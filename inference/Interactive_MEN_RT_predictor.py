import os
import numpy as np
import torch
from typing import Union, List, Tuple, Optional, Dict
from concurrent.futures import ThreadPoolExecutor
from time import time
import sys
import importlib
import math

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from batchgenerators.utilities.file_and_folder_operations import load_json, join, subdirs
from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice, crop_and_pad_nd
from torch import nn
import torch.nn.functional as F
from torch.nn.functional import interpolate
import nnunetv2

import nnInteractive
from nnInteractive.interaction.point import PointInteraction_stub
from nnInteractive.utils.bboxes import generate_bounding_boxes
from nnInteractive.utils.crop import crop_and_pad_into_buffer, paste_tensor, pad_cropped, crop_to_valid
from nnInteractive.utils.erosion_dilation import iterative_3x3_same_padding_pool3d
from nnInteractive.utils.rounding import round_to_nearest_odd


class InteractiveMENRTPredictor:
    """
    Interactive MEN RT Predictor for interactive segmentation with point, bbox, scribble, and lasso interactions.
    """
    
    def __init__(self,
                 device: torch.device = torch.device('cuda'),
                 use_torch_compile: bool = False,
                 verbose: bool = False,
                 torch_n_threads: int = 8,
                 do_autozoom: bool = True,
                 use_pinned_memory: bool = True
    ):
        """
        Only intended to work with nnInteractiveTrainerV2 and its derivatives
        """
        # set as part of initialization
        assert use_torch_compile is False, ('This implementation places the preprocessed image and the interactions '
                                            'into pinned memory for speed reasons. This is incompatible with '
                                            'torch.compile because of inconsistent strides in the memory layout. '
                                            'Note to self: .contiguous() on GPU could be a solution. Unclear whether '
                                            'that will yield a benefit though.')
        self.network = None
        self.label_manager = None
        self.dataset_json = None
        self.trainer_name = None
        self.configuration_manager = None
        self.plans_manager = None
        self.use_pinned_memory = use_pinned_memory
        self.device = device
        self.use_torch_compile = use_torch_compile
        
        # Interactive session state
        self.interactions: torch.Tensor = None
        self.preprocessed_image: torch.Tensor = None
        self.preprocessed_props = None
        self.target_buffer: Union[np.ndarray, torch.Tensor] = None
        
        self.pad_mode_data = self.preferred_scribble_thickness = self.point_interaction = None
        self.verbose = verbose
        
        self.do_autozoom: bool = do_autozoom
        torch.set_num_threads(min(torch_n_threads, os.cpu_count()))

        self.original_image_shape = None

        self.new_interaction_zoom_out_factors: List[float] = []
        self.new_interaction_centers = []
        self.has_positive_bbox = False

        # Create a thread pool executor for background tasks.
        # this only takes care of preprocessing and interaction memory initialization so there is no need to give it
        # more than 2 workers
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.preprocess_future = None
        self.interactions_future = None

    def set_image(self, image: np.ndarray, image_properties: dict = None):
        """
        Image must be 4D to satisfy nnU-Net needs: [c, x, y, z]
        Offload the processing to a background thread.
        """
        if image_properties is None:
            image_properties = {}
        self._reset_session()
        assert image.ndim == 4, f'expected a 4d image as input, got {image.ndim}d. Shape {image.shape}'
        if self.verbose:
            print(f'Initialize with raw image shape {image.shape}')

        # Offload all image preprocessing to a background thread.
        self.preprocess_future = self.executor.submit(self._background_set_image, image, image_properties)
        self.original_image_shape = image.shape

    def _finish_preprocessing_and_initialize_interactions(self):
        """
        Block until both the image preprocessing and the interactions tensor initialization
        are finished.
        """
        if self.preprocess_future is not None:
            # Wait for image preprocessing to complete.
            self.preprocess_future.result()
            del self.preprocess_future
            self.preprocess_future = None

    def set_target_buffer(self, target_buffer: Union[np.ndarray, torch.Tensor]):
        """
        Must be 3d numpy array or torch.Tensor
        """
        self.target_buffer = target_buffer

    def set_do_autozoom(self, do_propagation: bool, max_num_patches: Optional[int] = None):
        self.do_autozoom = do_propagation

    def _reset_session(self):
        self.interactions_future = None
        self.preprocess_future = None

        del self.preprocessed_image
        del self.target_buffer
        del self.interactions
        del self.preprocessed_props
        self.preprocessed_image = None
        self.target_buffer = None
        self.interactions = None
        self.preprocessed_props = None
        empty_cache(self.device)
        self.original_image_shape = None
        self.has_positive_bbox = False

    def _initialize_interactions(self, image_torch: torch.Tensor):
        if self.verbose:
            print(f'Initialize interactions. Pinned: {self.use_pinned_memory}')
        # Create the interaction tensor based on the target shape.
        self.interactions = torch.zeros(
            (7, *image_torch.shape[1:]),
            device='cpu',
            dtype=torch.float16,
            pin_memory=(self.device.type == 'cuda' and self.use_pinned_memory)
        )
    
    def _background_set_image(self, image: np.ndarray, image_properties: dict):
        """Background preprocessing of the image"""
        # Convert to torch tensor
        image_torch = torch.clone(torch.from_numpy(image))
        
        # Crop to nonzero region
        if self.verbose:
            print('Cropping input image to nonzero region')
        nonzero_idx = torch.where(image_torch != 0)
        bbox = [[i.min().item(), i.max().item() + 1] for i in nonzero_idx]
        
        # Ensure bbox is larger than patch_size
        if hasattr(self, 'configuration_manager') and self.configuration_manager is not None:
            patch_size = self.configuration_manager.patch_size
            for dim in range(1, len(bbox)):
                bbox_size = bbox[dim][1] - bbox[dim][0]
                if bbox_size < patch_size[dim - 1]:
                    # Center the bbox and extend it to patch_size
                    center = (bbox[dim][0] + bbox[dim][1]) // 2
                    bbox[dim][0] = max(0, center - patch_size[dim - 1] // 2)
                    bbox[dim][1] = min(image_torch.shape[dim], center + patch_size[dim - 1] // 2 + patch_size[dim - 1] % 2)
        
        del nonzero_idx
        slicer = bounding_box_to_slice(bbox)
        image_torch = image_torch[slicer].float()
        
        if self.verbose:
            print(f'Cropped image shape: {image_torch.shape}')
            
        # Initialize interactions tensor
        self._initialize_interactions(image_torch)
        
        # Normalize the image
        if self.verbose:
            print('Normalizing cropped image')
        image_torch -= image_torch.mean()
        image_torch /= image_torch.std()
        
        self.preprocessed_image = image_torch
        if self.use_pinned_memory and self.device.type == 'cuda':
            if self.verbose:
                print('Pin memory: image')
            self.preprocessed_image = self.preprocessed_image.pin_memory()
            
        self.preprocessed_props = {'bbox_used_for_cropping': bbox[1:]}

    def reset_interactions(self):
        """
        Use this to reset all interactions and start from scratch for the current image. This includes the initial
        segmentation!
        """
        if self.interactions is not None:
            self.interactions.fill_(0)

        if self.target_buffer is not None:
            if isinstance(self.target_buffer, np.ndarray):
                self.target_buffer.fill(0)
            elif isinstance(self.target_buffer, torch.Tensor):
                self.target_buffer.zero_()
        empty_cache(self.device)
        self.has_positive_bbox = False
    
    def add_bbox_interaction(self, bbox_coords, include_interaction: bool, run_prediction: bool = True) -> np.ndarray:
        if include_interaction:
            self.has_positive_bbox = True

        self._finish_preprocessing_and_initialize_interactions()

        lbs_transformed = [round(i) for i in transform_coordinates_noresampling([i[0] for i in bbox_coords],
                                                             self.preprocessed_props['bbox_used_for_cropping'])]
        ubs_transformed = [round(i) for i in transform_coordinates_noresampling([i[1] for i in bbox_coords],
                                                             self.preprocessed_props['bbox_used_for_cropping'])]
        transformed_bbox_coordinates = [[i, j] for i, j in zip(lbs_transformed, ubs_transformed)]

        if self.verbose:
            print(f'Added bounding box coordinates.\n'
                  f'Raw: {bbox_coords}\n'
                  f'Transformed: {transformed_bbox_coordinates}\n'
                  f"Crop Bbox: {self.preprocessed_props['bbox_used_for_cropping']}")

        # Prevent collapsed bounding boxes and clip to image shape
        image_shape = self.preprocessed_image.shape  # Assuming shape is (C, H, W, D) or similar

        for dim in range(len(transformed_bbox_coordinates)):
            transformed_start, transformed_end = transformed_bbox_coordinates[dim]

            # Clip to image boundaries
            transformed_start = max(0, transformed_start)
            transformed_end = min(image_shape[dim + 1], transformed_end)  # +1 to skip channel dim

            # Ensure the bounding box does not collapse to a single point
            if transformed_end <= transformed_start:
                if transformed_start == 0:
                    transformed_end = min(1, image_shape[dim + 1])
                else:
                    transformed_start = max(transformed_start - 1, 0)

            transformed_bbox_coordinates[dim] = [transformed_start, transformed_end]

        if self.verbose:
            print(f'Bbox coordinates after clip to image boundaries and preventing dim collapse:\n'
                  f'Bbox: {transformed_bbox_coordinates}\n'
                  f'Internal image shape: {self.preprocessed_image.shape}')

        self._add_patch_for_bbox_interaction(transformed_bbox_coordinates)

        # decay old interactions
        self.interactions[-6:-4] *= self.interaction_decay

        # place bbox
        slicer = tuple([slice(*i) for i in transformed_bbox_coordinates])
        channel = -6 if include_interaction else -5
        self.interactions[(channel, *slicer)] = 1

        # forward pass
        if run_prediction:
            self._predict()

    def add_point_interaction(self, coordinates: Tuple[int, ...], include_interaction: bool, run_prediction: bool = True):
        self._finish_preprocessing_and_initialize_interactions()

        transformed_coordinates = [round(i) for i in transform_coordinates_noresampling(coordinates,
                                                             self.preprocessed_props['bbox_used_for_cropping'])]

        self._add_patch_for_point_interaction(transformed_coordinates)

        # decay old interactions
        self.interactions[-4:-2] *= self.interaction_decay

        interaction_channel = -4 if include_interaction else -3
        self.interactions[interaction_channel] = self.point_interaction.place_point(
            transformed_coordinates, self.interactions[interaction_channel])
        if run_prediction:
            self._predict()

    def add_scribble_interaction(self, scribble_image: np.ndarray,  include_interaction: bool, run_prediction: bool = True):
        assert all([i == j for i, j in zip(self.original_image_shape[1:], scribble_image.shape)]), f'Given scribble image must match input image shape. Input image was: {self.original_image_shape[1:]}, given: {scribble_image.shape}'
        self._finish_preprocessing_and_initialize_interactions()

        scribble_image = torch.from_numpy(scribble_image)

        # crop (as in preprocessing)
        scribble_image = crop_and_pad_nd(scribble_image, self.preprocessed_props['bbox_used_for_cropping'])

        self._add_patch_for_scribble_interaction(scribble_image)

        # decay old interactions
        self.interactions[-2:] *= self.interaction_decay

        interaction_channel = -2 if include_interaction else -1
        torch.maximum(self.interactions[interaction_channel], scribble_image.to(self.interactions.device),
                      out=self.interactions[interaction_channel])
        del scribble_image
        empty_cache(self.device)
        if run_prediction:
            self._predict()

    def add_lasso_interaction(self, lasso_image: np.ndarray,  include_interaction: bool, run_prediction: bool = True):
        assert all([i == j for i, j in zip(self.original_image_shape[1:], lasso_image.shape)]), f'Given lasso image must match input image shape. Input image was: {self.original_image_shape[1:]}, given: {lasso_image.shape}'
        self._finish_preprocessing_and_initialize_interactions()

        lasso_image = torch.from_numpy(lasso_image)

        # crop (as in preprocessing)
        lasso_image = crop_and_pad_nd(lasso_image, self.preprocessed_props['bbox_used_for_cropping'])

        self._add_patch_for_lasso_interaction(lasso_image)

        # decay old interactions
        self.interactions[-6:-4] *= self.interaction_decay

        # lasso is written into bbox channel
        interaction_channel = -6 if include_interaction else -5
        torch.maximum(self.interactions[interaction_channel], lasso_image.to(self.interactions.device),
                      out=self.interactions[interaction_channel])
        del lasso_image
        empty_cache(self.device)
        if run_prediction:
            self._predict()

    def add_initial_seg_interaction(self, initial_seg: np.ndarray, run_prediction: bool = False):
        """
        WARNING THIS WILL RESET INTERACTIONS!
        """
        assert all([i == j for i, j in zip(self.original_image_shape[1:], initial_seg.shape)]), f'Given initial seg must match input image shape. Input image was: {self.original_image_shape[1:]}, given: {initial_seg.shape}'

        self._finish_preprocessing_and_initialize_interactions()

        self.reset_interactions()

        if isinstance(self.target_buffer, np.ndarray):
            self.target_buffer[:] = initial_seg

        initial_seg = torch.from_numpy(initial_seg)

        if isinstance(self.target_buffer, torch.Tensor):
            self.target_buffer[:] = initial_seg

        # crop (as in preprocessing)
        initial_seg = crop_and_pad_nd(initial_seg, self.preprocessed_props['bbox_used_for_cropping'])

        # initial seg is written into initial seg buffer
        interaction_channel = -7
        self.interactions[interaction_channel] = initial_seg
        empty_cache(self.device)
        if run_prediction:
            self._add_patch_for_initial_seg_interaction(initial_seg)
            del initial_seg
            self._predict()
        else:
            del initial_seg

    @torch.inference_mode()
    def _predict(self):
        """
        Perform prediction with interactions. The process follows the training procedure:
        1. Make initial prediction with current interactions
        2. Generate new interactions based on prediction errors
        3. Make final prediction with updated interactions
        """
        assert self.pad_mode_data == 'constant', 'pad modes other than constant are not implemented here'

        start_predict = time()
        with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            # Find the region containing all interactions
            interaction_mask = torch.any(self.interactions[1:] > 0, dim=0)  # Combine all interaction channels
            if not torch.any(interaction_mask):
                print('No interactions found, skipping prediction')
                return

            # Get bounding box of interaction region
            nonzero_indices = torch.nonzero(interaction_mask)
            min_coords = torch.min(nonzero_indices, dim=0)[0]
            max_coords = torch.max(nonzero_indices, dim=0)[0]
            
            # Initialize bbox with interaction region
            patch_size = self.configuration_manager.patch_size
            half_patch_size = [p // 2 for p in patch_size]
            image_shape = self.preprocessed_image.shape[1:]
            
            # For each dimension, calculate bbox ensuring:
            # 1. bbox start >= 0
            # 2. bbox end <= image_shape
            # 3. bbox size >= patch_size
            bbox = []
            for i, (min_c, max_c, h, p) in enumerate(zip(min_coords, max_coords, half_patch_size, patch_size)):
                start = max(0, min(image_shape[i] - p, (min_c + max_c) // 2 - p // 2))
                end = min(image_shape[i], start + p)
                bbox.append([start, end])

            # Calculate number of patches needed
            overlap = [64, 64, 64]                             # [O_z, O_y, O_x]
            num_patches = [
                1 if (b1 - b0) <= P 
                else math.ceil(((b1 - b0) - P) / (P - O)) + 1
                for (b0, b1), P, O in zip(bbox, patch_size, overlap)
            ]
            
            # Initialize prediction tensors for soft merging
            final_pred_soft = torch.zeros((2, *self.preprocessed_image.shape[1:]), dtype=torch.float32, device='cpu')
            prediction_count = torch.zeros(self.preprocessed_image.shape[1:], dtype=torch.float32, device='cpu')

            # Process each patch
            for x in range(num_patches[0]):
                for y in range(num_patches[1]):
                    for z in range(num_patches[2]):
                        # Calculate patch boundaries
                        step_index = [x, y, z]
                        start_coords = [bbox[i][0] + step_index[i] * p for i, p in zip([0, 1, 2], patch_size)]
                        end_coords = [min(bbox[i][1], start_coords[i] + p) for i, p in zip([0, 1, 2], patch_size)]
                        
                        for i in range(len(patch_size)):
                            if end_coords[i] - start_coords[i] < patch_size[i]:
                                if end_coords[i] >= bbox[i][1]:
                                    start_coords[i] = bbox[i][1] - patch_size[i]
                        
                        # Extract image patch
                        image_patch = self.preprocessed_image[:, start_coords[0]:end_coords[0], 
                                                           start_coords[1]:end_coords[1], 
                                                           start_coords[2]:end_coords[2]]
                        
                        # Extract interaction patches
                        interaction_patch = self.interactions[:, start_coords[0]:end_coords[0],
                                                           start_coords[1]:end_coords[1],
                                                           start_coords[2]:end_coords[2]]
                        
                        # Pad to patch_size if necessary
                        if not all([e - s == p for s, e, p in zip(start_coords, end_coords, patch_size)]):
                            pad_size = [(0, p - (e - s)) for s, e, p in zip(start_coords, end_coords, patch_size)]
                            image_patch = F.pad(image_patch, [item for sublist in reversed(pad_size) for item in sublist])
                            interaction_patch = F.pad(interaction_patch, [item for sublist in reversed(pad_size) for item in sublist])
                        
                        # Move to device
                        image_patch = image_patch.to(self.device, non_blocking=self.device.type == 'cuda')
                        interaction_patch = interaction_patch.to(self.device, non_blocking=self.device.type == 'cuda')
                        
                        # Concatenate image and interaction channels
                        input_for_predict = torch.cat((image_patch, interaction_patch))
                        
                        # Make prediction
                        pred_raw = self.network(input_for_predict[None])[0]
                        pred_prob = F.softmax(pred_raw, dim=0)
                        
                        del input_for_predict, pred_raw, image_patch, interaction_patch
                        
                        # Resize prediction if needed
                        if not all([e - s == p for s, e, p in zip(start_coords, end_coords, patch_size)]):
                            pred_prob = interpolate(pred_prob[None], 
                                                  [e - s for s, e in zip(start_coords, end_coords)], 
                                                  mode='trilinear')[0]

                        # Add to accumulated predictions
                        pred_prob = pred_prob.cpu()
                        final_pred_soft[:, start_coords[0]:end_coords[0],
                                      start_coords[1]:end_coords[1],
                                      start_coords[2]:end_coords[2]] += pred_prob
                        prediction_count[start_coords[0]:end_coords[0],
                                       start_coords[1]:end_coords[1],
                                       start_coords[2]:end_coords[2]] += 1

                        del pred_prob
                        empty_cache(self.device)

            # Average predictions and convert to binary
            final_pred_soft = final_pred_soft / prediction_count.clamp(min=1)
            # final_pred_soft = self._iterative_adjust_prediction(final_pred_soft, self.interactions)
            final_pred = (final_pred_soft[1] >= 0.5).to(torch.uint8)

            # Update interactions and target buffer
            self.interactions[0][:] = final_pred
            paste_tensor(self.target_buffer, final_pred, self.preprocessed_props['bbox_used_for_cropping'])

        print(f'Done. Total time {round(time() - start_predict, 3)}s')

        self.new_interaction_centers = []
        empty_cache(self.device)

    @torch.inference_mode()
    def _predict_without_interaction(self):
        """
        Perform prediction with interaction channels but without zooming. This is a simplified version of _predict that:
        1. Makes prediction on the entire image at once using interaction channels
        2. No zooming or refinement is performed
        3. Uses all interaction channels (previous segmentation, bbox, point, scribble)
        """
        assert self.pad_mode_data == 'constant', 'pad modes other than constant are not implemented here'
        
        start_predict = time()
        with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            # Get image dimensions
            image_shape = self.preprocessed_image.shape[1:]  # Remove channel dimension
            
            # Calculate number of patches needed
            patch_size = self.configuration_manager.patch_size
            bbox = [[0, i] for i in image_shape]

            # Calculate number of patches needed
            overlap = [64, 64, 64]                             # [O_z, O_y, O_x]
            num_patches = [
                1 if (b1 - b0) <= P 
                else math.ceil(((b1 - b0) - P) / (P - O)) + 1
                for (b0, b1), P, O in zip(bbox, patch_size, overlap)
            ]
            
            # Initialize prediction tensors for soft merging
            pred_soft = torch.zeros((2, *image_shape), dtype=torch.float32, device='cpu')  # 2 channels for binary segmentation
            pred_count = torch.zeros(image_shape, dtype=torch.float32, device='cpu')
            
            # Process each patch
            for x in range(num_patches[0]):
                for y in range(num_patches[1]):
                    for z in range(num_patches[2]):
                        # Calculate patch boundaries
                        step_index = [x, y, z]
                        start_coords = [bbox[i][0] + step_index[i] * p for i, p in zip([0, 1, 2], patch_size)]
                        end_coords = [min(bbox[i][1], start_coords[i] + p) for i, p in zip([0, 1, 2], patch_size)]
                        
                        for i in range(len(patch_size)):
                            if end_coords[i] - start_coords[i] < patch_size[i]:
                                if end_coords[i] >= bbox[i][1]:
                                    start_coords[i] = bbox[i][1] - patch_size[i]
                        
                        # Extract image patch
                        image_patch = self.preprocessed_image[:, start_coords[0]:end_coords[0], 
                                                           start_coords[1]:end_coords[1], 
                                                           start_coords[2]:end_coords[2]]
                        
                        # Extract interaction patches
                        interaction_patch = self.interactions[:, start_coords[0]:end_coords[0],
                                                           start_coords[1]:end_coords[1],
                                                           start_coords[2]:end_coords[2]]
                        
                        # Pad if necessary
                        if not all([e - s == p for s, e, p in zip(start_coords, end_coords, patch_size)]):
                            pad_size = [(0, p - (e - s)) for s, e, p in zip(start_coords, end_coords, patch_size)]
                            image_patch = F.pad(image_patch, [item for sublist in reversed(pad_size) for item in sublist])
                            interaction_patch = F.pad(interaction_patch, [item for sublist in reversed(pad_size) for item in sublist])
                        
                        # Move to device
                        image_patch = image_patch.to(self.device, non_blocking=self.device.type == 'cuda')
                        interaction_patch = interaction_patch.to(self.device, non_blocking=self.device.type == 'cuda')
                        
                        # Concatenate image and interaction channels
                        input_for_predict = torch.cat((image_patch, interaction_patch))
                        
                        # Make prediction and get soft probabilities
                        patch_pred = self.network(input_for_predict[None])[0]
                        patch_prob = F.softmax(patch_pred, dim=0)
                        
                        # Resize prediction to original patch size if necessary
                        if not all([e - s == p for s, e, p in zip(start_coords, end_coords, patch_size)]):
                            patch_prob = interpolate(patch_prob[None], 
                                                   [e - s for s, e in zip(start_coords, end_coords)], 
                                                   mode='trilinear')[0]
                        
                        # Add to accumulated predictions
                        pred_soft[:, start_coords[0]:end_coords[0], 
                                start_coords[1]:end_coords[1], 
                                start_coords[2]:end_coords[2]] += patch_prob.cpu()
                        pred_count[start_coords[0]:end_coords[0], 
                                 start_coords[1]:end_coords[1], 
                                 start_coords[2]:end_coords[2]] += 1
                        
                        del image_patch, interaction_patch, input_for_predict, patch_pred, patch_prob
                        empty_cache(self.device)
            
            # Average predictions and convert to binary
            pred_soft = pred_soft / pred_count.clamp(min=1)
            pred = (pred_soft[1] >= 0.5).to(torch.uint8)

            # Update interactions and target buffer
            self.interactions[0][:] = pred
            paste_tensor(self.target_buffer, pred, self.preprocessed_props['bbox_used_for_cropping'])
        
        print(f'Done. Total time {round(time() - start_predict, 3)}s')
        empty_cache(self.device)

    def _add_patch_for_point_interaction(self, coordinates):
        self.new_interaction_centers.append(coordinates)
        print(f'Added new point interaction: center {coordinates}')

    def _add_patch_for_bbox_interaction(self, bbox):
        bbox_center = [round((i[0] + i[1]) / 2) for i in bbox]
        bbox_size = [i[1]-i[0] for i in bbox]
        # we want to see some context, so the crop we see for the initial prediction should be patch_size / 3 larger
        requested_size = [i + j // 3 for i, j in zip(bbox_size, self.configuration_manager.patch_size)]
        self.new_interaction_zoom_out_factors.append(max(1, max([i / j for i, j in zip(requested_size, self.configuration_manager.patch_size)])))
        self.new_interaction_centers.append(bbox_center)
        print(f'Added new bbox interaction: center {bbox_center}')

    def _add_patch_for_scribble_interaction(self, scribble_image):
        return self._generic_add_patch_from_image(scribble_image)

    def _add_patch_for_lasso_interaction(self, lasso_image):
        return self._generic_add_patch_from_image(lasso_image)

    def _add_patch_for_initial_seg_interaction(self, initial_seg):
        return self._generic_add_patch_from_image(initial_seg)

    def _generic_add_patch_from_image(self, image: torch.Tensor):
        if not torch.any(image):
            print('Received empty image prompt. Cannot add patches for prediction')
            return
        nonzero_indices = torch.nonzero(image, as_tuple=False)
        mn = torch.min(nonzero_indices, dim=0)[0]
        mx = torch.max(nonzero_indices, dim=0)[0]
        roi = [[i.item(), x.item() + 1] for i, x in zip(mn, mx)]
        roi_center = [round((i[0] + i[1]) / 2) for i in roi]
        roi_size = [i[1]- i[0] for i in roi]
        requested_size = [i + j // 3 for i, j in zip(roi_size, self.configuration_manager.patch_size)]
        self.new_interaction_zoom_out_factors.append(max(1, max([i / j for i, j in zip(requested_size, self.configuration_manager.patch_size)])))
        self.new_interaction_centers.append(roi_center)
        print(f'Added new image interaction: scale {self.new_interaction_zoom_out_factors[-1]}, center {roi_center}')
        
    def initialize_from_trained_model_folder(self, 
                                           model_training_output_dir: str,
                                           use_fold: Union[int, str] = None,
                                           checkpoint_name: str = 'checkpoint_final.pth'):
        """
        Initialize the predictor from a trained model folder.
        """
        # Determine fold folder
        if use_fold is not None:
            use_fold = int(use_fold) if use_fold != 'all' else use_fold
            fold_folder = f'fold_{use_fold}'
        else:
            fldrs = subdirs(model_training_output_dir, prefix='fold_', join=False)
            assert len(fldrs) == 1, f'Attempted to infer fold but there is != 1 fold_ folders: {fldrs}'
            fold_folder = fldrs[0]
            
        # load trainer specific settings
        expected_json_file = join(model_training_output_dir, fold_folder, 'inference_session_class.json')
        json_content = load_json(expected_json_file)
        if isinstance(json_content, str):
            # Old convention where we only specified the inference class in this file
            point_interaction_radius = 4
            point_interaction_use_etd = True
            self.preferred_scribble_thickness = [2, 2, 2]
            self.point_interaction = PointInteraction_stub(
                point_interaction_radius,
                point_interaction_use_etd)
            self.pad_mode_data = "constant"
            self.interaction_decay = 0.9
        else:
            point_interaction_radius = json_content['point_radius']
            self.preferred_scribble_thickness = json_content['preferred_scribble_thickness']
            if not isinstance(self.preferred_scribble_thickness, (tuple, list)):
                self.preferred_scribble_thickness = [self.preferred_scribble_thickness] * 3
            self.interaction_decay = json_content['interaction_decay'] if 'interaction_decay' in json_content.keys() else 0.9
            point_interaction_use_etd = json_content['use_distance_transform'] if 'use_distance_transform' in json_content.keys() else True
            self.point_interaction = PointInteraction_stub(point_interaction_radius, point_interaction_use_etd)
            # padding mode for data. See nnInteractiveTrainerV2_nodelete_reflectpad
            self.pad_mode_data = json_content['pad_mode_image'] if 'pad_mode_image' in json_content.keys() else "constant"

        # Load dataset and plans
        dataset_json = load_json(join(model_training_output_dir, 'dataset.json'))
        plans = load_json(join(model_training_output_dir, 'plans.json'))
        plans_manager = PlansManager(plans)

        # Load checkpoint
        checkpoint = torch.load(join(model_training_output_dir, fold_folder, checkpoint_name),
                              map_location=self.device, weights_only=False)
        trainer_name = checkpoint['trainer_name']
        configuration_name = checkpoint['init_args']['configuration']
        parameters = checkpoint['network_weights']

        # Get configuration
        configuration_manager = plans_manager.get_configuration(configuration_name)
        
        # Restore network
        num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
        network = nnUNetTrainer.build_network_architecture(
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels,
            plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
            enable_deep_supervision=False
        ).to(self.device)
        network.load_state_dict(parameters)

        # Store necessary information
        self.plans_manager = plans_manager
        self.configuration_manager = configuration_manager
        self.network = network
        self.dataset_json = dataset_json
        self.trainer_name = trainer_name
        self.label_manager = plans_manager.get_label_manager(dataset_json)

        if self.use_torch_compile:
            print('Using torch.compile')
            self.network = torch.compile(self.network)

        if self.verbose:
            print(f"Loaded interactive config: point_radius={self.point_interaction.point_radius}, "
                  f"scribble_thickness={self.preferred_scribble_thickness}, "
                  f"interaction_decay={self.interaction_decay}")
            
    def manual_initialization(self, network: nn.Module, plans_manager: PlansManager,
                              configuration_manager: ConfigurationManager,
                              dataset_json: dict, trainer_name: str):
        """
        This is used by the nnUNetTrainer to initialize nnUNetPredictor for the final validation
        """
        self.plans_manager = plans_manager
        self.configuration_manager = configuration_manager
        self.network = network
        self.dataset_json = dataset_json
        self.trainer_name = trainer_name
        self.label_manager = plans_manager.get_label_manager(dataset_json)

        if self.use_torch_compile and not isinstance(self.network, OptimizedModule):
            print('Using torch.compile')
            self.network = torch.compile(self.network)

        if not self.use_torch_compile and isinstance(self.network, OptimizedModule):
            self.network = self.network._orig_mod

        self.network = self.network.to(self.device)

    @torch.inference_mode()
    def _predict_autozoom(self):
        """
        Perform prediction with interactions. The process follows the training procedure:
        1. Make initial prediction with current interactions
        2. Generate new interactions based on prediction errors
        3. Make final prediction with updated interactions
        """
        assert self.pad_mode_data == 'constant', 'pad modes other than constant are not implemented here'
        assert len(self.new_interaction_centers) == len(self.new_interaction_zoom_out_factors)
        if len(self.new_interaction_centers) > 1:
            print('It seems like more than one interaction was added since the last prediction. This is not '
                  'recommended and may cause unexpected behavior or inefficient predictions')

        start_predict = time()
        with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            for prediction_center, initial_zoom_out_factor in zip(self.new_interaction_centers, self.new_interaction_zoom_out_factors):
                # Store previous prediction for comparison
                previous_prediction = torch.clone(self.interactions[0])

                if not self.do_autozoom:
                    initial_zoom_out_factor = 1

                initial_zoom_out_factor = min(initial_zoom_out_factor, 4)
                zoom_out_factor = initial_zoom_out_factor
                max_zoom_out_factor = initial_zoom_out_factor

                start_autozoom = time()
                while zoom_out_factor is not None and zoom_out_factor <= 4:
                    print('Performing prediction at zoom out factor', zoom_out_factor)
                    max_zoom_out_factor = max(max_zoom_out_factor, zoom_out_factor)
                    
                    # Calculate patch size and bounding box
                    scaled_patch_size = [round(i * zoom_out_factor) for i in self.configuration_manager.patch_size]
                    scaled_bbox = [[int(c - p // 2), int(c + p // 2 + p % 2)] for c, p in zip(prediction_center, scaled_patch_size)]

                    # Crop and prepare input
                    crop_img, pad = crop_to_valid(self.preprocessed_image, scaled_bbox)
                    crop_img = crop_img.to(self.device, non_blocking=self.device.type == 'cuda')
                    crop_interactions, pad_interaction = crop_to_valid(self.interactions, scaled_bbox)

                    # Resize if needed
                    if not all([i == j for i, j in zip(self.configuration_manager.patch_size, scaled_patch_size)]):
                        crop_interactions_resampled_gpu = torch.empty((7, *self.configuration_manager.patch_size), dtype=torch.float16, device=self.device)
                        
                        # Handle previous segmentation and bbox channels
                        for i in range(0, 3):
                            if any([x for y in pad_interaction for x in y]):
                                tmp = pad_cropped(crop_interactions[i].to(self.device, non_blocking=self.device.type == 'cuda'), pad_interaction)
                            else:
                                tmp = crop_interactions[i].to(self.device)
                            crop_interactions_resampled_gpu[i] = interpolate(tmp[None, None], self.configuration_manager.patch_size, mode='area')[0][0]
                        empty_cache(self.device)

                        # Handle point and scribble channels with dilation
                        max_pool_ks = round_to_nearest_odd(zoom_out_factor * 2 - 1)
                        for i in range(3, 7):
                            if any([x for y in pad_interaction for x in y]):
                                tmp = pad_cropped(crop_interactions[i].to(self.device, non_blocking=self.device.type == 'cuda'), pad_interaction)
                            else:
                                tmp = crop_interactions[i].to(self.device, non_blocking=self.device.type == 'cuda')
                            if max_pool_ks > 1:
                                tmp = iterative_3x3_same_padding_pool3d(tmp[None, None], max_pool_ks)[0, 0]
                            crop_interactions_resampled_gpu[i] = interpolate(tmp[None, None], self.configuration_manager.patch_size, mode='area')[0][0]
                        del tmp

                        crop_img = interpolate(pad_cropped(crop_img, pad)[None] if any([x for y in pad_interaction for x in y]) else crop_img[None], 
                                             self.configuration_manager.patch_size, mode='trilinear')[0]
                        crop_interactions = crop_interactions_resampled_gpu
                        del crop_interactions_resampled_gpu
                        empty_cache(self.device)
                    else:
                        crop_img = pad_cropped(crop_img, pad) if any([x for y in pad_interaction for x in y]) else crop_img
                        crop_interactions = pad_cropped(crop_interactions.to(self.device, non_blocking=self.device.type == 'cuda'), pad_interaction) if any([x for y in pad_interaction for x in y]) else crop_interactions.to(self.device, non_blocking=self.device.type == 'cuda')

                    # Make prediction
                    input_for_predict = torch.cat((crop_img, crop_interactions))
                    del crop_img, crop_interactions
                    pred = self.network(input_for_predict[None])[0].argmax(0).detach()
                    del input_for_predict

                    # Check for changes at borders
                    previous_zoom_prediction = crop_and_pad_nd(self.interactions[0], scaled_bbox).to(self.device, non_blocking=self.device.type == 'cuda')
                    if not all([i == j for i, j in zip(pred.shape, previous_zoom_prediction.shape)]):
                        previous_zoom_prediction = interpolate(previous_zoom_prediction[None, None].to(float), pred.shape, mode='nearest')[0, 0]

                    # Determine if we need to continue zooming
                    continue_zoom = False
                    if zoom_out_factor < 4 and self.do_autozoom:
                        for dim in range(len(scaled_bbox)):
                            if continue_zoom:
                                break
                            for idx in [0, pred.shape[dim] - 1]:
                                slice_prev = previous_zoom_prediction.index_select(dim, torch.tensor(idx, device=self.device))
                                slice_curr = pred.index_select(dim, torch.tensor(idx, device=self.device))
                                pixels_prev = torch.sum(slice_prev)
                                pixels_current = torch.sum(slice_curr)
                                pixels_diff = torch.sum(slice_prev != slice_curr)
                                rel_change = max(pixels_prev, pixels_current) / max(min(pixels_prev, pixels_current), 1e-5) - 1
                                
                                if pixels_diff > 1500 or (pixels_diff > 100 and rel_change > 0.2):
                                    continue_zoom = True
                                    if self.verbose:
                                        print(f'Continuing zoom due to significant changes at borders')
                                    break
                                del slice_prev, slice_curr, pixels_prev, pixels_current, pixels_diff
                        del previous_zoom_prediction

                    # Resize prediction if needed
                    if not all([i == j for i, j in zip(pred.shape, scaled_patch_size)]):
                        pred = (interpolate(pred[None, None].to(float), scaled_patch_size, mode='trilinear')[0, 0] >= 0.5).to(torch.uint8)

                    # Update interactions and target buffer
                    if zoom_out_factor == 1 or not continue_zoom:
                        pred = pred.cpu()
                        paste_tensor(self.interactions[0], pred.half(), scaled_bbox)
                        
                        # Update target buffer
                        bbox = [[i[0] + bbc[0], i[1] + bbc[0]] for i, bbc in zip(scaled_bbox, self.preprocessed_props['bbox_used_for_cropping'])]
                        paste_tensor(self.target_buffer, pred, bbox)

                    del pred
                    empty_cache(self.device)

                    if continue_zoom:
                        zoom_out_factor *= 1.5
                        zoom_out_factor = min(4, zoom_out_factor)
                    else:
                        zoom_out_factor = None

                end = time()
                print(f'Auto zoom stage took {round(end - start_autozoom, ndigits=3)}s. Max zoom out factor was {max_zoom_out_factor}')

        print(f'Done. Total time {round(time() - start_predict, 3)}s')

        self.new_interaction_centers = []
        self.new_interaction_zoom_out_factors = []
        empty_cache(self.device)

    def _iterative_adjust_prediction(self, pred_prob: torch.Tensor, crop_interactions: torch.Tensor,
                                   max_iterations: int = 15, prob_increase_factor: float = 1.5) -> torch.Tensor:
        """
        Perform iterative prediction adjustment when positive interactions exist.
        
        Args:
            pred_prob: Probability prediction tensor [C, H, W, D]
            crop_interactions: Interaction tensor [7, H, W, D]
            max_iterations: Maximum number of iterations to try
            prob_increase_factor: Factor to increase foreground probability by in each iteration
            
        Returns:
            Adjusted prediction tensor
        """
        # Check if there are any positive interactions
        crop_interactions_pos = crop_interactions[1:7:2]
        pos_mask = torch.any(crop_interactions_pos > 0, dim=0)
        pos_mask_np = pos_mask.cpu().numpy()
        max_iterations = max_iterations if np.any(pos_mask_np) else 1
                
        iteration = 0
        while iteration < max_iterations:
            pred_prob = self._adjust_prediction_with_interactions(pred_prob, crop_interactions)
            pred_np = pred_prob[1].cpu().numpy()
            
            # If prediction is all zero, try again with adjusted probabilities
            if not np.any(pred_np):
                # Increase foreground probability for regions with positive interactions
                pred_prob[1, pos_mask] = torch.clamp(pred_prob[1, pos_mask] * prob_increase_factor, 0, 1)
                pred_prob[0, pos_mask] = 1 - pred_prob[1, pos_mask]
                iteration += 1
            else:
                break
                
        return pred_prob

    def _adjust_prediction_with_interactions(self, pred_prob: torch.Tensor, crop_interactions: torch.Tensor) -> torch.Tensor:
        """
        Adjust prediction based on interaction masks using superpixel segmentation.
        
        Args:
            pred_prob: Probability prediction tensor [C, H, W, D]
            crop_interactions: Interaction tensor [7, H, W, D]
            
        Returns:
            Adjusted prediction tensor
        """
        # Separate positive and negative interactions
        crop_interactions_pos = crop_interactions[1:7:2]
        crop_interactions_neg = crop_interactions[2:7:2]
        
        pos_mask = torch.any(crop_interactions_pos > 0, dim=0)
        neg_mask = torch.any(crop_interactions_neg > 0, dim=0)
        
        # Separate connected components
        import scipy.ndimage
        from skimage.segmentation import slic
        # Get initial prediction for labeling using threshold
        pred_np = (pred_prob[1].cpu().numpy() > 0.5).astype(np.uint8)
        labeled_pred, num_components = scipy.ndimage.label(pred_np)
        
        # Convert masks to numpy for overlap checking
        pos_mask_np = pos_mask.cpu().numpy()
        neg_mask_np = neg_mask.cpu().numpy()
        
        # Check overlap for each component and adjust pred_prob
        for comp_id in range(1, num_components + 1):
            comp_mask = (labeled_pred == comp_id).astype(np.uint8)
            
            # Check overlap with positive and negative masks
            overlap_pos = np.logical_and(comp_mask, pos_mask_np)
            overlap_neg = np.logical_and(comp_mask, neg_mask_np)
            
            # If component overlaps with both positive and negative masks
            if np.any(overlap_pos) and np.any(overlap_neg):
                # Get the bounding box of the component
                bbox = scipy.ndimage.find_objects(comp_mask)[0]
                comp_region = comp_mask[bbox]
                pos_region = overlap_pos[bbox]
                neg_region = overlap_neg[bbox]
                
                # Get pred_prob values for the region
                pred_region_prob = pred_prob[:, bbox[0], bbox[1], bbox[2]].cpu().numpy()
                
                # Create RGB image from probabilities
                pred_rgb = np.transpose(pred_region_prob, (1, 2, 3, 0))  # [H, W, D, C]
                # pred_rgb = np.mean(pred_rgb, axis=-1, keepdims=True)  # Average across channels
                # pred_rgb = np.repeat(pred_rgb, 3, axis=-1)  # Repeat for RGB
                
                # Create superpixels based on pred_prob values
                n_segments = min(100, np.sum(comp_region))  # Limit number of segments
                segments = slic(pred_rgb, n_segments=n_segments, compactness=10, channel_axis=-1)
                
                # Process each superpixel
                for seg_id in range(1, segments.max() + 1):
                    seg_mask = (segments == seg_id)
                    seg_pos = np.logical_and(seg_mask, pos_region)
                    seg_neg = np.logical_and(seg_mask, neg_region)
                    
                    # Get global coordinates for this segment
                    seg_coords = np.where(seg_mask)
                    global_coords = tuple(c + b for c, b in zip(seg_coords, [b.start for b in bbox]))
                    
                    # Assign values based on interaction overlap
                    if np.any(seg_pos) and not np.any(seg_neg):
                        pred_prob[0, global_coords] = 0.0
                        pred_prob[1, global_coords] = 1.0
                    elif np.any(seg_neg) and not np.any(seg_pos):
                        pred_prob[0, global_coords] = 1.0
                        pred_prob[1, global_coords] = 0.0
                    # If segment has both interactions, use the original prediction
                    else:
                        continue
            
            # If component only overlaps with positive mask, force it to foreground
            elif np.any(overlap_pos):
                pred_prob[0, comp_mask > 0] = 0.0  # Set background to 0
                pred_prob[1, comp_mask > 0] = 1.0  # Set foreground to 1
            
            # If component only overlaps with negative mask, force it to background
            elif np.any(overlap_neg):
                pred_prob[0, comp_mask > 0] = 1.0  # Set background to 1
                pred_prob[1, comp_mask > 0] = 0.0  # Set foreground to 0
                
            # # If component does not overlap with any masks, force it to background
            # else:
            #     pred_prob[0, comp_mask > 0] = 1.0  # Set background to 1
            #     pred_prob[1, comp_mask > 0] = 0.0  # Set foreground to 0

        # Return thresholded prediction
        return pred_prob


def transform_coordinates_noresampling(
        coords_orig: Union[List[int], Tuple[int, ...]],
        nnunet_preprocessing_crop_bbox: List[Tuple[int, int]]
) -> Tuple[int, ...]:
    """
    converts coordinates in the original uncropped image to the internal cropped representation. Man I really hate
    nnU-Net's crop to nonzero!
    """
    return tuple([coords_orig[d] - nnunet_preprocessing_crop_bbox[d][0] for d in range(len(coords_orig))])


