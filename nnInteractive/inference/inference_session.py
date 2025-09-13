from concurrent.futures import ThreadPoolExecutor
from os import cpu_count
from time import time
from typing import Union, List, Tuple, Optional

import numpy as np
import torch
from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice, crop_and_pad_nd
from batchgenerators.utilities.file_and_folder_operations import load_json, join, subdirs
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.helpers import dummy_context, empty_cache
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from torch import nn
from torch._dynamo import OptimizedModule
from torch.nn.functional import interpolate

import nnInteractive
from nnInteractive.interaction.point import PointInteraction_stub
from nnInteractive.trainer.nnInteractiveTrainer import nnInteractiveTrainer_stub
from nnInteractive.utils.bboxes import generate_bounding_boxes
from nnInteractive.utils.crop import crop_and_pad_into_buffer, paste_tensor, pad_cropped, crop_to_valid
from nnInteractive.utils.erosion_dilation import iterative_3x3_same_padding_pool3d
from nnInteractive.utils.rounding import round_to_nearest_odd


class nnInteractiveInferenceSession():
    def __init__(self,
                 device: torch.device = torch.device('cuda'),
                 use_torch_compile: bool = False,
                 verbose: bool = False,
                 torch_n_threads: int = 8,
                 do_autozoom: bool = True,
                 use_pinned_memory: bool = True,
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
        self.interaction_decay = None

        # image specific
        self.interactions: torch.Tensor = None
        self.preprocessed_image: torch.Tensor = None
        self.preprocessed_props = None
        self.target_buffer: Union[np.ndarray, torch.Tensor] = None

        # this will be set when loading the model (initialize_from_trained_model_folder)
        self.pad_mode_data = self.preferred_scribble_thickness = self.point_interaction = None

        self.verbose = verbose

        self.do_autozoom: bool = do_autozoom

        torch.set_num_threads(min(torch_n_threads, cpu_count()))

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
        # Convert and clone the image tensor.
        image_torch = torch.clone(torch.from_numpy(image))

        # Crop to nonzero region.
        if self.verbose:
            print('Cropping input image to nonzero region')
        nonzero_idx = torch.where(image_torch != 0)
        # Create bounding box: for each dimension, get the min and max (plus one) of the nonzero indices.
        bbox = [[i.min().item(), i.max().item() + 1] for i in nonzero_idx]
        del nonzero_idx
        slicer = bounding_box_to_slice(bbox)  # Assuming this returns a tuple of slices.
        image_torch = image_torch[slicer].float()
        if self.verbose:
            print(f'Cropped image shape: {image_torch.shape}')

        # As soon as we have the target shape, start initializing the interaction tensor in its own thread.
        self.interactions_future = self.executor.submit(self._initialize_interactions, image_torch)

        # Normalize the cropped image.
        if self.verbose:
            print('Normalizing cropped image')
        image_torch -= image_torch.mean()
        image_torch /= image_torch.std()

        self.preprocessed_image = image_torch
        if self.use_pinned_memory and self.device.type == 'cuda':
            if self.verbose:
                print('Pin memory: image')
            # Note: pin_memory() in PyTorch typically returns a new tensor.
            self.preprocessed_image = self.preprocessed_image.pin_memory()

        self.preprocessed_props = {'bbox_used_for_cropping': bbox[1:]}

        # we need to wait for this here I believe
        self.interactions_future.result()
        del self.interactions_future
        self.interactions_future = None

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
        This function is a smoking mess to read. This is deliberate. Initially it was super pretty and easy to
        understand. Then the run time optimization began.
        If it feels like we are excessively transferring tensors between CPU and GPU, this is deliberate as well.
        Our goal is to keep this tool usable even for people with smaller GPUs (8-10GB VRAM). In an ideal world
        everyone would have 24GB+ of VRAM and all tensors would like on GPU all the time.
        The amount of hours spent optimizing this function is substantial. Almost every line was turned and twisted
        multiple times. If something appears odd, it is probably so for a reason. Don't change things all willy nilly
        without first understanding what is going on. And don't make changes without verifying that the run time or
        VRAM consumption is not adversely affected.

        Returns:

        """
        assert self.pad_mode_data == 'constant', 'pad modes other than constant are not implemented here'
        assert len(self.new_interaction_centers) == len(self.new_interaction_zoom_out_factors)
        if len(self.new_interaction_centers) > 1:
            print('It seems like more than one interaction was added since the last prediction. This is not '
                  'recommended and may cause unexpected behavior or inefficient predictions')

        start_predict = time()
        with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            for prediction_center, initial_zoom_out_factor in zip(self.new_interaction_centers, self.new_interaction_zoom_out_factors):
                # make a prediction at initial zoom out factor. If more zoom out is required, do this until the
                # entire object fits the FOV. Then go back to original resolution and refine.

                # we need this later.
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
                    # initial prediction at initial_zoom_out_factor
                    scaled_patch_size = [round(i * zoom_out_factor) for i in self.configuration_manager.patch_size]
                    scaled_bbox = [[c - p // 2, c + p // 2 + p % 2] for c, p in zip(prediction_center, scaled_patch_size)]

                    crop_img, pad = crop_to_valid(self.preprocessed_image, scaled_bbox)
                    crop_img = crop_img.to(self.device, non_blocking=self.device.type == 'cuda')
                    crop_interactions, pad_interaction = crop_to_valid(self.interactions, scaled_bbox)

                    # resize input_for_predict (which may be larger than patch size) to patch size
                    # this implementation may not seem straightforward but it does save VRAM which is crucial here
                    if not all([i == j for i, j in zip(self.configuration_manager.patch_size, scaled_patch_size)]):
                        crop_interactions_resampled_gpu = torch.empty((7, *self.configuration_manager.patch_size), dtype=torch.float16, device=self.device)
                        # previous seg, bbox+, bbox-
                        for i in range(0, 3):
                            # this is area for a reason but I aint telling ya why
                            if any([x for y in pad_interaction for x in y]):
                                tmp = pad_cropped(crop_interactions[i].to(self.device, non_blocking=self.device.type == 'cuda'), pad_interaction)
                            else:
                                tmp = crop_interactions[i].to(self.device)
                            crop_interactions_resampled_gpu[i] = interpolate(tmp[None, None], self.configuration_manager.patch_size, mode='area')[0][0]
                        empty_cache(self.device)

                        max_pool_ks = round_to_nearest_odd(zoom_out_factor * 2 - 1)
                        # point+, point-, scribble+, scribble-
                        for i in range(3, 7):
                            if any([x for y in pad_interaction for x in y]):
                                tmp = pad_cropped(crop_interactions[i].to(self.device, non_blocking=self.device.type == 'cuda'), pad_interaction)
                            else:
                                tmp = crop_interactions[i].to(self.device, non_blocking=self.device.type == 'cuda')
                            if max_pool_ks > 1:
                                # dilate to preserve interactions after downsampling
                                tmp = iterative_3x3_same_padding_pool3d(tmp[None, None], max_pool_ks)[0, 0]
                            # this is 'area' for a reason but I aint telling ya why
                            crop_interactions_resampled_gpu[i] = interpolate(tmp[None, None], self.configuration_manager.patch_size, mode='area')[0][0]
                        del tmp

                        crop_img = interpolate(pad_cropped(crop_img, pad)[None] if any([x for y in pad_interaction for x in y]) else crop_img[None], self.configuration_manager.patch_size, mode='trilinear')[0]
                        crop_interactions = crop_interactions_resampled_gpu

                        del crop_interactions_resampled_gpu
                        empty_cache(self.device)
                    else:
                        # crop_img is already on device
                        crop_img = pad_cropped(crop_img, pad) if any([x for y in pad_interaction for x in y]) else crop_img
                        crop_interactions = pad_cropped(crop_interactions.to(self.device, non_blocking=self.device.type == 'cuda'), pad_interaction) if any([x for y in pad_interaction for x in y]) else crop_interactions.to(self.device, non_blocking=self.device.type == 'cuda')

                    input_for_predict = torch.cat((crop_img, crop_interactions))
                    del crop_img, crop_interactions

                    pred = self.network(input_for_predict[None])[0].argmax(0).detach()

                    del input_for_predict

                    # detect changes at borders
                    previous_zoom_prediction = crop_and_pad_nd(self.interactions[0], scaled_bbox).to(self.device, non_blocking=self.device.type == 'cuda')
                    if not all([i == j for i, j in zip(pred.shape, previous_zoom_prediction.shape)]):
                        previous_zoom_prediction = interpolate(previous_zoom_prediction[None, None].to(float), pred.shape, mode='nearest')[0, 0]

                    abs_pxl_change_threshold = 1500
                    rel_pxl_change_threshold = 0.2
                    min_pxl_change_threshold = 100
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
                                rel_change = max(pixels_prev, pixels_current) / max(min(pixels_prev, pixels_current),
                                                                                    1e-5) - 1
                                if pixels_diff > abs_pxl_change_threshold:
                                    continue_zoom = True
                                    if self.verbose:
                                        print(f'continue zooming because change at borders of {pixels_diff} > {abs_pxl_change_threshold}')
                                    break
                                if pixels_diff > min_pxl_change_threshold and rel_change > rel_pxl_change_threshold:
                                    continue_zoom = True
                                    if self.verbose:
                                        print(f'continue zooming because relative change of {rel_change} > {rel_pxl_change_threshold} and n_pixels {pixels_diff} > {min_pxl_change_threshold}')
                                    break
                                del slice_prev, slice_curr, pixels_prev, pixels_current, pixels_diff

                        del previous_zoom_prediction

                    # resize prediction to correct size and place in target buffer + interactions
                    if not all([i == j for i, j in zip(pred.shape, scaled_patch_size)]):
                        pred = (interpolate(pred[None, None].to(float), scaled_patch_size, mode='trilinear')[0, 0] >= 0.5).to(torch.uint8)

                    # if we do not continue zooming we need a difference map for sampling patches
                    if not continue_zoom and zoom_out_factor > 1:
                        # wow this circus saves ~30ms relative to naive implementation
                        previous_prediction = previous_prediction.to(self.device, non_blocking=self.device.type == 'cuda')
                        seen_bbox = [[max(0, i[0]), min(i[1], s)] for i, s in zip(scaled_bbox, previous_prediction.shape)]
                        bbox_tmp = [[i[0] - s[0], i[1] - s[0]] for i, s in zip(seen_bbox, scaled_bbox)]
                        bbox_tmp = [[max(0, i[0]), min(i[1], s)] for i, s in zip(bbox_tmp, scaled_patch_size)]
                        slicer = bounding_box_to_slice(seen_bbox)
                        slicer2 = bounding_box_to_slice(bbox_tmp)
                        diff_map = pred[slicer2] != previous_prediction[slicer]
                        # dont allocate new memory, just reuse previous_prediction. We don't need it anymore
                        previous_prediction.zero_()
                        diff_map = paste_tensor(previous_prediction, diff_map, seen_bbox)

                        # open the difference map to keep computational load in check (fewer refinement boxes)
                        # open distance map
                        diff_map[slicer] = iterative_3x3_same_padding_pool3d(diff_map[slicer][None, None], kernel_size=5, use_min_pool=True)[0, 0]
                        diff_map[slicer] = iterative_3x3_same_padding_pool3d(diff_map[slicer][None, None], kernel_size=5, use_min_pool=False)[0, 0]

                        has_diff = torch.any(diff_map[slicer])

                        del previous_prediction
                    else:
                        has_diff = False

                    if zoom_out_factor == 1 or (not continue_zoom and has_diff): # rare case where no changes are needed because of useless interaction. Need to check for not continue_zoom because otherwise diff_map wint exist
                        pred = pred.cpu()

                        if zoom_out_factor == 1:
                            paste_tensor(self.interactions[0], pred.half(), scaled_bbox)
                        else:
                            seen_bbox = [[max(0, i[0]), min(i[1], s)] for i, s in
                                         zip(scaled_bbox, diff_map.shape)]
                            bbox_tmp = [[i[0] - s[0], i[1] - s[0]] for i, s in zip(seen_bbox, scaled_bbox)]
                            bbox_tmp = [[max(0, i[0]), min(i[1], s)] for i, s in zip(bbox_tmp, scaled_patch_size)]
                            slicer = bounding_box_to_slice(seen_bbox)
                            slicer2 = bounding_box_to_slice(bbox_tmp)
                            mask = (diff_map[slicer] > 0).cpu()
                            self.interactions[0][slicer][mask] = pred[slicer2][mask].half()

                        # place into target buffer
                        bbox = [[i[0] + bbc[0], i[1] + bbc[0]] for i, bbc in
                                zip(scaled_bbox, self.preprocessed_props['bbox_used_for_cropping'])]
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

                if max_zoom_out_factor > 1 and has_diff:
                    start_refinement = time()
                    # only use the region that was previously looked at. Use last scaled_bbox
                    if self.has_positive_bbox:
                        # mask positive bbox channel with current segmentation to avoid bbox nonsense.
                        # Basically convert bbox to pseudo lasso
                        pos_bbox_idx = -6
                        self.interactions[pos_bbox_idx][(~(self.interactions[0] > 0.5)).cpu()] = 0
                        self.has_positive_bbox = False

                    bboxes_ordered = generate_bounding_boxes(diff_map, self.configuration_manager.patch_size, stride='auto', margin=(10, 10, 10), max_depth=3)

                    del diff_map
                    empty_cache(self.device)

                    if self.verbose:
                        print(f'Using {len(bboxes_ordered)} bounding boxes for refinement')

                    preallocated_input = torch.zeros((8, *self.configuration_manager.patch_size), device=self.device, dtype=torch.float)
                    for nref, refinement_bbox in enumerate(bboxes_ordered):
                        assert self.pad_mode_data == 'constant'
                        crop_and_pad_into_buffer(preallocated_input[0], refinement_bbox, self.preprocessed_image[0])
                        crop_and_pad_into_buffer(preallocated_input[1:], refinement_bbox, self.interactions)
                        
                        pred = self.network(preallocated_input[None])[0].argmax(0).detach().cpu()

                        paste_tensor(self.interactions[0], pred, refinement_bbox)
                        # place into target buffer
                        bbox = [[i[0] + bbc[0], i[1] + bbc[0]] for i, bbc in zip(refinement_bbox, self.preprocessed_props['bbox_used_for_cropping'])]
                        paste_tensor(self.target_buffer, pred, bbox)
                        del pred
                        preallocated_input.zero_()
                    del preallocated_input
                    empty_cache(self.device)
                    end_refinement = time()
                    print(f'Took {round(end_refinement - start_refinement, 3)} s for refining the segmentation with {len(bboxes_ordered)} bounding boxes')
                else:
                    print('No refinement necessary')
        print(f'Done. Total time {round(time() - start_predict, 3)}s')

        self.new_interaction_centers = []
        self.new_interaction_zoom_out_factors = []
        empty_cache(self.device)

    def _add_patch_for_point_interaction(self, coordinates):
        self.new_interaction_zoom_out_factors.append(1)
        self.new_interaction_centers.append(coordinates)
        print(f'Added new point interaction: center {self.new_interaction_zoom_out_factors[-1]}, scale {self.new_interaction_centers}')

    def _add_patch_for_bbox_interaction(self, bbox):
        bbox_center = [round((i[0] + i[1]) / 2) for i in bbox]
        bbox_size = [i[1]-i[0] for i in bbox]
        # we want to see some context, so the crop we see for the initial prediction should be patch_size / 3 larger
        requested_size = [i + j // 3 for i, j in zip(bbox_size, self.configuration_manager.patch_size)]
        self.new_interaction_zoom_out_factors.append(max(1, max([i / j for i, j in zip(requested_size, self.configuration_manager.patch_size)])))
        self.new_interaction_centers.append(bbox_center)
        print(f'Added new bbox interaction: center {self.new_interaction_zoom_out_factors[-1]}, scale {self.new_interaction_centers}')

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
        print(f'Added new image interaction: scale {self.new_interaction_zoom_out_factors[-1]}, center {self.new_interaction_centers}')

    def initialize_from_trained_model_folder(self, model_training_output_dir: str,
                                             use_fold: Union[int, str] = None,
                                             checkpoint_name: str = 'checkpoint_final.pth'):
        """
        This is used when making predictions with a trained model
        """
        # load trainer specific settings
        expected_json_file = join(model_training_output_dir, 'inference_session_class.json')
        json_content = load_json(expected_json_file)
        if isinstance(json_content, str):
            # old convention where we only specified the inference class in this file. Set defaults for stuff
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
            point_interaction_use_etd = True # so far this is not defined in that file so we stick with default
            self.point_interaction = PointInteraction_stub(point_interaction_radius, point_interaction_use_etd)
            # padding mode for data. See nnInteractiveTrainerV2_nodelete_reflectpad
            self.pad_mode_data = json_content['pad_mode_image'] if 'pad_mode_image' in json_content.keys() else "constant"

        dataset_json = load_json(join(model_training_output_dir, 'dataset.json'))
        plans = load_json(join(model_training_output_dir, 'plans.json'))
        plans_manager = PlansManager(plans)

        if use_fold is not None:
            use_fold = int(use_fold) if use_fold != 'all' else use_fold
            fold_folder = f'fold_{use_fold}'
        else:
            fldrs = subdirs(model_training_output_dir, prefix='fold_', join=False)
            assert len(fldrs) == 1, f'Attempted to infer fold but there is != 1 fold_ folders: {fldrs}'
            fold_folder = fldrs[0]

        checkpoint = torch.load(join(model_training_output_dir, fold_folder, checkpoint_name),
                                map_location=self.device, weights_only=False)
        trainer_name = checkpoint['trainer_name']
        configuration_name = checkpoint['init_args']['configuration']

        parameters = checkpoint['network_weights']

        configuration_manager = plans_manager.get_configuration(configuration_name)
        # restore network
        num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
        trainer_class = recursive_find_python_class(join(nnInteractive.__path__[0], "trainer"),
                                                    trainer_name, 'nnInteractive.trainer')
        if trainer_class is None:
            print(f'Unable to locate trainer class {trainer_name} in nnInteractive.trainer. '
                               f'Please place it there (in any .py file)!')
            print('Attempting to use default nnInteractiveTrainer_stub. If you encounter errors, this is where you need to look!')
            trainer_class = nnInteractiveTrainer_stub

        network = trainer_class.build_network_architecture(
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels,
            plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
            enable_deep_supervision=False
        ).to(self.device)
        network.load_state_dict(parameters)

        self.plans_manager = plans_manager
        self.configuration_manager = configuration_manager
        self.network = network
        self.dataset_json = dataset_json
        self.trainer_name = trainer_name
        self.label_manager = plans_manager.get_label_manager(dataset_json)
        if self.use_torch_compile and not isinstance(self.network, OptimizedModule):
            print('Using torch.compile')
            self.network = torch.compile(self.network)

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


def transform_coordinates_noresampling(
        coords_orig: Union[List[int], Tuple[int, ...]],
        nnunet_preprocessing_crop_bbox: List[Tuple[int, int]]
) -> Tuple[int, ...]:
    """
    converts coordinates in the original uncropped image to the internal cropped representation. Man I really hate
    nnU-Net's crop to nonzero!
    """
    return tuple([coords_orig[d] - nnunet_preprocessing_crop_bbox[d][0] for d in range(len(coords_orig))])


