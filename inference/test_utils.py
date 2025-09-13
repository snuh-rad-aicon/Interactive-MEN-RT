import os
import torch
import numpy as np
import random
from monai.transforms import (
    Compose, LoadImaged, SpatialPadd,
    EnsureChannelFirstd, Orientationd, NormalizeIntensityd
)
from skimage.draw import line, polygon
import skimage.segmentation
from scipy import ndimage

def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
def get_checkpoint_path(model_name):
    if model_name == "AutoInteractive_MEN_RT":
        return "/mnt/hdd3/hjang/scripts/MICCAI_2025_CLIP/checkpoints/nnUNetAutoInteractionTrainer__nnUNetPlans__3d_fullres"
    elif model_name == "AutoInteractive_MEN_RT_scratch":
        return "/mnt/hdd3/hjang/scripts/MICCAI_2025_CLIP/checkpoints/nnUNetAutoInteractionTrainer__nnUNetPlans__3d_fullres_scratch"
    if model_name == "Interactive_MEN_RT":
        return "/mnt/hdd3/hjang/scripts/MICCAI_2025_CLIP/checkpoints/Interactive_MEN_RT_point_manual_infer"
    elif model_name == "Interactive_MEN_RT_scratch":
        return "/mnt/hdd3/hjang/scripts/MICCAI_2025_CLIP/checkpoints/Interactive_MEN_RT_point_manual_infer"
    elif model_name == "nnUNet_MEN_RT":
        return "/mnt/hdd3/hjang/scripts/MICCAI_2025_CLIP/checkpoints/nnUNetTrainer__nnUNetPlans__3d_fullres_scratch"
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

def get_false_pos(gt, pred):
    """
    Get the largest 3 false positive regions from ground truth and predicted segmentations, ignoring components with <= 100 voxels
    """
    false_pos = (pred > 0) & (gt == 0)
    labeled, num = ndimage.label(false_pos)
    if num == 0:
        return false_pos  # all False
    sizes = ndimage.sum(false_pos, labeled, range(1, num+1))
    # Ignore components with <= 100 voxels
    valid_labels = np.where(sizes > 100)[0] + 1  # component labels are 1-based
    if len(valid_labels) == 0:
        return false_pos * False  # all False
    # Sort by size and pick largest 3
    largest = valid_labels[np.argsort(sizes[valid_labels-1])[-3:][::-1]]
    mask = np.isin(labeled, largest)
    return mask
    
def get_false_neg(gt, pred):
    """
    Get the largest 3 false negative regions from ground truth and predicted segmentations, ignoring components with <= 100 voxels
    
    Args:
        gt: Ground truth segmentation
        pred: Predicted segmentation
    Returns:
        false_neg_top3: Boolean mask with only the largest 3 false negative components (or fewer if less exist)
    """
    false_neg = (pred == 0) & (gt > 0)
    labeled, num = ndimage.label(false_neg)
    if num == 0:
        return false_neg  # all False
    sizes = ndimage.sum(false_neg, labeled, range(1, num+1))
    # Ignore components with <= 100 voxels
    valid_labels = np.where(sizes > 100)[0] + 1  # component labels are 1-based
    if len(valid_labels) == 0:
        return false_neg * False  # all False
    # Sort by size and pick largest 3
    largest = valid_labels[np.argsort(sizes[valid_labels-1])[-3:][::-1]]
    mask = np.isin(labeled, largest)
    return mask

def simulate_interaction(session, interaction_type, mask_np, include_interaction=True, run_prediction=True):
    # Label connected components
    labeled_mask, num_components = ndimage.label(mask_np)
    
    if num_components == 0:
        print("Warning: No components found in mask. Skipping interaction.")
        return
    
    # Process each component
    for component_id in range(1, num_components + 1):
        component_mask = (labeled_mask == component_id)
        coords = np.argwhere(component_mask)
        center = np.mean(coords, axis=0)
        center = np.round(center).astype(int)
        
        if interaction_type == "point":
            session.add_point_interaction(tuple(center), include_interaction=include_interaction, run_prediction=False)
        elif interaction_type == "bbox":
            mins = coords.min(axis=0)
            maxs = coords.max(axis=0) + 1
            bbox = [(int(mins[0]), int(maxs[0])), (int(mins[1]), int(maxs[1])), (int(mins[2]), int(maxs[2]))]
            session.add_bbox_interaction(bbox, include_interaction=include_interaction, run_prediction=False)
        elif interaction_type in ["scribble", "lasso"]:
            z = int(center[2])
            region_slice = component_mask[:, :, z]
            mask2d = np.zeros(mask_np.shape, dtype=np.uint8)
            H, W = region_slice.shape

            if interaction_type == "scribble":
                region_slice_tensor = torch.from_numpy(region_slice.astype(np.uint8)).to(session.device)
                xy_coords = torch.nonzero(region_slice_tensor, as_tuple=False)

                if xy_coords.shape[0] >= 2:
                    num_pts_to_select = min(6, xy_coords.shape[0])
                    
                    if xy_coords.shape[0] >=2 and num_pts_to_select < 2:
                        num_pts_to_select = 2
                    
                    if num_pts_to_select >= 2:
                        chosen_idx_tensor = torch.randperm(xy_coords.shape[0], device=session.device)[:num_pts_to_select]
                        pts = xy_coords[chosen_idx_tensor].cpu().numpy().astype(int)

                        if pts.shape[0] >= 2: 
                            order = np.lexsort((pts[:, 0], pts[:, 1])) 
                            pts = pts[order]

                            rr_list, cc_list = [], []
                            for i in range(pts.shape[0] - 1): 
                                r0, c0 = pts[i]
                                r1, c1 = pts[i + 1]
                                rr_slice, cc_slice = line(r0, c0, r1, c1)
                                rr_list.append(rr_slice)
                                cc_list.append(cc_slice)
                            
                            if rr_list: 
                                rr_combined = np.concatenate(rr_list)
                                cc_combined = np.concatenate(cc_list)
                                
                                valid_indices = (rr_combined >= 0) & (rr_combined < H) & \
                                                (cc_combined >= 0) & (cc_combined < W)
                                rr_final = rr_combined[valid_indices]
                                cc_final = cc_combined[valid_indices]
                                
                                if rr_final.size > 0 and cc_final.size > 0: 
                                    mask2d[rr_final, cc_final, z] = 1
            else:  # lasso
                boundary2d_np = skimage.segmentation.find_boundaries(region_slice.astype(bool), mode='outer')
                boundary2d_tensor = torch.from_numpy(boundary2d_np).to(session.device)
                
                xy_coords_boundary = torch.nonzero(boundary2d_tensor, as_tuple=False)

                if xy_coords_boundary.shape[0] >= 4:
                    num_pts_to_select = min(6, xy_coords_boundary.shape[0])
                    
                    if num_pts_to_select < 3 and xy_coords_boundary.shape[0] >=3:
                        num_pts_to_select = 3

                    if num_pts_to_select >=3: 
                        chosen_idx_tensor = torch.randperm(xy_coords_boundary.shape[0], device=session.device)[:num_pts_to_select]
                        pts = xy_coords_boundary[chosen_idx_tensor].cpu().numpy().astype(int)

                        if pts.shape[0] >= 3:
                            centroid = pts.mean(axis=0)
                            angles = np.arctan2(pts[:, 0] - centroid[0], pts[:, 1] - centroid[1])
                            order = np.argsort(angles)
                            pts_sorted = pts[order]

                            rr_poly, cc_poly = polygon(pts_sorted[:, 0], pts_sorted[:, 1], shape=(H, W))
                            
                            valid_indices_poly = (rr_poly >= 0) & (rr_poly < H) & \
                                                 (cc_poly >= 0) & (cc_poly < W)
                            rr_final_poly = rr_poly[valid_indices_poly]
                            cc_final_poly = cc_poly[valid_indices_poly]

                            if rr_final_poly.size > 0 and cc_final_poly.size > 0:
                                mask2d[rr_final_poly, cc_final_poly, z] = 1
            
            if np.sum(mask2d) == 0:
                print(f"Warning: No {interaction_type} was generated for component {component_id}. Adding a center point interaction instead.")
                session.add_point_interaction(tuple(center), include_interaction=include_interaction, run_prediction=False)
            elif interaction_type == "scribble":
                session.add_scribble_interaction(mask2d, include_interaction=include_interaction, run_prediction=False)
            else: # lasso
                session.add_lasso_interaction(mask2d, include_interaction=include_interaction, run_prediction=False)
        else:
            raise ValueError(f"Unknown interaction_type: {interaction_type}")
    
    if run_prediction:
        session._predict()

def simulate_interaction_variation(session, interaction_type, mask_np, include_interaction=True):
    POINT_VAR_NUM_POINTS = 1
    BBOX_VAR_MARGIN_RATIO_MIN = 0.05
    BBOX_VAR_MARGIN_RATIO_MAX = 0.15

    coords = np.argwhere(mask_np > 0)
    center = np.mean(coords, axis=0)
    center = np.round(center).astype(int)
        
    H_vol, W_vol, D_vol = mask_np.shape
    mask_tensor_full = torch.from_numpy(mask_np.astype(np.float32)).to(session.device)

    if interaction_type == "point":
        if coords.shape[0] == 0:
            print("Warning: No positive labels in mask_np for point variation. Adding center point.")
            session.add_point_interaction(tuple(center), include_interaction=True, run_prediction=False)
            return

        slice_sums_for_points = mask_tensor_full.sum(dim=(0, 1))
        z_coords_all = coords[:, 2]
        
        weights_for_coords = slice_sums_for_points[z_coords_all]
        weights_for_coords[weights_for_coords <= 0] = 1e-6

        if torch.sum(weights_for_coords) == 0:
            probs = torch.ones(coords.shape[0], device=session.device) / coords.shape[0]
        else:
            probs = weights_for_coords / torch.sum(weights_for_coords)
        
        num_actual_points = min(POINT_VAR_NUM_POINTS, coords.shape[0])
        if num_actual_points > 0:
            indices = torch.multinomial(probs, num_actual_points, replacement=False)
            chosen_pts_3d = coords[indices.cpu().numpy()]

            session.add_point_interaction(tuple(chosen_pts_3d[0].astype(int)), include_interaction=True, run_prediction=False)
        else:
            print("Warning: Could not select any points for point variation. Adding center point.")
            session.add_point_interaction(tuple(center), include_interaction=True, run_prediction=False)

    elif interaction_type == "bbox":
        if coords.shape[0] == 0:
            print("Warning: No positive labels in mask_np for bbox variation. Skipping bbox.")
            session.add_point_interaction(tuple(center), include_interaction=True, run_prediction=False)
            return

        mins_orig = coords.min(axis=0)
        maxs_orig = coords.max(axis=0) + 1
        
        mins = mins_orig.copy()
        maxs = maxs_orig.copy()

        mr = np.random.uniform(BBOX_VAR_MARGIN_RATIO_MIN, BBOX_VAR_MARGIN_RATIO_MAX)
        for i in range(3):
            size = maxs[i] - mins[i]
            margin_val = max(1, int(size * mr))
            mins[i] = max(0, mins[i] - margin_val)
            maxs[i] = min(mask_np.shape[i], maxs[i] + margin_val)

        for i in range(3):
            max_jitter = max(1, int(mask_np.shape[i] * 0.02))
            jitter = np.random.randint(-max_jitter, max_jitter + 1)
            mins[i] = max(0, mins[i] + jitter)
            maxs[i] = min(mask_np.shape[i], maxs[i] + jitter)
            if mins[i] >= maxs[i]:
                maxs[i] = mins[i] + 1 if mins[i] < mask_np.shape[i] -1 else mask_np.shape[i]
                mins[i] = maxs[i] -1 if maxs[i] > 0 else 0
        
        for i in range(3):
            if maxs[i] > mins[i]:
                size_var = np.random.uniform(0.8, 1.2)
                current_center = (mins[i] + maxs[i]) / 2
                half_size = (maxs[i] - mins[i]) / 2 * size_var
                mins[i] = max(0, int(current_center - half_size))
                maxs[i] = min(mask_np.shape[i], int(current_center + half_size))
                if mins[i] >= maxs[i]:
                    maxs[i] = mins[i] + 1 if mins[i] < mask_np.shape[i] -1 else mask_np.shape[i]
                    mins[i] = maxs[i] -1 if maxs[i] > 0 else 0
        
        valid_bbox = True
        for i in range(3):
            mins[i] = np.clip(mins[i], 0, mask_np.shape[i] -1)
            maxs[i] = np.clip(maxs[i], mins[i] + 1, mask_np.shape[i])
            if mins[i] >= maxs[i]:
                valid_bbox = False
                break
        
        if valid_bbox:
            bbox_interaction = [(int(mins[0]), int(maxs[0])), (int(mins[1]), int(maxs[1])), (int(mins[2]), int(maxs[2]))]
            session.add_bbox_interaction(bbox_interaction, include_interaction=True, run_prediction=False)
        else:
            print("Warning: BBox variation resulted in invalid dimensions. Using original bbox or fallback.")
            bbox_orig_interaction = [(int(mins_orig[0]), int(maxs_orig[0])), (int(mins_orig[1]), int(maxs_orig[1])), (int(mins_orig[2]), int(maxs_orig[2]))]
            session.add_bbox_interaction(bbox_orig_interaction, include_interaction=True, run_prediction=False)

    elif interaction_type in ["scribble", "lasso"]:
        mask2d_interaction = np.zeros(mask_np.shape, dtype=np.uint8)

        slice_sums = mask_tensor_full.sum(dim=(0, 1))
        positive_slices_indices = torch.nonzero(slice_sums > 0, as_tuple=False).flatten()

        if len(positive_slices_indices) == 0:
            print(f"Warning: No positive slices found for {interaction_type} variation. Adding center point.")
            session.add_point_interaction(tuple(center), include_interaction=True, run_prediction=False)
            return
        
        sums_for_positive_slices = slice_sums[positive_slices_indices].cpu().numpy()
        if np.sum(sums_for_positive_slices) == 0:
            slice_z_index = np.random.choice(positive_slices_indices.cpu().numpy())
        else:
            probabilities = sums_for_positive_slices / np.sum(sums_for_positive_slices)
            probabilities = probabilities / np.sum(probabilities)
            slice_z_index = np.random.choice(positive_slices_indices.cpu().numpy(), p=probabilities)

        region_slice_2d = mask_np[:, :, slice_z_index]
        H_slice, W_slice = region_slice_2d.shape

        if interaction_type == "scribble":
            region_slice_tensor = torch.from_numpy(region_slice_2d.astype(np.uint8)).to(session.device)
            xy_coords_slice = torch.nonzero(region_slice_tensor, as_tuple=False)

            if xy_coords_slice.shape[0] < 2:
                pass 
            else:
                max_pts = min(8, xy_coords_slice.shape[0])
                if max_pts < 2: num_pts = max_pts
                else: num_pts = np.random.randint(2, max_pts + 1)
                
                if num_pts >= 2:
                    chosen_idx = torch.randperm(xy_coords_slice.shape[0], device=session.device)[:num_pts]
                    pts = xy_coords_slice[chosen_idx].cpu().numpy().astype(int)

                    if np.random.rand() < 0.5:
                        order = np.argsort(pts[:, 1], kind='stable')
                    else:
                        order = np.lexsort((pts[:, 1], pts[:, 0]))
                    pts = pts[order]

                    jitter_amount = int(min(H_slice, W_slice) * 0.05)
                    jitter = np.random.randint(-jitter_amount, jitter_amount+1, size=pts.shape)
                    pts = np.clip(pts + jitter, a_min=[0, 0], a_max=[H_slice - 1, W_slice - 1])
                    
                    rr_list, cc_list = [], []
                    for i in range(pts.shape[0] - 1):
                        r0, c0 = pts[i, 0], pts[i, 1]
                        r1, c1 = pts[i+1, 0], pts[i+1, 1]
                        rr_segment, cc_segment = line(r0, c0, r1, c1)
                        rr_list.append(rr_segment)
                        cc_list.append(cc_segment)
                    
                    if rr_list:
                        rr = np.concatenate(rr_list)
                        cc = np.concatenate(cc_list)
                        wave_amount = int(min(H_slice,W_slice) * 0.02)
                        rr = np.clip(rr + np.random.randint(-wave_amount, wave_amount+1, size=rr.shape), 0, H_slice - 1)
                        cc = np.clip(cc + np.random.randint(-wave_amount, wave_amount+1, size=cc.shape), 0, W_slice - 1)
                        
                        valid_indices = (rr >= 0) & (rr < H_slice) & (cc >= 0) & (cc < W_slice)
                        mask2d_interaction[rr[valid_indices], cc[valid_indices], slice_z_index] = 1
        
        else:  # lasso
            boundary2d_np = skimage.segmentation.find_boundaries(region_slice_2d.astype(bool), mode='outer')
            xy_coords_boundary = torch.nonzero(torch.from_numpy(boundary2d_np).to(session.device), as_tuple=False)

            if xy_coords_boundary.shape[0] < 3:
                 pass
            else:
                n_min, n_max = 3, 12
                num_pts_to_select = np.random.randint(n_min, min(n_max, xy_coords_boundary.shape[0]) + 1)
                
                if num_pts_to_select >=3:
                    chosen_idx = torch.randperm(xy_coords_boundary.shape[0], device=session.device)[:num_pts_to_select]
                    pts = xy_coords_boundary[chosen_idx].cpu().numpy().astype(int)

                    jitter_amount = int(min(H_slice, W_slice) * 0.05)
                    jit = np.random.randint(-jitter_amount, jitter_amount+1, size=pts.shape)
                    pts = pts + jit
                    pts[:, 0] = np.clip(pts[:, 0], 0, H_slice - 1)
                    pts[:, 1] = np.clip(pts[:, 1], 0, W_slice - 1)

                    if pts.shape[0] > n_min and np.random.random() < 0.5:
                        drop = np.random.randint(0, pts.shape[0])
                        pts = np.delete(pts, drop, axis=0)
                    elif xy_coords_boundary.shape[0] > pts.shape[0] and pts.shape[0] < n_max and np.random.random() < 0.5:
                        extra_idx = torch.randperm(xy_coords_boundary.shape[0], device=session.device)[:1]
                        extra = xy_coords_boundary[extra_idx].cpu().numpy().astype(int)
                        pts = np.vstack([pts, extra])
                    
                    if pts.shape[0] >= 3:
                        centroid = pts.mean(axis=0)
                        angles = np.arctan2(pts[:, 0] - centroid[0], pts[:, 1] - centroid[1])
                        order = np.argsort(angles)
                        pts_sorted = pts[order]

                        rr_poly, cc_poly = polygon(pts_sorted[:, 0], pts_sorted[:, 1], shape=(H_slice, W_slice))
                        valid_indices_poly = (rr_poly >= 0) & (rr_poly < H_slice) & (cc_poly >= 0) & (cc_poly < W_slice)
                        mask2d_interaction[rr_poly[valid_indices_poly], cc_poly[valid_indices_poly], slice_z_index] = 1

        if np.sum(mask2d_interaction) == 0:
            print(f"Warning: No {interaction_type} (variation) was generated on slice {slice_z_index}. Adding a center point interaction instead.")
            session.add_point_interaction(tuple(center), include_interaction=True, run_prediction=False)
        elif interaction_type == "scribble":
            session.add_scribble_interaction(mask2d_interaction, include_interaction=True, run_prediction=False)
        else: # lasso
            session.add_lasso_interaction(mask2d_interaction, include_interaction=True, run_prediction=False)
    else:
        raise ValueError(f"Unknown interaction_type: {interaction_type}")

class MENRTDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, patch_size=(192,192,192), transform=None, mode='train'):
        self.data_dir = f'{data_dir}/{mode}'
        self.transform = transform
        self.patch_size = patch_size # patch_size is used, config is not directly used
        self.cases = sorted([d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))])

    def __len__(self):
        return len(self.cases)
    
    def __getitem__(self, idx):
        case = self.cases[idx]
        case_dir = os.path.join(self.data_dir, case)
        img_path = os.path.join(case_dir, f'{case}_t1c.nii.gz')
        seg_path = os.path.join(case_dir, f'{case}_gtv.nii.gz')
        
        data = {
            'image': img_path,
            'mask': seg_path,
            'case': case
        }
        
        if self.transform: # Apply transform only if it exists
            data = self.transform(data)
            if isinstance(data, list): # MONAI transforms can return a list
                data = data[0]

        return {
            'image': data['image'],
            'mask': data['mask'],
            'case': data['case']
        }

def get_transforms(config_patch_size): # Takes patch_size from config
    return Compose([
        LoadImaged(keys=['image', 'mask']),
        EnsureChannelFirstd(keys=['image', 'mask']),
        Orientationd(keys=['image', 'mask'], axcodes='RAS'),
        SpatialPadd(keys=['image', 'mask'], spatial_size=config_patch_size),
        # NormalizeIntensityd(keys=['image'], nonzero=True, channel_wise=True),
    ]) 