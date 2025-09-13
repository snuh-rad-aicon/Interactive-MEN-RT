import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import ants
from dataclasses import dataclass, field
from typing import List
from skimage.morphology import skeletonize

from Interactive_MEN_RT_predictor import InteractiveMENRTPredictor
from test_utils import (
    set_seed, 
    get_checkpoint_path,
    get_false_pos,
    get_false_neg,
    simulate_interaction, 
    simulate_interaction_variation, 
    MENRTDataset, 
    get_transforms
)

@dataclass
class TrainingConfig:
    # Data settings
    data_dir: str = "/mnt/hdd3/hjang/data/Meningioma/Interactive_MEN_RT/data"
    save_dir: str = "checkpoints"
    patch_size: tuple = (192, 192, 192)
    device: str = "cuda"
    num_workers: int = 16
    pin_memory: bool = True
    seed: int = 42

def main(model_name, interaction_type='None', variation: str = "auto", gpu_id: int = 0):
    config = TrainingConfig()
    set_seed(config.seed) 

    device_str = config.device
    if config.device == "cuda" and torch.cuda.is_available():
        device_str = f"cuda:{gpu_id}"
    elif config.device == "cuda":
        print("CUDA specified but not available. Falling back to CPU.")
        device_str = "cpu"
    
    current_device = torch.device(device_str)

    save_dir = os.path.join(config.save_dir, f"{model_name}_{interaction_type}_{variation}_infer")
    os.makedirs(save_dir, exist_ok=True)

    checkpoint_path = get_checkpoint_path(model_name)
    predictor = InteractiveMENRTPredictor(
        device=current_device,
        use_torch_compile=False,
        do_autozoom=False,
        verbose=False
    )
    predictor.initialize_from_trained_model_folder(
        model_training_output_dir=checkpoint_path,
        use_fold=0,
        checkpoint_name='checkpoint_best.pth'
    )
    
    val_transform = get_transforms(config.patch_size)
    val_dataset = MENRTDataset(config.data_dir, patch_size=config.patch_size, transform=val_transform, mode='val')
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )

    for batch in tqdm(val_loader, desc=f'{interaction_type}_{variation} Inference'):
        images = batch['image'].cpu().numpy() 
        masks = batch['mask'].cpu().numpy()   
        case = batch['case'][0] if isinstance(batch['case'], list) else batch['case']

        image_np = images[0]
        mask_np = masks[0, 0]
        if np.sum(mask_np) == 0:
            print(f"No tumor found in case {case}, skipping.")
            continue
        
        predictor.set_image(image_np)
        predictor.set_target_buffer(np.zeros(mask_np.shape, dtype=np.float32))
        predictor.reset_interactions()
        
        if variation == "auto":
            predictor._finish_preprocessing_and_initialize_interactions()
            predictor._predict_without_interaction()
        elif variation == "manual":
            simulate_interaction(predictor, interaction_type, mask_np, include_interaction=True)
        elif variation == "auto_manual":
            predictor._finish_preprocessing_and_initialize_interactions()
            predictor._predict_without_interaction()
            pred_np = predictor.target_buffer.astype(np.float32)
            false_pos = get_false_pos(mask_np, pred_np)
            simulate_interaction(predictor, interaction_type, false_pos, include_interaction=False, run_prediction=False)
            false_neg = get_false_neg(mask_np, pred_np)
            simulate_interaction(predictor, interaction_type, false_neg, include_interaction=True, run_prediction=False)
            predictor._predict()
            
        pred_np = predictor.target_buffer.astype(np.float32)
        
        ants_img = ants.from_numpy(image_np[0])
        ants_mask = ants.from_numpy(mask_np)
        ants_pred_original = ants.from_numpy(pred_np)
        
        ants.image_write(ants_img, os.path.join(save_dir, f'{case}_image.nii.gz'))
        ants.image_write(ants_mask, os.path.join(save_dir, f'{case}_mask.nii.gz'))
        ants.image_write(ants_pred_original, os.path.join(save_dir, f'{case}_pred.nii.gz'))

if __name__ == '__main__':
    model_names = ["Interactive_MEN_RT", "Interactive_MEN_RT_scratch"]
    checkpoint_paths = [
        "/mnt/hdd3/hjang/scripts/MICCAI_2025_CLIP/checkpoints/nnUNetInteractionTrainer__nnUNetPlans__3d_fullres",
        "/mnt/hdd3/hjang/scripts/MICCAI_2025_CLIP/checkpoints/nnUNetInteractionTrainer__nnUNetPlans__3d_fullres_scratch"
    ]

    interaction_types = ["point", "bbox", "scribble", "lasso"]
    variation_options = ['manual']
    gpu_id = 0
    
    for model_name in model_names:
        for var_option in variation_options:
            if var_option == 'auto':
                print(f"Running inference for model: {model_name}, mode: {var_option}")
                main(model_name, variation=var_option, gpu_id=gpu_id)
            else:
                for interaction_type in interaction_types:
                    print(f"Running inference for interaction type: {interaction_type}, mode: {var_option}")
                    main(model_name, interaction_type, variation=var_option, gpu_id=gpu_id)
