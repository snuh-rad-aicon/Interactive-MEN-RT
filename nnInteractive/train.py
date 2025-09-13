#!/usr/bin/env python3
import os
import argparse
import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import nibabel as nib
from monai.transforms import (
    Compose, LoadImaged, ScaleIntensityd, Rand3DElasticd, RandAffined, RandGaussianSmoothd,
    RandFlipd, RandScaleIntensityd, RandShiftIntensityd,
    RandAdjustContrastd, RandGaussianSharpend, RandHistogramShiftd,
    RandCoarseDropoutd, RandSpatialCropd, SpatialPadd,
    EnsureChannelFirstd, Orientationd, RandGaussianNoised, NormalizeIntensityd,
    RandCropByPosNegLabeld, 
)

from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, load_json
from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed, nnUNet_results
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name

from nnInteractive.trainer import nnInteractiveTrainer


class MENRTDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, patch_size=(192,192,192), transform=None, mode='train'):
        """
        Dataset for BraTS-MEN-RT data
        
        Args:
            data_dir: Path to the data directory
            patch_size: Size of the patches to extract
            transform: Transforms to apply to the data
            mode: 'train' or 'test'
        """
        self.data_dir = f'{data_dir}/{mode}'
        self.transform = transform
        self.patch_size = patch_size
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
        
        if self.transform:
            data = self.transform(data)
            if isinstance(data, list):
                data = data[0]
        
        # Convert data to the format expected by nnInteractiveTrainer
        if 'image' in data and 'mask' in data:
            return {
                'data': data['image'],
                'target': data['mask'],
                'case_id': data['case']
            }
        else:
            return data


def get_transforms(config, mode='train'):
    if mode == 'train':
        return Compose([
            LoadImaged(keys=['image', 'mask']),
            EnsureChannelFirstd(keys=['image', 'mask']),
            Orientationd(keys=['image', 'mask'], axcodes='RAS'),
            # Spacingd(keys=['image', 'mask'], pixdim=(1.0, 1.0, 1.0), mode=('bilinear', 'nearest')),
            SpatialPadd(keys=['image', 'mask'], spatial_size=config.patch_size),
            RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=0),
            RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=1),
            RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=2),
            RandAffined(
                keys=['image', 'mask'],
                rotate_range=(np.pi/6, np.pi/6, np.pi/6),  # ±30°
                scale_range=(0.1, 0.1, 0.1),                  # ±10%
                mode=('bilinear', 'nearest'),
                prob=0.2
            ),
            Rand3DElasticd(
                keys=['image', 'mask'],
                sigma_range=(5, 7),
                magnitude_range=(100, 200),
                mode=('bilinear', 'nearest'),
                prob=0.2
            ),
            RandGaussianSmoothd(
                keys=['image'],
                prob=0.2,
                sigma_x=(0.5, 1.0),
                sigma_y=(0.5, 1.0),
                sigma_z=(0.5, 1.0)
            ),
            RandScaleIntensityd(keys=['image'], prob=0.2, factors=0.1),
            RandShiftIntensityd(keys=['image'], prob=0.2, offsets=0.1),
            RandAdjustContrastd(keys=['image'], prob=0.2, gamma=(0.8, 1.2)),
            NormalizeIntensityd(keys=['image'], nonzero=True, channel_wise=True),
            # 가우시안 노이즈 (p=0.2)
            RandGaussianNoised(
                keys=['image'],
                prob=0.2,
                mean=0.0,
                std=0.05
            ),

            RandCropByPosNegLabeld(
                keys=['image', 'mask'],
                label_key='mask',
                spatial_size=config.patch_size,
                pos=1,
                neg=1,
                num_samples=1,
                allow_smaller=True,
            ),
        ])
    else:
        return Compose([
            LoadImaged(keys=['image', 'mask']),
            EnsureChannelFirstd(keys=['image', 'mask']),
            Orientationd(keys=['image', 'mask'], axcodes='RAS'),
            # Spacingd(keys=['image', 'mask'], pixdim=(1.0, 1.0, 1.0), mode=('bilinear', 'nearest')),
            SpatialPadd(keys=['image', 'mask'], spatial_size=config.patch_size),
            NormalizeIntensityd(keys=['image'], nonzero=True, channel_wise=True),
        ])


def setup_trainer(config):
    """
    Set up the trainer with the configuration
    
    Args:
        config: Configuration object
        
    Returns:
        Trainer, train_loader, val_loader
    """
    # Set random seeds for reproducibility
    if config.seed is not None:
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        np.random.seed(config.seed)
        random.seed(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set up paths
    dataset_name = maybe_convert_to_dataset_name(config.dataset_id)
    preprocessed_dataset_folder = join(nnUNet_preprocessed, dataset_name)
    plans_file = join(preprocessed_dataset_folder, "nnUNetPlans.json")
    
    # Load plans and dataset JSON
    plans = load_json(plans_file)
    dataset_json = load_json(join(preprocessed_dataset_folder, "dataset.json"))
    
    # Initialize trainer
    trainer = nnInteractiveTrainer(
        plans=plans,
        configuration="3d_fullres",
        fold=config.fold,
        dataset_json=dataset_json,
        device=device,
        point_radius=config.point_radius,
        preferred_scribble_thickness=config.scribble_thickness,
        interaction_decay=config.interaction_decay
    )
    
    # Initialize the trainer
    trainer.initialize()
    
    # Load pretrained weights if specified
    if config.pretrained_weights:
        trainer.load_checkpoint(config.pretrained_weights)
    
    # Setup data
    train_transform = get_transforms(config, mode='train')
    val_transform = get_transforms(config, mode='test')
    
    train_dataset = MENRTDataset(
        config.data_dir, 
        patch_size=config.patch_size, 
        transform=train_transform, 
        mode='train'
    )
    
    val_dataset = MENRTDataset(
        config.data_dir, 
        patch_size=config.patch_size, 
        transform=val_transform, 
        mode='test'
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    # Set up TensorBoard writer
    output_folder = trainer.output_folder
    log_dir = join(output_folder, "logs")
    maybe_mkdir_p(log_dir)
    tensorboard_writer = SummaryWriter(log_dir=log_dir)
    trainer.set_tensorboard_writer(tensorboard_writer)
    
    return trainer, train_loader, val_loader, tensorboard_writer


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train nnInteractive model on BraTS-MEN-RT dataset')
    
    # Dataset arguments
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the data directory')
    parser.add_argument('--dataset_id', type=str, required=True, help='Dataset ID for nnUNet')
    parser.add_argument('--fold', type=int, default=0, help='Fold to train on')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=1000, help='Number of epochs to train for')
    parser.add_argument('--patch_size', type=tuple, default=(128, 128, 128), help='Patch size for training')
    parser.add_argument('--spacing', type=tuple, default=(1.0, 1.0, 1.0), help='Spacing for resampling')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--pin_memory', type=bool, default=True, help='Use pinned memory for data loading')
    
    # Interactive segmentation arguments
    parser.add_argument('--point_radius', type=int, default=4, help='Radius for point interactions')
    parser.add_argument('--scribble_thickness', type=int, default=2, help='Thickness for scribble interactions')
    parser.add_argument('--interaction_decay', type=float, default=0.9, help='Decay factor for interactions')
    
    # Model arguments
    parser.add_argument('--pretrained_weights', type=str, default=None, help='Path to pretrained weights')
    parser.add_argument('--continue_training', action='store_true', help='Continue training from last checkpoint')
    parser.add_argument('--only_val', action='store_true', help='Only run validation')
    
    return parser.parse_args()


def main():
    """Main training function"""
    config = parse_args()
    trainer, train_loader, val_loader, tensorboard_writer = setup_trainer(config)
    
    # Print some info
    print(f"Training on {len(train_loader.dataset)} samples")
    print(f"Validating on {len(val_loader.dataset)} samples")
    
    if config.only_val:
        print("Running validation only...")
        trainer.run_validation()
        return
    
    # Training loop
    print(f"Starting training for {config.num_epochs} epochs...")
    trainer.on_train_start()
    trainer.train(config.num_epochs)
    
    # Final validation with best checkpoint
    trainer.load_checkpoint(join(trainer.output_folder, "checkpoint_best.pth"))
    trainer.run_validation()
    
    tensorboard_writer.close()
    print("Training completed!")


if __name__ == "__main__":
    main() 