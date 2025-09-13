#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
import torch
from torch.utils.tensorboard import SummaryWriter

from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, load_json
from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed, nnUNet_results
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class

from nnInteractive.trainer.nnInteractiveTrainer import nnInteractiveTrainer


def run_training(dataset_name_or_id: str,
                 configuration: str = "3d_fullres",
                 fold: int = 0,
                 trainer_class_name: str = "nnInteractiveTrainer",
                 plans_identifier: str = "nnUNetPlans",
                 pretrained_weights: str = None,
                 num_epochs: int = 1000,
                 continue_training: bool = False,
                 only_run_validation: bool = False,
                 disable_checkpointing: bool = False,
                 device: str = "cuda",
                 unpack_dataset: bool = True,
                 point_radius: int = 4,
                 scribble_thickness: int = 2,
                 interaction_decay: float = 0.9,
                 deterministic: bool = False,
                 npz: bool = False,
                 find_lr: bool = False,
                 val_with_best: bool = False,
                 ):
    """
    Run training for interactive segmentation
    
    Args:
        dataset_name_or_id: Dataset name or ID to train on
        configuration: nnUNet configuration to use
        fold: Fold to train on
        trainer_class_name: Name of the trainer class to use
        plans_identifier: Name of the plans file
        pretrained_weights: Path to pretrained weights
        num_epochs: Number of epochs to train for
        continue_training: Whether to continue training from the last checkpoint
        only_run_validation: Only run validation, no training
        disable_checkpointing: Disable saving checkpoints
        device: Device to train on
        unpack_dataset: Whether to unpack the dataset
        point_radius: Radius for point interactions
        scribble_thickness: Thickness for scribble interactions
        interaction_decay: Decay factor for past interactions
        deterministic: Use deterministic training
        npz: Save preprocessed data as npz instead of pkl
        find_lr: Run learning rate finder
        val_with_best: Run validation with the best checkpoint
    """
    
    # Set deterministic training if requested
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Convert dataset name if needed
    dataset_name = maybe_convert_to_dataset_name(dataset_name_or_id)
    
    # Set up paths
    preprocessed_dataset_folder = join(nnUNet_preprocessed, dataset_name)
    plans_file = join(preprocessed_dataset_folder, plans_identifier + ".json")
    output_folder = join(nnUNet_results, dataset_name, f"{trainer_class_name}__{plans_identifier}")
    
    if fold is not None:
        output_folder = join(output_folder, f"fold_{fold}")
        
    maybe_mkdir_p(output_folder)
    
    # Find the trainer class
    trainer_class = recursive_find_python_class(
        join(os.path.dirname(os.path.dirname(__file__)), "trainer"),
        trainer_class_name,
        "nnInteractive.trainer"
    )
    
    if trainer_class is None:
        raise RuntimeError(f"Could not find trainer class {trainer_class_name} in nnInteractive.trainer")
        
    # Get plans and dataset JSON
    plans = load_json(plans_file)
    dataset_json = load_json(join(preprocessed_dataset_folder, "dataset.json"))
    
    # Initialize the trainer
    trainer = trainer_class(
        plans=plans,
        configuration=configuration,
        fold=fold,
        dataset_json=dataset_json,
        unpack_dataset=unpack_dataset,
        device=torch.device(device),
        point_radius=point_radius,
        preferred_scribble_thickness=scribble_thickness,
        interaction_decay=interaction_decay
    )
    
    # Set some trainer properties
    if continue_training:
        trainer.load_checkpoint()
    elif pretrained_weights is not None:
        trainer.load_checkpoint(pretrained_weights)
    
    trainer.set_deep_supervision_enabled(True)
    
    # Set TensorBoard writer
    log_dir = join(output_folder, "logs")
    maybe_mkdir_p(log_dir)
    tensorboard_writer = SummaryWriter(log_dir=log_dir)
    trainer.set_tensorboard_writer(tensorboard_writer)
    
    # Run learning rate finder if requested
    if find_lr:
        trainer.find_lr()
        return
    
    if only_run_validation:
        trainer.run_validation()
        return
    
    # Run training
    if not disable_checkpointing:
        trainer.on_train_start()
        trainer.train(num_epochs)
    else:
        trainer.on_train_start()
        trainer.train(num_epochs, save_checkpoints=False)
        
    # Run validation with the best checkpoint if requested
    if val_with_best:
        trainer.load_checkpoint(join(output_folder, "checkpoint_best.pth"))
        trainer.run_validation()
        
    tensorboard_writer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name_or_id", type=str, help="Dataset name or ID")
    parser.add_argument("-c", "--configuration", type=str, default="3d_fullres", help="Configuration")
    parser.add_argument("-f", "--fold", type=int, default=0, help="Fold")
    parser.add_argument("-tr", "--trainer_class_name", type=str, default="nnInteractiveTrainer", 
                       help="Trainer class name")
    parser.add_argument("-p", "--plans_identifier", type=str, default="nnUNetPlans", 
                       help="Plans identifier")
    parser.add_argument("-pretrained_weights", type=str, default=None, 
                       help="Path to pretrained weights")
    parser.add_argument("-num_epochs", type=int, default=1000, help="Number of epochs")
    parser.add_argument("-continue_training", action="store_true", 
                       help="Continue training from checkpoint")
    parser.add_argument("-val", "--only_run_validation", action="store_true", 
                       help="Only run validation")
    parser.add_argument("-disable_checkpointing", action="store_true", 
                       help="Disable checkpointing")
    parser.add_argument("-device", type=str, default="cuda", help="Device to use")
    parser.add_argument("-no_unpack", action="store_false", dest="unpack_dataset", 
                       help="Do not unpack dataset")
    parser.add_argument("-point_radius", type=int, default=4, 
                       help="Radius for point interactions")
    parser.add_argument("-scribble_thickness", type=int, default=2, 
                       help="Thickness for scribble interactions")
    parser.add_argument("-interaction_decay", type=float, default=0.9, 
                       help="Decay factor for past interactions")
    parser.add_argument("-deterministic", action="store_true", 
                       help="Use deterministic training")
    parser.add_argument("-npz", action="store_true", help="Save data as npz")
    parser.add_argument("-find_lr", action="store_true", help="Run learning rate finder")
    parser.add_argument("-val_with_best", action="store_true", 
                       help="Run validation with best checkpoint")
    
    args = parser.parse_args()
    
    run_training(
        dataset_name_or_id=args.dataset_name_or_id,
        configuration=args.configuration,
        fold=args.fold,
        trainer_class_name=args.trainer_class_name,
        plans_identifier=args.plans_identifier,
        pretrained_weights=args.pretrained_weights,
        num_epochs=args.num_epochs,
        continue_training=args.continue_training,
        only_run_validation=args.only_run_validation,
        disable_checkpointing=args.disable_checkpointing,
        device=args.device,
        unpack_dataset=args.unpack_dataset,
        point_radius=args.point_radius,
        scribble_thickness=args.scribble_thickness,
        interaction_decay=args.interaction_decay,
        deterministic=args.deterministic,
        npz=args.npz,
        find_lr=args.find_lr,
        val_with_best=args.val_with_best
    )


if __name__ == "__main__":
    main() 