#!/usr/bin/env python3
import os
import argparse
import numpy as np
import torch
from pathlib import Path

from batchgenerators.utilities.file_and_folder_operations import join, isdir, maybe_mkdir_p

from nnInteractive.inference.inference_session import nnInteractiveInferenceSession


def run_inference(
    model_dir: str,
    input_image: np.ndarray,
    device: str = "cuda",
    fold: int = None,
    checkpoint_name: str = "checkpoint_final.pth",
    use_torch_compile: bool = False,
    do_autozoom: bool = True,
    verbose: bool = False,
    output_file: str = None
):
    """
    Run inference with the nnInteractiveInferenceSession
    
    Args:
        model_dir: Path to the trained model directory
        input_image: 4D image data [c, x, y, z]
        device: Device to run inference on
        fold: Fold to use for inference
        checkpoint_name: Name of the checkpoint file to use
        use_torch_compile: Whether to use torch.compile
        do_autozoom: Whether to use auto-zooming
        verbose: Whether to print verbose output
        output_file: Path to save the output segmentation
        
    Returns:
        The inference session object that can be used for interactive segmentation
    """
    # Create inference session
    inference_session = nnInteractiveInferenceSession(
        device=torch.device(device),
        use_torch_compile=use_torch_compile,
        verbose=verbose,
        do_autozoom=do_autozoom
    )
    
    # Initialize from model
    inference_session.initialize_from_trained_model_folder(
        model_dir,
        use_fold=fold,
        checkpoint_name=checkpoint_name
    )
    
    # Set the input image
    inference_session.set_image(input_image)
    
    # Initialize target buffer to store the result
    target_shape = input_image.shape[1:]
    target_buffer = np.zeros(target_shape, dtype=np.uint8)
    inference_session.set_target_buffer(target_buffer)
    
    print(f"Initialized inference session from {model_dir}")
    print(f"Input image shape: {input_image.shape}")
    
    # Return the initialized session for interactive use
    return inference_session


def demonstrate_interaction(inference_session, input_image):
    """
    Demonstrate the different types of interactions
    
    Args:
        inference_session: The initialized inference session
        input_image: The input image
    """
    shape = input_image.shape[1:]
    
    # Create a basic bounding box around the center of the image
    center = [s // 2 for s in shape]
    box_size = [s // 4 for s in shape]
    bbox_coords = [[c - s, c + s] for c, s in zip(center, box_size)]
    
    print("Adding bounding box interaction...")
    inference_session.add_bbox_interaction(bbox_coords, include_interaction=True)
    
    # Add a point interaction near the center
    point_coords = [c + np.random.randint(-5, 5) for c in center]
    
    print("Adding point interaction...")
    inference_session.add_point_interaction(point_coords, include_interaction=True)
    
    # Create a simple scribble
    scribble_image = np.zeros(shape, dtype=np.float32)
    scribble_start = [c - s // 2 for c, s in zip(center, box_size)]
    scribble_end = [c + s // 2 for c, s in zip(center, box_size)]
    
    # Create a line along one dimension
    for i in range(scribble_start[0], scribble_end[0]):
        coords = [i, center[1], center[2]]
        scribble_image[coords[0], coords[1], coords[2]] = 1
    
    print("Adding scribble interaction...")
    inference_session.add_scribble_interaction(scribble_image, include_interaction=True)
    
    print("Inference with interactions complete!")
    
    return inference_session.target_buffer


def main():
    parser = argparse.ArgumentParser(description="Run interactive inference with trained model")
    parser.add_argument("model_dir", type=str, help="Path to trained model directory")
    parser.add_argument("--input_file", type=str, help="Path to input image file (numpy array)")
    parser.add_argument("--fold", type=int, default=None, help="Fold to use")
    parser.add_argument("--checkpoint", type=str, default="checkpoint_final.pth", 
                       help="Checkpoint name")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--no_autozoom", action="store_false", dest="do_autozoom", 
                       help="Disable auto-zooming")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--output_file", type=str, help="Path to save output segmentation")
    parser.add_argument("--demo", action="store_true", help="Run a demo with sample interactions")
    
    args = parser.parse_args()
    
    # Check if model directory exists
    if not isdir(args.model_dir):
        raise ValueError(f"Model directory {args.model_dir} does not exist")
    
    # Create dummy input data if no input file is provided
    if args.input_file is None:
        print("No input file provided, creating dummy data...")
        # Create a 4D dummy image with a sphere in the center
        shape = (1, 128, 128, 128)
        input_image = np.zeros(shape, dtype=np.float32)
        
        # Create a sphere
        center = np.array([64, 64, 64])
        radius = 20
        
        x, y, z = np.ogrid[:128, :128, :128]
        dist = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
        input_image[0, dist <= radius] = 1.0
    else:
        # Load input image
        input_image = np.load(args.input_file)
        
        # Ensure 4D input
        if input_image.ndim == 3:
            input_image = input_image[np.newaxis]
        assert input_image.ndim == 4, f"Input image must be 4D, got shape {input_image.shape}"
    
    # Run inference
    inference_session = run_inference(
        model_dir=args.model_dir,
        input_image=input_image,
        device=args.device,
        fold=args.fold,
        checkpoint_name=args.checkpoint,
        do_autozoom=args.do_autozoom,
        verbose=args.verbose,
        output_file=args.output_file
    )
    
    # Run demo if requested
    if args.demo:
        segmentation = demonstrate_interaction(inference_session, input_image)
        
        # Save output if requested
        if args.output_file:
            output_dir = os.path.dirname(args.output_file)
            if output_dir:
                maybe_mkdir_p(output_dir)
            np.save(args.output_file, segmentation)
            print(f"Saved segmentation to {args.output_file}")
    else:
        print("Inference session initialized and ready for interaction")
        print("You can now add interactions using the inference_session object")
        
    return inference_session


if __name__ == "__main__":
    inference_session = main() 