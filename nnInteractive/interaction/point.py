from functools import lru_cache
from typing import Tuple, Optional

import numpy as np
import torch
from batchgeneratorsv2.helpers.scalar_type import sample_scalar, RandomScalar
from scipy.ndimage import distance_transform_edt
from skimage.morphology import disk, ball


@lru_cache(maxsize=5)
def build_point(radii, use_distance_transform, binarize):
    max_radius = max(radii)
    ndim = len(radii)

    # Create a spherical (or circular) structuring element with max_radius
    if ndim == 2:
        structuring_element = disk(max_radius)
    elif ndim == 3:
        structuring_element = ball(max_radius)
    else:
        raise ValueError("Unsupported number of dimensions. Only 2D and 3D are supported.")

    # Convert the structuring element to a tensor
    structuring_element = torch.from_numpy(structuring_element.astype(np.float32))

    # Create the target shape based on the sampled radii
    target_shape = [round(2 * r + 1) for r in radii]

    if any([i != j for i, j in zip(target_shape, structuring_element.shape)]):
        structuring_element_resized = torch.nn.functional.interpolate(
            structuring_element.unsqueeze(0).unsqueeze(0),  # Add batch and channel dimensions for interpolation
            size=target_shape,
            mode='trilinear' if ndim == 3 else 'bilinear',
            align_corners=False
        )[0, 0]  # Remove batch and channel dimensions after interpolation
    else:
        structuring_element_resized = structuring_element

    if use_distance_transform:
        # Convert the structuring element to a binary mask for distance transform computation
        binary_structuring_element = (structuring_element_resized >= 0.5).numpy()

        # Compute the Euclidean distance transform of the binary structuring element
        structuring_element_resized = distance_transform_edt(binary_structuring_element)

        # Normalize the distance transform to have values between 0 and 1
        structuring_element_resized /= structuring_element_resized.max()
        structuring_element_resized = torch.from_numpy(structuring_element_resized)

    if binarize and not use_distance_transform:
        # Normalize the resized structuring element to binary (values near 1 are treated as the point region)
        structuring_element_resized = (structuring_element_resized >= 0.5).float()
    return structuring_element_resized


class PointInteraction_stub():
    interaction_type = 'point'

    def __init__(self,
                 point_radius: RandomScalar,
                 use_distance_transform: bool = False):
        """
        Initializes the PointInteraction object.

        Parameters:
        point_radius (RandomScalar): Specifies the radius for the interaction points.
        use_distance_transform (bool): Determines whether to use a distance transform for smooth interactions.
        """
        super().__init__()
        self.point_radius = point_radius
        self.use_distance_transform = use_distance_transform

    def place_point(self,
                    position: Tuple[int, ...],
                    interaction_map: torch.Tensor,
                    binarize: bool = False) -> torch.Tensor:
        """
        Places a point on the interaction map around the specified position.

        Parameters:
        position (Tuple[int, ...]): The (x, y, z) coordinates where the point should be placed.
        interaction_map (torch.Tensor): A tensor representing the interaction map where the point
                                        should be placed. The shape should match the volume dimensions.
        binarize (bool): If True, inserts a binary mask. If False, may insert smooth values based on distance.

        Returns:
        torch.Tensor: Updated interaction map with the point added.
        """
        ndim = interaction_map.ndim

        # Determine the radius for each dimension
        radius = tuple([sample_scalar(self.point_radius, d, interaction_map.shape) for d in range(ndim)])

        strel = build_point(radius, self.use_distance_transform, binarize)

        # Calculate slice range in each dimension, ensuring it is within the bounds of the interaction map
        bbox = [[position[i] - strel.shape[i] // 2, position[i] + strel.shape[i] // 2 + strel.shape[i] % 2] for i in range(ndim)]
        # detect if bbox is completely outside interaction_map
        if any([i[1] < 0 for i in bbox]) or any([i[0] > s for i, s in zip(bbox, interaction_map.shape)]):
            print('Point is outside the interaction map! Ignoring')
            print(f'Position: {position}')
            print(f'Interaction map shape: {interaction_map.shape}')
            print(f'Point bbox would have been {bbox}')
            return interaction_map
        slices = tuple(slice(max(0, bbox[i][0]), min(interaction_map.shape[i], bbox[i][1])) for i in range(ndim))

        # Calculate where the resized structuring element should be placed within the slices
        structuring_slices = tuple([slice(max(0, -bbox[i][0]), slices[i].stop - slices[i].start + max(0, -bbox[i][0])) for i in range(ndim)])

        # Place the resized structuring element into the interaction map
        torch.maximum(interaction_map[slices], strel[structuring_slices].to(interaction_map.device), out=interaction_map[slices])
        return interaction_map

