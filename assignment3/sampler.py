import math
from typing import List

import torch
from ray_utils import RayBundle
from pytorch3d.renderer.cameras import CamerasBase


# Sampler which implements stratified (uniform) point sampling along rays
class StratifiedRaysampler(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.n_pts_per_ray = cfg.n_pts_per_ray
        self.min_depth = cfg.min_depth
        self.max_depth = cfg.max_depth

    def forward(
        self,
        ray_bundle,
    ):
        # TODO (Q1.4): Compute z values for self.n_pts_per_ray points uniformly sampled between [near, far]
        # a.k.a Computing the depth values
        z_vals = torch.linspace(self.min_depth, self.max_depth, self.n_pts_per_ray, device = "cuda")


        # TODO (Q1.4): Sample points from z values
        # Create Sample Points
            # unsqueeze(1) adds a new dimension to prepare for repetition
            # repeat duplicates the data along the sampling dimension
        origins = ray_bundle.origins.unsqueeze(1).repeat(1, z_vals.shape[0], 1)
        directions = ray_bundle.directions.unsqueeze(1).repeat(1, z_vals.shape[0], 1)

        # Compute Sample Positions
            # scales the direction vectors by each sampled depth value (z_vals)
        z_vals = z_vals.unsqueeze(0).unsqueeze(-1).repeat(ray_bundle.origins.shape[0], 1, 1)
        sample_points = origins + directions * z_vals

        # Return
        return ray_bundle._replace(
            sample_points=sample_points,
            sample_lengths=z_vals * torch.ones_like(sample_points[..., :1]),
        )


sampler_dict = {
    'stratified': StratifiedRaysampler
}