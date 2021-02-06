from geodesic_distance import _geodesic_distance

import torch
import math
import numpy as np
import functools
import typing

__all__ = ["GeodesicGaussian"]

def expand_dims(
    src:            torch.Tensor,
    dst:            torch.Tensor,
    start_index:    int=1,
) -> torch.Tensor:
    r"""
        Expands the source tensor to match the spatial dimensions of the destination tensor.
        
        Arguments:
            src (torch.Tensor): A tensor of [B, K, X(Y)(Z)] dimensions
            dst (torch.Tensor): A tensor of [B, X(Y)(Z), (D), (H), W] dimensions
            start_index (int, optional): An optional start index denoting the start of the spatial dimensions
        
        Returns:
            A torch.Tensor of [B, K, X(Y)(Z), (1), (1), 1] dimensions. 
    """
    return functools.reduce(
        lambda s, _: s.unsqueeze(-1), 
        [*dst.shape[start_index:]],
        src
    )
expand_spatial_dims = functools.partial(expand_dims, start_index=2)

def dim_list(
    tensor:         torch.Tensor,
    start_index:    int=1,
) -> typing.List[int]:
    return list(range(start_index, len(tensor.shape)))
spatial_dim_list = functools.partial(dim_list, start_index=2)

class GeodesicGaussian(torch.nn.Module):
    __C__ = math.sqrt(math.pi * 2.0)

    def __init__(self,
        std:            float=5.0,
        normalize:      bool=True,
    ):
        super(GeodesicGaussian, self).__init__()
        std = std / 100.0 * np.pi
        self.register_buffer("std", torch.scalar_tensor(std))
        self.normalize = normalize

    def forward(self,
        keypoints:      torch.Tensor, # [B, K, (S)UV or UV(S)] with K the number of keypoints
        spherical_grid: torch.Tensor, # [B, (S)UV or UV(S), (D), H, W]
    ) -> torch.Tensor:                # [B, K, (D), H, W]
        inv_denom = -0.5 * torch.reciprocal(self.std ** 2)
        g = spherical_grid
        centroids = expand_spatial_dims(keypoints, g)
        long1 = g[:, 0, ...].unsqueeze(1)
        lat1 = g[:, 1, ...].unsqueeze(1)
        long2 = centroids[:, :, 0] * np.pi
        lat2 = centroids[:, :, 1] * (0.5 * np.pi)
        dist = _geodesic_distance(long1, lat1, long2, lat2)        
        gaussian = torch.exp(dist * inv_denom)
        if self.normalize: # provide a normalized Gaussian summing to unity
            norm_dims = spatial_dim_list(g)
            gaussian = gaussian / torch.sum(
                gaussian, dim=norm_dims, keepdim=True
            )
        return gaussian

if __name__ == "__main__":
    from spherical_grid import SphericalGrid
    import cv2
    import sys

    std = 9.0 if len(sys.argv) < 2 else float(sys.argv[1])
    width = 512 if len(sys.argv) < 3 else int(sys.argv[2])
    grid = SphericalGrid(width=width)
    gg = GeodesicGaussian(std=std)

    B, K = 5, 4

    keypoints = torch.rand(5, 4, 2) * 2.0 - 1.0
    keypoints[0, 0, 0] = -0.9
    keypoints[0, 1, 0] = 0.99

    gaussian = gg.forward(keypoints, grid.forward(keypoints))
    for b in range(B):
        for k in range(K):
            img = gaussian[b, k, ...]
            img = (img - img.min()) / (img.max() - img.min())
            cv2.imshow(f"{b}_heatmap_{k}", img.numpy())
    cv2.waitKey(-1)