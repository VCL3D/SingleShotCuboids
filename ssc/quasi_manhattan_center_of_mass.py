import torch
import numpy as np

__all__ = ["QuasiManhattanCenterOfMass"]

class QuasiManhattanCenterOfMass(torch.nn.Module):
    def __init__(self,        
        mode:           str='periodic', # one of ['periodic', 'stamdard']
    ):
        super(QuasiManhattanCenterOfMass, self).__init__()
        self._extract_center_of_mass = self._extract_center_of_mass_periodic\
            if mode == 'periodic' else self._extract_center_of_mass_standard

    def _extract_center_of_mass_standard(self,
        grid: torch.Tensor,     # [B, (S)VU, (D), H, W], order is (S)VU not UV(S), y coord first channel
        heatmaps: torch.Tensor  # [B, K, (D), H, W], with its value across the spatial dimensions summing to unity
    ) -> torch.Tensor:          # [B, K, UV(S) or (S)VU]
        channels = heatmaps.shape[1]
        sum_dims = list(range(2, len(heatmaps.shape)))
        top_CoMs, bottom_CoMs = [], []
        for i in range(channels // 2):
            top_heatmap = heatmaps[:, i, ...].unsqueeze(1)
            bottom_heatmap = heatmaps[:, i + channels // 2, ...].unsqueeze(1)
            x_coord = torch.sum(
                top_heatmap * grid[:, 0, ...].unsqueeze(1)
                + bottom_heatmap * grid[:, 0, ...].unsqueeze(1),
                dim=sum_dims
            ) / 2.0
            top_y_coord_t = torch.sum(
                top_heatmap * grid[:, 1, ...].unsqueeze(1), dim=sum_dims
            )
            bottom_y_coord_t = torch.sum(
                bottom_heatmap * grid[:, 1, ...].unsqueeze(1), dim=sum_dims
            )
            top_CoMs.append(torch.cat([x_coord, top_y_coord_t], dim=1))
            bottom_CoMs.append(torch.cat([x_coord, bottom_y_coord_t], dim=1))
        top_corners_t = torch.stack(top_CoMs, dim=1)
        bottom_corners_t = torch.stack(bottom_CoMs, dim=1)
        return torch.cat([top_corners_t, bottom_corners_t], dim=1)

    def _extract_center_of_mass_periodic(self,
        grid: torch.Tensor,     # [B, (S)VU, (D), H, W], order is (S)VU not UV(S), y coord first channel
        heatmaps: torch.Tensor,  # [B, K, (D), H, W], with its value across the spatial dimensions summing to unity
        epsilon: float=1e-12,
    ) -> torch.Tensor:          # [B, K, UV(S) or (S)VU]
        b, c, h, w = heatmaps.size()
        theta_i = grid[:, 1, ...].unsqueeze(1) * (0.5 * np.pi) # in [-pi/2, pi/2]
        phi_i = grid[:, 0, ...].unsqueeze(1) * np.pi + np.pi # in [0, 2pi] for periodic boundary CoM
        chi_i = torch.cos(phi_i)
        zeta_i = torch.sin(phi_i)

        top_coords_t = []
        bottom_coords_t = []
        for i in range(c // 2):
            top_heatmap_t = heatmaps[:, i, ...].unsqueeze(1)
            bottom_heatmap_t = heatmaps[:, i + c // 2, ...].unsqueeze(1)
            
            chi = torch.sum(
                top_heatmap_t * chi_i + bottom_heatmap_t * chi_i,
                dim=[2, 3]
            ) / 2.0
            zeta = torch.sum(
                top_heatmap_t * zeta_i + bottom_heatmap_t * zeta_i,
                dim=[2, 3]
            ) / 2.0
            phi = torch.atan2(-zeta, -chi + epsilon)

            theta_t = torch.sum(top_heatmap_t * theta_i, dim=[2, 3])
            theta_b = torch.sum(bottom_heatmap_t * theta_i, dim=[2, 3])

            top_coords_t.append(torch.cat([phi / np.pi, theta_t / (np.pi * 0.5)], dim=1))
            bottom_coords_t.append(torch.cat([phi / np.pi, theta_b / (np.pi * 0.5)], dim=1))
        
        top_corners_t = torch.stack(top_coords_t, dim=1)
        bottom_corners_t = torch.stack(bottom_coords_t, dim=1)
        return torch.cat([top_corners_t, bottom_corners_t], dim=1)
                
    def forward(self, 
        grid: torch.Tensor,     # coordinates grid tensor of C coordinates
        heatmaps: torch.Tensor, # spatial probability tensor of K keypoints
    ) -> torch.Tensor:
            return self._extract_center_of_mass(grid, heatmaps)

if __name__ == "__main__":
    from geodesic_gaussian import GeodesicGaussian
    from spherical_grid import SphericalGrid, Grid
    import cv2
    import sys

    mode = sys.argv[1]
    
    sg = SphericalGrid(width=512)
    g = Grid(width=512, height=256)
    gg = GeodesicGaussian(std=9.0)
    # scom = QuasiManhattanCenterOfMass(mode='standard')
    # scom = QuasiManhattanCenterOfMass(mode='periodic')
    scom = QuasiManhattanCenterOfMass(mode=mode)

    B, K = 5, 4

    keypoints = torch.rand(5, 4, 2) * 2.0 - 1.0    
    keypoints[0, 0, 0] = -0.9
    keypoints[0, 1, 0] = 0.99
    keypoints[:, 2:, 0] = keypoints[:, :2, 0]

    sgrid = sg.forward(keypoints)
    grid = g.forward(keypoints)
    gaussian = gg.forward(keypoints, sgrid)
    corners = scom.forward(grid, gaussian)

    for b in range(B):
        for k in range(K):
            img = gaussian[b, k, ...]
            img = (img - img.min()) / (img.max() - img.min())
            corner_n_heatmap = img.unsqueeze(-1).repeat(1, 1, 3).numpy()
            corner_n_heatmap = (corner_n_heatmap * 255.0).astype(np.uint8)
            u, v = corners[b, k]
            x = (u + 1.0) / 2.0 * 512
            y = (v + 1.0) / 2.0 * 256
            cv2.circle(corner_n_heatmap, (x,y), 3, (255,0,0), -1)
            cv2.imshow(f"{b}_corner_n_heatmap_{k}", corner_n_heatmap)
    cv2.waitKey(-1)
