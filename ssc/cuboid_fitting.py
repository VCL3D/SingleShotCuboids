import torch
import numpy as np
import functools
import kornia

class CuboidAlignment(torch.nn.Module):
    def __init__(self,
        mode:               str='joint', # one of ['joint', 'floor', 'ceil', 'avg']
        floor_distance:     float=-1.6,
    ):
        super(CuboidAlignment, self).__init__()
        self.homography_func = functools.partial(
            self._homography_floor_svd,
            floor_z=floor_distance)\
            if mode == 'floor' else (
                functools.partial(
                    self._homography_ceil_svd,
                    ceil_z=-floor_distance
                ) if mode == 'ceil' else (functools.partial(
                    self._homography_avg_svd,
                    floor_z=floor_distance,
                    ceil_z=-floor_distance
                    ) if mode == 'avg' else functools.partial(
                        self._homography_joint_svd,
                        floor_z=floor_distance,
                        ceil_z=-floor_distance
                    )
                )
            )
        cuboid_axes = torch.Tensor([[
            [-1, 1],
            [-1, -1],
            [1, -1],
            [1, 1], 
        ]]).float()
        self.register_buffer("cuboid_axes", cuboid_axes)

    def _get_scale_all(self, coords: torch.Tensor, eps: float=1e-12) -> torch.Tensor:
        a_x1 = torch.linalg.norm(coords[:, 0, :] - coords[:, 1, :], ord=2, dim=1)
        a_y1 = torch.linalg.norm(coords[:, 1, :] - coords[:, 2, :], ord=2, dim=1)
        a_x2 = torch.linalg.norm(coords[:, 2, :] - coords[:, 3, :], ord=2, dim=1)
        a_y2 = torch.linalg.norm(coords[:, 3, :] - coords[:, 0, :], ord=2, dim=1)
        a_x = 0.5 * (a_x1 + a_x2)
        a_y = 0.5 * (a_y1 + a_y2)
        return torch.stack([a_y, a_x], dim=1)

    def _svd(self,
        points1: torch.Tensor,
        points2: torch.Tensor
    ) -> torch.Tensor:
        '''
            Computes a similarity transform (sR, t) that takes
            a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
            where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
            i.e. solves the orthogonal Procrutes problem.
        '''
        b, _, c = points1.shape
        # 1. Remove mean.
        points1 = torch.transpose(points1, -2, -1)
        points2 = torch.transpose(points2, -2, -1)
        centroid1 = points1.mean(dim=-1, keepdims=True)
        centroid2 = points1.mean(dim=-1, keepdims=True)
        centered1 = points1 - centroid1
        centered2 = points2 - centroid2
        # 2. Compute variance of X1 used for scale.
        variance = torch.sum(centered1 ** 2, dim=[1, 2])
        # 3. The outer product of X1 and X2.    
        K = centered1 @ torch.transpose(centered2, -2, -1)
        # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are singular vectors of K.
        U, s, V = torch.svd(K)
        # Construct Z that fixes the orientation of R to get det(R)=1.
        Z = torch.eye(c).to(U).unsqueeze(0).repeat(b, 1, 1)
        Z[:,-1, -1] *= torch.sign(torch.det(U @ torch.transpose(V, -2, -1)))
        # Construct R.
        rotation = V @ (Z @ torch.transpose(U, -2, -1))
        # 5. Recover scale.
        scale = torch.cat([torch.trace(x).unsqueeze(0) for x in (rotation @ K)]) / variance
        # 6. Recover translation.
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        translation = centroid2 - (scale * (rotation @ centroid1))
        return rotation, translation, scale

    def _transform_points(self,
        points: torch.Tensor,
        rotation: torch.Tensor,
        translation: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        xformed = scale * (rotation @ torch.transpose(points, -2, -1)) + translation
        return torch.transpose(xformed, -2, -1)

    def _homography_floor_svd(self,
        top_corners: torch.Tensor, # in [-1, 1]
        bottom_corners: torch.Tensor, # in [-1, 1]
        floor_z: float=-1.6,
    ):
        b, N, _ = top_corners.size()
        u = bottom_corners[:, :, 0] * np.pi
        v = bottom_corners[:, :, 1] * (-0.5 * np.pi)
        c = floor_z / torch.tan(v)
        x = c * torch.sin(u)
        y = -c * torch.cos(u)
        floor_xy = torch.stack([x, y], dim=-1)
        scale = self._get_scale_all(floor_xy)
        scale = scale / 2.0
        centroid = floor_xy.mean(dim=1)
        c = torch.linalg.norm(floor_xy, ord=2, dim=-1)
        v = top_corners[:, :, 1] * (-0.5 * np.pi)
        ceil_z = (c * torch.tan(v)).mean(dim=1, keepdim=True)
        ceil_z = ceil_z.unsqueeze(1).expand(b, 4, 1).contiguous()
        floor_xy = floor_xy - centroid.unsqueeze(1)
        inds = torch.sort(torch.atan2(floor_xy[..., 0], floor_xy[..., 1] + 1e-12))[1]
        axes = self.cuboid_axes[:, inds.squeeze(), :]
        homography = kornia.get_perspective_transform(floor_xy, axes)    
        homogeneous = torch.cat([floor_xy, torch.ones_like(floor_xy[..., -1:])], dim=2)
        xformed = (homography @ homogeneous.transpose(1, 2)).transpose(1, 2)
        xformed = xformed[:, :, :2] / xformed[:, :, 2].unsqueeze(-1)
        rect_floor_xy = xformed * scale.unsqueeze(1) + centroid.unsqueeze(1)
        original_xy = floor_xy + centroid.unsqueeze(1)
        R, t, s = self._svd(rect_floor_xy, original_xy[:, inds.squeeze(), :])
        rect_floor_xy = self._transform_points(rect_floor_xy, R, t, s)
        bottom_points = torch.cat([rect_floor_xy, floor_z * torch.ones_like(c.unsqueeze(-1))], dim=-1)
        top_points = torch.cat([rect_floor_xy, ceil_z], dim=-1)
        return top_points, bottom_points

    def _homography_joint_svd(self,
        top_corners: torch.Tensor, # in [-1, 1]
        bottom_corners: torch.Tensor, # in [-1, 1]
        floor_z: float=-1.6,
        ceil_z: float=1.6,
    ):
        b, N, _ = top_corners.size()
        floor_u = bottom_corners[:, :, 0] * np.pi
        floor_v = bottom_corners[:, :, 1] * (-0.5 * np.pi)
        floor_c = floor_z / torch.tan(floor_v)
        floor_x = floor_c * torch.sin(floor_u)
        floor_y = -floor_c * torch.cos(floor_u)
        floor_xy = torch.stack([floor_x, floor_y], dim=-1)
        floor_scale = self._get_scale_all(floor_xy)
        floor_scale = floor_scale / 2.0        
        floor_ceil_c = torch.linalg.norm(floor_xy, ord=2, dim=-1)
        floor_ceil_v = top_corners[:, :, 1] * (-0.5 * np.pi)
        floor_ceil_z = (floor_ceil_c * torch.tan(floor_ceil_v)).mean(dim=1, keepdim=True)
        floor_ceil_z = floor_ceil_z.unsqueeze(1).expand(b, 4, 1).contiguous()
        ceil_u_t = top_corners[:, :, 0] * np.pi
        ceil_v_t = top_corners[:, :, 1] * (-0.5 * np.pi)
        ceil_c = ceil_z / torch.tan(ceil_v_t)
        ceil_x = ceil_c * torch.sin(ceil_u_t)
        ceil_y = -ceil_c * torch.cos(ceil_u_t)
        ceil_xy = torch.stack([ceil_x, ceil_y], dim=-1)
        ceil_floor_c = torch.linalg.norm(ceil_xy, ord=2, dim=-1)
        ceil_v_b = bottom_corners[:, :, 1] * (-0.5 * np.pi)
        ceil_floor_z = (ceil_floor_c * torch.tan(ceil_v_b)).mean(dim=1, keepdim=True)
        fix_ceil = -ceil_z / ceil_floor_z
        ceil_z_fix = ceil_z * fix_ceil
        ceil_z_fix = ceil_z_fix.unsqueeze(1).expand(b, 4, 1).contiguous()
        ceil_floor_fixed_c = ceil_z_fix.squeeze(-1) / torch.tan(ceil_v_t)
        ceil_x = ceil_floor_fixed_c * torch.sin(ceil_u_t)
        ceil_y = -ceil_floor_fixed_c * torch.cos(ceil_u_t)
        ceil_xy = torch.stack([ceil_x, ceil_y], dim=-1)
        ceil_scale = self._get_scale_all(ceil_xy)
        ceil_scale = ceil_scale / 2.0
        joint_xy = 0.5 * (floor_xy + ceil_xy)
        joint_scale = 0.5 * (floor_scale + ceil_scale)        
        joint_centroid = joint_xy.mean(dim=1)
        joint_xy = joint_xy - joint_centroid.unsqueeze(1)
        inds = torch.sort(torch.atan2(joint_xy[..., 0], joint_xy[..., 1] + 1e-12))[1]
        axes = self.cuboid_axes[:, inds.squeeze(), :]
        homography = kornia.get_perspective_transform(joint_xy, axes)    
        homogeneous = torch.cat([joint_xy, torch.ones_like(joint_xy[..., -1:])], dim=2)
        xformed = (homography @ homogeneous.transpose(1, 2)).transpose(1, 2)
        xformed = xformed[:, :, :2] / xformed[:, :, 2].unsqueeze(-1)
        rect_joint_xy = xformed * joint_scale.unsqueeze(1) + joint_centroid.unsqueeze(1)
        original_xy = joint_xy + joint_centroid.unsqueeze(1)
        R, t, s = self._svd(rect_joint_xy, original_xy[:, inds.squeeze(), :])
        rect_joint_xy = self._transform_points(rect_joint_xy, R, t, s)
        bottom_points = torch.cat([rect_joint_xy, floor_z * torch.ones_like(floor_c.unsqueeze(-1))], dim=-1)
        top_points = torch.cat([rect_joint_xy, ceil_z_fix], dim=-1)
        return top_points, bottom_points

    def _homography_ceil_svd(self,
        top_corners: torch.Tensor, # in [-1, 1]
        bottom_corners: torch.Tensor, # in [-1, 1]
        ceil_z: float=1.6,
    ):
        b, N, _ = top_corners.size()
        u_t = top_corners[:, :, 0] * np.pi
        v_t = top_corners[:, :, 1] * (-0.5 * np.pi)
        c = ceil_z / torch.tan(v_t)
        x = c * torch.sin(u_t)
        y = -c * torch.cos(u_t)
        ceil_xy = torch.stack([x, y], dim=-1)
        c = torch.linalg.norm(ceil_xy, ord=2, dim=-1)
        v_b = bottom_corners[:, :, 1] * (-0.5 * np.pi)
        floor_z = (c * torch.tan(v_b)).mean(dim=1, keepdim=True)
        fix_ceil = -ceil_z / floor_z
        floor_z = -ceil_z
        ceil_z_fix = ceil_z * fix_ceil
        ceil_z_fix = ceil_z_fix.unsqueeze(1).expand(b, 4, 1).contiguous()
        c = ceil_z_fix.squeeze(-1) / torch.tan(v_t)
        x = c * torch.sin(u_t)
        y = -c * torch.cos(u_t)
        ceil_xy = torch.stack([x, y], dim=-1)
        scale = self._get_scale_all(ceil_xy)
        scale = scale / 2.0
        centroid = ceil_xy.mean(dim=1)
        ceil_xy = ceil_xy - centroid.unsqueeze(1)
        inds = torch.sort(torch.atan2(ceil_xy[..., 0], ceil_xy[..., 1] + 1e-12))[1]
        axes = self.cuboid_axes[:, inds.squeeze(), :]
        homography = kornia.get_perspective_transform(ceil_xy, axes)
        homogeneous = torch.cat([ceil_xy, torch.ones_like(ceil_xy[..., -1:])], dim=2)
        xformed = (homography @ homogeneous.transpose(1, 2)).transpose(1, 2)
        xformed = xformed[:, :, :2] / xformed[:, :, 2].unsqueeze(-1)
        rect_ceil_xy = xformed * scale.unsqueeze(1) + centroid.unsqueeze(1)
        original_xy = ceil_xy + centroid.unsqueeze(1)
        R, t, s = self._svd(rect_ceil_xy, original_xy[:, inds.squeeze(), :])
        rect_ceil_xy = self._transform_points(rect_ceil_xy, R, t, s)
        bottom_points = torch.cat([rect_ceil_xy, floor_z * torch.ones_like(c.unsqueeze(-1))], dim=-1)
        top_points = torch.cat([rect_ceil_xy, ceil_z_fix], dim=-1)
        return top_points, bottom_points

    def _homography_avg_svd(self,
        top_corners:        torch.Tensor, # in [-1, 1]
        bottom_corners:     torch.Tensor, # in [-1, 1]
        floor_z:            float=-1.6,
        ceil_z:             float=1.6,
    ):
        top_ceil, bottom_ceil = self._homography_ceil_svd(top_corners, bottom_corners, ceil_z)
        top_floor, bottom_floor = self._homography_floor_svd(top_corners, bottom_corners, floor_z)
        return (top_ceil + top_floor) * 0.5, (bottom_ceil + bottom_floor) * 0.5

    def _project_points(self,
        points3d:           torch.Tensor,
        epsilon:            float=1e-12,
    ):
        phi = torch.atan2(points3d[:, :, 0], -1.0 * points3d[:, :, 1] + epsilon) # [-pi, pi]
        xy_dist = torch.linalg.norm(points3d[:, :, :2], ord=2, dim=-1)
        theta = -1.0 * torch.atan2(points3d[:, :, 2], xy_dist + epsilon)  # [-pi / 2.0, pi / 2.0]
        u = phi / np.pi
        v = theta / (0.5 * np.pi)
        return torch.stack([u, v], dim=-1)

    def forward(self, corners: torch.Tensor) -> torch.Tensor:
        top, bottom = torch.chunk(corners, 2, dim=1)
        b = top.shape[0]
        aligned = []
        for i in range(b):
            t = top[i, ...].unsqueeze(0)
            b = bottom[i, ...].unsqueeze(0)
            try:
                t_xyz, b_xyz = self.homography_func(t, b)
                t_uv, b_uv = self._project_points(t_xyz), self._project_points(b_xyz)
                t_uv = t_uv[:, torch.argsort(t_uv[0, :, 0]), :]
                b_uv = b_uv[:, torch.argsort(b_uv[0, :, 0]), :]
                aligned_corners = torch.cat([t_uv, b_uv], dim=1).squeeze(0)
                aligned.append(aligned_corners)
            except RuntimeError as ex:
                aligned.append(corners[i, ...])
        return torch.stack(aligned, dim=0)


if __name__ == "__main__":
    from cuboid_test_utils import *
    from cuboid_tests import *

    modes = ['floor', 'ceil', 'joint', 'avg']
    for name, test in get_tests():
        if '15' not in name:
            continue
        for mode in modes:
            if 'floor' not in mode:
                continue
            alignment = CuboidAlignment(mode=mode)
            
            top, bottom = test()
            top = top.cuda()
            bottom = bottom.cuda()
            alignment = alignment.cuda()

            corners = torch.cat([top, bottom], dim=1)
            aligned = alignment.forward(corners)
            images = np.zeros([1, 256, 512, 3], dtype=np.uint8)
            top_pts2d, bottom_pts2d = torch.chunk(aligned, 2, dim=-2)
            draw_points(top_pts2d, images, [255, 0, 0])
            draw_points(bottom_pts2d, images, [255, 0, 0])

            top_pts2d, bottom_pts2d = torch.chunk(corners, 2, dim=-2)
            draw_points(top_pts2d, images, [0, 255, 0])
            draw_points(bottom_pts2d, images, [0, 255, 0])
            show_frozen(f"{mode} {name}", images[0])
            # show_playback(f"{mode} {name}", images[0])
    print("finished")