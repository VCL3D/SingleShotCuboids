import os
import cv2
import numpy as np
import torch
import open3d

colors = [
    [255, 0, 0],    # blue - floor
    [0, 255, 0],    # green - ceil
    [0, 0, 255],    # red - proj floor
    [255, 0, 255],  # purple - proj ceil
    [255, 255, 0],  # cyan - proj
]

def draw_points(
    points2d: torch.Tensor, # [B, N, 2]
    images: np.array, # [B, H, W, 3]
    color: np.array
):
    b, h, w, _ = images.shape
    N = points2d.shape[1]
    for i in range(b):
        img = images[i]
        pts2d = points2d[i]        
        for k in range(N):
            pt2d = pts2d[k]
            x = int((pt2d[0] * np.pi + np.pi) / (2.0 * np.pi) * w)
            y = int((pt2d[1] * np.pi * 0.5 + np.pi * 0.5) / np.pi * h)
            cv2.drawMarker(img, (x, y), color, cv2.MARKER_SQUARE, 15, 2)

def draw_points_coords(
    points2d: torch.Tensor, # [B, N, 2]
    images: np.array, # [B, H, W, 3]
    color: np.array
):
    b = images.shape[0]
    N = points2d.shape[1]
    for i in range(b):
        img = images[i]
        pts2d = points2d[i]        
        for k in range(N):
            pt2d = pts2d[k]
            x = int(pt2d[0])
            y = int(pt2d[1])
            cv2.drawMarker(img, (x, y),color, cv2.MARKER_SQUARE, 15, 2)

def show_frozen(name: str, img: np.array):
    cv2.imshow(name, img)
    cv2.waitKey(-1)

def show_playback(name: str, img: np.array):
    cv2.imshow(name, img)
    cv2.waitKey(33)

def write_points(
    top_pts3d: torch.Tensor,
    bottom_pts3d: torch.Tensor,
    path: str,
    key: str,
    color: int,
):
    b, N, _ = top_pts3d.size()
    for i in range(b):
        points = torch.cat([top_pts3d[i, ...], bottom_pts3d[i, ...]], dim=0)            
        wf_lines = [[i, (i+1)%N] for i in range(N)] +\
                [[i+N, (i+1)%N+N] for i in range(N)] +\
                [[i, i+N] for i in range(N)]
        wf_colors = [[c / 255.0 for c in colors[color]] for i in range(len(wf_lines))]
        wf_line_set = open3d.geometry.LineSet()
        wf_line_set.points = open3d.utility.Vector3dVector(points.detach().numpy())
        wf_line_set.lines = open3d.utility.Vector2iVector(wf_lines)
        wf_line_set.colors = open3d.utility.Vector3dVector(wf_colors)
        open3d.io.write_line_set(os.path.join(
                path, f"test_{key}.ply"
            ),
            wf_line_set
        )