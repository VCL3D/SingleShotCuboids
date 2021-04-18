import open3d
import numpy as np
import cv2
import typing
from panda3d.core import Triangulator

#NOTE: code borrowed from https://github.com/bertjiazheng/Structured3D

class MeshHandler(object):
    def __init__(self):
        pass

    def _coor2xy(self, coor, z=50, w=1024, h=512):
        '''
        coor: N x 2, index of array in (col, row) format
        '''
        coor = np.array(coor)
        u = self._coorx2u(coor[:, 0], w)
        v = self._coory2v(coor[:, 1], h)
        c = z / np.tan(v)
        x = c * np.sin(u)
        y = -c * np.cos(u)
        return np.hstack([x[:, None], y[:, None]])

    def _coorx2u(self, x: np.array, w=1024):
        return ((x + 0.5) / w - 0.5) * 2 * np.pi

    def _coory2v(self, coory, coorH=512):
        return -((coory + 0.5) / coorH - 0.5) * np.pi

    def _uv2xyz(self,
        floor: np.array,
        ceil: np.array,
        z_dist: float,
        w: int,
        h: int
    ):
        N = len(floor)
        floor_xyz = np.hstack([
            self._coor2xy(floor, z_dist, w, h),
            np.zeros((N, 1)) + z_dist,
        ])
        c = np.sqrt((floor_xyz[:, :2] ** 2).sum(1))
        v = self._coory2v(ceil[:, 1], h)
        ceil_z = c * np.tan(v)
        ceil_xyz = floor_xyz.copy()
        ceil_xyz[:, 2] = ceil_z
        return floor_xyz, ceil_xyz
    
    def _xyz2coorxy(self, xs, ys, zs, H=512, W=1024):
        us = np.arctan2(xs, ys)
        vs = -np.arctan(zs / np.sqrt(xs**2 + ys**2))
        coorx = (us / (2 * np.pi) + 0.5) * W
        coory = (vs / np.pi + 0.5) * H
        return coorx, coory
        
    def _equirectangular_to_perspective(self,
        image, corner_i, corner_j, 
        wall_height, camera, input_resolution=1024,
        output_resolution=512, is_wall=True
    ):
        corner_i = corner_i - camera
        corner_j = corner_j - camera
        if is_wall:
            xs = np.linspace(corner_i[0], corner_j[0], output_resolution)[None].repeat(output_resolution, 0)
            ys = np.linspace(corner_i[1], corner_j[1], output_resolution)[None].repeat(output_resolution, 0)
            zs = np.linspace(-camera[-1], wall_height - camera[-1], output_resolution)[:, None].repeat(output_resolution, 1)
        else:
            xs = np.linspace(corner_i[0], corner_j[0], output_resolution)[None].repeat(output_resolution, 0)
            ys = np.linspace(corner_i[1], corner_j[1], output_resolution)[:, None].repeat(output_resolution, 1)
            zs = np.zeros_like(xs) + wall_height - camera[-1]
        w, h = input_resolution, input_resolution // 2
        coorx, coory = self._xyz2coorxy(xs, ys, zs, H=h, W=w)
        return cv2.remap(image, coorx.astype(np.float32), coory.astype(np.float32), 
            cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP
        )

    def _create_plane_mesh(self,
        vertices:           np.array, 
        vertices_floor:     np.array, 
        textures:           typing.List[np.array],
        texture_floor:      np.array,
        texture_ceiling:    np.array,
        delta_height:       np.array,
        ignore_ceiling:     bool=False,
    ) -> open3d.geometry.TriangleMesh: # create mesh for 3D floorplan visualization
        triangles, triangle_uvs = [], []
        num_walls = len(vertices) # the number of vertical walls
        # 1. vertical wall (always rectangle)
        num_vertices = 0
        for i in range(len(vertices)):        
            triangle = np.array([[0, 2, 1], [2, 0, 3]]) # hardcode triangles for each vertical wall
            triangles.append(triangle + num_vertices)
            num_vertices += 4
            triangle_uv = np.array(
                [
                    [i / (num_walls + 2), 0], 
                    [i / (num_walls + 2), 1], 
                    [(i+1) / (num_walls + 2), 1], 
                    [(i+1) / (num_walls + 2), 0]
                ],
                dtype=np.float32
            )
            triangle_uvs.append(triangle_uv)
        # 2. floor and ceiling    
        tri = Triangulator() # Since the floor and ceiling may not be a rectangle, triangulate the polygon first.
        for i in range(len(vertices_floor)):
            tri.add_vertex(vertices_floor[i, 0], vertices_floor[i, 1])
        for i in range(len(vertices_floor)):
            tri.add_polygon_vertex(i)
        tri.triangulate() # polygon triangulation    
        triangle = []
        for i in range(tri.getNumTriangles()):
            triangle.append([tri.get_triangle_v0(i), tri.get_triangle_v1(i), tri.get_triangle_v2(i)])
        triangle = np.array(triangle)
        # add triangles for floor and ceiling
        triangles.append(triangle + num_vertices)
        num_vertices += len(np.unique(triangle))
        if not ignore_ceiling:
            triangles.append(triangle + num_vertices)
        # texture for floor and ceiling
        vertices_floor_min = np.min(vertices_floor[:, :2], axis=0)
        vertices_floor_max = np.max(vertices_floor[:, :2], axis=0)
        # normalize to [0, 1]
        triangle_uv = (vertices_floor[:, :2] - vertices_floor_min) / (vertices_floor_max - vertices_floor_min)
        triangle_uv[:, 0] = (triangle_uv[:, 0] + num_walls) / (num_walls + 2) 
        triangle_uvs.append(triangle_uv)
        # normalize to [0, 1]
        triangle_uv = (vertices_floor[:, :2] - vertices_floor_min) / (vertices_floor_max - vertices_floor_min)
        triangle_uv[:, 0] = (triangle_uv[:, 0] + num_walls + 1) / (num_walls + 2)
        triangle_uvs.append(triangle_uv)
        # 3. Merge wall, floor, and ceiling
        vertices.append(vertices_floor)
        vertices.append(vertices_floor + delta_height)
        vertices = np.concatenate(vertices, axis=0)
        triangles = np.concatenate(triangles, axis=0)
        textures.append(texture_floor)
        textures.append(texture_ceiling)
        textures = np.concatenate(textures, axis=1)
        triangle_uvs = np.concatenate(triangle_uvs, axis=0)
        mesh = open3d.geometry.TriangleMesh(
            vertices=open3d.utility.Vector3dVector(vertices),
            triangles=open3d.utility.Vector3iVector(triangles)
        )
        mesh.compute_vertex_normals()
        # mesh.compute_vertex_normals()    
        mesh.texture = open3d.geometry.Image(textures)
        mesh.triangle_uvs = np.array(triangle_uvs[triangles.reshape(-1), :], dtype=np.float64)
        return mesh

    def create_mesh(self,
        panorama:       np.array,
        corners:        np.array,
        floor_z:        float=-1.6,
        ignore_ceiling: bool=True,        
    ):
        H, W, C = panorama.shape
        top_corners, bottom_corners = corners[::2], corners[1::2]
        junctions_floor, junctions_ceiling = self._uv2xyz(
            bottom_corners, top_corners, floor_z, W, H
        )
        junctions_floor[:, -1] = 0.0
        wall_height = np.mean(junctions_ceiling, axis=0)[-1] + abs(floor_z)
        delta_height = np.array([0, 0, wall_height])
        corners, textures = [], [] # 3D coordinate & texture  for each wall
        camera_center = np.array([0.0, 0.0, abs(floor_z)])
        for corner_i, corner_j in zip(junctions_floor, np.roll(junctions_floor, shift=-1, axis=0)):
            corner_j, corner_i = corner_i, corner_j
            texture = self._equirectangular_to_perspective(
                panorama, corner_i, corner_j, wall_height, camera_center,
                input_resolution=W, output_resolution=W, is_wall=True
            )
            corner = np.array([
                corner_i, corner_i + delta_height, corner_j + delta_height, corner_j
            ])
            corners.append(corner)
            textures.append(texture)
        # the floor/ceiling texture is cropped by the maximum bounding box
        corner_min = np.min(junctions_floor, axis=0)
        corner_max = np.max(junctions_floor, axis=0)
        texture_floor = self._equirectangular_to_perspective(
            panorama, corner_min, corner_max, 0, camera_center,
            input_resolution=W, output_resolution=W, is_wall=False
        )
        texture_ceiling = self._equirectangular_to_perspective(
            panorama, corner_min, corner_max, wall_height, camera_center,
            input_resolution=W, output_resolution=W, is_wall=False
        )
        mesh = self._create_plane_mesh(
            corners, junctions_floor,
            textures, texture_floor, texture_ceiling,
            delta_height, ignore_ceiling=ignore_ceiling
        )
        return mesh