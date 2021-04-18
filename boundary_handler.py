import numpy as np
import numpy.matlib as matlib

#NOTE: code borrowed from https://github.com/bertjiazheng/Structured3D

class BoundaryHandler(object):
    def __init__(self):
        pass

    def _uv2xyzN(self, uv, planeID=1):
        ID1 = (int(planeID) - 1 + 0) % 3
        ID2 = (int(planeID) - 1 + 1) % 3
        ID3 = (int(planeID) - 1 + 2) % 3
        xyz = np.zeros((uv.shape[0], 3))
        xyz[:, ID1] = np.cos(uv[:, 1]) * np.sin(uv[:, 0])
        xyz[:, ID2] = np.cos(uv[:, 1]) * np.cos(uv[:, 0])
        xyz[:, ID3] = np.sin(uv[:, 1])
        return xyz

    def _coords2uv(self, coords, width, height):
        """
        Image coordinates (xy) to uv
        """
        middleX = width / 2 + 0.5
        middleY = height / 2 + 0.5
        uv = np.hstack([
            (coords[:, [0]] - middleX) / width * 2 * np.pi,
            -(coords[:, [1]] - middleY) / height * np.pi])
        return uv

    def _xyz2uvN(self, xyz, planeID=1):
        ID1 = (int(planeID) - 1 + 0) % 3
        ID2 = (int(planeID) - 1 + 1) % 3
        ID3 = (int(planeID) - 1 + 2) % 3
        normXY = np.sqrt(xyz[:, [ID1]] ** 2 + xyz[:, [ID2]] ** 2)
        normXY[normXY < 0.000001] = 0.000001
        normXYZ = np.sqrt(xyz[:, [ID1]] ** 2 + xyz[:, [ID2]] ** 2 + xyz[:, [ID3]] ** 2)
        v = np.arcsin(xyz[:, [ID3]] / normXYZ)
        u = np.arcsin(xyz[:, [ID1]] / normXY)
        valid = (xyz[:, [ID2]] < 0) & (u >= 0)
        u[valid] = np.pi - u[valid]
        valid = (xyz[:, [ID2]] < 0) & (u <= 0)
        u[valid] = -np.pi - u[valid]
        uv = np.hstack([u, v])
        uv[np.isnan(uv[:, 0]), 0] = 0
        return uv

    def _computeUVN(self, n, in_, planeID):
        """
        compute v given u and normal.
        """
        if planeID == 2:
            n = np.array([n[1], n[2], n[0]])
        elif planeID == 3:
            n = np.array([n[2], n[0], n[1]])
        bc = n[0] * np.sin(in_) + n[1] * np.cos(in_)
        bs = n[2]
        out = np.arctan(-bc / (bs + 1e-9))
        return out

    def _lineFromTwoPoint(self, pt1, pt2):
        """
        Generate line segment based on two points on panorama
        pt1, pt2: two points on panorama
        line:
            1~3-th dim: normal of the line
            4-th dim: the projection dimension ID
            5~6-th dim: the u of line segment endpoints in projection plane
        """
        numLine = pt1.shape[0]
        lines = np.zeros((numLine, 6))
        n = np.cross(pt1, pt2)
        n = n / (matlib.repmat(np.sqrt(np.sum(n ** 2, 1, keepdims=True)), 1, 3) + 1e-9)
        lines[:, 0:3] = n

        areaXY = np.abs(np.sum(n * matlib.repmat([0, 0, 1], numLine, 1), 1, keepdims=True))
        areaYZ = np.abs(np.sum(n * matlib.repmat([1, 0, 0], numLine, 1), 1, keepdims=True))
        areaZX = np.abs(np.sum(n * matlib.repmat([0, 1, 0], numLine, 1), 1, keepdims=True))
        planeIDs = np.argmax(np.hstack([areaXY, areaYZ, areaZX]), axis=1) + 1
        lines[:, 3] = planeIDs

        for i in range(numLine):
            uv = self._xyz2uvN(np.vstack([pt1[i, :], pt2[i, :]]), lines[i, 3])
            umax = uv[:, 0].max() + np.pi
            umin = uv[:, 0].min() + np.pi
            if umax - umin > np.pi:
                lines[i, 4:6] = np.array([umax, umin]) / 2 / np.pi
            else:
                lines[i, 4:6] = np.array([umin, umax]) / 2 / np.pi

        return lines

    def _lineIdxFromCors(self, cor_all, im_w, im_h):
        assert len(cor_all) % 2 == 0
        uv = self._coords2uv(cor_all, im_w, im_h)
        xyz = self._uv2xyzN(uv)
        lines = self._lineFromTwoPoint(xyz[0::2], xyz[1::2])
        num_sample = max(im_h, im_w)

        cs, rs = [], []
        for i in range(lines.shape[0]):
            n = lines[i, 0:3]
            sid = lines[i, 4] * 2 * np.pi
            eid = lines[i, 5] * 2 * np.pi
            if eid < sid:
                x = np.linspace(sid, eid + 2 * np.pi, num_sample)
                x = x % (2 * np.pi)
            else:
                x = np.linspace(sid, eid, num_sample)

            u = -np.pi + x.reshape(-1, 1)
            v = self._computeUVN(n, u, lines[i, 3])
            xyz = self._uv2xyzN(np.hstack([u, v]), lines[i, 3])
            uv = self._xyz2uvN(xyz, 1)

            r = np.minimum(np.floor((uv[:, 0] + np.pi) / (2 * np.pi) * im_w) + 1,
                        im_w).astype(np.int32)
            c = np.minimum(np.floor((np.pi / 2 - uv[:, 1]) / np.pi * im_h) + 1,
                        im_h).astype(np.int32)
            cs.extend(r - 1)
            rs.extend(c - 1)
        return rs, cs

    def _draw_boundary_from_cor_id(self, cor_id, img):
        im_h, im_w, c = img.shape
        cor_all_top = []
        cor_all_bot = []
        for i in range(len(cor_id)):
            if i%2 == 0:#top
                cor_all_top.append(cor_id[i, :])
                cor_all_top.append(cor_id[(i+2) % len(cor_id), :])
            else:
                cor_all_bot.append(cor_id[i, :])
                cor_all_bot.append(cor_id[(i+2) % len(cor_id), :])
        cor_all_top = np.vstack(cor_all_top)
        cor_all_bot = np.vstack(cor_all_bot)

        rs_top, cs_top = self._lineIdxFromCors(cor_all_top, im_w, im_h)
        rs_top = np.array(rs_top)
        cs_top = np.array(cs_top)

        rs_bot, cs_bot = self._lineIdxFromCors(cor_all_bot, im_w, im_h)
        rs_bot = np.array(rs_bot)
        cs_bot = np.array(cs_bot)

        for dx, dy in [[-1, 0], [1, 0], [0, 0], [0, 1], [0, -1]]:            
            img[np.clip(rs_top + dx, 0, im_h - 1), np.clip(cs_top + dy, 0, im_w - 1), 1] = 255
            img[np.clip(rs_top + dx, 0, im_h - 1), np.clip(cs_top + dy, 0, im_w - 1), 0] = 120
            img[np.clip(rs_bot + dx, 0, im_h - 1), np.clip(cs_bot + dy, 0, im_w - 1), 2] = 255
            img[np.clip(rs_bot + dx, 0, im_h - 1), np.clip(cs_bot + dy, 0, im_w - 1), 0] = 120
        return img, rs_top, cs_top

    def create_boundary(self,
        panorama:       np.array,
        corners:        np.array,
    ):
        pano, _, __ = self._draw_boundary_from_cor_id(
            corners, panorama
        )
        return pano