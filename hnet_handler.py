#NOTE: Code adapted from https://github.com/sunset1995/HorizonNet  

import json
import toolz
import sys
import numpy as np
import os
import logging
import torch
import io
import cv2
import typing
import requests
import open3d

logger = logging.getLogger(__name__)
logger.info("HorizonNet handler initialization.")

from PIL import Image
from scipy.ndimage.filters import maximum_filter
from scipy.spatial.distance import pdist, squareform

try:
    from hnet.model import HorizonNet
    logger.info("Handler spawned from torchserve.")
except ImportError:
    from model import HorizonNet
from shapely.geometry import Polygon
from obj_handler import ObjHandler
from usdz_exporter import UsdzExporter
from boundary_handler import BoundaryHandler

from urllib.parse import urlparse

def is_url(url):
  try:
    result = urlparse(url)
    return all([result.scheme, result.netloc])
  except ValueError:
    return False

class HNetHandler(ObjHandler, UsdzExporter, BoundaryHandler):
    PI = float(np.pi)

    def __init__(self):
        super(HNetHandler, self).__init__()

    def _np_coorx2u(self, coorx, coorW=1024):
        return ((coorx + 0.5) / coorW - 0.5) * 2 * HNetHandler.PI

    def _np_coory2v(self, coory, coorH=512):
        return -((coory + 0.5) / coorH - 0.5) * HNetHandler.PI
        
    def _mean_percentile(self, vec, p1=25, p2=75):
        vmin = np.percentile(vec, p1)
        vmax = np.percentile(vec, p2)
        return vec[(vmin <= vec) & (vec <= vmax)].mean()

    def _np_refine_by_fix_z(self, coory0, coory1, z0=50, coorH=512):
        '''
        Refine coory1 by coory0
        coory0 are assumed on given plane z
        '''
        v0 = self._np_coory2v(coory0, coorH)
        v1 = self._np_coory2v(coory1, coorH)

        c0 = z0 / np.tan(v0)
        z1 = c0 * np.tan(v1)
        z1_mean = self._mean_percentile(z1)
        v1_refine = np.arctan2(z1_mean, c0)
        coory1_refine = (-v1_refine / HNetHandler.PI + 0.5) * coorH - 0.5

        return coory1_refine, z1_mean

    def _find_N_peaks(self, signal, r=29, min_v=0.05, N=None):
        max_v = maximum_filter(signal, size=r, mode='wrap')
        pk_loc = np.where(max_v == signal)[0]
        pk_loc = pk_loc[signal[pk_loc] > min_v]
        if N is not None:
            order = np.argsort(-signal[pk_loc])
            pk_loc = pk_loc[order[:N]]
            pk_loc = pk_loc[np.argsort(pk_loc)]
        return pk_loc, signal[pk_loc]

    def _get_gpid(self, coorx, coorW):
        gpid = np.zeros(coorW)
        gpid[np.round(coorx).astype(int)] = 1
        gpid = np.cumsum(gpid).astype(int)
        gpid[gpid == gpid[-1]] = 0
        return gpid
     
    def _vote(self, vec, tol):
        vec = np.sort(vec)
        n = np.arange(len(vec))[::-1]
        n = n[:, None] - n[None, :] + 1.0
        l = squareform(pdist(vec[:, None], 'minkowski', p=1) + 1e-9)

        invalid = (n < len(vec) * 0.4) | (l > tol)
        if (~invalid).sum() == 0 or len(vec) < tol:
            best_fit = np.median(vec)
            p_score = 0
        else:
            l[invalid] = 1e5
            n[invalid] = -1
            score = n
            max_idx = score.argmax()
            max_row = max_idx // len(vec)
            max_col = max_idx % len(vec)
            assert max_col > max_row
            best_fit = vec[max_row:max_col+1].mean()
            p_score = (max_col - max_row + 1) / len(vec)

        l1_score = np.abs(vec - best_fit).mean()

        return best_fit, p_score, l1_score
        
    def _gen_ww_cuboid(self, xy, gpid, tol):
        xy_cor = []
        assert len(np.unique(gpid)) == 4
        # For each part seperated by wall-wall peak, voting for a wall
        for j in range(4):
            now_x = xy[gpid == j, 0]
            now_y = xy[gpid == j, 1]
            new_x, x_score, x_l1 = self._vote(now_x, tol)
            new_y, y_score, y_l1 = self._vote(now_y, tol)
            if (x_score, -x_l1) > (y_score, -y_l1):
                xy_cor.append({'type': 0, 'val': new_x, 'score': x_score})
            else:
                xy_cor.append({'type': 1, 'val': new_y, 'score': y_score})
        # Sanity fallback
        scores = [0, 0]
        for j in range(4):
            if xy_cor[j]['type'] == 0:
                scores[j % 2] += xy_cor[j]['score']
            else:
                scores[j % 2] -= xy_cor[j]['score']
        if scores[0] > scores[1]:
            xy_cor[0]['type'] = 0
            xy_cor[1]['type'] = 1
            xy_cor[2]['type'] = 0
            xy_cor[3]['type'] = 1
        else:
            xy_cor[0]['type'] = 1
            xy_cor[1]['type'] = 0
            xy_cor[2]['type'] = 1
            xy_cor[3]['type'] = 0

        return xy_cor

    def _np_x_u_solve_y(self, x, u, floorW=1024, floorH=512):
        c = (x - floorW / 2 + 0.5) / np.sin(u)
        return -c * np.cos(u) + floorH / 2 - 0.5


    def _np_y_u_solve_x(self, y, u, floorW=1024, floorH=512):
        c = -(y - floorH / 2 + 0.5) / np.cos(u)
        return c * np.sin(u) + floorW / 2 - 0.5

    def _np_xy2coor(self, xy, z=50, coorW=1024, coorH=512, floorW=1024, floorH=512):
        '''
        xy: N x 2
        '''
        x = xy[:, 0] - floorW / 2 + 0.5
        y = xy[:, 1] - floorH / 2 + 0.5

        u = np.arctan2(x, -y)
        v = np.arctan(z / np.sqrt(x**2 + y**2))

        coorx = (u / (2 * HNetHandler.PI) + 0.5) * coorW - 0.5
        coory = (-v / HNetHandler.PI + 0.5) * coorH - 0.5

        return np.hstack([coorx[:, None], coory[:, None]])

    def _gen_ww_general(self, init_coorx, xy, gpid, tol):
        xy_cor = []
        assert len(init_coorx) == len(np.unique(gpid))
        # Candidate for each part seperated by wall-wall boundary
        for j in range(len(init_coorx)):
            now_x = xy[gpid == j, 0]
            now_y = xy[gpid == j, 1]
            new_x, x_score, x_l1 = self._vote(now_x, tol)
            new_y, y_score, y_l1 = self._vote(now_y, tol)
            u0 = self._np_coorx2u(init_coorx[(j - 1 + len(init_coorx)) % len(init_coorx)])
            u1 = self._np_coorx2u(init_coorx[j])
            if (x_score, -x_l1) > (y_score, -y_l1):
                xy_cor.append({'type': 0, 'val': new_x, 'score': x_score, 'action': 'ori', 'gpid': j, 'u0': u0, 'u1': u1, 'tbd': True})
            else:
                xy_cor.append({'type': 1, 'val': new_y, 'score': y_score, 'action': 'ori', 'gpid': j, 'u0': u0, 'u1': u1, 'tbd': True})
        # Construct wall from highest score to lowest
        while True:
            # Finding undetermined wall with highest score
            tbd = -1
            for i in range(len(xy_cor)):
                if xy_cor[i]['tbd'] and (tbd == -1 or xy_cor[i]['score'] > xy_cor[tbd]['score']):
                    tbd = i
            if tbd == -1:
                break
            # This wall is determined
            xy_cor[tbd]['tbd'] = False
            p_idx = (tbd - 1 + len(xy_cor)) % len(xy_cor)
            n_idx = (tbd + 1) % len(xy_cor)

            num_tbd_neighbor = xy_cor[p_idx]['tbd'] + xy_cor[n_idx]['tbd']
            # Two adjacency walls are not determined yet => not special case
            if num_tbd_neighbor == 2:
                continue
            # Only one of adjacency two walls is determine => add now or later special case
            if num_tbd_neighbor == 1:
                if (not xy_cor[p_idx]['tbd'] and xy_cor[p_idx]['type'] == xy_cor[tbd]['type']) or\
                        (not xy_cor[n_idx]['tbd'] and xy_cor[n_idx]['type'] == xy_cor[tbd]['type']):
                    # Current wall is different from one determined adjacency wall
                    if xy_cor[tbd]['score'] >= -1:
                        # Later special case, add current to tbd
                        xy_cor[tbd]['tbd'] = True
                        xy_cor[tbd]['score'] -= 100
                    else:
                        # Fallback: forced change the current wall or infinite loop
                        if not xy_cor[p_idx]['tbd']:
                            insert_at = tbd
                            if xy_cor[p_idx]['type'] == 0:
                                new_val = self._np_x_u_solve_y(xy_cor[p_idx]['val'], xy_cor[p_idx]['u1'])
                                new_type = 1
                            else:
                                new_val = self._np_y_u_solve_x(xy_cor[p_idx]['val'], xy_cor[p_idx]['u1'])
                                new_type = 0
                        else:
                            insert_at = n_idx
                            if xy_cor[n_idx]['type'] == 0:
                                new_val = self._np_x_u_solve_y(xy_cor[n_idx]['val'], xy_cor[n_idx]['u0'])
                                new_type = 1
                            else:
                                new_val = self._np_y_u_solve_x(xy_cor[n_idx]['val'], xy_cor[n_idx]['u0'])
                                new_type = 0
                        new_add = {'type': new_type, 'val': new_val, 'score': 0, 'action': 'forced infer', 'gpid': -1, 'u0': -1, 'u1': -1, 'tbd': False}
                        xy_cor.insert(insert_at, new_add)
                continue
            # Below checking special case
            if xy_cor[p_idx]['type'] == xy_cor[n_idx]['type']:
                # Two adjacency walls are same type, current wall should be differen type
                if xy_cor[tbd]['type'] == xy_cor[p_idx]['type']:
                    # Fallback: three walls with same type => forced change the middle wall
                    xy_cor[tbd]['type'] = (xy_cor[tbd]['type'] + 1) % 2
                    xy_cor[tbd]['action'] = 'forced change'
                    xy_cor[tbd]['val'] = xy[gpid == xy_cor[tbd]['gpid'], xy_cor[tbd]['type']].mean()
            else:
                # Two adjacency walls are different type => add one
                tp0 = xy_cor[n_idx]['type']
                tp1 = xy_cor[p_idx]['type']
                if xy_cor[p_idx]['type'] == 0:
                    val0 = self._np_x_u_solve_y(xy_cor[p_idx]['val'], xy_cor[p_idx]['u1'])
                    val1 = self._np_y_u_solve_x(xy_cor[n_idx]['val'], xy_cor[n_idx]['u0'])
                else:
                    val0 = self._np_y_u_solve_x(xy_cor[p_idx]['val'], xy_cor[p_idx]['u1'])
                    val1 = self._np_x_u_solve_y(xy_cor[n_idx]['val'], xy_cor[n_idx]['u0'])
                new_add = [
                    {'type': tp0, 'val': val0, 'score': 0, 'action': 'forced infer', 'gpid': -1, 'u0': -1, 'u1': -1, 'tbd': False},
                    {'type': tp1, 'val': val1, 'score': 0, 'action': 'forced infer', 'gpid': -1, 'u0': -1, 'u1': -1, 'tbd': False},
                ]
                xy_cor = xy_cor[:tbd] + new_add + xy_cor[tbd+1:]

        return xy_cor

    def _np_coor2xy(self, coor, z=50, coorW=1024, coorH=512, floorW=1024, floorH=512):
        '''
        coor: N x 2, index of array in (col, row) format
        '''
        coor = np.array(coor)
        u = self._np_coorx2u(coor[:, 0], coorW)
        v = self._np_coory2v(coor[:, 1], coorH)
        c = z / np.tan(v)
        x = c * np.sin(u) + floorW / 2 - 0.5
        y = -c * np.cos(u) + floorH / 2 - 0.5
        return np.hstack([x[:, None], y[:, None]])

    def _gen_ww(self, init_coorx, coory, z=50, coorW=1024, coorH=512, floorW=1024, floorH=512, tol=3, force_cuboid=True):
        gpid = self._get_gpid(init_coorx, coorW)
        coor = np.hstack([np.arange(coorW)[:, None], coory[:, None]])
        xy = self._np_coor2xy(coor, z, coorW, coorH, floorW, floorH)
        # Generate wall-wall
        if force_cuboid:
            xy_cor = self._gen_ww_cuboid(xy, gpid, tol)
        else:
            xy_cor = self._gen_ww_general(init_coorx, xy, gpid, tol)
        # Ceiling view to normal view
        cor = []
        for j in range(len(xy_cor)):
            next_j = (j + 1) % len(xy_cor)
            if xy_cor[j]['type'] == 1:
                cor.append((xy_cor[next_j]['val'], xy_cor[j]['val']))
            else:
                cor.append((xy_cor[j]['val'], xy_cor[next_j]['val']))
        cor = self._np_xy2coor(np.array(cor), z, coorW, coorH, floorW, floorH)
        cor = np.roll(cor, -2 * cor[::2, 0].argmin(), axis=0)

        return cor, xy_cor

    def _infer_coory(self, coory0, h, z0=50, coorH=512):
        v0 = self._np_coory2v(coory0, coorH)
        c0 = z0 / np.tan(v0)
        z1 = z0 + h
        v1 = np.arctan2(z1, c0)
        return (-v1 / HNetHandler.PI + 0.5) * coorH - 0.5


    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return:
        """
        self._context = context
        self.manifest = context.manifest
        
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        serialized_file = self.manifest['model']['serializedFile']
        model_pt_path = os.path.join(model_dir, serialized_file)
        if not os.path.isfile(model_pt_path):
            error_message = f"Missing the serialized model weights file({model_pt_path})"
            logger.error(error_message)
            raise RuntimeError(error_message)

        checkpoint = torch.load(model_pt_path, map_location=self.device)['state_dict']
        self.model = HorizonNet()
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()
        self.initialized = True
        logger.info("Coarse Geometry Model Loaded Successfully.")

    def preprocess(self, data):
        """
        Transform raw input into model input data.
        :param batch: list of raw requests, should match batch size
        :return: list of preprocessed model input data
        """
        for row in data:
            if isinstance(row.get("data"), torch.Tensor):
                image = row.get("data").to(self.device)
                mesh_url = row.get('outputs', {}).get('mesh', '')
                viz_url = row.get('outputs', {}).get('boundary', '')
                metadata = row.get('Source', {'sceneId': 'test', 'type': 'panorama'})
                #metadata = row.get('Source')
                #logger.info(metadata)
                break
            elif 'data' in row and isinstance(row.get('data'), dict):
                json = row['data']
                logger.warning(f"json: {json}")
                color_url = json['inputs']['color']
                viz_url = json['outputs']['boundary']
                mesh_url = json['outputs']['mesh']
                metadata = json.get('Source', {'sceneId': 'test', 'type': 'panorama'})
                #metadata = json.get('Source')
                #logger.info(metadata)
                r = requests.get(color_url, timeout=1.0) #TODO: make timeout configurable
                image = r.content
            elif 'body' in row and isinstance(row.get('body'), dict):
                json = row['body']
                logger.warning(f"json: {json}")
                color_url = json['inputs']['color']
                viz_url = json['outputs']['boundary']
                mesh_url = json['outputs']['mesh']
                metadata = json.get('Source', {'sceneId': 'test', 'type': 'panorama'})
                #metadata = json.get('Source')
                #logger.info(metadata)
                r = requests.get(color_url, timeout=1.0) #TODO: make timeout configurable
                image = r.content
            else:
                image = row.get("data") or row.get("body")
                mesh_url, viz_url = '', ''
                metadata = row.get('Source', {'sceneId': 'test', 'type': 'panorama'})
                #metadata = row.get('Source')
                #logger.info(metadata)
            raw = io.BytesIO(image)
            image = Image.open(raw)
            image = np.array(image) # cvt color?
            image = image.transpose(2, 0, 1)
            image = torch.from_numpy(image).unsqueeze(0).float() / 255.0
            image = image.to(self.device)
            break
        logger.info(f"metadata : {metadata}")
        original = image.clone()
        resolution = image.shape[2:]
        image = torch.nn.functional.interpolate(
            image, size=[512, 1024], mode='bilinear', align_corners=True)
        return {
            'metadata': metadata,
            'panorama': {
                'original': original,
                'scaled': image,
            },
            'resolution': {
                'width': resolution[-1],
                'height': resolution[0],
            },
            'outputs': {
                'mesh': mesh_url,
                'boundary': viz_url,
            },
            'floor_distance': row.get('floor_distance', -1.6),
            'ignore_ceiling': row.get('remove_ceiling', True)
        }

    def inference(self, model_inputs):
        """
        Internal inference methods
        :param model_input: transformed model input data
        :return: list of inference output in NDArray
        """
        with torch.no_grad():
            y_bon, y_cor = self.model(model_inputs['panorama']['scaled'])
        return toolz.merge({
            'heights': y_bon,
            'corners': torch.sigmoid(y_cor),
            # 'original': model_inputs['panorama']['original'],
        }, model_inputs)

    def postprocess(self, inference_output):
        """
        Return inference result.
        :param inference_output: list of inference output
        :return: list of predict results
        """
        force_cuboid = False #TODO: add as param
        r = 0.05        
        W = 1024
        H = W // 2
        min_v = None
        # Take output from network and post-process to desired format
        y_bon_ = inference_output['heights']
        y_cor_ = inference_output['corners']
        img = inference_output['panorama']['scaled']
        # H, W = inference_output['resolution']['height'], inference_output['resolution']['width']
        y_bon_ = (y_bon_[0].cpu().numpy() / np.pi + 0.5) * H - 0.5
        y_cor_ = y_cor_[0, 0].cpu().numpy()

        # Init floor/ceil plane
        z0 = 50
        _, z1 = self._np_refine_by_fix_z(*y_bon_, z0)
        # Detech wall-wall peaks
        if min_v is None:
            min_v = 0 if force_cuboid else 0.05
        r = int(round(W * r / 2))
        N = 4 if force_cuboid else None
        xs_ = self._find_N_peaks(y_cor_, r=r, min_v=min_v, N=N)[0]
        # Generate wall-walls
        cor, xy_cor = self._gen_ww(xs_, y_bon_[0], z0, tol=abs(0.16 * z1 / 1.6), force_cuboid=force_cuboid)
        if not force_cuboid:
            # Check valid (for fear self-intersection)
            xy2d = np.zeros((len(xy_cor), 2), np.float32)
            for i in range(len(xy_cor)):
                xy2d[i, xy_cor[i]['type']] = xy_cor[i]['val']
                xy2d[i, xy_cor[i-1]['type']] = xy_cor[i-1]['val']
            if not Polygon(xy2d).is_valid:
                print(
                    'Fail to generate valid general layout!! '
                    'Generate cuboid as fallback.',
                    file=sys.stderr)
                xs_ = self._find_N_peaks(y_cor_, r=r, min_v=0, N=4)[0]
                cor, xy_cor = self._gen_ww(xs_, y_bon_[0], z0, tol=abs(0.16 * z1 / 1.6), force_cuboid=True)
        # Expand with btn coory
        cor = np.hstack([cor, self._infer_coory(cor[:, 1], z1 - z0, z0)[:, None]])
        # Collect corner position in equirectangular
        cor_id = np.zeros((len(cor)*2, 2), np.float32)
        for j in range(len(cor)):
            cor_id[j*2] = cor[j, 0], cor[j, 1]
            cor_id[j*2 + 1] = cor[j, 0], cor[j, 2]
        # Normalized to [0, 1]
        cor_id[:, 0] /= W
        cor_id[:, 1] /= H
        
        cor_id[:, 0] *= W
        cor_id[:, 1] *= H
        boundary_uri = inference_output['outputs']['boundary']
        mesh_uri = inference_output['outputs']['mesh']
        if boundary_uri or mesh_uri:
            img = cv2.cvtColor(
                img.cpu().numpy().squeeze().transpose(1, 2, 0),
                cv2.COLOR_BGR2RGB
            )
            img = (img * 255.0).astype(np.uint8)
        # mesh
        if mesh_uri:
            floor_z = inference_output.get('floor_distance', -1.6)
            ignore_ceiling = inference_output.get('remove_ceiling', True)
            mesh = self.create_obj_mesh(img, cor_id, floor_z, ignore_ceiling)
            out_file = io.BytesIO()
            tex = Image.fromarray(np.asarray(mesh.texture)) # np.asarray(mesh.texture)[:, :, ::-1]
            tex.save(out_file, 'JPEG')
            out_file.seek(0)
            scene_name = inference_output['metadata']['sceneId']
            if is_url(mesh_uri):
                requests.post(inference_output['outputs']['mesh'],                
                    files={
                        'json': (None, json.dumps({
                            'metadata': inference_output['metadata'],
                            'mesh': {
                                'vertices': np.asarray(mesh.vertices).tolist(),
                                'triangles': np.asarray(mesh.triangles).tolist(),
                                'normals': np.asarray(mesh.vertex_normals).tolist(),
                                'triangle_uvs': [uv.tolist() for uv in mesh.triangle_uvs],
                            }
                        }), 'application/json'),
                        'texture': ('test.obj', out_file, 'application/octet-stream'),
                        'mesh': (f'{scene_name}.usdz', self.export_usdz(mesh, scene_name, io.BytesIO()), 'application/octet-stream'),
                    }
                )
            elif os.path.exists(os.path.dirname(mesh_uri) or os.getcwd()):
                if '.obj' in mesh_uri:
                    open3d.io.write_triangle_mesh(mesh_uri, mesh)
                elif '.usdz' in mesh_uri:
                    self.export_usdz(mesh, scene_name)
                else:
                    logger.error(f'Mesh file type ({mesh_uri}) not supported.')
            else:
                logger.warning(f'Mesh URI ({mesh_uri}) is not valid.')
        if boundary_uri:
            pano = self.create_boundary(img, cor_id)
            out_img = Image.fromarray(pano.astype(np.uint8))
            out_file = io.BytesIO()
            out_img.save(out_file, 'JPEG')
            out_file.seek(0)
            if is_url(boundary_uri):
                requests.post(boundary_uri, files={
                    'json': (None, json.dumps({
                        'metadata': inference_output['metadata']
                    })),
                    'image': out_file
                })
            elif os.path.exists(os.path.dirname(boundary_uri) or os.getcwd()):
                with open(boundary_uri, 'wb') as f:
                    f.write(out_file.getbuffer())
            else:
                logger.warning(f'Boundary URI ({boundary_uri}) is not valid.')
        return [cor_id.tolist()]

    def handle(self, data, context):
        """
        Invoke by TorchServe for prediction request.
        Do pre-processing of data, prediction using model and postprocessing of prediciton output
        :param data: Input data for prediction
        :param context: Initial context contains model server system properties.
        :return: prediction output
        """
        model_input = self.preprocess(data)
        model_output = self.inference(model_input)
        return self.postprocess(model_output)