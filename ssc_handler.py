import cv2
import toolz
import json
import requests
import numpy as np
import os
import logging
import torch
import io
import open3d

from PIL import Image

try:
    from ssc.model import StackedHourglass as Model
    from ssc.cuboid_fitting import CuboidFitting
    from ssc.quasi_manhattan_center_of_mass import QuasiManhattanCenterOfMass as CoM
    from ssc.spherical_grid import Grid
except ImportError:
    from model import StackedHourglass as Model
    from cuboid_fitting import CuboidFitting
    from quasi_manhattan_center_of_mass import QuasiManhattanCenterOfMass as CoM
    from spherical_grid import Grid

from mesh_handler import MeshHandler
from boundary_handler import BoundaryHandler

logger = logging.getLogger(__name__)

from urllib.parse import urlparse

def is_url(url):
  try:
    result = urlparse(url)
    return all([result.scheme, result.netloc])
  except ValueError:
    return False

class SscHandler(MeshHandler, BoundaryHandler):
    PI = float(np.pi)

    def __init__(self,):
        super(SscHandler, self).__init__()

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

        checkpoint = torch.load(model_pt_path, map_location=self.device)
        self.model = Model()
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()
        self.cuboid = CuboidFitting().to(self.device)
        self.com = CoM().to(self.device)
        self.grid = Grid(width=512//4, height=256//4).to(self.device)
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
                break
            elif 'data' in row and isinstance(row.get('data'), dict):
                json = row['data']
                color_url = json['inputs']['color']
                viz_url = json['outputs']['boundary']
                mesh_url = json['outputs']['mesh']
                r = requests.get(color_url, timeout=1.0) #TODO: make timeout configurable
                image = r.content
            elif 'body' in row and isinstance(row.get('body'), dict):
                json = row['body']
                color_url = json['inputs']['color']
                viz_url = json['outputs']['boundary']
                mesh_url = json['outputs']['mesh']
                r = requests.get(color_url, timeout=1.0) #TODO: make timeout configurable
                image = r.content
            else:
                image = row.get("data") or row.get("body")
                mesh_url, viz_url = '', ''
            raw = io.BytesIO(image)
            image = Image.open(raw)
            image = np.array(image) # cvt color?
            image = image.transpose(2, 0, 1)
            image = torch.from_numpy(image).unsqueeze(0).float() / 255.0
            image = image.to(self.device)
            break        
        original = image.clone()
        if original.shape[-1] > 2048:
            original = torch.nn.functional.interpolate(
                original, 
                size=[1024, 2048], 
                mode='area',
                align_corners=None
            )
        resolution = original.shape[2:]
        image = torch.nn.functional.interpolate(
            image, 
            size=[256, 512], 
            mode='area' if image.shape[-1] > 512 else 'bilinear',
            align_corners=None if image.shape[-1] > 512 else True
        )
        return {
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
            heatmaps = toolz.last(self.model(model_inputs['panorama']['scaled']))
            gaussians = []
            hh, hw = heatmaps.shape[2:]
            for h in heatmaps.squeeze(0):
                gaussians.append(
                    torch.softmax(h.view(-1), dim=0).view(hh, hw)
                )
            gaussians = torch.stack(gaussians).unsqueeze(0)
            coords = self.com(self.grid(model_inputs['panorama']['scaled']), gaussians)
            self.cuboid.floor_distance = model_inputs['floor_distance']
            coords = self.cuboid(coords)
        return toolz.merge({            
            'coords': coords.squeeze(),             
        }, model_inputs)

    def postprocess(self, inference_outputs):
        """
        Return inference result.
        :param inference_output: list of inference output
        :return: list of predict results
        """
        panorama = inference_outputs['panorama']['original']
        coords = inference_outputs['coords']
        resolution = inference_outputs['resolution']
        # print(coords)
        coords[:, 0] = ((coords[:, 0] * np.pi + np.pi) / (2.0 * np.pi)) * resolution['width']
        coords[:, 1] = (coords[:, 1] * np.pi * 0.5 + np.pi * 0.5) / np.pi * resolution['height']
        cor_id = np.zeros((len(coords), 2), np.float32)
        coords = coords.cpu().numpy()
        cor_id[::2, :] = coords[:4]
        cor_id[1::2, :] = coords[4:]
        # boundary image
        boundary_uri = inference_outputs['outputs']['boundary']
        mesh_uri = inference_outputs['outputs']['mesh']
        if boundary_uri or mesh_uri:
            img = cv2.cvtColor(
                panorama.cpu().numpy().squeeze().transpose(1, 2, 0),
                cv2.COLOR_BGR2RGB
            )
            img = (img * 255.0).astype(np.uint8)
        # mesh
        if mesh_uri:
            floor_z = inference_outputs.get('floor_distance', -1.6)
            ignore_ceiling = inference_outputs.get('remove_ceiling', True)
            mesh = self.create_mesh(img, cor_id, floor_z, ignore_ceiling)
            out_file = io.BytesIO()
            tex = Image.fromarray(np.asarray(mesh.texture))
            tex.save(out_file, 'JPEG')
            out_file.seek(0)
            if is_url(mesh_uri):
                requests.post(inference_outputs['outputs']['mesh'],                
                    files={
                        'json': (None, json.dumps({'mesh': {                    
                            'vertices': np.asarray(mesh.vertices).tolist(),
                            'triangles': np.asarray(mesh.triangles).tolist(),
                            'normals': np.asarray(mesh.vertex_normals).tolist(),
                            'triangle_uvs': [uv.tolist() for uv in mesh.triangle_uvs],                        
                        }}), 'application/json'),
                        'texture': ('test.obj', out_file, 'application/octet-stream'),
                    }
                )
            elif os.path.exists(os.path.dirname(mesh_uri) or os.getcwd()):
                open3d.io.write_triangle_mesh(mesh_uri, mesh)
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