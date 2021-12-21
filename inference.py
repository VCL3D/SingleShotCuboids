from ssc_handler import SscHandler
from hnet_handler import HNetHandler

import os
import cv2
import torch
import open3d
import glob
import argparse
import sys
import json
import tqdm

# supress mesh writing warnings
open3d.utility.set_verbosity_level(open3d.utility.VerbosityLevel.Error)

_HANDLERS_ = {
    'ssc': SscHandler,
    'ssc_syn': SscHandler,
    'hnet': HNetHandler,
    'hnet_syn': HNetHandler,
}

class Context:
    manifest = None
    system_properties = None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('glob', type=str)
    parser.add_argument('--model', type=str, default='ssc', 
        choices=['ssc', 'hnet', 'ssc_syn', 'hnet_syn']
    )
    parser.add_argument('--output_path', type=str, default='')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--floor_distance', type=float, default=-1.6)
    parser.add_argument('--remove_ceiling', action='store_true')
    parser.add_argument('--save_boundary', action='store_true')
    parser.add_argument('--save_mesh', action='store_true')
    parser.add_argument('--mesh_type', type=str, default='obj', 
        choices=['usdz', 'obj']
    )
    args = parser.parse_args()

    if args.model not in _HANDLERS_:
        print(f"Model ({args.model}) not available, use one of {list(_HANDLERS_.keys())} with the --model argument.")
        sys.exit(-1)

    handler = _HANDLERS_[args.model]()
    context = Context()
    setattr(context, "manifest", {
        "model": {
            "serializedFile": os.path.join("ckpt", f"{args.model}.pth")
        }
    })
    setattr(context, "system_properties", {
        "model_dir": os.getcwd(),
        "gpu_id": args.gpu,
    })
    handler.initialize(context)
    
    filenames = glob.glob(args.glob)
    for filename in tqdm.tqdm(filenames, desc='Running inference ... '):
        img = torch.from_numpy(
            cv2.imread(filename).transpose(2, 0, 1)        
        ).unsqueeze(0) / 255.0
        name, ext = os.path.splitext(os.path.basename(filename))
        corners = handler.handle([{
            "data": img,
            'outputs': {
                'boundary': f'{os.path.join(args.output_path, name)}_viz.JPG' if args.save_boundary else '',
                'mesh': f'{os.path.join(args.output_path, name)}.{args.mesh_type}' if args.save_mesh else '',
            },
            'floor_distance': args.floor_distance,
            'remove_ceiling': args.remove_ceiling,
        }], None)
        with open(os.path.join(args.output_path, name + '.json'), 'w') as f:
            json.dump(corners[0], f)