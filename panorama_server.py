# FTT Simple webserver serving static files and print posted body to console
# Listen on port 5000

#NOTE: Adapted from the image_server created by Werner Bailer (wbailer@GitHub)

import argparse
import json
import sys
import os
import flask
import glob
import numpy as np
from PIL import Image
import open3d

# Image placeholder size for /
WIDTH = 50
HEIGHT = 40
# Image Page layout for /
TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title></title>
    <meta charset="utf-8" />
    <style>
body {
    margin: 0;
    background-color: #FFF;
}
.image {
    display: block;
    margin-left: 2em;
    background-color: #EEE;
    box-shadow: 0 0 5px rgba(0,0,0,0.3);
}
img {
    display: block;
}
    </style>
    <script src="https://code.jquery.com/jquery-1.10.2.min.js" charset="utf-8"></script>
    <script src="http://luis-almeida.github.io/unveil/jquery.unveil.min.js" charset="utf-8"></script>
    <script>
$(document).ready(function() {
    $('img').unveil(1000);
});
    </script>
</head>
<body>
    {% for image in images %}
        <a class="image" href="{{ image.src }}" style="width: {{ image.width }}px; height: {{ image.height }}px">
            <img src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" data-src="{{ image.src }}?w={{ image.width }}&amp;h={{ image.height }}" width="{{ image.width }}" height="{{ image.height }}" alt="{{image.src}}" />
        </a>
        <p>{{image.src}}</p>
    {% endfor %}
</body>
'''

def run(
    panorama_glob: str,
    output_path: str,
    ip: str,
    port: str,
):
    app = flask.Flask(__name__)

    @app.route('/help')
    def print_help():
        return flask.jsonify({'available endpoints': [
            'GET /', 
            'GET /<file>', 
            'POST /cout'
        ]})

    @app.route('/cout', methods=['POST'])
    def print_body():
        """Print posted body to stdout"""
        body = flask.request.get_data()  # binary
        print(f'---\n{body}\n')
        return ''

    @app.route('/save/<path:filename>', methods=['POST'])
    def save(filename):
        app.logger.info([k for k in flask.request.files.keys()])        
        json_obj = json.loads(flask.request.form['json'])
        app.logger.info(f"scene: {json_obj['metadata']['sceneId']}")
        if 'image' in flask.request.files:
            image = flask.request.files['image']
            image.save(os.path.join(output_path, filename))
        if 'texture' in flask.request.files:
            texture = flask.request.files['texture']
            image = np.array(Image.open(texture))
            mesh = open3d.geometry.TriangleMesh(
                vertices=open3d.utility.Vector3dVector(
                    np.array(json_obj['mesh']['vertices'])
                ),
                triangles=open3d.utility.Vector3iVector(
                    np.array(json_obj['mesh']['triangles'])
                ),
            )
            mesh.vertex_normals = open3d.utility.Vector3dVector(
                np.array(json_obj['mesh']['normals'])
            )
            mesh.texture = open3d.geometry.Image(image)
            mesh.triangle_uvs = np.array(json_obj['mesh']['triangle_uvs'])
            open3d.io.write_triangle_mesh(
                os.path.join(output_path, texture.filename), mesh
            )
        if 'mesh' in flask.request.files:
            mesh = flask.request.files['mesh']
            mesh.save(os.path.join(output_path, mesh.filename))
        return ''

    @app.route('/<path:filename>')
    def get_file(filename):
        """get static files from pwd root"""
        try:
            root = os.path.dirname(panorama_glob)
            return flask.send_from_directory(root, filename)
        except IOError:
            flask.abort(404)

    @app.route('/')
    def index():
        image_filenames = glob.glob(panorama_glob)
        images = []
        for filename in image_filenames:
            print(filename)
            im = Image.open(filename)
            w, h = im.size
            aspect = 1.0 * w / h
            if aspect > 1.0 * WIDTH / HEIGHT:
                width = min(w, WIDTH)
                height = width / aspect
            else:
                height = min(h, HEIGHT)
                width = height * aspect
            images.append({
                'width': int(width),
                'height': int(height),
                'src': f'{ip}:{port}/{os.path.basename(filename)}',
            })

        return flask.render_template_string(TEMPLATE, **{
            'images': images
        })

    return app.run(debug=True, host=ip, port=port)


if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('input_glob', type=str)
    parser.add_argument('--output_path', type=str, default='.')
    parser.add_argument('--ip', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=5000)
    args = parser.parse_args()
    sys.exit(run(args.input_glob, args.output_path, args.ip, args.port))
