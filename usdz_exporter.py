import numpy as np
import typing
import os
from pathlib import Path
import posixpath
from pxr import Usd, UsdGeom, Sdf, Vt, UsdShade
from PIL import Image
import open3d
from zipfile import ZipFile
import io

class UsdzExporter(object):
    def __init__(self):
        pass

    def export_usdz(self, 
        mesh:       open3d.geometry.TriangleMesh,
        mesh_name:  str,
        buffer:     io.BytesIO=None,
    ) -> bytes:
        verts = np.asarray(mesh.vertices)
        indices = np.asarray(mesh.triangles).reshape(-1)
        uvs = np.asarray(mesh.triangle_uvs)
        uvs[..., 1] = 1.0 - uvs[..., 1]
        normals = np.asarray(mesh.triangle_normals)
        stage = Usd.Stage.CreateNew(str(f'{mesh_name}.usda'))
        world = stage.DefinePrim('/World', 'Xform')
        stage.SetDefaultPrim(world)
        UsdGeom.SetStageUpAxis(stage, 'Y')
        mesh_prim = stage.DefinePrim(f'/{mesh_name}', 'Mesh')
        stage.SetDefaultPrim(stage.GetPrimAtPath(f'/{mesh_name}'))
        vset = mesh_prim.GetVariantSets().AddVariantSet("shadingVariant")
        vset.AddVariant('diffuse')
        usd_mesh = UsdGeom.Mesh.Define(stage, f'/{mesh_name}')
        time = Usd.TimeCode.Default()        
        face_vertex_counts = [3 for _ in range(len(indices) // 3)]
        usd_mesh.GetFaceVertexCountsAttr().Set(face_vertex_counts, time=time)
        usd_mesh.GetFaceVertexIndicesAttr().Set(indices.astype(np.uint64).tolist(), time=time)
        usd_mesh.GetPointsAttr().Set(Vt.Vec3fArray(verts.tolist()), time=time)
        pv = UsdGeom.PrimvarsAPI(usd_mesh.GetPrim()).CreatePrimvar(
            "st", Sdf.ValueTypeNames.Float2Array
        )
        pv.Set(uvs.tolist(), time=time)
        pv.SetIndices(Vt.IntArray(np.arange(0, len(indices)).tolist()), time=time)
        pv.SetInterpolation('faceVarying')
        usd_mesh.GetNormalsAttr().Set(normals, time=time)
        UsdGeom.PointBased(usd_mesh).SetNormalsInterpolation('faceVarying')
        vset = mesh_prim.GetVariantSets().GetVariantSet('shadingVariant')
        vset.SetVariantSelection('diffuse')
        with vset.GetVariantEditContext():
            file_path = f'{mesh_name}.usda'
            texture_dir = 'textures'
            texture_dir = Path(texture_dir).as_posix()
            if not os.path.exists(texture_dir):
                os.mkdir(texture_dir)
            """Write a USD Preview Surface material."""
            material = UsdShade.Material.Define(stage, '/Materials')
            shader = UsdShade.Shader.Define(stage, '/Materials/Shader')
            shader.CreateIdAttr('UsdPreviewSurface')
            # Create Inputs
            diffuse_input = shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f)
            diffuse_input.Set(tuple([1.0, 1.0, 1.0]), time=time)
            # Export textures and Connect textures to shader
            usd_dir = os.path.dirname(file_path)
            rel_filepath = posixpath.join(texture_dir, 'texture_diffuse.png')
            img_tensor_uint8 = np.asarray(mesh.texture)
            img = Image.fromarray(img_tensor_uint8)
            img.save(posixpath.join(usd_dir, rel_filepath))
            texture = UsdShade.Shader.Define(stage, '/Textures/diffuse_texture')
            texture.CreateIdAttr('UsdUVTexture')
            input = texture.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(rel_filepath, time=time)
            channels = ['r', 'b', 'g', 'a']
            out = texture.CreateOutput('rgb', Sdf.ValueTypeNames.Float3)
            diffuse_input.ConnectToSource(out)
            # create Usd Preview Surface Shader outputs
            shader.CreateOutput('surface', Sdf.ValueTypeNames.Token)
            shader.CreateOutput('displacement', Sdf.ValueTypeNames.Token)
            # create material
            material.CreateSurfaceOutput().ConnectToSource(shader.GetOutput('surface'))
            material.CreateDisplacementOutput().ConnectToSource(shader.GetOutput('displacement'))            
            binding_api = UsdShade.MaterialBindingAPI(mesh_prim)
            binding_api.Bind(UsdShade.Material(material))
        stage.Save()
        if buffer is not None:
            with ZipFile(buffer, 'w') as zf:
                zf.write(f'{mesh_name}.usda')
                zf.write(rel_filepath)
            buffer.seek(0)
        return buffer
