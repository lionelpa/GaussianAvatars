from pathlib import Path

from torch import nn
from PIL import Image
import numpy as np

from utils.pytorch3d_load_obj import load_obj

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LS7_MESH_PATH_LIONEL = PROJECT_ROOT / "ls7_model/assets/final_fit_invtrans_bodyscan.obj"
LS7_TEXTURE_PATH_LIONEL = PROJECT_ROOT / "ls7_model/assets/skin_basecolor_inv.png"



def load_mesh(mesh_path):
    verts, faces, aux = load_obj(mesh_path, load_textures=False)

    uvs = aux.verts_uvs

    faces_uvs = faces.textures_idx
    faces = faces.verts_idx

    return verts, uvs, faces, faces_uvs

def load_texture(texture_path):
    texture = Image.open(texture_path)
    texture.convert('RGB')
    texture_np = np.array(texture)

    return texture_np


class LS7Model(nn.Module):
    def __init__(self, ls7_mesh_path=LS7_MESH_PATH_LIONEL):
        super().__init__()

        self.verts, self.verts_uvs, self.faces, self.faces_uvs = load_mesh(ls7_mesh_path)
        self.texture = load_texture(LS7_TEXTURE_PATH_LIONEL)

        assert self.texture.shape[0] == 4096
        assert self.texture.shape[1] == 4096

        assert self.texture[10, 26][0] in [228, 200, 180]
        assert self.texture[10, 26][1] in [228,200,180]
        assert self.texture[10, 26][2] in [228,200,180]

        assert self.texture[11, 26][0] in [168, 188, 199]
        assert self.texture[11, 26][1] in [168, 188, 199]
        assert self.texture[11, 26][2] in [168, 188, 199]

        assert self.texture[1605, 26][0] in [228,200,180]
        assert self.texture[1605, 26][1] in [228,200,180]
        assert self.texture[1605, 26][2] in [228,200,180]

        assert self.texture[1606, 26][0] in [151,146,153]
        assert self.texture[1606, 26][1] in [151,146,153]
        assert self.texture[1606, 26][2] in [151,146,153]
