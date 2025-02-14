from pathlib import Path

import numpy as np
from PIL import Image
from torch import nn

from utils.pytorch3d_load_obj import load_obj

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LS7_MESH_PATH_LIONEL = PROJECT_ROOT / "ls7_model/assets/handy_pipeline/head/baked_mesh_reduced.obj"
#LS7_MESH_PATH_LIONEL = PROJECT_ROOT / "ls7_model/assets/final_fit_invtrans_bodyscan.obj"
LS7_TEXTURE_PATH_LIONEL = PROJECT_ROOT / "ls7_model/assets/handy_pipeline/head/baked_mesh_tex0.png"
# LS7_TEXTURE_PATH_LIONEL = PROJECT_ROOT / "ls7_model/assets/skin_basecolor_inv.png"



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
