# from .lbs import lbs, vertices2landmarks, blend_shapes, vertices2joints
import os
import re

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from PIL import Image

try:
    from pytorch3d.io import load_obj
except ImportError:
    from utils.pytorch3d_load_obj import load_obj

WB_HEAD_BASE_MESH_PATH = "wb_model/assets/0_head_foundational.obj"
WB_EYES_BASE_MESH_PATH = "wb_model/assets/0_eyes_foundational.obj"
WB_MESHES_PATH = "wb_model/assets/"
WB_HEAD_MESHES_NAME_FILTER_PATTERN="([0-9]+)_head\.obj"
WB_EYES_MESHES_NAME_FILTER_PATTERN="([0-9]+)_eyes\.obj"
WB_TEXTURE_PATH = "wb_model/assets/skin_basecolor.png"


def load_texture(texture_path):
    texture = Image.open(texture_path)
    texture.convert('RGB')
    texture_np = np.array(texture)

    return texture_np

class WBModel(nn.Module):



    def __init__(
            self,
            wb_head_base_mesh_path=WB_HEAD_BASE_MESH_PATH,
            wb_eyes_base_mesh_path=WB_EYES_BASE_MESH_PATH,
            wb_meshes_path=WB_MESHES_PATH,
            wb_head_meshes_name_filter_pattern=WB_HEAD_MESHES_NAME_FILTER_PATTERN,
            wb_eyes_meshes_name_filter_pattern=WB_EYES_MESHES_NAME_FILTER_PATTERN,
            wb_texture_path=WB_TEXTURE_PATH
    ):
        """
        Initializes the class with paths and filters for loading meshes.

        Args:
            wb_base_mesh_path (str): Path to the base mesh file.
            wb_meshes_path (str): Path to the directory containing the mesh files.
            wb_meshes_name_filter_pattern (str): Regex pattern to filter out wanted mesh files where first regex group locates timestep number.
        """
        super(WBModel, self).__init__()

        # Get face info from base meshes. Faces do not change throughout training and can be buffered
        head_verts, head_faces, head_aux = load_obj(wb_head_base_mesh_path, load_textures=False)
        _, eyes_faces, eyes_aux = load_obj(wb_eyes_base_mesh_path, load_textures=False)        

        # stack eyes faces under head faces and adjust indices
        faces = torch.vstack((head_faces.verts_idx, eyes_faces.verts_idx + head_verts.shape[0]))
        self.register_buffer("faces", faces, persistent=False)

        self.texture = load_texture(wb_texture_path)

        self.verts_uvs = torch.vstack((head_aux.verts_uvs, eyes_aux.verts_uvs))
        self.faces_uvs = torch.vstack((head_faces.textures_idx, eyes_faces.textures_idx))


        # Load each head.obj 
        head_file_pattern = re.compile(wb_head_meshes_name_filter_pattern)
        head_meshes = self.load_timestep2mesh_dict(wb_meshes_path, head_file_pattern, "heads")

        # Load each eyes.obj 
        eyes_file_pattern = re.compile(wb_eyes_meshes_name_filter_pattern)
        eyes_meshes = self.load_timestep2mesh_dict(wb_meshes_path, eyes_file_pattern, "eyes")
        
        # Merge dicts 
        for t, verts in tqdm(head_meshes.items(), desc="Merging meshes...", unit="merges"):         
            full_verts = torch.vstack((verts, eyes_meshes[t]))
            # move all mesh vert tensors to gpu
            self.timestep_to_mesh_dict[t] = full_verts.unsqueeze(0).float().cuda()

        self.num_timesteps = len(self.timestep_to_mesh_dict.values())
        self.start_timestep = min(self.timestep_to_mesh_dict.keys())
        self.end_timestep = max(self.timestep_to_mesh_dict.keys())

    def load_timestep2mesh_dict(self, meshes_path, pattern, unit=""):
        mesh_dict = {}
        mesh_file_names = sorted([f for f in os.listdir(meshes_path) if pattern.match(f)])

        # timestep -> mesh vertices
        self.timestep_to_mesh_dict = {}
        for mesh_file in tqdm(sorted(mesh_file_names, key= lambda x: int(pattern.match(x).group(1))), desc=f"Loading {unit}...", unit=f"{unit}"):
            full_path = os.path.join(meshes_path, mesh_file)

            match = pattern.match(mesh_file)
            timestep = int(match.group(1)) if match else None

            verts, _, _ = load_obj(full_path, load_textures=False)
            mesh_dict[timestep] = verts
        return mesh_dict

    def forward(self, timestep):
        return self.timestep_to_mesh_dict[timestep]


if __name__ == '__main__':
    wb_model = WBModel()
