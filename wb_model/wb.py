# from .lbs import lbs, vertices2landmarks, blend_shapes, vertices2joints
import os
import re

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

from utils.general_utils import print_triangle_area_info
from utils.pytorch3d import euler_angles_to_matrix
from utils.pytorch3d_load_obj import load_obj

WB_HEAD_BASE_MESH_PATH = "wb_model/assets/0_head_foundational.obj"
WB_EYES_BASE_MESH_PATH = "wb_model/assets/0_eyes_foundational.obj"
WB_MESH_FOR_CENTERING_PATH = "wb_model/assets/4_head.obj"
WB_MESHES_PATH = "wb_model/assets/"
WB_HEAD_MESHES_NAME_FILTER_PATTERN="([0-9]+)_head\.obj"
WB_EYES_MESHES_NAME_FILTER_PATTERN="([0-9]+)_eyes\.obj"
WB_TEXTURE_PATH = "wb_model/assets/skin_basecolor.png"
TARGET_HEIGHT = 0.34316921 # determined from flame base model height, used to rescale wb mesh




def load_texture(texture_path):
    texture = Image.open(texture_path)
    texture.convert('RGB')
    texture_np = np.array(texture)

    return texture_np

class WBModel(nn.Module):



    def __init__(
            self,
            center_and_scale,
            wb_head_base_mesh_path=WB_HEAD_BASE_MESH_PATH,
            wb_eyes_base_mesh_path=WB_EYES_BASE_MESH_PATH,
            wb_mesh_for_centering_path=WB_MESH_FOR_CENTERING_PATH,
            wb_meshes_path=WB_MESHES_PATH,
            wb_head_meshes_name_filter_pattern=WB_HEAD_MESHES_NAME_FILTER_PATTERN,
            wb_eyes_meshes_name_filter_pattern=WB_EYES_MESHES_NAME_FILTER_PATTERN,
            wb_texture_path=WB_TEXTURE_PATH,
            target_height=TARGET_HEIGHT
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
        self.n_head_verts = head_verts.shape[0]
        _, eyes_faces, eyes_aux = load_obj(wb_eyes_base_mesh_path, load_textures=False) 
        
        # for mesh centering and rescaling to approx setup like niessner 
        ## must not be the template because it is positioned differently from all n_head.objs which are used during training.
        ## thus its vertices are useless here
        vs, _, _ = load_obj(wb_mesh_for_centering_path, load_textures=False)
        
        self.raw_mesh_centroid = torch.zeros(3)
        self.rescale_factor = 1
        if center_and_scale:
            print("Calculating vector for centering and rescaling...")
            self.raw_mesh_centroid = torch.mean(vs, axis=0, keepdims=False)
            ys = vs[:,1]
            min_y = float(torch.min(ys))
            max_y = float(torch.max(ys))
            height = max_y - min_y
            self.rescale_factor = target_height / height    
        # self.rescale_factor = 1
        print("self.raw_mesh_centroid", self.raw_mesh_centroid)
        print("self.rescale_factor", self.rescale_factor)


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
        
        # Merge dicts (eyes and heads)
        for t, verts in tqdm(head_meshes.items(), desc="Merging meshes...", unit="merges"):         
            full_verts = torch.vstack((verts, eyes_meshes[t]))
            # center
            full_verts = full_verts - self.raw_mesh_centroid
            # rescale
            full_verts = self.rescale_factor * full_verts 
            # move all mesh vert tensors to gpu
            self.timestep_to_mesh_dict[t] = full_verts.float().cuda()


        print_triangle_area_info(self.timestep_to_mesh_dict[4].cpu().squeeze(), faces.cpu())
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
            break
        return mesh_dict

    def forward(self, timestep, rotation, scale, translation):
        '''
            rotation, scale, translation are all tensor 3
        '''
        # print("timestep", timestep)
        # print("scale", scale)
        # print("translation", translation)
        # print("rotation", rotation)

        v = self.timestep_to_mesh_dict[timestep]
        # mean only on head not on head+eyes
        mean = torch.mean(v[:self.n_head_verts], dim=0)

        R = euler_angles_to_matrix(rotation, convention="XYZ")
        # print("R", R)
        # print("mean", mean)
        # print("verts", v.shape)
        # print("verts-mean",(v - mean).shape)
        # print("scale* verts-mean",(scale * (v - mean)).shape)
        # print((scale * (verts - mean)).shape)
        # print(R.unsqueeze(0).shape)
        # transformed = torch.bmm((scale * (verts - mean)), R.unsqueeze(0))[0] + mean + translation

        # save_obj("./output/###1start.obj", v, self.faces)
        # centered = v - mean
        # save_obj("./output/###2centered.obj", centered, self.faces)
        # scaled = scale * centered
        # save_obj("./output/###3scaled.obj", scaled, self.faces)
        # rotated = scaled @ R
        # save_obj("./output/###4rotated.obj",rotated, self.faces)
        # recentered = rotated + mean
        # save_obj("./output/###5recentered.obj",recentered, self.faces)
        # final = recentered + translation
        # save_obj("./output/###6final.obj",final, self.faces)

        transformed = (scale * (v - mean)) @ R + mean + translation
        # save_obj("./output/###.obj", transformed, self.faces)
        return transformed


if __name__ == '__main__':
    wb_model = WBModel()
