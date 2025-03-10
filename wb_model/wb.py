# from .lbs import lbs, vertices2landmarks, blend_shapes, vertices2joints
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from utils.pytorch3d import euler_angles_to_matrix

try:
    from pytorch3d.io import load_obj
except ImportError:
    from utils.pytorch3d_load_obj import load_obj, save_obj

ROOT = "/home/lio/PycharmProjects/data/scanner_wb/video"
# WB_HEAD_BASE_MESH_PATH     = ROOT + "/meshes_weights/0_head_nicolas_neutral.obj"
# WB_EYES_BASE_MESH_PATH     = ROOT + "/meshes_weights/0_eyes_nicolas_neutral.obj"
WB_HEAD_BASE_MESH_PATH     = ROOT + "/smooth_new/0_head_nicolas_neutral.obj"
WB_EYES_BASE_MESH_PATH     = ROOT + "/smooth_new/0_eyes_nicolas_neutral.obj"
WB_MESH_FOR_CENTERING_PATH = ROOT + "/smooth_new/4_head.obj"
WB_BLENDSHAPES_PATH        = ROOT + "/bs" #"wb_model/assets/"

WB_FRAME_PARAMS_PATH       = "([0-9]+)_head\.obj"
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
            wb_texture_path=WB_TEXTURE_PATH,
            target_height=TARGET_HEIGHT,
            wb_blendshapes_path=WB_BLENDSHAPES_PATH,
    ):

        super(WBModel, self).__init__()

        ### VERTS & FACES - TEMPLATE ###
        # Get face info from base meshes. Faces do not change throughout training and can be buffered
        neutral_head_verts, neutral_head_faces, neutral_head_aux = load_obj(wb_head_base_mesh_path, load_textures=False)
        neutral_eyes_verts, neutral_eyes_faces, neutral_eyes_aux = load_obj(wb_eyes_base_mesh_path, load_textures=False)
        self.register_buffer("head_neutral_v", neutral_head_verts, persistent=False)
        self.register_buffer("eyes_neutral_v", neutral_eyes_verts, persistent=False)
        self.register_buffer("full_neutral_v", torch.vstack([neutral_head_verts, neutral_eyes_verts]), persistent=False)

        # stack eyes faces under head faces and adjust indices
        faces = torch.vstack((neutral_head_faces.verts_idx, neutral_eyes_faces.verts_idx + neutral_head_verts.shape[0]))
        self.register_buffer("verts", torch.vstack((neutral_head_verts, neutral_eyes_verts)), persistent=False)
        self.register_buffer("faces", faces, persistent=False)

        ### VERTS - BLENDSHAPES ###
        shapes, self.shapes_names = self.load_delta_blendshapes(neutral_head_verts, wb_blendshapes_path)
        self.register_buffer("shapes", shapes, persistent=False)

        ### TEXTURE ###
        self.texture = load_texture(wb_texture_path)
        self.verts_uvs = torch.vstack((neutral_head_aux.verts_uvs, neutral_eyes_aux.verts_uvs))
        self.faces_uvs = torch.vstack((neutral_head_faces.textures_idx, neutral_eyes_faces.textures_idx))

        self.raw_mesh_centroid = torch.zeros(3)
        self.rescale_factor = 1

    def load_delta_blendshapes(self, neutral, blendshapes_path, blendshape_order_file="blendshapes_order.txt"):
        shapes = []
        shapes_names = []
        with open(os.path.join(blendshapes_path, blendshape_order_file)) as f:
            for i, blendshape_name in enumerate(f.readlines()):
                blendshape_file_name = blendshape_name.strip() + ".obj"
                blendshape_file_path = os.path.join(blendshapes_path, blendshape_file_name)
                try:
                    shape_verts, _, _ = load_obj(blendshape_file_path, load_textures=False)
                    if neutral is not None:
                        shape_verts = shape_verts - neutral
                    shapes.append(shape_verts.unsqueeze(0))
                    shapes_names.append(blendshape_name.strip())
                except Exception as e:
                    assert False, f"Could not read {blendshape_file_path}\n{e}"
        return torch.vstack(shapes), shapes_names

    def load_blendshapes(self, neutral, blendshapes_path, blendshape_order_file="blendshapes_order.txt"):
        return self.load_delta_blendshapes(None, blendshapes_path, blendshape_order_file)

    # def extend_blend_shape_verts_by_eyes(self, blendshapes, neutral_eyes_verts):
    #     # blendshapes        = B x Vb x 3
    #     # neutral_eyes_verts =     Ve x 3
    #     eyes_expanded = neutral_eyes_verts.unsqueeze(0).repeat(blendshapes.shape[0], 1, 1)  # (B, Ve, 3)
    #     extended_blendshapes = torch.cat([blendshapes, eyes_expanded], dim=1)
    #     return  extended_blendshapes

    def forward(self, translation, rotation, scale, blendshape_weights, timestep, mean):
        # apply only to head
        w_0 = blendshape_weights.unsqueeze(0)
        t_0 = translation.unsqueeze(0)
        R_0 = rotation.unsqueeze(0)
        s_0 = scale.unsqueeze(0)
        bs = self.shapes.unsqueeze(0)

        # Transformations applied to reconstruct face and align with cameras are done on an already centered
        # neutral expression. Hence, we need to deduct the mean before transforming.
        neutral_mean = self.head_neutral_v.mean(dim=0)
        neutral = (self.head_neutral_v - neutral_mean).unsqueeze(0)

        w_0 = w_0.reshape(*(w_0.shape), 1, 1)
        weighted_bs = (w_0 * bs).sum(1) + neutral

        mean = mean.unsqueeze(0)
        matrix_R = euler_angles_to_matrix(R_0, "XYZ")

        rotated_pred = (torch.bmm(s_0 * (weighted_bs - mean), matrix_R)) + mean + t_0

        save_obj(f"./output/A_{timestep}_maxim.obj", verts=rotated_pred[0], faces=self.faces)
        # raise Exception("MÖP")
        # print("final",rotated_pred.shape)
        # print("final[0]",rotated_pred[0].shape)
        return rotated_pred



if __name__ == '__main__':
    wb_model = WBModel(False)
