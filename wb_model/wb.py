# from .lbs import lbs, vertices2landmarks, blend_shapes, vertices2joints
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch import einsum

import utils.pytorch3d

try:
    from pytorch3d.io import load_obj
except ImportError:
    from utils.pytorch3d_load_obj import load_obj

ROOT = "/home/lio/PycharmProjects/data/scanner_wb/video"
WB_HEAD_BASE_MESH_PATH     = ROOT + "/meshes_weights/0_head_nicolas_neutral.obj"
WB_EYES_BASE_MESH_PATH     = ROOT + "/meshes_weights/0_eyes_nicolas_neutral.obj"
WB_MESH_FOR_CENTERING_PATH = ROOT + "/meshes_weights/4_head.obj"
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
        self.register_buffer("v_neutral", neutral_head_verts, persistent=False)

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
                    shapes_names.append(blendshape_name)
                except Exception as e:
                    assert False, f"Could not read {blendshape_file_path}\n{e}"
        return torch.vstack(shapes), shapes_names

    def load_blendshapes(self, neutral, blendshapes_path, blendshape_order_file="blendshapes_order.txt"):
        return self.load_delta_blendshapes(None, blendshapes_path, blendshape_order_file)

    def forward(self, translation, rotation, scale, blendshape_weights):
        # apply blendshapes to head
        print("=========== FORWARD =============================")
        print(">>> params:")
        print("translation", translation.shape)
        print("rotation", rotation.shape)
        print("scale", scale.shape)
        print("blendshape_weights", blendshape_weights.shape)
        print("===")
        print("self.v_neutral:", self.v_neutral.shape)
        print("self.shapes:", self.shapes.shape)

        # DBS = DELTA_BLENDSHAPES
        DBS = self.v_neutral + einsum("w,wvc->vc", blendshape_weights, self.shapes)
        print("DBS", DBS.shape)

        # rotation und translation durchführen
        # (LBS - mean(LBS)) * R + mean(LBS) + t
        # mit scaling? (LBS - mean(LBS)) * S * R + mean(LBS) + t
        c = torch.mean(DBS, dim=0)
        print("c", c)
        centered = (DBS - c)
        print("centered", centered.shape)
        scaled = centered * scale
        print("scaled", scaled.shape)

        rot_mat = utils.pytorch3d.euler_angles_to_matrix(rotation, convention="XYZ")
        print("rot_mat", rot_mat.shape)
        rotated = scaled @ rot_mat
        print("rotated", rotated.shape)

        # für augen und head

        # kopf und augen zusammenfügen

        raise Exception("JAA")
        return



if __name__ == '__main__':
    wb_model = WBModel(False)
