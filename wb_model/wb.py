# from .lbs import lbs, vertices2landmarks, blend_shapes, vertices2joints
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from utils.pytorch3d import euler_angles_to_matrix
from utils.pytorch3d_load_obj import load_obj

ROOT = "/home/lionel.azevedo/data/wb_scanner" # "/home/lio/PycharmProjects/data/scanner_wb/video"
WB_HEAD_BASE_MESH_PATH     = ROOT + "/smooth2/0_head_nicolas_neutral.obj"
WB_EYES_BASE_MESH_PATH     = ROOT + "/smooth2/0_eyes_nicolas_neutral.obj"
WB_BLENDSHAPES_PATH        = ROOT + "/bs" #"wb_model/assets/bs"
WB_TEXTURE_PATH = "wb_model/assets/skin_basecolor.png"

# needed for adjustment of pos and scale to those of FLAME
TARGET_HEIGHT = 0.34316921 # determined from flame base model height, used to rescale wb mesh
WB_MESH_FOR_CENTERING_PATH = ROOT + "/smooth2/4_head.obj"



def load_texture(texture_path):
    texture = Image.open(texture_path)
    texture.convert('RGB')
    texture_np = np.array(texture)

    return texture_np


def assertTexture(texture):
    # check random points to match color value
    assert (texture[100, 100] == np.array([44,96,149])).all()
    assert (texture[200, 3800] == np.array([56,51,29])).all()
    assert (texture[3900, 300] == np.array([107,94,75])).all()
    assert (texture[3900, 3500] == np.array([165,130,114])).all()


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
        assertTexture(self.texture)
        self.verts_uvs = torch.vstack((neutral_head_aux.verts_uvs, neutral_eyes_aux.verts_uvs))
        self.faces_uvs = torch.vstack((neutral_head_faces.textures_idx, neutral_eyes_faces.textures_idx + neutral_head_aux.verts_uvs.shape[0]))

        self.raw_mesh_centroid = torch.zeros(3)
        self.rescale_factor = 1

        ### CENTERING PARAMS NEEDED TO ADJUST TO FLAME ###
        vs, _, _ = load_obj(wb_mesh_for_centering_path, load_textures=False)
        self.raw_mesh_centroid = torch.zeros(3).cuda()
        self.rescale_factor = torch.tensor(1).cuda()
        if center_and_scale:
            print("Calculating vector for centering and rescaling...")
            self.raw_mesh_centroid = torch.mean(vs, dim=0).cuda()
            ys = vs[:, 1]
            min_y = float(torch.min(ys))
            max_y = float(torch.max(ys))
            height = max_y - min_y
            self.rescale_factor = torch.tensor(target_height / height).cuda()
        print("self.raw_mesh_centroid", self.raw_mesh_centroid)
        print("self.rescale_factor", self.rescale_factor)


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

    def forward(self, translation, rotation, scale, blendshape_weights, timestep, mean, static_offset, dynamic_offset):
        # apply only to head
        w_0 = blendshape_weights.unsqueeze(0)
        t_0 = translation.unsqueeze(0)
        R_0 = rotation.unsqueeze(0)
        s_0 = scale.unsqueeze(0)
        bs = self.shapes.unsqueeze(0)
        static_offset = static_offset.unsqueeze(0)
        dynamic_offset = dynamic_offset.unsqueeze(0)
        mean = mean.unsqueeze(0)

        # Transformations applied to reconstruct face and align with cameras are done on an already centered
        # neutral expression. Hence, we need to deduct the mean before transforming.
        # clarification: center using computed mean -> later use passed mean param for transform
        neutral_head_mean = self.head_neutral_v.mean(dim=0)
        head_neutral_centered_v = (self.head_neutral_v - neutral_head_mean).unsqueeze(0)
        eyes_neutral_centered_v = (self.eyes_neutral_v - neutral_head_mean).unsqueeze(0)

        # apply blendshapes
        w_0 = w_0.reshape(*(w_0.shape), 1, 1)
        weighted_bs = (w_0 * bs).sum(1) + head_neutral_centered_v

        # combine with eyes
        verts_full_cano = torch.cat([weighted_bs, eyes_neutral_centered_v], dim=1)

        # apply static offset
        verts_full_with_offset = verts_full_cano + static_offset

        # apply dynamic offset based on bs
        # dyn_offset = 1 x B x V x 3
        if dynamic_offset is not None:
            dyn = (w_0 * dynamic_offset).sum(1)
            verts_full_with_offset = verts_full_with_offset + dyn

        # transform
        matrix_R = euler_angles_to_matrix(R_0, "XYZ")
        rotated_pred_cano = (torch.bmm(s_0 * (verts_full_cano - mean), matrix_R)) + mean + t_0
        rotated_pred = (torch.bmm(s_0 * (verts_full_with_offset - mean), matrix_R)) + mean + t_0

        # center and scale
        centroid = self.raw_mesh_centroid.unsqueeze(0)
        rotated_pred_cano = self.rescale_factor * (rotated_pred_cano - centroid)
        rotated_pred = self.rescale_factor * (rotated_pred - centroid)
        # save_obj(f"./output/A_{timestep}_maxim.obj", verts=rotated_pred[0], faces=self.faces)

        return rotated_pred, rotated_pred_cano


if __name__ == '__main__':
    wb_model = WBModel(False)
