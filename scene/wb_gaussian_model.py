#
# Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual
# property and proprietary rights in and to this software and related documentation.
# Any commercial use, reproduction, disclosure or distribution of this software and
# related documentation without an express license agreement from Toyota Motor Europe NV/SA
# is strictly prohibited.
#

from pathlib import Path

import numpy as np
import torch
from roma import rotmat_to_unitquat, quat_xyzw_to_wxyz

from utils.graphics_utils import compute_face_orientation
from wb_model.wb import WBModel
from .gaussian_model import GaussianModel


class WBGaussianModel(GaussianModel):
    def __init__(self, center_and_scale ,sh_degree: int):
        super().__init__(sh_degree)

        self.wb_model = WBModel(center_and_scale).cuda()
        self.num_timesteps = self.wb_model.num_timesteps
        self.min_timestep = self.wb_model.start_timestep
        self.max_timestep = self.wb_model.end_timestep

        self.verts = list(self.wb_model.timestep_to_mesh_dict.values())[0]
        self.verts_uvs = self.wb_model.verts_uvs
        self.faces = self.wb_model.faces
        self.faces_uvs = self.wb_model.faces_uvs
        self.texture = self.wb_model.texture

        self.raw_mesh_centroid = self.wb_model.raw_mesh_centroid
        self.rescale_factor = self.wb_model.rescale_factor


        # binding is initialized once the mesh topology is known
        if self.binding is None:
            self.binding = torch.arange(len(self.wb_model.faces)).cuda()
            self.binding_counter = torch.ones(len(self.wb_model.faces), dtype=torch.int32).cuda()

    def update_mesh_by_timestep(self, timestep):
        verts, verts_cano = self.wb_model(timestep)
        self.update_mesh_properties(verts, verts_cano)

    def select_mesh_by_timestep(self, timestep, original=False):
        self.timestep = timestep

        verts = self.wb_model(
            timestep=timestep,
            rotation=self.model_params['mesh_rotation'][timestep],
            scale=self.model_params['mesh_scale'][timestep],
            translation=self.model_params['mesh_translation'][timestep],
        )

        self.update_mesh_properties(verts)

    def update_mesh_properties(self, verts):
        faces = self.wb_model.faces
        triangles = verts[:, faces]

        # position
        self.face_center = triangles.mean(dim=-2).squeeze(0)

        # orientation and scale
        self.face_orien_mat, self.face_scaling = compute_face_orientation(verts.squeeze(0), faces.squeeze(0),
                                                                          return_scale=True)
        # self.face_orien_quat = matrix_to_quaternion(self.face_orien_mat)  # pytorch3d (WXYZ)
        self.face_orien_quat = quat_xyzw_to_wxyz(rotmat_to_unitquat(self.face_orien_mat))  # roma

        # for mesh rendering
        self.verts = verts
        self.faces = faces

    def compute_dynamic_offset_loss(self):
        # loss_dynamic = (self.flame_param['dynamic_offset'][[self.timestep]] - self.flame_param_orig['dynamic_offset'][[self.timestep]]).norm(dim=-1)
        loss_dynamic = self.flame_param['dynamic_offset'][[self.timestep]].norm(dim=-1)
        return loss_dynamic.mean()

    def compute_laplacian_loss(self):
        # offset = self.flame_param['static_offset'] + self.flame_param['dynamic_offset'][[self.timestep]]
        offset = self.flame_param['dynamic_offset'][[self.timestep]]
        verts_wo_offset = (self.verts_cano - offset).detach()
        verts_w_offset = verts_wo_offset + offset

        L = self.wb_model.laplacian_matrix[None, ...].detach()  # (1, V, V)
        lap_wo = L.bmm(verts_wo_offset).detach()
        lap_w = L.bmm(verts_w_offset)
        diff = (lap_wo - lap_w) ** 2
        diff = diff.sum(dim=-1, keepdim=True)
        return diff.mean()

    def load_meshes(self, train_meshes, test_meshes, tgt_train_meshes, tgt_test_meshes):
        T = self.max_timestep + 1
        print("Max timestep", T)

        # create model params to be saved for model reloading
        # if train and test frames are not continuous (have gaps) the tensor entries are 0s
        self.model_params = {
            'mesh_rotation': torch.zeros([T, 3]),
            'mesh_translation': torch.zeros([T, 3]),
            'mesh_scale': torch.ones([T, 3]),
        }

        for k, v in self.model_params.items():
            self.model_params[k] = v.float().cuda()

    def training_setup(self, training_args):
        super().training_setup(training_args)

        # rotation
        self.model_params['mesh_rotation'].requires_grad = True
        param_rotation = {'params': [self.model_params['mesh_rotation']], 'lr': training_args.flame_pose_lr,
                          "name": "mesh_rotation"}
        self.optimizer.add_param_group(param_rotation)

        # translation
        self.model_params['mesh_translation'].requires_grad = True
        param_translation = {'params': [self.model_params['mesh_translation']], 'lr': training_args.flame_trans_lr,
                             "name": "mesh_translation"}
        self.optimizer.add_param_group(param_translation)

        # scale
        self.model_params['mesh_scale'].requires_grad = True
        param_translation = {'params': [self.model_params['mesh_scale']], 'lr': training_args.flame_pose_lr,
                             "name": "mesh_scale"}
        self.optimizer.add_param_group(param_translation)

    def save_ply(self, path):
        super().save_ply(path)

        npz_path = Path(path).parent / "model_params.npz"
        params = {k: v.cpu().numpy() for k, v in self.model_params.items()}
        np.savez(str(npz_path), **params)

    def load_ply(self, path, **kwargs):
        super().load_ply(path)

        # Load rot scale trans learned in training
        npz_path = Path(path).parent / "model_params.npz"
        model_params = np.load(str(npz_path))
        model_params = {k: torch.from_numpy(v).cuda() for k, v in model_params.items()}

        self.model_params = model_params