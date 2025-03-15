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

        # needed for camera repositioning
        self.center_and_scale = center_and_scale
        self.raw_mesh_centroid = self.wb_model.raw_mesh_centroid
        self.rescale_factor = self.wb_model.rescale_factor

        self.verts = None
        self.verts_uvs = self.wb_model.verts_uvs
        self.faces = self.wb_model.faces
        self.faces_uvs = self.wb_model.faces_uvs
        self.texture = self.wb_model.texture

        # self.min_timestep = 4
        # self.max_timestep = 499

        # binding is initialized once the mesh topology is known
        if self.binding is None:
            self.binding = torch.arange(len(self.wb_model.faces)).cuda()
            self.binding_counter = torch.ones(len(self.wb_model.faces), dtype=torch.int32).cuda()


    def select_mesh_by_timestep(self, timestep, original=False):
        self.timestep = timestep

        verts = self.wb_model(
            timestep=timestep,
            rotation=self.model_params['mesh_rotation'][timestep],
            translation=self.model_params['mesh_translation'][timestep],
            scale=self.model_params['mesh_scale'][timestep],
            blendshape_weights=self.model_params['bs_weights'][timestep],
            mean=self.model_params['mesh_mean'][timestep],
            static_offset=self.model_params['static_offset'],
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


    def load_meshes(self, meshes):
        # needed for viewer gui
        self.num_timesteps = len(meshes)
        self.min_timestep = torch.min(torch.tensor([int(k) for k in meshes.keys()]))
        self.max_timestep = torch.max(torch.tensor([int(k) for k in meshes.keys()]))

        T = self.max_timestep + 1
        print("Max timestep", T)

        # create model params to be saved for model reloading
        # if train and test frames are not continuous (have gaps) the tensor entries are 0s
        self.model_params = {
            'center_and_scale': torch.tensor(int(self.center_and_scale)),
            'mesh_rotation': torch.zeros([T, 3]),
            'mesh_translation': torch.zeros([T, 3]),
            'mesh_scale': torch.ones([T, 3]),
            'bs_weights': torch.zeros([T, list(meshes.values())[0]['bs_weights'].shape[0]]),
            'mesh_mean': torch.ones([T, 3]),
            'static_offset': torch.zeros_like(self.wb_model.verts),
        }

        for timestep, mesh in meshes.items():
            self.model_params['mesh_rotation'][timestep] = mesh['rotation'].clone()
            self.model_params['mesh_translation'][timestep] = mesh['translation'].clone()
            self.model_params['mesh_scale'][timestep] = mesh['scale'].clone()
            self.model_params['bs_weights'][timestep] = mesh['bs_weights'].clone()
            self.model_params['mesh_mean'][timestep] = mesh['mean'].clone()

        for k, v in self.model_params.items():
            self.model_params[k] = v.float().cuda()

    def training_setup(self, training_args):
        super().training_setup(training_args)

        # rotation
        self.model_params['mesh_rotation'].requires_grad = True
        param_rotation = {'params': [self.model_params['mesh_rotation']], 'lr': training_args.flame_pose_lr, "name": "mesh_rotation"}
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

        # expression
        self.model_params['bs_weights'].requires_grad = True
        param_bs_weights = {'params': [self.model_params['bs_weights']], 'lr': training_args.flame_expr_lr, "name": "bs_weights"}
        self.optimizer.add_param_group(param_bs_weights)

        # static_offset
        self.model_params['static_offset'].requires_grad = True
        param_static_offset = {'params': [self.model_params['static_offset']], 'lr': 1e-6, "name": "static_offset"}
        self.optimizer.add_param_group(param_static_offset)

    def save_ply(self, path):
        super().save_ply(path)

        npz_path = Path(path).parent / "model_params.npz"
        params = {k: v.cpu().numpy() for k, v in self.model_params.items()}
        params["num_timesteps"] = np.array(self.num_timesteps)
        params["min_timestep"] = np.array(self.min_timestep)
        params["max_timestep"] = np.array(self.max_timestep)

        np.savez(str(npz_path), **params)

    def load_ply(self, path, **kwargs):
        super().load_ply(path)

        # Load rot scale trans learned in training
        npz_path = Path(path).parent / "model_params.npz"
        params = np.load(str(npz_path))
        self.model_params = {k: torch.from_numpy(v).cuda() for k, v in params.items()
                        if k not in ["num_timesteps", "min_timestep", "max_timestep"]}
        self.num_timesteps = int(params['num_timesteps'])
        self.min_timestep = int(params['min_timestep'])
        self.max_timestep = int(params['max_timestep'])