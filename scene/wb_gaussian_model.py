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
from pytorch3d.structures import Meshes
from roma import rotmat_to_unitquat, quat_xyzw_to_wxyz

from utils.graphics_utils import compute_face_orientation
from utils.pytorch3d import mesh_laplacian_smoothing_per_vertex
from wb_model.wb import WBModel
from .gaussian_model import GaussianModel
from utils.general_utils import inverse_sigmoid, get_expon_lr_func


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

        verts, verts_canonical = self.wb_model(
            timestep=timestep,
            rotation=self.model_params['mesh_rotation'][timestep],
            translation=self.model_params['mesh_translation'][timestep],
            scale=self.model_params['mesh_scale'][timestep],
            global_scale=self.model_params['global_mesh_scale'],
            blendshape_weights=self.model_params['bs_weights'][timestep] + self.model_params['add_bs_weights'][timestep],
            mean=self.model_params['mesh_mean'][timestep],
            static_offset=self.model_params['static_offset'],
            dynamic_offset=self.model_params['dynamic_offset'],
        )
        
        self.update_mesh_properties(verts, verts_canonical)

    def update_mesh_properties(self, verts, verts_cano):
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

        # for mesh regularization (e.g. laplacian loss)
        self.verts_cano = verts_cano


    def load_meshes(self, meshes):
        # needed for viewer gui
        self.num_timesteps = len(meshes)
        self.min_timestep = torch.min(torch.tensor([int(k) for k in meshes.keys()]))
        self.max_timestep = torch.max(torch.tensor([int(k) for k in meshes.keys()]))

        T = self.max_timestep + 1
        print("Max timestep", T)

        num_verts = self.wb_model.verts.shape[0]

        # create model params to be saved for model reloading
        # if train and test frames are not continuous (have gaps) the tensor entries are 0s
        self.model_params = {
            'mesh_translation': torch.zeros([T, 3]),
            'mesh_rotation': torch.zeros([T, 3]),
            'mesh_scale': torch.ones([T, 3]), # loaded per frame, not learned
            'bs_weights': torch.zeros([T, list(meshes.values())[0]['bs_weights'].shape[0]]), # loaded
            'add_bs_weights': torch.zeros([T, list(meshes.values())[0]['bs_weights'].shape[0]]), # learned
            'mesh_mean': torch.ones([T, 3]),
            'center_and_scale': torch.tensor(int(self.center_and_scale)),
            'global_mesh_scale': torch.ones([3]), # learned scale (global! not per frame)
            'static_offset': torch.zeros_like(self.wb_model.verts),
            'dynamic_offset': torch.zeros(list(meshes.values())[0]['bs_weights'].shape[0], num_verts, 3),
        }

        for timestep, mesh in meshes.items():
            self.model_params['mesh_rotation'][timestep] = mesh['rotation'].clone()
            self.model_params['mesh_translation'][timestep] = mesh['translation'].clone()
            self.model_params['mesh_scale'][timestep] = mesh['scale'].clone()
            self.model_params['bs_weights'][timestep] = mesh['bs_weights'].clone()
            self.model_params['mesh_mean'][timestep] = mesh['mean'].clone()

        for k, v in self.model_params.items():
            self.model_params[k] = v.float().cuda()
        
        self._add_bs_weights_original = self.model_params["add_bs_weights"].clone()
        self._static_offset_original = self.model_params["static_offset"].clone()


    def training_setup(self, training_args):
        super().training_setup(training_args)
        trans_lr = training_args.trans_lr
        rot_lr = training_args.rot_lr
        scale_lr = training_args.scale_lr
        bs_lr = training_args.bs_lr
        static_offset_lr = training_args.static_offset_lr
        dynamic_offset_lr = training_args.dynamic_offset_lr

        # translation
        self.model_params['mesh_translation'].requires_grad = True
        param_translation = {'params': [self.model_params['mesh_translation']], 'lr': trans_lr,
                             "name": "mesh_translation"}
        self.optimizer.add_param_group(param_translation)

        # rotation
        self.model_params['mesh_rotation'].requires_grad = True
        param_rotation = {'params': [self.model_params['mesh_rotation']], 'lr': rot_lr, "name": "mesh_rotation"}
        self.optimizer.add_param_group(param_rotation)

        # global scale (not per frame)
        self.model_params['global_mesh_scale'].requires_grad = True
        param_global_scale = {'params': [self.model_params['global_mesh_scale']], 'lr': scale_lr,
                             "name": "global_mesh_scale"}
        self.optimizer.add_param_group(param_global_scale)

        # expression learned
        self.model_params['add_bs_weights'].requires_grad = True
        param_bs_weights = {'params': [self.model_params['add_bs_weights']], 'lr': bs_lr, "name": "add_bs_weights"}
        self.optimizer.add_param_group(param_bs_weights)

        # static_offset
        self.model_params['static_offset'].requires_grad = True
        param_static_offset = {'params': [self.model_params['static_offset']], 'lr': static_offset_lr, "name": "static_offset"}
        self.optimizer.add_param_group(param_static_offset)

        # dynamic_offset
        self.model_params['dynamic_offset'].requires_grad = True
        param_dynamic_offset = {'params': [self.model_params['dynamic_offset']], 'lr': dynamic_offset_lr, "name": "dynamic_offset"}
        self.optimizer.add_param_group(param_dynamic_offset)

        self.trans_scheduler_args = get_expon_lr_func(lr_init=trans_lr,
                                                    lr_final= training_args.flame_trans_lr,
                                                    max_steps=training_args.reposition_until)
        self.rot_scheduler_args = get_expon_lr_func(lr_init=rot_lr,
                                                    lr_final= training_args.flame_pose_lr,
                                                    max_steps=training_args.reposition_until)
        self.scale_scheduler_args = get_expon_lr_func(lr_init=scale_lr,
                                                    lr_final= training_args.flame_pose_lr,
                                                    max_steps=training_args.reposition_until)
        self.bs_scheduler_args = get_expon_lr_func(lr_init=training_args.bs_lr,
                                                    lr_final= training_args.bs_lr,
                                                    max_steps=training_args.reposition_until,
                                                    lr_delay_steps=5000, lr_delay_mult=0)
                                                    

    def update_learning_rate(self, iteration):
        super().update_learning_rate(iteration)
        ''' Learning rate scheduling per step '''
        
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "mesh_translation":
                lr = self.trans_scheduler_args(iteration)
                param_group['lr'] = lr
            elif param_group["name"] == "mesh_rotation":
                lr = self.rot_scheduler_args(iteration)
                param_group['lr'] = lr
            elif param_group["name"] == "global_mesh_scale":
                lr = self.scale_scheduler_args(iteration)
                param_group['lr'] = lr
            elif param_group["name"] == "add_bs_weights":
                lr = self.bs_scheduler_args(iteration)
                param_group['lr'] = lr

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

    def compute_static_offset_laplace(self):
        total_offset = self.model_params["static_offset"]
        mesh = Meshes(verts=[total_offset], faces=[self.faces])
        lap = mesh_laplacian_smoothing_per_vertex(mesh, "cot") # nverts x 3
        return torch.sum((lap ** 2).sum(dim=1))

    def get_dynamic_offset(self):
        x = self.model_params["dynamic_offset"]
        w_0 = self.model_params["bs_weights"][self.timestep] + self.model_params["add_bs_weights"][self.timestep]
        w_0 = w_0.unsqueeze(0)
        w_0 = w_0.reshape(*(w_0.shape), 1, 1)
        dyn = (w_0 * x).sum(1)[0]

        return dyn

    def reset_gaussian_params(self):
        num_pts = self.binding.shape[0]
        
        # reset xyz
        xyz_new = torch.zeros((num_pts, 3)).float().cuda()
        optimizable_tensors = self.replace_tensor_to_optimizer(xyz_new, "xyz")
        self._xyz = optimizable_tensors["xyz"]

        # reset color
        self.active_sh_degree = 0

        optimizable_tensors = self.replace_tensor_to_optimizer(self._features_dc_original.clone(), "f_dc")
        self._features_dc = optimizable_tensors["f_dc"]

        optimizable_tensors = self.replace_tensor_to_optimizer(self._features_rest_original.clone(), "f_rest")
        self._features_rest = optimizable_tensors["f_rest"]
    
    def reset_all(self, training_args):
        num_pts = self.binding.shape[0]
        
        # reset xyz
        xyz_new = torch.zeros((num_pts, 3)).float().cuda()
        optimizable_tensors = self.replace_tensor_to_optimizer(xyz_new, "xyz")
        self._xyz = optimizable_tensors["xyz"]

        # reset rot
        optimizable_tensors = self.replace_tensor_to_optimizer(self._rotation_original.clone(), "rotation")
        self._scaling = optimizable_tensors["rotation"]

        # reset scale
        optimizable_tensors = self.replace_tensor_to_optimizer(self._scaling_original.clone(), "scaling")
        self._scaling = optimizable_tensors["scaling"]

        # reset color
        self.active_sh_degree = 0

        optimizable_tensors = self.replace_tensor_to_optimizer(self._features_dc_original.clone(), "f_dc")
        self._features_dc = optimizable_tensors["f_dc"]

        optimizable_tensors = self.replace_tensor_to_optimizer(self._features_rest_original.clone(), "f_rest")
        self._features_rest = optimizable_tensors["f_rest"]

        # reset opacity
        optimizable_tensors = self.replace_tensor_to_optimizer(self._opacity_original, "opacity")
        self._opacity = optimizable_tensors["opacity"]

        # reset bs_weights
        optimizable_tensors = self.replace_tensor_to_optimizer(self._add_bs_weights_original.clone(), "add_bs_weights")
        self.model_params["add_bs_weights"] = optimizable_tensors["add_bs_weights"]

        # # reset static offsets
        # optimizable_tensors = self.replace_tensor_to_optimizer(self._static_offset_original.clone(), "static_offset")
        # self.model_params["static_offset"] = optimizable_tensors["static_offset"]

        # reset LR scheduler for xyz 
        self.activate_xyz_learning(training_args)

    def activate_xyz_learning(self, training_args):
        # when this is called we are at iteration "training_args.reposition_until"
        def mock_reset_scheduler_lr_func(step):
            helper = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps - training_args.reposition_until) #this way and
            return helper(step - training_args.reposition_until) # this way we shift from 10000-600000 to 0-590000
        self.xyz_scheduler_args = mock_reset_scheduler_lr_func
        # print(f"FROM 0 TO {training_args.position_lr_max_steps - training_args.reposition_until} (590000)")
        print(f"[ACTIVATE] Successfully updated lr scheduler for xyz!")

    def deactivate_xyz_learning(self):
        def zero_func(step):
            return 0
        self.xyz_scheduler_args = zero_func
        print(f"[DEACTIVATE] Successfully set xyz_scheduler to zero_func!")
    
    def activate_bs_learning(self, training_args):
        # when this is called we are at iteration "training_args.reposition_until"
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "add_bs_weights":
                param_group['lr'] = training_args.bs_lr
                print(f"[ACTIVATE] Successfully set lr for add_bs_weights to {training_args.bs_lr}!")
                return
        raise Exception("Could not activate add_bs_weights lr!")

    def deactivate_bs_learning(self):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "add_bs_weights":
                param_group['lr'] = 0
                print(f"[DEACTIVATE] Successfully set lr for add_bs_weights to zero!")
                return
        raise Exception("Could not deactivate add_bs_weights lr!")

