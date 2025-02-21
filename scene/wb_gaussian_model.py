#
# Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual
# property and proprietary rights in and to this software and related documentation.
# Any commercial use, reproduction, disclosure or distribution of this software and
# related documentation without an express license agreement from Toyota Motor Europe NV/SA
# is strictly prohibited.
#

import torch
from roma import rotmat_to_unitquat, quat_xyzw_to_wxyz

from utils.graphics_utils import compute_face_orientation
from wb_model.wb import WBModel
from .gaussian_model import GaussianModel


class WBGaussianModel(GaussianModel):
    def __init__(self, sh_degree: int):
        super().__init__(sh_degree)

        self.wb_model = WBModel().cuda()
        self.num_timesteps = self.wb_model.num_timesteps
        self.min_timestep = self.wb_model.start_timestep
        self.max_timestep = self.wb_model.end_timestep

        self.verts = list(self.wb_model.timestep_to_mesh_dict.values())[0]
        self.verts_uvs = self.wb_model.verts_uvs
        self.faces = self.wb_model.faces
        self.faces_uvs = self.wb_model.faces_uvs
        self.texture = self.wb_model.texture


        # binding is initialized once the mesh topology is known
        if self.binding is None:
            self.binding = torch.arange(len(self.wb_model.faces)).cuda()
            self.binding_counter = torch.ones(len(self.wb_model.faces), dtype=torch.int32).cuda()

    def update_mesh_by_timestep(self, timestep):
        verts, verts_cano = self.wb_model(timestep)
        self.update_mesh_properties(verts, verts_cano)

    def select_mesh_by_timestep(self, timestep, original=False):
        self.timestep = timestep

        verts = self.wb_model(timestep)
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
        # meshes = {**train_meshes, **test_meshes}
        # tgt_meshes = {**tgt_train_meshes, **tgt_test_meshes}
        # print("len(meshes):", len(meshes))
        # print("len(tgt)   :", len(tgt_meshes))
        # pose_meshes = meshes if len(tgt_meshes) == 0 else tgt_meshes
        # print("len(pose)  :", len(pose_meshes))
        
        # self.num_timesteps = max(pose_meshes) + 1  # required by viewers and training view when evaluating test and val data
        # print("self.num_timesteps", self.num_timesteps)
        return
