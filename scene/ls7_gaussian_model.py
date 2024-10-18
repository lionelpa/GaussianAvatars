# 
# Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual 
# property and proprietary rights in and to this software and related documentation. 
# Any commercial use, reproduction, disclosure or distribution of this software and 
# related documentation without an express license agreement from Toyota Motor Europe NV/SA 
# is strictly prohibited.
#
import time
from pathlib import Path
import numpy as np
import torch
# from vht.model.flame import FlameHead
from flame_model.flame import FlameHead
from ls7_model.ls7_model import LS7Model

from .gaussian_model import GaussianModel
from utils.graphics_utils import compute_face_orientation
# from pytorch3d.transforms import matrix_to_quaternion
from roma import rotmat_to_unitquat, quat_xyzw_to_wxyz


class LS7GaussianModel(GaussianModel):
    def __init__(self, sh_degree : int, n_shape=300, n_expr=100):
        super().__init__(sh_degree)

        # load LS7 model and move it to GPU using cuda()
        self.ls7_model = LS7Model().cuda()
        self.verts = self.ls7_model.verts
        self.faces = self.ls7_model.faces

        # position
        self.face_center = self.calculate_face_centers(self.verts, self.faces)
        self.face_center = self.face_center.cuda()

        # orientation and scale
        self.face_orien_mat, self.face_scaling = compute_face_orientation(self.verts.squeeze(0), self.faces.squeeze(0), return_scale=True)
        # self.face_orien_quat = matrix_to_quaternion(self.face_orien_mat)  # pytorch3d (WXYZ)
        self.face_orien_quat = quat_xyzw_to_wxyz(rotmat_to_unitquat(self.face_orien_mat))  # roma
        
        self.face_orien_quat = self.face_orien_quat.cuda()
        self.face_orien_mat = self.face_orien_mat.cuda()
        self.face_scaling = self.face_scaling.cuda()

        # binding is initialized once the mesh topology is known
        # binding is a tensor that maps gaussian index to face index. Thus it initially has the length of 'self.flame_model.faces'
        if self.binding is None:
            self.binding = torch.arange(len(self.ls7_model.faces)).cuda()
            self.binding_counter = torch.ones(len(self.ls7_model.faces), dtype=torch.int32).cuda()


    def calculate_face_centers(self, verts, faces):
        num_faces = faces.shape[0]
        face_centers = []
        # Start time
        start_time = time.time()

        for i in range(num_faces):
            # Get the indices of the vertices for the current face
            v0, v1, v2 = faces[i]

            # Get the positions of the vertices
            triangle = verts[[v0, v1, v2], :]

            # Calculate the center of the triangle
            center = triangle.mean(dim=0)

            # Store the result
            face_centers.append(center)

        # Convert list of centers back to a tensor
        return torch.stack(face_centers)
    
    # def compute_dynamic_offset_loss(self):
    #     # loss_dynamic = (self.flame_param['dynamic_offset'][[self.timestep]] - self.flame_param_orig['dynamic_offset'][[self.timestep]]).norm(dim=-1)
    #     loss_dynamic = self.flame_param['dynamic_offset'][[self.timestep]].norm(dim=-1)
    #     return loss_dynamic.mean()
    #
    # def compute_laplacian_loss(self):
    #     # offset = self.flame_param['static_offset'] + self.flame_param['dynamic_offset'][[self.timestep]]
    #     offset = self.flame_param['dynamic_offset'][[self.timestep]]
    #     verts_wo_offset = (self.verts_cano - offset).detach()
    #     verts_w_offset = verts_wo_offset + offset
    #
    #     L = self.flame_model.laplacian_matrix[None, ...].detach()  # (1, V, V)
    #     lap_wo = L.bmm(verts_wo_offset).detach()
    #     lap_w = L.bmm(verts_w_offset)
    #     diff = (lap_wo - lap_w) ** 2
    #     diff = diff.sum(dim=-1, keepdim=True)
    #     return diff.mean()