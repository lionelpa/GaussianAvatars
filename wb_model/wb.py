# from .lbs import lbs, vertices2landmarks, blend_shapes, vertices2joints
import os
import re

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

try:
    from pytorch3d.io import load_obj
except ImportError:
    from utils.pytorch3d_load_obj import load_obj

WB_BASE_MESH_PATH = "wb_model/assets/4_head.obj"
WB_MESHES_PATH = "wb_model/assets/"
WB_MESHES_NAME_FILTER_PATTERN="([0-9]+)_head\.obj"



def to_tensor(array, dtype=torch.float32):
    if "torch.tensor" not in str(type(array)):
        return torch.tensor(array, dtype=dtype)


def to_np(array, dtype=np.float32):
    if "scipy.sparse" in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)


class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


def face_vertices(vertices, faces):
    """
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of faces, 3, 3]
    """
    assert vertices.ndimension() == 3
    assert faces.ndimension() == 3
    assert vertices.shape[0] == faces.shape[0]
    assert vertices.shape[2] == 3
    assert faces.shape[2] == 3

    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
    vertices = vertices.reshape((bs * nv, 3))
    # pytorch only supports long and byte tensors for indexing
    return vertices[faces.long()]


class WBModel(nn.Module):
    """
    Given flame parameters this class generates a differentiable FLAME function
    which outputs the a mesh and 2D/3D facial landmarks
    """

    def __init__(
            self,
            wb_base_mesh_path=WB_BASE_MESH_PATH,
            wb_meshes_path=WB_MESHES_PATH,
            wb_meshes_name_filter_pattern=WB_MESHES_NAME_FILTER_PATTERN,
    ):
        """
        Initializes the class with paths and filters for loading meshes.

        Args:
            wb_base_mesh_path (str): Path to the base mesh file.
            wb_meshes_path (str): Path to the directory containing the mesh files.
            wb_meshes_name_filter_pattern (str): Regex pattern to filter out wanted mesh files where first regex group locates timestep number.
        """
        super(WBModel, self).__init__()

        # Get num_verts and triangle info from base mesh. Tris do not change throughout training and can be buffered
        verts, faces, _ = load_obj(wb_base_mesh_path, load_textures=False)
        self.register_buffer("faces", faces.verts_idx, persistent=False)

        # Load each head.obj and store it in a buffer
        pattern = re.compile(wb_meshes_name_filter_pattern)
        mesh_file_names = sorted([f for f in os.listdir(wb_meshes_path) if pattern.match(f)])
        n_meshes = len(mesh_file_names)
        n_verts = len(verts)

        # timestep -> mesh vertices
        self.timestep_to_mesh_dict = {}
        for mesh_file in tqdm(sorted(mesh_file_names, key= lambda x: int(x.split("_head.")[0])), desc="Loading meshes", unit=" objs"):
            full_path = os.path.join(wb_meshes_path, mesh_file)

            match = pattern.match(mesh_file)
            timestep = int(match.group(1)) if match else None

            verts, _, _ = load_obj(full_path, load_textures=False)
            # move all mesh vert tensors to gpu
            self.timestep_to_mesh_dict[timestep] = verts.unsqueeze(0).float().cuda()

        # self.register_buffer("textures_idx", faces.textures_idx, persistent=False)
        # Check our template mesh faces match those of FLAME:

    def forward(self, timestep):
        return self.timestep_to_mesh_dict[timestep]


class BufferContainer(nn.Module):
    def __init__(self):
        super().__init__()

    def __repr__(self):
        main_str = super().__repr__() + '\n'
        for name, buf in self.named_buffers():
            main_str += f'    {name:20}\t{buf.shape}\t{buf.dtype}\n'
        return main_str

    def __iter__(self):
        for name, buf in self.named_buffers():
            yield name, buf

    def keys(self):
        return [name for name, buf in self.named_buffers()]

    def items(self):
        return [(name, buf) for name, buf in self.named_buffers()]


if __name__ == '__main__':
    wb_model = WBModel()
