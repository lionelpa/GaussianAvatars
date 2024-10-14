from torch import nn

from utils.pytorch3d_load_obj import load_obj

LS7_MESH_PATH_LIONEL = "/home/lio/PycharmProjects/GaussianAvatars/ls7_model/assets/result_noGlands.obj"
# LS7_MESH_PATH_LIONEL = "ls7_model/assets/result_noGlands.obj"



def load_mesh(mesh_path):
    verts, faces, aux = load_obj(mesh_path, load_textures=False)

    faces = faces.verts_idx

    return verts, faces


class LS7Model(nn.Module):
    def __init__(self,
                 ls7_mesh_path=LS7_MESH_PATH_LIONEL):
        super().__init__()

        self.verts = None
        self.faces = None

        verts, faces = load_mesh(ls7_mesh_path)
        self.verts = verts
        self.faces = faces
