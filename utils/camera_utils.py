#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import string

import numpy as np
import torch
from tqdm import tqdm

from scene.cameras import Camera, MiniCam
from utils.graphics_utils import fov2focal
from utils.pytorch3d_load_obj import save_obj

WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.width, cam_info.height

    if args.resolution in [1, 2, 4, 8]:
        image_width, image_height = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        image_width, image_height = (int(orig_w / scale), int(orig_h / scale))

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image_width=image_width, image_height=image_height,
                  bg=cam_info.bg, 
                  image=cam_info.image, 
                  image_path=cam_info.image_path,
                  image_name=cam_info.image_name, uid=id, 
                  timestep=cam_info.timestep, data_device=args.data_device,
                  trans=cam_info.trans, scale=cam_info.scale)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in tqdm(enumerate(cam_infos), total=len(cam_infos)):
        if args.select_camera_id != -1 and c.camera_id is not None:
            if c.camera_id != args.select_camera_id:
                continue
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry

def extract_c2w_mat_from_xml_string(matrix_string: string):
    # Convert the string into a list of floats
    matrix_values = list(map(float, matrix_string.split()))

    # Reshape the list into a 4x4 numpy array
    c2w_mat = np.array(matrix_values).reshape(4, 4)

    return c2w_mat


def camera2miniCam(camera: Camera):
    return MiniCam(
        width=camera.image_width,
        height=camera.image_height,
        fovx=camera.FoVx,
        fovy=camera.FoVy,
        znear=camera.znear,
        zfar=camera.zfar,
        world_view_transform=camera.world_view_transform,
        full_proj_transform=camera.full_proj_transform,
        timestep=camera.timestep,
    )

def save_cams_as_mesh(cameras: [Camera], scale=0.025):

    model_verts = np.array([[1, 1, 1], #0
                            [1, 1, -1],
                            [1, -1, 1], #2
                            [1, -1, -1],
                            [-1, 1, 1], #4
                            [-1, 1, -1],
                            [-1, -1, 1], # 6
                            [-1, -1, -1],
                            [0, 0, 5] #8
                            ]) * scale
    n_model_verts = model_verts.shape[0]

    model_faces = np.array([
        [4, 6, 2], [4, 2, 0],
        [1, 3, 7], [1, 7, 5],
        [5, 7, 6], [5, 6, 4],
        [0, 2, 3], [0, 3, 1],
        [4, 5, 1], [4, 1, 0],
        [6, 7, 3], [6, 3, 2],
        [8, 0, 2], [8, 2, 6], [8, 4, 6], [8,4,0]
    ])

    camera_verts=[]
    camera_faces=[]
    for i, c in enumerate(cameras):
        t = c.camera_center.numpy().reshape(1,3)

        w2c = c.world_view_transform
        R = np.linalg.inv(w2c)[:3,:3] # rot of c2w

        vs = (model_verts @ R) + t

        camera_verts.append(vs)
        camera_faces.append(model_faces + i * np.array([n_model_verts,n_model_verts,n_model_verts]))

    # Convert lists to proper NumPy arrays
    camera_verts = np.vstack(camera_verts)  # Stack along the first axis
    camera_faces = np.vstack(camera_faces)  # Stack face indices

    # Convert to tensors
    camera_verts = torch.from_numpy(camera_verts)
    camera_faces = torch.from_numpy(camera_faces)

    save_obj("./output/camera_models.obj", verts=camera_verts, faces= camera_faces)

def save_cams_as_pcd(cameras: [Camera], scale=0.025):

    lookdir_verts = np.array([[0, 0, 0], [0, 0, 1],[0, 0, 2],[0, 0, 3],[0, 0, 4],[0, 0, 5]]) * scale

    camera_verts=[]
    for i, c in enumerate(cameras):
        t = c.camera_center.numpy().reshape(1,3)

        w2c = c.world_view_transform
        R = np.linalg.inv(w2c)[:3,:3] # rot of c2w

        vs = (lookdir_verts @ R) + t

        camera_verts.append(vs)

    # Convert lists to proper NumPy arrays
    camera_verts = np.vstack(camera_verts)  # Stack along the first axis

    # Convert to tensors
    camera_verts = torch.from_numpy(camera_verts)

    save_obj("./output/camera_models_pcd.obj", verts=camera_verts, faces=torch.tensor([]))