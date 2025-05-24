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
import glob
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np
import torch
from PIL import Image
from plyfile import PlyData, PlyElement
from tqdm import tqdm

from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from scene.gaussian_model import BasicPointCloud
from utils.camera_utils import extract_c2w_mat_from_xml_string
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
from utils.sh_utils import SH2RGB


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: Optional[np.array]
    image_path: str
    image_name: str
    width: int
    height: int
    bg: np.array = np.array([0, 0, 0])
    timestep: Optional[int] = None
    camera_id: Optional[int] = None
    trans: np.array = np.array([0, 0, 0])
    scale: float = 1

class SceneInfo(NamedTuple):
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    point_cloud: Optional[BasicPointCloud]
    ply_path: Optional[str]
    val_cameras: list = []
    train_meshes: dict = {}
    test_meshes: dict = {}
    tgt_train_meshes: dict = {}
    tgt_test_meshes: dict = {}
    #
    meshes: dict = {}

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def getNerfppNormHylec(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)

        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        # todo 17.2.25 check if correct
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    # save_as_ply(np.hstack(cam_centers).transpose(), np.array([center]), path="./output/_plys/PPNormHylec.ply")
    radius = diagonal * 1.1

    translate = -center
    return {"translate": translate, "radius": radius}


def readMeshParamsForTimesteps(source_path):
    # We need to parse infos for all frames(timesteps) so when the model gets saved all params get saved. This is
    # necessary because if we only load the params for test and train frames we wouldnt be able to look at frames
    # outside them properly in the viewer.
    # TODO parameterize
    params_path = os.path.join(source_path, "smooth2")

    Rs = readMeshParam(params_path, "*_R.txt", 3)
    Ts = readMeshParam(params_path, "*_t.txt", 3)
    Ss = readMeshParam(params_path, "*_s.txt", 3)
    Ws = readMeshParam(params_path, "*_w.txt", 52)
    means = readMeshParam(params_path, "*_mean.txt", 3)
    assert len(Rs) == len(Ts) == len(Ss) == len(Ws) == len(means), "Number of params mismatch. Are some txt-files missing?"

    mesh_params = {t:{"rotation":Rs[t],
                           "translation":Ts[t],
                           "scale":Ss[t],
                           "bs_weights":Ws[t],
                           "mean":means[t]} for t in Rs.keys()}
    return mesh_params

def readMeshParam(params_path, param_filename_pattern, expected_param_dim):
    # load param from file matching Pattern
    file_pattern = os.path.join(params_path, param_filename_pattern)
    file_list = glob.glob(file_pattern)
    # loop through each matching file and print its contents
    param_dict = {}
    for file_path in file_list:
        # extract filename without path
        filename = os.path.basename(file_path)
        timestep = int(filename.split("_")[0])  # format like "{timestep}_R.txt"

        with open(file_path, 'r') as f:
            parsed_param_tensor = torch.tensor([float(x.strip()) for x in f.readlines()])
            param_dict[timestep] = parsed_param_tensor
            assert len(parsed_param_tensor) == expected_param_dim
    return param_dict


def readSceneInfoForScannerWB(source_path, images_folder_name, centroid, rescale_factor, val_cam_ids, train_frames, test_frames, eval):
    train_cam_infos, val_cam_infos, test_cam_infos = readWBCamerasFromXML(source_path, images_folder_name,
                                                                          "cameras.xml",
                                                                          centroid, rescale_factor, 
                                                                          val_cam_ids, train_frames, test_frames, eval)

    mesh_params = readMeshParamsForTimesteps(source_path)

    nerf_normalization = getNerfppNorm(train_cam_infos)

    scene_info = SceneInfo(point_cloud=None,
                           train_cameras=train_cam_infos,
                           val_cameras=val_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           meshes=mesh_params,
                           ply_path=None)
    return scene_info


def readWBCamerasFromXML(source_path, images_folder_name, cameras_xml_file_name, centroid, rescale_factor, val_cam_ids, train_frames, test_frames, eval):
    camsXML_path = os.path.join(source_path, cameras_xml_file_name)
    tree = ET.parse(camsXML_path)
    root = tree.getroot()

    # read sensors
    sensors_root = root.find("sensors") # first chunk contains sensors intrinsics
    sensors = dict()
    for s in sensors_root.findall("sensor"):
        sid = int(s.get("id"))

        c = s.find("calibration")
        r = c.find("resolution")
        height = int(r.get("height"))
        width = int(r.get("width"))
        focal_in_pix = float(c.find("f").text)

        # calculate fovs from
        fovX = focal2fov(focal_in_pix, width)
        fovY = focal2fov(focal_in_pix, height)

        sensor_info = CameraInfo(uid=sid, FovX=fovX, FovY=fovY, width=width, height=height,
                                 R=None, T=None, image=None, image_path=None, image_name=None)
        sensors.update({sid: sensor_info})
    # read cameras
    cameras_root = root.find("cameras")  # first chunk contains all extrinsics

    train_cam_infos = []
    val_cam_infos = []
    test_cam_infos = []
    discarded_frames = set()
    for c in cameras_root.findall("camera"):
        cam_id = int(c.get("id"))
        sid = int(c.get("sensor_id"))
        sensor = sensors[sid]

        mat_string = c.find(".//transform").text
        c2w = extract_c2w_mat_from_xml_string(mat_string)

        # for Camera obj we need R and T of the w2c matrix
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3, :3])   # R is stored transposed due to 'glm' in CUDA code
        # print("WOW")
        # print(w2c[:3, 3])
        # print(centroid.numpy())
        # print((w2c[:3, 3] - centroid.numpy()))
        # print((w2c[:3, 3] - centroid.numpy()) * rescale_factor)
        
        T = w2c[:3, 3]

        # bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])

        # create a camera for each timestep
        camera_images_folder_path = os.path.join(source_path, images_folder_name, str(cam_id))
        image_paths = sorted(glob.glob(os.path.join(camera_images_folder_path, "*.png")))
        # Loop through all images and create a camera for each
        for image_path in image_paths:
            image_name = os.path.basename(image_path)
            timestep = int(os.path.basename(image_path).split(".")[0])  # naming convention of image is {camera_id}/{timestep}.png

            # param 'image' is None since it is loaded dynamically by the DatasetLoader in train.py
            cam = CameraInfo(uid=cam_id, FovY=sensor.FovY, FovX=sensor.FovX, width=sensor.width, height=sensor.height,
                             R=R, T=T, image=None, image_path=image_path, image_name=image_name, timestep=timestep,
                             trans=-centroid, scale=rescale_factor)

            if timestep in test_frames:
                test_cam_infos.append(cam)
            elif timestep in train_frames:
                if cam_id in val_cam_ids:
                    val_cam_infos.append(cam)
                else:
                    train_cam_infos.append(cam)
            else:
                discarded_frames.add(timestep)

    print(f"===== Finished loading cameras from {camsXML_path}")
    print(f"#Train: {len(train_cam_infos)} total = {len(train_cam_infos)//len(train_frames)} cams x {len(train_frames)} frames")
    # print(f"#Val  : {len(val_cam_infos)} total = {len(val_cam_ids)} cams x {len(train_frames)} frames")
    # print(f"#Test : {len(test_cam_infos)} total = {len(test_cam_infos)//len(test_frames)} cams x {len(test_frames)} frames")
    print(f"[WARNING]: Discarded the following frames:\n{sorted(list(discarded_frames))}")

    if not eval:
        print("Eval param is false -> discarding testing cams")
        test_cam_infos = []
    return train_cam_infos, val_cam_infos, test_cam_infos


def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)
        width, height = image.size

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        if 'camera_angle_x' in contents:
            fovx_shared = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in tqdm(enumerate(frames), total=len(frames)):
            file_path = frame["file_path"]
            if extension not in frame["file_path"]:
                file_path += extension
            cam_name = os.path.join(path, file_path)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            
            if 'w' in frame and 'h' in frame:
                image = None
                width = frame['w']
                height = frame['h']
            else:
                image = Image.open(image_path)
                im_data = np.array(image.convert("RGBA"))
                norm_data = im_data / 255.0
                arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
                image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
                width, height = image.size

            if 'camera_angle_x' in frame:
                fovx = frame["camera_angle_x"]
            else:
                fovx = fovx_shared
            fovy = focal2fov(fov2focal(fovx, width), height)

            timestep = frame["timestep_index"] if 'timestep_index' in frame else None
            camera_id = frame["camera_index"] if 'camera_id' in frame else None
            
            cam_infos.append(CameraInfo(
                uid=idx, R=R, T=T, FovY=fovy, FovX=fovx, bg=bg, image=image, 
                image_path=image_path, image_name=image_name, 
                width=width, height=height, 
                timestep=timestep, camera_id=camera_id))
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readMeshesFromTransforms(path, transformsfile):
    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        frames = contents["frames"]
        
        mesh_infos = {}
        for idx, frame in tqdm(enumerate(frames), total=len(frames)):
            if not 'timestep_index' in frame or frame["timestep_index"] in mesh_infos:
                continue

            flame_param = dict(np.load(os.path.join(path, frame['flame_param_path']), allow_pickle=True))
            mesh_infos[frame["timestep_index"]] = flame_param
    return mesh_infos

def readDynamicNerfInfo(path, white_background, eval, extension=".png", target_path=""):
    print("Reading Training Transforms")
    if target_path != "":
        train_cam_infos = readCamerasFromTransforms(target_path, "transforms_train.json", white_background, extension)
    else:
        train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    
    print("Reading Training Meshes")
    train_mesh_infos = readMeshesFromTransforms(path, "transforms_train.json")
    if target_path != "":
        print("Reading Target Meshes (Training Division)")
        tgt_train_mesh_infos = readMeshesFromTransforms(target_path, "transforms_train.json")
    else:
        tgt_train_mesh_infos = {}
    
    print("Reading Validation Transforms")
    if target_path != "":
        val_cam_infos = readCamerasFromTransforms(target_path, "transforms_val.json", white_background, extension)
    else:
        val_cam_infos = readCamerasFromTransforms(path, "transforms_val.json", white_background, extension)
    
    print("Reading Test Transforms")
    if target_path != "":
        test_cam_infos = readCamerasFromTransforms(target_path, "transforms_test.json", white_background, extension)
    else:
        test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    print("Reading Test Meshes")
    test_mesh_infos = readMeshesFromTransforms(path, "transforms_test.json")
    if target_path != "":
        print("Reading Target Meshes (Test Division)")
        tgt_test_mesh_infos = readMeshesFromTransforms(target_path, "transforms_test.json")
    else:
        tgt_test_mesh_infos = {}
    
    if target_path != "" or not eval:
        train_cam_infos.extend(val_cam_infos)
        val_cam_infos = []
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []
        train_mesh_infos.update(test_mesh_infos)
        test_mesh_infos = {}

    nerf_normalization = getNerfppNorm(train_cam_infos)

    scene_info = SceneInfo(point_cloud=None,
                           train_cameras=train_cam_infos,
                           val_cameras=val_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=None,
                           train_meshes=train_mesh_infos,
                           test_meshes=test_mesh_infos,
                           tgt_train_meshes=tgt_train_mesh_infos,
                           tgt_test_meshes=tgt_test_mesh_infos)
    return scene_info

sceneLoadTypeCallbacks = {
    "ScannerWB": readSceneInfoForScannerWB,

    "Colmap": readColmapSceneInfo,
    "DynamicNerf" : readDynamicNerfInfo,
    "Blender" : readNerfSyntheticInfo,
}