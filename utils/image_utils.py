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

import torch
from matplotlib import cm
import numpy as np
from PIL import Image
import os

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def error_map(img1, img2):
    error = (img1 - img2).mean(dim=0) / 2 + 0.5
    cmap = cm.get_cmap("seismic")
    error_map = cmap(error.cpu())
    return torch.from_numpy(error_map[..., :3]).permute(2, 0, 1)

def save_tensor_as_image(tensor, cam_name, iteration, training_start_time, output_dir="output/cam_renders", force_tensor_to_cpu=True):
    """
    Convert a 3xHxW image tensor to a PIL image and save it to the specified path. Creates the directory if it doesn't exist.

    Args:
    - tensor (torch.Tensor): The input tensor with shape (3, H, W).
    - output_path (str): The path where the image should be saved.
    """
    # Ensure the tensor is on the CPU
    if force_tensor_to_cpu:
        tensor = tensor.cpu()

    # Permute the tensor to (H, W, C) format and convert to a numpy array
    image_np = tensor.permute(1, 2, 0).detach().numpy()

    # Scale the values to [0, 255] and convert to uint8
    image_np = (image_np * 255).astype(np.uint8)

    # Convert the numpy array to a PIL image
    image = Image.fromarray(image_np)

    # Create a directory with the current date and time if it doesn't exist
    # Get the parent directory of the script's location
    current_time = training_start_time.strftime("%Y-%m-%d_%H-%M-%S")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)  # Go up one level
    out_dir = os.path.join(parent_dir, output_dir, f"{current_time}_CAM-{cam_name}")
    os.makedirs(out_dir, exist_ok=True)

    # Save the image to the specified output path in the new directory
    image_name = f"cam{cam_name}_it{iteration}.png"
    output_file_path = os.path.join(out_dir, image_name)
    image.save(output_file_path)