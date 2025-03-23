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

import json
import os
import sys
import uuid
from argparse import ArgumentParser, Namespace

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_renderer import render, network_gui
from lpipsPyTorch import lpips
from mesh_renderer import NVDiffRenderer
from scene import Scene
from scene.wb_gaussian_model import WBGaussianModel
from utils.general_utils import safe_state
from utils.image_utils import psnr, error_map
from utils.loss_utils import l1_loss, ssim

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, render_meshes_iterations):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    if dataset.bind_to_mesh:
        gaussians = WBGaussianModel(dataset.center_and_scale, dataset.sh_degree)
        mesh_renderer = NVDiffRenderer()
    else:
        raise Exception("please use --bind-to-mesh flag")
    scene = Scene(dataset, gaussians)
    # gaussians.save_ply_for_SIBR(f"{dataset.model_path}/_init_gaussians.ply", scene, render_debug_origin=True)
    gaussians.training_setup(opt)
    if opt.reposition_until > 0:
        if opt.fixate_xyz_during_repos:
            gaussians.deactivate_xyz_learning()
        if opt.dont_learn_bs_during_repos:
            gaussians.deactivate_bs_learning()

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    loader_camera_train = DataLoader(scene.getTrainCameras(), batch_size=None, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    iter_camera_train = iter(loader_camera_train)

    # chosen_cams = set()
    # training_start_time = datetime.now()
    # for cam in tqdm(iter_camera_train, desc="Render images for cams", unit=" cams"):
    #     if cam.colmap_id not in chosen_cams and cam.timestep == 4:
    #         chosen_cams.add(cam.colmap_id)
    #         gaussians.select_mesh_by_timestep(cam.timestep)
    #
    #         # export mesh render
    #         out_dict = mesh_renderer.render_from_camera(gaussians.verts, gaussians.faces, cam)
    #         rgba_mesh = out_dict['rgba'].squeeze(0)  # (H, W, C)
    #         rgb_mesh = rgba_mesh[:, :, :3]
    #         image=rgb_mesh.permute(2,0,1)
    #         save_tensor_as_image(image, cam, -1, training_start_time)
    #
    #         ## export gaussian render
    #         render_pkg = render(cam, gaussians, pipe, background)
    #         image = render_pkg["render"]
    #         save_tensor_as_image(image, cam, -2, training_start_time)
    # raise Exception("Finished generating test images")

    # viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                # receive data
                net_image = None
                # custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer, use_original_mesh = network_gui.receive()
                custom_cam, msg = network_gui.receive()

                # render
                if custom_cam != None:
                    # mesh selection by timestep
                    if gaussians.binding != None:
                        gaussians.select_mesh_by_timestep(custom_cam.timestep, msg['use_original_mesh'])
                    
                    # gaussian splatting rendering
                    if msg['show_splatting']:
                        net_image = render(custom_cam, gaussians, pipe, background, msg['scaling_modifier'])["render"]
                    
                    # mesh rendering
                    if gaussians.binding != None and msg['show_mesh']:
                        out_dict = mesh_renderer.render_from_camera(gaussians.verts, gaussians.faces, custom_cam)

                        rgba_mesh = out_dict['rgba'].squeeze(0).permute(2, 0, 1)  # (C, W, H)
                        rgb_mesh = rgba_mesh[:3, :, :]
                        alpha_mesh = rgba_mesh[3:, :, :]

                        mesh_opacity = msg['mesh_opacity']
                        if net_image is None:
                            net_image = rgb_mesh
                        else:
                            net_image = rgb_mesh * alpha_mesh * mesh_opacity  + net_image * (alpha_mesh * (1 - mesh_opacity) + (1 - alpha_mesh))

                    # send data
                    net_dict = {'num_timesteps': gaussians.num_timesteps, 'num_points': gaussians._xyz.shape[0]}
                    network_gui.send(net_image, net_dict)
                if msg['do_training'] and ((iteration < int(opt.iterations)) or not msg['keep_alive']):
                    break
            except Exception as e:
                # print(e)
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # if we need to reset regularly during alignment phase
        if opt.reposition_until > 0:
            if iteration < opt.reposition_until and opt.reset_interval > 0 and iteration % opt.reset_interval == 0:
                gaussians.reset_gaussian_params()
                print(f"[ITER {iteration}] Reset gaussian params!")
            elif iteration == (opt.reposition_until + 1):
                gaussians.reset_all(opt)
                print(f"[ITER {iteration}] FULL RESET")
                if opt.fixate_xyz_during_repos: # make learnable after repos phase
                    gaussians.activate_xyz_learning(opt)
                if opt.dont_learn_bs_during_repos: # make learnable after repos phase
                    gaussians.activate_bs_learning(opt)
                gaussians.update_learning_rate(iteration)


        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        try:
            viewpoint_cam = next(iter_camera_train)
        except StopIteration:
            iter_camera_train = iter(loader_camera_train)
            viewpoint_cam = next(iter_camera_train)

        if gaussians.binding != None:
            gaussians.select_mesh_by_timestep(viewpoint_cam.timestep)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()

        losses = {}
        losses['l1'] = l1_loss(image, gt_image) * (1.0 - opt.lambda_dssim)
        losses['ssim'] = (1.0 - ssim(image, gt_image)) * opt.lambda_dssim

        if gaussians.binding != None:
            if opt.metric_xyz:
                losses['xyz'] = F.relu((gaussians._xyz*gaussians.face_scaling[gaussians.binding])[visibility_filter] - opt.threshold_xyz).norm(dim=1).mean() * opt.lambda_xyz
            else:
                # losses['xyz'] = gaussians._xyz.norm(dim=1).mean() * opt.lambda_xyz
                losses['xyz'] = F.relu(gaussians._xyz[visibility_filter].norm(dim=1) - opt.threshold_xyz).mean() * opt.lambda_xyz

            if opt.lambda_scale != 0:
                if opt.metric_scale:
                    losses['scale'] = F.relu(gaussians.get_scaling[visibility_filter] - opt.threshold_scale).norm(dim=1).mean() * opt.lambda_scale
                else:
                    # losses['scale'] = F.relu(gaussians._scaling).norm(dim=1).mean() * opt.lambda_scale
                    losses['scale'] = F.relu(torch.exp(gaussians._scaling[visibility_filter]) - opt.threshold_scale).norm(dim=1).mean() * opt.lambda_scale

            if opt.lambda_dynamic_offset != 0:
                losses['dy_off'] = gaussians.compute_dynamic_offset_loss() * opt.lambda_dynamic_offset

            if opt.lambda_dynamic_offset_std != 0:
                ti = viewpoint_cam.timestep
                t_indices =[ti]
                if ti > 0:
                    t_indices.append(ti-1)
                if ti < gaussians.num_timesteps - 1:
                    t_indices.append(ti+1)
                losses['dynamic_offset_std'] = gaussians.flame_param['dynamic_offset'].std(dim=0).mean() * opt.lambda_dynamic_offset_std
        
            if opt.lambda_laplacian != 0:
                losses['lap'] = gaussians.compute_laplacian_loss() * opt.lambda_laplacian

            # custom
            if opt.lambda_static_offset_laplacian != 0:
                losses["offset_lap"] = gaussians.compute_offset_laplacian_mse_loss() * opt.lambda_static_offset_laplacian
            if opt.lambda_offset_norm != 0:
                losses["offset_norm"] = gaussians.compute_offset_loss() * opt.lambda_offset_norm
        
        losses['total'] = sum([v for k, v in losses.items()])
        losses['total'].backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * losses['total'].item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                postfix = {"Loss": f"{ema_loss_for_log:.{7}f}"}
                if 'xyz' in losses:
                    postfix["xyz"] = f"{losses['xyz']:.{7}f}"
                if 'scale' in losses:
                    postfix["scale"] = f"{losses['scale']:.{7}f}"
                if 'dy_off' in losses:
                    postfix["dy_off"] = f"{losses['dy_off']:.{7}f}"
                if 'lap' in losses:
                    postfix["lap"] = f"{losses['lap']:.{7}f}"
                if 'dynamic_offset_std' in losses:
                    postfix["dynamic_offset_std"] = f"{losses['dynamic_offset_std']:.{7}f}"
                progress_bar.set_postfix(postfix)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, losses, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), mesh_renderer, render_meshes_iterations)
            if (iteration in saving_iterations):
                print("[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, losses, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, mesh_renderer, render_meshes_iterations):
    if tb_writer:
        tb_writer.add_scalar('1_train_loss_patches/l1_loss', losses['l1'].item(), iteration)
        tb_writer.add_scalar('1_train_loss_patches/ssim_loss', losses['ssim'].item(), iteration)
        if 'xyz' in losses:
            tb_writer.add_scalar('1_train_loss_patches/xyz_loss', losses['xyz'].item(), iteration)
        if 'scale' in losses:
            tb_writer.add_scalar('1_train_loss_patches/scale_loss', losses['scale'].item(), iteration)
        if 'dynamic_offset' in losses:
            tb_writer.add_scalar('1_train_loss_patches/dynamic_offset', losses['dynamic_offset'].item(), iteration)
        if 'laplacian' in losses:
            tb_writer.add_scalar('1_train_loss_patches/laplacian', losses['laplacian'].item(), iteration)
        if 'dynamic_offset_std' in losses:
            tb_writer.add_scalar('1_train_loss_patches/dynamic_offset_std', losses['dynamic_offset_std'].item(), iteration)
        if 'offset_lap' in losses:
            tb_writer.add_scalar('1_train_loss_patches/offset_lap_mse', losses['offset_lap'].item(), iteration)
        if 'offset_norm' in losses:
            tb_writer.add_scalar('1_train_loss_patches/offset_norm', losses['offset_norm'].item(), iteration)

        tb_writer.add_scalar('1_train_loss_patches/total_loss', losses['total'].item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

        if iteration % 100 == 0:
            tb_writer.add_scalar('1_transform/rotation',
                                 scene.gaussians.model_params['mesh_rotation'].norm(dim=-1).sum(),
                                 iteration)
            tb_writer.add_scalar('1_transform/scale',
                                 scene.gaussians.model_params['mesh_scale'].norm(dim=-1).sum(), iteration)
            tb_writer.add_scalar('1_transform/translation',
                                 scene.gaussians.model_params['mesh_translation'].norm(dim=-1).sum(),
                                 iteration)
            tb_writer.add_scalar('1_transform/bs_weights',
                                 scene.gaussians.model_params['bs_weights'].abs().sum(),
                                 iteration)
            tb_writer.add_scalar('1_transform/static_offset_norm_sum',
                                 scene.gaussians.model_params['static_offset'].norm(),
                                 iteration)
            tb_writer.add_scalar('1_transform/dynamic_offset_norm_sum',
                                 scene.gaussians.model_params['dynamic_offset'].norm(dim=-1).sum(),
                                 iteration)

    # if tb_writer:
    #     for viewpoint in scene.getValCameras():
    #         if viewpoint.timestep in [4, 31, 62, 134, 164, 224, 279, 399] and iteration in [1,50,100,150,200,250,300,500,1000,2000,5000,10000,20000,50000,100000,300000,600000]:
    #             out_dict = mesh_renderer.render_from_camera(scene.gaussians.verts, scene.gaussians.faces, camera2miniCam(viewpoint)), 0.0, 1.0
    #             rgba_mesh = torch.clamp(out_dict['rgba'].squeeze(0), 0.0, 1.0)  # (H, W, C)
    #             rgb_mesh = rgba_mesh[:, :, :3]
    #             alpha_mesh = rgba_mesh[:, :, 3:]
    #             mesh_opacity = torch.tensor([0.8]).cuda()
    #             rgb_gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)[:,:,:3]

    #             rgb = rgb_mesh * alpha_mesh * mesh_opacity  + rgb_gt_image * (alpha_mesh * (1 - mesh_opacity) + (1 - alpha_mesh))
    #             tb_writer.add_images(f"cam{viewpoint.uid}_timestep{viewpoint.timestep}", rgb[None], global_step=iteration)

    if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
            # tb_writer.add_scalar('static_offset_total', torch.sum(torch.norm(scene.gaussians.model_params['static_offset'])).cpu().numpy(), iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        print("[ITER {}] Evaluating".format(iteration))
        torch.cuda.empty_cache()
        validation_configs = (
            {'name': 'val', 'cameras' : scene.getValCameras()},
            {'name': 'test', 'cameras' : scene.getTestCameras()},
        )

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0
                num_vis_img = 10
                image_cache = []
                gt_image_cache = []
                vis_ct = 0
                for idx, viewpoint in tqdm(enumerate(DataLoader(config['cameras'], shuffle=False, batch_size=None, num_workers=8)), total=len(config['cameras'])):
                    if scene.gaussians.num_timesteps > 1:
                        scene.gaussians.select_mesh_by_timestep(viewpoint.timestep)
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx % (len(config['cameras']) // num_vis_img) == 0):
                        tb_writer.add_images(config['name'] + "_{}/render".format(vis_ct), image[None], global_step=iteration)
                        error_image = error_map(image, gt_image)
                        tb_writer.add_images(config['name'] + "_{}/error".format(vis_ct), error_image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_{}/ground_truth".format(vis_ct), gt_image[None], global_step=iteration)
                        vis_ct += 1
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()

                    image_cache.append(image)
                    gt_image_cache.append(gt_image)

                    if idx == len(config['cameras']) - 1 or len(image_cache) == 16:
                        batch_img = torch.stack(image_cache, dim=0)
                        batch_gt_img = torch.stack(gt_image_cache, dim=0)
                        lpips_test += lpips(batch_img, batch_gt_img).sum().double()
                        image_cache = []
                        gt_image_cache = []

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                lpips_test /= len(config['cameras'])          
                ssim_test /= len(config['cameras'])          
                print("[ITER {}] Evaluating {}: L1 {:.4f} PSNR {:.4f} SSIM {:.4f} LPIPS {:.4f}".format(iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)

    visible_train_cams = [2, 0, 13, 14, 12]
    mesh_cam = 13
    mesh_gt_overlay_cams = [2, 0, 13, 14, 12] # displayed in tb in same order from left to right
    mesh_gt_overlay_timesteps = [4, 223, 100, 200, 300, 400] # train only
    n = 100
    if tb_writer and (iteration in render_meshes_iterations or iteration in testing_iterations):
        # iterate once
        mesh_overlay_images = {t:dict() for t in mesh_gt_overlay_timesteps}
        relevant_cams = scene.getTrainCamerasWithIds(visible_train_cams + [mesh_cam] + mesh_gt_overlay_cams)
        for idx, viewpoint in tqdm(enumerate(DataLoader(relevant_cams, shuffle=False, batch_size=None, num_workers=8)), total=len(relevant_cams), desc=f"[ITER {iteration}] Rendering train, meshes and mesh overlays..."):
            scene.gaussians.select_mesh_by_timestep(viewpoint.timestep)
            # render train
            if iteration in testing_iterations:
                if viewpoint.colmap_id in visible_train_cams and viewpoint.timestep % n == 0:
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    tb_writer.add_images(f"train_c{viewpoint.colmap_id}_t{viewpoint.timestep}/render", image[None],
                                        global_step=iteration)
                    error_image = error_map(image, gt_image)
                    tb_writer.add_images(f"train_c{viewpoint.colmap_id}_t{viewpoint.timestep}/error", error_image[None],
                                        global_step=iteration)
                    if iteration == testing_iterations[0]:
                        tb_writer.add_images(f"train_c{viewpoint.colmap_id}_t{viewpoint.timestep}/ground_truth", gt_image[None],
                                            global_step=iteration)
            # render gt overlaid with mesh
            if iteration in render_meshes_iterations:
                #render mesh to visualize offsets
                if viewpoint.colmap_id == mesh_cam and viewpoint.timestep % n == 0:
                    # export mesh render
                    out_dict = mesh_renderer.render_from_camera(scene.gaussians.verts, scene.gaussians.faces, viewpoint)
                    rgba_mesh = out_dict['rgba'].squeeze(0)  # (H, W, C)
                    rgb_mesh = rgba_mesh[:, :, :3]
                    image=rgb_mesh.permute(2,0,1)
                    tb_writer.add_images(f"1_mesh/mesh_c{viewpoint.colmap_id}_t{viewpoint.timestep}", image[None],
                                    global_step=iteration)
                if viewpoint.colmap_id in mesh_gt_overlay_cams and viewpoint.timestep in mesh_gt_overlay_timesteps:
                    # get gt image
                    gt_image = viewpoint.original_image  # Assuming this is a PIL image or convertible
                    gt_image = gt_image.permute(1, 2, 0).cuda()

                    # get mesh image
                    out_dict = mesh_renderer.render_from_camera(scene.gaussians.verts, scene.gaussians.faces, viewpoint)
                    rgba_mesh = out_dict['rgba'].squeeze(0)  # (H, W, C)
                    rgb_mesh = rgba_mesh[:, :, :3]
                    alpha_mesh = rgba_mesh[:, :, 3:]
                    mesh_opacity = torch.tensor(0.5)

                    # aplpha blend
                    final = rgb_mesh * alpha_mesh * mesh_opacity + gt_image * (alpha_mesh * (1 - mesh_opacity) + (1 - alpha_mesh))
                    final = final.permute(2,0,1)
                    # final = Image.fromarray((final * 255).clip(0, 255).cpu().numpy().astype(np.uint8))
                    # tb_writer.add_images(f"1_mesh/1_overlay_c{viewpoint.colmap_id}_t{viewpoint.timestep}", final[None],
                    #                     global_step=iteration)
                    mesh_overlay_images[viewpoint.timestep][viewpoint.colmap_id] = final
        
        if iteration in render_meshes_iterations:
            for t, image_dict in mesh_overlay_images.items():
                if len(image_dict) != 0: # if the timestep is not in train cams and thus no img were generated
                    images = [image_dict[k] for k in mesh_gt_overlay_cams]
                    tb_writer.add_images(f"1_mesh/1_overlay_c{mesh_gt_overlay_cams}_t{t}", torch.stack(images)
                                        , global_step=iteration)

        torch.cuda.empty_cache()

def save_params_to_json(lp, op, pp, args, folder, filename="params_ga.json"):
    # Ensure the output folder exists
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)

    # Use `vars()` to get the dictionary of each parameter group
    params_dict = {
        "ModelParams": vars(lp.extract(args)),
        "OptimizationParams": vars(op.extract(args)),
        "PipelineParams": vars(pp.extract(args)),
        "GeneralArgs": vars(args)
    }

    # Write the dictionary to a JSON file
    with open(filepath, 'w') as f:
        json.dump(params_dict, f, indent=4)

    print(f"Parameters saved to {filepath}")

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--interval", type=int, default=60_000, help="A shared iteration interval for test and saving results and checkpoints.")
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    # custom
    parser.add_argument("--render_meshes_iterations", nargs="+", type=int, default=[])
    args = parser.parse_args(sys.argv[1:])

    if args.interval > op.iterations:
        args.interval = op.iterations // 5
    if len(args.test_iterations) == 0:
        args.test_iterations.extend(list(range(args.interval, args.iterations+1, args.interval)))
    if len(args.save_iterations) == 0:
        args.save_iterations.extend(list(range(args.interval, args.iterations+1, args.interval)))
    if len(args.checkpoint_iterations) == 0:
        args.checkpoint_iterations.extend(list(range(args.interval, args.iterations+1, args.interval)))
    # custom
    if len(args.render_meshes_iterations) == 0:
        args.render_meshes_iterations.extend(list(range(args.interval, args.iterations+1, args.interval)))
    
    args.test_iterations          = [1] + list(range(0, 10001, op.densification_interval//2)) + [10000, 12000, 15000, 20000, 30000, 70000, 80000, 100000] + args.test_iterations
    # args.test_iterations          = [1, 1000, 5000, 10000, 20000, 30000] + args.test_iterations
    args.save_iterations          = [1] + args.save_iterations
    args.render_meshes_iterations = [1] + list(range(0, 10001, op.densification_interval//2)) + [10000, 12000, 15000, 20000, 30000, 70000, 80000, 100000]  + args.render_meshes_iterations
    # args.render_meshes_iterations = [1, 500, 1000, 2000, 5000, 10000, 15000, 20000, 25000 , 30000] + args.render_meshes_iterations

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Save params as json
    save_params_to_json(lp, op, pp, args, args.model_path, filename="params_ga.json")

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        lp.extract(args), 
        op.extract(args), 
        pp.extract(args), 
        args.test_iterations, 
        args.save_iterations, 
        args.checkpoint_iterations, 
        args.start_checkpoint, 
        args.debug_from,
        args.render_meshes_iterations)

    # All done
    print("\nTraining complete.")
