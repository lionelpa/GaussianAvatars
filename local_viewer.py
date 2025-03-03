# 
# Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual 
# property and proprietary rights in and to this software and related documentation. 
# Any commercial use, reproduction, disclosure or distribution of this software and 
# related documentation without an express license agreement from Toyota Motor Europe NV/SA 
# is strictly prohibited.
#

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import dearpygui.dearpygui as dpg
import matplotlib
import numpy as np
import torch
import tyro
from PIL import Image

from gaussian_renderer import render
from mesh_renderer import NVDiffRenderer
from scene.wb_gaussian_model import WBGaussianModel
from utils.viewer_utils import Mini3DViewer, Mini3DViewerConfig


@dataclass
class PipelineConfig:
    debug: bool = False
    compute_cov3D_python: bool = False
    convert_SHs_python: bool = False


@dataclass
class Config(Mini3DViewerConfig):
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    """Pipeline settings for gaussian splatting rendering"""
    cam_convention: Literal["opengl", "opencv"] = "opencv"
    """Camera convention"""
    point_path: Optional[Path] = None
    """Path to the gaussian splatting file"""
    motion_path: Optional[Path] = None
    """Path to the motion file (npz)"""
    sh_degree: int = 3
    """Spherical Harmonics degree"""
    background_color: tuple[float] = (1., 1., 1.)
    """default GUI background color"""
    save_folder: Path = Path("./viewer_output")
    """default saving folder"""
    fps: int = 25
    """default fps for recording"""
    keyframe_interval: int = 1
    """default keyframe interval"""
    ref_json: Optional[Path] = None
    """ Path to a reference json file. We copy file paths from a reference json into 
    the exported trajectory json file as placeholders so that `render.py` can directly
    load it like a normal sequence. """
    demo_mode: bool = False
    """The UI will be simplified in demo mode."""

class LocalViewer(Mini3DViewer):
    def __init__(self, cfg: Config):
        self.cfg = cfg
        
        # recording settings
        self.keyframes = []  # list of state dicts of keyframes
        self.all_frames = {}  # state dicts of all frames {key: [num_frames, ...]}
        self.num_record_timeline = 0
        self.playing = False
        self.loop = False

        print("Initializing 3D Gaussians...")
        self.init_gaussians()

        if self.gaussians.binding is not None:
            # rendering settings
            self.mesh_color = torch.tensor([1, 1, 1, 0.5])
            self.face_colors = None
            print("Initializing mesh renderer...")
            self.mesh_renderer = NVDiffRenderer(use_opengl=False)
        
        # FLAME parameters
        super().__init__(cfg, 'GaussianAvatars - Local Viewer')

        if self.gaussians.binding is not None:
            self.num_timesteps = self.gaussians.num_timesteps
            self.min_timestep = self.gaussians.min_timestep
            self.max_timestep = self.gaussians.max_timestep
            dpg.configure_item("_slider_timestep", min_value=self.min_timestep, max_value=self.max_timestep)

            self.gaussians.select_mesh_by_timestep(self.min_timestep)

    def init_gaussians(self):
        # load gaussians
        self.gaussians = WBGaussianModel(True, self.cfg.sh_degree)
        unselected_fid = []
        
        if self.cfg.point_path is not None:
            if self.cfg.point_path.exists():
                self.gaussians.load_ply(self.cfg.point_path, has_target=False, motion_path=self.cfg.motion_path, disable_fid=unselected_fid)
            else:
                raise FileNotFoundError(f'{self.cfg.point_path} does not exist.')

    def refresh_stat(self):
        if self.last_time_fresh is not None:
            elapsed = time.time() - self.last_time_fresh
            fps = 1 / elapsed
            dpg.set_value("_log_fps", f'{int(fps):<4d}')
        self.last_time_fresh = time.time()


    def define_gui(self):
        super().define_gui()

        # window: rendering options ==================================================================================================
        with dpg.window(label="Render", tag="_render_window", autosize=True):

            with dpg.group(horizontal=True):
                dpg.add_text("FPS:", show=not self.cfg.demo_mode)
                dpg.add_text("0   ", tag="_log_fps", show=not self.cfg.demo_mode)

            dpg.add_text(f"number of points: {self.gaussians._xyz.shape[0]}")
            
            with dpg.group(horizontal=True):
                # show splatting
                def callback_show_splatting(sender, app_data):
                    self.need_update = True
                dpg.add_checkbox(label="show splatting", default_value=True, callback=callback_show_splatting, tag="_checkbox_show_splatting")

                dpg.add_spacer(width=10)

                if self.gaussians.binding is not None:
                    # show mesh
                    def callback_show_mesh(sender, app_data):
                        self.need_update = True
                    dpg.add_checkbox(label="show mesh", default_value=False, callback=callback_show_mesh, tag="_checkbox_show_mesh")

                    # # show original mesh
                    # def callback_original_mesh(sender, app_data):
                    #     self.original_mesh = app_data
                    #     self.need_update = True
                    # dpg.add_checkbox(label="original mesh", default_value=self.original_mesh, callback=callback_original_mesh)
            
            # timestep slider and buttons
            if self.num_timesteps != None:
                def callback_set_current_frame(sender, app_data):
                    if sender == "_slider_timestep":
                        self.timestep = app_data
                    elif sender in ["_button_timestep_plus", "_mvKey_Right"]:
                        self.timestep = min(self.timestep + 1, self.num_timesteps - 1)
                    elif sender in ["_button_timestep_minus", "_mvKey_Left"]:
                        self.timestep = max(self.timestep - 1, 0)
                    elif sender == "_mvKey_Home":
                        self.timestep = 0
                    elif sender == "_mvKey_End":
                        self.timestep = self.num_timesteps - 1

                    dpg.set_value("_slider_timestep", self.timestep)
                    self.gaussians.select_mesh_by_timestep(self.timestep)

                    self.need_update = True

                self.timestep = 0
                for i in range(100):
                    try:
                        self.gaussians.select_mesh_by_timestep(i)
                        self.timestep= i
                        print(f"Loaded initial mesh as timestep {self.timestep}")
                        break
                    except:
                        pass

                with dpg.group(horizontal=True):
                    dpg.add_button(label='-', tag="_button_timestep_minus", callback=callback_set_current_frame)
                    dpg.add_button(label='+', tag="_button_timestep_plus", callback=callback_set_current_frame)
                    dpg.add_slider_int(label="timestep", tag='_slider_timestep', width=153, min_value=self.timestep, max_value=self.num_timesteps - 1, format="%d", default_value=self.timestep, callback=callback_set_current_frame)

            # # render_mode combo
            # def callback_change_mode(sender, app_data):
            #     self.render_mode = app_data
            #     self.need_update = True
            # dpg.add_combo(('rgb', 'depth', 'opacity'), label='render mode', default_value=self.render_mode, callback=callback_change_mode)

            # scaling_modifier slider
            def callback_set_scaling_modifier(sender, app_data):
                self.need_update = True
            dpg.add_slider_float(label="Scale modifier", min_value=0, max_value=1, format="%.2f", width=200, default_value=1, callback=callback_set_scaling_modifier, tag="_slider_scaling_modifier")

            # fov slider
            def callback_set_fovy(sender, app_data):
                self.cam.fovy = app_data
                self.need_update = True
            dpg.add_slider_int(label="FoV (vertical)", min_value=1, max_value=120, width=200, format="%d deg", default_value=self.cam.fovy, callback=callback_set_fovy, tag="_slider_fovy", show=not self.cfg.demo_mode)

            if self.gaussians.binding is not None:
                # visualization options
                def callback_visual_options(sender, app_data):
                    if app_data == 'number of points per face':
                        value, ct = self.gaussians.binding.unique(return_counts=True)
                        ct = torch.log10(ct + 1)
                        ct = ct.float() / ct.max()
                        cmap = matplotlib.colormaps["plasma"]
                        self.face_colors = torch.from_numpy(cmap(ct.cpu())[None, :, :3]).to(self.gaussians.verts)
                    else:
                        self.face_colors = self.mesh_color[:3].to(self.gaussians.verts)[None, None, :].repeat(1, self.gaussians.face_center.shape[0], 1)  # (1, F, 3)
                    
                    dpg.set_value('_checkbox_show_mesh', True)
                    self.need_update = True
                dpg.add_combo(["none", "number of points per face"], default_value="none", label='visualization', width=200, callback=callback_visual_options, tag="_visual_options")

                # mesh_color picker
                def callback_change_mesh_color(sender, app_data):
                    self.mesh_color = torch.tensor(app_data, dtype=torch.float32)  # only need RGB in [0, 1]
                    if dpg.get_value("_visual_options") == 'none':
                        self.face_colors = self.mesh_color[:3].to(self.gaussians.verts)[None, None, :].repeat(1, self.gaussians.face_center.shape[0], 1)
                    self.need_update = True
                dpg.add_color_edit((self.mesh_color*255).tolist(), label="Mesh Color", width=200, callback=callback_change_mesh_color, show=not self.cfg.demo_mode)

            # # bg_color picker
            # def callback_change_bg(sender, app_data):
            #     self.bg_color = torch.tensor(app_data[:3], dtype=torch.float32)  # only need RGB in [0, 1]
            #     self.need_update = True
            # dpg.add_color_edit((self.bg_color*255).tolist(), label="Background Color", width=200, no_alpha=True, callback=callback_change_bg)

            # # near slider
            # def callback_set_near(sender, app_data):
            #     self.cam.znear = app_data
            #     self.need_update = True
            # dpg.add_slider_int(label="near", min_value=1e-8, max_value=2, format="%.2f", default_value=self.cam.znear, callback=callback_set_near, tag="_slider_near")

            # # far slider
            # def callback_set_far(sender, app_data):
            #     self.cam.zfar = app_data
            #     self.need_update = True
            # dpg.add_slider_int(label="far", min_value=1e-3, max_value=10, format="%.2f", default_value=self.cam.zfar, callback=callback_set_far, tag="_slider_far")
            
            # camera
            with dpg.group(horizontal=True):
                def callback_reset_camera(sender, app_data):
                    self.cam.reset()
                    self.need_update = True
                    dpg.set_value("_slider_fovy", self.cam.fovy)
                dpg.add_button(label="reset camera", tag="_button_reset_pose", callback=callback_reset_camera, show=not self.cfg.demo_mode)
                
                def callback_cache_camera(sender, app_data):
                    self.cam.save()
                dpg.add_button(label="cache camera", tag="_button_cache_pose", callback=callback_cache_camera, show=not self.cfg.demo_mode)

                def callback_clear_cache(sender, app_data):
                    self.cam.clear()
                dpg.add_button(label="clear cache", tag="_button_clear_cache", callback=callback_clear_cache, show=not self.cfg.demo_mode)

            def callback_set_record_timestep(sender, app_data):
                state_dict = self.get_state_dict_record()
                
                self.apply_state_dict(state_dict)
                self.need_update = True
            dpg.add_slider_int(label="timeline", tag='_slider_record_timestep', width=200, min_value=0, max_value=0, format="%d", default_value=0, callback=callback_set_record_timestep)
            
            with dpg.group(horizontal=True):
                def callback_play(sender, app_data):
                    self.playing = not self.playing
                    self.need_update = True
                dpg.add_button(label="play", tag="_button_play", callback=callback_play)

                def callback_loop(sender, app_data):
                    self.loop = dpg.get_value(sender)
                    # self.need_update = True
                dpg.add_checkbox(label="loop", tag="_checkbox_loop", callback=callback_loop)
            
            def callback_save_image(sender, app_data):
                if not self.cfg.save_folder.exists():
                    self.cfg.save_folder.mkdir(parents=True)
                path = self.cfg.save_folder / f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_{self.timestep}.png"
                print(f"Saving image to {path}")
                Image.fromarray((np.clip(self.render_buffer, 0, 1) * 255).astype(np.uint8)).save(path)
            with dpg.group(horizontal=True):
                dpg.add_button(label="save image", tag="_button_save_image", callback=callback_save_image)


        # widget-dependent handlers ========================================================================================
        with dpg.handler_registry():
            dpg.add_key_press_handler(dpg.mvKey_Left, callback=callback_set_current_frame, tag='_mvKey_Left')
            dpg.add_key_press_handler(dpg.mvKey_Right, callback=callback_set_current_frame, tag='_mvKey_Right')
            dpg.add_key_press_handler(dpg.mvKey_Home, callback=callback_set_current_frame, tag='_mvKey_Home')
            dpg.add_key_press_handler(dpg.mvKey_End, callback=callback_set_current_frame, tag='_mvKey_End')

            def callbackmouse_wheel_slider(sender, app_data):
                delta = app_data
                if dpg.is_item_hovered("_slider_timestep"):
                    self.timestep = min(max(self.timestep - delta, 0), self.max_timestep)
                    dpg.set_value("_slider_timestep", self.timestep)
                    self.gaussians.select_mesh_by_timestep(self.timestep)
                    self.need_update = True
            dpg.add_mouse_wheel_handler(callback=callbackmouse_wheel_slider)

    def prepare_camera(self):
        @dataclass
        class Cam:
            FoVx = float(np.radians(self.cam.fovx))
            FoVy = float(np.radians(self.cam.fovy))
            image_height = self.cam.image_height
            image_width = self.cam.image_width
            world_view_transform = torch.tensor(self.cam.world_view_transform).float().cuda().T  # the transpose is required by gaussian splatting rasterizer
            full_proj_transform = torch.tensor(self.cam.full_proj_transform).float().cuda().T  # the transpose is required by gaussian splatting rasterizer
            camera_center = torch.tensor(self.cam.pose[:3, 3]).cuda()
        return Cam

    @torch.no_grad()
    def run(self):
        print("Running LocalViewer...")

        while dpg.is_dearpygui_running():

            if self.need_update or self.playing:
                self.rendering = True
                cam = self.prepare_camera()

                if dpg.get_value("_checkbox_show_splatting"):
                    # rgb
                    rgb_splatting = render(cam, self.gaussians, self.cfg.pipeline, torch.tensor(self.cfg.background_color).cuda(), scaling_modifier=dpg.get_value("_slider_scaling_modifier"))["render"].permute(1, 2, 0).contiguous()

                    # opacity
                    # override_color = torch.ones_like(self.gaussians._xyz).cuda()
                    # background_color = torch.tensor(self.cfg.background_color).cuda() * 0
                    # rgb_splatting = render(cam, self.gaussians, self.cfg.pipeline, background_color, scaling_modifier=dpg.get_value("_slider_scaling_modifier"), override_color=override_color)["render"].permute(1, 2, 0).contiguous()

                if self.gaussians.binding is not None and dpg.get_value("_checkbox_show_mesh"):
                    out_dict = self.mesh_renderer.render_from_camera(self.gaussians.verts, self.gaussians.faces, cam, face_colors=self.face_colors)

                    rgba_mesh = out_dict['rgba'].squeeze(0)  # (H, W, C)
                    rgb_mesh = rgba_mesh[:, :, :3]
                    alpha_mesh = rgba_mesh[:, :, 3:]
                    mesh_opacity = self.mesh_color[3:].cuda()

                if dpg.get_value("_checkbox_show_splatting") and dpg.get_value("_checkbox_show_mesh"):
                    rgb = rgb_mesh * alpha_mesh * mesh_opacity  + rgb_splatting * (alpha_mesh * (1 - mesh_opacity) + (1 - alpha_mesh))
                elif dpg.get_value("_checkbox_show_splatting") and not dpg.get_value("_checkbox_show_mesh"):
                    rgb = rgb_splatting
                elif not dpg.get_value("_checkbox_show_splatting") and dpg.get_value("_checkbox_show_mesh"):
                    rgb = rgb_mesh
                else:
                    rgb = torch.ones([self.H, self.W, 3])

                self.render_buffer = rgb.cpu().numpy()
                dpg.set_value("_texture", self.render_buffer)

                self.refresh_stat()
                self.rendering = False
                self.need_update = False

                if self.playing:
                    if self.loop:
                        self.timestep = max(self.min_timestep, (self.timestep + 1) % self.max_timestep)
                    else:
                        self.timestep = min(self.timestep + 1, self.max_timestep )
                    dpg.set_value("_slider_timestep", self.timestep)
                    self.gaussians.select_mesh_by_timestep(self.timestep)

            dpg.render_dearpygui_frame()


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    gui = LocalViewer(cfg)
    gui.run()
