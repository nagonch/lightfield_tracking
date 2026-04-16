import math
from pathlib import Path

import numpy as np
import torch
import viser
from gsplat import rasterization

from .gaussian_splatting.gsplat_viewer import GsplatViewer


class SurfaceLFRefinementViewer:
    def __init__(self, enabled: bool = True, update_every: int = 10):
        self.enabled = enabled
        self.update_every = max(1, int(update_every))
        self._closed = False
        self._means = None
        self._harmonics = None
        self._rotations = None
        self._scales = None
        self._opacities = None

        if not self.enabled:
            self.server = None
            self.viewer = None
            self.iteration_handle = None
            self.loss_handle = None
            self.num_gaussians_handle = None
            return

        self.server = viser.ViserServer(verbose=False)

        @torch.no_grad()
        def viewer_render_fn(camera_state, render_tab_state):
            if self._means is None or self._means.shape[0] == 0:
                h = int(getattr(render_tab_state, "viewer_height", 480))
                w = int(getattr(render_tab_state, "viewer_width", 640))
                return np.zeros((h, w, 3), dtype=np.float32)

            if render_tab_state.preview_render:
                width = render_tab_state.render_width
                height = render_tab_state.render_height
            else:
                width = render_tab_state.viewer_width
                height = render_tab_state.viewer_height

            c2w = torch.from_numpy(camera_state.c2w).float().to(self._means.device)
            K = (
                torch.from_numpy(camera_state.get_K((width, height)))
                .float()
                .to(self._means.device)
            )
            viewmat = c2w.inverse()

            render_mode_map = {
                "rgb": "RGB",
                "depth(accumulated)": "D",
                "depth(expected)": "ED",
                "alpha": "RGB",
            }
            render_tab_state.backgrounds = (255.0, 255.0, 255.0)
            sh_degree = int(math.sqrt(self._harmonics.shape[1]) - 1)

            render_colors, _, info = rasterization(
                self._means,
                self._rotations,
                self._scales,
                self._opacities,
                self._harmonics,
                viewmat[None],
                K[None],
                width,
                height,
                sh_degree=min(render_tab_state.max_sh_degree, sh_degree),
                near_plane=render_tab_state.near_plane,
                far_plane=render_tab_state.far_plane,
                radius_clip=render_tab_state.radius_clip,
                eps2d=render_tab_state.eps2d,
                backgrounds=torch.tensor([render_tab_state.backgrounds]).to(
                    self._means.device
                )
                / 255.0,
                render_mode=render_mode_map[render_tab_state.render_mode],
                rasterize_mode=render_tab_state.rasterize_mode,
                camera_model=render_tab_state.camera_model,
                packed=False,
            )

            render_tab_state.total_gs_count = len(self._means)
            render_tab_state.rendered_gs_count = (
                (info["radii"] > 0).all(-1).sum().item()
            )
            render_colors = torch.clip(render_colors[0, ..., :3], min=0.0, max=1.0)
            return render_colors.detach().cpu().numpy()

        self.viewer = GsplatViewer(
            server=self.server,
            render_fn=viewer_render_fn,
            output_dir=Path("/tmp/gs_output"),
            mode="rendering",
        )

        with self.server.gui.add_folder("Refinement"):
            self.iteration_handle = self.server.gui.add_number(
                "Iteration", initial_value=0, disabled=True
            )
            self.loss_handle = self.server.gui.add_number(
                "Loss", initial_value=0.0, disabled=True
            )
            self.num_gaussians_handle = self.server.gui.add_number(
                "Gaussians", initial_value=0, disabled=True
            )

    def _to_display_image(self, image, gamma=None):
        if image is None:
            return None
        if torch.is_tensor(image):
            image = image.detach().float().cpu()
            if image.ndim == 2:
                image = image[..., None].repeat(1, 1, 3)
            # Normalize to [0, 1] if needed (e.g., for loss maps)
            if gamma is not None:
                image = image - image.min()
                image = image / (image.max() + 1e-8)
                image = torch.pow(image, gamma)
            image = torch.clamp(image, 0.0, 1.0).numpy()
        image = np.asarray(image)
        if image.dtype != np.uint8:
            image = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
        return image

    def _to_display_env_map(self, environment_map):
        if environment_map is None:
            return None
        if torch.is_tensor(environment_map):
            environment_map = environment_map.detach().float().cpu()
        environment_map = np.asarray(environment_map)

        # Accept either HWC or CHW env maps.
        if environment_map.ndim == 3 and environment_map.shape[0] == 3:
            environment_map = np.transpose(environment_map, (1, 2, 0))
        elif environment_map.ndim != 3 or environment_map.shape[-1] != 3:
            return None

        if environment_map.dtype != np.uint8:
            environment_map = (np.clip(environment_map, 0.0, 1.0) * 255.0).astype(
                np.uint8
            )
        return environment_map

    def _update_scene_images(
        self, rendered_image, target_image, loss_image=None, environment_map=None
    ):
        rendered_np = self._to_display_image(rendered_image)
        target_np = self._to_display_image(target_image)
        env_map_np = self._to_display_env_map(environment_map)
        if rendered_np is None or target_np is None:
            return

        h, w = rendered_np.shape[:2]
        fx = float(max(w, 1))
        fov = float(2.0 * np.arctan2(w / 2.0, fx))

        # Use provided loss image or compute from difference
        if loss_image is not None:
            loss_np = self._to_display_image(loss_image, gamma=0.5)
        else:
            loss_np = np.abs(
                rendered_np.astype(np.float32) - target_np.astype(np.float32)
            )
            loss_np = (np.clip(loss_np, 0.0, 1.0) * 255.0).astype(np.uint8)

        pose_rendered = np.eye(4, dtype=np.float32)
        pose_rendered[:3, 3] = np.array([-0.08, -0.02, 0.02], dtype=np.float32)

        pose_target = np.eye(4, dtype=np.float32)
        pose_target[:3, 3] = np.array([0.08, -0.02, 0.02], dtype=np.float32)

        pose_loss = np.eye(4, dtype=np.float32)
        pose_loss[:3, 3] = np.array([0.0, 0.06, 0.02], dtype=np.float32)

        pose_env_map = np.eye(4, dtype=np.float32)
        pose_env_map[:3, 3] = np.array([0.24, -0.02, 0.02], dtype=np.float32)

        self.server.scene.add_camera_frustum(
            name="refinement/rendered",
            aspect=w / max(h, 1),
            fov=fov,
            scale=0.03,
            line_width=1.0,
            image=rendered_np,
            wxyz=(1.0, 0.0, 0.0, 0.0),
            position=pose_rendered[:3, 3],
        )
        self.server.scene.add_camera_frustum(
            name="refinement/target",
            aspect=w / max(h, 1),
            fov=fov,
            scale=0.03,
            line_width=1.0,
            image=target_np,
            wxyz=(1.0, 0.0, 0.0, 0.0),
            position=pose_target[:3, 3],
        )
        self.server.scene.add_camera_frustum(
            name="refinement/loss",
            aspect=w / max(h, 1),
            fov=fov,
            scale=0.03,
            line_width=1.0,
            image=loss_np,
            wxyz=(1.0, 0.0, 0.0, 0.0),
            position=pose_loss[:3, 3],
        )

        if env_map_np is not None:
            env_h, env_w = env_map_np.shape[:2]
            env_fx = float(max(env_w, 1))
            env_fov = float(2.0 * np.arctan2(env_w / 2.0, env_fx))
            self.server.scene.add_camera_frustum(
                name="refinement/environment_map",
                aspect=env_w / max(env_h, 1),
                fov=env_fov,
                scale=0.03,
                line_width=1.0,
                image=env_map_np,
                wxyz=(1.0, 0.0, 0.0, 0.0),
                position=pose_env_map[:3, 3],
            )

    @torch.no_grad()
    def update(
        self,
        transformed_values,
        loss_value,
        iteration,
        rendered_image,
        target_image,
        environment_map=None,
        loss_image=None,
    ):
        if not self.enabled:
            return
        if self._closed:
            return
        if iteration % self.update_every != 0 and iteration != 0:
            return

        means = transformed_values["means"].detach().float()
        harmonics = transformed_values["harmonics"].detach().float()
        rotations = transformed_values["rotations"].detach().float()
        scales = transformed_values["scales"].detach().float()
        opacities = transformed_values["opacities"].detach().float()

        try:
            self.viewer.lock.acquire()
            self._means = means
            self._harmonics = harmonics
            self._rotations = rotations
            self._scales = scales
            self._opacities = opacities
        finally:
            self.viewer.lock.release()

        try:
            self.iteration_handle.value = int(iteration)
            self.loss_handle.value = float(loss_value)
            self.num_gaussians_handle.value = int(means.shape[0])
            self._update_scene_images(
                rendered_image, target_image, loss_image, environment_map
            )
            self.viewer.rerender(None)
        except RuntimeError:
            self._closed = True

    def close(self):
        if not self.enabled:
            return
        self._closed = True
        try:
            self.server.scene.reset()
            self.server.stop()
        except RuntimeError:
            pass
