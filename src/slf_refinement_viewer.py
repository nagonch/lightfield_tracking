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

    def _to_display_image(self, image):
        if image is None:
            return None
        if torch.is_tensor(image):
            image = image.detach().float().cpu()
            if image.ndim == 2:
                image = image[..., None].repeat(1, 1, 3)
            image = torch.clamp(image, 0.0, 1.0).numpy()
        image = np.asarray(image)
        if image.dtype != np.uint8:
            image = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
        return image

    def _update_scene_images(self, rendered_image, target_image):
        rendered_np = self._to_display_image(rendered_image)
        target_np = self._to_display_image(target_image)
        if rendered_np is None or target_np is None:
            return

        h, w = rendered_np.shape[:2]
        fx = float(max(w, 1))
        fov = float(2.0 * np.arctan2(w / 2.0, fx))

        pose_rendered = np.eye(4, dtype=np.float32)
        pose_rendered[:3, 3] = np.array([-0.08, -0.02, 0.02], dtype=np.float32)

        pose_target = np.eye(4, dtype=np.float32)
        pose_target[:3, 3] = np.array([0.08, -0.02, 0.02], dtype=np.float32)

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

    @torch.no_grad()
    def update(
        self, transformed_values, loss_value, iteration, rendered_image, target_image
    ):
        if not self.enabled:
            return
        if iteration % self.update_every != 0 and iteration != 0:
            return

        means = transformed_values["means"].detach().float()
        harmonics = transformed_values["harmonics"].detach().float()
        rotations = transformed_values["rotations"].detach().float()
        scales = transformed_values["scales"].detach().float()
        opacities = transformed_values["opacities"].detach().float()

        self.viewer.lock.acquire()
        self._means = means
        self._harmonics = harmonics
        self._rotations = rotations
        self._scales = scales
        self._opacities = opacities
        self.viewer.lock.release()

        self.iteration_handle.value = int(iteration)
        self.loss_handle.value = float(loss_value)
        self.num_gaussians_handle.value = int(means.shape[0])
        self._update_scene_images(rendered_image, target_image)

        self.viewer.rerender(None)

    def close(self):
        if not self.enabled:
            return
        self.server.scene.reset()
        self.server.stop()
