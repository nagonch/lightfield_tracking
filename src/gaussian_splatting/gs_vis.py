import argparse
import math
import os
import time

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import viser
from pathlib import Path
from gsplat._helper import load_test_data
from gsplat.distributed import cli
from gsplat.rendering import rasterization

from nerfview import CameraState, RenderTabState, apply_float_colormap
from gsplat_viewer import GsplatViewer, GsplatRenderTabState


def visualize_gs(means, colors, quats=None, scales=None, opacities=None):
    device = torch.device("cuda")
    means = means.float()
    colors = colors.float()
    sh_degree = int(math.sqrt(colors.shape[-2]) - 1)
    if quats is None:
        quats = torch.stack(
            [
                torch.tensor([1, 0, 0, 0]).cuda(),
            ]
            * means.shape[0]
        ).float()
    if scales is None:
        scales = torch.ones_like(means).float() / 1000
    if opacities is None:
        opacities = torch.ones_like(means[:, 0]).float()

    print("Number of Gaussians:", len(means))

    # register and open viewer
    @torch.no_grad()
    def viewer_render_fn(camera_state: CameraState, render_tab_state: RenderTabState):
        assert isinstance(render_tab_state, GsplatRenderTabState)
        if render_tab_state.preview_render:
            width = render_tab_state.render_width
            height = render_tab_state.render_height
        else:
            width = render_tab_state.viewer_width
            height = render_tab_state.viewer_height
        c2w = camera_state.c2w
        K = camera_state.get_K((width, height))
        c2w = torch.from_numpy(c2w).float().to(device)
        K = torch.from_numpy(K).float().to(device)
        viewmat = c2w.inverse()

        RENDER_MODE_MAP = {
            "rgb": "RGB",
            "depth(accumulated)": "D",
            "depth(expected)": "ED",
            "alpha": "RGB",
        }
        render_tab_state.backgrounds = (255.0, 255.0, 255.0)  # white background
        render_colors, render_alphas, info = rasterization(
            means,  # [N, 3]
            quats,  # [N, 4]
            scales,  # [N, 3]
            opacities,  # [N]
            colors,  # [N, S, 3]
            viewmat[None],  # [1, 4, 4]
            K[None],  # [1, 3, 3]
            width,
            height,
            sh_degree=(
                min(render_tab_state.max_sh_degree, sh_degree)
                if sh_degree is not None
                else None
            ),
            near_plane=render_tab_state.near_plane,
            far_plane=render_tab_state.far_plane,
            radius_clip=render_tab_state.radius_clip,
            eps2d=render_tab_state.eps2d,
            backgrounds=torch.tensor([render_tab_state.backgrounds], device=device)
            / 255.0,
            render_mode=RENDER_MODE_MAP[render_tab_state.render_mode],
            rasterize_mode=render_tab_state.rasterize_mode,
            camera_model=render_tab_state.camera_model,
            packed=False,
        )
        render_tab_state.total_gs_count = len(means)
        render_tab_state.rendered_gs_count = (info["radii"] > 0).all(-1).sum().item()
        render_colors = render_colors[0, ..., 0:3]
        render_colors = torch.clip(render_colors, min=0, max=1)
        renders = render_colors.cpu().numpy()
        return renders

    server = viser.ViserServer()
    _ = GsplatViewer(
        server=server,
        render_fn=viewer_render_fn,
        output_dir=Path("/gs_output"),
        mode="rendering",
    )
    print("Viewer running... Ctrl+C to exit.")
    time.sleep(100000)


if __name__ == "__main__":
    gaussians = torch.load("/home/ngoncharov/cvpr2026/LiFT-6DoF/gaussians.pt")
    visualize_gs(
        gaussians["means"].float(),
        gaussians["harmonics"].float(),
        quats=gaussians["rotations"].float(),
        scales=gaussians["scales"].float(),
        opacities=gaussians["opacities"].float(),
    )
