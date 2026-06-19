"""Viser demo — GT vs RealSense-synth vs our LF plane-sweep depth, as point clouds.

Loads one sequence, back-projects all three depths (gt / synth / ours) into the
central-camera frame, and shows them overlaid with a frame slider so you can scrub
through the sequence.  "Ours" is computed live by the LF plane-sweep estimator
(no pre-write needed); gt and synth are read from disk.

Run inside the lift6dof container, then open http://localhost:8080 :
    docker exec -e CUDA_VISIBLE_DEVICES=0 -w "$PWD" lift6dof python demo_depth.py
Pick the sequence in __main__ below.
"""

import os
import time

import numpy as np
import torch
import viser
from PIL import Image

from lf_depth import LFPlaneSweepDepth, _backproject, _central_mask, _turbo
from src.dataset import LFDataset
from utils import linear_to_srgb


def run(seq_path: str, port: int = 8080):
    ds = LFDataset(seq_path, depth_source="gt")
    est = LFPlaneSweepDepth()
    sy_dir = os.path.join(seq_path, "depth_synth")
    sy_fnames = sorted(os.listdir(sy_dir))
    n = len(ds)

    cache: dict[int, dict] = {}

    def frame_data(idx: int) -> dict:
        """Compute/load all three depths for one frame (cached on first view)."""
        if idx not in cache:
            fr = ds[idx]
            S, T, H, W, _ = fr["LF"].shape
            cmid = (S // 2) * T + (T // 2)
            mask = _central_mask(fr)
            ours = est.estimate(fr, mask=mask)["depth"].cpu().numpy()
            synth = (
                np.array(Image.open(os.path.join(sy_dir, sy_fnames[idx]))).astype(
                    np.float32
                )
                / 1000.0
            )
            rgb = (
                (linear_to_srgb(fr["LF"].reshape(-1, H, W, 3)[cmid]).clamp(0, 1) * 255)
                .to(torch.uint8)
                .cpu()
                .numpy()
            )
            cache[idx] = {
                "gt": fr["depth"].cpu().numpy(),
                "synth": synth,
                "ours": ours,
                "rgb": rgb,
                "mask": mask.cpu().numpy(),
                "K": fr["camera_matrix"][:3, :3].cpu().numpy(),
            }
        return cache[idx]

    server = viser.ViserServer(port=port)
    server.scene.set_up_direction("-y")
    slider = server.gui.add_slider("Frame", 0, n - 1, 1, 0)
    show_gt = server.gui.add_checkbox("GT (green)", True)
    show_ours = server.gui.add_checkbox("Ours (rgb)", True)
    show_synth = server.gui.add_checkbox("Synth (red)", True)
    obj_only = server.gui.add_checkbox("Object only", True)
    color_err = server.gui.add_checkbox("Colour ours by error", False)
    pt_size = server.gui.add_slider("Point size", 0.0005, 0.01, 0.0005, 0.002)
    info = server.gui.add_text("Object MAE vs GT", initial_value="")

    def render():
        d = frame_data(int(slider.value))
        K, gt, ours, synth, rgb = d["K"], d["gt"], d["ours"], d["synth"], d["rgb"]
        mask = d["mask"] if obj_only.value else np.ones_like(d["mask"])
        server.scene.reset()

        if show_gt.value:
            gp, gy, gx = _backproject(gt, mask, K)
            if len(gp):
                col = np.tile(np.array([[40, 200, 80]], np.uint8), (len(gp), 1))
                server.scene.add_point_cloud(
                    "/gt", gp, col, point_size=float(pt_size.value)
                )
        if show_ours.value:
            op, oy, ox = _backproject(ours, mask, K)
            if len(op):
                col = (
                    _turbo(np.abs(ours[oy, ox] - gt[oy, ox]) * 1000 / 20.0)
                    if color_err.value
                    else rgb[oy, ox]
                )
                server.scene.add_point_cloud(
                    "/ours", op, col, point_size=float(pt_size.value)
                )
        if show_synth.value:
            sp, sy, sx = _backproject(synth, mask, K)
            if len(sp):
                col = np.tile(np.array([[220, 60, 60]], np.uint8), (len(sp), 1))
                server.scene.add_point_cloud(
                    "/synth", sp, col, point_size=float(pt_size.value)
                )

        m = d["mask"] & (gt > 1e-3)
        ours_mae = np.abs(ours[m] - gt[m]).mean() * 1000 if m.any() else float("nan")
        sv = m & (synth > 1e-3)
        synth_mae = (
            np.abs(synth[sv] - gt[sv]).mean() * 1000 if sv.any() else float("nan")
        )
        synth_cov = sv.sum() / max(m.sum(), 1) * 100
        info.value = (
            f"frame {int(slider.value)}/{n - 1}   "
            f"ours {ours_mae:.1f}mm (100%)   "
            f"synth {synth_mae:.1f}mm ({synth_cov:.0f}%)"
        )

    for ctrl in (slider, show_gt, show_ours, show_synth, obj_only, color_err, pt_size):
        ctrl.on_update(lambda _: render())
    render()
    # viser picks the next free port if the requested one is busy — report the real one.
    print(
        f"viser at http://localhost:{server.get_port()}   ({n} frames)  —  {seq_path}"
    )
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    # Pick any sequence.  cube_1.0 (pure mirror) is the most dramatic: synth is
    # holey and ~1 m off, ours is a clean metric surface.
    SEQ = "/home/ngoncharov/SpecTrack_dataset/objects_0.0/bleach0"
    run(SEQ, port=8080)
