"""Live viser visualiser for ReLiFT-6DoF tracking.

Open http://localhost:8080 in a browser after launching.

3D scene layout
---------------
  /world                         world-frame axis gizmo
  /canonical/points              canonical PC at origin (object frame)
  /canonical/env_sphere          env map textured onto a UV sphere  (0, 0, 0.7)
  /canonical/env_flat            env map equirectangular panel       (2.2, 0, 0)
  /trajectory                    object translation trail (dot cloud)
  /frames/fNNNN/pts              per-frame observed PC   (plasma: purple=early, yellow=late)
  /frames/fNNNN/canonical_overlay canonical PC at this pose (subtle tint)
  /frames/fNNNN/pose             estimated pose axis gizmo
  /frames/fNNNN/diffuse          diffuse input image panel   (left of pose)
  /frames/fNNNN/rendered         canonical-relit render panel (right of pose)
  /refinement/compare            rendered vs target side-by-side (live)
  /refinement/loss_plot          log-scale loss + Δ-loss curves   (live)

GUI sidebar
-----------
  Tracking Status   frame index, canonical pts, last refinement loss
  Live Refinement   iteration counter, current loss, status text
  Legend            colour / panel explanation
"""

from __future__ import annotations

import threading
from io import BytesIO
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from scipy.spatial.transform import Rotation
import viser


# ── UV sphere geometry ─────────────────────────────────────────────────────────

def _uv_sphere_mesh(
    radius: float = 0.25, n_lat: int = 36, n_lon: int = 72
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(vertices [V,3] f32, faces [F,3] u32, uvs [V,2] f32)."""
    lats = np.linspace(np.pi / 2, -np.pi / 2, n_lat, dtype=np.float32)
    lons = np.linspace(-np.pi, np.pi, n_lon, dtype=np.float32)
    verts, uvs = [], []
    for la in lats:
        for lo in lons:
            verts.append([
                radius * np.cos(la) * np.cos(lo),
                radius * np.cos(la) * np.sin(lo),
                radius * np.sin(la),
            ])
            uvs.append([
                (lo + np.pi) / (2.0 * np.pi),
                (np.pi / 2.0 - la) / np.pi,
            ])
    verts = np.array(verts, dtype=np.float32)
    uvs   = np.array(uvs,   dtype=np.float32)
    faces = []
    for i in range(n_lat - 1):
        for j in range(n_lon - 1):
            a = i * n_lon + j
            b, c, d = a + 1, a + n_lon, a + n_lon + 1
            faces += [[a, b, c], [b, d, c]]
    return verts, np.array(faces, dtype=np.uint32), uvs


_SPHERE_VERTS, _SPHERE_FACES, _SPHERE_UVS = _uv_sphere_mesh()


def _env_vert_colors(env_hwc: np.ndarray) -> np.ndarray:
    """Bilinear-sample env map [H,W,3] float at sphere UVs → [V,3] uint8."""
    H, W = env_hwc.shape[:2]
    u = np.clip(_SPHERE_UVS[:, 0] * W - 0.5, 0, W - 1)
    v = np.clip(_SPHERE_UVS[:, 1] * H - 0.5, 0, H - 1)
    u0, v0 = u.astype(int), v.astype(int)
    u1, v1 = np.minimum(u0 + 1, W - 1), np.minimum(v0 + 1, H - 1)
    wu, wv = (u - u0)[:, None], (v - v0)[:, None]
    sampled = (
        (1 - wu) * (1 - wv) * env_hwc[v0, u0]
        +      wu * (1 - wv) * env_hwc[v0, u1]
        + (1 - wu) *      wv * env_hwc[v1, u0]
        +      wu  *      wv * env_hwc[v1, u1]
    )
    return np.clip(sampled * 255, 0, 255).astype(np.uint8)


# ── helpers ────────────────────────────────────────────────────────────────────

def _rot_to_wxyz(R: np.ndarray) -> tuple[float, float, float, float]:
    q = Rotation.from_matrix(R).as_quat()  # [x, y, z, w]
    return (float(q[3]), float(q[0]), float(q[1]), float(q[2]))


def _fig_to_uint8(fig: plt.Figure) -> np.ndarray:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=90)
    plt.close(fig)
    buf.seek(0)
    return np.array(PILImage.open(buf).convert("RGB"))


def _to_u8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    return np.clip(img * 255, 0, 255).astype(np.uint8)


_CMAP = plt.cm.plasma  # time-colour: purple=early, yellow=late


# ── visualiser ─────────────────────────────────────────────────────────────────

class ReLiFTVis:
    """Live viser visualiser — wire into the tracking loop via update_* calls."""

    def __init__(self, port: int = 8080):
        self.server = viser.ViserServer(port=port, verbose=False)
        self._lock = threading.Lock()

        self._refine_losses: list[float] = []
        self._refine_frame: int = 0
        self._last_refine_loss: Optional[float] = None

        self.total_frames: int = 1
        self._frame_idx: int = 0
        self._n_pts: int = 0
        self._trajectory: list[np.ndarray] = []

        self._setup_gui()
        self.server.scene.add_frame("/world", axes_length=0.08, axes_radius=0.004)
        print(f"[vis] Viser running at http://localhost:{port}")

    # ── GUI setup ─────────────────────────────────────────────────────────────

    def _setup_gui(self) -> None:
        with self.server.gui.add_folder("Tracking Status"):
            self._md_status = self.server.gui.add_markdown("Initialising…")

        with self.server.gui.add_folder("Live Refinement"):
            self._md_refine = self.server.gui.add_markdown("*Waiting for frame 1…*")

        with self.server.gui.add_folder("Legend"):
            self.server.gui.add_markdown(
                "**Canonical PC** — object frame, centred at origin  \n"
                "**Env sphere** — accumulated illumination at (0,0,0.7)  \n"
                "**Env panel** — equirectangular at (2.2,0,0)  \n"
                "**Frame PCs** — observed; plasma: purple=early, yellow=late  \n"
                "**Overlay PCs** — canonical at estimated pose  \n"
                "**Pose gizmos** — estimated object pose axes  \n"
                "**Image panels** — left=diffuse input, right=canonical render  \n"
                "**Trajectory** — translation trail of the object"
            )

    def _push_status(self) -> None:
        txt = (
            f"**Frame:** {self._frame_idx}  \n"
            f"**Canonical pts:** {self._n_pts:,}  \n"
        )
        if self._last_refine_loss is not None:
            txt += f"**Last refine loss:** {self._last_refine_loss:.5f}"
        self._md_status.content = txt

    # ── canonical model ───────────────────────────────────────────────────────

    def update_canonical(
        self,
        pts_obj: np.ndarray,
        colors: np.ndarray,
        env_hwc: Optional[np.ndarray],
        frame_idx: int,
    ) -> None:
        """Update canonical PC and env map. Call after each fuse_frame."""
        self._frame_idx = frame_idx
        self._n_pts = len(pts_obj)

        self.server.scene.add_point_cloud(
            "/canonical/points",
            points=pts_obj.astype(np.float32),
            colors=np.clip(colors * 255, 0, 255).astype(np.uint8),
            point_size=0.003,
        )

        if env_hwc is not None:
            # Textured UV sphere
            self.server.scene.add_mesh_simple(
                "/canonical/env_sphere",
                vertices=_SPHERE_VERTS,
                faces=_SPHERE_FACES,
                vertex_colors=_env_vert_colors(env_hwc),
                position=(0.0, 0.0, 0.7),
                wxyz=(1, 0, 0, 0),
            )
            # Flat equirectangular panel
            env_u8 = np.clip(env_hwc * 255, 0, 255).astype(np.uint8)
            H, W = env_u8.shape[:2]
            self.server.scene.add_image(
                "/canonical/env_flat",
                image=env_u8,
                render_width=2.0,
                render_height=2.0 * H / max(W, 1),
                position=(2.2, 0.0, 0.0),
                wxyz=(1, 0, 0, 0),
                format="jpeg",
            )

        self._push_status()

    # ── gradient refinement ───────────────────────────────────────────────────

    def reset_refinement(self, frame_idx: int) -> None:
        """Call immediately before starting refinement for a new frame."""
        with self._lock:
            self._refine_losses = []
            self._refine_frame = frame_idx
        self._md_refine.content = f"**Frame {frame_idx}** — refining…"

    def on_refine_step(
        self,
        iteration: int,
        loss: float,
        img_rendered: Optional[np.ndarray] = None,
        img_target: Optional[np.ndarray] = None,
    ) -> None:
        """Call each refinement step; expensive scene ops happen every 25 iters."""
        with self._lock:
            self._refine_losses.append(loss)
        self._md_refine.content = (
            f"**Frame {self._refine_frame}** — refining…  \n"
            f"iter {iteration:4d}  |  loss {loss:.5f}"
        )
        if iteration % 25 != 0:
            return

        if img_rendered is not None and img_target is not None:
            side = np.concatenate([_to_u8(img_rendered), _to_u8(img_target)], axis=1)
            self.server.scene.add_image(
                "/refinement/compare",
                image=side,
                render_width=2.0,
                render_height=1.0,
                position=(4.5, 0.0, 0.0),
                wxyz=(1, 0, 0, 0),
                format="jpeg",
            )
        self._push_loss_plot()

    def finalize_refinement(self, final_loss: float) -> None:
        """Call after refinement finishes."""
        with self._lock:
            self._last_refine_loss = final_loss
        self._push_loss_plot()
        self._md_refine.content = (
            f"**Frame {self._refine_frame}** — done  \n"
            f"final loss **{final_loss:.5f}**"
        )
        self._push_status()

    def _push_loss_plot(self) -> None:
        losses = list(self._refine_losses)
        if len(losses) < 2:
            return

        fig, axes = plt.subplots(1, 2, figsize=(8, 2.5), dpi=90)

        # Left: log-scale loss curve with fill
        ax = axes[0]
        iters = np.arange(len(losses))
        ax.semilogy(iters, losses, linewidth=1.5, color="steelblue")
        ax.fill_between(iters, losses, alpha=0.12, color="steelblue")
        ax.set_xlabel("Iteration", fontsize=8)
        ax.set_ylabel("Loss (log)", fontsize=8)
        ax.set_title(f"Refinement — frame {self._refine_frame}", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(True, which="both", alpha=0.3, linewidth=0.5)

        # Right: absolute per-step loss change
        ax2 = axes[1]
        deltas = np.abs(np.diff(losses))
        if len(deltas) > 0:
            ax2.plot(deltas, linewidth=1.2, color="tomato")
            ax2.fill_between(np.arange(len(deltas)), deltas, alpha=0.12, color="tomato")
            ax2.set_xlabel("Iteration", fontsize=8)
            ax2.set_ylabel("|Δ loss|", fontsize=8)
            ax2.set_title("Loss change rate", fontsize=9)
            ax2.tick_params(labelsize=7)
            ax2.grid(True, alpha=0.3, linewidth=0.5)

        fig.tight_layout(pad=0.5)
        self.server.scene.add_image(
            "/refinement/loss_plot",
            image=_fig_to_uint8(fig),
            render_width=3.0,
            render_height=1.25,
            position=(4.5, 1.5, 0.0),
            wxyz=(1, 0, 0, 0),
            format="jpeg",
        )

    # ── per-frame result ──────────────────────────────────────────────────────

    def add_estimated_frame(
        self,
        frame_idx: int,
        pose_abs: np.ndarray,
        pts_obj: np.ndarray,
        colors_obj: np.ndarray,
        pts_cam: np.ndarray,
        colors_cam: np.ndarray,
        img_rendered: Optional[np.ndarray] = None,
        img_diffuse: Optional[np.ndarray] = None,
    ) -> None:
        """Add a completed frame to the global 3D view.

        pts_obj / colors_obj  : canonical model (object frame) [N, 3].
        pts_cam / colors_cam  : observed PC (camera frame)     [M, 3] float [0,1].
        img_rendered          : [H, W, 3] float — canonical render at refined pose.
        img_diffuse           : [H, W, 3] uint8 or float — diffuse middle view.
        """
        t_frac = float(frame_idx) / max(self.total_frames - 1, 1)
        cr, cg, cb, _ = _CMAP(t_frac)
        tint = np.array([cr, cg, cb], dtype=np.float32)

        R    = pose_abs[:3, :3]
        t_v  = pose_abs[:3, 3]
        wxyz = _rot_to_wxyz(R)
        name = f"/frames/f{frame_idx:04d}"

        # Observed PC — plasma-tinted by time index
        pc_u8 = np.clip((colors_cam * 0.65 + tint * 0.35) * 255, 0, 255).astype(np.uint8)
        self.server.scene.add_point_cloud(
            f"{name}/pts",
            points=pts_cam.astype(np.float32),
            colors=pc_u8,
            point_size=0.002,
        )

        # Canonical PC projected to camera frame — overlaid on observed PC
        pts_at_pose = (R @ pts_obj.T).T + t_v
        can_u8 = np.clip((colors_obj * 0.8 + tint * 0.2) * 255, 0, 255).astype(np.uint8)
        self.server.scene.add_point_cloud(
            f"{name}/canonical_overlay",
            points=pts_at_pose.astype(np.float32),
            colors=can_u8,
            point_size=0.0015,
        )

        # Pose axis gizmo
        self.server.scene.add_frame(
            f"{name}/pose",
            axes_length=0.04,
            axes_radius=0.002,
            position=(float(t_v[0]), float(t_v[1]), float(t_v[2])),
            wxyz=wxyz,
        )

        # Floating image panels flanking the object centre
        side_l = R @ np.array([ 0.10, 0.0, 0.0]) + t_v  # left  = diffuse input
        side_r = R @ np.array([-0.10, 0.0, 0.0]) + t_v  # right = canonical render

        if img_diffuse is not None:
            self.server.scene.add_image(
                f"{name}/diffuse",
                image=_to_u8(img_diffuse),
                render_width=0.10,
                render_height=0.075,
                position=(float(side_l[0]), float(side_l[1]), float(side_l[2])),
                wxyz=(1, 0, 0, 0),
                format="jpeg",
            )

        if img_rendered is not None:
            self.server.scene.add_image(
                f"{name}/rendered",
                image=_to_u8(img_rendered),
                render_width=0.10,
                render_height=0.075,
                position=(float(side_r[0]), float(side_r[1]), float(side_r[2])),
                wxyz=(1, 0, 0, 0),
                format="jpeg",
            )

        # Translation trail
        self._trajectory.append(t_v.copy())
        trail = np.array(self._trajectory, dtype=np.float32)
        trail_fracs = np.linspace(0.0, 1.0, len(trail))
        trail_colors = np.array([_CMAP(f)[:3] for f in trail_fracs], dtype=np.float32)
        self.server.scene.add_point_cloud(
            "/trajectory",
            points=trail,
            colors=np.clip(trail_colors * 255, 0, 255).astype(np.uint8),
            point_size=0.006,
        )
