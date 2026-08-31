"""Viser inspection of the captured RealSense sequences (SpecTrack real capture).

Visualizes, in the (fixed) camera frame:
  * the camera frustum with the current RGB image (object axes projected onto it),
  * the RGB-D point cloud unprojected from depth/XXXXXX.png, optionally filtered by
    the segment masks (masks/XXXXXX.png, from segment_realsense_capture.py),
  * the corrected object mesh (captured/mustard_bottle_mesh) posed with
    poses_object/XXXXXX.txt (= poses_arm_endeffector @ ee_T_obj, fitted by
    correct_realsense_poses.py),
  * the object pose axes + full object trajectory,
  * the robot base frame and the end-effector pose/trajectory from
    poses_arm_endeffector/XXXXXX.txt (the dataset's original raw pose files),
  * per-frame mesh-vs-segment-depth alignment stats.

Run inside the lift6dof container, then open http://localhost:8080 :
    docker exec -w "$PWD" lift6dof python viz_realsense_capture.py
"""

import argparse
import os
import threading
import time

import cv2
import numpy as np
import trimesh
import viser
import viser.transforms as vtf
from PIL import Image
from scipy.spatial import cKDTree

ROOT = "/home/ngoncharov/SpecTrack_dataset/captured"  # holds realsense/ and epi/
MESH_PATH = "/home/ngoncharov/SpecTrack_dataset/captured/mustard_bottle_mesh/textured_simple.obj"
DEPTH_SCALE = 1000.0  # uint16 mm -> m


# --------------------------------------------------------------------------- #
# data loading
# --------------------------------------------------------------------------- #
def load_sequence(seq_dir: str) -> dict:
    K = np.loadtxt(os.path.join(seq_dir, "intrinsics.txt"))
    dpath = os.path.join(seq_dir, "distortion.txt")
    D = np.loadtxt(dpath) if os.path.exists(dpath) else None
    img_dir = os.path.join(seq_dir, "images")
    dep_dir = os.path.join(seq_dir, "depth")
    msk_dir = os.path.join(seq_dir, "masks")
    entries = sorted(os.listdir(img_dir))
    stems = [e.replace(".png", "") for e in entries]
    # RealSense: images/XXXXXX.png; EPI cross: images/XXXXXX/cam_4_4.png (center view)
    img_paths = [
        os.path.join(img_dir, e, "cam_4_4.png")
        if os.path.isdir(os.path.join(img_dir, e)) else os.path.join(img_dir, e)
        for e in entries
    ]

    def load_poses(sub):
        d = os.path.join(seq_dir, sub)
        return np.stack([np.loadtxt(os.path.join(d, s + ".txt")) for s in stems])

    poses_obj = load_poses("poses_object")
    poses_ee = load_poses("poses_arm_endeffector")
    # constant camera-side hand-eye correction fitted for this rig (epi): apply it to
    # the EE display so it stays consistent with poses_object (files on disk stay raw)
    cpath = os.path.join(seq_dir, "cam_correction.txt")
    if os.path.exists(cpath):
        poses_ee = np.einsum("ij,njk->nik", np.loadtxt(cpath), poses_ee)
    arm = np.load(os.path.join(seq_dir, "arm_poses_result.npy"))
    assert len(entries) == len(poses_obj) == len(arm), (
        f"count mismatch: {len(entries)} images, {len(poses_obj)} poses, {len(arm)} arm poses"
    )
    return dict(
        K=K,
        D=D,
        img_paths=img_paths,
        dep_paths=[os.path.join(dep_dir, s + ".png") for s in stems],
        msk_paths=[os.path.join(msk_dir, s + ".png") for s in stems],
        cam_T_obj=poses_obj,
        cam_T_ee=poses_ee,
        base_T_ee=arm,
    )


def unproject(depth: np.ndarray, rgb: np.ndarray, K: np.ndarray, stride: int,
              max_depth: float, mask: np.ndarray | None = None,
              D: np.ndarray | None = None):
    """Back-project an RGB-D frame to a colored point cloud in the camera frame."""
    d = depth[::stride, ::stride].astype(np.float32) / DEPTH_SCALE
    c = rgb[::stride, ::stride]
    H, W = d.shape
    u, v = np.meshgrid(np.arange(W) * stride, np.arange(H) * stride)
    valid = (d > 0) & (d < max_depth)
    if mask is not None:
        valid &= mask[::stride, ::stride]
    z = d[valid]
    uv = np.stack([u[valid], v[valid]], -1).astype(np.float64)
    if D is not None:  # distorted pixel grid -> normalized rays
        xy = cv2.undistortPoints(uv.reshape(-1, 1, 2), K, D).reshape(-1, 2)
    else:
        xy = (uv - K[:2, 2]) / np.array([K[0, 0], K[1, 1]])
    return np.concatenate([xy * z[:, None], z[:, None]], -1), c[valid]


def draw_axes(img: np.ndarray, K: np.ndarray, T: np.ndarray, length: float = 0.08,
              D: np.ndarray | None = None):
    """Draw the object coordinate frame (x=red, y=green, z=blue) onto the image."""
    pts = np.float64([[0, 0, 0], [length, 0, 0], [0, length, 0], [0, 0, length]])
    cam = pts @ T[:3, :3].T + T[:3, 3]
    if (cam[:, 2] <= 1e-6).any():
        return img
    if D is not None:
        uv, _ = cv2.projectPoints(cam, np.zeros(3), np.zeros(3), K, D)
        uv = uv.reshape(-1, 2).round().astype(int)
    else:
        uv = (cam @ K.T)
        uv = (uv[:, :2] / uv[:, 2:]).round().astype(int)
    for tip, color in zip(uv[1:], [(255, 40, 40), (40, 255, 40), (40, 120, 255)]):
        cv2.line(img, tuple(uv[0]), tuple(tip), color, 2, cv2.LINE_AA)
    cv2.circle(img, tuple(uv[0]), 3, (255, 255, 255), -1, cv2.LINE_AA)
    return img


# --------------------------------------------------------------------------- #
# viewer
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--mesh", default=MESH_PATH)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--check", action="store_true", help="headless validation only, no server")
    args = ap.parse_args()

    mesh = trimesh.load(args.mesh)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_mesh()
    mesh_pts = mesh.sample(20000)
    print(f"mesh: {len(mesh.vertices)} verts, extents {mesh.extents.round(3)} m")

    seqs = {}
    for cap in ["realsense", "epi"]:
        for s in ["diffuse", "reflective"]:
            seq_dir = os.path.join(args.root, cap, s)
            if os.path.isdir(os.path.join(seq_dir, "poses_object")):
                seqs[f"{cap}/{s}"] = load_sequence(seq_dir)

    # camera<->robot-base transform: poses_arm_endeffector = cam_T_base @ arm_poses
    # exactly, so recover it directly and report the spread as a consistency check.
    cam_T_base = {}
    for name, ds in seqs.items():
        Xs = np.einsum("nij,njk->nik", ds["cam_T_ee"], np.linalg.inv(ds["base_T_ee"]))
        cam_T_base[name] = Xs[0]
        eeT = np.linalg.inv(ds["cam_T_ee"][0]) @ ds["cam_T_obj"][0]
        print(f"[{name}] {len(ds['img_paths'])} frames | cam_T_base spread "
              f"{np.abs(Xs - Xs[0]).max():.2e} | ee_T_obj t = "
              f"{np.round(eeT[:3, 3] * 1000, 1)} mm")

    def frame_stats(ds, idx, stride=2, use_ee_pose=False):
        """Mesh-vs-depth alignment: distances of masked depth points to the posed mesh."""
        depth = np.asarray(Image.open(ds["dep_paths"][idx]))
        rgb = np.asarray(Image.open(ds["img_paths"][idx]))
        seg = np.asarray(Image.open(ds["msk_paths"][idx])) > 127
        pts, _ = unproject(depth, rgb, ds["K"], stride, 3.5, mask=seg, D=ds["D"])
        T = ds["cam_T_ee" if use_ee_pose else "cam_T_obj"][idx]
        posed = mesh_pts @ T[:3, :3].T + T[:3, 3]
        dist, _ = cKDTree(posed).query(pts, workers=-1)
        near = dist[dist < 0.03]
        return dict(n_near=len(near), n_seg=len(pts),
                    med_mm=float(np.median(near) * 1000) if len(near) else float("nan"),
                    mean_mm=float(np.mean(near) * 1000) if len(near) else float("nan"))

    if args.check:
        for name, ds in seqs.items():
            for idx in [0, len(ds["img_paths"]) // 2, len(ds["img_paths"]) - 1]:
                a = frame_stats(ds, idx, use_ee_pose=True)
                b = frame_stats(ds, idx)
                print(f"[{name}] frame {idx:3d}: ee pose {a['n_near']:5d}/{a['n_seg']:5d} "
                      f"seg pts near / {a['med_mm']:5.1f} mm median  |  object pose "
                      f"{b['n_near']:5d}/{b['n_seg']:5d} / {b['med_mm']:5.1f} mm median")
        return

    server = viser.ViserServer(port=args.port)
    cloud_cache: dict[tuple, tuple] = {}
    stats_cache: dict[tuple, dict] = {}

    first = next(iter(seqs))
    with server.gui.add_folder("Sequence"):
        gui_seq = server.gui.add_dropdown("sequence", list(seqs), initial_value=first)
        gui_frame = server.gui.add_slider("frame", 0, len(seqs[first]["img_paths"]) - 1, 1, 0)
        gui_play = server.gui.add_checkbox("play", False)
    with server.gui.add_folder("Display"):
        gui_stride = server.gui.add_slider("cloud stride", 1, 8, 1, 4)
        gui_maxd = server.gui.add_slider("max depth (m)", 0.5, 4.0, 0.1, 3.5)
        gui_psize = server.gui.add_slider("point size (mm)", 0.5, 10.0, 0.5, 3.0)
        gui_mesh = server.gui.add_checkbox("show mesh", True)
        gui_axes2d = server.gui.add_checkbox("axes on image", True)
        gui_seg = server.gui.add_checkbox("filter by segment", False)
        gui_objonly = server.gui.add_checkbox("object-only cloud", False)
        gui_arm = server.gui.add_checkbox("show arm/base", True)
    gui_stats = server.gui.add_markdown("")

    def rebuild(_=None):
        name = gui_seq.value
        ds = seqs[name]
        n = len(ds["img_paths"])
        gui_frame.max = n - 1
        idx = min(gui_frame.value, n - 1)
        K = ds["K"]
        T = ds["cam_T_obj"][idx]
        rgb = np.asarray(Image.open(ds["img_paths"][idx]))
        H, W = rgb.shape[:2]

        # camera frustum at the origin (world = camera frame, OpenCV convention)
        frustum_img = np.ascontiguousarray(rgb[::2, ::2])
        if gui_axes2d.value:
            K_half = K.copy()
            K_half[:2] /= 2
            frustum_img = draw_axes(frustum_img, K_half, T, D=ds["D"])
        server.scene.add_camera_frustum(
            "/camera", fov=2 * np.arctan2(H / 2, K[1, 1]), aspect=W / H,
            scale=0.12, image=frustum_img)

        key = (name, idx, gui_stride.value, round(gui_maxd.value, 1), gui_seg.value)
        if key not in cloud_cache:
            depth = np.asarray(Image.open(ds["dep_paths"][idx]))
            seg = (np.asarray(Image.open(ds["msk_paths"][idx])) > 127
                   if gui_seg.value else None)
            cloud_cache[key] = unproject(depth, rgb, K, gui_stride.value,
                                         gui_maxd.value, mask=seg, D=ds["D"])
        pts, cols = cloud_cache[key]
        if gui_objonly.value:
            posed = mesh_pts @ T[:3, :3].T + T[:3, 3]
            m = cKDTree(posed).query(pts, workers=-1)[0] < 0.05
            pts, cols = pts[m], cols[m]
        server.scene.add_point_cloud(
            "/cloud", pts, cols, point_size=gui_psize.value / 1000, point_shape="circle")

        # posed mesh + object axes + trajectory
        so3 = vtf.SO3.from_matrix(T[:3, :3])
        server.scene.add_mesh_trimesh("/object/mesh", mesh, wxyz=so3.wxyz,
                                      position=T[:3, 3], visible=gui_mesh.value)
        server.scene.add_frame("/object/axes", wxyz=so3.wxyz, position=T[:3, 3],
                               axes_length=0.08, axes_radius=0.003)
        traj = ds["cam_T_obj"][:, :3, 3]
        server.scene.add_spline_catmull_rom("/object/traj", traj, color=(255, 160, 0),
                                            line_width=2.0)
        server.scene.add_point_cloud("/object/traj_pts", traj,
                                     np.tile([[255, 160, 0]], (n, 1)), point_size=0.004)

        # robot base + end-effector (poses_arm_endeffector is already camera-frame)
        X = cam_T_base[name]
        base_so3 = vtf.SO3.from_matrix(X[:3, :3])
        server.scene.add_frame("/arm/base", wxyz=base_so3.wxyz, position=X[:3, 3],
                               axes_length=0.15, axes_radius=0.004, visible=gui_arm.value)
        cam_T_ee = ds["cam_T_ee"]
        ee_so3 = vtf.SO3.from_matrix(cam_T_ee[idx, :3, :3])
        server.scene.add_frame("/arm/ee", wxyz=ee_so3.wxyz, position=cam_T_ee[idx, :3, 3],
                               axes_length=0.06, axes_radius=0.003, visible=gui_arm.value)
        server.scene.add_spline_catmull_rom("/arm/traj", cam_T_ee[:, :3, 3],
                                            color=(0, 180, 255), line_width=2.0,
                                            visible=gui_arm.value)

        skey = (name, idx)
        if skey not in stats_cache:
            stats_cache[skey] = frame_stats(ds, idx)
        st = stats_cache[skey]
        gui_stats.content = (
            f"**{name}** frame {idx}/{n - 1} — object at z = {T[2, 3]:.3f} m  \n"
            f"mesh↔segment depth: {st['n_near']}/{st['n_seg']} pts within 3 cm, "
            f"median **{st['med_mm']:.1f} mm**, mean {st['mean_mm']:.1f} mm")

        # gravity up = robot base +z, for natural orbit controls
        server.scene.set_up_direction(tuple(X[:3, :3] @ np.array([0.0, 0.0, 1.0])))

    for h in (gui_seq, gui_frame, gui_stride, gui_maxd, gui_psize,
              gui_mesh, gui_axes2d, gui_seg, gui_objonly, gui_arm):
        h.on_update(rebuild)
    rebuild()

    def play_loop():
        while True:
            time.sleep(0.35)
            if gui_play.value:
                gui_frame.value = (gui_frame.value + 1) % (gui_frame.max + 1)

    threading.Thread(target=play_loop, daemon=True).start()
    print(f"viewer at http://localhost:{args.port}")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
