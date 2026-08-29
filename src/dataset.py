import os
import torch
import numpy as np
import json
import trimesh
from PIL import Image
from utils import srgb_to_linear

# ── real LiFT dataset options ─────────────────────────────────────────────────
# The real LiFT capture is a 9×9 grid at 1280×720 (vs 5×5 @ 640×480 synthetic),
# i.e. ~10× the pixels per frame. These module-level knobs let the driver
# (main_lift.py) thin the light field without touching track_sequence, which
# constructs LFDataset internally. A stride-2 view subsample keeps the full
# 40 mm baseline as a synthetic-shaped 5×5 grid. Selection is centred on the
# true central view (the one the depth maps and masks are referenced to).
LIFT_VIEW_STRIDE = 2
LIFT_IMAGE_SCALE = 1.0


def set_lift_options(view_stride: int | None = None, image_scale: float | None = None):
    global LIFT_VIEW_STRIDE, LIFT_IMAGE_SCALE
    if view_stride is not None:
        LIFT_VIEW_STRIDE = int(view_stride)
    if image_scale is not None:
        LIFT_IMAGE_SCALE = float(image_scale)


class LFDataset:
    def __init__(self, folder, poses_to_opencv=True, depth_source: str = "gt"):
        if "ycbv" in folder:
            self.flip = False
        else:
            self.flip = False
        self.poses_to_opencv = poses_to_opencv
        self.folder = folder

        # Determine dataset root and object name from folder path
        split_dir = os.path.dirname(folder)         # e.g. .../SpecTrack_dataset/cube_0.0
        split_name = os.path.basename(split_dir)    # e.g. cube_0.0
        seq_name = os.path.basename(folder)         # e.g. bleach_hard_00_03_chaitanya
        self.dataset_root = os.path.dirname(split_dir)

        # Real LiFT capture: sequences ship a gdino_prompt.txt (segmentor prompt)
        # instead of an object_meshes/ directory, poses are already in the OpenCV
        # camera convention, and the LF is 9×9 @ 720p (thinned via the module
        # options above).
        self.is_lift = os.path.isfile(os.path.join(folder, "gdino_prompt.txt"))
        self.view_stride = LIFT_VIEW_STRIDE if self.is_lift else 1
        self.image_scale = LIFT_IMAGE_SCALE if self.is_lift else 1.0

        if self.is_lift:
            with open(os.path.join(folder, "gdino_prompt.txt")) as f:
                prompt = f.read().strip().rstrip(".")
            # Stored with underscores so Segmentor's `.replace("_", " ")`
            # recovers the natural-language prompt.
            self.object_name = prompt.replace(" ", "_")
            self.mesh_dir = None
        elif split_name.startswith("cube_"):
            self.object_name = "cube"
        else:
            mesh_names = [
                m for m in os.listdir(os.path.join(self.dataset_root, "object_meshes"))
                if os.path.isdir(os.path.join(self.dataset_root, "object_meshes", m))
            ]
            self.object_name = max(
                mesh_names,
                key=lambda m: len(os.path.commonprefix([seq_name, m])),
            )

        if not self.is_lift:
            self.mesh_dir = os.path.join(
                self.dataset_root, "object_meshes", self.object_name
            )
        self.camera_matrix = torch.tensor(
            np.loadtxt(f"{self.folder}/camera_matrix.txt"), dtype=torch.float32
        )
        if self.image_scale != 1.0:
            self.camera_matrix = self.camera_matrix.clone()
            self.camera_matrix[0] *= self.image_scale
            self.camera_matrix[1] *= self.image_scale
        with open(f"{self.folder}/metadata.json", "r") as f:
            self.metadata = json.load(f)
        self.frames = list(
            sorted([item for item in os.listdir(self.folder) if "LF_" in item])
        )
        self.size = len(self.frames)
        self.camera_poses_dir = os.path.join(self.folder, "camera_poses")
        if self.is_lift:
            # Real capture has only the RealSense depth/ ("gt") and optional
            # depth_lf/. "synth" also maps to depth/ — track_sequence loads it
            # purely for frame indexing when computing live LF depth.
            depth_subfolder = {"lf": "depth_lf"}.get(depth_source, "depth")
        else:
            depth_subfolder = {"synth": "depth_synth", "lf": "depth_lf"}.get(
                depth_source, "depth"
            )
        self.depth_dir = os.path.join(self.folder, depth_subfolder)
        self.depth_fnames = list(sorted(os.listdir(self.depth_dir)))

        self.object_poses_dir = os.path.join(self.folder, "object_poses")
        self.object_poses_fnames = list(sorted(os.listdir(self.object_poses_dir)))

        self.camera_poses = []
        for pose_file in sorted(os.listdir(self.camera_poses_dir)):
            if pose_file.endswith(".txt"):
                pose_path = os.path.join(self.camera_poses_dir, pose_file)
                pose = torch.tensor(np.loadtxt(pose_path), dtype=torch.float32)
                self.camera_poses.append(pose)
        self.camera_poses = torch.stack(self.camera_poses, dim=0).reshape(
            self.metadata["n_views"][0],
            self.metadata["n_views"][1],
            *self.camera_poses[0].shape,
        )

        # View subsampling (real LiFT only): keep every `stride`-th row/column,
        # centred on the true central view so depth/ and the pipeline's
        # s_size//2 central index keep referring to the same physical camera.
        self._view_sel = None
        if self.view_stride > 1:
            S_full, T_full = self.metadata["n_views"]
            sel_s = list(range((S_full // 2) % self.view_stride, S_full, self.view_stride))
            sel_t = list(range((T_full // 2) % self.view_stride, T_full, self.view_stride))
            assert S_full // 2 in sel_s and T_full // 2 in sel_t
            assert len(sel_s) % 2 == 1 and len(sel_t) % 2 == 1
            self._view_sel = [s * T_full + t for s in sel_s for t in sel_t]
            self.camera_poses = self.camera_poses[sel_s][:, sel_t]
            self.metadata = dict(self.metadata)
            self.metadata["n_views"] = [len(sel_s), len(sel_t)]
            self.metadata["x_spacing"] = self.metadata["x_spacing"] * self.view_stride
            self.metadata["y_spacing"] = self.metadata["y_spacing"] * self.view_stride

        # Synthetic (Blender) object poses need the axis flip to land in OpenCV;
        # the real LiFT poses are already OpenCV, so the flip is identity there.
        self.to_opencv = (
            torch.eye(4, dtype=torch.float32)
            if self.is_lift
            else torch.tensor(
                [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
                dtype=torch.float32,
            )
        ).cuda()

    def get_mesh(self) -> trimesh.Trimesh:
        if self.mesh_dir is None:
            raise RuntimeError("The real LiFT dataset ships no object meshes")
        obj_path = os.path.join(self.mesh_dir, "textured_simple.obj")
        return trimesh.load(obj_path, force="mesh")

    def __len__(self):
        return self.size

    def _load_img(self, path: str, resample) -> Image.Image:
        img = Image.open(path)
        if self.image_scale != 1.0:
            W, H = img.size
            img = img.resize(
                (round(W * self.image_scale), round(H * self.image_scale)), resample
            )
        return img

    def __getitem__(self, idx):
        frame_path = os.path.join(self.folder, self.frames[idx])
        img_paths = sorted(
            [
                os.path.join(frame_path, f)
                for f in os.listdir(frame_path)
                if f.endswith(".png")
            ]
        )
        if self._view_sel is not None:
            img_paths = [img_paths[i] for i in self._view_sel]

        imgs = [
            torch.tensor(
                np.array(self._load_img(p, Image.BILINEAR)), dtype=torch.float32
            )
            for p in img_paths
        ]
        LF = torch.stack(imgs, dim=0)
        LF = LF.view(
            self.metadata["n_views"][0], self.metadata["n_views"][1], *imgs[0].shape
        )
        LF /= LF.max()
        LF = srgb_to_linear(LF)
        s_mid, t_mid = LF.shape[0] // 2, LF.shape[1] // 2
        # Depth resizes with NEAREST so RealSense holes (0) don't bleed into
        # valid metric depth.
        depth = np.array(
            self._load_img(
                os.path.join(self.depth_dir, self.depth_fnames[idx]), Image.NEAREST
            )
        )
        depth = torch.tensor(depth.astype(np.float32), dtype=torch.float32) / 1000.0
        object_pose = np.loadtxt(
            os.path.join(self.object_poses_dir, self.object_poses_fnames[idx])
        )
        object_pose = torch.tensor(object_pose, dtype=torch.float32)
        object_pose = torch.linalg.inv(self.camera_poses[s_mid, t_mid]) @ object_pose
        if self.poses_to_opencv:
            object_pose = object_pose.cuda() @ self.to_opencv

        masks_dir = os.path.join(frame_path, "masks")
        if os.path.exists(masks_dir):
            mask_paths = sorted(
                [
                    os.path.join(masks_dir, f)
                    for f in os.listdir(masks_dir)
                    if f.endswith(".png")
                ]
            )
            if self._view_sel is not None:
                mask_paths = [mask_paths[i] for i in self._view_sel]

            masks = [
                torch.tensor(
                    np.array(self._load_img(p, Image.NEAREST)), dtype=torch.bool
                )
                for p in mask_paths
            ]
            masks = torch.stack(masks, dim=0)
            masks = masks.view(
                self.metadata["n_views"][0],
                self.metadata["n_views"][1],
                imgs[0].shape[0],
                imgs[0].shape[1],
            ).cuda()
        else:
            masks = None
        predicted_depth_path = os.path.join(frame_path, "predicted_depth.npy")
        if os.path.exists(predicted_depth_path):
            predicted_depth = torch.tensor(
                np.load(predicted_depth_path), dtype=torch.float32
            ).cuda()
            if self.image_scale != 1.0:
                predicted_depth = torch.nn.functional.interpolate(
                    predicted_depth[None, None],
                    size=(imgs[0].shape[0], imgs[0].shape[1]),
                    mode="nearest",
                )[0, 0]
        else:
            predicted_depth = None
        LF = torch.flip(LF, dims=[0, 1]) if self.flip else LF
        return {
            "LF": LF.cuda(),
            "depth": depth.cuda(),
            "predicted_depth": predicted_depth,
            "frame_path": frame_path,
            "camera_matrix": self.camera_matrix.cuda(),
            "object_pose": object_pose.cuda(),
            "camera_poses": self.camera_poses.cuda(),
            "camera_poses_rel": (
                torch.linalg.inv(self.camera_poses[s_mid, t_mid]) @ self.camera_poses
            ).cuda(),
            "masks": masks,
            "baseline": self.metadata["x_spacing"],
        }


if __name__ == "__main__":
    dataset = LFDataset("/home/ngoncharov/cvpr2026/datasets/ycbv_lf/bleach")
    print(dataset[0].keys())
