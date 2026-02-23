import os
import torch
import numpy as np
import json
from PIL import Image


import os, json
import numpy as np
import torch
from torchvision.io import read_image  # faster than PIL in many cases


class LFDataset(torch.utils.data.Dataset):
    def __init__(
        self, folder, use_masks=True, use_predicted_depth=True, dtype=torch.float32
    ):
        self.folder = folder
        self.dtype = dtype

        self.camera_matrix = torch.from_numpy(
            np.loadtxt(os.path.join(folder, "camera_matrix.txt"))
        ).to(dtype)

        with open(os.path.join(folder, "metadata.json"), "r") as f:
            self.metadata = json.load(f)

        self.n_s, self.n_t = self.metadata["n_views"]
        self.s_mid, self.t_mid = self.n_s // 2, self.n_t // 2

        self.depth_dir = os.path.join(folder, "depth")
        self.object_poses_dir = os.path.join(folder, "object_poses")
        self.camera_poses_dir = os.path.join(folder, "camera_poses")

        # frames
        self.frames = sorted([d for d in os.listdir(folder) if d.startswith("LF_")])
        self.size = len(self.frames)

        # depth / object pose filenames
        self.depth_fnames = sorted(os.listdir(self.depth_dir))
        self.object_poses_fnames = sorted(os.listdir(self.object_poses_dir))

        # camera poses: load once
        camera_pose_paths = [
            os.path.join(self.camera_poses_dir, f)
            for f in sorted(os.listdir(self.camera_poses_dir))
            if f.endswith(".txt")
        ]
        camera_poses = [torch.from_numpy(np.loadtxt(p)) for p in camera_pose_paths]
        camera_poses = torch.stack(camera_poses, dim=0).to(dtype)
        camera_poses = camera_poses.reshape(self.n_s, self.n_t, *camera_poses[0].shape)
        self.camera_poses = camera_poses.contiguous()

        # precompute relative poses once
        center_inv = torch.linalg.inv(self.camera_poses[self.s_mid, self.t_mid])
        self.camera_poses_rel = (center_inv @ self.camera_poses).contiguous()

        # precompute image & mask paths per frame
        self.frame_image_paths = []
        self.frame_mask_paths = [] if use_masks else None
        self.frame_pred_depth_paths = [] if use_predicted_depth else None

        for frame_name in self.frames:
            frame_path = os.path.join(folder, frame_name)

            img_paths = sorted(
                os.path.join(frame_path, f)
                for f in os.listdir(frame_path)
                if f.endswith(".png")
            )
            self.frame_image_paths.append(img_paths)

            if use_masks:
                masks_dir = os.path.join(frame_path, "masks")
                if os.path.isdir(masks_dir):
                    mask_paths = sorted(
                        os.path.join(masks_dir, f)
                        for f in os.listdir(masks_dir)
                        if f.endswith(".png")
                    )
                else:
                    mask_paths = None
                self.frame_mask_paths.append(mask_paths)

            if use_predicted_depth:
                p = os.path.join(frame_path, "predicted_depth.npy")
                self.frame_pred_depth_paths.append(p if os.path.exists(p) else None)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        img_paths = self.frame_image_paths[idx]

        # read all views: returns uint8 CHW
        # stack -> [V, C, H, W]
        views = [read_image(p) for p in img_paths]
        lf = torch.stack(views, dim=0)  # uint8

        # reshape to [S, T, C, H, W] and normalize
        lf = lf.view(self.n_s, self.n_t, *lf.shape[1:]).contiguous()
        lf = lf.to(self.dtype).div_(255.0)

        depth_path = os.path.join(self.depth_dir, self.depth_fnames[idx])
        depth_img = read_image(
            depth_path
        )  # likely [1, H, W] uint8/uint16 depending on encoding
        # If your depth PNG is 16-bit, read_image will give uint16. Convert carefully:
        depth = depth_img.squeeze(0).to(self.dtype).div_(1000.0)

        object_pose_path = os.path.join(
            self.object_poses_dir, self.object_poses_fnames[idx]
        )
        object_pose = torch.from_numpy(np.loadtxt(object_pose_path)).to(self.dtype)

        masks = None
        if self.frame_mask_paths is not None:
            mask_paths = self.frame_mask_paths[idx]
            if mask_paths is not None and len(mask_paths) > 0:
                mask_views = [read_image(p) for p in mask_paths]  # [1, H, W] uint8
                masks = torch.stack(mask_views, dim=0).squeeze(1)  # [V, H, W]
                masks = masks.view(
                    self.n_s, self.n_t, masks.shape[-2], masks.shape[-1]
                ).contiguous()
                masks = masks > 0  # bool

        predicted_depth = None
        if self.frame_pred_depth_paths is not None:
            p = self.frame_pred_depth_paths[idx]
            if p is not None:
                # np.load is fine; if it’s big, consider mmap_mode="r"
                predicted_depth = torch.from_numpy(np.load(p, mmap_mode="r")).to(
                    self.dtype
                )

        frame_path = os.path.join(self.folder, self.frames[idx])

        return {
            "LF": lf.cuda(),
            "depth": depth.cuda(),
            "predicted_depth": (
                predicted_depth.cuda() if predicted_depth is not None else None
            ),
            "frame_path": frame_path,
            "camera_matrix": self.camera_matrix.cuda(),
            "object_pose": object_pose.cuda(),
            "camera_poses": self.camera_poses.cuda(),
            "camera_poses_rel": self.camera_poses_rel.cuda(),
            "masks": masks.cuda() if masks is not None else None,
            "baseline": self.metadata["x_spacing"],
        }


if __name__ == "__main__":
    dataset = LFDataset("/home/ngoncharov/cvpr2026/datasets/ycbv_lf/bleach")
    print(dataset[0].keys())
