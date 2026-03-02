import os
import torch
import numpy as np
import json
from PIL import Image


class LFDataset:
    def __init__(self, folder, poses_to_opencv=True):
        if "ycbv" in folder:
            self.flip = True
        else:
            self.flip = False
        self.poses_to_opencv = poses_to_opencv
        self.folder = folder
        self.camera_matrix = torch.tensor(
            np.loadtxt(f"{self.folder}/camera_matrix.txt"), dtype=torch.float32
        )
        with open(f"{self.folder}/metadata.json", "r") as f:
            self.metadata = json.load(f)
        self.frames = list(
            sorted([item for item in os.listdir(self.folder) if "LF_" in item])
        )
        self.size = len(self.frames)
        self.camera_poses_dir = os.path.join(self.folder, "camera_poses")
        self.depth_dir = os.path.join(self.folder, "depth")
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
        self.to_opencv = torch.tensor(
            [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
            dtype=torch.float32,
        ).cuda()

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        frame_path = os.path.join(self.folder, self.frames[idx])
        img_paths = sorted(
            [
                os.path.join(frame_path, f)
                for f in os.listdir(frame_path)
                if f.endswith(".png")
            ]
        )

        imgs = [
            torch.tensor(np.array(Image.open(p)), dtype=torch.float32)
            for p in img_paths
        ]
        LF = torch.stack(imgs, dim=0)
        LF = LF.view(
            self.metadata["n_views"][0], self.metadata["n_views"][1], *imgs[0].shape
        )
        LF /= LF.max()
        s_mid, t_mid = LF.shape[0] // 2, LF.shape[1] // 2
        depth = np.array(
            Image.open(os.path.join(self.depth_dir, self.depth_fnames[idx]))
        )
        depth = torch.tensor(depth, dtype=torch.float32) / 1000.0
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

            masks = [
                torch.tensor(np.array(Image.open(p)), dtype=torch.bool)
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
