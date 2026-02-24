from src.dataset import LFDataset
from time import time
from PIL import Image
import os
from segmentor import Segmentor
from depth_anything_3.api import DepthAnything3
import torch
from da3_functions import da3_run_from_tensors
from src.utilities import backproject_depth_to_pointcloud, Visualizer
import numpy as np
from src.disparity import get_LF_disparity


def main():
    dataset = LFDataset(
        "/home/ngoncharov/cvpr2026/datasets/LiFT_dataset/jug_motion_prod"
    )
    s_size, t_size = dataset.metadata["n_views"]
    segmentor = Segmentor(prompt="shiny metal jug.")

    device = torch.device("cuda")
    da3_model = DepthAnything3.from_pretrained("depth-anything/DA3-GIANT")
    da3_model = da3_model.to(device=device)

    v = Visualizer()

    time_now = time()
    for i, frame in enumerate(dataset):
        img_central = frame["LF"][s_size // 2, t_size // 2]

        camera_matrix = frame["camera_matrix"]
        camera_poses_resh = frame["camera_poses_rel"].reshape(-1, 4, 4)
        LF_resh = frame["LF"].reshape(-1, *frame["LF"].shape[2:])
        pred = da3_model.inference(
            [(image * 255.0).cpu().numpy().astype(np.uint8) for image in LF_resh],
            extrinsics=torch.linalg.inv(camera_poses_resh).cpu().numpy(),
            intrinsics=np.stack(
                [
                    camera_matrix.cpu().numpy(),
                ]
                * camera_poses_resh.shape[0]
            ),
            infer_gs=True,
        )
        # torch.save(pred.gaussians, "gaussians.pt")
        # raise
        depth = pred.depth
        depth = torch.tensor(depth).to(device=device, dtype=torch.float32)

        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1),  # [n,1,h',w']
            size=(img_central.shape[0], img_central.shape[1]),
            mode="bilinear",
            align_corners=False,
        )[:, 0]
        mask = segmentor(img_central)
        if i % 2 == 0:
            depth = depth[depth.shape[0] // 2]
            color = img_central[mask > 0].reshape(-1, 3)
            depth = depth * (mask > 0)
            pc = backproject_depth_to_pointcloud(
                pixel_indices=None,
                depths=depth,
                camera_matrix=camera_matrix,
            )
            pc = pc[(mask > 0).reshape(-1)]

            depth_gt = frame["depth"]
            depth_gt = depth_gt * (mask > 0)
            pc_gt = backproject_depth_to_pointcloud(
                pixel_indices=None,
                depths=depth_gt,
                camera_matrix=camera_matrix,
            )
            pc_gt = pc_gt[(mask > 0).reshape(-1)]

            v.add_point_cloud(
                f"testpc_{i}", points=pc.cpu().numpy(), colors=color.cpu().numpy()
            )
            v.add_point_cloud(
                f"testpc_{i}_gt", points=pc_gt.cpu().numpy(), colors=color.cpu().numpy()
            )
    v.run()


if __name__ == "__main__":
    main()
