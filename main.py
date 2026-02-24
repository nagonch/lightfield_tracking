from src.dataset import LFDataset
from time import time
from PIL import Image
import os
from segmentor import Segmentor
from depth_anything_3.api import DepthAnything3
import torch
from da3_functions import da3_run_from_tensors


def main():
    dataset = LFDataset("/home/ngoncharov/cvpr2026/datasets/ycbv_lf/mustard0")
    s_size, t_size = dataset.metadata["n_views"]
    segmentor = Segmentor(prompt="bottle.")

    device = torch.device("cuda")
    da3_model = DepthAnything3.from_pretrained("depth-anything/DA3-GIANT")
    da3_model = da3_model.to(device=device)

    time_now = time()
    for i, frame in enumerate(dataset):
        img_central = frame["LF"][s_size // 2, t_size // 2]

        camera_matrix = frame["camera_matrix"]
        camera_poses_resh = frame["camera_poses"].reshape(-1, 4, 4)
        LF_resh = frame["LF"].reshape(-1, *frame["LF"].shape[2:])
        result = da3_run_from_tensors(
            da3_model,
            images_mhw3=LF_resh,
            intrinsics_4x4=camera_matrix,
            poses_m44=camera_poses_resh,
            poses_are_c2w=True,
            process_res=512,
        )
        print(result)
        torch.save(result, f"frame_{i:04d}_da3.pt")
        raise
        mask = segmentor(img_central)

    time_per_frame = (time() - time_now) / len(dataset)
    print(time_per_frame)


if __name__ == "__main__":
    main()
