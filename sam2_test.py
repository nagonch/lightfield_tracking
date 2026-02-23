from src.dataset import LFDataset
from src.utilities import Visualizer, backproject_depth_to_pointcloud
from time import time
from SAM_functions import (
    get_dino_models,
    get_image_predictor,
    get_video_predictor,
    get_image_masks_from_boxes,
    get_dino_boxes,
)
from PIL import Image
import numpy as np


if __name__ == "__main__":
    dataset = LFDataset("/home/ngoncharov/cvpr2026/datasets/ycbv_lf/bleach0")
    s_size, t_size = dataset.metadata["n_views"]
    time_now = time()

    image_predictor = get_image_predictor()
    video_predictor = get_video_predictor()
    processor, grounding_model = get_dino_models()
    PROMPT = "bottle"

    for i, frame in enumerate(dataset):
        img_central = frame["LF"][s_size // 2, t_size // 2]
        if i == 0:
            image_pil = Image.fromarray(
                (img_central * 255).cpu().numpy().astype(np.uint8)
            )
            boxes = get_dino_boxes(
                image_pil,
                PROMPT,
                processor,
                grounding_model,
                return_full=False,
            )
            image_mask = get_image_masks_from_boxes(image_predictor, boxes, image_pil)
        else:


    time_per_frame = (time() - time_now) / len(dataset)
    print(time_per_frame)
