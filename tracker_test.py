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
import hydra
from cutie.inference.inference_core import InferenceCore
from cutie.utils.get_default_model import get_default_model
import torch
import os


@torch.inference_mode()
@torch.cuda.amp.autocast()
def main():
    dataset = LFDataset(
        "/home/ngoncharov/cvpr2026/datasets/LiFT_dataset/box_motion_prod"
    )
    s_size, t_size = dataset.metadata["n_views"]
    time_now = time()

    image_predictor = get_image_predictor()
    processor, grounding_model = get_dino_models()
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    cutie = get_default_model()
    cutie_processor = InferenceCore(cutie, cfg=cutie.cfg)

    PROMPT = "white and blue box."

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
            image_mask = get_image_masks_from_boxes(image_predictor, boxes, image_pil)[
                0
            ]
            del image_predictor, processor, grounding_model
            print(img_central.shape)
            out_prob = cutie_processor.step(
                img_central.permute(2, 0, 1),
                torch.from_numpy(np.array(image_mask)).cuda(),
                objects=[
                    1,
                ],
            )
        else:
            out_prob = cutie_processor.step(
                img_central.permute(2, 0, 1),
            )
        mask = cutie_processor.output_prob_to_mask(out_prob)
        Image.fromarray((mask.cpu().numpy() * 255).astype(np.uint8)).save(
            f"cutie_output/frame_{i:04d}.png"
        )
    time_per_frame = (time() - time_now) / len(dataset)
    print(time_per_frame)


if __name__ == "__main__":
    main()
