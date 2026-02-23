from SAM_functions import (
    get_dino_models,
    get_image_predictor,
    get_image_masks_from_boxes,
    get_dino_boxes,
)
from PIL import Image
import numpy as np
import hydra
from cutie.inference.inference_core import InferenceCore
from cutie.utils.get_default_model import get_default_model
import torch


class Segmentor:
    def __init__(self, prompt: str):
        self.prompt = prompt
        self.initialized = False

        self.image_predictor = get_image_predictor()
        self.processor, self.grounding_model = get_dino_models()

        hydra.core.global_hydra.GlobalHydra.instance().clear()
        cutie = get_default_model()
        self.cutie_processor = InferenceCore(cutie, cfg=cutie.cfg)

    @torch.inference_mode()
    @torch.amp.autocast("cuda:0")
    def __call__(self, img_tensor_hwc):
        """
        img_tensor_hwc : torch tensor HWC float [0,1]
        returns mask (torch tensor H,W)
        """

        if not self.initialized:
            image_pil = Image.fromarray(
                (img_tensor_hwc * 255).cpu().numpy().astype(np.uint8)
            )

            boxes = get_dino_boxes(
                image_pil,
                self.prompt,
                self.processor,
                self.grounding_model,
                return_full=False,
            )

            image_mask = get_image_masks_from_boxes(
                self.image_predictor, boxes, image_pil
            )[0]

            del self.image_predictor, self.processor, self.grounding_model

            out_prob = self.cutie_processor.step(
                img_tensor_hwc.permute(2, 0, 1),
                torch.from_numpy(np.array(image_mask)).cuda(),
                objects=[1],
            )

            self.initialized = True

        else:
            out_prob = self.cutie_processor.step(
                img_tensor_hwc.permute(2, 0, 1),
            )

        mask = self.cutie_processor.output_prob_to_mask(out_prob)
        return mask


if __name__ == "__main__":
    pass
