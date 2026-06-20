from SAM_functions import (
    get_dino_models,
    get_image_predictor,
    get_image_masks_from_boxes,
    get_dino_boxes,
)
from PIL import Image
import numpy as np
from hydra import initialize_config_module
from hydra.core.global_hydra import GlobalHydra
from cutie.inference.inference_core import InferenceCore
from cutie.utils.get_default_model import get_default_model
import torch


class Segmentor:
    def __init__(self, prompt: str, sam_image_predictor=None):
        self.prompt = prompt
        self.initialized = False

        # SAM2 and Cutie both configure Hydra's *global* config state, and only one
        # config module can be registered at a time. SAM2 registers its module on
        # first import, but a previous Segmentor's Cutie init (below) leaves Hydra
        # pointing at Cutie's config — so re-point Hydra at the sam2 config module
        # before building SAM2, then hand it back to Cutie afterwards.
        if sam_image_predictor is None:
            GlobalHydra.instance().clear()
            initialize_config_module("sam2", version_base="1.2")
        self.image_predictor = (
            sam_image_predictor
            if sam_image_predictor is not None
            else get_image_predictor()
        )
        self.processor, self.grounding_model = get_dino_models()

        GlobalHydra.instance().clear()
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
    import os

    from config import DATASET_ROOT, SPLIT_PREFIXES, REFLECTIVITIES
    from src.dataset import LFDataset
    from utils import linear_to_srgb

    OUT_ROOT = "/home/ngoncharov/cvpr2026/ReLiFT-6DoF/masks_vis"
    os.makedirs(OUT_ROOT, exist_ok=True)

    # Run every sequence (split_prefix × reflectivity × sequence) through the
    # segmentor, prompting on the object name used to locate the mesh (the same
    # string main.py feeds the segmentor). For each frame save a side-by-side
    # [ full central view | central_srgb * (mask == 1) ] so the segmentation can
    # be judged against the background. Frames go to
    # masks_vis/<split>_<refl>__<seq>/<i>.png; make_mask_gifs.sh turns each such
    # folder into a per-sequence GIF.
    GAP = 8  # black separator (px) between the two panels
    for split_prefix in SPLIT_PREFIXES:
        for reflectivity in REFLECTIVITIES:
            split_dir = os.path.join(DATASET_ROOT, f"{split_prefix}_{reflectivity}")
            if not os.path.isdir(split_dir):
                print(f"skip missing split: {split_dir}")
                continue

            for seq_name in sorted(os.listdir(split_dir)):
                seq_path = os.path.join(split_dir, seq_name)
                if not os.path.isdir(seq_path) or seq_name == "models":
                    continue

                key = f"{split_prefix}_{reflectivity}__{seq_name}"
                seq_out = os.path.join(OUT_ROOT, key)
                os.makedirs(seq_out, exist_ok=True)

                try:
                    dataset = LFDataset(seq_path, depth_source="gt")
                    s_size, t_size = dataset.metadata["n_views"]
                    # Mesh names use underscores; GroundingDINO detects the
                    # natural-language form ("bleach_cleanser" → "bleach cleanser").
                    prompt = dataset.object_name.replace("_", " ")
                    # One Segmentor per sequence: Cutie tracks temporally from frame 0.
                    segmentor = Segmentor(prompt=prompt)
                    print(f"[{key}] prompt='{prompt}' frames={len(dataset)}")

                    for i, frame in enumerate(dataset):
                        central_srgb = linear_to_srgb(
                            frame["LF"][s_size // 2, t_size // 2].clamp(0.0, 1.0)
                        )
                        mask = segmentor(central_srgb)
                        masked = central_srgb * (mask == 1)[..., None]
                        full = (
                            (central_srgb.clamp(0.0, 1.0) * 255)
                            .cpu()
                            .numpy()
                            .astype(np.uint8)
                        )
                        right = (
                            (masked.clamp(0.0, 1.0) * 255)
                            .cpu()
                            .numpy()
                            .astype(np.uint8)
                        )
                        gap = np.zeros((full.shape[0], GAP, 3), dtype=np.uint8)
                        arr = np.concatenate([full, gap, right], axis=1)
                        Image.fromarray(arr).save(
                            os.path.join(seq_out, f"{i:04d}.png")
                        )
                except Exception as exc:  # keep going over the other sequences
                    print(f"[{key}] FAILED: {exc}")

    print(f"done -> {OUT_ROOT}")
