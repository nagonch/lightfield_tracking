import os
import sys

import numpy as np
import torch
import torchvision

_code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(_code_dir, "LoFTR", "src"))
from loftr import LoFTR, default_cfg


_WEIGHTS = os.path.join(_code_dir, "LoFTR", "weights", "outdoor_ds.ckpt")
_RESIZE = 400
_MATCH_THR = 0.2
_BATCH = 64


class LoftrRunner:
    """Thin wrapper around LoFTR for batch image-pair matching."""

    def __init__(self, weights: str = _WEIGHTS, match_thr: float = _MATCH_THR):
        cfg = dict(default_cfg)
        cfg["match_coarse"] = dict(cfg["match_coarse"])
        cfg["match_coarse"]["thr"] = match_thr
        self.matcher = LoFTR(config=cfg)
        ckpt = torch.load(weights, map_location="cpu")
        self.matcher.load_state_dict(ckpt["state_dict"])
        self.matcher = self.matcher.eval().cuda()

    @torch.no_grad()
    def predict(self, rgbAs: np.ndarray, rgbBs: np.ndarray):
        """Find correspondences for a batch of image pairs.

        rgbAs, rgbBs : (N, H, W, 3) arrays.
        Returns a list of N arrays, each (M_i, 5): [x0, y0, x1, y1, conf].
        """
        image0 = torch.from_numpy(rgbAs.astype(np.float32)).permute(0, 3, 1, 2).cuda()
        image1 = torch.from_numpy(rgbBs.astype(np.float32)).permute(0, 3, 1, 2).cuda()
        # LoFTR expects grayscale in [0, 1]
        if image0.shape[1] == 3:
            image0 = torchvision.transforms.functional.rgb_to_grayscale(image0)
            image1 = torchvision.transforms.functional.rgb_to_grayscale(image1)
        # Accept both uint8 [0, 255] and float [0, 1] inputs
        if image0.max() > 2.0:
            image0 = image0 / 255.0
            image1 = image1 / 255.0

        ret_keys = ["mkpts0_f", "mkpts1_f", "mconf", "m_bids"]
        acc: dict = {}
        i_b = 0
        for b in range(0, len(image0), _BATCH):
            batch = {"image0": image0[b : b + _BATCH], "image1": image1[b : b + _BATCH]}
            with torch.amp.autocast("cuda", enabled=True):
                self.matcher(batch)
            batch["m_bids"] = batch["m_bids"] + i_b
            for k in ret_keys:
                acc.setdefault(k, []).append(batch[k])
            i_b += len(batch["image0"])

        for k in ret_keys:
            acc[k] = torch.cat(acc[k], dim=0)

        mkpts0 = acc["mkpts0_f"].cpu().numpy()
        mkpts1 = acc["mkpts1_f"].cpu().numpy()
        mconf = acc["mconf"].cpu().numpy()
        pair_ids = acc["m_bids"].cpu().numpy().astype(int)

        corres = np.concatenate(
            [mkpts0, mkpts1, mconf.reshape(-1, 1)], axis=-1
        ).astype(np.float32)

        return [corres[pair_ids == i] for i in range(len(rgbAs))]
