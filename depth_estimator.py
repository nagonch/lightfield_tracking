import torch
from depth_anything_3.api import DepthAnything3
import numpy as np


class DepthEstimator:
    def __init__(self, infer_gs=False):
        self.device = torch.device("cuda")
        self.da3_model = DepthAnything3.from_pretrained("depth-anything/DA3-GIANT")
        self.da3_model = self.da3_model.to(device=self.device)
        self.infer_gs = infer_gs

    def __call__(self, frame, object_mask, top_confidence=0.7):
        camera_matrix = frame["camera_matrix"]
        camera_poses_resh = frame["camera_poses_rel"].reshape(-1, 4, 4)
        LF_resh = frame["LF"].reshape(-1, *frame["LF"].shape[2:])
        pred = self.da3_model.inference(
            [(image * 255.0).cpu().numpy().astype(np.uint8) for image in LF_resh],
            extrinsics=torch.linalg.inv(camera_poses_resh).cpu().numpy(),
            intrinsics=np.stack(
                [
                    camera_matrix.cpu().numpy(),
                ]
                * camera_poses_resh.shape[0]
            ),
            infer_gs=self.infer_gs,
        )
        depth = pred.depth
        depth = torch.tensor(depth).to(device=self.device, dtype=torch.float32)
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1),  # [n,1,h',w']
            size=(LF_resh.shape[1], LF_resh.shape[2]),
            mode="bilinear",
            align_corners=False,
        )[:, 0]

        confidence = pred.conf
        confidence = torch.tensor(confidence).to(
            device=self.device, dtype=torch.float32
        )
        confidence = torch.nn.functional.interpolate(
            confidence.unsqueeze(1),  # [n,1,h',w']
            size=(LF_resh.shape[1], LF_resh.shape[2]),
            mode="bilinear",
            align_corners=False,
        )[:, 0]
        confidence = confidence[confidence.shape[0] // 2]
        confidence = confidence * object_mask
        threshold_value = torch.quantile(
            confidence[object_mask == 1], 1.0 - top_confidence
        )
        high_confidence_mask = confidence >= threshold_value

        return depth, high_confidence_mask


if __name__ == "__main__":
    pass
