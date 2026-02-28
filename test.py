from src.utilities import Visualizer
from src.dataset import LFDataset
import torch


class SurfaceLF:
    def __init__(self, LF, K, cam_poses, current_object_pose, pc, canonical_frame=True):
        self.LF = LF.reshape(-1, LF.shape[2], LF.shape[3], 3).permute(0, 3, 1, 2)
        self.K = K
        self.cam_poses = cam_poses.reshape(-1, 4, 4)
        self.current_object_pose = current_object_pose
        self.pc = pc
        if canonical_frame:
            self.cam_poses = torch.linalg.inv(current_object_pose) @ self.cam_poses
            pose_inv = torch.linalg.inv(current_object_pose).double()
            self.pc = (pose_inv[:3, :3] @ pc.T).T + pose_inv[:3, 3]
            self.current_object_pose = torch.eye(4).double()


if __name__ == "__main__":
    lf_debug = torch.load("lf_debug_0.pt")
    surface_lf = SurfaceLF(
        LF=lf_debug["LF"]
        .reshape(-1, lf_debug["LF"].shape[2], lf_debug["LF"].shape[3], 3)
        .permute(0, 3, 1, 2),
        cam_poses=lf_debug["cam_poses"].reshape(-1, 4, 4),
        K=lf_debug["K"],
        current_object_pose=lf_debug["current_object_pose"],
        pc=lf_debug["pc"],
    )
    v = Visualizer()
    LF = lf_debug["LF"]
    cam_poses = lf_debug["cam_poses"].reshape(-1, 4, 4)
    K = lf_debug["K"]
    current_object_pose = lf_debug["current_object_pose"]
    pc = lf_debug["pc"]

    for i, pose in enumerate(surface_lf.cam_poses):
        v.add_frame(f"cam_{i}", pose.cpu().numpy())

    v.add_point_cloud("pc", surface_lf.pc.cpu().numpy())
    v.add_frame("object", surface_lf.current_object_pose.cpu().numpy())
    v.run()
