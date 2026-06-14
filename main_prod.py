import os
import sys
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from icp import run_explorative_icp_with_centering, rebase_poses


DATASET_ROOT = "/home/ngoncharov/SpecTrack_dataset"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results_icp_base")

_TO_OPENCV = np.array(
    [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.float64
)


class SpecTrackSequence:
    """Load central-view RGB, depth, mask, and GT object poses for one LF sequence.

    5x5 light field; central view is at grid index (2,2) = flat index 12.
    Depth images are uint16 in millimetres; poses use the OpenCV camera convention.

    depth_mode: "gt" → depth/, "synth" → depth_synth/
    """

    CENTRAL_VIEW = 12  # 2*5 + 2
    DEPTH_DIRS = {"gt": "depth", "synth": "depth_synth"}

    def __init__(self, seq_dir: str, depth_mode: str = "gt"):
        assert depth_mode in self.DEPTH_DIRS, f"depth_mode must be 'gt' or 'synth', got {depth_mode!r}"
        self.seq_dir = seq_dir
        self.depth_mode = depth_mode
        self.K = np.loadtxt(os.path.join(seq_dir, "camera_matrix.txt"))  # (3,3)

        self.frames = sorted(d for d in os.listdir(seq_dir) if d.startswith("LF_"))
        synth_files = set(os.listdir(os.path.join(seq_dir, "depth_synth")))
        self._synth_files = synth_files
        self.obj_pose_files = sorted(
            os.listdir(os.path.join(seq_dir, "object_poses"))
        )

        # Central camera is fixed across frames (the LF rig doesn't move)
        self._cam_pose = np.loadtxt(
            os.path.join(seq_dir, "camera_poses", f"{self.CENTRAL_VIEW:04d}.txt")
        )
        self._inv_cam_pose = np.linalg.inv(self._cam_pose)

    def __len__(self) -> int:
        return len(self.frames)

    def get_gt_pose(self, idx: int) -> np.ndarray:
        """GT object-to-camera transform (4x4) in OpenCV convention."""
        world_obj = np.loadtxt(
            os.path.join(self.seq_dir, "object_poses", self.obj_pose_files[idx])
        )
        return self._inv_cam_pose @ world_obj @ _TO_OPENCV

    def get_frame(self, idx: int):
        """Return (rgb uint8 HxWx3, depth float64 metres HxW, mask bool HxW)."""
        frame_dir = os.path.join(self.seq_dir, self.frames[idx])
        vname = f"{self.CENTRAL_VIEW:04d}.png"
        rgb = np.array(Image.open(os.path.join(frame_dir, vname)))
        # Frame "LF_0138" → depth file "0138.png"
        depth_fname = self.frames[idx][3:] + ".png"
        if self.depth_mode == "synth" and depth_fname not in self._synth_files:
            depth_dir = "depth"  # fall back to GT depth for missing synth frames
        else:
            depth_dir = self.DEPTH_DIRS[self.depth_mode]
        depth = (
            np.array(
                Image.open(os.path.join(self.seq_dir, depth_dir, depth_fname))
            ).astype(np.float64)
            / 1000.0
        )
        mask = (
            np.array(Image.open(os.path.join(frame_dir, "masks", vname))) > 0
        )
        return rgb, depth, mask


class IcpBase:
    """Frame-to-frame ICP tracker using explorative ICP with rotation hypotheses."""

    def __init__(self, seq: SpecTrackSequence, voxel_size: float = 0.005):
        self.seq = seq
        self.voxel_size = voxel_size

    def _build_points(self, rgb, depth, mask):
        """Back-project masked pixels → (N,3) xyz and (N,3) rgb arrays."""
        valid = mask & (depth > 0.01) & (depth < 10.0)
        if valid.sum() < 50:
            return None, None

        ys, xs = np.where(valid)
        d = depth[valid]
        fx, fy = self.seq.K[0, 0], self.seq.K[1, 1]
        cx, cy = self.seq.K[0, 2], self.seq.K[1, 2]

        pts = np.stack(
            [(xs - cx) * d / fx, (ys - cy) * d / fy, d], axis=-1
        ).astype(np.float64)
        cols = rgb[valid].astype(np.float64) / 255.0
        return pts, cols

    def run(self) -> np.ndarray:
        """
        Track the full sequence.

        Returns
        -------
        poses : (N, 4, 4) float64
            Absolute object-to-camera poses, rebased so that poses[0] == GT poses[0].
        """
        n = len(self.seq)
        gt_poses = np.stack([self.seq.get_gt_pose(i) for i in range(n)])

        est_poses = []
        prev_pts = None
        prev_cols = None

        for idx in tqdm(range(n), desc="  frames", leave=False):
            rgb, depth, mask = self.seq.get_frame(idx)
            cur_pts, cur_cols = self._build_points(rgb, depth, mask)

            if idx == 0:
                current_pose = np.eye(4)
            else:
                if cur_pts is None or prev_pts is None:
                    current_pose = est_poses[-1].copy()
                else:
                    result = run_explorative_icp_with_centering(
                        source_points_xyz=prev_pts,
                        target_points_xyz=cur_pts,
                        source_colors_rgb=prev_cols,
                        target_colors_rgb=cur_cols,
                        max_correspondence_distance=self.voxel_size * 20,
                        coarse_voxel_size=self.voxel_size * 2,
                        final_voxel_size=self.voxel_size,
                    )
                    T_rel = result["transform_source_to_target"]
                    current_pose = T_rel @ est_poses[-1]

            est_poses.append(current_pose)
            prev_pts = cur_pts
            prev_cols = cur_cols

        est_poses = np.stack(est_poses)
        est_poses = rebase_poses(gt_poses, est_poses)
        return est_poses


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    splits = sorted(
        d
        for d in os.listdir(DATASET_ROOT)
        if os.path.isdir(os.path.join(DATASET_ROOT, d))
        and (d.startswith("cube_") or d.startswith("objects_"))
    )

    for depth_mode in ("gt", "synth"):
        for split in splits:
            split_dir = os.path.join(DATASET_ROOT, split)
            out_dir = os.path.join(RESULTS_DIR, depth_mode, split)
            os.makedirs(out_dir, exist_ok=True)

            sequences = sorted(
                s
                for s in os.listdir(split_dir)
                if os.path.isdir(os.path.join(split_dir, s)) and s != "models"
            )

            for seq_name in tqdm(sequences, desc=f"{depth_mode}/{split}"):
                out_path = os.path.join(out_dir, f"{seq_name}.npy")
                if os.path.exists(out_path):
                    tqdm.write(f"  {depth_mode}/{split}/{seq_name}: already done, skipping")
                    continue

                seq = SpecTrackSequence(os.path.join(split_dir, seq_name), depth_mode=depth_mode)
                tracker = IcpBase(seq)
                poses = tracker.run()
                np.save(out_path, poses)
                tqdm.write(f"  {depth_mode}/{split}/{seq_name}: {poses.shape} → {out_path}")
