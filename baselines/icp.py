import os
import numpy as np
import open3d as o3d
from PIL import Image
from tqdm import tqdm


DATASET_ROOT = "/home/ngoncharov/SpecTrack_dataset"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results_icp")

_TO_OPENCV = np.array(
    [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.float64
)


def rebase_poses(gt_poses, est_poses):
    """Align an estimated absolute-pose track so that poses[0] == gt_poses[0]."""
    est_to_gt = np.linalg.inv(est_poses[0]) @ gt_poses[0]
    est_poses = [p @ est_to_gt for p in est_poses]
    return np.stack(est_poses, axis=0)


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


class ColoredICPTracker:
    """Frame-to-frame colored ICP tracker (Park et al. 2017) using Open3D GPU pipeline."""

    def __init__(
        self,
        seq: SpecTrackSequence,
        voxel_size: float = 0.005,
        cuda_device: str = "CUDA:0",
    ):
        self.seq = seq
        self.voxel_size = voxel_size
        self.o3d_device = o3d.core.Device(cuda_device)
        self.f32 = o3d.core.Dtype.Float32

        self.estimation = (
            o3d.t.pipelines.registration.TransformationEstimationForColoredICP(
                lambda_geometric=0.968
            )
        )
        self.criteria = o3d.t.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=50
        )
        self.max_corr_dist = voxel_size * 20

    # ------------------------------------------------------------------
    def _build_pcd(self, rgb, depth, mask):
        """Back-project masked pixels → voxel-downsampled, normal-estimated PCD on GPU."""
        valid = mask & (depth > 0.01) & (depth < 10.0)
        if valid.sum() < 50:
            return None

        ys, xs = np.where(valid)
        d = depth[valid]
        fx, fy = self.seq.K[0, 0], self.seq.K[1, 1]
        cx, cy = self.seq.K[0, 2], self.seq.K[1, 2]

        pts = np.stack(
            [(xs - cx) * d / fx, (ys - cy) * d / fy, d], axis=-1
        ).astype(np.float32)
        cols = (rgb[valid].astype(np.float32) / 255.0)

        pcd = o3d.t.geometry.PointCloud(self.o3d_device)
        pcd.point.positions = o3d.core.Tensor(pts, dtype=self.f32, device=self.o3d_device)
        pcd.point.colors = o3d.core.Tensor(cols, dtype=self.f32, device=self.o3d_device)
        pcd = pcd.voxel_down_sample(self.voxel_size)
        pcd.estimate_normals(radius=self.voxel_size * 2, max_nn=30)
        return pcd

    def _register(self, source, target, init: np.ndarray) -> np.ndarray:
        """Colored ICP; returns source-to-target transform as (4,4) numpy array."""
        init_t = o3d.core.Tensor(init.astype(np.float64))
        try:
            result = o3d.t.pipelines.registration.icp(
                source,
                target,
                max_correspondence_distance=self.max_corr_dist,
                init_source_to_target=init_t,
                estimation_method=self.estimation,
                criteria=self.criteria,
                voxel_size=self.voxel_size,
            )
            return result.transformation.numpy()
        except Exception as e:
            print(f"\n  [ICP] registration failed: {e}; using init guess")
            return init

    # ------------------------------------------------------------------
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
        prev_pcd = None

        for idx in tqdm(range(n), desc="  frames", leave=False):
            rgb, depth, mask = self.seq.get_frame(idx)
            cur_pcd = self._build_pcd(rgb, depth, mask)

            if idx == 0:
                current_pose = np.eye(4)
            else:
                # Constant-velocity initialisation
                if idx == 1 or est_poses[-2] is None:
                    init = np.eye(4)
                else:
                    init = est_poses[-1] @ np.linalg.inv(est_poses[-2])

                if cur_pcd is None or prev_pcd is None:
                    current_pose = est_poses[-1].copy()
                else:
                    T_rel = self._register(prev_pcd, cur_pcd, init)
                    current_pose = T_rel @ est_poses[-1]

            est_poses.append(current_pose)
            prev_pcd = cur_pcd

        est_poses = np.stack(est_poses)

        # Rebase: translate the trajectory so est_poses[0] == gt_poses[0]
        est_to_gt = np.linalg.inv(est_poses[0]) @ gt_poses[0]
        est_poses = np.stack([p @ est_to_gt for p in est_poses])

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
                tracker = ColoredICPTracker(seq)
                poses = tracker.run()
                np.save(out_path, poses)
                tqdm.write(f"  {depth_mode}/{split}/{seq_name}: {poses.shape} → {out_path}")
