import os
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm


DATASET_ROOT = "/home/ngoncharov/SpecTrack_dataset"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results_pnp")

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
        self.K = np.loadtxt(os.path.join(seq_dir, "camera_matrix.txt"))

        self.frames = sorted(d for d in os.listdir(seq_dir) if d.startswith("LF_"))
        self._synth_files = set(os.listdir(os.path.join(seq_dir, "depth_synth")))
        self.obj_pose_files = sorted(os.listdir(os.path.join(seq_dir, "object_poses")))

        self._cam_pose = np.loadtxt(
            os.path.join(seq_dir, "camera_poses", f"{self.CENTRAL_VIEW:04d}.txt")
        )
        self._inv_cam_pose = np.linalg.inv(self._cam_pose)

    def __len__(self) -> int:
        return len(self.frames)

    def get_gt_pose(self, idx: int) -> np.ndarray:
        world_obj = np.loadtxt(
            os.path.join(self.seq_dir, "object_poses", self.obj_pose_files[idx])
        )
        return self._inv_cam_pose @ world_obj @ _TO_OPENCV

    def get_frame(self, idx: int):
        """Return (rgb uint8 HxWx3, depth float64 metres HxW, mask bool HxW)."""
        frame_dir = os.path.join(self.seq_dir, self.frames[idx])
        vname = f"{self.CENTRAL_VIEW:04d}.png"
        rgb = np.array(Image.open(os.path.join(frame_dir, vname)))

        depth_fname = self.frames[idx][3:] + ".png"  # "LF_0138" → "0138.png"
        if self.depth_mode == "synth" and depth_fname not in self._synth_files:
            depth_dir = "depth"  # fall back to GT depth for missing synth frames
        else:
            depth_dir = self.DEPTH_DIRS[self.depth_mode]
        depth = (
            np.array(Image.open(os.path.join(self.seq_dir, depth_dir, depth_fname)))
            .astype(np.float64) / 1000.0
        )

        mask = np.array(Image.open(os.path.join(frame_dir, "masks", vname))) > 0
        return rgb, depth, mask


class PnPTracker:
    """Frame-to-frame SIFT + PnP object tracker.

    Pipeline per frame:
      1. Detect SIFT keypoints inside the object mask.
      2. Back-project each keypoint to 3D using the depth map.
      3. Match current descriptors against the previous frame's (Lowe ratio test).
      4. Solve PnP RANSAC (EPnP) on the 3D(prev)–2D(curr) correspondences.
      5. Refine the result with Levenberg–Marquardt on RANSAC inliers.
      6. Fall back to the constant-velocity motion model on failure.
    """

    MIN_INLIERS = 6

    def __init__(self, seq: SpecTrackSequence):
        self.seq = seq
        self.K = seq.K.astype(np.float64)
        self.dist = np.zeros(4, dtype=np.float64)

        self.detector = cv2.SIFT_create(
            nfeatures=5000,
            contrastThreshold=0.01,
            edgeThreshold=5,
        )

        # FLANN-based kNN matcher — fast for float32 SIFT descriptors
        self.matcher = cv2.FlannBasedMatcher(
            {"algorithm": 1, "trees": 5},  # FLANN_INDEX_KDTREE
            {"checks": 50},
        )

    # ------------------------------------------------------------------
    def _extract(self, rgb, depth, mask):
        """Detect SIFT, back-project to 3D, filter by valid depth.

        Returns (pts3d, kp_pts, desc) as float32 arrays, or (None,None,None).
        """
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        mask_u8 = (mask.astype(np.uint8)) * 255
        kp, desc = self.detector.detectAndCompute(gray, mask=mask_u8)

        if desc is None or len(kp) == 0:
            return None, None, None

        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        H, W = depth.shape

        pts3d, kp_pts, valid_desc = [], [], []
        for i, k in enumerate(kp):
            u, v = int(round(k.pt[0])), int(round(k.pt[1]))
            if not (0 <= u < W and 0 <= v < H):
                continue
            z = depth[v, u]
            if z < 0.01 or z > 10.0:
                continue
            pts3d.append([(u - cx) * z / fx, (v - cy) * z / fy, z])
            kp_pts.append(k.pt)
            valid_desc.append(desc[i])

        if len(pts3d) < self.MIN_INLIERS:
            return None, None, None

        return (
            np.array(pts3d, dtype=np.float32),
            np.array(kp_pts, dtype=np.float32),
            np.array(valid_desc, dtype=np.float32),
        )

    def _register(self, prev_pts3d, prev_desc, curr_kp_pts, curr_desc):
        """Match, RANSAC PnP (EPnP), LM refinement.

        Returns source-to-target (4,4) transform or None on failure.
        """
        raw_matches = self.matcher.knnMatch(prev_desc, curr_desc, k=2)
        good = [
            m for pair in raw_matches
            if len(pair) == 2
            for m, n in [pair]
            if m.distance < 0.75 * n.distance
        ]

        if len(good) < self.MIN_INLIERS:
            return None

        obj_pts = np.array(
            [prev_pts3d[m.queryIdx] for m in good], dtype=np.float64
        )
        img_pts = np.array(
            [curr_kp_pts[m.trainIdx] for m in good], dtype=np.float64
        )

        try:
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                obj_pts,
                img_pts,
                self.K,
                self.dist,
                iterationsCount=1000,
                reprojectionError=2.0,
                confidence=0.999,
                flags=cv2.SOLVEPNP_EPNP,
            )
        except cv2.error:
            return None

        if not ok or inliers is None or len(inliers) < self.MIN_INLIERS:
            return None

        inlier_idx = inliers.ravel()
        try:
            rvec, tvec = cv2.solvePnPRefineLM(
                obj_pts[inlier_idx],
                img_pts[inlier_idx],
                self.K,
                self.dist,
                rvec.copy(),
                tvec.copy(),
            )
        except cv2.error:
            pass  # keep EPnP + RANSAC result

        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = tvec.ravel()

        # Guard against EPnP degeneracies (can produce astronomically large translations
        # while still passing the inlier count check)
        if np.linalg.norm(T[:3, 3]) > 3.0:
            return None

        return T

    # ------------------------------------------------------------------
    def run(self) -> np.ndarray:
        """Track the full sequence.

        Returns
        -------
        poses : (N, 4, 4) float64
            Absolute object-to-camera poses, rebased so that poses[0] == GT poses[0].
        """
        n = len(self.seq)
        gt_poses = np.stack([self.seq.get_gt_pose(i) for i in range(n)])

        est_poses = []
        prev_pts3d = prev_desc = None

        for idx in tqdm(range(n), desc="  frames", leave=False):
            rgb, depth, mask = self.seq.get_frame(idx)
            pts3d, kp_pts, desc = self._extract(rgb, depth, mask)

            if idx == 0:
                current_pose = np.eye(4)
            else:
                init = (
                    est_poses[-1] @ np.linalg.inv(est_poses[-2])
                    if len(est_poses) >= 2
                    else np.eye(4)
                )

                if pts3d is None or prev_pts3d is None:
                    current_pose = init @ est_poses[-1]
                else:
                    T_rel = self._register(prev_pts3d, prev_desc, kp_pts, desc)
                    current_pose = (T_rel @ est_poses[-1]) if T_rel is not None else (init @ est_poses[-1])

            est_poses.append(current_pose)

            if pts3d is not None:
                prev_pts3d, prev_desc = pts3d, desc

        est_poses = np.stack(est_poses)
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
                tracker = PnPTracker(seq)
                poses = tracker.run()
                np.save(out_path, poses)
                tqdm.write(f"  {depth_mode}/{split}/{seq_name}: {poses.shape} → {out_path}")
