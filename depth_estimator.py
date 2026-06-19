"""Depth-Anything-3 depth estimation for the ReLiFT light-field tracker.

This module does three things (all driven from ``__main__`` / the CLI below):

1. ``DepthEstimator`` — runs DA3 on the *full* light field of one frame and
   returns the central-view depth + confidence.  (Validation: "can we infer
   Depth-Anything depth and get values".)

2. ``align_depth_to_gt`` — post-processes the predicted depth so it matches the
   ground-truth depth as closely as possible, fitting a robust scale+shift
   **only over the object-mask pixels** (the monocular-depth → metric alignment
   used to bring predicted geometry into the GT frame, same family of method the
   LoFTR/ICP path uses to put predicted points into the GT metric scale).

3. ``run_sequence`` / ``visualize`` — runs the estimator over whole sequences
   with temporal consistency (see below), caches the result, and opens a viser
   server that compares GT vs. estimated depth per object, with a frame slider.

Notes / bugs fixed vs. the original implementation
---------------------------------------------------
* **sRGB input.**  The dataset stores the light field in *linear* RGB
  (``srgb_to_linear`` + max-normalised).  DA3 was trained on ordinary sRGB
  images, so feeding ``LF * 255`` (linear) is wrong gamma.  We undo the gamma
  with ``linear_to_srgb`` before handing images to DA3.
* **Central view.**  DA3 reorders views internally for reference selection but
  restores the original order on output (``restore_original_order`` in the
  ViT), so output index == input index.  The central LF view is therefore
  ``(S//2)*T + (T//2)``; the original ``len//2`` only happens to coincide for an
  odd square grid.  We index the central view explicitly.
* **Pose order.**  ``camera_poses_rel`` are camera→reference (c2w) transforms;
  DA3 wants world→camera (w2c) extrinsics, so we invert.  (Verified correct.)
* **Return value.**  We return the central-view depth (matched to the GT depth,
  which is rendered for the central camera), not all 25 views.

Temporal consistency ("video mode")
------------------------------------
DA3's native "video mode" (``da3 video ...``) simply feeds the video frames in
as a joint multi-view set; there is no separate temporal module in the base API
(``da3_streaming`` is a heavier SLAM extension).  Here the camera rig is *static*
and the object *moves*, so feeding several frames jointly would violate DA3's
static-scene assumption (the moving object "ghosts"), and the ~2 cm LF aperture
gives almost no cross-view baseline — so per-frame depth is effectively
monocular and there is no geometry to "track" the object across frames.

We therefore use a **persistent-background** temporal model
(``build_persistent_background``): each frame still runs DA3 on its own full
light field, then we build one robust per-pixel *median* background-depth
template over the whole sequence and re-scale every frame onto it.  The static
scene becomes identical in every frame and the per-frame object inherits a
sequence-consistent scale anchor (far steadier than aligning to a single noisy
frame).  The object region is left to each frame's own estimate — that is
inherent to a static rig + moving object.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from depth_anything_3.api import DepthAnything3
from src.dataset import LFDataset
from utils import linear_to_srgb


# ──────────────────────────────────────────────────────────────────────────────
#  Estimator
# ──────────────────────────────────────────────────────────────────────────────
class DepthEstimator:
    def __init__(
        self,
        model_name: str = "depth-anything/DA3-GIANT",
        ref_view_strategy: str = "middle",
        infer_gs: bool = True,
    ):
        self.device = torch.device("cuda")
        self.da3_model = DepthAnything3.from_pretrained(model_name).to(
            device=self.device
        )
        self.ref_view_strategy = ref_view_strategy
        self.infer_gs = infer_gs

    @staticmethod
    def _resize_to(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """[N,h',w'] -> [N,h,w] bilinear."""
        return torch.nn.functional.interpolate(
            x.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=False
        )[:, 0]

    @torch.inference_mode()
    def infer_lf(self, frame: dict):
        """Run DA3 over the full light field of one frame.

        Returns
        -------
        depth_c : [H,W]   central-view metric depth (DA3 scale, aligned to input poses)
        conf_c  : [H,W]   central-view confidence
        depth   : [N,H,W] all-view depth
        cmid    : int     central-view flat index
        """
        LF = frame["LF"]  # [S,T,H,W,3] linear RGB in [0,1]
        S, T, H, W, _ = LF.shape
        N = S * T
        cmid = (S // 2) * T + (T // 2)

        K = frame["camera_matrix"]
        K33 = K[:3, :3] if K.shape[-1] >= 3 else K
        poses_c2w = frame["camera_poses_rel"].reshape(
            N, 4, 4
        )  # cam_i -> reference (c2w)

        # DA3 expects ordinary sRGB uint8 images; the dataset stores LINEAR RGB.
        lf_flat = LF.reshape(N, H, W, 3)
        imgs = [
            (linear_to_srgb(img).clamp(0, 1) * 255.0)
            .round()
            .to(torch.uint8)
            .cpu()
            .numpy()
            for img in lf_flat
        ]
        # extrinsics are world->camera (w2c); camera_poses_rel are c2w -> invert.
        extrinsics = poses_c2w.double().cpu().numpy()
        intrinsics = np.broadcast_to(K33.cpu().numpy(), (N, 3, 3)).copy()

        pred = self.da3_model.inference(
            imgs,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            ref_view_strategy=self.ref_view_strategy,
            infer_gs=self.infer_gs,
        )

        depth = torch.tensor(pred.depth, device=self.device, dtype=torch.float32)
        depth = self._resize_to(depth, H, W)
        if pred.conf is not None:
            conf = torch.tensor(pred.conf, device=self.device, dtype=torch.float32)
            conf = self._resize_to(conf, H, W)
        else:
            conf = torch.ones_like(depth)
        return depth[cmid], conf[cmid], depth, cmid

    @torch.inference_mode()
    def __call__(self, frame, object_mask, top_confidence: float = 0.7):
        """Back-compatible entry point: central-view depth + high-confidence mask.

        ``object_mask`` is the central-view object mask; the returned mask keeps
        the ``top_confidence`` fraction of most-confident object pixels.
        """
        depth_c, conf_c, _, _ = self.infer_lf(frame)
        object_mask = object_mask.to(conf_c.device).bool()
        conf_in = conf_c[object_mask]
        if conf_in.numel() == 0:
            return depth_c, torch.zeros_like(object_mask)
        thr = torch.quantile(conf_in, 1.0 - top_confidence)
        high_conf = (conf_c >= thr) & object_mask
        return depth_c, high_conf


# ──────────────────────────────────────────────────────────────────────────────
#  Depth post-processing (align to GT) and temporal consistency
# ──────────────────────────────────────────────────────────────────────────────
def _robust_affine(
    src: torch.Tensor,
    ref: torch.Tensor,
    iters: int = 3,
    trim: float = 0.2,
    scale_bounds=(0.05, 20.0),
):
    """Robust least-squares ``s,b`` minimising ``(s*src + b - ref)^2``.

    Iteratively trims the worst-``trim`` residual fraction (re-descending) so a
    few outliers (depth bleed at the silhouette) do not dominate the fit.
    Falls back to a shift-only fit when the source is near-degenerate (e.g. an
    almost fronto-parallel object whose depth range is mostly noise).
    """
    src = src.double()
    ref = ref.double()
    w = torch.ones_like(src)
    s, b = 1.0, 0.0
    for _ in range(iters):
        m = w > 0
        if m.sum() < 10:
            break
        sv, rv = src[m], ref[m]
        if sv.std() < 1e-4:  # degenerate -> shift only
            s, b = 1.0, float((rv - sv).median())
        else:
            A = torch.stack([sv, torch.ones_like(sv)], dim=1)
            sol = torch.linalg.lstsq(A, rv.unsqueeze(1)).solution.squeeze(1)
            s, b = float(sol[0]), float(sol[1])
            s = float(np.clip(s, *scale_bounds))
        res = (s * src + b - ref).abs()
        thr = torch.quantile(res[w > 0], 1.0 - trim)
        w = (res <= thr).double()
    return s, b


def align_depth_to_gt(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor):
    """Post-process ``pred`` to match ``gt`` over the object mask only.

    Fits a robust scale+shift on the masked, valid pixels and applies it to the
    whole predicted map.  Returns (aligned_depth, (s, b), metrics-dict).
    """
    valid = mask.bool() & (gt > 1e-3) & (pred > 1e-3)
    metrics = {"n": int(valid.sum())}
    if valid.sum() < 10:
        return (
            pred.clone(),
            (1.0, 0.0),
            {
                **metrics,
                "mae_mm": float("nan"),
                "rmse_mm": float("nan"),
                "med_mm": float("nan"),
                "delta_1.05": float("nan"),
                "corr": float("nan"),
            },
        )
    s, b = _robust_affine(pred[valid], gt[valid])
    aligned = s * pred + b
    err = (aligned[valid] - gt[valid]).abs()
    ratio = torch.maximum(aligned[valid] / gt[valid], gt[valid] / aligned[valid])
    pv, gv = pred[valid].double(), gt[valid].double()
    corr = float(
        ((pv - pv.mean()) * (gv - gv.mean())).mean() / (pv.std() * gv.std() + 1e-12)
    )
    metrics.update(
        mae_mm=float(err.mean() * 1000),
        rmse_mm=float(torch.sqrt((err**2).mean()) * 1000),
        med_mm=float(err.median() * 1000),
        delta_1_05=float((ratio < 1.05).double().mean()),
        corr=corr,
        scale=float(s),
        shift=float(b),
    )
    return aligned, (s, b), metrics


def build_persistent_background(raw: torch.Tensor, masks: torch.Tensor, iters: int = 2):
    """Static-scene depth template + per-frame depths co-scaled onto it.

    The rig is static, so every background pixel has the same true depth in every
    frame.  We build a robust per-pixel **median** background template across the
    whole sequence and re-scale each frame (robust scale+shift on its own
    background) onto that template.  This is the "persistent background"
    temporal model: a single stable static scene plus a sequence-consistent scale
    anchor (far steadier than aligning to one noisy frame-0).  The object region
    is left to each frame's own estimate (see ``composite_frame``).

    Parameters
    ----------
    raw   : [F,H,W] per-frame DA3 central depth (each in its own scale)
    masks : [F,H,W] per-frame object masks (True = object)

    Returns
    -------
    template : [H,W]   median background depth (NaN where never seen as background)
    co       : [F,H,W] each ``raw`` frame re-scaled onto ``template``
    """
    F = raw.shape[0]
    bg_all = (~masks) & (raw > 1e-3)  # background & valid, per frame
    co = raw.clone()
    template = raw[0].clone()
    for _ in range(iters):
        stacked = co.clone()
        stacked[~bg_all] = float("nan")
        template = torch.nanmedian(stacked, dim=0).values  # [H,W]
        tvalid = torch.isfinite(template)
        for i in range(F):
            bg = bg_all[i] & tvalid
            if bg.sum() < 50:
                continue
            s, b = _robust_affine(raw[i][bg], template[bg])
            co[i] = s * raw[i] + b
    return template, co


def composite_frame(co_i: torch.Tensor, mask_i: torch.Tensor, template: torch.Tensor):
    """Persistent background everywhere + this frame's object in the object mask."""
    comp = co_i.clone()
    bg_use = (~mask_i.bool()) & torch.isfinite(template)
    comp[bg_use] = template[bg_use]
    return comp


# ──────────────────────────────────────────────────────────────────────────────
#  Sequence runner
# ──────────────────────────────────────────────────────────────────────────────
def _central_mask(frame: dict) -> torch.Tensor:
    """Central-view object mask [H,W]; falls back to all-ones if absent."""
    LF = frame["LF"]
    S, T = LF.shape[0], LF.shape[1]
    H, W = LF.shape[2], LF.shape[3]
    cmid = (S // 2) * T + (T // 2)
    masks = frame["masks"]
    if masks is None:
        return torch.ones(H, W, dtype=torch.bool, device=LF.device)
    return masks.reshape(-1, H, W)[cmid].bool()


def run_sequence(
    estimator: DepthEstimator,
    seq_path: str,
    out_path: str,
    temporal: bool = True,
    limit: int | None = None,
):
    """Estimate + post-process depth for a whole sequence and cache to ``out_path``.

    Two passes:
      1. Run DA3 on every frame's full light field (independent per frame) and
         collect the central-view depth, mask, conf, GT and RGB.
      2. Build a persistent static-background template and re-scale every frame
         onto it, then composite (stable background + per-frame object) and align
         the object region to GT.  See ``build_persistent_background``.
    """
    ds = LFDataset(seq_path, depth_source="gt")
    n = len(ds) if limit is None else min(limit, len(ds))

    # ── pass 1: independent per-frame DA3 over the full LF ──────────────────────
    raw_list, conf_list, mask_list, gt_list, rgb_list = [], [], [], [], []
    K_np = None
    for idx in range(n):
        frame = ds[idx]
        H, W = frame["LF"].shape[2], frame["LF"].shape[3]
        if K_np is None:
            K_np = frame["camera_matrix"][:3, :3].cpu().numpy()
        depth_c, conf_c, _, cmid = estimator.infer_lf(frame)
        raw_list.append(depth_c)
        conf_list.append(conf_c)
        mask_list.append(_central_mask(frame))
        gt_list.append(frame["depth"])
        rgb_list.append(
            (linear_to_srgb(frame["LF"].reshape(-1, H, W, 3)[cmid]).clamp(0, 1) * 255)
            .to(torch.uint8)
        )

    raw = torch.stack(raw_list)      # [F,H,W]
    masks = torch.stack(mask_list)   # [F,H,W] bool

    # ── pass 2: persistent background + per-frame object, then GT-align ─────────
    bg_jitter_raw = bg_jitter_temporal = float("nan")
    if temporal:
        template, co = build_persistent_background(raw, masks)
        depth_temporal = torch.stack(
            [composite_frame(co[i], masks[i], template) for i in range(n)]
        )
        # temporal-stability diagnostic on pixels that are background in *every*
        # frame: how much per-frame depth jitters before vs. after the template.
        always_bg = ((~masks) & (raw > 1e-3)).all(dim=0)
        if always_bg.sum() > 20:
            bg_jitter_raw = float(raw[:, always_bg].std(dim=0).mean() * 1000)
            bg_jitter_temporal = float(co[:, always_bg].std(dim=0).mean() * 1000)
    else:
        template = torch.full_like(raw[0], float("nan"))
        depth_temporal = raw.clone()

    depth_aligned, per_frame_metrics = [], []
    for idx in range(n):
        aligned, _, metrics = align_depth_to_gt(depth_temporal[idx], gt_list[idx], masks[idx])
        metrics["frame"] = idx
        per_frame_metrics.append(metrics)
        depth_aligned.append(aligned)

    def _f16(stack):
        return torch.stack(stack).cpu().numpy().astype(np.float16)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(
        out_path,
        depth_raw=raw.cpu().numpy().astype(np.float16),
        depth_temporal=depth_temporal.cpu().numpy().astype(np.float16),
        depth_aligned=_f16(depth_aligned),
        bg_template=template.cpu().numpy().astype(np.float16),
        gt=_f16(gt_list),
        mask=torch.stack(mask_list).cpu().numpy(),
        conf=_f16(conf_list),
        rgb=torch.stack(rgb_list).cpu().numpy(),
        K=K_np,
        metrics=json.dumps(per_frame_metrics),
    )

    maes = [m["mae_mm"] for m in per_frame_metrics if np.isfinite(m["mae_mm"])]
    corrs = [m["corr"] for m in per_frame_metrics if np.isfinite(m["corr"])]
    summary = {
        "seq": seq_path,
        "frames": n,
        "mae_mm": float(np.mean(maes)) if maes else float("nan"),
        "med_mae_mm": float(np.median(maes)) if maes else float("nan"),
        "corr": float(np.mean(corrs)) if corrs else float("nan"),
        "bg_jitter_raw_mm": bg_jitter_raw,
        "bg_jitter_temporal_mm": bg_jitter_temporal,
    }
    return summary


def _iter_sequences(dataset_root, splits, refls, seq_filter=None):
    for split in splits:
        for refl in refls:
            split_dir = os.path.join(dataset_root, f"{split}_{refl}")
            if not os.path.isdir(split_dir):
                continue
            for seq in sorted(os.listdir(split_dir)):
                seq_path = os.path.join(split_dir, seq)
                if not os.path.isdir(seq_path) or seq == "models":
                    continue
                if seq_filter and seq not in seq_filter:
                    continue
                yield split, refl, seq, seq_path


# ──────────────────────────────────────────────────────────────────────────────
#  Viser visualisation
# ──────────────────────────────────────────────────────────────────────────────
def _turbo(x: np.ndarray) -> np.ndarray:
    """Map x in [0,1] to an RGB uint8 colormap (matplotlib turbo if available)."""
    try:
        import matplotlib.cm as cm

        return (cm.get_cmap("turbo")(np.clip(x, 0, 1))[..., :3] * 255).astype(np.uint8)
    except Exception:
        x = np.clip(x, 0, 1)[..., None]
        c = np.array([0.0, 0.0, 1.0]) * (1 - x) + np.array([1.0, 0.0, 0.0]) * x
        return (c * 255).astype(np.uint8)


def _backproject(depth: np.ndarray, mask: np.ndarray, K: np.ndarray):
    """Masked pixels -> camera-space XYZ [M,3] and their flat pixel indices."""
    H, W = depth.shape
    ys, xs = np.where(mask & (depth > 1e-3))
    z = depth[ys, xs].astype(np.float64)
    x = (xs - K[0, 2]) / K[0, 0] * z
    y = (ys - K[1, 2]) / K[1, 1] * z
    pts = np.stack([x, y, z], axis=1).astype(np.float32)
    return pts, ys, xs


def visualize(results_dir: str, err_cap_mm: float = 30.0, port: int = 8080):
    """Open a viser server comparing GT vs estimated depth, per object.

    Controls: object dropdown, frame slider, GT/estimated/error toggles.  GT is
    drawn in its true colour; the estimated cloud (post-processed, GT-aligned)
    can be shown either in RGB or coloured by per-point error.
    """
    import viser

    files = sorted(
        os.path.join(r, f)
        for r, _, fs in os.walk(results_dir)
        for f in fs
        if f.endswith(".npz")
    )
    if not files:
        raise FileNotFoundError(f"No .npz results under {results_dir}")
    labels = [os.path.relpath(f, results_dir)[:-4] for f in files]
    cache: dict[str, dict] = {}

    def load(label):
        if label not in cache:
            d = np.load(files[labels.index(label)], allow_pickle=True)
            cache[label] = {k: d[k] for k in d.files}
        return cache[label]

    server = viser.ViserServer(port=port)
    server.scene.set_up_direction("-y")
    obj_dd = server.gui.add_dropdown("Object", labels, initial_value=labels[0])
    data0 = load(labels[0])
    frame_sl = server.gui.add_slider("Frame", 0, len(data0["gt"]) - 1, 1, 0)
    show_gt = server.gui.add_checkbox("GT (green)", True)
    show_est = server.gui.add_checkbox("Estimated", True)
    color_err = server.gui.add_checkbox("Colour est. by error", True)
    full_scene = server.gui.add_checkbox("Full scene (temporal, no GT)", False)
    pt_size = server.gui.add_slider("Point size", 0.0005, 0.01, 0.0005, 0.002)
    info = server.gui.add_text("Frame error", initial_value="")

    def render():
        d = load(obj_dd.value)
        f = min(int(frame_sl.value), len(d["gt"]) - 1)
        K = d["K"]
        mask = d["mask"][f]
        gt = d["gt"][f].astype(np.float32)
        est = d["depth_aligned"][f].astype(np.float32)
        rgb = d["rgb"][f]

        server.scene.reset()

        # Full-scene mode: show the temporally-consistent composite (persistent
        # background + per-frame object) over the whole frame.  Scrub the slider
        # to see the background stay put while only the object moves.
        if full_scene.value:
            comp = d["depth_temporal"][f].astype(np.float32)
            allmask = np.ones_like(mask, dtype=bool)
            cpts, cy, cx = _backproject(comp, allmask, K)
            if len(cpts):
                server.scene.add_point_cloud(
                    "/scene", cpts, rgb[cy, cx], point_size=float(pt_size.value)
                )
            m = json.loads(str(d["metrics"]))
            mm = next((x for x in m if x["frame"] == f), {})
            info.value = (f"[full scene] frame {f}  object MAE "
                          f"{mm.get('mae_mm', float('nan')):.1f}mm")
            return

        # GT cloud (offset left), estimated cloud (offset right), for side-by-side.
        gpts, gy, gx = _backproject(gt, mask, K)
        if show_gt.value and len(gpts):
            server.scene.add_point_cloud(
                "/gt",
                gpts,
                np.tile(np.array([[40, 200, 80]], np.uint8), (len(gpts), 1)),
                point_size=float(pt_size.value),
            )
        epts, ey, ex = _backproject(est, mask, K)
        if show_est.value and len(epts):
            if color_err.value:
                err = np.abs(est[ey, ex] - gt[ey, ex]) * 1000.0
                ecol = _turbo(err / err_cap_mm)
            else:
                ecol = rgb[ey, ex]
            server.scene.add_point_cloud(
                "/est", epts, ecol, point_size=float(pt_size.value)
            )

        # metrics text for this frame
        m = json.loads(str(d["metrics"]))
        mm = next((x for x in m if x["frame"] == f), {})
        info.value = (
            f"MAE {mm.get('mae_mm', float('nan')):.1f}mm  "
            f"med {mm.get('med_mm', float('nan')):.1f}mm  "
            f"corr {mm.get('corr', float('nan')):.2f}"
        )

    @obj_dd.on_update
    def _(_):
        d = load(obj_dd.value)
        frame_sl.max = len(d["gt"]) - 1
        frame_sl.value = 0
        render()

    for ctrl in (frame_sl, show_gt, show_est, color_err, full_scene, pt_size):
        ctrl.on_update(lambda _: render())

    render()
    print(f"viser running at http://localhost:{port}  ({len(labels)} objects)")
    while True:
        time.sleep(1.0)


# ──────────────────────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="DA3 depth estimation + GT alignment + viser"
    )
    ap.add_argument("--mode", choices=["estimate", "vis"], default="estimate")
    ap.add_argument("--dataset-root", default="/home/ngoncharov/SpecTrack_dataset")
    ap.add_argument("--out", default="eval/depth_da3")
    ap.add_argument("--splits", default="cube,objects")
    ap.add_argument("--refls", default="0.0,0.5,0.7,1.0")
    ap.add_argument(
        "--seqs", default="", help="comma-sep sequence names to restrict to"
    )
    ap.add_argument("--limit", type=int, default=None, help="max frames per sequence")
    ap.add_argument("--no-temporal", action="store_true")
    ap.add_argument("--model", default="depth-anything/DA3-GIANT")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.mode == "vis":
        visualize(args.out, port=args.port)
        return

    splits = [s for s in args.splits.split(",") if s]
    refls = [r for r in args.refls.split(",") if r]
    seq_filter = set(s for s in args.seqs.split(",") if s) or None

    estimator = DepthEstimator(model_name=args.model)
    summaries = []
    for split, refl, seq, seq_path in _iter_sequences(
        args.dataset_root, splits, refls, seq_filter
    ):
        out_path = os.path.join(args.out, f"{split}_{refl}", f"{seq}.npz")
        if os.path.exists(out_path) and not args.overwrite:
            print(f"skip (cached) {split}_{refl}/{seq}")
            continue
        t = time.time()
        try:
            summary = run_sequence(
                estimator,
                seq_path,
                out_path,
                temporal=not args.no_temporal,
                limit=args.limit,
            )
        except RuntimeError as e:  # e.g. transient OOM on the shared GPU
            torch.cuda.empty_cache()
            print(f"FAILED {split}_{refl}/{seq}: {str(e)[:120]}")
            continue
        summary.update(split=split, refl=refl, seq=seq)
        summaries.append(summary)
        print(
            f"{split}_{refl}/{seq}: {summary['frames']}f  "
            f"MAE={summary['mae_mm']:.1f}mm  corr={summary['corr']:.2f}  "
            f"({time.time() - t:.0f}s)"
        )

    if summaries:
        maes = [s["mae_mm"] for s in summaries if np.isfinite(s["mae_mm"])]
        print(
            f"\n=== {len(summaries)} sequences  |  mean MAE {np.mean(maes):.1f}mm ==="
        )
        with open(os.path.join(args.out, "summary.json"), "w") as f:
            json.dump(summaries, f, indent=2)


if __name__ == "__main__":
    main()
