"""Eval from cache: single-frame vs accumulated(p80) MAE, linear vs nonlinear cal."""
import pickle
import numpy as np

CACHE = "/tmp/alpha_cache.pkl"
SPLITS = ["cube", "objects"]
REFLS = ["0.0", "0.5", "0.7", "1.0"]
Q_IN, FP = 0.3, 2.0


def wpct(x, w, q):
    o = np.argsort(x); xs, ws = x[o], w[o]
    cw = (np.cumsum(ws) - 0.5 * ws) / max(ws.sum(), 1e-8)
    return float(np.interp(q, cw, xs))


def frame_stat(rec):
    w = (np.clip(rec["frontal"], 0, 1) ** FP) * rec["vc"]
    return wpct(rec["std"], w, Q_IN)


def fit_lin(xs, ys):  # stat = f + s*(1-a)
    A = np.vstack([np.ones_like(ys), ys]).T
    f, s = np.linalg.lstsq(A, xs, rcond=None)[0]
    return f, s


def main():
    recs = pickle.load(open(CACHE, "rb"))
    for r in recs:
        r["s"] = frame_stat(r)

    # ---------- single-frame ----------
    xs = np.array([r["s"] for r in recs])
    ys = np.array([1.0 - r["alpha"] for r in recs])
    f, s = fit_lin(xs, ys)
    print(f"[single-frame] FLOOR={f:.4f} SLOPE={s:.4f}")
    ae = []
    for split in SPLITS:
        for rr in REFLS:
            a = 1.0 - float(rr)
            sub = [x for x, r in zip(xs, recs) if r["split"] == split and r["alpha"] == a]
            ah = [1 - min(max((x - f) / s, 0), 1) for x in sub]
            e = [abs(x - a) for x in ah]
            ae += e
            print(f"  {split:8s} {a:4.2f}  aHat {np.mean(ah):5.2f}±{np.std(ah):4.2f}  MAE {np.mean(e):.3f}")
    print(f"  single-frame overall MAE {np.mean(ae):.3f}\n")

    # ---------- accumulated: running p80 over frames in temporal order ----------
    # group by (split,alpha,seq), sort by fi, compute estimate after k frames
    groups = {}
    for r in recs:
        groups.setdefault((r["split"], r["alpha"], r["seq"]), []).append(r)
    # calibration from full-sequence p80
    pk_x, pk_y = [], []
    for key, g in groups.items():
        st = np.array([x["s"] for x in sorted(g, key=lambda z: z["fi"])])
        pk_x.append(np.percentile(st, 80)); pk_y.append(1.0 - key[1])
    f2, s2 = fit_lin(np.array(pk_x), np.array(pk_y))
    print(f"[accumulated p80] FLOOR={f2:.4f} SLOPE={s2:.4f}")
    for K in [1, 3, 5, 10]:
        ae = []
        for key, g in groups.items():
            st = np.array([x["s"] for x in sorted(g, key=lambda z: z["fi"])][:K])
            est = np.percentile(st, 80)
            ah = 1 - min(max((est - f2) / s2, 0), 1)
            ae.append(abs(ah - key[1]))
        print(f"  after K={K:2d} frames: overall MAE {np.mean(ae):.3f}")

    # ---------- nonlinear (quadratic) single-frame calibration ----------
    # fit (1-a) = poly(stat) directly, clip
    coef = np.polyfit(xs, ys, 2)
    ae = []
    for split in SPLITS:
        for rr in REFLS:
            a = 1.0 - float(rr)
            sub = [x for x, r in zip(xs, recs) if r["split"] == split and r["alpha"] == a]
            ah = [1 - min(max(np.polyval(coef, x), 0), 1) for x in sub]
            ae += [abs(x - a) for x in ah]
    print(f"\n[single-frame quadratic] overall MAE {np.mean(ae):.3f}")


if __name__ == "__main__":
    main()
