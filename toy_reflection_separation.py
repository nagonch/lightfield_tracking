"""
toy_reflection_separation.py
============================
A tiny, self-contained illustration of the two ideas behind ReLiFT's
diffuse / reflection split, on 1-D toy data, rendered as one figure.

Generative model (1-D luminance, mirrors  y = a*D + (1-a)*E(reflect) ):

    y[p, v] = alpha * D[p] + (1 - alpha) * E(theta[p] + delta[v]) + noise

  D[p]      per-point diffuse luminance      (VIEW-INDEPENDENT -> constant over v)
  E(.)      one shared, distant 1-D world    (the environment map)
  theta[p]  a point's base reflected angle   (set by its surface normal)
  delta[v]  a tiny per-subview sweep of the  (the light-field baseline; small,
            reflected ray                     ~the same for every point)

THE ONE IDENTITY everything hangs on: D is constant across views, so the
per-point standard deviation over the M subviews is

    sigma[p] = std_v(y) = (1 - alpha) * std_v( E(theta[p] + delta) ).

So sigma is a direct read-out of (1 - alpha), scaled by the LOCAL CONTRAST of
the bit of world the point happens to reflect. Two robustness facts then turn
a noisy per-point quantity into one alpha per sequence:

  * within a frame, a few points OVER-read (bad normals / grazing Fresnel blow
    up their variance) -> take a LOW quantile (q=0.30) to dodge that high tail;
  * across frames, whole poses UNDER-read (they reflect a flat patch of the
    world) -> alpha is a material constant, so take a HIGH quantile (q=0.80)
    over the per-frame stats to recover it from the informative poses.

Run inside any env with numpy + matplotlib + scipy:
    python toy_reflection_separation.py
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# --------------------------------------------------------------------------- #
# Toy world + sensor
# --------------------------------------------------------------------------- #
RNG = np.random.default_rng(7)
TWO_PI = 2.0 * np.pi
M = 25                                  # 5x5 sub-aperture views
DELTA = 0.13                            # half-width of the per-subview ray sweep
DELTAS = np.linspace(-DELTA, DELTA, M)  # the light-field baseline, in radians
NOISE = 0.010                           # sensor noise -> the calibration FLOOR
N_BINS = 220                            # env-map resolution for the separation
WITHIN_Q = 0.30                         # low within-frame quantile (dodge artifacts)
ACROSS_Q = 0.80                         # high across-frame quantile (recover signal)


def make_env(theta):
    """A 1-D world: pervasive texture on a 'busy' side, two bright windows, and a
    flat/dim arc (near theta ~ 4.6) where reflections under-read."""
    t = np.asarray(theta, dtype=float) % TWO_PI
    busy = 0.5 + 0.5 * np.tanh(np.cos(t - 1.5) / 0.40)          # ~1 near 1.5, ~0 near 4.6
    tex = 0.12 * np.sin(5 * t + 0.3) + 0.07 * np.sin(11 * t + 1.0) + 0.05 * np.sin(19 * t)
    e = 0.20 + busy * tex                                       # pervasive contrast
    e = e + 0.75 * np.exp(-0.5 * ((t - 1.00) / 0.13) ** 2)      # bright light
    e = e + 0.45 * np.exp(-0.5 * ((t - 2.40) / 0.24) ** 2)      # softer window
    return np.clip(e, 0.0, 1.0)


def simulate_frame(alpha, phi, n=6000, artifact_frac=0.10):
    """One frame: the object, rotated by `phi`, turns an arc of normals to us."""
    theta = (phi + RNG.uniform(0.0, 3.0, n)) % TWO_PI    # reflected angle per point
    D = RNG.uniform(0.2, 0.8, n)                          # per-point diffuse colour
    artifact = RNG.random(n) < artifact_frac             # bad normals / grazing
    amp = np.where(artifact, RNG.uniform(2.5, 4.5, n), 1.0)   # inflate their sweep
    ang = theta[:, None] + amp[:, None] * DELTAS[None, :]     # [n, M] reflected dirs
    R = make_env(ang)                                        # reflected world / subview
    y = alpha * D[:, None] + (1.0 - alpha) * R + NOISE * RNG.standard_normal((n, M))
    return dict(y=y, theta=theta, D=D, artifact=artifact)


# --------------------------------------------------------------------------- #
# The alpha statistic
# --------------------------------------------------------------------------- #
def per_point_sigma(y):
    return y.std(axis=1)                 # sigma[p] = std over the M subviews


def frame_stat(y):
    # The real method also weights by frontal^2 * viewcount before this quantile
    # (suppressing grazing points); here the low quantile alone carries the load.
    return np.quantile(per_point_sigma(y), WITHIN_Q)


def accum_stat(alpha, n_frames=24):
    phis = np.linspace(0, TWO_PI, n_frames, endpoint=False)
    s = np.array([frame_stat(simulate_frame(alpha, p, n=4000)["y"]) for p in phis])
    return np.quantile(s, ACROSS_Q)


def calibrate(alphas=(0.0, 0.25, 0.5, 0.75, 1.0)):
    """Fit stat = floor + slope * (1 - alpha) over known reflectivities."""
    stats = np.array([accum_stat(a) for a in alphas])
    x = 1.0 - np.array(alphas)
    floor, slope = np.linalg.lstsq(np.vstack([np.ones_like(x), x]).T, stats, rcond=None)[0]
    return floor, slope, np.array(alphas), stats


def stat_to_alpha(stat, floor, slope):
    return float(np.clip(1.0 - (stat - floor) / slope, 0.03, 0.97))


# --------------------------------------------------------------------------- #
# The actual separation: recover per-point D and the shared map E (alternating
# least squares -- exactly the structure of the real joint fit, just tiny).
# --------------------------------------------------------------------------- #
def separate(y, theta, alpha, n_iter=12):
    ang = (theta[:, None] + DELTAS[None, :]) % TWO_PI
    b = np.clip((ang / TWO_PI * N_BINS).astype(int), 0, N_BINS - 1)
    bflat = b.ravel()
    D = y.mean(axis=1).copy()                       # init: per-point mean
    E = np.zeros(N_BINS)
    cov = np.zeros(N_BINS, bool)
    for _ in range(n_iter):
        r = (y - alpha * D[:, None]) / (1.0 - alpha)         # reflection estimate
        acc = np.zeros(N_BINS); cnt = np.zeros(N_BINS)
        np.add.at(acc, bflat, r.ravel())                     # splat into env bins
        np.add.at(cnt, bflat, 1.0)
        cov = cnt > 0
        E = np.where(cov, acc / np.maximum(cnt, 1.0), 0.0)
        D = np.clip(((y - (1.0 - alpha) * E[b]).mean(axis=1)) / alpha, 0.0, 1.0)
    return E, D, cov


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
C_WORLD = "#178A8A"   # the environment / reflection (teal)
C_DIFF  = "#E08A3C"   # diffuse (warm orange)
C_TRUE  = "#2B2B2B"   # ground truth (near-black)
C_EST   = "#C8453B"   # estimate (crimson)
C_FLOOR = "#7E5BA6"   # under-reading / diffuse floor (purple)
C_ART   = "#3F6DB5"   # artifact tail (blue)
C_HL    = "#E6A100"   # quantile lines (gold)


def main():
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 160, "font.family": "DejaVu Sans",
        "font.size": 10, "axes.titlesize": 11.5, "axes.titleweight": "bold",
        "axes.titlepad": 7, "axes.labelsize": 9.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.18, "grid.linewidth": 0.7,
        "axes.axisbelow": True, "legend.frameon": False, "legend.fontsize": 8,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    })

    # ---- precompute everything --------------------------------------------- #
    floor, slope, cal_a, cal_s = calibrate()
    ALPHA_TRUE = 0.45

    demo = simulate_frame(ALPHA_TRUE, phi=0.8, n=6000)
    sigma = per_point_sigma(demo["y"])

    alphas_show = [0.2, 0.5, 0.8]
    sig_by_alpha = {a: per_point_sigma(simulate_frame(a, phi=0.8, n=6000)["y"])
                    for a in alphas_show}

    n_frames = 24
    phis = np.linspace(0, TWO_PI, n_frames, endpoint=False)
    seq = [simulate_frame(ALPHA_TRUE, p, n=4000) for p in phis]
    seq_stats = np.array([frame_stat(f["y"]) for f in seq])
    running_alpha = np.array([
        stat_to_alpha(np.quantile(seq_stats[:k], ACROSS_Q) if k >= 4 else seq_stats[k - 1],
                      floor, slope)
        for k in range(1, n_frames + 1)
    ])

    E1, D1, cov1 = separate(seq[0]["y"], seq[0]["theta"], ALPHA_TRUE)
    y_all = np.concatenate([f["y"] for f in seq])
    th_all = np.concatenate([f["theta"] for f in seq])
    Eall, _, covall = separate(y_all, th_all, ALPHA_TRUE)
    bc = (np.arange(N_BINS) + 0.5) / N_BINS * TWO_PI

    fig, axes = plt.subplots(2, 3, figsize=(17, 9.6), constrained_layout=True)
    fig.suptitle("Reflection separation by example   —   view-variance reads out the "
                 "material;  motion fills in the world",
                 fontsize=15, fontweight="bold")

    # ---- (a) the world + the tiny sliding window --------------------------- #
    ax = axes[0, 0]
    th = np.linspace(0, TWO_PI, 1500)
    ax.fill_between(th, 0, make_env(th), color=C_WORLD, alpha=0.13, lw=0)
    ax.plot(th, make_env(th), color=C_WORLD, lw=2.2)
    for c, col, txt in [(1.0, C_EST, "contrasty patch\nbig swing\nreads reflective"),
                        (4.6, C_FLOOR, "flat patch\ntiny swing\nlooks diffuse")]:
        ax.axvspan(c - DELTA, c + DELTA, color=col, alpha=0.25, lw=0)
        ax.annotate(txt, xy=(c, float(make_env([c])[0])),
                    xytext=(c, 0.97 if c < 2 else 0.62),
                    ha="center", va="top", fontsize=8, color=col, fontweight="bold",
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.4))
    ax.set_title("(a)  One shared world; each point reflects a tiny sliding window")
    ax.set_xlabel("reflected ray angle  θ  (rad)")
    ax.set_ylabel("world radiance  E(θ)")
    ax.set_xlim(0, TWO_PI); ax.set_ylim(0, 1.07)

    # ---- (b) what one point sees across subviews --------------------------- #
    ax = axes[0, 1]
    order = np.argsort(sigma)
    picks = [(order[int(0.985 * len(order))], C_WORLD, "reflective · contrasty"),
             (order[int(0.55 * len(order))], "#6FB0B0", "reflective · mild"),
             (order[int(0.05 * len(order))], C_FLOOR, "reflects a flat patch")]
    for idx, col, lab in picks:
        ax.plot(np.arange(M), demo["y"][idx], "-o", ms=3.5, lw=1.6, color=col,
                label=f"{lab}   (σ={sigma[idx]:.3f})")
    ax.axhline(demo["y"][picks[2][0]].mean(), color=C_DIFF, lw=1.5, ls="--",
               label="pure diffuse (α=1): flat")
    ax.set_title("(b)  A diffuse colour stays put; a reflection swings")
    ax.set_xlabel("sub-aperture view index  v")
    ax.set_ylabel("observed luminance  y(v)")
    ax.legend(loc="upper left")
    ax.text(0.975, 0.04,
            r"$y_{p,v}=\alpha D_p+(1-\alpha)\,E(\theta_p+\delta_v)$" + "\n"
            r"swing amplitude $\propto (1-\alpha)\times$ local contrast",
            transform=ax.transAxes, fontsize=8.5, va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.8"))

    # ---- (c) the factorization: distribution shifts with alpha ------------- #
    ax = axes[0, 2]
    xs = np.linspace(0, float(sig_by_alpha[0.2].max()) * 1.02, 400)
    cmap = {0.2: "#1B4F72", 0.5: "#2E86C1", 0.8: "#9CCBEA"}
    for a in alphas_show:
        ys = gaussian_kde(sig_by_alpha[a])(xs)
        ax.fill_between(xs, ys, color=cmap[a], alpha=0.28, lw=0)
        ax.plot(xs, ys, color=cmap[a], lw=2, label=f"α = {a}")
        ax.axvline(np.median(sig_by_alpha[a]), color=cmap[a], lw=1.1, ls=":")
    ax.set_title("(c)  More diffuse (α↑) ⇒ the whole σ distribution collapses to 0")
    ax.set_xlabel(r"per-point view-std  $\sigma_p=\mathrm{std}_v\,y$")
    ax.set_ylabel("density")
    ax.legend(title=r"$\sigma_p=(1-\alpha)\,\mathrm{std}_v\,E$")

    # ---- (d) within-frame: the tails + the low quantile -------------------- #
    ax = axes[1, 0]
    bins = np.linspace(0, float(sigma.max()) * 1.02, 70)
    bw = bins[1] - bins[0]
    N = len(sigma)
    ax.hist(sigma, bins=bins, density=True, color="#D6DCE0",
            edgecolor="white", linewidth=0.3, label="all points")
    counts_art, _ = np.histogram(sigma[demo["artifact"]], bins=bins)
    ax.bar(bins[:-1], counts_art / (N * bw), width=bw, align="edge",
           color=C_ART, alpha=0.55, label="artifacts → high tail")
    xs2 = np.linspace(0, float(sigma.max()) * 1.02, 400)
    kde = gaussian_kde(sigma)
    ax.plot(xs2, kde(xs2), color=C_TRUE, lw=1.6)
    q30 = np.quantile(sigma, WITHIN_Q)
    ax.axvline(q30, color=C_HL, lw=2.4, ls="--", label=f"q=0.30 read = {q30:.3f}")
    ax.axvline(np.median(sigma), color="0.45", lw=1.2, ls=":",
               label=f"median = {np.median(sigma):.3f}")
    ymax = ax.get_ylim()[1]
    ax.annotate("diffuse floor\n(flat patches, noise)\nUNDER-reads",
                xy=(np.quantile(sigma, 0.05), 0.25 * ymax),
                xytext=(np.quantile(sigma, 0.04), 0.66 * ymax),
                color=C_FLOOR, fontsize=8, fontweight="bold", ha="left",
                arrowprops=dict(arrowstyle="-|>", color=C_FLOOR, lw=1.3))
    ax.annotate("artifact tail\nOVER-reads",
                xy=(np.quantile(sigma, 0.985), 0.10 * ymax),
                xytext=(np.quantile(sigma, 0.72), 0.5 * ymax),
                color=C_ART, fontsize=8, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=C_ART, lw=1.3))
    ax.set_title("(d)  Within a frame: a LOW quantile dodges the artifact tail")
    ax.set_xlabel(r"per-point view-std  $\sigma_p$")
    ax.set_ylabel("density")
    ax.legend(loc="upper right")

    # ---- (e) across-frame: poses under-read -> high quantile --------------- #
    ax = axes[1, 1]
    fr = np.arange(1, n_frames + 1)
    ax.plot(fr, seq_stats, "-o", color="#9AA5B1", ms=5, lw=1.4,
            label="per-frame stat $s_f$  (q=0.30)")
    low = seq_stats < np.quantile(seq_stats, 0.30)
    ax.scatter(fr[low], seq_stats[low], color=C_FLOOR, s=60, zorder=5,
               label="uninformative pose (flat reflection)")
    q80 = np.quantile(seq_stats, ACROSS_Q)
    ax.axhline(q80, color=C_HL, lw=2.4, ls="--", label=f"q=0.80 across frames = {q80:.3f}")
    ax.set_title("(e)  Across frames: poses under-read, so read a HIGH quantile")
    ax.set_xlabel("frame")
    ax.set_ylabel("frame statistic  $s_f$")
    ax.legend(loc="lower right")
    axin = ax.inset_axes([0.13, 0.60, 0.42, 0.36])
    axin.axhline(ALPHA_TRUE, color=C_TRUE, ls="--", lw=1.3)
    axin.plot(fr, running_alpha, color=C_EST, lw=1.9)
    axin.set_title(f"estimate → {running_alpha[-1]:.2f}   (true {ALPHA_TRUE})", fontsize=8)
    axin.set_xlabel("frames seen", fontsize=7)
    axin.set_ylabel("α estimate", fontsize=7)
    axin.set_ylim(0, 1); axin.tick_params(labelsize=6.5); axin.grid(alpha=0.15)

    # ---- (f) the decomposition result + fill-in ---------------------------- #
    ax = axes[1, 2]
    ax.plot(bc, make_env(bc), color=C_TRUE, lw=2.6, alpha=0.85, label="true world  E(θ)")
    ax.plot(bc[cov1], E1[cov1], color=C_DIFF, lw=1.9, label="recovered, 1 frame (partial)")
    ax.plot(bc, np.where(covall, Eall, np.nan), color=C_WORLD, lw=1.6,
            label="recovered, 24 frames (filled in)")
    ax.set_title("(f)  Joint fit recovers the shared map — and it fills in as it turns")
    ax.set_xlabel("reflected ray angle  θ")
    ax.set_ylabel("E(θ)")
    ax.set_xlim(0, TWO_PI); ax.set_ylim(0, 1.07)
    ax.legend(loc="upper right")
    axin = ax.inset_axes([0.60, 0.13, 0.36, 0.37])
    axin.scatter(seq[0]["D"], D1, s=5, alpha=0.22, color=C_DIFF, edgecolor="none")
    axin.plot([0, 1], [0, 1], color=C_TRUE, ls="--", lw=1)
    rmse = float(np.sqrt(np.mean((seq[0]["D"] - D1) ** 2)))
    axin.set_title(f"diffuse D  (RMSE {rmse:.3f})", fontsize=7.5)
    axin.set_xlabel("true", fontsize=7); axin.set_ylabel("recovered", fontsize=7)
    axin.set_xlim(0.15, 0.85); axin.set_ylim(0.15, 0.85)
    axin.tick_params(labelsize=6.5); axin.grid(alpha=0.15)

    out = "/home/ngoncharov/cvpr2026/ReLiFT-6DoF/toy_reflection_separation.png"
    fig.savefig(out)
    print("saved", out)
    print(f"calibration: floor={floor:.4f} slope={slope:.4f}  | "
          f"alpha est={running_alpha[-1]:.3f} (true {ALPHA_TRUE})  | diffuse RMSE={rmse:.3f}")


if __name__ == "__main__":
    main()
