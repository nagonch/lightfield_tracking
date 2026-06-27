"""
toy_epi_aggregation.py
======================
A *simulated* flatland illustration of the insight behind ``lf_depth.py``.

We model the optics for real: a row of pinhole cameras on a baseline, linear
ray tracing in a 2-D world, and one glossy patch whose radiance follows the
dichromatic model

        L(s) = alpha * D  +  (1 - alpha) * E(s)

with D a constant diffuse colour (view-INDEPENDENT) and E(s) the environment
colour reached by the reflected ray for camera ``s`` (view-DEPENDENT).

The reflected sources sit BEHIND the cameras, so the forward-looking cameras
never image them directly (no real EPI lines for them).  Their reflection still
appears on the patch as a sweeping colour; in the EPI we draw their *imaginary*
lines — the virtual images behind the mirror — which cross the patch line
exactly where each colour bump occurs.

Three matplotlib plots:
  (1) the flatland scene  (cameras, glossy patch, reflected sources behind);
  (2) the observed EPI  (white; the real patch line + imaginary source lines);
  (3) the per-channel radiance sampled ALONG the patch line, with smooth R/G/B
      curves — flat at alpha*D, with sparse reflection bumps a robust aggregator
      rejects.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────────────
#  Scene  (x = horizontal, z = depth; cameras on z = 0 looking towards +z)
# ──────────────────────────────────────────────────────────────────────────────
ALPHA = 0.5                                  # dichromatic mix
D_DIFFUSE = np.array([0.16, 0.55, 0.62])     # patch diffuse colour (muted cyan)
DISPLAY_BG = np.array([1.0, 1.0, 1.0])       # what a camera ray that hits nothing shows
REFLECT_BG = np.array([0.0, 0.0, 0.0])       # what the patch reflects in empty directions
PATCH_C = np.array([0.0, 1.20])              # glossy patch centre (x, z)
PATCH_HW = 0.085                             # patch half-width
PATCH_N = np.array([0.0, -1.0])              # patch normal (faces the cameras)

# bright reflected sources, placed BEHIND the cameras (z < 0): (x, z, half-width, colour)
EMITTERS = [
    (-0.45, -0.70, 0.06, np.array([1.00, 0.22, 0.20])),   # red
    ( 0.00, -0.70, 0.06, np.array([0.30, 0.95, 0.35])),   # green
    ( 0.45, -0.70, 0.06, np.array([0.35, 0.50, 1.00])),   # blue
]

# Flat segment list. kind 0 = solid source, 1 = glossy patch.
SEGS = []
for ex, ez, ehw, col in EMITTERS:
    SEGS.append((np.array([ex - ehw, ez]), np.array([ex + ehw, ez]), col, 0))
SEGS.append((np.array([PATCH_C[0] - PATCH_HW, PATCH_C[1]]),
             np.array([PATCH_C[0] + PATCH_HW, PATCH_C[1]]), D_DIFFUSE, 1))

A = np.array([s[0] for s in SEGS])
B = np.array([s[1] for s in SEGS])
COL = np.array([s[2] for s in SEGS])
KIND = np.array([s[3] for s in SEGS])
E = B - A
PATCH_IDX = len(SEGS) - 1


# ──────────────────────────────────────────────────────────────────────────────
#  Linear-optics ray tracer
# ──────────────────────────────────────────────────────────────────────────────
def nearest_hit(o, d, ignore=None):
    """Closest segment hit by ray o + t d (t>eps). Returns (idx, t) or (-1, inf)."""
    det = E[:, 0] * d[1] - E[:, 1] * d[0]
    r = A - o
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (-r[:, 0] * E[:, 1] + E[:, 0] * r[:, 1]) / det
        u = (d[0] * r[:, 1] - d[1] * r[:, 0]) / det
    ok = (np.abs(det) > 1e-12) & (t > 1e-5) & (u >= 0) & (u <= 1)
    if ignore is not None:
        ok[ignore] = False
    if not ok.any():
        return -1, np.inf
    t = np.where(ok, t, np.inf)
    i = int(np.argmin(t))
    return i, float(t[i])


def seg_normal(i, d):
    e = E[i] / np.linalg.norm(E[i])
    n = np.array([e[1], -e[0]])
    return -n if np.dot(n, d) > 0 else n


def shade(o, d):
    """Radiance along ray o + t d (cameras look forward; one reflection bounce)."""
    i, t = nearest_hit(o, d)
    if i < 0:
        return DISPLAY_BG.copy()
    if KIND[i] == 0:
        return COL[i].copy()
    p = o + t * d
    n = seg_normal(i, d)
    r = d - 2.0 * np.dot(d, n) * n
    j, _ = nearest_hit(p, r, ignore=i)
    env = COL[j] if (j >= 0 and KIND[j] == 0) else REFLECT_BG
    return ALPHA * COL[i] + (1.0 - ALPHA) * env


# ──────────────────────────────────────────────────────────────────────────────
#  Cameras + EPI render
# ──────────────────────────────────────────────────────────────────────────────
S = 121
Q = 360
s_vals = np.linspace(-0.55, 0.55, S)
q_vals = np.linspace(-0.85, 0.85, Q)

epi = np.empty((S, Q, 3))
for si, s in enumerate(s_vals):
    o = np.array([s, 0.0])
    for qi, q in enumerate(q_vals):
        d = np.array([q, 1.0]); d /= np.linalg.norm(d)
        epi[si, qi] = shade(o, d)
epi = np.clip(epi, 0, 1)

# Radiance ALONG the patch line: its dichromatic surface value L = a D + (1-a) E(s).
patch_rad = np.empty((S, 3))
q_patch = (PATCH_C[0] - s_vals) / PATCH_C[1]
n_patch = PATCH_N / np.linalg.norm(PATCH_N)
for si, s in enumerate(s_vals):
    o = np.array([s, 0.0])
    d = PATCH_C - o; d /= np.linalg.norm(d)
    r = d - 2.0 * np.dot(d, n_patch) * n_patch
    j, _ = nearest_hit(PATCH_C, r, ignore=PATCH_IDX)
    env = COL[j] if (j >= 0 and KIND[j] == 0) else REFLECT_BG
    patch_rad[si] = np.clip(ALPHA * D_DIFFUSE + (1.0 - ALPHA) * env, 0, 1)


def smooth(y, sigma):
    """Gaussian smoothing with edge padding (a simple curve fit through samples)."""
    half = max(1, int(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma) ** 2)
    k /= k.sum()
    return np.convolve(np.pad(y, half, mode="edge"), k, mode="valid")


# ──────────────────────────────────────────────────────────────────────────────
#  Plots
# ──────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans",
                     "axes.linewidth": 0.8, "pdf.fonttype": 42})
fig, (ax1, ax2, ax3) = plt.subplots(
    1, 3, figsize=(11.4, 4.0), gridspec_kw={"width_ratios": [1.0, 0.95, 1.15]})

# ---- (1) scene ----------------------------------------------------------------
ax1.set_title("1   flatland scene", loc="left", fontsize=10)
ax1.axhline(0, color="#dddddd", lw=0.8, zorder=0)
for ex, ez, ehw, col in EMITTERS:
    ax1.plot([ex - ehw, ex + ehw], [ez, ez], color=col, lw=5, solid_capstyle="round")
ax1.text(EMITTERS[-1][0] + 0.10, -0.70, "reflected sources\n(behind the cameras)",
         fontsize=7.5, va="center", color="#555")
ax1.plot([PATCH_C[0] - PATCH_HW, PATCH_C[0] + PATCH_HW], [PATCH_C[1]] * 2,
         color=D_DIFFUSE, lw=6, solid_capstyle="round")
ax1.text(PATCH_C[0] + PATCH_HW + 0.04, PATCH_C[1],
         f"glossy patch\n$\\alpha D+(1-\\alpha)E$\n$\\alpha={ALPHA}$",
         fontsize=7.5, va="center", color="#333")
for s in s_vals[::12]:
    ax1.add_patch(plt.Polygon([[s - 0.022, -0.07], [s + 0.022, -0.07], [s, 0.0]],
                              closed=True, facecolor="white", edgecolor="#222", lw=0.7))
ax1.text(-0.72, 0.16, "cameras  (baseline $s$)", fontsize=8, color="#9aa0a6", ha="left")
# reflected sample rays: camera -> patch -> source (behind)
for s in (-0.28, 0.0, 0.28):
    o = np.array([s, 0.0]); d = PATCH_C - o; d /= np.linalg.norm(d)
    i, t = nearest_hit(o, d); p = o + t * d
    n = seg_normal(i, d); r = d - 2 * np.dot(d, n) * n
    j, tj = nearest_hit(p, r); q2 = p + (tj if np.isfinite(tj) else 1.0) * r
    ax1.plot([o[0], p[0]], [o[1], p[1]], color="#999", lw=0.7, alpha=0.8)
    ax1.plot([p[0], q2[0]], [p[1], q2[1]], color=COL[j] if j >= 0 else "#999",
             lw=1.0, alpha=0.9)
ax1.set_xlim(-0.75, 0.95)
ax1.set_ylim(-1.0, 1.5)
ax1.set_xlabel("x", fontsize=8); ax1.set_ylabel("depth  z", fontsize=8)
ax1.set_aspect("equal", adjustable="box")
ax1.tick_params(length=2, labelsize=7)
for sp in ("top", "right"):
    ax1.spines[sp].set_visible(False)

# ---- (2) observed EPI ---------------------------------------------------------
ax2.set_title("2   observed EPI", loc="left", fontsize=10)
ax2.imshow(epi, origin="lower", aspect="auto",
           extent=[q_vals[0], q_vals[-1], s_vals[0], s_vals[-1]],
           interpolation="nearest")
ax2.plot(q_patch, s_vals, color="white", lw=0.8, ls=(0, (3, 2)))
ax2.text(q_patch[0] + 0.02, s_vals[0] + 0.03, "patch line\n(true depth)",
         fontsize=7, color="#222", ha="left", va="bottom")
# imaginary EPI lines = virtual images of the sources (mirror across z = Zp).
z_v = 2 * PATCH_C[1] - EMITTERS[0][1]
for ex, ez, ehw, col in EMITTERS:
    u_imag = (ex - s_vals) / z_v
    ax2.plot(u_imag, s_vals, color=col, lw=1.4, ls=(0, (2, 1.6)), alpha=0.9)
ax2.text(-0.82, 0.5, "imaginary lines of the\nreflected sources (behind)",
         fontsize=7, color="#555", ha="left", va="top")
ax2.set_xlim(q_vals[0], q_vals[-1]); ax2.set_ylim(s_vals[0], s_vals[-1])
ax2.set_xlabel("sensor coordinate  $u = (X-s)/Z$", fontsize=8)
ax2.set_ylabel("view  $s$", fontsize=8)
ax2.tick_params(length=2, labelsize=7)

# ---- (3) per-channel radiance along the patch line, with smooth curves ---------
ax3.set_title("3   radiance along the patch line", loc="left", fontsize=10)
chans = [("R", 0, "#c0392b"), ("G", 1, "#2e8b57"), ("B", 2, "#3b6fb0")]
for name, ci, col in chans:
    y = patch_rad[:, ci]
    ax3.scatter(s_vals, y, s=8, color=col, alpha=0.30, edgecolor="none", zorder=2)
    ax3.plot(s_vals, smooth(y, 2.5), color=col, lw=2.0, zorder=3,
             label=f"{name} reflected")
    ax3.axhline(ALPHA * D_DIFFUSE[ci], color=col, ls=":", lw=1.0, alpha=0.8, zorder=1)
ax3.text(0.0, patch_rad.max() + 0.04,
         "bumps = view-dependent reflections\n"
         "dotted = diffuse baseline $\\alpha D$  (what robust aggregation returns)",
         fontsize=7.5, color="#555", ha="center", va="bottom")
ax3.set_xlim(s_vals[0], s_vals[-1])
ax3.set_ylim(0, patch_rad.max() * 1.35)
ax3.set_xlabel("view  $s$", fontsize=8)
ax3.set_ylabel("channel radiance", fontsize=8)
ax3.tick_params(length=2, labelsize=7)
for sp in ("top", "right"):
    ax3.spines[sp].set_visible(False)
ax3.legend(fontsize=7, loc="upper right", frameon=False, handlelength=1.6,
           labelspacing=0.25, borderpad=0.1)

fig.tight_layout(w_pad=1.3)
out = "toy_epi_aggregation.pdf"
fig.savefig(out, bbox_inches="tight")
fig.savefig("toy_epi_aggregation.png", dpi=150, bbox_inches="tight")
print("wrote", out)
