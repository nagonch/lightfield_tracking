"""
toy_reflection_separation_flatland.py
=====================================
Reflection separation on the (single) glossy patch, in the same flatland as
``toy_reflection_motion.py``.

The patch is now slightly CURVED, so its surface normal — and therefore the bit
of environment it reflects — varies across its extent.  It also carries a
spatially-varying DIFFUSE albedo D(t) (teal -> purple) for generality.  The
dichromatic image of every surface point, across every view, is

        O(t, s) = alpha * D(t)  +  (1 - alpha) * E( reflect(t, s) )

with D view-INDEPENDENT (constant along a point's EPI line) and E the shared
environment map behind the cameras (view-DEPENDENT).  alpha and the patch mask
are assumed known.

Reflection separation = recover, from O alone, the per-point diffuse D(t) and the
shared environment E(.).  It is a (gauge-fixed) linear inverse problem: the same
environment bin is hit by many (point, view) pairs, so D and E can be untangled.

The static figure shows every ground-truth component.  ``--frames`` dumps the
separation optimisation evolving (an all-diffuse guess -> the true layers) for a GIF.
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────────────
#  Scene (shared with toy_reflection_motion.py)
# ──────────────────────────────────────────────────────────────────────────────
ALPHA = 0.5
ZP = 1.20                                    # patch nominal depth
ENV_Z = -0.70                                # coloured environment behind cameras
ENV_X0, ENV_X1 = -1.7, 1.7
BG_Z = 2.00
WARM = np.array([1.00, 0.94, 0.82])
TURBO = matplotlib.colormaps["turbo"]

# curved patch with spatially-varying diffuse
X0 = 0.0                                      # patch centre
W = 0.18                                      # patch half-extent (surface param t)
CURV = 1.1                                    # convex bulge toward cameras
P = 90                                        # surface points
DA = np.array([0.13, 0.55, 0.60])             # diffuse at left edge  (teal)
DB = np.array([0.46, 0.27, 0.52])             # diffuse at right edge (purple)
K = 110                                       # environment bins

S, Q = 121, 400
s_vals = np.linspace(-0.55, 0.55, S)
q_vals = np.linspace(-1.00, 1.00, Q)

_xt = np.linspace(-9, 9, 1600)
_rng = np.random.default_rng(0)


def _smooth(y, sig):
    half = int(3 * sig)
    k = np.exp(-0.5 * (np.arange(-half, half + 1) / sig) ** 2); k /= k.sum()
    return np.convolve(np.pad(y, half, mode="reflect"), k, mode="valid")


_prof = np.clip(0.60 + 0.16 * np.sin(2 * np.pi * _xt / 0.55)
                + 0.18 * _smooth(_rng.standard_normal(_xt.size), 2.5), 0.30, 0.92)


def bg_tex(X):
    inten = np.interp(np.ravel(X), _xt, _prof).reshape(np.shape(X))
    return inten[..., None] * WARM


def env_color(xh):
    t = np.clip((xh - ENV_X0) / (ENV_X1 - ENV_X0), 0, 1)
    return np.asarray(TURBO(t))[..., :3]


def diffuse_at(t):
    """Spatially-varying diffuse albedo across the patch (array-friendly)."""
    f = np.clip((np.asarray(t) + W) / (2 * W), 0, 1)[..., None]
    return DA * (1 - f) + DB * f


def reflect_xh(xq, z, t, s):
    """World-x where the env wall is hit by the ray camera(s)->point reflected
    about the curved patch normal at parameter t.  All args broadcast."""
    nx, nz = 2 * CURV * t, -np.ones_like(t * 1.0)
    nn = np.hypot(nx, nz); nx, nz = nx / nn, nz / nn
    dx, dz = xq - s, z * np.ones_like(xq)
    dn = np.hypot(dx, dz); dx, dz = dx / dn, dz / dn
    dot = dx * nx + dz * nz
    rx, rz = dx - 2 * dot * nx, dz - 2 * dot * nz
    tau = (ENV_Z - z) / rz
    return xq + rx * tau


# ── environment binning (range from the patch's reflected sweep) ───────────────
_t_pts = np.linspace(-W, W, P)
_xq = (X0 + _t_pts)[:, None]
_z = ZP + CURV * _t_pts[:, None] ** 2
_xh_ps = reflect_xh(_xq, _z, _t_pts[:, None], s_vals[None, :])    # [P, S]
XH_LO, XH_HI = float(_xh_ps.min()), float(_xh_ps.max())
# map the colour ramp to the actually-reflected range (full rainbow, no clipping)
_pad = 0.06 * (XH_HI - XH_LO)
ENV_X0, ENV_X1 = XH_LO - _pad, XH_HI + _pad
env_centers = np.linspace(XH_LO, XH_HI, K)
E_TRUE = env_color(env_centers)                                   # [K, 3]
D_TRUE = diffuse_at(_t_pts)                                       # [P, 3]


def bin_of(xh):
    return np.clip(np.round((xh - XH_LO) / (XH_HI - XH_LO) * (K - 1)).astype(int), 0, K - 1)


# ── observations O(point, view) and their env-bin index ────────────────────────
KIDX = bin_of(_xh_ps)                                             # [P, S]
O_OBS = ALPHA * D_TRUE[:, None, :] + (1 - ALPHA) * E_TRUE[KIDX]   # [P, S, 3]

# ── dense EPI grids (for display): which surface point / env bin each pixel sees ─
xq_d = s_vals[:, None] + q_vals[None, :] * ZP                     # [S, Q] x of surface point
t_d = xq_d - X0
MASK = np.abs(t_d) <= W
z_d = ZP + CURV * t_d ** 2
xh_d = reflect_xh(xq_d, z_d, t_d, s_vals[:, None])
KDENSE = bin_of(xh_d)                                             # [S, Q]
PIDX = np.clip(np.round((t_d + W) / (2 * W) * (P - 1)).astype(int), 0, P - 1)
BG = bg_tex(s_vals[:, None] + q_vals[None, :] * BG_Z)
WHITE = np.ones(3)


def layer(values, fill=WHITE):
    """Place per-pixel ``values`` [S,Q,3] inside the mask, ``fill`` outside."""
    return np.where(MASK[..., None], values, fill)


def epi_observed():
    return np.clip(np.where(MASK[..., None],
                            ALPHA * diffuse_at(t_d) + (1 - ALPHA) * E_TRUE[KDENSE], BG), 0, 1)


def epi_from(D_est, E_est):
    """Dense diffuse / reflection / reconstruction EPIs from current estimates."""
    diff = np.clip(layer(ALPHA * D_est[PIDX]), 0, 1)
    refl = np.clip(layer((1 - ALPHA) * E_est[KDENSE]), 0, 1)
    recon = np.clip(layer(ALPHA * D_est[PIDX] + (1 - ALPHA) * E_est[KDENSE], BG), 0, 1)
    return diff, refl, recon


# ──────────────────────────────────────────────────────────────────────────────
#  Separation optimisation  (gauge-fixed gradient descent; animate the evolution)
# ──────────────────────────────────────────────────────────────────────────────
def separate(n_iter=600, lr=12.0, snap_every=8):
    N = P * S
    D_est = O_OBS.mean(1) / ALPHA                 # all-diffuse guess (E = 0)
    E_est = np.zeros((K, 3))
    kflat = KIDX.ravel()
    snaps = []
    for it in range(n_iter + 1):
        recon = ALPHA * D_est[:, None, :] + (1 - ALPHA) * E_est[KIDX]
        res = recon - O_OBS
        loss = float((res ** 2).mean())
        if it % snap_every == 0 or it == n_iter:
            snaps.append((it, D_est.copy(), E_est.copy(), loss))
        gD = (2 * ALPHA / N) * res.sum(1)
        gE = np.zeros((K, 3))
        np.add.at(gE, kflat, (2 * (1 - ALPHA) / N) * res.reshape(-1, 3))
        D_est -= lr * gD
        E_est -= lr * gE
        # fix the additive gauge (anchor estimated env mean to the true env mean)
        c = E_TRUE.mean(0) - E_est.mean(0)
        E_est += c
        D_est -= (1 - ALPHA) / ALPHA * c
    return snaps


# ──────────────────────────────────────────────────────────────────────────────
#  Static figure: every component of the ground-truth decomposition
# ──────────────────────────────────────────────────────────────────────────────
def _strip(ax, colors, title, xlabel=None):
    ax.imshow(np.clip(colors[None], 0, 1), aspect="auto", extent=[0, 1, 0, 1])
    ax.set_title(title, fontsize=8.5, loc="left")
    ax.set_yticks([])
    ax.set_xticks([])
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=7.5)


def _epi(ax, img, title):
    ax.imshow(img, origin="lower", aspect="auto",
              extent=[q_vals[0], q_vals[-1], s_vals[0], s_vals[-1]], interpolation="nearest")
    ax.set_title(title, fontsize=8.5, loc="left")
    ax.set_xlabel("sensor coordinate $u$", fontsize=7.5)
    ax.set_ylabel("view $s$", fontsize=7.5)
    ax.tick_params(length=2, labelsize=6.5)


def _main_figure():
    plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans",
                         "axes.linewidth": 0.8, "pdf.fonttype": 42})
    fig = plt.figure(figsize=(12.0, 6.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.15], hspace=0.42, wspace=0.28)

    # (a) scene
    axs = fig.add_subplot(gs[0, 0])
    axs.set_title("scene: curved glossy patch", fontsize=9, loc="left")
    axs.axhline(0, color="#dddddd", lw=0.8, zorder=0)
    grad = env_color(np.linspace(ENV_X0, ENV_X1, 256))[None]
    axs.imshow(grad, extent=[-1.0, 1.0, ENV_Z - 0.06, ENV_Z + 0.06], aspect="auto", zorder=2)
    axs.text(0.0, ENV_Z - 0.14, "environment $E$", fontsize=6.5, color="#555",
             ha="center", va="top")
    for s in s_vals[::16]:
        axs.add_patch(plt.Polygon([[s - 0.025, -0.08], [s + 0.025, -0.08], [s, 0.0]],
                                  closed=True, facecolor="white", edgecolor="#222", lw=0.7))
    axs.text(-1.05, 0.18, "cameras", fontsize=7, color="#9aa0a6", ha="left")
    xa, za = X0 + _t_pts, ZP + CURV * _t_pts ** 2
    axs.scatter(xa, za, c=np.clip(D_TRUE, 0, 1), s=10, zorder=4)
    # a few normals + reflected rays
    for ti in (-0.15, 0.0, 0.15):
        xqi, zi = X0 + ti, ZP + CURV * ti ** 2
        nx, nz = 2 * CURV * ti, -1.0
        nn = np.hypot(nx, nz)
        axs.plot([xqi, xqi + 0.12 * nx / nn], [zi, zi + 0.12 * nz / nn], color="#888", lw=0.8)
        xh = reflect_xh(np.array(xqi), np.array(zi), np.array(ti), np.array(0.0))
        axs.plot([xqi, float(xh)], [zi, ENV_Z], color="#bbb", lw=0.6, zorder=1)
    axs.text(X0, ZP + 0.22, "patch", fontsize=7.5, color="#333", ha="center")
    axs.set_xlim(-1.15, 1.15); axs.set_ylim(-0.9, 1.6)
    axs.set_aspect("equal", adjustable="box")
    axs.set_xlabel("x", fontsize=7.5); axs.set_ylabel("depth z", fontsize=7.5)
    axs.tick_params(length=2, labelsize=6.5)
    for sp in ("top", "right"):
        axs.spines[sp].set_visible(False)

    _strip(fig.add_subplot(gs[0, 1]), E_TRUE, "environment map  $E$", "reflected direction →")
    _strip(fig.add_subplot(gs[0, 2]), D_TRUE, "diffuse albedo  $D(t)$  (varies on patch)",
           "patch surface  $t$ →")

    diff_t, refl_t, _ = epi_from(D_TRUE, E_TRUE)
    _epi(fig.add_subplot(gs[1, 0]), epi_observed(), r"observed EPI  $O=\alpha D+(1-\alpha)E$")
    _epi(fig.add_subplot(gs[1, 1]), diff_t, r"diffuse layer  $\alpha D$  (constant along lines)")
    _epi(fig.add_subplot(gs[1, 2]), refl_t, r"reflection layer  $(1-\alpha)E$  (view-dependent)")

    fig.suptitle("Reflection separation in flatland   ·   "
                 r"$\alpha=0.5$, known mask   ·   curved patch, spatially-varying diffuse",
                 fontsize=10.5, y=0.99)
    fig.savefig("toy_reflection_separation_flatland.pdf", bbox_inches="tight")
    fig.savefig("toy_reflection_separation_flatland.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote toy_reflection_separation_flatland.pdf")


# ──────────────────────────────────────────────────────────────────────────────
#  GIF frames: the separation optimisation converging
# ──────────────────────────────────────────────────────────────────────────────
def _gif_frames(outdir):
    os.makedirs(outdir, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "axes.linewidth": 0.8})
    snaps = separate()
    losses_it = [s[0] for s in snaps]
    losses_v = [s[3] for s in snaps]
    diff_t, refl_t, _ = epi_from(D_TRUE, E_TRUE)
    O_disp = epi_observed()
    ext = [q_vals[0], q_vals[-1], s_vals[0], s_vals[-1]]
    for fi, (it, D_est, E_est, loss) in enumerate(snaps):
        diff_e, refl_e, _ = epi_from(D_est, E_est)
        fig = plt.figure(figsize=(11.6, 6.0))
        gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 0.85], hspace=0.45, wspace=0.28)
        for col, (img, ttl) in enumerate([
                (O_disp, "observed  $O$  (input)"),
                (diff_e, r"estimated diffuse  $\alpha\hat D$"),
                (refl_e, r"estimated reflection  $(1-\alpha)\hat E$")]):
            ax = fig.add_subplot(gs[0, col])
            ax.imshow(img, origin="lower", aspect="auto", extent=ext, interpolation="nearest")
            ax.set_title(ttl, fontsize=9, loc="left")
            ax.set_xlabel("$u$", fontsize=7.5); ax.tick_params(length=2, labelsize=6.5)
            if col == 0:
                ax.set_ylabel("view $s$", fontsize=7.5)

        axd = fig.add_subplot(gs[1, 0])
        axd.imshow(np.clip(D_TRUE[None], 0, 1), aspect="auto", extent=[0, 1, 1, 2])
        axd.imshow(np.clip(D_est[None], 0, 1), aspect="auto", extent=[0, 1, 0, 1])
        axd.set_ylim(0, 2); axd.set_xticks([]); axd.set_yticks([0.5, 1.5])
        axd.set_yticklabels(["est", "true"], fontsize=7)
        axd.set_title("diffuse albedo  $D(t)$", fontsize=8.5, loc="left")

        axe = fig.add_subplot(gs[1, 1])
        axe.imshow(np.clip(E_TRUE[None], 0, 1), aspect="auto", extent=[0, 1, 1, 2])
        axe.imshow(np.clip(E_est[None], 0, 1), aspect="auto", extent=[0, 1, 0, 1])
        axe.set_ylim(0, 2); axe.set_xticks([]); axe.set_yticks([0.5, 1.5])
        axe.set_yticklabels(["est", "true"], fontsize=7)
        axe.set_title("environment map  $E$", fontsize=8.5, loc="left")

        axl = fig.add_subplot(gs[1, 2])
        axl.semilogy(losses_it, losses_v, color="#888", lw=1.0)
        axl.semilogy(losses_it[:fi + 1], losses_v[:fi + 1], color="#c0392b", lw=2.0)
        axl.scatter([it], [loss], color="#c0392b", s=26, zorder=5)
        axl.set_xlim(0, losses_it[-1]); axl.set_xlabel("iteration", fontsize=7.5)
        axl.set_title("reconstruction loss", fontsize=8.5, loc="left")
        axl.tick_params(length=2, labelsize=6.5)
        for sp in ("top", "right"):
            axl.spines[sp].set_visible(False)

        fig.suptitle(f"Reflection separation — iteration {it:3d}   "
                     f"(loss {loss:.4f})", fontsize=11, y=0.99)
        fig.savefig(os.path.join(outdir, f"frame_{fi:03d}.png"), dpi=96)
        plt.close(fig)
    print(f"wrote {len(snaps)} frames to {outdir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="", help="dir to dump GIF frames into")
    args = ap.parse_args()
    _main_figure()
    if args.frames:
        _gif_frames(args.frames)
