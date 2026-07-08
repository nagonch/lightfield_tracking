"""
toy_reflection_separation_flatland.py
=====================================
A flatland toy that explains EXACTLY what ``reflection_separation.py`` optimises,
in EXACTLY the state it optimises in.

The patch is a slightly CURVED glossy strip, so its surface normal -- and
therefore the bit of environment it reflects -- varies across its extent.  It
also carries a spatially-varying DIFFUSE albedo D(t).  The dichromatic image of
every surface point ``p`` across every view ``s`` is

        O(p, s) = alpha * D(p)  +  (1 - alpha) * E( reflect(p, s) )

with D view-INDEPENDENT (constant along a point's EPI line) and E the shared
environment behind the cameras (view-DEPENDENT).  ``alpha`` and the mask are known.

This file is a faithful 1-D mirror of ``DiffuseEnvModel`` / ``separate_reflection``
in ``reflection_separation.py``.  Every variable and every loss term has the same
name and the same role; the only change is that the equirect environment map
(H x W) collapses to a 1-D strip of K bins (so TV is 1-D and there is no v-axis):

    OPTIMISED VARIABLES (nn.Parameter)
      * ``diffuse_logits``  [P, 3]  ->  diffuse = sigmoid(logits)      (D, on the surface)
      * ``environment_map`` [K, 3]  ->  reflected = sample_env(E, dir) (E, behind cameras)

    INITIAL CONDITIONS (never use ground truth)
      * diffuse_0  = per-point mean of the observations                (logit-init)
      * env_0      = splat warm-start of the residual reflective estimate
                     (obs - alpha*diffuse_0)/(1 - alpha)               (holes where unobserved)
      * confidence = 1 - exp(-accum_w / ref)  per env bin              (observed vs interpolated)
      * weight     = valid * |cos(view, normal)|  per observation      (grazing down-weight)

    LOSS  =  loss_recon                                                (confidence/grazing-weighted MSE)
           + w_tv    * loss_env_tv      with  w_tv = floor + (1-floor)*(1-confidence)
           + w_range * loss_env_range   ( relu(-E) + relu(E-1) )
           + w_prior * loss_env_prior   ( ||E - prev_env||^2 over prev_valid; multi-frame )

NO ground truth is used by the solver: it sees only NOISY observations, is
initialised from the data, and minimises the loss above (Adam, the real LRs).

Run:
    python toy_reflection_separation_flatland.py
produces
    toy_reflection_separation_flatland.{png,pdf}   -- initial conditions + loss components
    toy_reflection_separation_flatland.gif         -- the optimisation evolving
"""

import argparse
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import torch
import torch.nn.functional as F

# ──────────────────────────────────────────────────────────────────────────────
#  Solver hyper-parameters -- copied verbatim from separate_reflection() defaults
# ──────────────────────────────────────────────────────────────────────────────
D_LR = 1e-2  # lr            (diffuse logits)
ENV_LR = 5e-2  # env_lr        (environment map)
W_TV = 1e-2  # weight_env_tv
W_RANGE = 1e-1  # weight_env_range
W_PRIOR = 1e-1  # weight_env_prior  (only active with a previous frame)
TV_FLOOR = 0.1  # tv_observed_floor
ITERS = 300  # iterations
SNAP_EVERY = 6  # snapshot cadence for the GIF
EPS = 1e-8

DEVICE = "cpu"
torch.manual_seed(0)


def T(x):
    return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=DEVICE)


# ──────────────────────────────────────────────────────────────────────────────
#  Scene  (curved glossy patch; shared with toy_reflection_motion.py)
# ──────────────────────────────────────────────────────────────────────────────
ALPHA = 0.5
ZP = 1.20  # patch nominal depth
ENV_Z = -0.70  # coloured environment behind cameras
ENV_X0, ENV_X1 = -1.7, 1.7
BG_Z = 2.00
WARM = np.array([1.00, 0.94, 0.82])
ENV_SEED, ENV_SIGMA, ENV_SPAN = 7, 95.0, 2.4

X0 = 0.0  # patch centre
W = 0.18  # patch half-extent (surface param t)
CURV = 1.1  # convex bulge toward cameras
P = 90  # surface points
D_COLORS = np.array(
    [
        [0.13, 0.55, 0.60],  # teal
        [0.46, 0.27, 0.52],  # purple
        [0.82, 0.58, 0.22],  # gold
        [0.30, 0.46, 0.26],  # olive
    ]
)
DIFF_SEED, DIFF_SIGMA, DIFF_SPAN = 29, 130.0, 3.0
K = 110  # environment bins

S, Q = 121, 400
s_vals = np.linspace(-0.55, 0.55, S)
q_vals = np.linspace(-1.00, 1.00, Q)

_xt = np.linspace(-9, 9, 1600)
_rng = np.random.default_rng(0)


def _smooth(y, sig):
    half = int(3 * sig)
    k = np.exp(-0.5 * (np.arange(-half, half + 1) / sig) ** 2)
    k /= k.sum()
    return np.convolve(np.pad(y, half, mode="reflect"), k, mode="valid")


_prof = np.clip(
    0.60
    + 0.16 * np.sin(2 * np.pi * _xt / 0.55)
    + 0.18 * _smooth(_rng.standard_normal(_xt.size), 2.5),
    0.30,
    0.92,
)


def _smooth_noise(n, seed, sigma, octaves=2):
    rng = np.random.default_rng(seed)
    y, amp = np.zeros(n), 1.0
    for o in range(octaves):
        y += amp * _smooth(rng.standard_normal(n), max(2.0, sigma / (2**o)))
        amp *= 0.55
    return (y - y.min()) / (y.max() - y.min() + 1e-9)


_ENV_N, _DIFF_N = 2048, 1024
_ENV_HUE = (_smooth_noise(_ENV_N, ENV_SEED, ENV_SIGMA) * ENV_SPAN) % 1.0
_DIFF_PH = _smooth_noise(_DIFF_N, DIFF_SEED, DIFF_SIGMA)


def bg_tex(X):
    inten = np.interp(np.ravel(X), _xt, _prof).reshape(np.shape(X))
    return inten[..., None] * WARM


def env_color(xh):
    t = np.clip((np.asarray(xh) - ENV_X0) / (ENV_X1 - ENV_X0), 0, 1)
    h = np.interp(np.ravel(t), np.linspace(0, 1, _ENV_N), _ENV_HUE).reshape(np.shape(t))
    hsv = np.stack([h, np.full_like(t, 0.95), np.full_like(t, 0.96)], axis=-1)
    return mcolors.hsv_to_rgb(hsv)


def diffuse_at(t):
    f = np.clip((np.asarray(t) + W) / (2 * W), 0, 1)
    ph = np.interp(np.ravel(f), np.linspace(0, 1, _DIFF_N), _DIFF_PH).reshape(
        np.shape(f)
    )
    n = len(D_COLORS)
    idx = (ph * DIFF_SPAN * n) % n
    i0 = np.floor(idx).astype(int) % n
    frac = (idx - np.floor(idx))[..., None]
    return D_COLORS[i0] * (1 - frac) + D_COLORS[(i0 + 1) % n] * frac


def reflect_xh(xq, z, t, s):
    """World-x where the env wall is hit by the ray camera(s)->point reflected
    about the curved patch normal at parameter t.  All args broadcast."""
    nx, nz = 2 * CURV * t, -np.ones_like(t * 1.0)
    nn = np.hypot(nx, nz)
    nx, nz = nx / nn, nz / nn
    dx, dz = xq - s, z * np.ones_like(xq)
    dn = np.hypot(dx, dz)
    dx, dz = dx / dn, dz / dn
    dot = dx * nx + dz * nz
    rx, rz = dx - 2 * dot * nx, dz - 2 * dot * nz
    tau = (ENV_Z - z) / rz
    return xq + rx * tau


# ── environment binning (range from the patch's reflected sweep) ───────────────
_t_pts = np.linspace(-W, W, P)
_xq = (X0 + _t_pts)[:, None]
_z = ZP + CURV * _t_pts[:, None] ** 2
_xh_ps = reflect_xh(_xq, _z, _t_pts[:, None], s_vals[None, :])  # [P, S]
XH_LO, XH_HI = float(_xh_ps.min()), float(_xh_ps.max())
_pad = 0.06 * (XH_HI - XH_LO)
ENV_X0, ENV_X1 = XH_LO - _pad, XH_HI + _pad
env_centers = np.linspace(XH_LO, XH_HI, K)
E_TRUE = env_color(env_centers)  # [K, 3]  (reference only; never seen by solver)
D_TRUE = diffuse_at(_t_pts)  # [P, 3]  (reference only)

# continuous env-bin coordinate of every (point, view) -- the analog of the
# reflected direction fed to sample_environment_map().
KF = (_xh_ps - XH_LO) / (XH_HI - XH_LO) * (K - 1)  # [P, S] float


def bin_of(xh):
    return np.clip(
        np.round((xh - XH_LO) / (XH_HI - XH_LO) * (K - 1)).astype(int), 0, K - 1
    )


# ── per-observation weight: valid * |cos(view, normal)| (grazing down-weight) ──
_xqP = X0 + _t_pts  # [P]
_zP = ZP + CURV * _t_pts**2  # [P]
_nxP, _nzP = 2 * CURV * _t_pts, -np.ones(P)
_nnP = np.hypot(_nxP, _nzP)
_nxP, _nzP = _nxP / _nnP, _nzP / _nnP
_dx = _xqP[:, None] - s_vals[None, :]
_dz = _zP[:, None] * np.ones((1, S))
_dn = np.hypot(_dx, _dz)
_dx, _dz = _dx / _dn, _dz / _dn
COSW = np.abs(_dx * _nxP[:, None] + _dz * _nzP[:, None])  # [P, S]  (== weight; valid=1)

# ── observations O(point, view) the solver actually fits (noisy) ───────────────
KIDX = bin_of(_xh_ps)  # [P, S]
O_OBS = ALPHA * D_TRUE[:, None, :] + (1 - ALPHA) * E_TRUE[KIDX]  # [P, S, 3] (clean)
NOISE_STD = 0.03
O_FIT = O_OBS + np.random.default_rng(1234).normal(0, NOISE_STD, O_OBS.shape)

# ── dense EPI grids (for display only) ─────────────────────────────────────────
xq_d = s_vals[:, None] + q_vals[None, :] * ZP  # [S, Q]
t_d = xq_d - X0
MASK = np.abs(t_d) <= W
z_d = ZP + CURV * t_d**2
xh_d = reflect_xh(xq_d, z_d, t_d, s_vals[:, None])
KDENSE = bin_of(xh_d)  # [S, Q]
PIDX = np.clip(np.round((t_d + W) / (2 * W) * (P - 1)).astype(int), 0, P - 1)
BG = bg_tex(s_vals[:, None] + q_vals[None, :] * BG_Z)
WHITE = np.ones(3)
_VIEW = np.arange(S)[:, None]  # [S, 1]  view index for each EPI row


def layer(values, fill=WHITE):
    return np.where(MASK[..., None], values, fill)


def to_dense_ps(arr_ps):
    """Scatter a per-(point, view) array [P, S] onto the dense EPI grid [S, Q]."""
    return arr_ps[PIDX, _VIEW]


def epi_observed():
    return np.clip(
        np.where(
            MASK[..., None], ALPHA * diffuse_at(t_d) + (1 - ALPHA) * E_TRUE[KDENSE], BG
        ),
        0,
        1,
    )


def epi_from(D_est, E_est):
    """Dense diffuse / reflection / reconstruction EPIs from current estimates.
    Layers are divided by their mixing weight so D and E show at full brightness."""
    diff = np.clip(layer(D_est[PIDX]), 0, 1)
    refl = np.clip(layer(E_est[KDENSE]), 0, 1)
    recon = np.clip(layer(ALPHA * D_est[PIDX] + (1 - ALPHA) * E_est[KDENSE], BG), 0, 1)
    return diff, refl, recon


# ──────────────────────────────────────────────────────────────────────────────
#  1-D mirrors of sample_environment_map / splat_environment_map
# ──────────────────────────────────────────────────────────────────────────────
def sample_env_1d(E, kf):
    """Differentiable linear sampling of the 1-D env strip E[K,3] at coords kf."""
    k0 = torch.floor(kf).long().clamp(0, K - 1)
    k1 = (k0 + 1).clamp(0, K - 1)
    frac = (kf - k0.to(kf.dtype)).unsqueeze(-1)
    return E[k0] * (1 - frac) + E[k1] * frac


def splat_env_1d(kf, colors, weights):
    """Nearest-bin weighted accumulation -> (env, accum_w). Seeds the env param."""
    k = kf.round().long().clamp(0, K - 1).reshape(-1)
    rgb = torch.zeros(K, 3, device=kf.device)
    wsum = torch.zeros(K, device=kf.device)
    rgb.index_add_(0, k, colors.reshape(-1, 3) * weights.reshape(-1, 1))
    wsum.index_add_(0, k, weights.reshape(-1))
    env = rgb / (wsum[:, None] + EPS)
    return env, wsum


# ──────────────────────────────────────────────────────────────────────────────
#  Solver -- a faithful 1-D copy of separate_reflection()
# ──────────────────────────────────────────────────────────────────────────────
def solve(iters=ITERS, snap_every=SNAP_EVERY):
    """Recover (D, E) from the NOISY observations exactly as separate_reflection
    does: sigmoid-diffuse + env-map Adam fit of weighted-recon + TV + range.

    Returns (snaps, info) where ``snaps`` is a list of
    (iter, D_est[P,3], E_est[K,3], loss_dict) and ``info`` holds every initial
    condition (numpy) for the static figure."""
    colors = T(O_FIT)  # [P, S, 3]
    kf = T(KF)  # [P, S]
    weight = T(COSW)  # [P, S]   (valid * |cos|; valid == 1 here)
    w_ext = weight[..., None]
    w_norm = w_ext.sum().clamp(min=1.0)

    # ---- diffuse init: per-point valid mean -> logit ----
    diffuse0 = colors.mean(1).clamp(1e-4, 1.0 - 1e-4)  # [P, 3]
    diffuse_logits = torch.log(diffuse0 / (1.0 - diffuse0)).clone().requires_grad_(True)

    # ---- env init: splat warm-start of the residual reflective estimate ----
    reflective_est = ((colors - ALPHA * diffuse0[:, None, :]) / (1.0 - ALPHA)).clamp(
        0.0, 1.0
    )
    env_splat, accum_w = splat_env_1d(kf, reflective_est, weight)
    environment_map = env_splat.clone().requires_grad_(True)

    # ---- observation confidence: 1 - exp(-accum_w / ref), ref = 0.3*median ----
    pos = accum_w > 0
    if bool(pos.any()):
        ref = (0.3 * torch.median(accum_w[pos])).clamp(min=EPS)
    else:
        ref = accum_w.new_tensor(1.0)
    env_confidence = 1.0 - torch.exp(-accum_w / ref)  # [K] in [0, 1]

    # confidence-weighted TV weight (fixed throughout the optimisation)
    w_tv = TV_FLOOR + (1.0 - TV_FLOOR) * (1.0 - env_confidence)  # [K]

    optimizer = torch.optim.Adam(
        [
            {"params": [diffuse_logits], "lr": D_LR},
            {"params": [environment_map], "lr": ENV_LR},
        ]
    )

    snaps = []
    for it in range(iters + 1):
        diffuse = torch.sigmoid(diffuse_logits)
        reflected = sample_env_1d(environment_map, kf)
        reconstruction = ALPHA * diffuse[:, None, :] + (1.0 - ALPHA) * reflected

        loss_recon = (((reconstruction - colors) ** 2) * w_ext).sum() / w_norm

        env = environment_map
        du = (env - torch.roll(env, shifts=1, dims=0)).abs().mean(-1)  # [K], 1-D TV
        loss_env_tv = (du * w_tv).mean()
        loss_env_range = (F.relu(-env) + F.relu(env - 1.0)).mean()

        loss = loss_recon + W_TV * loss_env_tv + W_RANGE * loss_env_range

        if it % snap_every == 0 or it == iters:
            snaps.append(
                (
                    it,
                    diffuse.detach().numpy().copy(),
                    env.detach().clamp(0.0, 1.0).numpy().copy(),
                    {
                        "recon": float(loss_recon.detach()),
                        "tv": float((W_TV * loss_env_tv).detach()),
                        "range": float((W_RANGE * loss_env_range).detach()),
                        "total": float(loss.detach()),
                    },
                )
            )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # ---- initial-condition arrays for the static figure ----
    with torch.no_grad():
        recon0 = ALPHA * diffuse0[:, None, :] + (1.0 - ALPHA) * sample_env_1d(
            env_splat, kf
        )
        res0 = (((recon0 - colors) ** 2) * w_ext).sum(-1)  # [P, S] weighted residual
        du0 = (env_splat - torch.roll(env_splat, 1, 0)).abs().mean(-1)  # [K]

    info = {
        "D0": diffuse0.numpy(),
        "env_init": env_splat.detach().numpy(),
        "conf": env_confidence.numpy(),
        "w_tv": w_tv.numpy(),
        "cosw": COSW,
        "accum_w": accum_w.numpy(),
        "res0": res0.numpy(),
        "dE0": du0.numpy(),
    }
    return snaps, info


# ──────────────────────────────────────────────────────────────────────────────
#  Shared plotting helpers
# ──────────────────────────────────────────────────────────────────────────────
_EXT = [q_vals[0], q_vals[-1], s_vals[0], s_vals[-1]]
_BIN_EXT = [0, 1, 0, 1]


def _strip(ax, colors, title, xlabel=None):
    ax.imshow(np.clip(colors[None], 0, 1), aspect="auto", extent=[0, 1, 0, 1])
    ax.set_title(title, fontsize=8.5, loc="left")
    ax.set_yticks([])
    ax.set_xticks([])
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=7.5)


def _epi(ax, img, title, cmap=None, vmin=None, vmax=None):
    im = ax.imshow(
        img,
        origin="lower",
        aspect="auto",
        extent=_EXT,
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title, fontsize=8.5, loc="left")
    ax.set_xlabel("sensor coordinate $u$", fontsize=7.5)
    ax.set_ylabel("view $s$", fontsize=7.5)
    ax.tick_params(length=2, labelsize=6.5)
    return im


def _draw_scene(axs):
    axs.set_title("scene: curved glossy patch", fontsize=9, loc="left")
    axs.axhline(0, color="#dddddd", lw=0.8, zorder=0)
    grad = env_color(np.linspace(ENV_X0, ENV_X1, 256))[None]
    axs.imshow(
        grad, extent=[-1.0, 1.0, ENV_Z - 0.06, ENV_Z + 0.06], aspect="auto", zorder=2
    )
    axs.text(
        0.0, ENV_Z - 0.14, "environment $E$ (reflected)", fontsize=6.5,
        color="#555", ha="center", va="top",
    )
    bgrad = bg_tex(np.linspace(-1.0, 1.0, 256))[None]
    axs.imshow(
        bgrad, extent=[-1.0, 1.0, BG_Z - 0.07, BG_Z + 0.07], aspect="auto", zorder=2
    )
    axs.text(
        0.0, BG_Z + 0.16, "textured background", fontsize=6.5,
        color="#777", ha="center", va="bottom",
    )
    for s in s_vals[::16]:
        axs.add_patch(
            plt.Polygon(
                [[s - 0.025, -0.08], [s + 0.025, -0.08], [s, 0.0]],
                closed=True, facecolor="white", edgecolor="#222", lw=0.7,
            )
        )
    axs.text(-1.05, 0.18, "cameras", fontsize=7, color="#9aa0a6", ha="left")
    xa, za = X0 + _t_pts, ZP + CURV * _t_pts**2
    axs.scatter(xa, za, c=np.clip(D_TRUE, 0, 1), s=10, zorder=4)
    for ti in (-0.15, 0.0, 0.15):
        xqi, zi = X0 + ti, ZP + CURV * ti**2
        nx, nz = 2 * CURV * ti, -1.0
        nn = np.hypot(nx, nz)
        axs.plot(
            [xqi, xqi + 0.12 * nx / nn], [zi, zi + 0.12 * nz / nn], color="#888", lw=0.8
        )
        xh = reflect_xh(np.array(xqi), np.array(zi), np.array(ti), np.array(0.0))
        axs.plot([xqi, float(xh)], [zi, ENV_Z], color="#bbb", lw=0.6, zorder=1)
    axs.text(X0, ZP + 0.22, "patch", fontsize=7.5, color="#333", ha="center")
    axs.set_xlim(-1.15, 1.15)
    axs.set_ylim(-0.95, 2.35)
    axs.set_aspect("equal", adjustable="box")
    axs.set_xlabel("x", fontsize=7.5)
    axs.set_ylabel("depth z", fontsize=7.5)
    axs.tick_params(length=2, labelsize=6.5)
    for sp in ("top", "right"):
        axs.spines[sp].set_visible(False)


# ──────────────────────────────────────────────────────────────────────────────
#  Static figure: INITIAL CONDITIONS + COMPONENTS OF THE LOSS FUNCTION
# ──────────────────────────────────────────────────────────────────────────────
def _main_figure(info, snaps):
    plt.rcParams.update(
        {
            "font.size": 9,
            "font.family": "DejaVu Sans",
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
        }
    )
    D0, env_init = info["D0"], info["env_init"]
    conf, w_tv = info["conf"], info["w_tv"]
    res0_dense = to_dense_ps(info["res0"])
    cosw_dense = to_dense_ps(info["cosw"])
    bins = np.arange(K)

    fig = plt.figure(figsize=(12.6, 9.4))
    gs = fig.add_gridspec(
        3, 3, height_ratios=[1.0, 1.0, 1.0], hspace=0.46, wspace=0.30
    )

    # ── Row 0 : the optimised variables in their INITIAL state ────────────────
    _draw_scene(fig.add_subplot(gs[0, 0]))
    _strip(
        fig.add_subplot(gs[0, 1]),
        D0,
        r"init diffuse  $\hat D_0=\mathrm{sigmoid}(\mathrm{logits}_0)$  (per-point mean)",
        "patch surface  $p$ →",
    )
    _strip(
        fig.add_subplot(gs[0, 2]),
        env_init,
        r"init env  $\hat E_0$  (splat warm-start of residual reflective est.)",
        "env bin  $k$  (reflected dir) →",
    )

    # ── Row 1 : confidence, per-observation weight, observed input ────────────
    axc = fig.add_subplot(gs[1, 0])
    axc.imshow(
        np.repeat(conf[None, :, None], 3, axis=2),
        aspect="auto", extent=[0, 1, 0, 1], vmin=0, vmax=1,
    )
    axc.set_title(
        r"env confidence  $c=1-e^{-\mathrm{accum\_w}/\mathrm{ref}}$  (observed→white)",
        fontsize=8.5, loc="left",
    )
    axc.set_xticks([])
    axc.set_yticks([])
    axc.set_xlabel("env bin  $k$ →", fontsize=7.5)

    axw = fig.add_subplot(gs[1, 1])
    im = _epi(
        axw, np.where(MASK, cosw_dense, np.nan),
        r"per-obs weight  $|\cos(\mathrm{view},n)|$  (grazing→0)",
        cmap="magma", vmin=0, vmax=1,
    )
    fig.colorbar(im, ax=axw, fraction=0.046, pad=0.02)

    _epi(
        fig.add_subplot(gs[1, 2]),
        np.clip(epi_observed() + _noise_layer(7), 0, 1),
        r"observed input  $O=\alpha D+(1-\alpha)E$  (noisy)",
    )

    # ── Row 2 : the three components of the loss function ─────────────────────
    axr = fig.add_subplot(gs[2, 0])
    im = _epi(
        axr, np.where(MASK, res0_dense, np.nan),
        r"$\mathcal{L}_{\mathrm{recon}}$: weighted residual $w\,(\hat O_0-O)^2$",
        cmap="inferno",
    )
    fig.colorbar(im, ax=axr, fraction=0.046, pad=0.02)

    axt = fig.add_subplot(gs[2, 1])
    axt.plot(bins, w_tv, color="#c0392b", lw=1.6, label=r"$w_{tv}=f+(1-f)(1-c)$")
    axt.plot(
        bins, info["dE0"] / (info["dE0"].max() + 1e-9), color="#2c7fb8", lw=1.0,
        label=r"$|\partial_k \hat E_0|$ (norm.)",
    )
    axt.axhline(TV_FLOOR, color="#999", ls=":", lw=0.8)
    axt.text(2, TV_FLOOR + 0.02, "floor", fontsize=6.5, color="#777")
    axt.set_title(
        rf"$\lambda_{{tv}}\,\mathcal{{L}}_{{tv}}$: confidence-weighted TV "
        rf"($\lambda_{{tv}}={W_TV:g}$)",
        fontsize=8.5, loc="left",
    )
    axt.set_xlabel("env bin  $k$", fontsize=7.5)
    axt.set_ylim(-0.03, 1.05)
    axt.legend(fontsize=6.3, loc="upper right", framealpha=0.9)
    axt.tick_params(length=2, labelsize=6.5)
    for sp in ("top", "right"):
        axt.spines[sp].set_visible(False)

    axg = fig.add_subplot(gs[2, 2])
    lum = (env_init * np.array([0.299, 0.587, 0.114])).sum(-1)
    axg.fill_between([0, K - 1], 0, 1, color="#eef5ee", zorder=0)
    axg.plot(bins, lum, color="#444", lw=1.2)
    axg.axhline(0.0, color="#27ae60", ls="--", lw=0.9)
    axg.axhline(1.0, color="#27ae60", ls="--", lw=0.9)
    axg.text(K * 0.5, 1.04, "valid range [0,1]", fontsize=6.5, color="#27ae60", ha="center")
    axg.set_title(
        rf"$\lambda_{{rng}}\,\mathcal{{L}}_{{rng}}=\lambda_{{rng}}\,$mean"
        rf"$(\mathrm{{relu}}(-E)+\mathrm{{relu}}(E-1))$ ($\lambda_{{rng}}={W_RANGE:g}$)",
        fontsize=8.0, loc="left",
    )
    axg.set_xlabel("env bin  $k$  (env luminance shown)", fontsize=7.5)
    axg.set_ylim(-0.25, 1.25)
    axg.tick_params(length=2, labelsize=6.5)
    for sp in ("top", "right"):
        axg.spines[sp].set_visible(False)
    axg.text(
        0.02, -0.42,
        r"+ multi-frame prior  $\lambda_{pr}\,\|E-E_{\mathrm{prev}}\|^2$ on observed bins"
        rf"  ($\lambda_{{pr}}={W_PRIOR:g}$; inactive on frame 0)",
        transform=axg.transAxes, fontsize=6.6, color="#777",
    )

    fig.suptitle(
        "Reflection separation — initial conditions & loss components   ·   "
        r"$\hat D=\mathrm{sigmoid}(\mathrm{logits})$, $\hat E=$ env map   ·   "
        rf"$\alpha={ALPHA}$, known mask",
        fontsize=11, y=0.995,
    )
    fig.savefig("toy_reflection_separation_flatland.pdf", bbox_inches="tight")
    fig.savefig("toy_reflection_separation_flatland.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote toy_reflection_separation_flatland.{pdf,png}")


def _noise_layer(seed):
    nz = np.zeros((S, Q, 3))
    nz[MASK] = np.random.default_rng(seed).normal(0, NOISE_STD, (int(MASK.sum()), 3))
    return nz


# ──────────────────────────────────────────────────────────────────────────────
#  GIF: the optimisation converging (the evolution of all of it)
# ──────────────────────────────────────────────────────────────────────────────
def _gif_frames(outdir, info, snaps):
    os.makedirs(outdir, exist_ok=True)
    plt.rcParams.update(
        {"font.size": 9, "font.family": "DejaVu Sans", "axes.linewidth": 0.8}
    )
    its = [s[0] for s in snaps]
    L_recon = [s[3]["recon"] for s in snaps]
    L_tv = [s[3]["tv"] for s in snaps]
    L_rng = [s[3]["range"] for s in snaps]
    L_tot = [s[3]["total"] for s in snaps]
    conf = info["conf"]
    O_disp = np.clip(epi_observed() + _noise_layer(99), 0, 1)

    paths = []
    for fi, (it, D_est, E_est, ld) in enumerate(snaps):
        diff_e, refl_e, _ = epi_from(D_est, E_est)
        fig = plt.figure(figsize=(11.8, 6.4))
        gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 0.85], hspace=0.5, wspace=0.28)

        for col, (img, ttl) in enumerate(
            [
                (O_disp, r"observed  $O$  (noisy input)"),
                (diff_e, r"estimated diffuse  $\hat D=\mathrm{sigmoid}(\mathrm{logits})$"),
                (refl_e, r"estimated reflection  $\hat E$  (env map)"),
            ]
        ):
            ax = fig.add_subplot(gs[0, col])
            ax.imshow(
                img, origin="lower", aspect="auto", extent=_EXT, interpolation="nearest"
            )
            ax.set_title(ttl, fontsize=9, loc="left")
            ax.set_xlabel("$u$", fontsize=7.5)
            ax.tick_params(length=2, labelsize=6.5)
            if col == 0:
                ax.set_ylabel("view $s$", fontsize=7.5)

        # diffuse: est vs true
        axd = fig.add_subplot(gs[1, 0])
        axd.imshow(np.clip(D_TRUE[None], 0, 1), aspect="auto", extent=[0, 1, 1, 2])
        axd.imshow(np.clip(D_est[None], 0, 1), aspect="auto", extent=[0, 1, 0, 1])
        axd.set_ylim(0, 2)
        axd.set_xticks([])
        axd.set_yticks([0.5, 1.5])
        axd.set_yticklabels(["est", "true"], fontsize=7)
        axd.set_title(r"diffuse  $\hat D(p)$  (the surface colour)", fontsize=8.5, loc="left")

        # env: est vs true + confidence
        axe = fig.add_subplot(gs[1, 1])
        axe.imshow(np.clip(E_TRUE[None], 0, 1), aspect="auto", extent=[0, 1, 2, 3])
        axe.imshow(np.clip(E_est[None], 0, 1), aspect="auto", extent=[0, 1, 1, 2])
        axe.imshow(
            np.repeat(conf[None, :, None], 3, axis=2),
            aspect="auto", extent=[0, 1, 0, 1], vmin=0, vmax=1,
        )
        axe.set_ylim(0, 3)
        axe.set_xticks([])
        axe.set_yticks([0.5, 1.5, 2.5])
        axe.set_yticklabels(["conf", "est", "true"], fontsize=7)
        axe.set_title(r"environment map  $\hat E$  + confidence", fontsize=8.5, loc="left")

        # loss components
        axl = fig.add_subplot(gs[1, 2])
        axl.semilogy(its, L_tot, color="#222", lw=1.4, label="total")
        axl.semilogy(its, L_recon, color="#c0392b", lw=1.0, label=r"$\mathcal{L}_{recon}$")
        axl.semilogy(its, L_tv, color="#2c7fb8", lw=1.0, label=r"$\lambda_{tv}\mathcal{L}_{tv}$")
        axl.semilogy(its, L_rng, color="#27ae60", lw=1.0, label=r"$\lambda_{rng}\mathcal{L}_{rng}$")
        axl.axvline(it, color="#999", lw=0.8, ls=":")
        axl.set_xlim(0, its[-1])
        axl.set_xlabel("iteration", fontsize=7.5)
        axl.set_title("loss components", fontsize=8.5, loc="left")
        axl.legend(fontsize=6.0, loc="lower left", ncol=2, framealpha=0.9)
        axl.tick_params(length=2, labelsize=6.5)
        for sp in ("top", "right"):
            axl.spines[sp].set_visible(False)

        fig.suptitle(
            rf"Reflection separation (Adam, noisy obs, no GT) — iter {it:3d}   "
            rf"loss {ld['total']:.4f}   ·   variables: $\mathrm{{logits}}$ (→$\hat D$), env map $\hat E$",
            fontsize=10.5, y=0.99,
        )
        p = os.path.join(outdir, f"frame_{fi:03d}.png")
        fig.savefig(p, dpi=96)
        plt.close(fig)
        paths.append(p)
    print(f"wrote {len(paths)} frames to {outdir}")
    return paths


def _assemble_gif(paths, gifpath, duration=110, hold=12):
    from PIL import Image

    frames = [Image.open(p).convert("P", palette=Image.ADAPTIVE) for p in paths]
    frames += [frames[-1]] * hold  # pause on the converged result
    frames[0].save(
        gifpath, save_all=True, append_images=frames[1:], duration=duration, loop=0
    )
    print(f"wrote {gifpath}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="frames_sep", help="dir to dump GIF frames into")
    ap.add_argument("--gif", default="toy_reflection_separation_flatland.gif")
    args = ap.parse_args()

    snaps, info = solve()
    _main_figure(info, snaps)
    paths = _gif_frames(args.frames, info, snaps)
    _assemble_gif(paths, args.gif)
