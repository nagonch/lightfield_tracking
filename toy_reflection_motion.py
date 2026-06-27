"""
toy_reflection_motion.py
========================
"Reflection as a signal for motion" — a simulated flatland illustration.

A static light-field camera watches a glossy patch sitting in front of a
TEXTURED background wall.  Behind the cameras is a coloured environment that the
patch reflects (dichromatic model  L = alpha*D + (1-alpha)*E).  The patch is
seen at two positions translated along x:

    patch 1 (start)  at x1  — reflects the environment one way
    patch 2 (target) at x2  — having moved, reflects it a DIFFERENT way

We recover the translation by sliding a *virtual* patch across x and CORRELATING
its EPI with the target EPI (subtracting the two EPIs and summing the residual).
Three ways to predict the virtual patch's colours give three losses:

  (1) RELIGHT  — recompute the reflection at the candidate position → matches the
                 target only at the true displacement → minimum reaches 0.
  (2) FREEZE   — carry patch-1's view-dependent colours rigidly → minimum at the
                 truth but on a photometric floor (colours don't match).
  (3) DIFFUSE  — average (diffuse) colour only, no view-dependence.  Still dips at
                 the truth: away from it each patch is differenced against the
                 BACKGROUND, at the truth the two patches are differenced against
                 each other (patch colour != background colour).

main.py writes the static figure (pdf/png).  `--frames` also dumps GIF frames.
"""

import argparse
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────────────
#  Scene
# ──────────────────────────────────────────────────────────────────────────────
ALPHA = 0.5
D_DIFFUSE = np.array([0.16, 0.55, 0.62])  # patch diffuse colour (dark teal)
ZP = 1.20  # patch depth (both positions)
PATCH_HW = 0.10  # patch half-width
X1, X2 = -0.22, 0.22  # start and target patch x
ENV_Z = -0.70  # coloured environment behind cameras
ENV_X0, ENV_X1 = -1.7, 1.7  # env extent -> colour ramp
BG_Z = 2.00  # textured background wall (behind patch)
WARM = np.array([1.00, 0.94, 0.82])  # background tint (distinct from env)
PATCH_N = np.array([0.0, -1.0])
TURBO = matplotlib.colormaps["turbo"]

S, Q = 121, 400
s_vals = np.linspace(-0.55, 0.55, S)
q_vals = np.linspace(-1.00, 1.00, Q)
SIG_U = PATCH_HW / ZP

# textured 1-D background profile (warm-grey stripes + smooth noise)
_xt = np.linspace(-9, 9, 1600)
_rng = np.random.default_rng(0)


def _smooth(y, sig):
    half = int(3 * sig)
    k = np.exp(-0.5 * (np.arange(-half, half + 1) / sig) ** 2)
    k /= k.sum()
    return np.convolve(np.pad(y, half, mode="reflect"), k, mode="valid")


_prof = (
    0.60
    + 0.16 * np.sin(2 * np.pi * _xt / 0.55)
    + 0.09 * np.sin(2 * np.pi * _xt / 0.19 + 1.3)
    + 0.18 * _smooth(_rng.standard_normal(_xt.size), 2.5)
)
_prof = np.clip(_prof, 0.30, 0.92)


def bg_tex(X):
    """Warm-grey textured wall colour at world-x ``X`` (array-friendly)."""
    inten = np.interp(np.ravel(X), _xt, _prof).reshape(np.shape(X))
    return inten[..., None] * WARM


def env_color(xh):
    t = np.clip((xh - ENV_X0) / (ENV_X1 - ENV_X0), 0, 1)
    return np.asarray(TURBO(t))[..., :3]


def patch_profile(x):
    """Per-view dichromatic appearance L(x, s) of a patch centred at (x, ZP)."""
    xh = x + ((ZP - ENV_Z) / ZP) * (x - s_vals)
    return np.clip(ALPHA * D_DIFFUSE + (1.0 - ALPHA) * env_color(xh), 0, 1)


# Background EPI (a ray from camera s through sensor u hits the wall at s + u*BG_Z)
B_EPI = bg_tex(s_vals[:, None] + q_vals[None, :] * BG_Z)  # [S, Q, 3]


def paint(img, x, colors):
    """Soft-edged patch band at depth ZP, position x, per-view ``colors`` [S,3]."""
    uc = (x - s_vals) / ZP
    a = np.exp(-0.5 * ((q_vals[None, :] - uc[:, None]) / SIG_U) ** 2)  # [S, Q]
    return img * (1 - a)[..., None] + colors[:, None, :] * a[..., None]


L1 = patch_profile(X1)
L2 = patch_profile(X2)
M1 = np.tile(L1.mean(0), (S, 1))  # patch-1 diffuse (mean) colour
M2 = np.tile(L2.mean(0), (S, 1))

T_REAL = paint(B_EPI, X2, L2)  # observed target EPI
T_MEAN = paint(B_EPI, X2, M2)  # diffuse target (mean colour)


def mse(a, b):
    return float(((a - b) ** 2).sum(2).mean())


# ──────────────────────────────────────────────────────────────────────────────
#  Loss curves  (correlate virtual EPI with target EPI)
# ──────────────────────────────────────────────────────────────────────────────
xv = np.linspace(-0.62, 0.62, 161)
loss_relight = np.array([mse(paint(B_EPI, x, patch_profile(x)), T_REAL) for x in xv])
loss_freeze = np.array([mse(paint(B_EPI, x, L1), T_REAL) for x in xv])
loss_diffuse = np.array([mse(paint(B_EPI, x, M1), T_MEAN) for x in xv])


def _main_figure():
    plt.rcParams.update(
        {
            "font.size": 9,
            "font.family": "DejaVu Sans",
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
        }
    )
    fig, (axA, axB, axC) = plt.subplots(
        1, 3, figsize=(12.4, 3.9), gridspec_kw={"width_ratios": [1.05, 0.95, 1.2]}
    )

    # (A) scene
    axA.set_title("1   patch translates along x", loc="left", fontsize=10)
    axA.axhline(0, color="#dddddd", lw=0.8, zorder=0)
    grad = env_color(np.linspace(-1.0, 1.0, 256))[None]
    axA.imshow(
        grad, extent=[-1.0, 1.0, ENV_Z - 0.05, ENV_Z + 0.05], aspect="auto", zorder=2
    )
    axA.text(
        0.0,
        ENV_Z - 0.12,
        "reflected environment (behind cameras)",
        fontsize=7,
        color="#555",
        ha="center",
        va="top",
    )
    bgrad = bg_tex(np.linspace(-1.0, 1.0, 256))[None]
    axA.imshow(
        bgrad, extent=[-1.0, 1.0, BG_Z - 0.06, BG_Z + 0.06], aspect="auto", zorder=2
    )
    axA.text(
        0.0,
        BG_Z + 0.13,
        "textured background",
        fontsize=7,
        color="#777",
        ha="center",
        va="bottom",
    )
    for s in s_vals[::12]:
        axA.add_patch(
            plt.Polygon(
                [[s - 0.022, -0.07], [s + 0.022, -0.07], [s, 0.0]],
                closed=True,
                facecolor="white",
                edgecolor="#222",
                lw=0.7,
            )
        )
    axA.text(-1.05, 0.16, "static cameras", fontsize=8, color="#9aa0a6", ha="left")
    for x in np.linspace(-0.5, 0.5, 9):
        axA.plot(
            [x - PATCH_HW, x + PATCH_HW],
            [ZP, ZP],
            color="#cccccc",
            lw=3,
            solid_capstyle="round",
            zorder=3,
        )
    axA.annotate(
        "",
        xy=(X2, ZP + 0.16),
        xytext=(X1, ZP + 0.16),
        arrowprops=dict(arrowstyle="->", color="#444", lw=1.0),
    )
    axA.text(
        (X1 + X2) / 2, ZP + 0.25, "translate", fontsize=7.5, color="#444", ha="center"
    )
    axA.plot(
        [X1 - PATCH_HW, X1 + PATCH_HW],
        [ZP] * 2,
        color="#2a9d8f",
        lw=6,
        solid_capstyle="round",
        zorder=4,
    )
    axA.plot(
        [X2 - PATCH_HW, X2 + PATCH_HW],
        [ZP] * 2,
        color="#1d3f8f",
        lw=6,
        solid_capstyle="round",
        zorder=4,
    )
    axA.text(
        X1,
        ZP - 0.16,
        "patch 1\n(start)",
        fontsize=7.5,
        color="#2a9d8f",
        ha="center",
        va="top",
    )
    axA.text(
        X2,
        ZP - 0.16,
        "patch 2\n(target)",
        fontsize=7.5,
        color="#1d3f8f",
        ha="center",
        va="top",
    )
    axA.set_xlim(-1.15, 1.15)
    axA.set_ylim(-0.95, 2.25)
    axA.set_xlabel("x", fontsize=8)
    axA.set_ylabel("depth  z", fontsize=8)
    axA.set_aspect("equal", adjustable="box")
    axA.tick_params(length=2, labelsize=7)
    for sp in ("top", "right"):
        axA.spines[sp].set_visible(False)

    # (B) the observed EPI (background texture + both patch lines)
    axB.set_title(
        "2   EPI: textured background\n     + reflective patches",
        loc="left",
        fontsize=10,
    )
    disp = paint(paint(B_EPI, X1, L1), X2, L2)
    axB.imshow(
        np.clip(disp, 0, 1),
        origin="lower",
        aspect="auto",
        extent=[q_vals[0], q_vals[-1], s_vals[0], s_vals[-1]],
        interpolation="nearest",
    )
    axB.text(
        X1 / ZP - 0.04,
        0.46,
        "patch 1",
        fontsize=7.5,
        color="#0b3d36",
        ha="right",
        va="center",
        rotation=-50,
    )
    axB.text(
        X2 / ZP + 0.04,
        0.46,
        "patch 2",
        fontsize=7.5,
        color="#0b1f4d",
        ha="left",
        va="center",
        rotation=-50,
    )
    axB.set_xlim(q_vals[0], q_vals[-1])
    axB.set_ylim(s_vals[0], s_vals[-1])
    axB.set_xlabel("sensor coordinate  $u$", fontsize=8)
    axB.set_ylabel("view  $s$", fontsize=8)
    axB.tick_params(length=2, labelsize=7)

    # (C) the three correlation losses
    axC.set_title("3   photometric loss vs candidate position", loc="left", fontsize=10)
    axC.plot(
        xv,
        loss_relight,
        color="#c0392b",
        lw=2.2,
        label="relight (recompute reflection)",
    )
    axC.plot(
        xv, loss_freeze, color="#e08a1e", lw=2.0, label="freeze (carry patch-1 colours)"
    )
    axC.plot(
        xv, loss_diffuse, color="#3b6fb0", lw=2.0, label="diffuse (mean colour only)"
    )
    axC.axvline(X1, color="#2a9d8f", lw=1.0, ls=":", alpha=0.8)
    axC.axvline(X2, color="#1d3f8f", lw=1.0, ls=":", alpha=0.8)
    ymax = max(loss_relight.max(), loss_freeze.max(), loss_diffuse.max()) * 1.16
    axC.text(
        X1 - 0.02,
        ymax * 0.5,
        "start $x_1$",
        color="#2a9d8f",
        fontsize=7.5,
        ha="right",
        va="center",
    )
    axC.text(
        X2 + 0.02,
        ymax * 0.5,
        "true disp. $x_2$",
        color="#1d3f8f",
        fontsize=7.5,
        ha="left",
        va="center",
    )
    for L, c in (
        (loss_relight, "#c0392b"),
        (loss_freeze, "#e08a1e"),
        (loss_diffuse, "#3b6fb0"),
    ):
        axC.scatter([xv[L.argmin()]], [L.min()], color=c, s=26, zorder=5)
    axC.set_xlim(xv[0], xv[-1])
    axC.set_ylim(0, ymax)
    axC.set_xlabel("virtual-patch position  $x_v$", fontsize=8)
    axC.set_ylabel("photometric loss", fontsize=8)
    axC.tick_params(length=2, labelsize=7)
    for sp in ("top", "right"):
        axC.spines[sp].set_visible(False)
    axC.legend(
        fontsize=7,
        loc="upper center",
        frameon=False,
        handlelength=1.8,
        labelspacing=0.3,
        borderpad=0.1,
    )

    fig.tight_layout(w_pad=1.4)
    fig.savefig("toy_reflection_motion.pdf", bbox_inches="tight")
    fig.savefig("toy_reflection_motion.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote toy_reflection_motion.pdf")
    print(
        f"relight_min={loss_relight.min():.3f}@{xv[loss_relight.argmin()]:+.3f}  "
        f"freeze_min={loss_freeze.min():.3f}@{xv[loss_freeze.argmin()]:+.3f}  "
        f"diffuse_min={loss_diffuse.min():.3f}@{xv[loss_diffuse.argmin()]:+.3f}  (x2={X2:+.2f})"
    )


def _gif_frames(outdir):
    """Dump grayscale-photometric-loss frames as the virtual EPI slides."""
    os.makedirs(outdir, exist_ok=True)
    plt.rcParams.update(
        {"font.size": 9, "font.family": "DejaVu Sans", "axes.linewidth": 0.8}
    )
    sweep = np.linspace(-0.55, 0.55, 91)
    emax = float(
        ((paint(B_EPI, xv[0], patch_profile(xv[0])) - T_REAL) ** 2).sum(2).max()
    )
    for fi, x in enumerate(sweep):
        V = paint(B_EPI, x, patch_profile(x))
        diff = ((V - T_REAL) ** 2).sum(2)  # grayscale photometric loss
        loss_now = float(diff.mean())
        fig, (a0, a1, a2) = plt.subplots(
            1, 3, figsize=(11.5, 3.4), gridspec_kw={"width_ratios": [1, 1, 1.25]}
        )
        ext = [q_vals[0], q_vals[-1], s_vals[0], s_vals[-1]]
        a0.imshow(
            np.clip(V, 0, 1),
            origin="lower",
            aspect="auto",
            extent=ext,
            interpolation="nearest",
        )
        a0.plot((X2 - s_vals) / ZP, s_vals, color="#222", lw=0.8, ls=(0, (3, 2)))
        a0.set_title("virtual EPI  (slides)  +  target line", fontsize=9, loc="left")
        a0.set_xlabel("u", fontsize=8)
        a0.set_ylabel("view s", fontsize=8)
        a0.tick_params(labelsize=7, length=2)
        a1.imshow(
            diff,
            origin="lower",
            aspect="auto",
            extent=ext,
            cmap="gray_r",
            vmin=0,
            vmax=emax,
        )
        a1.set_title("grayscale photometric loss  $|V-T|^2$", fontsize=9, loc="left")
        a1.set_xlabel("u", fontsize=8)
        a1.tick_params(labelsize=7, length=2)
        a2.plot(xv, loss_relight, color="#c0392b", lw=2.0, label="relight")
        a2.plot(xv, loss_freeze, color="#e08a1e", lw=1.6, label="freeze")
        a2.plot(xv, loss_diffuse, color="#3b6fb0", lw=1.6, label="diffuse")
        a2.axvline(x, color="#222", lw=1.2)
        a2.scatter([x], [loss_now], color="#c0392b", s=30, zorder=5)
        a2.axvline(X2, color="#1d3f8f", lw=0.8, ls=":", alpha=0.7)
        a2.set_xlim(xv[0], xv[-1])
        a2.set_ylim(
            0, max(loss_relight.max(), loss_freeze.max(), loss_diffuse.max()) * 1.1
        )
        a2.set_title(
            f"loss curves   ($x_v$={x:+.2f},  loss={loss_now:.2f})",
            fontsize=9,
            loc="left",
        )
        a2.set_xlabel("virtual-patch position $x_v$", fontsize=8)
        a2.tick_params(labelsize=7, length=2)
        a2.legend(
            fontsize=7, loc="upper center", frameon=False, ncol=3, handlelength=1.4
        )
        for sp in ("top", "right"):
            a2.spines[sp].set_visible(False)
        fig.tight_layout(w_pad=1.1)
        fig.savefig(os.path.join(outdir, f"frame_{fi:03d}.png"), dpi=96)
        plt.close(fig)
    print(f"wrote {len(sweep)} frames to {outdir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="", help="dir to dump GIF frames into")
    args = ap.parse_args()
    _main_figure()
    if args.frames:
        _gif_frames(args.frames)
