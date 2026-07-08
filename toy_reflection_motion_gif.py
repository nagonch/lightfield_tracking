"""
toy_reflection_motion_gif.py
============================
Animated (website) version of the paper figure `toy_reflection_motion.pdf`.

Layout mirrors the paper figure exactly:
  a) Scene composition   -- static, redrawn identically every frame
  b) EPI correlation     -- virtual patch band slides across the target EPI
                            (RGB correlation on top, per-pixel photometric loss below)
  c) Result loss         -- the corresponding loss curve is traced out

Three sweeps; every model is correlated against the same REAL observed target EPI,
only the virtual-patch colour model changes (matching the paper legend):
  1. diffuse loss          (orange) -- flat mean colour, no view dependence (classic vision)
  2. EPI loss              (blue)   -- carry the observed EPI colours rigidly
  3. relighted colour loss (red)    -- recompute the reflection at the candidate

Usage:  python toy_reflection_motion_gif.py [--outdir DIR] [--test]
        (--test renders 3 probe frames only)
Frames are dumped as PNG; assemble with ffmpeg palettegen (see bottom).
"""

import argparse
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from toy_reflection_motion import (
    ZP, X1, X2, ENV_Z, BG_Z,
    s_vals, q_vals, B_EPI,
    paint, patch_profile, env_color, bg_tex, mse,
    L1, L2, M1,
    T_REAL,
    xv, loss_relight, loss_freeze,
)

# diffuse model (flat mean colour) correlated against the REAL observed target,
# so all three losses share the identical target EPI
loss_diffuse_real = np.array([mse(paint(B_EPI, x, M1), T_REAL) for x in xv])

# paper colours (legend order: diffuse / EPI / relighted colour)
C_DIF, C_EPI, C_REL = "#e08a1e", "#3b6fb0", "#c0392b"
COL_P1, COL_P2 = "#9ec9e8", "#f6c489"  # patch fills (blue / orange, as paper)
EDGE_P1, EDGE_P2 = "#4a7fab", "#c8823a"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["cmr10", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 12,
        "axes.unicode_minus": False,
    }
)

YMAX = float(max(loss_relight.max(), loss_freeze.max(), loss_diffuse_real.max()))

# the three sweeps: (legend label, subtitle, colour, loss curve, virtual colours)
CASES = [
    ("diffuse loss", "diffuse", C_DIF, loss_diffuse_real, lambda x: M1),
    ("EPI loss", "EPI", C_EPI, loss_freeze, lambda x: L1),
    ("relighted colour loss", "relighted", C_REL, loss_relight, patch_profile),
]

# common scale for the per-pixel loss maps (worst residual over all sweeps)
EMAX = max(
    float(((paint(B_EPI, xv[0], virt(xv[0])) - T_REAL) ** 2).sum(2).max())
    for _, _, _, _, virt in CASES
)


# ──────────────────────────────────────────────────────────────────────────────
#  Panel a)  Scene composition  (static)
# ──────────────────────────────────────────────────────────────────────────────
def draw_scene(ax):
    ax.set_title("a) Scene composition", fontsize=16, pad=14)
    ax.set_xlim(-1.48, 1.50)
    ax.set_ylim(-1.62, 2.78)
    ax.axis("off")
    y0, xa0 = -1.28, -1.32  # axis origin

    # axis arrows
    ax.annotate("", xy=(1.44, y0), xytext=(xa0, y0),
                arrowprops=dict(arrowstyle="-|>", color="k", lw=1.1, mutation_scale=13))
    ax.annotate("", xy=(xa0, 2.62), xytext=(xa0, y0),
                arrowprops=dict(arrowstyle="-|>", color="k", lw=1.1, mutation_scale=13))
    ax.text(1.46, y0, "$x$", fontsize=14, ha="left", va="center")
    ax.text(xa0, 2.70, "$z$", fontsize=14, ha="center", va="bottom")

    # textured background wall (top)
    bgrad = bg_tex(np.linspace(-1.05, 1.05, 300))[None]
    ax.imshow(bgrad, extent=[-1.05, 1.05, BG_Z - 0.08, BG_Z + 0.08],
              aspect="auto", zorder=2)
    ax.text(0.0, BG_Z + 0.20, "Textured background", fontsize=12,
            ha="center", va="bottom")

    # dotted verticals to the x axis (behind everything)
    for x in (X1, X2):
        ax.plot([x, x], [ZP - 0.48, y0], color="#888", lw=0.9,
                ls=(0, (2, 3)), zorder=1)
    ax.text(X1, y0 - 0.12, r"$\mathbf{x}_t$", fontsize=13, ha="center", va="top")
    ax.text(X2, y0 - 0.12, r"$\mathbf{x}_{t+1}$", fontsize=13, ha="center", va="top")

    # the two patches (parallelograms) + reflection lobes + normals
    for x, fc, ec, lab in (
        (X1, COL_P1, EDGE_P1, r"$\mathcal{P}_t$"),
        (X2, COL_P2, EDGE_P2, r"$\mathcal{P}_{t+1}$"),
    ):
        quad = [(x - 0.20, ZP + 0.04), (x - 0.08, ZP + 0.17),
                (x + 0.20, ZP + 0.13), (x + 0.08, ZP - 0.00)]
        ax.add_patch(plt.Polygon(quad, closed=True, facecolor=fc,
                                 edgecolor=ec, lw=1.0, zorder=4))
        ax.text(x, ZP + 0.085, lab, fontsize=12, ha="center", va="center", zorder=5)
        # normal
        ax.annotate("", xy=(x, ZP - 0.44), xytext=(x, ZP + 0.02),
                    arrowprops=dict(arrowstyle="-|>", color="k", lw=1.0,
                                    mutation_scale=11), zorder=6)
        ax.text(x + 0.06, ZP - 0.38, r"$\mathbf{n}$", fontsize=12,
                ha="left", va="center", zorder=6)
        # reflection lobe, coloured by the environment it reflects
        t = np.linspace(-1, 1, 80)
        lx = x + 0.185 * t
        lz = ZP - 0.035 - 0.30 * np.exp(-2.2 * t**2)
        xh = x + ((ZP - ENV_Z) / ZP) * (x - np.linspace(0.55, -0.55, 80))
        cols = env_color(np.sort(xh))
        pts = np.stack([lx, lz], axis=1)
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, colors=cols[:-1], lw=3.2, capstyle="round", zorder=5)
        ax.add_collection(lc)

    # translation arrow between the patches
    ax.annotate("", xy=(X2 - 0.02, ZP + 0.33), xytext=(X1 + 0.02, ZP + 0.33),
                arrowprops=dict(arrowstyle="-|>", color="k", lw=1.2,
                                mutation_scale=13), zorder=6)
    ax.text((X1 + X2) / 2, ZP + 0.41, r"$\Delta\mathbf{x}$", fontsize=13,
            ha="center", va="bottom", zorder=6)

    # static light-field cameras at z = 0
    for s in np.linspace(-0.55, 0.55, 6):
        ax.add_patch(plt.Rectangle((s - 0.055, -0.125), 0.11, 0.085,
                                   facecolor="#5a5a5a", edgecolor="#222",
                                   lw=0.7, zorder=4))
        ax.add_patch(plt.Polygon(
            [(s - 0.033, -0.04), (s + 0.033, -0.04),
             (s + 0.058, 0.045), (s - 0.058, 0.045)],
            closed=True, facecolor="#5a5a5a", edgecolor="#222", lw=0.7, zorder=4))

    # environment illumination (behind the cameras)
    grad = env_color(np.linspace(-1.05, 1.05, 300))[None]
    ax.imshow(grad, extent=[-1.05, 1.05, ENV_Z - 0.08, ENV_Z + 0.08],
              aspect="auto", zorder=2)
    ax.text(0.0, ENV_Z - 0.20, "Environment illumination", fontsize=12,
            ha="center", va="top")


# ──────────────────────────────────────────────────────────────────────────────
#  Panel b)  EPI correlation
# ──────────────────────────────────────────────────────────────────────────────
def draw_epi(ax, rgb, lossmap, subtitle="", sub_color="k"):
    ax.set_title("b) EPI correlation", fontsize=16, pad=14)
    ax.set_xlim(-1.34, 1.22)
    ax.set_ylim(-1.42, 1.80)
    ax.axis("off")
    ox = -1.16  # axis x
    # correlation equation, then the current-model subtitle
    ax.text(0.0, 1.58, r"$L_t(s,u)\;\overset{\star}{\longrightarrow}\;L_{t+1}(s,u)$",
            fontsize=12.5, ha="center", va="bottom")
    if subtitle:
        ax.text(0.0, 1.24, subtitle, fontsize=13, ha="center", va="bottom",
                style="italic", color=sub_color)
    # RGB correlation (top) and per-pixel photometric loss (bottom)
    ax.imshow(np.clip(rgb, 0, 1), origin="lower", aspect="auto",
              extent=[q_vals[0], q_vals[-1], 0.10, 1.10],
              interpolation="nearest", zorder=2)
    ax.imshow(lossmap, origin="lower", aspect="auto",
              extent=[q_vals[0], q_vals[-1], -1.16, -0.16],
              cmap="gray_r", vmin=0, vmax=EMAX,
              interpolation="nearest", zorder=2)
    ax.text(1.06, 0.60, "RGB correlation", fontsize=9, color="#555",
            ha="left", va="center", rotation=270)
    ax.text(1.06, -0.66, "photometric loss", fontsize=9, color="#555",
            ha="left", va="center", rotation=270)
    # axes arrows: s alongside the RGB image, u under the loss map
    ax.annotate("", xy=(1.12, -1.30), xytext=(ox, -1.30),
                arrowprops=dict(arrowstyle="-|>", color="k", lw=1.1, mutation_scale=13))
    ax.annotate("", xy=(ox, 1.16), xytext=(ox, 0.10),
                arrowprops=dict(arrowstyle="-|>", color="k", lw=1.1, mutation_scale=13))
    ax.annotate("", xy=(ox, -0.10), xytext=(ox, -1.16),
                arrowprops=dict(arrowstyle="-|>", color="k", lw=1.1, mutation_scale=13))
    ax.text(1.15, -1.30, "$u$", fontsize=14, ha="left", va="center")
    ax.text(ox - 0.07, 1.12, "$s$", fontsize=14, ha="right", va="center")
    ax.text(ox - 0.07, -0.14, "$s$", fontsize=14, ha="right", va="center")


# ──────────────────────────────────────────────────────────────────────────────
#  Panel c)  Result loss
# ──────────────────────────────────────────────────────────────────────────────
def draw_loss(ax, done, current=None):
    """done: list of case indices fully drawn.  current: (case_idx, x_now) or None."""
    ax.set_title("c) Result loss", fontsize=16, pad=14)
    ax.set_xlim(-0.82, 0.84)
    ax.set_ylim(-0.30 * YMAX, 1.52 * YMAX)
    ax.axis("off")
    ox = -0.72
    ax.annotate("", xy=(0.78, 0), xytext=(ox, 0),
                arrowprops=dict(arrowstyle="-|>", color="k", lw=1.1, mutation_scale=13))
    ax.annotate("", xy=(ox, 1.46 * YMAX), xytext=(ox, 0),
                arrowprops=dict(arrowstyle="-|>", color="k", lw=1.1, mutation_scale=13))
    ax.text(0.81, 0, "$x$", fontsize=14, ha="left", va="center")
    ax.text(ox - 0.055, 0.66 * YMAX, "Photometric loss", fontsize=13,
            ha="center", va="center", rotation=90)

    # dotted markers of start / target position
    for x, lab in ((X1, r"$\mathbf{x}_t$"), (X2, r"$\mathbf{x}_{t+1}$")):
        ax.plot([x, x], [0, 1.30 * YMAX], color="#888", lw=0.9,
                ls=(0, (2, 3)), zorder=1)
        ax.text(x, -0.055 * YMAX, lab, fontsize=13, ha="center", va="top")

    # legend (always complete, as in the paper)
    handles = [Line2D([], [], color=c[2], lw=2.4, label=c[0]) for c in CASES]
    leg = ax.legend(handles=handles, fontsize=11, loc="upper center",
                    frameon=True, fancybox=True, borderpad=0.55,
                    labelspacing=0.35, handlelength=1.9)
    leg.get_frame().set_edgecolor("#aaaaaa")
    leg.get_frame().set_linewidth(0.9)

    for i in done:
        col, loss = CASES[i][2], CASES[i][3]
        ax.plot(xv, loss, color=col, lw=2.4, zorder=3)
        ax.scatter([xv[loss.argmin()]], [loss.min()], color=col, s=34,
                   zorder=5, edgecolors="white", linewidths=0.8)

    if current is not None:
        i, x_now = current
        col, loss = CASES[i][2], CASES[i][3]
        m = xv <= x_now + 1e-9
        ax.plot(xv[m], loss[m], color=col, lw=2.4, zorder=4)
        y_now = float(np.interp(x_now, xv, loss))
        ax.plot([x_now, x_now], [0, 1.30 * YMAX], color="#bbb", lw=0.9, zorder=1)
        ax.scatter([x_now], [y_now], color=col, s=42, zorder=6,
                   edgecolors="white", linewidths=0.9)


# ──────────────────────────────────────────────────────────────────────────────
#  Frames
# ──────────────────────────────────────────────────────────────────────────────
def render_frame(path, rgb, lossmap, done, current, subtitle="", sub_color="k"):
    fig, (axA, axB, axC) = plt.subplots(
        1, 3, figsize=(15.0, 5.1), gridspec_kw={"width_ratios": [1.08, 1.0, 1.04]}
    )
    draw_scene(axA)
    draw_epi(axB, rgb, lossmap, subtitle, sub_color)
    draw_loss(axC, done, current)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(path, dpi=100, facecolor="white")
    plt.close(fig)


def epi_for(case_idx, x_virtual):
    virt = CASES[case_idx][4]
    V = paint(B_EPI, x_virtual, virt(x_virtual))
    rgb = paint(paint(B_EPI, X2, L2), x_virtual, virt(x_virtual))
    lossmap = ((V - T_REAL) ** 2).sum(2)
    return rgb, lossmap


EPI_STATIC = paint(paint(B_EPI, X1, L1), X2, L2)  # the paper's static panel b)
LOSS_STATIC = ((paint(B_EPI, X1, L1) - T_REAL) ** 2).sum(2)  # initial misalignment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="frames_reflection_motion")
    ap.add_argument("--test", action="store_true", help="render 3 probe frames only")
    ap.add_argument("--sweep-frames", type=int, default=56)
    ap.add_argument("--hold-frames", type=int, default=10)
    ap.add_argument("--final-frames", type=int, default=22)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    if args.test:
        for name, (rgb, lm), done, cur, sub, sc in [
            ("test_static", (EPI_STATIC, LOSS_STATIC), [0, 1, 2], None, "", "k"),
            ("test_sweep0", epi_for(0, -0.10), [], (0, -0.10), CASES[0][1], CASES[0][2]),
            ("test_sweep1", epi_for(1, -0.05), [0], (1, -0.05), CASES[1][1], CASES[1][2]),
            ("test_sweep2", epi_for(2, 0.05), [0, 1], (2, 0.05), CASES[2][1], CASES[2][2]),
        ]:
            render_frame(os.path.join(args.outdir, name + ".png"),
                         rgb, lm, done, cur, sub, sc)
        print("wrote 4 test frames to", args.outdir)
        return

    fi = 0
    sweep = np.linspace(xv[0], xv[-1], args.sweep_frames)
    for ci, (_, sub, col, loss, _) in enumerate(CASES):
        done = list(range(ci))
        for x in sweep:
            rgb, lm = epi_for(ci, x)
            render_frame(os.path.join(args.outdir, f"frame_{fi:04d}.png"),
                         rgb, lm, done, (ci, x), sub, col)
            fi += 1
        # hold at the minimum, curve complete
        x_min = float(xv[loss.argmin()])
        rgb, lm = epi_for(ci, x_min)
        for _ in range(args.hold_frames):
            render_frame(os.path.join(args.outdir, f"frame_{fi:04d}.png"),
                         rgb, lm, done + [ci], None, sub, col)
            fi += 1
    # final hold: the exact paper figure (both patches, all curves)
    for _ in range(args.final_frames):
        render_frame(os.path.join(args.outdir, f"frame_{fi:04d}.png"),
                     EPI_STATIC, LOSS_STATIC, [0, 1, 2], None)
        fi += 1
    print(f"wrote {fi} frames to {args.outdir}")
    print("assemble:  ffmpeg -y -framerate 16 -i "
          f"{args.outdir}/frame_%04d.png -vf "
          '"split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4" '
          "toy_reflection_motion_anim.gif")


if __name__ == "__main__":
    main()
