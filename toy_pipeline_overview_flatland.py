"""
toy_pipeline_overview_flatland.py
=================================
A single CVPR-style method figure that UNIFIES the three stages of the system on
one diagram, by showing that they each recover a different unknown of the SAME
dichromatic surface-light-field (SLF) equation:

        O(p,s) = alpha * D(p) + (1 - alpha) * E( reflect(p, s; n) )
                  └─②─┘ └─③─┘            └─③─┘            └─①─┘

    ① Depth estimation  (LF plane-sweep)  ->  geometry, normals n(p), reflect()
    ② Alpha estimation  (view-variance)   ->  the diffuse fraction  alpha
    ③ Reflection separation (joint fit)   ->  diffuse D(p), env map E, confidence

The Surface Light Field O(p,s) is the shared hub: depth BUILDS it, alpha is a
STATISTIC of it, separation FACTORIZES it.

Reuses the flatland scene + faithful solver from
``toy_reflection_separation_flatland.py`` so the panels stay visually consistent.

Run:
    python toy_pipeline_overview_flatland.py
produces
    toy_pipeline_overview_flatland.{pdf,png}
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import toy_reflection_separation_flatland as toy

# ── stage colours (also colour-code the equation symbols) ──────────────────────
DEPTH_C = "#1f77b4"  # ① blue
ALPHA_C = "#e8821a"  # ② orange
SEP_C = "#2ca02c"  # ③ green
INK = "#222222"
_LUMW = np.array([0.299, 0.587, 0.114])


# ──────────────────────────────────────────────────────────────────────────────
#  colour-composited equation (left-to-right fragments, centred)
# ──────────────────────────────────────────────────────────────────────────────
def draw_equation(fig, ax, frags, fontsize=19, y=0.62):
    ax.axis("off")
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    axw = ax.get_window_extent(rend).width
    widths = []
    for txt, _ in frags:
        t = ax.text(0.5, y, txt, fontsize=fontsize, transform=ax.transAxes,
                    va="center", ha="left")
        widths.append(t.get_window_extent(rend).width)
        t.remove()
    x = 0.5 - (sum(widths) / axw) / 2.0
    for (txt, c), w in zip(frags, widths):
        ax.text(x, y, txt, fontsize=fontsize, color=c, transform=ax.transAxes,
                va="center", ha="left", fontweight="bold" if c != INK else "normal")
        x += w / axw


def _colour_spines(ax, c, lw=2.0):
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_edgecolor(c)
        sp.set_linewidth(lw)


# ──────────────────────────────────────────────────────────────────────────────
#  ① Depth: geometry scene + EPI disparity cue
# ──────────────────────────────────────────────────────────────────────────────
def stage_depth(ax):
    toy._draw_scene(ax)
    ax.set_title("①  Depth estimation  —  LF plane-sweep", fontsize=10,
                 loc="left", color=DEPTH_C, fontweight="bold")
    # EPI disparity inset: constant-surface-point lines, slope ∝ depth
    axin = ax.inset_axes([0.60, 0.28, 0.37, 0.34])
    axin.imshow(toy.epi_observed(), origin="lower", aspect="auto",
                extent=toy._EXT, interpolation="nearest")
    for xq in (-0.10, 0.0, 0.10):  # s = xq - q*ZP  (slope = -ZP ∝ depth)
        q = (xq - toy.s_vals) / toy.ZP
        axin.plot(q, toy.s_vals, color="#00e5ff", lw=1.1, alpha=0.95)
    axin.set_xlim(toy.q_vals[0], toy.q_vals[-1])
    axin.set_ylim(toy.s_vals[0], toy.s_vals[-1])
    axin.set_xticks([]); axin.set_yticks([])
    axin.set_title("EPI slope ∝ depth", fontsize=6.6, color="#333", pad=1.5)
    ax.text(0.02, 0.03,
            r"output:  surface points, normals $n(p)\Rightarrow\mathrm{reflect}(p,s)$",
            transform=ax.transAxes, fontsize=7.6, color=DEPTH_C,
            bbox=dict(boxstyle="round,pad=0.25", fc="#eaf2fb", ec=DEPTH_C, lw=0.8))


# ──────────────────────────────────────────────────────────────────────────────
#  ② Alpha: view-variance of the SLF + calibration to alpha
# ──────────────────────────────────────────────────────────────────────────────
def stage_alpha(ax):
    p0 = toy.P // 2
    lum_refl = (toy.O_FIT[p0] * _LUMW).sum(-1)  # reflective: swings across views
    base = float(lum_refl.mean())
    rng = np.random.default_rng(3)
    lum_diff = base + 0.012 * rng.standard_normal(toy.S)  # diffuse: ~flat
    s = toy.s_vals

    ax.plot(s, lum_refl, color="#7b2d8e", lw=1.4,
            label=r"reflective  $\alpha\!\approx\!0.1$  (high view-var.)")
    ax.plot(s, lum_diff, color="#3a7d44", lw=1.4,
            label=r"diffuse  $\alpha\!\approx\!0.9$  (flat)")
    # show the view-spread (std) of the reflective trace
    lo, hi = lum_refl.min(), lum_refl.max()
    ax.annotate("", xy=(s[3], hi), xytext=(s[3], lo),
                arrowprops=dict(arrowstyle="<->", color="#7b2d8e", lw=1.1))
    ax.text(s[5], (lo + hi) / 2, r"$\mathrm{std}_s(O)$", fontsize=7.5,
            color="#7b2d8e", va="center")
    ax.set_title("②  Reflectivity  α  estimation  —  view-variance", fontsize=10,
                 loc="left", color=ALPHA_C, fontweight="bold")
    ax.set_xlabel("view  $s$", fontsize=8)
    ax.set_ylabel("luminance  $O(p,s)$", fontsize=8)
    ax.tick_params(length=2, labelsize=6.5)
    ax.legend(fontsize=6.3, loc="upper right", framealpha=0.9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # calibration inset: stat -> (1 - alpha)
    axc = ax.inset_axes([0.13, 0.13, 0.34, 0.34])
    xs = np.linspace(0, 1, 50)
    axc.plot(xs, 0.06 + 0.9 * xs, color=ALPHA_C, lw=1.3)
    axc.scatter([0.85], [0.06 + 0.9 * 0.85], color="#7b2d8e", s=18, zorder=5)
    axc.scatter([0.12], [0.06 + 0.9 * 0.12], color="#3a7d44", s=18, zorder=5)
    axc.set_xlabel(r"$\mathrm{std}_s(O)$ stat", fontsize=6.2, labelpad=1)
    axc.set_ylabel(r"$1-\alpha$", fontsize=6.2, labelpad=1)
    axc.set_xticks([]); axc.set_yticks([])
    axc.set_title(r"$1-\alpha=\frac{\mathrm{stat}-\mathrm{F}}{\mathrm{S}}$",
                  fontsize=6.6, pad=2)
    for sp in ("top", "right"):
        axc.spines[sp].set_visible(False)
    ax.text(0.98, 0.02,
            r"output:  diffuse fraction $\alpha$",
            transform=ax.transAxes, fontsize=7.6, color=ALPHA_C, ha="right",
            bbox=dict(boxstyle="round,pad=0.25", fc="#fdf0e3", ec=ALPHA_C, lw=0.8))


# ──────────────────────────────────────────────────────────────────────────────
#  ③ Separation: the joint-fit result (D, E, confidence)
# ──────────────────────────────────────────────────────────────────────────────
def stage_separation(ax, D_est, E_est, conf):
    rows = [
        (np.clip(toy.D_TRUE[None], 0, 1), 5, 6, "true $D$"),
        (np.clip(D_est[None], 0, 1), 4, 5, r"est $\hat D$"),
        (np.clip(toy.E_TRUE[None], 0, 1), 2.5, 3.5, "true $E$"),
        (np.clip(E_est[None], 0, 1), 1.5, 2.5, r"est $\hat E$"),
        (np.repeat(conf[None, :, None], 3, axis=2), 0.4, 1.4, "conf $c$"),
    ]
    yt, yl = [], []
    for img, y0, y1, lab in rows:
        ax.imshow(img, aspect="auto", extent=[0, 1, y0, y1], vmin=0, vmax=1)
        yt.append((y0 + y1) / 2); yl.append(lab)
    ax.axhline(3.6, color="#bbb", lw=0.8)  # separate D block / E block
    ax.set_ylim(0.4, 6.0)
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_yticks(yt)
    ax.set_yticklabels(yl, fontsize=7.5)
    ax.text(0.5, 5.5, "diffuse  $D(p)$  (surface)", ha="center", fontsize=7,
            color="#444")
    ax.text(0.5, 3.0, "environment  $E$  (shared)", ha="center", fontsize=7,
            color="#444")
    ax.set_title("③  Reflection separation  —  joint Adam fit", fontsize=10,
                 loc="left", color=SEP_C, fontweight="bold")
    ax.text(0.5, -0.06,
            r"output:  $D(p)$,  $E$,  confidence $c$",
            transform=ax.transAxes, fontsize=7.6, color=SEP_C, ha="center",
            bbox=dict(boxstyle="round,pad=0.25", fc="#eaf6ea", ec=SEP_C, lw=0.8))


# ──────────────────────────────────────────────────────────────────────────────
#  data-flow bar:  LF -> SLF -> {alpha, separation} -> downstream
# ──────────────────────────────────────────────────────────────────────────────
def flow_bar(ax):
    ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    boxes = [
        (0.085, "Light field\n(sub-aperture views)", "#f2f2f2", "#888"),
        (0.355, "Surface Light Field\n$O(p,s)$", "#eef3f8", DEPTH_C),
        (0.635, "Diffuse $D$  +  Env $E$\n+  confidence $c$", "#eef6ee", SEP_C),
        (0.895, "Relighting  &\n6-DoF tracking", "#faf0f0", "#b03a2e"),
    ]
    cx = []
    for x, txt, fc, ec in boxes:
        ax.text(x, 0.5, txt, ha="center", va="center", fontsize=8.2, color=INK,
                bbox=dict(boxstyle="round,pad=0.5", fc=fc, ec=ec, lw=1.4))
        cx.append(x)
    labels = [
        (DEPTH_C, "①  build via depth"),
        (ALPHA_C + "/" + SEP_C, "②  α   ③  separate"),
        ("#b03a2e", "downstream"),
    ]
    arrow_cols = [DEPTH_C, SEP_C, "#b03a2e"]
    for i, (x0, x1) in enumerate(zip(cx[:-1], cx[1:])):
        ax.annotate("", xy=(x1 - 0.085, 0.5), xytext=(x0 + 0.085, 0.5),
                    arrowprops=dict(arrowstyle="-|>", color=arrow_cols[i], lw=2.0))
        ax.text((x0 + x1) / 2, 0.86, labels[i][1], ha="center", fontsize=7.3,
                color=arrow_cols[i] if i != 1 else INK)


# ──────────────────────────────────────────────────────────────────────────────
def main():
    plt.rcParams.update({
        "font.size": 9, "font.family": "DejaVu Sans",
        "axes.linewidth": 0.8, "pdf.fonttype": 42,
    })
    snaps, info = toy.solve()
    _, D_est, E_est, _ = snaps[-1]
    conf = info["conf"]

    fig = plt.figure(figsize=(14.5, 8.6))
    gs = fig.add_gridspec(
        3, 3, height_ratios=[0.92, 3.05, 1.05], hspace=0.30, wspace=0.24,
        left=0.045, right=0.985, top=0.99, bottom=0.02,
    )

    # ── unifying equation banner ──
    axeq = fig.add_subplot(gs[0, :])
    draw_equation(fig, axeq, [
        (r"$O(p,s)\;=\;$", INK),
        (r"$\alpha\,$", ALPHA_C),
        (r"$D(p)$", SEP_C),
        (r"$\;+\;(1-$", INK),
        (r"$\alpha$", ALPHA_C),
        (r"$)\;$", INK),
        (r"$E(\mathrm{reflect}(p,s;\,$", SEP_C),
        (r"$n$", DEPTH_C),
        (r"$))$", SEP_C),
    ])
    axeq.text(0.5, 0.12,
              "one dichromatic surface-light-field model  ·  each stage recovers a "
              "different unknown of the SAME equation",
              transform=axeq.transAxes, ha="center", fontsize=9.2, color="#555",
              style="italic")

    # ── three stages ──
    ax1 = fig.add_subplot(gs[1, 0]); stage_depth(ax1); _colour_spines(ax1, DEPTH_C, 1.2)
    ax2 = fig.add_subplot(gs[1, 1]); stage_alpha(ax2)
    ax3 = fig.add_subplot(gs[1, 2]); stage_separation(ax3, D_est, E_est, conf)
    _colour_spines(ax2, ALPHA_C, 1.2)
    _colour_spines(ax3, SEP_C, 1.2)

    # ── data-flow bar ──
    flow_bar(fig.add_subplot(gs[2, :]))

    fig.savefig("toy_pipeline_overview_flatland.pdf", bbox_inches="tight")
    fig.savefig("toy_pipeline_overview_flatland.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote toy_pipeline_overview_flatland.{pdf,png}")


if __name__ == "__main__":
    main()
