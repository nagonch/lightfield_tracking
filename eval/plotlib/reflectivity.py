"""Accuracy vs. reflectivity for the objects and the cube (drawn with seaborn)."""

import seaborn as sns

from .common import COLOR, REFL, long_frame, plt, save_fig

# (metric_key, axis label, log_y)
METRICS = [
    ("add_auc", "ADD  ($\\uparrow$)", False),
    ("ate_rmse", "ATE RMSE [m]  ($\\downarrow$)", True),
    ("mean_abs_rot_deg", "Rotation error [$^\\circ$]  ($\\downarrow$)", True),
]
GEOMS = [("objects", "Objects"), ("cube", "Cube")]
SENSOR = [("synth", "sensor")]


def reflectivity_figure(methods, out_name):
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.4), sharex=True)
    handles = labels = None

    for row, (geom, geom_title) in enumerate(GEOMS):
        for col, (mkey, mlabel, logy) in enumerate(METRICS):
            ax = axes[row, col]
            df = long_frame(methods, geom, mkey, SENSOR)

            sns.lineplot(
                data=df, x="Reflectivity", y="value",
                hue="Method", hue_order=methods, palette=COLOR,
                marker="o", markersize=5.5, linewidth=1.9, errorbar=None,
                ax=ax, legend=(handles is None),
            )
            if handles is None:  # capture the shared legend once, then drop it
                handles, labels = ax.get_legend_handles_labels()
                ax.legend_.remove()

            # emphasise Ours by overdrawing it thicker (still seaborn)
            sns.lineplot(
                data=df[df["Method"] == "Ours"], x="Reflectivity", y="value",
                color=COLOR["Ours"], marker="o", markersize=8.5, linewidth=3.4,
                errorbar=None, ax=ax, legend=False,
            )

            if logy:
                ax.set_yscale("log")
            ax.set_xticks(REFL)
            ax.set_xlim(-0.04, 1.04)
            ax.margins(y=0.08)
            ax.set_ylabel("")
            ax.set_xlabel("Reflectivity  $r$" if row == 1 else "")
            if row == 0:
                ax.set_title(mlabel, pad=8)
        axes[row, 0].annotate(
            geom_title, xy=(-0.22, 0.5), xycoords="axes fraction", rotation=90,
            ha="center", va="center", fontsize=12.5, fontweight="bold", color="#333333",
        )

    labels = [f"{x}  (ours)" if x == "Ours" else x for x in labels]
    fig.legend(handles, labels, loc="lower center", ncol=len(methods),
               frameon=True, framealpha=0.95, edgecolor="#cccccc",
               bbox_to_anchor=(0.5, -0.012), handlelength=2.2, columnspacing=1.6)
    fig.suptitle("Tracking accuracy vs. reflectivity", fontsize=14.5,
                 fontweight="bold", y=0.99)
    fig.tight_layout(rect=(0.02, 0.05, 1, 0.96))
    save_fig(fig, out_name)
