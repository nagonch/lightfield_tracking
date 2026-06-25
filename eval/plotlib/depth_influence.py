"""Ground-truth vs. sensor depth on the objects and the cube, ADD-S (seaborn)."""

import pandas as pd
import seaborn as sns

from .common import COLOR, REFL, long_frame, plt, save_fig

DEPTHS = [("synth", "sensor depth"), ("gt", "GT depth")]
MARKERS = {"sensor depth": "o", "GT depth": "s"}
DASHES = {"sensor depth": "", "GT depth": (4, 1.5)}
GEOMS = [("objects", "Objects"), ("cube", "Cube")]


def depth_influence_figure(methods, out_name):
    # objects | cube side by side -> square-ish panels, compact figure
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.7))
    handles = labels = None

    for col, (geom, geom_title) in enumerate(GEOMS):
        ax = axes[col]
        df = long_frame(methods, geom, "adds_auc", DEPTHS)
        df["Depth"] = pd.Categorical(df["Depth"], ["sensor depth", "GT depth"])

        sns.lineplot(
            data=df, x="Reflectivity", y="value", hue="Method",
            hue_order=methods, style="Depth", palette=COLOR,
            markers=MARKERS, dashes=DASHES, markersize=6.5, linewidth=1.9,
            errorbar=None, ax=ax, legend=(handles is None),
        )
        if handles is None:  # capture the shared Method+Depth legend once
            handles, labels = ax.get_legend_handles_labels()
            ax.legend_.remove()

        # emphasise Ours by overdrawing both depth curves thicker (still seaborn)
        sns.lineplot(
            data=df[df["Method"] == "Ours"], x="Reflectivity", y="value",
            style="Depth", color=COLOR["Ours"], markers=MARKERS, dashes=DASHES,
            markersize=7.5, linewidth=3.0, errorbar=None, ax=ax, legend=False,
        )

        ax.set_xticks(REFL)
        ax.set_xlim(-0.04, 1.04)
        ax.margins(y=0.08)
        ax.set_ylabel("ADD-S  ($\\uparrow$)")
        ax.set_xlabel("Reflectivity  $r$")
        ax.set_title(geom_title, pad=8)

    # one combined legend (Method + Depth), to the right so it never covers curves
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5),
               frameon=True, framealpha=0.95, edgecolor="#cccccc")
    fig.suptitle("Tracking quality: ground-truth vs. sensor depth",
                 fontsize=14.5, fontweight="bold", y=0.99)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    save_fig(fig, out_name)
