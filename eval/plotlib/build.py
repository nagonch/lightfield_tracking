"""Render every production figure."""

from .common import set_style
from .reflectivity import reflectivity_figure
from .depth_influence import depth_influence_figure


def main():
    set_style()
    reflectivity_figure(["FP", "BundleSDF", "LoFTR", "Ours"], "reflectivity_main")
    depth_influence_figure(["FP", "BundleSDF", "Ours"], "depth_influence")


if __name__ == "__main__":
    main()
