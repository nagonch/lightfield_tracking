#!/usr/bin/env python3
"""Production figures for the ReLiFT-6DoF evaluation (seaborn-styled).

Usage (inside container via bash run_container.sh):
    python eval/build_plot_prod.py

Outputs (eval/plots/):
    reflectivity_main.{pdf,png}  FP / BundleSDF / LoFTR / Ours, textured objects + cube
    depth_influence.{pdf,png}    FP / BundleSDF / Ours, ground-truth vs. sensor depth

The plotting code lives in the ``plotlib`` package (one module per figure type)
so new figures can be added without disturbing the existing ones.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from plotlib.build import main  # noqa: E402

if __name__ == "__main__":
    main()
