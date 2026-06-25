"""Production plotting library for the ReLiFT-6DoF evaluation figures.

Each plot type lives in its own module so new figures can be added without
touching the existing ones:

    common.py           shared style / palette / data loading / saving
    reflectivity.py     accuracy vs. reflectivity (objects + cube, 3 metrics)
    depth_influence.py  ground-truth vs. sensor depth comparison
    build.py            entry point that renders every figure

Render everything with ``python eval/build_plot_prod.py``.
"""
