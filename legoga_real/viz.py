"""Rysowanie ukladu (matplotlib)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .geometry import ELEMENTS, apply
from .layout import Layout

COLOR = {"S16": "#3b3b3b", "S24": "#3b3b3b", "CL": "#3b3b3b", "CR": "#3b3b3b",
         "WL": "#c8102e", "WR": "#c8102e", "XX": "#0057b8"}


def draw(lay: Layout, ax=None, title: str | None = None):
    T, _ = lay.place()
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))
    for i, tname in enumerate(lay.types):
        if T[i] is None:
            continue
        col = COLOR.get(tname, "#3b3b3b")
        lw = 3.0 if tname in ("WL", "WR", "XX") else 2.0
        for (_, _, line) in ELEMENTS[tname].centerline:
            pts = [apply(T[i], p) for p in line]
            ax.plot([p[0] for p in pts], [p[1] for p in pts],
                    color=col, lw=lw, solid_capstyle="round")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15)
    ax.set_xlabel("study")
    if title:
        ax.set_title(title, fontsize=10)
    return ax


def save_gallery(layouts, path, cols: int = 3, titles=None):
    n = len(layouts)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 5.2 * rows))
    import numpy as _np
    axes = list(_np.atleast_1d(axes).flatten())
    for k, lay in enumerate(layouts):
        draw(lay, axes[k], titles[k] if titles else None)
    for k in range(n, len(axes)):
        axes[k].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
