"""Small plotting helpers shared by all notebooks."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

RANDOM_STATE = 42


def save_fig(name: str, folder: str | Path = "images", dpi: int = 120) -> Path:
    """Save the current figure as ``<folder>/<name>.png`` (folder is relative to the notebook)."""
    path = Path(folder) / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    return path


def plot_digits(images, n_cols: int = 10, size: int = 28, titles=None):
    """Plot a grid of flattened square images (MNIST: size=28, digits: size=8, faces: size=64)."""
    images = list(images)
    n_rows = (len(images) - 1) // n_cols + 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 0.9, n_rows * 0.9 + 0.2))
    for i, ax in enumerate(getattr(axes, "flat", [axes])):
        ax.axis("off")
        if i < len(images):
            ax.imshow(images[i].reshape(size, size), cmap="binary" if size != 64 else "gray")
            if titles is not None:
                ax.set_title(str(titles[i]), fontsize=8)
    return fig
