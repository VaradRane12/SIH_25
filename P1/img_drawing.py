# kolam_generator.py
import os
import numpy as np
import matplotlib.pyplot as plt

def make_dot_field(n, img_size=512, margin=20, sigma=6.0, sign_pattern="checker"):
    """
    Create a Gaussian field from an n x n dot grid.
    
    Parameters:
        n (int): grid size (n x n dots).
        img_size (int): output matrix size in pixels.
        margin (int): pixel margin around dots.
        sigma (float): gaussian sigma in pixels.
        sign_pattern (str): "checker", "rows", or "allpos".
    Returns:
        field (2D numpy array)
    """
    xs = np.linspace(margin, img_size - margin, n)
    ys = np.linspace(margin, img_size - margin, n)
    X, Y = np.meshgrid(np.arange(img_size), np.arange(img_size))
    field = np.zeros_like(X, dtype=float)

    for i, xi in enumerate(xs):
        for j, yj in enumerate(ys):
            if sign_pattern == "checker":
                sign = 1.0 if (i + j) % 2 == 0 else -1.0
            elif sign_pattern == "rows":
                sign = 1.0 if i % 2 == 0 else -1.0
            else:  # all positive
                sign = 1.0
            field += sign * np.exp(-((X - xi) ** 2 + (Y - yj) ** 2) / (2 * sigma * sigma))
    return field


def add_scale_layers(field, n_layers=2, scale=0.5):
    """
    Add coarser/finer Gaussian layers to create nested loops.
    """
    out = field.copy()
    base = field.copy()
    for k in range(1, n_layers):
        # Add mirrored and scaled layers for complexity
        out += (scale ** k) * base[::-1, ::-1]
    return out


def enforce_symmetry(field, axes=("vertical", "horizontal")):
    """
    Enforce symmetry in the field.
    
    Parameters:
        field (2D numpy array).
        axes (tuple): can include "vertical", "horizontal".
    """
    out = field.copy()
    if "vertical" in axes:
        out = 0.5 * (out + out[:, ::-1])
    if "horizontal" in axes:
        out = 0.5 * (out + out[::-1, :])
    return out


def draw_kolam(n=5,
               img_size=512,
               sigma=6.0,
               level=0.15,
               sign_pattern="checker",
               mirror_axes=("vertical", "horizontal"),
               n_layers=2,
               out_path="kolam_generated.png"):
    """
    Draw a Kolam pattern and save as PNG.
    """
    field = make_dot_field(n, img_size=img_size, sigma=sigma, sign_pattern=sign_pattern)
    field = add_scale_layers(field, n_layers=n_layers)
    field = enforce_symmetry(field, axes=mirror_axes)

    # Normalize field
    field = (field - field.min()) / (field.max() - field.min())

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    ax.set_facecolor("white")
    ax.axis("off")

    # --- START OF FIX ---

    # Define the contour levels and the alternating line widths
    n_contours = 5
    levels = np.linspace(level, level * 3, n_contours)
    linewidths = [2.0 if i % 2 == 0 else 0.9 for i in range(n_contours)]

    # Pass the list of line widths directly to the contour function
    cs = ax.contour(field, levels=levels, linewidths=linewidths, antialiased=True, colors="black")

    # The loop is no longer needed, so you can remove it.
    # for i, collection in enumerate(cs.collections):
    #     lw = 2.0 if i % 2 == 0 else 0.9
    #     collection.set_linewidth(lw)

    # --- END OF FIX ---

    # Draw dot-grid
    xs = np.linspace(20, img_size - 20, n)
    ys = np.linspace(20, img_size - 20, n)
    for xi in xs:
        for yj in ys:
            ax.scatter(xi, yj, s=8, c="black", zorder=10)

    ax.set_xlim(0, field.shape[1])
    ax.set_ylim(0, field.shape[0])
    ax.invert_yaxis()
    plt.tight_layout(pad=0)
    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print("Saved", out_path)

if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    # Example: generate a 5x5 Kolam
    draw_kolam(n=5,
               img_size=700,
               sigma=12.0,
               level=0.12,
               sign_pattern="checker",
               mirror_axes=("vertical", "horizontal"),
               n_layers=3,
               out_path="outputs/kolam_5x5.png")
