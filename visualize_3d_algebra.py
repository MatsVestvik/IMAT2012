"""
Quick Matplotlib helpers for common 3D algebra visuals.

Run directly: python visualize_2d_algebra.py
Or import functions into a notebook for custom visuals.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


Vector = Tuple[float, float, float]


def _ensure_array(v: Sequence[float]) -> np.ndarray:
    arr = np.asarray(v, dtype=float)
    if arr.shape != (3,):
        raise ValueError("Expected a 3-element vector")
    return arr


@dataclass
class Basis:
    i: Vector = (1.0, 0.0, 0.0)
    j: Vector = (0.0, 1.0, 0.0)
    k: Vector = (0.0, 0.0, 1.0)


def plot_vectors(
    ax: plt.Axes,
    vectors: Iterable[Vector],
    *,
    origin: Vector = (0.0, 0.0, 0.0),
    colors: Iterable[str] | None = None,
    labels: Iterable[str] | None = None,
    arrow_ratio: float = 0.08,
) -> None:
    """
    Draw 3D arrows for each vector starting from a shared origin.
    """

    orig = _ensure_array(origin)
    colors_iter = colors if colors is not None else itertools.cycle("rgbcmyk")
    labels_iter = labels if labels is not None else itertools.repeat("")

    for vec, color, label in zip(vectors, colors_iter, labels_iter):
        v = _ensure_array(vec)
        ax.quiver(
            orig[0], orig[1], orig[2], v[0], v[1], v[2],
            arrow_length_ratio=arrow_ratio,
            color=color,
            label=label,
            linewidth=2.0,
        )


def plot_plane(
    ax: plt.Axes,
    point: Vector,
    normal: Vector,
    *,
    xlim: Tuple[float, float] = (-2, 2),
    ylim: Tuple[float, float] = (-2, 2),
    alpha: float = 0.25,
    color: str = "#6baed6",
    label: str | None = None,
) -> None:
    """
    Draw the plane passing through a point with the given normal.
    Uses z = (-d - ax - by) / c rearrangement of ax + by + cz + d = 0.
    """

    p0 = _ensure_array(point)
    n = _ensure_array(normal)
    if np.isclose(n[2], 0.0):
        raise ValueError("Plane normal's z component is zero; pick a different normal or rotate axes.")

    d = -np.dot(n, p0)
    xx, yy = np.meshgrid(
        np.linspace(*xlim, 10),
        np.linspace(*ylim, 10),
    )
    zz = (-d - n[0] * xx - n[1] * yy) / n[2]
    ax.plot_surface(xx, yy, zz, alpha=alpha, color=color, edgecolor="none", label=label)


def plot_line(
    ax: plt.Axes,
    point: Vector,
    direction: Vector,
    t_range: Tuple[float, float] = (-2, 2),
    *,
    color: str = "#31a354",
    label: str | None = None,
) -> None:
    """
    Draw parametric line p(t) = point + t * direction.
    """

    p0 = _ensure_array(point)
    d = _ensure_array(direction)
    t_vals = np.linspace(t_range[0], t_range[1], 50)
    pts = p0[:, None] + d[:, None] * t_vals
    ax.plot(pts[0], pts[1], pts[2], color=color, linewidth=2.0, label=label)


def plot_linear_transformation(
    ax: plt.Axes,
    basis: Basis,
    matrix: np.ndarray,
    *,
    origin: Vector = (0.0, 0.0, 0.0),
    colors: Sequence[str] = ("#e34a33", "#31a354", "#3182bd"),
) -> None:
    """
    Visualize how a 3x3 matrix moves the standard basis vectors.
    """

    mat = np.asarray(matrix, dtype=float)
    if mat.shape != (3, 3):
        raise ValueError("Matrix must be 3x3 for a 3D linear transformation")

    standard = np.eye(3)
    transformed = mat @ standard

    plot_vectors(ax, standard, origin=origin, colors=colors, labels=("e1", "e2", "e3"))
    plot_vectors(
        ax,
        transformed.T,  # columns are transformed basis vectors
        origin=origin,
        colors=colors,
        labels=("T(e1)", "T(e2)", "T(e3)"),
    )


def _add_axis_styling(ax: plt.Axes, title: str) -> None:
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.4)


def demo_vectors_and_plane() -> None:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    plot_vectors(
        ax,
        vectors=[(1, 1, 2), (-1, 2, 0.5), (0.5, -1, 1.5)],
        labels=["v1", "v2", "v3"],
    )
    plot_plane(ax, point=(0, 0, 1), normal=(0.5, -0.25, 1.0), label="plane")
    plot_line(ax, point=(0, 0, 0), direction=(1, -1, 1), label="line")

    _add_axis_styling(ax, "Vectors, plane, and line")
    plt.show()


def demo_linear_transformation() -> None:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    matrix = np.array(
        [
            [1.2, 0.2, 0.0],
            [0.0, 0.8, 0.3],
            [0.1, -0.2, 1.1],
        ]
    )
    plot_linear_transformation(ax, Basis(), matrix)

    _add_axis_styling(ax, "Linear transformation of basis")
    plt.show()


def demo_partial_derivatives_terrain() -> None:
    """
    Show how points with zero partial derivatives correspond to peaks (local maxima)
    and pits (local minima) on a smooth surface.

    The terrain is f(x, y) = x^4 + y^4 - 2(x^2 + y^2), whose partials are
    fx = 4x(x^2 - 1) and fy = 4y(y^2 - 1).
    Critical points (fx = 0 and fy = 0) occur at x, y in { -1, 0, 1 }.
    """

    def f(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return x**4 + y**4 - 2 * (x**2 + y**2)

    def grad(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return 4 * x * (x**2 - 1), 4 * y * (y**2 - 1)

    xs = np.linspace(-1.6, 1.6, 120)
    ys = np.linspace(-1.6, 1.6, 120)
    xx, yy = np.meshgrid(xs, ys)
    zz = f(xx, yy)

    # Critical points where both partial derivatives vanish.
    critical = []
    for cx in (-1.0, 0.0, 1.0):
        for cy in (-1.0, 0.0, 1.0):
            val = f(np.array(cx), np.array(cy)).item()
            if cx == 0.0 and cy == 0.0:
                kind = "peak (max)"
                color = "#e34a33"
            elif abs(cx) == 1.0 and abs(cy) == 1.0:
                kind = "pit (min)"
                color = "#3182bd"
            else:
                kind = "saddle"
                color = "#fdae6b"
            critical.append({"x": cx, "y": cy, "z": val, "kind": kind, "color": color})

    # Figure with a 3D surface and a 2D contour + gradient field.
    fig = plt.figure(figsize=(12, 5))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax2d = fig.add_subplot(1, 2, 2)

    ax3d.plot_surface(xx, yy, zz, cmap="viridis", alpha=0.8, edgecolor="none")
    for p in critical:
        ax3d.scatter(p["x"], p["y"], p["z"], color=p["color"], s=60, label=p["kind"])

    # Keep unique legend entries only.
    handles, labels = ax3d.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax3d.legend(by_label.values(), by_label.keys(), loc="upper left")
    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("z = f(x, y)")
    ax3d.set_title("Terrain with critical points (partials = 0)")
    ax3d.view_init(elev=25, azim=-60)

    # 2D contour with gradient (partial derivatives) arrows.
    ax2d.contourf(xx, yy, zz, levels=25, cmap="viridis")
    gx, gy = grad(xx, yy)
    step = 6  # thin quiver density for readability
    ax2d.quiver(xx[::step, ::step], yy[::step, ::step], gx[::step, ::step], gy[::step, ::step],
                color="white", pivot="mid", alpha=0.8, width=0.003)

    for p in critical:
        ax2d.scatter(p["x"], p["y"], color=p["color"], s=60, edgecolor="black")
        ax2d.text(p["x"] + 0.05, p["y"] + 0.05, p["kind"], color="white", fontsize=9,
                  bbox={"facecolor": "black", "alpha": 0.5, "pad": 2})

    ax2d.set_xlabel("x")
    ax2d.set_ylabel("y")
    ax2d.set_title("Gradient field: arrows show partial derivatives")
    ax2d.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Change which demo runs here
    demo_partial_derivatives_terrain()
    # demo_vectors_and_plane()
    # demo_linear_transformation()
