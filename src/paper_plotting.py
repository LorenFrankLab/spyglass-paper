"""
Shared plotting utilities for Spyglass paper figures.

This module provides publication-quality figure defaults, saving utilities,
and common visualization functions used across all figure scripts.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    import networkx as nx

# =============================================================================
# FIGURE DIMENSION CONSTANTS
# =============================================================================
MM_TO_INCHES = 1.0 / 25.4

ONE_COLUMN = 89.0 * MM_TO_INCHES
ONE_AND_HALF_COLUMN = 140.0 * MM_TO_INCHES
TWO_COLUMN = 178.0 * MM_TO_INCHES
PAGE_HEIGHT = 247.0 * MM_TO_INCHES
GOLDEN_RATIO = (np.sqrt(5) - 1.0) / 2.0


# =============================================================================
# FIGURE SETUP AND SAVING
# =============================================================================
def set_figure_defaults() -> None:
    """
    Set matplotlib defaults for publication-quality figures.

    Configures seaborn and matplotlib rcParams for consistent styling
    across all figure panels, including font sizes, tick sizes, and
    color settings suitable for journal publication.
    """
    rc_params = {
        "pdf.fonttype": 42,  # Make fonts editable in Adobe Illustrator
        "ps.fonttype": 42,  # Make fonts editable in Adobe Illustrator
        "axes.labelcolor": "#222222",
        "axes.labelsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "text.color": "#222222",
        "text.usetex": False,
        "figure.figsize": (7.2, 4.45),
        "xtick.major.size": 2,
        "xtick.bottom": True,
        "ytick.left": True,
        "ytick.major.size": 2,
        "axes.labelpad": 0.1,
    }
    sns.set(style="white", context="paper", rc=rc_params, font_scale=1.4)


def save_figure(
    figure_name: str,
    output_dir: str | Path | None = None,
    transparent: bool = True,
    facecolor: str | None = None,
) -> None:
    """
    Save the current matplotlib figure as both PDF and PNG.

    Parameters
    ----------
    figure_name : str
        Base name for the output files (without extension).
    output_dir : str or Path, optional
        Directory where figures will be saved. Created if it doesn't exist.
        Defaults to current working directory.
    transparent : bool, optional
        Whether to save with transparent background, by default True.
    facecolor : str, optional
        Background color for the figure. Only used if transparent is False.
    """
    if output_dir is None:
        output_dir = Path(".")
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = output_dir / f"{figure_name}.pdf"
    png_path = output_dir / f"{figure_name}.png"

    save_kwargs = {
        "dpi": 300,
        "bbox_inches": "tight",
        "transparent": transparent,
    }
    if facecolor is not None:
        save_kwargs["facecolor"] = facecolor

    plt.savefig(pdf_path, **save_kwargs)
    plt.savefig(png_path, **save_kwargs)
    print(f"Saved: {pdf_path} and {png_path}")


# =============================================================================
# PLOTTING HELPERS
# =============================================================================
def add_scalebar(
    ax: Axes,
    length: float,
    label: str,
    position: tuple[float, float] = (0.05, 0.05),
    linewidth: int = 3,
    color: str = "black",
    fontsize: int = 11,
    text_offset: float = -5.0,
) -> None:
    """
    Add a scale bar to a matplotlib Axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to add the scale bar to.
    length : float
        Length of the scale bar in data coordinates (typically cm for position
        data in this module).
    label : str
        Text label for the scale bar (e.g., "20 cm").
    position : tuple of float, optional
        Position of the scale bar in axes coordinates (0-1), by default (0.05, 0.05).
    linewidth : int, optional
        Width of the scale bar line in points, by default 3.
    color : str, optional
        Color of the scale bar and label, by default "black".
    fontsize : int, optional
        Font size for the label in points, by default 11.
    text_offset : float, optional
        Vertical offset for the label in data coordinates (typically cm),
        by default -5.0.
    """
    trans = ax.transAxes + ax.transData.inverted()
    x, y = trans.transform(position)

    bar = mpatches.Rectangle(
        (x, y),
        length,
        0,
        linewidth=linewidth,
        color=color,
        transform=ax.transData,
    )
    ax.add_patch(bar)

    ax.text(
        x + length / 2,
        y + text_offset,
        label,
        ha="center",
        va="top",
        color=color,
        fontsize=fontsize,
        transform=ax.transData,
    )


def plot_graph_as_1D(
    track_graph: nx.Graph,
    ax: Axes | None = None,
    edge_order: list[tuple[int, int]] | None = None,
    edge_spacing: float | list[float] = 0.0,
    reward_well_nodes: list[int] | None = None,
    other_axis_start: float = 0,
    edge_colors: np.ndarray | None = None,
    reward_well_size: int = 10,
    edge_linewidth: int = 2,
) -> None:
    """
    Plot track graph as 1D linearized representation.

    Draws the track graph edges as vertical line segments positioned
    sequentially to show the linearized track structure, with optional
    markers for reward well locations.

    Parameters
    ----------
    track_graph : networkx.Graph
        Track graph with edges containing 'distance' attributes (in cm).
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, uses current axes.
    edge_order : list of tuple of int, optional
        Order of edges for linearization. Each tuple is (node1, node2).
        If None, uses graph's natural edge order.
    edge_spacing : float or list of float, optional
        Spacing between edges in cm. If float, same spacing for all.
        If list, spacing after each edge (length n_edges - 1).
        By default 0.0.
    reward_well_nodes : list of int, optional
        Node indices that are reward wells (marked with scatter points).
    other_axis_start : float, optional
        X-position for the 1D representation in data coordinates (typically
        time in seconds), by default 0.
    edge_colors : ndarray, optional
        Array of RGB colors for each edge. If None, uses tab10 colormap.
    reward_well_size : int, optional
        Marker size for reward well points in points^2, by default 10.
    edge_linewidth : int, optional
        Line width for edge segments in points, by default 2.
    """
    if ax is None:
        ax = plt.gca()
    if edge_order is None:
        edge_order = list(track_graph.edges)
    if reward_well_nodes is None:
        reward_well_nodes = []
    if edge_colors is None:
        edge_colors = np.array(matplotlib.colormaps.get_cmap("tab10").colors)

    n_edges = len(edge_order)
    if isinstance(edge_spacing, (int, float)):
        edge_spacing = [edge_spacing] * (n_edges - 1)

    start_node_linear_position = 0.0

    for edge_ind, edge in enumerate(edge_order):
        end_node_linear_position = (
            start_node_linear_position + track_graph.edges[edge]["distance"]
        )
        ax.plot(
            (other_axis_start, other_axis_start),
            (start_node_linear_position, end_node_linear_position),
            color=edge_colors[edge_ind],
            clip_on=False,
            zorder=7,
            linewidth=edge_linewidth,
        )
        if edge[0] in reward_well_nodes:
            ax.scatter(
                other_axis_start,
                start_node_linear_position,
                color=edge_colors[edge_ind],
                s=reward_well_size,
                zorder=10,
                clip_on=False,
            )
        if edge[1] in reward_well_nodes:
            ax.scatter(
                other_axis_start,
                end_node_linear_position,
                color=edge_colors[edge_ind],
                s=reward_well_size,
                zorder=10,
                clip_on=False,
            )

        # Update position for next edge (skip spacing on last edge)
        if edge_ind < len(edge_spacing):
            start_node_linear_position += (
                track_graph.edges[edge]["distance"] + edge_spacing[edge_ind]
            )
        else:
            start_node_linear_position += track_graph.edges[edge]["distance"]


def plot_2D_track_graph(
    track_graph: nx.Graph,
    position_info,
    edge_order: list[tuple[int, int]] | None = None,
    reward_well_nodes: list[int] | None = None,
    edge_colors: np.ndarray | None = None,
    figsize: tuple[float, float] | None = None,
    position_names: tuple[str, str] = ("position_x", "position_y"),
    scalebar_length: float = 20,
    scalebar_label: str = "20 cm",
) -> tuple[plt.Figure, Axes]:
    """
    Plot 2D track graph with position trajectory overlay.

    Parameters
    ----------
    track_graph : networkx.Graph
        Track graph with nodes containing 'pos' attributes.
    position_info : pandas.DataFrame
        DataFrame containing position columns for trajectory overlay.
    edge_order : list of tuple of int, optional
        Order of edges. If None, uses graph's natural edge order.
    reward_well_nodes : list of int, optional
        Node indices that are reward wells (marked with scatter points).
    edge_colors : ndarray, optional
        Array of colors for each edge. If None, uses tab10 colormap.
    figsize : tuple of float, optional
        Figure size. If None, uses default based on TWO_COLUMN.
    position_names : tuple of str, optional
        Column names for (x, y) position in position_info.
    scalebar_length : float, optional
        Length of scale bar in data units, by default 20.
    scalebar_label : str, optional
        Label for scale bar, by default "20 cm".

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes
        The axes object.
    """
    if reward_well_nodes is None:
        reward_well_nodes = []
    if edge_colors is None:
        edge_colors = np.array(matplotlib.colormaps.get_cmap("tab10").colors)
    if edge_order is None:
        edge_order = list(track_graph.edges)
    if figsize is None:
        figsize = (TWO_COLUMN / 3 * 0.6, TWO_COLUMN / 3 * 0.6)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    ax.plot(
        position_info[position_names[0]],
        position_info[position_names[1]],
        color="lightgrey",
        alpha=0.7,
    )

    for edge_color, (node1, node2) in zip(edge_colors, edge_order):
        node1_pos = track_graph.nodes[node1]["pos"]
        node2_pos = track_graph.nodes[node2]["pos"]
        ax.plot(
            [node1_pos[0], node2_pos[0]],
            [node1_pos[1], node2_pos[1]],
            linewidth=2,
            color=edge_color,
        )
        if node1 in reward_well_nodes:
            ax.scatter(
                node1_pos[0],
                node1_pos[1],
                color=edge_color,
                s=30,
                zorder=10,
            )
        if node2 in reward_well_nodes:
            ax.scatter(
                node2_pos[0],
                node2_pos[1],
                color=edge_color,
                s=30,
                zorder=10,
            )

    add_scalebar(ax, scalebar_length, scalebar_label)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    return fig, ax
