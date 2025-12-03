"""
Supplemental Figure: Spyglass Neural Decoding Workflow

Generates a publication-ready flowchart showing the two decoding pipelines
used in Figure 5: clusterless decoding (Frank Lab) and sorted spikes decoding
(Buzsaki Lab).

Design follows paper style specifications from FIGURE5_SUPP.md.
"""

import os
import sys

# Add skill scripts to path
skill_paths = [
    os.path.expanduser("~/.claude/skills/scientific-figures-paper/scripts"),
    "/mnt/skills/user/scientific-figures-paper/scripts",
    "/mnt/skills/scientific-figures-paper/scripts",
]

for path in skill_paths:
    if os.path.exists(path) and path not in sys.path:
        sys.path.insert(0, path)
        break

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Try to import from skill, fallback to inline definitions
try:
    from figure_utilities import set_figure_defaults, save_figure
except ImportError:
    import matplotlib as mpl

    def set_figure_defaults(context="paper"):
        rc_params = {
            "font.family": "sans-serif",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.labelcolor": "#222222",
            "text.color": "#222222",
            "figure.facecolor": "white",
            "axes.linewidth": 0.8,
            "savefig.dpi": 300,
            "savefig.transparent": False,
        }
        mpl.rcParams.update(rc_params)

    def save_figure(figure_name, output_dir=None):
        if output_dir is None:
            output_dir = "."
        os.makedirs(output_dir, exist_ok=True)
        pdf_path = os.path.join(output_dir, f"{figure_name}.pdf")
        png_path = os.path.join(output_dir, f"{figure_name}.png")
        plt.savefig(pdf_path, dpi=300, bbox_inches="tight", transparent=False)
        plt.savefig(png_path, dpi=300, bbox_inches="tight", transparent=False)
        print(f"Saved: {pdf_path} and {png_path}")


# =============================================================================
# COLOR PALETTE (from FIGURE5_SUPP.md design specifications)
# =============================================================================
COLORS = {
    "position": "#1f77b4",  # Blue - Position stream
    "neural_frank": "#ff7f0e",  # Orange - Neural stream (Frank Lab)
    "neural_buzsaki": "#2ca02c",  # Green - Neural stream (Buzsaki Lab)
    "decoding": "#d62728",  # Red - Decoding/convergence
    "merge": "#9467bd",  # Purple - Merge tables
    "text": "#222222",  # Dark gray - Text/labels
    "background": "#ffffff",  # White - Background
    "arrow": "#7f7f7f",  # Gray - Arrows
    "nwb": "#e0e0e0",  # Light gray - NWB file boxes
}

# Box fill opacity
FILL_ALPHA = 0.2


# =============================================================================
# BOX GEOMETRY HELPERS
# =============================================================================
# Constants for box rendering
BOX_PAD = 0.02  # FancyBboxPatch pad extends visual box by this amount on each side
MERGE_BOX_OUTER_OFFSET = 0.015  # Extra offset for merge table outer border
ARROW_GAP = 0.004  # Small gap between arrow endpoint and box edge


def get_box_bounds(y_center, height, is_merge=False):
    """
    Calculate the visual top and bottom bounds of a box for arrow attachment.

    The FancyBboxPatch 'pad' parameter extends the visual box outward from
    the geometric bounds. Arrows should attach at the visual edge.

    Parameters
    ----------
    y_center : float
        Y coordinate of box center
    height : float
        Box height (geometric, before padding)
    is_merge : bool
        If True, account for merge table's outer border offset

    Returns
    -------
    dict with keys 'top' and 'bottom' representing visual bounds for arrow attachment
    """
    # Geometric bounds
    geo_top = y_center + height / 2
    geo_bottom = y_center - height / 2

    # Visual bounds include the BOX_PAD from FancyBboxPatch
    visual_top = geo_top + BOX_PAD
    visual_bottom = geo_bottom - BOX_PAD

    # Merge tables have an additional outer border
    if is_merge:
        visual_top += MERGE_BOX_OUTER_OFFSET
        visual_bottom -= MERGE_BOX_OUTER_OFFSET

    return {"top": visual_top, "bottom": visual_bottom}


def draw_vertical_arrow_between_boxes(
    ax,
    x: float,
    src_bounds: dict,
    dst_bounds: dict,
    color: str | None = None,
) -> None:
    """
    Draw a vertical arrow between two boxes, leaving a small visual gap
    from each box edge. Arrow points toward destination (downward flow).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    x : float
        X coordinate of the arrow.
    src_bounds : dict
        Bounds dict for the source box from get_box_bounds.
    dst_bounds : dict
        Bounds dict for the destination box from get_box_bounds.
    color : str, optional
        Arrow color. Defaults to COLORS["arrow"].
    """
    if color is None:
        color = COLORS["arrow"]

    # Leave ARROW_GAP below the source and above the destination
    # This creates symmetric spacing at both ends of the arrow
    start_y = src_bounds["bottom"] - ARROW_GAP
    end_y = dst_bounds["top"] + ARROW_GAP

    arrow = FancyArrowPatch(
        (x, start_y),
        (x, end_y),
        arrowstyle="-|>",
        mutation_scale=8,
        color=color,
        linewidth=1,
        shrinkA=0,
        shrinkB=0,
    )
    ax.add_patch(arrow)


# =============================================================================
# DRAWING HELPER FUNCTIONS
# =============================================================================
def draw_box(
    ax,
    x,
    y,
    width,
    height,
    text,
    color,
    is_merge=False,
    is_dashed=False,
    fontsize=7,
    ref_num=None,
):
    """
    Draw a styled box with optional merge table double-line or dashed border.

    Parameters
    ----------
    ax : matplotlib axes
    x, y : float
        Center coordinates of the box
    width, height : float
        Box dimensions
    text : str
        Label text
    color : str
        Box color (border and text)
    is_merge : bool
        If True, draw double-line border (merge table style)
    is_dashed : bool
        If True, draw dashed border
    fontsize : int
        Font size for label text
    ref_num : str or None
        Reference number to display (e.g., "[1]")
    """
    # Calculate corner position
    x0 = x - width / 2
    y0 = y - height / 2

    if is_dashed:
        # Dashed border (no fill)
        rect = FancyBboxPatch(
            (x0, y0),
            width,
            height,
            boxstyle=f"round,pad={BOX_PAD},rounding_size=0.02",
            facecolor="none",
            edgecolor=color,
            linewidth=1,
            linestyle="--",
        )
        ax.add_patch(rect)
    elif is_merge:
        # Merge table: double-line border effect
        # Outer rectangle
        rect_outer = FancyBboxPatch(
            (x0 - MERGE_BOX_OUTER_OFFSET, y0 - MERGE_BOX_OUTER_OFFSET),
            width + 2 * MERGE_BOX_OUTER_OFFSET,
            height + 2 * MERGE_BOX_OUTER_OFFSET,
            boxstyle=f"round,pad={BOX_PAD},rounding_size=0.02",
            facecolor="none",
            edgecolor=color,
            linewidth=1,
        )
        ax.add_patch(rect_outer)
        # Inner rectangle with fill
        rect_inner = FancyBboxPatch(
            (x0, y0),
            width,
            height,
            boxstyle=f"round,pad={BOX_PAD},rounding_size=0.02",
            facecolor=(*plt.cm.colors.hex2color(color), FILL_ALPHA),
            edgecolor=color,
            linewidth=1,
        )
        ax.add_patch(rect_inner)
    else:
        # Standard box
        rect = FancyBboxPatch(
            (x0, y0),
            width,
            height,
            boxstyle=f"round,pad={BOX_PAD},rounding_size=0.02",
            facecolor=(*plt.cm.colors.hex2color(color), FILL_ALPHA),
            edgecolor=color,
            linewidth=1,
        )
        ax.add_patch(rect)

    # Add text label
    # Offset text up slightly if there's a reference number, proportional to box height
    text_offset = height * 0.20 if ref_num else 0
    ax.text(
        x,
        y + text_offset,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=COLORS["text"],
        fontweight="normal",
        wrap=True,
    )

    # Add reference number below main text
    if ref_num:
        ref_offset = height * 0.55
        ax.text(
            x,
            y - ref_offset,
            ref_num,
            ha="center",
            va="center",
            fontsize=fontsize - 1,
            color=color,
            fontweight="bold",
        )


def draw_arrow(ax, start, end, color=None):
    """
    Draw an arrow from start to end position.

    Parameters
    ----------
    ax : matplotlib axes
    start : tuple (x, y)
    end : tuple (x, y)
    color : str, optional
    """
    if color is None:
        color = COLORS["arrow"]

    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=8,
        color=color,
        linewidth=1,
        shrinkA=0,
        shrinkB=0,  # Arrow tip reaches exactly the endpoint
    )
    ax.add_patch(arrow)


def draw_vertical_arrow(ax, x, y_start, y_end, color=None):
    """
    Draw a vertical arrow from y_start to y_end at x position.

    The arrow head points in the direction of the flow (toward y_end).

    Parameters
    ----------
    ax : matplotlib axes
    x : float
        X coordinate for the vertical arrow
    y_start : float
        Y coordinate where arrow begins (bottom of source box)
    y_end : float
        Y coordinate where arrow ends (top of target box)
    color : str, optional
    """
    if color is None:
        color = COLORS["arrow"]

    draw_arrow(ax, (x, y_start), (x, y_end), color)


def draw_split_arrow(ax, start, ends, color=None):
    """
    Draw arrows splitting from one point to multiple endpoints.

    Parameters
    ----------
    ax : matplotlib axes
    start : tuple (x, y)
    ends : list of tuples [(x1, y1), (x2, y2), ...]
    color : str, optional
    """
    if color is None:
        color = COLORS["arrow"]

    # Draw vertical line from start to split point
    mid_y = (start[1] + max(e[1] for e in ends)) / 2
    ax.plot(
        [start[0], start[0]], [start[1], mid_y], color=color, linewidth=1, zorder=1
    )

    # Draw horizontal line to span endpoints
    min_x = min(e[0] for e in ends)
    max_x = max(e[0] for e in ends)
    ax.plot([min_x, max_x], [mid_y, mid_y], color=color, linewidth=1, zorder=1)

    # Draw arrows to each endpoint
    for end in ends:
        ax.plot([end[0], end[0]], [mid_y, end[1] + 0.05], color=color, linewidth=1)
        draw_arrow(ax, (end[0], end[1] + 0.05), end, color)


def draw_merge_arrow(ax, starts, end, color=None):
    """
    Draw arrows merging from multiple points to one endpoint.

    Parameters
    ----------
    ax : matplotlib axes
    starts : list of tuples [(x1, y1), (x2, y2), ...]
    end : tuple (x, y)
    color : str, optional
    """
    if color is None:
        color = COLORS["arrow"]

    # Calculate merge point
    mid_y = (min(s[1] for s in starts) + end[1]) / 2

    # Draw from each start to horizontal line
    for start in starts:
        ax.plot(
            [start[0], start[0]], [start[1], mid_y], color=color, linewidth=1, zorder=1
        )

    # Draw horizontal line
    min_x = min(s[0] for s in starts)
    max_x = max(s[0] for s in starts)
    ax.plot([min_x, max_x], [mid_y, mid_y], color=color, linewidth=1, zorder=1)

    # Draw to endpoint
    mid_x = (min_x + max_x) / 2
    ax.plot([mid_x, mid_x], [mid_y, end[1] + 0.05], color=color, linewidth=1, zorder=1)
    draw_arrow(ax, (mid_x, end[1] + 0.05), end, color)


# =============================================================================
# MAIN FIGURE CREATION
# =============================================================================
def create_figure():
    """Create the supplemental figure showing decoding workflows."""

    set_figure_defaults(context="paper")

    # Figure dimensions (from design spec)
    fig_width = 7.0  # Two-column width
    fig_height = 8.5  # Increased to accommodate table reference

    fig, axes = plt.subplot_mosaic(
        [["A", "B"]],
        figsize=(fig_width, fig_height),
        width_ratios=[1, 1],
        dpi=300,
        constrained_layout=True,
    )

    # Configure both panels
    for ax_name in ["A", "B"]:
        ax = axes[ax_name]
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        # Don't set aspect="equal" - let the flowchart fill available space
        ax.axis("off")

    # ==========================================================================
    # SHARED BOX DIMENSIONS AND POSITIONS
    # ==========================================================================
    # Box dimensions
    bw = 0.35  # Box width (large)
    bw_small = 0.30  # Box width (small)

    # Calculate layout based on available space
    # We have 8 rows: nwb, input, output1, features, group, decoding, decode_out, figure
    n_rows = 8
    y_top = 0.86  # Center of first box (leave room for title above)
    y_bottom = 0.22  # Center of last box (leave room for legend + table reference below)
    total_height = y_top - y_bottom

    # We want: row_spacing = box_height + gap_between_boxes
    # With n_rows boxes: total_height = (n_rows - 1) * row_spacing
    # So: row_spacing = total_height / (n_rows - 1)
    row_spacing = total_height / (n_rows - 1)

    # Set box height to leave enough gap for arrows
    # Visual box height = bh + 2*BOX_PAD (due to FancyBboxPatch padding)
    # Use 30% of row_spacing: more vertical space between boxes for arrows
    bh = row_spacing * 0.30
    bh_figure = bh * 0.6  # Figure output box slightly smaller than others

    # Y positions (top to bottom) - these are box CENTER positions
    y_nwb = y_top - 0 * row_spacing
    y_input = y_top - 1 * row_spacing
    y_output1 = y_top - 2 * row_spacing
    y_features = y_top - 3 * row_spacing
    y_group = y_top - 4 * row_spacing
    y_decoding = y_top - 5 * row_spacing
    y_decode_out = y_top - 6 * row_spacing
    y_figure = y_top - 7 * row_spacing

    # X positions
    x_left = 0.30
    x_right = 0.70
    x_center = 0.50

    # ==========================================================================
    # PRECOMPUTE BOX BOUNDS FOR ALL ROWS
    # ==========================================================================
    # Each box type has its bounds calculated once
    bounds_nwb = get_box_bounds(y_nwb, bh, is_merge=False)
    bounds_input = get_box_bounds(y_input, bh, is_merge=False)
    bounds_output1 = get_box_bounds(y_output1, bh, is_merge=True)
    bounds_features = get_box_bounds(y_features, bh, is_merge=False)
    bounds_group = get_box_bounds(y_group, bh, is_merge=False)
    bounds_decoding = get_box_bounds(y_decoding, bh, is_merge=False)
    bounds_decode_out = get_box_bounds(y_decode_out, bh, is_merge=True)
    bounds_figure = get_box_bounds(y_figure, bh_figure, is_merge=False)

    # ==========================================================================
    # PANEL A: Frank Lab Dataset (Clusterless Decoding)
    # ==========================================================================
    ax = axes["A"]

    # Panel title
    ax.text(
        0.5,
        0.98,
        "UCSF Dataset",
        ha="center",
        va="top",
        fontsize=9,
        fontweight="bold",
        color=COLORS["text"],
    )
    ax.text(
        0.5,
        0.94,
        "(Clusterless Decoding)",
        ha="center",
        va="top",
        fontsize=7,
        color=COLORS["text"],
    )

    # --- NWB File ---
    draw_box(ax, x_center, y_nwb, bw, bh, "NWB File\nDANDI:000937", COLORS["nwb"])

    # Split arrow from NWB to Position and Neural
    # Line from NWB bottom down to a split point, then horizontal, then down to inputs
    # Position split_y between NWB bottom and input top (respecting ARROW_GAP)
    gap_between = bounds_nwb["bottom"] - bounds_input["top"]
    split_y = bounds_nwb["bottom"] - gap_between / 3  # Split point 1/3 down from NWB
    ax.plot(
        [x_center, x_center],
        [bounds_nwb["bottom"] - ARROW_GAP, split_y],
        color=COLORS["arrow"],
        linewidth=1,
    )
    # Horizontal line connecting left and right branches
    ax.plot(
        [x_left, x_right],
        [split_y, split_y],
        color=COLORS["arrow"],
        linewidth=1,
    )
    # Arrows down to input boxes (with gap above destination)
    end_y = bounds_input["top"] + ARROW_GAP
    draw_arrow(ax, (x_left, split_y), (x_left, end_y))
    draw_arrow(ax, (x_right, split_y), (x_right, end_y))

    # --- Input layer ---
    draw_box(
        ax,
        x_left,
        y_input,
        bw_small,
        bh,
        "Raw\nPosition",
        COLORS["position"],
        ref_num="[1]",
    )
    draw_box(
        ax,
        x_right,
        y_input,
        bw_small,
        bh,
        "Raw\n(ephys)",
        COLORS["neural_frank"],
        ref_num="[2]",
    )

    # Arrows to output layer
    draw_vertical_arrow_between_boxes(ax, x_left, bounds_input, bounds_output1)
    draw_vertical_arrow_between_boxes(ax, x_right, bounds_input, bounds_output1)

    # --- Output merge tables ---
    draw_box(
        ax,
        x_left,
        y_output1,
        bw_small,
        bh,
        "Position\nOutput",
        COLORS["merge"],
        is_merge=True,
        ref_num="[4]",
    )
    draw_box(
        ax,
        x_right,
        y_output1,
        bw_small,
        bh,
        "Spike Sorting\nOutput",
        COLORS["merge"],
        is_merge=True,
        ref_num="[5]",
    )

    # Arrow from Spike Sorting Output to Waveform Features
    draw_vertical_arrow_between_boxes(ax, x_right, bounds_output1, bounds_features)

    # --- Waveform features ---
    draw_box(
        ax,
        x_right,
        y_features,
        bw_small,
        bh,
        "Waveform\nFeatures",
        COLORS["neural_frank"],
        ref_num="[6]",
    )

    # Arrow from Position Output to Position Group (skips features row)
    draw_vertical_arrow_between_boxes(ax, x_left, bounds_output1, bounds_group)
    # Arrow from Waveform Features to Waveform Features Group
    draw_vertical_arrow_between_boxes(ax, x_right, bounds_features, bounds_group)

    # --- Grouping ---
    draw_box(
        ax,
        x_left,
        y_group,
        bw_small,
        bh,
        "Position\nGroup",
        COLORS["position"],
        ref_num="[8]",
    )
    draw_box(
        ax,
        x_right,
        y_group,
        bw_small,
        bh,
        "Waveform\nFeatures Group",
        COLORS["neural_frank"],
        ref_num="[9]",
    )

    # Merge arrows to decoding (both streams converge)
    # Position merge_y between group bottom and decoding top (respecting ARROW_GAP)
    gap_to_decoding = bounds_group["bottom"] - bounds_decoding["top"]
    merge_y = bounds_group["bottom"] - gap_to_decoding / 3  # Merge point 1/3 down
    ax.plot(
        [x_left, x_left],
        [bounds_group["bottom"] - ARROW_GAP, merge_y],
        color=COLORS["arrow"],
        linewidth=1,
    )
    ax.plot(
        [x_right, x_right],
        [bounds_group["bottom"] - ARROW_GAP, merge_y],
        color=COLORS["arrow"],
        linewidth=1,
    )
    ax.plot(
        [x_left, x_right],
        [merge_y, merge_y],
        color=COLORS["arrow"],
        linewidth=1,
    )
    # Arrow down to decoding box (with gap above destination)
    merge_end_y = bounds_decoding["top"] + ARROW_GAP
    draw_arrow(ax, (x_center, merge_y), (x_center, merge_end_y))

    # --- Decoding ---
    draw_box(
        ax,
        x_center,
        y_decoding,
        bw,
        bh,
        "Clusterless\nDecoding",
        COLORS["decoding"],
        ref_num="[10]",
    )

    # Arrow to decoding output
    draw_vertical_arrow_between_boxes(ax, x_center, bounds_decoding, bounds_decode_out)

    # --- Decoding output ---
    draw_box(
        ax,
        x_center,
        y_decode_out,
        bw,
        bh,
        "Decoding\nOutput",
        COLORS["merge"],
        is_merge=True,
        ref_num="[12]",
    )

    # Arrow to figure
    draw_vertical_arrow_between_boxes(ax, x_center, bounds_decode_out, bounds_figure)

    # --- Figure output ---
    draw_box(
        ax,
        x_center,
        y_figure,
        bw_small,
        bh_figure,
        "Figure 5A",
        COLORS["arrow"],
        is_dashed=True,
        fontsize=7,
    )

    # Panel label
    ax.text(
        0.02,
        0.98,
        "A",
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
    )

    # ==========================================================================
    # PANEL B: Buzsaki Lab Dataset (Sorted Spikes Decoding)
    # ==========================================================================
    ax = axes["B"]

    # Panel title
    ax.text(
        0.5,
        0.98,
        "NYU Dataset",
        ha="center",
        va="top",
        fontsize=9,
        fontweight="bold",
        color=COLORS["text"],
    )
    ax.text(
        0.5,
        0.94,
        "(Sorted Spikes Decoding)",
        ha="center",
        va="top",
        fontsize=7,
        color=COLORS["text"],
    )

    # --- NWB File ---
    draw_box(ax, x_center, y_nwb, bw, bh, "NWB File\nDANDI:000059", COLORS["nwb"])

    # Split arrow from NWB (reuse split_y calculated in Panel A for consistency)
    ax.plot(
        [x_center, x_center],
        [bounds_nwb["bottom"] - ARROW_GAP, split_y],
        color=COLORS["arrow"],
        linewidth=1,
    )
    ax.plot(
        [x_left, x_right],
        [split_y, split_y],
        color=COLORS["arrow"],
        linewidth=1,
    )
    # Arrows down to input boxes (with gap above destination)
    draw_arrow(ax, (x_left, split_y), (x_left, end_y))
    draw_arrow(ax, (x_right, split_y), (x_right, end_y))

    # --- Input layer ---
    draw_box(
        ax,
        x_left,
        y_input,
        bw_small,
        bh,
        "Raw\nPosition",
        COLORS["position"],
        ref_num="[1]",
    )
    draw_box(
        ax,
        x_right,
        y_input,
        bw_small,
        bh,
        "Imported\nSpike Sorting",
        COLORS["neural_buzsaki"],
        ref_num="[3]",
    )

    # Arrows to output layer
    draw_vertical_arrow_between_boxes(ax, x_left, bounds_input, bounds_output1)
    draw_vertical_arrow_between_boxes(ax, x_right, bounds_input, bounds_output1)

    # --- Output merge tables ---
    draw_box(
        ax,
        x_left,
        y_output1,
        bw_small,
        bh,
        "Position\nOutput",
        COLORS["merge"],
        is_merge=True,
        ref_num="[4]",
    )
    draw_box(
        ax,
        x_right,
        y_output1,
        bw_small,
        bh,
        "Spike Sorting\nOutput",
        COLORS["merge"],
        is_merge=True,
        ref_num="[5]",
    )

    # Arrow from Spike Sorting Output to Sorted Spikes Group
    draw_vertical_arrow_between_boxes(ax, x_right, bounds_output1, bounds_features)

    # --- Sorted spikes group ---
    draw_box(
        ax,
        x_right,
        y_features,
        bw_small,
        bh,
        "Sorted Spikes\nGroup",
        COLORS["neural_buzsaki"],
        ref_num="[7]",
    )

    # Arrow from Position Output to Position Group (skips features row)
    draw_vertical_arrow_between_boxes(ax, x_left, bounds_output1, bounds_group)

    # --- Position grouping ---
    draw_box(
        ax,
        x_left,
        y_group,
        bw_small,
        bh,
        "Position\nGroup",
        COLORS["position"],
        ref_num="[8]",
    )

    # Merge arrows to decoding (position group and sorted spikes group converge)
    # In Panel B, the merge comes from group (left) and features (right)
    # Position merge_y_b between the lower of the two sources and decoding top
    gap_to_decoding_b = bounds_group["bottom"] - bounds_decoding["top"]
    merge_y_b = bounds_group["bottom"] - gap_to_decoding_b / 3
    # Position Group vertical line down (with gap below source)
    ax.plot(
        [x_left, x_left],
        [bounds_group["bottom"] - ARROW_GAP, merge_y_b],
        color=COLORS["arrow"],
        linewidth=1,
    )
    # Sorted Spikes Group vertical line down (from features row, with gap below source)
    ax.plot(
        [x_right, x_right],
        [bounds_features["bottom"] - ARROW_GAP, merge_y_b],
        color=COLORS["arrow"],
        linewidth=1,
    )
    ax.plot(
        [x_left, x_right],
        [merge_y_b, merge_y_b],
        color=COLORS["arrow"],
        linewidth=1,
    )
    # Arrow down to decoding box (with gap above destination)
    merge_end_y_b = bounds_decoding["top"] + ARROW_GAP
    draw_arrow(ax, (x_center, merge_y_b), (x_center, merge_end_y_b))

    # --- Decoding ---
    draw_box(
        ax,
        x_center,
        y_decoding,
        bw,
        bh,
        "Sorted Spikes\nDecoding",
        COLORS["decoding"],
        ref_num="[11]",
    )

    # Arrow to decoding output
    draw_vertical_arrow_between_boxes(ax, x_center, bounds_decoding, bounds_decode_out)

    # --- Decoding output ---
    draw_box(
        ax,
        x_center,
        y_decode_out,
        bw,
        bh,
        "Decoding\nOutput",
        COLORS["merge"],
        is_merge=True,
        ref_num="[12]",
    )

    # Arrow to figure
    draw_vertical_arrow_between_boxes(ax, x_center, bounds_decode_out, bounds_figure)

    # --- Figure output ---
    draw_box(
        ax,
        x_center,
        y_figure,
        bw_small,
        bh_figure,
        "Figure 5B-D",
        COLORS["arrow"],
        is_dashed=True,
        fontsize=7,
    )

    # Panel label
    ax.text(
        0.02,
        0.98,
        "B",
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
    )

    # ==========================================================================
    # LEGEND
    # ==========================================================================
    # Add legend at bottom of figure
    legend_y = -0.02

    legend_elements = [
        (
            FancyBboxPatch(
                (0, 0),
                0.1,
                0.03,
                boxstyle="round,pad=0.01",
                facecolor=(*plt.cm.colors.hex2color(COLORS["position"]), FILL_ALPHA),
                edgecolor=COLORS["position"],
                linewidth=1,
            ),
            "Position processing",
        ),
        (
            FancyBboxPatch(
                (0, 0),
                0.1,
                0.03,
                boxstyle="round,pad=0.01",
                facecolor=(*plt.cm.colors.hex2color(COLORS["neural_frank"]), FILL_ALPHA),
                edgecolor=COLORS["neural_frank"],
                linewidth=1,
            ),
            "Neural (clusterless)",
        ),
        (
            FancyBboxPatch(
                (0, 0),
                0.1,
                0.03,
                boxstyle="round,pad=0.01",
                facecolor=(
                    *plt.cm.colors.hex2color(COLORS["neural_buzsaki"]),
                    FILL_ALPHA,
                ),
                edgecolor=COLORS["neural_buzsaki"],
                linewidth=1,
            ),
            "Neural (sorted)",
        ),
        (
            FancyBboxPatch(
                (0, 0),
                0.1,
                0.03,
                boxstyle="round,pad=0.01",
                facecolor=(*plt.cm.colors.hex2color(COLORS["decoding"]), FILL_ALPHA),
                edgecolor=COLORS["decoding"],
                linewidth=1,
            ),
            "Decoding",
        ),
        (
            FancyBboxPatch(
                (0, 0),
                0.1,
                0.03,
                boxstyle="round,pad=0.01",
                facecolor=(*plt.cm.colors.hex2color(COLORS["merge"]), FILL_ALPHA),
                edgecolor=COLORS["merge"],
                linewidth=1,
            ),
            "Merge table",
        ),
    ]

    # Create legend
    fig.legend(
        handles=[e[0] for e in legend_elements],
        labels=[e[1] for e in legend_elements],
        loc="lower center",
        ncol=5,
        fontsize=6,
        frameon=False,
        bbox_to_anchor=(0.5, 0.14),
        handlelength=2,
        handleheight=1.5,
    )

    # ==========================================================================
    # TABLE REFERENCE
    # ==========================================================================
    # Add table reference below the legend
    table_ref_y = 0.12  # Y position for table reference title

    # Table reference title
    fig.text(
        0.5,
        table_ref_y,
        "Table Reference",
        ha="center",
        va="top",
        fontsize=8,
        fontweight="bold",
        color=COLORS["text"],
    )

    # Table reference content - organized in columns
    # Left column: Sources and Aggregation
    # Right column: Feature Extraction & Grouping and Analysis
    ref_fontsize = 6
    line_spacing = 0.022
    col1_x = 0.08  # Left column x position
    col2_x = 0.55  # Right column x position

    # Column 1: Sources
    y_pos = table_ref_y - 0.035
    fig.text(col1_x, y_pos, "Sources", ha="left", va="top", fontsize=ref_fontsize,
             fontweight="bold", color=COLORS["text"])
    y_pos -= line_spacing
    fig.text(col1_x, y_pos, "[1]  TrodesPosV1 – Position processing", ha="left",
             va="top", fontsize=ref_fontsize, color=COLORS["text"])
    y_pos -= line_spacing
    fig.text(col1_x, y_pos, "[2]  SpikeSortingRecording – Spike detection & sorting",
             ha="left", va="top", fontsize=ref_fontsize, color=COLORS["text"])
    y_pos -= line_spacing
    fig.text(col1_x, y_pos, "[3]  ImportedSpikeSorting – Pre-sorted units from NWB",
             ha="left", va="top", fontsize=ref_fontsize, color=COLORS["text"])

    # Column 1: Aggregation (Merge Tables)
    y_pos -= line_spacing * 1.3
    fig.text(col1_x, y_pos, "Aggregation (Merge Tables)", ha="left", va="top",
             fontsize=ref_fontsize, fontweight="bold", color=COLORS["text"])
    y_pos -= line_spacing
    fig.text(col1_x, y_pos, "[4]  PositionOutput – Aggregates position sources",
             ha="left", va="top", fontsize=ref_fontsize, color=COLORS["text"])
    y_pos -= line_spacing
    fig.text(col1_x, y_pos, "[5]  SpikeSortingOutput – Aggregates spike sorting sources",
             ha="left", va="top", fontsize=ref_fontsize, color=COLORS["text"])
    y_pos -= line_spacing
    fig.text(col1_x, y_pos, "[12] DecodingOutput – Aggregates decoding results",
             ha="left", va="top", fontsize=ref_fontsize, color=COLORS["text"])

    # Column 2: Feature Extraction & Grouping
    y_pos2 = table_ref_y - 0.035
    fig.text(col2_x, y_pos2, "Feature Extraction & Grouping", ha="left", va="top",
             fontsize=ref_fontsize, fontweight="bold", color=COLORS["text"])
    y_pos2 -= line_spacing
    fig.text(col2_x, y_pos2, "[6]  UnitWaveformFeatures – Waveform amplitudes",
             ha="left", va="top", fontsize=ref_fontsize, color=COLORS["text"])
    y_pos2 -= line_spacing
    fig.text(col2_x, y_pos2, "[7]  SortedSpikesGroup – Group sorted units",
             ha="left", va="top", fontsize=ref_fontsize, color=COLORS["text"])
    y_pos2 -= line_spacing
    fig.text(col2_x, y_pos2, "[8]  PositionGroup – Group position data",
             ha="left", va="top", fontsize=ref_fontsize, color=COLORS["text"])
    y_pos2 -= line_spacing
    fig.text(col2_x, y_pos2, "[9]  UnitWaveformFeaturesGroup – Group waveform features",
             ha="left", va="top", fontsize=ref_fontsize, color=COLORS["text"])

    # Column 2: Analysis
    y_pos2 -= line_spacing * 1.3
    fig.text(col2_x, y_pos2, "Analysis", ha="left", va="top",
             fontsize=ref_fontsize, fontweight="bold", color=COLORS["text"])
    y_pos2 -= line_spacing
    fig.text(col2_x, y_pos2, "[10] ClusterlessDecodingV1 – Decode from waveform features",
             ha="left", va="top", fontsize=ref_fontsize, color=COLORS["text"])
    y_pos2 -= line_spacing
    fig.text(col2_x, y_pos2, "[11] SortedSpikesDecodingV1 – Decode from sorted spikes",
             ha="left", va="top", fontsize=ref_fontsize, color=COLORS["text"])

    return fig


def main():
    """Generate and save the supplemental figure."""
    # Get script directory for output
    script_dir = os.path.dirname(os.path.abspath(__file__))

    fig = create_figure()
    save_figure("figure5_supp", output_dir=script_dir)
    plt.close()

    print("Figure generation complete!")


if __name__ == "__main__":
    main()
