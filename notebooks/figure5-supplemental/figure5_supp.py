"""
Supplemental Figure: Spyglass Neural Decoding Workflow

Generates a publication-ready flowchart showing the two decoding pipelines
used in Figure 5: clusterless decoding (UCSF) and sorted spikes decoding (NYU).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


# =============================================================================
# FIGURE UTILITIES (standalone - no external dependencies)
# =============================================================================
def set_figure_defaults() -> None:
    """Set matplotlib defaults for publication-quality figures."""
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


def save_figure(figure_name: str, output_dir: str | None = None) -> None:
    """
    Save the current matplotlib figure as both PDF and PNG.

    Parameters
    ----------
    figure_name : str
        Base name for the output files (without extension).
    output_dir : str, optional
        Directory to save files to. Creates directory if it doesn't exist.
        Defaults to current working directory.

    Notes
    -----
    Files are saved at 300 DPI with tight bounding boxes.
    """
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
#: Color palette for figure elements following FIGURE5_SUPP.md design specs.
#: Keys correspond to pipeline components and UI elements.
COLORS: dict[str, str] = {
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
BOX_PAD = (
    0.02  # FancyBboxPatch pad extends visual box by this amount on each side
)
MERGE_BOX_OUTER_OFFSET = 0.015  # Extra offset for merge table outer border
ARROW_GAP = 0.004  # Small gap between arrow endpoint and box edge


# =============================================================================
# LAYOUT CONFIGURATION
# =============================================================================
@dataclass
class LayoutConfig:
    """Configuration for figure layout dimensions and positions."""

    # Figure dimensions
    fig_width: float = 7.0
    fig_height: float = 8.5

    # Vertical layout
    y_top: float = 0.86
    y_bottom: float = 0.22

    # Horizontal positions
    x_left: float = 0.30
    x_right: float = 0.70
    x_center: float = 0.50

    # Box dimensions
    box_width: float = 0.35
    box_width_small: float = 0.30

    # Table reference
    table_ref_y: float = 0.12
    ref_fontsize: float = 6
    line_spacing: float = 0.022
    col1_x: float = 0.08
    col2_x: float = 0.55

    # Derived layout values (computed from row spacing)
    n_rows: int = field(default=8, init=False)

    @property
    def total_height(self) -> float:
        """Total vertical space available for boxes."""
        return self.y_top - self.y_bottom

    @property
    def row_spacing(self) -> float:
        """Vertical spacing between box centers."""
        return self.total_height / (self.n_rows - 1)

    @property
    def box_height(self) -> float:
        """Standard box height (30% of row spacing for arrow visibility)."""
        return self.row_spacing * 0.30

    @property
    def box_height_figure(self) -> float:
        """Figure output box height (slightly smaller)."""
        return self.box_height * 0.6

    def get_y_positions(self) -> dict[str, float]:
        """Calculate Y positions for each row (top to bottom)."""
        return {
            "nwb": self.y_top - 0 * self.row_spacing,
            "input": self.y_top - 1 * self.row_spacing,
            "output1": self.y_top - 2 * self.row_spacing,
            "features": self.y_top - 3 * self.row_spacing,
            "group": self.y_top - 4 * self.row_spacing,
            "decoding": self.y_top - 5 * self.row_spacing,
            "decode_out": self.y_top - 6 * self.row_spacing,
            "figure": self.y_top - 7 * self.row_spacing,
        }


@dataclass
class BoxBounds:
    """Visual bounds of a box for arrow attachment."""

    top: float
    bottom: float


# Default layout configuration
DEFAULT_LAYOUT = LayoutConfig()


def get_box_bounds(
    y_center: float, height: float, is_merge: bool = False
) -> BoxBounds:
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
    BoxBounds
        Visual bounds for arrow attachment with 'top' and 'bottom' attributes
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

    return BoxBounds(top=visual_top, bottom=visual_bottom)


def draw_vertical_arrow_between_boxes(
    ax: Axes,
    x: float,
    src_bounds: BoxBounds,
    dst_bounds: BoxBounds,
    color: str | None = None,
) -> None:
    """
    Draw a vertical arrow between two boxes, leaving a small visual gap
    from each box edge. Arrow points toward destination (downward flow).

    Parameters
    ----------
    ax : Axes
        Axes to draw on.
    x : float
        X coordinate of the arrow.
    src_bounds : BoxBounds
        Bounds for the source box from get_box_bounds.
    dst_bounds : BoxBounds
        Bounds for the destination box from get_box_bounds.
    color : str, optional
        Arrow color. Defaults to COLORS["arrow"].
    """
    if color is None:
        color = COLORS["arrow"]

    # Leave ARROW_GAP below the source and above the destination
    # This creates symmetric spacing at both ends of the arrow
    start_y = src_bounds.bottom - ARROW_GAP
    end_y = dst_bounds.top + ARROW_GAP

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
    ax: Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    color: str,
    is_merge: bool = False,
    is_dashed: bool = False,
    fontsize: int = 7,
    ref_num: str | None = None,
) -> None:
    """
    Draw a styled box with optional merge table double-line or dashed border.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to draw on.
    x : float
        X center coordinate of the box.
    y : float
        Y center coordinate of the box.
    width : float
        Box width.
    height : float
        Box height.
    text : str
        Label text to display in the box.
    color : str
        Hex color string for box border and fill.
    is_merge : bool
        If True, draw double-line border (merge table style).
    is_dashed : bool
        If True, draw dashed border.
    fontsize : int
        Font size for label text.
    ref_num : str or None
        Reference number to display below text (e.g., "[1]").
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


def draw_arrow(
    ax: Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str | None = None,
) -> None:
    """
    Draw an arrow from start to end position.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to draw on.
    start : tuple[float, float]
        Starting (x, y) coordinates.
    end : tuple[float, float]
        Ending (x, y) coordinates.
    color : str, optional
        Arrow color. Defaults to COLORS["arrow"].
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


# =============================================================================
# PANEL CONFIGURATION
# =============================================================================
@dataclass
class PanelConfig:
    """Configuration for a single panel in the figure."""

    label: str
    title: str
    subtitle: str
    dandi_id: str
    neural_input_label: str
    neural_input_color: str
    neural_input_ref: str
    features_label: str
    features_ref: str
    group_label: str
    group_ref: str
    decoding_label: str
    decoding_ref: str
    figure_output: str
    has_waveform_group: bool = (
        True  # Panel A has waveform group, Panel B doesn't
    )


# Pre-defined panel configurations
PANEL_A_CONFIG = PanelConfig(
    label="A",
    title="UCSF Dataset",
    subtitle="(Clusterless Decoding)",
    dandi_id="DANDI:000937",
    neural_input_label="Raw\n(ephys)",
    neural_input_color="neural_frank",
    neural_input_ref="[2]",
    features_label="Waveform\nFeatures",
    features_ref="[6]",
    group_label="Waveform\nFeatures Group",
    group_ref="[9]",
    decoding_label="Clusterless\nDecoding",
    decoding_ref="[10]",
    figure_output="Figure 5A",
    has_waveform_group=True,
)

PANEL_B_CONFIG = PanelConfig(
    label="B",
    title="NYU Dataset",
    subtitle="(Sorted Spikes Decoding)",
    dandi_id="DANDI:000059",
    neural_input_label="Imported\nSpike Sorting",
    neural_input_color="neural_buzsaki",
    neural_input_ref="[3]",
    features_label="Sorted Spikes\nGroup",
    features_ref="[7]",
    group_label="",  # Not used in Panel B
    group_ref="",  # Not used in Panel B
    decoding_label="Sorted Spikes\nDecoding",
    decoding_ref="[11]",
    figure_output="Figure 5B-D",
    has_waveform_group=False,
)


# =============================================================================
# PANEL DRAWING FUNCTIONS
# =============================================================================
def _draw_panel_title(ax: Axes, title: str, subtitle: str) -> None:
    """
    Draw the title and subtitle for a panel.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to draw on.
    title : str
        Main title text (bold, larger font).
    subtitle : str
        Subtitle text (smaller font, below title).
    """
    ax.text(
        0.5,
        0.98,
        title,
        ha="center",
        va="top",
        fontsize=9,
        fontweight="bold",
        color=COLORS["text"],
    )
    ax.text(
        0.5,
        0.94,
        subtitle,
        ha="center",
        va="top",
        fontsize=7,
        color=COLORS["text"],
    )


def _draw_panel_label(ax: Axes, label: str) -> None:
    """
    Draw the panel label (A, B, etc.) in the top-left corner.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to draw on.
    label : str
        Panel label text (e.g., "A", "B").
    """
    ax.text(
        0.02,
        0.98,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
    )


def _draw_split_arrow(
    ax: Axes,
    bounds_src: BoxBounds,
    bounds_dst: BoxBounds,
    x_center: float,
    x_left: float,
    x_right: float,
) -> tuple[float, float]:
    """
    Draw a split arrow from one source box to two destination boxes.

    Draws a vertical line from source, horizontal line to span destinations,
    then arrows down to each destination box.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to draw on.
    bounds_src : BoxBounds
        Visual bounds of the source box.
    bounds_dst : BoxBounds
        Visual bounds of the destination boxes (assumes same row).
    x_center : float
        X coordinate of the source box center.
    x_left : float
        X coordinate of the left destination box center.
    x_right : float
        X coordinate of the right destination box center.

    Returns
    -------
    tuple[float, float]
        The (split_y, end_y) positions for potential reuse.
    """
    gap_between = bounds_src.bottom - bounds_dst.top
    split_y = bounds_src.bottom - gap_between / 3

    # Vertical line from source to split point
    ax.plot(
        [x_center, x_center],
        [bounds_src.bottom - ARROW_GAP, split_y],
        color=COLORS["arrow"],
        linewidth=1,
    )
    # Horizontal line connecting branches
    ax.plot(
        [x_left, x_right],
        [split_y, split_y],
        color=COLORS["arrow"],
        linewidth=1,
    )
    # Arrows to destination boxes
    end_y = bounds_dst.top + ARROW_GAP
    draw_arrow(ax, (x_left, split_y), (x_left, end_y))
    draw_arrow(ax, (x_right, split_y), (x_right, end_y))

    return split_y, end_y


def _draw_merge_arrow(
    ax: Axes,
    bounds_left: BoxBounds,
    bounds_right: BoxBounds,
    bounds_dst: BoxBounds,
    x_left: float,
    x_right: float,
    x_center: float,
) -> None:
    """
    Draw arrows that merge from two source boxes to one destination box.

    Draws vertical lines from each source to a merge point, horizontal line
    connecting them, then a single arrow down to the destination.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to draw on.
    bounds_left : BoxBounds
        Visual bounds of the left source box.
    bounds_right : BoxBounds
        Visual bounds of the right source box.
    bounds_dst : BoxBounds
        Visual bounds of the destination box.
    x_left : float
        X coordinate of the left source box center.
    x_right : float
        X coordinate of the right source box center.
    x_center : float
        X coordinate of the destination box center.
    """
    gap_to_dst = bounds_left.bottom - bounds_dst.top
    merge_y = bounds_left.bottom - gap_to_dst / 3

    # Vertical lines from sources to merge point
    ax.plot(
        [x_left, x_left],
        [bounds_left.bottom - ARROW_GAP, merge_y],
        color=COLORS["arrow"],
        linewidth=1,
    )
    ax.plot(
        [x_right, x_right],
        [bounds_right.bottom - ARROW_GAP, merge_y],
        color=COLORS["arrow"],
        linewidth=1,
    )
    # Horizontal line connecting branches
    ax.plot(
        [x_left, x_right],
        [merge_y, merge_y],
        color=COLORS["arrow"],
        linewidth=1,
    )
    # Arrow to destination
    merge_end_y = bounds_dst.top + ARROW_GAP
    draw_arrow(ax, (x_center, merge_y), (x_center, merge_end_y))


def _draw_group_layer(
    ax: Axes,
    y_pos: float,
    box_width: float,
    box_height: float,
    x_left: float,
    x_right: float,
    config: PanelConfig,
) -> None:
    """
    Draw the grouping layer for a panel.

    For clusterless panels (has_waveform_group=True): draws both position and
    waveform feature groups.
    For sorted spikes panels: draws only the position group.
    """
    # Position group is always drawn on the left
    draw_box(
        ax,
        x_left,
        y_pos,
        box_width,
        box_height,
        "Position\nGroup",
        COLORS["position"],
        ref_num="[8]",
    )
    # Waveform features group only for clusterless pipeline
    if config.has_waveform_group:
        draw_box(
            ax,
            x_right,
            y_pos,
            box_width,
            box_height,
            config.group_label,
            COLORS[config.neural_input_color],
            ref_num=config.group_ref,
        )


def _draw_panel(
    ax: Axes,
    config: PanelConfig,
    layout: LayoutConfig,
    y_positions: dict[str, float],
    bounds: dict[str, BoxBounds],
) -> None:
    """
    Draw a complete decoding pipeline panel.

    This renders the standard pipeline structure:
    1. NWB source file
    2. Split to position/neural input streams
    3. Merge to output tables
    4. Feature extraction (waveform features or sorted spikes group)
    5. Grouping (position group, optionally waveform features group)
    6. Merge streams to decoding
    7. Decoding output
    8. Figure output

    The differences between clusterless and sorted spikes pipelines are
    captured in the PanelConfig.
    """
    bw = layout.box_width
    bw_small = layout.box_width_small
    bh = layout.box_height
    bh_figure = layout.box_height_figure

    # Panel title and label
    _draw_panel_title(ax, config.title, config.subtitle)

    # Layer 1: NWB File
    draw_box(
        ax,
        layout.x_center,
        y_positions["nwb"],
        bw,
        bh,
        f"NWB File\n{config.dandi_id}",
        COLORS["nwb"],
    )

    # Arrow: NWB splits to position and neural inputs
    _draw_split_arrow(
        ax,
        bounds["nwb"],
        bounds["input"],
        layout.x_center,
        layout.x_left,
        layout.x_right,
    )

    # Layer 2: Input streams (position and neural)
    draw_box(
        ax,
        layout.x_left,
        y_positions["input"],
        bw_small,
        bh,
        "Raw\nPosition",
        COLORS["position"],
        ref_num="[1]",
    )
    draw_box(
        ax,
        layout.x_right,
        y_positions["input"],
        bw_small,
        bh,
        config.neural_input_label,
        COLORS[config.neural_input_color],
        ref_num=config.neural_input_ref,
    )

    # Arrows: Input to output layer
    draw_vertical_arrow_between_boxes(
        ax, layout.x_left, bounds["input"], bounds["output1"]
    )
    draw_vertical_arrow_between_boxes(
        ax, layout.x_right, bounds["input"], bounds["output1"]
    )

    # Layer 3: Output merge tables
    draw_box(
        ax,
        layout.x_left,
        y_positions["output1"],
        bw_small,
        bh,
        "Position\nOutput",
        COLORS["merge"],
        is_merge=True,
        ref_num="[4]",
    )
    draw_box(
        ax,
        layout.x_right,
        y_positions["output1"],
        bw_small,
        bh,
        "Spike Sorting\nOutput",
        COLORS["merge"],
        is_merge=True,
        ref_num="[5]",
    )

    # Arrow: Spike sorting output to features layer
    draw_vertical_arrow_between_boxes(
        ax, layout.x_right, bounds["output1"], bounds["features"]
    )

    # Layer 4: Features (waveform features or sorted spikes group)
    draw_box(
        ax,
        layout.x_right,
        y_positions["features"],
        bw_small,
        bh,
        config.features_label,
        COLORS[config.neural_input_color],
        ref_num=config.features_ref,
    )

    # Arrows: Position output skips to group; features to group
    draw_vertical_arrow_between_boxes(
        ax, layout.x_left, bounds["output1"], bounds["group"]
    )
    if config.has_waveform_group:
        # Clusterless: waveform features -> waveform features group
        draw_vertical_arrow_between_boxes(
            ax, layout.x_right, bounds["features"], bounds["group"]
        )

    # Layer 5: Grouping
    _draw_group_layer(
        ax,
        y_positions["group"],
        bw_small,
        bh,
        layout.x_left,
        layout.x_right,
        config,
    )

    # Arrows: Merge to decoding
    # For clusterless: both streams merge from group layer
    # For sorted spikes: position group + features layer merge
    bounds_right_source = (
        bounds["group"] if config.has_waveform_group else bounds["features"]
    )
    _draw_merge_arrow(
        ax,
        bounds["group"],
        bounds_right_source,
        bounds["decoding"],
        layout.x_left,
        layout.x_right,
        layout.x_center,
    )

    # Layer 6: Decoding
    draw_box(
        ax,
        layout.x_center,
        y_positions["decoding"],
        bw,
        bh,
        config.decoding_label,
        COLORS["decoding"],
        ref_num=config.decoding_ref,
    )

    # Arrow: Decoding to decoding output
    draw_vertical_arrow_between_boxes(
        ax, layout.x_center, bounds["decoding"], bounds["decode_out"]
    )

    # Layer 7: Decoding output
    draw_box(
        ax,
        layout.x_center,
        y_positions["decode_out"],
        bw,
        bh,
        "Decoding\nOutput",
        COLORS["merge"],
        is_merge=True,
        ref_num="[12]",
    )

    # Arrow: Decoding output to figure
    draw_vertical_arrow_between_boxes(
        ax, layout.x_center, bounds["decode_out"], bounds["figure"]
    )

    # Layer 8: Figure output
    draw_box(
        ax,
        layout.x_center,
        y_positions["figure"],
        bw_small,
        bh_figure,
        config.figure_output,
        COLORS["arrow"],
        is_dashed=True,
        fontsize=7,
    )

    _draw_panel_label(ax, config.label)


# =============================================================================
# LEGEND AND TABLE REFERENCE
# =============================================================================
def _create_legend_patch(color_key: str) -> FancyBboxPatch:
    """
    Create a legend patch for the given color key.

    Parameters
    ----------
    color_key : str
        Key into the COLORS dictionary.

    Returns
    -------
    FancyBboxPatch
        A styled patch for use in the figure legend.
    """
    return FancyBboxPatch(
        (0, 0),
        0.1,
        0.03,
        boxstyle="round,pad=0.01",
        facecolor=(*plt.cm.colors.hex2color(COLORS[color_key]), FILL_ALPHA),
        edgecolor=COLORS[color_key],
        linewidth=1,
    )


def _draw_legend(fig: Figure) -> None:
    """
    Draw the figure legend.

    Parameters
    ----------
    fig : Figure
        Matplotlib figure to draw the legend on.
    """
    legend_items = [
        ("position", "Position processing"),
        ("neural_frank", "Neural (clusterless)"),
        ("neural_buzsaki", "Neural (sorted)"),
        ("decoding", "Decoding"),
        ("merge", "Merge table"),
    ]

    handles = [_create_legend_patch(color_key) for color_key, _ in legend_items]
    labels = [label for _, label in legend_items]

    fig.legend(
        handles=handles,
        labels=labels,
        loc="lower center",
        ncol=5,
        fontsize=6,
        frameon=False,
        bbox_to_anchor=(0.5, 0.14),
        handlelength=2,
        handleheight=1.5,
    )


def _draw_table_reference_section(
    fig: Figure,
    x: float,
    start_y: float,
    title: str,
    entries: list[str],
    layout: LayoutConfig,
) -> float:
    """
    Draw a section of the table reference.

    Parameters
    ----------
    fig : Figure
        Matplotlib figure to draw on.
    x : float
        X coordinate for the section (figure coordinates).
    start_y : float
        Y coordinate for the section title (figure coordinates).
    title : str
        Section title text (displayed in bold).
    entries : list[str]
        List of entry strings to display below the title.
    layout : LayoutConfig
        Layout configuration with font sizes and spacing.

    Returns
    -------
    float
        The Y position after the last entry.
    """
    y_pos = start_y
    fig.text(
        x,
        y_pos,
        title,
        ha="left",
        va="top",
        fontsize=layout.ref_fontsize,
        fontweight="bold",
        color=COLORS["text"],
    )
    for entry in entries:
        y_pos -= layout.line_spacing
        fig.text(
            x,
            y_pos,
            entry,
            ha="left",
            va="top",
            fontsize=layout.ref_fontsize,
            color=COLORS["text"],
        )
    return y_pos


def _draw_table_reference(fig: Figure, layout: LayoutConfig) -> None:
    """
    Draw the table reference section below the legend.

    Parameters
    ----------
    fig : Figure
        Matplotlib figure to draw on.
    layout : LayoutConfig
        Layout configuration with positioning and font settings.
    """
    # Title
    fig.text(
        0.5,
        layout.table_ref_y,
        "Table Reference",
        ha="center",
        va="top",
        fontsize=8,
        fontweight="bold",
        color=COLORS["text"],
    )

    start_y = layout.table_ref_y - 0.035

    # Column 1: Sources
    sources = [
        "[1]  TrodesPosV1 - Position processing",
        "[2]  SpikeSortingRecording - Spike detection & sorting",
        "[3]  ImportedSpikeSorting - Pre-sorted units from NWB",
    ]
    y_pos = _draw_table_reference_section(
        fig, layout.col1_x, start_y, "Sources", sources, layout
    )

    # Column 1: Aggregation
    aggregation = [
        "[4]  PositionOutput - Aggregates position sources",
        "[5]  SpikeSortingOutput - Aggregates spike sorting sources",
        "[12] DecodingOutput - Aggregates decoding results",
    ]
    y_pos -= layout.line_spacing * 1.3
    _draw_table_reference_section(
        fig,
        layout.col1_x,
        y_pos,
        "Aggregation (Merge Tables)",
        aggregation,
        layout,
    )

    # Column 2: Feature Extraction & Grouping
    features = [
        "[6]  UnitWaveformFeatures - Waveform amplitudes",
        "[7]  SortedSpikesGroup - Group sorted units",
        "[8]  PositionGroup - Group position data",
        "[9]  UnitWaveformFeaturesGroup - Group waveform features",
    ]
    y_pos2 = _draw_table_reference_section(
        fig,
        layout.col2_x,
        start_y,
        "Feature Extraction & Grouping",
        features,
        layout,
    )

    # Column 2: Analysis
    analysis = [
        "[10] ClusterlessDecodingV1 - Decode from waveform features",
        "[11] SortedSpikesDecodingV1 - Decode from sorted spikes",
    ]
    y_pos2 -= layout.line_spacing * 1.3
    _draw_table_reference_section(
        fig, layout.col2_x, y_pos2, "Analysis", analysis, layout
    )


# =============================================================================
# MAIN FIGURE CREATION
# =============================================================================
def _compute_box_bounds(
    y_positions: dict[str, float], layout: LayoutConfig
) -> dict[str, BoxBounds]:
    """
    Compute visual bounds for all box rows.

    Parameters
    ----------
    y_positions : dict[str, float]
        Mapping of layer names to Y center coordinates.
    layout : LayoutConfig
        Layout configuration with box dimensions.

    Returns
    -------
    dict[str, BoxBounds]
        Mapping of layer names to visual bounds for arrow attachment.
    """
    bh = layout.box_height
    bh_figure = layout.box_height_figure

    return {
        "nwb": get_box_bounds(y_positions["nwb"], bh, is_merge=False),
        "input": get_box_bounds(y_positions["input"], bh, is_merge=False),
        "output1": get_box_bounds(y_positions["output1"], bh, is_merge=True),
        "features": get_box_bounds(y_positions["features"], bh, is_merge=False),
        "group": get_box_bounds(y_positions["group"], bh, is_merge=False),
        "decoding": get_box_bounds(y_positions["decoding"], bh, is_merge=False),
        "decode_out": get_box_bounds(
            y_positions["decode_out"], bh, is_merge=True
        ),
        "figure": get_box_bounds(
            y_positions["figure"], bh_figure, is_merge=False
        ),
    }


def _setup_figure(layout: LayoutConfig) -> tuple[Figure, dict[str, Axes]]:
    """
    Create and configure the figure and axes.

    Parameters
    ----------
    layout : LayoutConfig
        Layout configuration with figure dimensions.

    Returns
    -------
    tuple[Figure, dict[str, Axes]]
        The figure and a dictionary mapping panel labels to axes.
    """
    fig, axes = plt.subplot_mosaic(
        [["A", "B"]],
        figsize=(layout.fig_width, layout.fig_height),
        width_ratios=[1, 1],
        dpi=300,
        constrained_layout=True,
    )

    for ax in axes.values():
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    return fig, axes


def create_figure(layout: LayoutConfig | None = None) -> Figure:
    """
    Create the supplemental figure showing decoding workflows.

    Parameters
    ----------
    layout : LayoutConfig, optional
        Layout configuration. Uses default if not provided.

    Returns
    -------
    Figure
        The matplotlib figure object.
    """
    if layout is None:
        layout = DEFAULT_LAYOUT

    set_figure_defaults()

    fig, axes = _setup_figure(layout)
    y_positions = layout.get_y_positions()
    bounds = _compute_box_bounds(y_positions, layout)

    # Draw panels
    _draw_panel(axes["A"], PANEL_A_CONFIG, layout, y_positions, bounds)
    _draw_panel(axes["B"], PANEL_B_CONFIG, layout, y_positions, bounds)

    # Draw legend and table reference
    _draw_legend(fig)
    _draw_table_reference(fig, layout)

    return fig


def main() -> None:
    """
    Generate and save the supplemental figure.

    Creates the Figure 5 supplemental workflow diagram and saves it as both
    PDF and PNG in the same directory as this script.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))

    create_figure()
    save_figure("figure5_supp", output_dir=script_dir)
    plt.close()

    print("Figure generation complete!")


if __name__ == "__main__":
    main()
