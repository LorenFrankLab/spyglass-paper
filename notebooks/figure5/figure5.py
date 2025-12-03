"""
Figure 5: Neural Decoding with Spyglass

Generates publication-ready figures demonstrating Spyglass decoding pipelines:
- Panel A: UCSF/Frank Lab clusterless decoding (j1620210710)
- Panel B-D: NYU/Buzsaki Lab sorted spikes decoding (MS2220180629)

This script fetches pre-computed results from Spyglass database tables
and generates the visualizations. It assumes all decoding has been run
previously (see notebooks for data population examples).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns

# Add src to path for paper_plotting imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from paper_plotting import (
    PAGE_HEIGHT,
    TWO_COLUMN,
    plot_2D_track_graph,
    plot_graph_as_1D,
    save_figure,
    set_figure_defaults,
)

# Spyglass and analysis imports (deferred to avoid import-time database connections)
# These are imported at module level for visibility but guarded in functions
if TYPE_CHECKING:
    from typing import Any

    import networkx as nx
    import xarray as xr


# =============================================================================
# DATA STRUCTURES
# =============================================================================
@dataclass
class DecodingData:
    """
    Container for decoding analysis results.

    Using a dataclass instead of a dict provides:
    - Self-documenting field names
    - Attribute access instead of string keys
    - Type checking support
    """

    results: xr.Dataset
    posterior: xr.DataArray
    linear_position_info: pd.DataFrame
    multiunit_rate: np.ndarray
    classifier: Any  # ContFragClusterlessClassifier or ContFragSortedSpikesClassifier
    track_graph: nx.Graph
    track_graph_params: dict[str, Any]
    position_info: pd.DataFrame | None = None
    nwb_file_copy_name: str | None = None
    ahead_behind_distance: np.ndarray | None = None



# =============================================================================
# CONSTANTS
# =============================================================================
# Analysis constants
SPEED_THRESHOLD_CM_PER_S = 10.0  # Threshold for run vs immobility classification
BUZSAKI_SAMPLING_INTERVAL_S = 0.004  # 250 Hz sampling rate
SPECTRAL_SAMPLING_FREQUENCY_HZ = 250
SPECTRAL_TIME_WINDOW_S = 3.0
SPECTRAL_TIME_HALFBANDWIDTH = 1

# Visualization constants
AHEAD_BEHIND_YLIM_CM = (-30, 30)  # Y-axis limits for ahead/behind distance plot
DECODE_DISTANCE_YLIM_CM = (8, 14)  # Y-axis limits for decode distance error plot
DEFAULT_SCALEBAR_LENGTH_CM = 20  # Default scale bar length for track figures
CM_TO_M = 0.01  # Conversion factor from centimeters to meters
CONFIDENCE_INTERVAL_Z = 1.96  # Z-score for 95% confidence interval




# =============================================================================
# CONFIGURATION
# =============================================================================
@dataclass(frozen=True)
class DatasetConfig:
    """
    Configuration for a single dataset's decoding analysis.

    Parameters
    ----------
    name : str
        Short identifier for the dataset (e.g., "frank", "buzsaki").
    nwb_file_name : str
        NWB file name in the Spyglass database. For UCSF/Frank Lab data, this
        is already the copy filename (with underscore suffix). For NYU/Buzsaki
        Lab data, this is the original filename which gets converted to the
        copy filename via get_nwb_copy_filename().
    interval_list_name : str
        Name of the interval list for position data.
    position_params_name : str
        Name of position processing parameters.
    decoding_param_name : str
        Name of decoding parameters in DecodingParameters table.
    encoding_interval : str
        Interval name used for encoding the decoder.
    decoding_interval : str
        Interval name used for running decoding.
    group_name : str
        Name of the feature/position group in Spyglass tables.
    time_slice : slice
        Index slice for the time window to plot in decode figure.
    time_slice_description : str
        Human-readable description of what the time slice represents.
    output_prefix : str
        Prefix for output figure filenames.
    reward_well_nodes : tuple[int, ...]
        Node indices in track graph that are reward wells.
    edge_colormap : str
        Matplotlib colormap name for track edges.
    position_x_name : str
        Column name for x position in position dataframe.
    position_y_name : str
        Column name for y position in position dataframe.
    speed_name : str
        Column name for speed in position dataframe.
    track_graph_interval_list_name : str
        Interval list name for fetching track graph (Frank Lab only).
    """

    name: str
    nwb_file_name: str
    interval_list_name: str
    position_params_name: str
    decoding_param_name: str
    encoding_interval: str
    decoding_interval: str
    group_name: str
    time_slice: slice
    time_slice_description: str
    output_prefix: str
    reward_well_nodes: tuple[int, ...]
    edge_colormap: str
    position_x_name: str
    position_y_name: str
    speed_name: str
    track_graph_interval_list_name: str = ""

    def validate(self) -> None:
        """
        Validate configuration consistency.

        Raises
        ------
        ValueError
            If time_slice is invalid, NWB filename is malformed, or colormap
            doesn't exist.
        """
        # Validate time_slice
        if self.time_slice.start is None or self.time_slice.stop is None:
            raise ValueError(
                f"time_slice must have explicit start and stop, got {self.time_slice}"
            )
        if self.time_slice.start >= self.time_slice.stop:
            raise ValueError(
                f"time_slice.start ({self.time_slice.start}) must be less than "
                f"time_slice.stop ({self.time_slice.stop})"
            )
        if self.time_slice.start < 0:
            raise ValueError(
                f"time_slice.start must be non-negative, got {self.time_slice.start}"
            )

        # Validate NWB filename
        if not self.nwb_file_name.endswith(".nwb"):
            raise ValueError(
                f"nwb_file_name must end with '.nwb', got '{self.nwb_file_name}'"
            )

        # Validate colormap exists
        if self.edge_colormap not in plt.colormaps:
            raise ValueError(
                f"edge_colormap '{self.edge_colormap}' not found in matplotlib colormaps"
            )


# UCSF (Frank Lab) - Clusterless decoding
UCSF_CONFIG = DatasetConfig(
    name="ucsf",
    nwb_file_name="j1620210710_.nwb",
    interval_list_name="runs_noPrePostTrialTimes raw data valid times",
    position_params_name="default_decoding",
    decoding_param_name="j1620210710_contfrag_clusterless_1D",
    encoding_interval="06_r3 noPrePostTrialTimes",
    decoding_interval="06_r3 noPrePostTrialTimes",
    group_name="test_group",
    # 2-second window (1000 samples @ 500 Hz) during run epoch r3
    time_slice=slice(212_500, 213_500),
    time_slice_description="2s window during run epoch r3",
    output_prefix="figure5_frank",
    reward_well_nodes=tuple(range(6)),
    edge_colormap="tab10",
    position_x_name="head_position_x",
    position_y_name="head_position_y",
    speed_name="head_speed",
    track_graph_interval_list_name="pos 5 valid times",
)

# NYU (Buzsaki Lab) - Sorted spikes decoding
NYU_CONFIG = DatasetConfig(
    name="nyu",
    nwb_file_name="MS2220180629.nwb",
    interval_list_name="pos 0 valid times",
    position_params_name="single_led_decoding",
    decoding_param_name="MS2220180629_contfrag_sorted",
    encoding_interval="pos 0 valid times",
    decoding_interval="pos 0 valid times",
    group_name="MS2220180629",
    # 3.2-second window (800 samples @ 250 Hz) showing theta sequences
    time_slice=slice(229_700, 230_500),
    time_slice_description="3.2s window showing theta sequences",
    output_prefix="figure5_buzsaki",
    reward_well_nodes=(0,),
    edge_colormap="tab20",
    position_x_name="position_x",
    position_y_name="position_y",
    speed_name="speed",
    # Track graph is created manually for Buzsaki, not fetched from DB
    track_graph_interval_list_name="",
)


# =============================================================================
# VALIDATION HELPERS
# =============================================================================
def validate_time_slice(time_slice: slice, data_length: int, context: str) -> None:
    """
    Validate that a time slice is within bounds of the data.

    Parameters
    ----------
    time_slice : slice
        The slice to validate.
    data_length : int
        Length of the data array being sliced.
    context : str
        Description of the data being sliced for error messages.

    Raises
    ------
    ValueError
        If time_slice indices are out of bounds.
    """
    if time_slice.stop > data_length:
        raise ValueError(
            f"time_slice.stop ({time_slice.stop}) exceeds {context} length "
            f"({data_length}). Check that the configured time_slice is valid "
            f"for this dataset."
        )
    if time_slice.start >= data_length:
        raise ValueError(
            f"time_slice.start ({time_slice.start}) exceeds {context} length "
            f"({data_length}). Check that the configured time_slice is valid "
            f"for this dataset."
        )


def validate_non_empty_slice(
    sliced_data: pd.DataFrame | np.ndarray,
    context: str,
) -> None:
    """
    Validate that a sliced dataframe or array is non-empty.

    Parameters
    ----------
    sliced_data : pandas.DataFrame or numpy.ndarray
        The sliced data to validate.
    context : str
        Description of the data for error messages.

    Raises
    ------
    ValueError
        If the sliced data is empty.
    """
    if len(sliced_data) == 0:
        raise ValueError(
            f"Time slice produced empty {context}. Check that the configured "
            f"time_slice indices are valid for this dataset."
        )


# =============================================================================
# DATA LOADING
# =============================================================================
def _validate_decoding_results(
    results: xr.Dataset,
    config: DatasetConfig,
    decoding_type: str,
) -> None:
    """
    Validate that decoding results were successfully fetched.

    Parameters
    ----------
    results : xarray.Dataset
        Decoding results from Spyglass.
    config : DatasetConfig
        Dataset configuration for error messages.
    decoding_type : str
        Type of decoding ("clusterless" or "sorted spikes") for error messages.

    Raises
    ------
    ValueError
        If results are empty or invalid.
    """
    if results is None:
        raise ValueError(
            f"No {decoding_type} decoding results found for {config.name} dataset. "
            f"Check that decoding has been run for {config.nwb_file_name}."
        )
    if len(results.time) == 0:
        raise ValueError(
            f"Empty {decoding_type} decoding results for {config.name} dataset. "
            f"Results have 0 time points."
        )


def fetch_clusterless_decoding_data(config: DatasetConfig) -> DecodingData:
    """
    Fetch clusterless decoding results from Spyglass tables.

    Parameters
    ----------
    config : DatasetConfig
        Configuration for the dataset.

    Returns
    -------
    DecodingData
        Container with results, posterior, linear_position_info,
        multiunit_rate, classifier, track_graph, and track_graph_params.

    Raises
    ------
    ValueError
        If no decoding results are found for the configuration.
    """
    from spyglass.decoding.v1.clusterless import ClusterlessDecodingV1
    from spyglass.linearization.v0.main import IntervalLinearizedPosition, TrackGraph

    selection_key = {
        "waveform_features_group_name": config.group_name,
        "position_group_name": config.group_name,
        "decoding_param_name": config.decoding_param_name,
        "nwb_file_name": config.nwb_file_name,
        "encoding_interval": config.encoding_interval,
        "decoding_interval": config.decoding_interval,
        "estimate_decoding_params": True,
    }

    # Fetch and validate results
    results = (ClusterlessDecodingV1 & selection_key).fetch_results()
    _validate_decoding_results(results, config, "clusterless")
    posterior = results.acausal_posterior.unstack("state_bins").sum("state")

    # Fetch multiunit rate
    multiunit_rate = ClusterlessDecodingV1.get_firing_rate(
        selection_key, results.time.values, multiunit=True
    ).squeeze()

    # Fetch linear position info
    linear_position_info = ClusterlessDecodingV1.fetch_linear_position_info(
        selection_key
    )

    # Fetch classifier
    classifier = (ClusterlessDecodingV1 & selection_key).fetch_model()

    # Fetch track graph using interval from config
    linearization_key = {
        "position_info_param_name": config.position_params_name,
        "nwb_file_name": config.nwb_file_name,
        "interval_list_name": config.track_graph_interval_list_name,
        "linearization_param_name": "default",
    }
    track_graph_name = (IntervalLinearizedPosition() & linearization_key).fetch1(
        "track_graph_name"
    )
    track_graph = (
        TrackGraph() & {"track_graph_name": track_graph_name}
    ).get_networkx_track_graph()
    track_graph_params = (
        TrackGraph() & {"track_graph_name": track_graph_name}
    ).fetch1()

    return DecodingData(
        results=results,
        posterior=posterior,
        linear_position_info=linear_position_info,
        multiunit_rate=multiunit_rate,
        classifier=classifier,
        track_graph=track_graph,
        track_graph_params=track_graph_params,
    )


def fetch_sorted_spikes_decoding_data(config: DatasetConfig) -> DecodingData:
    """
    Fetch sorted spikes decoding results from Spyglass tables.

    Parameters
    ----------
    config : DatasetConfig
        Configuration for the dataset. Note: nwb_file_name is the original
        filename which will be converted to the copy filename internally.

    Returns
    -------
    DecodingData
        Container with results, posterior, linear_position_info,
        multiunit_rate, classifier, track_graph, track_graph_params,
        position_info, and nwb_file_copy_name.

    Raises
    ------
    ValueError
        If no decoding results are found for the configuration.
    """
    import spyglass.decoding as sgd
    import spyglass.position as sgp
    from position_tools import get_angle
    from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename

    # Convert original NWB filename to Spyglass copy filename
    # (adds underscore before .nwb extension)
    nwb_file_copy_name = get_nwb_copy_filename(config.nwb_file_name)

    decoding_selection_key = {
        "sorted_spikes_group_name": config.group_name,
        "position_group_name": config.group_name,
        "decoding_param_name": config.decoding_param_name,
        "nwb_file_name": nwb_file_copy_name,
        "encoding_interval": config.encoding_interval,
        "decoding_interval": config.decoding_interval,
        "estimate_decoding_params": True,
        "unit_filter_params_name": config.group_name,
    }

    # Fetch and validate results
    results = (sgd.SortedSpikesDecodingV1 & decoding_selection_key).fetch_results()
    _validate_decoding_results(results, config, "sorted spikes")
    posterior = results.acausal_posterior.unstack("state_bins").sum("state")

    # Fetch linear position info
    linear_position_info = sgd.SortedSpikesDecodingV1.fetch_linear_position_info(
        decoding_selection_key
    )

    # Fetch multiunit rate
    multiunit_rate = sgd.v1.sorted_spikes.SortedSpikesGroup.get_firing_rate(
        decoding_selection_key, results.time.values, multiunit=True
    ).squeeze()

    # Fetch classifier
    classifier = (sgd.SortedSpikesDecodingV1 & decoding_selection_key).fetch_model()

    # Fetch position info for speed/orientation
    pos_merge_key = (
        sgp.PositionOutput.TrodesPosV1
        & {
            "nwb_file_name": nwb_file_copy_name,
            "interval_list_name": config.interval_list_name,
        }
    ).fetch1("KEY")
    position_info = (sgp.PositionOutput & pos_merge_key).fetch1_dataframe()

    # Create track graph (Buzsaki figure-8 maze)
    track_graph, track_graph_params = create_buzsaki_track_graph()

    # Add orientation to linear_position_info.
    # Orientation is computed as the angle between consecutive position samples.
    # NaN values occur at the first sample (no previous position) and when the
    # animal is stationary (zero displacement). We set NaNs to 0.0 because:
    # 1. The first sample has no valid orientation - 0.0 is a neutral default
    # 2. When stationary, orientation is undefined - 0.0 preserves the last
    #    known heading direction conceptually (no rotation)
    # This affects <1% of samples and has negligible impact on analyses.
    orientation = get_angle(
        position_info[[config.position_x_name, config.position_y_name]]
        .shift(1)
        .to_numpy(),
        position_info[[config.position_x_name, config.position_y_name]].to_numpy(),
    )
    orientation[np.isnan(orientation)] = 0.0
    linear_position_info["orientation"] = orientation

    return DecodingData(
        results=results,
        posterior=posterior,
        linear_position_info=linear_position_info,
        multiunit_rate=multiunit_rate,
        classifier=classifier,
        track_graph=track_graph,
        track_graph_params=track_graph_params,
        position_info=position_info,
        nwb_file_copy_name=nwb_file_copy_name,
    )


# =============================================================================
# TRACK GRAPH CREATION
# =============================================================================
def create_buzsaki_track_graph() -> tuple[nx.Graph, dict[str, Any]]:
    """
    Create the Buzsaki figure-8 maze track graph.

    Constructs a networkx graph representing the figure-8 maze topology
    used in the Buzsaki lab experiments, with node positions and edge
    connectivity defined for linearization.

    Returns
    -------
    track_graph : networkx.Graph
        Graph with nodes containing 'pos' attributes and edges containing
        'distance' attributes.
    track_graph_params : dict
        Dictionary with 'linear_edge_order' (list of edge tuples) and
        'linear_edge_spacing' (float) for linearization.
    """
    import track_linearization as tl

    node_positions = np.array(
        [
            (0, -52),  # center bottom
            (0, 45),  # center top
            (-20, 45),
            (-40, 30),  # left middle
            (-50, 0),
            (-40, -30),  # right middle
            (-20, -45),
            (20, 45),
            (40, 30),
            (50, 0),
            (40, -30),
            (20, -45),
        ]
    )

    edges = np.array(
        [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 6),
            (6, 0),
            (1, 7),
            (7, 8),
            (8, 9),
            (9, 10),
            (10, 11),
            (11, 0),
        ]
    )

    linear_edge_order = [tuple(e) for e in edges]
    linear_edge_spacing = 0

    track_graph = tl.make_track_graph(node_positions, edges)

    track_graph_params = {
        "linear_edge_order": linear_edge_order,
        "linear_edge_spacing": linear_edge_spacing,
    }

    return track_graph, track_graph_params


# =============================================================================
# ANALYSIS HELPERS
# =============================================================================
def compute_ahead_behind_distance(
    posterior: xr.DataArray,
    linear_position_info: pd.DataFrame,
    classifier: Any,
    track_graph: nx.Graph,
    time_slice: slice | None = None,
) -> np.ndarray:
    """
    Compute the ahead/behind distance between decoded and actual position.

    Parameters
    ----------
    posterior : xr.DataArray
        Decoded posterior probability.
    linear_position_info : pd.DataFrame
        Linearized position information with projected positions and orientation.
    classifier : Any
        Trained decoder model.
    track_graph : nx.Graph
        Track graph for the environment.
    time_slice : slice, optional
        Time slice to compute for. If None, computes for all times.

    Returns
    -------
    np.ndarray
        Ahead/behind distance in cm (positive = ahead, negative = behind).
    """
    from non_local_detector.analysis import (
        get_ahead_behind_distance,
        get_trajectory_data,
    )

    # Determine orientation column name
    orientation_cols = linear_position_info.columns[
        linear_position_info.columns.isin(["orientation", "head_orientation"])
    ]
    if len(orientation_cols) == 0:
        raise ValueError(
            "No orientation column found in linear_position_info. "
            "Expected 'orientation' or 'head_orientation'."
        )
    orientation_name = orientation_cols[0]

    # Slice data if requested
    if time_slice is not None:
        posterior_slice = posterior.isel(time=time_slice)
        pos_info_slice = linear_position_info.iloc[time_slice]
    else:
        posterior_slice = posterior
        pos_info_slice = linear_position_info

    traj_data = get_trajectory_data(
        posterior=posterior_slice,
        track_graph=track_graph,
        decoder=classifier,
        actual_projected_position=pos_info_slice[
            ["projected_x_position", "projected_y_position"]
        ],
        track_segment_id=pos_info_slice["track_segment_id"],
        actual_orientation=pos_info_slice[orientation_name],
    )

    return get_ahead_behind_distance(track_graph, *traj_data)


def get_edge_colors_and_order(
    config: DatasetConfig,
    track_graph: nx.Graph,
    track_graph_params: dict[str, Any],
) -> tuple[np.ndarray, list[tuple[int, int]], float | list[float]]:
    """
    Extract edge colors, order, and spacing from config and track graph params.

    Parameters
    ----------
    config : DatasetConfig
        Dataset configuration containing the edge colormap name.
    track_graph : networkx.Graph
        Track graph (used as fallback for edge order).
    track_graph_params : dict
        Dictionary containing 'linear_edge_order' and 'linear_edge_spacing'.

    Returns
    -------
    edge_colors : ndarray
        Array of RGB colors for each edge from the configured colormap.
    edge_order : list of tuple of int
        Ordered list of edge tuples (node1, node2) for linearization.
    edge_spacing : float or list of float
        Spacing between edges for linearization.
    """
    edge_order = track_graph_params.get("linear_edge_order", list(track_graph.edges))
    edge_spacing = track_graph_params.get("linear_edge_spacing", 0)
    edge_colors = np.array(plt.colormaps[config.edge_colormap].colors)
    return edge_colors, edge_order, edge_spacing


# =============================================================================
# FIGURE GENERATION
# =============================================================================
def generate_track_figure(
    track_graph: nx.Graph,
    position_info: pd.DataFrame,
    edge_order: list[tuple[int, int]],
    reward_well_nodes: tuple[int, ...],
    edge_colors: np.ndarray,
    position_names: tuple[str, str],
    output_name: str,
    output_dir: Path,
) -> None:
    """
    Generate 2D track graph visualization.

    Creates a figure showing the track graph overlaid on the animal's
    position trajectory, with edges colored according to the colormap
    and reward wells marked.

    Parameters
    ----------
    track_graph : networkx.Graph
        Track graph with nodes containing 'pos' attributes.
    position_info : pandas.DataFrame
        DataFrame containing position columns for trajectory overlay.
    edge_order : list of tuple of int
        Ordered list of edge tuples (node1, node2).
    reward_well_nodes : tuple of int
        Node indices that are reward wells.
    edge_colors : ndarray
        Array of colors for each edge.
    position_names : tuple of str
        Column names for (x, y) position in position_info.
    output_name : str
        Base name for output figure files.
    output_dir : Path
        Directory to save figures.
    """
    plot_2D_track_graph(
        track_graph=track_graph,
        position_info=position_info,
        edge_order=edge_order,
        reward_well_nodes=list(reward_well_nodes),
        edge_colors=edge_colors,
        position_names=position_names,
        scalebar_length=DEFAULT_SCALEBAR_LENGTH_CM,
        scalebar_label=f"{DEFAULT_SCALEBAR_LENGTH_CM} cm",
    )

    save_figure(output_name, output_dir)
    plt.close()


def generate_decode_figure(
    config: DatasetConfig,
    data: DecodingData,
    output_dir: Path,
) -> None:
    """
    Generate main decoding visualization figure with four panels.

    Creates a multi-panel figure showing:

    1. Posterior probability heatmap with actual position overlay
    2. Ahead/behind distance from actual position
    3. Running speed
    4. Multiunit firing rate

    Parameters
    ----------
    config : DatasetConfig
        Dataset configuration with time slice and output settings.
    data : DecodingData
        Container with decoding results, posterior, position info, etc.
    output_dir : Path
        Directory to save figures.

    Raises
    ------
    ValueError
        If time_slice indices are out of bounds for the data.
    """
    time_slice_ind = config.time_slice

    # Validate time slice against data dimensions
    validate_time_slice(time_slice_ind, len(data.results.time), "results.time")
    validate_time_slice(
        time_slice_ind, len(data.linear_position_info), "linear_position_info"
    )
    validate_time_slice(time_slice_ind, len(data.multiunit_rate), "multiunit_rate")

    # Extract visualization parameters
    edge_colors, edge_order, edge_spacing = get_edge_colors_and_order(
        config, data.track_graph, data.track_graph_params
    )

    # Get actual time values for the slice
    time_slice = slice(
        data.results.time.values[time_slice_ind.start],
        data.results.time.values[time_slice_ind.stop],
    )

    # Compute ahead/behind distance for the time slice
    ahead_behind_distance = compute_ahead_behind_distance(
        posterior=data.posterior,
        linear_position_info=data.linear_position_info,
        classifier=data.classifier,
        track_graph=data.track_graph,
        time_slice=time_slice_ind,
    )

    # Determine speed column name with fallback
    speed_col = _get_speed_column(data.linear_position_info, config.speed_name)

    # Validate sliced data is non-empty
    sliced_position_info = data.linear_position_info.iloc[time_slice_ind]
    validate_non_empty_slice(sliced_position_info, "linear_position_info")

    # Create figure with 4 panels
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(TWO_COLUMN / 3, PAGE_HEIGHT * 0.5),
        height_ratios=[4, 1, 1, 1],
        sharex=True,
        constrained_layout=True,
    )

    # Panel 1: Posterior with position overlay
    _plot_posterior_panel(
        ax=axes[0],
        posterior=data.posterior,
        linear_position_info=data.linear_position_info,
        track_graph=data.track_graph,
        time_slice_ind=time_slice_ind,
        edge_order=edge_order,
        edge_spacing=edge_spacing,
        edge_colors=edge_colors,
        reward_well_nodes=config.reward_well_nodes,
    )

    # Panel 2: Ahead/behind distance
    _plot_ahead_behind_panel(
        ax=axes[1],
        time_values=data.results.isel(time=time_slice_ind).time.values,
        ahead_behind_distance=ahead_behind_distance,
    )

    # Panel 3: Speed
    _plot_speed_panel(
        ax=axes[2],
        linear_position_info=data.linear_position_info,
        time_slice_ind=time_slice_ind,
        speed_col=speed_col,
    )

    # Panel 4: Firing rate
    _plot_firing_rate_panel(
        ax=axes[3],
        time_values=data.posterior.isel(time=time_slice_ind).time.values,
        multiunit_rate=data.multiunit_rate[time_slice_ind],
    )

    # Configure x-axis
    duration = time_slice.stop - time_slice.start
    axes[-1].set_xticks(
        (time_slice.start, time_slice.stop),
        (str(0.0), f"{duration:.1f}"),
    )
    axes[-1].set_xlabel("Time [s]")

    sns.despine(offset=5)

    save_figure(f"{config.output_prefix}_decode", output_dir)
    plt.close()


def _get_speed_column(linear_position_info: pd.DataFrame, preferred: str) -> str:
    """
    Get the speed column name, with fallbacks.

    Parameters
    ----------
    linear_position_info : pandas.DataFrame
        DataFrame containing position and speed columns.
    preferred : str
        Preferred column name to look for first.

    Returns
    -------
    str
        Name of the speed column found in the DataFrame.

    Raises
    ------
    ValueError
        If no speed column is found.
    """
    if preferred in linear_position_info.columns:
        return preferred
    if "speed" in linear_position_info.columns:
        return "speed"
    if "head_speed" in linear_position_info.columns:
        return "head_speed"
    raise ValueError(
        f"No speed column found. Expected one of: {preferred}, 'speed', 'head_speed'."
    )


def _plot_posterior_panel(
    ax: plt.Axes,
    posterior: xr.DataArray,
    linear_position_info: pd.DataFrame,
    track_graph: nx.Graph,
    time_slice_ind: slice,
    edge_order: list[tuple[int, int]],
    edge_spacing: float | list[float],
    edge_colors: np.ndarray,
    reward_well_nodes: tuple[int, ...],
) -> None:
    """
    Plot posterior probability with actual position overlay.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on.
    posterior : xarray.DataArray
        Decoded posterior probability with 'time' and 'position' dimensions.
    linear_position_info : pandas.DataFrame
        DataFrame with 'linear_position' column and time index.
    track_graph : networkx.Graph
        Track graph for 1D representation.
    time_slice_ind : slice
        Index slice for the time window to plot.
    edge_order : list of tuple of int
        Edge order for linearization.
    edge_spacing : float or list of float
        Spacing between edges.
    edge_colors : ndarray
        Colors for each edge.
    reward_well_nodes : tuple of int
        Node indices for reward wells.
    """
    posterior.isel(time=time_slice_ind).plot(
        x="time",
        y="position",
        robust=True,
        ax=ax,
        cmap="bone_r",
        add_colorbar=False,
        rasterized=True,
    )
    ax.scatter(
        linear_position_info.iloc[time_slice_ind].index,
        linear_position_info.iloc[time_slice_ind].linear_position,
        color="magenta",
        s=1,
        clip_on=False,
        rasterized=True,
    )
    ax.set_ylabel("Position [cm]")
    ax.set_xlabel("")

    plot_graph_as_1D(
        track_graph,
        ax=ax,
        edge_order=edge_order,
        edge_spacing=edge_spacing,
        reward_well_nodes=reward_well_nodes,
        other_axis_start=linear_position_info.iloc[time_slice_ind].index[-1] + 0.3,
        edge_colors=edge_colors,
    )


def _plot_ahead_behind_panel(
    ax: plt.Axes,
    time_values: np.ndarray,
    ahead_behind_distance: np.ndarray,
) -> None:
    """
    Plot ahead/behind distance panel.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on.
    time_values : ndarray
        Time values for x-axis.
    ahead_behind_distance : ndarray
        Distance values in cm (positive = ahead, negative = behind).
    """
    ax.plot(time_values, ahead_behind_distance, color="black")
    ax.axhline(0.0, color="magenta", linestyle="--")
    ax.set_ylabel("Dist. [cm]")
    ax.set_ylim(AHEAD_BEHIND_YLIM_CM)
    ax.text(
        time_values[0],
        AHEAD_BEHIND_YLIM_CM[1],
        "Ahead",
        color="grey",
        fontsize=8,
        ha="left",
        va="top",
    )
    ax.text(
        time_values[0],
        AHEAD_BEHIND_YLIM_CM[0],
        "Behind",
        color="grey",
        fontsize=8,
        ha="left",
        va="bottom",
    )


def _plot_speed_panel(
    ax: plt.Axes,
    linear_position_info: pd.DataFrame,
    time_slice_ind: slice,
    speed_col: str,
) -> None:
    """
    Plot speed panel.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on.
    linear_position_info : pandas.DataFrame
        DataFrame with speed column and time index.
    time_slice_ind : slice
        Index slice for the time window to plot.
    speed_col : str
        Name of the speed column in the DataFrame.
    """
    ax.fill_between(
        linear_position_info.iloc[time_slice_ind].index,
        linear_position_info.iloc[time_slice_ind][speed_col],
        color="lightgrey",
    )
    ax.set_ylabel("Speed\n[cm/s]")


def _plot_firing_rate_panel(
    ax: plt.Axes,
    time_values: np.ndarray,
    multiunit_rate: np.ndarray,
) -> None:
    """
    Plot multiunit firing rate panel.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on.
    time_values : ndarray
        Time values for x-axis.
    multiunit_rate : ndarray
        Firing rate values in spikes/s.
    """
    ax.fill_between(time_values, multiunit_rate, color="black")
    ax.set_ylabel("Firing rate\n[spikes/s]")


# =============================================================================
# BUZSAKI-SPECIFIC ANALYSES
# =============================================================================
def get_trial_time_slice(trials: pd.DataFrame, trial_type: str) -> slice:
    """
    Get time slice for a specific trial type.

    Parameters
    ----------
    trials : pandas.DataFrame
        Trial information with 'cooling state', 'start_time', 'stop_time'.
    trial_type : str
        Trial type to filter (e.g., "Pre-Cooling", "Cooling on").

    Returns
    -------
    slice
        Time slice from start to stop of the trial type.
    """
    trial_subset = trials.loc[trials["cooling state"] == trial_type]
    return slice(trial_subset.start_time.min(), trial_subset.stop_time.max())


def compute_log_power_ratio(
    power: xr.DataArray,
    position_info: pd.DataFrame,
    trials: pd.DataFrame,
    trial_types: tuple[str, str] = ("Pre-Cooling", "Cooling on"),
) -> np.ndarray:
    """
    Compute log power ratio between two trial types during running.

    Parameters
    ----------
    power : xarray.DataArray
        Spectral power with 'time' and 'frequency' dimensions.
    position_info : pandas.DataFrame
        Position info with 'speed' column, interpolated to power time points.
    trials : pandas.DataFrame
        Trial information with 'cooling state', 'start_time', 'stop_time'.
    trial_types : tuple of str, optional
        Two trial types to compare, by default ("Pre-Cooling", "Cooling on").
        Returns ratio of second minus first.

    Returns
    -------
    ndarray
        Log power ratio (dB) as a function of frequency.
    """
    log_power_by_trial_type = []
    for trial_type in trial_types:
        trial_time_slice = get_trial_time_slice(trials, trial_type)
        speed = position_info.loc[trial_time_slice, "speed"].to_numpy().squeeze()
        speed_filter = speed > SPEED_THRESHOLD_CM_PER_S

        log_power_by_trial_type.append(
            10.0
            * np.log10(power)
            .sel(time=trial_time_slice)
            .isel(time=speed_filter)
            .mean("time")
            .values
        )

    return log_power_by_trial_type[1] - log_power_by_trial_type[0]


def plot_power_ratio_fill(
    ax: plt.Axes,
    frequency: np.ndarray,
    power_ratio: np.ndarray,
) -> None:
    """
    Plot power ratio with positive/negative fill coloring.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on.
    frequency : ndarray
        Frequency values for x-axis.
    power_ratio : ndarray
        Power ratio values (positive = green, negative = red).
    """
    ax.plot(frequency, power_ratio)
    ax.fill_between(
        frequency,
        power_ratio,
        0.0,
        where=power_ratio > 0,
        color="green",
        alpha=0.3,
        step="mid",
    )
    ax.fill_between(
        frequency,
        power_ratio,
        0.0,
        where=power_ratio <= 0,
        color="red",
        alpha=0.3,
        step="mid",
    )


def add_power_ratio_labels(
    ax: plt.Axes,
    x_pos: float,
    y_pos_positive: float,
    y_pos_negative: float,
) -> None:
    """
    Add labels for power ratio comparison.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to add labels to.
    x_pos : float
        X position for labels.
    y_pos_positive : float
        Y position for "Cooling on > Pre-Cooling" label.
    y_pos_negative : float
        Y position for "Cooling on < Pre-Cooling" label.
    """
    ax.text(
        x_pos,
        y_pos_positive,
        "Cooling on\n> Pre-Cooling",
        color="green",
        fontsize=8,
        ha="right",
        va="bottom",
    )
    ax.text(
        x_pos,
        y_pos_negative,
        "Cooling on\n< Pre-Cooling",
        color="red",
        fontsize=8,
        ha="right",
        va="top",
    )


def compute_trial_stats(
    values: np.ndarray,
    speed: np.ndarray,
    mobility_type: str = "run",
) -> tuple[float, float]:
    """
    Compute mean and SEM for values filtered by mobility type.

    Parameters
    ----------
    values : ndarray
        Values to compute statistics for.
    speed : ndarray
        Speed values for filtering.
    mobility_type : str, optional
        Either "run" (speed > threshold) or "all", by default "run".

    Returns
    -------
    mean : float
        Mean of filtered values.
    sem : float
        Standard error of mean of filtered values.
    """
    if mobility_type == "run":
        speed_filter = speed > SPEED_THRESHOLD_CM_PER_S
    else:
        speed_filter = np.ones_like(values, dtype=bool)

    filtered_values = values[speed_filter]
    return filtered_values.mean(), scipy.stats.sem(filtered_values)


def interpolate_to_new_time(
    df: pd.DataFrame,
    new_time: np.ndarray,
    method: str = "linear",
) -> pd.DataFrame:
    """
    Interpolate dataframe to new time index.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with time as index.
    new_time : ndarray
        New time values to interpolate to.
    method : str, optional
        Interpolation method passed to pandas interpolate, by default "linear".

    Returns
    -------
    pandas.DataFrame
        DataFrame reindexed to new_time with interpolated values.
    """
    old_time = df.index
    new_index = pd.Index(np.unique(np.concatenate((old_time, new_time))), name="time")
    return (
        df.reindex(index=new_index).interpolate(method=method).reindex(index=new_time)
    )


def fetch_buzsaki_trials(nwb_file_copy_name: str) -> pd.DataFrame:
    """
    Fetch trial information from Buzsaki NWB file.

    Parameters
    ----------
    nwb_file_copy_name : str
        Name of the NWB file copy in the Spyglass database.

    Returns
    -------
    pandas.DataFrame
        Trial intervals with 'start_time', 'stop_time', and 'cooling state' columns.
    """
    import pynwb
    import spyglass.common as sgc

    nwb_path = (sgc.Nwbfile & {"nwb_file_name": nwb_file_copy_name}).fetch1(
        "nwb_file_abs_path"
    )

    with pynwb.NWBHDF5IO(nwb_path, mode="r") as io:
        nwbfile = io.read()
        trials = nwbfile.intervals["trials"].to_dataframe()

    return trials


def generate_distance_speed_figure(
    data: DecodingData,
    trials: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Generate distance and speed error bar plots for Buzsaki dataset.

    Creates a two-panel figure showing mean decode distance and decode speed
    across different cooling trial types, with 95% confidence intervals.

    Parameters
    ----------
    data : DecodingData
        Container with position_info and pre-computed ahead_behind_distance.
    trials : pandas.DataFrame
        Trial information with 'cooling state', 'start_time', 'stop_time'.
    output_dir : Path
        Directory to save figures.

    Raises
    ------
    ValueError
        If position_info or ahead_behind_distance is None in data.
    """
    if data.position_info is None:
        raise ValueError("position_info is required for distance/speed figure")
    if data.ahead_behind_distance is None:
        raise ValueError("ahead_behind_distance must be pre-computed")

    ahead_behind_distance_stats = pd.DataFrame(
        {"ahead_behind_distance": data.ahead_behind_distance},
        index=data.position_info.index,
    )

    mobility_types = ["run"]
    trial_types = ["Pre-Cooling", "Cooling on", "Cooling off", "Post-Cooling"]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(
            0.9 * TWO_COLUMN / 3,
            PAGE_HEIGHT * 0.25 + TWO_COLUMN / 3 * 0.3,
        ),
        constrained_layout=True,
    )

    # Panel 1: Distance error
    sns.despine(offset=5, ax=axes[0])
    for mobility_type in mobility_types:
        mean_by_trial_type = []
        sem_by_trial_type = []

        for trial_type in trial_types:
            trial_time_slice = get_trial_time_slice(trials, trial_type)
            dist = np.abs(
                ahead_behind_distance_stats.loc[trial_time_slice].to_numpy().squeeze()
            )
            speed = data.position_info.loc[trial_time_slice, "speed"].to_numpy().squeeze()

            mean_val, sem_val = compute_trial_stats(dist, speed, mobility_type)
            mean_by_trial_type.append(mean_val)
            sem_by_trial_type.append(sem_val)

        axes[0].errorbar(
            trial_types,
            mean_by_trial_type,
            yerr=CONFIDENCE_INTERVAL_Z * np.asarray(sem_by_trial_type),
            label=mobility_type,
            color="black",
        )

    axes[0].set_xticks([])
    axes[0].set_ylim(DECODE_DISTANCE_YLIM_CM)
    axes[0].set_yticks(list(range(DECODE_DISTANCE_YLIM_CM[0], DECODE_DISTANCE_YLIM_CM[1] + 1, 2)))
    axes[0].set_ylabel("Decode\nDistance\n[cm]")

    # Panel 2: Speed error
    sns.despine(offset=5, ax=axes[1])
    for mobility_type in mobility_types:
        mean_by_trial_type = []
        sem_by_trial_type = []

        for trial_type in trial_types:
            trial_time_slice = get_trial_time_slice(trials, trial_type)
            # Compute decode speed as gradient of ahead/behind distance
            # Convert from cm/sample to m/s
            decode_speed = (
                np.abs(
                    np.gradient(
                        ahead_behind_distance_stats.loc[trial_time_slice]
                        .to_numpy()
                        .squeeze(),
                        BUZSAKI_SAMPLING_INTERVAL_S,
                    )
                )
                * CM_TO_M
            )
            speed = data.position_info.loc[trial_time_slice, "speed"].to_numpy().squeeze()

            mean_val, sem_val = compute_trial_stats(decode_speed, speed, mobility_type)
            mean_by_trial_type.append(mean_val)
            sem_by_trial_type.append(sem_val)

        axes[1].errorbar(
            trial_types,
            mean_by_trial_type,
            yerr=CONFIDENCE_INTERVAL_Z * np.asarray(sem_by_trial_type),
            label=mobility_type,
            color="black",
        )

    axes[1].set_xticklabels(trial_types, rotation=45, horizontalalignment="right")
    axes[1].set_ylabel("Decode\nSpeed\n[m/s]")

    save_figure("figure5_buzsaki_distance_speed", output_dir)
    plt.close()


def generate_power_figure(
    data: DecodingData,
    trials: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Generate power spectrum analysis figure for NYU dataset.

    Creates a two-panel figure comparing spectral power between Pre-Cooling
    and Cooling-on conditions for multiunit activity and decode distance.

    Parameters
    ----------
    data : DecodingData
        Container with multiunit_rate, position_info, and pre-computed
        ahead_behind_distance.
    trials : pandas.DataFrame
        Trial information with 'cooling state', 'start_time', 'stop_time'.
    output_dir : Path
        Directory to save figures.

    Raises
    ------
    ValueError
        If position_info or ahead_behind_distance is None in data.
    """
    from spectral_connectivity import multitaper_connectivity

    if data.position_info is None:
        raise ValueError("position_info is required for power figure")
    if data.ahead_behind_distance is None:
        raise ValueError("ahead_behind_distance must be pre-computed")

    fig, axes = plt.subplots(
        2,
        1,
        sharex=True,
        constrained_layout=True,
        figsize=(
            0.9 * TWO_COLUMN / 3,
            PAGE_HEIGHT * 0.25 + TWO_COLUMN / 3 * 0.3,
        ),
    )

    start_time = data.position_info.index[0]

    # Panel 1: Multiunit power
    multiunit_power = multitaper_connectivity(
        data.multiunit_rate[:, None],
        sampling_frequency=SPECTRAL_SAMPLING_FREQUENCY_HZ,
        time_halfbandwidth_product=SPECTRAL_TIME_HALFBANDWIDTH,
        time_window_duration=SPECTRAL_TIME_WINDOW_S,
        start_time=start_time,
        method="power",
    ).squeeze()
    interp_pos_info = interpolate_to_new_time(data.position_info, multiunit_power.time)

    multiunit_power_ratio = compute_log_power_ratio(
        multiunit_power, interp_pos_info, trials
    )

    plot_power_ratio_fill(axes[0], multiunit_power.frequency.values, multiunit_power_ratio)
    add_power_ratio_labels(axes[0], x_pos=13, y_pos_positive=3, y_pos_negative=-3)
    axes[0].set_xlim((5, 13))
    axes[0].set_ylim((-5, 5))
    axes[0].axhline(0.0, color="black", linestyle="--")
    axes[0].set_ylabel("Power\nChange [dB]")
    axes[0].set_title("Multiunit", fontsize=10)

    # Panel 2: Ahead/behind power
    ahead_behind_power = multitaper_connectivity(
        data.ahead_behind_distance[:, None],
        sampling_frequency=SPECTRAL_SAMPLING_FREQUENCY_HZ,
        time_halfbandwidth_product=SPECTRAL_TIME_HALFBANDWIDTH,
        time_window_duration=SPECTRAL_TIME_WINDOW_S,
        start_time=start_time,
        method="power",
    ).squeeze()
    interp_pos_info = interpolate_to_new_time(data.position_info, ahead_behind_power.time)

    ahead_behind_power_ratio = compute_log_power_ratio(
        ahead_behind_power, interp_pos_info, trials
    )

    plot_power_ratio_fill(
        axes[1], ahead_behind_power.frequency.values, ahead_behind_power_ratio
    )
    add_power_ratio_labels(axes[1], x_pos=13, y_pos_positive=2, y_pos_negative=-2)
    axes[1].set_xlim((5, 13))
    axes[1].set_ylim((-3, 3))
    axes[1].axhline(0.0, color="black", linestyle="--")
    axes[1].set_xlabel("Frequency [Hz]")
    axes[1].set_ylabel("Power\nChange [dB]")
    axes[1].set_title("Decode Distance", fontsize=10)
    axes[1].set_xticks([5, 7, 9, 11, 13])

    sns.despine(offset=5)

    save_figure("figure5_buzsaki_power", output_dir)
    plt.close()


# =============================================================================
# PIPELINE FUNCTIONS
# =============================================================================
def generate_ucsf_figures(output_dir: Path) -> None:
    """
    Generate all figures for UCSF (Frank Lab) dataset.

    Fetches clusterless decoding data and generates track and decode figures.

    Parameters
    ----------
    output_dir : Path
        Directory to save figures.

    Raises
    ------
    ValueError
        If configuration is invalid or data cannot be fetched.
    """
    print("Generating UCSF figures...")
    config = UCSF_CONFIG
    config.validate()

    print("  Fetching clusterless decoding data...")
    data = fetch_clusterless_decoding_data(config)

    edge_colors, edge_order, _ = get_edge_colors_and_order(
        config, data.track_graph, data.track_graph_params
    )

    # Track figure
    print("  Generating track figure...")
    generate_track_figure(
        track_graph=data.track_graph,
        position_info=data.linear_position_info,
        edge_order=edge_order,
        reward_well_nodes=config.reward_well_nodes,
        edge_colors=edge_colors,
        position_names=(config.position_x_name, config.position_y_name),
        output_name=f"{config.output_prefix}_track",
        output_dir=output_dir,
    )

    # Decode figure
    print("  Generating decode figure...")
    generate_decode_figure(config, data, output_dir)

    print("UCSF figures complete!")


def generate_nyu_figures(output_dir: Path) -> None:
    """
    Generate all figures for NYU (Buzsaki Lab) dataset.

    Fetches sorted spikes decoding data and generates track, decode,
    distance/speed, and power spectrum figures.

    Parameters
    ----------
    output_dir : Path
        Directory to save figures.

    Raises
    ------
    ValueError
        If configuration is invalid or data cannot be fetched.
    """
    print("Generating NYU figures...")
    config = NYU_CONFIG
    config.validate()

    print("  Fetching sorted spikes decoding data...")
    data = fetch_sorted_spikes_decoding_data(config)

    edge_colors, edge_order, _ = get_edge_colors_and_order(
        config, data.track_graph, data.track_graph_params
    )

    # Track figure
    print("  Generating track figure...")
    generate_track_figure(
        track_graph=data.track_graph,
        position_info=data.linear_position_info,
        edge_order=edge_order,
        reward_well_nodes=config.reward_well_nodes,
        edge_colors=edge_colors,
        position_names=(config.position_x_name, config.position_y_name),
        output_name=f"{config.output_prefix}_track",
        output_dir=output_dir,
    )

    # Decode figure
    print("  Generating decode figure...")
    generate_decode_figure(config, data, output_dir)

    # NYU-specific analyses require ahead_behind_distance computed once
    print("  Computing ahead/behind distance for full dataset...")
    ahead_behind_distance = compute_ahead_behind_distance(
        posterior=data.posterior,
        linear_position_info=data.linear_position_info,
        classifier=data.classifier,
        track_graph=data.track_graph,
    )
    # Update data with computed ahead_behind_distance
    data = replace(data, ahead_behind_distance=ahead_behind_distance)

    print("  Fetching trial information...")
    trials = fetch_buzsaki_trials(data.nwb_file_copy_name)

    print("  Generating distance/speed figure...")
    generate_distance_speed_figure(data, trials, output_dir)

    print("  Generating power figure...")
    generate_power_figure(data, trials, output_dir)

    print("NYU figures complete!")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
def main() -> None:
    """
    Generate Figure 5 panels with command-line interface.

    Parses command-line arguments to select dataset(s) and output directory,
    then generates the appropriate figures.
    """
    parser = argparse.ArgumentParser(
        description="Generate Figure 5 panels for Spyglass paper",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        choices=["ucsf", "nyu", "all"],
        default="all",
        help="Which dataset to generate figures for (ucsf=clusterless, nyu=sorted)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: same directory as script)",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="GPU device ID for JAX (default: use JAX default)",
    )

    args = parser.parse_args()

    # Suppress warnings for cleaner output during figure generation
    import warnings

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    # Set GPU device if specified
    if args.gpu is not None:
        import jax

        device = jax.devices()[args.gpu]
        jax.config.update("jax_default_device", device)
        print(f"Using GPU: {device}")

    # Determine output directory
    if args.output_dir is None:
        output_dir = Path(__file__).parent
    else:
        output_dir = args.output_dir

    # Set figure defaults
    set_figure_defaults()

    # Generate figures
    if args.dataset in ("ucsf", "all"):
        generate_ucsf_figures(output_dir)

    if args.dataset in ("nyu", "all"):
        generate_nyu_figures(output_dir)

    print(f"\nAll figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
