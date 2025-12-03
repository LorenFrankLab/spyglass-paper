"""Tests for paper_plotting module."""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pytest

from src.paper_plotting import (
    add_scalebar,
    plot_2D_track_graph,
    plot_graph_as_1D,
    save_figure,
    set_figure_defaults,
)


class TestSetFigureDefaults:
    """Test set_figure_defaults function."""

    def test_sets_rcparams(self):
        set_figure_defaults()
        assert plt.rcParams["pdf.fonttype"] == 42
        assert plt.rcParams["ps.fonttype"] == 42
        assert plt.rcParams["axes.labelsize"] == 9

    def test_does_not_raise(self):
        set_figure_defaults()


class TestSaveFigure:
    """Test save_figure function."""

    def test_creates_pdf_and_png(self, tmp_path):
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])

        save_figure("test_figure", output_dir=tmp_path)

        assert (tmp_path / "test_figure.pdf").exists()
        assert (tmp_path / "test_figure.png").exists()
        plt.close(fig)

    def test_creates_output_directory(self, tmp_path):
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])

        new_dir = tmp_path / "new_subdir"
        save_figure("test_figure", output_dir=new_dir)

        assert new_dir.exists()
        assert (new_dir / "test_figure.pdf").exists()
        plt.close(fig)


class TestAddScalebar:
    """Test add_scalebar function."""

    def test_adds_patch_to_axes(self):
        fig, ax = plt.subplots()
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)

        initial_patches = len(ax.patches)
        add_scalebar(ax, length=20, label="20 cm")

        assert len(ax.patches) == initial_patches + 1
        plt.close(fig)

    def test_adds_text_to_axes(self):
        fig, ax = plt.subplots()
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)

        initial_texts = len(ax.texts)
        add_scalebar(ax, length=20, label="20 cm")

        assert len(ax.texts) == initial_texts + 1
        plt.close(fig)


def _create_simple_track_graph():
    """Create a simple track graph for testing."""
    G = nx.Graph()
    G.add_node(0, pos=(0, 0))
    G.add_node(1, pos=(10, 0))
    G.add_node(2, pos=(10, 10))
    G.add_edge(0, 1, distance=10.0)
    G.add_edge(1, 2, distance=10.0)
    return G


class TestPlotGraphAs1D:
    """Test plot_graph_as_1D function."""

    def test_plots_edges(self):
        fig, ax = plt.subplots()
        track_graph = _create_simple_track_graph()

        plot_graph_as_1D(track_graph, ax=ax)

        assert len(ax.lines) == 2
        plt.close(fig)

    def test_plots_reward_wells(self):
        fig, ax = plt.subplots()
        track_graph = _create_simple_track_graph()

        plot_graph_as_1D(track_graph, ax=ax, reward_well_nodes=[0, 2])

        assert len(ax.collections) == 2
        plt.close(fig)

    def test_custom_edge_order(self):
        fig, ax = plt.subplots()
        track_graph = _create_simple_track_graph()

        plot_graph_as_1D(track_graph, ax=ax, edge_order=[(1, 2), (0, 1)])

        assert len(ax.lines) == 2
        plt.close(fig)


class TestPlot2DTrackGraph:
    """Test plot_2D_track_graph function."""

    def test_returns_figure_and_axes(self):
        track_graph = _create_simple_track_graph()
        position_info = pd.DataFrame(
            {
                "position_x": [0, 5, 10, 10],
                "position_y": [0, 0, 0, 5],
            }
        )

        fig, ax = plot_2D_track_graph(track_graph, position_info)

        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_plots_trajectory(self):
        track_graph = _create_simple_track_graph()
        position_info = pd.DataFrame(
            {
                "position_x": [0, 5, 10, 10],
                "position_y": [0, 0, 0, 5],
            }
        )

        fig, ax = plot_2D_track_graph(track_graph, position_info)

        assert len(ax.lines) >= 1
        plt.close(fig)

    def test_plots_reward_wells(self):
        track_graph = _create_simple_track_graph()
        position_info = pd.DataFrame(
            {
                "position_x": [0, 5, 10],
                "position_y": [0, 0, 0],
            }
        )

        fig, ax = plot_2D_track_graph(
            track_graph, position_info, reward_well_nodes=[0, 2]
        )

        assert len(ax.collections) >= 2
        plt.close(fig)
