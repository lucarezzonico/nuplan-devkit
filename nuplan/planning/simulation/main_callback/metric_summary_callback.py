import logging
import math
import time
from collections import defaultdict
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Union, Tuple, Optional

from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path as MplPath
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

from dataclasses import dataclass
import collections
import statistics

import matplotlib.cm as cmap
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import PercentFormatter
from tqdm import tqdm

from nuplan.planning.metrics.metric_dataframe import MetricStatisticsDataFrame
from nuplan.planning.nuboard.tabs.config.histogram_tab_config import (
    HistogramConstantConfig,
    HistogramTabFigureStyleConfig,
    HistogramTabMatPlotLibPlotStyleConfig,
)
from nuplan.planning.nuboard.utils.nuboard_histogram_utils import (
    aggregate_metric_aggregator_dataframe_histogram_data,
    aggregate_metric_statistics_dataframe_histogram_data,
    compute_histogram_edges,
    get_histogram_plot_x_range,
)
from nuplan.planning.nuboard.utils.utils import metric_aggregator_reader, metric_statistics_reader
from nuplan.planning.simulation.main_callback.abstract_main_callback import AbstractMainCallback

METRIC_DATAFRAME_TYPE = Dict[str, Union[MetricStatisticsDataFrame, pd.DataFrame]]

logger = logging.getLogger(__name__)



def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):

        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return MplPath(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)
            # self.set_thetagrids(np.degrees(theta), labels, rotation=30)
            
        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars, radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=MplPath.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5) + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


def example_data():
    data = [
        ['Metric 1', 'Metric 2', 'Metric 3', 'Metric 4', 'Metric 5'],
        ('Model Comparison based on AVAILABLE Metrics', [
            # [1.0, 1.0, 1.0, 1.0, 1.0],
            [0.8, 0.8, 0.8, 0.8, 0.8],
            [0.02, 0.01, 0.07, 0.01, 0.21],
            [0.01, 0.01, 0.02, 0.71, 0.74]]),
    ]
    return data

@dataclass
class IdxUnit:
    idx: int
    unit: str
    
@dataclass
class OLStats:
    """
    Parameters for RadarPlot.
    """
    planner_miss_rate_horizon_3: IdxUnit
    planner_miss_rate_horizon_5: IdxUnit
    planner_miss_rate_horizon_8: IdxUnit
    planner_miss_rate_within_bound: IdxUnit
    planner_expert_ADE_horizon_3: IdxUnit
    planner_expert_ADE_horizon_5: IdxUnit
    planner_expert_ADE_horizon_8: IdxUnit
    planner_expert_average_l2_error_within_bound: IdxUnit
    avg_planner_expert_ADE_over_all_horizons: IdxUnit
    planner_expert_FDE_horizon_3: IdxUnit
    planner_expert_FDE_horizon_5: IdxUnit
    planner_expert_FDE_horizon_8: IdxUnit
    planner_expert_final_l2_error_within_bound: IdxUnit
    avg_planner_expert_FDE_over_all_horizons: IdxUnit
    planner_expert_FHE_horizon_3: IdxUnit
    planner_expert_FHE_horizon_5: IdxUnit
    planner_expert_FHE_horizon_8: IdxUnit
    planner_expert_final_heading_error_within_bound: IdxUnit
    avg_planner_expert_FHE_over_all_horizons: IdxUnit
    planner_expert_AHE_horizon_3: IdxUnit
    planner_expert_AHE_horizon_5: IdxUnit
    planner_expert_AHE_horizon_8: IdxUnit
    planner_expert_average_heading_error_within_bound: IdxUnit
    avg_planner_expert_AHE_over_all_horizons: IdxUnit
    all: IdxUnit    
   
@dataclass
class CLStats:
    """
    Parameters for RadarPlot.
    """
    ego_is_comfortable: IdxUnit
    all: IdxUnit

class MetricSummaryCallback(AbstractMainCallback):
    """Callback to render histograms for metrics and metric aggregator."""

    def __init__(
        self,
        metric_save_path: str,
        metric_aggregator_save_path: str,
        summary_output_path: str,
        pdf_file_name: str,
        selected_ol_stats: List[str],
        selected_cl_stats: List[str],
        ol_stats: OLStats,
        cl_stats: CLStats,
        num_bins: int = 20,
        model_name: str = "Model",
        max_average_l2_error_threshold: float = 8.0,       # [m]
        max_final_l2_error_threshold: float = 8.0,         # [m]
        max_average_heading_error_threshold: float = 0.8,  # [rad]
        max_final_heading_error_threshold: float = 0.8,    # [rad]
    ):
        """Callback to handle metric files at the end of process."""
        self._metric_save_path = Path(metric_save_path)
        self._metric_aggregator_save_path = Path(metric_aggregator_save_path)
        self._summary_output_path = Path(summary_output_path)
        self._summary_output_path.mkdir(parents=True, exist_ok=True)
        self._pdf_file_name = pdf_file_name
        self._num_bins = num_bins

        self._color_index = 0
        color_palette = cmap.get_cmap('Set1').colors + cmap.get_cmap('Set2').colors + cmap.get_cmap('Set3').colors
        self._color_choices = [mcolors.rgb2hex(color) for color in color_palette]

        self._metric_aggregator_dataframes: Dict[str, pd.DataFrame] = {}
        self._metric_statistics_dataframes: Dict[str, MetricStatisticsDataFrame] = {}
        
        self.model_name = model_name
        self.max_average_l2_error_threshold = max_average_l2_error_threshold            # [m]
        self.max_final_l2_error_threshold = max_final_l2_error_threshold                # [m]
        self.max_average_heading_error_threshold = max_average_heading_error_threshold  # [rad]
        self.max_final_heading_error_threshold = max_final_heading_error_threshold      # [rad]
        self.selected_ol_stats = selected_ol_stats
        self.selected_cl_stats = selected_cl_stats
        self.ol_stats = ol_stats
        self.cl_stats = cl_stats

    @staticmethod
    def _read_metric_parquet_files(
        metric_save_path: Path, metric_reader: Callable[[Path], Any]
    ) -> METRIC_DATAFRAME_TYPE:
        """
        Read metric parquet files with different readers.
        :param metric_save_path: Metric save path.
        :param metric_reader: Metric reader to read metric parquet files.
        :return A dictionary of {file_index: {file_name: MetricStatisticsDataFrame or pandas dataframe}}.
        """
        metric_dataframes: Dict[str, Union[MetricStatisticsDataFrame, pd.DataFrame]] = defaultdict()
        metric_file = metric_save_path.rglob("*.parquet")
        for file_index, file in enumerate(metric_file):
            try:
                if file.is_dir():
                    continue
                data_frame = metric_reader(file)
                metric_dataframes[file.stem] = data_frame
            except (FileNotFoundError, Exception):
                # Ignore the file
                pass
        return metric_dataframes

    def _aggregate_metric_statistic_histogram_data(self) -> HistogramConstantConfig.HistogramDataType:
        """
        Aggregate metric statistic histogram data.
        :return A dictionary of metric names and their aggregated data.
        """
        data: HistogramConstantConfig.HistogramDataType = defaultdict(list)
        for dataframe_filename, dataframe in self._metric_statistics_dataframes.items():
            histogram_data_list = aggregate_metric_statistics_dataframe_histogram_data(
                metric_statistics_dataframe=dataframe,
                metric_statistics_dataframe_index=0,
                metric_choices=[],
                scenario_types=None,
            )
            if histogram_data_list:
                data[dataframe.metric_statistic_name] += histogram_data_list

        return data

    def _aggregate_scenario_type_score_histogram_data(self) -> HistogramConstantConfig.HistogramDataType:
        """
        Aggregate scenario type score histogram data.
        :return A dictionary of scenario type metric name and their scenario type scores.
        """
        data: HistogramConstantConfig.HistogramDataType = defaultdict(list)
        for index, (dataframe_filename, dataframe) in enumerate(self._metric_aggregator_dataframes.items()):
            histogram_data_list = aggregate_metric_aggregator_dataframe_histogram_data(
                metric_aggregator_dataframe=dataframe,
                metric_aggregator_dataframe_index=index,
                scenario_types=['all'],
                dataframe_file_name=dataframe_filename,
            )
            if histogram_data_list:
                data[
                    f'{HistogramConstantConfig.SCENARIO_TYPE_SCORE_HISTOGRAM_NAME}_{dataframe_filename}'
                ] += histogram_data_list

        return data

    def _assign_planner_colors(self) -> Dict[str, Any]:
        """
        Assign colors to planners.
        :return A dictionary of planner and colors.
        """
        planner_color_maps = {}
        for dataframe_filename, dataframe in self._metric_statistics_dataframes.items():
            planner_names = dataframe.planner_names
            for planner_name in planner_names:
                if planner_name not in planner_color_maps:
                    planner_color_maps[planner_name] = self._color_choices[self._color_index % len(self._color_choices)]
                    self._color_index += 1

        return planner_color_maps

    def _save_to_pdf(self, matplotlib_plots: List[Any]) -> None:
        """
        Save a list of matplotlib plots to a pdf file.
        :param matplotlib_plots: A list of matplotlib plots.
        """
        file_name = self._summary_output_path / self._pdf_file_name
        pp = PdfPages(file_name)
        # Save to pdf
        for fig in matplotlib_plots[::-1]:
            fig.savefig(pp, format='pdf')
        pp.close()
        
    def _save_to_png(self, matplotlib_plots: List[Any]) -> None:
        """
        Save a list of matplotlib plots to png files.
        :param matplotlib_plots: A list of matplotlib plots.
        """
        png_path = f"{self._summary_output_path}/png_figs"
        if not os.path.exists(png_path): os.makedirs(png_path)
        
        for i, fig in enumerate(matplotlib_plots[::-1]): # -1 specifies the step, in this case it goes backwards, so the entire list is returned in reverse order.
            file_name = f"{png_path}/fig_{i}.png"
            fig.savefig(file_name, format='png')
        
    @staticmethod
    def _render_ax_hist(
        ax: Any,
        x_values: npt.NDArray[np.float64],
        x_axis_label: str,
        y_axis_label: str,
        bins: npt.NDArray[np.float64],
        label: str,
        color: str,
        ax_title: str,
    ) -> None:
        """
        Render axis with histogram bins.
        :param ax: Matplotlib axis.
        :param x_values: An array of histogram x-axis values.
        :param x_axis_label: Label in the x-axis.
        :param y_axis_label: Label in the y-axis.
        :param bins: An array of histogram bins.
        :param label: Legend name for the bins.
        :param color: Color for the bins.
        :param ax_title: Axis title.
        """
        ax.hist(x=x_values, bins=bins, label=label, color=color, weights=np.ones(len(x_values)) / len(x_values))
        ax.set_xlabel(x_axis_label, fontsize=HistogramTabMatPlotLibPlotStyleConfig.x_axis_label_size)
        ax.set_ylabel(y_axis_label, fontsize=HistogramTabMatPlotLibPlotStyleConfig.y_axis_label_size)
        ax.set_title(ax_title, fontsize=HistogramTabMatPlotLibPlotStyleConfig.axis_title_size)
        ax.set_ylim(ymin=0)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.tick_params(axis='both', which='major', labelsize=HistogramTabMatPlotLibPlotStyleConfig.axis_ticker_size)
        ax.legend(fontsize=HistogramTabMatPlotLibPlotStyleConfig.legend_font_size)

    @staticmethod
    def _render_ax_bar_hist(
        ax: Any,
        x_values: Union[npt.NDArray[np.float64], List[str]],
        x_axis_label: str,
        y_axis_label: str,
        x_range: List[str],
        label: str,
        color: str,
        ax_title: str,
    ) -> None:
        """
        Render axis with bar histogram.
        :param ax: Matplotlib axis.
        :param x_values: An array of histogram x-axis values.
        :param x_axis_label: Label in the x-axis.
        :param y_axis_label: Label in the y-axis.
        :param x_range: A list of histogram category names.
        :param label: Legend name for the bins.
        :param color: Color for the bins.
        :param ax_title: Axis title.
        """
        value_categories = {key: 0.0 for key in x_range}
        for value in x_values:
            value_categories[str(value)] += 1.0

        category_names = list(value_categories.keys())
        category_values: List[float] = list(value_categories.values())
        num_scenarios = sum(category_values)
        if num_scenarios != 0:
            category_values = [(value / num_scenarios) * 100 for value in category_values]
            category_values = np.round(category_values, decimals=HistogramTabFigureStyleConfig.decimal_places)
        ax.bar(category_names, category_values, label=label, color=color)
        ax.set_xlabel(x_axis_label, fontsize=HistogramTabMatPlotLibPlotStyleConfig.x_axis_label_size)
        ax.set_ylabel(y_axis_label, fontsize=HistogramTabMatPlotLibPlotStyleConfig.y_axis_label_size)
        ax.set_title(ax_title, fontsize=HistogramTabMatPlotLibPlotStyleConfig.axis_title_size)
        ax.set_ylim(ymin=0)
        ax.tick_params(axis='both', which='major', labelsize=HistogramTabMatPlotLibPlotStyleConfig.axis_ticker_size)
        ax.legend(fontsize=HistogramTabMatPlotLibPlotStyleConfig.legend_font_size)

    def _draw_histogram_plots(
        self,
        planner_color_maps: Dict[str, Any],
        histogram_data_dict: HistogramConstantConfig.HistogramDataType,
        histogram_edges: HistogramConstantConfig.HistogramEdgesDataType,
        n_cols: int = 2,
    ) -> list[plt.Figure]:
        """
        :param planner_color_maps: Color maps from planner names.
        :param histogram_data_dict: A dictionary of histogram data.
        :param histogram_edges: A dictionary of histogram edges (bins) data.
        :param n_cols: Number of columns in subplot.
        """
        matplotlib_plots = []
        for histogram_title, histogram_data_list in tqdm(histogram_data_dict.items(), desc='Rendering histograms'):
            for histogram_data in histogram_data_list:
                # Get planner color
                color = planner_color_maps.get(histogram_data.planner_name, None)
                if not color:
                    planner_color_maps[histogram_data.planner_name] = self._color_choices[
                        self._color_index % len(self._color_choices)
                    ]
                    color = planner_color_maps.get(histogram_data.planner_name)
                    self._color_index += 1

                n_rows = math.ceil(len(histogram_data.statistics) / n_cols)
                fig_size = min(max(6, len(histogram_data.statistics) // 5 * 5), 24)
                fig, axs = plt.subplots(n_rows, n_cols, figsize=(fig_size, fig_size))
                flatten_axs = axs.flatten()
                fig.suptitle(histogram_title, fontsize=HistogramTabMatPlotLibPlotStyleConfig.main_title_size)

                for index, (statistic_name, statistic) in enumerate(histogram_data.statistics.items()):
                    unit = statistic.unit
                    bins: npt.NDArray[np.float64] = np.unique(
                        histogram_edges[histogram_title].get(statistic_name, None)
                    )
                    assert bins is not None, f"Count edge data for {statistic_name} cannot be None!"
                    x_range = get_histogram_plot_x_range(unit=unit, data=bins)
                    values = np.round(statistic.values, HistogramTabFigureStyleConfig.decimal_places)
                    if unit in ["count"]:
                        self._render_ax_bar_hist(
                            ax=flatten_axs[index],
                            x_values=values,
                            x_range=x_range,
                            x_axis_label=unit,
                            y_axis_label='Frequency (%)',
                            label=histogram_data.planner_name,
                            color=color,
                            ax_title=statistic_name,
                        )
                    elif unit in ["bool", "boolean"]:
                        values = ["True" if value else "False" for value in values]
                        self._render_ax_bar_hist(
                            ax=flatten_axs[index],
                            x_values=values,
                            x_range=x_range,
                            x_axis_label=unit,
                            y_axis_label='Frequency (%)',
                            label=histogram_data.planner_name,
                            color=color,
                            ax_title=statistic_name,
                        )
                    else:
                        self._render_ax_hist(
                            ax=flatten_axs[index],
                            x_values=values,
                            bins=bins,
                            x_axis_label=unit,
                            y_axis_label='Frequency (%)',
                            label=histogram_data.planner_name,
                            color=color,
                            ax_title=statistic_name,
                        )

                if n_rows * n_cols != len(histogram_data.statistics.values()):
                    flatten_axs[-1].set_axis_off()
                plt.tight_layout()
                matplotlib_plots.append(fig)

        # self._save_to_pdf(matplotlib_plots=matplotlib_plots)
        return matplotlib_plots
    
    
    @staticmethod
    def _get_count_or_bool_hist_values(
        x_values: Union[npt.NDArray[np.float64], List[str]],
        x_range: List[str],
    ) -> None:
        """
        Render axis with bar histogram.
        :param x_values: An array of histogram x-axis values.
        :param x_range: A list of histogram category names.
        """
        value_categories = {key: 0.0 for key in x_range}
        # count for how many scenarios the metric was False or True
        for value in x_values:
            value_categories[str(value)] += 1.0

        category_names = list(value_categories.keys())
        category_values: List[float] = list(value_categories.values())
        num_scenarios = sum(category_values)
        if num_scenarios != 0:
            category_values = [(value / num_scenarios) * 100 for value in category_values] 
            category_values = np.round(category_values, decimals=HistogramTabFigureStyleConfig.decimal_places)
        
        return category_names, category_values
    
    
    def _draw_radar_plots(
        self,
        planner_color_maps: Dict[str, Any],
        histogram_data_dict: HistogramConstantConfig.HistogramDataType,
        histogram_edges: HistogramConstantConfig.HistogramEdgesDataType,
        n_cols: int = 2,
    ) -> list[plt.Figure]:
        """
        :param planner_color_maps: Color maps from planner names.
        :param histogram_data_dict: A dictionary of histogram data.
        :param histogram_edges: A dictionary of histogram edges (bins) data.
        :param n_cols: Number of columns in subplot.
        """
        matplotlib_plots = []
        radar_plot_metrics = []
        radar_plot_values_mean_over_all_scenarios = []
        
        simulation_type = str(list(histogram_data_dict.keys())[-1])
        if "open_loop" in simulation_type:
            simulation_type = "Open-Loop"
        elif "closed_loop_nonreactive_agents" in simulation_type:
            simulation_type = "Closed-Loop Non-Reacive Agents"
        elif "closed_loop_reactive_agents" in simulation_type:
            simulation_type = "Closed-Loop Reacive Agents"
        else:
            raise ValueError(f"Simulation type {simulation_type} is not supported!")
            
        num_scenarios = len(list(histogram_data_dict.items())[-1][1][0].statistics["all"].scenarios)
        num_statistics = len(self.ol_stats.items()) if simulation_type == "Open-Loop" else len(self.cl_stats.items())
        
        radar_plot_scenario_statistics = np.zeros((num_scenarios, num_statistics))
        statistic_global_idx = 0
        
        scenario_names = list(histogram_data_dict.items())[-1][1][0].statistics["all"].scenarios
        scenario_types_keys = list(list(histogram_data_dict.items())[-1][1][0].statistics.keys())
        scenarios_by_type = list(list(histogram_data_dict.items())[-1][1][0].statistics.items())[1:]
        scenario_types_of_each_scenario = [item[0] for s in scenario_names for item in scenarios_by_type if s in item[1].scenarios]

        for histogram_title, histogram_data_list in tqdm(histogram_data_dict.items(), desc='Rendering radar plots'):
            for histogram_data in histogram_data_list:                
                for index, (statistic_name, statistic) in enumerate(histogram_data.statistics.items()):
                    unit = statistic.unit
                    bins: npt.NDArray[np.float64] = np.unique(
                        histogram_edges[histogram_title].get(statistic_name, None)
                    )
                    assert bins is not None, f"Count edge data for {statistic_name} cannot be None!"
                    x_range = get_histogram_plot_x_range(unit=unit, data=bins)
                    values_per_scenario = np.array(np.round(statistic.values, HistogramTabFigureStyleConfig.decimal_places)) # [num_scenarios]
                    # if unit in ["count"]:
                    #     category_names, category_values = self._get_count_or_bool_hist_values(x_values=values,
                    #                                                                           x_range=x_range)
                    #     # what is count?
                    # elif unit in ["bool", "boolean"]:
                    #     pass
                    # elif unit in ["meter"]:
                    #     pass
                    # else:
                    #     pass
                    
                    if "ADE" in statistic_name and unit in ["meter"]:
                        values_per_scenario = np.clip(1-values_per_scenario/self.max_average_l2_error_threshold, 0, None)
                    elif "FDE" in statistic_name and unit in ["meter"]:
                        values_per_scenario = np.clip(1-values_per_scenario/self.max_final_l2_error_threshold, 0, None)
                    elif "AHE" in statistic_name and unit in ["radian"]:
                        values_per_scenario = np.clip(1-values_per_scenario/self.max_average_heading_error_threshold, 0, None)
                    elif "FHE" in statistic_name and unit in ["radian"]:
                        values_per_scenario = np.clip(1-values_per_scenario/self.max_final_heading_error_threshold, 0, None)
                    # elif "all" == statistic_name and unit in ["scores"]:
                    #     values_per_scenario = values_per_scenario
                    # elif  unit in ["scores"]: # every scenario's score separately
                    #     values_per_scenario = values_per_scenario
                        
                    values_mean_over_all_scenarios = np.mean(values_per_scenario) # average on all scenarios values [num_scenarios] -> [1]
                    
                    stat_name = statistic_name
                    stat_name = stat_name.replace("planner_expert_", "")
                    stat_name = stat_name.replace("planner_", "")
                    stat_name = stat_name.replace("average_l2_error", "ADE")
                    stat_name = stat_name.replace("final_l2_error", "FDE")
                    stat_name = stat_name.replace("average_heading_error", "AHE")
                    stat_name = stat_name.replace("final_heading_error", "FHE")
                    radar_plot_metrics.append(stat_name)
                    
                    radar_plot_values_mean_over_all_scenarios.append(values_mean_over_all_scenarios)

                    radar_plot_scenario_statistics[:, statistic_global_idx] = values_per_scenario

                    statistic_global_idx += 1
                    
                    # stop after the "all" statistic, avoid scenario specific statistics
                    if statistic_global_idx == num_statistics:
                        break
        
        if simulation_type == "Open-Loop":
            selected_statistics_indices = [self.ol_stats[s]["idx"] for s in self.selected_ol_stats]
        else:
            selected_statistics_indices = [self.cl_stats[s]["idx"] for s in self.selected_cl_stats]
            
        radar_plot_values_scenarios = np.mean(radar_plot_scenario_statistics[:, selected_statistics_indices], axis=1)
        radar_plot_values_selected_statistics = np.mean(radar_plot_scenario_statistics[:, selected_statistics_indices], axis=0)
        radar_plot_keys_selected_statistics = np.array(radar_plot_metrics)[selected_statistics_indices].tolist()
        
        # for new plot
        # radar_plot_values_scenarios_2 = np.mean(radar_plot_scenario_statistics[:, :], axis=1) # computes the mean again
        all_idx = self.ol_stats["all"]["idx"] if simulation_type == "Open-Loop" else self.cl_stats["all"]["idx"]
        radar_plot_values_scenarios_2 = radar_plot_scenario_statistics[:, all_idx]
        
        scenario_types_values = self.compute_metrics_mean_for_each_scenario_type(scenario_types_of_each_scenario,
                                                                                 radar_plot_values_scenarios,
                                                                                 scenario_types_keys)
        scenario_types_values_2 = self.compute_metrics_mean_for_each_scenario_type(scenario_types_of_each_scenario,
                                                                                   radar_plot_values_scenarios_2,
                                                                                   scenario_types_keys)
        
        # each subplot has to use the same metrics (same number of radar corners)
        radar_plot_data = [
            [   # for Radar Plot 1
                radar_plot_keys_selected_statistics,
                ('Mean of a Selection of Metrics over all Scenarios',
                 [radar_plot_values_selected_statistics]),  # [radar_plot_values_model_1, radar_plot_values_model_2, radar_plot_values_model_3] for several models
            ],
            [   # for Radar Plot 2
                scenario_types_keys,
                ('Mean of the Selected Metrics for each Scenario Type',
                 [scenario_types_values]),  # [radar_plot_values_model_1, radar_plot_values_model_2, radar_plot_values_model_3] for several models
            ],
            [   # for Radar Plot 3
                scenario_types_keys,
                ('Average Score for each Scenario Type', # or 'Mean of all Metrics for each Scenario Type'
                 [scenario_types_values_2]),  # [radar_plot_values_model_1, radar_plot_values_model_2, radar_plot_values_model_3] for several models
                # ('Model Comparison based on AVAILABLE Metrics', [radar_plot_data]), # for other subpolts
            ]
        ]
        matplotlib_plots = self._render_radar_plot(radar_plot_data, simulation_type)

        # self._save_to_pdf(matplotlib_plots=matplotlib_plots)
        return matplotlib_plots
    
    def compute_metrics_mean_for_each_scenario_type(
        self,
        scenario_types_of_each_scenario,
        radar_plot_values_scenarios,
        scenario_types_keys
    ):
        # Combine scenario types and their respective values into a dictionary
        scenario_values = collections.defaultdict(list)
        for scenario_type, value in zip(scenario_types_of_each_scenario, radar_plot_values_scenarios):
            scenario_values[scenario_type].append(value)

        # Compute the mean of each scenario type's values
        scenario_means = {scenario_type: statistics.mean(values) for scenario_type, values in scenario_values.items()}
        scenario_means["all"] = statistics.mean(radar_plot_values_scenarios)
        scenario_types_values = [scenario_means[key] for key in scenario_types_keys]
        
        return scenario_types_values
        
    def _render_radar_plot(self, data: List[Tuple[str, List[np.ndarray]]], simulation_type: str) -> List[plt.Figure]:
        matplotlib_plots = []
        
        for data_i in data: # plot i figures
            theta = radar_factory(len(data_i[0]), frame='polygon')

            fig, ax = plt.subplots(figsize=(9, 9), nrows=1, ncols=1, subplot_kw=dict(projection='radar'))
            fig.subplots_adjust(wspace=0.25, hspace=0.10, top=0.85, bottom=0.05)

            colors = ['b', 'r', 'g', 'm', 'y']
            # Plot the four cases from the example data on separate axes
            # for i, ax in enumerate(ax):
            spoke_labels, (title, case_data) = data_i
            ax.set_rgrids([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
            # ax.set_theta_offset(2*np.pi / 15)
            # ax.set_theta_direction('clockwise')
            ax.set_ylim(0, 1)  # Set the limits for the radial axis
            ax.set_title(title, weight='bold', size='medium', position=(0.5, 1.1),
                            horizontalalignment='center', verticalalignment='center')
            for d, color in zip(case_data, colors):
                ax.plot(theta, d, color=color)
                ax.fill(theta, d, facecolor=color, alpha=0.25, label='_nolegend_')
            ax.set_varlabels(spoke_labels)
            # for label in ax.get_xticklabels():
            #     label.set_rotation(30)
            
            # add legend relative to top-left plot
            labels = (self.model_name, 'Model 2', 'Model 3', 'Model 4', 'Model 5')
            legend = ax.legend(labels, loc=(0.9, .95), labelspacing=0.1, fontsize='small')

            fig.text(0.5, 0.9, f"{simulation_type} Simulation",
                     horizontalalignment='center', color='black', weight='bold',
                     size='large')

            matplotlib_plots.append(fig)
            
        return matplotlib_plots

    def on_run_simulation_end(self) -> None:
        """Callback before end of the main function."""
        start_time = time.perf_counter()

        # Stop if no metric save path
        if not self._metric_save_path.exists() and not self._metric_aggregator_save_path.exists():
            return

        self._metric_aggregator_dataframes = self._read_metric_parquet_files(
            metric_save_path=self._metric_aggregator_save_path, metric_reader=metric_aggregator_reader
        )
        self._metric_statistics_dataframes = self._read_metric_parquet_files(
            metric_save_path=self._metric_save_path,
            metric_reader=metric_statistics_reader,
        )
        planner_color_maps = self._assign_planner_colors()

        # Aggregate histogram data
        histogram_data_dict = self._aggregate_metric_statistic_histogram_data()
        scenario_type_histogram_data_dict = self._aggregate_scenario_type_score_histogram_data()
        # Integrate them into the same dictionary
        histogram_data_dict.update(scenario_type_histogram_data_dict)

        # Compute edges
        histogram_edge_data = compute_histogram_edges(bins=self._num_bins, aggregated_data=histogram_data_dict)
        matplotlib_plots = self._draw_histogram_plots(
            planner_color_maps=planner_color_maps,
            histogram_data_dict=histogram_data_dict,
            histogram_edges=histogram_edge_data,
        )

        radar_matplotlib_plots = self._draw_radar_plots(
            planner_color_maps=planner_color_maps,
            histogram_data_dict=histogram_data_dict,
            histogram_edges=histogram_edge_data,
        )
        matplotlib_plots.extend(radar_matplotlib_plots)
        
        self._save_to_pdf(matplotlib_plots)
        self._save_to_png(matplotlib_plots)
        
        plt.close('all')  # This will close all open figures

        end_time = time.perf_counter()
        elapsed_time_s = end_time - start_time
        time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time_s))
        logger.info('Metric summary: {} [HH:MM:SS]'.format(time_str))

