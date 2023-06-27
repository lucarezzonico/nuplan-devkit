# import os
# from pathlib import Path
# import hydra
# from omegaconf import OmegaConf, DictConfig
# import tempfile
# from nuplan.planning.script.run_training import main as main_train
# from nuplan.planning.script.run_simulation import main as main_simulation
# from nuplan.planning.script.run_nuboard import main as main_nuboard
import yaml
# import glob
# from typing import List, Optional, cast
# import shutil
# from PIL import Image
# import logging

# from nuplan.planning.training.experiments.training import TrainingEngine

# import matplotlib.pyplot as plt


# logger = logging.getLogger(__name__)

# from nuplan.planning.simulation.main_callback.metric_summary_callback import (
#     radar_factory,
#     example_data,
#     IdxUnit,
#     OLStats,
#     CLStats,
#     MetricSummaryCallback,
# )

import logging
import math
import time
from collections import defaultdict, OrderedDict
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
            [0.8, 0.8, 0.8, 0.8, 0.8],          # Model 1
            [0.02, 0.01, 0.07, 0.01, 0.21],     # Model 2
            [0.01, 0.01, 0.02, 0.71, 0.74]]),   # Model 3
    ]
    return data


class MetricSummary():
    """Callback to render histograms for metrics and metric aggregator."""

    def __init__(
        self,
        metric_save_path: str,
        metric_aggregator_save_path: str,
        summary_output_path: str,
        pdf_file_name: str,
        selected_ol_stats: List[str],
        selected_cl_stats: List[str],
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
        
    def _save_to_png(self, matplotlib_plots: List[Any],  simulation_type: str) -> None:
        """
        Save a list of matplotlib plots to png files.
        :param matplotlib_plots: A list of matplotlib plots.
        """
        png_path = f"{self._summary_output_path}/png_figs"
        if not os.path.exists(png_path): os.makedirs(png_path)
        
        if simulation_type == "Open-Loop": sim_type = "open_loop_sim"
        elif simulation_type == "Closed-Loop Non-Reactive Agents": sim_type = "closed_loop_non_reactive_sim"
        elif simulation_type == "Closed-Loop Reactive Agents": sim_type = "closed_loop_reactive_sim"
        
        for i, fig in enumerate(matplotlib_plots[::-1]): # -1 specifies the step, in this case it goes backwards, so the entire list is returned in reverse order.
            # file_name = f"{png_path}/fig_{i}.png"
            if i == 0:
                file_name = f"{png_path}/{sim_type}_scenario_scores.png"
            elif i == 1:
                file_name = f"{png_path}/{sim_type}_metrics.png"
            else:
                raise ValueError("choose name for additial plot")
            fig.savefig(file_name, format='png')
        
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
        
    def _render_radar_plot(self, data: List[Tuple[str, List[np.ndarray]]], simulation_type: str, model_names: list[str]) -> List[plt.Figure]:
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
            labels = (model_names)
            legend = ax.legend(labels, loc=(0.79, 1.04), labelspacing=0.1, fontsize='small')

            fig.text(0.5, 0.9, f"{simulation_type} Simulation",
                     horizontalalignment='center', color='black', weight='bold',
                     size='large')

            matplotlib_plots.append(fig)
            
        return matplotlib_plots


    
    ######################################## Multiple Models on Radar Plot ########################################

    def _get_model_radar_data(self):
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
        
        radar_plot_data, simulation_type = self._get_radar_data(
            planner_color_maps=planner_color_maps,
            histogram_data_dict=histogram_data_dict,
            histogram_edges=histogram_edge_data,
        )
        
        return radar_plot_data, simulation_type
    
    def _get_radar_data(
        self,
        planner_color_maps: Dict[str, Any],
        histogram_data_dict: HistogramConstantConfig.HistogramDataType,
        histogram_edges: HistogramConstantConfig.HistogramEdgesDataType,
        n_cols: int = 2,
    ) -> list[list]:
        """
        :param planner_color_maps: Color maps from planner names.
        :param histogram_data_dict: A dictionary of histogram data.
        :param histogram_edges: A dictionary of histogram edges (bins) data.
        :param n_cols: Number of columns in subplot.
        """
                
        simulation_type = str(list(histogram_data_dict.keys())[-1])
        if "open_loop" in simulation_type:
            simulation_type = "Open-Loop"
        elif "closed_loop_nonreactive_agents" in simulation_type:
            simulation_type = "Closed-Loop Non-Reactive Agents"
        elif "closed_loop_reactive_agents" in simulation_type:
            simulation_type = "Closed-Loop Reactive Agents"
        else:
            raise ValueError(f"Simulation type {simulation_type} is not supported!")
        
        # dictionary to store the result
        stat_dict = OrderedDict()
        
        radar_plot_scenario_statistics = []
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
                    elif "miss_rate_horizon" in statistic_name and unit in ["ratio"]:
                        values_per_scenario = 1-values_per_scenario
                    elif "no_ego_at_fault_collisions" in statistic_name and unit in ["boolean"]:
                        values_per_scenario = 1-values_per_scenario
                    # elif "all" == statistic_name and unit in ["scores"]:
                    #     values_per_scenario = values_per_scenario
                    # elif  unit in ["scores"]: # every scenario's score separately
                    #     values_per_scenario = values_per_scenario

                    stat_name = statistic_name
                    # stat_name = stat_name.replace("planner_expert_", "")
                    # stat_name = stat_name.replace("planner_", "")
                    # stat_name = stat_name.replace("average_l2_error", "ADE")
                    # stat_name = stat_name.replace("final_l2_error", "FDE")
                    # stat_name = stat_name.replace("average_heading_error", "AHE")
                    # stat_name = stat_name.replace("final_heading_error", "FHE")
                    
                    # append the stat name and corresponding values to the dictionary
                    stat_dict[stat_name] = values_per_scenario
                    radar_plot_scenario_statistics.append(values_per_scenario)

                    statistic_global_idx += 1
                    
                    # stop after the "all" statistic, avoid scenario specific statistics
                    if stat_name == "all":
                        break
                    
        radar_plot_scenario_statistics = np.array(radar_plot_scenario_statistics).transpose(1,0)
        
        if simulation_type == "Open-Loop":
            selected_statistics_indices = self.get_dict_indices(stat_dict, self.selected_ol_stats)
        else:
            selected_statistics_indices = self.get_dict_indices(stat_dict, self.selected_cl_stats)
            
        radar_plot_values_scenarios = np.mean(radar_plot_scenario_statistics[:, selected_statistics_indices], axis=1)
        
        # selected metrics' mean of over all scenarios
        radar_plot_values_selected_statistics = np.mean(radar_plot_scenario_statistics[:, selected_statistics_indices], axis=0)
        radar_plot_keys_selected_statistics = self.selected_ol_stats if simulation_type == "Open-Loop" else self.selected_cl_stats
        
        # # each metric's mean over all scenarios
        # radar_plot_values_selected_statistics = np.mean(radar_plot_scenario_statistics[:, :], axis=0)
        # radar_plot_keys_selected_statistics = list(stat_dict.keys())
        
        # for new plot
        # radar_plot_values_scenarios_2 = np.mean(radar_plot_scenario_statistics[:, :], axis=1) # computes the mean again
        radar_plot_values_scenarios_2 = radar_plot_scenario_statistics[:, -1]
        
        # # reorder the scenario_types to avoid overlapping ing the final plot
        scenario_types_keys.sort(key=len)
        
        # scenario_types_keys = list(enumerate(scenario_types_keys))
        # scenario_types_keys.sort(key=lambda x: len(x[1]))
        
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
        
        return [radar_plot_keys_selected_statistics, radar_plot_values_selected_statistics,
                scenario_types_keys, scenario_types_values,
                scenario_types_keys, scenario_types_values_2], \
                simulation_type


    def get_dict_indices(self, dict, keys):
        return [list(dict.keys()).index(k) for k in keys]
    
    
def get_latex_table(data, models, best_val):
    # Create the DataFrame
    # data = {
    #     '& all': [0.39, 0.44, 0.48],
    #     '& changing lane': [0.34, 0.41, 0.45],
    #     '& stopping with lead': [0.79, 0.83, 0.71],
    #     '& starting left turn': [0.19, 0.29, 0.39],
    #     '& low magnitude speed': [0.49, 0.54, 0.43],
    #     '& starting right turn': [0.25, 0.35, 0.42],
    #     '& behind long vehicle': [0.64, 0.51, 0.59],
    #     '& high magnitude speed': [0.18, 0.43, 0.44],
    #     '& stationary in traffic': [0.84, 0.88, 0.81],
    #     '& near multiple vehicles': [0.67, 0.48, 0.65],
    #     '& following lane with lead': [0.25, 0.25, 0.29],
    #     '& traversing pickup drop off': [0.22, 0.29, 0.44],
    #     '& high lateral acceleration': [0.20, 0.08, 0.18],
    #     '& waiting for pedestrian to cross': [0.30, 0.43, 0.48],
    #     '& starting straight traffic light intersection traversal': [0.12, 0.35, 0.42],
    # }
    pd.set_option('display.max_colwidth', None)
    df = pd.DataFrame(data, index=models).T

    # Create a function to highlight the max in a series.
    def bold_max(s):
        if best_val == 'max':
            is_max = s == s.max()
        if best_val == 'min':
            is_max = s == s.min()
        
        return [' & '.join(['\\textbf{' + str(v) + '}' if vmax else str(v) for v, vmax in zip(s, is_max)])]

    # Apply the function to each row
    df = df.apply(bold_max, axis=1)

    # Convert to LaTeX
    latex_str = df.to_latex(header=False, escape=False)

    # Removing square brackets from the string
    latex_str = latex_str.replace('[', '').replace(']', '')

    # Formatting the string
    latex_table = """
    \\begin{{table}}[h]
    \\begin{{center}}
        \\begin{{tabular}}{{clccc}}
            \\Xhline{{2\\arrayrulewidth}}
            & & \\multicolumn{{3}}{{c}}{{Model}} \\\\
            & & \\rotatebox{{90}}{{AutoBotEgo}} & \\rotatebox{{90}}{{\\parbox{{2.1cm}}{{Urban Driver Closed-Loop}}}} & \\rotatebox{{90}}{{\\parbox{{2.1cm}}{{Urban Driver Closed-Loop with velocity}}}} \\\\
            \\Xhline{{0.5\\arrayrulewidth}}
            \\multirow{{14}}{{*}}{{\\rotatebox{{90}}{{Scenario Scores}}}}
            {} % Here goes the tabular data
            \\Xhline{{0.5\\arrayrulewidth}}
            \\multirow{{1}}{{*}}{{\\rotatebox{{90}}{{Scenario Scores}}}}
            & all & 0.39 & 0.44 & 0.48 \\\\
            \\Xhline{{2\\arrayrulewidth}}
        \\end{{tabular}}
    \\end{{center}}
    \\caption{{Open-Loop Simulation Results}}
    \\label{{tab:open_loop_simulation}}
    \\end{{table}}
    """.format(latex_str[latex_str.find("\\midrule")+8:latex_str.find("\\bottomrule")].strip()) # extracting only the tabular data

    # Print the LaTeX table
    latex_str = latex_str.replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ')

    print(latex_str)
    


if __name__ == '__main__':

    # cfgs = task.load_cfgs("default_config_autobotego")
    # task.main(cfgs)
    
    cfg = yaml.safe_load(open("tutorials/radar_plot_multiple_models.yaml"))
    exp_root = os.getenv('NUPLAN_EXP_ROOT')
    output_dir = f"{exp_root}/simulation"
    
    
    for model_dirs in ["open_loop_model_dirs", "closed_loop_nonreactive_model_dirs", "closed_loop_reactive_model_dirs"]:
        model_names = []
        plot_1_values = []
        plot_2_values = []
        plot_3_values = []
        
        for model_dir in cfg["multiple_models"][model_dirs]:
            
            cfg["metric_summary_callback"]["metric_save_path"] = f"{exp_root}/{model_dir}/metrics"              # Path to saved metric files
            cfg["metric_summary_callback"]["metric_aggregator_save_path"] = f"{exp_root}/{model_dir}/aggregator_metric"     # Path to saved aggregated files
            cfg["metric_summary_callback"]["summary_output_path"] = f"{output_dir}/summary"

            index = model_dir.find("_experiment")    # Find the position of the first occurrence of "_experiment" in model_dir
            model_name = model_dir[:index]
            model_name = model_name.split('/')
            model_name = model_name[-1]         # this is the string after the last '/'
            model_names.append(model_name)
            
            # # extract the name of the model
            # model_dir_split = model_dir.split('/')
            # model_name = model_dir_split[0] #this is the string before the first '/'
            # model_name = model_name.replace('_experiment', '')
            # model_names.append(model_name)
        
            msc = MetricSummary(**cfg["metric_summary_callback"])
            
            # first_model = True
            # if first_model:
            #     model_radar_plot_data, simulation_type = msc._get_model_radar_data()
            #     first_model = False
            # else:
            #     new_model_radar_plot_data, _ = msc._get_model_radar_data()[1][1]
            #     for i in range(len(new_model_radar_plot_data)):
            #         model_radar_plot_data[i][1][1].extend(new_model_radar_plot_data[i][1][1])
            
            
            [radar_plot_keys_selected_statistics, radar_plot_values_selected_statistics,
            scenario_types_keys, scenario_types_values,
            scenario_types_keys, scenario_types_values_2], \
            simulation_type = msc._get_model_radar_data()
                
            plot_1_values.extend([radar_plot_values_selected_statistics])
            plot_2_values.extend([scenario_types_values])
            plot_3_values.extend([scenario_types_values_2])
        
        selected_statistics_keys = radar_plot_keys_selected_statistics.copy()
        for i in range(len(selected_statistics_keys)):
            selected_statistics_keys[i] = selected_statistics_keys[i].replace("planner_expert_", "")
            selected_statistics_keys[i] = selected_statistics_keys[i].replace("planner_", "")
            selected_statistics_keys[i] = selected_statistics_keys[i].replace("average_l2_error", "ADE")
            selected_statistics_keys[i] = selected_statistics_keys[i].replace("final_l2_error", "FDE")
            selected_statistics_keys[i] = selected_statistics_keys[i].replace("average_heading_error", "AHE")
            selected_statistics_keys[i] = selected_statistics_keys[i].replace("final_heading_error", "FHE")
            selected_statistics_keys[i] = selected_statistics_keys[i].replace("avg_ADE_over_all_horizons", "ADE_horizon_avg")
            selected_statistics_keys[i] = selected_statistics_keys[i].replace("avg_FDE_over_all_horizons", "FDE_horizon_avg")
            selected_statistics_keys[i] = selected_statistics_keys[i].replace("avg_AHE_over_all_horizons", "AHE_horizon_avg")
            selected_statistics_keys[i] = selected_statistics_keys[i].replace("avg_FHE_over_all_horizons", "FHE_horizon_avg")
            selected_statistics_keys[i] = selected_statistics_keys[i].replace("miss_rate_horizon", "success_rate_horizon")
            selected_statistics_keys[i] = selected_statistics_keys[i].replace("no_ego_at_fault_collisions", "not_ego_at_fault_collisions")
            selected_statistics_keys[i] = selected_statistics_keys[i].replace("abs_ego_lon_acceleration_within_bounds", "                   abs_ego_lon_acceleration_within_bounds")
            selected_statistics_keys[i] = selected_statistics_keys[i].replace("time_to_collision_within_bound", "   time_to_collision_within_bound")
            selected_statistics_keys[i] = selected_statistics_keys[i].replace("ego_is_making_progress", "ego_is_making_progress   ")
            selected_statistics_keys[i] = selected_statistics_keys[i].replace("abs_ego_yaw_acceleration_within_bounds", "abs_ego_yaw_acceleration_within_bounds                                ")
            selected_statistics_keys[i] = selected_statistics_keys[i].replace("ego_lane_change_fail_rate_below_threshold", "ego_lane_change_fail_rate_below_threshold    ")
        
        scenarios_keys = scenario_types_keys.copy()
        for i in range(len(scenarios_keys)):
            scenarios_keys[i] = scenarios_keys[i].replace("traversing_pickup_dropoff", "traversing_pickup_dropoff       ")
            scenarios_keys[i] = scenarios_keys[i].replace("high_lateral_acceleration", "high_lateral_acceleration       ")

        # each subplot has to use the same metrics (same number of radar corners)
        models_radar_plot_data = [
            [   # for Radar Plot 1
                selected_statistics_keys,
                ('Mean of a Selection of Metrics over all Scenarios',
                plot_1_values),  # [radar_plot_values_model_1, radar_plot_values_model_2, radar_plot_values_model_3] for several models
            ],
            # [   # for Radar Plot 2
            #     scenarios_keys,
            #     ('Mean of the Selected Metrics for each Scenario Type',
            #      plot_2_values),  # [radar_plot_values_model_1, radar_plot_values_model_2, radar_plot_values_model_3] for several models
            # ],
            [   # for Radar Plot 3
                scenarios_keys,
                ('Average Score for each Scenario Type', # or 'Mean of all Metrics for each Scenario Type'
                plot_3_values),  # [radar_plot_values_model_1, radar_plot_values_model_2, radar_plot_values_model_3] for several models
                # ('Model Comparison based on AVAILABLE Metrics', [radar_plot_data]), # for other subpolts
            ]
        ]
        
        # print scenario latex table
        df_values_scenarios = list(np.array(np.round(plot_3_values, decimals=2)).transpose(1,0))
        df_scenarios = scenario_types_keys
        data = {'& '+ k.replace('_', ' ') : list(v) for k, v in zip(df_scenarios, df_values_scenarios)}
        get_latex_table(data, models=cfg["multiple_models"]["model_names"], best_val="max") # best_val to select which one to print in bold
        
        # print metric latex table
        df_values_metrics = list(np.array(np.round(plot_1_values, decimals=2)).transpose(1,0))
        df_metrics = selected_statistics_keys
        data = {'& '+ k.replace(' ', '').replace('_', ' ') : list(v) for k, v in zip(df_metrics, df_values_metrics)}
        get_latex_table(data, models=cfg["multiple_models"]["model_names"], best_val="max") # best_val to select which one to print in bold

        # create radar plots
        matplotlib_plots = msc._render_radar_plot(models_radar_plot_data, simulation_type, cfg["multiple_models"]["model_names"])
            
        # msc._save_to_pdf(matplotlib_plots)
        msc._save_to_png(matplotlib_plots, simulation_type)
        
        plt.close('all')  # This will close all open figures
        