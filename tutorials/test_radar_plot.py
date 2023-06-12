import torch
from typing import cast
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory
from nuplan.planning.training.preprocessing.features.trajectories import Trajectories
from nuplan.planning.training.preprocessing.features.tensor_target import TensorTarget
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path as MplPath
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D


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

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
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
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


def example_data():
    data = [
        [
            ['Metric 1', 'Metric 2', 'Metric 3', 'Metric 4', 'Metric 5'],
            ('Model Comparison based on AVAILABLE Metrics',
             [# [1.0, 1.0, 1.0, 1.0, 1.0],
              # [0.8, 0.8, 0.8, 0.8, 0.8],
              # [0.02, 0.01, 0.07, 0.01, 0.21],
              [0.01, 0.01, 0.02, 0.71, 0.74]]),
        ],
        [
            ['Metric 1', 'Metric 2', 'Metric 3', 'Metric 4'],
            ('Model Comparison based on AVAILABLE Metrics',
             [# [1.0, 1.0, 1.0, 1.0],
              # [0.8, 0.8, 0.8, 0.8],
              [0.02, 0.01, 0.07, 0.01],
              [0.01, 0.01, 0.02, 0.71]]),
        ]
    ]
    return data




if __name__ == '__main__':
    data = example_data()
    for data_i in data:
        theta = radar_factory(len(data_i[0]), frame='polygon')


        fig, ax = plt.subplots(figsize=(9, 9), nrows=1, ncols=1, subplot_kw=dict(projection='radar'))
        fig.subplots_adjust(wspace=0.25, hspace=0.10, top=0.85, bottom=0.05)

        colors = ['b', 'r', 'g', 'm', 'y']
        # Plot the four cases from the example data on separate axes
        # for i, ax in enumerate(ax):
        spoke_labels, (title, case_data) = data_i
        ax.set_rgrids([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_title(title, weight='bold', size='medium', position=(0.5, 1.1),
                        horizontalalignment='center', verticalalignment='center')
        for d, color in zip(case_data, colors):
            ax.plot(theta, d, color=color)
            ax.fill(theta, d, facecolor=color, alpha=0.25, label='_nolegend_')
        ax.set_varlabels(spoke_labels)

        # add legend relative to top-left plot
        labels = ('Model 1', 'Model 2', 'Model 3', 'Model 4', 'Model 5')
        legend = ax.legend(labels, loc=(0.9, .95), labelspacing=0.1, fontsize='small')

        # fig.text(0.5, 0.965, '5-Model Solution Profiles Across Four Scenarios',
        #          horizontalalignment='center', color='black', weight='bold',
        #          size='large')

        plt.show()