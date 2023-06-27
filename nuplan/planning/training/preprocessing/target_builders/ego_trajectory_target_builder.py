from __future__ import annotations

from typing import Type

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import AbstractModelFeature
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory
from nuplan.planning.training.preprocessing.features.trajectories import Trajectories
from nuplan.planning.training.preprocessing.features.trajectory_utils import convert_absolute_to_relative_poses
from nuplan.planning.training.preprocessing.target_builders.abstract_target_builder import AbstractTargetBuilder

from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.actor_state.state_representation import StateSE2
import numpy as np

class EgoTrajectoryTargetBuilder(AbstractTargetBuilder):
    """Trajectory builders constructed the desired ego's trajectory from a scenario."""

    def __init__(self, future_trajectory_sampling: TrajectorySampling) -> None:
        """
        Initializes the class.
        :param future_trajectory_sampling: parameters for sampled future trajectory
        """
        self._num_future_poses = future_trajectory_sampling.num_poses
        self._time_horizon = future_trajectory_sampling.time_horizon

    @classmethod
    def get_feature_unique_name(cls) -> str:
        """Inherited, see superclass."""
        return "trajectory"

    @classmethod
    def get_feature_type(cls) -> Type[AbstractModelFeature]:
        """Inherited, see superclass."""
        return Trajectory  # type: ignore

    def get_targets(self, scenario: AbstractScenario) -> Trajectory:
        """Inherited, see superclass."""
        current_absolute_state = scenario.initial_ego_state
        trajectory_absolute_states = scenario.get_ego_future_trajectory(
            iteration=0, num_samples=self._num_future_poses, time_horizon=self._time_horizon
        )

        # Get all future poses relative to the ego coordinate system
        trajectory_relative_poses = convert_absolute_to_relative_poses(
            current_absolute_state.rear_axle, [state.rear_axle for state in trajectory_absolute_states]
        )

        if len(trajectory_relative_poses) != self._num_future_poses:
            raise RuntimeError(f'Expected {self._num_future_poses} num poses but got {len(trajectory_absolute_states)}')

        return Trajectory(data=trajectory_relative_poses)

class AgentsTrajectoryTargetBuilder(AbstractTargetBuilder):
    """Trajectory builders constructed the desired ego's trajectory from a scenario."""

    def __init__(self, future_trajectory_sampling: TrajectorySampling) -> None:
        """
        Initializes the class.
        :param future_trajectory_sampling: parameters for sampled future trajectory
        """
        self._num_future_poses = future_trajectory_sampling.num_poses
        self._time_horizon = future_trajectory_sampling.time_horizon

    @classmethod
    def get_feature_unique_name(cls) -> str:
        """Inherited, see superclass."""
        return "trajectories"

    @classmethod
    def get_feature_type(cls) -> Type[AbstractModelFeature]:
        """Inherited, see superclass."""
        return Trajectories  # type: ignore

    def get_targets(self, scenario: AbstractScenario) -> Trajectories:
        """Inherited, see superclass."""
        pad_to = 30

        vehicles_objects_initial = scenario.initial_tracked_objects.tracked_objects.get_tracked_objects_of_type(TrackedObjectType.VEHICLE)
        vehicles_objects_initial = list(vehicles_objects_initial)
        vehicle_array_initial = []
        for v in vehicles_objects_initial:
            vehicle_array_initial.append(v.center.serialize())
        vehicle_array_initial = np.array(vehicle_array_initial)
        vehicle_array_initial = vehicle_array_initial[:pad_to]
        if pad_to > vehicle_array_initial.shape[0]:
            pad = np.zeros((pad_to-vehicle_array_initial.shape[0], vehicle_array_initial.shape[1])).tolist()
            vehicle_array_initial = np.concatenate((vehicle_array_initial, pad), axis=0)

        vehicles_absolute_states = scenario.get_future_tracked_objects(
            iteration=0, num_samples=self._num_future_poses, time_horizon=self._time_horizon
        )
        vehicles_absolute_states = list(vehicles_absolute_states)
        state_array = []
        for i, state_at_timestep in enumerate(vehicles_absolute_states):
            state_at_timestep.tracked_objects.get_tracked_objects_of_type(TrackedObjectType.VEHICLE)
            t_center = convert_absolute_to_relative_poses(
                StateSE2.deserialize(vehicle_array_initial[i]), [s.center for s in state_at_timestep.tracked_objects.tracked_objects]
            )
            t_center = t_center[:pad_to]
            if pad_to > t_center.shape[0]:
                pad = np.zeros((pad_to-t_center.shape[0], t_center.shape[1])).tolist()
                t_center = np.concatenate((t_center, pad), axis=0)
            state_array.append(t_center)
            if i == pad_to: break
        state_array = np.array(state_array).transpose(1, 0, 2).reshape((-1, 3))


        # if state_array.shape[1] != self._num_future_poses:
        #     raise RuntimeError(f'Expected {self._num_future_poses} num poses but got {len(state_array.shape[1])}')

        return Trajectories([Trajectory(data=state_array)])