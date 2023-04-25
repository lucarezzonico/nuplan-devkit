from typing import Dict, List, Tuple, Type, cast

import torch

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import TimePoint
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.scenario_builder.scenario_utils import sample_indices_with_time_horizon
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization, PlannerInput
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import AbstractModelFeature
from nuplan.planning.training.preprocessing.feature_builders.scriptable_feature_builder import ScriptableFeatureBuilder
from nuplan.planning.training.preprocessing.features.agents import Agents
from nuplan.planning.training.preprocessing.utils.agents_preprocessing import (
    build_ego_features_from_tensor,
    compute_yaw_rate_from_state_tensors,
    convert_absolute_quantities_to_relative,
    filter_agents_tensor,
    pack_agents_tensor,
    pad_agent_states,
    sampled_future_ego_states_to_tensor,
    sampled_future_timestamps_to_tensor,
    sampled_tracked_objects_to_tensor_list,
)


class ExpertFeatureBuilder(ScriptableFeatureBuilder):
    """Builder for constructing agent features during training and simulation."""

    def __init__(self, trajectory_sampling: TrajectorySampling) -> None:
        """
        Initializes ExpertFeatureBuilder.
        :param trajectory_sampling: Parameters of the sampled trajectory of every agent
        """
        super().__init__()
        self.num_future_poses = trajectory_sampling.num_poses
        self.future_time_horizon = trajectory_sampling.time_horizon

        self._agents_states_dim = Agents.agents_states_dim()

    @torch.jit.unused
    @classmethod
    def get_feature_unique_name(cls) -> str:
        """Inherited, see superclass."""
        return "expert"

    @torch.jit.unused
    @classmethod
    def get_feature_type(cls) -> Type[AbstractModelFeature]:
        """Inherited, see superclass."""
        return Agents  # type: ignore

    @torch.jit.unused
    def get_features_from_scenario(self, scenario: AbstractScenario) -> Agents:
        """Inherited, see superclass."""
        # Retrieve present/future ego states and agent boxes
        with torch.no_grad():
            
            # TODO: remove current ego pose from future trajectory
            # TODO: understand difference between get_ego_future_trajectory() and get_expert_ego_trajectory()
            # nb_iterations = scenario.get_number_of_iterations()
            # expert_goal_state = scenario.get_expert_goal_state()
            # expert_states = scenario.get_expert_ego_trajectory() # EXPERT
            # TODO: future trajectory (blue in simulation) is 8s (len 16), but ground truth (orange in simulation) is 15s
            
            anchor_ego_state = scenario.initial_ego_state

            future_ego_states = scenario.get_ego_future_trajectory(
                iteration=0, num_samples=self.num_future_poses, time_horizon=self.future_time_horizon
            )
            sampled_future_ego_states = list(future_ego_states) + [anchor_ego_state]
            time_stamps = list(
                scenario.get_future_timestamps(
                    iteration=0, num_samples=self.num_future_poses, time_horizon=self.future_time_horizon
                )
            ) + [scenario.start_time]
            # Retrieve present/future agent boxes
            present_tracked_objects = scenario.initial_tracked_objects.tracked_objects
            future_tracked_objects = [
                tracked_objects.tracked_objects
                for tracked_objects in scenario.get_future_tracked_objects(
                    iteration=0, time_horizon=self.future_time_horizon, num_samples=self.num_future_poses
                )
            ]

            # Extract and pad features
            sampled_future_observations = future_tracked_objects + [present_tracked_objects]

            assert len(sampled_future_ego_states) == len(sampled_future_observations), (
                "Expected the trajectory length of ego and agent to be equal. "
                f"Got ego: {len(sampled_future_ego_states)} and agent: {len(sampled_future_observations)}"
            )

            assert len(sampled_future_observations) > 2, (
                "Trajectory of length of " f"{len(sampled_future_observations)} needs to be at least 3"
            )

            tensors, list_tensors, list_list_tensors = self._pack_to_feature_tensor_dict(
                sampled_future_ego_states, time_stamps, sampled_future_observations
            )

            tensors, list_tensors, list_list_tensors = self.scriptable_forward(tensors, list_tensors, list_list_tensors)

            output: Agents = self._unpack_feature_from_tensor_dict(tensors, list_tensors, list_list_tensors)

            return output

    @torch.jit.unused
    def get_features_from_simulation(
        self, current_input: PlannerInput, initialization: PlannerInitialization
    ) -> Agents:
        """Inherited, see superclass."""
        with torch.no_grad():
            history = current_input.history
            assert isinstance(
                history.observations[0], DetectionsTracks
            ), f"Expected observation of type DetectionTracks, got {type(history.observations[0])}"

            present_ego_state, present_observation = history.current_state

            future_observations = history.observations[:-1]
            future_ego_states = history.ego_states[:-1]

            assert history.sample_interval, "SimulationHistoryBuffer sample interval is None"

            indices = sample_indices_with_time_horizon(
                self.num_future_poses, self.future_time_horizon, history.sample_interval
            )

            try:
                sampled_future_observations = [
                    cast(DetectionsTracks, future_observations[-idx]).tracked_objects for idx in reversed(indices)
                ]
                sampled_future_ego_states = [future_ego_states[-idx] for idx in reversed(indices)]
            except IndexError:
                raise RuntimeError(
                    f"SimulationHistoryBuffer duration: {history.duration} is "
                    f"too short for requested future_time_horizon: {self.future_time_horizon}. "
                    f"Please increase the simulation_buffer_duration in default_simulation.yaml"
                )

            sampled_future_observations = sampled_future_observations + [
                cast(DetectionsTracks, present_observation).tracked_objects
            ]
            sampled_future_ego_states = sampled_future_ego_states + [present_ego_state]
            time_stamps = [state.time_point for state in sampled_future_ego_states]

            tensors, list_tensors, list_list_tensors = self._pack_to_feature_tensor_dict(
                sampled_future_ego_states, time_stamps, sampled_future_observations
            )

            tensors, list_tensors, list_list_tensors = self.scriptable_forward(tensors, list_tensors, list_list_tensors)

            output: Agents = self._unpack_feature_from_tensor_dict(tensors, list_tensors, list_list_tensors)

            return output

    @torch.jit.unused
    def _pack_to_feature_tensor_dict(
        self,
        future_ego_states: List[EgoState],
        future_time_stamps: List[TimePoint],
        future_tracked_objects: List[TrackedObjects],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]], Dict[str, List[List[torch.Tensor]]]]:
        """
        Packs the provided objects into tensors to be used with the scriptable core of the builder.
        :param future_ego_states: The future states of the ego vehicle.
        :param future_time_stamps: The future time stamps of the input data.
        :param future_tracked_objects: The future tracked objects.
        :return: The packed tensors.
        """
        future_ego_states_tensor = sampled_future_ego_states_to_tensor(future_ego_states)
        future_time_stamps_tensor = sampled_future_timestamps_to_tensor(future_time_stamps)
        future_tracked_objects_tensor_list = sampled_tracked_objects_to_tensor_list(future_tracked_objects)

        return (
            {
                "future_ego_states": future_ego_states_tensor,
                "future_time_stamps": future_time_stamps_tensor,
            },
            {"future_tracked_objects": future_tracked_objects_tensor_list},
            {},
        )

    @torch.jit.unused
    def _unpack_feature_from_tensor_dict(
        self,
        tensor_data: Dict[str, torch.Tensor],
        list_tensor_data: Dict[str, List[torch.Tensor]],
        list_list_tensor_data: Dict[str, List[List[torch.Tensor]]],
    ) -> Agents:
        """
        Unpacks the data returned from the scriptable core into an Agents feature class.
        :param tensor_data: The tensor data output from the scriptable core.
        :param list_tensor_data: The List[tensor] data output from the scriptable core.
        :param list_tensor_data: The List[List[tensor]] data output from the scriptable core.
        :return: The packed Agents object.
        """
        ego_features = [list_tensor_data["agents.ego"][0].detach().numpy()]
        agent_features = [list_tensor_data["agents.agents"][0].detach().numpy()]

        return Agents(ego=ego_features, agents=agent_features)

    @torch.jit.export
    def scriptable_forward(
        self,
        tensor_data: Dict[str, torch.Tensor],
        list_tensor_data: Dict[str, List[torch.Tensor]],
        list_list_tensor_data: Dict[str, List[List[torch.Tensor]]],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]], Dict[str, List[List[torch.Tensor]]]]:
        """
        Inherited. See interface.
        """
        ego_future: torch.Tensor = tensor_data["future_ego_states"]
        time_stamps: torch.Tensor = tensor_data["future_time_stamps"]
        agents: List[torch.Tensor] = list_tensor_data["future_tracked_objects"]

        anchor_ego_state = ego_future[-1, :].squeeze()
        agent_future = filter_agents_tensor(agents, reverse=True)

        if agent_future[-1].shape[0] == 0:
            # Return zero array when there are no agents in the scene
            agents_tensor: torch.Tensor = torch.zeros((len(agent_future), 0, self._agents_states_dim)).float()
        else:
            padded_agent_states = pad_agent_states(agent_future, reverse=True)

            local_coords_agent_states = convert_absolute_quantities_to_relative(padded_agent_states, anchor_ego_state)

            # Calculate yaw rate
            # yaw_rate_horizon = compute_yaw_rate_from_state_tensors(padded_agent_states, time_stamps)
            yaw_rate_horizon = torch.zeros(len(padded_agent_states), padded_agent_states[0].size(0))

            agents_tensor = pack_agents_tensor(local_coords_agent_states, yaw_rate_horizon)

        ego_tensor = build_ego_features_from_tensor(ego_future, reverse=True)

        output_dict: Dict[str, torch.Tensor] = {}

        output_list_dict: Dict[str, List[torch.Tensor]] = {"agents.ego": [ego_tensor], "agents.agents": [agents_tensor]}

        output_list_list_dict: Dict[str, List[List[torch.Tensor]]] = {}

        return output_dict, output_list_dict, output_list_list_dict

    @torch.jit.export
    def precomputed_feature_config(self) -> Dict[str, Dict[str, str]]:
        """
        Inherited. See interface.
        """
        return {
            "expert_states": {
                "iteration": "0",
                "num_samples": str(self.num_future_poses),
                "time_horizon": str(self.future_time_horizon),
            },
            "future_time_stamps": {
                "iteration": "0",
                "num_samples": str(self.num_future_poses),
                "time_horizon": str(self.future_time_horizon),
            },
            "future_tracked_objects": {
                "iteration": "0",
                "time_horizon": str(self.future_time_horizon),
                "num_samples": str(self.num_future_poses),
            },
        }
