from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any, Dict, List, Union

import numpy as np
import torch
from pyquaternion import Quaternion

from nuplan.planning.script.builders.utils.utils_type import are_the_same_type, validate_type
from nuplan.planning.training.preprocessing.feature_builders.vector_builder_utils import LaneOnRouteStatusData
from nuplan.planning.training.preprocessing.features.abstract_model_feature import (
    AbstractModelFeature,
    FeatureDataType,
    to_tensor,
)

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, TimeDuration, TimePoint


@dataclass
class ScenarioType(AbstractModelFeature):
    """
    Vector map data struture, including:
        coords: List[<np.ndarray: num_lane_segments, 2, 2>].
            The (x, y) coordinates of the start and end point of the lane segments.
        lane_groupings: List[List[<np.ndarray: num_lane_segments_in_lane>]].
            Each lane grouping or polyline is represented by an array of indices of lane segments
            in coords belonging to the given lane. Each batch contains a List of lane groupings.
        multi_scale_connections: List[Dict of {scale: connections_of_scale}].
            Each connections_of_scale is represented by an array of <np.ndarray: num_connections, 2>,
            and each column in the array is [from_lane_segment_idx, to_lane_segment_idx].
        on_route_status: List[<np.ndarray: num_lane_segments, 2>].
            Binary encoding of on route status for lane segment at given index.
            Encoding: off route [0, 1], on route [1, 0], unknown [0, 0]
        traffic_light_data: List[<np.ndarray: num_lane_segments, 4>]
            One-hot encoding of on traffic light status for lane segment at given index.
            Encoding: green [1, 0, 0, 0] yellow [0, 1, 0, 0], red [0, 0, 1, 0], unknown [0, 0, 0, 1]

    In all cases, the top level List represent number of batches. This is a special feature where
    each batch entry can have different size. Similarly, each lane grouping within a batch can have
    a variable number of elements. For that reason, the feature can not be placed to a single tensor,
    and we batch the feature with a custom `collate` function
    """

    scenario_type_idx: int

    @cached_property
    def is_valid(self) -> bool:
        """Inherited, see superclass."""
        return (
            self.scenario_type_idx != 404
        )

    # @property
    # def num_of_batches(self) -> int:
    #     """
    #     :return: number of batches
    #     """
    #     return len(self.scenario_type)


    # def get_scenario_type(self, sample_idx: int) -> FeatureDataType:
    #     """
    #     Retrieve lane coordinates at given sample index.
    #     :param sample_idx: the batch index of interest.
    #     :return: lane coordinate features.
    #     """
    #     return self.scenario_type[sample_idx]

    # @classmethod
    # def collate(cls, batch: List[ScenarioType]) -> ScenarioType:
    #     """Implemented. See interface."""
    #     return ScenarioType(
    #         scenario_type=[data for sample in batch for data in sample.scenario_type],
    #     )

    def to_feature_tensor(self) -> ScenarioType:
        """Implemented. See interface."""
        return ScenarioType(
            scenario_type_idx=to_tensor(torch.tensor([self.scenario_type_idx])).contiguous(),
        )

    def to_device(self, device: torch.device) -> ScenarioType:
        """Implemented. See interface."""
        return ScenarioType(
            scenario_type_idx=self.scenario_type_idx.to(device=device),
        )

    @classmethod
    def deserialize(cls, data: Dict[FeatureDataType, Any]) -> ScenarioType:
        """Implemented. See interface."""
        return ScenarioType(
            scenario_type_idx=data["scenario_type_idx"],
        )

    def unpack(self) -> List[ScenarioType]:
        """Implemented. See interface."""
        return [
            ScenarioType(scenario_type_idx=self.scenario_type_idx)
        ]

@dataclass
class EgoGoal(AbstractModelFeature):
    """
    Vector map data struture, including:
        coords: List[<np.ndarray: num_lane_segments, 2, 2>].
            The (x, y) coordinates of the start and end point of the lane segments.
        lane_groupings: List[List[<np.ndarray: num_lane_segments_in_lane>]].
            Each lane grouping or polyline is represented by an array of indices of lane segments
            in coords belonging to the given lane. Each batch contains a List of lane groupings.
        multi_scale_connections: List[Dict of {scale: connections_of_scale}].
            Each connections_of_scale is represented by an array of <np.ndarray: num_connections, 2>,
            and each column in the array is [from_lane_segment_idx, to_lane_segment_idx].
        on_route_status: List[<np.ndarray: num_lane_segments, 2>].
            Binary encoding of on route status for lane segment at given index.
            Encoding: off route [0, 1], on route [1, 0], unknown [0, 0]
        traffic_light_data: List[<np.ndarray: num_lane_segments, 4>]
            One-hot encoding of on traffic light status for lane segment at given index.
            Encoding: green [1, 0, 0, 0] yellow [0, 1, 0, 0], red [0, 0, 1, 0], unknown [0, 0, 0, 1]

    In all cases, the top level List represent number of batches. This is a special feature where
    each batch entry can have different size. Similarly, each lane grouping within a batch can have
    a variable number of elements. For that reason, the feature can not be placed to a single tensor,
    and we batch the feature with a custom `collate` function
    """

    ego_goal: StateSE2

    @cached_property
    def is_valid(self) -> bool:
        """Inherited, see superclass."""
        return (
            True
        )

    # @property
    # def num_of_batches(self) -> int:
    #     """
    #     :return: number of batches
    #     """
    #     return len(self.scenario_type)


    # def get_scenario_type(self, sample_idx: int) -> FeatureDataType:
    #     """
    #     Retrieve lane coordinates at given sample index.
    #     :param sample_idx: the batch index of interest.
    #     :return: lane coordinate features.
    #     """
    #     return self.scenario_type[sample_idx]

    # @classmethod
    # def collate(cls, batch: List[ScenarioType]) -> ScenarioType:
    #     """Implemented. See interface."""
    #     return ScenarioType(
    #         scenario_type=[data for sample in batch for data in sample.scenario_type],
    #     )

    def to_feature_tensor(self) -> EgoGoal:
        """Implemented. See interface."""
        return EgoGoal(
            ego_goal=to_tensor(torch.tensor([self.ego_goal])).contiguous(),
        )

    def to_device(self, device: torch.device) -> EgoGoal:
        """Implemented. See interface."""
        return EgoGoal(
            ego_goal=self.ego_goal.to(device=device),
        )

    @classmethod
    def deserialize(cls, data: Dict[FeatureDataType, Any]) -> EgoGoal:
        """Implemented. See interface."""
        return EgoGoal(
            ego_goal=data["ego_goal"],
        )

    def unpack(self) -> List[EgoGoal]:
        """Implemented. See interface."""
        return [
            EgoGoal(ego_goal=self.ego_goal)
        ]