
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Type

import torch
from torch import Tensor

from nuplan.common.actor_state.state_representation import Point2D, StateSE2
from nuplan.common.geometry.torch_geometry import coordinates_to_local_frame
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario


from nuplan.planning.training.preprocessing.features.abstract_model_feature import AbstractModelFeature
from nuplan.planning.training.preprocessing.features.vector_map import VectorMap
from nuplan.planning.training.preprocessing.features.agents import Agents
from nuplan.planning.training.preprocessing.feature_builders.vector_map_feature_builder import VectorMapFeatureBuilder
from nuplan.planning.training.preprocessing.feature_builders.agents_feature_builder import AgentsFeatureBuilder
from nuplan.planning.training.preprocessing.target_builders.ego_trajectory_target_builder import EgoTrajectoryTargetBuilder
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory
from nuplan.planning.training.preprocessing.features.abstract_model_feature import AbstractModelFeature
from nuplan.planning.training.preprocessing.target_builders.abstract_target_builder import AbstractTargetBuilder
from nuplan.planning.training.preprocessing.features.tensor_target import TensorTarget

import numpy as np
from numpy.typing import NDArray
import itertools



from nuplan.planning.training.preprocessing.feature_builders.scriptable_feature_builder import ScriptableFeatureBuilder
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import AbstractFeatureBuilder
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization, PlannerInput
from nuplan.planning.training.preprocessing.features.scenario_type import ScenarioType


from nuplan.planning.training.preprocessing.features.abstract_model_feature import (
    FeatureDataType,
)


# def coords_to_map_attr(coords) -> NDArray:
#     """Map coordinates in VectorMap format to AutoBots features

#     Args:
#         coords NDArray with shape [num_segments, 2, 2]: one element in coords List of VectorMap

#     Returns:
#         point_feature_tab NDArray with shape [num_segments, 4]: 4 attributes are x, y, angles, existence mask
#     """
#     vec = np.squeeze(coords[:,1,:])-np.squeeze(coords[:,0,:])
#     angles=np.arctan2(vec[:,1], vec[:,0]).reshape((-1, 1))

#     # get point feature tablular of shape [p_total, 3]
#     point_feature_tab=np.concatenate((np.squeeze(coords[:,0,:]), angles), axis=1)
#     point_feature_tab=np.pad(point_feature_tab, ((0, 0), (0, 1)), "constant", constant_values=(1))

#     # [TODO] omit the first point as [0, 0, 0], greatly simplify the process
#     point_feature_tab[0,:]=np.zeros((1,4)) 
#     return point_feature_tab


# # This class is unused
# class AutobotsMapFeatureBuilder(VectorMapFeatureBuilder):
#     @torch.jit.unused
#     def get_feature_type(self) -> Type[AbstractModelFeature]:
#         """Inherited, see superclass."""
#         return Tensor # type: ignore

#     @torch.jit.unused
#     @classmethod
#     def get_feature_unique_name(cls) -> str:
#         """Inherited, see superclass."""
#         return "tensor_map"

#     @torch.jit.unused
#     def get_features_from_scenario(self, scenario: AbstractScenario) -> Tensor:
#         vec_map=super(AutobotsMapFeatureBuilder, self).get_features_from_scenario(scenario)
#         return self.VectorMapToAutobotsMapTensor(vec_map)

#     @torch.jit.unused
#     def get_features_from_simulation(
#         self, current_input: PlannerInput, initialization: PlannerInitialization
#     ) -> Tensor:
#         vec_map=super(AutobotsMapFeatureBuilder, self).get_features_from_simulation(current_input, initialization)
#         return self.VectorMapToAutobotsMapTensor(vec_map)

#     @torch.jit.unused 
#     def VectorMapToAutobotsMapTensor(self, vec_map: VectorMap):
#         """_summary_

#         Args:
#             scenario (AbstractScenario): see base class

#         Returns:
#             Tensor: shape [B, S, P, map_attr+1] example [64,100,40,4]
#         """

#         # B=len(vec_map.coords) # get the number of batches

#         # TODO: S and P dimension must be bigger than that of the original data
#         S=200 # 100
#         P=300 # 40

#         # to debug: check the maximum number of segment contained in one lane
#         lengths = [[len(x) for x in sublist] for sublist in vec_map.lane_groupings]
#         length_maxes=[ np.max(np.array(x)) for x in lengths]
#         max_length = np.max(length_maxes)

#         padded_list_list = [[np.pad(x, (0, max(P - len(x), 0)), 'constant') for x in sublist] for sublist in vec_map.lane_groupings]
#         # [TODO]if P < len(x) ??
#         # if you experience exception here, it may be the presence of P < len(x). Check max_length and P values.
#         list_of_idx_array = [ np.array(l, np.float64) for l in padded_list_list] # l's shape = [num_lane, P]
#         list_of_feature_array = [ coords_to_map_attr(coord_mat) for coord_mat in vec_map.coords]

#         lane_features = [ feature[idx.astype(np.int32)] for idx, feature in zip(list_of_idx_array, list_of_feature_array)]

#         # ((pad_top, pad_bottom), (pad_left, pad_right))
#         padded_list = [ np.pad(arr, ((0, S-arr.shape[0]), (0, 0), (0, 0)), 'constant')  for arr in lane_features] # get list of array of shape [S, P]

#         map_autobots=np.array(padded_list, np.float64) # map_autobots shape is [B, S, P, 4]

#         return Tensor(map_autobots)

# # This class is unused
# class AutobotsAgentsFeatureBuilder(AgentsFeatureBuilder):
#     @torch.jit.unused
#     def get_feature_type(self) -> Type[AbstractModelFeature]:
#         """Inherited, see superclass."""
#         return Tensor # type: ignore

#     @torch.jit.unused
#     @classmethod
#     def get_feature_unique_name(cls) -> str:
#         """Inherited, see superclass."""
#         return "tensor_agents"

#     @torch.jit.unused
#     def get_features_from_scenario(self, scenario: AbstractScenario) -> Tensor:
#         agent=super(AutobotsAgentsFeatureBuilder, self).get_features_from_scenario(scenario)
#         return self.AgentsToAutobotsAgentsTensor(agent)

#     @torch.jit.unused
#     def get_features_from_simulation(
#         self, current_input: PlannerInput, initialization: PlannerInitialization
#     ) -> Tensor:
#         agent=super(AutobotsAgentsFeatureBuilder, self).get_features_from_simulation(current_input, initialization)
#         return self.AgentsToAutobotsAgentsTensor(agent)

#     @torch.jit.unused 
#     def AgentsToAutobotsAgentsTensor(self, agents: Agents):
#         """_summary_

#         Args:
#             scenario (AbstractScenario): see base class

#         Returns:
#             Tensor: [B, T_obs, M-1, k_attr+1] example [64,4,7,3]
#         """

#         # every scenario may have different number of agents
#         M_minus_1 = 80 # maximum agent number

#         lengths = [x.shape[1] for x in agents.agents]
#         length_maxes=[ np.max(np.array(x)) for x in lengths]
#         max_length = np.max(length_maxes)

#         padded_list=[ np.pad(arr, ((0, 0), (0, max(M_minus_1-arr.shape[1], 0)), (0, 0)), 'constant')  for arr in agents.agents]

#         extended_list= [np.expand_dims(x, 0) for x in padded_list]

#         agents_ts= np.concatenate(extended_list, 0)

#         agents_ts=agents_ts[:,:,:,0:3] # take only x, y coordinates and an addtional dimension to be existence mask

#         agents_ts[:,:,:,2] = (agents_ts[:,:,:,2]!=0)


#         return Tensor(agents_ts)


class ScenarioTypeFeatureBuilder(AbstractFeatureBuilder):
    """
    Feature builder for constructing map features in a vector-representation.
    """

    def __init__(self) -> None:
        """
        Initialize vector map builder with configuration parameters.
        :param radius:  The query radius scope relative to the current ego-pose.
        :param connection_scales: Connection scales to generate. Use the 1-hop connections if it's left empty.
        :return: Vector map data including lane segment coordinates and connections within the given range.
        """
        super().__init__()
        self.test_scenario_types=[
            "starting_straight_traffic_light_intersection_traversal",
            "high_lateral_acceleration",
            "changing_lane",
            "high_magnitude_speed",
            "low_magnitude_speed",
            "starting_left_turn",
            "starting_right_turn",
            "stopping_with_lead",
            "following_lane_with_lead",
            "near_multiple_vehicles",
            "traversing_pickup_dropoff",
            "behind_long_vehicle",
            "waiting_for_pedestrian_to_cross",
            "stationary_in_traffic"
        ]

    @torch.jit.unused
    def get_feature_type(self) -> Type[AbstractModelFeature]:
        """Inherited, see superclass."""
        return ScenarioType  # type: ignore

    @torch.jit.unused
    @classmethod
    def get_feature_unique_name(cls) -> str:
        """Inherited, see superclass."""
        return "scenario_type"

    @torch.jit.unused
    def get_features_from_scenario(self, scenario: AbstractScenario) -> ScenarioType:
        """Inherited, see superclass."""
        with torch.no_grad():
            scenario_type = scenario.scenario_type

            try:
                scenario_type_idx: FeatureDataType = torch.Tensor([self.test_scenario_types.index(scenario_type)])
            except:
                print("scenario_type not in test_scenario_types")

            return ScenarioType(scenario_type=scenario_type_idx)
        
    @torch.jit.unused
    def get_features_from_simulation(self, current_input: PlannerInput, initialization: PlannerInitialization) -> ScenarioType:
        return ScenarioType(scenario_type="ADD_from_simulation")
    
#####################################################################################################3

# This class is unused
class AutobotsTargetBuilder(EgoTrajectoryTargetBuilder):
    """Trajectory builders constructed the desired ego's trajectory from a scenario."""
    @classmethod
    def get_feature_unique_name(cls) -> str:
        """Inherited, see superclass."""
        return "tensor_trajectory"

    @classmethod
    def get_feature_type(cls) -> Type[AbstractModelFeature]:
        """Inherited, see superclass."""
        return Tensor  # type: ignore

    def get_targets(self, scenario: AbstractScenario) -> Tensor:
        targets = super(AutobotsTargetBuilder, self).get_targets(scenario)

        # since in AutoBots, the last colomn of values are existence mask, not heading direction angles, 
        # we overwrite them all with 1

        targets=self.TrajectoryToAutobotsTarget(targets)

        return Tensor(targets)

    def TrajectoryToAutobotsTarget(self, target: Trajectory) -> Tensor:
        """_summary_
        Args:
            target (Trajectory): attribute data: either a [num_batches, num_states, 3] or [num_states, 3] representing the trajectory
                 where se2_state is [x, y, heading] with units [meters, meters, radians].
        Returns:
            Tensor: _description_
        """

        target_ts=torch.as_tensor(target.data)
        if target_ts.dim() == 2:
            target_ts=torch.unsqueeze(target_ts, 0) # if two dimension, unsqueeze to create one more "batch" dimension
        target_ts[:,:,2]=0
        return target_ts

class AutobotsPredNominalTargetBuilder(AbstractTargetBuilder):
    def get_feature_unique_name(cls) -> str:
        """Inherited, see superclass."""
        return "pred"

    @classmethod
    def get_feature_type(cls) -> Type[AbstractModelFeature]:
        """Inherited, see superclass."""
        return TensorTarget  # type: ignore

    def get_targets(self, scenario: AbstractScenario) -> Tensor:

        nominal_target = TensorTarget(data=np.zeros((2,2)))
        return nominal_target


class AutobotsModeProbsNominalTargetBuilder(AbstractTargetBuilder):
    def get_feature_unique_name(cls) -> str:
        """Inherited, see superclass."""
        return "mode_probs"

    @classmethod
    def get_feature_type(cls) -> Type[AbstractModelFeature]:
        """Inherited, see superclass."""
        return TensorTarget  # type: ignore

    def get_targets(self, scenario: AbstractScenario) -> Tensor:

        nominal_target = TensorTarget(data=np.zeros((2,2)))
        return nominal_target

