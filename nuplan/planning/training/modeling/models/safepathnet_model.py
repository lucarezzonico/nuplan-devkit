from dataclasses import dataclass
from typing import Dict, List, Tuple, cast, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

# from l5kit.planning.vectorized.common import (
#     build_matrix,
#     build_target_normalization,
#     pad_avail,
#     pad_points,
#     transform_points
# )

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.modeling.models.urban_driver_open_loop_model_utils import (
    LocalSubGraph,
    MultiheadAttentionGlobalHead,
    SinusoidalPositionalEmbedding,
    TypeEmbedding,
    pad_avails,
    pad_polylines,
    pad_avails_batch,
    pad_polylines_batch,
)

from nuplan.planning.training.modeling.models.urban_driver_closed_loop_model_utils import (
    transform_points,
    update_transformation_matrices,
    build_target_normalization
)

from nuplan.planning.training.modeling.models.safepathnet_model_utils import (
    MultimodalDecoder,
    TrajectoryMatcher,
    build_matrix
)

from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.modeling.types import FeaturesType, TargetsType
from nuplan.planning.training.preprocessing.feature_builders.generic_agents_feature_builder import (
    GenericAgentsFeatureBuilder,
)
from nuplan.planning.training.preprocessing.feature_builders.vector_set_map_feature_builder import (
    VectorSetMapFeatureBuilder,
)
from nuplan.planning.training.preprocessing.features.generic_agents import GenericAgents
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory
from nuplan.planning.training.preprocessing.features.trajectories import Trajectories
from nuplan.planning.training.preprocessing.features.vector_set_map import VectorSetMap
from nuplan.planning.training.preprocessing.target_builders.ego_trajectory_target_builder import (
    EgoTrajectoryTargetBuilder,
)

from nuplan.planning.training.preprocessing.feature_builders.generic_expert_feature_builder import GenericExpertFeatureBuilder
from nuplan.planning.training.preprocessing.target_builders.expert_trajectory_target_builder import (
    ExpertTrajectoryTargetBuilder,
)

from nuplan.planning.training.preprocessing.features.tensor_target import TensorTarget

import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class SafePathNetModelParams:
    """
    Parameters for UrbanDriverOpenLoop model.
        local_embedding_size: embedding dimensionality of local subgraph layers.
        global_embedding_size: embedding dimensionality of global attention layers.
        num_subgraph_layers: number of stacked PointNet-like local subgraph layers.
        global_head_dropout: float in range [0,1] for the dropout in the MHA global head. Set to 0 to disable it.
    """

    local_embedding_size: int
    global_embedding_size: int
    feedforward_embedding_size: int
    num_encoder_layers: int
    num_decoder_layers: int
    num_subgraph_layers: int
    global_head_dropout: float
    
    detach_unroll: bool        # if to detach between steps when training with unroll
    warmup_num_frames: int     # K "sample" warmup_num_steps by following the model's policy
    discount_factor: float     # discount future timesteps via discount_factor**t
    limit_predicted_yaw: bool
    
    agent_num_trajectories: int = 20
    use_perturbation: bool = False
    perturbation_probability: float = 1.0
    use_gaussian_perturbation: bool = False
    perturbation_min_displacement_m: float = 1.2
    cost_prob_coeff: float = 0.1


@dataclass
class SafePathNetModelFeatureParams:
    """
    Parameters for UrbanDriverOpenLoop features.
        feature_types: List of feature types (agent and map) supported by model. Used in type embedding layer.
        past_time_steps: maximum number of points per element, to maintain fixed sized features.
        future_time_steps: maximum number of points per element, to maintain fixed sized features.
        feature_dimension: feature size, to maintain fixed sized features.
        agent_features: Agent features to request from agent feature builder.
        ego_dimension: Feature dimensionality to keep from ego features.
        agent_dimension: Feature dimensionality to keep from agent features.
        max_agents: maximum number of agents, to maintain fixed sized features.
        past_trajectory_sampling: Sampling parameters for past trajectory.
        map_features: Map features to request from vector set map feature builder.
        max_elements: Maximum number of elements to extract per map feature layer.
        max_points: Maximum number of points per feature to extract per map feature layer.
        vector_set_map_feature_radius: The query radius scope relative to the current ego-pose.
        interpolation_method: Interpolation method to apply when interpolating to maintain fixed size map elements.
        disable_map: whether to ignore map.
        disable_agents: whether to ignore agents.
    """
    
    feature_types: Dict[str, int]
    total_max_points: int
    past_time_steps: int
    future_time_steps: int
    feature_dimension: int
    history_num_frames_ego: int
    history_num_frames_agents: int
    agent_features: List[str]
    ego_dimension: int
    agent_dimension: int
    max_agents: int
    past_trajectory_sampling: TrajectorySampling
    map_features: List[str]
    max_elements: Dict[str, int]
    max_points: Dict[str, int]
    vector_set_map_feature_radius: int
    interpolation_method: str
    disable_map: bool
    disable_agents: bool

    def __post_init__(self) -> None:
        """
        Sanitize feature parameters.
        :raise AssertionError if parameters invalid.
        """
        if not self.total_max_points > 0:
            raise AssertionError(f"Total max points must be >0! Got: {self.total_max_points}")
        
        if not self.past_time_steps > 0:
            raise AssertionError(f"past_time_steps must be >0! Got: {self.past_time_steps}")

        if not self.future_time_steps > 0:
            raise AssertionError(f"future_time_steps must be >0! Got: {self.future_time_steps}")

        if not self.feature_dimension >= 2:
            raise AssertionError(f"Feature dimension must be >=2! Got: {self.feature_dimension}")

        # sanitize feature types
        for feature_name in ["NONE", "EGO"]:
            if feature_name not in self.feature_types:
                raise AssertionError(f"{feature_name} must be among feature types! Got: {self.feature_types}")

        self._sanitize_agent_features()
        self._sanitize_map_features()

    def _sanitize_agent_features(self) -> None:
        """
        Sanitize agent feature parameters.
        :raise AssertionError if parameters invalid.
        """
        if "EGO" in self.agent_features:
            raise AssertionError("EGO must not be among agent features!")
        for feature_name in self.agent_features:
            if feature_name not in self.feature_types:
                raise AssertionError(f"Agent feature {feature_name} not in feature_types: {self.feature_types}!")

    def _sanitize_map_features(self) -> None:
        """
        Sanitize map feature parameters.
        :raise AssertionError if parameters invalid.
        """
        for feature_name in self.map_features:
            if feature_name not in self.feature_types:
                raise AssertionError(f"Map feature {feature_name} not in feature_types: {self.feature_types}!")
            if feature_name not in self.max_elements:
                raise AssertionError(f"Map feature {feature_name} not in max_elements: {self.max_elements.keys()}!")
            if feature_name not in self.max_points:
                raise AssertionError(f"Map feature {feature_name} not in max_points types: {self.max_points.keys()}!")


@dataclass
class SafePathNetModelTargetParams:
    """
    Parameters for UrbanDriverOpenLoop targets.
        num_output_features: number of target features.
        future_trajectory_sampling: Sampling parameters for future trajectory.
    """

    num_output_features: int
    future_trajectory_sampling: TrajectorySampling
    expert_trajectory_sampling: TrajectorySampling

def convert_predictions_to_trajectory(predictions: torch.Tensor) -> torch.Tensor:
    """
    Convert predictions tensor to Trajectory.data shape
    :param predictions: tensor from network
    :return: data suitable for Trajectory
    """
    num_batches = predictions.shape[0]
    return predictions.view(num_batches, -1, Trajectory.state_size())


class SafePathNetModel(TorchModuleWrapper):
    """
    Vector-based model that uses PointNet-based subgraph layers for collating loose collections of vectorized inputs
    into local feature descriptors to be used as input to a global Transformer.

    Adapted from L5Kit's implementation of "Urban Driver: Learning to Drive from Real-world Demonstrations
    Using Policy Gradients":
    https://github.com/woven-planet/l5kit/blob/master/l5kit/l5kit/planning/vectorized/open_loop_model.py
    Only the open-loop  version of the model is here represented, with slight modifications to fit the nuPlan framework.
    Changes:
        1. Use nuPlan features from NuPlanScenario
        2. Format model for using pytorch_lightning
    """

    def __init__(
        self,
        model_params: SafePathNetModelParams,
        feature_params: SafePathNetModelFeatureParams,
        target_params: SafePathNetModelTargetParams,
    ):
        """
        Initialize UrbanDriverOpenLoop model.
        :param model_params: internal model parameters.
        :param feature_params: agent and map feature parameters.
        :param target_params: target parameters.
        """
        super().__init__(
            feature_builders=[
                VectorSetMapFeatureBuilder(
                    map_features=feature_params.map_features,
                    max_elements=feature_params.max_elements,
                    max_points=feature_params.max_points,
                    radius=feature_params.vector_set_map_feature_radius,
                    interpolation_method=feature_params.interpolation_method,
                ),
                GenericAgentsFeatureBuilder(feature_params.agent_features, feature_params.past_trajectory_sampling),
                GenericExpertFeatureBuilder(feature_params.agent_features, target_params.expert_trajectory_sampling),
            ],
            target_builders=[
                EgoTrajectoryTargetBuilder(target_params.future_trajectory_sampling),
            ],
            future_trajectory_sampling=target_params.future_trajectory_sampling,
        )
        self._model_params = model_params
        self._feature_params = feature_params
        self._target_params = target_params

        self.feature_embedding = nn.Linear(in_features=self._feature_params.feature_dimension,
                                           out_features=self._model_params.local_embedding_size) # (self._vector_agent_length, self._d_local)
        self.positional_embedding = SinusoidalPositionalEmbedding(embedding_size=self._model_params.local_embedding_size)
        self.type_embedding = TypeEmbedding(embedding_dim=self._model_params.global_embedding_size,
                                            feature_types=self._feature_params.feature_types)
        self.local_subgraph = LocalSubGraph(num_layers=self._model_params.num_subgraph_layers,
                                            dim_in=self._model_params.local_embedding_size)
        # if self._model_params.global_embedding_size != self._model_params.local_embedding_size:
        #     self.global_from_local = nn.Linear(in_features=self._model_params.local_embedding_size,
        #                                        out_features=self._model_params.global_embedding_size)
        
        # num_timesteps = self.future_trajectory_sampling.num_poses
        # num_timesteps = 1
        # self.global_head = MultiheadAttentionGlobalHead(
        #     self._model_params.global_embedding_size,
        #     num_timesteps,
        #     self._target_params.num_output_features // num_timesteps,
        #     dropout=self._model_params.global_head_dropout,
        # )
        
        ########### Closed-Loop Model ###########

        self._history_num_frames_ego = 4
        self._history_num_frames_agents = 4
        
        ########### SafePathNet Model ###########
        
        self.global_head = MultimodalDecoder(
            dim_in=self._model_params.local_embedding_size,
            projection_dim=self._model_params.global_embedding_size,
            dim_feedforward=self._model_params.feedforward_embedding_size,
            num_layers=self._model_params.num_encoder_layers,
            num_decode_layers=self._model_params.num_decoder_layers,
            num_outputs=self._target_params.num_output_features,
            future_num_frames=self._feature_params.future_time_steps,
            agent_future_num_frames=self._feature_params.future_time_steps,
            agent_num_trajectories=self._model_params.agent_num_trajectories,
            max_num_agents=self._feature_params.max_agents
        )

        self.agent_traj_matcher = TrajectoryMatcher(cost_prob_coeff=self._model_params.cost_prob_coeff)
        self.eps = float(torch.finfo(torch.float).eps)
        
        self.criterion = torch.nn.modules.loss.L1Loss(reduction='none')
        
        
    def name(self) -> str:
        return self.__class__.__name__
    
    def extract_future_agent_features(
        self, ego_agent_features: GenericAgents, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract ego and agent features into format expected by network and build accompanying availability matrix.
        :param ego_agent_features: agent features to be extracted (ego + other agents)
        :param batch_size: number of samples in batch to extract
        :return:
            agent_features: <torch.FloatTensor: batch_size, num_elements (polylines) (1+max_agents*num_agent_types),
                num_points_per_element, feature_dimension>. Stacked ego, agent, and map features.
            agent_avails: <torch.BoolTensor: batch_size, num_elements (polylines) (1+max_agents*num_agent_types),
                num_points_per_element>. Bool specifying whether feature is available or zero padded.
        """
        agent_features = []  # List[<torch.FloatTensor: max_agents+1, total_max_points, feature_dimension>: batch_size]
        agent_avails = []  # List[<torch.BoolTensor: max_agents+1, total_max_points>: batch_size]

        # features have different size across batch so we use per sample feature extraction
        for sample_idx in range(batch_size):
            # Ego features
            # maintain fixed feature size through trimming/padding
            sample_ego_feature = ego_agent_features.ego[sample_idx][
                ..., : min(self._feature_params.ego_dimension, self._feature_params.feature_dimension)
            ].unsqueeze(0)
            if (
                min(self._feature_params.ego_dimension, GenericAgents.ego_state_dim())
                < self._feature_params.feature_dimension
            ):
                sample_ego_feature = pad_polylines(sample_ego_feature, self._feature_params.feature_dimension, dim=2)

            sample_ego_avails = torch.ones(
                sample_ego_feature.shape[0],
                sample_ego_feature.shape[1],
                dtype=torch.bool,
                device=sample_ego_feature.device,
            )

            # Don't reverse for future timesteps
            # reverse points so frames are in reverse chronological order, i.e. (t_0, t_-1, ..., t_-N)
            # sample_ego_feature = torch.flip(sample_ego_feature, dims=[1])

            # maintain fixed number of points per polyline
            sample_ego_feature = sample_ego_feature[:, : self._feature_params.total_max_points, ...]    # [1, 16, 3]
            sample_ego_avails = sample_ego_avails[:, : self._feature_params.total_max_points, ...]      # [1, 16]
            if sample_ego_feature.shape[1] < self._feature_params.total_max_points:
                sample_ego_feature = pad_polylines(sample_ego_feature, self._feature_params.total_max_points, dim=1)
                sample_ego_avails = pad_avails(sample_ego_avails, self._feature_params.total_max_points, dim=1)

            sample_features = [sample_ego_feature] # list([1, 16, 3])
            sample_avails = [sample_ego_avails]    # list([1, 16])

            # Agent features
            for feature_name in self._feature_params.agent_features: # VEHICLE
                # if there exist at least one valid agent in the sample
                if ego_agent_features.has_agents(feature_name, sample_idx):
                    # num_frames x num_agents x num_features -> num_agents x num_frames x num_features
                    sample_agent_features = torch.permute(
                        ego_agent_features.agents[feature_name][sample_idx], (1, 0, 2)
                    )
                    # maintain fixed feature size through trimming/padding
                    sample_agent_features = sample_agent_features[
                        ..., : min(self._feature_params.agent_dimension, self._feature_params.feature_dimension)
                    ]
                    if (
                        min(self._feature_params.agent_dimension, GenericAgents.agents_states_dim())
                        < self._feature_params.feature_dimension
                    ):
                        sample_agent_features = pad_polylines(
                            sample_agent_features, self._feature_params.feature_dimension, dim=2
                        )

                    sample_agent_avails = torch.ones(
                        sample_agent_features.shape[0],
                        sample_agent_features.shape[1],
                        dtype=torch.bool,
                        device=sample_agent_features.device,
                    )

                    # reverse points so frames are in reverse chronological order, i.e. (t_0, t_-1, ..., t_-N)
                    # sample_agent_features = torch.flip(sample_agent_features, dims=[1])

                    # maintain fixed number of points per polyline
                    sample_agent_features = sample_agent_features[:, : self._feature_params.total_max_points, ...]
                    sample_agent_avails = sample_agent_avails[:, : self._feature_params.total_max_points, ...]
                    if sample_agent_features.shape[1] < self._feature_params.total_max_points:
                        sample_agent_features = pad_polylines(
                            sample_agent_features, self._feature_params.total_max_points, dim=1
                        )
                        sample_agent_avails = pad_avails(
                            sample_agent_avails, self._feature_params.total_max_points, dim=1
                        )

                    # maintained fixed number of agent polylines of each type per sample
                    sample_agent_features = sample_agent_features[: self._feature_params.max_agents, ...]
                    sample_agent_avails = sample_agent_avails[: self._feature_params.max_agents, ...]
                    if sample_agent_features.shape[0] < (self._feature_params.max_agents):
                        sample_agent_features = pad_polylines(
                            sample_agent_features, self._feature_params.max_agents, dim=0
                        )
                        sample_agent_avails = pad_avails(sample_agent_avails, self._feature_params.max_agents, dim=0)

                else:
                    sample_agent_features = torch.zeros(
                        self._feature_params.max_agents,
                        self._feature_params.total_max_points,
                        self._feature_params.feature_dimension,
                        dtype=torch.float32,
                        device=sample_ego_feature.device,
                    )
                    sample_agent_avails = torch.zeros(
                        self._feature_params.max_agents,
                        self._feature_params.total_max_points,
                        dtype=torch.bool,
                        device=sample_agent_features.device,
                    )

                # add features, avails to sample
                sample_features.append(sample_agent_features)
                sample_avails.append(sample_agent_avails)

            sample_features = torch.cat(sample_features, dim=0)
            sample_avails = torch.cat(sample_avails, dim=0)

            agent_features.append(sample_features)
            agent_avails.append(sample_avails)
        agent_features = torch.stack(agent_features)
        agent_avails = torch.stack(agent_avails)

        return agent_features, agent_avails
    
    def extract_agent_features(
        self, ego_agent_features: GenericAgents, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract ego and agent features into format expected by network and build accompanying availability matrix.
        :param ego_agent_features: agent features to be extracted (ego + other agents)
        :param batch_size: number of samples in batch to extract
        :return:
            agent_features: <torch.FloatTensor: batch_size, num_elements (polylines) (1+max_agents*num_agent_types),
                num_points_per_element, feature_dimension>. Stacked ego, agent, and map features.
            agent_avails: <torch.BoolTensor: batch_size, num_elements (polylines) (1+max_agents*num_agent_types),
                num_points_per_element>. Bool specifying whether feature is available or zero padded.
        """
        agent_features = []  # List[<torch.FloatTensor: max_agents+1, total_max_points, feature_dimension>: batch_size]
        agent_avails = []  # List[<torch.BoolTensor: max_agents+1, total_max_points>: batch_size]

        # features have different size across batch so we use per sample feature extraction
        for sample_idx in range(batch_size):
            # Ego features
            # maintain fixed feature size through trimming/padding
            sample_ego_feature = ego_agent_features.ego[sample_idx][
                ..., : min(self._feature_params.ego_dimension, self._feature_params.feature_dimension)
            ].unsqueeze(0)
            if (
                min(self._feature_params.ego_dimension, GenericAgents.ego_state_dim())
                < self._feature_params.feature_dimension
            ):
                sample_ego_feature = pad_polylines(sample_ego_feature, self._feature_params.feature_dimension, dim=2)

            sample_ego_avails = torch.ones(
                sample_ego_feature.shape[0],
                sample_ego_feature.shape[1],
                dtype=torch.bool,
                device=sample_ego_feature.device,
            )

            # reverse points so frames are in reverse chronological order, i.e. (t_0, t_-1, ..., t_-N)
            sample_ego_feature = torch.flip(sample_ego_feature, dims=[1])

            # maintain fixed number of points per polyline
            sample_ego_feature = sample_ego_feature[:, : self._feature_params.total_max_points, ...]
            sample_ego_avails = sample_ego_avails[:, : self._feature_params.total_max_points, ...]
            if sample_ego_feature.shape[1] < self._feature_params.total_max_points:
                sample_ego_feature = pad_polylines(sample_ego_feature, self._feature_params.total_max_points, dim=1)
                sample_ego_avails = pad_avails(sample_ego_avails, self._feature_params.total_max_points, dim=1)

            sample_features = [sample_ego_feature]
            sample_avails = [sample_ego_avails]

            # Agent features
            for feature_name in self._feature_params.agent_features:
                # if there exist at least one valid agent in the sample
                if ego_agent_features.has_agents(feature_name, sample_idx):
                    # num_frames x num_agents x num_features -> num_agents x num_frames x num_features
                    sample_agent_features = torch.permute(
                        ego_agent_features.agents[feature_name][sample_idx], (1, 0, 2)
                    )
                    # maintain fixed feature size through trimming/padding
                    sample_agent_features = sample_agent_features[
                        ..., : min(self._feature_params.agent_dimension, self._feature_params.feature_dimension)
                    ]
                    if (
                        min(self._feature_params.agent_dimension, GenericAgents.agents_states_dim())
                        < self._feature_params.feature_dimension
                    ):
                        sample_agent_features = pad_polylines(
                            sample_agent_features, self._feature_params.feature_dimension, dim=2
                        )

                    sample_agent_avails = torch.ones(
                        sample_agent_features.shape[0],
                        sample_agent_features.shape[1],
                        dtype=torch.bool,
                        device=sample_agent_features.device,
                    )

                    # reverse points so frames are in reverse chronological order, i.e. (t_0, t_-1, ..., t_-N)
                    sample_agent_features = torch.flip(sample_agent_features, dims=[1])

                    # maintain fixed number of points per polyline
                    sample_agent_features = sample_agent_features[:, : self._feature_params.total_max_points, ...]
                    sample_agent_avails = sample_agent_avails[:, : self._feature_params.total_max_points, ...]
                    if sample_agent_features.shape[1] < self._feature_params.total_max_points:
                        sample_agent_features = pad_polylines(
                            sample_agent_features, self._feature_params.total_max_points, dim=1
                        )
                        sample_agent_avails = pad_avails(
                            sample_agent_avails, self._feature_params.total_max_points, dim=1
                        )

                    # maintained fixed number of agent polylines of each type per sample
                    sample_agent_features = sample_agent_features[: self._feature_params.max_agents, ...]
                    sample_agent_avails = sample_agent_avails[: self._feature_params.max_agents, ...]
                    if sample_agent_features.shape[0] < (self._feature_params.max_agents):
                        sample_agent_features = pad_polylines(
                            sample_agent_features, self._feature_params.max_agents, dim=0
                        )
                        sample_agent_avails = pad_avails(sample_agent_avails, self._feature_params.max_agents, dim=0)

                else:
                    sample_agent_features = torch.zeros(
                        self._feature_params.max_agents,
                        self._feature_params.total_max_points,
                        self._feature_params.feature_dimension,
                        dtype=torch.float32,
                        device=sample_ego_feature.device,
                    )
                    sample_agent_avails = torch.zeros(
                        self._feature_params.max_agents,
                        self._feature_params.total_max_points,
                        dtype=torch.bool,
                        device=sample_agent_features.device,
                    )

                # add features, avails to sample
                sample_features.append(sample_agent_features)
                sample_avails.append(sample_agent_avails)

            sample_features = torch.cat(sample_features, dim=0)
            sample_avails = torch.cat(sample_avails, dim=0)

            agent_features.append(sample_features)
            agent_avails.append(sample_avails)
        agent_features = torch.stack(agent_features)
        agent_avails = torch.stack(agent_avails)

        return agent_features, agent_avails

    def extract_map_features(
        self, vector_set_map_data: VectorSetMap, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract map features into format expected by network and build accompanying availability matrix.
        :param vector_set_map_data: VectorSetMap features to be extracted
        :param batch_size: number of samples in batch to extract
        :return:
            map_features: <torch.FloatTensor: batch_size, num_elements (polylines) (max_lanes),
                num_points_per_element, feature_dimension>. Stacked map features.
            map_avails: <torch.BoolTensor: batch_size, num_elements (polylines) (max_lanes),
                num_points_per_element>. Bool specifying whether feature is available or zero padded.
        """
        map_features = []  # List[<torch.FloatTensor: max_map_features, total_max_points, feature_dim>: batch_size]
        map_avails = []  # List[<torch.BoolTensor: max_map_features, total_max_points>: batch_size]

        # features have different size across batch so we use per sample feature extraction
        for sample_idx in range(batch_size):

            sample_map_features = []
            sample_map_avails = []

            for feature_name in self._feature_params.map_features:
                coords = vector_set_map_data.coords[feature_name][sample_idx]
                tl_data = (
                    vector_set_map_data.traffic_light_data[feature_name][sample_idx]
                    if feature_name in vector_set_map_data.traffic_light_data
                    else None
                )
                avails = vector_set_map_data.availabilities[feature_name][sample_idx]

                # add traffic light data if exists for feature
                if tl_data is not None:
                    coords = torch.cat((coords, tl_data), dim=2)

                # maintain fixed number of points per map element (polyline)
                coords = coords[:, : self._feature_params.total_max_points, ...]
                avails = avails[:, : self._feature_params.total_max_points]

                if coords.shape[1] < self._feature_params.total_max_points:
                    coords = pad_polylines(coords, self._feature_params.total_max_points, dim=1)
                    avails = pad_avails(avails, self._feature_params.total_max_points, dim=1)

                # maintain fixed number of features per point
                coords = coords[..., : self._feature_params.feature_dimension]
                if coords.shape[2] < self._feature_params.feature_dimension:
                    coords = pad_polylines(coords, self._feature_params.feature_dimension, dim=2)

                sample_map_features.append(coords)
                sample_map_avails.append(avails)

            map_features.append(torch.cat(sample_map_features))
            map_avails.append(torch.cat(sample_map_avails))

        map_features = torch.stack(map_features)
        map_avails = torch.stack(map_avails)

        return map_features, map_avails

    def forward_open_loop(self, features: FeaturesType) -> TargetsType:
        """
        Predict
        :param features: input features containing
                        {
                            "vector_set_map": VectorSetMap,
                            "generic_agents": GenericAgents,
                            "expert": GenericAgents, (future)
                        }
        :return: targets: predictions from network
                        {
                            "trajectory": Trajectory,
                            "target": Trajectory,
                        }
        """
        # Recover features
        vector_set_map_data = cast(VectorSetMap, features["vector_set_map"])
        ego_agent_features = cast(GenericAgents, features["generic_agents"])
        batch_size = ego_agent_features.batch_size
        # batchify expert's future trajectory
        target = torch.stack(features["generic_expert"].ego)[:,:-1,:3]
        future_ego_agent_features = cast(GenericAgents, features["generic_expert"])
        expert_features, expert_avails = self.extract_agent_features(future_ego_agent_features, batch_size)

        # Extract features across batch
        agent_features, agent_avails = self.extract_agent_features(ego_agent_features, batch_size)
        map_features, map_avails = self.extract_map_features(vector_set_map_data, batch_size)
        features = torch.cat([agent_features, map_features], dim=1)
        avails = torch.cat([agent_avails, map_avails], dim=1)

        # embed inputs
        feature_embedding = self.feature_embedding(features)

        # calculate positional embedding, then transform [num_points, 1, feature_dim] -> [1, 1, num_points, feature_dim]
        pos_embedding = self.positional_embedding(features).unsqueeze(0).transpose(1, 2)

        # invalid mask
        invalid_mask = ~avails
        invalid_polys = invalid_mask.all(-1)

        # local subgraph
        embeddings = self.local_subgraph(feature_embedding, invalid_mask, pos_embedding)
        if hasattr(self, "global_from_local"):
            embeddings = self.global_from_local(embeddings)
        embeddings = F.normalize(embeddings, dim=-1) * (self._model_params.global_embedding_size**0.5)
        embeddings = embeddings.transpose(0, 1)

        type_embedding = self.type_embedding(
            batch_size,
            self._feature_params.max_agents,
            self._feature_params.agent_features,
            self._feature_params.map_features,
            self._feature_params.max_elements,
            device=features.device,
        ).transpose(0, 1)

        # disable certain elements on demand
        if self._feature_params.disable_agents:
            invalid_polys[
                :, 1 : (1 + self._feature_params.max_agents * len(self._feature_params.agent_features))
            ] = 1  # agents won't create attention

        if self._feature_params.disable_map:
            invalid_polys[
                :, (1 + self._feature_params.max_agents * len(self._feature_params.agent_features)) :
            ] = 1  # map features won't create attention

        invalid_polys[:, 0] = 0  # make ego always available in global graph

        # global attention layers (transformer)
        outputs, attns = self.global_head(embeddings, type_embedding, invalid_polys)
        
        # self.plot_attention_weights(attn_weights=attns)

        return {
            "trajectory": Trajectory(data=convert_predictions_to_trajectory(outputs)),
            "target": Trajectory(data=convert_predictions_to_trajectory(target)),
        }
        
    def forward_closed_loop(self, features: FeaturesType) -> TargetsType:
        """
        Predict
        :param features: input features containing
                        {
                            "vector_set_map": VectorSetMap,
                            "generic_agents": GenericAgents,
                            "generic_expert": GenericAgents, (future)
                        }
        :return: targets: predictions from network
                        {
                            "ts_traj": Trajectory,
                            "trajectory": Trajectory,
                            "target": Trajectory,
                        }
        """
        # Recover features
        vector_set_map_data = cast(VectorSetMap, features["vector_set_map"])
        ego_agent_features = cast(GenericAgents, features["generic_agents"])
        batch_size = ego_agent_features.batch_size
        # batchify expert's future trajectory
        target = torch.stack(features["generic_expert"].ego)[:,:-1,:3]
        future_ego_agent_features = cast(GenericAgents, features["generic_expert"])

        # Extract features across batch
        agents_future_polys, agents_future_avail = self.extract_future_agent_features(future_ego_agent_features, batch_size) # [8, 31, 16, 3] & [8, 31, 16]
        agents_past_polys, agents_past_avail = self.extract_agent_features(ego_agent_features, batch_size) # [8, 31, 5, 3] & [8, 31, 5]
        map_features, map_avails = self.extract_map_features(vector_set_map_data, batch_size) # [8, 160, 5, 3] & [8, 160, 5]
        
        # ==== get additional info from the batch, or fall back to sensible defaults
        # future_num_frames = agents_future_polys.shape[2] #TODO: add new param, since agents_future_polys.shape[2] is already padded to 20
        max_num_vectors = map_features.shape[2]            #TODO: map_features.shape[1] is already padded to 160
        # max_num_vectors = self._feature_params.total_max_points # 21
        
        # adapt future
        # agents_past_polys_sum_past_t = agents_past_polys[:, :, -1, :] - agents_past_polys[:, :, -1, :].unsqueeze(dim=2)   # [batch_size, num_agents]
        # agents_future_polys += agents_past_polys_sum_past_t

        # Combine past and future agent information.
        # Future information is ordered [T+1, T+2, ...], past information [T, T-1, T-2, ...].
        # We thus flip past vectors and by concatenating get [..., T-2, T-1, T, T+1, T+2, ...].
        # Now, at each step T the current time window of interest simply is represented by the indices
        # T + agents_past_polys.shape[2] - window_size + 1: T + agents_past_polys.shape[2] + 1.
        # During the training loop, we will fetch this information, as well as map features,
        # which is all represented in the space of T = 0.
        # We then transform this into the space of T and feed this to the model.
        # Eventually, we shift our time window one step into the future.
        # See below for more information about used coordinate spaces.
        agents_polys = torch.cat([torch.flip(agents_past_polys, [2]), agents_future_polys], dim=2)
        agents_avail = torch.cat([torch.flip(agents_past_avail.contiguous(), [2]), agents_future_avail], dim=2)
        
        ego_full = agents_polys[:, 0, :, :]
        
        # features_now = torch.cat([agents_past_polys, map_features], dim=1)
        # avails = torch.cat([agents_past_avail, map_avails], dim=1)
        
        window_size = self._feature_params.past_time_steps + 1
        current_timestep = self._feature_params.past_time_steps

        outputs_ts = []  # buffer for predictions in local spaces
        gts_ts = []  # buffer for gts in local spaces
        outputs_t0 = []  # buffer for final prediction in t0 space (for eval only)
        attns = []

        batch_size = agents_polys.shape[0]

        type_embedding = self.type_embedding(
            batch_size,
            self._feature_params.max_agents,
            self._feature_params.agent_features,
            self._feature_params.map_features,
            self._feature_params.max_elements,
            device=agents_polys.device,
        ).transpose(0, 1)

        # one = torch.ones_like(data_batch["target_yaws"][:, 0])
        # zero = torch.zeros_like(data_batch["target_yaws"][:, 0])
        one = torch.ones_like(agents_future_polys[:,0,0,2]).unsqueeze(dim=1)
        zero = torch.zeros_like(agents_future_polys[:,0,0,2]).unsqueeze(dim=1)

        # ====== Transformation between local spaces
        # NOTE: we use the standard convention A_from_B to indicate that a matrix/yaw/translation
        # converts a point from the B space into the A space
        # e.g. if pB = (1,0) and A_from_B = (-1, 1) then pA = (0, 1)
        # NOTE: we use the following convention for names:
        # t0 -> space at 0, i.e. the space we pull out of the data for which ego is in (0, 0) with no yaw
        # ts -> generic space at step t = s > 0 (predictions at t=s are in this space)
        # tsplus -> space at s+1 (proposal new ts, built from prediction at t=s)
        # A_from_B -> indicate a full 2x3 RT matrix from B to A
        # yaw_A_from_B -> indicate a yaw from B to A
        # tr_A_from_B -> indicate a translation (XY) from B to A
        # NOTE: matrices (and yaw) we need to keep updated while we loop:
        # t0_from_ts -> bring a point from the current space into the data one (e.g. for visualisation)
        # ts_from_t0 -> bring a point from data space into the current one (e.g. to compute loss
        t0_from_ts = torch.eye(3, device=one.device).unsqueeze(0).repeat(batch_size, 1, 1) # (batch_size, 3, 3)
        ts_from_t0 = t0_from_ts.clone() # (batch_size, 3, 3)
        yaw_t0_from_ts = zero
        yaw_ts_from_t0 = zero

        for idx in range(self._feature_params.future_time_steps):
            # === STEP FORWARD ====
            # pick the right point in time [T, T-1, T-2, T-3, T-4] for window_size = 5
            agents_polys_step = torch.flip(agents_polys[:, :, current_timestep - window_size + 1: current_timestep + 1], [2]).clone() # [16, 31, 5, 3]
            agent_avails_step = torch.flip(agents_avail[:, :, current_timestep - window_size + 1: current_timestep + 1].contiguous(), [2]).clone() # [16, 31, 5, 3]
            # PAD
            agents_polys_step = pad_polylines_batch(agents_polys_step, max_num_vectors, dim=2)
            agent_avails_step = pad_avails_batch(agent_avails_step, max_num_vectors, dim=2)

            # crop agents history accordingly
            # NOTE: before padding, agent_polys_step has a number of elements equal to:
            # maxagents_polys_step& agents
            agents_polys_step[:, 0, self._history_num_frames_ego + 1:] = 0 # [batch_size, max_num_vectors-_history_num_frames_ego-1, 3]
            agent_avails_step[:, 0, self._history_num_frames_ego + 1:] = 0 # [batch_size, max_num_vectors-ego_past_frames-1]
            # agents
            agents_polys_step[:, 1:, self._history_num_frames_agents + 1:] = 0 # [batch_size, max_num_vectors-agents_past_frames-1, 3] = [16, 31, 20, 3]
            agent_avails_step[:, 1:, self._history_num_frames_agents + 1:] = 0 # [batch_size, max_num_vectors-agents_past_frames-1] = [16, 31, 20]

            # transform agents and maps into right coordinate system (ts)
            agent_features_step = transform_points(agents_polys_step, ts_from_t0, agent_avails_step, yaw_ts_from_t0) # [16, 31, 20, 3]
            map_avails_step = map_avails.clone() # [16, 160, 20]
            map_features_step = transform_points(map_features.clone(), ts_from_t0, map_avails_step) # [16, 160, 20, 3]



            ######## self.model_call ########
            # get predictions and attention of the model

            features_step = torch.cat([agent_features_step, map_features_step], dim=1) # [16, 160+31, 20, 3]
            avails_step = torch.cat([agent_avails_step, map_avails_step], dim=1) # [16, 160+31, 20]

            # embed inputs
            feature_embedding = self.feature_embedding(features_step) # [batch_size, num_elements, max_num_points, embed_dim]

            # calculate positional embedding, then transform [num_points, 1, feature_dim] -> [1, 1, num_points, feature_dim]
            pos_embedding = self.positional_embedding(features_step).unsqueeze(0).transpose(1, 2) # [1, 1, max_num_points, embed_dim]

            # invalid mask
            invalid_mask = ~avails_step
            invalid_polys = invalid_mask.all(-1)

            # local subgraph
            embeddings = self.local_subgraph(feature_embedding, invalid_mask, pos_embedding) # [batch_size, num_elements, max_num_points]
            if hasattr(self, "global_from_local"):
                embeddings = self.global_from_local(embeddings)
            embeddings = F.normalize(embeddings, dim=-1) * (self._model_params.global_embedding_size**0.5)
            embeddings = embeddings.transpose(0, 1)

            # disable certain elements on demand
            if self._feature_params.disable_agents:
                invalid_polys[
                    :, 1 : (1 + self._feature_params.max_agents * len(self._feature_params.agent_features))
                ] = 1  # agents won't create attention

            if self._feature_params.disable_map:
                invalid_polys[
                    :, (1 + self._feature_params.max_agents * len(self._feature_params.agent_features)) :
                ] = 1  # map features won't create attention

            invalid_polys[:, 0] = 0  # make ego always available in global graph

            # call and return global graph
            out, attn = self.global_head(embeddings, type_embedding, invalid_polys)
            ######## self.model_call ########
            
            

            # outputs are in ts space (optionally xy normalised)
            pred_xy_step = out[:, 0, :2]
            pred_yaw_step = out[:, 0, 2:3] if not self._model_params.limit_predicted_yaw else 0.3 * torch.tanh(out[:, 0, 2:3])

            pred_xy_step_unnorm = pred_xy_step

            # ==== SAVE PREDICTIONS & GT
            # (16 x 1 x 2) = (16 x 1 x 2) @ (16 x 2 x 2) + (16 x 1 x 2)
            gt_xy_step_ts = agents_future_polys[:, 0, idx: idx + 1, :2] @ ts_from_t0[..., :2, :2].transpose(
                1, 2
            ) + ts_from_t0[..., :2, -1:].transpose(1, 2)
            gt_xy_step_ts = gt_xy_step_ts[:, 0] # (16 x 2)
            gt_yaw_ts = (agents_future_polys[:, 0, idx, 2:3] + yaw_ts_from_t0) # (16 x 1) = (16 x 1) + (16 x 1)

            pred_xy_step_t0 = pred_xy_step_unnorm[:, None, :] @ t0_from_ts[..., :2, :2].transpose(1, 2) + t0_from_ts[
                ..., :2, -1:
            ].transpose(1, 2)
            pred_xy_step_t0 = pred_xy_step_t0[:, 0]
            pred_yaw_step_t0 = pred_yaw_step + yaw_t0_from_ts

            outputs_ts.append(torch.cat([pred_xy_step, pred_yaw_step], dim=-1))
            outputs_t0.append(torch.cat([pred_xy_step_t0, pred_yaw_step_t0], dim=-1))
            gts_ts.append(torch.cat([gt_xy_step_ts, gt_yaw_ts], dim=-1))
            if attn is not None:
                attns.append(attn)

            # clone as we might change in place
            pred_xy_step_unnorm = pred_xy_step_unnorm.clone()
            pred_yaw_step = pred_yaw_step.clone()

            # ==== UPDATE HISTORY WITH INFORMATION FROM PREDICTION

            # update transformation matrices
            t0_from_ts, ts_from_t0, yaw_t0_from_ts, yaw_ts_from_t0 = update_transformation_matrices(
                pred_xy_step_unnorm, pred_yaw_step, t0_from_ts, ts_from_t0, yaw_t0_from_ts, yaw_ts_from_t0, zero, one
            )

            # update AoI
            agents_polys[:, 0, current_timestep + 1, :2] = pred_xy_step_t0
            agents_polys[:, 0, current_timestep + 1, 2:3] = pred_yaw_step_t0
            agents_avail[:, 0, current_timestep + 1] = 1

            # move time window one step into the future
            current_timestep += 1

            # detach if requested, or if in initial sampling phase
            if self._model_params.detach_unroll or idx < self._model_params.warmup_num_frames:
                t0_from_ts.detach_()
                ts_from_t0.detach_()
                yaw_t0_from_ts.detach_()
                yaw_ts_from_t0.detach_()
                agents_polys.detach_()
                map_features.detach_()
                agents_avail.detach_()
                map_avails.detach_()

        # recombine predictions
        outputs_ts = torch.stack(outputs_ts, dim=1)
        outputs_t0 = torch.stack(outputs_t0, dim=1)
        targets = torch.stack(gts_ts, dim=1)
        attns = torch.cat(attns, dim=1)
        
        ts_pred = convert_predictions_to_trajectory(outputs_ts)
        t0_pred = convert_predictions_to_trajectory(outputs_t0)
        goal = convert_predictions_to_trajectory(targets)
        
        
        # goal = convert_predictions_to_trajectory(torch.stack(future_ego_agent_features.ego)[:,:-1,:3])
        # goal = convert_predictions_to_trajectory(ego_full)
        # goal = convert_predictions_to_trajectory(agents_future_polys[:, 0, 2:, :])
        # goal = convert_predictions_to_trajectory(agents_past_polys[:, 1, :, :])
        # goal = convert_predictions_to_trajectory(torch.cat([torch.flip(agents_past_polys, [2]), agents_future_polys], dim=2)[:,0])
        
        return {
            "ts_traj": Trajectory(data=ts_pred),
            "trajectory": Trajectory(data=t0_pred),
            "target": Trajectory(data=goal),
        }


    def forward(self, features: FeaturesType) -> TargetsType:
        """
        Predict
        :param features: input features containing
                        {
                            "vector_set_map": VectorSetMap,
                            "generic_agents": GenericAgents,
                            "generic_expert": GenericAgents, (future)
                        }
        :return: targets: predictions from network
                        {
                            "trajectory": Trajectory,
                        }
        """
        # Recover featuresTensorTarget
        vector_set_map_data = cast(VectorSetMap, features["vector_set_map"])
        ego_agent_features = cast(GenericAgents, features["generic_agents"])
        batch_size = ego_agent_features.batch_size  # 8
        # batchify expert's future trajectory
        # target = torch.stack(features["generic_expert"].ego)[:,:-1,:3]
        future_ego_agent_features = cast(GenericAgents, features["generic_expert"])

        # Extract features across batch
        agents_future_polys, agents_future_avail = self.extract_future_agent_features(future_ego_agent_features, batch_size) # [8, 51, 16, 3] & [8, 51, 16]
        agents_past_polys, agents_past_avail = self.extract_agent_features(ego_agent_features, batch_size) # [8, 51, 16, 3] & [8, 51, 16]
        map_features, map_avails = self.extract_map_features(vector_set_map_data, batch_size) # [8, 160, 16, 3] & [8, 160, 16]
        
        features = torch.cat([agents_past_polys, map_features], dim=1) # [8, 211, 16, 3]
        avails = torch.cat([agents_past_avail, map_avails], dim=1) # [8, 211, 16]

        # embed inputs
        feature_embedding = self.feature_embedding(features) # [8, 211, 16, 128]

        # calculate positional embedding, then transform [num_points, 1, feature_dim] -> [1, 1, num_points, feature_dim]
        pos_embedding = self.positional_embedding(features).unsqueeze(0).transpose(1, 2) # [1, 1, 16, 128]

        # invalid mask
        invalid_mask = ~avails # [8, 211, 16]
        invalid_polys = invalid_mask.all(-1) # [8, 211]

        # local subgraph
        embeddings = self.local_subgraph(feature_embedding, invalid_mask, pos_embedding) # [8, 211, 128]
        if hasattr(self, "global_from_local"):
            embeddings = self.global_from_local(embeddings)
        embeddings = F.normalize(embeddings, dim=-1) * (self._model_params.global_embedding_size**0.5) # [8, 211, 128]
        # embeddings = embeddings.transpose(0, 1)

        type_embedding = self.type_embedding(
            batch_size,
            self._feature_params.max_agents,
            self._feature_params.agent_features,
            self._feature_params.map_features,
            self._feature_params.max_elements,
            device=features.device,
        )#.transpose(0, 1) # [8, 211, 256]

        # disable certain elements on demand
        if self._feature_params.disable_agents:
            invalid_polys[
                :, 1 : (1 + self._feature_params.max_agents * len(self._feature_params.agent_features))
            ] = 1  # agents won't create attention

        if self._feature_params.disable_map:
            invalid_polys[
                :, (1 + self._feature_params.max_agents * len(self._feature_params.agent_features)) :
            ] = 1  # map features won't create attention

        invalid_polys[:, 0] = 0  # make ego always available in global graph

        # global attention layers (transformer)
        pred_agents, pred_traj_logits = self.global_head(embeddings, type_embedding, invalid_polys)
        # pred_agents: [8, 50, 6, 16, 3], pred_traj_logits: [8, 50, 6]
        # pred_agents: [batch_size, max_agents, agent_num_trajectories, future_time_steps, feature_dimension]

        # used to make target relative to agents / output relative to ego
        agents_t0_xy = agents_past_polys[:, 0:, 0, :2]  # all_other_agents_history_positions # [8, 50, 2]
        agents_t0_yaw = agents_past_polys[:, 0:, 0, 2:3].squeeze(-1)  # all_other_agents_history_yaws # [8, 50]
        
        
        ####### LOSS #######
        
        # ego
        # loss_ego = 0.  # we don't have a planning ego loss here

        # agents
        # gt is in ego-relative coords
        # [batch_size, num_agents, agent_num_timesteps, num_outputs]
        agents_xy = agents_future_polys[:, 0:, :, :2]  # all_other_agents_future_positions # [8, 50, 16, 2]
        agents_yaw = agents_future_polys[:, 0:, :, 2:3] # all_other_agents_future_yaws # [8, 50, 16, 1]

        # make targets relative to agents
        # get the transformation matrix
        agents_t0_xy_exp = agents_t0_xy.reshape(-1, 2)  # [400, 2]
        agents_t0_yaw_exp = agents_t0_yaw.reshape(-1)   # [400]
        _, inverse_matrix = build_matrix(translation=agents_t0_xy_exp, angle=agents_t0_yaw_exp) # [8, 50, 3, 3]
        # get the correct matrix shape
        inverse_matrix = inverse_matrix.view(list(agents_t0_xy.shape[:2]) + [3, 3]) # [8, 50, 3, 3]
        inverse_matrix = inverse_matrix.unsqueeze(2).expand(list(agents_t0_xy.shape[:-1]) + [self._feature_params.future_time_steps, 3, 3]).reshape(-1, 3, 3)  # [6400, 3, 3]
        # transform the points
        agents_points = torch.cat((agents_xy, agents_yaw), -1)  # [8, 50, 16, 3]
        agents_points_flattened = agents_points.reshape(-1, 1, 1, 3)    # [6400, 1, 1, 3]
        transformed_points = transform_points(agents_points_flattened, inverse_matrix,
                                                torch.ones_like(agents_points_flattened[..., 0], dtype=torch.bool),
                                                agents_yaw.reshape(-1, 1, 1, 1))  # [6400, 1, 1, 3]
        agents_xy = transformed_points[..., :2].reshape(agents_xy.shape)    # [8, 50, 16, 2]
        agents_yaw = transformed_points[..., 2:].reshape(agents_yaw.shape)  # [8, 50, 16, 1]

        target = torch.cat((agents_xy, agents_yaw), dim=-1)    # [8, 50, 16, 3]
        target[..., :2] /= 125.
        target[..., 2] /= np.math.pi  # no need for complex angle normalization, we predict offsets

        # [batch_size, num_agents, num_timesteps]
        target_avails = agents_future_avail[:, 0:, :]  # all_other_agents_future_availability # [8, 50, 16]

        # [batch_size, num_agents, agent_num_trajectories, num_timesteps, 3]
        target = target.unsqueeze(dim=2).expand(-1, -1, self.global_head.agent_num_trajectories, -1, -1)  # [8, 50, 6, 16, 3]
        # [batch_size, num_agents, agent_num_trajectories, num_timesteps]
        target_avails = target_avails.unsqueeze(dim=2).expand(-1, -1, self.global_head.agent_num_trajectories, -1)  # [8, 50, 6, 16]
        
        # # compute loss on trajectories
        # agent_loss = self.criterion(pred_agents, target)   # [16, 50, 20, 30, 3]
        # agent_loss *= target_avails.unsqueeze(-1)   # [16, 50, 20, 30, 3]
        # pred_num_valid_targets = target_avails.sum().float()    # [1]

        # pred_num_valid_targets /= self.global_head.agent_num_trajectories
        # any_target_avails = torch.any(target_avails, dim=-1) # [16, 50, 20]

        # # [batch_size, num_agents, agent_num_trajectories]
        # loss_pred_per_trajectory = agent_loss.sum(-1).sum(-1) * any_target_avails   # [16, 50, 20]

        # # [batch_size, num_agents]
        # pred_loss_argmin_idx = self.agent_traj_matcher(loss_pred_per_trajectory / (target_avails.sum(-1) + self.eps), pred_traj_logits) # [16, 50]

        # # compute loss on probability distribution for valid targets only
        # pred_prob_loss = F.cross_entropy(pred_traj_logits[any_target_avails[..., 0]], pred_loss_argmin_idx[any_target_avails[..., 0]])  # [1]

        # # compute the final loss only on the trajectory with the lowest loss
        # # [batch_size, num_agents, num_traj]
        # pred_traj_loss_batch = agent_loss.sum(-1).sum(-1)  # [16, 50, 20]
        # # [batch_size, num_agents]
        # # NOTE torch.gather can be non-deterministic -- from pytorch 1.9.0 torch.take_along_dim can be used instead
        # pred_traj_loss_batch = torch.gather(pred_traj_loss_batch, dim=-1, index=pred_loss_argmin_idx[:, :, None]).squeeze(-1)  # [16, 50]
        # # zero out invalid targets (agents)
        # pred_traj_loss_batch = pred_traj_loss_batch * any_target_avails[..., 0]  # [16, 50]
        # # [1]
        # pred_traj_loss = pred_traj_loss_batch.sum() / (pred_num_valid_targets + self.eps) # [1]
        # # compute overall loss
        # pred_loss = pred_traj_loss + pred_prob_loss * self.agent_traj_matcher.cost_prob_coeff # [1]


        # ######### VISUALIZE #########
        
        # ego
        # ego is ignored in these experiments (we don't include planning).
        # ego future is predicted considering ego as a road agent (ego is the first agent)

        # agents
        # [batch_size, num_agents, num_trajectories, num_timesteps, 3]
        pred_agents[..., :2] *= 125.
        pred_agents[..., 2] *= np.math.pi  # no need for complex angle normalization, we predict offsets

        # getting the agents availabilities at the current timestep
        # [batch_size, num_agents]
        pred_agents_avails = agents_past_avail[:, 0:, 0] # all_other_agents_history_availability # [16, 50]
        pred_agents[~pred_agents_avails] = 0.

        # make predictions relative to ego
        # get the transformation matrix
        agents_t0_xy_reshaped = agents_t0_xy.reshape(-1, 2)
        agents_t0_yaw_reshaped = agents_t0_yaw.reshape(-1)
        matrix, _ = build_matrix(translation=agents_t0_xy_reshaped, angle=agents_t0_yaw_reshaped)
        # get the correct matrix shape
        matrix = matrix.view(list(agents_t0_xy.shape[:2]) + [3, 3])
        matrix = matrix.unsqueeze(2).unsqueeze(2).expand(list(pred_agents.shape[:-1]) + [3, 3]).reshape(-1, 3, 3)
        # transform the points
        pred_agents_flattened = pred_agents.reshape(-1, 1, 1, 3)
        transformed_points = transform_points(pred_agents_flattened, matrix,
                                                torch.ones_like(pred_agents_flattened[..., 0], dtype=torch.bool),
                                                pred_agents_flattened[..., 2:])
        # [batch_size, num_agents, num_trajectories, num_timesteps, 3]
        transformed_points = transformed_points.reshape(pred_agents.shape)
        all_pred_agents_xy = transformed_points[..., :2]
        all_pred_agents_yaw = transformed_points[..., 2:]

        # get the highest probability trajectory
        # [batch_size, num_agents]
        if self.global_head.agent_multimodal_predictions:
            # just pick the trajectory with highest probability
            pred_traj_index = pred_traj_logits.argmax(dim=-1)
        else:
            # pick the only predicted trajectory
            pred_traj_index = torch.zeros(pred_agents.shape[:2], dtype=torch.long, device=pred_agents.device)
        # [batch_size, num_agents, 1, 1, 1]
        pred_traj_index_expanded = pred_traj_index[:, :, None, None, None]
        # [batch_size, num_agents, num_future_frames, 3]
        # NOTE torch.gather can be non-deterministic -- from pytorch 1.9.0 torch.take_along_dim can be used instead
        pred_agents_xy = torch.gather(
            all_pred_agents_xy, dim=2,
            index=pred_traj_index_expanded.expand(([-1, -1, -1] + list(all_pred_agents_xy.shape[-2:])))).squeeze(2)
        pred_agents_yaw = torch.gather(
            all_pred_agents_yaw, dim=2,
            index=pred_traj_index_expanded.expand(([-1, -1, -1] + list(all_pred_agents_yaw.shape[-2:])))).squeeze(2)

        # self.plot_attention_weights(attn_weights=attns)
        
        sorted_pred_trajs = self.sort_predictions(pred_agents, pred_traj_logits)
        # predicted_trajs = list(pred_agents.chunk(pred_agents.size(2), dim=2))
        # ego_trajs = [Trajectory(data=convert_predictions_to_trajectory(pred[:, 0])) for pred in predicted_trajs] # 6 ego trajs

        all_pred_agents = torch.cat([all_pred_agents_xy, all_pred_agents_yaw], dim=-1)
        all_pred_agents = self.sort_predictions(all_pred_agents, pred_traj_logits)
        
        return {
            "trajectories": TensorTarget(sorted_pred_trajs),    # [8, 50, 6, 16, 3] # ego_trajs = [6][8, 16, 3]
            "pred_agents": TensorTarget(pred_agents),           # [8, 50, 6, 16, 3]
            "pred_traj_logits": TensorTarget(pred_traj_logits), # [8, 50, 6]
            "target": TensorTarget(target),                     # [8, 50, 6, 16, 3]
            "target_avails": TensorTarget(target_avails),       # [8, 50, 6, 16]
            # "loss": TensorTarget(pred_loss),
            # "loss/agents_traj": TensorTarget(pred_traj_loss),
            # "loss/agents_traj_prob": TensorTarget(pred_prob_loss),
            "all_pred_agents": TensorTarget(all_pred_agents),   # [8, 50, 6, 16, 3]
        }

        # # calculate loss or return predicted position for inference
        # if self.training:
        #     if self.criterion is None:
        #         raise NotImplementedError("Loss function is undefined.")

        #     # ego
        #     # loss_ego = 0.  # we don't have a planning ego loss here

        #     # agents
        #     # gt is in ego-relative coords
        #     # [batch_size, num_agents, agent_num_timesteps, num_outputs]
        #     agents_xy = agents_future_polys[:, 1:, :, :2]  # all_other_agents_future_positions # [16, 50, 30, 2]
        #     agents_yaw = agents_future_polys[:, 1:, :, 2:3] # all_other_agents_future_yaws # [16, 50, 30, 1]

        #     # make targets relative to agents
        #     # get the transformation matrix
        #     agents_t0_xy_exp = agents_t0_xy.reshape(-1, 2)  # [800, 2]
        #     agents_t0_yaw_exp = agents_t0_yaw.reshape(-1)   # [800]
        #     _, inverse_matrix = build_matrix(translation=agents_t0_xy_exp, angle=agents_t0_yaw_exp) # [16, 50, 3, 3]
        #     # get the correct matrix shape
        #     inverse_matrix = inverse_matrix.view(list(agents_t0_xy.shape[:2]) + [3, 3]) # [16, 50, 3, 3]
        #     inverse_matrix = inverse_matrix.unsqueeze(2).expand(list(agents_t0_xy.shape[:-1]) + [self._feature_params.future_time_steps, 3, 3]).reshape(-1, 3, 3)  # [24000, 3, 3]
        #     # transform the points
        #     agents_points = torch.cat((agents_xy, agents_yaw), -1)  # [16, 50, 30, 3]
        #     agents_points_flattened = agents_points.reshape(-1, 1, 1, 3)    # [24000, 1, 1, 3]
        #     transformed_points = transform_points(agents_points_flattened, inverse_matrix,
        #                                           torch.ones_like(agents_points_flattened[..., 0], dtype=torch.bool),
        #                                           agents_yaw.reshape(-1, 1, 1, 1))  # [24000, 1, 1, 3]
        #     agents_xy = transformed_points[..., :2].reshape(agents_xy.shape)    # [16, 50, 30, 2]
        #     agents_yaw = transformed_points[..., 2:].reshape(agents_yaw.shape)  # [16, 50, 30, 1]

        #     targets = torch.cat((agents_xy, agents_yaw), dim=-1)    # [16, 50, 30, 3]
        #     targets[..., :2] /= 125.
        #     targets[..., 2] /= np.math.pi  # no need for complex angle normalization, we predict offsets

        #     # [batch_size, num_agents, num_timesteps]
        #     target_avails = agents_future_avail[:, 1:, :]  # all_other_agents_future_availability # [16, 50, 30]

        #     # [batch_size, num_agents, agent_num_trajectories, num_timesteps, 3]
        #     targets = targets.unsqueeze(dim=2).expand(-1, -1, self.global_head.agent_num_trajectories, -1, -1)  # [16, 50, 20, 30, 3]
        #     # [batch_size, num_agents, agent_num_trajectories, num_timesteps]
        #     target_avails = target_avails.unsqueeze(dim=2).expand(-1, -1, self.global_head.agent_num_trajectories, -1)  # [16, 50, 20, 30]

        #     # compute loss on trajectories
        #     agent_loss = self.criterion(pred_agents, targets)   # [16, 50, 20, 30, 3]
        #     agent_loss *= target_avails.unsqueeze(-1)   # [16, 50, 20, 30, 3]
        #     pred_num_valid_targets = target_avails.sum().float()    # [1]

        #     pred_num_valid_targets /= self.global_head.agent_num_trajectories
        #     any_target_avails = torch.any(target_avails, dim=-1) # [16, 50, 20]

        #     # [batch_size, num_agents, agent_num_trajectories]
        #     loss_pred_per_trajectory = agent_loss.sum(-1).sum(-1) * any_target_avails   # [16, 50, 20]

        #     # [batch_size, num_agents]
        #     pred_loss_argmin_idx = self.agent_traj_matcher(loss_pred_per_trajectory / (target_avails.sum(-1) + self.eps), pred_traj_logits) # [16, 50]

        #     # compute loss on probability distribution for valid targets only
        #     pred_prob_loss = F.cross_entropy(pred_traj_logits[any_target_avails[..., 0]], pred_loss_argmin_idx[any_target_avails[..., 0]])  # [1]

        #     # compute the final loss only on the trajectory with the lowest loss
        #     # [batch_size, num_agents, num_traj]
        #     pred_traj_loss_batch = agent_loss.sum(-1).sum(-1)  # [16, 50, 20]
        #     # [batch_size, num_agents]
        #     # NOTE torch.gather can be non-deterministic -- from pytorch 1.9.0 torch.take_along_dim can be used instead
        #     pred_traj_loss_batch = torch.gather(pred_traj_loss_batch, dim=-1, index=pred_loss_argmin_idx[:, :, None]).squeeze(-1)  # [16, 50]
        #     # zero out invalid targets (agents)
        #     pred_traj_loss_batch = pred_traj_loss_batch * any_target_avails[..., 0]  # [16, 50]
        #     # [1]
        #     pred_traj_loss = pred_traj_loss_batch.sum() / (pred_num_valid_targets + self.eps) # [1]
        #     # compute overall loss
        #     pred_loss = pred_traj_loss + pred_prob_loss * self.agent_traj_matcher.cost_prob_coeff # [1]

        #     train_dict = {
        #         "loss": pred_loss,
        #         "loss/agents_traj": pred_traj_loss,
        #         "loss/agents_traj_prob": pred_prob_loss,
        #     }
        #     return train_dict
        # else:
        #     # ego
        #     # ego is ignored in these experiments (we don't include planning).
        #     # ego future is predicted considering ego as a road agent (ego is the first agent)

        #     # agents
        #     # [batch_size, num_agents, num_trajectories, num_timesteps, 3]
        #     pred_agents[..., :2] *= 125.
        #     pred_agents[..., 2] *= np.math.pi  # no need for complex angle normalization, we predict offsets

        #     # getting the agents availabilities at the current timestep
        #     # [batch_size, num_agents]
        #     pred_agents_avails = agents_past_avail[:, 1:, 0] # all_other_agents_history_availability # [16, 50]
        #     pred_agents[~pred_agents_avails] = 0.

        #     # make predictions relative to ego
        #     # get the transformation matrix
        #     agents_t0_xy_reshaped = agents_t0_xy.reshape(-1, 2)
        #     agents_t0_yaw_reshaped = agents_t0_yaw.reshape(-1)
        #     matrix, _ = build_matrix(translation=agents_t0_xy_reshaped, angle=agents_t0_yaw_reshaped)
        #     # get the correct matrix shape
        #     matrix = matrix.view(list(agents_t0_xy.shape[:2]) + [3, 3])
        #     matrix = matrix.unsqueeze(2).unsqueeze(2).expand(list(pred_agents.shape[:-1]) + [3, 3]).reshape(-1, 3, 3)
        #     # transform the points
        #     pred_agents_flattened = pred_agents.reshape(-1, 1, 1, 3)
        #     transformed_points = transform_points(pred_agents_flattened, matrix,
        #                                           torch.ones_like(pred_agents_flattened[..., 0], dtype=torch.bool),
        #                                           pred_agents_flattened[..., 2:])
        #     # [batch_size, num_agents, num_trajectories, num_timesteps, 3]
        #     transformed_points = transformed_points.reshape(pred_agents.shape)
        #     all_pred_agents_xy = transformed_points[..., :2]
        #     all_pred_agents_yaw = transformed_points[..., 2:]

        #     # get the highest probability trajectory
        #     # [batch_size, num_agents]
        #     if self.global_head.agent_multimodal_predictions:
        #         # just pick the trajectory with highest probability
        #         pred_traj_index = pred_traj_logits.argmax(dim=-1)
        #     else:
        #         # pick the only predicted trajectory
        #         pred_traj_index = torch.zeros(pred_agents.shape[:2], dtype=torch.long, device=pred_agents.device)
        #     # [batch_size, num_agents, 1, 1, 1]
        #     pred_traj_index_expanded = pred_traj_index[:, :, None, None, None]
        #     # [batch_size, num_agents, num_future_frames, 3]
        #     # NOTE torch.gather can be non-deterministic -- from pytorch 1.9.0 torch.take_along_dim can be used instead
        #     pred_agents_xy = torch.gather(
        #         all_pred_agents_xy, dim=2,
        #         index=pred_traj_index_expanded.expand(([-1, -1, -1] + list(all_pred_agents_xy.shape[-2:])))).squeeze(2)
        #     pred_agents_yaw = torch.gather(
        #         all_pred_agents_yaw, dim=2,
        #         index=pred_traj_index_expanded.expand(([-1, -1, -1] + list(all_pred_agents_yaw.shape[-2:])))).squeeze(2)

        #     eval_dict = {
        #         "agent_positions": pred_agents_xy,
        #         "agent_yaws": pred_agents_yaw,
        #         "all_agent_positions": all_pred_agents_xy,
        #         "all_agent_yaws": all_pred_agents_yaw,
        #         "agent_traj_logits": pred_traj_logits,
        #     }

        #     return eval_dict
        
    
    def plot_attention_weights(self, attn_weights: torch.Tensor):
        """
        Plots the attention weights as a heatmap.

        Args:
            attn_weights (torch.Tensor): Tensor of shape [batch_size, target_sequence_length, source_sequence_length].
        """
        batch_size, target_sequence_length, source_sequence_length = attn_weights.size()

        # Plot the attention weights as a heatmap.
        fig, ax = plt.subplots(figsize=(90,6))
        sns.heatmap(attn_weights.squeeze(dim=1).cpu().detach().numpy(), cmap="YlGnBu", ax=ax)
        ax.set_xlabel("Source Sequence")
        ax.set_ylabel("Target Sequence")
        ax.set_title("Attention Weights for each batch")
        plt.savefig('attention_weights.png')
        # plt.show()
        
    def sort_predictions(self, all_pred_agents: torch.Tensor, pred_traj_logits: torch.Tensor) -> torch.Tensor:
        """select the trajectory with the largest probability for each batch of data
        Args:
            pred_obs: shape [c, T, B, 5] c trajectories for the ego agents with every point being the params of Bivariate Gaussian distribution.
            mode_probs: shape [B, c] mode probability predictions P(z|X_{1:T_obs})
        """
        
        # Sort the tensor based on the sorted indices
        if self.global_head.agent_multimodal_predictions:
            # Sort the trajectory probabilities in descending order
            _, pred_traj_index = pred_traj_logits.sort(dim=-1, descending=True)
        else:
            # Pick the only predicted trajectory
            pred_traj_index = torch.zeros(all_pred_agents.shape[:2], dtype=torch.long, device=all_pred_agents.device)

        # [batch_size, num_agents, 1, 1, 1]
        pred_traj_index_expanded = pred_traj_index[:, :, :, None, None]
        # [batch_size, num_agents, num_future_frames, 3]
        # NOTE: torch.gather can be non-deterministic -- from pytorch 1.9.0 torch.take_along_dim can be used instead
        all_pred_agents_sorted = torch.gather(
            all_pred_agents, dim=2,
            index=pred_traj_index_expanded.expand(([-1, -1, -1] + list(all_pred_agents.shape[-2:])))).squeeze(2)

        
        return all_pred_agents_sorted