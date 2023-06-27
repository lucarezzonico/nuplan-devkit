import math
from pathlib import Path
import os
import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.preprocessing.feature_builders.autobots_feature_builder import AutobotsPredNominalTargetBuilder, AutobotsModeProbsNominalTargetBuilder
from nuplan.planning.training.preprocessing.target_builders.ego_trajectory_target_builder import EgoTrajectoryTargetBuilder, AgentsTrajectoryTargetBuilder

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.preprocessing.feature_builders.vector_map_feature_builder import VectorMapFeatureBuilder
from nuplan.planning.training.preprocessing.features.agents import Agents
from nuplan.planning.training.preprocessing.features.vector_map import VectorMap
from nuplan.planning.training.preprocessing.features.tensor_target import TensorTarget
from nuplan.planning.training.preprocessing.feature_builders.agents_feature_builder import AgentsFeatureBuilder
from nuplan.planning.training.preprocessing.features.autobots_feature_conversion import NuplanToAutobotsConverter
from nuplan.planning.training.modeling.types import FeaturesType, TargetsType

from typing import List, Optional, cast, Dict, Tuple

from nuplan.planning.training.modeling.models.context_encoders import MapEncoderPtsMA, MapEncoderCNN, MapEncoderPts

from nuplan.planning.training.preprocessing.feature_builders.autobots_feature_builder import ScenarioTypeFeatureBuilder, EgoGoalFeatureBuilder
from nuplan.planning.training.preprocessing.feature_builders.expert_feature_builder import ExpertFeatureBuilder

from dataclasses import dataclass
from nuplan.planning.training.preprocessing.feature_builders.generic_agents_feature_builder import (
    GenericAgentsFeatureBuilder,
)
from nuplan.planning.training.preprocessing.feature_builders.vector_set_map_feature_builder import (
    VectorSetMapFeatureBuilder,
)
from nuplan.planning.training.preprocessing.feature_builders.generic_expert_feature_builder import (
    GenericExpertFeatureBuilder,
)
from nuplan.planning.training.preprocessing.target_builders.expert_trajectory_target_builder import (
    ExpertTrajectoryTargetBuilder,
)
from nuplan.planning.training.preprocessing.features.generic_agents import GenericAgents
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory
from nuplan.planning.training.preprocessing.features.trajectories import Trajectories
from nuplan.planning.training.preprocessing.features.vector_set_map import VectorSetMap
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


from nuplan.planning.training.callbacks.utils.visualization_utils import (
    get_raster_from_vector_map_with_agents, get_raster_from_vector_map_with_agents_multiple_trajectories
)
import cv2

@dataclass
class AutoBotJointModelParams:
    """
    Parameters for UrbanDriverOpenLoop model.
        local_embedding_size: embedding dimensionality of local subgraph layers.
        global_embedding_size: embedding dimensionality of global attention layers.
        num_subgraph_layers: number of stacked PointNet-like local subgraph layers.
        global_head_dropout: float in range [0,1] for the dropout in the MHA global head. Set to 0 to disable it.
    """
    
    d_k: int = 128 # hidden_size
    _M: int = 30 # num_other_agents
    c: int = 6 # num_modes
    T: int = 16
    L_enc: int = 4
    dropout: float = 0.1
    k_attr: int = 2
    map_attr: int = 3
    num_heads: int = 16
    L_dec: int = 4
    tx_hidden_size: int = 384
    use_map_img: bool = False
    use_map_lanes: bool = True
    predict_yaw: bool = False
    
    draw_visualizations: bool = False
    current_task: str = "training"
    log_dir: str = "/data1/nuplan/luca/exp/training/autobotego_experiment/autobotego_model"
    checkpoint_dir: str = "/data1/nuplan/luca/exp/training/autobotego_experiment"
    
@dataclass
class AutoBotJointFeatureParams:
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
class AutoBotJointTargetParams:
    """
    Parameters for UrbanDriverOpenLoop targets.
        num_output_features: number of target features.
        future_trajectory_sampling: Sampling parameters for future trajectory.
    """

    # num_output_features: int
    future_trajectory_sampling: TrajectorySampling
    expert_trajectory_sampling: TrajectorySampling



def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class PositionalEncoding(nn.Module):
    '''
    Standard positional encoding.
    ''' 
    def __init__(self, d_model, dropout=0.1, max_len=20):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        :param x: must be (T, B, H)
        :return:
        '''
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class OutputModel(nn.Module):
    '''
    This class operates on the output of AutoBot-Joint's decoder representation. It produces the parameters of a
    bivariate Gaussian distribution and possibly predicts the yaw.
    '''
    def __init__(self, d_k=64, predict_yaw=False):
        super(OutputModel, self).__init__()
        self.d_k = d_k
        self.predict_yaw = predict_yaw
        out_len = 5
        if predict_yaw:
            out_len = 6

        init_ = lambda m: init(m, nn.init.xavier_normal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))
        self.observation_model = nn.Sequential(
            init_(nn.Linear(d_k, d_k)), nn.ReLU(),
            init_(nn.Linear(d_k, d_k)), nn.ReLU(),
            init_(nn.Linear(d_k, out_len))
        )
        self.min_stdev = 0.01

    def forward(self, agent_latent_state):
        T = agent_latent_state.shape[0]
        BK = agent_latent_state.shape[1]
        pred_obs = self.observation_model(agent_latent_state.reshape(-1, self.d_k)).reshape(T, BK, -1)
        
        x_mean = pred_obs[:, :, 0]
        y_mean = pred_obs[:, :, 1]
        x_sigma = F.softplus(pred_obs[:, :, 2]) + self.min_stdev
        y_sigma = F.softplus(pred_obs[:, :, 3]) + self.min_stdev
        rho = torch.tanh(pred_obs[:, :, 4]) * 0.9  # for stability
        
        if self.predict_yaw:
            yaws = pred_obs[:, :, 5]  # for stability
            return torch.stack([x_mean, y_mean, x_sigma, y_sigma, rho, yaws], dim=2)
        else:
            return torch.stack([x_mean, y_mean, x_sigma, y_sigma, rho], dim=2)


class AutoBotJoint(TorchModuleWrapper):
    '''
    AutoBot-Joint Class.
    '''
    def __init__(
        self,
        model_params: AutoBotJointModelParams,
        feature_params: AutoBotJointFeatureParams,
        target_params: AutoBotJointTargetParams,
    ):
        # super(AutoBotJoint, self).__init__()
        
        self.img_num = 0
        
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
                # GenericExpertFeatureBuilder(feature_params.agent_features, target_params.expert_trajectory_sampling),
            ],
            target_builders=[
                EgoTrajectoryTargetBuilder(target_params.future_trajectory_sampling),
                AgentsTrajectoryTargetBuilder(target_params.future_trajectory_sampling),
                # ExpertTrajectoryTargetBuilder(target_params.expert_trajectory_sampling)
                AutobotsPredNominalTargetBuilder(),
                AutobotsModeProbsNominalTargetBuilder(),
            ],
            future_trajectory_sampling=target_params.future_trajectory_sampling,
        )
        self._model_params = model_params
        self._feature_params = feature_params
        self._target_params = target_params
        
        init_ = lambda m: init(m, nn.init.xavier_normal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))

        self.converter = NuplanToAutobotsConverter(_M=self._model_params._M)
        
        # INPUT ENCODERS
        self.agents_dynamic_encoder = nn.Sequential(init_(nn.Linear(self._model_params.k_attr, self._model_params.d_k)))

        # ============================== AutoBot-Joint ENCODER ==============================
        self.social_attn_layers = []
        self.temporal_attn_layers = []
        for _ in range(self._model_params.L_enc):
            tx_encoder_layer = nn.TransformerEncoderLayer(d_model=self._model_params.d_k,
                                                          nhead=self._model_params.num_heads,
                                                          dropout=self._model_params.dropout,
                                                          dim_feedforward=self._model_params.tx_hidden_size)
            self.temporal_attn_layers.append(nn.TransformerEncoder(tx_encoder_layer, num_layers=2))

            tx_encoder_layer = nn.TransformerEncoderLayer(d_model=self._model_params.d_k,
                                                          nhead=self._model_params.num_heads,
                                                          dropout=self._model_params.dropout,
                                                          dim_feedforward=self._model_params.tx_hidden_size)
            self.social_attn_layers.append(nn.TransformerEncoder(tx_encoder_layer, num_layers=1))

        self.temporal_attn_layers = nn.ModuleList(self.temporal_attn_layers)
        self.social_attn_layers = nn.ModuleList(self.social_attn_layers)

        # ============================== MAP ENCODER ==========================
        if self._model_params.use_map_lanes:
            self.map_encoder = MapEncoderPtsMA(d_k=self._model_params.d_k,
                                               map_attr=self._model_params.map_attr,
                                               dropout=self._model_params.dropout)
            self.map_attn_layers = nn.MultiheadAttention(self._model_params.d_k,
                                                         num_heads=self._model_params.num_heads,
                                                         dropout=self._model_params.dropout)

        # ============================== AGENT TYPES Encoders ==============================
        # self.emb_agent_types = nn.Sequential(init_(nn.Linear(num_agent_types, self._model_params.d_k)))
        self.dec_agenttypes_encoder = nn.Sequential(
            init_(nn.Linear(2 * self._model_params.d_k, self._model_params.d_k)), nn.ReLU(),
            init_(nn.Linear(self._model_params.d_k, self._model_params.d_k))
        )

        # ============================== AutoBot-Joint DECODER ==============================
        self.Q = nn.Parameter(torch.Tensor(self._model_params.T, 1, self._model_params.c, 1, self._model_params.d_k), requires_grad=True)
        nn.init.xavier_uniform_(self.Q)

        self.social_attn_decoder_layers = []
        self.temporal_attn_decoder_layers = []
        for _ in range(self._model_params.L_dec):
            tx_decoder_layer = nn.TransformerDecoderLayer(d_model=self._model_params.d_k,
                                                          nhead=self._model_params.num_heads,
                                                          dropout=self._model_params.dropout,
                                                          dim_feedforward=self._model_params.tx_hidden_size)
            self.temporal_attn_decoder_layers.append(nn.TransformerDecoder(tx_decoder_layer, num_layers=2))
            tx_encoder_layer = nn.TransformerEncoderLayer(d_model=self._model_params.d_k,
                                                          nhead=self._model_params.num_heads,
                                                          dropout=self._model_params.dropout,
                                                          dim_feedforward=self._model_params.tx_hidden_size)
            self.social_attn_decoder_layers.append(nn.TransformerEncoder(tx_encoder_layer, num_layers=1))

        self.temporal_attn_decoder_layers = nn.ModuleList(self.temporal_attn_decoder_layers)
        self.social_attn_decoder_layers = nn.ModuleList(self.social_attn_decoder_layers)

        # ============================== Positional encoder ==============================
        self.pos_encoder = PositionalEncoding(self._model_params.d_k, dropout=0.0)

        # ============================== OUTPUT MODEL ==============================
        self.output_model = OutputModel(d_k=self._model_params.d_k, predict_yaw=self._model_params.predict_yaw)

        # ============================== Mode Prob prediction (P(z|X_1:t)) ==============================
        self.P = nn.Parameter(torch.Tensor(self._model_params.c, 1, 1, self._model_params.d_k), requires_grad=True)  # Appendix C.2.
        nn.init.xavier_uniform_(self.P)

        if self._model_params.use_map_lanes:
            self.mode_map_attn = nn.MultiheadAttention(self._model_params.d_k,
                                                       num_heads=self._model_params.num_heads)

        self.prob_decoder = nn.MultiheadAttention(self._model_params.d_k,
                                                  num_heads=self._model_params.num_heads,
                                                  dropout=self._model_params.dropout)
        self.prob_predictor = init_(nn.Linear(self._model_params.d_k, 1))

        self.train()
    
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
        agent_features = []  # List[<torch.FloatTensor: max_agents+1, future_time_steps, feature_dimension>: batch_size]
        agent_avails = []  # List[<torch.BoolTensor: max_agents+1, future_time_steps>: batch_size]

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
            sample_ego_feature = sample_ego_feature[:, : self._feature_params.future_time_steps, ...]    # [1, 16, 3]
            sample_ego_avails = sample_ego_avails[:, : self._feature_params.future_time_steps, ...]      # [1, 16]
            if sample_ego_feature.shape[1] < self._feature_params.future_time_steps:
                sample_ego_feature = pad_polylines(sample_ego_feature, self._feature_params.future_time_steps, dim=1)
                sample_ego_avails = pad_avails(sample_ego_avails, self._feature_params.future_time_steps, dim=1)

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
                    sample_agent_features = sample_agent_features[:, : self._feature_params.future_time_steps, ...]
                    sample_agent_avails = sample_agent_avails[:, : self._feature_params.future_time_steps, ...]
                    if sample_agent_features.shape[1] < self._feature_params.future_time_steps:
                        sample_agent_features = pad_polylines(
                            sample_agent_features, self._feature_params.future_time_steps, dim=1
                        )
                        sample_agent_avails = pad_avails(
                            sample_agent_avails, self._feature_params.future_time_steps, dim=1
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
                        self._feature_params.future_time_steps,
                        self._feature_params.feature_dimension,
                        dtype=torch.float32,
                        device=sample_ego_feature.device,
                    )
                    sample_agent_avails = torch.zeros(
                        self._feature_params.max_agents,
                        self._feature_params.future_time_steps,
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

    def generate_decoder_mask(self, seq_len, device):
        ''' For masking out the subsequent info. '''
        subsequent_mask = (torch.triu(torch.ones((seq_len, seq_len), device=device), diagonal=1)).bool()
        return subsequent_mask

    def process_observations(self, ego, agents):
        # ego stuff
        ego_tensor = ego[:, :, :self._model_params.k_attr]
        env_masks = ego[:, :, -1]

        # Agents stuff
        temp_masks = torch.cat((torch.ones_like(env_masks.unsqueeze(-1)), agents[:, :, :, -1]), dim=-1)
        opps_masks = (1.0 - temp_masks).type(torch.BoolTensor).to(agents.device)  # only for agents.
        opps_tensor = agents[:, :, :, :self._model_params.k_attr]  # only opponent states

        return ego_tensor, opps_tensor, opps_masks, env_masks

    def temporal_attn_fn(self, agents_emb, agent_masks, layer):
        '''
        :param agents_emb: (T, B, N, H)
        :param agent_masks: (B, T, N)
        :return: (T, B, N, H)
        '''
        T_obs = agents_emb.size(0)
        B = agent_masks.size(0)
        agent_masks = agent_masks.permute(0, 2, 1).reshape(-1, T_obs)
        agent_masks[:, -1][agent_masks.sum(-1) == T_obs] = False  # Ensure agent's that don't exist don't throw NaNs.
        agents_temp_emb = layer(self.pos_encoder(agents_emb.reshape(T_obs, B * (self._model_params._M + 1), -1)),
                                src_key_padding_mask=agent_masks)
        return agents_temp_emb.view(T_obs, B, self._model_params._M+1, -1)

    def social_attn_fn(self, agents_emb, agent_masks, layer):
        '''
        :param agents_emb: (T, B, N, H)
        :param agent_masks: (B, T, N)
        :return: (T, B, N, H)
        '''
        T_obs = agents_emb.size(0)
        B = agent_masks.size(0)
        agents_emb = agents_emb.permute(2, 1, 0, 3).reshape(self._model_params._M + 1, B * T_obs, -1)
        agents_soc_emb = layer(agents_emb, src_key_padding_mask=agent_masks.view(-1, self._model_params._M+1))
        agents_soc_emb = agents_soc_emb.view(self._model_params._M+1, B, T_obs, -1).permute(2, 1, 0, 3)
        return agents_soc_emb

    def temporal_attn_decoder_fn(self, agents_emb, context, agent_masks, layer):
        '''
        :param agents_emb: (T, BK, N, H)
        :param context: (T_in, BK, N, H)
        :param agent_masks: (BK, T, N)
        :return: (T, BK, N, H)
        '''
        T_obs = context.size(0)
        BK = agent_masks.size(0)
        time_masks = self.generate_decoder_mask(seq_len=self._model_params.T, device=agents_emb.device)
        agent_masks = agent_masks.permute(0, 2, 1).reshape(-1, T_obs)
        agent_masks[:, -1][agent_masks.sum(-1) == T_obs] = False  # Ensure that agent's that don't exist don't make NaN.
        agents_emb = agents_emb.reshape(self._model_params.T, -1, self._model_params.d_k)  # [T, BxKxN, H]
        context = context.view(-1, BK*(self._model_params._M+1), self._model_params.d_k)

        agents_temp_emb = layer(agents_emb, context, tgt_mask=time_masks, memory_key_padding_mask=agent_masks)
        agents_temp_emb = agents_temp_emb.view(self._model_params.T, BK, self._model_params._M+1, -1)

        return agents_temp_emb

    def social_attn_decoder_fn(self, agents_emb, agent_masks, layer):
        '''
        :param agents_emb: (T, BK, N, H)
        :param agent_masks: (BK, T, N)
        :return: (T, BK, N, H)
        '''
        B = agent_masks.size(0)
        agent_masks = agent_masks[:, -1:].repeat(1, self._model_params.T, 1).view(-1, self._model_params._M + 1)  # take last timestep of all agents.
        agents_emb = agents_emb.permute(2, 1, 0, 3).reshape(self._model_params._M + 1, B * self._model_params.T, -1)
        agents_soc_emb = layer(agents_emb, src_key_padding_mask=agent_masks)
        agents_soc_emb = agents_soc_emb.view(self._model_params._M + 1, B, self._model_params.T, -1).permute(2, 1, 0, 3)
        return agents_soc_emb

    def forward(self, features: FeaturesType) -> TargetsType:
        """
        Predict
        :param features: input features containing
                        {
                            "vector_map": VectorMap,
                            "agents": Agents,
                            # "scenario_type": ScenarioType,
                            # "expert": Agents, (future)
                        }
        :return: targets: predictions from network
                        {
                            "trajectory": Trajectory,
                            "mode_probs": TensorTarget(data=mode_probs), 
                            "pred": TensorTarget(data=out_dists)
                        }
        """
        vector_set_map_data = cast(VectorSetMap, features["vector_set_map"])
        ego_agent_features = cast(GenericAgents, features["generic_agents"])
        batch_size = ego_agent_features.batch_size
        
        # Extract features across batch
        agent_features, agent_avails = self.extract_agent_features(ego_agent_features, batch_size)  # [8, 1+3*30, 20, 8], [8, 1+3*30, 20]
        num_ego_agents = agent_features.shape[1]
        map_in, map_avails = self.extract_map_features(vector_set_map_data, batch_size)             # [8, 1+3*30, 160, 20, 8], [8, 1+3*30, 160, 20]
        map_in, map_avails = map_in.unsqueeze(dim=1).repeat(1, num_ego_agents, 1, 1, 1), map_avails.unsqueeze(dim=1).repeat(1, num_ego_agents, 1, 1)

        roads = torch.cat((map_in, map_avails.unsqueeze(-1)), dim=-1) # [8, 1+3*30, 160, 20, 9]
        agents_in_and_ego = torch.cat((agent_features[:,:,:,:3], agent_avails.unsqueeze(-1)), dim=3).transpose(1, 2) # [8, 20, 91, 4] # agent features' feature dimension from 3 to 8 are padded with 0s.
        ego_in = agents_in_and_ego[:, :, 0, :] # [8, 20, 4]
        agents_in = agents_in_and_ego[:, :, 1:, :] # [8, 20, 1+3*30, 4]
        
        '''
        :param ego_in: one agent called ego, shape [B, T_obs, k_attr+1] [64,4,5] with last values being the existence mask.
        :param agents_in: other scene agents, shape [B, T_obs, M-1, k_attr+1] [64,4,7,5] with last values being the existence mask.
        :param roads: [B, M, S, P, map_attr+1] [64,8,100,40,4] representing the road network or
                      [B, 1, 1] if self.use_map_lanes is False.
        :param agent_types: [B, M, num_agent_types] [64,8,6] one-hot encoding of agent types, with the first agent idx being ego.
        :return:
            pred_obs: shape [c, T, B, M, 5(6)] c trajectories for all agents with every point being the params of
                                        Bivariate Gaussian distribution (and the yaw prediction if self.predict_yaw).
            mode_probs: shape [B, c] mode probability predictions P(z|X_{1:T_obs})
        '''
        B = ego_in.size(0) # 8

        # Encode all input observations
        ego_tensor, _agents_tensor, opps_masks, env_masks = self.process_observations(ego_in, agents_in) # [8, 20, 3], [8, 20, 3*30, 3], [8, 20, 1+3*30], [8, 20]
        agents_tensor = torch.cat((ego_tensor.unsqueeze(2), _agents_tensor), dim=2) # [8, 20, 1+3*30, 3]
        agents_emb = self.agents_dynamic_encoder(agents_tensor).permute(1, 0, 2, 3) # [20, 8, 91, 128]

        # Process through AutoBot's encoder
        for i in range(self._model_params.L_enc):
            agents_emb = self.temporal_attn_fn(agents_emb, opps_masks, layer=self.temporal_attn_layers[i]) # [20, 8, 91, 128]
            agents_emb = self.social_attn_fn(agents_emb, opps_masks, layer=self.social_attn_layers[i])     # [20, 8, 91, 128]

        # Process map information
        if self._model_params.use_map_lanes:
            orig_map_features, orig_road_segs_masks = self.map_encoder(roads, agents_emb)
            map_features = orig_map_features.unsqueeze(2).repeat(1, 1, self._model_params.c, 1, 1).view(-1, B * self._model_params.c * (self._model_params._M+1), self._model_params.d_k)
            road_segs_masks = orig_road_segs_masks.unsqueeze(2).repeat(1, self._model_params.c, 1, 1).view(B * self._model_params.c * (self._model_params._M+1), -1)

        # Repeat the tensors for the number of modes.
        opps_masks_modes = opps_masks.unsqueeze(1).repeat(1, self._model_params.c, 1, 1).view(B*self._model_params.c, ego_in.shape[1], -1)
        context = agents_emb.unsqueeze(2).repeat(1, 1, self._model_params.c, 1, 1)
        context = context.view(ego_in.shape[1], B*self._model_params.c, self._model_params._M+1, self._model_params.d_k)

        # # embed agent types
        # agent_types_features = self.emb_agent_types(agent_types).unsqueeze(1).\
        #     repeat(1, self._model_params.c, 1, 1).view(-1, self._model_params._M+1, self._model_params.d_k)
        # agent_types_features = agent_types_features.unsqueeze(0).repeat(self._model_params.T, 1, 1, 1)

        # AutoBot-Joint Decoding
        dec_parameters = self.Q.repeat(1, B, 1, self._model_params._M+1, 1).view(self._model_params.T, B*self._model_params.c, self._model_params._M+1, -1)
        # dec_parameters = torch.cat((dec_parameters, agent_types_features), dim=-1)
        # dec_parameters = self.dec_agenttypes_encoder(dec_parameters)
        agents_dec_emb = dec_parameters

        for d in range(self._model_params.L_dec):
            if self._model_params.use_map_lanes and d == 1:
                agents_dec_emb = agents_dec_emb.reshape(self._model_params.T, -1, self._model_params.d_k)
                agents_dec_emb_map = self.map_attn_layers(query=agents_dec_emb, key=map_features, value=map_features,
                                                          key_padding_mask=road_segs_masks)[0]
                agents_dec_emb = agents_dec_emb + agents_dec_emb_map
                agents_dec_emb = agents_dec_emb.reshape(self._model_params.T, B*self._model_params.c, self._model_params._M+1, -1)

            agents_dec_emb = self.temporal_attn_decoder_fn(agents_dec_emb, context, opps_masks_modes, layer=self.temporal_attn_decoder_layers[d])
            agents_dec_emb = self.social_attn_decoder_fn(agents_dec_emb, opps_masks_modes, layer=self.social_attn_decoder_layers[d])

        out_dists = self.output_model(agents_dec_emb.reshape(self._model_params.T, -1, self._model_params.d_k))
        out_dists = out_dists.reshape(self._model_params.T, B, self._model_params.c, self._model_params._M+1, -1).permute(2, 0, 1, 3, 4)

        # Mode prediction
        mode_params_emb = self.P.repeat(1, B, self._model_params._M+1, 1).view(self._model_params.c, -1, self._model_params.d_k)
        mode_params_emb = self.prob_decoder(query=mode_params_emb, key=agents_emb.reshape(-1, B*(self._model_params._M+1), self._model_params.d_k),
                                            value=agents_emb.reshape(-1, B*(self._model_params._M+1), self._model_params.d_k))[0]
        if self._model_params.use_map_lanes:
            orig_map_features = orig_map_features.view(-1, B*(self._model_params._M+1), self._model_params.d_k)
            orig_road_segs_masks = orig_road_segs_masks.view(B*(self._model_params._M+1), -1)
            mode_params_emb = self.mode_map_attn(query=mode_params_emb, key=orig_map_features, value=orig_map_features,
                                                 key_padding_mask=orig_road_segs_masks)[0] + mode_params_emb

        mode_probs = self.prob_predictor(mode_params_emb).squeeze(-1).view(self._model_params.c, B, self._model_params._M+1).sum(2).transpose(0, 1)
        mode_probs = F.softmax(mode_probs, dim=1)

        # convert outputs for plotting
        pred_traj_logits = self.prob_predictor(mode_params_emb).squeeze(-1).view(self._model_params.c, B, self._model_params._M+1).permute(1, 2, 0) # [8, 91, 6]
        all_pred_agents = out_dists.permute(2, 3, 0, 1, 4) # [8, 91, 6]

        all_pred_agents, pred_traj_logits = self.sort_predictions(all_pred_agents, pred_traj_logits)

        # return  [c, T, B, 5], [B, c]
        # return out_dists, mode_probs
        
        if self._model_params.draw_visualizations:
            if self._model_params.current_task == 'training':
                pass
            if self._model_params.current_task == 'simulating':
                if self.img_num == 0:
                    self.img_folder = f"{str(Path(self._model_params.log_dir).parent)}/images/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

                multimodal_traj_objects = Trajectory(all_pred_agents[0,0,:,:,:3]) # in simulation mode, first and only trajectory of the batch
                image_ndarray = get_raster_from_vector_map_with_agents_multiple_trajectories(
                    vector_set_map_data.to_device('cpu'),
                    ego_agent_features.to_device('cpu'),
                    target_trajectory = None,
                    predicted_trajectory = multimodal_traj_objects.to_device('cpu'),
                    pixel_size = 0.1
                )
                path = f"{self.img_folder}"
                if not os.path.exists(path): os.makedirs(path, exist_ok=True)
                cv2.imwrite(f"{path}/multimodal_vis_{self.img_num:06d}.png", image_ndarray)
                self.img_num += 1
                print("SAVED IMG NUM: ", self.img_num)

        return {
            "trajectory": Trajectory(all_pred_agents[:,0,0,:,:3]),   # [8, 16, 3] ego's most likely trajectory
            "trajectories": TensorTarget(all_pred_agents),        # [8, 50, 6, 16, 3] # ego_trajs = [6][8, 16, 3]
            "mode_probs": TensorTarget(data=mode_probs),
            "pred": TensorTarget(data=out_dists),
            "all_pred_agents": TensorTarget(all_pred_agents),       # [8, 50, 6, 16, 3]
            "pred_traj_logits": TensorTarget(pred_traj_logits),     # [8, 50, 6]
        }
    

    def sort_predictions(self, all_pred_agents: torch.Tensor, pred_traj_logits: torch.Tensor) -> torch.Tensor:
        """select the trajectory with the largest probability for each batch of data
        Args:
            all_pred_agents: shape [B, M, c, T, 5] c trajectories for the ego agents with every point being the params of Bivariate Gaussian distribution.
            pred_traj_logits: shape [B, M, c] mode probability predictions P(z|X_{1:T_obs})
        """
        
        # Sort the trajectory probabilities in descending order
        pred_traj_logits_sorted, pred_traj_index = pred_traj_logits.sort(dim=-1, descending=True)

        # [batch_size, num_agents, 1, 1, 1]
        pred_traj_index_expanded = pred_traj_index[:, :, :, None, None]
        # [batch_size, num_agents, num_future_frames, 3]
        # NOTE: torch.gather can be non-deterministic -- from pytorch 1.9.0 torch.take_along_dim can be used instead
        all_pred_agents_sorted = torch.gather(
            all_pred_agents, dim=2,
            index=pred_traj_index_expanded.expand(([-1, -1, -1] + list(all_pred_agents.shape[-2:])))
            ).squeeze(2)

        
        return all_pred_agents_sorted, pred_traj_logits_sorted

