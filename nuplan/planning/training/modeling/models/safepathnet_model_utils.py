from typing import Optional, Tuple

import torch
from torch import nn

from nuplan.planning.training.modeling.models.urban_driver_open_loop_model_utils import MLP

from nuplan.planning.training.modeling.models.safepathnet_transformer import (
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer
)


def build_matrix(translation: torch.Tensor, angle: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build transform matrix from translation and angle."""
    assert translation.shape[:-1] == angle.shape

    c = torch.cos(angle)
    s = torch.sin(angle)
    rotation = torch.zeros(list(angle.shape) + [2, 2], device=translation.device)
    rotation[..., 0, 0] = c
    rotation[..., 0, 1] = -s
    rotation[..., 1, 0] = s
    rotation[..., 1, 1] = c
    linear_t = torch.transpose(rotation, -1, -2)

    # Construct 3x3 matrix
    matrix = torch.zeros(list(angle.shape) + [3, 3], device=translation.device)
    matrix[..., 0:2, 0:2] = rotation
    matrix[..., 0:2, 2] = translation
    matrix[..., 2, 2] = 1.0

    # Construct 3x3 inverse matrix
    inverse_matrix = torch.zeros_like(matrix)
    inverse_matrix[..., 0:2, 0:2] = linear_t
    inverse_matrix[..., 0:2, 2] = -(torch.matmul(linear_t, translation.unsqueeze(-1))).squeeze(-1)
    inverse_matrix[..., 2, 2] = 1.0

    assert matrix is not None
    assert inverse_matrix is not None

    return matrix, inverse_matrix


class MultiheadAttentionGlobalHead(nn.Module):
    """
    Copied from L5Kit's implementation `MultiheadAttentionGlobalHead`:
    https://github.com/woven-planet/l5kit/blob/master/l5kit/l5kit/planning/vectorized/global_graph.py.
    Changes:
        1. Add input & output description for `__init__`, `forward`
        2. Add num_mlp_layers & hidden_size_scaling to adjust MLP layers
        3. Change input variable `d_model` to `global_embedding_size`

    Global graph making use of multi-head attention.
    """

    def __init__(
        self,
        global_embedding_size: int,
        num_timesteps: int,
        num_outputs: int,
        nhead: int = 8,
        dropout: float = 0.1,
        hidden_size_scaling: int = 4,
        num_mlp_layers: int = 3,
    ):
        """
        Constructs global multi-head attention layer.
        :param global_embedding_size: Feature size.
        :param num_timesteps: Number of output timesteps.
        :param num_outputs: Number of output features per timestep.
        :param nhead: Number of attention heads. Default 8: query=ego, keys=types,ego,agents,map, values=ego,agents,map.
        :param dropout: Float in range [0,1] for level of dropout. Set to 0 to disable it. Default 0.1.
        :param hidden_size_scaling: Controls hidden layer size, scales embedding dimensionality. Default 4.
        :param num_mlp_layers: Num MLP layers. Default 3.
        """
        super().__init__()
        self.num_timesteps = num_timesteps
        self.num_outputs = num_outputs
        self.encoder = nn.MultiheadAttention(global_embedding_size, nhead, dropout=dropout)
        self.output_embed = MLP(
            global_embedding_size,
            global_embedding_size * hidden_size_scaling,
            num_timesteps * num_outputs,
            num_mlp_layers,
        )

    def forward(
        self, inputs: torch.Tensor, type_embedding: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward of the module.
        :param inputs: Model inputs. [1 + N + M, batch_size, feature_dim]
        :param type_embedding: Type embedding describing the different input types. [1 + N + M, batch_size, feature_dim]
        :param mask: Availability mask. [batch_size, 1 + N + M]
        :return Tuple of outputs, attention.
        """
        # dot-product attention:
        #   - query is ego's vector
        #   - key is inputs plus type embedding
        #   - value is inputs
        out, attns = self.encoder(inputs[[0]], inputs + type_embedding, inputs, mask)
        outputs = self.output_embed(out[0]).view(-1, self.num_timesteps, self.num_outputs)
        return outputs, attns
    
    
class TrajectoryMatcher(nn.Module):
    """Picks the closest trajectory based on provided costs per trajectory and trajectory logits.
    """

    def __init__(self, cost_prob_coeff: float = 1):
        """Initializes the class.

        :param cost_prob_coeff: coefficient for the trajectory probability cost
        """
        super().__init__()
        self.cost_prob_coeff = cost_prob_coeff

    @torch.no_grad()
    def forward(self, trajectory_costs: torch.Tensor, trajectory_logits: torch.Tensor) -> torch.Tensor:
        """Returns the indices corresponding to the lowest-cost trajectories

        :param trajectory_costs: cost of each trajectory (e.g. loss per traj.). Shape [batch_size, num_trajectories] # [16, 50, 20]
        :param trajectory_logits: logits of the trajectory distribution. Shape [batch_size, num_trajectories] # [16, 50, 20]
        :return: indices of the trajectories with the lowest cost. Shape [batch_size]
        """
        # [batch_size, num_trajectories]
        traj_prob_cost = trajectory_logits.softmax(dim=-1)  # [16, 50, 20]
        # [batch_size]
        indices = torch.argmin(trajectory_costs - traj_prob_cost * self.cost_prob_coeff, dim=-1)  # [16, 50]
        return indices


class TransformerGlobalGraph(nn.Module):
    """This transformer encoder-decoder module implements self attention across polyline embeddings.
    This represents a global graph that generates attention between each polyline towards other polyline.
    Then, the decoder queries the global graph features with the given queries.
    """

    def __init__(self, dim_in: int = 128, projection_dim: int = 256, nhead: int = 8,
                 dim_feedforward: int = 1024, num_layers: int = 3, num_decode_layers: int = 3,
                 dropout: float = 0.1) -> None:
        """Initializes the transformer model for the global graph

        :param dim_in: The dimension of the polyline embeddings
        :param projection_dim: The dimension of the projection layer
        :param nhead: The number of parallel attention heads in multihead attention
        :param dim_feedforward: The dimension of the feedforward layer in the transformer encoder
        :param num_layers: The number of transformer encoder layers.
        :param num_decode_layers: The number of transformer decoder layers.
        :param dropout: The dropout probability for the mha layer
        """
        super(TransformerGlobalGraph, self).__init__()
        self.dim_in = dim_in

        self.encoder_projection_layer = nn.Linear(dim_in, projection_dim, bias=False)
        self.tdecoder_projection_layer = nn.Linear(dim_in, projection_dim, bias=False)

        encoder_layer = TransformerEncoderLayer(projection_dim, nhead=nhead,
                                                dim_feedforward=dim_feedforward,
                                                dropout=dropout)
        self.encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.encoder.reset_parameters()

        tdecoder_layer = TransformerDecoderLayer(projection_dim, nhead=nhead,
                                                 dim_feedforward=dim_feedforward,
                                                 dropout=dropout)
        self.tdecoder = TransformerDecoder(tdecoder_layer, num_layers=num_decode_layers)
        self.tdecoder.reset_parameters()

    def forward(self, polys_embed: torch.Tensor,
                polys_embed_decoder: torch.Tensor,
                invalid_mask: torch.Tensor,
                tgt_invalid_mask: Optional[torch.Tensor],
                types: torch.Tensor) -> torch.Tensor:
        """Forward pass for the transformer model based on polyline and type embeddings, the polyline invalid
          mask, the polyline decoder queries. The attention keys are the sum of polyline and type embedding.

        :param polys_embed: input tensor representing the polylines embeddings
            shape: [batch_size, num_polylines, dim_in] (float)
        :param polys_embed_decoder: input tensor representing the transformer decoder queries
            shape: [batch_size, num_polylines, dim_in] (float)
        :param invalid_mask: bool tensor representing the mask of polylines with no valid information
            shape: [batch_size, num_polylines] (bool)
        :param tgt_invalid_mask: bool tensor representing the mask of decoded polylines with no valid information
            shape: [batch_size, num_output_polylines] (bool)
        :param types: float tensor representing the polyline type embeddings
            shape: [batch_size, num_polylines] (float)
        """
        assert polys_embed.shape[-1] == self.dim_in, "The input embedding dimension must be the same as the layer"
        assert polys_embed[..., 0].shape == invalid_mask.shape, \
            "The shape of the polylines embeddings and invalid mask are not consistent"
        assert invalid_mask.dtype == torch.bool, "The invalid_mask type must be of type torch.bool"
        assert tgt_invalid_mask is None or tgt_invalid_mask.dtype == torch.bool, \
            "The tgt_invalid_mask type must be of type torch.bool"
        assert polys_embed.shape[:-1] == types.shape[:-1], \
            "The shape of the polylines embeddings and types are not consistent"

        projected_features = self.encoder_projection_layer(polys_embed)

        src = (projected_features + types).transpose(0, 1)

        # [num_polylines + (self.num_trajectories - 1), batch_size, projection_dim]
        enc_out = self.encoder(src, src_key_padding_mask=invalid_mask)

        # ego and agents decoder
        decoder_queries = self.tdecoder_projection_layer(polys_embed_decoder).transpose(0, 1)

        out = self.tdecoder.forward(tgt=decoder_queries,
                                    memory=enc_out,
                                    memory_key_padding_mask=invalid_mask,
                                    tgt_key_padding_mask=tgt_invalid_mask).transpose(0, 1)

        return out


class MultimodalTransformerGlobalGraph(nn.Module):
    """ This class extends the transformer global graph to the multimodal case (prediction of multiple trajectories).
    Each agent embedding is repeated `(agent_)num_trajectories` times and a set of `(agent_)num_trajectories` learned
    embeddings is added to the cloned ego features to obtain diverse queries.
    """

    def __init__(self, transformer_global_graph: TransformerGlobalGraph,
                 num_trajectories: int, agent_num_trajectories: int = 1, max_num_agents: int = 0) -> None:
        """Initializes the multimodal transformer global graph.

        :param transformer_global_graph: The standard transformer global graph
        :param num_trajectories: Number of different ego trajectories to predict (unused)
        :param agent_num_trajectories: Number of different agent trajectories to predict
        :param max_num_agents: Maximum number of agents in the scene (used to construct the prediction mask)
        """
        super().__init__()
        self.transformer_global_graph = transformer_global_graph
        self.num_trajectories = num_trajectories
        self.agent_num_trajectories = agent_num_trajectories
        self.max_num_agents = max_num_agents

        self.register_parameter('egos_embedding', nn.Parameter(
            torch.randn(num_trajectories, self.transformer_global_graph.dim_in)))

        self.register_parameter('agents_embedding', nn.Parameter(
            torch.randn(agent_num_trajectories, self.transformer_global_graph.dim_in)))

    def repeat_features(self, features: torch.Tensor, num_repeats: int) -> torch.Tensor:
        """Repeat input features num_repeats times on dimension 1 in a deterministic way
            (avoiding the usage of repeat_interleave).

        :param features: Input features to be repeated
        :param num_repeats: Number of times features will be repeated on dimension 1
        """
        assert features.ndim == 3
        tmp_new_shape = list(features.shape)
        tmp_new_shape.insert(2, num_repeats)
        repeated_features = torch.empty(tmp_new_shape, dtype=features.dtype, device=features.device)
        repeated_features[:, :, :] = features.unsqueeze(2)
        new_shape = list(features.shape)
        new_shape[1] *= num_repeats
        repeated_features = repeated_features.view(new_shape)
        # The same thing could be done with repeat_interleave, but that function is non-deterministic
        # repeated_features_nondeterministic = features.repeat_interleave(num_repeats, dim=1)
        # assert torch.allclose(repeated_features_nondeterministic, repeated_features)
        return repeated_features

    def repeat_ego_features(self, features: torch.Tensor) -> torch.Tensor:
        """Repeat ego features self.num_trajectories times"""
        return self.repeat_features(features, self.num_trajectories)

    def repeat_agent_features(self, features: torch.Tensor) -> torch.Tensor:
        """Repeat agents features self.num_trajectories times"""
        return self.repeat_features(features, self.agent_num_trajectories)

    def forward(self, polys_embed: torch.Tensor,
                invalid_mask: torch.Tensor,
                types: torch.Tensor) -> torch.Tensor:
        """Clones the agent features (self.num_trajectories - 1) times before calling the parent forward method.

        The model returns agent embeddings with shape [batch_size, num_agents, dim_in]

        :param polys_embed: input tensor representing the polylines embeddings
            shape: [batch_size, num_polylines, dim_in] (float)
        :param invalid_mask: bool tensor representing the mask of polylines with no valid information
            shape: [batch_size, num_polylines] (bool)
        :param types: float tensor representing the polylines type embedding
            shape: [batch_size, num_polylines, dim_in] (float)
        """
        assert polys_embed.shape[-1] == self.egos_embedding.shape[1], \
            "The input embedding dimension must be the same as the ego embedding."
        assert polys_embed[..., 0].shape == invalid_mask.shape, \
            "The shape of the polylines embeddings and invalid mask are not consistent"
        assert invalid_mask.dtype == torch.bool, "The invalid_mask type must be of type torch.bool"
        assert polys_embed.shape[:-1] == types.shape[:-1], \
            "The shape of the polylines embeddings and types are not consistent"

        # # [batch_size, 1, transformer_global_graph.dim_in]
        polys_embed_ego = polys_embed[:, :1]
        invalid_mask_ego = invalid_mask[:, :1]
        invalid_mask_ego_repeated = self.repeat_ego_features(invalid_mask_ego.unsqueeze(-1)).squeeze(-1)
        repeated_ego = self.repeat_ego_features(polys_embed_ego) + self.egos_embedding
        
        # set the prediction mask to get the agents
        # [num_polylines]
        prediction_mask = torch.zeros(polys_embed.shape[1], dtype=torch.bool, device=polys_embed.device)
        prediction_mask[1:1 + self.max_num_agents] = True   # prediction_mask[1:1 + self.max_num_agents] = True  # Solves Loss problem due to unused model outputs

        num_agents = int(prediction_mask.sum().item())
        # [batch_size, num_agents, transformer_global_graph.dim_in]
        polys_embed_agents = polys_embed[:, prediction_mask]
        invalid_mask_agents = invalid_mask[:, prediction_mask]
        invalid_mask_agents_repeated = self.repeat_agent_features(invalid_mask_agents.unsqueeze(-1)).squeeze(-1)
        agents_embedding_repeated = self.agents_embedding.repeat((num_agents, 1))
        repeated_agents = self.repeat_agent_features(polys_embed_agents) + agents_embedding_repeated
        decoder_invalid_mask = invalid_mask_agents_repeated
        
        # concatenate ego and agents
        repeated_ego_agents = torch.cat([repeated_ego, repeated_agents], dim=1)
        invalid_mask_ego_agents_repeated = torch.cat([invalid_mask_ego_repeated, invalid_mask_agents_repeated], dim=1)
        decoder_invalid_mask = invalid_mask_ego_agents_repeated
        
        out: torch.Tensor = self.transformer_global_graph(
            polys_embed=polys_embed, polys_embed_decoder=repeated_ego_agents,
            invalid_mask=invalid_mask, tgt_invalid_mask=decoder_invalid_mask, types=types)

        return out


class MultimodalDecoder(nn.Module):
    """The multimodal global graph model, based on the transformer encoder-decoder model.
    """

    def __init__(self, dim_in: int = 128, projection_dim: int = 256, nhead: int = 8,
                 dim_feedforward: int = 1024, num_layers: int = 3, num_decode_layers: int = 3,
                 dropout: float = 0.1, future_num_frames: int = 45,
                 agent_future_num_frames: int = 30, num_outputs: int = 3, num_mlp_layers: int = 3,
                 num_trajectories: int = 10, agent_num_trajectories: int = 5, max_num_agents: int = 30) -> None:
        """Initializes the model.

        :param dim_in: The dimension of the polyline embeddings
        :param projection_dim: The dimension of the projection layer
        :param nhead: The number of parallel attention heads in multihead attention
        :param dim_feedforward: The dimension of the feedforward layer in the transformer encoder
        :param num_layers: The number of transformer encoder layers.
        :param num_decode_layers: The number of transformer decoder layers.
        :param dropout: The dropout probability for the mha layer
        :param future_num_frames: number of predicted ego future steps
        :param agent_future_num_frames: number of predicted agent future steps
        :param num_outputs: number of outputs (3: x, y, yaw)
        :param num_mlp_layers: number of linear layers for the MLP decoder
        :param num_trajectories: Number of different ego trajectories to predict (unused)
        :param agent_num_trajectories: Number of different agent trajectories to predict
        :param max_num_agents: Maximum number of agents in the scene (used to construct the prediction mask)
        """
        super(MultimodalDecoder, self).__init__()
        self.num_timesteps = future_num_frames
        self.num_trajectories = num_trajectories
        self.num_outputs = num_outputs
        self.agent_num_timesteps = agent_future_num_frames
        self.agent_num_trajectories = agent_num_trajectories

        transformer_global_graph = TransformerGlobalGraph(dim_in, projection_dim, nhead, dim_feedforward, num_layers,
                                                          num_decode_layers, dropout)
        self.transformer = MultimodalTransformerGlobalGraph(transformer_global_graph, num_trajectories,
                                                            agent_num_trajectories, max_num_agents)

        self.ego_prediction_model = MLP(
            projection_dim,
            projection_dim * 2,
            future_num_frames * 3 + (1 if num_trajectories > 1 else 0),
            num_layers=num_mlp_layers,
        )
        
        self.agent_prediction_model = MLP(
            projection_dim,
            projection_dim * 2,
            agent_future_num_frames * 3 + (1 if agent_num_trajectories > 1 else 0),
            num_layers=num_mlp_layers,
        )

        self.multimodal_predictions = self.num_trajectories > 1
        self.agent_multimodal_predictions = self.agent_num_trajectories > 1

    def forward(self, polys_embed: torch.Tensor, types: torch.Tensor,
                invalid_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward of the model. Returns:
        - the agent predictions, shape:
            [batch_size, num_agents, agent_num_trajectories, agent_num_timesteps, num_outputs]
        - the associated trajectory probabilities, shape:
            [batch_size, num_agents, agent_num_trajectories]

        :param polys_embed: input tensor representing the polylines embeddings
            shape: [batch_size, num_polylines, dim_in] (float)
        :param types: float tensor representing the polylines type embedding
            shape: [batch_size, num_polylines, dim_in] (float)
        :param invalid_mask: bool tensor representing the mask of polylines with no valid information
            shape: [batch_size, num_polylines] (bool)
        """
        ego_agents_embedding = self.transformer(polys_embed, invalid_mask, types)
        ego_embedding = ego_agents_embedding[:, :self.num_trajectories]
        agents_embedding = ego_agents_embedding[:, self.num_trajectories:]
        
        # PREDICT EGO
        ego_decoder_outputs = self.ego_prediction_model(ego_embedding).unsqueeze(dim=1)
        if self.multimodal_predictions:
            # [batch_size, num_agents, num_trajectories, future_num_frames * 3]
            ego_predictions = ego_decoder_outputs[:, :, :, :-1]
            # [batch_size, num_agents, num_trajectories]
            ego_traj_probs = ego_decoder_outputs[:, :, :, -1]
        else:
            # [batch_size, num_agents, 1, future_num_frames * 3]
            ego_predictions = ego_decoder_outputs
            # [batch_size, num_agents, 1]
            ego_traj_probs = torch.ones_like(
                ego_decoder_outputs[..., 0])  # create a fake probability distrib.
            
        # [batch_size, num_agents, num_trajectories, num_timesteps, 3]
        ego_predictions = ego_predictions.view(
            list(ego_predictions.shape[:3]) + [self.num_timesteps, 3])
        
        # PREDICT AGENTS
        # [batch_size, num_agents, num_trajectories, num_features]
        agents_embedding = agents_embedding.view(agents_embedding.shape[0], -1, self.agent_num_trajectories,
                                                 agents_embedding.shape[-1])
        # [batch_size, num_agents, num_trajectories, future_num_frames * 3 (+ 1 if self.agent_multimodal_predictions)]
        agents_decoder_outputs = self.agent_prediction_model(agents_embedding)
        if self.agent_multimodal_predictions:
            # [batch_size, num_agents, num_trajectories, future_num_frames * 3]
            agents_predictions = agents_decoder_outputs[:, :, :, :-1]
            # [batch_size, num_agents, num_trajectories]
            agents_traj_probs = agents_decoder_outputs[:, :, :, -1]
        else:
            # [batch_size, num_agents, 1, future_num_frames * 3]
            agents_predictions = agents_decoder_outputs
            # [batch_size, num_agents, 1]
            agents_traj_probs = torch.ones_like(
                agents_decoder_outputs[..., 0])  # create a fake probability distrib.

        # [batch_size, num_agents, num_trajectories, num_timesteps, 3]
        agents_predictions = agents_predictions.view(
            list(agents_predictions.shape[:3]) + [self.agent_num_timesteps, 3])
        
        
        # concatenate ego and agents
        ego_agents_predictions = torch.cat([ego_predictions, agents_predictions], dim=1)
        ego_agents_traj_probs = torch.cat([ego_traj_probs, agents_traj_probs], dim=1)

        return ego_agents_predictions, ego_agents_traj_probs



