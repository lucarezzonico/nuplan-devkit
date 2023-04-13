from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Type

import torch
from torch import Tensor
import torch.nn.functional as F

from nuplan.planning.training.preprocessing.features.vector_map import VectorMap
from nuplan.planning.training.preprocessing.features.agents import Agents

from nuplan.planning.training.preprocessing.features.trajectory import Trajectory

import numpy as np

# F.pad info:
# pad last dimension by amount on left and right
# 1D: pad=(dim_0_padding_left, dim_0_padding_right)
# 2D: pad=(dim_1_padding_left, dim_1_padding_right, dim_0_padding_top, dim_0_padding_bottom)
# 3D: pad=(dim_2_padding_left, dim_2_padding_right, dim_1_padding_top, dim_1_padding_bottom, dim_0_padding_front, dim_0_padding_back)

# torch.nn.utils.rnn.pad_sequence info:
# stacks a list of Tensors along a new dimension (batchify list of tensors),
# and pads them to equal length

class NuplanToAutobotsConverter:
    def __init__(self, S=200, P=600, _M=100):
        """_summary_
        Args:
            S (int, optional): lane numbers of one scenario. Defaults to 200.
            P (int, optional): segments (points) numbers of one lane. Defaults to 600.
            _M (int, optional): agent number except ego vehicle. Defaults to 100.
        """
        self.S=S # 100
        self.P=P # 40
        self._M = _M # M-1 maximum agent number 80?


    def coords_to_map_attr(self, coords) -> Tensor:
        """Map coordinates in VectorMap format to AutoBots features
        Args:
            coords torch.Tensor with shape [num_segments, 2, 2]: one element in coords List of VectorMap
        Returns:
            point_feature_tab torch.Tensor with shape [num_segments, 4]: 4 attributes are x, y, angles, existence mask
        """
        segments_start_coords = torch.squeeze(coords[:, 0, :], dim=1)
        segments_end_coords = torch.squeeze(coords[:, 1, :], dim=1)
        segments_vector = segments_end_coords - segments_start_coords

        segments_angle = torch.atan2(segments_vector[:, 1], segments_vector[:, 0]).reshape((-1, 1))

        # get point feature tabular of shape [p_total, 3]
        point_feature_tab = torch.cat((segments_start_coords, segments_angle), dim=1)
        # pad the last dimension with existence mask all being 1
        point_feature_tab = F.pad(input=point_feature_tab,
                                  pad=(0, 1),
                                  mode="constant",
                                  value=1)

        # [NOTE] omit the first point as [0, 0, 0, 0], which greatly simplify the process and should not affect
        # too much the training
        point_feature_tab[0, :] = torch.zeros((1, 4))
        return point_feature_tab


    @torch.jit.unused
    def VectorMapToAutobotsMapTensor(self, vec_map: VectorMap) -> Tensor:
        """Pad and batchify lane coordinates and lane groupings
        Args:
            vec_map: lane_groupings: List[List[<np.ndarray: num_lane_segments_in_lane>]].
                Each lane grouping or polyline is represented by an array of indices of lane segments
                in coords belonging to the given lane. Each batch contains a List of lane groupings.
        Returns:
            Tensor: shape [B, S, P, map_attr+1] example [64,100,40,4]
        """

        # B = len(vec_map.coords) # get the number of batches

        # TODO: S and P dimension must be bigger than that of the original data.
        # Adapting dimension?

        # to debug: check the maximum number of segment contained in one lane

        # lengths = [[len(x) for x in sublist] for sublist in vec_map.lane_groupings]
        # length_maxes = [torch.max(torch.tensor(x)) for x in lengths]
        # max_p_num = torch.max(torch.tensor(length_maxes))

        # pad the lane_groupings
        # padded_list_list = [[F.pad(x, (0, max(P - len(x), 0)), 'constant', 0) for x in sublist] for sublist in vec_map.lane_groupings]
        list_idx = [torch.nn.utils.rnn.pad_sequence(sequences=sublist,
                                                    batch_first=True,
                                                    padding_value=0.) for sublist in vec_map.lane_groupings]
        # list_idx shape = List[Tensor[num_lane(varies), max_p_num(varies)]] 
        padded_list_idx = [F.pad(input=x[:, :min(x.shape[1], self.P)],
                                 pad=(0, max(self.P-x.shape[1], 0)),
                                 mode='constant',
                                 value=0) for x in list_idx]
        # list_idx shape = List[Tensor[num_lane(varies), P]]

        list_of_feature_array = [self.coords_to_map_attr(coord_mat) for coord_mat in vec_map.coords]
        # list_of_feature_array shape : List[Tensor [num_segment(varies), 4]]


        lane_features = [feature[idx.long()] for idx, feature in zip(padded_list_idx, list_of_feature_array)]  # List[Tensor(num_lane(varies), P, 4)]
        lane_features_tensor = torch.nn.utils.rnn.pad_sequence(sequences=lane_features,
                                                               batch_first=True,
                                                               padding_value=0) # Tensor(B, num_lane(varies), P, 4)

        lf_shape=lane_features_tensor.shape
        map_autobots =  F.pad(input=lane_features_tensor[:, :min(lf_shape[1], self.S), :, :],
                              pad=(0, 0, 0, 0, 0, max(self.S-lf_shape[1], 0)),
                              mode='constant',
                              value=0) # map_autobots shape is [B, S, P, 4]

        return map_autobots


    @torch.jit.unused
    def AgentsToAutobotsAgentsTensor(self, agents: Agents) -> Tensor:
        """Pads the number of agents to the max allowed agents _M (M-1)
           and builds a tensor of shape [B, T_obs, M-1, 3]
        Args:
            agents
            The structure inludes:
            ego: List[<np.ndarray: num_frames, 3>].
                The outer list is the batch dimension.
                The num_frames includes both present and past frames.
                The last dimension is the ego pose (x, y, heading) at time t.
                Example dimensions: 8 (batch_size) x 5 (1 present + 4 past frames) x 3
            agents: List[<np.ndarray: num_frames, num_agents, 8>].
                The outer list is the batch dimension.
                The num_frames includes both present and past frames.
                The num_agents is padded to fit the largest number of agents across all frames.
                The last dimension is the agent pose (x, y, heading) velocities (vx, vy, yaw rate)
                and size (length, width) at time t.
        Returns:
            Tensor: [B, T_obs, M-1, k_attr+1] example [64,5,100,3]
        """
        # every scenario may have different number of agents

        # lengths = [x.shape[1] for x in agents.agents]
        # length_maxes = [torch.max(torch.tensor(x)) for x in lengths]
        # max_length = torch.max(torch.tensor(length_maxes))

        # agents.agents: List[Tensor[T_obs, num_agents(varies), 8]]
        padded_list = [F.pad(input=x[:,:min(self._M, x.shape[1]),:],
                             pad=(0, 0, 0, max(self._M - x.shape[1],0)),
                             mode='constant',
                             value=0) for x in agents.agents]
        # padded_list: List[Tensor[T_obs, _M, 8]]
        agents_ts = torch.stack(padded_list)
        # agents_ts: Tensor[B, T_obs, _M, 8]
        agents_ts = agents_ts[:, :, :, :3]  # take only x, y coordinates and an additional dimension to be existence mask
        # agents_ts: Tensor[B, T_obs, _M, 3]
        agents_ts[:, :, :, 2] = (agents_ts[:, :, :, 2] != 0).float() # modify the last column to be existence mask
        # agents_ts: Tensor[B, T_obs, _M, 3]
        return agents_ts


    @torch.jit.unused 
    def AgentsToAutobotsEgoinTensor(self, agents: Agents):
        """Turns list into a tensor of B batches: [B, T_obs, 3]
        Args:
            agents
            The structure inludes:
            ego: List[<np.ndarray: num_frames, 3>].
                The outer list is the batch dimension.
                The num_frames includes both present and past frames.
                The last dimension is the ego pose (x, y, heading) at time t.
                Example dimensions: 8 (batch_size) x 5 (1 present + 4 past frames) x 3
            agents: List[<np.ndarray: num_frames, num_agents, 8>].
                The outer list is the batch dimension.
                The num_frames includes both present and past frames.
                The num_agents is padded to fit the largest number of agents across all frames.
                The last dimension is the agent pose (x, y, heading) velocities (vx, vy, yaw rate)
                and size (length, width) at time t.
        Returns:
            Tensor: [B, T_obs, k_attr+1] example [64,5,3]
        """

        ego_in=torch.stack(agents.ego)
        # if ego_in.dim == 2:
        #     ego_in=torch.unsqueeze(ego_in, 0) # if two dimension, unsqueeze to create one more "batch" dimension

        ego_in[:,:,2]=1
        return ego_in


    @torch.jit.unused
    def TrajectoryToAutobotsEgoin(self, traj: Trajectory) -> Tensor:
        target_ts=traj.data
        target_ts[:,:,2]=1
        return target_ts


    @torch.jit.unused
    def output_tensor_to_trajectory(self, pred_obs: Tensor, mode_probs: Tensor) -> Trajectory:
        """select the trajectory with the largest probability for each batch of data
        Args:
            pred_obs: shape [c, T, B, 5] c trajectories for the ego agents with every point being the params of Bivariate Gaussian distribution.
            mode_probs: shape [B, c] mode probability predictions P(z|X_{1:T_obs})
        """

        most_likely_idx=torch.argmax(mode_probs, 1)
        # for each batch, pick the trajectory with largest probability
        trajs=torch.stack([pred_obs[most_likely_idx[i],:,i,:] for i in range(pred_obs.shape[2])])

        trajs_3=trajs[:,:,:3]

        trajs_3[:,:,-1] = 0

        # ang_vec=trajs_3[:,1:,:2] - trajs_3[:,:-1,:2] 
        # ang = torch.atan2(ang_vec[:,:,0], ang_vec[:,:,1])
        # trajs_3[:,:-1,2] = ang
        # trajs_3[:,-1,2] = trajs_3[:,-2,2]

        return Trajectory(data=trajs_3)