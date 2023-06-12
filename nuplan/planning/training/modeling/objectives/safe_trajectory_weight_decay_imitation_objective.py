from typing import Dict, List, cast

import torch

from nuplan.planning.training.modeling.objectives.abstract_objective import AbstractObjective
from nuplan.planning.training.modeling.objectives.scenario_weight_utils import extract_scenario_type_weight
from nuplan.planning.training.modeling.types import FeaturesType, ScenarioListType, TargetsType
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory
from nuplan.planning.training.preprocessing.features.trajectories import Trajectories

from torch.nn import functional as F

from nuplan.planning.training.preprocessing.features.tensor_target import TensorTarget

from nuplan.planning.training.modeling.models.safepathnet_model_utils import (
    MultimodalDecoder,
    TrajectoryMatcher,
    build_matrix
)

class SafeTrajectoryWeightDecayImitationObjective(AbstractObjective):
    """
    Objective that drives the model to imitate the signals from expert behaviors/trajectories.
    When comparing model's predictions to expert's trajectory, it assigns more weight to earlier timestamps than later ones.
    Formula: mean(loss * exp(-index / poses))
    """

    def __init__(self, scenario_type_loss_weighting: Dict[str, float], weight: float = 1.0):
        """
        Initializes the class

        :param name: name of the objective
        :param weight: weight contribution to the overall loss
        """
        self._name = 'safe_trajectory_weight_decay_imitation_objective'
        self._weight = weight
        self._decay = 1.0

        self._fn_xy = torch.nn.modules.loss.L1Loss(reduction='none')
        self._fn_heading = torch.nn.modules.loss.L1Loss(reduction='none')
        self._scenario_type_loss_weighting = scenario_type_loss_weighting
        
        self.criterion = torch.nn.modules.loss.L1Loss(reduction='none')
        self.l1_loss = torch.nn.modules.loss.L1Loss(reduction='none')
        self.l2_loss = torch.nn.modules.loss.MSELoss(reduction='none')
        
        self.agent_traj_matcher = TrajectoryMatcher(cost_prob_coeff=0.1)
        self.eps = float(torch.finfo(torch.float).eps)

    def name(self) -> str:
        """
        Name of the objective
        """
        return self._name

    def get_list_of_required_target_types(self) -> List[str]:
        """Implemented. See interface."""
        return ["trajectory"]

    def compute(self, predictions: FeaturesType, targets: TargetsType, scenarios: ScenarioListType) -> torch.Tensor:
        """
        Computes the objective's loss given the ground truth targets and the model's predictions
        and weights it based on a fixed weight factor.

        :param predictions: model's predictions (model's outputs)
        :param targets: ground truth targets from the dataset (according to target_builders in model's init)
        :return: loss scalar tensor
        """
        predicted_trajectory = cast(Trajectory, Trajectory(predictions["trajectories"].data[:,0,0,:,:]))
        targets_trajectory = cast(Trajectory, targets["trajectory"])    # predictions["target"] for closed loop, same for metrics
        loss_weights = extract_scenario_type_weight(
            scenarios, self._scenario_type_loss_weighting, device=predicted_trajectory.xy.device
        )

        # Add exponential decay of loss such that later error induce less penalty
        planner_output_steps = predicted_trajectory.xy.shape[1]
        decay_weight = torch.ones_like(predicted_trajectory.xy)
        decay_value = torch.exp(-torch.Tensor(range(planner_output_steps)) / (planner_output_steps * self._decay))
        decay_weight[:, :] = decay_value.unsqueeze(1)

        broadcast_shape_xy = tuple([-1] + [1 for _ in range(predicted_trajectory.xy.dim() - 1)])
        broadcasted_loss_weights_xy = loss_weights.view(broadcast_shape_xy)
        broadcast_shape_heading = tuple([-1] + [1 for _ in range(predicted_trajectory.heading.dim() - 1)])
        broadcasted_loss_weights_heading = loss_weights.view(broadcast_shape_heading)

        weighted_xy_loss = self._fn_xy(predicted_trajectory.xy, targets_trajectory.xy) * broadcasted_loss_weights_xy
        weighted_heading_loss = (
            self._fn_heading(predicted_trajectory.heading, targets_trajectory.heading)
            * broadcasted_loss_weights_heading
        )

        # Assert that broadcasting was done correctly
        assert weighted_xy_loss.size() == predicted_trajectory.xy.size()
        assert weighted_heading_loss.size() == predicted_trajectory.heading.size()

        return self._weight * (
            torch.mean(weighted_xy_loss * decay_weight) + torch.mean(weighted_heading_loss * decay_weight[:, :, 0])
        )
        
    def compute_new(self, predictions: FeaturesType, targets: TargetsType, scenarios: ScenarioListType) -> torch.Tensor:
        """
        Computes the objective's loss given the ground truth targets and the model's predictions
        and weights it based on a fixed weight factor.

        :param predictions: model's predictions (model's outputs)
        :param targets: ground truth targets from the dataset (according to target_builders in model's init)
        :return: loss scalar tensor
        """
        
        # trajectories = cast(Trajectories, predictions["trajectories"]).trajectories
        pred_agents = cast(TensorTarget, predictions["pred_agents"]).data
        pred_traj_logits = cast(TensorTarget, predictions["pred_traj_logits"]).data
        target = cast(TensorTarget, predictions["target"]).data
        target_avails = cast(TensorTarget, predictions["target_avails"]).data
        
        # compute loss on trajectories
        agent_loss = self.criterion(pred_agents, target)   # [16, 50, 20, 30, 3]
        # agent_loss_xy = self.l2_loss(pred_agents[:,:,:,:,:2], target[:,:,:,:,:2])
        # agent_loss_yaw = self.l1_loss(pred_agents[:,:,:,:,2:3], target[:,:,:,:,2:3])
        # agent_loss = torch.cat([agent_loss_xy, agent_loss_yaw], dim=-1)
        agent_loss *= target_avails.unsqueeze(-1)   # [16, 50, 20, 30, 3]
        pred_num_valid_targets = target_avails.sum().float()    # [1]

        pred_num_valid_targets /= 6 # self.global_head.agent_num_trajectories
        any_target_avails = torch.any(target_avails, dim=-1) # [16, 50, 20]

        # [batch_size, num_agents, agent_num_trajectories]
        loss_pred_per_trajectory = agent_loss.sum(-1).sum(-1) * any_target_avails   # [16, 50, 20]

        # [batch_size, num_agents]
        pred_loss_argmin_idx = self.agent_traj_matcher(loss_pred_per_trajectory / (target_avails.sum(-1) + self.eps), pred_traj_logits) # [16, 50]

        # compute loss on probability distribution for valid targets only
        pred_prob_loss = F.cross_entropy(pred_traj_logits[any_target_avails[..., 0]], pred_loss_argmin_idx[any_target_avails[..., 0]])  # [1]

        # compute the final loss only on the trajectory with the lowest loss
        # [batch_size, num_agents, num_traj]
        pred_traj_loss_batch = agent_loss.sum(-1).sum(-1)  # [16, 50, 20]
        # [batch_size, num_agents]
        # NOTE torch.gather can be non-deterministic -- from pytorch 1.9.0 torch.take_along_dim can be used instead
        pred_traj_loss_batch = torch.gather(pred_traj_loss_batch, dim=-1, index=pred_loss_argmin_idx[:, :, None]).squeeze(-1)  # [16, 50]
        # zero out invalid targets (agents)
        pred_traj_loss_batch = pred_traj_loss_batch * any_target_avails[..., 0]  # [16, 50]
        # [1]
        pred_traj_loss = pred_traj_loss_batch.sum() / (pred_num_valid_targets + self.eps) # [1]
        # compute overall loss
        pred_loss = pred_traj_loss + pred_prob_loss * self.agent_traj_matcher.cost_prob_coeff # [1]

        # train_dict = {
        #     "loss": pred_loss,
        #     "loss/agents_traj": pred_traj_loss,
        #     "loss/agents_traj_prob": pred_prob_loss,
        # }
        
        return pred_loss # cast(torch.Tensor, predictions["loss"]).data
    
