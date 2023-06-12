from typing import Dict, List, cast, Tuple

import torch

from nuplan.planning.training.modeling.objectives.abstract_objective import AbstractObjective
from nuplan.planning.training.modeling.objectives.scenario_weight_utils import extract_scenario_type_weight
from nuplan.planning.training.modeling.types import FeaturesType, ScenarioListType, TargetsType
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory
from nuplan.planning.training.preprocessing.features.trajectories import Trajectories
from nuplan.planning.training.preprocessing.features.tensor_target import TensorTarget


class TrajectoryWeightDecayImitationObjective(AbstractObjective):
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
        self._name = 'trajectory_weight_decay_imitation_objective'
        self._weight = weight
        self._decay = 1.0

        self._fn_xy = torch.nn.modules.loss.L1Loss(reduction='none')
        self._fn_heading = torch.nn.modules.loss.L1Loss(reduction='none')
        self._scenario_type_loss_weighting = scenario_type_loss_weighting
        
        self.alpha = 1.0
        self.beta = 1.0

    def name(self) -> str:
        """
        Name of the objective
        """
        return self._name

    def get_list_of_required_target_types(self) -> List[str]:
        """Implemented. See interface."""
        return ["trajectory"]

    def compute_unimodal(self, predictions: FeaturesType, targets: TargetsType, scenarios: ScenarioListType) -> torch.Tensor:
        """
        Computes the objective's loss given the ground truth targets and the model's predictions
        and weights it based on a fixed weight factor.

        :param predictions: model's predictions (model's outputs)
        :param targets: ground truth targets from the dataset (according to target_builders in model's init)
        :return: loss scalar tensor
        """
        predicted_trajectory = cast(Trajectory, predictions["trajectory"])
        # predicted_trajectories = cast(TensorTarget, predictions["multimodal_outputs"])
        targets_trajectory = cast(Trajectory, targets["trajectory"])    # predictions["target"] for closed loop, same for metrics
        loss_weights = extract_scenario_type_weight(scenarios, self._scenario_type_loss_weighting, device=predicted_trajectory.xy.device)
        # loss_weights = loss_weights.unsqueeze(dim=1).repeat(1,predicted_trajectories.data.shape[1]).view(-1)
        
        # # Reshape predicted and target trajectories to be of shape (batch_size * num_predictions, num_timesteps, num_features)
        # predicted_trajectory = Trajectory(predicted_trajectories.data.reshape(-1, predicted_trajectories.data.shape[-2], predicted_trajectories.data.shape[-1]))
        # targets_trajectory = Trajectory(targets_trajectory.data.unsqueeze(dim=1).repeat(1,predicted_trajectories.data.shape[1],1,1).view(-1, targets_trajectory.data.shape[-2], targets_trajectory.data.shape[-1]))
        
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
        
    def min_ade(self, traj: torch.Tensor, traj_gt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes average displacement error for the best trajectory is a set, with respect to ground truth
        :param traj: predictions, shape [batch_size, num_modes, sequence_length, 2]
        :param traj_gt: ground truth trajectory, shape [batch_size, sequence_length, 2]
        :param masks: masks for varying length ground truth, shape [batch_size, sequence_length]
        :return errs, inds: errors and indices for modes with min error, shape [batch_size]
        """
        num_modes = traj.shape[1]
        sequence_length = traj.shape[2]

        traj_gt_rpt = traj_gt.unsqueeze(1).repeat(1, num_modes, 1, 1)
        err = traj_gt_rpt[:, :, :, 0:3] - traj[:, :, :, 0:3]
        err = torch.pow(err, exponent=2)
        err = torch.sum(err, dim=3)
        err = torch.pow(err, exponent=0.5)
        err = torch.sum(err, dim=2) / sequence_length
        err, inds = torch.min(err, dim=1)

        return err, inds
    
    def min_ahe(self, traj: torch.Tensor, traj_gt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes average heading error for the best trajectory is a set, with respect to ground truth
        :param traj: predictions, shape [batch_size, num_modes, sequence_length, 1]
        :param traj_gt: ground truth trajectory, shape [batch_size, sequence_length, 1]
        :param masks: masks for varying length ground truth, shape [batch_size, sequence_length]
        :return errs, inds: errors and indices for modes with min error, shape [batch_size]
        """
        num_modes = traj.shape[1]
        sequence_length = traj.shape[2]

        traj_gt_rpt = traj_gt.unsqueeze(1).repeat(1, num_modes, 1, 1)
        err = traj_gt_rpt[:, :, :, -1:] - traj[:, :, :, -1:]
        err = torch.pow(err, exponent=2)
        err = torch.sum(err, dim=3)
        err = torch.pow(err, exponent=0.5)
        err = torch.sum(err, dim=2) / sequence_length
        err, inds = torch.min(err, dim=1)

        return err, inds
        
    def compute(self, predictions: FeaturesType, targets: TargetsType, scenarios: ScenarioListType) -> torch.Tensor:
        # Unpack arguments
        traj = cast(TensorTarget, predictions['multimodal_outputs']).data
        log_probs = cast(TensorTarget, predictions['probs']).data
        traj_gt = cast(Trajectory, targets["trajectory"]).data    # predictions["target"] for closed loop, same for metrics

        # Useful variables
        batch_size = traj.shape[0]
        sequence_length = traj.shape[2]
        pred_params = 3

        # Obtain mode with minimum ADE with respect to ground truth:
        errs, inds = self.min_ade(traj, traj_gt) # [8]
        inds_rep = inds.repeat(sequence_length, pred_params, 1, 1).permute(3, 2, 0, 1)  # [8, 6, 16, 3]

        # Calculate MSE or NLL loss for trajectories corresponding to selected outputs:
        traj_best = traj.gather(1, inds_rep).squeeze(dim=1) # [8, 16, 3]

        l_reg = errs

        # Compute classification loss
        l_class = - torch.squeeze(log_probs.gather(1, inds.unsqueeze(1))) # [8]

        loss = self.beta * l_reg + self.alpha * l_class # [8]
        loss = torch.mean(loss) # []

        return loss