from typing import Dict, List, cast

import torch

from nuplan.planning.training.modeling.objectives.abstract_objective import AbstractObjective
from nuplan.planning.training.modeling.objectives.scenario_weight_utils import extract_scenario_type_weight
from nuplan.planning.training.modeling.types import FeaturesType, ScenarioListType, TargetsType
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory


class ImitationObjective(AbstractObjective):
    """
    Objective that drives the model to imitate the signals from expert behaviors/trajectories.
    """

    def __init__(self, scenario_type_loss_weighting: Dict[str, float], weight: float = 1.0):
        """
        Initializes the class

        :param name: name of the objective
        :param weight: weight contribution to the overall loss
        """
        self._name = 'imitation_objective'
        self._weight = weight
        self._fn_xy = torch.nn.modules.loss.MSELoss(reduction='none')
        self._fn_heading = torch.nn.modules.loss.L1Loss(reduction='none')
        self._fn_miss_rate = torch.nn.modules.loss.L1Loss(reduction='none')
        self._scenario_type_loss_weighting = scenario_type_loss_weighting

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

        :param predictions: model's predictions
        :param targets: ground truth targets from the dataset
        :return: loss scalar tensor
        """
        predicted_trajectory = cast(Trajectory, predictions["trajectory"])
        targets_trajectory = cast(Trajectory, targets["trajectory"])
        loss_weights = extract_scenario_type_weight(
            scenarios, self._scenario_type_loss_weighting, device=predicted_trajectory.xy.device
        )

        weighted_xy_loss = self._compute_xy_loss(predicted_trajectory, targets_trajectory, loss_weights)
        weighted_heading_loss = self._compute_heading_loss(predicted_trajectory, targets_trajectory, loss_weights)
        weighted_miss_rate_loss = self._compute_miss_rate_loss(predicted_trajectory, targets_trajectory, loss_weights)
        
        return self._weight * (torch.mean(weighted_xy_loss) +
                               torch.mean(weighted_heading_loss) +
                               torch.mean(weighted_miss_rate_loss))
    
    def _compute_xy_loss(self, predicted_trajectory: Trajectory, targets_trajectory: Trajectory, loss_weights: torch.Tensor) -> torch.Tensor:
        broadcast_shape_xy = tuple([-1] + [1 for _ in range(predicted_trajectory.xy.dim() - 1)])
        broadcasted_loss_weights_xy = loss_weights.view(broadcast_shape_xy)
        weighted_xy_loss = self._fn_xy(predicted_trajectory.xy, targets_trajectory.xy) * broadcasted_loss_weights_xy
        assert weighted_xy_loss.size() == predicted_trajectory.xy.size() # Assert that broadcasting was done correctly
        return weighted_xy_loss
    
    def _compute_heading_loss(self, predicted_trajectory: Trajectory, targets_trajectory: Trajectory, loss_weights: torch.Tensor) -> torch.Tensor:
        broadcast_shape_heading = tuple([-1] + [1 for _ in range(predicted_trajectory.heading.dim() - 1)])
        broadcasted_loss_weights_heading = loss_weights.view(broadcast_shape_heading)
        weighted_heading_loss = (self._fn_heading(predicted_trajectory.heading, targets_trajectory.heading) * broadcasted_loss_weights_heading)
        assert weighted_heading_loss.size() == predicted_trajectory.heading.size() # Assert that broadcasting was done correctly
        return weighted_heading_loss
    
    def _compute_miss_rate_loss(self, predicted_trajectory: Trajectory, targets_trajectory: Trajectory, loss_weights: torch.Tensor) -> torch.Tensor:
        max_displacement_threshold = 0.5
        predicted_trajectory.miss_rate = (torch.norm(predicted_trajectory.xy - targets_trajectory.xy, dim=-1) > max_displacement_threshold).bool().to(torch.float16)
        targets_trajectory.miss_rate = torch.zeros_like(predicted_trajectory.miss_rate)
        broadcast_shape_miss_rate = tuple([-1] + [1 for _ in range(predicted_trajectory.miss_rate.dim() - 1)])
        broadcasted_loss_weights_miss_rate = loss_weights.view(broadcast_shape_miss_rate)
        weighted_miss_rate_loss = self._fn_miss_rate(predicted_trajectory.miss_rate, targets_trajectory.miss_rate) * broadcasted_loss_weights_miss_rate
        assert weighted_miss_rate_loss.size() == predicted_trajectory.miss_rate.size() # Assert that broadcasting was done correctly
        return weighted_miss_rate_loss
