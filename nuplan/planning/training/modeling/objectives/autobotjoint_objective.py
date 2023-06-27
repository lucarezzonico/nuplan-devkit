
from typing import Dict, List, cast

import torch

from nuplan.planning.training.modeling.objectives.abstract_objective import AbstractObjective
from nuplan.planning.training.modeling.objectives.scenario_weight_utils import extract_scenario_type_weight
from nuplan.planning.training.modeling.types import FeaturesType, ScenarioListType, TargetsType
from nuplan.planning.training.preprocessing.features.tensor_target import TensorTarget
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory
from nuplan.planning.training.preprocessing.features.trajectories import Trajectories

from nuplan.planning.training.modeling.objectives.autobots_train_helpers import nll_loss_multimodes, nll_loss_multimodes_joint
from torch import Tensor

class AutobotsObjective(AbstractObjective):
    """
    Autobots ego objective
    """

    def __init__(self, scenario_type_loss_weighting: Dict[str, float], entropy_weight, kl_weight, use_FDEADE_aux_loss):
        """
        Initializes the class
        :param name: name of the objective
        :param weight: weight contribution to the overall loss
        """
        self._name = 'autobots_ego_objective'
        self.entropy_weight=entropy_weight
        self.kl_weight=kl_weight
        self.use_FDEADE_aux_loss=use_FDEADE_aux_loss
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

        pred_obs = cast(TensorTarget, predictions["pred"]).data
        mode_probs = cast(TensorTarget, predictions["mode_probs"]).data
        targets_xy = cast(Trajectory, targets["trajectory"]).data
        targets_xy_agents = cast(Trajectory, targets["agent_trajectories"]).data
        
        loss_weights = extract_scenario_type_weight(scenarios, self._scenario_type_loss_weighting, device=pred_obs.device) # [B]

        num_agents = pred_obs.shape[3]-1
        targets_xy_fake = targets_xy.unsqueeze(dim=2).repeat(1, 1, num_agents, 1)
        nll_loss, kl_loss, post_entropy, adefde_loss = nll_loss_multimodes_joint(pred_obs, targets_xy[:, :, :2], targets_xy_fake[:, :, :, :2], mode_probs,
                                                                                   entropy_weight=self.entropy_weight,
                                                                                   kl_weight=self.kl_weight,
                                                                                   use_FDEADE_aux_loss=self.use_FDEADE_aux_loss)
        
        total_loss=nll_loss + adefde_loss + kl_loss
        # how to implement the gradient clip?

        # nll_loss: 

        return total_loss