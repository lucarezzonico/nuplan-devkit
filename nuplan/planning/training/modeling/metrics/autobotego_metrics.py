from typing import Dict, List, cast
import torch

from nuplan.planning.training.modeling.metrics.abstract_training_metric import AbstractTrainingMetric
from nuplan.planning.training.modeling.types import TargetsType
from nuplan.planning.training.preprocessing.features.tensor_target import TensorTarget
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory

from nuplan.planning.training.modeling.objectives.autobots_train_helpers import nll_loss_multimodes


class AutobotsNllLoss(AbstractTrainingMetric):
    """
    
    """

    def __init__(self, entropy_weight, kl_weight, use_FDEADE_aux_loss, predict_yaw):
        """
        Initializes the class
        :param name: name of the objective
        :param weight: weight contribution to the overall loss
        """
        self._name = 'nll_loss'
        self.entropy_weight=entropy_weight
        self.kl_weight=kl_weight
        self.use_FDEADE_aux_loss=use_FDEADE_aux_loss
        self.predict_yaw = predict_yaw

    def name(self) -> str:
        """
        Name of the objective
        """
        return self._name

    def get_list_of_required_target_types(self) -> List[str]:
        """Implemented. See interface."""
        return ["trajectory", "pred", "mode_probs"]

    def compute(self, predictions: TargetsType, targets: TargetsType) -> torch.Tensor:
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
        

        nll_loss, kl_loss, post_entropy, adefde_loss = nll_loss_multimodes(pred_obs, targets_xy, mode_probs,
                                                                            entropy_weight=self.entropy_weight,
                                                                            kl_weight=self.kl_weight,
                                                                            use_FDEADE_aux_loss=self.use_FDEADE_aux_loss,
                                                                            predict_yaw=self.predict_yaw)

        return nll_loss


class AutobotsKlLoss(AbstractTrainingMetric):
    """
    
    """

    def __init__(self, entropy_weight, kl_weight, use_FDEADE_aux_loss, predict_yaw):
        """
        Initializes the class
        :param name: name of the objective
        :param weight: weight contribution to the overall loss
        """
        self._name = 'kl_loss'
        self.entropy_weight=entropy_weight
        self.kl_weight=kl_weight
        self.use_FDEADE_aux_loss=use_FDEADE_aux_loss
        self.predict_yaw = predict_yaw

    def name(self) -> str:
        """
        Name of the objective
        """
        return self._name

    def get_list_of_required_target_types(self) -> List[str]:
        """Implemented. See interface."""
        return ["trajectory", "pred", "mode_probs"]

    def compute(self, predictions: TargetsType, targets: TargetsType) -> torch.Tensor:
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
        

        nnll_loss, kl_loss, post_entropy, adefde_loss = nll_loss_multimodes(pred_obs, targets_xy, mode_probs,
                                                                            entropy_weight=self.entropy_weight,
                                                                            kl_weight=self.kl_weight,
                                                                            use_FDEADE_aux_loss=self.use_FDEADE_aux_loss,
                                                                            predict_yaw=self.predict_yaw)

        return kl_loss

class AutobotsPostEntropy(AbstractTrainingMetric):
    """
    
    """

    def __init__(self, entropy_weight, kl_weight, use_FDEADE_aux_loss, predict_yaw):
        """
        Initializes the class
        :param name: name of the objective
        :param weight: weight contribution to the overall loss
        """
        self._name = 'post_entropy'
        self.entropy_weight=entropy_weight
        self.kl_weight=kl_weight
        self.use_FDEADE_aux_loss=use_FDEADE_aux_loss
        self.predict_yaw = predict_yaw

    def name(self) -> str:
        """
        Name of the objective
        """
        return self._name

    def get_list_of_required_target_types(self) -> List[str]:
        """Implemented. See interface."""
        return ["trajectory", "pred", "mode_probs"]

    def compute(self, predictions: TargetsType, targets: TargetsType) -> torch.Tensor:
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
        

        nll_loss, kl_loss, post_entropy, adefde_loss = nll_loss_multimodes(pred_obs, targets_xy, mode_probs,
                                                                            entropy_weight=self.entropy_weight,
                                                                            kl_weight=self.kl_weight,
                                                                            use_FDEADE_aux_loss=self.use_FDEADE_aux_loss,
                                                                            predict_yaw=self.predict_yaw)

        return post_entropy

class AutobotsADEFDELoss(AbstractTrainingMetric):
    """
    
    """

    def __init__(self, entropy_weight, kl_weight, use_FDEADE_aux_loss, predict_yaw):
        """
        Initializes the class
        :param name: name of the objective
        :param weight: weight contribution to the overall loss
        """
        self._name = 'ade_fde_loss'
        self.entropy_weight=entropy_weight
        self.kl_weight=kl_weight
        self.use_FDEADE_aux_loss=use_FDEADE_aux_loss
        self.predict_yaw=predict_yaw
        self.predict_yaw = predict_yaw

    def name(self) -> str:
        """
        Name of the objective
        """
        return self._name

    def get_list_of_required_target_types(self) -> List[str]:
        """Implemented. See interface."""
        return ["trajectory", "pred", "mode_probs"]

    def compute(self, predictions: TargetsType, targets: TargetsType) -> torch.Tensor:
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
        

        nll_loss, kl_loss, post_entropy, adefde_loss = nll_loss_multimodes(pred_obs, targets_xy, mode_probs,
                                                                            entropy_weight=self.entropy_weight,
                                                                            kl_weight=self.kl_weight,
                                                                            use_FDEADE_aux_loss=self.use_FDEADE_aux_loss,
                                                                            predict_yaw=self.predict_yaw)

        return adefde_loss

