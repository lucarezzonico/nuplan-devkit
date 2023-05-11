import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from nuplan.planning.script.builders.lr_scheduler_builder import build_lr_scheduler
from nuplan.planning.training.modeling.metrics.planning_metrics import AbstractTrainingMetric
from nuplan.planning.training.modeling.objectives.abstract_objective import aggregate_objectives
from nuplan.planning.training.modeling.objectives.imitation_objective import AbstractObjective
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.modeling.types import FeaturesType, ScenarioListType, TargetsType

logger = logging.getLogger(__name__)


class LightningModuleWrapper(pl.LightningModule):
    """
    Lightning module that wraps the training/validation/testing procedure and handles the objective/metric computation.
    """

    def __init__(
        self,
        model: TorchModuleWrapper,
        objectives: List[AbstractObjective],
        metrics: List[AbstractTrainingMetric],
        batch_size: int,
        optimizer: Optional[DictConfig] = None,
        lr_scheduler: Optional[DictConfig] = None,
        warm_up_lr_scheduler: Optional[DictConfig] = None,
        objective_aggregate_mode: str = 'mean',
        nb_scenarios: int = 500,
        nb_epochs: int = 2,
    ) -> None:
        """
        Initializes the class.

        :param model: pytorch model
        :param objectives: list of learning objectives used for supervision at each step
        :param metrics: list of planning metrics computed at each step
        :param batch_size: batch_size taken from dataloader config
        :param optimizer: config for instantiating optimizer. Can be 'None' for older models.
        :param lr_scheduler: config for instantiating lr_scheduler. Can be 'None' for older models and when an lr_scheduler is not being used.
        :param warm_up_lr_scheduler: config for instantiating warm up lr scheduler. Can be 'None' for older models and when a warm up lr_scheduler is not being used.
        :param objective_aggregate_mode: how should different objectives be combined, can be 'sum', 'mean', and 'max'.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.objectives = objectives
        self.metrics = metrics
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.warm_up_lr_scheduler = warm_up_lr_scheduler
        self.objective_aggregate_mode = objective_aggregate_mode
        self.nb_scenarios = nb_scenarios
        self.nb_epochs = nb_epochs

        # Validate metrics objectives and model
        model_targets = {builder.get_feature_unique_name() for builder in model.get_list_of_computed_target()}
        for objective in self.objectives:
            for feature in objective.get_list_of_required_target_types():
                assert feature in model_targets, f"Objective target: \"{feature}\" is not in model computed targets!"
        for metric in self.metrics:
            for feature in metric.get_list_of_required_target_types():
                assert feature in model_targets, f"Metric target: \"{feature}\" is not in model computed targets!"

        self.list_loss : list = []
        self.count_epoch: int = -1
        self.count_train_step: int = 0
        self.count_train_step_total: int = 0
        self.count_val_step: int = 0
        self.count_val_step_total: int = 0
        self.previous_prefix: str = "val"

    def _step(self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], prefix: str) -> torch.Tensor:
        """
        Propagates the model forward and backwards and computes/logs losses and metrics.

        This is called either during training, validation or testing stage.

        :param batch: input batch consisting of features and targets
        :param prefix: prefix prepended at each artifact's name during logging
        :return: model's scalar loss
        """
        features, targets, scenarios = batch

        predictions = self.forward(features)
        objectives = self._compute_objectives(predictions, targets, scenarios)
        metrics = self._compute_metrics(predictions, targets)
        loss = aggregate_objectives(objectives, agg_mode=self.objective_aggregate_mode)

        self._log_step(loss, objectives, metrics, prefix)

        return loss

    def _compute_objectives(
        self, predictions: TargetsType, targets: TargetsType, scenarios: ScenarioListType
    ) -> Dict[str, torch.Tensor]:
        """
        Computes a set of learning objectives used for supervision given the model's predictions and targets.

        :param predictions: model's output signal
        :param targets: supervisory signal
        :return: dictionary of objective names and values
        """
        return {objective.name(): objective.compute(predictions, targets, scenarios) for objective in self.objectives}

    def _compute_metrics(self, predictions: TargetsType, targets: TargetsType) -> Dict[str, torch.Tensor]:
        """
        Computes a set of planning metrics given the model's predictions and targets.

        :param predictions: model's predictions
        :param targets: ground truth targets
        :return: dictionary of metrics names and values
        """
        return {metric.name(): metric.compute(predictions, targets) for metric in self.metrics}

    def _log_step(
        self,
        loss: torch.Tensor,
        objectives: Dict[str, torch.Tensor],
        metrics: Dict[str, torch.Tensor],
        prefix: str,
        loss_name: str = 'loss',
    ) -> None:
        """
        Logs the artifacts from a training/validation/test step.

        :param loss: scalar loss value
        :type objectives: [type]
        :param metrics: dictionary of metrics names and values
        :param prefix: prefix prepended at each artifact's name
        :param loss_name: name given to the loss for logging
        """
        self.log(f'loss/{prefix}_{loss_name}', loss)
        
        self.custom_epoch_loss_log(prefix, loss, loss_name)
            
        for key, value in objectives.items():
            self.log(f'objectives/{prefix}_{key}', value)

        for key, value in metrics.items():
            self.log(f'metrics/{prefix}_{key}', value)
            
    def custom_epoch_loss_log(self, prefix: str, loss: torch.Tensor, loss_name: str = 'loss') -> None:
        
        if prefix == 'train' and self.previous_prefix == 'val':
            self.previous_prefix = 'train'
            self.count_epoch += 1
            
        if prefix == 'train':
            self.previous_prefix = 'train'
            self.list_loss.append(loss) # keep track of loss for each step during one epoch
            
            self.count_val_step = 0
            self.count_train_step += 1
            self.count_train_step_total += 1
            logger.info(f'Training Epoch {self.count_epoch}, Loss: {loss}, Step: {self.count_train_step}, Total Steps: {self.count_train_step_total}')
        
        if prefix == 'val' and self.previous_prefix == 'train':
            self.previous_prefix = 'val'
            loss_epoch_last_iteration = self.list_loss[-1] # select loss of last train last iteration
            if not isinstance(self.list_loss, list): self.list_loss = [self.list_loss] # in case of only 1 iteration
            loss_epoch_mean = torch.mean(torch.stack(self.list_loss, dim=0), dim=0) # select mean loss of all iterations
            self.list_loss = []
            self.log(f'loss_epoch/train_{loss_name}_epoch_last_iteration', loss_epoch_last_iteration)
            self.log(f'loss_epoch/train_{loss_name}_epoch_mean', loss_epoch_mean)
            
        if prefix == 'val':
            self.previous_prefix = 'val'
            self.count_train_step = 0
            self.count_val_step += 1
            self.count_val_step_total += 1
            logger.info(f'Validation Epoch {self.count_epoch}, Loss: {loss}, Step: {self.count_val_step}, Total Steps: {self.count_val_step_total}')
        
    def training_step(self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], batch_idx: int) -> torch.Tensor:
        """
        Step called for each batch example during training.

        :param batch: example batch
        :param batch_idx: batch's index (unused)
        :return: model's loss tensor
        """
        return self._step(batch, 'train')

    def validation_step(
        self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], batch_idx: int
    ) -> torch.Tensor:
        """
        Step called for each batch example during validation.

        :param batch: example batch
        :param batch_idx: batch's index (unused)
        :return: model's loss tensor
        """
        return self._step(batch, 'val')

    def test_step(self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], batch_idx: int) -> torch.Tensor:
        """
        Step called for each batch example during testing.

        :param batch: example batch
        :param batch_idx: batch's index (unused)
        :return: model's loss tensor
        """
        return self._step(batch, 'test')

    def forward(self, features: FeaturesType) -> TargetsType:
        """
        Propagates a batch of features through the model.

        :param features: features batch
        :return: model's predictions
        """
        return self.model(features)

    def configure_optimizers(
        self,
    ) -> Union[Optimizer, Dict[str, Union[Optimizer, _LRScheduler]]]:
        """
        Configures the optimizers and learning schedules for the training.

        :return: optimizer or dictionary of optimizers and schedules
        """
        if self.optimizer is None:
            raise RuntimeError("To train, optimizer must not be None.")

        # Get optimizer
        optimizer: Optimizer = instantiate(
            config=self.optimizer,
            params=self.parameters(),
            lr=self.optimizer.lr,  # Use lr found from lr finder; otherwise use optimizer config
        )
        # Log the optimizer used
        logger.info(f'Using optimizer: {self.optimizer._target_}')

        # Get lr_scheduler
        lr_scheduler_params: Dict[str, Union[_LRScheduler, str, int]] = build_lr_scheduler(
            optimizer=optimizer,
            lr=self.optimizer.lr,
            warm_up_lr_scheduler_cfg=self.warm_up_lr_scheduler,
            lr_scheduler_cfg=self.lr_scheduler,
        )

        optimizer_dict: Dict[str, Any] = {}
        optimizer_dict['optimizer'] = optimizer
        if lr_scheduler_params:
            logger.info(f'Using lr_schedulers {lr_scheduler_params}')
            optimizer_dict['lr_scheduler'] = lr_scheduler_params

        return optimizer_dict if 'lr_scheduler' in optimizer_dict else optimizer_dict['optimizer']

    def name(self) -> str:
        """
        Returns the model's name.

        :return: model's name
        """
        return self.model.__class__.__name__
    
    def get_checkpoint_dir(self) -> str:
        """
        Returns the model's checkpoint directory.

        :return: model's checkpoint directory
        """
        return self.checkpoint_dir
    
    def set_checkpoint_dir(self, checkpoint_dir: str) -> None:
        """
        Sets the model's checkpoint directory.

        :param checkpoint_dir: model's checkpoint directory
        """
        self.checkpoint_dir = checkpoint_dir