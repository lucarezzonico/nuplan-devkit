import logging
from dataclasses import dataclass

import pytorch_lightning as pl
from omegaconf import DictConfig

from nuplan.planning.script.builders.model_builder import build_torch_module_wrapper
from nuplan.planning.script.builders.training_builder import (
    build_lightning_datamodule,
    build_lightning_module,
    build_trainer,
)
from nuplan.planning.script.builders.utils.utils_config import scale_cfg_for_distributed_training
from nuplan.planning.utils.multithreading.worker_pool import WorkerPool

from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingEngine:
    """Lightning training engine dataclass wrapping the lightning trainer, model and datamodule."""

    trainer: pl.Trainer  # Trainer for models
    model: pl.LightningModule  # Module describing NN model, loss, metrics, visualization
    datamodule: pl.LightningDataModule  # Loading data

    def __repr__(self) -> str:
        """
        :return: String representation of class without expanding the fields.
        """
        return f"<{type(self).__module__}.{type(self).__qualname__} object at {hex(id(self))}>"


def build_training_engine(cfg: DictConfig, worker: WorkerPool) -> TrainingEngine:
    """
    Build the three core lightning modules: LightningDataModule, LightningModule and Trainer
    :param cfg: omegaconf dictionary
    :param worker: Worker to submit tasks which can be executed in parallel
    :return: TrainingEngine
    """
    logger.info('Building training engine...')

    # Create model
    torch_module_wrapper = build_torch_module_wrapper(cfg.model)

    # Build the datamodule
    datamodule = build_lightning_datamodule(cfg, worker, torch_module_wrapper)

    if cfg.lightning.trainer.params.accelerator == 'ddp':  # Update the learning rate parameters to suit ddp
        cfg = scale_cfg_for_distributed_training(cfg, datamodule=datamodule, worker=worker)
    else:
        logger.info(
            f'Updating configs based on {cfg.lightning.trainer.params.accelerator} strategy is currently not supported. Optimizer and LR Scheduler configs will not be updated.'
        )

    # Build lightning module
    model = build_lightning_module(cfg, torch_module_wrapper)

    # Build trainer
    trainer = build_trainer(cfg)
    datamodule = build_lightning_datamodule(cfg, worker, torch_module_wrapper)
    
    # Add Model Graph To Tensorboard
    # trainer.logger.experiment.add_graph(model(), datamodule.from_datasets)
    # trainer.logger.log_graph(model, datamodule) # datamodule should be of type FeaturesType
    # writer = SummaryWriter(cfg.output_dir)
    # writer.add_graph(model, datamodule)
    # writer.close()

    engine = TrainingEngine(trainer=trainer, datamodule=datamodule, model=model)

    return engine
