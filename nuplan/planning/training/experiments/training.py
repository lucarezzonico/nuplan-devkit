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
from torchviz import make_dot

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

    def save_visualize_info(self, path) -> None:
        """
        To save the related information to later visualize the model via TensorBoard.
        """
        print(self.model)

        # the way trying to visualize it through tensorboard failed. cuz it doesn't
        # support non-standard input format, which in this case, is [VectorMap, Agents]

        # writer = SummaryWriter(path)
        # trainloader=self.datamodule.train_dataloader()
        # dataiter = iter(trainloader)
        # datainput= next(dataiter)
        # # datainput:[{'vector_map': VectorMap(coords=[te...ing_dim=2), 'agents': Agents(ego=[tensor([...e+00]]])])}, {'trajectory': Trajectory(data=tens....3670]]]))}, [<nuplan.planning.sce...7e0131a30>]]
        # writer.add_graph(self.model, datainput[0])
        # writer.flush()
        # writer.close()

        # try to visualize through torchviz
        trainloader=self.datamodule.train_dataloader()
        dataiter = iter(trainloader)
        datainput= next(dataiter)
        
        # look at input outpu size
        ModelInput = datainput[0]
        ModelOutput = self.model(datainput[0])
        
        # datainput:[{'vector_map': VectorMap(coords=[te...ing_dim=2), 'agents': Agents(ego=[tensor([...e+00]]])])}, {'trajectory': Trajectory(data=tens....3670]]]))}, [<nuplan.planning.sce...7e0131a30>]]
        graph_obj=make_dot(self.model(datainput[0])['trajectory'].data, params=dict(self.model.named_parameters()), show_attrs=True, show_saved=True)
        graph_obj.render(path + "/model_vis/model_vis.dot", view=True)
        # .pdf as the same name is stored under the same directory. 
        print("Model graph rendered.")

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

    engine = TrainingEngine(trainer=trainer, datamodule=datamodule, model=model)

    return engine
