import random
from typing import Any, List, Optional, cast

import numpy as np
import numpy.typing as npt
import pytorch_lightning as pl
import torch
import torch.utils.data

from nuplan.planning.training.callbacks.utils.visualization_utils import (
    get_raster_from_vector_map_with_agents,
    get_raster_from_vector_map_with_agents_multimodal,
    get_raster_with_trajectories_as_rgb,
)
from nuplan.planning.training.modeling.types import FeaturesType, TargetsType, move_features_type_to_device
from nuplan.planning.training.preprocessing.feature_collate import FeatureCollate

from torchvision.utils import save_image
import os
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory
from nuplan.planning.training.preprocessing.features.trajectories import Trajectories

class VisualizationCallback(pl.Callback):
    """
    Callback that visualizes planner model inputs/outputs and logs them in Tensorboard.
    """

    def __init__(
        self,
        images_per_tile: int,
        num_train_tiles: int,
        num_val_tiles: int,
        pixel_size: float,
        all_agents: bool,
        all_modes: bool,
    ):
        """
        Initialize the class.

        :param images_per_tile: number of images per tiles to visualize
        :param num_train_tiles: number of tiles from the training set
        :param num_val_tiles: number of tiles from the validation set
        :param pixel_size: [m] size of pixel in meters
        """
        super().__init__()

        self.custom_batch_size = images_per_tile
        self.num_train_images = num_train_tiles * images_per_tile
        self.num_val_images = num_val_tiles * images_per_tile
        self.pixel_size = pixel_size
        self.all_agents = all_agents
        self.all_modes = all_modes

        self.train_dataloader: Optional[torch.utils.data.DataLoader] = None
        self.val_dataloader: Optional[torch.utils.data.DataLoader] = None
        
        self.singleagent_multimodal_models: list[str] = ["AutoBotEgo", "UrbanDriverOpenLoopMultimodal"]
        self.multiagents_multimodal_models: list[str] = ["SafePathNetModel"]

    def _initialize_dataloaders(self, datamodule: pl.LightningDataModule) -> None:
        """
        Initialize the dataloaders. This makes sure that the same examples are sampled
        every time for comparison during visualization.

        :param datamodule: lightning datamodule
        """
        train_set = datamodule.train_dataloader().dataset
        val_set = datamodule.val_dataloader().dataset

        self.train_dataloader = self._create_dataloader(train_set, self.num_train_images)
        self.val_dataloader = self._create_dataloader(val_set, self.num_val_images)

    def _create_dataloader(self, dataset: torch.utils.data.Dataset, num_samples: int) -> torch.utils.data.DataLoader:
        dataset_size = len(dataset)
        num_keep = min(dataset_size, num_samples)
        sampled_idxs = random.sample(range(dataset_size), num_keep)
        subset = torch.utils.data.Subset(dataset=dataset, indices=sampled_idxs)
        return torch.utils.data.DataLoader(
            dataset=subset, batch_size=self.custom_batch_size, collate_fn=FeatureCollate()
        )

    def _log_from_dataloader(
        self,
        pl_module: pl.LightningModule,
        dataloader: torch.utils.data.DataLoader,
        loggers: List[Any],
        training_step: int,
        prefix: str,
    ) -> None:
        """
        Visualizes and logs all examples from the input dataloader.

        :param pl_module: lightning module used for inference
        :param dataloader: torch dataloader
        :param loggers: list of loggers from the trainer
        :param training_step: global step in training
        :param prefix: prefix to add to the log tag
        """
        for batch_idx, batch in enumerate(dataloader):
            features: FeaturesType = batch[0]
            targets: TargetsType = batch[1]
            predictions = self._infer_model(pl_module, move_features_type_to_device(features, pl_module.device))

            self._log_batch(loggers, features, targets, predictions, batch_idx, training_step, prefix, pl_module)

    def _log_batch(
        self,
        loggers: List[Any],
        features: FeaturesType,
        targets: TargetsType,
        predictions: TargetsType,
        batch_idx: int,
        training_step: int,
        prefix: str,
        pl_module: pl.LightningModule,
    ) -> None:
        """
        Visualizes and logs a batch of data (features, targets, predictions) from the model.

        :param loggers: list of loggers from the trainer
        :param features: tensor of model features
        :param targets: tensor of model targets
        :param predictions: tensor of model predictions
        :param batch_idx: index of total batches to visualize
        :param training_step: global training step
        :param prefix: prefix to add to the log tag
        :param pl_module: lightning module used for inference
        """
        # if 'trajectory' not in targets or 'trajectory' not in predictions:
        #     return

        if 'raster' in features:
            image_batch = self._get_images_from_raster_features(features, targets, predictions)
        elif ('vector_map' in features or 'vector_set_map' in features) and (
            'agents' in features or 'generic_agents' in features
        ):
            # image_batch = self._get_images_from_vector_features(features, targets, predictions, pl_module.name())
            
            # if pl_module.name() in self.singleagent_multimodal_models:
            image_batch = self._get_images_from_vector_features_multimodal(features, targets, predictions, pl_module.name(),
                                                                           all_agents=self.all_agents, all_modes=self.all_modes)
            # expert_image_batch = self._get_expert_images_from_vector_features(features, targets, predictions, pl_module.name())
        else:
            return

        tag = f'{prefix}_visualization_{batch_idx}'
        
        # save images to do gifs
        self._save_images(torch.from_numpy(image_batch), tag, training_step, pl_module)
        
        # if pl_module.name() in self.singleagent_multimodal_models:
        #     self._save_images(torch.from_numpy(image_batch_multimodal), "multimodal_"+tag, training_step, pl_module)
        # self._save_images(torch.from_numpy(expert_image_batch), "expert_"+tag, training_step, pl_module)

        for logger in loggers:
            if isinstance(logger, torch.utils.tensorboard.writer.SummaryWriter):
                logger.add_images(
                    tag=tag,
                    img_tensor=torch.from_numpy(image_batch),
                    global_step=training_step,
                    dataformats='NHWC',
                )
                
    def _save_images(
        self, image_batch: torch.Tensor, tag: str, training_step: int, pl_module: pl.LightningModule
    ) -> None:
        """
        Visualizes and logs a batch of data (features, targets, predictions) from the model.

        :param image_batch: tensor of images
        :param tag: tag for folder name to save images
        :param training_step: global training step
        :param pl_module: lightning module used for inference
        """
        image_batch = torch.permute(image_batch, (0, 3, 1, 2))
        exp_root = os.getenv('NUPLAN_EXP_ROOT')
        path = f'{pl_module.get_checkpoint_dir()}/training_visualization/{tag}'      #TODO:save images to the current training folder
        if not os.path.exists(path): os.makedirs(path, exist_ok=True)
        save_image(image_batch.float()/255.0, f'{path}/step_{training_step}.png')

    def _get_images_from_raster_features(
        self, features: FeaturesType, targets: TargetsType, predictions: TargetsType
    ) -> npt.NDArray[np.uint8]:
        """
        Create a list of RGB raster images from a batch of model data of raster features.

        :param features: tensor of model features
        :param targets: tensor of model targets
        :param predictions: tensor of model predictions
        :return: list of raster images
        """
        images = list()

        for raster, target_trajectory, predicted_trajectory in zip(
            features['raster'].unpack(), targets['trajectory'].unpack(), predictions['trajectory'].unpack()
        ):
            image = get_raster_with_trajectories_as_rgb(
                raster,
                target_trajectory,
                predicted_trajectory,
                pixel_size=self.pixel_size,
            )

            images.append(image)

        return np.asarray(images)

    def _get_images_from_vector_features(
        self, features: FeaturesType, targets: TargetsType, predictions: TargetsType, model_name: str
    ) -> npt.NDArray[np.uint8]:
        """
        Create a list of RGB raster images from a batch of model data of vectormap and agent features.

        :param features: tensor of model features
        :param targets: tensor of model targets
        :param predictions: tensor of model predictions
        :return: list of raster images
        """
        images = list()
        vector_map_feature = 'vector_map' if 'vector_map' in features else 'vector_set_map'
        agents_feature = 'agents' if 'agents' in features else 'generic_agents'
        
        target_traj = targets['trajectory'].unpack()
        predicted_traj = predictions['trajectory'].unpack()

        for vector_map, agents, target_trajectory, predicted_trajectory in zip(
            features[vector_map_feature].unpack(),
            features[agents_feature].unpack(),
            target_traj,
            predicted_traj,
        ):
            image = get_raster_from_vector_map_with_agents(
                vector_map,
                agents,
                target_trajectory,
                predicted_trajectory,
                pixel_size=self.pixel_size,
                vector_map_feature=vector_map_feature
            )

            images.append(image)

        return np.asarray(images)
    
    def _get_images_from_vector_features_multimodal(
        self, features: FeaturesType, targets: TargetsType, predictions: TargetsType, model_name: str, all_agents=True, all_modes=True
    ) -> npt.NDArray[np.uint8]:
        """
        Create a list of RGB raster images from a batch of model data of vectormap and agent features.

        :param features: tensor of model features
        :param targets: tensor of model targets
        :param predictions: tensor of model predictions
        :return: list of raster images
        """
        images = list()
        vector_map_feature = 'vector_map' if 'vector_map' in features else 'vector_set_map'
        agents_feature = 'agents' if 'agents' in features else 'generic_agents'
        
        predicted_trajs = []
        if model_name in self.multiagents_multimodal_models:
            target_traj = targets['trajectory'].unpack()
            # predicted_trajs = predictions["trajectories"].unpack()
            # predicted_tensors = list(predictions["past"].data[:,:,:,:5,:].chunk(predictions["past"].data[:,:,:,:5,:].size(0), dim=0)) # to plot all agents targets/pasts
            # predicted_tensors = list(predictions["future"].data.chunk(predictions["future"].data.size(0), dim=0)) # [8][50, 6, 16, 3]
            predicted_tensors = list(predictions["all_pred_agents"].data.chunk(predictions["all_pred_agents"].data.size(0), dim=0)) # [8][50, 6, 16, 3]
            # # list of 8 Trajectories objects containing 50 Trajectory objects of size (6,16,3): [[8]Trajectories([50]Trajectory(6,16,3))]
            predicted_trajs = [self.compute_trajectories(batch_tensors.squeeze(dim=0), model_name, all_agents=all_agents, all_modes=all_modes)
                               for batch_tensors in predicted_tensors]
            
        elif model_name in self.singleagent_multimodal_models:
            target_traj = targets['trajectory'].unpack()
            predicted_trajs = [self.compute_trajectories(pred_traj.data, model_name, all_modes=all_modes) for pred_traj in predictions["trajectories"].trajectories]
        
        else: # singleagent unimodal models
            target_traj = targets['trajectory'].unpack()
            predicted_trajs = Trajectories(predictions['trajectory'].unpack()).unpack()
        
        for vector_map, agents, target_trajectory, predicted_trajectories in zip(
            features[vector_map_feature].unpack(),
            features[agents_feature].unpack(),
            target_traj,
            predicted_trajs,
        ):
            image = get_raster_from_vector_map_with_agents_multimodal(
                vector_map,
                agents,
                target_trajectory,
                predicted_trajectories,
                pixel_size=self.pixel_size,
                vector_map_feature=vector_map_feature
            )

            images.append(image)

        return np.asarray(images)
    
    def _get_expert_images_from_vector_features(
        self, features: FeaturesType, targets: TargetsType, predictions: TargetsType, model_name: str
    ) -> npt.NDArray[np.uint8]:
        """
        Create a list of RGB raster images from a batch of model data of vectormap and agent features.

        :param features: tensor of model features
        :param targets: tensor of model targets
        :param predictions: tensor of model predictions
        :return: list of raster images
        """
        images = list()
        vector_map_feature = 'vector_map' if 'vector_map' in features else 'vector_set_map'
        agents_feature = 'agents' if 'agents' in features else 'generic_agents'
        expert_feature = 'expert' if 'expert' in features else 'generic_expert'
        
        expert_trajectory = Trajectory(data=torch.stack(features[expert_feature].ego)[:,:-1,:3])
        
        if model_name == 'UrbanDriverClosedLoopModel':
            target_traj = predictions['target'].unpack()
            predicted_traj = predictions['ts_traj'].unpack()
        else:
            target_traj = targets['trajectory'].unpack()
            predicted_traj = predictions['trajectory'].unpack()

        for vector_map, agents, target_trajectory, expert in zip(
            features[vector_map_feature].unpack(),
            features[agents_feature].unpack(),
            target_traj,
            predicted_traj,
        ):
            image = get_raster_from_vector_map_with_agents(
                vector_map,
                agents,
                target_trajectory,
                expert,
                pixel_size=self.pixel_size,
                vector_map_feature=vector_map_feature,
            )

            images.append(image)

        return np.asarray(images)

    def _infer_model(self, pl_module: pl.LightningModule, features: FeaturesType) -> TargetsType:
        """
        Make an inference of the input batch features given a model.

        :param pl_module: lightning model
        :param features: model inputs
        :return: model predictions
        """
        with torch.no_grad():
            pl_module.eval()
            predictions = move_features_type_to_device(pl_module(features), torch.device('cpu'))
            pl_module.train()

        return predictions

    def on_train_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        unused: Optional = None,  # type: ignore
    ) -> None:
        """
        Visualizes and logs training examples at the end of the epoch.

        :param trainer: lightning trainer
        :param pl_module: lightning module
        """
        assert hasattr(trainer, 'datamodule'), "Trainer missing datamodule attribute"
        assert hasattr(trainer, 'global_step'), "Trainer missing global_step attribute"

        if self.train_dataloader is None:
            self._initialize_dataloaders(trainer.datamodule)

        self._log_from_dataloader(
            pl_module,
            self.train_dataloader,
            trainer.logger.experiment,
            trainer.global_step,
            'train',
        )

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        unused: Optional = None,  # type: ignore
    ) -> None:
        """
        Visualizes and logs validation examples at the end of the epoch.

        :param trainer: lightning trainer
        :param pl_module: lightning module
        """
        assert hasattr(trainer, 'datamodule'), "Trainer missing datamodule attribute"
        assert hasattr(trainer, 'global_step'), "Trainer missing global_step attribute"

        if self.val_dataloader is None:
            self._initialize_dataloaders(trainer.datamodule)

        self._log_from_dataloader(
            pl_module,
            self.val_dataloader,
            trainer.logger.experiment,
            trainer.global_step,
            'val',
        )

    
    def compute_trajectories(
        self,
        predicted_batch: torch.Tensor,
        model_name: str,
        all_agents: bool = False,
        all_modes: bool = False,
    ) -> Trajectories:
        """
        Merge number of agents and number of multimodal trajectories dimensions into one.
        
        :param predicted_batch: tensor of shape (num_agents, num_modes, num_points, num_features) or (num_modes, num_points, num_features)
        :param all_agents: if all agents trajectories should be returned of just the ego's
        :param all_modes: if all mutilmodal trajectories should be returned of just the most likely
        :return: a trajectories object containing num_agents trajectory objects of size (num_modes, num_points, num_features)
        """
        
        if model_name in self.multiagents_multimodal_models:
            num_agents, num_modes, num_points, num_features = predicted_batch.size()
            # create agents and mutimodal trajectories mask
            agent_mask = torch.ones_like(predicted_batch)
            if not all_agents: agent_mask[1:, :, :, :] *= 0
            traj_mask = torch.ones_like(predicted_batch)
            if not all_modes: traj_mask[:, 1:, :, :] *= 0
            agent_traj_mask = torch.mul(agent_mask, traj_mask)
            agent_traj_mask_bool = agent_traj_mask.bool()
            trajs = predicted_batch[agent_traj_mask_bool].view(-1, num_modes if all_modes else 1, num_points, num_features) # (num_agents or 1, num_modes or 1, num_points, num_features)
            predicted_tensors = list(trajs.chunk(trajs.size(0), dim=0)) # [50][6, 16, 3]
            pred_trajs = Trajectories([Trajectory(agent_tensors.squeeze(dim=0)) for agent_tensors in predicted_tensors])
            # pred_trajs: a list of 50 trajectories, each containing a Trajectory(6, 16, 3)
        elif model_name in self.singleagent_multimodal_models:
            num_modes, num_points, num_features = predicted_batch.size()
            # create mutimodal trajectories mask
            traj_mask = torch.ones_like(predicted_batch)
            if not all_modes: traj_mask[1:, :, :] *= 0
            traj_mask_bool = traj_mask.bool()
            trajs = predicted_batch[traj_mask_bool].view(-1, num_points, num_features) # (num_modes or 1, num_points, num_features)
            pred_trajs = Trajectories([Trajectory(trajs)])
            # pred_trajs: a list of 50 trajectories, each containing a Trajectory(6, 16, 3)
        
        return pred_trajs