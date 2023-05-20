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

        self.train_dataloader: Optional[torch.utils.data.DataLoader] = None
        self.val_dataloader: Optional[torch.utils.data.DataLoader] = None

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
            image_batch = self._get_images_from_vector_features(features, targets, predictions, pl_module.name())
            image_batch_multimodal = self._get_images_from_vector_features_multimodal(features, targets, predictions, pl_module.name(),
                                                                                      all_agents=True, all_trajectories=False)
            # expert_image_batch = self._get_expert_images_from_vector_features(features, targets, predictions, pl_module.name())
        else:
            return

        tag = f'{prefix}_visualization_{batch_idx}'
        
        self._save_images(torch.from_numpy(image_batch), tag, training_step, pl_module)
        self._save_images(torch.from_numpy(image_batch_multimodal), "multimodal_"+tag, training_step, pl_module)
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
        
        if model_name == 'UrbanDriverClosedLoopModel':
            target_traj = targets['trajectory'].unpack()
            predicted_traj = predictions['trajectory'].unpack()
        elif model_name == 'SafePathNetModel':
            target_traj = targets['trajectory'].unpack()
            predicted_traj = predictions["trajectories"].trajectories[0].unpack()
        else:
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
            )

            images.append(image)

        return np.asarray(images)
    
    def _get_images_from_vector_features_multimodal(
        self, features: FeaturesType, targets: TargetsType, predictions: TargetsType, model_name: str, all_agents=True, all_trajectories=True
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
        if model_name == 'UrbanDriverClosedLoopModel':
            target_traj = targets['trajectory'].unpack()
            predicted_trajs = predictions['trajectory'].unpack()
        elif model_name == 'SafePathNetModel':
            target_traj = targets['trajectory'].unpack()
            # predicted_trajs = predictions["trajectories"].unpack()
            predicted_tensors = list(predictions["all_pred_agents"].data.chunk(predictions["all_pred_agents"].data.size(0), dim=0))
            predicted_trajs = [self.compute_trajectories(batch_tensors.squeeze(dim=0), all_agents=all_agents, all_trajectories=all_trajectories)
                               for batch_tensors in predicted_tensors]
            
            # pred_multimodal_trajectories = Trajectories([t for traj in pred_multimodal_trajectories.trajectories for t in traj.unpack() for i in t.unpack()])
        elif model_name == 'AutoBotEgo':
            target_traj = targets['trajectory'].unpack()
            
            # _, sorted_indices = torch.sort(predictions["mode_probs"].data, dim=1)
            # # for each batch, pick the trajectory with largest probability
            # trajs=torch.stack([predictions["pred"].data[sorted_indices[i],:,i,:] for i in range(predictions["pred"].data.shape[2])])
            # trajs_3=trajs[:,:,:,:3]
            # trajs_3[:,:,:,-1] = 0
            
            # ego_trajs = list(trajs_3.chunk(trajs_3.size(1), dim=1))
            # predicted_trajs = [Trajectory(data=pred[:, 0]).unpack() for pred in ego_trajs] # ego trajs
            
            predicted_trajs = predictions["trajectories"].unpack()
            
        else:
            target_traj = targets['trajectory'].unpack()
            for traj in predictions['trajectories']:
                predicted_trajs.append(traj.unpack())

        for vector_map, agents, target_trajectory, predicted_trajectories in zip(
            features[vector_map_feature].unpack(),
            features[agents_feature].unpack(),
            target_traj,
            predicted_trajs,
        ):
            # if model_name == 'SafePathNetModel':
            #     predicted_trajectories = Trajectories([Trajectory(predicted_trajectories.squeeze(dim=0)[a,b,:,:].unsqueeze(dim=0))
            #                                            for a in range(predicted_trajectories.squeeze(dim=0).shape[0])
            #                                            for b in range(predicted_trajectories.squeeze(dim=0).shape[1])])

            image = get_raster_from_vector_map_with_agents_multimodal(
                vector_map,
                agents,
                target_trajectory,
                predicted_trajectories,
                pixel_size=self.pixel_size,
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
        all_agents: bool = False,
        all_trajectories: bool = False,
    ) -> Trajectories:
        _, _, num_points, num_features = predicted_batch.size()
        
        # create agents and mutimodal trajectories mask
        agent_mask = torch.ones_like(predicted_batch)
        if not all_agents: agent_mask[1:, :, :, :] *= 0
        traj_mask = torch.ones_like(predicted_batch)
        if not all_trajectories: traj_mask[:, 1:, :, :] *= 0
        agent_traj_mask = torch.mul(agent_mask, traj_mask)
        agent_traj_mask_bool = agent_traj_mask.bool()
        
        trajs = predicted_batch[agent_traj_mask_bool].view(-1, num_points, num_features)
        
        pred_trajs = Trajectories(Trajectory(trajs).unpack())
        
        return pred_trajs