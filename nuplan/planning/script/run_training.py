import logging
import os
from typing import Optional

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig

from nuplan.planning.script.builders.folder_builder import build_training_experiment_folder
from nuplan.planning.script.builders.logging_builder import build_logger
from nuplan.planning.script.builders.utils.utils_config import update_config_for_training
from nuplan.planning.script.builders.worker_pool_builder import build_worker
from nuplan.planning.script.profiler_context_manager import ProfilerContextManager
from nuplan.planning.script.utils import set_default_path
from nuplan.planning.training.experiments.caching import cache_data
from nuplan.planning.training.experiments.training import TrainingEngine, build_training_engine

logging.getLogger('numba').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# If set, use the env. variable to overwrite the default dataset and experiment paths
set_default_path()

# If set, use the env. variable to overwrite the Hydra config
CONFIG_PATH = os.getenv('NUPLAN_HYDRA_CONFIG_PATH', 'config/training')

if os.environ.get('NUPLAN_HYDRA_CONFIG_PATH') is not None:
    CONFIG_PATH = os.path.join('../../../../', CONFIG_PATH)

if os.path.basename(CONFIG_PATH) != 'training':
    CONFIG_PATH = os.path.join(CONFIG_PATH, 'training')
CONFIG_NAME = 'default_training'


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> Optional[TrainingEngine]:
    """
    Main entrypoint for training/validation experiments.
    :param cfg: omegaconf dictionary
    """
    # Fix random seed
    pl.seed_everything(cfg.seed, workers=True)

    # Configure logger
    build_logger(cfg)

    # Override configs based on setup, and print config
    update_config_for_training(cfg)

    # Create output storage folder
    build_training_experiment_folder(cfg=cfg)

    # Build worker
    worker = build_worker(cfg)

    if cfg.py_func == 'train':
        # Build training engine
        with ProfilerContextManager(cfg.output_dir, cfg.enable_profiling, "build_training_engine"):
            engine = build_training_engine(cfg, worker)

        # Run training
        logger.info('Starting training...')
        with ProfilerContextManager(cfg.output_dir, cfg.enable_profiling, "training"):
            engine.trainer.fit(model=engine.model, datamodule=engine.datamodule)
        return engine
    elif cfg.py_func == 'test':
        # Build training engine
        with ProfilerContextManager(cfg.output_dir, cfg.enable_profiling, "build_training_engine"):
            engine = build_training_engine(cfg, worker)

        # Test model
        logger.info('Starting testing...')
        with ProfilerContextManager(cfg.output_dir, cfg.enable_profiling, "testing"):
            engine.trainer.test(model=engine.model, datamodule=engine.datamodule)
        return engine
    elif cfg.py_func == 'cache':
        # Precompute and cache all features
        logger.info('Starting caching...')
        with ProfilerContextManager(cfg.output_dir, cfg.enable_profiling, "caching"):
            cache_data(cfg=cfg, worker=worker)
        return None
    else:
        raise NameError(f'Function {cfg.py_func} does not exist')


if __name__ == '__main__':
    cfg = dict(
        # Location of path with all simulation configs
        CONFIG_PATH = '../nuplan/planning/script/config/training',
        CONFIG_NAME = 'default_training',

        # Name of the experiment
        EXPERIMENT = 'vector_experiment',
        JOB_NAME = 'vector_model',
        TRAINING_MODEL = 'training_vector_model',

        # Training params
        PY_FUNC = 'train', # ['train','test','cache']
        SCENARIO_BUILDER = 'nuplan', # ['nuplan','nuplan_challenge','nuplan_mini']
        SCENARIO_SELECTION = 500,
        MAX_EPOCHS = 2,
        BATCH_SIZE = 8,

        # add save directory
        SAVE_DIR = os.getenv('NUPLAN_EXP_ROOT')
    )
    hydra.core.global_hydra.GlobalHydra.instance().clear()  # reinitialize hydra if already initialized
    hydra.initialize(config_path='../nuplan/planning/script/config/training')
    
    cfg = hydra.compose(config_name=cfg["CONFIG_NAME"], overrides=[
        f'group={str(cfg.SAVE_DIR)}/training',
        f'cache.cache_path={str(cfg.SAVE_DIR)}/cache',
        f'experiment_name={cfg.EXPERIMENT}',
        f'job_name={cfg.JOB_NAME}',
        f'py_func={cfg.PY_FUNC}',
        f'+training={cfg.TRAINING_MODEL}',  # raster model that consumes ego, agents and map raster layers and regresses the ego's trajectory
        f'scenario_builder={cfg.SCENARIO_BUILDER}',  # use nuplan mini database  # ['nuplan','nuplan_challenge','nuplan_mini']
        f'scenario_filter.limit_total_scenarios={cfg.SCENARIO_SELECTION}',  # Choose 500 scenarios to train with
        f'lightning.trainer.params.accelerator={cfg.lightning_accelerator}',  # ddp is not allowed in interactive environment, using ddp_spawn instead - this can bottleneck the data pipeline, it is recommended to run training outside the notebook
        f'lightning.trainer.params.max_epochs={cfg.MAX_EPOCHS}',
        f'data_loader.params.batch_size={cfg.BATCH_SIZE}',
        f'data_loader.params.num_workers=8',
    ])
    
    main(cfg)
