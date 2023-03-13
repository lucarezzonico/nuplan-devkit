import os
from pathlib import Path
import hydra
from nuplan.planning.script.run_training import main as main_train
from omegaconf import DictConfig
import tempfile


def visualize(sim_dict: dict) -> str:
    # Location of path with all simulation configs
    CONFIG_PATH = sim_dict['CONFIG_PATH']
    CONFIG_NAME = sim_dict['CONFIG_NAME']
    
    # add save directory
    SAVE_DIR = sim_dict['SAVE_DIR']
    # Name of the experiment
    EXPERIMENT = sim_dict['EXPERIMENT']
    JOB_NAME = sim_dict['JOB_NAME']
    TRAINING_MODEL = sim_dict['TRAINING_MODEL']
    
    # Training params
    PY_FUNC = sim_dict['PY_FUNC']
    SCENARIO_BUILDER = sim_dict['SCENARIO_BUILDER']
    SCENARIO_SELECTION = sim_dict['SCENARIO_SELECTION']
    MAX_EPOCHS = sim_dict['MAX_EPOCHS']
    BATCH_SIZE = sim_dict['BATCH_SIZE']
    
    LOG_DIR = str(Path(SAVE_DIR) / EXPERIMENT / JOB_NAME)
    print('__LOG__' + LOG_DIR)

    # Initialize configuration management system
    hydra.core.global_hydra.GlobalHydra.instance().clear()  # reinitialize hydra if already initialized
    hydra.initialize(config_path=CONFIG_PATH)
    
    # Compose the configuration
    cfg = hydra.compose(config_name=CONFIG_NAME, overrides=[
        f'group={str(SAVE_DIR)}',
        f'cache.cache_path={str(SAVE_DIR)}/cache',
        f'experiment_name={EXPERIMENT}',
        f'job_name={JOB_NAME}',
        f'py_func={PY_FUNC}', # ['train','test','cache']
        f'+training={TRAINING_MODEL}',  # raster model that consumes ego, agents and map raster layers and regresses the ego's trajectory
        f'scenario_builder={SCENARIO_BUILDER}',  # use nuplan mini database  # ['nuplan','nuplan_challenge','nuplan_mini']
        f'scenario_filter.limit_total_scenarios={SCENARIO_SELECTION}',  # Choose 500 scenarios to train with
        'lightning.trainer.params.accelerator=ddp_spawn',  # ddp is not allowed in interactive environment, using ddp_spawn instead - this can bottleneck the data pipeline, it is recommended to run training outside the notebook
        f'lightning.trainer.params.max_epochs={MAX_EPOCHS}',
        f'data_loader.params.batch_size={BATCH_SIZE}',
        'data_loader.params.num_workers=8',
    ])
    
    # Run the training loop, optionally inspect training artifacts through tensorboard (above cell)
    engine=main_train(cfg)
    engine.save_visualize_info('/data1/nuplan/jiale/model_vis')
    print("done. ")
    
    
if __name__ == '__main__': 
    train_dicts = []
    # # Raster Model
    # train_dicts.append(
    #     dict(
    #         # Location of path with all simulation configs
    #         CONFIG_PATH = '../nuplan/planning/script/config/training',
    #         CONFIG_NAME = 'default_training',
        
    #         # Name of the experiment
    #         EXPERIMENT = 'raster_experiment',
    #         JOB_NAME = 'raster_model',
    #         TRAINING_MODEL = 'training_raster_model',
            
    #         # Training params
    #         PY_FUNC = 'train', # ['train','test','cache']
    #         SCENARIO_BUILDER = 'nuplan_mini', # ['nuplan','nuplan_challenge','nuplan_mini']
    #         SCENARIO_SELECTION = 500,
    #         MAX_EPOCHS = 10,
    #         BATCH_SIZE = 8,
            
    #         # add save directory
    #         SAVE_DIR = os.getenv('NUPLAN_EXP_ROOT')
    #     )
    # )
    # # Simple Vector Model
    # train_dicts.append(
    #     dict(
    #         # Location of path with all simulation configs
    #         CONFIG_PATH = '../nuplan/planning/script/config/training',
    #         CONFIG_NAME = 'default_training',
        
    #         # Name of the experiment
    #         EXPERIMENT = 'simple_vector_experiment',
    #         JOB_NAME = 'simple_vector_model',
    #         TRAINING_MODEL = 'training_simple_vector_model',
            
    #         # Training params
    #         PY_FUNC = 'train', # ['train','test','cache']
    #         SCENARIO_BUILDER = 'nuplan_mini', # ['nuplan','nuplan_challenge','nuplan_mini']
    #         SCENARIO_SELECTION = 500,
    #         MAX_EPOCHS = 10,
    #         BATCH_SIZE = 8,
            
    #         # add save directory
    #         SAVE_DIR = os.getenv('NUPLAN_EXP_ROOT')
    #     )
    # )
    # # Vector Model
    train_dicts.append(
        dict(
            # Location of path with all simulation configs
            CONFIG_PATH = '../nuplan/planning/script/config/training',
            CONFIG_NAME = 'default_training',
        
            # Name of the experiment
            EXPERIMENT = 'vector_experiment',
            JOB_NAME = 'vector_model',
            TRAINING_MODEL = 'training_vector_model',
            
            # Training params
            PY_FUNC = 'train', # ['train','test','cache']
            SCENARIO_BUILDER = 'nuplan_mini', # ['nuplan','nuplan_challenge','nuplan_mini']
            SCENARIO_SELECTION = 20,
            MAX_EPOCHS = 1,
            BATCH_SIZE = 1,
            
            # add save directory
            SAVE_DIR = os.getenv('NUPLAN_EXP_ROOT')
        )
    )

    # train_dicts.append(
    #     dict(
    #         # Location of path with all simulation configs
    #         CONFIG_PATH = '../nuplan/planning/script/config/training',
    #         CONFIG_NAME = 'default_training',
        
    #         # Name of the experiment
    #         EXPERIMENT = 'vector_experiment',
    #         JOB_NAME = 'vector_model',
    #         TRAINING_MODEL = 'training_vector_model',
            
    #         # Training params
    #         PY_FUNC = 'train', # ['train','test','cache']
    #         SCENARIO_BUILDER = 'nuplan', # ['nuplan','nuplan_challenge','nuplan_mini']
    #         SCENARIO_SELECTION = 200000, # paper: 0.2M 
    #         MAX_EPOCHS = 32,
    #         BATCH_SIZE = 8, # paper:128
            
    #         # add save directory
    #         SAVE_DIR = os.getenv('NUPLAN_EXP_ROOT')
    #     )
    # )

    for train_dict in train_dicts:
        visualize(train_dict)