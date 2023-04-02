import os
from pathlib import Path
import hydra
from omegaconf import DictConfig
import tempfile
from nuplan.planning.script.run_training import main as main_train
from nuplan.planning.script.run_simulation import main as main_simulation
from nuplan.planning.script.run_nuboard import main as main_nuboard

def train(sim_dict: dict) -> str:
    # Location of path with all simulation configs
    CONFIG_PATH = sim_dict['CONFIG_PATH']+'/training'
    CONFIG_NAME = sim_dict['CONFIG_NAME']+'_training'
    
    # add save directory
    SAVE_DIR = sim_dict['SAVE_DIR']+'/training'
    # Name of the experiment
    EXPERIMENT = sim_dict['EXPERIMENT']
    MODEL = sim_dict['MODEL']
    TRAINING_MODEL = sim_dict['TRAINING_MODEL']
    LR_SCHEDULER = sim_dict['LR_SCHEDULER']
    
    # Training params
    PY_FUNC = sim_dict['PY_FUNC']
    SCENARIO_BUILDER = sim_dict['SCENARIO_BUILDER']
    SCENARIO_SELECTION = sim_dict['SCENARIO_SELECTION']
    MAX_EPOCHS = sim_dict['MAX_EPOCHS']
    BATCH_SIZE = sim_dict['BATCH_SIZE']
    
    LOG_DIR = str(Path(SAVE_DIR) / EXPERIMENT / MODEL)
    print('__LOG__' + LOG_DIR)

    # Initialize configuration management system
    hydra.core.global_hydra.GlobalHydra.instance().clear()  # reinitialize hydra if already initialized
    hydra.initialize(config_path=CONFIG_PATH)
    
    # Compose the configuration
    cfg = hydra.compose(config_name=CONFIG_NAME, overrides=[
        f'group={str(SAVE_DIR)}',
        f'cache.cache_path={str(SAVE_DIR)}/cache',
        f'experiment_name={EXPERIMENT}',
        f'job_name={MODEL}',
        f'py_func={PY_FUNC}',
        f'+training={TRAINING_MODEL}',  # raster model that consumes ego, agents and map raster layers and regresses the ego's trajectory
        f'lr_scheduler={LR_SCHEDULER}',
        f'scenario_builder={SCENARIO_BUILDER}',  # use nuplan mini database  # ['nuplan','nuplan_challenge','nuplan_mini']
        f'scenario_filter.limit_total_scenarios={SCENARIO_SELECTION}',  # Choose 500 scenarios to train with
        'lightning.trainer.params.accelerator=ddp_spawn',  # ddp is not allowed in interactive environment, using ddp_spawn instead - this can bottleneck the data pipeline, it is recommended to run training outside the notebook
        f'lightning.trainer.params.max_epochs={MAX_EPOCHS}',
        f'data_loader.params.batch_size={BATCH_SIZE}',
        'data_loader.params.num_workers=8',
    ])
    
    # Run the training loop, optionally inspect training artifacts through tensorboard (above cell)
    main_train(cfg)

def simulate(sim_dict: dict) -> str:
    # Location of path with all simulation configs
    CONFIG_PATH = sim_dict['CONFIG_PATH']+'/simulation'
    CONFIG_NAME = sim_dict['CONFIG_NAME']+'_simulation'

    # Select the planner and simulation challenge
    PLANNER = sim_dict['PLANNER']  # [simple_planner, ml_planner]
    CHALLENGE = sim_dict['CHALLENGE']  # [open_loop_boxes, closed_loop_nonreactive_agents, closed_loop_reactive_agents]
    SCENARIO_BUILDER = sim_dict['SCENARIO_BUILDER']
    DATASET_PARAMS = sim_dict['DATASET_PARAMS']
    
    # Name of the experiment
    EXPERIMENT = sim_dict['EXPERIMENT']
    
    # add save directory
    SAVE_DIR = sim_dict['SAVE_DIR']+'/simulation'
    
    simulation_folder = ''
    
    # Initialize configuration management system
    hydra.core.global_hydra.GlobalHydra.instance().clear()  # reinitialize hydra if already initialized
    hydra.initialize(config_path=CONFIG_PATH)
    
    if PLANNER == 'simple_planner':
        # Compose the configuration
        cfg = hydra.compose(config_name=CONFIG_NAME, overrides=[
            f'experiment_name={EXPERIMENT}',
            f'group={SAVE_DIR}',
            f'planner={PLANNER}',
            f'+simulation={CHALLENGE}',
            f'scenario_builder={SCENARIO_BUILDER}',
            *DATASET_PARAMS,
        ])
        
        # Run the simulation loop
        main_simulation(cfg)
    
        # Simulation folder for visualization in nuBoard
        simulation_folder = cfg.output_dir
        
    elif PLANNER == 'ml_planner':
        
        MODEL = sim_dict['MODEL']
        
        LOG_DIR = str(Path(sim_dict['SAVE_DIR']) / 'training' / EXPERIMENT / MODEL)
        print(LOG_DIR)

        # Get the checkpoint of the trained model
        # last_experiment = sorted(os.listdir(LOG_DIR))[-1]
        
        i = -1
        train_experiment_dir = sorted(Path(LOG_DIR).iterdir())[i]  # get last experiment
        while not (train_experiment_dir / 'checkpoints').exists():
            i -= 1
            train_experiment_dir = sorted(Path(LOG_DIR).iterdir())[i]  # get last experiment
            if i == -10: break
        checkpoint = sorted((train_experiment_dir / 'checkpoints').iterdir())[-1]  # get last checkpoint
        
        MODEL_PATH = str(checkpoint).replace("=", "\=")

        
        # Compose the configuration
        cfg = hydra.compose(config_name=CONFIG_NAME, overrides=[
            f'experiment_name={EXPERIMENT}',
            f'group={SAVE_DIR}',
            f'planner={PLANNER}',
            f'model={MODEL}',
            'planner.ml_planner.model_config=${model}',  # hydra notation to select model config
            f'planner.ml_planner.checkpoint_path={MODEL_PATH}',  # this path can be replaced by the checkpoint of the model trained in the previous section
            f'+simulation={CHALLENGE}',
            f'scenario_builder={SCENARIO_BUILDER}',
            *DATASET_PARAMS,
        ])
    
        # Run the simulation loop
        main_simulation(cfg)
    
        # Simulation folder for visualization in nuBoard
        simulation_folder = cfg.output_dir
    
    return simulation_folder


def open_nuboard(sim_dict: dict, simulation_folders: list[str]) -> None:
    # Location of path with all nuBoard configs
    CONFIG_PATH = sim_dict['CONFIG_PATH']+'/nuboard'
    CONFIG_NAME = sim_dict['CONFIG_NAME']+'_nuboard'

    # Initialize configuration management system
    hydra.core.global_hydra.GlobalHydra.instance().clear()  # reinitialize hydra if already initialized
    hydra.initialize(config_path=CONFIG_PATH)

    # Compose the configuration
    cfg = hydra.compose(config_name=CONFIG_NAME, overrides=[
        'scenario_builder=nuplan_mini',  # set the database (same as simulation) used to fetch data for visualization
        f'simulation_path={simulation_folders}',  # nuboard file path(s), if left empty the user can open the file inside nuBoard
    ])
    
    # Run nuBoard
    main_nuboard(cfg)
    
if __name__ == '__main__':
    train_sim_dicts = []
    # # Raster Model
    # train_sim_dicts.append(
    #     dict(
    #         ## TRAINING
    #         # Location of path with all simulation configs
    #         CONFIG_PATH = '../nuplan/planning/script/config',
    #         CONFIG_NAME = 'default',
            
    #         # Name of the experiment
    #         EXPERIMENT = 'raster_experiment',
    #         MODEL = 'raster_model',
    #         TRAINING_MODEL = 'training_raster_model',
            
    #         # Training params
    #         PY_FUNC = 'train', # ['train','test','cache']
    #         SCENARIO_BUILDER = 'nuplan_mini', # ['nuplan','nuplan_challenge','nuplan_mini']
    #         SCENARIO_SELECTION = 500,
    #         MAX_EPOCHS = 2,
    #         BATCH_SIZE = 8,
            
    #         # add save directory
    #         SAVE_DIR = os.getenv('NUPLAN_EXP_ROOT'),
            
    #         ## SIMULATION
    #         # Select the planner and simulation challenge
    #         PLANNER = 'ml_planner',  # [simple_planner, ml_planner]
    #         CHALLENGE = 'open_loop_boxes',  # [open_loop_boxes, closed_loop_nonreactive_agents, closed_loop_reactive_agents]
    #         DATASET_PARAMS = [
    #             'scenario_filter=all_scenarios',  # initially select all scenarios in the database
    #             'scenario_filter.scenario_types=[near_multiple_vehicles, on_pickup_dropoff, starting_unprotected_cross_turn, high_magnitude_jerk]',  # select scenario types
    #             'scenario_filter.num_scenarios_per_type=10',  # use 10 scenarios per scenario type
    #         ],
    #     )
    # )
    # # Simple Vector Model
    # train_sim_dicts.append(
    #     dict(
    #         ## TRAINING
    #         # Location of path with all simulation configs
    #         CONFIG_PATH = '../nuplan/planning/script/config',
    #         CONFIG_NAME = 'default',
            
    #         # Name of the experiment
    #         EXPERIMENT = 'simple_vector_experiment',
    #         MODEL = 'simple_vector_model',
    #         TRAINING_MODEL = 'training_simple_vector_model',
            
    #         # Training params
    #         PY_FUNC = 'train', # ['train','test','cache']
    #         SCENARIO_BUILDER = 'nuplan_mini', # ['nuplan','nuplan_challenge','nuplan_mini']
    #         SCENARIO_SELECTION = 500,
    #         MAX_EPOCHS = 10,
    #         BATCH_SIZE = 8,
            
    #         # add save directory
    #         SAVE_DIR = os.getenv('NUPLAN_EXP_ROOT'),
            
    #         ## SIMULATION
    #         # Select the planner and simulation challenge
    #         PLANNER = 'ml_planner',  # [simple_planner, ml_planner]
    #         CHALLENGE = 'open_loop_boxes',  # [open_loop_boxes, closed_loop_nonreactive_agents, closed_loop_reactive_agents]
    #         DATASET_PARAMS = [
    #             'scenario_filter=all_scenarios',  # initially select all scenarios in the database
    #             'scenario_filter.scenario_types=[near_multiple_vehicles, on_pickup_dropoff, starting_unprotected_cross_turn, high_magnitude_jerk]',  # select scenario types
    #             'scenario_filter.num_scenarios_per_type=10',  # use 10 scenarios per scenario type
    #         ],
    #     )
    # )
    # # Vector Model
    train_sim_dicts.append(
        dict(
            ## TRAINING
            # Location of path with all simulation configs
            CONFIG_PATH = '../nuplan/planning/script/config',
            CONFIG_NAME = 'default',
            
            # Name of the experiment
            EXPERIMENT = 'vector_experiment',
            MODEL = 'vector_model',
            TRAINING_MODEL = 'training_vector_model',
            LR_SCHEDULER = 'multistep_lr',
            
            # Training params
            PY_FUNC = 'train', # ['train','test','cache']
            SCENARIO_BUILDER = 'nuplan', # ['nuplan','nuplan_challenge','nuplan_mini']
            SCENARIO_SELECTION = 32000,
            MAX_EPOCHS = 36,
            BATCH_SIZE = 8,
            
            # add save directory
            SAVE_DIR = os.getenv('NUPLAN_EXP_ROOT'),
            
            ## SIMULATION
            # Select the planner and simulation challenge
            PLANNER = 'ml_planner',  # [simple_planner, ml_planner]
            CHALLENGE = 'open_loop_boxes',  # [open_loop_boxes, closed_loop_nonreactive_agents, closed_loop_reactive_agents]
            DATASET_PARAMS = [
                'scenario_filter=all_scenarios',  # initially select all scenarios in the database
                'scenario_filter.scenario_types=[near_multiple_vehicles, on_pickup_dropoff, starting_unprotected_cross_turn, high_magnitude_jerk]',  # select scenario types
                'scenario_filter.num_scenarios_per_type=10',  # use 10 scenarios per scenario type
            ],
        )
    )
    
    simulation_folders = []
    for train_sim_dict in train_sim_dicts:
        train(train_sim_dict)
        simulation_folder = simulate(train_sim_dict)
        simulation_folders.append(simulation_folder)
    # open_nuboard(train_sim_dict, simulation_folder)