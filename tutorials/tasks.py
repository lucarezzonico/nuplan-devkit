import os
from pathlib import Path
import hydra
from omegaconf import DictConfig
import tempfile
from nuplan.planning.script.run_training import main as main_train
from nuplan.planning.script.run_simulation import main as main_simulation
from nuplan.planning.script.run_nuboard import main as main_nuboard
import yaml
import glob
from typing import List, Optional, cast
import shutil
from PIL import Image

from nuplan.planning.script.utils import set_default_path

# If set, use the env. variable to overwrite the default dataset and experiment paths
set_default_path()

# If set, use the env. variable to overwrite the Hydra config
CONFIG_PATH = '../nuplan/planning/script'
CONFIG_NAME = 'default_config'


def train(cfg: DictConfig) -> None:
    # add save directory
    save_dir_training = os.getenv('NUPLAN_EXP_ROOT') + '/training'
    
    log_dir = str(Path(save_dir_training) / cfg.experiment / cfg.model)
    print('__LOG__' + log_dir)
    
    # Initialize configuration management system
    hydra.core.global_hydra.GlobalHydra.instance().clear()  # reinitialize hydra if already initialized
    hydra.initialize(config_path=cfg.config_path_training)
    
    # remove previous scenario_visualization folder
    shutil.rmtree(f'{save_dir_training}/scenario_visualization', ignore_errors=True)
    
    # Compose the configuration
    # cfg = hydra.compose(config_name=cfg.config_name_training, overrides=[
    #     f'group={str(save_dir_training)}',
    #     f'cache.cache_path={str(save_dir_training)}/cache',
    #     f'experiment_name={cfg.experiment}',
    #     f'job_name={cfg.model}',
    #     f'py_func={cfg.py_func}',
    #     f'+training={cfg.training_model}',  # raster model that consumes ego, agents and map raster layers and regresses the ego's trajectory
    #     f'lr_scheduler={cfg.lr_scheduler}',
    #     f'scenario_builder={cfg.scenario_builder}',  # use nuplan mini database  # ['nuplan','nuplan_challenge','nuplan_mini']
    #     f'scenario_filter.limit_total_scenarios={cfg.limit_total_scenarios}',  # Choose 500 scenarios to train with
    #     f'scenario_filter.scenario_types={cfg.scenario_types}',
    #     f'lightning.trainer.params.accelerator={cfg.lightning_accelerator}',  # ddp is not allowed in interactive environment, using ddp_spawn instead - this can bottleneck the data pipeline, it is recommended to run training outside the notebook
    #     f'lightning.trainer.params.max_epochs={cfg.max_epochs}',
    #     f'data_loader.params.batch_size={cfg.batch_size}',
    #     f'data_loader.params.num_workers={cfg.num_workers}',
    # ])
    
    # Run the training loop, optionally inspect training artifacts through tensorboard (above cell)
    main_train(cfg)
    
    scenario_visualzation()

    
def simulate(cfg: DictConfig) -> str:
    # add save directory
    save_dir_simulation = os.getenv('NUPLAN_EXP_ROOT')+'/simulation'
    
    simulation_folder = ''
    
    # Initialize configuration management system
    hydra.core.global_hydra.GlobalHydra.instance().clear()  # reinitialize hydra if already initialized
    hydra.initialize(config_path=cfg.config_path_simulation)
    
    if cfg.planner == 'simple_planner':
        # Compose the configuration
        cfg = hydra.compose(config_name=cfg.config_name_simulation, overrides=[
            f'experiment_name={cfg.experiment}',
            f'group={save_dir_simulation}',
            # f'ego_controller={cfg.ego_controller}',
            # f'observation={cfg.observation}',
            f'planner={cfg.planner}',
            f'+simulation={cfg.challenge}',
            f'scenario_builder={cfg.scenario_builder}',
            f'scenario_filter={cfg.scenarios}',
            f'scenario_filter.scenario_types={cfg.scenario_types}',
            f'scenario_filter.num_scenarios_per_type={cfg.scenarios_per_type}',
        ])
        
        # Run the simulation loop
        main_simulation(cfg)
    
        # Simulation folder for visualization in nuBoard
        simulation_folder = cfg.output_dir
        
    elif cfg.planner == 'ml_planner':
        
        cfg.model = cfg.model
        
        log_dir = str(Path(os.getenv('NUPLAN_EXP_ROOT')) / 'training' / cfg.experiment / cfg.model)
        print(log_dir)

        # Get the checkpoint of the trained model
        # last_experiment = sorted(os.listdir(LOG_DIR))[-1]
        
        i = -1
        train_experiment_dir = sorted(Path(log_dir).iterdir())[i]  # get last experiment
        while not (train_experiment_dir / 'checkpoints').exists():
            i -= 1
            train_experiment_dir = sorted(Path(log_dir).iterdir())[i]  # get last experiment
            if i == -10: break
        checkpoint = sorted((train_experiment_dir / 'checkpoints').iterdir())[-1]  # get last checkpoint
        
        cfg.model_path = str(checkpoint).replace("=", "\=")

        
        # Compose the configuration
        cfg = hydra.compose(config_name=cfg.config_name_simulation, overrides=[
            f'experiment_name={cfg.experiment}',
            f'group={save_dir_simulation}',
            # f'ego_controller={cfg.ego_controller}',
            # f'observation={cfg.observation}',
            f'planner={cfg.planner}',
            f'model={cfg.model}',
            'planner.ml_planner.model_config=${model}',  # hydra notation to select model config
            f'planner.ml_planner.checkpoint_path={cfg.model_path}',  # this path can be replaced by the checkpoint of the model trained in the previous section
            f'+simulation={cfg.challenge}',
            f'scenario_builder={cfg.scenario_builder}',
            f'scenario_filter={cfg.scenarios}',
            f'scenario_filter.scenario_types={cfg.scenario_types}',
            f'scenario_filter.num_scenarios_per_type={cfg.scenarios_per_type}',
        ])
    
        # Run the simulation loop
        main_simulation(cfg)
    
        # Simulation folder for visualization in nuBoard
        simulation_folder = cfg.output_dir
    
    return simulation_folder


def open_nuboard(cfg: DictConfig, simulation_folders: List[str]) -> None:
    # Initialize configuration management system
    hydra.core.global_hydra.GlobalHydra.instance().clear()  # reinitialize hydra if already initialized
    hydra.initialize(config_path=cfg.config_path_nuboard)

    # Compose the configuration
    cfg = hydra.compose(config_name=cfg.config_name_nuboard, overrides=[
        f'scenario_builder={cfg.scenario_builder}',  # set the database (same as simulation) used to fetch data for visualization
        f'simulation_path={simulation_folders}',  # nuboard file path(s), if left empty the user can open the file inside nuBoard
    ])
    
    # Run nuBoard
    main_nuboard(cfg)
    

def scenario_visualzation():
    # Create the frames
    frames = []
    exp_root = os.getenv('NUPLAN_EXP_ROOT')
    gif_root = f'{exp_root}/training/scenario_visualization'
    if not os.path.exists(gif_root): os.makedirs(gif_root, exist_ok=True)
    
    # Get the list of all files and folders in the specified directory
    dir_contents = os.listdir(gif_root)
    # Filter out only the folders from the list
    dirs = sorted([f for f in dir_contents if os.path.isdir(os.path.join(gif_root, f))])
    
    for dir in dirs:
        time_sorted_images = sorted(glob.glob(f'{gif_root}/{dir}/*.png'), key=os.path.getmtime)
        frames = [Image.open(i) for i in time_sorted_images]
        # Save into a GIF file that loops forever
        rest_images = [frames[0]] * 3 + frames[1:]
        try:
            frames[0].save(f'{gif_root}/{dir}.gif', format='GIF', append_images=rest_images, save_all=True, duration=500, loop=0)
        except:
            print(f'Could not create {dir}.gif :(')

def load_cfgs(names: Optional[List[str]]=None) -> List[DictConfig]:
    
    if names is None:
        # Load all .yaml files in tutorials/config
        file_list = glob.glob("tutorials/config/*.yaml")
        # file_list = os.listdir("tutorials/config")
    else:
        if isinstance(names, str): names = [names]
        # Load given .yaml files from tutorials/config
        file_list = []
        for name in names:
            for root, dirs, files in os.walk("tutorials/config"):
                file = name if name.endswith(".yaml") else name+".yaml"
                if file in files:
                    filename = os.path.join(root, file)
                    file_list.append(filename)
    print("Configs to run tasks: ", file_list)
    cfgs = []           
    for filename in file_list:
        if filename.endswith(".yaml"): cfgs.append(yaml.safe_load(open(filename)))
        
        # # Initialize configuration management system
        # hydra.core.global_hydra.GlobalHydra.instance().clear()  # reinitialize hydra if already initialized
        # hydra.initialize(config_path="config")
        # filename_without_ext = os.path.splitext(filename)[0]
        # # Compose the configuration
        # cfg = hydra.compose(config_name=filename_without_ext)
        # cfgs.append(cfg)
    if len(cfgs) == 0: print("No config to run tasks!")
    return cfgs

@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg: DictConfig):
    #TODO save cfg to self
    # self.cfg = cfg
    
    simulation_folders = []
    OPEN_NUBOARD = False
    
    if "train" in cfg.tasks:
        train(cfg)
    if "simulate" in cfg.tasks:
        simulation_folder = simulate(cfg)
        if "open_nuboard" in cfg.tasks:
            OPEN_NUBOARD = True
            simulation_folders.append(simulation_folder)

    if OPEN_NUBOARD: open_nuboard(cfg, simulation_folders)


if __name__ == '__main__':
    # cfgs = load_cfgs() # ["default_config"]
    
    # Initialize configuration management system
    hydra.core.global_hydra.GlobalHydra.instance().clear()  # reinitialize hydra if already initialized
    hydra.initialize(config_path=CONFIG_PATH)
    # Compose the configuration
    cfg = hydra.compose(config_name=CONFIG_NAME)            # build hydra config file tree
    
    main(cfg)