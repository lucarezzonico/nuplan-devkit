import os
from pathlib import Path
import hydra
from omegaconf import OmegaConf, DictConfig
import tempfile
from nuplan.planning.script.run_training import main as main_train
from nuplan.planning.script.run_simulation import main as main_simulation
from nuplan.planning.script.run_nuboard import main as main_nuboard
import yaml
import glob
from typing import List, Optional, cast
import shutil
from PIL import Image


class Tasks():
    
    def __init__(self) -> None:
        self.all_scenario_types = ['accelerating_at_crosswalk', 'accelerating_at_stop_sign', 'accelerating_at_stop_sign_no_crosswalk', 'accelerating_at_traffic_light', 'accelerating_at_traffic_light_with_lead', 'accelerating_at_traffic_light_without_lead', 'behind_bike', 'behind_long_vehicle', 'behind_pedestrian_on_driveable', 'behind_pedestrian_on_pickup_dropoff', 'changing_lane', 'changing_lane_to_left', 'changing_lane_to_right', 'changing_lane_with_lead', 'changing_lane_with_trail', 'crossed_by_bike', 'crossed_by_vehicle', 'following_lane_with_lead', 'following_lane_with_slow_lead', 'following_lane_without_lead', 'high_lateral_acceleration', 'high_magnitude_jerk', 'high_magnitude_speed', 'low_magnitude_speed', 'medium_magnitude_speed', 'near_barrier_on_driveable', 'near_construction_zone_sign', 'near_high_speed_vehicle', 'near_long_vehicle', 'near_multiple_bikes', 'near_multiple_pedestrians', 'near_multiple_vehicles', 'near_pedestrian_at_pickup_dropoff', 'near_pedestrian_on_crosswalk', 'near_pedestrian_on_crosswalk_with_ego', 'near_trafficcone_on_driveable', 'on_all_way_stop_intersection', 'on_carpark', 'on_intersection', 'on_pickup_dropoff', 'on_stopline_crosswalk', 'on_stopline_stop_sign', 'on_stopline_traffic_light', 'on_traffic_light_intersection', 'starting_high_speed_turn', 'starting_left_turn', 'starting_low_speed_turn', 'starting_protected_cross_turn', 'starting_protected_noncross_turn', 'starting_right_turn', 'starting_straight_stop_sign_intersection_traversal', 'starting_straight_traffic_light_intersection_traversal', 'starting_u_turn', 'starting_unprotected_cross_turn', 'starting_unprotected_noncross_turn', 'stationary', 'stationary_at_crosswalk', 'stationary_at_traffic_light_with_lead', 'stationary_at_traffic_light_without_lead', 'stationary_in_traffic', 'stopping_at_crosswalk', 'stopping_at_stop_sign_no_crosswalk', 'stopping_at_stop_sign_with_lead', 'stopping_at_stop_sign_without_lead', 'stopping_at_traffic_light_with_lead', 'stopping_at_traffic_light_without_lead', 'stopping_with_lead', 'traversing_crosswalk', 'traversing_intersection', 'traversing_narrow_lane', 'traversing_pickup_dropoff', 'traversing_traffic_light_intersection', 'waiting_for_pedestrian_to_cross'] # all 73 scenario types
        self.selected_scenario_types = '[near_multiple_vehicles, on_pickup_dropoff, starting_unprotected_cross_turn, high_magnitude_jerk]' # select scenario types
    
    def train(self, cfg: DictConfig) -> None:
        # Location of path with all simulation configs
        cfg.config_path_training = cfg.config_path + '/training'
        cfg.config_name_training = cfg.config_name + '_training'
        
        # add save directory
        cfg.save_dir_training = os.getenv('NUPLAN_EXP_ROOT') + '/training'
        
        cfg.log_dir = str(Path(cfg.save_dir_training) / cfg.experiment / cfg.model)
        print('__LOG__' + cfg.log_dir)
        
        # Initialize configuration management system
        hydra.core.global_hydra.GlobalHydra.instance().clear()  # reinitialize hydra if already initialized
        hydra.initialize(config_path=cfg.config_path_training)
        
        # remove previous scenario_visualization folder
        shutil.rmtree(f'{cfg.save_dir_training}/scenario_visualization', ignore_errors=True)
        
        # Compose the configuration
        cfg = hydra.compose(config_name=cfg.config_name_training, overrides=[
            f'group={str(cfg.save_dir_training)}',
            f'cache.cache_path={str(cfg.save_dir_training)}/cache',
            f'experiment_name={cfg.experiment}',
            f'job_name={cfg.model}',
            f'py_func={cfg.py_func}',
            f'+training={cfg.training_model}',  # raster model that consumes ego, agents and map raster layers and regresses the ego's trajectory
            f'lr_scheduler={cfg.lr_scheduler}',
            # f'optimizer.lr={5e-4}',
            f'scenario_builder={cfg.scenario_builder}',  # use nuplan mini database  # ['nuplan','nuplan_challenge','nuplan_mini']
            f'scenario_filter.limit_total_scenarios={cfg.limit_total_scenarios}',  # Choose 500 scenarios to train with
            f'scenario_filter.scenario_types={cfg.scenario_types}',
            f'lightning.trainer.params.accelerator={cfg.lightning_accelerator}',  # ddp is not allowed in interactive environment, using ddp_spawn instead - this can bottleneck the data pipeline, it is recommended to run training outside the notebook
            f'lightning.trainer.params.max_epochs={cfg.max_epochs}',
            f'data_loader.params.batch_size={cfg.batch_size}',
            f'data_loader.params.num_workers={cfg.num_workers}',
        ])
        
        # Run the training loop, optionally inspect training artifacts through tensorboard (above cell)
        main_train(cfg)
        
        self.scenario_visualzation()

        
    def simulate(self, cfg: DictConfig) -> str:
        # Location of path with all simulation configs
        cfg.config_path_simulation = cfg.config_path + '/simulation'
        cfg.config_name_simulation = cfg.config_name + '_simulation'
        
        # add save directory
        cfg.save_dir_simulation = os.getenv('NUPLAN_EXP_ROOT')+'/simulation'
        
        simulation_folder = ''
        
        # Initialize configuration management system
        hydra.core.global_hydra.GlobalHydra.instance().clear()  # reinitialize hydra if already initialized
        hydra.initialize(config_path=cfg.config_path_simulation)
        
        if cfg.planner == 'simple_planner':
            # Compose the configuration
            cfg = hydra.compose(config_name=cfg.config_name_simulation, overrides=[
                f'experiment_name={cfg.experiment}',
                f'group={cfg.save_dir_simulation}',
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
            
            cfg.log_dir = str(Path(os.getenv('NUPLAN_EXP_ROOT')) / 'training' / cfg.experiment / cfg.model)
            print(cfg.log_dir)

            # Get the checkpoint of the trained model
            # last_experiment = sorted(os.listdir(LOG_DIR))[-1]
            
            i = -1
            ## GET LAST MODEL
            # train_experiment_dir = sorted(Path(cfg.log_dir).iterdir())[i]  # get last experiment
            # while not (train_experiment_dir / 'checkpoints').exists():
            #     i -= 1
            #     train_experiment_dir = sorted(Path(cfg.log_dir).iterdir())[i]  # get last experiment
            #     if i == -10: break
            # checkpoint = sorted((train_experiment_dir / 'checkpoints').iterdir())[-1]  # get last checkpoint
            
            ## GET BEST MODEL
            while not (train_experiment_dir / 'best_model').exists():
                i -= 1
                train_experiment_dir = sorted(Path(cfg.log_dir).iterdir())[i]  # get last experiment
                if i == -10: break
            checkpoint = sorted((train_experiment_dir / 'best_model').iterdir())[-1]
            
            cfg.model_path = str(checkpoint).replace("=", "\=")

            
            # Compose the configuration
            cfg = hydra.compose(config_name=cfg.config_name_simulation, overrides=[
                f'experiment_name={cfg.experiment}',
                f'group={cfg.save_dir_simulation}',
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


    def open_nuboard(self, cfg: DictConfig, simulation_folders: List[str]) -> None:
        # Location of path with all nuBoard configs
        cfg.config_path_nuboard = cfg.config_path+'/nuboard'
        cfg.config_name_nuboard = cfg.config_name+'_nuboard'

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
        

    def scenario_visualzation(self):
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

    def load_cfgs(self, names: Optional[List[str]]=None) -> List[DictConfig]:
        
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
    
    # @hydra.main(config_path="config", config_name="default_config")
    def main(self, cfgs: List[DictConfig]):
        #TODO save cfg to self
        # self.cfg = cfg
        
        simulation_folders = []
        OPEN_NUBOARD = False
        
        cfg = DictConfig(None)
        for cfg in cfgs:
            cfg = DictConfig(cfg) # convert dict to DictConfig
            if "train" in cfg.tasks:
                task.train(cfg)
            if "simulate" in cfg.tasks:
                simulation_folder = task.simulate(cfg)
                if "open_nuboard" in cfg.tasks:
                    OPEN_NUBOARD = True
                    simulation_folders.append(simulation_folder)

        if OPEN_NUBOARD: task.open_nuboard(cfg, simulation_folders)


if __name__ == '__main__':
    task = Tasks()
    cfgs = task.load_cfgs() # ["default_config"]    
    task.main(cfgs)