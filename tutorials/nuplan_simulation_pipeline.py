import os
from pathlib import Path
import hydra
from nuplan.planning.script.run_simulation import main as main_simulation
from nuplan.planning.script.run_nuboard import main as main_nuboard
from omegaconf import DictConfig

all_scenario_types = ['accelerating_at_crosswalk', 'accelerating_at_stop_sign', 'accelerating_at_stop_sign_no_crosswalk', 'accelerating_at_traffic_light', 'accelerating_at_traffic_light_with_lead', 'accelerating_at_traffic_light_without_lead', 'behind_bike', 'behind_long_vehicle', 'behind_pedestrian_on_driveable', 'behind_pedestrian_on_pickup_dropoff', 'changing_lane', 'changing_lane_to_left', 'changing_lane_to_right', 'changing_lane_with_lead', 'changing_lane_with_trail', 'crossed_by_bike', 'crossed_by_vehicle', 'following_lane_with_lead', 'following_lane_with_slow_lead', 'following_lane_without_lead', 'high_lateral_acceleration', 'high_magnitude_jerk', 'high_magnitude_speed', 'low_magnitude_speed', 'medium_magnitude_speed', 'near_barrier_on_driveable', 'near_construction_zone_sign', 'near_high_speed_vehicle', 'near_long_vehicle', 'near_multiple_bikes', 'near_multiple_pedestrians', 'near_multiple_vehicles', 'near_pedestrian_at_pickup_dropoff', 'near_pedestrian_on_crosswalk', 'near_pedestrian_on_crosswalk_with_ego', 'near_trafficcone_on_driveable', 'on_all_way_stop_intersection', 'on_carpark', 'on_intersection', 'on_pickup_dropoff', 'on_stopline_crosswalk', 'on_stopline_stop_sign', 'on_stopline_traffic_light', 'on_traffic_light_intersection', 'starting_high_speed_turn', 'starting_left_turn', 'starting_low_speed_turn', 'starting_protected_cross_turn', 'starting_protected_noncross_turn', 'starting_right_turn', 'starting_straight_stop_sign_intersection_traversal', 'starting_straight_traffic_light_intersection_traversal', 'starting_u_turn', 'starting_unprotected_cross_turn', 'starting_unprotected_noncross_turn', 'stationary', 'stationary_at_crosswalk', 'stationary_at_traffic_light_with_lead', 'stationary_at_traffic_light_without_lead', 'stationary_in_traffic', 'stopping_at_crosswalk', 'stopping_at_stop_sign_no_crosswalk', 'stopping_at_stop_sign_with_lead', 'stopping_at_stop_sign_without_lead', 'stopping_at_traffic_light_with_lead', 'stopping_at_traffic_light_without_lead', 'stopping_with_lead', 'traversing_crosswalk', 'traversing_intersection', 'traversing_narrow_lane', 'traversing_pickup_dropoff', 'traversing_traffic_light_intersection', 'waiting_for_pedestrian_to_cross'] # all 73 scenario types
selected_scenario_types = '[near_multiple_vehicles, on_pickup_dropoff, starting_unprotected_cross_turn, high_magnitude_jerk]' # select scenario types

def simulate(sim_dict: dict) -> str:
    # Location of path with all simulation configs
    CONFIG_PATH = sim_dict['CONFIG_PATH']
    CONFIG_NAME = sim_dict['CONFIG_NAME']

    # Select the planner and simulation challenge
    PLANNER = sim_dict['PLANNER']  # [simple_planner, ml_planner]
    CHALLENGE = sim_dict['CHALLENGE']  # [open_loop_boxes, closed_loop_nonreactive_agents, closed_loop_reactive_agents]
    
    DATASET = sim_dict['DATASET']
    SCENARIOS = sim_dict['SCENARIOS']
    SCENARIO_TYPES = sim_dict['SCENARIO_TYPES']
    SCENARIOS_PER_TYPE = sim_dict['SCENARIOS_PER_TYPE']
    
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
            f'scenario_builder={DATASET}',
            f'scenario_filter={SCENARIOS}',
            f'scenario_filter.scenario_types={SCENARIO_TYPES}',
            f'scenario_filter.num_scenarios_per_type={SCENARIOS_PER_TYPE}',
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
        last_experiment = sorted(os.listdir(LOG_DIR))[-1]
        
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
            f'scenario_builder={DATASET}',
            f'scenario_filter={SCENARIOS}',
            f'scenario_filter.scenario_types={SCENARIO_TYPES}',
            f'scenario_filter.num_scenarios_per_type={SCENARIOS_PER_TYPE}',
        ])
    
        # Run the simulation loop
        main_simulation(cfg)
    
        # Simulation folder for visualization in nuBoard
        simulation_folder = cfg.output_dir
    
    return simulation_folder


def open_nuboard(simulation_folders: list[str]) -> None:
    # Location of path with all nuBoard configs
    CONFIG_PATH = '../nuplan/planning/script/config/nuboard'
    CONFIG_NAME = 'default_nuboard'

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
    sim_dicts = []
    # # Simple Planner
    # sim_dicts.append(
    #     dict(
    #         # Location of path with all simulation configs
    #         CONFIG_PATH = '../nuplan/planning/script/config/simulation',
    #         CONFIG_NAME = 'default_simulation',

    #         # Select the planner and simulation challenge
    #         PLANNER = 'simple_planner',  # [simple_planner, ml_planner]
    #         CHALLENGE = 'open_loop_boxes',  # [open_loop_boxes, closed_loop_nonreactive_agents, closed_loop_reactive_agents]
    #         DATASET_PARAMS = [
    #             'scenario_builder=nuplan_mini',  # use nuplan mini database
    #             'scenario_filter=all_scenarios',  # initially select all scenarios in the database
    #             'scenario_filter.scenario_types=[near_multiple_vehicles, on_pickup_dropoff, starting_unprotected_cross_turn, high_magnitude_jerk]',  # select scenario types
    #             'scenario_filter.num_scenarios_per_type=10',  # use 10 scenarios per scenario type
    #         ],
        
    #         # Name of the experiment
    #         EXPERIMENT = 'simulation_simple_experiment',

    #         # add save directory
    #         SAVE_DIR = os.getenv('NUPLAN_EXP_ROOT'),
            
    #         # for ML Planner only
    #         MODEL = None
    #     )
    # )
    # # Raster Model
    # sim_dicts.append(
    #     dict(
    #         # Location of path with all simulation configs
    #         CONFIG_PATH = '../nuplan/planning/script/config/simulation',
    #         CONFIG_NAME = 'default_simulation',

    #         # Select the planner and simulation challenge
    #         PLANNER = 'ml_planner',  # [simple_planner, ml_planner]
    #         CHALLENGE = 'open_loop_boxes',  # [open_loop_boxes, closed_loop_nonreactive_agents, closed_loop_reactive_agents]
    #         DATASET_PARAMS = [
    #             'scenario_builder=nuplan_mini',  # use nuplan mini database
    #             'scenario_filter=all_scenarios',  # initially select all scenarios in the database
    #             'scenario_filter.scenario_types=[near_multiple_vehicles, on_pickup_dropoff, starting_unprotected_cross_turn, high_magnitude_jerk]',  # select scenario types
    #             'scenario_filter.num_scenarios_per_type=10',  # use 10 scenarios per scenario type
    #         ],
        
    #         # Name of the experiment
    #         EXPERIMENT = 'raster_experiment',
            
    #         # add save directory
    #         SAVE_DIR = os.getenv('NUPLAN_EXP_ROOT'),
            
    #         # for ML Planner only
    #         MODEL = 'raster_model'
    #     )
    # )
    # # Simple Vector Model
    # sim_dicts.append(
    #     dict(
    #         # Location of path with all simulation configs
    #         CONFIG_PATH = '../nuplan/planning/script/config/simulation',
    #         CONFIG_NAME = 'default_simulation',

    #         # Select the planner and simulation challenge
    #         PLANNER = 'ml_planner',  # [simple_planner, ml_planner]
    #         CHALLENGE = 'open_loop_boxes',  # [open_loop_boxes, closed_loop_nonreactive_agents, closed_loop_reactive_agents]
    #         DATASET_PARAMS = [
    #             'scenario_builder=nuplan_mini',  # use nuplan mini database
    #             'scenario_filter=all_scenarios',  # initially select all scenarios in the database
    #             'scenario_filter.scenario_types=[near_multiple_vehicles, on_pickup_dropoff, starting_unprotected_cross_turn, high_magnitude_jerk]',  # select scenario types
    #             'scenario_filter.num_scenarios_per_type=10',  # use 10 scenarios per scenario type
    #         ],
        
    #         # Name of the experiment
    #         EXPERIMENT = 'simple_vector_experiment',
            
    #         # add save directory
    #         SAVE_DIR = os.getenv('NUPLAN_EXP_ROOT'),
            
    #         # for ML Planner only
    #         MODEL = 'simple_vector_model'
    #     )
    # )
    # # Vector Model
    sim_dicts.append(
        dict(
            # Location of path with all simulation configs
            CONFIG_PATH = '../nuplan/planning/script/config/simulation',
            CONFIG_NAME = 'default_simulation',

            # Select the planner and simulation challenge
            PLANNER = 'ml_planner',  # [simple_planner, ml_planner]
            CHALLENGE = 'open_loop_boxes',  # [open_loop_boxes, closed_loop_nonreactive_agents, closed_loop_reactive_agents]

            DATASET = 'nuplan_mini', # use nuplan mini database
            SCENARIOS = 'all_scenarios', # initially select all scenarios in the database
            SCENARIO_TYPES = selected_scenario_types,  # select scenario types
            SCENARIOS_PER_TYPE = 10,  # use 10 scenarios per scenario type
        
            # Name of the experiment
            EXPERIMENT = 'vector_experiment',
            
            # add save directory
            SAVE_DIR = os.getenv('NUPLAN_EXP_ROOT'),
            
            # for ML Planner only
            MODEL = 'vector_model',
        )
    )
    
    simulation_folders = []
    for sim_dict in sim_dicts:
        simulation_folder = simulate(sim_dict)
        simulation_folders.append(simulation_folder)
        
    open_nuboard(simulation_folders)
    

    
