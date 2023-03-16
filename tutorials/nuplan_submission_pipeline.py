import os
from pathlib import Path
import hydra
from nuplan.planning.script.run_simulation import main as main_simulation
from nuplan.planning.script.run_nuboard import main as main_nuboard
from nuplan.planning.script.run_submission_planner import main as main_submission
from omegaconf import DictConfig


def submit(sim_dict: dict) -> None:
    # Location of path with all simulation configs
    CONFIG_PATH = sim_dict['CONFIG_PATH']
    CONFIG_NAME = sim_dict['CONFIG_NAME']

    # Select the planner and simulation challenge
    PLANNER = sim_dict['PLANNER']  # [simple_planner, ml_planner]
    
    # Initialize configuration management system
    hydra.core.global_hydra.GlobalHydra.instance().clear()  # reinitialize hydra if already initialized
    hydra.initialize(config_path=CONFIG_PATH)
    
    # Compose the configuration
    cfg = hydra.compose(config_name=CONFIG_NAME, overrides=[
        f'planner={PLANNER}',
    ])
    
    # Run the simulation loop
    main_submission(cfg)
    
    
if __name__ == '__main__':
    sim_dicts = []
    # # Simple Vector Model
    sim_dicts.append(
        dict(
            # Location of path with all simulation configs
            CONFIG_PATH = '../nuplan/planning/script/config/simulation',
            CONFIG_NAME = 'default_submission_planner',

            # Select the planner and simulation challenge
            PLANNER = 'ml_planner',  # [simple_planner, ml_planner]
        )
    )
    
    simulation_folders = []
    for sim_dict in sim_dicts:
        simulation_folder = submit(sim_dict)
        simulation_folders.append(simulation_folder)

    
