import os
from pathlib import Path
import hydra
from omegaconf import DictConfig
from nuplan.planning.script.run_nuboard import main as main_nuboard

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
    simulation_folder = [] # browse from nuboard directly
    open_nuboard(simulation_folder)