from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_filter_utils import discover_log_dbs
from tutorials.utils.tutorial_utils import get_scenario_type_token_map

from tutorials.utils.tutorial_utils import visualize_nuplan_scenarios
from tutorials.utils.tutorial_utils import visualize_scenario

from tutorials.utils.tutorial_utils import get_default_scenario_from_token


import os
import glob
NUPLAN_DATA_ROOT = os.getenv('NUPLAN_DATA_ROOT')
NUPLAN_MAPS_ROOT = os.getenv('NUPLAN_MAPS_ROOT')
NUPLAN_DB_FILES = os.getenv('NUPLAN_DB_FILES',
             glob.glob('/home/luca/Documents/nuplan/dataset/nuplan-v1.1/mini' + '/*.db'))
NUPLAN_MAP_VERSION = os.getenv('NUPLAN_MAP_VERSION', 'nuplan-maps-v1.0')


log_db_files = discover_log_dbs(NUPLAN_DB_FILES)
scenario_type_token_map = get_scenario_type_token_map(log_db_files)

scenario_type_list = sorted(scenario_type_token_map.keys())
log_db_file, token = scenario_type_token_map["high_lateral_acceleration"][0]

scenario = get_default_scenario_from_token(NUPLAN_DATA_ROOT, log_db_file, token,
                                           NUPLAN_MAPS_ROOT, NUPLAN_MAP_VERSION)

visualize_scenario(scenario, bokeh_port=8888)
