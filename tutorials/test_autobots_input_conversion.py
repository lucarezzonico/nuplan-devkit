
import pickle
from nuplan.planning.training.preprocessing.feature_builders.autobots_feature_builder import AutobotsTargetBuilder
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.preprocessing.features.autobots_feature_conversion import NuplanToAutobotsConverter
from nuplan.planning.training.preprocessing.feature_builders.autobots_feature_builder import AutobotsModeProbsNominalTargetBuilder, AutobotsTargetBuilder, AutobotsPredNominalTargetBuilder

from nuplan.planning.training.preprocessing.feature_builders.vector_map_feature_builder import VectorMapFeatureBuilder
from nuplan.planning.training.preprocessing.feature_builders.agents_feature_builder import AgentsFeatureBuilder
from nuplan.planning.training.preprocessing.target_builders.ego_trajectory_target_builder import EgoTrajectoryTargetBuilder

import os
import torch
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import (
    DEFAULT_SCENARIO_NAME,
    ScenarioExtractionInfo,
)


# file_name="./project_records/laneGCN_input_sample.obj" # one batch with batch size 1
    
def loadSample():
    file_name="./project_records/laneGCN_input_sample_batch.pkl" # one batch with batch size 2

    with open(file_name, 'rb') as file:
        datainput=pickle.load(file)
        print(f'Object successfully loaded to "{file_name}"')
    return datainput

def testMap(datainput):

    vec_map=datainput[0]['vector_map']

    ts=NuplanToAutobotsConverter.VectorMapToAutobotsMapTensor(vec_map)

    print(ts)

def testAgents(datainput):

    ag=datainput[0]['agents']

    ts=NuplanToAutobotsConverter.AgentsToAutobotsAgentsTensor(ag)

    print(ts)

def testTrajectory(datainput):
    # fb=AutobotsTargetBuilder(TrajectorySampling(num_poses=10, time_horizon=5.0))
    datainput=loadSample()
    ag=datainput[1]['trajectory']
    c=NuplanToAutobotsConverter()
    ts=c.TrajectoryToAutobotsEgoin(ag)

    print(ts)
    return ts

def group1():
    datainput=loadSample()
    trajs_3=testTrajectory(datainput)
    c=NuplanToAutobotsConverter()

    ang_vec=trajs_3[:,1:,:2] - trajs_3[:,:-1,:2] 
    ang = torch.atan2(ang_vec[:,:,0], ang_vec[:,:,1])
    trajs_3[:,:-1,2] = ang
    trajs_3[:,-1,2] = trajs_3[:,-2,2]

    # c.output_tensor_to_trajectory(n)
    # testAgents(datainput)
    # testMap(datainput)

def group2():
    scenario = NuPlanScenario(
        data_root='/data1/nuplan/dataset/nuplan-v1.1/trainval',
        log_file_load_path='/data1/nuplan/dataset/nuplan-v1.1/trainval/2021.08.09.17.55.59_veh-28_00021_00307.db',
        initial_lidar_token='01602e046c4753b0',
        initial_lidar_timestamp=1628531925250106,
        scenario_type=DEFAULT_SCENARIO_NAME,
        map_root='stationary',
        map_version='nuplan-maps-v1.0',
        map_name='us-ma-boston',
        scenario_extraction_info=ScenarioExtractionInfo() ,
        ego_vehicle_parameters=get_pacifica_parameters(),
    )

    # scenario = NuPlanScenario(
    #     data_root='/data1/nuplan/dataset/nuplan-v1.1/trainval',
    #     log_file_load_path='/data1/nuplan/dataset/nuplan-v1.1/trainval/2021.08.20.18.15.01_veh-28_00016_00436',
    #     initial_lidar_token='f35c81eeb76759fc',
    #     initial_lidar_timestamp=1628531925250106,
    #     scenario_type=DEFAULT_SCENARIO_NAME,
    #     map_root='stationary',
    #     map_version='nuplan-maps-v1.0',
    #     map_name='us-ma-boston',
    #     scenario_extraction_info=ScenarioExtractionInfo() ,
    #     ego_vehicle_parameters=get_pacifica_parameters(),
    # )

    # VectorMapFeatureBuilder.get_features_from_scenario(scenario)

    # 实例化各个类
    autobots_target_builder = AutobotsTargetBuilder(TrajectorySampling(num_poses=4, time_horizon=1.5))
    autobots_mode_probs_nominal_target_builder = AutobotsModeProbsNominalTargetBuilder()
    autobots_pred_nominal_target_builder = AutobotsPredNominalTargetBuilder()

    # 调用程序
    a=autobots_target_builder.get_targets(scenario)
    b=autobots_mode_probs_nominal_target_builder.get_targets(scenario)
    c=autobots_pred_nominal_target_builder.get_targets(scenario)


    print("done")


# def group3():
#     import traceback

# try:
#     # your code that might raise an exception
#     x = 1 / 0
# except Exception as e:
#     # handle the exception and get the traceback information
#     print("Exception type:", type(e).__name__)
#     print("Exception message:", str(e))
#     traceback.print_tb(e.__traceback__)


group1()
