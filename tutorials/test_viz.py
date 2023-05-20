import torch
from typing import cast
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory
from nuplan.planning.training.preprocessing.features.trajectories import Trajectories
from nuplan.planning.training.preprocessing.features.tensor_target import TensorTarget

def compute_trajectories(predicted_batch: torch.Tensor, all_agents: bool=False, all_trajectories: bool=False) -> Trajectories:
    _, _, num_points, num_features = predicted_batch.size()
    
    # create agents and mutimodal trajectories mask
    agent_mask = torch.ones_like(predicted_batch)
    if not all_agents: agent_mask[1:, :, :, :] *= 0
    traj_mask = torch.ones_like(predicted_batch)
    if not all_trajectories: traj_mask[:, 1:, :, :] *= 0
    agent_traj_mask = torch.mul(agent_mask, traj_mask)
    agent_traj_mask_bool = agent_traj_mask.bool()
    
    trajs = predicted_batch[agent_traj_mask_bool].view(-1, num_points, num_features)
    print(trajs.size())

    pred_trajs = Trajectories(Trajectory(trajs).unpack())
    
    return pred_trajs

def tensor_to_trajectories(pred_tensor: TensorTarget) -> Trajectories:
    
    
    
    predicted_batch_trajs = list(pred_tensor.data.chunk(pred_tensor.data.size(0), dim=0))
    
    for predicted_batch in predicted_batch_trajs:
        # predicted_agents_trajs = list(predicted_batch.squeeze(dim=0).chunk(predicted_batch.squeeze(dim=0).size(0), dim=0))
        # pred_multiagent_multimodal_trajectories = Trajectories([Trajectory(data=pred.squeeze(dim=0)) for pred in predicted_agents_trajs]) # 8 x [6,16,3]
        pred_trajs = Trajectories([Trajectory(predicted_batch.squeeze(dim=0)[a,b,:,:].unsqueeze(dim=0))
                                   for a in range(predicted_batch.squeeze(dim=0).shape[0])
                                   for b in range(predicted_batch.squeeze(dim=0).shape[1])])
    
    pred_trajs = Trajectories([Trajectory(pred_tensor.data[a,b,c,:,:]) for a in range(pred_tensor.data.shape[0]) for b in range(pred_tensor.data.shape[1]) for c in range(pred_tensor.data.shape[2])])

    
    pred_trajs = pred_trajspredicted_trajs = list(pred_tensor.data.chunk(pred_tensor.data.size(0), dim=0))
    
    
    pred_multimodal_trajectories = Trajectories([Trajectory(data=pred.squeeze(dim=0)) for pred in predicted_trajs]) # 8 x [6,16,3]
    
    pred_multimodal_trajectories_2 = Trajectories([traj.unpack() for traj in pred_multimodal_trajectories.trajectories])
    
    pred_multimodal_trajectories_3 = Trajectories([t for traj in pred_multimodal_trajectories.trajectories for t in traj.unpack() for i in t.unpack()])
    

    
    pred_multimodal_trajectories.trajectories[0].unpack()
    
    return pred_tensor
    

if __name__ == '__main__':
    pred_tensor = TensorTarget(torch.rand(8, 51, 6, 16, 3)) # pred_tensor = predictions["pred_agents"]
    # pred_tensor = TensorTarget(torch.rand(51, 6, 16, 3)) # pred_tensor = predictions["pred_agents"]
    pred_traj = Trajectory(torch.rand(6, 16, 3))
    
    predicted_batch_trajs = list(pred_tensor.data.chunk(pred_tensor.data.size(0), dim=0))
    
    for predicted_batch in predicted_batch_trajs:
        compute_trajectories(predicted_batch.squeeze(dim=0))
    
    tensor_to_trajectories(pred_tensor)
    
