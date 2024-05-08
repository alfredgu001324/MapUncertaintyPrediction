import os
from itertools import permutations
from itertools import product
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.data import Dataset
from tqdm import tqdm

from utils import TemporalData
import pickle

class NuscenesDataset(Dataset):

    def __init__(self,
                root: str,
                split: str,
                transform: Optional[Callable] = None,
                local_radius: float = 50,
                centerline=False) -> None:

        self._split = split
        self._local_radius = local_radius
        self.centerline_used = centerline

        if split == 'train':
            self._directory = 'train'
        elif split == 'train_val':
            self._directory = 'train_val'
        elif split == 'val':
            self._directory = 'val'
        elif split == 'test':
            self._directory = 'test'
        elif split == 'mini_val':
            self._directory = 'mini_val'
        elif split == 'mini_train':
            self._directory = 'mini_train'
        else:
            raise ValueError(split + ' is not valid')
        self.root = root
        self._raw_file_names = os.listdir(self.raw_dir)
        self._processed_file_names = [os.path.splitext(f)[0].split('-')[1] + '.pt' for f in self.raw_file_names]
        self._processed_paths = [os.path.join(self.processed_dir, f) for f in self._processed_file_names]
        super(NuscenesDataset, self).__init__(root, transform=transform)

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self._directory, 'data')

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self._directory, 'processed')

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return self._raw_file_names

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return self._processed_file_names

    @property
    def processed_paths(self) -> List[str]:
        return self._processed_paths
    
    def process(self) -> None:
        for raw_path in tqdm(self.raw_paths):
            kwargs = process_nuscenes(self._split, raw_path, self.centerline_used)
            data = TemporalData(**kwargs)
            torch.save(data, os.path.join(self.processed_dir, str(kwargs['seq_id']) + '.pt'))

    def len(self) -> int:
        return len(self._raw_file_names)

    def get(self, idx) -> Data:
        return torch.load(self.processed_paths[idx])

def process_nuscenes(split, raw_path, centerline_used):
    with open(raw_path, 'rb') as handle:
        data = pickle.load(handle)
    
    agent_indices = np.where(data['agent_type'] == 1)[0]
    num_nodes = len(agent_indices) + 1
    
    av_index = 0
    agent_index = 0

    origin = torch.tensor(data['ego_hist'][-1], dtype=torch.float)
    av_heading_vector = origin - torch.tensor(data['ego_hist'][-2], dtype=torch.float)
    theta = torch.atan2(av_heading_vector[1], av_heading_vector[0])
    rotate_mat = torch.tensor([[torch.cos(theta), -torch.sin(theta)],
                               [torch.sin(theta), torch.cos(theta)]])
    
    x = torch.zeros(num_nodes, 50, 2, dtype=torch.float)
    edge_index = torch.LongTensor(list(permutations(range(num_nodes), 2))).t().contiguous()
    padding_mask = torch.ones(num_nodes, 50, dtype=torch.bool)
    bos_mask = torch.zeros(num_nodes, 20, dtype=torch.bool)
    rotate_angles = torch.zeros(num_nodes, dtype=torch.float)

    padding_mask[0, 0:50] = False
    xy = np.vstack([data['ego_hist'], data['ego_fut']])
    x[0, :] = torch.matmul(torch.tensor(xy) - origin, rotate_mat)
    heading_vector = x[0, 19] - x[0, 18]
    rotate_angles[0] = torch.atan2(heading_vector[1], heading_vector[0])
    for i, index in enumerate(agent_indices):
        i += 1
        padding_mask[i, 0:50] = False
        xy = np.vstack([data['agent_hist'][index][:,:2], data['agent_fut'][index]])
        x[i, :] = torch.matmul(torch.tensor(xy) - origin, rotate_mat)
        heading_vector = x[i, 19] - x[i, 18]
        rotate_angles[i] = torch.atan2(heading_vector[1], heading_vector[0])
    
    bos_mask[:, 0] = ~padding_mask[:, 0]
    bos_mask[:, 1: 20] = padding_mask[:, : 19] & ~padding_mask[:, 1: 20]

    positions = x.clone()
    x[:, 20:] = torch.where((padding_mask[:, 19].unsqueeze(-1) | padding_mask[:, 20:]).unsqueeze(-1),
                            torch.zeros(num_nodes, 30, 2),
                            x[:, 20:] - x[:, 19].unsqueeze(-2))
    x[:, 1: 20] = torch.where((padding_mask[:, : 19] | padding_mask[:, 1: 20]).unsqueeze(-1),
                              torch.zeros(num_nodes, 19, 2),
                              x[:, 1: 20] - x[:, : 19])
    x[:, 0] = torch.zeros(num_nodes, 2)

    ego_pos = np.vstack([data['ego_hist'], data['ego_fut']])
    ego_pos = ego_pos.reshape(1, *ego_pos.shape)
    all_agent_pos = np.concatenate((data['agent_hist'][:,:,:2], data['agent_fut']), axis=1)[agent_indices]
    all_node_pos = np.vstack([ego_pos, all_agent_pos])
    node_positions_19 = torch.tensor(all_node_pos[:,19,:])

    (lane_vectors, lane_actor_index, lane_actor_vectors) = get_lane_features(
        np.arange(0, all_node_pos.shape[0]),
        node_positions_19, 
        origin, 
        rotate_mat, 
        data['predicted_map'],
        centerline_used
    )
    y = None if split == 'test' else x[:, 20:]
    seq_id = os.path.splitext(os.path.basename(raw_path))[0].split('-')[1]

    return {
        'x': x[:, : 20],  # [N, 20, 2]
        'positions': positions,  # [N, 50, 2]
        'edge_index': edge_index,  # [2, N x N - 1]
        'y': y,  # [N, 30, 2]
        'num_nodes': num_nodes,
        'padding_mask': padding_mask,  # [N, 50]
        'bos_mask': bos_mask,  # [N, 20]
        'rotate_angles': rotate_angles,  # [N]
        'lane_vectors': lane_vectors,  # [L, 2]
        'lane_actor_index': lane_actor_index,  # [2, E_{A-L}]
        'lane_actor_vectors': lane_actor_vectors,  # [E_{A-L}, 2]
        'seq_id': int(seq_id),
        'av_index': av_index,
        'agent_index': agent_index,
        'origin': origin.unsqueeze(0),
        'theta': theta,
    }


def get_lane_features(node_inds,
                      node_positions, 
                      origin: torch.Tensor,
                      rotate_mat: torch.Tensor,
                      predicted_map,
                      centerline_used=False):
    lane_positions = []
    lane_position_betas = []
    lane_ids = set()

    node_positions = torch.matmul(node_positions - origin, rotate_mat).float()
    node_positions = torch.cat([node_positions, torch.zeros_like(node_positions)], dim=-1)

    if centerline_used:
        centerlines = predicted_map['centerline']
        centerline_betas = predicted_map['centerline_betas']
    else:
        boundary_scores = predicted_map['boundary_scores']
        boundary_indices = np.where(boundary_scores > 0)[0]  #0.4
        boundary = np.array(predicted_map['boundary'])[boundary_indices]

        divider_scores = predicted_map['divider_scores']
        divider_indices = np.where(divider_scores > 0)[0]
        divider = np.array(predicted_map['divider'])[divider_indices]

        if boundary.shape[0] == 0:
            centerlines = divider
            centerline_betas = predicted_map['divider_betas'][divider_indices]
        else:
            centerlines = boundary
            centerline_betas = predicted_map['boundary_betas'][boundary_indices]

    for i, centerline in enumerate(centerlines):
        lane_centerline = torch.matmul(torch.tensor(centerline, dtype=torch.float32) - origin, rotate_mat)
        lane_positions.append(lane_centerline)
        lane_position_betas.append(torch.tensor(centerline_betas[i]))
        count = len(lane_centerline) - 1

    lane_positions = torch.cat(lane_positions, dim=0)
    lane_position_betas = torch.cat(lane_position_betas, dim=0)
    lane_positions = torch.cat((lane_positions, lane_position_betas), dim=-1)

    lane_actor_index = torch.LongTensor(list(product(torch.arange(lane_positions.size(0)), node_inds))).t().contiguous()
    lane_actor_vectors = \
        lane_positions.repeat_interleave(len(node_inds), dim=0) - node_positions.repeat(lane_positions.size(0), 1)

    return lane_positions, lane_actor_index, lane_actor_vectors
