# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
#  Modified by Shihao Wang
# ---------------------------------------------
import math
import torch
from torch.utils.data import DistributedSampler as _DistributedSampler
from .sampler import SAMPLER
import numpy as np

@SAMPLER.register_module()
class DistributedSampler(_DistributedSampler):

    def __init__(self,
                 dataset=None,
                 num_replicas=None,
                 rank=None,
                 shuffle=True,
                 seed=0):
        super().__init__(
            dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        # for the compatibility from PyTorch 1.3+
        self.seed = seed if seed is not None else 0
        self.flag = self.dataset.flag
        self.group_sizes = np.bincount(self.flag)
        self.groups_num = len(self.group_sizes)
        self.groups = list(set(self.flag))
        assert self.groups == list(range(self.groups_num))

        # Now, for efficiency, make a dict {group_idx: List[dataset sample_idxs]}
        self.group_idx_to_sample_idxs = {
            group_idx: np.where(self.flag == group_idx)[0].tolist()
            for group_idx in range(self.groups_num)}  

        num_groups_per_gpu = math.ceil(len(self.groups) / self.num_replicas)
        # assign groups (continuous videos) to each gpu rank
        # self.sample_group_idx = self.groups[self.rank*num_groups_per_gpu: min(len(self.groups), (self.rank+1)*num_groups_per_gpu)]
        self.sample_group_idx = self.groups[self.rank::self.num_replicas]
        
        self.sample_idxs = []
        for i in self.sample_group_idx:
            self.sample_idxs.extend(self.group_idx_to_sample_idxs[i])

        self.num_samples = len(self.sample_idxs)
        self.total_size = len(self.dataset)

    def __iter__(self):
        # only used for validation/testing 
        # only support batchsize = 1
        if self.shuffle:
            assert False
        # else:
        #     indices = torch.arange(len(self.dataset)).tolist()

        # # add extra samples to make it evenly divisible
        # # in case that indices is shorter than half of total_size
        # indices = (indices *
        #            math.ceil(self.total_size / len(indices)))[:self.total_size]
        # assert len(indices) == self.total_size

        # # subsample
        # per_replicas = self.total_size//self.num_replicas
        # # indices = indices[self.rank:self.total_size:self.num_replicas]
        # indices = indices[self.rank*per_replicas:(self.rank+1)*per_replicas]
        # assert len(indices) == self.num_samples

        return iter(self.sample_idxs)
