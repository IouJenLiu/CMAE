from scipy.special import entr
import numpy as np
import torch
from torch.distributions import Categorical
import collections

def norm_ent(count):
    """
    :param count:  (1-d array) stores the count of each outcome
    :return: (float) norm_ent
    """
    n = len(count)
    dist = count / np.sum(count)
    ent = entr(dist)
    h = np.sum(ent)
    h_max = np.log(n)
    ret = h / h_max
    return ret


class Count(object):

    def __init__(self, var_sizes, group_sizes):
        """
        Assume the state is one-hot encoded. A state contains multiple groups. The last group is the environment info.
        Each group consists of multiple variables.
        :param var_sizes: list contains length of each variable (i.e. # of bits in each vars)
        :param group_sizes: list contains length of each group (i.e. # of variable in each group)
        Ex. s = [0,0,1,0, 0,0,1, 0,0,1]
        var_sizes = [4, 3, 3]
        group_sizes = [2, 1]
        self.var_start_idx = [0, 4, 7]
        self.n_vars_in_firt_k_group = [0,2,3]
        """
        self.n_bits = np.sum(var_sizes)
        self.n_vars = len(var_sizes)
        self.n_groups = len(group_sizes)
        self.var_sizes = torch.LongTensor(var_sizes)
        self.var_start_idx = torch.cat((torch.zeros(1, dtype=torch.long), torch.cumsum(self.var_sizes, dim=0)))[:-1]
        self.group_sizes = torch.LongTensor(group_sizes)
        self.n_vars_in_firt_k_group = torch.cat((torch.zeros(1, dtype=torch.long), torch.cumsum(self.group_sizes, dim=0)))
        self.var_to_bits_idxs = [list(range(self.var_start_idx[i], self.var_start_idx[i] + self.var_sizes[i])) for i in range(self.n_vars)]
        self.agent_s_range = np.prod(var_sizes[:group_sizes[0]])
        self.env_s_range = np.prod(var_sizes[-group_sizes[-1]:])
        self.env_n_bits = np.sum(var_sizes[-group_sizes[-1]:])
        self.agent_s_group_size = group_sizes[0]
        self.env_s_group_size = group_sizes[-1]
        self.counts = [[torch.zeros(self.agent_s_range ** level, dtype=torch.long),
                        torch.zeros(self.agent_s_range ** (level - 1) * self.env_s_range, dtype=torch.long)] for level in range(1, self.n_groups)]
        self.counts = [torch.zeros(np.sum(var_sizes), dtype=torch.long)] + self.counts + [[None, torch.zeros(np.prod(var_sizes), dtype=torch.long)]]
        self.tarvar_var_ids = []
        for level in range(self.n_groups):
            if level == 0:
                self.tarvar_var_ids.append([np.array(i) for i in range(self.n_vars)])
            else:
                o_vars1 = list(range(self.n_vars_in_firt_k_group[level]))
                o_vars2 = list(range(self.n_vars_in_firt_k_group[level - 1]))
                e_vars = list(range(self.n_vars_in_firt_k_group[-2], self.n_vars_in_firt_k_group[- 1]))
                self.tarvar_var_ids.append([np.array(o_vars1), np.array(o_vars2 + e_vars)])
        self.tarvar_var_ids.append(np.array(range(self.n_vars)))

    def update_count(self, s_batch, level):
        """
        :param s_batch: [bz, s_size] tensor, onehot encoding
        """
        bz = s_batch.size()[0]
        int_s_batch = torch.nonzero(s_batch, as_tuple=True)[1].view(bz, -1).cpu() - self.var_start_idx

        if level == 0:
            self.counts[level] += torch.sum(s_batch, dim=0, dtype=torch.long).cpu()
        elif level == self.n_groups:
            linear_idx_batch = self.to_linear_idx(int_s_batch, self.var_sizes)
            self.counts[level][1].index_add_(0, linear_idx_batch, torch.ones_like(linear_idx_batch))
        else:
            # o1_group
            linear_idx_batch = self.to_linear_idx(int_s_batch[:, self.tarvar_var_ids[level][0]], self.var_sizes[self.tarvar_var_ids[level][0]])
            self.counts[level][0].index_add_(0, linear_idx_batch, torch.ones_like(linear_idx_batch))
            # env_group
            linear_idx_batch = self.to_linear_idx(int_s_batch[:, self.tarvar_var_ids[level][1]], self.var_sizes[self.tarvar_var_ids[level][1]])
            self.counts[level][1].index_add_(0, linear_idx_batch, torch.ones_like(linear_idx_batch))

    def to_linear_idx(self, idx_batch, dims):
        """
        :param idx_batch: [bz, n_idx] tensor
        :param dims: tensor of size n_idx, which gives dim of each idx
        :return: linear_idx_batch: [bz, 1]
        """
        linear_idx_batch = np.ravel_multi_index(idx_batch.numpy().transpose(), list(dims.numpy()))
        return torch.LongTensor(linear_idx_batch)

    def compute_norm_ent(self, level):
        """
        construct dists. according to count and the desired level
        :param level:
        """
        assert level <= self.n_groups, 'level must be smaller or equal to number of groups'
        dists = []
        norm_ents = []
        if level == 0:
            # agent obs
            for i in range(self.n_vars):
                prob = self.counts[level][self.var_start_idx[i]:self.var_start_idx[i] + self.var_sizes[i]]
                dist = Categorical(prob.float())
                norm_ent = (dist.entropy() / np.log(dist._num_events)).item()
                dists.append(dist)
                norm_ents.append(norm_ent)

            target_var_ids = [np.random.choice(np.where(norm_ents == np.min(norm_ents))[0])]
            dist = dists[target_var_ids[0]]

        elif level == self.n_groups:
            dist = None
            target_var_ids = None
        else:
            # dist1: level agent state
            norm_ent1, dist1 = self.get_norm_ent_(level, 0)
            # dist2: level - 1 agent + env. state
            norm_ent2, dist2 = self.get_norm_ent_(level, 1)
            if norm_ent1 >= norm_ent2:
                # dist2 has lower ent
                target_var_ids = self.tarvar_var_ids[level][1]
                dist = dist2
            else:
                # dist1 has lower ent
                target_var_ids = self.tarvar_var_ids[level][0]
                dist = dist1

        return dist, target_var_ids

    def get_norm_ent_(self, level, target_id):
        if level == 0:
            prob = self.counts[level][self.var_start_idx[target_id]:self.var_start_idx[target_id] + self.var_sizes[target_id]]
        else:
            prob = self.counts[level][target_id]
        dist = Categorical(prob.float())
        norm_ent = dist.entropy() / np.log(dist._num_events)
        return norm_ent, dist

    def compute_p_batch(self, s_batch, dist, target_var_ids):
        """
        :param s_batch: [bz, s_dim] tensor
        :return: [bz, 1] prob. of target_batch
        """
        int_s_batch = self.to_int_batch(s_batch)
        target_batch = int_s_batch[:, target_var_ids]
        flat_target_batch = np.ravel_multi_index(target_batch.numpy().transpose(), self.var_sizes[target_var_ids].numpy())
        p_batch = dist.probs[flat_target_batch]
        return p_batch.view(-1, 1)

    def to_int_batch(self, s_batch):
        bz = s_batch.size()[0]
        int_s_batch = torch.nonzero(s_batch, as_tuple=True)[1].view(bz, -1).cpu() - self.var_start_idx
        return int_s_batch

    def share_memory(self):
        for level in range(len(self.counts)):
            if level == 0:
                self.counts[level].share_memory_()
            elif level == len(self.counts) - 1:
                self.counts[level][1].share_memory_()
            else:
                for target in range(len(self.counts[level])):
                    self.counts[level][target].share_memory_()









