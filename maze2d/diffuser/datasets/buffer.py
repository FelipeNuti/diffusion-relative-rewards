import numpy as np
import torch

def atleast_2d(x):
    while x.ndim < 2:
        x = torch.unsqueeze(x, dim = -1)
    return x

class ReplayBuffer:

    def __init__(self, max_n_episodes, max_path_length, termination_penalty):
        self._dict = {
            'path_lengths': torch.zeros(max_n_episodes, dtype=torch.int),
        }
        self._count = 0
        self.max_n_episodes = max_n_episodes
        self.max_path_length = max_path_length
        self.termination_penalty = termination_penalty

    def __repr__(self):
        return '[ datasets/buffer ] Fields:\n' + '\n'.join(
            f'    {key}: {val.shape}'
            for key, val in self.items()
        )

    def __getitem__(self, key):
        return self._dict[key]

    def __setitem__(self, key, val):
        self._dict[key] = val
        self._add_attributes()

    @property
    def n_episodes(self):
        return self._count

    @property
    def n_steps(self):
        return sum(self['path_lengths'])

    def _add_keys(self, path):
        if hasattr(self, 'keys'):
            return
        self.keys = list(path.keys())
        if "trajectories" not in self.keys:
            self.keys.append("trajectories")

    def _add_attributes(self):
        '''
            can access fields with `buffer.observations`
            instead of `buffer['observations']`
        '''
        for key, val in self._dict.items():
            setattr(self, key, val)

    def items(self):
        return {k: v for k, v in self._dict.items()
                if k != 'path_lengths'}.items()

    def _allocate(self, key, array):
        assert key not in self._dict
        dim = array.shape[-1]
        shape = (self.max_n_episodes, self.max_path_length, dim)
        self._dict[key] = torch.zeros(shape, dtype=torch.float32)

    def add_path(self, path):
        path_length = len(path['observations'])
        assert path_length <= self.max_path_length
        if path_length <= 1:
            return
        self._add_keys(path)
        for key in path.keys():
            array = atleast_2d(torch.tensor(path[key]))
            if key not in self._dict: self._allocate(key, array)
            self._dict[key][self._count, :path_length] = array

            if path_length < self.max_path_length and path_length > 0 and key == "observations":
                self._dict[key][self._count, path_length:] = array[-1]

        traj = torch.cat(
            [self._dict["actions"][self._count, :], self._dict["observations"][self._count, :]], 
            dim=-1
        )

        if "trajectories" not in self._dict: self._allocate("trajectories", traj)
        self._dict["trajectories"][self._count, :] = traj
        if path['terminals'].any() and self.termination_penalty is not None:
            assert not path['timeouts'].any(), 'Penalized a timeout episode for early termination'
            self._dict['rewards'][self._count, path_length - 1] += self.termination_penalty
        self._dict['path_lengths'][self._count] = path_length
        self._count += 1

    def truncate_path(self, path_ind, step):
        old = self._dict['path_lengths'][path_ind]
        new = min(step, old)
        self._dict['path_lengths'][path_ind] = new

    def remove_extra_slots(self):
        for key in self.keys + ['path_lengths']:
            self._dict[key] = self._dict[key][:self._count]

    def finalize(self):
        ## remove extra slots
        self.remove_extra_slots()
        self._add_attributes()
        print(f'[ datasets/buffer ] Finalized replay buffer | {self._count} episodes')

    def clean_unused(self, used):
        new_d = {}
        new_keys = []
        for k in self._dict.keys():
            if k in used:
                new_d[k] = self._dict[k]
                new_keys.append(k)
            elif hasattr(self, k):
                delattr(self, k)
        self._dict = new_d
        self.keys = new_keys
        self._add_attributes()

    def save(self, path):
        torch.save({
            "_dict": self._dict,
            "keys": self.keys,
            "count": self._count,
            "max_n_episodes": self.max_n_episodes,
            "max_path_length": self.max_path_length,
            "termination_penalty": self.termination_penalty,
        }, path)

    def load(self, path):
        saved = torch.load(path)
        self._dict = saved["_dict"]
        self.keys = saved["keys"]
        self._count = saved["count"]
        self.max_n_episodes = saved["max_n_episodes"]
        self.max_path_length = saved["max_path_length"]
        self.termination_penalty = saved["termination_penalty"]

        self._add_attributes()

    def to(self, device = "cuda:0"):
        for k, v in self._dict.items():
            self._dict[k] = v.to(device = device)
            setattr(self, k, self._dict[k])
        return self
