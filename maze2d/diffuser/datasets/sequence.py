from collections import namedtuple
import numpy as np
import torch
import pdb

from .preprocessing import get_preprocess_fn
from .d4rl import load_environment, sequence_dataset
from .normalization import DatasetNormalizer
from .buffer import ReplayBuffer

Batch = namedtuple('Batch', 'trajectories conditions')
ValueBatch = namedtuple('ValueBatch', 'trajectories conditions values')

class SequenceDataset(torch.utils.data.Dataset):

    def __init__(self, env='hopper-medium-replay', horizon=64, fields_load_path = None,
        normalizer='LimitsNormalizer', preprocess_fns=[], max_path_length=1000,
        max_n_episodes=10000, termination_penalty=0, use_padding=True, h5path=None):
        self.preprocess_fn = get_preprocess_fn(preprocess_fns, env)
        self.env = env = load_environment(env)
        self.horizon = horizon
        self.max_path_length = max_path_length
        self.use_padding = use_padding

        fields = ReplayBuffer(max_n_episodes, max_path_length, termination_penalty)
        if fields_load_path is None:
            itr = sequence_dataset(env, self.preprocess_fn, h5path=h5path)
            for i, episode in enumerate(itr):
                fields.add_path(episode)
            fields.remove_extra_slots()
        else:
            fields.load(fields_load_path)
            self.max_path_length = fields.max_path_length

        self.observation_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.fields = fields
        self.n_episodes = fields.n_episodes
        self.path_lengths = fields["path_lengths"]

        self.normalizer = DatasetNormalizer(
            fields, 
            normalizer,
            self.observation_dim,
            self.action_dim,
            path_lengths=fields['path_lengths']
        )
        self.indices = self.make_indices(self.path_lengths, horizon)
        self.normalize()

        fields._add_attributes()
        print(fields)

    def used_fields(self, for_saving = False):
        if for_saving:
            return [
                "path_lengths",
                "observations",
                "trajectories",
            ]
        else:
            return [
                "path_lengths",
                "normed_trajectories"
            ]

    def clean_fields(self, for_saving = False):
        self.fields.clean_unused(self.used_fields(for_saving = for_saving))

    def normalize(self, keys=['observations', 'actions', "trajectories"]):
        '''
            normalize fields that will be predicted by the diffusion model
        '''
        for key in keys:
            if key not in self.fields.keys:
                continue
            array = self.fields[key].reshape(self.n_episodes*self.max_path_length, -1)
            normed = self.normalizer(array, key)
            self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_path_length, -1)
    
    def renormalize(self, new_normalizer, keys = ["trajectories"]):
        """
        Renormalize trajectories with new normalizer
        """
        for key in keys:
            normed_key = f"normed_{key}"
            array = self.fields[normed_key].reshape(self.n_episodes*self.max_path_length, -1)
            unnormed = self.normalizer.unnormalize(array, key)
            renormed = new_normalizer.normalize(unnormed, key)
            self.fields[normed_key] = renormed.reshape(self.n_episodes, self.max_path_length, -1)
        
        self.normalizer = new_normalizer
        self.fields._add_attributes()

    def _compute_max_start(self, path_length, horizon):
        max_start = min(path_length, self.max_path_length - horizon)
        if not self.use_padding:
            max_start = max(min(max_start, path_length - horizon), 1)
        return max_start

    def _compute_all_max_starts(self, path_lengths, horizon):
        max_starts = np.zeros((len(path_lengths, )), dtype = int)

        for i, path_length in enumerate(path_lengths):
            max_starts[i] = self._compute_max_start(path_length, horizon)

        return np.array(max_starts)
    
    def _compute_idx_repeats(self, max_starts, most_subtrajectories):
        return np.maximum(1, most_subtrajectories // max_starts)
    
    def make_indices(self, path_lengths, horizon):
        '''
            makes indices for sampling from dataset;
            each index maps to a datapoint
        '''
        indices = []
        max_starts = self._compute_all_max_starts(path_lengths, horizon)
        most_subtrajectories = max_starts.max()

        idx_repeats = self._compute_idx_repeats(max_starts, most_subtrajectories)
        indices = np.zeros(((max_starts * idx_repeats).sum(), 3), dtype = int)
        positions = np.arange(0, most_subtrajectories, dtype = int)
        idx = 0
        for i, path_length in enumerate(path_lengths):
            max_start = max_starts[i]
            for _ in range(idx_repeats[i]):
                indices[idx:idx+max_start, 0] = i
                indices[idx:idx+max_start, 1] = positions[:max_start]
                indices[idx:idx+max_start, 2] = positions[:max_start] + horizon
                idx += max_start
        return indices

    def get_conditions(self, trajectories):
        '''
            condition on current observation for planning
        '''
        return {0: trajectories[0, :, self.action_dim:]}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]

        trajectories = self.fields.normed_trajectories[path_ind, start:end]

        conditions = self.get_conditions(trajectories)
        batch = Batch(trajectories, conditions)
        return batch

    def to(self, device = "cuda:0"):
        self.normalizer = self.normalizer.to(device = device)
        self.fields = self.fields.to(device = device)

        return self

class GoalDataset(SequenceDataset):

    def get_conditions(self, observations):
        '''
            condition on both the current observation and the last observation in the plan
        '''
        return {
            0: observations[0, self.action_dim:],
            self.horizon - 1: observations[-1, self.action_dim:],
        }

class ValueDataset(SequenceDataset):
    '''
        adds a value field to the datapoints for training the value function
    '''

    def __init__(self, *args, discount=0.99, **kwargs):
        super().__init__(*args, **kwargs)
        self.discount = discount
        self.discounts = self.discount ** torch.arange(self.max_path_length)[:,None]

    def used_fields(self, **kwargs):
        return [*super().used_fields(**kwargs), "rewards"]

    def __getitem__(self, idx):
        batch = super().__getitem__(idx)
        path_ind, start, end = self.indices[idx]
        rewards = self.fields['rewards'][path_ind, start:]
        discounts = self.discounts[:len(rewards)]
        value = (discounts * rewards).sum()
        value = torch.array([value], dtype=torch.float32)
        value_batch = ValueBatch(*batch, value)
        return value_batch
