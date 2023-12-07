from collections import namedtuple
import torch
import einops
import pdb

import diffuser.utils as utils
from diffuser.datasets.preprocessing import get_policy_preprocess_fn


Trajectories = namedtuple('Trajectories', 'actions observations values')


class GuidedPolicy:

    def __init__(self, guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs):
        self.guide = guide
        self.diffusion_model = diffusion_model
        self.normalizer = normalizer
        self.action_dim = diffusion_model.action_dim
        self.preprocess_fn = get_policy_preprocess_fn(preprocess_fns)
        self.sample_kwargs = sample_kwargs

    def __call__(self, conditions, batch_size=1, verbose=True):
        conditions = {k: self.preprocess_fn(v) for k, v in conditions.items()}
        conditions = self._format_conditions(conditions, batch_size)

        ## run reverse diffusion process
        samples = self.diffusion_model(conditions, guide=self.guide, verbose=verbose, **self.sample_kwargs)
        trajectories = utils.to_np(samples.trajectories)

        ## extract action [ batch_size x horizon x transition_dim ]
        actions = trajectories[:, :, :self.action_dim]
        actions = self.normalizer.unnormalize(actions, 'actions')

        ## extract first action
        action = actions[0, 0]

        normed_observations = trajectories[:, :, self.action_dim:]
        observations = self.normalizer.unnormalize(normed_observations, 'observations')

        trajectories = Trajectories(actions, observations, samples.values)
        return action, trajectories

    @property
    def device(self):
        parameters = list(self.diffusion_model.parameters())
        return parameters[0].device

    def _format_conditions(self, conditions, batch_size):
        conditions = utils.apply_dict(
            self.normalizer.normalize,
            conditions,
            'observations',
        )
        conditions = utils.to_torch(conditions, dtype=torch.float32, device='cuda:0')
        conditions = utils.apply_dict(
            einops.repeat,
            conditions,
            'd -> repeat d', repeat=batch_size,
        )
        return conditions
    
class GuidedPolicyParallel:

    def __init__(self, n_processes, guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs):
        self.n_processes = n_processes
        self.guide = guide
        self.diffusion_model = diffusion_model
        self.normalizer = normalizer
        self.action_dim = diffusion_model.action_dim
        self.preprocess_fn = get_policy_preprocess_fn(preprocess_fns)
        self.sample_kwargs = sample_kwargs

    def __call__(self, conditions, batch_size=1, verbose=True):
        conditions = {k: self.preprocess_fn(v) for k, v in conditions.items()}
        conditions = self._format_conditions(conditions, batch_size)

        ## run reverse diffusion process
        samples = self.diffusion_model(conditions, guide=self.guide, verbose=verbose, return_sorted = False, **self.sample_kwargs)
        actions, trajectories = self._format_outputs(samples.trajectories, samples.values)

        ## extract first action
        action = actions[:, 0, 0, :]

        return action, trajectories

    @property
    def device(self):
        parameters = list(self.diffusion_model.parameters())
        return parameters[0].device

    def _format_conditions(self, conditions, batch_size):
        conditions = utils.apply_dict(
            self.normalizer.normalize,
            conditions,
            'observations',
        )
        conditions = utils.to_torch(conditions, dtype=torch.float32, device='cuda:0')
        conditions = utils.apply_dict(
            einops.repeat,
            conditions,
            'n d -> (n repeat) d', repeat=batch_size,
        )
        return conditions

    def _format_outputs(self, normed_trajectories, values):
        normed_trajectories = utils.to_np(normed_trajectories)

        normed_actions = normed_trajectories[:, :, :self.action_dim]
        normed_observations = normed_trajectories[:, :, self.action_dim:]

        actions = self.normalizer.unnormalize(normed_actions, 'actions')
        observations = self.normalizer.unnormalize(normed_observations, 'observations')

        actions = einops.rearrange(actions, "(n r) h d -> n r h d", n = self.n_processes)
        observations = einops.rearrange(observations, "(n r) h d -> n r h d", n = self.n_processes)
        values = einops.rearrange(values, "(n r) -> n r", n = self.n_processes)

        trajectories = Trajectories(actions, observations, values)
        return actions, trajectories

