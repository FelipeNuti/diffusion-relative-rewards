# %load_ext autoreload

import typing as t
import os 
import diffuser.utils as utils
from diffuser.models import ValueFunction, TemporalValue, SimpleMLPValue
from diffuser.utils.arrays import batch_to_device, to_np, to_device, apply_dict
from diffuser.utils.training import cycle
from diffuser.utils.rendering import MAZE_BOUNDS
import diffuser.rrf_diffusion.gradient_matching as gradient_matching
from diffuser.rrf_diffusion import GradientMatchingTrainer
import diffuser.utils as utils
import torch
import wandb
import dill as pickle
import copy
from tqdm import tqdm
import einops
import matplotlib.pyplot as plt
import numpy as np

expert_experiment = utils.load_diffusion(
    "/scratch/shared/beegfs/<username>/maze-experiments/v1/maze2d-large-v1/10k/goal1/diffusion/H256_T100",
    fields_load_path = "/scratch/shared/beegfs/<username>/d4rl_preprocessed_buffers/maze2d-large-v1_10k_v1_goal1.pkl"
)

general_experiment = utils.load_diffusion(
    "/scratch/shared/beegfs/<username>/maze-experiments/v1/maze2d-large-v1/10k/general/diffusion/H256_T100",
    fields_load_path = "/scratch/shared/beegfs/<username>/d4rl_preprocessed_buffers/maze2d-large-v1_10k_v1_general.pkl"
)
s_general = general_experiment.diffusion
general_dataset = general_experiment.dataset
expert_dataset = expert_experiment.dataset

gradient_matching_experiment = utils.serialization.load_gradient_matching(
    "/work/<username>/scratch_backup/maze-experiments/v1/maze2d-large-v1/10k/goal1/gradient_matching/apricot-sweep-23",
    dataset = general_dataset,
    diffusion = copy.deepcopy(s_general)
)

renderer = gradient_matching_experiment.renderer
gradient_matching = gradient_matching_experiment.model
trainer = gradient_matching_experiment.trainer
action_dim = general_dataset.action_dim
s_general.n_timesteps
expert_eval = cycle(torch.utils.data.DataLoader(
            expert_dataset, batch_size=1, num_workers=0, shuffle=True, pin_memory=False
        ))
general_eval = cycle(torch.utils.data.DataLoader(
            general_dataset, batch_size=1, num_workers=0, shuffle=True, pin_memory=False
        ))
wandb.init(mode = "disabled")
x_exp = next(expert_eval)
x_general = next(general_eval)
trainer.render_progressive_noising(x_exp.trajectories, x_general.trajectories, savepath = "/work/<username>/test4.png")