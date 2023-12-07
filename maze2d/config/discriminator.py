import socket

from diffuser.utils import watch

#------------------------ base ------------------------#

## automatically make experiment names for planning
## by labelling folders with these args

discriminator_args_to_watch = [
    ('prefix', ''),
]


base = {

    "discriminator": {
        "s_expert_path": None,
        "s_general_path": None,
        "expert_buffer_path": None,
        "general_buffer_path": None,
        "alpha": 1e-3,
        "dim": 8,
        "embed_dim": 8,
        "dim_mults": (8, 4, 2, 1),
        "stride": 1,
        "kernel_size": 3,
        "train_frac": 0.9,

        "seed": 0,

        ## value dataset
        'loader': 'datasets.ValueDataset',
        'h5path': None,
        'termination_penalty': None,
        'normalizer': 'LimitsNormalizer',
        'preprocess_fns': ['maze2d_set_terminals'],
        'clip_denoised': True,
        'use_padding': False,
        'max_path_length': 40000,

        ## serialization
        'logbase': 'logs',
        'prefix': 'discriminator/',
        'exp_name': watch(discriminator_args_to_watch),

        ## training
        'n_steps_per_epoch': 10000,
        'loss_type': 'l2',
        'n_train_steps': 2e6,
        'batch_size': 32,
        'learning_rate': 2e-4,
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'save_freq': 1000,
        'sample_freq': 1000,
        'log_freq': 100,
        'n_saves': 50,
        'save_parallel': False,
        'bucket': None,
        'device': 'cuda',

    }

}

#------------------------ overrides ------------------------#

'''
    maze2d maze episode steps:
        umaze: 150
        medium: 250
        large: 600
'''

maze2d_umaze_v1 = {
    'diffusion': {
        'horizon': 128,
        'n_diffusion_steps': 64,
        
    },
    'plan': {
        'horizon': 128,
        'n_diffusion_steps': 64,
    },
}

maze2d_large_v1 = {
    'diffusion': {
        'horizon': 384,
        'n_diffusion_steps': 256,
    },
    'plan': {
        'horizon': 384,
        'n_diffusion_steps': 256,
    },
}
