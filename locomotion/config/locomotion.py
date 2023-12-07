import socket

from diffuser.utils import watch

#------------------------ base ------------------------#

## automatically make experiment names for planning
## by labelling folders with these args

args_to_watch = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
    ## value kwargs
    ('discount', 'd'),
    ('description', '')
]

logbase = './locomotion_diffusion_logs/logs'

base = {
    'diffusion': {
        ## model
        'model': 'models.TemporalUnet',
        'diffusion': 'models.GaussianDiffusion',
        'horizon': 32,
        'n_diffusion_steps': 20,
        'action_weight': 10,
        'loss_weights': None,
        'loss_discount': 1,
        'predict_epsilon': False,
        'dim_mults': (1, 2, 4, 8),
        'attention': False,
        'renderer': 'utils.MuJoCoRenderer',

        ## dataset
        'loader': 'datasets.SequenceDataset',
        'normalizer': 'GaussianNormalizer',
        'preprocess_fns': [],
        'clip_denoised': False,
        'use_padding': True,
        'max_path_length': 1000,

        "tag": None,
        'noise_level': None,
        'noise_seed': None,

        ## serialization
        'logbase': logbase,
        'prefix': 'diffusion/defaults',
        'exp_name': watch(args_to_watch),

        'noise_level': 0.0,

        ## training
        'n_steps_per_epoch': 10000,
        'loss_type': 'l2',
        'n_train_steps': 1e6,
        'batch_size': 32,
        'learning_rate': 2e-4,
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'save_freq': 20000,
        'sample_freq': 20000,
        'n_saves': 5,
        'save_parallel': False,
        'n_reference': 8,
        'bucket': None,
        'device': 'cuda',
        'seed': None,
    },

    'values': {
        'model': 'models.ValueFunction',
        'diffusion': 'models.ValueDiffusion',
        'horizon': 32,
        'n_diffusion_steps': 20,
        'dim_mults': (1, 2, 4, 8),
        'renderer': 'utils.MuJoCoRenderer',

        ## value-specific kwargs
        'discount': 0.99,
        'termination_penalty': -100,
        'normed': False,

        ## dataset
        'loader': 'datasets.ValueDataset',
        'normalizer': 'GaussianNormalizer',
        'preprocess_fns': [],
        'use_padding': True,
        'max_path_length': 1000,

        "tag": None,
        'noise_level': None,
        'noise_seed': None,

        ## serialization
        'logbase': logbase,
        'prefix': 'values/defaults',
        'exp_name': watch(args_to_watch),

        ## training
        'n_steps_per_epoch': 10000,
        'loss_type': 'value_l2',
        'n_train_steps': 200e3,
        'batch_size': 32,
        'learning_rate': 2e-4,
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'save_freq': 1000,
        'sample_freq': 0,
        'n_saves': 5,
        'save_parallel': False,
        'n_reference': 8,
        'bucket': None,
        'device': 'cuda',
        'seed': None,
    },

    'plan': {
        'guide': 'sampling.ValueGuide',
        'policy': 'sampling.GuidedPolicy',
        'max_episode_length': 1000,
        'batch_size': 64,
        'preprocess_fns': [],
        'device': 'cuda',
        'seed': None,

        ## sample_kwargs
        'n_guide_steps': 2,
        'scale': 0.1,
        't_stopgrad': 2,
        'scale_grad_by_std': True,

        ## serialization
        'loadbase': None,
        'logbase': logbase,
        'prefix': 'plans/',
        'exp_name': watch(args_to_watch),
        'vis_freq': 100,
        'max_render': 8,

        ## diffusion model
        'horizon': 32,
        'n_diffusion_steps': 20,

        "tag": None,
        'noise_level': None,
        'noise_seed': None,

        ## value function
        'discount': 0.997,

        ## loading
        'diffusion_loadpath': 'f:{loadbase}/{dataset}/diffusion/defaults_H{horizon}_T{n_diffusion_steps}',
        'value_loadpath': 'f:{loadbase}/{dataset}/values/defaults_H{horizon}_T{n_diffusion_steps}_d{discount}',

        'description': None,

        'diffusion_epoch': 'latest',
        'value_epoch': 'latest',

        'verbose': True,
        'suffix': '0',
    },

    'plan-gradient-matching': {
        'value_class': "models.SingleStepRewardMLP",
        'guide': 'sampling.ValueGuide',
        'policy': 'sampling.GuidedPolicy',
        'max_episode_length': 1000,
        'batch_size': 64,
        'preprocess_fns': [],
        'device': 'cuda',
        'seed': 0,

        'value_horizon': 32,
        'value_train_dataset': 'mixed',
        'value_dim_mults': '1,4,8',

        ## sample_kwargs
        'n_guide_steps': 1,
        'scale': 0.1,
        't_stopgrad': 0,
        'scale_grad_by_std': False,

        "dataset_name": None,
        "tag": None,

        "gradient_clipping": None,
        "learning_rate": None,
        "weight_clipping": None,
        "l2_reg": None,

        ## serialization
        'loadbase': None,
        'logbase': logbase,
        'prefix': 'plans-gradient-matching/',
        'exp_name': watch(args_to_watch),
        'vis_freq': 998,
        'max_render': 1,

        'noise_level': 0.0,
        'noise_seed': None,

        ## diffusion model
        'horizon': 32,
        'n_diffusion_steps': 20,

        ## value function
        'discount': 0.99,

        'n_seed': 1,
        'n_processes': 1,

        ## loading
        'diffusion_loadpath': 'f:{loadbase}/{dataset}/diffusion/defaults_H{horizon}_T{n_diffusion_steps}',
        'value_loadpath': 'f:{loadbase}/{dataset}/gradient_matching/H{value_horizon}_D{value_train_dataset}_DIMS{value_dim_mults}_ARCH{value_class}{description}',

        'description': "",

        'diffusion_epoch': 'latest',
        'value_epoch': 'latest',

        'verbose': False,
        'suffix': 'f:{seed}',
    },
}


#------------------------ overrides ------------------------#


hopper_medium_expert_v2 = {
    'plan': {
        'scale': 0.0001,
        't_stopgrad': 4,
    },
    'plan-gradient-matching': {
        'dim': 4,
        'horizon': 4,
        'value_horizon': 4,
        'n_diffusion_steps': 20,
        'half_performance_epoch': 12000,
    }
}

hopper_expert_v2 = hopper_medium_v2 = hopper_medium_replay_v2 = hopper_random_v2 = {
    'plan-gradient-matching': {
        'dim': 4,
        'horizon': 32,
        'value_horizon': 32,
        'n_diffusion_steps': 20,
        'half_performance_epoch': 12000,
    }
}

walker2d_expert_v2 = walker2d_medium_expert_v2 = walker2d_medium_v2 = walker2d_medium_replay_v2 = walker2d_random_v2 = {
    'plan-gradient-matching': {
        'dim': 4,
        'horizon': 32,
        'value_horizon': 32,
        'n_diffusion_steps': 20,
        'half_performance_epoch': 16000,
    }
}

halfcheetah_medium_replay_v2 = halfcheetah_medium_v2 = halfcheetah_medium_expert_v2 = halfcheetah_random_v2 = halfcheetah_expert_v2 = {
    'diffusion': {
        'horizon': 4,
        'dim_mults': (1, 4, 8),
        'attention': True,
    },
    'values': {
        'horizon': 4,
        'dim_mults': (1, 4, 8),
    },
    'plan': {
        'horizon': 4,
        'scale': 0.001,
        't_stopgrad': 4,
    },
    'plan-gradient-matching': {
        'dim': 4,
        'horizon': 4,
        'value_horizon': 4,
        'n_diffusion_steps': 20,
        'half_performance_epoch': 12000,
    }
}
