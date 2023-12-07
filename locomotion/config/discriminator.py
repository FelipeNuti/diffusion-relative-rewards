import socket

from diffuser.utils import watch

#------------------------ base ------------------------#

## automatically make experiment names for planning
## by labelling folders with these args

discriminator_args_to_watch = [
    ('prefix', ''),
    ('model_horizon', 'H'),
    ('train_dataset', 'D'),
    ('dim_mults', 'DIMS'),
    ('model_class', "ARCH"),
    ('description', '')
]


base = {

    "discriminator": {
        "model_class": "models.SingleStepRewardMLP",
        # "discriminator_class": "rrf_diffusion.GradientRewardRegressor",

        "s_expert_path": "f:{logbase}/{dataset}/diffusion/defaults_H{model_horizon}_T{n_diffusion_steps}",
        "s_general_path": None,
        "expert_buffer_path": None,
        "general_buffer_path": None,
        "diffusion_predicts_mean": True,
        "alpha": 1e-3,
        'clip_denoised': True,
        'bias': True,
        "gradient_clipping": None,
        "weight_clipping": None,
        "eps_loss": 0,
        "model_horizon": None,
        "n_diffusion_steps": 20,
        "dim": 4,
        "embed_dim": 32,
        "dim_mults": "8,4,2,1",
        "stride": 1,
        "kernel_size": 5,
        "scale_scores": False,
        "train_frac": 0.9,

        "tag": None,
        "noise_level": None,
        "noise_seed": None,

        "dataset_name": None,
        "expert_epoch": "latest",
        "general_epoch": "latest",
        'half_performance_epoch': "latest",

        "activation": "leaky_relu",
        "norm": "none",
        "seed": 0,
        'renderer': 'utils.MuJoCoRenderer',

        ## heatmap dataset
        'train_dataset': 'mixed',
        'loader': 'datasets.GoalDataset',
        'h5path': None,
        'termination_penalty': None,
        'normalizer': 'LimitsNormalizer',
        'preprocess_fns': [],
        'use_padding': False,
        'max_path_length': 40000,
        'heatmap_horizon': 64,
        'heatmap_max_path_length': 40000,
        'heatmap_buffer_path': None,
        'shape_factor_heatmap': 1,

        'description': 'base',

        ## serialization
        'logbase': './locomotion_diffusion_logs/logs',
        'prefix': 'discriminator/',
        'exp_name': watch(discriminator_args_to_watch),

        ## training
        'n_steps_per_epoch': 10000,
        'loss_type': 'l2',
        'n_train_steps': 2e6,
        'batch_size': 32,
        'learning_rate': 2e-4,
        'gradient_accumulate_every': 1,
        'l2_reg': None,
        'ema_decay': 0.995,
        'save_freq': 24998,
        'sample_freq': 2500,
        'render_freq': 2500,
        'num_samples': 25,
        'num_trajectories_heatmap': 1000,
        'log_freq': 100,
        'n_saves': 4,
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

halfcheetah_expert_v2 = halfcheetah_medium_expert_v2 = halfcheetah_medium_v2 = halfcheetah_medium_replay_v2 = halfcheetah_random_v2 = {"discriminator": {
    'dim': 4,
    'model_horizon': 4,
    'n_diffusion_steps': 20,
    'embed_dim': 32,
    'half_performance_epoch': 12000,
}}

hopper_expert_v2 = hopper_medium_expert_v2 = hopper_medium_v2 = hopper_medium_replay_v2 = hopper_random_v2 = {"discriminator": {
    'dim': 4,
    'model_horizon': 32,
    'n_diffusion_steps': 20,
    'embed_dim': 32,
    'half_performance_epoch': 12000,
}}

walker2d_expert_v2 = walker2d_medium_expert_v2 = walker2d_medium_v2 = walker2d_medium_replay_v2 = walker2d_random_v2 = {"discriminator": {
    'dim': 4,
    'model_horizon': 32,
    'n_diffusion_steps': 20,
    'embed_dim': 32,
    'half_performance_epoch': 16000,
}}

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