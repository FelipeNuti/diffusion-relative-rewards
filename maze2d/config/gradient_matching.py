import socket

from diffuser.utils import watch

#------------------------ base ------------------------#

## automatically make experiment names for planning
## by labelling folders with these args

gradient_matching_args_to_watch = [
    ('prefix', ''),
]


base = {

    "gradient_matching": {
        "model_class": "models.SingleStepRewardMLP",
        "gradient_matching_class": "rrf_diffusion.GradientRewardSingleStep",

        "dataset_size": "10m",
        "goal": "unspecified",
        "buffer_dir": None,
        "version": "v0",

        "diffusion_spec_string": "H256_T100",

        "s_expert_path": 'f:{goal}/diffusion/{diffusion_spec_string}',
        "s_general_path": 'f:general/diffusion/{diffusion_spec_string}',
        "expert_buffer_path": 'f:{buffer_dir}/{dataset}_{dataset_size}_{version}_{goal}.pkl',
        "general_buffer_path": 'f:{buffer_dir}/{dataset}_{dataset_size}_{version}_general.pkl',
        "diffusion_predicts_mean": True,
        "alpha": 1e-3,
        'clip_denoised': True,
        "gradient_clipping": 1e8,
        "eps_loss": 0,
        "model_horizon": None,
        "dim": 4,
        "embed_dim": 32,
        "dim_mults": "8,4,2,1",
        "stride": 1,
        "kernel_size": 5,
        "scale_scores": False,
        "train_frac": 0.9,

        "activation": "leaky_relu",
        "seed": 0,
        'renderer': 'utils.Maze2dClassificationRenderer',

        ## heatmap dataset
        'train_dataset': 'expert',
        'loader': 'datasets.GoalDataset',
        'h5path': None,
        'termination_penalty': None,
        'normalizer': 'LimitsNormalizer',
        'preprocess_fns': ['maze2d_set_terminals'],
        'use_padding': False,
        'max_path_length': 40000,
        'heatmap_horizon': 64,
        'heatmap_max_path_length': 40000,
        'heatmap_dataset_size': "10m",
        'heatmap_buffer_path': 'f:{buffer_dir}/{dataset}_{heatmap_dataset_size}_{version}_general.pkl',
        'shape_factor_heatmap': 10,

        ## serialization
        'logbase': 'logs',
        'prefix': 'gradient_matching/',
        'exp_name': watch(gradient_matching_args_to_watch, wandb_run=True),

        ## training
        'n_steps_per_epoch': 10000,
        'loss_type': 'l2',
        'n_train_steps': 2e6,
        'batch_size': 256,
        'learning_rate': 2e-4,
        'gradient_accumulate_every': 1,
        'ema_decay': 0.995,
        'save_freq': 1000,
        'sample_freq': 1000,
        'render_freq': 1000,
        'num_samples': 20,
        'num_trajectories_heatmap': 10000,
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
