import diffuser.utils as utils
import argparse
import pathlib
import os 

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type = str, default = '', help = 'Dataset path, e.g. maze2d-umaze-v1')
parser.add_argument('--dataset_size', type = str, default = '', help = 'Dataset size, e.g. 10m, 10k, 1m')
parser.add_argument('--goal', type = str, default = '', help = 'Name of goal, e.g. topleft, general')
parser.add_argument('--version', type = str, default = '', help = 'Version being used, e.g. v1, v2')
parser.add_argument('--horizon', type = int, default = 256, help = 'Horizon for the diffusion model. Needs to be a power of 2')
parser.add_argument('--datadir', type = str, default = '/scratch/shared/beegfs/<username>/d4rl_datasets', help = "Directory containing folder with data for experiment.")
parser.add_argument('--savedir', type = str, default = '/scratch/shared/beegfs/<username>/d4rl_preprocessed_buffers', help = "Directory in which to save the buffer.")
args = parser.parse_args()

pathlib.Path(os.path.abspath(args.savedir)).mkdir(exist_ok = True, parents = True)
save_path = f"{args.savedir}/{args.dataset}_{args.dataset_size}_{args.version}_{args.goal}.pkl"
h5path = f"{args.datadir}/{args.dataset}_{args.dataset_size}_{args.version}/{args.dataset}_{args.goal}.hdf5"
new_dataset_config = utils.Config(
    'datasets.GoalDataset',
    savepath=('/tmp', 'dataset_config.pkl'),
    env=args.dataset,
    horizon=args.horizon,
    normalizer='LimitsNormalizer',
    preprocess_fns=['maze2d_set_terminals'],
    use_padding=False,
    max_path_length=1000,
    max_n_episodes=300000,
    h5path=h5path,
)
new_dataset = new_dataset_config()
new_dataset.clean_fields(for_saving = True)
new_dataset.fields.save(save_path)