import typing as t
import os 

import sys, os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from rrf_diffusion import ValueUnet, GradientRewardRegressor, GradientMatchingTrainer
from rrf_diffusion.dataset import batch_to_device
import rrf_diffusion.utils as utils

import torch
import wandb
# import dill as pickle
from tqdm import tqdm

import argparse

"""
Example command:
python ./scripts/train_gradient_matching.py --batch_size 64 --dim 8
"""

#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
parser.add_argument('--gradient_clipping', type=float, default=None, help='max absolute value for each gradient element')
parser.add_argument('--sample_freq', type=int, default=1000, help='sampling frequency')
parser.add_argument('--n_train_steps', type=int, default=1000000, help='number of training steps')
parser.add_argument('--n_steps_per_epoch', type=int, default=1000, help='number of steps per epoch')
parser.add_argument('--train_frac', type=float, default=0.9, help='fraction of data assigned for training')
parser.add_argument('--dim', type=int, default=4, help="multiplier of ValueUnet layer dimensions")
parser.add_argument('--seed', type=int, default=0, help="seed")
parser.add_argument('--debug', type=bool, default=False, help="Put wandb in disabled mode")
parser.add_argument('--test_overfit', type=bool, default=False, help="Test model ability to overfit to a few datapoints")
parser.add_argument('--base_path', type=str, default='./safe_stable_diffusion_data/train', help='directory with training data')


args = parser.parse_args()

utils.set_seed(args.seed)

print(args.__dict__)

base_path = args.base_path
load_file = "merged.pt" if not args.debug else "latents_0_100.pt"

print("Loading datasets (expert)...")
expert_dataset = torch.load(os.path.join(base_path, f"expert/{load_file}"), map_location=torch.device("cpu"))
print("Loading datasets (general)...")
general_dataset = torch.load(os.path.join(base_path, f"general/{load_file}"), map_location=torch.device("cpu"))
print("Finished loading")

model = ValueUnet(args.dim, dim_mults=(1, 2, 4, 8), channels = 4, resnet_block_groups=8)
gradient_matching = GradientRewardRegressor(model)

utils.report_parameters(model)

savepath = "./safe_stable_diffusion_logs"
debug_str = "" if not args.debug else "_debug"
savepath = os.path.join(savepath, f"lr{args.lr}_dim{args.dim}_seed{args.seed}{debug_str}/")

os.makedirs(savepath, exist_ok=True)

device = "cuda:0"
model = model.to(device = device)
gradient_matching = gradient_matching.to(device = device)

trainer = GradientMatchingTrainer(
    gradient_matching, 
    expert_dataset, 
    general_dataset,
    train_lr=args.lr,
    gradient_clipping=args.gradient_clipping,
    train_batch_size=args.batch_size,
    sample_freq=args.sample_freq,
    train_frac=args.train_frac,
    test_overfit=args.test_overfit,
    results_folder=savepath,
)
print('Testing forward...', end=' ', flush=True)
batch = next(trainer.dataloader)
batch = batch_to_device(batch)
loss, info = gradient_matching.loss(*batch)
loss.backward()
print('✓')

wandb.init(
    project = "diffusion_relative_rewards", 
    mode = "online" if not args.debug else "disabled",
    config = args.__dict__
)


#-----------------------------------------------------------------------------#
#--------------------------------- main loop ---------------------------------#
#-----------------------------------------------------------------------------#

n_epochs = int(args.n_train_steps // args.n_steps_per_epoch)

for i in range(n_epochs):
    print(f'Epoch {i} / {n_epochs} | {savepath}')
    trainer.train(n_train_steps=args.n_steps_per_epoch)







