import os
import argparse
import json
import subprocess

parser = argparse.ArgumentParser(description='A simple script with two arguments.')

parser.add_argument('--dataset', help='One of maze2d-open-v0, maze2d-umaze-v0, maze2d-medium-v1, maze2d-large-v1')
parser.add_argument('--goal', help='One of goal1, goal2, . . ., goal8')
parser.add_argument('--mode', help='buffer -> preprocess buffer; diffusion -> train diffusion model; rrf -> train relative reward')
parser.add_argument('--to_train', default="", help='If training the diffusion model, specify whether the expert or base (general) should be trained')

args = parser.parse_args()

paths = {
    "diffusion": "./diffusion_training_commands",
    "rrf": "./experiment_commands",
    "buffer": "./dataset_generation"
}

commands_path = paths[args.mode]

with open(os.path.join(commands_path, f"{args.dataset}.json")) as f:
    commands = json.load(f)

key = args.goal

if args.mode in ["buffer", "diffusion"]: 
    if args.to_train == "base":
        key = "general"
    else:
        assert args.to_train == "expert", "to_train must be expert or base"


print(commands[key])
subprocess.call(commands[key], shell = True)