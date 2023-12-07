import os
import argparse
import json
import subprocess

parser = argparse.ArgumentParser(description='A simple script with two arguments.')

parser.add_argument('--dataset', help='One of halfcheetah, hopper, or walker2d')
parser.add_argument('--action', help='train_expert, train_base, train_rrf, or rollouts')

args = parser.parse_args()

commands_path = os.path.abspath('./experiment_commands/main_results.json')

with open(commands_path) as f:
    commands = json.load(f)

cmd = commands[args.dataset][args.action]
print(cmd)
subprocess.call(cmd, shell = True)