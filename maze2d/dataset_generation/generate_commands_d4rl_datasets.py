import os
import json

targets_named = {
    "maze2d-large-v1": {
        "bottomleft": [7, 1],
        "bottomright": [7, 10],
        "topleft": [1, 1],
        "topright": [1, 10],
        "open1": [3, 6],
        "open2": [5, 4],
        "open3": [5, 2],
        "open4": [3, 8],
    },
    "maze2d-medium-v1": {
        "bottomleft": [6, 1],
        "bottomright": [6, 6],
        "topleft": [1, 1],
        "topright": [1, 6],
        "open1": [2, 2],
        "open2": [4, 6],
        "open3": [5, 3],
        "open4": [4, 4],
    },
    "maze2d-umaze-v0": {
        "bottomleft": [3, 1],
        "bottomright": [3, 3],
        "topleft": [1, 1],
        "topright": [1, 3],
        "open1": [1, 2],
        "open2": [2, 3],
        "open3": [3, 2],
        "open4": [1.3, 2.7],
    },
    "maze2d-open-v0": {
        "bottomleft": [3, 1],
        "bottomright": [3, 5],
        "topleft": [1, 1],
        "topright": [1, 5],
        "open1": [2, 3],
        "open2": [1.3, 1.7],
        "open3": [2.7, 3.1],
        "open4": [2.3, 4.3],
    }
}

targets = {
    env: {
        f"goal{i+1}": pos for i, pos in enumerate(sorted(d.values()))
    } for env, d in targets_named.items()
}

# print("#"*50)
# print("targets: ")
# print(json.dumps(targets, indent=1))
# print("#"*50)

def many_dataset_generation_commands(datasets, *args, **kwargs):
    for dataset in datasets:
        dataset_generation_commands(dataset, *args, **kwargs)

def dataset_generation_commands(
    dataset,
    version,
    dataset_size,
    max_steps = 1000,
    num_samples = 1e6, 
    wall_dist_thresh = 0.26, 
    print_only = None, 
    base_save_dir = ".", 
    parts = None
):
    num_samples = int(num_samples)
    to_print = []
    save_dir = os.path.join(base_save_dir, f"{dataset}_{dataset_size}_{version}")

    s_slurm = ""
    s_base = f"python scripts/generation/generate_maze2d_datasets.py --env_name {dataset} --savedir \"{save_dir}\" --num_samples {num_samples} "
    s_extras = f"--print_spec True  --override_max_episode_steps {max_steps} --wall_dist_thresh {wall_dist_thresh} "
    if "open" in dataset:
        s_extras += "--openspace True "
    else:
        s_extras += " --fill_square "

    if parts is None:
        for name, target in targets[dataset].items():
            s_goal = f"--keep_target True --set_target \"{','.join([str(x) for x in target])}\" --description \"{name}\" "
            s = s_slurm + s_base + s_goal + s_extras
            to_print.append(s)
        s_general = f"--description \"general\"  --reset_when_reached True "
        to_print.append(s_slurm + s_base + s_general + s_extras)
    else:
        for part in range(1, parts+1):
            for name, target in targets[dataset].items():
                if print_only is not None and name not in print_only:
                    continue
                s_base = f"python scripts/generation/generate_maze2d_datasets.py --env_name {dataset} --savedir \"{save_dir}/{name}\" --num_samples {num_samples} "
                s_goal = f"--keep_target True --set_target \"{','.join([str(x) for x in target])}\" --description \"{name}_pt_{part}\" "
                s = s_slurm + s_base + s_goal + s_extras
                to_print.append(s)
            if print_only is None or "general" in print_only:
                s_base = f"python scripts/generation/generate_maze2d_datasets.py --env_name {dataset} --savedir \"{save_dir}/general\" --num_samples {num_samples} "
                s_general = f"--description \"general_pt_{part}\"  --reset_when_reached True "
                s = s_slurm + s_base + s_general + s_extras
                to_print.append(s)
    to_print = sorted(to_print)
    for s in to_print:
        print(s)
        print()

many_dataset_generation_commands(
    ["maze2d-open-v0", "maze2d-umaze-v0", "maze2d-medium-v1", "maze2d-large-v1"],
    "v1",
    "10m",
    num_samples=1e7,
    parts = None,
    base_save_dir="./d4rl_datasets"
)