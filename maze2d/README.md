# Installation

Download mujoco200 from https://www.roboti.us/download.html, extract it and copy it to `~/.mujoco/mujoco200`.
Download mujoco key file from https://www.roboti.us/file/mjkey.txt and add it to "~/.mujoco"

Add `~/.mujoco/mujoco200` to `LD_LIBRARY_PATH`:
```
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/users/<username>/.mujoco/mujoco200/bin" >> ~/.bashrc
echo "export CPATH=$CONDA_PREFIX/include:/users/<username>/.mujoco/mujoco200/include" >> ~/.bashrc
```

<!-- conda env create -f environment.yml -->
```
conda create --name maze_env_diffuser python=3.8
conda activate maze_env_diffuser

pip install setuptools==65.5.0
pip install torch==1.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
pip install -e .
```

Now install further required libraries:

```
source ~/.bashrc
conda activate maze_env_diffuser

conda install -c conda-forge glew 
conda install -c conda-forge mesalib 
conda install -c conda-forge glfw
conda install -c anaconda pyopengl

CPATH=$CPATH:~/anaconda3/envs/maze_env_diffuser/include pip install mujoco-py==2.0.2.5

pip install gym==0.18.0
```
<!-- pip install mujoco-py==2.0.2.13 -->

Finally, as we use `wandb` for the logging, make sure to login to Weights and Biases CLI and create a project named `diffusion_relative_rewards`.

# Downloading the data

To download the rollouts for training the diffusion models and relative reward functions, run:

```
chmod +x download_extract_dataset.sh
./download_extract_dataset.sh
```

The folder structure of the extracted file should look like:
```
d4rl_datasets/
├── maze2d-large-v1_10m_v1
├── maze2d-medium-v1_10m_v1
├── maze2d-open-v0_10m_v1
└── maze2d-umaze-v0_10m_v1
```

# Run main experiments from the paper

The script `run.py` takes care of running the correct command for each experiment. The commands themselves are in `maze2d/diffusion_training_commands` and `maze2d/experiment_commands`. For example, to run the experiments for the `maze2d-large-v1` maze with goal position 1, run the following, in this order:

```
# Run from the locomotion directory

# Preprocess the buffers if you haven't already
python run.py --dataset maze2d-large-v1  --goal goal1 --mode buffer --to_train base
python run.py --dataset maze2d-large-v1  --goal goal1 --mode buffer --to_train expert

# Train base model
python run.py --dataset maze2d-large-v1  --goal goal1 --mode diffusion --to_train base

# Train expert model
python run.py --dataset maze2d-large-v1  --goal goal1 --mode diffusion --to_train expert

# Once the expert and base models are trained, train the relative reward function
python run.py --dataset maze2d-large-v1  --goal goal1 --mode rrf
```

Observation: the base model is referenced as "general model" throughout the codebase.


