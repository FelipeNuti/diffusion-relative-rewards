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
conda create --name locomotion_env_diffuser python=3.8
conda activate locomotion_env_diffuser

pip install setuptools==65.5.0
pip install torch==1.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
pip install -e .
```

Now install further required libraries:

```
source ~/.bashrc
conda activate locomotion_env_diffuser

conda install -c conda-forge glew 
conda install -c conda-forge mesalib 
conda install -c conda-forge glfw
conda install -c anaconda pyopengl

CPATH=$CPATH:~/anaconda3/envs/locomotion_env_diffuser/include pip install mujoco-py==2.0.2.5

pip install gym==0.18.0
```
<!-- pip install mujoco-py==2.0.2.13 -->

Finally, as we use `wandb` for the logging, make sure to login to Weights and Biases CLI and create a project named `diffusion_relative_rewards`.

# Run main experiments from the paper

The script `run.py` takes care of running the correct command for each experiment. The commands themselves are in `locomotion/experiment_commands/main_results.json`. For example, to run the experiments for the `halfcheetah`, run the following, in this order:

```
# Run from the maze2d directory

# Train base model
python run.py --dataset halfcheetah --action train_base

# Train expert model
python run.py --dataset halfcheetah --action train_expert

# Once the expert and base models are trained, train the relative reward function
python run.py --dataset halfcheetah --action train_rrf

# Run rollouts by steering the base model with the learned relative reward
python run.py --dataset halfcheetah --action rollouts
```

Observation: the base model is referenced as "general model" throughout the codebase.


