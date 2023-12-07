import pdb
import copy
import diffuser.sampling as sampling
import diffuser.utils as utils
import wandb
import time
import numpy as np


#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    dataset: str = 'walker2d-medium-replay-v2'
    config: str = 'config.locomotion'
    debug: bool = False

parser = Parser()
args = parser.parse_args('plan-gradient-matching')

#-----------------------------------------------------------------------------#
#---------------------------------- loading ----------------------------------#
#-----------------------------------------------------------------------------#
run = wandb.init(
    project = "diffusion_relative_rewards", 
    # entity="rrf-diffusion",
    # dir = "/scratch/shared/beegfs/<username>/wandb",
    mode = "online" if not args.debug else "disabled",
    tags = [f"rollout_gm_{args.tag}"] if args.tag is not None else [],
    config = args._dict
)

diffusion_experiment = utils.load_diffusion(
    args.diffusion_loadpath,
    epoch=args.diffusion_epoch, seed=args.seed,
)
diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer

value_experiment = utils.load_discriminator(
    args.value_loadpath,
    dataset = dataset,
    diffusion = copy.deepcopy(diffusion),
    epoch=args.value_epoch, seed=args.seed
)

## ensure that the diffusion model and value function are compatible with each other
utils.check_compatibility(diffusion_experiment, value_experiment)
value_function = value_experiment.ema
guide_config = utils.Config(args.guide, model=value_function, verbose=False)
guide = guide_config()

if abs(args.scale) < 1e-10:
    args.n_guide_steps = 0

## policies are wrappers around an unconditional diffusion model and a value guide
policy_config = utils.Config(
    "sampling.GuidedPolicyParallel",
    guide=guide,
    scale=args.scale,
    diffusion_model=diffusion,
    normalizer=dataset.normalizer,
    preprocess_fns=args.preprocess_fns,
    ## sampling kwargs
    sample_fn=sampling.n_step_guided_p_sample,
    n_guide_steps=args.n_guide_steps,
    t_stopgrad=args.t_stopgrad,
    scale_grad_by_std=args.scale_grad_by_std,
    n_processes=args.n_processes,
    verbose=False,
)
policy = policy_config()


#-----------------------------------------------------------------------------#
#--------------------------------- main loop ---------------------------------#
#-----------------------------------------------------------------------------#

logger_config = utils.Config(
    utils.Logger,
    renderer=renderer,
    logpath=args.savepath,
    vis_freq=args.vis_freq,
    max_render=args.max_render,
)
logger = logger_config()

obs_dim = dataset.env.observation_space.shape[0]
envs = []
rollouts = np.zeros((args.n_processes, args.max_episode_length+1, obs_dim))
total_rewards = np.zeros((args.n_processes,))
scores = np.zeros((args.n_processes,))
done = []
steps = []

for i in range(args.n_processes):
    env = copy.deepcopy(dataset.env)
    env.seed(int(args.seed) + i)
    observation = env.reset()

    envs.append(env)
    rollouts[i, 0, :] = observation.copy()
    done.append(False)
    steps.append(args.max_episode_length)

## observations for rendering

wandb.log({"run_state": 0}, commit = True)
for t in range(args.max_episode_length):
    if t % 100 == 0:
        wandb.log({"total_reward": total_rewards.mean(), "score": scores.mean()})

    if all(done):
        break

    conditions = {0: rollouts[:, t, :]}
    action, samples = policy(conditions, batch_size=args.batch_size, verbose=args.verbose)

    for i in range(args.n_processes):
        if done[i]:
            continue

        next_observation, reward, terminal, _ = envs[i].step(action[i])

        ## print reward and score
        total_rewards[i] += reward
        scores[i] = dataset.env.get_normalized_score(total_rewards[i])
        if i % 100 == 0:
            print(
                f't: {t} | i: {i} | r: {reward:.2f} |  R: {total_rewards[i]:.2f} | score: {scores[i]:.4f} | '
                f'values: {samples.values[i].item():.2f} | scale: {args.scale}',
                flush=True,
            )

        ## update rollout observations
        rollouts[i, t+1, :] = next_observation.copy()

        if terminal or total_rewards[i] < -50:
            done[i] = True
            steps[i] = t

## write results to json file at `args.savepath`
logger.finish(t, scores.tolist(), total_rewards.tolist(), terminal, diffusion_experiment, value_experiment)

data = np.zeros((args.n_processes, 3))
data[:, 0] = np.sort(total_rewards)[::-1]
data[:, 1] = np.sort(scores)[::-1]
data[:, 2] = np.array(steps)

wandb.log({"total_reward": total_rewards.mean(), "score": scores.mean(), "run_state": 1}, commit = True)
wandb.log({"runs": wandb.Table(data = data, columns = ["total_reward", "score", "steps"])}, commit = True)
wandb.finish()
time.sleep(20.0)