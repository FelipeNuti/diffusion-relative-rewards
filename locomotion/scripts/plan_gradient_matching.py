import pdb
import copy
import diffuser.sampling as sampling
import diffuser.utils as utils
import wandb
import time


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

value_experiment = utils.load_gradient_matching(
    args.value_loadpath,
    dataset = dataset,
    diffusion = copy.deepcopy(diffusion),
    epoch=args.value_epoch, seed=args.seed
)

## ensure that the diffusion model and value function are compatible with each other
utils.check_compatibility(diffusion_experiment, value_experiment)

## initialize value guide
value_function = value_experiment.ema
guide_config = utils.Config(args.guide, model=value_function, verbose=False)
guide = guide_config()

if abs(args.scale) < 1e-10:
    args.n_guide_steps = 0

## policies are wrappers around an unconditional diffusion model and a value guide
policy_config = utils.Config(
    args.policy,
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

env = dataset.env
env.seed(args.seed)
observation = env.reset()

## observations for rendering
rollout = [observation.copy()]

total_reward = 0
wandb.log({"run_state": 0}, commit = True)
for t in range(args.max_episode_length):

    if t % 10 == 0: print(args.savepath, flush=True)

    ## save state for rendering only
    state = env.state_vector().copy()

    ## format current observation for conditioning
    conditions = {0: observation}
    action, samples = policy(conditions, batch_size=args.batch_size, verbose=args.verbose)

    ## execute action in environment
    next_observation, reward, terminal, _ = env.step(action)

    ## print reward and score
    total_reward += reward
    score = env.get_normalized_score(total_reward)
    print(
        f't: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | '
        f'values: {samples.values} | scale: {args.scale}',
        flush=True,
    )

    if t % 100 == 0:
        wandb.log({
            "reward": reward,
            "total_reward": total_reward,
            "score": score,
        })

    ## update rollout observations
    rollout.append(next_observation.copy())

    ## render every `args.vis_freq` steps
    logger.log(t, samples, state, rollout)

    if terminal or total_reward < -50:
        break

    observation = next_observation

## write results to json file at `args.savepath`
logger.finish(t, score, total_reward, terminal, diffusion_experiment, value_experiment)
wandb.log({"total_reward": total_reward, "score": env.get_normalized_score(total_reward), "run_state": 1}, commit = True)
wandb.finish()
time.sleep(20.0)