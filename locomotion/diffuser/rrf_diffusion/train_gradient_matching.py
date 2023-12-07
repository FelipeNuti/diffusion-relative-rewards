import typing as t
import os 
import diffuser.utils as utils
import diffuser.utils as utils
import torch
import wandb
from tqdm import tqdm

#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    dataset: str = 'maze2d-large-v1'
    config: str = 'config.gradient_matching'
    debug: bool = False
    offline: bool = False
    profile: bool = False
    profile_wait:int = 1
    profile_warmup:int = 1
    profile_active:int = 13
    profile_repeat:int = 3

args = Parser().parse_args('gradient_matching', prepare_dirs=False)

wandb.init(
    project = "diffusion_relative_rewards", 
    # entity="rrf-diffusion",
    # dir = "/scratch/shared/beegfs/<username>/wandb",
    mode = "online" if not args.debug else "disabled",
    tags = [f"rollout_gm_{args.tag}"] if args.tag is not None else [],
    config = args._dict
)

args.prepare_dirs(args)

expert_experiment = utils.load_diffusion(
    args.s_expert_path, 
    epoch=args.expert_epoch,
    #fields_load_path = args.expert_buffer_path
)
general_experiment = utils.load_diffusion(
    args.s_general_path,
    epoch=args.general_epoch
)

s_expert = expert_experiment.ema
s_general = general_experiment.ema

expert_dataset = expert_experiment.dataset#.to(device = "cuda:0")
general_dataset = general_experiment.dataset#.to(device = "cuda:0")

expert_dataset.renormalize(general_dataset.normalizer)

horizon = s_expert.horizon if args.model_horizon is None else args.model_horizon

render_config = utils.Config(
    args.renderer,
    savepath=(args.savepath, 'render_config.pkl'),
    env=args.dataset,
)

dim_mults = [int(p) for p in args.dim_mults.split(",")]
print("Final dim_mults:", dim_mults)

model_config = utils.Config(
    args.model_class,
    savepath=(args.savepath, 'model_config.pkl'),
    dim = args.dim,
    horizon = horizon,
    stride = args.stride,
    kernel_size = args.kernel_size,
    dim_mults = dim_mults,
    embed_dim = args.embed_dim,
    activation = args.activation,
    norm = args.norm,
    bias = args.bias,
)
model = model_config(s_expert.transition_dim)

trainer_config = utils.Config(
    'rrf_diffusion.GradientMatchingTrainer',
    train_dataset = args.train_dataset,
    savepath=(args.savepath, 'trainer_config.pkl'),
    train_batch_size=args.batch_size,
    train_frac=args.train_frac,
    train_lr=args.learning_rate,
    l2_reg=args.l2_reg,
    gradient_accumulate_every=args.gradient_accumulate_every,
    gradient_clipping = args.gradient_clipping,
    weight_clipping = args.weight_clipping,
    ema_decay=args.ema_decay,
    sample_freq=args.sample_freq,
    save_freq=args.save_freq,
    num_samples=args.num_samples,
    num_trajectories_heatmap=args.num_trajectories_heatmap,
    shape_factor_heatmap = args.shape_factor_heatmap,
    label_freq=1, #int(args.n_train_steps // args.n_saves),
    log_freq = args.log_freq,
    render_freq = args.render_freq,
    save_parallel=args.save_parallel,
    results_folder=args.savepath,
    bucket=args.bucket,
    debug = args.debug
)
gradient_matching_config = utils.Config(
    args.gradient_matching_class,
    savepath=(args.savepath, 'gradient_matching_config.pkl'),
    diffusion_predicts_mean=args.diffusion_predicts_mean,
    eps_loss = args.eps_loss,
    scale_scores=args.scale_scores,
    alpha = args.alpha
)

gradient_matcher = gradient_matching_config(
    model, 
    s_expert, 
    s_general).to(device = "cuda:0")
renderer = render_config()
trainer = trainer_config(
    gradient_matcher, 
    s_expert, 
    s_general, 
    expert_dataset, 
    general_dataset,
    renderer,
    None,
)

utils.report_parameters(model)
print('Testing forward...', end=' ', flush=True)
batch = next(trainer.dataloader)
batch = utils.batch_to_device(batch)
loss, info = gradient_matcher.loss(*batch)
loss.backward()
print('✓')


#-----------------------------------------------------------------------------#
#--------------------------------- main loop ---------------------------------#
#-----------------------------------------------------------------------------#

n_epochs = int(args.n_train_steps // args.n_steps_per_epoch)

if not args.profile:
    for i in range(n_epochs):
        print(f'Epoch {i} / {n_epochs} | {args.savepath}')
        trainer.train(n_train_steps=args.n_steps_per_epoch)
else:
    profile_path = os.path.join('./profiling/gradient_matching', args.dataset, args.exp_name)
    waiting, warmup, active, repeat = args.profile_wait, args.profile_warmup, args.profile_active, args.profile_repeat

    with torch.profiler.profile(
        schedule = torch.profiler.schedule(wait = waiting, warmup = warmup, active = active, repeat = repeat),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_path),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        print("Profiling started")
        print("Profile path:", os.path.abspath(profile_path))
        print(f"wait = {waiting}, warmup = {warmup}, active = {active}, repeat = {repeat}")
        for i in tqdm(range(repeat * (waiting + warmup + active))):
            trainer.simple_step()
            prof.step()






