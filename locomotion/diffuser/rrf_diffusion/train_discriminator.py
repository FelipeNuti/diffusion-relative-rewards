# For debugging purposes
import typing as t
import diffuser.utils as utils
from diffuser.models import ValueFunction, SimpleMLPValue
from diffuser.utils.arrays import batch_to_device, to_np, to_device, apply_dict
from diffuser.utils.training import cycle
from diffuser.rrf_diffusion.discriminator import TrajectoryDiscriminator
from diffuser.rrf_diffusion.discriminator_trainer import DiscriminatorTrainer
import diffuser.utils as utils
import torch
import wandb


#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    dataset: str = 'maze2d-large-v1'
    config: str = 'config.discriminator'
    debug: bool = False

args = Parser().parse_args('discriminator', prepare_dirs=False)

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
)
general_experiment = utils.load_diffusion(
    args.s_general_path,
)

s_expert = expert_experiment.ema
s_general = general_experiment.ema

expert_dataset = expert_experiment.dataset#.to(device = "cuda:0")
general_dataset = general_experiment.dataset#.to(device = "cuda:0")

expert_dataset.renormalize(general_dataset.normalizer)

dim_mults = [int(p) for p in args.dim_mults.split(",")]
print("Final dim_mults:", dim_mults)

horizon = s_expert.horizon if args.model_horizon is None else args.model_horizon

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
    DiscriminatorTrainer,
    savepath=(args.savepath, 'discriminator_trainer_config.pkl'),
    train_batch_size=args.batch_size,
    train_frac=args.train_frac,
    train_lr=args.learning_rate,
    n_timesteps=s_expert.n_timesteps,
    gradient_accumulate_every=args.gradient_accumulate_every,
    ema_decay=args.ema_decay,
    sample_freq=args.sample_freq,
    save_freq=args.save_freq,
    label_freq=1, #int(args.n_train_steps // args.n_saves),
    log_freq = args.log_freq,
    save_parallel=args.save_parallel,
    results_folder=args.savepath,
    bucket=args.bucket,
)
discriminator = TrajectoryDiscriminator(model, s_expert).to(device = "cuda:0")
trainer = trainer_config(discriminator, expert_dataset, general_dataset)

utils.report_parameters(model)

print('Testing forward...', end=' ', flush=True)
batch = next(trainer.dataloader)
batch = utils.batch_to_device(batch)
loss, info = discriminator.loss(*batch)
loss.backward()
print('✓')


#-----------------------------------------------------------------------------#
#--------------------------------- main loop ---------------------------------#
#-----------------------------------------------------------------------------#

n_epochs = int(args.n_train_steps // args.n_steps_per_epoch)

for i in range(n_epochs):
    print(f'Epoch {i} / {n_epochs} | {args.savepath}')
    trainer.train(n_train_steps=args.n_steps_per_epoch)





