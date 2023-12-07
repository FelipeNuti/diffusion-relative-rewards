# For debugging purposes
import typing as t
import diffuser.utils as utils
from diffuser.models import ValueFunction, TemporalValue, SimpleMLPValue
from diffuser.utils.arrays import batch_to_device, to_np, to_device, apply_dict
from diffuser.utils.training import cycle
from discriminator import TrajectoryDiscriminator
from discriminator_trainer import DiscriminatorTrainer
import diffuser.utils as utils
import torch
import wandb


#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    dataset: str = 'maze2d-large-v1'
    config: str = 'config.discriminator'
    dim_mults: t.List[int] = None
    debug: bool = False

args = Parser().parse_args('discriminator', prepare_dirs=False)

wandb.init(
    # project = "discriminator_benchmark", 
    project = "diffusion_relative_rewards", 
    # entity="rrf-diffusion",
    # dir = "/scratch/shared/beegfs/<username>/wandb",
    mode = "online" if not args.debug else "disabled",
    config = args._dict
)

args.prepare_dirs(args)

expert_experiment = utils.load_diffusion(
    args.s_expert_path, 
    fields_load_path = args.expert_buffer_path
)
general_experiment = utils.load_diffusion(
    args.s_general_path,
    fields_load_path = args.general_buffer_path
)

s_expert = expert_experiment.ema
s_general = general_experiment.ema

expert_dataset = expert_experiment.dataset#.to(device = "cuda:0")
general_dataset = general_experiment.dataset#.to(device = "cuda:0")

expert_dataset.clean_fields()
general_dataset.clean_fields()

expert_dataset.renormalize(general_dataset.normalizer)

model = SimpleMLPValue(
    s_expert.horizon,
    s_expert.transition_dim,
    dim = args.dim,
    stride = args.stride,
    kernel_size = args.kernel_size,
    dim_mults = args.dim_mults,
    embed_dim = args.embed_dim
)

trainer_config = utils.Config(
    DiscriminatorTrainer,
    savepath=(args.savepath, 'trainer_config.pkl'),
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
batch = utils.batchify(trainer.dataset[0])
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





