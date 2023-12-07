import os
import copy
import numpy as np
import torch
import einops
import pdb
import wandb

from collections import namedtuple

from diffuser.utils.arrays import batch_to_device, to_np, to_device, apply_dict
from diffuser.datasets.sequence import Batch
from diffuser.utils.timer import Timer
from diffuser.utils.cloud import sync_logs
from diffuser.sampling.functions import n_step_guided_p_sample
from diffuser.utils.debugging import check_nan
from diffuser.utils.datasets import MixedDataset

def cycle(dl):
    while True:
        for data in dl:
            yield data

class EMA():
    '''
        empirical moving average
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class DiscriminatorTrainer(object):
    def __init__(
        self,
        discriminator,
        expert_dataset,
        general_dataset,
        train_frac = 0.9,
        n_timesteps=1000,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=2e-5,
        gradient_accumulate_every=2,
        step_start_ema=2000,
        update_ema_every=10,
        log_freq=100,
        sample_freq=1000,
        save_freq=1000,
        label_freq=100000,
        save_parallel=False,
        results_folder='./results',
        bucket=None,
    ):
        super().__init__()

        self.model = discriminator
        self.ema = EMA(ema_decay)

        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.t_max_frac = 0.1
        self.n_timesteps = n_timesteps

        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.label_freq = label_freq
        self.save_parallel = save_parallel

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        mixed_dataset = MixedDataset(
            expert_dataset, 
            general_dataset, 
            split = "train",
            frac = train_frac,
            add_labels=True
        )

        self.dataset = mixed_dataset
        self.dataloader = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=train_batch_size, num_workers=2, shuffle=True, pin_memory=True
        ))

        self.dataset_eval = MixedDataset(
            expert_dataset, 
            general_dataset, 
            split = "test",
            frac = 1 - train_frac,
            add_labels=True
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=train_lr)

        self.dataloader_eval = cycle(torch.utils.data.DataLoader(
            self.dataset_eval, batch_size=self.batch_size, num_workers=0, shuffle=True, pin_memory=True
        ))

        self.logdir = results_folder
        self.bucket = bucket

        self.reset_parameters()
        self.step = 0

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def monitor_gradient(self):
        """
        Computes L2 norm of gradients and parameters
        """
        params = [p for p in self.model.parameters() if p.grad is not None and p.requires_grad]
        grad_norm = 0.0
        param_norm = 0.0
        param_max = 0.0
        grad_max = 0.0

        for p in params:
            check_nan(p.grad, "gradient")()
            param_norm += p.detach().norm(2).data.item() ** 2
            grad_norm += p.grad.detach().norm(2).data.item() ** 2
            grad_max = max(grad_max, p.grad.detach().abs().max().item())
            param_max = max(param_max, p.detach().abs().max().item())
        
        grad_norm = grad_norm ** 0.5
        param_norm = param_norm ** 0.5
        return {
            "grad_norm": grad_norm, 
            "param_norm": param_norm,
            "param_max": param_max,
            "grad_max": grad_max
        }

    def print_info(self, **kwargs):
        s = f"Step {self.step} "
        for k,v in kwargs.items():
            s += f"| {k} {v:8.6} |"
        print(s)

        wandb.log({k:v for k, v in kwargs.items() if k != "loss"})


    #-----------------------------------------------------------------------------#
    #------------------------------------ api ------------------------------------#
    #-----------------------------------------------------------------------------#

    def train(self, n_train_steps):
        timer = Timer()
        for step in range(n_train_steps):
            for i in range(self.gradient_accumulate_every):
                batch = next(self.dataloader)
                batch = batch_to_device(batch)

                loss, accuracy = self.model.loss(*batch)
                loss = loss / self.gradient_accumulate_every
                loss.backward()

            if self.step % self.log_freq == 0:
                monitor_dict = self.monitor_gradient()
                self.print_info(
                    loss= loss, 
                    validation_loss = self.eval_validation_loss(),
                    **monitor_dict,
                    accuracy_arbitrary_noise = accuracy,
                    t = timer()
                )

            wandb.log({
                "loss": loss.item()
            })

            self.optimizer.step()
            self.optimizer.zero_grad()
            self.model.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step % self.save_freq == 0:
                label = self.step // self.label_freq * self.label_freq
                self.save(label)

            if self.step % self.sample_freq == 0:
                self.eval_accuracy_without_noise()

            self.step += 1

    def save(self, epoch):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        savepath = os.path.join(self.logdir, f'state_{epoch}.pt')
        torch.save(data, savepath)
        print(f'[ utils/training ] Saved model to {savepath}')
        if self.bucket is not None:
            sync_logs(self.logdir, bucket=self.bucket, background=self.save_parallel)

    def load(self, epoch):
        '''
            loads model and ema from disk
        '''
        loadpath = os.path.join(self.logdir, f'state_{epoch}.pt')
        data = torch.load(loadpath)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])
    def eval_validation_loss(self):
        avg_loss = 0.0
        n_batches = 4
        for i in range(n_batches):
            batch = next(self.dataloader_eval)
            batch = batch_to_device(batch)
            loss, info = self.model.loss(*batch)
            avg_loss += loss.item() / n_batches
        
        return avg_loss
    
    @torch.no_grad()
    def eval_accuracy_without_noise(self):
        mixed_dataloader_eval = cycle(torch.utils.data.DataLoader(
            self.dataset_eval, batch_size=self.batch_size, num_workers=0, shuffle=True, pin_memory=True
        ))
        total_correct = 0
        total = 0

        for _ in range(100):
            batch = next(mixed_dataloader_eval)
            batch = batch_to_device(batch)
            batch_size = batch.trajectories.shape[0]

            t = torch.randint(
                0, int(self.t_max_frac * self.n_timesteps), 
                (batch_size,), 
                device=batch.trajectories.device).long()

            loss, accuracy = self.model.loss(*batch, t)

            total_correct += int(batch_size * accuracy)
            total += batch_size

        accuracy = total_correct/total
        wandb.log({
            "accuracy_controlled_noise": accuracy
        })
        print(f'[ utils/training ] Predicted relative preference with accuracy {accuracy}')

        mixed_dataloader_eval.close()
