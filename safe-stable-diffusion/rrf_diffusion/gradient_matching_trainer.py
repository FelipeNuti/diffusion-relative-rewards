import os
import copy
import numpy as np
import torch
import einops
import pdb
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats.mstats import winsorize

from .models import cycle
from .dataset import GradientMatchingDataset, Batch, LabeledBatch
from .dataset import  batch_to_device
from .utils import check_nan, plot2img, to_np
from .utils import Timer, ActivationExtractor

from einops import rearrange
import wandb
from sklearn.linear_model import LogisticRegression

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

class GradientMatchingTrainer(object):
    def __init__(
        self,
        gradient_matching,
        expert_dataset,
        general_dataset,
        train_dataset = "mixed",
        train_frac = 0.90,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=2e-5,
        l2_reg=None,
        gradient_accumulate_every=1,
        gradient_clipping=0.025,
        weight_clipping=None,
        test_overfit=False,
        step_start_ema=2000,
        update_ema_every=10,
        log_freq=100,
        sample_freq=1000,
        save_freq=1000,
        render_freq=1000,
        label_freq=10000,
        num_trajectories_heatmap=1000,
        shape_factor_heatmap=1,
        num_samples=10,
        save_parallel=False,
        results_folder='./safe_stable_diffusion_logs',
        bucket=None,
        debug = False,
    ):
        super().__init__()

        self.model = gradient_matching
        self.ema = EMA(ema_decay)

        self.debug = debug

        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.render_freq = render_freq
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.label_freq = label_freq
        self.num_samples = num_samples
        self.num_trajectories_heatmap = num_trajectories_heatmap
        self.shape_factor_heatmap = shape_factor_heatmap
        self.save_parallel = save_parallel
        self.gradient_clipping = gradient_clipping
        self.weight_clipping = weight_clipping
        self.l2_reg = l2_reg
        self.test_overfit = test_overfit

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.expert_dataset = expert_dataset
        self.general_dataset = general_dataset

        mixed_dataset_train = GradientMatchingDataset(
            expert_dataset, 
            general_dataset, 
            split = "train",
            balanced = True,
            frac = train_frac,
            add_labels=False
        )

        self.dataset = mixed_dataset_train
        self.dataloader = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=train_batch_size, num_workers=2, shuffle=True, pin_memory=True,
            drop_last=True,
        ))

        if not self.test_overfit:
            self.dataset_eval = GradientMatchingDataset(
                expert_dataset, 
                general_dataset, 
                split = "test",
                normalizer=mixed_dataset_train.normalizer,
                n_diffusion_timesteps=1, # only eval on later diffusion timesteps
                balanced = True,
                frac = 1 - train_frac,
                add_labels=True
            )
        else:
            print("Running in Test Overfit mode - evaluating on training set")
            self.dataset_eval = GradientMatchingDataset(
                expert_dataset, 
                general_dataset, 
                split = "train",
                normalizer=mixed_dataset_train.normalizer,
                n_diffusion_timesteps=1, # only eval on later diffusion timesteps
                balanced = True,
                frac = train_frac,
                add_labels=True
            )
        
        self.dataloader_eval = cycle(torch.utils.data.DataLoader(
            self.dataset_eval, batch_size=train_batch_size, num_workers=0, shuffle=True, pin_memory=True
        ))

        if self.l2_reg is None:
            self.optimizer = torch.optim.Adam(self.model.model.parameters(), lr=train_lr)
        else:
            self.optimizer = torch.optim.Adam(self.model.model.parameters(), lr=train_lr, weight_decay=2*l2_reg)

        self.logdir = results_folder
        self.bucket = bucket

        self.num_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

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

    def stats_from_info(self, info, nan = False):
        prefix = "" if not nan else "nan_"

        d = {
            "min_t": info["t"].min().float(),
            "max_t": info["t"].max().float(),
            "avg_t": info["t"].float().mean(),
            "target_norm": (info["target"] ** 2).mean(),
            "target_max_abs": info["target"].abs().max(),
            "output_mean":info["out"].mean(),
            "grad_wrt_x":(info["grad_wrt_x"]**2).sum() ** 0.5,
            "fraction_grad_nonzero": (info["grad_wrt_x"] != 0).float().mean(),
        }

        return {prefix+k: v for k, v in d.items()}


    def print_info(self, **kwargs):
        s = f"Step {self.step} "
        for k,v in kwargs.items():
            s += f"| {k} {v:8.6} |"
        print(s)

        wandb.log({k:v for k, v in kwargs.items()})

    def simple_step(self):
        batch = next(self.dataloader)
        batch = batch_to_device(batch)
        loss, info = self.model.loss(*batch)
        loss = loss #/ self.gradient_accumulate_every
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.model.zero_grad()

    def train(self, n_train_steps):
        timer = Timer()
        wandb.watch(self.model.model, log = "all")
        activation_extractor = ActivationExtractor(self.model.model, enabled = not self.debug)
        for step in range(n_train_steps):
            for i in range(self.gradient_accumulate_every):
                activation_extractor.clear()
                batch = next(self.dataloader)
                batch = batch_to_device(batch)
                loss, info = self.model.loss(*batch)
                loss = loss / self.gradient_accumulate_every

                try:
                    loss.backward()
                except Exception as e:
                    print(e)
                    self.print_info(
                        loss = loss, 
                        **self.stats_from_info(info, nan = True),
                        max_activation = activation_extractor.max_activation(),
                        t = timer()
                    )
                    raise e

            if self.gradient_clipping is not None:
                torch.nn.utils.clip_grad_value_(
                    self.model.parameters(), 
                    self.gradient_clipping
                )

            if self.step % self.log_freq == 0:
                monitor_dict = self.monitor_gradient()
                self.print_info(
                    loss = loss, 
                    validation_loss = self.eval_validation_loss(),
                    **monitor_dict,
                    **self.stats_from_info(info, nan = False),
                    max_activation = activation_extractor.max_activation(),
                    t = timer()
                )

            self.optimizer.step()
            self.optimizer.zero_grad()
            self.model.zero_grad()

            if self.weight_clipping is not None:
                with torch.no_grad():
                    for param in self.model.parameters():
                        param.clamp_(-self.weight_clipping, self.weight_clipping)

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if (self.step + 1) % self.save_freq == 0:
                label = self.step // self.label_freq * self.label_freq
                self.save(label)

            if self.sample_freq and self.step % self.sample_freq == 0:
                self.eval_preference_prediction()
            
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

    def _adapt_dict(self, module, data):
        module_keys = module.state_dict().keys()
        return {k: data[k] for k in data.keys() if k in module_keys}

    def load(self, epoch):
        '''
            loads model and ema from disk
        '''
        loadpath = os.path.join(self.logdir, f'state_{epoch}.pt')
        data = torch.load(loadpath)
        data['model'] = self._adapt_dict(self.model, data['model'])
        data['ema'] = self._adapt_dict(self.ema_model, data['ema'])
        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])

    #-----------------------------------------------------------------------------#
    #--------------------------- eval reward prediction --------------------------#
    #-----------------------------------------------------------------------------#

    @torch.no_grad()
    def _get_preds(self, x_t, t):
        N = x_t.shape[0]
        return self.model(x_t, t), x_t, N

    def plot_output_hist(self, outs, labels, decision_boundary):
        plt.clf()
        fig, ax = plt.subplots()
        fig.set_size_inches(5, 5)

        label_names = list(map(lambda x: "expert" if x == 1 else "general", labels))

        palette = {
            "expert":"C0",
            "general": "C1"
        }

        median = np.median(outs)
        if decision_boundary is None:
            print("Encountered decision_boundary = None - Displaying median instead")
            decision_boundary = median

        sns.histplot(x=outs, hue = label_names, ax = ax, palette = palette, stat = "density")
        sns.kdeplot(x=outs, hue = label_names, fill = True, ax = ax.twinx(), palette = palette, cbar = True)
        ax.axvline(x = decision_boundary, c='r')
        ax.axvline(x = median, c='b')

        img = plot2img(fig, remove_margins=False)
        plt.close()
        wandb.log({
            "output_hist": wandb.Image(img)
        })

    def eval_validation_loss(self):
        avg_loss = 0.0
        n_batches = (512 // self.batch_size)
        for i in range(n_batches):
            batch = next(self.dataloader_eval)
            batch = Batch(batch.latents, batch.t, batch.next_general_latents, batch.next_expert_latents)
            batch = batch_to_device(batch)
            loss, info = self.model.loss(*batch, eval_mode = True)
            avg_loss += loss.item() / n_batches
        
        return avg_loss

    @torch.no_grad()
    def eval_preference_prediction(self):
        mixed_dataloader_eval = cycle(torch.utils.data.DataLoader(
            self.dataset_eval, batch_size=128, num_workers=0, shuffle=True, pin_memory=True
        ))

        all_outs = []   
        all_labels = []

        for _ in range(200):
            batch = next(mixed_dataloader_eval)
            batch = batch_to_device(batch)
            x_t, t, _, _, labels = batch
            out, _, N = self._get_preds(x_t, t)


            out = to_np(out)
            labels = to_np(labels).flatten()

            all_outs.append(out.flatten())
            all_labels.append(labels)

        all_outs = np.concatenate(all_outs).reshape((-1, 1))
        all_labels = np.concatenate(all_labels).reshape((-1,))

        all_outs = winsorize(all_outs, limits = (0.005, 0.005), nan_policy = 'raise').data

        sample_weights = np.ones(all_labels.shape)
        n_samples = all_labels.shape[0]
        n_expert = all_labels.sum()
        n_general = n_samples  - n_expert
        sample_weights[all_labels == 1] = n_samples/n_expert if n_expert > 0 else 1
        sample_weights[all_labels == 0] = n_samples/n_general if n_general > 0 else 1

        clf = LogisticRegression(class_weight="balanced")
        clf.fit(all_outs, all_labels)
        accuracy = clf.score(all_outs, all_labels, sample_weight = sample_weights)

        decision_boundary = -clf.intercept_[0]/clf.coef_[0] if abs(clf.coef_[0]) >= 1e-8 else None

        wandb.log({
            "accuracy": accuracy,
            "logistic_reg_intercept": clf.intercept_[0],
            "logistic_reg_coef": clf.coef_[0]
        })

        self.plot_output_hist(all_outs.flatten(), all_labels.flatten(), decision_boundary)

        print(f'[ utils/training ] Predicted relative preference with accuracy {accuracy}')
        mixed_dataloader_eval.close()