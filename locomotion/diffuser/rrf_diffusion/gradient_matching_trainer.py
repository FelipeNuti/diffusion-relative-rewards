import os
import copy
import numpy as np
import torch
import einops
import pdb
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats.mstats import winsorize

from diffuser.utils.arrays import batch_to_device, to_np#, to_device, apply_dict
from ..datasets.sequence import Batch
from diffuser.utils.timer import Timer
from diffuser.utils.cloud import sync_logs
from diffuser.sampling.functions import n_step_guided_p_sample
from diffuser.utils.debugging import check_nan, ActivationExtractor
from diffuser.utils.datasets import MixedDataset
from diffuser.utils.rendering import plot2img
from einops import rearrange
import wandb
from sklearn.linear_model import LogisticRegression

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

class GradientMatchingTrainer(object):
    def __init__(
        self,
        gradient_matching,
        s_expert,
        s_general,
        expert_dataset,
        general_dataset,
        renderer,
        heatmap_dataset,
        train_dataset = "expert",
        train_frac = 0.90,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=2e-5,
        l2_reg=None,
        gradient_accumulate_every=2,
        gradient_clipping=None,
        weight_clipping=None,
        step_start_ema=2000,
        update_ema_every=10,
        log_freq=100,
        sample_freq=1000,
        save_freq=1000,
        render_freq=1000,
        label_freq=100000,
        num_trajectories_heatmap=1000,
        shape_factor_heatmap=1,
        num_samples=10,
        save_parallel=False,
        results_folder='./results',
        bucket=None,
        debug = False,
    ):
        super().__init__()

        self.model = gradient_matching
        self.s_expert = s_expert
        self.s_general = s_general
        self.renderer = renderer
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

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.expert_dataset = expert_dataset
        self.general_dataset = general_dataset

        if heatmap_dataset is None:
            self.heatmap_dataset = general_dataset
        else:
            print("Using different dataset for heatmap")
            self.heatmap_dataset = heatmap_dataset

        mixed_dataset_train = MixedDataset(
            expert_dataset, 
            general_dataset, 
            split = "train",
            balanced = True,
            frac = train_frac,
            add_labels=False
        )

        if train_dataset == "expert":
            self.dataset = expert_dataset
        elif train_dataset == "general":
            self.dataset = general_dataset
        elif train_dataset == "mixed":
            self.dataset = mixed_dataset_train
        self.dataloader = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=train_batch_size, num_workers=2, shuffle=True, pin_memory=True
        ))

        self.dataset_eval = MixedDataset(
            expert_dataset, 
            general_dataset, 
            split = "test",
            balanced = True,
            frac = 1 - train_frac,
            add_labels=True
        )
        
        self.dataloader_eval = cycle(torch.utils.data.DataLoader(
            self.dataset_eval, batch_size=1024, num_workers=0, shuffle=True, pin_memory=True
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

        wandb.log({k:v for k, v in kwargs.items() if k != "loss"})


    #-----------------------------------------------------------------------------#
    #------------------------------------ api ------------------------------------#
    #-----------------------------------------------------------------------------#

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

            wandb.log({
                "loss": loss.item()
            })

            if self.gradient_clipping is not None:
                torch.nn.utils.clip_grad_value_(
                    self.model.parameters(), 
                    self.gradient_clipping #/ (self.num_trainable ** 0.5)
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

            if self.step == 0 or (self.step + 1) % self.save_freq == 0:
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
        if self.bucket is not None:
            sync_logs(self.logdir, bucket=self.bucket, background=self.save_parallel)

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

    def _get_preds(self, x_start, conds, t_max_frac = 1.0, predict_per_point = False):
        N = x_start.shape[0]

        t_max = int(t_max_frac * self.model.n_timesteps) if t_max_frac is not None else 1
        t = torch.randint(0, t_max, (N,), device=x_start.device).long()

        x_t = self.s_expert.q_sample(x_start, t)

        if not predict_per_point:
            return self.model(x_t, conds, t), x_t, N
        else:
            model_horizon = self.model.model.horizon

            padding = torch.zeros((x_t.shape[0], model_horizon - 1, x_t.shape[2]), device=x_start.device)
            padding[:, :, :] = x_t[:, -1:, :]
            x_t_pad = torch.cat([x_t, padding], dim = 1)

            x_windows = x_t_pad.unfold(1, model_horizon, 1)
            n_windows = x_windows.shape[1]

            assert n_windows == x_t.shape[1], "Unfolding changed horizon"

            x_windows = einops.rearrange(x_windows, "b nw c h -> (b nw) h c")
            t_windows = t.reshape((-1, 1)).repeat((1, n_windows)).flatten()

            preds_windows = self.model(x_windows, conds, t_windows)

            preds = einops.rearrange(preds_windows, "(b nw) 1 -> b nw 1", nw = n_windows)
            return preds, x_t, N


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

        sns.histplot(x=outs, hue = label_names, ax = ax, palette = palette, stat = "density")
        sns.kdeplot(x=outs, hue = label_names, fill = True, ax = ax.twinx(), palette = palette, cbar = True)
        ax.axvline(x = decision_boundary, c='r')
        ax.axvline(x = median, c='b')

        img = plot2img(fig, remove_margins=False)
        plt.close()
        wandb.log({
            "output_hist": wandb.Image(img)
        })

    @torch.no_grad()
    def plot_reward_heatmap(self):
        general_dataloader_eval = cycle(torch.utils.data.DataLoader(
            self.heatmap_dataset, batch_size=self.num_trajectories_heatmap, num_workers=0, shuffle=True, pin_memory=True
        ))

        batch = next(general_dataloader_eval)
        batch = batch_to_device(batch)

        x_start, conds = batch
        
        x_start = rearrange(x_start, "b (mh nw) c -> (b nw) mh c", mh = self.model.model.horizon)
        preds, x, _ = self._get_preds(x_start, conds, t_max_frac = None)
        preds = preds.cpu().flatten()
        
        preds = torch.tensor(winsorize(to_np(preds), limits = (0.005, 0.005), nan_policy = 'raise').data, device = "cpu")
        preds = (preds - preds.min()) / (preds.max() - preds.min())
        
        x = self.dataset.normalizer.unnormalize(x.cpu(), "trajectories")
        x = x[:, :, self.dataset.action_dim:]

        reward_heatmap, reward_field, counts_heatmap, counts_field = \
            self.renderer.render_reward_heatmap(x, preds, shape_factor = self.shape_factor_heatmap, samples_thresh = 5)
        
        wandb.log({
            "reward_heatmap": wandb.Image(reward_heatmap),
            "counts_heatmap": wandb.Image(counts_heatmap),
            "counts_field": counts_field,
            "reward_field": reward_field
        })
    def eval_validation_loss(self):
        avg_loss = 0.0
        n_batches = 4
        for i in range(n_batches):
            batch = next(self.dataloader_eval)
            batch = Batch(batch.trajectories, batch.conditions)
            batch = batch_to_device(batch)

            t = torch.randint(0, self.model.n_timesteps, (batch.trajectories.shape[0],), device=batch.trajectories.device).long()
            loss, info = self.model.loss(*batch, t = t, eval_mode = True)
            avg_loss += loss.item() / n_batches
        
        return avg_loss

    @torch.no_grad()
    def eval_preference_prediction(self):
        mixed_dataloader_eval = cycle(torch.utils.data.DataLoader(
            self.dataset_eval, batch_size=256, num_workers=0, shuffle=True, pin_memory=True
        ))

        all_outs = []   
        all_labels = []

        for _ in range(100):
            batch = next(mixed_dataloader_eval)
            batch = batch_to_device(batch)
            x_start, conds, labels = batch
            out, _, N = self._get_preds(x_start, conds, t_max_frac=0.1)

            out = to_np(out)
            labels = to_np(labels).flatten()

            all_outs.append(out.flatten())
            all_labels.append(labels)

        all_outs = np.concatenate(all_outs).reshape((-1, 1))
        all_labels = np.concatenate(all_labels).reshape((-1,))

        all_outs = winsorize(all_outs, limits = (0.01, 0.01), nan_policy = 'raise').data

        sample_weights = np.ones(all_labels.shape)
        n_samples = all_labels.shape[0]
        n_expert = all_labels.sum()
        n_general = n_samples  - n_expert
        sample_weights[all_labels == 1] = n_samples/n_expert if n_expert > 0 else 1
        sample_weights[all_labels == 0] = n_samples/n_general if n_general > 0 else 1

        clf = LogisticRegression(class_weight="balanced")
        clf.fit(all_outs, all_labels)
        accuracy = clf.score(all_outs, all_labels, sample_weight = sample_weights)

        decision_boundary = -clf.intercept_[0]/clf.coef_[0] if clf.coef_[0] != 0.0 else 0

        wandb.log({
            "accuracy": accuracy,
            "logistic_reg_intercept": clf.intercept_[0],
            "logistic_reg_coef": clf.coef_[0]
        })

        self.plot_output_hist(all_outs.flatten(), all_labels.flatten(), decision_boundary)

        print(f'[ utils/training ] Predicted relative preference with accuracy {accuracy}')
        mixed_dataloader_eval.close()

    #-----------------------------------------------------------------------------#
    #--------------------------- visualize predictions ---------------------------#
    #-----------------------------------------------------------------------------#

    def render_reference(self):
        '''
            renders training points
        '''
        mixed_dataloader_eval = cycle(torch.utils.data.DataLoader(
            self.dataset_eval, batch_size=self.num_samples, num_workers=0, shuffle=True, pin_memory=True
        ))
        batch = next(mixed_dataloader_eval)
        mixed_dataloader_eval.close()

        batch = batch_to_device(batch)
        x_start, conds, labels = batch
        preds, x_t, _ = self._get_preds(x_start, conds, t_max_frac=0.1, predict_per_point=True)

        trajectories = to_np(x_t)
        conditions = to_np(conds[0])[:, None]
        labels = to_np(labels)
        preds = to_np(preds)

        
        preds = winsorize(preds, limits = (0.005, 0.005), axis = 1, nan_policy = 'raise').data
        
        preds = (preds - preds.min(axis = 1, keepdims = 1)) / \
            (preds.max(axis = 1, keepdims = 1) - preds.min(axis = 1, keepdims = 1))
        titles = ["expert" if label else "general" for label in labels.flatten()]
        unnormed_trajectories = self.dataset.normalizer.unnormalize(torch.tensor(trajectories), 'trajectories')
        observations = unnormed_trajectories[:, :, self.dataset.action_dim:]

        savepath = os.path.join(self.logdir, f'samples_step{self.step}.png')
        self.renderer.composite(savepath, to_np(observations), to_np(preds), save_wandb = True, title = titles)
    