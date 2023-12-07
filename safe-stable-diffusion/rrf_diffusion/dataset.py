import torch
import numpy as np
import os 
from collections import namedtuple
import einops

LabeledBatch = namedtuple('LabeledBatch', 'latents t next_general_latents next_expert_latents label')
Batch = namedtuple('Batch', 'latents t next_general_latents next_expert_latents')

def batch_to_device(batch, device='cuda:0', dtype=torch.float32):
    vals = []
    for field in batch._fields:
        if field != "t":
            vals.append(getattr(batch, field).to(device = device, dtype = dtype))
        else:
            vals.append(getattr(batch, field).flatten().to(device = device, dtype = dtype))
    return type(batch)(*vals)

class LimitsNormalizer(torch.nn.Module):
    def __init__(self, xs):
        super().__init__()

        self.maxs = [einops.reduce(data, "b t idx c h w -> 1 t c h w", "max") for data in xs]
        self.mins = [einops.reduce(data, "b t idx c h w -> 1 t c h w", "min") for data in xs]

        self.maxs = einops.reduce(torch.cat(self.maxs, dim = 0), "n t c h w -> t 1 c h w", "max")
        self.mins = einops.reduce(torch.cat(self.mins, dim = 0), "n t c h w -> t 1 c h w", "min")

    def normalize(self, x, t, inplace = False):
        if inplace:
            x.sub_(self.mins[t])
            x.div_(self.maxs[t] - self.mins[t])
            x.mul_(2)
            x.sub_(1)
            return x
        else:
            return (x - self.mins[t]) / (self.maxs[t] - self.mins[t])

    def unnormalize(self, ans, t, inplace = False):
        if inplace:
            ans.add_(1)
            ans.div_(2)
            ans.mul_(self.maxs[t] - self.mins[t])
            ans.add_(self.mins[t])
            return ans
        else:
            ans = (ans + 1) / 2
            ans = (self.maxs[t] - self.mins[t]) * ans + self.mins[t]

class StandardNormalizer(torch.nn.Module):
    def __init__(self, xs):
        super().__init__()

        total_len = np.sum([len(data) for data in xs])
        self.means = [einops.reduce(data, "b t idx c h w -> 1 t c h w", "mean").to(dtype = torch.float32) * len(data) / total_len for data in xs]
        self.stdev = [torch.std(data, dim = (0, 2), keepdim = False).unsqueeze(0).to(dtype = torch.float32) ** 2 * len(data) for data in xs]

        self.means = einops.reduce(torch.cat(self.means, dim = 0), "n t c h w -> t 1 c h w", "sum")
        self.stdev = torch.sqrt(einops.reduce(torch.cat(self.stdev, dim = 0), "n t c h w -> t 1 c h w", "sum") / total_len + 1e-8)

        print(f"Normalizer: means {self.means.abs().min()} - {self.means.abs().max()} | std {self.stdev.abs().min()} - {self.stdev.abs().max()}")

    def normalize(self, x, t, inplace = False):
        if inplace:
            x.sub_(self.means[t])
            x.div_(self.stdev[t])
            return x
        else:
            return (x.to(dtype = torch.float32) - self.means[t]) / self.stdev[t]

    def unnormalize(self, ans, t, inplace = False):
        if inplace:
            ans.mul_(self.stdev[t])
            ans.add_(self.means[t])
        else:
            ans = self.stdev[t] * ans + self.means[t]
        return ans

class GradientMatchingDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        expert_dataset,
        general_dataset,
        base_path = "./safe_stable_diffusion_data/train",
        n_diffusion_timesteps = 50,
        split = "all",
        normalizer = None,
        frac = 1.0,
        balanced = False,
        add_labels = True
    ):
        self.expert_dataset = expert_dataset
        self.general_dataset = general_dataset
        self.n_diffusion_timesteps = n_diffusion_timesteps

        self.split = split
        self.frac = frac
        self.balanced = balanced

        self.make_lengths()

        if split != "test":
            self.normalizer = StandardNormalizer([expert_dataset, general_dataset])
        else:
            self.normalizer = normalizer

        self.add_labels = add_labels

        self.len = (self.n_expert + self.n_general) * self.n_diffusion_timesteps

        print(f"[utils/datasets.py] Lengths - Expert dataset: {self.n_expert} - General dataset: {self.n_general}")
        print(f"[utils/datasets.py] Ranges - Expert dataset: {self.expert_dataset.min()}-{self.expert_dataset.max()}"+
               f"- General dataset: {self.general_dataset.min()}-{self.general_dataset.max()}")

    def __len__(self):
        return self.len

    def make_lengths(self):
        self.n_expert = len(self.expert_dataset)
        self.n_general = len(self.general_dataset)
        self.start_expert = 0
        self.start_general = 0

        share_expert = self.n_expert
        share_general = self.n_general

        if self.split != "all":
            share_expert = int(self.n_expert * self.frac)
            share_general = int(self.n_general * self.frac)

        if self.balanced:
            share_expert = min(share_expert, share_general)
            share_general = share_expert

        if self.split == "train":
            self.expert_dataset = self.expert_dataset[:share_expert]
            self.general_dataset = self.general_dataset[:share_general]

        elif self.split == "test":
            self.expert_dataset = self.expert_dataset[self.n_expert - share_expert:]
            self.general_dataset = self.general_dataset[self.n_general - share_general:]

        self.n_expert = share_expert
        self.n_general = share_general

        self.n_expert_base = self.n_expert
        self.n_general_base = self.n_general

        if self.balanced:
            self.n_expert = max(self.n_expert, self.n_general)
            self.n_general = self.n_expert
    
    def __getitem__(self, idx):
        if idx >= self.len and idx < 0:
            raise IndexError()
        
        idx_latent = idx // self.n_diffusion_timesteps
        idx_t = idx % self.n_diffusion_timesteps
        idx_next_t = min(idx_t + 1, self.n_diffusion_timesteps - 1)
        
        if idx_latent < self.n_expert:
            data = self.expert_dataset[idx_latent, idx_t]
            prev_latent = self.expert_dataset[idx_latent, idx_next_t , 1]
            label = torch.tensor([1])
        else:
            data = self.general_dataset[idx_latent - self.n_expert, idx_t]
            prev_latent = self.general_dataset[idx_latent - self.n_expert, idx_next_t , 0]
            label = torch.tensor([0])

        data = self.normalizer.normalize(data, idx_t)
        prev_latent = self.normalizer.normalize(prev_latent[None, :, :, :], idx_next_t)[0]

        if self.add_labels:
            return LabeledBatch(prev_latent, torch.tensor([idx_t]), data[0], data[1], label)
        else:
            return Batch(prev_latent, torch.tensor([idx_t]), data[0], data[1])