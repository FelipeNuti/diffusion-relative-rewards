import torch
from collections import namedtuple

LabeledBatch = namedtuple('LabeledBatch', 'trajectories conditions label')
Batch = namedtuple('Batch', 'trajectories conditions')
class MixedDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        expert_dataset, 
        general_dataset, 
        split = "all",
        frac = 1.0,
        balanced = False,
        add_labels = True
    ):
        self.expert_dataset = expert_dataset
        self.general_dataset = general_dataset

        self.split = split
        self.frac = frac
        self.balanced = balanced

        self.make_lengths()

        self.normalizer = self.expert_dataset.normalizer
        self.action_dim = self.expert_dataset.action_dim

        self.add_labels = add_labels

        self.len = self.n_expert + self.n_general

        print(f"[utils/datasets.py] Lengths - Expert dataset: {self.n_expert} - General dataset: {self.n_general}")

    def __len__(self):
        return self.len

    def make_lengths(self):
        self.n_expert = len(self.expert_dataset)
        self.n_general = len(self.general_dataset)
        self.start_expert = 0
        self.start_general = 0

        if self.split != "all":
            share_expert = int(self.n_expert * self.frac)
            share_general = int(self.n_general * self.frac)

            self.n_expert = share_expert
            self.n_general = share_general

            if self.split == "test":
                self.start_expert = self.n_expert - share_expert
                self.start_general = self.n_general - share_general

        self.n_expert_base = self.n_expert
        self.n_general_base = self.n_general

        if self.balanced:
            self.n_expert = max(self.n_expert, self.n_general)
            self.n_general = self.n_expert
    
    def __getitem__(self, idx):
        if idx >= self.len and idx < 0:
            raise IndexError()
        
        if idx < self.n_expert:
            data = self.expert_dataset[self.start_expert + (idx % self.n_expert_base)]
            label = torch.tensor([1])
        else:
            data = self.general_dataset[(idx - self.n_expert) % self.n_general_base + self.start_general]
            label = torch.tensor([0])

        if self.add_labels:
            return LabeledBatch(*data, label)
        else:
            return data