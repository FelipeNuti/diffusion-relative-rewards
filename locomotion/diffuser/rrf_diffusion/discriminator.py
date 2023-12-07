import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad, Variable

class TrajectoryDiscriminator(nn.Module):
    def __init__(
        self,
        model,
        diffusion
    ):
        super().__init__()
        torch.autograd.set_detect_anomaly(True)
        self.model = model 
        self.diffusion = diffusion

    def forward(self, x_t, cond, t):
        out = self.model(x_t, cond, t)
        return F.sigmoid(out)

    def loss(self, x_start, cond, labels, t = None):
        batch_size = x_start.shape[0]

        if t is None:
            t = torch.randint(0, self.diffusion.n_timesteps, (batch_size,), device=x_start.device).long()

        x_t = self.diffusion.q_sample(x_start, t)

        probs = self.model(x_t, cond, t)
        preds = (probs >= 0.5)
        loss = F.binary_cross_entropy_with_logits(probs.float(), labels.float())
        accuracy = (labels == preds).sum() / batch_size

        return loss, accuracy


