import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad, Variable
import einops

def check_nan(x, name):
    if torch.isnan(x).any():
        print(f"{name} is NAN")
        return breakpoint
    else:
        return lambda: None


class GradientRewardRegressor(nn.Module):
    def __init__(
        self,
        model,
        n_timesteps=50,
    ):
        super().__init__()
        torch.autograd.set_detect_anomaly(True)
        self.model = model
        self.n_timesteps = n_timesteps
        self.diffusion_predicts_mean = True 
        self.scale_scores = False 
    
    def forward(self, x_t, t):
        return self.model(x_t, t)
    
    def _get_gradients(self, latent, t, eval_mode = False):
        latent = Variable(latent, requires_grad = True)

        out = self.forward(latent, t)
        assert eval_mode or out.requires_grad

        gradients = grad(
            out, latent, 
            retain_graph=True, 
            create_graph=True,
            allow_unused = False,
            grad_outputs=torch.ones(out.size(), device = latent.device)
        )[0]

        return gradients, {"out": out, "grad_wrt_x": gradients}
    
    def loss(self, latent, t, next_latent_general, next_latent_expert, eval_mode = False):
        batch_size = latent.shape[0]

        gradients, info = self._get_gradients(latent, t, eval_mode = eval_mode)
        assert gradients.shape == latent.shape

        difference = next_latent_expert - next_latent_general

        info["t"] = t
        info["target"] = difference

        loss = F.mse_loss(
            gradients,
            difference
        )

        return loss, info
