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
        s_expert,
        s_general,
        alpha=1e-3,
        eps_loss=1e-5,
        diffusion_predicts_mean=True,
        scale_scores = False,
    ):
        super().__init__()
        torch.autograd.set_detect_anomaly(True)
        self.model = model # use temporal value net
        self.s_expert =  s_expert
        self.s_general = s_general
        self.alpha = alpha
        self.n_timesteps = s_expert.n_timesteps
        self.diffusion_predicts_mean = diffusion_predicts_mean
        self.eps_loss = eps_loss
        self.scale_scores = scale_scores

    def forward(self, x_t, cond, t):
        return self.model(x_t, cond, t)
    
    def get_diffusion_out(self, diffusion_model, *args):
        if self.diffusion_predicts_mean:
            return diffusion_model.p_mean_variance(*args)
        else:
            return diffusion_model.p_score(*args, scaled = self.scale_scores)

    def _get_gradients(self, x_t, cond, t, eval_mode = False):
        x_t = Variable(x_t, requires_grad = True)

        out = self.forward(x_t, cond, t)
        assert eval_mode or out.requires_grad

        gradients = grad(
            out, x_t, 
            retain_graph=True, 
            create_graph=True,
            allow_unused = False,
            grad_outputs=torch.ones(out.size(), device = x_t.device)
        )[0]

        return gradients, {"out": out, "grad_wrt_x": gradients}
    
    def loss(self, x_start, cond, t = None, eval_mode = False):
        batch_size = x_start.shape[0]

        if t is None:
            t = torch.randint(0, self.n_timesteps, (batch_size,), device=x_start.device).long()
        
        x_t = self.s_expert.q_sample(x_start, t)

        with torch.no_grad():
            out_expert, _, posterior_log_variance = self.get_diffusion_out(self.s_expert, x_t, cond, t)
            out_general, _, _ = self.get_diffusion_out(self.s_general, x_t, cond, t)

            stdev = torch.exp(0.5 * posterior_log_variance)

        gradients, info = self._get_gradients(out_general, cond, t, eval_mode = eval_mode)
        assert gradients.shape == x_t.shape

        if self.diffusion_predicts_mean:
            difference = out_expert - out_general
        else:
            difference = out_general - out_expert

        info["t"] = t
        info["target"] = difference

        loss = F.mse_loss(
            gradients,# + self.eps_loss,
            difference
        )

        loss += self.eps_loss * (info["out"] * info["out"]).mean() #(gradients ** 2).mean()

        return loss, info

class GradientRewardSingleStep(GradientRewardRegressor):

    def forward(self, x_t, cond, t):
        sample_horizon = x_t.shape[1]
        model_horizon = self.model.horizon

        assert sample_horizon % model_horizon == 0, f"Sample horizon ({sample_horizon}) must be a multiple of model horizon {model_horizon}"
        n_windows = sample_horizon // model_horizon

        x_flat = einops.rearrange(x_t, "b (mh nw) c -> (b nw) mh c", nw = n_windows, mh = model_horizon) 
        t_flat = t.reshape((-1, 1)).repeat((1, n_windows)).flatten()
        out_flat = self.model(x_flat, cond, t_flat)

        out = einops.reduce(out_flat, "(b nw) 1 -> b 1", "mean", nw = n_windows)
        return out

