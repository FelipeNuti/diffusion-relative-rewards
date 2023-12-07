import torch

def check_nan(x, name):
    if torch.isnan(x).any():
        print(f"{name} is NAN")
        return breakpoint
    else:
        return lambda: None

class ActivationExtractor:
    def __init__(self, model, enabled = True):
        self.activations = {}

        if enabled:
            hooked = self.register(model, name = "base_model")
            print(f"[utils/debugging] {hooked} modules hooked")
        else:
            print(f"[utils/debugging] Activation Extraction Disabled")

    def get_hook(self, name):
        def hook(model, input, output):
            self.activations[name] = output.detach()
        return hook

    def register(self, p, name = ""):
        hooked = 0
        if hasattr(p, "register_forward_hook"): #and hasattr(p, "requires_grad") and p.requires_grad:
            p.register_forward_hook(self.get_hook(name))
            hooked += 1
        for child_name, child in p.named_children():
            hooked += self.register(child, name = child_name)
        return hooked


    def stats(self, name):
        out = self.activations[name]
        return {
            "max": out.max(),
            "min": out.min(),
            "l1_mean": out.abs().mean(),
            "is_nan": torch.isnan(out).any()
        }

    def all_stats(self):
        return {k: self.stats(k) for k in self.activations.keys()}

    def max_activation(self):
        sup = 0.0

        for name, out in self.activations.items():
            sup = max(sup, out.abs().max())

        return sup
    
    def clear(self):
        self.activations = {}