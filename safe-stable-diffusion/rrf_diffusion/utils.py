import time
import torch
import numpy as np
import random

def check_nan(x, name):
    if torch.isnan(x).any():
        print(f"{name} is NAN")
        return breakpoint
    else:
        return lambda: None

def to_np(x):
	if torch.is_tensor(x):
		x = x.detach().cpu().numpy()
	return x

def param_to_module(param):
	module_name = param[::-1].split('.', maxsplit=1)[-1][::-1]
	return module_name

def _to_str(num):
	if num >= 1e6:
		return f'{(num/1e6):.2f} M'
	else:
		return f'{(num/1e3):.2f} k'

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def report_parameters(model, topk=10):
	counts = {k: p.numel() for k, p in model.named_parameters()}
	n_parameters = sum(counts.values())
	print(f'[ utils/arrays ] Total parameters: {_to_str(n_parameters)}')

	modules = dict(model.named_modules())
	sorted_keys = sorted(counts, key=lambda x: -counts[x])
	max_length = max([len(k) for k in sorted_keys])
	for i in range(topk):
		key = sorted_keys[i]
		count = counts[key]
		module = param_to_module(key)
		print(' '*8, f'{key:10}: {_to_str(count)} | {modules[module]}')

	remaining_parameters = sum([counts[k] for k in sorted_keys[topk:]])
	print(' '*8, f'... and {len(counts)-topk} others accounting for {_to_str(remaining_parameters)} parameters')
	return n_parameters

class Timer:

	def __init__(self):
		self._start = time.time()

	def __call__(self, reset=True):
		now = time.time()
		diff = now - self._start
		if reset:
			self._start = now
		return diff

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
        if hasattr(p, "register_forward_hook"): 
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

def plot2img(fig, remove_margins=True):
    # https://stackoverflow.com/a/35362787/2912349
    # https://stackoverflow.com/a/54334430/2912349

    from matplotlib.backends.backend_agg import FigureCanvasAgg

    if remove_margins:
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    img_as_string, (width, height) = canvas.print_to_buffer()
    return np.fromstring(img_as_string, dtype='uint8').reshape((height, width, 4))