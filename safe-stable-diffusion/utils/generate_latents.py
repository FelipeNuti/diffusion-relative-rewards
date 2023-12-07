import argparse
import os
import torch
from diffusers import StableDiffusionPipelineSafe
from diffusers.pipelines.stable_diffusion_safe import SafetyConfig
from datasets import load_dataset

parser = argparse.ArgumentParser(description='Generate images using stable diffusion and save intermediate latents')
parser.add_argument('job_index', type=int, help='job index')
parser.add_argument('dataset', type=str, help='expert or general')
parser.add_argument('--batch-size', type=int, default=16, help='batch size')
parser.add_argument('--num-batches', type=int, default=10, help='number of batches to generate')
parser.add_argument('--prompt-length', type=int, default=-1, help='length of prompt')
parser.add_argument('--device', type=str, default='cuda:0', help='device')
parser.add_argument('--half-precision', type=bool, default=True, help="use torch.float16 instead of torch.float32")
parser.add_argument('--safe-cache-dir', type=str, default='./huggingface_cache', help='cache directory for safe model')
parser.add_argument('--unsafe-cache-dir', type=str, default='./huggingface_cache', help='cache directory for unsafe model')
args = parser.parse_args()

device = torch.device(args.device)
save_dirs = {
    "general": './safe_stable_diffusion_data/train/general',
    "expert": './safe_stable_diffusion_data/train/expert',
}
save_dir = save_dirs[args.dataset]

print(f"Save dir: {save_dir}")

def get_prompt(i):
    s = dataset["train"][i]["prompt"]

    if args.prompt_length != -1:
        s = s[:args.prompt_length]

    return s

dataset_path = "AIML-TUDA/i2p" #dataset_paths[args.dataset]
dataset = load_dataset(dataset_path, cache_dir="./huggingface_cache")
print("Dataset loaded")

follow_safe_latent = args.dataset == "expert"

# Test dataset
print(get_prompt(0))

pipeline_orig = StableDiffusionPipelineSafe.from_pretrained(
    "AIML-TUDA/stable-diffusion-safe", 
    torch_dtype=torch.float16 if args.half_precision else torch.float32, 
    device_map="auto",
    safety_checker = None,
)
pipeline = pipeline_orig.to(device)
print("Model loaded")

print(f"Set max prompt length to {args.prompt_length}")

if not os.path.exists(save_dir):
    print(f"Path {save_dir} not found")
    os.makedirs(save_dir)
    print(f"Path {save_dir} was created")

def generate_images_and_latents(device, start_index, end_index):
    current_idx = start_index

    for batch_index in range(args.num_batches):
        torch.cuda.empty_cache()
        if current_idx >= end_index:
            return
        next_idx = min(end_index, current_idx+args.batch_size)
        prompts = [get_prompt(i) for i in range(current_idx, next_idx)]

        diffusion_step_latents_safe = []
        diffusion_step_latents_unsafe = []

        def callback(i, t, latents_unsafe, latents_safe):
            diffusion_step_latents_unsafe.append(latents_unsafe)
            diffusion_step_latents_safe.append(latents_safe)

        with torch.no_grad():
            pipeline.get_deltas(follow_safe_latent=follow_safe_latent, width=512, height=512, num_inference_steps = 50, prompt=prompts, callback=callback, **SafetyConfig.MAX)
            torch.cuda.empty_cache()

        diffusion_step_latents_safe.reverse()
        diffusion_step_latents_unsafe.reverse()
        batch_latents_safe = torch.stack(diffusion_step_latents_safe, dim = 1)
        batch_latents_unsafe = torch.stack(diffusion_step_latents_unsafe, dim = 1)
        latents = torch.stack([batch_latents_unsafe, batch_latents_safe], dim = 2)

        filename = f"latents_{current_idx}_{next_idx}.pt"
        savepath = os.path.join(save_dir, filename)
        torch.save(latents, savepath)
        print(f"Saved latent differences (shape {latents.shape}) to {savepath}")

        del latents, diffusion_step_latents_safe, diffusion_step_latents_unsafe
        del batch_latents_safe, batch_latents_unsafe

        current_idx = next_idx


if __name__ == "__main__":

    job_index = args.job_index
    start_index = job_index * args.batch_size * args.num_batches
    end_index = min((job_index + 1) * args.batch_size * args.num_batches, len(dataset["train"]))

    generate_images_and_latents(device, start_index, end_index)


