# Environment setup

Create a new conda environment with python 3.8:
```
conda create --nama safe_env_diffuser python=3.8
conda activate safe_env_diffuser
```

Install the requirements:
```
pip install -r requirements.txt
```

Install our version of the `diffusers` library, modified to allow for extracting the intermediate latents of the diffusion process of the expert and base models:
```
cd diffusers
pip install -e .
```

# Obtaining the latents 

## Downloading the latents

To download the latents, run:

```
chmod +x download_extract_safe_data.sh
./download_extract_safe_data.sh
```

The folder structure of the extracted file should look like:
```
safe_stable_diffusion_data/                                                                                               
├── train                                                                                                                 
    ├── expert                                                                                                            
    │   ├── merged.pt  
    ├── general                                                                                                            
    │   ├── merged.pt  
```

## Generating the latents

Generate the latents using `utils/generate_latents.py`. The script supports having several processes dumping latents in parallel. 

You will need to login to Huggingface using `huggingface-cli login` in order to access Safe Stable Diffusion.

These are the exact commands we used:
```
python /users/nuti/code_remote/safe-stable-diffusion-irl/utils/generate_latents.py 0 expert --batch-size 100 --num-batches 12 --half-precision True
python /users/nuti/code_remote/safe-stable-diffusion-irl/utils/generate_latents.py 1 expert --batch-size 100 --num-batches 12 --half-precision True
python /users/nuti/code_remote/safe-stable-diffusion-irl/utils/generate_latents.py 2 expert --batch-size 100 --num-batches 12 --half-precision True
python /users/nuti/code_remote/safe-stable-diffusion-irl/utils/generate_latents.py 3 expert --batch-size 100 --num-batches 12 --half-precision True

python /users/nuti/code_remote/safe-stable-diffusion-irl/utils/generate_latents.py 0 general --batch-size 100 --num-batches 12 --half-precision True
python /users/nuti/code_remote/safe-stable-diffusion-irl/utils/generate_latents.py 1 general --batch-size 100 --num-batches 12 --half-precision True
python /users/nuti/code_remote/safe-stable-diffusion-irl/utils/generate_latents.py 2 general --batch-size 100 --num-batches 12 --half-precision True
python /users/nuti/code_remote/safe-stable-diffusion-irl/utils/generate_latents.py 3 general --batch-size 100 --num-batches 12 --half-precision True
```

Once these latents are dumped, merge them using the `utils/merge_latents.py` script:
```
python ./utils/merge_latents.py ./safe_stable_diffusion_data/train
```

# Reproducing the main plot in the paper

```
python ./scripts/train_gradient_matching.py --batch_size 64 --dim 32 --lr 0.0001 --train_frac 0.6 --seed 4 --gradient_clipping 0.002
```