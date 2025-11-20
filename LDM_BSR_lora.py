import argparse
import glob
import os
import random

import numpy as np
import torch
import torchvision
from omegaconf import OmegaConf
from PIL import Image

import IA_diff_util
import lora
import my_utils
from StableDiffusion.ldm.models.diffusion.ddim import DDIMSampler
from StableDiffusion.main import instantiate_from_config
from StableDiffusion.notebook_helpers import get_cond, get_cond_options, get_model, run

args = argparse.ArgumentParser()
args.add_argument("--in_dir", type=str)
args.add_argument("--out_dir", type=str)
args.add_argument("--steps", type=int, default=500)
args.add_argument("--sf", type=int, default=4)
args.add_argument("--IA_iterations", type=int, default=0)
args.add_argument('--IA_batch_size', type=int, default=1)
args.add_argument('--IA_loss', type=str, default='L2', choices=['L1', 'L2'])
args.add_argument('--IA_opt', type=str, default='Adam', choices=['Adam', 'SGD', 'MomSGD'])
args.add_argument('--n_imgs', type=int, default=50)
args.add_argument('--lora_r', type=int, default=8)
args.add_argument('--lora_alpha', type=float, default=1.0)
args.add_argument('--IA_t_range', type=int, default=[0, 1000], nargs=2)
args.add_argument('--IA_lr', type=float, default=1e-4)
args.add_argument('--EMA_weight', type=float, default=0.8)
args.add_argument('--suffix', type=str, default='')
args.add_argument('--num_IA_imgs', type=int, default=0)
args.add_argument('--generate_baseline', action='store_true')
args.add_argument('--img_size', type=int, default=1024)
args.add_argument('--start_img', type=int, default=0)
args.add_argument('--seed', type=int, default=0)
args.add_argument('--latent_scale', type=int, default=4)
args.add_argument('--verbose', action='store_true')
args.add_argument('--IA_use_sample_timesteps', action='store_true')
args.add_argument('--IA_VAE', action='store_true')
args.add_argument('--use_lpips', action='store_true')
args.add_argument('--rnd_knn', action='store_true')
args.add_argument('--VAE_EMA_weight', type=float, default=0)
args.add_argument('--VAE_IA_lr', type=float, default=1e-4)
args.add_argument('--IA_crop_size', type=int, default=256)
args.add_argument('--IA_crop', action='store_true')
args.add_argument('--IA_data_dir', type=str, default='')
args = args.parse_args()

if not os.path.isdir(args.out_dir):
    os.mkdir(args.out_dir)

torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

images = [os.path.join(args.in_dir, f) for f in os.listdir(args.in_dir) if '.png' in f or '.jpg' in f or '.JPEG' in f]
images.sort()
images = images[args.start_img:args.n_imgs]
print(f"Found {len(images)} inputs")

# config = OmegaConf.load("StableDiffusion/models/ldm/bsr_sr/config.yaml")
# model = instantiate_from_config(config.model)
# state_dict = torch.load("StableDiffusion/models/ldm/bsr_sr/last.ckpt")["state_dict"]
# model.load_state_dict(state_dict, strict=False)

dir, options = get_cond_options('superresolution')
model = get_model('superresolution')["model"]
state_dict = model.state_dict()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
args.device = device
model = model.to(args.device)

model.requires_grad_(False)

model = lora.inject_lora(model, alpha=args.lora_alpha, r=args.lora_r)
lora.reset_lora(model)

sampler = DDIMSampler(model)

# Define sampling operator
s = my_utils.get_bicubic(args.sf).to(args.device)
s_norm = s / s.sum()
S = lambda x: my_utils.fft_Down_(x, s_norm, args.sf).real
St = lambda x: my_utils.fft_Up_(x, s, args.sf).real

Transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(args.img_size, interpolation=torchvision.transforms.functional.InterpolationMode.BICUBIC),
    torchvision.transforms.CenterCrop(args.img_size),
    torchvision.transforms.ToTensor()
])

suffix = ""

if args.IA_iterations > 0:
    suffix += f"_IA{args.IA_iterations}_{args.num_IA_imgs}NN"
    if args.num_IA_imgs > 0:
        from clip import clip
        clip_model = clip.load("ViT-B/16", device=args.device, jit=False)

suffix += args.suffix

for idx, img_name in enumerate(images):
    I = Image.open(img_name).convert("RGB")
    x = Transforms(I).unsqueeze(0).to(args.device)
    y = S(x)
    y = 2 * y - 1

    print(f"Processing batch {idx}/{len(images)}")

    args.image_name = f'{idx:03d}'

    if args.generate_baseline:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        with torch.no_grad():
            shape = (y.shape[1],)+y.shape[2:]
            # samples_ddim, intermediates = sampler.sample(S=args.steps, conditioning=y, batch_size=y.shape[0], shape=shape, verbose=False)
            samples_ddim, intermediates = sampler.sample(args.steps, batch_size=y.shape[0], shape=shape, conditioning=y, callback=None,
                                         normals_sequence=None, quantize_x0=False, eta=0.0, mask=None, x0=None,
                                         temperature=1.0, verbose=False, score_corrector=None, corrector_kwargs=None, x_T=None)
            x_samples_ddim = model.decode_first_stage(samples_ddim)
            predicted_image = torch.clamp(0.5 * x_samples_ddim + 0.5, 0, 1)
            torchvision.utils.save_image(predicted_image, os.path.join(args.out_dir, f"{args.image_name}_LDM_SRx{args.sf}_config0.png"))
            torchvision.utils.save_image(torch.clamp(0.5 * y + 0.5, 0, 1), os.path.join(args.out_dir, f"{args.image_name}_SRx{args.sf}_y.png"))
            torchvision.utils.save_image(St(torch.clamp(0.5 * y + 0.5, 0, 1)), os.path.join(args.out_dir, f"{args.image_name}_SRx{args.sf}_Sty.png"))
            torchvision.utils.save_image(x, os.path.join(args.out_dir, f"{args.image_name}_GT.png"))


    if args.IA_iterations > 0:
        if args.num_IA_imgs > 0:
            with torch.no_grad():
                NN_imgs_tensor = IA_diff_util.find_img_NN(0.5*y + 0.5, clip_model=clip_model, args=args,
                                                      Transforms=torchvision.transforms.ToTensor())
            y_ = [2*I - 1 for I in NN_imgs_tensor]
        else:
            y_ = y

        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        def model_fn(z_, t_, c, **kwargs):
            return model.apply_model(z_, t_, c)
        if args.IA_use_sample_timesteps:
            sampler.make_schedule(ddim_num_steps=args.steps, ddim_eta=0, verbose=False)
        IA_diff_util.IA_latent(y_, model, model.encode_first_stage, args, model_fn=model_fn,
                               timesteps=torch.tensor(sampler.ddim_timesteps, device=args.device) if args.IA_use_sample_timesteps else None, S=S)
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        with torch.no_grad():
            shape = (y.shape[1],)+y.shape[2:]
            # samples_ddim, intermediates = sampler.sample(S=args.steps,
            #                                  conditioning=y,
            #                                  batch_size=y.shape[0],
            #                                  shape=shape,
            #                                  verbose=False)
            samples_ddim, intermediates = sampler.sample(args.steps, batch_size=y.shape[0], shape=shape, conditioning=y, callback=None,
                                         normals_sequence=None, quantize_x0=False, eta=0.0, mask=None, x0=None,
                                         temperature=1.0, verbose=False, score_corrector=None, corrector_kwargs=None, x_T=None)
            x_samples_ddim = model.decode_first_stage(samples_ddim)
            predicted_image = torch.clamp(0.5 * x_samples_ddim + 0.5, 0, 1)
            y = torch.clamp(0.5 * y + 0.5, 0, 1)

            torchvision.utils.save_image(predicted_image, os.path.join(args.out_dir, f"{args.image_name}_LDM_SRx{args.sf}_{suffix}.png"))
            torchvision.utils.save_image(y, os.path.join(args.out_dir, f"{args.image_name}_SRx{args.sf}_y.png"))
            torchvision.utils.save_image(St(y), os.path.join(args.out_dir, f"{args.image_name}_SRx{args.sf}_Sty.png"))
            torchvision.utils.save_image(x, os.path.join(args.out_dir, f"{args.image_name}_GT.png"))

        lora.reset_lora(model)

print('Finished')
