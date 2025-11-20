import argparse
import os
import random

import numpy as np
import torch
import torchvision
from PIL import Image

import IA_diff_util
import lora
import my_utils
from GuidedDiffusion.guided_diffusion.gaussian_diffusion import get_named_beta_schedule
from GuidedDiffusion.guided_diffusion.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    classifier_defaults,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

args = argparse.ArgumentParser()
args.add_argument('--out_dir', type=str, default='')
args.add_argument('--in_dir', type=str, default='')
args.add_argument('--IA_data_dir', type=str, default='')
args.add_argument('--cond_type', type=str, default='LS', choices=['LS', 'L1'])
args.add_argument('--IA_loss', type=str, default='L1', choices=['L1', 'L2', 'MS_SSIM'])
args.add_argument('--IA_opt', type=str, default='Adam', choices=['Adam', 'SGD', 'MomSGD'])
args.add_argument('--T_fidelity_high', type=int, default=1000)
args.add_argument('--start_img', type=int, default=0)
args.add_argument('--n_imgs', type=int, default=50)
args.add_argument('--lora_r', type=int, default=8)
args.add_argument('--lora_alpha', type=float, default=1.0)
args.add_argument('--T_fidelity_low', type=int, default=0)
args.add_argument('--img_size', type=int, default=256)
args.add_argument('--fidelity_scale', type=float, default=10)
args.add_argument('--IA_t_range', type=int, default=[0, 1000], nargs=2)
args.add_argument('--num_workers', type=int, default=0)
args.add_argument('--lambda_TV', type=float, default=0)
args.add_argument('--IA_lr', type=float, default=1e-4)
args.add_argument('--EMA_weight', type=float, default=0.8)
args.add_argument('--IA_iterations', type=int, default=0)
args.add_argument('--IA_batch_size', type=int, default=1)
args.add_argument('--average', action='store_true')
args.add_argument('--IA_use_At', action='store_true')
args.add_argument('--generate_baseline', action='store_true')
args.add_argument('--num_IA_imgs', type=int, default=0)
args.add_argument('--seed', type=int, default=0)
args.add_argument('--verbose', action='store_true')
args.add_argument('--IA_crop_size', type=int, default=256)
args.add_argument('--IA_crop', action='store_true')
args.add_argument('--use_lpips', action='store_true')
args.add_argument('--IA_resized_crop', action='store_true')
args.add_argument('--suffix', type=str, default='')
# Define Diffusion model
defaults = dict(
    clip_denoised=True,
    num_samples=10000,
    batch_size=1,
    use_ddim=False,
    model_path="",
    classifier_path="",
    classifier_scale=1.0,
)
defaults.update(model_and_diffusion_defaults())
defaults.update(classifier_defaults())
add_dict_to_argparser(args, defaults)
args = args.parse_args()
args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
args.attention_resolutions = "32,16,8"
args.class_cond = False
args.diffusion_steps = 1000
args.image_size = 256
args.timestep_respacing = '1000'
args.learn_sigma = True
args.noise_schedule = 'linear'
args.num_channels = 256
args.num_head_channels = 64
args.num_res_blocks = 2
args.resblock_updown = True
args.use_fp16 = False
args.use_scale_shift_norm = True
args.rescale_timesteps = True

args.classifier_scale = 10.0
args.classifier_path = ''

if not os.path.isdir(args.out_dir):
    os.mkdir(args.out_dir)

args.dtype = torch.float
model, diffusion = create_model_and_diffusion(
    **args_to_dict(args, model_and_diffusion_defaults().keys())
)
model.to(args.device)
state_dict = torch.load(args.model_path, map_location="cpu")
model.load_state_dict(
    state_dict
)
model.to(args.device)
model.eval()
model.type(args.dtype)

model.requires_grad_(False)

model = lora.inject_lora(model, alpha=args.lora_alpha, r=args.lora_r)
lora.reset_lora(model)

# Define sampling operator
def S(x):
    return 0.3 * x[:,0:1] + 0.59 * x[:,1:2] + 0.11 * x[:,2:3]
def St(x):
    return x.repeat(1, 3, 1, 1)

Transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(args.img_size, interpolation=torchvision.transforms.functional.InterpolationMode.BICUBIC),
    torchvision.transforms.CenterCrop(args.img_size),
    torchvision.transforms.ToTensor()
])

suffix = f"s{args.fidelity_scale}_{args.cond_type}"

if args.lambda_TV > 0:
    suffix += f"_TV{args.lambda_TV}"
if args.IA_iterations > 0:
    suffix += f"_IA{args.IA_iterations}_{args.num_IA_imgs}NN"

from clip import clip

clip_model = clip.load("ViT-B/16", device=args.device, jit=False)


suffix += args.suffix

images = [os.path.join(args.in_dir, f) for f in os.listdir(args.in_dir) if '.png' in f or '.jpg' in f or '.JPEG' in f]
images.sort()
images = images[args.start_img:args.n_imgs]
print(f"Found {len(images)} inputs")

betas = np.array(get_named_beta_schedule(args.noise_schedule, args.IA_t_range[1]))
alphas = 1.0 - betas
model.alphas = torch.tensor(alphas).float().to(args.device)
model.alphas_cumprod = torch.tensor(np.cumprod(alphas, axis=0)).float().to(args.device)
def model_fn(x, t, y=None, **kwargs):
    assert y is not None
    return model(x, t, y if args.class_cond else None)

def cond_fn(xt, t, **kwargs):
    with torch.no_grad():
        out = kwargs["model"](xt, t)
        eps, __ = torch.split(out, 3, 1)
        a_bar = kwargs["alphas_cumprod"][int(torch.div(t, args.diffusion_steps/int(args.timestep_respacing), rounding_mode='trunc'))]
        yt = torch.sqrt(a_bar) * kwargs["y"] + torch.sqrt(1 - a_bar) * kwargs["S"](eps)
    return - my_utils.gLS(xt, yt, kwargs["S"], kwargs["St"]) * args.fidelity_scale

for idx, img_name in enumerate(images):
    I = Image.open(img_name).convert("RGB")
    x = Transforms(I).unsqueeze(0).to(args.device)
    y = S(x)
    y = 2 * y - 1

    if args.verbose:
        print(f"Processing batch {args.start_img + idx}/{len(images)}")

    args.image_name = f'{args.start_img + idx:03d}'

    if args.generate_baseline:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        with torch.no_grad():
            xt = diffusion.p_sample_loop(model_fn, (1, 3, args.img_size, args.img_size), cond_fn=cond_fn, device=args.device, progress=args.verbose,
                                         model_kwargs={"model": model, "y": y, "S": S, "St": St, "alphas_cumprod": model.alphas_cumprod, "alphas": model.alphas})
            xt = torch.clamp(0.5 * xt + 0.5, 0, 1)
            torchvision.utils.save_image(xt, args.out_dir + f"{1 + idx:05d}_color_s{args.fidelity_scale}_config0.png")


    if args.IA_iterations > 0:
        if args.num_IA_imgs > 0:
            with torch.no_grad():
                NN_imgs = IA_diff_util.find_img_NN(0.5*y + 0.5, clip_model=clip_model, args=args,
                                                      Transforms=torchvision.transforms.ToTensor() if args.IA_crop else Transforms)
            if args.IA_crop or args.IA_resized_crop:
                y_ = [2 * I - 1 for I in NN_imgs]
            else:
                y_ = 2*NN_imgs - 1
        else:
            y_ = y
        IA_diff_util.IA(y_, model, args)
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        with torch.no_grad():
            xt = diffusion.p_sample_loop(model_fn, (1, 3, args.img_size, args.img_size), cond_fn=cond_fn, device=args.device, progress=args.verbose,
                                         model_kwargs={"model": model, "y": y, "S": S, "St": St, "alphas_cumprod": model.alphas_cumprod, "alphas": model.alphas})

        xt = torch.clamp(0.5 * xt + 0.5, 0, 1)
        y = torch.clamp(0.5 * y + 0.5, 0, 1)
        torchvision.utils.save_image(xt, os.path.join(args.out_dir, f"{1 + idx:05d}_color_{suffix}.png"))
        torchvision.utils.save_image(y, os.path.join(args.out_dir, f"{1 + idx:05d}_color_y.png"))
        torchvision.utils.save_image(x, os.path.join(args.out_dir, f"{1 + idx:05d}_GT.png"))

        if args.IA_iterations > 0:
            lora.reset_lora(model)
        #     model.load_state_dict(state_dict)

print('Finished')
