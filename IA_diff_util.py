import copy
import os
import time

import kornia
import kornia.augmentation as K
import lpips
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image

import my_utils


def IA(IA_imgs, denoiser, args, loss_mask=None, model_fn=None, timesteps=None, **kwargs):
    if args.IA_loss == 'L1':
        objective = torch.nn.L1Loss()
    elif args.IA_loss == 'L2':
        objective = torch.nn.MSELoss()
    elif args.IA_loss == 'MS_SSIM':
        objective = kornia.losses.MS_SSIMLoss().to(args.device)
    if args.use_lpips:
        percep_obj = lpips.LPIPS(net='vgg').to(args.device)
    optimizer = torch.optim.Adam(denoiser.parameters(), lr=args.IA_lr)

    if args.EMA_weight > 0:
        state_dict_ema = copy.deepcopy(denoiser.state_dict())

    if args.IA_crop:
        T = K.RandomCrop((args.IA_crop_size, args.IA_crop_size)).to(args.device)
        e = torch.randn(min(args.IA_batch_size, args.num_IA_imgs), 3, args.IA_crop_size, args.IA_crop_size, device=args.device)
    elif  args.IA_resized_crop:
        T = K.RandomResizedCrop((args.IA_crop_size, args.IA_crop_size), resample=kornia.constants.Resample.BICUBIC, scale=(0.25, 1)).to(args.device)
        e = torch.randn(min(args.IA_batch_size, args.num_IA_imgs), 3, args.IA_crop_size, args.IA_crop_size, device=args.device)
    else:
        e = torch.randn(min(args.IA_batch_size, args.num_IA_imgs), 3, IA_imgs.shape[2], IA_imgs.shape[3], device=args.device)

    for i in range(args.IA_iterations):
        tik = time.time()
        with torch.no_grad():
            e.normal_()
            if timesteps is not None:
                t = timesteps[torch.randint(low=0, high=timesteps.shape[0], size=(min(args.IA_batch_size, IA_imgs.shape[0]),))]
            else:
                t = torch.randint(low=args.IA_t_range[0], high=args.IA_t_range[1] - 1, size=(1,), device=args.device)
                if args.num_IA_imgs > 0:
                    t = t.repeat(min(args.IA_batch_size, args.num_IA_imgs))
            a_bar = denoiser.alphas_cumprod.index_select(0, t).view(-1, 1, 1, 1)
            if args.num_IA_imgs > 0:
                b_idx = torch.randperm(args.num_IA_imgs)[:min(args.IA_batch_size, args.num_IA_imgs)]
                if args.IA_crop or args.IA_resized_crop:
                    imgs = [IA_imgs[n] for n in b_idx]
                    x_0 = torch.cat([T(I) for I in imgs], dim=0)
                else:
                    x_0 = IA_imgs[b_idx]
            else:
                x_0 = IA_imgs
            xt = torch.sqrt(a_bar) * x_0 + torch.sqrt(1 - a_bar) * e

        optimizer.zero_grad(set_to_none=True)

        if model_fn is None:
            out = denoiser(xt, t.float(), **kwargs)
        else:
            out = model_fn(xt, t.float(), **kwargs)
        epsilon = out[:, :x_0.shape[1], :, :]
        if loss_mask is not None:
            loss = objective(loss_mask * epsilon, loss_mask * e)
        else:
            loss = objective(epsilon, e) + (0.1 * torch.mean(percep_obj(epsilon, e)) if args.use_lpips else 0)

        loss.backward()
        optimizer.step()
        if args.EMA_weight > 0:
            state_dict_ema = my_utils.EMA_update(state_dict_ema, denoiser.state_dict(), args.EMA_weight)
            denoiser.load_state_dict(state_dict_ema)
        tok = time.time()
        if args.verbose:
            print(
                f"\r Image adaptive iterations: {i}/{args.IA_iterations} | loss = {loss.item(): .5f}| time = {tok - tik: .4f}",
                end='')
    optimizer.zero_grad()
    denoiser.eval()
    print('\n')


def IA_latent(IA_imgs, denoiser, encoder, args, loss_mask=None, model_fn=None, timesteps=None, embeddings=None, optimizer=None, **kwargs):
    if args.IA_loss == 'L1':
        objective = torch.nn.L1Loss()
    elif args.IA_loss == 'MS_SSIM':
        objective = kornia.losses.MS_SSIMLoss().to(args.device)
    else:
        objective = torch.nn.MSELoss()
    if args.use_lpips:
        percep_obj = lpips.LPIPS(net='vgg').to(args.device)

    if optimizer is None:
        optimizer = torch.optim.Adam(denoiser.parameters(), lr=args.IA_lr)

    if args.EMA_weight > 0:
        state_dict_ema = copy.deepcopy(denoiser.module.state_dict()) if isinstance(denoiser, torch.nn.DataParallel) else copy.deepcopy(denoiser.state_dict())

    if args.IA_crop or args.IA_resized_crop:
        if args.IA_crop:
            T = K.RandomCrop((args.IA_crop_size, args.IA_crop_size)).to(args.device)
        elif  args.IA_resized_crop:
            T = K.RandomResizedCrop((args.IA_crop_size, args.IA_crop_size), resample=kornia.constants.Resample.BICUBIC).to(args.device)
        e = torch.randn(min(args.IA_batch_size, args.num_IA_imgs) if args.num_IA_imgs > 0 else args.num_IA_imgs, 4, args.IA_crop_size//args.latent_scale, args.IA_crop_size//args.latent_scale, device=args.device, dtype=IA_imgs[0].dtype)
    else:
        e = torch.randn(min(args.IA_batch_size, args.num_IA_imgs) if args.num_IA_imgs > 0 else 1, 3, IA_imgs.shape[2]//args.latent_scale, IA_imgs.shape[3]//args.latent_scale, device=args.device, dtype=IA_imgs[0].dtype)
        with torch.no_grad():
            z_NN = encoder(IA_imgs)

    for i in range(args.IA_iterations):
        tik = time.time()
        with torch.no_grad():
            e.normal_()
            if timesteps is not None:
                t = timesteps[torch.randint(low=0, high=timesteps.shape[0], size=(min(args.IA_batch_size, IA_imgs.shape[0]),))]
            else:
                t = torch.randint(low=args.IA_t_range[0], high=args.IA_t_range[1] - 1, size=(1,), device=args.device)
                t = t.repeat(min(args.IA_batch_size, args.num_IA_imgs)).type(IA_imgs[0].dtype)
            a_bar = denoiser.alphas_cumprod.index_select(0, t.int()).view(-1, 1, 1, 1).type(IA_imgs[0].dtype)
            b_idx = torch.randperm(args.num_IA_imgs)[:min(args.IA_batch_size, args.num_IA_imgs)]
            if args.IA_crop or args.IA_resized_crop:
                imgs = [IA_imgs[n] for n in b_idx]
                x_0 = torch.cat([T(I) for I in imgs], dim=0)
                z_0 = encoder(x_0)
            else:
                z_0 = z_NN[b_idx]
                x_0 = IA_imgs[b_idx]
            if embeddings is not None:
                kwargs["embeddings"] = embeddings[b_idx]
            if 'S' in kwargs.keys():
                c = kwargs['S'](x_0)
                kwargs['c'] = c
            else:
                kwargs['x_0'] = x_0
            zt = torch.sqrt(a_bar) * z_0 + torch.sqrt(1 - a_bar) * e

        optimizer.zero_grad(set_to_none=True)
        if model_fn is None:
            epsilon = denoiser(zt.detach(), t.float(), **kwargs)
        else:
            epsilon = model_fn(zt.detach(), t.float(), **kwargs)

        if loss_mask is not None:
            loss = objective(loss_mask * epsilon, loss_mask * e)
        else:
            if args.use_lpips:
                loss = objective(epsilon, e) + (0.1 * torch.mean(percep_obj(epsilon, e)) if args.use_lpips else 0)
            else:
                loss = objective(epsilon, e)
        loss.backward(retain_graph=False)
        optimizer.step()
        if args.EMA_weight > 0:
            with torch.no_grad():
                new_state_dict = denoiser.module.state_dict() if isinstance(denoiser, torch.nn.DataParallel) else denoiser.state_dict()
                state_dict_ema = my_utils.EMA_update(state_dict_ema, new_state_dict, args.EMA_weight)
                if isinstance(denoiser, torch.nn.DataParallel):
                    denoiser.module.load_state_dict(state_dict_ema)
                else:
                    denoiser.load_state_dict(state_dict_ema)
        tok = time.time()
        if args.verbose:
            print(
                f"\r Image adaptive iterations: {i}/{args.IA_iterations} | loss = {loss.item(): .5f}| time = {tok - tik: .4f}",
                end='', flush=True)
    optimizer.zero_grad(set_to_none=True)
    print('\n')


def IA_Enc(imgs, latents, vae, args):
    objective = lpips.LPIPS(pretrained=True, net='vgg', lpips=False).to(args.device)
    for p in vae.decoder.parameters():
        p.requires_grad = True
    vae.decoder.train()
    optimizer = torch.optim.Adam(vae.decoder.parameters(), lr=args.VAE_IA_lr)

    if args.VAE_EMA_weight > 0:
        state_dict_ema = copy.deepcopy(vae.decoder.state_dict())

    for i in range(args.IA_iterations):
        with torch.no_grad():
            idx = torch.randint(low=0, high=imgs.shape[0], size=(min(args.IA_batch_size, imgs.shape[0]), ), device=args.device)
            x = imgs[idx, :, :, :]
            y = latents[idx, :, :, :]
            tik = time.time()
        optimizer.zero_grad(set_to_none=True)
        x_hat = vae.decode(y)
        loss = torch.mean(objective(x_hat, x))
        loss.backward()
        optimizer.step()
        if args.VAE_EMA_weight > 0:
            state_dict_ema = my_utils.EMA_update(state_dict_ema, vae.decoder.state_dict(), args.VAE_EMA_weight)
            vae.decoder.load_state_dict(state_dict_ema)
        tok = time.time()
        print(
            f"\r VAE adaptation iterations: {i}/{args.IA_iterations} | loss = {loss.item(): .5f}| time = {tok - tik: .4f}",
            end='')
    optimizer.zero_grad()
    print('\n')



def find_prompt_NN(clip_model, args, Transforms, return_embd=False):
    NN = []
    if os.path.isdir(os.path.join(args.out_dir, 'NN')):
        existing_imgs = [f for f in os.listdir(os.path.join(args.out_dir, 'NN')) \
                         if f"Nearest_Neighbor_{args.IA_prompt.replace(' ', '_')}_0" in f]
    else:
        existing_imgs = []
        os.mkdir(os.path.join(args.out_dir, 'NN'))
    if len(existing_imgs) >= args.num_IA_imgs:
        existing_imgs = existing_imgs[:args.num_IA_imgs]
        existing_imgs.sort()
        for I in existing_imgs:
            NN.append(['', '', os.path.join(args.out_dir, 'NN', I)])
        if args.verbose:
            print(f'Loaded existing NN with prompt \"{args.IA_prompt}\"')
    else:
        encoded_text = clip_model[0].encode_text(clip.tokenize(args.IA_prompt).to(args.device))
        encoded_text = F.normalize(encoded_text.float().cpu(), dim=-1).float()
        encoded_imgs = [[os.path.join(args.IA_data_dir, f), 0, ''] for f in os.listdir(args.IA_data_dir)]
        with torch.no_grad():
            for idx, img in enumerate(encoded_imgs):
                state = torch.load(img[0], map_location="cpu")
                img_embedding, img_file = state['Encoded Image'], state['Directory']
                img_embedding = F.normalize(img_embedding.float(), dim=-1)
                dist = (img_embedding - encoded_text).norm(dim=-1).div(2).arcsin().pow(2).mul(2)
                img[1] = dist
                img[2] = img_file
                if args.verbose:
                    print(f"\r Computing CLIP distance {idx}/{len(encoded_imgs)}", end="")

        mins = [[np.nan, np.inf] for __ in range(args.num_IA_imgs)]
        if args.verbose:
            print(f"\n Looking for top {args.num_IA_imgs} images")
        for m, min in enumerate(mins):
            for idx, img in enumerate(encoded_imgs):
                if img[1] < min[1]:
                    if m == 0:
                        min[0], min[1] = idx, img[1]
                    elif mins[m - 1][1] < img[1]:
                        min[0], min[1] = idx, img[1]
            NN.append(encoded_imgs[min[0]])
    NN_imgs = []
    for idx, nn in enumerate(NN):
        I = Image.open(nn[2]).convert('RGB')
        NN_imgs.append(Transforms(I).unsqueeze(0).to(args.device))

    if not (args.IA_crop or args.IA_resized_crop):
        NN_imgs = torch.cat(NN_imgs, dim=0)

    if len(existing_imgs) < args.num_IA_imgs:
        if not os.path.isdir(os.path.join(args.out_dir, f'NN')):
                os.mkdir(os.path.join(args.out_dir, f'NN'))
        if args.IA_crop or args.IA_resized_crop:
            for i, I in enumerate(NN_imgs):
                torchvision.utils.save_image(I, os.path.join(args.out_dir, "NN", f"Nearest_Neighbor_{args.IA_prompt.replace(' ', '_')}_{i:03d}.png"))
        else:
            for i in range(NN_imgs.shape[0]):
                torchvision.utils.save_image(NN_imgs[i:i+1], os.path.join(args.out_dir, "NN", f"Nearest_Neighbor_{args.IA_prompt.replace(' ', '_')}_{i:03d}.png"))

    if return_embd:
        T_clip = torchvision.transforms.Compose([
        K.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                K.CenterCrop(clip_model[0].visual.input_resolution),
                K.Resize(clip_model[0].visual.input_resolution, resample=kornia.constants.Resample.BICUBIC)])
        T_NN_imgs = torch.cat([T_clip(I) for I in NN_imgs], dim=0)
        encoded_imgs = clip_model[0].encode_image(T_NN_imgs)
        encoded_imgs = F.normalize(encoded_imgs, dim=-1).float()
        return NN_imgs, encoded_imgs

    return NN_imgs


def find_img_NN(img, clip_model, args, Transforms=None):
    NN = []
    if not os.path.isdir(os.path.join(args.out_dir, f'NN_{int(args.img_size)}')):
        os.mkdir(os.path.join(args.out_dir, f'NN_{int(args.img_size)}'))
    if hasattr(args, 'image_name') and os.path.isdir(os.path.join(args.out_dir, f'NN_{int(args.img_size)}')):
        existing_imgs = [f for f in os.listdir(os.path.join(args.out_dir, f'NN_{int(args.img_size)}')) \
                         if f"{args.image_name.split('.')[0]}_NearestNeighbor_" in f]
    else:
        existing_imgs = []

    if args.rnd_knn:
        print(f'Loading random images')
        imgs_dirs = [os.path.join(args.IA_data_dir, f) for f in os.listdir(args.IA_data_dir) if '.png' in f.lower() or '.jpeg' in f.lower() or '.jpg' in f.lower()]
        knn_dirs = [imgs_dirs[i] for i in np.random.randint(0, len(imgs_dirs), size=args.num_IA_imgs)]
        NN = [(0,0,f) for f in knn_dirs]

    elif args.mse_knn:
        T = torchvision.transforms.Compose([
            K.Resize(clip_model[0].visual.input_resolution, resample=kornia.constants.Resample.BICUBIC),
            K.CenterCrop(clip_model[0].visual.input_resolution)
        ])
        img_ = T(img).cpu()
        imgs_dirs = [os.path.join(args.IA_data_dir, f) for f in os.listdir(args.IA_data_dir) if '.png' in f.lower() or '.jpeg' in f.lower() or '.jpg' in f.lower()]
        imgs_dirs = imgs_dirs[:10000]
        encoded_imgs = []
        with torch.no_grad():
            for idx, img_dir in enumerate(imgs_dirs):
                I = Image.open(img_dir).convert("RGB")
                I = torchvision.transforms.ToTensor()(I).unsqueeze(0)
                I = T(I)
                dist = torch.nn.functional.mse_loss(img_, I)
                encoded_imgs.append([img_dir, dist])
                if args.verbose:
                    print(f"\r Computing MSE distance {idx}/{len(imgs_dirs)}", end="")

        mins = [[np.nan, np.inf] for __ in range(args.num_IA_imgs)]
        if args.verbose:
            print(f"\n Looking for top {args.num_IA_imgs} images")
        for m, min in enumerate(mins):
            for idx, img in enumerate(encoded_imgs):
                if img[1] < min[1]:
                    if m == 0:
                        min[0], min[1] = idx, img[1]
                    elif mins[m - 1][1] < img[1]:
                        min[0], min[1] = idx, img[1]
            NN.append(['', '', encoded_imgs[min[0]][0]])

    elif len(existing_imgs) >= args.num_IA_imgs:
        existing_imgs = existing_imgs[:args.num_IA_imgs]
        existing_imgs.sort()
        for I in existing_imgs:
            NN.append(['', '', os.path.join(args.out_dir, f'NN_{int(args.img_size)}', I)])

        print(f'Loaded existing NN')
    else:
        img_ = K.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))(
            K.CenterCrop(clip_model[0].visual.input_resolution)(
                K.Resize(clip_model[0].visual.input_resolution, resample=kornia.constants.Resample.BICUBIC)
                (img.to(args.device))
            ))
        encoded_img = clip_model[0].encode_image(img_)
        encoded_img = F.normalize(encoded_img, dim=-1).float().cpu()
        encoded_imgs = [[os.path.join(args.IA_data_dir, f), 0, ''] for f in os.listdir(args.IA_data_dir)]
        with torch.no_grad():
            for idx, img in enumerate(encoded_imgs):
                state = torch.load(img[0], map_location="cpu")
                img_embedding, img_file = state['Encoded Image'], state['Directory']
                img_embedding = F.normalize(img_embedding.float(), dim=-1)
                dist = (img_embedding.half() - encoded_img).norm(dim=-1).div(2).arcsin().pow(2).mul(2)
                img[1] = dist
                img[2] = img_file
                if args.verbose:
                    print(f"\r Computing CLIP distance {idx}/{len(encoded_imgs)}", end="")

        mins = [[np.nan, np.inf] for __ in range(args.num_IA_imgs)]
        if args.verbose:
            print(f"\n Looking for top {args.num_IA_imgs} images")
        for m, min in enumerate(mins):
            for idx, img in enumerate(encoded_imgs):
                if img[1] < min[1]:
                    if m == 0:
                        min[0], min[1] = idx, img[1]
                    elif mins[m - 1][1] < img[1]:
                        min[0], min[1] = idx, img[1]
            NN.append(encoded_imgs[min[0]])

    NN_imgs = []
    for idx, nn in enumerate(NN):
        I = Image.open(nn[2]).convert('RGB')
        NN_imgs.append(Transforms(I).unsqueeze(0).to(args.device))

    if not (args.IA_crop or args.IA_resized_crop):
        NN_imgs = torch.cat(NN_imgs, dim=0)

    if len(existing_imgs) < args.num_IA_imgs:
        if not os.path.isdir(os.path.join(args.out_dir, f'NN_{int(args.img_size)}')):
                os.mkdir(os.path.join(args.out_dir, f'NN_{int(args.img_size)}'))
        if args.IA_crop or args.IA_resized_crop:
            for i, I in enumerate(NN_imgs):
                torchvision.utils.save_image(I, os.path.join(args.out_dir, f'NN_{int(args.img_size)}', args.image_name + f'_NearestNeighbor_{i:03d}.png'))
        else:
            for i in range(NN_imgs.shape[0]):
                torchvision.utils.save_image(NN_imgs[i:i+1], os.path.join(args.out_dir, f'NN_{int(args.img_size)}', args.image_name + f'_NearestNeighbor_{i:03d}.png'))
    return NN_imgs


def find_face_NN(img, facenet_model, args, Transforms=None):
    NN = []
    if not os.path.isdir(os.path.join(args.out_dir, f'NN_{int(args.img_size)}')):
        os.mkdir(os.path.join(args.out_dir, f'NN_{int(args.img_size)}'))
    if hasattr(args, 'image_name') and os.path.isdir(os.path.join(args.out_dir, f'NN_{int(args.img_size)}')):
        existing_imgs = [f for f in os.listdir(os.path.join(args.out_dir, f'NN_{int(args.img_size)}')) \
                         if f"{args.image_name.split('.')[0]}_NearestNeighbor_" in f]
    else:
        existing_imgs = []

    if args.rnd_knn:
        print(f'Loading random images')
        imgs_dirs = [os.path.join(args.IA_data_dir, f) for f in os.listdir(args.IA_data_dir) if '.png' in f.lower() or '.jpeg' in f.lower() or '.jpg' in f.lower()]
        knn_dirs = [imgs_dirs[i] for i in np.random.randint(0, len(imgs_dirs), size=args.num_IA_imgs)]
        NN = [(0,0,f) for f in knn_dirs]

    elif len(existing_imgs) >= args.num_IA_imgs:
        existing_imgs = existing_imgs[:args.num_IA_imgs]
        existing_imgs.sort()
        for I in existing_imgs:
            NN.append(['', '', os.path.join(args.out_dir, f'NN_{int(args.img_size)}', I)])

        print(f'Loaded existing NN')
    else:
        imgs = [os.path.join(args.IA_data_dir, f) for f in os.listdir(args.IA_data_dir)]
        dists = []
        with torch.no_grad():
            for idx, img in enumerate(imgs):
                dists.append([idx, facenet_model.verify(img, os.path.join(args.out_dir, f"{args.image_name}_SRx{args.sf}_Sty.png"), enforce_detection=False)["distance"], img])
                if args.verbose:
                    print(f"\r Computing distance {idx}/{len(imgs)}", end="")
        mins = [[np.nan, np.inf] for __ in range(args.num_IA_imgs)]
        if args.verbose:
            print(f"\n Looking for top {args.num_IA_imgs} images")
        for m, min in enumerate(mins):
            for idx, img in enumerate(dists):
                if img[1] < min[1]:
                    if m == 0:
                        min[0], min[1] = idx, img[1]
                    elif mins[m - 1][1] < img[1]:
                        min[0], min[1] = idx, img[1]
            NN.append(dists[min[0]])

    NN_imgs = []
    for idx, nn in enumerate(NN):
        I = Image.open(nn[2]).convert('RGB')
        NN_imgs.append(Transforms(I).unsqueeze(0).to(args.device))

    if not (args.IA_crop or args.IA_resized_crop):
        NN_imgs = torch.cat(NN_imgs, dim=0)

    if len(existing_imgs) < args.num_IA_imgs:
        if not os.path.isdir(os.path.join(args.out_dir, f'NN_{int(args.img_size)}')):
                os.mkdir(os.path.join(args.out_dir, f'NN_{int(args.img_size)}'))
        if args.IA_crop or args.IA_resized_crop:
            for i, I in enumerate(NN_imgs):
                torchvision.utils.save_image(I, os.path.join(args.out_dir, f'NN_{int(args.img_size)}', args.image_name + f'_NearestNeighbor_{i:03d}.png'))
        else:
            for i in range(NN_imgs.shape[0]):
                torchvision.utils.save_image(NN_imgs[i:i+1], os.path.join(args.out_dir, f'NN_{int(args.img_size)}', args.image_name + f'_NearestNeighbor_{i:03d}.png'))
    return NN_imgs


class CLIPImageRetriever:
    def __init__(self, data_dir, hash_indices_dir="hash_indices", embedding_dim=512, num_buckets=1000, num_hashes=1):
        self.data_dir = data_dir
        self.hash_indices_dir = hash_indices_dir
        self.embedding_dim = embedding_dim
        self.num_buckets = num_buckets
        self.num_hashes = num_hashes
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.dataset = torchvision.datasets.DatasetFolder(data_dir, loader=self.image_loader, extensions=("jpg", "jpeg", "png"))
        self.hash = [[] for __ in range(num_buckets)]

        # Initialize projection matrix and bias
        self.projection_matrix = torch.randn(self.embedding_dim, self.num_hashes, device=self.device, dtype=self.model.dtype)
        self.bias = torch.rand(self.num_hashes, device=self.device, dtype=self.model.dtype) * self.num_buckets
        self.hash_dataset()

    def image_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    @torch.no_grad
    def hash_embeddings(self, embeddings):
        hashed_indices = torch.matmul(embeddings, self.projection_matrix)
        hashed_indices += self.bias
        hashed_indices = hashed_indices.int() % self.num_buckets
        return hashed_indices

    @torch.no_grad
    def save_hash_indices(self):
        for idx, (img, label) in enumerate(self.dataset):
            emb = self.model.encode_image(self.preprocess(img).unsqueeze(0).to(self.model.dtype).to(self.device))
            # emb = emb / emb.norm(2, dim=-1)
            hashed_embeddings = self.hash_embeddings(emb)
            hash_index = hashed_embeddings.item()
            with open(os.path.join(self.hash_indices_dir, f"hash_{hash_index}.txt"), "a") as f:
                f.write(f"{idx} {self.dataset.samples[idx][0]}\n")

    @torch.no_grad
    def hash_dataset(self):
        print("Hashing dataset..")
        for idx, (img, label) in enumerate(self.dataset):
            emb = self.model.encode_image(self.preprocess(img).unsqueeze(0).to(self.model.dtype).to(self.device))
            # emb = emb / emb.norm(2, dim=-1)
            hashed_embeddings = self.hash_embeddings(emb)
            hash_index = hashed_embeddings.item()
            self.hash[hash_index].append(idx)
            print(f"\r [{idx + 1} / {len(self.dataset)}]", end="")

    @torch.no_grad
    def dist(self, query_emb, emb):
        # query_emb = query_emb / query_emb.norm(2, dim=-1)
        # emb = emb / emb.norm(2, dim=-1)
        return (query_emb - emb).norm(dim=-1).div(2).arcsin().pow(2).mul(2)

    @torch.no_grad
    def retrieve_nearest_neighbors(self, query_embedding, num_neighbors=5):
        # query_embedding = query_embedding / query_embedding.norm(2, dim=-1)
        hashed_query = self.hash_embeddings(query_embedding)
        nearest_neighbors = []
        hash_index = hashed_query.item()
        for bin_idx, img_idx in enumerate(self.hash[hash_index]):
            emb = self.model.encode_image(self.preprocess(self.dataset[img_idx][0]).unsqueeze(0).to(self.device).to(self.model.dtype))
            # emb = emb / emb.norm(2, dim=-1)
            nearest_neighbors.append((img_idx, self.dist(query_embedding, emb)))
            print(f"\r [{bin_idx}/{len(self.hash[hash_index])}]", end="")
        nearest_neighbors.sort(key=lambda x: x[1])
        return nearest_neighbors[:min(num_neighbors, len(nearest_neighbors))]

    @torch.no_grad
    def find_nearest_images(self, query_image, args, Transforms=torchvision.transforms.ToTensor()):
        if args.verbose:
            print("Rerieving NN images..")
        query_image = K.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))(
            K.CenterCrop(self.model.visual.input_resolution)(
                K.Resize(self.model.visual.input_resolution, resample=kornia.constants.Resample.BICUBIC)
                (query_image.to(args.device))
            ))
        query_embedding = self.model.encode_image(query_image.to(self.model.dtype))
        nearest_neighbors = self.retrieve_nearest_neighbors(query_embedding, num_neighbors=args.num_IA_imgs)
        NN_imgs = []
        for idx, nn in enumerate(nearest_neighbors):
            print(self.dataset.samples[nn[0]])
            I = Image.open(self.dataset.samples[nn[0]][0]).convert('RGB')
            NN_imgs.append(Transforms(I).unsqueeze(0).to(args.device))

        if not (args.IA_crop or args.IA_resized_crop):
            NN_imgs = torch.cat(NN_imgs, dim=0)

        if not os.path.isdir(os.path.join(args.out_dir, f'NN_hash_{int(args.img_size)}')):
                os.mkdir(os.path.join(args.out_dir, f'NN_hash_{int(args.img_size)}'))
        if args.IA_crop or args.IA_resized_crop:
            for i, I in enumerate(NN_imgs):
                torchvision.utils.save_image(I, os.path.join(args.out_dir, f'NN_hash_{int(args.img_size)}', args.image_name + f'_NearestNeighbor_{i:03d}.png'))
        else:
            for i in range(NN_imgs.shape[0]):
                torchvision.utils.save_image(NN_imgs[i:i+1], os.path.join(args.out_dir, f'NN_hash_{int(args.img_size)}', args.image_name + f'_NearestNeighbor_{i:03d}.png'))
        return NN_imgs
