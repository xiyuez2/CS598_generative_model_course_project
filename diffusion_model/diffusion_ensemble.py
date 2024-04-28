#-*- coding:utf-8 -*-
#
# Original code is here: https://github.com/lucidrains/denoising-diffusion-pytorch
#
# import from 2D model
from model_2D.core.logger import VisualWriter, InfoLogger
from model_2D.models import create_model, define_network
import model_2D.core.praser as Praser
# main import
import math
import copy
import torch
from torch import nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
from torch.utils import data
from pathlib import Path
from torch.optim import Adam
from torchvision import transforms, utils
from PIL import Image
import nibabel as nib
import numpy as np
from tqdm import tqdm
from einops import rearrange
from torch.utils.tensorboard import SummaryWriter
import datetime
import time
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from utils.util import make_video, make_plot


# for multi GPU training
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group,destroy_process_group

try:
    from apex import amp
    APEX_AVAILABLE = True
    print("APEX: ON")
except:
    APEX_AVAILABLE = False
    print("APEX: OFF")

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def cycle(dl):
    while True:
        for data in dl:
            yield data

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

# small helper modules

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()
  
def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, a_min = 0, a_max = 0.999)

def make_beta_schedule(schedule, n_timestep, linear_start=1e-6, linear_end=1e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'cos_old':
        betas = cosine_beta_schedule(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


class GaussianDiffusion_ensemble(nn.Module):
    def __init__(
        self,
        denoise_fn,
        *,
        image_size,
        depth_size,
        channels = 1,
        timesteps = 250,
        loss_type = 'l1',
        betas = None,
        with_condition = False,
        with_pairwised = False,
        apply_bce = False,
        lambda_bce = 0.0,
        fp16 = False,
        model_2D_2_opt = {},
        model_2D_3_opt = {},
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.depth_size = depth_size
        self.denoise_fn = denoise_fn
        self.with_condition = with_condition
        self.with_pairwised = with_pairwised
        self.apply_bce = apply_bce
        self.lambda_bce = lambda_bce
        self.fp16 = fp16
        self.self_consistency_config = {}
        self.model_2D_2_opt = model_2D_2_opt
        self.model_2D_3_opt = model_2D_3_opt
        
        if exists(betas):
            print("using existing betas, not supported since the beta needs to be consistent with 2D model", betas)
            raise
            betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        else:
            print("please make sure this is consistent with 2D model in config")
            betas = make_beta_schedule("linear", n_timestep=1000, linear_start=1e-4, linear_end=0.09)

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0) # gamma
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1]) # gamma_prev

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef3', to_torch(
            1. - (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def q_mean_variance(self, x_start, t, c=None):
        x_hat = 0
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + x_hat
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise, c=None):
        x_hat = 0.
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise -
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_hat
        )

    def q_posterior(self, x_start, x_t, t, c=None):
        x_hat = 0.
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t +
            extract(self.posterior_mean_coef3, t, x_t.shape) * x_hat
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, c = None):
        if self.with_condition:
            x_recon = self.predict_start_from_noise(x, t=t, noise=self.denoise_fn(torch.cat([x, c], 1), t))
        else:
            x_recon = self.predict_start_from_noise(x, t=t, noise=self.denoise_fn(x, t))

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t, c=c)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, condition_tensors=None, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, c=condition_tensors, clip_denoised=clip_denoised)
        if len(self.self_consistency_config) > 0:
            # enforce cycle consistency here
            for target_ind in self.self_consistency_config.keys():
                target_tensor = model_mean[:,target_ind]
                source_ind, input_name = self.self_consistency_config[target_ind]
                source_tensor = condition_tensors[:,source_ind]
                modality, downsample = input_name.split("_")
                downsample, downsampling_factor, *_ = downsample.split("x") + [None]
                if downsample == "pool":
                    target_tensor_pool = torch.nn.functional.avg_pool3d(target_tensor.unsqueeze(1), int(downsampling_factor), int(downsampling_factor))
                    target_tensor_pool = torch.repeat_interleave(target_tensor_pool, int(downsampling_factor), dim=2)
                    target_tensor_pool = torch.repeat_interleave(target_tensor_pool, int(downsampling_factor), dim=3)
                    target_tensor_pool = torch.repeat_interleave(target_tensor_pool, int(downsampling_factor), dim=4)
                    # print("consistency error:", torch.mean(target_tensor_pool),target_tensor_pool.shape)
                    model_mean[:,source_ind] -= target_tensor_pool[:,0]

                else:
                    print("cycle consistentcy not implemented for:")
                    print(self.self_consistency_config)
                    pass
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, condition_tensors = None):
        device = self.betas.device

        b = shape[0]
        img = torch.randn(shape, device=device)

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            if self.with_condition:
                t = torch.full((b,), i, device=device, dtype=torch.long)
                img = self.p_sample(img, t, condition_tensors=condition_tensors)
                # print("debugging:", img.shape)
                if len(self.debug) > 0 and i % 300 == 0:
                    for j in range(img.shape[1]):
                        cur_images = img[:, j, ...]
                        cur_images = torch.squeeze(cur_images)
                        cur_images = cur_images.transpose(2, 0)
                        sampleImage = cur_images.cpu().numpy()
                        sampleImage = np.transpose(sampleImage, (2, 1, 0))
                        make_plot(sampleImage, self.debug + f"/res-{i}-{j}")
                # print(i)
                # print("img", img.min(), img.max(), img.mean())
                # print("condition_tensors", condition_tensors.min(), condition_tensors.max(), condition_tensors.mean())
            else:
                img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))

        return img

    @torch.no_grad()
    def sample(self, batch_size = 2, condition_tensors = None,self_consistency_config = {}, debug = None):
        image_size = self.image_size
        depth_size = self.depth_size
        channels = self.channels
        self.debug = debug
        if len(self_consistency_config) > 0:
            self.self_consistency_config = self_consistency_config
        res = self.p_sample_loop((batch_size, channels, depth_size, image_size, image_size), condition_tensors = condition_tensors)
        self.debug = None
        return res

    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))
        return img

    def q_sample(self, x_start, t, noise=None, c=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_hat = 0.
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise + x_hat
        )

    def p_losses(self, x_start, t, condition_tensors = None, noise = None, mask = None):
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled = self.fp16):
            # print("fp16:", self.fp16)
            if mask is None:
                mask = torch.ones_like(x_start).cuda()
            b, c, h, w, d = x_start.shape
            noise = default(noise, lambda: torch.randn_like(x_start))

            if self.with_condition:
                x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
                x_recon = self.denoise_fn(torch.cat([x_noisy, condition_tensors], dim = 1), t)
            else:
                x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
                x_recon = self.denoise_fn(x_noisy, t)
            
            x_recon = x_recon * mask + noise * (1 - mask)
            # print("during training:")
            # print("x_noisy:", x_noisy.min(), x_noisy.max(), x_noisy.mean())
            # print("x_recon:", x_recon.min(), x_recon.max(), x_recon.mean())
            # print("conditional_tensors:", condition_tensors.min(), condition_tensors.max(),condition_tensors.mean())

            if self.loss_type == 'l1':
                loss = (noise - x_recon).abs().mean()
            elif self.loss_type == 'l2':
                loss = F.mse_loss(x_recon, noise)
            elif self.loss_type == 'hybrid':
                loss1 = (noise - x_recon).abs().mean()
                loss2 = F.mse_loss(x_recon, noise)
                # loss2 = F.binary_cross_entropy_with_logits(x_recon, noise)
                loss = loss1 * 5 + loss2 * 150
            else:
                raise NotImplementedError()

            return loss

    def forward(self, x, condition_tensors=None, mask = None, *args, **kwargs):
        # auto cast with torch.autocast
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled = self.fp16):
            if mask is None:
                mask = torch.ones_like(x).cuda()
            b, c, d, h, w, device, img_size, = *x.shape, x.device, self.image_size
            # assert h == img_size and w == img_size, f'height and width of image must be {img_size}, but now h={h} and w={w}'
            t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
            return self.p_losses(x, t, condition_tensors=condition_tensors, mask = mask, *args, **kwargs) 
