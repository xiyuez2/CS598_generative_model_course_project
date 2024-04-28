#-*- coding:utf-8 -*-
#
# Original code is here: https://github.com/lucidrains/denoising-diffusion-pytorch
#

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

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group,destroy_process_group
# for fast sampling

# from fast_sampling.inference_utils import *

def scale(data,data_min = -1, data_max = 1):
    data = (data - data_min) / np.abs(data_max - data_min)
    return data
try:
    from apex import amp
    APEX_AVAILABLE = True
    print("APEX: ON")
except:
    APEX_AVAILABLE = False
    print("in trainer brast APEX: OFF")
    # print("APEX: OFF")

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

class GaussianDiffusion(nn.Module):
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
        fp16 = False
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

        if exists(betas):
            betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        else:
            betas = cosine_beta_schedule(timesteps)

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

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
            x_recon = self.predict_start_from_noise(x, t=t, noise=self.denoise_fn(torch.cat([c, x], 1), t))
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
                x_recon = self.denoise_fn(torch.cat([condition_tensors, x_noisy], 1), t)
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
                loss = loss1 * 100 + loss2 * 150
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


# trainer class

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        val_dataset,
        ema_decay = 0.995,
        image_size = 128,
        depth_size = 128,
        train_batch_size = 1,
        train_lr = 2e-6,
        train_num_steps = 100000,
        gradient_accumulate_every = 1,
        fp16 = False,
        step_start_ema = 2000,
        update_ema_every = 10,
        save_and_sample_every = 1000,
        results_folders = [],
        res_folder = "results",
        with_condition = False,
        with_pairwised = False,
        gpu_id = 0,
        world_size = 1,
        writer = None,
        residual_training = False,
        skip_input_viz = 0,
        self_consistency_config = {},
        num_workers=4,
        val_num_steps = 50,):
        super().__init__()
        self.skip_input_viz = skip_input_viz
        self.val_num_steps = val_num_steps
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every
        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every
        self.batch_size = train_batch_size
        self.image_size = diffusion_model.image_size
        self.residual_training = residual_training
        self.depth_size = depth_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps
        self.self_consistency_config = self_consistency_config

        self.ds = dataset
        self.val_ds = val_dataset
        self.dl = cycle(data.DataLoader(
            self.ds, 
            batch_size = train_batch_size, 
            shuffle=False, 
            num_workers=num_workers, 
            pin_memory=True, 
            sampler=DistributedSampler(self.ds)))
        self.val_dl = data.DataLoader(
            self.val_ds, 
            batch_size = train_batch_size, 
            shuffle=False, 
            num_workers=num_workers, 
            pin_memory=True, 
            sampler=DistributedSampler(self.val_ds))
        
        
        self.opt = Adam(diffusion_model.parameters(), lr=train_lr)
        self.train_lr = train_lr
        self.train_batch_size = train_batch_size
        self.with_condition = with_condition

        self.step = 0

        # assert not fp16 or fp16 and APEX_AVAILABLE, 'Apex must be installed in order for mixed precision training to be turned on'

        self.fp16 = fp16
        # if fp16:
        #     (self.model, self.ema_model), self.opt = amp.initialize([self.model, self.ema_model], self.opt, opt_level='O1')
        
        results_folders[0] = res_folder + "/" + results_folders[0]
        self.results_folders = results_folders
        self.results_folder = Path(results_folders[0])
        # for i in range(1,len(results_folders)):
        #     os.makedirs(f"{results_folders[0]}/{results_folders[i]}", exist_ok=True)
        os.makedirs(f"{results_folders[0]}/debug_mask", exist_ok=True)
        os.makedirs(f"{results_folders[0]}/model", exist_ok=True)
                
        self.ref = nib.load("reference_brats.nii.gz")
        
        self.world_size = world_size
        self.gpu_id = gpu_id
        if self.gpu_id == self.world_size - 1:
            self.log_dir = self.create_log_dir()
            self.writer = SummaryWriter(log_dir=self.log_dir)#"./logs")
        else:
            self.writer = None

        self.model = DDP(self.model,device_ids=[gpu_id])
        self.ema_model = DDP(self.ema_model,device_ids=[gpu_id])
        self.reset_parameters()

    def create_log_dir(self):
        now = datetime.datetime.now().strftime("%y-%m-%dT%H%M%S")
        log_dir = os.path.join("logs", now)
        os.makedirs(log_dir, exist_ok=True)
        return log_dir

    def reset_parameters(self):
        self.ema_model.module.load_state_dict(self.model.module.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self, milestone):
        data = {
            'step': self.step,
            'model': self.model.module.state_dict(),
            'ema': self.ema_model.module.state_dict()
        }
        torch.save(data, str(self.results_folder/ f'model/model-{milestone}.pt'))

    def load(self, resume_weight):

        data = torch.load(resume_weight, map_location='cuda')
        # self.step = data['step']
        self.model.module.load_state_dict(data['model'])
        self.ema_model.module.load_state_dict(data['ema'])

    def viz(self, cur_images, path, video = True, plot = True, file = True):
        # cur_images of shape 1 d w h
        cur_images = torch.squeeze(cur_images)
        cur_images = cur_images.transpose(2, 0)
        sampleImage = cur_images.cpu().numpy()
        nifti_img = nib.Nifti1Image(sampleImage, affine=self.ref.affine)
        if file:
            nib.save(nifti_img, path + '.nii.gz')
        sampleImage = np.transpose(sampleImage, (2, 1, 0))
        if plot:
            make_plot(sampleImage, path)
        if video:
            make_video(sampleImage, path)

        return 1

    def train(self):
        start_time = time.time()
        if self.fp16:
            scaler = torch.cuda.amp.GradScaler()
        else:
            scaler = None

        accumulated_loss = []
        while self.step < self.train_num_steps:
            for i in range(self.gradient_accumulate_every):
                if self.with_condition:
                    data = next(self.dl)
                    input_tensors = data['input'].cuda()
                    target_tensors = data['target'].cuda()
                    mask = data['mask'].cuda()
                    # if mask is all 0, skip this iteration
                    if mask.sum() == 0:
                        print("mask is all 0, skip this iteration")
                        continue
                    # if self.fp16:
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled = self.fp16):
                        loss = self.model(target_tensors, condition_tensors=input_tensors, mask = mask)
                    # else:
                    #     loss = self.model(target_tensors, condition_tensors=input_tensors, mask = mask)
                else:
                    data = next(self.dl).cuda()
                    with torch.autocast(device_type="cuda",dtype=torch.float16, enabled = self.fp16):
                        loss = self.model(data)
                
                loss = loss.sum()/self.batch_size
                accumulated_loss.append(loss.item())
                print(f'[GPU{self.gpu_id}] train loss at step {self.step}: {loss.item()}')
                
                loss = loss / self.gradient_accumulate_every

                if self.fp16:
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled = self.fp16):
                        scaler.scale(loss).backward()
                else:
                    loss.backward()

            if self.fp16:
                # with torch.autocast(device_type="cuda", dtype=torch.float16, enabled = self.fp16):
                scaler.step(self.opt)
                scaler.update()
            else:
                self.opt.step()

            self.opt.zero_grad()
            average_loss = 0
            print("accumulated_loss", len(accumulated_loss), np.mean(accumulated_loss))

            if len(accumulated_loss) > 1000:
                average_loss = np.mean(accumulated_loss[-1000:])
            else:
                average_loss = np.mean(accumulated_loss)

            end_time = time.time()

            # Record here
            if self.gpu_id == self.world_size - 1:
                self.writer.add_scalar("training_loss_for_last_1k_samples", average_loss, self.step)
            
            if self.step % self.update_ema_every == 0:
                self.step_ema()

            # if self.step % 100 == 0: # save viz for training
            #     for i in range(input_tensors.shape[1]):
            #         self.viz(input_tensors[:, i, ...], str(f'debug/train-input-{self.step}_GPU{self.gpu_id}'),video=False,plot=True,file=False)
            #     for i in range(target_tensors.shape[1]):
            #         self.viz(target_tensors[:, i, ...], str(f'debug/train-GT-{self.step}_GPU{self.gpu_id}'),video=False,plot=True,file=False)

            # step == 0 for debug
            if self.step % self.save_and_sample_every == 3: # and self.step != 0:
                print("training loss epoch(past 1k samples):", average_loss, len(accumulated_loss))
                self.ema_model.eval()
                self.model.eval()
                with torch.no_grad():
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled = self.fp16):
                        milestone = self.step // self.save_and_sample_every
                        # save model
                        if self.gpu_id == self.world_size - 1:
                            self.save(milestone)
                            print("saved at step ", self.step)
                                                
                        # validate
                        print("validating")
                        val_loss_total = []
                        for j in tqdm(range(self.val_num_steps)):
                            for val_data in tqdm(self.val_dl):
                                val_input_tensors = val_data['input'].cuda()
                                val_target_tensors = val_data['target'].cuda()
                                val_mask = val_data['mask'].cuda()
                                # if self.fp16:
                                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled = self.fp16):
                                    val_loss = self.model(val_target_tensors, condition_tensors=val_input_tensors, mask = val_mask)

                                val_loss = val_loss.sum()/self.batch_size
                                val_loss_total.append(val_loss.item())

                        print(f'[GPU{self.gpu_id}] val loss epoch at milestone {milestone}: {np.mean(val_loss_total)}, steps: {len(val_loss_total)}')

                        # sample data in validation/train set
                        if self.gpu_id < 2:
                            input_data, target_data, mask = self.ds.sample_conditions(batch_size=1)
                        else:
                            input_data, target_data, mask = self.val_ds.sample_conditions(batch_size=1)
                        # inference
                        if self.with_condition:
                            print("start inference with condition on GPU ", self.gpu_id)
                            all_images = self.ema_model.module.sample(batch_size=1, condition_tensors=input_data, self_consistency_config = self.self_consistency_config, debug = "debug")
                        else:
                            all_images = self.ema_model.module.sample(batch_size=1)

                        # save and viz inference result
                        
                        for i in range(all_images.shape[1]):
                            self.viz(all_images[:, i, ...], str(self.results_folder / f'sample-{milestone}-GPU{self.gpu_id}-inferece-modal-{i}'))
                            if self.residual_training:
                                self.viz(2 * all_images[:, i, ...] + input_data[:, i, ...], str(self.results_folder / f'sample-{milestone}-GPU{self.gpu_id}-inferece-modal-final-{i}'))
                        
                        # save and viz input
                        for i in range(input_data.shape[1] - self.skip_input_viz):
                            self.viz(input_data[:, i, ...], str(self.results_folder / f'sample-{milestone}-GPU{self.gpu_id}-input-modal-{i}'))                        
                        
                        # save and viz GT
                        for i in range(target_data.shape[1]):
                            self.viz(target_data[:, i, ...], str(self.results_folder / f'sample-{milestone}-GPU{self.gpu_id}-GT-modal-{i}'))
                            if self.residual_training:
                                self.viz(2 * target_data[:, i, ...] + input_data[:, i, ...], str(self.results_folder / f'sample-{milestone}-GPU{self.gpu_id}-GT-modal-final-{i}'))
                        for i in range(mask.shape[1]):
                            self.viz(mask[:, i, ...], str(self.results_folder / f'sample-{milestone}-GPU{self.gpu_id}-mask-modal-{i}'))

                        self.viz(mask[:,0, ...], str(self.results_folder / f'debug_mask/sample-{milestone}-GPU{self.gpu_id}'))
                        
                        
                self.ema_model.train()
                self.model.train()
            self.step += 1

        print('training completed')
        end_time = time.time()
        execution_time = (end_time - start_time)/3600
        if self.gpu_id == self.world_size - 1:
            self.writer.add_hparams(
                {
                    "lr": self.train_lr,
                    "batchsize": self.train_batch_size,
                    "image_size": self.image_size,
                    "depth_size": self.depth_size,
                    "execution_time (hour)": execution_time
                },
                {"last_loss":average_loss}
            )
            self.writer.close()
        
    def validate(self):
        start_time = time.time()
        self.results_folder_old = self.results_folder
        os.makedirs(str(self.results_folder)+ "/val_res", exist_ok=True)
        
        self.results_folder = str(self.results_folder)+ "/val_res"

        self.ema_model.eval()
        self.model.eval()
        with torch.no_grad():
            total_metrics = []
            for i, data in enumerate(tqdm(self.val_dl)):
                input_tensors = data['input'].cuda()
                target_tensors = data['target'].cuda()
                mask = data['mask'].cuda()
                batch_size = input_tensors.shape[0]
                # for debug
                # metrics_k = self.val_ds.evaluate(torch.randn_like(target_tensors)[0,0,...].cpu() , target_tensors[0, 0, ...].cpu())
                # print(metrics_k)
                # for debug
                if self.with_condition:
                    print("start inference with condition on GPU ", self.gpu_id)
                    all_images = self.ema_model.module.sample(batch_size=batch_size, condition_tensors=input_tensors, debug = "debug")
                else:
                    all_images = self.ema_model.module.sample(batch_size=batch_size)
                # print("debug", all_images.shape)
                # only viz a few inference result                       
                if i < 3:
                    for j in range(all_images.shape[1]):
                        if self.residual_training:
                            self.viz(2 * all_images[:, j, ...] + input_tensors[:, j, ...], str(self.results_folder + f'/sample-{i}-GPU{self.gpu_id}-inferece-modal-final-{j}'))
                        else:
                            self.viz(all_images[:, j, ...], str(self.results_folder + f'/validation-{i}-GPU{self.gpu_id}-inferece-modal-{j}'))
                    
                    # save and viz input
                    for j in range(input_tensors.shape[1]):
                        self.viz(input_tensors[:, j, ...], str(self.results_folder + f'/sample-{i}-GPU{self.gpu_id}-input-modal-{j}'))

                    # save and viz GT
                    for j in range(target_tensors.shape[1]):
                        if self.residual_training:
                            self.viz(2 * target_tensors[:, j, ...] + input_tensors[:, j, ...], str(self.results_folder + f'/sample-{i}-GPU{self.gpu_id}-GT-modal-final-{j}'))
                        else:
                            self.viz(target_tensors[:, j, ...], str(self.results_folder + f'/sample-{i}-GPU{self.gpu_id}-GT-modal-{j}'))
                    
                # cal metrics
                b, c, d, w, h = all_images.shape
                
                for j in range(len(all_images)):
                    current_metrics = [] # modality x 3
                    for k in range(all_images.shape[1]):
                        if not self.residual_training:
                            cur_res = all_images[j, k, ...] / 2. + 1
                            cur_target = target_tensors[j, k, ...] / 2. + 1
                        else:
                            cur_res = (2 * all_images[j, k, ...] + input_tensors[j, k, ...]) / 2. + 0.5
                            cur_target = (2 * target_tensors[j, k, ...] + input_tensors[j, k, ...]) / 2. + 0.5
                        
                        # cur_res = scale(cur_res.cpu().numpy())
                        # cur_target = scale(cur_target.cpu().numpy())
                        print("debug data range")
                        print(np.min(cur_res), np.max(cur_res), np.mean(cur_res))
                        print(np.min(cur_target), np.max(cur_target), np.mean(cur_target))
                        # save res to disk
                        cur_path = str(self.results_folder / f'sample-{i*b + j}-GPU{self.gpu_id}-GT-modal-{k}')
                        np.save(cur_path, cur_target)
                        cur_path = str(self.results_folder / f'sample-{i*b + j}-GPU{self.gpu_id}-inferece-modal-{k}')
                        np.save(cur_path, cur_res)

                        # metrics_k = self.val_ds.evaluate(cur_res, cur_target) # len: 3
                        # L2 loss 
                        metrics_k = np.mean((cur_res - cur_target) ** 2)
                        current_metrics.append(metrics_k)
                    
                    total_metrics.append(current_metrics) # num_samples x modality x 3
                    print(f"GPU{self.gpu_id}")
                    print(total_metrics)
                    # currently metrics is just L2

        self.results_folder = self.results_folder_old

    def fast_sample(self,sampling_step = 50):
        
        
        # for debug only!!
        # sampling_step = 5
        start_time = time.time()
        self.results_folder_old = self.results_folder
        os.makedirs(str(self.results_folder)+ "/fast_sample", exist_ok=True)
        
        self.results_folder = str(self.results_folder)+ "/fast_sample"

        self.ema_model.eval()
        self.model.eval()
        old_betas = self.model.module.betas.cpu().detach()
        
        with torch.no_grad():
            total_metrics = []
            for i, data in enumerate(tqdm(self.val_dl)):
                input_tensors = data['input'].cuda()
                target_tensors = data['target'].cuda()
                mask = data['mask'].cuda()
                batch_size = input_tensors.shape[0]

                diffusion = make_diffusion(old_betas, 1000, sampling_step)
                wrap = Wrap(self.model.module.denoise_fn, input_tensors).cuda()
                # print(target_tensors.shape)
                # print(input_tensors.shape)

                if self.with_condition:
                    print("start inference with condition on GPU ", self.gpu_id)
                    all_images = diffusion.p_sample_loop(wrap, target_tensors.shape[1:], progress=True, self_consistency_config = self.self_consistency_config)
                    #self.ema_model.module.sample(batch_size=batch_size, condition_tensors=input_tensors, debug = "debug")
                else:
                    all_images = diffusion.p_sample_loop(wrap, target_tensors.shape[1:], progress=True, self_consistency_config = self.self_consistency_config)
                    #self.ema_model.module.sample(batch_size=batch_size)
                print(all_images.shape)
                all_images = all_images.unsqueeze(0)
                # print(all_images.shape, all_images.min(), all_images.max(), all_images.mean())
                # print("debugging!!",all_images.shape)
                if i < 3:
                    for j in range(all_images.shape[1]):
                        if self.residual_training:
                            self.viz(2 * all_images[:, j, ...] + input_tensors[:, j, ...], str(self.results_folder + f'/sample-{i}-GPU{self.gpu_id}-inferece-modal-final-{j}'))
                        else:
                            self.viz(all_images[:, j, ...], str(self.results_folder + f'/validation-{i}-GPU{self.gpu_id}-inferece-modal-{j}'))
                    
                    # save and viz input
                    for j in range(input_tensors.shape[1]):
                        self.viz(input_tensors[:, j, ...], str(self.results_folder + f'/sample-{i}-GPU{self.gpu_id}-input-modal-{j}'))

                    # save and viz GT
                    for j in range(target_tensors.shape[1]):
                        if self.residual_training:
                            self.viz(2 * target_tensors[:, j, ...] + input_tensors[:, j, ...], str(self.results_folder + f'/sample-{i}-GPU{self.gpu_id}-GT-modal-final-{j}'))
                        else:
                            self.viz(target_tensors[:, j, ...], str(self.results_folder + f'/sample-{i}-GPU{self.gpu_id}-GT-modal-{j}'))
                    
                # cal metrics
                b, c, d, w, h = all_images.shape
                
                for j in range(len(all_images)):
                    current_metrics = [] # modality x 3
                    for k in range(all_images.shape[1]):
                        if not self.residual_training:
                            cur_res = all_images[j, k, ...] / 2. + 0.5
                            cur_target = target_tensors[j, k, ...] / 2. + 0.5
                        else:
                            # debugging = torch.nn.functional.avg_pool3d(all_images[j, k, ...].unsqueeze(0),4, 4)
                            
                            cur_res = (2 * all_images[j, k, ...] + input_tensors[j, k, ...]) / 2. + 0.5 
                            cur_target = (2 * target_tensors[j, k, ...] + input_tensors[j, k, ...]) / 2. + 0.5
                        
                        
                        
                        # check self consistency:
                        # cur_res, cur_target, input_tensors
                        
                        downsampled = torch.nn.functional.avg_pool3d(cur_res.unsqueeze(0),4, 4) # 1 d, w, h
                        downsampled = torch.repeat_interleave(downsampled, 4, dim=1)
                        downsampled = torch.repeat_interleave(downsampled, 4, dim=2)
                        downsampled = torch.repeat_interleave(downsampled, 4, dim=3)
                        
                        print("debugging")
                        print(torch.min(downsampled), torch.max(downsampled), torch.mean(downsampled))
                        print(torch.min(cur_res), torch.max(cur_res), torch.mean(cur_res))
                        print(torch.min(cur_target), torch.max(cur_target), torch.mean(cur_target))
                        print(torch.min(input_tensors[j,k,...]), torch.max(input_tensors[j,k,...]), torch.mean(input_tensors[j,k,...]))

                        print("check self consistency, this should be 0: ",torch.mean((downsampled - (input_tensors[j,k,...].unsqueeze(0) / 2 + 0.5) ) ** 2))
                        
                        downsampled = torch.nn.functional.avg_pool3d(cur_target.unsqueeze(0),4, 4) # 1 d, w, h
                        downsampled = torch.repeat_interleave(downsampled, 4, dim=1)
                        downsampled = torch.repeat_interleave(downsampled, 4, dim=2)
                        downsampled = torch.repeat_interleave(downsampled, 4, dim=3)
                        print("check self consistency, this have to be 0: ",torch.mean((downsampled - (input_tensors[j,k,...].unsqueeze(0) / 2 + 0.5) ) ** 2))
                        
                        # cur_res = scale(cur_res.cpu().numpy())
                        # cur_target = scale(cur_target.cpu().numpy())
                        cur_res = cur_res.cpu().numpy()
                        cur_target = cur_target.cpu().numpy()
                        print("debug data range")
                        # print(cur_res)
                        print(cur_res.shape)
                        print("L2", np.mean((cur_res - cur_target) ** 2))
                        print(np.min(cur_res), np.max(cur_res), np.mean(cur_res))
                        print(np.min(cur_target), np.max(cur_target), np.mean(cur_target))
                        # save res to disk
                        cur_path = str(self.results_folder + f'/sample-{i*b + j}-GPU{self.gpu_id}-GT-modal-{k}')
                        np.save(cur_path, cur_target)
                        cur_path = str(self.results_folder + f'/sample-{i*b + j}-GPU{self.gpu_id}-inferece-modal-{k}')
                        np.save(cur_path, cur_res)

                        # metrics_k = self.val_ds.evaluate(cur_res, cur_target) # len: 3
                        # L2 loss 
                        metrics_k = np.mean((cur_res - cur_target) ** 2)
                        current_metrics.append(metrics_k)
                    
                    total_metrics.append(current_metrics) # num_samples x modality x 3
                    print(f"GPU{self.gpu_id}")
                    print(total_metrics)

        self.results_folder = self.results_folder_old
