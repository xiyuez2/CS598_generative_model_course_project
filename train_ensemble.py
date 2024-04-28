#-*- coding:utf-8 -*-
# +
# for 2d 
import json
from collections import OrderedDict
from pathlib import Path
from model_2D.core.praser import mkdirs, dict_to_nonedict, write_json
from model_2D.core.logger import VisualWriter, InfoLogger
from model_2D.models import create_model as create_model_2D
from model_2D.models import define_network
import model_2D.core.praser as Praser

# main imports
from torchvision.transforms import RandomCrop, Compose, ToPILImage, Resize, ToTensor, Lambda
from diffusion_model.trainer_brats import GaussianDiffusion, Trainer
from diffusion_model.diffusion_ensemble import GaussianDiffusion_ensemble
from diffusion_model.unet_brats import create_model, model_ensemble
from dataset_brats import NiftiImageGenerator, NiftiPairImageGenerator
import argparse
import torch
import datetime
from torch.utils.tensorboard import SummaryWriter
import yaml

import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group,destroy_process_group

def ddp_setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def parse_2D_config(config, args):
    json_str = ''
    print("using 2D config file:", config)
    with open(config, 'r') as f:
        for line in f:
            line = line.split('//')[0] + '\n'
            json_str += line
    opt = json.loads(json_str, object_pairs_hook=OrderedDict)
    opt['phase'] = "test"
    opt['datasets'][opt['phase']]['dataloader']['args']['batch_size'] = args.batchsize
    opt['gpu_ids'] = list(range(args.gpus))
    if len(opt['gpu_ids']) > 1:
        opt['distributed'] = True
    else:
        opt['distributed'] = False
    
    opt['name'] = args.input_modality.replace(" ", "&") + "-to-" + args.target_modality.replace(" ", "=")
    experiments_root = args.res_folder + "/"+ opt['name']
    opt['model']['which_model']['args']['set_device'] = False

    opt['path']['experiments_root'] = experiments_root
    opt['path']['experiments_root'] = experiments_root
    for key, path in opt['path'].items():
        if 'resume' not in key and 'base' not in key and 'root' not in key:
            opt['path'][key] = os.path.join(experiments_root, path)
            mkdirs(opt['path'][key])
    mkdirs(experiments_root + "/config")
    write_json(opt, '{}/{}'.format(experiments_root,config))
    
    return dict_to_nonedict(opt)


def main(rank: int, world_size: int, args):
    ddp_setup(rank, world_size)
    try: 
        input_size = args.input_size
        depth_size = args.depth_size
        num_channels = args.num_channels
        num_res_blocks = args.num_res_blocks
        save_and_sample_every = args.save_and_sample_every
        with_condition = args.with_condition
        resume_weight = args.resume_weight
        if args.random_crop_xy == 0:
            random_crop_xy = input_size
        else:
            random_crop_xy = args.random_crop_xy
            
        if args.random_crop_z == 0:
            random_crop_z = depth_size
        else:
            random_crop_z = args.random_crop_z
            
        random_crop_size = (args.random_crop_xy, args.random_crop_xy, args.random_crop_z)
        if args.random_crop_xy == 0 and args.random_crop_z == 0:
            random_crop_size = ()
            use_ckpt = True
        else:
            use_ckpt = False

        input_modality = args.input_modality.split()
        target_modality = args.target_modality.split()
        print("starting image to image generation training")
        print("input modality:", input_modality)
        print("target modality:", target_modality)
        # define the folder names for saving results
        # res_folder_name is the name of the folder that contains all the results
        # res_folder_names is a list of names of the res_folder_name and its sub folders
        # each sub folder contains the results of one modality
        # seg mask is a special case, it will be saved in 4 different folders, one for each channel
        res_folder_name = args.input_modality.replace(" ", "&") + "-to-" + args.target_modality.replace(" ", "=")
        res_folder_names = [res_folder_name]
        # self consistency config
        self_consistency_config = {}
        if not args.no_self_consistency:
            for i, name in enumerate(target_modality):
                if not name in ["flair", "t1", "t1ce", "t2"]:
                    continue
                for j, input_name in enumerate(input_modality):
                    if name + "_" in input_name:
                        self_consistency_config[i] = (j, input_name)

        print("self consistency enforce:", self_consistency_config)
        print("note that self consistency must be used with residual = True")
        cond_channels = len(input_modality)
        if "seg" in input_modality:
            cond_channels -= 1
        if args.glob_pos_emb:
            cond_channels += 3
        if args.none_zero_mask:
            cond_channels += 1

        target_channels = len(target_modality)

        in_channels = cond_channels + target_channels if with_condition or with_pairwised else 1
        # inchannels added for 2D model input
        if not args.baseline == "3D_only":
            in_channels += (target_channels*2)

        out_channels = target_channels

        if with_condition:
            dataset = NiftiPairImageGenerator(
                dataset_folder=args.dataset_folder,
                input_modality = input_modality,
                target_modality = target_modality,
                input_size=input_size,
                depth_size=depth_size,
                transform= None, #input_transform if with_condition else transform,
                target_transform=None, #transform,
                full_channel_mask=True,
                random_crop_size=random_crop_size,
                global_pos_emb=args.glob_pos_emb,
                none_zero_mask=args.none_zero_mask,
                residual_training=args.residual_training
            )
            val_dataset = NiftiPairImageGenerator(
                dataset_folder=args.dataset_folder,
                input_modality=input_modality,
                target_modality=target_modality,
                input_size=input_size,
                depth_size=depth_size,
                transform=None, #input_transform if with_condition else transform,
                target_transform=None, #transform,
                full_channel_mask=True,
                train=False,
                random_crop_size=random_crop_size,
                global_pos_emb=args.glob_pos_emb,
                none_zero_mask=args.none_zero_mask,
                residual_training=args.residual_training
            )
        else:
            print("Please modify your code to unconditional generation")
        skip_input_viz = 0
        if args.glob_pos_emb:
            skip_input_viz += 3
        if args.none_zero_mask:
            skip_input_viz += 1


        input_size, depth_size = random_crop_xy, random_crop_z
        if len(resume_weight) > 0:
            print("changing model architecture to match the resume weight")
            # TODO automate this
            input_size_model = 64
            # use_ckpt = False
        else:
            input_size_model = input_size
        
        # emsenble unet and base 2D models as denoise_fn
        

        model_2D_2_opt = parse_2D_config(args.model_2D_2_config, args)
        model_2D_3_opt = parse_2D_config(args.model_2D_3_config, args)
        phase_logger = InfoLogger(model_2D_2_opt)
        phase_writer = VisualWriter(model_2D_2_opt, phase_logger)
        phase_logger.info('Create the log file in directory {}.\n'.format(model_2D_2_opt['path']['experiments_root']))
        # def add_2D_name(item_opt):
        #     network_name = item_opt['name']
        #     print("network_name:", network_name)
        #     network_name[0] += "model_2D"
        #     item_opt[name] = network_name
        #     return item_opt
        networks_2D_2 = [define_network(phase_logger, model_2D_2_opt, item_opt) for item_opt in model_2D_2_opt['model']['which_networks']]
        networks_2D_3 = [define_network(phase_logger, model_2D_3_opt, item_opt) for item_opt in model_2D_3_opt['model']['which_networks']]
        model2D_2 = create_model_2D(
            opt = model_2D_2_opt,
            networks = networks_2D_2,
            phase_loader = [],
            val_loader = [],
            losses = [lambda x: 0],
            metrics = [lambda x: 0],
            logger = None,
            writer = None
        )
        model2D_3 = create_model_2D(
            opt = model_2D_3_opt,
            networks = networks_2D_3,
            phase_loader = [],
            val_loader = [],
            losses = [lambda x: 0],
            metrics = [lambda x: 0],
            logger = None,
            writer = None
        )

        model_3D = create_model(input_size_model, 
            num_channels, 
            num_res_blocks, 
            in_channels=in_channels, 
            out_channels=out_channels, 
            use_checkpoint = use_ckpt, 
            attention_resolutions = "",
            use_fp16=args.fp16
        ) #.cuda()
        if args.validation:
            batch_size_2D_inference = 8
        else:
            batch_size_2D_inference = 8
        model = model_ensemble(model_3D, model2D_2, model2D_3, batch_size_2D_inference = batch_size_2D_inference, time_step=args.timesteps, baseline=args.baseline).cuda()

        diffusion = GaussianDiffusion_ensemble(
            model,
            image_size = input_size,
            depth_size = depth_size,
            timesteps = args.timesteps,   # number of steps
            loss_type = 'hybrid',    # L1 and L2
            with_condition=with_condition,
            channels=out_channels,
            fp16=args.fp16,
            model_2D_2_opt = model_2D_2_opt,
            model_2D_3_opt = model_2D_3_opt
        ).cuda()

        if len(resume_weight) > 0:
            weight = torch.load(resume_weight, map_location='cuda')
            diffusion.load_state_dict(weight['ema'])
            print("Model Loaded!")

        trainer = Trainer(
            diffusion,
            dataset,
            val_dataset = val_dataset,
            image_size = input_size,
            depth_size = depth_size,
            residual_training=args.residual_training,
            train_batch_size = args.batchsize,
            train_lr = 1e-5,
            train_num_steps = args.epochs,                              # total training steps
            gradient_accumulate_every = args.gradient_accumulate,       # gradient accumulation steps
            ema_decay = 0.995,                                          # exponential moving average decay
            fp16 = args.fp16, #True,                                        # turn on mixed precision training with apex
            save_and_sample_every = save_and_sample_every,
            results_folders = res_folder_names,
            res_folder = args.res_folder,
            with_condition = with_condition,
            gpu_id=rank,
            world_size=world_size,
            num_workers=args.num_workers,
            self_consistency_config = self_consistency_config,
            skip_input_viz = skip_input_viz
        )
        if not args.validation:
            if len(resume_weight) > 0:
                # weight = torch.load(resume_weight, map_location='cuda')
                # diffusion.load_state_dict(weight['ema'])
                # print("Model Loaded!")
                trainer.load(args.resume_weight)
            trainer.train()
        else:
            if args.fast_sample:
                trainer.fast_sample()
            else:
                trainer.validate()

    except Exception as e:
        destroy_process_group()
        raise e

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_folder', type=str, default="Brast21")
    parser.add_argument('-i', '--input_modality', type=str, default=["seg"])
    parser.add_argument('-t', '--target_modality', type=str, default=["flair"])
    parser.add_argument('-R', '--res_folder', type=str, default="results")
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--glob_pos_emb', action='store_true')
    parser.add_argument('--none_zero_mask', action='store_true')
    parser.add_argument('--residual_training', action='store_true')
    parser.add_argument('--validation', action='store_true')
    parser.add_argument('--fast_sample', action='store_true')
    parser.add_argument('--no_self_consistency', action='store_true')
    parser.add_argument('--input_size', type=int, default=192)
    parser.add_argument('--depth_size', type=int, default=144)
    parser.add_argument('--num_channels', type=int, default=64)
    parser.add_argument('--num_res_blocks', type=int, default=2)
    parser.add_argument('--batchsize', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=10000000)
    parser.add_argument('--timesteps', type=int, default=250)
    parser.add_argument('--save_and_sample_every', type=int, default=1000)
    parser.add_argument('--with_condition', action='store_true')
    parser.add_argument('-r', '--resume_weight', type=str, default="")
    parser.add_argument('-g', '--gradient_accumulate', type=int, default=1)
    parser.add_argument('-w', '--num_workers', type=int, default=4)
    parser.add_argument('--random_crop_xy', type=int, default=0)
    parser.add_argument('--random_crop_z', type=int, default=0)
    parser.add_argument('--model_2D_2_config', type=str, default="config/brast_SR_2_resume_3D.json")
    parser.add_argument('--model_2D_3_config', type=str, default="config/brast_SR_3_resume_3D.json")
    parser.add_argument('--baseline', type=str, default="3D")

    args = parser.parse_args()

    # now = datetime.datetime.now().strftime("%y-%m-%dT%H%M%S")
    # log_dir = os.path.join("logs", now)
    # os.makedirs(log_dir, exist_ok=True)
    # writer = SummaryWriter(log_dir=log_dir)
    # args.writer = writer
    print("check GPU status and CUDA_VISIBLE_DEVICES")
    print(torch.cuda.is_available())
    print("num GPUs:", torch.cuda.device_count())
    if args.fp16:
        print("using fp16")
    if args.validation:
        print("!! validation mode")
        args.res_folder = "validation_" + args.res_folder

    args.res_folder = args.res_folder + "_ensemble"
    results_folder_exp = args.res_folder +"/"+ args.input_modality.replace(" ", "&") + "-to-" + args.target_modality.replace(" ", "=")
    os.makedirs(args.res_folder, exist_ok=True)
    os.makedirs(results_folder_exp, exist_ok=True)

    yaml_file_path = results_folder_exp + '/config.yaml'
    args_dict = vars(args)
    with open(yaml_file_path, 'w') as file:
        yaml.dump(args_dict, file, default_flow_style=False)

    # exit()
    mp.spawn(main, args=(args.gpus, args), nprocs=args.gpus)
