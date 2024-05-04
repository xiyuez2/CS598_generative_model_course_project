#-*- coding:utf-8 -*-
# +
from torchvision.transforms import RandomCrop, Compose, ToPILImage, Resize, ToTensor, Lambda
from diffusion_model.trainer_brats import GaussianDiffusion, Trainer
from diffusion_model.unet_brats import create_model
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

        # input tensor: (B, 1, H, W, D)  value range: [-1, 1]
        # transform = Compose([
        #     Lambda(lambda t: torch.tensor(t).float()),
        #     Lambda(lambda t: t.permute(3, 0, 1, 2)),
        #     Lambda(lambda t: t.transpose(3, 1)),
        # ])

        # input_transform = Compose([
        #     Lambda(lambda t: torch.tensor(t).float()),
        #     Lambda(lambda t: t.permute(3, 0, 1, 2)),
        #     Lambda(lambda t: t.transpose(3, 1)),
        # ])

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
        for i, name in enumerate(target_modality):
            if not name in ["flair", "t1", "t1ce", "t2"]:
                continue
            for j, input_name in enumerate(input_modality):
                if name + "_" in input_name:
                    self_consistency_config[i] = (j, input_name)

        print("self consistency enforce:", self_consistency_config)


        cond_channels = len(input_modality)
        if "seg" in input_modality or "reverse" in input_modality:
            cond_channels -= 1
        if args.glob_pos_emb:
            cond_channels += 3
        if args.none_zero_mask:
            cond_channels += 1

        target_channels = len(target_modality)
        # if "seg" in target_modality:
        #     target_channels += 3

        in_channels = cond_channels + target_channels if with_condition or with_pairwised else 1
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

        model = create_model(input_size_model, 
            num_channels, 
            num_res_blocks, 
            in_channels=in_channels, 
            out_channels=out_channels, 
            use_checkpoint = use_ckpt, 
            attention_resolutions = "",
            use_fp16=args.fp16
        ).cuda()

        diffusion = GaussianDiffusion(
            model,
            image_size = input_size,
            depth_size = depth_size,
            timesteps = args.timesteps,   # number of steps
            loss_type = 'hybrid',    # L1 and L2
            with_condition=with_condition,
            channels=out_channels,
            fp16=args.fp16
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
            trainer.train()
        else:
            trainer.fast_sample()

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
    parser.add_argument('--input_size', type=int, default=192)
    parser.add_argument('--depth_size', type=int, default=152)
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
        print("validation mode")
        args.res_folder = "validation_" + args.res_folder
    
    
    results_folder_exp = args.res_folder +"/"+ args.input_modality.replace(" ", "&") + "-to-" + args.target_modality.replace(" ", "=")
    os.makedirs(args.res_folder, exist_ok=True)
    os.makedirs(results_folder_exp, exist_ok=True)

    yaml_file_path = results_folder_exp + '/config.yaml'
    args_dict = vars(args)
    with open(yaml_file_path, 'w') as file:
        yaml.dump(args_dict, file, default_flow_style=False)
    
    # exit()
    mp.spawn(main, args=(args.gpus, args), nprocs=args.gpus)
    # main(0, 1, args)
    