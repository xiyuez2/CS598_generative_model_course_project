#-*- coding:utf-8 -*-
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor
from glob import glob
from utils.dtypes_brats import LabelEnum
import matplotlib.pyplot as plt
import nibabel as nib
import torchio as tio
import numpy as np
import torch
import re
import os
from scipy.ndimage import zoom
from utils.util import bbox2mask, random_bbox
from utils.fft import clear, get_mask, fft2, ifft2
import math

from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, MeanSquaredError
import torch

# Define evaluation Metrics
psnr = PeakSignalNoiseRatio()
ssim = StructuralSimilarityIndexMeasure()
mse = MeanSquaredError()


class NiftiImageGenerator(Dataset):
    def __init__(self, imagefolder, input_size, depth_size, transform=None):
        self.imagefolder = imagefolder
        self.input_size = input_size
        self.depth_size = depth_size
        self.inputfiles = glob(os.path.join(imagefolder, '*.nii.gz'))
        self.scaler = MinMaxScaler()
        self.transform = transform

    def read_image(self, file_path):
        img = nib.load(file_path).get_fdata()
        img = self.scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape) # 0 -> 1 scale
        return img

    def plot_samples(self, n_slice=15, n_row=4):
        samples = [self[index] for index in np.random.randint(0, len(self), n_row*n_row)]
        for i in range(n_row):
            for j in range(n_row):
                sample = samples[n_row*i+j]
                sample = sample[0]
                plt.subplot(n_row, n_row, n_row*i+j+1)
                plt.imshow(sample[:, :, n_slice])
        plt.show()

    def __len__(self):
        return len(self.inputfiles)

    def __getitem__(self, index):
        inputfile = self.inputfiles[index]
        img = self.read_image(inputfile)
        h, w, d= img.shape
        if h != self.input_size or w != self.input_size or d != self.depth_size:
            img = tio.ScalarImage(inputfile)
            cop = tio.Resize((self.input_size, self.input_size, self.depth_size))
            img = np.asarray(cop(img))[0]

        if self.transform is not None:
            img = self.transform(img)
        return img

class NiftiPairImageGenerator(Dataset):
    def __init__(self,
            dataset_folder: str,
            input_modality,
            target_modality,
            input_size: int,
            depth_size: int,
            input_channel: int = 8,
            transform=None,
            target_transform=None,
            full_channel_mask=False,
            combine_output=False,
            train = True,
            random_crop_size = (),
            global_pos_emb = False,
            none_zero_mask = False,
            residual_training = False
        ):
        self.dataset_folder = dataset_folder
        folders = sorted(glob(os.path.join(dataset_folder, '*')))
        if train:
            self.input_folders = folders[:int(0.8*len(folders))]
        else:
            self.input_folders = folders[int(0.8*len(folders)):]
        
        # if input modality is a str
        # split it into a list of modalities
        if isinstance(input_modality, str):
            input_modality = input_modality.split(' ')
        if isinstance(target_modality, str):
            target_modality = target_modality.split(' ')
        self.input_modality = input_modality
        self.target_modality = target_modality
        self.input_size = input_size
        self.depth_size = depth_size
        self.input_channel = input_channel
        self.transform = transform
        self.target_transform = target_transform
        self.full_channel_mask = full_channel_mask
        self.combine_output = combine_output
        self.random_crop_size = random_crop_size
        self.global_pos_emb = global_pos_emb
        self.none_zero_mask = none_zero_mask
        self.residual_training = residual_training
        print("1. when using mask for inpainting, please check the mask is correct")
        print("mask will be used in computing the loss")
        print("2. random crop with inpainting not supported yet since it is to be dicussed")

    def label2value(self, masked_img):
        result_img = masked_img.copy()
        result_img[result_img==LabelEnum.BACKGROUND.value] = 0.0
        result_img[result_img==LabelEnum.TUMORAREA1.value] = 0.25
        result_img[result_img==LabelEnum.TUMORAREA2.value] = 0.5
        result_img[result_img==LabelEnum.TUMORAREA3.value] = 0.75
        result_img[result_img==LabelEnum.BRAINAREA.value] = 1.0
        #result_img = self.scaler.fit_transform(result_img.reshape(-1, result_img.shape[-1])).reshape(result_img.shape)
        return result_img

    def label2masks(self, masked_img):
        result_img = np.zeros(masked_img.shape + (4,))   # ( (H, W, D) + (2,)  =  (H, W, D, 2)  -> (B, 2, H, W, D))
        result_img[masked_img==LabelEnum.TUMORAREA1.value, 0] = 1
        result_img[masked_img==LabelEnum.TUMORAREA2.value, 1] = 1
        result_img[masked_img==LabelEnum.TUMORAREA3.value, 2] = 1
        result_img[masked_img==LabelEnum.BRAINAREA.value, 3] = 1
        return result_img

    def combine_mask_channels(self, masks): # masks: (2, H, W, D), 2->mask channels
        result_img = np.zeros(masks.shape[1:])
        result_img += LabelEnum.TUMORAREA1.value * masks[0]
        result_img += LabelEnum.TUMORAREA2.value * masks[1]
        result_img += LabelEnum.TUMORAREA3.value * masks[2]
        result_img += LabelEnum.BRAINAREA.value * masks[3]
        return result_img

    def read_image(self, file_path):
        img = nib.load(file_path).get_fdata()
        return img

    def downsample(self, img, downsample,mask = None):
        # input shape: (H, W, D), a 3D image
        # output shape: (H, W, D), a 3D image with the same shape, 
        # but downsampled according to the downsample method
        d,w,h = img.shape
        # split downsample into downsample method and its factor
        # if no downsample factor, factor is None
        downsample, downsampling_factor, *_ = downsample.split("x") + [None]
        
        # Bicubic downsampling
        if downsample == 'bicubic':
            downsampling_factor = int(downsampling_factor or 2) # default downsampling factor is 2
            downsampling_factor = 1/int(downsampling_factor)
            downsampled_img = zoom(img, downsampling_factor, order=3)
            downsampled_img = zoom(downsampled_img, 1/downsampling_factor, order=0)
            return downsampled_img, mask
        elif downsample == "pool":
            # turn img to torch tensor from numoy
            img = torch.from_numpy(img).float()
            downsampling_factor = int(downsampling_factor or 2) # default downsampling factor is 2
            # avg pool img
            img = img.unsqueeze(0).unsqueeze(0)
            img = torch.nn.functional.avg_pool3d(img,downsampling_factor, downsampling_factor) # kernel size and stride
            img = img.squeeze(0).squeeze(0).numpy()
            # resize it back to original size
            img = np.repeat(img, downsampling_factor, axis=0)
            img = np.repeat(img, downsampling_factor, axis=1)
            img = np.repeat(img, downsampling_factor, axis=2)

            return img, mask

        elif downsample == "compressed":
            downsampling_factor = int(downsampling_factor or 24) # default conpressed sensing acc factor is 24
            # print("compressed sensing not debugged yet")
            # raise
            # print(img.shape)
            img = torch.from_numpy(img).float().transpose(0, 2).unsqueeze(1) # 152 1 192 192
            label_kspace = fft2(img)
            forward_mask = get_mask(torch.zeros(1, 1, 192, 192), 192, 1, #1 is batch size
                            type='poisson', acc_factor=downsampling_factor, center_fraction=None)
            forward_mask_full = forward_mask.repeat(152, 1, 1, 1)
            # print("forward_mask_full shape: ", forward_mask_full.shape)
            # print("label_kspace shape: ", label_kspace.shape)
            measure_kspace = label_kspace * forward_mask_full
            measure_dagger = torch.real(ifft2(measure_kspace))
            measure_dagger = measure_dagger.squeeze().transpose(0, 2).numpy() # 192 192 152
            # print("measure_dagger shape:", measure_dagger.shape)
            return measure_dagger, mask

        elif downsample == "CSSR":
            downsampling_factor = int(downsampling_factor or 24) # default conpressed sensing acc factor is 24
            # print("compressed sensing not debugged yet")
            # raise
            # print(img.shape)
            img = torch.from_numpy(img).float().transpose(0, 2).unsqueeze(1) # 152 1 192 192
            label_kspace = fft2(img)
            # forward_mask = get_mask(torch.zeros(1, 1, 192, 192), 192, 1, #1 is batch size
            #                 type='poisson', acc_factor=downsampling_factor, center_fraction=None)
            forward_mask = torch.zeros(1, 1, 192, 192)
            # the center part should be 1 and others 0
            center_len = math.ceil((w * h / downsampling_factor) ** 0.5)
            forward_mask[0, 0, (w - center_len) // 2: (w + center_len) // 2, (h - center_len) // 2: (h + center_len) // 2] = 1
            forward_mask_full = forward_mask.repeat(152, 1, 1, 1)
            # print("forward_mask_full shape: ", forward_mask_full.shape)
            # print("label_kspace shape: ", label_kspace.shape)
            measure_kspace = label_kspace * forward_mask_full
            measure_dagger = torch.real(ifft2(measure_kspace))
            measure_dagger = measure_dagger.squeeze().transpose(0, 2).numpy() # 192 192 152
            # print("measure_dagger shape:", measure_dagger.shape)
            return measure_dagger, mask

        elif downsample == "mask":
            if mask is None:
                downsampling_factor = downsampling_factor or "center" # default mask method is "center"
                if downsampling_factor == "center":
                    mask = bbox2mask(img.shape, (d//4, w//4, h//4, d//2, w//2,h//2))
                elif downsampling_factor == "center2":
                    mask = bbox2mask(img.shape, ((d*3)//8, (w*3)//8, (w*3)//8, d//4, w//4, h//4))
                elif downsampling_factor == "bbox":
                    mask = bbox2mask(img.shape, random_bbox())
                else:
                    print("mask method not implemented")
                    raise NotImplementedError
                # mask = (1. - mask)
            
            downsampled_img = img * (1. - mask) + mask * np.random.randn(*img.shape)
            return downsampled_img, mask
        else:
            print("downsample method not implemented")
            raise NotImplementedError

    def plot(self, index, n_slice=30):
        data = self[index]
        input_img = data['input']
        target_img = data['target']
        plt.subplot(1, 2, 1)
        plt.imshow(input_img[:, :, n_slice])
        plt.subplot(1, 2, 2)
        plt.imshow(target_img[:, :, n_slice])
        plt.show()

    def center_crop(self, img):
        h, w, d = img.shape
        if h != self.input_size or w != self.input_size or d != self.depth_size:
            # img = tio.ScalarImage(tensor=img[np.newaxis, ...])
            # cop = tio.Resize((self.input_size, self.input_size, self.depth_size))
            # img = np.asarray(cop(img))[0]
            # center crop
            h_start = (h - self.input_size) // 2
            w_start = (w - self.input_size) // 2
            d_start = (d - self.depth_size) // 2
            img = img[h_start:h_start+self.input_size, w_start:w_start+self.input_size, d_start:d_start+self.depth_size]
        return img

    def center_crop_4d(self, input_img):
        h, w, d, c = input_img.shape
        # print(input_img.shape)
        result_img = np.zeros((self.input_size, self.input_size, self.depth_size, c))
        if h != self.input_size or w != self.input_size or d != self.depth_size:
            for ch in range(c):
                buff = input_img.copy()[..., ch]
                # center crop
                h_start = (h - self.input_size) // 2
                w_start = (w - self.input_size) // 2
                d_start = (d - self.depth_size) // 2
                buff = buff[h_start:h_start+self.input_size, w_start:w_start+self.input_size, d_start:d_start+self.depth_size]
                result_img[..., ch] += buff
            return result_img
        else:
            return input_img

    def sample_conditions(self, batch_size: int):
        indexes = np.random.randint(0, len(self), batch_size)
        input_tensors = []
        target_tensors = []
        masks_tensors = []
        # data = 
        for index in indexes:
            data = self[index]
            input_tensors.append(data['input'])
            target_tensors.append(data['target'])
            masks_tensors.append(data['mask'])

        # print(np.shape(input_tensors[0]), np.shape(target_tensors[0]), np.shape(masks_tensors[0]))
        # plt.imshow(input_tensors[0][0,:,:,32])
        # plt.savefig("input_img_sampled.png")
        # plt.imshow(target_tensors[0][0,:,:,32])
        # plt.savefig("target_img_sampled.png")
        # print(np.shape(target_tensors[0]),np.shape(masks_tensors[0]))
        target_tensor = torch.cat(target_tensors, 0).cuda().unsqueeze(0)
        input_tensor = torch.cat(input_tensors, 0).cuda().unsqueeze(0)
        masks_tensor = torch.cat(masks_tensors, 0).cuda().unsqueeze(0)

        return input_tensor, target_tensor, masks_tensor # 1 4 152 192 192

    def __len__(self):
        return len(self.input_folders)

    def get_bound(self, data, return_coord=False):
        """
        get the boundary of image z y x
        data is padded with 0
        """
        data_0 = data - data.min()

        z, y, x = np.where(data_0)
        z_start, z_end = np.min(z), np.max(z)
        y_start, y_end = np.min(y), np.max(y)
        x_start, x_end = np.min(x), np.max(x)

        indicator = np.ones_like(data, dtype=bool)
        indicator[z_start:z_end, y_start:y_end, x_start:x_end] = False
        if return_coord:
            return z_start, z_end, y_start, y_end, x_start, x_end, indicator
        return indicator

    def mri_data_norm(self, data, scale=6.0, return_v=False):
        
        X = np.array(data.astype(float))
        X_std = np.array((X - np.min(X)) / (np.max(X) - np.min(X)))
        X_std = X_std * 2. - 1.

        return X_std
        
        # we do center crop instead of crop out zeros uncomment # !!! # lines
        # borrowed from Song Yang's repo
        # to use crop out zero instead
        # important to transfer datatype to keep division works
        data = data.astype(float)
        # get a box mask to remove background
        min_z, max_z, min_y, max_y, min_x, max_x, indicator = self.get_bound(data, return_coord=True)
        # !!! # crop_data = np.array(data[min_z:max_z, min_y:max_y, min_x:max_x] * 1.0)
        crop_data = np.array(data * 1.0)
        
        # clip outliers
        # print(np.max(data), ";", np.min(data), ";", np.mean(data),";",np.std(data))
        # mean, std = np.mean(crop_data), np.std(crop_data)
        if 1 : #np.max(crop_data) > 6000:
            print("warning max value larger than 6000: ", np.max(crop_data), np.min(crop_data), np.mean(crop_data), np.std(crop_data))
        crop_data = np.clip(crop_data, 0, 6000)
        # print("after clip")
        # print(np.max(crop_data), ";", np.min(crop_data), ";", np.mean(crop_data),";",np.std(crop_data))
        
        # normalize scale [-1,1]
        min_v = 0
        crop_data = np.array(crop_data - min_v)
        max_v = np.max(crop_data) * 1.0
        crop_data = np.array(crop_data) / max_v
        # data = crop_data * 2 - 1 # [0,1]  ->  [-1, 1] 

        # !!! # data[min_z: -max_z, min_y:max_y, min_x:max_x] = np.array(crop_data)
        # !!! # data[indicator] = 0

        if return_v:
            return np.array(crop_data), [min_v, max_v, np.float(min_y), np.float(max_y), np.float(min_x), np.float(max_x)]
        else:
            return np.array(crop_data)


    def get_modality(self, folder, modality, mask = None):
        # split modality into modality and its downsample method
        # if no downsample method, downsample is None
        modality, downsample, *_ = modality.split("_") + [None]
        # read in files
        if modality in ['t1', 't1ce', 't2', 'flair', 'seg']:
            input_file = glob(os.path.join(folder, '*' + modality + '.nii.gz'))
            
            if len(input_file) != 1:
                print("no file or more than 1 files", input_file)
                print(os.path.join(folder, '*'+modality+'.nii.gz'))
                raise NotImplementedError
            img = self.read_image(input_file[0])
        else:
            print(modality + " is not supported")
            raise NotImplementedError
        
        # modality specific pre-processing
        if modality == 'seg':
            img = self.label2masks(img) if self.full_channel_mask else img
            img = self.center_crop(img) if not self.full_channel_mask else self.center_crop_4d(img)
        elif modality in ['t1', 't1ce', 't2', 'flair']:
            # print(img.shape)
            img = self.center_crop(img)
            # print(modality)
            # print(img.shape)
            img = img[..., np.newaxis]

        # to be discussed
        if modality in ['t1', 't1ce', 't2', 'flair']:
            for i in range(img.shape[-1]):
                img[...,i] = self.mri_data_norm(img[...,i])
        
        # print(img.shape)
        for i in range(img.shape[-1]):
            img[...,i], mask = self.downsample(img[...,i], downsample, mask=mask) if downsample is not None else (img[...,i], mask)
        if not mask is None:
            mask = mask[..., np.newaxis]
        return img, mask #, meta_data if modality in ['t1', 't1ce', 't2', 'flair'] else None

    def evaluate(self, gt_image: torch.Tensor, output: torch.Tensor, mask = None, normalize=False):
        """Computes MSE, PSNR and SSIM between two images only in the masked region.

        Normalizes the two images to [0;1] based on the gt_image maximal value in the masked region.
        Requires input to have shape (1,1, X,Y,Z), meaning only one sample and one channel.
        For MSE and PSNR we use the respective torchmetrics libraries on the voxels that are covered by the mask.
        For SSIM, we first zero all non-mask voxels, then we apply regular SSIM on the complete volume. In the end we take
        the "full SSIM" image from torchmetrics and only take the values relating to voxels within the mask.
        The main difference between the original torchmetrics SSIM and this substitude for masked images is that we pad
        with zeros while torchmetrics does reflection padding at the cuboid borders.
        This does slightly bias the SSIM voxel values at the mask surface but does not influence the resulting participant
        ranking as all submission underlie the same bias.

        Args:
            gt_image (torch.Tensor): The t1n ground truth image (t1n.nii.gz)
            output (torch.Tensor): The inferred t1n image
            mask (torch.Tensor): The inference mask (mask.nii.gz)
            normalize (bool): Normalizes the input by dividing trough the maximal value of the gt_image in the masked
                region. Defaults to True

        Raises:
            UserWarning: If you dimensions do not match the (torchmetrics) requirements: 1,1,X,Y,Z

        Returns:
            _type_: MSE, PSNR, SSIM as float each
        """
        # while gt and out dim less than 5, add 1 to the front
        # if gt is not numpy, convert it to numpy


        while len(gt_image.shape) < 5:
            gt_image = gt_image.unsqueeze(0)
        
        while len(output.shape) < 5:
            output = output.unsqueeze(0)

        if not (output.shape[0] == 1 and output.shape[1] == 1):
            raise UserWarning(f"All inputs have to be 5D with the first two dimensions being 1. Your output dimension: {output.shape}")

        # Get Infill region (we really are only interested in the infill region)
        if mask is None or not (mask.shape == gt_image.shape):
            # print("creating all-one mask")
            mask = torch.ones_like(gt_image).float()

        output_infill = output * mask
        gt_image_infill = gt_image * mask

        # Normalize to [0;1] based on GT (otherwise MSE will depend on the image intensity range)
        if normalize:
            v_max = gt_image_infill.max()
            output_infill /= v_max
            gt_image_infill /= v_max
        

        # SSIM - apply on complete masked image but only take values from masked region
        ssim_idx = ssim(gt_image_infill, output_infill)

        SSIM = ssim_idx.mean()

        # only voxels that are to be inferred (-> flat array)
        # gt_image_infill = gt_image_infill[mask]
        # output_infill = output_infill[mask]

        # MSE
        MSE = mse(gt_image_infill, output_infill) / mask.mean()

        # PSNR - similar to pytorch PeakSignalNoiseRatio until 4 digits after decimal point
        PSNR = 10.0 * torch.log10((torch.max(gt_image_infill) - torch.min(gt_image_infill)) ** 2 / MSE)

        return float(MSE), float(PSNR), float(SSIM)


    def __getitem__(self, index):
        cur_folder = self.input_folders[index]
        input_data = []
        mask = None
        for input_m in self.input_modality:
            cur_im, mask = self.get_modality(cur_folder, input_m, mask = mask)
            input_data.append(cur_im)

        # add global position embedding
        if self.global_pos_emb:
            i, j, k = np.meshgrid(np.arange(self.input_size), np.arange(self.input_size), np.arange(self.depth_size), indexing='ij')
            pos_emb = np.zeros((self.input_size, self.input_size, self.depth_size, 3))
            pos_emb[..., 0] = i / self.input_size
            pos_emb[..., 1] = j / self.input_size
            pos_emb[..., 2] = k / self.depth_size
            pos_emb = pos_emb * 2. - 1.
            input_data.append(pos_emb)
        # add none zero mask
        if self.none_zero_mask:
            none_zero_mask = input_data[0] > -0.99
            none_zero_mask = none_zero_mask * 2. - 1.
            input_data.append(none_zero_mask)
        input_img = np.concatenate(input_data, axis=-1)

        if mask is None:
            mask = np.ones_like(input_img[...])
        else:
            final_mask = np.zeros_like(input_img[...])
            for i in range(np.shape(input_img)[-1]):
                final_mask[...,i] = mask[...,0]
            mask = final_mask

        target_data = []
        for target_m in self.target_modality:
            cur_im, target_mask = self.get_modality(cur_folder, target_m)
            target_data.append(cur_im)
        target_img = np.concatenate(target_data, axis=-1)
        
        if len(self.random_crop_size) > 0:
            # randomly crop long three axis:
            x_start = np.random.randint(0, input_img.shape[0] - self.random_crop_size[0])
            y_start = np.random.randint(0, input_img.shape[1] - self.random_crop_size[1])
            z_start = np.random.randint(0, input_img.shape[2] - self.random_crop_size[2])

            input_img = input_img[x_start:x_start+self.random_crop_size[0], y_start:y_start+self.random_crop_size[1], z_start:z_start+self.random_crop_size[2], :]
            target_img = target_img[x_start:x_start+self.random_crop_size[0], y_start:y_start+self.random_crop_size[1], z_start:z_start+self.random_crop_size[2], :]
            mask = mask[x_start:x_start+self.random_crop_size[0], y_start:y_start+self.random_crop_size[1], z_start:z_start+self.random_crop_size[2], :]
            # save image to disk with plt.savefig()
            
        # target/input specific pre-processing
        if self.transform is not None:
            print("using none default input transform")
            print("input transform: ", self.transform)
            input_img = self.transform(input_img)
            mask = self.transform(mask)

        else:
            input_img = torch.from_numpy(input_img).float()
            input_img = input_img.permute(3, 0, 1, 2)
            input_img = input_img.transpose(3, 1)
            mask = torch.from_numpy(mask).float()
            mask = mask.permute(3, 0, 1, 2)
            mask = mask.transpose(3, 1)

        if self.target_transform is not None:
            print("using none default target transform")
            print("target transform: ", self.target_transform)
            target_img = self.target_transform(target_img)
        else:
            target_img = torch.from_numpy(target_img).float()
            target_img = target_img.permute(3, 0, 1, 2)
            target_img = target_img.transpose(3, 1)

        if self.combine_output:
            return torch.cat([target_img, input_img], 0)
        
        if self.residual_training:
            for i in range(target_img.shape[0]):
                target_img[i] = (target_img[i] - input_img[i]) / 2.
        # plt.imshow(input_img[0,:,:,32])
        # plt.savefig("input_img_item.png")
        # plt.imshow(target_img[0,:,:,32])
        # plt.savefig("target_img_item.png")
        # plt.imshow(mask[0,:,:,32])
        # plt.savefig("mask.png")
        
        return {'input':input_img, 'target':target_img, 'mask':mask}

# if this is the main file
    
if __name__ == '__main__':
    dataset = NiftiPairImageGenerator(
        dataset_folder='Brast21',
        input_modality='flair_poolx4',
        target_modality='flair',
        input_size=192,
        depth_size=152,
        transform=None,
        target_transform=None,
        global_pos_emb = True,
        none_zero_mask = True,
        residual_training = True
    )
    # loop through the dataset
    # make dataloader
    print(len(dataset))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    print(len(dataloader))
    for data in dataloader:
        for k in data.keys():
            print(k, data[k].shape)
        


