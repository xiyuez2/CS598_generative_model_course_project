import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
from skimage.metrics import structural_similarity as ssim_sk
import os
import argparse
import glob
from tqdm import tqdm
from sklearn import metrics

def mmd_linear(X, Y):
    """MMD using linear kernel (i.e., k(x,y) = <x,y>)
    Note that this is not the original linear MMD, only the reformulated and faster version.
    The original version is:
        def mmd_linear(X, Y):
            XX = np.dot(X, X.T)
            YY = np.dot(Y, Y.T)
            XY = np.dot(X, Y.T)
            return XX.mean() + YY.mean() - 2 * XY.mean()

    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]

    Returns:
        [scalar] -- [MMD value]
    """
    delta = X.mean(0) - Y.mean(0)
    return delta.dot(delta.T)


def mmd_rbf(X, Y, gamma=1.0):
    """MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))

    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]

    Keyword Arguments:
        gamma {float} -- [kernel parameter] (default: {1.0})

    Returns:
        [scalar] -- [MMD value]
    """
    XX = metrics.pairwise.rbf_kernel(X, X, gamma)
    YY = metrics.pairwise.rbf_kernel(Y, Y, gamma)
    XY = metrics.pairwise.rbf_kernel(X, Y, gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()


def mmd_poly(X, Y, degree=2, gamma=1, coef0=0):
    """MMD using polynomial kernel (i.e., k(x,y) = (gamma <X, Y> + coef0)^degree)

    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]

    Keyword Arguments:
        degree {int} -- [degree] (default: {2})
        gamma {int} -- [gamma] (default: {1})
        coef0 {int} -- [constant item] (default: {0})

    Returns:
        [scalar] -- [MMD value]
    """
    XX = metrics.pairwise.polynomial_kernel(X, X, degree, gamma, coef0)
    YY = metrics.pairwise.polynomial_kernel(Y, Y, degree, gamma, coef0)
    XY = metrics.pairwise.polynomial_kernel(X, Y, degree, gamma, coef0)
    return XX.mean() + YY.mean() - 2 * XY.mean()



def ssim(gt, res):
    # if mask is None:
    #     return ssim_sk(gt,res,data_range = 1)
    # else:
    #     #TODO: add support for masked ssim here
    #     return ssim_sk(gt,res,data_range = 1)
    return ssim_sk(gt,res,data_range = 1)

class ResDataset(Dataset):
    def __init__(self, args):
        self.gt_files = sorted(glob.glob(args.res + '/*GT*.npy'))

    def __len__(self):
        return len(self.gt_files)

    def __getitem__(self, idx):
        gt_file = self.gt_files[idx]
        res_file = gt_file.replace('GT', 'inferece')

        gt_img = np.load(gt_file)
        result_img = np.load(res_file)
        
        # normalize
        max_v, min_v = np.max(gt_img), np.min(gt_img)
        if min_v != 0 or max_v != 1:
            print(f"min value is not 0 or max value is not 1, {min_v}, {max_v}")

        gt_img = (gt_img - min_v)/(max_v - min_v)
        result_img = (result_img - min_v)/(max_v - min_v)

        # L2
        L2 = ((gt_img - result_img) ** 2).mean()
        PSNR = 10 * np.log10(1 / L2)
        SSIM = ssim(gt_img, result_img)
        # down sample gt and res img using avg pooling
        gt_img = torch.tensor(gt_img).unsqueeze(0).unsqueeze(0)
        result_img = torch.tensor(result_img).unsqueeze(0).unsqueeze(0)
        # gt_img = torch.nn.functional.avg_pool3d(gt_img, 2)
        # result_img = torch.nn.functional.avg_pool3d(result_img, 2)
        gt_img = gt_img.squeeze().squeeze().numpy()
        result_img = result_img.squeeze().squeeze().numpy()
        # flatten the image
        gt_img = gt_img.flatten()
        result_img = result_img.flatten()

        return SSIM, PSNR, L2, gt_img, result_img


def main():
    parser = argparse.ArgumentParser(description='Calculate metrics norm between NII.GZ files.')
    # parser.add_argument('--gt_dir', type=str, default = '/home/shirui/INPAINT/data/augmented_data')
    # parser.add_argument('--mask', type=str, default = '')
    # parser.add_argument('--target', type=str, default = 'flair')

    parser.add_argument('--res', type=str, required=True, help='Directory containing result NII.GZ files')
    parser.add_argument('--batch_size', type=int, default = 4)
    parser.add_argument('--num_workers', type=int, default = 16)
    
    args = parser.parse_args()

    dataset = ResDataset(args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    ssim_values = []
    l2_values = []
    psnr_values = []
    ratio = 1
    gt_imgs = np.zeros((len(dataset), 192*192*152//(ratio**3) ))
    res_imgs = np.zeros((len(dataset), 192*192*152//(ratio**3) ))
    for j, batch in enumerate(tqdm(dataloader)):
        SSIM, PSNR, L2, gt_img, res_img = batch
        for i in range(len(SSIM)):
            ssim_values.append(SSIM[i].item())
            l2_values.append(L2[i].item())
            psnr_values.append(PSNR[i].item())
            gt_imgs[j*args.batch_size + i] = gt_img[i]
            res_imgs[j*args.batch_size + i] = res_img[i]

    average_ssim = np.mean(ssim_values)
    average_l2 = np.mean(l2_values)
    average_psnr = np.mean(psnr_values)

    print(f"Average SSIM: {average_ssim}")
    print(f"Average L2: {average_l2}")
    print(f"Average PSNR: {average_psnr}")
    # MMD
    mmd = mmd_linear(gt_imgs, res_imgs)
    print(f"mmd_linear: {mmd}")
    mmd = mmd_rbf(gt_imgs, res_imgs)
    print(f"mmd_rbf: {mmd}")
    mmd = mmd_poly(gt_imgs, res_imgs)
    print(f"mmd_poly: {mmd}")

if __name__ == '__main__':
    main()


