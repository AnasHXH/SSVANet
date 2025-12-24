# evaluate.py

import os
import lpips
import torch
import numpy as np
import torchvision
import torchvision.transforms.functional as F
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage.color import rgb2lab
import yaml

def to_tensor(img):
    img = Image.fromarray(img)
    img_t = F.to_tensor(img).float()
    img_t = img_t.unsqueeze(dim=0)
    return img_t

def calc_rmse(real_img, fake_img):
    # Convert to LAB color space
    real_lab = rgb2lab(real_img)
    fake_lab = rgb2lab(fake_img)
    rmse = np.sqrt(((real_lab - fake_lab) ** 2).mean())
    return rmse

def metric(gt, pre, loss_fn_vgg):
    transf = torchvision.transforms.Compose(
        [torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
    )
    lpips_value = loss_fn_vgg(transf(pre[0]).cuda(), transf(gt[0]).cuda()).item()

    pre = pre * 255.0
    pre = pre.permute(0, 2, 3, 1)
    pre = pre.detach().cpu().numpy().astype(np.uint8)[0]

    gt = gt * 255.0
    gt = gt.permute(0, 2, 3, 1)
    gt = gt.cpu().detach().numpy().astype(np.uint8)[0]

    psnr = compare_psnr(gt, pre)
    ssim = compare_ssim(gt, pre, data_range=255, channel_axis=-1)
    rmse = calc_rmse(gt, pre)

    return psnr, ssim, lpips_value, rmse

def evaluation():
    # Load configuration
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    gt_root = config['val_gt_dir']
    #pred_root = os.path.join(config['output_dir'], 'test_results')
    pred_root = config['output_dir']

    fnames = os.listdir(gt_root)
    fnames.sort()

    psnr_all_list, ssim_all_list, lpips_all_list, rmse_all_list = [], [], [], []

    loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()  # vgg is used in the paper

    for fname in fnames:
        gt_path = os.path.join(gt_root, fname)
        pre_path = os.path.join(pred_root, fname)

        if not os.path.exists(pre_path):
            continue

        gt_image = np.array(Image.open(gt_path).convert('RGB'))
        pre_image = np.array(Image.open(pre_path).convert('RGB'))

        # Resize pre_image to gt_image size if necessary
        if pre_image.shape != gt_image.shape:
            pre_image = np.array(Image.fromarray(pre_image).resize(gt_image.shape[1::-1], Image.BILINEAR))

        gt = to_tensor(gt_image)
        pre = to_tensor(pre_image)

        psnr_all, ssim_all, lpips_all, rmse_all = metric(gt, pre, loss_fn_vgg)

        psnr_all_list.append(psnr_all)
        ssim_all_list.append(ssim_all)
        lpips_all_list.append(lpips_all)
        rmse_all_list.append(rmse_all)

    print('-----------------------------------------------------------------------------')
    print(f'All PSNR: {round(np.average(psnr_all_list), 4)}')
    print(f'All SSIM: {round(np.average(ssim_all_list), 4)}')
    print(f'All LPIPS: {round(np.average(lpips_all_list), 4)}')
    print(f'All RMSE: {round(np.average(rmse_all_list), 4)}')

if __name__ == "__main__":
    evaluation()
