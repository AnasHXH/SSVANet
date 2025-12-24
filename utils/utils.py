# utils/utils.py

import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def calculate_psnr_ssim(pred, target):
    pred_np = pred.cpu().numpy().transpose(0, 2, 3, 1)
    target_np = target.cpu().numpy().transpose(0, 2, 3, 1)
    total_psnr = 0
    total_ssim = 0
    batch_size = pred_np.shape[0]
    for i in range(batch_size):
        psnr_value = psnr(target_np[i], pred_np[i], data_range=1)
        ssim_value = ssim(target_np[i], pred_np[i], data_range=1, channel_axis=-1)
        total_psnr += psnr_value
        total_ssim += ssim_value
    avg_psnr = total_psnr / batch_size
    avg_ssim = total_ssim / batch_size
    return avg_psnr, avg_ssim
