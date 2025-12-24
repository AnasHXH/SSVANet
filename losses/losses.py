# losses/losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class VGGLoss(nn.Module):
    def __init__(self, device):
        super(VGGLoss, self).__init__()
        vgg = models.vgg19(pretrained=True).features.to(device).eval()
        self.vgg_layers = nn.Sequential(*list(vgg)[:36]).to(device)
        for param in self.vgg_layers.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        x_vgg = self.vgg_layers(x)
        y_vgg = self.vgg_layers(y)
        return F.l1_loss(x_vgg, y_vgg)

class FocalFrequencyLoss(nn.Module):
    def __init__(self, alpha=1.0, device=None):
        super(FocalFrequencyLoss, self).__init__()
        self.alpha = alpha
        self.device = device

    def forward(self, input, target):
        input_fft = torch.fft.fft2(input)
        target_fft = torch.fft.fft2(target)
        diff = input_fft - target_fft
        abs_diff = torch.abs(diff)
        loss = torch.pow(abs_diff, self.alpha)
        return torch.mean(loss)

class CombinedLoss(nn.Module):
    def __init__(self, lambda_vgg=0.01, lambda_ff=0.1, device=None):
        super(CombinedLoss, self).__init__()
        self.lambda_vgg = lambda_vgg
        self.lambda_ff = lambda_ff
        self.l1_loss = nn.L1Loss()
        self.vgg_loss = VGGLoss(device=device)
        self.ff_loss = FocalFrequencyLoss(device=device)

    def forward(self, input, target):
        l1 = self.l1_loss(input, target)
        vgg = self.vgg_loss(input, target)
        ff = self.ff_loss(input, target)
        return l1 + self.lambda_vgg * vgg + self.lambda_ff * ff

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
        return loss
