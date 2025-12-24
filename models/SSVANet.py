# models/mambair.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return x1 * x2

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False)
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False)
        self.conv4 = nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv5 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.conv_out = nn.Conv2d(out_channels * 5, out_channels, 1, bias=False)
        self.norm = nn.LayerNorm(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        size = x.shape[-2:]
        feat1 = self.conv1(x)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = F.interpolate(self.conv5(self.pool(x)), size=size, mode='bilinear', align_corners=False)
        out = torch.cat((feat1, feat2, feat3, feat4, feat5), dim=1)
        out = self.conv_out(out)
        out = out.permute(0, 2, 3, 1)  # Change to (B, H, W, C)
        out = self.norm(out)
        out = out.permute(0, 3, 1, 2)  # Change back to (B, C, H, W)
        return self.act(out)

class CWSA(nn.Module):
    def __init__(self, dim):
        super(CWSA, self).__init__()
        self.dim = dim
        self.query_conv = nn.Linear(dim, dim)
        self.key_conv = nn.Linear(dim, dim)
        self.value_conv = nn.Linear(dim, dim)
        self.scale = dim ** -0.5
        self.pos_embedding = nn.Parameter(torch.randn(1, 1, 1, dim))

    def forward(self, x):
        # x: (B, H, W, C)
        B, H, W, C = x.shape
        x = x + self.pos_embedding  # Positional embedding
        x = x.view(B, H * W, C)     # Reshape to (B, N, C)

        # Linear projections
        q = self.query_conv(x)      # (B, N, C)
        k = self.key_conv(x)        # (B, N, C)
        v = self.value_conv(x)      # (B, N, C)

        # Compute attention over channels at each spatial location
        q = q.view(B, H * W, 1, C)  # (B, N, 1, C)
        k = k.view(B, H * W, C, 1)  # (B, N, C, 1)
        attn = torch.matmul(q, k).squeeze(2) * self.scale  # (B, N, C)
        attn = attn.softmax(dim=-1)  # Softmax over channels

        # Apply attention to values
        out = attn * v               # Element-wise multiplication
        out = out.view(B, H, W, C)   # Reshape back to (B, H, W, C)
        return out

class SSVAModule(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=3, expand=2.0, dt_rank=64, dt_min=0.001, dt_max=0.1, dt_init="random", dt_scale=1.0):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)  # self.d_inner = 2 * d_model
        self.dt_rank = dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2)
        self.conv2d = nn.Conv2d(self.d_inner, self.d_inner, kernel_size=d_conv, padding=(d_conv - 1) // 2, groups=self.d_inner)
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(self.d_inner, self.d_inner * 2)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner)

        self.out_norm = nn.LayerNorm(self.d_inner)

        self.out_proj = nn.Linear(self.d_inner // 2, d_model)

        # New components
        self.simple_gate = SimpleGate()
        self.aspp = ASPP(d_model, d_model)
        self.channel_attn = CWSA(d_model)

    def forward(self, x):
        B, H, W, C = x.shape

        # Apply ASPP
        x_aspp = self.aspp(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        # Original SS2D operations
        x = self.in_proj(x)
        x, z = x.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2)
        x = self.conv2d(x)
        x = x.permute(0, 2, 3, 1)
        x = self.act(x)
        y = self.selective_scan(x)
        y = self.out_norm(y)
        y = y * F.silu(z)

        # Apply SimpleGate
        y = self.simple_gate(y)

        # Apply Channel-wise Self-Attention
        y = self.channel_attn(y)

        # Combine with ASPP output
        y = y + x_aspp

        out = self.out_proj(y)
        return out

    def selective_scan(self, x):
        B, H, W, C = x.shape
        x_flat = x.reshape(B, H * W, C)
        x_dbl = self.x_proj(x_flat)
        x_dbl = x_dbl.view(B, H, W, -1)
        dt, x_proj = x_dbl.chunk(2, dim=-1)
        dt = F.softplus(self.dt_proj(dt))
        y = x * torch.sigmoid(dt) + x_proj * torch.tanh(x_proj)
        return y

class SSVABlock(nn.Module):
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.ss2d = SSVAModule(d_model, d_state)
        self.ln_2 = nn.LayerNorm(d_model)
        self.conv_blk = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1)
        )

    def forward(self, x):
        residual = x
        x = self.ln_1(x)
        x = residual + self.ss2d(x)
        residual = x
        x = self.ln_2(x)
        x = x.permute(0, 3, 1, 2)
        x = self.conv_blk(x)
        x = x.permute(0, 2, 3, 1)
        x = residual + x
        return x

class SSVA_Net(nn.Module):
    def __init__(self, img_channel=3, width=32, middle_blk_num=1, enc_blk_nums=[1, 1, 1, 1], dec_blk_nums=[1, 1, 1, 1], d_state=64):
        super().__init__()
        self.intro = nn.Conv2d(img_channel, width, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        self.ending = nn.Conv2d(width, img_channel, kernel_size=3, padding=1, stride=1, groups=1, bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(*[SSVABlock(chan, d_state) for _ in range(num)])
            )
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan = chan * 2

        self.middle_blks = nn.Sequential(
            *[SSVABlock(chan, d_state) for _ in range(middle_blk_num)]
        )

        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(
                nn.Conv2d(chan, chan * 2, 1, bias=False),
                nn.PixelShuffle(2)
            ))
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(*[SSVABlock(chan, d_state) for _ in range(num)])
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)
        x = self.intro(inp)
        x = x.permute(0, 2, 3, 1)

        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = x.permute(0, 3, 1, 2)
            x = down(x)
            x = x.permute(0, 2, 3, 1)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = x.permute(0, 3, 1, 2)
            x = up(x)
            x = x.permute(0, 2, 3, 1)
            x = x + enc_skip
            x = decoder(x)

        x = x.permute(0, 3, 1, 2)
        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x
