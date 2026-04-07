```python
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from pathlib import Path
from torchvision import transforms
from PIL import Image
import argparse
import requests
from tqdm import tqdm

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=time.device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.dim = channels // 8
        self.q_conv = nn.Conv2d(channels, self.dim, 1)
        self.k_conv = nn.Conv2d(channels, self.dim, 1)
        self.v_conv = nn.Conv2d(channels, channels, 1)
        self.out_conv = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        q = self.q_conv(x).view(b, self.dim, -1)
        k = self.k_conv(x).view(b, self.dim, -1)
        v = self.v_conv(x).view(b, c, -1)
        attn = torch.softmax(torch.bmm(q.transpose(1, 2), k) / (self.dim ** 0.5), dim=-1)
        out = torch.bmm(v, attn.transpose(1, 2)).view(b, c, h, w)
        out = self.out_conv(out)
        return x + out

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, class_emb_dim, up=False, down=False, use_attention=False):
        super().__init__()
        self.use_attention = use_attention
        self.time_mlp = nn.Linear(time_emb_dim, out_ch) if time_emb_dim else None
        self.class_mlp = nn.Linear(class_emb_dim, out_ch) if class_emb_dim else None
        if up:
            self.conv1 = nn.Conv2d(2*in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        elif down:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Identity()

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU()

        if use_attention:
            self.attention = SelfAttention(out_ch)
        else:
            self.attention = nn.Identity()

    def forward(self, x, t=None, class_labels=None):
        h = self.act(self.bnorm1(self.conv1(x)))
        if self.time_mlp and t is not None:
            h = h + self.time_mlp(t)[:, :, None, None]
        if self.class_mlp and class_labels is not None:
            h = h + self.class_mlp(class_labels)[:, :, None, None]
        h = self.act(self.bnorm2(self.conv2(h)))
        h = self.attention(h)
        return self.transform(h)

class SimpleUNet(nn.Module):
    def __init__(self, image_channels=3, time_emb_dim=128, class_emb_dim=10, num_classes=10):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )
        self.class_mlp = nn.Embedding(num_classes, class_emb_dim)

        self.down1 = Block(image_channels, 64, time_emb_dim, class_emb_dim, down=True, use_attention=False)
        self.down2 = Block(64, 128, time_emb_dim, class_emb_dim, down=True, use_attention=True)
        self.down3 = Block(128, 256, time_emb_dim, class_emb_dim, down=True, use_attention=False)
        self.bottleneck = Block(256, 256, time_emb_dim, class_emb_dim, use_attention=True)
        self.up3 = Block(256, 128, time_emb_dim, class_emb_dim, up=True, use_attention=False)
        self.up2 = Block(128, 64, time_emb_dim, class_emb_dim, up=True, use_attention=True)
        self.up1 = Block(64, 64, time_emb_dim, class_emb_dim, up=True, use_attention=False)
        self.out = nn.Conv2d(64, image_channels, 1)

    def forward(self, x, t, class_labels):
        t_emb = self.time_mlp(t)
        c_emb = self.class_mlp(class_labels)
        d1 = self.down1(x, t_emb, c_emb)
        d2 = self.down2(d1, t_emb, c_emb)
        d3 = self.down3(d2, t_emb, c_emb)
        b = self.bottleneck(d3, t_emb, c_emb)
        u3 = self.up3(torch.cat([b, d3], dim=1), t_emb, c_emb)
        u2 = self.up2(torch.cat([u3, d2], dim=1), t_emb, c_emb)
        u1 = self.up1(torch.cat([u2, d1], dim=1), t_emb, c_emb)
        return self.out(u1)

class NoiseScheduler:
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02, schedule='linear'):
        self.num_timesteps = num_timesteps
        if schedule == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif schedule == 'cosine':
            s = 0.008
            steps = num_timesteps + 1
            x = torch.linspace(0, num_timesteps, steps)
            alphas_cumprod = torch.cos(((x / num_timesteps) + s) / (1 + s) * math.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clip(betas, 0.0001, 0.9999)
            self.betas = betas
        else:
            raise ValueError(f'Unknown schedule {schedule}')
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)

    def p_sample(self, model, xt, t, classes, clip_denoised=True):
        t_batch = torch.full((xt.shape[0],), t, device=xt.device, dtype=torch.long)
        predicted_noise = model(xt, t_batch, classes)
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1,1,1,1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1,1,1,1)
        pred_x0 = (xt - sqrt_one_minus_alpha_cumprod_t * predicted_noise) / sqrt_alpha_cumprod_t
        if clip_denoised:
            pred_x0 = torch.clamp(pred_x0, -1., 1.)
        mean = self.posterior_mean_coef1[t].view(-1,1,1,1) * pred_x0 + \
               self.posterior_mean_coef2[t].view(-1,1,1,1) * xt
        variance = self.posterior_variance[t]
        if t == 0:
            return mean
        else:
            noise = torch.randn_like(xt)
            return mean + torch.sqrt(variance).view(-1,1,1,1) * noise

    def sample(self, model, num_samples, classes, img_shape=(3,32,32), device='cuda', clip_denoised=True):
        model.eval()
        with torch.no_grad():
            xt = torch.randn((num_samples,) + img_shape, device=device)
            for t in tqdm(reversed(range(self.num_timesteps)), desc='Sampling'):
                xt = self.p_sample(model, xt, t, classes, clip_denoised)
            return xt

    def to(self, device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        self.posterior_mean_coef1 = self.posterior_mean_coef1.to(device)
        self.posterior_mean_coef2 = self.posterior_mean_coef2.to(device)
        return self

def download_model(url, dest):
    if not os.path.exists(dest):
        print(f'Downloading model from {url} ...')
        response = requests.get(url, stream=True)
        with open(dest, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print('Model downloaded.')

def main():
    parser = argparse.ArgumentParser(description='Generate SVHN digits with diffusion model')
    parser.add_argument('--class', dest='class_id', type=int, required=True, choices=range(10),
                        help='Digit class to generate (0-9)')
    parser.add_argument('--num', type=int, default=10, help='Number of images to generate')
    parser.add_argument('--out_dir', type=str, default='generated', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device (cuda/cpu)')
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_url = 'https://github.com/David-Z-ai/svhn-digits-generation/raw/main/unet_ep120_bs64_t1000.pth'
    model_path = 'unet_ep120_bs64_t1000.pth'
    download_model(model_url, model_path)

    model = SimpleUNet(image_channels=3, time_emb_dim=128, class_emb_dim=10, num_classes=10).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    noise_scheduler = NoiseScheduler(num_timesteps=1000, schedule='linear')
    noise_scheduler.to(device)

    classes = torch.full((args.num,), args.class_id, device=device, dtype=torch.long)
    print(f'Generating {args.num} images of class {args.class_id} ...')
    images = noise_scheduler.sample(model, args.num, classes, img_shape=(3,32,32), device=device, clip_denoised=True)
    images = (images + 1) / 2
    images = torch.clamp(images, 0, 1)

    for idx, img_tensor in enumerate(images):
        img_pil = transforms.ToPILImage()(img_tensor.cpu())
        filename = out_dir / f'class_{args.class_id}_{idx:04d}.png'
        img_pil.save(filename)
        print(f'Saved: {filename}')

    print('Done.')

if __name__ == '__main__':
    main()
