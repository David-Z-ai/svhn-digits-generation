import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torchvision import transforms
from PIL import Image
import argparse
import requests
from tqdm import tqdm

class VAEEncoder(nn.Module):
    def __init__(self, latent_dim=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.fc_mu = nn.Linear(64 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(64 * 4 * 4, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class VAEDecoder(nn.Module):
    def __init__(self, latent_dim=10):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 64 * 4 * 4)
        self.deconv1 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 64, 4, 4)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = torch.tanh(self.deconv3(x))
        return x

class VAE(nn.Module):
    def __init__(self, latent_dim=10):
        super().__init__()
        self.encoder = VAEEncoder(latent_dim)
        self.decoder = VAEDecoder(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

def download_model(url, dest):
    if not os.path.exists(dest):
        print(f'Downloading model from {url} ...')
        response = requests.get(url, stream=True)
        with open(dest, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print('Model downloaded.')

def generate_images(vae, num_samples, latent_dim, device):
    vae.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim, device=device)
        images = vae.decoder(z)
        images = (images + 1) / 2
        images = torch.clamp(images, 0, 1)
    return images

def main():
    parser = argparse.ArgumentParser(description='Generate random images with a pretrained VAE')
    parser.add_argument('--num', type=int, default=10, help='Number of images to generate')
    parser.add_argument('--out_dir', type=str, default='generated_vae', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device (cuda/cpu)')
    parser.add_argument('--latent_dim', type=int, default=10, help='Latent space dimension')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_url = 'https://github.com/David-Z-ai/svhn-digits-generation/raw/main/vae_ep30_bs64_latent10.pth'
    model_path = 'vae_model.pth'
    download_model(model_url, model_path)

    vae = VAE(latent_dim=args.latent_dim).to(device)
    state_dict = torch.load(model_path, map_location=device)
    vae.load_state_dict(state_dict)
    vae.eval()

    print(f'Generating {args.num} random images with latent_dim={args.latent_dim}...')
    images = generate_images(vae, args.num, args.latent_dim, device)

    for idx, img_tensor in enumerate(images):
        img_pil = transforms.ToPILImage()(img_tensor.cpu())
        filename = out_dir / f'vae_gen_{idx:04d}.png'
        img_pil.save(filename)
        print(f'Saved: {filename}')

    print('Done.')

if __name__ == '__main__':
    main()
