# generate_kolam.py
import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, utils
import torch.nn.functional as F

# Config (match vae_kolam.py)
DATA_DIR = "kolam_images"
OUTPUT_DIR = "Creation"
MODEL_PATH = os.path.join(OUTPUT_DIR, "vae_kolam.pth")
IMG_SIZE = 128
LATENT_DIM = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model definition (must match the model used during training)
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(True),
            nn.Flatten()
        )
        conv_out = 256 * (IMG_SIZE // 16) * (IMG_SIZE // 16)
        self.fc_mu = nn.Linear(conv_out, latent_dim)
        self.fc_logvar = nn.Linear(conv_out, latent_dim)

    def forward(self, x):
        x = self.conv(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        conv_out = 256 * (IMG_SIZE // 16) * (IMG_SIZE // 16)
        self.fc = nn.Linear(latent_dim, conv_out)
        self.deconv = nn.Sequential(
            nn.Unflatten(1, (256, IMG_SIZE // 16, IMG_SIZE // 16)),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.fc(z)
        x = self.deconv(x)
        return x

class VAE(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.enc = Encoder(latent_dim)
        self.dec = Decoder(latent_dim)

    def reparameterize(self, mu, logvar):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.enc(x)
        z = self.reparameterize(mu, logvar)
        recon = self.dec(z)
        return recon, mu, logvar

# Utility: load dataset (grayscale, same transform as training)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
class_names = dataset.classes

# Load model
model = VAE(LATENT_DIM).to(DEVICE)
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Compute latent stats (mu mean and covariance diagonal) per class
def compute_class_stats(dataloader, model, device):
    stats = {c: [] for c in class_names}
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            mu, logvar = model.enc(imgs)
            for i, lbl in enumerate(labels):
                stats[class_names[lbl]].append(mu[i].cpu())
    class_means = {}
    class_vars = {}
    for c, mus in stats.items():
        if len(mus) == 0:
            class_means[c] = torch.zeros(LATENT_DIM)
            class_vars[c] = torch.ones(LATENT_DIM)
            continue
        stack = torch.stack(mus)
        class_means[c] = stack.mean(dim=0)
        class_vars[c] = stack.var(dim=0) + 1e-6
    return class_means, class_vars

class_means, class_vars = compute_class_stats(dataloader, model, DEVICE)

# Sample functions
def sample_from_prior(n=16):
    z = torch.randn(n, LATENT_DIM).to(DEVICE)
    with torch.no_grad():
        gen = model.dec(z)
    return gen.cpu()

def sample_from_class(class_name, n=8):
    if class_name not in class_means:
        raise ValueError("Unknown class")
    mu = class_means[class_name].to(DEVICE)
    var = class_vars[class_name].to(DEVICE)
    z = mu.unsqueeze(0) + torch.randn(n, LATENT_DIM, device=DEVICE) * var.sqrt().unsqueeze(0)
    with torch.no_grad():
        gen = model.dec(z)
    return gen.cpu()

def interpolate_between_two_images(img_idx1, img_idx2, steps=8):
    # Encode images into mu
    img1, _ = dataset[img_idx1]
    img2, _ = dataset[img_idx2]
    img1 = img1.unsqueeze(0).to(DEVICE)
    img2 = img2.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        mu1, _ = model.enc(img1)
        mu2, _ = model.enc(img2)
    mus = []
    for alpha in torch.linspace(0, 1, steps):
        z = (1-alpha) * mu1 + alpha * mu2
        mus.append(z.squeeze(0))
    z_all = torch.stack(mus).to(DEVICE)
    with torch.no_grad():
        gen = model.dec(z_all)
    return gen.cpu()

# Save helpers
def save_grid(tensor_batch, filename, nrow=4):
    utils.save_image(utils.make_grid(tensor_batch, nrow=nrow, normalize=True, pad_value=1), filename)

# Main generation tasks
if __name__ == "__main__":
    # 1) Samples from prior
    prior_gen = sample_from_prior(n=16)
    save_grid(prior_gen, os.path.join(OUTPUT_DIR, "sample_prior.png"), nrow=4)

    # 2) Class-conditioned samples for each class
    for c in class_names:
        gen = sample_from_class(c, n=8)
        save_grid(gen, os.path.join(OUTPUT_DIR, f"sample_class_{c.replace(' ','_')}.png"), nrow=4)

    # 3) Interpolate between two dataset images (choose indices)
    if len(dataset) >= 2:
        interp = interpolate_between_two_images(0, min(15, len(dataset)-1), steps=12)
        save_grid(interp, os.path.join(OUTPUT_DIR, "interp_0_15.png"), nrow=6)

    print("Saved samples to", OUTPUT_DIR)
