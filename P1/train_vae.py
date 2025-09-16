import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

# --- Configuration ---
DATA_DIR = "kolam_dataset"  # Directory containing the 'images' subfolder
OUTPUT_DIR = "vae_output"
MODEL_PATH = os.path.join(OUTPUT_DIR, "vae_kolam.pth")
IMG_SIZE = 64
LATENT_DIM = 32  # Dimensionality of the learned "style" space
BATCH_SIZE = 32
EPOCHS = 100 # Increase for better results (e.g., 200-500)
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 1. VAE Model Definition ---
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), nn.ReLU(), # 64 -> 32
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(), # 32 -> 16
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(), # 16 -> 8
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(), # 8 -> 4
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(256 * 4 * 4, LATENT_DIM)
        self.fc_logvar = nn.Linear(256 * 4 * 4, LATENT_DIM)

        # Decoder
        self.decoder_fc = nn.Linear(LATENT_DIM, 256 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1), nn.Sigmoid() # Sigmoid for pixel values [0, 1]
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(self.decoder_fc(z).view(-1, 256, 4, 4))
        return recon, mu, logvar

# --- Loss Function ---
def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# --- 2. Data Loading and Preprocessing ---
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(IMG_SIZE),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(), # Converts to [0, 1] and (C, H, W)
])

dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- 3. Training ---
model = VAE().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f"🚀 Starting training on {DEVICE}...")
for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(dataloader):
        data = data.to(DEVICE)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    
    avg_loss = train_loss / len(dataset)
    print(f"====> Epoch: {epoch} Average loss: {avg_loss:.4f}")

    # Save a sample of reconstructed images every 10 epochs
    if epoch % 10 == 0:
        with torch.no_grad():
            recon_sample, _, _ = model(data[:8])
            comparison = torch.cat([data[:8], recon_sample])
            save_image(comparison.cpu(), os.path.join(OUTPUT_DIR, f'reconstruction_{epoch}.png'), nrow=8)

# --- 4. Save Model and Generate New Kolams ---
torch.save(model.state_dict(), MODEL_PATH)
print(f"💾 Model saved to {MODEL_PATH}")

print("🎨 Generating new Kolams from random noise...")
model.eval()
with torch.no_grad():
    # Sample from the latent space (a standard normal distribution)
    random_z = torch.randn(64, LATENT_DIM).to(DEVICE)
    generated_samples = model.decoder(model.decoder_fc(random_z).view(-1, 256, 4, 4)).cpu()
    save_image(generated_samples, os.path.join(OUTPUT_DIR, 'generated_kolams.png'), nrow=8)

print(f"✨ Generation complete! Check the '{OUTPUT_DIR}' folder.")