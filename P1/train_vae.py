import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

# --- Configuration ---
DATA_DIR = "kolam_images"
OUTPUT_DIR = "vae_output_sharp"
MODEL_PATH = os.path.join(OUTPUT_DIR, "vae_kolam_sharp.pth")
IMG_SIZE = 64
LATENT_DIM = 64
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-4
BETA = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model and Loss Function Definitions can stay outside the main block ---

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(256 * 4 * 4, LATENT_DIM)
        self.fc_logvar = nn.Linear(256 * 4 * 4, LATENT_DIM)

        # Decoder
        self.decoder_fc = nn.Linear(LATENT_DIM, 256 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (256, 4, 4)),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
            nn.Tanh()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(self.decoder_fc(z))
        return recon, mu, logvar

def loss_function(recon_x, x, mu, logvar, beta):
    MSE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + beta * KLD

# --- Main execution block ---
if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(IMG_SIZE),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
    # The DataLoader must be created inside this block
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    model = VAE().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"🚀 Starting training on {DEVICE} for {EPOCHS} epochs...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0
        for data, _ in dataloader:
            data = data.to(DEVICE)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar, BETA)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        
        avg_loss = train_loss / len(dataset.samples)
        print(f"====> Epoch: {epoch} Average loss: {avg_loss:.4f}")

        if epoch % 50 == 0:
            with torch.no_grad():
                fixed_noise = torch.randn(16, LATENT_DIM).to(DEVICE)
                generated_samples = model.decoder(model.decoder_fc(fixed_noise)).cpu()
                save_image(generated_samples.view(16, 1, IMG_SIZE, IMG_SIZE),
                           os.path.join(OUTPUT_DIR, f'sample_epoch_{epoch}.png'),
                           normalize=True)

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"💾 Model saved to {MODEL_PATH}")

    print("🎨 Generating final image sheet...")
    model.eval()
    with torch.no_grad():
        random_z = torch.randn(64, LATENT_DIM).to(DEVICE)
        final_samples = model.decoder(model.decoder_fc(random_z)).cpu()
        save_image(final_samples, os.path.join(OUTPUT_DIR, 'final_generation.png'), normalize=True)

    print(f"✨ Generation complete! Check the '{OUTPUT_DIR}' folder.")