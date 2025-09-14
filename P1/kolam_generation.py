import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image

# Generator
class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_shape=(1, 64, 64)):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, int(torch.prod(torch.tensor(img_shape)))),
            nn.Tanh()
        )
        self.img_shape = img_shape

    def forward(self, z):
        img = self.model(z)
        return img.view(img.size(0), *self.img_shape)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, img_shape=(1, 64, 64)):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(int(torch.prod(torch.tensor(img_shape))), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.model(img)

# Settings
latent_dim = 100
img_shape = (1, 64, 64)  # grayscale Kolam images resized
batch_size = 64
epochs = 100

# Data
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize(img_shape[1:]),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = datasets.ImageFolder("kolam_images", transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
generator = Generator(latent_dim, img_shape).to(device)
discriminator = Discriminator(img_shape).to(device)

optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
adversarial_loss = nn.BCELoss()

# Training loop
for epoch in range(epochs):
    for imgs, _ in dataloader:
        real = imgs.to(device)
        batch_size = real.size(0)

        # Train Discriminator
        z = torch.randn(batch_size, latent_dim).to(device)
        fake = generator(z).detach()
        real_loss = adversarial_loss(discriminator(real), torch.ones(batch_size, 1).to(device))
        fake_loss = adversarial_loss(discriminator(fake), torch.zeros(batch_size, 1).to(device))
        d_loss = (real_loss + fake_loss) / 2
        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        # Train Generator
        z = torch.randn(batch_size, latent_dim).to(device)
        gen_imgs = generator(z)
        g_loss = adversarial_loss(discriminator(gen_imgs), torch.ones(batch_size, 1).to(device))
        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

    print(f"[Epoch {epoch}/{epochs}] D_loss: {d_loss.item()} G_loss: {g_loss.item()}")
    if epoch % 10 == 0:
        save_image(gen_imgs[:25], f"outputs/kolam_{epoch}.png", nrow=5, normalize=True)
