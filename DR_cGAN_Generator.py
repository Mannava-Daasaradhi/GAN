# DR_cGAN_Generator.py

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
import torchvision.utils as vutils
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# =============== CONFIGURATION AND PATHS ===============
# ---- Paths ----
DATA_DIR = "./data/train_images"
LABELS_CSV = "./data/train.csv"
# This OUT_DIR MUST be the same one used by the classifier script
OUT_DIR = "./DR_Phase4_MultiTask"
os.makedirs(OUT_DIR, exist_ok=True)
SYN_DIR = os.path.join(OUT_DIR, "synthetic_cgan")
os.makedirs(SYN_DIR, exist_ok=True)
RESULTS_DIR = os.path.join(OUT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---- Hyperparameters ----
# Using a smaller image size for GAN training is common as they are hard to train
IMG_SIZE = 128
LATENT_DIM = 100
COND_DIM = 2
BATCH_SIZE = 64
EPOCHS_CGAN = 100 # Increase for better quality, e.g., to 200
LR = 0.0002
BETA1 = 0.5 # Recommended for Adam optimizer in GANs
N_SYN = 1500 # Number of synthetic images to generate post-training

# ---- GPU Setup ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =============== 1. DATA LOADING FOR GAN ===============
# We only need the images WITH Diabetic Retinopathy to train the GAN
df = pd.read_csv(LABELS_CSV)
dr_ids = df[df['diagnosis'] > 0]['id_code'].values

# Normalize images to [-1, 1] for tanh activation
gan_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

class RetinopathyDataset(Dataset):
    def __init__(self, ids, directory, transform=None):
        self.ids = ids
        self.directory = directory
        self.transform = transform
        # Simulate risk factors for the real images
        self.hba1c_risks = 6.0 + 8.0 * np.random.beta(5, 2, size=len(ids))
        self.duration_risks = 2.0 + 28.0 * np.random.beta(4, 3, size=len(ids))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_name = f"{self.ids[idx]}.png"
        img_path = os.path.join(self.directory, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        # Normalize risk factors to be used as conditions
        hba1c_norm = self.hba1c_risks[idx] / 15.0
        duration_norm = self.duration_risks[idx] / 30.0
        condition = torch.tensor([hba1c_norm, duration_norm], dtype=torch.float32)
        
        return image, condition

dr_dataset = RetinopathyDataset(dr_ids, DATA_DIR, transform=gan_transform)
dr_loader = DataLoader(dr_dataset, batch_size=BATCH_SIZE, shuffle=True)
print(f"Loaded {len(dr_dataset)} images with DR for cGAN training.")

# =============== 2. CGAN MODEL ARCHITECTURE ===============
# Custom weights initialization
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Input is Z (latent) + C (condition), going into a convolution
            nn.ConvTranspose2d(LATENT_DIM + COND_DIM, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # state size. 512 x 4 x 4
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # state size. 256 x 8 x 8
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # state size. 128 x 16 x 16
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # state size. 64 x 32 x 32
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            # state size. 32 x 64 x 64
            nn.ConvTranspose2d(32, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # final state size. 3 x 128 x 128
        )
    def forward(self, noise, cond):
        inp = torch.cat((noise, cond), dim=1)
        # Reshape for convolutional layers
        inp = inp.view(inp.size(0), -1, 1, 1)
        return self.main(inp)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # Image processing path
        self.img_path = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # Final combined path
        self.final_path = nn.Sequential(
            nn.Conv2d(512 + COND_DIM, 1, 4, 1, 0, bias=False),
        )

    def forward(self, img, cond):
        img_features = self.img_path(img)
        # Replicate condition to match spatial dimensions
        cond_rep = cond.view(cond.size(0), -1, 1, 1).expand(-1, -1, img_features.size(2), img_features.size(3))
        # Concatenate along the channel dimension
        x = torch.cat((img_features, cond_rep), dim=1)
        return self.final_path(x)


# =============== 3. CGAN TRAINING ===============
netG = Generator().to(device); netG.apply(weights_init)
netD = Discriminator().to(device); netD.apply(weights_init)

criterion = nn.BCEWithLogitsLoss()
optimizerD = optim.Adam(netD.parameters(), lr=LR, betas=(BETA1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=LR, betas=(BETA1, 0.999))

# Fixed noise and conditions for visualizing generator progress
fixed_noise = torch.randn(4, LATENT_DIM, device=device)
fixed_conds_raw = torch.tensor([[8/15, 5/30], [8/15, 25/30], [13/15, 5/30], [13/15, 25/30]], dtype=torch.float32, device=device)
fixed_conds_labels = ["H:8 D:5", "H:8 D:25", "H:13 D:5", "H:13 D:25"]


print("Starting Training Loop...")
for epoch in range(1, EPOCHS_CGAN + 1):
    for i, (real_imgs, real_conds) in enumerate(dr_loader):
        real_imgs, real_conds = real_imgs.to(device), real_conds.to(device)
        b_size = real_imgs.size(0)
        
        # ---- Train Discriminator ----
        netD.zero_grad()
        # Real batch
        label = torch.full((b_size,), 1.0, dtype=torch.float, device=device)
        output = netD(real_imgs, real_conds).view(-1)
        errD_real = criterion(output, label)
        errD_real.backward()
        # Fake batch
        noise = torch.randn(b_size, LATENT_DIM, device=device)
        fake_imgs = netG(noise, real_conds) # Use same conditions as real batch
        label.fill_(0.0)
        output = netD(fake_imgs.detach(), real_conds).view(-1)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        errD = errD_real + errD_fake
        optimizerD.step()

        # ---- Train Generator ----
        netG.zero_grad()
        label.fill_(1.0) # Generator wants discriminator to think fake is real
        output = netD(fake_imgs, real_conds).view(-1)
        errG = criterion(output, label)
        errG.backward()
        optimizerG.step()

    print(f'[{epoch}/{EPOCHS_CGAN}] Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f}')

    # Save a grid of generated images to check progress
    if epoch % 10 == 0:
        with torch.no_grad():
            fake_samples = netG(fixed_noise, fixed_conds_raw).detach().cpu()
        grid = vutils.make_grid(fake_samples, padding=2, normalize=True)
        plt.figure(figsize=(6, 6))
        plt.imshow(np.transpose(grid, (1, 2, 0)))
        plt.title(f"Generated Images at Epoch {epoch}\n{fixed_conds_labels}")
        plt.axis("off")
        plt.savefig(os.path.join(RESULTS_DIR, f"cgan_progress_epoch_{epoch}.png"))
        plt.close()

print("✅ CGAN training complete.")

# Save the generator model
generator_path = os.path.join(OUT_DIR, "cgan_generator.pth")
torch.save(netG.state_dict(), generator_path)
print(f"Generator model saved to {generator_path}")

# =============== 4. GENERATE SYNTHETIC DATASET ===============
print(f"Generating {N_SYN} synthetic images...")
netG.eval()
synthetic_index = []
with torch.no_grad():
    for i in tqdm(range(N_SYN), desc="Generating Images"):
        # Create random noise and a new random condition
        noise = torch.randn(1, LATENT_DIM, device=device)
        hba1c = 6.0 + 8.0 * np.random.beta(5, 2)
        duration = 2.0 + 28.0 * np.random.beta(4, 3)
        hba1c_norm = hba1c / 15.0
        duration_norm = duration / 30.0
        condition = torch.tensor([[hba1c_norm, duration_norm]], dtype=torch.float32, device=device)

        # Generate image
        fake_img = netG(noise, condition).detach().cpu()
        
        # Save image
        fname = f"syn_{i:05d}.png"
        vutils.save_image(fake_img, os.path.join(SYN_DIR, fname), normalize=True)
        
        # Record metadata
        synthetic_index.append((fname, hba1c, duration))

# Save the index file
syn_df = pd.DataFrame(synthetic_index, columns=["filename", "HbA1c", "Duration"])
syn_df.to_csv(os.path.join(OUT_DIR, "synthetic_index.csv"), index=False)

print(f"✅ Synthetic dataset saved to {SYN_DIR} and index saved to {OUT_DIR}.")
print("\nYou can now run the classifier script.")