import os
import zipfile
import cv2
import pandas as pd
from tqdm import tqdm
import numpy as np

IMAGE_SIZE = 256
RESIZE_INTERPOLATION = cv2.INTER_AREA
OUTPUT_DIR = '/kaggle/working/split_dataset/'

EXTRACT_PATH1 = '/kaggle/input/ti-6al-4v-imgs/ArunBaskaran Image-Driven-Machine-Learning-Approach-for-Microstructure-Classification-and-Segmentation-Ti-6Al-4V master Images1'
EXTRACT_PATH2 = '/kaggle/input/ti-6al-4v-imgs/ArunBaskaran Image-Driven-Machine-Learning-Approach-for-Microstructure-Classification-and-Segmentation-Ti-6Al-4V master Images2'

LABELS_CSV = r'/kaggle/input/labelling-of-microstruct/labels.xlsx'

os.makedirs(OUTPUT_DIR, exist_ok=True)

labels_df = pd.read_excel(LABELS_CSV, header=None)
labels_df.columns = ['image_name', 'label']
labels_dict = dict(zip(labels_df['image_name'], labels_df['label']))

### Albumentations augmentation pipeline

augmenter = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomBrightnessContrast(p=0.3),
    A.GaussNoise(p=0.2)
])

ERROR_LOG = os.path.join(OUTPUT_DIR, "error_log.txt")

def log_error(msg):
    with open(ERROR_LOG, 'a') as f:
        f.write(msg + '\n')

def preprocess_and_split(image_path, image_name, label):
    img = cv2.imread(image_path)
    if img is None:
        log_error(f"❌ Could not read image: {image_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))
    enhanced = clahe.apply(gray)

    enhanced_3ch = cv2.merge([enhanced]*3)

    h, w, _ = enhanced_3ch.shape
    left = enhanced_3ch[:, :w // 2]
    right = enhanced_3ch[:, w // 2:]

    left = cv2.resize(left, (IMAGE_SIZE, IMAGE_SIZE), interpolation=RESIZE_INTERPOLATION)
    right = cv2.resize(right, (IMAGE_SIZE, IMAGE_SIZE), interpolation=RESIZE_INTERPOLATION)

    # Normalize 
    left = cv2.normalize(left, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    right = cv2.normalize(right, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    for idx, crop in enumerate([left, right]):
        # augmentations (convert to RGB)
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        augmented = augmenter(image=crop_rgb)['image']
        # back to BGR for cv2.imwrite
        final_img = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)

        sub = 'a' if idx == 0 else 'b'
        fname = f"{image_name}_{sub}.png"
        label_dir = os.path.join(OUTPUT_DIR, str(label))
        os.makedirs(label_dir, exist_ok=True)

        # png saving compression lvl 3
        save_path = os.path.join(label_dir, fname)
        success = cv2.imwrite(save_path, final_img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        if not success:
            log_error(f"❌ Failed to save image: {save_path}")

# Validate keys (label)
image_files1 = [f for f in os.listdir(EXTRACT_PATH1) if f.lower().endswith(('.png','.tif','.jpg','.jpeg'))]
image_files2 = [f for f in os.listdir(EXTRACT_PATH2) if f.lower().endswith(('.png','.tif','.jpg','.jpeg'))]

all_images = image_files1 + image_files2

# Parse image ids 
def extract_image_id(filename):
    try:
        base = filename.split('.')[0]
        parts = base.split('_')
        return int(parts[-1])
    except Exception:
        return None

unmatched = []
for fname in all_images:
    img_id = extract_image_id(fname)
    if img_id is None or img_id not in labels_dict:
        unmatched.append(fname)

if unmatched:
    print(f"⚠️ Unmatched images (no label found or bad format): {len(unmatched)}")
    for u in unmatched[:10]:
        print(" -", u)
for folder_path in [EXTRACT_PATH1, EXTRACT_PATH2]:
    print(f"\nProcessing folder: {folder_path}")
    for img_name in tqdm(sorted(os.listdir(folder_path))):
        if not img_name.lower().endswith(('.png','.tif','.jpg','.jpeg')):
            continue

        image_id = extract_image_id(img_name)
        if image_id is None:
            log_error(f"❌ Invalid image name format: {img_name}")
            continue

        label = labels_dict.get(image_id, None)
        if label is None:
            log_error(f"⚠️ Skipping (label missing): {img_name}")
            continue

        img_path = os.path.join(folder_path, img_name)
        preprocess_and_split(img_path, image_id, label)

print(list(labels_dict.items())[:5])  
from IPython.display import Image, display
import random
import os

sample_paths = []
for root, _, files in os.walk(OUTPUT_DIR):
    for f in files:
        if f.lower().endswith(".png"):
            sample_paths.append(os.path.join(root, f))

# Display 5 random pics
num_samples = min(5, len(sample_paths))
for path in random.sample(sample_paths, num_samples):
    display(Image(filename=path))

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import glob
from tqdm import tqdm

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 8
IMAGE_SIZE = 256
NUM_CLASSES = 4  
EPOCHS = 10
LEARNING_RATE = 2e-4
TIMESTEPS = 1000

DATA_DIR = '/kaggle/working/split_dataset/'
CHECKPOINT_DIR = './checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
# Dataset 
class MicrostructureDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.samples = []
        for label_name in os.listdir(root_dir):
            label_path = os.path.join(root_dir, label_name)
            if not os.path.isdir(label_path):
                continue
            label = int(label_name)  
            for img_path in glob.glob(os.path.join(label_path, '*.png')):
                self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

# Transforms 
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),  
])
# Noise Scheduler 
class GaussianNoiseScheduler:
    def __init__(self, timesteps=1000):
        self.timesteps = timesteps
        self.betas = torch.linspace(1e-4, 0.02, timesteps).to(DEVICE)  
        self.alphas = 1. - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha_cumprod = self.alpha_cumprod[t].sqrt().view(-1, 1, 1, 1)
        sqrt_one_minus_alpha = (1 - self.alpha_cumprod[t]).sqrt().view(-1, 1, 1, 1)
        return sqrt_alpha_cumprod * x_start + sqrt_one_minus_alpha * noise
import torch
import torch.nn as nn
import torch.nn.functional as F

TIMESTEPS = 1000  

# UNet with One-Hot Label Conditioning 
class UNet2D(nn.Module):
    def __init__(self, num_classes, base_channels=64, timesteps=1000):
        super().__init__()
        self.num_classes = num_classes
        self.base_channels = base_channels

        # Embed timestep 
        self.time_embed = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, base_channels)
        )

        # Label embedding
        self.label_emb = nn.Sequential(
            nn.Linear(num_classes, base_channels),
            nn.SiLU(),
            nn.Linear(base_channels, 3)  # Projecting 3 channels
        )

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, base_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1),
            nn.ReLU()
        )
        self.enc3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels * 4, base_channels * 4, 3, padding=1),
            nn.ReLU()
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels * 4, base_channels * 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels * 8, base_channels * 8, 3, padding=1),
            nn.ReLU()
        )

        # Decoder
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(base_channels * 8, base_channels * 4, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels * 4, base_channels * 4, 3, padding=1),
            nn.ReLU()
        )
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1),
            nn.ReLU()
        )
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU()
        )

        self.final = nn.Conv2d(base_channels, 3, 1)  

    def forward(self, x, t, labels):

        # Time embedding
        t = t.float().unsqueeze(-1) / TIMESTEPS  
        t_emb = self.time_embed(t)  

        # Label conditioning
        labels_onehot = F.one_hot(labels, num_classes=self.num_classes).float()  
        label_emb = self.label_emb(labels_onehot)  

        # Injecting label 
        label_emb = label_emb.unsqueeze(-1).unsqueeze(-1) 
        x = x + label_emb  

        # Encode
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b = self.bottleneck(e3)

        # Decode
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = self.final(d1)
        return out

# Helper Func
def get_noise_schedule(timesteps=1000):
    return GaussianNoiseScheduler(timesteps=timesteps)

def sample_timesteps(batch_size, timesteps):
    return torch.randint(0, timesteps, (batch_size,), device=DEVICE)

# Sampling Gen
@torch.no_grad()
def generate_samples(model, noise_scheduler, label, num_samples=4):
    model.eval()
    label = torch.tensor([label] * num_samples, device=DEVICE)
    imgs = torch.randn(num_samples, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)  

    for t in reversed(range(TIMESTEPS)):
        t_batch = torch.full((num_samples,), t, device=DEVICE, dtype=torch.long)
        predicted_noise = model(imgs, t_batch, label)
        alpha_t = noise_scheduler.alpha_cumprod[t]
        alpha_prev = noise_scheduler.alpha_cumprod[t-1] if t > 0 else torch.tensor(1.0, device=DEVICE)

        x0_pred = (imgs - (1 - alpha_t).sqrt() * predicted_noise) / alpha_t.sqrt()

        mean = alpha_prev.sqrt() * x0_pred + (1 - alpha_prev).sqrt() * predicted_noise

        # Adding noise 
        if t > 0:
            noise = torch.randn_like(imgs)
        else:
            noise = torch.zeros_like(imgs)

        imgs = mean + noise * noise_scheduler.betas[t].sqrt()

    imgs = torch.clamp(imgs, 0, 1)
    return imgs.cpu()


### Training Loop

def train():
    dataset = MicrostructureDataset(DATA_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

    model = UNet2D(NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    noise_scheduler = get_noise_schedule(TIMESTEPS)

    model.train()
    for epoch in range(EPOCHS):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for imgs, labels in pbar:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            batch_size = imgs.size(0)

            t = sample_timesteps(batch_size, TIMESTEPS)
            noise = torch.randn_like(imgs)
            noisy_imgs = noise_scheduler.q_sample(imgs, t, noise)

            preds = model(noisy_imgs, t, labels)
            loss = F.mse_loss(preds, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=loss.item())

        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f'model_epoch_{epoch+1}.pth'))

if __name__ == "__main__":
    train()


