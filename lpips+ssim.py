import os
import cv2
import pandas as pd
from tqdm import tqdm
import numpy as np
import albumentations as A
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
import torchmetrics  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ssim_metric = torchmetrics.StructuralSimilarityIndexMeasure(data_range=2.0).to(device)
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

import os
import torch
import lpips
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# AlexNet backbone
print("Setting up [LPIPS] perceptual loss: trunk [alex], v[0.1], spatial [off]")
lpips_model = lpips.LPIPS(net='alex').to(device)
lpips_model.eval()
from pytorch_msssim import ssim as ssim_metric
import torch
import torch.nn.functional as F
import lpips
from typing import Tuple, Optional, List, Dict
from skimage.metrics import structural_similarity as ssim


def compute_ssim(pred_img: torch.Tensor, target_img: torch.Tensor) -> float:
   
    pred = pred_img.squeeze().permute(1, 2, 0).cpu().numpy()
    target = target_img.squeeze().permute(1, 2, 0).cpu().numpy()

    pred = (pred + 1) / 2
    target = (target + 1) / 2

    ssim_val = ssim(pred, target, channel_axis=-1, data_range=1.0)
    return ssim_val


def compute_lpips_and_saliency(pred_img: torch.Tensor, target_img: torch.Tensor) -> Tuple[float, float, np.ndarray]:
    pred_img = pred_img.clone().detach().to(device).requires_grad_(True)
    target_img = target_img.clone().detach().to(device)

    if pred_img.grad is not None:
        pred_img.grad.zero_()

    lpips_score = lpips_model(pred_img, target_img)
    lpips_score.sum().backward()

    # L2 channels for saliency
    saliency = pred_img.grad.detach()
    saliency = torch.norm(saliency, p=2, dim=1).squeeze(0)
    saliency -= saliency.min()
    saliency /= (saliency.max() + 1e-10)
    saliency = saliency.clamp(0, 1)

    ssim_score = compute_ssim(pred_img.detach(), target_img.detach())

    return lpips_score.item(), ssim_score, saliency.cpu().numpy()


def visualize_metrics(pred_img, target_img, saliency_map, lpips_score, ssim_score):
    def tensor_to_img(t):
        img = t.squeeze().permute(1, 2, 0).cpu().numpy()
        img = (img + 1) / 2
        return np.clip(img, 0, 1)

    pred_np = tensor_to_img(pred_img)
    target_np = tensor_to_img(target_img)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title("Predicted Image")
    plt.imshow(pred_np)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Target Image")
    plt.imshow(target_np)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title(f"Saliency\nLPIPS: {lpips_score:.4f}\nSSIM: {ssim_score:.4f}")
    plt.imshow(saliency_map, cmap='hot')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def compute_metrics_batch(pred_imgs: torch.Tensor, target_imgs: torch.Tensor) -> Tuple[List[float], List[float]]:
    lpips_scores = []
    ssim_scores = []
    with torch.no_grad():
        for i in range(pred_imgs.shape[0]):
            pred = pred_imgs[i:i+1].to(device)
            target = target_imgs[i:i+1].to(device)
            lpips_val = lpips_model(pred, target).item()
            ssim_val = compute_ssim(pred, target)
            lpips_scores.append(lpips_val)
            ssim_scores.append(ssim_val)
    return lpips_scores, ssim_scores


def plot_metric_bar(metric_dict: Dict[str, List[float]], metric_name: str, color: str = 'skyblue'):
    names = list(metric_dict.keys())
    scores = [np.mean(vals) for vals in metric_dict.values()]
    plt.figure(figsize=(10, 5))
    plt.bar(names, scores, color=color)
    plt.ylabel(f"Mean {metric_name}")
    plt.title(f"{metric_name} per Class/Texture")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_metric_heatmap(metric_dict: Dict[str, List[float]], metric_name: str, cmap='coolwarm'):
    names = list(metric_dict.keys())
    scores = [np.mean(vals) for vals in metric_dict.values()]
    plt.figure(figsize=(8, 6))
    sns.heatmap(np.array([scores]), annot=True, fmt=".3f",
                xticklabels=names, yticklabels=[metric_name], cmap=cmap)
    plt.title(f"{metric_name} Heatmap")
    plt.tight_layout()
    plt.show()


def main():
    H, W = 128, 128
    target_img = torch.clamp(torch.randn(1, 3, H, W), -1, 1).to(device)
    pred_img = torch.clamp(target_img + torch.randn_like(target_img) * 0.01, -1, 1)

    lpips_score, ssim_score, saliency_map = compute_lpips_and_saliency(pred_img, target_img)
    visualize_metrics(pred_img, target_img, saliency_map, lpips_score, ssim_score)

    lpips_per_class = {'alpha': [0.032, 0.041], 'beta': [0.025], 'lamellar': [0.045, 0.038]}
    ssim_per_class = {'alpha': [0.91, 0.93], 'beta': [0.95], 'lamellar': [0.88, 0.90]}

    lpips_per_texture = {'fine': [0.031], 'coarse': [0.042, 0.037], 'mixed': [0.029]}
    ssim_per_texture = {'fine': [0.94], 'coarse': [0.89, 0.91], 'mixed': [0.92]}

    # Bar Charts
    plot_metric_bar(lpips_per_class, "LPIPS")
    plot_metric_bar(ssim_per_class, "SSIM", color='green')

    plot_metric_bar(lpips_per_texture, "LPIPS")
    plot_metric_bar(ssim_per_texture, "SSIM", color='green')

    # Heatmap
    plot_metric_heatmap(lpips_per_class, "LPIPS")
    plot_metric_heatmap(ssim_per_class, "SSIM", cmap='Greens')

    plot_metric_heatmap(lpips_per_texture, "LPIPS")
    plot_metric_heatmap(ssim_per_texture, "SSIM", cmap='Greens')


if __name__ == "__main__":
    main()
from typing import Tuple, List
from torchmetrics.functional import structural_similarity_index_measure as ssim_fn


def compute_lpips_ssim_saliency_batch(
    pred_imgs: torch.Tensor,
    target_imgs: torch.Tensor
) -> Tuple[List[float], List[float], torch.Tensor]:
    pred_imgs = pred_imgs.clone().detach().to(device).requires_grad_(True)
    target_imgs = target_imgs.clone().detach().to(device)

    # LPIPS
    lpips_scores = lpips_model(pred_imgs, target_imgs).view(-1).tolist()

    # Saliency 
    lpips_model(pred_imgs, target_imgs).sum().backward()
    saliency = pred_imgs.grad.detach()
    saliency_maps = torch.norm(saliency, p=2, dim=1)

    saliency_maps = (saliency_maps - torch.amin(saliency_maps, dim=(1, 2), keepdim=True)) / \
                    (torch.amax(saliency_maps, dim=(1, 2), keepdim=True) + 1e-10)
    saliency_maps = saliency_maps.clamp(0, 1)

    # SSIM
    ssim_scores = []
    for i in range(pred_imgs.shape[0]):
        ssim_val = ssim_fn(pred_imgs[i].unsqueeze(0), target_imgs[i].unsqueeze(0))
        ssim_scores.append(ssim_val.item())

    return lpips_scores, ssim_scores, saliency_maps


def tensor_to_img(t: torch.Tensor) -> np.ndarray:
    img = t.squeeze().permute(1, 2, 0).cpu().numpy()
    return np.clip((img + 1) / 2, 0, 1)


def visualize_batch(
    preds: torch.Tensor,
    targets: torch.Tensor,
    saliencies: torch.Tensor,
    lpips_scores: List[float],
    ssim_scores: List[float],
    save_dir: Optional[str] = None,
    prefix: str = "_sample"
):
    preds = preds.detach().cpu()
    targets = targets.detach().cpu()
    saliencies = saliencies.detach().cpu()

    for i in range(preds.shape[0]):
        fig = plt.figure(figsize=(15, 5))
        titles = [
            'Predicted Image',
            'Target Image',
            f'Saliency\nLPIPS: {lpips_scores[i]:.4f}, SSIM: {ssim_scores[i]:.4f}'
        ]

        imgs = [
            (preds[i].permute(1, 2, 0).numpy() + 1) / 2,   
            (targets[i].permute(1, 2, 0).numpy() + 1) / 2, 
            saliencies[i].numpy()                         
        ]

        for j in range(3):
            plt.subplot(1, 3, j + 1)
            plt.title(titles[j])
            plt.imshow(imgs[j], cmap='hot' if j == 2 else None)
            plt.axis('off')

        plt.tight_layout()
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, f"{prefix}{i}.png")
            plt.savefig(path)
            print(f"Saved: {path}")
            plt.close()
        else:
            plt.show()



#  Eg Simu
if __name__ == "__main__":
    B, C, H, W = 4, 3, 128, 128
    save_dir = "./batch_lpips_vis"

    target_imgs = torch.randn(B, C, H, W).clamp(-1, 1)
    noise = torch.randn_like(target_imgs) * 0.02
    pred_imgs = (target_imgs + noise).clamp(-1, 1)

    lpips_scores, ssim_scores, saliency_maps = compute_lpips_ssim_saliency_batch(pred_imgs, target_imgs)
    visualize_batch(pred_imgs, target_imgs, saliency_maps, lpips_scores, ssim_scores, save_dir=save_dir)

print(f"Avg LPIPS: {np.mean(lpips_scores):.4f}")
print(f"Avg SSIM : {np.mean(ssim_scores):.4f}")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_similarity_heatmap(per_property_scores: dict, save_path: str = "similarity_heatmap.png"):

    properties = list(per_property_scores.keys())
    avg_lpips = [np.mean(per_property_scores[p]['lpips']) for p in properties]
    avg_ssim = [np.mean(per_property_scores[p]['ssim']) for p in properties]

    data = np.array([avg_lpips, avg_ssim])  
    plt.figure(figsize=(12, 4))
    sns.heatmap(
        data,
        annot=True,
        fmt=".4f",
        cmap="YlGnBu",
        xticklabels=properties,
        yticklabels=["LPIPS", "SSIM"]
    )
    plt.title("Heatmap: Avg LPIPS vs SSIM per Property")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"Saved heatmap to {save_path}")
per_property_scores = {
    'ClassA': {'lpips': [0.01, 0.015], 'ssim': [0.94, 0.945]},
    'ClassB': {'lpips': [0.002, 0.0015], 'ssim': [0.98, 0.982]},
    'ClassC': {'lpips': [0.004, 0.005], 'ssim': [0.96, 0.962]},
}

plot_similarity_heatmap(per_property_scores)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from typing import List, Tuple


def dual_metric_2d_heatmap(
    lpips_ssim_data: List[Tuple[str, str, float, float]],
    title: str = "Class vs Texture: LPIPS and SSIM Heatmaps"
):
    lpips_dict = defaultdict(lambda: defaultdict(list))
    ssim_dict = defaultdict(lambda: defaultdict(list))

    for cls, tex, lpips, ssim in lpips_ssim_data:
        lpips_dict[cls][tex].append(lpips)
        ssim_dict[cls][tex].append(ssim)

    all_classes = sorted(set(lpips_dict.keys()) | set(ssim_dict.keys()))
    all_textures = sorted({
        tex for d in list(lpips_dict.values()) + list(ssim_dict.values())
        for tex in d
    })

    lpips_array = np.zeros((len(all_classes), len(all_textures)))
    ssim_array = np.zeros((len(all_classes), len(all_textures)))

    for i, cls in enumerate(all_classes):
        for j, tex in enumerate(all_textures):
            lp_scores = lpips_dict[cls].get(tex, [])
            ss_scores = ssim_dict[cls].get(tex, [])
            lpips_array[i, j] = np.mean(lp_scores) if lp_scores else np.nan
            ssim_array[i, j] = np.mean(ss_scores) if ss_scores else np.nan

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.heatmap(
        lpips_array, annot=True, fmt=".3f", cmap="magma",
        xticklabels=all_textures, yticklabels=all_classes,
        ax=axes[0], linewidths=0.5, linecolor='gray'
    )
    axes[0].set_title("LPIPS Heatmap")
    axes[0].set_xlabel("Texture")
    axes[0].set_ylabel("Class")

    sns.heatmap(
        ssim_array, annot=True, fmt=".3f", cmap="YlGnBu",
        xticklabels=all_textures, yticklabels=all_classes,
        ax=axes[1], linewidths=0.5, linecolor='gray'
    )
    axes[1].set_title("SSIM Heatmap")
    axes[1].set_xlabel("Texture")
    axes[1].set_ylabel("")

    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def generate_dual_report_table(
    lpips_ssim_data: List[Tuple[str, str, float, float]]
) -> pd.DataFrame:
    report = defaultdict(list)

    for cls, tex, lpips, ssim in lpips_ssim_data:
        report[(cls, tex)].append((lpips, ssim))

    records = []
    for (cls, tex), scores in report.items():
        lp_arr = np.array([s[0] for s in scores])
        ss_arr = np.array([s[1] for s in scores])
        records.append({
            'Class': cls,
            'Texture': tex,
            'Count': len(scores),
            'Mean LPIPS': np.mean(lp_arr),
            'Std LPIPS': np.std(lp_arr),
            'Min LPIPS': np.min(lp_arr),
            'Max LPIPS': np.max(lp_arr),
            'Mean SSIM': np.mean(ss_arr),
            'Std SSIM': np.std(ss_arr),
            'Min SSIM': np.min(ss_arr),
            'Max SSIM': np.max(ss_arr)
        })

    df = pd.DataFrame(records)
    df = df.sort_values(by=["Class", "Texture"]).reset_index(drop=True)
    return df
if __name__ == "__main__":
    lpips_ssim_data = [
        ("alpha", "coarse", 0.035, 0.94),
        ("alpha", "coarse", 0.038, 0.93),
        ("alpha", "fine",   0.029, 0.95),
        ("beta",  "coarse", 0.040, 0.96),
        ("beta",  "mixed",  0.034, 0.97),
        ("lamellar", "fine", 0.036, 0.98),
        ("lamellar", "coarse", 0.043, 0.92),
        ("lamellar", "mixed", 0.037, 0.91),
        ("beta",  "fine", 0.031, 0.99),
        ("alpha", "mixed", 0.032, 0.93)
    ]

    dual_metric_2d_heatmap(lpips_ssim_data)

    report_df = generate_dual_report_table(lpips_ssim_data)
    print("\n--- LPIPS & SSIM Report Table ---\n")
    print(report_df.to_string(index=False))

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_combined_lpips_ssim_heatmaps(df, save_dir="./visuals", prefix="combined"):
    os.makedirs(save_dir, exist_ok=True)

    pivot_lpips = df.pivot(index='Class', columns='Texture', values='Mean LPIPS')
    pivot_ssim = df.pivot(index='Class', columns='Texture', values='Mean SSIM')

    # HM1: LPIPS heatmap with SSIM overlay
    plt.figure(figsize=(10, 6))
    ax = sns.heatmap(
        pivot_lpips,
        annot=pivot_ssim.round(3), fmt=".3f",
        cmap="rocket_r",
        linewidths=0.5, linecolor='gray',
        cbar_kws={'label': 'LPIPS (lower = better)'}
    )
    plt.title("Combined Heatmap: LPIPS (Color) + SSIM (Text)")
    plt.xlabel("Texture")
    plt.ylabel("Class")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{prefix}_lpips_ssim_combined_heatmap.png")
    plt.show()
    
    # Heatmap 2: SSIM - LPIPS qlt score
    df['SSIM - LPIPS'] = df['Mean SSIM'] - df['Mean LPIPS']
    pivot_score = df.pivot(index='Class', columns='Texture', values='SSIM - LPIPS')

    plt.figure(figsize=(10, 6))
    sns.heatmap(
        pivot_score, annot=True, fmt=".3f",
        cmap='coolwarm', center=0,
        linewidths=0.5, linecolor='gray',
        cbar_kws={'label': 'SSIM - LPIPS'}
    )
    plt.title("Visual Quality Score (SSIM - LPIPS)")
    plt.xlabel("Texture")
    plt.ylabel("Class")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{prefix}_quality_score_heatmap.png")
    plt.show()

report_df["Mean SSIM"] = [0.935, 0.950, 0.930, 0.960, 0.990, 0.970, 0.920, 0.980, 0.910]

plot_combined_lpips_ssim_heatmaps(report_df, save_dir="./visuals", prefix="microstructure")

