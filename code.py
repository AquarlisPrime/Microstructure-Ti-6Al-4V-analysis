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

import torch
import torch.nn.functional as F
import lpips
import numpy as np
import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load pretrained LPIPS model (AlexNet backbone)
lpips_model = lpips.LPIPS(net='alex').to(device)
lpips_model.eval()

def compute_lpips_and_saliency(pred_img, target_img):
    """
    Compute LPIPS perceptual distance and saliency heatmap showing
    pixel regions that contribute most to perceptual differences.
    """
    pred_img = pred_img.clone().detach().to(device).requires_grad_(True)
    target_img = target_img.clone().detach().to(device)

    if pred_img.grad is not None:
        pred_img.grad.zero_()

    lpips_score = lpips_model(pred_img, target_img)
    lpips_score.backward()

    saliency = pred_img.grad.abs().mean(dim=1).squeeze(0)
    saliency -= saliency.min()
    saliency /= (saliency.max() + 1e-10)

    return lpips_score.item(), saliency.cpu().numpy()

def visualize_lpips_saliency(pred_img, target_img, saliency_map, lpips_score):
    """Visualize predicted image, target image, and saliency heatmap."""
    def tensor_to_img(t):
        img = t.squeeze().permute(1, 2, 0).cpu().numpy()
        img = (img + 1) / 2
        return np.clip(img, 0, 1)

    pred_np = tensor_to_img(pred_img)
    target_np = tensor_to_img(target_img)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1); plt.title("Predicted Image"); plt.imshow(pred_np); plt.axis('off')
    plt.subplot(1, 3, 2); plt.title("Target Image"); plt.imshow(target_np); plt.axis('off')
    plt.subplot(1, 3, 3); plt.title(f"Saliency Map\nLPIPS: {lpips_score:.4f}"); plt.imshow(saliency_map, cmap='hot'); plt.axis('off')
    plt.tight_layout(); plt.show()

def plot_classwise_lpips(lpips_per_class):
    class_names = list(lpips_per_class.keys())
    mean_scores = [np.mean(scores) for scores in lpips_per_class.values()]
    plt.figure(figsize=(10, 5))
    plt.bar(class_names, mean_scores, color='skyblue')
    plt.ylabel("Mean LPIPS Score")
    plt.title("LPIPS per Class")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_texturewise_lpips(lpips_per_texture):
    texture_names = list(lpips_per_texture.keys())
    mean_scores = [np.mean(scores) for scores in lpips_per_texture.values()]
    plt.figure(figsize=(10, 5))
    plt.bar(texture_names, mean_scores, color='orange')
    plt.ylabel("Mean LPIPS Score")
    plt.title("LPIPS per Texture Cluster")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def heatmap_classwise_lpips(lpips_per_class):
    class_names = list(lpips_per_class.keys())
    mean_scores = [np.mean(scores) for scores in lpips_per_class.values()]
    plt.figure(figsize=(8, 6))
    sns.heatmap(np.array([mean_scores]), annot=True, fmt=".3f",
                xticklabels=class_names, yticklabels=['LPIPS'], cmap='coolwarm')
    plt.title("Class-wise LPIPS Heatmap")
    plt.tight_layout()
    plt.show()

def heatmap_texturewise_lpips(lpips_per_texture):
    texture_names = list(lpips_per_texture.keys())
    mean_scores = [np.mean(scores) for scores in lpips_per_texture.values()]
    plt.figure(figsize=(8, 6))
    sns.heatmap(np.array([mean_scores]), annot=True, fmt=".3f",
                xticklabels=texture_names, yticklabels=['LPIPS'], cmap='YlGnBu')
    plt.title("Texture Cluster-wise LPIPS Heatmap")
    plt.tight_layout()
    plt.show()

# Example usage
def main():
    H, W = 128, 128

    # Generate target image
    target_img = torch.randn(1, 3, H, W).to(device)
    target_img = torch.clamp(target_img, -1, 1)

    # Generate pred_img very close to target_img (small Gaussian noise)
    noise = torch.randn_like(target_img) * 0.01
    pred_img = torch.clamp(target_img + noise, -1, 1)

    # Compute LPIPS and saliency
    lpips_score, saliency_map = compute_lpips_and_saliency(pred_img, target_img)
    visualize_lpips_saliency(pred_img, target_img, saliency_map, lpips_score)

    # Dummy LPIPS scores (for heatmaps)
    lpips_per_class = {'alpha': [0.032, 0.041], 'beta': [0.025], 'lamellar': [0.045, 0.038]}
    lpips_per_texture = {'fine': [0.031], 'coarse': [0.042, 0.037], 'mixed': [0.029]}

    plot_classwise_lpips(lpips_per_class)
    plot_texturewise_lpips(lpips_per_texture)
    heatmap_classwise_lpips(lpips_per_class)
    heatmap_texturewise_lpips(lpips_per_texture)

if __name__ == "__main__":
    main()

def compute_lpips_and_saliency(
    pred_img: torch.Tensor,
    target_img: torch.Tensor
) -> Tuple[float, np.ndarray]:
    pred_img = pred_img.clone().detach().to(device).requires_grad_(True)
    target_img = target_img.clone().detach().to(device)

    if pred_img.grad is not None:
        pred_img.grad.zero_()

    lpips_score = lpips_model(pred_img, target_img)
    lpips_score.sum().backward()  

    # L2 channels for saliency
    saliency = pred_img.grad.detach()
    saliency = torch.norm(saliency, p=2, dim=1).squeeze(0)

    # Norm
    saliency -= saliency.min()
    saliency /= (saliency.max() + 1e-10)
    saliency = saliency.clamp(0, 1)

    return lpips_score.item(), saliency.cpu().numpy()



def compute_lpips_batch(
    pred_imgs: torch.Tensor,
    target_imgs: torch.Tensor,
) -> List[float]:
    lpips_scores = []
    with torch.no_grad():
        for i in range(pred_imgs.shape[0]):
            score = lpips_model(pred_imgs[i:i+1].to(device), target_imgs[i:i+1].to(device))
            lpips_scores.append(score.item())
    return lpips_scores


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    filepath: str
) -> None:
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict()
    }
    if optimizer:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    filepath: str,
    device: torch.device = device
) -> Tuple[int, torch.nn.Module, Optional[torch.optim.Optimizer]]:
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Checkpoint loaded from {filepath}, epoch {epoch}")
    return epoch, model, optimizer


class LPIPSVisualizer:
    def __init__(self, save_dir: Optional[str] = None):
        self.save_dir = save_dir
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def tensor_to_img(self, t: torch.Tensor) -> np.ndarray:
        img = t.squeeze().permute(1, 2, 0).cpu().numpy()
        img = (img + 1) / 2  
        return np.clip(img, 0, 1)

    def visualize_lpips_saliency(
        self,
        pred_img: torch.Tensor,
        target_img: torch.Tensor,
        saliency_map: np.ndarray,
        lpips_score: float,
        filename: Optional[str] = None
    ) -> None:
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.title("Predicted Image")
        plt.imshow(self.tensor_to_img(pred_img))
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title("Target Image")
        plt.imshow(self.tensor_to_img(target_img))
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title(f"Saliency Map\nLPIPS: {lpips_score:.4f}")
        plt.imshow(saliency_map, cmap='hot')
        plt.axis('off')

        plt.tight_layout()
        if filename and self.save_dir:
            filepath = os.path.join(self.save_dir, filename)
            plt.savefig(filepath)
            print(f"Saved visualization to {filepath}")
            plt.close()
        else:
            plt.show()

    def plot_classwise_lpips(self, lpips_per_class: Dict[str, List[float]], filename: Optional[str] = None) -> None:
        class_names = list(lpips_per_class.keys())
        mean_scores = [np.mean(scores) for scores in lpips_per_class.values()]
        plt.figure(figsize=(10, 5))
        plt.bar(class_names, mean_scores, color='skyblue')
        plt.ylabel("Mean LPIPS Score")
        plt.title("LPIPS per Class")
        plt.xticks(rotation=45)
        plt.tight_layout()
        self._save_or_show_plot(filename, "classwise_lpips.png")

    def plot_texturewise_lpips(self, lpips_per_texture: Dict[str, List[float]], filename: Optional[str] = None) -> None:
        texture_names = list(lpips_per_texture.keys())
        mean_scores = [np.mean(scores) for scores in lpips_per_texture.values()]
        plt.figure(figsize=(10, 5))
        plt.bar(texture_names, mean_scores, color='orange')
        plt.ylabel("Mean LPIPS Score")
        plt.title("LPIPS per Texture Cluster")
        plt.xticks(rotation=45)
        plt.tight_layout()
        self._save_or_show_plot(filename, "texturewise_lpips.png")

    def heatmap_classwise_lpips(self, lpips_per_class: Dict[str, List[float]], filename: Optional[str] = None) -> None:
        class_names = list(lpips_per_class.keys())
        mean_scores = [np.mean(scores) for scores in lpips_per_class.values()]
        plt.figure(figsize=(8, 6))
        sns.heatmap(np.array([mean_scores]), annot=True, fmt=".3f",
                    xticklabels=class_names, yticklabels=['LPIPS'], cmap='coolwarm')
        plt.title("Class-wise LPIPS Heatmap")
        plt.tight_layout()
        self._save_or_show_plot(filename, "heatmap_classwise.png")

    def heatmap_texturewise_lpips(self, lpips_per_texture: Dict[str, List[float]], filename: Optional[str] = None) -> None:
        texture_names = list(lpips_per_texture.keys())
        mean_scores = [np.mean(scores) for scores in lpips_per_texture.values()]
        plt.figure(figsize=(8, 6))
        sns.heatmap(np.array([mean_scores]), annot=True, fmt=".3f",
                    xticklabels=texture_names, yticklabels=['LPIPS'], cmap='YlGnBu')
        plt.title("Texture Cluster-wise LPIPS Heatmap")
        plt.tight_layout()
        self._save_or_show_plot(filename, "heatmap_texturewise.png")

    def _save_or_show_plot(self, filename: Optional[str], default_name: str):
        if filename and self.save_dir:
            filepath = os.path.join(self.save_dir, filename)
            plt.savefig(filepath)
            print(f"Saved plot to {filepath}")
            plt.close()
        elif self.save_dir:
            filepath = os.path.join(self.save_dir, default_name)
            plt.savefig(filepath)
            print(f"Saved plot to {filepath}")
            plt.close()
        else:
            plt.show()


# Eg and simulation
if __name__ == "__main__":
    H, W = 128, 128
    save_dir = "./lpips_vis"
    visualizer = LPIPSVisualizer(save_dir)

    splits = {
        'train': [(torch.randn(1, 3, H, W), 0.01), (torch.randn(1, 3, H, W), 0.02)],
        'test': [(torch.randn(1, 3, H, W), 0.01)],
        'val': [(torch.randn(1, 3, H, W), 0.015), (torch.randn(1, 3, H, W), 0.005)],
    }

    lpips_scores_per_split = {}
    for split_name, examples in splits.items():
        scores = []
        for i, (target_img, noise_level) in enumerate(examples):
            target_img = target_img.to(device).clamp(-1, 1)
            noise = torch.randn_like(target_img) * noise_level
            pred_img = (target_img + noise).clamp(-1, 1)
            score, saliency = compute_lpips_and_saliency(pred_img, target_img)

            if split_name == 'train' and i == 0:
                visualizer.visualize_lpips_saliency(pred_img, target_img, saliency, score, filename="train_sample_0.png")

            scores.append(score)
        lpips_scores_per_split[split_name] = scores

    # LPIPS scores
    print("LPIPS scores per split:")
    for split, scores in lpips_scores_per_split.items():
        print(f"{split}: {scores}")

    # Save 
    dummy_model = torch.nn.Linear(10, 10)
    save_checkpoint(dummy_model, None, epoch=1, filepath="./lpips_checkpoint.pth")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from typing import List, Tuple

def lpips_2d_heatmap(
    lpips_data: List[Tuple[str, str, float]],
    title: str = "Combined LPIPS Heatmap (Class vs. Texture)"
):
    
    heatmap_dict = defaultdict(lambda: defaultdict(list))
    for cls, tex, score in lpips_data:
        heatmap_dict[cls][tex].append(score)

    all_classes = sorted(heatmap_dict.keys())
    all_textures = sorted({tex for d in heatmap_dict.values() for tex in d})

    heatmap_array = np.zeros((len(all_classes), len(all_textures)))

    for i, cls in enumerate(all_classes):
        for j, tex in enumerate(all_textures):
            scores = heatmap_dict[cls].get(tex, [])
            heatmap_array[i, j] = np.mean(scores) if scores else np.nan

    # heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(
        heatmap_array,
        annot=True,
        fmt=".3f",
        xticklabels=all_textures,
        yticklabels=all_classes,
        cmap='magma',
        linewidths=0.5,
        linecolor='gray'
    )
    plt.title(title)
    plt.xlabel("Texture Cluster")
    plt.ylabel("Class Label")
    plt.tight_layout()
    plt.show()


def generate_lpips_report_table(
    lpips_data: List[Tuple[str, str, float]]
) -> pd.DataFrame:
   
    report = defaultdict(list)

    for cls, tex, score in lpips_data:
        report[(cls, tex)].append(score)

    records = []
    for (cls, tex), scores in report.items():
        scores_np = np.array(scores)
        records.append({
            'Class': cls,
            'Texture': tex,
            'Count': len(scores),
            'Mean LPIPS': np.mean(scores_np),
            'Std LPIPS': np.std(scores_np),
            'Min LPIPS': np.min(scores_np),
            'Max LPIPS': np.max(scores_np)
        })

    df = pd.DataFrame(records)
    df = df.sort_values(by=["Class", "Texture"]).reset_index(drop=True)
    return df


# Eg
if __name__ == "__main__":
    lpips_data = [
        ("alpha", "coarse", 0.035), ("alpha", "coarse", 0.038),
        ("alpha", "fine", 0.029),   ("beta", "coarse", 0.040),
        ("beta", "mixed", 0.034),   ("lamellar", "fine", 0.036),
        ("lamellar", "coarse", 0.043), ("lamellar", "mixed", 0.037),
        ("beta", "fine", 0.031),    ("alpha", "mixed", 0.032)
    ]

    lpips_2d_heatmap(lpips_data)

    # Gen n display
    report_df = generate_lpips_report_table(lpips_data)
    print("\n--- LPIPS Report Table ---\n")
    print(report_df.to_string(index=False))
