# src/train.py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import rasterio
import rasterio.features
import geopandas as gpd
from shapely.geometry import box
import os
from tqdm import tqdm
from model import UNet

# CONFIG
PATCH_SIZE   = 32
STRIDE       = 16
BATCH_SIZE   = 4
EPOCHS       = 25
LR           = 1e-3
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IMAGE_PATH   = 'data/processed/delhi_s2_clipped.tif'
OSM_PATH     = 'data/osm/delhi_roads.geojson'
MODEL_SAVE   = 'outputs/unet_road.pth'


# ── Dataset ────────────────────────────────────────────────
def create_road_mask(image_path, osm_path):
    with rasterio.open(image_path) as src:
        shape     = (src.height, src.width)
        transform = src.transform
        crs       = src.crs

    roads = gpd.read_file(osm_path).to_crs(crs)
    roads['geometry'] = roads.geometry.buffer(8)  # ~8m buffer

    mask = rasterio.features.rasterize(
        [(geom, 1) for geom in roads.geometry if geom is not None],
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype='uint8'
    )
    print(f"Road mask created: {mask.sum()} road pixels / {mask.size} total")
    return mask


def extract_patches(image, mask, patch_size, stride):
    patches_img  = []
    patches_mask = []
    _, H, W = image.shape

    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            img_patch  = image[:, y:y+patch_size, x:x+patch_size]
            mask_patch = mask[y:y+patch_size, x:x+patch_size]
            patches_img.append(img_patch)
            patches_mask.append(mask_patch)

    print(f"Extracted {len(patches_img)} patches")
    return np.array(patches_img), np.array(patches_mask)


class RoadDataset(Dataset):
    def __init__(self, images, masks):
        self.images = torch.tensor(images, dtype=torch.float32)
        self.masks  = torch.tensor(masks,  dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.masks[idx]


# ── Normalize ──────────────────────────────────────────────
def normalize(img):
    out = np.zeros_like(img, dtype=np.float32)
    for i in range(img.shape[0]):
        band = img[i]
        out[i] = (band - band.min()) / (band.max() - band.min() + 1e-8)
    return out


# ── Loss ───────────────────────────────────────────────────
class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        bce_loss  = self.bce(pred, target)
        pred_sig  = torch.sigmoid(pred)
        inter     = (pred_sig * target).sum()
        dice_loss = 1 - (2 * inter + 1) / (pred_sig.sum() + target.sum() + 1)
        return bce_loss + dice_loss


# ── Training Loop ──────────────────────────────────────────
def train():
    print(f"Using device: {DEVICE}\n")

    # Load image
    with rasterio.open(IMAGE_PATH) as src:
        image = src.read().astype(np.float32)
    image = normalize(image)

    # Create road mask from OSM
    mask = create_road_mask(IMAGE_PATH, OSM_PATH)

    # Extract patches
    patches_img, patches_mask = extract_patches(image, mask, PATCH_SIZE, STRIDE)

    # Dataset and loader
    dataset    = RoadDataset(patches_img, patches_mask)
    loader     = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Model
    model      = UNet(in_channels=4, out_channels=1).to(DEVICE)
    optimizer  = torch.optim.Adam(model.parameters(), lr=LR)
    criterion  = DiceBCELoss()
    scheduler  = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    os.makedirs('outputs', exist_ok=True)
    best_loss  = float('inf')

    # Train
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0

        for imgs, masks in tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False):
            imgs  = imgs.to(DEVICE)
            masks = masks.to(DEVICE)

            optimizer.zero_grad()
            preds = model(imgs)
            loss  = criterion(preds, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        scheduler.step()

        print(f"Epoch {epoch:02d}/{EPOCHS} | Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), MODEL_SAVE)

    print(f"\nBest model saved -> {MODEL_SAVE}")
    print(f"Best loss: {best_loss:.4f}")


if __name__ == '__main__':
    print("=== Training U-Net for Road Extraction ===\n")
    train()
