# src/predict.py
import numpy as np
import torch
import rasterio
from rasterio.transform import from_bounds
import matplotlib.pyplot as plt
import os
from skimage.morphology import skeletonize
from model import UNet
from skimage.morphology import skeletonize, remove_small_objects
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# CONFIG
PATCH_SIZE  = 32
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGE_PATH  = 'data/processed/delhi_s2_clipped.tif'
MODEL_PATH  = 'outputs/unet_road.pth'
MASK_OUTPUT = 'outputs/masks/road_mask.tif'
PRED_VISUAL = 'outputs/masks/prediction_visual.png'


def normalize(img):
    out = np.zeros_like(img, dtype=np.float32)
    for i in range(img.shape[0]):
        band = img[i]
        out[i] = (band - band.min()) / (band.max() - band.min() + 1e-8)
    return out


def predict():
    print("Loading model...")
    model = UNet(in_channels=4, out_channels=1).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    print("Loading image...")
    with rasterio.open(IMAGE_PATH) as src:
        image     = src.read().astype(np.float32)
        meta      = src.meta.copy()
        transform = src.transform
        crs       = src.crs

    image = normalize(image)
    _, H, W = image.shape

    print(f"Image shape: {image.shape}")
    print("Running inference...")

    # Full image prediction using sliding window
    pred_map   = np.zeros((H, W), dtype=np.float32)
    count_map  = np.zeros((H, W), dtype=np.float32)

    with torch.no_grad():
        for y in range(0, H - PATCH_SIZE + 1, PATCH_SIZE // 2):
            for x in range(0, W - PATCH_SIZE + 1, PATCH_SIZE // 2):
                patch = image[:, y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                patch_tensor = torch.tensor(patch).unsqueeze(0).to(DEVICE)
                output = torch.sigmoid(model(patch_tensor))
                pred_map[y:y+PATCH_SIZE, x:x+PATCH_SIZE]  += output.squeeze().cpu().numpy()
                count_map[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += 1

    # Average overlapping predictions
    count_map[count_map == 0] = 1
    pred_map  = pred_map / count_map

    # Threshold to binary mask
    binary_mask = (pred_map > 0.80).astype(np.uint8)

    print(f"Road pixels detected: {binary_mask.sum()}")

    # Save GeoTIFF mask
    os.makedirs('outputs/masks', exist_ok=True)
    meta.update({'count': 1, 'dtype': 'uint8', 'driver': 'GTiff'})
    with rasterio.open(MASK_OUTPUT, 'w', **meta) as dst:
        dst.write(binary_mask[np.newaxis, :, :])
    print(f"Road mask saved -> {MASK_OUTPUT}")

    # Visualization
    with rasterio.open(IMAGE_PATH) as src:
        rgb_raw = src.read([3, 2, 1]).astype(np.float32)

    def norm(b):
        return (b - b.min()) / (b.max() - b.min() + 1e-8)

    rgb = np.stack([norm(rgb_raw[0]),
                    norm(rgb_raw[1]),
                    norm(rgb_raw[2])], axis=-1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(rgb)
    axes[0].set_title('Sentinel-2 RGB')
    axes[0].axis('off')

    axes[1].imshow(pred_map, cmap='hot')
    axes[1].set_title('Road Probability Map')
    axes[1].axis('off')

    binary_clean = remove_small_objects(binary_mask.astype(bool), min_size=50)
    skeleton = skeletonize(binary_clean).astype(np.uint8)
    skeleton = skeletonize(binary_mask).astype(np.uint8)
    overlay = (rgb * 255).astype(np.uint8)       
    overlay[skeleton == 1] = [139, 0, 0]         
    axes[2].imshow(overlay)
    axes[2].set_title('Detected Roads Overlay')
    axes[2].axis('off')


    plt.tight_layout()
    plt.savefig(PRED_VISUAL, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Visualization saved -> {PRED_VISUAL}")

    return binary_mask, transform, crs


if __name__ == '__main__':
    print("=== Road Prediction ===\n")
    predict()
    print("\n=== Prediction complete ===")
