# src/postprocess.py
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import rasterio
from rasterio.features import shapes
from skimage.morphology import skeletonize, remove_small_objects, binary_closing, disk
import geopandas as gpd
from shapely.geometry import shape, MultiLineString
import matplotlib.pyplot as plt

# CONFIG
MASK_PATH    = 'outputs/masks/road_mask.tif'
GEOJSON_OUT  = 'outputs/roads.geojson'
VISUAL_OUT   = 'outputs/masks/postprocess_visual.png'


def postprocess():
    print("Loading road mask...")
    with rasterio.open(MASK_PATH) as src:
        mask      = src.read(1).astype(np.uint8)
        transform = src.transform
        crs       = src.crs

    print(f"Mask shape: {mask.shape}, Road pixels: {mask.sum()}")

    # --- Step 1: Morphological closing to reconnect broken segments ---
    print("Applying morphological closing...")
    closed = binary_closing(mask.astype(bool), disk(1))

    # --- Step 2: Remove small noise blobs ---
    print("Removing small objects...")
    cleaned = remove_small_objects(closed, min_size=80)

    # --- Step 3: Skeletonize to single-pixel-width roads ---
    print("Skeletonizing...")
    skeleton = skeletonize(cleaned).astype(np.uint8)
    print(f"Skeleton pixels: {skeleton.sum()}")

    # --- Step 4: Vectorize skeleton → polygons → GeoJSON ---
    print("Vectorizing...")
    geometries = []
    for geom, value in shapes(skeleton, transform=transform):
        if value == 1:
            geometries.append(shape(geom))

    print(f"Road geometries extracted: {len(geometries)}")

    # --- Step 5: Save as GeoJSON ---
    os.makedirs('outputs', exist_ok=True)
    gdf = gpd.GeoDataFrame({'geometry': geometries}, crs=crs)
    gdf.to_file(GEOJSON_OUT, driver='GeoJSON')
    print(f"GeoJSON saved -> {GEOJSON_OUT}")

    # --- Step 6: Visualization ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(mask, cmap='gray')
    axes[0].set_title('Raw Mask')
    axes[0].axis('off')

    axes[1].imshow(cleaned, cmap='gray')
    axes[1].set_title('Cleaned + Closed')
    axes[1].axis('off')

    axes[2].imshow(skeleton, cmap='hot')
    axes[2].set_title('Final Skeleton')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(VISUAL_OUT, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Visualization saved -> {VISUAL_OUT}")

        # Final overlay on RGB
    with rasterio.open('data/processed/delhi_s2_clipped.tif') as src:
        rgb_raw = src.read([3, 2, 1]).astype(np.float32)

    def norm(b):
        return (b - b.min()) / (b.max() - b.min() + 1e-8)

    rgb = np.stack([norm(rgb_raw[0]), norm(rgb_raw[1]), norm(rgb_raw[2])], axis=-1)
    overlay = (rgb * 255).astype(np.uint8)
    overlay[skeleton == 1] = [255, 0, 0]   # bright red for final report

    fig2, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(overlay)
    ax.set_title('Final Road Network — Connaught Place, Delhi', fontsize=13)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig('outputs/final_road_overlay.png', dpi=200, bbox_inches='tight')
    plt.show()
    print("Final overlay saved -> outputs/final_road_overlay.png")


    return gdf


if __name__ == '__main__':
    print("=== Road Postprocessing ===\n")
    gdf = postprocess()
    print(f"\nTotal road features: {len(gdf)}")
    print("\n=== Postprocessing complete ===")
