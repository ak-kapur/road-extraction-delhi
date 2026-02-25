# src/preprocess.py
import rasterio
import rasterio.mask
import numpy as np
from shapely.geometry import box
import geopandas as gpd
import os
from tqdm import tqdm

# CONFIG - Connaught Place AOI
AOI = [77.213, 28.631, 77.225, 28.641]  # [W, S, E, N]

BAND_FILES = {
    'B02': 'data/raw/T43RGM_20241129T053201_B02_10m.jp2',
    'B03': 'data/raw/T43RGM_20241129T053201_B03_10m.jp2',
    'B04': 'data/raw/T43RGM_20241129T053201_B04_10m.jp2',
    'B08': 'data/raw/T43RGM_20241129T053201_B08_10m.jp2',
}

OUTPUT_STACKED = 'data/processed/delhi_s2_stacked.tif'
OUTPUT_CLIPPED = 'data/processed/delhi_s2_clipped.tif'


def stack_bands():
    print("Stacking bands...")
    os.makedirs('data/processed', exist_ok=True)

    with rasterio.open(list(BAND_FILES.values())[0]) as src:
        meta = src.meta.copy()
        meta.update(
            count=4,
            dtype='float32',
            driver='GTiff'   # force GeoTIFF output
        )

    with rasterio.open(OUTPUT_STACKED, 'w', **meta) as dst:
        for i, (band_name, path) in enumerate(
            tqdm(BAND_FILES.items(), desc="Stacking bands", unit="band"), start=1
        ):
            with rasterio.open(path) as src:
                data = src.read(1).astype('float32')
                dst.write(data, i)

    print(f"Stacked saved -> {OUTPUT_STACKED}")


def clip_to_aoi():
    print("Clipping to AOI...")

    aoi_geom = box(AOI[0], AOI[1], AOI[2], AOI[3])

    with rasterio.open(OUTPUT_STACKED) as src:
        aoi_gdf = gpd.GeoDataFrame(
            geometry=[aoi_geom], crs='EPSG:4326'
        ).to_crs(src.crs)

        clipped, transform = rasterio.mask.mask(
            src,
            aoi_gdf.geometry,
            crop=True
        )

        meta = src.meta.copy()
        meta.update({
            'height': clipped.shape[1],
            'width':  clipped.shape[2],
            'transform': transform,
            'dtype': 'float32'
        })

    with rasterio.open(OUTPUT_CLIPPED, 'w', **meta) as dst:
        dst.write(clipped.astype('float32'))

    print(f"Clipped saved -> {OUTPUT_CLIPPED}")
    print(f"Clipped shape: {clipped.shape}")


def normalize(img):
    out = np.zeros_like(img, dtype=np.float32)
    for i in range(img.shape[0]):
        band = img[i]
        out[i] = (band - band.min()) / (band.max() - band.min() + 1e-8)
    return out


def load_processed():
    with rasterio.open(OUTPUT_CLIPPED) as src:
        img  = src.read().astype(np.float32)
        meta = src.meta.copy()
        transform = src.transform
        crs  = src.crs
    img = normalize(img)
    print(f"Normalized image shape: {img.shape}")
    print(f"Value range: min={img.min():.3f}, max={img.max():.3f}")
    return img, meta, transform, crs


if __name__ == '__main__':
    print("=== Preprocessing Pipeline ===\n")
    stack_bands()
    clip_to_aoi()
    load_processed()
    print("\n=== Preprocessing complete ===")
