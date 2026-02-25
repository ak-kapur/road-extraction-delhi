# src/acquire.py
import geopandas as gpd
import os
from shapely.geometry import box
from tqdm import tqdm
import time

# CONFIG
AOI = [77.213, 28.631, 77.225, 28.641]  

SHP_PATH = 'data/osm/gis_osm_roads_free_1.shp'


def get_osm_roads():
    steps = [
        "Reading shapefile",
        "Clipping to AOI",
        "Cleaning columns",
        "Saving GeoJSON"
    ]

    with tqdm(total=len(steps), desc="OSM Roads", unit="step") as pbar:

        # Step 1
        pbar.set_description(steps[0])
        roads = gpd.read_file(SHP_PATH)
        pbar.update(1)

        # Step 2
        pbar.set_description(steps[1])
        aoi_geom = box(AOI[0], AOI[1], AOI[2], AOI[3])
        roads    = roads[roads.geometry.intersects(aoi_geom)].copy()
        pbar.update(1)

        # Step 3
        pbar.set_description(steps[2])
        roads = roads[['name', 'fclass', 'geometry']].reset_index(drop=True)
        roads = roads.rename(columns={'fclass': 'highway'})
        pbar.update(1)

        # Step 4
        pbar.set_description(steps[3])
        roads.to_file('data/osm/delhi_roads.geojson', driver='GeoJSON')
        pbar.update(1)

    print(f"OSM roads saved: {len(roads)} segments -> data/osm/delhi_roads.geojson")


if __name__ == '__main__':
    print("=== Data Acquisition ===\n")
    get_osm_roads()
    print("\n=== Acquisition complete ===")
