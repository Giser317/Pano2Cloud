import os
import csv
import math
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
from PIL import Image
import cv2
from multiprocessing import Pool, cpu_count
from functools import partial


W, H, yaw = 2048, 1024, 1536
building_geojson = None


def initializer(building_geojson_path):
    global building_geojson
    building_geojson = gpd.read_file(building_geojson_path)


def pixel_to_xyz(w, h, c, d, x_ref, y_ref):
    x_angle = (w - yaw) * (360 / W) + c
    x_angle_rad = math.radians(x_angle)
    if h > (H / 2):
        z_angle = (h - (H / 2)) * (180 / H)
        z_pixel = -math.tan(math.radians(z_angle)) * d + 2.15
    else:
        z_angle = ((H / 2) - h) * (180 / H)
        z_pixel = d * math.tan(math.radians(z_angle)) + 2.15
    X = x_ref + d * math.sin(x_angle_rad)
    Y = y_ref + d * math.cos(x_angle_rad)
    return X, Y, z_pixel


def compute_distance(point_coord, angle):
    global building_geojson
    point_geom = Point(point_coord)
    angle_rad = math.radians(angle % 360)
    dx, dy = math.sin(angle_rad), math.cos(angle_rad)
    ray = LineString([point_geom, (point_geom.x + 1e6 * dx, point_geom.y + 1e6 * dy)])
    distances = [point_geom.distance(pt) for geom in building_geojson.geometry
                 for pt in ray.intersection(geom.boundary).geoms] if not building_geojson.empty else []
    return min(distances) if distances else None


def process_row(row, out_dir, img_dir, mask_dir):
    fid, x_ref, y_ref, c_val = int(row["FID"]), row["x"], row["y"], row["north_angle"]
    origin_path = os.path.join(img_dir, f"{fid}.jpg")
    mask_path = os.path.join(mask_dir, f"{fid}.png")
    if not (os.path.exists(origin_path) and os.path.exists(mask_path)):
        return {"FID": fid, "status": "missing"}

    origin_img = Image.open(origin_path)
    mask_img = cv2.imread(mask_path)
    white_pixels = np.where(np.all(mask_img == [255, 255, 255], axis=-1))

    result = []
    for h, w in zip(*white_pixels):
        rgb = origin_img.getpixel((w, h))
        north = (yaw - (c_val * (W / 360))) % W
        angle = (w - north) * (360 / W) if w >= north else 360 - (north - w) * (360 / W)
        d = compute_distance((x_ref, y_ref), angle)
        if d:
            X, Y, Z = pixel_to_xyz(w, h, c_val, d, x_ref, y_ref)
            result.append([fid, X, Y, Z, *rgb, d])

    out_csv = os.path.join(out_dir, f"{fid}.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["FID", "X", "Y", "Z", "R", "G", "B", "Depth"])
        writer.writerows(result)
    return {"FID": fid, "status": "done", "num_points": len(result)}


def main(points_csv, building_geojson_path, out_dir, img_dir, mask_dir):
    os.makedirs(out_dir, exist_ok=True)
    data = pd.read_csv(points_csv)
    pool = Pool(processes=cpu_count(), initializer=initializer, initargs=(building_geojson_path,))
    process_func = partial(process_row, out_dir=out_dir, img_dir=img_dir, mask_dir=mask_dir)
    results = pool.map(process_func, [row for _, row in data.iterrows()])
    pool.close()
    pool.join()
    print(f"Processed {sum(r['status']=='done' for r in results)} FIDs")
