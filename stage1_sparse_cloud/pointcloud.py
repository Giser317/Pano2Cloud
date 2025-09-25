import cv2
import numpy as np
import csv
import math
from PIL import Image
import pandas as pd
from tqdm import tqdm


def hori_distance(h, H, d):
    """Convert pixel-to-camera distance into horizontal distance."""
    angle = (180 / H) * abs(h - H / 2)
    angle_rad = math.radians(angle)
    return d * math.cos(angle_rad)


def pixel_to_xyz(w, h, W, H, yaw, heading, depth, x, y):
    """Convert image pixel coordinates into 3D world coordinates."""
    d = hori_distance(h, H, depth)
    x_angle = (w - yaw) * (360 / W) + heading
    x_angle_rad = math.radians(x_angle)

    if h > (H / 2):
        z_angle = (h - (H / 2)) * (180 / H)
        z = -math.tan(math.radians(z_angle)) * d + 2.15
    else:
        z_angle = ((H / 2) - h) * (180 / H)
        z = d * math.tan(math.radians(z_angle)) + 2.15

    X = x + d * math.sin(x_angle_rad)
    Y = y + d * math.cos(x_angle_rad)
    return X, Y, z


def extract_depth_coordinates(depth_img_path, mask_img_path, output_csv, W, H, yaw, x, y, heading):
    """Extract 3D coordinates from depth and mask images, save as CSV."""
    depth_img = np.array(Image.open(depth_img_path)).astype(np.float32) / 256.0
    mask_img = cv2.imread(mask_img_path)
    white_pixels = np.where(np.all(mask_img == [255, 255, 255], axis=-1))

    result = []
    for h, w in zip(white_pixels[0], white_pixels[1]):
        depth_value = depth_img[h, w]
        X, Y, Z = pixel_to_xyz(w, h, W, H, yaw, heading, depth_value, x, y)
        result.append([w, h, depth_value, X, Y, Z])

    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Pixel_X", "Pixel_Y", "Depth_Value", "X", "Y", "Z"])
        writer.writerows(result)


def batch_process(camera_file, depth_dir, mask_dir, output_dir, W=2048, H=1024, yaw=1536):
    """Batch process multiple panoramas based on camera metadata."""
    data = pd.read_excel(camera_file)
    for _, row in tqdm(data.iterrows(), total=len(data)):
        idx = int(row["FID"])
        x, y, heading = row["x"], row["y"], row["north_angle"]

        depth_img_path = f"{depth_dir}/{idx}.png"
        mask_img_path = f"{mask_dir}/{idx}.png"
        output_csv = f"{output_dir}/{idx}_XYZ_originDepthMap.csv"

        try:
            extract_depth_coordinates(
                depth_img_path, mask_img_path, output_csv, W, H, yaw, x, y, heading
            )
        except Exception as e:
            print(f"Error processing {idx}: {e}")
            continue


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate initial 3D point clouds from depth maps")
    parser.add_argument("--camera_file", type=str, required=True, help="Camera metadata (Excel file)")
    parser.add_argument("--depth_dir", type=str, required=True, help="Directory of depth images")
    parser.add_argument("--mask_dir", type=str, required=True, help="Directory of mask images")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for output CSVs")
    parser.add_argument("--W", type=int, default=2048, help="Image width")
    parser.add_argument("--H", type=int, default=1024, help="Image height")
    parser.add_argument("--yaw", type=int, default=1536, help="Yaw offset in pixels")
    args = parser.parse_args()

    batch_process(args.camera_file, args.depth_dir, args.mask_dir, args.output_dir, args.W, args.H, args.yaw)
