import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString
from scipy.spatial import cKDTree


def resample_polygon(coords, segment_length=1.0):
    line = LineString(coords)
    length = line.length
    num_points = int(np.ceil(length / segment_length))
    distances = np.linspace(0, length, num_points, endpoint=False)
    resampled = [line.interpolate(d) for d in distances]
    return np.array([(p.x, p.y) for p in resampled])


def detect_corners(contour, threshold=0.5):
    prev, nxt = np.roll(contour, 1, axis=0), np.roll(contour, -1, axis=0)
    vectors = nxt - prev
    curvature = np.linalg.norm(vectors, axis=1)
    idx = np.where(curvature > threshold)[0]
    return contour[idx]


def compute_internal_energy(points, alpha, beta):
    prev, nxt = np.roll(points, 1, axis=0), np.roll(points, -1, axis=0)
    d2 = nxt - 2 * points + prev
    return alpha * (nxt - prev) + beta * d2


def compute_external_energy(points, tree, radius, corners, weight=2.0):
    external = np.zeros_like(points)
    for i, p in enumerate(points):
        idx = tree.query_ball_point(p, r=radius)
        if idx:
            centroid = np.mean(tree.data[idx], axis=0)
            direction = centroid - p
            if any(np.linalg.norm(p - c) < 1e-3 for c in corners):
                direction *= weight
            external[i] = direction
    return external


def active_contour(point_cloud, init_contour, corners,
                   alpha=0.05, beta=0.25,
                   segment_length=1.0, radius=2.0,
                   iterations=100, step=0.1, corner_weight=2.0):
    contour = resample_polygon(init_contour, segment_length)
    tree = cKDTree(point_cloud)
    for i in range(iterations):
        internal = compute_internal_energy(contour, alpha, beta)
        external = compute_external_energy(contour, tree, radius, corners, corner_weight)
        contour += step * (internal + external)
    return contour


def to_polygon(points):
    poly = Polygon(points)
    if not poly.exterior.is_closed:
        poly = Polygon(np.vstack([points, points[0]]))
    return poly
