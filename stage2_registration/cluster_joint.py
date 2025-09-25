import pandas as pd
import numpy as np
from kmodes.kprototypes import KPrototypes
from sklearn.preprocessing import StandardScaler


def read_correspondence_points(file_path):
    data = pd.read_csv(file_path)
    points_a = data[["Pixel_X1", "Pixel_Y1", "X1", "Y1", "Z1"]].values
    points_b = data[["Pixel_X2", "Pixel_Y2", "X2", "Y2", "Z2"]].values
    types_a = data["type1"].values
    types_b = data["type2"].values
    return points_a, points_b, types_a, types_b


def prepare_joint_data(points_a, points_b, types_a, types_b):
    scaler = StandardScaler()
    points_a_scaled = scaler.fit_transform(points_a)
    points_b_scaled = scaler.fit_transform(points_b)
    return np.hstack([points_a_scaled, points_b_scaled, types_a.reshape(-1, 1), types_b.reshape(-1, 1)])


def joint_clustering(data, min_clusters=2, max_clusters=10):
    best_k, best_cost, best_clusters = None, float("inf"), None
    for k in range(min_clusters, max_clusters + 1):
        kproto = KPrototypes(n_clusters=k, init="Cao", n_init=5, verbose=0)
        clusters = kproto.fit_predict(data, categorical=[10, 11])
        if kproto.cost_ < best_cost:
            best_k, best_cost, best_clusters = k, kproto.cost_, clusters
    return best_clusters, best_k
