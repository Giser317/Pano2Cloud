import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from skimage.feature import local_binary_pattern
from skimage.color import rgb2lab
from sklearn.preprocessing import StandardScaler


def compute_lbp_histogram(lbp_image, coords, window_size=5, n_bins=10):
    half = window_size // 2
    features = []
    H, W = lbp_image.shape
    for (x, y) in coords:
        x_min, x_max = max(x - half, 0), min(x + half + 1, H)
        y_min, y_max = max(y - half, 0), min(y + half + 1, W)
        patch = lbp_image[x_min:x_max, y_min:y_max].ravel()
        hist, _ = np.histogram(patch, bins=n_bins, range=(0, n_bins))
        hist = hist.astype(np.float32)
        if hist.sum() > 0:
            hist /= hist.sum()
        features.append(hist)
    return np.array(features)


def extract_features(img, mask, window_size=5):
    lab_img = rgb2lab(img)
    gray_img = np.mean(img, axis=2).astype(np.uint8)
    lbp = local_binary_pattern(gray_img, P=8, R=1, method="uniform")
    coords = np.argwhere(mask > 128)

    L, a, b = lab_img[mask > 128, 0], lab_img[mask > 128, 1], lab_img[mask > 128, 2]
    lbp_hists = compute_lbp_histogram(lbp, coords, window_size)

    features = np.hstack([L.reshape(-1, 1), a.reshape(-1, 1), b.reshape(-1, 1), lbp_hists])
    return features, coords, img.shape[:2]


def find_optimal_clusters(features, min_clusters=2, max_clusters=10, sample_size=5000):
    if features.shape[0] > sample_size:
        idx = np.random.choice(features.shape[0], sample_size, replace=False)
        sample_data = features[idx, :]
    else:
        sample_data = features

    best_k, best_score = min_clusters, -1
    for k in range(min_clusters, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42).fit(sample_data)
        score = silhouette_score(sample_data, kmeans.labels_)
        if score > best_score:
            best_k, best_score = k, score
    return best_k
