import pandas as pd
import numpy as np


def calculate_intersection(x1, y1, z1, theta1, alpha1, x2, y2, z2, theta2, alpha2):
    theta1, alpha1 = np.radians(theta1), np.radians(alpha1)
    theta2, alpha2 = np.radians(theta2), np.radians(alpha2)

    dir1 = np.array([np.sin(theta1) * np.cos(alpha1), np.cos(theta1) * np.cos(alpha1), np.sin(alpha1)])
    dir2 = np.array([np.sin(theta2) * np.cos(alpha2), np.cos(theta2) * np.cos(alpha2), np.sin(alpha2)])

    P1, P2 = np.array([x1, y1, z1]), np.array([x2, y2, z2])
    A, b = np.array([dir1, -dir2]).T, P2 - P1

    t = np.linalg.lstsq(A, b, rcond=None)[0]
    return P1 + t[0] * dir1


def process_file(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    results = []

    for _, row in df.iterrows():
        x1, y1, z1 = row["x(m)_1"], row["y(m)_1"], row["z(m)_1"]
        t1, a1 = row["horizen_1"], row["elevation_1"]
        x2, y2, z2 = row["x(m)_2"], row["y(m)_2"], row["z(m)_2"]
        t2, a2 = row["horizen_2"], row["elevation_2"]

        X, Y, Z = calculate_intersection(x1, y1, z1, t1, a1, x2, y2, z2, t2, a2)
        results.append(list(row) + [X, Y, Z])

    cols = list(df.columns) + ["X(m)", "Y(m)", "Z(m)"]
    pd.DataFrame(results, columns=cols).to_csv(output_csv, index=False)
    print(f"Saved: {output_csv}")
