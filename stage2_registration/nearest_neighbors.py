import pandas as pd
from scipy.spatial import KDTree
import os
import re
from concurrent.futures import ProcessPoolExecutor
import numpy as np


def read_csv(filename):
    return pd.read_csv(filename)


def get_coordinates(df):
    return df[["X", "Y", "Z"]].values


def find_nearest_neighbors(file1, file2, max_distance=1.0):
    df1, df2 = read_csv(file1), read_csv(file2)
    coords1, coords2 = get_coordinates(df1), get_coordinates(df2)

    tree1, tree2 = KDTree(coords1), KDTree(coords2)

    distances_1_to_2, indices_1_to_2 = tree2.query(coords1, k=1)
    mask_1_to_2 = distances_1_to_2 <= max_distance

    distances_2_to_1, indices_2_to_1 = tree1.query(coords2, k=1)
    mask_2_to_1 = distances_2_to_1 <= max_distance

    results = []
    for i, valid in enumerate(mask_1_to_2):
        if valid:
            j = indices_1_to_2[i]
            results.append({
                "X1": df1.iloc[i]["X"], "Y1": df1.iloc[i]["Y"], "Z1": df1.iloc[i]["Z"],
                "X2": df2.iloc[j]["X"], "Y2": df2.iloc[j]["Y"], "Z2": df2.iloc[j]["Z"]
            })

    for i, valid in enumerate(mask_2_to_1):
        if valid:
            j = indices_2_to_1[i]
            results.append({
                "X1": df1.iloc[j]["X"], "Y1": df1.iloc[j]["Y"], "Z1": df1.iloc[j]["Z"],
                "X2": df2.iloc[i]["X"], "Y2": df2.iloc[i]["Y"], "Z2": df2.iloc[i]["Z"]
            })

    return pd.DataFrame(results)


def process_files(pair, max_distance, output_dir):
    file1, file2 = pair
    result_df = find_nearest_neighbors(file1, file2, max_distance)
    output_file = os.path.join(output_dir, f"{os.path.basename(file1)}_{os.path.basename(file2)}.csv")
    result_df.to_csv(output_file, index=False)
    print(f"Saved: {output_file}")


def run_in_parallel(pairs, input_dir, output_dir, max_distance=1.0):
    with ProcessPoolExecutor() as executor:
        futures = []
        for f1, f2 in pairs:
            file1, file2 = os.path.join(input_dir, f1), os.path.join(input_dir, f2)
            futures.append(executor.submit(process_files, (file1, file2), max_distance, output_dir))
        for future in futures:
            future.result()
