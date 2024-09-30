"""
This module computes metrics based on simulation saved states and updates the
results of the simulation. The primary function is to analyze a list of
simulation runs, using their saved states to calculate new metrics.
"""

import os
import csv
import json
import sys

import numpy as np
import pandas as pd

from utils import circularity_metric


def list_sim_paths(root: str):
    """
    List all simulation paths in the given root directory.
    """
    sim_paths = []

    def recursive_add_simpaths(root: str):
        paths = []
        if "metric.csv" in os.listdir(root) and "pos.csv" in os.listdir(root):
            return [root]
        for path in os.listdir(root):
            subpath = os.path.join(root, path)
            if os.path.isdir(subpath):
                paths += recursive_add_simpaths(subpath)
        return paths

    return recursive_add_simpaths(root)


def load_params(path: str) -> dict:
    if not os.path.exists(path):
        raise Exception(f"File {path} does not exist")
    with open(path) as f:
        params = json.load(f)
    return params


def load_points_from_poscsv(path: str, frame: int) -> list:
    points = []
    with open(path, "r") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if int(row["Frame"]) == frame:
                point = (float(row["PosX"]), float(row["PosY"]))
                points.append(point)
    return points


def add_metric_to_metriccsv(
    metriccsv_path: str, metric_name: str, metric_data: np.ndarray
) -> None:
    try:
        df = pd.read_csv(metriccsv_path)
    except Exception as e:
        raise Exception("Error reading metriccsv file: " + str(e))

    # Sanity check: Check if lengths of first column match the length of this new column
    if len(df[df.columns[0]]) != len(metric_data):
        raise ValueError(
            "Lengths of first column do not match the length of the new column"
        )

    # Add the new column to the DataFrame
    df[metric_name] = metric_data

    # Write the DataFrame back to the CSV file
    df.to_csv(metriccsv_path, index=False)


def compute_metric_circularity_1frame(sim_path: str, frame: int) -> float:
    points = load_points_from_poscsv(os.path.join(sim_path, "pos.csv"), frame)
    return circularity_metric(points)


if __name__ == "__main__":
    root = sys.argv[1]
    if not os.path.exists(root):
        raise Exception(f"Directory {root} does not exist")
    sim_paths = list_sim_paths(root)
    for sim_path in sim_paths:
        params = load_params(os.path.join(sim_path, "params.json"))
        new_metric = np.full(params["n_sim_steps"], np.nan)
        metric_circularity = compute_metric_circularity_1frame(
            sim_path, params["n_sim_steps"] - 1
        )
        new_metric[-1] = metric_circularity
        add_metric_to_metriccsv(
            os.path.join(sim_path, "metric.csv"), "circularity", new_metric
        )
        print(f"{sim_path} done!")
