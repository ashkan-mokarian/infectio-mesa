"""
Functions to evaluate and compare simulation results/metrics against reference
distributions e.g. extracted from dataset. 
"""

from typing import List, Union
import argparse
import pandas as pd
import numpy as np
import os


def bulk_evaluate(
    ref_file,
    target_root,
    ref_mean_colnames,
    ref_std_colnames,
    target_colnames,
    time_colname="t",
) -> pd.DataFrame:
    refdf = pd.read_csv(ref_file)
    refdf.set_index(time_colname, inplace=True)
    columns = ["target_folder"] + [s + "-dist" for s in target_colnames]
    eval_results = []
    for target_folder in os.listdir(target_root):
        if target_folder.startswith("."):
            print("- folder starts with ., skipping folder: ", target_folder)
            continue
        if "metric.csv" not in os.listdir(os.path.join(target_root, target_folder)):
            print("- no metric.csv, skipping folder: ", target_folder)
            continue
        row = {}
        row["target_folder"] = target_folder
        targetdf = pd.read_csv(os.path.join(target_root, target_folder, "metric.csv"))
        # TODO: Temporary, next batch of experiemnts have correct col name
        targetdf = targetdf.rename(columns={"Frame": time_colname})
        targetdf[time_colname] = targetdf[time_colname] * 1 / 6
        targetdf.set_index(time_colname, inplace=True)
        # ////////////////////////////////////////////////////////////////
        dists = evaluate_simulation_against_reference(
            refdf,
            targetdf,
            ref_mean_colnames,
            ref_std_colnames,
            target_colnames,
            time_colname,
        )
        for colname, dist in zip(columns[1:], dists):
            row[colname] = dist
        eval_results.append(row)

    return pd.DataFrame(eval_results, index=None)


def evaluate_simulation_against_reference(
    refdf: pd.DataFrame,
    targetdf: pd.DataFrame,
    ref_mean_colnames: Union[str, List[str]],
    ref_std_colnames: Union[str, List[str]],
    target_colnames: Union[str, List[str]],
    time_colname: str = "t",
) -> List[float]:
    """Computes distance measures of a time-series to distribution for a list of metrics.

    Args:
        refdf (pd.DataFrame): Reference DataFrame containing mean and std values of time-series distributions.
        targetdf (pd.DataFrame): Dataframe containing the simulation results.
        ref_mean_colname (Union[str, List[str]]): Column names of means of reference timeseries dist.
        ref_std_colname (Union[str, List[str]]): Column names of stds of reference timeseries dist.
        target_colname (Union[str, List[str]]): Column name of the target metric.
        time_colname (str, optional): Column name of the time. Defaults to "t".

    Returns:
        List[float]: distance measures of each metric.
    """

    def _to_list(x):
        return x if isinstance(x, list) else [x]

    ref_mean_colnames = _to_list(ref_mean_colnames)
    ref_std_colnames = _to_list(ref_std_colnames)
    target_colnames = _to_list(target_colnames)
    assert (
        len(ref_mean_colnames) == len(ref_std_colnames) == len(target_colnames)
    ), "Length of the three colname lists must be the same."

    distances = []
    for ref_metric_mean, ref_metric_std, target_metric in zip(
        ref_mean_colnames, ref_std_colnames, target_colnames
    ):
        standard_refdf = refdf.loc[:, [ref_metric_mean, ref_metric_std]].rename(
            columns={ref_metric_mean: "mean", ref_metric_std: "std"}
        )
        standard_targetdf = targetdf.loc[:, [target_metric]].rename(
            columns={target_metric: "value"}
        )
        distances.append(
            timeseries_point2distribution_distance(standard_refdf, standard_targetdf)
        )
    return distances


def timeseries_point2distribution_distance(refdf, targetdf):
    """computes distance measure of a time-series to distribution.

    Args:
        refdf (pd.DataFrame): Reference DataFrame with mean and std columns vs time index.
        targetdf (pd.DataFrame): Target values vs time index.

    Returns:
        float: distance measure.
    """
    # First align the two data frames based on time_colname
    aligned_df = pd.merge_asof(refdf, targetdf, left_index=True, right_index=True)

    return aligned_timeseries_point2distribution_distance(
        aligned_df["mean"],
        aligned_df["std"],
        aligned_df["value"],
    )


def aligned_timeseries_point2distribution_distance(ref_mean, ref_std, target) -> float:
    assert (
        len(ref_mean) == len(ref_std) == len(target)
    ), "All inputs must be aligned and have the same length."
    return np.sum(np.abs(target - ref_mean) / ref_std)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Bulk evaluation of simulation results."
    )
    parser.add_argument("--reference", type=str, help="path to reference csv file")
    parser.add_argument(
        "--root",
        type=str,
        help="path to simulation results where metric.csv can be found.",
    )
    parser.add_argument(
        "--tcol", nargs="+", type=str, help="column names of target metrics."
    )
    parser.add_argument(
        "--rmeancol",
        nargs="+",
        type=str,
        help="column names of reference metric means.",
    )
    parser.add_argument(
        "--rstdcol",
        nargs="+",
        type=str,
        help="column names of reference metric stds.",
    )
    parser.add_argument("--timecol", type=str, help="column name of time.", default="t")
    parser.add_argument("--output", type=str, help="output save file.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    eval_results = bulk_evaluate(
        args.reference, args.root, args.rmeancol, args.rstdcol, args.tcol
    )
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    eval_results.to_csv(args.output)
