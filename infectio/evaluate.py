"""
Functions to evaluate and compare simulation results/metrics against reference
distributions e.g. extracted from dataset. 
"""

from typing import List, Union
import argparse
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import os


def bulk_evaluate(
    ref_file,
    target_root,
    ref_mean_colnames,
    ref_std_colnames,
    target_colnames,
    time_colname="t",
    single_experiments=False,
) -> pd.DataFrame:
    refdf = pd.read_csv(ref_file)
    refdf.set_index(time_colname, inplace=True)
    dist_eval_results = []
    endpoint_eval_results = []

    if single_experiments:
        columns = ["target_folder"] + [s + "-dist" for s in target_colnames]
    else:
        columns = ["target_folder"] + [s + "-mean-dist" for s in target_colnames]

    for target_folder in os.listdir(target_root):
        if target_folder.startswith("."):
            print("- folder starts with ., skipping folder: ", target_folder)
            continue
        if not os.path.isdir(os.path.join(target_root, target_folder)):
            print("- not a folder, skipping folder: ", target_folder)
            continue

        dist_row = {}
        dist_row["target_folder"] = target_folder
        endpoint_row = {}
        endpoint_row["target_folder"] = target_folder
        experiment_root = os.path.join(target_root, target_folder)

        if single_experiments:
            dist_dists, endpoint_dists = evaluate_single_experiments(
                experiment_root,
                refdf,
                ref_mean_colnames,
                ref_std_colnames,
                target_colnames,
                time_colname,
            )
        else:
            dist_dists, endpoint_dists = evaluate_multiple_experiments(
                experiment_root,
                refdf,
                ref_mean_colnames,
                ref_std_colnames,
                target_colnames,
                time_colname,
            )
        if dist_dists is None:
            continue

        for colname, dist in zip(columns[1:], dist_dists):
            dist_row[colname] = dist
        for colname, dist in zip(columns[1:], endpoint_dists):
            endpoint_row[colname] = dist

        dist_eval_results.append(dist_row)
        endpoint_eval_results.append(endpoint_row)

    dist_df = pd.DataFrame(dist_eval_results, index=None)
    add_normalized_sum_to_df(dist_df)
    endpoint_df = pd.DataFrame(endpoint_eval_results, index=None)
    add_normalized_sum_to_df(endpoint_df)

    return dist_df, endpoint_df


def add_normalized_sum_to_df(df):
    dist_cols = [col for col in df.columns if col.endswith("-dist")]
    max_values = {col: df[col].max() for col in dist_cols}
    normalized_cols = {col: df[col] / max_val for col, max_val in max_values.items()}
    df_normalized = pd.DataFrame(normalized_cols)
    normalized_sum = df_normalized.sum(axis=1)
    new_row_label = f"normalized-sum-dist({', '.join(map(str, max_values.values()))})"
    df[new_row_label] = normalized_sum


def evaluate_multiple_experiments(
    experiment_root,
    refdf,
    ref_mean_colnames,
    ref_std_colnames,
    target_colnames,
    time_colname,
):
    # Report multiple criterias of selection, e.g. naive sum of dists, dist2dist
    # distance such as KL or wasserstein, etc.
    # Probably useful to aggregate metrics first but now for a quick result,
    # just do multiple single experiment runs and sum over them
    multiple_dist_dists = []
    multiple_endpoint_dists = []
    for exp_path in os.listdir(experiment_root):
        dist_dists, endpoint_dists = evaluate_single_experiments(
            os.path.join(experiment_root, exp_path),
            refdf,
            ref_mean_colnames,
            ref_std_colnames,
            target_colnames,
            time_colname,
        )
        multiple_dist_dists.append(dist_dists)
        multiple_endpoint_dists.append(endpoint_dists)
    multiple_dist_dists = np.vstack(multiple_dist_dists)
    multiple_endpoint_dists = np.vstack(multiple_endpoint_dists)
    return (
        np.mean(multiple_dist_dists, axis=0).tolist(),
        np.mean(multiple_endpoint_dists, axis=0).tolist(),
    )


def evaluate_single_experiments(
    experiment_root,
    refdf,
    ref_mean_colnames,
    ref_std_colnames,
    target_colnames,
    time_colname,
):
    if "metric.csv" not in os.listdir(experiment_root):
        print("- no metric.csv, skipping folder: ", experiment_root)
        return None
    targetdf = pd.read_csv(os.path.join(experiment_root, "metric.csv"))
    targetdf.set_index(time_colname, inplace=True)
    dist_dists, endpoint_dists = evaluate_simulation_against_reference(
        refdf, targetdf, ref_mean_colnames, ref_std_colnames, target_colnames
    )
    return dist_dists, endpoint_dists


def evaluate_simulation_against_reference(
    refdf: pd.DataFrame,
    targetdf: pd.DataFrame,
    ref_mean_colnames: Union[str, List[str]],
    ref_std_colnames: Union[str, List[str]],
    target_colnames: Union[str, List[str]],
):
    """Computes distance measures of a time-series to distribution for a list of metrics.

    Args:
        refdf (pd.DataFrame): Reference DataFrame containing mean and std values of time-series distributions.
        targetdf (pd.DataFrame): Dataframe containing the simulation results.
        ref_mean_colname (Union[str, List[str]]): Column names of means of reference timeseries dist.
        ref_std_colname (Union[str, List[str]]): Column names of stds of reference timeseries dist.
        target_colname (Union[str, List[str]]): Column name of the target metric.

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

    dist_distances = []
    endpoint_distances = []
    for ref_metric_mean, ref_metric_std, target_metric in zip(
        ref_mean_colnames, ref_std_colnames, target_colnames
    ):
        standard_refdf = refdf.loc[:, [ref_metric_mean, ref_metric_std]].rename(
            columns={ref_metric_mean: "mean", ref_metric_std: "std"}
        )
        standard_targetdf = targetdf.loc[:, [target_metric]].rename(
            columns={target_metric: "value"}
        )
        distances = point2distribution_distance(standard_refdf, standard_targetdf)
        dist_distances.append(distances[0])
        endpoint_distances.append(distances[1])
    return dist_distances, endpoint_distances


def point2distribution_distance(refdf, targetdf):
    """computes distance measure of a time-series to distribution.

    Args:
        refdf (pd.DataFrame): Reference DataFrame with mean and std columns vs time index.
        targetdf (pd.DataFrame): Target values vs time index.

    Returns:
        float: distance measure of comparing timeseries vs timeseries dist.
        float: distance measure of comparing last timepoint of timeseries vs dist.
    """
    # First align the two data frames based on time_colname
    aligned_df = pd.merge_asof(refdf, targetdf, left_index=True, right_index=True)

    dist_distance = aligned_timeseries_point2distribution_distance(
        aligned_df["mean"], aligned_df["std"], aligned_df["value"]
    )
    endpoint_distance = aligned_point2distribution_distance(
        aligned_df["mean"], aligned_df["std"], aligned_df["value"]
    )
    return dist_distance, endpoint_distance


def aligned_point2distribution_distance(ref_mean, ref_std, target) -> float:
    # this only compares the last frame
    return np.abs(target.iloc[-1] - ref_mean.iloc[-1]) / ref_std.iloc[-1]


def aligned_timeseries_point2distribution_distance(ref_mean, ref_std, target) -> float:
    assert (
        len(ref_mean) == len(ref_std) == len(target)
    ), "All inputs must be aligned and have the same length."
    return np.sum(np.abs(target - ref_mean) / ref_std)


def visualize_bulk_evaluation_results(eval_results: pd.DataFrame):
    df = replace_experiment_name_with_params(eval_results)

    sum_dist_col_name = [
        col for col in df.columns if col.startswith("normalized-sum-dist")
    ][0]

    fig = go.Figure(
        data=go.Parcoords(
            line=dict(
                color=df[sum_dist_col_name], colorscale="Blackbody", showscale=True
            ),
            dimensions=[
                dict(values=df[c], label=c, tickvals=df[c].unique())
                for c in df.columns
                if "-dist" not in c
            ]
            + [dict(values=df[c], label=c) for c in df.columns if "-dist" in c],
            unselected=dict(line=dict(opacity=0)),
        ),
    )
    return fig


def replace_experiment_name_with_params(df, column_name="target_folder"):
    return pd.concat(
        [
            pd.json_normalize(df["target_folder"].apply(extract_parameters_from_name)),
            df.drop(["target_folder"], axis=1),
        ],
        axis=1,
    )


def extract_parameters_from_name(name):
    name = name.split("-")
    params = {}
    for p in name:
        if "=" in p:
            k, v = p.split("=")
            params[k] = v
    return params


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
        "--single_experiments",
        type=bool,
        default=False,
        help="choose if each parameter choice has only one experiment in it, \
        otherwise assumes multiple experiments for each set of parameters and \
        uses dist2dist scores.",
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
    parser.add_argument("--output", type=str, help="output save file.")
    parser.add_argument("--timecol", type=str, help="column name of time.", default="t")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dist_eval_results, endpoint_eval_results = bulk_evaluate(
        args.reference,
        args.root,
        args.rmeancol,
        args.rstdcol,
        args.tcol,
        args.timecol,
        args.single_experiments,
    )
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    dist_eval_results.to_csv(args.output)
    print("-- Evaluation results saved to: ", args.output)
    endpoint_output = args.output.replace(".csv", "_endpoint.csv")
    endpoint_eval_results.to_csv(endpoint_output)
    print("-- and                        : ", endpoint_output)
    for colname in dist_eval_results.columns:
        if "-dist" in colname:
            print(f"-- Top experiments based on: {colname}")
            print(colname, dist_eval_results[colname].describe())
            print(dist_eval_results.sort_values(colname, ascending=True).head())

    dist_fig = visualize_bulk_evaluation_results(dist_eval_results)

    from plotly.offline import plot

    plot(
        dist_fig,
        filename=os.path.join(
            os.path.dirname(args.output),
            "dist_bulk_eval_parallel_coordinate_plotly.html",
        ),
    )
    endpoint_fig = visualize_bulk_evaluation_results(endpoint_eval_results)
    plot(
        endpoint_fig,
        filename=os.path.join(
            os.path.dirname(args.output),
            "endpoint_bulk_eval_parallel_coordinate_plotly.html",
        ),
    )
