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
from scipy import stats

# TODO: change these to one list, with difference being adding -mean and -std
# to the end whenever you change the reference datasets and recompute them again
# Also change experiment colnames to be the same as the other two -> infected to inf
target_colnames_mean = [
    "inf-count-mean",
    "area-mean(um2)",
    "radial-velocity-mean(um/min)",
]
target_colnames_std = ["inf-count-std", "area-std(um2)", "radial-velocity-std(um/min)"]
experiment_colnames = ["infected-count", "area(um2)", "radial-velocity(um/min)"]
time_colname = "t"


def evaluate_experiments(
    experiments_root,
    target_dataset_csv,
    target_N,  # Number of measured points of the reference dataset used for Corrected standardard deviation
) -> pd.DataFrame:
    target_df = pd.read_csv(target_dataset_csv)
    target_df.set_index(time_colname, inplace=True)
    eval_results = []
    for exp_folder in os.listdir(experiments_root):
        if exp_folder.startswith("."):
            print("- folder starts with ., skipping folder: ", exp_folder)
            continue
        exp_path = os.path.join(experiments_root, exp_folder)
        if not os.path.isdir(exp_path):
            print("- not a folder, skipping folder: ", exp_path)
            continue

        row = {}
        row["experiment_name"] = exp_folder

        experiment_metriccsv_paths = get_metric_paths_for_experiment(exp_path)
        if len(experiment_metriccsv_paths) == 0:
            print("- no metric csvs found for ", exp_folder)
            continue
        if len(experiment_metriccsv_paths) < 3:
            print("- less than 3 metric csvs found for ", exp_folder)
            continue

        # Add evaluation scores here
        # 1. zscore = (val-mean)/std (summed for time-series)
        # 2. corrected xi2 score (described in here: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10617697/#pone.0289619.s001)
        # 3. t-test ONLY for lastframe which compares only based on mean of the target distribution
        # 4. z-test is based on zscore with an additional division by sqrt(n) where n is the number of experiments

        # infected count
        z, zlast = evaluate_zscore(
            target_df,
            target_colnames_mean[0],
            target_colnames_std[0],
            experiment_metriccsv_paths,
            experiment_colnames[0],
            also_return_only_last_frame=True,
        )
        row["infected-count-zscore"] = z
        row["infected-count-zscore-mean"] = sum(z) / len(z)
        row["infected-count-lastframe-zscore"] = zlast
        row["infected-count-lastframe-zscore-mean"] = sum(zlast) / len(zlast)

        xi2, xi2last = evaluate_corrected_xi2_score(
            target_df,
            target_colnames_mean[0],
            target_colnames_std[0],
            experiment_metriccsv_paths,
            experiment_colnames[0],
            target_N,
            also_return_only_last_frame=True,
        )
        row["infected-count-corxi2-pval"] = xi2
        row["infected-count-lastframe-corxi2-pval"] = xi2last

        ttest_pval_lastframe = evaluate_t_test_lastframe(
            target_df,
            target_colnames_mean[0],
            experiment_metriccsv_paths,
            experiment_colnames[0],
        )
        row["infected-count-lastframe-t-test-pvalue"] = ttest_pval_lastframe

        ztest_pval_lastframe = evaluate_z_test_lastframe(
            target_df,
            target_colnames_mean[0],
            target_colnames_std[0],
            experiment_metriccsv_paths,
            experiment_colnames[0],
        )
        row["infected-count-lastframe-z-test-pvalue"] = ztest_pval_lastframe

        # area
        z, zlast = evaluate_zscore(
            target_df,
            target_colnames_mean[1],
            target_colnames_std[1],
            experiment_metriccsv_paths,
            experiment_colnames[1],
            also_return_only_last_frame=True,
        )
        row["area-zscore"] = z
        row["area-zscore-mean"] = sum(z) / len(z)
        row["area-lastframe-zscore"] = zlast
        row["area-lastframe-zscore-mean"] = sum(zlast) / len(zlast)

        xi2, xi2last = evaluate_corrected_xi2_score(
            target_df,
            target_colnames_mean[1],
            target_colnames_std[1],
            experiment_metriccsv_paths,
            experiment_colnames[1],
            target_N,
            also_return_only_last_frame=True,
        )
        row["area-corxi2-pval"] = xi2
        row["area-lastframe-corxi2-pval"] = xi2last

        ttest_pval_lastframe = evaluate_t_test_lastframe(
            target_df,
            target_colnames_mean[1],
            experiment_metriccsv_paths,
            experiment_colnames[1],
        )
        row["area-lastframe-t-test-pvalue"] = ttest_pval_lastframe

        ztest_pval_lastframe = evaluate_z_test_lastframe(
            target_df,
            target_colnames_mean[1],
            target_colnames_std[1],
            experiment_metriccsv_paths,
            experiment_colnames[1],
        )
        row["area-lastframe-z-test-pvalue"] = ztest_pval_lastframe

        # radial velocity
        z, zlast = evaluate_zscore(
            target_df,
            target_colnames_mean[2],
            target_colnames_std[2],
            experiment_metriccsv_paths,
            experiment_colnames[2],
            also_return_only_last_frame=True,
        )
        row["radial-velocity-zscore"] = z
        row["radial-velocity-zscore-mean"] = sum(z) / len(z)
        row["radial-velocity-lastframe-zscore"] = zlast
        row["radial-velocity-lastframe-zscore-mean"] = sum(zlast) / len(zlast)

        xi2, xi2last = evaluate_corrected_xi2_score(
            target_df,
            target_colnames_mean[2],
            target_colnames_std[2],
            experiment_metriccsv_paths,
            experiment_colnames[2],
            target_N,
            also_return_only_last_frame=True,
        )
        row["radial-velocity-corxi2-pval"] = xi2
        row["radial-velocity-lastframe-corxi2-pval"] = xi2last

        ttest_pval_lastframe = evaluate_t_test_lastframe(
            target_df,
            target_colnames_mean[2],
            experiment_metriccsv_paths,
            experiment_colnames[2],
        )
        row["radial-velocity-lastframe-t-test-pvalue"] = ttest_pval_lastframe

        ztest_pval_lastframe = evaluate_z_test_lastframe(
            target_df,
            target_colnames_mean[2],
            target_colnames_std[2],
            experiment_metriccsv_paths,
            experiment_colnames[2],
        )
        row["radial-velocity-lastframe-z-test-pvalue"] = ztest_pval_lastframe

        # Circularity
        circularity = evaluate_circularity_lastframe(
            experiment_metriccsv_paths, exp_colname="circularity"
        )
        row["circularity-lastframe"] = circularity

        # END of EVALUATION SCORES

        eval_results.append(row)

    eval_df = pd.DataFrame(eval_results, index=None)

    # Add sum of the scores
    add_normalized_sum_to_df(
        eval_df,
        new_colname="zscores-normalized-sum",
        colnames_to_sum=[
            "infected-count-zscore-mean",
            "area-zscore-mean",
            "radial-velocity-zscore-mean",
        ],
    )
    add_normalized_sum_to_df(
        eval_df,
        new_colname="zscores-lastframe-normalized-sum",
        colnames_to_sum=[
            "infected-count-lastframe-zscore-mean",
            "area-lastframe-zscore-mean",
            "radial-velocity-lastframe-zscore-mean",
        ],
    )
    return eval_df


def get_metric_paths_for_experiment(exp_path):
    metric_paths = []
    for root, dirs, files in os.walk(exp_path):
        if "metric.csv" in files:
            metric_paths.append(os.path.join(root, "metric.csv"))
    return metric_paths


def evaluate_zscore(
    target_df,
    target_df_mean_colname,
    target_df_std_colname,
    metriccsv_paths,
    exp_colname,
    also_return_only_last_frame=True,
):
    zscores = []
    zscores_only_last_frame = []
    for csv in metriccsv_paths:
        exp_df = pd.read_csv(csv)
        exp_df.set_index(time_colname, inplace=True)
        aligned_df = pd.merge(
            exp_df, target_df, left_index=True, right_index=True, how="inner"
        )
        aligned_df.dropna()
        zscore_all = (
            aligned_df[exp_colname] - aligned_df[target_df_mean_colname]
        ).abs() / aligned_df[target_df_std_colname]
        zscores.append(zscore_all.sum())
        if also_return_only_last_frame:
            zscores_only_last_frame.append(zscore_all.iloc[-1])
    if also_return_only_last_frame:
        return zscores, zscores_only_last_frame
    else:
        return zscores


def evaluate_corrected_xi2_score(
    target_df,
    target_df_mean_colname,
    target_df_std_colname,
    metriccsv_paths,
    exp_colname,
    target_N,
    also_return_only_last_frame=True,
):
    df_list = []
    exp_N = len(metriccsv_paths)
    for csv in metriccsv_paths:
        df = pd.read_csv(csv, index_col=time_colname)
        df_list.append(df[exp_colname])
    concatenated_df = pd.concat(df_list, axis=1)
    mean_df = concatenated_df.mean(axis=1)
    std_df = concatenated_df.std(axis=1)
    exp_df = pd.DataFrame({"mean": mean_df, "std": std_df})
    exp_df.index.name = time_colname
    # exp_df.set_index(time_colname, inplace=True)
    target_df = target_df[[target_df_mean_colname, target_df_std_colname]]
    aligned_df = pd.merge(
        exp_df, target_df, left_index=True, right_index=True, how="inner"
    )
    aligned_df.dropna()
    aligned_df["corrected_std_exp"] = aligned_df["std"] / np.sqrt(exp_N)
    aligned_df["corrected_std_target"] = aligned_df[target_df_std_colname] / np.sqrt(
        target_N
    )
    aligned_df["abs_mean_diff"] = np.abs(
        aligned_df["mean"] - aligned_df[target_df_mean_colname]
    )
    aligned_df["std_diff"] = np.sqrt(
        (aligned_df["corrected_std_exp"] ** 2)
        + (aligned_df["corrected_std_target"] ** 2)
    )
    aligned_df["t-value"] = aligned_df["abs_mean_diff"] / aligned_df["std_diff"]
    N_small = exp_N if exp_N < target_N else target_N
    dF = 2 * N_small - 2

    aligned_df["probs"] = 2 * (
        1 - stats.t.cdf(aligned_df["t-value"], dF)
    )  # survivala function 1-cdf for t-distribution multiplied by two for two tailed computation

    aligned_df["corrected_xi2"] = stats.chi2.ppf(1 - aligned_df["probs"], 1)
    lastframe_value = aligned_df.iloc[-1][
        "probs"
    ]  # returning the probability value p-value

    sum_xi2 = aligned_df["corrected_xi2"].sum()  # T3
    num_timepoints = len(aligned_df)
    right_tail_probability_of_all_timepoints = 1 - stats.chi2.cdf(
        sum_xi2, num_timepoints
    )  # this corresponds to T4
    if also_return_only_last_frame:
        return right_tail_probability_of_all_timepoints, lastframe_value
    else:
        return right_tail_probability_of_all_timepoints


def evaluate_t_test_lastframe(
    target_df, target_df_mean_colname, metriccsv_paths, exp_colname
):
    exp_df_list = []
    for csv in metriccsv_paths:
        exp_df = pd.read_csv(csv, index_col=time_colname)
        exp_df_list.append(exp_df[exp_colname])
    concatenated_exp_df = pd.concat(exp_df_list, axis=1)
    aligned_df = pd.merge(
        concatenated_exp_df, target_df, left_index=True, right_index=True, how="inner"
    )
    aligned_df.dropna()
    aligned_df_lastframe = aligned_df.iloc[-1]
    values = aligned_df_lastframe[exp_colname].values
    t_stat, p_value = stats.ttest_1samp(
        values, popmean=aligned_df_lastframe[target_df_mean_colname]
    )
    return p_value


def evaluate_z_test_lastframe(
    target_df,
    target_df_mean_colname,
    target_df_std_colname,
    metriccsv_paths,
    exp_colname,
):
    exp_df_list = []
    for csv in metriccsv_paths:
        exp_df = pd.read_csv(csv, index_col=time_colname)
        exp_df_list.append(exp_df[exp_colname])
    concatenated_exp_df = pd.concat(exp_df_list, axis=1)
    aligned_df = pd.merge(
        concatenated_exp_df, target_df, left_index=True, right_index=True, how="inner"
    )
    aligned_df.dropna()
    aligned_df_lastframe = aligned_df.iloc[-1]
    values = aligned_df_lastframe[exp_colname].values
    n = len(values)
    target_mean = aligned_df_lastframe[target_df_mean_colname]
    target_std = aligned_df_lastframe[target_df_std_colname]
    z_score = (values.mean() - target_mean) / (target_std / np.sqrt(n))
    p_value = 1 - stats.norm.cdf(abs(z_score))
    return p_value


def evaluate_circularity_lastframe(metriccsv_paths, exp_colname="circularity"):
    circularity = []
    for csv in metriccsv_paths:
        exp_df = pd.read_csv(csv, index_col=time_colname)
        circularity.append(exp_df.iloc[-1][exp_colname])
    return circularity


def add_normalized_sum_to_df(df, new_colname, colnames_to_sum):
    assert all([col in df.columns for col in colnames_to_sum])
    max_values = {col: df[col].max() for col in colnames_to_sum}
    normalized_cols = {col: df[col] / max_val for col, max_val in max_values.items()}
    df_normalized = pd.DataFrame(normalized_cols)
    normalized_sum = df_normalized.sum(axis=1)
    df[new_colname] = normalized_sum
    return df


def visualize_bulk_evaluation_results(eval_results: pd.DataFrame):

    df = replace_experiment_name_with_params(eval_results)

    def is_numeric_and_not_list(column):
        # Check if the column is numeric and does not contain lists
        return pd.api.types.is_numeric_dtype(column) and not any(
            isinstance(i, list) for i in column
        )

    numeric_columns = [c for c in df.columns if is_numeric_and_not_list(df[c])]
    fig = go.Figure(
        data=go.Parcoords(
            line=dict(
                color=df["zscores-lastframe-normalized-sum"],
                colorscale="Blackbody",
                showscale=True,
            ),
            dimensions=[
                dict(values=df[c], label=c, tickvals=df[c].unique())
                for c in numeric_columns
            ],
            # + [dict(values=df[c], label=c) for c in df.columns if "-dist" in c],
            # unselected=dict(line=dict(opacity=0)),
        ),
    )
    return fig


def replace_experiment_name_with_params(df, column_name="experiment_name"):
    return pd.concat(
        [
            pd.json_normalize(
                df["experiment_name"].apply(extract_parameters_from_name)
            ),
            df.drop(["experiment_name"], axis=1),
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
    parser.add_argument("--target_csv", type=str, help="path to target csv file")
    parser.add_argument(
        "--root",
        type=str,
        help="path to simulation results where metric.csv can be found.",
    )
    parser.add_argument(
        "--n_dataset",
        type=int,
        help="Number of measured points of the reference dataset used for Corrected standardard deviation",
    )
    parser.add_argument("--output", type=str, help="output save file.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if os.path.exists(args.output):
        print("-- Output already exists.")
        exit(1)

    df = evaluate_experiments(args.root, args.target_csv, args.n_dataset)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output)
    print("-- Evaluation results saved to: ", args.output)

    dist_fig = visualize_bulk_evaluation_results(df)

    from plotly.offline import plot

    plot(
        dist_fig,
        filename=os.path.join(
            os.path.dirname(args.output),
            "dist_bulk_eval_parallel_coordinate_plotly.html",
        ),
    )
