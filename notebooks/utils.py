import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# This function handles the cases of lists when reading from a csv file where lists with multiple values are read as strings but with one value come without double quptions
def parse_list_string(val):
    if isinstance(val, str):
        val = val.strip("[]")  # Remove square brackets
        return [float(x) for x in val.split(",")] if val else []
    else:
        return val


def read_and_clean_df(eval_csv):
    df = pd.read_csv(eval_csv, index_col=0)
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.max_columns", None)

    df["infected-count-zscore"] = df["infected-count-zscore"].apply(parse_list_string)
    df["infected-count-lastframe-zscore"] = df["infected-count-lastframe-zscore"].apply(
        parse_list_string
    )
    df["area-zscore"] = df["area-zscore"].apply(parse_list_string)
    df["area-lastframe-zscore"] = df["area-lastframe-zscore"].apply(parse_list_string)
    df["radial-velocity-zscore"] = df["radial-velocity-zscore"].apply(parse_list_string)
    df["circularity-lastframe"] = df["circularity-lastframe"].apply(parse_list_string)

    df = df[df["infected-count-zscore"].apply(lambda x: len(x) == 5)]

    # also add some handy columns to the dataframe
    df.loc[:, "circularity-lastframe-mean"] = df["circularity-lastframe"].apply(
        lambda x: sum(x) / len(x)
    )
    return df


def add_param_cols_from_experiment_name(df):
    for index, row in df.iterrows():
        experiment_name = row["experiment_name"]

        param_vals = experiment_name.split("-")

        for p_v in param_vals:
            p = p_v.split("=")[0].strip()
            v = p_v.split("=")[1].strip()

            # Add new columns to the dataframe with the extracted parameters
            if not df.columns.tolist().count(p):
                df[p] = 0.0

            df.loc[index, p] = float(v)
    return df


def create_dotted_plot(df, param_name, score_name):
    plt.figure(figsize=(10, 6))

    for i, score in enumerate(score_name):
        df.plot.scatter(x=param_name, y=score, marker="o", label=score, alpha=0.5)

    plt.title("Dotted Plot")
    plt.xlabel(param_name)
    plt.ylabel(score_name[0])
    plt.legend(loc="upper right")

    plt.show()


def dotted_plot_overlaying_scores(df, param_name, score_names, xlim=None):
    plt.figure(figsize=(10, 6))

    # Generate a list of colors for the different scores
    colors = plt.cm.viridis(np.linspace(0, 1, len(score_names)))

    for i, score in enumerate(score_names):
        plt.scatter(df[param_name], df[score], color=colors[i], label=score, alpha=0.5)

    # Set x-axis range if provided
    if xlim:
        plt.xlim(xlim)

    plt.title("Overlayed Score Plot")
    plt.xlabel(param_name)
    plt.ylabel("Scores")
    plt.legend(loc="upper right")

    plt.show()
