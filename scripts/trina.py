import os
import pandas as pd
from collections import defaultdict
from pathlib import Path

ROOT = Path(
    "/Users/ashkanhzdr/workspace/infectio-mesa/output/top10_mae_minmaxnormalized"
)  # Replace with actual path
SAVE_ROOT = Path(
    "/Users/ashkanhzdr/workspace/infectio-mesa/output/trina_csv"
)  # Replace with actual path
SAVE_ROOT.mkdir(parents=True, exist_ok=True)

COLUMN_MAP = {
    "infected-count": "inf-count-list",
    "area(um2)": "area-list(um2)",
    "radial-velocity(um/min)": "radial-velocity-list(um/min)",
}


def merge_metrics(exp_dir):
    subfolders = [f for f in exp_dir.iterdir() if f.is_dir()]
    merged_data = defaultdict(lambda: defaultdict(list))
    time_order = None
    success = False

    print(f"Running for: {exp_dir}")
    num_samples = 0
    for sub in subfolders:
        metric_path = sub / "metric.csv"
        if not metric_path.exists():
            print(f"[WARN] Missing: {metric_path}")
            continue

        try:
            df = pd.read_csv(metric_path)
        except Exception as e:
            print(f"[WARN] Could not read {metric_path}: {e}")
            continue

        if "t" not in df.columns:
            print(f"[WARN] No 't' column in {metric_path}")
            continue

        if time_order is None:
            time_order = df["t"].tolist()
        elif df["t"].tolist() != time_order:
            print(f"[WARN] Mismatched 't' in {metric_path}")
            continue

        for _, row in df.iterrows():
            t = row["t"]
            for col in df.columns:
                if col == "t":
                    continue
                merged_data[t][col].append(row[col])

        success = True
        num_samples += 1

    if not success or time_order is None:
        print(f"[INFO] No valid data to merge in: {exp_dir}")
        return
    print(f"number of samples: {num_samples}")

    result_rows = []
    for t in time_order:
        row = {"t": t}
        for col, values in merged_data[t].items():
            new_col = COLUMN_MAP.get(col, col + "-list")
            row[new_col] = values
        result_rows.append(row)

    result_df = pd.DataFrame(result_rows)
    save_path = SAVE_ROOT / f"{exp_dir.name}.csv"
    result_df.to_csv(save_path, index=False)
    print(f"[OK] Saved: {save_path}")


# Main loop
for exp_dir in ROOT.iterdir():
    if exp_dir.is_dir():
        merge_metrics(exp_dir)
