import os
import pandas as pd
import ast

# Constants
CSV_ROOT = "/Users/ashkanhzdr/workspace/infectio-mesa/output/trina_csv"
NUM_EXPERIMENTS = 100  # <-- Set your expected number
PARAM_COMBINATIONS = "wide_make_100_param_combinations.txt"

# Clear the file before writing
with open(PARAM_COMBINATIONS, "w") as f:
    pass

# Process each CSV file
for filename in os.listdir(CSV_ROOT):
    if filename.endswith(".csv"):
        path = os.path.join(CSV_ROOT, filename)
        try:
            df = pd.read_csv(path)
            if "inf-count-list" not in df.columns:
                continue
            # Parse first valid row
            lengths = (
                df["inf-count-list"].dropna().apply(lambda x: len(ast.literal_eval(x)))
            )
            if not lengths.empty:
                actual_len = lengths.iloc[0]
                if actual_len < NUM_EXPERIMENTS:
                    base_name = os.path.splitext(filename)[0]
                    missing = NUM_EXPERIMENTS - actual_len
                    with open(PARAM_COMBINATIONS, "a") as f:
                        for _ in range(missing):
                            f.write(base_name + "\n")
                    print(f"{base_name}: {missing}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")
