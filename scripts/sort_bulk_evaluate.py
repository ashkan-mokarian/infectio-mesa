import pandas as pd
import os

# Replace 'your_file.csv' with the path to your CSV file
root_path = "/Users/ashkanhzdr/workspace/infectio-mesa/output/dVGFdF11"
file_path = os.path.join(root_path, "1907/0_evaluation/bulk_evaluate.csv")
save_path = os.path.join(root_path, "1907/0_evaluation/top")
os.makedirs(save_path, exist_ok=True)

# Load the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Adjust display settings to show full column names
pd.set_option("display.max_colwidth", None)

# Sort the DataFrame by the 'Infected_Count-dist' column in ascending order
dist_cols = [col for col in df.columns if "-dist" in col]
for col in dist_cols:
    print(f"Sorting by {col}:")
    df_sorted = df.sort_values(by=col)
    print(df_sorted.head(5))

    # copy the folders to a new location
    tops = df_sorted.head(5)["target_folder"].tolist()
    for top in tops:
        src = os.path.join(root_path, top)
        dst = os.path.join(save_path, top)
        os.system(f"cp -r {src} {dst}")
