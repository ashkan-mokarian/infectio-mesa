import pandas as pd

# Replace 'your_file.csv' with the path to your CSV file
file_path = "/Users/ashkanhzdr/workspace/infectio-mesa/output/dVGFdF11/1906/0_evaluation/bulk_evaluate.csv"

# Load the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Adjust display settings to show full column names
pd.set_option("display.max_colwidth", None)

# Sort the DataFrame by the 'Infected_Count-dist' column in ascending order
df_sorted = df.sort_values(by="Sum-dist")

# Get the top 5 rows with the smallest values in 'Infected_Count-dist'
top_5_smallest = df_sorted.head(5)

# Display the result
print(top_5_smallest)
