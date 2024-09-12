import os
from dotenv import load_dotenv

# Replace 'your_file.csv' with the path to your CSV file
load_dotenv()
PROJECT_PATH = os.getenv("PROJECT_PATH")
root_path = os.path.join(PROJECT_PATH, "output/dVGFdF11")
save_path = os.path.join(PROJECT_PATH, "output", "download")
folders_list = []

if os.path.exists(save_path):
    print("Save path already exists")
    exit(1)
os.makedirs(save_path)

for f in folders_list:
    src = os.path.join(root_path, f)
    dst = os.path.join(save_path, f)
    os.system(f"cp -r {src} {dst}")
