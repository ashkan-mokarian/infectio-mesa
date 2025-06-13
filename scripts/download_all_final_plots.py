import os
import shutil


def collect_plot_final(root_dir, destination_folder):
    print("hello")
    if not os.path.exists(destination_folder):
        print("create")
        os.makedirs(destination_folder)

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            print(f"dirpath={dirpath}, dirname={dirnames}, filename={filename}")
            if filename == "plot_final.png":
                # Identify the parameter folder and subfolder
                parts = dirpath.split(os.sep)
                param_folder = parts[-2] if len(parts) > 1 else parts[-1]
                subfolder = parts[-1] if len(parts) > 1 else ""

                # Construct the new filename
                new_filename = (
                    f"{param_folder}-{subfolder}.png"
                    if subfolder
                    else f"{param_folder}.png"
                )
                destination_path = os.path.join(destination_folder, new_filename)

                # Avoid overwriting files
                if os.path.exists(destination_path):
                    count = 1
                    while os.path.exists(
                        f"{destination_folder}/{param_folder}-{subfolder}-{count}.png"
                    ):
                        count += 1
                    destination_path = (
                        f"{destination_folder}/{param_folder}-{subfolder}-{count}.png"
                    )

                # Copy the file
                shutil.copy2(os.path.join(dirpath, filename), destination_path)
                print(f"Copied: {filename} to {destination_path}")


# Example usage
root_directory = "/home/mokari27/workspace/infectio-mesa/output/dVGF/wide_range/"  # Replace with your actual root directory path
destination_folder = "/home/mokari27/workspace/infectio-mesa/output/download/"  # Replace with your desired destination folder
collect_plot_final(root_directory, destination_folder)
