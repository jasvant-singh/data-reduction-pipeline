#!/usr/bin/env python3
import os
import shutil

# Get the directory where this script is located: /home/jasvant/data-reduction/source_code
script_dir = os.path.dirname(os.path.abspath(__file__))

# Go one level up: /home/jasvant/data-reduction
project_root = os.path.dirname(script_dir)

# ~ # Read the input path from paths.txt (expected to be in project_root/)
# ~ txt_path = os.path.join(project_root, 'paths.txt')

# ~ with open(txt_path, 'r') as f:
    # ~ for line in f:
        # ~ if line.startswith("Input-data-directory"):
            # ~ source_dir = line.split("Input-data-directory", 1)[1].strip()
            # ~ break
    # ~ else:
        # ~ raise ValueError("No line starting with 'Input-data-directory' found.")
source_dir = os.path.join(project_root, 'input')
# Destination path: /home/jasvant/data-reduction/intermediate
destination_dir = os.path.join(project_root, 'intermediate')

# Classification logic
def classify_file(filename):
    lower = filename.lower()
    if 'bias' in lower:
        return 'bias'
    elif 'dark' in lower:
        return 'dark'
    elif 'calib' in lower:
        return 'calib'
    elif 'sky' in lower:
        return 'sky'
    else:
        return 'science'

# Sort files from input → intermediate/<category>
for filename in os.listdir(source_dir):
    file_path = os.path.join(source_dir, filename)

    if os.path.isfile(file_path):
        category = classify_file(filename)
        category_dir = os.path.join(destination_dir, category)
        os.makedirs(category_dir, exist_ok=True)

        dest_path = os.path.join(category_dir, filename)
        shutil.copy2(file_path, dest_path)  # use shutil.move(...) to move instead
        print(f"Copied: {filename} → {category}/")
