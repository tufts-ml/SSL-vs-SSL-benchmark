import os
import csv
import pickle
from PIL import Image

LABELED = False

# Input Pickle File
pkl_file = "/cluster/tufts/hugheslab/datasets/AIROGS/train.pkl"

# Output Paths
output_folder = "/cluster/tufts/hugheslab/datasets/AIROGS/Images"
csv_file = "/cluster/tufts/hugheslab/datasets/AIROGS/train.csv"

# Create the Images directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Load the pickle file
with open(pkl_file, 'rb') as f:
    data = pickle.load(f)

# Extract images (ignore labels)
images = data['images']  # NumPy array of images

# Save images and filenames
with open(csv_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["filename"])  # CSV header (no labels)

    for idx, image in enumerate(images):
        img_filename = f"train_{idx:05d}.png"  # Example: 'train_00000.png'
        img_path = os.path.join(output_folder, img_filename)

        # Convert NumPy array to a PIL image
        image = Image.fromarray(image)
        
        if LABELED:
            image_label = data['labels'][idx]

        # Save image
        image.save(img_path)

        # Write filename to CSV
        writer.writerow([img_filename, image_label] if LABELED else [img_filename])

print(
    f"✅ Reformatted dataset: {len(images)} unlabeled images saved in '{output_folder}/' and filenames in '{csv_file}'.")
