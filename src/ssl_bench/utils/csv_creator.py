# For a folder of images, create a CSV file with the filenames.

import os
import csv
from tqdm import tqdm

def create_csv_from_images(image_folder, csv_file):
        """
        Create a CSV file from images in a folder.
        
        Parameters:
        - image_folder: Path to the folder containing images.
        - csv_file: Path to the output CSV file.
        """
        # Create the CSV file and write the header
        with open(csv_file, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["filename"])  # CSV header
        
                # Iterate through all files in the image folder
                for filename in tqdm(os.listdir(image_folder)):
                        if filename.endswith(('.png', '.jpg', '.jpeg')):  # Check for image files
                                writer.writerow([filename])  # Write filename to CSV
        print(f"✅ CSV file '{csv_file}' created with image filenames from '{image_folder}'.")
        return csv_file

if __name__ == "__main__":
        # Example usage
        image_folder = "/cluster/tufts/hugheslab/datasets/AIROGS/Images_0/0"  # Replace with your image folder path
        csv_file = "/cluster/tufts/hugheslab/datasets/AIROGS/train_0.csv"  # Replace with your desired output CSV file path

        create_csv_from_images(image_folder, csv_file)
        