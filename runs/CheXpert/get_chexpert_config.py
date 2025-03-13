import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

IMAGE_PATH = "/cluster/tufts/hugheslab/datasets/chexpert_sample_data/images"
LABEL_PATH = "/cluster/tufts/hugheslab/datasets/chexpert_sample_data/sampled_train.csv"

all_images = pd.read_csv(LABEL_PATH)

# Drop samples with -1 label instead of casting them to 1
all_images = all_images[all_images['Pleural Effusion'] != -1]

image_paths = all_images['Path']

# Only Pleural Effusion labels
labels = all_images['Pleural Effusion'].fillna(0).astype(np.int32).values  # Replace NaN values with 0

pixel_sum = np.zeros(3)
pixel_squared_sum = np.zeros(3)
pixel_count = 0
class_counts = np.zeros(5)

for path, label in tqdm(zip(image_paths, labels), total=len(image_paths)):
    path = IMAGE_PATH + '/' + path
    image = Image.open(path).convert('RGB')
    img_array = np.array(image, dtype=np.float32) / 255.0

    pixel_sum += img_array.sum(axis=(0, 1))
    pixel_squared_sum += np.square(img_array).sum(axis=(0, 1))
    pixel_count += img_array.shape[0] * img_array.shape[1]

    class_counts[label] += 1

dataset_mean = pixel_sum / pixel_count
dataset_std = np.sqrt(pixel_squared_sum / pixel_count - dataset_mean ** 2)

total_samples = len(labels)
class_weights = [total_samples / count if count > 0 else 0 for count in class_counts]
class_weights = np.array(class_weights) / np.sum(class_weights)

config = {
    'CheXpert': {
        'dataset_mean': tuple(dataset_mean),
        'dataset_std': tuple(dataset_std),
        'image_size': (320, 390),
        'class_weights': class_weights.tolist(),
        'nimg_per_epoch': 3500,
        'num_classes': 2
    }
}

print(config)
