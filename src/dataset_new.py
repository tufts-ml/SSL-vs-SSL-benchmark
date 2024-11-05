import os
import torch
import pandas as pd
from skimage import io
import numpy as np
from torch.utils.data import VisionDataset

class CustomDataset(VisionDataset):
    def __init__(self, csv_file, root_dir, transform=None):
        super().__init__(root=root_dir, transform=transform)
        self.labels = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        # Using shape[0] for explicit row count
        return self.labels.shape[0]
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.item()

        img_name = os.path.join(self.root_dir, self.labels.iloc[index, 0])
        image = io.imread(img_name + '.jpg')

        # Assuming single-class label at index 1
        label = self.labels.iloc[index, 1]
        label = np.float32(label)  # Convert label to float32 for PyTorch compatibility

        if self.transform:
            image = self.transform(image)

        # Return as a tuple (image, label) for compatibility
        return image, label
