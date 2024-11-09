import os
import torch
import pandas as pd
from skimage import io
import numpy as np
from torchvision.datasets import VisionDataset


class ImageCSVDataset(VisionDataset):
    def __init__(self, csv_file, root_dir, transforms=None, transform=None, target_transform=None):
        super().__init__(root=root_dir, transforms=transforms, transform=transform,
                         target_transform=target_transform)
        # TODO: rename labels later
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        # Using shape[0] for explicit row count
        return self.data.shape[0]

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.item()

        img_name = os.path.join(self.root_dir, self.data.iloc[index, 0])
        image = io.imread(img_name)

        # Assuming single-class label at index 1
        label = self.data.iloc[index, 1]
        label = np.int32(label)

        # Return as a tuple (image, label) for compatibility
        if self.transforms is not None:
            return self.transforms(image, label)
           
        return (image, label)
