import os
import torch
import pandas as pd
from skimage import io
import numpy as np
from torchvision.datasets import VisionDataset
from PIL import Image


class ImageCSVDataset(VisionDataset):
    def __init__(self, csv_file, root_dir, transforms=None, transform=None, target_transform=None):
        super().__init__(root=root_dir, transforms=transforms, transform=transform,
                         target_transform=target_transform)
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.item()

        img_name = os.path.join(self.root_dir, self.data.iloc[index, 0])
        image = io.imread(img_name)
        # Convert the image to PIL format if it’s not already
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Assuming single-class label at index 1
        label = self.data.iloc[index, 1]
        label = np.int32(label)

        # Return as a tuple (image, label) for compatibility
        if self.transforms is not None:
            return self.transforms(image, label)
           
        return (image, label)


class UnlabeledImageCSVDataset(ImageCSVDataset):
    def __init__(self, csv_file, root_dir, transforms=None, transform=None, target_transform=None):
        super().__init__(csv_file, root_dir, transforms, transform, target_transform)

    # TODO can the below two functions be inherited or use super function calls?
    # should be possible if the CSV is still expected to have 2 columns
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.item()

        img_name = os.path.join(self.root_dir, self.data.iloc[index, 0])
        image = io.imread(img_name)

        # Convert the image to PIL format if it’s not already
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Return data
        if self.transforms is not None:
            return self.transforms(image)

        return image


class CheXpertDataset(VisionDataset):
    # TODO how could inheritance reduce the amount of code in these first 2 functions?
    def __init__(self, csv_file, root_dir, transforms=None, transform=None, target_transform=None):
        super().__init__(root=root_dir, transforms=transforms, transform=transform,
                         target_transform=target_transform)
        self.labels = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.label_columns = ['Edema', 'Consolidation', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion']


    def __len__(self):
        return self.labels.shape[0]


    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.item()

        img_path = self.labels.iloc[index, 0]
        img_name = os.path.join(self.root_dir, img_path)

        if not os.path.exists(img_name):
            print(f"File not found: {img_name}")
            return None, None

        image = io.imread(img_name)
        labels = self.labels.loc[index, self.label_columns]
        labels = labels.fillna(0)  # Replace NaN values with 0
        labels = labels.astype(np.int32).values

        if self.transform is not None:
            return self.transform(image), labels
        return image, labels
