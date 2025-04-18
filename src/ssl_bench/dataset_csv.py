import os
import torch
import pandas as pd
from skimage import io
import numpy as np
from torchvision.datasets import VisionDataset
from PIL import Image


class LabeledImageCSVDataset(VisionDataset):
    """Dataset class for loading images from a CSV file.

    Args:
        csv_file (str): Path to the CSV file with image paths and labels.
        root_dir (str): Directory with all the images.
        transforms (callable, optional): Optional transform to be applied on a
            sample.
        transform (callable, optional): Optional transform to be applied on an
            image.
        target_transform (callable, optional): Optional transform to be applied
            on a label.
    """

    def __init__(self, csv_file, root_dir, transforms=None, transform=None,
                 target_transform=None):
        super().__init__(root=root_dir, transforms=transforms,
                         transform=transform,
                         target_transform=target_transform)
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def get_image(self, index):
        if torch.is_tensor(index):
            index = index.item()

        img_name = os.path.join(self.root_dir, self.data.iloc[index, 0])
        image = io.imread(img_name)

        # Convert the image to PIL format if it’s not already
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

    def __getitem__(self, index):
        image = self.get_image(index)

        rows, cols = self.data.shape

        # Working with labeled data
        if cols > 1:
            # Assuming single-class label at index 1
            label = self.data.iloc[index, 1]
            label = torch.tensor(label, dtype=torch.long)

            # Return as a tuple (image, label) for compatibility
            if self.transforms is not None:
                return self.transforms(image, label)
            return (image, label)

        # Working with unlabeled data
        else:
            if self.transform is not None:
                image = self.transform(image)  # Apply image transformations
            return image


class UnlabeledImageCSVDataset(LabeledImageCSVDataset):
    """Dataset class for loading images from a CSV file without labels.

    Args:
        csv_file (str): Path to the CSV file with image paths.
        root_dir (str): Directory with all the images.
        transform (callable, optional): Optional transform to be applied on an
            image.
    """

    def __init__(self, csv_file, root_dir, transform=None):
        super().__init__(csv_file, root_dir, transform=transform)


class CheXpertDataset(LabeledImageCSVDataset):
    """Dataset class for loading images from the CheXpert dataset.

    Args:
        csv_file (str): Path to the CSV file with image paths and labels.
        root_dir (str): Directory with all the images.
        transforms (callable, optional): Optional transform to be applied on a
            sample.
        transform (callable, optional): Optional transform to be applied on an
            image.
        target_transform (callable, optional): Optional transform to be applied
            on a label.
    """

    def __init__(self, csv_file, root_dir, transforms=None, transform=None,
                 target_transform=None):
        super().__init__(root=root_dir, transforms=transforms,
                         transform=transform,
                         target_transform=target_transform)
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.label_columns = ['Edema', 'Consolidation',
                              'Atelectasis', 'Pneumothorax',
                              'Pleural Effusion']

    def __getitem__(self, index):
        image = self.get_image(index)
        labels = self.data.loc[index, self.label_columns]
        labels = labels.fillna(0)  # Replace NaN values with 0
        labels = labels.astype(np.int32).values

        if self.transform is not None:
            return self.transform(image), labels
        return image, labels
