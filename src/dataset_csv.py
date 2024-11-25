import os
import torch
import pandas as pd
from torchvision.datasets import VisionDataset
from PIL import Image


class ImageCSVDataset(VisionDataset):
    def __init__(self, csv_file, root_dir, transforms=None, transform=None, target_transform=None):
        """Initializes a dataset from a CSV file containing image paths and labels.

        Args:
            csv_file (_type_): _description_
            root_dir (_type_): _description_
            transforms (_type_, optional): _description_. Defaults to None.
            transform (_type_, optional): _description_. Defaults to None.
            target_transform (_type_, optional): _description_. Defaults to None.
        """
        super().__init__(root=root_dir, transforms=transforms, transform=transform,
                         target_transform=target_transform)
        self.root_dir = root_dir
        self.transform = transform

        dataframe = pd.read_csv(csv_file)
        self.img_paths = dataframe.iloc[:, 0]
        self.labels = dataframe.iloc[:, 1]

    def __len__(self):
        """Returns the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.img_paths)

    def __getitem__(self, index):
        """Returns the image and label at the specified index.

        Args:
            index (int): The index of the sample to return.

        Returns:
            tuple: A tuple containing the image and label.
        """
        if torch.is_tensor(index):
            index = index.item()

        # Load image from file
        img_name = os.path.join(self.root_dir, self.img_paths[index])
        image = Image.open(img_name)
        # Select label
        label = int(self.labels[index])

        # Return as a tuple (image, label) for compatibility
        if self.transforms is not None:
            return self.transforms(image, label)

        return (image, label)
