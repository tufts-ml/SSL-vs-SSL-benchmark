import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class CIFAR100Dataset(Dataset):
    def __init__(self, file_path, transform=None, target_transform=None, use_coarse_labels=False):
        """
        Args:
            file_path (str): Path to the CIFAR-100 dataset file.
            transform (callable, optional): Optional transform to be applied on an image.
            target_transform (callable, optional): Optional transform to be applied on a label.
            use_coarse_labels (bool): Whether to use coarse labels instead of fine labels.
        """
        self.file_path = file_path
        self.transform = transform
        self.target_transform = target_transform
        self.use_coarse_labels = use_coarse_labels

        # Load data
        self.data, self.labels = self._load_data()

    def _unpickle(self, file):
        """Load CIFAR-100 dataset from pickle file."""
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    def _load_data(self):
        """Extracts image data and labels from the dataset file."""
        dataset_dict = self._unpickle(self.file_path)
        data = dataset_dict[b'data']  # Shape: (50000, 3072)
        if self.use_coarse_labels:
            labels = dataset_dict[b'coarse_labels']
        else:
            labels = dataset_dict[b'fine_labels']

        # Reshape data to (50000, 3, 32, 32)
        data = data.reshape(-1, 3, 32, 32)  # Convert to channel-first format for PyTorch

        return data, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.data[index]  # Shape: (3, 32, 32)
        label = self.labels[index]

        # Convert numpy array to PIL Image
        image = Image.fromarray(np.transpose(image, (1, 2, 0)))  # Convert (C, H, W) → (H, W, C)

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, torch.tensor(label, dtype=torch.long)
