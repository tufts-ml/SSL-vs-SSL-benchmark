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
        self.labels = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.label_columns = self.labels.columns[5:]

    def __len__(self):
        # Using shape[0] for explicit row count
        return self.labels.shape[0]

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.item()

        img_path = self.labels.iloc[index, 0]

        img_path = img_path.replace('view1_frontal.jpg', 'viewfrontal.jpg.jpg')
        img_name = os.path.join(self.root_dir, img_path)

        if not os.path.exists(img_name):
            print(f"File not found: {img_name}")
            return None, None
        
        image = io.imread(img_name)

        labels = self.labels.iloc[index, 5:] 
        
        
        labels = labels.fillna(0)  # Replace NaN values with 0
       
        labels = labels.astype(np.int32).values

        if self.transform is not None:
            return self.transform(image), labels
           
        return image, labels

