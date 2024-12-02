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
        self.label_columns = ['Atelectasis']

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.item()

        img_path = self.labels.iloc[index, 0]

        #img_path = img_path.replace('view1_frontal.jpg', 'viewfrontal.jpg.jpg')
        img_name = os.path.join(self.root_dir, img_path)

     

        if not os.path.exists(img_name):
            print(f"File not found: {img_name}")
            return None
        
        image = io.imread(img_name)
        
        if image is None:
            print(f"Error loading image: {img_name}")
            return None
            
        labels = self.labels.loc[index, self.label_columns] 
        
        pd.set_option('future.no_silent_downcasting', True)
        labels = labels.fillna(0).infer_objects()   # Replace NaN values with 0

        labels[labels == -1] = 1
        labels = labels.astype(np.int32).values

        if len(self.label_columns) == 1:
            labels = labels[0] 
        labels = torch.tensor(labels, dtype=torch.long)



        if self.transform is not None:
            return self.transform(image), labels

           
        return image, labels

