import os
import torch
import pandas as pd
from skimage import io
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class data(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        print("csv_file: ", csv_file)
        print("root_dir: ", root_dir)
        self.labels = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        img_name = os.path.join(self.root_dir, self.labels.iloc[index, 0])
        image = io.imread(img_name + '.jpg')

        #TODO: Update this to work for multi-class label data
        labels = self.labels.iloc[index, 1] 
        labels = np.array([labels], dtype=np.float32)
        sample = {'image': image, 'label': labels}

        if self.transform:
            sample = self.transform(sample)
        
        return sample