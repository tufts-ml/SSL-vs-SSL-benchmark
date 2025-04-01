import numpy as np
import os
from PIL import Image

# from torchvision import transforms

# from .randaugment import RandAugmentMC



class data:
    def __init__(self, dataset_name, dataset_path, transform_fn=None):
        if dataset_name in ['TissueMNIST','PathMNIST']:
            self.dataset = np.load(dataset_path, allow_pickle=True).item() #need to use HWC version of the data
        elif dataset_name == 'TMED2':
            self.dataset = np.load(dataset_path, allow_pickle=True) #need to use HWC version of the data
        else:
            raise NameError('Check')
            
        self.transform_fn = transform_fn
        
    def __getitem__(self, idx):
        image = self.dataset["images"][idx]
        label = self.dataset["labels"][idx]
        
        image = Image.fromarray(image)
        if self.transform_fn is not None:
            image = self.transform_fn(image)
            
        return idx, image, label

    def __len__(self):
        return len(self.dataset["images"])
    
    
