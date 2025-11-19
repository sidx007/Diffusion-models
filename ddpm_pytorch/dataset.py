import pandas as pd
import numpy as np
import torchvision
from torch.utils.data.dataset import Dataset

class CustomMnistDataset(Dataset):
    def __init__(self, csv_path, num_datapoints = None):
        super(CustomMnistDataset, self).__init__()
        
        self.df = pd.read_csv(csv_path)
        
        if num_datapoints is not None:
            self.df = self.df.iloc[0:num_datapoints]
      
    def __len__(self):
        return len(self.df)
    
    def  __getitem__(self, index):
        img = self.df.iloc[index].filter(regex='pixel').values
        img =  np.reshape(img, (28, 28)).astype(np.uint8)
        
        img_tensor = torchvision.transforms.ToTensor()(img)
        img_tensor = 2*img_tensor - 1
        
        return img_tensor
