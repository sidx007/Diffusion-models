import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from datasets import load_dataset

def img_to_tensor(im):
  return torch.tensor(np.array(im.convert('RGB'))/255).permute(2, 0, 1).unsqueeze(0) * 2 - 1

def tensor_to_image(t):
  return Image.fromarray(np.array(((t.squeeze().permute(1, 2, 0)+1)/2).clip(0, 1)*255).astype(np.uint8))

def getTransforms():
    return  transforms.Compose([
        transforms.ToTensor(),
    ])

class CIFAR10Dataset(Dataset):
    def __init__(self, transforms=None):
        super(CIFAR10Dataset, self).__init__()
        self.transforms = transforms
        self.cifar10 = load_dataset('cifar10')

    def __len__(self):
        return len(self.cifar10['train'])


    def __getitem__(self, index):
        image = self.cifar10['train'][index]['img']
        image = self.transforms(image)
        return image
