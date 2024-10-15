import torch
import os
from torch.utils.data import Dataset
from torchvision.io import read_image


class ImageDataset(Dataset):
    def __init__(self,path_dir,transform=None):
        self.path_dir = path_dir
        self.image_files = os.listdir(path=path_dir)
        #self.image_files.remove('desktop.ini')

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        filename = self.image_files[index]
        filepath = os.path.join(self.path_dir,filename)
        img = read_image(filepath)
        img = img.float()
        img = (img / 127.5) - 1 # Normalize image from -1 to 1
        return img