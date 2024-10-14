import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader


def get_data_loaders(dataset, batch_size, shuffle):
    data_loader = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=shuffle)
    return data_loader


def imshow(img):
    img = torch.permute(img,(1,2,0))
    plt.imshow(img)
    plt.show()