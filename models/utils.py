import torch
import torch.nn as nn
from torchsummary import summary


def downsample(in_channel, out_channel, kernel_size=4, stride=2, padding=1):
    layer = nn.Sequential()
    layer.add_module('Conv',nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride, padding=padding))
    layer.add_module('Batch_Norm', nn.BatchNorm2d(num_features=out_channel))
    layer.add_module('LeakyReLU', nn.LeakyReLU())
    return layer


def upsample(in_channel, out_channel, kernel_size=4, stride=2, padding=1,apply_dropout=False):
    layer = nn.Sequential()
    layer.add_module('Conv_Transpose',nn.ConvTranspose2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride, padding=padding))
    layer.add_module('Batch_Norm', nn.BatchNorm2d(num_features=out_channel))
    if apply_dropout:
        layer.add_module('Dropout', nn.Dropout())
    layer.add_module('LeakyReLU', nn.LeakyReLU())
    return layer

def conv_stack(in_channel, out_channel, kernel_size=4, stride=2, padding=1, apply_pooling=True):
    layer = nn.Sequential()
    layer.add_module('Conv',nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride, padding=padding))
    if apply_pooling:
        layer.add_module('MaxPooling',nn.MaxPool2d(kernel_size=3, stride=1))
    layer.add_module('Batch_Norm', nn.BatchNorm2d(num_features=out_channel))
    layer.add_module('LeakyReLU', nn.LeakyReLU())
    return layer


def display(mdl):
    summary(mdl, torch.zeros(32, 3, 256, 256))
