import torch
import torch.nn as nn
from .utils import downsample, upsample, conv_stack


class Generator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        num_layers = 8
        in_channels = [3, 64, 128, 256, 512, 512, 512, 512]
        out_channels = [64, 128, 256, 512, 512, 512, 512, 512]
        self.downsample_stack = nn.ModuleList()
        for i in range(num_layers):
            self.downsample_stack.append(downsample(in_channels[i], out_channels[i]))

        in_channels = [512, 1024, 1024, 1024, 1024, 512, 256, 128]
        out_channels = [512, 512, 512, 512, 256, 128, 64, 3]
        self.upsample_stack = nn.ModuleList()
        for i in range(num_layers):
            if i < 3:
                apply_dropout = True
            else:
                apply_dropout = False
            self.upsample_stack.append(upsample(in_channels[i], out_channels[i], apply_dropout = apply_dropout))

    def forward(self, x):
        layer_outs = []
        for layer in self.downsample_stack:
            x = layer(x)
            layer_outs.append(x)

        layer_outs = layer_outs[::-1]
        n = len(layer_outs)

        for i, layer in enumerate(self.upsample_stack):
            x = layer(x)
            if i < n-1:
                x = torch.cat((x, layer_outs[i+1]), dim=1)

        # Apply tanh to bring the output to [-1, 1], then normalize to [0, 1]
        x = torch.tanh(x)  # Output is now in range [-1, 1]
        x = (x + 1) / 2  # Normalize to [0, 1]

        return x


class Discriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        num_conv_layers = 7
        in_channels = [3, 32, 64, 128, 256, 512, 512]
        out_channels = [32, 64, 128, 256, 512, 512, 512]
        self.layers = nn.ModuleList()
        for i in range(num_conv_layers):
            if i < 5:
                apply_pooling = True
            else:
                apply_pooling = False
            self.layers.append(conv_stack(in_channels[i], out_channels[i], apply_pooling = apply_pooling))
        self.layers.append(nn.Flatten(start_dim=1,end_dim=3))
        self.layers.append(nn.Linear(512, 128))
        self.layers.append(nn.Linear(128, 1))
        self.layers.append(nn.Sigmoid())

    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
        return x
