import torch
import torch.nn as nn
import torch.nn.functional as F


class DemoNet(nn.Module):
    def __init__(self, in_channels, out_channels, backbone=None):
        super(DemoNet, self).__init__()
        self.backbone = backbone
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.backbone_network = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.ReLU(),
            nn.Linear(64, out_channels)
        )

    def forward(self, x):
        output = self.backbone_network(x)

        return output
