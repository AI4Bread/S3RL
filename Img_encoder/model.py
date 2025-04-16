import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50


class Model(nn.Module):
    def __init__(self, dim_out=128, dim_mid=256):
        super(Model, self).__init__()

        self.f = []
        for name, module in resnet50().named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        # reduce dimension
        self.e = nn.Sequential(nn.Linear(2048, dim_mid, bias=True), nn.BatchNorm1d(dim_mid), nn.ReLU(inplace=True))
        # projection head
        self.g = nn.Sequential(nn.Linear(256, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, dim_out, bias=True))

    def forward(self, x):
        x = self.f(x)
        feature = self.e(torch.flatten(x, start_dim=1))
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)
