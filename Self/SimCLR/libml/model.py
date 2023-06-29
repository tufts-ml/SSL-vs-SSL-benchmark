import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet


class Model(nn.Module):
    def __init__(self, backbone):
        super(Model, self).__init__()
        self.backbone = backbone
        self.projection_head = nn.Sequential(nn.Linear(512, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, 128, bias=True))
                               
    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        out = self.projection_head(x)
        #print(x.shape)
        return F.normalize(x, dim=-1), F.normalize(out, dim=-1)