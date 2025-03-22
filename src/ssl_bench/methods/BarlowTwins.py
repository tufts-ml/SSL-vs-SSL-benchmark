import torch.nn as nn
from lightly.models.modules import BarlowTwinsProjectionHead
from lightly.loss import BarlowTwinsLoss


class BarlowTwins(nn.Module):
    def __init__(self, backbone, projection_dim=2048, lambd=5e-3):
        super(BarlowTwins, self).__init__()
        self.backbone = backbone
        self.projection_head = BarlowTwinsProjectionHead(512, 2048, 2048)
        self.barlow_twins_loss = BarlowTwinsLoss()

    def forward(self, l_data, l_labels, u_data):
        self.backbone.train()

        u1, u2 = u_data
        z1 = self.backbone(u1)
        z2 = self.backbone(u2)
        z1 = self.projection_head(z1)
        z2 = self.projection_head(z2)
        loss = self.barlow_twins_loss(z1, z2)

        return None, loss, 0, loss

    def eval_forward(self, l_data, l_labels):
        self.backbone.eval()

        z1 = self.backbone(l_data)
        z1 = self.projection_head(z1)

        return None, None, z1
