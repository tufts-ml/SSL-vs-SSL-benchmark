from ssl_bench.methods.MethodWrapper import MethodWrapper
from lightly.models.modules import BarlowTwinsProjectionHead
from lightly.loss import BarlowTwinsLoss


class BarlowTwins(MethodWrapper):
    def __init__(self, backbone, args):
        super(BarlowTwins, self).__init__(backbone, args)  # Pass required arguments
        self.backbone = backbone
        self.projection_head = BarlowTwinsProjectionHead(
            512, 1024, 1024)
        self.barlow_twins_loss = BarlowTwinsLoss(
            lambda_param=args.lambd if hasattr(args, 'lambd') else 5e-3)

    def forward(self, l_data, l_labels, u_data):
        self.backbone.train()

        u1, u2 = u_data
        device = next(self.backbone.parameters()).device  # Get the device of the model

        u1, u2 = u1.to(device), u2.to(device)  # Move inputs to the correct device

        z1_features = self.backbone(u1)
        z2_features = self.backbone(u2)

        z1 = self.projection_head(z1_features)
        z2 = self.projection_head(z2_features)

        loss = self.barlow_twins_loss(z1, z2)

        return None, loss, 0, loss

    def eval_forward(self, l_data, l_labels):
        self.backbone.eval()

        device = next(self.backbone.parameters()).device
        l_data = l_data.to(device)

        z1 = self.projection_head(self.backbone(l_data))
        return z1, 0
