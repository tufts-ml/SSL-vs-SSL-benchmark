from ssl_bench.methods.MethodWrapper import MethodWrapper
import torch.nn as nn
import torch.nn.functional as f
import torch


class SimCLR(MethodWrapper):
    def __init__(self, backbone, args):
        super(SimCLR, self).__init__(backbone, args)
        self.backbone = backbone
        self.backbone.fc = nn.Identity()
        self.projection_head = nn.Sequential(nn.Linear(512, 512, bias=False), nn.BatchNorm1d(512),
                                             nn.ReLU(inplace=True), nn.Linear(512, 128, bias=True))
        self.temperature = args.temperature
        self.device = args.device

    def forward(self, l_data, l_labels, u_data):
        self.backbone.train()

        u1, u2 = u_data
        device = next(self.backbone.parameters()).device

        u1, u2 = u1.to(device), u2.to(device)

        rep1 = self.backbone(u1).flatten(start_dim=1)
        rep2 = self.backbone(u2).flatten(start_dim=1)
        out1 = self.projection_head(rep1)
        out2 = self.projection_head(rep2)

        out1 = f.normalize(out1, dim=1)
        out2 = f.normalize(out2, dim=1)
        rep1 = f.normalize(rep1, dim=1)
        rep2 = f.normalize(rep2, dim=1)

        loss = simclr_loss(self.args, out1, out2, u1.size(0))

        return None, loss, 0, loss

    def eval_forward(self, l_data, l_labels):
        self.backbone.eval()

        device = next(self.backbone.parameters()).device
        l_data = l_data.to(device)

        z1 = self.backbone(l_data).flatten(start_dim=1)

        return z1, 0


def get_negative_mask(batch_size):
    negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0

    negative_mask = torch.cat((negative_mask, negative_mask), 0)
    return negative_mask


def simclr_loss(args, out_1, out_2, batch_size):
    # neg score
    out = torch.cat([out_1, out_2], dim=0)
    neg = torch.exp(torch.mm(out, out.t().contiguous()) / args.temperature)
    mask = get_negative_mask(batch_size).to(args.device)
    neg = neg.masked_select(mask).view(2 * batch_size, -1)

    # pos score
    pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / args.temperature)
    pos = torch.cat([pos, pos], dim=0)

    neg = neg.sum(dim=-1)

    # contrastive loss
    loss = (- torch.log(pos / (pos + neg))).mean()

    return loss
