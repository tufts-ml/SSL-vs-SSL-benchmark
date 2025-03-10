from ssl_bench.methods.MethodWrapper import MethodWrapper
from torch.nn import functional as func
import torch
import numpy as np


# TODO replace following two functions with PyTorch implementation
# https://github.com/hysts/pytorch_mixup/blob/master/utils.py
# also similar style as google mixmatch repo mixup baseline
def onehot(label, n_classes):
    return torch.zeros(label.size(0), n_classes).scatter_(1, label.view(-1, 1).long(), 1)


def mixup(data, data2, targets, alpha, n_classes):
    indices = torch.randperm(data2.size(0))

    data2 = data2[indices]
    targets2 = targets[indices]

    targets = onehot(targets, n_classes)
    targets2 = onehot(targets2, n_classes)

    lam = np.random.beta(alpha, alpha)
    created_data = data * lam + data2 * (1 - lam)
    created_targets = targets * lam + targets2 * (1 - lam)

    return created_data, created_targets


# TODO should this inherit from LabelOnlyBaseline?
class MixUp(MethodWrapper):
    # TODO do we want u_data=None in the base class?
    def forward(self, l_data, l_labels, u_data=None):
        self.backbone.train()

        l_input, l_input2 = l_data

        created_l_input, created_l_labels = mixup(
            l_input, l_input2, l_labels, self.args.alpha, self.args.num_classes)

        created_l_input, created_l_labels = created_l_input.to(
            self.args.device).float(), created_l_labels.to(self.args.device).float()

        logits = self.backbone(created_l_input)
        output = func.softmax(logits, dim=1)

        s_loss = func.cross_entropy(logits, created_l_labels,
                                    weight=self.args.weights, reduction='mean')

        return output, s_loss, s_loss, 0  # No unsupervised loss

    def eval_forward(self, l_data, l_labels):
        self.backbone.eval()

        logits = self.backbone(l_data)
        output = func.softmax(logits, dim=1)
        
        s_loss = func.cross_entropy(logits, l_labels, weight=self.args.weights, reduction='mean')

        return output, s_loss
