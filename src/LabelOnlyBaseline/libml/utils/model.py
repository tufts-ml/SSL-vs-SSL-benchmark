from src.methods import MethodWrapper
from torch.nn import functional as func


class SupervisedMethod(MethodWrapper):
    def forward(self, l_data, l_labels, u_data=None):
        # Forward pass
        logits = self.backbone(l_data)
        s_loss = func.cross_entropy(logits, l_labels, weight=self.args.weights, reduction='mean')
        return s_loss, s_loss, 0  # No unsupervised loss