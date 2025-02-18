from src.methods.methods import MethodWrapper
from torch.nn import functional as func


class LabelOnlyBaseline(MethodWrapper):
    def forward(self, l_data, l_labels, u_data=None):
        # Forward pass
        logits = self.backbone(l_data)

        if self.args.dataset_name == "CheXpert":
            s_loss = func.binary_cross_entropy_with_logits(
                logits, l_labels.float(), weight=self.args.weights, reduction='mean')
        else:
            s_loss = func.cross_entropy(
                logits, l_labels, weight=self.args.weights, reduction='mean')

        # s_loss = func.cross_entropy(logits, l_labels, weight=self.args.weights, reduction='mean')

        return logits, s_loss, s_loss, 0  # No unsupervised loss
