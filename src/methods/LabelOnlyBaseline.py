from src.methods.methods import MethodWrapper
from torch.nn import functional as func


class LabelOnlyBaseline(MethodWrapper):
    def forward(self, l_data, l_labels, u_data=None):
        l_data, l_labels = l_data.to(self.args.device), l_labels.to(self.args.device)

        logits = self.backbone(l_data)

        s_loss = func.cross_entropy(logits, l_labels, weight=self.args.weights, reduction='mean')

        return logits, s_loss, s_loss, 0  # No unsupervised loss

    def eval(self, l_data, l_labels):
        return self.forward(l_data, l_labels)
