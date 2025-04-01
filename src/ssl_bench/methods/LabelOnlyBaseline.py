from ssl_bench.methods.MethodWrapper import MethodWrapper 
from torch.nn import functional as func


class LabelOnlyBaseline(MethodWrapper):
    def forward(self, l_data, l_labels):
        l_data, l_labels = l_data.to(self.args.device), l_labels.to(self.args.device)

        logits = self.backbone(l_data)
        class_probs = func.softmax(logits, dim=1)

        s_loss = func.cross_entropy(logits, l_labels, weight=self.args.weights, reduction='mean')

        return class_probs, s_loss, s_loss, 0  # No unsupervised loss

    def eval_forward(self, l_data, l_labels):
        class_probs, s_loss, _, _ = self.forward(l_data, l_labels)
        return class_probs, s_loss
