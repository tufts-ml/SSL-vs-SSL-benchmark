from ssl_bench.methods.MethodWrapper import MethodWrapper
import torch
from torch.nn import functional as func


class PseudoLabeling(MethodWrapper):
    def forward(self, l_data, l_labels, u_data_weak=None, u_data_strong=None):
        self.backbone.train()

        l_data, l_labels = l_data.to(self.args.device).float(), l_labels.to(self.args.device).long()
        if u_data_weak is not None and u_data_strong is not None:
            u_data_weak = u_data_weak.to(self.args.device).float()
            u_data_strong = u_data_strong.to(self.args.device).float()

        l_logits = self.backbone(l_data)
        sup_loss = func.cross_entropy(l_logits, l_labels, weight=self.args.weights, reduction='mean')

        unsup_loss = torch.tensor(0.0, device=l_data.device)
        if u_data_weak is not None and u_data_strong is not None:
            self.backbone.eval()
            with torch.no_grad():
                u_logits_weak = self.backbone(u_data_weak)
                u_probs = func.softmax(u_logits_weak, dim=1)
                u_confidences, u_pseudo_labels = torch.max(u_probs, dim=1)
                mask = u_confidences.ge(self.args.confidence_threshold).float()
            self.backbone.train()

            u_logits_strong = self.backbone(u_data_strong)
            per_sample_loss = func.cross_entropy(u_logits_strong, u_pseudo_labels, reduction='none')
            unsup_loss = (per_sample_loss * mask).sum() / (mask.sum() + 1e-8)

        total_loss = sup_loss + self.args.lambda_u * unsup_loss

        return func.softmax(l_logits, dim=1), sup_loss, total_loss, unsup_loss

    def eval_forward(self, l_data, l_labels):
        self.backbone.eval()

        l_data = l_data.to(self.args.device).float()
        l_labels = l_labels.to(self.args.device).long()

        logits = self.backbone(l_data)
        class_probs = func.softmax(logits, dim=1)
        loss = func.cross_entropy(logits, l_labels, weight=self.args.weights, reduction='mean')

        return class_probs, loss
