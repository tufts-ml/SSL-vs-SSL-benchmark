from ssl_bench.methods.MethodWrapper import MethodWrapper
import torch
from torch.nn import functional as func
import torchvision.transforms.v2 as transforms


class MixMatch(MethodWrapper):
    def forward(self, l_data, l_labels, u_data=None):
        self.backbone.train()

        if u_data is None:
            print("Warning: u_data is None")
            u_weak = u_strong = torch.zeros_like(l_data)
        else:
            u_weak, u_strong = u_data

        l_data = l_data.to(self.args.device).float()
        l_labels = l_labels.to(self.args.device).long()
        u_weak = u_weak.to(self.args.device).float()
        u_strong = u_strong.to(self.args.device).float()

        num_classes = self.args.num_classes
        l_labels_onehot = torch.zeros(l_labels.size(0), num_classes, device=l_labels.device)
        l_labels_onehot.scatter_(1, l_labels.view(-1, 1), 1)

        with torch.no_grad():
            u_logits = self.backbone(u_weak)
            u_probs = func.softmax(u_logits, dim=1)
            u_probs = u_probs ** (1.0 / self.args.temperature)
            u_probs = u_probs / u_probs.sum(dim=1, keepdim=True)

        all_inputs = torch.cat([l_data, u_weak, u_strong], dim=0)
        all_targets = torch.cat([l_labels_onehot, u_probs, u_probs], dim=0)

        idx = torch.randperm(all_inputs.size(0))
        mixed_input = self.mixup(all_inputs, all_inputs[idx], self.args.alpha)
        mixed_target = self.mixup(all_targets, all_targets[idx], self.args.alpha)

        logits = self.backbone(mixed_input)
        loss = func.cross_entropy(logits, mixed_target.argmax(dim=1), reduction='mean')

        l_logits = logits[:l_data.size(0)]
        l_probs = func.softmax(l_logits, dim=1)
        sup_loss = func.cross_entropy(l_logits, l_labels, weight=self.args.weights, reduction='mean')

        return l_probs, sup_loss, loss, torch.tensor(0.0, device=loss.device)

    def eval_forward(self, l_data, l_labels):
        self.backbone.eval()

        l_data = l_data.to(self.args.device).float()
        l_labels = l_labels.to(self.args.device).long()

        logits = self.backbone(l_data)
        class_probs = func.softmax(logits, dim=1)

        loss = func.cross_entropy(logits, l_labels, weight=self.args.weights, reduction='mean')

        return class_probs, loss

    def mixup(self, x1, x2, alpha):
        lam = torch.distributions.Beta(alpha, alpha).sample().to(x1.device)
        return lam * x1 + (1 - lam) * x2