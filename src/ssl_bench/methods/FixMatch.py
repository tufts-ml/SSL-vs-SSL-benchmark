from ssl_bench.methods.MethodWrapper import MethodWrapper
from torch.nn import functional as func
import torchvision.transforms.v2 as transforms
import torch


class FixMatch(MethodWrapper):
    def forward(self, l_data, l_labels, u_data=None):
        self.backbone.train()

        if u_data is None:
            print("Warning: u_data is None")
            u_weak = u_strong = torch.zeros_like(l_data)  
        else:
            u_weak, u_strong = u_data
            # print(f"u_data (before forward): {u_data}") 

        l_data = l_data.to(self.args.device).float()
        l_labels = l_labels.to(self.args.device).long()
        l_logits = self.backbone(l_data)
        sup_loss = func.cross_entropy(l_logits, l_labels,
                                      weight=self.args.weights, reduction='mean')
        l_probs = func.softmax(l_logits, dim=1)

        u_weak = u_weak.to(self.args.device).float()
        u_strong = u_strong.to(self.args.device).float()

        with torch.no_grad():
            u_logits_weak = self.backbone(u_weak)
            u_probs_weak = func.softmax(u_logits_weak, dim=1)
            max_probs, pseudo_labels = torch.max(u_probs_weak, dim=1)
            mask = max_probs.ge(self.args.conf_threshold).float()

        u_logits_strong = self.backbone(u_strong)
        unsup_loss = func.cross_entropy(u_logits_strong, pseudo_labels,
                                        reduction='none')
        unsup_loss = (unsup_loss * mask).mean()

        total_loss = sup_loss + self.args.unsup_weight * unsup_loss
        return l_probs, sup_loss, total_loss, unsup_loss

    def eval_forward(self, l_data, l_labels):
        self.backbone.eval()

        l_data = l_data.to(self.args.device).float()
        l_labels = l_labels.to(self.args.device).long()

        logits = self.backbone(l_data)
        class_probs = func.softmax(logits, dim=1)

        loss = func.cross_entropy(logits, l_labels,
                                  weight=self.args.weights, reduction='mean')

        return class_probs, loss
