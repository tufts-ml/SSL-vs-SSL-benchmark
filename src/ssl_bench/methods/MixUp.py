from ssl_bench.methods.MethodWrapper import MethodWrapper
from torch.nn import functional as func
import torchvision.transforms.v2 as transforms


class MixUp(MethodWrapper):
    def forward(self, l_data, l_labels, u_data=None):
        self.backbone.train()

        mixup_transform = transforms.MixUp(alpha=self.args.alpha, num_classes=self.args.num_classes)

        created_l_input, created_l_labels = mixup_transform((l_data, l_labels))
        created_l_input, created_l_labels = created_l_input.to(
            self.args.device).float(), created_l_labels.to(self.args.device).float()

        logits = self.backbone(created_l_input)
        class_probs = func.softmax(logits, dim=1)

        s_loss = func.cross_entropy(logits, created_l_labels,
                                    weight=self.args.weights, reduction='mean')

        return class_probs, s_loss, s_loss, 0  # No unsupervised loss

    def eval_forward(self, l_data, l_labels):
        self.backbone.eval()

        logits = self.backbone(l_data)
        class_probs = func.softmax(logits, dim=1)

        loss = func.cross_entropy(logits, l_labels,
                                  weight=self.args.weights, reduction='mean')

        return class_probs, loss
