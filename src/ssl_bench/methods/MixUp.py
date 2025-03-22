from ssl_bench.methods.MethodWrapper import MethodWrapper
from torch.nn import functional as func
from torchvision.transforms.v2 import MixUp as mixup


# TODO should this inherit from LabelOnlyBaseline?
class MixUp(MethodWrapper):
    def forward(self, l_data, l_labels, u_data=None):
        self.backbone.train()

        mixup_transform = mixup(alpha=self.args.alpha, num_classes=self.args.num_classes)

        print("l_data shape: ", l_data.shape)
        print("l_labels shape: ", l_labels.shape)

        created_l_input, created_l_labels = mixup_transform((l_data, l_labels))

        print("created_l_input shape: ", created_l_input.shape)
        print("created_l_labels shape: ", created_l_labels
              .shape)

        created_l_input, created_l_labels = created_l_input.to(
            self.args.device).float(), created_l_labels.to(self.args.device).float()

        logits = self.backbone(created_l_input)
        output = func.softmax(logits, dim=1)

        print("output shape: ", output.shape)

        s_loss = func.cross_entropy(logits, created_l_labels,
                                    weight=self.args.weights, reduction='mean')

        return output, s_loss, s_loss, 0  # No unsupervised loss

    def eval_forward(self, l_data, l_labels):
        self.backbone.eval()

        logits = self.backbone(l_data)
        output = func.softmax(logits, dim=1)

        loss = func.cross_entropy(logits, l_labels,
                                    weight=self.args.weights, reduction='mean')

        return output, loss
