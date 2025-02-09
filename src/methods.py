import torch.nn as nn


class MethodWrapper(nn.Module):
    def __init__(self, backbone, args) -> None:
        """Wrapper for SSL method code

        Args:
            backbone (torch.nn.Module): Backbone architecture
            args (Namespace): parsed arguments
        """
        super().__init__()
        self.backbone = backbone
        self.args = args
        self.backbone.fc = nn.Linear(512, args.num_classes)

    def forward(self, l_data, l_labels, u_data):
        """Forward pass for a learning method

        Args:
            l_data (torch.Tensor): Labeled data batch
            l_labels (torch.tensor): Labeled batch labels
            u_data (Any): Unlabeled data batch, may be Tensor or tuple of two batches of Tensors

        Returns:
            tuple:
                loss: loss value to get backpropagated
                s_loss: loss value from supervised loss (0 if no supervised loss)
                u_loss: loss value from unsupervised loss (0 if no unsupervised loss)
        """
        raise NotImplementedError()
