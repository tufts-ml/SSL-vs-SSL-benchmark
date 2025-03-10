import torch
import torch.nn as nn
import torch.nn.functional as F

class PL(nn.Module):
    def __init__(self, threshold, num_classes):
        super().__init__()
        self.th = threshold
        self.num_classes = num_classes

    def forward(self, x, y, model, mask):
        y_probs = y.softmax(1)

        
        onehot_label = self.__make_one_hot(y_probs.max(1)[1]).float()
        
        gt_mask = (y_probs > self.th).float()
        
        gt_mask = gt_mask.max(1)[0] # reduce_any
        
        lt_mask = 1 - gt_mask # logical not
        
        p_target = gt_mask[:,None] * self.num_classes * onehot_label + lt_mask[:,None] * y_probs

        p_target = p_target.detach()
        

        
#         loss = (-(p_target.detach() * F.log_softmax(output, 1)).sum(1)*mask).mean()
        loss = (-(p_target * F.log_softmax(y, 1)).sum(1)*mask).mean()


        return loss, gt_mask

    def __make_one_hot(self, y):
        return torch.eye(self.num_classes)[y].to(y.device)
