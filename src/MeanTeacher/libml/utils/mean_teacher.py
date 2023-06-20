import torch
import torch.nn as nn
import torch.nn.functional as F

class MT(nn.Module):
    def __init__(self):
        super().__init__()
       

    def forward(self, x, y, ema_model, mask):
        
#         print('y requires_grad: {}'.format(y.requires_grad))
        y_hat = ema_model(x)
#         print('y_hat requires_grad: {}'.format(y_hat.requires_grad))
        
        masked_y_predictions = y.softmax(1)[mask]
        return F.mse_loss(masked_y_predictions, y_hat.softmax(1).detach(), reduction="none").mean(1).mean()
