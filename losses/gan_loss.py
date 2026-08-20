import torch
import torch.nn as nn


class HingeGANLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def d_loss(self, real_pred, fake_pred):
        loss_real = torch.mean(torch.relu(1.0 - real_pred))
        loss_fake = torch.mean(torch.relu(1.0 + fake_pred))
        return loss_real + loss_fake

    def g_loss(self, fake_pred):
        return -torch.mean(fake_pred)