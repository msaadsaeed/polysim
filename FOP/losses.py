import torch
import torch.nn as nn
import torch.nn.functional as F

class OrthogonalProjectionLoss(nn.Module):
    def forward(self, feats, labels):
        feats = F.normalize(feats, dim=1)
        labels = labels.unsqueeze(1)

        mask = labels.eq(labels.T)
        eye = torch.eye(len(labels), device=labels.device).bool()

        pos = (mask & ~eye).float()
        neg = (~mask).float()

        dot = feats @ feats.T

        pos_mean = (pos * dot).sum() / (pos.sum() + 1e-6)
        neg_mean = (neg * dot).abs().sum() / (neg.sum() + 1e-6)

        loss = (1 - pos_mean) + 0.7 * neg_mean
        return loss
