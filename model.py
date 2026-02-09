import torch
import torch.nn as nn
import torch.nn.functional as F

def fc_block(in_dim, out_dim):
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.BatchNorm1d(out_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5)
    )

class EmbedBranch(nn.Module):
    def __init__(self, feat_dim, emb_dim):
        super().__init__()
        self.fc = fc_block(feat_dim, emb_dim)

    def forward(self, x):
        x = self.fc(x)
        return F.normalize(x, dim=1)

class LinearFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Parameter(torch.rand(1))
        self.w2 = nn.Parameter(torch.rand(1))

    def forward(self, face, voice):
        return self.w1 * face + self.w2 * voice

class FOP(nn.Module):
    def __init__(self, config, face_dim, voice_dim):
        super().__init__()
        self.face_branch = EmbedBranch(face_dim, config.embedding_dim)
        self.voice_branch = EmbedBranch(voice_dim, config.embedding_dim)

        self.fusion = LinearFusion()
        self.classifier = nn.Linear(config.embedding_dim, config.resolved_num_classes)

    def forward(self, face, voice):
        face_e = self.face_branch(face)
        voice_e = self.voice_branch(voice)
        fused = self.fusion(face_e, voice_e)
        logits = self.classifier(fused)
        return fused, logits, face_e, voice_e
