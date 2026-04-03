import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------
# Utility blocks
# --------------------------------------------------

def fc_block(in_dim, out_dim, p=0.5):
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.BatchNorm1d(out_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(p),
    )


class EmbedBranch(nn.Module):
    def __init__(self, feat_dim, emb_dim):
        super().__init__()
        self.fc = fc_block(feat_dim, emb_dim)

    def forward(self, x):
        x = self.fc(x)
        return F.normalize(x, dim=1)


# --------------------------------------------------
# Linear fusion
# --------------------------------------------------

class LinearFusion(nn.Module):
    """
    Learnable weighted sum fusion
    """
    def __init__(self):
        super().__init__()
        self.w_face = nn.Parameter(torch.rand(1))
        self.w_audio = nn.Parameter(torch.rand(1))

    def forward(self, face, audio):
        fused = self.w_face * face + self.w_audio * audio
        return fused, face, audio


# --------------------------------------------------
# Gated fusion
# --------------------------------------------------

class ForwardBlock(nn.Module):
    def __init__(self, in_dim, out_dim, p=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
        )

    def forward(self, x):
        return self.block(x)


class GatedFusion(nn.Module):
    """
    Gated multimodal fusion
    """
    def __init__(self, emb_dim, mid_dim=128):
        super().__init__()

        self.attention = nn.Sequential(
            ForwardBlock(emb_dim * 2, mid_dim),
            nn.Linear(mid_dim, emb_dim),
        )

        self.face_proj = nn.Linear(emb_dim, emb_dim)
        self.audio_proj = nn.Linear(emb_dim, emb_dim)

    def forward(self, face, audio):
        concat = torch.cat([face, audio], dim=1)
        gate = torch.sigmoid(self.attention(concat))

        face_t = torch.tanh(self.face_proj(face))
        audio_t = torch.tanh(self.audio_proj(audio))

        fused = gate * face_t + (1.0 - gate) * audio_t
        return fused, face_t, audio_t
