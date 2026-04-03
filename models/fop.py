import torch
import torch.nn as nn

from .model import (
    EmbedBranch,
    LinearFusion,
    GatedFusion,
)


# --------------------------------------------------
# Main model
# --------------------------------------------------

class FOP(nn.Module):
    def __init__(self, config, face_dim, audio_dim):
        super().__init__()

        emb_dim = config.embedding_dim
        num_classes = config.resolved_num_classes

        self.face_branch = EmbedBranch(face_dim, emb_dim)
        self.audio_branch = EmbedBranch(audio_dim, emb_dim)

        # --------------------------------------------------
        # Fusion selection
        # --------------------------------------------------
        if config.fusion == "linear":
            self.fusion = LinearFusion()
            fusion_dim = emb_dim

        elif config.fusion == "gated":
            self.fusion = GatedFusion(emb_dim)
            fusion_dim = emb_dim

        elif config.fusion == "concat":
            self.fusion = None
            fusion_dim = emb_dim * 2

        else:
            raise ValueError(f"Unknown fusion type: {config.fusion}")

        # --------------------------------------------------
        # Classifier
        # --------------------------------------------------
        self.classifier = nn.Linear(fusion_dim, num_classes)

    def forward(self, face, audio):
        face_e = self.face_branch(face)
        audio_e = self.audio_branch(audio)

        # --------------------------------------------------
        # Fusion
        # --------------------------------------------------
        if self.fusion is None:
            fused = torch.cat([face_e, audio_e], dim=1)
        else:
            fused, face_e, audio_e = self.fusion(face_e, audio_e)

        logits = self.classifier(fused)

        return fused, logits, face_e, audio_e
