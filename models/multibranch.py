import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import (
    EmbedBranch,
    LinearFusion,
    GatedFusion,
)

class MultiBranchFOP(nn.Module):
    """
    Multi-branch multimodal model with:
    - Face-only head
    - Voice-only head
    - Fusion head

    Fusion options:
    - linear
    - gated
    - concat
    """

    def __init__(self, config, face_dim, voice_dim):
        super().__init__()

        self.config = config
        emb = config.embedding_dim
        num_classes = config.resolved_num_classes

        # -------------------------
        # Embedding branches
        # -------------------------
        self.face_branch = EmbedBranch(face_dim, emb)
        self.voice_branch = EmbedBranch(voice_dim, emb)

        # -------------------------
        # Unimodal classifiers
        # -------------------------
        self.face_classifier = nn.Linear(emb, num_classes)
        self.voice_classifier = nn.Linear(emb, num_classes)

        # -------------------------
        # Fusion
        # -------------------------
        if config.fusion == "linear":
            self.fusion = LinearFusion()
            fusion_dim = emb

        elif config.fusion == "gated":
            self.fusion = GatedFusion(emb)
            fusion_dim = emb

        elif config.fusion == "concat":
            self.fusion = None
            fusion_dim = emb * 2

        else:
            raise ValueError(f"Unknown fusion type: {config.fusion}")

        self.fusion_classifier = nn.Linear(fusion_dim, num_classes)

    def forward(self, face, voice):
        # -------------------------
        # Embeddings
        # -------------------------
        face_e = self.face_branch(face)
        voice_e = self.voice_branch(voice)

        # -------------------------
        # Unimodal logits
        # -------------------------
        face_logits = self.face_classifier(face_e)
        voice_logits = self.voice_classifier(voice_e)

        # -------------------------
        # Fusion
        # -------------------------
        if self.fusion is None:
            fused = torch.cat([face_e, voice_e], dim=1)
        else:
            fused, _, _ = self.fusion(face_e, voice_e)

        fusion_logits = self.fusion_classifier(fused)

        return {
            "face_logits": face_logits,
            "voice_logits": voice_logits,
            "fusion_logits": fusion_logits,
            "face_embed": face_e,
            "voice_embed": voice_e,
            "fusion_embed": fused,
        }
