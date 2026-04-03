import torch
import torch.nn as nn

from .model import (
    EmbedBranch,
    LinearFusion,
    GatedFusion,
)

class MultiBranchFOP(nn.Module):
    """
    Multi-branch multimodal model with:
    - Face-only head
    - Audio-only head
    - Fusion head

    Fusion options:
    - linear
    - gated
    - concat
    """

    def __init__(self, config, face_dim, audio_dim):
        super().__init__()

        self.config = config
        emb = config.embedding_dim
        num_classes = config.resolved_num_classes

        # -------------------------
        # Embedding branches
        # -------------------------
        self.face_branch = EmbedBranch(face_dim, emb)
        self.audio_branch = EmbedBranch(audio_dim, emb)

        # -------------------------
        # Unimodal classifiers
        # -------------------------
        self.face_classifier = nn.Linear(emb, num_classes)
        self.audio_classifier = nn.Linear(emb, num_classes)

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

    def forward(self, face, audio):
        # -------------------------
        # Embeddings
        # -------------------------
        face_e = self.face_branch(face)
        audio_e = self.audio_branch(audio)

        # -------------------------
        # Unimodal logits
        # -------------------------
        face_logits = self.face_classifier(face_e)
        audio_logits = self.audio_classifier(audio_e)

        # -------------------------
        # Fusion
        # -------------------------
        if self.fusion is None:
            fused = torch.cat([face_e, audio_e], dim=1)
        else:
            fused, _, _ = self.fusion(face_e, audio_e)

        fusion_logits = self.fusion_classifier(fused)

        return {
            "face_logits": face_logits,
            "audio_logits": audio_logits,
            "fusion_logits": fusion_logits,
            "face_embed": face_e,
            "audio_embed": audio_e,
            "fusion_embed": fused,
        }
