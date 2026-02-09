import torch
from tqdm import tqdm
from .losses import OrthogonalProjectionLoss


class Trainer:
    def __init__(self, model, config):
        self.model = model.to(config.device)
        self.config = config

        self.ce = torch.nn.CrossEntropyLoss()
        self.opl = OrthogonalProjectionLoss()
        self.opt = torch.optim.Adam(self.model.parameters(), lr=config.lr)

    def train_epoch(self, loader, alpha, logger=None, epoch=None):
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(
            loader,
            desc=f"Epoch {epoch}",
            disable=not self.config.debug,
            leave=False,
        )

        for audio, face, labels in pbar:
            audio = audio.to(self.config.device, non_blocking=True)
            face = face.to(self.config.device, non_blocking=True)
            labels = labels.to(self.config.device, non_blocking=True)

            # --------------------------------------------------
            # TRAIN-TIME missing modality (batch-level)
            # --------------------------------------------------
            if (
                self.config.train_missing_modality is not None
                and self.config.missing_ratio > 0
            ):
                B = labels.size(0)
                k = int(self.config.missing_ratio * B)

                if k > 0:
                    idx = torch.randperm(B, device=labels.device)[:k]

                    if self.config.train_missing_modality == "audio":
                        audio[idx] = 0
                    elif self.config.train_missing_modality == "face":
                        face[idx] = 0

            # --------------------------------------------------
            # Forward
            # --------------------------------------------------
            out = self.model(face, audio)

            # --------------------------------------------------
            # Losses
            # --------------------------------------------------
            if isinstance(out, dict):
                # -------- MultiBranchFOP --------
                loss_face = self.ce(out["face_logits"], labels)
                loss_voice = self.ce(out["voice_logits"], labels)
                loss_fusion = self.ce(out["fusion_logits"], labels)

                loss = (
                    self.config.loss_face * loss_face
                    + self.config.loss_voice * loss_voice
                    + self.config.loss_fusion * loss_fusion
                )

                # Optional OPL on fusion embedding
                if alpha > 0:
                    loss = loss + alpha * self.opl(
                        out["fusion_embed"], labels
                    )

            else:
                # -------- FOP (baseline) --------
                fused, logits, _, _ = out
                loss = self.ce(logits, labels)

                if alpha > 0:
                    loss = loss + alpha * self.opl(fused, labels)

            # --------------------------------------------------
            # Backprop
            # --------------------------------------------------
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step()

            total_loss += loss.item()

        return total_loss / len(loader)
