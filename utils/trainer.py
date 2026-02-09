import torch
from tqdm import tqdm
from losses import OrthogonalProjectionLoss

class Trainer:
    def __init__(self, model, config):
        self.model = model.to(config.device)
        self.config = config

        self.ce = torch.nn.CrossEntropyLoss()
        self.opl = OrthogonalProjectionLoss()
        self.opt = torch.optim.Adam(model.parameters(), lr=config.lr)

    def train_epoch(self, loader, alpha, logger=None, epoch=None):
        self.model.train()
        total_loss = 0.0

        # tqdm only if debug or explicitly enabled
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

            fused, logits, _, _ = self.model(face, audio)

            loss = (
                self.ce(logits, labels)
                + alpha * self.opl(fused, labels)
            )

            # loss = self.ce(logits, labels)

            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        return avg_loss
