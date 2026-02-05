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

            fused, logits, _, _ = self.model(face, audio)

            loss = (
                self.ce(logits, labels)
                + alpha * self.opl(fused, labels)
            )

            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        return avg_loss
