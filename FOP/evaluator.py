import torch
import logging


class Evaluator:
    def __init__(self, model, config):
        self.model = model
        self.config = config

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(config.log_level)

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "[%(levelname)s][%(name)s] %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def accuracy(self, loader):
        """
        Compute accuracy for the given loader.

        Missing-modality behavior is defined entirely by the dataset
        via `config.missing_modality` and `config.missing_ratio`.
        """
        self.model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for audio, face, labels in loader:
                audio = audio.to(self.config.device, non_blocking=True)
                face = face.to(self.config.device, non_blocking=True)
                labels = labels.to(self.config.device, non_blocking=True)

                _, logits, _, _ = self.model(face, audio)
                preds = logits.argmax(dim=1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

        acc = 100.0 * correct / total

        return acc
