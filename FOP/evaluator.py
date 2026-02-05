import torch

class Evaluator:
    def __init__(self, model, config):
        self.model = model
        self.config = config

        # Cache tensors (lazy init)
        self._cached = {}

    def _get_tensors(self, dataset):
        """
        Cache tensors to avoid repeated torch.from_numpy calls.
        """
        key = id(dataset)
        if key not in self._cached:
            self._cached[key] = (
                torch.from_numpy(dataset.face_feats).float(),
                torch.from_numpy(dataset.audio_feats).float(),
                torch.from_numpy(dataset.labels).long(),
            )
        return self._cached[key]

    def accuracy_from_tensors(self, face, audio, labels):
        self.model.eval()
        with torch.no_grad():
            _, logits, _, _ = self.model(face, audio)
            preds = logits.argmax(dim=1)
            correct = (preds == labels).sum().item()
        return 100.0 * correct / labels.size(0)


    def accuracy(self, dataset):
        """
        Vectorized accuracy computation (FAST).

        Missing-modality behavior is defined by the dataset
        or can be applied externally via tensor ops.
        """
        self.model.eval()

        face, audio, labels = self._get_tensors(dataset)

        with torch.no_grad():
            face = face.to(self.config.device, non_blocking=True)
            audio = audio.to(self.config.device, non_blocking=True)
            labels = labels.to(self.config.device, non_blocking=True)

            _, logits, _, _ = self.model(face, audio)
            preds = logits.argmax(dim=1)

            correct = (preds == labels).sum().item()

        return 100.0 * correct / labels.size(0)
