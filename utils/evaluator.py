import torch


class Evaluator:
    def __init__(self, model, config):
        self.model = model
        self.config = config

        # Cache tensors (lazy init)
        self._cached = {}

    # --------------------------------------------------
    # Dataset → tensor cache
    # --------------------------------------------------
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

    # --------------------------------------------------
    # Core accuracy from tensors
    # --------------------------------------------------
    def accuracy_from_tensors(
        self,
        face,
        audio,
        labels,
        head="fusion",   # "fusion" | "face" | "audio"
    ):
        """
        Vectorized accuracy from tensors.

        head:
            - "fusion" (default)
            - "face"
            - "audio"
        """
        self.model.eval()

        with torch.no_grad():
            out = self.model(face, audio)

            # ---------------------------
            # MultiBranchFOP
            # ---------------------------
            if isinstance(out, dict):
                if head == "fusion":
                    logits = out["fusion_logits"]
                elif head == "face":
                    logits = out["face_logits"]
                elif head == "audio":
                    logits = out["audio_logits"]
                else:
                    raise ValueError(f"Unknown head: {head}")

            # ---------------------------
            # Baseline FOP
            # ---------------------------
            else:
                _, logits, _, _ = out

            preds = logits.argmax(dim=1)
            correct = (preds == labels).sum().item()

        return 100.0 * correct / labels.size(0)

    # --------------------------------------------------
    # Dataset-level accuracy
    # --------------------------------------------------
    def accuracy(self, dataset, head="fusion"):
        """
        Vectorized accuracy computation (FAST).

        head:
            - "fusion" (default)
            - "face"
            - "audio" (only valid for MultiBranchFOP)
        """
        face, audio, labels = self._get_tensors(dataset)

        face = face.to(self.config.device, non_blocking=True)
        audio = audio.to(self.config.device, non_blocking=True)
        labels = labels.to(self.config.device, non_blocking=True)

        return self.accuracy_from_tensors(
            face,
            audio,
            labels,
            head=head,
        )
