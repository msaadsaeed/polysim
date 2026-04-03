from dataclasses import dataclass
import logging

@dataclass
class ExperimentConfig:
    home_dir = "/feats"
    seed: int = 1
    device: str = "cuda"
    lr: float = 1e-3
    batch_size: int = 32
    max_epochs: int = 300
    num_workers = 0
    alpha: float = 0.0
    embedding_dim: int = 512

    model_type: str = "fop"   # "fop" | "multibranch"
    fusion: str = "linear"   # "linear" | "gated" | "concat"
    
    loss_face: float = 1.0
    loss_audio: float = 1.0
    loss_fusion: float = 1.0

    version: str = "v3"
    seen_lang: str = "English"

    test_missing_modality = "face" # face, audio
    test_alpha=0.0

    debug: bool = False
    log_level = logging.DEBUG if debug else logging.INFO

    early_stop: bool = True
    early_stop_patience: int = 10      # tolerance (epochs)
    early_stop_min_delta: float = 0.3 # minimum improvement
    early_stop_metric: str = "seen"    # "seen" | "unseen"

    @property
    def resolved_num_classes(self):
        if self.version == "v1":
            return 70
        elif self.version == "v2":
            return 84
        elif self.version == "v3":
            return 36
        else:
            raise ValueError(f"Unknown version '{self.version}'")

    @property
    def unseen_lang(self):
        mapping = {
            ("v1", "English"): "Urdu",
            ("v1", "Urdu"):    "English",
            ("v2", "English"): "Hindi",
            ("v2", "Hindi"):   "English",
            ("v3", "English"): "German",
            ("v3", "German"):  "English",
        }
        key = (self.version, self.seen_lang)
        if key not in mapping:
            raise ValueError(f"Invalid version '{self.version}' or seen_lang '{self.seen_lang}'.")
        return mapping[key]
