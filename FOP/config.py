from dataclasses import dataclass
from typing import List
import logging

@dataclass
class ExperimentConfig:
    home_dir = "D:/mavceleb/"
    seed: int = 1
    device: str = "cuda"
    lr: float = 1e-3
    batch_size: int = 32
    max_epochs: int = 300
    num_workers = 8
    alpha_list: List[float] = (1.0,)
    embedding_dim: int = 512
    fusion: str = "linear"   # linear | gated
    
    version: str = "v3"
    seen_lang: str = "English"

    train_missing_modality = "face"
    missing_ratio = 0.1 # 0.0 - 1.0

    debug: bool = False
    log_level = logging.DEBUG if debug else logging.INFO

    early_stop: bool = True
    early_stop_patience: int = 10      # tolerance (epochs)
    early_stop_min_delta: float = 0.2 # minimum improvement
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
        if self.version == "v1" and self.seen_lang == "English":
            return "Urdu"
        elif self.version == "v2" and self.seen_lang == "English":
            return "Hindi"
        elif self.version == "v3" and self.seen_lang == "English":
            return "German"
        else:
            raise ValueError(f"Invalid version '{self.version}' or seen_lang '{self.seen_lang}'.")
        return "English"
