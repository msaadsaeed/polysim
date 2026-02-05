import os
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from torch.utils.data import Dataset


class LoadData(Dataset):
    """
    Audiovisual dataset with controlled missing-modality simulation.

    Configuration is fully driven by `config`.

    Rules:
    - Missing modality allowed ONLY for: audio OR face
    - Missing BOTH modalities is forbidden
    """

    def __init__(
        self,
        csv_path: str,
        config,
        audio_encoder: str,
        modality: str = "audiovisual",
    ):
        # ---- logger ----
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(config.log_level)

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "[%(levelname)s][%(name)s] %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        # ---- sanity checks ----
        assert modality == "audiovisual", (
            "Missing-modality simulation only makes sense for audiovisual data."
        )

        assert config.missing_modality in [None, "audio", "face"], (
            "config.missing_modality must be None, 'audio', or 'face'."
        )

        assert 0.0 <= config.missing_ratio <= 1.0, (
            "config.missing_ratio must be in [0, 1]."
        )

        # ---- config ----
        self.config = config
        self.audio_encoder = audio_encoder
        self.modality = modality
        self.home_dir = config.home_dir

        # ---- data ----
        self.df = pd.read_csv(csv_path)
        self.num_samples = len(self.df)

        # ---- missing-modality setup (fixed across epochs) ----
        self.missing_indices = set()
        if config.missing_modality is not None and config.missing_ratio > 0:
            rng = np.random.RandomState(config.seed)
            num_missing = int(self.num_samples * config.missing_ratio)
            self.missing_indices = set(
                rng.choice(self.num_samples, num_missing, replace=False)
            )

        # ---- logging summary ----
        self.logger.info(
            "Loaded dataset: %s | samples=%d | missing=%s | ratio=%.2f",
            csv_path,
            self.num_samples,
            config.missing_modality,
            config.missing_ratio,
        )

        if self.missing_indices:
            self.logger.info(
                "Missing-modality indices: %d / %d",
                len(self.missing_indices),
                self.num_samples,
            )

    def __len__(self):
        return self.num_samples

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_path(self, path: str) -> str:
        """Resolve feature path relative to home_dir if needed."""
        p = Path(path)

        if not p.is_absolute():
            p = Path(self.home_dir) / p

        resolved = p.resolve()

        if self.config.debug:
            self.logger.debug("Resolved path: %s -> %s", path, resolved)

        if not resolved.exists():
            self.logger.error("Feature file not found: %s", resolved)
            raise FileNotFoundError(resolved)

        return str(resolved)

    def _load_audio(self, row):
        path = self._resolve_path(row[self.audio_encoder])
        return np.load(path).astype("float32")

    def _load_face(self, row):
        path = self._resolve_path(row["facenet_feats_path"])
        return np.load(path).astype("float32")

    # ------------------------------------------------------------------
    # Main access
    # ------------------------------------------------------------------

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = int(row["label"])

        audio = self._load_audio(row)
        face = self._load_face(row)

        # Apply missing modality (audio OR face, never both)
        if idx in self.missing_indices:
            if self.config.missing_modality == "audio":
                audio = np.zeros_like(audio)
                if self.config.debug:
                    self.logger.debug("Audio missing at idx=%d", idx)

            elif self.config.missing_modality == "face":
                face = np.zeros_like(face)
                if self.config.debug:
                    self.logger.debug("Face missing at idx=%d", idx)

        return audio, face, label

if __name__ == "__main__":
    import torch
    from config import ExperimentConfig
    config = ExperimentConfig()
    torch.manual_seed(config.seed)
    
    dataset = LoadData(
        csv_path="./feature_tracker/v1_test_English.csv",
        config=config,
        audio_encoder="ecappa_feats_path"
    )

    for x in dataset:
        print(x[0].shape, x[1].shape, x[2])
        break

