import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset


class LoadData(Dataset):
    """
    Audiovisual dataset (fully in-memory).

    - No missing-modality logic
    - Designed for fast training & full-dataset evaluation
    """

    def __init__(
        self,
        csv_path: str,
        config,
        audio_encoder: str,
        modality: str = "audiovisual",
    ):
        # ---- sanity check ----
        assert modality == "audiovisual", (
            "This loader supports audiovisual data only."
        )

        self.audio_encoder = audio_encoder
        self.modality = modality

        # ---- read CSV once ----
        df = pd.read_csv(csv_path)
        self.num_samples = len(df)

        # ---- resolve paths ONCE ----
        audio_paths = [
            str((Path(config.home_dir) / p).resolve())
            for p in df[audio_encoder]
        ]
        face_paths = [
            str((Path(config.home_dir) / p).resolve())
            for p in df["facenet_feats_path"]
        ]
        labels = df["label"].astype(int).to_numpy()

        # ---- load EVERYTHING into memory ----
        audio_feats = []
        face_feats = []

        for i in range(self.num_samples):
            audio_feats.append(
                np.load(audio_paths[i]).astype("float32")
            )
            face_feats.append(
                np.load(face_paths[i]).astype("float32")
            )

        # ---- stack for cache-friendly access ----
        self.audio_feats = np.stack(audio_feats)   # (N, Da)
        self.face_feats = np.stack(face_feats)     # (N, Df)
        self.labels = labels                       # (N,)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return (
            self.audio_feats[idx],
            self.face_feats[idx],
            self.labels[idx],
        )


if __name__ == "__main__":
    import torch
    from config import ExperimentConfig

    config = ExperimentConfig()
    torch.manual_seed(config.seed)

    dataset = LoadData(
        csv_path="./feature_tracker/v1_test_English.csv",
        config=config,
        audio_encoder="ecappa_feats_path",
    )

    a, f, y = dataset[0]
    print(a.shape, f.shape, y)
