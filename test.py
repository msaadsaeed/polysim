import torch
import numpy as np

from config import ExperimentConfig
from utils.featLoader import LoadData
from models.fop import FOP
from utils.evaluator import Evaluator


def apply_missing(face, audio, pct, modality, seed=0):
    """
    Zero out `pct`% of samples for the given modality.
    """
    assert modality in ["voice", "face"]

    N = face.shape[0]
    k = int((pct / 100.0) * N)

    rng = torch.Generator(device=face.device)
    rng.manual_seed(seed)

    idx = torch.randperm(N, generator=rng, device=face.device)[:k]

    face_m = face.clone()
    audio_m = audio.clone()

    if modality == "voice":
        audio_m[idx] = 0
    else:
        face_m[idx] = 0

    return face_m, audio_m


def sweep_missing(
    evaluator,
    dataset,
    modality,
    step=5,
    seed=0,
):
    """
    Sweep missing modality percentage and report accuracy.
    """
    face, audio, labels = evaluator._get_tensors(dataset)

    face = face.to(evaluator.config.device)
    audio = audio.to(evaluator.config.device)
    labels = labels.to(evaluator.config.device)

    results = []

    for pct in range(0, 101, step):
        face_m, audio_m = apply_missing(
            face, audio, pct, modality, seed
        )

        acc = evaluator.accuracy_from_tensors(
            face_m, audio_m, labels
        )

        results.append((pct, acc))
        print(f"[{modality.upper()} MISS {pct:3d}%] ACC = {acc:.2f}")

    return results


def main():
    # --------------------------------------------------
    # Config
    # --------------------------------------------------
    config = ExperimentConfig()
    config.debug = False
    device = torch.device(config.device)

    torch.manual_seed(config.seed)

    # --------------------------------------------------
    # Load test dataset (in-memory)
    # --------------------------------------------------
    test_csv = f"./feature_tracker/{config.version}_test_{config.seen_lang}.csv"
    unseen_test_csv = f"./feature_tracker/{config.version}_test_{config.unseen_lang}.csv"

    test_dataset = LoadData(
        csv_path=test_csv,
        config=config,
        audio_encoder="ecappa_feats_path",
        modality="audiovisual",
    )

    unseen_test_dataset = LoadData(
        csv_path=unseen_test_csv,
        config=config,
        audio_encoder="ecappa_feats_path",
        modality="audiovisual",
    )

    # --------------------------------------------------
    # Load model
    # --------------------------------------------------
    face_dim = test_dataset.face_feats.shape[1]
    audio_dim = test_dataset.audio_feats.shape[1]

    model = FOP(
        config=config,
        face_dim=face_dim,
        voice_dim=audio_dim,
    ).to(device)

    checkpoint_path = f"./checkpoints/{config.version}_{config.seen_lang}_alpha{config.test_alpha}_best.pt"
    ckpt = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    evaluator = Evaluator(model, config)

    # --------------------------------------------------
    # Missing-modality sweeps
    # --------------------------------------------------
    print(f"\n=== Version: {config.version} | Seen language: {config.seen_lang} | {config.test_missing_modality.upper()} missing sweep ===")
    sweep_missing(
        evaluator,
        test_dataset,
        modality=f"{config.test_missing_modality}",
        step=5,
        seed=config.seed,
    )

    print(f"\n=== Version: {config.version} | Unseen language {config.unseen_lang} | {config.test_missing_modality.upper()} missing sweep ===")
    sweep_missing(
        evaluator,
        unseen_test_dataset,
        modality=f"{config.test_missing_modality}",
        step=5,
        seed=config.seed,
    )

if __name__ == "__main__":
    main()
