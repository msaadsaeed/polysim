import logging
import torch
from torch.utils.data import DataLoader

from config import ExperimentConfig
from featLoader import LoadData
from model import FOP
from trainer import Trainer
from evaluator import Evaluator
from earlystop import EarlyStopping

def setup_logger(config):
    logger = logging.getLogger("Experiment")
    logger.setLevel(config.log_level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(levelname)s][%(name)s] %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def make_loader(csv_path, config, shuffle=False, logger=None):
    # if logger:
    #     logger.info("Creating DataLoader for %s (shuffle=%s)", csv_path, shuffle)

    dataset = LoadData(
        csv_path=csv_path,
        config=config,
        audio_encoder="ecappa_feats_path",
        modality="audiovisual",
    )

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
    )


def main():
    # --------------------------------------------------
    # Config & reproducibility
    # --------------------------------------------------
    config = ExperimentConfig()
    torch.manual_seed(config.seed)

    if config.device == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    logger = setup_logger(config)

    logger.info("=== Experiment started ===")
    logger.info(
        "Seed=%d | Device=%s | Fusion=%s | Version=%s | Train_Lang=%s | #Classes=%d | UnSeen_Lang=%s | Missing=%s | Ratio=%.2f",
        config.seed,
        config.device,
        config.fusion,
        config.version,
        config.seen_lang,
        config.resolved_num_classes,
        config.unseen_lang,
        config.missing_modality,
        config.missing_ratio,
    )

    # --------------------------------------------------
    # CSV paths
    # --------------------------------------------------
    train_csv = f"./feature_tracker/{config.version}_train_{config.seen_lang}.csv"
    test_csv = f"./feature_tracker/{config.version}_test_{config.seen_lang}.csv"
    unseen_csv = f"./feature_tracker/{config.version}_test_{config.unseen_lang}.csv"

    logger.info("Train CSV: %s", train_csv)
    logger.info("Test  CSV: %s", test_csv)
    logger.info("Unseen CSV: %s", unseen_csv)

    # --------------------------------------------------
    # DataLoaders
    # --------------------------------------------------
    train_loader = make_loader(train_csv, config, shuffle=True, logger=logger)
    test_loader = make_loader(test_csv, config, shuffle=False, logger=logger)
    unseen_loader = make_loader(unseen_csv, config, shuffle=False, logger=logger)

    # --------------------------------------------------
    # Infer feature dimensions
    # --------------------------------------------------
    audio, face, _ = next(iter(train_loader))
    logger.info(
        "Feature dimensions | Audio=%d | Face=%d",
        audio.shape[1],
        face.shape[1],
    )

    model = FOP(
        config=config,
        face_dim=face.shape[1],
        voice_dim=audio.shape[1],
    )

    logger.info(
        "Model initialized | Params=%.2fM",
        sum(p.numel() for p in model.parameters()) / 1e6,
    )

    # --------------------------------------------------
    # Trainer & evaluator
    # --------------------------------------------------
    trainer = Trainer(model, config)
    evaluator = Evaluator(model, config)

    # --------------------------------------------------
    # Training loop
    # --------------------------------------------------
    for alpha in config.alpha_list:
        logger.info("=== Training with alpha=%.3f ===", alpha)

        early_stopper = EarlyStopping(
        patience=config.early_stop_patience,
        min_delta=config.early_stop_min_delta,
        )

        for epoch in range(config.max_epochs):
            loss = trainer.train_epoch(train_loader, alpha)

            acc_seen = evaluator.accuracy(test_loader)
            acc_unseen = evaluator.accuracy(unseen_loader)

            monitor_value = (
                acc_seen
                if config.early_stop_metric == "seen"
                else acc_unseen
            )

            logger.info(
                "[α=%.3f] Epoch %03d | Loss %.4f | Seen %.2f | Unseen %.2f",
                alpha,
                epoch,
                loss,
                acc_seen,
                acc_unseen,
            )

            if config.early_stop:
                if early_stopper.step(monitor_value):
                    logger.info(
                        "Early stopping triggered at epoch %d",
                        "(best %s accuracy = %0.2f)".
                        epoch,
                        config.early_stop_metric,
                        early_stopper.best_score,

                    )
                    break

    logger.info("=== Experiment finished ===")

if __name__ == "__main__":
    main()
