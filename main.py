import logging
import torch
from torch.utils.data import DataLoader

from config import ExperimentConfig

from utils.featLoader import LoadData
from utils.trainer import Trainer
from utils.evaluator import Evaluator
from utils.earlystop import EarlyStopping

from models.fop import FOP
from models.multibranch import MultiBranchFOP

import os
import json

def save_checkpoint(model, optimizer, config, epoch, metric_value, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "metric": metric_value,
        "early_stop_metric": config.early_stop_metric,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer else None,
        "config": vars(config),  # full experiment snapshot
    }

    torch.save(checkpoint, save_path)


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

    dataset = LoadData(
        csv_path=csv_path,
        config=config,
        audio_encoder="ecappa_feats_path",
        modality="audiovisual",
    )

    loader =  DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return dataset, loader


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
        "Seed=%d | Device=%s | Model=%s | Fusion=%s | Version=%s | Train_Lang=%s \
        \n#Classes=%d | UnSeen_Lang=%s | Missing=%s | Ratio=%.2f",
        config.seed,
        config.device,
        config.model_type,
        config.fusion,
        config.version,
        config.seen_lang,
        config.resolved_num_classes,
        config.unseen_lang,
        config.train_missing_modality,
        config.missing_ratio,
    )

    # --------------------------------------------------
    # CSV paths
    # --------------------------------------------------
    train_csv = f"./feature_tracker/{config.version}_train_{config.seen_lang}.csv"
    test_csv = f"./feature_tracker/{config.version}_val_{config.seen_lang}.csv"
    unseen_csv = f"./feature_tracker/{config.version}_val_{config.unseen_lang}.csv"

    logger.info("Train CSV: %s", train_csv)
    logger.info("Test  CSV: %s", test_csv)
    logger.info("Unseen CSV: %s", unseen_csv)

    # --------------------------------------------------
    # DataLoaders
    # --------------------------------------------------
    _, train_loader = make_loader(train_csv, config, shuffle=True, logger=logger)
    test_dataset, _ = make_loader(test_csv, config, shuffle=False, logger=logger)
    unseen_test_dataset, _ = make_loader(unseen_csv, config, shuffle=False, logger=logger)

    # --------------------------------------------------
    # Infer feature dimensions
    # --------------------------------------------------
    audio, face, _ = next(iter(train_loader))
    logger.info(
        "Feature dimensions | Audio=%d | Face=%d",
        audio.shape[1],
        face.shape[1],
    )

    if config.model_type == "FOP":

        model = FOP(
            config=config,
            face_dim=face.shape[1],
            voice_dim=audio.shape[1],
        )

        # print(model)
    
    elif config.model_type == "multibranch":
        model = MultiBranchFOP(
            config=config,
            face_dim=face.shape[1],
            voice_dim=audio.shape[1]
        )
    else:
        raise ValueError(f"Unknown model_type: {config.model_type}")

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

        best_metric = -float("inf")
        best_epoch = -1

        save_path = (
            f"./checkpoints/"
            f"{config.version}_"
            f"{config.seen_lang}_"
            f"alpha{alpha}_"
            f"best.pt"
        )

        early_stopper = EarlyStopping(
        patience=config.early_stop_patience,
        min_delta=config.early_stop_min_delta,
        )

        for epoch in range(config.max_epochs):
            loss = trainer.train_epoch(train_loader, alpha)

            acc_seen = evaluator.accuracy(test_dataset)
            acc_unseen = evaluator.accuracy(unseen_test_dataset)

            monitor_value = (
                acc_seen
                if config.early_stop_metric == "seen"
                else acc_unseen
            )

            if monitor_value > best_metric:
                best_metric = monitor_value
                best_epoch = epoch

                save_checkpoint(
                    model=model,
                    optimizer=trainer.opt,
                    config=config,
                    epoch=epoch,
                    metric_value=monitor_value,
                    save_path=save_path,
                )

            logger.info(
                "[α=%.3f] Epoch %03d | Loss %.4f | Seen Val %.2f | Unseen Val %.2f",
                alpha,
                epoch,
                loss,
                acc_seen,
                acc_unseen,
            )

            if config.early_stop:
                if early_stopper.step(monitor_value):
                    logger.info(
                        "Early stopping triggered at epoch %d" \
                        "(best %s accuracy = %0.2f)",
                        epoch,
                        config.early_stop_metric,
                        early_stopper.best_score,

                    )
                    break

    logger.info("=== Experiment finished ===")

if __name__ == "__main__":
    main()
