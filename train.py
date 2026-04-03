import hydra
import wandb
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig

from src.datamodule import XrayDataModule, LABELS
from src.lightning_module import XrayClassifier

import torch
torch.serialization.add_safe_globals([DictConfig])


@hydra.main(config_path="configs", config_name="config", version_base=None)
def train(cfg: DictConfig):
    L.seed_everything(42)

    dm     = XrayDataModule(cfg.data)
    model  = XrayClassifier(cfg.model, num_classes=len(LABELS), max_epochs=cfg.train.max_epochs)
    logger = WandbLogger(project=cfg.wandb.project, name=cfg.wandb.name)

    trainer = L.Trainer(
        max_epochs=cfg.train.max_epochs,
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        strategy=cfg.train.strategy,
        precision=cfg.train.precision,
        log_every_n_steps=cfg.train.log_every_n_steps,
        enable_progress_bar=cfg.train.enable_progress_bar,
        enable_model_summary=cfg.train.enable_model_summary,
        logger=logger,
        callbacks=[
            ModelCheckpoint(monitor="val/auc", mode="max", save_top_k=1, filename="best"),
            EarlyStopping(monitor="val/auc",   mode="max", patience=5),
            TQDMProgressBar(refresh_rate=50),
        ],
        gradient_clip_val=1.0,
    )

    trainer.fit(model, dm)
    trainer.test(model, dm, ckpt_path="best")
    wandb.finish()


if __name__ == "__main__":
    train()