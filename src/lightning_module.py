import torch
import torch.nn as nn
import torchxrayvision as xrv
from torchmetrics import AUROC, Accuracy, Precision, Recall, F1Score, MetricCollection
from omegaconf import OmegaConf
import lightning as L
import math


class XrayClassifier(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))
        self.cfg = cfg

        # ── backbone ──────────────────────────────────────────────────────────
        if cfg.backbone == "densenet121_chex":
            backbone = xrv.models.DenseNet(weights="densenet121-res224-chex")
        elif cfg.backbone == "densenet121_mimic":
            backbone = xrv.models.DenseNet(weights="densenet121-res224-mimic_ch")
        else:
            backbone = xrv.models.DenseNet(weights=None)
        backbone.classifier = nn.Linear(backbone.classifier.in_features, 1)
        self.model  = backbone

        # Freeze the first `cfg.frozen_layers` layers (default 50)
        self._freeze_layers(getattr(cfg, "frozen_layers", 50))

        self.loss = nn.BCEWithLogitsLoss(pos_weight=cfg.get("pos_weight", 1.0))

        # ── separate MetricCollections per stage ──────────────────────────────
        def _metrics():
            return MetricCollection({
                "auc":    AUROC(task="binary"),
                "acc":    Accuracy(task="binary"),
                "prec":   Precision(task="binary"),
                "recall": Recall(task="binary"),
                "f1":     F1Score(task="binary"),
            })

        self.train_metrics = _metrics()
        self.val_metrics   = _metrics()
        self.test_metrics  = _metrics()
    def on_train_epoch_end(self):
        self.train_metrics.reset()

    def on_val_epoch_end(self):
        self.val_metrics.reset()

    # ── layer freezing ────────────────────────────────────────────────────────
    def _freeze_layers(self, n: int):
        params = list(self.model.parameters())
        n = min(n, len(params))
        for param in params[:n]:
            param.requires_grad = False
        print(f"[XrayClassifier] Frozen {n}/{len(params)} parameter tensors.")

    def unfreeze_all(self):
        for param in self.model.parameters():
            param.requires_grad = True

    # ── forward ───────────────────────────────────────────────────────────────
    def forward(self, x):
        return self.model(x)

    # ── shared step ───────────────────────────────────────────────────────────
    def _step(self, batch, stage: str):
        x, y   = batch
        logits = self(x).squeeze(1)
        loss   = self.loss(logits, y.float())
        probs  = torch.sigmoid(logits)

        metrics = {
            "train": self.train_metrics,
            "val":   self.val_metrics,
            "test":  self.test_metrics,
        }[stage]

        # update accumulators
        metrics.update(probs, y.int())

        # pass metric OBJECTS (not .compute()) to self.log_dict
        # Lightning syncs internal state across DDP ranks before computing
        self.log(f"{stage}/loss", loss,
                 prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log_dict(
            {f"{stage}/{k}": metrics[k] for k in metrics},
            prog_bar=True, on_step=False, on_epoch=True, sync_dist=True,
        )

        return loss

    def training_step(self, batch, _):   return self._step(batch, "train")
    def validation_step(self, batch, _): return self._step(batch, "val")
    def test_step(self, batch, _):       return self._step(batch, "test")

    # ── optimiser + scheduler ─────────────────────────────────────────────────
    def configure_optimizers(self):
        trainable = [p for p in self.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(trainable, lr=self.cfg.lr,
                                weight_decay=self.cfg.weight_decay)

        warmup_steps = max(1, int(self.trainer.max_epochs * 0.1))

        def lr_lambda(epoch):
            if epoch < warmup_steps:
                return epoch / warmup_steps
            progress = (epoch - warmup_steps) / max(1, self.trainer.max_epochs - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return {
            "optimizer":    opt,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }