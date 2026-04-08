import torch
import torch.nn as nn
from torchvision import models
from torchmetrics import AUROC, Accuracy, Precision, Recall, F1Score
from torchmetrics import MetricCollection
from omegaconf import OmegaConf
import lightning as L


class XrayClassifier(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))
        self.cfg = cfg

        # backbone 
        backbone    = getattr(models, cfg.backbone)(weights="DEFAULT" if cfg.pretrained else None)
        backbone.fc = nn.Linear(backbone.fc.in_features, 1)
        self.model  = backbone

        # Freeze the first layers (default 50)
        self._freeze_layers(cfg.frozen_layers)

        self.loss = nn.BCEWithLogitsLoss()

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

    # ── layer freezing ────────────────────────────────────────────────────────
    def _freeze_layers(self, n: int):
        params = list(self.model.parameters())
        n = min(n, len(params))          # guard against overshooting
        for param in params[:n]:
            param.requires_grad = False
        print(f"[XrayClassifier] Frozen {n}/{len(params)} parameter tensors.")

    def unfreeze_all(self):
        for param in self.model.parameters():
            param.requires_grad = True

    # forward
    def forward(self, x):
        return self.model(x)

    # shared step
    def _step(self, batch, stage: str):
        x, y   = batch
        logits = self(x).squeeze(1)
        loss   = self.loss(logits, y.float())
        probs  = torch.sigmoid(logits).detach()

        {"train": self.train_metrics,
         "val":   self.val_metrics,
         "test":  self.test_metrics}[stage].update(probs, y.int())

        self.log(f"{stage}/loss", loss, prog_bar=True,
                 on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def training_step(self, batch, _):   return self._step(batch, "train")
    def validation_step(self, batch, _): return self._step(batch, "val")
    def test_step(self, batch, _):       return self._step(batch, "test")

    #  epoch-end hooks (log accumulated metrics) 
    def on_train_epoch_end(self):
        self.log_dict({f"train/{k}": v for k, v in self.train_metrics.compute().items()},
                      prog_bar=True, sync_dist=True,)
        self.train_metrics.reset()

    def on_validation_epoch_end(self):
        self.log_dict({f"val/{k}": v for k, v in self.val_metrics.compute().items()},
                      prog_bar=True, sync_dist=True)
        self.val_metrics.reset()

    def on_test_epoch_end(self):
        self.log_dict({f"test/{k}": v for k, v in self.test_metrics.compute().items()},
                      prog_bar=True, sync_dist=True,)
        self.test_metrics.reset()

    # optimiser + scheduler
    def configure_optimizers(self):
        trainable = [p for p in self.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(trainable, lr=self.cfg.lr,
                                weight_decay=self.cfg.weight_decay)

        # Warmup (default 10%) of training, then cosine decay — much smoother than
        warmup_steps = max(1, int(self.trainer.max_epochs * self.cfg.warmup))
        def lr_lambda(epoch):
            if epoch < warmup_steps:
                return epoch / warmup_steps          # linear warm-up
            progress = (epoch - warmup_steps) / max(1, self.trainer.max_epochs - warmup_steps)
            return 0.5 * (1.0 + torch.cos(torch.tensor(3.14159 * progress)).item())

        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return {
            "optimizer":    opt,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }