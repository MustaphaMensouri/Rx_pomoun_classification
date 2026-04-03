import torch
import torch.nn as nn
from torchvision import models
from torchmetrics import AUROC, Accuracy, Precision, Recall, F1Score

import lightning as L

class XrayClassifier(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(ignore=["cfg"])
        self.cfg = cfg

        backbone    = getattr(models, cfg.backbone)(weights="DEFAULT" if cfg.pretrained else None)
        backbone.fc = nn.Linear(backbone.fc.in_features, 1)
        self.model  = backbone

        self.loss = nn.BCEWithLogitsLoss()
        self.auc  = AUROC(task="binary")
        self.acc  = Accuracy(task="binary")
        self.prec = Precision(task="binary")
        self.recall = Recall(task="binary")
        self.f1 = F1Score(task="binary")

    def _step(self, batch, stage):
        x, y   = batch
        logits = self(x).squeeze(1)
        loss   = self.loss(logits, y)
        probs  = torch.sigmoid(logits)
        self.log_dict(
            {f"{stage}/loss": loss, f"{stage}/auc": self.auc(probs, y.int()), f"{stage}/acc": self.acc(probs, y.int()), f"{stage}/prec": self.prec(probs, y.int()), f"{stage}/recall": self.recall(probs, y.int()), f"{stage}/f1": self.f1(probs, y.int())},
            prog_bar=True, on_step=False, on_epoch=True, sync_dist=True,
        )
        return loss

    def training_step(self, batch, _):   return self._step(batch, "train")
    def validation_step(self, batch, _): return self._step(batch, "val")
    def test_step(self, batch, _):       return self._step(batch, "test")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.max_epochs)
        return {"optimizer": opt, "lr_scheduler": scheduler}