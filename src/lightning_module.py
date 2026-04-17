import torch
import torch.nn as nn
from torchvision import models
from torchmetrics import AUROC, Accuracy, Precision, Recall, F1Score, MetricCollection
from omegaconf import OmegaConf
from transformers import ViTForImageClassification, ViTModel
import lightning as L


class XrayClassifier(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))
        self.cfg = cfg

        # ── backbone ──────────────────────────────────────────────────────────
        vit = ViTModel.from_pretrained(cfg.vit_checkpoint)
        vit.gradient_checkpointing_enable()
        vit.pooler = None
        self.model = vit
        hidden_size = vit.config.hidden_size  # 768 for base, 1024 for large
        self.classifier = nn.Linear(hidden_size, 1)

        # Freeze the first `cfg.frozen_layers` layers (default 50)
        self._freeze_layers(getattr(cfg, "frozen_layers", 0))
        self.loss = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([cfg.get("pos_weight", 1.0)])  # handles class imbalance in NIH
        )


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

    # ── layer freezing ────────────────────────────────────────────────────────
    def _freeze_layers(self, n: int):
        for i, layer in enumerate(self.model.encoder.layer):
            if i < n:
                for param in layer.parameters():
                    param.requires_grad = False
        print(f"[XrayClassifier] Frozen first {n} ViT encoder blocks.")
    def unfreeze_all(self):
        for param in self.model.parameters():
            param.requires_grad = True

    # ── forward ───────────────────────────────────────────────────────────────
    def forward(self, x):
        outputs = self.model(pixel_values=x)
        cls_token = outputs.last_hidden_state[:, 0]  # [CLS] token
        return self.classifier(cls_token)

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
        param_groups = self._get_vit_param_groups()
        opt = torch.optim.AdamW(param_groups,
                                lr=self.cfg.lr,
                                weight_decay=self.cfg.weight_decay)

        warmup_steps = max(1, int(self.trainer.max_epochs * 0.1))

        def lr_lambda(epoch):
            if epoch < warmup_steps:
                return epoch / warmup_steps
            progress = (epoch - warmup_steps) / max(1, self.trainer.max_epochs - warmup_steps)
            return 0.5 * (1.0 + torch.cos(torch.tensor(3.14159 * progress)).item())

        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return {
            "optimizer":    opt,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
    
    def _get_vit_param_groups(self):
        """Layer-wise LR decay: deeper layers get higher LR."""
        decay = getattr(self.cfg, "layer_lr_decay", 0.85)
        num_layers = self.model.config.num_hidden_layers  # 12 for base

        param_groups = []

        # Classifier head — full LR
        param_groups.append({
            "params": list(self.classifier.parameters()),
            "lr": self.cfg.lr
        })

        # Encoder layers — decayed LR
        for i, layer in enumerate(reversed(self.model.encoder.layer)):
            lr = self.cfg.lr * (decay ** (i + 1))
            param_groups.append({
                "params": [p for p in layer.parameters() if p.requires_grad],
                "lr": lr
            })

        # Embeddings — lowest LR
        param_groups.append({
            "params": [p for p in self.model.embeddings.parameters() if p.requires_grad],
            "lr": self.cfg.lr * (decay ** (num_layers + 1))
        })

        return param_groups