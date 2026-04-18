import torch
import pandas as pd
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import lightning as L
import torchxrayvision as xrv


LABELS = [
    "Atelectasis", "Consolidation", "Edema", "Effusion", "Emphysema",
    "Fibrosis", "Infiltration", "Mass", "Nodule",
    "Pleural_Thickening", "Pneumonia", "Pneumothorax", "No Finding",
]


class XrayDataset(Dataset):
    def __init__(self, csv_path, data_dir, transform):
        self.df        = pd.read_csv(csv_path)
        self.data_dir  = Path(data_dir)
        self.transform = transform
        self.labels    = [l for l in LABELS if l in self.df.columns]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        img   = Image.open(self.data_dir / row["image_path"]).convert("L")
        label = torch.tensor(1.0 - float(row["No Finding"]), dtype=torch.float32)
        return self.transform(img), label


class XrayDataModule(L.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg      = cfg
        self.train_tf = transforms.Compose([
            transforms.Resize((224, 224)),     
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
            transforms.RandomAutocontrast(p=0.3),
            transforms.ToTensor(),
            transforms.Lambda(
                lambda x: xrv.datasets.normalize(x, maxval=1.0, reshape=True)
            ),
        ])
        self.val_tf   = transforms.Compose([
            transforms.Resize((224, 224)),  
            transforms.ToTensor(),
            transforms.Lambda(
                lambda x: xrv.datasets.normalize(x, maxval=1.0, reshape=True)
            ),
        ])

    def _loader(self, split, transform, shuffle=False):
        ds = XrayDataset(f"{self.cfg.data_dir}/{split}.csv", self.cfg.data_dir, transform)
        return DataLoader(
            ds,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=shuffle,
            pin_memory=self.cfg.pin_memory,
            prefetch_factor=self.cfg.prefetch_factor,
            persistent_workers=self.cfg.persistent_workers,
        )

    def train_dataloader(self): return self._loader("train", self.train_tf, shuffle=True)
    def val_dataloader(self):   return self._loader("val",   self.val_tf)
    def test_dataloader(self):  return self._loader("test",  self.val_tf)