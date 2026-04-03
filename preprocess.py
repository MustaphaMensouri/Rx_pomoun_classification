import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


LUNG_LABELS = [
    "Atelectasis", "Consolidation", "Edema", "Effusion", "Emphysema",
    "Fibrosis", "Infiltration", "Mass", "Nodule",
    "Pleural_Thickening", "Pneumonia", "Pneumothorax",
]
KEEP_LABELS = LUNG_LABELS + ["No Finding"]


def resize_one(args):
    src, dst = args
    try:
        with Image.open(src) as img:
            img.convert("RGB").resize((512, 512), Image.LANCZOS).save(dst, format="JPEG")
    except Exception as e:
        print(f"[warn] {src.name}: {e}")


def main(data_root, output_dir, val_split=0.1, num_workers=4):
    data_root, output_dir = Path(data_root), Path(output_dir)
    (output_dir / "images").mkdir(parents=True, exist_ok=True)

    # ── labels ───────────────────────────────────────────────────────────────
    df = pd.read_csv(data_root / "Data_Entry_2017.csv").rename(columns=str.strip)
    df = df[["Image Index", "Finding Labels", "Patient ID", "Patient Age", "Patient Gender", "View Position"]]

    labels = df["Finding Labels"].str.get_dummies(sep="|")
    df = pd.concat([df.drop(columns="Finding Labels"), labels], axis=1)

    keep_cols = [l for l in KEEP_LABELS if l in df.columns]
    df = df[df[keep_cols].any(axis=1)][["Image Index", "Patient ID", "Patient Age", "Patient Gender", "View Position", *keep_cols]]
    print(f"  {len(df):,} images kept ({df['No Finding'].sum():,} normal, {(~df['No Finding'].astype(bool)).sum():,} pathology)")

    # ── splits ────────────────────────────────────────────────────────────────
    train_val = set((data_root / "train_val_list.txt").read_text().split())
    test = set((data_root / "test_list.txt").read_text().split())

    tv_df = df[df["Image Index"].isin(train_val)]
    patients = tv_df["Patient ID"].unique()
    np.random.default_rng(42).shuffle(patients)
    cut = int(len(patients) * val_split)

    splits = {
        "train": tv_df[~tv_df["Patient ID"].isin(patients[:cut])],
        "val":   tv_df[ tv_df["Patient ID"].isin(patients[:cut])],
        "test":  df[df["Image Index"].isin(test)],
    }

    # ── resize ────────────────────────────────────────────────────────────────
    src_index = {p.name: p for d in data_root.glob("images_*/images") for p in d.glob("*.png")}
    all_files = set().union(*[set(s["Image Index"]) for s in splits.values()])
    tasks     = [(src_index[f], output_dir / "images" / f.replace(".png", ".jpg")) for f in all_files if f in src_index]

    with ProcessPoolExecutor(num_workers) as pool:
        list(tqdm(pool.map(resize_one, tasks), total=len(tasks), desc="resizing"))

    # ── save csvs ─────────────────────────────────────────────────────────────
    for name, split_df in splits.items():
        out = split_df.copy()
        out.insert(0, "image_path", "images/" + out["Image Index"].str.replace(".png", ".jpg", regex=False))
        out.to_csv(output_dir / f"{name}.csv", index=False)
        print(f"  {name}.csv → {len(out):,} rows")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",   required=True)
    p.add_argument("--output_dir",  default="./data/preprocessed")
    p.add_argument("--val_split",   type=float, default=0.1)
    p.add_argument("--num_workers", type=int,   default=4)
    args = p.parse_args()
    main(args.data_root, args.output_dir, args.val_split, args.num_workers)