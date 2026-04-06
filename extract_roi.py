import argparse
import os
import warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_gamma_lut(gamma: float = 1.5) -> np.ndarray:
    inv_gamma = 1.0 / gamma
    return np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in range(256)],
        dtype=np.uint8,
    )

def apply_gamma(rgb: np.ndarray, lut: np.ndarray) -> np.ndarray:
    return cv2.LUT(rgb, lut)

def refine_mask(mask: np.ndarray, dilation_kernel: int = 30) -> np.ndarray:
    kernel  = np.ones((dilation_kernel, dilation_kernel), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return dilated
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
    combined = np.vstack(contours)
    hull     = cv2.convexHull(combined)
    hull_mask = np.zeros_like(mask)
    cv2.drawContours(hull_mask, [hull], -1, 1, thickness=cv2.FILLED)
    return hull_mask

def save_roi(
    rgb_512:      np.ndarray,
    mask_512:     np.ndarray,
    dst_path:     Path,
    final_size:   int,
    jpeg_quality: int,
) -> bool:
    try:
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        if mask_512.ndim == 3:
            mask_512 = np.squeeze(mask_512, axis=-1)
        binary = (mask_512 > 0.5).astype(np.uint8)

        hull = refine_mask(binary)

        roi = (rgb_512 * hull[:, :, np.newaxis]).astype(np.uint8)
        roi = cv2.resize(roi, (final_size, final_size), interpolation=cv2.INTER_LANCZOS4)

        bgr = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(dst_path), bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        return True
    except Exception as exc:
        warnings.warn(f"[warn] {dst_path.name}: {exc}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# tf.data loader
# ─────────────────────────────────────────────────────────────────────────────

def make_dataset(src_paths, seg_size, gamma_lut, batch_size, num_workers):

    def load_and_preprocess(path):
        raw = tf.io.read_file(path)
        img = tf.image.decode_image(raw, channels=1, expand_animations=False)
        img = tf.image.resize(img, (seg_size, seg_size))
        img = tf.cast(img, tf.float32)
        img_rgb = tf.image.grayscale_to_rgb(img)  # (H,W,3)

        # Normalize exactly like training: divide by 255
        model_input = img_rgb / 255.0

        # Keep a uint8 copy for saving the ROI crop
        img_uint8 = tf.cast(img_rgb, tf.uint8)

        return img_uint8, model_input

    return (
        tf.data.Dataset.from_tensor_slices(src_paths)
        .map(load_and_preprocess, num_parallel_calls=num_workers, deterministic=True)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Split processor
# ─────────────────────────────────────────────────────────────────────────────

def process_split(
    csv_path:        Path,
    prep_dir:        Path,
    out_dir:         Path,
    model,
    gamma_lut:       np.ndarray,
    seg_size:        int,
    final_size:      int,
    batch_size:      int,
    jpeg_quality:    int,
    num_workers:     int,
    checkpoint_file: Path,
) -> None:
    split_name = csv_path.stem
    df = pd.read_csv(csv_path)

    all_records = [
        (prep_dir / row["image_path"],
         out_dir / "images" / (Path(row["image_path"]).stem + ".jpg"))
        for _, row in df.iterrows()
    ]

    done: set = set()
    if checkpoint_file.exists():
        done = set(checkpoint_file.read_text().splitlines())
        print(f"  [{split_name}] resuming – {len(done):,} already done, "
              f"{len(all_records) - len(done):,} remaining")

    records = [(s, d) for s, d in all_records if d.name not in done]
    if not records:
        print(f"  [{split_name}] nothing to do.")
        _write_csv(df, out_dir, split_name, failed=set())
        return

    src_paths = [str(s) for s, _ in records]
    dst_paths = [d       for _, d in records]

    ds = make_dataset(src_paths, seg_size, gamma_lut, batch_size, num_workers)

    failed_stems: set = set()
    processed = 0

    with ThreadPoolExecutor(max_workers=num_workers) as pool, \
         open(checkpoint_file, "a") as ckpt_f:

        pending_futures: dict = {}
        pbar = tqdm(total=len(records), desc=split_name, unit="img")

        for imgs_gamma_batch, model_input_batch in ds:
            batch_n  = imgs_gamma_batch.shape[0]
            masks_np = model.predict(model_input_batch, verbose=0)
            imgs_np  = imgs_gamma_batch.numpy()

            for i in range(batch_n):
                dst = dst_paths[processed + i]
                future = pool.submit(
                    save_roi,
                    imgs_np[i],
                    masks_np[i],
                    dst,
                    final_size,
                    jpeg_quality,
                )
                pending_futures[future] = dst

            processed += batch_n

            still_pending: dict = {}
            for future, dst in pending_futures.items():
                if future.done():
                    ok = future.result()
                    if ok:
                        ckpt_f.write(dst.name + "\n")
                        ckpt_f.flush()
                    else:
                        failed_stems.add(dst.stem)
                    pbar.update(1)
                else:
                    still_pending[future] = dst
            pending_futures = still_pending

        for future in as_completed(pending_futures):
            dst = pending_futures[future]
            ok  = future.result()
            if ok:
                ckpt_f.write(dst.name + "\n")
                ckpt_f.flush()
            else:
                failed_stems.add(dst.stem)
            pbar.update(1)

        pbar.close()

    _write_csv(df, out_dir, split_name, failed_stems)


def _write_csv(df: pd.DataFrame, out_dir: Path, split_name: str, failed: set) -> None:
    out_df = df.copy()
    out_df["image_path"] = out_df["image_path"].apply(
        lambda p: "images/" + Path(p).stem + ".jpg"
    )
    if failed:
        before = len(out_df)
        out_df = out_df[
            ~out_df["image_path"].apply(lambda p: Path(p).stem).isin(failed)
        ]
        print(f"  [{split_name}] dropped {before - len(out_df)} failed images")

    out_csv = out_dir / f"{split_name}.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"  [{split_name}] saved {len(out_df):,} rows → {out_csv}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preprocessed_dir", required=True)
    p.add_argument("--output_dir",       default="./data/segmented")
    p.add_argument("--model_path",       required=True)
    p.add_argument("--splits",     nargs="+", default=["train", "val", "test"])
    p.add_argument("--batch_size",   type=int,   default=32)
    p.add_argument("--num_workers",  type=int,   default=8)
    p.add_argument("--gamma",        type=float, default=1.5)
    p.add_argument("--seg_size",     type=int,   default=512)
    p.add_argument("--final_size",   type=int,   default=224)
    p.add_argument("--jpeg_quality", type=int,   default=95)
    args = p.parse_args()

    prep_dir = Path(args.preprocessed_dir)
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "images").mkdir(exist_ok=True)

    # ── Load full model from .keras file ─────────────────────────────────
    print(f"Loading model: {args.model_path}")
    model = tf.keras.models.load_model(args.model_path)
    model.trainable = False
    gamma_lut = build_gamma_lut(args.gamma)

    for split in args.splits:
        csv_path = prep_dir / f"{split}.csv"
        if not csv_path.exists():
            print(f"  [skip] {csv_path} not found")
            continue
        print(f"\n── {split} ──────────────────────────────────────────────")
        checkpoint_file = out_dir / f".{split}_done.txt"
        process_split(
            csv_path        = csv_path,
            prep_dir        = prep_dir,
            out_dir         = out_dir,
            model           = model,
            gamma_lut       = gamma_lut,
            seg_size        = args.seg_size,
            final_size      = args.final_size,
            batch_size      = args.batch_size,
            jpeg_quality    = args.jpeg_quality,
            num_workers     = args.num_workers,
            checkpoint_file = checkpoint_file,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()