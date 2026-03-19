# src/data/download_images.py
#
# Downloads Legacy Survey JPEG cutouts for a stratified sample of labeled galaxies.
# Resumable: skips files that already exist.
#
# Usage:
#   uv run python -m src.data.download_images
#   uv run python -m src.data.download_images --per-class 10000 --workers 8

import argparse
import time
import urllib.request
from pathlib import Path

import pandas as pd
from tqdm import tqdm


TABLES    = Path("input/tables")
IMG_DIR   = Path("input/images")
FAIL_LOG  = TABLES / "image_download_failures.txt"

LABELED_INDEX = TABLES / "labeled_index.csv"

# Legacy Survey cutout API
# pixscale=0.262 gives ~53x53 arcsec field at 224px (good for ResNet input)
IMG_URL = (
    "https://www.legacysurvey.org/viewer/jpeg-cutout"
    "?ra={ra}&dec={dec}&size=224&layer=ls-dr10&pixscale=0.262"
)


def build_sample(labeled_index: pd.DataFrame, per_class: int) -> pd.DataFrame:
    """Stratified sample: up to per_class rows per label."""
    parts = []
    for label, group in labeled_index.groupby("label"):
        n = min(len(group), per_class)
        parts.append(group.sample(n=n, random_state=42))
    sample = pd.concat(parts).reset_index(drop=True)
    print(f"Download sample: {len(sample):,} galaxies")
    print(sample["label"].value_counts().to_string())
    return sample


def download_images(
    df: pd.DataFrame,
    img_dir: Path = IMG_DIR,
    fail_log: Path = FAIL_LOG,
    sleep: float = 0.05,
    workers: int = 1,
) -> None:
    """
    Download Legacy Survey JPEG cutouts.

    Files saved as: input/images/{objid}.jpeg
    Failed objids logged to: input/tables/image_download_failures.txt
    """
    img_dir.mkdir(parents=True, exist_ok=True)
    fail_log.parent.mkdir(parents=True, exist_ok=True)

    todo = df[~df["objid"].apply(lambda x: (img_dir / f"{x}.jpeg").exists())]
    print(f"{len(df) - len(todo):,} already downloaded, {len(todo):,} remaining")

    failures = []

    if workers > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _fetch(row):
            return _download_one(row, img_dir, sleep)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_fetch, row): row for _, row in todo.iterrows()}
            for future in tqdm(as_completed(futures), total=len(futures), desc="images"):
                objid, ok = future.result()
                if not ok:
                    failures.append(objid)
    else:
        for _, row in tqdm(todo.iterrows(), total=len(todo), desc="images"):
            objid, ok = _download_one(row, img_dir, sleep)
            if not ok:
                failures.append(objid)

    if failures:
        with open(fail_log, "a") as f:
            for objid in failures:
                f.write(f"{objid}\n")
        print(f"{len(failures):,} failures logged to {fail_log}")

    print(f"Done. {len(todo) - len(failures):,} downloaded, {len(failures):,} failed.")


def _download_one(row, img_dir: Path, sleep: float) -> tuple:
    objid = row["objid"]
    out = img_dir / f"{objid}.jpeg"
    if out.exists():
        return objid, True
    url = IMG_URL.format(ra=row["ra"], dec=row["dec"])
    try:
        urllib.request.urlretrieve(url, out)
        time.sleep(sleep)
        return objid, True
    except Exception:
        return objid, False


def main():
    parser = argparse.ArgumentParser(description="Download Legacy Survey galaxy images")
    parser.add_argument("--input",     default=str(LABELED_INDEX))
    parser.add_argument("--per-class", type=int, default=10_000)
    parser.add_argument("--sleep",     type=float, default=0.05)
    parser.add_argument("--workers",   type=int, default=1,
                        help="Parallel download threads (be polite: <=8)")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    sample = build_sample(df, args.per_class)

    # Save the sample so download_spectra uses the same set
    sample_path = TABLES / "download_sample.csv"
    sample.to_csv(sample_path, index=False)
    print(f"Sample saved to {sample_path}")

    download_images(sample, sleep=args.sleep, workers=args.workers)


if __name__ == "__main__":
    main()
