# src/data/download_spectra.py
#
# Downloads SDSS DR19 spectrum FITS files for the download sample.
# Must be run after download_images.py (reuses download_sample.csv).
# Resumable: skips files that already exist.
#
# Usage:
#   uv run python -m src.data.download_spectra
#   uv run python -m src.data.download_spectra --workers 8

import argparse
import time
import urllib.request
from pathlib import Path

import pandas as pd
from tqdm import tqdm


TABLES     = Path("input/tables")
SPEC_DIR   = Path("input/spectra")
FAIL_LOG   = TABLES / "spectra_download_failures.txt"

DOWNLOAD_SAMPLE = TABLES / "download_sample.csv"

# SDSS DR19 spectra archive (lite = no sky/model extensions, smaller files)
SPEC_URL = (
    "https://data.sdss.org/sas/dr19/spectro/sdss/redux/26/"
    "spectra/lite/{plate:04d}/"
    "spec-{plate:04d}-{spec_mjd}-{fiberid:04d}.fits"
)


def spec_filename(plate: int, spec_mjd: int, fiberid: int) -> str:
    return f"spec-{plate:04d}-{spec_mjd}-{fiberid:04d}.fits"


def download_spectra(
    df: pd.DataFrame,
    spec_dir: Path = SPEC_DIR,
    fail_log: Path = FAIL_LOG,
    sleep: float = 0.05,
    workers: int = 1,
) -> None:
    """
    Download SDSS FITS spectra.

    Files saved as: input/spectra/spec-{plate}-{spec_mjd}-{fiberid}.fits
    Failed identifiers logged to: input/tables/spectra_download_failures.txt

    Note: ~2% of plate/mjd/fiberid combinations return 404. These are logged
    and handled in the dataset loader (zero-filled spectrum + has_spectrum=0 flag).
    """
    spec_dir.mkdir(parents=True, exist_ok=True)
    fail_log.parent.mkdir(parents=True, exist_ok=True)

    # Check which files are already present
    def _exists(row):
        fname = spec_filename(int(row["plate"]), int(row["spec_mjd"]), int(row["fiberid"]))
        return (spec_dir / fname).exists()

    todo = df[~df.apply(_exists, axis=1)]
    print(f"{len(df) - len(todo):,} already downloaded, {len(todo):,} remaining")

    failures = []

    if workers > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _fetch(row):
            return _download_one(row, spec_dir, sleep)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_fetch, row): row for _, row in todo.iterrows()}
            for future in tqdm(as_completed(futures), total=len(futures), desc="spectra"):
                key, ok = future.result()
                if not ok:
                    failures.append(key)
    else:
        for _, row in tqdm(todo.iterrows(), total=len(todo), desc="spectra"):
            key, ok = _download_one(row, spec_dir, sleep)
            if not ok:
                failures.append(key)

    if failures:
        with open(fail_log, "a") as f:
            for key in failures:
                f.write(f"{key}\n")
        print(f"{len(failures):,} failures logged to {fail_log}")

    print(f"Done. {len(todo) - len(failures):,} downloaded, {len(failures):,} failed.")


def _download_one(row, spec_dir: Path, sleep: float) -> tuple:
    plate    = int(row["plate"])
    spec_mjd = int(row["spec_mjd"])
    fiberid  = int(row["fiberid"])
    fname    = spec_filename(plate, spec_mjd, fiberid)
    out      = spec_dir / fname
    key      = fname

    if out.exists():
        return key, True

    url = SPEC_URL.format(plate=plate, spec_mjd=spec_mjd, fiberid=fiberid)
    try:
        urllib.request.urlretrieve(url, out)
        time.sleep(sleep)
        return key, True
    except Exception:
        return key, False


def build_working_set(df: pd.DataFrame, img_dir: Path, spec_dir: Path) -> pd.DataFrame:
    """
    After downloads, build working_set.csv flagging which files are present.

    has_image    : Legacy Survey JPEG exists
    has_spectrum : SDSS FITS spectrum exists
    in_working_set : has_image (spectrum can be zero-filled)
    """
    df = df.copy()

    df["has_image"] = df["objid"].apply(
        lambda x: (img_dir / f"{x}.jpeg").exists()
    )

    df["spec_path"] = df.apply(
        lambda r: str(spec_dir / spec_filename(
            int(r["plate"]), int(r["spec_mjd"]), int(r["fiberid"])
        )),
        axis=1,
    )
    df["has_spectrum"] = df["spec_path"].apply(lambda p: Path(p).exists())
    df["in_working_set"] = df["has_image"]

    return df


def main():
    parser = argparse.ArgumentParser(description="Download SDSS DR19 spectrum FITS files")
    parser.add_argument("--input",   default=str(DOWNLOAD_SAMPLE))
    parser.add_argument("--sleep",   type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel download threads (be polite: <=8)")
    parser.add_argument("--build-working-set", action="store_true",
                        help="After download, write input/tables/working_set.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    download_spectra(df, sleep=args.sleep, workers=args.workers)

    if args.build_working_set:
        working = build_working_set(df, Path("input/images"), SPEC_DIR)
        out = TABLES / "working_set.csv"
        working.to_csv(out, index=False)
        print(f"\nWorking set saved to {out}")
        print(f"  has_image    : {working['has_image'].sum():,}")
        print(f"  has_spectrum : {working['has_spectrum'].sum():,}")
        print(f"  both         : {(working['has_image'] & working['has_spectrum']).sum():,}")


if __name__ == "__main__":
    main()
