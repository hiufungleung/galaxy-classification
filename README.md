# Galaxy Morphology Classification — DATA7901 (UQ)

Multi-modal galaxy morphology classification using SDSS DR19 data. Three modalities are trained independently and combined via late fusion:

| Modality | Model | Macro F1 | Notes |
| --- | --- | --- | --- |
| Tabular photometry | XGBoost | 0.775 | CPU, ~5 min |
| Galaxy images | ResNet-18 | 0.807 | GPU, ~1–2 hr |
| Optical spectra | 1D-CNN | 0.563 | GPU, ~30 min |
| **Late fusion** | MLP head | **0.873** | GPU, ~1 hr |

Three morphological classes: **elliptical**, **spiral**, **merger** (Galaxy Zoo 1 vote-fraction labels, threshold = 0.60).

---

## Table of Contents

1. [Repository Layout](#repository-layout)
2. [Environment Setup](#environment-setup)
3. [Data Acquisition](#data-acquisition)
4. [Running the Pipeline](#running-the-pipeline)
5. [Training Individual Models](#training-individual-models)
6. [Notebooks](#notebooks)
7. [Results & Evaluation](#results--evaluation)
8. [Project Architecture](#project-architecture)
9. [Troubleshooting](#troubleshooting)

---

## Repository Layout

```text
.
├── input/
│   ├── queries/                    # SQL files for SDSS CasJobs
│   │   ├── DATA7901_DR19_casjobs_merged.sql   # recommended: explicit columns, aliases
│   │   └── ...                    # exploratory / partial queries
│   ├── tables/                    # CSV exports (gitignored — large files)
│   │   ├── DATA7901_DR19_merged.csv           # 507k rows, 723 columns (main table)
│   │   ├── labeled_index.csv                  # 214k labeled rows (output of Step 0)
│   │   ├── working_set.csv                    # 142k rows with both image + spectrum
│   │   ├── split_train.csv / split_val.csv / split_test.csv   # working-set split
│   │   └── split_train_full.csv / split_val_full.csv          # full-data split
│   ├── images/                    # JPEG cutouts from DESI Legacy Survey (gitignored)
│   └── spectra/                   # FITS files from SDSS DR19 archive (gitignored)
│
├── src/
│   ├── data/
│   │   ├── rename.py              # Column rename map (resolves CasJobs collisions)
│   │   ├── labels.py              # make_labels() — threshold-based label assignment
│   │   ├── features.py            # engineer_features(), handle_missing(), TABULAR_FEATURES
│   │   └── split.py               # Stratified 70/15/15 split → input/tables/
│   ├── datasets/
│   │   ├── image.py               # GalaxyImageDataset (PyTorch)
│   │   ├── spectral.py            # SpectralDataset + load_spectrum() resampler
│   │   └── multimodal.py          # MultiModalDataset (all three modalities)
│   ├── models/
│   │   ├── tabular.py             # TabularPipeline (StandardScaler + XGBoost)
│   │   ├── image.py               # ImageClassifier (ResNet-18 encoder + Linear head)
│   │   ├── spectral.py            # SpectralClassifier (1D-CNN encoder + Linear head)
│   │   └── fusion.py              # LateFusionModel (705-d concat → MLP → 3 classes)
│   ├── train/
│   │   ├── tabular.py             # XGBoost training + optional grid search
│   │   ├── image.py               # ResNet-18 training loop (configs A–D)
│   │   ├── spectral.py            # 1D-CNN training loop (configs A–D)
│   │   ├── fusion.py              # Two-phase late fusion training + ablation
│   │   └── physical.py            # Stretch task: physical (star-forming / passive)
│   ├── evaluate.py                # Unified evaluation across all four models
│   └── interpret.py               # XGBoost feature importance + GradCAM
│
├── checkpoints/                   # Saved weights and results (gitignored)
│   ├── tabular/                   # xgb_model_full.pkl, test_results*.pkl, grid_search_results.csv
│   ├── image/                     # resnet18_configC_best.pth, test_results*.pkl, gradcam_samples.png
│   ├── spectral/                  # cnn1d_best.pth, test_results*.pkl
│   ├── fusion/                    # late_fusion_best.pth, test_results*.pkl
│   ├── physical/                  # tabular_results.pkl, spectral_results.pkl
│   └── results_summary.csv        # Final four-model comparison table
│
├── notebooks/
│   ├── 00_run_pipeline.ipynb      # Run the full pipeline from a notebook
│   ├── 01_label_audit.ipynb       # Step 0: class counts, vote distributions, sentinel check
│   ├── 02_results.ipynb           # Final results figures for the report
│   ├── 03_walkthrough.ipynb       # Full presentation walkthrough (training, encoding, viz)
│   └── explore_tables.ipynb       # Initial data exploration (starter kit)
│
├── scripts/
│   ├── download_images.py         # Batch JPEG downloader (Legacy Survey)
│   └── download_spectra.py        # Batch FITS downloader (SDSS archive)
│
├── run_pipeline.py                # One-shot pipeline runner (all stages, resumable)
├── pyproject.toml                 # uv project config + pinned dependencies
└── uv.lock                        # Fully locked dependency graph
```

---

## Environment Setup

### Recommended: `uv` (locked, reproducible)

```bash
# Install uv if not already installed
pip install uv

# Install all dependencies from the lock file (Python 3.12)
uv sync

# Verify installation
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

> **GPU note:** `torch` and `torchvision` are installed from the PyTorch CUDA 12.8 index. If you are on CPU-only or a different CUDA version, edit `pyproject.toml` and `uv.lock` accordingly, or install PyTorch separately.

### Alternative: pip + venv

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install jupyter numpy pandas matplotlib astropy scikit-learn xgboost tqdm Pillow
```

---

## Data Acquisition

### Step 1 — Run the SQL query on SDSS CasJobs

1. Go to [https://casjobs.sdss.org/casjobs/](https://casjobs.sdss.org/casjobs/) and sign in.
2. Select **DR19** from the Context dropdown.
3. Open `input/queries/DATA7901_DR19_casjobs_merged.sql` and paste it into the query editor.
4. Submit the job. When finished, export `mydb.DATA7901_DR19_merged` as CSV.
5. Save the file to:

```text
input/tables/DATA7901_DR19_merged.csv
```

> **Why the merged query?** The wildcard query (`p.*, s.*, v.*`) silently renames duplicate columns (e.g., `SpecObj.z` → `Column8`). The merged query uses explicit aliases (`s.z AS spec_z`, `s.mjd AS spec_mjd`) so all column names are unambiguous. See `CLAUDE.md` for the full column collision audit.
>
> **Size note:** The full table is ~507k rows. CasJobs caps export at ~500 MB; if the export is split across multiple files, concatenate them:

```bash
{ head -1 MERGED1.csv; tail -n +2 MERGED1.csv; tail -n +2 MERGED2.csv; tail -n +2 MERGED3.csv; } > DATA7901_DR19_merged.csv
```

### Step 2 — Download galaxy images

```bash
uv run python scripts/download_images.py
# Images saved to input/images/<objid>.jpeg
# Failed downloads logged to input/tables/image_download_failures.txt
```

Stratified sampling (up to 10k per class) is applied by default. The script is resumable — already-downloaded files are skipped.

### Step 3 — Download spectra

```bash
uv run python scripts/download_spectra.py
# Spectra saved to input/spectra/spec-<plate>-<mjd>-<fiberid>.fits
# Failed downloads logged to input/tables/spectra_download_failures.txt
# Also writes input/tables/working_set.csv
```

---

## Running the Pipeline

The simplest way to run everything is the one-shot pipeline script. It is **resumable** — each stage checks whether its output already exists and skips it.

```bash
# Run all stages (skip completed ones)
uv run python run_pipeline.py

# Re-run everything from scratch
uv run python run_pipeline.py --force
```

### Pipeline stages (in order)

| Stage | Script | Output | Hardware |
| --- | --- | --- | --- |
| 1. Full-data split | `src.data.split` | `split_train_full.csv`, `split_val_full.csv` | CPU, <1 min |
| 2. Tabular baseline | `src.train.tabular --grid-search` | `checkpoints/tabular/` | CPU, ~15 min |
| 3. Image baseline (A–D) | `src.train.image` | `checkpoints/image/` | GPU, ~2–4 hr |
| 4. Spectral baseline (A–D) | `src.train.spectral` | `checkpoints/spectral/` | GPU, ~2 hr |
| 5. Late fusion | `src.train.fusion` | `checkpoints/fusion/` | GPU, ~1 hr |
| 6. Ablation study | `src.train.fusion --ablate` | `checkpoints/fusion/*_ablate_*.pkl` | GPU, ~3 hr |
| 7. Interpretability | `src.interpret` | Feature importance + GradCAM | CPU/GPU, ~5 min |
| 8. Physical classification | `src.train.physical` | `checkpoints/physical/` | CPU/GPU, ~10 min |
| 9. Evaluation summary | `src.evaluate` | `checkpoints/results_summary.csv` | CPU, <1 min |

---

## Training Individual Models

Each training script can also be run standalone.

### Tabular (XGBoost)

```bash
# Default: working-set split, no grid search
uv run python -m src.train.tabular

# Full-data split (all 214k labeled rows) + hyperparameter grid search
uv run python -m src.train.tabular --full-data --grid-search
```

Outputs: `checkpoints/tabular/xgb_model_full.pkl`, `test_results_full.pkl`, `grid_search_results.csv`

### Image (ResNet-18)

```bash
# Config A — baseline (lr=1e-4, no dropout, pretrained ImageNet)
uv run python -m src.train.image

# Config C — best (lr=1e-4, dropout=0.3, pretrained ImageNet)
uv run python -m src.train.image --dropout 0.3 --suffix configC

# Config B — high learning rate
uv run python -m src.train.image --lr 5e-5 --suffix configB

# Config D — random init (no ImageNet pretraining)
uv run python -m src.train.image --no-pretrain --dropout 0.3 --suffix configD
```

| Config | LR | Dropout | Pretrained | Macro F1 | Merger F1 |
| --- | --- | --- | --- | --- | --- |
| A | 1e-4 | 0.0 | Yes | 0.806 | 0.492 |
| B | 1e-3 | 0.0 | Yes | 0.762 | 0.389 |
| **C** | **1e-4** | **0.3** | **Yes** | **0.807** | **0.501** |
| D | 1e-4 | 0.0 | No | 0.775 | 0.413 |

Best checkpoint: `checkpoints/image/resnet18_configC_best.pth`

### Spectral (1D-CNN)

```bash
# Config A — baseline (lr=1e-4, 32 base filters, no dropout)
uv run python -m src.train.spectral

# Config C — wider filters
uv run python -m src.train.spectral --base-filters 64 --suffix configC

# Config D — with dropout
uv run python -m src.train.spectral --dropout 0.3 --suffix configD
```

| Config | LR | Filters | Dropout | Macro F1 | Merger F1 |
| --- | --- | --- | --- | --- | --- |
| **A** | **1e-4** | **32** | **0.0** | **0.563** | **0.118** |
| B | 1e-3 | 32 | 0.0 | 0.557 | 0.121 |
| C | 1e-4 | 64 | 0.0 | 0.543 | 0.082 |
| D | 1e-4 | 32 | 0.3 | 0.492 | 0.065 |

Best checkpoint: `checkpoints/spectral/cnn1d_best.pth`

### Late Fusion

```bash
# Train fusion with best encoders
uv run python -m src.train.fusion \
    --img-ckpt  checkpoints/image/resnet18_configC_best.pth \
    --spec-ckpt checkpoints/spectral/cnn1d_best.pth \
    --suffix    _bestenc

# Ablation: zero out one modality during training and test
uv run python -m src.train.fusion --ablate spectral --suffix _ablate_nospec
uv run python -m src.train.fusion --ablate image    --suffix _ablate_noimg
uv run python -m src.train.fusion --ablate tabular  --suffix _ablate_notab
```

| Fusion variant | Macro F1 | Merger F1 |
| --- | --- | --- |
| **Full (img + spec + tab)** | **0.873** | **0.679** |
| No spectral (img + tab) | 0.852 | 0.625 |
| No tabular (img + spec) | 0.817 | 0.525 |
| No image (spec + tab) | 0.572 | 0.061 |

### Physical Classification (stretch task)

```bash
uv run python -m src.train.physical
```

Classifies galaxies as **star-forming** (g−r < 0.65) vs **passive** (g−r ≥ 0.65). The spectral 1D-CNN achieves macro F1 = 0.935 on this task vs 0.563 on morphological classification — confirming that spectra encode physical type rather than visual morphology.

### Interpretability

```bash
uv run python -m src.interpret
```

Produces:

- `checkpoints/tabular/feature_importance.csv` — XGBoost gain-based feature importance
- `checkpoints/image/gradcam_samples.png` — GradCAM activation maps (2 galaxies per class)

---

## Notebooks

Open Jupyter Lab with:

```bash
uv run jupyter lab
```

| Notebook | Purpose | Run time |
| --- | --- | --- |
| `01_label_audit.ipynb` | Class counts, vote distributions, sentinel (-9999) check, working set summary | ~5 min (CPU) |
| `02_results.ipynb` | Final performance figures for the report (confusion matrices, per-class F1, ablation) | ~2 min (CPU) |
| `03_walkthrough.ipynb` | Complete presentation walkthrough: raw data → preprocessing → training demo → encoding → GradCAM → t-SNE → fusion → results | ~15 min (GPU recommended) |
| `explore_tables.ipynb` | Initial EDA: load CSV, validate Galaxy Zoo fields, download and visualise sample images and spectra | ~5 min (CPU) |

To execute a notebook non-interactively:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 notebooks/03_walkthrough.ipynb
```

---

## Results & Evaluation

After all checkpoints are produced, generate the final comparison table:

```bash
uv run python -m src.evaluate
```

Output: `checkpoints/results_summary.csv`

### Final model comparison (test set, 32,158 galaxies)

| Model | Macro F1 | Accuracy | Elliptical F1 | Spiral F1 | Merger F1 |
| --- | --- | --- | --- | --- | --- |
| Tabular (XGBoost) | 0.775 | 94.0% | 0.944 | 0.942 | 0.439 |
| Image (ResNet-18) | 0.807 | 95.7% | 0.967 | 0.959 | 0.501 |
| Spectral (1D-CNN) | 0.563 | 77.7% | 0.770 | 0.800 | 0.118 |
| **Fusion** | **0.873** | **96.5%** | **0.970** | **0.967** | **0.679** |

Key findings:

- **Image > Tabular** (+0.032 macro F1): spatial morphology carries information beyond aggregated photometric measurements.
- **Spectral underperforms** (0.563): spectra encode physical properties (star formation, velocity dispersion), not spatial shape — they are an indirect morphology proxy.
- **Fusion achieves the best performance** across all metrics. The largest gain is in merger F1 (+0.178 over image alone): mergers show both disturbed morphology *and* anomalous spectral signatures that are complementary signals.

---

## Project Architecture

### Label construction

Galaxy Zoo 1 vote fractions (from `zooVotes`) are thresholded at 0.60:

```python
# Assign label only where one morphology dominates
df.loc[df['p_el'] > 0.60, 'label'] = 'elliptical'
df.loc[df['p_cs'] > 0.60, 'label'] = 'spiral'   # p_cs = p_cw + p_acw + p_edge
df.loc[df['p_mg'] > 0.60, 'label'] = 'merger'
# Rows with no dominant class, p_dk >= 0.30, or nvote_tot < 20 are excluded
```

**Vote fractions are never used as input features** — they are the source of the labels and including them would produce trivially perfect accuracy (label leakage).

### Feature engineering

44 tabular features from `PhotoObj` + `SpecObj`. Key groups:

| Feature group | Examples | What it encodes |
| --- | --- | --- |
| Extinction-corrected colours | `dered_g - dered_r`, `dered_r - dered_i` | Stellar population age (red = elliptical, blue = spiral) |
| Profile weight | `fracDeV_r` | 1.0 = de Vaucouleurs bulge (elliptical), 0.0 = exponential disc (spiral) |
| Concentration | `petroR90_r / petroR50_r` | Ellipticals ~3.5–5; spirals ~2–3.5 |
| Shape | `deVAB_r`, `expAB_r`, `mE1_r`, `mE2_r` | Axis ratios; adaptive moment ellipticity (merger indicator) |
| Spectroscopic | `spec_z`, `velDisp`, `theta_0–4` | Redshift, velocity dispersion, eigenspectrum PCA |

Four `lnL*` profile log-likelihood columns were **dropped** after audit — 67–84% sentinel values (-9999) mean they carry no signal for the majority of galaxies.

### Fusion architecture

```text
Image (224×224 JPEG) → ResNet-18 encoder  → 512-d
Spectrum (2700 bins) → 1D-CNN encoder     → 128-d
Tabular (44 features) → MLP encoder       → 64-d
has_spectrum (0/1)                         → 1-d
                           concat          → 705-d
                           Linear(256) → ReLU → Dropout(0.3)
                           Linear(3)   → logits
```

**Two-phase training:**

1. **Phase 1** (10 epochs): freeze image and spectral encoders; cache their outputs in RAM; train only the fusion MLP head and tabular encoder (`batch_size=2048`).
2. **Phase 2** (until early stopping): unfreeze all parameters; joint fine-tuning at `lr=1e-5` with `CosineAnnealingLR`.

### Spectrum preprocessing

Each FITS spectrum is resampled to a uniform 3800–9200 Å grid (2700 bins at 2 Å/px), flux values are clipped at ±5σ, then normalised by median absolute flux. Missing spectra are zero-filled; a binary `has_spectrum` flag is passed to the fusion model.

---

## Troubleshooting

**`input/tables/DATA7901_DR19_merged.csv` not found**
The data files are gitignored. Follow the [Data Acquisition](#data-acquisition) steps to generate them.

**CUDA out of memory**
Reduce `BATCH_SIZE` in the training script, or use gradient accumulation. Image training default is 256; try 64 or 128.

**Zombie CUDA process (Windows)**
If a previous training run was killed abruptly, the CUDA context may still be held. Run `nvidia-smi` to find the PID and kill it:

```powershell
Stop-Process -Id <PID> -Force
```

**Merger class F1 near zero**
Mergers are ~0.7% of the labeled set. Check that class-weighted loss is being used (`CrossEntropyLoss(weight=class_weights)`). If merger count is < 500, consider lowering the merger threshold to 0.40 in `src/data/labels.py`.

**`KeyError: 'spec_z'` or `'Column8'`**
The raw merged CSV uses CasJobs column aliases. `src/data/rename.py` maps `Column8` → `spec_z`. Ensure `apply_renames(df)` is called after loading the CSV.

**Spectra 404 errors**
Not all `plate/mjd/fiberid` combinations exist at the SDSS archive. The downloader logs failures to `input/tables/spectra_download_failures.txt`. The fusion model handles missing spectra via zero-fill + `has_spectrum=0`.

**`uv sync` fails (torch CUDA version mismatch)**
Edit `pyproject.toml` to change the PyTorch index URL to match your CUDA version (e.g., `cu121`, `cpu`). Then re-run `uv sync`.

---

## Data Sources

- **SDSS DR19 CasJobs** — SQL queries for `PhotoObj`, `SpecObj`, `zooVotes`: [https://casjobs.sdss.org/casjobs/](https://casjobs.sdss.org/casjobs/)
- **DESI Legacy Survey** — galaxy image cutouts (JPEG): [https://www.legacysurvey.org/](https://www.legacysurvey.org/)
- **SDSS DR19 archive** — spectral FITS files: [https://data.sdss.org/sas/dr19/](https://data.sdss.org/sas/dr19/)
- **Galaxy Zoo 1** — Lintott, C. et al. (2011), MNRAS, 410, 166. [ADS](https://ui.adsabs.harvard.edu/abs/2011MNRAS.410..166L/abstract)
