# src/data/labels.py
#
# Assigns morphological labels from Galaxy Zoo vote fractions.
# Labels are threshold-based (not argmax) to discard ambiguous cases.
#
# Three classes:
#   elliptical — p_el > threshold
#   spiral     — p_cs > threshold  (p_cs = p_cw + p_acw + p_edge, combined spiral)
#   merger     — p_mg > threshold
#
# Rows matching none, or matching more than one, are excluded.
# p_cs is used instead of individual spiral subtypes (cw/acw/edge) because
# distinguishing spiral handedness has <70% inter-rater agreement in Galaxy Zoo.

import pandas as pd


LABEL_COLS = ["p_el", "p_cw", "p_acw", "p_edge", "p_mg", "p_cs"]

LABEL_MAP = {
    "elliptical": 0,
    "spiral":     1,
    "merger":     2,
}


def make_labels(
    df: pd.DataFrame,
    min_votes: int = 20,
    threshold: float = 0.60,
    max_dk: float = 0.30,
    merger_threshold: float | None = None,
    max_redshift: float = 0.10,
) -> pd.DataFrame:
    """
    Filter rows and assign morphological labels from Galaxy Zoo vote fractions.

    Exclusion criteria (applied in order):
      1. class != 'GALAXY'          — removes stars and QSOs
      2. sdssPrimary != 1           — removes duplicate observations (prevents train/test leakage)
      3. nvote_tot < min_votes      — too few votes, label unreliable
      4. p_dk >= max_dk             — high don't-know: artifact, bad image, or ambiguous
      5. spec_z > max_redshift      — distant galaxies are poorly resolved in images and
                                      have lower S/N spectra; morphology classification
                                      degrades significantly beyond z~0.10. Galaxy Zoo 1
                                      papers (Lintott 2011) use z < 0.085 for their clean
                                      sample. Default 0.10 is a conservative version of this.

    Label assignment (only one label per row; rows matching none are excluded):
      p_el > threshold           -> 'elliptical'
      p_cs > threshold           -> 'spiral'
      p_mg > merger_threshold    -> 'merger'  (defaults to threshold)

    Parameters
    ----------
    df : DataFrame with Galaxy Zoo columns present
    min_votes : minimum total vote count
    threshold : vote fraction required for elliptical and spiral assignment
    max_dk : maximum allowed don't-know fraction
    merger_threshold : separate threshold for mergers (default: same as threshold).
        Set lower (e.g. 0.40) if merger count is too small (<500 examples).
    max_redshift : upper redshift limit. Galaxies beyond this are excluded because
        (a) angular size is too small for reliable image-based morphology (<2 arcsec
        at z>0.15 for a 10 kpc galaxy), and (b) key spectral lines shift out of the
        SDSS detector window. Set to None to disable this filter.

    Returns
    -------
    DataFrame with a 'label' column (str) and 'label_id' column (int).
    All Galaxy Zoo vote columns are retained for audit purposes but must
    never be used as ML input features.
    """
    if merger_threshold is None:
        merger_threshold = threshold

    mask = (
        (df["class"] == "GALAXY") &
        (df["sdssPrimary"] == 1) &
        (df["nvote_tot"] >= min_votes) &
        (df["p_dk"] < max_dk)
    )
    if max_redshift is not None and "spec_z" in df.columns:
        mask &= (df["spec_z"] <= max_redshift)

    df = df[mask].copy()

    df["label"] = None
    df.loc[df["p_el"] > threshold,          "label"] = "elliptical"
    df.loc[df["p_cs"] > threshold,          "label"] = "spiral"
    df.loc[df["p_mg"] > merger_threshold,   "label"] = "merger"

    # Rows where more than one threshold is exceeded: exclude (genuinely ambiguous)
    multi = (
        (df["p_el"] > threshold).astype(int) +
        (df["p_cs"] > threshold).astype(int) +
        (df["p_mg"] > merger_threshold).astype(int)
    ) > 1
    df.loc[multi, "label"] = None

    labeled = df.dropna(subset=["label"]).copy()
    labeled["label_id"] = labeled["label"].map(LABEL_MAP)

    counts = labeled["label"].value_counts()
    z_note = f"  (spec_z <= {max_redshift})" if max_redshift is not None else ""
    print(f"Labeled galaxies: {len(labeled):,} / {len(df):,} filtered rows{z_note}")
    for cls in ["elliptical", "spiral", "merger"]:
        n = counts.get(cls, 0)
        pct = 100 * n / len(labeled) if len(labeled) > 0 else 0
        print(f"  {cls:12s}: {n:6,}  ({pct:.1f}%)")

    if counts.get("merger", 0) < 500:
        print(
            f"\nWARNING: only {counts.get('merger', 0)} merger examples. "
            "Consider merger_threshold=0.40 or dropping the merger class."
        )

    return labeled
