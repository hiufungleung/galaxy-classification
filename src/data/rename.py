# src/data/rename.py
#
# CasJobs wildcard queries silently rename duplicate column names across joined tables.
# This module defines the single source-of-truth rename map and a helper to apply it.
#
# Confirmed collisions (from 10k wildcard test run):
#   PhotoObj.z   (z-band model mag, ~16-17)  vs  SpecObj.z   (redshift, ~0.02-0.15) -> Column8
#   PhotoObj.mjd (imaging epoch)              vs  SpecObj.mjd (spec epoch)           -> Column2
#   SpecObj.objid / ra / dec / cx / cy / cz  -> Column1, Column3-7
#   zooVotes.objid / ra / dec / ...          -> Column9-14

RENAME = {
    # Critical: used as ML features and download identifiers
    "Column8":  "spec_z",    # SpecObj.z — spectroscopic redshift
    "Column2":  "spec_mjd",  # SpecObj.mjd — used in spectrum file path

    # Case normalisation
    "objID":    "objid",
    "fiberID":  "fiberid",

    # Remaining collision columns — renamed for traceability, dropped before training
    "Column1":  "spec_objid_dup",
    "Column3":  "spec_ra_dup",
    "Column4":  "spec_dec_dup",
    "Column5":  "spec_cx_dup",
    "Column6":  "spec_cy_dup",
    "Column7":  "spec_cz_dup",
    "Column9":  "zoo_objid_dup",
    "Column10": "zoo_ra_dup",
    "Column11": "zoo_dec_dup",
    "Column12": "zoo_col12_dup",
    "Column13": "zoo_col13_dup",
    "Column14": "zoo_col14_dup",
}


def apply_renames(df):
    """Rename collision columns. Only renames columns that exist."""
    existing = {k: v for k, v in RENAME.items() if k in df.columns}
    return df.rename(columns=existing)
