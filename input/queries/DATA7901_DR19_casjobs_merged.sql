-- DATA7901 SDSS DR19 — Merged query (explicit columns, aliased to avoid clashes)
-- Combines PhotoObj (photometric features), SpecObj (spectral metadata + redshift),
-- and zooVotes (morphology labels) into a single clean table.
--
-- Usage:
--   1. Go to https://casjobs.sdss.org/casjobs/ and sign in
--   2. Set Context to DR19
--   3. Paste this query and submit
--   4. Export mydb.DATA7901_DR19_merged as CSV
--   5. Save as input/tables/DATA7901_DR19_merged.csv
--
-- Estimated output: ~507k rows, ~75 columns, ~200-250 MB (within 500 MB quota)
--
-- WHY explicit columns and aliases (not p.*, s.*, v.*):
--   Wildcards cause CasJobs to silently rename duplicate column names to Column1, Column2, ...
--   Verified collisions from DATA7901_DR19_MERGED_10000.csv (wildcard run):
--     - SpecObj.z  (redshift)  renamed to Column8  — clashes with PhotoObj.z (z-band magnitude)
--     - SpecObj.mjd (spec MJD) renamed to Column2  — clashes with PhotoObj.mjd (imaging MJD)
--     - ra, dec, objid from SpecObj/zooVotes renamed to Column3-Column14
--   Aliases below (spec_z, spec_mjd) produce unambiguous column names in the output CSV.

SELECT DISTINCT

  -- === Identifiers ===
  p.objid,
  p.ra,
  p.dec,
  s.specObjID,
  s.plate,
  s.mjd    AS spec_mjd,   -- aliased: PhotoObj.mjd = imaging date; this is spectroscopic MJD
  s.fiberid,

  -- === Filtering columns (NOT ML features — row selection only) ===
  s.class,        -- 'GALAXY', 'STAR', 'QSO'; keep == 'GALAXY'
  s.sdssPrimary,  -- 1 = primary SDSS observation; keep == 1 to avoid duplicate objects in train/test
  p.clean,        -- 1 = clean photometry (no deblending issues, not near edge); keep == 1

  -- === Galaxy Zoo labels (NEVER input features — they ARE the classification target) ===
  v.nvote_tot,    -- total volunteer votes; reliability filter: keep >= 20
  v.p_el,         -- fraction voting elliptical
  v.p_cw,         -- fraction voting clockwise spiral
  v.p_acw,        -- fraction voting anticlockwise spiral
  v.p_edge,       -- fraction voting edge-on disk
  v.p_dk,         -- fraction voting don't know / artifact; exclude rows where >= 0.3
  v.p_mg,         -- fraction voting merger
  v.p_cs,         -- combined spiral = p_cw + p_acw + p_edge; use for spiral label threshold

  -- === Spectroscopic features (from SpecObj) ===
  -- CRITICAL: SpecObj.z is spectroscopic REDSHIFT, but clashes with PhotoObj.z (z-band mag).
  -- Without alias, CasJobs renames it to Column8. Aliased here as spec_z.
  s.z          AS spec_z,   -- spectroscopic redshift; ML feature and physical context
  s.zErr,                   -- redshift uncertainty
  s.velDisp,                -- stellar velocity dispersion km/s; higher = more massive = elliptical
  s.velDispErr,             -- velocity dispersion uncertainty
  s.velDispChi2,            -- chi^2 of velDisp fit; flag unreliable measurements (velDispChi2 > 10)
  s.subClass,               -- spectral subclass: STARFORMING, AGN, BROADLINE, null; categorical
  s.snMedian_r,             -- median S/N per pixel in r-band; data quality indicator
  s.snMedian,               -- overall median S/N per pixel
  s.wCoverage,              -- fraction of spectral wavelength range covered; data quality flag
  -- Eigenspectrum coefficients: PCA decomposition onto SDSS galaxy eigenvectors.
  -- theta_0 tracks early/late-type spectral shape (absorption vs emission dominated).
  -- Tabular spectral information without needing raw FITS files.
  s.theta_0,
  s.theta_1,
  s.theta_2,
  s.theta_3,
  s.theta_4,

  -- === Petrosian magnitudes (from PhotoObj) ===
  -- Robust total-flux measurement for extended sources; r-band used in WHERE filter.
  -- Sentinel value -9999.0 = failed measurement; treat as NaN in preprocessing.
  p.petroMag_u,
  p.petroMag_g,
  p.petroMag_r,
  p.petroMag_i,
  p.petroMag_z,
  p.petroMagErr_u,
  p.petroMagErr_g,
  p.petroMagErr_r,
  p.petroMagErr_i,
  p.petroMagErr_z,

  -- === Model magnitudes (from PhotoObj) ===
  -- Best-fit composite profile magnitude; better colour estimates for extended galaxies.
  -- Prefer over petroMag for colour indices (g-r, r-i, u-g).
  p.modelMag_u,
  p.modelMag_g,
  p.modelMag_r,
  p.modelMag_i,
  p.modelMag_z,
  p.modelMagErr_u,
  p.modelMagErr_g,
  p.modelMagErr_r,
  p.modelMagErr_i,
  p.modelMagErr_z,

  -- === Extinction-corrected magnitudes (from PhotoObj) ===
  -- dered_* = modelMag_* - extinction_* (Schlegel 1998 Galactic dust map).
  -- Best choice for colour measurements; corrects Milky Way dust reddening.
  p.dered_u,
  p.dered_g,
  p.dered_r,
  p.dered_i,
  p.dered_z,

  -- === Dust extinction per band (from PhotoObj) ===
  p.extinction_u,
  p.extinction_g,
  p.extinction_r,
  p.extinction_i,
  p.extinction_z,

  -- === Petrosian size measurements — r-band only (reference band) ===
  p.petroR50_r,     -- half-light radius in arcsec (50% of Petrosian flux)
  p.petroR50Err_r,
  p.petroR90_r,     -- 90%-light radius in arcsec
  p.petroR90Err_r,
  -- Derived: concentration = petroR90_r / petroR50_r
  --   ellipticals ~3.5-5.0  |  spirals ~2.0-3.5

  -- === Profile fit likelihoods — r and g bands (from PhotoObj) ===
  -- lnLDeV: log-likelihood of de Vaucouleurs r^(1/4) fit (bulge-dominated = elliptical)
  -- lnLExp: log-likelihood of exponential disc fit (disc-dominated = spiral)
  -- Derived: deV_exp_ratio = lnLDeV_r - lnLExp_r  (> 0 -> elliptical-like)
  p.lnLDeV_r,
  p.lnLExp_r,
  p.lnLDeV_g,
  p.lnLExp_g,

  -- === Composite model profile weight (from PhotoObj) ===
  -- fracDeV_r: weight of de Vaucouleurs component in the best-fit composite model.
  --   1.0 = pure de Vaucouleurs bulge (elliptical)
  --   0.0 = pure exponential disc (spiral)
  -- Single strongest photometric morphology indicator.
  p.fracDeV_r,
  p.fracDeV_g,

  -- === Profile shape parameters — r-band (from PhotoObj) ===
  p.deVAB_r,    -- de Vaucouleurs axis ratio b/a (1=round, 0=edge-on); elliptical shape/roundness
  p.expAB_r,    -- exponential disc axis ratio b/a; disc inclination angle
  p.deVRad_r,   -- de Vaucouleurs effective radius (arcsec)
  p.expRad_r,   -- exponential disc effective radius (arcsec)

  -- === Adaptive moment ellipticity — r-band (from PhotoObj) ===
  -- Gaussian-weighted 2nd moments; sensitive to asymmetric/disturbed morphology.
  -- Useful for mergers which have irregular shapes (elevated |mE1|, |mE2|).
  p.mE1_r,
  p.mE2_r

INTO mydb.DATA7901_DR19_merged

FROM PhotoObj AS p
  LEFT JOIN SpecObj AS s ON s.bestobjid = p.objid
  LEFT JOIN zooVotes AS v ON v.objid    = p.objid

WHERE
  p.petroMag_r BETWEEN 10.0 AND 17.7          -- bright enough to resolve, faint enough to be distant
  AND s.zWarning = 0                           -- clean spectroscopic fit
  AND s.z BETWEEN 0.003333 AND 0.1500000001;  -- nearby galaxies only (z < 0.15)
