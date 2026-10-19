"""
=============================================================================
  SMAP Irregularity Injector  â€”  Phase 2 of the SEE Detection Pipeline
  -----------------------------------------------------------------------
  IEEE RadarConf 2027  |  ISRO-realistic gap injection into NASA SMAP data

  WHAT THIS SCRIPT DOES (no data fabrication, no sensor value modification):
  --------------------------------------------------------------------------
  1. Loads all 55 SMAP channels from the downloaded dataset (test + train)
  2. Applies a proportionally faithful LEO burst-gap pattern derived from
     real ISRO telemetry statistics:
       â€¢ 1 large gap per ~16 rows  (6.25% gap rate)
       â€¢ Burst length: Uniform[10, 22] rows  (real variation around mean 16)
       â€¢ Gap size distribution: 52% â†’ 148 units, 47.9% â†’ 149 units
         (proportional, not exact â€” scaled to SMAP's observation units)
       â€¢ Occasional small within-burst gaps (2â€“5 units, 5% probability)
  3. ONLY removes rows â€” sensor values are NEVER modified or invented
  4. Computes exact delta_t (elapsed units since previous observation)
     for every surviving row â€” this becomes Layer 2's time-gap signal
  5. Preserves all ground-truth anomaly labels (Is_Anomaly) for surviving rows
  6. Saves:
       â€¢ Per-channel CSVs  â†’  nasa_smap_data/irregular/<chan_id>.csv
       â€¢ Combined parquet  â†’  nasa_smap_data/all_smap_irregular.parquet
       â€¢ Summary stats     â†’  nasa_smap_data/injection_summary.csv

  ISRO-DERIVED PROPORTIONS USED:
  --------------------------------
  Source: Real ISRO spacecraft telemetry (Febâ€“Jul 2018)
    Total rows  : 1,517,153
    Large gaps  : 94,823  â†’  rate = 6.25% (1 per 16 rows)
    148-unit gaps: 52.0% of all large gaps
    149-unit gaps: 47.9% of all large gaps
    Other         :  0.1%
  These proportions are preserved; exact magnitudes are scaled to SMAP.

=============================================================================
"""

import ast
import numpy as np
import pandas as pd
from pathlib import Path


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# CONFIGURATION
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
DATA_ROOT    = Path("nasa_smap_data")
TEST_DIR     = DATA_ROOT / "data" / "data" / "test"
TRAIN_DIR    = DATA_ROOT / "data" / "data" / "train"
META_CSV     = DATA_ROOT / "labeled_anomalies.csv"
OUT_DIR      = DATA_ROOT / "irregular"
PARQUET_OUT  = DATA_ROOT / "all_smap_irregular.parquet"
SUMMARY_OUT  = DATA_ROOT / "injection_summary.csv"

SPACECRAFT   = "SMAP"
START_TS     = pd.Timestamp("2026-01-01 00:00:00")

# â”€â”€ Gap model (derived proportionally from real ISRO telemetry) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# ISRO: 1 large gap per 16 rows, 52% are 148-unit gaps, 47.9% are 149-unit gaps
# We use proportional scaling â€” not hard-coded to ISRO's exact dataset size.

BURST_MIN    = 10     # minimum rows in one burst (real ISRO shortest runs ~10)
BURST_MAX    = 22     # maximum rows in one burst (spread around mean 16)

# Large gap sizes: proportional to ISRO's 148/149 sec modal gap
GAP_148_PROB = 0.521  # 52.1% of large gaps â†’ 148 observation-units skipped
GAP_149_PROB = 0.479  # 47.9% of large gaps â†’ 149 observation-units skipped

# Small within-burst gaps (communication micro-interruptions)
SMALL_GAP_PROB    = 0.05   # 5% chance each within-burst transition has a micro-gap
SMALL_GAP_MIN     = 2
SMALL_GAP_MAX     = 5

RANDOM_SEED  = 42     # reproducibility â€” different per channel via chan_seed


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 1: Load SMAP metadata
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def load_metadata(meta_csv: Path, spacecraft: str) -> pd.DataFrame:
    df = pd.read_csv(meta_csv)
    df_sc = df[df["spacecraft"] == spacecraft].reset_index(drop=True)
    print(f"[META]   {spacecraft} channels: {len(df_sc)}")
    return df_sc


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 2: Load one channel's telemetry + ground truth
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def load_channel(chan_id: str, anomaly_str: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (data, mask):
      data  : np.ndarray (n_rows, n_features)
      mask  : np.ndarray (n_rows,) int8  â€” 1 = anomaly, 0 = normal
    """
    npy_path = TEST_DIR / f"{chan_id}.npy"
    if not npy_path.exists():
        raise FileNotFoundError(f"Missing: {npy_path}")

    data = np.load(npy_path)                        # (n, 25)
    n    = data.shape[0]

    # Build anomaly mask from labeled sequences
    mask = np.zeros(n, dtype=np.int8)
    sequences = ast.literal_eval(anomaly_str)
    for s, e in sequences:
        mask[max(0, s) : min(n, e + 1)] = 1

    return data, mask


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 3: Generate a burst-gap keep-mask
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def generate_keep_mask(n_rows: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generates a boolean array of shape (n_rows,):
      True  â†’ row is "observed" (survives)
      False â†’ row falls inside a gap (never sent / never received)

    Pattern: [burst of 10â€“22 rows] â†’ [gap of 148 or 149 rows] â†’ repeat

    This models the real ISRO LEO burst-gap telemetry pattern proportionally:
      â€¢ Burst  mean = 16 rows  (Uniform[10,22] â†’ mean 16.0)
      â€¢ Gap    mean = 148.48 rows  (52.1% Ã— 148 + 47.9% Ã— 149)
      â€¢ Survival rate = 16 / (16 + 148.48) = 9.73%  â† matches ISRO proportion

    Small within-burst micro-gaps are also injected (5% probability).
    """
    keep = np.zeros(n_rows, dtype=bool)
    idx  = 0

    while idx < n_rows:
        # â”€â”€ Burst phase: observe rows â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        burst_len = int(rng.integers(BURST_MIN, BURST_MAX + 1))

        for i in range(burst_len):
            if idx >= n_rows:
                break
            keep[idx] = True
            idx += 1

            # Small within-burst micro-gap (rare)
            if i < burst_len - 1 and rng.random() < SMALL_GAP_PROB:
                skip = int(rng.integers(SMALL_GAP_MIN, SMALL_GAP_MAX + 1))
                idx += skip   # these rows are "dropped" (not received)

        # â”€â”€ Gap phase: skip rows â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if idx < n_rows:
            # Gap size: 52.1% â†’ 148, 47.9% â†’ 149 (proportional to ISRO)
            gap_len = 148 if rng.random() < GAP_148_PROB else 149
            idx += gap_len    # skip these rows entirely

    return keep


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 4: Build the irregular DataFrame for one channel
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def build_irregular_df(
    chan_id     : str,
    data        : np.ndarray,    # (n_rows, n_features)
    mask        : np.ndarray,    # (n_rows,) int8 anomaly labels
    keep        : np.ndarray,    # (n_rows,) bool keep-mask
    start_ts    : pd.Timestamp,
) -> pd.DataFrame:
    """
    Assembles the irregular telemetry DataFrame.

    For each surviving row:
      - original_row_idx  : integer index in the original .npy (for traceability)
      - timestamp         : datetime at original_row_idx seconds from start_ts
      - delta_t           : actual elapsed seconds since previous surviving row
                            (= 1 for consecutive rows, 148/149 across a gap)
      - feat_00 â€¦ feat_24 : real sensor values (UNCHANGED)
      - Is_Anomaly        : 0 or 1 from ground-truth label
    """
    n_features  = data.shape[1]
    feature_cols = [f"feat_{i:02d}" for i in range(n_features)]

    surviving_indices = np.where(keep)[0]          # row indices that survived

    # Compute original timestamps (treating each original row as 1 second)
    timestamps = start_ts + pd.to_timedelta(surviving_indices, unit="s")

    # Compute delta_t: elapsed seconds since previous surviving row
    delta_t = np.diff(surviving_indices, prepend=surviving_indices[0])
    # First row: delta_t = 0 (no predecessor)
    delta_t[0] = 0

    df = pd.DataFrame(
        data[surviving_indices],
        columns=feature_cols,
    )
    df.insert(0, "original_row_idx", surviving_indices)
    df.insert(1, "timestamp",        timestamps)
    df.insert(2, "delta_t",          delta_t.astype(np.float32))
    df["Is_Anomaly"] = mask[surviving_indices].astype(np.int8)
    df["chan_id"]    = chan_id
    df = df.set_index("timestamp")

    return df


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STEP 5: Summarise the injection for reporting
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def summarise(chan_id: str, original_n: int, df: pd.DataFrame) -> dict:
    survived   = len(df)
    large_gaps = (df["delta_t"] > 10).sum()
    gap_148    = (df["delta_t"] == 148).sum()
    gap_149    = (df["delta_t"] == 149).sum()
    anom_orig  = 0    # computed from mask; updated by caller
    anom_surv  = int(df["Is_Anomaly"].sum())

    return {
        "chan_id"          : chan_id,
        "original_rows"   : original_n,
        "survived_rows"   : survived,
        "survival_pct"    : round(survived / original_n * 100, 2),
        "large_gaps_gt10" : int(large_gaps),
        "gap_148_units"   : int(gap_148),
        "gap_149_units"   : int(gap_149),
        "anomaly_rows_survived": anom_surv,
    }


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# MAIN PIPELINE
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def main():
    print("\n" + "=" * 70)
    print("  ISRO-PROPORTIONAL IRREGULARITY INJECTOR")
    print("  IEEE RadarConf 2027  |  Phase 2: Data Preparation")
    print("=" * 70)
    print(f"\n  Burst length : Uniform[{BURST_MIN}, {BURST_MAX}]  (mean = {(BURST_MIN+BURST_MAX)/2:.0f} rows)")
    print(f"  Gap dist     : {GAP_148_PROB*100:.1f}% -> 148 units  |  "
          f"{GAP_149_PROB*100:.1f}% -> 149 units")
    print(f"  Expected survival rate : ~9.7%  (mirrors ISRO 6.25% gap rate)\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df_meta = load_metadata(META_CSV, SPACECRAFT)

    all_dfs  = []
    summaries = []

    for idx, row in df_meta.iterrows():
        chan_id     = row["chan_id"]
        anom_str    = row["anomaly_sequences"]

        # Skip channels whose .npy doesn't exist in test/
        npy_path = TEST_DIR / f"{chan_id}.npy"
        if not npy_path.exists():
            print(f"  [SKIP]  {chan_id:6s} â€” test .npy not found")
            continue

        # Load real data + ground-truth mask
        data, mask = load_channel(chan_id, anom_str)
        n_orig     = len(data)

        # Per-channel reproducible seed (different gap pattern per channel,
        # since in real life each sensor stream has independent dropouts)
        chan_seed = RANDOM_SEED + idx
        rng = np.random.default_rng(chan_seed)

        # Generate keep-mask (only removes rows, never changes values)
        keep = generate_keep_mask(n_orig, rng)

        # Safety: ensure at least 1 row survives
        if keep.sum() == 0:
            keep[0] = True

        # Build the irregular DataFrame
        df_irr = build_irregular_df(chan_id, data, mask, keep, START_TS)

        # Save per-channel CSV
        out_path = OUT_DIR / f"{chan_id}.csv"
        df_irr.to_csv(out_path)

        # Collect summary
        s = summarise(chan_id, n_orig, df_irr)
        s["anomaly_rows_original"] = int(mask.sum())
        summaries.append(s)
        all_dfs.append(df_irr)

        print(
            f"  {chan_id:6s}  orig={n_orig:5d}  survived={len(df_irr):4d} "
            f"({s['survival_pct']:5.1f}%)  "
            f"gapsâ‰¥148: {s['gap_148_units']+s['gap_149_units']:4d}  "
            f"anom_survived={s['anomaly_rows_survived']:4d}"
        )

    # â”€â”€ Combine all channels into one parquet â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print(f"\n[COMBINE]  Merging {len(all_dfs)} channels ...")
    df_all = pd.concat(all_dfs, axis=0)
    df_all = df_all.sort_values(["chan_id", "original_row_idx"])
    df_all.to_parquet(PARQUET_OUT)
    print(f"[OK]       Saved â†’ {PARQUET_OUT}")
    print(f"           Combined shape: {df_all.shape}")

    # â”€â”€ Save summary CSV â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    df_sum = pd.DataFrame(summaries)
    df_sum.to_csv(SUMMARY_OUT, index=False)
    print(f"[OK]       Summary â†’ {SUMMARY_OUT}")

    # â”€â”€ Final statistics â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\n" + "=" * 70)
    print("  INJECTION SUMMARY")
    print("=" * 70)
    print(f"  Channels processed     : {len(summaries)}")
    print(f"  Total original rows    : {df_sum['original_rows'].sum():,}")
    print(f"  Total survived rows    : {df_sum['survived_rows'].sum():,}")
    print(f"  Overall survival rate  : "
          f"{df_sum['survived_rows'].sum()/df_sum['original_rows'].sum()*100:.2f}%")
    print(f"  Total anomaly rows (orig)    : {df_sum['anomaly_rows_original'].sum():,}")
    print(f"  Total anomaly rows (survived): {df_sum['anomaly_rows_survived'].sum():,}")
    print(f"  Anomaly survival rate  : "
          f"{df_sum['anomaly_rows_survived'].sum()/df_sum['anomaly_rows_original'].sum()*100:.1f}%")

    print(f"\n  delta_t distribution in combined dataset:")
    dt_series = df_all["delta_t"]
    print(f"    delta_t == 1   (normal)  : {(dt_series==1).sum():,} rows")
    print(f"    delta_t 2â€“10  (micro-gap): {((dt_series>1)&(dt_series<=10)).sum():,} rows")
    print(f"    delta_t == 148           : {(dt_series==148).sum():,} rows")
    print(f"    delta_t == 149           : {(dt_series==149).sum():,} rows")
    print(f"    delta_t > 10  (large gap): {(dt_series>10).sum():,} rows")
    print(f"    delta_t max              : {int(dt_series.max())}")
    print("=" * 70)

    print("\n[DONE]  Data is ready for modelling.")
    print("        Next: layer1_isolation_forest.py")
    print("        Input file: nasa_smap_data/all_smap_irregular.parquet\n")

    return df_all, df_sum


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if __name__ == "__main__":
    df_irregular, df_summary = main()
