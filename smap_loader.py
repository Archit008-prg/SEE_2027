"""
=============================================================================
  NASA SMAP Telemetry Anomaly Loader
  ------------------------------------
  Target: IEEE RadarConf 2027  |  Layer-1 / Layer-2 anomaly detection pipeline
  Dataset: patrickfleith/nasa-anomaly-detection-dataset-smap-msl (Kaggle)
  Author: Lead AI Researcher + Python Data Engineer

  What this script does (step-by-step):
    1. Downloads the dataset from Kaggle via its official CLI API
    2. Extracts it into ./nasa_smap_data/
    3. Reads labeled_anomalies.csv, filters to SMAP spacecraft only
    4. Picks the FIRST SMAP channel (e.g. 'P-1') from the CSV
    5. Loads the corresponding raw .npy telemetry file from test/
    6. Parses the 'anomaly_sequences' column -> integer index ranges
    7. Builds a clean pandas DataFrame:
         Timestamp      -> continuous datetime from 2026-01-01 @ 1-second cadence
         Raw_Telemetry  -> actual sensor float values from the .npy file
         Is_Anomaly     -> 1 inside every anomaly window, 0 everywhere else
    8. Prints shape + a 10-row slice that straddles the first anomaly boundary
=============================================================================
"""

import subprocess
import sys
import zipfile
import os
import ast
import numpy as np
import pandas as pd
from pathlib import Path


# -----------------------------------------------------------------------------
# 0.  CONFIGURATION
# -----------------------------------------------------------------------------
KAGGLE_DATASET   = "patrickfleith/nasa-anomaly-detection-dataset-smap-msl"
DOWNLOAD_DIR     = Path("nasa_smap_data")          # everything lives here
ZIP_NAME         = "nasa-anomaly-detection-dataset-smap-msl.zip"
METADATA_CSV     = "labeled_anomalies.csv"
SPACECRAFT       = "SMAP"
START_TIMESTAMP  = "2026-01-01 00:00:00"
FREQ             = "1s"                            # 1-second cadence


# -----------------------------------------------------------------------------
# 1.  DATA FETCHING  --  Download & extract via Kaggle API
# -----------------------------------------------------------------------------
def download_dataset(download_dir: Path, dataset: str, zip_name: str) -> None:
    """
    Downloads the Kaggle dataset (if not already present) and extracts it.

    Prerequisite: ~/.kaggle/kaggle.json  must exist with your API credentials.
    Install Kaggle CLI:  pip install kaggle
    """
    download_dir.mkdir(parents=True, exist_ok=True)
    zip_path = download_dir / zip_name

    if zip_path.exists():
        print(f"[CACHE HIT]  ZIP already present -> {zip_path}")
    else:
        print(f"[DOWNLOAD]   Fetching dataset '{dataset}' from Kaggle ...")
        cmd = [
            sys.executable, "-m", "kaggle",
            "datasets", "download",
            "--dataset", dataset,
            "--path", str(download_dir),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"Kaggle download failed.\n"
                f"STDOUT: {result.stdout}\n"
                f"STDERR: {result.stderr}\n\n"
                "Make sure you have:\n"
                "  1. pip install kaggle\n"
                "  2. ~/.kaggle/kaggle.json  (API key from kaggle.com/account)"
            )
        print(f"[OK]         Download complete -> {zip_path}")

    # Extract only if the test/ folder is missing
    test_dir = download_dir / "test"
    if test_dir.exists():
        print(f"[CACHE HIT]  Already extracted -> {test_dir}")
    else:
        print(f"[EXTRACT]    Unzipping ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(download_dir)
        print(f"[OK]         Extracted to -> {download_dir}")


# -----------------------------------------------------------------------------
# 2.  METADATA PARSING  --  Filter labeled_anomalies.csv for SMAP only
# -----------------------------------------------------------------------------
def load_smap_metadata(download_dir: Path, metadata_csv: str, spacecraft: str) -> pd.DataFrame:
    """
    Reads the labeled anomaly CSV and returns rows for the given spacecraft.
    The CSV has at minimum:
        chan_id, spacecraft, anomaly_sequences, num_values, ...
    """
    # The CSV might be at the top-level of the ZIP or inside a sub-folder.
    # Search recursively so we are robust to both layouts.
    candidates = list(download_dir.rglob(metadata_csv))
    if not candidates:
        raise FileNotFoundError(
            f"Could not find '{metadata_csv}' anywhere under {download_dir}. "
            "Make sure the dataset extracted correctly."
        )
    csv_path = candidates[0]
    print(f"[METADATA]   Loading -> {csv_path}")

    df_meta = pd.read_csv(csv_path)
    print(f"             Full dataset: {len(df_meta)} channels, "
          f"spacecraft types: {df_meta['spacecraft'].unique().tolist()}")

    df_smap = df_meta[df_meta["spacecraft"] == spacecraft].reset_index(drop=True)
    print(f"             {spacecraft} channels found: {len(df_smap)}")
    print(f"             Channel list: {df_smap['chan_id'].tolist()}")
    return df_smap


# -----------------------------------------------------------------------------
# 3.  TELEMETRY EXTRACTION  --  Load the raw .npy file for the first channel
# -----------------------------------------------------------------------------
def load_channel_telemetry(download_dir: Path, channel_id: str) -> np.ndarray:
    """
    The dataset stores each channel's telemetry as <channel_id>.npy inside test/.
    We prefer the file inside test/ (multi-variate raw telemetry) over files
    in smoothed_errors/ or train/ which may be 1-D.
    Returns the array as-is (1-D or 2-D); build_dataframe handles both shapes.
    """
    all_candidates = list(download_dir.rglob(f"{channel_id}.npy"))
    if not all_candidates:
        raise FileNotFoundError(
            f"Could not find '{channel_id}.npy' anywhere under {download_dir}. "
            "Check that the dataset extracted correctly and the channel ID matches."
        )

    # Prefer files whose parent folder is named 'test'
    test_candidates = [p for p in all_candidates if p.parent.name == "test"]
    npy_path = test_candidates[0] if test_candidates else all_candidates[0]

    print(f"\n[TELEMETRY]  Channel: {channel_id}")
    print(f"             Loading  -> {npy_path}")
    print(f"             (all matches: {[str(p) for p in all_candidates]})")

    data = np.load(npy_path)
    print(f"             Raw .npy shape: {data.shape}  "
          f"({'1-D time-series' if data.ndim == 1 else f'{data.shape[1]}-feature multivariate'})")
    return data


# -----------------------------------------------------------------------------
# 4.  GROUND TRUTH MAPPING  --  Parse anomaly_sequences -> index ranges
# -----------------------------------------------------------------------------
def parse_anomaly_sequences(raw_string: str) -> list:
    """
    Converts the string "[[2149, 2349], [4000, 4200]]"
    into a Python list of (start, end) integer tuples.

    Uses ast.literal_eval -- safe, no eval(), no regex hacks.
    """
    parsed = ast.literal_eval(raw_string)   # -> [[2149, 2349], ...]
    return [(int(seq[0]), int(seq[1])) for seq in parsed]


def build_anomaly_mask(n_rows: int, sequences: list) -> np.ndarray:
    """
    Creates an integer array of shape (n_rows,) that is:
      * 1  inside every [start, end] range (inclusive on both ends)
      * 0  everywhere else
    """
    mask = np.zeros(n_rows, dtype=np.int8)
    for start, end in sequences:
        # Clamp to valid array bounds -- defensive programming
        s = max(0, start)
        e = min(n_rows - 1, end)
        mask[s : e + 1] = 1
    return mask


# -----------------------------------------------------------------------------
# 5.  DATAFRAME CONSTRUCTION
# -----------------------------------------------------------------------------
def build_dataframe(
    telemetry: np.ndarray,
    anomaly_mask: np.ndarray,
    start_ts: str,
    freq: str,
) -> pd.DataFrame:
    """
    Assembles the unified DataFrame.

    Columns:
        Timestamp      -- DatetimeIndex (start_ts, freq, len = n_rows)
        Raw_Telemetry  -- first feature column from the .npy array
        Is_Anomaly     -- 0 / 1 integer flag from the ground-truth mask
    """
    n_rows = len(telemetry)

    timestamps = pd.date_range(start=start_ts, periods=n_rows, freq=freq)

    # Handle both shapes:
    #   1-D (n_timesteps,)         -> already the primary sensor signal
    #   2-D (n_timesteps, n_feats) -> column 0 is the target channel
    if telemetry.ndim == 1:
        primary_sensor = telemetry.astype(np.float64)
    else:
        primary_sensor = telemetry[:, 0].astype(np.float64)

    df = pd.DataFrame({
        "Timestamp"     : timestamps,
        "Raw_Telemetry" : primary_sensor,
        "Is_Anomaly"    : anomaly_mask.astype(np.int8),
    })
    df = df.set_index("Timestamp")
    return df


# -----------------------------------------------------------------------------
# 6.  PREVIEW  --  Shape + anomaly-transition slice
# -----------------------------------------------------------------------------
def preview_dataframe(df: pd.DataFrame, sequences: list) -> None:
    """
    Prints:
      * Overall shape
      * Basic statistics on Raw_Telemetry
      * Anomaly label distribution
      * A 10-row window straddling the first anomaly transition (0->1)
    """
    print("\n" + "=" * 70)
    print("  DATAFRAME OVERVIEW")
    print("=" * 70)
    print(f"  Shape            : {df.shape}   (rows x cols)")
    print(f"  Index start      : {df.index[0]}")
    print(f"  Index end        : {df.index[-1]}")
    print(f"  Anomaly rows     : {df['Is_Anomaly'].sum():,}  "
          f"({df['Is_Anomaly'].mean() * 100:.2f}% of total)")
    print(f"  Normal rows      : {(df['Is_Anomaly'] == 0).sum():,}")
    print()
    print("  Raw_Telemetry statistics:")
    print(df["Raw_Telemetry"].describe().to_string(header=False))

    # -- Find the first anomaly window boundary --------------------------------
    if not sequences:
        print("\n  [NOTE] No anomaly sequences found for this channel.")
        return

    first_start, first_end = sequences[0]
    print(f"\n  First anomaly window : rows [{first_start} -> {first_end}]")

    # Show 5 rows BEFORE the anomaly starts + 5 rows INSIDE it
    window_start = max(0, first_start - 5)
    window_end   = min(len(df) - 1, first_start + 4)

    print("\n" + "=" * 70)
    print(f"  10-ROW ANOMALY TRANSITION SLICE  "
          f"(rows {window_start} -> {window_end})")
    print("=" * 70)

    slice_df = df.iloc[window_start : window_end + 1].copy()
    # Add a plain integer row index column for readability
    slice_df.insert(0, "Row_Index", range(window_start, window_end + 1))

    with pd.option_context(
        "display.max_columns", None,
        "display.width", 120,
        "display.float_format", "{:.6f}".format,
    ):
        print(slice_df.to_string(index=True))

    print(f"\n  ^ Is_Anomaly flips 0->1 at row index {first_start}")
    print("=" * 70)


# -----------------------------------------------------------------------------
# MAIN PIPELINE
# -----------------------------------------------------------------------------
def main() -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("  NASA SMAP TELEMETRY ANOMALY LOADER")
    print("  IEEE RadarConf 2027  |  Layer-1 Isolation Forest pipeline")
    print("=" * 70 + "\n")

    # Step 1: Fetch
    download_dataset(DOWNLOAD_DIR, KAGGLE_DATASET, ZIP_NAME)

    # Step 2: Metadata
    df_smap = load_smap_metadata(DOWNLOAD_DIR, METADATA_CSV, SPACECRAFT)

    # Step 3: Pick first channel & load telemetry
    first_channel_row = df_smap.iloc[0]
    channel_id        = first_channel_row["chan_id"]          # e.g. 'P-1'
    raw_anomaly_str   = first_channel_row["anomaly_sequences"]

    telemetry = load_channel_telemetry(DOWNLOAD_DIR, channel_id)
    n_rows    = telemetry.shape[0]

    # Step 4: Ground truth mapping
    print(f"\n[LABELS]     Raw anomaly_sequences string:")
    print(f"             {raw_anomaly_str}")
    sequences    = parse_anomaly_sequences(raw_anomaly_str)
    anomaly_mask = build_anomaly_mask(n_rows, sequences)
    print(f"             Parsed windows  : {sequences}")
    print(f"             Anomalous rows  : {anomaly_mask.sum():,} / {n_rows:,}")

    # Step 5: Build DataFrame
    print("\n[BUILD]      Assembling unified DataFrame ...")
    df = build_dataframe(telemetry, anomaly_mask, START_TIMESTAMP, FREQ)
    print(f"[OK]         DataFrame ready: shape = {df.shape}")

    # Step 6: Preview
    preview_dataframe(df, sequences)

    return df   # expose for downstream Layer-1 / Layer-2 code


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    df_telemetry = main()
    print(f"\n[DONE]  'df_telemetry' is ready -- shape {df_telemetry.shape}")
    print("        Next step -> feed into Isolation Forest (Layer 1)\n")
