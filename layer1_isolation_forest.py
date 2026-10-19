"""
=============================================================================
  Layer 1 — High-Sensitivity Physics & Contextual Isolation Forest (v3)
  ---------------------------------------------------------------------
  IEEE RadarConf 2027 | Two-Tier Asynchronous Telemetry Triage System

  Architecture
  ------------
  Communication gaps (delta_t > 1) are treated as hard burst boundaries.
  The Isolation Forest monitors rate-of-change, baseline drift, and command-
  physics consistency *within* each continuous burst. There is no interpolation
  across gaps — preserving raw data integrity for Layer 2 (PhysDiff).

  Pipeline Highlights (v3)
  ------------------------
  1. 8-Feature Vector:
     - f0: Raw telemetry x_t (feat_00)
     - f1: Baseline z-score |x_t - mu_0| / sigma_0 (calibrated on nominal train/)
     - f2: |dx| (within-burst first difference with boundary backfill)
     - f3: |x_t - roll_mean_5| (within-burst 5-step rolling mean deviation)
     - f4: roll_std_5 (within-burst 5-step rolling std)
     - f5: roll_range_10 (within-burst 10-step rolling range max - min)
     - f6: cmd_mismatch_err (absolute residual from Ridge predicting x_t
           from [x_{t-1}, roll_mean_5, feat_01..feat_24])
     - f7: aux_novelty (max normalized deviation across commands feat_01..24)
  2. Per-Channel IF with Train-Derived Tail Calibration:
     - contamination='auto' on 100% nominal train/ data.
     - Dual thresholding: tau_high (98.5th percentile) and tau_low (92.0th percentile).
  3. Burst-Aware Hysteresis & Denoising:
     - 3-step median smoothing with SEE spike preservation (> tau_high + 2.0).
     - Hysteresis activation + 1-to-2 step gap bridging.
     - Suppression of isolated borderline 1-step noise.
     - 30% channel safety cap.
  4. Layer 2 Candidate Windows & Metrics:
     - W = 32-step trigger windows extracted per continuous burst.
     - Step-Level, Point-Adjusted (PA), and Event-Level Recall metrics.

  Outputs
  -------
    nasa_smap_data/layer1_results.parquet   -- all test rows with IF scores & flags
    nasa_smap_data/layer1_windows.parquet   -- candidate trigger windows for Layer 2
    nasa_smap_data/layer1_summary.csv       -- per-channel performance summary
=============================================================================
"""

import ast
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_ROOT     = Path("nasa_smap_data")

if (DATA_ROOT / "train").exists():
    TRAIN_DIR = DATA_ROOT / "train"
elif (DATA_ROOT / "data" / "data" / "train").exists():
    TRAIN_DIR = DATA_ROOT / "data" / "data" / "train"
else:
    TRAIN_DIR = Path("train")

META_CSV      = DATA_ROOT / "labeled_anomalies.csv"
IRREGULAR_PKL = DATA_ROOT / "all_smap_irregular.parquet"

OUT_RESULTS   = DATA_ROOT / "layer1_results.parquet"
OUT_WINDOWS   = DATA_ROOT / "layer1_windows.parquet"
OUT_SUMMARY   = DATA_ROOT / "layer1_summary.csv"

SPACECRAFT    = "SMAP"
PRIMARY_FEAT  = "feat_00"
CMD_COLS      = [f"feat_{i:02d}" for i in range(1, 25)]

# Isolation Forest hyperparameters
N_ESTIMATORS  = 150
MAX_SAMPLES   = 256
RANDOM_SEED   = 42

# Burst segmentation and trigger window
GAP_THRESHOLD = 1
WINDOW_SIZE   = 32     # candidate window rows routed to Layer 2

FEATURE_NAMES = [
    "x_t",
    "z_level",
    "abs_dx",
    "roll_mean_dev",
    "roll_std_5",
    "roll_range_10",
    "cmd_mismatch_err",
    "aux_novelty",
]


# =============================================================================
# 1. HIGH-SIGNAL PHYSICS & CONTEXTUAL FEATURE EXTRACTION
# =============================================================================
def extract_channel_features(train_arr: np.ndarray, test_df: pd.DataFrame) -> tuple:
    """
    Extract the 8-feature matrix F for both train and test data.
    - train/ is a single continuous burst (delta_t = 1).
    - test is partitioned into bursts via (delta_t > 1).cumsum().
    - Dynamic features (dx, rolling stats, cmd_flips) are backfilled at burst starts.
    - Fits Ridge(alpha=1.0) on train to predict x_t from past state & commands.
    """
    x_tr = train_arr[:, 0].astype(np.float32)
    n_tr = len(x_tr)
    cmd_tr = (train_arr[:, 1:25].astype(np.float32)
              if train_arr.shape[1] > 1 else np.zeros((n_tr, 24), dtype=np.float32))

    # Baseline statistics on nominal train
    mu_0 = float(np.mean(x_tr))
    sigma_0 = float(max(np.std(x_tr), 1e-3))

    # Fit fast Ridge physics predictor on train
    s_xtr = pd.Series(x_tr)
    x_prev_tr = s_xtr.shift(1).bfill().fillna(x_tr[0]).to_numpy(dtype=np.float32)
    rm5_tr = s_xtr.rolling(5, min_periods=1).mean().to_numpy(dtype=np.float32)
    X_ridge_tr = np.column_stack([x_prev_tr, rm5_tr, cmd_tr])
    ridge = Ridge(alpha=1.0).fit(X_ridge_tr, x_tr)

    # Auxiliary command baseline statistics on train
    mu_aux = np.mean(cmd_tr, axis=0)
    sigma_aux = np.maximum(np.std(cmd_tr, axis=0), 1e-2)

    # Build F_train (8 features)
    f0_tr = x_tr
    f1_tr = (np.abs(x_tr - mu_0) / sigma_0).astype(np.float32)
    dx_tr = s_xtr.diff().bfill().fillna(0.0).to_numpy(dtype=np.float32)
    f2_tr = np.abs(dx_tr)
    f3_tr = np.abs(x_tr - rm5_tr).astype(np.float32)
    f4_tr = s_xtr.rolling(5, min_periods=2).std().bfill().fillna(0.0).to_numpy(dtype=np.float32)
    rmax_tr = s_xtr.rolling(10, min_periods=2).max()
    rmin_tr = s_xtr.rolling(10, min_periods=2).min()
    f5_tr = (rmax_tr - rmin_tr).bfill().fillna(0.0).to_numpy(dtype=np.float32)
    f6_tr = np.abs(x_tr - ridge.predict(X_ridge_tr)).astype(np.float32)
    f7_tr = np.max(np.abs(cmd_tr - mu_aux) / sigma_aux, axis=1).astype(np.float32)

    F_train = np.column_stack([f0_tr, f1_tr, f2_tr, f3_tr, f4_tr, f5_tr, f6_tr, f7_tr])

    # Partition test data into continuous bursts
    test_df_clean = test_df.copy().reset_index(drop=True)
    test_df_clean["_burst_id"] = (test_df_clean["delta_t"] > GAP_THRESHOLD).cumsum()

    bursts = []
    burst_F_list = []

    for _, b_df in test_df_clean.groupby("_burst_id"):
        b_df = b_df.reset_index(drop=True)
        bursts.append(b_df)
        x_te = b_df[PRIMARY_FEAT].to_numpy(dtype=np.float32)
        n_te = len(x_te)
        avail_cmds = [c for c in CMD_COLS if c in b_df.columns]
        cmd_te = (b_df[avail_cmds].to_numpy(dtype=np.float32)
                  if len(avail_cmds) > 0 else np.zeros((n_te, 24), dtype=np.float32))

        s_xte = pd.Series(x_te)
        f0_te = x_te
        f1_te = (np.abs(x_te - mu_0) / sigma_0).astype(np.float32)
        dx_te = s_xte.diff().bfill().fillna(0.0).to_numpy(dtype=np.float32)
        f2_te = np.abs(dx_te)

        x_prev_te = s_xte.shift(1).bfill().fillna(x_te[0]).to_numpy(dtype=np.float32)
        rm5_te = s_xte.rolling(5, min_periods=1).mean().to_numpy(dtype=np.float32)
        f3_te = np.abs(x_te - rm5_te).astype(np.float32)
        f4_te = s_xte.rolling(5, min_periods=2).std().bfill().fillna(0.0).to_numpy(dtype=np.float32)
        rmax_te = s_xte.rolling(10, min_periods=2).max()
        rmin_te = s_xte.rolling(10, min_periods=2).min()
        f5_te = (rmax_te - rmin_te).bfill().fillna(0.0).to_numpy(dtype=np.float32)

        X_ridge_te = np.column_stack([x_prev_te, rm5_te, cmd_te])
        f6_te = np.abs(x_te - ridge.predict(X_ridge_te)).astype(np.float32)
        f7_te = np.max(np.abs(cmd_te - mu_aux) / sigma_aux, axis=1).astype(np.float32)

        F_b = np.column_stack([f0_te, f1_te, f2_te, f3_te, f4_te, f5_te, f6_te, f7_te])
        burst_F_list.append(F_b)

    F_test = np.vstack(burst_F_list) if burst_F_list else np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
    return F_train, F_test, bursts, burst_F_list


# =============================================================================
# 2. SCORING, HYSTERESIS, & WINDOW TRIGGERING
# =============================================================================
def score_and_filter_channel(clf: IsolationForest,
                             scaler: StandardScaler,
                             F_train: np.ndarray,
                             F_test: np.ndarray,
                             bursts: list,
                             chan_id: str) -> tuple:
    """
    Compute calibrated anomaly scores and apply burst-aware hysteresis.
    Extracts W=32 candidate trigger windows.
    """
    F_train_scaled = np.clip(scaler.transform(F_train), -10, 10)
    F_test_scaled  = np.clip(scaler.transform(F_test), -10, 10)

    # Continuous raw scores (higher = more anomalous)
    raw_train = -clf.score_samples(F_train_scaled)

    t0_score = time.perf_counter()
    raw_test  = -clf.score_samples(F_test_scaled)
    t_score   = time.perf_counter() - t0_score
    lat_us    = (t_score / len(F_test) * 1e6) if len(F_test) > 0 else 0.0

    # Direct empirical tail score
    mean_tr = np.mean(F_train, axis=0)
    std_tr  = np.std(F_train, axis=0) + 1e-3
    z_emp   = np.max((F_test - mean_tr) / std_tr, axis=1)

    # Standardize against nominal training distribution
    mu_rt    = np.mean(raw_train)
    sigma_rt = np.std(raw_train) + 1e-6
    s_train  = (raw_train - mu_rt) / sigma_rt
    s_test   = (raw_test - mu_rt) / sigma_rt + 0.35 * np.clip(z_emp - 3.0, 0, 15)

    # Thresholds derived strictly from healthy train/ distribution
    tau_high = max(float(np.percentile(s_train, 98.5)), 2.2)
    tau_low  = max(float(np.percentile(s_train, 92.0)), 1.3)

    # Burst-aware hysteresis & denoising
    burst_flags = []
    offset = 0

    for b_df in bursts:
        b_len = len(b_df)
        s_b = s_test[offset : offset + b_len]
        offset += b_len

        if b_len == 1:
            val = s_b[0]
            if val > tau_high + 2.0:
                fl = np.array([1], dtype=np.int8)
            elif val >= tau_high:
                fl = np.array([1], dtype=np.int8)
            else:
                fl = np.array([0], dtype=np.int8)
            burst_flags.append(fl)
            continue

        s_series = pd.Series(s_b)
        s_smooth = s_series.rolling(3, min_periods=1, center=True).median().to_numpy()
        spike_mask = s_b > (tau_high + 2.0)
        s_smooth[spike_mask] = s_b[spike_mask]

        has_high    = np.any(s_b >= tau_high)
        has_two_low = np.sum(s_b >= (tau_low + 0.4)) >= 2

        if has_high or has_two_low:
            fl = (s_smooth > tau_low).astype(np.int8)
            fl[s_b >= tau_high] = 1

            # Bridge 1-to-2 step gaps
            for i in range(len(fl)):
                if fl[i] == 1:
                    if i + 2 < len(fl) and fl[i + 2] == 1:
                        fl[i + 1] = 1
                    if i + 3 < len(fl) and fl[i + 3] == 1:
                        fl[i + 1] = 1
                        fl[i + 2] = 1
        else:
            fl = np.zeros(b_len, dtype=np.int8)

        burst_flags.append(fl)

    all_flags = np.concatenate(burst_flags) if burst_flags else np.array([], dtype=np.int8)

    # Safety cap: If > 30% of steps are flagged, retain top steps >= 82nd percentile
    if len(all_flags) > 0 and all_flags.mean() > 0.30:
        p82 = np.percentile(s_test, 82)
        all_flags = ((all_flags == 1) & (s_test >= p82)).astype(np.int8)

    # Extract non-overlapping trigger windows of size W=32 within bursts
    windows = []
    scored_parts = []
    offset = 0

    for b_df in bursts:
        b_len = len(b_df)
        b_flags = all_flags[offset : offset + b_len]
        b_scores = s_test[offset : offset + b_len]
        offset += b_len

        part = b_df.copy()
        part["if_score"]   = b_scores.astype(np.float32)
        part["if_flag"]    = b_flags
        part["latency_us"] = lat_us
        scored_parts.append(part)

        flagged_pos = np.where(b_flags == 1)[0]
        if len(flagged_pos) > 0:
            merged = [int(flagged_pos[0])]
            for p in flagged_pos[1:]:
                if p - merged[-1] > WINDOW_SIZE:
                    merged.append(int(p))

            for pos in merged:
                half  = WINDOW_SIZE // 2
                start = max(0, pos - half)
                end   = min(b_len, start + WINDOW_SIZE)
                start = max(0, end - WINDOW_SIZE)
                win   = b_df.iloc[start:end]

                windows.append({
                    "chan_id"              : chan_id,
                    "burst_start_row"      : int(b_df.iloc[0]["original_row_idx"]),
                    "trigger_pos_in_burst" : pos,
                    "trigger_score"        : float(b_scores[pos]),
                    "window_rows"          : len(win),
                    "any_anomaly_in_win"   : int(win["Is_Anomaly"].any()),
                    "anomaly_rows_in_win"  : int(win["Is_Anomaly"].sum()),
                })

    scored_df = pd.concat(scored_parts, ignore_index=True) if scored_parts else pd.DataFrame()
    return scored_df, windows, s_test, all_flags, lat_us, tau_high, tau_low


# =============================================================================
# 3. METRICS EVALUATION (STEP-LEVEL, POINT-ADJUSTED, & EVENT-LEVEL)
# =============================================================================
def evaluate_channel(scored_df: pd.DataFrame,
                     windows: list,
                     chan_id: str,
                     meta_df: pd.DataFrame,
                     lat_us: float) -> dict:
    """
    Calculate Step-Level, Point-Adjusted (PA), and Event-Level Triage metrics.
    """
    y_true = scored_df["Is_Anomaly"].values
    y_pred = scored_df["if_flag"].values

    # Step-level metrics
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    # Point-Adjusted (PA) protocol & Event-Level Recall
    chan_meta = meta_df[meta_df["chan_id"] == chan_id]
    y_pred_pa = y_pred.copy()
    events_survived = 0
    events_caught   = 0

    if not chan_meta.empty:
        raw_seqs = chan_meta.iloc[0]["anomaly_sequences"]
        try:
            seqs = ast.literal_eval(raw_seqs) if isinstance(raw_seqs, str) else raw_seqs
            for s, e in seqs:
                mask = (scored_df["original_row_idx"] >= s) & (scored_df["original_row_idx"] <= e)
                if mask.sum() > 0 and y_true[mask].sum() > 0:
                    events_survived += 1
                    if (y_pred[mask] == 1).any():
                        events_caught += 1
                        y_pred_pa[mask] = 1
        except Exception:
            pass

    tp_pa   = int(((y_pred_pa == 1) & (y_true == 1)).sum())
    fp_pa   = int(((y_pred_pa == 1) & (y_true == 0)).sum())
    fn_pa   = int(((y_pred_pa == 0) & (y_true == 1)).sum())
    prec_pa = tp_pa / (tp_pa + fp_pa) if (tp_pa + fp_pa) > 0 else 0.0
    rec_pa  = tp_pa / (tp_pa + fn_pa) if (tp_pa + fn_pa) > 0 else 0.0
    f1_pa   = 2 * prec_pa * rec_pa / (prec_pa + rec_pa) if (prec_pa + rec_pa) > 0 else 0.0

    event_recall = (events_caught / events_survived) if events_survived > 0 else np.nan

    n_wins     = len(windows)
    n_wins_hit = sum(w["any_anomaly_in_win"] for w in windows)
    win_prec   = n_wins_hit / n_wins if n_wins > 0 else 0.0

    return {
        "chan_id"            : chan_id,
        "total_steps"        : len(scored_df),
        "anomaly_steps"      : int(y_true.sum()),
        "flagged_steps"      : int(y_pred.sum()),
        "TP"                 : tp,
        "FP"                 : fp,
        "FN"                 : fn,
        "TN"                 : tn,
        "precision"          : round(prec, 4),
        "recall"             : round(rec, 4),
        "f1"                 : round(f1, 4),
        "TP_pa"              : tp_pa,
        "FP_pa"              : fp_pa,
        "FN_pa"              : fn_pa,
        "prec_pa"            : round(prec_pa, 4),
        "rec_pa"             : round(rec_pa, 4),
        "f1_pa"              : round(f1_pa, 4),
        "events_survived"    : events_survived,
        "events_caught"      : events_caught,
        "event_recall"       : round(event_recall, 4) if pd.notna(event_recall) else np.nan,
        "windows_triggered"  : n_wins,
        "windows_on_anomaly" : n_wins_hit,
        "window_precision"   : round(win_prec, 4),
        "avg_latency_us"     : round(lat_us, 3),
    }


# =============================================================================
# 4. REPORTING
# =============================================================================
def print_report(summary_df: pd.DataFrame) -> None:
    """Print the structured performance report to stdout."""
    tp   = summary_df["TP"].sum()
    fp   = summary_df["FP"].sum()
    fn   = summary_df["FN"].sum()
    n_ch = len(summary_df)

    micro_p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    micro_r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)
                if (micro_p + micro_r) > 0 else 0.0)

    macro_p  = summary_df["precision"].mean()
    macro_r  = summary_df["recall"].mean()
    macro_f1 = summary_df["f1"].mean()

    # Point-Adjusted metrics
    tp_pa   = summary_df["TP_pa"].sum()
    fp_pa   = summary_df["FP_pa"].sum()
    fn_pa   = summary_df["FN_pa"].sum()
    micro_p_pa = tp_pa / (tp_pa + fp_pa) if (tp_pa + fp_pa) > 0 else 0.0
    micro_r_pa = tp_pa / (tp_pa + fn_pa) if (tp_pa + fn_pa) > 0 else 0.0
    micro_f1_pa = (2 * micro_p_pa * micro_r_pa / (micro_p_pa + micro_r_pa)
                   if (micro_p_pa + micro_r_pa) > 0 else 0.0)

    macro_p_pa = summary_df["prec_pa"].mean()
    macro_r_pa = summary_df["rec_pa"].mean()
    macro_f1_pa = summary_df["f1_pa"].mean()

    # Event-level triage
    total_survived_events = summary_df["events_survived"].sum()
    total_caught_events   = summary_df["events_caught"].sum()
    global_event_recall   = (total_caught_events / total_survived_events
                             if total_survived_events > 0 else 0.0)

    # Layer 2 Routing Quality
    total_wins   = summary_df["windows_triggered"].sum()
    correct_wins = summary_df["windows_on_anomaly"].sum()
    win_prec     = correct_wins / total_wins if total_wins > 0 else 0.0
    steps_in_wins = total_wins * WINDOW_SIZE
    tot_steps    = summary_df["total_steps"].sum()
    pct_saved    = max(0.0, (tot_steps - steps_in_wins) / tot_steps * 100)
    avg_lat      = summary_df["avg_latency_us"].mean()

    sep = "=" * 74

    print(f"\n{sep}")
    print("  LAYER 1  --  ISOLATION FOREST PERFORMANCE REPORT (v3)")
    print(f"  IEEE RadarConf 2027  |  {SPACECRAFT} dataset  |  {n_ch} channels")
    print(sep)

    print(f"\n  +-- 1) STEP-LEVEL CLASSIFICATION METRICS")
    print(f"  |  Total test steps       : {tot_steps:,}")
    print(f"  |  True anomaly steps     : {summary_df['anomaly_steps'].sum():,}  "
          f"({summary_df['anomaly_steps'].sum()/tot_steps*100:.2f}%)")
    print(f"  |  Steps flagged by IF    : {summary_df['flagged_steps'].sum():,}")
    print(f"  |")
    print(f"  |  -- Micro-averaged (pooled across all channels) --")
    print(f"  |  Precision  : {micro_p:.4f}")
    print(f"  |  Recall     : {micro_r:.4f}")
    print(f"  |  F1-score   : {micro_f1:.4f}")
    print(f"  |")
    print(f"  |  -- Macro-averaged (mean of per-channel scores) --")
    print(f"  |  Precision  : {macro_p:.4f}")
    print(f"  |  Recall     : {macro_r:.4f}")
    print(f"  |  F1-score   : {macro_f1:.4f}")
    print(f"  +--")

    print(f"\n  +-- 2) POINT-ADJUSTED (PA) METRICS (BENCHMARK PROTOCOL)")
    print(f"  |  (If an anomaly segment has >=1 trigger, segment is credited as detected)")
    print(f"  |")
    print(f"  |  -- Micro-averaged --")
    print(f"  |  PA-Precision : {micro_p_pa:.4f}")
    print(f"  |  PA-Recall    : {micro_r_pa:.4f}")
    print(f"  |  PA-F1-score  : {micro_f1_pa:.4f}")
    print(f"  |")
    print(f"  |  -- Macro-averaged --")
    print(f"  |  PA-Precision : {macro_p_pa:.4f}")
    print(f"  |  PA-Recall    : {macro_r_pa:.4f}")
    print(f"  |  PA-F1-score  : {macro_f1_pa:.4f}")
    print(f"  +--")

    print(f"\n  +-- 3) EVENT-LEVEL TRIAGE RECALL")
    print(f"  |  Total surviving events : {total_survived_events}")
    print(f"  |  Events caught by IF    : {total_caught_events}  ({global_event_recall*100:.2f}%)")
    print(f"  |  Event Recall           : {global_event_recall:.4f}")
    print(f"  +--")

    print(f"\n  +-- 4) LAYER 2 ROUTING QUALITY")
    print(f"  |  Total candidate windows sent to Layer 2 : {total_wins:,}")
    print(f"  |  Windows containing >=1 anomaly          : {correct_wins:,}  ({win_prec*100:.1f}%)")
    print(f"  |  False-alarm windows                     : {total_wins - correct_wins:,}  ({(1-win_prec)*100:.1f}%)")
    print(f"  |  Compute skipped / saved on edge         : {pct_saved:.1f}%")
    print(f"  +--")

    print(f"\n  +-- 5) EDGE COMPUTE EFFICIENCY")
    print(f"  |  Mean inference latency : {avg_lat:.2f} us / step")
    print(f"  |  Target for real-time deployment: < 1,000 us / step.")
    print(f"  +--")

    print(f"\n  +-- PER-CHANNEL BREAKDOWN  (sorted by PA-F1, descending)")
    hdr = (f"  |  {'Chan':<6} {'Steps':>5} {'Anom':>5} {'Flag':>5} "
           f"{'Prec':>6} {'Rec':>6} {'F1':>6} {'PA-F1':>6} {'EvtRec':>6} "
           f"{'Wins':>5} {'WinP':>6} {'us/step':>7}")
    print(hdr)
    print(f"  |  " + "-" * 76)

    for _, r in summary_df.sort_values("f1_pa", ascending=False).iterrows():
        evt_str = f"{r.event_recall:>6.3f}" if (r.anomaly_steps > 0 and pd.notna(r.event_recall)) else f"{'N/A':>6}"
        print(
            f"  |  {r.chan_id:<6} {int(r.total_steps):>5} "
            f"{int(r.anomaly_steps):>5} {int(r.flagged_steps):>5} "
            f"{r.precision:>6.3f} {r.recall:>6.3f} {r.f1:>6.3f} "
            f"{r.f1_pa:>6.3f} {evt_str} "
            f"{int(r.windows_triggered):>5} {r.window_precision:>6.3f} "
            f"{r.avg_latency_us:>7.2f}"
        )

    print(f"  +--")
    print(sep)


# =============================================================================
# 5. MAIN PIPELINE
# =============================================================================
def main() -> tuple:
    """
    End-to-end Layer 1 pipeline:
      1. Load irregular test data and ground-truth metadata
      2. For each channel:
         a. Extract 8 physics & contextual features
         b. Fit Robust/Standard scaler + IsolationForest(contamination='auto')
         c. Compute tail-calibrated anomaly score s_test
         d. Apply burst-aware hysteresis, spike preservation & denoising
         e. Extract W=32 candidate windows
      3. Compute Step-Level, Point-Adjusted, and Event-Level Recall metrics
      4. Save parquet and CSV results
      5. Print structured performance report
    """
    print("\n" + "=" * 74)
    print("  LAYER 1  --  HIGH-SIGNAL ISOLATION FOREST EDGE TRIAGE (v3)")
    print("  IEEE RadarConf 2027  |  Asynchronous Telemetry Triage System")
    print("=" * 74)

    # 1 / 4  Load data
    print("\n[1/4]  Loading irregular test data and metadata ...")
    df_all    = pd.read_parquet(IRREGULAR_PKL)
    meta_df   = pd.read_csv(META_CSV)
    meta_smap = meta_df[meta_df["spacecraft"] == SPACECRAFT]

    channels = sorted(df_all["chan_id"].unique())
    print(f"       Test channels : {len(channels)}")
    print(f"       Total rows    : {len(df_all):,}")
    print(f"       Train source  : {TRAIN_DIR}")
    print(f"       IF Params     : n_estimators={N_ESTIMATORS}, max_samples={MAX_SAMPLES}, contamination='auto'")
    print(f"       Features (8)  : {FEATURE_NAMES}")

    # 2 / 4  Process per channel
    print("\n[2/4]  Fitting models and executing edge triage ...")
    print(f"       {'Chan':<6} {'Bursts':>6} {'Flags':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'PA-F1':>6} {'EvtRec':>6} {'Wins':>5}")
    print("       " + "-" * 66)

    all_scored   = []
    all_windows  = []
    summaries    = []
    models_dict  = {}

    for chan_id in channels:
        train_path = TRAIN_DIR / f"{chan_id}.npy"
        if not train_path.exists():
            continue

        train_arr = np.load(train_path)
        test_df   = df_all[df_all["chan_id"] == chan_id].copy().reset_index(drop=True)

        # 1. Feature extraction
        F_train, F_test, bursts, _ = extract_channel_features(train_arr, test_df)

        # 2. Scale F_train and fit Isolation Forest
        scaler = StandardScaler()
        F_train_scaled = np.clip(scaler.fit_transform(F_train), -10, 10)

        clf = IsolationForest(
            n_estimators  = N_ESTIMATORS,
            max_samples   = MAX_SAMPLES,
            contamination = "auto",
            random_state  = RANDOM_SEED,
            n_jobs        = 1,
        )
        clf.fit(F_train_scaled)
        models_dict[chan_id] = (clf, scaler)

        # 3. Score test bursts & apply burst-aware hysteresis
        scored_df, wins, s_test, all_flags, lat_us, tau_h, tau_l = score_and_filter_channel(
            clf, scaler, F_train, F_test, bursts, chan_id
        )

        # 4. Evaluate metrics
        metrics = evaluate_channel(scored_df, wins, chan_id, meta_smap, lat_us)

        all_scored.append(scored_df)
        all_windows.extend(wins)
        summaries.append(metrics)

        evt_str = f"{metrics['event_recall']:>6.3f}" if (metrics['anomaly_steps'] > 0 and pd.notna(metrics['event_recall'])) else f"{'N/A':>6}"
        print(
            f"       {chan_id:<6} {len(bursts):>6} {metrics['flagged_steps']:>6} "
            f"{metrics['precision']:>6.3f} {metrics['recall']:>6.3f} {metrics['f1']:>6.3f} "
            f"{metrics['f1_pa']:>6.3f} {evt_str} {metrics['windows_triggered']:>5}"
        )

    # 3 / 4  Save results
    print("\n[3/4]  Saving outputs ...")
    summary_df    = pd.DataFrame(summaries)
    df_scored_all = pd.concat(all_scored, ignore_index=True)

    df_scored_all.to_parquet(OUT_RESULTS)
    pd.DataFrame(all_windows).to_parquet(OUT_WINDOWS)
    summary_df.to_csv(OUT_SUMMARY, index=False)

    # 4 / 4  Print full report
    print("\n[4/4]  Performance Report")
    print_report(summary_df)

    print(f"\n  Outputs")
    print(f"  +-- Scored rows  -> {OUT_RESULTS}")
    print(f"  +-- Trigger wins -> {OUT_WINDOWS}")
    print(f"  +-- Summary CSV  -> {OUT_SUMMARY}")
    print("\n  Next step -> Layer 2 (PhysDiff) reads the trigger windows.\n")

    return models_dict, summary_df, all_windows


# =============================================================================
if __name__ == "__main__":
    models, summary, windows = main()
