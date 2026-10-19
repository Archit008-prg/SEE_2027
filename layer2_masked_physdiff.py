"""
Layer 2: Masked PhysDiff (Physics-Guided Conditional Masked Diffusion)
Edge Anomaly Detector & Root-Cause Classifier for Irregular Spacecraft Telemetry
Designed for IEEE RadarConf 2027 Benchmark Pipeline

Inputs:
  - nasa_smap_data/all_smap_irregular.parquet
  - nasa_smap_data/layer1_results.parquet
  - nasa_smap_data/layer1_windows.parquet
  - nasa_smap_data/labeled_anomalies.csv
  - train/{chan_id}.npy

Outputs:
  - nasa_smap_data/layer2_results.parquet
  - nasa_smap_data/final_benchmark_summary.csv
"""

import os
import ast
import time
import math
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score

warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIGURATION & REPRODUCIBILITY
# ==============================================================================
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUT_DIR = "nasa_smap_data"
IRREG_PATH = os.path.join(OUT_DIR, "all_smap_irregular.parquet")
L1_RES_PATH = os.path.join(OUT_DIR, "layer1_results.parquet")
L1_WIN_PATH = os.path.join(OUT_DIR, "layer1_windows.parquet")
L1_SUM_PATH = os.path.join(OUT_DIR, "layer1_summary.csv")

CANDIDATE_DATA_DIRS = [
    os.path.join(OUT_DIR, "data", "data"),
    os.path.join(OUT_DIR, "data"),
    "data/data",
    "data",
    OUT_DIR,
    "."
]

def find_path(rel_target):
    for d in CANDIDATE_DATA_DIRS:
        p = os.path.join(d, rel_target)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Cannot locate {rel_target} across candidate directories")

TRAIN_DIR = find_path("train")
LABELS_CSV = find_path("labeled_anomalies.csv")

# Hyperparameters
WIN_SIZE = 32
DIFF_STEPS = 20
HIDDEN_DIM = 64
BATCH_SIZE = 64
MAX_EPOCHS = 8
LR = 3e-3
PATIENCE = 3


# ==============================================================================
# 1. HELPER PARSERS & AUTO-METADATA DISCOVERY
# ==============================================================================
def parse_class_list(val):
    if pd.isna(val):
        return []
    s = str(val).strip("[] \t\r\n")
    return [x.strip(" '\"") for x in s.split(",") if x.strip(" '\"")]

def parse_seq_list(val):
    if pd.isna(val):
        return []
    if isinstance(val, (list, tuple)):
        return list(val)
    try:
        return ast.literal_eval(val)
    except Exception:
        import json
        try:
            return json.loads(val)
        except Exception:
            return []

def load_and_discover_layer1():
    df_irreg = pd.read_parquet(IRREG_PATH)
    df_l1_res = pd.read_parquet(L1_RES_PATH)
    df_l1_win = pd.read_parquet(L1_WIN_PATH)
    df_labels_raw = pd.read_csv(LABELS_CSV)
    df_labels_raw = df_labels_raw[df_labels_raw["spacecraft"] == "SMAP"].copy().reset_index(drop=True)

    # 1. Deduplicate channels in labeled_anomalies.csv (combines duplicate rows like P-2)
    consolidated_records = []
    for cid, grp in df_labels_raw.groupby("chan_id", sort=False):
        classes = []
        seqs = []
        for _, r in grp.iterrows():
            classes.extend(parse_class_list(r["class"]))
            seqs.extend(parse_seq_list(r["anomaly_sequences"]))
        p_type = "point" if "point" in classes and "contextual" not in classes else "contextual"
        consolidated_records.append({
            "chan_id": cid,
            "class": classes,
            "primary_type": p_type,
            "anomaly_sequences": seqs
        })
    df_labels = pd.DataFrame(consolidated_records)

    feat_cols = sorted([c for c in df_irreg.columns if c.startswith("feat_")])

    # Adapt to df_l1_res columns
    res_cols = df_l1_res.columns.tolist()
    flag_col = next((c for c in ["if_flag", "flag", "pred", "preds"] if c in res_cols), "if_flag")
    score_col = next((c for c in ["if_score", "score", "anom_score", "s_test"] if c in res_cols), "if_score")
    idx_col = next((c for c in ["original_row_idx", "orig_idx", "row_idx"] if c in res_cols), "original_row_idx")

    # Adapt to df_l1_win columns
    win_cols = df_l1_win.columns.tolist()
    b_start_col = next((c for c in ["burst_start_row", "burst_start", "b_start"] if c in win_cols), None)
    t_pos_col = next((c for c in ["trigger_pos_in_burst", "trigger_pos", "t_pos"] if c in win_cols), None)
    w_rows_col = next((c for c in ["window_rows", "win_rows", "w_rows"] if c in win_cols), None)
    w_start_col = next((c for c in ["win_start_pos", "start_pos", "w_start"] if c in win_cols), None)
    w_end_col = next((c for c in ["win_end_pos", "end_pos", "w_end"] if c in win_cols), None)
    orig_s_col = next((c for c in ["orig_idx_start", "orig_start", "start_orig_idx"] if c in win_cols), None)
    orig_e_col = next((c for c in ["orig_idx_end", "orig_end", "end_orig_idx"] if c in win_cols), None)
    win_score_col = next((c for c in ["trigger_score", "score", "anom_score"] if c in win_cols), None)

    meta = {
        "feat_cols": feat_cols,
        "flag_col": flag_col,
        "score_col": score_col,
        "idx_col": idx_col,
        "b_start_col": b_start_col,
        "t_pos_col": t_pos_col,
        "w_rows_col": w_rows_col,
        "w_start_col": w_start_col,
        "w_end_col": w_end_col,
        "orig_s_col": orig_s_col,
        "orig_e_col": orig_e_col,
        "win_score_col": win_score_col,
    }
    return df_irreg, df_l1_res, df_l1_win, df_labels, meta


# ==============================================================================
# 2. MASKED FREQUENCY DECOMPOSITION (P_low, P_high) ON OBSERVED TIMESTAMPS
# ==============================================================================
def compute_masked_priors_torch(X, M, kernel_sigma=2.5):
    """
    X: (B, W, D) observed telemetry on physical grid (0.0 where missing)
    M: (B, W, D) binary mask (1.0 where observed, 0.0 where radio blackout)
    Returns:
      P_low:  (B, W, D) Low-frequency physical trend via Nadaraya-Watson masked smoother
      P_high: (B, W, D) High-frequency transient residual strictly on observed steps
    """
    W = X.shape[1]
    t = torch.arange(W, device=X.device, dtype=X.dtype)
    dist_sq = (t.unsqueeze(0) - t.unsqueeze(1)) ** 2
    K = torch.exp(-dist_sq / (2.0 * (kernel_sigma ** 2)))

    XM = X * M
    num = torch.einsum("ij,bjd->bid", K, XM)
    den = torch.einsum("ij,bjd->bid", K, M) + 1e-6

    P_low = (num / den) * M
    P_high = (X - P_low) * M
    return P_low, P_high


# ==============================================================================
# 3. LIGHTWEIGHT CONDITIONAL MASKED DIFFUSION (Frequency-Routing + DDPM)
# ==============================================================================
class SinusoidalStepEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

class LiteMaskedPhysDiff(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, n_steps=20):
        super().__init__()
        self.in_dim = in_dim
        self.n_steps = n_steps

        betas = torch.linspace(1e-4, 0.02, n_steps)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)

        self.step_emb = nn.Sequential(
            SinusoidalStepEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )

        self.x_proj = nn.Linear(in_dim * 2, hidden_dim)
        self.pl_proj = nn.Linear(in_dim, hidden_dim)
        self.ph_proj = nn.Linear(in_dim, hidden_dim)

        self.gate_net = nn.Linear(hidden_dim * 3, 2)

        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.out_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, in_dim)
        )

    def forward(self, x_t, t_step, P_low, P_high, M):
        h_x = self.x_proj(torch.cat([x_t * M, M], dim=-1))
        h_pl = self.pl_proj(P_low)
        h_ph = self.ph_proj(P_high)

        gates = torch.softmax(self.gate_net(torch.cat([h_x, h_pl, h_ph], dim=-1)), dim=-1)
        g_low = gates[..., 0:1]
        g_high = gates[..., 1:2]

        t_e = self.step_emb(t_step).unsqueeze(1)
        h = h_x + g_low * h_pl + g_high * h_ph + t_e

        h_conv = F.silu(self.conv1(h.transpose(1, 2)).transpose(1, 2))
        h = self.norm1(h + h_conv)
        h_attn, _ = self.attn(h, h, h)
        h = self.norm2(h + h_attn)

        eps_pred = self.out_head(h)
        return eps_pred * M

    def compute_masked_loss(self, x0, M):
        B = x0.shape[0]
        P_low, P_high = compute_masked_priors_torch(x0, M)
        t = torch.randint(0, self.n_steps, (B,), device=x0.device)
        a_bar = self.alpha_bar[t].view(B, 1, 1)

        eps = torch.randn_like(x0) * M
        x_t = (torch.sqrt(a_bar) * x0 + torch.sqrt(1.0 - a_bar) * eps) * M

        eps_pred = self.forward(x_t, t, P_low, P_high, M)

        obs_count = M.sum() + 1e-6
        l_diff = (((eps_pred - eps) ** 2) * M).sum() / obs_count

        x0_hat = ((x_t - torch.sqrt(1.0 - a_bar) * eps_pred) / (torch.sqrt(a_bar) + 1e-6)).clamp(-10.0, 10.0) * M
        pl_hat, ph_hat = compute_masked_priors_torch(x0_hat, M)
        l_phys = ((((pl_hat - P_low) ** 2) + ((ph_hat - P_high) ** 2)) * M).sum() / obs_count

        return l_diff + 0.25 * l_phys

    @torch.no_grad()
    def reconstruct_window(self, x0, M):
        P_low, P_high = compute_masked_priors_torch(x0, M)
        B = x0.shape[0]
        eval_steps = [6, 12, 18]
        x0_preds = []

        for step_idx in eval_steps:
            t = torch.full((B,), step_idx, device=x0.device, dtype=torch.long)
            a_bar = self.alpha_bar[t].view(B, 1, 1)
            torch.manual_seed(SEED + step_idx)
            eps = torch.randn_like(x0) * M
            x_t = (torch.sqrt(a_bar) * x0 + torch.sqrt(1.0 - a_bar) * eps) * M
            eps_pred = self.forward(x_t, t, P_low, P_high, M)
            x0_hat = ((x_t - torch.sqrt(1.0 - a_bar) * eps_pred) / (torch.sqrt(a_bar) + 1e-6)) * M
            x0_preds.append(x0_hat)

        x0_rec = torch.stack(x0_preds, dim=0).mean(dim=0)
        pl_rec, ph_rec = compute_masked_priors_torch(x0_rec, M)
        return x0_rec, P_low, P_high, pl_rec, ph_rec


# ==============================================================================
# 4. POINT-ADJUSTED (PA) PROTOCOL IDENTICAL FOR BOTH LAYERS
# ==============================================================================
def compute_pa_predictions_event(y_true, y_pred, orig_idx, gt_seqs):
    """
    Standard Point-Adjusted (PA) protocol on SMAP anomaly events:
    If any predicted flag intersects the event interval [s, e], all true anomaly
    timesteps within that interval are credited as detected.
    """
    pa_pred = y_pred.copy()
    for s, e in gt_seqs:
        mask = (orig_idx >= s) & (orig_idx <= e)
        if np.any(mask) and np.any(y_true[mask] == 1):
            if np.any(y_pred[mask] == 1):
                pa_pred[mask & (y_true == 1)] = 1
    return pa_pred


# ==============================================================================
# 5. LAYER 2 PIPELINE EXECUTION
# ==============================================================================
def run_layer2():
    print("=" * 78, flush=True)
    print("  LAYER 2 -- MASKED PHYSDIFF (PHYSICS-GUIDED DIFFUSION CASCADE)", flush=True)
    print(f"  IEEE RadarConf 2027 | Device: {DEVICE} | Window W={WIN_SIZE}, Steps T={DIFF_STEPS}", flush=True)
    print("=" * 78, flush=True)

    df_irreg, df_l1_res, df_l1_win, df_labels, meta = load_and_discover_layer1()
    feat_cols = meta["feat_cols"]
    D = len(feat_cols)

    print(f"\n[1/4] Layer 1 Schema & Candidate Windows:", flush=True)
    print(f"      Irregular rows : {len(df_irreg):,} | Features D={D}", flush=True)
    print(f"      Unique channels: {len(df_labels)} (deduplicated P-2)", flush=True)
    print(f"      L1 Windows     : {len(df_l1_win):,} candidate trigger windows", flush=True)

    all_l2_rows = []
    summary_records = []

    total_windows_evaluated = 0
    total_windows_rejected = 0
    true_anom_wins_kept = 0
    false_alarm_wins_rejected = 0

    correct_root_cause = 0
    total_classified_events = 0

    t_pipeline_start = time.perf_counter()

    print(f"\n[2/4] Training Masked PhysDiff & Evaluating ONLY Routed Candidate Windows ...", flush=True)
    print(f"{'Chan':<6} {'L1_Win':>6} {'Kept':>5} {'FA_Rej':>6} {'L1_F1':>6} {'L2_F1':>6} {'L2_PAF1':>7} {'PredType':<11} {'GT_Type':<11}", flush=True)
    print("-" * 78, flush=True)

    for _, row in df_labels.iterrows():
        chan_id = row["chan_id"]
        gt_primary_type = row["primary_type"]
        gt_seqs = row["anomaly_sequences"]

        df_c_irreg = df_irreg[df_irreg["chan_id"] == chan_id].sort_values("original_row_idx").reset_index(drop=True)
        df_c_res = df_l1_res[df_l1_res["chan_id"] == chan_id].sort_values(meta["idx_col"]).reset_index(drop=True)
        df_c_win = df_l1_win[df_l1_win["chan_id"] == chan_id].reset_index(drop=True)

        if df_c_irreg.empty or df_c_res.empty:
            continue

        X_test_raw = df_c_irreg[feat_cols].values.astype(np.float32)
        y_true = df_c_irreg["Is_Anomaly"].values.astype(int)
        delta_t = df_c_irreg["delta_t"].values.astype(np.float32)
        orig_idx = df_c_irreg["original_row_idx"].values.astype(int)

        l1_flags = df_c_res[meta["flag_col"]].values.astype(int)
        l1_scores = df_c_res[meta["score_col"]].values.astype(np.float32) if meta["score_col"] in df_c_res else l1_flags.astype(np.float32)

        l1_f1 = f1_score(y_true, l1_flags, zero_division=0)
        l1_pa_preds = compute_pa_predictions_event(y_true, l1_flags, orig_idx, gt_seqs)
        l1_paf1 = f1_score(y_true, l1_pa_preds, zero_division=0)

        # If Layer 1 routed 0 windows, Layer 2 completely sleeps
        if df_c_win.empty or np.sum(l1_flags) == 0:
            print(f"{chan_id:<6} {0:>6} {0:>5} {'0%':>6} {l1_f1:>6.3f} {l1_f1:>6.3f} {l1_paf1:>7.3f} {'none':<11} {gt_primary_type:<11}", flush=True)
            df_c_out = df_c_res.copy()
            df_c_out["l2_flag"] = 0
            df_c_out["l2_phys_score"] = 0.0
            df_c_out["pred_anomaly_type"] = "none"
            df_c_out["gt_anomaly_type"] = gt_primary_type
            all_l2_rows.append(df_c_out)
            continue

        # ----------------------------------------------------------------------
        # A. Load Nominal train/{chan_id}.npy & Standardize
        # ----------------------------------------------------------------------
        train_path = os.path.join(TRAIN_DIR, f"{chan_id}.npy")
        train_mat = np.load(train_path).astype(np.float32)

        train_mu = np.mean(train_mat, axis=0, keepdims=True)
        train_sigma = np.std(train_mat, axis=0, keepdims=True)
        train_sigma_safe = np.where(train_sigma < 1e-3, 1.0, train_sigma)

        train_mu_p0 = float(train_mu[0, 0])
        train_std_p0 = float(train_sigma[0, 0])

        X_train_norm = np.clip((train_mat - train_mu) / train_sigma_safe, -8.0, 8.0)
        X_test_norm = np.clip((X_test_raw - train_mu) / train_sigma_safe, -8.0, 8.0)

        # Slice training windows with simulated blackout masks
        train_wins, train_masks = [], []
        stride = 12
        for s in range(0, max(1, len(X_train_norm) - WIN_SIZE + 1), stride):
            w = X_train_norm[s:s + WIN_SIZE]
            if len(w) < WIN_SIZE:
                pad = np.zeros((WIN_SIZE - len(w), D), dtype=np.float32)
                w = np.vstack([w, pad])
            m = (np.random.rand(WIN_SIZE, 1) > 0.20).astype(np.float32)
            m[0, 0] = 1.0
            m_full = np.repeat(m, D, axis=1)
            train_wins.append(w * m_full)
            train_masks.append(m_full)

        t_X = torch.tensor(np.array(train_wins), dtype=torch.float32, device=DEVICE)
        t_M = torch.tensor(np.array(train_masks), dtype=torch.float32, device=DEVICE)

        # ----------------------------------------------------------------------
        # B. Train LiteMaskedPhysDiff
        # ----------------------------------------------------------------------
        model = LiteMaskedPhysDiff(in_dim=D, hidden_dim=HIDDEN_DIM, n_steps=DIFF_STEPS).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

        loader = DataLoader(TensorDataset(t_X, t_M), batch_size=BATCH_SIZE, shuffle=True)
        best_loss = float("inf")
        patience_cnt = 0

        model.train()
        for epoch in range(MAX_EPOCHS):
            ep_loss = 0.0
            for bx, bm in loader:
                optimizer.zero_grad()
                loss = model.compute_masked_loss(bx, bm)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                ep_loss += loss.item() * len(bx)
            ep_loss /= max(1, len(t_X))
            if ep_loss < best_loss - 1e-4:
                best_loss = ep_loss
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= PATIENCE:
                    break

        # ----------------------------------------------------------------------
        # C. Calibrate Separate Primary vs Auxiliary Reconstruction Errors
        # ----------------------------------------------------------------------
        model.eval()
        with torch.no_grad():
            ones_M = torch.ones_like(t_X)
            x0_rec_tr, pl_in_tr, ph_in_tr, pl_rec_tr, ph_rec_tr = model.reconstruct_window(t_X, ones_M)
            
            # Primary sensor (feat_00) diffusion reconstruction errors
            err_p0_hf_tr = torch.abs(ph_in_tr[:, :, 0] - ph_rec_tr[:, :, 0]).cpu().numpy()
            err_p0_lf_tr = torch.abs(pl_in_tr[:, :, 0] - pl_rec_tr[:, :, 0]).cpu().numpy()
            err_p0_tr = 0.5 * err_p0_hf_tr + 0.5 * err_p0_lf_tr
            
            # Auxiliary command channel errors
            err_aux_tr = torch.mean(torch.abs(t_X[:, :, 1:] - x0_rec_tr[:, :, 1:]), dim=-1).cpu().numpy()

            # Calibrate 97.5th and 96.5th percentiles strictly on nominal training distribution
            tau_p0 = float(np.percentile(err_p0_tr, 97.5))
            tau_win_p0 = float(np.percentile(np.percentile(err_p0_tr, 90, axis=1), 96.5))

        # ----------------------------------------------------------------------
        # D. Precision False-Alarm Filter & Selective Window Confirmation
        # ----------------------------------------------------------------------
        l2_flags = np.zeros_like(l1_flags)
        l2_scores = np.zeros(len(X_test_norm), dtype=np.float32)

        burst_id = (delta_t > 1.0).cumsum()
        chan_wins_kept = 0
        chan_wins_total = len(df_c_win)

        orig_to_pos = {idx: i for i, idx in enumerate(orig_idx)}

        # Channel candidate window scores distribution
        win_scores_list = []
        if meta["win_score_col"] and meta["win_score_col"] in df_c_win:
            chan_l1_p45 = float(df_c_win[meta["win_score_col"]].quantile(0.45))
        else:
            chan_l1_p45 = 0.0

        all_test_e_p0 = np.zeros(len(X_test_norm), dtype=np.float32)
        all_test_e_p0_hf = np.zeros(len(X_test_norm), dtype=np.float32)
        all_test_e_p0_lf = np.zeros(len(X_test_norm), dtype=np.float32)
        all_test_e_aux = np.zeros(len(X_test_norm), dtype=np.float32)

        for _, w_row in df_c_win.iterrows():
            total_windows_evaluated += 1

            if meta["b_start_col"] and meta["t_pos_col"]:
                b_start = int(w_row[meta["b_start_col"]])
                t_pos = int(w_row[meta["t_pos_col"]])
                t_orig = b_start + t_pos
            elif meta["w_start_col"] and meta["w_end_col"]:
                ws = int(w_row[meta["w_start_col"]])
                t_orig = orig_idx[min(ws, len(orig_idx) - 1)]
            else:
                os_idx = int(w_row[meta["orig_s_col"]])
                t_orig = os_idx

            g_start = max(0, t_orig - (WIN_SIZE // 2))
            g_end = g_start + WIN_SIZE

            w_arr = np.zeros((1, WIN_SIZE, D), dtype=np.float32)
            m_arr = np.zeros((1, WIN_SIZE, D), dtype=np.float32)

            observed_local_indices = []
            observed_test_indices = []

            for k in range(WIN_SIZE):
                physical_t = g_start + k
                if physical_t in orig_to_pos:
                    pos = orig_to_pos[physical_t]
                    w_arr[0, k, :] = X_test_norm[pos, :]
                    m_arr[0, k, :] = 1.0
                    observed_local_indices.append(k)
                    observed_test_indices.append(pos)

            if len(observed_test_indices) == 0:
                continue

            wx = torch.tensor(w_arr, device=DEVICE)
            wm = torch.tensor(m_arr, device=DEVICE)

            with torch.no_grad():
                x0_r, pl_i, ph_i, pl_r, ph_r = model.reconstruct_window(wx, wm)
                e_p0_hf = torch.abs(ph_i[0, observed_local_indices, 0] - ph_r[0, observed_local_indices, 0]).cpu().numpy()
                e_p0_lf = torch.abs(pl_i[0, observed_local_indices, 0] - pl_r[0, observed_local_indices, 0]).cpu().numpy()
                e_p0 = 0.5 * e_p0_hf + 0.5 * e_p0_lf
                e_aux = torch.mean(torch.abs(wx[0, observed_local_indices, 1:] - x0_r[0, observed_local_indices, 1:]), dim=-1).cpu().numpy()

            obs_test_pos = np.array(observed_test_indices)
            all_test_e_p0[obs_test_pos] = np.maximum(all_test_e_p0[obs_test_pos], e_p0)
            all_test_e_p0_hf[obs_test_pos] = np.maximum(all_test_e_p0_hf[obs_test_pos], e_p0_hf)
            all_test_e_p0_lf[obs_test_pos] = np.maximum(all_test_e_p0_lf[obs_test_pos], e_p0_lf)
            all_test_e_aux[obs_test_pos] = np.maximum(all_test_e_aux[obs_test_pos], e_aux)

            sub_raw_p0 = X_test_raw[obs_test_pos, 0]
            raw_p0_range = float(np.ptp(sub_raw_p0))
            raw_p0_dev = float(np.max(np.abs(sub_raw_p0 - train_mu_p0)) / (train_std_p0 + 1e-4))

            win_p0_p90 = float(np.percentile(e_p0, 90))
            win_l1_score = float(w_row[meta["win_score_col"]]) if meta["win_score_col"] and meta["win_score_col"] in w_row else 0.0

            # 3. Precision False-Alarm Filter
            # Reject window if:
            # a) Primary sensor is dead-flat/unchanged (raw_p0_range < 1e-3 and raw_p0_dev < 1.5)
            # b) Both primary diffusion error and Layer 1 score are weak (bottom 45% of triggers without primary deviation)
            is_flat_sensor = (raw_p0_range < 1e-3 and raw_p0_dev < 1.5)
            is_weak_diff_and_l1 = (win_p0_p90 < tau_win_p0 and win_l1_score <= chan_l1_p45 and raw_p0_dev < 2.5)

            reject_window = is_flat_sensor or is_weak_diff_and_l1
            has_gt = bool(np.any(y_true[obs_test_pos] == 1))

            if not reject_window:
                chan_wins_kept += 1
                if has_gt:
                    true_anom_wins_kept += 1

                # Keep union of Layer 1 high-confidence flags and steps where primary diffusion error exceeds threshold
                active_steps = (e_p0 > tau_p0 * 0.85) | (l1_flags[obs_test_pos] == 1)
                confirmed_pos = obs_test_pos[active_steps]
                l2_flags[confirmed_pos] = 1
                l2_scores[obs_test_pos] = np.maximum(l2_scores[obs_test_pos], e_p0)

                # Burst propagation: if >35% of burst is confirmed, confirm burst
                for b_id in np.unique(burst_id[confirmed_pos]):
                    b_mask = (burst_id == b_id)
                    if np.mean(l2_flags[b_mask]) >= 0.35:
                        l2_flags[b_mask] = 1
            else:
                total_windows_rejected += 1
                if not has_gt:
                    false_alarm_wins_rejected += 1

        # ----------------------------------------------------------------------
        # E. Accurate Physics Root-Cause Classifier (Point vs Contextual)
        # ----------------------------------------------------------------------
        aux_activity = float(np.mean(np.std(X_test_raw[:, 1:], axis=0) > 0.05))
        flagged_mask = (l2_flags == 1)

        if np.sum(l2_flags) > 0:
            p0_kurtosis = float(np.max(all_test_e_p0_hf) / (np.mean(all_test_e_p0_hf) + 1e-5))
            lf_to_hf = float(np.mean(all_test_e_p0_lf) / (np.mean(all_test_e_p0_hf) + 1e-5))
            cmd_corr_ratio = float(np.mean(all_test_e_aux[flagged_mask]) / (np.mean(all_test_e_p0[flagged_mask]) + 1e-5))

            if (cmd_corr_ratio > 0.45 and aux_activity > 0.12) or (p0_kurtosis < 2.8 and aux_activity > 0.20):
                pred_type = "contextual"
            else:
                pred_type = "point"
        else:
            pred_type = "none"

        if np.any(y_true == 1) and pred_type != "none":
            total_classified_events += 1
            if pred_type == gt_primary_type:
                correct_root_cause += 1

        # Channel Metrics
        l2_f1 = f1_score(y_true, l2_flags, zero_division=0)
        l2_pa_preds = compute_pa_predictions_event(y_true, l2_flags, orig_idx, gt_seqs)
        l2_paf1 = f1_score(y_true, l2_pa_preds, zero_division=0)

        rej_pct = ((chan_wins_total - chan_wins_kept) / max(1, chan_wins_total)) * 100.0
        print(f"{chan_id:<6} {chan_wins_total:>6} {chan_wins_kept:>5} {rej_pct:>5.1f}% "
              f"{l1_f1:>6.3f} {l2_f1:>6.3f} {l2_paf1:>7.3f} {pred_type:<11} {gt_primary_type:<11}", flush=True)

        df_c_out = df_c_res.copy()
        df_c_out["l2_flag"] = l2_flags
        df_c_out["l2_phys_score"] = l2_scores
        df_c_out["pred_anomaly_type"] = pred_type
        df_c_out["gt_anomaly_type"] = gt_primary_type
        all_l2_rows.append(df_c_out)

        summary_records.append({
            "chan_id": chan_id,
            "steps": len(y_true),
            "true_anom": int(np.sum(y_true)),
            "l1_flags": int(np.sum(l1_flags)),
            "l2_flags": int(np.sum(l2_flags)),
            "l1_windows": chan_wins_total,
            "l2_windows_kept": chan_wins_kept,
            "l1_step_f1": l1_f1,
            "l2_step_f1": l2_f1,
            "l1_pa_f1": l1_paf1,
            "l2_pa_f1": l2_paf1,
            "pred_type": pred_type,
            "gt_type": gt_primary_type
        })

    # ==============================================================================
    # 6. FINAL BENCHMARK AGGREGATION & REPORT (IEEE RADARCONF 2027)
    # ==============================================================================
    total_time = time.perf_counter() - t_pipeline_start
    df_all_out = pd.concat(all_l2_rows, ignore_index=True)
    df_sum_out = pd.DataFrame(summary_records)

    # Save Results
    df_all_out.to_parquet(os.path.join(OUT_DIR, "layer2_results.parquet"), index=False)
    df_sum_out.to_csv(os.path.join(OUT_DIR, "final_benchmark_summary.csv"), index=False)

    y_all_true = df_all_out["Is_Anomaly"].values.astype(int)
    y_l1_pred = df_all_out[meta["flag_col"]].values.astype(int)
    y_l2_pred = df_all_out["l2_flag"].values.astype(int)

    # Compute Point-Adjusted (PA) vectors per channel using exact ground-truth sequences
    pa_l1_list, pa_l2_list = [], []
    for cid, grp in df_all_out.groupby("chan_id", sort=False):
        yt = grp["Is_Anomaly"].values.astype(int)
        orig = grp[meta["idx_col"]].values.astype(int)
        seqs = df_labels[df_labels["chan_id"] == cid].iloc[0]["anomaly_sequences"]
        
        pa_l1_list.append(compute_pa_predictions_event(yt, grp[meta["flag_col"]].values.astype(int), orig, seqs))
        pa_l2_list.append(compute_pa_predictions_event(yt, grp["l2_flag"].values.astype(int), orig, seqs))

    y_l1_pa = np.concatenate(pa_l1_list)
    y_l2_pa = np.concatenate(pa_l2_list)

    # Step-Level Micro Metrics
    l1_p = precision_score(y_all_true, y_l1_pred, zero_division=0)
    l1_r = recall_score(y_all_true, y_l1_pred, zero_division=0)
    l1_f1 = f1_score(y_all_true, y_l1_pred, zero_division=0)

    l2_p = precision_score(y_all_true, y_l2_pred, zero_division=0)
    l2_r = recall_score(y_all_true, y_l2_pred, zero_division=0)
    l2_f1 = f1_score(y_all_true, y_l2_pred, zero_division=0)

    # Point-Adjusted Micro Metrics
    l1_pap = precision_score(y_all_true, y_l1_pa, zero_division=0)
    l1_par = recall_score(y_all_true, y_l1_pa, zero_division=0)
    l1_paf1 = f1_score(y_all_true, y_l1_pa, zero_division=0)

    l2_pap = precision_score(y_all_true, y_l2_pa, zero_division=0)
    l2_par = recall_score(y_all_true, y_l2_pa, zero_division=0)
    l2_paf1 = f1_score(y_all_true, y_l2_pa, zero_division=0)

    fa_rej_rate = (total_windows_rejected / max(1, total_windows_evaluated)) * 100.0
    cls_acc = (correct_root_cause / max(1, total_classified_events)) * 100.0

    dense_windows = len(df_all_out)
    cascade_speedup = dense_windows / max(1, total_windows_evaluated)
    compute_saved_pct = (1.0 - (total_windows_evaluated / dense_windows)) * 100.0

    print("\n" + "=" * 78, flush=True)
    print("  FINAL END-TO-END BENCHMARK REPORT (IEEE RADARCONF 2027 TABLE 1)", flush=True)
    print("=" * 78, flush=True)
    print(f"  {'Metric':<34} | {'Layer 1 Alone':>17} | {'Layer 1 + Layer 2':>17}", flush=True)
    print("  " + "-" * 72, flush=True)
    print(f"  {'Step-Level Precision':<34} | {l1_p:>17.4f} | {l2_p:>17.4f}", flush=True)
    print(f"  {'Step-Level Recall':<34} | {l1_r:>17.4f} | {l2_r:>17.4f}", flush=True)
    print(f"  {'Step-Level F1-Score':<34} | {l1_f1:>17.4f} | {l2_f1:>17.4f}", flush=True)
    print("  " + "-" * 72, flush=True)
    print(f"  {'Point-Adjusted Precision (PA-P)':<34} | {l1_pap:>17.4f} | {l2_pap:>17.4f}", flush=True)
    print(f"  {'Point-Adjusted Recall (PA-R)':<34} | {l1_par:>17.4f} | {l2_par:>17.4f}", flush=True)
    print(f"  {'Point-Adjusted F1-Score (PA-F1)':<34} | {l1_paf1:>17.4f} | {l2_paf1:>17.4f}", flush=True)
    print("  " + "-" * 72, flush=True)
    print(f"  Total Candidate Windows Routed     : {total_windows_evaluated:,} (vs {dense_windows:,} dense sliding windows)", flush=True)
    print(f"  Diffusion Evaluations Saved        : {compute_saved_pct:.2f}% ({cascade_speedup:.1f}x Cascade Speedup)", flush=True)
    print(f"  Layer 1 Candidate Windows Rejected : {total_windows_rejected:,} / {total_windows_evaluated:,} ({fa_rej_rate:.1f}% Filtered)", flush=True)
    print(f"  Root-Cause Classification Accuracy : {correct_root_cause} / {total_classified_events} ({cls_acc:.1f}% Point vs Contextual)", flush=True)
    print(f"  Total Layer 2 Execution Time       : {total_time:.2f} seconds", flush=True)
    print("=" * 78, flush=True)
    print(f"  Saved -> {os.path.join(OUT_DIR, 'layer2_results.parquet')}", flush=True)
    print(f"  Saved -> {os.path.join(OUT_DIR, 'final_benchmark_summary.csv')}", flush=True)


if __name__ == "__main__":
    run_layer2()
