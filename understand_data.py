"""
Data Understanding Script
--------------------------
Shows you exactly what the NASA SMAP data is,
what the anomalies look like, and what gap injection did.
Generates 4 clear plots saved as PNG files you can open.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # save to file (no display window needed)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import ast

# ── Load the raw original data for P-1 ──────────────────────────────────────
meta     = pd.read_csv('nasa_smap_data/labeled_anomalies.csv')
p1_meta  = meta[meta['chan_id'] == 'P-1'].iloc[0]

raw_test  = np.load('nasa_smap_data/data/data/test/P-1.npy')   # (8505, 25)
raw_train = np.load('nasa_smap_data/data/data/train/P-1.npy')  # (2872, 25)

# Build ground truth anomaly mask
sequences = ast.literal_eval(p1_meta['anomaly_sequences'])
mask = np.zeros(len(raw_test), dtype=int)
for s, e in sequences:
    mask[s:e+1] = 1

# Build full original DataFrame
df_orig = pd.DataFrame(raw_test, columns=[f'feat_{i:02d}' for i in range(25)])
df_orig['Is_Anomaly'] = mask
df_orig['row']        = df_orig.index

# Load the irregular version
df_irr = pd.read_parquet('nasa_smap_data/all_smap_irregular.parquet')
df_p1_irr = df_irr[df_irr['chan_id'] == 'P-1'].reset_index()

# ── Print a plain-English data walkthrough ───────────────────────────────────
print("=" * 65)
print("  WHAT IS THIS DATA?")
print("=" * 65)
print("""
The NASA SMAP satellite monitors itself while orbiting Earth.
It continuously measures things like:
  - how much current its solar panels are producing
  - battery voltage levels
  - temperature of components
  - whether certain commands executed
  - attitude / orientation readings

Every minute, it logs 25 of these measurements.
That one row of 25 numbers = one snapshot of the satellite's health.

The dataset gives us 8,505 such snapshots for the P-1 sensor channel.
That's about 5.9 days of data.

NASA engineers later reviewed these snapshots and marked:
  - rows 2149 to 2349: something went wrong here
  - rows 3539 to 3779: something went wrong here
  - rows 4536 to 4844: something went wrong here

We don't know the exact physical cause (NASA keeps that proprietary),
but these are REAL anomalies that were investigated.
""")

print("=" * 65)
print("  WHAT DO THE 25 FEATURES ACTUALLY LOOK LIKE?")
print("=" * 65)

# Show a few rows clearly
sample = df_orig[['row','feat_00','feat_01','feat_05','feat_06','Is_Anomaly']].iloc[2144:2155]
print("\nActual sensor readings around the first anomaly (rows 2144-2154):")
print("  feat_00 = primary monitored sensor")
print("  feat_01 = binary flag (command state: 0 or 1)")
print("  feat_05 = correlated sensor (r=0.60 with feat_00)")
print("  feat_06 = most correlated sensor (r=0.67 with feat_00)")
print()
print(f"  {'row':>5}  {'feat_00':>10}  {'feat_01':>8}  {'feat_05':>8}  {'feat_06':>8}  {'ANOMALY':>8}")
print(f"  {'---':>5}  {'-------':>10}  {'-------':>8}  {'-------':>8}  {'-------':>8}  {'-------':>8}")
for _, r in sample.iterrows():
    flag = "  <-- ANOMALY STARTS" if r['row'] == 2149 else ""
    marker = "***" if r['Is_Anomaly'] == 1 else "   "
    print(f"  {int(r['row']):>5}  {r['feat_00']:>10.4f}  {r['feat_01']:>8.1f}  "
          f"{r['feat_05']:>8.1f}  {r['feat_06']:>8.1f}  {marker}{flag}")

print("""
Notice:
  - feat_00 doesn't spike dramatically (this is a CONTEXTUAL anomaly)
  - feat_01 stays 0 (binary flag unchanged)
  - feat_05 and feat_06 flip from 0 to 1 at the anomaly start
  - the anomaly is visible in the COMBINATION of sensors, not one spike
""")

print("=" * 65)
print("  WHAT DID GAP INJECTION DO?")
print("=" * 65)
print(f"""
Original P-1 data:    8,505 rows  (every row = 1 minute apart, uniform)
After gap injection:    876 rows  (only 10.3% of rows kept)

What we did:
  - Kept rows 0-9     (burst: 10 rows observed)
  - DELETED rows 10-157  (gap: 148 rows = unobserved time)
  - Kept rows 158-175  (next burst)
  - DELETED rows 176-324
  ... and so on

The DELETED rows are NOT fabricated. They're just "what the satellite
sent but no ground station was in range to receive."
The surviving rows still have their REAL sensor values.

The delta_t column tells you: "how long since the last observation?"
  delta_t = 1   means: this row is 1 unit after the previous row (normal)
  delta_t = 149 means: 149 units of time passed (big gap just ended)
""")

print(f"  First 15 rows of irregular P-1 data:")
print(f"  {'row_idx':>7}  {'delta_t':>8}  {'feat_00':>10}  {'feat_05':>8}  {'Is_Anomaly':>10}")
print(f"  {'-------':>7}  {'-------':>8}  {'-------':>10}  {'-------':>8}  {'----------':>10}")
for _, r in df_p1_irr.head(15).iterrows():
    gap_note = "  << GAP ENDED" if r['delta_t'] > 10 else ""
    print(f"  {int(r['original_row_idx']):>7}  {r['delta_t']:>8.0f}  "
          f"{r['feat_00']:>10.4f}  {r['feat_05']:>8.1f}  "
          f"{'YES' if r['Is_Anomaly'] else 'no':>10}{gap_note}")

# ── PLOT 1: Full original signal with anomaly windows ───────────────────────
print("\n[PLOTTING] Generating 4 plots...")

fig, axes = plt.subplots(4, 1, figsize=(16, 18))
fig.patch.set_facecolor('#0f1117')

colors = {'bg': '#0f1117', 'normal': '#4fc3f7', 'anomaly': '#ff5252',
          'gap': '#ffd54f', 'grid': '#2a2d3e', 'text': '#e0e0e0'}

# ── PLOT 1: Original full signal ─────────────────────────────────────────────
ax = axes[0]
ax.set_facecolor(colors['bg'])
rows = df_orig['row'].values
vals = df_orig['feat_00'].values
anom = df_orig['Is_Anomaly'].values

# Plot normal regions
ax.plot(rows, vals, color=colors['normal'], linewidth=0.8, alpha=0.9, label='Normal telemetry')

# Shade anomaly windows
for s, e in sequences:
    ax.axvspan(s, e, color=colors['anomaly'], alpha=0.25, label='Anomaly window')
    ax.plot(rows[s:e+1], vals[s:e+1], color=colors['anomaly'], linewidth=1.2)

ax.set_title('PLOT 1: Full Original P-1 Telemetry (8,505 rows — raw NASA data, uniform 1-unit spacing)',
             color=colors['text'], fontsize=11, pad=10)
ax.set_xlabel('Row index (time steps)', color=colors['text'])
ax.set_ylabel('feat_00 sensor value\n(NASA normalized [-1, +1])', color=colors['text'])
ax.tick_params(colors=colors['text'])
ax.grid(color=colors['grid'], linewidth=0.4, alpha=0.6)
for spine in ax.spines.values():
    spine.set_edgecolor(colors['grid'])

# Add anomaly window labels
for i, (s, e) in enumerate(sequences):
    ax.annotate(f'Anomaly {i+1}\nrows {s}-{e}', xy=((s+e)/2, ax.get_ylim()[1]*0.85),
                color=colors['anomaly'], fontsize=8, ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=colors['bg'], edgecolor=colors['anomaly']))

handles = [mpatches.Patch(color=colors['normal'], label='Normal'),
           mpatches.Patch(color=colors['anomaly'], alpha=0.5, label='Anomaly window')]
ax.legend(handles=handles, facecolor=colors['bg'], labelcolor=colors['text'], fontsize=9)

# ── PLOT 2: Zoom into first anomaly transition ────────────────────────────────
ax = axes[1]
ax.set_facecolor(colors['bg'])
zoom_start, zoom_end = 2100, 2420
z_rows = rows[zoom_start:zoom_end]
z_vals = vals[zoom_start:zoom_end]
z_anom = anom[zoom_start:zoom_end]

for i in range(len(z_rows)-1):
    color = colors['anomaly'] if z_anom[i] else colors['normal']
    ax.plot(z_rows[i:i+2], z_vals[i:i+2], color=color, linewidth=1.5)

ax.axvline(x=2149, color=colors['anomaly'], linestyle='--', linewidth=1.5, alpha=0.8)
ax.axvline(x=2349, color=colors['anomaly'], linestyle='--', linewidth=1.5, alpha=0.8)
ax.axvspan(2149, 2349, color=colors['anomaly'], alpha=0.1)

ax.annotate('Anomaly starts\nrow 2149', xy=(2149, ax.get_ylim()[0]),
            xytext=(2170, -0.4), color=colors['anomaly'], fontsize=9,
            arrowprops=dict(arrowstyle='->', color=colors['anomaly']))

# Also show feat_05 and feat_06
feat05 = raw_test[zoom_start:zoom_end, 5]
feat06 = raw_test[zoom_start:zoom_end, 6]
ax.plot(z_rows, feat05 * 0.3 - 0.8, color='#a5d6a7', linewidth=1.0,
        linestyle=':', alpha=0.7, label='feat_05 (scaled)')
ax.plot(z_rows, feat06 * 0.3 - 0.9, color='#ce93d8', linewidth=1.0,
        linestyle=':', alpha=0.7, label='feat_06 (scaled)')

ax.set_title('PLOT 2: Zoom into Anomaly Window 1 (rows 2100-2420) — Notice feat_05 & feat_06 flip at anomaly start',
             color=colors['text'], fontsize=11, pad=10)
ax.set_xlabel('Row index', color=colors['text'])
ax.set_ylabel('Sensor value', color=colors['text'])
ax.tick_params(colors=colors['text'])
ax.grid(color=colors['grid'], linewidth=0.4, alpha=0.6)
for spine in ax.spines.values():
    spine.set_edgecolor(colors['grid'])
ax.legend(facecolor=colors['bg'], labelcolor=colors['text'], fontsize=9)

# ── PLOT 3: Original vs Irregular (gap effect) ────────────────────────────────
ax = axes[2]
ax.set_facecolor(colors['bg'])

irr_rows = df_p1_irr['original_row_idx'].values
irr_vals = df_p1_irr['feat_00'].values
irr_anom = df_p1_irr['Is_Anomaly'].values
irr_dt   = df_p1_irr['delta_t'].values

# Original (faint)
ax.plot(rows, vals, color=colors['normal'], linewidth=0.4, alpha=0.2, label='Original (all 8505 rows)')
# Irregular surviving points
normal_mask_irr = irr_anom == 0
anom_mask_irr   = irr_anom == 1
ax.scatter(irr_rows[normal_mask_irr], irr_vals[normal_mask_irr],
           color=colors['normal'], s=6, alpha=0.7, label='Survived (normal)', zorder=3)
ax.scatter(irr_rows[anom_mask_irr], irr_vals[anom_mask_irr],
           color=colors['anomaly'], s=12, alpha=0.9, label='Survived (anomaly)', zorder=4)

# Mark gap positions with vertical ticks at bottom
gap_positions = irr_rows[irr_dt > 10]
ax.vlines(gap_positions, ymin=-1.02, ymax=-0.95, color=colors['gap'],
          linewidth=0.8, alpha=0.7, label='Gap boundary (delta_t>10)')

for s, e in sequences:
    ax.axvspan(s, e, color=colors['anomaly'], alpha=0.08)

ax.set_title('PLOT 3: Gap Injection Effect — Only 876 of 8,505 rows survive (dots = observed, yellow ticks = gap boundaries)',
             color=colors['text'], fontsize=11, pad=10)
ax.set_xlabel('Original row index', color=colors['text'])
ax.set_ylabel('feat_00 sensor value', color=colors['text'])
ax.tick_params(colors=colors['text'])
ax.grid(color=colors['grid'], linewidth=0.4, alpha=0.6)
for spine in ax.spines.values():
    spine.set_edgecolor(colors['grid'])
ax.legend(facecolor=colors['bg'], labelcolor=colors['text'], fontsize=9, markerscale=2)

# ── PLOT 4: delta_t distribution + what it means ─────────────────────────────
ax = axes[3]
ax.set_facecolor(colors['bg'])

dt_vals = df_p1_irr['delta_t'].values
bins = [0, 1.5, 10, 100, 155]
labels_bins = ['delta_t=1\n(consecutive)', 'delta_t 2-10\n(micro-gap)', 'delta_t 11-100\n(medium gap)', 'delta_t 149-150\n(major gap)']
counts = [
    (dt_vals == 1).sum(),
    ((dt_vals > 1) & (dt_vals <= 10)).sum(),
    ((dt_vals > 10) & (dt_vals <= 100)).sum(),
    (dt_vals > 100).sum(),
]
bar_colors = [colors['normal'], '#fff176', '#ffb74d', colors['anomaly']]

bars = ax.bar(labels_bins, counts, color=bar_colors, width=0.5, edgecolor=colors['bg'], linewidth=2)
for bar, count in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            str(count), ha='center', va='bottom', color=colors['text'], fontsize=11, fontweight='bold')

ax.set_title('PLOT 4: Distribution of delta_t in Irregular P-1 Data — '
             'Shows how often gaps occur and their sizes',
             color=colors['text'], fontsize=11, pad=10)
ax.set_ylabel('Number of rows', color=colors['text'])
ax.set_xlabel('Time gap type', color=colors['text'])
ax.tick_params(colors=colors['text'])
ax.grid(color=colors['grid'], linewidth=0.4, alpha=0.6, axis='y')
for spine in ax.spines.values():
    spine.set_edgecolor(colors['grid'])

# Add explanation text inside the plot
ax.text(0.98, 0.85,
        "delta_t tells Layer 2:\n"
        "\"How long was the satellite\n"
        "invisible before this reading?\"\n\n"
        "delta_t=1   → normal burst\n"
        "delta_t=149 → contact window gap\n"
        "              (149 minutes of data\n"
        "               we never received)",
        transform=ax.transAxes, ha='right', va='top',
        color=colors['text'], fontsize=9,
        bbox=dict(boxstyle='round', facecolor=colors['grid'], edgecolor=colors['normal'], alpha=0.8))

plt.tight_layout(pad=2.5)
plt.savefig('data_understanding.png', dpi=150, bbox_inches='tight', facecolor=colors['bg'])
print("[SAVED]  data_understanding.png  <-- open this file")
print()
print("=" * 65)
print("  SUMMARY: WHAT YOU HAVE")
print("=" * 65)
print(f"""
1. RAW NASA DATA (original, untouched):
   File: nasa_smap_data/data/data/test/P-1.npy
   Shape: 8,505 rows x 25 sensor readings
   - Every row = 1 snapshot of satellite health (every minute)
   - feat_00 = the sensor being monitored
   - feat_01-24 = other sensors that provide CONTEXT
   - NASA labeled rows 2149-2349, 3539-3779, 4536-4844 as anomalies

2. INJECTED (IRREGULAR) VERSION:
   File: nasa_smap_data/all_smap_irregular.parquet
   Shape: 876 rows for P-1 (10.3% of original)
   - Same real sensor values — nothing changed or invented
   - 149 rows simply "deleted" in gaps (simulating satellite out of range)
   - delta_t column added: tells you how long since last observation
   - Is_Anomaly labels preserved for whatever rows survived

3. THE CHALLENGE:
   - 80 of the original 751 anomaly rows survived the gaps
   - The model must find those 80 anomaly rows among 796 normal rows
   - AND it must handle the fact that consecutive rows may be
     149 units apart (not 1 unit) without knowing what happened in between
""")
