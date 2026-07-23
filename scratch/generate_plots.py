import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add src to sys.path to reuse grading / data_status if needed
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "polymarket_weather"))

from grading import resolves_yes
import config

out_dir = Path(__file__).resolve().parent.parent / "src" / "polymarket_weather" / "output"
cal_csv = out_dir / "opportunities_evaluation_calibrated.csv"
ens_csv = out_dir / "opportunities_evaluation_ensemble.csv"

# Load & Grade Calibrated Markets
df_cal = pd.read_csv(cal_csv)
df_cal = df_cal.sort_values("fetched_at").groupby("condition_id").last().reset_index()
outcomes = []
for _, r in df_cal.iterrows():
    ry = resolves_yes(r["city"], r["target_date"], r["question"], r["bin_temp_c"])
    outcomes.append(None if ry is None else int(ry))
df_cal["outcome"] = outcomes
ml = df_cal.dropna(subset=["outcome"]).reset_index(drop=True)

# Load & Grade Ensemble Markets
df_ens = pd.read_csv(ens_csv)
df_ens = df_ens.sort_values("fetched_at").groupby("condition_id").last().reset_index()
outcomes_ens = []
for _, r in df_ens.iterrows():
    ry = resolves_yes(r["city"], r["target_date"], r["question"], r["bin_temp_c"])
    outcomes_ens.append(None if ry is None else int(ry))
df_ens["outcome"] = outcomes_ens
ens = df_ens.dropna(subset=["outcome"]).reset_index(drop=True)

print(f"Loaded {len(ml)} calibrated graded markets, {len(ens)} ensemble graded markets.")

# Set consistent publication style
plt.rcParams.update({
    'font.sans-serif': 'Helvetica',
    'font.family': 'sans-serif',
    'figure.dpi': 300,
    'axes.edgecolor': '#cccccc',
    'axes.linewidth': 0.8,
    'grid.color': '#eeeeee',
    'grid.linestyle': '--',
    'grid.alpha': 0.7
})

# ==========================================
# PLOT 1: Win Rate & ROI by Price Bucket
# ==========================================
fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)

buckets = ['[0.0, 0.1]', '(0.1, 0.2]', '(0.2, 0.3]', '(0.3, 0.5]', '(0.5, 1.0]']
n_markets = [82, 73, 55, 40, 14]
roi_pct = [-38.8, -12.2, -7.9, 6.5, -52.6]
# Let's calculate exact win rates per bucket from dataset
bins = [0.0, 0.1, 0.2, 0.3, 0.5, 1.0]
ml['price_bucket'] = pd.cut(ml['market_prob_raw'], bins=bins, labels=buckets, include_lowest=True)

win_rates = []
for b in buckets:
    sub = ml[ml['price_bucket'] == b]
    won = ((sub['bet_side'] == 'Yes') & (sub['outcome'] == 1)) | ((sub['bet_side'] == 'No') & (sub['outcome'] == 0))
    win_rates.append(won.mean() * 100 if len(sub) > 0 else 0)

x = np.arange(len(buckets))
bars = ax1.bar(x, roi_pct, width=0.45, color=['#d9534f' if r < 0 else '#5cb85c' for r in roi_pct], alpha=0.85, edgecolor='black', linewidth=0.5, label='ROI (%)')

ax1.set_ylabel('Realized ROI (%)', fontsize=12, fontweight='bold', color='#333333')
ax1.set_xlabel('Raw Market Price Bucket (market_prob_raw)', fontsize=12, fontweight='bold', labelpad=10)
ax1.set_xticks(x)
ax1.set_xticklabels([f"{b}\n(N={n})" for b, n in zip(buckets, n_markets)], fontsize=10)
ax1.axhline(0, color='black', linewidth=1, linestyle='-')
ax1.set_ylim(-65, 25)
ax1.grid(True, axis='y', alpha=0.5)

# Add values on top/bottom of bars
for bar in bars:
    height = bar.get_height()
    va = 'bottom' if height >= 0 else 'top'
    y_pos = height + (1 if height >= 0 else -3)
    ax1.annotate(f'{height:+.1f}%',
                 xy=(bar.get_x() + bar.get_width() / 2, y_pos),
                 xytext=(0, 0), textcoords="offset points",
                 ha='center', va=va, fontsize=10, fontweight='bold')

# Secondary axis for Win Rate
ax2 = ax1.twinx()
ax2.plot(x, win_rates, color='#1f77b4', marker='o', linewidth=2.5, markersize=8, label='Win Rate (%)')
ax2.set_ylabel('Bet Win Rate (%)', fontsize=12, fontweight='bold', color='#1f77b4')
ax2.set_ylim(0, 100)
ax2.grid(False)

for i, wr in enumerate(win_rates):
    ax2.annotate(f'{wr:.1f}%', (x[i], wr + 3), ha='center', fontsize=9, fontweight='bold', color='#1f77b4')

# Callout annotation
ax1.annotate('Only Profitable Bucket:\n(30%, 50%] +6.5% ROI', xy=(3, 6.5), xytext=(2.2, 16),
             arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=6),
             fontsize=9, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="#e6ffe6", ec="#5cb85c", lw=1.5))

ax1.annotate('Severe Tail Losses:\n-38.8% (<10%) & -52.6% (>50%)', xy=(4, -52.6), xytext=(2.8, -45),
             arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=6),
             fontsize=9, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="#ffe6e6", ec="#d9534f", lw=1.5))

plt.title('Prediction Market Bot: Realized ROI & Win Rate by Market Price Bucket (N=264)', fontsize=13, fontweight='bold', pad=15)
fig.tight_layout()
plt.savefig('scratch/win_rate_by_price_bucket.png', dpi=300)
plt.close()
print("Saved scratch/win_rate_by_price_bucket.png")

# ==========================================
# PLOT 2: Brier Score Comparison
# ==========================================
fig, ax = plt.subplots(figsize=(11, 6), dpi=300)

cities = ['Chicago', 'Hong Kong', 'London', 'NYC', 'Seoul', 'Overall']
# Brier scores by city
# Model Brier
brier_model_cities = [0.1202, 0.0970, 0.1649, 0.1564, 0.1742, 0.1546]
# Market Brier
brier_mkt_cities   = [0.0803, 0.0403, 0.1363, 0.1215, 0.1449, 0.1213]
# Ensemble Brier (overall 0.1502, city paired levels)
# Let's compute exact ensemble Brier per city on paired set
common_ids = set(ml['condition_id']) & set(ens['condition_id'])
ml_common = ml[ml['condition_id'].isin(common_ids)].set_index('condition_id')
ens_common = ens[ens['condition_id'].isin(common_ids)].set_index('condition_id')

brier_ens_cities = []
for c in ['Chicago', 'Hong Kong', 'London', 'NYC', 'Seoul']:
    sub_ens = ens_common[ens_common['city'] == c]
    if len(sub_ens) > 0:
        b_ens = sum((p - y)**2 for p, y in zip(sub_ens['forecast_prob'], sub_ens['outcome'])) / len(sub_ens)
        brier_ens_cities.append(b_ens)
    else:
        brier_ens_cities.append(np.nan)
brier_ens_cities.append(0.1502) # Overall ensemble Brier

x = np.arange(len(cities))
width = 0.25

rects1 = ax.bar(x - width, brier_model_cities, width, label='Calibrated Model (EMOS)', color='#d9534f', alpha=0.85, edgecolor='black', linewidth=0.5)
rects2 = ax.bar(x, brier_mkt_cities, width, label='Market Price (Raw)', color='#5cb85c', alpha=0.85, edgecolor='black', linewidth=0.5)
rects3 = ax.bar(x + width, brier_ens_cities, width, label='Raw NWP Ensemble', color='#428bca', alpha=0.85, edgecolor='black', linewidth=0.5)

ax.set_ylabel('Brier Score (Lower is Better)', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(cities, fontsize=11, fontweight='bold')
ax.axhline(0.1213, color='#5cb85c', linestyle='--', linewidth=1.5, label='Overall Market Baseline (0.1213)')
ax.set_ylim(0, 0.22)
ax.grid(True, axis='y', alpha=0.5)
ax.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9)

# Value annotations on Overall bars
ax.annotate('+0.0332 Deficit\n(t = +3.97, p < 0.001)', xy=(5 - width, 0.1546), xytext=(4.3, 0.185),
            arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=5),
            fontsize=9, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="#ffe6e6", ec="#d9534f"))

plt.title('Brier Score Benchmarking: Model vs Market vs Ensemble (By City & Overall)', fontsize=13, fontweight='bold', pad=15)
fig.tight_layout()
plt.savefig('scratch/brier_score_comparison.png', dpi=300)
plt.close()
print("Saved scratch/brier_score_comparison.png")

# ==========================================
# PLOT 3: Probability Gap & Brier Deficit
# ==========================================
fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)

gap_buckets = ['(0.05, 0.10]', '(0.10, 0.15]', '(0.15, 0.25]', '(0.25, 1.00]']
n_gaps = [100, 72, 64, 28]
model_brier_gaps = [0.0925, 0.1754, 0.1820, 0.2600]
mkt_brier_gaps   = [0.0892, 0.1438, 0.1396, 0.1364]
brier_deficits   = [0.0032, 0.0316, 0.0424, 0.1236]

x = np.arange(len(gap_buckets))
width = 0.35

rects1 = ax1.bar(x - width/2, model_brier_gaps, width, label='Model Brier', color='#d9534f', alpha=0.85, edgecolor='black', linewidth=0.5)
rects2 = ax1.bar(x + width/2, mkt_brier_gaps, width, label='Market Brier', color='#5cb85c', alpha=0.85, edgecolor='black', linewidth=0.5)

ax1.set_ylabel('Brier Score (Lower is Better)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Model-Market Probability Gap Size |P_model - P_market|', fontsize=12, fontweight='bold', labelpad=10)
ax1.set_xticks(x)
ax1.set_xticklabels([f"{g}\n(N={n})" for g, n in zip(gap_buckets, n_gaps)], fontsize=10)
ax1.set_ylim(0, 0.30)
ax1.grid(True, axis='y', alpha=0.5)

ax2 = ax1.twinx()
ax2.plot(x, brier_deficits, color='#cc0000', marker='s', linewidth=2.5, markersize=8, linestyle='-', label='Brier Deficit (+Δ)')
ax2.set_ylabel('Brier Deficit (Model - Market)', fontsize=12, fontweight='bold', color='#cc0000')
ax2.set_ylim(0, 0.15)
ax2.grid(False)

for i, d in enumerate(brier_deficits):
    ax2.annotate(f'+{d:.4f}', (x[i], d + 0.006), ha='center', fontsize=9, fontweight='bold', color='#cc0000')

# Callout annotation
ax1.annotate('Disagreement Paradox:\nLarge Gap (>25%) → Brier Explodes to 0.2600\n(Model error, not market mispricing)',
             xy=(3 - width/2, 0.2600), xytext=(1.2, 0.23),
             arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=6),
             fontsize=9, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="#fff0f0", ec="#cc0000", lw=1.5))

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=True, facecolor='white', framealpha=0.9)

plt.title('The Disagreement Paradox: Brier Score Deficit vs Model-Market Probability Gap', fontsize=13, fontweight='bold', pad=15)
fig.tight_layout()
plt.savefig('scratch/probability_gap_brier_deficit.png', dpi=300)
plt.close()
print("Saved scratch/probability_gap_brier_deficit.png")

# ==========================================
# PLOT 4: Reliability Diagram (Calibration Curve)
# ==========================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), dpi=300, gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

# Generate calibration data using _calibration from evaluate_oos logic
def get_cal_curve(probs, outcomes, n_bins=10):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    mean_preds = []
    realized_freqs = []
    counts = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i+1]
        mask = (probs >= lo) & (probs < hi) if i < n_bins - 1 else (probs >= lo) & (probs <= hi)
        sub_p = probs[mask]
        sub_y = outcomes[mask]
        counts.append(len(sub_p))
        if len(sub_p) > 0:
            mean_preds.append(sub_p.mean())
            realized_freqs.append(sub_y.mean())
        else:
            mean_preds.append(bin_centers[i])
            realized_freqs.append(np.nan)
    return bin_centers, np.array(mean_preds), np.array(realized_freqs), np.array(counts)

p_model = ml['forecast_prob'].values
p_mkt = ml['market_prob_raw'].values
y_true = ml['outcome'].values

bc_mod, mp_mod, rf_mod, cnt_mod = get_cal_curve(p_model, y_true, 10)
bc_mkt, mp_mkt, rf_mkt, cnt_mkt = get_cal_curve(p_mkt, y_true, 10)

# Diagonal reference
ax1.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Perfect Calibration (y = x)')

# Plot Model and Market curves
ax1.plot(mp_mod, rf_mod, 'o-', color='#d9534f', linewidth=2.5, markersize=7, label='Calibrated Model (EMOS)')
ax1.plot(mp_mkt, rf_mkt, 's-', color='#5cb85c', linewidth=2.5, markersize=7, label='Market Price (Raw)')

ax1.set_ylabel('Realized YES Frequency', fontsize=12, fontweight='bold')
ax1.set_ylim(-0.02, 1.02)
ax1.set_xlim(-0.02, 1.02)
ax1.grid(True, alpha=0.5)
ax1.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9)

# Annotations on low probability bin
# [0, 0.1): n=98, pred=0.037, realized=0.163
ax1.annotate('Low Probability Under-Confidence:\nPred 3.7% vs Realized 16.3% (n=98)',
             xy=(0.037, 0.163), xytext=(0.18, 0.30),
             arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=6),
             fontsize=9, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="#ffe6e6", ec="#d9534f", lw=1.5))

ax1.set_title('Reliability Diagram: Model vs Market Calibration Curves (N=264)', fontsize=13, fontweight='bold', pad=15)

# Subplot: Sample distribution counts
width = 0.035
ax2.bar(bc_mod - width/2, cnt_mod, width=width, color='#d9534f', alpha=0.7, label='Model Predictions')
ax2.bar(bc_mkt + width/2, cnt_mkt, width=width, color='#5cb85c', alpha=0.7, label='Market Predictions')
ax2.set_xlabel('Forecast Probability Bin', fontsize=12, fontweight='bold')
ax2.set_ylabel('Sample Count (N)', fontsize=10, fontweight='bold')
ax2.grid(True, alpha=0.5)
ax2.legend(loc='upper right', fontsize=8, frameon=True)

fig.tight_layout()
plt.savefig('scratch/reliability_diagram.png', dpi=300)
plt.close()
print("Saved scratch/reliability_diagram.png")

print("All 4 plots generated successfully!")
