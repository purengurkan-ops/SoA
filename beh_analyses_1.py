# =============================================================================
# BEHAVIORAL ANALYSES: CONTROL DETECTION & MEMORY
# =============================================================================

# --- Packages ---
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import norm
import polars as pl
from pymer4.models import glmer, lmer
import pingouin as pg

# =============================================================================
# PHASE 1: LOAD ALL DATA
# =============================================================================
print("\n" + "="*40)
print("PHASE 1: LOADING DATA")
print("="*40)

DATA_DIR = Path(r"/Users/purengurkan/Desktop/SoA/SoA/CDmem/data/main_Data")
OUTPUT_DIR = Path("analysis_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PARTICIPANT_FILTER = [4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22]

# 1A. Load Main Task Data
all_files = list(DATA_DIR.glob("CDmem_1_*.csv"))
df_list = []
for file_path in all_files:
    df_list.append(pd.read_csv(file_path))

if PARTICIPANT_FILTER:
    df_list = [df for df in df_list if df["participant"].iloc[0] in PARTICIPANT_FILTER]

if df_list:
    data = pd.concat(df_list, ignore_index=True)
    print(f"Main data loaded! Total rows: {len(data)}")
else:
    raise ValueError("Warning: No main data files found.")

# 1B. Load Recognition Data
recog_files = list(DATA_DIR.glob("CDmem_*_recognition.csv"))
recog_list = []
for file_path in recog_files:
    recog_list.append(pd.read_csv(file_path))

if recog_list:
    recog_data = pd.concat(recog_list, ignore_index=True)
    print(f"Recognition data loaded! Total rows: {len(recog_data)}")
else:
    raise ValueError("Warning: No recognition data files found.")

# --- DEMOGRAPHICS CHECK ---
participant_df = data.drop_duplicates(subset=['participant'])

print("\n--- DEMOGRAPHICS ---")
print("Total Participants:", len(participant_df))

if 'age' in participant_df.columns and 'gender' in participant_df.columns:
    print("Mean Age:", participant_df['age'].mean())
    print("Age SD:", participant_df['age'].std())
    print("Gender Distribution:\n", participant_df['gender'].value_counts())
else:
    print("Warning: 'age' and 'gender' columns not found in the dataset.")

main_pts = set(data["participant"].unique())
recog_pts = set(recog_data["participant"].unique())
missing = main_pts - recog_pts
print("Missing from recognition:", missing) 

from pathlib import Path
DATA_DIR = Path(r"/Users/purengurkan/Desktop/SoA/SoA/CDmem/data/main_Data")
print("Recognition files the glob found:")
for f in sorted(DATA_DIR.glob("CDmem_*_recognition.csv")):
    print("  ", f.name)

print("\nEverything in the folder with 'recog' in the name:")
for f in sorted(DATA_DIR.iterdir()):
    if "recog" in f.name.lower():
        print("  ", f.name)


# =============================================================================
# PHASE 2: DATA CLEANING & EXCLUSIONS
# =============================================================================
print("\n" + "="*40)
print("PHASE 2: EXCLUSION CRITERIA")
print("="*40)

# --- Excl 1: Timeout Rate (Main Data) ---
TIMEOUT_THRESHOLD = 0.50
test_data = data[data["phase"] == "test"].copy()
test_data["is_timeout"] = test_data["is_timeout"].astype(str).str.strip().str.lower() == "true"
timeout_rate = test_data.groupby(["participant", "control_condition"])["is_timeout"].mean().reset_index()
failed_rows = timeout_rate[timeout_rate["is_timeout"] >= TIMEOUT_THRESHOLD]
excluded_timeout_pts = failed_rows["participant"].unique().tolist()
print(f"Criterion 1 (Timeout > 50%): Excluded {len(excluded_timeout_pts)} participants {excluded_timeout_pts}")
data = data[~data["participant"].isin(excluded_timeout_pts)].copy()

# --- Excl 2: Detection Accuracy 2.5 SD Outliers (Main Data) ---
pt_acc_by_cond = data.groupby(['participant', 'control_condition'])['detection_accuracy'].mean().reset_index()
group_stats = pt_acc_by_cond.groupby('control_condition')['detection_accuracy'].agg(['mean', 'std']).reset_index()
pt_acc_merged = pt_acc_by_cond.merge(group_stats, on='control_condition', suffixes=('_pt', '_group'))
pt_acc_merged['is_outlier'] = abs(pt_acc_merged['detection_accuracy'] - pt_acc_merged['mean']) > (2.5 * pt_acc_merged['std'])
excluded_acc_pts = pt_acc_merged[pt_acc_merged['is_outlier']]['participant'].unique().tolist()
data = data[~data['participant'].isin(excluded_acc_pts)]
print(f"Criterion 2 (Det Acc > 2.5 SD per condition): Excluded {len(excluded_acc_pts)} participants {excluded_acc_pts}")

# --- Excl 3: Calibration Failure (Numerical SD Check) ---
calib_data = data[data["phase"] == "calibration"].copy()
calib_data['quest_alpha_sd'] = pd.to_numeric(calib_data['quest_alpha_sd'], errors='coerce')
excluded_calib_pts = []
px_convergence = {}  # To track how many staircases converged per participant to inform future analyses (Excl 5)

for px, px_df in calib_data.groupby("participant"):
    n_converged = 0
    for target, target_df in px_df.groupby("calib_target"):
        if not target_df.empty:
            final_sd = target_df["quest_alpha_sd"].iloc[-1]
            if not pd.isna(final_sd) and final_sd < 0.20:
                n_converged += 1

    px_convergence[px] = n_converged
    if n_converged == 0:  # If both staircases failed to converge, exclude the participant
        excluded_calib_pts.append(px)

print(f"Criterion 3 (Calibration Fail): Excluded {len(excluded_calib_pts)} participants {excluded_calib_pts}")
data = data[~data["participant"].isin(excluded_calib_pts)].copy()

# --- RT Outlier Trimming (Recog Data) ---
participant_mean_rt = recog_data.groupby("participant")["mem_rt"].transform("mean")
participant_sd_rt = recog_data.groupby("participant")["mem_rt"].transform("std")
valid_trials_mask = (recog_data["mem_rt"] >= (participant_mean_rt - 3 * participant_sd_rt)) & \
                    (recog_data["mem_rt"] <= (participant_mean_rt + 3 * participant_sd_rt))
clean_recog_data = recog_data[valid_trials_mask].copy()
print(f"RT Trimming: Removed {len(recog_data) - len(clean_recog_data)} trials outside of 3SD.")

print(f"\nFINAL CLEAN SAMPLE: {data['participant'].nunique()} Participants")

# --- Excl 5: Individual Manipulation Failure
from scipy.stats import ttest_ind

test_data_manip = data[data["phase"] == "test"].copy()
test_data_manip["detection_accuracy"] = pd.to_numeric(test_data_manip["detection_accuracy"], errors="coerce")
test_data_manip["agency_rating"] = pd.to_numeric(test_data_manip["agency_rating"], errors="coerce")

excluded_manip = []

for px in sorted(data["participant"].unique()):
    px_test = test_data_manip[test_data_manip["participant"] == px]
    high = px_test[px_test["control_condition"] == "high"]
    low = px_test[px_test["control_condition"] == "low"]

    if len(high) == 0 or len(low) == 0:
        continue

    acc_diff = high["detection_accuracy"].mean() - low["detection_accuracy"].mean()
    ag_diff = high["agency_rating"].mean() - low["agency_rating"].mean()
    both_converged = px_convergence.get(px, 0) == 2

    # (A) If both calibration staircases failed to converge and the differences are in the wrong direction, exclude
    if acc_diff <= 0 and ag_diff <= 0:
        excluded_manip.append(px)
        continue

    # (B) If both calibration staircases converged and the differences are in the correct direction, include
    if both_converged:
        continue

    # (C) If only one calibration staircase converged, use t-test to check
    t_acc, p_acc = ttest_ind(high["detection_accuracy"].dropna(), low["detection_accuracy"].dropna())
    t_ag, p_ag = ttest_ind(high["agency_rating"].dropna(), low["agency_rating"].dropna())

    acc_ok = acc_diff > 0 and p_acc < 0.05
    ag_ok = ag_diff > 0 and p_ag < 0.05

    # If either t-test is not significant or the difference is in the wrong direction, exclude
    if not acc_ok or not ag_ok:
        excluded_manip.append(px)

print(f"Criterion 5 (Manipulation Fail): Excluded {len(excluded_manip)} participants {excluded_manip}")
data = data[~data["participant"].isin(excluded_manip)].copy()

# --- FIX: keep recognition/memory data in sync with the final analyzed sample ---
# `clean_recog_data` was built from `recog_data`, which was never subjected to the
# participant-level exclusions above (timeout, accuracy outliers, calibration
# failure, manipulation failure). Without this step, excluded participants'
# recognition trials still flow into `model_data` in Phase 4, and because their
# images have no match in the (already-filtered) `data` lookup table, those trials
# get assigned control_condition = NaN, which downstream gets silently contrast-
# coded as low control (-0.5). This line removes that leak.
final_pts = data["participant"].unique()
clean_recog_data = clean_recog_data[clean_recog_data["participant"].isin(final_pts)].copy()
print(f"Recognition data restricted to final sample: {clean_recog_data['participant'].nunique()} participants, "
      f"{len(clean_recog_data)} trials remain.")


# =============================================================================
# PHASE 3: MANIPULATION CHECKS
# =============================================================================
print("\n" + "="*40)
print("PHASE 3: SANITY / MANIPULATION CHECKS")
print("="*40)

mani_data = data[(data["phase"] == "test") & (data['control_condition'].isin(['high', 'low']))].copy()
pt_means = mani_data.groupby(['participant', 'control_condition'])[['detection_accuracy', 'agency_rating']].mean().reset_index()
manipulation_summary = pt_means.groupby('control_condition')[['detection_accuracy', 'agency_rating']].mean().reset_index()
manipulation_summary['detection_accuracy'] = manipulation_summary['detection_accuracy'] * 100

print("\nMANIPULATION CHECK SUMMARY")
print("-" * 40)
print(manipulation_summary.round(2).to_string(index=False))

# Pivot to wide format for paired t-tests
wide_pt_means = pt_means.pivot(index='participant', columns='control_condition', values=['agency_rating', 'detection_accuracy']).dropna()

# Means and SDs per condition (participant-level)
desc = wide_pt_means.agg(['mean', 'std'])
desc[('detection_accuracy', 'high')] *= 100
desc[('detection_accuracy', 'low')]  *= 100
print("\n--- MEANS AND SDs BY CONDITION ---")
print(desc.round(2).to_string())

# SD of the paired difference (useful for the effect-size sentence)
for var in ['detection_accuracy', 'agency_rating']:
    diff = wide_pt_means[(var, 'high')] - wide_pt_means[(var, 'low')]
    print(f"{var}: M_diff = {diff.mean():.3f}, SD_diff = {diff.std(ddof=1):.3f}")

print("\n--- AGENCY RATING T-TEST (High vs. Low) ---")
agency_ttest = pg.ttest(wide_pt_means[('agency_rating', 'high')], wide_pt_means[('agency_rating', 'low')], paired=True)
print(agency_ttest.to_string())

print("\n--- DETECTION ACCURACY T-TEST (High vs. Low) ---")
acc_ttest = pg.ttest(wide_pt_means[('detection_accuracy', 'high')], wide_pt_means[('detection_accuracy', 'low')], paired=True)
print(acc_ttest.to_string())


# =============================================================================
# PHASE 4: VARIABLE DERIVATION (REVERTED TO SEEN vs UNSEEN)
# =============================================================================
print("\n" + "="*40)
print("PHASE 4: VARIABLE DERIVATION")
print("="*40)

# Veriyi küçük/büyük harf veya boşluk hatalarından temizleyelim
clean_recog_data["controlled"] = clean_recog_data["controlled"].astype(str).str.strip().str.lower()

# --- 1. Foolproof Lookup: Resimleri kontrol ve tespit verisiyle eşleştir ---
targets = clean_recog_data[clean_recog_data['mem_ground_truth'] == 'seen'].copy()

lookup_A = data[['participant', 'img_A_name', 'control_condition', 'detection_accuracy']].rename(columns={'img_A_name': 'mem_filename'})
lookup_B = data[['participant', 'img_B_name', 'control_condition', 'detection_accuracy']].rename(columns={'img_B_name': 'mem_filename'})
img_lookup = pd.concat([lookup_A, lookup_B], ignore_index=True).drop_duplicates(subset=['participant', 'mem_filename'])

if 'control_condition' in targets.columns:
    targets = targets.drop(columns=['control_condition'])

# Güvenli eşleştirme (Inner merge)
targets = targets.merge(img_lookup, on=['participant', 'mem_filename'], how='inner')

# ITEM_TYPE = SEEN
targets['item_type'] = 'seen'

# --- 2. Assign Foils (Dummy Conditions) ---
foils = clean_recog_data[clean_recog_data['mem_ground_truth'] == 'unseen'].copy()
foils = foils.sort_values(by=['participant', 'mem_filename'])
row_numbers = foils.groupby('participant').cumcount()
total_foils = foils.groupby('participant')['mem_ground_truth'].transform('count')

foils['control_condition'] = np.where(row_numbers < (total_foils / 2), 'high', 'low')

# ITEM_TYPE = UNSEEN
foils['item_type'] = 'unseen'

np.random.seed(42)
foils['detection_accuracy'] = np.random.choice([1.0, 0.0], size=len(foils))

# --- 3. Combine & Contrast Code ---
model_data = pd.concat([targets, foils], ignore_index=True)
model_data['said_old_int'] = model_data['mem_response'].str.lower().map({'yes': 1, 'no': 0})
model_data['log_mem_rt'] = np.log(model_data['mem_rt'])

# PREDICTORS (SEEN vs UNSEEN Mantığı)
model_data['control_c'] = np.where(model_data['control_condition'] == 'high', 0.5, -0.5)
model_data['item_type_c'] = np.where(model_data['item_type'] == 'seen', 0.5, -0.5)
model_data['detection_accuracy_c'] = np.where(model_data['detection_accuracy'] == 1, 0.5, 
                                     np.where(model_data['detection_accuracy'] == 0, -0.5, 0))

# --- 4. Compute d-prime (Sadece High vs Low olarak havuzlandı) ---
def compute_d_prime(hits, misses, false_alarms, correct_rejections):
    hit_rate = (hits + 0.5) / (hits + misses + 1)
    fa_rate = (false_alarms + 0.5) / (false_alarms + correct_rejections + 1)
    return norm.ppf(hit_rate) - norm.ppf(fa_rate)

rows = []
for participant, px_data in model_data.groupby("participant"):
    
    # False Alarm (Tüm Foil'ler kullanılarak)
    fa_data = px_data[px_data["mem_ground_truth"] == "unseen"]
    fa = (fa_data["said_old_int"] == 1).sum()
    cr = (fa_data["said_old_int"] == 0).sum()
    
    # Hits (Control Condition'a göre ikiye ayrılmış Target'lar)
    seen_data = px_data[px_data["mem_ground_truth"] == "seen"]
    for condition, cond_data in seen_data.groupby("control_condition"):
        if condition not in {"high", "low"}: continue
        
        hits = (cond_data["said_old_int"] == 1).sum()
        misses = (cond_data["said_old_int"] == 0).sum()
        
        rows.append({
            "participant": participant, 
            "control_condition": condition, 
            "hits": hits, 
            "misses": misses,
            "false_alarms": fa, 
            "correct_rejections": cr, 
            "d_prime": compute_d_prime(hits, misses, fa, cr)
        })

results_df = pd.DataFrame(rows)

if not results_df.empty:
    results_df['hit_rate'] = results_df['hits'] / (results_df['hits'] + results_df['misses'])
    results_df.to_csv(OUTPUT_DIR / "dprime_by_condition.csv", index=False)
else:
    print("WARNING: results_df is empty, matching problem.")


# =============================================================================
# PHASE 5: STATISTICAL MODELS
# =============================================================================
print("\n" + "="*40)
print("PHASE 5: STATISTICAL MODELS")
print("="*40)

model_data_pl = pl.DataFrame(model_data)

print("\n--- RQ1 Analysis A: Binomial GLMM (said_old ~ item_type * control) ---")
# RQ1: (Seen + Unseen)
rq1_bin = glmer("said_old_int ~ item_type_c * control_c + (1 | participant)", data=model_data_pl, family="binomial")
rq1_bin.fit()
print(rq1_bin.result_fit)

print("\n--- RQ1 Analysis B: Gaussian LMM (log_RT ~ item_type * control) ---")
rq1_rt = lmer("log_mem_rt ~ item_type_c * control_c + (1 | participant)", data=model_data_pl)
rq1_rt.fit()
print(rq1_rt.result_fit)


print("\n--- RQ2 Analysis C: Binomial GLMM (Target Items Only) ---")
# RQ2: Sadece Ekranda Olanlar (Seen). Çünkü Foil'lerin detection'ı dummy-coded.
targets_only = model_data[model_data['item_type'] == 'seen'].copy()
targets_only_pl = pl.DataFrame(targets_only)

rq2_formula_bin = "said_old_int ~ detection_accuracy_c * control_c + (1 | participant)"
rq2_bin = glmer(rq2_formula_bin, data=targets_only_pl, family="binomial")
rq2_bin.fit()
print(rq2_bin.result_fit)

print("\n--- RQ2 Analysis D: Gaussian LMM RT (Target Items Only) ---")
rq2_formula_rt = "log_mem_rt ~ detection_accuracy_c * control_c + (1 | participant)"
rq2_rt = lmer(rq2_formula_rt, data=targets_only_pl)
rq2_rt.fit()
print(rq2_rt.result_fit)


# =============================================================================
# PHASE 6: GENERATING PLOTS (FINAL CLEAN VERSION - NO BAR PLOTS)
# =============================================================================
print("\n" + "="*40)
print("PHASE 6: GENERATING PLOTS")
print("="*40)

import warnings
warnings.filterwarnings("ignore", message="Setting a gradient palette using color=")

sns.set_theme(style="whitegrid")
PLOT_DIR = OUTPUT_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

COND_COLORS = {'high': '#3a76af', 'low': '#e38041'}
BOX_COLORS = {'Detected': '#98df8a', 'Undetected': '#ff9896'}
SWARM_COLORS = {'Detected': '#2ca02c', 'Undetected': '#d62728'}
COND_ORDER = ['high', 'low']

unique_pts = model_data['participant'].unique()
pt_palette = sns.color_palette("tab10", n_colors=len(unique_pts))
PT_COLORS = {pt: color for pt, color in zip(unique_pts, pt_palette)}

# H1 Plot Function
def plot_h1_metric(df, y_col, ylabel, title, filename):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=16)

    # Box + Swarm Plot
    sns.boxplot(data=df, x='control_condition', y=y_col, order=COND_ORDER, width=0.4, palette=COND_COLORS, hue='control_condition', fliersize=0, ax=axes[0], legend=False)
    sns.swarmplot(data=df, x='control_condition', y=y_col, order=COND_ORDER, color='black', alpha=0.6, size=5, ax=axes[0])
    axes[0].set_title(f"Box & Swarm of {ylabel}")
    axes[0].set_ylabel(ylabel)
    axes[0].set_xlabel("Control Condition")
    axes[0].set_xticklabels(['High Control', 'Low Control'])

    # Line Plot 
    sns.lineplot(data=df, x='control_condition', y=y_col, hue='participant', palette=PT_COLORS, marker='o', linewidth=2, errorbar=None, ax=axes[1])
    axes[1].set_title(f"Per Participant {ylabel}")
    axes[1].set_xlabel("Control Condition")
    axes[1].set_xticklabels(['High Control', 'Low Control'])
    axes[1].legend(title='Participant', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(PLOT_DIR / filename, dpi=300, bbox_inches='tight')
    plt.close()

# H2 Plot Function
def plot_h2_metrics(data, y_col, title, filename):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title, fontsize=16)
    
    # Box + Swarm Plot
    sns.boxplot(data=data, x='control_condition', y=y_col, hue='Detection Status', order=['high', 'low'], palette=BOX_COLORS, fliersize=0, ax=axes[0])
    sns.swarmplot(data=data, x='control_condition', y=y_col, hue='Detection Status', order=['high', 'low'], dodge=True, palette=SWARM_COLORS, size=5, alpha=0.8, ax=axes[0])
    axes[0].set_title("Box & Swarm Distribution")
    axes[0].set_ylabel(y_col)
    axes[0].set_xlabel("Control Condition")
    
    # Line Plot 
    sns.lineplot(data=data, x='control_condition', y=y_col, hue='participant', palette=PT_COLORS, marker='o', linewidth=2, errorbar=None, ax=axes[1])
    axes[1].set_title("Per Participant Performance")
    axes[1].set_xlabel("Control Condition")
    axes[1].legend(title='Participant', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(PLOT_DIR / filename, dpi=300, bbox_inches='tight')
    plt.close()

results_df['control_condition'] = pd.Categorical(results_df['control_condition'], categories=['high', 'low'], ordered=True)
plot_h1_metric(results_df, 'd_prime', "d'", "Memory Performance (d') by Control", "H1_Control_Dprime.png")
plot_h1_metric(results_df, 'hit_rate', "Hit Rate", "Hit Rate by Control", "H1_Control_HitRate.png")

rt_by_control = targets_only.groupby(['participant', 'control_condition'])['log_mem_rt'].mean().reset_index()
plot_h1_metric(rt_by_control, 'log_mem_rt', "Recognition RT (log)", "Recognition RT by Control", "H1_Control_RT.png")

hit_rates_target = targets_only.groupby(['participant', 'control_condition', 'detection_accuracy'])['said_old_int'].mean().reset_index()
hit_rates_target.rename(columns={'said_old_int': 'hit_rate'}, inplace=True)
hit_rates_target['Detection Status'] = hit_rates_target['detection_accuracy'].map({1.0: 'Detected', 0.0: 'Undetected'})
plot_h2_metrics(hit_rates_target, 'hit_rate', "Hit Rate by Detection Status", "H2_Detection_HitRate.png")

rt_target = targets_only.groupby(['participant', 'control_condition', 'detection_accuracy'])['mem_rt'].mean().reset_index()
rt_target['Detection Status'] = rt_target['detection_accuracy'].map({1.0: 'Detected', 0.0: 'Undetected'})
plot_h2_metrics(rt_target, 'mem_rt', "Recognition RT by Detection Status", "H2_Detection_RT.png")

print("All plots generated successfully (Reverted to Seen vs Unseen).")