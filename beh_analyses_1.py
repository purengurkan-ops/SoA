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
px_convergence = {} # To track how many staircases converged per participant to inform future analyses (Excl 5)

for px, px_df in calib_data.groupby("participant"):
    n_converged = 0
    for target, target_df in px_df.groupby("calib_target"):
        if not target_df.empty:
            final_sd = target_df["quest_alpha_sd"].iloc[-1]
            if not pd.isna(final_sd) and final_sd < 0.20:
                n_converged += 1
    
    px_convergence[px] = n_converged
    if n_converged == 0: # If both staircases failed to converge, exclude the participant
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


# =============================================================================
# PHASE 3: MANIPULATION CHECKS
# =============================================================================
print("\n" + "="*40)
print("PHASE 3: SANITY / MANIPULATION CHECKS")
print("="*40)

mani_data = data[data['control_condition'].isin(['high', 'low'])].copy()
pt_means = mani_data.groupby(['participant', 'control_condition'])[['detection_accuracy', 'agency_rating']].mean().reset_index()
manipulation_summary = pt_means.groupby('control_condition')[['detection_accuracy', 'agency_rating']].mean().reset_index()
manipulation_summary['detection_accuracy'] = manipulation_summary['detection_accuracy'] * 100

print("\nMANIPULATION CHECK SUMMARY")
print("-" * 40)
print(manipulation_summary.round(2).to_string(index=False)) 

wide_pt_means = pt_means.pivot(index='participant', columns='control_condition', values=['agency_rating', 'detection_accuracy']).dropna()

print("\n--- AGENCY RATING T-TEST (High vs. Low) ---")
agency_ttest = pg.ttest(wide_pt_means[('agency_rating', 'high')], wide_pt_means[('agency_rating', 'low')], paired=True)
print(agency_ttest.to_string())

print("\n--- DETECTION ACCURACY T-TEST (High vs. Low) ---")
acc_ttest = pg.ttest(wide_pt_means[('detection_accuracy', 'high')], wide_pt_means[('detection_accuracy', 'low')], paired=True)
print(acc_ttest.to_string())


# =============================================================================
# PHASE 4: VARIABLE DERIVATION
# =============================================================================
print("\n" + "="*40)
print("PHASE 4: VARIABLE DERIVATION")
print("="*40)

clean_recog_data["controlled"] = clean_recog_data["controlled"].astype(str).str.strip().str.lower()

# --- 1. Merge Main Data vars into Recognition Targets ---
targets = clean_recog_data[clean_recog_data['mem_ground_truth'] == 'seen'].copy()

lookup_A = data[['participant', 'img_A_name', 'control_condition', 'detection_accuracy']].rename(columns={'img_A_name': 'mem_filename'})
lookup_B = data[['participant', 'img_B_name', 'control_condition', 'detection_accuracy']].rename(columns={'img_B_name': 'mem_filename'})
img_lookup = pd.concat([lookup_A, lookup_B], ignore_index=True).drop_duplicates()

if 'control_condition' in targets.columns:
    targets = targets.drop(columns=['control_condition'])

targets = targets.merge(img_lookup, on=['participant', 'mem_filename'], how='left')

# --- 2. Assign Foils (Dummy Conditions) ---
foils = clean_recog_data[clean_recog_data['mem_ground_truth'] == 'unseen'].copy()
foils = foils.sort_values(by=['participant', 'mem_filename'])
row_numbers = foils.groupby('participant').cumcount()
total_foils = foils.groupby('participant')['mem_ground_truth'].transform('count')

foils['control_condition'] = np.where(row_numbers < (total_foils / 2), 'high', 'low')
foils['controlled'] = np.where(row_numbers % 2 == 0, 'yes', 'no')

# --- 3. Combine Model Data & Contrast Code ---
model_data = pd.concat([targets, foils], ignore_index=True)

model_data['said_old_int'] = model_data['mem_response'].map({'yes': 1, 'no': 0})
model_data['log_mem_rt'] = np.log(model_data['mem_rt']) 

# Predictor 1: Control Condition (+0.5 High, -0.5 Low)
model_data['control_c'] = np.where(model_data['control_condition'] == 'high', 0.5, -0.5)

# Predictor 2: Item Type (+0.5 SEEN, -0.5 UNSEEN) - FIXED BASED ON SUPERVISOR FEEDBACK
model_data['item_type_c'] = np.where(model_data['mem_ground_truth'] == 'seen', 0.5, -0.5)

# Predictor 3: Detection Accuracy
model_data['detection_accuracy_c'] = np.where(model_data['detection_accuracy'] == 1, 0.5, 
                                     np.where(model_data['detection_accuracy'] == 0, -0.5, 0))

# --- 4. Compute d-prime (Now calculating based on High vs Low Control) ---
def compute_d_prime(hits, misses, false_alarms, correct_rejections):
    hit_rate = (hits + 0.5) / (hits + misses + 1)
    fa_rate = (false_alarms + 0.5) / (false_alarms + correct_rejections + 1)
    return norm.ppf(hit_rate) - norm.ppf(fa_rate)

false_alarm_stats = (
    model_data[model_data["mem_ground_truth"] == "unseen"]
    .groupby("participant")["mem_response"]
    .value_counts().unstack(fill_value=0)
    .rename(columns={"yes": "false_alarms", "no": "correct_rejections"}).reset_index()
)
for col in ["false_alarms", "correct_rejections"]:
    if col not in false_alarm_stats.columns: false_alarm_stats[col] = 0

rows = []
for participant, subject_data in model_data.groupby("participant"):
    fa_row = false_alarm_stats[false_alarm_stats["participant"] == participant]
    if fa_row.empty: continue
    fa, cr = int(fa_row["false_alarms"].iloc[0]), int(fa_row["correct_rejections"].iloc[0])

    # Grouping by control_condition for our main effect hypothesis!
    for condition, condition_data in subject_data[subject_data["mem_ground_truth"] == "seen"].groupby("control_condition"):
        if condition not in {"high", "low"}: continue
        hits, misses = (condition_data["mem_response"] == "yes").sum(), (condition_data["mem_response"] == "no").sum()
        rows.append({"participant": participant, "control_condition": condition, "hits": hits, "misses": misses,
                     "false_alarms": fa, "correct_rejections": cr, "d_prime": compute_d_prime(hits, misses, fa, cr)})

results_df = pd.DataFrame(rows)
results_df.to_csv(OUTPUT_DIR / "dprime_by_condition.csv", index=False)
print("Saved d' summaries to analysis_output/dprime_by_condition.csv")

# --- 5. Above-Chance Memory Check (One-Sample T-Test) ---
mean_dprime_per_pt = results_df.groupby('participant')['d_prime'].mean()
print("\n--- D-PRIME ONE-SAMPLE T-TEST (Is memory above chance?) ---")
dprime_ttest = pg.ttest(mean_dprime_per_pt, 0)
print(dprime_ttest.to_string())


# =============================================================================
# PHASE 5: GLMMs & LMMs (RQ1 & RQ2)
# =============================================================================
print("\n" + "="*40)
print("PHASE 5: STATISTICAL MODELS")
print("="*40)

model_data_pl = pl.DataFrame(model_data)

print("\n--- RQ1 Analysis A: Binomial GLMM (said_old ~ item_type * control) ---")
rq1_bin = glmer("said_old_int ~ item_type_c * control_c + (1 | participant)", data=model_data_pl, family="binomial")
rq1_bin.fit()
print(rq1_bin.result_fit)
print("\n--- Random Effects (Participant Variability) ---")
print(rq1_bin.ranef_var) 

print("\n--- RQ1 Analysis B: Gaussian LMM (log_RT ~ item_type * control) ---")
rq1_rt = lmer("log_mem_rt ~ item_type_c * control_c + (1 | participant)", data=model_data_pl)
rq1_rt.fit()
print(rq1_rt.result_fit)

rq2_data_pl = pl.DataFrame(model_data)

print("\n--- RQ2 Analysis C: Binomial GLMM (3-Way Interaction) ---")
rq2_formula_bin = "said_old_int ~ detection_accuracy_c * control_c * item_type_c + (1 | participant)"
rq2_bin = glmer(rq2_formula_bin, data=rq2_data_pl, family="binomial")
rq2_bin.fit()
print(rq2_bin.result_fit)

print("\n--- RQ2 Analysis D: Gaussian LMM (3-Way Interaction) ---")
rq2_formula_rt = "log_mem_rt ~ detection_accuracy_c * control_c * item_type_c + (1 | participant)"
rq2_rt = lmer(rq2_formula_rt, data=rq2_data_pl)
rq2_rt.fit()
print(rq2_rt.result_fit)

# =============================================================================
# PHASE 6: PLOTTING (UPDATED FOR MAIN EFFECT)
# =============================================================================
print("\n" + "="*40)
print("PHASE 6: GENERATING PLOTS")
print("="*40)

sns.set_theme(style="whitegrid")
PLOT_DIR = OUTPUT_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Condition Colors (Blue for High, Orange for Low)
COND_COLORS = {'high': '#3a76af', 'low': '#e38041'}

unique_pts = pt_means['participant'].unique()
pt_palette = sns.color_palette("tab10", n_colors=len(unique_pts))
PT_COLORS = {pt: color for pt, color in zip(unique_pts, pt_palette)}

# Sort condition alphabetically so 'high' always comes before 'low' in plots
results_df['control_condition'] = pd.Categorical(results_df['control_condition'], categories=['high', 'low'], ordered=True)

# --- H3: Sanity Check Plots ---
mani_data['detection_accuracy_pct'] = mani_data['detection_accuracy'] * 100

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Sanity Checks: Agency Rating & Detection Accuracy', fontsize=16)

sns.barplot(data=mani_data, x='control_condition', y='agency_rating', order=['high', 'low'], capsize=.1, errorbar='se', palette=COND_COLORS, hue='control_condition', ax=axes[0, 0])
axes[0, 0].set_title('Pooled Agency Rating')
axes[0, 0].set_ylabel('Agency Rating')

sns.lineplot(data=pt_means, x='control_condition', y='agency_rating', hue='participant', palette=PT_COLORS, marker='o', linewidth=2, ax=axes[0, 1])
axes[0, 1].set_title('Per Participant Agency Rating')
axes[0, 1].legend(title='Participant', bbox_to_anchor=(1.05, 1), loc='upper left')

sns.barplot(data=mani_data, x='control_condition', y='detection_accuracy_pct', order=['high', 'low'], capsize=.1, errorbar='se', palette=COND_COLORS, hue='control_condition', ax=axes[1, 0])
axes[1, 0].set_title('Pooled Detection Accuracy (%)')
axes[1, 0].set_ylabel('Accuracy (%)')

pt_means['detection_accuracy_pct'] = pt_means['detection_accuracy'] * 100
sns.lineplot(data=pt_means, x='control_condition', y='detection_accuracy_pct', hue='participant', palette=PT_COLORS, marker='o', linewidth=2, ax=axes[1, 1])
axes[1, 1].set_title('Per Participant Accuracy (%)')
axes[1, 1].legend_.remove() 

plt.tight_layout()
plt.savefig(PLOT_DIR / "H3_Sanity_Checks.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved H3 Sanity Check Plots.")

# --- H1: d' Plots ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Memory Performance (d')", fontsize=16)

# Swapped x to control_condition, updated palette
sns.barplot(data=results_df, x='control_condition', y='d_prime', order=['high', 'low'], capsize=.1, errorbar='se', palette=COND_COLORS, hue='control_condition', ax=axes[0])
axes[0].set_title('Pooled Mean d-prime')
axes[0].set_ylabel("d'")
axes[0].set_xticks([0, 1])
axes[0].set_xticklabels(['High Control', 'Low Control'])

# Box + Strip Plot (Honest Distribution)
sns.boxplot(data=results_df, x='control_condition', y='d_prime', order=['high', 'low'], width=0.4, palette=COND_COLORS, hue='control_condition', fliersize=0, ax=axes[1])
sns.stripplot(data=results_df, x='control_condition', y='d_prime', order=['high', 'low'], color='black', alpha=0.6, jitter=True, ax=axes[1])
axes[1].set_title("Box & Swarm of d'")
axes[1].set_ylabel("d'")
axes[1].set_xticks([0, 1])
axes[1].set_xticklabels(['High Control', 'Low Control'])

sns.lineplot(data=results_df, x='control_condition', y='d_prime', hue='participant', palette=PT_COLORS, marker='o', linewidth=2, ax=axes[2])
axes[2].set_title("Per Participant d'")
axes[2].set_xticks([0, 1])
axes[2].set_xticklabels(['High Control', 'Low Control'])
axes[2].legend(title='Participant', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig(PLOT_DIR / "H1_Dprime.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved H1 d' Plots.")

# --- H2: Hit Rate Plots ---
results_df['hit_rate'] = results_df['hits'] / (results_df['hits'] + results_df['misses'])

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Memory Performance (Hit Rate)", fontsize=16)

# Swapped x to control_condition, updated palette
sns.barplot(data=results_df, x='control_condition', y='hit_rate', order=['high', 'low'], capsize=.1, errorbar='se', palette=COND_COLORS, hue='control_condition', ax=axes[0])
axes[0].set_title('Pooled Mean Hit Rate')
axes[0].set_ylabel("Hit Rate")
axes[0].set_xticks([0, 1])
axes[0].set_xticklabels(['High Control', 'Low Control'])

# Box + Strip Plot 
sns.boxplot(data=results_df, x='control_condition', y='hit_rate', order=['high', 'low'], width=0.4, palette=COND_COLORS, hue='control_condition', fliersize=0, ax=axes[1])
sns.stripplot(data=results_df, x='control_condition', y='hit_rate', order=['high', 'low'], color='black', alpha=0.6, jitter=True, ax=axes[1])
axes[1].set_title("Box & Swarm of Hit Rate")
axes[1].set_ylabel("Hit Rate")
axes[1].set_xticks([0, 1])
axes[1].set_xticklabels(['High Control', 'Low Control'])

sns.lineplot(data=results_df, x='control_condition', y='hit_rate', hue='participant', palette=PT_COLORS, marker='o', linewidth=2, ax=axes[2])
axes[2].set_title("Per Participant Hit Rate")
axes[2].set_xticks([0, 1])
axes[2].set_xticklabels(['High Control', 'Low Control'])
axes[2].legend(title='Participant', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig(PLOT_DIR / "H2_HitRate.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved H2 Hit Rate Plots.")