# =============================================================================
# BEHAVIORAL ANALYSES: CONTROL DETECTION & MEMORY
# =============================================================================

# --- Packages ---
#packages for file handling
import os
#packages for numerical operations
import numpy as np
#packages for data manipulation
import pandas as pd
#packages for stats and modeling
import matplotlib.pyplot as plt
#packages for stats and modeling
import seaborn as sns
#packages for stats and modeling
from pathlib import Path
#packages for stats and modeling
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

# Define paths and participant filter
DATA_DIR = Path(r"/Users/purengurkan/Desktop/SoA/SoA/CDmem/data/subjects")
#define output directory for results and plots
OUTPUT_DIR = Path("analysis_output")
#create output directory if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
#filter to include only specific participants (if empty, includes all)
PARTICIPANT_FILTER = [12, 14, 15, 16, 17] # Note: Removed duplicate '14' from your list

# 1A. Load Main Task Data
all_files = list(DATA_DIR.glob("CDmem_1_*.csv"))
df_list = []
for file_path in all_files:
    df_list.append(pd.read_csv(file_path))
# Apply participant filter if specified (checks the first row's participant ID in each file)
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
# We assume 'is_timeout' is a column in the test phase data that indicates whether each trial was a timeout (True/False or 1/0).
test_data = data[data["phase"] == "test"].copy()
# Ensure 'is_timeout' is boolean (if it's not already)
test_data["is_timeout"] = test_data["is_timeout"].astype(str).str.strip().str.lower() == "true"
# Calculate timeout rate per participant and condition
timeout_rate = test_data.groupby(["participant", "control_condition"])["is_timeout"].mean().reset_index()
# Identify participants who exceed the timeout threshold in either condition
failed_rows = timeout_rate[timeout_rate["is_timeout"] >= TIMEOUT_THRESHOLD]
# Extract unique participant IDs to exclude
excluded_timeout_pts = failed_rows["participant"].unique().tolist()
print(f"Criterion 1 (Timeout > 50%): Excluded {len(excluded_timeout_pts)} participants {excluded_timeout_pts}")
# Formally exclude them from the main dataset (Uncommented for active use)
data = data[~data["participant"].isin(excluded_timeout_pts)].copy()

# --- Excl 2: Detection Accuracy 2.5 SD Outliers (Main Data) ---
# We calculate each participant's overall detection accuracy across all test trials, then identify those whose accuracy is more than 2.5 SDs from the group mean (either too low or too high).
test_data = data[data["phase"] == "test"].copy()
# Ensure 'detection_accuracy' is numeric (if it's not already)
pt_accuracy = test_data.groupby("participant")["detection_accuracy"].mean().reset_index()
# Calculate group mean and SD
group_mean_acc, group_sd_acc = pt_accuracy["detection_accuracy"].mean(), pt_accuracy["detection_accuracy"].std()
# Define bounds for outliers
lower_bound_acc = group_mean_acc - (2.5 * group_sd_acc)
# Identify participants outside these bounds
upper_bound_acc = group_mean_acc + (2.5 * group_sd_acc)
# Participants with accuracy < lower_bound_acc or > upper_bound_acc are flagged for exclusion
failed_acc_rows = pt_accuracy[(pt_accuracy["detection_accuracy"] < lower_bound_acc) | (pt_accuracy["detection_accuracy"] > upper_bound_acc)]
# Extract unique participant IDs to exclude
excluded_acc_pts = failed_acc_rows["participant"].tolist()
# Print summary of this exclusion step
print(f"Criterion 2 (Det Acc > 2.5 SD): Excluded {len(excluded_acc_pts)} participants {excluded_acc_pts}")
# Formally exclude them from the main dataset (Uncommented for active use)
data = data[~data["participant"].isin(excluded_acc_pts)].copy()

# --- Excl 3: Calibration Failure (Main Data) ---
calib_data = data[data["phase"] == "calibration"].copy()
calib_data["quest_low_converged"] = calib_data["quest_low_converged"].astype(str).str.strip().str.lower() == "true"
calib_data["quest_high_converged"] = calib_data["quest_high_converged"].astype(str).str.strip().str.lower() == "true"
conv_low = calib_data.groupby("participant")["quest_low_converged"].any()
conv_high = calib_data.groupby("participant")["quest_high_converged"].any()
convergence_check = pd.DataFrame({"low_converged": conv_low, "high_converged": conv_high}).reset_index()
failed_calib_rows = convergence_check[(convergence_check["low_converged"] == False) & (convergence_check["high_converged"] == False)]
excluded_calib_pts = failed_calib_rows["participant"].tolist()
print(f"Criterion 3 (Calibration Fail): Excluded {len(excluded_calib_pts)} participants {excluded_calib_pts}")
data = data[~data["participant"].isin(excluded_calib_pts)].copy()

# --- Sync Recognition Data to Main Data (Pre-trimming) ---
valid_main_participants = data["participant"].unique()
# Ensure recog_data only includes participants who passed the main data exclusions, so we don't do extra work on recognition data for participants who are already excluded.
recog_data = recog_data[recog_data["participant"].isin(valid_main_participants)].copy()

# --- RT Outlier Trimming (Recog Data) ---
# We calculate the mean and SD of memory RTs for each participant, then exclude any trials where the RT is more than 3 SDs away from that participant's mean (either too fast or too slow).
participant_mean_rt = recog_data.groupby("participant")["mem_rt"].transform("mean")
# Calculate SD for each participants RTs to apply the 3 SD trimming
participant_sd_rt = recog_data.groupby("participant")["mem_rt"].transform("std")
valid_trials_mask = (recog_data["mem_rt"] >= (participant_mean_rt - 3 * participant_sd_rt)) & \
                    (recog_data["mem_rt"] <= (participant_mean_rt + 3 * participant_sd_rt))
clean_recog_data = recog_data[valid_trials_mask].copy()
print(f"RT Trimming: Removed {len(recog_data) - len(clean_recog_data)} trials outside of 3SD.")

# --- Excl 4: Memory Floor/Ceiling (Recog Data) ---
# We calculate each participant's overall memory accuracy (proportion of correct responses) across all recognition trials, then identify those whose accuracy is less than 55% (floor) or greater than 95% (ceiling), as these may indicate non-compliance or lack of engagement with the task.
clean_recog_data['mem_response'] = clean_recog_data['mem_response'].astype(str).str.strip().str.lower()
# Ensure 'mem_ground_truth' is also standardized to match the response formatting
clean_recog_data['mem_ground_truth'] = clean_recog_data['mem_ground_truth'].astype(str).str.strip().str.lower()

clean_recog_data['is_mem_correct'] = (
    ((clean_recog_data['mem_ground_truth'] == 'seen') & (clean_recog_data['mem_response'] == 'yes')) |
    ((clean_recog_data['mem_ground_truth'] == 'unseen') & (clean_recog_data['mem_response'] == 'no'))
)
# Calculate memory accuracy per participant
mem_accuracy = clean_recog_data.groupby('participant')['is_mem_correct'].mean().reset_index()
# Identify participants with accuracy <= 0.55 or >= 0.95
failed_mem_rows = mem_accuracy[(mem_accuracy['is_mem_correct'] <= 0.55) | (mem_accuracy['is_mem_correct'] >= 0.95)]
# Extract unique participant IDs to exclude
excluded_mem_pts = failed_mem_rows['participant'].tolist()
# Print summary of this exclusion step
print(f"Criterion 4 (Floor/Ceiling): Identified {len(excluded_mem_pts)} participants {excluded_mem_pts}")

# Formally exclude them from both datasets (Uncommented for active use)
# We exclude them from the recognition data first, then also ensure they're excluded from the main data to keep everything in sync for later analyses.
clean_recog_data = clean_recog_data[~clean_recog_data["participant"].isin(excluded_mem_pts)].copy()
# We also need to exclude them from the main data to ensure that all subsequent analyses are based on the same final sample of participants.
data = data[~data["participant"].isin(excluded_mem_pts)].copy()

print(f"\nFINAL CLEAN SAMPLE: {data['participant'].nunique()} Participants")


# =============================================================================
# PHASE 3: MANIPULATION CHECKS
# =============================================================================
print("\n" + "="*40)
print("PHASE 3: SANITY / MANIPULATION CHECKS")
print("="*40)

# For the manipulation checks, we focus on the test phase data and compare the 'high' vs. 'low' control conditions on both detection accuracy and agency ratings. 
# We will report both pooled means and per-participant means, along with paired t-tests to confirm that the manipulations had the intended effects.
mani_data = data[data['control_condition'].isin(['high', 'low'])].copy()
# Calculate pooled means and standard errors for each condition
pt_means = mani_data.groupby(['participant', 'control_condition'])[['detection_accuracy', 'agency_rating']].mean().reset_index()
# Calculate pooled means and standard errors for each condition
manipulation_summary = pt_means.groupby('control_condition')[['detection_accuracy', 'agency_rating']].mean().reset_index()
# Convert detection accuracy to percentage for easier interpretation
manipulation_summary['detection_accuracy'] = manipulation_summary['detection_accuracy'] * 100

print("\nMANIPULATION CHECK SUMMARY")
print("-" * 40)
print(manipulation_summary.round(2).to_string(index=False)) 

# For the paired t-tests, we need to reshape the data so that we have one column for the 'high' condition and one for the 'low' condition for both agency ratings and detection accuracy. 
# This allows us to directly compare the two conditions within each participant.
wide_pt_means = pt_means.pivot(index='participant', columns='control_condition', values=['agency_rating', 'detection_accuracy']).dropna()

print("\n--- AGENCY RATING T-TEST (High vs. Low) ---")
# We perform a paired t-test comparing the agency ratings in the 'high' vs. 'low' conditions across participants.
agency_ttest = pg.ttest(wide_pt_means[('agency_rating', 'high')], wide_pt_means[('agency_rating', 'low')], paired=True)
# The output includes the t-statistic, degrees of freedom, p-value, and confidence intervals for the mean difference
# which will help us determine if there is a statistically significant difference in perceived agency between the two conditions.
print(agency_ttest.to_string())

print("\n--- DETECTION ACCURACY T-TEST (High vs. Low) ---")
# We perform a paired t-test comparing the detection accuracy in the 'high' vs. 'low' conditions across participants 
# to confirm that the manipulation effectively influenced their ability to detect the targets.
acc_ttest = pg.ttest(wide_pt_means[('detection_accuracy', 'high')], wide_pt_means[('detection_accuracy', 'low')], paired=True)
print(acc_ttest.to_string())


# =============================================================================
# PHASE 4: VARIABLE DERIVATION
# =============================================================================
print("\n" + "="*40)
print("PHASE 4: VARIABLE DERIVATION")
print("="*40)

clean_recog_data["controlled"] = clean_recog_data["controlled"].astype(str).str.strip().str.lower()

# --- Compute d-prime ---
def compute_d_prime(hits, misses, false_alarms, correct_rejections):
    hit_rate = (hits + 0.5) / (hits + misses + 1)
    fa_rate = (false_alarms + 0.5) / (false_alarms + correct_rejections + 1)
    return norm.ppf(hit_rate) - norm.ppf(fa_rate)

false_alarm_stats = (
    clean_recog_data[clean_recog_data["mem_ground_truth"] == "unseen"]
    .groupby("participant")["mem_response"]
    .value_counts().unstack(fill_value=0)
    .rename(columns={"yes": "false_alarms", "no": "correct_rejections"}).reset_index()
)
for col in ["false_alarms", "correct_rejections"]:
    if col not in false_alarm_stats.columns: false_alarm_stats[col] = 0

rows = []
for participant, subject_data in clean_recog_data.groupby("participant"):
    fa_row = false_alarm_stats[false_alarm_stats["participant"] == participant]
    if fa_row.empty: continue
    fa, cr = int(fa_row["false_alarms"].iloc[0]), int(fa_row["correct_rejections"].iloc[0])

    for condition, condition_data in subject_data[subject_data["mem_ground_truth"] == "seen"].groupby("controlled"):
        if condition not in {"yes", "no"}: continue
        hits, misses = (condition_data["mem_response"] == "yes").sum(), (condition_data["mem_response"] == "no").sum()
        rows.append({"participant": participant, "controlled": condition, "hits": hits, "misses": misses,
                     "false_alarms": fa, "correct_rejections": cr, "d_prime": compute_d_prime(hits, misses, fa, cr)})

results_df = pd.DataFrame(rows)
results_df.to_csv(OUTPUT_DIR / "dprime_by_condition.csv", index=False)
print("Saved d' summaries to analysis_output/dprime_by_condition.csv")

# --- Merge Main Data vars into Recognition Targets ---
targets = clean_recog_data[clean_recog_data['mem_ground_truth'] == 'seen'].copy()

# Pull detection accuracy from the main task data
lookup_A = data[['participant', 'img_A_name', 'control_condition', 'detection_accuracy']].rename(columns={'img_A_name': 'mem_filename'})
lookup_B = data[['participant', 'img_B_name', 'control_condition', 'detection_accuracy']].rename(columns={'img_B_name': 'mem_filename'})
img_lookup = pd.concat([lookup_A, lookup_B], ignore_index=True).drop_duplicates()

if 'control_condition' in targets.columns:
    targets = targets.drop(columns=['control_condition'])

targets = targets.merge(img_lookup, on=['participant', 'mem_filename'], how='left')

# --- Assign Foils (Dummy Conditions) ---
foils = clean_recog_data[clean_recog_data['mem_ground_truth'] == 'unseen'].copy()
foils = foils.sort_values(by=['participant', 'mem_filename'])
row_numbers = foils.groupby('participant').cumcount()
total_foils = foils.groupby('participant')['mem_ground_truth'].transform('count')

# Assign Dummy Control Condition (50% high, 50% low)
foils['control_condition'] = np.where(row_numbers < (total_foils / 2), 'high', 'low')

# Assign Dummy Item Type (50% controlled/yes, 50% uncontrolled/no)
# We use modulo 2 to alternate yes/no evenly across the foils
foils['controlled'] = np.where(row_numbers % 2 == 0, 'yes', 'no')

# --- Combine Model Data & Contrast Code ---
model_data = pd.concat([targets, foils], ignore_index=True)

# Dependent Variables
model_data['said_old_int'] = model_data['mem_response'].map({'yes': 1, 'no': 0})
model_data['log_mem_rt'] = np.log(model_data['mem_rt']) # Step D7

# Predictor 1: Control Condition (+0.5 High, -0.5 Low)
model_data['control_c'] = np.where(model_data['control_condition'] == 'high', 0.5, -0.5)

# Predictor 2: Item Type (+0.5 Controlled, -0.5 Uncontrolled)
model_data['item_type_c'] = np.where(model_data['controlled'] == 'yes', 0.5, -0.5)

# Predictor 3: Detection Accuracy (+0.5 Correct, -0.5 Incorrect) - Step D6
# We use np.nan for foils since they don't have detection accuracy data
model_data['detection_accuracy_c'] = np.where(model_data['detection_accuracy'] == 1, 0.5, 
                                     np.where(model_data['detection_accuracy'] == 0, -0.5, np.nan))


## =============================================================================
# PHASE 5: GLMMs & LMMs (RQ1 & RQ2)
# =============================================================================
print("\n" + "="*40)
print("PHASE 5: STATISTICAL MODELS")
print("="*40)

model_data_pl = pl.DataFrame(model_data)

# -----------------------------------------------------------------------------
# 5A&B. RQ1: Control Level to Recognition (Data: All recognition items)
# -----------------------------------------------------------------------------
print("\n--- RQ1 Analysis A: Binomial GLMM (said_old ~ item_type * control) ---")
# Using the fallback model as standard; you can swap to maximal if your full dataset supports it
rq1_bin = glmer("said_old_int ~ item_type_c * control_c + (1 | participant)", data=model_data_pl, family="binomial")
rq1_bin.fit()
print(rq1_bin.result_fit)

print("\n--- RQ1 Analysis B: Gaussian LMM (log_RT ~ item_type * control) ---")
rq1_rt = lmer("log_mem_rt ~ item_type_c * control_c + (1 | participant)", data=model_data_pl)
rq1_rt.fit()
print(rq1_rt.result_fit)

# -----------------------------------------------------------------------------
# 5C&D. RQ2: Detection Accuracy to Recognition (Data: Targets only)
# -----------------------------------------------------------------------------
# Filter for target items with valid detection data (drops the foils)
rq2_data = model_data[(model_data['mem_ground_truth'] == 'seen')].dropna(subset=['detection_accuracy_c']).copy()
rq2_data_pl = pl.DataFrame(rq2_data)

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
# PHASE 6: PLOTTING (WITH CUSTOM COLOR PALETTES)
# =============================================================================
print("\n" + "="*40)
print("PHASE 6: GENERATING PLOTS")
print("="*40)

import seaborn as sns
import matplotlib.pyplot as plt

# Set global seaborn styling
sns.set_theme(style="whitegrid")
PLOT_DIR = OUTPUT_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# DEFINE COLOR PALETTES 
# -----------------------------------------------------------------------------
# 1. Condition Colors (Blue for High, Orange for Low)
COND_COLORS = {'high': '#3a76af', 'low': '#e38041'}

# 2. Item Type Colors (Teal for Controlled/Yes, Peach for Uncontrolled/No)
ITEM_COLORS = {'yes': '#88b5a1', 'no': '#df987b'}

# 3. Participant Colors (Assign a unique, consistent color to each participant)
unique_pts = pt_means['participant'].unique()
pt_palette = sns.color_palette("tab10", n_colors=len(unique_pts))
PT_COLORS = {pt: color for pt, color in zip(unique_pts, pt_palette)}

# -----------------------------------------------------------------------------
# H3: Sanity Check Plots (2x2 Grid)
# -----------------------------------------------------------------------------
mani_data['detection_accuracy_pct'] = mani_data['detection_accuracy'] * 100

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Sanity Checks: Agency Rating & Detection Accuracy', fontsize=16)

# Top Left: Pooled Agency Rating
sns.barplot(data=mani_data, x='control_condition', y='agency_rating', capsize=.1, errorbar='se', palette=COND_COLORS, hue='control_condition', ax=axes[0, 0])
axes[0, 0].set_title('Pooled Agency Rating')
axes[0, 0].set_ylabel('Agency Rating')

# Top Right: Per Participant Agency Rating
sns.lineplot(data=pt_means, x='control_condition', y='agency_rating', hue='participant', palette=PT_COLORS, marker='o', linewidth=2, ax=axes[0, 1])
axes[0, 1].set_title('Per Participant Agency Rating')
axes[0, 1].legend(title='Participant', bbox_to_anchor=(1.05, 1), loc='upper left')

# Bottom Left: Pooled Detection Accuracy
sns.barplot(data=mani_data, x='control_condition', y='detection_accuracy_pct', capsize=.1, errorbar='se', palette=COND_COLORS, hue='control_condition', ax=axes[1, 0])
axes[1, 0].set_title('Pooled Detection Accuracy (%)')
axes[1, 0].set_ylabel('Accuracy (%)')

# Bottom Right: Per Participant Detection Accuracy
pt_means['detection_accuracy_pct'] = pt_means['detection_accuracy'] * 100
sns.lineplot(data=pt_means, x='control_condition', y='detection_accuracy_pct', hue='participant', palette=PT_COLORS, marker='o', linewidth=2, ax=axes[1, 1])
axes[1, 1].set_title('Per Participant Accuracy (%)')
axes[1, 1].legend_.remove() # Remove redundant legend

plt.tight_layout()
plt.savefig(PLOT_DIR / "H3_Sanity_Checks.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved H3 Sanity Check Plots.")

# -----------------------------------------------------------------------------
# H1: d' Plots (3-Panel Figure)
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Memory Performance (d')", fontsize=16)

# 1. Pooled Barplot
sns.barplot(data=results_df, x='controlled', y='d_prime', capsize=.1, errorbar='se', palette=ITEM_COLORS, hue='controlled', ax=axes[0])
axes[0].set_title('Pooled Mean d-prime')
axes[0].set_ylabel("d'")
axes[0].set_xticklabels(['Uncontrolled (No)', 'Controlled (Yes)'])

# 2. Violin Plot (Distribution)
sns.violinplot(data=results_df, x='controlled', y='d_prime', inner='point', palette=ITEM_COLORS, hue='controlled', ax=axes[1])
axes[1].set_title("Distribution of d'")
axes[1].set_ylabel("d'")
axes[1].set_xticklabels(['Uncontrolled', 'Controlled'])

# 3. Per Participant Lineplot
sns.lineplot(data=results_df, x='controlled', y='d_prime', hue='participant', palette=PT_COLORS, marker='o', linewidth=2, ax=axes[2])
axes[2].set_title("Per Participant d'")
axes[2].set_xticks([0, 1])
axes[2].set_xticklabels(['Uncontrolled', 'Controlled'])
axes[2].legend(title='Participant', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig(PLOT_DIR / "H1_Dprime.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved H1 d' Plots.")

# -----------------------------------------------------------------------------
# H2: Hit Rate Plots (3-Panel Figure)
# -----------------------------------------------------------------------------
results_df['hit_rate'] = results_df['hits'] / (results_df['hits'] + results_df['misses'])

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Memory Performance (Hit Rate)", fontsize=16)

# 1. Pooled Barplot
sns.barplot(data=results_df, x='controlled', y='hit_rate', capsize=.1, errorbar='se', palette=ITEM_COLORS, hue='controlled', ax=axes[0])
axes[0].set_title('Pooled Mean Hit Rate')
axes[0].set_ylabel("Hit Rate")
axes[0].set_xticklabels(['Uncontrolled', 'Controlled'])

# 2. Violin Plot
sns.violinplot(data=results_df, x='controlled', y='hit_rate', inner='point', palette=ITEM_COLORS, hue='controlled', ax=axes[1])
axes[1].set_title("Distribution of Hit Rate")
axes[1].set_ylabel("Hit Rate")
axes[1].set_xticklabels(['Uncontrolled', 'Controlled'])

# 3. Per Participant Lineplot
sns.lineplot(data=results_df, x='controlled', y='hit_rate', hue='participant', palette=PT_COLORS, marker='o', linewidth=2, ax=axes[2])
axes[2].set_title("Per Participant Hit Rate")
axes[2].set_xticks([0, 1])
axes[2].set_xticklabels(['Uncontrolled', 'Controlled'])
axes[2].legend(title='Participant', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig(PLOT_DIR / "H2_HitRate.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved H2 Hit Rate Plots.")