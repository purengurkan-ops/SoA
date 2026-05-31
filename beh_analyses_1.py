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
# PHASE 4: VARIABLE DERIVATION & D-PRIME
# =============================================================================
print("\n" + "="*40)
print("PHASE 4: VARIABLE DERIVATION")
print("="*40)

# Before we can compute d-prime, we need to ensure that the recognition data is clean and standardized.
# We will standardize the 'mem_response' and 'mem_ground_truth' columns to ensure that they are in a consistent format (e.g., all lowercase, no extra spaces) so that we can accurately count hits, misses, false alarms, and correct rejections.
clean_recog_data["controlled"] = clean_recog_data["controlled"].astype(str).str.strip().str.lower()

# --- Compute d-prime ---
# The d-prime calculation requires us to count the number of hits, misses, false alarms, and correct rejections for each participant and condition.
def compute_d_prime(hits, misses, false_alarms, correct_rejections):
    hit_rate = (hits + 0.5) / (hits + misses + 1)
    fa_rate = (false_alarms + 0.5) / (false_alarms + correct_rejections + 1)
    return norm.ppf(hit_rate) - norm.ppf(fa_rate)

# First, we calculate the false alarm and correct rejection counts for each participant based on the 'unseen' trials 
# since these are the trials where a "yes" response would be a false alarm and a "no" response would be a correct rejection.
false_alarm_stats = (
    clean_recog_data[clean_recog_data["mem_ground_truth"] == "unseen"]
    .groupby("participant")["mem_response"]
    .value_counts().unstack(fill_value=0)
    .rename(columns={"yes": "false_alarms", "no": "correct_rejections"}).reset_index()
)
# We need to ensure that if a participant has no false alarms or no correct rejections, we still have those columns in the dataframe with a count of 0, otherwise our d-prime calculation will fail for those participants.
for col in ["false_alarms", "correct_rejections"]:
    if col not in false_alarm_stats.columns: false_alarm_stats[col] = 0

# Next, we iterate through each participant and condition in the 'seen' trials to count hits and misses, and then merge those counts with the false alarm stats to compute d-prime for each participant and condition.
rows = []
# We loop through each participant's data in the recognition dataset, and for each participant, we retrieve their false alarm and correct rejection counts from the false_alarm_stats dataframe.
for participant, subject_data in clean_recog_data.groupby("participant"):
    fa_row = false_alarm_stats[false_alarm_stats["participant"] == participant]
    if fa_row.empty: continue
    fa, cr = int(fa_row["false_alarms"].iloc[0]), int(fa_row["correct_rejections"].iloc[0])
# Then, for each condition (high vs. low control) within the 'seen' trials, we count the number of hits (where mem_response is "yes") and misses (where mem_response is "no"), and use those counts along with the false alarm and correct rejection counts to compute d-prime for that participant and condition.
    for condition, condition_data in subject_data[subject_data["mem_ground_truth"] == "seen"].groupby("controlled"):
        if condition not in {"yes", "no"}: continue
        hits, misses = (condition_data["mem_response"] == "yes").sum(), (condition_data["mem_response"] == "no").sum()
        rows.append({"participant": participant, "controlled": condition, "hits": hits, "misses": misses,
                     "false_alarms": fa, "correct_rejections": cr, "d_prime": compute_d_prime(hits, misses, fa, cr)})

results_df = pd.DataFrame(rows)
results_df.to_csv(OUTPUT_DIR / "dprime_by_condition.csv", index=False)
print("Saved d' summaries to analysis_output/dprime_by_condition.csv")

print("\nD-prime results by participant and condition:")
print(results_df.groupby("controlled")["d_prime"].describe())

# --- Z-Score Agency Ratings ---
# To make the agency ratings more comparable across participants, we will z-score them within each participant. 
# This means that for each participant, we will subtract their mean agency rating from each of their ratings and then divide by their standard deviation. 
# This standardization allows us to interpret the agency ratings in terms of how many standard deviations they are above or below that participant's average rating, which can help account for individual differences in how participants use the rating scale.
data['agency_z'] = data.groupby('participant')['agency_rating'].transform(lambda x: (x - x.mean()) / x.std())

# --- Merge Main Data vars into Recognition Targets ---
# We need to merge the main data variables (like control condition, detection accuracy, and agency rating) into the recognition data for the 'seen' trials (targets) so that we can use those variables as predictors in our GLMMs.
targets = clean_recog_data[clean_recog_data['mem_ground_truth'] == 'seen'].copy()

# Remove 'trial_level' from the lists below, just keeping 'control_condition'
# We create two lookup tables: one for the 'img_A_name' and one for the 'img_B_name', which contain the participant ID, image filename, control condition, detection accuracy, and z-scored agency rating.
# We then concatenate these two lookup tables together and drop duplicates to create a single lookup table that we can merge with the recognition data based on participant ID and image filename.
lookup_A = data[['participant', 'img_A_name', 'control_condition', 'detection_accuracy', 'agency_z']].rename(columns={'img_A_name': 'mem_filename'})
lookup_B = data[['participant', 'img_B_name', 'control_condition', 'detection_accuracy', 'agency_z']].rename(columns={'img_B_name': 'mem_filename'})
img_lookup = pd.concat([lookup_A, lookup_B], ignore_index=True).drop_duplicates()

# If the recognition data already has a blank/old control_condition column, drop it so they merge cleanly
if 'control_condition' in targets.columns:
    targets = targets.drop(columns=['control_condition'])

targets = targets.merge(img_lookup, on=['participant', 'mem_filename'], how='left')

# --- Assign Foils ---
# For the 'unseen' trials (foils), we will assign a control condition based on the order in which they appear for each participant.
# We will sort the foils by participant and image filename, then assign the first half of the foils for each participant to the 'high' control condition and the second half to the 'low' control condition. 
# This is a somewhat arbitrary assignment, but it allows us to include the foils in our analyses with a control condition variable that we can use in the GLMMs, even though the foils were not actually presented during the test phase. 
# This way, we can compare the recognition performance for targets and foils across the two control conditions, which is important for our research questions about how the manipulations affect memory performance.
foils = clean_recog_data[clean_recog_data['mem_ground_truth'] == 'unseen'].copy()
# We sort the foils by participant and image filename to ensure that the assignment of control conditions is consistent within each participant.ß
foils = foils.sort_values(by=['participant', 'mem_filename'])
# We use the groupby and cumcount functions to assign a row number to each foil trial within each participant, which allows us to determine which trials belong to the first half and which belong to the second half for the purpose of assigning control conditions.
row_numbers = foils.groupby('participant').cumcount()
# We calculate the total number of foils for each participant using the transform function, which allows us to assign the 'high' control condition to the first half of the foils and the 'low' control condition to the second half for each participant.
total_foils = foils.groupby('participant')['mem_ground_truth'].transform('count')
# We use the np.where function to assign the control condition based on whether the row number for each foil trial is less than half of the total number of foils for that participant.
foils['control_condition'] = np.where(row_numbers < (total_foils / 2), 'high', 'low')

# --- Combine Model Data & Contrast Code ---
# Finally, we concatenate the targets and foils back together to create a single dataset that we can use for our GLMM analyses.
# We also create contrast-coded variables for the item type (target vs. foil) and control condition (high vs. low), as well as a binary variable for whether the participant said "old" or "new" and a log-transformed variable for memory RT, which will be used as the dependent variable in our GLMMs.
model_data = pd.concat([targets, foils], ignore_index=True)
# For the item type contrast code, we assign a value of 0.5 to the 'seen' trials (targets) and -0.5 to the 'unseen' trials (foils).
model_data['item_type_c'] = np.where(model_data['mem_ground_truth'] == 'seen', 0.5, -0.5)
# For the control condition contrast code, we assign a value of 0.5 to the 'high' control condition and -0.5 to the 'low' control condition.ß
model_data['control_c'] = np.where(model_data['control_condition'] == 'high', 0.5, -0.5)
# We create a binary variable for whether the participant said "old" (1) or "new" (0) based on their memory response, which will be used as the dependent variable in our binomial GLMMs.
model_data['said_old_int'] = model_data['mem_response'].map({'yes': 1, 'no': 0})
# We create a log-transformed variable for memory RT to use as the dependent variable in our Gaussian LMMs, which can help normalize the distribution of RTs and reduce the influence of outliers.
# We add a small constant (e.g., 1) to the RTs before taking the log to avoid issues with log(0) if there are any trials with an RT of 0.
# Note: If there are any negative RTs (which shouldn't happen but just in case), we would need to handle those separately, perhaps by excluding them or setting them to a small positive value before log transformation.
model_data['log_mem_rt'] = np.log(model_data['mem_rt'])


# =============================================================================
# PHASE 5: GLMMs & LMMs (RQ1, RQ2, RQ3)
# =============================================================================
print("\n" + "="*40)
print("PHASE 5: STATISTICAL MODELS")
print("="*40)

# --- 5A. RQ1: Motor Control -> Memory (All Data) ---
print("\n--- RQ1: Binomial GLMM (said_old ~ item_type * control) ---")
model_data_pl = pl.DataFrame(model_data)
rq1_bin = glmer("said_old_int ~ item_type_c * control_c + (1 | participant)", data=model_data_pl, family="binomial")
rq1_bin.fit()
print(rq1_bin.result_fit)

print("\n--- RQ1: Gaussian LMM (log_RT ~ item_type * control) ---")
rq1_rt = lmer("log_mem_rt ~ item_type_c * control_c + (1 | participant)", data=model_data_pl)
rq1_rt.fit()
print(rq1_rt.result_fit)