#-------------------------
# This file includes pseudocode / guidelines for the behavioral analyses of the control detection & memory project
# --------------------

# The packages you need 
import os #the package that allows python to read files from local folders
import numpy as np #the package that allows you to perform numerical operations
import pandas as pd #the package that allows you to work with dataframes
import matplotlib.pyplot as plt #the package that allows you to create plots
import seaborn as sns #the package that allows you to create more advanced plots
from pathlib import Path #the package that allows you to work with file paths in a more flexible way
from scipy.stats import norm #the package that allows you to perform statistical analyses (e.g., t-tests, ANOVAs, etc.)
import polars as pl
from pymer4.models import glmer

import pingouin as pg #the package that allows you to perform statistical analyses (e.g., t-tests, ANOVAs, etc.)


#-------------------------
# Import data
#-------------------------

DATA_DIR = Path(r"pilot_data")

# same goes for the output directory. I have just created a new empty folder called analysis_output in our repo. set that as OUTPUT_DIR below

OUTPUT_DIR = Path("analysis_output")

# Participant filter: set to a non-empty list to restrict analyses to specific
# participant IDs, e.g. PARTICIPANT_FILTER = [2, 3].
# Leave as an empty list [] to include ALL participants found in data files.
PARTICIPANT_FILTER = []

#-------------------------
# Load data
#-------------------------

# 1. Find all data files matching the specific naming pattern. 
# DATA_DIR.glob tells Python to look inside DATA_DIR and fetch files ending with .csv
# CDmem_1_* is the naming pattern for data coming from the control detection task NOT recognition. 
all_files = list(DATA_DIR.glob("CDmem_1_*.csv"))

# 2. Create an empty list to store the dataframes for each participant
# a list is the same structure as the participant filter above :)

df_list = []

# 3. Loop through each file path that we found
# a for loop does whatever is indented below it, for each item in the list
# so this one will do the indented actions for everything in all_files
for file_path in all_files:
    
    # Read the CSV file into a pandas DataFrame
    # a pandas dataframe is basically a table with rows and columns
    # you need to give it a name, usually df
    df = pd.read_csv(file_path)
    
    # Add this DataFrame to the empty list you created.
    # you do this via writing the name of the empty list and then .append(df), single line of code
    # .append() is a "method" that adds the item in parantheses to the end of the list
    
    df_list.append(df)


# up to this point, we have loaded all the data in the data folder.
# 4.  now we need to filter participants if you set numbers in PARTICIPANT_FILTER list above

if PARTICIPANT_FILTER: 
    # Use list comprehension to keep only data for included participants
    # change "thelistyoucreated" to whatever name you gave your list above
    df_list = [df for df in df_list if df["participant"].iloc[0] in PARTICIPANT_FILTER]

# 5. recall that we have everything in a list. but for stats, we need everything in a dataframe (i.e., rows, columns, etc. so we can use column names etc later)
# we do this with pd.concat()
# pd.concat() is a function that takes a list of dataframes and concatenates them into a single dataframe
# ignore_index = True means that we want to create a new index for the combined dataframe

# change "thelistyoucreated" to whatever name you gave your list above
if df_list: # This checks if the list is not empty
    data = pd.concat(df_list, ignore_index=True)
    print(f"Successfully loaded {len(df_list)} data files! Total rows: {len(data)}")
    # len() gives the length of a list. len(thelistyoucreated) gives the number of lists in the listyoucreated. recall that thelistyoucreated was created to store all data from the folder.
    # similarly, len(data) gives the number of rows in the dataframe.
else:
    print("Warning: No data files found. Please check your DATA_DIR folder.") 

# ============================================================================
# EXCLUSION CRITERION 1 - TIMEOUT RATE
# ============================================================================
# We want to exclude participants who have 50% or more timeout trials 
# in either the 'low' or 'high' control condition during the 'test' phase.
# set a threshold for the timeout rate. basically a variable that equals 0.50

TIMEOUT_THRESHOLD = 0.50

# 1. Filter the data to only include the "test" phase 
# the .copy one creates a copy so that the original doesnt get modified
test_data = data[data["phase"] == "test"].copy()

# 2. Make sure the "is_timeout" column contains only True or False
# Sometimes it loads as text like "True" or "False", so we force it to be a boolean
# so it does not take it as the word "true" but the logical true
test_data["is_timeout"] = test_data["is_timeout"].astype(str).str.strip().str.lower() == "true"

# 3. Calculate the timeout rate per participant per condition. 
# We can do this by taking the mean of the "is_timeout" column (since True is 1 and False is 0)

# timeout_rate = ........
timeout_rate = test_data.groupby(["participant", "control_condition"])["is_timeout"].mean().reset_index()

# 4. Find the rows where the timeout rate is greater than or equal to our threshold

failed_rows = timeout_rate[timeout_rate["is_timeout"] >= TIMEOUT_THRESHOLD]

# 5. Get a simple list of their unique participant IDs
listofexcludedparticipants = failed_rows["participant"].unique().tolist()

print("\nEXCLUSION CRITERION 1: Timeout Rate")
# change "listofexcludedparticipants" to whatever you named that list at step 5
if len(listofexcludedparticipants) > 0:
    print(f"  -> Excluded {len(listofexcludedparticipants)} participants: {listofexcludedparticipants}")
else:
    print("  -> No participants were excluded.")

# 6. Remove these participants from our main 'data' dataframe
# similar to how we filtered stuff in the first step, but this time we use the '~' symbol
# The '~' symbol means "NOT" (keep data where participant is NOT in the list)
# don't forget to use .copy() again

data = data[~data["participant"].isin(listofexcludedparticipants)].copy()

#-----------------------------------
# NOW TRY TO LOAD AND FILTER THE RECOGNITION DATA
# file naming convention is: CDmem_*_recognition.csv
# recall that the exclusion criteria for recognition data is trial based:
# following Ren et al., 2026, any trial that has a RT of +- 3SD of the **participant mean** should be excluded

# 1. load the data (similar to how we loaded the main data, but with a different naming pattern)
recog_files = list(DATA_DIR.glob("CDmem_*_recognition.csv"))
recog_list = []

for file_path in recog_files:
    df = pd.read_csv(file_path)
    recog_list.append(df)

if recog_list:
    recog_data = pd.concat(recog_list, ignore_index=True)
    print(f"\nSuccessfully loaded {len(recog_list)} recognition files! Total rows: {len(recog_data)}")
else:
    print("\nWarning: No recognition data files found.")

# calculate mean and SD of RT for each participant
participant_mean_rt = recog_data.groupby("participant")["mem_rt"].transform("mean")
participant_sd_rt = recog_data.groupby("participant")["mem_rt"].transform("std")

# calculate upper and lower bounds for each trial based on participant mean and SD
upper_bound = participant_mean_rt + (3 * participant_sd_rt)
lower_bound = participant_mean_rt - (3 * participant_sd_rt)

# create a boolean mask to identify valid trials (those within the bounds)
valid_trials_mask = (recog_data["mem_rt"] >= lower_bound) & (recog_data["mem_rt"] <= upper_bound)

# apply the mask to filter out invalid trials
clean_recog_data = recog_data[valid_trials_mask].copy()

# report how many trials were removed
trials_removed = len(recog_data) - len(clean_recog_data)
print(f"EXCLUSION CRITERION 2: Removed {trials_removed} recognition trials outside of 3SD.")


# NEXT STEP: MATCH EXCLUSIONS FROM BOTH DATASETS AND RUN ANALYSES
# for t-test calculate d' for each participant and condition, then run a paired t-test comparing d' between controlled and uncontrolled items

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Make sure recognition data only includes the same participants that survived the main timeout exclusion.
valid_participants = data["participant"].unique()
print(f"Participants after timeout exclusion: {len(valid_participants)}")

if len(valid_participants) == 0:
    raise ValueError("No valid participants remain after timeout exclusions. Cannot continue recognition analysis.")

clean_recog_data = clean_recog_data[clean_recog_data["participant"].isin(valid_participants)].copy()

if clean_recog_data.empty:
    print("Warning: Recognition data contains no trials after matching exclusions. Check participant filtering and recognition files.")
else:
    print(f"Recognition trials after matching exclusions: {len(clean_recog_data)}")

# Standardize recognition labels
clean_recog_data["mem_response"] = clean_recog_data["mem_response"].astype(str).str.strip().str.lower()
clean_recog_data["mem_ground_truth"] = clean_recog_data["mem_ground_truth"].astype(str).str.strip().str.lower()
clean_recog_data["controlled"] = clean_recog_data["controlled"].astype(str).str.strip().str.lower()

# Function to compute d-prime with loglinear correction

def compute_d_prime(hits, misses, false_alarms, correct_rejections):
    hit_rate = (hits + 0.5) / (hits + misses + 1)
    fa_rate = (false_alarms + 0.5) / (false_alarms + correct_rejections + 1)
    return norm.ppf(hit_rate) - norm.ppf(fa_rate)

# Compute false alarm counts per participant using all new/unseen trials
false_alarm_stats = (
    clean_recog_data[clean_recog_data["mem_ground_truth"] == "unseen"]
    .groupby("participant")["mem_response"]
    .value_counts()
    .unstack(fill_value=0)
    .rename(columns={"yes": "false_alarms", "no": "correct_rejections"})
    .reset_index()
)

for col in ["false_alarms", "correct_rejections"]:
    if col not in false_alarm_stats.columns:
        false_alarm_stats[col] = 0

rows = []
for participant, subject_data in clean_recog_data.groupby("participant"):
    fa_row = false_alarm_stats[false_alarm_stats["participant"] == participant]
    if fa_row.empty:
        continue
    false_alarms = int(fa_row["false_alarms"].iloc[0])
    correct_rejections = int(fa_row["correct_rejections"].iloc[0])

    for condition, condition_data in subject_data[subject_data["mem_ground_truth"] == "seen"].groupby("controlled"):
        if condition not in {"yes", "no"}:
            continue
        hits = (condition_data["mem_response"] == "yes").sum()
        misses = (condition_data["mem_response"] == "no").sum()
        d_prime = compute_d_prime(hits, misses, false_alarms, correct_rejections)
        rows.append({
            "participant": participant,
            "controlled": condition,
            "hits": hits,
            "misses": misses,
            "false_alarms": false_alarms,
            "correct_rejections": correct_rejections,
            "d_prime": d_prime,
        })

if not rows:
    raise ValueError("No d-prime rows were generated. Check recognition labels and controlled annotation.")

results_df = pd.DataFrame(rows)
results_df.to_csv(OUTPUT_DIR / "dprime_by_condition.csv", index=False)

print("\nD-prime results by participant and condition:")
print(results_df.groupby("controlled")["d_prime"].describe())

wide_dprime = results_df.pivot(index="participant", columns="controlled", values="d_prime").dropna()
print(f"\nParticipants with both conditions available: {len(wide_dprime)}")

if len(wide_dprime) < 2:
    print("Not enough paired participants to run a paired t-test.")
else:
    ttest_results = pg.ttest(wide_dprime["yes"], wide_dprime["no"], paired=True)
    print("\nPaired t-test comparing d' for controlled=yes vs controlled=no")
    print(ttest_results)


import polars as pl
from pymer4.models import glmer

# =============================================================================
# STEP 2: Prepare the TARGET trials (old items)
# =============================================================================
# "Targets" are images that were ACTUALLY shown to participants during encoding.
# In the recognition file, these are rows where mem_ground_truth == 'seen'.
#
# We also need to know WHICH control condition each image came from
# (HIGH or LOW), because that's our key predictor.
# That information lives in the main task file, not in the recognition file.

# --- 2a. Pull out only the "seen" images from recognition ---
targets = recog_data[recog_data['mem_ground_truth'] == 'seen'].copy()

# --- 2b. Build a lookup table: image filename → control condition ---
# We extract Image A and Image B from the main task data (named 'data'), 
# grab their condition, and rename the columns so they match the recognition data.
lookup_A = data[['participant', 'img_A_name', 'control_condition']].copy()
lookup_A = lookup_A.rename(columns={'img_A_name': 'mem_filename'})

lookup_B = data[['participant', 'img_B_name', 'control_condition']].copy()
lookup_B = lookup_B.rename(columns={'img_B_name': 'mem_filename'})

# Combine them and drop any duplicates to make a clean dictionary/lookup table
img_lookup = pd.concat([lookup_A, lookup_B], ignore_index=True).drop_duplicates()

# --- 2c. Merge condition info into the targets dataframe ---
targets = targets.merge(img_lookup, on=['participant', 'mem_filename'], how='left')

# --- 2d. For Analysis 4 (PRIMARY analysis), we only care about
#         whether the trial was HIGH vs LOW control.
#         The 'control_condition' column is the trial_level for controlled items,
#         and 'uncontrolled' for uncontrolled items.
#         (For this primary analysis, we keep both item types but label them by
#          trial condition, not by whether the item itself was controlled.)

# Overwrite the 'control_condition' with the overall trial condition.
# Note: Ensure that 'trial_level' is the exact name of the column from your main data 
# that holds the overarching "high" or "low" status for the whole trial!
targets['control_condition'] = targets['trial_level']

# Convert the yes/no memory response to a binary integer (1 = said "old", 0 = said "new")
# We use .str.lower() just to be safe, in case there are any capitalized "Old" or "New" entries.
# Then we map those text values to integers.
targets['mem_response_bin'] = targets['mem_response'].str.lower().map({'old': 1, 'new': 0})

# =============================================================================
# STEP 3: Prepare the FOIL trials (new items)
# =============================================================================
# "Foils" are images that were NEVER shown during encoding.
# In the recognition file, these are rows where mem_ground_truth == 'unseen'.
#
# The key difference from targets: foils have NO real control condition,
# because they were never part of the main task.
#
# For the interaction model, we need to assign foils to a control condition
# anyway (so the model has something to estimate). We do this with a
# BALANCED DUMMY ASSIGNMENT — split foils 50/50 into "high" and "low"
# per participant. Because the assignment is balanced, it doesn't bias
# the estimate of the interaction, but it IS a modelling choice you
# should report in your methods section.

foils = recog_data[recog_data['mem_ground_truth'] == 'unseen'].copy()

# Sort within participant so the split is deterministic

foils = foils.sort_values(by=['participant', 'mem_filename'])

# Assign dummy control condition: first half of each participant's foils → "high",
# second half → "low"
# this is because the foils were not shown in the control detection task so they dont belong to any condition, however for the model to work we need to assign them artificial conditions

row_numbers = foils.groupby('participant').cumcount()
total_foils = foils.groupby('participant')['mem_ground_truth'].transform('count')
foils['control_condition'] = np.where(row_numbers < (total_foils / 2), 'high', 'low')

# Optional: Print a quick check to verify the split worked!
print("Foil dummy condition assignment complete:")
print(foils[['participant', 'mem_filename', 'control_condition']].head(10))

# =============================================================================
# STEP 4: Combine targets and foils; add contrast codes
# =============================================================================

# 1. Combine targets and foils into a single dataframe
# ignore_index=True ensures the new dataframe has a clean row count from 0 to the end
model_data = pd.concat([targets, foils], ignore_index=True)

# 2. Apply contrast coding for Item Type
# If the item is a target ('seen'), assign +0.5. Otherwise (it's a foil), assign -0.5.
model_data['item_type_c'] = np.where(model_data['mem_ground_truth'] == 'seen', 0.5, -0.5)

# 3. Apply contrast coding for Control Condition
# If the condition is 'high', assign +0.5. Otherwise (it's 'low'), assign -0.5.
model_data['control_c'] = np.where(model_data['control_condition'] == 'high', 0.5, -0.5)

# 4. Convert the finalized pandas dataframe into a polars dataframe for pymer4
model_data_pl = pl.DataFrame(model_data)

# Print a quick check to verify our final dataset is ready!
print("\n--- STEP 4 COMPLETE ---")
print(f"Final modeling dataset created with {len(model_data_pl)} rows.")
print("First 5 rows of contrast codes:")
print(model_data[['mem_ground_truth', 'item_type_c', 'control_condition', 'control_c']].head())