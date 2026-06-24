import os
import sys
import numpy as np
import pandas as pd
import mne

# Force UTF-8 output so special characters (e.g. ✓) print cleanly in
# Windows PowerShell terminals (which default to cp1252/cp850)
sys.stdout.reconfigure(encoding='utf-8')

# ──────────────────────────────────────────────────────────────
# Which participant(s) do you want to process?
# ──────────────────────────────────────────────────────────────

plist = [4,6,7,8,9,10,12,13,14,15,16,17,19,20,21,22]



# CHANGE AS NEEDED:
eeg_path = r"H:\PHD\control_detection\main_data\eeg\eeg3_clean" # clean data is in our CDmem_data cloud
log_path = r"H:\PHD\control_detection\main_data\behavioral"

# Accumulates one summary dict per participant 
# so we can do a cross-participant trial-count check after the loop.
all_summaries = []

# Loop through selected participants
for sub in plist:
    sub_id = f"{sub:04d}"
    epo_file = os.path.join(eeg_path, f"CDmem_{sub_id}-epo.fif")
    
    # Find behavioral log file
    import glob
    log_files = glob.glob(os.path.join(log_path, f"CDmem_1_{sub}*.csv"))
    log_files = [f for f in log_files if 'kinematics' not in f and 'recognition' not in f]

    # Check that files exist
    if not os.path.exists(epo_file):
        print(f"Cleaned epochs file not found: {epo_file}")
        continue
    if not log_files:
        print(f"Behavioral log file not found for participant {sub}")
        continue
        
    log_file = log_files[0]

    print(f"\n{'='*60}")
    print(f"  Participant {sub}")
    print(f"{'='*60}")

    # ── 1. Load data ──────────────────────────────────────────
    # Load cleaned EEG epochs
    epochs = mne.read_epochs(epo_file, preload=True, verbose=False)
    
    # Load behavioral log file
    logdat = pd.read_csv(log_file)
    print(f"  Loaded EEG: {len(epochs)} trials")
    print(f"  Loaded Log: {len(logdat)} trials")

    # ── 2. Match behavioral and EEG ───────────────────────────
    #
    # The epochs in the cleaned file are the motion-onset-trigger trials that
    # survived artifact rejection, in chronological order. The log has ALL trials in the same chronological
    # order, with trigger values.
    #
    # We match by finding which log rows (by their sequential trigger position)
    # correspond to the surviving EEG epochs.
    #
    # NOTE: epochs.selection is NOT used here because it indexes into the
    # original all-markers events array (fixations, responses, etc.), not just
    # the feedback events. Its values are far out of range for logdat.

    # Build a reverse map: event_id int -> trigger number (e.g. 99 -> 99)
    event_id_rev = {v: int(k.split('S ')[1]) for k, v in epochs.event_id.items()}

    # Get the trigger sequence from surviving EEG epochs (chronological order)
    eeg_triggers = np.array([event_id_rev[e] for e in epochs.events[:, 2]])

    # Get the trigger sequence from the full log (chronological order)
    log_triggers = logdat['trigger_motion_start'].values  # shape (352,) for all trials

    # Walk through the log and mark which rows have a matching EEG trigger
    
    eeg_idx = 0  # pointer into eeg_triggers
    log_survived = np.zeros(len(logdat), dtype=bool)

    for log_idx, ltrig in enumerate(log_triggers):
        if eeg_idx < len(eeg_triggers) and eeg_triggers[eeg_idx] == ltrig:
            log_survived[log_idx] = True
            eeg_idx += 1

    if eeg_idx != len(eeg_triggers):
        print(f"  WARNING: Only matched {eeg_idx}/{len(eeg_triggers)} EEG epochs "
              f"to log rows. Check trigger alignment!")
    else:
        print(f"  ✓ All {eeg_idx} EEG epochs matched to log rows via trigger sequence.")

    logdat = logdat[log_survived].copy()
    print(f"  Trials surviving preprocessing: {len(logdat)}")

   

    # ── 3. Sanity Check: Match Triggers ────────────────────
    # Compare logdat.trigger_motion_start with epochs.events[:, 2]
    
    # Let's find the values MNE used for these triggers.
    event_id_rev = {v: k for k, v in epochs.event_id.items()}
    
    mismatch_found = False
    for i in range(len(epochs)):
        eeg_trigger_str = event_id_rev[epochs.events[i, 2]] # e.g. 'Stimulus/S 77'
        eeg_trigger_val = int(eeg_trigger_str.split('S ')[1])
        log_trigger_val = logdat.iloc[i]['trigger_motion_start']
        
        if eeg_trigger_val != log_trigger_val:
            print(f"  ERROR: Mismatch in Trigger Sequence at trial index {i}!")
            print(f"         EEG Trigger: {eeg_trigger_val}, Log Trigger: {log_trigger_val}")
            mismatch_found = True
            break
            
    if not mismatch_found:
        print(f"  ✓ Sanity Check passed: Logfile and EEG triggers match!")

    # ── 4. Narrow time window ──────────────────────────────
    # Following Wen et al., 2017
    epochs.crop(tmin=-0.30, tmax=1.20)
    print(f"  ✓ EEG cropped to window: [-0.30, 1.20] s")

    # ── 5. ERP analysis per condition ─────────────────────
    # Conditions: control_condition
    #   control_condition  'high' = high control   'low' = low control
    # → 2 conditions total: high_control, low_control

    # Output folder — create it if it doesn't exist
    erp_out_path = r"H:\PHD\control_detection\main_data\eeg\eeg4_ERPSummaries"
    os.makedirs(erp_out_path, exist_ok=True)

    # We need a reset positional index on logdat so boolean masks align with
    # epochs (which are already in the same order after HV filtering above).
    logdat_reset = logdat.reset_index(drop=True)

    eeg_dat   = {}   # dict of condition_name → mne.Evoked     
    cond_name = []   # list of condition label strings       
    summary   = {'sub': sub}  # trial counts per condition  

    cnum = 0
    for control in ['high', 'low']:  # high control, low control
        cnum += 1

        # Build condition label, e.g. 'high_control'
        label = f"{control}_control"
        cond_name.append(label)

        # Boolean mask over the current (filtered, preprocessed) trials
        mask = (logdat_reset['control_condition'] == control)

        n_trials = mask.sum()

        # Store trial count in summary
        summary[f"num_{label}"] = int(n_trials)

        if n_trials == 0:
            print(f"  WARNING: no trials for condition {label} — skipping.")
            eeg_dat[label] = None
            continue

        # Select the matching epochs and average across trials → Evoked object
        epochs_cond = epochs[mask.values]
        evoked = epochs_cond.average()
        evoked.comment = label   # label the Evoked so it's identifiable when saved
        eeg_dat[label] = evoked

        print(f"  [{label}]  {n_trials} trials  →  ERP computed")

    # ── 6. Save ERP results ───────────────────────────────

    # Save all 4 Evoked objects in a single FIF file (one file per participant)
    evoked_list = [ev for ev in eeg_dat.values() if ev is not None]
    evoked_file = os.path.join(erp_out_path, f"CDmem_{sub_id}-erp-ave.fif")
    mne.write_evokeds(evoked_file, evoked_list, overwrite=True, verbose=False)
    print(f"  ✓ Saved ERP file: {evoked_file}")

    # Save summary (trial counts per condition) as CSV alongside the ERP file
    summary_file = os.path.join(erp_out_path, f"CDmem_{sub_id}-erp-summary.csv")
    pd.DataFrame([summary]).to_csv(summary_file, index=False)
    print(f"  ✓ Saved summary:  {summary_file}")

    print(f"  DONE! Conditions: {cond_name}  |  Trial counts: "
          f"{ {k: summary[f'num_{k}'] for k in cond_name} }")

    # Append this participant's summary to the cross-participant list
    all_summaries.append(summary)

print("\nAll selected participants processed.")

# ── Cross-participant trial-count check ───────────────────────

# Identifies any participant who has fewer than 25 trials in at least one
# experimental condition — these may need to be excluded from group analysis.
THRESHOLD = 25

if all_summaries:
    summary_df = pd.DataFrame(all_summaries)  # one row per participant

    # Condition columns only (exclude 'sub' identifier column)

    cond_cols = [c for c in summary_df.columns if c.startswith('num_')]
    trial_counts = summary_df[cond_cols]

    # Flag rows where ANY condition has fewer than THRESHOLD trials
    flagged_mask = (trial_counts < THRESHOLD).any(axis=1)
    flagged_subs = summary_df.loc[flagged_mask, 'sub'].tolist()
    num_flagged  = len(flagged_subs)

    print(f"\n{'='*60}")
    print(f"  TRIAL COUNT CHECK  (threshold: < {THRESHOLD} trials per condition)")
    print(f"{'='*60}")
    
    print(f"  Participants with < {THRESHOLD} trials in >= 1 condition: {num_flagged}")

    if num_flagged == 0:
        print("  All participants meet the minimum trial threshold.")
    else:
        
        print("  Flagged participant(s):")
        for psub in flagged_subs:
            psub_id = f"{psub:04d}"
            row = summary_df.loc[summary_df['sub'] == psub, cond_cols].iloc[0]
            low_conds = row[row < THRESHOLD].to_dict()
            print(f"    Sub {psub_id}: {low_conds}")
    print(f"{'='*60}\n")
