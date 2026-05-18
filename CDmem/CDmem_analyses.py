#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CDmem Analysis Script
=====================

Analysis script for the CDmem (Control Detection + Memory) experiment.
Loads all subject data files and performs various analyses.

Data file naming convention: CDmem_1_{participant}.csv
Located in: CDmem/data/subjects/

- add cd accuracy analysis -> see run_cd_accuracy_analysis() below.

- ren et al 2026: Trims RT outliers on a rolling, per-participant basis. Any trial with an RT greater than mean + 3 * SD for that specific participant is discarded. // CDmem_analyses.py: Uses an absolute cutoff across the board. Any recognition trial with an RT greater than 20 seconds is discarded unconditionally (following Haridi et al., 2025).

- TODO 

(supplementary): Add a serial/trial-order analysis of recognition memory. Images are presented
  in counterbalanced blocks (all High trials together, all Low trials together), so there may be
  primacy/recency effects on memory. Adding a trial_position covariate (z-scored serial position
  within block) to Analysis 3 would allow testing whether the control-level effect persists after
  controlling for encoding order.
"""

import os
import sys

# Set R_HOME for pymer4/rpy2 on Windows
os.environ["R_HOME"] = r"C:\Program Files\R\R-4.5.2"

import re
import glob
import numpy as np
import pandas as pd
import polars as pl
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend: writes directly to file, no GUI window.
                       # Ensures plots are saved reliably on Windows regardless of whether
                       # a display is available or image viewer has the file open.
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.stats import norm
from scipy.optimize import curve_fit
from statsmodels.stats.anova import AnovaRM
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from pymer4.models import lmer, glmer

# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR = Path(__file__).parent
# DATA_DIR = SCRIPT_DIR / "data" / "subjects"
DATA_DIR = Path(r"C:\Users\elifg\Desktop\PHD\control_detection\pilot_data")
OUTPUT_DIR          = SCRIPT_DIR / "analysis_output"
POOLED_DIR          = OUTPUT_DIR / "pooled"
PER_PARTICIPANT_DIR = OUTPUT_DIR / "per_participant"
for _d in (OUTPUT_DIR, POOLED_DIR, PER_PARTICIPANT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Participant filter: set to a non-empty list to restrict analyses to specific
# participant IDs, e.g. PARTICIPANT_FILTER = [2, 3].
# Leave as an empty list [] to include ALL participants found in data files.
PARTICIPANT_FILTER = [2,3,4,5,6,7]  # e.g. [2, 3]
# ============================================================================
# DATA LOADING
# ============================================================================

def load_all_data(data_dir=DATA_DIR):
    """
    Load all CDmem subject data files and combine into a single DataFrame.

    File naming pattern: CDmem_1_{participant}.csv
    (e.g. CDmem_1_99.csv for participant 99)

    The function deliberately excludes:
      - CDmem_test.csv style files (intermediate output)
      - CDmem_*_kinematics.csv  (kinematics data - separate analysis)
      - CDmem_*_recognition.csv (recognition data - separate analysis)

    As new participants complete the study and their CSV is placed in the
    data folder, they are automatically included on the next run.
    No code changes are required.

    Parameters
    ----------
    data_dir : Path
        Directory to search for CSV files.

    Returns
    -------
    pd.DataFrame or None
        Combined DataFrame of all loaded participants, or None if no files found.
    """

    pattern = str(data_dir / "CDmem_*.csv")
    all_files = glob.glob(pattern)

    strict_pattern = re.compile(r"^CDmem_1_\d+(?:_\d+)?\.csv$")
    all_files = [
        f for f in all_files
        if strict_pattern.match(os.path.basename(f))
    ]

    if not all_files:
        print(f"[ERROR] No matching data files found in: {data_dir}")
        return None

    print(f"Found {len(all_files)} data file(s):")
    for f in sorted(all_files):
        print(f"  {os.path.basename(f)}")

    dfs = []
    for f in sorted(all_files):
        try:
            df = pd.read_csv(f)
            df["source_file"] = os.path.basename(f)
            dfs.append(df)
        except Exception as e:
            print(f"  [WARNING] Could not load {os.path.basename(f)}: {e}")

    if not dfs:
        print("[ERROR] No files could be loaded.")
        return None

    data = pd.concat(dfs, ignore_index=True)

    if PARTICIPANT_FILTER:
        before = data['participant'].nunique()
        data = data[data['participant'].isin(PARTICIPANT_FILTER)].copy()
        after = data['participant'].nunique()
        print(f"  PARTICIPANT_FILTER applied: retained {after}/{before} participant(s).")
        print(f"  Retained: {sorted(data['participant'].unique().tolist())}\n")

    print(f"\nLoaded {len(dfs)} file(s) -> {len(data)} total rows combined.\n")
    return data


# ============================================================================
# EXCLUSION CRITERION 1 - TIMEOUT RATE
# ============================================================================
#
# Criterion (Haridi et al., 2025):
#   A participant is excluded if 50% or more of their TEST-PHASE trials in
#   EITHER control condition (low or high) are timeout trials.
#
# Relevant columns:
#   'phase'             : 'calibration' or 'test'
#   'control_condition' : 'low' or 'high'
#   'is_timeout'        : True / False (or 'True' / 'False' strings)
#   'participant'       : unique participant identifier
# ============================================================================

TIMEOUT_THRESHOLD = 0.50  # 50% Haridi et al., 2025


def exclude_timeout_participants(data):
    """
    Exclude participants whose timeout rate in the test phase reaches or
    exceeds TIMEOUT_THRESHOLD in either the 'low' or 'high' control condition.

    Parameters
    ----------
    data : pd.DataFrame
        Full combined dataset (all participants, all phases).

    Returns
    -------
    data_clean : pd.DataFrame
        Dataset with flagged participants fully removed (all phases).
    excluded_ids : list
        List of participant IDs excluded by this criterion.
    """

    test_data = data[data["phase"] == "test"].copy()

    test_data["is_timeout"] = test_data["is_timeout"].map(
        lambda x: str(x).strip().lower() == "true"
    )

    timeout_stats = (
        test_data
        .groupby(["participant", "control_condition"])["is_timeout"]
        .agg(
            total_trials="count",
            timeout_trials="sum"
        )
        .reset_index()
    )

    timeout_stats["timeout_rate"] = (
        timeout_stats["timeout_trials"] / timeout_stats["total_trials"]
    )

    flagged = timeout_stats[timeout_stats["timeout_rate"] >= TIMEOUT_THRESHOLD]
    excluded_ids = flagged["participant"].unique().tolist()

    print("=" * 60)
    print("EXCLUSION CRITERION 1: Timeout Rate")
    print(f"  Threshold : >= {TIMEOUT_THRESHOLD * 100:.0f}% timeouts in either condition")
    print(f"  Reference : Haridi et al., 2025")
    print("=" * 60)

    if len(excluded_ids) == 0:
        print("  -> No participants excluded by this criterion.\n")
    else:
        print(f"  -> {len(excluded_ids)} participant(s) excluded:\n")
        for _, row in flagged.iterrows():
            print(
                f"     Participant {row['participant']} | "
                f"Condition: {row['control_condition']} | "
                f"Timeout rate: {row['timeout_rate']:.1%} "
                f"({int(row['timeout_trials'])}/{int(row['total_trials'])} trials)"
            )
        print()

    data_clean = data[~data["participant"].isin(excluded_ids)].copy()

    return data_clean, excluded_ids


# ============================================================================
# EXCLUSION CRITERION 2: ACCURACY OUTLIERS
# ============================================================================
#
# Criterion:
#   Expected mean accuracy is 55% for the LOW control condition and 85% for
#   the HIGH control condition (reflecting the QUEST staircase targets).
#   A participant is excluded if their accuracy in EITHER condition is more
#   than 2.5 standard deviations away from the GROUP mean.
#
# Relevant columns:
#   'phase'              : we look only at test-phase trials
#   'control_condition'  : 'low' or 'high'
#   'detection_accuracy' : 0 = incorrect, 1 = correct
#   'participant'        : unique participant identifier
# ============================================================================

EXPECTED_ACCURACY = {
    "low":  0.55,
    "high": 0.85,
}

ACCURACY_SD_THRESHOLD = 2.5


def exclude_accuracy_outliers(data):
    """
    Exclude participants whose mean test-phase accuracy in either control
    condition deviates more than ACCURACY_SD_THRESHOLD SDs from the group mean.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset after previous exclusion steps.

    Returns
    -------
    data_clean : pd.DataFrame
    excluded_ids : list
    report_df : pd.DataFrame
    """

    test_data = data[data["phase"] == "test"].copy()

    acc_per_px = (
        test_data
        .groupby(["participant", "control_condition"])["detection_accuracy"]
        .mean()
        .reset_index()
        .rename(columns={"detection_accuracy": "mean_accuracy"})
    )

    group_stats = (
        acc_per_px
        .groupby("control_condition")["mean_accuracy"]
        .agg(group_mean="mean", group_sd="std")
        .reset_index()
    )

    acc_per_px = acc_per_px.merge(group_stats, on="control_condition")

    acc_per_px["z_score"] = (
        (acc_per_px["mean_accuracy"] - acc_per_px["group_mean"])
        / acc_per_px["group_sd"]
    )

    flagged = acc_per_px[acc_per_px["z_score"].abs() > ACCURACY_SD_THRESHOLD]
    excluded_ids = flagged["participant"].unique().tolist()

    print("=" * 60)
    print("EXCLUSION CRITERION 2: Accuracy Outliers")
    print(f"  Threshold : |z| > {ACCURACY_SD_THRESHOLD} SD from group mean")
    print(f"  Expected accuracies: "
          f"low = {EXPECTED_ACCURACY['low']*100:.0f}%, "
          f"high = {EXPECTED_ACCURACY['high']*100:.0f}%")
    print("=" * 60)

    print("\n  Per-participant accuracy by condition (with group z-scores):\n")
    print(f"  {'Participant':<15} {'Condition':<10} {'Accuracy':>10} "
          f"{'Group Mean':>12} {'Group SD':>10} {'z-score':>9}")
    print("  " + "-" * 70)
    for _, row in acc_per_px.sort_values(["control_condition", "participant"]).iterrows():
        flag = " *** EXCLUDED" if row["participant"] in excluded_ids else ""
        print(
            f"  {str(row['participant']):<15} {row['control_condition']:<10} "
            f"{row['mean_accuracy']:>10.3f} {row['group_mean']:>12.3f} "
            f"{row['group_sd']:>10.3f} {row['z_score']:>9.2f}{flag}"
        )

    print()
    if len(excluded_ids) == 0:
        print("  -> No participants excluded by this criterion.\n")
    else:
        print(f"  -> {len(excluded_ids)} participant(s) excluded: {excluded_ids}\n")

    assert data is not None
    data_clean = data[~data["participant"].isin(excluded_ids)].copy()

    return data_clean, excluded_ids, acc_per_px


# ============================================================================
# RECOGNITION DATA LOADING
# ============================================================================

def load_recognition_data(data_dir=DATA_DIR):
    """
    Load all CDmem recognition CSV files and combine into a single DataFrame.

    File naming pattern: CDmem_{participant}_{session}_{block}_recognition.csv

    Parameters
    ----------
    data_dir : Path

    Returns
    -------
    pd.DataFrame or None
    """

    pattern = str(Path(data_dir) / "CDmem_*_recognition.csv")
    all_files = glob.glob(pattern)

    # Keep only files matching the canonical pattern: CDmem_1_<number>_recognition.csv or CDmem_1_<number>_<session>_recognition.csv
    strict_pattern = re.compile(r"^CDmem_1_\d+(?:_\d+)?_recognition\.csv$")
    all_files = [
        f for f in all_files
        if strict_pattern.match(os.path.basename(f))
    ]

    if not all_files:
        print(f"[ERROR] No recognition files found in: {data_dir}")
        return None

    print(f"Found {len(all_files)} recognition file(s):")
    for f in sorted(all_files):
        print(f"  {os.path.basename(f)}")

    dfs = []
    for f in sorted(all_files):
        try:
            df = pd.read_csv(f)
            df["source_file"] = os.path.basename(f)
            dfs.append(df)
        except Exception as e:
            print(f"  [WARNING] Could not load {os.path.basename(f)}: {e}")

    if not dfs:
        print("[ERROR] No recognition files could be loaded.")
        return None

    recog_data = pd.concat(dfs, ignore_index=True)
    print(f"\nLoaded {len(dfs)} recognition file(s) -> {len(recog_data)} total rows.\n")
    return recog_data


# ============================================================================
# RECOGNITION EXCLUSION: LONG RT TRIALS
# ============================================================================
#
# Criterion (Ren et al., 2026):
#   Recognition trials with a response time (mem_rt) greater than each
#   participant's mean recognition RT + 3 * SD are excluded at the TRIAL
#   level (not the participant level).
# ============================================================================

RT_SD_MULTIPLIER = 3.0


def exclude_long_rt_trials(recog_data):
    """
    Remove individual recognition trials where mem_rt exceeds each
    participant's mean RT + RT_SD_MULTIPLIER * SD.

    Parameters
    ----------
    recog_data : pd.DataFrame

    Returns
    -------
    recog_clean : pd.DataFrame
    n_excluded_trials : int
    exclusion_details : pd.DataFrame
    """

    recog_data = recog_data.copy()
    recog_data["mem_rt"] = pd.to_numeric(recog_data["mem_rt"], errors="coerce")

    # Compute per-participant mean and SD of mem_rt
    px_stats = (
        recog_data
        .groupby("participant")["mem_rt"]
        .agg(px_mean="mean", px_sd="std")
        .reset_index()
    )
    recog_data = recog_data.merge(px_stats, on="participant", how="left")
    recog_data["rt_threshold"] = recog_data["px_mean"] + RT_SD_MULTIPLIER * recog_data["px_sd"]

    long_rt_mask = recog_data["mem_rt"] > recog_data["rt_threshold"]
    missing_rt_mask = recog_data["mem_rt"].isna()
    exclude_mask = long_rt_mask | missing_rt_mask

    exclusion_details = recog_data[exclude_mask][
        ["participant", "overall_trial_num", "mem_rt", "rt_threshold"]
    ].copy()
    n_excluded_trials = len(exclusion_details)

    print("=" * 60)
    print("RECOGNITION EXCLUSION: Long RT Trials")
    print(f"  RT cutoff : per-participant mean + {RT_SD_MULTIPLIER:.0f} SD")
    print(f"  Reference : Ren et al., 2026")
    print("=" * 60)

    if n_excluded_trials == 0:
        print("  -> No trials excluded by this criterion.\n")
    else:
        print(f"  -> {n_excluded_trials} trial(s) excluded:\n")
        for _, row in exclusion_details.iterrows():
            print(
                f"     Participant {row['participant']} | "
                f"Trial {int(row['overall_trial_num'])} | "
                f"RT = {row['mem_rt']:.2f} s "
                f"(threshold = {row['rt_threshold']:.2f} s)"
            )
        print()

    recog_clean = recog_data[~exclude_mask].drop(
        columns=["px_mean", "px_sd", "rt_threshold"]
    ).copy()

    return recog_clean, n_excluded_trials, exclusion_details


# ============================================================================
# CROSS-DATASET PARTICIPANT SYNCHRONISATION
# ============================================================================

def sync_participant_ids(data, recog_data):
    """
    Ensure both DataFrames contain exactly the same set of participant IDs.

    Parameters
    ----------
    data : pd.DataFrame
    recog_data : pd.DataFrame

    Returns
    -------
    data_synced : pd.DataFrame
    recog_synced : pd.DataFrame
    """

    main_pxs  = set(data['participant'].unique())
    recog_pxs = set(recog_data['participant'].unique())
    common_pxs = main_pxs & recog_pxs
    only_in_main  = main_pxs  - recog_pxs
    only_in_recog = recog_pxs - main_pxs

    print("=" * 60)
    print("CROSS-DATASET PARTICIPANT SYNCHRONISATION")
    print("=" * 60)

    if not only_in_main and not only_in_recog:
        print("  Both datasets already contain the same participant IDs.")
        print(f"  Participants in both: {sorted(common_pxs)}\n")
    else:
        if only_in_main:
            print(f"  In main data but NOT in recognition (removed from main data):")
            print(f"    {sorted(only_in_main)}")
        if only_in_recog:
            print(f"  In recognition data but NOT in main data (removed from recog_data):")
            print(f"    {sorted(only_in_recog)}")
        print(f"  Participants retained in both: {sorted(common_pxs)}\n")

    data_synced  = data[data['participant'].isin(common_pxs)].copy()
    recog_synced = recog_data[recog_data['participant'].isin(common_pxs)].copy()

    return data_synced, recog_synced


# ============================================================================
# HELPER: d-prime calculation
# ============================================================================

def calc_dprime(hr, far, clip_val=0.01):
    """
    Calculate d-prime from hit rate (hr) and false alarm rate (far).
    Includes clipping to avoid infinity when hr=1.0 or far=0.0.
    """
    hr_clipped  = np.clip(hr,  clip_val, 1.0 - clip_val)
    far_clipped = np.clip(far, clip_val, 1.0 - clip_val)
    return norm.ppf(hr_clipped) - norm.ppf(far_clipped)


# ============================================================================
# HELPER: fit GLMM with maximal random effects, fall back to intercept-only
# ============================================================================

def fit_glmm_with_fallback(formula_maximal, formula_minimal, data_pl, family="binomial"):
    """
    Attempt to fit a GLMM with maximal random effects structure.
    If the model produces a singular fit warning, fall back to a random-intercept
    only model and report which structure was used.

    Singular fit is detected by checking whether any random-effect variance
    components are estimated at (or very near) zero.

    Parameters
    ----------
    formula_maximal : str
        Full model formula including random slopes (e.g. "(1 + x | subj)").
    formula_minimal : str
        Fallback formula with random intercept only (e.g. "(1 | subj)").
    data_pl : pl.DataFrame
        Data in Polars format (required by pymer4 >= 0.8).
    family : str
        Model family (default: "binomial").

    Returns
    -------
    model : fitted pymer4 model
    random_structure_used : str
        Either "maximal" or "intercept-only", for transparent reporting.
    """
    print(f"  Attempting maximal model: {formula_maximal}")
    try:
        model = glmer(formula_maximal, data=data_pl, family=family)
        result = model.fit()

        # Check for singular fit: any variance component near zero
        singular = False
        if hasattr(model, 'ranef_var') and model.ranef_var is not None:
            variances = model.ranef_var.select(pl.col("^Var.*$")).to_numpy().flatten()
            if np.any(variances < 1e-6):
                singular = True

        if singular:
            print("  [WARNING] Singular fit detected in maximal model.")
            print("  Falling back to random-intercept-only model.")
            raise ValueError("Singular fit")

        print("  -> Maximal random effects model converged without singular fit.")
        print(result)
        return model, "maximal"

    except Exception as e:
        if "Singular fit" not in str(e):
            print(f"  [WARNING] Maximal model failed ({e}). Falling back to intercept-only.")

    print(f"  Fitting intercept-only model: {formula_minimal}")
    try:
        model = glmer(formula_minimal, data=data_pl, family=family)
        result = model.fit()
        print("  -> Intercept-only model used (report this in methods).")
        print(result)
        return model, "intercept-only"
    except Exception as e:
        print(f"  [ERROR] Intercept-only model also failed: {e}")
        return None, None


# ============================================================================
# RECOGNITION DATA PREPARATION
# ============================================================================

def analyze_recognition(main_data, recog_data):
    """
    Calculate recognition performance (HR, FA, d-prime) per participant
    and per control condition (including uncontrolled items).

    Returns
    -------
    mem_results : pd.DataFrame
        Participant-level summary (primary: high vs. low vs. uncontrolled).
    targets : pd.DataFrame
        Trial-level recognition data for old items with condition labels.
    supp_mem_results : pd.DataFrame
        Participant-level summary for 2x2 (trial_level x item_type).
    """
    recog_data = recog_data.copy()
    
    # 1. Convert "yes"/"no" responses to True/False for calculation
    recog_data['said_old'] = recog_data['mem_response'].str.lower() == 'yes'

    # 2. Calculate FALSE ALARM RATE (Noise Baseline)
    foils = recog_data[recog_data['mem_ground_truth'] == 'unseen']
    
    if len(foils) == 0:
        print("[WARNING] No foil ('unseen') trials found in recognition data.")
        return None, None, None
        
    fa_rates = foils.groupby('participant')['said_old'].mean().reset_index()
    fa_rates.rename(columns={'said_old': 'FA_rate'}, inplace=True)

    # 3. Process targets
    targets = recog_data[recog_data['mem_ground_truth'] == 'seen'].copy()
    
    # Build image -> condition lookup from main_data to get encoding RTs and handle fallbacks
    if 'phase' in main_data.columns:
        test_data = main_data[main_data['phase'] == 'test'].copy()
    else:
        test_data = main_data.copy()
        
    img_lookups = []
    for _, row in test_data.iterrows():
        px       = row['participant']
        cond     = row['control_condition']
        target_img = row['true_controlled']

        _agency = row['agency_rating'] if 'agency_rating' in row.index and pd.notna(row.get('agency_rating')) else np.nan
        img_lookups.append({
            'participant':      px,
            'mem_filename':     row['img_A_name' if target_img == 'img_A' else 'img_B_name'],
            'trial_level':      cond,
            'item_type':        'controlled',
            'agency_rating':    _agency
        })
        img_lookups.append({
            'participant':      px,
            'mem_filename':     row['img_B_name' if target_img == 'img_A' else 'img_A_name'],
            'trial_level':      cond,
            'item_type':        'uncontrolled',
            'agency_rating':    _agency
        })
    img_lookup = pd.DataFrame(img_lookups)
    img_lookup.dropna(subset=['mem_filename'], inplace=True)
    img_lookup.drop_duplicates(subset=['participant', 'mem_filename'], inplace=True)

    # Check if we have the new logging (item_type/trial_level) in the CSV
    has_tracking = 'item_type' in targets.columns and 'trial_level' in targets.columns
    
    if not has_tracking:
        print("[INFO] Recognition data lacks condition tracking columns. Falling back to filename matching.")
        targets = targets.merge(img_lookup, on=['participant', 'mem_filename'], how='left')
    else:
        # Merge agency_rating if tracking columns already exist
        targets = targets.merge(
            img_lookup[['participant', 'mem_filename', 'agency_rating']],
            on=['participant', 'mem_filename'],
            how='left'
        )

    # Derive 'control_condition' exactly as the old script expected for PRIMARY analysis:
    # If controlled -> 'high' or 'low'. If uncontrolled -> 'uncontrolled'
    targets['control_condition'] = targets.apply(
        lambda row: row['trial_level'] if row['item_type'] == 'controlled' else 'uncontrolled',
        axis=1
    )

    # 4. Primary Hit Rates
    hit_rates = (
        targets
        .groupby(['participant', 'control_condition'])['said_old']
        .mean()
        .reset_index()
        .rename(columns={'said_old': 'Hit_rate'})
    )

    results = hit_rates.merge(fa_rates, on='participant', how='left')
    results['d_prime'] = results.apply(
        lambda row: calc_dprime(row['Hit_rate'], row['FA_rate']), axis=1
    )

    # 5. Supplementary Analysis 2x2 Hit Rates (trial_level x item_type)
    results_supp = (
        targets.groupby(['participant', 'trial_level', 'item_type'])['said_old']
        .mean().reset_index().rename(columns={'said_old': 'Hit_rate'})
        .merge(fa_rates, on='participant', how='left')
    )
    results_supp['d_prime'] = results_supp.apply(
        lambda row: calc_dprime(row['Hit_rate'], row['FA_rate']), axis=1
    )

    return results, targets, results_supp


# ==============================================================================
# PRIMARY ANALYSES (High vs. Low Control)
# ==============================================================================

# ------------------------------------------------------------------------------
# ANALYSIS 1: Paired t-test on d-prime
# ------------------------------------------------------------------------------

def run_analysis_1_dprime_ttest(mem_results):
    """
    Paired t-test comparing d-prime between high and low control conditions.
    """
    if mem_results is None:
        print("[WARNING] mem_results is None. Skipping Analysis 1.")
        return None

    pivoted = mem_results.pivot(index='participant', columns='control_condition', values='d_prime')

    if 'high' not in pivoted.columns or 'low' not in pivoted.columns:
        print("[WARNING] Missing condition data for d-prime t-test.")
        return None

    pivoted = pivoted.dropna(subset=['high', 'low'])

    t_stat, p_val = stats.ttest_rel(pivoted['high'], pivoted['low'])
    diff = pivoted['high'] - pivoted['low']
    cohens_d = diff.mean() / (diff.std() + 1e-9)

    return {
        'analysis':   'Analysis 1: d-prime paired t-test',
        't_stat':     t_stat,
        'p_val':      p_val,
        'mean_high':  pivoted['high'].mean(),
        'mean_low':   pivoted['low'].mean(),
        'sd_high':    pivoted['high'].std(),
        'sd_low':     pivoted['low'].std(),
        'cohens_d':   cohens_d,
        'n':          len(pivoted)
    }


# ------------------------------------------------------------------------------
# ANALYSIS 2: Paired t-test on hit rates
# ------------------------------------------------------------------------------

def run_analysis_2_hitrate_ttest(mem_results):
    """
    Paired t-test comparing hit rates between high and low control conditions.
    """
    if mem_results is None:
        print("[WARNING] mem_results is None. Skipping Analysis 2.")
        return None

    pivoted = mem_results.pivot(index='participant', columns='control_condition', values='Hit_rate')

    if 'high' not in pivoted.columns or 'low' not in pivoted.columns:
        print("[WARNING] Missing condition data for hit rate t-test.")
        return None

    pivoted = pivoted.dropna(subset=['high', 'low'])

    t_stat, p_val = stats.ttest_rel(pivoted['high'], pivoted['low'])
    diff = pivoted['high'] - pivoted['low']
    cohens_d = diff.mean() / (diff.std() + 1e-9)

    return {
        'analysis':   'Analysis 2: Hit rate paired t-test',
        't_stat':     t_stat,
        'p_val':      p_val,
        'mean_high':  pivoted['high'].mean(),
        'mean_low':   pivoted['low'].mean(),
        'sd_high':    pivoted['high'].std(),
        'sd_low':     pivoted['low'].std(),
        'cohens_d':   cohens_d,
        'n':          len(pivoted)
    }


# ------------------------------------------------------------------------------
# ANALYSIS 3: GLMM on target trials only (High vs. Low)
# ------------------------------------------------------------------------------

def run_analysis_3_glmm(targets):
    """
    Trial-level GLMM (Binomial, logit link) on target (old) trials only.

    Model formula:
      Maximal : said_old_int ~ control_c + (1 + control_c | participant)
      Fallback: said_old_int ~ control_c + (1 | participant)

    control_c is contrast-coded: High = +0.5, Low = -0.5.
    The maximal model is attempted first; if singular, falls back to
    intercept-only and reports which was used.
    """
    if targets is None or len(targets) == 0:
        print("[WARNING] No target data for Analysis 3. Skipping.")
        return None

    df = targets.copy()
    df['said_old_int'] = df['said_old'].astype(int)

    # Contrast code: High = +0.5, Low = -0.5
    df['control_c'] = df['control_condition'].map({'high': 0.5, 'low': -0.5})
    df = df.dropna(subset=['said_old_int', 'control_c', 'participant'])

    if len(df) == 0:
        print("[WARNING] No valid trials after preprocessing for Analysis 3.")
        return None

    print("\nAnalysis 3: GLMM (Binomial, logit link) on target trials (High vs Low)...")
    print(f"  Trials: {len(df)} | Participants: {df['participant'].nunique()}")
    print(f"  Contrast coding: High = +0.5, Low = -0.5")

    df_pl = pl.from_pandas(df)

    model, structure = fit_glmm_with_fallback(
        formula_maximal="said_old_int ~ control_c + (1 + control_c | participant)",
        formula_minimal="said_old_int ~ control_c + (1 | participant)",
        data_pl=df_pl,
        family="binomial"
    )
    if model is not None:
        print(f"  [REPORT] Random effects structure used: {structure}")
    return model


# ------------------------------------------------------------------------------
# ANALYSIS 4: GLMM Interaction (item_type x control) on all trials
# ------------------------------------------------------------------------------

def run_analysis_4_interaction_glmm(targets, recog_data):
    """
    Trial-level GLMM (Binomial, logit link) on ALL trials (targets + foils).

    Model formula:
      Maximal : said_old_int ~ item_type_c * control_c + (1 + item_type_c * control_c | participant)
      Fallback: said_old_int ~ item_type_c * control_c + (1 | participant)

    item_type_c  : Target = +0.5, Foil = -0.5 (contrast-coded)
    control_c    : High   = +0.5, Low  = -0.5 (contrast-coded)

    NOTE: For foils, control_c is assigned via a balanced alternating split
    (even trials = +0.5, odd = -0.5) because foils have no experimental
    condition. This dummy assignment is balanced by design so it does not
    systematically bias the foil main effect, but the control_c term for
    foils is not experimentally meaningful. The interaction is therefore
    driven primarily by the target trials. This should be noted in reporting.
    """
    if targets is None or recog_data is None:
        print("[WARNING] Missing data for Analysis 4. Skipping.")
        return None

    # Prepare foils with balanced dummy control assignment.
    # Foils were never shown during the control task and have no genuine
    # condition. We split them equally (first half +0.5, second half -0.5)
    # per participant. Because the FA rate is identical for both dummy
    # conditions by construction, this does not bias the interaction
    # estimate — but it is an approximation (see allpossible_power_CDmem.py).
    foils = recog_data[recog_data['mem_ground_truth'] == 'unseen'].copy()
    if len(foils) == 0:
        print("[WARNING] No foil data for Analysis 4.")
        return None

    foils['said_old_int'] = (foils['mem_response'].str.lower() == 'yes').astype(int)
    foils['item_type_c']  = -0.5
    foils = foils.sort_values(['participant', 'overall_trial_num'])

    def _assign_foil_control_c(px_df):
        """First half of foils per participant get +0.5, second half -0.5."""
        n = len(px_df)
        half = n // 2
        codes = np.array([0.5] * half + [-0.5] * (n - half))
        return pd.Series(codes, index=px_df.index)

    foils['_rank'] = foils.groupby('participant').cumcount()
    foils['_n']    = foils.groupby('participant')['_rank'].transform('count')
    foils['control_c'] = np.where(foils['_rank'] < foils['_n'] // 2, 0.5, -0.5)
    foils.drop(columns=['_rank', '_n'], inplace=True)

    # Prepare targets with real contrast codes
    targets_df = targets.copy()
    targets_df['said_old_int'] = targets_df['said_old'].astype(int)
    targets_df['item_type_c']  = 0.5
    targets_df['control_c']    = targets_df['control_condition'].map({'high': 0.5, 'low': -0.5})

    cols = ['participant', 'said_old_int', 'item_type_c', 'control_c']
    df   = pd.concat([targets_df[cols], foils[cols]], ignore_index=True)
    df   = df.dropna()

    if len(df) == 0:
        print("[WARNING] No valid trials for Analysis 4.")
        return None

    print("\nAnalysis 4: GLMM Interaction (item_type x control) on all trials...")
    print(f"  Trials: {len(df)} (T={len(targets_df)}, F={len(foils)}) | Pxs: {df['participant'].nunique()}")
    print(f"  Contrast coding: Target=+0.5/Foil=-0.5, High=+0.5/Low=-0.5")
    print(f"  NOTE: control_c for foils is a balanced dummy assignment (not experimental).")
    print(f"  Target cell counts (item_type_c x control_c):")
    for (it, cc), grp in targets_df.groupby(['item_type_c', 'control_c']):
        lbl_it = 'Target' if it > 0 else 'Foil'
        lbl_cc = 'High'   if cc > 0 else 'Low'
        print(f"    {lbl_it} / {lbl_cc}: N trials={len(grp)}, N pxs={grp['participant'].nunique()}")

    df_pl = pl.from_pandas(df)

    model, structure = fit_glmm_with_fallback(
        formula_maximal="said_old_int ~ item_type_c * control_c + (1 + item_type_c * control_c | participant)",
        formula_minimal="said_old_int ~ item_type_c * control_c + (1 | participant)",
        data_pl=df_pl,
        family="binomial"
    )
    if model is not None:
        print(f"  [REPORT] Random effects structure used: {structure}")

    # -------------------------------------------------------------------------
    # ALTERNATIVE APPROACH: NESTED INTERACTION MODEL (Ren et al., 2026)
    # -------------------------------------------------------------------------
    # Why this was added: 
    # To provide a comparison between the full-factorial sum-coded model (above) 
    # and a nested/conditional effects model commonly used in literature (e.g., Ren et al., 2026).
    #
    # PROS of the Full-Factorial Approach (current active model):
    # - Matches standard ANOVA logic perfectly (orthogonal main effects and interactions).
    # - Because of the (-0.5, 0.5) dummy balancing on foils, the main effect of item_type_c 
    #   directly tests overall memory sensitivity (Hits vs FAs, i.e., overall d-prime) 
    #   independently of the control condition.
    # - Strict test: forces the model to prove that the control condition changes Hit Rates 
    #   significantly more than it randomly fluctuates False Alarm rates.
    #
    # CONS of the Full-Factorial Approach:
    # - Relies on a "dummy hack": Foils are artificially split into "High" and "Low" 
    #   conditions they never experienced, which can be unintuitive.
    #
    # PROS of the Nested/Conditional Approach (R script style below):
    # - Conceptually cleaner and strictly logical: Foils (new items) do not have an 
    #   encoding condition, so the model only estimates the control effect where it 
    #   genuinely exists (on old items).
    # - The interaction coefficient directly corresponds to the effect of control on Hit Rates.
    # - No artificial dummy data manipulation for foils.
    #
    # CONS of the Nested/Conditional Approach:
    # - You don't get a single, pure "Main Effect of Memory" (a direct analogue to overall d-prime) 
    #   because the baseline shifts. Doesn't strictly test condition differences relative to FAs.
    # -------------------------------------------------------------------------
    #
    # # Code to run the alternative nested model (mimicking R script `is_old + is_old:control_val`):
    # alt_foils = foils.copy()
    # alt_foils['item_is_old'] = 0
    # alt_foils['control_val'] = 0  # Foils have no condition
    # 
    # alt_targets = targets_df.copy()
    # alt_targets['item_is_old'] = 1
    # alt_targets['control_val'] = alt_targets['control_condition'].map({'high': 0.5, 'low': -0.5})
    # 
    # cols_alt = ['participant', 'said_old_int', 'item_is_old', 'control_val']
    # df_alt = pd.concat([alt_targets[cols_alt], alt_foils[cols_alt]], ignore_index=True).dropna()
    # 
    # print("\nAlternative Analysis 4: Nested Interaction GLMM...")
    # alt_model, alt_structure = fit_glmm_with_fallback(
    #     formula_maximal="said_old_int ~ item_is_old + item_is_old:control_val + (1 + item_is_old:control_val | participant)",
    #     formula_minimal="said_old_int ~ item_is_old + item_is_old:control_val + (1 | participant)",
    #     data_pl=pl.from_pandas(df_alt),
    #     family="binomial"
    # )
    
    return model


# ==============================================================================
# SUPPLEMENTARY ANALYSES (2x2: Trial Level x Item Type)
# ==============================================================================

def _check_2x2_completeness(df, subject_col='participant',
                             factor1='trial_level', factor2='item_type'):
    """
    Return the subset of participants who have exactly one observation in each
    of the four cells of the 2x2 design. Uses per-cell grouping rather than a
    simple row count, which would pass malformed data silently.

    Parameters
    ----------
    df : pd.DataFrame
    subject_col : str
    factor1, factor2 : str

    Returns
    -------
    valid_subjs : pd.Index
    df_valid : pd.DataFrame
    """
    valid_mask = (
        df.groupby(subject_col)
        .apply(lambda x: x.groupby([factor1, factor2]).ngroups == 4)
    )
    valid_subjs = valid_mask[valid_mask].index
    df_valid    = df[df[subject_col].isin(valid_subjs)]
    return valid_subjs, df_valid


# ------------------------------------------------------------------------------
# SUPPLEMENTARY 1: 2x2 Repeated-Measures ANOVA on d-prime
# ------------------------------------------------------------------------------

def run_supp_analysis_1_dprime_2x2_anova(supp_mem_results):
    """
    2x2 RM ANOVA on d-prime: Trial Level (High vs Low) x Item Type (Controlled vs Uncontrolled).
    """
    if supp_mem_results is None:
        print("[WARNING] supp_mem_results is None. Skipping Supp Analysis 1.")
        return None

    df_anova = supp_mem_results.dropna(subset=['d_prime', 'trial_level', 'item_type'])
    valid_subjs, df_anova = _check_2x2_completeness(df_anova)

    if len(valid_subjs) < 2:
        print("[WARNING] Not enough participants with all 4 cells for Supp Analysis 1.")
        return None

    print(f"\nSupplementary Analysis 1: 2x2 RM ANOVA on d-prime (N={len(valid_subjs)})...")
    try:
        anova = AnovaRM(
            data=df_anova, depvar='d_prime',
            subject='participant', within=['trial_level', 'item_type']
        )
        res = anova.fit()
        print(res.summary())
        return res
    except Exception as e:
        print(f"  [ERROR] AnovaRM failed: {e}")
        return None


# ------------------------------------------------------------------------------
# SUPPLEMENTARY 2: 2x2 Repeated-Measures ANOVA on hit rates
# ------------------------------------------------------------------------------

def run_supp_analysis_2_hitrate_2x2_anova(supp_mem_results):
    """
    2x2 RM ANOVA on hit rates: Trial Level (High vs Low) x Item Type (Controlled vs Uncontrolled).

    Note: Hit rates are bounded [0,1]; the linear model assumption of AnovaRM
    is approximate. The GLMM in Supp Analysis 3 is the primary trial-level test;
    this ANOVA is retained as a summary-statistic complement.
    """
    if supp_mem_results is None:
        print("[WARNING] supp_mem_results is None. Skipping Supp Analysis 2.")
        return None

    df_anova = supp_mem_results.dropna(subset=['Hit_rate', 'trial_level', 'item_type'])
    valid_subjs, df_anova = _check_2x2_completeness(df_anova)

    if len(valid_subjs) < 2:
        print("[WARNING] Not enough participants with all 4 cells for Supp Analysis 2.")
        return None

    print(f"\nSupplementary Analysis 2: 2x2 RM ANOVA on hit rates (N={len(valid_subjs)})...")
    try:
        anova = AnovaRM(
            data=df_anova, depvar='Hit_rate',
            subject='participant', within=['trial_level', 'item_type']
        )
        res = anova.fit()
        print(res.summary())
        return res
    except Exception as e:
        print(f"  [ERROR] AnovaRM failed: {e}")
        return None


# ------------------------------------------------------------------------------
# SUPPLEMENTARY 3: 2x2 GLMM on all old items (targets only)
# ------------------------------------------------------------------------------

def run_supp_analysis_3_glmm_2x2(targets):
    """
    Trial-level 2x2 GLMM on target (old) items.

    Model formula:
      Maximal : said_old_int ~ trial_level_c * item_type_c + (1 + trial_level_c * item_type_c | participant)
      Fallback: said_old_int ~ trial_level_c * item_type_c + (1 | participant)

    trial_level_c : High = +0.5, Low = -0.5 (contrast-coded)
    item_type_c   : Controlled = +0.5, Uncontrolled = -0.5 (contrast-coded)

    Main effects are interpretable at the grand mean (not at a reference level).
    The interaction tests whether the trial-level control effect differs between
    controlled and uncontrolled items.
    """
    if targets is None or len(targets) == 0:
        print("[WARNING] targets is None or empty. Skipping Supp Analysis 3.")
        return None

    df = targets.copy()
    df['said_old_int'] = df['said_old'].astype(int)

    # Contrast codes
    df['trial_level_c'] = df['trial_level'].map({'high': 0.5, 'low': -0.5})
    df['item_type_c']   = df['item_type'].map({'controlled': 0.5, 'uncontrolled': -0.5})

    df = df.dropna(subset=['said_old_int', 'trial_level_c', 'item_type_c', 'participant'])

    if len(df) == 0:
        print("[WARNING] No valid trials after preprocessing for Supp Analysis 3.")
        return None

    print("\nSupplementary Analysis 3: 2x2 GLMM on all old items...")
    print(f"  Trials: {len(df)} | Participants: {df['participant'].nunique()}")
    print(f"  Contrast coding: High=+0.5/Low=-0.5, Controlled=+0.5/Uncontrolled=-0.5")

    df_pl = pl.from_pandas(df)

    model, structure = fit_glmm_with_fallback(
        formula_maximal="said_old_int ~ trial_level_c * item_type_c + (1 + trial_level_c * item_type_c | participant)",
        formula_minimal="said_old_int ~ trial_level_c * item_type_c + (1 | participant)",
        data_pl=df_pl,
        family="binomial"
    )
    if model is not None:
        print(f"  [REPORT] Random effects structure used: {structure}")
    return model


# ------------------------------------------------------------------------------
# SUPPLEMENTARY 4: GLMM Interaction (Is_Old x Trial_Level x Item_Type)
# ------------------------------------------------------------------------------

def run_supp_analysis_4_glmm_foils(targets, recog_data):
    """
    Trial-level GLMM on ALL trials (targets + foils) with 3-way interaction.

    Model formula:
      Maximal : said_old_int ~ item_is_old_c * trial_level_c * item_type_c + (1 + item_is_old_c | participant)
      Fallback: said_old_int ~ item_is_old_c * trial_level_c * item_type_c + (1 | participant)

    item_is_old_c : Old = +0.5, Foil = -0.5 (contrast-coded)
    trial_level_c : High = +0.5, Low = -0.5 (contrast-coded)
    item_type_c   : Controlled = +0.5, Uncontrolled = -0.5 (contrast-coded)

    NOTE: For foils, trial_level_c and item_type_c are assigned via a balanced
    rotating dummy scheme (cycling through all 4 combinations). These assignments
    are not experimentally meaningful for foils. The interaction terms involving
    item_is_old_c are the theoretically relevant estimates; they test whether the
    2x2 target structure differs from the foil baseline.
    """
    if targets is None or recog_data is None:
        print("[WARNING] Missing data for Supp Analysis 4. Skipping.")
        return None

    # Prepare foils with balanced dummy 2x2 assignment.
    # Foils are split into four equal batches per participant, one for each
    # cell of the 2x2: (high, controlled), (high, uncontrolled),
    # (low, controlled), (low, uncontrolled). Because the FA rate is the same
    # across all four dummy cells by construction, this does not bias the
    # 3-way interaction estimate (see allpossible_power_CDmem.py, Supp 4).
    foils = recog_data[recog_data['mem_ground_truth'] == 'unseen'].copy()
    if len(foils) == 0:
        print("[WARNING] No foil data for Supp Analysis 4.")
        return None

    foils['said_old_int']  = (foils['mem_response'].str.lower() == 'yes').astype(int)
    foils['item_is_old_c'] = -0.5
    foils = foils.sort_values(['participant', 'overall_trial_num'])

    foils['_rank'] = foils.groupby('participant').cumcount()
    foils['_n']    = foils.groupby('participant')['_rank'].transform('count')
    foils['_q']    = foils['_n'] // 4
    foils['trial_level_c'] = np.select(
        [foils['_rank'] < foils['_q'],
         foils['_rank'] < foils['_q'] * 2,
         foils['_rank'] < foils['_q'] * 3],
        [0.5, 0.5, -0.5], default=-0.5
    )
    foils['item_type_c'] = np.select(
        [foils['_rank'] < foils['_q'],
         foils['_rank'] < foils['_q'] * 2,
         foils['_rank'] < foils['_q'] * 3],
        [0.5, -0.5, 0.5], default=-0.5
    )
    foils.drop(columns=['_rank', '_n', '_q'], inplace=True)

    # Prepare targets with real contrast codes
    old_items = targets.copy()
    old_items['said_old_int']  = old_items['said_old'].astype(int)
    old_items['item_is_old_c'] = 0.5
    old_items['trial_level_c'] = old_items['trial_level'].map({'high': 0.5, 'low': -0.5})
    old_items['item_type_c']   = old_items['item_type'].map({'controlled': 0.5, 'uncontrolled': -0.5})

    cols = ['participant', 'said_old_int', 'item_is_old_c', 'trial_level_c', 'item_type_c']

    # Assert all required columns are present before concatenating
    for c in cols:
        assert c in old_items.columns, f"Missing column in old_items: {c}"
        assert c in foils.columns,     f"Missing column in foils: {c}"

    df = pd.concat([old_items[cols], foils[cols]], ignore_index=True)
    df = df.dropna(subset=cols)

    if len(df) == 0:
        print("[WARNING] No valid trials for Supp Analysis 4.")
        return None

    print("\nSupplementary Analysis 4: GLMM Interaction (Is_Old x Trial_Level x Item_Type)...")
    print(f"  Trials: {len(df)} (Old={len(old_items)}, Foils={len(foils)})")
    print(f"  Contrast coding: Old=+0.5/Foil=-0.5, High=+0.5/Low=-0.5, Ctrl=+0.5/Unctrl=-0.5")
    print(f"  NOTE: trial_level_c and item_type_c for foils are balanced dummy assignments.")
    print(f"  Old-item cell counts (trial_level x item_type):")
    for (tl, it), grp in old_items.groupby(['trial_level_c', 'item_type_c']):
        lbl_tl = 'High' if tl > 0 else 'Low'
        lbl_it = 'Controlled' if it > 0 else 'Uncontrolled'
        print(f"    {lbl_tl} / {lbl_it}: N trials={len(grp)}, N pxs={grp['participant'].nunique()}")

    df_pl = pl.from_pandas(df)

    model, structure = fit_glmm_with_fallback(
        formula_maximal="said_old_int ~ item_is_old_c * trial_level_c * item_type_c + (1 + item_is_old_c | participant)",
        formula_minimal="said_old_int ~ item_is_old_c * trial_level_c * item_type_c + (1 | participant)",
        data_pl=df_pl,
        family="binomial"
    )
    if model is not None:
        print(f"  [REPORT] Random effects structure used: {structure}")
    return model


# ------------------------------------------------------------------------------
# SUPPLEMENTARY 5: False Alarm Rate Manipulation Check
# ------------------------------------------------------------------------------

def run_supp_analysis_5_fa_check(recog_data):
    """
    Sanity check: verify that false alarm (FA) rates are low and do not differ
    significantly from zero, confirming that participants were not systematically
    biased toward responding 'old' to new items.

    A significant FA rate would indicate a response bias problem that could
    inflate hit rates and distort d-prime estimates.

    Tests:
      1. One-sample t-test: FA rate vs. 0 (are participants above chance bias?)
      2. Descriptive summary per participant.

    Parameters
    ----------
    recog_data : pd.DataFrame
        Full (cleaned) recognition dataset.
    """
    if recog_data is None:
        print("[WARNING] recog_data is None. Skipping Supp Analysis 5.")
        return None

    foils = recog_data[recog_data['mem_ground_truth'] == 'unseen'].copy()
    if len(foils) == 0:
        print("[WARNING] No foil trials found. Skipping Supp Analysis 5.")
        return None

    foils['said_old'] = foils['mem_response'].str.lower() == 'yes'

    fa_per_px = foils.groupby('participant')['said_old'].mean().reset_index()
    fa_per_px.rename(columns={'said_old': 'FA_rate'}, inplace=True)

    group_mean = fa_per_px['FA_rate'].mean()
    group_sd   = fa_per_px['FA_rate'].std()
    group_min  = fa_per_px['FA_rate'].min()
    group_max  = fa_per_px['FA_rate'].max()

    # One-sample t-test against 0
    t_stat, p_val = stats.ttest_1samp(fa_per_px['FA_rate'], popmean=0)

    print("\nSupplementary Analysis 5: False Alarm Rate Manipulation Check")
    print("=" * 60)
    print(f"  N participants : {len(fa_per_px)}")
    print(f"  FA rate M (SD) : {group_mean:.3f} ({group_sd:.3f})")
    print(f"  FA rate range  : {group_min:.3f} - {group_max:.3f}")
    print(f"\n  One-sample t-test (FA rate vs. 0):")
    print(f"    t({len(fa_per_px)-1}) = {t_stat:.3f}, p = {p_val:.4f}")
    if p_val < 0.05:
        print("    -> FA rate is significantly above 0. Check for response bias.")
    else:
        print("    -> FA rate does not significantly differ from 0. No evidence of response bias.")

    print("\n  Per-participant FA rates:")
    for _, row in fa_per_px.sort_values('participant').iterrows():
        print(f"    Participant {row['participant']}: FA = {row['FA_rate']:.3f}")

    return fa_per_px


# ------------------------------------------------------------------------------
# ANALYSIS 5a: Agency as a function of current accuracy and control level
# (adapted from MT_Inference_Analysis.py Analysis 5a)
# ------------------------------------------------------------------------------

def run_agency_current_trial_analysis(test_data):
    """
    OLS regression of agency_rating on current-trial accuracy and control level,
    run separately for each control condition (High vs Low).

    Model (per condition):
      agency_rating ~ detection_accuracy + prop_used + detection_accuracy:prop_used

    Tests whether subjective agency tracks current performance (detection_accuracy)
    and control level (prop_used), and whether they interact, within each condition.

    NOTE: The active OLS model is taken directly from MT_Inference_Analysis.py
    (Analysis 5a). It pools all trials within a condition without accounting for
    participant-level clustering. A mixed-effects alternative (smf.mixedlm with
    participant as random intercept) is provided below in commented-out form --
    uncomment it if you want to account for between-participant variance.

    Parameters
    ----------
    test_data : pd.DataFrame
        Test-phase main data (after exclusions), must contain:
        'agency_rating', 'detection_accuracy', 'prop_used', 'control_condition'.
    """
    if test_data is None or len(test_data) == 0:
        print("[WARNING] test_data is None or empty. Skipping Analysis 5a.")
        return None

    if 'agency_rating' not in test_data.columns or test_data['agency_rating'].isna().all():
        print("[INFO] No agency rating data available. Skipping Analysis 5a.")
        return None

    df_agency = test_data.dropna(subset=['agency_rating', 'detection_accuracy', 'prop_used']).copy()

    if len(df_agency) == 0:
        print("[WARNING] No valid trials with agency ratings. Skipping Analysis 5a.")
        return None

    print("\n" + "=" * 60)
    print("ANALYSIS 5a: Agency ~ Accuracy + Control Level (OLS per Condition)")
    print("=" * 60)
    print("  Model: agency_rating ~ detection_accuracy + prop_used + detection_accuracy:prop_used")
    print("  Source: adapted from MT_Inference_Analysis.py (Analysis 5a).")
    print("  Run separately for High and Low control conditions.")

    results = {}

    for condition in ['high', 'low']:
        df_cond = df_agency[df_agency['control_condition'] == condition]

        if len(df_cond) < 30:
            print(f"\n  [{condition.upper()}] Insufficient data (N={len(df_cond)} trials). Skipping.")
            continue

        print(f"\n  [{condition.upper()} condition] N = {len(df_cond)} trials, "
              f"{df_cond['participant'].nunique()} participant(s)")

        # ------------------------------------------------------------------
        # Active model: OLS (from MT_Inference_Analysis.py Analysis 5a)
        # Pools all trials within condition; ignores participant clustering.
        # ------------------------------------------------------------------
        try:
            formula = "agency_rating ~ detection_accuracy + prop_used + detection_accuracy:prop_used"
            model = smf.ols(formula, data=df_cond).fit()
            results[f'{condition}_agency_model'] = model

            print(f"  R\u00b2 = {model.rsquared:.3f}, F({int(model.df_model)},{int(model.df_resid)}) "
                  f"= {model.fvalue:.3f}, p = {model.f_pvalue:.4f}")
            print(f"  {'Predictor':<35} {'Coef':>8} {'SE':>8} {'t':>7} {'p':>8}")
            print("  " + "-" * 70)
            for term in model.params.index:
                coef  = model.params[term]
                se    = model.bse[term]
                tval  = model.tvalues[term]
                pval  = model.pvalues[term]
                label = term.replace('detection_accuracy:prop_used', 'accuracy \u00d7 prop_used')
                sig   = " *" if pval < 0.05 else ""
                print(f"  {label:<35} {coef:>8.3f} {se:>8.3f} {tval:>7.3f} {pval:>8.4f}{sig}")

        except Exception as e:
            print(f"  [ERROR] OLS model failed for {condition} condition: {e}")

        # ------------------------------------------------------------------
        # Alternative model: Mixed LMM with participant as random intercept.
        # Uncomment this block (and comment out the OLS block above) if you
        # want to account for between-participant variance in agency ratings.
        # Requires smf.mixedlm (already imported as smf).
        # ------------------------------------------------------------------
        # try:
        #     formula_mixed = ("agency_rating ~ detection_accuracy + prop_used "
        #                      "+ detection_accuracy:prop_used")
        #     model_mixed = smf.mixedlm(
        #         formula_mixed, df_cond, groups=df_cond['participant']
        #     ).fit()
        #     results[f'{condition}_agency_model'] = model_mixed
        #
        #     print(f"  Mixed LMM [{condition.upper()}]:")
        #     print(f"  {'Predictor':<35} {'Coef':>8} {'SE':>8} {'z':>7} {'p':>8}")
        #     print("  " + "-" * 70)
        #     for term in model_mixed.params.index:
        #         coef  = model_mixed.params[term]
        #         se    = model_mixed.bse[term]
        #         zval  = model_mixed.tvalues[term]
        #         pval  = model_mixed.pvalues[term]
        #         label = term.replace('detection_accuracy:prop_used', 'accuracy x prop_used')
        #         sig   = " *" if pval < 0.05 else ""
        #         print(f"  {label:<35} {coef:>8.3f} {se:>8.3f} {zval:>7.3f} {pval:>8.4f}{sig}")
        #
        # except Exception as e:
        #     print(f"  [ERROR] Mixed LMM failed for {condition} condition: {e}")

    print()
    return results


# ============================================================================
# PRINT HELPERS
# ============================================================================

def print_stat_results(res):
    """Print formatted t-test results."""
    if res is None:
        return
    print(f"\n{res['analysis']}:")
    print(f"  t({res['n']-1}) = {res['t_stat']:.3f}, p = {res['p_val']:.4f}")
    print(f"  Mean (SD) High: {res['mean_high']:.3f} ({res['sd_high']:.3f})")
    print(f"  Mean (SD) Low:  {res['mean_low']:.3f} ({res['sd_low']:.3f})")
    if 'cohens_d' in res:
        print(f"  Cohen's d       : {res['cohens_d']:.3f}")


def run_recognition_stats(mem_results, targets, supp_mem_results, recog_data):
    """
    Run all recognition statistical analyses.

    Parameters
    ----------
    mem_results      : pd.DataFrame  Primary participant-level results.
    targets          : pd.DataFrame  Trial-level target data.
    supp_mem_results : pd.DataFrame  Supplementary 2x2 participant-level results.
    recog_data       : pd.DataFrame  Full (cleaned) recognition data including foils.
                                     Passed explicitly to avoid reliance on global scope.
    """
    print("\n" + "=" * 60)
    print("RECOGNITION STATISTICAL ANALYSES")
    print("=" * 60)

    # --- Primary ---
    res1 = run_analysis_1_dprime_ttest(mem_results)
    print_stat_results(res1)

    res2 = run_analysis_2_hitrate_ttest(mem_results)
    print_stat_results(res2)

    run_analysis_3_glmm(targets)

    run_analysis_4_interaction_glmm(targets, recog_data)

    # --- Supplementary ---
    print("\n" + "-" * 60)
    print("SUPPLEMENTARY ANALYSES (2x2 Factorial + FA Check)")
    print("-" * 60)

    s1 = run_supp_analysis_1_dprime_2x2_anova(supp_mem_results)
    s2 = run_supp_analysis_2_hitrate_2x2_anova(supp_mem_results)
    run_supp_analysis_3_glmm_2x2(targets)
    run_supp_analysis_4_glmm_foils(targets, recog_data)
    run_supp_analysis_5_fa_check(recog_data)

    # --- Bonferroni correction summary for supplementary ANOVAs ---
    k_supp = 4  # Supp 1-4 constitute the supplementary family
    alpha_adj = 0.05 / k_supp
    print("\n" + "-" * 60)
    print(f"BONFERRONI CORRECTION (Supplementary Analyses, k={k_supp})")
    print(f"  Corrected alpha = 0.05 / {k_supp} = {alpha_adj:.4f}")
    print("-" * 60)
    for label, anova in [("Supp 1 (d-prime 2x2 ANOVA)", s1),
                         ("Supp 2 (hit rate 2x2 ANOVA)", s2)]:
        if anova is None:
            print(f"  {label}: not available.")
            continue
        try:
            tbl = anova.anova_table
            int_keys = [idx for idx in tbl.index if ':' in str(idx)]
            if int_keys:
                p_raw = tbl.loc[int_keys[0], 'Pr > F']
                p_adj = min(1.0, p_raw * k_supp)
                sig   = " *" if p_adj < 0.05 else ""
                print(f"  {label}: interaction p_uncorr={p_raw:.4f}, "
                      f"p_Bonferroni={p_adj:.4f}{sig}")
        except Exception as e:
            print(f"  {label}: could not extract p-values ({e}).")
    print(f"  NOTE: Supp 3 & 4 GLMM p-values should also be compared against alpha={alpha_adj:.4f}.")


    print("\n" + "=" * 60)




# ==============================================================================
# ANALYSIS 7: Agency Rating -> Recognition Memory
# (Continuous predictor: within-participant z-scored agency_rating)
# ==============================================================================

def prepare_targets_agency(targets):
    """
    Z-score agency_rating within each participant and attach as 'agency_rating_z'.

    Agency ratings are treated as a continuous predictor (1–7 Likert scale).
    Within-participant z-scoring removes individual differences in scale use
    (e.g. a participant who always rates 5–7 vs one who uses 1–4), so the
    slope in the GLMM captures trial-to-trial within-person variation only.

    Parameters
    ----------
    targets : pd.DataFrame
        Trial-level recognition data for old items.

    Returns
    -------
    targets_agency : pd.DataFrame
        Copy of targets with 'agency_rating_z' column added.
        Rows with missing agency_rating are dropped.
    """
    if targets is None or len(targets) == 0:
        print("[INFO] No target data for agency memory preparation.")
        return None
    if 'agency_rating' not in targets.columns or targets['agency_rating'].isna().all():
        print("[INFO] No agency_rating data. Skipping agency memory preparation.")
        return None

    df = targets.copy()
    df['agency_rating'] = pd.to_numeric(df['agency_rating'], errors='coerce')
    df = df.dropna(subset=['agency_rating', 'said_old', 'participant'])

    # Z-score within participant
    df['agency_rating_z'] = df.groupby('participant')['agency_rating'].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-9)
    )

    print("\n  Agency rating z-scoring (within participant):")
    print(f"    Trials with agency rating : {len(df)}")
    print(f"    Participants              : {df['participant'].nunique()}")
    for px, px_df in df.groupby('participant'):
        print(
            f"    Participant {px}: "
            f"M = {px_df['agency_rating'].mean():.2f}, "
            f"SD = {px_df['agency_rating'].std():.2f}, "
            f"range = {px_df['agency_rating'].min():.0f}–{px_df['agency_rating'].max():.0f}"
        )

    return df


# ------------------------------------------------------------------------------
# ANALYSIS 7: GLMM — continuous agency_rating_z -> said_old
# ------------------------------------------------------------------------------

def run_analysis_7_agency_glmm(targets_agency):
    """
    Trial-level GLMM (Binomial, logit link) testing whether within-participant
    variation in agency_rating predicts subsequent recognition memory.

    Predictor: agency_rating_z — agency_rating z-scored within each participant.
    Within-participant z-scoring removes between-participant differences in
    rating scale use, so the slope reflects trial-to-trial variation only.

    Model formula:
      Maximal : said_old_int ~ agency_rating_z + (1 + agency_rating_z | participant)
      Fallback: said_old_int ~ agency_rating_z + (1 | participant)
    """
    if targets_agency is None or len(targets_agency) == 0:
        print("[WARNING] No target data for Analysis 7. Skipping.")
        return None

    df = targets_agency.copy()
    df['said_old_int'] = df['said_old'].astype(int)
    df = df.dropna(subset=['said_old_int', 'agency_rating_z', 'participant'])

    if len(df) == 0:
        print("[WARNING] No valid trials for Analysis 7 after preprocessing.")
        return None

    print("\nAnalysis 7: GLMM (Binomial, logit link) — continuous agency_rating_z -> said_old")
    print(f"  Trials: {len(df)} | Participants: {df['participant'].nunique()}")
    print(f"  Predictor: agency_rating_z (within-participant z-scored, continuous 1–7 scale)")

    df_pl = pl.from_pandas(df[['participant', 'said_old_int', 'agency_rating_z']])

    model, structure = fit_glmm_with_fallback(
        formula_maximal="said_old_int ~ agency_rating_z + (1 + agency_rating_z | participant)",
        formula_minimal="said_old_int ~ agency_rating_z + (1 | participant)",
        data_pl=df_pl,
        family="binomial"
    )
    if model is not None:
        print(f"  [REPORT] Random effects structure used: {structure}")
    return model


def run_agency_recognition_stats(targets):
    """
    Run Analysis 7: continuous agency_rating -> recognition memory.

    Parameters
    ----------
    targets : pd.DataFrame
        Trial-level target data (from analyze_recognition).

    Returns
    -------
    targets_agency : pd.DataFrame
        Targets with agency_rating_z column; None if no agency data.
    """
    print("\n" + "=" * 60)
    print("ANALYSIS 7: Agency Rating -> Recognition Memory")
    print("  (Continuous predictor: within-participant z-scored agency_rating)")
    print("=" * 60)

    targets_agency = prepare_targets_agency(targets)
    run_analysis_7_agency_glmm(targets_agency)

    print("\n" + "=" * 60)
    return targets_agency


# ============================================================================
# DESCRIPTIVE STATISTICS
# ============================================================================

def print_descriptives(data):
    """
    Print demographic descriptives for the final sample.
    (N, gender counts/percentages, age mean, SD, min, max)
    """
    demographics = data.drop_duplicates(subset=["participant"])[
        ["participant", "age", "gender"]
    ].copy()

    n_total = len(demographics)

    demographics["gender"] = demographics["gender"].astype(str).str.strip().str.lower()
    gender_counts = demographics["gender"].value_counts()
    gender_pcts   = demographics["gender"].value_counts(normalize=True) * 100

    demographics["age"] = pd.to_numeric(demographics["age"], errors="coerce")
    age_mean = demographics["age"].mean()
    age_sd   = demographics["age"].std()
    age_min  = demographics["age"].min()
    age_max  = demographics["age"].max()

    print("=" * 60)
    print("DESCRIPTIVE STATISTICS (Final Sample)")
    print("=" * 60)
    print(f"  Total N = {n_total}\n")

    print("  Gender:")
    for g in gender_counts.index:
        count = gender_counts[g]
        pct   = gender_pcts[g]
        label = "Female" if g == "f" else "Male" if g == "m" else g.capitalize()
        print(f"    {label:<10} : {count:>3} ({pct:>4.1f}%)")

    print("\n  Age:")
    if pd.notna(age_mean):
        print(f"    Mean (SD)  : {age_mean:.1f} ({age_sd:.1f})")
        print(f"    Range      : {age_min:.0f} - {age_max:.0f}")
    else:
        print("    No valid age data found.")
    print("=" * 60 + "\n")


# ============================================================================
# CALIBRATION CONVERGENCE REPORT
# ============================================================================

def report_calibration_convergence(data):
    if 'phase' not in data.columns:
        print("[WARNING] 'phase' column missing.")
        return

    calib_data = data[data['phase'] == 'calibration'].copy()
    if len(calib_data) == 0:
        return

    print("\n" + "=" * 60)
    print("CALIBRATION CONVERGENCE")
    print("=" * 60)
    print("\n| Participant | Target | Trials | Converged | Final Alpha SD |")
    print("|-------------|--------|--------|-----------|----------------|")

    for px, px_df in calib_data.groupby('participant'):
        for target, target_df in px_df.groupby('calib_target'):
            n_trials = len(target_df)
            final_alpha_sd = target_df['quest_alpha_sd'].iloc[-1]
            
            # Direct threshold check: converged if alpha_sd < 0.2
            is_converged = final_alpha_sd < 0.20
            
            cond_label = f"Low ({target:.2f})" if target < 0.7 else f"High ({target:.2f})"
            
            print(f"| {px:<11} | {cond_label:<10} | {n_trials:<6} | {str(is_converged):<9} | {final_alpha_sd:<14.3f} |")
    print()


def _px_label(data, col='participant'):
    """
    Build a compact participant label string for plot titles.
    Single participant  -> '(p. 3)'
    Multiple participants -> '(p. 2, 3, 7)'
    """
    if data is None or col not in data.columns:
        return ''
    ids = sorted(data[col].dropna().unique().tolist())
    ids_str = ', '.join(str(int(i)) if float(i).is_integer() else str(i) for i in ids)
    return f'(p. {ids_str})'


def plot_calibration_convergence(data, output_dir, suffix=""):
    """
    Plot the trajectory of QUEST alpha_sd over calibration trials.
    """
    if 'phase' not in data.columns or 'quest_alpha_sd' not in data.columns:
        return
        
    calib_data = data[data['phase'] == 'calibration'].copy()
    if len(calib_data) == 0:
        return

    calib_data['trial_in_block'] = pd.to_numeric(calib_data['trial_in_block'], errors='coerce')
    calib_data['quest_alpha_sd'] = pd.to_numeric(calib_data['quest_alpha_sd'], errors='coerce')
    calib_data['calib_target_label'] = np.where(calib_data['calib_target'] < 0.7, 'Low (~0.55)', 'High (~0.85)')
    
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
    plt.figure(figsize=(8, 6))

    # Plot average trajectory with error bands
    sns.lineplot(
        data=calib_data, 
        x='trial_in_block', 
        y='quest_alpha_sd', 
        hue='calib_target_label', 
        marker='o',
        errorbar='se'
    )
    
    # Plot individual trajectories very lightly
    for px in calib_data['participant'].unique():
        px_data = calib_data[calib_data['participant'] == px]
        sns.lineplot(
            data=px_data, 
            x='trial_in_block', 
            y='quest_alpha_sd', 
            hue='calib_target_label', 
            alpha=0.15, 
            legend=False
        )

    plt.axhline(y=0.2, color='red', linestyle='--', alpha=0.5, label='Convergence Threshold (0.2)')
    plt.title(f'Calibration Phase: QUEST Alpha SD Trajectory  {_px_label(data)}')
    plt.xlabel('Trial in Block')
    plt.ylabel('Alpha SD')
    
    # handle duplicated legends from individual lines
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), title='Target Condition')

    plt.tight_layout()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'calibration_convergence{suffix}.png'
    plt.savefig(out_path, dpi=150)
    plt.close()
    print("Saved plot: " + str(out_path))


def plot_recognition_performance(supp_mem_results, output_dir, suffix=""):
    """
    Create a bar plot for recognition hit rates.
    X-axis: Trial Level (High vs Low), Hue: Item Type (Controlled vs Uncontrolled).
    """
    if supp_mem_results is None or len(supp_mem_results) == 0:
        print("[WARNING] No supplementary recognition data for plotting.")
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    
    plt.figure(figsize=(10, 7))
    
    # Sort for consistent plotting
    plot_df = supp_mem_results.copy()
    if 'trial_level' in plot_df.columns and 'item_type' in plot_df.columns:
        plot_df['trial_level'] = pd.Categorical(plot_df['trial_level'], categories=['high', 'low'], ordered=True)
        plot_df['item_type'] = pd.Categorical(plot_df['item_type'], categories=['controlled', 'uncontrolled'], ordered=True)
        plot_df = plot_df.sort_values(['trial_level', 'item_type'])

        # Create the grouped bar plot
        # High Control Blue, Low Control Orange (standard colors from sanity check)
        # But here we have two factors, so using a standard palette like 'Set2'
        ax = sns.barplot(
            data=plot_df, 
            x='trial_level', 
            y='Hit_rate', 
            hue='item_type',
            errorbar='se',
            palette='Set2',
            capsize=0.1
        )

        plt.title(f'Recognition Performance: Hit Rate by Condition  {_px_label(plot_df)}',
                  fontsize=14, fontweight='bold')
        plt.xlabel('Control Task Level', fontsize=12)
        plt.ylabel('Hit Rate (Mean Accuracy to Old Items)', fontsize=12)
        plt.ylim(0, 1)
        plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
        
        plt.legend(title='Item Type', frameon=True, loc='upper right')

        plt.tight_layout()
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f'recognition_performance{suffix}.png'
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved recognition plot: {out_path}")
    else:
        print("[WARNING] Missing required columns for recognition plot.")


# ============================================================================
# DATA INTEGRITY CHECKS
# ============================================================================

def check_image_uniqueness(data):
    """
    Confirm that all images shown during the test phase were unique
    *within each participant* across the img_A_name and img_B_name columns.

    Uniqueness is evaluated per participant so that the same image appearing
    for two different participants does not falsely trigger a failure.
    """
    if ('phase' not in data.columns
            or 'img_A_name' not in data.columns
            or 'img_B_name' not in data.columns
            or 'participant' not in data.columns):
        print("[WARNING] Missing columns for image uniqueness check.")
        return

    test_data = data[data['phase'] == 'test'].copy()
    if len(test_data) == 0:
        print("[WARNING] No test phase trials found for image uniqueness check.")
        return

    print("\n" + "=" * 60)
    print("IMAGE UNIQUENESS CHECK (Test Phase, per participant)")
    print("=" * 60)

    any_failure = False
    for px, px_df in test_data.groupby('participant'):
        from collections import Counter
        img_a = px_df['img_A_name'].dropna().tolist()
        img_b = px_df['img_B_name'].dropna().tolist()
        all_images = img_a + img_b
        n_total  = len(all_images)
        n_unique = len(set(all_images))
        if n_total != n_unique:
            counts     = Counter(all_images)
            duplicates = [img for img, cnt in counts.items() if cnt > 1]
            print(f"  FAILURE - Participant {px}: {n_total - n_unique} duplicate(s)")
            for d in duplicates:
                print(f"    Duplicate: {d} (shown {counts[d]} times)")
            any_failure = True

    if not any_failure:
        print("  -> SUCCESS: All images were unique within every participant.")
    print("=" * 60)


# ============================================================================
# SANITY CHECKS & PLOTS
# ============================================================================

def sanity_check(df):
    """Compute sanity check statistics for performance and agency ratings."""

    results = {}

    if 'control_condition' in df.columns:
        results['accuracy_by_condition'] = (
            df.groupby('control_condition')['detection_accuracy']
            .agg(['mean', 'std', 'count'])
        )
    else:
        results['accuracy'] = df['detection_accuracy'].agg(['mean', 'std', 'count'])

    if 'agency_rating' in df.columns and df['agency_rating'].notna().any():
        if 'control_condition' in df.columns:
            results['agency_by_condition'] = (
                df.groupby('control_condition')['agency_rating']
                .agg(['mean', 'std', 'count'])
            )
            results['agency_by_accuracy'] = (
                df.groupby(['control_condition', 'detection_accuracy'])['agency_rating']
                .agg(['mean', 'std', 'count'])
            )
        else:
            results['agency'] = df['agency_rating'].agg(['mean', 'std', 'count'])
            results['agency_by_accuracy'] = (
                df.groupby('detection_accuracy')['agency_rating']
                .agg(['mean', 'std', 'count'])
            )

    df_copy = df.copy()
    df_copy['prop_used'] = pd.to_numeric(df_copy['prop_used'], errors='coerce')
    if df_copy['prop_used'].nunique() > 10:
        df_copy['prop_bin'] = pd.cut(df_copy['prop_used'], bins=10, duplicates='drop')
    else:
        df_copy['prop_bin'] = df_copy['prop_used']

    if 'control_condition' in df_copy.columns:
        results['psychometric'] = (
            df_copy.groupby(['prop_bin', 'control_condition'], observed=False)
            ['detection_accuracy'].agg(['mean', 'std', 'count'])
        )
    else:
        results['psychometric'] = (
            df_copy.groupby('prop_bin', observed=False)
            ['detection_accuracy'].agg(['mean', 'std', 'count'])
        )

    return results


def plot_sanity_check(data, output_dir, suffix=""):
    """Create plots for sanity check statistics."""

    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Define consistent colors for conditions
    color_map = {'high': '#1f77b4', 'low': '#ff7f0e'} # Blue for High, Orange for Low

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    data = data.copy()

    # Ensure numeric types for critical columns
    numeric_cols = ['detection_accuracy', 'prop_used', 'agency_rating', 'rt_choice', 'calib_target']
    for col in numeric_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')

    # Split into calibration and test data
    calib_data = data[data['phase'] == 'calibration'].copy() if 'phase' in data.columns else pd.DataFrame()
    test_data  = data[data['phase'] == 'test'].copy()        if 'phase' in data.columns else data.copy()

    # 1. Psychometric function (using Calibration data fit + Test markers)
    ax = axes[0, 0]
    if not calib_data.empty:
        # Define psychometric fit (sigmoid starting at 0.5)
        def psychometric_func(x, alpha, beta):
            return 0.5 + 0.48 / (1 + np.exp(-np.clip(beta, 0.1, 100) * (x - alpha)))

        # Detect number of participants to decide on binning strategy
        n_participants = data['participant'].nunique() if 'participant' in data.columns else 1

        # 1a. Global sigmoid fit (pooling all calibration data)
        try:
            if calib_data['prop_used'].nunique() >= 2:
                popt, _ = curve_fit(psychometric_func, calib_data['prop_used'], calib_data['detection_accuracy'], 
                                    p0=[calib_data['prop_used'].mean(), 10], 
                                    bounds=([0, 0], [1, 200]), maxfev=5000)
                x_fit = np.linspace(0, 1, 100)
                y_fit = psychometric_func(x_fit, *popt)
                ax.plot(x_fit, y_fit, color='gray', linestyle='-', linewidth=2, alpha=0.6, label='Global Sigmoid Fit')
        except Exception as e:
            print(f"[Plotting] Global sigmoid fit failed: {e}")

        # 1b. Staircase-specific data points
        if 'calib_target' in calib_data.columns:
            calib_data['calib_condition'] = np.where(calib_data['calib_target'] < 0.7, 'low', 'high')
            for condition in ['high', 'low']:
                subset = calib_data[calib_data['calib_condition'] == condition].copy()
                if subset.empty: continue
                
                # Binning Strategy: use raw values for single participant, bins for multiple
                if n_participants == 1:
                    psychometric = subset.groupby('prop_used', observed=False)['detection_accuracy'].agg(['mean', 'sem']).reset_index()
                    psychometric.rename(columns={'prop_used': 'prop_bin'}, inplace=True)
                else:
                    bins = np.linspace(0, 1, 11)
                    bin_centers = (bins[:-1] + bins[1:]) / 2
                    subset['prop_bin'] = pd.cut(subset['prop_used'], bins=bins, labels=bin_centers)
                    psychometric = subset.groupby('prop_bin', observed=False)['detection_accuracy'].agg(['mean', 'sem']).reset_index()
                    psychometric['prop_bin'] = psychometric['prop_bin'].astype(float)
                
                psychometric = psychometric.dropna(subset=['mean'])
                
                ax.errorbar(psychometric['prop_bin'], psychometric['mean'], yerr=psychometric['sem'],
                            marker='o', capsize=3, linestyle='None', color=color_map[condition], alpha=0.7,
                            label=f'{condition.capitalize()} {"Samples" if n_participants == 1 else "Bins"}')
                # Add target horizontal lines
                target_acc = 0.55 if condition == 'low' else 0.85
                ax.axhline(y=target_acc, color=color_map[condition], linestyle=':', alpha=0.3)

        # 1c. Overlay Test Results
        if not test_data.empty:
            for condition in ['high', 'low']:
                test_subset = test_data[test_data['control_condition'] == condition]
                if not test_subset.empty:
                    mean_prop = test_subset['prop_used'].mean()
                    mean_acc  = test_subset['detection_accuracy'].mean()
                    ax.plot(mean_prop, mean_acc, marker='*', markersize=15, 
                            color=color_map[condition], markeredgecolor='black',
                            linestyle='None', label=f'{condition.capitalize()} (Test)')
        
        ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize='small', loc='lower right')
    else:
        ax.text(0.5, 0.5, 'No calibration data available', transform=ax.transAxes, ha='center')

    ax.set_xlabel('Control Level (prop self-motion)')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Psychometric Function ({"Raw" if n_participants == 1 else "Binned"} & Global Fit)')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlim([-0.05, 1.05])

    # 2. Accuracy by condition (using Test data)
    ax = axes[0, 1]
    has_condition = 'control_condition' in test_data.columns
    if has_condition and not test_data.empty:
        # Sort to ensure consistent order: high, low
        accuracy_data = (
            test_data.groupby('control_condition')['detection_accuracy']
            .agg(['mean', 'sem']).reindex(['high', 'low']).dropna().reset_index()
        )
        colors = [color_map.get(str(c), '#666666') for c in accuracy_data['control_condition']]
        ax.bar(accuracy_data['control_condition'].astype(str), accuracy_data['mean'],
               yerr=accuracy_data['sem'], capsize=5, color=colors)
        ax.set_xlabel('Control Condition')
    elif not test_data.empty:
        ax.bar(['Overall'], [test_data['detection_accuracy'].mean()],
               yerr=[test_data['detection_accuracy'].sem()], capsize=5, color='gray')
    
    ax.set_ylabel('Mean Accuracy')
    ax.set_title('Accuracy (Test Phase)')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
    ax.set_ylim([0, 1])

    # 3. Agency ratings by accuracy (using Test data)
    ax = axes[1, 0]
    if 'agency_rating' in test_data.columns and test_data['agency_rating'].notna().any():
        if has_condition:
            sns.barplot(data=test_data, x='control_condition', y='agency_rating',
                        hue='detection_accuracy', ax=ax, errorbar='se',
                        palette={0: '#cccccc', 1: '#999999'}, order=['high', 'low'])
            ax.set_xlabel('Control Condition')
        else:
            sns.barplot(data=test_data, x='phase', y='agency_rating',
                        hue='detection_accuracy', ax=ax, errorbar='se')
        ax.set_ylabel('Agency Rating (1-7)')
        ax.set_title('Agency Ratings by Accuracy (Test Phase)')
        ax.legend(title='Correct (1) / Incorrect (0)')
    else:
        ax.text(0.5, 0.5, 'No agency rating data available', transform=ax.transAxes, ha='center')

    # 4. RT distribution (using Test data)
    ax = axes[1, 1]
    if 'rt_choice' in test_data.columns and test_data['rt_choice'].notna().any():
        df_rt = test_data[test_data['rt_choice'] > 0]
        if has_condition and len(df_rt) > 0:
            for condition in ['high', 'low']:
                subset = df_rt[df_rt['control_condition'] == condition]['rt_choice']
                if len(subset) > 0:
                    ax.hist(subset, bins=30, alpha=0.4, label=condition, 
                            density=True, color=color_map.get(condition))
            ax.legend()
        elif len(df_rt) > 0:
            ax.hist(df_rt['rt_choice'], bins=30, alpha=0.5, density=True, color='gray')
        ax.set_xlabel('Reaction Time (s)')
        ax.set_ylabel('Density')
        ax.set_title('RT Distribution (Test Phase)')
    else:
        ax.text(0.5, 0.5, 'No RT data available', transform=ax.transAxes, ha='center')

    px_lbl = _px_label(data)
    fig.suptitle(f'Sanity Check  {px_lbl}', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / f'sanity_check{suffix}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved plot: " + str(out_dir / f'sanity_check{suffix}.png'))


# ============================================================================
# CONTROL DETECTION ACCURACY ANALYSIS (Manipulation Check)
# ============================================================================

def run_cd_accuracy_analysis(data):
    """
    Verify the control manipulation worked by testing whether detection accuracy
    differs significantly between High and Low conditions in the test phase.

    This is a prerequisite check: if accuracy does not differ, the memory
    analyses cannot be attributed to a control-level effect.

    Analyses:
      1. Descriptive summary (mean, SD per condition).
      2. Paired t-test: High vs. Low (participant-level means) + Cohen's d.
      3. Binomial GLMM: detection_accuracy ~ control_c + (1 + control_c | participant).
    """
    if data is None or len(data) == 0:
        print("[WARNING] No data for CD accuracy analysis. Skipping.")
        return

    test_data = data[data['phase'] == 'test'].copy() if 'phase' in data.columns else data.copy()
    test_data['detection_accuracy'] = pd.to_numeric(test_data['detection_accuracy'], errors='coerce')

    if 'control_condition' not in test_data.columns:
        print("[WARNING] 'control_condition' column missing. Skipping CD accuracy analysis.")
        return

    print("\n" + "=" * 60)
    print("CONTROL DETECTION ACCURACY (Manipulation Check)")
    print("=" * 60)
    print(f"  Expected: High \u2248 {EXPECTED_ACCURACY['high']*100:.0f}%, Low \u2248 {EXPECTED_ACCURACY['low']*100:.0f}%")

    # 1. Descriptives
    desc = (
        test_data.groupby('control_condition')['detection_accuracy']
        .agg(mean='mean', sd='std', n='count')
        .reindex(['high', 'low']).dropna()
    )
    print("\n  Descriptives:")
    for cond, row in desc.iterrows():
        print(f"    {cond.capitalize()}: M = {row['mean']:.3f}, SD = {row['sd']:.3f}, N = {int(row['n'])} trials")

    # 2. Paired t-test (participant-level means)
    acc_px = (
        test_data
        .groupby(['participant', 'control_condition'])['detection_accuracy']
        .mean().reset_index()
        .pivot(index='participant', columns='control_condition', values='detection_accuracy')
        .dropna(subset=['high', 'low'])
    )
    if len(acc_px) < 2:
        print("\n  [WARNING] Too few participants for paired t-test. Skipping.")
        return

    t_stat, p_val = stats.ttest_rel(acc_px['high'], acc_px['low'])
    diff = acc_px['high'] - acc_px['low']
    cohens_d = diff.mean() / (diff.std() + 1e-9)

    print(f"\n  Paired t-test (High vs. Low, participant-level means):")
    print(f"    t({len(acc_px)-1}) = {t_stat:.3f}, p = {p_val:.4f}")
    print(f"    Cohen's d = {cohens_d:.3f}")
    print(f"    Mean (SD) High: {acc_px['high'].mean():.3f} ({acc_px['high'].std():.3f})")
    print(f"    Mean (SD) Low:  {acc_px['low'].mean():.3f} ({acc_px['low'].std():.3f})")
    if p_val < 0.05:
        print("    -> Significant: manipulation successfully differentiated accuracy levels.")
    else:
        print("    -> NOT significant: accuracy did not differ between conditions. "
              "Interpret memory effects with caution.")

    # 3. Binomial GLMM
    print("\n  Binomial GLMM: detection_accuracy ~ control_c + (1 + control_c | participant)")
    df = test_data.dropna(subset=['detection_accuracy', 'control_condition', 'participant']).copy()
    df['control_c'] = df['control_condition'].map({'high': 0.5, 'low': -0.5})
    df = df.dropna(subset=['control_c'])
    df['detection_accuracy'] = df['detection_accuracy'].astype(int)
    if df['participant'].nunique() < 2:
        print("  [WARNING] Need \u22652 participants for GLMM. Skipping.")
        return
    df_pl = pl.from_pandas(df[['participant', 'detection_accuracy', 'control_c']])
    fit_glmm_with_fallback(
        formula_maximal="detection_accuracy ~ control_c + (1 + control_c | participant)",
        formula_minimal="detection_accuracy ~ control_c + (1 | participant)",
        data_pl=df_pl,
        family="binomial"
    )
    print("=" * 60 + "\n")


# ============================================================================
# PLOT: d-prime by condition (2x2)
# ============================================================================

def plot_dprime_by_condition(supp_mem_results, output_dir, suffix=""):
    """
    Bar chart of mean d' grouped by the 2x2 design:
      Trial Level (High / Low) x Item Type (Controlled / Uncontrolled).
    A d' = 0 reference line marks chance memory performance.
    """
    if supp_mem_results is None or len(supp_mem_results) == 0:
        print("[WARNING] No supplementary data for d-prime plot.")
        return
    if 'trial_level' not in supp_mem_results.columns or 'item_type' not in supp_mem_results.columns:
        print("[WARNING] Missing required columns for d-prime plot.")
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    plot_df = supp_mem_results.copy()
    plot_df['trial_level'] = pd.Categorical(
        plot_df['trial_level'], categories=['high', 'low'], ordered=True
    )
    plot_df['item_type'] = pd.Categorical(
        plot_df['item_type'], categories=['controlled', 'uncontrolled'], ordered=True
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(
        data=plot_df, x='trial_level', y='d_prime', hue='item_type',
        errorbar='se', palette='Set2', capsize=0.1, ax=ax,
        order=['high', 'low'], hue_order=['controlled', 'uncontrolled']
    )
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1.2, alpha=0.6,
               label="d\u2032 = 0 (chance)")
    ax.set_title(f"Recognition Memory Sensitivity (d\u2032) by Condition  {_px_label(supp_mem_results)}",
                 fontsize=14, fontweight='bold')
    ax.set_xlabel("Control Task Level", fontsize=12)
    ax.set_ylabel("d\u2032 (Mean \u00b1 SE)", fontsize=12)
    ax.set_ylim(0, 2)
    ax.legend(title='Item Type', frameon=True, loc='lower right')
    plt.tight_layout()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'dprime_by_condition{suffix}.png'
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved d-prime plot: {out_path}")


# ============================================================================
# PLOT: Agency rating vs. recognition memory
# ============================================================================

def plot_agency_recognition(targets, output_dir, suffix=""):
    """
    Scatterplot of mean hit rate per participant per agency rating (1-7),
    ignoring control condition, with a single logistic regression trend line.

    X-axis : Agency rating 1-7 (discrete Likert values)
    Y-axis : Mean hit rate (proportion of targets recognised as 'old')
    Shows  : Whether memory performance increases as SoA ratings increase.
    """
    if targets is None or len(targets) == 0:
        print("[WARNING] No target data for agency_recognition plot.")
        return
    if 'agency_rating' not in targets.columns or targets['agency_rating'].isna().all():
        print("[INFO] No agency_rating data. Skipping agency_recognition plot.")
        return

    df = targets.copy()
    df['said_old_int'] = df['said_old'].astype(int)
    df['agency_rating'] = pd.to_numeric(df['agency_rating'], errors='coerce')
    df = df.dropna(subset=['agency_rating', 'said_old_int'])

    if len(df) == 0:
        print("[WARNING] No valid rows for agency_recognition plot.")
        return

    plt.style.use('seaborn-v0_8-whitegrid')

    rating_vals = sorted(df['agency_rating'].dropna().unique())

    # Get participant-level means for each agency rating
    px_means = (
        df.groupby(['participant', 'agency_rating'])['said_old_int']
        .mean()
        .reset_index()
        .rename(columns={'said_old_int': 'mean_hit_rate'})
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    # Add jitter to x-axis for participant points to avoid overlap
    jitter = np.random.uniform(-0.15, 0.15, size=len(px_means))
    x_jittered = px_means['agency_rating'] + jitter
    
    # Plot individual participant means as scattered dots
    ax.scatter(
        x_jittered, px_means['mean_hit_rate'],
        color='#1f77b4', alpha=0.4, edgecolor='none', s=40,
        label='Participant Mean', zorder=2
    )

    # Calculate overall means and SEM per agency rating for bold summary points
    overall_stats = (
        px_means.groupby('agency_rating')['mean_hit_rate']
        .agg(['mean', 'sem'])
        .reindex(rating_vals)
        .reset_index()
    )

    # Plot overall mean points and error bars
    ax.errorbar(
        overall_stats['agency_rating'], overall_stats['mean'], yerr=overall_stats['sem'],
        fmt='o', color='#d62728', markersize=8, capsize=5, linewidth=2,
        label='Group Mean ± SE', zorder=4
    )

    # Single logistic regression trend line across full rating range
    try:
        from scipy.special import expit
        if df['agency_rating'].nunique() >= 3:
            # Fit logistic regression on the raw binary trial data
            lr       = smf.logit('said_old_int ~ agency_rating', data=df).fit(disp=False)
            x_line   = np.linspace(min(rating_vals), max(rating_vals), 200)
            log_odds = lr.params['Intercept'] + lr.params['agency_rating'] * x_line
            ax.plot(x_line, expit(log_odds),
                    color='#2ca02c', linewidth=3, linestyle='-', alpha=0.9,
                    label='Overall Logistic Trend', zorder=3)
    except Exception as e:
        print(f"[WARNING] Could not fit overall logistic trend: {e}")

    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='Chance')
    ax.set_xticks(rating_vals)
    ax.set_xlabel('Agency Rating at Encoding (1 = Low SoA → 7 = High SoA)', fontsize=12)
    ax.set_ylabel('Hit Rate (Proportion Recognised)', fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(
        f'Sense of Agency vs Recognition Memory {_px_label(targets)}',
        fontsize=13, fontweight='bold'
    )
    
    # Clean up legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), frameon=True, loc='lower right')
    
    plt.tight_layout()

    out_dir  = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'agency_recognition{suffix}.png'
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved agency_recognition plot: {out_path}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    data = load_all_data()

    if data is None:
        print("No data loaded. Exiting.")
        sys.exit(1)

    assert data is not None

    # ------------------------------------------------------------------
    # 2. Print column names and overall shape
    # ------------------------------------------------------------------
    print("=" * 60)
    print("COLUMN NAMES")
    print("=" * 60)
    print(f"Total columns: {len(data.columns)}\n")
    for i, col in enumerate(data.columns, start=1):
        print(f"  {i:>3}. {col}")

    print("\nFirst 10 rows:")
    print(data.head(10).to_string())
    print()

    # ------------------------------------------------------------------
    # 2b. Image Uniqueness Check
    # ------------------------------------------------------------------
    check_image_uniqueness(data)

    print("\n" + "=" * 60)
    print("DATAFRAME SHAPE  (raw, before exclusions)")
    print("=" * 60)
    print(f"  Rows         : {data.shape[0]}")
    print(f"  Columns      : {data.shape[1]}")
    print(f"  Participants : {data['participant'].nunique()}")
    print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    # 3. Exclusion criterion 1 - Timeout rate
    # ------------------------------------------------------------------
    data, excluded_timeout = exclude_timeout_participants(data)

    # ------------------------------------------------------------------
    # 4. Exclusion criterion 2 - Accuracy outliers
    # ------------------------------------------------------------------
    data, excluded_accuracy, accuracy_report = exclude_accuracy_outliers(data)

    # ------------------------------------------------------------------
    # 5. Load recognition data
    # ------------------------------------------------------------------
    n_rt_excluded = 0  # default; overwritten inside the else-block below
    print("\n" + "=" * 60)
    print("LOADING RECOGNITION DATA")
    print("=" * 60)
    recog_data = load_recognition_data()

    if recog_data is None:
        print("No recognition data loaded. Skipping recognition analyses.")
    else:
        print(f"\nRecognition column names (total: {len(recog_data.columns)}):")
        for i, col in enumerate(recog_data.columns, start=1):
            print(f"  {i:>3}. {col}")
        print("\nFirst 10 rows of recognition data:")
        print(recog_data.head(10).to_string())
        print()

        # ------------------------------------------------------------------
        # 6. Recognition exclusion - long RTs (> 20 s)
        # ------------------------------------------------------------------
        recog_data, n_rt_excluded, rt_exclusion_details = exclude_long_rt_trials(recog_data)

    # ------------------------------------------------------------------
    # 7. Cross-dataset synchronisation
    # ------------------------------------------------------------------
    if recog_data is not None:
        data, recog_data = sync_participant_ids(data, recog_data)
    else:
        print("Skipping synchronisation (no recognition data loaded).")

    # ------------------------------------------------------------------
    # 8. Final summary
    # ------------------------------------------------------------------
    print("=" * 60)
    print("FINAL EXCLUSION SUMMARY")
    print("=" * 60)
    print("  [Main task data]")
    print(f"    Excluded -- timeout     : {excluded_timeout if excluded_timeout else 'none'}")
    print(f"    Excluded -- accuracy    : {excluded_accuracy if excluded_accuracy else 'none'}")
    print(f"    Participants remaining  : {data['participant'].nunique()}")
    print(f"    Rows remaining          : {len(data)}")
    if recog_data is not None:
        print("  [Recognition data]")
        print(f"    Trials excluded (RT>20s): {n_rt_excluded}")
        print(f"    Trials remaining         : {len(recog_data)}")
        print(f"    Participants remaining    : {recog_data['participant'].nunique()}")
    print("  [Final participant IDs in both datasets]")
    print(f"    {sorted(data['participant'].unique().tolist())}")
    print("=" * 60 + "\n")
    # ------------------------------------------------------------------
    # 9. Calibration Convergence Reporting
    # ------------------------------------------------------------------
    report_calibration_convergence(data)
   
    # ------------------------------------------------------------------
    # 10. Descriptive Statistics
    # ------------------------------------------------------------------
    print_descriptives(data)
    sys.stdout.flush()

    # ------------------------------------------------------------------
    # 11. Sanity Checks & Plots
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SANITY CHECKS & PLOTS")
    print("=" * 60)

    test_data = data[data['phase'] == 'test'].copy() if 'phase' in data.columns else data.copy()
    test_data['detection_accuracy'] = pd.to_numeric(test_data['detection_accuracy'], errors='coerce')
    test_data['prop_used']          = pd.to_numeric(test_data['prop_used'],          errors='coerce')

    s_results = sanity_check(test_data)
    if 'accuracy_by_condition' in s_results:
        print("\nAccuracy by Condition:")
        print(s_results['accuracy_by_condition'].to_string())
        print()

    plot_sanity_check(data, POOLED_DIR)
    plot_calibration_convergence(data, POOLED_DIR)
    sys.stdout.flush()

    # ------------------------------------------------------------------
    # 12. Recognition Memory Analysis
    # ------------------------------------------------------------------
    if recog_data is not None:
        print("\n" + "=" * 60)
        print("RECOGNITION MEMORY (D-PRIME)")
        print("=" * 60)

        mem_results, targets, supp_mem_results = analyze_recognition(data, recog_data)

        if mem_results is not None:
            print("\nPer-participant Hit Rate, FA Rate, and D-prime (Primary):")
            print("-" * 60)
            print(mem_results.to_string(index=False))
            print("-" * 60)

            print("\nGroup Summary (Primary):")
            summary = mem_results.groupby('control_condition')[
                ['Hit_rate', 'FA_rate', 'd_prime']
            ].agg(['mean', 'std'])
            print(summary.to_string())

            if supp_mem_results is not None:
                print("\nGroup Summary (Supplementary 2x2):")
                summary_supp = supp_mem_results.groupby(
                    ['trial_level', 'item_type']
                )[['Hit_rate', 'd_prime']].agg(['mean', 'std'])
                print(summary_supp.to_string())

            sys.stdout.flush()

            # Statistical Analyses (Long running - fitting models)
            run_recognition_stats(mem_results, targets, supp_mem_results, recog_data)

            # Recognition Performance Plots (pooled)
            plot_recognition_performance(supp_mem_results, POOLED_DIR)
            plot_dprime_by_condition(supp_mem_results, POOLED_DIR)

            # Analysis 7: Agency -> Memory (continuous agency_rating_z)
            targets_agency = run_agency_recognition_stats(targets)
            plot_agency_recognition(targets, POOLED_DIR)

    # ------------------------------------------------------------------
    # Per-participant plots
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PER-PARTICIPANT PLOTS")
    print("=" * 60)

    for _px in sorted(data['participant'].unique()):
        _px_suffix = f"_{int(_px)}"
        _px_data   = data[data['participant'] == _px].copy()
        print(f"\n  Participant {int(_px)} -> {PER_PARTICIPANT_DIR}")

        plot_sanity_check(_px_data, PER_PARTICIPANT_DIR, suffix=_px_suffix)
        plot_calibration_convergence(_px_data, PER_PARTICIPANT_DIR, suffix=_px_suffix)

        if recog_data is not None and supp_mem_results is not None:
            _px_supp = supp_mem_results[
                supp_mem_results['participant'] == _px
            ].copy()
            if not _px_supp.empty:
                plot_recognition_performance(_px_supp, PER_PARTICIPANT_DIR, suffix=_px_suffix)
                plot_dprime_by_condition(_px_supp, PER_PARTICIPANT_DIR, suffix=_px_suffix)
            else:
                print(f"    [WARNING] No supplementary recognition data for participant {int(_px)}. Skipping recognition plots.")

            if targets is not None:
                _px_targets = targets[targets['participant'] == _px].copy()
                if not _px_targets.empty:
                    plot_agency_recognition(_px_targets, PER_PARTICIPANT_DIR, suffix=_px_suffix)

    # ------------------------------------------------------------------
    # 13. Control Detection Accuracy (Manipulation Check)
    # ------------------------------------------------------------------
    run_cd_accuracy_analysis(data)

    # ------------------------------------------------------------------
    # 14. Analysis 5a: Agency ~ Accuracy + Control Level (Test Phase)
    # ------------------------------------------------------------------
    run_agency_current_trial_analysis(test_data)
