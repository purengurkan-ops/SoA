#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CDmem_1.py
==========
Control Detection  Experiment — Version 1

"""

import os
import sys
import math
import random
import pathlib
import datetime
import atexit
import hashlib
import json
import subprocess

import serial #for sending triggers via triggerbox
import time 


# ─────────────────────────────────────────────────────────────────────────────
#  PYTHON INTERPRETER CHECK
#  PsychoPy requires specific packages (numpy, pandas, psychopy). If they are
#  not available in the current Python environment, this block tries to find
#  a compatible Python installation and re-launches the script with it.
# ─────────────────────────────────────────────────────────────────────────────

def check_and_run_with_correct_python():
    """
    Checks whether the required packages are available. If not, searches
    for a Python interpreter that has them and re-launches this script.
    Returns False if packages are available (script should continue normally).
    """
    try:
        import numpy as np
        import pandas as pd
        from psychopy import visual, event, core, data, gui
        return False  # All good — continue with current interpreter
    except ImportError as e:
        print(f"Missing required packages: {e}")
        print("Searching for a compatible Python interpreter...")

        # Common locations for Python with PsychoPy installed
        python_paths = [
            "C:/Program Files/PsychoPy/python.exe",           # Standalone PsychoPy
            "C:/Users/knogl/Miniconda3/envs/psychopy_env/python.exe",
            "C:/Users/knogl/Miniconda3/python.exe",
            "/opt/anaconda3/bin/python",                       # macOS Anaconda
            "/usr/bin/python3",                                # Linux
        ]

        for path in python_paths:
            if os.path.exists(path):
                print(f"Found Python at: {path}")
                result = subprocess.run([path] + sys.argv, check=False)
                sys.exit(result.returncode)

        print("Error: No compatible Python found. Please install psychopy, numpy, and pandas.")
        sys.exit(1)


# Run the check — if it returns True, the script was re-launched and we exit.
if check_and_run_with_correct_python():
    sys.exit(0)

# ---------------------------------------------------------------------------
# INITIALIZE TRIGGERBOX
USE_TRIGGERS = False  # Set to False to disable EEG triggers manually

# Replace 'COM3' with the actual port found in Device Manager
try:
    if USE_TRIGGERS:
        port = serial.Serial('COM3') 
        port.write(b'\x00')  # Ensure it starts at zero
        TRIGGERBOX_READY = True
    else:
        print("EEG triggers are DISABLED via USE_TRIGGERS flag.")
        TRIGGERBOX_READY = False
        port = None
except Exception as e:
    print(f"WARNING: Could not connect to TriggerBox on COM3: {e}")
    print("Experiment will continue without EEG triggers.")
    TRIGGERBOX_READY = False
    port = None

def send_trigger(val):
    """Sends a trigger byte and resets it after a short delay."""
    if not TRIGGERBOX_READY or port is None:
        return
    try:
        port.write(bytes([val]))      # Set the trigger lines
        time.sleep(0.01)              # Wait 10ms (standard pulse width)
        port.write(b'\x00')           # Reset all lines to zero
    except Exception as e:
        print(f"Error sending trigger {val}: {e}")

# ---------------------------------------------------------------------------
#  IMPORTS (available after interpreter check)
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
from psychopy import visual, event, core, data, gui
from scipy.signal import butter, sosfilt_zi, sosfilt


# ─────────────────────────────────────────────────────────────────────────────
#  GLOBAL KINEMATICS STORAGE
#  Frame-by-frame mouse and shape position data are collected during each trial
#  and saved to a separate kinematics CSV at the end of the experiment.
# ─────────────────────────────────────────────────────────────────────────────

kinematics_data = []       # List of dicts, one per frame across all trials
kinematics_csv_path = ""   # Will be set after participant dialog
recognition_data = []      # List of dicts, one per trial in memory test
recognition_csv_path = ""  # Will be set after participant dialog


# ─────────────────────────────────────────────────────────────────────────────
#  AUTO-SAVE ON QUIT
#  If the experiment is closed unexpectedly (e.g. crash or Escape key),
#  this function saves whatever data has been collected so far.
# ─────────────────────────────────────────────────────────────────────────────

_saved = False  # Flag to prevent saving twice

def _save():
    """Save main trial data and kinematics data to CSV files."""
    global _saved
    if not _saved:
        if 'thisExp' in globals() and thisExp is not None:
            thisExp.saveAsWideText(csv_path)
            print("Main data auto-saved ->", csv_path)
            if kinematics_data:
                kinematics_df = pd.DataFrame(kinematics_data)
                kinematics_df.to_csv(kinematics_csv_path, index=False)
                print("Kinematics data auto-saved ->", kinematics_csv_path)
            if recognition_data:
                recognition_df = pd.DataFrame(recognition_data)
                recognition_df.to_csv(recognition_csv_path, index=False)
                print("Recognition data auto-saved ->", recognition_csv_path)
        else:
            print("Experiment not initialized — no data to save.")
        _saved = True

atexit.register(_save)  # Register _save to run automatically when Python exits


# ─────────────────────────────────────────────────────────────────────────────
#  PARTICIPANT DIALOG & EXPERIMENT SETTINGS
#  A dialog box collects participant ID and session number.
#  Two special modes are available:
#    - simulate: runs the experiment with a virtual mouse (no real input needed)
#    - check_mode: uses fewer trials for quick testing of the script
# ─────────────────────────────────────────────────────────────────────────────

expName = "CDmem_1"

# AUTO_TEST mode: skip the dialog entirely and run a fast simulation.
# Activated by setting the environment variable CDT_AUTO_TEST=true or
# passing --autotest as a command-line argument.
AUTO_TEST = os.environ.get('CDT_AUTO_TEST', '').lower() == 'true' or '--autotest' in sys.argv

if AUTO_TEST:
    print("AUTO-TEST MODE: Skipping dialog, running simulation")
    expInfo = {"participant": "AUTO_TEST", "session": "001"}
    SIMULATE = True
    CHECK_MODE = True
else:
    expInfo = {"participant": "", "session": "001", "age": "", "gender": "", "handedness": "", "simulate": False, "check_mode": False}
    dlg = gui.DlgFromDict(expInfo, order=["participant", "session", "age", "gender", "handedness", "simulate", "check_mode"], title=expName)
    if not dlg.OK:
        core.quit()  # User pressed Cancel
    SIMULATE = bool(expInfo.pop("simulate"))
    CHECK_MODE = bool(expInfo.pop("check_mode"))
    if SIMULATE:
        expInfo["participant"] = "SIM"

# Set trial counts depending on mode.
# CHECK_MODE uses minimal trials so the experimenter can quickly verify
# that the script runs correctly end-to-end.
if CHECK_MODE:
    CHECK_CALIBRATION_TRIALS = 6     # Minimum trials for QUEST+ (matches MTI check mode)
    CHECK_TEST_TRIALS_PER_LEVEL = 5  # Trials per difficulty level per block
else:
    CHECK_CALIBRATION_TRIALS = 60    # Full QUEST+ calibration (matches MTI)
    CHECK_TEST_TRIALS_PER_LEVEL = 20 # 20 trials per miniblock (6 miniblocks × 20 = 120 total)

if CHECK_MODE:
    print("=" * 60)
    print("** CHECK MODE ENABLED — Running minimal trials **")
    print(f"   Calibration: {CHECK_CALIBRATION_TRIALS} trials")
    print(f"   Test: {CHECK_TEST_TRIALS_PER_LEVEL} trials/miniblock × 6 miniblocks = {CHECK_TEST_TRIALS_PER_LEVEL * 6} total")
    print("=" * 60)
else:
    print("Running FULL experiment mode")
    print(f"   Calibration: {CHECK_CALIBRATION_TRIALS} trials")
    print(f"   Test: {CHECK_TEST_TRIALS_PER_LEVEL} trials/miniblock × 6 miniblocks = {CHECK_TEST_TRIALS_PER_LEVEL * 6} total")

# Start tracking total experiment time from this point (after dialog)
global_clock = core.Clock()

# ─────────────────────────────────────────────────────────────────────────────
#  SCREENSHOT SETTINGS
#  Screenshots are saved automatically at the first frame of each phase.
#  Saved to: CDmem/screenshots/
# ─────────────────────────────────────────────────────────────────────────────

SCREENSHOTS_DIR = pathlib.Path(__file__).parent / "screenshots"
SCREENSHOTS_DIR.mkdir(exist_ok=True)

# Tracks which phases have already had a screenshot saved this session.
# Keys: e.g. 'calibration_frame001', 'test_frame001', 'memory_test'
_screenshots_saved = set()


# ─────────────────────────────────────────────────────────────────────────────
#  MOTION LIBRARY
#  Pre-recorded cursor movement snippets are stored as velocity arrays in
#  core_pool.npy. Each snippet is a sequence of (dx, dy) displacements.
#  These are used to drive the autonomous movement of shapes on screen.
#
#  Additional files:
#    core_pool_feats.npy  — feature vectors for each snippet (used for clustering)
#    core_pool_labels.npy — cluster labels (k=4 clusters of movement styles)
#    scaler_params.json   — mean/std used to normalize features
#    cluster_centroids.json — centroid coordinates for each cluster
# ─────────────────────────────────────────────────────────────────────────────

script_dir = pathlib.Path(__file__).parent  # Directory containing this script

LIB_NAME    = script_dir / "Motion_Library" / "core_pool.npy"
FEATS_NAME  = script_dir / "Motion_Library" / "core_pool_feats.npy"
LABELS_NAME = script_dir / "Motion_Library" / "core_pool_labels.npy"

motion_pool      = np.load(LIB_NAME)    # Shape: (n_snippets, snippet_length, 2)
snippet_features = np.load(FEATS_NAME)  # Shape: (n_snippets, n_features)
snippet_labels   = np.load(LABELS_NAME) # Shape: (n_snippets,) — cluster IDs

SNIP_LEN    = motion_pool.shape[1]  # Number of frames per snippet
TOTAL_SNIPS = motion_pool.shape[0]  # Total number of snippets
K_CLUST     = 4                     # Number of movement clusters

print(f"Loaded {TOTAL_SNIPS} snippets × {SNIP_LEN} frames from {LIB_NAME}")
print(f"Cluster distribution: {np.bincount(snippet_labels)}")

# Load feature scaler parameters (used to normalize snippet features)
with open(script_dir / "Motion_Library" / "scaler_params.json", "r") as f:
    scp = json.load(f)
scaler_mean = np.array(scp["mean"],  dtype=np.float32)
scaler_std  = np.array(scp["scale"], dtype=np.float32)

# Load cluster centroids (used for trajectory quality scoring)
with open(script_dir / "Motion_Library" / "cluster_centroids.json", "r") as f:
    CLUSTER_CENTROIDS = np.array(json.load(f), dtype=np.float32)

# Create a participant-specific random number generator seeded by participant ID.
# Using a deterministic seed means the same participant always gets the same
# random sequence — useful for reproducibility.
participant_clusters = None
seed = int(hashlib.sha256(expInfo["participant"].encode()).hexdigest(), 16) & 0xFFFFFFFF
rng = np.random.default_rng(seed)


# ─────────────────────────────────────────────────────────────────────────────
#  TRAJECTORY QUALITY FUNCTIONS
#  Before using snippets in the experiment, we filter out low-quality ones
#  (e.g. snippets with too little movement, too much jitter, or erratic speed).
#  Valid snippets are then normalized to a consistent speed range.
# ─────────────────────────────────────────────────────────────────────────────

def get_trajectory_signature(trajectory):
    """
    Compute key movement statistics for a trajectory (position array).
    Used to score trajectory quality and select the best snippets.

    Parameters
    ----------
    trajectory : np.ndarray, shape (T, 2)
        Array of (x, y) positions over time.

    Returns
    -------
    dict with keys: mean_speed, speed_variability, path_length,
                    net_displacement, speed_percentiles
    """
    velocities = np.diff(trajectory, axis=0)  # Frame-to-frame displacements
    if len(velocities) == 0:
        return {'mean_speed': 0, 'speed_variability': 0, 'path_length': 0,
                'net_displacement': 0, 'speed_percentiles': np.array([0, 0, 0])}
    speeds = np.linalg.norm(velocities, axis=1)  # Scalar speed per frame
    return {
        'mean_speed':        np.mean(speeds),
        'speed_variability': np.std(speeds),
        'path_length':       np.sum(speeds),
        'net_displacement':  np.linalg.norm(trajectory[-1] - trajectory[0]),
        'speed_percentiles': np.percentile(speeds, [25, 50, 75])
    }


def analyze_trajectory_quality(trajectory):
    """
    Compute detailed quality metrics for a trajectory.

    Parameters
    ----------
    trajectory : np.ndarray, shape (T, 2)
        Position array (cumulative sum of velocity snippet).

    Returns
    -------
    dict with quality metrics used by is_trajectory_valid().
    """
    velocities = np.diff(trajectory, axis=0)
    speeds = np.linalg.norm(velocities, axis=1)
    mean_speed = np.mean(speeds)
    std_speed  = np.std(speeds)
    max_speed  = np.max(speeds)
    min_speed  = np.min(speeds)

    # Fraction of frames where the shape is nearly stationary
    zero_movement_ratio = np.sum(speeds < 0.5) / len(speeds)

    # Fraction of frames with unusually high speed (outliers)
    high_jitter_ratio = np.sum(speeds > mean_speed + 3 * std_speed) / len(speeds)

    # Compute direction changes (jerkiness) from unit velocity vectors
    if len(velocities) > 1:
        unit_velocities = velocities / (speeds.reshape(-1, 1) + 1e-9)
        angle_changes = np.arccos(
            np.clip(np.sum(unit_velocities[:-1] * unit_velocities[1:], axis=1), -1, 1)
        )
        mean_angle_change = np.mean(angle_changes)
        jerkiness = np.std(angle_changes)
    else:
        mean_angle_change = 0
        jerkiness = 0

    return {
        'mean_speed':          mean_speed,
        'std_speed':           std_speed,
        'zero_movement_ratio': zero_movement_ratio,
        'high_jitter_ratio':   high_jitter_ratio,
        'mean_angle_change':   mean_angle_change,
        'jerkiness':           jerkiness,
        'speed_range':         max_speed - min_speed
    }


def is_trajectory_valid(trajectory, min_speed=1.0, max_zero_ratio=0.3,
                         max_jitter_ratio=0.1, max_jerkiness=1.5):
    """
    Decide whether a trajectory meets quality standards.

    A trajectory is rejected if:
      - Mean speed is too low (shape barely moves)
      - Too many near-zero-speed frames (shape stalls)
      - Too many high-jitter frames (speed spikes)
      - Too jerky (erratic direction changes)

    Returns
    -------
    (bool, str) — (is_valid, reason_if_invalid)
    """
    quality = analyze_trajectory_quality(trajectory)
    if quality['mean_speed'] < min_speed:
        return False, "mean_speed_too_low"
    if quality['zero_movement_ratio'] > max_zero_ratio:
        return False, "too_much_zero_movement"
    if quality['high_jitter_ratio'] > max_jitter_ratio:
        return False, "too_much_jitter"
    if quality['jerkiness'] > max_jerkiness:
        return False, "too_jerky"
    return True, "valid"


def normalize_trajectory(trajectory, target_speed_range=(5.0, 15.0), smooth_factor=0.35):
    """
    Rescale and smooth a trajectory so all snippets have comparable speed.

    Steps:
      1. Scale all velocities so mean speed matches the target range midpoint.
      2. Apply exponential smoothing to reduce abrupt speed changes.

    Parameters
    ----------
    trajectory    : np.ndarray, shape (T, 2) — position array
    target_speed_range : (min, max) target mean speed in pixels/frame
    smooth_factor : weight given to the previous frame's velocity (0–1)

    Returns
    -------
    np.ndarray, shape (T, 2) — normalized position array
    """
    if len(trajectory) < 2:
        return trajectory
    velocities = np.diff(trajectory, axis=0)
    speeds = np.linalg.norm(velocities, axis=1)
    current_mean_speed = np.mean(speeds)
    if current_mean_speed > 0:
        target_mean_speed = np.mean(target_speed_range)
        speed_scale = target_mean_speed / current_mean_speed
        velocities = velocities * speed_scale

    # Exponential smoothing: each frame's velocity is a blend of the
    # previous smoothed velocity and the current raw velocity.
    smoothed_velocities = velocities.copy()
    for i in range(1, len(velocities)):
        smoothed_velocities[i] = (smooth_factor * smoothed_velocities[i - 1]
                                  + (1 - smooth_factor) * velocities[i])

    # Reconstruct position array from smoothed velocities
    normalized_trajectory = [trajectory[0]]
    for vel in smoothed_velocities:
        normalized_trajectory.append(normalized_trajectory[-1] + vel)
    return np.array(normalized_trajectory)


def preprocess_motion_pool():
    """
    Filter and normalize all snippets in the motion pool.

    For each snippet:
      1. Convert velocity array to position array (cumulative sum).
      2. Check quality — discard if it fails any criterion.
      3. Normalize speed and smoothness.
      4. Convert back to velocity array (differences of positions).

    Updates the global motion_pool, snippet_features, and snippet_labels
    in-place, keeping only valid snippets.

    Returns
    -------
    list of int — indices of valid snippets (0 to N-1 after filtering)
    """
    global motion_pool, snippet_features, snippet_labels, SNIP_LEN
    print("Preprocessing motion pool for quality control...")
    initial_count = len(motion_pool)

    processed_snippets  = []
    processed_features  = []
    processed_labels    = []

    for i, snippet in enumerate(motion_pool):
        # Convert velocity snippet to position trajectory for quality checks
        trajectory = np.cumsum(snippet, axis=0)
        is_valid, reason = is_trajectory_valid(trajectory)

        if is_valid:
            normalized_trajectory = normalize_trajectory(trajectory)
            # Convert back to velocities (differences between consecutive positions)
            velocities = np.diff(normalized_trajectory, axis=0)
            processed_snippets.append(velocities)
            processed_features.append(snippet_features[i])
            processed_labels.append(snippet_labels[i])
        else:
            print(f"  Removed snippet {i}: {reason}")

    # Replace global arrays with filtered versions
    motion_pool      = np.array(processed_snippets)
    snippet_features = np.array(processed_features)
    snippet_labels   = np.array(processed_labels)
    SNIP_LEN = motion_pool.shape[1] if len(motion_pool) > 0 else 0

    print(f"Motion pool preprocessed: kept {len(processed_snippets)}/{initial_count} snippets")
    return list(range(len(processed_snippets)))


# Run preprocessing immediately at startup
valid_snippet_indices = preprocess_motion_pool()


# ─────────────────────────────────────────────────────────────────────────────
#  UNIVERSAL TRAJECTORY SET
#  To ensure all participants see trajectories of comparable quality, we
#  pre-select a fixed set of the best snippets (ranked by a quality score).
#  This set is the same for every participant (seeded with 42, not the
#  participant seed), ensuring cross-participant comparability.
#
#  Primary set:  1,240 best snippets (used first)
#  Overflow set: next 40 snippets (used if primary runs out)
# ─────────────────────────────────────────────────────────────────────────────

# These globals are populated by select_universal_trajectory_set()
universal_trajectory_set_primary  = []
universal_trajectory_set_overflow = []
universal_trajectory_set          = []
used_trajectory_indices           = set()   # Tracks which snippets have been used
trajectory_usage_stats            = {"used_count": 0, "total_needed": 2000}

# Global trial counter — increments across all phases for consistent numbering
global_trial_counter = 0


def select_universal_trajectory_set():
    """
    Score all valid snippets by movement quality and select the top 1,240
    as the primary set (plus up to 40 overflow).

    Quality score = speed_score × variability_score × length_score
      - speed_score:       how close mean speed is to 8 px/frame
      - variability_score: how close speed std is to 3 px/frame
      - length_score:      path length relative to 100 px (capped at 1.0)

    Uses a fixed random seed (42) so the selection is identical for all
    participants.
    """
    global valid_snippet_indices
    global universal_trajectory_set_primary, universal_trajectory_set_overflow
    global universal_trajectory_set

    total_valid = len(valid_snippet_indices)
    if total_valid < 1240:
        print(f"Warning: Only {total_valid} valid trajectories (fewer than 1,240 needed)")
        universal_trajectory_set_primary  = valid_snippet_indices.copy()
        universal_trajectory_set_overflow = []
        universal_trajectory_set          = universal_trajectory_set_primary.copy()
        return universal_trajectory_set.copy()

    selection_rng = np.random.default_rng(42)  # Fixed seed — same for all participants
    print("Selecting universal trajectory sets (Primary 1,240 + Overflow 40)...")

    trajectory_scores = []
    for idx in valid_snippet_indices:
        trajectory = motion_pool[idx]
        traj_cumsum = np.cumsum(trajectory, axis=0)
        sig = get_trajectory_signature(traj_cumsum)

        speed_score       = 1.0 / (1.0 + abs(sig['mean_speed'] - 8.0))
        variability_score = 1.0 / (1.0 + abs(sig['speed_variability'] - 3.0))
        length_score      = min(1.0, sig['path_length'] / 100.0)
        overall_score     = speed_score * variability_score * length_score
        trajectory_scores.append((overall_score, idx))

    trajectory_scores.sort(reverse=True)  # Best snippets first

    primary_indices  = [idx for score, idx in trajectory_scores[:1240]]
    overflow_indices = ([idx for score, idx in trajectory_scores[1240:1280]]
                        if total_valid >= 1280 else [])

    universal_trajectory_set_primary  = primary_indices
    universal_trajectory_set_overflow = overflow_indices
    universal_trajectory_set          = primary_indices + overflow_indices

    print(f"  Primary: {len(primary_indices)} snippets (best score={trajectory_scores[0][0]:.3f})")
    if overflow_indices:
        print(f"  Overflow: {len(overflow_indices)} snippets")
    else:
        print("  No overflow set (valid < 1,280)")

    return universal_trajectory_set.copy()


# Initialize the universal set after preprocessing
universal_trajectory_set = select_universal_trajectory_set()
print(f"Universal set: Primary={len(universal_trajectory_set_primary)}, "
      f"Overflow={len(universal_trajectory_set_overflow)}, "
      f"Total={len(universal_trajectory_set)}")


def get_trajectory_indices(n_trajectories):
    """
    Select n unique trajectory indices for a single trial.

    Preference order:
      1. Unused snippets from the primary set
      2. Unused snippets from the overflow set (if primary exhausted)
      3. Any snippet from the combined set (if both exhausted)
      4. Emergency fallback: any valid snippet

    Parameters
    ----------
    n_trajectories : int — how many snippets to select (2 for this experiment)

    Returns
    -------
    list of int — selected snippet indices
    """
    global used_trajectory_indices, trajectory_usage_stats

    available_primary  = [i for i in universal_trajectory_set_primary
                          if i not in used_trajectory_indices]
    available_overflow = [i for i in universal_trajectory_set_overflow
                          if i not in used_trajectory_indices]
    available_indices  = (available_primary if len(available_primary) >= n_trajectories
                          else available_primary + available_overflow)

    if len(available_indices) >= n_trajectories:
        selected = rng.choice(available_indices, size=n_trajectories, replace=False)
        for idx in selected:
            used_trajectory_indices.add(idx)
        trajectory_usage_stats["used_count"] += n_trajectories
        return list(selected)
    else:
        # Fallback: reuse from combined set
        combined = universal_trajectory_set_primary + universal_trajectory_set_overflow
        if len(combined) >= n_trajectories:
            return list(rng.choice(combined, size=n_trajectories, replace=False))
        else:
            # Emergency: use any valid snippet
            return list(rng.choice(valid_snippet_indices, size=n_trajectories, replace=False))


# ─────────────────────────────────────────────────────────────────────────────
#  TRAJECTORY SMOOTHING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def apply_consistent_smoothing(trajectory1, trajectory2):
    """
    Apply a moving-average window to two trajectories simultaneously.
    This ensures both the target and distractor have similar smoothness,
    preventing participants from using smoothness as a cue.

    Parameters
    ----------
    trajectory1, trajectory2 : np.ndarray, shape (T, 2) — velocity arrays

    Returns
    -------
    (vel1, vel2) — smoothed velocity arrays
    """
    def smooth_trajectory(traj, window_size=3):
        if len(traj) < window_size:
            return traj
        smoothed = traj.copy()
        for i in range(len(traj)):
            start = max(0, i - window_size // 2)
            end   = min(len(traj), i + window_size // 2 + 1)
            smoothed[i] = np.mean(traj[start:end], axis=0)
        return smoothed

    # Convert velocity → position → smooth → back to velocity
    pos1 = np.cumsum(trajectory1, axis=0)
    pos2 = np.cumsum(trajectory2, axis=0)
    smooth_pos1 = smooth_trajectory(pos1)
    smooth_pos2 = smooth_trajectory(pos2)
    vel1 = np.diff(smooth_pos1, axis=0)
    vel2 = np.diff(smooth_pos2, axis=0)
    return vel1, vel2


def smooth_single_trajectory(trajectory):
    """Apply moving-average smoothing to a single velocity array."""
    def smooth_traj(traj, window_size=3):
        if len(traj) < window_size:
            return traj
        smoothed = traj.copy()
        for i in range(len(traj)):
            start = max(0, i - window_size // 2)
            end   = min(len(traj), i + window_size // 2 + 1)
            smoothed[i] = np.mean(traj[start:end], axis=0)
        return smoothed

    pos = np.cumsum(trajectory, axis=0)
    smooth_pos = smooth_traj(pos)
    vel = np.diff(smooth_pos, axis=0)
    return vel


# ─────────────────────────────────────────────────────────────────────────────
#  DIRECTION MIXING FUNCTION
#  This is the core of the control manipulation.
#
#  The target shape moves at the same speed as the distractor (trajectory
#  speed), but its DIRECTION is a weighted blend of:
#    - The participant's mouse direction (weight = prop)
#    - The pre-recorded trajectory direction (weight = 1 - prop)
#
#  By keeping speed constant and only varying direction, the task difficulty
#  is controlled purely by how much the mouse direction "leaks" into the
#  target's movement — making it harder to detect at low prop values.
# ─────────────────────────────────────────────────────────────────────────────

def mix_direction_only(mouse_dx, mouse_dy, traj_dx, traj_dy, prop):
    """
    Blend mouse and trajectory directions while preserving trajectory speed.

    Parameters
    ----------
    mouse_dx, mouse_dy : float — mouse displacement this frame
    traj_dx, traj_dy   : float — trajectory velocity this frame
    prop               : float in [0, 1] — proportion of mouse influence
                         (0 = fully autonomous, 1 = fully mouse-driven)

    Returns
    -------
    (vx, vy) : float — blended velocity with trajectory magnitude
    """
    traj_speed = math.hypot(traj_dx, traj_dy)
    if traj_speed < 0.01:
        return traj_dx, traj_dy  # Shape is nearly stationary — no mixing needed

    mouse_mag = math.hypot(mouse_dx, mouse_dy)
    if mouse_mag < 0.01:
        return traj_dx, traj_dy  # Mouse is stationary — follow trajectory only

    # Compute unit direction vectors
    m_dir_x, m_dir_y = mouse_dx / mouse_mag, mouse_dy / mouse_mag
    t_dir_x, t_dir_y = traj_dx / traj_speed, traj_dy / traj_speed

    # Blend directions: prop controls how much mouse direction is used
    mix_x = prop * m_dir_x + (1 - prop) * t_dir_x
    mix_y = prop * m_dir_y + (1 - prop) * t_dir_y

    # Normalize the blended direction and scale to trajectory speed
    mix_mag = math.hypot(mix_x, mix_y)
    if mix_mag > 0.01:
        return (mix_x / mix_mag) * traj_speed, (mix_y / mix_mag) * traj_speed
    return traj_dx, traj_dy


# ─────────────────────────────────────────────────────────────────────────────
#  DISPLAY & STIMULUS CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

OFFSET_X         = 300    # Horizontal distance from center to shape start position (px)
CHOICE_OFFSET_X  = 120    # Stimuli are moved closer to center during response (A/S) phase
OFFSET_Y         = 150    # Vertical distance from center to shape start position (px)
LOWPASS          = 0.2    # Low-pass filter weight: lower = less smoothing, more responsive
SPEED_MULTIPLIER = 1.3    # Multiply trajectory velocities to make shapes move faster

# ── Visual Angle / Stimulus Size Setup ───────────────────────────────────────
DIST_CM    = 90.0    # Distance from eyes to monitor
WIDTH_CM   = 53.0    # Physical width of the monitor
STIM_SZ_DEG = 3.0  # Desired visual angle size of the stimuli in degrees

def vis_ang_to_pix(deg, dist_cm, width_cm, win_width_pix):
    """
    Convert visual angle (degrees) to pixels based on monitor physics.
    Requires math.tan.
    """
    size_cm = 2 * dist_cm * math.tan(math.radians(deg) / 2)
    pix_per_cm = win_width_pix / width_cm
    return int(round(size_cm * pix_per_cm))


# ─────────────────────────────────────────────────────────────────────────────
#  DATASET CONSTANTS
#  Familiar images are sampled from CARA_prep chosen_stimuli dataset.
#  They are paired (e.g., alpaca_03s.jpg and alpaca_07s.jpg). We use one from
#  each pair in the test phase and the other in the memory test phase.
# ─────────────────────────────────────────────────────────────────────────────

STIM_DIR     = pathlib.Path(r"C:\Users\F11IT\Desktop\CDmem\chosen_stimuli_nolures")
IMAGE_SEED   = 42         # Fixed seed for reproducible sampling across runs
# IMAGE_SIZE will be calculated dynamically based on window width after win is created.

# Number of unique pairs needed for TEST phase = 120 trials (full) or 30 trials (check mode)
# Since each trial needs 2 UNIQUE images (Target & Distractor), we need 240 object concepts
# to run 120 trials with NO REPEATS.
N_IMAGES     = (CHECK_TEST_TRIALS_PER_LEVEL * 6 * 2) if CHECK_MODE else 240
IMAGE_LOG    = pathlib.Path(__file__).parent / "image_stimuli_log.json"


# ─────────────────────────────────────────────────────────────────────────────
#  IMAGE SAMPLING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def sample_and_log_images(img_dir, n, seed):
    """
    Split all images in `img_dir` into two non-overlapping groups.

    - All .jpg files are sorted (for determinism) then shuffled with `seed`.
    - First `n` images  → test phase (shown during motion trials).
    - Remaining images  → foils for the yes/no recognition memory test.

    No image appears in both phases.

    Returns
    -------
    test_images      : list of dicts {'filename', 'path'} — n images for test
    recognition_foils: list of dicts {'filename', 'path'} — remaining images
    """
    all_imgs = sorted(img_dir.glob("*.jpg"))  # sorted for determinism before shuffle

    if len(all_imgs) < n:
        raise ValueError(
            f"Not enough images in {img_dir}: found {len(all_imgs)}, need at least {n}."
        )

    rng_img  = random.Random(seed)   # isolated RNG — does not affect the experiment RNG
    shuffled = list(all_imgs)
    rng_img.shuffle(shuffled)

    test_imgs = shuffled[:n]
    foil_imgs = shuffled[n:n * 2]   # exactly n foils — any extras beyond n*2 are discarded

    test_images       = [{'filename': p.stem, 'path': str(p)} for p in test_imgs]
    recognition_foils = [{'filename': p.stem, 'path': str(p)} for p in foil_imgs]

    records = {
        'seed':        seed,
        'n_test':      n,
        'n_foils':     len(recognition_foils),
        'test_images': test_images,
        'foil_images': recognition_foils,
    }

    with open(IMAGE_LOG, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2)

    print(f"[Stimuli] {n} test + {len(recognition_foils)} foil images. Log → {IMAGE_LOG}")
    return test_images, recognition_foils




def make_image_pairs(images, seed):
    """
    Pair `test_images` into non-overlapping pairs for simultaneous presentation.
    
    Since each object concept only appears once in `test_images`, any random 
    pairing of two different list elements will naturally pair two DIFFERENT objects.
    
    Returns a list of tuples: [(dict_A, dict_B), ...]
    """
    assert len(images) % 2 == 0, "Need an even number of images to form pairs"
    
    # Shuffle locally to create random pairings
    rng_img = random.Random(seed)
    shuffled = list(images)
    rng_img.shuffle(shuffled)
    
    pairs = [(shuffled[i], shuffled[i + 1]) for i in range(0, len(shuffled), 2)]
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
#  FILE PATHS & EXPERIMENT HANDLER
#  PsychoPy's ExperimentHandler manages trial-by-trial data logging.
#  We also define paths for the main CSV and the kinematics CSV.
# ─────────────────────────────────────────────────────────────────────────────

root         = pathlib.Path(__file__).parent / "data"
subjects_dir = root / "subjects"
subjects_dir.mkdir(parents=True, exist_ok=True)

participant_id = expInfo['participant']
base_filename  = f"CDmem_1_{participant_id}"
csv_path             = subjects_dir / f"{base_filename}.csv"
kinematics_csv_path  = subjects_dir / f"{base_filename}_kinematics.csv"
recognition_csv_path = subjects_dir / f"{base_filename}_recognition.csv"

# If a file with this name already exists, append a number to avoid overwriting
i = 1
while csv_path.exists():
    new_filename         = f"CDmem_1_{participant_id}_{i}"
    csv_path             = subjects_dir / f"{new_filename}.csv"
    kinematics_csv_path  = subjects_dir / f"{new_filename}_kinematics.csv"
    recognition_csv_path = subjects_dir / f"{new_filename}_recognition.csv"
    i += 1

thisExp = data.ExperimentHandler(
    name=expName, extraInfo=expInfo,
    savePickle=False, saveWideText=False,
    dataFileName=str(root / base_filename)
)


# ─────────────────────────────────────────────────────────────────────────────
#  PSYCHOPY WINDOW & STIMULI
# ─────────────────────────────────────────────────────────────────────────────

# Open a fullscreen window (or windowed in simulate mode for easier debugging)
win = visual.Window((1920, 1080), fullscr=not SIMULATE, color=[0.5] * 3,
                    units="pix", allowGUI=True)
win.setMouseVisible(False)

# Compute scalable text sizes based on window resolution
# Base height factor: 0.05 means ~54 pixels on a 1080p screen (twice as large as original 26)
# SCALED_TEXT_HEIGHT = int(win.size[1] * 0.048)   # For instructions and messages
# SCALED_WRAP_WIDTH  = int(win.size[0] * 0.8)     # 80% of screen width for text wrapping

# Calculate dynamic stimulus size based on STIM_SZ_DEG visual degrees
# At 90 cm viewing distance, 53 cm monitor width, 1920px wide screen:
#   1.7° ≈  97 px  (previous — too small)
#   4.0° ≈ 228 px  (current  — between old test and old recognition size)
#  300px  (old recognition hardcoded)
target_size_pix = vis_ang_to_pix(STIM_SZ_DEG, DIST_CM, WIDTH_CM, win.size[0])
IMAGE_SIZE = (target_size_pix, target_size_pix)

# The two shapes used in CALIBRATION trials. Both will take up the computed size.
# Square gets width and height. Circle gets radius = width/2.
square = visual.Rect(win, target_size_pix, target_size_pix, fillColor="black", lineColor="black")
dot    = visual.Circle(win, target_size_pix / 2, fillColor="black", lineColor="black")

# ── Sample IMAGINE images and build trial pairs ───────────────────────────────
# Done here (after the window is open) so the log is always written before
# the experiment starts. The participant dialog has already run at this point.
sampled_test_images, foil_images = sample_and_log_images(STIM_DIR   , N_IMAGES, IMAGE_SEED)

# Build non-overlapping pairs for the test phase trials
# Use participant seed so the simultaneous pairings are random but reproducible per participant
_pair_seed = int(hashlib.md5(expInfo['participant'].encode()).hexdigest(), 16) % (2**32)
image_pairs = make_image_pairs(sampled_test_images, _pair_seed)

# Global index into image_pairs — incremented by each test trial
pair_index = 0

# Mapping to track if an image was controlled ('yes') or distractor ('no')
# Used to store the 'yes'/'no' control state and 'high'/'low' condition for memory test images
image_control_map = {}
image_condition_map = {}
# ── Foil images for the memory test ──────────────────────────────────────────
# We already populated `foil_images` from the paired subset sampled above.
# The `foil_images` list contains exactly N_IMAGES (240) unseen images that are 
# the paired counterparts to the `sampled_test_images`.

# Fixation cross shown at the start of each trial
fix = visual.TextStim(win, "+", color="#616161", height=60)

# General-purpose message text (instructions, response prompts, etc.)
msg = visual.TextStim(win, "", color="#616161", height=26, wrapWidth=1000, bold=True)

# Feedback text shown after calibration trials ("Right" / "Wrong")
feedbackTxt = visual.TextStim(win, "", color="#616161", height=40)

# Helper: confine a position to a circle of radius l around the screen center.
# Prevents shapes from flying off-screen.
confine = lambda p, l=400: p if (r := math.hypot(*p)) <= l else (p[0] * l / r, p[1] * l / r)


# ─────────────────────────────────────────────────────────────────────────────
#  SIMULATION HELPERS
#  When SIMULATE=True, a virtual mouse replaces real mouse input and key
#  presses are generated automatically. This allows the script to be tested
#  without a human participant.
# ─────────────────────────────────────────────────────────────────────────────

class SimulatedMouse:
    """A fake mouse that drifts randomly, used in simulation mode."""
    def __init__(self):
        self._pos = np.array([0.0, 0.0], dtype=float)

    def setPos(self, pos=(0, 0)):
        self._pos = np.array(pos, dtype=float)

    def getPos(self):
        # Add small random noise to simulate natural hand movement
        self._pos += rng.normal(0, 3, 2)
        return self._pos.tolist()


def wait_keys(keys=None):
    """
    Wait for a key press. In simulation mode, returns a random valid key
    immediately without waiting for real input.
    """
    if SIMULATE:
        if keys is None:
            core.wait(0.2)
            return ["space"]
        allowed = [k for k in keys if k != "escape"] or ["space"]
        return [rng.choice(allowed)]
    return event.waitKeys(keyList=keys)


# ─────────────────────────────────────────────────────────────────────────────
#  BREAK SCREEN
#  Shown automatically every 50 trials within a phase, and between blocks.
# ─────────────────────────────────────────────────────────────────────────────

def show_break_screen(trials_completed, total_trials_in_block, block_label):
    """
    Display a 30-second countdown break screen.

    Parameters
    ----------
    trials_completed      : int — how many trials done so far in this phase
    total_trials_in_block : int — total trials in this phase
    block_label           : str — descriptive label for the current block
    """
    break_msg = visual.TextStim(
        win=win,
        text=f"BREAK TIME\n\nCompleted {trials_completed} trials.\n"
             f"Progress: {trials_completed}/{total_trials_in_block} ({block_label})\n\n"
             f"Break time remaining: 30 seconds",
        pos=(0, 50), color='#616161', height=30, wrapWidth=800, bold=True
    )
    countdown_text = visual.TextStim(win=win, text='30', pos=(0, -100), color='yellow', height=60, bold=True)

    break_clock = core.Clock()
    while break_clock.getTime() < 30.0:
        remaining = 30 - int(break_clock.getTime())
        countdown_text.text = str(remaining)
        break_msg.text = (f"BREAK TIME\n\nCompleted {trials_completed} trials.\n"
                          f"Progress: {trials_completed}/{total_trials_in_block} ({block_label})\n\n"
                          f"Break time remaining: {remaining} seconds")
        break_msg.draw()
        countdown_text.draw()
        win.flip()

        if not SIMULATE:
            if event.getKeys(['escape']):
                _save(); core.quit()
        core.wait(0.1)

    # After countdown: show "press space to continue"
    visual.TextStim(
        win=win,
        text=f"BREAK COMPLETE\n\nCompleted {trials_completed}/{total_trials_in_block} trials.\n\nPress SPACE to continue.",
        pos=(0, 0), color='#616161', height=30, wrapWidth=800, bold=True
    ).draw()
    win.flip()
    wait_keys(['space', 'escape'])


# ─────────────────────────────────────────────────────────────────────────────
#  QUEST+ HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def logit(x):
    """Convert a probability to logit (log-odds) scale."""
    x = float(np.clip(x, 1e-6, 1 - 1e-6))
    return float(np.log(x / (1 - x)))


def inv_logit(z):
    """Convert a logit value back to probability."""
    return float(1.0 / (1.0 + np.exp(-z)))


def clamp_prop(s):
    """Clamp a prop value to the valid experiment range [0.02, 0.90]."""
    return float(np.clip(s, 0.02, 0.90))


# ─────────────────────────────────────────────────────────────────────────────
#  QUEST+ STAIRCASE
#  Bayesian adaptive procedure to estimate the psychometric function.
#  Uses entropy-minimising stimulus selection.
#  Adapted from: MT Inference.py (SimonKnogler / GitHub)
# ─────────────────────────────────────────────────────────────────────────────

class QuestPlusStaircase:
    def __init__(self, target_type):
        """
        QUEST+ implementation with entropy-based stimulus selection.

        Parameters
        ----------
        target_type : str — which accuracy target this staircase aims for:
            'low'     — targets ~55% correct; prior centred on a low prop
                        (logit(0.40)), probing the hard end of the curve
            'high'    — targets ~85% correct; prior centred on a high prop
                        (logit(0.80)), probing the easy end of the curve
            'neutral' — legacy; centred at logit(0.40), not used in main flow
        """
        self.s_grid      = np.linspace(logit(0.05), logit(0.90), 61)
        self.alpha_grid  = np.linspace(logit(0.05), logit(0.90), 61)
        self.beta_grid   = np.geomspace(1.0, 12.0, 25)
        self.lambda_grid = np.array([0.00, 0.01, 0.02, 0.04, 0.06])
        self.gamma = 0.5
        self.target_type = target_type

        if target_type == "high":
            # Prior centred at prop=0.80 — easy region for ~85% correct
            # changed to 0.70 to make it harder since a participant got 100% in the pilot
            alpha_mu = logit(0.70)
        elif target_type == "low":
            # Prior centred at prop=0.40 — hard region for ~55% correct
            alpha_mu = logit(0.40)
        else:  # 'neutral' — legacy calibration mode
            alpha_mu = logit(0.40)
        alpha_sd = 1.0

        self.prior_alpha = np.exp(-0.5 * ((self.alpha_grid - alpha_mu) / alpha_sd) ** 2)
        self.prior_alpha /= self.prior_alpha.sum()

        beta_mean, beta_gsd = 2.5, 2.0
        self.prior_beta = np.exp(-0.5 * ((np.log(self.beta_grid) - np.log(beta_mean)) / np.log(beta_gsd)) ** 2)
        self.prior_beta /= self.prior_beta.sum()

        self.prior_lambda = np.ones_like(self.lambda_grid) / len(self.lambda_grid)

        self.post_alpha  = self.prior_alpha.copy()
        self.post_beta   = self.prior_beta.copy()
        self.post_lambda = self.prior_lambda.copy()

        self.trial_count = 0
        self.responses   = []

    def psychometric(self, s_logit, alpha, beta, lapse):
        """p(correct | s; α, β, λ) = γ + (1 − γ − λ) σ(β [s − α])"""
        sigmoid = 1.0 / (1.0 + np.exp(-beta * (s_logit - alpha)))
        return self.gamma + (1.0 - self.gamma - lapse) * sigmoid

    def compute_entropy(self, posterior):
        """Shannon entropy of a probability distribution."""
        posterior = posterior + 1e-12
        return -np.sum(posterior * np.log(posterior))

    def select_stimulus_entropy_fast(self):
        """Select next stimulus by expected-entropy minimisation (subsampled grid for speed)."""
        s_grid_subset   = self.s_grid[::3]
        current_entropy = self.compute_entropy(self.post_alpha)
        best_stimulus, max_info_gain = None, -np.inf

        alpha_mean  = np.sum(self.alpha_grid  * self.post_alpha)
        beta_mean   = np.sum(self.beta_grid   * self.post_beta)
        lambda_mean = np.sum(self.lambda_grid * self.post_lambda)

        for s_logit in s_grid_subset:
            p_correct   = self.psychometric(s_logit, alpha_mean, beta_mean, lambda_mean)
            p_incorrect = 1.0 - p_correct
            if p_correct < 1e-6 or p_incorrect < 1e-6:
                continue

            post_c = np.zeros_like(self.post_alpha)
            post_i = np.zeros_like(self.post_alpha)
            for idx, alpha in enumerate(self.alpha_grid):
                lc = self.psychometric(s_logit, alpha, beta_mean, lambda_mean)
                post_c[idx] = self.post_alpha[idx] * lc
                post_i[idx] = self.post_alpha[idx] * (1.0 - lc)
            post_c /= (post_c.sum() + 1e-12)
            post_i /= (post_i.sum() + 1e-12)

            expected_entropy = (p_correct * self.compute_entropy(post_c) +
                                p_incorrect * self.compute_entropy(post_i))
            info_gain = current_entropy - expected_entropy
            if info_gain > max_info_gain:
                max_info_gain = info_gain
                best_stimulus = s_logit

        if best_stimulus is None:
            best_stimulus = self.s_grid[len(self.s_grid) // 2]
        return clamp_prop(inv_logit(best_stimulus))

    def select_stimulus_entropy(self):
        """Select stimulus using fast entropy approximation."""
        return self.select_stimulus_entropy_fast()

    def update(self, stimulus_prop, correct):
        """Bayesian update of the posterior after observing a response."""
        s_logit  = logit(clamp_prop(stimulus_prop))
        new_post = np.zeros((len(self.alpha_grid), len(self.beta_grid), len(self.lambda_grid)))
        for i, alpha in enumerate(self.alpha_grid):
            for j, beta in enumerate(self.beta_grid):
                for k, lapse in enumerate(self.lambda_grid):
                    w = self.post_alpha[i] * self.post_beta[j] * self.post_lambda[k]
                    p = self.psychometric(s_logit, alpha, beta, lapse)
                    new_post[i, j, k] = w * (p if correct else 1.0 - p)
        new_post /= (new_post.sum() + 1e-12)
        self.post_alpha  = new_post.sum(axis=(1, 2))
        self.post_beta   = new_post.sum(axis=(0, 2))
        self.post_lambda = new_post.sum(axis=(0, 1))
        self.trial_count += 1
        self.responses.append((s_logit, correct))

    def get_threshold_sd(self):
        """Standard deviation of the alpha (threshold) posterior in logits."""
        mu  = np.sum(self.alpha_grid * self.post_alpha)
        var = np.sum(self.post_alpha * (self.alpha_grid - mu) ** 2)
        return float(np.sqrt(var))

    def get_threshold_mean(self):
        """Mean of the alpha (threshold) posterior in logits."""
        return float(np.sum(self.alpha_grid * self.post_alpha))

    def posterior_summary(self):
        """Summary statistics of alpha, beta, and lambda posteriors."""
        a_mu  = np.sum(self.alpha_grid  * self.post_alpha)
        a_sd  = np.sqrt(np.sum(self.post_alpha  * (self.alpha_grid  - a_mu) ** 2))
        b_mu  = np.sum(self.beta_grid   * self.post_beta)
        b_sd  = np.sqrt(np.sum(self.post_beta   * (self.beta_grid   - b_mu) ** 2))
        l_mu  = np.sum(self.lambda_grid * self.post_lambda)
        l_sd  = np.sqrt(np.sum(self.post_lambda * (self.lambda_grid - l_mu) ** 2))
        return {
            'alpha_mean':  float(a_mu),  'alpha_sd':  float(a_sd),
            'beta_mean':   float(b_mu),  'beta_sd':   float(b_sd),
            'lambda_mean': float(l_mu),  'lambda_sd': float(l_sd),
        }

    def threshold_for_target(self, p_target):
        """Find the prop value that yields p_target% correct under the current posterior."""
        lam_hat = np.sum(self.lambda_grid * self.post_lambda)
        if p_target > 1.0 - lam_hat:
            p_target = min(0.85, 1.0 - lam_hat - 0.02)
        best_diff, best_s = float('inf'), 0.5
        for s_logit in self.s_grid:
            p_pred = 0.0
            for i, alpha in enumerate(self.alpha_grid):
                for j, beta in enumerate(self.beta_grid):
                    for k, lapse in enumerate(self.lambda_grid):
                        w = self.post_alpha[i] * self.post_beta[j] * self.post_lambda[k]
                        p_pred += w * self.psychometric(s_logit, alpha, beta, lapse)
            diff = abs(p_pred - p_target)
            if diff < best_diff:
                best_diff = diff
                best_s    = inv_logit(s_logit)
        return clamp_prop(best_s)



# ─────────────────────────────────────────────────────────────────────────────
#  TRIAL FUNCTION: run_trial_2shapes
#  This function runs a single experimental trial with 2 shapes.
#
#  Trial timeline:
#    1. Fixation cross (1 second, or random interval if cue_dur_range given)
#    2. Shapes appear; wait for mouse movement to begin
#    3. Motion phase (3 seconds): shapes move, mouse influences target
#    4. Response phase: participant presses A (square) or S (circle)
#    5. Feedback ("Right"/"Wrong") — calibration trials only
#    6. Agency rating (1–7) — test trials only
#
#  Returns a dict of trial results for logging.
# ─────────────────────────────────────────────────────────────────────────────

# Minimum mouse speed (px/frame) to consider the mouse "moving".
# Below this threshold, the target follows its trajectory without mouse mixing.
MOUSE_MOVE_THRESHOLD = 0.5

# ── Butterworth anti-jiggle filter (ported from CDmem_jigglefree.py) ────────
# Applied to the raw mouse DISPLACEMENT each frame. Jiggle = rapid alternating
# ±displacement that cancels out above CUTOFF_HZ. Deliberate movement = low-
# frequency sustained displacement that passes through.
CUTOFF_HZ    = 3.5   # Hz  — frequencies above this are attenuated
BUTTER_ORDER = 2     # 2nd order → -40 dB/decade roll-off past cutoff
SAMPLE_HZ    = 60.0  # approximate monitor refresh rate
_butter_sos  = butter(BUTTER_ORDER, CUTOFF_HZ, btype='low', fs=SAMPLE_HZ, output='sos')
_butter_zi   = sosfilt_zi(_butter_sos)  # template state vector (n_sections × 2)
# ────────────────────────────────────────────────────────────────────────────

# Fixed duration of the motion phase in seconds.
MOTION_DURATION = 3.0


def run_trial_2shapes(trial_in_block, phase, mode, block_num=1,
                      prop_override=None, cue_dur_range=None, motion_dur=None,
                      response_window=None, control_condition=None,
                      image_pair=None):
    # image_pair : tuple of two dicts {'filename': str, 'path': str}, or None.
    #              When provided (test phase), images replace the shapes.
    #              When None (calibration phase), the original shapes are used.
    """
    Run a single 2-shape trial.

    Parameters
    ----------
    trial_in_block  : int   — trial number within the current phase (resets per block)
    phase           : str   — 'calibration' or 'test'
    mode            : str   — 'staircase' or 'test' (informational only)
    block_num       : int   — current block number (for logging)
    prop_override   : float — self-proportion to use (overrides staircase)
    cue_dur_range   : tuple — (min, max) seconds for fixation duration
    motion_dur      : float — motion phase duration (not used; MOTION_DURATION is fixed)
    response_window : float — response time limit (not used; unlimited response time)
    control_condition: str   — level label for logging (e.g. 'low')

    Returns
    -------
    dict — trial results including accuracy, RT, ratings, and evidence metrics
    """

    # ── Set the self-proportion for this trial ──────────────────────────────
    if prop_override is not None:
        prop = float(np.clip(prop_override, 0.02, 0.90))
    else:
        prop = 0.40

    # ── Decide whether this is an image trial or a shape trial ─────────────
    use_images = (image_pair is not None)

    if use_images:
        img_A_info, img_B_info = image_pair
        stim_A = visual.ImageStim(win, image=img_A_info['path'], size=IMAGE_SIZE)
        stim_B = visual.ImageStim(win, image=img_B_info['path'], size=IMAGE_SIZE)
        stim_left_label  = 'img_A'
        stim_right_label = 'img_B'
    else:
        stim_A = square
        stim_B = dot
        stim_left_label  = 'square'
        stim_right_label = 'dot'

    # ── Fixation cross ───────────────────────────────────────────────────────
    fix.color = "#616161"
    if not use_images:
        square.fillColor = square.lineColor = "black"
        dot.fillColor    = dot.lineColor    = "black"

    fix.draw(); win.flip()
    if cue_dur_range is not None:
        core.wait(float(rng.uniform(cue_dur_range[0], cue_dur_range[1])))
    else:
        core.wait(1.0)

    # ── Position stimuli ─────────────────────────────────────────────────────
    # Randomly place stim_A on left or right; stim_B takes the other side.
    left_stim = random.choice(['A', 'B'])
    if left_stim == 'A':
        stim_A.pos = (-OFFSET_X, 0); stim_B.pos = (OFFSET_X, 0)
        left_label  = stim_left_label   # 'img_A' or 'square'
        right_label = stim_right_label  # 'img_B' or 'dot'
    else:
        stim_A.pos = (OFFSET_X, 0);  stim_B.pos = (-OFFSET_X, 0)
        left_label  = stim_right_label  # 'img_B' or 'dot'
        right_label = stim_left_label   # 'img_A' or 'square'

    # Save starting positions so we can reset the images at response time
    start_pos_A = tuple(stim_A.pos)
    start_pos_B = tuple(stim_B.pos)

    stim_A.draw(); stim_B.draw(); win.flip()

    # ── EEG Triggers: Stimulus Onset ────────────────────────────────────────
    trigger_stim_onset = np.nan
    if phase == "test" and control_condition:
        try:
            level_idx = 1 if control_condition == 'low' else 3
            trigger_stim_onset = 10 + level_idx
            send_trigger(trigger_stim_onset)
        except (ValueError, IndexError):
            pass

    # ── Wait for mouse movement to start ────────────────────────────────────
    mouse = SimulatedMouse() if SIMULATE else event.Mouse(win=win, visible=False)
    mouse.setPos((0, 0))
    last = mouse.getPos()
    while True:
        stim_A.draw(); stim_B.draw(); win.flip()
        x, y = mouse.getPos()
        if math.hypot(x - last[0], y - last[1]) > 0 or SIMULATE:
            break
        if not SIMULATE and event.getKeys(["escape"]):
            _save(); core.quit()

    # ── Select target and trajectories ──────────────────────────────────────
    # FIX: target is now chosen from [left_label, right_label] so it is always
    # in the same coordinate system as response_controlled (physical side),
    # rather than from [stim_left_label, stim_right_label] (image identity).
    target = random.choice([left_label, right_label])

    # If using images, record which image was controlled for later memory analysis.
    # We derive this from whether target matches the label on each side.
    if use_images:
        image_control_map[img_A_info['filename']] = 'yes' if target == left_label and left_label == 'img_A' \
                                                    else 'yes' if target == right_label and right_label == 'img_A' \
                                                    else 'no'
        image_control_map[img_B_info['filename']] = 'yes' if target == left_label and left_label == 'img_B' \
                                                    else 'yes' if target == right_label and right_label == 'img_B' \
                                                    else 'no'
        image_condition_map[img_A_info['filename']] = control_condition
        image_condition_map[img_B_info['filename']] = control_condition
    # Get 2 unique trajectory snippets: one for target, one for distractor.
    trajectory_indices = get_trajectory_indices(2)
    target_snippet_idx, distractor_snippet_idx = trajectory_indices[0], trajectory_indices[1]

    target_snippet     = motion_pool[target_snippet_idx]
    distractor_snippet = motion_pool[distractor_snippet_idx]
    target_snippet, distractor_snippet = apply_consistent_smoothing(
        target_snippet, distractor_snippet
    )

    # ── Motion phase setup ───────────────────────────────────────────────────
    trial_kinematics = []
    clk   = core.Clock()
    frame = 0
    vt    = np.zeros(2, np.float32)
    vd    = np.zeros(2, np.float32)

    # ── Butterworth filter state — reset each trial ──────────────────────────
    # Initialised to zero because displacement starts at 0 (mouse stationary).
    zi_x = _butter_zi.copy() * 0.0
    zi_y = _butter_zi.copy() * 0.0
    last = list(mouse.getPos())  # track RAW position for displacement
    # ────────────────────────────────────────────────────────────────────────

    event.clearEvents(eventType='keyboard')

    # ── EEG Triggers: Motion Start ───────────────────────────────────────────
    trigger_motion_start = np.nan
    if phase == "test" and control_condition:
        try:
            level_idx = 1 if control_condition == 'low' else 3
            trigger_motion_start = 20 + level_idx
            send_trigger(trigger_motion_start)
        except (ValueError, IndexError):
            pass

    _SCREENSHOT_FRAMES = {1: 'frame001', 30: 'frame030'}

    # ── Motion phase: accumulate time only while mouse is actively moving ─────
    # Shapes freeze (and MOTION_DURATION countdown pauses) whenever the
    # participant stops moving. Trial ends after MOTION_DURATION seconds of
    # *actual* mouse movement have elapsed.
    active_motion_time = 0.0   # Seconds of mouse-active movement accumulated
    last_flip_time     = clk.getTime()  # Wall-clock time at last frame

    while active_motion_time < MOTION_DURATION:
        x, y = mouse.getPos()
        now = clk.getTime()
        frame_dt = now - last_flip_time   # Elapsed wall-clock time this frame
        last_flip_time = now

        # ── Butterworth anti-jiggle filter on mouse displacement ─────────────
        raw_dx = x - last[0]
        raw_dy = y - last[1]
        last = [x, y]  # always update from RAW position

        filt_x, zi_x = sosfilt(_butter_sos, [raw_dx], zi=zi_x)
        filt_y, zi_y = sosfilt(_butter_sos, [raw_dy], zi=zi_y)
        dx, dy = float(filt_x[0]), float(filt_y[0])
        # ────────────────────────────────────────────────────────────────────

        mouse_speed = math.hypot(dx, dy)
        MAX_SPEED = 20.0
        if mouse_speed > MAX_SPEED:
            scale_factor = MAX_SPEED / mouse_speed
            dx *= scale_factor
            dy *= scale_factor
            mouse_speed = MAX_SPEED

        mouse_is_moving = mouse_speed > MOUSE_MOVE_THRESHOLD

        if mouse_is_moving:
            # Advance trajectory frame and accumulate active time
            active_motion_time += frame_dt
            frame += 1

            target_traj_dx,     target_traj_dy     = target_snippet[frame % len(target_snippet)]
            distractor_traj_dx, distractor_traj_dy = distractor_snippet[frame % len(distractor_snippet)]

            target_traj_dx     *= SPEED_MULTIPLIER
            target_traj_dy     *= SPEED_MULTIPLIER
            distractor_traj_dx *= SPEED_MULTIPLIER
            distractor_traj_dy *= SPEED_MULTIPLIER

            tdx, tdy = mix_direction_only(dx, dy, target_traj_dx, target_traj_dy, prop)
            ddx, ddy = distractor_traj_dx, distractor_traj_dy

            vt = LOWPASS * vt + (1 - LOWPASS) * np.array([tdx, tdy])
            vd = LOWPASS * vd + (1 - LOWPASS) * np.array([ddx, ddy])

            vm       = np.array([dx, dy], dtype=float)
            vm_speed = np.linalg.norm(vm) + 1e-9

            vt_disp = np.array(vt, dtype=float)
            vd_disp = np.array(vd, dtype=float)

            ut = vt_disp / (np.linalg.norm(vt_disp) + 1e-9)
            ud = vd_disp / (np.linalg.norm(vd_disp) + 1e-9)

            cos_T = np.dot(vm, ut) / vm_speed
            cos_D = np.dot(vm, ud) / vm_speed

            evidence = (cos_T - cos_D) * mouse_speed

            # Update shape positions only when mouse is moving
            if (left_stim == 'A' and target == left_label) or \
               (left_stim == 'B' and target == right_label):
                stim_A.pos = confine(tuple(np.array(stim_A.pos) + vt))
                stim_B.pos = confine(tuple(np.array(stim_B.pos) + vd))
            else:
                stim_B.pos = confine(tuple(np.array(stim_B.pos) + vt))
                stim_A.pos = confine(tuple(np.array(stim_A.pos) + vd))

        else:
            # Mouse is stationary: shapes stay frozen; no position update
            evidence = 0.0

        trial_kinematics.append({
            'timestamp':       clk.getTime(),
            'frame':           frame,
            'mouse_x':         x,
            'mouse_y':         y,
            'mouse_speed':     mouse_speed,
            'mouse_is_moving': mouse_is_moving,
            'active_motion_time': active_motion_time,
            'stim_A_x':        stim_A.pos[0],
            'stim_A_y':        stim_A.pos[1],
            'stim_B_x':        stim_B.pos[0],
            'stim_B_y':        stim_B.pos[1],
            'evidence':        evidence
        })

        if not SIMULATE:
            if event.getKeys(['escape']):
                _save(); core.quit()

        stim_A.draw(); stim_B.draw(); win.flip()

        if frame in _SCREENSHOT_FRAMES:
            key = f"{phase}_{_SCREENSHOT_FRAMES[frame]}"
            if key not in _screenshots_saved:
                fname = SCREENSHOTS_DIR / f"screenshot_{key}.png"
                win.getMovieFrame(buffer='front')
                win.saveMovieFrames(str(fname))
                _screenshots_saved.add(key)
                print(f"[Screenshot] Saved: {fname}")

    # ── RESPONSE PHASE ───────────────────────────────────────────────────────
    event.clearEvents(eventType='keyboard')

    # Key mapping: always based on which image/shape is on which side
    key_to_label = {'a': left_label, 's': right_label}

    CHOICE_DURATION = 5.0

    if start_pos_A[0] < 0:
        stim_A.pos = (-CHOICE_OFFSET_X, 0)
        stim_B.pos = (CHOICE_OFFSET_X, 0)
        left_img_x, right_img_x = -CHOICE_OFFSET_X, CHOICE_OFFSET_X
    else:
        stim_A.pos = (CHOICE_OFFSET_X, 0)
        stim_B.pos = (-CHOICE_OFFSET_X, 0)
        left_img_x, right_img_x = -CHOICE_OFFSET_X, CHOICE_OFFSET_X

    img_y = 0
    img_half_h = IMAGE_SIZE[1] / 2
    label_y = img_y - img_half_h - 40

    key_label_A_stim = visual.TextStim(
        win, text="A", pos=(left_img_x, label_y),
        height=30, color='#616161', bold=True, alignText='center'
    )
    key_label_S_stim = visual.TextStim(
        win, text="S", pos=(right_img_x, label_y),
        height=30, color='#616161', bold=True, alignText='center'
    )

    question_text = "Which image did you control?" if use_images else "Which shape did you control?"
    choice_question = visual.TextStim(
        win, text=question_text,
        pos=(0, int(win.size[1] * 0.2)), height=26, color='#616161', wrapWidth=1000, bold=True
    )

    def draw_choice_screen():
        stim_A.draw()
        stim_B.draw()
        key_label_A_stim.draw()
        key_label_S_stim.draw()
        choice_question.draw()
        win.flip()

    # ── EEG Triggers: Response Screen Onset ─────────────────────────────────
    trigger_resp_onset = np.nan
    if phase == "test" and control_condition:
        try:
            level_idx = 1 if control_condition == 'low' else 3
            trigger_resp_onset = 30 + level_idx
            send_trigger(trigger_resp_onset)
        except (ValueError, IndexError):
            pass

    response_controlled = None
    rt_choice  = np.nan
    response_clock = core.Clock()

    draw_choice_screen()
    response_start_time = response_clock.getTime()

    if SIMULATE:
        sim_rt = random.uniform(1.5, 4.5)
        core.wait(sim_rt)
        # FIX: simulate response from [left_label, right_label] to match
        # the same coordinate system as target
        response_controlled = rng.choice([left_label, right_label])
        rt_choice  = response_clock.getTime()
        remaining  = CHOICE_DURATION - rt_choice
        if remaining > 0:
            core.wait(remaining)
    else:
        while response_clock.getTime() < CHOICE_DURATION:
            elapsed = response_clock.getTime()
            draw_choice_screen()
            keys = event.getKeys(['a', 's', 'escape'], timeStamped=True)
            if keys:
                key, key_time = keys[0]
                if key == 'escape':
                    _save(); core.quit()
                elif key in key_to_label and response_controlled is None:
                    response_controlled = key_to_label[key]
                    rt_choice  = elapsed
            core.wait(0.01)

        if response_controlled is None:
            response_controlled = 'timeout'
            timeout_msg = visual.TextStim(
                win, text="Please answer faster!",
                pos=(0, 0), height=30, color='yellow', bold=True
            )
            timeout_msg.draw(); win.flip()
            core.wait(2.0)

    # Accuracy: 1 if participant identified the correct target, 0 otherwise.
    correct = int(response_controlled == target)

    # ── EEG Triggers: Response Value ─────────────────────────────────────────
    trigger_resp_val = np.nan
    if phase == "test":
        trigger_resp_val = 41 if correct else 42
        send_trigger(trigger_resp_val)

    # ── FEEDBACK (calibration trials only) ───────────────────────────────────
    if phase == "calibration":
        feedbackTxt.text = "Right" if correct else "Wrong"
        feedbackTxt.draw(); win.flip(); core.wait(0.8)
        win.flip(); core.wait(0.3)

    # ── AGENCY RATING (test trials only) ─────────────────────────────────────
    agency_rating = np.nan
    if phase == "test":
        if SIMULATE:
            agency_rating = float(rng.integers(1, 8))
            core.wait(0.5)
        else:
            event.clearEvents(eventType='keyboard')
            msg.text = "How much control did you feel over the shape's movement?"
            msg.pos = (0, int(win.size[1] * 0.2))

            scale_text_height = 20
            scale_width = int(win.size[0] * 0.8)
            spacing = scale_width / 6
            start_x = -(scale_width / 2)

            line_y = -100
            text_y = line_y - scale_text_height * 1.5

            scale_positions = [(start_x + i * spacing, text_y) for i in range(7)]
            scale_labels = ["1\nVery weak", "2\nWeak", "3\nSomewhat weak", "4\nModerate",
                            "5\nSomewhat strong", "6\nStrong", "7\nVery strong"]

            scale_line = visual.Line(
                win, start=(start_x, line_y), end=(-start_x, line_y),
                lineColor='#616161', lineWidth=5
            )
            tick_half_height = scale_text_height * 0.3
            scale_ticks = [
                visual.Line(win, start=(x, line_y + tick_half_height), end=(x, line_y - tick_half_height),
                            lineColor='#616161', lineWidth=5)
                for x, _ in scale_positions
            ]
            scale_stimuli = [
                visual.TextStim(win, text=label, pos=pos, height=scale_text_height,
                                color='#616161', alignText='center', bold=True)
                for pos, label in zip(scale_positions, scale_labels)
            ]

            rating = None
            while rating is None:
                msg.draw()
                scale_line.draw()
                for tick in scale_ticks:
                    tick.draw()
                for stim in scale_stimuli:
                    stim.draw()
                win.flip()
                keys = event.getKeys(['1', '2', '3', '4', '5', '6', '7', 'escape'])
                if keys:
                    if 'escape' in keys:
                        _save(); core.quit()
                    else:
                        rating = int(keys[0])
                        msg.draw()
                        scale_line.draw()
                        for tick in scale_ticks:
                            tick.draw()
                        for stim in scale_stimuli:
                            stim.draw()
                        win.flip()
                        core.wait(0.5)
                core.wait(0.01)
            agency_rating = rating
            msg.pos = (0, 0)

    # ── COMPUTE SUMMARY EVIDENCE METRICS ─────────────────────────────────────
    frame_evidence = [d['evidence'] for d in trial_kinematics]
    mean_evidence  = np.mean(frame_evidence) if frame_evidence else np.nan
    sum_evidence   = np.sum(frame_evidence)  if frame_evidence else np.nan
    var_evidence   = np.var(frame_evidence)  if frame_evidence else np.nan

    # ── ADD TRIAL METADATA TO KINEMATICS ─────────────────────────────────────
    for frame_data in trial_kinematics:
        frame_data.update({
            'overall_trial_num':  global_trial_counter,
            'trial_in_block':     trial_in_block,
            'phase':              phase,
            'n_shapes':           2,
            'target':             target,
            'prop_used':          prop,
            'block_num':          block_num,
            'control_condition':  control_condition,
            'participant':        expInfo.get('participant'),
            'session':            expInfo.get('session'),
            'age':                expInfo.get('age'),
            'gender':             expInfo.get('gender'),
            'handedness':         expInfo.get('handedness')
        })
    kinematics_data.extend(trial_kinematics)

    return dict(
        n_shapes=2,
        target_snippet_id=target_snippet_idx,
        distractor_snippet_ids=[distractor_snippet_idx],
        phase=phase,
        block_num=block_num,
        true_controlled=target,
        response_controlled=response_controlled,
        detection_accuracy=correct,
        rt_choice=rt_choice,
        agency_rating=agency_rating,
        prop_used=prop,
        early_response=False,
        mean_evidence=mean_evidence,
        sum_evidence=sum_evidence,
        var_evidence=var_evidence,
        control_condition=control_condition,
        img_A_name=img_A_info['filename'] if use_images else np.nan,
        img_B_name=img_B_info['filename'] if use_images else np.nan,
        trigger_stim_onset=trigger_stim_onset,
        trigger_motion_start=trigger_motion_start,
        trigger_resp_onset=trigger_resp_onset,
        trigger_resp_val=trigger_resp_val
    )


# ─────────────────────────────────────────────────────────────────────────────
#  CALIBRATION PHASE RUNNER (QUEST+)
#  Runs a QUEST+ Bayesian adaptive staircase to find the prop (cursor-mix
#  proportion) that yields a specific accuracy target (55% or 85% correct).
#
#  Adaptive stopping:
#    • Run at least min_trials (default 40)
#    • Stop when posterior SD < 0.20 AND trials ≥ min_trials
#    • If not converged by min_trials, add up to 20 extra trials (max 60)
#    • If still not converged after max_trials, show a failure screen and
#      continue with the best estimate available
# ─────────────────────────────────────────────────────────────────────────────

def run_calibration_quest(target_accuracy, num_trials, block_num=0):
    """
    Run a single QUEST+ calibration staircase targeting a specific accuracy.

    Parameters
    ----------
    target_accuracy : float — the proportion correct to calibrate to (0.55 or 0.85)
    num_trials      : int   — minimum number of calibration trials
    block_num       : int   — block number for logging (use 0 for first, -1 for second)

    Returns
    -------
    quest     : QuestPlusStaircase — fitted object
    prop      : float — estimated prop (cursor-mix) that yields target_accuracy correct
    converged : bool  — whether the staircase converged within max_trials
    """
    global global_trial_counter

    # Select the appropriate prior based on the accuracy target
    target_type = 'low' if target_accuracy <= 0.60 else 'high'
    quest        = QuestPlusStaircase(target_type)
    min_trials   = num_trials
    max_trials   = num_trials + 20
    sd_threshold = 0.20

    print(f"\nStarting QUEST+ calibration (target={target_accuracy:.0%}, "
          f"min={min_trials}, max={max_trials} trials, SD threshold={sd_threshold})")

    trial_num = 0
    converged = False
    while trial_num < max_trials:
        trial_num            += 1
        global_trial_counter += 1

        s_candidate = quest.select_stimulus_entropy()

        res = run_trial_2shapes(
            trial_num, "calibration", mode="staircase",
            prop_override=s_candidate, cue_dur_range=(0.5, 0.8),
            control_condition="calibration", block_num=block_num
        )

        # Update QUEST only for valid (non-timeout) responses
        if res.get('response_controlled') != 'timeout':
            quest.update(s_candidate, int(res.get('detection_accuracy', 0)))

        # Compute QUEST summary every 10 trials (expensive); otherwise fast SD
        if trial_num % 10 == 0 or trial_num < 10:
            summ           = quest.posterior_summary()
            quest_alpha_sd = summ['alpha_sd']
        else:
            quest_alpha_sd = quest.get_threshold_sd()
            summ           = None

        # ── Log trial data ────────────────────────────────────────────────────
        thisExp.addData('overall_trial_num',      global_trial_counter)
        thisExp.addData('participant',             expInfo['participant'])
        thisExp.addData('session',                 expInfo['session'])
        thisExp.addData('age',                     expInfo['age'])
        thisExp.addData('gender',                  expInfo['gender'])
        thisExp.addData('handedness',              expInfo['handedness'])
        thisExp.addData('phase',                   'calibration')
        thisExp.addData('calib_target',            target_accuracy)   # NEW: which staircase
        thisExp.addData('n_shapes',                2)
        thisExp.addData('block_num',               block_num)
        thisExp.addData('trial_in_block',          trial_num)
        thisExp.addData('prop_used',               s_candidate)
        thisExp.addData('stimulus_logit',          logit(s_candidate))
        thisExp.addData('detection_accuracy',      res.get('detection_accuracy', 0))
        thisExp.addData('is_timeout',              res.get('response_controlled') == 'timeout')
        thisExp.addData('rt_choice',               res.get('rt_choice', np.nan))
        thisExp.addData('early_response',          res.get('early_response', False))
        thisExp.addData('true_controlled',         res.get('true_controlled', ''))
        thisExp.addData('response_controlled',     res.get('response_controlled', ''))
        thisExp.addData('quest_alpha_sd',          quest_alpha_sd)
        thisExp.addData('mean_evidence',           res.get('mean_evidence', np.nan))
        thisExp.addData('sum_evidence',            res.get('sum_evidence', np.nan))
        thisExp.addData('var_evidence',            res.get('var_evidence', np.nan))
        thisExp.addData('target_snippet_id',       res.get('target_snippet_id', np.nan))
        thisExp.addData('distractor_snippet_ids',  str(res.get('distractor_snippet_ids', [])))

        if summ is not None:
            thisExp.addData('quest_alpha_mean',   summ['alpha_mean'])
            thisExp.addData('quest_beta_mean',    summ['beta_mean'])
            thisExp.addData('quest_beta_sd',      summ['beta_sd'])
            thisExp.addData('quest_lambda_mean',  summ['lambda_mean'])
            thisExp.addData('quest_lambda_sd',    summ['lambda_sd'])
        else:
            thisExp.addData('quest_alpha_mean',   np.nan)
            thisExp.addData('quest_beta_mean',    np.nan)
            thisExp.addData('quest_beta_sd',      np.nan)
            thisExp.addData('quest_lambda_mean',  np.nan)
            thisExp.addData('quest_lambda_sd',    np.nan)

        thisExp.nextEntry()

        if trial_num % 10 == 0:
            print(f"  Trial {trial_num}: prop={s_candidate:.3f}, "
                  f"alpha_sd={quest_alpha_sd:.4f}")

        # Adaptive stopping: converge after min_trials if posterior SD is small
        if trial_num >= min_trials and quest_alpha_sd < sd_threshold:
            print(f"  QUEST+ converged after {trial_num} trials "
                  f"(alpha_sd={quest_alpha_sd:.4f} < {sd_threshold})")
            converged = True
            break

    # ── Calibration failure screen ────────────────────────────────────────────
    if not converged:
        print(f"  WARNING: QUEST+ did NOT converge after {trial_num} trials "
              f"(alpha_sd={quest_alpha_sd:.4f}). Using best estimate.")

    prop    = quest.threshold_for_target(target_accuracy)
    summary = quest.posterior_summary()
    print(f"\nCalibration complete (target={target_accuracy:.0%}):")
    print(f"  Trials: {trial_num}, converged: {converged}, alpha_sd: {summary['alpha_sd']:.4f}")
    print(f"  Calibrated prop (cursor mix for {target_accuracy:.0%} correct): {prop:.3f}")

    return quest, prop, converged


# ─────────────────────────────────────────────────────────────────────────────
#  TEST BLOCK RUNNER
#  Runs all trials for a single difficulty level (one block).
#  Unlike the original MT Inference.py (which shuffled all 4 levels together),
#  here each block contains only one control level — enabling analysis of
#  within-block history effects at a fixed difficulty.
# ─────────────────────────────────────────────────────────────────────────────

def run_test_block_for_level(threshold_75, level_name, prop_value,
                              num_trials, block_num, angle_bias=0):
    """
    Run a test block where all trials use the same difficulty level.

    Parameters
    ----------
    threshold_75 : float — calibrated 75% threshold (logged for reference)
    level_name   : str   — control condition label (e.g. 'low')
    prop_value   : float — self-proportion for all trials in this block
    num_trials   : int   — number of trials in this block
    block_num    : int   — block number (1–4) for logging
    angle_bias   : int   — rotation applied to mouse input (0 = none)
    """
    global global_trial_counter, pair_index

    print(f"\nTest Block {block_num}: {level_name} (prop={prop_value:.3f}, "
          f"{num_trials} trials)")

    for trial_num in range(1, num_trials + 1):
        global_trial_counter += 1

        # Fetch the next unique image pair for this trial.
        # pair_index is a global counter so pairs never repeat across blocks.
        current_pair = image_pairs[pair_index % len(image_pairs)]
        pair_index += 1

        # Run one trial at the fixed prop_value for this block, using images
        res = run_trial_2shapes(
            trial_num, "test", mode="test",
            prop_override=prop_value, cue_dur_range=(0.5, 0.8),
            control_condition=level_name, block_num=block_num,
            image_pair=current_pair
        )

        # ── Log trial data ────────────────────────────────────────────────────
        thisExp.addData('overall_trial_num',      global_trial_counter)
        thisExp.addData('participant',             expInfo['participant'])
        thisExp.addData('session',                 expInfo['session'])
        thisExp.addData('age',                     expInfo['age'])
        thisExp.addData('gender',                  expInfo['gender'])
        thisExp.addData('handedness',              expInfo['handedness'])
        thisExp.addData('phase',                   'test')
        thisExp.addData('n_shapes',                2)
        thisExp.addData('block_num',               block_num)
        thisExp.addData('trial_in_block',          trial_num)
        thisExp.addData('control_condition',       level_name)
        thisExp.addData('prop_used',               prop_value)
        thisExp.addData('threshold_75',            threshold_75)
        thisExp.addData('detection_accuracy',      res.get('detection_accuracy', 0))
        thisExp.addData('is_timeout',              res.get('response_controlled') == 'timeout')
        thisExp.addData('rt_choice',               res.get('rt_choice', np.nan))
        thisExp.addData('agency_rating',           res.get('agency_rating', np.nan))
        thisExp.addData('early_response',          res.get('early_response', False))
        thisExp.addData('true_controlled',         res.get('true_controlled', ''))
        thisExp.addData('response_controlled',     res.get('response_controlled', ''))
        thisExp.addData('mean_evidence',           res.get('mean_evidence', np.nan))
        thisExp.addData('sum_evidence',            res.get('sum_evidence', np.nan))
        thisExp.addData('var_evidence',            res.get('var_evidence', np.nan))
        thisExp.addData('target_snippet_id',       res.get('target_snippet_id', np.nan))
        thisExp.addData('distractor_snippet_ids',  str(res.get('distractor_snippet_ids', [])))
        thisExp.addData('img_A_name',              res.get('img_A_name', np.nan))
        thisExp.addData('img_B_name',              res.get('img_B_name', np.nan))
        # Log EEG triggers
        thisExp.addData('trigger_stim_onset',      res.get('trigger_stim_onset', np.nan))
        thisExp.addData('trigger_motion_start',    res.get('trigger_motion_start', np.nan))
        thisExp.addData('trigger_resp_onset',      res.get('trigger_resp_onset', np.nan))
        thisExp.addData('trigger_resp_val',        res.get('trigger_resp_val', np.nan))
        thisExp.nextEntry()

        # Offer a break every 50 trials within a block
        if trial_num % 50 == 0 and trial_num < num_trials:
            show_break_screen(trial_num, num_trials, f"Block {block_num} ({level_name})")


# ─────────────────────────────────────────────────────────────────────────────
#  MEMORY TEST
#  After all 4 test blocks, participants complete a yes/no recognition test.
#  480 images are shown one at a time (240 seen during the experiment +
#  240 unseen foils). For each image, participants press:
#    A = Yes (I saw this during the experiment)
#    S = No  (I did not see this during the experiment)
#  Logged per item: filename, seen/unseen ground truth, response, accuracy, RT.
# ─────────────────────────────────────────────────────────────────────────────

def run_memory_test(seen_images, foil_images_list):
    """
    Run the yes/no recognition memory test.

    Parameters
    ----------
    seen_images      : list of dicts {'filename', 'path'} — the 240 images
                       shown during the experiment (ground truth = 'seen')
    foil_images_list : list of dicts {'filename', 'path'} — 240 new images
                       never shown during the experiment (ground truth = 'unseen')

    Each item is shown one at a time. Participant presses:
      A = Yes (seen before)   |   S = No (not seen before)
    No time limit per item.

    Logged per item (appended to the main CSV via thisExp):
      n_trial         : item number within the memory test (1–400)
      mem_filename    : image filename stem
      mem_ground_truth: 'seen' or 'unseen'
      mem_response    : 'yes' or 'no'
      mem_rt          : response time in seconds
    """
    global global_trial_counter

    # Build the full list: tag each image with its ground truth
    # Total count = 480 items (240 seen + 240 unseen)
    mem_items = (
        [{'filename': d['filename'], 'path': d['path'], 'ground_truth': 'seen'}
         for d in seen_images] +
        [{'filename': d['filename'], 'path': d['path'], 'ground_truth': 'unseen'}
         for d in foil_images_list]
    )

    # Shuffle with a participant-specific seed for reproducibility
    _mem_rng = random.Random(
        int(hashlib.md5((expInfo['participant'] + '_mem').encode()).hexdigest(), 16) % (2**32)
    )
    _mem_rng.shuffle(mem_items)

    print(f"\nMemory test: {len(mem_items)} items ({len(seen_images)} seen + {len(foil_images_list)} unseen)")

    # Pre-create a large ImageStim; we'll update its image each trial.
    # Use IMAGE_SIZE so the recognition phase matches the test phase exactly.
    mem_img_stim = visual.ImageStim(win, size=IMAGE_SIZE)

    # Question text sits below the image
    mem_question = visual.TextStim(
        win,
        text="Have you seen this image during the experiment before?",
        pos=(0, int(-win.size[1] * 0.18)), color='#616161', height=26, wrapWidth=1000, bold=True
    )
    # Key labels: Y = Yes (left), N = No (right)
    mem_key_Y = visual.TextStim(
        win, text="Y\nYes", pos=(-int(win.size[0] * 0.05), int(-win.size[1] * 0.25)),
        height=30, color='#616161', bold=True, alignText='center'
    )
    mem_key_N = visual.TextStim(
        win, text="N\nNo", pos=(int(win.size[0] * 0.05), int(-win.size[1] * 0.25)),
        height=30, color='#616161', bold=True, alignText='center'
    )

    # Fixation cross for inter-trial interval
    mem_fix = visual.TextStim(win, text='+', pos=(0, 0), color='#616161',
                              height=60, bold=True)

    for item_num, item in enumerate(mem_items, start=1):
        global_trial_counter += 1

        # Load image for this item
        mem_img_stim.image = item['path']
        mem_img_stim.pos   = (0, 80)   # Slightly above centre; prompt sits below

        # Fixation cross — random duration 0.5–0.8 s
        fix_dur = random.uniform(0.5, 0.8)
        mem_fix.draw(); win.flip()
        core.wait(fix_dur)

        # Reset key label colours to white for every new trial
        mem_key_Y.color = '#616161'
        mem_key_N.color = '#616161'

        # Draw image + key labels and start timing
        event.clearEvents(eventType='keyboard')
        mem_img_stim.draw()
        mem_question.draw()
        mem_key_Y.draw()
        mem_key_N.draw()
        win.flip()
        item_onset = core.getTime()

        # EEG Triggers: Recognition Stimulus Onset
        mem_trigger_onset = 51 if item['ground_truth'] == 'seen' else 52
        send_trigger(mem_trigger_onset)

        # Screenshot: capture the very first memory test item
        if item_num == 1 and 'memory_test' not in _screenshots_saved:
            fname = SCREENSHOTS_DIR / "screenshot_memory_test.png"
            win.getMovieFrame(buffer='front')
            win.saveMovieFrames(str(fname))
            _screenshots_saved.add('memory_test')
            print(f"[Screenshot] Saved: {fname}")

        # Wait for Y or N (no time limit)
        mem_response = None
        mem_rt       = np.nan

        if SIMULATE:
            sim_rt = random.uniform(1.0, 5.0)
            core.wait(sim_rt)
            mem_response = _mem_rng.choice(['yes', 'no'])
            mem_rt       = sim_rt
        else:
            while mem_response is None:
                keys = event.getKeys(['y', 'n', 'escape'], timeStamped=True)
                if keys:
                    key, key_time = keys[0]
                    if key == 'escape':
                        _save(); core.quit()
                    elif key == 'y':
                        mem_response = 'yes'
                        mem_rt       = key_time - item_onset
                        # Turn selected key feedback removed (no longer turns green)
                        mem_img_stim.draw()
                        mem_question.draw()
                        mem_key_Y.draw()
                        mem_key_N.draw()
                        win.flip()
                        core.wait(0.3)
                    elif key == 'n':
                        mem_response = 'no'
                        mem_rt       = key_time - item_onset
                        # Turn selected key feedback removed (no longer turns green)
                        mem_img_stim.draw()
                        mem_question.draw()
                        mem_key_Y.draw()
                        mem_key_N.draw()
                        win.flip()
                        core.wait(0.3)
                core.wait(0.01)

        # Accuracy: correct if 'yes' for seen, 'no' for unseen
        ground_truth  = item['ground_truth']
        mem_correct   = int(
            (mem_response == 'yes' and ground_truth == 'seen') or
            (mem_response == 'no'  and ground_truth == 'unseen')
        )

        # EEG Triggers: Recognition Participant Response
        # 61 = Correct (Hit or CR), 62 = Incorrect (Miss or FA)
        mem_trigger_resp = 61 if mem_correct else 62
        send_trigger(mem_trigger_resp)

        win.flip()  # clear screen; next trial's fixation cross follows immediately

        # Log to recognition data list (saved separately)
        recognition_data.append({
            'participant':      expInfo.get('participant'),
            'session':          expInfo.get('session'),
            'age':              expInfo.get('age'),
            'gender':           expInfo.get('gender'),
            'handedness':       expInfo.get('handedness'),
            'phase':            'memory_test',
            'overall_trial_num': global_trial_counter,
            'trial_in_block':   item_num,
            'mem_filename':     item['filename'],
            'mem_ground_truth': ground_truth,
            'controlled':       image_control_map.get(item['filename'], float('nan')),
            'trial_level':      image_condition_map.get(item['filename'], float('nan')),
            'item_type':        'controlled' if image_control_map.get(item['filename']) == 'yes' else 'uncontrolled' if image_control_map.get(item['filename']) == 'no' else float('nan'),
            'mem_response':     mem_response,
            'mem_rt':           mem_rt,
            'mem_trigger_onset': mem_trigger_onset,
            'mem_trigger_resp':  mem_trigger_resp
        })

        # Optional mid-test break every 100 items (timed: 30 s max)
        if item_num % 100 == 0 and item_num < len(mem_items):
            BREAK_DURATION = 30.0
            break_clock = core.Clock()
            event.clearEvents(eventType='keyboard')
            while break_clock.getTime() < BREAK_DURATION:
                remaining = int(BREAK_DURATION - break_clock.getTime()) + 1
                msg.text = (f"Take a short break if needed.\n\n"
                            f"The experiment will continue automatically in {remaining} s.\n\n"
                            f"Press SPACE to continue earlier.")
                msg.draw(); win.flip()
                keys = event.getKeys(['space', 'escape'])
                if 'escape' in keys:
                    _save(); core.quit()
                if 'space' in keys:
                    break
                core.wait(0.1)

    print(f"Memory test complete. {len(mem_items)} items judged.")


# ─────────────────────────────────────────────────────────────────────────────
#  INSTRUCTION SCREENS
# ─────────────────────────────────────────────────────────────────────────────

def show_initial_instructions():
    """Display the welcome and general task instructions."""
    instructions = [
        """Dear participant, welcome to the study!

This task involves moving images on the screen using your finger on the touchpad, and figuring out which image is under your control. The experiment consists of 6 blocks with 20 trials each.

In each trial, you will see two images on the screen.
Please use your right index finger on the touchpad to move the images. One of these images will be moving randomly, and the other one will be partially controlled by your finger movement. After a certain duration, you will be asked to report which image you were controlling, by pressing [A] or [S] on the keyboard. 

After each trial, you will be asked to indicate your feeling of control (on a scale of 1 to 7, where 1 is no control and 7 is full control). Please answer by pressing the corresponding number on the keyboard.

Please respond as accurately as possible throughout the whole experiment. If unsure, make your best guess.

Please feel free to ask any questions to the experimenter now.

Before the main experiment, you will practice the task with simple shapes in two practice blocks. During the practice blocks, you will receive feedback on whether your response was correct or incorrect. You won't be asked to indicate your feeling of control.

Please press SPACE to start the practice blocks."""
    ]

    for instruction in instructions:
        msg.text = instruction
        msg.draw(); win.flip()
        keys = wait_keys(['space', 'escape'])
        if 'escape' in keys:
            _save(); core.quit()



# def show_calibration_instructions():
#     """Display instructions for the calibration (practice) phase."""
#     msg.text = """PRACTICE PHASE

# In this phase, you will practice the task.
# After each trial, you will receive feedback: "Right" or "Wrong".

# This helps us calibrate the task difficulty to your individual level.

# Response keys:
#   A = Square
#   S = Circle

# Press SPACE to start the practice..."""
#     msg.draw(); win.flip()
#     wait_keys(['space', 'escape'])


def show_test_phase_instructions():
    """
    Display transition instructions shown ONCE after calibration,
    before the first miniblock of test trials.
    """
    msg.text = """Well done — the practice block is now complete!

You will now see pairs of images on screen.
Use the touchpad to move the images and decide which image was the one you controlled.

Indicate your decision by pressing [A] or [S].
After each trial, rate your feeling of control on a scale of 1 to 7
(1 = no control, 7 = full control) by pressing the corresponding number.

No feedback will be shown during the main experiment.

Please press SPACE to start."""
    msg.draw(); win.flip()
    wait_keys(['space', 'escape'])


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN EXPERIMENT
# ─────────────────────────────────────────────────────────────────────────────

# ── Step 1: Show initial instructions ────────────────────────────────────────
show_initial_instructions()

# ── Step 2: Two-stage calibration ──────────────────────────────────────────────
# Run two QUEST+ staircases: one targeting 55% correct (low control condition)
# and one targeting 85% correct (high control condition).
# Order is counterbalanced by participant parity (same logic as test miniblocks):
#   Odd  participant → Practice block 1 = 55%, Practice block 2 = 85%
#   Even participant → Practice block 1 = 85%, Practice block 2 = 55%
try:
    participant_num = int(expInfo["participant"])
except ValueError:
    participant_num = int(hashlib.sha256(expInfo["participant"].encode()).hexdigest(), 16)

starts_low_first = (participant_num % 2 == 1)   # odd → 55% block first

if starts_low_first:
    calib_sequence = [
        (0.55, 0,  "Practice Block 1 of 2"),
        (0.85, -1, "Practice Block 2 of 2"),
    ]
else:
    calib_sequence = [
        (0.85, -1, "Practice Block 1 of 2"),
        (0.55, 0,  "Practice Block 2 of 2"),
    ]

prop_55 = None
prop_85 = None

for calib_idx, (calib_target, calib_block_num, calib_label) in enumerate(calib_sequence):
    # Show opening screen for the first calibration block only
    # (The second block starts directly after the between-block break screen)
    if calib_idx == 0:
        msg.text = (f"{calib_label}\n\n"
                    "In this block, you will practice the task.\n"
                    "You will receive feedback after each response.\n\n"
                    "Press SPACE to start.")
        msg.draw(); win.flip(); wait_keys()

    quest_calib, prop_calib, converged_calib = run_calibration_quest(
        target_accuracy=calib_target,
        num_trials=CHECK_CALIBRATION_TRIALS,
        block_num=calib_block_num
    )

    # Store calibrated prop under the correct condition label
    if calib_target <= 0.60:
        prop_55 = prop_calib
        expInfo['quest_prop_low']  = prop_55
        expInfo['quest_low_converged'] = converged_calib
    else:
        prop_85 = prop_calib
        expInfo['quest_prop_high'] = prop_85
        expInfo['quest_high_converged'] = converged_calib

    # Between-block message (shown only after the FIRST calibration block)
    if calib_idx == 0:
        msg.text = ("Well done — practice block 1 is now complete!\n\n"
                    "You can take a short break now.\n\n"
                    "Please press SPACE to continue with Practice block 2.")
        msg.draw(); win.flip(); wait_keys()

# Assemble difficulty levels from the two directly calibrated props
levels = {'low': prop_55, 'high': prop_85}

print(f"\nControl conditions (directly calibrated):")
print(f"  low  (55% target): prop={levels['low']:.3f}")
print(f"  high (85% target): prop={levels['high']:.3f}")

# ── Step 3: Determine miniblock order by participant parity ───────────────────
# Odd participant number  → starts with low: [low, high, low, high, low, high]
# Even participant number → starts with high: [high, low, high, low, high, low]
# participant_num already computed above (same variable)
starts_with_low = (participant_num % 2 == 1)   # odd → low first
level_A = 'low' if starts_with_low else 'high'
level_B = 'high' if starts_with_low else 'low'

miniblock_sequence = [level_A, level_B, level_A, level_B, level_A, level_B]

expInfo['miniblock_order'] = str(miniblock_sequence)
expInfo['starts_with']     = level_A

print(f"\nParticipant {expInfo['participant']} (num={participant_num}, "
      f"{'odd' if starts_with_low else 'even'}): "
      f"Miniblock order = {miniblock_sequence}")

# ── Step 5: Show test phase instructions (once, after calibration) ────────────
show_test_phase_instructions()

# ── Step 6: Run 6 miniblocks ──────────────────────────────────────────────────
# Miniblocks alternate between low and high (order set by parity above).
# Each miniblock contains CHECK_TEST_TRIALS_PER_LEVEL trials (20 in full mode).
TOTAL_MINIBLOCKS = 6

for mb_idx, level_name in enumerate(miniblock_sequence):
    miniblock_num = mb_idx + 1
    prop_value    = levels[level_name]

    # Run all trials for this miniblock
    run_test_block_for_level(
        threshold_75=prop_value,  # calibrated prop for this condition (logged for reference)
        level_name=level_name,
        prop_value=prop_value,
        num_trials=CHECK_TEST_TRIALS_PER_LEVEL,
        block_num=miniblock_num
    )

    # Show a break screen between miniblocks (not after the last one)
    if miniblock_num < TOTAL_MINIBLOCKS:
        msg.text = (f"Block {miniblock_num} of {TOTAL_MINIBLOCKS} complete!\n\n"
                    f"You can take a short break.\n\n"
                    f"Press SPACE when you are ready to continue.")
        msg.draw(); win.flip(); wait_keys()


# ─────────────────────────────────────────────────────────────────────────────
#  MEMORY TEST PHASE
# ─────────────────────────────────────────────────────────────────────────────

# Show transition instructions before the memory test
msg.text = """

The main experiment is now complete. One last part! 

You will now see a series of objects on the screen, one at a time.
For each object, decide whether you saw it during the experiment (Yes) or not (No). If you have seen the object before (Yes), press [Y]. If you have not seen the object before (No), press [N].

Please try to respond as accurately as possible. If unsure, make your best guess.

Please press SPACE to start."""
msg.draw(); win.flip()
wait_keys(['space', 'escape'])

# Run the memory test (N_IMAGES seen + N_IMAGES foils, shuffled)
run_memory_test(sampled_test_images, foil_images)


# ─────────────────────────────────────────────────────────────────────────────
#  END OF EXPERIMENT
# ─────────────────────────────────────────────────────────────────────────────

final_used = len(used_trajectory_indices)
tot_time_secs = global_clock.getTime()
tot_mins = int(tot_time_secs // 60)
tot_secs = int(tot_time_secs % 60)

print(f"\n============================================================")
print(f"EXPERIMENT COMPLETE")
print(f"Total Duration: {tot_mins} minutes {tot_secs} seconds")
print(f"============================================================")
print(f"  Total trajectories used: {final_used}")
print(f"  Low  prop (55% target): {levels['low']:.3f}")
print(f"  High prop (85% target): {levels['high']:.3f}")
print(f"  Miniblock order: {miniblock_sequence}")

msg.text = f"""Thank you for participating!

Your data have been recorded. 

Press SPACE to exit."""
msg.draw(); win.flip(); wait_keys()

# Save all data and close
_save()
win.close()
core.quit()
