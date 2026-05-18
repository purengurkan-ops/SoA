#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CDmem_jigglefree.py
===================
Isolated demo of the CDmem control detection trial — with EWMA anti-jiggle filter.

Purpose
-------
This script is a stripped-down version of CDmem_1.py for testing the effect
of a low-pass (EWMA) filter on the mouse input. It uses the same Motion Library,
the same direction-mixing logic, and the same velocity smoothing as CDmem_1.py.

What is NEW vs CDmem_1.py:
  - An EWMA filter is applied to the RAW MOUSE INPUT before it enters
    mix_direction_only(). This destroys high-frequency jiggle movements
    before they can influence the shape's direction.

What is KEPT from CDmem_1.py:
  - Motion Library loading and preprocessing
  - mix_direction_only() — direction blending, speed preserved from trajectory
  - LOWPASS = 0.2 applied to velocity output (smooths shape movement)
  - confine() boundary clamping
  - SPEED_MULTIPLIER = 1.3
  - Fixation → Motion (3 sec) → Response (A/S) trial structure

What is REMOVED:
  - Calibration / QUEST+ staircase
  - Recognition / memory test phase
  - Image stimuli (shapes only)
  - EEG triggers
  - Participant dialog and data logging

Conditions
----------
  High Control : prop_used = 0.23  (easy — strong mouse influence)
  Low Control  : prop_used = 0.49  (hard — weaker mouse influence)
  5 trials per condition, randomly interleaved.

Two filters at work
-------------------
  1. ALPHA (EWMA on mouse INPUT — anti-jiggle, NEW):
       filtered_mouse = ALPHA * filtered_mouse + (1 - ALPHA) * raw_mouse
       This is an Exponentially Weighted Moving Average (EWMA), also called a
       Low-Pass Filter. It kills fast, high-frequency jiggle movements at the
       source, before they ever enter the direction-mixing step. A jigging hand
       averages out to near-zero displacement, so the shape simply ignores it.

  2. LOWPASS (on velocity OUTPUT — trajectory smoothing, from CDmem_1.py):
       vt = LOWPASS * vt + (1 - LOWPASS) * new_velocity
       This makes the shape's on-screen movement look smooth and natural.
       It does NOT prevent jiggling from influencing the direction — it just
       makes the resulting motion look less jerky.

Play around with ALPHA below. Try:
  0.00 → no mouse filtering (jiggle works fully)
  0.70 → light smoothing
  0.85 → strong anti-jiggle (recommended starting point)
  0.95 → very heavy smoothing, noticeable lag
"""

import os
import sys
import math
import random
import pathlib
import json

import numpy as np
from psychopy import visual, event, core
from scipy.signal import butter, sosfilt_zi, sosfilt

# =============================================================================
# ▶▶  TUNE THESE TWO PARAMETERS  ◀◀
# =============================================================================

# CUTOFF_HZ: Butterworth low-pass cutoff frequency for the anti-jiggle filter.
#   Frequencies ABOVE this are attenuated. Lower = more aggressive jiggle removal.
#   Deliberate hand movement ≈ 0–2 Hz. Hand tremor / jiggle ≈ 6–20 Hz.
#   Try: 6.0 (gentle), 3.5 (moderate ← current), 2.0 (very aggressive, some lag)
CUTOFF_HZ   = 3.5    # Hz
BUTTER_ORDER = 2     # Filter order (2 = -40 dB/decade roll-off, good balance)
SAMPLE_HZ   = 60.0  # Approximate monitor refresh rate (frames per second)

# Design the filter ONCE at startup (efficient — reuse the sos across trials)
_butter_sos = butter(BUTTER_ORDER, CUTOFF_HZ, btype='low', fs=SAMPLE_HZ, output='sos')
_butter_zi  = sosfilt_zi(_butter_sos)  # template initial-state (shape: n_sections × 2)

# LOWPASS: smoothing on shape velocity output (same as CDmem_1.py, line 1459)
# This makes the shape's motion look smooth. Does NOT stop jiggling.
LOWPASS = 0.2

# =============================================================================
#  OTHER CONSTANTS  (mirror CDmem_1.py)
# =============================================================================

SPEED_MULTIPLIER    = 1.3    # Scale trajectory velocities
MOTION_DURATION     = 3.0    # Seconds per motion phase
MOUSE_MOVE_THRESHOLD = 0.5   # px/frame below which mouse is considered stationary
OFFSET_X            = 300    # Horizontal distance from center to shape start (px)
MAX_MOUSE_SPEED     = 20.0   # Cap on raw mouse speed per frame (px)

# Conditions
PROP_HIGH_CONTROL = 0.49   # High control (easy — shape strongly follows mouse)
PROP_LOW_CONTROL  = 0.23   # Low control (hard — shape weakly follows mouse)
N_TRIALS_PER_COND = 5

# =============================================================================
#  MOTION LIBRARY
# =============================================================================

script_dir  = pathlib.Path(__file__).parent
LIB_NAME    = script_dir / "Motion_Library" / "core_pool.npy"
FEATS_NAME  = script_dir / "Motion_Library" / "core_pool_feats.npy"
LABELS_NAME = script_dir / "Motion_Library" / "core_pool_labels.npy"

motion_pool      = np.load(LIB_NAME)
snippet_features = np.load(FEATS_NAME)
snippet_labels   = np.load(LABELS_NAME)

SNIP_LEN    = motion_pool.shape[1]
TOTAL_SNIPS = motion_pool.shape[0]

print(f"Loaded {TOTAL_SNIPS} snippets × {SNIP_LEN} frames from {LIB_NAME}")

rng = np.random.default_rng(42)

# =============================================================================
#  TRAJECTORY QUALITY & PREPROCESSING  (identical to CDmem_1.py)
# =============================================================================

def analyze_trajectory_quality(trajectory):
    velocities = np.diff(trajectory, axis=0)
    speeds = np.linalg.norm(velocities, axis=1)
    mean_speed = np.mean(speeds)
    std_speed  = np.std(speeds)

    zero_movement_ratio = np.sum(speeds < 0.5) / len(speeds)
    high_jitter_ratio   = np.sum(speeds > mean_speed + 3 * std_speed) / len(speeds)

    if len(velocities) > 1:
        unit_velocities = velocities / (speeds.reshape(-1, 1) + 1e-9)
        angle_changes   = np.arccos(
            np.clip(np.sum(unit_velocities[:-1] * unit_velocities[1:], axis=1), -1, 1)
        )
        jerkiness = np.std(angle_changes)
    else:
        jerkiness = 0

    return {
        'mean_speed':          mean_speed,
        'zero_movement_ratio': zero_movement_ratio,
        'high_jitter_ratio':   high_jitter_ratio,
        'jerkiness':           jerkiness,
    }


def is_trajectory_valid(trajectory, min_speed=1.0, max_zero_ratio=0.3,
                         max_jitter_ratio=0.1, max_jerkiness=1.5):
    q = analyze_trajectory_quality(trajectory)
    if q['mean_speed']          < min_speed:         return False, "mean_speed_too_low"
    if q['zero_movement_ratio'] > max_zero_ratio:    return False, "too_much_zero_movement"
    if q['high_jitter_ratio']   > max_jitter_ratio:  return False, "too_much_jitter"
    if q['jerkiness']           > max_jerkiness:     return False, "too_jerky"
    return True, "valid"


def normalize_trajectory(trajectory, target_speed_range=(5.0, 15.0), smooth_factor=0.35):
    if len(trajectory) < 2:
        return trajectory
    velocities = np.diff(trajectory, axis=0)
    speeds     = np.linalg.norm(velocities, axis=1)
    current_mean = np.mean(speeds)
    if current_mean > 0:
        target_mean  = np.mean(target_speed_range)
        velocities   = velocities * (target_mean / current_mean)

    smoothed = velocities.copy()
    for i in range(1, len(velocities)):
        smoothed[i] = smooth_factor * smoothed[i - 1] + (1 - smooth_factor) * velocities[i]

    normalized = [trajectory[0]]
    for vel in smoothed:
        normalized.append(normalized[-1] + vel)
    return np.array(normalized)


def preprocess_motion_pool():
    global motion_pool, snippet_features, snippet_labels, SNIP_LEN
    print("Preprocessing motion pool...")
    initial_count = len(motion_pool)
    proc_snips, proc_feats, proc_labels = [], [], []

    for i, snippet in enumerate(motion_pool):
        trajectory = np.cumsum(snippet, axis=0)
        is_valid, reason = is_trajectory_valid(trajectory)
        if is_valid:
            norm_traj = normalize_trajectory(trajectory)
            proc_snips.append(np.diff(norm_traj, axis=0))
            proc_feats.append(snippet_features[i])
            proc_labels.append(snippet_labels[i])

    motion_pool      = np.array(proc_snips)
    snippet_features = np.array(proc_feats)
    snippet_labels   = np.array(proc_labels)
    SNIP_LEN = motion_pool.shape[1] if len(motion_pool) > 0 else 0

    print(f"Motion pool: kept {len(proc_snips)}/{initial_count} snippets")
    return list(range(len(proc_snips)))


valid_snippet_indices = preprocess_motion_pool()
used_indices = set()


def get_two_trajectories():
    """Pick 2 unique unused snippets; fall back to any valid ones if exhausted."""
    available = [i for i in valid_snippet_indices if i not in used_indices]
    if len(available) >= 2:
        selected = list(rng.choice(available, size=2, replace=False))
    else:
        selected = list(rng.choice(valid_snippet_indices, size=2, replace=False))
    for idx in selected:
        used_indices.add(idx)
    return selected


# =============================================================================
#  CONSISTENT SMOOTHING  (identical to CDmem_1.py)
# =============================================================================

def apply_consistent_smoothing(traj1, traj2, window_size=3):
    def smooth(traj):
        s = traj.copy()
        for i in range(len(traj)):
            start = max(0, i - window_size // 2)
            end   = min(len(traj), i + window_size // 2 + 1)
            s[i]  = np.mean(traj[start:end], axis=0)
        return s

    pos1 = np.cumsum(traj1, axis=0); pos2 = np.cumsum(traj2, axis=0)
    vel1 = np.diff(smooth(pos1), axis=0)
    vel2 = np.diff(smooth(pos2), axis=0)
    return vel1, vel2


# =============================================================================
#  DIRECTION MIXING  (identical to CDmem_1.py)
# =============================================================================

def mix_direction_only(mouse_dx, mouse_dy, traj_dx, traj_dy, prop):
    """
    Blend mouse and trajectory DIRECTIONS while preserving trajectory SPEED.
    prop=0 → shape follows trajectory only.
    prop=1 → shape follows mouse direction only (at trajectory speed).
    """
    traj_speed = math.hypot(traj_dx, traj_dy)
    if traj_speed < 0.01:
        return traj_dx, traj_dy

    mouse_mag = math.hypot(mouse_dx, mouse_dy)
    if mouse_mag < 0.01:
        return traj_dx, traj_dy

    m_dir_x, m_dir_y = mouse_dx / mouse_mag, mouse_dy / mouse_mag
    t_dir_x, t_dir_y = traj_dx / traj_speed, traj_dy / traj_speed

    mix_x = prop * m_dir_x + (1 - prop) * t_dir_x
    mix_y = prop * m_dir_y + (1 - prop) * t_dir_y

    mix_mag = math.hypot(mix_x, mix_y)
    if mix_mag > 0.01:
        return (mix_x / mix_mag) * traj_speed, (mix_y / mix_mag) * traj_speed
    return traj_dx, traj_dy


# =============================================================================
#  BOUNDARY CLAMPING  (identical to CDmem_1.py)
# =============================================================================

confine = lambda p, l=400: p if (r := math.hypot(*p)) <= l else (p[0] * l / r, p[1] * l / r)


# =============================================================================
#  WINDOW & STIMULI
# =============================================================================

win = visual.Window((1200, 800), fullscr=False, color=[0.5] * 3, units='pix', allowGUI=True)
win.setMouseVisible(False)

square = visual.Rect(win,   40, 40, fillColor='black', lineColor='black')
dot    = visual.Circle(win, 20,     fillColor='black', lineColor='black')
fix    = visual.TextStim(win, '+',  color='white', height=60)

# Condition label shown throughout the motion phase
cond_label_stim = visual.TextStim(
    win, text='', color='yellow', height=24, pos=(0, 340), bold=True
)

# On-screen instructions between trials
msg = visual.TextStim(win, '', color='white', height=26, wrapWidth=900, bold=True)

# Key labels shown during response
label_A = visual.TextStim(win, 'A', color='white', height=28, bold=True, pos=(-120, -80))
label_S = visual.TextStim(win, 'S', color='white', height=28, bold=True, pos=( 120, -80))


# =============================================================================
#  TRIAL SEQUENCE
# =============================================================================

trials = (
    [{'condition': 'High Control', 'prop': PROP_HIGH_CONTROL}] * N_TRIALS_PER_COND +
    [{'condition': 'Low Control',  'prop': PROP_LOW_CONTROL}]  * N_TRIALS_PER_COND
)
random.shuffle(trials)

# =============================================================================
#  MAIN TRIAL LOOP
# =============================================================================

for trial_num, trial in enumerate(trials):
    prop      = trial['prop']
    cond_name = trial['condition']

    # ── INTER-TRIAL INSTRUCTIONS ──────────────────────────────────────────────
    msg.text = (
        f"Trial {trial_num + 1} / {len(trials)}\n\n"
        f"Condition: {cond_name}  (prop = {prop})\n"
        f"Butterworth cutoff = {CUTOFF_HZ} Hz (order {BUTTER_ORDER})    LOWPASS (velocity) = {LOWPASS}\n\n"
        f"One shape is yours. Try to figure out which one!\n"
        f"Press  A  (left shape)  or  S  (right shape)\n\n"
        f"Press SPACE to start."
    )
    msg.draw()
    win.flip()
    event.waitKeys(keyList=['space', 'escape'])
    if event.getKeys(['escape']):
        win.close(); core.quit()

    # ── FIXATION ──────────────────────────────────────────────────────────────
    fix.draw(); win.flip()
    core.wait(1.0)

    # ── SHAPE STARTING POSITIONS ──────────────────────────────────────────────
    # Randomly assign square and dot to left/right
    left_is_square = random.choice([True, False])
    if left_is_square:
        square.pos = (-OFFSET_X, 0); dot.pos = (OFFSET_X, 0)
        left_label_name = 'square'; right_label_name = 'dot'
    else:
        dot.pos = (-OFFSET_X, 0); square.pos = (OFFSET_X, 0)
        left_label_name = 'dot'; right_label_name = 'square'

    # Randomly assign which shape is the target (controlled)
    target_name = random.choice([left_label_name, right_label_name])

    # ── TRAJECTORIES ─────────────────────────────────────────────────────────
    t_idx, d_idx       = get_two_trajectories()
    target_snip        = motion_pool[t_idx]
    distractor_snip    = motion_pool[d_idx]
    target_snip, distractor_snip = apply_consistent_smoothing(target_snip, distractor_snip)

    # ── WAIT FOR MOUSE MOVEMENT ───────────────────────────────────────────────
    mouse = event.Mouse(win=win, visible=False)
    mouse.setPos((0, 0))
    last_raw = list(mouse.getPos())
    square.draw(); dot.draw(); win.flip()

    while True:
        x, y = mouse.getPos()
        if math.hypot(x - last_raw[0], y - last_raw[1]) > 0:
            break
        if event.getKeys(['escape']):
            win.close(); core.quit()
        square.draw(); dot.draw(); win.flip()

    # ── MOTION PHASE ──────────────────────────────────────────────────────────
    clk   = core.Clock()
    frame = 0

    # Velocity accumulators for LOWPASS smoothing (from CDmem_1.py)
    vt = np.zeros(2, np.float32)
    vd = np.zeros(2, np.float32)

    # Butterworth filter state — reset each trial to zero (no startup transient
    # because displacement starts at 0 when the mouse is stationary).
    # Separate state for X and Y axes.
    zi_x = _butter_zi.copy() * 0.0
    zi_y = _butter_zi.copy() * 0.0

    last_raw = list(mouse.getPos())  # track RAW position

    event.clearEvents(eventType='keyboard')

    while clk.getTime() < MOTION_DURATION:

        # 1. RAW MOUSE POSITION & RAW DISPLACEMENT
        raw_pos = np.array(mouse.getPos(), dtype=float)
        raw_dx  = raw_pos - np.array(last_raw)   # per-frame displacement
        last_raw = list(raw_pos)                  # always track RAW position

        # 2. BUTTERWORTH LOW-PASS FILTER ON MOUSE DISPLACEMENT (anti-jiggle)
        #    Applied to the velocity (displacement per frame), not the position.
        #    Jiggle = rapid ±displacement → high frequency → attenuated by filter.
        #    Deliberate movement = slow, sustained displacement → low frequency → passes.
        #    sosfilt() processes one sample at a time and updates the filter state (zi)
        #    so it works correctly frame-by-frame in real time.
        filt_x, zi_x = sosfilt(_butter_sos, [raw_dx[0]], zi=zi_x)
        filt_y, zi_y = sosfilt(_butter_sos, [raw_dx[1]], zi=zi_y)
        dx, dy = float(filt_x[0]), float(filt_y[0])

        # 3. CAP MOUSE SPEED
        mouse_speed = math.hypot(dx, dy)
        if mouse_speed > MAX_MOUSE_SPEED:
            scale = MAX_MOUSE_SPEED / mouse_speed
            dx *= scale; dy *= scale
            mouse_speed = MAX_MOUSE_SPEED

        mouse_is_moving = mouse_speed > MOUSE_MOVE_THRESHOLD

        # 4. GET TRAJECTORY VELOCITIES (scaled)
        t_dx, t_dy = target_snip[frame % len(target_snip)]
        d_dx, d_dy = distractor_snip[frame % len(distractor_snip)]
        t_dx *= SPEED_MULTIPLIER; t_dy *= SPEED_MULTIPLIER
        d_dx *= SPEED_MULTIPLIER; d_dy *= SPEED_MULTIPLIER
        frame += 1

        # 5. MIX DIRECTION FOR TARGET
        if mouse_is_moving:
            tdx, tdy = mix_direction_only(dx, dy, t_dx, t_dy, prop)
        else:
            tdx, tdy = t_dx, t_dy

        ddx, ddy = d_dx, d_dy

        # 6. LOWPASS ON VELOCITY OUTPUT (from CDmem_1.py — smooths shape motion)
        #    This makes the shape's trajectory look smooth. It does NOT prevent
        #    jiggling from affecting direction — that is handled by ALPHA above.
        vt = LOWPASS * vt + (1 - LOWPASS) * np.array([tdx, tdy])
        vd = LOWPASS * vd + (1 - LOWPASS) * np.array([ddx, ddy])

        # 7. UPDATE POSITIONS
        if target_name == left_label_name:
            # Target is on the left
            if left_is_square:
                square.pos = confine(tuple(np.array(square.pos) + vt))
                dot.pos    = confine(tuple(np.array(dot.pos)    + vd))
            else:
                dot.pos    = confine(tuple(np.array(dot.pos)    + vt))
                square.pos = confine(tuple(np.array(square.pos) + vd))
        else:
            # Target is on the right
            if left_is_square:
                dot.pos    = confine(tuple(np.array(dot.pos)    + vt))
                square.pos = confine(tuple(np.array(square.pos) + vd))
            else:
                square.pos = confine(tuple(np.array(square.pos) + vt))
                dot.pos    = confine(tuple(np.array(dot.pos)    + vd))

        # 8. DRAW (condition label visible during motion phase)
        cond_label_stim.text = f"[ {cond_name} ]"
        square.draw(); dot.draw(); cond_label_stim.draw(); win.flip()

        if event.getKeys(['escape']):
            win.close(); core.quit()

    # ── RESPONSE PHASE ────────────────────────────────────────────────────────
    # Move shapes closer to center for response
    if left_is_square:
        square.pos = (-120, 0); dot.pos = (120, 0)
    else:
        dot.pos = (-120, 0); square.pos = (120, 0)

    msg.text = "Which shape did you control?   A = left    S = right"
    msg.pos  = (0, 150)

    event.clearEvents(eventType='keyboard')
    response = None
    resp_clock = core.Clock()

    while response is None:
        square.draw(); dot.draw()
        label_A.draw(); label_S.draw()
        msg.draw()
        win.flip()

        keys = event.getKeys(['a', 's', 'escape'])
        if 'escape' in keys:
            win.close(); core.quit()
        if 'a' in keys:
            response = left_label_name
        elif 's' in keys:
            response = right_label_name

    msg.pos = (0, 0)

    # ── FEEDBACK ──────────────────────────────────────────────────────────────
    correct = (response == target_name)
    feedback_text = f"{'✓ Correct!' if correct else '✗ Wrong'}   (Target was the {target_name})"
    feedback = visual.TextStim(win, feedback_text, color='white', height=30, bold=True)
    feedback.draw(); win.flip()
    core.wait(1.0)

# ── END SCREEN ────────────────────────────────────────────────────────────────
msg.text = "All done! Thank you.\n\nPress any key to exit."
msg.draw(); win.flip()
event.waitKeys()

win.close()
core.quit()
