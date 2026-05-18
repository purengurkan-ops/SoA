# Trial & Block Structure — CDmem (CDmem_1.py)

## Overview

The experiment is divided into 3 phases: **Calibration**, **Test**, and **Memory Test**. Calibration and Test trials all share the same single-trial structure (described below). The Memory Test phase has a different structure.


## Trial Count Summary

| Phase | Condition | Trials | Notes |
|-------|----------|--------|-------|
| **Calibration** | - | 60–80 | Adaptive stop when alpha SD < 0.20; hard cap at 80 |
| **Test — low** | 55% accuracy | 60 | 3 miniblocks × 20 trials. Show cases 120 images. |
| **Test — high** | 75% accuracy | 60 | 3 miniblocks × 20 trials. Show cases 120 images. |
| **Test (total)** | - | **120** | **6 miniblocks × 20 trials. 240 total images shown.** |
| **Memory test — seen** | Old | 240 | 2 images per trial × 120 test trials |
| **Memory test — foils** | New | 240 | Paired counterparts never shown |
| **Memory test (total)** | - | **480** | **Shuffled together** |
| **Grand total** | - | **~680** | **Calibration mid-range (70) + 120 test + 480 memory** |

---


## Single Trial Structure (Calibration & Test)

Each trial proceeds through the following steps in order.

### 1. Fixation cross — **Timed (0.5–0.8 s, random uniform)**
A white fixation cross (`+`) appears at the centre of a grey screen.
- No input required from the participant.
- Duration is randomly sampled on every trial.

### 2. Motion phase — **Timed (3.0 s, no input required)**
Two stimuli appear on screen: one on the left (`−300 px`) and one on the right (`+300 px`) of centre, vertically centred.

- **Calibration phase**: stimuli are a black **Square** (40 × 40 px) and a black **Circle** (radius 20 px).
- **Test phase**: stimuli are two **object images** (200 × 200 px) from the CARA stimulus set.

The other stimulus (the **distractor**) follows its pre-recorded trajectory fully independently of the mouse.

One stimulus is the **target** (randomly chosen each trial): its direction is a weighted blend of the participant's **mouse direction** and a pre-recorded **trajectory direction**. The blend weight (`prop`) is set by QUEST+ in calibration, or by the fixed control condition in the test phase.

The other stimulus (the **distractor**) follows its pre-recorded trajectory fully independently of the mouse.

> **Direction-only mixing**: speed is always equal to the trajectory speed. Only the *direction* of the target is influenced by the mouse. This keeps speed identical for both stimuli, preventing speed from being used as a cue.

- Mouse movement is tracked every frame; if the mouse is stationary (speed < 0.5 px/frame), the target also follows its trajectory autonomously.
- Stimuli are confined within a circle of radius 400 px around the centre to prevent them flying off-screen.
- **No keypress is required during this phase** — it proceeds automatically for exactly 3 seconds.
- EEG triggers are sent at stimulus onset and motion start (test phase only).

### 3. Response — **Self-paced (5 s window)**
After the motion phase ends, a response screen appears.

- **Calibration**: text prompt — *"Which shape did you control? A = Square   S = Circle"*
- **Test**: images are reset to their starting positions; the question *"Which image did you control?"* appears above them. Key labels **A** and **S** appear below the left and right images respectively.

The participant presses **A** or **S** to indicate which stimulus they controlled. A 5-second response window is enforced; if no response is given, a "Please answer faster!" message is shown and the trial is logged as a timeout.

> **Note**: the response is time-limited to 5 s (unlike the photo task, which is fully self-paced). RT and accuracy are logged.

### 4. Feedback — **Timed (0.8 s) — Calibration only**
Immediately after the response, *"Right"* or *"Wrong"* is displayed for 0.8 s.
- Only shown during the calibration phase.
- No feedback is given during the test phase.

### 5. Agency rating — **Self-paced (no time limit) — Test only**
After the response (test phase only), the question appears:
> *"How much control did you feel over the shape's movement?"*

A 7-point labelled scale is shown:

| Key | Label |
|-----|-------|
| 1 | Very weak |
| 2 | Weak |
| 3 | Somewhat weak |
| 4 | Moderate |
| 5 | Somewhat strong |
| 6 | Strong |
| 7 | Very strong |

Participant presses a **number key 1–7**. After keypress, the next trial's fixation cross appears immediately.

---

## Phase 1 — Calibration (Practice)

**Purpose:** Estimate each participant's psychometric function via a QUEST+ Bayesian adaptive staircase, and find the `prop` value at which they can identify the controlled stimulus ~75% of the time and 55% of the time. We now run two separate staircases for each participant, with different starting points for the psychometric function for the two target accuracies (i.e., 55% and 75%).

**Stimuli:** Plain Square and Circle (no images).

**Feedback:** Given after every trial ("Right" / "Wrong").

**No agency rating** is collected.

### Trial count

| Mode | Minimum | Hard cap |
|------|---------|----------|
| Full | 60 | 80 |
| Check mode | 6 | 26 |

QUEST+ stops early once the posterior SD of the threshold (`alpha_sd`) drops below **0.20**, provided the minimum number of trials has been reached.

### Mid-block break
After half the minimum trials, a self-paced break screen is shown. Participant presses **SPACE** to continue.

### How QUEST+ Works in Calibration

The calibration phase uses the QUEST+ Bayesian adaptive staircase to efficiently estimate the participant's psychometric function.

1.  **The Psychometric Function:**
    It assumes the probability of correctly identifying the controlled shape follows a logistic psychometric function:
    `p(correct | s) = γ + (1 − γ − λ) * (1 / (1 + exp(-β * (s - α))))`
    *   `s` = stimulus intensity (proportion of mouse control, in logit scale).
    *   `α` (alpha) = the threshold (stimulus level yielding ~75% accuracy when γ=0.5 and λ=0).
    *   `β` (beta) = the slope of the psychometric function.
    *   `γ` (gamma) = the guess rate (fixed at 0.5 for a 2-alternative forced choice).
    *   `λ` (lambda) = the lapse rate (probability of making a mistake regardless of stimulus intensity).

2.  **The Prior Grid:**
    QUEST+ maintains a discrete probability distribution (the "posterior") over a 3D grid of possible parameter values:
    *   `α` (alpha) grid: 61 values linearly spaced between `logit(0.05)` and `logit(0.90)`. Prior is a Gaussian centered around `logit(0.40)`.
    *   `β` (beta) grid: 25 values logarithmically spaced from `1.0` to `12.0`. Prior is log-normal.
    *   `λ` (lambda) grid: 5 values `[0.00, 0.01, 0.02, 0.04, 0.06]`. Prior is uniform.

3.  **Stimulus Selection (Entropy Minimization):**
    On each trial, QUEST+ selects the next stimulus intensity (`s`) that is expected to provide the most information.
    *   It calculates the Shannon entropy of the current 3D posterior.
    *   For a subset of possible stimuli (every 3rd value in the `s` grid to speed up computation), it predicts the probability of a "correct" or "incorrect" response.
    *   It simulates how the posterior would update given each possible response.
    *   It selects the `s` value that minimizes the *expected* entropy of the updated posterior (i.e., maximizes expected information gain).

4.  **Posterior Update:**
    After the participant responds, QUEST+ updates the probability of each parameter combination in the 3D grid using Bayes' rule.
    *   `P(parameters | response) ∝ P(response | parameters, s) * P(parameters)`
    *   This shifts the posterior distribution towards the most likely values of `α`, `β`, and `λ` given the participant's actual performance.

5.  **Adaptive Stopping Criterion:**
    The staircase stops dynamically when it is confident enough in its estimate of the threshold.
    *   It continuously calculates the standard deviation (SD) of the marginal posterior distribution for `α` (`quest_alpha_sd`).
    *   Calibration stops as soon as `alpha_sd < 0.20`, provided the minimum number of trials (60 in full mode) has been reached.
    *   If convergence isn't reached, it hits a hard cap of 80 trials.

6.  **Control Condition Derivation:**
    After calibration ends, the final 3D posterior is used to calculate the specific stimulus intensities (`prop` values) required for the test phase:
    *   `low` (Hard, ~55% accuracy target).
    *   `high` (Medium-hard, ~85% accuracy target).
    These values are derived by finding the `prop` that yields the target probability across the entire weighted parameter space.

### After calibration
The QUEST+ posterior is used to derive two control conditions for the test phase. A completion screen is shown; participant presses **SPACE** to continue to the test phase.

---

## Phase 2 — Test

**Purpose:** Measure perceived control and sense of agency across two control conditions, using paired object images as stimuli.

**Stimuli:** Two object images per trial (one target, one distractor), drawn from the CARA stimulus set (200 × 200 px).

**No feedback** is given.

**Agency rating** collected after every trial.

### Control conditions

Derived from the QUEST+ calibration posterior via `threshold_for_target()`:

| Condition | Target accuracy | Description |
|-------|----------------|-------------|
| `low` | ~55 % correct | Hardest |
| `high` | ~85 % correct | Medium-hard |

### Image stimuli

- 120 unique object-concept pairs are sampled from the `chosen_stimuli` folder at startup (fixed seed = 42 for reproducibility).
- From each pair, one image is assigned to the **test group** (shown during test trials) and the other to the **recognition group** (used as foils in the memory test).
- Images within the test group are randomly paired (two different object concepts per trial) to create 60 simultaneous-display pairs.
- Each image pair is used exactly once — no repeats across the 6 miniblocks.

### Block structure

- Total: **6 miniblocks × 20 trials = 120 test trials** (full mode), or **6 × 5 = 30** in check mode.
- Miniblocks alternate between `low` and `high`.
- The starting level is **counterbalanced by participant number parity**:
  - **Odd** participant number → starts with `low`: `[low, high, low, high, low, high]`
  - **Even** participant number → starts with `high`: `[high, low, high, low, high, low]`

### Breaks between miniblocks
After each miniblock (except the last), a self-paced break screen is shown. Participant presses **SPACE** to continue.

### EEG triggers (test phase only)

| Event | Trigger value |
|-------|--------------|
| Stimulus onset | 10 + level index (11 or 13) |
| Motion start | 20 + level index (21 or 23) |
| Response screen onset | 30 + level index (31 or 33) |
| Correct response | 41 |
| Incorrect response | 42 |

---

## Phase 3 — Memory Test

**Purpose:** Assess recognition memory for the object images seen during the test phase.

Two sets of images are presented in a single shuffled sequence:
- **Seen images** — the exact images shown during the test phase (120 items in full mode, 30 in check mode).
- **Foil (unseen) images** — the paired counterparts that were *not* shown during the test phase (120 items in full mode, 30 in check mode).

Total: **240 items** (or **60 items** in check mode), presented one at a time in a participant-specific random order.

### Instructions (self-paced screen)
Participant reads instructions and presses **SPACE** to begin.

### Per-item trial structure

#### 1. Fixation cross — **Timed (0.5–0.8 s, random uniform)**
Same as in the navigation phases.

#### 2. Recognition judgement — **Self-paced (no time limit)**
A single object image (300 × 300 px) is shown slightly above screen centre. Below it:
> *"Have you seen this object during the experiment before?"*

Key labels: **A = Old**, **S = New**

- Participant presses **A** (seen before) or **S** (not seen before).
- After keypress, the screen clears and the next trial's fixation cross appears immediately.

### EEG triggers (memory test phase)

| Event | Trigger value |
|-------|--------------|
| Seen image onset | 51 |
| Foil image onset | 52 |
| Correct response (Hit or Correct Rejection) | 61 |
| Incorrect response (Miss or False Alarm) | 62 |

### Optional mid-test break
Every **100 items**, a self-paced break screen is shown. Participant presses **SPACE** to continue.

---

## End of Experiment

A completion screen is shown:
> *"Thank you for participating! Your data have been recorded."*

Participant presses **SPACE** to exit. All data (main CSV + kinematics CSV) is saved automatically.

---

## Output Files

| File | Contents |
|------|----------|
| `data/subjects/CDmem_1_<ID>.csv` | Trial-by-trial responses (Calibration & Test phases): accuracy, RT, agency ratings, QUEST+ parameters, image filenames, EEG triggers, true_controlled, response_controlled, overall_trial_num (session-wide), trial_in_block (resets per block), and full participant metadata |
| `data/subjects/CDmem_1_<ID>_kinematics.csv` | Frame-by-frame mouse position, shape positions, per-frame evidence, overall_trial_num, trial_in_block, and full participant metadata |
| `data/subjects/CDmem_1_<ID>_recognition.csv` | Recognition memory test results: filename, ground truth, controlled (yes/no), participant response, RT, EEG triggers, overall_trial_num, trial_in_block, and full participant metadata |
| `image_stimuli_log.json` | Record of which images were sampled and assigned to test vs. recognition groups |

---

## Running Modes

| Mode | How to activate | Effect |
|------|----------------|--------|
| Normal | Run `CDmem_1.py` and fill in the dialog | Full trial counts |
| `check_mode` | Tick checkbox in dialog | Minimal trials (6 calibration, 5/miniblock) |
| `simulate` | Tick checkbox in dialog | Virtual mouse and auto-keypresses — no human needed |
| `AUTO_TEST` | `CDT_AUTO_TEST=true python CDmem_1.py` | Skips dialog, runs full simulation |
