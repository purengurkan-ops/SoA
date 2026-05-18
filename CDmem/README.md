# CDmem — Control Detection & Memory Experiment

A PsychoPy experiment investigating how people detect control over moving shapes, and how memory for controlled stimuli persists after the experiment.

Adapted from [`MTI/MT Inference.py`](../MTI/MT%20Inference.py).

---

## Overview

On each trial, two shapes (a Square and a Circle) move across the screen. One shape (the **target**) has its direction partially guided by the participant's mouse. The other (the **distractor**) follows a pre-recorded trajectory autonomously.

After the motion phase, participants:
1. **Identify** which shape they controlled
2. **Rate** their confidence and sense of agency

In the **Test Phase**, the shapes are replaced by images from the [IMAGINE dataset](https://doi.org/10.3758/s13428-019-01284-4). After all test blocks, participants complete a **recognition memory test** for these images.

---

## Experiment Structure

| Phase | Description |
|---|---|
| **Calibration** | 3-up-1-down staircase → finds each participant's 75% accuracy threshold (`prop`) |
| **Test** (4 blocks) | Four difficulty levels derived from the calibrated `prop`, counterbalanced across blocks |
| **Memory Test** | Recognition memory for images seen during the test phase |

**Control manipulation**: `prop` ∈ [0, 1] blends the participant's mouse direction with the pre-recorded trajectory direction, while keeping speed constant. Lower `prop` = harder to detect control.

---

## Repository Structure

```
CDmem/
├── CDmem_1.py               # Main experiment script
├── image_stimuli_log.json   # Log of images sampled per session
├── Motion_Library/          # Pre-recorded motion trajectories
│   ├── core_pool.npy            # Velocity snippets (n_snippets × frames × 2)
│   ├── core_pool_feats.npy      # Feature vectors per snippet
│   ├── core_pool_labels.npy     # K=4 cluster labels
│   ├── scaler_params.json       # Feature normalization parameters
│   └── cluster_centroids.json   # Cluster centroid coordinates
├── screenshots/             # Auto-saved screenshots per phase
└── data/
    └── subjects/            # Per-participant CSV files (trial data + kinematics)
```

---

## Running the Experiment

```bash
python CDmem_1.py
```

**Special modes** (set via dialog on startup):
- `simulate` — runs with a virtual mouse (no real input needed)
- `check_mode` — minimal trials for quick script verification

**Environment variable shortcut:**
```bash
CDT_AUTO_TEST=true python CDmem_1.py   # Fully automated test run
```

---

## Output Files

| File | Contents |
|---|---|
| `data/subjects/CDmem_1_<ID>.csv` | Trial-by-trial responses (accuracy, RT, agency ratings) + both overall and block-wise trial numbers + full metadata |
| `data/subjects/CDmem_1_<ID>_kinematics.csv` | Frame-by-frame mouse and shape positions + both overall and block-wise trial numbers + full metadata |
| `data/subjects/CDmem_1_<ID>_recognition.csv` | Recognition memory test results + both overall and block-wise trial numbers + full metadata |
| `screenshots/` | Auto-saved PNG screenshots of each phase |

---

## Requirements

- Python 3.8+
- PsychoPy, NumPy, Pandas
- IMAGINE dataset (familiar images, PNG format) — path configured in `CDmem_1.py`

---

## License

For academic/research use. Contact the author for permissions.
