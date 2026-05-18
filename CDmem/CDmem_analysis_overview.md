# CDmem Analysis Overview

**Control Detection + Memory Experiment**
Analysis Pipeline: Exclusion Criteria, Statistical Analyses & Formulas

---

## Experimental Design

Participants completed a **cursor control detection task** (CDmem). On each test-phase trial they saw two images side-by-side and controlled a cursor steered by a mix of their own movements and computer noise. The proportion of self-motion (`prop_used`) was titrated via a QUEST staircase to achieve **55% accuracy (Low control)** or **85% accuracy (High control)**. After the task, an unexpected **recognition memory test** probed memory for the images seen during encoding (targets) and for new images (foils).

**Two images per trial:** one *controlled* item (the cursor tracked it) and one *uncontrolled* item, enabling a 2×2 design: Trial Level (High / Low) × Item Type (Controlled / Uncontrolled).

---

## 1. Exclusion Criteria

### 1A. Timeout Rate — Participant Level

| Attribute | Detail |
|---|---|
| Level | Participant (removed entirely from all analyses) |
| Scope | Test-phase trials only (calibration phase excluded) |
| Threshold | ≥ 50% timeout trials in *either* the Low or High control condition |
| Reference | Haridi et al., 2025 |
| Variable | `is_timeout` — Boolean flag per trial |

A participant is flagged if their per-condition timeout rate (timeouts / total trials) reaches or exceeds 0.50 in *at least one* condition. All data from that participant are removed before any further analyses.

---

### 1B. Accuracy Outliers — Participant Level

| Attribute | Detail |
|---|---|
| Level | Participant (removed entirely from all analyses) |
| Scope | Test-phase detection accuracy per condition |
| Threshold | \|z\| > 2.5 SD from the group mean, evaluated per condition |
| Expected accuracies | Low ≈ 55%, High ≈ 85% (QUEST staircase targets) |
| Reference | Standard outlier removal (z-score criterion) |
| Variable | `detection_accuracy` — 0/1 per trial, averaged per participant × condition |

Per-participant mean accuracy is computed for High and Low conditions separately. A group mean and SD are calculated across participants for each condition. Any participant whose mean deviates by more than 2.5 SD (in either condition) is excluded.

---

### 1C. Recognition RT Outliers — Trial Level

| Attribute | Detail |
|---|---|
| Level | Trial (individual recognition trials removed; participant retained) |
| Scope | Recognition memory phase only |
| Threshold | `mem_rt` > participant's mean RT + 3 × SD  (per-participant rolling) |
| Reference | Ren et al., 2026 |
| Variable | `mem_rt` — response time on recognition trial (seconds) |
| Also excludes | Trials with missing (NaN) response time |

For each participant, mean and SD of `mem_rt` are computed across all their recognition trials. Any trial whose RT exceeds mean + 3 SD is discarded at the trial level. This is a participant-specific, rolling threshold — it adapts to individual response speed distributions.

---

## 2. Key Variables

| Variable | Source | Description |
|---|---|---|
| `participant` | Main / Recog | Unique participant ID |
| `phase` | Main | `'calibration'` or `'test'` |
| `control_condition` | Main | `'high'` (85% target) or `'low'` (55% target) |
| `detection_accuracy` | Main | 0 = incorrect, 1 = correct on each trial |
| `prop_used` | Main | Proportion of cursor motion from participant (0–1) |
| `agency_rating` | Main | Subjective sense of agency rating (1–7 Likert) |
| `is_timeout` | Main | Boolean — trial timed out before response |
| `rt_choice` | Main | Choice reaction time in the control task (s) |
| `mem_response` | Recognition | `'yes'` (old) or `'no'` (new) |
| `mem_rt` | Recognition | Response time on recognition trial (s) |
| `mem_ground_truth` | Recognition | `'seen'` (target) or `'unseen'` (foil) |
| `trial_level` | Recognition | Encoding trial's control level: `'high'` or `'low'` |
| `item_type` | Recognition | `'controlled'` or `'uncontrolled'` item per trial |
| `img_A_name` / `img_B_name` | Main | Filenames of the two images shown per trial |
| `true_controlled` | Main | Which image (`'img_A'` or `'img_B'`) the cursor tracked |

---

## 3. Derived Measures

| Measure | Formula / Method | Used in |
|---|---|---|
| Hit Rate (HR) | Mean of `said_old` (= 1) across target trials per participant × condition | Analyses 1, 2, 7a, 7b |
| False Alarm Rate (FAR) | Mean of `said_old` across foil trials per participant | All d′ calculations |
| d′ (d-prime) | Φ⁻¹(HR_clipped) − Φ⁻¹(FAR_clipped); Clipping: [0.01, 0.99] to avoid ±∞ | Analyses 1, 7a, Supp 1 |
| `control_c` | Contrast code: High = +0.5, Low = −0.5 | Analyses 3, 4, CD accuracy GLMM |
| `agency_rating_z` | `agency_rating` z-scored within each participant: `(x − participant_mean) / participant_SD`. Removes individual differences in scale use; slope captures trial-to-trial within-person variation only. | Analysis 7 |
| `trial_level_c` | Contrast code: High = +0.5, Low = −0.5 | Supp 3, Supp 4 |
| `item_type_c` | Contrast code: Controlled = +0.5, Uncontrolled = −0.5 | Supp 3, Supp 4 |
| `item_is_old_c` | Contrast code: Old (target) = +0.5, Foil = −0.5 | Analysis 4, Supp 4 |
| Cohen's d | (mean_high − mean_low) / SD_diff, applied to paired differences | Analyses 1, 2, 7a, 7b, CD check |

---

## 4. Statistical Analyses

### Primary Analyses — Control Level (High vs. Low)

---

#### Analysis 1 — d′ Paired t-test

| Attribute | Detail |
|---|---|
| Analysis type | Paired-samples t-test |
| Package | `scipy.stats.ttest_rel` |
| Independent variable | Control condition (High vs. Low) — within-subject |
| Dependent variable | d′ per participant per condition |
| Notes | Effect size: Cohen's d on paired differences. Participant-level summary statistic. |

---

#### Analysis 2 — Hit Rate Paired t-test

| Attribute | Detail |
|---|---|
| Analysis type | Paired-samples t-test |
| Package | `scipy.stats.ttest_rel` |
| Independent variable | Control condition (High vs. Low) |
| Dependent variable | Hit rate per participant per condition |
| Notes | Same structure as Analysis 1. Hit rate is bounded [0,1]; use d′ (Analysis 1) as primary. |

---

#### Analysis 3 — Recognition GLMM: High vs. Low (target trials only)

| Attribute | Detail |
|---|---|
| Analysis type | Generalized Linear Mixed Model (Binomial, logit link) |
| Package | `pymer4` (lme4/R via rpy2) |
| Independent variable | `control_c`: contrast-coded control condition (High = +0.5, Low = −0.5) |
| Dependent variable | `said_old_int`: binary recognition response (0 = no, 1 = yes) |

**Model formulas:**
```
Maximal:  said_old_int ~ control_c + (1 + control_c | participant)
Fallback: said_old_int ~ control_c + (1 | participant)
```

> Maximal random effects attempted first. Falls back to random-intercept-only if singular fit detected.

---

#### Analysis 4 — Recognition GLMM: Item Type × Control Interaction (all trials)

| Attribute | Detail |
|---|---|
| Analysis type | Generalized Linear Mixed Model (Binomial, logit link) |
| Package | `pymer4` (lme4/R via rpy2) |
| Independent variables | `item_type_c`: Target = +0.5, Foil = −0.5; `control_c`: High = +0.5, Low = −0.5; `item_type_c × control_c` interaction |
| Dependent variable | `said_old_int`: binary recognition response (0 = no, 1 = yes) |

**Model formulas:**
```
Maximal:  said_old_int ~ item_type_c * control_c + (1 + item_type_c * control_c | participant)
Fallback: said_old_int ~ item_type_c * control_c + (1 | participant)
```

> Includes both targets and foils. Foils receive a balanced dummy `control_c` assignment (first half +0.5, second half −0.5 per participant) since they have no genuine condition. FA rate is equal across dummy conditions by construction, so the interaction is driven by target trials. **This must be noted in reporting.**

---

### Supplementary Analyses — 2×2 Factorial (Trial Level × Item Type)

---

#### Supplementary Analysis 1 — 2×2 RM ANOVA on d′

| Attribute | Detail |
|---|---|
| Analysis type | Repeated-Measures ANOVA |
| Package | `statsmodels.stats.anova.AnovaRM` |
| Independent variables | Trial Level (High / Low) × Item Type (Controlled / Uncontrolled) — both within-subject |
| Dependent variable | d′ per participant per 2×2 cell |
| Notes | Linear model approximation on a bounded measure. Use Supp 3 GLMM as primary trial-level test. |

---

#### Supplementary Analysis 2 — 2×2 RM ANOVA on Hit Rate

| Attribute | Detail |
|---|---|
| Analysis type | Repeated-Measures ANOVA |
| Package | `statsmodels.stats.anova.AnovaRM` |
| Independent variables | Trial Level (High / Low) × Item Type (Controlled / Uncontrolled) |
| Dependent variable | Hit rate per participant per 2×2 cell |
| Notes | Same caveats as Supp 1. Retained as a summary-statistic complement. |

---

#### Supplementary Analysis 3 — 2×2 GLMM on target trials (Item Type × Trial Level)

| Attribute | Detail |
|---|---|
| Analysis type | Generalized Linear Mixed Model (Binomial, logit link) |
| Package | `pymer4` (lme4/R via rpy2) |
| Independent variables | `trial_level_c`: High = +0.5, Low = −0.5; `item_type_c`: Controlled = +0.5, Uncontrolled = −0.5; interaction |
| Dependent variable | `said_old_int`: binary recognition response |

**Model formulas:**
```
Maximal:  said_old_int ~ trial_level_c * item_type_c + (1 + trial_level_c * item_type_c | participant)
Fallback: said_old_int ~ trial_level_c * item_type_c + (1 | participant)
```

> Target trials only. Tests whether the encoding-level effect (high vs. low control) differs between controlled and uncontrolled items.

---

#### Supplementary Analysis 4 — 3-Way GLMM: Is_Old × Trial Level × Item Type (all trials)

| Attribute | Detail |
|---|---|
| Analysis type | Generalized Linear Mixed Model (Binomial, logit link) |
| Package | `pymer4` (lme4/R via rpy2) |
| Independent variables | `item_is_old_c`: Old = +0.5, Foil = −0.5; `trial_level_c`: High = +0.5, Low = −0.5; `item_type_c`: Controlled = +0.5, Uncontrolled = −0.5; all interactions |
| Dependent variable | `said_old_int`: binary recognition response |

**Model formulas:**
```
Maximal:  said_old_int ~ item_is_old_c * trial_level_c * item_type_c + (1 + item_is_old_c | participant)
Fallback: said_old_int ~ item_is_old_c * trial_level_c * item_type_c + (1 | participant)
```

> Includes foils with balanced dummy 2×2 assignment (rotating through all 4 cells per participant). FA rate is equal across dummy cells by construction. Theoretically relevant estimates are interactions involving `item_is_old_c`.

---

#### Supplementary Analysis 5 — False Alarm Rate Manipulation Check

| Attribute | Detail |
|---|---|
| Analysis type | One-sample t-test |
| Package | `scipy.stats.ttest_1samp` |
| Independent variable | None (tests FA rate against 0) |
| Dependent variable | False alarm rate per participant (proportion of foils called 'old') |
| Notes | Tests whether participants were systematically biased toward 'old' responses on new items. A significant FA rate would inflate hit rates and distort d′ estimates. |

---

### Analysis 7 — Sense of Agency → Recognition Memory (Continuous)

Agency_rating (1–7) is used as a **continuous predictor**, z-scored *within each participant*. Within-participant z-scoring removes individual differences in rating scale use (some participants habitually rate higher or lower than others), so the GLMM slope captures trial-to-trial variation within a person only.

---

#### Analysis 7 — Recognition GLMM: Continuous agency_rating_z (target trials only)

| Attribute | Detail |
|---|---|
| Analysis type | Generalized Linear Mixed Model (Binomial, logit link) |
| Package | `pymer4` (lme4/R via rpy2) |
| Independent variable | `agency_rating_z`: `agency_rating` z-scored within each participant (continuous) |
| Dependent variable | `said_old_int`: binary recognition response (0 = no, 1 = yes) |

**Model formulas:**
```
Maximal:  said_old_int ~ agency_rating_z + (1 + agency_rating_z | participant)
Fallback: said_old_int ~ agency_rating_z + (1 | participant)
```

> The positive slope estimate indicates that on trials where a participant felt *more* agency than usual, they were more likely to subsequently recognise the image. This is a within-person effect, not a between-person effect.

---

### Manipulation Check & Additional Analyses

---

#### CD — Control Detection Accuracy (Manipulation Check)

| Attribute | Detail |
|---|---|
| Analysis type | Paired t-test + GLMM (Binomial) |
| Package | `scipy.stats.ttest_rel` + `pymer4` |
| Independent variable | `control_c`: High = +0.5, Low = −0.5 |
| Dependent variable | `detection_accuracy`: 0/1 per trial (test phase only) |

**Model formulas:**
```
t-test:        participant-level mean accuracy, High vs. Low
GLMM Maximal:  detection_accuracy ~ control_c + (1 + control_c | participant)
GLMM Fallback: detection_accuracy ~ control_c + (1 | participant)
```

> Prerequisite check — if accuracy does not differ between conditions, memory effects cannot be attributed to the control-level manipulation.

---

#### Analysis 5a — Agency ~ Accuracy + Control Level (per condition)

| Attribute | Detail |
|---|---|
| Analysis type | Ordinary Least Squares (OLS) Regression |
| Package | `statsmodels.formula.api.ols` |
| Independent variables | `detection_accuracy` (0/1); `prop_used` (proportion of self-motion, 0–1); `detection_accuracy × prop_used` interaction |
| Dependent variable | `agency_rating` (1–7 Likert) |

**Model formula:**
```
agency_rating ~ detection_accuracy + prop_used + detection_accuracy:prop_used
```

> Run separately for High and Low control conditions. Adapted from MT_Inference_Analysis.py (Analysis 5a). Pools all trials within a condition without participant clustering. A commented-out mixed LMM alternative (`smf.mixedlm`) is retained in the code.

---

## 5. Multiple Comparisons Correction

| Family | Tests included | Correction | Corrected α |
|---|---|---|---|
| Supplementary (Supp 1–4) | Supp 1 d′ ANOVA, Supp 2 HR ANOVA, Supp 3 GLMM, Supp 4 GLMM | Bonferroni (k = 4) | 0.05 / 4 = **0.0125** |
| Primary (Analyses 1–4) | Analyses 1–4 | No correction (pre-registered primary family) | α = 0.05 |
| Agency (7a–7c) | Analyses 7a–7c | No correction (exploratory) | α = 0.05 |

The Bonferroni-corrected p-values for Supp 1 and Supp 2 interaction terms are computed automatically (`p_corrected = min(1.0, p_uncorr × 4)`). GLMM p-values (Supp 3 & 4) should also be compared against α = 0.0125.

---

## 6. Model Fitting Strategy (GLMMs)

| Step | Action |
|---|---|
| 1. Attempt maximal model | Fit GLMM with full random effects (random intercept + random slope per participant). |
| 2. Singular fit check | If any random-effect variance component is ≈ 0 (< 1e-6), the model is flagged as singular. |
| 3. Fallback | If singular or if the maximal model fails entirely, refit with random-intercept only. |
| 4. Reporting | Always report which random effects structure was used (`'maximal'` or `'intercept-only'`). |

**Package:** `pymer4` ≥ 0.8 (Python wrapper for lme4/R); data passed as Polars DataFrames.
**Family:** Binomial with logit link for all recognition GLMMs.

---

## 7. Contrast Coding Summary

All predictors use **sum/deviation contrast coding (±0.5)** so that main effects are interpretable at the grand mean — not at a reference category.

| Variable | Level A (code = +0.5) | Level B (code = −0.5) |
|---|---|---|
| `control_c` | High | Low |
| `agency_c` | High agency | Low agency |
| `trial_level_c` | High | Low |
| `item_type_c` | Controlled | Uncontrolled |
| `item_is_old_c` | Old (target) | Foil |
| `item_type_c` (foils in A4/S4) | Dummy: rotating 4-cell balanced assignment | Not experimentally meaningful |

---

## 8. Sanity Checks & Data Integrity

| Check | Description |
|---|---|
| Image uniqueness | Confirms all images shown during the test phase were unique within each participant (no image repeated across `img_A_name` / `img_B_name` columns). |
| Calibration convergence | Reports whether the QUEST staircase converged (final `alpha_SD` < 0.20) for each participant × target condition. Plots QUEST `alpha_SD` trajectory over calibration trials. |
| Psychometric function | Plots detection accuracy vs. `prop_used` (calibration data) with a fitted sigmoid and overlaid test-phase mean markers per condition. |
| Accuracy by condition | Bar chart of mean detection accuracy (High vs. Low) in the test phase. |
| Agency ratings | Bar chart of agency ratings by accuracy (correct/incorrect) × condition in the test phase. |
| RT distribution | Histogram of choice reaction times in the test phase, split by condition. |
| False alarm rate (Supp 5) | One-sample t-test confirming FA rates do not significantly differ from 0 (tests for response bias in the recognition phase). |

---

## 9. Output Files

| File / Directory | Contents |
|---|---|
| `analysis_output/pooled/` | Pooled plots across all participants: `sanity_check.png`, `calibration_convergence.png`, `recognition_performance.png`, `dprime_by_condition.png`, `agency_recognition.png` |
| `analysis_output/per_participant/` | Same plots generated per participant (suffix: `_{participant_id}.png`) |
| stdout / console | All statistical results, exclusion logs, and model outputs printed to console. |

---

*Generated from `CDmem_analyses.py`*
*References: Ren et al. (2026); Haridi et al. (2025)*
