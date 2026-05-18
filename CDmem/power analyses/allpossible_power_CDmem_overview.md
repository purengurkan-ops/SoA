# CDmem Power Analysis Overview

**File:** `allpossible_power_CDmem.py`  
Monte Carlo simulation-based power analysis for the Control Detection + Memory (CDmem) experiment.

---

## Purpose

Estimates statistical power for detecting a **control-level effect on recognition memory** across four primary analyses, four supplementary analyses, and one exploratory analysis (Analysis 7). Analyses are crossed with two effect sizes and four sample sizes.

All analysis functions in this script are designed to **mirror the corresponding analyses in `CDmem_analyses.py`** — same design, same predictors, same contrast coding. The only difference is the estimator: **GEE** is used instead of GLMM for simulation stability (see below).

---

## Simulation Settings

| Parameter | Value | Notes |
|---|---|---|
| `N_SIMULATIONS` | 1000 | Monte Carlo iterations per cell. Use 200 for quick testing, 5000 for final results. |
| `SAMPLE_SIZES` | [30, 40, 50, 60] | Target range for an EEG study with practical constraints |
| `ALPHA` | 0.05 | Two-tailed significance threshold |
| `N_TRIALS_PER_CONDITION` | 60 | Target (old) trials per condition (High / Low control) |
| `N_FOILS` | 240 | New (never-seen) foil trials total |
| `N_UNCONTROLLED_PER_CONDITION` | 60 | Uncontrolled old items per condition of origin |
| `N_UNCONTROLLED` | 120 | Total uncontrolled old items (2 × 60) |
| `np.random.seed` | 42 | Fixed for reproducibility |

---

## Parameters from the Literature

| Parameter | Value | Source |
|---|---|---|
| `BASELINE_DPRIME` | 0.74 | Average of congruent (0.78) and incongruent (0.70) conditions — Schreiner et al. (2024) |
| `DPRIME_SD` | 0.75 | Pooled SD across conditions — Schreiner et al. (2024) |
| `BASELINE_HIT_RATE` | 0.62 | Schreiner et al. (2024), congruent condition; confirmed by Wu et al. (2025), Exp. 4 |
| `HIT_RATE_SD` | 0.20 | Schreiner et al. (2024) |
| `BASELINE_FA_RATE` | 0.20 | Not directly reported; assumed as plausible value. Only enters Analyses 4 and Supp 4. |
| `RI_SD` | 1.36 | Random intercept SD estimated by converting d′ SD to log-odds: `DPRIME_SD × (π / √3) ≈ 1.36` |
| `WITHIN_PERSON_CORR` | 0.50 | Within-participant correlation across conditions, used for RM ANOVA simulation |
| `AGENCY_SLOPE_SMALL` | 0.20 | Log-odds per +1 SD agency (small effect, ~+5% hit rate at baseline) |
| `AGENCY_SLOPE_MEDIUM` | 0.50 | Log-odds per +1 SD agency (medium effect, ~+12% hit rate at baseline) |

---

## Why GEE (Not GLMM) for Trial-Level Simulations?

GEE is used for trial-level power simulations instead of GLMM because:

1. **Computational stability** — GEE almost never fails to converge across 1000+ iterations; GLMM convergence failures require complex handling and reduce effective simulation count.
2. **Conservative estimates** — GEE coefficients are attenuated relative to GLMM subject-specific coefficients due to the **non-collapsibility of the logit link** (Skrondal & Rabe-Hesketh, 2004; Diggle et al., 2002). Therefore: **actual GLMM power ≥ GEE-estimated power**. Using GEE is conservative and safe for planning.
3. **Established precedent** — Rochon (1998) and Liu & Liang (1997) explicitly recommend GEE-based simulation for power analysis with correlated binary outcomes.

> **Implication for methods section:** Note that GEE was used for power estimation (conservative approximation), while the confirmatory analyses use GLMM (pymer4/lme4).

---

## Helper Functions

| Function | Description |
|---|---|
| `cohens_d_to_dprime_diff(d)` | Converts Cohen's d to a raw d′ difference using `DPRIME_SD` |
| `cohens_d_to_hr_diff(d)` | Converts Cohen's d to a raw hit rate difference using `HIT_RATE_SD` |
| `hr_to_log_odds(hr)` | Converts a probability to log-odds; clipped to [0.001, 0.999] to avoid ±∞ |
| `_sim_2x2_dprime(n, ...)` | Generates correlated 2×2 d-prime data via random-intercept decomposition (rho = 0.50) |
| `_sim_2x2_hitrate(n, ...)` | Same as above but using `HIT_RATE_SD` |

---

## Effect Size Conversions

Cohen's d is converted to raw differences via:
- **d′ difference:** `Δd′ = Cohen's d × DPRIME_SD`
- **Hit rate difference:** `ΔHR = Cohen's d × HIT_RATE_SD`

| Cohen's d | Δd′ | d′ Low vs. High | ΔHR | HR Low vs. High |
|---|---|---|---|---|
| 0.2 (Small) | 0.150 | 0.665 vs. 0.815 | 0.040 | 0.600 vs. 0.640 |
| 0.5 (Medium) | 0.375 | 0.553 vs. 0.928 | 0.100 | 0.570 vs. 0.670 |

---

## Primary Analyses

These test the **High vs. Low control** contrast on controlled target items only. Mirror `run_analysis_1` to `run_analysis_4` in `CDmem_analyses.py`.

---

### Analysis 1 — d′ Paired t-test

| Attribute | Detail |
|---|---|
| Type | Paired-samples t-test |
| Package | `scipy.stats.ttest_rel` |
| IV | Control condition (High vs. Low) — within-subject |
| DV | d′ per participant per condition |
| Data generating process | d′ per participant drawn from N(mean_condition, DPRIME_SD); means separated by Δd′ |
| Reference | Standard SDT approach — Schreiner et al. (2024) |

---

### Analysis 2 — Hit Rate Paired t-test

| Attribute | Detail |
|---|---|
| Type | Paired-samples t-test |
| Package | `scipy.stats.ttest_rel` |
| IV | Control condition (High vs. Low) |
| DV | Mean hit rate per participant per condition |
| Data generating process | Per-participant hit rate drawn from N(mean_condition, HIT_RATE_SD) |
| Reference | Wu et al. (2025); supplemental analysis in Schreiner et al. (2024) |

---

### Analysis 3 — GEE: Target Trials Only

| Attribute | Detail |
|---|---|
| Type | GEE, Binomial / logit / Exchangeable |
| Package | `statsmodels.formula.api.gee` |
| IV | `control`: contrast-coded (High = +0.5, Low = −0.5) |
| DV | `recognized`: binary hit/miss per trial (0/1) |
| Test statistic | p-value for `control` coefficient |

**Model formula:**
```
recognized ~ control
groups = participant
```

**Data generating process:**
1. Each participant gets a random intercept from N(0, RI_SD)
2. Trial log-odds: `intercept + RI + true_slope × control`
3. Hit/miss sampled from Bernoulli with p = logistic(log-odds)

> Avoids the foil dummy-assignment problem entirely by analysing only target trials.

---

### Analysis 4 — GEE: Item Type × Control Interaction (All Trials)

| Attribute | Detail |
|---|---|
| Type | GEE, Binomial / logit / Exchangeable |
| Package | `statsmodels.formula.api.gee` |
| IVs | `item_type_c` (Target = +0.5, Foil = −0.5); `control_c` (High = +0.5, Low = −0.5); interaction |
| DV | `said_old`: binary 'old' response per trial |
| Test statistic | p-value for `item_type:control` interaction |

**Model formula:**
```
said_old ~ item_type * control
groups = participant
```

> Foils have no genuine control condition and are split equally (120 per dummy condition). FA rate is identical across dummy conditions by construction — the interaction is driven entirely by the change in hit rate.

---

## Supplementary Analyses

These analyses exploit the **2×2 trial structure: Trial Level (High/Low) × Item Type (Controlled/Uncontrolled)**. Mirror `run_supp_analysis_1` to `run_supp_analysis_4` in `CDmem_analyses.py`.

**Design rationale:** On each encoding trial, participants see two images — one *controlled* item (cursor tracked it) and one *uncontrolled* item (moving randomly). Both appear at test alongside foils, yielding four old-item cells:

| Cell | N per participant | Trial Level | Item Type |
|---|---|---|---|
| High × Controlled | 60 | High | Controlled |
| Low × Controlled | 60 | Low | Controlled |
| High × Uncontrolled | 60 | High | Uncontrolled |
| Low × Uncontrolled | 60 | Low | Uncontrolled |

> **Key test in ALL supplementary analyses:** The **Trial Level × Item Type interaction**. A significant interaction means motor control per se — not mere co-presence on a trial — drives memory (controlled items show a trial-level effect; uncontrolled items do not).

> **Conservative data generating process:** Uncontrolled items have the **grand mean** recognition probability — no trial-level effect in either direction. Power would be higher if uncontrolled items are remembered more poorly than controlled items.

> **Caveat:** Uncontrolled items were viewed in the same context as controlled items — they are not a clean "no encoding" baseline. Interpret supplementary results with caution.

---

### Supplementary Analysis 1 — 2×2 RM ANOVA on d′

| Attribute | Detail |
|---|---|
| Type | Repeated-Measures ANOVA |
| Package | `statsmodels.stats.anova.AnovaRM` (same estimator as `CDmem_analyses.py`) |
| IVs | `trial_level` (High / Low) × `item_type` (Controlled / Uncontrolled) — both within-subject |
| DV | d′ per participant per 2×2 cell (4 values per participant) |
| Key test | `trial_level:item_type` interaction |
| Within-person correlation | Modelled via random-intercept decomposition: `total_SD² = between_SD² + within_SD²`, rho = 0.50 |

**DGP cell means:**

| Cell | Mean d′ |
|---|---|
| High, Controlled | BASELINE_DPRIME + Δd′/2 |
| Low, Controlled | BASELINE_DPRIME − Δd′/2 |
| High, Uncontrolled | BASELINE_DPRIME (no effect) |
| Low, Uncontrolled | BASELINE_DPRIME (no effect) |

> **Replaces the old one-way 3-level `f_oneway` approximation.** AnovaRM properly removes between-subject variance from the error term, giving **accurate** power estimates for the 2×2 interaction.

---

### Supplementary Analysis 2 — 2×2 RM ANOVA on Hit Rate

| Attribute | Detail |
|---|---|
| Type | Repeated-Measures ANOVA |
| Package | `statsmodels.stats.anova.AnovaRM` |
| IVs | `trial_level` × `item_type` (both within-subject) |
| DV | Hit rate per participant per 2×2 cell |
| Key test | `trial_level:item_type` interaction |

> Same 2×2 design and correlated DGP as Supp 1, using `HIT_RATE_SD`.

---

### Supplementary Analysis 3 — 2×2 GEE: All Old Items

| Attribute | Detail |
|---|---|
| Type | GEE, Binomial / logit / Exchangeable |
| Package | `statsmodels.formula.api.gee` |
| IVs | `trial_level_c` (High = +0.5, Low = −0.5); `item_type_c` (Controlled = +0.5, Uncontrolled = −0.5); interaction |
| DV | `recognized`: binary hit/miss per trial |
| Groups | Participant |
| Key test | `trial_level_c:item_type_c` interaction |
| N trials | 240 per participant (4 cells × 60 trials) |

**Model formula:**
```
recognized ~ trial_level_c * item_type_c
groups = participant
```

**Why the interaction equals the key effect:** With ±0.5 contrast coding:
```
β_int = lo(Hi,Ctrl) − lo(Lo,Ctrl) − lo(Hi,Unc) + lo(Lo,Unc)
      = (lo_high − lo_low) − 0 = true_slope_controlled
```
Because uncontrolled items have identical log-odds regardless of trial level, the interaction coefficient equals the full trial-level slope on controlled items.

> **Replaces the old 3-level `is_high + is_low` dummy-coded model.**

---

### Supplementary Analysis 4 — 3-Way GEE: Is_Old × Trial Level × Item Type (All Trials)

| Attribute | Detail |
|---|---|
| Type | GEE, Binomial / logit / Exchangeable |
| Package | `statsmodels.formula.api.gee` |
| IVs | `item_is_old_c` (Old = +0.5, Foil = −0.5); `trial_level_c` (High = +0.5, Low = −0.5); `item_type_c` (Controlled = +0.5, Uncontrolled = −0.5); all interactions |
| DV | `said_old`: binary 'old' response |
| Groups | Participant |
| Key test | `item_is_old_c:trial_level_c:item_type_c` 3-way interaction |
| N trials | 480 per participant (240 old + 240 foils) |

**Model formula:**
```
said_old ~ item_is_old_c * trial_level_c * item_type_c
groups = participant
```

**Foil dummy assignment:** 240 foils split into 4 balanced cells of 60 (matching `run_supp_analysis_4_glmm_foils`). All foil cells share the same FA rate — dummy assignment does not bias the 3-way interaction.

**Why the 3-way = the key effect:**
- Old items show a 2×2 interaction (controlled: Hi > Lo; uncontrolled: no effect)
- Foils show no trial/type pattern (FA rate is constant)
- The 3-way tests whether the 2×2 pattern in old items differs from the (flat) foil pattern → significant whenever old items show the expected interaction

> **Replaces the old 3-level `item_is_old * is_high + item_is_old * is_low` model.**

---

## Analysis 7 — Agency Rating → Recognition Memory (Exploratory)

Mirrors `run_analysis_7_agency_glmm` in `CDmem_analyses.py`. **Labelled exploratory** — no prior literature provides a point estimate for this paradigm.

| Attribute | Detail |
|---|---|
| Type | GEE, Binomial / logit / Exchangeable |
| Package | `statsmodels.formula.api.gee` |
| IV | `agency_z`: agency rating z-scored within each participant (continuous, SD = 1) |
| DV | `recognized`: binary recognition response |
| Groups | Participant |
| Key test | p-value for `agency_z` slope |
| N trials | 120 per participant (60 High + 60 Low target trials) |

**Model formula:**
```
recognized ~ agency_z
groups = participant
```

**Effect size parameterisation (log-odds slope per +1 SD agency):**

| Label | Value | Interpretation |
|---|---|---|
| Small (`AGENCY_SLOPE_SMALL`) | 0.20 | +1 SD agency → ~+5% hit rate at baseline (0.62 → 0.67) |
| Medium (`AGENCY_SLOPE_MEDIUM`) | 0.50 | +1 SD agency → ~+12% hit rate at baseline (0.62 → 0.74) |

**Data generating process:**
1. Each participant has 120 target trials. Agency z-scores drawn i.i.d. from N(0, 1).
2. Hit probability: `p = expit(lo_baseline + RI + slope × agency_z)`
3. GEE tests whether the slope ≠ 0.

> Within-participant z-scoring is implicit: the simulated predictor already has mean = 0, SD = 1 per participant, matching the z-scoring performed in the actual script.

---

## Grid of Simulations

| Dimension | Values |
|---|---|
| Effect sizes (primary + supp) | Small (d = 0.2), Medium (d = 0.5) |
| Effect sizes (Analysis 7) | Small (b = 0.20), Medium (b = 0.50) |
| Sample sizes | N = 30, 40, 50, 60 |
| Primary analyses | 4 |
| Supplementary analyses | 4 |
| Analysis 7 | 1 |
| **Total cells** | (2 × 4 × 4) + (2 × 4 × 4) + (2 × 4) = **72** |

Each cell runs `N_SIMULATIONS` = 1000 Monte Carlo iterations. Power = proportion of iterations yielding p < 0.05.

---

## Output

| Output | Description |
|---|---|
| Console | Progress log per cell: analysis, effect size, N, power, runtime |
| `results_CDmem.md` | Formatted markdown tables: parameters, analysis descriptions, effect size conversions, power tables (Primary / Supplementary / Analysis 7), notes |

Power cells meeting the **80% threshold** are flagged with `[80%+]` in the output tables.

---

## Notes and Limitations

| Limitation | Detail |
|---|---|
| GEE vs. GLMM | Power simulations use GEE; confirmatory analyses use GLMM (pymer4/lme4). GEE estimates are conservative (actual GLMM power ≥ estimated). See Skrondal & Rabe-Hesketh (2004), Rochon (1998). |
| RM ANOVA estimator | AnovaRM (statsmodels) used for Supp 1 & 2 — same estimator as `CDmem_analyses.py`. Within-person correlation modelled via random-intercept decomposition (rho = 0.50). |
| Random intercept SD | Estimated indirectly from d′ SD via log-odds conversion. Actual between-participant variability is unknown. |
| False alarm rate | Assumed 0.20. Only enters Analyses 4 and Supp 4 via foil log-odds. |
| Conservative DGP | Uncontrolled items have the grand mean recognition probability in the simulation. Power for Supp 1–4 is likely an underestimate. |
| Analysis 7 | Exploratory. No prior estimate for agency-memory slope. Power reported to inform sample size planning only; do not treat as confirmatory. |
| Literature source | Parameters from Schreiner et al.'s choice paradigm — true effect in a cursor control paradigm may differ. |

---

## Recommendation

> For **small effects (d = 0.2 or b = 0.20):** N = 30–60 is likely underpowered across all analyses. Frame as exploratory.
>
> For **medium effects (d = 0.5):** Trial-level GEE analyses (Analyses 3 & 4, Supp 3 & 4) offer more power than paired t-tests or RM ANOVAs by leveraging within-participant trial variance.

---

*Generated from `allpossible_power_CDmem.py`*  
*References: Schreiner et al. (2024); Wu et al. (2025); Rochon (1998); Skrondal & Rabe-Hesketh (2004)*
