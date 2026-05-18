# CDmem Preregistration: Recommended Analyses

## Experiment Overview (for context)

Participants detect which of two moving shapes they are controlling (via partial mouse-direction coupling), across **High** (~85% accuracy target) and **Low** (~55% accuracy target) control conditions calibrated individually with QUEST+. After the control-detection task, a surprise recognition memory test probes whether images encoded during High vs. Low control trials are remembered differently. A foil set (never-seen images) provides the false-alarm baseline for SDT measures.

Key design features relevant to analysis choice:
- **Within-participant 2-condition design** (High vs. Low control at the trial level).
- Each trial also has two image roles: the **controlled** image (the one the participant moved) vs. the **uncontrolled** image (distractor).
- Agency ratings (1–7 Likert) are collected per trial.
- Trial-level binary recognition responses are the primary DV.

---

## Recommended Pre-registered Analyses

### 0. Participant Exclusion Criteria *(must be pre-registered)*

These are already in the script and well-justified — include both verbatim.

| Criterion | Threshold | Reference |
|---|---|---|
| Timeout rate ≥ 50% in either condition (test phase) | `TIMEOUT_THRESHOLD = 0.50` | Haridi et al., 2025 |
| Accuracy z-score > ±2.5 SD from group mean in either condition | `ACCURACY_SD_THRESHOLD = 2.5` | Standard outlier criterion |

**Trial-level:** recognition trials with RT > participant mean + 3 SD excluded (Ren et al., 2026).

---

### 1. Manipulation Check — Control Detection Accuracy (`run_cd_accuracy_analysis`)

> **Pre-register as a prerequisite.** If control detection does not differ between conditions, the memory effect cannot be attributed to the control manipulation.

**Test:** Paired t-test on participant-level mean accuracy (High vs. Low) + binomial GLMM with contrast-coded `control_c` (High = +0.5, Low = −0.5) and random intercepts/slopes per participant.

**Why appropriate:**
- The QUEST+ staircase targets 85% (High) and 55% (Low) — these targets must be reached for the memory comparison to be interpretable.
- Paired t-test is standard for 2-condition within-subject designs and matches the power analysis.
- The GLMM handles trial-level binary data correctly and accounts for participant random effects.

---

### 2. Primary Analysis — d-prime: High vs. Low Control (`run_analysis_1_dprime_ttest`)

> **Pre-register as the primary confirmatory test.**

**Test:** Paired t-test comparing participant-level d-prime between the **High-control** and **Low-control** conditions (controlled items only). Report Cohen's *d*.

**Why this is the primary measure:**
- d-prime is the gold-standard SDT measure; it corrects hit rates for response bias (false alarm rate), making it a pure index of memory discriminability — critical when participants might show yes-bias.
- It directly operationalises the core hypothesis: *"Items encoded under high subjective control are better recognised."*
- The paired t-test is the natural test for the within-participant 2-condition comparison and is fully powered by your Monte Carlo simulation.
- Precedent: virtually all recognition memory × agency studies use d-prime as the primary DV (e.g., Eitam & Higgins, 2010; Chambon et al., 2014; Humphreys et al., 2023).

> [!IMPORTANT]
> d-prime from **controlled items only** (rows where `item_type == 'controlled'`), using the shared foil false alarm rate as the noise baseline. This is the cleanest test of the hypothesis.

---

### 3. Primary Analysis (trial-level) — GLMM Interaction on All Trials (`run_analysis_4_interaction_glmm`)

> **Pre-register as confirmatory, alongside Analysis 1.**

**Test:** Binomial GLMM on **all recognition trials** (targets + foils):
```
said_old ~ item_type_c * control_c + (1 + item_type_c * control_c | participant)
```
Contrast coding: Target = +0.5 / Foil = −0.5; High = +0.5 / Low = −0.5.  
Use maximal-then-fallback random effects structure (report which was used).

**Why Analysis 4, not Analysis 3, is the correct trial-level primary test:**

Analysis 3 (`said_old ~ control_c`, targets only) has **no false-alarm baseline** — its fixed effect of `control_c` captures only how often participants say "old" to old items, i.e., it is a **hit-rate GLMM**, not a d-prime analog.

Analysis 4 includes foils and therefore:
- The **`item_type_c` main effect** captures overall recognition sensitivity (hit rate vs. FA rate — the logit-scale equivalent of overall d-prime).
- The **`item_type_c × control_c` interaction** captures whether the hit-minus-FA gap is larger in the High condition — which is exactly what d-prime tests at the participant level, now estimated at the trial level via a logit link.

This is the proper trial-level signal-detection GLM (see DeCarlo, 1998; Sheu & Chen, 2024 for SDT-GLMM equivalences).

> [!NOTE]
> **Foil dummy-coding caveat:** Foils have no real control condition (they were never shown during the control task). In the script they receive a balanced dummy `control_c` assignment (first half +0.5, second half −0.5 per participant). Because FA rates are expected to be identical across both dummy cells by design, the interaction is empirically driven by the **target trials only**, even though foils provide the FA baseline. State this explicitly in the Methods.

---

### 4. Secondary Analysis — Hit Rate Comparison (`run_analysis_2_hitrate_ttest`)

> **Pre-register as secondary / supporting.**

**Test:** Paired t-test on participant-level hit rates (High vs. Low, controlled items only).

**Why secondary (not primary):**
- Hit rates are not corrected for response bias; they can be inflated by liberal responding.
- However, including it alongside d-prime is standard practice (readers expect both), and it provides the descriptive means most easily communicated.
- **Do not treat this as a separate hypothesis** — it tests the same thing as Analysis 2 in a less controlled way. Report it as a consistency check.

---

### 5. Secondary Analysis — False Alarm Rate Sanity Check (`run_supp_analysis_5_fa_check`)

> **Pre-register as a manipulation/data quality check.**

**Test:** One-sample t-test of FA rates against 0.

**Why include:**
- The entire SDT approach depends on foils being genuinely unseen. If FA rates are high or variable, d-prime estimates are unreliable.
- Pre-registering this is good practice and increases reviewer confidence.
- It's a quick, unambiguous check.

---

### 6. Planned Secondary — 2×2 GLMM: Trial Level × Item Type (`run_supp_analysis_3_glmm_2x2`)

> **Pre-register as a planned secondary analysis** (not primary — it goes beyond the core hypothesis but is theoretically motivated).

**Test:** Binomial GLMM on all old items (controlled + uncontrolled):
```
said_old ~ trial_level_c * item_type_c + (1 + trial_level_c * item_type_c | participant)
```
Contrast coding: High = +0.5 / Low = −0.5; Controlled = +0.5 / Uncontrolled = −0.5.

**Why this matters scientifically:**
- This is arguably the most theoretically informative analysis. It tests whether the memory advantage is specifically for the **controlled** object (the one actively moved), or whether it generalises to the co-present uncontrolled object.
- A **main effect of trial_level only** (no interaction) → control level at encoding generally boosts memory for all items in the trial (arousal/attention account).
- An **interaction** → the memory boost is specific to the controlled item, implicating action-binding or selective attention (agency account).
- This dissociation is central to distinguishing agency accounts (Elsner & Hommel, 2001) from general arousal/attention accounts.
- Compatible papers (Ren et al., 2026; Chambon et al., 2014) report exactly this 2×2 structure.

> [!NOTE]
> Pre-register the **GLMM version** (Supp Analysis 3) as the primary test of this 2×2 question. The RM-ANOVA versions (Supp 1 & 2) can be mentioned as summary-statistics complements but are less appropriate for binary DVs.

---

### 7. Exploratory (Pre-registered as Exploratory) — Agency → Memory (`run_analysis_7_agency_glmm`)

> **Pre-register as an exploratory analysis** (openly, not as confirmatory). Label it clearly in the paper.

**Test:** Binomial GLMM on old items with continuous, within-participant z-scored agency rating as predictor:
```
said_old ~ agency_rating_z + (1 + agency_rating_z | participant)
```

**Why include:**
- Agency ratings are the subjective counterpart to the objective control manipulation. Testing whether *felt* agency predicts memory — over and above the objective condition — is a key mechanistic question.
- Pre-registering it as exploratory is honest and protects against inflated Type I error, while signalling you intended to test it.
- Multiple studies show that within-trial SoA ratings predict memory independently of accuracy (e.g., Damen et al., 2024; Humphreys et al., 2023 style analyses).
- Within-participant z-scoring (already implemented) is best practice to remove scale-use bias.

---

## What NOT to Pre-register (run post-hoc / exploratory only)

| Analysis | Reason to exclude from preregistration |
|---|---|
| **Analysis 3** (GLMM, targets only) | Without foils, the `control_c` effect is a hit-rate GLMM, not a d-prime analog. Superseded by Analysis 4 as the primary trial-level test. Can still be reported as a supplementary sensitivity check. |
| **Supp Analysis 4** (3-way GLMM with foils) | Model is very complex (3-way interaction), singular fits likely with small N. The foil dummy 2×2 assignment is harder to defend. Run as exploratory only. |
| **Analysis 5a** (OLS: agency ~ accuracy + prop_used) | Pools across participants without accounting for clustering — mixed LMM is preferable and would need more justification. |
| **Supp 1 & 2** (RM-ANOVAs on d-prime / hit rate) | RM-ANOVA on bounded/binary-derived DVs is less appropriate than GLMM. Include as descriptive summaries only. |

---

## Recommended Preregistration Structure

```
1. Exclusion Criteria
   1a. Timeout rate ≥ 50% per condition (participant-level)
   1b. Accuracy z-score > ±2.5 SD (participant-level)
   1c. Recognition RT > mean + 3 SD (trial-level)

2. Manipulation Check
   2a. Paired t-test: detection accuracy (High vs. Low)
   2b. Binomial GLMM: detection_accuracy ~ control_c + (1 + control_c | participant)

3. Primary Confirmatory Analyses (Memory: High vs. Low Control)
   3a. Analysis 1 — Paired t-test on d-prime
         (controlled items only, shared FA rate; participant-level)
   3b. Analysis 4 — Binomial GLMM on ALL recognition trials (targets + foils)
         said_old ~ item_type_c * control_c + (1 + item_type_c * control_c | participant)
         → item_type_c × control_c interaction = trial-level d-prime analog
         (Note foil dummy-coding in methods)

4. Secondary Analyses
   4a. Analysis 2 — Paired t-test on hit rate (controlled items; consistency check for 3a)
   4b. FA rate sanity check (one-sample t-test vs. 0)

5. Planned Secondary: 2×2 Design (Trial Level × Item Type)
   5a. Supp Analysis 3 — 2×2 GLMM on all old items
         said_old ~ trial_level_c * item_type_c + (1 + trial_level_c * item_type_c | participant)

6. Exploratory (pre-registered as exploratory)
   6a. Analysis 7 — Continuous agency → memory GLMM
         said_old ~ agency_rating_z + (1 + agency_rating_z | participant)
```

---

## Key Literature Supporting These Choices

- **d-prime as primary:** Macmillan & Creelman (2004); Green & Swets (1966) — SDT is the standard for recognition memory.
- **SDT-GLMM equivalence:** DeCarlo (1998, Psychological Methods); Sheu & Chen (2024) — including foils as a separate item_type level in a GLMM is the trial-level equivalent of signal detection theory.
- **Control → memory effect:** Eitam & Higgins (2010, Psychological Review) — "relevance" drives memory encoding; control is a key relevance cue.
- **Controlled vs. uncontrolled item distinction:** Elsner & Hommel (2001, JEP:General) — action-effect binding selectively links outcomes to their causative actions.
- **Agency ratings → memory:** Chambon et al. (2014, Cognition); Humphreys et al. (2023) — within-person SoA variation predicts subsequent memory.
- **GLMM for binary recognition:** Ren et al. (2026) — maximal random effects structure with binomial link for recognition data.
- **Within-participant z-scoring of agency ratings:** removes between-person scale-use variance before entering as predictor (standard practice in SoA / metacognition literature).
