# Power Analysis Results

## Parameters

- Simulations per cell: 1000
- Alpha level: 0.05
- Target trials per condition: 60
- Foil trials total: 240
- Uncontrolled old items total: 120 (2 x 60)
- Baseline hit rate: 0.62 (Wu et al., 2025, Exp. 4)
- Baseline false alarm rate: 0.2 (assumed)
- Baseline d-prime: 0.74 (avg of Schreiner et al. conditions)
- D-prime pooled SD: 0.75 (Schreiner et al.)
- Hit rate SD: 0.2 (Schreiner et al.)
- Random intercept SD: 1.36 (converted from d-prime SD via log-odds)
- Within-person correlation (RM ANOVA): 0.5
- Agency slope small: 0.2 log-odds/SD
- Agency slope medium: 0.5 log-odds/SD

## Primary Analysis Descriptions

Analyses 1-4 mirror run_analysis_1 to run_analysis_4 in CDmem_analyses.py.
Effect: High > Low control on recognition of controlled (target) items.

1. **d-prime paired t-test**: D-prime per participant per condition, two-tailed paired t-test (Schreiner et al., 2024).
2. **Hit rate paired t-test**: Mean hit rate per participant per condition, paired t-test (Wu et al., 2025).
3. **GEE hits only**: Trial-level GEE on target trials only. recognized ~ control_level (±0.5). Binomial/logit/exchangeable.
4. **GEE item_type x control**: Trial-level GEE on all trials. said_old ~ item_type_c * control_c. Foils dummy-assigned equally (balanced).

## Supplementary Analysis Descriptions

Analyses Supp1-4 mirror run_supp_analysis_1 to run_supp_analysis_4 in CDmem_analyses.py.
Design: 2x2 factorial — trial_level (High/Low) x item_type (Controlled/Uncontrolled).
**Key test in all supplementary analyses: trial_level x item_type INTERACTION.**
  Significant interaction => motor control per se (not co-presence) drives memory.
**Conservative assumption**: Uncontrolled items have the grand mean hit rate (no trial-level effect).

S1. **d-prime 2x2 RM ANOVA**: Four d-primes per participant (2x2 cells), statsmodels AnovaRM. Same estimator as CDmem_analyses.py.
S2. **Hit rate 2x2 RM ANOVA**: Same 2x2 design on hit rates.
S3. **GEE 2x2 old items**: recognized ~ trial_level_c * item_type_c on all 240 old trials. Interaction = motor control effect.
S4. **GEE 3-way all trials**: said_old ~ item_is_old_c * trial_level_c * item_type_c. Foils dummy-assigned to balanced 2x2 cells. 3-way interaction is key test.

## Analysis 7 Description (Exploratory)

Mirrors run_analysis_7_agency_glmm in CDmem_analyses.py.
Model: recognized ~ agency_rating_z (within-participant z-scored agency).
Effect parameterised as log-odds slope per 1 SD of agency (no prior estimate; exploratory).
Small slope: b=0.2 (~+5% hit rate per SD). Medium: b=0.5 (~+12% per SD).

## Effect Size Conversions

- Cohen's d = 0.2: delta d-prime = 0.150 (0.665 vs 0.815), delta hit rate = 0.040 (0.600 vs 0.640)
- Cohen's d = 0.5: delta d-prime = 0.375 (0.552 vs 0.927), delta hit rate = 0.100 (0.570 vs 0.670)

## Primary Analysis Results

### Small (d=0.2)

| Analysis | N=30 | N=40 | N=50 | N=60 |
|---|---|---|---|---|
| Analysis 1: d-prime paired t-test | 0.137 | 0.128 | 0.165 | 0.209 |
| Analysis 2: Hit rate paired t-test | 0.109 | 0.159 | 0.138 | 0.176 |
| Analysis 3: GEE hits only | 0.575 | 0.712 | 0.814 [80%+] | 0.853 [80%+] |
| Analysis 4: GEE item_type x control | 0.433 | 0.492 | 0.596 | 0.684 |

[80%+] = meets 80% power threshold

### Medium (d=0.5)

| Analysis | N=30 | N=40 | N=50 | N=60 |
|---|---|---|---|---|
| Analysis 1: d-prime paired t-test | 0.439 | 0.574 | 0.681 | 0.771 |
| Analysis 2: Hit rate paired t-test | 0.450 | 0.586 | 0.675 | 0.784 |
| Analysis 3: GEE hits only | 0.999 [80%+] | 1.000 [80%+] | 1.000 [80%+] | 1.000 [80%+] |
| Analysis 4: GEE item_type x control | 0.985 [80%+] | 0.995 [80%+] | 1.000 [80%+] | 1.000 [80%+] |

[80%+] = meets 80% power threshold

## Supplementary Analysis Results

### Small (d=0.2)

| Analysis | N=30 | N=40 | N=50 | N=60 |
|---|---|---|---|---|
| Supp 1: d-prime 2x2 RM ANOVA (interaction) | 0.124 | 0.156 | 0.152 | 0.212 |
| Supp 2: Hit rate 2x2 RM ANOVA (interaction) | 0.118 | 0.138 | 0.163 | 0.179 |
| Supp 3: GEE 2x2 old items (interaction) | 0.313 | 0.434 | 0.544 | 0.600 |
| Supp 4: GEE 3-way all trials (3-way interaction) | 0.204 | 0.232 | 0.265 | 0.311 |

[80%+] = meets 80% power threshold

### Medium (d=0.5)

| Analysis | N=30 | N=40 | N=50 | N=60 |
|---|---|---|---|---|
| Supp 1: d-prime 2x2 RM ANOVA (interaction) | 0.471 | 0.582 | 0.707 | 0.764 |
| Supp 2: Hit rate 2x2 RM ANOVA (interaction) | 0.476 | 0.582 | 0.698 | 0.763 |
| Supp 3: GEE 2x2 old items (interaction) | 0.961 [80%+] | 0.995 [80%+] | 1.000 [80%+] | 1.000 [80%+] |
| Supp 4: GEE 3-way all trials (3-way interaction) | 0.691 | 0.827 [80%+] | 0.898 [80%+] | 0.943 [80%+] |

[80%+] = meets 80% power threshold

## Analysis 7 (Exploratory) Analysis Results

### Small (b=0.2)

| Analysis | N=30 | N=40 | N=50 | N=60 |
|---|---|---|---|---|
| Analysis 7: GEE agency_z -> recognized | 0.998 [80%+] | 1.000 [80%+] | 1.000 [80%+] | 1.000 [80%+] |

[80%+] = meets 80% power threshold

### Medium (b=0.5)

| Analysis | N=30 | N=40 | N=50 | N=60 |
|---|---|---|---|---|
| Analysis 7: GEE agency_z -> recognized | 1.000 [80%+] | 1.000 [80%+] | 1.000 [80%+] | 1.000 [80%+] |

[80%+] = meets 80% power threshold

## Notes and Limitations

- **GEE vs GLMM**: Power simulation uses GEE (statsmodels); confirmatory analyses use GLMM (pymer4/lme4). GEE power estimates are conservative due to non-collapsibility of the logit link (Skrondal & Rabe-Hesketh, 2004). Actual GLMM power >= estimated.
- **GEE citation**: Rochon (1998) and Liu & Liang (1997) recommend GEE simulation for planning studies with correlated binary outcomes.
- **RM ANOVA**: Supp 1-2 use AnovaRM (statsmodels), matching CDmem_analyses.py. Within-person correlation modelled via random intercept decomposition (rho=0.50).
- **Random intercept SD**: Estimated from d-prime SD via log-odds conversion. Actual between-participant variability is unknown.
- **False alarm rate**: Assumed 0.20. Affects only Analysis 4 (foil log-odds) and Supp 4 (foil dummy cells).
- **Supplementary interaction**: Power reported for trial_level x item_type interaction (2x2). Conservative: uncontrolled items have no trial-level effect in DGP.
- **Analysis 7**: Exploratory. No prior estimate for agency-memory slope; power reported to inform sample size planning only.
- **Recommendation**: For small effects (d=0.2 or b=0.20), N=30-60 is likely underpowered. Frame as exploratory. For medium effects, GEE/GLMM trial-level analyses offer more power by leveraging within-participant trial variance.