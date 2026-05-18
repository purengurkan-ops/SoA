"""
===============================================================================
Power Analysis for Recognition Memory Experiment
===============================================================================
This script estimates statistical power for detecting a control-level effect
on recognition memory. Analyses mirror CDmem_analyses.py exactly:

PRIMARY ANALYSES (High vs. Low control — controlled targets only)
  1. Paired t-test on d-prime         (run_analysis_1_dprime_ttest)
  2. Paired t-test on hit rates        (run_analysis_2_hitrate_ttest)
  3. GEE on target trials only         (run_analysis_3_glmm)
  4. GEE with item_type x control      (run_analysis_4_interaction_glmm)

SUPPLEMENTARY ANALYSES (2x2: Trial Level x Item Type)
  Supp 1. 2x2 RM ANOVA on d-prime     (run_supp_analysis_1_dprime_2x2_anova)
            Trial Level (High/Low) x Item Type (Controlled/Uncontrolled)
            Key test: interaction (trial_level x item_type)
  Supp 2. 2x2 RM ANOVA on hit rates   (run_supp_analysis_2_hitrate_2x2_anova)
  Supp 3. 2x2 GEE on all old items    (run_supp_analysis_3_glmm_2x2)
            said_old ~ trial_level_c * item_type_c
  Supp 4. 3-way GEE on all trials     (run_supp_analysis_4_glmm_foils)
            said_old ~ item_is_old_c * trial_level_c * item_type_c
            Foils dummy-assigned to balanced 2x2 cells (same as actual script)

ANALYSIS 7: Agency -> Memory  (run_analysis_7_agency_glmm)
  Continuous within-participant z-scored agency_rating predicts said_old.
  Two effect sizes: small (b=0.20 log-odds/SD) and medium (b=0.50).
  Labelled exploratory; no prior literature provides a point estimate.

GEE IS USED INSTEAD OF GLMM FOR SIMULATION BECAUSE:
  - GEE simulation is computationally stable across thousands of iterations.
  - GEE-based power estimates are CONSERVATIVE relative to GLMM due to
    non-collapsibility of the logit link (Skrondal & Rabe-Hesketh, 2004;
    Diggle et al., 2002; Rochon, 1998), so actual GLMM power >= estimated.

PARAMETERS sourced from:
  - Schreiner et al. (2024): d-prime means/SDs, hit rate means/SDs
  - Wu et al. (2025): baseline hit rate from Experiment 4

OUTPUT:
  - Console progress output
  - results_CDmem.md saved in the current working directory
===============================================================================
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit   # logistic function: 1 / (1 + exp(-x))
from scipy.stats import norm      # normal distribution for d-prime conversions
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.anova import AnovaRM   # proper within-subjects RM ANOVA
import warnings
import time
warnings.filterwarnings('ignore')

np.random.seed(42)


# ==============================================================================
# SIMULATION SETTINGS
# ==============================================================================

N_SIMULATIONS          = 1000   # Monte Carlo iterations per cell
                                 # Use 200 for quick testing, 5000 for final results
SAMPLE_SIZES           = [30, 40, 50, 60]
ALPHA                  = 0.05   # two-tailed significance threshold
N_TRIALS_PER_CONDITION = 60     # target trials per condition (high / low control)
N_FOILS                = 240    # new (never-seen) foil trials total


# ==============================================================================
# PARAMETERS FROM THE LITERATURE
# ==============================================================================

# D-prime values from Schreiner et al. (2024):
#   Congruent condition:   M = 0.78, SD = 0.75
#   Incongruent condition: M = 0.70, SD = 0.74
BASELINE_DPRIME = 0.74   # average of the two conditions
DPRIME_SD       = 0.75   # pooled SD — used to convert Cohen's d to delta d-prime

# Hit rate values from Schreiner et al. (2024) and Wu et al. (2025):
#   Schreiner congruent: M = 0.62, SD ≈ 0.20
#   Wu et al. Exp 4 intermediate condition: ~0.62
BASELINE_HIT_RATE = 0.62
HIT_RATE_SD       = 0.20

# False alarm rate — not directly reported in the available literature.
# Assumed to be 0.20 as a plausible value for old/new recognition tasks.
# Affects only the d-prime-to-hit-rate conversion.
BASELINE_FA_RATE = 0.20

# Random intercept SD for the GEE data generating process.
# Captures between-participant variability in overall recognition tendency.
# Estimated by converting d-prime SD to log-odds scale:
#   RI_SD ~ DPRIME_SD x (pi / sqrt(3)) ~ 0.75 x 1.814 ~ 1.36
# This is a rough approximation — the actual value is unknown.
RI_SD = 1.36

# Within-participant correlation for 2x2 RM ANOVA simulation.
# Decomposed as: total_SD^2 = between_SD^2 + within_SD^2.
# With rho = 0.50: between_SD = within_SD = DPRIME_SD * sqrt(0.5).
# Removing between-subject variance from ANOVA error increases power
# relative to f_oneway (independent groups), giving more accurate estimates.
WITHIN_PERSON_CORR = 0.50

# Agency -> Memory effect sizes (Analysis 7).
# Defined directly as log-odds slope per 1 SD of within-participant agency.
# Small (0.20): +1 SD agency => ~5% increase in hit rate at baseline.
# Medium (0.50): +1 SD agency => ~12% increase — an appreciable relationship.
# No prior literature provides a point estimate; treat as exploratory.
AGENCY_SLOPE_SMALL  = 0.20
AGENCY_SLOPE_MEDIUM = 0.50


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def cohens_d_to_dprime_diff(cohens_d):
    """
    Convert Cohen's d to a raw d-prime difference.
    Cohen's d = delta_dprime / SD_pooled  =>  delta_dprime = Cohen's d x SD_pooled
    """
    return cohens_d * DPRIME_SD


def cohens_d_to_hr_diff(cohens_d):
    """
    Convert Cohen's d to a raw hit rate difference.
    Cohen's d = delta_hit_rate / SD  =>  delta_hit_rate = Cohen's d x SD
    """
    return cohens_d * HIT_RATE_SD


def hr_to_log_odds(hr):
    """
    Convert a probability (hit rate) to log odds.
    log_odds = log(p / (1 - p))
    Clipped to avoid log(0) or log(inf).
    """
    hr = np.clip(hr, 0.001, 0.999)
    return np.log(hr / (1 - hr))


# ==============================================================================
# ANALYSIS 1: Paired t-test on d-prime
# ==============================================================================

def power_dprime_ttest(n, effect_d, n_sim=N_SIMULATIONS):
    """
    Simulate power for a paired t-test comparing d-prime between
    high and low control conditions.

    This is the standard aggregated SDT approach (Schreiner et al., 2024).
    D-prime is computed per participant per condition, then compared with
    a two-tailed paired t-test.

    Data generating process:
      - Each participant has a d-prime value drawn from N(mean_condition, SD)
        for each condition, reflecting natural between-person variability.
      - The two condition means differ by the specified effect size.

    Parameters
    ----------
    n        : int   -- number of participants
    effect_d : float -- Cohen's d for the d-prime difference
    n_sim    : int   -- number of Monte Carlo simulations
    """
    dprime_diff = cohens_d_to_dprime_diff(effect_d)
    mean_high   = BASELINE_DPRIME + dprime_diff / 2
    mean_low    = BASELINE_DPRIME - dprime_diff / 2

    significant = 0
    for _ in range(n_sim):
        dp_high = np.random.normal(mean_high, DPRIME_SD, n)
        dp_low  = np.random.normal(mean_low,  DPRIME_SD, n)
        _, p    = stats.ttest_rel(dp_high, dp_low)
        if p < ALPHA:
            significant += 1

    return significant / n_sim


# ==============================================================================
# ANALYSIS 2: Paired t-test on hit rates
# ==============================================================================

def power_hitrate_ttest(n, effect_d, n_sim=N_SIMULATIONS):
    """
    Simulate power for a paired t-test comparing mean hit rates between
    high and low control conditions.

    This mirrors Wu et al. (2025) primary analysis and the supplemental
    analysis in Schreiner et al. (2024). Hit rate does not account for
    false alarms, but since foils have no condition structure in this
    design, it is a clean and interpretable measure.

    Data generating process:
      - Each participant has a mean hit rate drawn from N(mean_condition, SD).
    """
    hr_diff  = cohens_d_to_hr_diff(effect_d)
    hr_high  = np.clip(BASELINE_HIT_RATE + hr_diff / 2, 0.01, 0.99)
    hr_low   = np.clip(BASELINE_HIT_RATE - hr_diff / 2, 0.01, 0.99)

    significant = 0
    for _ in range(n_sim):
        hrs_high = np.random.normal(hr_high, HIT_RATE_SD, n)
        hrs_low  = np.random.normal(hr_low,  HIT_RATE_SD, n)
        _, p     = stats.ttest_rel(hrs_high, hrs_low)
        if p < ALPHA:
            significant += 1

    return significant / n_sim


# ==============================================================================
# ANALYSIS 3: GEE on target trials only
# ==============================================================================

def power_gee_hits_only(n, effect_d, n_sim=N_SIMULATIONS):
    """
    Simulate power for a trial-level GEE on target (old) trials only.

    Model:
        recognized ~ control_level
        groups: participant
        family: Binomial, logit link
        working correlation: Exchangeable

    Where:
        recognized    : 1 = hit, 0 = miss
        control_level : contrast-coded (+0.5 = high control, -0.5 = low control)

    By analyzing only target trials we avoid the foil dummy-assignment
    problem entirely. The key test is whether control_level significantly
    predicts recognition — i.e., are hit rates higher in the high condition?

    The exchangeable working correlation assumes all trials within a participant
    are equally correlated. This is a reasonable approximation and GEE is
    robust even if this assumption is violated.

    Data generating process:
      1. Each participant gets a random intercept drawn from N(0, RI_SD),
         capturing their overall tendency to recognize items.
      2. Trial-level hit/miss responses are drawn from Bernoulli distributions
         with probabilities: p = expit(intercept + ri_i + slope x control)
    """
    hr_diff    = cohens_d_to_hr_diff(effect_d)
    hr_high    = np.clip(BASELINE_HIT_RATE + hr_diff / 2, 0.01, 0.99)
    hr_low     = np.clip(BASELINE_HIT_RATE - hr_diff / 2, 0.01, 0.99)
    lo_high    = hr_to_log_odds(hr_high)
    lo_low     = hr_to_log_odds(hr_low)
    true_slope = lo_high - lo_low           # true effect of control in log-odds
    intercept  = (lo_high + lo_low) / 2    # grand mean log-odds

    # Pre-build trial structure — identical for every simulation
    n_total      = N_TRIALS_PER_CONDITION * 2
    participants = np.repeat(np.arange(n), n_total)
    controls     = np.tile(
        np.concatenate([
            np.repeat( 0.5, N_TRIALS_PER_CONDITION),   # high control trials
            np.repeat(-0.5, N_TRIALS_PER_CONDITION)    # low control trials
        ]), n
    )

    significant = 0
    for _ in range(n_sim):
        ri     = np.random.normal(0, RI_SD, n)
        ri_rep = np.repeat(ri, n_total)

        lo         = intercept + ri_rep + true_slope * controls
        probs      = expit(lo)
        recognized = np.random.binomial(1, probs)

        df_sim = pd.DataFrame({
            'participant': participants,
            'control':     controls,
            'recognized':  recognized
        })

        try:
            gee = smf.gee(
                "recognized ~ control",
                groups="participant",
                data=df_sim,
                family=sm.families.Binomial(),
                cov_struct=sm.cov_struct.Exchangeable()
            ).fit()

            if gee.pvalues['control'] < ALPHA:
                significant += 1

        except Exception:
            # Convergence failure — conservative choice is to not count as significant
            pass

    return significant / n_sim


# ==============================================================================
# ANALYSIS 4: GEE with item_type x control interaction
# ==============================================================================

def power_gee_interaction(n, effect_d, n_sim=N_SIMULATIONS):
    """
    Simulate power for a trial-level GEE including both target and foil
    trials, with an item_type x control interaction term.

    Model:
        said_old ~ item_type * control
        groups: participant
        family: Binomial, logit link
        working correlation: Exchangeable

    Where:
        said_old  : 1 = responded 'old', 0 = responded 'new'
        item_type : contrast-coded (+0.5 = target, -0.5 = foil)
        control   : contrast-coded (+0.5 = high,   -0.5 = low)

    WHY THE INTERACTION CAPTURES D-PRIME LOGIC:
      - Hit rate (targets) changes across control conditions
        => item_type x control interaction is non-zero
      - False alarm rate (foils) does NOT change across conditions
        => item_type x control interaction gets no contribution from foils
      - The interaction therefore selectively reflects the change in hit rate,
        which is equivalent to a change in d' (given constant FA rate)

    FOIL DUMMY ASSIGNMENT NOTE:
      Foils were never shown during the control task and have no genuine
      control condition. We split them equally (60 per dummy condition).
      Because the FA rate is identical for both dummy conditions by construction,
      this does not bias the interaction estimate — but it is an approximation.

    Data generating process:
      - Targets: hit rate differs between high and low control conditions
      - Foils: false alarm rate is identical regardless of dummy condition
      - Random intercepts capture between-participant variability
    """
    hr_diff  = cohens_d_to_hr_diff(effect_d)
    hr_high  = np.clip(BASELINE_HIT_RATE + hr_diff / 2, 0.01, 0.99)
    hr_low   = np.clip(BASELINE_HIT_RATE - hr_diff / 2, 0.01, 0.99)
    lo_high  = hr_to_log_odds(hr_high)
    lo_low   = hr_to_log_odds(hr_low)
    lo_fa    = hr_to_log_odds(BASELINE_FA_RATE)

    nf      = N_FOILS // 2
    n_total = N_TRIALS_PER_CONDITION * 2 + N_FOILS

    # Pre-build trial structure
    participants = np.repeat(np.arange(n), n_total)

    item_types = np.tile(np.concatenate([
        np.repeat( 0.5, N_TRIALS_PER_CONDITION),   # high control targets
        np.repeat( 0.5, N_TRIALS_PER_CONDITION),   # low control targets
        np.repeat(-0.5, nf),                        # foils (dummy: high)
        np.repeat(-0.5, nf)                         # foils (dummy: low)
    ]), n)

    controls = np.tile(np.concatenate([
        np.repeat( 0.5, N_TRIALS_PER_CONDITION),
        np.repeat(-0.5, N_TRIALS_PER_CONDITION),
        np.repeat( 0.5, nf),
        np.repeat(-0.5, nf)
    ]), n)

    # True log-odds per trial type in the data generating process
    base_lo = np.tile(np.concatenate([
        np.repeat(lo_high, N_TRIALS_PER_CONDITION),  # high targets: higher hit rate
        np.repeat(lo_low,  N_TRIALS_PER_CONDITION),  # low targets: lower hit rate
        np.repeat(lo_fa,   nf),                       # foils: FA rate (no condition effect)
        np.repeat(lo_fa,   nf)
    ]), n)

    significant = 0
    for _ in range(n_sim):
        ri       = np.random.normal(0, RI_SD, n)
        ri_rep   = np.repeat(ri, n_total)
        lo       = base_lo + ri_rep
        probs    = expit(lo)
        said_old = np.random.binomial(1, probs)

        df_sim = pd.DataFrame({
            'participant': participants,
            'item_type':   item_types,
            'control':     controls,
            'said_old':    said_old
        })

        try:
            gee = smf.gee(
                "said_old ~ item_type * control",
                groups="participant",
                data=df_sim,
                family=sm.families.Binomial(),
                cov_struct=sm.cov_struct.Exchangeable()
            ).fit()

            # item_type:control is the interaction term
            # Significant = hit rate effect of control differs from FA rate effect
            # = d-prime differs across conditions
            if gee.pvalues['item_type:control'] < ALPHA:
                significant += 1

        except Exception:
            pass

    return significant / n_sim


# ==============================================================================
# SUPPLEMENTARY ANALYSES
# ==============================================================================
# These analyses exploit the richer trial structure of the design:
# on each encoding trial, participants see TWO items simultaneously —
# one partially controlled (target item) and one moving randomly (uncontrolled
# item). Both are shown in the recognition test as "old" items alongside
# completely new foils.
#
# This gives FOUR old-item categories (2 x 2):
#   trial_level : High control / Low control   (which trial the item was on)
#   item_type   : Controlled   / Uncontrolled  (was the item under motor control?)
#
# The 2x2 design matches run_supp_analysis_1-4 in CDmem_analyses.py.
# The KEY TEST is the trial_level x item_type INTERACTION:
#   - Controlled items should show a trial_level effect (high > low memory)
#   - Uncontrolled items should show NO trial_level effect (no motor control)
#   => Significant interaction = motor control per se drives memory
#
# Data generating process (conservative):
#   High controlled:   BASELINE + effect/2  (e.g. higher d-prime)
#   Low  controlled:   BASELINE - effect/2
#   High uncontrolled: BASELINE              (no trial-level effect)
#   Low  uncontrolled: BASELINE
# ==============================================================================

N_UNCONTROLLED_PER_CONDITION = N_TRIALS_PER_CONDITION   # 60 per condition origin
N_UNCONTROLLED               = N_UNCONTROLLED_PER_CONDITION * 2   # 120 total


# ------------------------------------------------------------------------------
# Helper: simulate correlated 2x2 within-subjects d-prime data
# ------------------------------------------------------------------------------

def _sim_2x2_dprime(n, mean_hi_ctrl, mean_lo_ctrl, mean_hi_unc, mean_lo_unc):
    """
    Draw n participants' 2x2 d-prime values using a random-intercept
    decomposition that produces the within-participant correlation
    WITHIN_PERSON_CORR between any two conditions.

    Total SD per cell = DPRIME_SD.
    Between-participant SD = DPRIME_SD * sqrt(WITHIN_PERSON_CORR).
    Within-participant (residual) SD = DPRIME_SD * sqrt(1 - WITHIN_PERSON_CORR).

    Returns four arrays of length n (one per cell).
    """
    between_sd = DPRIME_SD * np.sqrt(WITHIN_PERSON_CORR)
    within_sd  = DPRIME_SD * np.sqrt(1.0 - WITHIN_PERSON_CORR)
    ri = np.random.normal(0, between_sd, n)   # shared random intercept
    dp_hi_ctrl = mean_hi_ctrl + ri + np.random.normal(0, within_sd, n)
    dp_lo_ctrl = mean_lo_ctrl + ri + np.random.normal(0, within_sd, n)
    dp_hi_unc  = mean_hi_unc  + ri + np.random.normal(0, within_sd, n)
    dp_lo_unc  = mean_lo_unc  + ri + np.random.normal(0, within_sd, n)
    return dp_hi_ctrl, dp_lo_ctrl, dp_hi_unc, dp_lo_unc


def _sim_2x2_hitrate(n, mean_hi_ctrl, mean_lo_ctrl, mean_hi_unc, mean_lo_unc):
    """Same as _sim_2x2_dprime but uses HIT_RATE_SD."""
    between_sd = HIT_RATE_SD * np.sqrt(WITHIN_PERSON_CORR)
    within_sd  = HIT_RATE_SD * np.sqrt(1.0 - WITHIN_PERSON_CORR)
    ri = np.random.normal(0, between_sd, n)
    hr_hi_ctrl = mean_hi_ctrl + ri + np.random.normal(0, within_sd, n)
    hr_lo_ctrl = mean_lo_ctrl + ri + np.random.normal(0, within_sd, n)
    hr_hi_unc  = mean_hi_unc  + ri + np.random.normal(0, within_sd, n)
    hr_lo_unc  = mean_lo_unc  + ri + np.random.normal(0, within_sd, n)
    return hr_hi_ctrl, hr_lo_ctrl, hr_hi_unc, hr_lo_unc


# ------------------------------------------------------------------------------
# Supplementary Analysis 1: 2x2 RM ANOVA on d-prime
# (Trial Level x Item Type; mirrors run_supp_analysis_1_dprime_2x2_anova)
# ------------------------------------------------------------------------------

def power_supp_dprime_2x2_anova(n, effect_d, n_sim=N_SIMULATIONS):
    """
    Simulate power for a 2x2 repeated-measures ANOVA on d-prime.

    Factors:
      trial_level : High vs. Low  (which encoding trial the item appeared on)
      item_type   : Controlled vs. Uncontrolled

    Key test: trial_level x item_type INTERACTION.
      The effect of interest is that controlled items show a trial_level
      effect (high>low) while uncontrolled items do not.
      A significant interaction therefore indicates that motor control per se
      (not mere co-presence on a control trial) drives memory.

    Data generating process (conservative):
      High, Controlled  : BASELINE_DPRIME + dprime_diff/2
      Low,  Controlled  : BASELINE_DPRIME - dprime_diff/2
      High, Uncontrolled: BASELINE_DPRIME  (no trial-level effect)
      Low,  Uncontrolled: BASELINE_DPRIME  (no trial-level effect)

    Correlation across the 4 conditions within each participant is modelled
    via a shared random intercept (rho = WITHIN_PERSON_CORR = 0.50).
    Uses statsmodels.AnovaRM — the same estimator as CDmem_analyses.py.
    """
    dprime_diff  = cohens_d_to_dprime_diff(effect_d)
    mean_hi_ctrl = BASELINE_DPRIME + dprime_diff / 2
    mean_lo_ctrl = BASELINE_DPRIME - dprime_diff / 2
    mean_hi_unc  = BASELINE_DPRIME   # no trial-level effect for uncontrolled
    mean_lo_unc  = BASELINE_DPRIME

    significant = 0
    subj_ids = np.arange(n)

    for _ in range(n_sim):
        dp_hc, dp_lc, dp_hu, dp_lu = _sim_2x2_dprime(
            n, mean_hi_ctrl, mean_lo_ctrl, mean_hi_unc, mean_lo_unc
        )
        # Long-format DataFrame required by AnovaRM
        df_sim = pd.DataFrame({
            'participant': np.tile(subj_ids, 4),
            'trial_level': np.repeat(['high', 'high', 'low', 'low'], n),
            'item_type':   np.repeat(['controlled', 'uncontrolled',
                                      'controlled', 'uncontrolled'], n),
            'd_prime':     np.concatenate([dp_hc, dp_hu, dp_lc, dp_lu])
        })
        try:
            res = AnovaRM(
                data=df_sim, depvar='d_prime',
                subject='participant',
                within=['trial_level', 'item_type']
            ).fit()
            tbl = res.anova_table
            # Key test: interaction
            int_key = [k for k in tbl.index if ':' in str(k)]
            if int_key and tbl.loc[int_key[0], 'Pr > F'] < ALPHA:
                significant += 1
        except Exception:
            pass

    return significant / n_sim


# ------------------------------------------------------------------------------
# Supplementary Analysis 2: 2x2 RM ANOVA on hit rates
# (Trial Level x Item Type; mirrors run_supp_analysis_2_hitrate_2x2_anova)
# ------------------------------------------------------------------------------

def power_supp_hitrate_2x2_anova(n, effect_d, n_sim=N_SIMULATIONS):
    """
    Simulate power for a 2x2 repeated-measures ANOVA on hit rates.

    Same 2x2 design and key test (interaction) as Supp Analysis 1.
    Hit rates are bounded [0,1]; the linear ANOVA is an approximation,
    matching the same caveat in CDmem_analyses.py.

    Data generating process:
      High, Controlled  : BASELINE_HIT_RATE + hr_diff/2
      Low,  Controlled  : BASELINE_HIT_RATE - hr_diff/2
      High, Uncontrolled: BASELINE_HIT_RATE
      Low,  Uncontrolled: BASELINE_HIT_RATE
    """
    hr_diff      = cohens_d_to_hr_diff(effect_d)
    mean_hi_ctrl = np.clip(BASELINE_HIT_RATE + hr_diff / 2, 0.01, 0.99)
    mean_lo_ctrl = np.clip(BASELINE_HIT_RATE - hr_diff / 2, 0.01, 0.99)
    mean_hi_unc  = BASELINE_HIT_RATE
    mean_lo_unc  = BASELINE_HIT_RATE

    significant = 0
    subj_ids = np.arange(n)

    for _ in range(n_sim):
        hr_hc, hr_lc, hr_hu, hr_lu = _sim_2x2_hitrate(
            n, mean_hi_ctrl, mean_lo_ctrl, mean_hi_unc, mean_lo_unc
        )
        df_sim = pd.DataFrame({
            'participant': np.tile(subj_ids, 4),
            'trial_level': np.repeat(['high', 'high', 'low', 'low'], n),
            'item_type':   np.repeat(['controlled', 'uncontrolled',
                                      'controlled', 'uncontrolled'], n),
            'hit_rate':    np.concatenate([hr_hc, hr_hu, hr_lc, hr_lu])
        })
        try:
            res = AnovaRM(
                data=df_sim, depvar='hit_rate',
                subject='participant',
                within=['trial_level', 'item_type']
            ).fit()
            tbl = res.anova_table
            int_key = [k for k in tbl.index if ':' in str(k)]
            if int_key and tbl.loc[int_key[0], 'Pr > F'] < ALPHA:
                significant += 1
        except Exception:
            pass

    return significant / n_sim


# ------------------------------------------------------------------------------
# Supplementary Analysis 3: 2x2 GEE on all old items
# (mirrors run_supp_analysis_3_glmm_2x2 in CDmem_analyses.py)
# ------------------------------------------------------------------------------

def power_supp_gee_2x2_old_items(n, effect_d, n_sim=N_SIMULATIONS):
    """
    Simulate power for a 2x2 trial-level GEE on all old items.

    Model:
        recognized ~ trial_level_c * item_type_c
        groups: participant
        family: Binomial, logit link
        working correlation: Exchangeable

    Contrast coding:
        trial_level_c : High = +0.5, Low = -0.5
        item_type_c   : Controlled = +0.5, Uncontrolled = -0.5

    Key test: trial_level_c:item_type_c INTERACTION

    With contrast (±0.5) coding, the interaction coefficient in log-odds equals:
        β_int = lo(Hi,Ctrl) - lo(Lo,Ctrl) - lo(Hi,Unc) + lo(Lo,Unc)
              = (lo_high - lo_low) - 0  =  true_slope_controlled
    because uncontrolled items have identical log-odds regardless of trial level.

    240 old trials per participant: 4 cells x 60 trials.
    """
    hr_diff   = cohens_d_to_hr_diff(effect_d)
    hr_high   = np.clip(BASELINE_HIT_RATE + hr_diff / 2, 0.01, 0.99)
    hr_low    = np.clip(BASELINE_HIT_RATE - hr_diff / 2, 0.01, 0.99)
    lo_high   = hr_to_log_odds(hr_high)
    lo_low    = hr_to_log_odds(hr_low)
    lo_base   = (lo_high + lo_low) / 2   # uncontrolled: grand mean, no trial effect

    # Trial structure per participant (240 old items)
    n_total = N_TRIALS_PER_CONDITION * 2 + N_UNCONTROLLED   # 60+60+60+60
    participants = np.repeat(np.arange(n), n_total)

    # trial_level_c: High=+0.5, Low=-0.5
    trial_level_c = np.tile(np.concatenate([
        np.repeat( 0.5, N_TRIALS_PER_CONDITION),          # High controlled
        np.repeat(-0.5, N_TRIALS_PER_CONDITION),          # Low  controlled
        np.repeat( 0.5, N_UNCONTROLLED_PER_CONDITION),    # High uncontrolled
        np.repeat(-0.5, N_UNCONTROLLED_PER_CONDITION)     # Low  uncontrolled
    ]), n)

    # item_type_c: Controlled=+0.5, Uncontrolled=-0.5
    item_type_c = np.tile(np.concatenate([
        np.repeat( 0.5, N_TRIALS_PER_CONDITION),           # controlled
        np.repeat( 0.5, N_TRIALS_PER_CONDITION),
        np.repeat(-0.5, N_UNCONTROLLED_PER_CONDITION),     # uncontrolled
        np.repeat(-0.5, N_UNCONTROLLED_PER_CONDITION)
    ]), n)

    # True log-odds per cell (uncontrolled items have no trial-level effect)
    base_lo = np.tile(np.concatenate([
        np.repeat(lo_high, N_TRIALS_PER_CONDITION),
        np.repeat(lo_low,  N_TRIALS_PER_CONDITION),
        np.repeat(lo_base, N_UNCONTROLLED_PER_CONDITION),
        np.repeat(lo_base, N_UNCONTROLLED_PER_CONDITION)
    ]), n)

    significant = 0
    for _ in range(n_sim):
        ri     = np.random.normal(0, RI_SD, n)
        ri_rep = np.repeat(ri, n_total)
        lo     = base_lo + ri_rep
        probs  = expit(lo)
        recognized = np.random.binomial(1, probs)

        df_sim = pd.DataFrame({
            'participant':   participants,
            'trial_level_c': trial_level_c,
            'item_type_c':   item_type_c,
            'recognized':    recognized
        })

        try:
            gee = smf.gee(
                "recognized ~ trial_level_c * item_type_c",
                groups="participant",
                data=df_sim,
                family=sm.families.Binomial(),
                cov_struct=sm.cov_struct.Exchangeable()
            ).fit()
            if gee.pvalues['trial_level_c:item_type_c'] < ALPHA:
                significant += 1
        except Exception:
            pass

    return significant / n_sim


# ------------------------------------------------------------------------------
# Supplementary Analysis 4: 3-way GEE on all trials
# (mirrors run_supp_analysis_4_glmm_foils in CDmem_analyses.py)
# ------------------------------------------------------------------------------

def power_supp_gee_3way_all_trials(n, effect_d, n_sim=N_SIMULATIONS):
    """
    Simulate power for a 3-way trial-level GEE on ALL trials (old + foils).

    Model:
        said_old ~ item_is_old_c * trial_level_c * item_type_c
        groups: participant
        family: Binomial, logit link
        working correlation: Exchangeable

    Contrast coding:
        item_is_old_c : Old = +0.5, Foil = -0.5
        trial_level_c : High = +0.5, Low = -0.5
        item_type_c   : Controlled = +0.5, Uncontrolled = -0.5

    Key test: 3-way interaction item_is_old_c:trial_level_c:item_type_c
      This tests whether the 2x2 pattern in old items (trial_level x item_type)
      is different from the 2x2 pattern in new foils.
      For foils the FA rate does not vary with any factor, so if old items
      show the 2x2 interaction, the 3-way is significant.

    Foil dummy assignment: 240 foils split into 4 balanced cells of 60
    (matching the rotating scheme in run_supp_analysis_4_glmm_foils).
    All foil cells share the same FA rate, so the dummy assignment does not
    bias the 3-way interaction estimate.

    Uncontrolled items: same mean log-odds for High and Low trial origin —
    no trial-level effect on memory for items never under participant control.
    """
    hr_diff   = cohens_d_to_hr_diff(effect_d)
    hr_high   = np.clip(BASELINE_HIT_RATE + hr_diff / 2, 0.01, 0.99)
    hr_low    = np.clip(BASELINE_HIT_RATE - hr_diff / 2, 0.01, 0.99)
    lo_high   = hr_to_log_odds(hr_high)
    lo_low    = hr_to_log_odds(hr_low)
    lo_base   = (lo_high + lo_low) / 2   # uncontrolled: no trial-level effect
    lo_fa     = hr_to_log_odds(BASELINE_FA_RATE)

    nf4       = N_FOILS // 4    # 60 foils per dummy 2x2 cell
    n_total   = N_TRIALS_PER_CONDITION * 2 + N_UNCONTROLLED + N_FOILS  # 480

    participants = np.repeat(np.arange(n), n_total)

    # item_is_old_c: Old=+0.5, Foil=-0.5
    item_is_old_c = np.tile(np.concatenate([
        np.repeat( 0.5, N_TRIALS_PER_CONDITION),         # High controlled old
        np.repeat( 0.5, N_TRIALS_PER_CONDITION),         # Low  controlled old
        np.repeat( 0.5, N_UNCONTROLLED_PER_CONDITION),   # High uncontrolled old
        np.repeat( 0.5, N_UNCONTROLLED_PER_CONDITION),   # Low  uncontrolled old
        np.repeat(-0.5, N_FOILS)                          # Foils
    ]), n)

    # trial_level_c: High=+0.5, Low=-0.5  (foils balanced: 2 cells high, 2 low)
    trial_level_c = np.tile(np.concatenate([
        np.repeat( 0.5, N_TRIALS_PER_CONDITION),
        np.repeat(-0.5, N_TRIALS_PER_CONDITION),
        np.repeat( 0.5, N_UNCONTROLLED_PER_CONDITION),
        np.repeat(-0.5, N_UNCONTROLLED_PER_CONDITION),
        np.repeat( 0.5, nf4), np.repeat( 0.5, nf4),     # foil dummy: 2 high cells
        np.repeat(-0.5, nf4), np.repeat(-0.5, nf4)      # foil dummy: 2 low cells
    ]), n)

    # item_type_c: Controlled=+0.5, Uncontrolled=-0.5  (foils balanced)
    item_type_c = np.tile(np.concatenate([
        np.repeat( 0.5, N_TRIALS_PER_CONDITION),          # controlled
        np.repeat( 0.5, N_TRIALS_PER_CONDITION),
        np.repeat(-0.5, N_UNCONTROLLED_PER_CONDITION),    # uncontrolled
        np.repeat(-0.5, N_UNCONTROLLED_PER_CONDITION),
        np.repeat( 0.5, nf4), np.repeat(-0.5, nf4),      # foil dummy: ctrl, unctrl
        np.repeat( 0.5, nf4), np.repeat(-0.5, nf4)       # foil dummy: ctrl, unctrl
    ]), n)

    # True log-odds per trial group
    base_lo = np.tile(np.concatenate([
        np.repeat(lo_high, N_TRIALS_PER_CONDITION),
        np.repeat(lo_low,  N_TRIALS_PER_CONDITION),
        np.repeat(lo_base, N_UNCONTROLLED_PER_CONDITION),
        np.repeat(lo_base, N_UNCONTROLLED_PER_CONDITION),
        np.repeat(lo_fa,   N_FOILS)                       # all foils same FA rate
    ]), n)

    significant = 0
    for _ in range(n_sim):
        ri     = np.random.normal(0, RI_SD, n)
        ri_rep = np.repeat(ri, n_total)
        lo     = base_lo + ri_rep
        probs  = expit(lo)
        said_old = np.random.binomial(1, probs)

        df_sim = pd.DataFrame({
            'participant':   participants,
            'item_is_old_c': item_is_old_c,
            'trial_level_c': trial_level_c,
            'item_type_c':   item_type_c,
            'said_old':      said_old
        })

        try:
            gee = smf.gee(
                "said_old ~ item_is_old_c * trial_level_c * item_type_c",
                groups="participant",
                data=df_sim,
                family=sm.families.Binomial(),
                cov_struct=sm.cov_struct.Exchangeable()
            ).fit()
            # 3-way interaction is the key test
            if gee.pvalues['item_is_old_c:trial_level_c:item_type_c'] < ALPHA:
                significant += 1
        except Exception:
            pass

    return significant / n_sim


# ==============================================================================
# ANALYSIS 7: Agency Rating -> Recognition Memory
# (mirrors run_analysis_7_agency_glmm in CDmem_analyses.py)
# ==============================================================================

def power_agency_memory_gee(n, agency_slope, n_sim=N_SIMULATIONS):
    """
    Simulate power for a trial-level GEE testing whether within-participant
    variation in agency_rating predicts subsequent recognition memory.

    Model:
        recognized ~ agency_rating_z
        groups: participant
        family: Binomial, logit link
        working correlation: Exchangeable

    Predictor: agency_rating_z — agency rating z-scored within each participant.
    Within-participant z-scoring is done in the actual analysis; here it is
    implicit because the simulated predictor already has mean=0, SD=1 per
    participant.

    Effect size parameterisation:
        agency_slope = change in log-odds per 1 SD increase in agency rating.
        Small  (AGENCY_SLOPE_SMALL  = 0.20): ~+5%  hit rate at baseline.
        Medium (AGENCY_SLOPE_MEDIUM = 0.50): ~+12% hit rate at baseline.
        No published estimate exists for this paradigm; Analysis 7 is
        exploratory and should be labelled as such.

    Data generating process:
      1. Each participant has n_total target trials (120: 60 high + 60 low).
         Agency rating z-scores are drawn i.i.d. N(0,1) per trial.
      2. Hit probability: p = expit(intercept + ri_i + slope * agency_z)
      3. GEE tests whether slope != 0.

    Parameters
    ----------
    n            : int   -- number of participants
    agency_slope : float -- log-odds per 1 SD of agency (AGENCY_SLOPE_*)
    n_sim        : int   -- number of Monte Carlo simulations
    """
    lo_baseline = hr_to_log_odds(BASELINE_HIT_RATE)
    n_total     = N_TRIALS_PER_CONDITION * 2   # 60 high + 60 low target trials
    participants = np.repeat(np.arange(n), n_total)

    significant = 0
    for _ in range(n_sim):
        ri     = np.random.normal(0, RI_SD, n)
        ri_rep = np.repeat(ri, n_total)
        # Agency z-scores: i.i.d. within participant (after z-scoring, SD=1)
        agency_z = np.random.normal(0, 1, n * n_total)
        lo       = lo_baseline + ri_rep + agency_slope * agency_z
        probs    = expit(lo)
        recognized = np.random.binomial(1, probs)

        df_sim = pd.DataFrame({
            'participant': participants,
            'agency_z':   agency_z,
            'recognized': recognized
        })

        try:
            gee = smf.gee(
                "recognized ~ agency_z",
                groups="participant",
                data=df_sim,
                family=sm.families.Binomial(),
                cov_struct=sm.cov_struct.Exchangeable()
            ).fit()
            if gee.pvalues['agency_z'] < ALPHA:
                significant += 1
        except Exception:
            pass

    return significant / n_sim


# ==============================================================================
# RUN ALL SIMULATIONS
# ==============================================================================

def run_all():
    """
    Run the full power analysis grid.

    Primary:       2 effect sizes x 4 analyses x 4 sample sizes = 32 cells
    Supplementary: 2 effect sizes x 4 analyses x 4 sample sizes = 32 cells
    Analysis 7:    2 agency slopes  x 4 sample sizes             =  8 cells
    Total: 72 cells.

    All analyses mirror the corresponding functions in CDmem_analyses.py.
    """
    effect_sizes = {
        'Small (d=0.2)':  0.2,
        'Medium (d=0.5)': 0.5
    }

    # Primary analyses: high vs low control, controlled targets only
    primary_analyses = {
        'Analysis 1: d-prime paired t-test':  power_dprime_ttest,
        'Analysis 2: Hit rate paired t-test': power_hitrate_ttest,
        'Analysis 3: GEE hits only':          power_gee_hits_only,
        'Analysis 4: GEE item_type x control': power_gee_interaction,
    }

    # Supplementary: 2x2 factorial (Trial Level x Item Type)
    supp_analyses = {
        'Supp 1: d-prime 2x2 RM ANOVA (interaction)':    power_supp_dprime_2x2_anova,
        'Supp 2: Hit rate 2x2 RM ANOVA (interaction)':   power_supp_hitrate_2x2_anova,
        'Supp 3: GEE 2x2 old items (interaction)':        power_supp_gee_2x2_old_items,
        'Supp 4: GEE 3-way all trials (3-way interaction)': power_supp_gee_3way_all_trials,
    }

    # Analysis 7: agency -> memory (exploratory; effect in log-odds/SD)
    agency_effect_sizes = {
        f'Small (b={AGENCY_SLOPE_SMALL})':  AGENCY_SLOPE_SMALL,
        f'Medium (b={AGENCY_SLOPE_MEDIUM})': AGENCY_SLOPE_MEDIUM,
    }

    results = []

    # --- Primary and Supplementary ---
    for section_label, analyses in [('Primary', primary_analyses),
                                     ('Supplementary', supp_analyses)]:
        total = len(effect_sizes) * len(analyses) * len(SAMPLE_SIZES)
        done  = 0
        print(f"\n{'='*65}")
        print(f"  {section_label} Analyses")
        print(f"{'='*65}")
        t_start = time.time()

        for effect_label, effect_d in effect_sizes.items():
            for analysis_label, func in analyses.items():
                for n in SAMPLE_SIZES:
                    done += 1
                    print(f"[{done}/{total}] {analysis_label} | {effect_label} | N={n}...",
                          flush=True)
                    t0    = time.time()
                    power = func(n, effect_d)
                    print(f"       -> Power = {power:.3f}  ({time.time()-t0:.1f}s)",
                          flush=True)
                    results.append({
                        'Section':     section_label,
                        'Effect Size': effect_label,
                        'Analysis':    analysis_label,
                        'N':           n,
                        'Power':       round(power, 3)
                    })

        print(f"  Section runtime: {(time.time()-t_start)/60:.1f} min")

    # --- Analysis 7: agency -> memory ---
    print(f"\n{'='*65}")
    print("  Analysis 7: Agency -> Memory (Exploratory)")
    print(f"{'='*65}")
    t_start = time.time()
    total7  = len(agency_effect_sizes) * len(SAMPLE_SIZES)
    done7   = 0

    for effect_label, slope in agency_effect_sizes.items():
        for n in SAMPLE_SIZES:
            done7 += 1
            lbl = f'Analysis 7: GEE agency_z -> recognized'
            print(f"[{done7}/{total7}] {lbl} | {effect_label} | N={n}...",
                  flush=True)
            t0    = time.time()
            power = power_agency_memory_gee(n, slope)
            print(f"       -> Power = {power:.3f}  ({time.time()-t0:.1f}s)",
                  flush=True)
            results.append({
                'Section':     'Analysis 7 (Exploratory)',
                'Effect Size': effect_label,
                'Analysis':    lbl,
                'N':           n,
                'Power':       round(power, 3)
            })

    print(f"  Section runtime: {(time.time()-t_start)/60:.1f} min")

    return pd.DataFrame(results)


# ==============================================================================
# SAVE RESULTS TO MARKDOWN
# ==============================================================================

def save_markdown(df, path='results_CDmem.md'):
    """
    Save power analysis results as a formatted markdown file.
    Primary, supplementary, and Analysis 7 results are in separate sections.
    [80%+] marks cells meeting the 80% power threshold.
    """
    lines = []
    lines.append("# Power Analysis Results\n")

    lines.append("## Parameters\n")
    lines.append(f"- Simulations per cell: {N_SIMULATIONS}")
    lines.append(f"- Alpha level: {ALPHA}")
    lines.append(f"- Target trials per condition: {N_TRIALS_PER_CONDITION}")
    lines.append(f"- Foil trials total: {N_FOILS}")
    lines.append(f"- Uncontrolled old items total: {N_UNCONTROLLED} (2 x {N_UNCONTROLLED_PER_CONDITION})")
    lines.append(f"- Baseline hit rate: {BASELINE_HIT_RATE} (Wu et al., 2025, Exp. 4)")
    lines.append(f"- Baseline false alarm rate: {BASELINE_FA_RATE} (assumed)")
    lines.append(f"- Baseline d-prime: {BASELINE_DPRIME} (avg of Schreiner et al. conditions)")
    lines.append(f"- D-prime pooled SD: {DPRIME_SD} (Schreiner et al.)")
    lines.append(f"- Hit rate SD: {HIT_RATE_SD} (Schreiner et al.)")
    lines.append(f"- Random intercept SD: {RI_SD} (converted from d-prime SD via log-odds)")
    lines.append(f"- Within-person correlation (RM ANOVA): {WITHIN_PERSON_CORR}")
    lines.append(f"- Agency slope small: {AGENCY_SLOPE_SMALL} log-odds/SD")
    lines.append(f"- Agency slope medium: {AGENCY_SLOPE_MEDIUM} log-odds/SD\n")

    lines.append("## Primary Analysis Descriptions\n")
    lines.append("Analyses 1-4 mirror run_analysis_1 to run_analysis_4 in CDmem_analyses.py.")
    lines.append("Effect: High > Low control on recognition of controlled (target) items.\n")
    lines.append("1. **d-prime paired t-test**: D-prime per participant per condition, two-tailed paired t-test (Schreiner et al., 2024).")
    lines.append("2. **Hit rate paired t-test**: Mean hit rate per participant per condition, paired t-test (Wu et al., 2025).")
    lines.append("3. **GEE hits only**: Trial-level GEE on target trials only. recognized ~ control_level (±0.5). Binomial/logit/exchangeable.")
    lines.append("4. **GEE item_type x control**: Trial-level GEE on all trials. said_old ~ item_type_c * control_c. Foils dummy-assigned equally (balanced).\n")

    lines.append("## Supplementary Analysis Descriptions\n")
    lines.append("Analyses Supp1-4 mirror run_supp_analysis_1 to run_supp_analysis_4 in CDmem_analyses.py.")
    lines.append("Design: 2x2 factorial — trial_level (High/Low) x item_type (Controlled/Uncontrolled).")
    lines.append("**Key test in all supplementary analyses: trial_level x item_type INTERACTION.**")
    lines.append("  Significant interaction => motor control per se (not co-presence) drives memory.")
    lines.append("**Conservative assumption**: Uncontrolled items have the grand mean hit rate (no trial-level effect).\n")
    lines.append("S1. **d-prime 2x2 RM ANOVA**: Four d-primes per participant (2x2 cells), statsmodels AnovaRM. Same estimator as CDmem_analyses.py.")
    lines.append("S2. **Hit rate 2x2 RM ANOVA**: Same 2x2 design on hit rates.")
    lines.append("S3. **GEE 2x2 old items**: recognized ~ trial_level_c * item_type_c on all 240 old trials. Interaction = motor control effect.")
    lines.append("S4. **GEE 3-way all trials**: said_old ~ item_is_old_c * trial_level_c * item_type_c. Foils dummy-assigned to balanced 2x2 cells. 3-way interaction is key test.\n")

    lines.append("## Analysis 7 Description (Exploratory)\n")
    lines.append("Mirrors run_analysis_7_agency_glmm in CDmem_analyses.py.")
    lines.append("Model: recognized ~ agency_rating_z (within-participant z-scored agency).")
    lines.append("Effect parameterised as log-odds slope per 1 SD of agency (no prior estimate; exploratory).")
    lines.append(f"Small slope: b={AGENCY_SLOPE_SMALL} (~+5% hit rate per SD). Medium: b={AGENCY_SLOPE_MEDIUM} (~+12% per SD).\n")

    lines.append("## Effect Size Conversions\n")
    for d in [0.2, 0.5]:
        dd = cohens_d_to_dprime_diff(d)
        hd = cohens_d_to_hr_diff(d)
        lines.append(
            f"- Cohen's d = {d}: "
            f"delta d-prime = {dd:.3f} ({BASELINE_DPRIME-dd/2:.3f} vs {BASELINE_DPRIME+dd/2:.3f}), "
            f"delta hit rate = {hd:.3f} ({BASELINE_HIT_RATE-hd/2:.3f} vs {BASELINE_HIT_RATE+hd/2:.3f})"
        )
    lines.append("")

    # Write results for each section
    for section in ['Primary', 'Supplementary', 'Analysis 7 (Exploratory)']:
        sec_label = {
            'Primary': 'Primary',
            'Supplementary': 'Supplementary',
            'Analysis 7 (Exploratory)': 'Analysis 7 (Exploratory)'
        }[section]
        lines.append(f"## {sec_label} Analysis Results\n")
        section_df = df[df['Section'] == section]
        if section_df.empty:
            lines.append("*(no results)*\n")
            continue

        for effect_label in section_df['Effect Size'].unique():
            lines.append(f"### {effect_label}\n")
            subset = section_df[section_df['Effect Size'] == effect_label].copy()
            pivot  = subset.pivot(index='Analysis', columns='N', values='Power')
            pivot.columns = [f'N={c}' for c in pivot.columns]
            lines.append('| Analysis | ' + ' | '.join(pivot.columns) + ' |')
            lines.append('|' + '---|' * (len(pivot.columns) + 1))
            for analysis, row in pivot.iterrows():
                vals = [f"{v:.3f}{' [80%+]' if v >= 0.80 else ''}" for v in row.values]
                lines.append(f"| {analysis} | " + " | ".join(vals) + " |")
            lines.append("\n[80%+] = meets 80% power threshold\n")

    lines.append("## Notes and Limitations\n")
    lines.append("- **GEE vs GLMM**: Power simulation uses GEE (statsmodels); confirmatory analyses use GLMM (pymer4/lme4). GEE power estimates are conservative due to non-collapsibility of the logit link (Skrondal & Rabe-Hesketh, 2004). Actual GLMM power >= estimated.")
    lines.append("- **GEE citation**: Rochon (1998) and Liu & Liang (1997) recommend GEE simulation for planning studies with correlated binary outcomes.")
    lines.append("- **RM ANOVA**: Supp 1-2 use AnovaRM (statsmodels), matching CDmem_analyses.py. Within-person correlation modelled via random intercept decomposition (rho=0.50).")
    lines.append("- **Random intercept SD**: Estimated from d-prime SD via log-odds conversion. Actual between-participant variability is unknown.")
    lines.append("- **False alarm rate**: Assumed 0.20. Affects only Analysis 4 (foil log-odds) and Supp 4 (foil dummy cells).")
    lines.append("- **Supplementary interaction**: Power reported for trial_level x item_type interaction (2x2). Conservative: uncontrolled items have no trial-level effect in DGP.")
    lines.append("- **Analysis 7**: Exploratory. No prior estimate for agency-memory slope; power reported to inform sample size planning only.")
    lines.append("- **Recommendation**: For small effects (d=0.2 or b=0.20), N=30-60 is likely underpowered. Frame as exploratory. For medium effects, GEE/GLMM trial-level analyses offer more power by leveraging within-participant trial variance.")

    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"\nResults saved to: {path}")


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("Power Analysis: 8 Analyses x 2 Effect Sizes x 4 Sample Sizes")
    print(f"  (4 primary + 4 supplementary)")
    print(f"Simulations per cell: {N_SIMULATIONS}")
    print("=" * 65 + "\n")

    df_results = run_all()

    print("\n" + "=" * 65)
    print("FINAL RESULTS")
    print("=" * 65)
    print(df_results.to_string(index=False))

    save_markdown(df_results, 'results_CDmem.md')