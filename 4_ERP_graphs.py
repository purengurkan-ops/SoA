import os
import sys
import numpy as np
import pandas as pd
import mne

# Force UTF-8 output so special characters print cleanly on Windows PowerShell
sys.stdout.reconfigure(encoding='utf-8')

# ──────────────────────────────────────────────────────────────
# Which participant(s) do you want to process? - REMEMBER TO REMOVE LOW TRIAL COUNT PARTICIPANTS 
# ──────────────────────────────────────────────────────────────
plist = [4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 19, 20]

# ──────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────
dfolder = os.path.join(os.getcwd(), "eeg4_ERPSummaries")

# Path for saving figures
save_to = os.path.join(os.getcwd(), "eeg5_figures")
os.makedirs(save_to, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# Electrode and time window selection for P500 (Wen et al., 2017)
# ──────────────────────────────────────────────────────────────

# electrodes_select = ['FCz', 'Cz', 'CPz', 'Pz']  # (Wen et al., 2017)
electrodes_select = ['Fz', 'FCz', 'FC1', 'FC2']  # (data-driven electrode selection - seems like positive voltage activity is more frontal in our case, following Giersiepen et al., 2024, 2025)

# Time window for P500 analysis and topoplots. 450-650 ms after movement onset (Wen et al., 2017)
time_select = [0.45, 0.65]

# ──────────────────────────────────────────────────────────────
# Condition layout (matches ERP_calculation.py output)
# ──────────────────────────────────────────────────────────────
num_cond = 2
cond_names = ['high_control', 'low_control']

# ──────────────────────────────────────────────────────────────
# Load data for all participants + build grand-average structures
# ─────────────────────────────────────────────────────────────
#
# In MNE, mne.read_evokeds() loads the FIF saved by ERP_calculation.py.
# Each file contains both Evoked objects (one per condition), identified
# by their .comment attribute (set to the condition label in 3_ERP_calculation.py).

# alleeg[p][cond_label] → Evoked for participant p, condition cond_label
alleeg       = {}   
all_summaries = []   

for p_idx, pnum in enumerate(plist):
    sub_id = f"{pnum:04d}"
    erp_file = os.path.join(dfolder, f"CDmem_{sub_id}-erp-ave.fif")
    csv_file  = os.path.join(dfolder, f"CDmem_{sub_id}-erp-summary.csv")

    if not os.path.exists(erp_file):
        print(f"  ERP file not found, skipping participant {pnum}: {erp_file}")
        continue

    print(f"Loading participant {pnum}")

    # Load all Evoked objects for this participant
    evoked_list = mne.read_evokeds(erp_file, verbose=False)

    # Index by condition label (stored in evoked.comment during 3_ERP_calculation.py)
    alleeg[p_idx] = {ev.comment: ev for ev in evoked_list}

    # Load trial-count summary
    if os.path.exists(csv_file):
        summary_row = pd.read_csv(csv_file).iloc[0].to_dict()
        all_summaries.append(summary_row)

    # ─────────────────────────────────────────────────────────────

# Print behavioral summary table across participants
if all_summaries:
    print("\nBehavioral summary (trial counts per condition):")
    print(pd.DataFrame(all_summaries).to_string(index=False))

# ──────────────────────────────────────────────────────────────
# Grand averages across participants
# ──────────────────────────────────────────────────────────────
# MNE: mne.grand_average() averages across a list of Evoked objects.
# It normalizes by number of trials under the hood

print("\nComputing grand averages...")
GA_dat = {}   

for cond_label in cond_names:
    # Collect this condition's Evoked across all loaded participants
    evokeds_this_cond = [alleeg[p][cond_label]
                         for p in alleeg if cond_label in alleeg[p]]
    if evokeds_this_cond:
        GA_dat[cond_label] = mne.grand_average(evokeds_this_cond)
        GA_dat[cond_label].comment = cond_label
        print(f"  GA [{cond_label}]: averaged across {len(evokeds_this_cond)} participant(s)")


print("\nGrand averages ready.")
print(f"  Electrodes selected : {electrodes_select}")
print(f"  Time window         : {time_select[0]:.2f} - {time_select[1]:.2f} s")

# ──────────────────────────────────────────────────────────────
# Average activity across electrodes per participant / condition
# ──────────────────────────────────────────────────────────────
# In MNE, evoked.data has shape [n_channels, n_timepoints] in Volts.
# We pick() the selected electrodes, then average across channels to get
# one time series per participant × condition.  Multiply by 1e6 → µV.

import matplotlib
matplotlib.use('TkAgg')   # interactive window; change to 'Agg' for headless/batch saving only
import matplotlib.pyplot as plt

loaded_plist = list(alleeg.keys()) # participant indices that were successfully loaded
n_participants = len(loaded_plist)

# Get shared time axis (same for all participants / conditions)
times = alleeg[loaded_plist[0]][cond_names[0]].times # in seconds

# Verify that all requested electrodes actually exist in the data
available_ch = alleeg[loaded_plist[0]][cond_names[0]].ch_names
picked_channels = [ch for ch in electrodes_select if ch in available_ch]
missing = [ch for ch in electrodes_select if ch not in available_ch]
if missing:
    print(f"  WARNING: electrode(s) not found in data and will be skipped: {missing}")
print(f"  Using electrodes: {picked_channels}")

n_timepoints = len(times)

# ──  pMeanList ──────────────────────────────
pMeanList = np.zeros((n_participants, num_cond, n_timepoints))   # [P, C, T]

for p_idx, p_key in enumerate(loaded_plist):
    for cond_idx, cond_label in enumerate(cond_names):
        # Pick selected electrodes and convert to µV
        evoked = alleeg[p_key][cond_label].copy().pick(picked_channels)
        data_uv = evoked.data * 1e6                        # [n_channels, n_timepoints]
        pMeanList[p_idx, cond_idx, :] = np.mean(data_uv, axis=0)   # average over electrodes

# ── Compute grandMean / subjMean for error-bar correction ──────

# grandMean is used only in the Cousineau-Morey within-subject error correction
# (see plotting loops below). For single-participant plots there are no error bars.

if n_participants == 1:
    grandMean = np.squeeze(np.mean(pMeanList, axis=1))   # [T]
    subjMean  = grandMean                                 # unused for 1 participant
else:
    subjMean  = np.squeeze(np.mean(pMeanList, axis=1))   # [P, T] -- mean over conditions
    grandMean = np.mean(subjMean, axis=0)                 # [T]    -- mean over participants

# ──────────────────────────────────────────────────────────────
# Plot settings
# ──────────────────────────────────────────────────────────────

ylimits  = [-3, 3]   # µV y-axis range

# Line colors per condition [blue for high, purple for low]
colors      = [[0, 0.44, 0.69], [0.8, 0.47, 0.65]]
linestyles  = ['-', '-']   
linecol_colors = ['blue', 'magenta']

# Figure dimensions
fig_w_cm, fig_h_cm = 30, 20
fig_w_in = fig_w_cm / 2.54
fig_h_in = fig_h_cm / 2.54
font_size = 20   

plt.close('all')

# ──────────────────────────────────────────────────────────────
# MAIN FIGURE
# ──────────────────────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(fig_w_in, fig_h_in))

H1 = []

# ──Main Effect of Control: 2 conditions ───────────────────
labels = ['High Control', 'Low Control']
for cond_idx, cond_label in enumerate(cond_names):
    y = pMeanList[:, cond_idx, :]   # [P, T]

    if n_participants == 1:
        h, = ax1.plot(times, y[0, :],
                      linestyle='-', color=linecol_colors[cond_idx])
    else:
        # Cousineau-Morey correction (factor for 2 conditions is 2/1 = 2.0)
        yCorrect = y - subjMean
        yCorrect = (yCorrect + grandMean) * (2.0)
        errbar   = np.std(yCorrect, axis=0, ddof=0) / np.sqrt(n_participants)

        h, = ax1.plot(times, np.mean(y, axis=0),
                      linestyle=linestyles[cond_idx],
                      color=colors[cond_idx])

    H1.append(h)
save_name = '00_lineplot_main_effects.svg'


# ── Formatting ────────────────────────────────────────────────


ax1.legend(H1, labels, loc='best', fontsize=font_size)
ax1.set_ylim(ylimits)
ax1.invert_yaxis()           # EEG convention: negative up
ax1.axhline(0, color='black', linestyle='--', linewidth=0.5)
ax1.axvline(0, color='black', linestyle='--', linewidth=0.5)

ax1.set_xlabel('Time (s)', fontsize=font_size)
ax1.set_ylabel('µV', fontsize=font_size)
ax1.set_xticks(np.arange(0, times[-1] + 0.01, 0.1)) 
ax1.tick_params(labelsize=font_size, width=1)
for spine in ax1.spines.values():
    spine.set_linewidth(1)
try:
    plt.rcParams['font.family'] = 'Times New Roman'
except Exception:
    pass   # fall back to matplotlib default if font not installed

fig1.patch.set_facecolor('white')
ax1.set_facecolor('white')

# Title hidden 
ax1.set_title('')

plt.tight_layout()

# ── Save figure ───────────────────────────────────────────────
# FieldTrip: print(gcf, '-dsvg', [save_to '00_lineplot_...'], dpi)
# dpi=600 matches FieldTrip's -r600; SVG is vector so DPI mainly affects rasterised elements.
save_path = os.path.join(save_to, save_name)
fig1.savefig(save_path, format='svg', dpi=600, bbox_inches='tight',
             facecolor='white')
print(f"\n  Figure saved: {save_path}")

# ──────────────────────────────────────────────────────────────
# Topographical Plots
# ──────────────────────────────────────────────────────────────
#
# In MNE, we use the `mask` parameter of plot_topomap() to highlight
# the selected electrodes (equivalent to cfg.highlightchannel).
# MNE handles the 2D projection internally so positions always match.

# matplotlib's 'RdBu_r' is the reversed version (blue for negative, red for positive).
topo_cmap = plt.cm.RdBu_r

# Color limits matching FieldTrip: cfg.zlim = [-1.5 1.5]   (µV)
# Adjust these if needed for visibility.
# topo_vlim = (-1.5, 1.5)   # in µV — will be converted to Volts for MNE
topo_vlim = (-3, 3)   # seems like we have positive activity more than 1.5 µV in a condition, so changed this way. 


# Figure size: FieldTrip PaperPosition [0 0 8 8] (cm) → 8×8 cm per topo
topo_fig_in = 8 / 2.54   # ~3.15 inches

# Time window for averaging: FieldTrip cfg.xlim = time_select
t_min, t_max = time_select

# Build a boolean mask for the highlighted electrodes.
# MNE: mask = boolean array [n_channels], True = highlight with mask_params style.
def make_highlight_mask(evoked, highlight_names):
    """Return a boolean array (n_channels,) — True for channels to highlight."""
    mask = np.array([ch in highlight_names for ch in evoked.ch_names])
    return mask

# Style for highlighted channels
mask_params = dict(marker='*', markerfacecolor='black', markeredgecolor='black',
                   markersize=10, zorder=10)

# ──Main Effect of Condition: one topo per condition ──────────
topo_labels = ['High Control', 'Low Control']
topo_prefix = '00_topo_main_effects'

for cond_idx, cond_label in enumerate(cond_names):
    evoked_topo = GA_dat[cond_label].copy()
    evoked_topo.crop(tmin=t_min, tmax=t_max)
    topo_data = evoked_topo.data.mean(axis=1)   # [n_channels] in Volts

    fig_topo, ax_topo = plt.subplots(figsize=(topo_fig_in, topo_fig_in))

    highlight_mask = make_highlight_mask(evoked_topo, picked_channels)
    mne.viz.plot_topomap(
        topo_data, evoked_topo.info,
        axes=ax_topo,
        cmap=topo_cmap,
        vlim=(topo_vlim[0] * 1e-6, topo_vlim[1] * 1e-6),   # µV → V
        mask=highlight_mask,
        mask_params=mask_params,
        show=False,
        contours=6,
    )

    ax_topo.set_title(topo_labels[cond_idx], fontsize=12)
    fig_topo.patch.set_facecolor('white')
    fig_topo.tight_layout()

    # Save as SVG and TIFF
    for fmt in ['svg', 'tiff']:
        topo_save = os.path.join(save_to, f"{topo_prefix}_{cond_label}.{fmt}")
        fig_topo.savefig(topo_save, format=fmt, dpi=600,
                         bbox_inches='tight', facecolor='white')
    print(f"  Topo saved: {topo_prefix}_{cond_label} (.svg + .tiff)")

plt.show()   # display main-effect figures interactively; close windows to continue

