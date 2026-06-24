import os
import sys
import numpy as np
import pandas as pd
import mne
from mne.stats import permutation_cluster_1samp_test, permutation_cluster_test
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Force UTF-8 output so special characters print cleanly on Windows PowerShell
sys.stdout.reconfigure(encoding='utf-8')

# ──────────────────────────────────────────────────────────────
# Which participant(s) do you want to process?
# ──────────────────────────────────────────────────────────────
plist = [4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22]

# ──────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────
dfolder = r"H:\PHD\control_detection\main_data\eeg\eeg4_ERPSummaries"

# ──────────────────────────────────────────────────────────────
# Load data for all participants
# ──────────────────────────────────────────────────────────────
# We load  the main ERP fif files 
eeg_set = {}

for p_idx, pnum in enumerate(plist):
    sub_id = f"{pnum:04d}"
    erp_file = os.path.join(dfolder, f"CDmem_{sub_id}-erp-ave.fif")
    
    if not os.path.exists(erp_file):
        print(f"  ERP file not found, skipping participant {pnum}: {erp_file}")
        continue
    

    print(f"Loading participant {pnum}")

    # Load all Evoked objects for this participant
    evokeds_main = mne.read_evokeds(erp_file, verbose=False)
    
    
    # Store all by comment/condition label
    eeg_set[p_idx] = {ev.comment: ev for ev in evokeds_main}

# ──────────────────────────────────────────────────────────────
# Electrode and time window selection (matches 4_ERP_graphs.py)
# ──────────────────────────────────────────────────────────────
# elec_include = ['FCz', 'Cz', 'CPz', 'Pz']  # (Wen et al., 2017)
elec_include = ['Fz', 'FCz', 'FC1', 'FC2']  # (following Giersiepen et al., 2024, 2025)
plot_time = [-0.3, 1.2]    # Full time window for the line graphs (matches 4_ERP_graphs.py)
test_time = [0.0, 1.0]     # Time window restricted for the permutation test

# Get shared time axis from first loaded participant
t_axis_full = None
for p_idx in eeg_set:
    if 'high_control' in eeg_set[p_idx]:
        t_axis_full = eeg_set[p_idx]['high_control'].copy().crop(tmin=plot_time[0], tmax=plot_time[1]).times
        break

if t_axis_full is None:
    raise ValueError("No valid ERP data loaded to determine the time axis.")

# Helper to average over picked channels and convert to µV
def extract_p_data(evoked, electrodes, tmin, tmax):
    available_ch = evoked.ch_names
    picked_ch = [ch for ch in electrodes if ch in available_ch]
    ev_crop = evoked.copy().crop(tmin=tmin, tmax=tmax).pick(picked_ch)
    return ev_crop.data.mean(axis=0) * 1e6  # Average over channels, convert to µV

# Extract time-series for all participants
extracted_data = {}
for p_idx in eeg_set:
    p_data = {}
    for cond_name, evoked in eeg_set[p_idx].items():
        p_data[cond_name] = extract_p_data(evoked, elec_include, plot_time[0], plot_time[1])
    extracted_data[p_idx] = p_data

# Identify valid participant subsets for each test
# Test 1 needs: 'high_control', 'low_control'
subs_test1 = [p for p in extracted_data if 'high_control' in extracted_data[p] and 'low_control' in extracted_data[p]]


# Output folder for figures
save_to = r"H:\PHD\control_detection\main_data\eeg\eeg5_figures"
os.makedirs(save_to, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# Helper: extract time indices from an MNE cluster object
# ──────────────────────────────────────────────────────────────
# In MNE 1.11+, clusters are returned as tuples of slices,
# e.g. (slice(32, 45),), regardless of out_type. This helper
# converts any cluster format into a plain integer index array.
def _get_cluster_inds(cluster):
    """Return a 1-D integer array of time-point indices for *cluster*."""
    # cluster is a tuple like (slice(start, stop),) or (np.array([...]),)
    if isinstance(cluster, tuple):
        obj = cluster[0]
        if isinstance(obj, slice):
            return np.arange(obj.start, obj.stop)
        else:
            return np.asarray(obj).ravel()
    # boolean mask
    if isinstance(cluster, np.ndarray) and cluster.dtype == bool:
        return np.flatnonzero(cluster)
    # fallback
    return np.flatnonzero(np.asarray(cluster))

# ──────────────────────────────────────────────────────────────
# Topoplot Settings & Helpers (Matching 4_ERP_graphs.py)
# ──────────────────────────────────────────────────────────────
topo_cmap = plt.cm.RdBu_r
topo_vlim = (-1.5, 1.5)
topo_fig_in = 8 / 2.54

def make_highlight_mask(evoked, highlight_names):
    return np.array([ch in highlight_names for ch in evoked.ch_names])

mask_params = dict(marker='*', markerfacecolor='black', markeredgecolor='black',
                   markersize=10, zorder=10)

# ──────────────────────────────────────────────────────────────
# Helper function to run permutation test & plot/save results
# ──────────────────────────────────────────────────────────────
def run_permutation_test(X_condA, X_condB, label_A, label_B, title, save_filename, colors, linestyles, p_indices, evs_A=None, evs_B=None):
    # Find indices for the test window
    test_inds = np.where((t_axis_full >= test_time[0] - 1e-5) & (t_axis_full <= test_time[1] + 1e-5))[0]
    
    # Extract data just for the test
    X_test_A = X_condA[:, test_inds]
    X_test_B = X_condB[:, test_inds]
    
    # Paired difference (A - B) for 1-sample cluster test
    X_diff = X_test_A - X_test_B
    
    n_permutations = 1000
    alpha = 0.05
    
    print(f"\n" + "="*60)
    print(f"  RUNNING TEST: {title}")
    print(f"  N = {len(p_indices)} participants: {[plist[p] for p in p_indices]}")
    print("="*60)
    
    T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(
        X_diff, 
        n_permutations=n_permutations, 
        tail=0,             # two-sided
        n_jobs=-1,
        seed=42
    )
    
    if clusters is None:
        clusters = []
        cluster_p_values = []
        
    good_cluster_inds = np.where(cluster_p_values < alpha)[0]
    
    txt_out = []
    txt_out.append(f"RESULTS FOR: {title}")
    txt_out.append(f"----------------------------------------")
    txt_out.append(f"Total clusters found: {len(clusters)}")
    txt_out.append(f"Significant clusters (p < {alpha}): {len(good_cluster_inds)}")
    txt_out.append(f"----------------------------------------\n")
    
    for i_clu in range(len(clusters)):
        clu_inds = _get_cluster_inds(clusters[i_clu])
        # clu_inds are indices within the cropped X_diff
        real_time_inds = test_inds[clu_inds]
        c_tmin = t_axis_full[real_time_inds[0]]
        c_tmax = t_axis_full[real_time_inds[-1]]
        p_val  = cluster_p_values[i_clu]
        avg_T = np.mean(T_obs[clu_inds])
        sum_T = np.sum(T_obs[clu_inds])
        direction = "Negative" if avg_T > 0 else "Positive"
        sig_marker = " ★ SIGNIFICANT" if p_val < alpha else ""
        
        # Compute Cohen's d and statistics for every cluster
        cluster_diff_data = X_diff[:, clu_inds]
        participant_mean_diff = np.mean(cluster_diff_data, axis=1)
        mean_diff = np.mean(participant_mean_diff)
        std_diff = np.std(participant_mean_diff, ddof=1)
        cohens_d = mean_diff / std_diff if std_diff > 0 else np.nan
        
        txt_out.append(f"  Cluster {i_clu+1}: {direction} cluster from {c_tmin:.3f} s to {c_tmax:.3f} s  (p = {p_val:.4f}){sig_marker}")
        txt_out.append(f"    n_timepoints:    {len(clu_inds)}")
        txt_out.append(f"    Mean T-obs:      {avg_T:8.4f}")
        txt_out.append(f"    Sum T-obs:       {sum_T:8.4f}")
        txt_out.append(f"    Mean difference: {mean_diff:8.4f} µV")
        txt_out.append(f"    SD difference:   {std_diff:8.4f} µV")
        txt_out.append(f"    Cohen's d:       {cohens_d:8.4f}\n")
    
    if len(clusters) == 0:
        txt_out.append(f"  No clusters found at all.\n")
        
    final_text = "\n".join(txt_out)
    print(f"\n{final_text}")
    
    # Save the text to a file
    txt_filename = save_filename.replace('.png', '.txt')
    txt_filepath = os.path.join(save_to, txt_filename)
    with open(txt_filepath, 'w', encoding='utf-8') as f:
        f.write(final_text)
            
    # Plotting Setup
    # Plotting Setup
    fontsz = 20
    fig_w_in, fig_h_in = 30 / 2.54, 20 / 2.54  # 30x20 cm
    fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in))
    ax.set_title(title, fontsize=fontsz)
    
    # # Calculate y-limits dynamically
    # X_all = np.vstack((X_condA, X_condB))
    # max_val = np.max(np.abs(X_all))
    # ylim_val = np.ceil(max_val * 1.1)
    # ylim_val = max(ylim_val, 3.0)  # at least 3 µV for matching 4_ERP_graphs
    # ylimits = [-ylim_val, ylim_val]
    ylimits = [-3,3]

    
    ax.set_ylim(ylimits)
    ax.invert_yaxis()  # Reverse positive & negative poles (standard EEG view)
    ax.set_ylabel('Activity (µV)', fontsize=fontsz)
    
    # X-Axis Setup
    ax.set_xlim([plot_time[0], plot_time[1]])
    ax.set_xlabel('Time (s)', fontsize=fontsz)
    
    # Add horizontal dotted line at y=0 and vertical at t=0
    ax.axhline(0, color='black', linestyle='--', linewidth=0.5)
    if plot_time[0] <= 0 <= plot_time[1]:
        ax.axvline(0, color='black', linestyle='--', linewidth=0.5)
        
    # Fill area of significant clusters with light grey
    for i_clu, clu_idx in enumerate(good_cluster_inds):
        clu_inds = _get_cluster_inds(clusters[clu_idx])
        real_time_inds = test_inds[clu_inds]
        c_tmin = t_axis_full[real_time_inds[0]]
        c_tmax = t_axis_full[real_time_inds[-1]]
        
        ax.fill_between([c_tmin, c_tmax], ylimits[0], ylimits[1],
                        color=[0.7, 0.7, 0.7], alpha=0.5, edgecolor='none')
        ax.axvline(c_tmin, color='black', linestyle='--', linewidth=0.5)
        ax.axvline(c_tmax, color='black', linestyle='--', linewidth=0.5)
        
    # Plot ERP Lines (No standard error shading, to match 4_ERP_graphs.py exactly)
    grandMean_A = np.mean(X_condA, axis=0)
    grandMean_B = np.mean(X_condB, axis=0)
    
    h_A, = ax.plot(t_axis_full, grandMean_A, color=colors[0], linestyle=linestyles[0], linewidth=2)
    h_B, = ax.plot(t_axis_full, grandMean_B, color=colors[1], linestyle=linestyles[1], linewidth=2)
    
    ax.legend([h_A, h_B], [label_A, label_B], loc='upper left', fontsize=fontsz)
    ax.tick_params(labelsize=fontsz)
    
    # Matching ticks and spines
    ax.set_xticks(np.arange(-0.3, 1.2 + 0.01, 0.1))
    
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    for spine in ax.spines.values():
        spine.set_linewidth(1)
        
    plt.tight_layout()
    
    save_path = os.path.join(save_to, save_filename)
    fig.savefig(save_path, format='png', dpi=600, bbox_inches='tight', facecolor='white')
    print(f"Figure saved to: {save_path}")
    print("----------------------------------------")

    # ── Topoplots for significant clusters ──
    if len(good_cluster_inds) > 0 and evs_A is not None and evs_B is not None:
        print("\n  Generating topoplots for significant clusters...")
        GA_A = mne.grand_average(evs_A)
        GA_B = mne.grand_average(evs_B)
        
        for i_clu, clu_idx in enumerate(good_cluster_inds):
            clu_inds = _get_cluster_inds(clusters[clu_idx])
            real_time_inds = test_inds[clu_inds]
            c_tmin = t_axis_full[real_time_inds[0]]
            c_tmax = t_axis_full[real_time_inds[-1]]
            
            for GA, label in zip([GA_A, GA_B], [label_A, label_B]):
                ev_topo = GA.copy().crop(tmin=c_tmin, tmax=c_tmax)
                topo_data = ev_topo.data.mean(axis=1)
                
                fig_t, ax_t = plt.subplots(figsize=(topo_fig_in, topo_fig_in))
                h_mask = make_highlight_mask(ev_topo, elec_include)
                mne.viz.plot_topomap(
                    topo_data, ev_topo.info,
                    axes=ax_t, cmap=topo_cmap,
                    vlim=(topo_vlim[0] * 1e-6, topo_vlim[1] * 1e-6),
                    mask=h_mask, mask_params=mask_params,
                    show=False, contours=6
                )
                ax_t.set_title(f"{label}\n({c_tmin*1000:.0f} - {c_tmax*1000:.0f} ms)", fontsize=12)
                fig_t.patch.set_facecolor('white')
                fig_t.tight_layout()
                
                topo_name = save_filename.replace('.png', f'_topo_cluster{i_clu+1}_{label.replace(" ", "_")}')
                tsave = os.path.join(save_to, f"{topo_name}.png")
                fig_t.savefig(tsave, format='png', dpi=600, bbox_inches='tight', facecolor='white')
                print(f"    Saved: {topo_name}")


# ──────────────────────────────────────────────────────────────
# RUN THE TESTS
# ──────────────────────────────────────────────────────────────

# --- TEST 1: Main Effect of Condition (High vs. Low Control) ---
X1_A = np.array([extracted_data[p]['high_control'] for p in subs_test1])
X1_B = np.array([extracted_data[p]['low_control'] for p in subs_test1])
evs1_A = [eeg_set[p]['high_control'] for p in subs_test1]
evs1_B = [eeg_set[p]['low_control'] for p in subs_test1]

run_permutation_test(
    X_condA=X1_A, 
    X_condB=X1_B, 
    label_A='High Control', 
    label_B='Low Control', 
    title='Main Effect of Condition (High vs. Low Control)', 
    save_filename='01_permut_main_effect_condition.png', 
    colors=[(0.00, 0.44, 0.69), (0.80, 0.47, 0.65)], 
    linestyles=['-', '-'],
    p_indices=subs_test1,
    evs_A=evs1_A,
    evs_B=evs1_B
)


print("\nALL PERMUTATION TESTS COMPLETED SUCCESSFULLY!")
plt.show()
