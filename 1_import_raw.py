
import os
import gc
import numpy as np
import mne
from mne.preprocessing import ICA

# ──────────────────────────────────────────────────────────────
# Which participant(s) to process?
# Use a single-element list (e.g. [31]) to test one participant,
# or list(range(5, 38)) to run all of them.
# ──────────────────────────────────────────────────────────────
plist = [3]  # <-- change this as needed

# ──────────────────────────────────────────────────────────────
# Bad channels per participant (Step 6 from FieldTrip workflow)
# Identified after visual ICA inspection. If a channel is extremely
# noisy, add it here and re-run. Empty list = no bad channels.
# ──────────────────────────────────────────────────────────────
bad_channels = {
    7: ['P2']
    # 21: ['P8', 'TP8'],
   
}

# Input and output folders
input_path = r"/Users/purengurkan/Desktop/SoA/SoA/EEG/eeg"
output_path = r"/Users/purengurkan/Desktop/SoA/SoA/EEG/eeg_ica"

# Create output folder if it doesn't exist
os.makedirs(output_path, exist_ok=True)

# Loop through selected participants 
for sub in plist:
    
    sub_id = f"{sub:04d}"  # formats 1 -> 0001, 32 -> 0032
    filename = f"CDmem_{sub_id}.vhdr"
    filepath = os.path.join(input_path, filename)
    
    if not os.path.exists(filepath):
        print(f"File not found: {filename}")
        continue
    
    print(f"Loading {filename}")
    
    try:
        # Load BrainVision file
        # This automatically reads the .vhdr, .vmrk, and .eeg files as a single Raw object.
        raw = mne.io.read_raw_brainvision(filepath, preload=True)
        
        # Set electrode positions from the actual actiCAP .bvef file.
        bvef_path = r"/Users/purengurkan/Desktop/SoA/SoA/EEG/CACS-64_REF_new.bvef"
        montage = mne.channels.read_custom_montage(bvef_path)
        # The .bvef file names the online reference "REF", but in our data it's
        # called "FCz" (added later via add_reference_channels). Rename so
        # FCz gets a valid position instead of NaN.
        montage.rename_channels({'REF': 'FCz'})
        raw.set_montage(montage, on_missing='warn')
        
        # Remove bad channels 
        bads = bad_channels.get(sub, [])
        if bads:
            raw.drop_channels(bads)
            print(f"  Dropped bad channels: {bads}")
        
        # Apply band-pass filter
        # In MNE, l_freq is the lower cutoff frequency and h_freq is the upper cutoff frequency. 
        # So, l_freq specifies the high-pass filter and h_freq specifies the low-pass filter.
       
        # Note: We currently apply a 1.0 Hz high-pass filter. Wen et al. (2017) used a 0.1 Hz filter.
        # The P500 is a "late, slow" brain wave. High-pass filters at 1.0 Hz or higher are known to 
        # severely attenuate and distort late slow waves. 
        # If the P500 looks small/non-existent, consider using the commented out 0.1 Hz filter instead.
        # raw.filter(l_freq=0.1, h_freq=40.)
        raw.filter(
            l_freq=1.,
            h_freq=40.,
            method='iir',
            iir_params=dict(order=4, ftype='butter')
        )
        
        # Re-reference to average of all electrodes
       
        # Note: Wen et al. (2017) used Earlobe referencing. Because the P500 is a broadly 
        # distributed positive component, the Common Average Reference may subtract it out, 
        # causing it to appear smaller in amplitude compared to an earlobe/mastoid reference.
        #
        # Step 1: Add the implicit reference channel (FCz) back to the data.
        #   During recording, FCz was the online reference so it's not in the data.
        #   This adds it as a flat (all zeros) channel → 64 becomes 65 channels.
        raw.add_reference_channels('FCz')
        raw.set_montage(montage, on_missing='warn')  # re-apply so FCz gets its position
        
        # Step 2: Re-reference to the average of all 65 electrodes.
        #   Each channel = original − mean(all channels).
        #   FCz goes from zeros to −mean(all), recovering its actual signal.
        raw.set_eeg_reference('average')
        
        # Epoch the data around movement onset
        # Step 1: Extract events from BrainVision annotations.
        #   MNE reads the .vmrk markers as annotations. events_from_annotations
        #   converts them to an (N, 3) array of [sample, 0, event_id].
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        
        # Step 2: Select only our triggers of interest (movement onset)
        #   S 21 = low control / left,  S 22 = low control / right
        #   S 23 = high control / left, S 24 = high control / right
        # We include both left- and right-side targets because the ERP
        # analysis only contrasts low vs high control (target side is irrelevant).
        # Note: BrainVision annotations in MNE use the full 'Stimulus/S XX' format.
        wanted_triggers = ['Stimulus/S 21', 'Stimulus/S 22', 'Stimulus/S 23', 'Stimulus/S 24']
        triggers = {}
        for t in wanted_triggers:
            if t in event_id:
                triggers[t] = event_id[t]
            else:
                print(f"  WARNING: trigger '{t}' not found in {filename}")
        
        if not triggers:
            print(f"  SKIPPED — no matching triggers found")
            continue
        
        # Step 3: Filter events to ONLY include our target triggers.
        # This is CRITICAL so that epochs.selection (used in later matching)
        # corresponds to the trial index (0, 1, 2...) rather than the 
        # index in the full list of all EEG markers (fixations, etc.).
        mask = np.isin(events[:, 2], list(triggers.values()))
        events = events[mask]

        # Step 4: Create epochs: -2.5 s before to +2.5 s after movement onset
        epochs = mne.Epochs(
            raw, events, event_id=triggers,
            tmin=-2.5, tmax=2.5,
            preload=True, verbose=False,
            baseline = None # we will do baseline correction later explicitly to match FieldTrip
        )
        
        # Downsample to 250 Hz
        # MNE automatically applies an anti-aliasing lowpass filter before
        # downsampling (1000 Hz → 250 Hz = 4× reduction).
        # We already filtered at 40 Hz, well below the new Nyquist (125 Hz).
        epochs.resample(250, verbose=False)
        
        # Count triggers for display
        if hasattr(epochs, 'metadata') and epochs.metadata is not None:
            counts = dict(epochs.metadata['event_name'].value_counts())
        else:
            counts = {k: (epochs.events[:, 2] == v).sum() for k, v in triggers.items()}
        
        print(f"  {len(epochs)} epochs, {epochs.info['sfreq']:.0f} Hz  {counts}")
        
        # ICA for artifact correction (eyeblinks)
        #
        # ICA decomposes the data into independent components. Components
        # capturing eyeblinks/eye movements will be identified and rejected
        # in a later interactive step.
        
        # ica = ICA(
        #     method='infomax',           # same as runica
        #     fit_params=dict(extended=False),  # matches cfg.runica.extended = 0
        #     random_state=42             # for reproducibility
        # )

        ica = ICA(
            method='picard',
            fit_params=dict(ortho=False, extended=False),  # ortho=False, extended=False = standard Infomax
            random_state=42
        )
        ica.fit(epochs, verbose=False)
        print(f"  ICA fitted: {ica.n_components_} components")
        
        # Save both the epoched data and the ICA solution
        epo_name = f"CDmem_{sub_id}-epo.fif"
        ica_name = f"CDmem_{sub_id}-ica.fif"
        epochs.save(os.path.join(output_path, epo_name), overwrite=True, fmt='single')
        ica.save(os.path.join(output_path, ica_name), overwrite=True)
        
        print(f"  Saved {epo_name} + {ica_name}")
        
        # Free RAM before loading the next participant
        del raw, epochs, ica
        gc.collect()
        
    except Exception as e:
        print(f"  FAILED {filename}: {e}")

print("All done!")
