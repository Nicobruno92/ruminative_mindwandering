# Spectral Topography Exploration: MNE vs Junifer

## Overview

Two scripts are available for exploring spectral topography:

1. **`explore_spectral_topography.py`** - Original script using MNE's native PSD calculation
2. **`explore_spectral_topography_junifer.py`** - New script using Junifer's SpectralPower marker

## Key Differences

### PSD Calculation Method

**MNE Version:**
- Uses `mne.time_frequency.psd_array_welch()` for raw data
- Uses `epochs.compute_psd()` for epoched data
- Direct control over Welch parameters

**Junifer Version:**
- Uses `SpectralPower` marker from `spectral_power.py`
- Leverages Junifer's standardized marker interface
- Includes trial aggregation using trimmed mean (80%)
- Automatically handles EEG channel filtering

### Configuration

Both scripts use the same configuration variables at the top:
- `DATA_PATH`: Base directory for BIDS derivatives
- `SUBJECT_ID`: Participant ID
- `SART_SESSION`: Session identifier or "ALL" for averaging
- `DATA_TYPE`: 'raw', 'evoked', or 'state'
- `FREQ_BANDS`: Dictionary of frequency bands

### Output

**MNE Version:**
- Saves plots to `./spectral_topography_plots/`
- Filenames: `sub-{id}_task-{session}_desc-{type}_spectral_topography.png`

**Junifer Version:**
- Saves plots to `./spectral_topography_plots_junifer/`
- Filenames: `sub-{id}_task-{session}_desc-{type}_spectral_topography_junifer.png`

## Usage

### Activate Junifer Environment

```bash
conda activate junifer
```

### Run the Junifer Version

```bash
cd /Volumes/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering/Preprocessing_pipeline_new/Explore_files
python explore_spectral_topography_junifer.py
```

### Modify Configuration

Edit the configuration section at the top of the script:

```python
SUBJECT_ID = "07"  # Change to your subject
SART_SESSION = "ALL"  # Or "Sart1", "Sart2", etc.
DATA_TYPE = "evoked"  # Or "raw", "state"
```

## Technical Details

### Junifer Marker Parameters

The Junifer version uses these SpectralPower parameters:
- `fmin`, `fmax`: Band-specific frequency range
- `normalize=False`: Absolute power (not relative)
- `dB=False`: Linear scale (not decibels)
- `n_per_seg=int(sfreq)`: ~1 second window
- `n_overlap=int(sfreq * 0.5)`: 50% overlap
- `trial_aggregation_method=['trim_mean80']`: Robust aggregation across trials

### Trial Aggregation

The Junifer version automatically aggregates across trials using a trimmed mean (removing top/bottom 10%), which provides robust estimates less sensitive to outliers.

## Expected Differences

The two methods may produce slightly different results due to:
1. **Trial aggregation**: Junifer uses trimmed mean by default
2. **Channel filtering**: Junifer automatically filters to EEG channels
3. **Implementation details**: Minor differences in Welch parameter handling

For most analyses, the results should be highly similar, with the Junifer version providing more standardized and reproducible outputs.
