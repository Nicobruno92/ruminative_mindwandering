# H5 to PKL Batch Conversion

This directory contains scripts for batch converting Junifer HDF5 marker files + MNE FIF metadata into organized PKL files following BIDS structure.

## Files

### Core Scripts
- **`h5_to_pkl_converter.py`**: Main converter script (handles single file conversion)

### Batch Processing (Sequential)
- **`batch_convert_h5_to_pkl.sh`**: SLURM script - processes elements one by one
- **`batch_convert_h5_to_pkl_local.sh`**: Local script for testing/small batches

### Parallel Processing (Recommended)
- **`batch_convert_h5_to_pkl_parallel.sh`**: SLURM job array - 20 jobs in parallel
- **`launch_parallel.sh`**: Launch the parallel processing
- **`monitor_progress.sh`**: Monitor progress in real-time
- **`summary_report.sh`**: Generate final summary after completion

## Output Structure

PKL files are saved following BIDS format:
```
BIDS_ROOT/features/sub-XX/eeg/junifer/sub-XX_task-SartX_desc-[evoked,state]_markers.pkl
```

## Usage

### ⚡ Option 1: Parallel Processing (RECOMMENDED - Much Faster!)

Process all 167 elements in parallel with up to 20 jobs running simultaneously:

```bash
cd junifer_markers/2.h5_to_pkl

# Launch parallel processing
./launch_parallel.sh

# Monitor progress in real-time
watch -n 5 ./monitor_progress.sh

# Or check once
./monitor_progress.sh

# After completion, generate summary
./summary_report.sh
```

**Advantages:**
- ⚡ **20x faster**: Processes 20 elements simultaneously
- 🎯 **Efficient**: Each job takes ~5-10 minutes, total time ~40-60 minutes
- 🔄 **Auto-retry**: Easy to identify and re-run failed tasks
- 📊 **Progress tracking**: Real-time monitoring with progress bars

**Configuration:**
- Max parallel jobs: 20 (adjustable in script with `--array=1-167%20`)
- Time per job: 1 hour
- Memory per job: 8GB

### Option 2: Sequential SLURM (Slower)

Submit a single job to process all elements sequentially. **By default, it overwrites existing PKL files** (force mode).

```bash
cd junifer_markers/2.h5_to_pkl
sbatch batch_convert_h5_to_pkl.sh
```

To skip existing files instead, edit the script and set `FORCE_OVERWRITE=false` at the top.

Monitor the job:
```bash
squeue -u $USER
tail -f logs/h5_to_pkl_*.out
```

⚠️ **Note**: This processes elements one at a time. Estimated time: ~12-16 hours for all 167 elements.

### Option 3: Local Execution (Testing)

For testing or processing a subset locally:

```bash
cd junifer_markers/2.h5_to_pkl

# Make script executable (run once on cluster)
chmod +x batch_convert_h5_to_pkl_local.sh

# Dry run to see what would be processed
./batch_convert_h5_to_pkl_local.sh --dry-run

# Process first 5 elements for testing
./batch_convert_h5_to_pkl_local.sh --limit 5

# Force overwrite existing files
./batch_convert_h5_to_pkl_local.sh --force

# Full processing
./batch_convert_h5_to_pkl_local.sh
```

### Option 4: Single File Conversion

To convert a single element manually:

```bash
python h5_to_pkl_converter.py \
    /path/to/sub-XX_task-SartX_desc-evoked_epo.fif \
    /path/to/element_sub-XX_SartX_evoked_markers.h5 \
    /path/to/output/sub-XX_task-SartX_desc-evoked_markers.pkl
```

## Configuration

All paths are configured in the scripts:

```bash
BIDS_ROOT="/network/iss/cenir/analyse/meeg/CYBERSART/BIDS"
LOCAL_ROOT="/network/iss/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering"
DERIVATIVES_DIR="${BIDS_ROOT}/derivatives"
H5_FEATURES_DIR="${BIDS_ROOT}/features/junifer"
PKL_FEATURES_DIR="${BIDS_ROOT}/features"
ELEMENTS_FILE="${LOCAL_ROOT}/junifer_markers/1.markers_h5_creation/elements"
```

## Input Files

For each element (subject + task + desc combination), the script expects:

1. **FIF file**: `DERIVATIVES_DIR/{subject}/eeg/{subject}_task-{task}_desc-{desc}_epo.fif`
2. **H5 file**: `H5_FEATURES_DIR/element_{subject}_{task}_{desc}_markers.h5`

## Output

PKL files contain:
- **Hierarchical structure**: markers → epochs → channels
- **Full metadata**: epoch annotations, behavioral data, channel info
- **All Junifer markers**: spectral, connectivity, time-locked, information theory

Access pattern:
```python
import pickle

# Load PKL file
with open("sub-31_task-Sart4_desc-evoked_markers.pkl", "rb") as f:
    data = pickle.load(f)

# Access markers
markers = data["markers"]
metadata = data["metadata"]

# Hierarchical access: marker → epoch → channel
marker_name = "psd_bands"
epoch_idx = 0
channel_name = "Fp1"

value = markers[marker_name][epoch_idx][channel_name].data
```

## Processing Statistics

The script will report:
- Total elements processed
- Successful conversions
- Failed conversions
- Skipped (already existing files)
- **File sizes** for input (FIF, H5) and output (PKL) files

Example log output:
```
Processing desc=evoked...
  Converting: sub-25_task-Sart3_desc-evoked
    FIF: /path/to/file.fif (45M)
    H5:  /path/to/file.h5 (2.3M)
    PKL: /path/to/file.pkl
  ✓  Success - PKL size: 2.5M
```

## Logs

- **SLURM**: Logs saved to `logs/h5_to_pkl_*.out` and `logs/h5_to_pkl_*.err`
- **Local**: Output printed to terminal

## Troubleshooting

### Missing input files
```
⚠️  FIF file not found: ...
⚠️  H5 file not found: ...
```
Check that preprocessing and Junifer marker computation completed successfully for that element.

### Permission errors
Make scripts executable on the cluster:
```bash
chmod +x batch_convert_h5_to_pkl.sh
chmod +x batch_convert_h5_to_pkl_local.sh
```

### Environment issues
Ensure the `eeg` conda environment is activated and has required packages:
```bash
source activate eeg
pip install mne junifer h5py
```

## Next Steps

After conversion, PKL files can be used for:
1. Aggregating markers across epochs/conditions
2. Statistical analysis
3. Machine learning pipelines
4. Visualization

See `junifer_markers/3.aggregate_probes/` for probe-aligned aggregation scripts.
