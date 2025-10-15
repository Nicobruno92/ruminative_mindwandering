# Pipeline 2: PKL Creation from H5 and FIF Files

This pipeline converts HDF5 marker files and FIF metadata files into PKL (pickle) files with a standardized structure for downstream analysis.

## Overview

After Pipeline 1 generates HDF5 feature files, this pipeline:
1. Reads metadata from original FIF epoch files
2. Reads computed features from HDF5 files
3. Combines them into structured PKL files with proper channel/epoch organization
4. Outputs BIDS-compliant PKL files for each subject/task/desc combination

## Pipeline Components

### 1. Batch Processing Script (`batch_create_pkl_from_pipeline.py`)
Main orchestration script that:
- Parses elements file to get list of (subject, task) pairs
- Constructs input/output paths following BIDS conventions
- Processes each element-desc combination
- Handles error reporting and dry-run mode

### 2. Conversion Logic (`create_pkl_from_h5_fif.py`)
Core conversion module that:
- Reads FIF metadata (epochs, channels, annotations, behavioral parameters)
- Reads H5 marker data using Junifer storage interface
- Creates structured PKL data with proper access patterns:
  - **Spectral markers**: channel × epoch × band
  - **ERP markers**: channel × epoch
  - **Connectivity markers**: channel_pair × epoch
- Embeds complete metadata and annotations

### 3. HDF5 Reader (`junifer_hdf5_reader_final.py`)
Utility module that:
- Uses Junifer's HDF5FeatureStorage class
- Lists and reads features from HDF5 files
- Handles MD5 hashing and metadata automatically
- Provides summary statistics

### 4. SLURM Job Script (`batch_create_pkl.sh`)
Cluster submission script for batch processing:
- Can be run standalone or chained after Pipeline 1
- Configurable resources and paths
- Proper conda environment activation

## Input Requirements

### Directory Structure
```
/network/iss/cenir/analyse/meeg/CYBERSART/BIDS/
├── derivatives/
│   └── {subject}/
│       └── eeg/
│           └── {subject}_task-{task}_desc-{desc}_epo.fif
└── features/
    └── junifer/
        └── markers.h5  # From Pipeline 1
```

### Elements File
Same format as Pipeline 1:
```
sub-31,Sart4,evoked
sub-31,Sart4,state
...
```

## Usage

### Basic Usage (Local)

1. **Process all elements**:
   ```bash
   cd /path/to/junifer_markers/pipeline_2_pkl_creation
   python batch_create_pkl_from_pipeline.py \
     --elements-file /path/to/elements \
     --desc both
   ```

2. **Dry run (test)**:
   ```bash
   python batch_create_pkl_from_pipeline.py \
     --elements-file /path/to/elements \
     --desc both \
     --dry-run
   ```

3. **Process only evoked**:
   ```bash
   python batch_create_pkl_from_pipeline.py \
     --elements-file /path/to/elements \
     --desc evoked
   ```

4. **Limit to first N elements (testing)**:
   ```bash
   python batch_create_pkl_from_pipeline.py \
     --elements-file /path/to/elements \
     --desc both \
     --limit 5
   ```

### Usage on Cluster (SLURM)

**Standalone submission**:
```bash
cd /network/iss/home/nicolas.bruno/Junifer
sbatch junifer_markers/pipeline_2_pkl_creation/batch_create_pkl.sh
```

**With custom parameters**:
```bash
JOBNAME=CYBERSART_features \
WORKDIR=/network/iss/home/nicolas.bruno/Junifer \
ELEMENTS_FILE=/path/to/elements \
sbatch junifer_markers/pipeline_2_pkl_creation/batch_create_pkl.sh
```

**Automatic chaining from Pipeline 1**:
```bash
# This is done automatically by Pipeline 1's submit_slurm_array.sh
# No action needed - PKL creation runs after H5 collection completes
```

### Single File Conversion

For debugging or one-off conversions:
```bash
python create_pkl_from_h5_fif.py \
  /path/to/markers.h5 \
  /path/to/sub-31_task-Sart4_desc-evoked_epo.fif \
  /path/to/output.pkl \
  --subject sub-31 \
  --task Sart4 \
  --desc evoked
```

## Output

### PKL File Structure
```
/network/iss/cenir/analyse/meeg/CYBERSART/BIDS/features/
└── {subject}/
    └── eeg/
        └── junifer/
            ├── {subject}_task-{task}_desc-evoked_markers.pkl
            └── {subject}_task-{task}_desc-state_markers.pkl
```

### PKL Content Structure
Each PKL file contains a dictionary:

```python
{
    "markers": {
        "psd_bands": {
            "access_pattern": "channel_epoch",
            "channel_names": [...],
            "epoch_annotations": [...],
            "data": {
                "Fp1": {"delta_epoch_0000": value, ...},
                "Fp2": {...},
                ...
            }
        },
        "wsmi_theta": {
            "access_pattern": "channel_pair_epoch",
            ...
        },
        "P1": {
            "access_pattern": "channel_epoch",
            ...
        },
        ...
    },
    "annotations": {
        "epochs": [
            {
                "epoch_index": 0,
                "epoch_id": "epoch_0000",
                "event_time": ...,
                "event_code": ...,
                "event_description": ...,
                "response_type": "go",
                "accuracy": "correct",
                "reaction_time": 450,
                ...
            },
            ...
        ]
    },
    "info": {
        "n_epochs": 100,
        "n_channels": 64,
        "channel_names": ["Fp1", "Fp2", ...],
        "created_at": "2025-10-08 12:34:56",
        "n_markers": 15,
        "access_patterns": {
            "psd_bands": "channel_epoch",
            "wsmi_theta": "channel_pair_epoch",
            ...
        }
    }
}
```

## Marker Types and Access Patterns

### 1. Spectral Power (channel_epoch)
- **Markers**: `psd_bands`, `psd_relative`
- **Structure**: `data[channel][band_epoch]`
- **Example**: `data["Fp1"]["delta_epoch_0042"]`

### 2. Event-Related Potentials (channel_epoch)
- **Markers**: `P1`, `N1`, `P2`, `P3a`, `P3b`, `PE_theta`, etc.
- **Structure**: `data[channel][epoch]`
- **Example**: `data["Cz"]["epoch_0042"]`

### 3. Connectivity (channel_pair_epoch)
- **Markers**: `wsmi_theta`, `wsmi_alpha`, `wsmi_beta`, `wsmi_gamma`
- **Structure**: `data[channel_pair][epoch]`
- **Example**: `data["Fp1-Fp2"]["epoch_0042"]`

### 4. Complexity Measures (channel_epoch)
- **Markers**: `kolmogorov_complexity`
- **Structure**: `data[channel][epoch]`
- **Example**: `data["Fp1"]["epoch_0042"]`

## Command Line Arguments

### batch_create_pkl_from_pipeline.py

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--elements-file` | Path | `junifer_jobs/CYBERSART_features/elements` | Path to elements file |
| `--desc` | Choice | `both` | Description type: `evoked`, `state`, or `both` |
| `--output-dir` | Path | _(computed)_ | Override output directory |
| `--dry-run` | Flag | False | Test without processing |
| `--limit` | Int | _(none)_ | Limit number of elements for testing |

### create_pkl_from_h5_fif.py

| Argument | Required | Description |
|----------|----------|-------------|
| `h5_file` | Yes | Path to HDF5 file |
| `fif_file` | Yes | Path to FIF file |
| `output_pkl` | Yes | Path for output PKL file |
| `--subject` | No | Subject ID (e.g., sub-31) |
| `--task` | No | Task name (e.g., Sart4) |
| `--desc` | No | Description (e.g., evoked, state) |

## Environment Variables (SLURM)

| Variable | Default | Description |
|----------|---------|-------------|
| `JOBNAME` | `CYBERSART_features` | Job name |
| `WORKDIR` | `/network/iss/home/nicolas.bruno/Junifer` | Working directory |
| `CONDA_ENV` | `junifer` | Conda environment |
| `ELEMENTS_FILE` | `${WORKDIR}/junifer_jobs/${JOBNAME}/elements` | Elements file |
| `PIPELINE_DIR` | `${WORKDIR}/junifer_markers/pipeline_2_pkl_creation` | Pipeline directory |

## Resource Requirements

### Typical Job
- **CPUs**: 4
- **Memory**: 16GB
- **Time**: 4 hours (for ~336 elements)
- **Walltime per element**: ~1-2 minutes

### Recommendations
- For large cohorts: increase memory to 32GB
- For testing: use `--limit` to process subset
- Use `--dry-run` first to verify paths

## Troubleshooting

### Missing Input Files
```bash
# Check FIF files exist
ls /network/iss/cenir/analyse/meeg/CYBERSART/BIDS/derivatives/sub-*/eeg/*.fif

# Check H5 file exists
ls /network/iss/cenir/analyse/meeg/CYBERSART/BIDS/features/junifer/markers.h5
```

### Element Mismatch Errors
If features from H5 don't match elements:
1. Verify H5 file was created by Pipeline 1 with same elements
2. Check element dict keys: subject, task, desc
3. Re-run Pipeline 1 if necessary

### Import Errors
```bash
# Ensure proper PYTHONPATH
export PYTHONPATH=/path/to/Junifer:/path/to/pipeline_2_pkl_creation:$PYTHONPATH

# Check junifer_eeg is importable
python -c "import junifer_eeg; print('OK')"
```

### Memory Issues
If jobs fail with OOM errors:
- Increase `--mem` in SLURM script
- Process fewer elements per job
- Check for memory leaks in conversion code

### Debugging Specific Element
```bash
# Run single element with verbose output
python create_pkl_from_h5_fif.py \
  /path/to/markers.h5 \
  /path/to/sub-31_task-Sart4_desc-evoked_epo.fif \
  /tmp/test_output.pkl \
  --subject sub-31 \
  --task Sart4 \
  --desc evoked

# Inspect output
python -c "import pickle; print(pickle.load(open('/tmp/test_output.pkl', 'rb')).keys())"
```

## Dependencies

### Python Packages
- junifer (with HDF5FeatureStorage)
- junifer_eeg (custom markers)
- mne
- numpy
- pandas
- pickle (stdlib)

### File Dependencies
- Elements file from Pipeline 1
- HDF5 markers file from Pipeline 1
- Original FIF epoch files from preprocessing

## Integration with Analysis Pipeline

The PKL files are designed for downstream analysis:

```python
import pickle

# Load PKL file
with open("sub-31_task-Sart4_desc-evoked_markers.pkl", "rb") as f:
    data = pickle.load(f)

# Access markers
psd_bands = data["markers"]["psd_bands"]
wsmi_theta = data["markers"]["wsmi_theta"]
p3b = data["markers"]["P3b"]

# Access annotations
epochs = data["annotations"]["epochs"]

# Filter by behavioral parameters
go_trials = [e for e in epochs if e.get("response_type") == "go"]
correct_trials = [e for e in epochs if e.get("accuracy") == "correct"]

# Extract specific channel/epoch
fp1_delta_epoch0 = psd_bands["data"]["Fp1"]["delta_epoch_0000"]
cz_p3b_epoch0 = p3b["data"]["Cz"]["epoch_0000"]
```

## Performance Notes

- **Batch processing**: Processes ~168 elements × 2 desc = 336 files in ~4 hours
- **Per-element time**: 1-2 minutes (mostly I/O)
- **Output size**: ~1-5 MB per PKL file (depends on number of epochs)
- **Parallelization**: Currently sequential, could be parallelized by desc type

## Future Improvements

1. **Parallel processing**: Use multiprocessing for desc types
2. **Incremental updates**: Only process new/modified elements
3. **Validation**: Add PKL file validation/checksums
4. **Compression**: Option for compressed PKL storage
5. **Metadata**: Embed pipeline version and config in PKL

## Notes

- PKL files preserve all behavioral annotations from FIF events
- Access patterns are documented in each marker for easy data retrieval
- Element information is used to select correct features from multi-element H5 files
- The pipeline handles missing fields gracefully (fills with np.nan)
- Output directory structure follows BIDS derivatives specification

## References

- Junifer storage documentation: https://juaml.github.io/junifer/main/api_storage.html
- MNE epochs documentation: https://mne.tools/stable/generated/mne.Epochs.html
- Python pickle documentation: https://docs.python.org/3/library/pickle.html
