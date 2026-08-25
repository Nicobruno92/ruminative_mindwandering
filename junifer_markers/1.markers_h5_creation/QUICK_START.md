# Quick Start Guide

## Unified Management Scripts

All pipeline management is now handled by **two scripts**:

### 1. `manage_pipeline.sh` - Complete Workflow Management

```bash
# Show all commands
bash manage_pipeline.sh

# Run everything (recommended)
bash manage_pipeline.sh full

# Individual commands
bash manage_pipeline.sh diagnose    # Check for issues
bash manage_pipeline.sh fix         # Fix elements file
bash manage_pipeline.sh check-h5    # Check H5 files status
bash manage_pipeline.sh clean-h5    # Delete all H5 files
bash manage_pipeline.sh submit      # Submit SLURM jobs
```

### 2. `manage_elements.py` - Elements File Management

```bash
# Diagnose elements file
python manage_elements.py diagnose

# Fix elements file (remove duplicates and missing files)
python manage_elements.py fix
```

## Typical Workflow

```bash
# On the cluster
cd <PATH_TO_YOUR_REPO_ROOT>

# Activate environment
conda activate junifer

# Run full workflow
bash junifer_markers/1.markers_h5_creation/manage_pipeline.sh full
```

This will:
1. Diagnose elements file
2. Fix any issues (duplicates, missing files)
3. Optionally clean existing H5 files
4. Submit SLURM array jobs

## Monitor Jobs

```bash
# Watch job queue
watch squeue -u $USER

# Count H5 files being created
watch -n 60 'ls <PATH_TO_YOUR_BIDS_FEATURES_ROOT>/junifer/element_*.h5 | wc -l'

# Check logs
tail -f logs/CYBERSART_features_*_*.out
```

## Output

With `single_output: false`, creates **individual H5 files**:
```
<PATH_TO_YOUR_BIDS_FEATURES_ROOT>/junifer/
├── element_sub-02_Sart1_evoked_markers.h5
├── element_sub-02_Sart1_state_markers.h5
└── ... (332 files total)
```

Each file contains 15 markers.
