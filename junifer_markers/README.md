# Junifer EEG Feature Extraction Pipelines

This directory contains two complete, production-ready pipelines for extracting EEG features from BIDS-formatted data using Junifer with SLURM cluster support.

## Quick Navigation

- **[Pipeline 1: H5 Markers Creation](./1.markers_h5_creation/)** - Extract EEG features to HDF5 format
- **[Pipeline 2: PKL Creation](./2.h5_to_pkl/)** - Convert H5 + FIF to structured PKL files

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    BIDS Input Data                          │
│  derivatives/{subject}/eeg/{subject}_task-{task}_desc-      │
│                  {desc}_epo.fif                             │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              PIPELINE 1: H5 Markers Creation                │
│  • Reads FIF files from BIDS derivatives                    │
│  • Computes features using Junifer markers                  │
│  • Outputs to HDF5 storage                                  │
│  • Parallelized via SLURM array jobs                        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                 features/junifer/markers.h5                 │
│  • All computed features in single HDF5 file                │
│  • Element-specific data with metadata                      │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              PIPELINE 2: PKL Creation                       │
│  • Reads H5 features + FIF metadata                         │
│  • Combines into structured PKL format                      │
│  • BIDS-compliant output per subject/task/desc             │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              features/{subject}/eeg/junifer/                │
│              {subject}_task-{task}_desc-{desc}_             │
│                    markers.pkl                              │
│  • Ready for analysis                                       │
│  • Structured access patterns                               │
│  • Complete metadata + annotations                          │
└─────────────────────────────────────────────────────────────┘
```

## Features Computed

Both pipelines compute the following EEG markers:

### 1. Spectral Analysis
- **Absolute Power (dB)**: Delta, Theta, Alpha, Beta, Gamma bands
- **Relative Power**: Normalized frequency band power

### 2. Connectivity
- **Weighted Symbolic Mutual Information (WSMI)**: 
  - Theta (tau=8, kernel=3)
  - Alpha (tau=4, kernel=3)
  - Beta (tau=2, kernel=3)
  - Gamma (tau=1, kernel=3)

### 3. Information Theory
- **Permutation Entropy (PE)**: Per frequency band
- **Kolmogorov Complexity**: Algorithmic complexity measure

### 4. Event-Related Potentials (ERPs)
- **Early components**: P1 (80-130ms), N1 (140-200ms), P2 (180-250ms)
- **Late components**: P3a (250-350ms), P3b (300-500ms)

## Quick Start

### Prerequisites
```bash
# Ensure conda environment with junifer is available
conda activate junifer

# Verify installations
python -c "import junifer, junifer_eeg, mne; print('All imports OK')"
```

### End-to-End Execution

```bash
**Usage**:
```bash
cd /network/iss/home/nicolas.bruno/Junifer
./junifer_markers/1.markers_h5_creation/submit_slurm_array.sh --queue
```

# This will:
#  - Generate elements file
#  - Submit SLURM array for H5 creation (parallel)
#  - Submit collection job (after array completes)
#  - Submit PKL creation job (after collection completes)

# 3. Monitor progress
squeue -u $USER
watch squeue -u $USER

# 4. Check logs
tail -f logs/CYBERSART_features_*.out
```

### Running Pipelines Separately

**Only H5 creation**:
```bash
CREATE_PKL=no ./junifer_markers/1.markers_h5_creation/submit_slurm_array.sh --queue
```

**Only PKL creation** (if H5 already exists):
```bash
sbatch junifer_markers/2.h5_to_pkl/batch_create_pkl.sh
```

## Directory Structure

```
junifer_markers/
├── README.md                          # This file
├── 1.markers_h5_creation/
│   ├── README.md                      # Pipeline 1 documentation
│   ├── config.yaml                    # Junifer pipeline configuration
│   ├── slurm_array_junifer.sh         # SLURM array job script
│   └── submit_slurm_array.sh          # Job submission helper
└── 2.h5_to_pkl/
    ├── README.md                      # Pipeline 2 documentation
    ├── batch_create_pkl_from_pipeline.py  # Main batch processor
    ├── create_pkl_from_h5_fif.py      # Conversion logic
    ├── junifer_hdf5_reader_final.py   # HDF5 reader utility
    └── batch_create_pkl.sh            # SLURM job script
```

## Configuration

### Pipeline 1 Configuration

Edit `1.markers_h5_creation/config.yaml` to customize:
- Data grabber patterns
- Marker parameters (frequency bands, time windows, etc.)
- Storage location
- Preprocessing options

### Environment Variables

Both pipelines support these environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `WORKDIR` | `/network/iss/home/nicolas.bruno/Junifer` | Working directory |
| `CONDA_ENV` | `junifer` | Conda environment name |
| `JOBNAME` | `CYBERSART_features` | Job/project name |
| `CPUS` | `4` | CPUs per task |
| `MEM` | `8G` (P1) / `16G` (P2) | Memory per task |
| `TIME` | `08:00:00` (P1) / `04:00:00` (P2) | Time limit |

## Resource Requirements

### Pipeline 1 (H5 Creation)
- **Per element**: 4 CPUs, 8GB RAM, ~2-4 hours
- **For 168 elements**: 672 CPUs, 1.3TB RAM total (parallel)
- **Walltime**: ~8 hours (all parallel)

### Pipeline 2 (PKL Creation)
- **Total job**: 4 CPUs, 16GB RAM, ~4 hours
- **Per element**: ~1-2 minutes (sequential)
- **For 336 files**: ~4 hours total

## Output Formats

### HDF5 Output (Pipeline 1)
```
features/junifer/markers.h5
```
- Single file with all features
- Element-based organization
- Junifer storage format

### PKL Output (Pipeline 2)
```
features/{subject}/eeg/junifer/{subject}_task-{task}_desc-{desc}_markers.pkl
```
- One file per subject/task/desc
- Structured dictionaries with:
  - `markers`: All computed features
  - `annotations`: Epoch annotations + behavioral data
  - `info`: Metadata (channels, n_epochs, etc.)

## Data Access Patterns

### Spectral Power
```python
data["markers"]["psd_bands"]["data"]["Fp1"]["delta_epoch_0042"]
```

### Event-Related Potentials
```python
data["markers"]["P3b"]["data"]["Cz"]["epoch_0042"]
```

### Connectivity
```python
data["markers"]["wsmi_theta"]["data"]["Fp1-Fp2"]["epoch_0042"]
```

## Troubleshooting

### Common Issues

**1. Elements file not found**
```bash
# Regenerate elements file
junifer queue junifer_markers/1.markers_h5_creation/config.yaml \
  --overwrite --verbose info
```

**2. Import errors**
```bash
# Verify PYTHONPATH
export PYTHONPATH=/path/to/Junifer:$PYTHONPATH

# Test imports
python -c "import junifer_eeg; print('OK')"
```

**3. SLURM jobs failing**
```bash
# Check logs
ls -lh logs/CYBERSART_features_*
tail logs/CYBERSART_features_*_*.err

# Re-run specific failed indices
sbatch --array=5,12,23 junifer_markers/1.markers_h5_creation/slurm_array_junifer.sh
```

**4. Memory issues**
```bash
# Increase memory allocation
MEM=32G sbatch batch_create_pkl.sh
```

### Getting Help

1. Check individual pipeline READMEs for detailed documentation
2. Review SLURM logs in `logs/` directory
3. Test with `--dry-run` flag (Pipeline 2)
4. Use `--limit` to test on small subset (Pipeline 2)

## Testing

### Test Pipeline 1 (single element)
```bash
# Run one element manually
ELEMENT="sub-31,Sart4,evoked"
junifer run junifer_markers/1.markers_h5_creation/config.yaml \
  --element "$ELEMENT" --verbose info
```

### Test Pipeline 2 (dry run)
```bash
cd junifer_markers/2.h5_to_pkl
python batch_create_pkl_from_pipeline.py \
  --elements-file /path/to/elements \
  --desc evoked \
  --limit 5 \
  --dry-run
```

## Performance Optimization

### Pipeline 1
- Use SLURM array for parallel processing
- Constrain threading (OMP_NUM_THREADS=1) to avoid contention
- Monitor cluster load and adjust batch size

### Pipeline 2
- Currently sequential (by design for I/O)
- Can process evoked and state separately for parallelization
- Consider increasing CPUs if I/O becomes bottleneck

## Dependencies

### Core Requirements
- Python 3.8+
- junifer (with HDF5FeatureStorage)
- junifer_eeg (custom markers)
- mne
- numpy
- scipy
- pandas
- h5py

### Cluster Requirements
- SLURM workload manager
- Conda/Miniconda
- Shared filesystem (for BIDS data)

### Installation
```bash
# Create conda environment
conda create -n junifer python=3.10
conda activate junifer

# Install core packages
pip install junifer mne numpy scipy pandas h5py

# Install junifer_eeg (custom markers)
cd /path/to/Junifer/junifer_eeg
pip install -e .
```

## BIDS Compliance

Both pipelines follow BIDS derivatives specification:

### Input (BIDS Derivatives)
```
derivatives/{subject}/eeg/{subject}_task-{task}_desc-{desc}_epo.fif
```

### Output (BIDS Features)
```
features/{subject}/eeg/junifer/{subject}_task-{task}_desc-{desc}_markers.pkl
```

## Advanced Usage

### Custom Marker Configuration

Edit `1.markers_h5_creation/config.yaml`:
```yaml
markers:
  - kind: SpectralPower
    name: custom_bands
    bands:
      custom1: [2.0, 5.0]
      custom2: [5.0, 10.0]
    on: EEG
```

### Custom Elements File

Create custom elements file:
```bash
cat > custom_elements.txt << EOF
sub-01,TaskA,evoked
sub-01,TaskA,state
sub-02,TaskA,evoked
EOF

# Use with pipeline
./submit_slurm_array.sh
# Then edit ELEMENTS_FILE variable or use:
ELEMENTS_FILE=custom_elements.txt ./submit_slurm_array.sh
```

### Batch Processing Specific Subjects

```bash
# Extract specific subjects from elements file
grep "sub-31\|sub-32" junifer_jobs/CYBERSART_features/elements > subset_elements.txt

# Process subset
cd junifer_markers/2.h5_to_pkl
python batch_create_pkl_from_pipeline.py \
  --elements-file subset_elements.txt \
  --desc both
```

## Version History

- **v1.0** (2025-10-08): Initial production pipeline
  - Complete H5 → PKL conversion
  - SLURM array support
  - BIDS-compliant output
  - Comprehensive documentation

## Contributing

To modify or extend these pipelines:

1. **Add new markers**: Edit `config.yaml` in Pipeline 1
2. **Modify conversion**: Edit `create_pkl_from_h5_fif.py` in Pipeline 2
3. **Test changes**: Use `--dry-run` and `--limit` flags
4. **Update documentation**: Keep READMEs in sync

## References

- **Junifer**: https://juaml.github.io/junifer/
- **MNE-Python**: https://mne.tools/
- **BIDS**: https://bids.neuroimaging.io/
- **SLURM**: https://slurm.schedmd.com/

## Contact

For questions or issues:
- Check pipeline-specific READMEs
- Review SLURM logs
- Consult Junifer documentation

## License

These pipelines are part of the CYBERSART project.

---

**Last Updated**: October 8, 2025
**Pipeline Version**: 1.0
**Author**: Nicolas Bruno (with Copilot assistance)
