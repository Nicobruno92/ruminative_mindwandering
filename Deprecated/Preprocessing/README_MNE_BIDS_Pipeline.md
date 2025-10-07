# MNE-BIDS-Pipeline Implementation

This directory contains scripts to run EEG preprocessing using the [MNE-BIDS-Pipeline](https://mne.tools/mne-bids-pipeline/stable/features/overview.html) instead of the custom robust preprocessing pipeline.

## Files

- `config_mne_bids_pipeline.py`: Configuration file for MNE-BIDS-Pipeline
- `run_mne_bids_wrapper.py`: Python wrapper script for running the pipeline
- `run_mne_bids_pipeline.sh`: SLURM batch script for cluster execution

## Key Differences from Robust Preprocessing Pipeline

### Advantages of MNE-BIDS-Pipeline

1. **Standardized Workflow**: Uses the established MNE-BIDS-Pipeline framework
2. **BIDS Compliance**: Automatically handles BIDS-compliant input/output
3. **Caching**: Built-in intelligent caching system
4. **Extensive Reporting**: Comprehensive HTML reports with interactive plots
5. **Community Support**: Well-maintained with regular updates
6. **Documentation**: Extensive documentation and examples

### Limitations Compared to Custom Pipeline

1. **Two-Copy ICA Strategy**: MNE-BIDS-Pipeline doesn't natively support the two-copy ICA approach
   - **Workaround**: Uses standard approach with different filters for ICA training vs analysis
   
2. **PREP Integration**: Limited direct integration with pyprep for bad channel detection
   - **Workaround**: Manual bad channel specification or use MNE's built-in methods

3. **Custom Trigger Processing**: Less flexibility for custom trigger correction
   - **Workaround**: Ensure BIDS events.tsv files are properly formatted

## Prerequisites

Install MNE-BIDS-Pipeline:

```bash
pip install mne-bids-pipeline
# or
conda install -c conda-forge mne-bids-pipeline
```

Ensure your data is in BIDS format with proper `events.tsv` files.

## Configuration

The configuration replicates the settings from the robust preprocessing pipeline:

- **Filtering**: 0.1 Hz high-pass for analysis, 1.0 Hz for ICA training, 40 Hz low-pass
- **Notch**: 50 Hz line noise removal
- **ICA**: Infomax algorithm with automatic artifact detection
- **Epoching**: -0.3 to 1.2 seconds around events
- **AutoReject**: Automated epoch rejection and interpolation
- **Reference**: Average reference

## Usage

### Single Subject/Task

```bash
python Preprocessing/run_mne_bids_wrapper.py 02 Sart1
```

### Batch Processing with SLURM

```bash
sbatch Preprocessing/run_mne_bids_pipeline.sh
```

The SLURM script processes all subjects (02-43) and all tasks (Sart1-4) using array jobs.

## Output

Results are saved to `/network/iss/cenir/analyse/meeg/CYBERSART/derivatives_mne_bids/` with:

- Preprocessed epochs in BIDS format
- ICA solutions
- AutoReject logs
- Comprehensive HTML reports
- Processing status CSV

## Monitoring Progress

Check the status CSV file:

```bash
cat /network/iss/cenir/analyse/meeg/CYBERSART/derivatives_mne_bids/preprocessing_status.csv
```

## Troubleshooting

### Common Issues

1. **BIDS Validation Errors**: Ensure your raw data follows BIDS conventions
2. **Missing Events**: Check that `events.tsv` files exist and contain go/nogo events
3. **Memory Issues**: Adjust `--mem` in SLURM script if needed
4. **Timeout**: Increase time limit in SLURM script for large datasets

### Logs

Check SLURM logs in the `logs/` directory:
- `logs/mne_bids_pipeline_[JOBID]_[ARRAYID].out`
- `logs/mne_bids_pipeline_[JOBID]_[ARRAYID].err`

## Comparison with Robust Pipeline

| Feature | Robust Pipeline | MNE-BIDS-Pipeline |
|---------|----------------|-------------------|
| Two-copy ICA | ✅ Native | ⚠️ Approximated |
| PREP integration | ✅ Full | ⚠️ Limited |
| Custom triggers | ✅ Flexible | ⚠️ BIDS-dependent |
| BIDS compliance | ⚠️ Manual | ✅ Automatic |
| Caching | ❌ Manual | ✅ Intelligent |
| Reporting | ⚠️ Basic | ✅ Comprehensive |
| Community support | ❌ Custom | ✅ Established |
| Maintenance | ❌ Manual | ✅ Community |

## Recommendations

1. **For Research**: Use MNE-BIDS-Pipeline for standardization and reproducibility
2. **For Custom Needs**: Use robust pipeline when specific preprocessing steps are required
3. **For Learning**: Start with MNE-BIDS-Pipeline to understand standard workflows
4. **For Production**: Consider MNE-BIDS-Pipeline for long-term maintainability

## Configuration Adjustments

Modify `config_mne_bids_pipeline.py` to adjust:

- Filter parameters
- ICA settings
- AutoReject thresholds
- Epoch timing
- Analysis channels
- Parallel processing settings

## Support

- [MNE-BIDS-Pipeline Documentation](https://mne.tools/mne-bids-pipeline/stable/)
- [MNE Community Forum](https://mne.discourse.group/)
- [GitHub Issues](https://github.com/mne-tools/mne-bids-pipeline/issues) 