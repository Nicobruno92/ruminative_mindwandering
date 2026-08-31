# Code for: Disentangling the Phenomenology of Mind-Wandering

Multidimensional EEG characterization and decoding of ongoing thought.
Bruno et al., 2026.

This repository contains the analysis code behind the paper's methods,
figures, and tables. It is a **code-only** release; processed/raw data
are provided as a separate release (see the paper's Data Availability
statement).

## Layout

The folder layout mirrors the structure of the working research repository,
pruned to the code that generates the paper's reported results. Each
top-level folder corresponds to a stage of the pipeline described in the
paper's Methods section:

| Folder                                                    | Paper section                               | What it does                                                                                                                                      |
| --------------------------------------------------------- | ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `data_harmonization/`                                   | Methods 2.2-2.3                             | Raw BrainVision -> BIDS conversion, trial/probe metadata, resampling to 250 Hz                                                                    |
| `Preprocessing_pipeline_new/`                           | Methods 2.3                                 | EEG preprocessing: ASR, PyPREP bad-channel detection, ICA/ICLabel, epoching, AutoReject                                                           |
| `junifer_markers/`                                      | Methods 2.4, Table 1, Suppl. 9.2            | Extraction of the 23 EEG markers (evoked, spectral, information-theoretic, connectivity, slow-wave)                                               |
| `Behavior/Objective_Markers/`                           | Methods 2.5, Fig 2, Fig S1                  | Behavioral GLMMs (omission rate, commission rate, RTCV)                                                                                           |
| `Behavior/Extract_data_from_eeg/`                       | --                                          | Upstream probe/behavioral extraction from BIDS raw`events.tsv`                                                                                  |
| `Behavior/Probe_analysis/probe_dimension_cloud_plot.py` | Fig 1C/D                                    | Probe distributions and pairwise Spearman correlations                                                                                            |
| `Behavior/Demographics/`                                | --                                          | Sample demographics/psychometrics characterization                                                                                                |
| `Stats_CBPT/`, `Statistics/plot_results.py`           | Methods 2.6, Fig 3, Suppl. 9.3              | Cluster-based permutation test (CBPT) and cross-dimension omnibus test                                                                            |
| `mw_classification_pipeline/`                           | Methods 2.7, Fig 4-7, S2-S8, Suppl. 9.4-9.6 | Decoding analysis: within-subject and LOSO classification, residualized contrasts, spatial searchlight, SHAP importance, Type-I error calibration |
| `utils/`                                                | --                                          | Shared project utilities (BIDS I/O, epoch filtering)                                                                                              |
| `results/`                                              | --                                          | Final figures and source-data tables referenced above (not the full per-contrast diagnostic output)                                               |

## Setup

Each pipeline stage has its own dependencies; see the `README.md` (and, where
present, `requirements.txt`) inside each top-level folder.

Several scripts read raw BIDS data or reference external paths that are
placeholders in this release (`<PATH_TO_YOUR_...>`) -- set these to your own
environment before running. See the paper's Methods for expected data
layout, and `NOTES.md`-equivalent context in each folder's own README where
present.

## Citation

If you use this code, please cite:

Bruno, N., Guesdon, A., Leto, C., Poux, L., Fosatti, P., Andrillon, T.,
Tagliazucchi, E., Sitt, J., & Valero-Cabre, A. (2026). Disentangling the
phenomenology of mind-wandering: Multidimensional EEG characterization and
decoding of ongoing thought.
