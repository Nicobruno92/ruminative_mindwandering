# Data Harmonization Pipeline

## Overview
This pipeline represents the first step in the CyberSART EEG data processing workflow. It reads raw BrainVision EEG data, applies marker correction/harmonization, sets up the expected montage (e.g., CACS-64), optionally resamples the data, and writes BIDS-compliant output (`.fif` format to preserve montage/electrode positions) ready for the main deterministic preprocessing pipeline.

## How It Works (Funcionamiento)
The pipeline is designed to solve a few common issues with raw EEG recordings: correcting missing or poorly formatted annotations, embedding electrode positions directly into the file, standardizing output names using the BIDS format, and unifying sampling frequencies before complex preprocessing.

Here is the step-by-step internal workflow when you run `data_harmonization.py`:

1. **Data Discovery (`io_helpers.py` > `DataFinder`)**:
   - The script scans the defined `data_root` looking for `.vhdr` BrainVision header files matching the pattern given in `config.yaml` (e.g., `CYBERSART_*_{task}.vhdr`).
   - It intelligently handles zero-padded subject names (e.g. `sub-02`, `sub-S002`, `sub-S02`) to find the corresponding `.vhdr` files.

2. **Safe Reading of Raw Data**:
   - BrainVision `.vhdr` files sometimes have internal pointers to `.eeg` and `.vmrk` files that use relative paths or incorrect casing. `read_brainvision_safe` creates a temporary, cleaned `.vhdr` header that resolves these paths to absolute paths so `mne.io.read_raw_brainvision` never fails due to malformed header references.

3. **Applying Montages (`io_helpers.py` > `MontageHelper`)**:
   - The original BrainVision files lack 3D positions for the electrodes. 
   - The pipeline applies a `.bvef` custom montage file (like `CACS-64_REF.bvef`) specifically used for this study.
   - It also automatically identifies auxiliary channels like `VEOG` and `HEOG` and correctly tags their MNE channel type as `eog`.

4. **Trigger Correction & Event Harmonization (`trigger_correction.py`)**:
   - This is the most complex step. `TriggerCorrector` iterates over all the annotations (markers) in the raw recording and processes them over multiple passes:
     - **Trial tagging**: Re-labels standard markers (e.g., `S 41`, `S 42`) into `go` and `nogo` trials.
     - **Accuracy & RT logic**: Looks at the next marker to determine if the participant answered correctly (`S 44`) or incorrectly (`S 45`). It computes the reaction time (RT) in milliseconds and attaches it to the response tag (e.g. `response/correct/rt345`). It also captures second button presses.
     - **Thought Probe Processing**: It identifies `S 31` through `S 36` markers as thought probe questions (e.g., on-task/off-task, self/other, valence, time, confidence). It aggregates the answers (which come from markers `S 100` to `S 200`) into a single continuous scale (0-100).
     - **Retro-propagation**: The pipeline rolls the continuous thought-probe values *backwards* onto all the preceding trials that occurred since the *last* probe.
     - **Binarization**: Based on the `thought_thresholds` defined in `config.yaml` (usually 50), it tags each trial and probe with binary labels too (e.g., `onoffBin1`, `gonogoBin0`, `ONTASK`, `OFFTASK`).
     - **Distance from Probe**: It numbers the trials strictly based on how far away they are from the probe (e.g. `-5`, `-1`) and attaches the ordinal probe index (`probe15`).
   - All these modifications replace the old event descriptions *without altering the original onset times, durations, or absolute `orig_time`*, guaranteeing temporal alignment.

5. **Rest Block Processing**:
   - The script automatically finds the rest periods (starting at `S 31` and ending at the next `S 41`/`S 42`/`S 43`) and explicitly adds a `BAD_rest` annotation spanning that entire time block, making it easy to drop these segments during downstream ICA.

6. **Resampling**:
   - The script checks the raw sampling frequency against `resample_hz` set in `config.yaml` (default `500`).
   - If it differs, the continuous raw data is loaded into memory and resampled using MNE's high-quality polyphase resampler to guarantee data homogeneity across all recording centers.

7. **BIDS Export (`bids_compliance_harmonized.py`)**:
   - Armed with standardized annotations and a 3D montage, the pipeline exports the result.
   - Importantly, it **does not save as BrainVision again**. To preserve the newly mapped custom 3D electrode positions safely, it exports the raw continuous data into standard MNE `.fif` files inside the `BIDS/raw/` folder structure.
   - During export, it creates the BIDS-compliant JSON sidecars, `channels.tsv`, `events.tsv`, and generates `dataset_description.json`.

8. **Quality Assurance (QA) Reporting (`qa_report.py`)**:
   - The script creates cumulative `harmonization_qa_report` artifacts (`.html`, `.json`, and `.csv`) in the output directory.
   - It summarizes whether the harmonization was a `success`, `skipped`, or resulted in an `error`, reporting the final number of annotations, EEG channels, and generated files, allowing operators to easily visually assess the validity of the batch.

## Key Components
- `data_harmonization.py`: Main entry point. Orchestrates the process using the YAML config.
- `BIDSComplianceHarmonized` (in `utils/bids_compliance_harmonized.py`): Unified BIDS I/O handler. Writes to `.fif` inside the BIDS/raw folder, generates extensive JSON sidecars and `channels.tsv`/`events.tsv`.
- `io_helpers.py`: Contains `DataFinder`, `MontageHelper`, `EventRecoder`, and `read_brainvision_safe`.
- `trigger_correction.py`: Contains `TriggerCorrector` for the entire annotation string manipulation sequence.
- `qa_report.py`: Generates comprehensive HTML, CSV, and JSON QA reports.

## Requirements
- Python 3.10+
- `mne`, `mne-bids`, `pyyaml`, `pandas`, `numpy`

## Configuration (`config.yaml`)
Driven by `config.yaml` located within `data_harmonization/`. Important keys include:
- `project`: Paths for `data_root`, `raw_root` (BIDS output folder), `brainvision_pattern`, and string references to montages setups (`CACS-64_REF.bvef`).
- `annotation_harmonization`: Threshold indicators (defaulting to 50) and choice of scale dimension to assign the master `ONTASK` and `OFFTASK` tags.
- `thought_thresholds`: Thresholds for each dimension to binarize ratings.
- `resample_hz`: Target sampling rate (e.g. 500 Hz).

## Execution
Run on a single subject/task:
```bash
python data_harmonization.py --config config.yaml --subject 04 --task Sart1
```
Run batch across the entire cluster iteratively via SLURM array:
```bash
sbatch run_harmonize_only_slurm.sh
```
Run locally iteratively (for all configured subjects/tasks):
```bash
python data_harmonization.py --config config.yaml
```

## Detailed File References
For specific derivatives nomenclature, layout logic, and BIDS entity naming standards applied downstream or output by this phase, review `FILE_TYPES_REFERENCE.md`. If maintaining the core annotation tagging conventions directly, see `utils/ANNOTATION_HARMONIZATION.md`.
