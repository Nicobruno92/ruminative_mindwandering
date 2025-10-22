# SART Trial Extraction from BIDS

## Overview

This script extracts trial-level behavioral data from SART tasks in BIDS format. It processes the `events.tsv` files from EEG recordings and creates structured CSV files with behavioral information for each subject and task.

## Output Format

For each subject and task, a CSV file is created at:
```
sub-XX/beh/sub-XX_task-SartX.csv
```

### Output Columns

- **subject**: Subject ID (integer)
- **sart**: Task name (e.g., 'Sart1', 'Sart2', etc.)
- **trial_number**: Sequential trial number within the task
- **distance_to_probe**: Number of trials until the next probe (negative values)
- **trial_class**: Trial type ('go' or 'nogo')
- **response**: Boolean indicating if a button press occurred
- **rt**: Reaction time in milliseconds (if response occurred)
- **correct**: Boolean indicating correct performance
  - For 'go' trials: correct if response present
  - For 'nogo' trials: correct if no response
- **probe_number**: Identifier for the upcoming probe
- **onoff**: Probe rating for on-task/off-task (0-100 scale)
- **valence**: Probe rating for emotional valence (0-100 scale)
- **time**: Probe rating for temporal focus (0-100 scale)
- **selfother**: Probe rating for self/other focus (0-100 scale)
- **confidence**: Probe rating for confidence (0-100 scale)

## Usage

### On the Cluster (Recommended)

```bash
# Submit the batch job
cd /network/iss/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering/Behavior
sbatch run_extract_sart_trials.sh
```

### Local/Interactive Execution

```bash
# Process all subjects and tasks
python extract_sart_trials_from_bids.py

# Process specific subjects
python extract_sart_trials_from_bids.py --subjects 02 03 04

# Process specific tasks
python extract_sart_trials_from_bids.py --tasks Sart1 Sart2

# Test mode (first subject/task only)
python extract_sart_trials_from_bids.py --test-mode

# Custom paths
python extract_sart_trials_from_bids.py \
    --bids-raw-root /path/to/raw \
    --bids-output-root /path/to/output
```

## How It Works

1. **Parse Events**: Reads `events.tsv` files from BIDS raw data
2. **Identify Trials**: Separates stimulus events (go/nogo) from response events
3. **Match Responses**: Links responses to stimuli based on timing (within 2s window)
4. **Extract Probe Info**: Parses probe ratings from trial_type strings
5. **Compute Metrics**: Calculates correctness, RTs, and distance to probes
6. **Fix Distance Values**: Recalculates distance_to_probe to count forward to the NEXT probe (raw values count backward from previous probe)
7. **Save BIDS Format**: Outputs structured CSV files in BIDS behavioral directory

## Input Data Structure

The script expects events.tsv files with trial_type strings formatted as:
```
go/correct/onoff99/selfother81/valence97/time46/confidence99/average81/onoffBin1/selfotherBin1/valenceBin1/timeBin0/confidenceBin1/averageBin1/gonogoBin1/-30/probe1/ontask
```

Response events are formatted as:
```
response/correct/rt551
```

## Validation

The script performs automatic validation and reports:
- Total trials extracted
- Trial class distribution (go/nogo)
- Response rate and accuracy
- Reaction time statistics
- Missing value checks
- Probe information coverage

## Notes

- Responses are matched to stimuli within a 2-second window
- Probe information is propagated across trials leading up to each probe
- All probe ratings are extracted from the trial_type strings
- Output follows BIDS behavioral data conventions

## Related Scripts

- `extract_probes_from_bids.py`: Extracts probe-level data (one row per probe)
- This script: Extracts trial-level data (one row per trial)
