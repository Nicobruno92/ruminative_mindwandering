================================================================================
ANNOTATION HARMONIZATION - MAINTENANCE DOCUMENTATION
================================================================================

This document describes the annotation harmonization system for the CyberSART
EEG study. The goal is to standardize event annotations so that preprocessing
and marker analysis scripts work consistently across all subjects.

================================================================================
1. INPUT FORMAT (Original Annotations from BrainVision)
================================================================================

Raw stimulus codes from BrainVision:
  - Stimulus/S 41, S 43: Go trials
  - Stimulus/S 42: NoGo trials
  - Stimulus/S 44: Correct response (button press after go)
  - Stimulus/S 45: Incorrect response (button press after nogo)
  - Stimulus/S 31: First thought probe (onoff dimension)
  - Stimulus/S 32: selfother probe
  - Stimulus/S 33: time probe
  - Stimulus/S 34: valence probe
  - Stimulus/S 35: confidence probe
  - Stimulus/S 36: average probe (end of probe sequence)
  - Stimulus/S 100-200: Probe responses (value = code - 100)

================================================================================
2. INTERMEDIATE FORMAT (After TriggerCorrector Processing)
================================================================================

Before harmonization, annotations look like:
  go/correct/onoff91/selfother51/valence72/time68/confidence76/average71/ontask/-5/probe5/12

Where:
  - go/nogo: Trial type
  - correct/incorrect: Response accuracy
  - onoff91: On-task/off-task rating (0-100 scale)
  - selfother51: Self vs other focus (0-100 scale)
  - valence72: Emotional valence (0-100 scale)
  - time68: Temporal focus (0-100 scale)
  - confidence76: Confidence rating (0-100 scale)
  - average71: Average immersion (0-100 scale)
  - ontask/offtask: Binary label (lowercase)
  - -5: Distance to probe (negative = before probe)
  - probe5: Probe number
  - 12: Trial number within segment

================================================================================
3. OUTPUT FORMAT (Harmonized Annotations)
================================================================================

Trial annotations are harmonized to:
  go/correct/probe-5/trial-12/distance-5/ONTASK/onoff91/selfother51/valence72/time68/confidence76/average71/ontask

Where:
  - go/nogo: Trial type (preserved)
  - correct/incorrect: Response accuracy (preserved)
  - probe-N: Probe number (standardized prefix with hyphen)
  - trial-N: Trial number within segment (standardized prefix)
  - distance-N: Distance to probe end (standardized prefix, always positive)
  - ONTASK/OFFTASK: Binary classification (uppercase, based on configurable threshold)
  - onoff91/selfother51/...: Original dimension values (preserved)
  - ontask: Original lowercase label (preserved)

Thought probe events are harmonized to:
  THOUGHT_PROBE/probe-5/ONTASK/onoff91/selfother51/valence72/time68/confidence76/average71

================================================================================
4. ONTASK/OFFTASK CLASSIFICATION
================================================================================

Configured in config.yaml under annotation_harmonization:

  annotation_harmonization:
    enabled: true
    ontask_threshold: 50
    ontask_dimension: "onoff"

Classification logic:
  - ONTASK = onoff >= ontask_threshold (default: >= 50)
  - OFFTASK = onoff < ontask_threshold (default: < 50)

The dimension used for classification can be changed via ontask_dimension.

================================================================================
5. THOUGHT PROBE DIMENSIONS (0-100 Scale)
================================================================================

All dimensions are continuous scales from 0 to 100:

  - onoff: On-task (100) vs Off-task (0)
  - selfother: Self-focused (100) vs Other-focused (0)
  - valence: Positive (100) vs Negative (0)
  - time: Future (100) vs Past (0)
  - confidence: High confidence (100) vs Low confidence (0)
  - average: High immersion (100) vs Low immersion (0)

================================================================================
6. FILES INVOLVED
================================================================================

config.yaml
  - annotation_harmonization.enabled: Enable/disable harmonization
  - annotation_harmonization.ontask_threshold: Threshold for ONTASK classification
  - annotation_harmonization.ontask_dimension: Dimension used for classification

utils/trigger_correction.py
  - TriggerCorrector class handles all annotation processing
  - harmonize_annotations(): Main harmonization function
  - _harmonize_trial(): Harmonizes go/nogo trial annotations
  - _harmonize_thought_probe(): Harmonizes THOUGHT_PROBE annotations
  - _get_ontask_label(): Determines ONTASK/OFFTASK based on threshold
  - _load_harmonization_config(): Loads config from YAML

data_harmonization.py
  - Calls TriggerCorrector.process_annotations() which includes harmonization

================================================================================
7. USAGE
================================================================================

Harmonization is applied automatically during data_harmonization.py if:
  annotation_harmonization:
    enabled: true

To disable, set enabled: false in config.yaml.

To run harmonization:
  python data_harmonization.py --config config.yaml

================================================================================
8. EXAMPLE TRANSFORMATIONS
================================================================================

Input (intermediate format):
  go/correct/onoff91/selfother51/valence72/time68/confidence76/average71/ontask/onoffBin1/selfotherBin1/valenceBin1/timeBin1/confidenceBin1/averageBin1/gonogoBin1/-5/probe5/12

Output (harmonized format):
  go/correct/probe-5/trial-12/distance-5/ONTASK/onoff91/selfother51/valence72/time68/confidence76/average71/ontask/onoffBin1/selfotherBin1/valenceBin1/timeBin1/confidenceBin1/averageBin1/gonogoBin1

Input (THOUGHT_PROBE):
  THOUGHT_PROBE/onoff91/selfother51/valence72/time68/confidence76/average71/probe5

Output (harmonized THOUGHT_PROBE):
  THOUGHT_PROBE/probe-5/ONTASK/onoff91/selfother51/valence72/time68/confidence76/average71

================================================================================
9. NOTES FOR MAINTENANCE
================================================================================

- Original annotation content is PRESERVED in the harmonized format
- The harmonization adds standardized prefixes (probe-, trial-, distance-)
- ONTASK/OFFTASK (uppercase) is a binary classification added for easy filtering
- The lowercase ontask/offtask labels from earlier processing are also preserved
- Non-trial annotations (responses, probe questions, etc.) are kept unchanged
- Binary labels (onoffBin1, etc.) are preserved for backward compatibility

To change ONTASK classification threshold:
  1. Edit ontask_threshold in config.yaml
  2. Re-run data_harmonization.py

To use a different dimension for ONTASK classification:
  1. Edit ontask_dimension in config.yaml (e.g., "confidence")
  2. Re-run data_harmonization.py

================================================================================
10. PARSING HARMONIZED ANNOTATIONS
================================================================================

To parse harmonized trial annotations in Python:

    parts = annotation.split('/')
    trial_type = parts[0]      # 'go' or 'nogo'
    correctness = parts[1]     # 'correct' or 'incorrect'
    
    for part in parts:
        if part.startswith('probe-'):
            probe_num = int(part[6:])
        elif part.startswith('trial-'):
            trial_num = int(part[6:])
        elif part.startswith('distance-'):
            distance = int(part[9:])
        elif part in ['ONTASK', 'OFFTASK']:
            ontask_status = part
        elif part.startswith('onoff'):
            onoff_value = int(''.join(filter(str.isdigit, part)))

================================================================================
