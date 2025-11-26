CYBERSART Junifer Pipeline - State and Evoked Epochs
=====================================================

The pipeline has been split into two separate configs to handle different epoch types:

1. STATE EPOCHS (desc-state)
   - No baseline period (data starts at t=0)
   - Config: config_state.yaml
   - Elements: elements_state.csv
   - Output: markers_state.h5
   - Includes: Spectral power, connectivity, entropy, complexity

2. EVOKED EPOCHS (desc-evoked)
   - With baseline period ([-0.2, 0] seconds)
   - Config: config_evoked.yaml
   - Elements: elements_evoked.csv
   - Output: markers_evoked.h5
   - Includes: All state markers + ERP components (P1, N1, P2, P3a, P3b)

WORKFLOW
========

1. Generate element files (first time or after data changes):
   ./submit_slurm_array.sh --queue

   This will:
   - Run junifer queue for config_state.yaml → creates elements_state.csv
   - Run junifer queue for config_evoked.yaml → creates elements_evoked.csv

2. Submit both jobs:
   ./submit_slurm_array.sh

   This will:
   - Submit SLURM array for state epochs
   - Submit SLURM array for evoked epochs
   - Both run in parallel

3. Check/fix element files (optional):
   python manage_elements.py diagnose        # Check both
   python manage_elements.py diagnose state  # Check state only
   python manage_elements.py fix all         # Fix both

ENVIRONMENT VARIABLES
====================

You can override defaults:
  WORKDIR=/path/to/workdir
  CONDA_ENV=junifer
  CPUS=4
  MEM=8G
  TIME=08:00:00
  PARTITION=gpu  # optional

Example:
  CPUS=8 MEM=16G ./submit_slurm_array.sh

FILES STRUCTURE
===============

Config files:
  - config_state.yaml   : State epochs config (no baseline)
  - config_evoked.yaml  : Evoked epochs config (with baseline)

Element files (auto-generated):
  - elements_state.csv  : List of state epoch files to process
  - elements_evoked.csv : List of evoked epoch files to process

Scripts:
  - submit_slurm_array.sh  : Main submission script (handles both)
  - slurm_array_junifer.sh : SLURM worker script
  - manage_elements.py     : Diagnose/fix element files

Output:
  - markers_state.h5  : State epoch features
  - markers_evoked.h5 : Evoked epoch features
