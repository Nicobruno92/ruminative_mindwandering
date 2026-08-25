#!/usr/bin/env python
"""
Run generate_pipeline_plots.py for all LOSO and WithinSubject result families.

USAGE (from project root):
    python mw_classification_pipeline/scripts/run_all_plots.py
"""

import subprocess
import sys
import os

# Full path to the ML conda environment Python — avoids conda activation issues
# on systems where `conda run` or sys.executable resolves to system Python 3.6.
_CONDA_PYTHON = "<PATH_TO_YOUR_ML_CONDA_ENV>/bin/python"
PYTHON_EXE = _CONDA_PYTHON if os.path.isfile(_CONDA_PYTHON) else sys.executable

# =============================================================================
# CONFIGURATION - modify if result dirs change
# =============================================================================
ROOT = os.path.join(
    os.path.dirname(__file__), "..", "results", "MW_Classification"
)
ROOT = os.path.abspath(ROOT)

FAMILY_DIRS = [
    os.path.join(ROOT, "LOSO", "ON_vs_OFF_global_median", "all"),
    os.path.join(ROOT, "LOSO", "ON_vs_OFF_global_median", "connectivity"),
    os.path.join(ROOT, "LOSO", "ON_vs_OFF_global_median", "information_theory"),
    os.path.join(ROOT, "LOSO", "ON_vs_OFF_global_median", "spectral"),
    os.path.join(ROOT, "LOSO", "ON_vs_OFF_within_median", "all"),
    os.path.join(ROOT, "LOSO", "ON_vs_OFF_within_median", "connectivity"),
    os.path.join(ROOT, "LOSO", "ON_vs_OFF_within_median", "information_theory"),
    os.path.join(ROOT, "LOSO", "ON_vs_OFF_within_median", "spectral"),
    os.path.join(ROOT, "WithinSubject", "on_vs_off_within_median", "all"),
    os.path.join(ROOT, "WithinSubject", "on_vs_off_within_median", "connectivity"),
    os.path.join(ROOT, "WithinSubject", "on_vs_off_within_median", "spectral"),
]

TOP_N_FEATURES = 20
SCRIPT = os.path.join(os.path.dirname(__file__), "generate_pipeline_plots.py")

# =============================================================================
# RUNNER
# =============================================================================

def run_plots_for_dir(family_dir: str) -> bool:
    """
    Run generate_pipeline_plots.py for a single family directory.

    Parameters
    ----------
    family_dir : str
        Path to the family-level directory (parent of model dirs like lr/, rf/).

    Returns
    -------
    bool
        True if the script completed without errors.
    """
    label = "/".join(os.path.normpath(family_dir).split(os.sep)[-2:])
    print(f"\n{'=' * 60}", flush=True)
    print(f"  {label}", flush=True)
    print("=" * 60, flush=True)

    if not os.path.isdir(family_dir):
        print(f"  SKIP – directory does not exist: {family_dir}", flush=True)
        return False

    result = subprocess.run(
        [PYTHON_EXE, SCRIPT, "--results_dir", family_dir,
         "--top_n_features", str(TOP_N_FEATURES)],
        capture_output=False,  # stream output directly
        text=True,
    )
    return result.returncode == 0


def main() -> None:
    failures = []
    for family_dir in FAMILY_DIRS:
        success = run_plots_for_dir(family_dir)
        if not success:
            failures.append(family_dir)

    print(f"\n{'=' * 60}", flush=True)
    if failures:
        print(f"COMPLETED with {len(failures)} failure(s):")
        for f in failures:
            print(f"  FAILED: {f}")
    else:
        print(f"All {len(FAMILY_DIRS)} family directories processed successfully.")
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
