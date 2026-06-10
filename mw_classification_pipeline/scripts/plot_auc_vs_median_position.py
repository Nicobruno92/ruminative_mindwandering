#!/usr/bin/env python
"""
LOSO: AUC vs median position — exploratory analysis.

Two figures relating LOSO decodability (per-subject AUC) to within-subject
median ratings on the 5 phenomenology dimensions (onoff, valence, selfother,
time, confidence):

  Fig A (dimension_auc_vs_median_variability):
      One point per dimension. x = SD across subjects of each subject's
      within-subject median rating. y = mean LOSO AUC for that dimension.

  Fig B (auc_vs_median_distance_from_50_faceted):
      One panel per dimension, one point per subject. x = |subject's median
      rating - 50| (distance from scale midpoint). y = subject's LOSO AUC.

Status: EXPLORATORY (n=5 dimensions for Fig A; no confirmatory claims).

Usage (from project root):
    /path/to/miniforge3/envs/ML/bin/python \
        mw_classification_pipeline/scripts/plot_auc_vs_median_position.py
"""

# =============================================================================
# Imports
# =============================================================================

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.plotting_utils import COLORS  # noqa: E402


# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOSO_RESULTS_ROOT = (
    PROJECT_ROOT / "mw_classification_pipeline" / "results" / "MW_Classification" / "LOSO"
)
PROBE_DATA_PATH = (
    PROJECT_ROOT / "results" / "Behavior" / "probe_data" / "probe_level_aggregated_data.csv"
)
OUTPUT_DIR = LOSO_RESULTS_ROOT / "median_position_analysis"

SCALE_MIDPOINT = 50.0
CHANCE_AUC = 0.5
MIN_POINTS_FOR_FIT = 3

# (contrast directory name, display label, marker color) — color scheme
# matches scripts/generate_combined_classification_figure.py
DIMENSIONS: List[Dict[str, str]] = [
    {"contrast": "ON_vs_OFF_within_median", "label": "On/Off-Task", "color": "#DE237B"},
    {"contrast": "valence_within_median", "label": "Valence", "color": "#7B4FBA"},
    {"contrast": "selfother_within_median", "label": "Self/Other", "color": "#E67E22"},
    {"contrast": "confidence_within_median", "label": "Confidence", "color": "#27AE60"},
    {"contrast": "time_within_median", "label": "Time", "color": "#2980B9"},
]
