"""Tests for scripts/plot_auc_vs_median_position.py. Synthetic data + real LOSO fixtures."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.plot_auc_vs_median_position import LOSO_RESULTS_ROOT
