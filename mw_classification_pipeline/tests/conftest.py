"""Shared fixtures for spatial decoding tests. Synthetic data only."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Allow `from utils.spatial_decoding_utils import ...`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def make_per_channel_X(channels, markers, n_samples=40, random_seed=42):
    """Build a synthetic per-channel feature matrix with `{channel}_{marker}` columns."""
    rng = np.random.default_rng(random_seed)
    cols = [f"{ch}_{mk}" for ch in channels for mk in markers]
    data = rng.standard_normal((n_samples, len(cols)))
    return pd.DataFrame(data, columns=cols)


@pytest.fixture
def per_channel_X():
    # Includes P1 and P10 to guard against substring matching bugs.
    return make_per_channel_X(
        channels=["Fz", "Pz", "P1", "P10"],
        markers=["psd_bands_delta_mean", "P3b", "slowwaves_density"],
    )
