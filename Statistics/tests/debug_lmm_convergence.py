"""Debug script to understand why LMM is not converging in tests."""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lmm_model import run_lmm_per_channel
from statsmodels.formula.api import mixedlm

# Recreate the test data
rng = np.random.RandomState(42)
n_subjects = 25
n_trials_per_subject = 50
n_channels = 10

subjects = np.repeat(np.arange(n_subjects), n_trials_per_subject)
trials = np.tile(np.arange(n_trials_per_subject), n_subjects)
onoff = rng.binomial(1, 0.5, size=len(subjects))

# More realistic power data with subject-specific baselines
power_data = np.zeros((len(subjects), n_channels))
for subj in range(n_subjects):
    subj_mask = subjects == subj
    # Subject-specific baseline (random intercept)
    subj_baseline = rng.randn() * 2.0
    # Add baseline + noise
    power_data[subj_mask] = subj_baseline + rng.randn(n_trials_per_subject, n_channels) * 3.0

# Add strong effect in channels 2-4 when onoff=1
effect_size = 1.5
for subj_idx in range(n_subjects):
    subj_mask = subjects == subj_idx
    onoff_mask = subj_mask & (onoff == 1)
    subject_effect = rng.randn() * 0.3
    power_data[onoff_mask, 2:5] += effect_size + subject_effect

df_behavioral = pd.DataFrame({
    'subject': subjects,
    'trial': trials,
    'onoff': onoff
})

print("=" * 80)
print("DATA STRUCTURE")
print("=" * 80)
print(f"Power data shape: {power_data.shape}")
print(f"Behavioral data shape: {df_behavioral.shape}")
print(f"Number of subjects: {df_behavioral['subject'].nunique()}")
print(f"Observations per subject: {len(df_behavioral) / df_behavioral['subject'].nunique():.1f}")
print(f"Onoff distribution: {df_behavioral['onoff'].value_counts().to_dict()}")
print(f"\nFirst few rows of behavioral data:")
print(df_behavioral.head(10))
print(f"\nPower data stats:")
print(f"  Mean: {power_data.mean():.3f}")
print(f"  Std: {power_data.std():.3f}")
print(f"  Min: {power_data.min():.3f}")
print(f"  Max: {power_data.max():.3f}")

# Try fitting a single channel manually
print("\n" + "=" * 80)
print("MANUAL LMM FIT FOR CHANNEL 0")
print("=" * 80)

ch_idx = 0
df_ch = df_behavioral.copy()
df_ch['power'] = power_data[:, ch_idx]
df_ch['subject'] = df_ch['subject'].astype(str)

print(f"Channel data prepared:")
print(f"  Shape: {df_ch.shape}")
print(f"  Columns: {list(df_ch.columns)}")
print(f"  Subject type: {df_ch['subject'].dtype}")
print(f"  Power stats: mean={df_ch['power'].mean():.3f}, std={df_ch['power'].std():.3f}")
print(f"\nFirst few rows:")
print(df_ch.head(10))

try:
    print("\nFitting LMM...")
    model = mixedlm(
        formula="power ~ onoff",
        data=df_ch,
        groups=df_ch["subject"]
    )
    
    result = model.fit(method='REML', maxiter=1000, disp=True)
    
    print("\n✓ Model converged successfully!")
    print(f"\nModel summary:")
    print(result.summary())
    
    print(f"\nT-statistic for 'onoff': {result.tvalues['onoff']:.3f}")
    print(f"P-value for 'onoff': {result.pvalues['onoff']:.6f}")
    
except Exception as e:
    print(f"\n✗ Model failed with error:")
    print(f"  {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# Now try with the full function
print("\n" + "=" * 80)
print("TESTING run_lmm_per_channel FUNCTION")
print("=" * 80)

formula = "power ~ onoff + (1|subject)"
predictor = "onoff"

print(f"Formula: {formula}")
print(f"Predictor: {predictor}")

t_stats, p_values, diagnostics = run_lmm_per_channel(
    power_data=power_data,
    df_behavioral=df_behavioral,
    formula=formula,
    predictor_of_interest=predictor,
    return_diagnostics=True
)

print(f"\nResults:")
print(f"  T-stats: {t_stats}")
print(f"  P-values: {p_values}")
print(f"\nDiagnostics:")
for key, value in diagnostics.items():
    if key not in ['convergence_warnings', 'aic', 'bic', 'log_likelihood', 
                   'conditional_r2', 'shapiro_p_values', 'residual_variance', 
                   'breusch_pagan_p']:
        print(f"  {key}: {value}")

if diagnostics['convergence_warnings']:
    print(f"\nFirst 3 convergence warnings:")
    for warning in diagnostics['convergence_warnings'][:3]:
        print(f"  - {warning}")
