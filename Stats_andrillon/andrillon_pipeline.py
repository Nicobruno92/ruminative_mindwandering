"""
Andrillon 2020 Pipeline - Main Workflow

Complete pipeline implementing Andrillon et al. (2020) cluster-permutation methodology.

Workflow:
1. Load configuration
2. Load marker data (reuse reader.py from current pipeline)
3. Preprocess data (normalization)
4. For each marker:
   a. Fit LMM per electrode with permutations
   b. Detect clusters
   c. Apply multiple comparisons correction
5. Save results
6. Generate visualizations
"""

import sys
import os
import yaml
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
import mne
from scipy import sparse
from scipy import stats as sp_stats
from joblib import Parallel, delayed

# Add Statistics directory to path to import existing modules
stats_dir = Path(__file__).parent.parent / "Statistics"
sys.path.insert(0, str(stats_dir))

# Import ALL functions from the proven Statistics pipeline
from reader import load_all_probe_data, prepare_data_for_lmm, get_available_markers
from cluster_test import get_channel_adjacency, find_clusters_from_pvalues
from lmm_model import (
    run_lmm_per_channel,
    fit_reduced_model_per_channel,
    freedman_lane_permutation,
)
from helpers import get_model_folder_name, normalize_by_subject

# Import from local Stats_andrillon modules
from plot_results import create_results_report, create_raw_topography_report
from generate_summary_report import generate_summary_report

# Threshold for flagging an individual permutation as "degenerate" — i.e.
# the LMM failed on a majority of channels. Mirrors the
# `abort_on_high_failure` cutoff inside run_lmm_per_channel. If a substantial
# fraction of permutations cross this threshold (see DEGENERATE_RATE_WARN),
# the simple permutation scheme is breaking the design matrix and the user
# should switch to Freedman-Lane.
DEGENERATE_RATE_THRESHOLD = 0.5
DEGENERATE_RATE_WARN = 0.01  # warn if >1% of permutations are degenerate

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    logger.info(f"Loading configuration from {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def _resolve_formula_for_predictor(config: Dict, predictor: str) -> str:
    """
    Return the formula to use for a given predictor of interest.

    Checks ``lmm.per_predictor_formula`` for an override keyed by the
    predictor name.  Falls back to ``lmm.formula`` when no override exists.

    This lets interaction predictors (e.g. ``"onoff:confidence"``) specify a
    multiplicative formula (``power ~ onoff * confidence + ...``) while
    main-effect predictors keep the default additive formula.

    Parameters
    ----------
    config : dict
        Pipeline configuration.
    predictor : str
        The predictor of interest (e.g. ``"onoff"`` or ``"onoff:confidence"``).

    Returns
    -------
    str
        Formula string to pass to ``prepare_data_for_lmm`` and
        ``run_lmm_per_channel``.

    Raises
    ------
    ValueError
        If a ``per_predictor_formula`` override exists for the predictor but
        the predictor (or its components) are absent from that formula — which
        would make the test nonsensical.
    """
    default_formula = config['lmm']['formula']
    overrides = config['lmm'].get('per_predictor_formula') or {}
    formula = overrides.get(predictor, default_formula)

    # Validate that the chosen formula actually contains the predictor.
    # For interaction terms (e.g. "onoff:confidence") check that at least the
    # multiplicative shorthand "*" or the explicit ":" form is present.
    if ':' in predictor:
        components = [v.strip() for v in predictor.split(':')]
        has_star = all(c in formula for c in components) and (
            '*' in formula or predictor in formula
        )
        if not has_star:
            raise ValueError(
                f"Formula override for predictor '{predictor}' does not "
                f"contain the interaction term. Formula: '{formula}'. "
                f"Add '{components[0]} * {components[1]}' or the explicit "
                f"'{predictor}' term to the formula."
            )
    elif predictor not in formula:
        raise ValueError(
            f"Formula for predictor '{predictor}' does not contain the "
            f"predictor as a term. Formula: '{formula}'."
        )

    return formula


def _get_derived_ratios_lookup(config: Dict) -> Dict[str, Dict]:
    """
    Build a lookup `{ "<marker_type>/<name>": spec }` for derived ratio markers
    declared in `config.derived_ratios`. A ratio spec has keys:
      - name        : output marker name (e.g. "psd_bands_theta_beta_ratio")
      - marker_type : type the ratio belongs to ("sleep", "evoked", ...)
      - numerator   : base marker name to use as numerator
      - denominator : base marker name to use as denominator

    Ratios are computed at load time by channel-wise division of the numerator
    and denominator arrays for matching (subject, task, probe) observations.
    """
    ratios = config.get('derived_ratios', []) or []
    lookup: Dict[str, Dict] = {}
    for spec in ratios:
        required = {'name', 'marker_type', 'numerator', 'denominator'}
        missing = required - set(spec.keys())
        if missing:
            raise ValueError(
                f"derived_ratios entry is missing required keys {missing}: {spec}"
            )
        key = f"{spec['marker_type']}/{spec['name']}"
        if key in lookup:
            raise ValueError(f"Duplicate derived_ratios entry for {key}")
        lookup[key] = spec
    return lookup


def get_marker_list(config: Dict) -> List[str]:
    """
    Get list of markers to analyze based on config.

    Returns list of marker names based on config.project.markers setting plus
    any derived ratio markers declared in config.derived_ratios. Uses
    get_available_markers from Statistics/reader.py (proven implementation).
    """
    markers_setting = config['project']['markers']
    features_root = config['project']['features_root']
    subjects = config['project']['subjects']
    tasks = config['project']['tasks']
    marker_types = config['project']['marker_types']

    # Derived ratios are appended to whatever marker list is in use.
    derived_keys = list(_get_derived_ratios_lookup(config).keys())

    if markers_setting == 'all':
        # Auto-discover all available markers using proven function
        markers = get_available_markers(
            features_root=features_root,
            marker_types=marker_types
        )

        # Format as "marker_type/marker_name" for consistency
        formatted_markers = []
        for marker_type, marker_names in markers.items():
            for marker_name in marker_names:
                # Exclude pair-wise matrix markers to avoid memory & channel count issues
                if marker_name.endswith('_pairs'):
                    continue
                formatted_markers.append(f"{marker_type}/{marker_name}")

        formatted_markers.extend(derived_keys)
        logger.info(
            f"Auto-discovered {len(formatted_markers) - len(derived_keys)} markers "
            f"across {len(markers)} types + {len(derived_keys)} derived ratios"
        )
        return formatted_markers
    else:
        # Use specified markers from config plus derived ratios
        return list(markers_setting) + derived_keys


def _load_ratio_marker_data(
    spec: Dict,
    config: Dict,
    formula: str,
    predictor: str,
) -> Tuple[np.ndarray, pd.DataFrame, List[str]]:
    """
    Load numerator and denominator markers and return their channel-wise ratio.

    Same return signature as `prepare_data_for_lmm`:
        power_ratio : (n_obs, n_ch) array of numerator / denominator
        df_behavioral : behavioral metadata for the aligned observations
        channels : channel names corresponding to columns of power_ratio

    Observations are aligned by (subject, task, probe_number) via inner join —
    a probe present only in the numerator or only in the denominator is dropped
    so the ratio is always defined. Channels are inner-joined for the same reason.
    NaN propagates through the division; explicit zero in the denominator
    becomes NaN (we set inf/-inf to NaN) so excluded channels stay excluded.

    `marker_type` is NOT used as a merge key — both numerator and denominator
    have their own marker_type column from the source files, so merging on it
    would create `_x` / `_y` collisions and split observations spuriously.
    """
    num_name = spec['numerator']
    den_name = spec['denominator']
    marker_type = spec['marker_type']

    preprocessing_cfg = config.get('preprocessing', {})
    project_cfg = config.get('project', {})
    onoff_max_value = preprocessing_cfg.get(
        'onoff_max_value', project_cfg.get('onoff_max_value')
    )
    min_predictor_variability = preprocessing_cfg.get(
        'min_predictor_variability',
        project_cfg.get('min_predictor_variability'),
    )

    df_all = load_all_probe_data(
        features_root=config['project']['features_root'],
        subjects=config['project'].get('subjects'),
        tasks=config['project'].get('tasks'),
        marker_types=[marker_type],
        specific_markers=[num_name, den_name],
        behavioral_data_path=config['project'].get('behavioral_data_path'),
    )
    df_all = df_all[df_all['marker_type'] == marker_type]

    df_num = df_all[df_all['marker'] == num_name]
    df_den = df_all[df_all['marker'] == den_name]
    if df_num.empty:
        raise ValueError(
            f"Derived ratio '{spec['name']}': numerator '{num_name}' "
            f"not found for marker_type='{marker_type}'."
        )
    if df_den.empty:
        raise ValueError(
            f"Derived ratio '{spec['name']}': denominator '{den_name}' "
            f"not found for marker_type='{marker_type}'."
        )

    power_num, df_beh_num, channels_num = prepare_data_for_lmm(
        df=df_num,
        marker_name=num_name,
        formula=formula,
        include_channels=preprocessing_cfg.get('include_channels'),
        exclude_channels=preprocessing_cfg.get('exclude_channels'),
        onoff_max_value=onoff_max_value,
        min_predictor_variability=min_predictor_variability,
        predictor_of_interest=predictor,
    )
    power_den, df_beh_den, channels_den = prepare_data_for_lmm(
        df=df_den,
        marker_name=den_name,
        formula=formula,
        include_channels=preprocessing_cfg.get('include_channels'),
        exclude_channels=preprocessing_cfg.get('exclude_channels'),
        onoff_max_value=onoff_max_value,
        min_predictor_variability=min_predictor_variability,
        predictor_of_interest=predictor,
    )

    # Inner-join channels in alphabetical order so the result is deterministic.
    common_channels = sorted(set(channels_num) & set(channels_den))
    if not common_channels:
        raise ValueError(
            f"No common channels between numerator '{num_name}' "
            f"and denominator '{den_name}'."
        )
    num_ch_idx = [channels_num.index(c) for c in common_channels]
    den_ch_idx = [channels_den.index(c) for c in common_channels]
    power_num = power_num[:, num_ch_idx]
    power_den = power_den[:, den_ch_idx]

    # Inner-join rows by (subject, task, probe_number).
    def _obs_key(df: pd.DataFrame) -> pd.Series:
        return (
            df['subject'].astype(str)
            + '|' + df['task'].astype(str)
            + '|' + df['probe_number'].astype(str)
        )

    df_beh_num = df_beh_num.copy()
    df_beh_den = df_beh_den.copy()
    df_beh_num['_obs_key'] = _obs_key(df_beh_num)
    df_beh_den['_obs_key'] = _obs_key(df_beh_den)

    common_keys = sorted(
        set(df_beh_num['_obs_key']) & set(df_beh_den['_obs_key'])
    )
    if not common_keys:
        raise ValueError(
            f"No common (subject, task, probe) observations between "
            f"numerator '{num_name}' and denominator '{den_name}'."
        )

    num_pos = {k: i for i, k in enumerate(df_beh_num['_obs_key'])}
    den_pos = {k: i for i, k in enumerate(df_beh_den['_obs_key'])}
    num_positions = [num_pos[k] for k in common_keys]
    den_positions = [den_pos[k] for k in common_keys]

    power_num_aligned = power_num[num_positions, :]
    power_den_aligned = power_den[den_positions, :]

    # Channel-wise ratio. 0/0 → NaN; x/0 → +/-inf → NaN (excluded from cluster).
    with np.errstate(divide='ignore', invalid='ignore'):
        power_ratio = power_num_aligned / power_den_aligned
    power_ratio[~np.isfinite(power_ratio)] = np.nan

    df_behavioral = (
        df_beh_num.iloc[num_positions]
        .copy()
        .reset_index(drop=True)
        .drop(columns=['_obs_key'])
    )

    n_finite = int(np.sum(np.isfinite(power_ratio)))
    logger.info(
        "  Derived ratio %s = %s / %s: %d obs × %d channels (%.1f%% finite values)",
        spec['name'], num_name, den_name,
        power_ratio.shape[0], power_ratio.shape[1],
        100.0 * n_finite / max(1, power_ratio.size),
    )

    return power_ratio, df_behavioral, list(common_channels)


def run_marker_analysis(
    marker_name: str,
    config: Dict,
    montage_path: str,
) -> Dict:
    """
    Run complete analysis for a single marker.
    
    Parameters
    ----------
    marker_name : str
        Name of the marker to analyze
    config : dict
        Configuration dictionary
    montage_path : str
        Path to montage file for building adjacency matrix
        
    Returns
    -------
    results : dict
        Analysis results including clusters and statistics
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Analyzing marker: {marker_name}")
    logger.info(f"{'='*80}\n")
    
    # 1. Load data
    logger.info("Step 1: Loading marker data...")

    # Parse marker_name to extract type and name
    # Format: "marker_type/marker_name" (e.g., "evoked/wsmi_theta")
    if '/' in marker_name:
        marker_type, marker_base_name = marker_name.split('/', 1)
    else:
        # Fallback: assume no type prefix
        marker_type = None
        marker_base_name = marker_name

    logger.info(f"  Marker type: {marker_type}")
    logger.info(f"  Marker name: {marker_base_name}")

    # Preprocessing & project config used below (shared between regular and
    # derived ratio code paths).
    preprocessing_cfg = config.get('preprocessing', {})
    project_cfg = config.get('project', {})
    onoff_max_value = preprocessing_cfg.get('onoff_max_value', project_cfg.get('onoff_max_value'))
    min_predictor_variability = preprocessing_cfg.get('min_predictor_variability', project_cfg.get('min_predictor_variability'))

    # Detect whether this is a derived ratio marker. The ratio lookup is built
    # from `config.derived_ratios` and keyed by "<marker_type>/<name>".
    ratio_lookup = _get_derived_ratios_lookup(config)
    ratio_spec = ratio_lookup.get(marker_name)

    if ratio_spec is not None:
        logger.info(
            f"  Derived ratio detected: {ratio_spec['numerator']} / "
            f"{ratio_spec['denominator']}"
        )
        _effective_formula = _resolve_formula_for_predictor(
            config, config['lmm']['predictor_of_interest']
        )
        power_data, df_behavioral, channels = _load_ratio_marker_data(
            spec=ratio_spec,
            config=config,
            formula=_effective_formula,
            predictor=config['lmm']['predictor_of_interest'],
        )
    else:
        # Standard single-marker loading path.
        marker_types_to_load = config['project']['marker_types']
        if marker_type is not None:
            marker_types_to_load = [marker_type]
        df_all = load_all_probe_data(
            features_root=config['project']['features_root'],
            subjects=config['project'].get('subjects'),
            tasks=config['project'].get('tasks'),
            marker_types=marker_types_to_load,
            specific_markers=[marker_base_name],
            behavioral_data_path=project_cfg.get('behavioral_data_path'),
        )

        # Filter by marker type if specified
        if marker_type:
            logger.info(f"  Filtering data for marker_type: {marker_type}")
            df_filtered = df_all[df_all['marker_type'] == marker_type]
            logger.info(f"  Filtered from {len(df_all)} to {len(df_filtered)} rows")
        else:
            df_filtered = df_all

        power_data, df_behavioral, channels = prepare_data_for_lmm(
            df=df_filtered,
            marker_name=marker_base_name,
            formula=_resolve_formula_for_predictor(
                config, config['lmm']['predictor_of_interest']
            ),
            include_channels=preprocessing_cfg.get('include_channels'),
            exclude_channels=preprocessing_cfg.get('exclude_channels'),
            onoff_max_value=onoff_max_value,
            min_predictor_variability=min_predictor_variability,
            predictor_of_interest=config['lmm']['predictor_of_interest'],
        )

    # Ensure subject is string type (critical for statsmodels mixedlm)
    df_behavioral['subject'] = df_behavioral['subject'].astype(str)

    logger.info(f"  Loaded {power_data.shape[0]} observations × {power_data.shape[1]} channels")

    # Apply per-subject normalization if requested. Without it, between-subject
    # scale differences propagate into the LMM residual variance and bias the
    # slope estimate of the predictor toward subjects with the largest amplitude,
    # producing spatially extended clusters that reflect individual baseline
    # topographies rather than the predictor effect.
    if preprocessing_cfg.get('normalize_by_subject', False):
        norm_method = preprocessing_cfg.get('normalization_method', 'zscore')
        norm_channel_wise = preprocessing_cfg.get('channel_wise', False)
        norm_subject_col = preprocessing_cfg.get('subject_column', 'subject')
        logger.info(
            f"  Normalizing power per subject: method={norm_method}, "
            f"channel_wise={norm_channel_wise}"
        )
        power_data = normalize_by_subject(
            power_data=power_data,
            df_behavioral=df_behavioral,
            method=norm_method,
            subject_col=norm_subject_col,
            channel_wise=norm_channel_wise,
            verbose=True,
        )

    # Create output directory consistent with main Statistics pipeline
    output_root = Path(config['project']['output_path'])

    predictor = config['lmm'].get('predictor_of_interest', 'auto')
    model_folder = get_model_folder_name(
        _resolve_formula_for_predictor(config, predictor), predictor
    )

    if marker_type:
        safe_marker_name = marker_base_name.replace('/', '_').replace(' ', '_')
        marker_folder = f"{marker_type}_{safe_marker_name}"
    else:
        marker_folder = marker_base_name.replace('/', '_').replace(' ', '_')
    output_dir = output_root / model_folder / marker_folder
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create Info object aligned to the original channel order for raw topographies
    if montage_path.endswith('.bvef'):
        montage = mne.channels.read_custom_montage(montage_path)
    else:
        montage = mne.channels.make_standard_montage(montage_path)
    info_raw = mne.create_info(ch_names=list(channels), sfreq=250, ch_types='eeg')
    info_raw.set_montage(montage, on_missing='ignore')

    # Generate raw topography report before statistical analysis
    create_raw_topography_report(
        power_data=power_data,
        df_behavioral=df_behavioral,
        ch_names=channels,
        info=info_raw,
        marker_name=marker_name,
        output_dir=str(output_dir),
        subject_col='subject',
        predictor_of_interest=config['lmm']['predictor_of_interest'],
    )
    
    # ── Backend dispatch ────────────────────────────────────────────────────
    # Select LMM backend (python/statsmodels or r/lme4) from config and bind
    # the two functions used throughout this function to local names so every
    # call site below is backend-agnostic.
    _backend = config['lmm'].get('backend', 'python')
    if _backend == 'r':
        from lmm_r_backend import (
            run_lmm_per_channel as _run_lmm,
            fit_reduced_model_per_channel as _fit_reduced,
        )
        logger.info("Using R backend (lme4::lmer via rpy2)")
    else:
        _run_lmm = run_lmm_per_channel
        _fit_reduced = fit_reduced_model_per_channel
        logger.info("Using Python backend (statsmodels MixedLM)")

    # ── Observation weights (WLS-LMM) ───────────────────────────────────────
    weight_col = config['lmm'].get('obs_weight_col')
    obs_weights: Optional[np.ndarray] = None
    if weight_col:
        if _backend != 'r':
            raise ValueError(
                f"obs_weight_col='{weight_col}' requires backend='r' for exact "
                "WLS-LMM. Set lmm.backend: 'r' or set obs_weight_col: null."
            )
        from lmm_r_backend import compute_obs_weights
        weight_norm = config['lmm'].get('weight_normalization', 'subject')
        obs_weights = compute_obs_weights(
            df_behavioral=df_behavioral,
            weight_col=weight_col,
            normalization=weight_norm,
            subject_col='subject',
        )
        logger.info(
            f"  Observation weights: col='{weight_col}', "
            f"normalization='{weight_norm}', "
            f"mean={obs_weights.mean():.3f}, std={obs_weights.std():.3f}"
        )

    # 2. Fit LMM per electrode (using proven implementation from Statistics/)
    print("Step 2: Fitting LMM per electrode...", flush=True)
    logger.info("Step 2: Fitting LMM per electrode...")
    predictor = config['lmm']['predictor_of_interest']
    formula = _resolve_formula_for_predictor(config, predictor)
    
    print(f"  Formula: {formula}", flush=True)
    print(f"  Predictor: {predictor}", flush=True)
    print(f"  Channels: {len(channels)}", flush=True)
    logger.info(f"  Formula: {formula}")
    logger.info(f"  Predictor: {predictor}")
    logger.info(f"  Channels: {len(channels)}")
    
    # Use the selected backend to fit LMM per channel
    t_stats, p_values, diagnostics = _run_lmm(
        power_data=power_data,
        df_behavioral=df_behavioral,
        formula=formula,
        predictor_of_interest=predictor,
        method=config['lmm']['method'],
        maxiter=config['lmm']['maxiter'],
        random_state=config['andrillon_clustering']['seed'],
        return_diagnostics=True,
        obs_weights=obs_weights,
    )
    
    print(f"  LMM fitting complete: {diagnostics['n_converged']}/{len(channels)} channels converged", flush=True)
    logger.info(f"  LMM fitting complete: {diagnostics['n_converged']}/{len(channels)} channels converged")
    
    # 3. Generate permutations (Andrillon 2020 method)
    print("Step 3: Generating permutations...", flush=True)
    logger.info("Step 3: Generating permutations...")
    n_permutations = config['andrillon_clustering']['n_permutations']
    print(f"  Running {n_permutations} permutations", flush=True)
    logger.info(f"  Running {n_permutations} permutations")
    
    # Store permutation t- and p-stats.
    # NaN (not zero/one) so that failed-channel entries are excluded from
    # cluster formation in the null distribution, matching the observed data.
    perm_t_stats = np.full((n_permutations, len(channels)), np.nan)
    perm_p_values = np.full((n_permutations, len(channels)), np.nan)
    permutation_within = config['andrillon_clustering']['permutation_within']

    # Use the same LMM method and maxiter for permuted fits as for the observed
    # fit — different optimizers produce systematically different t-statistics and
    # convergence rates, which creates an asymmetric null distribution and
    # inflates false positives.
    lmm_method = config['lmm']['method']
    lmm_maxiter = config['lmm']['maxiter']
    base_seed = config['andrillon_clustering']['seed']

    # Permutation method: "simple" shuffles the predictor inside permutation_within
    # groups (Andrillon default, biased when covariates are correlated with the
    # predictor); "freedman_lane" permutes residuals from the reduced model and
    # refits the full model on the reconstructed Y*.
    permutation_method = config['andrillon_clustering'].get(
        'permutation_method', 'simple'
    )
    if permutation_method not in ('simple', 'freedman_lane'):
        raise ValueError(
            f"Unsupported permutation_method '{permutation_method}'. "
            "Use 'simple' or 'freedman_lane'."
        )
    print(f"  Permutation method: {permutation_method}", flush=True)
    logger.info(f"  Permutation method: {permutation_method}")

    # If Freedman-Lane: fit the reduced model once to obtain residuals/fitted
    # values, which are then reused across all permutations. This is what
    # preserves the covariance structure under H0.
    fl_residuals = None
    fl_fitted = None
    if permutation_method == 'freedman_lane':
        print("  Fitting reduced model for Freedman-Lane permutation...", flush=True)
        logger.info("  Fitting reduced model for Freedman-Lane permutation...")
        fl_residuals, fl_fitted = _fit_reduced(
            power_data=power_data,
            df_behavioral=df_behavioral,
            formula=formula,
            predictor_of_interest=predictor,
            method=lmm_method,
            maxiter=lmm_maxiter,
            random_state=base_seed,
            obs_weights=obs_weights,
        )
        n_nan_resid = int(np.all(np.isnan(fl_residuals), axis=0).sum())
        print(
            f"  Reduced model fitted. Channels with all-NaN residuals: "
            f"{n_nan_resid}/{len(channels)}",
            flush=True,
        )
        logger.info(
            f"  Reduced model fitted. Channels with all-NaN residuals: "
            f"{n_nan_resid}/{len(channels)}"
        )

    # Per-permutation convergence diagnostics. The observed fit's diagnostics
    # are tracked separately. We compare aggregate convergence between observed
    # and permuted runs so that any asymmetry (which would deflate the null and
    # inflate FPR) is visible at the end of the run.
    n_jobs = config['andrillon_clustering'].get('n_jobs', -1)
    print(f"  Running permutations in parallel with n_jobs={n_jobs}", flush=True)
    logger.info(f"  Running permutations in parallel with n_jobs={n_jobs}")

    # Closure over the permutation context. Returning all required state per
    # call keeps each worker pure — joblib spawns processes that do not share
    # memory, so we cannot rely on perm_t_stats / perm_p_values mutation.
    def _run_one_permutation(perm_idx: int) -> Tuple[int, np.ndarray, np.ndarray, Dict]:
        # Deterministic per-permutation RNG. Seed namespace offset by
        # `n_permutations + 1` so it cannot collide with the observed fit's
        # seed (`base_seed`) — otherwise perm_idx=0 would call into the LMM
        # with the same global numpy seed as the observed fit, creating
        # spurious correlation between the two via any np.random consumer
        # in the optimizer stack.
        perm_seed = base_seed + 1 + perm_idx + n_permutations

        if permutation_method == 'simple':
            perm_rng = np.random.default_rng(perm_seed)

            # Permute predictor within `permutation_within` groups (e.g. subject)
            df_perm = df_behavioral.copy()
            groups = df_perm.groupby(permutation_within).groups
            for group_key, indices in groups.items():
                predictor_values = df_perm.loc[indices, predictor].values
                permuted_values = perm_rng.permutation(predictor_values)
                df_perm.loc[indices, predictor] = permuted_values

            power_perm_input = power_data
            df_perm_input = df_perm
        else:
            # Freedman-Lane: permute residuals within subject and reconstruct
            # Y* = Ŷ_reduced + R_perm. The full model is then refit on Y* with
            # the ORIGINAL predictor labels — so the predictor's covariance
            # with the other covariates is preserved under H0.
            power_perm_input = freedman_lane_permutation(
                residuals=fl_residuals,
                fitted_values=fl_fitted,
                df_behavioral=df_behavioral,
                seed=perm_seed,
            )
            df_perm_input = df_behavioral

        # Fit LMM with permuted data — identical settings to the observed fit.
        # Request diagnostics so we can verify convergence symmetry between
        # observed and permuted fits (asymmetry would bias the null).
        # abort_on_high_failure=False: a single degenerate permutation must
        # NOT kill the run. Failed channels become NaN and are excluded from
        # both null and observed cluster formation; the per-permutation
        # degenerate rate is tracked aggregately below.
        perm_t, perm_p, perm_diag = _run_lmm(
            power_data=power_perm_input,
            df_behavioral=df_perm_input,
            formula=formula,
            predictor_of_interest=predictor,
            method=lmm_method,
            maxiter=lmm_maxiter,
            random_state=perm_seed,
            return_diagnostics=True,
            abort_on_high_failure=False,
            obs_weights=obs_weights,
        )

        # A permutation is "degenerate" when more than 50% of the channels
        # fail to fit — same threshold the observed fit would have aborted
        # on. We do not abort here (NaNs propagate symmetrically), but we
        # count these draws so an asymmetric tail of pathological null
        # permutations is visible at the end of the run.
        n_attempted_perm = len(channels) - perm_diag.get('n_insufficient_data', 0)
        is_degenerate = (
            n_attempted_perm > 0
            and (perm_diag.get('n_failed', 0) / n_attempted_perm) > DEGENERATE_RATE_THRESHOLD
        )

        diag_record = {
            'perm_idx': perm_idx,
            'n_converged': perm_diag.get('n_converged', 0),
            'n_warnings': perm_diag.get('n_warnings', 0),
            'n_failed': perm_diag.get('n_failed', 0),
            'n_insufficient_data': perm_diag.get('n_insufficient_data', 0),
            'convergence_rate': perm_diag.get('convergence_rate', 0.0),
            'n_nan_t': int(np.sum(np.isnan(perm_t))),
            'is_degenerate': bool(is_degenerate),
        }
        return perm_idx, perm_t, perm_p, diag_record

    if n_jobs == 1:
        # Sequential execution (debug-friendly traceback)
        perm_results = []
        for perm_idx in range(n_permutations):
            if (perm_idx + 1) % 100 == 0 or perm_idx == 0:
                print(f"  Permutation {perm_idx+1}/{n_permutations}", flush=True)
            perm_results.append(_run_one_permutation(perm_idx))
    else:
        # Parallel execution. loky backend spawns isolated processes, so the
        # global numpy seed set inside `run_lmm_per_channel` cannot leak across
        # permutations — each worker has its own state.
        perm_results = Parallel(n_jobs=n_jobs, backend='loky', verbose=10)(
            delayed(_run_one_permutation)(perm_idx)
            for perm_idx in range(n_permutations)
        )

    # Reassemble in deterministic order. Even though joblib preserves order
    # by default, we re-index by perm_idx so a future change to backend or
    # ordering cannot silently corrupt the null distribution.
    perm_diag_records: List[Dict] = [None] * n_permutations
    for perm_idx, perm_t, perm_p, diag_record in perm_results:
        perm_t_stats[perm_idx, :] = perm_t
        perm_p_values[perm_idx, :] = perm_p
        perm_diag_records[perm_idx] = diag_record

    print(f"  Permutations complete", flush=True)
    logger.info(f"  Permutations complete")

    # Aggregate permutation convergence and compare with observed.
    # If the permuted runs converge less often than the observed, the null
    # has more NaN channels → fewer eligible nodes for clusters → null
    # distribution is contracted → observed clusters look more significant.
    perm_diag_df = pd.DataFrame(perm_diag_records)
    obs_n_nan = int(np.sum(np.isnan(t_stats)))
    n_degenerate = int(perm_diag_df['is_degenerate'].sum())
    degenerate_rate = n_degenerate / max(1, n_permutations)
    perm_diag_summary = {
        'n_channels': len(channels),
        'observed_n_converged': diagnostics['n_converged'],
        'observed_n_warnings': diagnostics['n_warnings'],
        'observed_n_failed': diagnostics['n_failed'],
        'observed_n_insufficient': diagnostics['n_insufficient_data'],
        'observed_n_nan_t': obs_n_nan,
        'observed_convergence_rate': diagnostics['convergence_rate'],
        'perm_n_converged_mean': float(perm_diag_df['n_converged'].mean()),
        'perm_n_converged_std': float(perm_diag_df['n_converged'].std()),
        'perm_n_warnings_mean': float(perm_diag_df['n_warnings'].mean()),
        'perm_n_failed_mean': float(perm_diag_df['n_failed'].mean()),
        'perm_n_nan_t_mean': float(perm_diag_df['n_nan_t'].mean()),
        'perm_convergence_rate_mean': float(perm_diag_df['convergence_rate'].mean()),
        'perm_convergence_rate_std': float(perm_diag_df['convergence_rate'].std()),
        'perm_n_degenerate': n_degenerate,
        'perm_degenerate_rate': float(degenerate_rate),
        'perm_degenerate_threshold': DEGENERATE_RATE_THRESHOLD,
    }

    if degenerate_rate > DEGENERATE_RATE_WARN:
        warn_msg = (
            f"WARNING: {n_degenerate}/{n_permutations} permutations "
            f"({100 * degenerate_rate:.1f}%) had >"
            f"{int(100 * DEGENERATE_RATE_THRESHOLD)}% LMM failures. "
            "This indicates the permutation scheme is producing many "
            "degenerate design matrices. "
            "Consider switching to permutation_method='freedman_lane' "
            "which permutes residuals from the reduced model instead and "
            "preserves the original predictor-covariate covariance under H0."
        )
        print(f"  {warn_msg}", flush=True)
        logger.warning(warn_msg)
    nan_asymmetry = obs_n_nan - perm_diag_summary['perm_n_nan_t_mean']
    perm_diag_summary['nan_asymmetry_obs_minus_perm_mean'] = float(nan_asymmetry)

    print(
        "  Convergence symmetry check:\n"
        f"    Observed: {diagnostics['n_converged']}/{len(channels)} converged "
        f"({100*diagnostics['convergence_rate']:.1f}%), "
        f"NaN t-stats: {obs_n_nan}\n"
        f"    Perm mean: {perm_diag_summary['perm_n_converged_mean']:.1f}/"
        f"{len(channels)} converged "
        f"({100*perm_diag_summary['perm_convergence_rate_mean']:.1f}%), "
        f"NaN t-stats: {perm_diag_summary['perm_n_nan_t_mean']:.1f}\n"
        f"    NaN asymmetry (obs - perm_mean): {nan_asymmetry:+.1f}",
        flush=True,
    )
    logger.info(
        "Convergence symmetry: obs=%d/%d (%.1f%%) NaN=%d | "
        "perm_mean=%.1f/%d (%.1f%%) NaN=%.1f | NaN_asym=%+.1f",
        diagnostics['n_converged'], len(channels),
        100 * diagnostics['convergence_rate'], obs_n_nan,
        perm_diag_summary['perm_n_converged_mean'], len(channels),
        100 * perm_diag_summary['perm_convergence_rate_mean'],
        perm_diag_summary['perm_n_nan_t_mean'], nan_asymmetry,
    )
    
    # 4. Build adjacency matrix for these channels
    print("Step 4: Building adjacency matrix...", flush=True)
    logger.info("Step 4: Building adjacency matrix...")
    adjacency, ch_names_ordered, channel_indices = get_channel_adjacency(montage_path, channels)
    logger.info(f"  Adjacency matrix shape: {adjacency.shape}")

    # Hard invariant: the indices we are about to use to permute t-stats must,
    # when applied to the original channel list, exactly reproduce the channel
    # order of the adjacency matrix. If this ever fails, t-stats and adjacency
    # rows/columns are misaligned and clusters would be spatially garbage.
    if [channels[idx] for idx in channel_indices] != list(ch_names_ordered):
        raise RuntimeError(
            "Adjacency-to-data channel mapping is broken: applying channel_indices "
            "to the input channels does not reproduce the adjacency channel order. "
            "Refusing to proceed — clusters would be spatially misaligned."
        )
    if adjacency.shape[0] != len(ch_names_ordered):
        raise RuntimeError(
            f"Adjacency shape {adjacency.shape} inconsistent with channel order "
            f"length {len(ch_names_ordered)}."
        )

    # Reorder statistics and channel list to match adjacency / montage order.
    # This mirrors the main Statistics pipeline to ensure spatial alignment.
    t_stats = t_stats[channel_indices]
    p_values = p_values[channel_indices]
    perm_t_stats = perm_t_stats[:, channel_indices]
    perm_p_values = perm_p_values[:, channel_indices]
    channels = [channels[idx] for idx in channel_indices]

    # Post-reorder invariant: channels must now match the adjacency order exactly.
    if list(channels) != list(ch_names_ordered):
        raise RuntimeError(
            "After reordering, `channels` does not match `ch_names_ordered`. "
            "Spatial alignment broken."
        )

    # Build exclusion mask for clustering based on configuration.
    # Excluded channels are still tested but cannot form or connect clusters.
    channel_excl_cfg = config.get('channel_exclusion', {})
    exclude_mask = None
    if channel_excl_cfg.get('enabled', False):
        excluded_names = set(channel_excl_cfg.get('channel_names', []))
        exclude_mask = np.array([ch in excluded_names for ch in channels], dtype=bool)

    # Create MNE Info object for visualization using the canonical channel order
    # from the adjacency/montage, but restricted to the channels present here.
    if montage_path.endswith('.bvef'):
        montage = mne.channels.read_custom_montage(montage_path)
    else:
        montage = mne.channels.make_standard_montage(montage_path)

    info = mne.create_info(ch_names=list(channels), sfreq=250, ch_types='eeg')
    # Ignore missing positions but keep available ones for plotting
    info.set_montage(montage, on_missing='ignore')
    
    # 5. Detect clusters (using proven find_clusters from Statistics/)
    print("Step 5: Detecting clusters...", flush=True)
    logger.info("Step 5: Detecting clusters...")

    cluster_alpha = float(config['andrillon_clustering']['cluster_alpha'])

    clusters_raw = find_clusters_from_pvalues(
        t_stats=t_stats,
        p_values=p_values,
        perm_t_stats=perm_t_stats,
        perm_p_values=perm_p_values,
        adjacency=adjacency,
        cluster_alpha=cluster_alpha,
        tail=0,
        n_permutations=n_permutations,
        stat_fun='sum',
        t_power=1.0,
        separate_signs=True,
        exclude=exclude_mask,
    )
    
    print(f"  Found {len(clusters_raw)} significant clusters", flush=True)
    logger.info(f"  Found {len(clusters_raw)} significant clusters")

    # Convert cluster dictionaries to arrays and summary statistics to match
    # the main Statistics pipeline expectations
    cluster_channels: List[np.ndarray] = []
    cluster_stats: List[float] = []
    cluster_p_values: List[float] = []
    for cluster in clusters_raw:
        if isinstance(cluster, dict):
            ch_idx = np.array(cluster.get('channels', []), dtype=int)
            stat_val = float(cluster.get('stat')) if cluster.get('stat') is not None else np.nan
            p_val = float(cluster.get('p_value')) if cluster.get('p_value') is not None else np.nan
        else:
            # Fallback for ClusterResult-like objects
            ch_idx = np.array(getattr(cluster, 'electrodes', []), dtype=int)
            stat_val = float(getattr(cluster, 'cluster_stat', np.nan))
            p_val = float(getattr(cluster, 'p_value', np.nan))

        cluster_channels.append(ch_idx)
        cluster_stats.append(stat_val)
        cluster_p_values.append(p_val)

    cluster_stats_arr = np.array(cluster_stats, dtype=float) if cluster_stats else np.array([], dtype=float)
    cluster_p_values_arr = np.array(cluster_p_values, dtype=float) if cluster_p_values else np.array([], dtype=float)

    # Compute number of significant clusters using Andrillon Monte Carlo alpha
    montecarlo_alpha = config['andrillon_clustering']['montecarlo_alpha']
    n_clusters = int(cluster_stats_arr.size)
    n_sig_clusters = int(np.sum(cluster_p_values_arr < montecarlo_alpha)) if n_clusters > 0 else 0

    # Visualization threshold for |t|: use the two-sided z-critical value
    # corresponding to cluster_alpha. With cluster_alpha=0.025 this is
    # z ≈ 2.24 (one-tailed) — the same threshold the Andrillon paper would
    # report. Previously this was the cluster_alpha-th quantile of the
    # observed |t|-distribution, which is data-dependent: each marker had
    # its own threshold making topoplots non-comparable across markers and
    # non-reproducible across runs.
    threshold_t = float(
        sp_stats.norm.ppf(1.0 - float(cluster_alpha))
    )

    # 6. Package results in a structure compatible with the Statistics pipeline
    results = {
        # Core identifiers
        'marker_name': marker_name,
        'marker_type': marker_type,
        'marker_base_name': marker_base_name,

        # Channel-wise statistics
        't_stats': t_stats,
        'p_values': p_values,
        'real_t_stats': t_stats,        # Backwards compatibility
        'real_p_values': p_values,      # Backwards compatibility

        # Cluster information
        'clusters': cluster_channels,   # List[np.ndarray] for plotting/reporting
        'cluster_stats': cluster_stats_arr,
        'cluster_p_values': cluster_p_values_arr,
        'clusters_raw': clusters_raw,   # Original objects from find_clusters
        'n_clusters': n_clusters,
        'n_sig_clusters': n_sig_clusters,

        # Channel / montage info
        'ch_names': channels,
        'channels': channels,           # Backwards compatibility
        'info': info,

        # Data shape and counts
        'power_data_shape': power_data.shape,
        'n_subjects': df_behavioral['subject'].nunique(),
        'n_observations': power_data.shape[0],
        'n_electrodes': len(channels),

        # Model and clustering metadata
        'config': config,
        'formula': formula,
        'predictor_of_interest': predictor,
        'analysis_timestamp': datetime.now().isoformat(),
        'clustering_method': 'andrillon_permutation',
        'permutation_method': permutation_method,
        'threshold': threshold_t,
        'alpha': montecarlo_alpha,
        'n_permutations': n_permutations,

        # Diagnostics from LMM fitting (observed)
        'diagnostics': diagnostics,

        # Permutation-side convergence diagnostics. Used to detect asymmetry
        # in the LMM fit between observed and permuted runs that would
        # contract the null distribution and inflate FPR.
        'perm_diagnostics': perm_diag_records,
        'perm_diagnostics_summary': perm_diag_summary,
    }

    return results


def save_results(results: Dict, output_dir: Path, marker_name: str):
    """Save analysis results.

    Uses a filesystem-safe marker name so that path separators in
    marker identifiers (e.g., "state/wsmi_theta") do not create
    unintended subdirectories.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create a safe filename component from the marker name
    # Examples:
    #   "state/wsmi_theta" -> "state_wsmi_theta"
    #   "evoked/wsmi_theta" -> "evoked_wsmi_theta"
    safe_marker_name = marker_name.replace('/', '_')

    # Save pickle (Andrillon-specific filename)
    pickle_path = output_dir / f"{safe_marker_name}_results.pkl"
    logger.info(f"Saving results to {pickle_path}")
    with open(pickle_path, 'wb') as f:
        pickle.dump(results, f)

    # Save generic results.pkl compatible with Statistics/generate_summary_report.py
    generic_pickle_path = output_dir / "results.pkl"
    logger.info(f"Saving generic results to {generic_pickle_path}")
    with open(generic_pickle_path, 'wb') as f:
        pickle.dump(results, f)

    # Save per-permutation convergence diagnostics as CSV for quick inspection
    # without unpickling. The summary row (single line) is also written so the
    # observed-vs-permuted asymmetry is grep-able from the cluster output dir.
    perm_records = results.get('perm_diagnostics')
    perm_summary = results.get('perm_diagnostics_summary')
    if perm_records:
        perm_csv = output_dir / f"{safe_marker_name}_perm_diagnostics.csv"
        pd.DataFrame(perm_records).to_csv(perm_csv, index=False)
        logger.info(f"Saved per-permutation diagnostics to {perm_csv}")
    if perm_summary:
        summary_csv = output_dir / f"{safe_marker_name}_convergence_summary.csv"
        pd.DataFrame([perm_summary]).to_csv(summary_csv, index=False)
        logger.info(f"Saved convergence symmetry summary to {summary_csv}")

    # Save CSV summary using the numeric cluster representation
    clusters = results.get('clusters', [])
    cluster_stats = np.asarray(results.get('cluster_stats', []), dtype=float)
    cluster_p_values = np.asarray(results.get('cluster_p_values', []), dtype=float)

    if clusters and cluster_stats.size == len(clusters) and cluster_p_values.size == len(clusters):
        csv_path = output_dir / f"{safe_marker_name}_clusters.csv"
        logger.info(f"Saving cluster summary to {csv_path}")

        cluster_data = []
        for i, ch_idx in enumerate(clusters):
            # ch_idx is an array of channel indices into results['ch_names'] / 'channels'
            ch_idx = np.asarray(ch_idx, dtype=int)
            # Map indices to channel names when available
            ch_names = results.get('ch_names', results.get('channels', []))
            if ch_names:
                cluster_channels = [ch_names[j] for j in ch_idx]
            else:
                cluster_channels = ch_idx.tolist()

            stat_val = float(cluster_stats[i]) if i < cluster_stats.size else np.nan
            p_val = float(cluster_p_values[i]) if i < cluster_p_values.size else np.nan
            cluster_type = 'positive' if np.isfinite(stat_val) and stat_val >= 0 else 'negative'

            cluster_data.append({
                'cluster_id': i,
                'cluster_type': cluster_type,
                'n_electrodes': len(cluster_channels),
                'electrodes': ','.join(map(str, cluster_channels)),
                'cluster_stat': stat_val,
                'p_value': p_val,
            })

        df = pd.DataFrame(cluster_data)
        df.to_csv(csv_path, index=False)
    else:
        logger.info("No significant clusters found - skipping CSV")

    # Optionally generate figures using the shared plotting utilities
    config = results.get('config', {})
    output_cfg = config.get('output', {}) if isinstance(config, dict) else {}
    if output_cfg.get('save_figures', True):
        t_stats = results.get('t_stats')
        clusters = results.get('clusters', [])
        cluster_stats = results.get('cluster_stats')
        cluster_p_values = results.get('cluster_p_values')
        info = results.get('info')
        threshold = results.get('threshold')
        alpha = results.get('alpha', 0.05)

        if threshold is None or (isinstance(threshold, float) and np.isnan(threshold)):
            threshold = 0.0

        if t_stats is not None and info is not None and cluster_stats is not None and cluster_p_values is not None:
            create_results_report(
                t_stats=t_stats,
                clusters=clusters,
                cluster_stats=cluster_stats,
                cluster_p_values=cluster_p_values,
                info=info,
                threshold=threshold,
                alpha=alpha,
                marker_name=marker_name,
                output_dir=str(output_dir),
                config=config if isinstance(config, dict) else None,
            )


def run_andrillon_pipeline(
    config_path: str,
    marker_name: Optional[str] = None,
):
    """
    Run complete Andrillon 2020 analysis pipeline.
    
    Parameters
    ----------
    config_path : str
        Path to configuration YAML file
    marker_name : str, optional
        If provided, analyze only this marker.
        If None, analyze all markers specified in config.
    """
    print("\n" + "="*80, flush=True)
    print("ANDRILLON 2020 CLUSTER-PERMUTATION PIPELINE", flush=True)
    print("="*80, flush=True)
    logger.info("="*80)
    logger.info("ANDRILLON 2020 CLUSTER-PERMUTATION PIPELINE")
    logger.info("="*80)
    
    # Load configuration
    print(f"Loading configuration from {config_path}", flush=True)
    logger.info(f"Loading configuration from {config_path}")
    config = load_config(config_path)

    # `predictor_of_interest` may be a list in the YAML to drive the
    # multi-predictor SLURM loop (submit_andrillon_predictor_loop.sh). When the
    # pipeline is invoked directly for a single run we need a single predictor
    # string — abort early with an actionable message rather than failing deep
    # in helpers.get_model_folder_name with an opaque AttributeError.
    poi = config['lmm'].get('predictor_of_interest', 'auto')
    if isinstance(poi, list):
        if len(poi) == 1:
            # A single-element list is unambiguous — unwrap it and continue.
            config['lmm']['predictor_of_interest'] = poi[0]
            logger.info(
                f"  predictor_of_interest was a single-element list; using '{poi[0]}'"
            )
        else:
            raise ValueError(
                f"`lmm.predictor_of_interest` is a list of {len(poi)} predictors "
                f"({poi}) but a single-run invocation needs exactly one. Either:\n"
                "  • pass --predictor-of-interest <name> on the CLI, or\n"
                "  • use Stats_andrillon/submit_andrillon_predictor_loop.sh to "
                "fan out one job per predictor, or\n"
                "  • set a single string in the YAML."
            )

    # Get adjacency matrix
    logger.info("Loading electrode adjacency matrix...")
    montage_path = Path(config['project']['montage_path'])
    
    # We'll get channel names from the first marker to build adjacency
    # This is a placeholder - adjacency will be built per marker with actual channels
    adjacency = None  # Will be built per marker
    logger.info("  Adjacency matrix will be built per marker based on available channels")
    
    # Get markers to analyze
    if marker_name:
        markers = [marker_name]
        logger.info(f"Analyzing single marker: {marker_name}")
    else:
        markers = get_marker_list(config)
        logger.info(f"Analyzing {len(markers)} markers")
    
    # Create output directory
    output_root = Path(config['project']['output_path'])
    
    # Determine model folder name from formula using the same helper
    # as the main Statistics pipeline to ensure identical directory names
    predictor = config['lmm'].get('predictor_of_interest', 'auto')
    model_folder = get_model_folder_name(
        _resolve_formula_for_predictor(config, predictor), predictor
    )
    
    # Run analysis for each marker
    all_results = {}
    for i, marker in enumerate(markers):
        print(f"\n{'#'*80}", flush=True)
        print(f"MARKER {i+1}/{len(markers)}: {marker}", flush=True)
        print(f"{'#'*80}\n", flush=True)
        logger.info(f"\n{'#'*80}")
        logger.info(f"MARKER {i+1}/{len(markers)}: {marker}")
        logger.info(f"{'#'*80}\n")
        
        # Run analysis — errors surface immediately so root causes are visible
        results = run_marker_analysis(marker, config, str(montage_path))

        all_results[marker] = results

        # Derive marker-specific output directory consistent with
        # the main Statistics pipeline: <marker_type>_<marker_name>
        if '/' in marker:
            marker_type, marker_base_name = marker.split('/', 1)
        else:
            marker_type = None
            marker_base_name = marker

        safe_marker_name = marker_base_name.replace('/', '_').replace(' ', '_')
        if marker_type:
            marker_folder = f"{marker_type}_{safe_marker_name}"
        else:
            marker_folder = safe_marker_name

        output_dir = output_root / model_folder / marker_folder
        print(f"Saving results to: {output_dir}", flush=True)
        logger.info(f"Saving results to: {output_dir}")
        save_results(results, output_dir, marker)
        print(f"Results saved successfully", flush=True)
    
    # Final summary
    print("\n" + "="*80, flush=True)
    print("PIPELINE COMPLETE", flush=True)
    print("="*80, flush=True)
    print(f"Analyzed {len(all_results)} markers successfully", flush=True)

    total_clusters = 0
    for res in all_results.values():
        cpv = np.asarray(res.get('cluster_p_values', []), dtype=float)
        total_clusters += int(cpv.size)
    print(f"Total clusters detected (before MCC): {total_clusters}", flush=True)

    model_dir = output_root / model_folder
    print(f"\nResults saved to: {model_dir}", flush=True)

    logger.info("\n" + "="*80)
    logger.info("PIPELINE COMPLETE")
    logger.info("="*80)
    logger.info(f"Analyzed {len(all_results)} markers successfully")
    logger.info(f"Total clusters detected (before MCC): {total_clusters}")
    logger.info(f"\nResults saved to: {model_dir}")

    # Generate summary report only when running the full set of markers
    if marker_name is None and len(all_results) > 0:
        print("\n" + "-"*80, flush=True)
        print("Generating Andrillon summary report (topoplots + tables)...", flush=True)
        print("-"*80, flush=True)
        generate_summary_report(
            model_dir=model_dir,
            alpha=config['andrillon_clustering']['montecarlo_alpha'],
            use_corrected=True,
            verbose=True,
        )

    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run Andrillon 2020 cluster-permutation pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--marker",
        type=str,
        default=None,
        help="Specific marker to analyze (optional, analyzes all if not provided)"
    )
    
    args = parser.parse_args()
    
    run_andrillon_pipeline(args.config, args.marker)
