"""
Moderation Analysis Pipeline: Phenomenological Dimensions × onoff → EEG.

Extension of the main LMM cluster permutation pipeline that tests whether
phenomenological probe dimensions (valence, selfother, time, confidence)
MODERATE the effect of onoff on EEG markers.

For each moderator, fits per channel:
    power ~ onoff * moderator + (1|subject)
and runs spatial cluster-based permutation testing on the interaction term
onoff:moderator.

After all individual tests, applies FDR-BH correction across all
moderator × marker combinations and generates a cross-moderator summary.

Usage
-----
Local (single moderator):
    python Statistics/run_moderation_pipeline.py --moderator valence

Local (all moderators):
    python Statistics/run_moderation_pipeline.py

Cluster (SLURM):
    sbatch Statistics/submit_moderation.sh

Author: Nicolas Bruno
"""

import argparse
import copy
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

import yaml

# Re-use the existing pipeline infrastructure
from run_pipeline import (
    load_config,
    process_single_marker,
    main as run_main_pipeline,
)
from helpers import get_model_folder_name
from statsmodels.stats.multitest import multipletests


# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_CONFIG_PATH = "Statistics/config.yaml"
MODERATION_CONFIG_PATH = "Statistics/config_moderation.yaml"


# =============================================================================
# HELPERS
# =============================================================================


def load_moderation_config(moderation_config_path: str,
                           base_config_path: str) -> dict:
    """
    Load moderation config and merge with the base pipeline config.

    Parameters
    ----------
    moderation_config_path : str
        Path to config_moderation.yaml.
    base_config_path : str
        Path to the base config.yaml (inherited settings).

    Returns
    -------
    dict
        Merged configuration with moderation-specific overrides applied.
    """
    base_config = load_config(base_config_path)
    mod_config = load_config(moderation_config_path)

    # Override base config with moderation-specific settings
    if 'lmm' in mod_config:
        for key, val in mod_config['lmm'].items():
            base_config['lmm'][key] = val

    if 'clustering' in mod_config and mod_config['clustering'] is not None:
        for key, val in mod_config['clustering'].items():
            base_config['clustering'][key] = val

    # Attach moderation-specific section
    base_config['moderation'] = mod_config.get('moderation', {})

    return base_config


def build_moderation_formula(base_predictor: str, moderator: str) -> str:
    """
    Build the interaction formula for a given moderator.

    Parameters
    ----------
    base_predictor : str
        Main predictor (e.g., "onoff").
    moderator : str
        Moderator variable (e.g., "valence").

    Returns
    -------
    str
        Formula string, e.g. "power ~ onoff * valence + (1|subject)".
    """
    return f"power ~ {base_predictor} * {moderator} + (1|subject)"


def build_interaction_term(base_predictor: str, moderator: str) -> str:
    """
    Build the interaction term name as statsmodels expects it.

    Parameters
    ----------
    base_predictor : str
        Main predictor (e.g., "onoff").
    moderator : str
        Moderator variable (e.g., "valence").

    Returns
    -------
    str
        Interaction term, e.g. "onoff:valence".
    """
    return f"{base_predictor}:{moderator}"


def collect_moderation_results(output_base: Path,
                               moderators: list,
                               base_predictor: str,
                               alpha: float = 0.05,
                               mcc_method: str = "fdr_bh") -> pd.DataFrame:
    """
    Collect all cluster results across moderators, apply FDR correction.

    Reads the per-moderator pipeline_summary.csv files, combines them,
    and applies FDR-BH across all moderator × marker tests.

    Parameters
    ----------
    output_base : Path
        Base output directory containing all model folders.
    moderators : list of str
        List of moderator names that were tested.
    base_predictor : str
        Base predictor name (e.g., "onoff").
    alpha : float
        Significance threshold for FDR.
    mcc_method : str
        Multiple comparisons method (e.g., "fdr_bh").

    Returns
    -------
    pd.DataFrame
        Combined summary with columns: moderator, marker_name, marker_type,
        n_sig_clusters_uncorrected, min_cluster_p, p_fdr, significant_fdr.
    """
    all_rows = []

    for moderator in moderators:
        formula = build_moderation_formula(base_predictor, moderator)
        interaction = build_interaction_term(base_predictor, moderator)
        model_folder = get_model_folder_name(formula, interaction)
        model_dir = output_base / model_folder

        summary_path = model_dir / "pipeline_summary.csv"
        if not summary_path.exists():
            print(f"  Warning: {summary_path} not found — skipping {moderator}")
            continue

        df = pd.read_csv(summary_path)
        df['moderator'] = moderator
        df['interaction_term'] = interaction
        df['moderation_formula'] = formula
        all_rows.append(df)

    if not all_rows:
        print("No moderation results found.")
        return pd.DataFrame()

    combined = pd.concat(all_rows, ignore_index=True)

    # Extract the minimum cluster p-value per marker × moderator
    # (used as the test-level p-value for FDR)
    combined['min_cluster_p'] = np.nan
    for idx, row in combined.iterrows():
        model_folder = get_model_folder_name(
            row['moderation_formula'], row['interaction_term']
        )
        model_dir = output_base / model_folder
        safe_name = row['marker_name'].replace('/', '_').replace(' ', '_')
        marker_folder = f"{row['marker_type']}_{safe_name}"
        cluster_csv = model_dir / marker_folder / "cluster_summary_uncorrected.csv"

        if cluster_csv.exists():
            cluster_df = pd.read_csv(cluster_csv)
            if len(cluster_df) > 0 and 'p_value' in cluster_df.columns:
                combined.at[idx, 'min_cluster_p'] = cluster_df['p_value'].min()
            else:
                combined.at[idx, 'min_cluster_p'] = 1.0
        else:
            combined.at[idx, 'min_cluster_p'] = 1.0

    # Apply FDR correction across ALL moderator × marker tests
    valid_mask = combined['min_cluster_p'].notna()
    p_vals = combined.loc[valid_mask, 'min_cluster_p'].values

    if len(p_vals) > 0:
        _, p_fdr, _, _ = multipletests(p_vals, method=mcc_method, alpha=alpha)
        combined['p_fdr'] = np.nan
        combined.loc[valid_mask, 'p_fdr'] = p_fdr
        combined['significant_fdr'] = combined['p_fdr'] < alpha
    else:
        combined['p_fdr'] = np.nan
        combined['significant_fdr'] = False

    return combined


# =============================================================================
# MAIN
# =============================================================================


def run_moderation_pipeline(moderation_config_path: str = MODERATION_CONFIG_PATH,
                            base_config_path: str = BASE_CONFIG_PATH,
                            moderator: str = None,
                            marker_index: int = None) -> None:
    """
    Run the full moderation analysis pipeline.

    Parameters
    ----------
    moderation_config_path : str
        Path to config_moderation.yaml.
    base_config_path : str
        Path to the base config.yaml.
    moderator : str, optional
        Single moderator to run. If None, runs all moderators from config.
    marker_index : int, optional
        SLURM array index for processing a single marker.
    """
    print("=" * 80)
    print("MODERATION ANALYSIS: Phenomenological Dimensions x onoff -> EEG")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Load merged config
    config = load_moderation_config(moderation_config_path, base_config_path)

    mod_settings = config['moderation']
    moderators = mod_settings['moderators']
    base_predictor = mod_settings['base_predictor']
    mcc_method = mod_settings.get('mcc_method', 'fdr_bh')
    mcc_alpha = mod_settings.get('mcc_alpha', 0.05)
    output_base = Path(config['project']['output_path'])

    # If a single moderator is specified, run only that one
    if moderator is not None:
        if moderator not in moderators:
            print(f"Warning: '{moderator}' not in configured moderators {moderators}.")
            print("Running anyway...")
        moderators_to_run = [moderator]
    else:
        moderators_to_run = moderators

    print(f"Base predictor: {base_predictor}")
    print(f"Moderators to test: {moderators_to_run}")
    print(f"MCC method: {mcc_method} (alpha={mcc_alpha})")
    print(f"Output base: {output_base}\n")

    # Run the standard pipeline for each moderator
    for mod in moderators_to_run:
        formula = build_moderation_formula(base_predictor, mod)
        interaction = build_interaction_term(base_predictor, mod)

        print("\n" + "=" * 80)
        print(f"MODERATOR: {mod}")
        print(f"Formula: {formula}")
        print(f"Predictor of interest: {interaction}")
        print("=" * 80)

        # Create a per-moderator config copy
        mod_config = copy.deepcopy(config)
        mod_config['lmm']['formula'] = formula
        mod_config['lmm']['predictor_of_interest'] = interaction

        # Run via the existing pipeline main()
        # We write a temporary config and call main() with it
        tmp_config_path = output_base / f"_tmp_config_moderation_{mod}.yaml"
        tmp_config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp_config_path, 'w') as f:
            yaml.dump(mod_config, f, default_flow_style=False, sort_keys=False)

        try:
            run_main_pipeline(
                config_path=str(tmp_config_path),
                marker_index=marker_index,
                predictor_of_interest=interaction,
            )
            print(f"\n✓ Completed moderator: {mod}")
        except Exception as e:
            print(f"\n✗ Failed moderator {mod}: {e}")
        finally:
            # Clean up temporary config
            if tmp_config_path.exists():
                tmp_config_path.unlink()

    # Cross-moderator FDR correction (only when running all moderators)
    if moderator is None and marker_index is None:
        print("\n" + "=" * 80)
        print("CROSS-MODERATOR FDR CORRECTION")
        print("=" * 80)

        summary = collect_moderation_results(
            output_base=output_base,
            moderators=moderators,
            base_predictor=base_predictor,
            alpha=mcc_alpha,
            mcc_method=mcc_method,
        )

        if len(summary) > 0:
            # Save summary
            summary_dir = output_base / "moderation_summary"
            summary_dir.mkdir(parents=True, exist_ok=True)

            summary_path = summary_dir / "moderation_cross_moderator_summary.csv"
            summary.to_csv(summary_path, index=False)
            print(f"✓ Cross-moderator summary saved: {summary_path}")

            # Print results
            n_sig = summary['significant_fdr'].sum()
            n_total = len(summary)
            print(f"\nResults: {n_sig}/{n_total} marker × moderator tests "
                  f"significant after {mcc_method} (alpha={mcc_alpha})")

            if n_sig > 0:
                sig_rows = summary[summary['significant_fdr']].copy()
                display_cols = ['moderator', 'marker_name', 'marker_type',
                                'min_cluster_p', 'p_fdr', 'n_sig_clusters_uncorrected']
                print("\nSignificant moderation effects:")
                print(sig_rows[display_cols].to_string(index=False))
            else:
                print("No significant moderation effects after FDR correction.")
        else:
            print("No results to summarize.")

    # Done
    print("\n" + "=" * 80)
    print("MODERATION PIPELINE COMPLETED")
    print(f"Ended at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Moderation analysis: phenomenological dimensions x onoff -> EEG"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=MODERATION_CONFIG_PATH,
        help="Path to moderation configuration YAML file",
    )
    parser.add_argument(
        "--base-config",
        type=str,
        default=BASE_CONFIG_PATH,
        dest="base_config",
        help="Path to the base pipeline config.yaml",
    )
    parser.add_argument(
        "--moderator",
        type=str,
        default=None,
        help="Single moderator to run (e.g., 'valence'). If omitted, runs all.",
    )
    parser.add_argument(
        "--marker-index",
        type=int,
        default=None,
        dest="marker_index",
        help="Index of marker to process (for SLURM array jobs)",
    )

    args = parser.parse_args()

    run_moderation_pipeline(
        moderation_config_path=args.config,
        base_config_path=args.base_config,
        moderator=args.moderator,
        marker_index=args.marker_index,
    )
