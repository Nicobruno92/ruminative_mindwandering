"""
Post-processing script to apply multiple comparisons correction (MCC) after SLURM array jobs.

This script:
1. Loads all individual marker results from pickle files
2. Groups markers by type (evoked/state)
3. Applies FDR/Bonferroni correction across markers
4. Saves corrected cluster summaries
5. Generates correction summary report

Usage:
    python Statistics/apply_mcc_postprocessing.py <model_dir> [--config CONFIG_PATH]
    
Example:
    python Statistics/apply_mcc_postprocessing.py results/lmm_cluster/onoff
"""

import argparse
import pickle
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple

from multiple_comparisons import correct_cluster_p_values, create_correction_summary
from helpers import summarize_clusters
from plot_results import plot_cluster_topomap


# Transient key holding the directory a result was actually loaded from.
# Fix FIND-002: the marker directory must NEVER be reconstructed from
# ``marker_name``. The Andrillon pipeline stores marker_name as
# "<type>/<name>" (e.g. "evoked/P3b"), so rebuilding the path as
# f"{marker_type}_{marker_name.replace('/', '_')}" produced "evoked_evoked_P3b",
# a directory that does not exist. Every downstream write silently became a
# no-op and the reports fell back to UNCORRECTED p-values. Carrying the real
# source directory removes that entire class of bug.
SOURCE_DIR_KEY = '_mcc_source_dir'

# Marker types recognised by the correction step, in report order.
MARKER_TYPES = ('evoked', 'state', 'sleep')


def load_marker_results(model_dir: Path, verbose: bool = True) -> List[Dict]:
    """
    Load all marker results from a model directory.

    Parameters
    ----------
    model_dir : Path
        Path to model directory containing marker subdirectories
    verbose : bool
        Print progress information

    Returns
    -------
    List[Dict]
        List of result dictionaries, each carrying ``SOURCE_DIR_KEY`` with the
        directory it was loaded from.

    Raises
    ------
    ValueError
        If a marker's type cannot be determined. Silently dropping such a
        marker would change the denominator of the FDR correction without
        leaving any trace in the outputs.
    """
    results = []

    # Find all marker subdirectories
    marker_dirs = sorted(d for d in model_dir.iterdir() if d.is_dir())

    if verbose:
        print(f"Scanning {len(marker_dirs)} marker directories...")

    for marker_dir in marker_dirs:
        results_pkl = marker_dir / "results.pkl"

        if not results_pkl.exists():
            if verbose:
                print(f"  ⚠ Skipping {marker_dir.name}: no results.pkl found")
            continue

        # Fix FIND-001: no try/except. A pickle that fails to load is a real
        # problem — swallowing it silently shrinks the family size used by the
        # FDR correction and inflates significance without any warning.
        with open(results_pkl, 'rb') as f:
            result = pickle.load(f)

        # Ensure marker_type is present, inferring from the directory name.
        if 'marker_type' not in result or result['marker_type'] is None:
            inferred = next(
                (t for t in MARKER_TYPES if marker_dir.name.startswith(f"{t}_")),
                None
            )
            if inferred is None:
                raise ValueError(
                    f"Cannot determine marker_type for {marker_dir}. "
                    f"Expected the directory name to start with one of "
                    f"{MARKER_TYPES}, or the pickle to carry a 'marker_type' key."
                )
            result['marker_type'] = inferred

        result[SOURCE_DIR_KEY] = marker_dir
        results.append(result)

        if verbose:
            marker_name = result.get('marker_name', 'unknown')
            marker_type = result['marker_type']
            n_clusters = result.get('n_clusters', 0)
            print(f"  ✓ Loaded {marker_name} ({marker_type}): {n_clusters} clusters")

    if verbose:
        print(f"\n✓ Loaded {len(results)} marker results")

    return results


def resolve_expected_fragments(config: Dict) -> Dict[str, List[str]]:
    """
    Resolve the marker-name fragments the config declares, per marker type.

    Mirrors the family resolution in ``Stats_andrillon/andrillon_pipeline.py``
    (``selected_markers`` -> ``feature_families`` -> fragments) so the family
    used for the correction is the family the analysis *declared*, not whatever
    happens to be present on disk.

    Parameters
    ----------
    config : Dict
        Parsed configuration.

    Returns
    -------
    Dict[str, List[str]]
        Marker type -> list of name fragments. An empty list means "accept
        every marker of this type" (the ``all`` family, defined as null).

    Raises
    ------
    ValueError
        If ``selected_markers`` names a family absent from ``feature_families``.
        Left as a hard error because a typo or a half-finished config edit would
        otherwise silently shrink the family and loosen the correction.
    """
    feature_families = config.get('feature_families', {})
    selected = config.get('selected_markers', {})

    expected: Dict[str, List[str]] = {}
    for marker_type, families in selected.items():
        fragments: List[str] = []
        accept_all = False
        for family_name in families:
            if family_name not in feature_families:
                raise ValueError(
                    f"Family '{family_name}' (in selected_markers['{marker_type}']) "
                    f"is not defined in feature_families. "
                    f"Available: {sorted(feature_families.keys())}"
                )
            members = feature_families[family_name]
            if members is None:
                accept_all = True
            else:
                fragments.extend(members)
        expected[marker_type] = [] if accept_all else fragments

    return expected


def partition_results_by_config(
    results: List[Dict],
    config: Dict,
    verbose: bool = True,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Keep only the markers the config declares, and report what does not line up.

    Fix FIND-004: the family size used by the correction was previously "however
    many marker directories happen to exist", so a marker that crashed, was
    never launched, or was left over from an older config silently changed the
    denominator — and therefore every corrected p-value in that family. Nobody
    decided that; it emerged from the state of the filesystem.

    Parameters
    ----------
    results : List[Dict]
        Marker results loaded from disk.
    config : Dict
        Parsed configuration declaring the intended markers.
    verbose : bool
        Print the family composition.

    Returns
    -------
    Tuple[List[Dict], List[Dict]]
        ``(in_family, excluded)``. Markers in ``excluded`` are stripped of any
        correction provenance so a stale result from an earlier run, under a
        different family or method, can never be mistaken for a current one.

    Raises
    ------
    ValueError
        If a declared fragment matched no marker on disk, unless
        ``multiple_comparisons.require_complete_family`` is set to false.
    """
    if not config.get('selected_markers'):
        print(
            "\n⚠ This config does not declare 'selected_markers', so the correction "
            "family cannot be anchored to the analysis plan and falls back to "
            "whatever is on disk. The family size is then not reproducible."
        )
        return results, []

    expected = resolve_expected_fragments(config)
    mcc_config = config.get('multiple_comparisons', {})
    require_complete = mcc_config.get('require_complete_family', True)

    in_family: List[Dict] = []
    excluded: List[Dict] = []
    matched_fragments: Dict[str, set] = {mt: set() for mt in expected}

    for result in results:
        marker_type = result['marker_type']
        if marker_type not in expected:
            excluded.append(result)
            continue

        fragments = expected[marker_type]
        base_name = result.get('marker_name', '').split('/')[-1]

        if not fragments:  # the "all" family
            in_family.append(result)
            continue

        hit = next((f for f in fragments if f in base_name), None)
        if hit is None:
            excluded.append(result)
        else:
            matched_fragments[marker_type].add(hit)
            in_family.append(result)

    missing = {
        mt: sorted(set(frags) - matched_fragments[mt])
        for mt, frags in expected.items()
        if frags and set(frags) - matched_fragments[mt]
    }

    if verbose:
        print("\nFamily composition (anchored to config, not to disk):")
        for marker_type in sorted(expected):
            n = len([r for r in in_family if r['marker_type'] == marker_type])
            print(f"  {marker_type:8s} m = {n}")
        if excluded:
            print(f"  ⚠ {len(excluded)} marker(s) on disk are NOT declared by the current "
                  f"config and are EXCLUDED from the correction:")
            for r in excluded:
                print(f"      - {r.get('marker_name', '?')}")

    if missing:
        detail = "; ".join(f"{mt}: {frags}" for mt, frags in missing.items())
        message = (
            f"The config declares marker(s) that produced no results on disk, so "
            f"the correction family would be smaller than the analysis plan and "
            f"the correction correspondingly weaker. Missing -> {detail}. "
            f"Re-run the missing markers, or set "
            f"multiple_comparisons.require_complete_family: false to proceed "
            f"deliberately with a reduced family."
        )
        if require_complete:
            raise ValueError(message)
        print(f"\n⚠ {message}")

    # Strip correction provenance from excluded markers. Without this, a marker
    # dropped from the family keeps the corrected p-values written by an earlier
    # run — under a different family size and possibly a different method — and
    # the summary report, which reads every directory on disk, silently counts
    # them as current results.
    stale_keys = (
        'cluster_p_values_corrected', 'cluster_rejected', 'correction_method',
        'correction_alpha', 'correction_unit', 'correction_family_size',
        'marker_p_value', 'marker_p_value_corrected', 'marker_rejected',
    )
    for result in excluded:
        for key in stale_keys:
            result.pop(key, None)
        result['in_correction_family'] = False
    for result in in_family:
        result['in_correction_family'] = True

    return in_family, excluded


def write_family_composition(
    results: List[Dict],
    model_dir: Path,
    verbose: bool = True,
) -> None:
    """
    Record exactly which markers formed each correction family.

    Makes the family size auditable after the fact: without this, a reader
    cannot tell whether a corrected p-value came from a family of 25 or 17.

    Parameters
    ----------
    results : List[Dict]
        Markers that entered the correction.
    model_dir : Path
        Model directory to write into.
    verbose : bool
        Print the output path.
    """
    rows = [
        {
            'marker_name': r.get('marker_name', 'unknown'),
            'marker_type': r.get('marker_type', 'unknown'),
            'n_clusters': r.get('n_clusters', 0),
            'n_permutations': r.get('n_permutations', pd.NA),
            'correction_family_size': r.get('correction_family_size', pd.NA),
            'correction_method': r.get('correction_method', 'none'),
            'correction_unit': r.get('correction_unit', 'none'),
        }
        for r in results
    ]
    df = pd.DataFrame(rows).sort_values(['marker_type', 'marker_name'])
    path = model_dir / "mcc_family_composition.csv"
    df.to_csv(path, index=False)
    if verbose:
        print(f"✓ Family composition saved to {path}")


def persist_corrected_pvalues(
    results: List[Dict],
    model_dir: Path,
    mcc_alpha: float,
    verbose: bool = True,
    add_missing_keys: bool = True,
) -> None:
    """
    Write corrected cluster p-values back into each marker's pickle files.

    ``generate_summary_report`` reads ``results.pkl`` and only enters its
    corrected-p-value branch when the ``cluster_p_values_corrected`` key is
    present. The correction is computed in-memory by ``correct_cluster_p_values``
    but was never persisted, so the summary report silently fell back to the
    UNCORRECTED p-values while ``multiple_comparisons_summary.csv`` held the
    corrected ones — two outputs that could disagree. This function closes that
    gap by writing the corrected keys into both ``results.pkl`` (the file the
    report reads) and the marker-specific ``<marker>_results.pkl`` so they stay
    consistent.

    Markers with no clusters receive an empty corrected array plus correction
    metadata, so every marker uniformly carries the same correction provenance.

    Parameters
    ----------
    results : List[Dict]
        Corrected result dictionaries (already mutated in-place with
        ``cluster_p_values_corrected`` where clusters existed).
    model_dir : Path
        Path to the model directory containing the marker subdirectories.
    mcc_alpha : float
        Alpha level used for the correction (stored as provenance).
    verbose : bool
        Print progress information.
    """
    if verbose:
        print("\nPersisting corrected p-values back into marker pickles...")

    n_written = 0
    for result in results:
        # Ensure corrected keys exist even for markers with no clusters, so the
        # summary report uniformly enters its corrected-p-value branch. Skipped
        # for excluded markers, which must stay free of correction provenance.
        if add_missing_keys and 'cluster_p_values_corrected' not in result:
            p_vals = np.asarray(result.get('cluster_p_values', []), dtype=float)
            result['cluster_p_values_corrected'] = p_vals.copy()
            result['cluster_rejected'] = p_vals <= mcc_alpha
            result['correction_method'] = result.get('correction_method', 'none')
            result['correction_alpha'] = mcc_alpha

        # Fix FIND-002: use the directory the result was actually loaded from
        # instead of rebuilding it from marker_name (see SOURCE_DIR_KEY).
        marker_dir = result[SOURCE_DIR_KEY]

        # Drop transient keys that loaders attach so the round-tripped pickle
        # stays clean and never carries machine-specific paths.
        persisted = {
            k: v for k, v in result.items()
            if k not in (SOURCE_DIR_KEY, 'marker_dir')
        }

        # Update the generic results.pkl (read by generate_summary_report) and
        # every marker-specific "<dir>_results.pkl" so the two never diverge.
        pkl_paths = [marker_dir / "results.pkl"] + sorted(marker_dir.glob("*_results.pkl"))
        for pkl_path in pkl_paths:
            if pkl_path.exists():
                with open(pkl_path, 'wb') as f:
                    pickle.dump(persisted, f)
                n_written += 1

    if verbose:
        print(f"  ✓ Updated {n_written} pickle file(s) with corrected p-values")


def save_corrected_summaries(results: List[Dict], model_dir: Path,
                            mcc_alpha: float, verbose: bool = True) -> None:
    """
    Save corrected cluster summaries for each marker.
    
    Parameters
    ----------
    results : List[Dict]
        List of corrected result dictionaries
    model_dir : Path
        Path to model directory
    mcc_alpha : float
        Alpha level for significance
    verbose : bool
        Print progress information
    """
    if verbose:
        print("\nSaving corrected cluster summaries...")
    
    for result in results:
        if result.get('n_clusters', 0) == 0:
            continue

        marker_name = result['marker_name']
        marker_type = result.get('marker_type', 'unknown')

        # Fix FIND-002: use the real source directory, never a reconstructed one.
        marker_dir = result[SOURCE_DIR_KEY]

        # Create corrected cluster summary
        cluster_summary_corrected = summarize_clusters(
            result['clusters'],
            result['cluster_stats'],
            result['cluster_p_values_corrected'],
            result['ch_names'],
            mcc_alpha
        )
        
        # Add metadata
        cluster_summary_corrected['marker_name'] = marker_name
        cluster_summary_corrected['marker_type'] = marker_type
        cluster_summary_corrected['correction_method'] = result.get('correction_method', 'none')
        cluster_summary_corrected['correction_alpha'] = result.get('correction_alpha', mcc_alpha)
        cluster_summary_corrected['analysis_timestamp'] = result.get('analysis_timestamp', datetime.now().isoformat())
        
        # Always persist BOTH the uncorrected and the corrected p-value for
        # every cluster, so any downstream reader can reproduce the correction
        # and see exactly what it cost.
        cluster_summary_corrected['p_value_uncorrected'] = result['cluster_p_values']
        cluster_summary_corrected['p_value_corrected'] = result['cluster_p_values_corrected']
        cluster_summary_corrected['significant_uncorrected'] = result['cluster_p_values'] <= mcc_alpha
        cluster_summary_corrected['significant_corrected'] = result['cluster_rejected']

        # Marker-level provenance (unit="marker"): the max-stat cluster p-value
        # that represents this marker in the family-wise correction, its
        # corrected counterpart, and whether the marker survived the gate.
        if 'marker_p_value' in result:
            cluster_summary_corrected['marker_p_value_uncorrected'] = result['marker_p_value']
            cluster_summary_corrected['marker_p_value_corrected'] = result['marker_p_value_corrected']
            cluster_summary_corrected['marker_rejected'] = result['marker_rejected']
        cluster_summary_corrected['correction_unit'] = result.get('correction_unit', 'cluster')

        # Save corrected summary
        csv_path_corrected = marker_dir / "cluster_summary_corrected.csv"
        cluster_summary_corrected.to_csv(csv_path_corrected, index=False)

        # Regenerate topomap plot with both uncorrected and corrected p-values.
        # Fix FIND-001: no try/except — a plot that cannot be drawn signals a
        # malformed result and must surface, not be reported as a warning while
        # the pipeline claims success.
        safe_marker_name = marker_name.replace('/', '_').replace(' ', '_')
        plot_title = f"Spatial Cluster Test - {marker_name} (MCC Applied)"
        topomap_path = marker_dir / f"results_{safe_marker_name}_topomap_corrected.png"

        plot_cluster_topomap(
            t_stats=result['t_stats'],
            clusters=result['clusters'],
            cluster_p_values=result['cluster_p_values_corrected'],  # Corrected as primary
            info=result['info'],
            alpha=mcc_alpha,
            title=plot_title,
            save_path=str(topomap_path),
            cluster_p_values_uncorrected=result['cluster_p_values']  # Show uncorrected as empty circles
        )

        if verbose:
            n_sig_before = np.sum(result['cluster_p_values'] <= mcc_alpha)
            n_sig_after = np.sum(result['cluster_rejected'])
            print(f"  ✓ {marker_name}: {n_sig_before} → {n_sig_after} significant clusters")


def main():
    parser = argparse.ArgumentParser(
        description="Apply multiple comparisons correction to SLURM array job results"
    )
    parser.add_argument(
        "model_dir",
        type=str,
        help="Path to model directory (e.g., results/lmm_cluster/onoff)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="Statistics/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print detailed progress information"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("MULTIPLE COMPARISONS CORRECTION POST-PROCESSING")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Convert to Path
    model_dir = Path(args.model_dir)
    
    if not model_dir.exists():
        print(f"✗ Error: Model directory not found: {model_dir}")
        return 1
    
    print(f"Model directory: {model_dir}")
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"✗ Error: Config file not found: {config_path}")
        return 1
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get correction settings
    mcc_config = config.get('multiple_comparisons', {})

    # Determine alpha level in a config-agnostic way so this script can be
    # reused for both the original Statistics pipeline and the Andrillon
    # pipeline.
    if 'alpha' in mcc_config and mcc_config['alpha'] is not None:
        mcc_alpha = mcc_config['alpha']
    else:
        # Fallbacks: legacy "clustering.alpha" (Statistics) or
        # "andrillon_clustering.montecarlo_alpha" (Andrillon pipeline).
        if 'clustering' in config and 'alpha' in config['clustering']:
            mcc_alpha = config['clustering']['alpha']
        elif 'andrillon_clustering' in config and 'montecarlo_alpha' in config['andrillon_clustering']:
            mcc_alpha = config['andrillon_clustering']['montecarlo_alpha']
        else:
            # Sensible default if nothing is specified
            mcc_alpha = 0.05

    # Unit of correction: what counts as one hypothesis in the family.
    # "marker" (one max-statistic p-value per marker) is the coherent choice
    # for cluster-permutation results; "cluster" is the legacy behaviour.
    mcc_unit = mcc_config.get('unit', 'cluster')

    print(f"Configuration loaded from: {config_path}")
    print(f"Alpha level: {mcc_alpha}")
    print(f"Correction unit: {mcc_unit}")
    
    # Load all marker results
    print("\n" + "-"*80)
    print("STEP 1: Loading marker results")
    print("-"*80)
    
    results = load_marker_results(model_dir, verbose=args.verbose)

    if len(results) == 0:
        print("✗ No results found to process")
        return 1

    # Fix FIND-004: the correction family is defined by the config, not by
    # whatever marker directories happen to exist.
    results, excluded = partition_results_by_config(results, config, verbose=args.verbose)

    if len(results) == 0:
        print("✗ No results match the markers declared in the config")
        return 1


    # Apply correction separately for evoked and state markers
    print("\n" + "-"*80)
    print("STEP 2: Applying multiple comparisons correction")
    print("-"*80)
    
    corrected_results = []
    
    for marker_type in MARKER_TYPES:
        # Get correction method for this marker type
        correction_method = mcc_config.get(marker_type, False)

        # Filter results for this marker type
        type_results = [r for r in results if r.get('marker_type') == marker_type]

        if len(type_results) == 0:
            print(f"\nNo {marker_type} markers to correct")
            continue

        markers_with_clusters = [r for r in type_results if r.get('n_clusters', 0) > 0]

        print(f"\n{marker_type.upper()} markers:")
        print(f"  Number of markers: {len(type_results)}")
        print(f"  Markers with clusters: {len(markers_with_clusters)}")
        print(f"  Correction method: {correction_method}")

        # With unit="cluster" there is nothing to correct when no marker
        # produced a cluster. With unit="marker" every tested marker is part of
        # the family regardless, so the correction always runs and every marker
        # ends up carrying uniform correction provenance.
        if mcc_unit == 'cluster' and len(markers_with_clusters) == 0:
            print(f"  No clusters found in any marker - skipping correction")
            corrected_results.extend(type_results)
            continue

        # Apply correction
        type_corrected = correct_cluster_p_values(
            results_list=type_results,
            correction_method=correction_method,
            alpha=mcc_alpha,
            unit=mcc_unit,
            verbose=True
        )

        corrected_results.extend(type_corrected)

    # Every loaded marker must belong to exactly one corrected family. A marker
    # silently falling outside MARKER_TYPES would be dropped from the report
    # without changing any visible count.
    if len(corrected_results) != len(results):
        uncovered = sorted({
            r.get('marker_type', 'unknown') for r in results
        } - set(MARKER_TYPES))
        raise ValueError(
            f"{len(results) - len(corrected_results)} marker(s) were not covered "
            f"by any correction family. Unrecognised marker_type(s): {uncovered}. "
            f"Expected one of {MARKER_TYPES}."
        )

    # Persist corrected p-values back into the marker pickles so that
    # generate_summary_report (which reads results.pkl) reports CORRECTED
    # significance instead of silently falling back to uncorrected.
    print("\n" + "-"*80)
    print("STEP 3: Persisting corrected p-values into marker pickles")
    print("-"*80)

    persist_corrected_pvalues(corrected_results, model_dir, mcc_alpha, verbose=args.verbose)

    # Excluded markers are written back too, so the correction keys stripped
    # above actually disappear from disk. Otherwise a marker dropped from the
    # family keeps p-values from a previous run and the report counts them.
    if excluded:
        print(f"\nClearing stale correction keys from {len(excluded)} excluded marker(s)...")
        persist_corrected_pvalues(
            excluded, model_dir, mcc_alpha, verbose=False, add_missing_keys=False
        )

    # Save corrected summaries
    print("\n" + "-"*80)
    print("STEP 4: Saving corrected cluster summaries")
    print("-"*80)
    
    save_corrected_summaries(corrected_results, model_dir, mcc_alpha, verbose=args.verbose)
    
    # Save correction summary
    print("\n" + "-"*80)
    print("STEP 5: Creating correction summary report")
    print("-"*80)
    
    correction_summary_path = model_dir / "multiple_comparisons_summary.csv"
    correction_summary = create_correction_summary(
        corrected_results,
        output_path=str(correction_summary_path)
    )
    
    write_family_composition(corrected_results, model_dir, verbose=args.verbose)

    print("\n✓ Correction summary:")
    print(correction_summary.to_string(index=False))
    
    # Update pipeline summary if it exists
    pipeline_summary_path = model_dir / "pipeline_summary.csv"
    if pipeline_summary_path.exists():
        print("\n" + "-"*80)
        print("STEP 6: Updating pipeline summary")
        print("-"*80)
        
        try:
            summary_df = pd.read_csv(pipeline_summary_path)
            
            # Update with corrected values
            for result in corrected_results:
                marker_name = result['marker_name']
                marker_type = result.get('marker_type', 'unknown')
                
                # Find matching row
                mask = (summary_df['marker_name'] == marker_name) & \
                       (summary_df['marker_type'] == marker_type)
                
                if mask.any():
                    # Update corrected counts
                    if 'cluster_rejected' in result:
                        n_sig_corrected = np.sum(result['cluster_rejected'])
                    else:
                        n_sig_corrected = result.get('n_sig_clusters', 0)
                    
                    summary_df.loc[mask, 'n_sig_clusters_corrected'] = n_sig_corrected
                    summary_df.loc[mask, 'correction_method'] = result.get('correction_method', 'none')
            
            # Save updated summary
            summary_df.to_csv(pipeline_summary_path, index=False)
            print(f"✓ Updated pipeline summary: {pipeline_summary_path}")
            
        except Exception as e:
            print(f"⚠ Warning: Could not update pipeline summary: {e}")
    
    # Print completion message
    print("\n" + "="*80)
    print("CORRECTION POST-PROCESSING COMPLETED")
    print("="*80)
    print(f"Ended at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Processed: {len(corrected_results)} markers")
    print(f"Results saved to: {model_dir}")
    print("="*80)
    
    return 0


if __name__ == "__main__":
    exit(main())
