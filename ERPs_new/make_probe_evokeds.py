import os
import sys
import argparse
from typing import Dict, List

import numpy as np
import pandas as pd
import mne

# Local importsy
try:
    from helpers import (
        load_yaml_config,
        read_derivative_epochs,
        parse_events_tsv,
        enrich_events_with_parsed_fields,
        select_trials_for_probes,
        label_probe_onoff,
        detect_outlier_epochs,
        save_evoked,
        write_probe_report_html,
        load_qa_summary,
        get_qa_exclusions,
        should_process_subject_task,
    )
except Exception:
    _this_dir = os.path.dirname(__file__)
    if _this_dir not in sys.path:
        sys.path.append(_this_dir)
    from helpers import (
        load_yaml_config,
        read_derivative_epochs,
        parse_events_tsv,
        enrich_events_with_parsed_fields,
        select_trials_for_probes,
        label_probe_onoff,
        detect_outlier_epochs,
        save_evoked,
        write_probe_report_html,
        load_qa_summary,
        get_qa_exclusions,
        should_process_subject_task,
    )


def process_subject_task(cfg: Dict, subject: str, task: str) -> pd.DataFrame:
    """
    Process a single subject-task combination to generate probe-specific evoked responses.
    
    This function now includes robust outlier detection using a configurable baseline range
    that is typically longer than the aggregation range to provide more reliable statistics.
    Outlier detection is applied before epoch aggregation using z-score thresholding.
    
    Parameters
    ----------
    cfg : Dict
        Configuration dictionary containing all processing parameters
    subject : str
        Subject identifier
    task : str
        Task identifier
        
    Returns
    -------
    pd.DataFrame
        Summary dataframe with probe information and outlier statistics
    """
    project = cfg.get("project", {})
    derivatives_root = project.get("derivatives_root")
    evoked_desc_in = project.get("input_evoked_desc", "evoked")
    bids_root = project.get("bids_root")
    features_root = project.get("features_root", derivatives_root)

    if not derivatives_root:
        raise ValueError("'project.derivatives_root' must be set in config")

    trial_sel = cfg.get("trial_selection", {})
    only_go_correct = bool(trial_sel.get("only_go_correct", True))
    distance_min = int(trial_sel.get("distance_min", -5))
    distance_max = int(trial_sel.get("distance_max", -1))
    min_req = int(trial_sel.get("min_required_distances", 3))

    labeling = cfg.get("labeling", {})
    onoff_thr = int(labeling.get("onoff_threshold", 50))

    # Outlier detection configuration
    outlier_cfg = cfg.get("outlier_detection", {})
    epoch_z_threshold = float(outlier_cfg.get("epoch_z_threshold", 3.0))
    baseline_distance_min = int(outlier_cfg.get("baseline_distance_min", -10))
    baseline_distance_max = int(outlier_cfg.get("baseline_distance_max", -1))
    min_baseline_epochs = int(outlier_cfg.get("min_baseline_epochs", 5))

    evk_cfg = cfg.get("evoked", {})
    apply_baseline = bool(evk_cfg.get("apply_baseline", False))
    baseline = tuple(evk_cfg.get("baseline", (-0.1, 0.0)))

    out_cfg = cfg.get("output", {})
    desc_prefix = str(out_cfg.get("desc_prefix", "probe"))
    save_summary_csv = bool(out_cfg.get("save_summary_csv", True))
    overwrite = bool(out_cfg.get("overwrite", True))

    # Load epochs and events from derivatives
    try:
        epochs, events_tsv, events_json = read_derivative_epochs(
            derivatives_root=derivatives_root,
            subject=subject,
            task=task,
            desc=evoked_desc_in,
            preload=True,
            bids_root=bids_root,
        )
    except FileNotFoundError as exc:
        print(f"[WARN] Missing input epochs for sub-{subject} task-{task}: {exc}")
        return pd.DataFrame()

    # Parse events
    events_df = parse_events_tsv(events_tsv)
    events_df = enrich_events_with_parsed_fields(events_df)
    # Rename row_index to epoch_index for consistency with downstream functions
    events_df = events_df.rename(columns={"row_index": "epoch_index"})

    # Select trials
    selected = select_trials_for_probes(
        events_df,
        only_go_correct=only_go_correct,
        distance_min=distance_min,
        distance_max=distance_max,
        min_required_distances=min_req,
    )

    if selected.empty:
        # Helpful diagnostics
        try:
            total_n = len(events_df)
            gogood_n = int(((events_df["trial_type"] == "go") & (events_df["correctness"] == "correct")).sum())
            dist_series = events_df.loc[
                (events_df["trial_type"] == "go") & (events_df["correctness"] == "correct") & (events_df["distance_to_probe"].notna()),
                "distance_to_probe",
            ]
            unique_dists = sorted(set(int(x) for x in dist_series.values.tolist()))
            print(
                f"[WARN] No trials selected for sub-{subject} task-{task}. Diagnostics: total_events={total_n}, go_correct={gogood_n}, "
                f"unique_distances_go_correct={unique_dists}, min_required_distances={min_req} in range [{distance_min},{distance_max}]"
            )
        except Exception:
            pass
        return pd.DataFrame()

    # Direct mapping: events.tsv row order matches epochs order
    merged = selected.copy()
    # epoch_index column already exists from the rename above

    # Label probes
    probe_labels = label_probe_onoff(events_df=selected, onoff_threshold=onoff_thr)

    # Aggregate by probe: pick epochs for each probe and average
    outputs: List[Dict] = []
    for probe_num, grp in merged.groupby("probe_number", sort=False):
        grp = grp.sort_values("distance_to_probe", ascending=True)
        epoch_indices = grp["epoch_index"].astype(int).tolist()
        if not epoch_indices:
            continue

        # Apply outlier detection using broader baseline range
        valid_indices, outlier_stats = detect_outlier_epochs(
            epochs=epochs,
            events_df=events_df,  # Use full events_df for baseline computation
            probe_number=probe_num,
            baseline_distance_min=baseline_distance_min,
            baseline_distance_max=baseline_distance_max,
            min_baseline_epochs=min_baseline_epochs,
            z_threshold=epoch_z_threshold,
        )

        # Filter epoch_indices to only include valid (non-outlier) epochs that are also in the aggregation range
        final_indices = [idx for idx in epoch_indices if idx in valid_indices]
        
        if not final_indices:
            print(f"[WARN] No valid epochs remaining for probe {probe_num} after outlier removal "
                  f"(removed {outlier_stats['n_outliers']}/{outlier_stats['n_total']} baseline epochs)")
            continue

        try:
            ep = epochs.copy()[final_indices]
        except Exception as exc:
            print(f"[WARN] Failed to subset epochs for probe {probe_num}: {exc}")
            continue

        if apply_baseline:
            try:
                ep.apply_baseline(baseline=baseline)
            except Exception:
                pass

        evk = ep.average()
        label = probe_labels.get(probe_num, "unknown") if hasattr(probe_labels, "get") else (
            probe_labels.loc[probe_num] if probe_num in probe_labels.index else "unknown"
        )
        label_str = str(label)

        desc = f"{desc_prefix}-{int(probe_num):03d}_{label_str}"
        # Prepare extra metadata for the sidecar (list of distances aggregated)
        distances_used = grp.loc[grp["epoch_index"].isin(final_indices), "distance_to_probe"].astype(int).tolist()
        out_path = save_evoked(
            evoked=evk,
            features_root=features_root,
            subject=subject,
            task=task,
            desc=desc,
            overwrite=overwrite,
            extra_meta={
                "probe_number": int(probe_num),
                "label": label_str,
                "distances_used": distances_used,
                "n_trials": int(len(final_indices)),
                "n_trials_before_outlier_removal": int(len(epoch_indices)),
                "outlier_detection": outlier_stats,
            },
        )

        # Extract original epoch annotations for this probe (only for epochs that were kept)
        # Get aggregated values across epochs for this probe
        kept_grp = grp.loc[grp["epoch_index"].isin(final_indices)]
        original_annotations = {
            "original_descriptions": kept_grp["description"].tolist(),
            "original_event_ids": kept_grp["event_id"].tolist(),
            "original_onsets": kept_grp["onset"].tolist(),
            "original_durations": kept_grp["duration"].tolist() if "duration" in kept_grp.columns else [],
            "original_trial_types": kept_grp["trial_type"].tolist(),
            "original_correctness": kept_grp["correctness"].tolist(),
            "original_distances": kept_grp["distance_to_probe"].tolist(),
        }
        
        # Add optional annotation fields if they exist
        for field in ["onoff", "selfother", "valence", "time", "confidence", "average"]:
            if field in kept_grp.columns:
                original_annotations[f"original_{field}"] = kept_grp[field].tolist()

        outputs.append(
            {
                "subject": subject,
                "task": task,
                "probe_number": int(probe_num),
                "n_trials": int(len(final_indices)),
                "n_trials_before_outlier_removal": int(len(epoch_indices)),
                "n_outliers_removed": outlier_stats["n_outliers"],
                "n_baseline_epochs": outlier_stats["n_baseline_epochs"],
                "outlier_detection_mean": outlier_stats["mean"],
                "outlier_detection_std": outlier_stats["std"],
                "outlier_z_threshold": outlier_stats["z_threshold"],
                "label": label_str,
                "desc": desc,
                "path": out_path,
                **original_annotations,
            }
        )

    df_out = pd.DataFrame(outputs)

    # Optional summary CSV: keep alongside features (not in results)
    if not df_out.empty and save_summary_csv:
        out_dir = os.path.join(features_root, f"sub-{subject}")
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, f"sub-{subject}_task-{task}_probe_evokeds.csv")
        df_out.to_csv(csv_path, index=False)
        print(f"[OK] Saved summary: {csv_path}")

    # Write an HTML report with full annotation rows per probe
    try:
        report_path = write_probe_report_html(
            selected_events=selected,
            probe_labels=probe_labels,
            subject=subject,
            task=task,
            features_root=features_root,
        )
        print(f"[OK] Saved probe trials report: {report_path}")
    except Exception as exc:
        print(f"[WARN] Could not write probe report HTML: {exc}")

    return df_out


def main():
    parser = argparse.ArgumentParser(description="Generate per-probe Evoked from preprocessed evoked epochs")
    parser.add_argument("--config", default="./ERPs_new/config.yaml", help="Path to YAML config")
    parser.add_argument("--subject", help="Process only this subject")
    parser.add_argument("--task", help="Process only this task")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)

    subjects_cfg = cfg.get("subjects", [])
    if args.subject:
        subjects = [str(args.subject)]
    else:
        subjects = [str(s) for s in subjects_cfg]

    tasks_cfg = cfg.get("tasks", [])
    if args.task:
        tasks = [str(args.task)]
    else:
        tasks = [str(t) for t in tasks_cfg]

    # Load QA summary if configured
    project = cfg.get("project", {})
    qa_summary_path = project.get("qa_summary_path", None)
    exclude_failed_qa = project.get("exclude_failed_qa", False)
    
    qa_exclusions = None
    if qa_summary_path and exclude_failed_qa:
        try:
            print("\n" + "="*80)
            print("Loading QA summary for filtering...")
            print("="*80)
            qa_df = load_qa_summary(qa_summary_path, verbose=True)
            qa_exclusions = get_qa_exclusions(qa_df, epoch_type='evoked')
            print(f"\nQA Filtering: {len(qa_exclusions)} subject-task combinations will be excluded")
            if len(qa_exclusions) > 0 and len(qa_exclusions) <= 10:
                print(f"Excluded: {sorted(list(qa_exclusions))}")
            print("="*80 + "\n")
        except Exception as e:
            print(f"Warning: Failed to load QA summary: {e}")
            print("Continuing without QA filtering...\n")
            qa_exclusions = None
    elif qa_summary_path and not exclude_failed_qa:
        print("\nQA summary path provided but exclude_failed_qa is False")
        print("Skipping QA filtering...\n")
    
    # Process subjects and tasks
    n_processed = 0
    n_skipped = 0
    
    for sub in subjects:
        for t in tasks:
            # Check QA exclusions
            if qa_exclusions is not None and not should_process_subject_task(sub, t, qa_exclusions):
                print(f"[SKIP] sub-{sub} task-{t}: Failed QA")
                n_skipped += 1
                continue
            
            _ = process_subject_task(cfg, sub, t)
            n_processed += 1
    
    # Print summary
    print("\n" + "="*80)
    print("PROCESSING COMPLETE")
    print("="*80)
    print(f"Processed: {n_processed} subject-task combinations")
    if qa_exclusions is not None:
        print(f"Skipped (QA): {n_skipped} subject-task combinations")
    print("="*80)


if __name__ == "__main__":
    main()


