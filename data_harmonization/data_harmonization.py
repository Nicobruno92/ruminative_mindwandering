"""
Data Harmonization Pipeline for CyberSART EEG Study

This script reads raw BrainVision EEG data, applies marker correction/harmonization,
sets up the montage, and writes BIDS-compliant output ready for preprocessing.

The pipeline:
1. Reads BrainVision files from data root
2. Applies CACS-64 montage
3. Runs TriggerCorrector for annotation harmonization
4. Resamples if needed
5. Writes BIDS-compliant EEG data with QA reporting

Usage:
    python data_harmonization.py --config config.yaml
    python data_harmonization.py --subject 02 --task Sart1
    python data_harmonization.py --subject 02 --session a --task Sart1
"""

import os
import sys
from typing import Dict, Any, List, Optional

import mne

# YAML config
try:
    import yaml
except ImportError as exc:
    raise ImportError("PyYAML is required. Please install with `pip install pyyaml`.") from exc

# Add utils directory to path
_this_dir = os.path.dirname(os.path.abspath(__file__))
_utils_dir = os.path.join(_this_dir, "utils")
if _utils_dir not in sys.path:
    sys.path.insert(0, _utils_dir)
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

from utils.bids_compliance_harmonized import BIDSComplianceHarmonized
from utils.io_helpers import DataFinder, MontageHelper, EventRecoder, maybe_set_meas_date_from_name, read_brainvision_safe
from utils.metadata_helpers import load_metadata, write_participants
from utils.trigger_correction import TriggerCorrector
from utils.qa_report import HarmonizationQAReport


# Default configuration
DEFAULT_CONFIG = {
    "project": {
        "data_root": "/network/iss/cenir/analyse/meeg/CYBERSART/_RAW_DATA",
        "raw_root": "/network/iss/cenir/analyse/meeg/CYBERSART/BIDS/raw",
        "dataset_name": "CyberSART EEG Dataset",
        "montage": "CACS-64_REF.bvef",
        "resample_hz": 500,
        "line_freq_hz": 50,
        "overwrite": True,
    },
    "tasks": ["Sart1", "Sart2", "Sart3", "Sart4"],
}


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Parameters
    ----------
    config_path : str
        Path to YAML configuration file
        
    Returns
    -------
    dict
        Configuration dictionary
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    # Merge with defaults
    merged = DEFAULT_CONFIG.copy()
    if cfg:
        for key, value in cfg.items():
            if isinstance(value, dict) and key in merged:
                merged[key].update(value)
            else:
                merged[key] = value
    
    return merged


def process_subject_task(
    cfg: Dict[str, Any],
    subject: str,
    task: str,
    session: Optional[str] = None
) -> Dict[str, Any]:
    """
    Process a single subject-task (and optional session) combination.
    
    Parameters
    ----------
    cfg : dict
        Configuration dictionary
    subject : str
        Subject identifier
    task : str
        Task name
    session : str, optional
        Session identifier
        
    Returns
    -------
    dict
        Processing result with status and metadata
    """
    result = {
        "subject": subject,
        "session": session or "n/a",
        "task": task,
        "status": "success",
        "message": None,
        "n_eeg_channels": None,
        "sfreq_original": None,
        "sfreq_final": None,
        "n_annotations": None,
        "annotation_harmonization_applied": False,
    }
    
    project = cfg["project"]
    data_root = project["data_root"]
    raw_root = project["raw_root"]
    brainvision_pattern = project.get("brainvision_pattern", "CYBERSART_*_{task}.vhdr")
    montage_filename = project.get("montage", "CACS-64_REF.bvef")
    resample_hz = int(project.get("resample_hz", 500))
    line_freq_hz = project.get("line_freq_hz", 50)
    overwrite = bool(project.get("overwrite", True))

    finder = DataFinder(data_root, brainvision_pattern)
    montage_helper = MontageHelper(montage_filename, line_freq_hz)
    bids_handler = BIDSComplianceHarmonized(
        bids_root=raw_root, 
        dataset_name=project.get("dataset_name")
    )

    try:
        # Locate and read raw
        vhdr_path = finder.find_brainvision(subject, task, session=session)
        print(f"[INFO] Reading BrainVision: {vhdr_path}")
        raw = read_brainvision_safe(vhdr_path)
        
        result["n_eeg_channels"] = len([ch for ch in raw.ch_names if 'EOG' not in ch.upper()])
        result["sfreq_original"] = raw.info["sfreq"]

        # Optional meas_date from filename
        maybe_set_meas_date_from_name(raw, vhdr_path)

        # Montage and channel types
        raw = montage_helper.apply(raw)
        print(f"[INFO] Montage applied: {montage_filename}")

        # Run trigger correction to add harmonized annotations
        try:
            tc = TriggerCorrector(raw)
            _events, _event_id = tc.process_annotations()
            raw._trigger_corrector_df = tc.df
            result["annotation_harmonization_applied"] = True
            print(f"[INFO] Annotation harmonization applied")
        except Exception as exc:
            print(f"[WARN] TriggerCorrector failed: {exc}")
            raw._trigger_corrector_df = None
            result["message"] = f"TriggerCorrector warning: {exc}"

        # Ensure BAD_rest periods are explicitly marked
        try:
            ann = raw.annotations
            descriptions = list(ann.description)
            onsets = list(ann.onset)
            durations = list(ann.duration)

            existing_bad_intervals = set(
                (float(onsets[i]), float(durations[i]))
                for i, d in enumerate(descriptions)
                if isinstance(d, str) and d.lower().startswith('bad')
            )
            to_add_onsets, to_add_durs, to_add_desc = [], [], []
            for i, d in enumerate(descriptions):
                if d == 'BAD_rest':
                    key = (float(onsets[i]), float(durations[i]))
                    if key not in existing_bad_intervals:
                        to_add_onsets.append(float(onsets[i]))
                        to_add_durs.append(float(durations[i]))
                        to_add_desc.append('bad_rest')
            if to_add_onsets:
                extra_bad = mne.Annotations(
                    onset=to_add_onsets,
                    duration=to_add_durs,
                    description=to_add_desc,
                    orig_time=ann.orig_time,
                )
                raw.set_annotations(ann + extra_bad)
        except Exception as exc:
            print(f"[WARN] Marking BAD_rest as bad annotations failed: {exc}")

        # Resampling
        if raw.info["sfreq"] != resample_hz:
            print(f"[INFO] Resampling from {raw.info['sfreq']} Hz to {resample_hz} Hz")
            raw.load_data()
            raw.resample(sfreq=resample_hz)
        
        result["sfreq_final"] = raw.info["sfreq"]
        result["n_annotations"] = len(raw.annotations)

        # Write BIDS
        session_str = f", ses-{session}" if session else ""
        print(f"[INFO] Writing BIDS raw for sub-{subject}{session_str}, task-{task} -> {raw_root}")
        bids_handler.write_raw(
            raw=raw, 
            subject=subject, 
            task=task, 
            session=session,
            overwrite=overwrite
        )
        print(f"[OK] Saved sub-{subject}{session_str}_task-{task}")
        
    except FileNotFoundError as e:
        result["status"] = "skipped"
        result["message"] = str(e)
        print(f"[SKIP] {e}")
        
    except Exception as e:
        result["status"] = "error"
        result["message"] = str(e)
        print(f"[ERROR] {e}")
    
    return result


def process_all(
    cfg: Dict[str, Any],
    qa_report: Optional[HarmonizationQAReport] = None
) -> List[Dict[str, Any]]:
    """
    Process all subjects, sessions, and tasks based on configuration.
    
    Parameters
    ----------
    cfg : dict
        Configuration dictionary
    qa_report : HarmonizationQAReport, optional
        QA report object to log results to
        
    Returns
    -------
    list
        List of processing results
    """
    project = cfg["project"]
    data_root = project["data_root"]
    
    finder = DataFinder(data_root)
    results = []
    
    # Get subjects from config or discover
    subjects_cfg = cfg.get("subjects", None)
    if subjects_cfg:
        if isinstance(subjects_cfg, str):
            subjects_str = subjects_cfg.strip()
            if subjects_str.startswith("(") and subjects_str.endswith(")"):
                subjects_str = subjects_str[1:-1]
            subjects = [
                s.strip() for s in subjects_str.replace(",", " ").split() 
                if s.strip()
            ]
        else:
            subjects = [str(s) for s in subjects_cfg]
    else:
        subjects = finder.list_subjects()
    
    # Get sessions from config (None means no sessions)
    sessions_cfg = cfg.get("sessions", None)
    
    # Get tasks
    tasks = cfg.get("tasks", ["Sart1", "Sart2", "Sart3", "Sart4"])
    
    print(f"[INFO] Processing {len(subjects)} subjects")
    print(f"[INFO] Tasks: {tasks}")
    if sessions_cfg:
        print(f"[INFO] Sessions: {sessions_cfg}")
    
    for subject in subjects:
        # Get sessions for this subject
        if sessions_cfg:
            sessions = sessions_cfg if isinstance(sessions_cfg, list) else [sessions_cfg]
        else:
            # Check if subject has sessions
            discovered_sessions = finder.list_sessions(subject)
            sessions = discovered_sessions if discovered_sessions else [None]
        
        for session in sessions:
            for task in tasks:
                result = process_subject_task(cfg, subject, task, session=session)
                results.append(result)
                
                if qa_report is not None:
                    qa_report.add_result(result)
    
    return results


def print_summary(results: List[Dict[str, Any]]) -> None:
    """Print processing summary."""
    success = sum(1 for r in results if r["status"] == "success")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    errors = sum(1 for r in results if r["status"] == "error")
    
    print("\n" + "=" * 60)
    print("PROCESSING SUMMARY")
    print("=" * 60)
    print(f"  Success: {success}")
    print(f"  Skipped: {skipped}")
    print(f"  Errors:  {errors}")
    print("=" * 60)
    
    if errors > 0:
        print("\nErrors:")
        for r in results:
            if r["status"] == "error":
                session_str = f"_ses-{r['session']}" if r.get('session') and r['session'] != 'n/a' else ""
                print(f"  - sub-{r['subject']}{session_str}_task-{r['task']}: {r['message']}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Data harmonization pipeline for CyberSART EEG"
    )
    parser.add_argument(
        "--config", 
        default="./config.yaml", 
        help="Path to YAML config file"
    )
    parser.add_argument("--subject", help="Process only this subject")
    parser.add_argument("--session", help="Process only this session (optional)")
    parser.add_argument("--task", help="Process only this task")
    parser.add_argument(
        "--no-qa-report",
        action="store_true",
        help="Disable QA report generation"
    )
    args = parser.parse_args()
    
    # Load config
    if os.path.exists(args.config):
        cfg = load_config(args.config)
    else:
        print(f"[WARN] Config file not found: {args.config}, using defaults")
        cfg = DEFAULT_CONFIG.copy()
    
    project = cfg["project"]
    raw_root = project["raw_root"]
    
    # Single subject/task mode
    if args.subject and args.task:
        result = process_subject_task(
            cfg, args.subject, args.task, session=args.session
        )
        print_summary([result])
        
        if not args.no_qa_report:
            qa_report = HarmonizationQAReport(
                output_dir=raw_root,
                report_name="harmonization_qa_report"
            )
            qa_report.add_result(result)
            saved_paths = qa_report.save_all()
            print(f"\n[QA REPORT] Saved/updated report:")
            for fmt, path in saved_paths.items():
                print(f"  - {fmt.upper()}: {path}")
        return
    
    # Optionally write participants.tsv/json using metadata
    metadata_csv = cfg.get("project", {}).get("metadata_csv")
    if metadata_csv and os.path.exists(metadata_csv):
        subjects_cfg = cfg.get("subjects", [])
        if isinstance(subjects_cfg, str):
            subjects_str = subjects_cfg.strip()
            if subjects_str.startswith("(") and subjects_str.endswith(")"):
                subjects_str = subjects_str[1:-1]
            subjects_list = [s.strip() for s in subjects_str.replace(",", " ").split() if s.strip()]
        else:
            subjects_list = [str(s) for s in subjects_cfg]
        
        tasks = cfg.get("tasks", [])
        df_meta = load_metadata(metadata_csv)
        write_participants(raw_root, subjects_list, tasks, df_meta)
    
    # Process all - create cumulative QA report
    qa_report = None
    if not args.no_qa_report:
        qa_report = HarmonizationQAReport(
            output_dir=raw_root,
            report_name="harmonization_qa_report"
        )
    
    results = process_all(cfg, qa_report=qa_report)
    
    # Save QA report
    if qa_report is not None:
        saved_paths = qa_report.save_all()
        qa_report.print_summary()
        print(f"\n[QA REPORT] Saved cumulative report to:")
        for fmt, path in saved_paths.items():
            print(f"  - {fmt.upper()}: {path}")
    else:
        print_summary(results)


if __name__ == "__main__":
    main()


