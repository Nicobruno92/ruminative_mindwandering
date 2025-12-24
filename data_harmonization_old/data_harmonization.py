import os
import sys

import mne

# YAML config
try:
    import yaml
except ImportError as exc:
    raise ImportError("PyYAML is required. Please install with `pip install pyyaml`.") from exc

from bids_compliance import BIDSCompliance
from io_helpers import DataFinder, MontageHelper, EventRecoder, maybe_set_meas_date_from_name, read_brainvision_safe
from metadata_helpers import load_metadata, write_participants

# Ensure we can import utilities from project root
try:
    from utils.trigger_correction import TriggerCorrector
except Exception:
    # Add project root (parent of this file's directory) to sys.path and retry
    _this_dir = os.path.dirname(__file__)
    _project_root = os.path.abspath(os.path.join(_this_dir, os.pardir))
    if _project_root not in sys.path:
        sys.path.append(_project_root)
    from utils.trigger_correction import TriggerCorrector


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def process_subject_task(cfg: dict, subject: str, task: str) -> None:
    # Instantiate helpers based on config
    project = cfg["project"]
    data_root = project["data_root"]
    raw_root = project["raw_root"]
    brainvision_pattern = project.get("brainvision_pattern", "CYBERSART_*_{task}.vhdr")
    montage_filename = project.get("montage", "CACS-64_withREF.bvef")
    resample_hz = int(project.get("resample_hz", 500))
    line_freq_hz = project.get("line_freq_hz", 50)
    overwrite = bool(project.get("overwrite", True))

    finder = DataFinder(data_root, brainvision_pattern)
    montage = MontageHelper(montage_filename, line_freq_hz)
    recoder = EventRecoder()
    bidsio = BIDSCompliance(bids_root=raw_root, dataset_name=project.get("dataset_name"))

    # Locate and read raw
    vhdr_path = finder.find_brainvision(subject, task)
    print(f"[INFO] Reading BrainVision: {vhdr_path}")
    raw = read_brainvision_safe(vhdr_path)

    # Optional meas_date from filename
    maybe_set_meas_date_from_name(raw, vhdr_path)

    # Montage and channel types
    raw = montage.apply(raw)

    # Run trigger correction to add harmonized annotations, including BAD_rest and THOUGHT_PROBE
    try:
        tc = TriggerCorrector(raw)
        _events, _event_id = tc.process_annotations()
        # BAD_rest segments are now present in raw.annotations and will be treated as bad
        # Store the TriggerCorrector dataframe for later use (contains rt, timestamps, etc.)
        raw._trigger_corrector_df = tc.df
    except Exception as exc:
        print(f"[WARN] TriggerCorrector failed: {exc}")
        raw._trigger_corrector_df = None

    # Ensure BAD_rest periods are explicitly marked as bad annotations for downstream processing
    try:
        ann = raw.annotations
        descriptions = list(ann.description)
        onsets = list(ann.onset)
        durations = list(ann.duration)

        # Build set of existing bad intervals to avoid duplicates
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

    # Recode events
    recoder.recode_inplace(raw)

    # Resampling
    if raw.info["sfreq"] != resample_hz:
        print(f"[INFO] Resampling from {raw.info['sfreq']} Hz to {resample_hz} Hz")
        raw.load_data()
        raw.resample(sfreq=resample_hz)

    # Write BIDS
    print(f"[INFO] Writing BIDS raw for sub-{subject}, task-{task} -> {raw_root}")
    bidsio.write_raw(raw, subject=subject, task=task, overwrite=overwrite)
    print("[OK] Saved")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Data harmonization I/O for EEG to BIDS")
    parser.add_argument("--config", default="./Preprocessing_pipeline_new/config.yaml", help="Path to YAML config")
    parser.add_argument("--subject", help="Process only this subject")
    parser.add_argument("--task", help="Process only this task")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.subject and args.task:
        process_subject_task(cfg, args.subject, args.task)
        return

    subjects_cfg = cfg.get("subjects", [])
    if not subjects_cfg:
        raise ValueError("Config must contain a 'subjects' list")

    # Accept subjects as list or string like "(02 03 04)" or "02 03 04"
    if isinstance(subjects_cfg, str):
        subjects_str = subjects_cfg.strip()
        if subjects_str.startswith("(") and subjects_str.endswith(")"):
            subjects_str = subjects_str[1:-1]
        subjects_list = [s.strip() for s in subjects_str.replace(",", " ").split() if s.strip()]
    else:
        subjects_list = [str(s) for s in subjects_cfg]

    tasks = cfg.get("tasks", [])
    if not tasks:
        raise ValueError("No tasks provided in config under 'tasks'")

    # Optionally write participants.tsv/json using metadata_experiment.csv
    metadata_csv = cfg.get("project", {}).get("metadata_csv")
    if metadata_csv and os.path.exists(metadata_csv):
        df_meta = load_metadata(metadata_csv)
        write_participants(cfg["project"]["raw_root"], subjects_list, tasks, df_meta)

    for subject_id in subjects_list:
        for task in tasks:
            process_subject_task(cfg, subject_id, str(task))


if __name__ == "__main__":
    main()


