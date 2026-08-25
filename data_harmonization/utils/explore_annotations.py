#%%
import os
import sys
import argparse
from pathlib import Path
import pandas as pd

import mne
from mne.viz import set_browser_backend
from mne_bids import BIDSPath, read_raw_bids

# Make project utils importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Preprocessing_pipeline_new.io_helpers import read_brainvision_safe


def load_raw(vhdr: str | None, bids_root: str | None, subject: str | None, task: str | None, preload: bool) -> mne.io.BaseRaw:
    if vhdr:
        raw = read_brainvision_safe(vhdr)
        if preload:
            raw.load_data()
        return raw
    if bids_root and subject and task:
        bp = BIDSPath(root=bids_root, subject=str(subject), task=str(task), datatype='eeg', suffix='eeg', extension='.vhdr', check=False)
        raw = read_raw_bids(bp, verbose=False)
        if preload:
            raw.load_data()
        return raw
    raise ValueError("Provide either --vhdr or the trio --bids-root --subject --task")


def summarize_annotations(raw: mne.io.BaseRaw) -> pd.DataFrame:
    ann = raw.annotations
    if ann is None or len(ann) == 0:
        print("No annotations found.")
        return pd.DataFrame(columns=["onset", "duration", "description"])  # empty

    df = ann.to_data_frame()
    print("\nAnnotation summary (top 30 by count):")
    counts = df["description"].value_counts().head(30)
    for desc, cnt in counts.items():
        print(f"  {desc}: {cnt}")
    return df


def summarize_events(raw: mne.io.BaseRaw) -> tuple[pd.DataFrame, dict]:
    try:
        events, event_id = mne.events_from_annotations(raw, event_id='auto')
    except Exception as e:
        print(f"Failed to extract events from annotations: {e}")
        return pd.DataFrame(columns=["sample", "previous", "event_id"]), {}

    df_events = pd.DataFrame(events, columns=["sample", "previous", "event_id"])  # sample index space
    print(f"\nEvents extracted: {len(df_events)} unique ids={len(event_id)}")
    # invert mapping for readability
    id_to_desc = {v: k for k, v in event_id.items()}
    top_ids = df_events["event_id"].value_counts().head(20)
    print("Top event IDs:")
    for eid, cnt in top_ids.items():
        label = id_to_desc.get(int(eid), "?")
        print(f"  {eid} ({label}): {cnt}")
    return df_events, event_id


def plot_inspection(raw: mne.io.BaseRaw, df_ann: pd.DataFrame, df_events: pd.DataFrame) -> None:
    try:
        raw.plot(block=False, title="Raw with annotations")
    except Exception as e:
        print(f"raw.plot failed: {e}")
    try:
        if len(df_events) > 0:
            ev = df_events[["sample", "previous", "event_id"]].to_numpy()
            mne.viz.plot_events(ev, sfreq=raw.info["sfreq"], first_samp=raw.first_samp)
    except Exception as e:
        print(f"plot_events failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Explore EEG annotations and events with MNE")
    parser.add_argument("--vhdr", help="Path to a BrainVision .vhdr file")
    parser.add_argument("--bids-root", help="Path to BIDS root (raw)")
    parser.add_argument("--subject", help="Subject ID (e.g., 02)")
    parser.add_argument("--task", help="Task label (e.g., Sart1)")
    parser.add_argument("--config", default="./Preprocessing_pipeline_new/config.yaml", help="Path to YAML config (optional; used for defaults)")
    parser.add_argument("--preload", action="store_true", help="Preload raw data")
    parser.add_argument("--plot", action="store_true", help="Show interactive plots")
    parser.add_argument("--browser-backend", default="auto", choices=["auto", "qt", "matplotlib", "pyqtgraph"], help="MNE browser backend to use")
    parser.add_argument("--save-html", help="If plotting fails or for headless use, save an HTML report to this path")
    parser.add_argument("--save-annots", help="Optional path to save annotations TSV")
    parser.add_argument("--save-events", help="Optional path to save events TSV")

    args = parser.parse_args()

    # If no explicit inputs provided, fall back to config + defaults (subject=02, task=Sart1)
    if not args.vhdr and not (args.bids_root and args.subject and args.task):
        try:
            import yaml  # lazy import
            cfg_path = Path(args.config).resolve()
            with open(cfg_path, "r") as f:
                cfg = yaml.safe_load(f)
            # Pull raw_root from config
            project = cfg.get("project", {})
            args.bids_root = args.bids_root or project.get("raw_root")
            # Defaults for quick-run
            args.subject = args.subject or "02"
            args.task = args.task or "Sart1"
            print(f"[INFO] Using defaults: bids_root={args.bids_root}, subject={args.subject}, task={args.task}")
        except Exception as e:
            raise ValueError("Provide either --vhdr or the trio --bids-root --subject --task (and ensure config if using defaults)") from e

    raw = load_raw(args.vhdr, args.bids_root, args.subject, args.task, preload=args.preload)

    # Summaries
    df_ann = summarize_annotations(raw)
    df_events, event_id = summarize_events(raw)

    # Try interactive plot, with backend selection and headless fallback
    plotted = False
    try:
        if args.browser_backend != "auto":
            set_browser_backend(args.browser_backend)
        elif "DISPLAY" not in os.environ:
            # likely headless; prefer matplotlib backend to avoid Qt
            try:
                set_browser_backend("matplotlib")
            except Exception:
                pass
        raw.plot(block=True, title=f"Raw sub-{args.subject} task-{args.task}")
        plotted = True
    except Exception as e:
        print(f"raw.plot failed: {e}")

    # Fallback: save HTML report for headless environments
    if not plotted or args.save_html:
        try:
            report_path = args.save_html or os.path.abspath("explore_raw_report.html")
            rep = mne.Report(title=f"Explore Raw sub-{args.subject} task-{args.task}")
            rep.add_raw(raw=raw, title="Raw", psd=True)
            rep.save(report_path, overwrite=True, open_browser=False)
            print(f"Saved HTML report to: {report_path}")
        except Exception as e:
            print(f"Saving HTML report failed: {e}")

    # Optional save
    if args.save_annots:
        # Keep core columns if present
        cols = [c for c in ["onset", "duration", "description"] if c in df_ann.columns]
        df_ann[cols].to_csv(args.save_annots, sep='\t', index=False)
        print(f"Saved annotations TSV to {args.save_annots}")
    if args.save_events:
        df_events.to_csv(args.save_events, sep='\t', index=False)
        print(f"Saved events TSV to {args.save_events}")

    # Optional plots
    if args.plot:
        plot_inspection(raw, df_ann, df_events)


if __name__ == "__main__":
    main()



