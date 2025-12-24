import os
import glob
from datetime import datetime

import mne

# Ensure utils import for trigger correction
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.trigger_correction import TriggerCorrector
import tempfile


class DataFinder:
    """Helper to locate input EEG files following a given pattern."""

    def __init__(self, data_root: str, brainvision_pattern: str = "CYBERSART_*_{task}.vhdr") -> None:
        self.data_root = os.path.abspath(data_root)
        self.pattern = brainvision_pattern

    def _subject_variants(self, subject: str) -> list[str]:
        s = str(subject)
        variants = []
        if s.upper().startswith("S"):
            core = s[1:]
            variants.extend([s, f"S0{core}", f"S00{core}"])
        else:
            # ensure numeric core with zero padding
            try:
                num = int(s)
            except Exception:
                num = s
            variants.extend([
                str(s),
                f"S{num}",
                f"S{num:02d}" if isinstance(num, int) else f"S0{s}",
                f"S{num:03d}" if isinstance(num, int) else f"S00{s}",
            ])
        # de-duplicate preserving order
        seen = set()
        out = []
        for v in variants:
            if v not in seen:
                seen.add(v)
                out.append(v)
        return out

    def find_brainvision(self, subject: str, task: str) -> str:
        # Try multiple subject variants and extensions
        tried = []
        patterns = [
            self.pattern,
            self.pattern.replace('.vhdr', '.VHDR'),
            self.pattern.replace('{task}.vhdr', '{task}*.vhdr'),
            self.pattern.replace('{task}.vhdr', '{task}*.VHDR'),
        ]
        for s_var in self._subject_variants(subject):
            eeg_dir = os.path.join(self.data_root, f"sub-{s_var}", "eeg")
            for pat in patterns:
                search = pat.format(task=task)
                full = os.path.join(eeg_dir, search)
                matches = glob.glob(full)
                tried.append(full)
                if matches:
                    if len(matches) > 1:
                        print(f"[WARN] Multiple matches found. Using the first one: {matches[0]}")
                    return matches[0]
        raise FileNotFoundError("No BrainVision file found. Tried: " + "; ".join(tried))


class MontageHelper:
    """Helper to set montage and channel types on Raw objects."""

    def __init__(self, montage_filename: str, line_freq_hz: int | None = 50) -> None:
        self.montage_filename = montage_filename
        self.line_freq_hz = line_freq_hz

    def resolve_montage_path(self) -> str:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidate = os.path.join(script_dir, self.montage_filename)
        if not os.path.exists(candidate):
            raise FileNotFoundError(f"Montage file not found: {candidate}")
        return candidate

    def apply(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        # Mark EOG if present
        eog_candidates = {}
        for ch_name in ["VEOG", "HEOG"]:
            if ch_name in raw.ch_names:
                eog_candidates[ch_name] = "eog"
        if eog_candidates:
            raw.set_channel_types(eog_candidates)

        montage_path = self.resolve_montage_path()
        montage = mne.channels.read_custom_montage(montage_path)
        raw.set_montage(montage)
        if self.line_freq_hz is not None:
            raw.info["line_freq"] = int(self.line_freq_hz)
        return raw


class EventRecoder:
    """Helper to recode annotations on a Raw object using TriggerCorrector."""

    def __init__(self) -> None:
        pass

    def recode_inplace(self, raw: mne.io.BaseRaw) -> None:
        processor = TriggerCorrector(raw)
        processor.recode_annotations()
        processor.retropropagate_tp_values()
        processor.retropropagate_tp_to_trials()
        processor.propagate_trial_info()
        processor.fix_probe_distances()
        # Ensure THOUGHT_PROBE rows aggregate all answers and trials carry ontask/offtask
        processor.aggregate_thought_probe_info()
        processor.add_onoff_label()
        # Add binarized labels based on config thresholds
        processor.append_binary_labels()

        # Replace only descriptions; keep original onset/duration/orig_time to avoid timebase issues
        df = processor.df
        if "recoded" not in df.columns:
            raise RuntimeError("TriggerCorrector did not produce 'recoded' column")
        recoded_descriptions = [str(x) for x in df["recoded"].tolist()]

        ann = raw.annotations
        raw.set_annotations(
            mne.Annotations(
                onset=ann.onset.copy(),
                duration=ann.duration.copy(),
                description=recoded_descriptions,
                orig_time=ann.orig_time,
            )
        )


def maybe_set_meas_date_from_name(raw: mne.io.BaseRaw, filename: str) -> None:
    parts = os.path.basename(filename).split("_")
    if len(parts) >= 2 and parts[1].isdigit() and len(parts[1]) >= 8:
        try:
            date = datetime.strptime(parts[1][:10], "%Y-%m-%d")
            raw.set_meas_date(date)
        except Exception:
            pass


def _clean_ref_value(value: str, exts: tuple[str, ...]) -> str:
    v = value.strip().strip('"').strip("'")
    # If extension appears more than once, keep up to the first occurrence
    for ext in exts:
        if ext in v:
            first = v.split(ext)[0] + ext
            v = first
            break
    # Reduce to basename
    return os.path.basename(v)


def read_brainvision_safe(vhdr_path: str) -> mne.io.BaseRaw:
    """Read BrainVision file, sanitizing header references to .vmrk and .eeg if needed.

    Creates a temporary cleaned .vhdr that points to absolute .vmrk and .eeg paths to
    avoid malformed references in original headers.
    """
    vhdr_path = os.path.abspath(vhdr_path)
    base_dir = os.path.dirname(vhdr_path)

    with open(vhdr_path, 'r', encoding='latin-1') as f:
        lines = f.readlines()

    new_lines = []
    data_abs = None
    mrk_abs = None
    for line in lines:
        if line.strip().lower().startswith('datafile='):
            val = line.split('=', 1)[1]
            clean = _clean_ref_value(val, ('.eeg', '.EEG'))
            cand = os.path.join(base_dir, clean)
            if not os.path.exists(cand):
                # try to locate by glob
                stem = os.path.splitext(os.path.basename(vhdr_path))[0]
                alt = glob.glob(os.path.join(base_dir, f"{stem}*.eeg"))
                cand = alt[0] if alt else cand
            data_abs = os.path.abspath(cand)
            new_lines.append(f"DataFile={data_abs}\n")
        elif line.strip().lower().startswith('markerfile='):
            val = line.split('=', 1)[1]
            clean = _clean_ref_value(val, ('.vmrk', '.VMRK'))
            cand = os.path.join(base_dir, clean)
            if not os.path.exists(cand):
                stem = os.path.splitext(os.path.basename(vhdr_path))[0]
                alt = glob.glob(os.path.join(base_dir, f"{stem}*.vmrk"))
                cand = alt[0] if alt else cand
            mrk_abs = os.path.abspath(cand)
            new_lines.append(f"MarkerFile={mrk_abs}\n")
        else:
            new_lines.append(line)

    # If nothing was changed, read normally
    if data_abs is None and mrk_abs is None:
        return mne.io.read_raw_brainvision(vhdr_path, preload=False, verbose=False)

    # Write cleaned header to temp file
    with tempfile.NamedTemporaryFile('w', suffix='.vhdr', delete=False) as tmp:
        tmp.write(''.join(new_lines))
        tmp_vhdr = tmp.name

    try:
        raw = mne.io.read_raw_brainvision(tmp_vhdr, preload=False, verbose=False)
    finally:
        try:
            os.remove(tmp_vhdr)
        except Exception:
            pass
    return raw


