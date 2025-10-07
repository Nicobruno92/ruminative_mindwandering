import os
import json
from datetime import datetime

import mne
import numpy as np
import pandas as pd
from mne_bids import BIDSPath, write_raw_bids, read_raw_bids


class BIDSCompliance:
    """Utility class to handle BIDS-compliant I/O for EEG raw data."""

    def __init__(self, bids_root: str, dataset_name: str | None = None) -> None:
        self.bids_root = os.path.abspath(bids_root)
        self.dataset_name = dataset_name or "EEG Multicenter Harmonized Dataset"
        self._ensure_dataset_description()

    def _ensure_dataset_description(self) -> None:
        os.makedirs(self.bids_root, exist_ok=True)
        desc_path = os.path.join(self.bids_root, "dataset_description.json")
        if os.path.exists(desc_path):
            return
        content = {
            "Name": self.dataset_name,
            "BIDSVersion": "1.8.0",
            "DatasetType": "raw",
            "GeneratedBy": [
                {
                    "Name": "Custom EEG data harmonization pipeline",
                    "Version": "1.0",
                    "Description": "Loads raw EEG, applies harmonization, and writes BIDS-compliant data.",
                }
            ],
            "Date": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        with open(desc_path, "w") as f:
            json.dump(content, f, indent=2)

    def build_bids_path(self, subject: str, task: str) -> BIDSPath:
        return BIDSPath(
            subject=str(subject),
            task=str(task),
            datatype="eeg",
            suffix="eeg",
            extension=".vhdr",
            root=self.bids_root,
            check=False,
        )

    def write_raw(self, raw: mne.io.BaseRaw, subject: str, task: str, overwrite: bool = True) -> None:
        bids_path = self.build_bids_path(subject, task)
        write_raw_bids(
            raw=raw,
            bids_path=bids_path,
            format="BrainVision",
            allow_preload=True,
            overwrite=overwrite,
            events=None,
            verbose=False,
        )
        # Ensure events.tsv and events.json exist with full metadata
        self._write_raw_events(
            bids_root=str(bids_path.root), subject=subject, task=task, raw=raw
        )

    def read_raw(self, subject: str, task: str, preload: bool = False) -> mne.io.BaseRaw:
        """Read BIDS raw. Compatible with mne-bids versions without 'preload' arg.

        If preload=True, calls raw.load_data() after reading.
        """
        bids_path = self.build_bids_path(subject, task)
        # Some mne-bids versions do not accept 'preload' kwarg.
        raw = read_raw_bids(bids_path=bids_path, verbose=False)
        if preload:
            try:
                raw.load_data()
            except Exception:
                pass
        return raw

    def _write_raw_events(self, bids_root: str, subject: str, task: str, raw: mne.io.BaseRaw) -> None:
        """Write events.tsv and events.json for raw data from annotations.

        The TSV will include onset, duration, sample, value, trial_type.
        The JSON will include the BIDS field descriptions and an event_id map.
        """
        try:
            annotations = getattr(raw, "annotations", None)
            if annotations is None or len(annotations) == 0:
                return

            sf = float(raw.info.get("sfreq", 0.0) or 0.0)
            # Derive numeric codes using MNE's mapping
            _, event_id = mne.events_from_annotations(raw)
            id_map = {str(k): int(v) for k, v in event_id.items()} if event_id else {}

            rows = []
            for ann in annotations:
                desc = ann["description"]
                onset = float(ann["onset"])  # seconds
                duration = float(ann["duration"]) if ann["duration"] is not None else 0.0
                sample = int(round(onset * sf)) if sf > 0 else 0
                value = id_map.get(str(desc), None)
                rows.append({
                    "onset": onset,
                    "duration": duration,
                    "sample": sample,
                    "value": value if value is not None else "n/a",
                    "trial_type": str(desc),
                })

            df = pd.DataFrame(rows)
            events_tsv = os.path.join(
                os.path.abspath(bids_root), f"sub-{subject}", "eeg", f"sub-{subject}_task-{task}_events.tsv"
            )
            df.to_csv(events_tsv, sep="\t", index=False)

            # JSON sidecar with standard descriptions plus event_id mapping
            events_json = events_tsv.replace(".tsv", ".json")
            meta = {
                "onset": {
                    "Description": (
                        "Onset (in seconds) of the event from the beginning of the first datapoint. "
                        "Negative onsets account for events before the first stored data point."
                    ),
                    "Units": "s",
                },
                "duration": {
                    "Description": (
                        "Duration of the event in seconds from onset. Must be zero, positive, or 'n/a' if unavailable. "
                        "A zero value indicates an impulse event. "
                    ),
                    "Units": "s",
                },
                "sample": {
                    "Description": "The event onset time in number of sampling points.First sample is 0.",
                },
                "value": {
                    "Description": "The event code (also known as trigger code or event ID) associated with the event.",
                },
                "trial_type": {
                    "Description": "The type, category, or name of the event.",
                },
            }
            if id_map:
                meta["event_id"] = id_map
            with open(events_json, "w") as f:
                json.dump(meta, f, indent=2)
        except Exception:
            pass

    def _write_complete_channels_tsv(self, out_path: str, info: mne.Info, bads_set: set[str] | None = None) -> None:
        """Write a complete channels.tsv alongside a derivative file.

        Columns: name, type, units, low_cutoff, high_cutoff, description,
        sampling_frequency, status, status_description

        Parameters
        ----------
        out_path : str
            Base path to the saved derivative file ('.fif'). The TSV will use
            the same base name with suffix '_channels.tsv'.
        info : mne.Info
            MNE info containing channel metadata.
        bads_set : set[str] | None
            Set of channel names considered bad at save time.
        """
        try:
            channel_names = list(info.get("ch_names", []))
            if not channel_names:
                return

            sampling_frequency = float(info.get("sfreq", 0.0) or 0.0)
            low_cutoff = info.get("highpass", None)
            high_cutoff = info.get("lowpass", None)

            def _describe_and_units(channel_type: str) -> tuple[str, str]:
                ch = (channel_type or "").upper()
                if ch == "EEG":
                    return "ElectroEncephaloGram", "µV"
                if ch == "EOG":
                    return "ElectroOculoGram", "µV"
                if ch == "EMG":
                    return "ElectroMyoGram", "µV"
                if ch == "ECG":
                    return "ElectroCardioGram", "µV"
                return ch if ch else "unknown", "µV"

            rows = []
            bads = set(bads_set or [])
            for idx, name in enumerate(channel_names):
                ch_type = mne.io.pick.channel_type(info, idx)
                desc, units = _describe_and_units(ch_type)
                status = "bad" if name in bads else "good"
                status_desc = "pre-interpolation bad" if status == "bad" else "n/a"
                rows.append({
                    "name": name,
                    "type": ch_type.upper() if isinstance(ch_type, str) else str(ch_type),
                    "units": units,
                    "low_cutoff": float(low_cutoff) if low_cutoff is not None else None,
                    "high_cutoff": float(high_cutoff) if high_cutoff is not None else None,
                    "description": desc,
                    "sampling_frequency": sampling_frequency,
                    "status": status,
                    "status_description": status_desc,
                })

            df = pd.DataFrame(rows)
            tsv_path = out_path.replace(".fif", "_channels.tsv")
            df.to_csv(tsv_path, sep="\t", index=False)
        except Exception:
            pass

    def _compute_channel_type_counts(self, info: mne.Info) -> dict[str, int]:
        """Compute counts per channel type from MNE info."""
        counts: dict[str, int] = {}
        for idx, _ in enumerate(info.get("chs", [])):
            try:
                t = mne.channel_type(info, idx)
            except Exception:
                t = "unknown"
            t_u = t.upper() if isinstance(t, str) else str(t)
            counts[t_u] = counts.get(t_u, 0) + 1
        return counts

    def _build_extended_sidecar(self, info: mne.Info, task: str, recording_type: str, recording_duration: float | None) -> dict:
        """Build extended JSON sidecar fields shared across derivatives."""
        counts = self._compute_channel_type_counts(info)
        sidecar = {
            "TaskName": task,
            "Manufacturer": info.get("manufacturer", "n/a") or "n/a",
            "PowerLineFrequency": info.get("line_freq", None),
            "SamplingFrequency": info.get("sfreq", None),
            "SoftwareFilters": "n/a",
            "RecordingDuration": float(recording_duration) if recording_duration is not None else None,
            "RecordingType": recording_type,
            "EEGReference": "n/a",
            "EEGGround": info.get("custom_ref_applied", "n/a") if isinstance(info.get("custom_ref_applied", None), str) else "n/a",
            "EEGPlacementScheme": "based on the extended 10/20 system",
            "EEGChannelCount": counts.get("EEG", 0),
            "EOGChannelCount": counts.get("EOG", 0),
            "ECGChannelCount": counts.get("ECG", 0),
            "EMGChannelCount": counts.get("EMG", 0),
            "MiscChannelCount": counts.get("MISC", 0),
            "TriggerChannelCount": counts.get("STIM", 0) + counts.get("TRIG", 0),
        }
        return sidecar

    # ------------- Derivatives I/O -------------
    def _deriv_dir(self, derivatives_root: str, subject: str, datatype: str = "eeg") -> str:
        out_dir = os.path.join(os.path.abspath(derivatives_root), f"sub-{subject}", datatype)
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def _make_deriv_fname(self, subject: str, task: str, suffix: str, desc: str | None, extension: str) -> str:
        if desc:
            return f"sub-{subject}_task-{task}_desc-{desc}_{suffix}{extension}"
        return f"sub-{subject}_task-{task}_{suffix}{extension}"

    def write_derivative_raw(
        self,
        raw: mne.io.BaseRaw,
        derivatives_root: str,
        subject: str,
        task: str,
        desc: str | None = None,
        overwrite: bool = True,
        bad_channels_pre_interp: list[str] | None = None,
    ) -> str:
        out_dir = self._deriv_dir(derivatives_root, subject, datatype="eeg")
        fname = self._make_deriv_fname(subject, task, suffix="eeg", desc=desc, extension=".fif")
        out_path = os.path.join(out_dir, fname)
        raw.save(out_path, overwrite=overwrite)
        # Extended sidecar metadata for continuous data
        duration = (float(raw.n_times) / float(raw.info.get("sfreq", 1.0))) if raw.n_times and raw.info.get("sfreq", None) else None
        sidecar = self._build_extended_sidecar(
            info=raw.info,
            task=task,
            recording_type="continuous",
            recording_duration=duration,
        )
        # Reference information (if average projector applied)
        sidecar["EEGReference"] = (
            "average (applied)" if any(p.get("desc", "") == "Average" for p in raw.info.get("projs", [])) else sidecar.get("EEGReference", "n/a")
        )
        with open(out_path.replace(".fif", ".json"), "w") as f:
            json.dump(sidecar, f, indent=2)

        # Write comprehensive channels.tsv
        self._write_complete_channels_tsv(
            out_path=out_path,
            info=raw.info,
            bads_set=set(bad_channels_pre_interp or []),
        )
        return out_path

    def write_derivative_ica(
        self,
        ica: mne.preprocessing.ICA,
        derivatives_root: str,
        subject: str,
        task: str,
        desc: str | None = None,
        overwrite: bool = True,
    ) -> str:
        out_dir = self._deriv_dir(derivatives_root, subject, datatype="eeg")
        if desc is None:
            desc = "ica"
        fname = self._make_deriv_fname(subject, task, suffix="ica", desc=desc, extension=".fif")
        out_path = os.path.join(out_dir, fname)
        ica.save(out_path, overwrite=overwrite)
        # Ensure JSON-serializable types (avoid numpy.int64)
        try:
            excl = [int(x) for x in (getattr(ica, "exclude", []) or [])]
        except Exception:
            excl = []
        meta = {
            "n_components": int(getattr(ica, "n_components_", 0)),
            "exclude": excl,
        }
        with open(out_path.replace(".fif", ".json"), "w") as f:
            json.dump(meta, f, indent=2)
        return out_path

    def write_derivative_epochs(
        self,
        epochs: mne.Epochs,
        derivatives_root: str,
        subject: str,
        task: str,
        desc: str | None = None,
        overwrite: bool = True,
    ) -> str:
        out_dir = self._deriv_dir(derivatives_root, subject, datatype="eeg")
        fname = self._make_deriv_fname(subject, task, suffix="epo", desc=desc, extension=".fif")
        out_path = os.path.join(out_dir, fname)
        epochs.save(out_path, overwrite=overwrite)
        # Extended sidecar for epoched data
        epoch_len = float(epochs.tmax - epochs.tmin) if epochs.tmax is not None and epochs.tmin is not None else None
        total_duration = float(len(epochs)) * epoch_len if (epoch_len is not None) else None
        sidecar = self._build_extended_sidecar(
            info=epochs.info,
            task=task,
            recording_type="epoched",
            recording_duration=total_duration,
        )
        # Add epoch-specific fields
        sidecar.update({
            "EpochCount": int(len(epochs)),
            "Tmin": float(epochs.tmin),
            "Tmax": float(epochs.tmax),
        })
        with open(out_path.replace(".fif", ".json"), "w") as f:
            json.dump(sidecar, f, indent=2)

        # Write channels.tsv for epochs using epochs.info
        self._write_complete_channels_tsv(
            out_path=out_path,
            info=epochs.info,
            bads_set=set(epochs.info.get("bads", [])),
        )

        # Write an events.tsv alongside, mapping sample-based events to onsets
        try:
            ev = np.array(epochs.events)
            if ev.size:
                sf = float(epochs.info["sfreq"])
                onsets = ev[:, 0] / sf
                durations = np.zeros(len(onsets))
                ids = ev[:, 2]
                # reverse map to strings if available
                id_to_desc = {v: k for k, v in (epochs.event_id or {}).items()}
                descriptions = [id_to_desc.get(int(i), str(int(i))) for i in ids]
                df = pd.DataFrame({
                    "onset": onsets,
                    "duration": durations,
                    "description": descriptions,
                    "event_id": ids.astype(int),
                })
                tsv_path = out_path.replace(".fif", "_events.tsv")
                df.to_csv(tsv_path, sep="\t", index=False)
                # Save mapping JSON
                meta_path = tsv_path.replace(".tsv", ".json")
                with open(meta_path, "w") as f:
                    json.dump({"event_id": {str(k): int(v) for k, v in (epochs.event_id or {}).items()}}, f, indent=2)
        except Exception:
            pass
        return out_path

    # --------- Public helpers for reading derivative epochs ---------
    def build_derivative_epochs_path(
        self,
        derivatives_root: str,
        subject: str,
        task: str,
        desc: str | None = None,
    ) -> str:
        out_dir = self._deriv_dir(derivatives_root, subject, datatype="eeg")
        fname = self._make_deriv_fname(subject, task, suffix="epo", desc=desc, extension=".fif")
        return os.path.join(out_dir, fname)

    def read_derivative_epochs(
        self,
        derivatives_root: str,
        subject: str,
        task: str,
        desc: str | None = None,
        preload: bool = False,
        proj: bool | str = True,
    ) -> mne.Epochs:
        path = self.build_derivative_epochs_path(
            derivatives_root=derivatives_root, subject=subject, task=task, desc=desc
        )
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        # proj can be True, False or 'delayed' (MNE behavior)
        try:
            epochs = mne.read_epochs(path, proj=proj, preload=preload, verbose=False)
        except TypeError:
            # Older MNE versions may not support proj kwarg in read_epochs
            epochs = mne.read_epochs(path, preload=preload, verbose=False)
        return epochs



