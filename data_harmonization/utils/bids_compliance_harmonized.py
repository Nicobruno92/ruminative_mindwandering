"""
Harmonized BIDS Compliance Utility for CyberSART EEG Dataset

This module provides a unified BIDS-compliant I/O interface that supports:
- Multiple data types (EEG, gaze, behavioral)
- Optional sessions and runs
- Task-specific organization
- Comprehensive metadata handling
- Derivative data management

Extends the base BIDSCompliance class with session/run support for standardization
across different EEG studies.
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Union, Any

import mne
import numpy as np
import pandas as pd
from mne_bids import BIDSPath, write_raw_bids, read_raw_bids


class BIDSComplianceHarmonized:
    """
    Unified BIDS compliance handler with session/run support.
    
    Supports raw data, epochs, evoked responses, and derivative data
    with full session/run/task hierarchy and comprehensive metadata.
    """

    def __init__(self, bids_root: str, dataset_name: Optional[str] = None) -> None:
        """
        Initialize BIDS compliance handler.
        
        Parameters
        ----------
        bids_root : str
            Root directory for BIDS dataset
        dataset_name : str, optional
            Name of the dataset for description file
        """
        self.bids_root = os.path.abspath(bids_root)
        self.dataset_name = dataset_name or "CyberSART EEG Dataset"
        self._ensure_dataset_description()

    def _ensure_dataset_description(self) -> None:
        """Create dataset_description.json if it doesn't exist."""
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
                    "Name": "CyberSART Data Harmonization Pipeline",
                    "Version": "1.0",
                    "Description": "Harmonized BIDS-compliant data processing for CyberSART study",
                }
            ],
            "Date": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        with open(desc_path, "w") as f:
            json.dump(content, f, indent=2)

    def build_bids_path(
        self, 
        subject: str, 
        task: str, 
        session: Optional[str] = None,
        run: Optional[str] = None,
        datatype: str = "eeg",
        suffix: Optional[str] = None,
        extension: str = ".fif"
    ) -> BIDSPath:
        """
        Build a BIDS path with full session/run/task support.
        
        Parameters
        ----------
        subject : str
            Subject identifier (e.g., '01', 'sub-01')
        task : str
            Task name (e.g., 'Sart1', 'Sart2')
        session : str, optional
            Session identifier (e.g., '01', 'baseline')
        run : str, optional
            Run number (e.g., '01', '02')
        datatype : str
            Data type (e.g., 'eeg', 'gaze', 'beh')
        suffix : str, optional
            File suffix (defaults to datatype)
        extension : str
            File extension
        
        Returns
        -------
        BIDSPath
            Configured BIDS path object
        """
        subject = str(subject).replace('sub-', '')
        
        bids_path = BIDSPath(
            subject=subject,
            session=session,
            run=run,
            task=task,
            datatype=datatype,
            suffix=suffix or datatype,
            extension=extension,
            root=self.bids_root,
            check=False,
        )
        return bids_path

    def write_raw(
        self, 
        raw: mne.io.BaseRaw, 
        subject: str, 
        task: str,
        session: Optional[str] = None,
        run: Optional[str] = None,
        datatype: str = "eeg",
        overwrite: bool = True,
    ) -> str:
        """
        Write raw data in FIF format to preserve montage information.
        
        FIF format is used instead of BrainVision because it preserves
        electrode positions (montage) which are lost in BrainVision format.
        
        Parameters
        ----------
        raw : mne.io.BaseRaw
            Raw data object
        subject : str
            Subject identifier
        task : str
            Task name
        session : str, optional
            Session identifier
        run : str, optional
            Run identifier
        datatype : str
            Data type (eeg, gaze, etc.)
        overwrite : bool
            Whether to overwrite existing files
            
        Returns
        -------
        str
            Path to saved file
        """
        subject = str(subject).replace('sub-', '')
        
        # Build output directory path
        out_parts = [self.bids_root, f"sub-{subject}"]
        if session:
            out_parts.append(f"ses-{session}")
        out_parts.append(datatype)
        out_dir = os.path.join(*out_parts)
        os.makedirs(out_dir, exist_ok=True)
        
        # Build filename
        fname_parts = [f"sub-{subject}"]
        if session:
            fname_parts.append(f"ses-{session}")
        fname_parts.append(f"task-{task}")
        if run:
            fname_parts.append(f"run-{run}")
        fname_parts.append(f"{datatype}.fif")
        fname = "_".join(fname_parts)
        
        out_path = os.path.join(out_dir, fname)
        
        # Save in FIF format to preserve montage
        raw.save(out_path, overwrite=overwrite)
        print(f"[INFO] Saved raw FIF (preserves montage): {out_path}")
        
        # Write JSON sidecar with extended metadata
        sidecar = self._build_extended_sidecar(
            info=raw.info,
            task=task,
            recording_type="continuous",
            recording_duration=raw.times[-1] if raw.times is not None else None,
        )
        
        # Add montage info to sidecar
        montage = raw.get_montage()
        if montage is not None:
            sidecar["MontagePreserved"] = True
            sidecar["MontageName"] = getattr(montage, 'name', 'custom') or 'custom'
        else:
            sidecar["MontagePreserved"] = False
        
        json_path = out_path.replace(".fif", ".json")
        with open(json_path, "w") as f:
            json.dump(sidecar, f, indent=2)
        
        # Write channels.tsv
        self._write_complete_channels_tsv(
            out_path=out_path,
            info=raw.info,
            bads_set=set(raw.info.get("bads", [])),
        )
        
        # Write events from annotations
        self._write_raw_events(
            bids_root=self.bids_root, subject=subject, task=task, 
            session=session, run=run, raw=raw
        )
        
        return out_path

    def read_raw(
        self, 
        subject: str, 
        task: str,
        session: Optional[str] = None,
        run: Optional[str] = None,
        datatype: str = "eeg",
        preload: bool = False
    ) -> mne.io.BaseRaw:
        """
        Read raw FIF data with preserved montage.
        
        Reads FIF format files which preserve electrode positions (montage).
        Falls back to BrainVision format for backward compatibility.
        
        Parameters
        ----------
        subject : str
            Subject identifier
        task : str
            Task name
        session : str, optional
            Session identifier
        run : str, optional
            Run identifier
        datatype : str
            Data type
        preload : bool
            Whether to preload data
        
        Returns
        -------
        mne.io.BaseRaw
            Loaded raw data with montage preserved
        """
        subject = str(subject).replace('sub-', '')
        
        # Build file path for FIF format
        path_parts = [self.bids_root, f"sub-{subject}"]
        if session:
            path_parts.append(f"ses-{session}")
        path_parts.append(datatype)
        
        fname_parts = [f"sub-{subject}"]
        if session:
            fname_parts.append(f"ses-{session}")
        fname_parts.append(f"task-{task}")
        if run:
            fname_parts.append(f"run-{run}")
        fname_parts.append(f"{datatype}.fif")
        fname = "_".join(fname_parts)
        
        fif_path = os.path.join(*path_parts, fname)
        
        # Try FIF first (preserves montage)
        if os.path.exists(fif_path):
            raw = mne.io.read_raw_fif(fif_path, preload=preload, verbose=False)
            print(f"[INFO] Loaded raw FIF (montage preserved): {fif_path}")
            return raw
        
        # Fallback to BrainVision format for backward compatibility
        vhdr_fname = fname.replace(".fif", ".vhdr")
        vhdr_path = os.path.join(*path_parts[:-1], datatype, vhdr_fname)
        
        if os.path.exists(vhdr_path):
            print(f"[WARN] Loading BrainVision format (montage may be lost): {vhdr_path}")
            raw = mne.io.read_raw_brainvision(vhdr_path, preload=preload, verbose=False)
            return raw
        
        # Try using mne-bids as last resort
        try:
            bids_path = self.build_bids_path(
                subject=subject, task=task, session=session, run=run, 
                datatype=datatype, extension=".vhdr"
            )
            raw = read_raw_bids(bids_path=bids_path, verbose=False)
            if preload:
                raw.load_data()
            print(f"[WARN] Loaded via mne-bids (montage may be lost)")
            return raw
        except Exception as e:
            raise FileNotFoundError(
                f"Could not find raw data for sub-{subject}, task-{task}. "
                f"Tried FIF ({fif_path}) and BrainVision formats. Error: {e}"
            )

    # ---------- Derivatives I/O ----------
    
    def _deriv_dir(
        self, 
        derivatives_root: str, 
        subject: str, 
        session: Optional[str] = None,
        datatype: str = "eeg"
    ) -> str:
        """Create derivatives directory path."""
        parts = [os.path.abspath(derivatives_root), f"sub-{subject}"]
        if session:
            parts.append(f"ses-{session}")
        parts.append(datatype)
        
        out_dir = os.path.join(*parts)
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def _make_deriv_fname(
        self, 
        subject: str, 
        task: str, 
        session: Optional[str] = None,
        run: Optional[str] = None,
        suffix: str = "eeg", 
        desc: Optional[str] = None, 
        extension: str = ".fif"
    ) -> str:
        """Create derivative filename with full hierarchy."""
        parts = [f"sub-{subject}"]
        
        if session:
            parts.append(f"ses-{session}")
        if task:
            parts.append(f"task-{task}")
        if run:
            parts.append(f"run-{run}")
        if desc:
            parts.append(f"desc-{desc}")
            
        parts.append(f"{suffix}{extension}")
        return "_".join(parts)

    def _make_bids_basename(
        self, 
        subject: str, 
        session: Optional[str], 
        task: str, 
        run: Optional[str] = None,
        suffix: str = "eeg", 
        extension: str = ".vhdr"
    ) -> str:
        """Create BIDS-compliant basename with full hierarchy."""
        parts = [f"sub-{subject}"]
        
        if session:
            parts.append(f"ses-{session}")
        if task:
            parts.append(f"task-{task}")
        if run:
            parts.append(f"run-{run}")
            
        parts.append(f"{suffix}{extension}")
        return "_".join(parts)

    def write_derivative_raw(
        self,
        raw: mne.io.BaseRaw,
        derivatives_root: str,
        subject: str,
        task: str,
        session: Optional[str] = None,
        run: Optional[str] = None,
        desc: Optional[str] = None,
        overwrite: bool = True,
        bad_channels_pre_interp: Optional[List[str]] = None
    ) -> str:
        """
        Write processed raw data to derivatives folder.
        
        Parameters
        ----------
        raw : mne.io.BaseRaw
            Raw data object to save
        derivatives_root : str
            Root directory for derivatives
        subject : str
            Subject identifier
        task : str
            Task name
        session : str, optional
            Session identifier
        run : str, optional
            Run identifier
        desc : str, optional
            Description suffix (e.g., 'icaClean', 'csd')
        overwrite : bool
            Whether to overwrite existing files
        bad_channels_pre_interp : List[str], optional
            List of bad channels before interpolation (for metadata)
        
        Returns
        -------
        str
            Path to saved file
        """
        out_dir = self._deriv_dir(
            derivatives_root, subject, session=session, datatype="eeg"
        )
        fname = self._make_deriv_fname(
            subject, task, session=session, run=run,
            suffix="eeg", desc=desc, extension=".fif"
        )
        out_path = os.path.join(out_dir, fname)
        raw.save(out_path, overwrite=overwrite)
        
        sidecar = self._build_extended_sidecar(
            info=raw.info,
            task=task,
            recording_type="continuous",
            recording_duration=raw.times[-1] if raw.times is not None else None,
        )
        
        if desc:
            sidecar["ProcessingDescription"] = desc
        
        if bad_channels_pre_interp:
            sidecar["BadChannelsPreInterpolation"] = bad_channels_pre_interp
        
        with open(out_path.replace(".fif", ".json"), "w") as f:
            json.dump(sidecar, f, indent=2)

        self._write_complete_channels_tsv(
            out_path=out_path,
            info=raw.info,
            bads_set=set(raw.info.get("bads", [])),
        )
        
        return out_path

    def write_derivative_epochs(
        self,
        epochs: mne.Epochs,
        derivatives_root: str,
        subject: str,
        task: str,
        session: Optional[str] = None,
        run: Optional[str] = None,
        desc: Optional[str] = None,
        overwrite: bool = True,
        condition_metadata: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Write epoched data with comprehensive metadata.
        
        Parameters
        ----------
        epochs : mne.Epochs
            Epochs object to save
        derivatives_root : str
            Root directory for derivatives
        subject : str
            Subject identifier
        task : str
            Task name
        session : str, optional
            Session identifier
        run : str, optional
            Run identifier
        desc : str, optional
            Description suffix
        overwrite : bool
            Whether to overwrite existing files
        condition_metadata : dict, optional
            Additional condition metadata
        
        Returns
        -------
        str
            Path to saved file
        """
        out_dir = self._deriv_dir(
            derivatives_root, subject, session=session, datatype="eeg"
        )
        fname = self._make_deriv_fname(
            subject, task, session=session, run=run, 
            suffix="epo", desc=desc, extension=".fif"
        )
        out_path = os.path.join(out_dir, fname)
        epochs.save(out_path, overwrite=overwrite)
        
        epoch_len = float(epochs.tmax - epochs.tmin) if epochs.tmax is not None and epochs.tmin is not None else None
        total_duration = float(len(epochs)) * epoch_len if epoch_len is not None else None
        
        sidecar = self._build_extended_sidecar(
            info=epochs.info,
            task=task,
            recording_type="epoched",
            recording_duration=total_duration,
        )
        
        sidecar.update({
            "EpochCount": int(len(epochs)),
            "Tmin": float(epochs.tmin),
            "Tmax": float(epochs.tmax),
        })
        
        if condition_metadata:
            sidecar.update(condition_metadata)
        
        with open(out_path.replace(".fif", ".json"), "w") as f:
            json.dump(sidecar, f, indent=2)

        self._write_complete_channels_tsv(
            out_path=out_path,
            info=epochs.info,
            bads_set=set(epochs.info.get("bads", [])),
        )

        self._write_epochs_events(out_path, epochs)
        
        return out_path

    def write_derivative_ica(
        self,
        ica: mne.preprocessing.ICA,
        derivatives_root: str,
        subject: str,
        task: str,
        session: Optional[str] = None,
        run: Optional[str] = None,
        desc: Optional[str] = None,
        overwrite: bool = True,
    ) -> str:
        """Write ICA decomposition to derivatives folder."""
        out_dir = self._deriv_dir(derivatives_root, subject, session=session, datatype="eeg")
        if desc is None:
            desc = "ica"
        fname = self._make_deriv_fname(
            subject, task, session=session, run=run,
            suffix="ica", desc=desc, extension=".fif"
        )
        out_path = os.path.join(out_dir, fname)
        ica.save(out_path, overwrite=overwrite)
        
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

    # ---------- Reading Methods ----------
    
    def read_derivative_epochs(
        self,
        derivatives_root: str,
        subject: str,
        task: str,
        session: Optional[str] = None,
        run: Optional[str] = None,
        desc: Optional[str] = None,
        preload: bool = False,
        proj: Union[bool, str] = True,
    ) -> mne.Epochs:
        """
        Read derivative epochs.
        
        Parameters
        ----------
        derivatives_root : str
            Root directory for derivatives
        subject : str
            Subject identifier
        task : str
            Task name
        session : str, optional
            Session identifier
        run : str, optional
            Run identifier
        desc : str, optional
            Description suffix
        preload : bool
            Whether to preload data
        proj : bool or str
            Whether to apply projections
        
        Returns
        -------
        mne.Epochs
            Loaded epochs
        """
        out_dir = self._deriv_dir(
            derivatives_root, subject, session=session, datatype="eeg"
        )
        fname = self._make_deriv_fname(
            subject, task, session=session, run=run,
            suffix="epo", desc=desc, extension=".fif"
        )
        path = os.path.join(out_dir, fname)
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        
        try:
            epochs = mne.read_epochs(path, proj=proj, preload=preload, verbose=False)
        except TypeError:
            epochs = mne.read_epochs(path, preload=preload, verbose=False)
        return epochs

    # ---------- Helper Methods ----------

    def _write_raw_events(
        self, 
        bids_root: str, 
        subject: str, 
        task: str,
        session: Optional[str] = None,
        run: Optional[str] = None,
        raw: mne.io.BaseRaw = None
    ) -> None:
        """Write events.tsv and events.json for raw data."""
        try:
            annotations = getattr(raw, "annotations", None)
            if annotations is None or len(annotations) == 0:
                return

            sf = float(raw.info.get("sfreq", 0.0) or 0.0)
            _, event_id = mne.events_from_annotations(raw)
            id_map = {str(k): int(v) for k, v in event_id.items()} if event_id else {}

            tc_df = getattr(raw, '_trigger_corrector_df', None)
            
            rows = []
            for idx, ann in enumerate(annotations):
                desc = ann["description"]
                onset = float(ann["onset"])
                duration = float(ann["duration"]) if ann["duration"] is not None else 0.0
                sample = int(round(onset * sf)) if sf > 0 else 0
                value = id_map.get(str(desc), None)
                
                row = {
                    "onset": onset,
                    "duration": duration,
                    "sample": sample,
                    "value": value if value is not None else "n/a",
                    "trial_type": str(desc),
                }
                
                if tc_df is not None and idx < len(tc_df):
                    extra_cols = ['rt', 'stim_timestamp', 'response_timestamp', 
                                  'rt_2nd', 'stim_timestamp_2nd', 'response_timestamp_2nd',
                                  'correctness']
                    for col in extra_cols:
                        if col in tc_df.columns:
                            val = tc_df.iloc[idx][col]
                            if pd.notna(val):
                                if hasattr(val, 'isoformat'):
                                    row[col] = val.isoformat()
                                else:
                                    row[col] = val
                            else:
                                row[col] = None
                
                rows.append(row)

            df = pd.DataFrame(rows)
            
            events_dir = os.path.join(os.path.abspath(bids_root), f"sub-{subject}")
            if session:
                events_dir = os.path.join(events_dir, f"ses-{session}")
            events_dir = os.path.join(events_dir, "eeg")
            
            events_basename = self._make_bids_basename(
                subject=subject, session=session, task=task, run=run,
                suffix="events", extension=".tsv"
            )
            events_tsv = os.path.join(events_dir, events_basename)
            
            os.makedirs(os.path.dirname(events_tsv), exist_ok=True)
            df.to_csv(events_tsv, sep="\t", index=False)

            events_json = events_tsv.replace(".tsv", ".json")
            meta = {
                "onset": {
                    "Description": (
                        "Onset (in seconds) of the event from the beginning of the first datapoint."
                    ),
                    "Units": "s",
                },
                "duration": {
                    "Description": (
                        "Duration of the event in seconds from onset."
                    ),
                    "Units": "s",
                },
                "sample": {
                    "Description": "The event onset time in number of sampling points.",
                },
                "value": {
                    "Description": "The event code associated with the event.",
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

    def _write_epochs_events(self, out_path: str, epochs: mne.Epochs) -> None:
        """Write events.tsv for epochs."""
        try:
            ev = np.array(epochs.events)
            if ev.size:
                sf = float(epochs.info["sfreq"])
                onsets = ev[:, 0] / sf
                durations = np.zeros(len(onsets))
                ids = ev[:, 2]
                
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
                
                meta_path = tsv_path.replace(".tsv", ".json")
                with open(meta_path, "w") as f:
                    json.dump(
                        {"event_id": {str(k): int(v) for k, v in (epochs.event_id or {}).items()}}, 
                        f, indent=2
                    )
        except Exception:
            pass

    def _write_complete_channels_tsv(
        self, 
        out_path: str, 
        info: mne.Info, 
        bads_set: Optional[set] = None
    ) -> None:
        """Write comprehensive channels.tsv file."""
        try:
            channel_names = list(info.get("ch_names", []))
            if not channel_names:
                return

            sampling_frequency = float(info.get("sfreq", 0.0) or 0.0)
            low_cutoff = info.get("highpass", None)
            high_cutoff = info.get("lowpass", None)

            def _describe_and_units(channel_type: str) -> tuple:
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

    def _compute_channel_type_counts(self, info: mne.Info) -> Dict[str, int]:
        """Compute counts per channel type from MNE info."""
        counts: Dict[str, int] = {}
        for idx, _ in enumerate(info.get("chs", [])):
            try:
                t = mne.channel_type(info, idx)
            except Exception:
                t = "unknown"
            t_u = t.upper() if isinstance(t, str) else str(t)
            counts[t_u] = counts.get(t_u, 0) + 1
        return counts

    def _build_extended_sidecar(
        self, 
        info: mne.Info, 
        task: str, 
        recording_type: str, 
        recording_duration: Optional[float]
    ) -> dict:
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


# ---------- Backward Compatibility Functions ----------

def make_bids_basename(
    subject: str, 
    session: Optional[str], 
    task: str, 
    suffix: str, 
    extension: str, 
    desc: Optional[str] = None,
    run: Optional[str] = None
) -> str:
    """Create a BIDS-compliant basename (backward compatibility function)."""
    parts = [f"sub-{subject}"]
    
    if session:
        parts.append(f"ses-{session}")
    if task:
        parts.append(f"task-{task}")
    if run:
        parts.append(f"run-{run}")
    if desc:
        parts.append(f"desc-{desc}")
        
    parts.append(f"{suffix}{extension}")
    return "_".join(parts)


def save_raw_bids_compliant(
    subject: str, 
    session: Optional[str], 
    task: str, 
    data_type: str, 
    raw: mne.io.BaseRaw, 
    root_folder: str,
    run: Optional[str] = None
) -> None:
    """Save raw data in BIDS-compliant format (backward compatibility function)."""
    bids_handler = BIDSComplianceHarmonized(root_folder)
    bids_handler.write_raw(
        raw=raw, 
        subject=subject, 
        task=task, 
        session=session, 
        run=run,
        datatype=data_type
    )
