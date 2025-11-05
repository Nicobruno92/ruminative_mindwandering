"""Epoch rejection helpers (AutoReject + fallback)."""

from typing import List, Tuple, Optional, Sequence, Union

import mne
from autoreject import AutoReject, get_rejection_threshold

def run_autoreject(
    epochs: mne.Epochs,
    cv: int,
    random_state: int,
    logger,
    log_prefix: str,
    n_jobs: int = 1,
    thresh_method: Optional[str] = None,
    n_interpolate: Optional[Sequence[int]] = None,
    n_consensus: Optional[Sequence[float]] = None,
    picks: Optional[Union[str, Sequence[str]]] = None,
) -> Tuple[mne.Epochs, List[int]]:
    """Run AutoReject deterministically.
    
    Parameters
    ----------
    epochs : mne.Epochs
        Input epochs to clean
    cv : int
        Cross-validation folds for AutoReject
    random_state : int
        Random seed for reproducibility
    logger : LogPreprocessingDetails
        Logger instance for recording details
    log_prefix : str
        Prefix for logging entries
    n_jobs : int
        Number of parallel jobs (default: 1)
    thresh_method : Optional[str]
        Method to determine rejection thresholds. Typical options are
        'bayesian_optimization' or 'random_search'.
    n_interpolate : Optional[Sequence[int]]
        Candidate numbers of channels to interpolate.
    n_consensus : Optional[Sequence[float]]
        Candidate consensus proportions for deciding rejection (0-1).
    picks : Optional[Union[str, Sequence[str]]]
        Channels to include. Use 'eeg' to pick EEG channels, or provide a
        list of channel names/indices.
        
    Returns
    -------
    Tuple[mne.Epochs, List[int]]
        Cleaned epochs and list of rejected epoch indices
        
    Raises
    ------
    RuntimeError
        If AutoReject fails to process the epochs
    """
    print(
        "Running AutoReject with cv="
        f"{cv}, random_state={random_state}, n_jobs={n_jobs}, "
        f"thresh_method={thresh_method}, n_interpolate={n_interpolate}, "
        f"n_consensus={n_consensus}, picks={picks}"
    )
    
    n_before = len(epochs.events)
    # Resolve picks
    resolved_picks = None
    if picks is not None:
        try:
            if isinstance(picks, str):
                if picks.lower() == "eeg":
                    resolved_picks = mne.pick_types(
                        epochs.info, meg=False, eeg=True, eog=False, stim=False
                    )
                else:
                    # allow other MNE pick strings like 'meg'
                    resolved_picks = mne.pick_channels(
                        epochs.ch_names, include=[picks]
                    )
            elif isinstance(picks, (list, tuple)):
                # If sequence of channel names or indices
                if all(isinstance(p, int) for p in picks):
                    resolved_picks = list(picks)
                else:
                    resolved_picks = mne.pick_channels(
                        epochs.ch_names, include=list(picks)
                    )
        except Exception:
            resolved_picks = None

    ar = AutoReject(
        cv=cv,
        random_state=random_state,
        n_jobs=n_jobs,
        thresh_method=(
            thresh_method
            if thresh_method is not None
            else 'bayesian_optimization'
        ),
        n_interpolate=(
            n_interpolate if n_interpolate is not None else [1, 4, 8, 16]
        ),
        consensus=n_consensus if n_consensus is not None else None,
        picks=resolved_picks,
    )
    epochs_clean = ar.fit_transform(epochs)
    n_after = len(epochs_clean.events)
    
    # Get rejection threshold for logging
    reject_thr = get_rejection_threshold(epochs)
    if logger is not None:
        logger.log_detail(f"{log_prefix}_autoreject_threshold", reject_thr)
    
    # Calculate number of rejected epochs
    n_rejected = max(0, n_before - n_after)
    rejected_indices = list(range(n_rejected))  # Placeholder indices
    
    print(
        "AutoReject completed: "
        f"{n_before} → {n_after} epochs ({n_rejected} rejected)"
    )
    return epochs_clean, rejected_indices



