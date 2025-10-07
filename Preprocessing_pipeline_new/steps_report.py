"""Reporting helpers to keep the main pipeline clean and readable."""

import mne
import matplotlib.pyplot as plt


def compute_psd_figure(
    raw: mne.io.BaseRaw,
    fmin: float = 0.5,
    fmax: float = 100.0,
    n_fft: int = 2048,
) -> object:
    """
    Compute and return a PSD figure for the given raw data.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Raw EEG data.
    fmin : float, optional
        Minimum frequency for PSD (default: 0.5 Hz).
    fmax : float, optional
        Maximum frequency for PSD (default: 100.0 Hz).
    n_fft : int, optional
        Length of FFT (default: 2048).

    Returns
    -------
    object
        Matplotlib figure object with the PSD plot.
    """
    psd = raw.compute_psd(fmin=fmin, fmax=fmax, n_fft=n_fft, verbose=False)
    fig = psd.plot(average=True, show=False)
    return fig


def add_psd_figures(report: mne.Report, fig_pre, fig_post) -> None:
    """Add PSD figures to report and clean up memory."""
    # Close figures after adding to free memory
    try:
        if fig_pre is not None:
            report.add_figure(fig_pre, title="PSD before notch filter")
            try:
                plt.close(fig_pre)
            except Exception:
                pass
    except Exception:
        try:
            plt.close(fig_pre)
        except Exception:
            pass
    try:
        if fig_post is not None:
            report.add_figure(fig_post, title="PSD after notch filter")
            try:
                plt.close(fig_post)
            except Exception:
                pass
    except Exception:
        try:
            plt.close(fig_post)
        except Exception:
            pass


def add_excluded_ics(
    report: mne.Report, ica: mne.preprocessing.ICA, inst
) -> None:
    """Add excluded ICA components to report."""
    try:
        if ica is not None and ica.exclude:
            fig = ica.plot_components(picks=ica.exclude, inst=inst, show=False)
            if isinstance(fig, list):
                for i, f in enumerate(fig):
                    report.add_figure(f, title=f"Excluded ICs {i}")
                    try:
                        plt.close(f)
                    except Exception:
                        pass
            else:
                report.add_figure(fig, title="Excluded ICs")
                try:
                    plt.close(fig)
                except Exception:
                    pass
    except Exception:
        pass


def add_drop_log(
    report: mne.Report, epochs, title: str = "Epochs drop log"
) -> None:
    """Add epochs drop log figure to report."""
    try:
        fig = epochs.plot_drop_log(show=False)
        report.add_figure(fig, title=title)
        try:
            plt.close(fig)
        except Exception:
            pass
    except Exception:
        pass


def add_epoch_rejection_summary(
    report: mne.Report,
    label: str,
    n_input_total: int | None,
    n_after_construct: int | None,
    n_dropped_pre_ar: int | None,
    n_rejected_by_ar: int | None,
    n_dropped_post_ar: int | None,
    include_qa_metrics: bool = True,
):
    """Add a concise text summary of epoch rejections to the report.

    Parameters
    ----------
    label : str
        Section label (e.g., 'Evoked', 'State')
    n_input_total : int | None
        Number of candidate epochs at construction (len(drop_log))
    n_after_construct : int | None
        Epochs constructed (len(epochs) after creation)
    n_dropped_pre_ar : int | None
        Drops at construction due to annotations/bounds
    n_rejected_by_ar : int | None
        Epochs rejected by AutoReject
    n_dropped_post_ar : int | None
        Final drops after calling drop_bad on cleaned epochs
    include_qa_metrics : bool
        Whether to include QA-relevant metrics in summary
    """
    try:
        def _pct(numer: int | None, denom: int | None) -> str:
            if numer is None or denom is None or denom == 0:
                return "N/A"
            return f"{100.0 * numer / denom:.1f}%"

        lines = []
        lines.append(f"=== {label} Epochs Rejection Summary ===")
        
        if n_input_total is not None:
            lines.append(f"Total candidate epochs: {n_input_total}")
        if n_after_construct is not None:
            lines.append(f"Epochs after construction: {n_after_construct}")
            if n_input_total is not None:
                excluded_at_construct = n_input_total - n_after_construct
                lines.append(f"Excluded at construction: {excluded_at_construct} "
                            f"({_pct(excluded_at_construct, n_input_total)})")
        if n_dropped_pre_ar is not None and n_dropped_pre_ar > 0:
            lines.append(f"Dropped pre-AutoReject: {n_dropped_pre_ar}")
        if n_rejected_by_ar is not None:
            lines.append(f"Rejected by AutoReject: {n_rejected_by_ar}")
            if n_after_construct is not None:
                lines.append(f"AutoReject rejection rate: "
                           f"{_pct(n_rejected_by_ar, n_after_construct)}")
        if n_dropped_post_ar is not None and n_dropped_post_ar > 0:
            lines.append(f"Dropped post-AutoReject: {n_dropped_post_ar}")
        
        if include_qa_metrics:
            lines.append("")
            lines.append("*** QA Assessment ***")
            if n_rejected_by_ar is not None and n_after_construct is not None:
                rejection_rate = n_rejected_by_ar / max(n_after_construct, 1)
                lines.append(f"Epoch rejection rate: {rejection_rate:.3f} "
                           f"({_pct(n_rejected_by_ar, n_after_construct)})")
                
        summary_text = "\n".join(lines)
        report.add_text(summary_text, title=f"{label} Epochs Summary")
    except Exception:
        pass


def add_qa_metrics_summary(
    report: mne.Report,
    evoked_metrics: dict | None = None,
    state_metrics: dict | None = None,
    global_metrics: dict | None = None,
):
    """Add a comprehensive QA metrics summary distinguishing evoked vs state.
    
    Parameters
    ----------
    evoked_metrics : dict, optional
        Dictionary containing evoked epoch metrics:
        - n_after_construct: epochs after construction
        - n_rejected_by_ar: epochs rejected by AutoReject
        - rejection_pct: percentage of epochs rejected
    state_metrics : dict, optional
        Dictionary containing state epoch metrics (same structure as evoked)
    global_metrics : dict, optional
        Dictionary containing global QA metrics:
        - bad_channels_ratio: ratio of bad channels
        - ica_excluded_ratio: ratio of ICA components excluded
        - line_noise_improvement: dB improvement in line noise
    """
    try:
        lines = []
        lines.append("=== Comprehensive QA Metrics Summary ===")
        lines.append("")
        
        if global_metrics:
            lines.append("GLOBAL METRICS:")
            if 'bad_channels_ratio' in global_metrics:
                ratio = global_metrics['bad_channels_ratio']
                n_bad = global_metrics.get('n_bad_channels', 0)
                bad_list = global_metrics.get('bad_channels', [])
                lines.append(f"Bad channels: {n_bad} ({ratio:.1%})")
                if bad_list:
                    lines.append(f"Bad channel names: {', '.join(bad_list)}")
            
            if 'ica_excluded_ratio' in global_metrics:
                ratio = global_metrics['ica_excluded_ratio']
                n_excluded = global_metrics.get('n_ica_excluded', 0)
                excluded_list = global_metrics.get('ica_excluded_indices', [])
                lines.append(f"ICA excluded: {n_excluded} ({ratio:.1%})")
                if excluded_list:
                    lines.append(f"Excluded ICA indices: {excluded_list}")
            
            if 'line_noise_improvement' in global_metrics:
                improvement = global_metrics['line_noise_improvement']
                lines.append(f"Line noise improvement: {improvement:.2f} dB")
            lines.append("")
        
        if evoked_metrics:
            lines.append("EVOKED EPOCHS (used for QA):")
            lines.append(f"Constructed: {evoked_metrics.get('n_after_construct', 0)}")
            lines.append(f"Rejected by AutoReject: {evoked_metrics.get('n_rejected_by_ar', 0)}")
            lines.append(f"Rejection rate: {evoked_metrics.get('rejection_pct', 0):.1f}%")
            if 'flag_too_many_bad_epochs' in evoked_metrics and evoked_metrics['flag_too_many_bad_epochs'] is not None:
                flag = evoked_metrics['flag_too_many_bad_epochs']
                lines.append(f"Flag: too many bad evoked epochs = {bool(flag)}")
            lines.append("")
        
        if state_metrics:
            lines.append("STATE EPOCHS (not used for QA):")
            lines.append(f"Constructed: {state_metrics.get('n_after_construct', 0)}")
            lines.append(f"Rejected by AutoReject: {state_metrics.get('n_rejected_by_ar', 0)}")
            lines.append(f"Rejection rate: {state_metrics.get('rejection_pct', 0):.1f}%")
            if 'flag_too_many_bad_epochs' in state_metrics and state_metrics['flag_too_many_bad_epochs'] is not None:
                flag = state_metrics['flag_too_many_bad_epochs']
                lines.append(f"Flag: too many bad state epochs = {bool(flag)}")
        
        summary_text = "\n".join(lines)
        report.add_text(summary_text, title="QA Metrics Summary")
    except Exception:
        pass


def add_epochs_sections(
    report: mne.Report,
    epochs_before,  # mne.Epochs | None
    epochs_after,   # mne.Epochs | None
    label: str,
) -> None:
    """Add before/after epochs snapshots to the report.

    Parameters
    ----------
    epochs_before : mne.Epochs | None
        Epochs prior to cleaning (e.g., pre-AutoReject)
    epochs_after : mne.Epochs | None
        Epochs after cleaning
    label : str
        Section label prefix (e.g., 'Evoked', 'State')
    """
    try:
        if epochs_before is not None:
            report.add_epochs(epochs_before, title=f"{label} epochs (before)")
    except Exception:
        pass
    try:
        if epochs_after is not None:
            report.add_epochs(epochs_after, title=f"{label} epochs (after)")
    except Exception:
        pass


def add_evoked_comparison(report: mne.Report, epochs_clean) -> None:
    """Add go vs nogo evoked and a P300-like difference if available.

    This is robust to label variants: it prefers '/correct' but falls back
    to any label that starts with 'go' or 'nogo'.
    """
    try:
        # Find go/nogo conditions with preference for '/correct' suffix
        go_labels = [k for k in epochs_clean.event_id.keys() 
                    if isinstance(k, str) and k.lower().startswith('go')]
        nogo_labels = [k for k in epochs_clean.event_id.keys() 
                      if isinstance(k, str) and k.lower().startswith('nogo')]
        
        # Prefer '/correct' variants
        go_correct = [k for k in go_labels if '/correct' in k.lower()]
        nogo_correct = [k for k in nogo_labels if '/correct' in k.lower()]
        
        go_key = go_correct[0] if go_correct else (go_labels[0] if go_labels else None)
        nogo_key = nogo_correct[0] if nogo_correct else (nogo_labels[0] if nogo_labels else None)
        
        evokeds = []
        titles = []
        
        if go_key and len(epochs_clean[go_key]) > 0:
            evk_go = epochs_clean[go_key].average()
            evokeds.append(evk_go)
            titles.append(f"Go ({go_key})")
        else:
            evk_go = None
            
        if nogo_key and len(epochs_clean[nogo_key]) > 0:
            evk_nogo = epochs_clean[nogo_key].average()
            evokeds.append(evk_nogo)
            titles.append(f"NoGo ({nogo_key})")
        else:
            evk_nogo = None
        
        # Add individual evokeds
        if evokeds:
            report.add_evokeds(evokeds, titles=titles)
        
        # Add difference wave if both available
        if evk_go is not None and evk_nogo is not None:
            try:
                diff = mne.combine_evoked([evk_go, evk_nogo], weights=[1, -1])
                diff.comment = f"Difference: {go_key} - {nogo_key}"
                report.add_evokeds([diff], titles=["Go - NoGo Difference"])
            except Exception:
                pass
                
    except Exception:
        pass