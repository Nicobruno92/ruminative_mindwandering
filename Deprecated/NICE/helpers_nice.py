import numpy as np
from nice.markers import (
    KolmogorovComplexity, PowerSpectralDensityEstimator, PowerSpectralDensitySummary,
    PowerSpectralDensity, SymbolicMutualInformation, PermutationEntropy
)

def compute_all_nice_markers(epochs, tmin, tmax, reduction_mode='all', roi_dict=None):
    """
    Compute all NICE markers per epoch, with reduction over channels/ROIs but NOT over epochs.
    reduction_mode: 'whole', 'all', or ROI dict
    Returns: dict of marker_name -> np.ndarray (shape: n_epochs, n_channels x n_epochs, or n_rois x n_epochs)
    """
    # Reduction configs: remove reduction over epochs!
    if reduction_mode == 'whole':
        spectral_reduction = [
            {'axis': 'frequency', 'function': np.sum},
            {'axis': 'channels', 'function': np.mean},
        ]
        summary_reduction = [
            {'axis': 'channels', 'function': np.mean},
        ]
        wsmi_reduction = [
            {'axis': 'channels_y', 'function': np.median},
            {'axis': 'channels', 'function': np.mean},
        ]
        other_reduction = [
            {'axis': 'channels', 'function': np.mean},
        ]
    elif reduction_mode == 'all':
        spectral_reduction = [
            {'axis': 'channels', 'function': lambda x, axis=None, **kwargs: x},
            {'axis': 'frequency', 'function': np.sum},
        ]
        summary_reduction = [
            {'axis': 'channels', 'function': lambda x, axis=None, **kwargs: x},
        ]
        wsmi_reduction = [
            {'axis': 'channels', 'function': lambda x, axis=None, **kwargs: x},
            {'axis': 'channels_y', 'function': np.median},
        ]
        other_reduction = [
            {'axis': 'channels', 'function': lambda x, axis=None, **kwargs: x},
        ]
    elif isinstance(reduction_mode, dict):
        # ROI dict
        spectral_reduction = [
            {'axis': 'channels', 'function': lambda x, axis=None, **kwargs: x},
            {'axis': 'frequency', 'function': np.sum},
        ]
        summary_reduction = [
            {'axis': 'channels', 'function': lambda x, axis=None, **kwargs: x},
        ]
        wsmi_reduction = [
            {'axis': 'channels', 'function': lambda x, axis=None, **kwargs: x},
            {'axis': 'channels_y', 'function': np.median},
        ]
        other_reduction = [
            {'axis': 'channels', 'function': lambda x, axis=None, **kwargs: x},
        ]
        # Pick only the channels specified in ROI dict
        all_roi_channels = []
        for roi_chs in reduction_mode.values():
            all_roi_channels.extend(roi_chs)
        all_roi_channels = list(dict.fromkeys(all_roi_channels))
        picks = [ch for ch in all_roi_channels if ch in epochs.ch_names]
        if picks:
            epochs = epochs.pick_channels(picks)
    else:
        raise ValueError("reduction_mode must be 'whole', 'all', or a dictionary of ROIs")

    def process_roi_results(data, ch_names):
        if not isinstance(reduction_mode, dict):
            return data
        roi_results = {}
        for roi_name, roi_chs in reduction_mode.items():
            ch_indices = [ch_names.index(ch) for ch in roi_chs if ch in ch_names]
            if ch_indices:
                arr = np.asarray(data)
                if arr.ndim == 2:
                    roi_results[roi_name] = np.mean(arr[ch_indices, :], axis=0)
                elif arr.ndim == 1:
                    roi_results[roi_name] = np.mean(arr[ch_indices])
        return roi_results

    results = {}
    psds_params = dict(n_fft=4096, n_overlap=100, n_jobs='auto', nperseg=128)
    base_psd = PowerSpectralDensityEstimator(
        psd_method='welch', tmin=tmin, tmax=tmax, fmin=1., fmax=45.,
        psd_params=psds_params, comment='default')
    # Alpha
    try:
        alpha = PowerSpectralDensity(
            estimator=base_psd, fmin=8., fmax=13., normalize=True, comment='alpha')
        alpha.fit(epochs)
        arr = alpha._reduce_to(spectral_reduction, target='epochs', picks=None)
        arr = np.asarray(arr)
        if arr.ndim == 1:
            arr = arr[None, :]
        results['a_n'] = process_roi_results(arr, epochs.ch_names)
    except Exception as e:
        results['a_n'] = np.nan
    try:
        alpha = PowerSpectralDensity(
            estimator=base_psd, fmin=8., fmax=13., normalize=False, comment='alpha')
        alpha.fit(epochs)
        arr = alpha._reduce_to(spectral_reduction, target='epochs', picks=None)
        arr = np.asarray(arr)
        if arr.ndim == 1:
            arr = arr[None, :]
        results['a'] = process_roi_results(arr, epochs.ch_names)
    except Exception as e:
        results['a'] = np.nan
    # Delta
    try:
        delta = PowerSpectralDensity(
            estimator=base_psd, fmin=1., fmax=4., normalize=True, comment='delta')
        delta.fit(epochs)
        arr = delta._reduce_to(spectral_reduction, target='epochs', picks=None)
        arr = np.asarray(arr)
        if arr.ndim == 1:
            arr = arr[None, :]
        results['d_n'] = process_roi_results(arr, epochs.ch_names)
    except Exception as e:
        results['d_n'] = np.nan
    try:
        delta = PowerSpectralDensity(
            estimator=base_psd, fmin=1., fmax=4, normalize=False, comment='delta')
        delta.fit(epochs)
        arr = delta._reduce_to(spectral_reduction, target='epochs', picks=None)
        arr = np.asarray(arr)
        if arr.ndim == 1:
            arr = arr[None, :]
        results['d'] = process_roi_results(arr, epochs.ch_names)
    except Exception as e:
        results['d'] = np.nan
    # Theta
    try:
        theta = PowerSpectralDensity(
            estimator=base_psd, fmin=4., fmax=8., normalize=True, comment='theta')
        theta.fit(epochs)
        arr = theta._reduce_to(spectral_reduction, target='epochs', picks=None)
        arr = np.asarray(arr)
        if arr.ndim == 1:
            arr = arr[None, :]
        results['t_n'] = process_roi_results(arr, epochs.ch_names)
    except Exception as e:
        results['t_n'] = np.nan
    try:
        theta = PowerSpectralDensity(
            estimator=base_psd, fmin=4., fmax=8, normalize=False, comment='theta')
        theta.fit(epochs)
        arr = theta._reduce_to(spectral_reduction, target='epochs', picks=None)
        arr = np.asarray(arr)
        if arr.ndim == 1:
            arr = arr[None, :]
        results['t'] = process_roi_results(arr, epochs.ch_names)
    except Exception as e:
        results['t'] = np.nan
    # Gamma
    try:
        gamma = PowerSpectralDensity(
            estimator=base_psd, fmin=30., fmax=45., normalize=True, comment='gamma')
        gamma.fit(epochs)
        arr = gamma._reduce_to(spectral_reduction, target='epochs', picks=None)
        arr = np.asarray(arr)
        if arr.ndim == 1:
            arr = arr[None, :]
        results['g_n'] = process_roi_results(arr, epochs.ch_names)
    except Exception as e:
        results['g_n'] = np.nan
    try:
        gamma = PowerSpectralDensity(
            estimator=base_psd, fmin=30., fmax=45, normalize=False, comment='gamma')
        gamma.fit(epochs)
        arr = gamma._reduce_to(spectral_reduction, target='epochs', picks=None)
        arr = np.asarray(arr)
        if arr.ndim == 1:
            arr = arr[None, :]
        results['g'] = process_roi_results(arr, epochs.ch_names)
    except Exception as e:
        results['g'] = np.nan
    # Beta
    try:
        beta = PowerSpectralDensity(
            estimator=base_psd, fmin=13., fmax=30., normalize=True, comment='beta')
        beta.fit(epochs)
        arr = beta._reduce_to(spectral_reduction, target='epochs', picks=None)
        arr = np.asarray(arr)
        if arr.ndim == 1:
            arr = arr[None, :]
        results['b_n'] = process_roi_results(arr, epochs.ch_names)
    except Exception as e:
        results['b_n'] = np.nan
    try:
        beta = PowerSpectralDensity(
            estimator=base_psd, fmin=13., fmax=30, normalize=False, comment='beta')
        beta.fit(epochs)
        arr = beta._reduce_to(spectral_reduction, target='epochs', picks=None)
        arr = np.asarray(arr)
        if arr.ndim == 1:
            arr = arr[None, :]
        results['b'] = process_roi_results(arr, epochs.ch_names)
    except Exception as e:
        results['b'] = np.nan
    # Spectral Entropy
    try:
        se = PowerSpectralDensity(
            estimator=base_psd, fmin=1., fmax=45., normalize=False, comment='summary_se')
        se.fit(epochs)
        arr = se._reduce_to(spectral_reduction, target='epochs', picks=None)
        arr = np.asarray(arr)
        if arr.ndim == 1:
            arr = arr[None, :]
        results['se'] = process_roi_results(arr, epochs.ch_names)
    except Exception as e:
        results['se'] = np.nan
    # Spectral Summary
    try:
        msf = PowerSpectralDensitySummary(
            estimator=base_psd, fmin=1., fmax=45., percentile=.5, comment='summary_msf')
        msf.fit(epochs)
        arr = msf._reduce_to(summary_reduction, target='epochs', picks=None)
        arr = np.asarray(arr)
        if arr.ndim == 1:
            arr = arr[None, :]
        results['msf'] = process_roi_results(arr, epochs.ch_names)
    except Exception as e:
        results['msf'] = np.nan
    try:
        sef90 = PowerSpectralDensitySummary(
            estimator=base_psd, fmin=1., fmax=45., percentile=.9, comment='summary_sef90')
        sef90.fit(epochs)
        arr = sef90._reduce_to(summary_reduction, target='epochs', picks=None)
        arr = np.asarray(arr)
        if arr.ndim == 1:
            arr = arr[None, :]
        results['sef90'] = process_roi_results(arr, epochs.ch_names)
    except Exception as e:
        results['sef90'] = np.nan
    try:
        sef95 = PowerSpectralDensitySummary(
            estimator=base_psd, fmin=1., fmax=45., percentile=.95, comment='summary_sef95')
        sef95.fit(epochs)
        arr = sef95._reduce_to(summary_reduction, target='epochs', picks=None)
        arr = np.asarray(arr)
        if arr.ndim == 1:
            arr = arr[None, :]
        results['sef95'] = process_roi_results(arr, epochs.ch_names)
    except Exception as e:
        results['sef95'] = np.nan
    # Kolmogorov complexity
    try:
        komplexity = KolmogorovComplexity(tmin=tmin, tmax=tmax, backend='openmp')
        komplexity.fit(epochs)
        arr = komplexity._reduce_to(other_reduction, target='epochs', picks=None)
        arr = np.asarray(arr)
        if arr.ndim == 1:
            arr = arr[None, :]
        results['k'] = process_roi_results(arr, epochs.ch_names)
    except Exception as e:
        results['k'] = np.nan
    # Permutation entropy
    for tau in [1, 2, 4, 8]:
        try:
            p_e = PermutationEntropy(tmin=tmin, tmax=tmax, kernel=3, tau=tau)
            p_e.fit(epochs)
            arr = p_e._reduce_to(other_reduction, target='epochs', picks=None)
            arr = np.asarray(arr)
            if arr.ndim == 1:
                arr = arr[None, :]
            results[f'p_e_{tau}'] = process_roi_results(arr, epochs.ch_names)
        except Exception as e:
            results[f'p_e_{tau}'] = np.nan
    # wSMI
    for tau in [1, 2, 4, 8]:
        try:
            wSMI = SymbolicMutualInformation(
                tmin=tmin, tmax=tmax, kernel=3, tau=tau,
                backend="python", method_params=None,
                method='weighted', comment='default'
            )
            wSMI.fit(epochs)
            arr = wSMI._reduce_to(wsmi_reduction, target='epochs', picks=None)
            arr = np.asarray(arr)
            if arr.ndim == 1:
                arr = arr[None, :]
            results[f'wSMI_{tau}'] = process_roi_results(arr, epochs.ch_names)
        except Exception as e:
            results[f'wSMI_{tau}'] = np.nan
    return results 