import numpy as np
from nice.markers import (KolmogorovComplexity, TimeLockedContrast, PowerSpectralDensityEstimator, PowerSpectralDensitySummary,
                         PowerSpectralDensity, SymbolicMutualInformation, PermutationEntropy, TimeLockedTopography, ContingentNegativeVariation)

import os
import re
import warnings
import functools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm

import mne

# Add a caching decorator to avoid redundant computations
def cache_computation(func):
    """Decorator to cache computation results"""
    cache = {}
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Create a key based on function arguments
        key = (func.__name__, str(args), str(sorted(kwargs.items())))
        if key in cache:
            return cache[key]
        result = func(*args, **kwargs)
        cache[key] = result
        return result
    
    return wrapper


def all_markers(epochs, tmin, tmax, target='epochs', picks=None, roi='whole', per_epoch=True):
    """
    Computes all the markers for given epochs.
    epochs: the epochs from which to compute the markers
    tmin: min time for computing markers 
    tmax: max time to compute markers
    target: reduction target, epochs or topography
    picks: channels to pick, if None uses all channels
    roi: Region of interest specification:
         - 'whole': one value for all electrodes per marker
         - 'all': one value per electrode per marker
         - dict: dictionary with ROIs like {'frontal': ['Fz', 'Fpz']}
    per_epoch: whether to compute markers for each epoch separately (True) or average across epochs (False)
    """       
    from scipy.stats import trim_mean
    
    # Downsample to 500 Hz if sampling rate is higher
    current_sfreq = epochs.info['sfreq']
    if current_sfreq > 500:
        print(f"Downsampling from {current_sfreq} Hz to 500 Hz")
        epochs = epochs.resample(500)
    
    def trim_mean80(a, axis=0, **kwargs):
        return trim_mean(a, proportiontocut=.1, axis=axis)

    # Configure reduction based on ROI parameter and per_epoch flag
    if roi == 'whole':
        # For spectral markers (PSD) - For whole brain, the order is frequency first
        spectral_reduction = [
            {'axis': 'frequency', 'function': np.sum},
            {'axis': 'channels', 'function': np.mean}
        ]
        
        # If we want per-epoch data, don't reduce over epochs dimension
        if not per_epoch:
            # Add epochs reduction (averaging across epochs)
            spectral_reduction.append(
                {'axis': 'epochs', 'function': trim_mean80}
            )
        
        # For summary markers
        summary_reduction = [
            {'axis': 'channels', 'function': np.mean}
        ]
        if not per_epoch:
            summary_reduction.append(
                {'axis': 'epochs', 'function': trim_mean80}
            )
        
        # For wSMI - For whole brain, channels_y first
        wsmi_reduction = [
            {'axis': 'channels_y', 'function': np.median},
            {'axis': 'channels', 'function': np.mean}
        ]
        if not per_epoch:
            wsmi_reduction.append(
                {'axis': 'epochs', 'function': trim_mean80}
            )
        
        # For other markers (K, PE)
        other_reduction = [
            {'axis': 'channels', 'function': np.mean}
        ]
        if not per_epoch:
            other_reduction.append(
                {'axis': 'epochs', 'function': trim_mean80}
            )
    elif roi == 'all':
        if per_epoch:
            # Do not reduce over channels or epochs, preserve both dimensions
            spectral_reduction = [
                {'axis': 'channels', 'function': lambda x, axis=None, **kwargs: x},  # Preserve channels
                {'axis': 'frequency', 'function': np.sum},
                {'axis': 'epochs', 'function': lambda x, axis=None, **kwargs: x}     # Preserve epochs
            ]
            summary_reduction = [
                {'axis': 'channels', 'function': lambda x, axis=None, **kwargs: x},
                {'axis': 'epochs', 'function': lambda x, axis=None, **kwargs: x}
            ]
            wsmi_reduction = [
                {'axis': 'channels', 'function': lambda x, axis=None, **kwargs: x},
                {'axis': 'channels_y', 'function': np.median},
                {'axis': 'epochs', 'function': lambda x, axis=None, **kwargs: x}
            ]
            other_reduction = [
                {'axis': 'channels', 'function': lambda x, axis=None, **kwargs: x},
                {'axis': 'epochs', 'function': lambda x, axis=None, **kwargs: x}
            ]
        else:
            # Existing logic for per_epoch=False
            spectral_reduction = [
                {'axis': 'channels', 'function': lambda x, axis=None, **kwargs: x},
                {'axis': 'frequency', 'function': np.sum},
                {'axis': 'epochs', 'function': trim_mean80}
            ]
            summary_reduction = [
                {'axis': 'channels', 'function': lambda x, axis=None, **kwargs: x},
                {'axis': 'epochs', 'function': trim_mean80}
            ]
            wsmi_reduction = [
                {'axis': 'channels', 'function': lambda x, axis=None, **kwargs: x},
                {'axis': 'channels_y', 'function': np.median},
                {'axis': 'epochs', 'function': trim_mean80}
            ]
            other_reduction = [
                {'axis': 'channels', 'function': lambda x, axis=None, **kwargs: x},
                {'axis': 'epochs', 'function': trim_mean80}
            ]
    elif isinstance(roi, dict):
        # For spectral markers (PSD) - For ROIs, channels first then frequency
        spectral_reduction = [
            {'axis': 'channels', 'function': lambda x, axis=None, **kwargs: x},  # Preserve channels
            {'axis': 'frequency', 'function': np.sum}
        ]
        if not per_epoch:
            spectral_reduction.append({'axis': 'epochs', 'function': trim_mean80})
        
        # For summary markers
        summary_reduction = [
            {'axis': 'channels', 'function': lambda x, axis=None, **kwargs: x}  # Preserve channels
        ]
        if not per_epoch:
            summary_reduction.append({'axis': 'epochs', 'function': trim_mean80})
        
        # For wSMI - For ROIs, the order is channels first
        wsmi_reduction = [
            {'axis': 'channels', 'function': lambda x, axis=None, **kwargs: x},  # Preserve channels
            {'axis': 'channels_y', 'function': np.median},
            {'axis': 'epochs', 'function': trim_mean80}
        ]
        
        # For other markers (K, PE)
        other_reduction = [
            {'axis': 'channels', 'function': lambda x, axis=None, **kwargs: x},  # Preserve channels
            {'axis': 'epochs', 'function': trim_mean80}
        ]
        
        # Pick only the channels specified in ROI dict
        all_roi_channels = []
        for roi_chs in roi.values():
            all_roi_channels.extend(roi_chs)
        # Remove duplicates while preserving order
        all_roi_channels = list(dict.fromkeys(all_roi_channels))
        # Filter to only include channels that exist in the data
        picks = [ch for ch in all_roi_channels if ch in epochs.ch_names]
    else:
        raise ValueError("roi must be 'whole', 'all', or a dictionary of ROIs")

    # Pick channels if specified
    if picks is not None:
        # Filter picks to only include channels that exist in the data
        valid_picks = [ch for ch in picks if ch in epochs.ch_names]
        if not valid_picks:
            raise ValueError(f"None of the specified channels {picks} exist in the data")
        epochs = epochs.pick_channels(valid_picks)

    # =============================================================================
    # SPECTRAL MARKERS
    # =============================================================================
    # Create cached versions of compute-intensive functions
    psds_params = dict(n_fft=4096, n_overlap=100, n_jobs='auto', nperseg=128)
    base_psd = PowerSpectralDensityEstimator(
        psd_method='welch', tmin=tmin, tmax=tmax, fmin=1., fmax=45.,
        psd_params=psds_params, comment='default')
        
    # Cache the PSD computation since it's used repeatedly
    @cache_computation
    def compute_psd(epochs, base_psd):
        # Fit the base PSD estimator
        base_psd.fit(epochs)
        return base_psd
    
    # Compute the base PSD once and reuse
    base_psd = compute_psd(epochs, base_psd)

    # Process ROI results if using dictionary
    def process_roi_results(data, ch_names):
        if not isinstance(roi, dict):
            return data
            
        roi_results = {}
        for roi_name, roi_chs in roi.items():
            # Find indices of channels in this ROI that exist in the data
            ch_indices = []
            for ch in roi_chs:
                if ch in ch_names:
                    try:
                        ch_indices.append(ch_names.index(ch))
                    except ValueError:
                        continue
            
            if ch_indices:
                # Handle both 1D and 2D data arrays
                if isinstance(data, np.ndarray):
                    if data.ndim == 1:
                        # For 1D arrays, just take values at indices
                        if len(data) > max(ch_indices):
                            roi_results[roi_name] = np.mean(data[ch_indices])
                    elif data.ndim == 2:
                        # For 2D arrays, take the first dimension at indices
                        if len(data) > max(ch_indices):
                            roi_results[roi_name] = np.mean(data[ch_indices], axis=0)
                else:
                    # If not a numpy array, try to convert it
                    try:
                        data_array = np.asarray(data)
                        if data_array.ndim == 1 and len(data_array) > max(ch_indices):
                            roi_results[roi_name] = np.mean(data_array[ch_indices])
                        elif data_array.ndim == 2 and len(data_array) > max(ch_indices):
                            roi_results[roi_name] = np.mean(data_array[ch_indices], axis=0)
                    except:
                        # If conversion fails, just return the original data
                        roi_results[roi_name] = data
        
        return roi_results

    # Collect results
    results = {}
    
    # Optimize computation by creating and fitting PSD markers once for each frequency band
    # This avoids redundant computations for normalized and unnormalized versions
    
    # Compute all frequency bands at once
    try:
        print("Computing spectral markers...")
        # Alpha band (8-13 Hz)
        alpha_norm = PowerSpectralDensity(
            estimator=base_psd, fmin=8., fmax=13., 
            normalize=True, comment='alpha'
        )
        alpha_norm.fit(epochs)
        results['a_n'] = process_roi_results(
            alpha_norm._reduce_to(spectral_reduction, target=target, picks=None), 
            epochs.ch_names
        )
        
        alpha = PowerSpectralDensity(
            estimator=base_psd, fmin=8., fmax=13., 
            normalize=False, comment='alpha'
        )
        alpha.fit(epochs)
        results['a'] = process_roi_results(
            alpha._reduce_to(spectral_reduction, target=target, picks=None), 
            epochs.ch_names
        )
        
        # Delta band (1-4 Hz)
        delta_norm = PowerSpectralDensity(
            estimator=base_psd, fmin=1., fmax=4., 
            normalize=True, comment='delta'
        )
        delta_norm.fit(epochs)
        results['d_n'] = process_roi_results(
            delta_norm._reduce_to(spectral_reduction, target=target, picks=None), 
            epochs.ch_names
        )
        
        delta = PowerSpectralDensity(
            estimator=base_psd, fmin=1., fmax=4, 
            normalize=False, comment='delta'
        )
        delta.fit(epochs)
        results['d'] = process_roi_results(
            delta._reduce_to(spectral_reduction, target=target, picks=None), 
            epochs.ch_names
        )
        
        # Theta band (4-8 Hz)
        theta_norm = PowerSpectralDensity(
            estimator=base_psd, fmin=4., fmax=8., 
            normalize=True, comment='theta'
        )
        theta_norm.fit(epochs)
        results['t_n'] = process_roi_results(
            theta_norm._reduce_to(spectral_reduction, target=target, picks=None), 
            epochs.ch_names
        )
        
        theta = PowerSpectralDensity(
            estimator=base_psd, fmin=4., fmax=8, 
            normalize=False, comment='theta'
        )
        theta.fit(epochs)
        results['t'] = process_roi_results(
            theta._reduce_to(spectral_reduction, target=target, picks=None), 
            epochs.ch_names
        )
        
        # Gamma band (30-45 Hz)
        gamma_norm = PowerSpectralDensity(
            estimator=base_psd, fmin=30., fmax=45., 
            normalize=True, comment='gamma'
        )
        gamma_norm.fit(epochs)
        results['g_n'] = process_roi_results(
            gamma_norm._reduce_to(spectral_reduction, target=target, picks=None), 
            epochs.ch_names
        )
        
        gamma = PowerSpectralDensity(
            estimator=base_psd, fmin=30., fmax=45, 
            normalize=False, comment='gamma'
        )
        gamma.fit(epochs)
        results['g'] = process_roi_results(
            gamma._reduce_to(spectral_reduction, target=target, picks=None), 
            epochs.ch_names
        )
        
        # Beta band (13-30 Hz)
        beta_norm = PowerSpectralDensity(
            estimator=base_psd, fmin=13., fmax=30., 
            normalize=True, comment='beta'
        )
        beta_norm.fit(epochs)
        results['b_n'] = process_roi_results(
            beta_norm._reduce_to(spectral_reduction, target=target, picks=None), 
            epochs.ch_names
        )
        
        beta = PowerSpectralDensity(
            estimator=base_psd, fmin=13., fmax=30, 
            normalize=False, comment='beta'
        )
        beta.fit(epochs)
        results['b'] = process_roi_results(
            beta._reduce_to(spectral_reduction, target=target, picks=None), 
            epochs.ch_names
        )
        
        # Spectral Entropy
        se = PowerSpectralDensity(
            estimator=base_psd, fmin=1., fmax=45.,
            normalize=False, comment='summary_se'
        )
        se.fit(epochs)
        results['se'] = process_roi_results(
            se._reduce_to(spectral_reduction, target=target, picks=None), 
            epochs.ch_names
        )
        
        # Spectral Summary
        msf = PowerSpectralDensitySummary(
            estimator=base_psd, fmin=1., fmax=45.,
            percentile=.5, comment='summary_msf'
        )
        msf.fit(epochs)
        results['msf'] = process_roi_results(
            msf._reduce_to(summary_reduction, target=target, picks=None), 
            epochs.ch_names
        )
        
        sef90 = PowerSpectralDensitySummary(
            estimator=base_psd, fmin=1., fmax=45.,
            percentile=.9, comment='summary_sef90'
        )
        sef90.fit(epochs)
        results['sef90'] = process_roi_results(
            sef90._reduce_to(summary_reduction, target=target, picks=None), 
            epochs.ch_names
        )
        
        sef95 = PowerSpectralDensitySummary(
            estimator=base_psd, fmin=1., fmax=45.,
            percentile=.95, comment='summary_sef95'
        )
        sef95.fit(epochs)
        results['sef95'] = process_roi_results(
            sef95._reduce_to(summary_reduction, target=target, picks=None), 
            epochs.ch_names
        )
    except Exception as e:
        warnings.warn(f"Error computing spectral markers: {str(e)}")

    # =============================================================================
    # INFORMATION THEORY MARKERS
    # =============================================================================
    
    # Kolmogorov complexity
    try:
        print("Computing Kolmogorov complexity...")
        try:
            komplexity = KolmogorovComplexity(tmin=tmin, tmax=tmax, backend='openmp')
            komplexity.fit(epochs)
        except Exception as e:
            if "invalid ELF header" in str(e) or "not supported" in str(e):
                print("OpenMP backend failed, falling back to Python backend for Kolmogorov complexity")
                komplexity = KolmogorovComplexity(tmin=tmin, tmax=tmax, backend='python')
                komplexity.fit(epochs)
            else:
                raise e
        
        # Define reduction based on per_epoch parameter
        k_reduction = other_reduction.copy()
        if not per_epoch and 'epochs' in [r['axis'] for r in k_reduction]:
            # Keep the epochs reduction
            pass
        elif per_epoch:
            # Remove any epochs reduction
            k_reduction = [r for r in k_reduction if r['axis'] != 'epochs']
            
        results['k'] = process_roi_results(
            komplexity._reduce_to(k_reduction, target=target, picks=None), 
            epochs.ch_names
        )
    except Exception as e:
        warnings.warn(f"Error computing Kolmogorov complexity: {str(e)}")
        results['k'] = np.nan
    
    # Permutation entropy with different taus
    try:
        print("Computing permutation entropy...")
        for tau in [1, 2, 4, 8]:
            try:
                try:
                    p_e = PermutationEntropy(tmin=tmin, tmax=tmax, kernel=3, tau=tau, backend='openmp')
                    p_e.fit(epochs)
                except Exception as e:
                    if "invalid ELF header" in str(e) or "not supported" in str(e):
                        print(f"OpenMP backend failed, falling back to Python backend for permutation entropy (tau={tau})")
                        p_e = PermutationEntropy(tmin=tmin, tmax=tmax, kernel=3, tau=tau, backend='python')
                        p_e.fit(epochs)
                    else:
                        raise e
                
                results[f'p_e_{tau}'] = process_roi_results(
                    p_e._reduce_to(other_reduction, target=target, picks=None), 
                    epochs.ch_names
                )
            except Exception as e:
                warnings.warn(f"Error computing permutation entropy (tau={tau}): {str(e)}")
                results[f'p_e_{tau}'] = np.nan
    except Exception as e:
        warnings.warn(f"Error during permutation entropy computation: {str(e)}")

    # =============================================================================
    # wSMI MARKERS
    # =============================================================================
    
    # wSMI with different taus
    try:
        print("Computing wSMI markers...")
        for tau in [1, 2, 4, 8]:
            try:
                try:
                    wSMI = SymbolicMutualInformation(
                        tmin=tmin, tmax=tmax, kernel=3, tau=tau,
                        backend="openmp", method_params=None,
                        method='weighted', comment='default'
                    )
                    wSMI.fit(epochs)
                except Exception as e:
                    if "invalid ELF header" in str(e) or "not supported" in str(e):
                        print(f"OpenMP backend failed, falling back to Python backend for wSMI (tau={tau})")
                        wSMI = SymbolicMutualInformation(
                            tmin=tmin, tmax=tmax, kernel=3, tau=tau,
                            backend="python", method_params=None,
                            method='weighted', comment='default'
                        )
                        wSMI.fit(epochs)
                    else:
                        raise e
                
                results[f'wSMI_{tau}'] = process_roi_results(
                    wSMI._reduce_to(wsmi_reduction, target=target, picks=None),
                    epochs.ch_names
                )
            except Exception as e:
                warnings.warn(f"Error computing wSMI (tau={tau}): {str(e)}")
                results[f'wSMI_{tau}'] = np.nan
    except Exception as e:
        warnings.warn(f"Error during wSMI computation: {str(e)}")

    # =============================================================================
    # EVOKED MARKERS
    # =============================================================================
    
    # Define channel ROIs for ERP components
    # These will be used for whole-brain analysis
    cnv_chs = ['AF3', 'AFz', 'AF4', 'F1', 'Fz', 'F2', 'FC1', 'FCz', 'FC2']
    p1_chs = ['AF3', 'AFz', 'AF4', 'F1', 'Fz', 'F2', 'FC1', 'FCz', 'FC2']  # Same as CNV
    n1_chs = ['FC1', 'FCz', 'FC2', 'C1', 'Cz', 'C2', 'CP1', 'CPz', 'CP2']  # Central
    p2_chs = ['FC1', 'FCz', 'FC2', 'C1', 'Cz', 'C2', 'CP1', 'CPz', 'CP2']  # Same as N1
    p3a_chs = ['AF3', 'AFz', 'AF4', 'F1', 'Fz', 'F2', 'FC1', 'FCz', 'FC2']  # Same as CNV
    p3b_chs = ['FC1', 'FCz', 'FC2', 'C1', 'Cz', 'C2', 'CP1', 'CPz', 'CP2']  # Same as N1

    # Define top_args for TimeLockedTopography components
    top_args = {}  # Empty dict for default arguments

    # For evoked markers, we need to handle the different ROI modes
    try:
        print("Computing ERP components...")
        if roi == 'whole':
            try:
                # Contingent Negative Variation (CNV)
                compute_erp_component('cnv', epochs, cnv_chs, -0.004, 0.596, per_epoch, target, results, trim_mean80)
                
                # P1
                compute_erp_component('p1', epochs, p1_chs, 0.100, 0.150, per_epoch, target, results, trim_mean80)
                
                # N1
                compute_erp_component('n1', epochs, n1_chs, 0.150, 0.200, per_epoch, target, results, trim_mean80)
                
                # P2
                compute_erp_component('p2', epochs, p2_chs, 0.200, 0.275, per_epoch, target, results, trim_mean80)
                
                # P3a
                compute_erp_component('p3a', epochs, p3a_chs, 0.275, 0.375, per_epoch, target, results, trim_mean80)
                
                # P3b
                compute_erp_component('p3b', epochs, p3b_chs, 0.375, 0.600, per_epoch, target, results, trim_mean80)
            except Exception as e:
                warnings.warn(f"Error computing ERP components: {str(e)}")
                
        elif roi == 'all':
            # Define top_args for TimeLockedTopography components
            top_args = {}  # Empty dict for default arguments
            
            try:
                # Contingent Negative Variation (CNV)
                cnv = ContingentNegativeVariation(tmin=-0.004, tmax=0.596)
                cnv.fit(epochs)
                
                # For all channels, use a reduction that preserves channels
                cnv_reduction = [
                    {'axis': 'channels', 'function': lambda x, axis=None, **kwargs: x}  # Preserve channels
                ]
                if not per_epoch:
                    cnv_reduction.append({'axis': 'epochs', 'function': trim_mean80})
                
                # Important: Use None for picks to process all channels
                results['cnv'] = cnv._reduce_to(cnv_reduction, target=target, picks=None)
            except Exception as e:
                warnings.warn(f"Error computing CNV: {str(e)}")
                
            # Implement all other ERP components
            try:
                # P1
                p1 = TimeLockedTopography(tmin=0.100, tmax=0.150, **top_args)
                p1.fit(epochs)
                
                p1_reduction = [
                    {'axis': 'channels', 'function': lambda x, axis=None, **kwargs: x}  # Preserve channels
                ]
                if not per_epoch:
                    p1_reduction.append({'axis': 'epochs', 'function': trim_mean80})
                p1_reduction.append({'axis': 'times', 'function': np.mean})
                
                results['p1'] = p1._reduce_to(p1_reduction, target=target, picks=None)
            except Exception as e:
                warnings.warn(f"Error computing P1: {str(e)}")
                
            try:
                # N1
                n1 = TimeLockedTopography(tmin=0.150, tmax=0.200, **top_args)
                n1.fit(epochs)
                
                n1_reduction = [
                    {'axis': 'channels', 'function': lambda x, axis=None, **kwargs: x}  # Preserve channels
                ]
                if not per_epoch:
                    n1_reduction.append({'axis': 'epochs', 'function': trim_mean80})
                n1_reduction.append({'axis': 'times', 'function': np.mean})
                
                results['n1'] = n1._reduce_to(n1_reduction, target=target, picks=None)
            except Exception as e:
                warnings.warn(f"Error computing N1: {str(e)}")
                
            try:
                # P2
                p2 = TimeLockedTopography(tmin=0.200, tmax=0.275, **top_args)
                p2.fit(epochs)
                
                p2_reduction = [
                    {'axis': 'channels', 'function': lambda x, axis=None, **kwargs: x}  # Preserve channels
                ]
                if not per_epoch:
                    p2_reduction.append({'axis': 'epochs', 'function': trim_mean80})
                p2_reduction.append({'axis': 'times', 'function': np.mean})
                
                results['p2'] = p2._reduce_to(p2_reduction, target=target, picks=None)
            except Exception as e:
                warnings.warn(f"Error computing P2: {str(e)}")
                
            try:
                # P3a
                p3a = TimeLockedTopography(tmin=0.275, tmax=0.375, **top_args)
                p3a.fit(epochs)
                
                p3a_reduction = [
                    {'axis': 'channels', 'function': lambda x, axis=None, **kwargs: x}  # Preserve channels
                ]
                if not per_epoch:
                    p3a_reduction.append({'axis': 'epochs', 'function': trim_mean80})
                p3a_reduction.append({'axis': 'times', 'function': np.mean})
                
                results['p3a'] = p3a._reduce_to(p3a_reduction, target=target, picks=None)
            except Exception as e:
                warnings.warn(f"Error computing P3a: {str(e)}")
                
            try:
                # P3b
                p3b = TimeLockedTopography(tmin=0.375, tmax=0.600, **top_args)
                p3b.fit(epochs)
                
                p3b_reduction = [
                    {'axis': 'channels', 'function': lambda x, axis=None, **kwargs: x}  # Preserve channels
                ]
                if not per_epoch:
                    p3b_reduction.append({'axis': 'epochs', 'function': trim_mean80})
                p3b_reduction.append({'axis': 'times', 'function': np.mean})
                
                results['p3b'] = p3b._reduce_to(p3b_reduction, target=target, picks=None)
            except Exception as e:
                warnings.warn(f"Error computing P3b: {str(e)}")

        elif isinstance(roi, dict):
            # Define top_args for TimeLockedTopography components
            top_args = {}  # Empty dict for default arguments
            
            # Process each ROI for CNV
            try:
                cnv = ContingentNegativeVariation(tmin=-0.004, tmax=0.596)
                cnv.fit(epochs)
                
                cnv_roi_results = {}
                for roi_name, channels in roi.items():
                    # Get indices of channels that exist in the data
                    available_channels = [ch for ch in channels if ch in epochs.ch_names]
                    if available_channels:
                        # Important: Use np.array and epochs.ch_names.index
                        roi_indices = np.array([epochs.ch_names.index(ch) for ch in available_channels])
                        
                        # Define reduction based on per_epoch parameter
                        cnv_reduction = []
                        if not per_epoch:
                            cnv_reduction.append({'axis': 'epochs', 'function': trim_mean80})
                        cnv_reduction.append({'axis': 'channels', 'function': np.mean})
                        
                        cnv_roi_results[roi_name] = cnv._reduce_to(cnv_reduction, target=target, picks={
                            'epochs': None,
                            'channels': roi_indices
                        })
                
                # Store the results dictionary
                if cnv_roi_results:
                    results['cnv'] = cnv_roi_results
            except Exception as e:
                warnings.warn(f"Error computing CNV for ROIs: {str(e)}")
                
            # Similar updates for other ERP components...
    except Exception as e:
        warnings.warn(f"Error during ERP components computation: {str(e)}")

    print("Marker computation completed")
    return results


def compute_markers_per_file(file, folder, results_folder):
    target = 'epochs'
    pattern = r"S(\d+)_eegmw_(\d+).fif"

    match = re.search(pattern, file)
    subject_number = match.group(1)
    round_number = match.group(2)

    #############################
    epochs = mne.read_epochs(os.path.join(folder, file), verbose=False)

    epochs.info['description'] = 'smarting/24' #necessary for wSMI 
    epochs =  epochs.pick_types(eeg = True) #EOGs break everything\
    
    
    df_markers = pd.DataFrame.from_dict(all_markers(epochs, 0, 2, target))
    df_markers['participant'] = subject_number
    df_markers['round'] = round_number
        

    df_markers.to_csv(os.path.join(results_folder, f'{subject_number}_{round_number}_all_markers.csv'))
    

# Define the context manager
import contextlib
import joblib
from joblib import Parallel, delayed,  parallel_backend
@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

@cache_computation
def compute_erp_component(component_name, epochs, ch_list, tmin, tmax, per_epoch, target, results, trim_mean80):
    """Helper function to compute ERP components consistently with caching"""
    try:
        # Define top_args for TimeLockedTopography components
        top_args = {}  # Empty dict for default arguments
        
        # Create the component object
        if component_name == 'cnv':
            component = ContingentNegativeVariation(tmin=tmin, tmax=tmax)
        else:
            component = TimeLockedTopography(tmin=tmin, tmax=tmax, **top_args)
        
        # Check which channels are actually available
        available_chs = [ch for ch in ch_list if ch in epochs.ch_names]
        
        if available_chs:  # Only proceed if we have at least one channel
            # Fit the component
            component.fit(epochs)
            
            # Use available channels only
            roi_indices = np.array([epochs.ch_names.index(ch) for ch in available_chs])
            
            # Define reduction based on per_epoch parameter
            reduction = []
            if not per_epoch:
                reduction.append({'axis': 'epochs', 'function': trim_mean80})
            reduction.append({'axis': 'channels', 'function': np.mean})
            
            # Add times reduction for TimeLockedTopography components
            if component_name != 'cnv':
                reduction.append({'axis': 'times', 'function': np.mean})
            
            # Set up picks dict
            picks = {
                'epochs': None,
                'channels': roi_indices
            }
            
            # Add times pick for TimeLockedTopography components
            if component_name != 'cnv':
                picks['times'] = None
                
            # Check if roi_indices is valid (not out of range)
            if np.max(roi_indices) >= len(epochs.ch_names):
                warnings.warn(f"Channel indices out of bounds for {component_name.upper()}: max index {np.max(roi_indices)} >= {len(epochs.ch_names)}")
                return None
                
            # Compute and store result
            return component._reduce_to(reduction, target=target, picks=picks)
        else:
            warnings.warn(f"No channels found in data for {component_name.upper()} (requested: {ch_list}, available: {epochs.ch_names})")
            return None
    except Exception as e:
        warnings.warn(f"Error computing {component_name.upper()}: {str(e)}")
        return None
