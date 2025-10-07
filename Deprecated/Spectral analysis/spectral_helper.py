import os
import pickle
import numpy as np 
import pandas as pd

import plotly.graph_objects as go

import mne

from joblib import Parallel, delayed
import re

import sys
# sys.path.insert(0, './')
sys.path.insert(0, '../')
from utils.analysis_helpers import compute_erps, fit_lmm_for_time_bins, plot_erp
from utils.bids_compliance import read_epochs, save_psd_epochs, read_psd_epochs
from utils.analysis_helpers import filter_epochs_by_distance_to_probe, classify_onoff_epochs

class PSDProcessor:
    def __init__(self, root, sart_tasks = '1', metrics = 'highlow', data="eeg", distance=5, n_jobs=4):
        self.root = root
        self.derivatives_folder = os.path.join(root, "derivatives_nico")
        self.tasks = sart_tasks
        self.metrics = metrics
        self.data = data
        self.distance = distance
        self.n_jobs = n_jobs

    def classify_all_metrics(self, subject_epochs):
        """
        Classify epochs with multiple metrics.
        """
        classified_epochs_dict = {}
        for metric in self.metrics:
            classified_epochs_dict[metric] = classify_onoff_epochs(subject_epochs.copy(), split=metric)
        return classified_epochs_dict

    def process_subject_for_metrics(self, subject):
        """
        Process and classify epochs for a single subject.
        """
        epochs_tasks = []
        for task in self.tasks:
            try:
                epochs, events = read_epochs(self.derivatives_folder, subject, task, self.data, desc="autoPreproc")
                epochs_tasks.append(epochs.copy())
            except Exception as e:
                print(f"Skipping {subject} {task}: {e}")

        if not epochs_tasks:
            print(f"No data for subject {subject}")
            return None

        try:
            epochs_concat = mne.concatenate_epochs(epochs_tasks)
            filtered_epochs = filter_epochs_by_distance_to_probe(epochs_concat, self.distance)
        except Exception as e:
            print(f"Failed concatenating or filtering epochs for subject {subject}: {e}")
            return None
        
        classified_epochs_dict = self.classify_all_metrics(filtered_epochs)

        return classified_epochs_dict
    
    def generate_save_epochs_psd_allmetrics(self, subject, method = 'multitaper', tmin=None, tmax=None,):
        """
        Generate and save PSDs for different metrics.
        """
        
        try:
            classified_epochs_dict = self.process_subject_for_metrics(subject)
            for metric, epochs_classified in classified_epochs_dict.items():
                epoch_psds = epochs_classified.compute_psd(method = method, tmin=tmin, tmax=tmax)

                save_psd_epochs(epoch_psds, self.derivatives_folder, subject, self.data, desc=metric)
                print(f"PSDs saved for subject {subject} using metric {metric}")
        except Exception as e:
            print(f"Failed processing subject {subject}: {e}")
            
    def process_epochs_psd_subjects_parallel(self, subjects,  method = 'multitaper', fmin = 0.5, fmax = 40, tmin=None, tmax=None):
        """
        Parallelized processing for multiple subjects.
        """
        
        Parallel(n_jobs=self.n_jobs)(
            delayed(self.generate_save_epochs_psd_allmetrics)(subject,method = method, tmin=tmin, tmax=tmax) for subject in subjects
        )
    
    def generate_and_average_psd_for_probes(self, epochs_psd, stimulus_condition=['go', 'nogo'], response_condition=['correct', 'incorrect'], mind_condition=['ontask', 'offtask']):
        """
        Generate and save evokeds for different metrics without reloading epochs.
        """
        average_psds_dict = {'ontask': [], 'offtask': []}
        average_unique_events = set([re.findall(r'average\d+', value)[0] for value in epochs_psd.event_id.keys()])

        for stim in stimulus_condition:
            for resp in response_condition:
                for mind in mind_condition:
                    for probe in average_unique_events:
                        event_str = f"{stim}/{resp}/{mind}/{probe}"
                        try:
                            selected_epochs = epochs_psd[event_str]
                            evoked = selected_epochs.average()
                            average_psds_dict[mind].append(evoked)
                        except Exception as e:
                            continue

        return average_psds_dict

    def average_psds_by_band(self, psds_obj, freqs=None, bands=None, normalize=False, dB=True, agg_fun=None):
        """
        Compute the average PSD for each band and each channel.

        Parameters
        ----------
        psds : array of float, shape (n_channels, n_freqs)
            Power spectral densities.
        freqs : array of float, shape (n_freqs,)
            Frequencies corresponding to the PSD values.
        bands : dict | None
            Frequency bands to average over. Keys are band names, values are tuples of (fmin, fmax).
            If None, uses the default bands:
                - Delta: 0-4 Hz
                - Theta: 4-8 Hz
                - Alpha: 8-12 Hz
                - Beta: 12-30 Hz
                - Gamma: 30-45 Hz
        normalize : bool
            Whether to normalize the PSDs before averaging.
        dB : bool
            Whether to convert the averaged PSDs to decibels (dB).
        agg_fun : callable | None
            Aggregation function for PSD values within each band. Defaults to `np.mean`.

        Returns
        -------
        band_averages : dict
            Dictionary with band names as keys and arrays of averaged PSDs (shape: n_channels) as values.
        """
        psds = psds_obj.get_data()
        if freqs is None:
            freqs = psds_obj.freqs

        if bands is None:
            bands = {
                "Delta (0-4 Hz)": (0, 4),
                "Theta (4-8 Hz)": (4, 8),
                "Alpha (8-12 Hz)": (8, 12),
                "Beta (12-30 Hz)": (12, 30),
                "Gamma (30-45 Hz)": (30, 45),
            }

        if normalize:
            psds /= psds.sum(axis=-1, keepdims=True)

        if agg_fun is None:
            agg_fun = np.sum if normalize else np.mean

        band_data = {band: agg_fun(psds[:, (fmin < freqs) & (freqs < fmax)], axis=1) for band, (fmin, fmax) in bands.items()}
        return band_data

    def create_all_subjects_bands_df(self, subjects, conditions, bands, desc='highlow', normalize = False, save=False):
        """
        Create a DataFrame with all the subjects and all the bands for each task and condition.
        
        Parameters:
        derivatives_folder : str
            Path to the folder containing the PSD data.
        subjects : list of str
            List of subject identifiers.
        conditions : list of str
            List of experimental conditions (e.g., 'ontask', 'offtask').
        data : str
            Data type (e.g., 'eeg').
        bands : dict
            Frequency bands to average over. Keys are band names, and values are tuples of (fmin, fmax).
        desc : str, optional
            Description tag to locate specific PSD files. Default is 'highlow'.

        Returns:
        band_dfs : dict
            Dictionary where keys are band names and values are DataFrames containing the data for each band.
        """
        band_dfs = {band: [] for band in bands.keys()}
        for subject in subjects:
            try:
                psd_epochs = read_psd_epochs(self.derivatives_folder, subject, self.data, desc=desc)
                average_psd = self.generate_and_average_psd_for_probes(psd_epochs)
                for condition in conditions:
                    for probe_psd in average_psd[condition]:
                        band_averages = self.average_psds_by_band(probe_psd, bands=bands, normalize=normalize)
                        for band_name, band_data in band_averages.items():
                            band_df_row = {
                                'subject': subject,
                                'condition': condition
                            }
                            band_df_row.update({f'{i}': avg for i, avg in enumerate(band_data)})
                            band_dfs[band_name].append(band_df_row)
            except Exception as e:
                print(f"Failed for subject {subject}: {e}")
                continue

        for band in band_dfs:
            band_dfs[band] = pd.DataFrame(band_dfs[band])
            if save:
                band_dfs[band].to_csv(f'../results/PSD/{band}_band.csv', index=False)

        return band_dfs


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import mne
from mne.viz.utils import (
    _check_sphere,
    plt_show,
    _format_units_psd,
    _validate_if_list_of_axes,
    _plot_topomap_multi_cbar
)

def plot_psds_topomap_from_bands(
    band_dict, 
    info, 
    *,
    vlim=(None, None), 
    cmap=None,
    ch_type="eeg",
    sensors=True,
    names=None,
    mask=None,
    mask_params=None,
    contours=6,
    outlines="head",
    sphere=None,
    image_interp="linear",
    extrapolate="local",
    border="mean",
    res=64,
    size=1,
    cnorm=None,
    colorbar=True,
    cbar_fmt="%0.1f",
    unit=None,
    axes=None,
    show=True,
):
    """
    Plot spatial maps of PSDs for precomputed bands.

    Parameters
    ----------
    band_dict : dict
        Dictionary with band names as keys and arrays of averaged PSDs (shape: n_channels) as values.
    info : instance of mne.Info
        Measurement info object containing channel information.
    vlim : tuple | None
        Min and max values for color limits. If "joint", the min/max values are computed jointly across bands.
    cmap : matplotlib colormap | None
        Colormap for the plots.
    ... (other parameters are the same as the original function).

    Returns
    -------
    fig : instance of matplotlib.figure.Figure
        Figure with a topomap subplot for each band.
    """
    # Extract channel positions
    picks = mne.pick_types(info, meg=False, eeg=True, exclude="bads")
    pos = np.array([info['chs'][pick]['loc'][:2] for pick in picks])

    # Handle defaults
    sphere = _check_sphere(sphere)
    if cbar_fmt == "auto":
        cbar_fmt = "%0.1f"

    band_data = list(band_dict.values())
    band_names = list(band_dict.keys())

    # Handle vmin/vmax
    joint_vlim = vlim == "joint"
    if joint_vlim:
        vlim = (np.min(band_data), np.max(band_data))

    # Set up unit label
    if unit is None:
        unit = "dB" if dB and not normalize else "power"
    else:
        _dB = dB and not normalize
        unit = _format_units_psd(unit, dB=_dB)

    # Set up figure / axes
    n_axes = len(band_dict)
    user_passed_axes = axes is not None
    if user_passed_axes:
        if isinstance(axes, Axes):
            axes = [axes]
        _validate_if_list_of_axes(axes, n_axes)
        fig = axes[0].figure
    else:
        fig, axes = plt.subplots(1, n_axes, figsize=(2 * n_axes, 1.5), layout="constrained")
        if n_axes == 1:
            axes = [axes]

    # Loop over subplots/frequency bands
    for ax, band, data in zip(axes, band_names, band_data):
        plot_colorbar = colorbar and ((not joint_vlim) or (ax == axes[-1]))
        _plot_topomap_multi_cbar(
            data,
            pos,
            ax,
            title=band,
            vlim=vlim,
            cmap=cmap,
            outlines=outlines,
            colorbar=plot_colorbar,
            unit=unit,
            cbar_fmt=cbar_fmt,
            sphere=sphere,
            ch_type=ch_type,
            sensors=sensors,
            names=names,
            mask=mask,
            mask_params=mask_params,
            contours=contours,
            image_interp=image_interp,
            extrapolate=extrapolate,
            border=border,
            res=res,
            size=size,
            cnorm=cnorm,
        )

    if not user_passed_axes:
        fig.canvas.draw()
        plt_show(show)

    return fig
