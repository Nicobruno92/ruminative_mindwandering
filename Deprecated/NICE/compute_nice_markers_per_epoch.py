import sys
import os
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from tqdm import tqdm

from utils.bids_compliance import read_epochs
from NICE.helpers_nice import compute_all_nice_markers

# Example ROI dictionary (can be customized)
ROIS = {
    'prefrontal': ['Fp1', 'Fp2',  'AFz'],
    'frontal': ['F3', 'F4', 'F7', 'F8', 'Fz'],
    'central': ['C3', 'C4', 'Cz', 'CPz'],
    'temporal': ['T7', 'T8'],
    'parietal': ['P3', 'P4', 'P7', 'P8', 'Pz'],
    'occipital': ['O1', 'O2', 'POz'],
    'right': ['Fp2', 'F4', 'F8', 'C4','T8', 'P4', 'P8', 'O2'],
    'left': ['Fp1', 'F3', 'F7', 'C3', 'T7', 'P3', 'P7', 'O1'],
    'midline': ['Fz', 'Cz', 'Pz', 'POz', 'Oz']
}

def process_one(root, subject, task, data_type, desc, reduction_mode, output_dir, tmin=0, tmax=2):
    os.makedirs(output_dir, exist_ok=True)
    results = []
    try:
        epochs, events = read_epochs(
            os.path.join(root, 'derivatives_nico'), subject, task, data_type, desc=desc
        )
    except Exception as e:
        print(f"Skipping {subject} {task}: {e}")
        return
    markers = compute_all_nice_markers(
        epochs, tmin=tmin, tmax=tmax, reduction_mode=reduction_mode, roi_dict=ROIS if isinstance(reduction_mode, dict) else None
    )
    event_ids = epochs.events[:, 2] if hasattr(epochs, 'events') else [None]*len(epochs)
    event_names = []
    if hasattr(epochs, 'event_id') and hasattr(epochs, 'events'):
        inv_event_id = {v: k for k, v in epochs.event_id.items()}
        for eid in event_ids:
            event_names.append(inv_event_id.get(eid, 'unknown'))
    else:
        event_names = ['unknown']*len(event_ids)
    for marker, marker_data in markers.items():
        if reduction_mode == 'whole':
            for ep_idx, value in enumerate(marker_data):
                results.append({
                    'marker': marker,
                    'channel': 'whole',
                    'value': value,
                    'subject_id': subject,
                    'task': task,
                    'epoch': ep_idx,
                    'event_id': event_ids[ep_idx] if ep_idx < len(event_ids) else None,
                    'event_name': event_names[ep_idx] if ep_idx < len(event_names) else None
                })
        elif reduction_mode == 'all':
            channels = epochs.ch_names
            for ch_idx, ch in enumerate(channels):
                for ep_idx in range(marker_data.shape[1]):
                    results.append({
                        'marker': marker,
                        'channel': ch,
                        'value': marker_data[ch_idx, ep_idx],
                        'subject_id': subject,
                        'task': task,
                        'epoch': ep_idx,
                        'event_id': event_ids[ep_idx] if ep_idx < len(event_ids) else None,
                        'event_name': event_names[ep_idx] if ep_idx < len(event_names) else None
                    })
        elif isinstance(reduction_mode, dict):
            for roi_name, roi_values in marker_data.items():
                for ep_idx, value in enumerate(roi_values):
                    results.append({
                        'marker': marker,
                        'channel': roi_name,
                        'value': value,
                        'subject_id': subject,
                        'task': task,
                        'epoch': ep_idx,
                        'event_id': event_ids[ep_idx] if ep_idx < len(event_ids) else None,
                        'event_name': event_names[ep_idx] if ep_idx < len(event_names) else None
                    })
    if reduction_mode == 'all':
        suffix = 'per_electrode'
    elif reduction_mode == 'whole':
        suffix = 'whole_brain'
    elif isinstance(reduction_mode, dict):
        suffix = 'per_roi'
    else:
        suffix = 'markers'
    outname = f"sub-{subject}_task-{task}_{suffix}.csv"
    outpath = os.path.join(output_dir, outname)
    df = pd.DataFrame(results)
    df.to_csv(outpath, index=False)
    print(f"Saved NICE markers to {outpath}")

def main(
    root,
    subjects,
    tasks,
    data_type='eeg',
    desc='autoPreproc',
    reduction_mode='all',
    output_dir='results/nice_markers',
    tmin=0,
    tmax=2
):
    os.makedirs(output_dir, exist_ok=True)
    for subject in tqdm(subjects, desc='Subjects'):
        for task in tasks:
            process_one(root, subject, task, data_type, desc, reduction_mode, output_dir, tmin, tmax)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute NICE markers per epoch for EEG data.')
    parser.add_argument('--subject', type=str, default=None, help='Subject ID (e.g., 02)')
    parser.add_argument('--task', type=str, default=None, help='Task name (e.g., Sart1)')
    parser.add_argument('--output-dir', type=str, default='results/nice_markers', help='Output directory')
    parser.add_argument('--reduction-mode', type=str, default='all', help="Reduction mode: 'all', 'whole', or 'roi'")
    parser.add_argument('--root', type=str, default='/network/iss/cenir/analyse/meeg/CYBERSART/', help='Root data directory')
    parser.add_argument('--tmin', type=float, default=0, help='Start time for marker computation')
    parser.add_argument('--tmax', type=float, default=2, help='End time for marker computation')
    parser.add_argument('--desc', type=str, default='autoPreproc', help='Description for the data')
    args = parser.parse_args()

    # Handle reduction_mode
    if args.reduction_mode == 'roi':
        reduction_mode = ROIS
    elif args.reduction_mode == 'whole':
        reduction_mode = 'whole'
    else:
        reduction_mode = 'all'

    if args.subject and args.task:
        process_one(args.root, args.subject, args.task, 'eeg', args.desc, reduction_mode, args.output_dir, args.tmin, args.tmax)
    else:
        # Default: process all subjects and tasks
        subjects = [f"{i:02}" for i in range(2, 43)]
        tasks = ['Sart1', 'Sart2', 'Sart3', 'Sart4']
        main(args.root, subjects, tasks, 'eeg', args.desc, reduction_mode, args.output_dir, args.tmin, args.tmax) 