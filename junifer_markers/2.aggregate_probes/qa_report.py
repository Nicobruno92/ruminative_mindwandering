#!/usr/bin/env python3
"""
QA Report Generator for Junifer Marker Aggregation Pipeline.

This module generates an interactive HTML report summarizing the aggregation
pipeline results, including:
- Per-subject/session/task statistics
- Probe counts (total, processed, discarded)
- Epoch counts per probe
- Failure tracking and error messages
- Visual summaries with tables and charts
- Topography visualizations for each marker (average, per-subject, ON/OFF task)
"""

import base64
import io
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Optional plotting imports
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for server-side rendering
    import matplotlib.pyplot as plt
    import mne
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False


# =============================================================================
# TOPOGRAPHY UTILITIES
# =============================================================================

def _create_mne_info(channel_names: List[str], montage_name: str = "standard_1020") -> Optional[Any]:
    """Create MNE Info object from channel names."""
    if not HAS_PLOTTING:
        return None
    
    # Filter out non-EEG channels
    eog_channels = ['VEOG', 'HEOG', 'EOG', 'ECG', 'EMG']
    eeg_channels = [ch for ch in channel_names if ch.upper() not in eog_channels]
    
    try:
        info = mne.create_info(eeg_channels, sfreq=256.0, ch_types='eeg')
        montage = mne.channels.make_standard_montage(montage_name)
        info.set_montage(montage, on_missing='ignore')
        return info
    except Exception as e:
        print(f"Warning: Could not create MNE info: {e}")
        return None


def _fig_to_base64(fig: Any) -> str:
    """Convert matplotlib figure to base64 encoded PNG."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64


def _plot_topomap(
    values: np.ndarray,
    info: Any,
    channel_names: List[str],
    title: str = "",
    cmap: str = "RdBu_r",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    figsize: Tuple[float, float] = (4, 4),
) -> Optional[str]:
    """
    Plot a single topomap and return as base64 encoded image.
    
    Returns None if plotting fails.
    """
    if not HAS_PLOTTING or info is None:
        return None
    
    try:
        # Match channels between info and data
        available_channels = [ch for ch in info.ch_names if ch in channel_names]
        if len(available_channels) < 3:
            return None
        
        # Get indices and reorder values
        indices = [channel_names.index(ch) for ch in available_channels]
        ordered_values = values[indices]
        
        # Filter out NaN values for valid channels
        valid_mask = ~np.isnan(ordered_values)
        if valid_mask.sum() < 3:
            return None
        
        # Create subset info
        picks = mne.pick_channels(info.ch_names, include=available_channels, exclude=[])
        info_subset = mne.pick_info(info, picks, copy=True)
        
        # Set color limits
        valid_values = ordered_values[valid_mask]
        if vmin is None:
            vmin = np.percentile(valid_values, 5)
        if vmax is None:
            vmax = np.percentile(valid_values, 95)
        
        # Symmetric color scale for difference plots
        if 'diff' in title.lower() or 'difference' in title.lower():
            abs_max = max(abs(vmin), abs(vmax))
            vmin, vmax = -abs_max, abs_max
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Replace NaN with 0 for plotting
        plot_values = np.nan_to_num(ordered_values, nan=0.0)
        
        im, _ = mne.viz.plot_topomap(
            plot_values,
            info_subset,
            axes=ax,
            show=False,
            cmap=cmap,
            contours=6,
            vlim=(vmin, vmax),
        )
        
        ax.set_title(title, fontsize=10, fontweight='bold')
        plt.colorbar(im, ax=ax, shrink=0.8, format='%.2f')
        
        return _fig_to_base64(fig)
        
    except Exception as e:
        print(f"Warning: Topomap plotting failed for '{title}': {e}")
        return None


def _load_aggregated_probe_data(
    features_root: str,
    summary_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Load all aggregated probe CSV files and combine into a single DataFrame.
    
    Returns DataFrame with columns: subject, session, task, probe_number, label, 
                                     channel, and all marker columns.
    """
    all_data = []
    
    if summary_df.empty or 'output_path' not in summary_df.columns:
        return pd.DataFrame()
    
    for _, row in summary_df.iterrows():
        output_path = row.get('output_path', '')
        if not output_path or not Path(output_path).exists():
            continue
        
        try:
            probe_df = pd.read_csv(output_path)
            # Add identifying columns if not present
            probe_df['subject'] = row.get('subject', '')
            probe_df['session'] = row.get('session', '')
            probe_df['task'] = row.get('task', '')
            probe_df['probe_number'] = row.get('probe_number', 0)
            probe_df['label'] = row.get('label', 'unknown')
            probe_df['marker_type'] = row.get('marker_type', 'unknown')
            all_data.append(probe_df)
        except Exception as e:
            print(f"Warning: Could not load {output_path}: {e}")
            continue
    
    if not all_data:
        return pd.DataFrame()
    
    return pd.concat(all_data, ignore_index=True)


def _get_marker_columns(df: pd.DataFrame) -> List[str]:
    """Get list of marker columns (excluding metadata columns)."""
    metadata_cols = [
        'subject', 'session', 'task', 'probe_number', 'label', 'n_epochs',
        'channel', 'marker_type', 'ontask_label', 'content', 
        'confidence_level', 'depth_level'
    ]
    return [c for c in df.columns if c not in metadata_cols]


def _compute_topography_data(
    probe_data: pd.DataFrame,
    marker_columns: List[str],
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Compute average topography data for each marker.
    
    Returns dict: marker_name -> {
        'grand_avg': array (channels,),
        'per_subject': {subject: array (channels,)},
        'ontask_avg': array (channels,),
        'offtask_avg': array (channels,),
        'diff': array (channels,),  # ontask - offtask
        'channel_names': list
    }
    """
    if probe_data.empty or 'channel' not in probe_data.columns:
        return {}
    
    channel_names = probe_data['channel'].unique().tolist()
    topo_data = {}
    
    for marker in marker_columns:
        if marker not in probe_data.columns:
            continue
        
        try:
            # Grand average across all probes
            grand_avg = probe_data.groupby('channel')[marker].mean()
            
            # Per-subject averages
            per_subject = {}
            for subject in probe_data['subject'].unique():
                sub_df = probe_data[probe_data['subject'] == subject]
                sub_avg = sub_df.groupby('channel')[marker].mean()
                per_subject[subject] = sub_avg.reindex(channel_names).values
            
            # On-task vs Off-task
            ontask_df = probe_data[probe_data['label'] == 'onTask']
            offtask_df = probe_data[probe_data['label'] == 'offTask']
            
            ontask_avg = ontask_df.groupby('channel')[marker].mean() if not ontask_df.empty else pd.Series()
            offtask_avg = offtask_df.groupby('channel')[marker].mean() if not offtask_df.empty else pd.Series()
            
            # Reindex to ensure consistent channel order
            grand_avg = grand_avg.reindex(channel_names).values
            ontask_avg = ontask_avg.reindex(channel_names).values if not ontask_avg.empty else np.full(len(channel_names), np.nan)
            offtask_avg = offtask_avg.reindex(channel_names).values if not offtask_avg.empty else np.full(len(channel_names), np.nan)
            
            # Compute difference
            diff = ontask_avg - offtask_avg
            
            topo_data[marker] = {
                'grand_avg': grand_avg,
                'per_subject': per_subject,
                'ontask_avg': ontask_avg,
                'offtask_avg': offtask_avg,
                'diff': diff,
                'channel_names': channel_names,
                'n_ontask': len(ontask_df['probe_number'].unique()) if not ontask_df.empty else 0,
                'n_offtask': len(offtask_df['probe_number'].unique()) if not offtask_df.empty else 0,
            }
            
        except Exception as e:
            print(f"Warning: Could not compute topography for {marker}: {e}")
            continue
    
    return topo_data


def _generate_topography_pages(
    topo_data: Dict[str, Dict[str, np.ndarray]],
    mne_info: Any,
) -> Dict[str, str]:
    """
    Generate HTML content for topography pages.
    
    Returns dict: marker_name -> html_content
    """
    if not HAS_PLOTTING or not topo_data:
        return {}
    
    pages = {}
    
    for marker_name, data in topo_data.items():
        channel_names = data['channel_names']
        per_subject = data['per_subject']
        
        # Generate images
        images = {}
        
        # Grand average
        grand_avg_img = _plot_topomap(
            data['grand_avg'], mne_info, channel_names,
            title=f"Grand Average (all probes)",
            cmap="RdBu_r"
        )
        if grand_avg_img:
            images['grand_avg'] = grand_avg_img
        
        # On-task average
        if data['n_ontask'] > 0:
            ontask_img = _plot_topomap(
                data['ontask_avg'], mne_info, channel_names,
                title=f"ON-TASK (n={data['n_ontask']})",
                cmap="RdBu_r"
            )
            if ontask_img:
                images['ontask'] = ontask_img
        
        # Off-task average
        if data['n_offtask'] > 0:
            offtask_img = _plot_topomap(
                data['offtask_avg'], mne_info, channel_names,
                title=f"OFF-TASK (n={data['n_offtask']})",
                cmap="RdBu_r"
            )
            if offtask_img:
                images['offtask'] = offtask_img
        
        # Difference (ON - OFF)
        if data['n_ontask'] > 0 and data['n_offtask'] > 0:
            diff_img = _plot_topomap(
                data['diff'], mne_info, channel_names,
                title="DIFFERENCE (ON - OFF)",
                cmap="RdBu_r"
            )
            if diff_img:
                images['diff'] = diff_img
        
        # Per-subject topomaps
        subject_images = {}
        for subject, values in sorted(per_subject.items()):
            sub_img = _plot_topomap(
                values, mne_info, channel_names,
                title=f"sub-{subject}",
                cmap="RdBu_r",
                figsize=(3, 3)
            )
            if sub_img:
                subject_images[subject] = sub_img
        
        # Build page HTML
        page_html = _build_topography_page(marker_name, images, subject_images, data)
        pages[marker_name] = page_html
    
    return pages


def _build_topography_page(
    marker_name: str,
    images: Dict[str, str],
    subject_images: Dict[str, str],
    data: Dict,
) -> str:
    """Build HTML content for a single marker's topography page."""
    
    # Placeholder for lazy loading
    placeholder = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E%3Crect width='100%25' height='100%25' fill='%23f1f5f9'/%3E%3C/svg%3E"
    
    # Main comparison section
    comparison_html = '<div class="topo-comparison">'
    
    if 'grand_avg' in images:
        comparison_html += f'''
        <div class="topo-card">
            <img class="lazy-load" src="{placeholder}" data-src="data:image/png;base64,{images['grand_avg']}" alt="Grand Average">
        </div>
        '''
    
    if 'ontask' in images:
        comparison_html += f'''
        <div class="topo-card">
            <img class="lazy-load" src="{placeholder}" data-src="data:image/png;base64,{images['ontask']}" alt="ON-TASK">
        </div>
        '''
    
    if 'offtask' in images:
        comparison_html += f'''
        <div class="topo-card">
            <img class="lazy-load" src="{placeholder}" data-src="data:image/png;base64,{images['offtask']}" alt="OFF-TASK">
        </div>
        '''
    
    if 'diff' in images:
        comparison_html += f'''
        <div class="topo-card highlight-diff">
            <img class="lazy-load" src="{placeholder}" data-src="data:image/png;base64,{images['diff']}" alt="Difference">
        </div>
        '''
    
    comparison_html += '</div>'
    
    # Per-subject grid
    subjects_html = ''
    if subject_images:
        subjects_html = '''
        <div class="section">
            <h3 class="collapsible" onclick="toggleContent(this)">Per-Subject Topographies</h3>
            <div class="content">
                <div class="subject-grid">
        '''
        for subject, img in sorted(subject_images.items()):
            subjects_html += f'''
            <div class="subject-topo">
                <img class="lazy-load" src="{placeholder}" data-src="data:image/png;base64,{img}" alt="sub-{subject}">
            </div>
            '''
        subjects_html += '</div></div></div>'
    
    # Statistics summary
    stats_html = f'''
    <div class="topo-stats">
        <span class="stat-item">ON-TASK probes: {data.get('n_ontask', 0)}</span>
        <span class="stat-item">OFF-TASK probes: {data.get('n_offtask', 0)}</span>
        <span class="stat-item">Channels: {len(data.get('channel_names', []))}</span>
    </div>
    '''
    
    return f'''
    <div class="topo-page" id="topo-{marker_name}">
        <h2>🧠 {marker_name}</h2>
        {stats_html}
        <h3>Grand Average & ON/OFF Task Comparison</h3>
        {comparison_html}
        {subjects_html}
    </div>
    '''


def generate_html_report(
    summary_df: pd.DataFrame,
    stats: Dict[str, Any],
    config: Dict,
    output_path: str,
) -> str:
    """
    Generate a comprehensive HTML QA report for the aggregation pipeline.
    
    Parameters
    ----------
    summary_df : pd.DataFrame
        DataFrame with all processed probes (from aggregation outputs)
    stats : Dict[str, Any]
        Dictionary containing aggregation statistics:
        - discarded_probes: List of dicts with discarded probe info
        - errors: List of dicts with error info
        - warnings: List of dicts with warning info
        - per_subject_stats: Dict with per-subject statistics
    config : Dict
        Configuration dictionary used for the run
    output_path : str
        Path where the HTML report will be saved
        
    Returns
    -------
    str
        Path to the generated HTML report
    """
    # Compute summary statistics
    total_subjects = len(summary_df['subject'].unique()) if not summary_df.empty else 0
    total_sessions = len(summary_df['session'].unique()) if not summary_df.empty else 0
    total_tasks = len(summary_df['task'].unique()) if not summary_df.empty else 0
    total_probes_processed = len(summary_df) if not summary_df.empty else 0
    
    discarded_probes = stats.get('discarded_probes', [])
    total_probes_discarded = len(discarded_probes)
    total_probes = total_probes_processed + total_probes_discarded
    
    errors = stats.get('errors', [])
    warnings = stats.get('warnings', [])
    epoch_mismatches = stats.get('epoch_mismatches', [])
    skipped_markers = stats.get('skipped_markers', [])
    
    # Get epoch type breakdown
    epoch_type_counts = {}
    if not summary_df.empty and 'marker_type' in summary_df.columns:
        epoch_type_counts = summary_df.groupby('marker_type').size().to_dict()
    
    # Get label distribution
    label_counts = {}
    if not summary_df.empty and 'label' in summary_df.columns:
        label_counts = summary_df.groupby('label').size().to_dict()
    
    # Compute per-subject statistics
    subject_stats = []
    if not summary_df.empty:
        for subject in sorted(summary_df['subject'].unique()):
            sub_df = summary_df[summary_df['subject'] == subject]
            sub_discarded = [d for d in discarded_probes if d.get('subject') == subject]
            
            # Get sessions for this subject
            sessions = sub_df['session'].unique().tolist()
            
            for session in sessions:
                ses_df = sub_df[sub_df['session'] == session]
                ses_discarded = [d for d in sub_discarded if d.get('session') == session]
                
                for task in ses_df['task'].unique():
                    task_df = ses_df[ses_df['task'] == task]
                    task_discarded = [d for d in ses_discarded if d.get('task') == task]
                    
                    # Get epoch type breakdown
                    evoked_count = len(task_df[task_df['marker_type'] == 'evoked'])
                    state_count = len(task_df[task_df['marker_type'] == 'state'])
                    sleep_count = len(task_df[task_df['marker_type'] == 'sleep'])
                    
                    # Get average epochs per probe
                    avg_epochs = task_df['n_epochs'].mean() if 'n_epochs' in task_df.columns else 0
                    
                    # Get on/off task distribution
                    on_task = len(task_df[task_df['label'] == 'onTask'])
                    off_task = len(task_df[task_df['label'] == 'offTask'])
                    unknown_label = len(task_df[task_df['label'] == 'unknown'])
                    
                    subject_stats.append({
                        'subject': subject,
                        'session': session,
                        'task': task,
                        'probes_processed': len(task_df),
                        'probes_discarded': len(task_discarded),
                        'evoked_probes': evoked_count,
                        'state_probes': state_count,
                        'sleep_probes': sleep_count,
                        'avg_epochs': round(avg_epochs, 1),
                        'on_task': on_task,
                        'off_task': off_task,
                        'unknown_label': unknown_label,
                    })
    
    # ==========================================================================
    # TOPOGRAPHY GENERATION
    # ==========================================================================
    topography_collections = {}
    
    if HAS_PLOTTING and not summary_df.empty:
        print("  Generating topography visualizations...")
        
        features_root = config.get('project', {}).get('features_root', '')
        
        # Identify marker types present in the data
        marker_types = summary_df['marker_type'].unique().tolist() if 'marker_type' in summary_df.columns else ['unknown']
        
        for m_type in marker_types:
            print(f"    Processing marker type: {m_type}")
            
            # Filter summary for this marker type
            type_df = summary_df[summary_df['marker_type'] == m_type]
            
            # Load data for this marker type
            probe_data = _load_aggregated_probe_data(features_root, type_df)
            
            if not probe_data.empty:
                # Get marker columns
                marker_columns = _get_marker_columns(probe_data)
                print(f"      Found {len(marker_columns)} marker columns")
                
                # Compute topography data
                topo_data = _compute_topography_data(probe_data, marker_columns)
                
                # Create MNE info for plotting
                if 'channel' in probe_data.columns:
                    channel_names = probe_data['channel'].unique().tolist()
                    mne_info = _create_mne_info(channel_names)
                    
                    if mne_info is not None:
                        # Generate topography pages
                        pages = _generate_topography_pages(topo_data, mne_info)
                        if pages:
                            topography_collections[m_type] = pages
                            print(f"      Generated {len(pages)} topography pages for {m_type}")
    else:
        if not HAS_PLOTTING:
            print("  Skipping topographies (matplotlib/mne not available)")
    
    # Build HTML
    html = _build_html_document(
        total_subjects=total_subjects,
        total_sessions=total_sessions,
        total_tasks=total_tasks,
        total_probes=total_probes,
        total_probes_processed=total_probes_processed,
        total_probes_discarded=total_probes_discarded,
        epoch_type_counts=epoch_type_counts,
        label_counts=label_counts,
        subject_stats=subject_stats,
        discarded_probes=discarded_probes,
        errors=errors,
        warnings=warnings,
        epoch_mismatches=epoch_mismatches,
        skipped_markers=skipped_markers,
        missing_epoch_types=stats.get('missing_epoch_types', []),
        config=config,
        summary_df=summary_df,
        topography_collections=topography_collections,
    )
    
    # Save report
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path_obj, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"\n[QA REPORT] Generated: {output_path_obj}")
    return str(output_path_obj)


def _build_html_document(
    total_subjects: int,
    total_sessions: int,
    total_tasks: int,
    total_probes: int,
    total_probes_processed: int,
    total_probes_discarded: int,
    epoch_type_counts: Dict[str, int],
    label_counts: Dict[str, int],
    subject_stats: List[Dict],
    discarded_probes: List[Dict],
    errors: List[Dict],
    warnings: List[Dict],
    epoch_mismatches: List[Dict],
    skipped_markers: List[Dict],
    missing_epoch_types: List[Dict],
    config: Dict,
    summary_df: pd.DataFrame,
    topography_collections: Optional[Dict[str, Dict[str, str]]] = None,
) -> str:
    """Build the complete HTML document with optional topography pages."""
    
    topography_collections = topography_collections or {}
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Calculate success rate
    success_rate = (total_probes_processed / total_probes * 100) if total_probes > 0 else 0
    
    # Determine overall status
    if len(errors) > 0:
        status_class = "status-error"
        status_text = "Completed with Errors"
    elif total_probes_discarded > 0 or len(warnings) > 0:
        status_class = "status-warning"
        status_text = "Completed with Warnings"
    else:
        status_class = "status-success"
        status_text = "Completed Successfully"
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Junifer Marker Aggregation QA Report</title>
    <style>
        :root {{
            --primary-color: #2563eb;
            --success-color: #16a34a;
            --warning-color: #d97706;
            --error-color: #dc2626;
            --bg-color: #f8fafc;
            --card-bg: #ffffff;
            --text-color: #1e293b;
            --text-muted: #64748b;
            --border-color: #e2e8f0;
        }}
        
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            line-height: 1.6;
            padding: 2rem;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        header {{
            text-align: center;
            margin-bottom: 2rem;
            padding: 2rem;
            background: linear-gradient(135deg, var(--primary-color), #7c3aed);
            color: white;
            border-radius: 1rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }}
        
        header h1 {{
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }}
        
        header p {{
            opacity: 0.9;
        }}
        
        .status-badge {{
            display: inline-block;
            padding: 0.5rem 1.5rem;
            border-radius: 2rem;
            font-weight: 600;
            margin-top: 1rem;
        }}
        
        .status-success {{
            background-color: var(--success-color);
        }}
        
        .status-warning {{
            background-color: var(--warning-color);
        }}
        
        .status-error {{
            background-color: var(--error-color);
        }}
        
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}
        
        .summary-card {{
            background: var(--card-bg);
            border-radius: 0.75rem;
            padding: 1.5rem;
            text-align: center;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
            border: 1px solid var(--border-color);
        }}
        
        .summary-card .value {{
            font-size: 2.5rem;
            font-weight: 700;
            color: var(--primary-color);
        }}
        
        .summary-card .label {{
            color: var(--text-muted);
            font-size: 0.875rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        
        .summary-card.success .value {{
            color: var(--success-color);
        }}
        
        .summary-card.warning .value {{
            color: var(--warning-color);
        }}
        
        .summary-card.error .value {{
            color: var(--error-color);
        }}
        
        .section {{
            background: var(--card-bg);
            border-radius: 0.75rem;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
            border: 1px solid var(--border-color);
        }}
        
        .section h2 {{
            font-size: 1.25rem;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid var(--primary-color);
            color: var(--text-color);
        }}
        
        .section h3 {{
            font-size: 1rem;
            margin: 1rem 0 0.5rem 0;
            color: var(--text-muted);
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.875rem;
        }}
        
        th, td {{
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }}
        
        th {{
            background-color: var(--bg-color);
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            font-size: 0.75rem;
            letter-spacing: 0.05em;
        }}
        
        tr:hover {{
            background-color: var(--bg-color);
        }}
        
        .bar-chart {{
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }}
        
        .bar-row {{
            display: flex;
            align-items: center;
            gap: 1rem;
        }}
        
        .bar-label {{
            min-width: 100px;
            font-size: 0.875rem;
        }}
        
        .bar-container {{
            flex: 1;
            height: 24px;
            background-color: var(--bg-color);
            border-radius: 4px;
            overflow: hidden;
        }}
        
        .bar {{
            height: 100%;
            background: linear-gradient(90deg, var(--primary-color), #7c3aed);
            border-radius: 4px;
            transition: width 0.3s ease;
        }}
        
        .bar-value {{
            min-width: 60px;
            text-align: right;
            font-weight: 600;
            font-size: 0.875rem;
        }}
        
        .pills {{
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
        }}
        
        .pill {{
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 1rem;
            font-size: 0.75rem;
            font-weight: 500;
        }}
        
        .pill-success {{
            background-color: #dcfce7;
            color: var(--success-color);
        }}
        
        .pill-warning {{
            background-color: #fef3c7;
            color: var(--warning-color);
        }}
        
        .pill-error {{
            background-color: #fee2e2;
            color: var(--error-color);
        }}
        
        .pill-info {{
            background-color: #dbeafe;
            color: var(--primary-color);
        }}
        
        .collapsible {{
            cursor: pointer;
            user-select: none;
        }}
        
        .collapsible::before {{
            content: '▶';
            display: inline-block;
            margin-right: 0.5rem;
            transition: transform 0.2s;
        }}
        
        .collapsible.active::before {{
            transform: rotate(90deg);
        }}
        
        .content {{
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.3s ease-out;
        }}
        
        .content.show {{
            max-height: 2000px;
        }}
        
        .error-box {{
            background-color: #fee2e2;
            border: 1px solid var(--error-color);
            border-radius: 0.5rem;
            padding: 1rem;
            margin: 0.5rem 0;
        }}
        
        .warning-box {{
            background-color: #fef3c7;
            border: 1px solid var(--warning-color);
            border-radius: 0.5rem;
            padding: 1rem;
            margin: 0.5rem 0;
        }}
        
        .config-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1rem;
        }}
        
        .config-item {{
            display: flex;
            justify-content: space-between;
            padding: 0.5rem 0;
            border-bottom: 1px dashed var(--border-color);
        }}
        
        .config-key {{
            font-weight: 500;
            color: var(--text-muted);
        }}
        
        .config-value {{
            font-family: monospace;
            color: var(--text-color);
        }}
        
        .progress-ring {{
            display: inline-block;
            width: 120px;
            height: 120px;
        }}
        
        .two-col {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
        }}
        
        @media (max-width: 768px) {{
            .two-col {{
                grid-template-columns: 1fr;
            }}
        }}
        
        footer {{
            text-align: center;
            padding: 2rem;
            color: var(--text-muted);
            font-size: 0.875rem;
        }}
        
        .scroll-table {{
            overflow-x: auto;
        }}
        
        .highlight {{
            background-color: #fef3c7;
        }}
        
        .no-data {{
            text-align: center;
            padding: 2rem;
            color: var(--text-muted);
            font-style: italic;
        }}
        
        /* Navigation tabs for pages */
        .nav-tabs {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-bottom: 1.5rem;
            padding: 1rem;
            background: var(--card-bg);
            border-radius: 0.75rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }}
        
        .nav-tab {{
            padding: 0.5rem 1rem;
            border: 2px solid var(--border-color);
            border-radius: 0.5rem;
            cursor: pointer;
            font-weight: 500;
            transition: all 0.2s;
            background: var(--bg-color);
        }}
        
        .nav-tab:hover {{
            border-color: var(--primary-color);
            color: var(--primary-color);
        }}
        
        .nav-tab.active {{
            background: var(--primary-color);
            color: white;
            border-color: var(--primary-color);
        }}
        
        .page {{
            display: none;
        }}
        
        .page.active {{
            display: block;
        }}
        
        /* Topography styles */
        .topo-comparison {{
            display: flex;
            flex-wrap: wrap;
            gap: 1.5rem;
            justify-content: center;
            margin: 1.5rem 0;
        }}
        
        .topo-card {{
            background: var(--card-bg);
            border-radius: 0.75rem;
            padding: 1rem;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            text-align: center;
        }}
        
        .topo-card img {{
            max-width: 100%;
            height: auto;
        }}
        
        .topo-card.highlight-diff {{
            border: 3px solid var(--primary-color);
            background: linear-gradient(135deg, #f0f9ff, #e0f2fe);
        }}
        
        .subject-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 1rem;
            margin-top: 1rem;
        }}
        
        .subject-topo {{
            background: var(--card-bg);
            border-radius: 0.5rem;
            padding: 0.5rem;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
            text-align: center;
        }}
        
        .subject-topo img {{
            max-width: 100%;
            height: auto;
        }}
        
        .topo-stats {{
            display: flex;
            gap: 2rem;
            justify-content: center;
            flex-wrap: wrap;
            margin: 1rem 0;
            padding: 1rem;
            background: var(--bg-color);
            border-radius: 0.5rem;
        }}
        
        .stat-item {{
            font-weight: 500;
            color: var(--text-muted);
        }}
        
        .topo-page {{
            padding: 1rem 0;
        }}
        
        .topo-page h2 {{
            font-size: 1.5rem;
            margin-bottom: 1rem;
            color: var(--text-color);
        }}
        
        .topo-page h3 {{
            font-size: 1.1rem;
            margin: 1.5rem 0 0.5rem 0;
            color: var(--text-muted);
        }}
        /* Tabs Styles */
        /* Tabs Styles */
        .tabs {{
            display: flex;
            flex-wrap: wrap;
            border-bottom: 2px solid var(--border-color);
            margin-bottom: 1.5rem;
            gap: 0.5rem;
        }}
        
        .tab-btn {{
            padding: 0.5rem 1rem;
            cursor: pointer;
            background: none;
            border: none;
            border-bottom: 3px solid transparent;
            font-weight: 600;
            color: var(--text-muted);
            font-size: 0.9rem;
            transition: all 0.2s;
            margin-bottom: -2px;
            white-space: nowrap;
        }}
        
        .tab-btn:hover {{
            color: var(--primary-color);
            background-color: #f1f5f9;
            border-radius: 0.5rem 0.5rem 0 0;
        }}
        
        .tab-btn.active {{
            color: var(--primary-color);
            border-bottom-color: var(--primary-color);
        }}
        
        .tab-content {{
            display: none;
            animation: fadeIn 0.3s ease;
        }}
        
        .tab-content.active {{
            display: block;
        }}
        
        /* Dropdown Styles for Markers */
        .marker-selector-container {{
            margin-bottom: 2rem;
            padding: 1.25rem;
            background: #f1f5f9;
            border-radius: 0.75rem;
            border: 1px solid var(--border-color);
            display: flex;
            align-items: center;
            gap: 1.5rem;
            box-shadow: inset 0 2px 4px rgba(0,0,0,0.05);
        }}

        .marker-selector-label {{
            font-weight: 700;
            color: var(--text-color);
            font-size: 1rem;
            white-space: nowrap;
        }}

        .marker-selector-wrapper {{
            flex: 1;
            position: relative;
            max-width: 600px;
        }}

        .marker-selector {{
            width: 100%;
            padding: 0.75rem 1.25rem;
            border-radius: 0.5rem;
            border: 2px solid var(--border-color);
            background: white;
            font-size: 1.1rem;
            font-weight: 600;
            color: var(--primary-color);
            cursor: pointer;
            transition: all 0.2s;
            appearance: none;
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 24 24' stroke='%232563eb'%3E%3Cpath stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M19 9l-7 7-7-7'%3E%3C/path%3E%3C/svg%3E");
            background-repeat: no-repeat;
            background-position: right 1.25rem center;
            background-size: 1.5rem;
            padding-right: 3.5rem;
        }}

        .marker-selector:hover {{
            border-color: var(--primary-color);
            background-color: #f8fafc;
        }}

        .marker-selector:focus {{
            outline: none;
            border-color: var(--primary-color);
            box-shadow: 0 0 0 4px rgba(37, 99, 235, 0.1);
        }}
        
        @keyframes fadeIn {{
            from {{ opacity: 0; }}
            to {{ opacity: 1; }}
        }}
    </style>
    <script>
        function toggleContent(header) {{
            header.classList.toggle('active');
            var content = header.nextElementSibling;
            if (content.style.display === 'block') {{
                content.style.display = 'none';
            }} else {{
                content.style.display = 'block';
            }}
        }}

        function loadLazyImages(container) {{
            var images = container.querySelectorAll('img.lazy-load');
            images.forEach(function(img) {{
                if (img.dataset.src) {{
                    img.src = img.dataset.src;
                    img.classList.remove('lazy-load');
                    img.removeAttribute('data-src');
                }}
            }});
        }}

        function openTab(evt, tabName) {{
            // Hide all tab content
            var i, tabcontent, tablinks;
            tabcontent = document.getElementsByClassName("tab-content");
            for (i = 0; i < tabcontent.length; i++) {{
                tabcontent[i].style.display = "none";
                tabcontent[i].classList.remove("active");
            }}

            // Remove active class from all tab buttons
            tablinks = document.getElementsByClassName("tab-btn");
            for (i = 0; i < tablinks.length; i++) {{
                tablinks[i].classList.remove("active");
            }}

            // Show the current tab, and add an "active" class to the button
            var activeTab = document.getElementById(tabName);
            activeTab.style.display = "block";
            activeTab.classList.add("active");
            evt.currentTarget.classList.add("active");
            
            // Trigger lazy load for this tab
            loadLazyImages(activeTab);
        }}

        function switchMarker(selectElement) {{
            var tabName = selectElement.value;
            var i, tabcontent;
            
            // Hide all tab contents in the current page
            var page = selectElement.closest('.section');
            tabcontent = page.getElementsByClassName("tab-content");
            for (i = 0; i < tabcontent.length; i++) {{
                tabcontent[i].style.display = "none";
                tabcontent[i].classList.remove("active");
            }}

            // Show selected
            var activeTab = document.getElementById(tabName);
            if (activeTab) {{
                activeTab.style.display = "block";
                activeTab.classList.add("active");
                loadLazyImages(activeTab);
            }}
        }}
        
        // Initial load for active tab
        document.addEventListener("DOMContentLoaded", function() {{
            var activeTabs = document.querySelectorAll('.tab-content.active, .page.active');
            activeTabs.forEach(function(tab) {{
                loadLazyImages(tab);
            }});
        }});
    </script>
</head>
<body>
    <div class="container">
        <header>
            <h1>🧠 Junifer Marker Aggregation QA Report</h1>
            <p>Mind-Wandering EEG Analysis Pipeline</p>
            <p>Generated: {timestamp}</p>
            <span class="status-badge {status_class}">{status_text}</span>
        </header>
        
        <!-- Navigation Tabs -->
        <div class="nav-tabs">
            <div class="nav-tab active" onclick="showPage('main')">📊 Summary</div>
            {_build_topo_nav_tabs(topography_collections)}
        </div>
        
        <!-- Main Summary Page -->
        <div class="page active" id="page-main">
        
        <!-- Summary Cards -->
        <div class="summary-grid">
            <div class="summary-card">
                <div class="value">{total_subjects}</div>
                <div class="label">Subjects</div>
            </div>
            <div class="summary-card">
                <div class="value">{total_sessions}</div>
                <div class="label">Sessions</div>
            </div>
            <div class="summary-card">
                <div class="value">{total_tasks}</div>
                <div class="label">Tasks</div>
            </div>
            <div class="summary-card success">
                <div class="value">{total_probes_processed}</div>
                <div class="label">Probes Processed</div>
            </div>
            <div class="summary-card {{'warning' if total_probes_discarded > 0 else ''}}">
                <div class="value">{total_probes_discarded}</div>
                <div class="label">Probes Discarded</div>
            </div>
            <div class="summary-card">
                <div class="value">{success_rate:.1f}%</div>
                <div class="label">Success Rate</div>
            </div>
        </div>
        
        <!-- Epoch Type & Label Distribution -->
        <div class="two-col">
            <div class="section">
                <h2>📊 Epoch Type Distribution</h2>
                {_build_bar_chart(epoch_type_counts, 'Epoch Type')}
            </div>
            <div class="section">
                <h2>🏷️ Label Distribution (On-Task vs Off-Task)</h2>
                {_build_bar_chart(label_counts, 'Label')}
            </div>
        </div>
        
        <!-- Per-Subject Statistics -->
        <div class="section">
            <h2>👥 Per-Subject Statistics</h2>
            {_build_subject_table(subject_stats)}
        </div>
        
        <!-- Epochs per Probe Distribution -->
        <div class="section">
            <h2>📈 Epochs Aggregated per Probe</h2>
            {_build_epochs_distribution(summary_df)}
        </div>
        
        <!-- Discarded Probes -->
        <div class="section">
            <h2 class="collapsible" onclick="toggleContent(this)">⚠️ Discarded Probes ({len(discarded_probes)})</h2>
            <div class="content">
                {_build_discarded_table(discarded_probes)}
            </div>
        </div>
        
        <!-- Errors -->
        {_build_errors_section(errors)}
        
        <!-- Warnings -->
        {_build_warnings_section(warnings)}
        
        <!-- Missing Epoch Types -->
        {_build_missing_epoch_types_section(missing_epoch_types)}
        
        <!-- Data Issues (Epoch Mismatches and Skipped Markers) -->
        {_build_data_issues_section(epoch_mismatches, skipped_markers)}
        
        <!-- Configuration Summary -->
        <div class="section">
            <h2 class="collapsible" onclick="toggleContent(this)">⚙️ Configuration Used</h2>
            <div class="content">
                {_build_config_section(config)}
            </div>
        </div>
        
        </div><!-- End of main page -->
        
        <!-- Topography Pages -->
        {_build_all_topo_pages(topography_collections)}
        
        <footer>
            <p>Junifer Marker Aggregation Pipeline | Mind-Wandering EEG Analysis</p>
            <p>Report generated on {timestamp}</p>
        </footer>
    </div>
    
    <script>
        function toggleContent(element) {{
            element.classList.toggle('active');
            var content = element.nextElementSibling;
            content.classList.toggle('show');
        }}
        
        function showPage(pageId) {{
            // Hide all pages
            document.querySelectorAll('.page').forEach(function(page) {{
                page.classList.remove('active');
            }});
            // Show selected page
            var selectedPage = document.getElementById('page-' + pageId);
            if (selectedPage) {{
                selectedPage.classList.add('active');
                // Trigger lazy loading for images in this page
                loadLazyImages(selectedPage);
                
                // If this is a topo page, also ensure current marker is loaded
                var activeMarker = selectedPage.querySelector('.tab-content.active');
                if (activeMarker) {{
                    loadLazyImages(activeMarker);
                }}
            }}
            // Update tabs
            document.querySelectorAll('.nav-tab').forEach(function(tab) {{
                tab.classList.remove('active');
            }});
            event.target.classList.add('active');
        }}
        
        // Auto-expand if there are errors
        document.addEventListener('DOMContentLoaded', function() {{
            var errorSections = document.querySelectorAll('.section h2');
            errorSections.forEach(function(h2) {{
                if (h2.textContent.includes('Error') && !h2.textContent.includes('(0)')) {{
                    h2.click();
                }}
            }});
        }});
    </script>
</body>
</html>
"""
    return html


def _build_bar_chart(data: Dict[str, int], title: str) -> str:
    """Build a horizontal bar chart from a dictionary."""
    if not data:
        return '<p class="no-data">No data available</p>'
    
    max_val = max(data.values()) if data.values() else 1
    
    rows = []
    for label, value in sorted(data.items()):
        pct = (value / max_val * 100) if max_val > 0 else 0
        rows.append(f"""
        <div class="bar-row">
            <div class="bar-label">{label}</div>
            <div class="bar-container">
                <div class="bar" style="width: {pct}%;"></div>
            </div>
            <div class="bar-value">{value}</div>
        </div>
        """)
    
    return f'<div class="bar-chart">{"".join(rows)}</div>'


def _build_topo_nav_tabs(topography_collections: Dict[str, Dict[str, str]]) -> str:
    """Build navigation tabs for marker types."""
    if not topography_collections:
        return ''
    
    tabs = []
    # Sort types: evoked, state, sleep, then others
    preferred_order = ['evoked', 'state', 'sleep']
    types = sorted(topography_collections.keys(), key=lambda x: (preferred_order.index(x) if x in preferred_order else 999, x))
    
    for m_type in types:
        label = m_type.capitalize()
        tabs.append(
            f'<div class="nav-tab" onclick="showPage(\'type-{m_type}\')">🧠 {label}</div>'
        )
    
    return ''.join(tabs)


def _build_all_topo_pages(topography_collections: Dict[str, Dict[str, str]]) -> str:
    """Build topography pages for each marker type."""
    if not topography_collections:
        return ''
    
    pages = []
    for m_type, markers_dict in topography_collections.items():
        if not markers_dict:
            continue
            
        # Build options for the dropdown and content sections
        options = []
        internal_contents = []
        
        marker_names = sorted(markers_dict.keys())
        for i, m_name in enumerate(marker_names):
            active_cls = " active" if i == 0 else ""
            # Replace characters that might break HTML ID/Select value
            safe_name = m_name.replace("'", "").replace(" ", "_").replace(".", "_")
            content_id = f"content-{m_type}-{safe_name}"
            
            options.append(f'<option value="{content_id}">{m_name}</option>')
            
            display_style = "block" if i == 0 else "none"
            internal_contents.append(f'''
                <div id="{content_id}" class="tab-content{active_cls}" style="display: {display_style};">
                    {markers_dict[m_name]}
                </div>
            ''')
            
        page_html = f'''
        <div class="page" id="page-type-{m_type}">
            <div class="section">
                <h2>{m_type.capitalize()} Markers Analysis</h2>
                
                <div class="marker-selector-container">
                    <div class="marker-selector-label">📍 Select Marker to Visualize:</div>
                    <div class="marker-selector-wrapper">
                        <select class="marker-selector" onchange="switchMarker(this)">
                            {''.join(options)}
                        </select>
                    </div>
                </div>

                <div class="marker-content-area">
                    {''.join(internal_contents)}
                </div>
            </div>
        </div>
        '''
        pages.append(page_html)
    
    return ''.join(pages)


def _build_subject_table(subject_stats: List[Dict]) -> str:
    """Build the per-subject statistics table."""
    if not subject_stats:
        return '<p class="no-data">No subject data available</p>'
    
    rows = []
    for stat in subject_stats:
        row_class = 'highlight' if stat['probes_discarded'] > 0 else ''
        rows.append(f"""
        <tr class="{row_class}">
            <td>{stat['subject']}</td>
            <td>{stat['session']}</td>
            <td>{stat['task']}</td>
            <td>{stat['probes_processed']}</td>
            <td>{stat['probes_discarded']}</td>
            <td>{stat['evoked_probes']}</td>
            <td>{stat['state_probes']}</td>
            <td>{stat['sleep_probes']}</td>
            <td>{stat['avg_epochs']}</td>
            <td>
                <span class="pill pill-success">{stat['on_task']} On</span>
                <span class="pill pill-warning">{stat['off_task']} Off</span>
                {f'<span class="pill pill-info">{stat["unknown_label"]} Unk</span>' if stat['unknown_label'] > 0 else ''}
            </td>
        </tr>
        """)
    
    return f"""
    <div class="scroll-table">
        <table>
            <thead>
                <tr>
                    <th>Subject</th>
                    <th>Session</th>
                    <th>Task</th>
                    <th>Processed</th>
                    <th>Discarded</th>
                    <th>Evoked</th>
                    <th>State</th>
                    <th>Sleep</th>
                    <th>Avg Epochs</th>
                    <th>Labels</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
    </div>
    """


def _build_epochs_distribution(summary_df: pd.DataFrame) -> str:
    """Build epochs distribution statistics."""
    if summary_df.empty or 'n_epochs' not in summary_df.columns:
        return '<p class="no-data">No epoch data available</p>'
    
    # Compute statistics per marker type
    stats_by_type = []
    for marker_type in ['evoked', 'state', 'sleep']:
        type_df = summary_df[summary_df['marker_type'] == marker_type]
        if not type_df.empty:
            stats_by_type.append({
                'type': marker_type,
                'min': int(type_df['n_epochs'].min()),
                'max': int(type_df['n_epochs'].max()),
                'mean': round(type_df['n_epochs'].mean(), 1),
                'median': round(type_df['n_epochs'].median(), 1),
                'std': round(type_df['n_epochs'].std(), 2),
                'count': len(type_df),
            })
    
    if not stats_by_type:
        return '<p class="no-data">No epoch statistics available</p>'
    
    rows = []
    for stat in stats_by_type:
        rows.append(f"""
        <tr>
            <td><strong>{stat['type'].capitalize()}</strong></td>
            <td>{stat['count']}</td>
            <td>{stat['min']}</td>
            <td>{stat['max']}</td>
            <td>{stat['mean']}</td>
            <td>{stat['median']}</td>
            <td>{stat['std']}</td>
        </tr>
        """)
    
    return f"""
    <table>
        <thead>
            <tr>
                <th>Marker Type</th>
                <th>N Probes</th>
                <th>Min Epochs</th>
                <th>Max Epochs</th>
                <th>Mean Epochs</th>
                <th>Median Epochs</th>
                <th>Std</th>
            </tr>
        </thead>
        <tbody>
            {''.join(rows)}
        </tbody>
    </table>
    """


def _build_discarded_table(discarded_probes: List[Dict]) -> str:
    """Build the discarded probes table."""
    if not discarded_probes:
        return '<p class="no-data">No probes were discarded 🎉</p>'
    
    rows = []
    for probe in discarded_probes:
        rows.append(f"""
        <tr>
            <td>{probe.get('subject', 'N/A')}</td>
            <td>{probe.get('session', 'N/A')}</td>
            <td>{probe.get('task', 'N/A')}</td>
            <td>{probe.get('probe_number', 'N/A')}</td>
            <td>{probe.get('epoch_type', 'N/A')}</td>
            <td>{probe.get('n_epochs', 'N/A')}</td>
            <td><span class="pill pill-warning">{probe.get('reason', 'Unknown')}</span></td>
        </tr>
        """)
    
    return f"""
    <div class="scroll-table">
        <table>
            <thead>
                <tr>
                    <th>Subject</th>
                    <th>Session</th>
                    <th>Task</th>
                    <th>Probe #</th>
                    <th>Epoch Type</th>
                    <th>N Epochs</th>
                    <th>Reason</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
    </div>
    """


def _build_errors_section(errors: List[Dict]) -> str:
    """Build the errors section."""
    if not errors:
        return ""
    
    error_boxes = []
    for error in errors:
        error_boxes.append(f"""
        <div class="error-box">
            <strong>{error.get('context', 'Unknown Context')}</strong><br>
            <code>{error.get('message', 'No message')}</code>
        </div>
        """)
    
    return f"""
    <div class="section">
        <h2>❌ Errors ({len(errors)})</h2>
        {''.join(error_boxes)}
    </div>
    """


def _build_warnings_section(warnings: List[Dict]) -> str:
    """Build the warnings section."""
    if not warnings:
        return ""
    
    warning_boxes = []
    for warning in warnings:
        warning_boxes.append(f"""
        <div class="warning-box">
            <strong>{warning.get('context', 'Unknown Context')}</strong><br>
            {warning.get('message', 'No message')}
        </div>
        """)
    
    return f"""
    <div class="section">
        <h2 class="collapsible" onclick="toggleContent(this)">⚠️ Warnings ({len(warnings)})</h2>
        <div class="content">
            {''.join(warning_boxes)}
        </div>
    </div>
    """


def _build_missing_epoch_types_section(missing_epoch_types: List[Dict]) -> str:
    """Build the missing epoch types section."""
    if not missing_epoch_types:
        return ""
    
    # Group by subject/session/task to show complete picture
    grouped = {}
    for item in missing_epoch_types:
        key = (item['subject'], item['session'], item['task'])
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(item)
    
    # Build table rows
    rows = []
    for (subject, session, task), items in sorted(grouped.items()):
        session_str = session if session else "N/A"
        missing_types = ", ".join(sorted(set(item['epoch_type'] for item in items)))
        reasons = "<br>".join(f"<small>{item['epoch_type']}: {item['reason']}</small>" for item in items)
        
        rows.append(f"""
        <tr>
            <td>{subject}</td>
            <td>{session_str}</td>
            <td>{task}</td>
            <td><span class="badge badge-warning">{missing_types}</span></td>
            <td>{reasons}</td>
        </tr>
        """)
    
    return f"""
    <div class="section">
        <h2 class="collapsible" onclick="toggleContent(this)">📂 Missing Epoch Types ({len(missing_epoch_types)})</h2>
        <div class="content">
            <p class="info-text">
                These subjects are missing one or more epoch types (evoked, state, sleep). 
                This could indicate incomplete preprocessing or missing data files.
            </p>
            <table>
                <thead>
                    <tr>
                        <th>Subject</th>
                        <th>Session</th>
                        <th>Task</th>
                        <th>Missing Types</th>
                        <th>Reason</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows)}
                </tbody>
            </table>
        </div>
    </div>
    """


def _build_data_issues_section(
    epoch_mismatches: List[Dict],
    skipped_markers: List[Dict],
) -> str:
    """Build the data issues section for epoch mismatches and skipped markers."""
    total_issues = len(epoch_mismatches) + len(skipped_markers)
    
    if total_issues == 0:
        return ""
    
    # Epoch mismatches table
    mismatch_html = ""
    if epoch_mismatches:
        rows = []
        for m in epoch_mismatches:
            rows.append(f"""
            <tr>
                <td>{m.get('context', '-')}</td>
                <td>{m.get('marker_type', '-')}</td>
                <td class="text-center">{m.get('h5_epochs', '-')}</td>
                <td class="text-center">{m.get('events_epochs', '-')}</td>
                <td class="text-center {'error-text' if m.get('difference', 0) != 0 else ''}">{m.get('difference', '-')}</td>
            </tr>
            """)
        
        mismatch_html = f"""
        <h3>🔄 Epoch Count Mismatches ({len(epoch_mismatches)})</h3>
        <p class="description">
            These files have different epoch counts between the H5 markers and events.tsv.
            This indicates the H5 file was generated from a different epochs file.
            <strong>Solution:</strong> Re-run Junifer marker extraction with the correct epochs file,
            or regenerate the events.tsv from the same source.
        </p>
        <table>
            <thead>
                <tr>
                    <th>Context</th>
                    <th>Marker Type</th>
                    <th>H5 Epochs</th>
                    <th>Events Epochs</th>
                    <th>Difference</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
        """
    
    # Skipped markers table
    skipped_html = ""
    if skipped_markers:
        rows = []
        for s in skipped_markers:
            rows.append(f"""
            <tr>
                <td>{s.get('context', '-')}</td>
                <td><code>{s.get('marker_name', '-')}</code></td>
                <td class="text-center">{s.get('expected_length', '-')}</td>
                <td class="text-center">{s.get('actual_length', '-')}</td>
                <td>{s.get('reason', '-')}</td>
            </tr>
            """)
        
        skipped_html = f"""
        <h3>⏭️ Skipped Markers ({len(skipped_markers)})</h3>
        <p class="description">
            These markers were skipped because their data length didn't match the expected length.
            This is usually caused by epoch count mismatches.
        </p>
        <table>
            <thead>
                <tr>
                    <th>Context</th>
                    <th>Marker</th>
                    <th>Expected Length</th>
                    <th>Actual Length</th>
                    <th>Reason</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
        """
    
    return f"""
    <div class="section data-issues">
        <h2 class="collapsible" onclick="toggleContent(this)">🔧 Data Issues ({total_issues})</h2>
        <div class="content">
            {mismatch_html}
            {skipped_html}
        </div>
    </div>
    """


def _build_config_section(config: Dict) -> str:
    """Build the configuration summary section."""
    if not config:
        return '<p class="no-data">No configuration available</p>'
    
    def format_value(v):
        if isinstance(v, list):
            return ", ".join(str(x) for x in v)
        elif isinstance(v, dict):
            return str(v)
        return str(v)
    
    sections = []
    
    for section_name, section_data in config.items():
        if isinstance(section_data, dict):
            items = []
            for key, value in section_data.items():
                items.append(f"""
                <div class="config-item">
                    <span class="config-key">{key}</span>
                    <span class="config-value">{format_value(value)}</span>
                </div>
                """)
            sections.append(f"""
            <div>
                <h3>{section_name}</h3>
                {''.join(items)}
            </div>
            """)
        else:
            sections.append(f"""
            <div class="config-item">
                <span class="config-key">{section_name}</span>
                <span class="config-value">{format_value(section_data)}</span>
            </div>
            """)
    
    return f'<div class="config-grid">{"".join(sections)}</div>'


class AggregationStats:
    """
    Class to track statistics during the aggregation pipeline.
    
    Usage:
        stats = AggregationStats()
        stats.add_discarded_probe(subject='03', session='a', task='sart', ...)
        stats.add_error(context='Loading H5', message='File not found')
        stats.add_epoch_mismatch(context='sub-03 ses-a evoked', h5_epochs=462, events_epochs=423)
        stats.add_skipped_marker(context='sub-03 ses-a evoked', marker='power_delta', ...)
        stats.to_dict()  # Get all stats for report generation
    """
    
    def __init__(self):
        self.discarded_probes: List[Dict] = []
        self.errors: List[Dict] = []
        self.warnings: List[Dict] = []
        self.epoch_mismatches: List[Dict] = []
        self.skipped_markers: List[Dict] = []
        self.missing_epoch_types: List[Dict] = []
        self.per_subject_stats: Dict[str, Dict] = {}
    
    def add_discarded_probe(
        self,
        subject: str,
        session: Optional[str],
        task: str,
        probe_number: int,
        epoch_type: str,
        n_epochs: int,
        reason: str,
    ):
        """Record a discarded probe."""
        self.discarded_probes.append({
            'subject': subject,
            'session': session or '',
            'task': task,
            'probe_number': probe_number,
            'epoch_type': epoch_type,
            'n_epochs': n_epochs,
            'reason': reason,
        })
    
    def add_error(self, context: str, message: str):
        """Record an error."""
        self.errors.append({
            'context': context,
            'message': str(message),
        })
    
    def add_warning(self, context: str, message: str):
        """Record a warning."""
        self.warnings.append({
            'context': context,
            'message': str(message),
        })
    
    def add_epoch_mismatch(
        self,
        context: str,
        h5_epochs: int,
        events_epochs: int,
        marker_type: str = "",
    ):
        """
        Record an epoch count mismatch between H5 file and events.tsv.
        
        This indicates a data integrity issue where the H5 markers were
        computed with a different epochs file than the events.tsv.
        """
        self.epoch_mismatches.append({
            'context': context,
            'h5_epochs': h5_epochs,
            'events_epochs': events_epochs,
            'marker_type': marker_type,
            'difference': h5_epochs - events_epochs,
        })
    
    def add_skipped_marker(
        self,
        context: str,
        marker_name: str,
        expected_length: int,
        actual_length: int,
        reason: str = "length mismatch",
    ):
        """
        Record a marker that was skipped during aggregation.
        
        This typically happens when a marker has a different number of epochs
        than expected from the events.tsv.
        """
        self.skipped_markers.append({
            'context': context,
            'marker_name': marker_name,
            'expected_length': expected_length,
            'actual_length': actual_length,
            'reason': reason,
        })
    
    def add_missing_epoch_type(
        self,
        subject: str,
        session: Optional[str],
        task: str,
        epoch_type: str,
        reason: str,
    ):
        """
        Record a missing epoch type for a subject/session/task.
        
        This helps identify incomplete datasets where some epoch types
        (evoked, state, sleep) are missing.
        """
        self.missing_epoch_types.append({
            'subject': subject,
            'session': session or '',
            'task': task,
            'epoch_type': epoch_type,
            'reason': reason,
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """Return all statistics as a dictionary."""
        return {
            'discarded_probes': self.discarded_probes,
            'errors': self.errors,
            'warnings': self.warnings,
            'epoch_mismatches': self.epoch_mismatches,
            'skipped_markers': self.skipped_markers,
            'missing_epoch_types': self.missing_epoch_types,
            'per_subject_stats': self.per_subject_stats,
        }


if __name__ == "__main__":
    # Test with dummy data
    test_df = pd.DataFrame({
        'subject': ['03', '03', '04', '04'],
        'session': ['a', 'a', 'a', 'b'],
        'task': ['sartauditiva', 'sartvisual', 'sartauditiva', 'sartvisual'],
        'probe_number': [1, 2, 1, 2],
        'label': ['onTask', 'offTask', 'onTask', 'unknown'],
        'marker_type': ['evoked', 'state', 'sleep', 'evoked'],
        'n_epochs': [5, 8, 1, 4],
        'n_markers': [10, 10, 10, 10],
        'output_path': ['path1', 'path2', 'path3', 'path4'],
    })
    
    test_stats = {
        'discarded_probes': [
            {'subject': '03', 'session': 'b', 'task': 'sartauditiva', 
             'probe_number': 5, 'epoch_type': 'evoked', 'n_epochs': 2, 
             'reason': 'Insufficient epochs (2 < 3)'},
        ],
        'errors': [],
        'warnings': [
            {'context': 'sub-05 ses-a sartvisual', 'message': 'H5 file not found'},
        ],
    }
    
    test_config = {
        'project': {'derivatives_root': '/path/to/derivatives'},
        'trial_selection': {'evoked_distance_max': 5},
    }
    
    generate_html_report(
        summary_df=test_df,
        stats=test_stats,
        config=test_config,
        output_path='/tmp/test_qa_report.html',
    )
    print("Test report generated at /tmp/test_qa_report.html")
