"""
QA Report Generator for Data Harmonization Pipeline.

Generates a summary table of all harmonized files with status and failure reasons.
Similar to the EEG preprocessing QA report system.
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import fcntl
except Exception:  # pragma: no cover
    fcntl = None


class HarmonizationQAReport:
    """
    QA report generator for data harmonization pipeline.
    
    Stores processing results and generates summary reports in multiple formats
    (JSON, CSV, HTML).
    
    Parameters
    ----------
    output_dir : str
        Directory where QA reports will be saved
    report_name : str
        Base name for report files (without extension)
    """
    
    def __init__(
        self, 
        output_dir: str, 
        report_name: str = "harmonization_qa_report",
        load_existing: bool = True
    ) -> None:
        self.output_dir = output_dir
        self.report_name = report_name
        self.results: List[Dict[str, Any]] = []
        self.start_time = datetime.now()
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Load existing results from JSON if present (for cumulative reports)
        if load_existing:
            self._load_existing_results()
    
    def _load_existing_results(self) -> None:
        """Load existing results from JSON file if it exists."""
        json_path = os.path.join(self.output_dir, f"{self.report_name}.json")
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                if 'results' in data and isinstance(data['results'], list):
                    self.results = data['results']
            except (json.JSONDecodeError, IOError):
                pass

    def _result_key(self, result: Dict[str, Any]) -> tuple:
        """Create a stable key for identifying unique entries in the report."""
        subj = str(result.get('subject', ''))
        sess = str(result.get('session', ''))
        task = str(result.get('task', ''))
        return subj, sess, task

    def _merge_results(self, base: List[Dict[str, Any]], incoming: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge two result lists, deduplicating by (subject, session, task)."""
        merged: Dict[tuple, Dict[str, Any]] = {}
        for r in base:
            if isinstance(r, dict):
                merged[self._result_key(r)] = r
        for r in incoming:
            if isinstance(r, dict):
                merged[self._result_key(r)] = r
        return list(merged.values())

    def _load_results_from_path(self, json_path: str) -> List[Dict[str, Any]]:
        """Load results list from a JSON file path, returning [] on failure."""
        if not os.path.exists(json_path):
            return []
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            if 'results' in data and isinstance(data['results'], list):
                return data['results']
        except (json.JSONDecodeError, IOError):
            return []
        return []

    def _acquire_lock(self, lock_path: str):
        """Acquire an exclusive lock on the given path (POSIX only)."""
        lock_fh = open(lock_path, 'a+')
        if fcntl is not None:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
        return lock_fh

    def _release_lock(self, lock_fh) -> None:
        """Release a previously acquired lock."""
        try:
            if lock_fh is not None and fcntl is not None:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
        finally:
            try:
                if lock_fh is not None:
                    lock_fh.close()
            except Exception:
                pass
    
    def _convert_to_serializable(self, obj: Any) -> Any:
        """Convert numpy types and other non-serializable objects."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        if isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, tuple):
            return list(obj)
        if isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._convert_to_serializable(i) for i in obj]
        if isinstance(obj, datetime):
            return obj.isoformat()
        return obj
    
    def add_result(self, result: Dict[str, Any]) -> None:
        """
        Add or update a processing result in the report.
        
        If a result with the same subject/session/task already exists,
        it will be updated. Otherwise, a new entry is added.
        
        Parameters
        ----------
        result : dict
            Processing result with keys:
            - subject: str
            - session: str
            - task: str
            - status: str ('success', 'skipped', 'error')
            - message: str or None (error/skip reason)
            - Additional optional fields (n_channels, sfreq, etc.)
        """
        result = self._convert_to_serializable(result)
        result['timestamp'] = datetime.now().isoformat()
        
        # Check if entry already exists (same subject/session/task)
        subj, sess, task = self._result_key(result)
        
        for i, existing in enumerate(self.results):
            if self._result_key(existing) == (subj, sess, task):
                # Update existing entry
                self.results[i] = result
                return
        
        # No existing entry found, append new one
        self.results.append(result)
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Compute summary statistics from all results.
        
        Returns
        -------
        dict
            Summary statistics including counts by status
        """
        total = len(self.results)
        success = sum(1 for r in self.results if r.get('status') == 'success')
        partial = sum(1 for r in self.results if r.get('status') == 'partial')
        skipped = sum(1 for r in self.results if r.get('status') == 'skipped')
        errors = sum(1 for r in self.results if r.get('status') == 'error')
        
        return {
            'total_processed': total,
            'success': success,
            'partial': partial,
            'skipped': skipped,
            'errors': errors,
            'success_rate': (success + partial) / total * 100 if total > 0 else 0,
            'start_time': self.start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert results to pandas DataFrame.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with all processing results
        """
        if not self.results:
            return pd.DataFrame()
        
        # Define column order
        columns = ['subject', 'session', 'task', 'status', 'message']
        
        # Add any additional columns found in results
        all_keys = set()
        for r in self.results:
            all_keys.update(r.keys())
        extra_cols = sorted(all_keys - set(columns) - {'timestamp'})
        columns.extend(extra_cols)
        columns.append('timestamp')
        
        df = pd.DataFrame(self.results)
        
        # Reorder columns (only include those that exist)
        existing_cols = [c for c in columns if c in df.columns]
        df = df[existing_cols]
        
        return df
    
    def save_json(self, path: Optional[str] = None) -> str:
        """
        Save results to JSON file.
        
        Parameters
        ----------
        path : str, optional
            Output path. If None, uses default naming.
            
        Returns
        -------
        str
            Path to saved file
        """
        if path is None:
            path = os.path.join(self.output_dir, f"{self.report_name}.json")

        lock_path = f"{path}.lock"
        lock_fh = None
        try:
            lock_fh = self._acquire_lock(lock_path)

            disk_results = self._load_results_from_path(path)
            before_n = len(self.results)
            disk_n = len(disk_results)
            self.results = self._merge_results(disk_results, self.results)
            after_n = len(self.results)
            if after_n < max(before_n, disk_n):
                print(
                    f"[QA REPORT][WARN] Results count decreased unexpectedly on merge: "
                    f"disk={disk_n}, mem={before_n}, merged={after_n}"
                )

            report_data = {
                'summary': self.get_summary_stats(),
                'results': self.results,
            }

            tmp_path = f"{path}.tmp"
            with open(tmp_path, 'w') as f:
                json.dump(report_data, f, indent=2)
            os.replace(tmp_path, path)
        finally:
            self._release_lock(lock_fh)

        return path
    
    def save_csv(self, path: Optional[str] = None) -> str:
        """
        Save results to CSV file.
        
        Parameters
        ----------
        path : str, optional
            Output path. If None, uses default naming.
            
        Returns
        -------
        str
            Path to saved file
        """
        if path is None:
            path = os.path.join(self.output_dir, f"{self.report_name}.csv")
        
        df = self.to_dataframe()
        tmp_path = f"{path}.tmp"
        df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, path)
        
        return path
    
    def save_html(self, path: Optional[str] = None) -> str:
        """
        Save results to HTML report with styled table and visual summary.
        
        Parameters
        ----------
        path : str, optional
            Output path. If None, uses default naming.
            
        Returns
        -------
        str
            Path to saved file
        """
        if path is None:
            path = os.path.join(self.output_dir, f"{self.report_name}.html")
        
        df = self.to_dataframe()
        stats = self.get_summary_stats()
        
        # Calculate additional metrics
        total = stats['total_processed']
        partial_rate = (stats['partial'] / total * 100) if total > 0 else 0
        success_only_rate = (stats['success'] / total * 100) if total > 0 else 0
        skipped_rate = (stats['skipped'] / total * 100) if total > 0 else 0
        error_rate = (stats['errors'] / total * 100) if total > 0 else 0
        
        # Count by task
        task_counts = {}
        if not df.empty and 'task' in df.columns:
            task_counts = df.groupby('task')['status'].value_counts().unstack(fill_value=0).to_dict('index')
        
        # Build HTML with modern styling
        html_parts = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "<meta charset='utf-8'>",
            "<title>Data Harmonization QA Report</title>",
            "<style>",
            "body { font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 20px; background: #f0f4f8; }",
            ".container { max-width: 1400px; margin: 0 auto; }",
            "h1 { color: #1a365d; border-bottom: 3px solid #4299e1; padding-bottom: 10px; }",
            "h2 { color: #2d3748; margin-top: 30px; }",
            ".summary-cards { display: flex; gap: 20px; flex-wrap: wrap; margin: 20px 0; }",
            ".card { background: white; border-radius: 12px; padding: 25px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); min-width: 160px; text-align: center; transition: transform 0.2s; }",
            ".card:hover { transform: translateY(-5px); }",
            ".card-value { font-size: 42px; font-weight: bold; }",
            ".card-label { color: #718096; font-size: 14px; margin-top: 8px; text-transform: uppercase; letter-spacing: 1px; }",
            ".success { color: #38a169; }",
            ".error { color: #e53e3e; }",
            ".skipped { color: #d69e2e; }",
            ".neutral { color: #4299e1; }",
            ".progress-container { background: white; border-radius: 12px; padding: 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); margin: 20px 0; }",
            ".progress-bar { background: #e2e8f0; border-radius: 10px; height: 40px; overflow: hidden; display: flex; }",
            ".progress-segment { height: 100%; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 14px; transition: width 0.5s; }",
            ".progress-success { background: linear-gradient(90deg, #38a169, #48bb78); }",
            ".progress-skipped { background: linear-gradient(90deg, #d69e2e, #ecc94b); }",
            ".progress-error { background: linear-gradient(90deg, #e53e3e, #fc8181); }",
            ".legend { display: flex; gap: 20px; margin-top: 15px; justify-content: center; }",
            ".legend-item { display: flex; align-items: center; gap: 8px; font-size: 14px; }",
            ".legend-color { width: 16px; height: 16px; border-radius: 4px; }",
            ".metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }",
            ".metric-item { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.08); }",
            ".metric-value { font-size: 28px; font-weight: bold; color: #2d3748; }",
            ".metric-label { color: #718096; font-size: 13px; margin-top: 5px; }",
            "table { width: 100%; border-collapse: collapse; background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 15px rgba(0,0,0,0.1); margin: 20px 0; }",
            "th { background: linear-gradient(135deg, #4299e1, #3182ce); color: white; padding: 15px 10px; text-align: left; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; }",
            "td { padding: 12px 10px; border-bottom: 1px solid #e2e8f0; font-size: 13px; }",
            "tr:hover { background: #f7fafc; }",
            ".status-badge { padding: 5px 12px; border-radius: 20px; font-weight: bold; font-size: 11px; text-transform: uppercase; }",
            ".badge-success { background: #c6f6d5; color: #22543d; }",
            ".badge-error { background: #fed7d7; color: #742a2a; }",
            ".badge-skipped { background: #fefcbf; color: #744210; }",
            ".badge-partial { background: #bee3f8; color: #2a4365; }",
            ".task-badge { display: inline-block; padding: 3px 10px; border-radius: 15px; font-size: 11px; font-weight: 600; }",
            ".task-sartauditiva { background: #e9d8fd; color: #553c9a; }",
            ".task-sartvisual { background: #bee3f8; color: #2a4365; }",
            ".message-cell { max-width: 350px; word-wrap: break-word; font-size: 12px; color: #4a5568; }",
            ".error-section { background: #fff5f5; border: 1px solid #feb2b2; border-radius: 12px; padding: 20px; margin: 20px 0; }",
            ".error-section h3 { color: #c53030; margin-top: 0; }",
            ".error-list { list-style: none; padding: 0; }",
            ".error-list li { padding: 10px; background: white; margin: 8px 0; border-radius: 8px; border-left: 4px solid #e53e3e; }",
            ".skipped-section { background: #fffff0; border: 1px solid #f6e05e; border-radius: 12px; padding: 20px; margin: 20px 0; }",
            ".skipped-section h3 { color: #b7791f; margin-top: 0; }",
            ".skipped-list li { border-left-color: #d69e2e; }",
            ".timestamp { color: #a0aec0; font-size: 12px; }",
            ".data-info { display: flex; gap: 15px; flex-wrap: wrap; margin: 10px 0; }",
            ".data-tag { background: #edf2f7; padding: 5px 12px; border-radius: 15px; font-size: 12px; color: #4a5568; }",
            "</style>",
            "</head>",
            "<body>",
            "<div class='container'>",
            "<h1>Data Harmonization QA Report</h1>",
            f"<p class='timestamp'>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
            "",
            "<!-- Summary Cards -->",
            "<div class='summary-cards'>",
            f"<div class='card'><div class='card-value neutral'>{stats['total_processed']}</div><div class='card-label'>Total Files</div></div>",
            f"<div class='card'><div class='card-value success'>{stats['success']}</div><div class='card-label'>Success</div></div>",
            f"<div class='card'><div class='card-value neutral'>{stats['partial']}</div><div class='card-label'>Partial</div></div>",
            f"<div class='card'><div class='card-value skipped'>{stats['skipped']}</div><div class='card-label'>Skipped</div></div>",
            f"<div class='card'><div class='card-value error'>{stats['errors']}</div><div class='card-label'>Errors</div></div>",
            "</div>",
            "",
            "<!-- Progress Bar -->",
            "<div class='progress-container'>",
            "<h2 style='margin-top:0'>Processing Status</h2>",
            "<div class='progress-bar'>",
        ]
        
        # Add progress segments
        if success_only_rate > 0:
            html_parts.append(f"<div class='progress-segment progress-success' style='width:{success_only_rate}%'>{stats['success']}</div>")
        if partial_rate > 0:
            html_parts.append(f"<div class='progress-segment' style='width:{partial_rate}%;background:linear-gradient(90deg,#3182ce,#63b3ed)'>{stats['partial']}</div>")
        if skipped_rate > 0:
            html_parts.append(f"<div class='progress-segment progress-skipped' style='width:{skipped_rate}%'>{stats['skipped']}</div>")
        if error_rate > 0:
            html_parts.append(f"<div class='progress-segment progress-error' style='width:{error_rate}%'>{stats['errors']}</div>")
        
        html_parts.extend([
            "</div>",
            "<div class='legend'>",
            "<div class='legend-item'><div class='legend-color' style='background:#38a169'></div>Success</div>",
            "<div class='legend-item'><div class='legend-color' style='background:#3182ce'></div>Partial</div>",
            "<div class='legend-item'><div class='legend-color' style='background:#d69e2e'></div>Skipped</div>",
            "<div class='legend-item'><div class='legend-color' style='background:#e53e3e'></div>Error</div>",
            "</div>",
            "</div>",
        ])
        
        # Data metrics section
        if not df.empty:
            html_parts.append("<h2>Data Metrics</h2>")
            html_parts.append("<div class='metrics-grid'>")
            
            # Calculate averages for successful harmonizations
            success_df = df[df['status'] == 'success']
            if not success_df.empty:
                if 'n_eeg_channels' in success_df.columns:
                    avg_channels = success_df['n_eeg_channels'].dropna().mean()
                    html_parts.append(f"<div class='metric-item'><div class='metric-value'>{avg_channels:.0f}</div><div class='metric-label'>Avg EEG Channels</div></div>")
                if 'sfreq_final' in success_df.columns:
                    avg_sfreq = success_df['sfreq_final'].dropna().mean()
                    html_parts.append(f"<div class='metric-item'><div class='metric-value'>{avg_sfreq:.0f} Hz</div><div class='metric-label'>Final Sample Rate</div></div>")
                if 'n_annotations' in success_df.columns:
                    avg_annot = success_df['n_annotations'].dropna().mean()
                    html_parts.append(f"<div class='metric-item'><div class='metric-value'>{avg_annot:.0f}</div><div class='metric-label'>Avg Annotations</div></div>")
                if 'has_gaze' in success_df.columns:
                    gaze_count = success_df['has_gaze'].sum()
                    html_parts.append(f"<div class='metric-item'><div class='metric-value'>{gaze_count}</div><div class='metric-label'>With Gaze Data</div></div>")
                if 'marker_correction_applied' in success_df.columns:
                    marker_count = success_df['marker_correction_applied'].sum()
                    html_parts.append(f"<div class='metric-item'><div class='metric-value'>{marker_count}</div><div class='metric-label'>Markers Corrected</div></div>")
            
            html_parts.append("</div>")

        # Subject-level summary table (subjects x sessions/tasks)
        html_parts.append("<h2>Subject Summary</h2>")
        if (not df.empty and
            all(col in df.columns for col in ['subject', 'session', 'task', 'status'])):
            try:
                # Classify subjects: completely good / partially bad / completely bad
                # "success" and "partial" both count as usable data
                subj_status = df.groupby('subject')['status'].apply(list)
                n_complete_good = 0
                n_partial_bad = 0
                n_complete_bad = 0
                for _subj, statuses in subj_status.items():
                    has_usable = any(s in ('success', 'partial') for s in statuses)
                    has_non_usable = any(s not in ('success', 'partial') for s in statuses)
                    if has_usable and not has_non_usable:
                        n_complete_good += 1
                    elif not has_usable and has_non_usable:
                        n_complete_bad += 1
                    elif has_usable and has_non_usable:
                        n_partial_bad += 1

                html_parts.append("<div class='summary-cards'>")
                html_parts.append(
                    f"<div class='card'><div class='card-value success'>{n_complete_good}</div><div class='card-label'>Subjects Completely OK</div></div>"
                )
                html_parts.append(
                    f"<div class='card'><div class='card-value neutral'>{n_partial_bad}</div><div class='card-label'>Subjects Partially Problematic</div></div>"
                )
                html_parts.append(
                    f"<div class='card'><div class='card-value error'>{n_complete_bad}</div><div class='card-label'>Subjects Completely Problematic</div></div>"
                )
                html_parts.append("</div>")

                summary = df.pivot_table(
                    index='subject',
                    columns=['session', 'task'],
                    values='status',
                    aggfunc='first'
                )
                # Ensure deterministic order
                summary = summary.sort_index(axis=0)
                sessions = []
                tasks_by_session = {}
                if summary.columns.nlevels == 2:
                    for sess, task in summary.columns:
                        if sess not in tasks_by_session:
                            tasks_by_session[sess] = []
                            sessions.append(sess)
                        if task not in tasks_by_session[sess]:
                            tasks_by_session[sess].append(task)

                html_parts.append("<table>")
                # Header row: sessions
                html_parts.append("<tr>")
                html_parts.append("<th rowspan='2'>Subject</th>")
                for sess in sessions:
                    n_tasks = len(tasks_by_session.get(sess, []))
                    html_parts.append(
                        f"<th colspan='{n_tasks}'>ses-{sess}</th>"
                    )
                html_parts.append("</tr>")

                # Header row: tasks
                html_parts.append("<tr>")
                for sess in sessions:
                    for task in tasks_by_session.get(sess, []):
                        html_parts.append(f"<th>{task}</th>")
                html_parts.append("</tr>")

                # Rows per subject
                for subj in summary.index:
                    html_parts.append("<tr>")
                    html_parts.append(f"<td>sub-{subj}</td>")
                    for sess in sessions:
                        for task in tasks_by_session.get(sess, []):
                            val = None
                            if (sess, task) in summary.columns:
                                try:
                                    val = summary.loc[subj, (sess, task)]
                                except Exception:
                                    val = None
                            if pd.isna(val):
                                val = None

                            cell_html = ""
                            if isinstance(val, str) and val:
                                if val in ['success', 'error', 'skipped', 'partial']:
                                    badge_class = f"badge-{val}"
                                    cell_html = (
                                        f"<span class='status-badge {badge_class}'>{val}</span>"
                                    )
                                else:
                                    cell_html = val
                            html_parts.append(f"<td>{cell_html}</td>")
                    html_parts.append("</tr>")

                html_parts.append("</table>")
            except Exception:
                html_parts.append("<p>Could not compute subject summary.</p>")
        else:
            html_parts.append("<p>No subject summary available.</p>")

        # Detailed table
        html_parts.append("<h2>Detailed Results</h2>")
        if not df.empty:
            # Select key columns
            display_cols = ['subject', 'session', 'task', 'status', 'message', 
                          'n_eeg_channels', 'sfreq_final', 'n_annotations', 'has_gaze']
            display_cols = [c for c in display_cols if c in df.columns]
            
            html_parts.append("<table>")
            html_parts.append("<tr>")
            col_labels = {
                'subject': 'Subject', 'session': 'Session', 'task': 'Task',
                'status': 'Status', 'message': 'Message', 'n_eeg_channels': 'Channels',
                'sfreq_final': 'Sfreq', 'n_annotations': 'Annotations', 'has_gaze': 'Gaze'
            }
            for col in display_cols:
                html_parts.append(f"<th>{col_labels.get(col, col)}</th>")
            html_parts.append("</tr>")
            
            for _, row in df.iterrows():
                html_parts.append("<tr>")
                for col in display_cols:
                    val = row.get(col, '')
                    if pd.isna(val):
                        val = ''
                    
                    # Format specific columns
                    if col == 'status':
                        badge_class = f"badge-{val}" if val in ['success', 'error', 'skipped', 'partial'] else ''
                        val = f"<span class='status-badge {badge_class}'>{val}</span>"
                    elif col == 'task':
                        task_class = f"task-{val}" if val else ''
                        val = f"<span class='task-badge {task_class}'>{val}</span>"
                    elif col == 'message':
                        val = f"<span class='message-cell'>{val}</span>" if val else ''
                    elif col == 'has_gaze':
                        val = "Yes" if val else "No"
                    elif col in ['sfreq_final', 'n_eeg_channels', 'n_annotations']:
                        try:
                            val = f"{float(val):.0f}" if val else ''
                        except:
                            pass
                    
                    html_parts.append(f"<td>{val}</td>")
                html_parts.append("</tr>")
            
            html_parts.append("</table>")
        else:
            html_parts.append("<p>No results to display.</p>")
        
        # Partial details section
        partials = [r for r in self.results if r.get('status') == 'partial']
        if partials:
            html_parts.append("<div class='progress-container' style='border-left:4px solid #3182ce'>")
            html_parts.append(f"<h3 style='color:#2a4365;margin-top:0'>Partial Matches ({len(partials)} files)</h3>")
            html_parts.append("<ul class='error-list' style='list-style:none;padding:0'>")
            for p in partials:
                subj = p.get('subject', '?')
                sess = p.get('session', '?')
                task = p.get('task', '?')
                n_matched = p.get('n_markers_matched', '?')
                n_expected = p.get('n_markers_expected', '?')
                seq_ratio = p.get('sequence_match_ratio', None)
                seq_text = f" | Sequence validated: {seq_ratio*100:.1f}%" if seq_ratio else ""
                html_parts.append(
                    f"<li style='border-left:4px solid #3182ce'><strong>sub-{subj}_ses-{sess}_task-{task}</strong><br/>"
                    f"Matched {n_matched}/{n_expected} markers{seq_text}</li>"
                )
            html_parts.append("</ul>")
            html_parts.append("</div>")
        
        # Error details section
        errors = [r for r in self.results if r.get('status') == 'error']
        if errors:
            html_parts.append("<div class='error-section'>")
            html_parts.append(f"<h3>Error Details ({len(errors)} files)</h3>")
            html_parts.append("<ul class='error-list'>")
            for e in errors:
                subj = e.get('subject', '?')
                sess = e.get('session', '?')
                task = e.get('task', '?')
                msg = e.get('message', 'Unknown error')
                html_parts.append(
                    f"<li><strong>sub-{subj}_ses-{sess}_task-{task}</strong><br/>{msg}</li>"
                )
            html_parts.append("</ul>")
            html_parts.append("</div>")
        
        # Skipped details section
        skipped = [r for r in self.results if r.get('status') == 'skipped']
        if skipped:
            html_parts.append("<div class='skipped-section'>")
            html_parts.append(f"<h3>Skipped Files ({len(skipped)} files)</h3>")
            html_parts.append("<ul class='error-list skipped-list'>")
            for s in skipped:
                subj = s.get('subject', '?')
                sess = s.get('session', '?')
                task = s.get('task', '?')
                msg = s.get('message', 'Unknown reason')
                html_parts.append(
                    f"<li><strong>sub-{subj}_ses-{sess}_task-{task}</strong><br/>{msg}</li>"
                )
            html_parts.append("</ul>")
            html_parts.append("</div>")
        
        html_parts.extend([
            "</div>",
            "</body>",
            "</html>",
        ])
        
        tmp_path = f"{path}.tmp"
        with open(tmp_path, 'w') as f:
            f.write("\n".join(html_parts))
        os.replace(tmp_path, path)
        
        return path
    
    def save_all(self) -> Dict[str, str]:
        """
        Save reports in all formats (JSON, CSV, HTML).
        
        Returns
        -------
        dict
            Dictionary with format names as keys and paths as values
        """
        json_path = os.path.join(self.output_dir, f"{self.report_name}.json")
        csv_path = os.path.join(self.output_dir, f"{self.report_name}.csv")
        html_path = os.path.join(self.output_dir, f"{self.report_name}.html")

        lock_path = f"{json_path}.lock"
        lock_fh = None
        try:
            lock_fh = self._acquire_lock(lock_path)

            disk_results = self._load_results_from_path(json_path)
            before_n = len(self.results)
            disk_n = len(disk_results)
            self.results = self._merge_results(disk_results, self.results)
            after_n = len(self.results)
            if after_n < max(before_n, disk_n):
                print(
                    f"[QA REPORT][WARN] Results count decreased unexpectedly on merge: "
                    f"disk={disk_n}, mem={before_n}, merged={after_n}"
                )

            report_data = {
                'summary': self.get_summary_stats(),
                'results': self.results,
            }

            json_tmp = f"{json_path}.tmp"
            with open(json_tmp, 'w') as f:
                json.dump(report_data, f, indent=2)
            os.replace(json_tmp, json_path)

            df = self.to_dataframe()
            csv_tmp = f"{csv_path}.tmp"
            df.to_csv(csv_tmp, index=False)
            os.replace(csv_tmp, csv_path)

            self.save_html(path=html_path)

        finally:
            self._release_lock(lock_fh)

        return {
            'json': json_path,
            'csv': csv_path,
            'html': html_path,
        }
    
    def print_summary(self) -> None:
        """Print processing summary to console."""
        stats = self.get_summary_stats()
        
        print("\n" + "=" * 60)
        print("DATA HARMONIZATION QA REPORT")
        print("=" * 60)
        print(f"  Total processed: {stats['total_processed']}")
        print(f"  Success:         {stats['success']}")
        print(f"  Skipped:         {stats['skipped']}")
        print(f"  Errors:          {stats['errors']}")
        print(f"  Success rate:    {stats['success_rate']:.1f}%")
        print("=" * 60)
        
        # Print errors
        errors = [r for r in self.results if r.get('status') == 'error']
        if errors:
            print("\nERRORS:")
            for e in errors:
                print(f"  - sub-{e.get('subject')}_ses-{e.get('session')}_"
                      f"task-{e.get('task')}: {e.get('message')}")
        
        # Print skipped
        skipped = [r for r in self.results if r.get('status') == 'skipped']
        if skipped:
            print("\nSKIPPED:")
            for s in skipped:
                print(f"  - sub-{s.get('subject')}_ses-{s.get('session')}_"
                      f"task-{s.get('task')}: {s.get('message')}")
