import yaml
import sys
from pathlib import Path
import re
import pandas as pd

def main():
    config_file = "Statistics/config.yaml"
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    features_root = config['project']['features_root']
    selected_markers = config.get('selected_markers', {})
    feature_families = config.get('feature_families', {})
    
    features_path = Path(features_root)
    csv_files = []
    csv_files.extend(list(features_path.rglob("sub-*_task-*_desc-probe-*_aggMarkers.csv")))
    csv_files.extend(list(features_path.rglob("sub-*_task-*_probe-*_*.csv")))
    
    unique_files = {}
    for f in csv_files:
        if f.name.endswith(".csv") and not f.name.startswith("summary") and not f.name.startswith("qa"):
            unique_files[str(f)] = f
    csv_files = list(unique_files.values())
    
    if not csv_files:
        print(f"No CSV files found in {features_root}", file=sys.stderr)
        sys.exit(1)
        
    sample_files_by_type = {}
    for f in csv_files:
        m1 = re.search(r'_desc-probe-\d+_(\w+)_aggMarkers\.csv$', f.name)
        m2 = re.search(r'_probe-\d+_\w+_(\w+)(?:_connectivity)?\.csv$', f.name)
        mtype = m1.group(1) if m1 else (m2.group(1) if m2 else None)
        if mtype and mtype not in sample_files_by_type:
            sample_files_by_type[mtype] = f
            
    markers_by_type = {}
    for mtype, sample_file in sample_files_by_type.items():
        df = pd.read_csv(sample_file, nrows=0)
        cols = df.columns.tolist()
        # Find columns ending in _trimmean
        found_markers = [c.replace('_trimmean', '') for c in cols if c.endswith('_trimmean')]
        # If no _trimmean but has 'marker' column (old format)
        if not found_markers and 'marker' in cols:
             markers_by_type[mtype] = sorted(pd.read_csv(sample_file, usecols=['marker'])['marker'].unique().tolist())
        else:
             markers_by_type[mtype] = sorted(found_markers)
    
    n_markers = 0
    for marker_type, family_list in selected_markers.items():
        if marker_type not in markers_by_type: continue
        available = markers_by_type[marker_type]
        if isinstance(family_list, list) and len(family_list) == 1 and str(family_list[0]).lower() == 'all':
            n_markers += len(available)
            continue
        fragments = []
        null_family = False
        for family_name in family_list:
            if family_name not in feature_families: continue
            ffrags = feature_families[family_name]
            if ffrags is None:
                null_family = True
                break
            fragments.extend(ffrags)
        if null_family: n_markers += len(available)
        else:
            matched = [mk for mk in available if any(frag in mk for frag in fragments)]
            n_markers += len(matched)
    print(n_markers)

if __name__ == "__main__":
    main()
