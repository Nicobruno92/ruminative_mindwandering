#!/usr/bin/env python3
"""
Unified elements management script.
Handles diagnosis and fixing of elements files for state, evoked, and sleep epochs.

Usage:
    python manage_elements.py diagnose [state|evoked|sleep|all]
    python manage_elements.py fix [state|evoked|sleep|all]
"""

import os
import sys
import shutil
from pathlib import Path
from datetime import datetime
from collections import Counter
import pandas as pd

# ============================================================================
# CONFIGURATION
# ============================================================================
BASE_DIR = "/network/iss/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering/junifer_markers/1.markers_h5_creation"
DERIVATIVES_DIR = "/network/iss/cenir/analyse/meeg/CYBERSART/BIDS/derivatives"

# Config for each epoch type
CONFIGS = {
    'state': {
        'elements_file': f"{BASE_DIR}/elements_state.csv",
        'desc': 'state',
        'pattern': '{subject}/eeg/{subject}_task-{task}_desc-state_epo.fif'
    },
    'evoked': {
        'elements_file': f"{BASE_DIR}/elements_evoked.csv",
        'desc': 'evoked',
        'pattern': '{subject}/eeg/{subject}_task-{task}_desc-evoked_epo.fif'
    },
    'sleep': {
        'elements_file': f"{BASE_DIR}/elements_sleep.csv",
        'desc': 'sleep',
        'pattern': '{subject}/eeg/{subject}_task-{task}_desc-sleep_epo.fif'
    }
}

# ============================================================================
# FUNCTIONS
# ============================================================================

def read_elements_csv(elements_file: str) -> pd.DataFrame:
    """Read elements CSV file into DataFrame."""
    return pd.read_csv(elements_file)


def build_file_path(subject: str, task: str, desc: str) -> Path:
    """Build expected file path from element components."""
    filename = f"{subject}_task-{task}_desc-{desc}_epo.fif"
    return Path(DERIVATIVES_DIR) / subject / "eeg" / filename


def diagnose_single(config_type: str):
    """Diagnose elements file for a specific config type."""
    config = CONFIGS[config_type]
    elements_file = config['elements_file']
    desc = config['desc']
    
    print("=" * 80)
    print(f"DIAGNOSING {config_type.upper()} ELEMENTS FILE")
    print("=" * 80)
    
    # Check if file exists
    if not Path(elements_file).exists():
        print(f"\n[WARN] Elements file not found: {elements_file}")
        print("       Run 'junifer queue' with the appropriate config to generate it.")
        return
    
    # Read elements CSV file
    print(f"\n1. Reading elements CSV file: {elements_file}")
    df = read_elements_csv(elements_file)
    
    print(f"   Total elements in file: {len(df)}")
    
    # Validate required columns
    required_cols = ['subject', 'task']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"   ERROR: Missing required columns: {missing_cols}")
        return
    
    # Convert to list of tuples
    elements = list(df[['subject', 'task']].itertuples(index=False, name=None))
    print(f"   Valid elements parsed: {len(elements)}")
    
    # Check for duplicates
    print(f"\n2. Checking for duplicate elements...")
    element_counts = Counter(elements)
    duplicates = {elem: count for elem, count in element_counts.items() if count > 1}
    
    if duplicates:
        print(f"   FOUND {len(duplicates)} DUPLICATE ELEMENTS:")
        for elem, count in sorted(duplicates.items()):
            subject, task = elem
            print(f"      {subject},{task} appears {count} times")
    else:
        print("   ✓ No duplicates found")
    
    # Check file existence
    print(f"\n3. Checking which derivative files exist...")
    missing_files = []
    existing_files = []
    
    for subject, task in elements:
        file_path = build_file_path(subject, task, desc)
        if file_path.exists():
            existing_files.append((subject, task))
        else:
            missing_files.append((subject, task, str(file_path)))
    
    print(f"   Existing files: {len(existing_files)}")
    print(f"   Missing files: {len(missing_files)}")
    
    if missing_files:
        print(f"\n   MISSING FILES (will cause 'invalid element' errors):")
        by_subject = {}
        for subject, task, path in missing_files:
            if subject not in by_subject:
                by_subject[subject] = []
            by_subject[subject].append(task)
        
        for subject in sorted(by_subject.keys()):
            print(f"\n      {subject}:")
            for task in by_subject[subject]:
                print(f"         {task}")
    else:
        print("   ✓ All files exist")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total elements in file:        {len(df)}")
    print(f"Unique elements:               {len(element_counts)}")
    print(f"Duplicate occurrences:         {len(elements) - len(element_counts)}")
    print(f"Elements with existing files:  {len(existing_files)}")
    print(f"Elements with missing files:   {len(missing_files)}")
    print(f"Expected successful runs:      {len(set(existing_files))}")
    
    if duplicates or missing_files:
        print("\n" + "=" * 80)
        print("RECOMMENDATIONS")
        print("=" * 80)
        if duplicates:
            print(f"\n⚠️  {len(duplicates)} duplicate elements found")
            print("   Run: python manage_elements.py fix")
        if missing_files:
            print(f"\n⚠️  {len(missing_files)} elements have missing derivative files")
            print("   Run: python manage_elements.py fix")
    else:
        print("\n✓ No issues detected!")


def fix_single(config_type: str):
    """Fix elements file for a specific config type."""
    config = CONFIGS[config_type]
    elements_file = config['elements_file']
    desc = config['desc']
    
    print("=" * 80)
    print(f"FIXING {config_type.upper()} ELEMENTS FILE")
    print("=" * 80)
    
    # Check if file exists
    if not Path(elements_file).exists():
        print(f"\n[ERROR] Elements file not found: {elements_file}")
        print("        Run 'junifer queue' with the appropriate config to generate it.")
        return
    
    # Backup original file
    backup_file = elements_file.replace('.csv', '') + datetime.now().strftime(".backup_%Y%m%d_%H%M%S.csv")
    print(f"\n1. Creating backup: {backup_file}")
    shutil.copy2(elements_file, backup_file)
    print("   ✓ Backup created")
    
    # Read elements CSV
    print(f"\n2. Reading elements CSV file: {elements_file}")
    df = read_elements_csv(elements_file)
    
    print(f"   Original elements: {len(df)}")
    
    # Process elements
    print(f"\n3. Processing elements...")
    
    # Track statistics
    original_count = len(df)
    duplicates_removed = 0
    missing_files_removed = 0
    
    # Remove duplicates
    df_before_dedup = df.copy()
    df = df.drop_duplicates(subset=['subject', 'task'], keep='first')
    duplicates_removed = len(df_before_dedup) - len(df)
    
    if duplicates_removed > 0:
        print(f"   Removed {duplicates_removed} duplicate entries")
    
    # Check file existence
    valid_rows = []
    for idx, row in df.iterrows():
        subject, task = row['subject'], row['task']
        file_path = build_file_path(subject, task, desc)
        
        if not file_path.exists():
            print(f"   Removing (file not found): {subject},{task}")
            missing_files_removed += 1
        else:
            valid_rows.append(row)
    
    # Create cleaned DataFrame
    df_clean = pd.DataFrame(valid_rows)
    elements = list(df_clean[['subject', 'task']].itertuples(index=False, name=None))
    
    print(f"\n   Elements after processing: {len(elements)}")
    print(f"   Duplicates removed: {duplicates_removed}")
    print(f"   Missing files removed: {missing_files_removed}")
    
    # Write cleaned CSV file
    print(f"\n4. Writing cleaned elements CSV file...")
    df_clean.to_csv(elements_file, index=False)
    
    print(f"   ✓ Written {len(elements)} elements to {elements_file}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Original elements:     {original_count}")
    print(f"Cleaned elements:      {len(elements)}")
    print(f"Elements removed:      {original_count - len(elements)}")
    print(f"  - Duplicates:        {duplicates_removed}")
    print(f"  - Missing files:     {missing_files_removed}")
    print(f"\nBackup saved to:       {backup_file}")
    print(f"Cleaned file:          {elements_file}")
    print(f"\n✓ {config_type.upper()} elements CSV file has been fixed!")


def diagnose(config_type: str = 'all'):
    """Diagnose elements files."""
    if config_type == 'all':
        for ctype in CONFIGS.keys():
            diagnose_single(ctype)
            print("\n")
    else:
        diagnose_single(config_type)


def fix(config_type: str = 'all'):
    """Fix elements files."""
    if config_type == 'all':
        for ctype in CONFIGS.keys():
            fix_single(ctype)
            print("\n")
    else:
        fix_single(config_type)


def show_usage():
    """Show usage information."""
    print("Usage: python manage_elements.py [COMMAND] [TYPE]")
    print()
    print("Commands:")
    print("  diagnose [state|evoked|sleep|all]    Check elements file(s) for issues (default: all)")
    print("  fix [state|evoked|sleep|all]         Fix elements file(s) (default: all)")
    print()
    print("Examples:")
    print("  python manage_elements.py diagnose")
    print("  python manage_elements.py diagnose state")
    print("  python manage_elements.py fix evoked")
    print("  python manage_elements.py fix sleep")
    print("  python manage_elements.py fix all")


# ============================================================================
# MAIN
# ============================================================================

def main():
    if len(sys.argv) < 2:
        show_usage()
        sys.exit(1)
    
    command = sys.argv[1]
    config_type = sys.argv[2] if len(sys.argv) > 2 else 'all'
    
    # Validate config_type
    if config_type not in ['state', 'evoked', 'sleep', 'all']:
        print(f"Unknown type: {config_type}")
        show_usage()
        sys.exit(1)
    
    if command == "diagnose":
        diagnose(config_type)
    elif command == "fix":
        fix(config_type)
    else:
        print(f"Unknown command: {command}")
        show_usage()
        sys.exit(1)


if __name__ == "__main__":
    main()
