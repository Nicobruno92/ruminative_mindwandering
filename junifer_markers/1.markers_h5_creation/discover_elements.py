#!/usr/bin/env python3
"""
Discover elements by scanning the derivatives directory.

This script scans the data directory to find all existing epoch files
and returns element information based on command-line arguments.

Configuration is read from pipeline_config.yaml in the same directory.

Usage:
    # Count elements for a description type
    python discover_elements.py --desc evoked --count
    
    # Get element at specific index (0-based)
    python discover_elements.py --desc evoked --index 5
    
    # List all elements for a description type
    python discover_elements.py --desc evoked --list
"""

import argparse
import re
import sys
from pathlib import Path

import yaml


def load_config(config_path: Path) -> dict:
    """Load the pipeline configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# Default config path (same directory as this script)
DEFAULT_CONFIG = Path(__file__).parent / 'pipeline_config.yaml'


def pattern_to_regex(pattern: str) -> re.Pattern:
    """Convert a file pattern with placeholders to a regex pattern."""
    escaped = re.escape(pattern)
    
    seen_placeholders = set()
    placeholder_patterns = {
        'subject': r'(sub-\d+)',
        'session': r'(ses-[a-z]+)',
        'task': r'([a-zA-Z0-9]+)',  # Allow digits for Sart1, Sart2, etc.
        'desc': r'([a-zA-Z]+)',
    }
    
    result = escaped
    for placeholder, regex in placeholder_patterns.items():
        escaped_placeholder = r'\{' + placeholder + r'\}'
        if escaped_placeholder in result:
            count = result.count(escaped_placeholder)
            if count > 0:
                if placeholder not in seen_placeholders:
                    named_regex = f'(?P<{placeholder}>{regex[1:-1]})'
                    result = result.replace(escaped_placeholder, named_regex, 1)
                    seen_placeholders.add(placeholder)
                    count -= 1
                while count > 0:
                    result = result.replace(escaped_placeholder, f'(?P={placeholder})', 1)
                    count -= 1
    
    return re.compile(result)


def discover_elements(datadir: Path, pattern: str, desc_filter: str,
                      subjects: list = None, sessions: list = None, 
                      tasks: list = None) -> list[dict]:
    """Scan the data directory and discover all valid elements for a description type."""
    pattern_regex = pattern_to_regex(pattern)
    elements = []
    
    # Convert filters to sets for fast lookup (empty list = no filter)
    subject_filter = set(subjects) if subjects else None
    session_filter = set(sessions) if sessions else None
    task_filter = set(tasks) if tasks else None
    
    for fif_path in datadir.rglob("*_epo.fif"):
        try:
            rel_path = fif_path.relative_to(datadir)
        except ValueError:
            continue
        
        match = pattern_regex.match(str(rel_path))
        if not match:
            continue
        
        element = match.groupdict()
        
        # Apply filters
        if element['desc'] != desc_filter:
            continue
        if subject_filter and element['subject'] not in subject_filter:
            continue
        if session_filter and element['session'] not in session_filter:
            continue
        if task_filter and element['task'] not in task_filter:
            continue
        
        elements.append(element)
    
    # Sort for reproducibility (session may not exist in all patterns)
    elements.sort(key=lambda x: (x['subject'], x.get('session', ''), x['task']))
    
    return elements


def main():
    parser = argparse.ArgumentParser(description="Discover elements from derivatives directory")
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG,
                        help='Path to pipeline_config.yaml')
    parser.add_argument('--desc', '-d', required=True,
                        help='Description type to filter by (e.g., evoked, state, sleep)')
    parser.add_argument('--count', '-c', action='store_true',
                        help='Print count of elements')
    parser.add_argument('--index', '-i', type=int, default=None,
                        help='Get element at specific index (0-based)')
    parser.add_argument('--list', '-l', action='store_true',
                        help='List all elements')
    
    args = parser.parse_args()
    
    # Load config
    if not args.config.exists():
        print(f"ERROR: Config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)
    
    config = load_config(args.config)
    datadir = Path(config['datadir'])
    pattern = config['pattern']
    
    # Get filters from config (empty list = no filter)
    subjects = config.get('subjects') or []
    sessions = config.get('sessions') or []
    tasks = config.get('tasks') or []
    
    elements = discover_elements(datadir, pattern, args.desc, subjects, sessions, tasks)
    
    if args.count:
        print(len(elements))
    elif args.index is not None:
        if args.index < 0 or args.index >= len(elements):
            print(f"ERROR: Index {args.index} out of range (0-{len(elements)-1})", file=sys.stderr)
            sys.exit(1)
        elem = elements[args.index]
        # Dynamic output: print only keys present in the element, in standard order
        standard_keys = ['subject', 'session', 'task', 'desc']
        output_values = [elem[k] for k in standard_keys if k in elem]
        print(",".join(output_values))
    elif args.list:
        for elem in elements:
            standard_keys = ['subject', 'session', 'task', 'desc']
            output_values = [elem[k] for k in standard_keys if k in elem]
            print(",".join(output_values))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
