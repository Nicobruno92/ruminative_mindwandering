"""
CyberSART Data Harmonization Pipeline

This package converts raw BrainVision EEG data to BIDS-compliant format
with annotation harmonization and QA reporting.

Main entry point: data_harmonization.py

Usage:
    python data_harmonization.py --config config.yaml
    python data_harmonization.py --subject 02 --task Sart1
"""

__version__ = "1.0.0"
