"""
Utils package for CyberSART Data Harmonization Pipeline.

This package provides:
- bids_compliance_harmonized: BIDS-compliant I/O with session/run support
- io_helpers: Data finding, montage setup, event recoding
- trigger_correction: Annotation harmonization and marker processing
- qa_report: QA report generation for harmonization results
- metadata_helpers: Participant metadata handling
"""

from .bids_compliance_harmonized import BIDSComplianceHarmonized
from .io_helpers import DataFinder, MontageHelper, EventRecoder, read_brainvision_safe
from .trigger_correction import TriggerCorrector
from .qa_report import HarmonizationQAReport

__all__ = [
    'BIDSComplianceHarmonized',
    'DataFinder',
    'MontageHelper',
    'EventRecoder',
    'TriggerCorrector',
    'HarmonizationQAReport',
    'read_brainvision_safe',
]
