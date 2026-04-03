
import sys
import time
import warnings
import collections
from datetime import datetime
from contextlib import contextmanager

class AnalysisLogger:
    """
    Logger for the analysis pipeline that tracks progress, timing,
    and manages/suppresses specific warnings.
    """
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.start_time = time.time()
        self.warning_counts = collections.defaultdict(lambda: collections.defaultdict(int))
        # Warnings to always suppress and count instead
        self.suppressed_warnings = [
            "ConvergenceWarning: The MLE may be on the boundary of the parameter space."
        ]
        
    def log(self, message):
        """Print message with timestamp."""
        if self.verbose:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"[{timestamp}] {message}")
            sys.stdout.flush()

    def warning(self, message):
        """Print warning message with timestamp."""
        self.log(f"WARNING: {message}")

    def log_section(self, title):
        """Print a section header."""
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"{title}")
            print(f"{'='*60}")
            sys.stdout.flush()
            
    @contextmanager
    def capture_warnings(self, context="Global"):
        """
        Context manager to capture and count specific warnings 
        instead of printing them to stderr.
        """
        # We use record=False so we can intercept showwarning dynamically.
        # If record=True, showwarning is replaced by a recorder that suppresses output,
        # which might conflict with our custom logic or other library behaviors.
        with warnings.catch_warnings(record=False):
            warnings.simplefilter("always") # Cause all warnings to be caught and passed to showwarning
            
            original_showwarning = warnings.showwarning
            
            def custom_showwarning(message, category, filename, lineno, file=None, line=None):
                msg_str = str(message)
                
                # Check if this is one of our target warnings to suppress
                is_target = False
                
                cat_name = category.__name__
                
                # Broad matching for the specific warning user requested
                if "ConvergenceWarning" in cat_name:
                    if "MLE may be on the boundary" in msg_str:
                         is_target = True
                
                # Also try matching just the message if category name varies
                if "MLE may be on the boundary" in msg_str:
                    is_target = True

                
                if is_target:
                    self.warning_counts[context]["ConvergenceWarning"] += 1
                else:
                     # Pass through other warnings to the original handler (usually prints to stderr)
                     if original_showwarning:
                        original_showwarning(message, category, filename, lineno, file, line)
                     else:
                        # Fallback if original was None (unlikely but safe)
                        sys.stderr.write(warnings.formatwarning(message, category, filename, lineno, line))
            
            warnings.showwarning = custom_showwarning
            try:
                yield
            finally:
                warnings.showwarning = original_showwarning

    def print_warning_summary(self):
        """Print a summary of caught warnings."""
        if not self.warning_counts:
            return

        self.log_section("WARNINGS SUMMARY")
        total_warnings = 0
        for context, counts in self.warning_counts.items():
            if counts:
                print(f"Context: {context}")
                for warning_type, count in counts.items():
                    print(f"  - {warning_type}: {count} occurrences")
                    total_warnings += count
        
        if total_warnings == 0:
            print("No significant warnings captured.")
        print("-" * 60)

    def print_execution_time(self):
        """Print total execution time."""
        total_time = time.time() - self.start_time
        print(f"\nTotal script execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
