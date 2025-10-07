---
applyTo: '**'
---
# Cursor General Rule: Clean & Maintainable Python Code

## Core Principles
- **Simplicity over complexity**: Always choose the most straightforward solution
- **Readability first**: Code should be self-explanatory to any developer
- **Maintainability**: Write code that's easy to modify and extend
- **Single responsibility**: Each function/class should have one clear purpose

## Code Style Guidelines

### Structure & Organization
- Keep functions small (max 20-30 lines when possible)
- Use clear, descriptive variable and function names
- Follow PEP 8 conventions consistently
- Group related functionality together
- Avoid deep nesting (max 3-4 levels)

### Documentation Requirements
- **Always** add docstrings to functions and classes using numpy/scipy style
- Include type hints for function parameters and return values
- Add inline comments for non-obvious logic or complex operations
- Document any assumptions or limitations

### Example Function Template:
```python
def process_neural_data(data: np.ndarray, sampling_rate: float) -> Dict[str, Any]:
    """
    Process neural time series data for mindwandering analysis.
    
    Parameters
    ----------
    data : np.ndarray
        Raw neural data with shape (n_channels, n_timepoints)
    sampling_rate : float
        Sampling frequency in Hz
        
    Returns
    -------
    Dict[str, Any]
        Processed results containing power spectra and connectivity metrics
        
    Notes
    -----
    Assumes data is already preprocessed and artifact-free.
    """
    # Clear implementation here
    pass
```

## Script Design Philosophy
### Configuration Style
- **In-script configuration**: All parameters should be easily modifiable variables at the top of the script
- **No CLI dependency**: Scripts should work by simply running them, not requiring command-line arguments
- **Plug-and-play approach**: User should only need to modify clearly marked variables and run
- **Optional CLI support**: Can include argparse as secondary option, but never as the primary interface

### Example Configuration Section:
```python
# =============================================================================
# CONFIGURATION - Modify these variables as needed
# =============================================================================
DATA_PATH = "path/to/your/data.csv"
OUTPUT_DIR = "results/"
SAMPLING_RATE = 1000  # Hz
WINDOW_SIZE = 2.0     # seconds
ANALYSIS_TYPE = "connectivity"  # options: 'connectivity', 'power', 'both'
# =============================================================================
```

## What to AVOID
- **No example scripts**: Never create demonstration or example files
- **No multiple versions**: Create only the requested script, no test variants
- **No CLI-focused design**: Avoid scripts that require terminal arguments to function
- **No over-engineering**: Avoid unnecessary abstractions or complex patterns
- **No verbose solutions**: Choose concise approaches over lengthy ones
- **No convoluted logic**: Break complex operations into simple steps
- **No undocumented code**: Every non-trivial piece needs explanation

## Data Science Specific
- Use pandas/numpy idiomatically (vectorized operations)
- Prefer sklearn/scipy built-in functions over custom implementations
- Keep analysis pipelines linear and easy to follow
- Use meaningful variable names for data transformations
- Comment on statistical assumptions and methods used

## File Management & Testing
- **Single script delivery**: Create only the requested main script
- **No example files**: Never generate demonstration or tutorial scripts
- **Debugging workflow**: Can create temporary test scripts during development for debugging purposes
- **Cleanup requirement**: Always delete any test/debug scripts before task completion
- **Clear filenames**: Use descriptive, meaningful names for the final script
- **Organized imports**: Standard library, third-party, then local imports at the top

## Error Handling
- Add basic error checking for common issues
- Use informative error messages
- Validate input parameters when necessary
- Handle edge cases gracefully

## Final Check
Before completing any task:
1. Is the code as simple as possible?
2. Are all functions and variables clearly named?
3. Is everything properly documented?
4. Can the user modify parameters easily at the top of the script?
5. Does the script work by simply running it (no CLI arguments required)?
6. Have I deleted any test/debug scripts created during development?
7. Have I created only the main script requested (no examples)?
8. Would another researcher easily understand and modify this code?