#!/bin/bash
# Unified pipeline management script
# Handles: diagnosis, fixing, H5 cleanup, and job submission

set -euo pipefail

# ============================================================================
# CONFIGURATION
# ============================================================================
WORKDIR="/network/iss/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering"
ELEMENTS_FILE="${WORKDIR}/junifer_markers/1.markers_h5_creation/elements.csv"
CONFIG_FILE="${WORKDIR}/junifer_markers/1.markers_h5_creation/config.yaml"
FEATURES_DIR="/network/iss/cenir/analyse/meeg/CYBERSART/BIDS/features/junifer"
CONDA_ENV="junifer"

# SLURM parameters
CPUS=${CPUS:-4}
MEM=${MEM:-8G}
TIME=${TIME:-08:00:00}
PARTITION=${PARTITION:-}

# ============================================================================
# FUNCTIONS
# ============================================================================

show_usage() {
    cat << EOF
Usage: $(basename "$0") [COMMAND]

Commands:
    diagnose    Check elements file and H5 files status
    fix         Fix elements file (remove duplicates and missing files)
    clean-h5    Delete all H5 files to start fresh
    check-h5    Show H5 files status and count
    submit      Submit SLURM array jobs
    full        Run full workflow: diagnose -> fix -> clean-h5 -> submit

Examples:
    $(basename "$0") diagnose
    $(basename "$0") full
    CPUS=8 MEM=16G $(basename "$0") submit

EOF
}

diagnose_elements() {
    echo "=========================================="
    echo "DIAGNOSING ELEMENTS FILE"
    echo "=========================================="
    
    cd "$WORKDIR"
    
    # Activate conda
    if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniforge3/etc/profile.d/conda.sh"
        conda activate "$CONDA_ENV" 2>/dev/null || true
    fi
    
    python junifer_markers/1.markers_h5_creation/diagnose_elements.py
}

fix_elements() {
    echo "=========================================="
    echo "FIXING ELEMENTS FILE"
    echo "=========================================="
    
    cd "$WORKDIR"
    
    # Activate conda
    if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniforge3/etc/profile.d/conda.sh"
        conda activate "$CONDA_ENV" 2>/dev/null || true
    fi
    
    python junifer_markers/1.markers_h5_creation/fix_elements.py
}

check_h5_files() {
    echo "=========================================="
    echo "CHECKING H5 FILES"
    echo "=========================================="
    
    if [ ! -d "$FEATURES_DIR" ]; then
        echo "Features directory does not exist: $FEATURES_DIR"
        echo "Creating it..."
        mkdir -p "$FEATURES_DIR"
        echo "✓ Created"
        return 0
    fi
    
    echo "Features directory: $FEATURES_DIR"
    echo ""
    
    # Count H5 files
    H5_COUNT=$(find "$FEATURES_DIR" -name "element_*.h5" -type f 2>/dev/null | wc -l)
    echo "Total H5 files found: $H5_COUNT"
    
    if [ "$H5_COUNT" -eq 0 ]; then
        echo "No H5 files exist yet."
        return 0
    fi
    
    echo ""
    echo "First 10 H5 files:"
    find "$FEATURES_DIR" -name "element_*.h5" -type f | head -10
    
    echo ""
    echo "Total disk usage:"
    du -sh "$FEATURES_DIR"
    
    echo ""
    echo "File size distribution:"
    find "$FEATURES_DIR" -name "element_*.h5" -type f -exec ls -lh {} \; | awk '{print $5}' | sort | uniq -c
}

clean_h5_files() {
    echo "=========================================="
    echo "CLEANING H5 FILES"
    echo "=========================================="
    
    if [ ! -d "$FEATURES_DIR" ]; then
        echo "Features directory does not exist: $FEATURES_DIR"
        return 0
    fi
    
    H5_COUNT=$(find "$FEATURES_DIR" -name "element_*.h5" -type f 2>/dev/null | wc -l)
    
    if [ "$H5_COUNT" -eq 0 ]; then
        echo "No H5 files to delete."
        return 0
    fi
    
    echo "Found $H5_COUNT H5 files"
    echo "This will DELETE all element_*.h5 files in:"
    echo "  $FEATURES_DIR"
    echo ""
    read -p "Are you sure? (yes/no) " -r
    
    if [[ ! $REPLY == "yes" ]]; then
        echo "Aborted."
        return 1
    fi
    
    echo "Deleting H5 files..."
    find "$FEATURES_DIR" -name "element_*.h5" -type f -delete
    echo "✓ Deleted $H5_COUNT files"
}

submit_jobs() {
    echo "=========================================="
    echo "SUBMITTING SLURM JOBS"
    echo "=========================================="
    
    cd "$WORKDIR"
    
    # Count elements
    if [ ! -f "$ELEMENTS_FILE" ]; then
        echo "ERROR: Elements file not found: $ELEMENTS_FILE"
        return 1
    fi
    
    # Count lines excluding header
    N=$(tail -n +2 "$ELEMENTS_FILE" | wc -l)
    END=$((N - 1))
    
    echo "Total elements: $N"
    echo "Array indices: 0-${END}"
    echo ""
    
    # Build sbatch command
    SBATCH_CMD="sbatch --array=0-${END}"
    SBATCH_CMD="$SBATCH_CMD --cpus-per-task=${CPUS}"
    SBATCH_CMD="$SBATCH_CMD --mem=${MEM}"
    SBATCH_CMD="$SBATCH_CMD --time=${TIME}"
    
    if [ -n "$PARTITION" ]; then
        SBATCH_CMD="$SBATCH_CMD --partition=${PARTITION}"
    fi
    
    SBATCH_CMD="$SBATCH_CMD junifer_markers/1.markers_h5_creation/slurm_array_junifer.sh"
    
    echo "Submitting with command:"
    echo "$SBATCH_CMD"
    echo ""
    
    JOB_ID=$(eval $SBATCH_CMD | grep -oP '\d+')
    echo ""
    echo "✓ Jobs submitted! Job ID: $JOB_ID"
    echo ""
    echo "Monitor with:"
    echo "  squeue -u \$USER"
    echo "  watch squeue -u \$USER"
    echo ""
    echo "Check logs:"
    echo "  tail -f logs/CYBERSART_features_${JOB_ID}_*.out"
    echo ""
    echo "Check H5 files:"
    echo "  watch -n 60 'ls $FEATURES_DIR/element_*.h5 | wc -l'"
}

run_full_workflow() {
    echo "=========================================="
    echo "FULL WORKFLOW"
    echo "=========================================="
    echo ""
    
    # Step 1: Diagnose
    diagnose_elements
    echo ""
    read -p "Continue with fixing? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        return 1
    fi
    
    # Step 2: Fix
    echo ""
    fix_elements
    
    # Step 3: Check H5 files
    echo ""
    check_h5_files
    echo ""
    read -p "Delete existing H5 files and start fresh? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        clean_h5_files
    fi
    
    # Step 4: Submit
    echo ""
    read -p "Submit jobs now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        submit_jobs
    else
        echo "Skipped submission."
        echo ""
        echo "To submit manually, run:"
        echo "  $(basename "$0") submit"
    fi
    
    echo ""
    echo "=========================================="
    echo "DONE!"
    echo "=========================================="
}

# ============================================================================
# MAIN
# ============================================================================

COMMAND=${1:-}

case "$COMMAND" in
    diagnose)
        diagnose_elements
        ;;
    fix)
        fix_elements
        ;;
    clean-h5)
        clean_h5_files
        ;;
    check-h5)
        check_h5_files
        ;;
    submit)
        submit_jobs
        ;;
    full)
        run_full_workflow
        ;;
    *)
        show_usage
        exit 1
        ;;
esac
