#!/usr/bin/env python3
"""
Memory profiling script for connectivity pipeline.
Helps identify which step is causing OOM issues.
"""

import gc
import psutil
import os
import yaml
from pathlib import Path

def get_memory_mb():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def main():
    print("=" * 70)
    print("  CONNECTIVITY PIPELINE MEMORY PROFILING")
    print("=" * 70)
    
    mem_start = get_memory_mb()
    print(f"\nInitial memory: {mem_start:.1f} MB")
    
    # Step 1: Load config
    print("\n[1/5] Loading config...")
    with open("Statistics_connectivity/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    mem_config = get_memory_mb()
    print(f"  Memory after config: {mem_config:.1f} MB (+{mem_config - mem_start:.1f} MB)")
    
    # Step 2: Import modules
    print("\n[2/5] Importing modules...")
    from reader import load_connectivity_data
    from lmm_connectivity import run_lmm_per_connection, apply_fdr_correction
    mem_imports = get_memory_mb()
    print(f"  Memory after imports: {mem_imports:.1f} MB (+{mem_imports - mem_config:.1f} MB)")
    
    # Step 3: Load ONE band of data
    print("\n[3/5] Loading connectivity data (theta band only)...")
    project = config.get("project", {})
    features_root = project["features_root"]
    
    try:
        band_df = load_connectivity_data(
            features_root=features_root,
            subjects=project.get("subjects"),
            tasks=project.get("tasks"),
            epoch_types=config.get("epoch_types"),
            bands=["theta"],  # Only one band
            verbose=True,
        )
        mem_data = get_memory_mb()
        print(f"  Memory after data load: {mem_data:.1f} MB (+{mem_data - mem_imports:.1f} MB)")
        print(f"  Data shape: {band_df.shape}")
        
        # Step 4: Prepare for LMM
        print("\n[4/5] Preparing data for LMM...")
        from reader import prepare_connectivity_for_lmm
        
        df_wide, connection_ids = prepare_connectivity_for_lmm(
            df=band_df,
            band="theta",
            epoch_type=None,
            onoff_max_value=project.get("onoff_max_value"),
            min_predictor_variability=project.get("min_predictor_variability"),
            min_minority_ratio=project.get("min_minority_ratio"),
        )
        mem_prep = get_memory_mb()
        print(f"  Memory after prep: {mem_prep:.1f} MB (+{mem_prep - mem_data:.1f} MB)")
        print(f"  Wide data shape: {df_wide.shape}")
        print(f"  Number of connections: {len(connection_ids)}")
        
        # Step 5: Run LMM (no permutations)
        print("\n[5/5] Running LMM (no permutations)...")
        lmm_cfg = config.get("lmm", {})
        
        results = run_lmm_per_connection(
            df_wide=df_wide,
            connection_ids=connection_ids[:10],  # Only first 10 connections for testing
            formula=lmm_cfg.get("formula", "power ~ onoff + (1|subject)"),
            predictor_of_interest=lmm_cfg.get("predictor_of_interest", "onoff"),
            method=lmm_cfg.get("method", "powell"),
            maxiter=lmm_cfg.get("maxiter", 500),
            random_state=lmm_cfg.get("random_state", 42),
        )
        mem_lmm = get_memory_mb()
        print(f"  Memory after LMM: {mem_lmm:.1f} MB (+{mem_lmm - mem_prep:.1f} MB)")
        
        # Cleanup
        del band_df, df_wide, results
        gc.collect()
        mem_cleanup = get_memory_mb()
        print(f"\n  Memory after cleanup: {mem_cleanup:.1f} MB")
        
        print("\n" + "=" * 70)
        print("  MEMORY PROFILE SUMMARY")
        print("=" * 70)
        print(f"  Peak memory usage: {mem_lmm:.1f} MB")
        print(f"  Data loading: +{mem_data - mem_imports:.1f} MB")
        print(f"  Data prep: +{mem_prep - mem_data:.1f} MB")
        print(f"  LMM (10 connections): +{mem_lmm - mem_prep:.1f} MB")
        print(f"  Estimated for all ~300 connections: ~{(mem_lmm - mem_prep) * 30:.1f} MB")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n  ERROR: {e}")
        import traceback
        traceback.print_exc()
        mem_error = get_memory_mb()
        print(f"\n  Memory at error: {mem_error:.1f} MB")

if __name__ == "__main__":
    main()
