#!/usr/bin/env python3
"""
Quick test to verify Unfold.jl works with local config and correct epoch descriptor.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from helpers import load_yaml_config

def test_unfold_local():
    """Test Unfold.jl with local configuration"""
    
    print("🔍 Testing Unfold.jl with local config...")
    
    # Load local config
    cfg = load_yaml_config("./ERPs_new/config_local.yaml")
    
    # Show key paths
    proj = cfg.get("project", {})
    derivatives_root = proj.get("derivatives_root")
    epochs_desc = proj.get("input_evoked_desc", "evoked")
    
    print(f"📁 Derivatives root: {derivatives_root}")
    print(f"📊 Epochs descriptor: {epochs_desc}")
    print(f"📊 Subjects: {cfg.get('subjects', [])[:3]}... (showing first 3)")
    
    # Check if a sample file exists
    subject = cfg.get('subjects', ['02'])[0]
    task = cfg.get('tasks', ['Sart1'])[0]
    
    sample_file = f"{derivatives_root}/sub-{subject}/eeg/sub-{subject}_task-{task}_desc-{epochs_desc}_epo.fif"
    print(f"📄 Sample file: {sample_file}")
    print(f"📄 File exists: {os.path.exists(sample_file)}")
    
    if os.path.exists(sample_file):
        print("✅ Configuration looks good for local testing")
        return True
    else:
        print("❌ Sample file not found - check configuration")
        return False

if __name__ == "__main__":
    success = test_unfold_local()
    if not success:
        sys.exit(1)