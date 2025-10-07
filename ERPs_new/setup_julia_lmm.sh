#!/bin/bash

"""
Simple Julia LMM Setup Script

This script installs the minimal required Julia packages for temporal LMM analysis.
Works on both local machines and cluster environments.

Usage:
    # For local machines:
    ./setup_julia_lmm.sh

    # For clusters:
    module load julia
    ./setup_julia_lmm.sh

Author: AI Assistant
"""
module load julia

set -e  # Exit on any error

echo "🔬 Setting up Julia for temporal LMM analysis"
echo "=============================================="

# Check if Julia is available
if ! command -v julia &> /dev/null; then
    echo "❌ Julia not found in PATH"
    echo ""
    echo "📋 Installation options:"
    echo "  Local: Download from https://julialang.org/downloads/"
    echo "  Cluster: module load julia"
    echo "  Ubuntu: sudo apt install julia"
    echo "  macOS: brew install julia"
    exit 1
fi

echo "✅ Found Julia: $(julia --version)"

# Install required packages
echo ""
echo "📦 Installing required Julia packages..."

julia -e '
using Pkg

# Required packages for temporal LMM
packages = [
    "DataFrames",
    "CSV", 
    "StatsModels",
    "MixedModels",
    "CategoricalArrays",
    "ArgParse"
]

println("Installing packages...")
for pkg in packages
    try
        Pkg.add(pkg)
        println("✅ $pkg")
    catch e
        println("❌ Failed to install $pkg: $e")
        exit(1)
    end
end

# Test installation
println("\n🧪 Testing packages...")
try
    using DataFrames, CSV, StatsModels, MixedModels, CategoricalArrays, ArgParse
    println("✅ All packages loaded successfully!")
catch e
    println("❌ Package loading failed: $e")
    exit(1)
end
'

echo ""
echo "🎉 Julia LMM setup completed!"
echo ""
echo "📋 Next steps:"
echo "  1. Enable Julia LMM in config.yaml:"
echo "     lmm_analysis:"
echo "       use_julia: true"
echo ""
echo "  2. Run analysis:"
echo "     python lmm_analysis.py --config config.yaml"
