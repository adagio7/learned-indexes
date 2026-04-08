#!/bin/bash

# Workload-Aware Learned Index: Unified Benchmark Runner

set -e # Exit on error

# 1. Setup Environment
echo "--- Setting up build environment ---"
mkdir -p build
mkdir -p figures

# 2. Build Project
echo "--- Compiling Project (Release Mode) ---"
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(sysctl -n hw.ncpu)
echo "--- Build Complete ---"

# 3. Run Experiments
echo "--- Running Adversarial Stress Test ---"
./adversarial_search
echo "--- Running Traditional Throughput Benchmark ---"
# ./benchmark_runner # Assuming this generates results.csv if implemented

# 4. Generate Plots
echo "--- Generating Performance Visualizations ---"
cd ..
source venv/bin/activate 2>/dev/null || echo "Venv not found, using system python"

python3 scripts/plot_adversarial.py
# python3 scripts/plot.py # Uncomment if results.csv exists
# python3 scripts/generate_heatmap.py # Uncomment to generate profiling heatmap

echo ""
echo "--- Benchmark Cycle Complete! ---"
echo "Results summary: adversarial_results.csv"
echo "New plots saved to: figures/"
ls -lh figures/
