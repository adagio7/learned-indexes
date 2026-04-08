# Workload-Aware Learned Index

A high-performance, **Workload-Aware Learned Index** featuring a lock-free query path, adaptive segment sharpening, and a high-concurrency Skip List delta buffer. This index adapts its structure in real-time based on query frequency to minimize search latency for "hot" key ranges.

## 🚀 Features

*   **Lock-Free Search Path**: Uses `std::atomic` snapshots and `std::shared_ptr` to eliminate blocking on the query path.
*   **Adaptive Sharpening**: A background monitoring thread tracks query patterns using a **Count-Min Sketch** and "sharpens" models (reducing error bounds) for frequently accessed segments.
*   **Concurrent Delta Buffer**: Powered by a **Concurrent Skip List** with atomic shared pointers, allowing non-blocking reads even during heavy ingestion.
*   **Copy-on-Write (CoW)**: Optimized background re-segmentation that only clones modified segments, minimizing memory bandwidth pressure.
*   **Gapped Array Storage**: Efficient data layout with physical gaps for faster in-place insertions.

## 🛠 Prerequisites

*   **Compiler**: C++17 compatible (GCC 8+, Clang 7+, or Apple Clang 11+).
*   **Build System**: CMake 3.14 or later.
*   **Python 3**: For plotting results (requires `matplotlib` and `pandas`).

## 🏗 Setup & Build

```bash
# Create build directory
mkdir -p build
cd build

# Configure and build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j
```

## 📈 Running Benchmarks

We provide a unified script to run the full build-test-plot lifecycle:

```bash
chmod +x run_benchmarks.sh
./run_benchmarks.sh
```

Alternatively, run individual benchmarks:

```bash
cd build
./adversarial_search
# Results are in build/adversarial_results.csv

# Generate plots
cd ..
python3 scripts/plot_adversarial.py
```

All plots are saved to the `figures/` directory.

## 📂 Project Structure

*   `include/`: Header-only core library (LearnedIndex, ConcurrentSkipList, GappedArray, etc.).
*   `benchmarks/`: Main benchmarking logic and workload generators.
*   `scripts/`: Python analysis and plotting scripts.
*   `figures/`: Generated performance visualizations.
*   `tests/`: Comprehensive unit and integration tests.

## 📝 License

This project was developed as part of COMP0252 - Algorithms for Computer Systems (Year 4 Term 2).
