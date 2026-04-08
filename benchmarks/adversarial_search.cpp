#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <thread>
#include "WorkloadGenerator.h"
#include "../include/LearnedIndex.h"
#include "BaselineIndex.h"

void run_scientific_test(std::ofstream& out, const std::string& name, 
                        const std::vector<uint64_t>& keys, 
                        const std::vector<uint64_t>& queries, 
                        size_t default_eps) {
    
    std::vector<std::pair<uint64_t, uint64_t>> data;
    for (auto k : keys) {
        data.push_back({k, k * 2});
    }
    
    BaselineIndex<uint64_t, uint64_t> baseline(data, default_eps);
    LearnedIndex<uint64_t, uint64_t> learned(data, default_eps, 0.8);
    
    std::cout << "--- Distribution: " << name << " ---" << std::endl;
    
    // 1. Measure Baseline
    auto t1 = std::chrono::high_resolution_clock::now();
    for (auto q : queries) baseline.search(q);
    auto t2 = std::chrono::high_resolution_clock::now();
    double b_ms = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0;
    out << name << ",Baseline," << b_ms << "\n";
    
    // 2. Train/Adapt
    std::cout << "  Adapting Learned Index (30s)..." << std::endl;
    for (int i=0; i<3; ++i) {
        for (auto q : queries) learned.search(q);
        std::this_thread::sleep_for(std::chrono::seconds(11));
    }
    
    // 3. STOP Background Monitoring
    std::cout << "  Stabilizing Index (Stopping Monitor)..." << std::endl;
    learned.stop_adaptation();
    std::this_thread::sleep_for(std::chrono::seconds(1));
    
    // 4. Measure Final Performance (Pure search path)
    t1 = std::chrono::high_resolution_clock::now();
    for (auto q : queries) learned.search(q);
    t2 = std::chrono::high_resolution_clock::now();
    double l_ms = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0;
    out << name << ",Learned," << l_ms << "\n";
    
    std::cout << "  Baseline: " << b_ms << " ms" << std::endl;
    std::cout << "  Learned:  " << l_ms << " ms" << std::endl;
    std::cout << "  Speedup:  " << (l_ms > 0 ? b_ms / l_ms : 0) << "x" << std::endl;
}

int main() {
    std::ofstream out("adversarial_results.csv");
    out << "Distribution,Index,TimeMS\n";
    
    size_t n = 1000000;
    size_t q_count = 500000;

    // 1. Lognormal Skew
    auto log_keys = WorkloadGenerator::generate_lognormal_keys(n, 10.0, 1.0);
    auto log_queries = WorkloadGenerator::generate_zipfian_queries(
        std::vector<uint64_t>(log_keys.end() - 10000, log_keys.end()), q_count, 1.95);
    run_scientific_test(out, "Lognormal_Skew", log_keys, log_queries, 64);

    // 2. Sine Wave (The Non-linear Killer)
    auto wave_keys = WorkloadGenerator::generate_wave_keys(n, 100.0, 0.01);
    auto wave_queries = WorkloadGenerator::generate_zipfian_queries(
        std::vector<uint64_t>(wave_keys.begin(), wave_keys.begin() + 10000), q_count, 1.95);
    run_scientific_test(out, "Sine_Wave", wave_keys, wave_queries, 128);

    // 3. Gaussian Cluster
    auto gau_keys = WorkloadGenerator::generate_gaussian_keys(n, 1000000.0, 50000.0);
    auto gau_queries = WorkloadGenerator::generate_zipfian_queries(gau_keys, q_count, 1.95);
    run_scientific_test(out, "Gaussian_Cluster", gau_keys, gau_queries, 64);

    out.close();
    std::cout << "\nFull multi-distribution suite complete. Results in adversarial_results.csv" << std::endl;
    return 0;
}
