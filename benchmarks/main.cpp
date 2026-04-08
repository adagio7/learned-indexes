#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <thread>
#include <vector>

#include "../include/LearnedIndex.h"
#include "BaselineIndex.h"
#include "WorkloadGenerator.h"

int main() {
  std::cout << "Generating Data (Lognormal Adversarial Scheme)..." << std::endl;
  // Generate 10M lognormal keys
  auto keys = WorkloadGenerator::generate_lognormal_keys(10000000, 10.0, 2.0);
  std::vector<std::pair<uint64_t, uint64_t>> data;
  for (auto k : keys) {
    data.push_back({k, k * 2}); // Dummy payload
  }

  std::cout << "Data size: " << data.size() << std::endl;

  std::cout << "Generating Queries (Zipfian)..." << std::endl;
  // Generate 1M queries with higher skew (1.6)
  auto queries =
      WorkloadGenerator::generate_zipfian_queries(keys, 1000000, 1.6);
  size_t block_size = 100000;
  size_t num_blocks = queries.size() / block_size;

  std::vector<double> baseline_times(num_blocks);
  std::vector<double> learned_times(num_blocks);

  // Baseline Assessment
  std::cout << "\n--- Baseline Index ---" << std::endl;
  BaselineIndex<uint64_t, uint64_t> baseline(data, 64, 0.8);

  size_t found_baseline = 0;
  for (size_t b = 0; b < num_blocks; ++b) {
    size_t start_idx = b * block_size;
    size_t end_idx = start_idx + block_size;

    auto start = std::chrono::high_resolution_clock::now();
    size_t b_found = 0;
    for (size_t i = start_idx; i < end_idx; ++i) {
      auto res = baseline.search(queries[i]);
      if (res.has_value())
        b_found++;
    }
    auto end = std::chrono::high_resolution_clock::now();
    baseline_times[b] =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count() /
        1000.0;
    found_baseline += b_found;
    std::cout << "Block " << b + 1 << " Time: " << baseline_times[b] << " ms"
              << std::endl;
  }
  std::cout << "Baseline found: " << found_baseline << " / " << queries.size()
            << std::endl;

  // Workload-Aware Learned Index Assessment
  std::cout << "\n--- Workload-Aware Learned Index ---" << std::endl;
  LearnedIndex<uint64_t, uint64_t> workload_index(data, 64, 0.8);

  size_t found_learned = 0;
  for (size_t b = 0; b < num_blocks; ++b) {
    size_t start_idx = b * block_size;
    size_t end_idx = start_idx + block_size;

    auto start = std::chrono::high_resolution_clock::now();
    size_t b_found = 0;
    for (size_t i = start_idx; i < end_idx; ++i) {
      auto res = workload_index.search(queries[i]);
      if (res.has_value())
        b_found++;
    }
    auto end = std::chrono::high_resolution_clock::now();
    learned_times[b] =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count() /
        1000.0;
    found_learned += b_found;

    std::cout << "Block " << b + 1 << " Time: " << learned_times[b] << " ms"
              << std::endl;

    // Sleep to let monitoring thread trigger re-segmentation
    if (b < num_blocks - 1) {
      std::cout << "(Sleeping 10s for background thread...)" << std::endl;
      std::this_thread::sleep_for(std::chrono::seconds(10));
    }
  }
  std::cout << "Learned segments: " << workload_index.get_segment_count() << std::endl;

  std::cout << "\n--- Concurrent Write/Read Stress (Delta Buffer) ---"
            << std::endl;
  // Measure latency while 4 threads are flooding the index with insertions
  std::atomic<bool> stop_stress{false};
  std::vector<std::thread> writers;
  for (int i = 0; i < 4; ++i) {
    writers.emplace_back([&workload_index, &stop_stress, i]() {
      uint64_t key_base = 200000000 + (i * 1000000);
      uint64_t counter = 0;
      while (!stop_stress) {
        workload_index.insert(key_base + counter, counter);
        counter++;
      }
    });
  }

  auto stress_start = std::chrono::high_resolution_clock::now();
  size_t stress_queries = 100000;
  size_t stress_found = 0;
  for (size_t i = 0; i < stress_queries; ++i) {
    auto res = workload_index.search(queries[i % queries.size()]);
    if (res.has_value())
      stress_found++;
  }
  auto stress_end = std::chrono::high_resolution_clock::now();
  double stress_ms = std::chrono::duration_cast<std::chrono::microseconds>(
                         stress_end - stress_start)
                         .count() /
                     1000.0;

  stop_stress = true;
  for (auto &t : writers)
    t.join();

  std::cout << "100K queries during high-concurrency writes: " << stress_ms
            << " ms" << std::endl;
  std::cout << "Delta Buffer Size after stress: "
            << workload_index.get_size_bytes() << " (conceptually)"
            << std::endl;

  // Write to CSV
  std::ofstream out("results.csv");
  out << "Block,BaselineTime,LearnedTime\n";
  for (size_t b = 0; b < num_blocks; ++b) {
    out << b + 1 << "," << baseline_times[b] << "," << learned_times[b] << "\n";
  }
  out.close();
  std::cout << "\nResults written to results.csv\n";

  return 0;
}
