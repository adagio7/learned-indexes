#pragma once

#include <vector>
#include <random>
#include <cmath>
#include <algorithm>

class WorkloadGenerator {
public:
    static std::vector<uint64_t> generate_uniform_keys(size_t num_keys, uint64_t max_val) {
        std::vector<uint64_t> keys(num_keys);
        std::mt19937_64 gen(42);
        std::uniform_int_distribution<uint64_t> dist(1, max_val);
        for (size_t i = 0; i < num_keys; ++i) {
            keys[i] = dist(gen);
        }
        std::sort(keys.begin(), keys.end());
        // Remove duplicates for simplicity
        keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
        return keys;
    }

    static std::vector<uint64_t> generate_lognormal_keys(size_t num_keys, double mean = 10.0, double stddev = 2.0) {
        std::vector<uint64_t> keys(num_keys);
        std::mt19937_64 gen(42);
        std::lognormal_distribution<double> dist(mean, stddev);
        
        for (size_t i = 0; i < num_keys; ++i) {
            double raw = dist(gen);
            keys[i] = static_cast<uint64_t>(raw * 100) + 1; 
        }
        
        std::sort(keys.begin(), keys.end());
        keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
        return keys;
    }

    static std::vector<uint64_t> generate_gaussian_keys(size_t num_keys, double mean = 1000000.0, double stddev = 100000.0) {
        std::vector<uint64_t> keys(num_keys);
        std::mt19937_64 gen(42);
        std::normal_distribution<double> dist(mean, stddev);
        
        for (size_t i = 0; i < num_keys; ++i) {
            double raw = dist(gen);
            // Ensure positive keys
            keys[i] = static_cast<uint64_t>(std::abs(raw)) + 1; 
        }
        
        std::sort(keys.begin(), keys.end());
        keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
        return keys;
    }

    static std::vector<uint64_t> generate_wave_keys(size_t num_keys, double amplitude = 50.0, double frequency = 0.001) {
        std::vector<uint64_t> keys(num_keys);
        // We use a simple linear sequence + sine wave
        for (size_t i = 0; i < num_keys; ++i) {
            double wave = std::sin(static_cast<double>(i) * frequency) * amplitude;
            keys[i] = static_cast<uint64_t>(static_cast<double>(i) * 10.0 + wave) + 100;
        }
        
        std::sort(keys.begin(), keys.end());
        keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
        return keys;
    }

    // Generate queries based on Zipfian distribution to simulate hot/cold zones
    static std::vector<uint64_t> generate_zipfian_queries(const std::vector<uint64_t>& data, size_t num_queries, double skew = 1.2) {
        std::vector<uint64_t> queries(num_queries);
        size_t n = data.size();
        
        // Simple Zipfian approximation generator
        std::vector<double> cdf(n, 0.0);
        double c = 0.0;
        for (size_t i = 1; i <= n; ++i) {
            c += (1.0 / std::pow(i, skew));
        }
        c = 1.0 / c;
        
        double sum = 0.0;
        for (size_t i = 0; i < n; ++i) {
            sum += c / std::pow(i + 1, skew);
            cdf[i] = sum;
        }

        std::mt19937_64 gen(42);
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        // Map the zipf distribution onto the keys in a randomized way.
        // Instead of rank 1 always being keys[0], we shuffle the index mapping.
        std::vector<size_t> index_mapping(n);
        std::iota(index_mapping.begin(), index_mapping.end(), 0);
        std::shuffle(index_mapping.begin(), index_mapping.end(), gen);

        for (size_t i = 0; i < num_queries; ++i) {
            double p = dist(gen);
            // Binary search in CDF
            auto it = std::lower_bound(cdf.begin(), cdf.end(), p);
            size_t zipf_rank = std::distance(cdf.begin(), it);
            if (zipf_rank >= n) zipf_rank = n - 1;
            
            queries[i] = data[index_mapping[zipf_rank]];
        }

        return queries;
    }
};
