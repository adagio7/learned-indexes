#pragma once

#include <memory>
#include <vector>
#include "GappedArray.h"

template <typename KeyType = uint64_t, typename PayloadType = uint64_t>
class IndexSegment {
public:
    KeyType start_key;
    KeyType end_key;
    
    // Linear model: pos = slope * (key - intercept_key) + intercept_pos
    // Or simplified: pos = slope * key + intercept
    // To handle large integers without precision loss, usually: pos = (key - min_key) * slope
    double slope;
    size_t intercept;

    size_t epsilon;
    
    // Workload statistics (aligned to prevent false sharing)
    alignas(64) mutable std::atomic<uint64_t> query_count{0};
    alignas(64) mutable std::atomic<uint64_t> total_search_steps{0};

    // The actual data array for this segment. Since we want an Atomic Swap,
    // we manage data inside the segment, or via a shared_ptr if shared.
    // For simplicity, we use unique_ptr.
    std::unique_ptr<GappedArray<KeyType, PayloadType>> data_array;

    IndexSegment(KeyType start, KeyType end, double slp, size_t icept, size_t eps)
        : start_key(start), end_key(end), slope(slp), intercept(icept), epsilon(eps) {
        data_array = std::make_unique<GappedArray<KeyType, PayloadType>>();
    }
    
    // For predictions
    size_t predict(KeyType key) const {
        if (key < start_key) return 0;
        double pred = (static_cast<double>(key) - start_key) * slope + intercept;
        if (pred < 0) return 0;
        return static_cast<size_t>(pred);
    }
    
    double get_avg_steps() const {
        uint64_t count = query_count.load(std::memory_order_relaxed);
        if (count == 0) return 0.0;
        return static_cast<double>(total_search_steps.load(std::memory_order_relaxed)) / count;
    }
    
    double get_pain_score() const {
        uint64_t count = query_count.load(std::memory_order_relaxed);
        if (count < 100) return 0.0; // Wait for stable sample size
        
        double avg_steps = get_avg_steps();
        // If search is already efficient (avg < 2), don't flag as "painful"
        if (avg_steps < 2.0) return 0.0;
        
        // Pain = Query Volume * Inefficiency
        return static_cast<double>(count) * (avg_steps - 1.0);
    }
    
    void reset_stats() {
        query_count = 0;
        total_search_steps = 0;
    }

    size_t get_size_bytes() const {
        size_t bytes = sizeof(IndexSegment);
        if (data_array) {
            bytes += data_array->capacity() * sizeof(typename GappedArray<KeyType, PayloadType>::Element);
        }
        return bytes;
    }
};
