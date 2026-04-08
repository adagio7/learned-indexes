#pragma once

#include <vector>
#include <limits>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

template <typename KeyType = uint64_t, typename PayloadType = uint64_t>
class GappedArray {
public:
    static constexpr KeyType EMPTY_KEY = std::numeric_limits<KeyType>::max();
    static constexpr KeyType TOMBSTONE_KEY = std::numeric_limits<KeyType>::max() - 1;

    struct Element {
        KeyType key;
        PayloadType payload;

        bool isEmpty() const { return key == EMPTY_KEY; }
        bool isTombstone() const { return key == TOMBSTONE_KEY; }
        bool isValid() const { return key != EMPTY_KEY && key != TOMBSTONE_KEY; }
    };

private:
    std::vector<Element> data_;
    size_t count_;
    double target_density_; // 1.0 means full, 0.0 means empty

public:
    GappedArray() : count_(0), target_density_(1.0) {}

    // Build from sorted data, leaving spaces evenly distributed to reach target_density
    // target_density = num_elements / total_capacity
    void build(const std::vector<std::pair<KeyType, PayloadType>>& sorted_data, double target_density) {
        if (target_density <= 0.0 || target_density > 1.0) {
            throw std::invalid_argument("target_density must be in (0, 1]");
        }
        target_density_ = target_density;
        count_ = sorted_data.size();
        size_t capacity = static_cast<size_t>(count_ / target_density);
        
        data_.assign(capacity, {EMPTY_KEY, PayloadType{}});

        if (count_ == 0) return;

        double step = static_cast<double>(capacity) / count_;
        double current_pos = 0.0;
        for (const auto& item : sorted_data) {
            size_t idx = static_cast<size_t>(current_pos);
            if (idx >= capacity) idx = capacity - 1;
            while (idx < capacity && data_[idx].key != EMPTY_KEY) {
                idx++;
            }
            if (idx < capacity) {
                data_[idx] = {item.first, item.second};
            }
            current_pos += step;
        }
    }

    // Try to insert in-place. If shifting distance exceeds limit, return false.
    bool insert(KeyType key, PayloadType payload, size_t predicted_pos, size_t max_shift = 32) {
        if (predicted_pos >= data_.size()) {
             predicted_pos = data_.size() > 0 ? data_.size() - 1 : 0;
        }

        // Find nearest empty slot
        size_t empty_idx = predicted_pos;
        int shift_dir = 0; // 0 for right, 1 for left
        size_t dist_right = 0;
        size_t dist_left = 0;

        // Search right
        while (predicted_pos + dist_right < data_.size() && data_[predicted_pos + dist_right].isValid()) {
            dist_right++;
        }
        // Search left
        while (predicted_pos >= dist_left && data_[predicted_pos - dist_left].isValid()) {
            dist_left++;
        }

        bool right_valid = (predicted_pos + dist_right < data_.size());
        bool left_valid = (predicted_pos >= dist_left);

        if (!right_valid && !left_valid) return false;

        size_t min_dist = data_.size() + 1;
        if (right_valid) min_dist = dist_right;
        if (left_valid && dist_left < min_dist) {
            min_dist = dist_left;
            shift_dir = 1;
            empty_idx = predicted_pos - dist_left;
        } else if (right_valid) {
            empty_idx = predicted_pos + dist_right;
        } else {
            return false;
        }

        if (min_dist > max_shift) {
            return false;
        }

        // Shift elements to make room at the correct insertion point
        // First find actual insertion point
        size_t insert_idx = predicted_pos;

        // Note: For a real learned b-tree, you'd do binary/exponential search to find exact insert point,
        // then shift everything up to the nearest gap. 
        // For simplicity in this structure block, we just assume caller handles it or we do it here.
        // Let's implement a simple shift to keep elements sorted.
        
        // Find insert pos
        // ... (We'll expand this later if we need strict in-place insertion with shifting)
        // Actually, let's keep it simple: if there's no gap exactly where we need it, we let the DeltaBuffer take it.
        // Or we just binary search the true position, and shift.

        // True binary search using bounds:
        auto it = std::upper_bound(data_.begin(), data_.end(), key,
            [](KeyType k, const Element& e) {
                if (!e.isValid()) return false; 
                return k < e.key;
            });
        
        insert_idx = std::distance(data_.begin(), it);

        // ... To be fully completed for shifting ...
        return false; // Force to delta buffer for now
    }

    bool remove(KeyType key) {
        // Binary search-like for key, mark as TOMBSTONE
        for (auto& item : data_) {
            if (item.key == key) {
                item.key = TOMBSTONE_KEY;
                count_--;
                return true;
            }
        }
        return false;
    }

    size_t size() const { return count_; }
    size_t capacity() const { return data_.size(); }
    double density() const { return capacity() > 0 ? static_cast<double>(count_) / capacity() : 0.0; }

    const std::vector<Element>& get_data() const { return data_; }

    // ARM NEON Vectorized Search
    size_t find_index_simd(KeyType target_key, size_t start, size_t end) const {
#ifdef __ARM_NEON
        if (target_key == EMPTY_KEY || target_key == TOMBSTONE_KEY) return end;
        
        uint64x2_t v_target = vdupq_n_u64(target_key);
        size_t i = start;
        
        // Process 2 elements at a time
        for (; i + 1 < end; i += 2) {
            // vld2q_u64 loads interleaved keys and payloads:
            // v_keys: [data_[i].key, data_[i+1].key]
            // v_pals: [data_[i].payload, data_[i+1].payload]
            uint64x2x2_t v_elements = vld2q_u64(reinterpret_cast<const uint64_t*>(&data_[i]));
            uint64x2_t v_keys = v_elements.val[0];
            
            // Compare keys
            uint64x2_t v_cmp = vceqq_u64(v_keys, v_target);
            
            // Extract match mask
            uint64_t match0 = vgetq_lane_u64(v_cmp, 0);
            uint64_t match1 = vgetq_lane_u64(v_cmp, 1);
            
            if (match0) return i;
            if (match1) return i + 1;
        }
        
        // Scalar fallback for the last element
        for (; i < end; ++i) {
            if (data_[i].key == target_key) return i;
        }
#else
        // Generic scalar fallback
        for (size_t i = start; i < end; ++i) {
            if (data_[i].key == target_key) return i;
        }
#endif
        return end;
    }
};
