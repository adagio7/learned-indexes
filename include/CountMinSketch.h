#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <functional>

class CountMinSketch {
private:
    size_t width_;
    size_t depth_;
    size_t mask_;
    std::vector<uint32_t> table_;

    uint32_t hash(uint64_t item, size_t i) const {
        uint64_t h = item * (0x517cc1b727220a95ULL + i * 0x71b1a19b9071a5ULL);
        h ^= h >> 32;
        return static_cast<uint32_t>(h & mask_);
    }

public:
    CountMinSketch(size_t width, size_t depth) : width_(width), depth_(depth) {
        // Enforce power of two for mask optimization
        if ((width & (width - 1)) != 0) {
            // Find next power of two
            size_t p = 1;
            while (p < width) p <<= 1;
            width_ = p;
        }
        mask_ = width_ - 1;
        table_.assign(width_ * depth_, 0);
    }

    void add(uint64_t item, uint32_t count = 1) {
        for (size_t i = 0; i < depth_; ++i) {
            table_[i * width_ + hash(item, i)] += count;
        }
    }

    uint32_t estimate(uint64_t item) const {
        uint32_t min_count = std::numeric_limits<uint32_t>::max();
        for (size_t i = 0; i < depth_; ++i) {
            min_count = std::min(min_count, table_[i * width_ + hash(item, i)]);
        }
        return min_count;
    }

    void decay(uint32_t factor = 2) {
        for (auto& val : table_) {
            val /= factor;
        }
    }

    void reset() {
        std::fill(table_.begin(), table_.end(), 0);
    }
};
