#pragma once

#include <vector>
#include <memory>
#include <cmath>
#include <limits>
#include "IndexSegment.h"

template <typename KeyType = uint64_t, typename PayloadType = uint64_t>
class PLA_Builder {
public:
    // Build a set of index segments for the given sorted data using optimal cone (shrink) algorithm
    static std::vector<std::unique_ptr<IndexSegment<KeyType, PayloadType>>> build(
            const std::vector<std::pair<KeyType, PayloadType>>& sorted_data,
            size_t epsilon,
            double target_density) {
        
        std::vector<std::unique_ptr<IndexSegment<KeyType, PayloadType>>> segments;

        if (sorted_data.empty()) return segments;

        size_t n = sorted_data.size();
        size_t current_segment_start_idx = 0;

        while (current_segment_start_idx < n) {
            // Start a new segment
            double lower_slope = 0.0;
            double upper_slope = std::numeric_limits<double>::max();
            
            KeyType start_key = sorted_data[current_segment_start_idx].first;
            
            // To properly track indices inside a GappedArray for this segment, 
            // we first need to estimate how many elements go into this segment.
            // But the cone algorithm decides when to cut! 
            // It assumes dense packing during construction, y = index.
            // After we cut, we know the elements. Then we can multiply index by (1/target_density)
            // or just build the cone over y = (index / target_density) directly!
            // Let's build the cone assuming y = index / target_density.
            
            size_t i = current_segment_start_idx + 1;
            for (; i < n; ++i) {
                KeyType x = sorted_data[i].first - start_key;
                double y = static_cast<double>(i - current_segment_start_idx) / target_density;
                
                double cur_lower = (y - epsilon) / x;
                double cur_upper = (y + epsilon) / x;

                if (cur_lower > lower_slope) { lower_slope = cur_lower; }
                if (cur_upper < upper_slope) { upper_slope = cur_upper; }

                if (lower_slope > upper_slope) {
                    // Intersection is empty, break
                    break;
                }
            }

            // Segment goes from current_segment_start_idx to i-1
            size_t segment_length = i - current_segment_start_idx;
            
            // Calculate a valid slope for the segment (middle of the valid range)
            double chosen_slope;
            if (upper_slope == std::numeric_limits<double>::max()) {
                chosen_slope = 0.0; // Flat line if only 1 point
            } else {
                chosen_slope = (lower_slope + upper_slope) / 2.0;
            }

            KeyType end_key = (i < n) ? sorted_data[i].first : std::numeric_limits<KeyType>::max();
            
            auto segment = std::make_unique<IndexSegment<KeyType, PayloadType>>(
                start_key, end_key, chosen_slope, 0, epsilon);
                
            // Extract the data for this segment
            std::vector<std::pair<KeyType, PayloadType>> segment_data(
                sorted_data.begin() + current_segment_start_idx,
                sorted_data.begin() + i);
                
            segment->data_array->build(segment_data, target_density);
            
            segments.push_back(std::move(segment));
            current_segment_start_idx = i;
        }

        return segments;
    }
};
