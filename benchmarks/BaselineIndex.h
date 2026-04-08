#pragma once

#include <vector>
#include <memory>
#include <optional>
#include "../include/IndexSegment.h"
#include "../include/PLA.h"

template <typename KeyType = uint64_t, typename PayloadType = uint64_t>
class BaselineIndex {
private:
    std::vector<std::unique_ptr<IndexSegment<KeyType, PayloadType>>> segments_;
    size_t epsilon_;

public:
    BaselineIndex(const std::vector<std::pair<KeyType, PayloadType>>& initial_data, 
                  size_t epsilon = 64, double density = 1.0)
        : epsilon_(epsilon) {
        segments_ = PLA_Builder<KeyType, PayloadType>::build(initial_data, epsilon_, density);
    }

    std::optional<PayloadType> search(KeyType key) const {
        if (segments_.empty()) return std::nullopt;

        auto it = std::upper_bound(segments_.begin(), segments_.end(), key,
            [](KeyType k, const std::unique_ptr<IndexSegment<KeyType, PayloadType>>& seg) {
                return k < seg->start_key;
            });
            
        if (it != segments_.begin()) {
            --it;
        }
        
        const auto& segment = *it;
        if (key < segment->start_key) return std::nullopt;

        size_t pred_pos = segment->predict(key);
        const auto& data = segment->data_array->get_data();
        if (data.empty()) return std::nullopt;

        size_t start_bound = (pred_pos > segment->epsilon) ? pred_pos - segment->epsilon : 0;
        size_t end_bound = std::min(pred_pos + segment->epsilon + 1, data.size());
        
        for (size_t i = start_bound; i < end_bound; ++i) {
            if (data[i].isValid() && data[i].key == key) {
                return data[i].payload;
            } else if (data[i].isValid() && data[i].key > key) {
                break;
            }
        }
        return std::nullopt;
    }

    size_t get_size_bytes() const {
        size_t bytes = sizeof(BaselineIndex);
        bytes += segments_.size() * sizeof(IndexSegment<KeyType, PayloadType>);
        for (const auto& seg : segments_) {
            bytes += seg->data_array->capacity() * sizeof(typename GappedArray<KeyType, PayloadType>::Element);
        }
        return bytes;
    }
};
