#pragma once
#include <vector>
#include <optional>
#include <atomic>
#include "ConcurrentSkipList.h"

template <typename KeyType = uint64_t, typename PayloadType = uint64_t>
class DeltaBuffer {
private:
    ConcurrentSkipList<KeyType, PayloadType> skip_list_;

public:
    bool empty() const {
        return skip_list_.size() == 0;
    }

    void insert(KeyType key, PayloadType payload) {
        skip_list_.insert(key, payload);
    }

    bool remove(KeyType key) {
        return skip_list_.remove(key);
    }

    std::optional<PayloadType> search(KeyType key) const {
        return skip_list_.find(key);
    }

    std::vector<std::pair<KeyType, PayloadType>> extract_all() {
        return skip_list_.extract_all();
    }

    size_t size() const {
        return skip_list_.size();
    }

    size_t get_size_bytes() const {
        return skip_list_.get_size_bytes();
    }
};
