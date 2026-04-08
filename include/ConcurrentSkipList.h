#pragma once

#include <vector>
#include <atomic>
#include <memory>
#include <random>
#include <mutex>
#include <optional>

template <typename KeyType, typename PayloadType>
class ConcurrentSkipList {
public:
    static constexpr int MAX_LEVEL = 16;

    struct Node {
        KeyType key;
        PayloadType payload;
        int height;
        std::shared_ptr<Node> next[MAX_LEVEL];

        Node(KeyType k, PayloadType p, int h) 
            : key(k), payload(p), height(h) {
            for (int i = 0; i < MAX_LEVEL; ++i) {
                next[i] = nullptr;
            }
        }
    };

private:
    std::shared_ptr<Node> head_;
    std::atomic<size_t> size_{0};
    mutable std::mutex write_mtx_; 

public:
    ConcurrentSkipList() {
        head_ = std::make_shared<Node>(KeyType{}, PayloadType{}, MAX_LEVEL);
    }

    void insert(KeyType key, PayloadType payload) {
        std::lock_guard<std::mutex> lock(write_mtx_);
        
        std::shared_ptr<Node> update[MAX_LEVEL];
        auto curr = head_;

        for (int i = MAX_LEVEL - 1; i >= 0; --i) {
            auto next_ptr = std::atomic_load(&curr->next[i]);
            while (next_ptr && next_ptr->key < key) {
                curr = next_ptr;
                next_ptr = std::atomic_load(&curr->next[i]);
            }
            update[i] = curr;
        }

        auto next_at_0 = std::atomic_load(&curr->next[0]);
        if (next_at_0 && next_at_0->key == key) {
            next_at_0->payload = payload;
            return;
        }

        int height = random_level();
        auto new_node = std::make_shared<Node>(key, payload, height);

        for (int i = 0; i < height; ++i) {
            auto next_at_i = std::atomic_load(&update[i]->next[i]);
            std::atomic_store(&new_node->next[i], next_at_i);
            std::atomic_store(&update[i]->next[i], new_node);
        }
        size_.fetch_add(1);
    }

    std::optional<PayloadType> find(KeyType key) const {
        auto curr = head_;
        for (int i = MAX_LEVEL - 1; i >= 0; --i) {
            auto next_ptr = std::atomic_load(&curr->next[i]);
            while (next_ptr && next_ptr->key < key) {
                curr = next_ptr;
                next_ptr = std::atomic_load(&curr->next[i]);
            }
        }

        auto target = std::atomic_load(&curr->next[0]);
        if (target && target->key == key) {
            return target->payload;
        }
        return std::nullopt;
    }

    bool remove(KeyType key) {
        std::lock_guard<std::mutex> lock(write_mtx_);
        
        std::shared_ptr<Node> update[MAX_LEVEL];
        auto curr = head_;

        for (int i = MAX_LEVEL - 1; i >= 0; --i) {
            auto next_ptr = std::atomic_load(&curr->next[i]);
            while (next_ptr && next_ptr->key < key) {
                curr = next_ptr;
                next_ptr = std::atomic_load(&curr->next[i]);
            }
            update[i] = curr;
        }

        auto target = std::atomic_load(&curr->next[0]);
        if (!target || target->key != key) {
            return false;
        }

        for (int i = 0; i < target->height; ++i) {
            if (std::atomic_load(&update[i]->next[i]) != target) break;
            auto target_next = std::atomic_load(&target->next[i]);
            std::atomic_store(&update[i]->next[i], target_next);
        }
        size_.fetch_sub(1);
        return true;
    }

    std::vector<std::pair<KeyType, PayloadType>> extract_all() {
        std::lock_guard<std::mutex> lock(write_mtx_);
        std::vector<std::pair<KeyType, PayloadType>> result;
        
        auto curr = std::atomic_load(&head_->next[0]);
        while (curr) {
            result.push_back({curr->key, curr->payload});
            curr = std::atomic_load(&curr->next[0]);
        }

        // Clear the skip list
        for (int i = 0; i < MAX_LEVEL; ++i) {
            std::atomic_store(&head_->next[i], std::shared_ptr<Node>(nullptr));
        }
        size_.store(0);
        return result;
    }

    size_t size() const {
        return size_.load();
    }

    size_t get_size_bytes() const {
        size_t bytes = sizeof(ConcurrentSkipList);
        // Approximation: count nodes
        bytes += size_.load() * (sizeof(Node) + MAX_LEVEL * sizeof(std::shared_ptr<Node>));
        return bytes;
    }

private:
    int random_level() {
        static thread_local std::mt19937 gen(std::random_device{}());
        static std::uniform_real_distribution<float> dist(0, 1);
        int lvl = 1;
        while (dist(gen) < 0.5f && lvl < MAX_LEVEL) {
            lvl++;
        }
        return lvl;
    }
};
