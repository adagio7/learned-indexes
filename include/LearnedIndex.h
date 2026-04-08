#pragma once

#include <vector>
#include <memory>
#include <shared_mutex>
#include <thread>
#include <atomic>
#include <chrono>
#include <optional>
#include <iostream>
#include <algorithm>

#include "IndexSegment.h"
#include "DeltaBuffer.h"
#include "CountMinSketch.h"
#include "PLA.h"

struct ProfilingResult {
    uint64_t sync_ns;
    uint64_t selection_ns;
    uint64_t model_ns;
    uint64_t total_ns;
};

template <typename KeyType = uint64_t, typename PayloadType = uint64_t>
struct CompactMetadata {
    KeyType start_key;
    double slope;
    size_t intercept;
    size_t epsilon;
    IndexSegment<KeyType, PayloadType>* segment;
    
    // For std::upper_bound
    bool operator<(KeyType key) const { return start_key < key; }
};

template <typename KeyType = uint64_t, typename PayloadType = uint64_t>
class LearnedIndex {
private:
    using SegmentList = std::vector<std::shared_ptr<IndexSegment<KeyType, PayloadType>>>;
    using MetadataList = std::vector<CompactMetadata<KeyType, PayloadType>>;
    
    struct IndexState {
        std::shared_ptr<SegmentList> segments;
        std::shared_ptr<MetadataList> metadata;
    };
    
    std::atomic<IndexState*> state_ptr_{nullptr};
    mutable std::vector<std::shared_ptr<IndexState>> history_;
    mutable std::mutex history_mtx_;
    
    DeltaBuffer<KeyType, PayloadType> delta_buffer_;
    mutable CountMinSketch sketch_;
    
    std::atomic<bool> stop_monitor_{false};
    std::thread monitor_thread_;
    
    size_t default_epsilon_;
    double default_density_;
    
    // Configurable thresholds
    double pain_threshold_ = 5000.0; // Trigger on hot regions after sampled query counts
    size_t hot_epsilon_ = 8;          // Sharpen aggressively
    double hot_density_ = 0.8;      // 20% gap
    
    size_t cold_epsilon_ = 128;
    double cold_density_ = 0.95; // 5% gap
    uint64_t merge_threshold_ = 0; // Threshold for coarsening (0 = disabled)

public:
    LearnedIndex(const std::vector<std::pair<KeyType, PayloadType>>& initial_data, 
                 size_t default_eps = 64, double default_den = 0.8)
        : sketch_(1024, 4), default_epsilon_(default_eps), default_density_(default_den) {
        
        auto initial_segments_raw = PLA_Builder<KeyType, PayloadType>::build(initial_data, default_epsilon_, default_density_);
        auto segments = std::make_shared<SegmentList>();
        auto metadata = std::make_shared<MetadataList>();
        
        for (auto& seg_ptr : initial_segments_raw) {
            auto shared_seg = std::shared_ptr<IndexSegment<KeyType, PayloadType>>(std::move(seg_ptr));
            segments->push_back(shared_seg);
            metadata->push_back({
                shared_seg->start_key, shared_seg->slope, shared_seg->intercept, 
                shared_seg->epsilon, shared_seg.get()
            });
        }
        
        auto initial_state = std::make_shared<IndexState>();
        initial_state->segments = segments;
        initial_state->metadata = metadata;
        state_ptr_.store(initial_state.get(), std::memory_order_release);
        history_.push_back(initial_state);
        
        // Start monitoring thread
        monitor_thread_ = std::thread(&LearnedIndex::monitoring_loop, this);
    }
    
    ~LearnedIndex() {
        stop_adaptation();
    }

    void stop_adaptation() {
        bool expected = false;
        if (stop_monitor_.compare_exchange_strong(expected, true)) {
            if (monitor_thread_.joinable()) {
                monitor_thread_.join();
            }
        }
    }

    std::optional<PayloadType> search(KeyType key) const {
        auto state = state_ptr_.load(std::memory_order_acquire);
        if (!state || state->metadata->empty()) return std::nullopt;

        // 1. Check DeltaBuffer
        if (!delta_buffer_.empty()) {
            auto delta_res = delta_buffer_.search(key);
            if (delta_res.has_value()) return delta_res;
        }

        // 2. Selection (Flat Binary Search)
        const auto& metadata = *(state->metadata);
        auto it = std::upper_bound(metadata.begin(), metadata.end(), key,
            [](KeyType k, const CompactMetadata<KeyType, PayloadType>& m) {
                return k < m.start_key;
            });
            
        if (it != metadata.begin()) --it;
        const auto& meta = *it;
        if (key < meta.start_key) return std::nullopt;

        // 3. Model Search (Inlined + SIMD)
        auto* segment = meta.segment;
        double d_pred = (static_cast<double>(key) - meta.start_key) * meta.slope + meta.intercept;
        size_t pred_pos = (d_pred < 0) ? 0 : static_cast<size_t>(d_pred);
        
        const auto& data_vec = segment->data_array->get_data();
        if (data_vec.empty()) return std::nullopt;

        size_t start_bound = (pred_pos > meta.epsilon) ? pred_pos - meta.epsilon : 0;
        size_t end_bound = std::min(pred_pos + meta.epsilon + 1, data_vec.size());
        
        size_t found_idx = segment->data_array->find_index_simd(key, start_bound, end_bound);
        
        if (found_idx < end_bound) {
            // Occasional stats update (Sampled 1/32)
            thread_local uint64_t q_counter = 0;
            if ((++q_counter & 31) == 0) {
                segment->query_count.fetch_add(32, std::memory_order_relaxed);
                segment->total_search_steps.fetch_add((found_idx - start_bound + 1) * 32, std::memory_order_relaxed);
                sketch_.add(key, 1);
            }
            return data_vec[found_idx].payload;
        }
        return std::nullopt;
    }

    std::optional<PayloadType> search_profiled(KeyType key, ProfilingResult* prof = nullptr) const {
        auto t_start = std::chrono::high_resolution_clock::now();
        
        // 1. Synchronization
        auto t_sync_start = std::chrono::high_resolution_clock::now();
        auto state = state_ptr_.load(std::memory_order_acquire);
        auto t_sync_end = std::chrono::high_resolution_clock::now();
        
        if (!state || state->metadata->empty()) return std::nullopt;

        // 2. Check DeltaBuffer
        if (!delta_buffer_.empty()) {
            auto delta_res = delta_buffer_.search(key);
            if (delta_res.has_value()) return delta_res;
        }

        // 2. Selection (Flat)
        auto t_selection_start = std::chrono::high_resolution_clock::now();
        const auto& metadata = *(state->metadata);
        auto it = std::upper_bound(metadata.begin(), metadata.end(), key,
            [](KeyType k, const CompactMetadata<KeyType, PayloadType>& m) {
                return k < m.start_key;
            });
        if (it != metadata.begin()) --it;
        const auto& meta = *it;
        auto t_selection_end = std::chrono::high_resolution_clock::now();
        
        if (key < meta.start_key) return std::nullopt;

        // 3. Model Search (Inlined + SIMD)
        auto t_model_start = std::chrono::high_resolution_clock::now();
        auto* segment = meta.segment;
        double d_pred = (static_cast<double>(key) - meta.start_key) * meta.slope + meta.intercept;
        size_t pred_pos = (d_pred < 0) ? 0 : static_cast<size_t>(d_pred);

        const auto& data_vec = segment->data_array->get_data();
        size_t start_bound = (pred_pos > meta.epsilon) ? pred_pos - meta.epsilon : 0;
        size_t end_bound = std::min(pred_pos + meta.epsilon + 1, data_vec.size());
        
        size_t found_idx = segment->data_array->find_index_simd(key, start_bound, end_bound);
        
        std::optional<PayloadType> res = std::nullopt;
        uint64_t search_steps = 0;
        if (found_idx < end_bound) {
            res = data_vec[found_idx].payload;
            search_steps = found_idx - start_bound + 1;
        }
        auto t_model_end = std::chrono::high_resolution_clock::now();

        if (prof) {
            prof->sync_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t_sync_end - t_sync_start).count();
            prof->selection_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t_selection_end - t_selection_start).count();
            prof->model_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t_model_end - t_model_start).count();
            prof->total_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t_model_end - t_start).count();
        }

        // Occasional stats update (Sampled 1/32)
        thread_local uint64_t q_counter = 0;
        if ((++q_counter & 31) == 0) {
            segment->query_count.fetch_add(32, std::memory_order_relaxed);
            segment->total_search_steps.fetch_add(search_steps * 32, std::memory_order_relaxed);
            sketch_.add(key, 1);
        }

        return res;
    }

    void insert(KeyType key, PayloadType payload) {
        delta_buffer_.insert(key, payload);
    }

    size_t get_segment_count() const {
        auto state = state_ptr_.load(std::memory_order_acquire);
        return state ? state->segments->size() : 0;
    }

    size_t get_size_bytes() const {
        size_t bytes = sizeof(LearnedIndex);
        bytes += delta_buffer_.get_size_bytes();
        auto state = state_ptr_.load(std::memory_order_acquire);
        if (state) {
            bytes += state->segments->size() * sizeof(std::shared_ptr<IndexSegment<KeyType, PayloadType>>);
            bytes += state->metadata->size() * sizeof(CompactMetadata<KeyType, PayloadType>);
            for (const auto& seg : *(state->segments)) {
                bytes += seg->get_size_bytes();
            }
        }
        return bytes;
    }

private:
    void monitoring_loop() {
        std::cout << "--- Monitor Thread Started ---" << std::endl;
        while (!stop_monitor_) {
            std::this_thread::sleep_for(std::chrono::seconds(2));
            if (stop_monitor_) break;
            
            std::cout << "--- Monitor Check (" << delta_buffer_.size() << " deltas) ---" << std::endl;
            if (delta_buffer_.size() > 1000) {
                merge_delta_buffer();
            }

            // Analyze and Re-segment
            analyze_and_resegment();
            std::cout.flush();
        }
    }
    
    void merge_delta_buffer() {
        auto current_state = state_ptr_.load(std::memory_order_acquire);
        if (!current_state) return;
        
        auto deltas = delta_buffer_.extract_all();
        if (deltas.empty()) return;
        
        std::vector<std::pair<KeyType, PayloadType>> all_data;
        for (const auto& seg : *(current_state->segments)) {
            auto seg_data = seg->data_array->get_data();
            for (const auto& element : seg_data) {
                if (!element.isEmpty()) {
                    all_data.push_back({element.key, element.payload});
                }
            }
        }
        
        for (const auto& kv : deltas) {
            all_data.push_back(kv);
        }
        
        std::sort(all_data.begin(), all_data.end(), [](const auto& a, const auto& b){
            return a.first < b.first;
        });
        
        auto new_segments_raw = PLA_Builder<KeyType, PayloadType>::build(all_data, default_epsilon_, default_density_);
        auto new_segments = std::make_shared<SegmentList>();
        auto new_metadata = std::make_shared<MetadataList>();
        
        for (auto& seg_ptr : new_segments_raw) {
            auto shared_seg = std::shared_ptr<IndexSegment<KeyType, PayloadType>>(std::move(seg_ptr));
            new_segments->push_back(shared_seg);
            new_metadata->push_back({
                shared_seg->start_key, shared_seg->slope, shared_seg->intercept, 
                shared_seg->epsilon, shared_seg.get()
            });
        }
        
        auto new_state = std::make_shared<IndexState>();
        new_state->segments = new_segments;
        new_state->metadata = new_metadata;
        {
            std::lock_guard<std::mutex> lock(history_mtx_);
            state_ptr_.store(new_state.get(), std::memory_order_release);
            history_.push_back(new_state);
        }
        std::cout << "--- Delta Merge Complete. New Segments: " << new_segments->size() << " ---" << std::endl;
    }

    void analyze_and_resegment() {
        auto current_state = state_ptr_.load(std::memory_order_acquire);
        if (!current_state || current_state->metadata->empty()) return;
        
        std::vector<std::pair<size_t, double>> segments_to_retrain;
        const auto& old_segments = *(current_state->segments);
        for (size_t i = 0; i < old_segments.size(); ++i) {
            double score = old_segments[i]->get_pain_score();
            if (score > pain_threshold_) {
                segments_to_retrain.push_back({i, score});
            }
        }
        
        sketch_.decay(2);
        
        std::vector<std::pair<size_t, size_t>> cold_runs;
        size_t run_start = 0;
        bool in_run = false;
        
        if (merge_threshold_ > 0) {
            for (size_t i = 0; i < old_segments.size(); ++i) {
                uint64_t count = old_segments[i]->query_count.load(std::memory_order_relaxed);
                bool is_cold = (count < merge_threshold_) && (old_segments[i]->epsilon < default_epsilon_);
                
                if (is_cold) {
                    if (!in_run) {
                        run_start = i;
                        in_run = true;
                    }
                } else {
                    if (in_run) {
                        if (i - run_start > 1) {
                            cold_runs.push_back({run_start, i});
                        }
                        in_run = false;
                    }
                }
            }
            if (in_run && (old_segments.size() - run_start > 1)) {
                cold_runs.push_back({run_start, old_segments.size()});
            }
        }
        
        if (segments_to_retrain.empty() && cold_runs.empty()) return;
        
        auto t_retrain_start = std::chrono::high_resolution_clock::now();
        if (!segments_to_retrain.empty()) {
            std::cout << "--- Retraining " << segments_to_retrain.size() << " hot regions ---" << std::endl;
        }
        if (!cold_runs.empty()) {
            std::cout << "--- Merging " << cold_runs.size() << " cold runs ---" << std::endl;
        }

        auto new_segments = std::make_shared<SegmentList>();
        auto new_metadata = std::make_shared<MetadataList>();
        
        for (size_t i = 0; i < old_segments.size(); ) {
            bool merged = false;
            for (auto& run : cold_runs) {
                if (i == run.first) {
                    std::vector<std::pair<KeyType, PayloadType>> merged_data;
                    for (size_t j = run.first; j < run.second; ++j) {
                        const auto& seg_data_ref = old_segments[j]->data_array->get_data();
                        for (const auto& elem : seg_data_ref) {
                            if (!elem.isEmpty()) {
                                merged_data.push_back({elem.key, elem.payload});
                            }
                        }
                    }
                    
                    auto coarse_segments = PLA_Builder<KeyType, PayloadType>::build(
                        merged_data, default_epsilon_, 1.0);
                        
                    for (auto& raw_seg : coarse_segments) {
                        auto shared_seg = std::shared_ptr<IndexSegment<KeyType, PayloadType>>(std::move(raw_seg));
                        new_segments->push_back(shared_seg);
                        new_metadata->push_back({
                            shared_seg->start_key, shared_seg->slope, shared_seg->intercept, 
                            shared_seg->epsilon, shared_seg.get()
                        });
                    }
                    
                    i = run.second;
                    merged = true;
                    break;
                }
            }
            if (merged) continue;

            bool retrain = false;
            for (auto& target : segments_to_retrain) {
                if (target.first == i) { retrain = true; break; }
            }
            
            if (retrain) {
                std::vector<std::pair<KeyType, PayloadType>> seg_data;
                const auto& old_data_ref = old_segments[i]->data_array->get_data();
                for (const auto& elem : old_data_ref) {
                    if (!elem.isEmpty()) {
                        seg_data.push_back({elem.key, elem.payload});
                    }
                }
                
                auto localized_segments = PLA_Builder<KeyType, PayloadType>::build(
                    seg_data, hot_epsilon_, hot_density_);
                    
                for (auto& raw_seg : localized_segments) {
                    auto shared_seg = std::shared_ptr<IndexSegment<KeyType, PayloadType>>(std::move(raw_seg));
                    new_segments->push_back(shared_seg);
                    new_metadata->push_back({
                        shared_seg->start_key, shared_seg->slope, shared_seg->intercept, 
                        shared_seg->epsilon, shared_seg.get()
                    });
                }
            } else {
                auto shared_seg = old_segments[i];
                new_segments->push_back(shared_seg);
                new_metadata->push_back({
                    shared_seg->start_key, shared_seg->slope, shared_seg->intercept, 
                    shared_seg->epsilon, shared_seg.get()
                });
            }
            i++;
        }
        
        auto new_state = std::make_shared<IndexState>();
        new_state->segments = new_segments;
        new_state->metadata = new_metadata;
        {
            std::lock_guard<std::mutex> lock(history_mtx_);
            state_ptr_.store(new_state.get(), std::memory_order_release);
            history_.push_back(new_state);
        }
        
        auto t_retrain_end = std::chrono::high_resolution_clock::now();
        auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(t_retrain_end - t_retrain_start).count();
        std::cout << "--- Retraining Complete (" << dur << "ms). Segments: " 
                  << old_segments.size() << " -> " << new_segments->size() << " ---" << std::endl;
        std::cout.flush();
    }
};
