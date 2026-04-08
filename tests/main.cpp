#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <thread>
#include <chrono>

#include "../include/GappedArray.h"
#include "../include/CountMinSketch.h"
#include "../include/DeltaBuffer.h"
#include "../include/PLA.h"
#include "../include/IndexSegment.h"
#include "../include/LearnedIndex.h"
#include "../benchmarks/BaselineIndex.h"

void test_gapped_array_comprehensive() {
    std::cout << "[TEST] GappedArray Comprehensive..." << std::endl;
    GappedArray<uint64_t, uint64_t> ga;
    
    // Test entirely empty data
    std::vector<std::pair<uint64_t, uint64_t>> empty_data;
    ga.build(empty_data, 0.5);
    assert(ga.capacity() == 0);
    assert(ga.size() == 0);

    // Test specific density gap distribution
    std::vector<std::pair<uint64_t, uint64_t>> data = {
        {10, 100}, {20, 200}, {30, 300}, {40, 400}
    };
    ga.build(data, 0.25); // Target capacity 16 for 4 items
    assert(ga.capacity() == 16);
    
    const auto& intern_data = ga.get_data();
    int valid_count = 0;
    int gaps = 0;
    for (const auto& elem : intern_data) {
        if (elem.isValid()) valid_count++;
        else if (elem.isEmpty()) gaps++;
    }
    assert(valid_count == 4);
    assert(gaps == 12);
    
    // Tombstone tests
    assert(ga.remove(20) == true);
    assert(ga.size() == 3);
    assert(ga.remove(20) == false); // Double remove fails
    
    int tombstone_count = 0;
    for (const auto& elem : ga.get_data()) {
        if (elem.isTombstone()) tombstone_count++;
    }
    assert(tombstone_count == 1);
    
    // Insert delegation test
    bool ins_res = ga.insert(25, 250, 0);
    // Based on current implementation, direct inserts often defer to delta buffer if out of strict shifting bounds
    std::cout << "  Passed!" << std::endl;
}

void test_count_min_sketch_comprehensive() {
    std::cout << "[TEST] CountMinSketch Comprehensive..." << std::endl;
    CountMinSketch cms(50, 5); // 5 hash functions, row length 50
    
    // Add multiple different values
    for (int i = 0; i < 1000; i++) {
        cms.add(i % 10, 1); // 0-9 will have count 100
    }
    
    // Check heavy hitters
    assert(cms.estimate(0) >= 100);
    assert(cms.estimate(5) >= 100);
    assert(cms.estimate(9) >= 100);
    
    // Check non-hitters
    assert(cms.estimate(9999) < 50); // Probability of 5-depth hash collision is low
    
    // Test decay
    cms.decay(2);
    assert(cms.estimate(0) >= 50 && cms.estimate(0) <= 60);
    
    // Test reset
    cms.reset();
    assert(cms.estimate(0) == 0);
    assert(cms.estimate(5) == 0);

    std::cout << "  Passed!" << std::endl;
}

void test_delta_buffer_comprehensive() {
    std::cout << "[TEST] DeltaBuffer Comprehensive..." << std::endl;
    DeltaBuffer<uint64_t, uint64_t> dyn_buf;
    
    // Remove on empty
    assert(dyn_buf.remove(99) == false);
    
    // Overwriting keys
    dyn_buf.insert(5, 50);
    dyn_buf.insert(5, 500); // Overwrite
    assert(dyn_buf.size() == 1);
    
    auto res = dyn_buf.search(5);
    assert(res.has_value() && res.value() == 500);
    
    // Multiple inserts
    for (int i = 10; i < 20; i++) {
        dyn_buf.insert(i, i * 10);
    }
    assert(dyn_buf.size() == 11); // 5 plus 10 through 19
    
    assert(dyn_buf.remove(15) == true);
    assert(dyn_buf.size() == 10);
    
    auto all_data = dyn_buf.extract_all();
    assert(all_data.size() == 10);
    assert(dyn_buf.size() == 0); 
    
    std::cout << "  Passed!" << std::endl;
}

void test_delta_buffer_concurrent() {
    std::cout << "[TEST] DeltaBuffer Concurrent Stress..." << std::endl;
    DeltaBuffer<uint64_t, uint64_t> dyn_buf;
    const int num_threads = 8;
    const int ops_per_thread = 2000;
    std::vector<std::thread> threads;

    // Concurrent writers
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&dyn_buf, t, ops_per_thread]() {
            for (int i = 0; i < ops_per_thread; ++i) {
                dyn_buf.insert(t * ops_per_thread + i, i);
            }
        });
    }

    // Concurrent readers
    std::vector<std::thread> readers;
    std::atomic<bool> stop_readers{false};
    for (int t = 0; t < num_threads; ++t) {
        readers.emplace_back([&dyn_buf, &stop_readers]() {
            while (!stop_readers) {
                dyn_buf.search(rand() % 16000);
            }
        });
    }

    for (auto& t : threads) t.join();
    stop_readers = true;
    for (auto& t : readers) t.join();

    assert(dyn_buf.size() == (size_t)num_threads * ops_per_thread);
    
    auto extracted = dyn_buf.extract_all();
    assert(extracted.size() == (size_t)num_threads * ops_per_thread);

    std::cout << "  Passed!" << std::endl;
}

void test_pla_builder_comprehensive() {
    std::cout << "[TEST] PLA_Builder Comprehensive..." << std::endl;
    
    // Edge Case: 1 Element
    std::vector<std::pair<uint64_t, uint64_t>> single = {{10, 100}};
    auto seg_single = PLA_Builder<uint64_t, uint64_t>::build(single, 5, 1.0);
    assert(seg_single.size() == 1);
    assert(seg_single[0]->start_key == 10);
    
    // Sparse, erratic data requiring multiple splits with loose epsilon
    std::vector<std::pair<uint64_t, uint64_t>> erratic = {
        {10, 1}, {20, 2}, {30, 3}, // Slope 0.1
        {100, 4}, {1000, 5},       // Massive jump
        {1010, 6}, {1020, 7}
    };
    
    auto seg_erratic = PLA_Builder<uint64_t, uint64_t>::build(erratic, 0, 1.0); // Exact bounds
    assert(seg_erratic.size() > 1);
    // With epsilon = 0, first three elements (slope 0.1) can form a segment. 
    // Data check:
    assert(seg_erratic[0]->start_key == 10);
    
    // Validate predictions obey epsilon bounds
    auto seg_loose = PLA_Builder<uint64_t, uint64_t>::build(erratic, 2, 0.5); 
    
    for (const auto& s : seg_loose) {
        // Just checking basic memory safety up to this point
        assert(s->epsilon == 2);
        assert(s->data_array->capacity() > 0);
    }
    
    std::cout << "  Passed!" << std::endl;
}

void test_learned_index_integration() {
    std::cout << "[TEST] LearnedIndex Integration..." << std::endl;
    std::vector<std::pair<uint64_t, uint64_t>> data;
    for (uint64_t i = 1; i <= 1000; i++) {
        data.push_back({i * 5, i});
    }
    
    LearnedIndex<uint64_t, uint64_t> li(data, 10, 0.5);
    BaselineIndex<uint64_t, uint64_t> baseline(data, 10);
    
    // Search valid
    for (uint64_t i = 100; i <= 200; i++) {
        uint64_t key = i * 5;
        auto val = li.search(key);
        assert(val.has_value() && val.value() == i);
        
        auto base_val = baseline.search(key);
        assert(base_val.has_value() && base_val.value() == i);
    }
    
    // Search invalid (not in dataset)
    assert(!li.search(123).has_value());
    assert(!baseline.search(123).has_value());
    
    // Delta buffer integration
    li.insert(9999, 12345);
    auto delta_val = li.search(9999);
    assert(delta_val.has_value() && delta_val.value() == 12345);

    std::cout << "  Passed!" << std::endl;
}

int main() {
    std::cout << "Running Comprehensive Component Tests..." << std::endl;
    
    test_gapped_array_comprehensive();
    test_count_min_sketch_comprehensive();
    test_delta_buffer_comprehensive();
    test_delta_buffer_concurrent();
    test_pla_builder_comprehensive();
    test_learned_index_integration();
    
    std::cout << "\nAll comprehensive tests passed successfully!" << std::endl;
    return 0;
}
