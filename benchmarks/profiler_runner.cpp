#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <thread>
#include "../include/LearnedIndex.h"
#include "WorkloadGenerator.h"

int main() {
    std::cout << "--- Performance Profiling Initialization (Sharpened State) ---" << std::endl;
    
    // 1. Generate data
    auto keys = WorkloadGenerator::generate_lognormal_keys(2000000, 10.0, 2.0);
    std::vector<std::pair<uint64_t, uint64_t>> data;
    for (auto k : keys) {
        data.push_back({k, k * 2});
    }

    // 2. Initialize Index
    LearnedIndex<uint64_t, uint64_t> index(data, 64, 0.8);
    
    // 3. Warming (Trigger Retraining)
    std::cout << "Triggering retraining with warm-up queries..." << std::endl;
    auto warm_queries = WorkloadGenerator::generate_zipfian_queries(keys, 500000, 1.5);
    for (auto q : warm_queries) {
        index.search(q);
    }
    
    std::cout << "Waiting 12 seconds for background thread to sharpen index..." << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(12));
    std::cout << "Index status: " << index.get_segment_count() << " segments." << std::endl;

    // 4. Profile
    auto queries = WorkloadGenerator::generate_zipfian_queries(keys, 10000, 1.5);
    std::cout << "Profiling 10,000 queries on sharpened index..." << std::endl;
    
    std::ofstream out("profile_data.csv");
    out << "QueryID,Sync_ns,Selection_ns,ModelSearch_ns,Total_ns\n";
    
    for (size_t i = 0; i < queries.size(); ++i) {
        ProfilingResult prof;
        index.search_profiled(queries[i], &prof);
        
        out << i << "," 
            << prof.sync_ns << "," 
            << prof.selection_ns << "," 
            << prof.model_ns << "," 
            << prof.total_ns << "\n";
    }
    
    out.close();
    std::cout << "Profiling complete. Data saved to profile_data.csv" << std::endl;
    
    return 0;
}
