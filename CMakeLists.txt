cmake_minimum_required(VERSION 3.10)
project(learned_b_tree)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")

# Threads
find_package(Threads REQUIRED)

include_directories(include benchmarks)

add_executable(benchmark_runner benchmarks/main.cpp)
target_link_libraries(benchmark_runner PRIVATE Threads::Threads)

add_executable(profiler_runner benchmarks/profiler_runner.cpp)
target_link_libraries(profiler_runner PRIVATE Threads::Threads)

add_executable(adversarial_search benchmarks/adversarial_search.cpp)
target_link_libraries(adversarial_search PRIVATE Threads::Threads)
