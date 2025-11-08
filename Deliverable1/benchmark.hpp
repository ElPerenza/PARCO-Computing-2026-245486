#ifndef BENCHMARK_HPP
#define BENCHMARK_HPP

#include <functional>
#include <vector>

struct benchmark_results {
    long average_time;
    long fastest_time;
    long slowest_time;
    long ninetieth_percentile_time;
    std::vector<long> times;
};

/// @brief Execute the given function a number of times equal to `runs` and report the execution times.
/// @param f the function to benchmark
/// @param runs the number of times to execute the function
/// @return the benchmark results
benchmark_results benchmark(std::function<void ()> f, int runs);

#endif