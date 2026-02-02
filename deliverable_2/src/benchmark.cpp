#include <algorithm>
#include <cmath>
#include <chrono>
#include <functional>
#include <numeric>
#include <vector>

#include "benchmark.hpp"

benchmark_results benchmark(std::function<void ()> f, int runs, int warmup_runs) {

    benchmark_results results;

    for(int i = 0; i < warmup_runs; i++) {
        f();
    }

    for(int i = 0; i < runs; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        f();
        auto end = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::milli> duration = end - start;
        results.times.push_back(duration.count());
    }

    results.fastest_time = *std::min_element(results.times.begin(), results.times.end());
    results.slowest_time = *std::max_element(results.times.begin(), results.times.end());
    results.average_time = std::accumulate(results.times.begin(), results.times.end(), 0) / runs;
    std::vector<double> sorted_times = results.times; // this calls the copy costructor
    std::sort(sorted_times.begin(), sorted_times.end());

    results.ninetieth_percentile_time = sorted_times[std::round<int>(90.0 / 100.0 * sorted_times.size()) - 1]; // nearest-rank percentile

    return results;
}