#include <algorithm>
#include <cmath>
#include <chrono>
#include <functional>
#include <numeric>
#include <vector>

#include "benchmark.hpp"

benchmark_results benchmark(std::function<void ()> f, int runs) {

    benchmark_results results;

    for(int i = 0; i < runs; i++) {
        std::chrono::time_point start = std::chrono::high_resolution_clock::now();

        f();

        std::chrono::time_point end = std::chrono::high_resolution_clock::now();
        long duration_millis = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        results.times.push_back(duration_millis);
    }

    results.fastest_time = *std::min_element(results.times.begin(), results.times.end());
    results.slowest_time = *std::max_element(results.times.begin(), results.times.end());
    results.average_time = std::accumulate(results.times.begin(), results.times.end(), 0) / runs;
    std::vector<long> sorted_times = results.times; // this calls the copy costructor
    std::sort(sorted_times.begin(), sorted_times.end());

    results.ninetieth_percentile_time = sorted_times[std::round<int>(90.0 / 100.0 * sorted_times.size()) - 1]; // nearest-rank percentile

    return results;
}