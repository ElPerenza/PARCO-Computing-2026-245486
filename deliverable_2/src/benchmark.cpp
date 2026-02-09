#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <tuple>
#include <vector>

#include "benchmark.hpp"

/**********************/
/***  Private API  ****/
/**********************/

/// @brief Compute the fastest, slowest, average and 90th percentile time for the given times.
/// @param stats the statisitics object containing the list of times and where the computed stats will be stored.
/// @return the passed `time_statistics` object
time_statistics compute_statistics(time_statistics& stats) {

    stats.fastest = *std::min_element(stats.times.begin(), stats.times.end());
    stats.slowest = *std::max_element(stats.times.begin(), stats.times.end());
    stats.average = std::accumulate(stats.times.begin(), stats.times.end(), 0) / stats.times.size();

    std::vector<double> sorted_times = stats.times; // this calls the copy costructor
    std::sort(sorted_times.begin(), sorted_times.end());
    stats.ninetieth_percentile = sorted_times[std::round<int>(90.0 / 100.0 * sorted_times.size()) - 1]; // nearest-rank percentile
    
    return stats;
}

/*********************/
/***  Public API  ****/
/*********************/

benchmark_results benchmark(std::function<std::tuple<double, double, int> ()> f, int runs, int warmup_runs) {

    benchmark_results results;

    for(int i = 0; i < warmup_runs; i++) {
        f();
    }

    for(int i = 0; i < runs; i++) {
        std::tuple<double, double, int> times_flops = f();
        results.communication_time.times.push_back(std::get<0>(times_flops));
        results.computation_time.times.push_back(std::get<1>(times_flops));
        results.flops.push_back(std::get<2>(times_flops) / (std::get<1>(times_flops) / 1000.0));
    }

    compute_statistics(results.communication_time);
    compute_statistics(results.computation_time);
    results.flops_average = std::accumulate(results.flops.begin(), results.flops.end(), 0) / results.flops.size();
    return results;
}