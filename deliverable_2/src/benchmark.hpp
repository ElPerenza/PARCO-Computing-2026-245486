#ifndef BENCHMARK_HPP
#define BENCHMARK_HPP

#include <functional>
#include <tuple>
#include <vector>

struct time_statistics {
    double average;
    double fastest;
    double slowest;
    double ninetieth_percentile;
    std::vector<double> times;
};

struct benchmark_results {
    time_statistics communication_time;
    time_statistics computation_time;
    std::vector<double> flops;
    double flops_average;
};

/// @brief Execute the given function a number of times equal to `runs` and report statistics on the execution times.
/// @param f the function to benchmark. The function must return a tuple of 2 doubles and 1 int - the first double denoting the execution time used for MPI communication, the second the execution time used for actual computation, and the int the flops of the computation.
/// @param runs the number of times to execute the function
/// @param warmup_runs the number of times to execute the function before starting the benchmark
/// @return the benchmark results
benchmark_results benchmark(std::function<std::tuple<double, double, int> ()> f, int runs, int warmup_runs);

#endif