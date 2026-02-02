#include <iomanip>
#include <iostream>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <stdexcept>
#include <vector>

#include "benchmark.hpp"
#include "matrix.hpp"

#define CACHE_LINE_SIZE 64 // x86-64 CPUs have 64 bytes for cache line size
#define BENCHMARK_RUNS 10
#define BENCHMARK_WARMUP 1

/// @brief Generate an integer array randomly filled with values between -100 and 100.
/// @param size the size of the array
/// @return the generated array
std::vector<long> generate_integer_array(size_t size) {
    
    std::random_device seed;
    std::mt19937 generator(seed());
    std::uniform_int_distribution<long> distribution(-100, 100);

    std::vector<long> result(size);
    for(long &val : result) {
        val = distribution(generator);
    }
    return result;
}

/// @brief Generate a double array randomly filled with values between -1.0 and 1.0.
/// @param size the size of the array
/// @return the generated array
std::vector<double> generate_real_array(size_t size) {
    
    std::random_device seed;
    std::mt19937 generator(seed());
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);

    std::vector<double> result(size);
    for(double &val : result) {
        val = distribution(generator);
    }
    return result;
}

/// @brief Perform a matrix-vector multiplication
/// @tparam T the field type of the matrix and vector 
/// @param matrix the matrix to multiply
/// @param array the vector to multiply
/// @return the result of the multiplication
template<typename T> std::unique_ptr<T[]> matrix_vector_multiplication(csr_matrix<T>& matrix, std::vector<T>& array) {
    
    alignas(CACHE_LINE_SIZE) T* result = new T[array.size()];

    for(size_t i = 0; i < array.size(); i++) {
        long row_start = matrix.row_indices[i];
        long row_end = matrix.row_indices[i + 1];
        long accumulator = 0;
        for(long value_index = row_start; value_index < row_end; value_index++) {
            accumulator += (matrix.values[value_index] * array[i]);
        }
        result[i] = accumulator;
    }

    return std::unique_ptr<T[]>(result);
}

void print_results(benchmark_results results) {
    std::cout << "RESULTS:\n";
    std::cout << "Fastest: " << results.fastest_time << "ms\n";
    std::cout << "Slowest: " << results.slowest_time << "ms\n";
    std::cout << "Average: " << results.average_time << "ms\n";
    std::cout << "90th percentile: " << results.ninetieth_percentile_time << "ms\n";
    std::cout << "Run times: ";
    for(long v : results.times) {
        std::cout << v << "ms, ";
    }
    std::cout << "\n";
}

void write_results_to_file(std::string filename, benchmark_results results) {
    std::ofstream output(filename);
    output << "fastest,slowest,ninetieth\n";
    output << results.fastest_time << "," << results.slowest_time << "," << results.ninetieth_percentile_time << "\n";
    output.flush();
    output.close();
}

int main(int argc, char* argv[]) {

    if(argc != 3) {
        std::cout << "USAGE:" << argv[0] << "<matrix-path> <results-output-path>\n";
        return 0;
    }

    std::string matrix_path(argv[2]);
    std::string results_path(argv[3]);
    std::ifstream file(matrix_path);
    std::string header;
    std::getline(file, header);
    matrix_metadata metadata = identify_matrix(header);

    switch(metadata.field_values) {
        case field_type::integer: {

            std::cout << "Loading matrix...\n";
            csr_matrix<long> m = read_integer_matrix(file, metadata);
            std::cout << "Matrix sparsity: " << std::setprecision(10) << m.sparsity << "%\n";

            std::cout << "Generating vector...\n";
            std::vector<long> test_arr = generate_integer_array(m.n_rows);

            // This is the part that needs to be parallelized with MPI.
            // Worker processes are assigned rows based on their rank -> rank == row_index % num_processes.
            // How to get the matrix to the worker processes?
            // - option 1: the whole matrix is broadcasted to the collective, each process only calculates on its assigned rows. Big network bottleneck if matrix is large/has many NNZ values.
            // - option 2: the matrix is split into sections by the master and each worker receives only the section it will work on. Matrix splitting is done sequentially and not in parallel, need to send data to each worker which might still have a big network bottleneck.
            // The benchmark funcition logic needs to be modified to properly accumulate the result of all worker processes. We don't necessarily care about the output as long as the computation is correct, but the accumulation needs to be counted in the benchmarking time.
            std::function<void ()> benchmark_function;
            benchmark_function = [&m, &test_arr]() { 
                matrix_vector_multiplication(m, test_arr); 
            };

            std::cout << "Starting benchmark...\n";
            benchmark_results results = benchmark(benchmark_function, BENCHMARK_RUNS, BENCHMARK_WARMUP);
            print_results(results);
            write_results_to_file(results_path, results);
            break;
        }

        case field_type::real: {
            
            std::cout << "Loading matrix...\n";
            csr_matrix<double> m = read_real_matrix(file, metadata);
            std::cout << "Matrix sparsity: " << std::setprecision(10) << m.sparsity << "%\n";

            std::cout << "Generating vector...\n";
            std::vector<double> test_arr = generate_real_array(m.n_rows);

            std::function<void ()> benchmark_function;
            benchmark_function = [&m, &test_arr]() { 
                matrix_vector_multiplication(m, test_arr); 
            };
        
            std::cout << "Starting benchmark...\n";
            benchmark_results results = benchmark(benchmark_function, BENCHMARK_RUNS, BENCHMARK_WARMUP);
            print_results(results);
            write_results_to_file(results_path, results);
            break;
        }
    }

    return 0;
}