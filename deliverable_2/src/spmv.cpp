#include <chrono>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <memory>
#include <mpi.h>
#include <random>
#include <string>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "benchmark.hpp"
#include "matrix.hpp"

#define CACHE_LINE_SIZE 64 // x86-64 CPUs have 64 bytes for cache line size
#define BENCHMARK_RUNS 10
#define BENCHMARK_WARMUP 1

#define ROOT_RANK 0

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
/// @param flops
/// @return the result of the multiplication
template<typename T> std::unique_ptr<T[]> matrix_vector_multiplication(partial_csr_matrix<T>& matrix, std::vector<T>& array, int& flops) {

    alignas(CACHE_LINE_SIZE) T* result = new T[array.size()];

    for(size_t i = 0; i < array.size(); i++) {
        long row_start = matrix.row_indices[i];
        long row_end = matrix.row_indices[i + 1];
        long accumulator = 0;
        for(long value_index = row_start; value_index < row_end; value_index++) {
            accumulator += (matrix.values[value_index] * array[i]);
            flops++;
        }
        result[i] = accumulator;
    }

    return std::unique_ptr<T[]>(result);
}

std::vector<int> generate_displacements(const int block_sizes[], size_t length) {
    std::vector<int> displacements(length, 0);
    for(size_t i = 1; i < displacements.size(); i++) {
        displacements[i] = displacements[i - 1] + block_sizes[i - 1];
    }
    return displacements;
}

std::vector<int> generate_displacements(const std::vector<int>& block_sizes) {
    return generate_displacements(block_sizes.data(), block_sizes.size());
}

template<typename T> void scatter_spmv_data(
        int rows_counts[], const T array_blocks[], const long row_indices_blocks[], 
        int values_counts[], const T values_blocks[], 
        MPI_Datatype datatype, int root_rank, MPI_Comm communicator,
        std::vector<T>& received_array, partial_csr_matrix<T>& received_matrix
) {

    int n_processes;
    MPI_Comm_size(communicator, &n_processes);
    std::vector<int> displacements;

    // In order:
    // 1. Scatter number of rows assigned to each process
    // 2. Scatter blocks of dense array assigned to each process
    // 3. Scatter blocks of row indices assigned to each process
    // 4. Scatter number of values assigned to each process
    // 5. Scatter blocks of values assigned to each process
    
    if(rows_counts != nullptr) {
        displacements = generate_displacements(rows_counts, n_processes);
    }
    int n_rows;
    MPI_Scatter(rows_counts, 1, MPI_INT, &n_rows, 1, MPI_INT, root_rank, communicator);
    std::unique_ptr<T[]> array_ptr(new T[n_rows]);
    MPI_Scatterv(array_blocks, rows_counts, displacements.data(), datatype, array_ptr.get(), n_rows, datatype, root_rank, communicator);
    
    // row_indices is number of rows + 1
    // Null check becourse rows_counts is used only by the scatter root, not receivers
    if(rows_counts != nullptr) {
        for(int i = 0; i < n_processes; i++) {
            rows_counts[i]++;
        }
        displacements = generate_displacements(rows_counts, n_processes);
    }
    std::unique_ptr<long[]> row_indices_ptr(new long[n_rows + 1]);
    MPI_Scatterv(row_indices_blocks, rows_counts, displacements.data(), MPI_LONG, row_indices_ptr.get(), n_rows + 1, MPI_LONG, root_rank, communicator);
    
    if(values_counts != nullptr) {
        displacements = generate_displacements(values_counts, n_processes);
    }
    int n_values;
    MPI_Scatter(values_counts, 1, MPI_INT, &n_values, 1, MPI_INT, root_rank, communicator);
    std::unique_ptr<T[]> values_ptr(new T[n_values]);
    MPI_Scatterv(values_blocks, values_counts, displacements.data(), datatype, values_ptr.get(), n_values, datatype, root_rank, communicator);

    received_array = std::vector<T>(array_ptr.get(), array_ptr.get() + n_rows);
    received_matrix.row_indices = std::vector<long>(row_indices_ptr.get(), row_indices_ptr.get() + n_rows + 1);
    received_matrix.values = std::vector<T>(values_ptr.get(), values_ptr.get() + n_values);
}

template<typename T> void distribute_data_to_processes(
        const csr_matrix<T>& matrix, const std::vector<T>& array, 
        MPI_Datatype datatype, MPI_Comm communicator, 
        partial_csr_matrix<T>& distributed_matrix, std::vector<T>& distributed_array, std::vector<int>& n_rows_per_process
) {

    int my_rank, n_processes;
    MPI_Comm_rank(communicator, &my_rank);
    MPI_Comm_size(communicator, &n_processes);

    std::vector<int> rows_per_process; // number of rows for each process rank
    std::vector<int> values_per_process; // number of values for each process rank
    partial_csr_matrix<T> partitioned_matrix;
    std::vector<T> partitioned_array;

    for(int process_rank = 0; process_rank < n_processes; process_rank++) {          

        int rows_read = 0, values_read = 0;
        partitioned_matrix.row_indices.push_back(0);
        for(size_t i = 0; i < matrix.n_rows; i++) {
            if((i % n_processes) == process_rank) {

                long row_start = matrix.row_indices[i];
                long row_end = matrix.row_indices[i + 1];
                for(long j = row_start; j < row_end; j++) {
                    partitioned_matrix.values.push_back(matrix.values[j]);
                    values_read++;
                }

                partitioned_matrix.row_indices.push_back(values_read);
                partitioned_array.push_back(array[i]);
                rows_read++;
            }
        }

        rows_per_process.push_back(rows_read);
        values_per_process.push_back(values_read);
    }

    n_rows_per_process = rows_per_process; // copy constructor
    scatter_spmv_data<T>(rows_per_process.data(), partitioned_array.data(), partitioned_matrix.row_indices.data(), values_per_process.data(), partitioned_matrix.values.data(), datatype, my_rank, communicator, distributed_array, distributed_matrix);
}

template<typename T> std::unique_ptr<T[]> reorder_results(const T result_blocks[], const std::vector<int>& rows_per_process, long n_rows) {
    
    std::unique_ptr<T[]> final_result(new T[n_rows]);
    int start_index = 0;
    size_t n_processes = rows_per_process.size();
    for(long process_rank = 0; process_rank < n_processes; process_rank++) {

        for(int j = start_index; j < start_index + rows_per_process[process_rank]; j++) {
            int local_index = j - start_index;
            final_result[(local_index * n_processes) + process_rank] = result_blocks[j];
        }

        start_index += rows_per_process[process_rank];
    }

    return final_result;
}

void print_results(benchmark_results results) {
    std::cout << "RESULTS:\n";
    std::cout << "Fastest: " << results.computation_time.fastest << "ms\n";
    std::cout << "Slowest: " << results.computation_time.slowest << "ms\n";
    std::cout << "Average: " << results.computation_time.average << "ms\n";
    std::cout << "90th percentile: " << results.computation_time.ninetieth_percentile << "ms\n";
    std::cout << "Computation times: ";
    for(double v : results.computation_time.times) {
        std::cout << v << "ms, ";
    }
    std::cout << "\n";
    std::cout << "FLOPS: ";
    for(double v : results.flops) {
        std::cout << v << ", ";
    }
    std::cout << "\n";
    std::cout << "Communication times: ";
    for(double v : results.communication_time.times) {
        std::cout << v << "ms, ";
    }
    std::cout << "\n";
}

void write_results_to_file(std::string filename, benchmark_results results) {
    std::ofstream output(filename);
    output << "comp_fastest,comp_slowest,comp_average,comp_ninetieth,comm_fastest,comm_slowest,comm_average,comm_ninetieth,flops_average\n";
    output << results.computation_time.fastest << "," << results.computation_time.slowest << "," << results.computation_time.average << "," << results.computation_time.ninetieth_percentile << ",";
    output << results.communication_time.fastest << "," << results.communication_time.slowest << "," << results.communication_time.average << "," << results.communication_time.ninetieth_percentile << ",";
    output << results.flops_average << "\n";
    output.flush();
    output.close();
}

void print_usage(std::string program_name) {
    std::cout << "USAGE: " << program_name << " [sequential | parallel] [<matrix-path> | generated] <results-output-path> <num-rows> <sparsity>\n";
    std::cout << "Note: <num-rows> and <sparsity> must only be provided if the second parameter is \"generated\"\n";
}

int main(int argc, char* argv[]) {

    if(argc != 4 && argc != 6) {
        print_usage(argv[0]);
        return 0;
    }
    std::string matrix_path(argv[2]);
    if(matrix_path == "generated" && argc != 6) {
        print_usage(argv[0]);
        return 0;
    }
    std::string execution_mode(argv[1]);
    if(execution_mode != "sequential" && execution_mode != "parallel") {
        std::cout << "Invalid execution mode (must be 'sequential' or 'parallel'): " << execution_mode << "\n";
        return -1;
    }
    std::string results_path(argv[3]);

    if(MPI_Init(&argc, &argv) != MPI_SUCCESS) {
        std::cerr << "Failed to initialize MPI!\n";
        return 1;
    }

    int n_processes, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &n_processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if(matrix_path == "generated") {

        if(rank == ROOT_RANK) {

            std::cout << "Generating matrix...\n";
            csr_matrix<double> m = generate_real_matrix(std::stol(argv[4]), std::stod(argv[5]));
            std::cout << "Matrix sparsity: " << std::setprecision(10) << m.sparsity << "%\n";

            std::cout << "Generating vector...\n";
            std::vector<double> test_arr = generate_real_array(m.n_rows);

            std::function<std::tuple<double, double, int> ()> benchmark_function;

            if(execution_mode == "sequential") {

                benchmark_function = [&m, &test_arr]() {
                    auto start = std::chrono::high_resolution_clock::now();

                    int flops = 0;
                    matrix_vector_multiplication(m, test_arr, flops);

                    auto end = std::chrono::high_resolution_clock::now();
                    const std::chrono::duration<double, std::milli> duration = end - start;
                    return std::make_tuple(0, duration.count(), flops);
                };
                
            } else if(execution_mode == "parallel") {

                benchmark_function = [&m, &test_arr, &n_processes, &rank]() {
                    auto start = std::chrono::high_resolution_clock::now();

                    partial_csr_matrix<double> root_rank_matrix;
                    std::vector<double> root_rank_array;
                    std::vector<int> rows_per_process;
                    distribute_data_to_processes<double>(m, test_arr, MPI_DOUBLE, MPI_COMM_WORLD, root_rank_matrix, root_rank_array, rows_per_process);

                    int flops = 0, total_flops = 0;
                    // Barriers needed to make sure the computation is timed properly
                    MPI_Barrier(MPI_COMM_WORLD);
                    auto computation_start = std::chrono::high_resolution_clock::now();
                    std::unique_ptr<double[]> result = matrix_vector_multiplication(root_rank_matrix, root_rank_array, flops);
                    MPI_Barrier(MPI_COMM_WORLD);
                    auto computation_end = std::chrono::high_resolution_clock::now();

                    std::unique_ptr<double[]> result_blocks(new double[m.n_rows]);
                    std::vector<int> displacements = generate_displacements(rows_per_process);
                    MPI_Gatherv(result.get(), root_rank_array.size(), MPI_DOUBLE, result_blocks.get(), rows_per_process.data(), displacements.data(), MPI_DOUBLE, ROOT_RANK, MPI_COMM_WORLD);
                    MPI_Reduce(&flops, &total_flops, 1, MPI_INT, MPI_SUM, ROOT_RANK, MPI_COMM_WORLD);

                    reorder_results(result_blocks.get(), rows_per_process, m.n_rows);

                    MPI_Barrier(MPI_COMM_WORLD);
                    auto end = std::chrono::high_resolution_clock::now();
                    const std::chrono::duration<double, std::milli> computation_duration = computation_end - computation_start;
                    const std::chrono::duration<double, std::milli> total_duration = end - start;
                    return std::make_tuple((total_duration - computation_duration).count(), computation_duration.count(), total_flops);
                };

            }

            std::cout << "Starting benchmark...\n";
            benchmark_results results = benchmark(benchmark_function, BENCHMARK_RUNS, BENCHMARK_WARMUP);
            print_results(results);
            write_results_to_file(results_path, results);

        } else {

            if(execution_mode == "sequential") {
                MPI_Finalize();
                return 0;
            }

            std::function<std::tuple<double, double, int> ()> fn = [&rank, &n_processes]() {

                partial_csr_matrix<double> matrix;
                std::vector<double> array;
                scatter_spmv_data<double>(nullptr, nullptr, nullptr, nullptr, nullptr, MPI_LONG, ROOT_RANK, MPI_COMM_WORLD, array, matrix);

                int flops = 0;
                MPI_Barrier(MPI_COMM_WORLD);
                std::unique_ptr<double[]> result = matrix_vector_multiplication(matrix, array, flops);
                MPI_Barrier(MPI_COMM_WORLD);

                MPI_Gatherv(result.get(), array.size(), MPI_DOUBLE, nullptr, nullptr, nullptr, MPI_DOUBLE, ROOT_RANK, MPI_COMM_WORLD);
                MPI_Reduce(&flops, nullptr, 1, MPI_INT, MPI_SUM, ROOT_RANK, MPI_COMM_WORLD);

                MPI_Barrier(MPI_COMM_WORLD);
                return std::make_tuple(-1, -1, -1);
            };

            benchmark(fn, BENCHMARK_RUNS, BENCHMARK_WARMUP);

        }

    } else {

        std::ifstream file(matrix_path);
        std::string header;
        std::getline(file, header);
        matrix_metadata metadata = identify_matrix(header);

        switch(metadata.field_values) {
            case field_type::integer: {

                if(rank == ROOT_RANK) {

                    std::cout << "Loading matrix...\n";
                    csr_matrix<long> m = read_integer_matrix(file, metadata);
                    std::cout << "Matrix sparsity: " << std::setprecision(10) << m.sparsity << "%\n";

                    std::cout << "Generating vector...\n";
                    std::vector<long> test_arr = generate_integer_array(m.n_rows);

                    std::function<std::tuple<double, double, int> ()> benchmark_function;

                    if(execution_mode == "sequential") {

                        benchmark_function = [&m, &test_arr]() {
                            auto start = std::chrono::high_resolution_clock::now();

                            int iops = 0;
                            matrix_vector_multiplication(m, test_arr, iops);

                            auto end = std::chrono::high_resolution_clock::now();
                            const std::chrono::duration<double, std::milli> duration = end - start;
                            return std::make_tuple(0, duration.count(), iops);
                        };
                        
                    } else if(execution_mode == "parallel") {

                        benchmark_function = [&m, &test_arr, &n_processes, &rank]() { 
                            auto start = std::chrono::high_resolution_clock::now();

                            partial_csr_matrix<long> root_rank_matrix;
                            std::vector<long> root_rank_array;
                            std::vector<int> rows_per_process;
                            distribute_data_to_processes<long>(m, test_arr, MPI_LONG, MPI_COMM_WORLD, root_rank_matrix, root_rank_array, rows_per_process);

                            int iops = 0, total_iops = 0;
                            MPI_Barrier(MPI_COMM_WORLD);
                            auto computation_start = std::chrono::high_resolution_clock::now();
                            std::unique_ptr<long[]> result = matrix_vector_multiplication(root_rank_matrix, root_rank_array, iops);
                            MPI_Barrier(MPI_COMM_WORLD);
                            auto computation_end = std::chrono::high_resolution_clock::now();

                            std::unique_ptr<long[]> result_blocks(new long[m.n_rows]);
                            std::vector<int> displacements = generate_displacements(rows_per_process);
                            MPI_Gatherv(result.get(), root_rank_array.size(), MPI_LONG, result_blocks.get(), rows_per_process.data(), displacements.data(), MPI_LONG, ROOT_RANK, MPI_COMM_WORLD);
                            MPI_Reduce(&iops, &total_iops, 1, MPI_INT, MPI_SUM, ROOT_RANK, MPI_COMM_WORLD);

                            reorder_results(result_blocks.get(), rows_per_process, m.n_rows);

                            MPI_Barrier(MPI_COMM_WORLD);
                            auto end = std::chrono::high_resolution_clock::now();
                            const std::chrono::duration<double, std::milli> computation_duration = computation_end - computation_start;
                            const std::chrono::duration<double, std::milli> total_duration = end - start;
                            return std::make_tuple((total_duration - computation_duration).count(), computation_duration.count(), total_iops);
                        };

                    }

                    std::cout << "Starting benchmark...\n";
                    benchmark_results results = benchmark(benchmark_function, BENCHMARK_RUNS, BENCHMARK_WARMUP);
                    print_results(results);
                    write_results_to_file(results_path, results);

                } else {

                    if(execution_mode == "sequential") {
                        MPI_Finalize();
                        return 0;
                    }

                    std::function<std::tuple<double, double, int> ()> fn = [&rank, &n_processes]() {

                        partial_csr_matrix<long> matrix;
                        std::vector<long> array;
                        scatter_spmv_data<long>(nullptr, nullptr, nullptr, nullptr, nullptr, MPI_LONG, ROOT_RANK, MPI_COMM_WORLD, array, matrix);

                        int iops = 0;
                        MPI_Barrier(MPI_COMM_WORLD);
                        std::unique_ptr<long[]> result = matrix_vector_multiplication(matrix, array, iops);
                        MPI_Barrier(MPI_COMM_WORLD);

                        MPI_Gatherv(result.get(), array.size(), MPI_LONG, nullptr, nullptr, nullptr, MPI_LONG, ROOT_RANK, MPI_COMM_WORLD);
                        MPI_Reduce(&iops, nullptr, 1, MPI_INT, MPI_SUM, ROOT_RANK, MPI_COMM_WORLD);

                        MPI_Barrier(MPI_COMM_WORLD);
                        return std::make_tuple(-1, -1, -1);
                    };

                    benchmark(fn, BENCHMARK_RUNS, BENCHMARK_WARMUP);

                }
                
                break;
            }

            case field_type::real: {

                if(rank == ROOT_RANK) {

                    std::cout << "Loading matrix...\n";
                    csr_matrix<double> m = read_real_matrix(file, metadata);
                    std::cout << "Matrix sparsity: " << std::setprecision(10) << m.sparsity << "%\n";

                    std::cout << "Generating vector...\n";
                    std::vector<double> test_arr = generate_real_array(m.n_rows);

                    std::function<std::tuple<double, double, int> ()> benchmark_function;

                    if(execution_mode == "sequential") {

                        benchmark_function = [&m, &test_arr]() {
                            auto start = std::chrono::high_resolution_clock::now();

                            int flops = 0;
                            matrix_vector_multiplication(m, test_arr, flops);

                            auto end = std::chrono::high_resolution_clock::now();
                            const std::chrono::duration<double, std::milli> duration = end - start;
                            return std::make_tuple(0, duration.count(), flops);
                        };
                        
                    } else if(execution_mode == "parallel") {

                        benchmark_function = [&m, &test_arr, &n_processes, &rank]() {
                            auto start = std::chrono::high_resolution_clock::now();

                            partial_csr_matrix<double> root_rank_matrix;
                            std::vector<double> root_rank_array;
                            std::vector<int> rows_per_process;
                            distribute_data_to_processes<double>(m, test_arr, MPI_DOUBLE, MPI_COMM_WORLD, root_rank_matrix, root_rank_array, rows_per_process);

                            int flops = 0, total_flops = 0;
                            // Barriers needed to make sure the computation is timed properly
                            MPI_Barrier(MPI_COMM_WORLD);
                            auto computation_start = std::chrono::high_resolution_clock::now();
                            std::unique_ptr<double[]> result = matrix_vector_multiplication(root_rank_matrix, root_rank_array, flops);
                            MPI_Barrier(MPI_COMM_WORLD);
                            auto computation_end = std::chrono::high_resolution_clock::now();

                            std::unique_ptr<double[]> result_blocks(new double[m.n_rows]);
                            std::vector<int> displacements = generate_displacements(rows_per_process);
                            MPI_Gatherv(result.get(), root_rank_array.size(), MPI_DOUBLE, result_blocks.get(), rows_per_process.data(), displacements.data(), MPI_DOUBLE, ROOT_RANK, MPI_COMM_WORLD);
                            MPI_Reduce(&flops, &total_flops, 1, MPI_INT, MPI_SUM, ROOT_RANK, MPI_COMM_WORLD);

                            reorder_results(result_blocks.get(), rows_per_process, m.n_rows);

                            MPI_Barrier(MPI_COMM_WORLD);
                            auto end = std::chrono::high_resolution_clock::now();
                            const std::chrono::duration<double, std::milli> computation_duration = computation_end - computation_start;
                            const std::chrono::duration<double, std::milli> total_duration = end - start;
                            return std::make_tuple((total_duration - computation_duration).count(), computation_duration.count(), total_flops);
                        };

                    }

                    std::cout << "Starting benchmark...\n";
                    benchmark_results results = benchmark(benchmark_function, BENCHMARK_RUNS, BENCHMARK_WARMUP);
                    print_results(results);
                    write_results_to_file(results_path, results);

                } else {

                    if(execution_mode == "sequential") {
                        MPI_Finalize();
                        return 0;
                    }

                    std::function<std::tuple<double, double, int> ()> fn = [&rank, &n_processes]() {

                        partial_csr_matrix<double> matrix;
                        std::vector<double> array;
                        scatter_spmv_data<double>(nullptr, nullptr, nullptr, nullptr, nullptr, MPI_LONG, ROOT_RANK, MPI_COMM_WORLD, array, matrix);

                        int flops = 0;
                        MPI_Barrier(MPI_COMM_WORLD);
                        std::unique_ptr<double[]> result = matrix_vector_multiplication(matrix, array, flops);
                        MPI_Barrier(MPI_COMM_WORLD);

                        MPI_Gatherv(result.get(), array.size(), MPI_DOUBLE, nullptr, nullptr, nullptr, MPI_DOUBLE, ROOT_RANK, MPI_COMM_WORLD);
                        MPI_Reduce(&flops, nullptr, 1, MPI_INT, MPI_SUM, ROOT_RANK, MPI_COMM_WORLD);

                        MPI_Barrier(MPI_COMM_WORLD);
                        return std::make_tuple(-1, -1, -1);
                    };

                    benchmark(fn, BENCHMARK_RUNS, BENCHMARK_WARMUP);

                }
            }
        }
    }

    MPI_Finalize();
    return 0;
}