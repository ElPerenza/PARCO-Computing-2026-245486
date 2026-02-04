#include <iomanip>
#include <iostream>
#include <fstream>
#include <memory>
#include <mpi.h>
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
template<typename T> std::unique_ptr<T[]> matrix_vector_multiplication(const csr_matrix<T>& matrix, const std::vector<T>& array) {
    
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

void extract_process_partition(const csr_matrix<long>& old_matrix, const std::vector<long>& old_array, int process_rank, int n_processes, csr_matrix<long>& new_matrix, std::vector<long>& new_array) {

    new_matrix.values.push_back(0);
    long values_read = 0;
    for(size_t i = 0; i < old_matrix.n_rows; i++) {
        if((i % n_processes) == process_rank) {

            new_array.push_back(old_array[i]);

            long row_start = old_matrix.row_indices[i];
            long row_end = old_matrix.row_indices[i + 1];
            for(long j = row_start; j < row_end; j++) {
                new_matrix.values.push_back(old_matrix.values[j]);
                values_read++;
            }
            new_matrix.row_indices.push_back(values_read);
        }
    }
}

void distribute_data_to_processes(const csr_matrix<long>& matrix, const std::vector<long>& array, const MPI_Comm& communicator) {

    int my_rank;
    MPI_Comm_rank(communicator, &my_rank);
    int n_processes;
    MPI_Comm_size(communicator, &n_processes);
    for(int process_rank = 0; process_rank < n_processes; process_rank++) {          
        
        // do not distribute data to process that called this function, as the calling process will do that itself
        if(process_rank == my_rank) {
            continue;
        }

        std::vector<long> partial_array;
        csr_matrix<long> partial_matrix;
        extract_process_partition(matrix, array, my_rank, n_processes, partial_matrix, partial_array);

        //TODO: I don't like sending size_t values and just saying they're unsigned longs, even if it's technically the same on x64 GCC
        // Send row index array of the CSR matrix
        size_t row_indices_size = partial_matrix.row_indices.size();
        MPI_Send(&row_indices_size, 1, MPI_UNSIGNED_LONG, process_rank, 0, communicator);
        MPI_Send(partial_matrix.row_indices.data(), row_indices_size, MPI_LONG, process_rank, 0, communicator);
        // Send values array of the CSR matrix
        size_t values_size = partial_matrix.values.size();
        MPI_Send(&values_size, 1, MPI_UNSIGNED_LONG, process_rank, 0, communicator);
        MPI_Send(partial_matrix.values.data(), values_size, MPI_LONG, process_rank, 0, communicator);
        // Send the dense array
        size_t array_size = partial_array.size();
        MPI_Send(&array_size, 1, MPI_UNSIGNED_LONG, process_rank, 0, communicator);
        MPI_Send(partial_array.data(), array_size, MPI_LONG, process_rank, 0, communicator);
    }
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
        std::cout << "USAGE: " << argv[0] << " <matrix-path> <results-output-path>\n";
        return 0;
    }
    if(MPI_Init(&argc, &argv) != MPI_SUCCESS) {
        std::cerr << "Failed to initialize MPI!\n";
        return 1;
    }

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(rank == 0) {
        // Master code here

        int n_processes;
        MPI_Comm_size(MPI_COMM_WORLD, &n_processes);

        std::string matrix_path(argv[1]);
        std::string results_path(argv[2]);
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
                std::function<void ()> benchmark_function = [&m, &test_arr, &n_processes, &rank]() { 

                    std::vector<long> rank_0_array;
                    csr_matrix<long> rank_0_matrix;
                    extract_process_partition(m, test_arr, rank, n_processes, rank_0_matrix, rank_0_array);

                    distribute_data_to_processes(m, test_arr, MPI_COMM_WORLD);

                    matrix_vector_multiplication(rank_0_matrix, rank_0_array);
                    MPI_Barrier(MPI_COMM_WORLD);
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

    } else {
        // Workers code here

        std::function<void ()> fn = [&rank]() {

            csr_matrix<long> matrix;

            size_t row_indices_size;
            MPI_Recv(&row_indices_size, 1, MPI_UNSIGNED_LONG, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            std::unique_ptr<long[]> row_indices_pointer = std::make_unique<long[]>(row_indices_size);
            MPI_Recv(row_indices_pointer.get(), row_indices_size, MPI_UNSIGNED_LONG, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            matrix.row_indices = std::vector<long>(row_indices_pointer.get(), row_indices_pointer.get() + row_indices_size);

            size_t values_size;
            MPI_Recv(&values_size, 1, MPI_UNSIGNED_LONG, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            std::unique_ptr<long[]> a = std::make_unique<long[]>(values_size);
            MPI_Recv(a.get(), values_size, MPI_UNSIGNED_LONG, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            matrix.values = std::vector<long>(a.get(), a.get() + values_size);

            size_t array_size;
            MPI_Recv(&array_size, 1, MPI_UNSIGNED_LONG, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            std::unique_ptr<long[]> b = std::make_unique<long[]>(array_size);
            MPI_Recv(b.get(), array_size, MPI_UNSIGNED_LONG, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            std::vector<long> array = std::vector<long>(b.get(), b.get() + array_size);

            matrix_vector_multiplication(matrix, array);
            MPI_Barrier(MPI_COMM_WORLD);
        };

        benchmark(fn, BENCHMARK_RUNS, BENCHMARK_WARMUP);
    }

    MPI_Finalize();
    return 0;
}