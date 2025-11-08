#include <iostream>
#include <fstream>
#include <omp.h>
#include <random>
#include <string>
#include <stdexcept>
#include <vector>

#include "benchmark.hpp"
#include "matrix.hpp"

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
template<typename T> std::vector<T> matrix_vector_multiplication(csr_matrix<T>& matrix, std::vector<T>& array) {
    
    std::vector<T> result(array.size());

    for(size_t i = 0; i < array.size(); i++) {
        long row_start = matrix.row_indices[i];
        long row_end = matrix.row_indices[i + 1];
        for(long value_index = row_start; value_index < row_end; value_index++) {
            result[i] += (matrix.values[value_index] * array[i]);
        }
    }

    return result;
}

/// @brief Perform a matrix-vector multiplication
/// @tparam T the field type of the matrix and vector 
/// @param matrix the matrix to multiply
/// @param array the vector to multiply
/// @return the result of the multiplication
template<typename T> std::vector<T> matrix_vector_multiplication_parallel(csr_matrix<T>& matrix, std::vector<T>& array) {
    
    std::vector<T> result(array.size());

    #pragma omp parallel for
    for(size_t i = 0; i < array.size(); i++) {
        long row_start = matrix.row_indices[i];
        long row_end = matrix.row_indices[i + 1];
        for(long value_index = row_start; value_index < row_end; value_index++) {
            result[i] += (matrix.values[value_index] * array[i]);
        }
    }

    return result;
}

int main(int argc, char* argv[]) {

    if(argc == 1) {
        std::cout << "USAGE: spmv <matrix-path>\n";
        return 0;
    }
    if(argc != 2) {
        throw std::invalid_argument("No file path present.");
    }

    std::ifstream file(argv[1]);
    std::string header;
    std::getline(file, header);
    matrix_metadata metadata = identify_matrix(header);

    switch(metadata.field_values) {
        case field_type::integer: {

            std::cout << "Loading matrix...\n";
            csr_matrix<long> m = read_integer_matrix(file, metadata);

            std::cout << "Generating vector...\n";
            std::vector<long> test_arr = generate_integer_array(m.n_rows);

            std::cout << "Starting sequential benchmark...\n";
            benchmark_results results = benchmark([&m, &test_arr]() { matrix_vector_multiplication(m, test_arr); }, 10);
            std::cout << "\nRESULTS:\n";
            std::cout << "Fastest: " << results.fastest_time << "ms\n";
            std::cout << "Slowest: " << results.slowest_time << "ms\n";
            std::cout << "Average: " << results.average_time << "ms\n";
            std::cout << "90th percentile: " << results.ninetieth_percentile_time << "ms\n";
            std::cout << "Run times: ";
            for(long v : results.times) {
                std::cout << v << "ms, ";
            }
            std::cout << "\n\n";

            std::cout << "Starting parallel benchmark...\n";
            benchmark_results results2 = benchmark([&m, &test_arr]() { matrix_vector_multiplication_parallel(m, test_arr); }, 10);
            std::cout << "\nRESULTS:\n";
            std::cout << "Fastest: " << results2.fastest_time << "ms\n";
            std::cout << "Slowest: " << results2.slowest_time << "ms\n";
            std::cout << "Average: " << results2.average_time << "ms\n";
            std::cout << "90th percentile: " << results2.ninetieth_percentile_time << "ms\n";
            std::cout << "Run times: ";
            for(long v : results2.times) {
                std::cout << v << "ms, ";
            }
            std::cout << "\n";

            break;
        }
        case field_type::real: {
            
            std::cout << "Loading matrix...\n";
            csr_matrix<double> m = read_real_matrix(file, metadata);

            std::cout << "Generating vector...\n";
            std::vector<double> test_arr = generate_real_array(m.n_rows);

            std::cout << "Starting sequential benchmark...\n";
            benchmark_results results = benchmark([&m, &test_arr]() { matrix_vector_multiplication(m, test_arr); }, 10);
            std::cout << "\nRESULTS:\n";
            std::cout << "Fastest: " << results.fastest_time << "ms\n";
            std::cout << "Slowest: " << results.slowest_time << "ms\n";
            std::cout << "Average: " << results.average_time << "ms\n";
            std::cout << "90th percentile: " << results.ninetieth_percentile_time << "ms\n";
            std::cout << "Run times: ";
            for(long v : results.times) {
                std::cout << v << "ms, ";
            }
            std::cout << "\n\n";

            std::cout << "Starting parallel benchmark...\n";
            benchmark_results results2 = benchmark([&m, &test_arr]() { matrix_vector_multiplication_parallel(m, test_arr); }, 10);
            std::cout << "\nRESULTS:\n";
            std::cout << "Fastest: " << results2.fastest_time << "ms\n";
            std::cout << "Slowest: " << results2.slowest_time << "ms\n";
            std::cout << "Average: " << results2.average_time << "ms\n";
            std::cout << "90th percentile: " << results2.ninetieth_percentile_time << "ms\n";
            std::cout << "Run times: ";
            for(long v : results2.times) {
                std::cout << v << "ms, ";
            }
            std::cout << "\n";

            break;
        }
    }

    return 0;
}