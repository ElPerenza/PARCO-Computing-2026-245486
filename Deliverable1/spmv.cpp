#include <iostream>
#include <fstream>
#include <string>
#include <stdexcept>
#include <vector>

#include "matrix.hpp"

int* vector_multiply_integer(csr_matrix<int>& matrix, int vector[], size_t vector_size) {
    
    int result[vector_size];

    for(size_t i = 0; i < vector_size; i++) {
        int row_start = matrix.row_indices[i];
        int row_end = matrix.row_indices[i + 1];

        for(int value_index = row_start; value_index < row_end; value_index++) {
            result[i] += (matrix.values[value_index] * vector[i]);
        }
    }

    return result;
}

double* vector_multiply_double(csr_matrix<double>& matrix, double vector[], size_t vector_size) {
    
    double result[vector_size];

    for(size_t i = 0; i < vector_size; i++) {
        int row_start = matrix.row_indices[i];
        int row_end = matrix.row_indices[i + 1];

        for(int value_index = row_start; value_index < row_end; value_index++) {
            result[i] += (matrix.values[value_index] * vector[i]);
        }
    }

    return result;
}

int main() {

    std::ifstream file("test.mtx");
    std::string header;
    std::getline(file, header);
    matrix_metadata metadata = identify_matrix(header);

    switch(metadata.field_values) {
        case field_type::integer: {
            csr_matrix<long> m = read_integer_matrix(file, metadata);

            std::cout << "Values: [";
            for(auto v : m.values) {
                std::cout << v << ",";
            }
            std::cout << "]\n";

            std::cout << "Row indices: [";
            for(auto v : m.row_indices) {
                std::cout << v << ",";
            }
            std::cout << "]\n";

            std::cout << "Column indices: [";
            for(auto v : m.column_indices) {
                std::cout << v << ",";
            }
            std::cout << "]\n";

            break;
        }
        case field_type::real: {
            csr_matrix<double> m = read_real_matrix(file, metadata);
            break;
        }
    }

    return 0;
}