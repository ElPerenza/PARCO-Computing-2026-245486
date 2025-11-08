#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <iostream>
#include <string>
#include <vector>

enum field_type {
    integer,
    real
};

enum symmetry_type {
    general,
    symmetric,
    skew_symmetric
};

struct matrix_metadata {
    field_type field_values;
    symmetry_type symmetry;
};

template<typename T> struct csr_matrix {
    size_t n_rows;
    size_t n_columns;
    std::vector<long> column_indices;
    std::vector<long> row_indices;
    std::vector<T> values;
};

/// @brief Identify if a Matrix Market file is supported by this program based on its header.
/// @param mm_header the header string
/// @return the matrix's metadata extracted from the header
/// @throws `illegal_argument` if the header doesn't represent a supported matrix
matrix_metadata identify_matrix(std::string mm_header);

/// @brief Read a Matrix Market file containing integer values, returning data in a compressed sparse row format.
/// @param mtx_file the Matrix Market file contents, with the header already removed
/// @param metadata the matrix metadata extracted by `identify_matrix`
/// @return the matrix in CSR format
csr_matrix<long> read_integer_matrix(std::istream& mtx_file, matrix_metadata metadata);

/// @brief Read a Matrix Market file containing real values, returning data in a compressed sparse row format.
/// @param mtx_file the Matrix Market file contents, with the header already removed
/// @param metadata the matrix metadata extracted by `identify_matrix`
/// @return the matrix in CSR format
csr_matrix<double> read_real_matrix(std::istream& mtx_file, matrix_metadata metadata);

#endif