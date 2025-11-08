#include <algorithm>
#include <functional>
#include <iostream>
#include <map>
#include <string>
#include <stdexcept>
#include <utility>
#include <vector>

#include "matrix.hpp"

/**********************/
/***  Private API  ****/
/**********************/

/// @brief Split the string around the given delimiter.
/// @param str string to split
/// @param delimiter delimiter for splitting
/// @return array of strings created by splitting
std::vector<std::string> split(std::string str, std::string delimiter) {
    std::vector<std::string> tokens;
    int pos = 0;
    while(pos != std::string::npos) {
        pos = str.find(delimiter);
        tokens.push_back(str.substr(0, pos));
        str.erase(0, pos + delimiter.length());
    }
    return tokens;
}

/// @brief Return the lowercase version of a string.
/// @param str the string
/// @return lowercase version of `str`
std::string to_lowercase(std::string str) {
    std::string lower_str;
    for(char c : str) {
        lower_str += std::tolower(c);
    }
    return lower_str;
}

/// @brief Convert a matrix stored as a map of coordinates and associated values to a Compressed Sparse Row representation.
/// @tparam T the type of the values in the matrix
/// @param matrix the matrix to convert
/// @return a CSR representation of `matrix`
template<typename T> csr_matrix<T> map_to_csr_matrix(std::map<std::pair<long, long>, T> matrix) {
    
    csr_matrix<T> sparse_matrix;
    sparse_matrix.row_indices.push_back(0);

    long latest_row = 0;
    long values_read = 0;
    for(std::pair<std::pair<long, long>, T> kv : matrix) {
        long row = kv.first.first;
        long column = kv.first.second;

        if(latest_row != row) {
            latest_row = row;
            sparse_matrix.row_indices.push_back(values_read);
        }

        sparse_matrix.values.push_back(kv.second);
        sparse_matrix.column_indices.push_back(column);
        values_read++;
    }

    sparse_matrix.row_indices.push_back(values_read);    
    return sparse_matrix;
}

/// @brief Read a Matrix Market file, returning data in a compressed sparse row format.
/// @tparam T the type of the matrix cell values contained in the file
/// @param mtx_file the Matrix Market file contents, with the header already removed
/// @param symmetry the symmetry of the matrix
/// @param value_extractor the function used to convert the string containing a cell value to type `T`
/// @return the matrix in CSR format
template<typename T> csr_matrix<T> read_coordinate_matrix(std::istream& mtx_file, symmetry_type symmetry, std::function<T (std::string)> value_extractor) {
    
    size_t rows = 0;
    size_t columns = 0;
    size_t nonzeros = 0;

    // read matrix size line
    for(std::string line; std::getline(mtx_file, line);) {
        if(line.substr(0,1) == "%" || line.length() == 0) {
            continue;
        }

        std::vector<std::string> tokens = split(line, " ");
        if(tokens.size() != 3) {
            throw std::invalid_argument("Size line does not have correct syntax");
        }
        rows = std::stol(tokens[0]);
        columns = std::stol(tokens[1]);
        nonzeros = std::stol(tokens[2]);
        break;
    }

    std::map<std::pair<long, long>, T> map_matrix;

    // read and store values in a 2D array matrix
    long n_nonzeroes = 0;
    for(std::string line; std::getline(mtx_file, line);) {
        if(line.substr(0,1) == "%" || line.length() == 0) {
            continue;
        }

        std::vector<std::string> tokens = split(line, " ");
        if(tokens.size() != 3) {
            throw std::invalid_argument("Size line does not have correct syntax");
        }
        int row = std::stoi(tokens[0]) - 1;
        int column = std::stoi(tokens[1]) - 1;
        T value = value_extractor(tokens[2]);
        n_nonzeroes++;

        if(row > rows || column > columns) {
            throw std::out_of_range("Value's coordinates are outside the dimensions defined in the size line");
        }

        map_matrix[std::make_pair(row, column)] = value;
        if(row != column && symmetry == symmetry_type::symmetric) {
            map_matrix[std::make_pair(column, row)] = value;
        }
        if(row != column && symmetry == symmetry_type::skew_symmetric) {
            map_matrix[std::make_pair(column, row)] = -value;
        }
    }

    if(n_nonzeroes != nonzeros) {
        throw std::invalid_argument("Number of nonzero values in size line and actual given values are not equal");
    }

    csr_matrix<T> sparse_matrix = map_to_csr_matrix(map_matrix);
    sparse_matrix.n_rows = rows;
    sparse_matrix.n_columns = columns;
    return sparse_matrix;
}

/*********************/
/***  Public API  ****/
/*********************/

matrix_metadata identify_matrix(std::string mm_header) {

    std::vector<std::string> header_tokens = split(to_lowercase(mm_header), " ");

    if(header_tokens.size() != 5 || (header_tokens[0] != "%matrixmarket" && header_tokens[0] != "%%matrixmarket")) {
        throw std::invalid_argument("Header does not represent a Matrix Market file");
    }

    if(header_tokens[1] != "matrix") {
        throw std::invalid_argument("Object type '" + header_tokens[1] + "' is not supported");
    }

    std::string format = header_tokens[2];
    std::string field_type = header_tokens[3];
    std::string symmetry_string = header_tokens[4];

    matrix_metadata metadata;

    if(field_type == "integer") {
        metadata.field_values = field_type::integer;
    } else if(field_type == "real") {
        metadata.field_values = field_type::real;
    } else {
        throw std::invalid_argument("Field type '" + field_type + "' is not supported");
    }

    if(symmetry_string == "general") {
        metadata.symmetry = symmetry_type::general;
    } else if(symmetry_string == "symmetric") {
        metadata.symmetry = symmetry_type::symmetric;
    } else if(symmetry_string == "skew-symmetric") {
        metadata.symmetry = symmetry_type::skew_symmetric;
    } else {
        throw std::invalid_argument("Symmetry '" + symmetry_string + "' is not supported");
    }

    if(format != "coordinate") {
        throw std::invalid_argument("Format '" + format + "' is not supported");
    }

    return metadata;
}

csr_matrix<long> read_integer_matrix(std::istream& mtx_file, matrix_metadata metadata) {
    std::function<long (std::string)> extractor = [](std::string s) { return std::stol(s); };
    return read_coordinate_matrix(mtx_file, metadata.symmetry, extractor);
}

csr_matrix<double> read_real_matrix(std::istream& mtx_file, matrix_metadata metadata) {
    std::function<double (std::string)> extractor = [](std::string s) { return std::stod(s); };
    return read_coordinate_matrix(mtx_file, metadata.symmetry, extractor);
}