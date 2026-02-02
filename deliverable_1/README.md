# Parallel Computing Course - Deliverable 1
This repository contains the code developed for the scientific paper of the first deliverable of the Parallel Computing course.  
Note: the working directory (`./`) mentioned in this README refers to the `deliverable_1` folder, not the repository root. 

## Getting the matrices
Since the files are too large to upload on GitHub, the matrices used for benchmarks must be downloaded separately. The downloaded `.mtx` files should be placed in the `./matrices` folder (needs to be created). The matrices are:
- `mouse_gene`: https://suitesparse-collection-website.herokuapp.com/MM/Belcastro/mouse_gene.tar.gz
- `vas_stokes_2M`: https://suitesparse-collection-website.herokuapp.com/MM/VLSI/vas_stokes_2M.tar.gz
- `circuit5M`: https://suitesparse-collection-website.herokuapp.com/MM/Freescale/circuit5M.tar.gz
- `kron_g500-logn17`: https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/kron_g500-logn17.tar.gz
- `kron_g500-logn20`: https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/kron_g500-logn20.tar.gz
- `mawi_201512020000`: https://suitesparse-collection-website.herokuapp.com/MM/MAWI/mawi_201512020000.tar.gz

## Compiling the executable
The source code of the program is contained in the `./src` directory. All files are documented to explain their purpose. To compile it, GCC 9.1.0 is needed. The command:
```
g++ -fopenmp -O3 -march=native -mtune=native -o ./spmv ./src/spmv.cpp ./src/matrix.cpp ./src/benchmark.cpp
```
Will compile the executable with the same settings used in the paper and place it in the root directory of the repository.

## Using the executable
The exeutable's syntax is:
```
./spmv [sequential | parallel] <matrix_path> <results_path>
```
Where the arguments are:
- `[sequential | parallel]` - decide whether the SpMV algorithm will be benchmarked sequentially or in parallel (using all cores made available to OpenMP)
- `matrix_path` - the path to the file where the matrix data is stored (using the Matrix Market format).
- `results_path` - the path to the file where the benchmark data will be saved, in CSV format. The benchmark results contain the fastest, dlowest and 90th percentile execution time, all in milliseconds. **Make sure the folder in which the file will reside already exists before running the program.** 

## Running on the cluster
To run all benchmarks on the UniTN HPC cluster, the file `./scripts/benchmark_all.pbs` can be used with `qsub`. This file compiles the executable and then executes all sequential and parallel benchmarks for the 6 matrices found in `./matrices`. Resources don't need to be specified as they're already inside the file.

## Generating the plot
To generate the plot found in the paper, use the file `./scripts/create_speedup_plot.py`. The only dependencies are Python `3.11` and matplotlib `3.10.7`, as specified in `./requirements.txt`. The syntax is:
```
python ./scripts/create_speedup_plot.py <results_path> <output_path>
```
Where the arguments are:
- `results_path` - the path to the folder where the result CSV files from the benchmarks made with `./spmv` are stored.
- `output_path` - the path to the image file where the plot will be saved. It can be any extension supported by matplotlib.