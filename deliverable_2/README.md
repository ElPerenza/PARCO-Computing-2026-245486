# Parallel Computing Course - Deliverable 2
This repository contains the code developed for the scientific paper of the second deliverable of the Parallel Computing course.  
Note: the working directory (`./`) mentioned in this README refers to the `deliverable_2` folder, not the repository root. 

## Getting the matrices
Since the files are too large to upload on GitHub, the matrices used for benchmarks must be downloaded separately. The downloaded `.mtx` files should be placed in the `./matrices` folder (needs to be created). The matrices are:
- `mouse_gene`: https://suitesparse-collection-website.herokuapp.com/MM/Belcastro/mouse_gene.tar.gz
- `vas_stokes_2M`: https://suitesparse-collection-website.herokuapp.com/MM/VLSI/vas_stokes_2M.tar.gz
- `circuit5M`: https://suitesparse-collection-website.herokuapp.com/MM/Freescale/circuit5M.tar.gz
- `kron_g500-logn17`: https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/kron_g500-logn17.tar.gz
- `kron_g500-logn20`: https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/kron_g500-logn20.tar.gz
- `mawi_201512020000`: https://suitesparse-collection-website.herokuapp.com/MM/MAWI/mawi_201512020000.tar.gz

## Compiling the executable
The source code of the program is contained in the `./src` directory. All files are documented to explain their purpose. To compile it, OpenMPI 4.1.6 with GCC 13.2.0 is needed. The command:
```
mpicxx -O3 -march=native -mtune=native -o ./spmv ./src/spmv.cpp ./src/matrix.cpp ./src/benchmark.cpp
```
Will compile the executable with the same settings used in the paper and place it in the root directory of the repository.

## Using the executable
The exeutable's syntax is:
```
./spmv [sequential | parallel] [<matrix-path> | generated] <results-path> <num-rows> <sparsity>
```
Where the arguments are:
- `[sequential | parallel]` - decide whether the SpMV algorithm will be benchmarked sequentially or in parallel (using all processors made available to MPI)
- `<matrix-path> | generated` - the path to the file where the matrix data is stored (using the Matrix Market format), or the string `generated`, in which case the matrix will instead be generated using additional arguments explained below.
- `<results-path>` - the path to the file where the benchmark data will be saved, in CSV format. The benchmark results contain statistics on the communication/computation times, all in milliseconds, and the average FLOPS (or IOPS) for the SpMV kernel. **Make sure the folder in which the file will reside already exists before running the program.** 
- `<num-rows>` - only used in conjunction with `generated` argument. Specifies the number of rows (and columns, since it will always generate a square matrix) the generated matrix will have.
- `<sparsity>` - only used in conjunction with `generated` argument. Specifies the sparsity (as a decimal number between 0 and 1) the generated matrix will have.

## Running on the cluster
To run all benchmarks on the UniTN HPC cluster, the file `./scripts/benchmark_all.pbs` can be used with `qsub`. This file compiles the executable and then executes all sequential and parallel benchmarks for the 6 matrices found in `./matrices`, plus addional weak scaling benchmarks with generated matrices. Resources don't need to be specified as they're already inside the file.

## Generating the plot
To generate the plot found in the paper, use the file `./scripts/create_speedup_plot.py`. The only dependencies are Python `3.11` and matplotlib `3.10.7`, as specified in `./requirements.txt`. The syntax is:
```
python ./scripts/create_speedup_plot.py <results_path> <output_path>
```
Where the arguments are:
- `results_path` - the path to the folder where the result CSV files from the benchmarks made with `./spmv` are stored.
- `output_path` - the path to the image file where the plot will be saved. It can be any extension supported by matplotlib.