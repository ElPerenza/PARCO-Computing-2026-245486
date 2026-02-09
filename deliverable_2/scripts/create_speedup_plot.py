from argparse import ArgumentParser
import csv
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import sys

def get_matrix_name(file_name: str) -> str:
    index = file_name.rfind("_")
    return file_name[:index]

def get_csv_values(file_path: str) -> list[str]:
    with open(file_path) as f:
        reader = csv.reader(f)
        next(reader) # ignore headers
        return next(reader)

parser = ArgumentParser("create_speedup_plot")
parser.add_argument("results_path", help="Path to the directory containg the result files")
parser.add_argument("output_path", help="Path to where the generated plot will be saved as an image")
args = parser.parse_args()

fig, ax = plt.subplots(1, 1, layout="constrained", figsize=[8.4, 4.8])

# CSV structure: comp_fastest,comp_slowest,comp_average,comp_ninetieth,comm_fastest,comm_slowest,comm_average,comm_ninetieth,flops_average
csv_files: list[str] = [file for file in os.listdir(args.results_path) if file.endswith(".csv")]
for matrix_name, matrix_iter in itertools.groupby(csv_files, get_matrix_name):
    if matrix_name == "generated":
        continue
    matrix_files = list(matrix_iter)

    seq_matches = [file for file in matrix_files if file.rfind("_sequential.csv") != -1]
    if len(seq_matches) == 0:
        print(f"No sequential benchmark found for {matrix_name}")
        sys.exit(-1)
    matrix_files.remove(seq_matches[0])

    seq_path = os.path.join(args.results_path, seq_matches[0])
    sequential_time = float(get_csv_values(seq_path)[3]) # get only the 90th percentile time (2nd index as CSV is fastest,slowest,90th)

    execution_times: dict[int, float] = dict()
    for parallel_path in (os.path.join(args.results_path, f) for f in matrix_files):
        parallel_time = float(get_csv_values(parallel_path)[3])
        exec_time_percent = parallel_time / sequential_time * 100
        num_procs = int(re.findall(f"{matrix_name}_([0-9]+)procs.csv", parallel_path)[0])
        execution_times[num_procs] = exec_time_percent 

    sorted_times = sorted(execution_times.items(), key=lambda kv: kv[0])
    x_values = [kv[0] for kv in sorted_times]
    y_values = [kv[1] for kv in sorted_times]
    ax.plot(x_values, y_values, "o-", label=matrix_name)

ax.set_xlim(0, plt.xlim()[1])
ax.set_ylim(0, max(plt.ylim()[1], 30))
ax.set_xticks(np.arange(0, 129, 8))

#ax.plot([1, 128], [100, 100], label="Sequential baseline")

ax.set_title("Computation time by increasing the number of processes")
ax.set_xlabel("Number of MPI processes")
ax.set_ylabel("Execution time [%]")
ax.grid(True)
fig.legend(loc="outside right upper") 
fig.savefig(args.output_path)