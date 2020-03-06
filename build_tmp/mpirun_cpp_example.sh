#!/bin/bash
LD_LIBRARY_PATH=. mpirun -np 2 ./cpp_example
#LD_LIBRARY_PATH=. perf record -F 99 mpirun -np 2 ./cpp_example
