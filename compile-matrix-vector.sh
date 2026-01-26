#!/bin/bash

GRID=/global/cfs/cdirs/mp13/lundstrum/Grid-231225/Grid-install-GPU-261225/bin/grid-config

# or -x cu if text.cc
#nvcc matrix-vector-test.cu \
nvcc disconnected_test.cu \
  $($GRID --cxxflags) \
  $($GRID --ldflags) \
  $($GRID --libs) \
  -O3 \
  -o test


# example multiple MPI rank execute: srun -n 2 -N 1 --ntasks-per-node=2 --gpus-per-task=1 --gpu-bind=single:1 ./test --mpi 1.1.1.2
