#!/bin/bash

GRID=/global/cfs/cdirs/mp13/lundstrum/Grid-231225/Grid-install-GPU-041726/bin/grid-config

# or -x cu if text.cc
#nvcc matrix-vector-test.cu \
nvcc FFT-test.cu \
  $($GRID --cxxflags) \
  $($GRID --ldflags) \
  $($GRID --libs) \
  -O3 \
  -o FFttest


# example multiple MPI rank execute: srun -n 2 -N 1 --ntasks-per-node=2 --gpus-per-task=1 --gpu-bind=none ./test --mpi 1.1.1.2
