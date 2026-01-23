#!/bin/bash

GRID=/global/cfs/cdirs/mp13/lundstrum/Grid-231225/Grid-install-GPU-261225/bin/grid-config

nvcc ErikVsClaudeGPUBattle.cu \
  $($GRID --cxxflags) \
  $($GRID --ldflags) \
  $($GRID --libs) \
  -O3 \
  -o gpu-battle

echo "Compiled successfully. Run with:"
echo "  ./gpu-battle --grid 24.24.24.64"
echo ""
echo "Or with MPI:"
echo "  srun -n 2 -N 1 --ntasks-per-node=2 --gpus-per-task=1 --gpu-bind=single:1 ./gpu-battle --mpi 1.1.1.2 --grid 24.24.24.64"
