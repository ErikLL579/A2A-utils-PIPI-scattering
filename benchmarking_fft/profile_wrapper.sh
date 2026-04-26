#!/bin/bash
# profile_wrapper.sh
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID

nsys profile \
    --trace=cuda,nvtx,mpi,osrt \
    --mpi-impl=mpich \
    --cuda-memory-usage=true \
    --force-overwrite=true \
    -o $PSCRATCH/fft_trace_rank%q{SLURM_PROCID} \
    "$@"
