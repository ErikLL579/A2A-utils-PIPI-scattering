srun -n 4 -N 1 --ntasks-per-node=4 --gpus-per-task=1 --gpu-bind=none ./FFttest --grid 24.24.24.64 --mpi 1.1.1.4
