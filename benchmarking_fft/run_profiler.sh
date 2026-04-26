srun -n 8 -N 2 --ntasks-per-node=4 --gpu-bind=none ./profile_wrapper.sh ../FFttest --grid 24.24.24.64 --mpi 1.1.1.8
