
# old command assigned all MPI nodes to a single device!
#srun -n 4 -N 1 --ntasks-per-node=4 --gpus-per-task=1 --gpu-bind=none ./FFttest --grid 24.24.24.64 --mpi 1.1.1.4

# new command that resolves this
srun -n 8 -N 2 --ntasks-per-node=4 --gpu-bind=none bash -c 'export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID; exec ./FFttest --grid 24.24.24.64 --mpi 1.1.1.8'
