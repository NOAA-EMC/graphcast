#!/bin/bash

#SBATCH --job-name=hres1AR
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --account=gpu-ai4wp
#SBATCH --time=12:00:00
#SBATCH --partition=u1-h100
#SBATCH --qos=gpu
#SBATCH --output=slurm/train_hres.%j.out
#SBATCH --error=slurm/train_hres.%j.err
#SBATCH --exclusive

module purge 

unset X509_USER_PROXY

module load hpc-x/2.18.1-mt-gcc
module load cuda/12.9.1   

export SLURM_CPU_BIND=none

# NCCL Settings for mutli-node communication
export NCCL_DEBUG=INFO

export UCX_NET_DEVICES=mlx5_0:1
export UCX_TLS=rc,sm,cuda_copy,cuda_ipc
export UCX_IB_GID_INDEX=3
export UCX_MEMTYPE_CACHE=n

export CUDA_DEVICE_MAX_CONNECTIONS=1

export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export TF_FORCE_UNIFIED_MEMORY=1

#AMD Genoa CPU optimization
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

export MPI4JAX_USE_CUDA_MPI=1

echo "Graphcast train started at : $(date)"
echo "Running on nodes: $SLURM_JOB_NODELIST"

source /scratch3/NCEPDEV/nems/Linlin.Cui/venvs/mpi4jax-gcc/bin/activate

mpirun -np ${SLURM_NTASKS} \
    --bind-to none \
    --map-by slot \
    -x NCCL_DEBUG \
    -x UCX_NET_DEVICES \
    -x UCX_TLS \
    -x JAX_PLATFORMS \
    -x XLA_PYTHON_CLIENT_PREALLOCATE \
    -x XLA_PYTHON_CLIENT_MEM_FRACTION \
    -x TF_FORCE_UNIFIED_MEMORY \
    numactl --interleave=all python train.py --config=configs/finetune_hres_hres_13pl.yaml

echo ""
echo "Job finished at: $(date)"

