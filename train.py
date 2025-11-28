"""
GraphCast Distributed Training with MPI on HPC
"""

import dataclasses
import datetime
import functools
import os
import sys
from typing import Optional
import random
import pickle
from time import time
import argparse
import yaml
#import logging

import numpy as np
import xarray as xr
import jax
import jax.numpy as jnp
from jax.experimental import multihost_utils
import haiku as hk
import optax
from mpi4py import MPI
import mpi4jax

# GraphCast imports
from graphcast import autoregressive
from graphcast import casting
from graphcast import checkpoint
from graphcast import data_utils
from graphcast import graphcast
from graphcast import normalization
from graphcast import xarray_jax
from graphcast import xarray_tree

import graphcast.loss_utils
from graphcast.data_generator import SingleZarrDataGenerator

#logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# =============================================================================
# MPI Setup
# =============================================================================

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
local_rank = rank % 2  # 2 GPUs per node

def print_rank0(msg):
    """Print only from rank 0"""
    if rank == 0:
        print(msg)
        #logging.info(msg)
        sys.stdout.flush()

# =============================================================================
# JAX Distributed Setup
# =============================================================================

def setup_jax_distributed():
    """Configure JAX for distributed training with MPI"""
    # Broadcast coordinator address from rank 0
    if rank == 0:
        coordinator_address = MPI.Get_processor_name()
    else:
        coordinator_address = None
    
    coordinator_address = comm.bcast(coordinator_address, root=0)
    
    # Initialize JAX distributed
    jax.distributed.initialize(
        coordinator_address=f"{coordinator_address}:12345",
        num_processes=size,
        process_id=rank,
    )
    
    # Set GPU visibility
    os.environ['CUDA_VISIBLE_DEVICES'] = str(local_rank)
    
    print_rank0(f"JAX distributed initialized with {size} processes")
    print_rank0(f"JAX devices per process: {jax.local_device_count()}")
    print_rank0(f"Total JAX devices: {jax.device_count()}")

# =============================================================================
# Memory Optimization
# =============================================================================

def configure_memory():
    """Configure memory settings for H100-NVL"""
    # Enable unified memory for host RAM as extra GPU memory
    os.environ['TF_FORCE_UNIFIED_MEMORY'] = '1'
    
    # Set memory fraction (>1.0 enables unified memory)
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.85'
    
    # Enable memory preallocation
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true'
    
    # Optimize XLA for H100
    os.environ['XLA_FLAGS'] = (
        '--xla_gpu_triton_gemm_any=true '
        '--xla_gpu_enable_latency_hiding_scheduler=true '
        '--xla_gpu_enable_highest_priority_async_stream=true '
        '--xla_gpu_enable_cudnn_fmha=true '
    )
    
    # NCCL settings for multi-node
    os.environ['NCCL_DEBUG'] = 'WARN'
    os.environ['NCCL_IB_DISABLE'] = '0'
    os.environ['NCCL_NET_GDR_LEVEL'] = '5'
    os.environ['NCCL_IB_GID_INDEX'] = '3'
    os.environ['NCCL_IB_TIMEOUT'] = '22'

    #UCX settings for GPU direct
    os.environ['UCX_MEMTYPE_CACHE'] = 'n'
    os.environ['UCX_TLS'] = 'rc,sm,cuda_copy,cuda_ipc'

    # Reduce CUDA stream memory overhead
    os.environ['CUDA_DEVICE_MAX_CONNECTION'] = '1'
    
    print_rank0("Memory optimization configured")

# =============================================================================
# Model Construction
# =============================================================================

def construct_wrapped_graphcast(
    model_config: graphcast.ModelConfig,
    task_config: graphcast.TaskConfig,
    diffs_stddev_by_level: xr.Dataset,
    mean_by_level: xr.Dataset,
    stddev_by_level: xr.Dataset
):
    """Constructs and wraps the GraphCast Predictor."""
    predictor = graphcast.GraphCast(model_config, task_config)
    predictor = casting.Bfloat16Cast(predictor)
    predictor = normalization.InputsAndResiduals(
        predictor,
        diffs_stddev_by_level=diffs_stddev_by_level,
        mean_by_level=mean_by_level,
        stddev_by_level=stddev_by_level
    )
    predictor = autoregressive.Predictor(predictor, gradient_checkpointing=True)
    return predictor

# =============================================================================
# Loss and Gradient Functions
# =============================================================================

def create_loss_fn(diffs_stddev_by_level, mean_by_level, stddev_by_level, custom_loss_fn=None):
    """Create loss function with normalization params"""
    
    @hk.transform_with_state
    def loss_fn(model_config, task_config, inputs, targets, forcings):
        predictor = construct_wrapped_graphcast(
            model_config, task_config,
            diffs_stddev_by_level, mean_by_level, stddev_by_level
        )
        if (custom_loss_fn is not None):
            loss, diagnostics = custom_loss_fn(predictor(inputs,targets,forcings),targets)
        else:
            loss, diagnostics = predictor.loss(inputs, targets, forcings)
        return xarray_tree.map_structure(
            lambda x: xarray_jax.unwrap_data(x.mean(), require_jax=True),
            (loss, diagnostics)
        )
    
    return loss_fn

def create_grads_fn(loss_fn, model_config, task_config):
    """Create gradient function with MPI synchronization"""
    
    def grads_fn(params, state, inputs, targets, forcings):
        def _aux(params, state, i, t, f):
            (loss, diagnostics), next_state = loss_fn.apply(
                params, state, jax.random.PRNGKey(0),
                model_config, task_config, i, t, f
            )
            return loss, (diagnostics, next_state)
        
        (loss, (diagnostics, next_state)), grads = jax.value_and_grad(
            _aux, has_aux=True
        )(params, state, inputs, targets, forcings)
        
        # Synchronize gradients across all MPI ranks
        grads = jax.tree.map(
            lambda g: mpi4jax.allreduce(g, op=MPI.SUM, comm=comm),
            grads
        )
        
        # Average gradients
        grads = jax.tree.map(lambda g: g / size, grads)
        
        # Synchronize loss
        loss = mpi4jax.allreduce(
            loss, op=MPI.SUM, comm=comm
        )
        loss = loss / size
        
        return loss, diagnostics, next_state, grads
    
    return grads_fn

# =============================================================================
# Checkpointing
# =============================================================================

def save_checkpoint(params, opt_state, step, train_loss, val_loss, checkpoint_dir):
    """Save checkpoint (only from rank 0)"""
    if rank != 0:
        return
    
    train_loss_str = f"{train_loss:.4f}"
    val_loss_str = f"{val_loss:.4f}"
    
    checkpoint_data = {
        'params': params,
        'opt_state': opt_state,
        'step': step,
        'train_loss': train_loss,
        'val_loss': val_loss,
    }
    
    checkpoint_path = os.path.join(
        checkpoint_dir,
        f'checkpoint_step_{step}_loss_{train_loss_str}_val_{val_loss_str}.pkl'
    )
    
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    
    print_rank0(f"Checkpoint saved: {checkpoint_path}")

def load_checkpoint(checkpoint_path):
    """Load checkpoint"""
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'rb') as f:
            checkpoint_data = pickle.load(f)
        print_rank0(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint_data
    return None

# =============================================================================
# Training Loop
# =============================================================================

def train_graphcast(
    config: dict,
):
    """Main training function with MPI"""
    
    # Setup
    configure_memory()
    setup_jax_distributed()

    batch_size = config['batch_size']
    num_steps = config['total_training_steps']

    checkpoint_dir = f"{config['checkpoint_dir']}_{config['val_steps']}AR"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print_rank0(f"\n{'='*70}")
    print_rank0("GraphCast Distributed Training with MPI")
    print_rank0(f"{'='*70}")
    print_rank0(f"MPI Processes: {size}")
    print_rank0(f"Effective batch size: {config['batch_size'] * size}")
    print_rank0(f"Learning schedule: {config['lr_scheduler']}")
    print_rank0(f"{'='*70}\n")
    
    # Load normalization stats (all ranks)
    print_rank0("Loading normalization statistics...")
    diffs_stddev_by_level = xr.load_dataset(
        os.path.join(config['norm_stats_path'], 'diffs_stddev_by_level.nc')
    ).compute()
    mean_by_level = xr.load_dataset(
        os.path.join(config['norm_stats_path'], 'mean_by_level.nc')
    ).compute()
    stddev_by_level = xr.load_dataset(
        os.path.join(config['norm_stats_path'], 'stddev_by_level.nc')
    ).compute()
    
    # Load initial checkpoint
    print_rank0("Loading initial model configuration...")
    ckpt = checkpoint.load(
        config['initial_params'],
        graphcast.CheckPoint
    )

    model_config = ckpt.model_config
    task_config = ckpt.task_config
    #params = ckpt.params
    state = {}
    
    # read params from pkl file if last_checkpoint is provided
    if config['last_checkpoint'] is not None and os.path.exists(config['last_checkpoint']):
        print_rank0(f"Loading initial parameters from {config['last_checkpoint']}...")
        with open(config['last_checkpoint'], 'rb') as f:
            new_ckpt = pickle.load(f)
            params = new_ckpt['params']
    else:
        # Use checkpoint params
        params = ckpt.params
    
    
    if config['loss_metric'] == "AMSE":
        custom_loss = graphcast.loss_utils.make_loss_new(
            model_config,
            task_config,
            diffs_stddev_by_level,
            mean_by_level,
            stddev_by_level,
            not (rank == 0),
        )
    else:
        custom_loss = None

    # Create loss and gradient functions
    loss_fn = create_loss_fn(diffs_stddev_by_level, mean_by_level, stddev_by_level, custom_loss_fn=custom_loss)

    grads_fn = create_grads_fn(loss_fn, model_config, task_config)
    
    # JIT compile gradient function
    grads_fn_jit = jax.jit(grads_fn)
    
    if config['lr_scheduler'] == "constant":

        lr = float(config['learning_rate'])

        optimizer = optax.adamw(
            learning_rate=lr,
            b1=config['b1'],
            b2=config['b2'],
            weight_decay=config['weight_decay'],
        )

    elif config['lr_scheduler'] == "cosine_decay":

        lr_scheduler = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=config['cosine_peak_lr'],
            warmup_steps=config['warmup_steps'],
            decay_steps=num_steps - config['warmup_steps'], # Total steps *after* warmup
            end_value=config['cosine_end_lr'],
            exponent=config['exponent'],
        )

        optimizer = optax.adamw(
            learning_rate=lr_scheduler,
            b1=config['b1'],
            b2=config['b2'],
            weight_decay=config['weight_decay'],
        )

        #if training was stopped due to walltime limit, set resume to True in the yaml file to restart training
        if config['resume']:
            opt_state = new_ckpt['opt_state']
            start_step = new_ckpt['step'] + 1
        else:
            opt_state = optimizer.init(params)
            start_step = 0

    try:
        print(f"Getting data generator on rank {rank}...\n")
        train_generator = SingleZarrDataGenerator(
            zarr_path=config['train_zarr_path'],
            task_config=task_config,
            n_target_steps=config['val_steps'],
            batch_size=config['batch_size'],
            prefetch_size=config['prefetch_size'],
            num_workers=config['num_workers'],
            rank=rank,
            size=size,
        )

        if config['validate']:
            valid_generator = SingleZarrDataGenerator(
                zarr_path=config['val_zarr_path'],
                task_config=task_config,
                n_target_steps=config['val_steps'],
                batch_size=config['batch_size'],
                prefetch_size=config['prefetch_size'],
                num_workers=config['num_workers'],
                rank=rank,
                size=size,
            )

        print_rank0("Starting training...\n")
        
        # Training loop
        for step in range(start_step, num_steps):

            t0 = time()
            lr = lr_scheduler(step)
         
            inputs, targets, forcings = train_generator.generate()

            elapsed_time = time() - t0
            print_rank0(f"\nElapsed time for loading data to memory for step {step+1}/{num_steps}: {elapsed_time} seconds")
            
            t0 = time()
            # Compute gradients
            loss_raw, diagnostics, next_state, grads = grads_fn_jit(
                params, state, inputs, targets, forcings
            )
            train_loss = float(loss_raw)
            
            elapsed_time = time() - t0
            print_rank0(f"\nElapsed time for computation for step {step+1}/{num_steps}: {elapsed_time} seconds")
            print_rank0(f"  Training Loss: {train_loss:.6f}, lr: {lr:.2e}")
            
            # Validation
            if config['validate'] and (step % config['val_frequency'] == 0):
             
                inputs_val, targets_val, forcings_val = valid_generator.generate()
                
                val_loss_raw, _, _, _ = grads_fn_jit(
                    params, state, inputs_val, targets_val, forcings_val
                )
                val_loss = float(val_loss_raw)
                
                print_rank0(f"  Validation Loss: {val_loss:.6f}")

            else:
                val_loss = 0.0 #if validate is False, set val_loss to 0 for use in the ckpt file
            
            # Update parameters
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            
            # Checkpoin0t
            if (step%config['checkpoint_frequency'] == 0) | (step == num_steps-1):
                save_checkpoint(
                    params, opt_state, step,
                    train_loss, val_loss, checkpoint_dir
                )
            
            # Synchronize all ranks
            comm.Barrier()

    finally:
        # cleanup generators
        if train_generator is not None:
            train_generator.stop_prefetching()

        if config['validate'] and (valid_generator is not None):
            valid_generator.stop_prefetching()
    
    # Cleanup
    jax.distributed.shutdown()

    print_rank0("\nTraining completed!")
# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Load and parse configuration file.')
    parser.add_argument('--config', type=str, required=True, help='config.yaml file')
    args = parser.parse_args()
    
    with open(args.config) as f:
        config_dict = yaml.full_load(f)

    # AMSE related
    if config_dict['loss_metric'] = "AMSE":
        error_group = parser.add_argument_group('Error options')

        error_group.add_argument('--error-weights',type=str,dest='error_weight_file',default=None,
                            help='File containing non-default variable and level weights')
        error_group.add_argument('--wind-speed',action='store_true',dest='wind_speed',default=False,
                            help='Add wind speed variable to loss function')
        error_group.add_argument('--time-bias',action='store_true',dest='time_bias',default=False,
                            help='Add time-averaged term to loss function')
        error_group.add_argument('--mean-bias',action='store_true',dest='mean_bias',default=False,
                            help='Add global mean bias term to loss function')
        error_group.add_argument('--spectral-amse',action='store_true', dest='spectral_amse',default=True,
                            help='Compute loss in spectral space, with correlation/ampltidue adjustment')
        error_group.add_argument('--mae',action='store_true', dest='mae',default=False,
                            help='Compute loss with mean absolute error rather than MSE')
        args = parser.parse_args()
        graphcast.loss_utils.parse_arguments(args)
    
    try:
        train_graphcast(
            config_dict,
        )
    except Exception as e:
        print(f"Rank {rank} error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        comm.Abort(1)

