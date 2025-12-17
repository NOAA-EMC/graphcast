import os
import xarray as xr
import numpy as np
import dataclasses
import threading
import dask
from queue import Queue, Empty
import time

from graphcast import data_utils
from graphcast import graphcast
from graphcast import checkpoint

#layerquantizer was used to save data. To use it, import it first
import layerquantizer

dask.config.set(**{'array.slicing.split_large_chunks': False})

class SeedGenerator:
    """Thread-safe random seed generator"""
    def __init__(self, initial_seed=0):
        self.seed = initial_seed
        self.lock = threading.Lock()
    
    def get_seed(self):
        with self.lock:
            seed = self.seed
            self.seed += 1
        return seed


class SingleZarrDataGenerator:
    """
    Data generator with robust prefetching
    Uses direct threading instead of ThreadPoolExecutor to avoid shutdown issues
    """
    
    def __init__(
        self, 
        zarr_path: str,
        task_config, 
        n_target_steps: int = 1, 
        batch_size: int = 32, 
        prefetch_size: int = 2,
        num_workers: int = 2,
        random_seed: int = 123,
        rank: int = 0,
        size: int = 1,
    ):
        self.rank = rank
        self.size = size
        self.zarr_path = zarr_path
        
        # Open zarr
        print(f"Rank {rank}: Opening zarr dataset...")
        self.dataset = xr.open_zarr(zarr_path, chunks='auto')
        
        self.n_samples = self.dataset.sizes['time']
        random_seed = random_seed + rank * 10000
        
        self.task_config = task_config
        self.batch_size = batch_size 
        self.n_target_steps = n_target_steps
        self.prefetch_size = prefetch_size
        self.num_workers = num_workers
        
        # Thread-safe components
        self.lock = threading.Lock()
        self.seed_generator = SeedGenerator(initial_seed=random_seed)
        
        # Prefetch queue
        self.batch_queue = Queue(maxsize=prefetch_size)
        
        # Control flags
        self.is_running = False
        self.worker_threads = []
        self.batches_generated = 0
        
        print(f"Rank {rank}: Generator initialized")
        print(f"  Samples: {self.n_samples}")
        print(f"  Batch size: {batch_size}")
        print(f"  Prefetch: {prefetch_size}, Workers: {num_workers}")
    
    def _get_random_indices(self, seed):
        """Generate random indices for batch sampling"""
        rs = np.random.RandomState(seed)
        max_start_idx = self.n_samples - self.n_target_steps - 2
        
        if max_start_idx <= 0:
            raise ValueError(f"Not enough samples. Need at least {self.n_target_steps + 2}")
        
        # Add a guard if max_start_idx < batch_size
        size = min(self.batch_size, max_start_idx)
        random_indices = rs.choice(max_start_idx, size=size, replace=False)
        return random_indices
    
    def _load_single_sample(self, start_idx):
        """Load a single sample efficiently"""
        end_idx = start_idx + self.n_target_steps + 2
        dataset_slice = self.dataset.isel(time=slice(start_idx, end_idx))

        #Hres has missing values in geopotential, if exists, do interpolation
        if dataset_slice.geopotential.isnull().any().any():
            print(f"dataset {dataset_slice.time.values} has NaNs!")
            dataset_slice = dataset_slice.interpolate_na(dim='lon', method='linear', fill_value='extrapolate')

        dataset_slice = dataset_slice.assign_coords(datetime=dataset_slice.time)
        dataset_slice['time'] = dataset_slice['time'] - dataset_slice.time[0]
        return dataset_slice
    
    def _generate_batch(self, seed):
        """Generate a single batch"""
        try:
            random_indices = self._get_random_indices(seed)
            
            # Load samples
            sample_slices = []
            for idx in random_indices:
                sample = self._load_single_sample(int(idx))
                sample_slices.append(sample)
            
            # Concatenate
            if len(sample_slices) > 1:
                dataset_slice = xr.concat(sample_slices, dim='batch')
            else:
                dataset_slice = sample_slices[0].expand_dims(dim='batch')
            
            # Fix datetime
            if 'datetime' in dataset_slice.coords:
                if 'batch' not in dataset_slice['datetime'].dims:
                    dataset_slice['datetime'] = dataset_slice['datetime'].expand_dims(dim='batch')
            
            # Squeeze static variables
            for var in ['geopotential_at_surface', 'land_sea_mask']:
                if var in dataset_slice and 'batch' in dataset_slice[var].dims:
                    dataset_slice[var] = dataset_slice[var].isel(batch=0)
            
            ## Add precipitation if not exists
            #if 'total_precipitation_6hr' not in dataset_slice:
            #    DIMS = ['batch', 'time', 'lat', 'lon']
            #    zeros_shape = tuple(dataset_slice.sizes[dim] for dim in DIMS)
            #    zeros_array = np.zeros(zeros_shape, dtype=np.float32)
            #    dataset_slice['total_precipitation_6hr'] = (DIMS, zeros_array)
            
            # Convert to static
            dataset_slice = self._to_static_vars(dataset_slice)
            
            # Extract
            inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
                dataset_slice,
                target_lead_times=slice("6h", f"{self.n_target_steps*6}h"),
                **dataclasses.asdict(self.task_config)
            )
            
            # Compute
            inputs, targets, forcings = dask.compute(inputs, targets, forcings)
            
            return inputs, targets, forcings
            
        except Exception as e:
            print(f"Rank {self.rank}: Error generating batch: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _to_static_vars(self, dataset, variables=['geopotential_at_surface', 'land_sea_mask']):
        """Convert time-varying variables to static"""
        for var in variables:
            if var in dataset and 'time' in dataset[var].dims:
                static_var = dataset[var].isel(time=0).drop_vars(['time'])
                dataset = dataset.drop_vars(var)
                dataset[var] = static_var
        return dataset
    
    def _worker_loop(self):
        """Worker thread loop"""
        while self.is_running:
            try:
                # Check if queue is full
                if self.batch_queue.qsize() >= self.prefetch_size:
                    time.sleep(0.1)
                    continue
                
                # Generate batch
                seed = self.seed_generator.get_seed()
                batch = self._generate_batch(seed)
                
                if batch is not None and self.is_running:
                    try:
                        self.batch_queue.put(batch, timeout=10)
                        with self.lock:
                            self.batches_generated += 1
                    except:
                        if self.is_running:
                            print(f"Rank {self.rank}: Queue put timeout")
                
            except Exception as e:
                if self.is_running:
                    print(f"Rank {self.rank}: Worker error: {e}")
                time.sleep(0.5)
    
    def start_prefetching(self):
        """Start background prefetching"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start worker threads
        for i in range(self.num_workers):
            thread = threading.Thread(target=self._worker_loop, daemon=True)
            thread.start()
            self.worker_threads.append(thread)
        
        # Wait for first batch
        time.sleep(1.0)
        
        if self.rank == 0:
            print(f"Prefetching started with {self.num_workers} workers")
    
    def stop_prefetching(self):
        """Stop background prefetching"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Wait for workers to finish
        for thread in self.worker_threads:
            if thread.is_alive():
                thread.join(timeout=2.0)
        
        self.worker_threads = []
        
        # Clear queue
        while not self.batch_queue.empty():
            try:
                self.batch_queue.get_nowait()
            except:
                break
        
        if self.rank == 0:
            print("Prefetching stopped")
    
    def generate(self, timeout=60):
        """Get next batch"""
        # Ensure prefetching is running
        if not self.is_running:
            self.start_prefetching()
        
        try:
            batch = self.batch_queue.get(timeout=timeout)
            return batch
        except Empty:
            print(f"Rank {self.rank}: Warning - queue empty, loading synchronously")
            seed = self.seed_generator.get_seed()
            return self._generate_batch(seed)
    
    def get_statistics(self):
        """Get loading statistics"""
        return {
            'batches_generated': self.batches_generated,
            'queue_size': self.batch_queue.qsize(),
            'workers_alive': sum(1 for t in self.worker_threads if t.is_alive()),
        }
    
    def __del__(self):
        """Cleanup"""
        try:
            self.stop_prefetching()
        except:
            pass

    def __len__(self):
        return self.n_samples
        
        
# Example usage
if __name__ == "__main__":
    from mpi4py import MPI
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Example configuration
    zarr_path = "/scratch3/NAGAPE/gpu-ai4wp/Linlin.Cui/graphcast_37/data/hres_13pl_2016-2021-6h-1440x721.zarr"
    initial_params_path = "/scratch3/NCEPDEV/nems/Linlin.Cui/gc_weights/params"

    ckpt = checkpoint.load(
        f'{initial_params_path}/GraphCast_operational - ERA5-HRES 1979-2021 - '
        'resolution 0.25 - pressure levels 13 - mesh 2to6 - precipitation output only.npz',
        graphcast.CheckPoint
    )
    task_config = ckpt.task_config
    
    try:
        # Create generator
        generator = SingleZarrDataGenerator(
            zarr_path=zarr_path,
            task_config=task_config,
            n_target_steps=1,
            batch_size=1,
            prefetch_size=1,
            num_workers=1,
            rank=rank,
            size=size,
        )
        
        # Start prefetching
        generator.start_prefetching()
        
        # Generate batches
        for i in range(10):
            inputs, targets, forcings = generator.generate()
            print(f"Rank {rank}: Generated batch {i+1}")
            comm.Barrier()
        
    finally:
        # Cleanup
        if generator is not None:
            generator.stop_prefetching()
