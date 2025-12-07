# OpenMP thread management
# With 3 workers, auto thread count works fine (no OMP warnings observed)
# Only limit threads if using 4+ workers to prevent memory contention
import os
import json

# Load config to get thread count setting
with open('config.json', 'r') as f:
    config = json.load(f)

# Get thread setting from config (None/"auto" means don't set, use system default)
omp_threads = config.get('gpu', {}).get('omp_threads_per_worker', 'auto')

if omp_threads != 'auto':
    omp_threads = str(omp_threads)
    os.environ['OMP_NUM_THREADS'] = omp_threads
    os.environ['MKL_NUM_THREADS'] = omp_threads
    os.environ['OPENBLAS_NUM_THREADS'] = omp_threads
    os.environ['NUMEXPR_NUM_THREADS'] = omp_threads

import multiprocessing

# CRITICAL: Set multiprocessing to 'spawn' mode for CUDA compatibility
if __name__ == "__main__":
    if multiprocessing.get_start_method(allow_none=True) != 'spawn':
        multiprocessing.set_start_method('spawn', force=True)

from learning.training_loop import AlphaZeroTrainer

if __name__ == "__main__":
    print("Initializing AlphaZero Training Pipeline...")
    
    # This will load config.json, setup the environment, and run the self-play loop
    trainer = AlphaZeroTrainer(config_path="config.json")
    
    # Start Training
    trainer.run_training_loop()