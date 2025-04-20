#!/usr/bin/env python
"""
Example script demonstrating how to launch Nanotron with different backends.

This script serves as a reference for users who want to:
1. Launch with NCCL backend for GPU training (recommended for performance)
2. Launch with Gloo backend for CPU-only or debugging environments
3. Configure based on available hardware

Usage:
    # For GPU training with NCCL backend (recommended)
    python launch_with_backend.py --backend nccl
    
    # For CPU-only training or debugging with Gloo backend
    python launch_with_backend.py --backend gloo
    
    # Let the system choose based on available hardware
    python launch_with_backend.py --backend auto
"""

import argparse
import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from nanotron.parallel import ParallelContext


def parse_args():
    parser = argparse.ArgumentParser(description="Launch example with different backends")
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["nccl", "gloo", "auto"],
        help="Backend to use for distributed training"
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Size of tensor parallelism"
    )
    parser.add_argument(
        "--pipeline_parallel_size",
        type=int,
        default=1,
        help="Size of pipeline parallelism"
    )
    parser.add_argument(
        "--data_parallel_size",
        type=int,
        default=1,
        help="Size of data parallelism"
    )
    return parser.parse_args()


def run_worker(rank, world_size, args):
    """Worker function for each process."""
    
    # Set environment variables for distributed training
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank % torch.cuda.device_count() if torch.cuda.is_available() else rank)
    os.environ["LOCAL_WORLD_SIZE"] = str(min(world_size, torch.cuda.device_count() if torch.cuda.is_available() else world_size))
    
    # Set environment variables for torch.distributed initialization
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    
    # Select backend based on args and available hardware
    backend = args.backend
    if backend == "auto":
        backend = "nccl" if torch.cuda.is_available() else "gloo"
    
    # Initialize ParallelContext - it will handle warnings and validations internally
    parallel_context = ParallelContext(
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        data_parallel_size=args.data_parallel_size,
        backend=backend,
    )
    
    # Print information about the process
    if rank == 0:
        print(f"Initialized with backend: {backend}")
        print(f"World size: {world_size}")
        print(f"Tensor parallel size: {args.tensor_parallel_size}")
        print(f"Pipeline parallel size: {args.pipeline_parallel_size}")
        print(f"Data parallel size: {args.data_parallel_size}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"CUDA current device: {torch.cuda.current_device()}")
    
    # Barrier for synchronized output
    dist.barrier()
    
    # Clean up
    parallel_context.destroy()


def main():
    args = parse_args()
    
    # Calculate world size
    world_size = args.tensor_parallel_size * args.pipeline_parallel_size * args.data_parallel_size
    
    # Validate configuration
    if args.tensor_parallel_size > 1 and (args.backend == "gloo" or not torch.cuda.is_available()):
        print("WARNING: Tensor parallelism > 1 requires NCCL backend and GPU availability.")
        print("Training will fail with this configuration.")
        if not torch.cuda.is_available():
            print("No GPUs detected. Please use tensor_parallel_size=1 for CPU-only training.")
            return 1
    
    # Launch processes
    if world_size > 1:
        mp.spawn(run_worker, args=(world_size, args), nprocs=world_size, join=True)
    else:
        run_worker(0, 1, args)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())