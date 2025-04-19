"""
Memory optimization utilities for Nanotron.

This module provides utilities for optimizing memory usage during training,
particularly for large language models where memory can be a bottleneck.
"""

import gc
import torch
from typing import Optional, List, Dict, Any, Union, Callable


def log_memory_usage(logger=None, prefix: str = ""):
    """
    Log current GPU memory usage.
    
    Args:
        logger: Logger to use. If None, print to stdout.
        prefix: Prefix for the log message.
    """
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    max_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)
    
    message = (
        f"{prefix} Memory: "
        f"Allocated: {allocated:.2f}GB (Peak: {max_allocated:.2f}GB), "
        f"Reserved: {reserved:.2f}GB (Peak: {max_reserved:.2f}GB)"
    )
    
    if logger:
        logger.info(message)
    else:
        print(message)


def clear_memory_cache(tensors_to_clear: Optional[List[torch.Tensor]] = None):
    """
    Clear memory cache and optionally delete specific tensors.
    
    Args:
        tensors_to_clear: List of tensors to explicitly delete before clearing cache.
    """
    if tensors_to_clear:
        for tensor in tensors_to_clear:
            if tensor is not None:
                del tensor
    
    # Run garbage collection
    gc.collect()
    
    # Clear CUDA cache
    torch.cuda.empty_cache()


class MemoryOptimizedFunction:
    """
    Context manager for memory-optimized function execution.
    
    This context manager helps optimize memory usage by:
    1. Clearing memory before execution
    2. Optionally using mixed precision during execution
    3. Clearing memory after execution
    4. Logging memory usage before and after
    
    Example:
        with MemoryOptimizedFunction(logger=logger, name="loss_calculation"):
            loss = model.compute_loss(logits, labels)
    """
    
    def __init__(
        self, 
        logger=None, 
        name: str = "function", 
        clear_before: bool = True, 
        clear_after: bool = True,
        use_mixed_precision: bool = False
    ):
        self.logger = logger
        self.name = name
        self.clear_before = clear_before
        self.clear_after = clear_after
        self.use_mixed_precision = use_mixed_precision
        self.amp_context = torch.cuda.amp.autocast() if use_mixed_precision else None
    
    def __enter__(self):
        if self.clear_before:
            clear_memory_cache()
        
        log_memory_usage(self.logger, f"Before {self.name}")
        
        if self.use_mixed_precision and self.amp_context:
            self.amp_context.__enter__()
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.use_mixed_precision and self.amp_context:
            self.amp_context.__exit__(exc_type, exc_val, exc_tb)
        
        if self.clear_after:
            clear_memory_cache()
        
        log_memory_usage(self.logger, f"After {self.name}")


def chunk_batch(
    batch: Dict[str, torch.Tensor], 
    chunk_size: int
) -> List[Dict[str, torch.Tensor]]:
    """
    Split a batch into smaller chunks to reduce memory usage.
    
    Args:
        batch: Dictionary of tensors representing a batch
        chunk_size: Maximum size of each chunk
        
    Returns:
        List of dictionaries, each representing a chunk of the original batch
    """
    batch_size = next(iter(batch.values())).size(0)
    num_chunks = (batch_size + chunk_size - 1) // chunk_size  # Ceiling division
    
    chunks = []
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, batch_size)
        
        chunk = {k: v[start_idx:end_idx] for k, v in batch.items()}
        chunks.append(chunk)
    
    return chunks


def process_in_chunks(
    batch: Dict[str, torch.Tensor],
    process_fn: Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]],
    chunk_size: int,
    reduction_fn: Optional[Callable[[List[Dict[str, torch.Tensor]]], Dict[str, torch.Tensor]]] = None
) -> Dict[str, torch.Tensor]:
    """
    Process a batch in chunks to reduce memory usage.
    
    Args:
        batch: Dictionary of tensors representing a batch
        process_fn: Function to process each chunk
        chunk_size: Maximum size of each chunk
        reduction_fn: Function to combine results from all chunks
                     If None, results will be concatenated along dim 0
                     
    Returns:
        Dictionary of processed results
    """
    chunks = chunk_batch(batch, chunk_size)
    chunk_results = []
    
    for chunk in chunks:
        # Process chunk and clear memory after each chunk
        with torch.cuda.amp.autocast(enabled=True):
            result = process_fn(chunk)
        chunk_results.append(result)
        clear_memory_cache()
    
    # Combine results
    if reduction_fn is not None:
        return reduction_fn(chunk_results)
    
    # Default reduction: concatenate tensors along dim 0
    combined_result = {}
    for key in chunk_results[0].keys():
        if isinstance(chunk_results[0][key], torch.Tensor):
            combined_result[key] = torch.cat([r[key] for r in chunk_results], dim=0)
        else:
            # For non-tensor values, just use the first chunk's value
            combined_result[key] = chunk_results[0][key]
    
    return combined_result
