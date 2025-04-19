"""
Utility functions for Nanotron.
"""

import random
import socket
import torch
from packaging import version

from .memory_utils import (
    log_memory_usage,
    clear_memory_cache,
    MemoryOptimizedFunction,
    chunk_batch,
    process_in_chunks,
)

# Import classes and functions from the main utils.py
class Singleton(type):
    """
    Singleton metaclass.
    Create objects using this class as the metaclass to enable singleton behaviour.
    For instance:
    ```
    class Logger(metaclass=Singleton):
      ...
    ```
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
def find_free_port(min_port: int = 2000, max_port: int = 65000) -> int:
    while True:
        port = random.randint(min_port, max_port)
        try:
            with socket.socket() as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(("localhost", port))
                return port
        except OSError:
            continue

def get_untyped_storage(tensor: torch.Tensor) -> torch.UntypedStorage:
    if version.parse(torch.__version__) >= version.parse("2.0"):
        return tensor.untyped_storage()
    else:
        return tensor.storage().untyped()

def tensor_from_untyped_storage(untyped_storage: torch.UntypedStorage, dtype: torch.dtype):
    # Figure out what's the best Pytorch way of building a tensor from a storage.
    device = untyped_storage.device
    tensor = torch.empty([], dtype=dtype, device=device)
    tensor.set_(source=untyped_storage)
    return tensor

__all__ = [
    "log_memory_usage",
    "clear_memory_cache",
    "MemoryOptimizedFunction",
    "chunk_batch",
    "process_in_chunks",
    "find_free_port",
    "get_untyped_storage",
    "tensor_from_untyped_storage",
    "Singleton",
]
