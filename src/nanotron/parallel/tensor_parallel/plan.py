"""Module for defining and implementing tensor parallel plans.

A tensor parallel plan specifies which model components should be parallelized 
and how they should be sharded (row-wise vs column-wise).
"""

from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple, Union

import torch.nn as nn

from nanotron import distributed as dist
from nanotron.parallel.tensor_parallel.enum import TensorParallelLinearMode


class ShardingType(Enum):
    """The type of sharding for a tensor parallel component."""
    ROW = auto()  # Sharded along input dimension
    COLUMN = auto()  # Sharded along output dimension
    EMBEDDING = auto()  # Special case for embedding tables
    NONE = auto()  # No sharding (replicated)


class TPPlanEntry:
    """An entry in a tensor parallel plan.
    
    Specifies how a specific module or parameter should be sharded.
    """
    
    def __init__(
        self,
        name: str,
        sharding_type: ShardingType,
        mode: TensorParallelLinearMode = TensorParallelLinearMode.ALL_REDUCE,
        async_communication: bool = False,
        contiguous_chunks: Optional[Tuple[int, ...]] = None,
        tp_recompute_allgather: bool = True,
    ):
        """Initialize a tensor parallel plan entry.
        
        Args:
            name: The name of the module or parameter in the model
            sharding_type: How to shard the module (ROW, COLUMN, EMBEDDING, NONE)
            mode: The tensor parallel mode (ALL_REDUCE or REDUCE_SCATTER)
            async_communication: Whether to use asynchronous communication
            contiguous_chunks: Optional specification of contiguous chunks for sharding
            tp_recompute_allgather: Whether to recompute the all-gather operation in backward pass
        """
        self.name = name
        self.sharding_type = sharding_type
        self.mode = mode
        self.async_communication = async_communication
        self.contiguous_chunks = contiguous_chunks
        self.tp_recompute_allgather = tp_recompute_allgather
    
    def __repr__(self) -> str:
        return (
            f"TPPlanEntry(name={self.name}, sharding_type={self.sharding_type}, "
            f"mode={self.mode}, async_communication={self.async_communication}, "
            f"contiguous_chunks={self.contiguous_chunks}, "
            f"tp_recompute_allgather={self.tp_recompute_allgather})"
        )


class TensorParallelPlan:
    """A plan for tensor parallelism, specifying which components to parallelize and how."""
    
    def __init__(self):
        """Initialize an empty tensor parallel plan."""
        self.entries: Dict[str, TPPlanEntry] = {}
    
    def add_entry(self, entry: TPPlanEntry) -> None:
        """Add an entry to the plan.
        
        Args:
            entry: The tensor parallel plan entry to add
        """
        self.entries[entry.name] = entry
    
    def get_entry(self, name: str) -> Optional[TPPlanEntry]:
        """Get the entry for a named module if it exists.
        
        Args:
            name: The name of the module in the model
        
        Returns:
            The tensor parallel plan entry if exists, None otherwise
        """
        return self.entries.get(name)
    
    def get_all_entries(self) -> Dict[str, TPPlanEntry]:
        """Get all entries in the plan.
        
        Returns:
            Dictionary mapping module names to tensor parallel plan entries
        """
        return self.entries
    
    def get_entries_by_type(self, sharding_type: ShardingType) -> Dict[str, TPPlanEntry]:
        """Get all entries of a specific sharding type.
        
        Args:
            sharding_type: The type of sharding to filter by
        
        Returns:
            Dictionary mapping module names to tensor parallel plan entries
        """
        return {name: entry for name, entry in self.entries.items() 
                if entry.sharding_type == sharding_type}
    
    def apply_to_model(
        self, 
        model: nn.Module, 
        tp_pg: dist.ProcessGroup,
        parallel_config: Optional[Dict] = None,
    ) -> None:
        """Apply the tensor parallel plan to a model.
        
        This method traverses the model and replaces modules with their tensor parallel
        equivalents according to the plan.
        
        Args:
            model: The model to apply tensor parallelism to
            tp_pg: The tensor parallel process group
            parallel_config: Optional configuration for tensor parallelism
        """
        # This is a stub implementation - in practice, this would iterate through the model
        # and replace modules with their tensor parallel equivalents according to the plan
        
        raise NotImplementedError(
            "The apply_to_model method must be implemented by subclasses based on the specific model architecture."
        )