# Tensor Parallelism Improvements in Nanotron

This document describes the improvements made to the tensor parallelism functionality in Nanotron.

## 1. Overview of Changes

We've made several enhancements to Nanotron's tensor parallelism capabilities:

1. **Opt-in Tensor Parallelism**: Model classes can now explicitly opt-in to tensor parallelism support.
2. **Improved ParallelContext API**: Replaced direct world_rank_matrix access with a more intuitive API.
3. **Centralized TP Initialization**: Relocated the TP initialization hook to the base NanotronModel class.
4. **Backend Flexibility**: Added support for different communication backends (NCCL/Gloo).
5. **TensorParallelPlan**: A structured way to define and apply tensor parallelism to models.
6. **Clear Documentation**: Added comprehensive documentation for tensor parallel implementation.

## 2. Opt-in Tensor Parallelism

Model classes can now opt-in to tensor parallelism by:

1. Setting a class attribute `supports_tensor_parallel = True`
2. Inheriting from `TensorParallelMixin` before the base model class
3. Implementing the `_init_tensor_parallel()` method

```python
from nanotron.parallel.tensor_parallel.nn import TensorParallelMixin
from nanotron.models import NanotronModel

class MyModel(TensorParallelMixin, NanotronModel):
    supports_tensor_parallel = True
    
    def __init__(self, config, parallel_context):
        super().__init__(config)
        self._init_tensor_parallel(parallel_context)
        
    def _init_tensor_parallel(self, parallel_context):
        # Model-specific tensor parallelism initialization
        if not hasattr(self, "supports_tensor_parallel") or not self.supports_tensor_parallel:
            return
            
        if parallel_context is None or parallel_context.tensor_parallel_size <= 1:
            return
            
        # TP initialization logic here...
```

## 3. Improved ParallelContext API

We've modernized the ParallelContext API by adding a `get_global_rank` method that provides a more intuitive interface than direct indexing of the `world_rank_matrix`.

**Old approach (deprecated):**
```python
global_rank = parallel_context.world_rank_matrix[
    expert_parallel_rank,
    pipeline_parallel_rank, 
    data_parallel_rank,
    context_parallel_rank,
    tensor_parallel_rank
]
```

**New approach:**
```python
global_rank = parallel_context.get_global_rank(
    expert_parallel_rank=expert_parallel_rank,
    pipeline_parallel_rank=pipeline_parallel_rank,
    data_parallel_rank=data_parallel_rank,
    context_parallel_rank=context_parallel_rank,
    tensor_parallel_rank=tensor_parallel_rank
)
```

The new approach:
- Provides better documentation through parameter names
- Adds type safety
- Allows for future changes to the implementation without breaking API
- Returns an `int` type instead of `np.int64`

## 4. TensorParallelPlan

The `TensorParallelPlan` class provides a structured way to define and apply tensor parallelism to models:

```python
from nanotron.parallel.tensor_parallel.plan import TensorParallelPlan, TPPlanEntry, ShardingType
from nanotron.parallel.tensor_parallel.enum import TensorParallelLinearMode

# Create a tensor parallel plan for a model
plan = TensorParallelPlan()

# Add entries for different components
plan.add_entry(TPPlanEntry(
    name="attention.qkv_proj",
    sharding_type=ShardingType.COLUMN,
    mode=TensorParallelLinearMode.ALL_REDUCE,
    async_communication=False
))

plan.add_entry(TPPlanEntry(
    name="attention.o_proj",
    sharding_type=ShardingType.ROW,
    mode=TensorParallelLinearMode.ALL_REDUCE,
    async_communication=False
))

# Apply plan to model implementation
```

Each plan entry specifies:
- The module or parameter name
- The type of sharding (ROW, COLUMN, EMBEDDING, NONE)
- The tensor parallel mode (ALL_REDUCE or REDUCE_SCATTER)
- Additional configuration parameters

## 5. Centralized TP Initialization

We've relocated the tensor parallelism initialization hook to the base NanotronModel class:

```python
# In NanotronModel.__init__
def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.parallel_context: "ParallelContext" = kwargs.get("parallel_context", None)
    # ... other initialization code ...
    
    # Initialize tensor parallelism if the model supports it
    if hasattr(self, "_init_tensor_parallel"):
        self._init_tensor_parallel(self.parallel_context)
```

Benefits:
- Consistent initialization across all model types
- Prevents subclasses from forgetting to call the initialization method
- Standardizes the timing of tensor parallelism setup

## 6. Backend Flexibility

We've improved support for different communication backends:

```python
# Automatic backend selection based on hardware
if torch.cuda.is_available():
    if backend != "nccl":
        warnings.warn("Switching backend from {backend} to nccl for GPU execution")
        backend = "nccl"
else:
    if backend == "nccl":
        warnings.warn("NCCL backend requested but no GPUs available, using gloo instead")
        backend = "gloo"
```

- **NCCL**: Used automatically when GPUs are available (best performance)
- **Gloo**: Used for CPU-only training or when requested explicitly
- **Validation**: Prevents invalid configurations (e.g., TP > 1 with Gloo)

We've created a sample launch script showing how to use different backends:
```python
# Launch with NCCL for GPU training
python launch_with_backend.py --backend nccl
    
# Launch with Gloo for CPU-only training
python launch_with_backend.py --backend gloo
    
# Let the system choose based on hardware
python launch_with_backend.py --backend auto
```

## 7. Testing and Documentation

We've added:
- Unit tests for the new `get_global_rank` method
- Support for testing on CPU-only environments with Gloo
- Comprehensive documentation in CONTRIBUTING.md
- A detailed guide on implementing tensor parallelism

## 8. Usage in Models like LLaVA

LLaVA models can now leverage tensor parallelism by:
1. Implementing the TensorParallelMixin
2. Creating a tensor parallel plan for the vision encoder and cross-attention modules
3. Appropriately sharding linear layers, attention blocks, and embedding tables

This enables efficient training of large vision-language models across multiple GPUs.