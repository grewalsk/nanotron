import numpy as np
import pytest
import torch.distributed as dist
from helpers.utils import (
    available_gpus,
    get_all_3d_configurations,
    init_distributed,
    rerun_if_address_is_in_use,
)
from nanotron.parallel import ParallelContext
from torch.distributed import ProcessGroup


def _test_init_parallel_context(parallel_context: ParallelContext):
    assert dist.is_initialized() is True
    assert isinstance(parallel_context.world_pg, ProcessGroup)
    assert isinstance(parallel_context.tp_pg, ProcessGroup) if parallel_context.tensor_parallel_size > 1 else True
    assert isinstance(parallel_context.pp_pg, ProcessGroup) if parallel_context.pipeline_parallel_size > 1 else True
    assert isinstance(parallel_context.dp_pg, ProcessGroup) if parallel_context.data_parallel_size > 1 else True

    world_rank = dist.get_rank(parallel_context.world_pg)

    assert isinstance(parallel_context.world_rank_matrix, np.ndarray)
    assert isinstance(parallel_context.world_ranks_to_pg, dict)

    local_rank = tuple(i.item() for i in np.where(parallel_context.world_rank_matrix == world_rank))
    global_rank = parallel_context.get_global_rank(*local_rank)
    assert isinstance(global_rank, np.int64), f"The type of global_rank is {type(global_rank)}"

    assert global_rank == dist.get_rank()

    parallel_context.destroy()
    assert dist.is_initialized() is False


@pytest.mark.parametrize(
    "tp,dp,pp",
    [
        pytest.param(*all_3d_configs)
        for gpus in range(1, min(available_gpus(), 4) + 1)
        for all_3d_configs in get_all_3d_configurations(gpus)
    ],
)
@rerun_if_address_is_in_use()
def test_init_parallel_context(tp: int, dp: int, pp: int):
    init_distributed(tp=tp, dp=dp, pp=pp)(_test_init_parallel_context)()


def _test_get_global_rank(parallel_context: ParallelContext):
    """Test the new get_global_rank method against direct world_rank_matrix access."""
    
    # Test various combinations of ranks
    for ep_rank in range(parallel_context.expert_parallel_size):
        for pp_rank in range(parallel_context.pipeline_parallel_size):
            for dp_rank in range(parallel_context.data_parallel_size):
                for cp_rank in range(parallel_context.context_parallel_size):
                    for tp_rank in range(parallel_context.tensor_parallel_size):
                        # Direct access to world_rank_matrix (to be deprecated)
                        direct_global_rank = int(parallel_context.world_rank_matrix[
                            ep_rank, pp_rank, dp_rank, cp_rank, tp_rank
                        ])
                        
                        # New API method using named parameters
                        api_global_rank = parallel_context.get_global_rank(
                            expert_parallel_rank=ep_rank,
                            pipeline_parallel_rank=pp_rank,
                            data_parallel_rank=dp_rank,
                            context_parallel_rank=cp_rank,
                            tensor_parallel_rank=tp_rank,
                        )
                        
                        # Check they match
                        assert direct_global_rank == api_global_rank, (
                            f"Rank mismatch for ep={ep_rank}, pp={pp_rank}, dp={dp_rank}, "
                            f"cp={cp_rank}, tp={tp_rank}: "
                            f"world_rank_matrix gives {direct_global_rank}, "
                            f"get_global_rank gives {api_global_rank}"
                        )
    
    # Test the current process's rank
    current_ranks = parallel_context.get_local_ranks(dist.get_rank())
    current_global_rank = parallel_context.get_global_rank(
        expert_parallel_rank=current_ranks["ep"],
        pipeline_parallel_rank=current_ranks["pp"],
        data_parallel_rank=current_ranks["dp"],
        context_parallel_rank=current_ranks["cp"],
        tensor_parallel_rank=current_ranks["tp"],
    )
    
    assert current_global_rank == dist.get_rank(), (
        f"Current global rank mismatch: dist.get_rank()={dist.get_rank()}, "
        f"get_global_rank gives {current_global_rank}"
    )
    
    parallel_context.destroy()


@pytest.mark.parametrize(
    "tp,dp,pp",
    [
        pytest.param(*all_3d_configs)
        for gpus in range(1, min(available_gpus(), 4) + 1)
        for all_3d_configs in get_all_3d_configurations(gpus)
    ],
)
@rerun_if_address_is_in_use()
def test_get_global_rank(tp: int, dp: int, pp: int):
    """Test the new get_global_rank API for the ParallelContext class."""
    init_distributed(tp=tp, dp=dp, pp=pp)(_test_get_global_rank)()
