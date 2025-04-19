import os
import sys
import torch
import pickle
import torch.distributed as dist

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath("nanotron/src"))

from nanotron.config import Config
from nanotron.trainer import DistributedTrainer
from nanotron.logging import log_rank, get_logger

logger = get_logger(__name__)

def profile_memory_usage():
    # Initialize distributed environment if not already done
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    # Load your configuration
    config_path = "nanotron/examples/config_tiny_llama.yaml"  # Using an existing config file
    config = Config.from_file(config_path)

    # Limit to just one iteration for profiling
    original_train_steps = config.tokens.train_steps
    config.tokens.train_steps = 1

    # Initialize the trainer
    trainer = DistributedTrainer(config)

    # Initialize the dataloader
    # We'll use a simple dummy dataloader for testing
    import torch

    def dummy_dataloader():
        batch_size = config.tokens.micro_batch_size
        seq_len = config.tokens.sequence_length
        vocab_size = config.model.model_config.vocab_size

        while True:
            # Create dummy batch
            batch = {
                "input_ids": torch.randint(0, vocab_size, (batch_size, seq_len), device="cuda"),
                "input_mask": torch.ones(batch_size, seq_len, device="cuda", dtype=torch.bool),
                "label_ids": torch.randint(0, vocab_size, (batch_size, seq_len), device="cuda"),
                "label_mask": torch.ones(batch_size, seq_len, device="cuda", dtype=torch.bool),
            }
            yield batch

    dataloader = dummy_dataloader()

    # Set the iteration step to match your initial_iter_step
    trainer.iteration_step = trainer.initial_iter_step

    # Create a directory for memory snapshots if it doesn't exist
    os.makedirs("memory_snapshots", exist_ok=True)

    # Enable CUDA memory recording right before the training step
    log_rank("Enabling CUDA memory recording...", logger=logger)
    torch.cuda.memory._record_memory_history(max_entries=100000)

    # Run a single training step
    log_rank("Running a single training step with memory profiling...", logger=logger)
    try:
        outputs, loss_avg, z_loss_avg = trainer.training_step(dataloader=dataloader)
        log_rank(f"Training step completed successfully. Loss: {loss_avg.item() if loss_avg is not None else 'N/A'}",
                logger=logger)
    except Exception as e:
        log_rank(f"Error during training step: {str(e)}", logger=logger)
        import traceback
        log_rank(traceback.format_exc(), logger=logger)

    # Dump the memory snapshot
    rank = dist.get_rank() if dist.is_initialized() else 0
    snapshot_path = f"memory_snapshots/memory_snapshot_rank{rank}.pkl"
    log_rank(f"Dumping memory snapshot to {snapshot_path}...", logger=logger)
    torch.cuda.memory._dump_snapshot(snapshot_path)

    # Disable memory recording
    torch.cuda.memory._record_memory_history(enabled=None)

    log_rank(f"Memory profiling complete. Snapshot saved to {snapshot_path}", logger=logger)

    # Restore original train steps
    config.tokens.train_steps = original_train_steps

if __name__ == "__main__":
    profile_memory_usage()