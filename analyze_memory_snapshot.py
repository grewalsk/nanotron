import pickle
import os
import sys
import argparse
from collections import defaultdict
import torch

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath("nanotron/src"))

def load_snapshot(snapshot_path):
    """Load a memory snapshot from a pickle file."""
    with open(snapshot_path, 'rb') as f:
        return pickle.load(f)

def analyze_snapshot(snapshot, top_n=20):
    """Analyze a memory snapshot and print the top N allocations."""
    # Group allocations by stack trace
    allocations_by_stack = defaultdict(int)
    stack_to_info = {}

    for item in snapshot:
        if item['action'] == 'a':  # Allocation
            size = item['size']
            stack_id = item['stack_id']
            stack = item['stack']

            allocations_by_stack[stack_id] += size
            if stack_id not in stack_to_info:
                stack_to_info[stack_id] = {
                    'stack': stack,
                    'size': 0,
                    'count': 0
                }
            stack_to_info[stack_id]['size'] += size
            stack_to_info[stack_id]['count'] += 1

    # Sort allocations by size (descending)
    sorted_allocations = sorted(
        [(stack_id, info['size'], info['count'], info['stack'])
         for stack_id, info in stack_to_info.items()],
        key=lambda x: x[1],
        reverse=True
    )

    # Print the top N allocations
    print(f"Top {top_n} memory allocations:")
    print("-" * 100)

    for i, (stack_id, size, count, stack) in enumerate(sorted_allocations[:top_n]):
        print(f"#{i+1}: {size/1024**2:.2f} MB ({count} allocations)")

        # Print the stack trace (focusing on the most relevant parts)
        relevant_frames = []
        for frame in stack:
            filename = frame[0]
            lineno = frame[1]
            name = frame[2]

            # Skip frames from PyTorch internals
            if 'torch/autograd' in filename and name != 'backward':
                continue

            # Focus on nanotron code
            if 'nanotron' in filename:
                relevant_frames.append(frame)

        # If we didn't find any relevant frames, show the first few frames
        if not relevant_frames:
            relevant_frames = stack[:3]

        for frame in relevant_frames:
            filename = frame[0]
            lineno = frame[1]
            name = frame[2]
            print(f"  {filename}:{lineno} - {name}")

        print("-" * 100)

    # Calculate total memory usage
    total_memory = sum(size for _, size, _, _ in sorted_allocations)
    print(f"Total allocated memory: {total_memory/1024**2:.2f} MB")

    # Check for specific functions of interest
    check_functions = [
        "sharded_cross_entropy",
        "Loss.forward",
        "LossWithZLoss.forward",
        "masked_mean",
        "_ShardedCrossEntropy.forward",
        "_ShardedCrossEntropyWithZLoss.forward"
    ]

    print("\nMemory usage by specific functions:")
    for func_name in check_functions:
        func_memory = 0
        for _, size, _, stack in sorted_allocations:
            for frame in stack:
                if func_name in frame[2]:
                    func_memory += size
                    break
        print(f"{func_name}: {func_memory/1024**2:.2f} MB")

    return {
        "total_memory": total_memory,
        "top_allocations": sorted_allocations[:top_n],
        "function_memory": {func: sum(size for _, size, _, stack in sorted_allocations
                                    if any(func in frame[2] for frame in stack))
                           for func in check_functions}
    }

def compare_snapshots(before_path, after_path, top_n=20):
    """Compare before and after snapshots to measure optimization impact."""
    before_snapshot = load_snapshot(before_path)
    after_snapshot = load_snapshot(after_path)

    print("=" * 50)
    print("BEFORE OPTIMIZATION")
    print("=" * 50)
    before_stats = analyze_snapshot(before_snapshot, top_n)

    print("\n" + "=" * 50)
    print("AFTER OPTIMIZATION")
    print("=" * 50)
    after_stats = analyze_snapshot(after_snapshot, top_n)

    # Calculate improvement percentages
    if before_stats["total_memory"] > 0:
        total_improvement = (1 - after_stats["total_memory"] / before_stats["total_memory"]) * 100
        print(f"\nTotal memory reduction: {total_improvement:.2f}%")

        print("\nMemory reduction by function:")
        for func in before_stats["function_memory"]:
            before_mem = before_stats["function_memory"][func]
            after_mem = after_stats["function_memory"].get(func, 0)
            if before_mem > 0:
                improvement = (1 - after_mem / before_mem) * 100
                print(f"{func}: {improvement:.2f}% reduction ({before_mem/1024**2:.2f} MB → {after_mem/1024**2:.2f} MB)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze memory snapshots")
    parser.add_argument("--snapshot", type=str, help="Path to the memory snapshot pickle file")
    parser.add_argument("--before", type=str, help="Path to the 'before optimization' snapshot")
    parser.add_argument("--after", type=str, help="Path to the 'after optimization' snapshot")
    parser.add_argument("--top_n", type=int, default=20, help="Number of top allocations to show")

    args = parser.parse_args()

    if args.before and args.after:
        compare_snapshots(args.before, args.after, args.top_n)
    elif args.snapshot:
        snapshot = load_snapshot(args.snapshot)
        analyze_snapshot(snapshot, args.top_n)
    else:
        print("Please provide either a single snapshot or before/after snapshots to compare")
