#!/usr/bin/env python3
"""
CLI tool to find unused GPUs by checking nvidia-smi memory usage.
GPUs with memory usage below 300MB are considered unused.
"""

import argparse
import subprocess
import sys
from typing import List, Dict, Any


def get_gpu_info() -> List[Dict[str, Any]]:
    """Query nvidia-smi and return GPU information as structured data."""
    try:
        # Use nvidia-smi with JSON output for easy parsing
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            check=True
        )

        gpus = []
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                parts = [part.strip() for part in line.split(',')]
                if len(parts) >= 4:
                    gpu_info = {
                        'index': int(parts[0]),
                        'name': parts[1],
                        'memory_used_mb': int(parts[2]),
                        'memory_total_mb': int(parts[3])
                    }
                    gpus.append(gpu_info)

        return gpus

    except subprocess.CalledProcessError as e:
        print(f"Error running nvidia-smi: {e}", file=sys.stderr)
        if e.stderr:
            print(f"nvidia-smi stderr: {e.stderr}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print("Error: nvidia-smi not found. Make sure NVIDIA drivers are installed.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


def find_free_gpus(gpu_info: List[Dict[str, Any]], threshold_mb: int = 300) -> List[Dict[str, Any]]:
    """Find GPUs with memory usage below the threshold."""
    free_gpus = []
    for gpu in gpu_info:
        if gpu['memory_used_mb'] < threshold_mb:
            free_gpus.append(gpu)
    return free_gpus


def format_output(free_gpus: List[Dict[str, Any]], verbose: bool = False) -> str:
    """Format the output for displaying free GPUs."""
    if not free_gpus:
        return "No free GPUs found."

    if verbose:
        output_lines = ["Free GPUs found:"]
        for gpu in free_gpus:
            output_lines.append(
                f"  GPU {gpu['index']}: {gpu['name']} "
                f"({gpu['memory_used_mb']}MB / {gpu['memory_total_mb']}MB used)"
            )
        return '\n'.join(output_lines)
    else:
        # Just return the indexes
        indexes = [str(gpu['index']) for gpu in free_gpus]
        return ' '.join(indexes)


def main():
    parser = argparse.ArgumentParser(
        description="Find GPUs that are currently not in use (memory usage < threshold).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Print indexes of free GPUs
  %(prog)s -v                 # Verbose output with details
  %(prog)s -t 500             # Use 500MB threshold instead of 300MB
  %(prog)s -q                 # Quiet mode (no output if no free GPUs)
        """
    )

    parser.add_argument(
        '-t', '--threshold',
        type=int,
        default=300,
        help='Memory usage threshold in MB to consider GPU as free (default: 300)'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show detailed information about free GPUs'
    )

    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Do not output anything if no free GPUs are found'
    )

    args = parser.parse_args()

    # Get GPU information
    gpu_info = get_gpu_info()

    if not gpu_info:
        print("No GPUs detected.", file=sys.stderr)
        sys.exit(1)

    # Find free GPUs
    free_gpus = find_free_gpus(gpu_info, args.threshold)

    # Output results
    if free_gpus:
        output = format_output(free_gpus, args.verbose)
        print(output)
    elif not args.quiet:
        print("No free GPUs found.")


if __name__ == "__main__":
    main()
