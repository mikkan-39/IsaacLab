#!/usr/bin/env python3
"""Launch TensorBoard for IsaacLab training logs."""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Launch TensorBoard for IsaacLab training logs")
    parser.add_argument(
        "--logdir",
        type=str,
        default="logs/rsl_rl/",
        help="Path to training logs directory (default: logs/rsl_rl/)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=6006,
        help="Port to serve TensorBoard on (default: 6006)",
    )
    
    args = parser.parse_args()
    
    # Check if logdir exists
    logdir_path = Path(args.logdir)
    if not logdir_path.exists():
        print(f"Error: Log directory '{args.logdir}' does not exist.")
        sys.exit(1)
    
    # Simple: just tell the user how to run it
    logdir_abs = logdir_path.resolve()
    print(f"TensorBoard launcher for IsaacLab")
    print(f"  Log directory: {logdir_abs}")
    print(f"  Port: {args.port}")
    print()
    print(f"Run this in a terminal:")
    print(f"  tensorboard --logdir {logdir_abs} --port {args.port} --bind_all")
    print()
    print(f"Then open:")
    print(f"  http://localhost:{args.port}/")
    print()


if __name__ == "__main__":
    main()


