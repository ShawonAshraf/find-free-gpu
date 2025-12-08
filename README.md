# find-free-gpu

A CLI tool that queries `nvidia-smi` to find unused GPUs by checking memory usage. GPUs with memory usage below 300MB are considered free.

## Usage

```bash
python main.py          # Print indexes of free GPUs
python main.py -v       # Verbose output with details
python main.py -t 500   # Use 500MB threshold
python main.py -q       # Quiet mode
```
