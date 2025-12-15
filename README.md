# find-free-gpu

| Tests | Coverage |
|--------|----------|
| [![Tests](https://github.com/ShawonAshraf/find-free-gpu/actions/workflows/test.yml/badge.svg)](https://github.com/ShawonAshraf/find-free-gpu/actions/workflows/test.yml) | [![codecov](https://codecov.io/gh/ShawonAshraf/find-free-gpu/graph/badge.svg?token=dvkfy4iEdd)](https://codecov.io/gh/ShawonAshraf/find-free-gpu) |


A CLI tool that queries `nvidia-smi` to find unused GPUs by checking memory usage. GPUs with memory usage below 300MB are considered free.

## Usage

```bash
python main.py          # Print indexes of free GPUs
python main.py -v       # Verbose output with details
python main.py -t 500   # Use 500MB threshold
python main.py -q       # Quiet mode
```

## Dev

```bash
# install uv from https://docs.astral.sh/uv/getting-started/installation/
uv sync
uv run main.py

# for testing with coverage
uv run pytest tests/ -v --cov --cov-report=xml
# without coverage
uv run pytest
```
