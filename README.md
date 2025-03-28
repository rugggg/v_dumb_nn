# v_dumb_nn

A basic neural network implemented in pure NumPy and JAX.

## Overview

This project contains implementations of a simple neural network for MNIST classification:
- `base_nn_layers.py`: Implementation using NumPy
- `jax_nn_layers.py`: Implementation using JAX

## Setup

### Using uv (recommended)

```bash
# Install uv if you haven't already
pip install uv

# Create and activate virtual environment
uv venv

# Activate the environment (Unix/macOS)
source .venv/bin/activate
# Or on Windows
# .venv\Scripts\activate

# Install dependencies
uv pip install -e .
```

### Using traditional pip

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Unix/macOS
# venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

Run the NumPy implementation:
```bash
python base_nn_layers.py
```

Run the JAX implementation:
```bash
python jax_nn_layers.py
```

## Development

Lint code with ruff:
```bash
ruff check .
```

Format code:
```bash
ruff format .
```