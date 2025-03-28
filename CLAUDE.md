# CLAUDE.md - Development Guidelines

## Setup & Commands
- **Environment with uv**: `uv venv && source .venv/bin/activate`
- **Install with uv**: `uv pip install -e .`
- **Traditional Setup**: `python -m venv venv && source venv/bin/activate && pip install -r requirements.txt`
- **Run NumPy Version**: `python base_nn_layers.py`
- **Run JAX Version**: `python jax_nn_layers.py`
- **Lint**: `ruff check .`
- **Format**: `ruff format .`

## Code Style
- **Imports**: Standard library first, then external packages, then local modules
- **Types**: Use type hints for function parameters and return values
- **Naming**: snake_case for variables/functions, CamelCase for classes
- **Documentation**: Docstrings for classes and non-trivial functions
- **Error Handling**: Use assertions for validation, exceptions for runtime errors
- **Structure**: Organize code with classes inheriting from proper base classes
- **Constants**: UPPER_CASE for constants and config values

## Best Practices
- Keep numerical computations vectorized using NumPy/JAX
- Use appropriate activation functions based on layer purpose
- Validate input/output shapes with assertions
- Use proper initialization for weights to avoid vanishing/exploding gradients
- For JAX implementation, respect immutability of arrays