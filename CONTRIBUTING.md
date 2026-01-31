# Contributing to SADL

Before submitting changes, read this guideline and make sure you follow all rules.
Yes, I know the code quality checks can be annoying and might be overly strict, but this ensures the code remains readable.

## Setup

```bash
# Clone the repository
git clone https://github.com/timcares/sadl.git
cd sadl

# Install dev dependencies (requires uv)
uv sync --extra dev

# Or with GPU support (requires CUDA)
uv sync --extra dev --extra gpu

# Install pre-commit hooks (recommended)
make bootstrap
```

## Commit Message Guidelines

This project uses [Conventional Commits](https://www.conventionalcommits.org/) for commit messages. This enables automatic versioning and changelog generation.

### Format

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

### Types

| Type       | Description                                      | Version Bump |
|------------|--------------------------------------------------|--------------|
| `feat`     | A new feature                                    | Minor        |
| `fix`      | A bug fix                                        | Patch        |
| `perf`     | Performance improvement                          | Patch        |
| `docs`     | Documentation only changes                       | None         |
| `style`    | Code style (formatting, semicolons, etc.)        | None         |
| `refactor` | Code change that neither fixes a bug nor adds a feature | None  |
| `test`     | Adding or correcting tests                       | None         |
| `chore`    | Maintenance tasks (deps, build, etc.)            | None         |
| `ci`       | CI/CD configuration changes                      | None         |

### Examples

```bash
# Feature (bumps minor version: 1.0.0 -> 1.1.0)
feat(tensor): add support for complex numbers

# Bug fix (bumps patch version: 1.0.0 -> 1.0.1)
fix(grad_ops): correct gradient computation for matmul with broadcasting

# Breaking change (bumps major version: 1.0.0 -> 2.0.0)
feat(api)!: rename Tensor.backward to Tensor.compute_gradients

BREAKING CHANGE: The backward method has been renamed.

# Documentation (no version bump)
docs: update installation instructions

# Performance improvement (bumps patch version)
perf(optimizer): reduce memory allocation in SGD step
```

### Using Commitizen

For interactive commit message creation:

```bash
make commit
# or
uv run cz commit
```

This guides you through creating a properly formatted commit message.

## Development Commands

```bash
make help          # Show all available commands
make check         # Run all checks (lint, typecheck, test)
make format        # Format code
make lint          # Run linter (ruff)
make typecheck     # Run type checker (mypy)
make test          # Run tests (pytest)
make fix           # Auto-fix linting issues and format
make commit        # Create a conventional commit (interactive)
```

## Code Quality

```bash
make check
```

This runs:
1. **Linting** (ruff) -> Code style and potential errors
2. **Type checking** (mypy) -> Static type analysis
3. **Tests** (pytest) -> Unit and integration tests

## Version Management

This project uses [Python Semantic Release](https://python-semantic-release.readthedocs.io/) for automated versioning and [Commitizen](https://commitizen-tools.github.io/commitizen/) for commit message validation.

### Commands

```bash
make version           # Show current version
make bump              # Preview next version bump (dry-run)
make bump-apply        # Apply version bump (creates commit and tag)
make changelog         # Generate/update CHANGELOG.md
make changelog-preview # Preview changelog changes
make release-dry       # Preview full release process
make release           # Full release (requires GH_TOKEN)
```

### Manual Version Bumps

```bash
make bump-patch        # Bump patch version (x.x.X)
make bump-minor        # Bump minor version (x.X.0)
make bump-major        # Bump major version (X.0.0)
```

### CI/CD Integration

For GitHub Actions, set up the `GH_TOKEN` secret with a personal access token that has `contents` write permission. The release workflow will:

1. Analyze commits since the last release
2. Determine the appropriate version bump
3. Update the version in `pyproject.toml`
4. Generate/update the changelog
5. Create a git tag
6. Optionally create a GitHub release

## Building and Publishing

### Build Package

```bash
# Build package (runs make check first)
make build

# Build without checks (for quick iteration)
make build-only
```

Distribution files are created in `dist/`.

### Test Locally

```bash
# Install from local build
uv pip install dist/sadl-*.whl

# Or install in editable mode
uv pip install -e .
```

### Publish

```bash
# Test on TestPyPI first
make publish-test

# Publish to PyPI
make publish
```

Both commands run `make check` and `make build` first to ensure code quality.

## Project Structure

```
sadl/
├── __init__.py         # Public API
├── pyproject.toml      # Package configuration
├── README.md           # Documentation
├── CONTRIBUTING.md     # This file
├── Makefile            # Development commands
├── docs/
│   └── API_REFERENCE.md
└── src/
    ├── __init__.py     # Internal exports
    ├── backend.py      # NumPy/CuPy abstraction
    ├── tensor.py       # Tensor, Parameter, serialization
    ├── grad_ops.py     # Gradient operation registry
    ├── function.py     # Neural network layers
    ├── optimizer.py    # Optimizer, SGD, backpropagation
    └── utils.py        # Device transfer utilities
```

## Adding New Features

### New Gradient Operation

1. Add the backward function in `src/grad_ops.py`:

```python
@register_grad_op
def my_op_backward(*inputs, compute_grad, grad_out, **kwargs):
    x = inputs[0]
    grad_x = ... if compute_grad[0] else None
    return (grad_x,)
```

2. The operation will automatically be available for tensors.

### New Layer

1. Create a class in `src/function.py` that inherits from `Function`:

```python
class MyLayer(Function):
    def __init__(self, ...):
        self.param = Parameter(...)

    def __call__(self, x: Tensor) -> Tensor:
        return ...
```

2. Export it in `src/__init__.py` and `__init__.py`.

### New Optimizer

1. Create a class in `src/optimizer.py` that inherits from `Optimizer`:

```python
class MyOptimizer(Optimizer):
    @no_grad_fn
    def step(self) -> None:
        for param in self.params:
            param -= self.lr * param.grad
```

Optionally, if you need extra states, you should override `Optimizer.__init__` in you optimizer.

2. Export it in `src/__init__.py` and `__init__.py`.
