#!/usr/bin/env python3
"""Verify that sadl is correctly installed and functional.

This script is run by test_install.sh in an isolated environment
to verify the built wheel works correctly.
"""

import sadl


def test_version() -> None:
    """Verify version is accessible."""
    print(f"sadl version: {sadl.__version__}")
    assert sadl.__version__, "Version should not be empty"


def test_tensor_operations() -> None:
    """Test basic tensor creation and operations."""
    x = sadl.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = x * 2
    loss = y.sum()
    assert float(loss) == 12.0, f"Expected 12.0, got {float(loss)}"


def test_model_forward() -> None:
    """Test basic model creation and forward pass."""
    model = sadl.Mlp(
        [
            sadl.Linear(dim_in=3, dim_out=4),
            sadl.ReLU(),
            sadl.Linear(dim_in=4, dim_out=1),
        ]
    )

    x = sadl.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    out = model(x)

    assert out.shape == (1, 1), f"Expected (1, 1), got {out.shape}"


def main() -> None:
    """Run all verification tests."""
    print("Running installation verification tests...")
    print("-" * 40)

    test_version()
    test_tensor_operations()
    test_model_forward()

    print("-" * 40)
    print("All installation tests passed!")


if __name__ == "__main__":
    main()
