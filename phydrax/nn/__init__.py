"""Composable neural models, tensor layers, and neural-operator runtimes.

Ownership is explicit:

- :mod:`phydrax.nn.models` contains pointwise and structured finite-dimensional models.
- :mod:`phydrax.nn.layers` contains reusable tensor-to-tensor layers.
- :mod:`phydrax.nn.operator` contains operator data, engines, adapters, and runtime policy.
"""

from . import activations, layers, models, operator


__all__ = ["activations", "layers", "models", "operator"]
