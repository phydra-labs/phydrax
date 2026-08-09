"""Composable neural models, tensor layers, and neural-operator runtimes.

Ownership is explicit:

- :mod:`phydrax.nn.models` contains pointwise and structured finite-dimensional models.
- :mod:`phydrax.nn.layers` contains reusable tensor-to-tensor layers.
- :mod:`phydrax.nn.parameters` contains physical parameter transforms and selections.
- :mod:`phydrax.nn.operator` contains operator data, engines, adapters, and runtime policy.
"""

from . import activations, layers, models, operator, parameters


__all__ = ["activations", "layers", "models", "operator", "parameters"]
