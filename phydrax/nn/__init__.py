"""Composable neural models, tensor layers, and neural-operator runtimes.

Ownership is explicit:

- :mod:`phydrax.nn.atomistic` contains finite-molecule equivariant potentials.
- :mod:`phydrax.nn.models` contains pointwise and structured finite-dimensional models.
- :mod:`phydrax.nn.layers` contains reusable tensor-to-tensor layers.
- :mod:`phydrax.nn.parameters` contains physical transforms, subspaces, selections, and adaptations.
- :mod:`phydrax.nn.quantum` contains antisymmetric continuum-electron amplitudes.
- :mod:`phydrax.nn.operator` contains operator data, engines, adapters, and runtime policy.
"""

from . import (
    activations,
    atomistic,
    latent,
    layers,
    models,
    operator,
    parameters,
    quantum,
)


__all__ = [
    "activations",
    "atomistic",
    "latent",
    "layers",
    "models",
    "operator",
    "parameters",
    "quantum",
]
