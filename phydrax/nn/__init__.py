"""Composable neural models, tensor layers, and neural-operator runtimes.

Ownership is explicit:

- :mod:`phydrax.nn.atomistic` contains finite-molecule equivariant potentials.
- :mod:`phydrax.nn.models` contains pointwise and structured finite-dimensional models.
- :mod:`phydrax.nn.layers` contains reusable tensor-to-tensor layers.
- :mod:`phydrax.nn.parameters` contains physical transforms, subspaces, selections, and adaptations.
- :mod:`phydrax.nn.population` constructs physical population codes and diagnosed decoders.
- :mod:`phydrax.nn.quantum` contains antisymmetric continuum-electron amplitudes.
- :mod:`phydrax.nn.operator` contains operator data, engines, adapters, and runtime policy.

`PeriodicInputCertificate` is the construction certificate by which models
such as certified periodic Fourier embeddings declare exact input periodicity.
"""

from importlib import import_module
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from . import (
        activations,
        atomistic,
        flows,
        latent,
        layers,
        models,
        neural_tangent,
        operator,
        parameters,
        population,
        quantum,
    )

from .._model import PeriodicInputCertificate


_MODULE_EXPORTS = frozenset(
    {
        "activations",
        "atomistic",
        "flows",
        "latent",
        "layers",
        "models",
        "neural_tangent",
        "operator",
        "parameters",
        "population",
        "quantum",
    }
)


def __getattr__(name: str) -> object:
    if name not in _MODULE_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = import_module(f".{name}", __name__)
    globals()[name] = value
    return value


__all__ = [
    "activations",
    "atomistic",
    "flows",
    "latent",
    "layers",
    "neural_tangent",
    "models",
    "operator",
    "parameters",
    "PeriodicInputCertificate",
    "population",
    "quantum",
]
