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

from typing import TYPE_CHECKING

from .._deferred_exports import deferred_exports


# Re-exports load on first use, so importing one owner never imports the
# whole package and its import cycles.
_DEFERRED_EXPORTS = {
    "PeriodicInputCertificate": ".._model",
    "activations": "module:.activations",
    "atomistic": "module:.atomistic",
    "flows": "module:.flows",
    "latent": "module:.latent",
    "layers": "module:.layers",
    "models": "module:.models",
    "neural_tangent": "module:.neural_tangent",
    "operator": "module:.operator",
    "parameters": "module:.parameters",
    "population": "module:.population",
    "quantum": "module:.quantum",
}


__getattr__, __dir__ = deferred_exports(__name__, globals(), _DEFERRED_EXPORTS)


if TYPE_CHECKING:
    from .._model import PeriodicInputCertificate
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
