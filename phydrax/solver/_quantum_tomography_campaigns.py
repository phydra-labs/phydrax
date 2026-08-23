#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import ArrayLike

from ..uq import QuantumTomographyData, tetrahedral_qubit_povm
from ._quantum_tomography import QuantumTomographyProblem


def tetrahedral_qubit_tomography(
    true_density: ArrayLike,
    /,
    *,
    shots: int = 1000,
) -> QuantumTomographyProblem:
    """Create a deterministic expected-count qubit tomography campaign."""
    if int(shots) < 1:
        raise ValueError("shots must be positive.")
    povm = tetrahedral_qubit_povm()
    probabilities = povm.probabilities(true_density)
    data = QuantumTomographyData(
        int(shots) * probabilities,
        data_id=f"tetrahedral-expected-counts:{int(shots)}",
    )
    return QuantumTomographyProblem(
        povm,
        data,
        0.5 * jnp.eye(2, dtype=complex),
        problem_id="tetrahedral-qubit-tomography",
    )


__all__ = ["tetrahedral_qubit_tomography"]
