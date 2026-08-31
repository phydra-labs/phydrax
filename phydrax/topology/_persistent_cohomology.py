#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._coefficients import PrimeField
from ._filtration import CellFiltration
from ._homology import compute_homology, FiniteFieldBasis
from ._persistence import compute_persistence, PersistenceResult
from ._resources import TopologyResourcePolicy


class TerminalCocycleAnnotation(StrictModule, NonTrainableState):
    """Terminal cohomology basis and compatible essential interval indices."""

    basis: FiniteFieldBasis
    essential_pair_indices: Array

    def __init__(
        self,
        basis: FiniteFieldBasis,
        essential_pair_indices,
        /,
    ):
        indices = jnp.asarray(essential_pair_indices, dtype=jnp.int32)
        if indices.shape != (basis.generator_count,):
            raise ValueError(
                "Terminal cocycle annotations must align with basis generators."
            )
        self.basis = basis
        self.essential_pair_indices = indices


class PersistentCohomologyResult(StrictModule, NonTrainableState):
    """Exact persistence intervals with terminal cocycle representatives."""

    persistence: PersistenceResult
    terminal_cocycles: tuple[FiniteFieldBasis, ...]
    annotations: tuple[TerminalCocycleAnnotation, ...]
    field: PrimeField
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        persistence: PersistenceResult,
        terminal_cocycles: tuple[FiniteFieldBasis, ...],
        annotations: tuple[TerminalCocycleAnnotation, ...],
        field: PrimeField,
        /,
    ):
        self.persistence = persistence
        self.terminal_cocycles = terminal_cocycles
        self.annotations = annotations
        self.field = field
        self.result_id = canonical_fingerprint(
            {
                "kind": "persistent-cohomology-result",
                "persistence": persistence.result_id,
                "field": field.field_id,
                "terminal_cocycles": [value.basis_id for value in terminal_cocycles],
                "annotations": [
                    {
                        "basis": value.basis.basis_id,
                        "pairs": np.asarray(value.essential_pair_indices).tolist(),
                    }
                    for value in annotations
                ],
            }
        )


def compute_persistent_cohomology(
    filtration: CellFiltration,
    /,
    *,
    coefficients: PrimeField,
    max_degree: int | None = None,
    resources: TopologyResourcePolicy | None = None,
) -> PersistentCohomologyResult:
    """Compute field-equivalent intervals and exact cocycles at the terminal complex."""
    maximum = filtration.max_degree if max_degree is None else int(max_degree)
    persistence = compute_persistence(
        filtration,
        coefficients=coefficients,
        max_degree=maximum,
        resources=resources,
    )
    terminal = compute_homology(
        filtration.complex,
        coefficients=coefficients,
        degrees=tuple(range(maximum + 1)),
        representatives="cocycles",
        resources=resources,
    )
    cocycles = tuple(
        value.cocycles for value in terminal.degrees if value.cocycles is not None
    )
    degrees = np.asarray(persistence.pairing.degrees)
    finite = np.asarray(persistence.pairing.has_finite_death)
    annotations = []
    for basis in cocycles:
        candidates = np.flatnonzero((degrees == basis.degree) & ~finite).astype(np.int32)
        if candidates.size != basis.generator_count:
            raise RuntimeError(
                "Terminal cocycle dimension does not match essential intervals."
            )
        annotations.append(TerminalCocycleAnnotation(basis, candidates))
    return PersistentCohomologyResult(
        persistence,
        cocycles,
        tuple(annotations),
        coefficients,
    )


__all__ = [
    "PersistentCohomologyResult",
    "TerminalCocycleAnnotation",
    "compute_persistent_cohomology",
]
