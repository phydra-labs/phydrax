#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ._canonical import canonicalize_mps
from ._core import MatrixProductState
from ._evolution import apply_two_site_gate, TensorTruncationEvidence


class NearestNeighborHamiltonian(StrictModule):
    terms: tuple[Array, ...]
    physical_dimensions: tuple[int, ...] = eqx.field(static=True)
    hamiltonian_id: str = eqx.field(static=True)

    def __init__(
        self,
        terms: Sequence[ArrayLike],
        physical_dimensions: Sequence[int],
        /,
        *,
        hamiltonian_id: str,
    ):
        dimensions = tuple(int(value) for value in physical_dimensions)
        values = tuple(jnp.asarray(term) for term in terms)
        if len(values) != len(dimensions) - 1:
            raise ValueError("One two-site term is required for every neighboring bond.")
        for index, term in enumerate(values):
            size = dimensions[index] * dimensions[index + 1]
            if term.shape != (size, size):
                raise ValueError("Two-site Hamiltonian term has the wrong shape.")
        self.terms = values
        self.physical_dimensions = dimensions
        self.hamiltonian_id = str(hamiltonian_id)

    def gate(self, bond: int, step: ArrayLike, /) -> Array:
        index = int(bond)
        left = self.physical_dimensions[index]
        right = self.physical_dimensions[index + 1]
        return jsp.linalg.expm(-1j * jnp.asarray(step) * self.terms[index]).reshape(
            (left, right, left, right)
        )


class TEBDEvidence(StrictModule):
    discarded_weights: Array
    cumulative_discarded_weight: Array
    norm_residual: Array
    valid: Array
    trotter_order: int = eqx.field(static=True)

    def __init__(
        self,
        discarded_weights: ArrayLike,
        norm_residual: ArrayLike,
        /,
        *,
        trotter_order: int,
    ):
        self.discarded_weights = jnp.asarray(discarded_weights)
        self.cumulative_discarded_weight = jnp.sum(self.discarded_weights)
        self.norm_residual = jnp.asarray(norm_residual)
        self.valid = (
            jnp.all(jnp.isfinite(self.discarded_weights))
            & (self.cumulative_discarded_weight >= 0.0)
            & jnp.isfinite(self.norm_residual)
        )
        self.trotter_order = int(trotter_order)


def _layer(
    state: MatrixProductState,
    hamiltonian: NearestNeighborHamiltonian,
    bonds: range,
    step: Array,
    maximum_bond_dimension: int,
    normalize: bool,
):
    current = state
    evidence: list[TensorTruncationEvidence] = []
    for bond in bonds:
        current, local = apply_two_site_gate(
            current,
            bond,
            hamiltonian.gate(bond, step),
            maximum_bond_dimension=maximum_bond_dimension,
            normalize=normalize,
        )
        evidence.append(local)
    return current, evidence


def tebd_step(
    state: MatrixProductState,
    hamiltonian: NearestNeighborHamiltonian,
    step_size: ArrayLike,
    /,
    *,
    maximum_bond_dimension: int,
    order: int = 2,
    normalize: bool = True,
) -> tuple[MatrixProductState, TEBDEvidence]:
    if tuple(state.physical_dimensions) != hamiltonian.physical_dimensions:
        raise ValueError("MPS and Hamiltonian physical dimensions differ.")
    step = jnp.asarray(step_size, dtype=float).reshape(())
    if order not in (1, 2):
        raise ValueError("TEBD order must be one or two.")
    records: list[TensorTruncationEvidence] = []
    current = state
    if order == 1:
        current, local = _layer(
            current,
            hamiltonian,
            range(0, state.site_count - 1, 2),
            step,
            maximum_bond_dimension,
            normalize,
        )
        records.extend(local)
        current, local = _layer(
            current,
            hamiltonian,
            range(1, state.site_count - 1, 2),
            step,
            maximum_bond_dimension,
            normalize,
        )
        records.extend(local)
    else:
        current, local = _layer(
            current,
            hamiltonian,
            range(0, state.site_count - 1, 2),
            0.5 * step,
            maximum_bond_dimension,
            normalize,
        )
        records.extend(local)
        current, local = _layer(
            current,
            hamiltonian,
            range(1, state.site_count - 1, 2),
            step,
            maximum_bond_dimension,
            normalize,
        )
        records.extend(local)
        current, local = _layer(
            current,
            hamiltonian,
            range(0, state.site_count - 1, 2),
            0.5 * step,
            maximum_bond_dimension,
            normalize,
        )
        records.extend(local)
    current, _ = canonicalize_mps(
        current, center=state.site_count // 2, normalize=normalize
    )
    discarded = jnp.stack([record.discarded_weight for record in records])
    return current, TEBDEvidence(
        discarded,
        jnp.abs(current.norm() - 1.0),
        trotter_order=order,
    )


__all__ = [
    "NearestNeighborHamiltonian",
    "TEBDEvidence",
    "tebd_step",
]
