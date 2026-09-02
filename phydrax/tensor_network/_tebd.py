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
from ._evolution import apply_two_site_gate
from ._split import TensorTruncationEvidence


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
        identifier = str(hamiltonian_id)
        if not dimensions or any(value < 1 for value in dimensions):
            raise ValueError("Hamiltonian physical dimensions must be positive.")
        if len(values) != len(dimensions) - 1:
            raise ValueError("One two-site term is required for every neighboring bond.")
        for index, term in enumerate(values):
            size = dimensions[index] * dimensions[index + 1]
            if term.shape != (size, size):
                raise ValueError("Two-site Hamiltonian term has the wrong shape.")
            if not bool(
                jnp.all(jnp.isfinite(term))
                & jnp.allclose(term, jnp.conj(term.T), rtol=1e-10, atol=1e-12)
            ):
                raise ValueError(
                    "Two-site Hamiltonian terms must be finite and Hermitian."
                )
        if not identifier:
            raise ValueError("hamiltonian_id must be non-empty.")
        self.terms = values
        self.physical_dimensions = dimensions
        self.hamiltonian_id = identifier

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
    precision_policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        discarded_weights: ArrayLike,
        norm_residual: ArrayLike,
        /,
        *,
        trotter_order: int,
        precision_policy_id: str,
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
        self.precision_policy_id = str(precision_policy_id)


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
    step = jnp.asarray(step_size, dtype=state.tensors[0].real.dtype).reshape(())
    if order not in (1, 2):
        raise ValueError("TEBD order must be one or two.")
    if int(maximum_bond_dimension) < 1:
        raise ValueError("Maximum bond dimension must be positive.")
    step = eqx.error_if(
        step,
        (~jnp.isfinite(step)) | (step < 0.0),
        "TEBD step size must be finite and nonnegative.",
    )
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
    discarded = (
        jnp.stack([record.discarded_weight for record in records])
        if records
        else jnp.zeros((0,), dtype=step.dtype)
    )
    return current, TEBDEvidence(
        discarded,
        jnp.abs(current.norm() - 1.0),
        trotter_order=order,
        precision_policy_id=state.precision.policy_id,
    )


__all__ = [
    "NearestNeighborHamiltonian",
    "TEBDEvidence",
    "tebd_step",
]
