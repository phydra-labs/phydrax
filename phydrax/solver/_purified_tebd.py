#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..tensor_network import LocallyPurifiedDensity, NearestNeighborHamiltonian
from ..tensor_network._canonical import canonicalize_lpdo
from ._purified_lindblad import apply_local_kraus_channel, LocalKrausChannel


class LPDOBondEvidence(StrictModule):
    retained_rank: int
    available_rank: int
    discarded_weight: Array
    valid: Array

    def __init__(
        self, retained_rank: int, available_rank: int, discarded_weight: ArrayLike, /
    ):
        self.retained_rank = int(retained_rank)
        self.available_rank = int(available_rank)
        self.discarded_weight = jnp.asarray(discarded_weight)
        self.valid = jnp.isfinite(self.discarded_weight) & (self.discarded_weight >= 0.0)


def apply_lpdo_two_site_unitary(
    state: LocallyPurifiedDensity,
    bond: int,
    gate: ArrayLike,
    /,
    *,
    maximum_bond_dimension: int,
) -> tuple[LocallyPurifiedDensity, LPDOBondEvidence]:
    index = int(bond)
    left = state.tensors[index]
    right = state.tensors[index + 1]
    gate_ = jnp.asarray(gate)
    if gate_.shape != (left.shape[1], right.shape[1], left.shape[1], right.shape[1]):
        raise ValueError("LPDO unitary gate shape is invalid.")
    theta = jnp.tensordot(left, right, axes=(-1, 0))
    theta = jnp.einsum("abij,likjmr->lakbmr", gate_, theta)
    matrix = theta.reshape(
        (
            left.shape[0] * left.shape[1] * left.shape[2],
            right.shape[1] * right.shape[2] * right.shape[-1],
        )
    )
    u, singular_values, vh = jnp.linalg.svd(matrix, full_matrices=False)
    available = singular_values.shape[0]
    retained = min(int(maximum_bond_dimension), available)
    discarded = jnp.sum(singular_values[retained:] ** 2)
    new_left = u[:, :retained].reshape(
        (left.shape[0], left.shape[1], left.shape[2], retained)
    )
    new_right = (singular_values[:retained, None] * vh[:retained]).reshape(
        (retained, right.shape[1], right.shape[2], right.shape[-1])
    )
    tensors = list(state.tensors)
    tensors[index] = new_left
    tensors[index + 1] = new_right
    result = LocallyPurifiedDensity(tuple(tensors))
    return result, LPDOBondEvidence(retained, available, discarded)


class PurifiedStrangProblem(StrictModule):
    initial_state: LocallyPurifiedDensity
    hamiltonian: NearestNeighborHamiltonian
    half_step_channels: tuple[LocalKrausChannel, ...]
    problem_id: str

    def __init__(
        self,
        initial_state: LocallyPurifiedDensity,
        hamiltonian: NearestNeighborHamiltonian,
        half_step_channels: Sequence[LocalKrausChannel],
        /,
        *,
        problem_id: str = "purified-strang",
    ):
        if tuple(initial_state.physical_dimensions) != hamiltonian.physical_dimensions:
            raise ValueError("LPDO and Hamiltonian dimensions differ.")
        channels = tuple(half_step_channels)
        if not channels:
            raise ValueError("At least one dissipative channel is required.")
        self.initial_state = initial_state
        self.hamiltonian = hamiltonian
        self.half_step_channels = channels
        self.problem_id = str(problem_id)


class PurifiedStrangResult(StrictModule):
    final_state: LocallyPurifiedDensity
    raw_trace_history: Array
    bond_discarded_history: Array
    kraus_discarded_history: Array
    canonical_residual_history: Array
    valid: Array
    problem_id: str

    def __init__(
        self,
        final_state: LocallyPurifiedDensity,
        raw_trace_history: ArrayLike,
        bond_discarded_history: ArrayLike,
        kraus_discarded_history: ArrayLike,
        canonical_residual_history: ArrayLike,
        /,
        *,
        problem_id: str,
    ):
        self.final_state = final_state
        self.raw_trace_history = jnp.asarray(raw_trace_history)
        self.bond_discarded_history = jnp.asarray(bond_discarded_history)
        self.kraus_discarded_history = jnp.asarray(kraus_discarded_history)
        self.canonical_residual_history = jnp.asarray(canonical_residual_history)
        self.valid = (
            jnp.all(jnp.isfinite(self.raw_trace_history))
            & jnp.all(self.bond_discarded_history >= 0.0)
            & jnp.all(self.kraus_discarded_history >= 0.0)
        )
        self.problem_id = str(problem_id)


def solve_purified_strang(
    problem: PurifiedStrangProblem,
    /,
    *,
    step_size: ArrayLike,
    steps: int,
    maximum_bond_dimension: int,
    maximum_purification_dimension: int,
) -> PurifiedStrangResult:
    state = problem.initial_state
    step = jnp.asarray(step_size, dtype=float).reshape(())
    traces = [state.raw_trace()]
    bond_records = []
    kraus_records = []
    canonical_records = []
    for _ in range(int(steps)):
        for channel in problem.half_step_channels:
            state, evidence = apply_local_kraus_channel(
                state,
                channel,
                maximum_purification_dimension=maximum_purification_dimension,
            )
            kraus_records.append(evidence.discarded_weight)
        for bond in range(0, state.site_count - 1, 2):
            state, evidence = apply_lpdo_two_site_unitary(
                state,
                bond,
                problem.hamiltonian.gate(bond, 0.5 * step),
                maximum_bond_dimension=maximum_bond_dimension,
            )
            bond_records.append(evidence.discarded_weight)
        for bond in range(1, state.site_count - 1, 2):
            state, evidence = apply_lpdo_two_site_unitary(
                state,
                bond,
                problem.hamiltonian.gate(bond, step),
                maximum_bond_dimension=maximum_bond_dimension,
            )
            bond_records.append(evidence.discarded_weight)
        for bond in range(0, state.site_count - 1, 2):
            state, evidence = apply_lpdo_two_site_unitary(
                state,
                bond,
                problem.hamiltonian.gate(bond, 0.5 * step),
                maximum_bond_dimension=maximum_bond_dimension,
            )
            bond_records.append(evidence.discarded_weight)
        for channel in problem.half_step_channels:
            state, evidence = apply_local_kraus_channel(
                state,
                channel,
                maximum_purification_dimension=maximum_purification_dimension,
            )
            kraus_records.append(evidence.discarded_weight)
        state, canonical = canonicalize_lpdo(state, center=state.site_count // 2)
        canonical_records.append(
            jnp.maximum(
                jnp.max(canonical.left_residuals), jnp.max(canonical.right_residuals)
            )
        )
        traces.append(state.raw_trace())
    return PurifiedStrangResult(
        state,
        jnp.stack(traces),
        jnp.stack(bond_records) if bond_records else jnp.zeros((0,)),
        jnp.stack(kraus_records) if kraus_records else jnp.zeros((0,)),
        jnp.stack(canonical_records) if canonical_records else jnp.zeros((0,)),
        problem_id=problem.problem_id,
    )


__all__ = [
    "LPDOBondEvidence",
    "PurifiedStrangProblem",
    "PurifiedStrangResult",
    "apply_lpdo_two_site_unitary",
    "solve_purified_strang",
]
