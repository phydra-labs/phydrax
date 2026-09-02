#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import prod

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

import phydrax.ein as ein

from .._fingerprint import canonical_fingerprint
from .._precision import precision_itemsize
from .._strict import StrictModule
from ..linalg import DenseLinearOperator, FactorizationPolicy, factorize
from ._peps import PEPS


class BoundaryMPSPolicy(StrictModule):
    maximum_bond_dimension: int = eqx.field(static=True)
    singular_value_cutoff: float = eqx.field(static=True)
    maximum_tensor_elements: int = eqx.field(static=True)
    maximum_workspace_bytes: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        maximum_bond_dimension: int,
        /,
        *,
        singular_value_cutoff: float = 0.0,
        maximum_tensor_elements: int = 100_000_000,
        maximum_workspace_bytes: int = 2**31,
    ):
        bond = int(maximum_bond_dimension)
        cutoff = float(singular_value_cutoff)
        tensor_elements = int(maximum_tensor_elements)
        workspace = int(maximum_workspace_bytes)
        if bond < 1 or cutoff < 0.0 or tensor_elements < 1 or workspace < 1:
            raise ValueError(
                "Boundary-MPS policy values are outside their finite positive ranges."
            )
        self.maximum_bond_dimension = bond
        self.singular_value_cutoff = cutoff
        self.maximum_tensor_elements = tensor_elements
        self.maximum_workspace_bytes = workspace
        self.policy_id = canonical_fingerprint(
            {
                "kind": "boundary-mps-policy",
                "maximum_bond_dimension": bond,
                "singular_value_cutoff": cutoff,
                "maximum_tensor_elements": tensor_elements,
                "maximum_workspace_bytes": workspace,
            }
        )


class BoundaryMPSEvidence(StrictModule):
    state_id: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)
    replay_id: str = eqx.field(static=True)
    row_count: int = eqx.field(static=True)
    truncation_errors: Array
    retained_ranks: Array
    gauge_residuals: Array
    logarithmic_scale: Array
    finite: Array
    exact: Array
    accepted: Array
    claim: str = eqx.field(static=True)
    global_error_bound_claimed: bool = eqx.field(static=True)
    admitted_peak_elements: int = eqx.field(static=True)
    admitted_peak_bytes: int = eqx.field(static=True)


class BoundaryMPSResult(StrictModule):
    value: Array
    boundary: tuple[Array, ...]
    evidence: BoundaryMPSEvidence


def _double_tensor(tensor: Array, /) -> Array:
    shape = tuple(int(value * value) for value in tensor.shape[:4])
    return ein.contract(
        "urdlp,URDLp->uUrRdDlL",
        jnp.conj(tensor),
        tensor,
        optimize="greedy",
    ).reshape(shape)


def _admit_boundary(state: PEPS, policy: BoundaryMPSPolicy, /) -> tuple[int, int]:
    itemsize = precision_itemsize(str(state.tensors[0].dtype))
    maximum = 1
    total_transfers = 0
    for tensor in state.tensors:
        doubled = tuple(int(value * value) for value in tensor.shape[:4])
        transfer_elements = prod(doubled)
        total_transfers += transfer_elements
        combined = (
            policy.maximum_bond_dimension
            * doubled[3]
            * doubled[2]
            * policy.maximum_bond_dimension
            * doubled[1]
        )
        maximum = max(maximum, transfer_elements, combined)
    if total_transfers > policy.maximum_tensor_elements:
        raise MemoryError("Boundary-MPS transfer tensors exceed maximum_tensor_elements.")
    if maximum > policy.maximum_tensor_elements:
        raise MemoryError("Boundary-MPS intermediate exceeds maximum_tensor_elements.")
    peak_bytes = (total_transfers + maximum) * itemsize
    if peak_bytes > policy.maximum_workspace_bytes:
        raise MemoryError("Boundary-MPS contraction exceeds maximum_workspace_bytes.")
    return maximum, peak_bytes


def _compress_boundary(
    tensors: tuple[Array, ...], policy: BoundaryMPSPolicy, /
) -> tuple[tuple[Array, ...], tuple[Array, ...], tuple[Array, ...], tuple[Array, ...]]:
    values = list(tensors)
    errors = []
    ranks = []
    gauges = []
    for column in range(len(values) - 1):
        left = values[column]
        matrix = left.reshape((left.shape[0] * left.shape[1], left.shape[2]))
        decomposition = factorize(DenseLinearOperator(matrix), FactorizationPolicy("svd"))
        svd = decomposition.prepared_solve.state
        u, singular, vh = svd.u, svd.singular_values, svd.vh
        retained = min(policy.maximum_bond_dimension, singular.shape[0])
        kept_singular = singular[:retained]
        keep = kept_singular > policy.singular_value_cutoff
        discarded = jnp.sum(jnp.square(jnp.abs(singular[retained:])))
        discarded = discarded + jnp.sum(
            jnp.where(keep, 0.0, jnp.square(jnp.abs(kept_singular)))
        )
        kept_u = u[:, :retained]
        kept_vh = vh[:retained, :]
        identity = jnp.eye(retained, dtype=kept_u.dtype)
        gauge = jnp.max(jnp.abs(jnp.conj(kept_u.T) @ kept_u - identity))
        values[column] = kept_u.reshape((left.shape[0], left.shape[1], retained))
        transfer = jnp.where(keep, kept_singular, 0.0)[:, None] * kept_vh
        values[column + 1] = ein.contract(
            "ka,adb->kdb", transfer, values[column + 1], optimize=False
        )
        errors.append(discarded)
        ranks.append(jnp.sum(keep, dtype=jnp.int32))
        gauges.append(gauge)
    return tuple(values), tuple(errors), tuple(ranks), tuple(gauges)


def contract_peps_boundary_mps(
    state: PEPS,
    policy: BoundaryMPSPolicy,
    /,
) -> BoundaryMPSResult:
    """Contract PEPS norm row-by-row with explicit SVD boundary truncation."""

    if not isinstance(state, PEPS) or not isinstance(policy, BoundaryMPSPolicy):
        raise TypeError("state and policy have invalid types.")
    peak_elements, peak_bytes = _admit_boundary(state, policy)
    transfers = tuple(_double_tensor(tensor) for tensor in state.tensors)
    boundary = tuple(
        jnp.ones((1, transfers[column].shape[0], 1), dtype=transfers[column].dtype)
        for column in range(state.columns)
    )
    truncation_errors = []
    retained_ranks = []
    gauge_residuals = []
    logarithmic_scale = jnp.asarray(0.0, dtype=jnp.real(transfers[0]).dtype)

    for row in range(state.rows):
        applied = []
        for column in range(state.columns):
            transfer = transfers[row * state.columns + column]
            old = boundary[column]
            candidate = ein.contract("aub,urdl->aldrb", old, transfer, optimize="greedy")
            applied.append(
                candidate.reshape(
                    (
                        old.shape[0] * transfer.shape[3],
                        transfer.shape[2],
                        old.shape[2] * transfer.shape[1],
                    )
                )
            )
        boundary, errors, ranks, gauges = _compress_boundary(tuple(applied), policy)
        truncation_errors.extend(errors)
        retained_ranks.extend(ranks)
        gauge_residuals.extend(gauges)
        normalized = []
        for tensor in boundary:
            norm = jnp.linalg.norm(tensor)
            safe = jnp.where(norm > 0.0, norm, 1.0)
            normalized.append(tensor / safe)
            logarithmic_scale = logarithmic_scale + jnp.log(safe)
        boundary = tuple(normalized)

    message = jnp.ones((1,), dtype=boundary[0].dtype)
    for tensor in boundary:
        message = ein.contract("a,adb->db", message, tensor, optimize=False).reshape(
            (-1,)
        )
    scalar = jnp.sum(message) * jnp.exp(logarithmic_scale)
    errors_array = (
        jnp.stack(tuple(truncation_errors))
        if truncation_errors
        else jnp.zeros((0,), dtype=jnp.real(scalar).dtype)
    )
    ranks_array = (
        jnp.stack(tuple(retained_ranks))
        if retained_ranks
        else jnp.zeros((0,), dtype=jnp.int32)
    )
    gauges_array = (
        jnp.stack(tuple(gauge_residuals))
        if gauge_residuals
        else jnp.zeros((0,), dtype=jnp.real(scalar).dtype)
    )
    finite = (
        jnp.all(jnp.isfinite(scalar))
        & jnp.all(jnp.isfinite(errors_array))
        & jnp.all(jnp.isfinite(gauges_array))
    )
    exact = finite & jnp.all(errors_array == 0.0)
    replay_id = canonical_fingerprint(
        {
            "kind": "boundary-mps-replay",
            "state": state.state_id,
            "policy": policy.policy_id,
        }
    )
    evidence = BoundaryMPSEvidence(
        state.state_id,
        policy.policy_id,
        replay_id,
        state.rows,
        errors_array,
        ranks_array,
        gauges_array,
        logarithmic_scale,
        finite,
        exact,
        finite,
        "boundary-MPS approximation unless every measured discarded weight is zero; no global error bound",
        False,
        peak_elements,
        peak_bytes,
    )
    return BoundaryMPSResult(scalar, boundary, evidence)


__all__ = [
    "BoundaryMPSEvidence",
    "BoundaryMPSPolicy",
    "BoundaryMPSResult",
    "contract_peps_boundary_mps",
]
