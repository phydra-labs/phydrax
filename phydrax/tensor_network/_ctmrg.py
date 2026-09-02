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


class CTMRGPolicy(StrictModule):
    environment_dimension: int = eqx.field(static=True)
    maximum_iterations: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    maximum_tensor_elements: int = eqx.field(static=True)
    maximum_workspace_bytes: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        environment_dimension: int,
        maximum_iterations: int,
        /,
        *,
        tolerance: float = 1e-8,
        maximum_tensor_elements: int = 100_000_000,
        maximum_workspace_bytes: int = 2**31,
    ):
        chi = int(environment_dimension)
        iterations = int(maximum_iterations)
        tolerance_ = float(tolerance)
        tensor_limit = int(maximum_tensor_elements)
        workspace = int(maximum_workspace_bytes)
        if (
            chi < 1
            or iterations < 1
            or tolerance_ <= 0.0
            or tensor_limit < 1
            or workspace < 1
        ):
            raise ValueError("CTMRG policy requires positive finite bounds.")
        self.environment_dimension = chi
        self.maximum_iterations = iterations
        self.tolerance = tolerance_
        self.maximum_tensor_elements = tensor_limit
        self.maximum_workspace_bytes = workspace
        self.policy_id = canonical_fingerprint(
            {
                "kind": "ctmrg-policy",
                "environment_dimension": chi,
                "maximum_iterations": iterations,
                "tolerance": tolerance_,
                "maximum_tensor_elements": tensor_limit,
                "maximum_workspace_bytes": workspace,
            }
        )


class CTMRGEnvironment(StrictModule):
    corners: tuple[Array, Array, Array, Array]
    edges: tuple[Array, Array, Array, Array]
    logarithmic_scale: Array
    environment_id: str = eqx.field(static=True)


class CTMRGEvidence(StrictModule):
    state_id: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)
    replay_id: str = eqx.field(static=True)
    residual_history: Array
    active_mask: Array
    convergence_mask: Array
    truncation_errors: Array
    gauge_residual: Array
    converged: Array
    finite: Array
    accepted: Array
    exact: Array
    claim: str = eqx.field(static=True)
    global_error_bound_claimed: bool = eqx.field(static=True)
    admitted_peak_elements: int = eqx.field(static=True)
    admitted_peak_bytes: int = eqx.field(static=True)


class CTMRGResult(StrictModule):
    value: Array
    environment: CTMRGEnvironment
    evidence: CTMRGEvidence


def _double_tensor(tensor: Array, /) -> Array:
    shape = tuple(int(value * value) for value in tensor.shape[:4])
    return ein.contract(
        "urdlp,URDLp->uUrRdDlL",
        jnp.conj(tensor),
        tensor,
        optimize="greedy",
    ).reshape(shape)


def _pad_matrix(matrix: Array, dimension: int, /) -> Array:
    result = jnp.zeros((dimension, dimension), dtype=matrix.dtype)
    return result.at[: matrix.shape[0], : matrix.shape[1]].add(matrix)


def _positive_channel(channel: Array, /) -> Array:
    value = channel @ jnp.conj(channel.T)
    trace = jnp.real(jnp.trace(value))
    return value / jnp.where(trace > 0.0, trace, 1.0)


def _renormalize_corner(
    corner: Array, channel: Array, retained: int, /
) -> tuple[Array, Array]:
    candidate = channel @ corner @ channel
    candidate = 0.5 * (candidate + jnp.conj(candidate.T))
    decomposition = factorize(DenseLinearOperator(candidate), FactorizationPolicy("svd"))
    svd = decomposition.prepared_solve.state
    values = jnp.maximum(jnp.real(svd.singular_values), 0.0)
    vectors = svd.u
    keep = jnp.arange(values.shape[0]) < retained
    discarded = jnp.sum(jnp.where(keep, 0.0, values))
    truncated = (vectors * jnp.where(keep, values, 0.0)[None, :]) @ jnp.conj(vectors.T)
    trace = jnp.real(jnp.trace(truncated))
    return truncated / jnp.where(trace > 0.0, trace, 1.0), discarded


def contract_peps_ctmrg(state: PEPS, policy: CTMRGPolicy, /) -> CTMRGResult:
    """Run a fixed-capacity four-corner CTMRG environment iteration."""

    if not isinstance(state, PEPS) or not isinstance(policy, CTMRGPolicy):
        raise TypeError("state and policy have invalid types.")
    doubled_shapes = tuple(
        tuple(int(value * value) for value in tensor.shape[:4])
        for tensor in state.tensors
    )
    dimension = max(max(shape) for shape in doubled_shapes)
    transfer_elements = sum(prod(shape) for shape in doubled_shapes)
    environment_elements = 8 * dimension * dimension
    peak_elements = transfer_elements + environment_elements * 3
    if (
        transfer_elements > policy.maximum_tensor_elements
        or peak_elements > policy.maximum_tensor_elements
    ):
        raise MemoryError(
            "CTMRG tensors exceed maximum_tensor_elements before allocation."
        )
    peak_bytes = peak_elements * precision_itemsize(str(state.tensors[0].dtype))
    if peak_bytes > policy.maximum_workspace_bytes:
        raise MemoryError("CTMRG exceeds maximum_workspace_bytes before allocation.")

    transfers = tuple(_double_tensor(tensor) for tensor in state.tensors)
    horizontal = jnp.zeros((dimension, dimension), dtype=transfers[0].dtype)
    vertical = jnp.zeros_like(horizontal)
    logarithmic_scale = jnp.asarray(0.0, dtype=jnp.real(horizontal).dtype)
    for transfer in transfers:
        norm = jnp.linalg.norm(transfer)
        safe = jnp.where(norm > 0.0, norm, 1.0)
        normalized = transfer / safe
        logarithmic_scale = logarithmic_scale + jnp.log(safe)
        horizontal = horizontal + _pad_matrix(jnp.sum(normalized, axis=(0, 2)), dimension)
        vertical = vertical + _pad_matrix(jnp.sum(normalized, axis=(1, 3)), dimension)
    horizontal = _positive_channel(horizontal)
    vertical = _positive_channel(vertical)
    channels = (horizontal, jnp.conj(horizontal.T), vertical, jnp.conj(vertical.T))
    identity = jnp.eye(dimension, dtype=horizontal.dtype) / dimension
    corners = (identity, identity, identity, identity)
    retained = min(policy.environment_dimension, dimension)
    active = jnp.asarray(True)
    residual_history = []
    active_history = []
    convergence_history = []
    truncation_history = []

    for _ in range(policy.maximum_iterations):
        active_history.append(active)
        candidates = []
        discarded = []
        for corner, channel in zip(corners, channels, strict=True):
            candidate, error = _renormalize_corner(corner, channel, retained)
            candidates.append(candidate)
            discarded.append(error)
        residual = jnp.max(
            jnp.stack(
                tuple(
                    jnp.max(jnp.abs(candidate - corner))
                    for candidate, corner in zip(candidates, corners, strict=True)
                )
            )
        )
        converged_now = active & (residual <= policy.tolerance)
        corners = tuple(
            jnp.where(active, candidate, corner)
            for candidate, corner in zip(candidates, corners, strict=True)
        )
        residual_history.append(jnp.where(active, residual, 0.0))
        convergence_history.append(converged_now)
        truncation_history.append(jnp.sum(jnp.stack(tuple(discarded))))
        active = active & ~converged_now

    left_right = jnp.real(jnp.trace(corners[0] @ horizontal @ corners[1] @ horizontal))
    up_down = jnp.real(jnp.trace(corners[2] @ vertical @ corners[3] @ vertical))
    reduced_value = jnp.sqrt(jnp.maximum(jnp.abs(left_right * up_down), 0.0))
    value = reduced_value * jnp.exp(logarithmic_scale)
    residuals = jnp.stack(tuple(residual_history))
    active_mask = jnp.stack(tuple(active_history))
    convergence_mask = jnp.stack(tuple(convergence_history))
    truncation = jnp.stack(tuple(truncation_history))
    gauge = jnp.max(
        jnp.stack(
            tuple(
                jnp.maximum(
                    jnp.max(jnp.abs(corner - jnp.conj(corner.T))),
                    jnp.abs(jnp.real(jnp.trace(corner)) - 1.0),
                )
                for corner in corners
            )
        )
    )
    converged = jnp.any(convergence_mask)
    finite = (
        jnp.all(jnp.isfinite(value))
        & jnp.all(jnp.isfinite(residuals))
        & jnp.all(jnp.isfinite(truncation))
        & jnp.isfinite(gauge)
    )
    accepted = finite & converged
    exact = accepted & jnp.asarray(dimension == 1)
    environment_id = canonical_fingerprint(
        {"kind": "ctmrg-environment", "state": state.state_id, "policy": policy.policy_id}
    )
    environment = CTMRGEnvironment(corners, channels, logarithmic_scale, environment_id)
    replay_id = canonical_fingerprint(
        {
            "kind": "ctmrg-replay",
            "environment": environment_id,
            "iterations": policy.maximum_iterations,
        }
    )
    evidence = CTMRGEvidence(
        state.state_id,
        policy.policy_id,
        replay_id,
        residuals,
        active_mask,
        convergence_mask,
        truncation,
        gauge,
        converged,
        finite,
        accepted,
        exact,
        "finite CTMRG approximation; exact only for unit virtual dimension; no global error bound",
        False,
        peak_elements,
        peak_bytes,
    )
    return CTMRGResult(value, environment, evidence)


__all__ = [
    "CTMRGEvidence",
    "CTMRGEnvironment",
    "CTMRGPolicy",
    "CTMRGResult",
    "contract_peps_ctmrg",
]
