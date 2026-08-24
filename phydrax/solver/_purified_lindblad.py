#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..tensor_network import LocallyPurifiedDensity, prepare_local_lindblad_channel


class LocalKrausChannel(StrictModule):
    kraus: Array
    site: int = eqx.field(static=True)
    channel_id: str = eqx.field(static=True)

    def __init__(self, site: int, kraus: ArrayLike, /, *, channel_id: str):
        values = jnp.asarray(kraus)
        if values.ndim != 3 or values.shape[-2] != values.shape[-1]:
            raise ValueError("Kraus operators require shape (count,d,d).")
        self.kraus = values
        self.site = int(site)
        self.channel_id = str(channel_id)

    def completeness_residual(self) -> Array:
        total = sum(jnp.conj(operator.T) @ operator for operator in self.kraus)
        return jnp.max(jnp.abs(total - jnp.eye(total.shape[0], dtype=total.dtype)))


def local_kraus_channel_from_lindblad(
    site: int,
    hamiltonian: ArrayLike,
    jump_operators: ArrayLike,
    step_size: ArrayLike,
    /,
    *,
    channel_id: str,
    tolerance: float = 1e-9,
) -> LocalKrausChannel:
    prepared = prepare_local_lindblad_channel(
        hamiltonian,
        jump_operators,
        step_size,
        tolerance=tolerance,
    )
    if not bool(jax.device_get(prepared.evidence.valid)):
        raise ValueError("Local Lindblad-to-Kraus preparation failed certification.")
    return LocalKrausChannel(site, prepared.kraus, channel_id=channel_id)


class PurificationTruncationEvidence(StrictModule):
    site: int
    available_rank: int
    retained_rank: int
    discarded_weight: Array
    channel_completeness_residual: Array
    valid: Array

    def __init__(
        self,
        site: int,
        available_rank: int,
        retained_rank: int,
        discarded_weight: ArrayLike,
        channel_completeness_residual: ArrayLike,
        /,
    ):
        self.site = int(site)
        self.available_rank = int(available_rank)
        self.retained_rank = int(retained_rank)
        self.discarded_weight = jnp.asarray(discarded_weight)
        self.channel_completeness_residual = jnp.asarray(channel_completeness_residual)
        self.valid = (
            jnp.isfinite(self.discarded_weight)
            & (self.discarded_weight >= 0.0)
            & (self.channel_completeness_residual <= 1e-8)
        )


def apply_local_kraus_channel(
    state: LocallyPurifiedDensity,
    channel: LocalKrausChannel,
    /,
    *,
    maximum_purification_dimension: int,
) -> tuple[LocallyPurifiedDensity, PurificationTruncationEvidence]:
    if not 0 <= channel.site < state.site_count:
        raise ValueError("Kraus-channel site is outside the purification.")
    tensor = state.tensors[channel.site]
    if tensor.shape[1] != channel.kraus.shape[-1]:
        raise ValueError("Kraus and physical dimensions differ.")
    transformed = oe.contract("aoi,likr->loakr", channel.kraus, tensor)
    transformed = transformed.reshape(
        (tensor.shape[0], tensor.shape[1], -1, tensor.shape[-1])
    )
    matrix = jnp.transpose(transformed, (0, 1, 3, 2)).reshape((-1, transformed.shape[2]))
    u, singular_values, _ = jnp.linalg.svd(matrix, full_matrices=False)
    available = singular_values.shape[0]
    retained = min(int(maximum_purification_dimension), available)
    discarded = jnp.sum(singular_values[retained:] ** 2)
    compressed = (u[:, :retained] * singular_values[:retained]).reshape(
        (tensor.shape[0], tensor.shape[1], tensor.shape[-1], retained)
    )
    compressed = jnp.transpose(compressed, (0, 1, 3, 2))
    tensors = list(state.tensors)
    tensors[channel.site] = compressed
    result = LocallyPurifiedDensity(tuple(tensors))
    evidence = PurificationTruncationEvidence(
        channel.site,
        available,
        retained,
        discarded,
        channel.completeness_residual(),
    )
    return result, evidence


class PurifiedLindbladProblem(StrictModule):
    initial_state: LocallyPurifiedDensity
    channels: tuple[LocalKrausChannel, ...]
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        initial_state: LocallyPurifiedDensity,
        channels: Sequence[LocalKrausChannel],
        /,
        *,
        problem_id: str = "purified-lindblad",
    ):
        channels_ = tuple(channels)
        if not channels_:
            raise ValueError("At least one local Kraus channel is required.")
        self.initial_state = initial_state
        self.channels = channels_
        self.problem_id = str(problem_id)


class PurifiedLindbladResult(StrictModule):
    final_state: LocallyPurifiedDensity
    trace_history: Array
    discarded_weight_history: Array
    valid: Array
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        final_state: LocallyPurifiedDensity,
        trace_history: ArrayLike,
        discarded_weight_history: ArrayLike,
        /,
        *,
        problem_id: str,
    ):
        self.final_state = final_state
        self.trace_history = jnp.asarray(trace_history)
        self.discarded_weight_history = jnp.asarray(discarded_weight_history)
        self.valid = (
            jnp.all(jnp.isfinite(self.trace_history))
            & jnp.all(jnp.abs(self.trace_history - 1.0) <= 1e-6)
            & jnp.all(self.discarded_weight_history >= 0.0)
        )
        self.problem_id = str(problem_id)


def solve_purified_lindblad(
    problem: PurifiedLindbladProblem,
    /,
    *,
    steps: int,
    maximum_purification_dimension: int,
) -> PurifiedLindbladResult:
    state = problem.initial_state
    traces = [state.raw_trace()]
    discarded = []
    for _ in range(int(steps)):
        for channel in problem.channels:
            state, evidence = apply_local_kraus_channel(
                state,
                channel,
                maximum_purification_dimension=maximum_purification_dimension,
            )
            discarded.append(evidence.discarded_weight)
        traces.append(state.raw_trace())
    return PurifiedLindbladResult(
        state,
        jnp.stack(traces),
        jnp.stack(discarded) if discarded else jnp.zeros((0,)),
        problem_id=problem.problem_id,
    )


__all__ = [
    "LocalKrausChannel",
    "PurificationTruncationEvidence",
    "PurifiedLindbladProblem",
    "PurifiedLindbladResult",
    "apply_local_kraus_channel",
    "local_kraus_channel_from_lindblad",
    "solve_purified_lindblad",
]
