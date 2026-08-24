#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from .._sampling import derive_key, SampleAddress
from .._strict import StrictModule
from ..nonlinear import scalar_root, ScalarRootProblem, TOMS748
from ..tensor_network import MatrixProductState, NearestNeighborHamiltonian, tebd_step


class LocalMPSJump(StrictModule):
    operator: Array
    site: int = eqx.field(static=True)
    jump_id: str = eqx.field(static=True)

    def __init__(self, site: int, operator: ArrayLike, /, *, jump_id: str):
        value = jnp.asarray(operator)
        if value.ndim != 2 or value.shape[0] != value.shape[1]:
            raise ValueError("Local jump operator must be square.")
        self.operator = value
        self.site = int(site)
        self.jump_id = str(jump_id)

    def apply(
        self, state: MatrixProductState, /, *, normalize: bool
    ) -> MatrixProductState:
        if not 0 <= self.site < state.site_count:
            raise ValueError("Jump site is outside the MPS.")
        tensor = state.tensors[self.site]
        if self.operator.shape[1] != tensor.shape[1]:
            raise ValueError("Jump physical dimension does not match the MPS site.")
        operator = state.precision.contraction(self.operator)
        updated = oe.contract(
            "oi,lir->lor",
            operator,
            state.precision.contraction(tensor),
        )
        tensors = list(state.tensors)
        tensors[self.site] = state.precision.storage(updated)
        result = MatrixProductState(tuple(tensors), precision=state.precision)
        return result.normalized() if normalize else result

    def rate(self, state: MatrixProductState, /) -> Array:
        transformed = self.apply(state, normalize=False)
        return transformed.norm() ** 2


class MPSQuantumJumpProblem(StrictModule):
    hamiltonian: NearestNeighborHamiltonian
    jumps: tuple[LocalMPSJump, ...]
    initial_state: MatrixProductState
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        hamiltonian: NearestNeighborHamiltonian,
        jumps: Sequence[LocalMPSJump],
        initial_state: MatrixProductState,
        /,
        *,
        problem_id: str = "mps-quantum-jump",
    ):
        jumps_ = tuple(jumps)
        if not jumps_:
            raise ValueError("At least one MPS jump operator is required.")
        if tuple(initial_state.physical_dimensions) != hamiltonian.physical_dimensions:
            raise ValueError("MPS and Hamiltonian dimensions differ.")
        self.hamiltonian = hamiltonian
        self.jumps = jumps_
        self.initial_state = initial_state.normalized()
        self.problem_id = str(problem_id)


class MPSQuantumTrajectoryResult(StrictModule):
    final_state: MatrixProductState
    jump_times: Array
    jump_channels: Array
    active_events: Array
    discarded_weight_history: Array
    root_residuals: Array
    root_ambiguous: Array
    event_capacity_saturated: Array
    event_count: int
    maximum_events: int
    valid: Array
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        final_state: MatrixProductState,
        jump_times: ArrayLike,
        jump_channels: ArrayLike,
        active_events: ArrayLike,
        discarded_weight_history: ArrayLike,
        root_residuals: ArrayLike,
        root_ambiguous: ArrayLike,
        event_capacity_saturated: ArrayLike,
        /,
        *,
        problem_id: str,
    ):
        self.final_state = final_state
        self.jump_times = jnp.asarray(jump_times)
        self.jump_channels = jnp.asarray(jump_channels, dtype=jnp.int32)
        self.active_events = jnp.asarray(active_events, dtype=bool)
        self.discarded_weight_history = jnp.asarray(discarded_weight_history)
        self.root_residuals = jnp.asarray(root_residuals)
        self.root_ambiguous = jnp.asarray(root_ambiguous, dtype=bool)
        self.event_capacity_saturated = jnp.asarray(
            event_capacity_saturated, dtype=bool
        )
        self.event_count = int(jnp.sum(self.active_events))
        self.maximum_events = int(self.active_events.shape[0])
        self.valid = (
            jnp.isfinite(final_state.norm())
            & (jnp.abs(final_state.norm() - 1.0) <= 1e-6)
            & jnp.all(jnp.isfinite(self.discarded_weight_history))
            & jnp.all(
                jnp.where(
                    self.active_events,
                    jnp.isfinite(self.root_residuals),
                    True,
                )
            )
            & ~jnp.any(self.root_ambiguous)
            & ~self.event_capacity_saturated
        )
        self.problem_id = str(problem_id)


def _nonhermitian_mps_step(
    problem: MPSQuantumJumpProblem,
    state: MatrixProductState,
    duration: Array,
    maximum_bond_dimension: int,
):
    normals: dict[int, Array] = {}
    for jump in problem.jumps:
        normal = jnp.conj(jump.operator.T) @ jump.operator
        normals[jump.site] = normals.get(
            jump.site, jnp.zeros_like(normal)
        ) + normal

    def damp(current, scale):
        result = current
        for site, normal in sorted(normals.items()):
            gate = jsp.linalg.expm(-scale * normal)
            result = LocalMPSJump(
                site,
                gate,
                jump_id=f"{problem.problem_id}:effective-site-{site}",
            ).apply(result, normalize=False)
        return result

    evolved = damp(state, 0.25 * duration)
    evolved, evidence = tebd_step(
        evolved,
        problem.hamiltonian,
        duration,
        maximum_bond_dimension=maximum_bond_dimension,
        order=2,
        normalize=False,
    )
    return damp(evolved, 0.25 * duration), evidence


def solve_mps_quantum_jump(
    problem: MPSQuantumJumpProblem,
    key: Array,
    /,
    *,
    step_size: ArrayLike,
    steps: int,
    maximum_bond_dimension: int,
    maximum_events: int = 128,
    root_truncation_tolerance: float = 1e-6,
    root_residual_tolerance: float = 1e-8,
) -> MPSQuantumTrajectoryResult:
    step = jnp.asarray(step_size, dtype=float).reshape(())
    step_count = int(steps)
    event_limit = int(maximum_events)
    bond_limit = int(maximum_bond_dimension)
    if (
        not bool(jnp.isfinite(step))
        or float(step) <= 0.0
        or step_count <= 0
        or event_limit <= 0
        or bond_limit <= 0
        or root_truncation_tolerance < 0.0
        or root_residual_tolerance < 0.0
    ):
        raise ValueError("MPS trajectory policy values are invalid.")
    state = problem.initial_state
    times = jnp.zeros((event_limit,), dtype=step.dtype)
    channels = -jnp.ones((event_limit,), dtype=jnp.int32)
    active = jnp.zeros((event_limit,), dtype=bool)
    root_residuals = jnp.full((event_limit,), jnp.nan, dtype=step.dtype)
    root_ambiguous = jnp.zeros((event_limit,), dtype=bool)
    discarded = []
    event_count = 0
    capacity_saturated = False
    threshold_address = SampleAddress(
        "quantum-trajectory",
        "mps-jump-threshold",
        target=problem.problem_id,
        role="threshold",
    )
    channel_address = SampleAddress(
        "quantum-trajectory",
        "mps-jump-channel",
        target=problem.problem_id,
        role="channel",
    )
    threshold = jax.random.uniform(derive_key(key, threshold_address, 0))
    for index in range(step_count):
        remaining = float(step)
        elapsed = 0.0
        while remaining > 1e-15:
            start = state
            duration = jnp.asarray(remaining, dtype=step.dtype)
            candidate, evidence = _nonhermitian_mps_step(
                problem,
                start,
                duration,
                bond_limit,
            )
            discarded.append(evidence.cumulative_discarded_weight)
            crossing = candidate.norm() ** 2 <= threshold
            if not bool(jax.device_get(crossing)):
                state = candidate
                break
            if event_count >= event_limit:
                capacity_saturated = True
                state = candidate
                remaining = 0.0
                break

            def survival_residual(event_duration, args):
                del args
                probe, _ = _nonhermitian_mps_step(
                    problem,
                    start,
                    event_duration,
                    bond_limit,
                )
                return probe.norm() ** 2 - threshold

            root_problem = ScalarRootProblem(
                survival_residual,
                bracket=(jnp.asarray(0.0), duration),
                problem_id=f"{problem.problem_id}:mps-survival-root",
            )
            root = scalar_root(root_problem, method=TOMS748())
            event_duration = root.root
            event_state, event_evidence = _nonhermitian_mps_step(
                problem,
                start,
                event_duration,
                bond_limit,
            )
            discarded.append(event_evidence.cumulative_discarded_weight)
            residual = jnp.abs(root.value)
            ambiguous = (
                ~root.successful
                | ~jnp.isfinite(event_duration)
                | (event_duration <= 0.0)
                | (event_duration > duration)
                | (
                    jnp.sqrt(event_evidence.cumulative_discarded_weight)
                    > root_truncation_tolerance
                )
                | (residual > root_residual_tolerance)
            )
            root_residuals = root_residuals.at[event_count].set(residual)
            if bool(jax.device_get(ambiguous)):
                root_ambiguous = root_ambiguous.at[event_count].set(True)
                state = candidate
                remaining = 0.0
                break
            normalized_event = event_state.normalized()
            rates = jnp.stack([jump.rate(normalized_event) for jump in problem.jumps])
            total = jnp.sum(rates)
            if not bool(
                jnp.all(jnp.isfinite(rates))
                & jnp.all(rates >= 0.0)
                & (total > 0.0)
            ):
                root_ambiguous = root_ambiguous.at[event_count].set(True)
                state = candidate
                remaining = 0.0
                break
            local_key = derive_key(key, channel_address, event_count)
            channel = jax.random.categorical(local_key, jnp.log(rates / total))
            state = problem.jumps[int(channel)].apply(
                normalized_event, normalize=True
            )
            event_time = index * step + elapsed + event_duration
            times = times.at[event_count].set(event_time)
            channels = channels.at[event_count].set(
                jnp.asarray(channel, dtype=jnp.int32)
            )
            active = active.at[event_count].set(True)
            event_count += 1
            elapsed += float(event_duration)
            remaining -= float(event_duration)
            threshold = jax.random.uniform(
                derive_key(key, threshold_address, event_count)
            )
        if capacity_saturated or bool(jnp.any(root_ambiguous)):
            break
    final_state = state.normalized()
    return MPSQuantumTrajectoryResult(
        final_state,
        times,
        channels,
        active,
        jnp.stack(discarded) if discarded else jnp.zeros((0,)),
        root_residuals,
        root_ambiguous,
        capacity_saturated,
        problem_id=problem.problem_id,
    )


__all__ = [
    "LocalMPSJump",
    "MPSQuantumJumpProblem",
    "MPSQuantumTrajectoryResult",
    "solve_mps_quantum_jump",
]
