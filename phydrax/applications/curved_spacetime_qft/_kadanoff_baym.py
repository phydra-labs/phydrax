#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""First-gradient Wigner Kadanoff--Baym transport with finite memory."""

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...solver._dark_sector_epoch_runtime import DarkSectorEpochPlan, DarkSectorEpochState
from ._off_shell_transport import (
    off_shell_evidence,
    OffShellTransportPlan,
    QuasiparticleOffShellState,
)


def _identifier(value: str, name: str, /) -> str:
    identifier = str(value).strip()
    if not identifier:
        raise ValueError(f"{name} must be non-empty.")
    return identifier


def _central_periodic(value: Array, axis: int, spacing: float, /) -> Array:
    return (jnp.roll(value, -1, axis=axis) - jnp.roll(value, 1, axis=axis)) / (
        2.0 * spacing
    )


class WignerGradientPlan(StrictModule, NonTrainableState):
    phase_space_shape: tuple[int, ...] = eqx.field(static=True)
    spacetime_spacings: tuple[float, ...] = eqx.field(static=True)
    momentum_spacings: tuple[float, ...] = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    boundary: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        phase_space_shape: Sequence[int],
        spacetime_spacings: Sequence[float],
        momentum_spacings: Sequence[float],
        /,
        *,
        boundary: str = "periodic",
    ):
        shape = tuple(int(value) for value in phase_space_shape)
        x_steps = tuple(float(value) for value in spacetime_spacings)
        p_steps = tuple(float(value) for value in momentum_spacings)
        if not x_steps or len(x_steps) != len(p_steps):
            raise ValueError(
                "Wigner phase space requires paired spacetime/momentum axes."
            )
        dimension = len(x_steps)
        if len(shape) != 2 * dimension or any(value < 3 for value in shape):
            raise ValueError(
                "phase_space_shape must provide at least three nodes on each paired axis."
            )
        if any(not np.isfinite(value) or value <= 0.0 for value in x_steps + p_steps):
            raise ValueError("Wigner grid spacings must be finite and positive.")
        if boundary != "periodic":
            raise ValueError(
                "First-gradient Wigner differences currently require periodic support."
            )
        self.phase_space_shape = shape
        self.spacetime_spacings = x_steps
        self.momentum_spacings = p_steps
        self.dimension = dimension
        self.boundary = boundary
        self.plan_id = canonical_fingerprint(
            {
                "kind": "centered-first-gradient-wigner-plan",
                "phase_space_shape": shape,
                "spacetime_spacings": x_steps,
                "momentum_spacings": p_steps,
                "boundary": boundary,
                "order": 2,
            }
        )

    def poisson_bracket(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        """Evaluate the canonical first-gradient Poisson bracket without materialization."""

        left_ = jnp.asarray(left)
        right_ = jnp.asarray(right)
        if (
            left_.shape != right_.shape
            or left_.shape[: 2 * self.dimension] != self.phase_space_shape
        ):
            raise ValueError(
                "Poisson-bracket operands must share the declared leading phase-space shape."
            )
        result = jnp.zeros_like(left_ * right_)
        for axis in range(self.dimension):
            dx_left = _central_periodic(left_, axis, self.spacetime_spacings[axis])
            dx_right = _central_periodic(right_, axis, self.spacetime_spacings[axis])
            momentum_axis = self.dimension + axis
            dp_left = _central_periodic(
                left_, momentum_axis, self.momentum_spacings[axis]
            )
            dp_right = _central_periodic(
                right_, momentum_axis, self.momentum_spacings[axis]
            )
            result = result + dx_left * dp_right - dp_left * dx_right
        return result


class KadanoffBaymProfile(StrictModule, NonTrainableState):
    support: tuple[str, ...] = eqx.field(static=True)
    refusals: tuple[str, ...] = eqx.field(static=True)
    differentiation: str = eqx.field(static=True)
    rights_id: str = eqx.field(static=True)
    provenance_ids: tuple[str, ...] = eqx.field(static=True)
    production_evidence_ids: tuple[str, ...] = eqx.field(static=True)
    production_ready: bool = eqx.field(static=True)
    profile_id: str = eqx.field(static=True)

    def __init__(self, production_evidence_ids: Sequence[str] = (), /):
        evidence = tuple(
            _identifier(value, "production_evidence_id")
            for value in production_evidence_ids
        )
        if len(set(evidence)) != len(evidence):
            raise ValueError("Production evidence identities must be unique.")
        support = (
            "first-gradient-poisson-bracket",
            "fixed-depth-causal-memory-ring",
            "durable-finite-epoch-continuation",
        )
        refusals = (
            "unbounded-device-memory",
            "silent-tail-discard",
            "negative-occupation-clipping",
            "checkpoint-identity-mismatch",
        )
        differentiation = "algorithmic-fixed-depth-ring-and-centered-gradient"
        rights_id = "phydra-native-no-external-provider"
        provenance_ids = ("first-gradient-kadanoff-baym-finite-memory",)
        self.support = support
        self.refusals = refusals
        self.differentiation = differentiation
        self.rights_id = rights_id
        self.provenance_ids = provenance_ids
        self.production_evidence_ids = evidence
        self.production_ready = bool(evidence)
        self.profile_id = canonical_fingerprint(
            {
                "kind": "kadanoff-baym-dark-transport-profile",
                "support": support,
                "refusals": refusals,
                "differentiation": differentiation,
                "rights_id": rights_id,
                "provenance_ids": provenance_ids,
                "production_evidence_ids": evidence,
            }
        )


class KadanoffBaymTransportPlan(StrictModule, NonTrainableState):
    off_shell: OffShellTransportPlan
    gradient: WignerGradientPlan
    epoch: DarkSectorEpochPlan
    profile: KadanoffBaymProfile
    time_step: float = eqx.field(static=True)
    memory_depth: int = eqx.field(static=True)
    maximum_memory_bytes: int = eqx.field(static=True)
    capacity_revision_id: str = eqx.field(static=True)
    compile_signature_id: str = eqx.field(static=True)
    support_id: str = eqx.field(static=True)
    frame_id: str = eqx.field(static=True)
    frame_realization_id: str = eqx.field(static=True)
    unit_contract_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        off_shell: OffShellTransportPlan,
        gradient: WignerGradientPlan,
        epoch: DarkSectorEpochPlan,
        /,
        *,
        time_step: float,
        memory_depth: int,
        maximum_memory_bytes: int = 512_000_000,
        production_evidence_ids: Sequence[str] = (),
    ):
        if not isinstance(off_shell, OffShellTransportPlan):
            raise TypeError("off_shell must be an OffShellTransportPlan.")
        if not isinstance(gradient, WignerGradientPlan):
            raise TypeError("gradient must be a WignerGradientPlan.")
        if not isinstance(epoch, DarkSectorEpochPlan):
            raise TypeError("epoch must be a DarkSectorEpochPlan.")
        if (
            off_shell.spectral_shape[: 2 * gradient.dimension]
            != gradient.phase_space_shape
        ):
            raise ValueError("Wigner gradient axes do not match off-shell support.")
        width = float(time_step)
        depth = int(memory_depth)
        maximum = int(maximum_memory_bytes)
        if not np.isfinite(width) or width <= 0.0 or depth < 1 or maximum < 1:
            raise ValueError(
                "Kadanoff--Baym time, depth, and memory capacity are invalid."
            )
        required = depth * int(np.prod(off_shell.spectral_shape)) * 3 * 8
        if required > maximum:
            raise ValueError(
                f"Kadanoff--Baym memory requires {required} bytes; capacity is {maximum}."
            )
        profile = KadanoffBaymProfile(production_evidence_ids)
        self.off_shell = off_shell
        self.gradient = gradient
        self.epoch = epoch
        self.profile = profile
        self.time_step = width
        self.memory_depth = depth
        self.maximum_memory_bytes = maximum
        self.capacity_revision_id = epoch.capacity_revision_id
        self.compile_signature_id = epoch.compile_signature_id
        self.support_id = off_shell.support_id
        self.frame_id = off_shell.frame_id
        self.frame_realization_id = off_shell.frame_realization_id
        self.unit_contract_id = off_shell.unit_contract_id
        self.plan_id = canonical_fingerprint(
            {
                "kind": "finite-memory-first-gradient-kadanoff-baym",
                "off_shell": off_shell.plan_id,
                "gradient": gradient.plan_id,
                "epoch_plan": epoch.plan_id,
                "capacity_revision": epoch.capacity_revision_id,
                "compile_signature": epoch.compile_signature_id,
                "frame_realization": off_shell.frame_realization_id,
                "time_step": width,
                "memory_depth": depth,
                "maximum_memory_bytes": maximum,
                "profile": profile.profile_id,
            }
        )


class KBMemoryState(StrictModule):
    source_history: Array
    kernel_history: Array
    initial_correlation_history: Array
    sample_times: Array
    valid: Array
    head: Array
    committed_samples: Array
    discarded_tail_bound: Array
    epoch_sequence: Array
    frame_token: Array
    frame_time: Array
    frame_scale_factor: Array
    frame_realization_id: str = eqx.field(static=True)
    parent_epoch_manifest_id: str | None = eqx.field(static=True)
    runtime_plan_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class KBMemoryEvidence(StrictModule):
    convolution: Array
    initial_correlation_contribution: Array
    initial_correlation_norm: Array
    discarded_tail_bound: Array
    active_depth: Array
    finite: Array
    causal: Array
    tail_reported: Array
    epoch_sequence: Array
    frame_token: Array
    frame_time: Array
    frame_scale_factor: Array
    frame_realization_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class KadanoffBaymStepEvidence(StrictModule):
    memory: KBMemoryEvidence
    poisson_bracket_norm: Array
    drift_poisson_norm: Array
    backflow_poisson_norm: Array
    collision_norm: Array
    occupation_change_norm: Array
    number_change: Array
    finite: Array
    accepted: Array
    rolled_back: Array
    status: Array
    frame_token: Array
    frame_time: Array
    frame_scale_factor: Array
    frame_realization_id: str = eqx.field(static=True)
    differentiation: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    support_id: str = eqx.field(static=True)
    frame_id: str = eqx.field(static=True)
    unit_contract_id: str = eqx.field(static=True)


class KadanoffBaymStepResult(StrictModule):
    proposed_state: QuasiparticleOffShellState
    accepted_state: QuasiparticleOffShellState
    proposed_memory: KBMemoryState
    accepted_memory: KBMemoryState
    evidence: KadanoffBaymStepEvidence
    plan_id: str = eqx.field(static=True)


class KadanoffBaymCheckpoint(StrictModule, NonTrainableState):
    state: QuasiparticleOffShellState
    memory: KBMemoryState
    frame_token: Array
    frame_time: Array
    frame_scale_factor: Array
    epoch_manifest_id: str = eqx.field(static=True)
    checkpoint_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    frame_id: str = eqx.field(static=True)
    frame_realization_id: str = eqx.field(static=True)
    unit_contract_id: str = eqx.field(static=True)


def initialize_kadanoff_baym_memory(
    plan: KadanoffBaymTransportPlan,
    runtime_state: DarkSectorEpochState,
    /,
) -> KBMemoryState:
    if not isinstance(plan, KadanoffBaymTransportPlan):
        raise TypeError("plan must be KadanoffBaymTransportPlan.")
    if (
        not isinstance(runtime_state, DarkSectorEpochState)
        or runtime_state.plan.plan_id != plan.epoch.plan_id
    ):
        raise ValueError("Runtime epoch state belongs to a different epoch plan.")
    history_shape = (plan.memory_depth,) + plan.off_shell.spectral_shape
    zeros = jnp.zeros(history_shape, dtype=plan.off_shell.energy_nodes.dtype)
    return KBMemoryState(
        zeros,
        zeros,
        zeros,
        jnp.zeros((plan.memory_depth,), dtype=plan.off_shell.energy_nodes.dtype),
        jnp.zeros((plan.memory_depth,), dtype=bool),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int64),
        jnp.asarray(0.0, dtype=plan.off_shell.energy_nodes.dtype),
        jnp.asarray(runtime_state.epoch_sequence, dtype=jnp.int64),
        plan.off_shell.frame.frame_token,
        plan.off_shell.frame.time,
        plan.off_shell.frame.scale_factor,
        plan.frame_realization_id,
        runtime_state.parent_epoch_manifest_id,
        runtime_state.plan.plan_id,
        plan.plan_id,
    )


def _append_memory(
    plan: KadanoffBaymTransportPlan,
    memory: KBMemoryState,
    source: Array,
    kernel: Array,
    initial_correlation: Array,
    sample_time: Array,
    /,
) -> KBMemoryState:
    if memory.plan_id != plan.plan_id:
        raise ValueError("Memory state belongs to a different Kadanoff--Baym plan.")
    expected = plan.off_shell.spectral_shape
    if (
        source.shape != expected
        or kernel.shape != expected
        or initial_correlation.shape != expected
    ):
        raise ValueError(
            "Memory source, kernel, and initial correlation must match off-shell support."
        )
    latest_time = jnp.max(jnp.where(memory.valid, memory.sample_times, -jnp.inf))
    sample_time = eqx.error_if(
        sample_time,
        (~jnp.isfinite(sample_time))
        | (jnp.any(memory.valid) & (sample_time < latest_time)),
        "Kadanoff--Baym memory samples must be finite and causal.",
    )
    index = memory.head
    overwritten = memory.valid[index]
    evicted = plan.time_step * (
        memory.kernel_history[index] * memory.source_history[index]
        + memory.initial_correlation_history[index]
    )
    evicted_bound = jnp.sum(jnp.abs(evicted))
    tail = memory.discarded_tail_bound + jnp.where(overwritten, evicted_bound, 0.0)
    return KBMemoryState(
        memory.source_history.at[index].set(source),
        memory.kernel_history.at[index].set(kernel),
        memory.initial_correlation_history.at[index].set(initial_correlation),
        memory.sample_times.at[index].set(sample_time),
        memory.valid.at[index].set(True),
        (index + 1) % plan.memory_depth,
        memory.committed_samples + 1,
        tail,
        memory.epoch_sequence,
        memory.frame_token,
        memory.frame_time,
        memory.frame_scale_factor,
        memory.frame_realization_id,
        memory.parent_epoch_manifest_id,
        memory.runtime_plan_id,
        memory.plan_id,
    )


def kadanoff_baym_memory_evidence(
    plan: KadanoffBaymTransportPlan,
    memory: KBMemoryState,
    /,
) -> KBMemoryEvidence:
    if not isinstance(plan, KadanoffBaymTransportPlan) or memory.plan_id != plan.plan_id:
        raise ValueError("Memory evidence requires matching Kadanoff--Baym plan/state.")
    mask = memory.valid.reshape(
        (plan.memory_depth,) + (1,) * len(plan.off_shell.spectral_shape)
    )
    convolution_terms = plan.time_step * memory.kernel_history * memory.source_history
    initial_terms = plan.time_step * memory.initial_correlation_history
    convolution = jnp.sum(jnp.where(mask, convolution_terms + initial_terms, 0.0), axis=0)
    initial = jnp.sum(jnp.where(mask, initial_terms, 0.0), axis=0)
    finite = (
        jnp.all(jnp.isfinite(convolution))
        & jnp.all(jnp.isfinite(memory.sample_times))
        & jnp.isfinite(memory.discarded_tail_bound)
    )
    pair_active = memory.valid[:, None] & memory.valid[None, :]
    distinct = ~jnp.eye(plan.memory_depth, dtype=bool)
    duplicate_time = (
        pair_active
        & distinct
        & (memory.sample_times[:, None] == memory.sample_times[None, :])
    )
    causal = finite & ~jnp.any(duplicate_time)
    return KBMemoryEvidence(
        convolution,
        initial,
        jnp.sum(jnp.abs(initial)),
        memory.discarded_tail_bound,
        jnp.sum(memory.valid.astype(jnp.int32)),
        finite,
        causal,
        memory.discarded_tail_bound >= 0.0,
        memory.epoch_sequence,
        memory.frame_token,
        memory.frame_time,
        memory.frame_scale_factor,
        memory.frame_realization_id,
        plan.plan_id,
    )


def _raw_state(
    plan: OffShellTransportPlan,
    template: QuasiparticleOffShellState,
    occupation: Array,
    time: Array,
    /,
) -> QuasiparticleOffShellState:
    evidence = off_shell_evidence(
        plan,
        template.spectral_function,
        occupation,
        template.real_retarded_self_energy,
        template.width,
        template.lesser_self_energy,
        template.greater_self_energy,
        template.pole_energy,
        template.inverse_temperature,
        template.chemical_potentials,
    )
    return QuasiparticleOffShellState(
        template.spectral_function,
        occupation,
        template.real_retarded_self_energy,
        template.width,
        template.lesser_self_energy,
        template.greater_self_energy,
        template.pole_energy,
        template.inverse_temperature,
        template.chemical_potentials,
        time,
        evidence,
        template.support_id,
        template.energy_quadrature_id,
        template.momentum_quadrature_id,
        template.frame_id,
        template.unit_contract_id,
        template.plan_id,
    )


def _select_memory(
    accepted: Array, candidate: KBMemoryState, old: KBMemoryState, /
) -> KBMemoryState:
    select = lambda new, previous: jnp.where(accepted, new, previous)
    return KBMemoryState(
        select(candidate.source_history, old.source_history),
        select(candidate.kernel_history, old.kernel_history),
        select(candidate.initial_correlation_history, old.initial_correlation_history),
        select(candidate.sample_times, old.sample_times),
        select(candidate.valid, old.valid),
        select(candidate.head, old.head),
        select(candidate.committed_samples, old.committed_samples),
        select(candidate.discarded_tail_bound, old.discarded_tail_bound),
        old.epoch_sequence,
        old.frame_token,
        old.frame_time,
        old.frame_scale_factor,
        old.frame_realization_id,
        old.parent_epoch_manifest_id,
        old.runtime_plan_id,
        old.plan_id,
    )


def advance_kadanoff_baym(
    plan: KadanoffBaymTransportPlan,
    state: QuasiparticleOffShellState,
    memory: KBMemoryState,
    runtime_state: DarkSectorEpochState,
    /,
    *,
    collision_source: ArrayLike,
    statistical_self_energy: ArrayLike,
    real_retarded_propagator: ArrayLike,
    memory_kernel: ArrayLike,
    initial_correlation: ArrayLike,
) -> KadanoffBaymStepResult:
    """Execute one rollback-safe first-gradient KB transaction."""

    if not isinstance(plan, KadanoffBaymTransportPlan):
        raise TypeError("plan must be KadanoffBaymTransportPlan.")
    if (
        not isinstance(state, QuasiparticleOffShellState)
        or state.plan_id != plan.off_shell.plan_id
    ):
        raise ValueError("Off-shell state belongs to a different KB plan.")
    if memory.plan_id != plan.plan_id:
        raise ValueError("Memory state belongs to a different KB plan.")
    if (
        not isinstance(runtime_state, DarkSectorEpochState)
        or runtime_state.plan.plan_id != plan.epoch.plan_id
    ):
        raise ValueError("KB memory and runtime epoch plans disagree.")
    checked_sequence = eqx.error_if(
        memory.epoch_sequence,
        memory.epoch_sequence != runtime_state.epoch_sequence,
        "KB memory and runtime epoch sequences disagree.",
    )
    memory = eqx.tree_at(
        lambda value: value.epoch_sequence,
        memory,
        checked_sequence,
    )
    collision = jnp.asarray(collision_source, dtype=state.occupation.dtype)
    statistical_sigma = jnp.asarray(statistical_self_energy, dtype=state.occupation.dtype)
    real_retarded = jnp.asarray(real_retarded_propagator, dtype=state.occupation.dtype)
    kernel = jnp.asarray(memory_kernel, dtype=state.occupation.dtype)
    initial = jnp.asarray(initial_correlation, dtype=state.occupation.dtype)
    expected = plan.off_shell.spectral_shape
    if any(
        value.shape != expected
        for value in (
            collision,
            statistical_sigma,
            real_retarded,
            kernel,
            initial,
        )
    ):
        raise ValueError(
            "KB collision, gradient, and memory terms must match off-shell support."
        )
    inverse_retarded_real = (
        plan.off_shell.energy_nodes[None, None, None, :]
        - state.pole_energy[None, :, :, None]
        - state.real_retarded_self_energy
    )
    drift_poisson = plan.gradient.poisson_bracket(inverse_retarded_real, state.occupation)
    backflow_poisson = plan.gradient.poisson_bracket(statistical_sigma, real_retarded)
    poisson = drift_poisson - backflow_poisson
    candidate_memory = _append_memory(
        plan,
        memory,
        collision,
        kernel,
        initial,
        state.time,
    )
    memory_evidence = kadanoff_baym_memory_evidence(plan, candidate_memory)
    increment = plan.time_step * (collision - poisson + memory_evidence.convolution)
    proposed_occupation = state.occupation + increment
    proposed = _raw_state(
        plan.off_shell,
        state,
        proposed_occupation,
        state.time + plan.time_step,
    )
    finite = proposed.evidence.valid & memory_evidence.finite & memory_evidence.causal
    accepted = finite
    accepted_occupation = jnp.where(accepted, proposed_occupation, state.occupation)
    accepted_time = jnp.where(accepted, proposed.time, state.time)
    accepted_state = _raw_state(
        plan.off_shell,
        state,
        accepted_occupation,
        accepted_time,
    )
    accepted_memory = _select_memory(accepted, candidate_memory, memory)
    rolled_back = ~accepted
    before_number = jnp.sum(state.occupation)
    after_number = jnp.sum(accepted_state.occupation)
    status = jnp.where(
        accepted, jnp.asarray(0, dtype=jnp.int32), jnp.asarray(1, dtype=jnp.int32)
    )
    evidence = KadanoffBaymStepEvidence(
        memory_evidence,
        jnp.sqrt(jnp.sum(jnp.abs(poisson) ** 2)),
        jnp.sqrt(jnp.sum(jnp.abs(drift_poisson) ** 2)),
        jnp.sqrt(jnp.sum(jnp.abs(backflow_poisson) ** 2)),
        jnp.sqrt(jnp.sum(jnp.abs(collision) ** 2)),
        jnp.sqrt(jnp.sum(jnp.abs(accepted_state.occupation - state.occupation) ** 2)),
        after_number - before_number,
        finite,
        accepted,
        rolled_back,
        status,
        plan.off_shell.frame.frame_token,
        plan.off_shell.frame.time,
        plan.off_shell.frame.scale_factor,
        plan.frame_realization_id,
        plan.profile.differentiation,
        plan.plan_id,
        plan.support_id,
        plan.frame_id,
        plan.unit_contract_id,
    )
    return KadanoffBaymStepResult(
        proposed,
        accepted_state,
        candidate_memory,
        accepted_memory,
        evidence,
        plan.plan_id,
    )


def checkpoint_kadanoff_baym(
    plan: KadanoffBaymTransportPlan,
    state: QuasiparticleOffShellState,
    memory: KBMemoryState,
    /,
    *,
    epoch_manifest_id: str,
) -> KadanoffBaymCheckpoint:
    if state.plan_id != plan.off_shell.plan_id or memory.plan_id != plan.plan_id:
        raise ValueError("Cannot checkpoint state from another Kadanoff--Baym plan.")
    manifest = _identifier(epoch_manifest_id, "epoch_manifest_id")
    checkpoint_id = canonical_fingerprint(
        {
            "kind": "kadanoff-baym-memory-checkpoint",
            "plan": plan.plan_id,
            "epoch_manifest": manifest,
            "state": array_tree_fingerprint(state),
            "memory": array_tree_fingerprint(memory),
            "frame": plan.frame_id,
            "frame_realization": plan.frame_realization_id,
            "frame_token": array_tree_fingerprint(plan.off_shell.frame.frame_token),
            "frame_time": array_tree_fingerprint(plan.off_shell.frame.time),
            "frame_scale_factor": array_tree_fingerprint(
                plan.off_shell.frame.scale_factor
            ),
            "units": plan.unit_contract_id,
        }
    )
    return KadanoffBaymCheckpoint(
        state,
        memory,
        plan.off_shell.frame.frame_token,
        plan.off_shell.frame.time,
        plan.off_shell.frame.scale_factor,
        manifest,
        checkpoint_id,
        plan.plan_id,
        plan.frame_id,
        plan.frame_realization_id,
        plan.unit_contract_id,
    )


def _checkpoint_frame_matches(
    plan: KadanoffBaymTransportPlan,
    checkpoint: KadanoffBaymCheckpoint,
    /,
) -> bool:
    frame = plan.off_shell.frame
    return (
        checkpoint.frame_id == plan.frame_id
        and checkpoint.frame_realization_id == plan.frame_realization_id
        and checkpoint.unit_contract_id == plan.unit_contract_id
        and np.array_equal(
            np.asarray(checkpoint.frame_token), np.asarray(frame.frame_token)
        )
        and np.array_equal(np.asarray(checkpoint.frame_time), np.asarray(frame.time))
        and np.array_equal(
            np.asarray(checkpoint.frame_scale_factor),
            np.asarray(frame.scale_factor),
        )
    )


def restore_kadanoff_baym(
    plan: KadanoffBaymTransportPlan,
    checkpoint: KadanoffBaymCheckpoint,
    runtime_state: DarkSectorEpochState,
    /,
) -> tuple[QuasiparticleOffShellState, KBMemoryState]:
    if (
        not isinstance(checkpoint, KadanoffBaymCheckpoint)
        or checkpoint.plan_id != plan.plan_id
    ):
        raise ValueError("Kadanoff--Baym checkpoint identity does not match this plan.")
    if not _checkpoint_frame_matches(plan, checkpoint):
        raise ValueError("Kadanoff--Baym checkpoint frame realization does not match.")
    if (
        not isinstance(runtime_state, DarkSectorEpochState)
        or runtime_state.plan.plan_id != plan.epoch.plan_id
    ):
        raise ValueError("Kadanoff--Baym restore requires the exact runtime epoch plan.")
    if runtime_state.epoch_sequence != int(np.asarray(checkpoint.memory.epoch_sequence)):
        raise ValueError("Checkpoint and runtime epoch sequences disagree.")
    expected = canonical_fingerprint(
        {
            "kind": "kadanoff-baym-memory-checkpoint",
            "plan": plan.plan_id,
            "epoch_manifest": checkpoint.epoch_manifest_id,
            "state": array_tree_fingerprint(checkpoint.state),
            "memory": array_tree_fingerprint(checkpoint.memory),
            "frame": plan.frame_id,
            "frame_realization": plan.frame_realization_id,
            "frame_token": array_tree_fingerprint(checkpoint.frame_token),
            "frame_time": array_tree_fingerprint(checkpoint.frame_time),
            "frame_scale_factor": array_tree_fingerprint(checkpoint.frame_scale_factor),
            "units": plan.unit_contract_id,
        }
    )
    if checkpoint.checkpoint_id != expected:
        raise ValueError("Kadanoff--Baym checkpoint content fingerprint is inconsistent.")
    return checkpoint.state, checkpoint.memory


def continue_kadanoff_baym_epoch(
    plan: KadanoffBaymTransportPlan,
    checkpoint: KadanoffBaymCheckpoint,
    runtime_state: DarkSectorEpochState,
    /,
) -> tuple[QuasiparticleOffShellState, KBMemoryState]:
    """Carry the finite ring into the next durable epoch without growing device state."""

    if (
        not isinstance(checkpoint, KadanoffBaymCheckpoint)
        or checkpoint.plan_id != plan.plan_id
    ):
        raise ValueError(
            "Kadanoff--Baym continuation requires an exact checkpoint identity."
        )
    if not _checkpoint_frame_matches(plan, checkpoint):
        raise ValueError("Kadanoff--Baym checkpoint frame realization does not match.")
    expected_checkpoint_id = canonical_fingerprint(
        {
            "kind": "kadanoff-baym-memory-checkpoint",
            "plan": plan.plan_id,
            "epoch_manifest": checkpoint.epoch_manifest_id,
            "state": array_tree_fingerprint(checkpoint.state),
            "memory": array_tree_fingerprint(checkpoint.memory),
            "frame": plan.frame_id,
            "frame_realization": plan.frame_realization_id,
            "frame_token": array_tree_fingerprint(checkpoint.frame_token),
            "frame_time": array_tree_fingerprint(checkpoint.frame_time),
            "frame_scale_factor": array_tree_fingerprint(checkpoint.frame_scale_factor),
            "units": plan.unit_contract_id,
        }
    )
    if checkpoint.checkpoint_id != expected_checkpoint_id:
        raise ValueError("Kadanoff--Baym checkpoint content fingerprint is inconsistent.")
    if (
        not isinstance(runtime_state, DarkSectorEpochState)
        or runtime_state.plan.plan_id != plan.epoch.plan_id
    ):
        raise ValueError(
            "Kadanoff--Baym continuation requires the exact runtime epoch plan."
        )
    next_sequence = runtime_state.epoch_sequence
    previous_sequence = int(np.asarray(checkpoint.memory.epoch_sequence))
    if next_sequence != previous_sequence + 1:
        raise ValueError(
            "Durable Kadanoff--Baym epochs must continue by one sequence number."
        )
    if runtime_state.parent_epoch_manifest_id != checkpoint.epoch_manifest_id:
        raise ValueError(
            "Runtime epoch parent does not match the Kadanoff--Baym checkpoint manifest."
        )
    old = checkpoint.memory
    continued = KBMemoryState(
        old.source_history,
        old.kernel_history,
        old.initial_correlation_history,
        old.sample_times,
        old.valid,
        old.head,
        old.committed_samples,
        old.discarded_tail_bound,
        jnp.asarray(next_sequence, dtype=jnp.int64),
        old.frame_token,
        old.frame_time,
        old.frame_scale_factor,
        old.frame_realization_id,
        runtime_state.parent_epoch_manifest_id,
        runtime_state.plan.plan_id,
        old.plan_id,
    )
    return checkpoint.state, continued


__all__ = [
    "KadanoffBaymCheckpoint",
    "KadanoffBaymProfile",
    "KadanoffBaymStepEvidence",
    "KadanoffBaymStepResult",
    "KadanoffBaymTransportPlan",
    "KBMemoryEvidence",
    "KBMemoryState",
    "WignerGradientPlan",
    "advance_kadanoff_baym",
    "checkpoint_kadanoff_baym",
    "continue_kadanoff_baym_epoch",
    "initialize_kadanoff_baym_memory",
    "kadanoff_baym_memory_evidence",
    "restore_kadanoff_baym",
]
