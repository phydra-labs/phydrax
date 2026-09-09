#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Source-bound local reactions and moving, no-flux tridiagonal cable diffusion."""

from __future__ import annotations

from enum import IntFlag
from math import isfinite

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._identity import NumericRevision, SemanticProvenance, strict_module_payload
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....linalg import solve_tridiagonal_lines
from ....solver import DifferentialProblem, solve_diffrax
from ._bundle import PrescribedFiberStimulusSchedule
from ._reaction import AbstractFiberReaction
from ._territories import MotorUnitEndplateStimulus


class StructuredFiberResponseStatus(IntFlag):
    SUCCESS = 0
    NONFINITE = 1
    REACTION_FAILURE = 2
    DIFFUSION_FAILURE = 4
    INADMISSIBLE = 8
    INVALID_STEP = 16
    UNALIGNED_EVENT = 32
    INVALID_GEOMETRY = 64
    SOURCE_GEOMETRY_MISMATCH = 128


def _real_array(value: ArrayLike, /, *, dtype=None) -> Array:
    result = jnp.asarray(value)
    if jnp.issubdtype(result.dtype, jnp.complexfloating):
        raise TypeError("Fiber geometry, times and coefficients must be real.")
    if dtype is not None:
        return result.astype(dtype)
    return result if jnp.issubdtype(result.dtype, jnp.floating) else result.astype(float)


def _dual_lengths(segment_lengths: Array, /) -> Array:
    zero = jnp.zeros_like(segment_lengths[..., :1])
    return 0.5 * (
        jnp.concatenate((zero, segment_lengths), axis=-1)
        + jnp.concatenate((segment_lengths, zero), axis=-1)
    )


class MovingFiberGeometry1D(StrictModule, NonTrainableState):
    """Current arclength metric of a material polyline in a fixed world frame.

    Dual lengths are lumped linear-element masses. Conductance is the spatial
    diffusivity divided by current segment length. No area/dilution law or
    mechanical force is inferred from these purely one-dimensional metrics.
    """

    node_positions_mm: Array
    segment_lengths_mm: Array
    dual_lengths_mm: Array
    segment_jacobian: Array
    node_jacobian: Array
    segment_conductance_mm_per_ms: Array
    valid: Array


def _geometry(
    positions: Array,
    reference_segments: Array,
    diffusivity: Array,
    minimum_segment_length_mm: float,
    /,
) -> MovingFiberGeometry1D:
    segments = jnp.sqrt(jnp.sum(jnp.diff(positions, axis=-2) ** 2, axis=-1))
    dual = _dual_lengths(segments)
    valid = jnp.all(jnp.isfinite(positions)) & jnp.all(
        segments > minimum_segment_length_mm
    )
    safe_segments = jnp.where(segments > minimum_segment_length_mm, segments, 1.0)
    return MovingFiberGeometry1D(
        positions,
        segments,
        dual,
        segments / reference_segments,
        dual / _dual_lengths(reference_segments),
        diffusivity / safe_segments,
        valid,
    )


def _path_nondegenerate(path: Array, minimum_length: float, /) -> Array:
    # The minimum of |a + theta*b| on [0,1] detects an interior collapse even
    # when both supplied endpoint geometries have positive segment lengths.
    segments = jnp.diff(path, axis=-2)
    a = segments[:-1]
    b = segments[1:] - a
    squared_rate = jnp.sum(b * b, axis=-1)
    theta = jnp.clip(
        -jnp.sum(a * b, axis=-1) / jnp.where(squared_rate > 0.0, squared_rate, 1.0),
        0.0,
        1.0,
    )
    minimum_squared = jnp.sum((a + theta[..., None] * b) ** 2, axis=-1)
    return jnp.all(jnp.isfinite(path)) & jnp.all(minimum_squared > minimum_length**2)


class StructuredFiberResponseState(StrictModule, NonTrainableState):
    time_ms: Array
    values: Array
    node_positions_mm: Array
    accepted_steps: Array
    prepared_id: str = eqx.field(static=True)


class StructuredFiberResponseEvidence(StrictModule, NonTrainableState):
    status: Array
    reaction_successful: Array
    diffusion_successful: Array
    admissible: Array
    finite: Array
    event_aligned: Array
    geometry_valid: Array
    source_geometry_matches: Array
    reaction_solver_steps: Array
    diffusion_relative_residual: Array
    diffusion_balance_error: Array
    minimum_diffusion_pivot: Array
    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.status == int(StructuredFiberResponseStatus.SUCCESS)


class StructuredFiberResponseCandidate(StrictModule, NonTrainableState):
    source_state: StructuredFiberResponseState
    candidate_state: StructuredFiberResponseState
    geometry: MovingFiberGeometry1D
    geometry_path_mm: Array
    evidence: StructuredFiberResponseEvidence

    def commit(
        self, current_state: StructuredFiberResponseState, /
    ) -> StructuredFiberResponseState:
        """Compare the whole source snapshot, then select all fields atomically.

        A stale proposal returns the supplied current state, never its old source
        snapshot. The comparison includes geometry, values, time and counter.
        """
        if current_state.prepared_id != self.evidence.prepared_id:
            raise ValueError("Fiber candidate belongs to a different prepared response.")
        source = self.source_state
        current_leaves = jax.tree.leaves(current_state)
        source_leaves = jax.tree.leaves(source)
        if any(
            a.shape != b.shape for a, b in zip(current_leaves, source_leaves, strict=True)
        ):
            raise ValueError("Current fiber state does not match the candidate topology.")
        matches = jnp.all(
            jnp.stack(
                tuple(
                    jnp.array_equal(a, b)
                    for a, b in zip(current_leaves, source_leaves, strict=True)
                )
            )
        )
        accepted = self.evidence.successful & matches
        return jax.tree.map(
            lambda proposed, current: jnp.where(accepted, proposed, current),
            self.candidate_state,
            current_state,
        )


class StructuredFiberResponsePlan(StrictModule, NonTrainableState):
    """Fixed substeps and geometry semantics, distinct from the dense oracle.

    The numerical identity is Strang local Kvaerno5 / midpoint-geometry
    Crank–Nicolson diffusion / local Kvaerno5. Events must coincide with supplied
    substep boundaries. There is no dense, explicit or smaller-step fallback.
    """

    fiber_ids: tuple[str, ...] = eqx.field(static=True)
    node_count: int = eqx.field(static=True)
    reference_node_positions_mm: Array
    reference_segment_lengths_mm: Array
    substep_abscissae: Array
    stimulus: PrescribedFiberStimulusSchedule | MotorUnitEndplateStimulus
    geometry_source_id: str = eqx.field(static=True)
    relative_tolerance: float = eqx.field(static=True)
    absolute_tolerance: float = eqx.field(static=True)
    initial_reaction_step_ms: float = eqx.field(static=True)
    maximum_step_ms: float = eqx.field(static=True)
    maximum_reaction_steps: int = eqx.field(static=True)
    diffusion_tolerance: float = eqx.field(static=True)
    pivot_tolerance: float = eqx.field(static=True)
    minimum_segment_length_mm: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        fiber_ids: tuple[str, ...],
        reference_node_positions_mm: ArrayLike,
        stimulus: PrescribedFiberStimulusSchedule | MotorUnitEndplateStimulus,
        substep_abscissae: ArrayLike,
        /,
        *,
        geometry_source_id: str,
        relative_tolerance: float = 1.0e-6,
        absolute_tolerance: float = 1.0e-8,
        initial_reaction_step_ms: float = 1.0e-5,
        maximum_step_ms: float = 0.5,
        maximum_reaction_steps: int = 4096,
        diffusion_tolerance: float = 1.0e-8,
        pivot_tolerance: float = 1.0e-14,
        minimum_segment_length_mm: float = 1.0e-10,
    ):
        ids = tuple(str(value).strip() for value in fiber_ids)
        if not ids or any(not value for value in ids) or len(set(ids)) != len(ids):
            raise ValueError("fiber_ids must be nonempty and unique.")
        positions = _real_array(reference_node_positions_mm)
        if (
            positions.ndim != 3
            or positions.shape[0] != len(ids)
            or positions.shape[-1] != 3
            or positions.shape[1] < 3
        ):
            raise ValueError("Reference positions must have shape (fiber, node>=3, 3).")
        theta = _real_array(substep_abscissae, dtype=positions.dtype)
        if theta.ndim != 1 or theta.size < 2:
            raise ValueError(
                "substep_abscissae must be a vector with at least two entries."
            )
        host_theta = np.asarray(theta)
        if (
            not np.all(np.isfinite(host_theta))
            or host_theta[0] != 0.0
            or host_theta[-1] != 1.0
            or not np.all(np.diff(host_theta) > 0.0)
        ):
            raise ValueError("Substep abscissae must increase strictly from zero to one.")
        if not isinstance(
            stimulus, (PrescribedFiberStimulusSchedule, MotorUnitEndplateStimulus)
        ):
            raise TypeError(
                "stimulus must be an explicit fiber schedule or bound events."
            )
        if stimulus.fiber_count != len(ids) or stimulus.node_count != positions.shape[1]:
            raise ValueError("Stimulus support must match fiber/node topology.")
        if not isinstance(geometry_source_id, str) or not geometry_source_id.strip():
            raise ValueError(
                "geometry_source_id must identify the caller's geometry/frame asset."
            )
        policy = {
            "relative_tolerance": float(relative_tolerance),
            "absolute_tolerance": float(absolute_tolerance),
            "initial_reaction_step_ms": float(initial_reaction_step_ms),
            "maximum_step_ms": float(maximum_step_ms),
            "diffusion_tolerance": float(diffusion_tolerance),
            "pivot_tolerance": float(pivot_tolerance),
            "minimum_segment_length_mm": float(minimum_segment_length_mm),
        }
        if any(not isfinite(value) or value <= 0.0 for value in policy.values()):
            raise ValueError(
                "Numerical tolerances and step/geometry bounds must be positive."
            )
        if initial_reaction_step_ms > maximum_step_ms:
            raise ValueError(
                "Initial reaction step cannot exceed the candidate step bound."
            )
        if (
            isinstance(maximum_reaction_steps, bool)
            or not isinstance(maximum_reaction_steps, int)
            or maximum_reaction_steps <= 0
        ):
            raise ValueError("maximum_reaction_steps must be a positive integer.")
        segments = jnp.sqrt(jnp.sum(jnp.diff(positions, axis=1) ** 2, axis=-1))
        if not np.all(np.isfinite(np.asarray(positions))) or not np.all(
            np.asarray(segments) > minimum_segment_length_mm
        ):
            raise ValueError(
                "Reference geometry must be finite with nondegenerate segments."
            )
        self.fiber_ids = ids
        self.node_count = positions.shape[1]
        self.reference_node_positions_mm = positions
        self.reference_segment_lengths_mm = segments
        self.substep_abscissae = theta
        self.stimulus = stimulus
        self.geometry_source_id = geometry_source_id.strip()
        self.relative_tolerance = policy["relative_tolerance"]
        self.absolute_tolerance = policy["absolute_tolerance"]
        self.initial_reaction_step_ms = policy["initial_reaction_step_ms"]
        self.maximum_step_ms = policy["maximum_step_ms"]
        self.maximum_reaction_steps = maximum_reaction_steps
        self.diffusion_tolerance = policy["diffusion_tolerance"]
        self.pivot_tolerance = policy["pivot_tolerance"]
        self.minimum_segment_length_mm = policy["minimum_segment_length_mm"]
        self.plan_id = canonical_fingerprint(
            {
                "kind": "structured-source-bound-fiber-response",
                "fiber_ids": ids,
                "geometry_source": self.geometry_source_id,
                "reference_positions": array_tree_fingerprint(positions),
                "substep_abscissae": array_tree_fingerprint(theta),
                "stimulus": stimulus.schedule_id,
                "policy": policy,
                "maximum_reaction_steps": maximum_reaction_steps,
                "reaction": "independent-local-kvaerno5-direct-adjoint",
                "split": "strang-reaction-diffusion-reaction",
                "diffusion": "no-flux-lumped-linear-fem-crank-nicolson-native-tridiagonal",
                "metric": "midpoint-current-arclength-spatial-diffusivity",
                "geometry_path": "piecewise-affine-world-node-positions-mm",
                "kinematic_path": "piecewise-affine-dual-length-stretch-endpoint-rate-per-ms",
                "voltage_transport": "material-derivative-no-dilution",
                "event_support": "left-closed-right-open-fixed-substep-boundaries",
            }
        )

    @property
    def substep_count(self) -> int:
        return self.substep_abscissae.size - 1

    def prepare(
        self, reaction: AbstractFiberReaction, diffusivity_mm2_per_ms: ArrayLike, /
    ) -> PreparedStructuredFiberResponse:
        return PreparedStructuredFiberResponse(self, reaction, diffusivity_mm2_per_ms)


class _LocalReactionDrift(StrictModule):
    reaction: AbstractFiberReaction
    current: Array
    stretch_start: Array
    stretch_rate: Array
    start_ms: Array

    def __call__(self, time: Array, values: Array, args: object, /) -> Array:
        del args
        stretch = self.stretch_start + (time - self.start_ms) * self.stretch_rate
        return self.reaction.rhs(time, values, self.current, stretch, self.stretch_rate)


class PreparedStructuredFiberResponse(StrictModule):
    plan: StructuredFiberResponsePlan
    reaction: AbstractFiberReaction
    diffusivity_mm2_per_ms: Array
    semantic_id: str = eqx.field(static=True)
    numeric_revision_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan, reaction, diffusivity_mm2_per_ms, /):
        if not isinstance(plan, StructuredFiberResponsePlan):
            raise TypeError("plan must be StructuredFiberResponsePlan.")
        if not isinstance(reaction, AbstractFiberReaction):
            raise TypeError(
                "reaction must be an explicitly source-bound AbstractFiberReaction."
            )
        if (
            not reaction.source_id
            or reaction.state_count < 1
            or not 0 <= reaction.voltage_index < reaction.state_count
        ):
            raise ValueError(
                "Reaction source identity and voltage/state layout must be valid."
            )
        initial = reaction.initialize(())
        if initial.shape != (reaction.state_count,) or not jnp.issubdtype(
            initial.dtype, jnp.floating
        ):
            raise ValueError(
                "Reaction initial state must be a real vector matching its layout."
            )
        diffusion = _real_array(diffusivity_mm2_per_ms)
        dtype = jnp.result_type(
            diffusion.dtype, initial.dtype, plan.reference_node_positions_mm.dtype
        )
        diffusion = diffusion.astype(dtype)
        expected = (len(plan.fiber_ids), plan.node_count - 1)
        if diffusion.shape != expected:
            raise ValueError(f"Spatial segment diffusivity must have shape {expected}.")
        if not np.all(np.isfinite(np.asarray(diffusion))) or not np.all(
            np.asarray(diffusion) >= 0.0
        ):
            raise ValueError("Spatial diffusivity must be finite and nonnegative.")
        binding = strict_module_payload(reaction)
        semantic = SemanticProvenance(
            {
                "kind": "prepared-structured-fiber-response",
                "plan": plan.plan_id,
                "reaction": binding["semantic_content_id"],
                "diffusivity_units": "mm2/ms-current-segment",
            }
        )
        numeric = NumericRevision(
            semantic,
            {
                "reaction": reaction,
                "diffusivity": diffusion,
            },
        )
        self.plan = plan
        self.reaction = reaction
        self.diffusivity_mm2_per_ms = diffusion
        self.semantic_id = semantic.semantic_id
        self.numeric_revision_id = numeric.revision_id
        self.prepared_id = canonical_fingerprint(
            {
                "semantic": semantic.semantic_id,
                "numeric": numeric.revision_id,
            }
        )

    def geometry(self, node_positions_mm: ArrayLike, /) -> MovingFiberGeometry1D:
        positions = _real_array(
            node_positions_mm, dtype=self.diffusivity_mm2_per_ms.dtype
        )
        if positions.shape != self.plan.reference_node_positions_mm.shape:
            raise ValueError("Current node positions must match the reference topology.")
        return _geometry(
            positions,
            self.plan.reference_segment_lengths_mm,
            self.diffusivity_mm2_per_ms,
            self.plan.minimum_segment_length_mm,
        )

    def initialize(self, time_ms: ArrayLike = 0.0, /) -> StructuredFiberResponseState:
        values = self.reaction.initialize(
            (len(self.plan.fiber_ids), self.plan.node_count)
        )
        if values.shape != (
            len(self.plan.fiber_ids),
            self.plan.node_count,
            self.reaction.state_count,
        ):
            raise ValueError(
                "Reaction initialization does not match its declared state layout."
            )
        values = values.astype(self.diffusivity_mm2_per_ms.dtype)
        time = _real_array(time_ms, dtype=values.dtype)
        if time.shape != ():
            raise ValueError("time_ms must be scalar.")
        return StructuredFiberResponseState(
            time,
            values,
            self.plan.reference_node_positions_mm.astype(values.dtype),
            jnp.asarray(0, dtype=jnp.int32),
            self.prepared_id,
        )

    def linear_geometry_path(
        self, state: StructuredFiberResponseState, end_node_positions_mm: ArrayLike, /
    ) -> Array:
        """Construct the declared fixed-grid affine path; same endpoint is stationary."""
        end = _real_array(end_node_positions_mm, dtype=state.node_positions_mm.dtype)
        if end.shape != state.node_positions_mm.shape:
            raise ValueError("Endpoint geometry must match the source node topology.")
        theta = self.plan.substep_abscissae[:, None, None, None]
        return (
            state.node_positions_mm[None] + theta * (end - state.node_positions_mm)[None]
        )

    def _reaction_step(self, values, start, end, full_start, current, stretch, rate, /):
        def solve_local(y, injected, initial_stretch, stretch_rate):
            problem = DifferentialProblem(
                _LocalReactionDrift(
                    self.reaction, injected, initial_stretch, stretch_rate, full_start
                ),
                y,
                t0=start,
                t1=end,
                problem_id=f"structured-fiber-local-reaction:{self.prepared_id}",
            )
            solution = solve_diffrax(
                problem,
                save_times=end[None],
                solver=dfx.Kvaerno5(),
                adjoint=dfx.DirectAdjoint(),
                dt0=jnp.minimum(self.plan.initial_reaction_step_ms, end - start),
                rtol=self.plan.relative_tolerance,
                atol=self.plan.absolute_tolerance,
                max_steps=self.plan.maximum_reaction_steps,
                throw=False,
                solver_configuration_id="structured-fiber-local-kvaerno5-direct-adjoint",
            )
            value = solution.states[-1]
            admissible = self.reaction.admissible(
                end,
                value,
                injected,
                initial_stretch + (end - full_start) * stretch_rate,
                stretch_rate,
            )
            return (
                value,
                solution.backend_successful & jnp.all(solution.valid),
                admissible,
                solution.stats["num_steps"],
            )

        result, successful, admissible, steps = jax.vmap(solve_local)(
            values.reshape((-1, self.reaction.state_count)),
            current.reshape(-1),
            stretch.reshape(-1),
            rate.reshape(-1),
        )
        return (
            result.reshape(values.shape),
            jnp.all(successful),
            jnp.all(admissible),
            jnp.sum(steps, dtype=jnp.int32),
        )

    def _diffusion_step(self, voltage, geometry, step_ms, /):
        conductance = geometry.segment_conductance_mm_per_ms
        zero = jnp.zeros_like(conductance[:, :1])
        left = jnp.concatenate((zero, conductance), axis=1)
        right = jnp.concatenate((conductance, zero), axis=1)
        mass = geometry.dual_lengths_mm
        half = 0.5 * step_ms
        flux = conductance * jnp.diff(voltage, axis=1)
        action = jnp.concatenate((flux, zero), axis=1) - jnp.concatenate(
            (zero, flux), axis=1
        )
        rhs = mass * voltage + half * action
        solved = solve_tridiagonal_lines(
            -half * left,
            mass + half * (left + right),
            -half * right,
            rhs,
            pivot_tolerance=self.plan.pivot_tolerance,
        )
        relative = solved.residual_norm / jnp.maximum(1.0, jnp.sqrt(jnp.sum(rhs**2)))
        balance = jnp.max(jnp.abs(jnp.sum(mass * (solved.value - voltage), axis=1)))
        balance_scale = jnp.maximum(
            1.0, jnp.max(jnp.sum(jnp.abs(mass * voltage), axis=1))
        )
        successful = (
            solved.successful
            & (relative <= self.plan.diffusion_tolerance)
            & (balance <= self.plan.diffusion_tolerance * balance_scale)
        )
        return solved.value, successful, relative, balance, solved.minimum_pivot

    def candidate(
        self,
        state: StructuredFiberResponseState,
        step_ms: ArrayLike,
        geometry_path_mm: ArrayLike,
        /,
    ) -> StructuredFiberResponseCandidate:
        if not isinstance(state, StructuredFiberResponseState):
            raise TypeError("state must be StructuredFiberResponseState.")
        if state.prepared_id != self.prepared_id:
            raise ValueError("Fiber state belongs to a different prepared response.")
        expected = (
            len(self.plan.fiber_ids),
            self.plan.node_count,
            self.reaction.state_count,
        )
        if (
            state.values.shape != expected
            or state.node_positions_mm.shape
            != self.plan.reference_node_positions_mm.shape
            or state.time_ms.shape != ()
            or state.accepted_steps.shape != ()
        ):
            raise ValueError("Fiber state does not match the prepared topology.")
        if not jnp.issubdtype(state.accepted_steps.dtype, jnp.integer):
            raise TypeError("accepted_steps must have integer dtype.")
        if any(
            value.dtype != self.diffusivity_mm2_per_ms.dtype
            for value in (state.values, state.time_ms, state.node_positions_mm)
        ):
            raise TypeError(
                "Fiber state real dtype must match the prepared numerical realization."
            )
        path = _real_array(geometry_path_mm, dtype=state.node_positions_mm.dtype)
        if path.shape != (self.plan.substep_count + 1,) + state.node_positions_mm.shape:
            raise ValueError(
                "geometry_path_mm must include every fixed substep endpoint."
            )
        step = _real_array(step_ms, dtype=state.time_ms.dtype)
        if step.shape != ():
            raise ValueError("step_ms must be scalar.")
        step_valid = (
            jnp.isfinite(step) & (step > 0.0) & (step <= self.plan.maximum_step_ms)
        )
        # Invalid requests never run an alternative solve. Safe times only keep
        # traced, unexecuted branch arithmetic defined for the rejection path.
        safe_step = jnp.where(step_valid, step, self.plan.initial_reaction_step_ms)
        times = state.time_ms + self.plan.substep_abscissae * safe_step
        midpoints = times[:-1] + 0.5 * jnp.diff(times)
        step_valid = (
            step_valid
            & jnp.all(jnp.isfinite(times))
            & jnp.all((midpoints > times[:-1]) & (midpoints < times[1:]))
        )
        events = self.plan.stimulus.event_boundaries_ms().astype(times.dtype)
        event_aligned = ~jnp.any(
            (events[:, None] > times[:-1]) & (events[:, None] < times[1:])
        )
        source_matches = jnp.array_equal(path[0], state.node_positions_mm)
        geometry_valid = _path_nondegenerate(path, self.plan.minimum_segment_length_mm)
        source_finite = jnp.isfinite(state.time_ms) & jnp.all(jnp.isfinite(state.values))
        coefficient_valid = jnp.all(jnp.isfinite(self.diffusivity_mm2_per_ms)) & jnp.all(
            self.diffusivity_mm2_per_ms >= 0.0
        )
        valid_counter = (state.accepted_steps >= 0) & (
            state.accepted_steps < jnp.iinfo(state.accepted_steps.dtype).max
        )
        initial_geometry = self.geometry(state.node_positions_mm)
        initial_current = self.plan.stimulus.current(state.time_ms)
        first_duration = times[1] - times[0]
        initial_rate = (
            self.geometry(path[1]).node_jacobian - initial_geometry.node_jacobian
        ) / jnp.where(first_duration > 0.0, first_duration, 1.0)
        initial_admissible = jnp.all(
            jax.vmap(self.reaction.admissible, in_axes=(None, 0, 0, 0, 0))(
                state.time_ms,
                state.values.reshape((-1, self.reaction.state_count)),
                initial_current.reshape(-1),
                initial_geometry.node_jacobian.reshape(-1),
                initial_rate.reshape(-1),
            )
        )
        can_solve = (
            step_valid
            & valid_counter
            & source_finite
            & coefficient_valid
            & initial_admissible
            & event_aligned
            & geometry_valid
            & source_matches
        )

        def advance(_):
            def substep(values, inputs):
                start, end, positions_start, positions_end = inputs
                duration = end - start
                midpoint = start + 0.5 * duration
                g0 = self.geometry(positions_start)
                g1 = self.geometry(positions_end)
                gm = self.geometry(0.5 * (positions_start + positions_end))
                stretch = g0.node_jacobian
                rate = (g1.node_jacobian - stretch) / duration
                current = self.plan.stimulus.current(midpoint)
                reacted, ok0, admissible0, steps0 = self._reaction_step(
                    values, start, midpoint, start, current, stretch, rate
                )
                voltage, diffusion_ok, residual, balance, pivot = self._diffusion_step(
                    reacted[..., self.reaction.voltage_index], gm, duration
                )
                diffused = reacted.at[..., self.reaction.voltage_index].set(voltage)
                result, ok1, admissible1, steps1 = self._reaction_step(
                    diffused, midpoint, end, start, current, stretch, rate
                )
                evidence = (
                    ok0 & ok1,
                    diffusion_ok,
                    admissible0 & admissible1,
                    steps0 + steps1,
                    residual,
                    balance,
                    pivot,
                )
                return result, evidence

            return jax.lax.scan(
                substep, state.values, (times[:-1], times[1:], path[:-1], path[1:])
            )

        def reject(_):
            shape = (self.plan.substep_count,)
            false = jnp.zeros(shape, dtype=bool)
            zero = jnp.zeros(shape, dtype=state.values.dtype)
            return state.values, (
                false,
                false,
                false,
                jnp.zeros(shape, dtype=jnp.int32),
                zero,
                zero,
                zero,
            )

        (
            values,
            (reaction_ok, diffusion_ok, admissible, steps, residual, balance, pivot),
        ) = jax.lax.cond(can_solve, advance, reject, operand=None)
        finite = source_finite & jnp.all(jnp.isfinite(values))
        source_supported = coefficient_valid & initial_admissible & jnp.all(admissible)
        status = (
            jnp.where(finite, 0, int(StructuredFiberResponseStatus.NONFINITE))
            | jnp.where(
                jnp.all(reaction_ok),
                0,
                int(StructuredFiberResponseStatus.REACTION_FAILURE),
            )
            | jnp.where(
                jnp.all(diffusion_ok),
                0,
                int(StructuredFiberResponseStatus.DIFFUSION_FAILURE),
            )
            | jnp.where(
                source_supported, 0, int(StructuredFiberResponseStatus.INADMISSIBLE)
            )
            | jnp.where(
                step_valid & valid_counter,
                0,
                int(StructuredFiberResponseStatus.INVALID_STEP),
            )
            | jnp.where(
                event_aligned, 0, int(StructuredFiberResponseStatus.UNALIGNED_EVENT)
            )
            | jnp.where(
                geometry_valid, 0, int(StructuredFiberResponseStatus.INVALID_GEOMETRY)
            )
            | jnp.where(
                source_matches,
                0,
                int(StructuredFiberResponseStatus.SOURCE_GEOMETRY_MISMATCH),
            )
        ).astype(jnp.int32)
        evidence = StructuredFiberResponseEvidence(
            status,
            reaction_ok,
            diffusion_ok,
            source_supported,
            finite,
            event_aligned,
            geometry_valid,
            source_matches,
            steps,
            residual,
            balance,
            pivot,
            self.plan.plan_id,
            self.prepared_id,
            self.reaction.source_id,
        )
        proposed = StructuredFiberResponseState(
            state.time_ms + step,
            values,
            path[-1],
            state.accepted_steps + jnp.asarray(1, dtype=state.accepted_steps.dtype),
            self.prepared_id,
        )
        return StructuredFiberResponseCandidate(
            state, proposed, self.geometry(path[-1]), path, evidence
        )

    def commit(self, state, candidate, /) -> StructuredFiberResponseState:
        if candidate.evidence.prepared_id != self.prepared_id:
            raise ValueError("Fiber candidate belongs to a different prepared response.")
        return candidate.commit(state)


__all__ = [
    "MovingFiberGeometry1D",
    "PreparedStructuredFiberResponse",
    "StructuredFiberResponseCandidate",
    "StructuredFiberResponseEvidence",
    "StructuredFiberResponsePlan",
    "StructuredFiberResponseState",
    "StructuredFiberResponseStatus",
]
