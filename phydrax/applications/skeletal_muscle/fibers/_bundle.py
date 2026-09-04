#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fixed-shape one-dimensional Shorten fast-twitch fiber bundles."""

from __future__ import annotations

from enum import IntFlag
from math import isfinite
from numbers import Integral

import diffrax as dfx
import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....solver import DifferentialProblem, solve_diffrax
from ..cellular import ShortenFastTwitchModel
from ._territories import MotorUnitEndplateStimulus


class SkeletalFiberBundleStatus(IntFlag):
    SUCCESS = 0
    NONFINITE = 1
    SOLVER_FAILURE = 2
    INADMISSIBLE = 4
    INVALID_STEP = 8
    STIMULUS_EVENT_CROSSED = 16


def _array(value: ArrayLike, name: str, ndim: int, /) -> Array:
    result = jnp.asarray(value)
    if result.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}.")
    if not jnp.issubdtype(result.dtype, jnp.inexact):
        result = result.astype(float)
    return result


class PrescribedFiberStimulusSchedule(StrictModule, NonTrainableState):
    """Fixed pulse capacity and source-supported node masks."""

    onset_ms: Array
    duration_ms: Array
    amplitude_uA_per_cm2: Array
    target_mask: Array
    schedule_id: str = eqx.field(static=True)

    def __init__(
        self,
        onset_ms: ArrayLike,
        duration_ms: ArrayLike,
        amplitude_uA_per_cm2: ArrayLike,
        target_mask: ArrayLike,
        /,
    ):
        onset = _array(onset_ms, "onset_ms", 1)
        duration = _array(duration_ms, "duration_ms", 1)
        amplitude = _array(amplitude_uA_per_cm2, "amplitude_uA_per_cm2", 1)
        mask = jnp.asarray(target_mask, dtype=bool)
        pulse_count = onset.shape[0]
        if duration.shape != (pulse_count,) or amplitude.shape != (pulse_count,):
            raise ValueError("Pulse time and amplitude arrays must agree in shape.")
        if mask.ndim != 3 or mask.shape[0] != pulse_count:
            raise ValueError("target_mask must have shape (pulse, fiber, node).")
        if not bool(
            np.all(np.isfinite(np.asarray(onset)))
            and np.all(np.isfinite(np.asarray(duration)))
            and np.all(np.isfinite(np.asarray(amplitude)))
            and np.all(np.asarray(duration) > 0.0)
            and np.all(np.diff(np.asarray(onset)) >= 0.0)
        ):
            raise ValueError("Pulse schedules require finite ordered onsets and duration > 0.")
        self.onset_ms = onset
        self.duration_ms = duration
        self.amplitude_uA_per_cm2 = amplitude
        self.target_mask = mask
        self.schedule_id = canonical_fingerprint(
            {
                "kind": "prescribed-skeletal-fiber-stimulus",
                "onset": array_tree_fingerprint(onset),
                "duration": array_tree_fingerprint(duration),
                "amplitude": array_tree_fingerprint(amplitude),
                "mask": array_tree_fingerprint(mask),
                "endpoint": "left-closed-right-open",
            }
        )

    @property
    def fiber_count(self) -> int:
        return self.target_mask.shape[1]

    @property
    def node_count(self) -> int:
        return self.target_mask.shape[2]

    def current(self, time_ms: ArrayLike, /) -> Array:
        time = jnp.asarray(time_ms, dtype=self.onset_ms.dtype)
        active = (time >= self.onset_ms) & (
            time < self.onset_ms + self.duration_ms
        )
        return jnp.sum(
            jnp.where(
                active[:, None, None] & self.target_mask,
                self.amplitude_uA_per_cm2[:, None, None],
                0.0,
            ),
            axis=0,
        )

    def event_boundaries_ms(self, /) -> Array:
        return jnp.sort(
            jnp.concatenate((self.onset_ms, self.onset_ms + self.duration_ms))
        )


class SkeletalFiberBundleState(StrictModule, NonTrainableState):
    time_ms: Array
    values: Array

    def __init__(self, time_ms: ArrayLike, values: ArrayLike, /):
        time = jnp.asarray(time_ms)
        state = jnp.asarray(values)
        if time.shape != ():
            raise ValueError("time_ms must be scalar.")
        if state.ndim != 3 or state.shape[-1] != 56:
            raise ValueError("Fiber state must have shape (fiber, node, 56).")
        self.time_ms = time.astype(state.dtype)
        self.values = state


class SkeletalFiberBundleOutput(StrictModule, NonTrainableState):
    membrane_potential_mV: Array
    transverse_tubule_potential_mV: Array
    cytosolic_calcium_uM: Array
    force_bearing_crossbridge_uM: Array
    stimulus_current_uA_per_cm2: Array


class SkeletalFiberBundleEvidence(StrictModule, NonTrainableState):
    status: Array
    solver_successful: Array
    finite: Array
    admissible: Array
    step_admissible: Array
    event_aligned: Array
    solver_steps: Array
    plan_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.status == int(SkeletalFiberBundleStatus.SUCCESS)


class SkeletalFiberBundleCandidate(StrictModule, NonTrainableState):
    source_state: SkeletalFiberBundleState
    candidate_state: SkeletalFiberBundleState
    output: SkeletalFiberBundleOutput
    evidence: SkeletalFiberBundleEvidence

    def commit(self, /) -> SkeletalFiberBundleState:
        accepted = self.evidence.successful
        return SkeletalFiberBundleState(
            jnp.where(
                accepted,
                self.candidate_state.time_ms,
                self.source_state.time_ms,
            ),
            jnp.where(
                accepted,
                self.candidate_state.values,
                self.source_state.values,
            ),
        )


class SkeletalFiberBundlePlan(StrictModule, NonTrainableState):
    fiber_ids: tuple[str, ...] = eqx.field(static=True)
    node_count: int = eqx.field(static=True)
    fiber_length_mm: Array
    diffusivity_mm2_per_ms: Array
    stimulus: PrescribedFiberStimulusSchedule | MotorUnitEndplateStimulus
    relative_tolerance: float = eqx.field(static=True)
    absolute_tolerance: float = eqx.field(static=True)
    initial_step_ms: float = eqx.field(static=True)
    maximum_step_ms: float = eqx.field(static=True)
    maximum_solver_steps: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        fiber_ids: tuple[str, ...],
        node_count: int,
        fiber_length_mm: ArrayLike,
        diffusivity_mm2_per_ms: ArrayLike,
        stimulus: PrescribedFiberStimulusSchedule | MotorUnitEndplateStimulus,
        /,
        *,
        relative_tolerance: float = 1.0e-6,
        absolute_tolerance: float = 1.0e-8,
        initial_step_ms: float = 1.0e-5,
        maximum_step_ms: float = 0.5,
        maximum_solver_steps: int = 131_072,
    ):
        ids = tuple(str(value).strip() for value in fiber_ids)
        if not ids or any(not value for value in ids) or len(set(ids)) != len(ids):
            raise ValueError("fiber_ids must be nonempty and unique.")
        if isinstance(node_count, bool) or not isinstance(node_count, Integral):
            raise TypeError("node_count must be an integer.")
        nodes = int(node_count)
        if nodes < 3:
            raise ValueError("Fiber bundles require at least three nodes per fiber.")
        lengths = _array(fiber_length_mm, "fiber_length_mm", 1)
        diffusion = _array(diffusivity_mm2_per_ms, "diffusivity_mm2_per_ms", 1)
        if lengths.shape != (len(ids),) or diffusion.shape != (len(ids),):
            raise ValueError("Fiber geometry/diffusivity must match fiber_ids.")
        if not bool(
            np.all(np.isfinite(np.asarray(lengths)))
            and np.all(np.asarray(lengths) > 0.0)
            and np.all(np.isfinite(np.asarray(diffusion)))
            and np.all(np.asarray(diffusion) >= 0.0)
        ):
            raise ValueError("Fiber lengths must be positive and diffusivities nonnegative.")
        if not isinstance(
            stimulus, (PrescribedFiberStimulusSchedule, MotorUnitEndplateStimulus)
        ):
            raise TypeError(
                "stimulus must be a prescribed schedule or bound motor-unit events."
            )
        if stimulus.fiber_count != len(ids) or stimulus.node_count != nodes:
            raise ValueError("Stimulus support must match fiber topology.")
        tolerances = tuple(
            float(value)
            for value in (
                relative_tolerance,
                absolute_tolerance,
                initial_step_ms,
                maximum_step_ms,
            )
        )
        if any(not isfinite(value) or value <= 0.0 for value in tolerances):
            raise ValueError("Solver tolerances and step bounds must be positive.")
        if initial_step_ms > maximum_step_ms:
            raise ValueError("initial_step_ms cannot exceed maximum_step_ms.")
        if not isinstance(maximum_solver_steps, int) or maximum_solver_steps <= 0:
            raise ValueError("maximum_solver_steps must be a positive integer.")
        self.fiber_ids = ids
        self.node_count = nodes
        self.fiber_length_mm = lengths
        self.diffusivity_mm2_per_ms = diffusion
        self.stimulus = stimulus
        self.relative_tolerance = tolerances[0]
        self.absolute_tolerance = tolerances[1]
        self.initial_step_ms = tolerances[2]
        self.maximum_step_ms = tolerances[3]
        self.maximum_solver_steps = maximum_solver_steps
        self.plan_id = canonical_fingerprint(
            {
                "kind": "shorten-2007-uniform-node-fiber-bundle",
                "fiber_ids": ids,
                "node_count": nodes,
                "lengths": array_tree_fingerprint(lengths),
                "diffusivity": array_tree_fingerprint(diffusion),
                "stimulus": stimulus.schedule_id,
                "relative_tolerance": self.relative_tolerance.hex(),
                "absolute_tolerance": self.absolute_tolerance.hex(),
                "initial_step_ms": self.initial_step_ms.hex(),
                "maximum_step_ms": self.maximum_step_ms.hex(),
                "maximum_solver_steps": maximum_solver_steps,
            }
        )

    def prepare(
        self, model: ShortenFastTwitchModel | None = None, /
    ) -> PreparedSkeletalFiberBundle:
        return PreparedSkeletalFiberBundle(
            self,
            ShortenFastTwitchModel() if model is None else model,
        )


class _FiberBundleDrift(StrictModule):
    model: ShortenFastTwitchModel
    plan: SkeletalFiberBundlePlan

    def __call__(self, time: Array, state: Array, args: object, /) -> Array:
        del args
        stimulus = self.plan.stimulus.current(time)
        reaction = self.model.rhs(
            time,
            state,
            stimulus_current_uA_per_cm2=stimulus,
        )
        potential = state[..., 0]
        spacing = self.plan.fiber_length_mm / (self.plan.node_count - 1)
        inverse_square = 1.0 / (spacing * spacing)
        interior = (
            potential[:, :-2] - 2.0 * potential[:, 1:-1] + potential[:, 2:]
        ) * inverse_square[:, None]
        left = 2.0 * (potential[:, 1] - potential[:, 0]) * inverse_square
        right = 2.0 * (potential[:, -2] - potential[:, -1]) * inverse_square
        laplacian = jnp.concatenate(
            (left[:, None], interior, right[:, None]), axis=1
        )
        diffusion = self.plan.diffusivity_mm2_per_ms[:, None] * laplacian
        return reaction.at[..., 0].add(diffusion)


class PreparedSkeletalFiberBundle(StrictModule):
    plan: SkeletalFiberBundlePlan
    model: ShortenFastTwitchModel
    solver: dfx.Kvaerno5
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self, plan: SkeletalFiberBundlePlan, model: ShortenFastTwitchModel, /
    ):
        if not isinstance(plan, SkeletalFiberBundlePlan):
            raise TypeError("plan must be SkeletalFiberBundlePlan.")
        if not isinstance(model, ShortenFastTwitchModel):
            raise TypeError("model must be ShortenFastTwitchModel.")
        self.plan = plan
        self.model = model
        self.solver = dfx.Kvaerno5()
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-skeletal-fiber-bundle",
                "plan": plan.plan_id,
                "model": model.model_id,
            }
        )

    def initialize(self, time_ms: ArrayLike = 0.0, /) -> SkeletalFiberBundleState:
        values = self.model.initialize(
            (len(self.plan.fiber_ids), self.plan.node_count)
        )
        return SkeletalFiberBundleState(time_ms, values)

    def output(self, state: SkeletalFiberBundleState, /) -> SkeletalFiberBundleOutput:
        stimulus = self.plan.stimulus.current(state.time_ms)
        evaluation = self.model.evaluate(
            state.time_ms,
            state.values,
            stimulus_current_uA_per_cm2=stimulus,
        )
        return SkeletalFiberBundleOutput(
            state.values[..., 0],
            state.values[..., 1],
            evaluation.cytosolic_calcium_uM,
            evaluation.force_bearing_crossbridge_uM,
            stimulus,
        )

    def candidate(
        self, state: SkeletalFiberBundleState, step_ms: ArrayLike, /
    ) -> SkeletalFiberBundleCandidate:
        if not isinstance(state, SkeletalFiberBundleState):
            raise TypeError("state must be SkeletalFiberBundleState.")
        expected = (len(self.plan.fiber_ids), self.plan.node_count, 56)
        if state.values.shape != expected:
            raise ValueError(f"Fiber state must have shape {expected}.")
        step = jnp.asarray(step_ms, dtype=state.values.dtype)
        if step.shape != ():
            raise ValueError("step_ms must be scalar.")
        step_valid = jnp.isfinite(step) & (step > 0.0) & (
            step <= self.plan.maximum_step_ms
        )
        safe_step = jnp.where(
            step_valid,
            step,
            jnp.asarray(self.plan.initial_step_ms, dtype=state.values.dtype),
        )
        end = state.time_ms + safe_step
        events = self.plan.stimulus.event_boundaries_ms().astype(state.values.dtype)
        event_crossed = jnp.any(
            (events > state.time_ms + 1.0e-12) & (events < end - 1.0e-12)
        )
        problem = DifferentialProblem(
            _FiberBundleDrift(self.model, self.plan),
            state.values,
            t0=state.time_ms,
            t1=end,
            problem_id=f"skeletal-fiber-bundle-step:{self.prepared_id}",
        )
        solution = solve_diffrax(
            problem,
            save_times=jnp.reshape(end, (1,)),
            solver=self.solver,
            dt0=self.plan.initial_step_ms,
            rtol=self.plan.relative_tolerance,
            atol=self.plan.absolute_tolerance,
            max_steps=self.plan.maximum_solver_steps,
            throw=False,
            solver_configuration_id="shorten-fiber-bundle-kvaerno5",
        )
        proposed_values = solution.states[-1]
        proposed = SkeletalFiberBundleState(end, proposed_values)
        output = self.output(proposed)
        finite = jnp.all(jnp.isfinite(proposed_values)) & jnp.all(
            jnp.isfinite(output.membrane_potential_mV)
        )
        admissible = jnp.all(
            self.model.admissible(
                proposed_values,
                precomputed_rates=self.model.rhs(
                    end,
                    proposed_values,
                    stimulus_current_uA_per_cm2=self.plan.stimulus.current(end),
                ),
                t0=end,
            )
        )
        solver_ok = jnp.asarray(solution.backend_successful) & jnp.all(solution.valid)
        event_aligned = ~event_crossed
        status = (
            jnp.where(finite, 0, int(SkeletalFiberBundleStatus.NONFINITE))
            | jnp.where(solver_ok, 0, int(SkeletalFiberBundleStatus.SOLVER_FAILURE))
            | jnp.where(admissible, 0, int(SkeletalFiberBundleStatus.INADMISSIBLE))
            | jnp.where(step_valid, 0, int(SkeletalFiberBundleStatus.INVALID_STEP))
            | jnp.where(
                event_aligned,
                0,
                int(SkeletalFiberBundleStatus.STIMULUS_EVENT_CROSSED),
            )
        ).astype(jnp.int32)
        evidence = SkeletalFiberBundleEvidence(
            status,
            solver_ok,
            finite,
            admissible,
            step_valid,
            event_aligned,
            jnp.asarray(solution.stats["num_steps"], dtype=jnp.int32),
            self.plan.plan_id,
        )
        return SkeletalFiberBundleCandidate(state, proposed, output, evidence)


__all__ = [
    "PrescribedFiberStimulusSchedule",
    "PreparedSkeletalFiberBundle",
    "SkeletalFiberBundleCandidate",
    "SkeletalFiberBundleEvidence",
    "SkeletalFiberBundleOutput",
    "SkeletalFiberBundlePlan",
    "SkeletalFiberBundleState",
    "SkeletalFiberBundleStatus",
]
