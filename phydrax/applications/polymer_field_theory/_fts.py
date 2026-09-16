#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, ArrayLike, Key

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..sign_problem import (
    ComplexLangevinPlan,
    ComplexLangevinResult,
    prepare_complex_langevin,
    PreparedComplexLangevin,
    sample_complex_langevin,
)
from ._scft import PreparedSCFT


class PartialSaddleFTSPlan(StrictModule, NonTrainableState):
    num_steps: int = eqx.field(static=True)
    step_size: float = eqx.field(static=True)
    mobility: float = eqx.field(static=True)
    pressure_iterations: int = eqx.field(static=True)
    pressure_damping: float = eqx.field(static=True)
    burn_in: int = eqx.field(static=True)
    thinning: int = eqx.field(static=True)
    maximum_incompressibility: float = eqx.field(static=True)
    realization_id: int = eqx.field(static=True)
    maximum_state_size: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        num_steps: int,
        step_size: float,
        mobility: float = 1.0,
        pressure_iterations: int = 4,
        pressure_damping: float = 0.2,
        burn_in: int = 0,
        thinning: int = 1,
        maximum_incompressibility: float = 1.0e-4,
        realization_id: int = 0,
        maximum_state_size: int = 1_000_000,
    ):
        steps = int(num_steps)
        step = float(step_size)
        mobility_ = float(mobility)
        pressure = int(pressure_iterations)
        damping = float(pressure_damping)
        burn = int(burn_in)
        thin = int(thinning)
        incompressibility = float(maximum_incompressibility)
        realization = int(realization_id)
        maximum = int(maximum_state_size)
        if (
            steps <= 0
            or not math.isfinite(step)
            or step <= 0.0
            or not math.isfinite(mobility_)
            or mobility_ <= 0.0
            or pressure <= 0
            or not math.isfinite(damping)
            or not 0.0 < damping <= 1.0
            or burn < 0
            or burn >= steps
            or thin <= 0
            or not math.isfinite(incompressibility)
            or incompressibility <= 0.0
            or realization < 0
            or maximum <= 0
        ):
            raise ValueError("Partial-saddle FTS controls are invalid.")
        self.num_steps = steps
        self.step_size = step
        self.mobility = mobility_
        self.pressure_iterations = pressure
        self.pressure_damping = damping
        self.burn_in = burn
        self.thinning = thin
        self.maximum_incompressibility = incompressibility
        self.realization_id = realization
        self.maximum_state_size = maximum
        self.plan_id = canonical_fingerprint(
            {
                "kind": "partial-saddle-fts-plan",
                "num_steps": steps,
                "step_size": step,
                "mobility": mobility_,
                "pressure_iterations": pressure,
                "pressure_damping": damping,
                "burn_in": burn,
                "thinning": thin,
                "maximum_incompressibility": incompressibility,
                "realization_id": realization,
                "maximum_state_size": maximum,
            }
        )

    @property
    def output_count(self) -> int:
        return len(range(self.burn_in, self.num_steps, self.thinning))

    def prepare(self, scft: PreparedSCFT, /) -> "PreparedPartialSaddleFTS":
        return PreparedPartialSaddleFTS(self, scft)


class PreparedPartialSaddleFTS(StrictModule, NonTrainableState):
    plan: PartialSaddleFTSPlan
    scft: PreparedSCFT
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: PartialSaddleFTSPlan, scft: PreparedSCFT, /):
        if not isinstance(plan, PartialSaddleFTSPlan):
            raise TypeError("plan must be PartialSaddleFTSPlan.")
        if not isinstance(scft, PreparedSCFT):
            raise TypeError("scft must be PreparedSCFT.")
        if int(np.prod(scft.field_shape)) > plan.maximum_state_size:
            raise ValueError("Partial-saddle FTS state exceeds maximum_state_size.")
        self.plan = plan
        self.scft = scft
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-partial-saddle-fts",
                "plan": plan.plan_id,
                "scft": scft.prepared_id,
            }
        )


class PartialSaddleFTSState(StrictModule):
    fields: Array
    step_index: Array
    key_data: Array
    successful: Array
    prepared_id: str = eqx.field(static=True)


class PartialSaddleFTSResult(StrictModule):
    samples: Array
    free_energies: Array
    incompressibility_norms: Array
    final_state: PartialSaddleFTSState
    successful: Array
    prepared_id: str = eqx.field(static=True)
    claim: str = eqx.field(static=True)


def initialize_partial_saddle_fts(
    prepared: PreparedPartialSaddleFTS,
    fields: ArrayLike,
    /,
    *,
    key: Key[Array, ""],
) -> PartialSaddleFTSState:
    if not isinstance(prepared, PreparedPartialSaddleFTS):
        raise TypeError("prepared must be PreparedPartialSaddleFTS.")
    value = prepared.scft.project_initial_fields(fields)
    if value.shape != prepared.scft.field_shape:
        raise ValueError("Partial-saddle initial fields have the wrong shape.")
    evaluation = prepared.scft.evaluate(value)
    return PartialSaddleFTSState(
        value,
        jnp.zeros((), dtype=jnp.int32),
        jr.key_data(key),
        evaluation.successful,
        prepared.prepared_id,
    )


def sample_partial_saddle_fts(
    prepared: PreparedPartialSaddleFTS,
    initial_state: PartialSaddleFTSState,
    /,
) -> PartialSaddleFTSResult:
    if not isinstance(prepared, PreparedPartialSaddleFTS):
        raise TypeError("prepared must be PreparedPartialSaddleFTS.")
    if not isinstance(initial_state, PartialSaddleFTSState):
        raise TypeError("initial_state must be PartialSaddleFTSState.")
    if initial_state.prepared_id != prepared.prepared_id:
        raise ValueError("Partial-saddle state belongs to another prepared runtime.")
    current = initial_state.fields
    active = initial_state.successful
    root_key = jr.wrap_key_data(initial_state.key_data)
    samples = []
    free_energies = []
    incompressibility_norms = []
    for local_step in range(prepared.plan.num_steps):
        pressure_relaxed = current
        for _ in range(prepared.plan.pressure_iterations):
            pressure_evaluation = prepared.scft.evaluate(pressure_relaxed)
            pressure_residual = pressure_evaluation.residual[..., -1]
            pressure_relaxed = pressure_relaxed + (
                prepared.plan.pressure_damping * pressure_residual[..., None]
            )
            pressure_relaxed = pressure_relaxed - jnp.mean(pressure_relaxed)
        evaluation = prepared.scft.evaluate(pressure_relaxed)
        drift = evaluation.residual - jnp.mean(
            evaluation.residual, axis=-1, keepdims=True
        )
        step_index = initial_state.step_index + local_step
        noise_key = jr.fold_in(root_key, prepared.plan.realization_id)
        noise_key = jr.fold_in(noise_key, step_index)
        noise = jr.normal(noise_key, prepared.scft.field_shape, dtype=current.dtype)
        noise = noise - jnp.mean(noise, axis=-1, keepdims=True)
        candidate = (
            pressure_relaxed
            - prepared.plan.mobility * prepared.plan.step_size * drift
            + jnp.sqrt(2.0 * prepared.plan.mobility * prepared.plan.step_size) * noise
        )
        candidate = candidate - jnp.mean(candidate)
        candidate_evaluation = prepared.scft.evaluate(candidate)
        finite = candidate_evaluation.successful & jnp.all(jnp.isfinite(candidate))
        commit = active & finite
        current = jnp.where(commit, candidate, current)
        active = active & finite
        if local_step >= prepared.plan.burn_in and (
            (local_step - prepared.plan.burn_in) % prepared.plan.thinning == 0
        ):
            accepted_evaluation = prepared.scft.evaluate(current)
            samples.append(current)
            free_energies.append(accepted_evaluation.free_energy)
            incompressibility_norms.append(
                jnp.max(jnp.abs(accepted_evaluation.incompressibility_residual))
            )
    sample_array = jnp.stack(samples)
    free_energy_array = jnp.stack(free_energies)
    incompressibility_array = jnp.stack(incompressibility_norms)
    successful = (
        active
        & jnp.all(jnp.isfinite(sample_array))
        & jnp.all(jnp.isfinite(free_energy_array))
        & jnp.all(incompressibility_array <= prepared.plan.maximum_incompressibility)
    )
    final_state = PartialSaddleFTSState(
        current,
        initial_state.step_index + prepared.plan.num_steps,
        initial_state.key_data,
        successful,
        prepared.prepared_id,
    )
    return PartialSaddleFTSResult(
        sample_array,
        free_energy_array,
        incompressibility_array,
        final_state,
        successful,
        prepared.prepared_id,
        "real-exchange-field/conditional-incompressibility-saddle",
    )


class ComplexFTSPlan(StrictModule, NonTrainableState):
    langevin: ComplexLangevinPlan
    plan_id: str = eqx.field(static=True)

    def __init__(self, langevin: ComplexLangevinPlan, /):
        if not isinstance(langevin, ComplexLangevinPlan):
            raise TypeError("langevin must be ComplexLangevinPlan.")
        self.langevin = langevin
        self.plan_id = canonical_fingerprint(
            {"kind": "complex-fts-plan", "langevin": langevin.plan_id}
        )

    def prepare(self, scft: PreparedSCFT, /) -> "PreparedComplexFTS":
        return PreparedComplexFTS(self, scft)


class PreparedComplexFTS(StrictModule, NonTrainableState):
    plan: ComplexFTSPlan
    scft: PreparedSCFT
    langevin: PreparedComplexLangevin
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: ComplexFTSPlan, scft: PreparedSCFT, /):
        if not isinstance(plan, ComplexFTSPlan):
            raise TypeError("plan must be ComplexFTSPlan.")
        if not isinstance(scft, PreparedSCFT):
            raise TypeError("scft must be PreparedSCFT.")

        def action(fields):
            gauge_fixed = fields - jnp.mean(fields)
            value = scft.evaluate(gauge_fixed).free_energy
            return jnp.asarray(value, dtype=fields.dtype)

        langevin = prepare_complex_langevin(
            plan.langevin,
            action,
            configuration_shape=scft.field_shape,
            action_id=f"polymer-fts:{scft.prepared_id}",
        )
        self.plan = plan
        self.scft = scft
        self.langevin = langevin
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-complex-fts",
                "plan": plan.plan_id,
                "scft": scft.prepared_id,
                "langevin": langevin.prepared_id,
            }
        )


class ComplexFTSResult(StrictModule):
    trajectory: ComplexLangevinResult
    gauge_residuals: Array
    successful: Array
    prepared_id: str = eqx.field(static=True)


def sample_complex_fts(
    prepared: PreparedComplexFTS,
    initial_fields: ArrayLike,
    /,
    *,
    key: Key[Array, ""],
) -> ComplexFTSResult:
    if not isinstance(prepared, PreparedComplexFTS):
        raise TypeError("prepared must be PreparedComplexFTS.")
    fields = jnp.asarray(initial_fields)
    if fields.shape != prepared.scft.field_shape or not jnp.iscomplexobj(fields):
        raise TypeError(
            "Complex FTS initial_fields must match the SCFT field shape and be complex."
        )
    gauge_fixed = fields - jnp.mean(fields)
    trajectory = sample_complex_langevin(prepared.langevin, gauge_fixed, key=key)
    gauge_residuals = jnp.abs(
        jnp.mean(trajectory.samples, axis=tuple(range(1, trajectory.samples.ndim)))
    )
    successful = trajectory.successful & jnp.all(jnp.isfinite(gauge_residuals))
    return ComplexFTSResult(trajectory, gauge_residuals, successful, prepared.prepared_id)


__all__ = [
    "ComplexFTSPlan",
    "ComplexFTSResult",
    "PartialSaddleFTSPlan",
    "PartialSaddleFTSResult",
    "PartialSaddleFTSState",
    "PreparedComplexFTS",
    "PreparedPartialSaddleFTS",
    "initialize_partial_saddle_fts",
    "sample_complex_fts",
    "sample_partial_saddle_fts",
]
