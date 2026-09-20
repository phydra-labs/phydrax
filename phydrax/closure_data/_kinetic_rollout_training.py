#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from jaxtyping import Array, Key

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._model import AbstractArrayModel, model_structure_recipe
from .._numerics._checkpointed_scan import (
    checkpointed_scan,
    CheckpointedScanMode,
    PreparedReplaySchedule,
)
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.discrete_velocity._smooth_compressible import (
    SmoothCompressibleKineticState,
)
from ..discretization.discrete_velocity._spatial import (
    PreparedSmoothCompressibleD2V17SpatialDynamics,
)
from ._kinetic_equilibrium import LearnedEnergyEquilibriumBindingPlan
from ._kinetic_rollout import (
    PreparedSmoothCompressibleRolloutDataset,
    SmoothCompressibleRolloutStatistics,
    SmoothCompressibleRolloutWindow,
)


KineticRolloutReplayMode: TypeAlias = CheckpointedScanMode
KineticRolloutTermination: TypeAlias = Literal["maximum_attempts", "curriculum_complete"]
_DEFAULT_CURRICULUM = (1, 2, 4, 8, 16, 25)


def _positive_integer(value: int, role: str, /) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{role} must be an integer.")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{role} must be positive.")
    return result


def _nonnegative_finite(value: float, role: str, /) -> float:
    result = float(value)
    if not isfinite(result) or result < 0.0:
        raise ValueError(f"{role} must be finite and non-negative.")
    return result


def _model_structure_id(model: AbstractArrayModel, /) -> str:
    if not isinstance(model, AbstractArrayModel):
        raise TypeError("model must be an AbstractArrayModel.")
    return canonical_fingerprint(
        {
            "kind": "kinetic-rollout-model-structure",
            "recipe": model_structure_recipe(model),
        }
    )


def _tree_finite(tree: Any, /) -> Array:
    leaves = tuple(value for value in jax.tree.leaves(tree) if eqx.is_array(value))
    if not leaves:
        return jnp.asarray(True)
    return jnp.all(jnp.stack(tuple(jnp.all(jnp.isfinite(value)) for value in leaves)))


class KineticRolloutTrainingPlan(StrictModule, NonTrainableState):
    """Static curriculum, optimizer, replay, and physical-runtime contract."""

    dynamics: PreparedSmoothCompressibleD2V17SpatialDynamics
    binding_plan: LearnedEnergyEquilibriumBindingPlan
    curriculum_horizons: tuple[int, ...] = eqx.field(static=True)
    accepted_updates_per_horizon: tuple[int, ...] = eqx.field(static=True)
    batch_size: int = eqx.field(static=True)
    guard_batch_size: int = eqx.field(static=True)
    learning_rate: float = eqx.field(static=True)
    macro_weight: float = eqx.field(static=True)
    energy_population_weight: float = eqx.field(static=True)
    equilibrium_weight: float = eqx.field(static=True)
    maximum_scaled_flux_error: float = eqx.field(static=True)
    replay_mode: KineticRolloutReplayMode = eqx.field(static=True)
    replay_block_size: int | None = eqx.field(static=True)
    replay_schedules: tuple[PreparedReplaySchedule, ...]
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        dynamics: PreparedSmoothCompressibleD2V17SpatialDynamics,
        binding_plan: LearnedEnergyEquilibriumBindingPlan,
        /,
        *,
        accepted_updates_per_horizon: int | tuple[int, ...],
        batch_size: int,
        guard_batch_size: int,
        learning_rate: float,
        macro_weight: float = 1.0,
        energy_population_weight: float = 1.0,
        equilibrium_weight: float = 0.1,
        maximum_scaled_flux_error: float = 6.0e-2,
        curriculum_horizons: tuple[int, ...] = _DEFAULT_CURRICULUM,
        replay_mode: KineticRolloutReplayMode = "step",
        replay_block_size: int | None = None,
        replay_schedules: tuple[PreparedReplaySchedule, ...] = (),
    ):
        if not isinstance(dynamics, PreparedSmoothCompressibleD2V17SpatialDynamics):
            raise TypeError(
                "dynamics must be PreparedSmoothCompressibleD2V17SpatialDynamics."
            )
        if not isinstance(binding_plan, LearnedEnergyEquilibriumBindingPlan):
            raise TypeError("binding_plan must be LearnedEnergyEquilibriumBindingPlan.")
        horizons = tuple(
            _positive_integer(value, "curriculum horizon")
            for value in curriculum_horizons
        )
        if horizons != _DEFAULT_CURRICULUM:
            raise ValueError("The rollout curriculum must be exactly 1→2→4→8→16→25.")
        if isinstance(accepted_updates_per_horizon, (int, np.integer)) and not isinstance(
            accepted_updates_per_horizon, (bool, np.bool_)
        ):
            updates = (
                _positive_integer(accepted_updates_per_horizon, "accepted updates"),
            ) * len(horizons)
        else:
            updates = tuple(
                _positive_integer(value, "accepted updates")
                for value in accepted_updates_per_horizon
            )
            if len(updates) != len(horizons):
                raise ValueError(
                    "accepted_updates_per_horizon must align with the curriculum."
                )
        batch = _positive_integer(batch_size, "batch_size")
        guard_batch = _positive_integer(guard_batch_size, "guard_batch_size")
        rate = float(learning_rate)
        if not isfinite(rate) or rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive.")
        weights = tuple(
            _nonnegative_finite(value, role)
            for value, role in (
                (macro_weight, "macro_weight"),
                (energy_population_weight, "energy_population_weight"),
                (equilibrium_weight, "equilibrium_weight"),
            )
        )
        if not any(value > 0.0 for value in weights):
            raise ValueError("At least one rollout loss weight must be positive.")
        flux_gate = _nonnegative_finite(
            maximum_scaled_flux_error, "maximum_scaled_flux_error"
        )
        if flux_gate <= 0.0:
            raise ValueError("maximum_scaled_flux_error must be positive.")
        if replay_mode not in ("full", "step", "block", "scheduled"):
            raise ValueError("Unknown kinetic-rollout replay mode.")
        block = (
            None
            if replay_block_size is None
            else _positive_integer(replay_block_size, "replay_block_size")
        )
        schedules = tuple(replay_schedules)
        if replay_mode == "block":
            if block is None or schedules:
                raise ValueError("Block replay requires only replay_block_size.")
        elif replay_mode == "scheduled":
            if block is not None or len(schedules) != len(horizons):
                raise ValueError(
                    "Scheduled replay requires one prepared schedule per horizon."
                )
            if any(
                not isinstance(schedule, PreparedReplaySchedule)
                or schedule.step_count != horizon
                for schedule, horizon in zip(schedules, horizons, strict=True)
            ):
                raise ValueError(
                    "Prepared replay schedules must match curriculum horizons."
                )
        elif block is not None or schedules:
            raise ValueError("Full and step replay accept no block or schedule.")
        if (
            dynamics.method.quadrature.quadrature_id
            != binding_plan.equilibrium_plan.quadrature.quadrature_id
            or dynamics.energy_plan.plan_id != binding_plan.equilibrium_plan.plan_id
            or dynamics.method.material.material_id != binding_plan.material.material_id
        ):
            raise ValueError(
                "Dynamics and learned-equilibrium binding physical identities differ."
            )
        self.dynamics = dynamics
        self.binding_plan = binding_plan
        self.curriculum_horizons = horizons
        self.accepted_updates_per_horizon = updates
        self.batch_size = batch
        self.guard_batch_size = guard_batch
        self.learning_rate = rate
        self.macro_weight, self.energy_population_weight, self.equilibrium_weight = (
            weights
        )
        self.maximum_scaled_flux_error = flux_gate
        self.replay_mode = replay_mode
        self.replay_block_size = block
        self.replay_schedules = schedules
        self.plan_id = canonical_fingerprint(
            {
                "kind": "kinetic-rollout-training-plan",
                "dynamics": dynamics.prepared_id,
                "binding_plan": binding_plan.plan_id,
                "curriculum_horizons": list(horizons),
                "accepted_updates_per_horizon": list(updates),
                "batch_size": batch,
                "guard_batch_size": guard_batch,
                "learning_rate": rate,
                "loss_weights": list(weights),
                "maximum_scaled_flux_error": flux_gate,
                "replay_mode": replay_mode,
                "replay_block_size": block,
                "replay_schedules": [value.schedule_id for value in schedules],
            }
        )

    def optimizer(self, /) -> optax.GradientTransformation:
        return optax.adam(self.learning_rate)

    def scan_arguments(self, horizon: int, /) -> dict[str, Any]:
        horizon_ = _positive_integer(horizon, "horizon")
        if horizon_ not in self.curriculum_horizons:
            raise ValueError("The horizon is not a curriculum stage.")
        if self.replay_mode == "block":
            return {"mode": "block", "block_size": self.replay_block_size}
        if self.replay_mode == "scheduled":
            index = self.curriculum_horizons.index(horizon_)
            return {"mode": "scheduled", "schedule": self.replay_schedules[index]}
        return {"mode": self.replay_mode}


class KineticRolloutObjectiveEvidence(StrictModule):
    """Per-trajectory objective and compact hard-gate diagnostics."""

    total_loss: Array
    macro_loss: Array
    energy_population_loss: Array
    equilibrium_loss: Array
    successful: Array
    maximum_energy_residual: Array
    maximum_particle_stress_residual: Array
    maximum_flux_residual: Array
    maximum_scaled_flux_error: Array
    maximum_conservation_residual: Array
    minimum_particle_population: Array
    minimum_energy_population: Array
    minimum_support_margin: Array
    horizon: int = eqx.field(static=True)


class _KineticRolloutScanCarry(StrictModule):
    particle_populations: Array
    total_energy_populations: Array
    successful: Array
    macro_loss: Array
    energy_population_loss: Array
    equilibrium_loss: Array
    maximum_energy_residual: Array
    maximum_particle_stress_residual: Array
    maximum_flux_residual: Array
    maximum_scaled_flux_error: Array
    maximum_conservation_residual: Array
    minimum_particle_population: Array
    minimum_energy_population: Array
    minimum_support_margin: Array


def _minimum_support_margin(evidence: Any, /) -> Array:
    return jnp.min(
        jnp.stack(
            (
                evidence.rho_margin,
                evidence.u_x_margin,
                evidence.u_y_margin,
                evidence.temperature_margin,
                evidence.mach_margin,
                evidence.hull_margin,
                evidence.particle_equilibrium_margin,
            ),
            axis=0,
        )
    )


def _single_trajectory_objective(
    model: AbstractArrayModel,
    plan: KineticRolloutTrainingPlan,
    statistics: SmoothCompressibleRolloutStatistics,
    f_initial: Array,
    g_initial: Array,
    g_targets: Array,
    U_targets: Array,
    horizon: int,
    /,
) -> tuple[Array, tuple[Array, ...]]:
    dtype = g_initial.dtype
    initial = _KineticRolloutScanCarry(
        particle_populations=f_initial,
        total_energy_populations=g_initial,
        successful=jnp.asarray(True),
        macro_loss=jnp.asarray(0.0, dtype=dtype),
        energy_population_loss=jnp.asarray(0.0, dtype=dtype),
        equilibrium_loss=jnp.asarray(0.0, dtype=dtype),
        maximum_energy_residual=jnp.asarray(0.0, dtype=dtype),
        maximum_particle_stress_residual=jnp.asarray(0.0, dtype=dtype),
        maximum_flux_residual=jnp.asarray(0.0, dtype=dtype),
        maximum_scaled_flux_error=jnp.asarray(0.0, dtype=dtype),
        maximum_conservation_residual=jnp.asarray(0.0, dtype=dtype),
        minimum_particle_population=jnp.asarray(jnp.inf, dtype=dtype),
        minimum_energy_population=jnp.asarray(jnp.inf, dtype=dtype),
        minimum_support_margin=jnp.asarray(jnp.inf, dtype=dtype),
    )
    dt = jnp.asarray(plan.dynamics.required_step_size, dtype=dtype)
    U_scale = jnp.asarray(statistics.U_scale, dtype=dtype)
    g_scale = jnp.asarray(statistics.g_scale, dtype=dtype)

    def step(carry: _KineticRolloutScanCarry, targets: tuple[Array, Array]):
        g_target, U_target = targets
        state = SmoothCompressibleKineticState(
            carry.particle_populations, carry.total_energy_populations
        )
        moments = plan.dynamics.method.moments(state)
        target_flux = (moments.total_energy + moments.pressure)[
            ..., None
        ] * moments.velocity
        oracle = plan.dynamics.energy_plan.solve(moments.total_energy, target_flux)
        result, support = plan.dynamics.step_with_model(
            state, dt, model, plan.binding_plan
        )
        next_state = result.accepted_state
        predicted_U = plan.dynamics.method.moments(next_state).conserved
        macro_increment = jnp.mean(jnp.square((predicted_U - U_target) / U_scale))
        energy_increment = jnp.mean(
            jnp.square((next_state.total_energy_populations - g_target) / g_scale)
        )
        equilibrium_increment = jnp.mean(
            jnp.square(
                (
                    result.evidence.equilibrium.energy.recovered_flux
                    - jax.lax.stop_gradient(oracle.evidence.recovered_flux)
                )
                / jnp.maximum(
                    jnp.abs(moments.total_energy[..., None]), jnp.finfo(dtype).tiny
                )
            )
        )

        equilibrium = result.evidence.equilibrium
        energy = equilibrium.energy
        energy_scale = jnp.maximum(jnp.max(jnp.abs(energy.target_total_energy)), 1.0)
        energy_tolerance = (
            jnp.asarray(plan.dynamics.energy_plan.residual_tolerance, dtype=dtype)
            + 256.0 * jnp.finfo(dtype).eps * energy_scale
        )
        particle_stress_scale = jnp.maximum(
            jnp.max(jnp.abs(equilibrium.target_particle_momentum_flux)), 1.0
        )
        particle_stress_tolerance = 256.0 * jnp.finfo(dtype).eps * particle_stress_scale
        conservation_scale = jnp.maximum(
            jnp.max(jnp.abs(result.evidence.conservation.pre_step_content)), 1.0
        )
        conservation_tolerance = (
            jnp.asarray(plan.dynamics.conservation_tolerance, dtype=dtype)
            + 512.0 * jnp.finfo(dtype).eps * conservation_scale
        )
        energy_residual = jnp.max(jnp.abs(energy.total_energy_residual))
        particle_stress_residual = (
            equilibrium.maximum_absolute_particle_momentum_flux_residual
        )
        flux_residual = jnp.max(energy.flux_error_norm)
        scaled_flux_error = jnp.max(
            energy.flux_error_norm
            / jnp.maximum(jnp.abs(energy.target_total_energy), jnp.finfo(dtype).tiny)
        )
        conservation_residual = result.evidence.conservation.maximum_absolute_residual
        particle_minimum = jnp.minimum(
            jnp.min(equilibrium.minimum_particle_equilibrium_population),
            jnp.min(
                result.evidence.post_transport_realizability.minimum_particle_population
            ),
        )
        energy_minimum = jnp.minimum(
            jnp.min(energy.minimum_population),
            jnp.min(
                result.evidence.post_transport_realizability.minimum_total_energy_population
            ),
        )
        support_margin = _minimum_support_margin(support)
        physical = (
            jnp.all(support.successful)
            & jnp.all(oracle.evidence.successful & oracle.evidence.converged)
            & jnp.all(result.successful)
            & jnp.all(energy.successful)
            & (energy_residual <= energy_tolerance)
            & (particle_stress_residual <= particle_stress_tolerance)
            & (scaled_flux_error <= plan.maximum_scaled_flux_error)
            & (conservation_residual <= conservation_tolerance)
            & (particle_minimum >= plan.dynamics.population_floor)
            & (energy_minimum >= plan.dynamics.population_floor)
        )
        return (
            _KineticRolloutScanCarry(
                particle_populations=next_state.particle_populations,
                total_energy_populations=next_state.total_energy_populations,
                successful=carry.successful & physical,
                macro_loss=carry.macro_loss + macro_increment,
                energy_population_loss=(carry.energy_population_loss + energy_increment),
                equilibrium_loss=carry.equilibrium_loss + equilibrium_increment,
                maximum_energy_residual=jnp.maximum(
                    carry.maximum_energy_residual, energy_residual
                ),
                maximum_particle_stress_residual=jnp.maximum(
                    carry.maximum_particle_stress_residual,
                    particle_stress_residual,
                ),
                maximum_flux_residual=jnp.maximum(
                    carry.maximum_flux_residual, flux_residual
                ),
                maximum_scaled_flux_error=jnp.maximum(
                    carry.maximum_scaled_flux_error, scaled_flux_error
                ),
                maximum_conservation_residual=jnp.maximum(
                    carry.maximum_conservation_residual,
                    conservation_residual,
                ),
                minimum_particle_population=jnp.minimum(
                    carry.minimum_particle_population, particle_minimum
                ),
                minimum_energy_population=jnp.minimum(
                    carry.minimum_energy_population, energy_minimum
                ),
                minimum_support_margin=jnp.minimum(
                    carry.minimum_support_margin, support_margin
                ),
            ),
            None,
        )

    final, _ = checkpointed_scan(
        step,
        initial,
        (g_targets, U_targets),
        length=horizon,
        **plan.scan_arguments(horizon),
    )
    denominator = jnp.asarray(horizon, dtype=dtype)
    macro = final.macro_loss / denominator
    energy_population = final.energy_population_loss / denominator
    equilibrium = final.equilibrium_loss / denominator
    total = (
        plan.macro_weight * macro
        + plan.energy_population_weight * energy_population
        + plan.equilibrium_weight * equilibrium
    )
    return total, (
        macro,
        energy_population,
        equilibrium,
        final.successful,
        final.maximum_energy_residual,
        final.maximum_particle_stress_residual,
        final.maximum_flux_residual,
        final.maximum_scaled_flux_error,
        final.maximum_conservation_residual,
        final.minimum_particle_population,
        final.minimum_energy_population,
        final.minimum_support_margin,
    )


def _window_arrays(
    windows: tuple[SmoothCompressibleRolloutWindow, ...],
    horizon: int,
    /,
) -> tuple[Array, Array, Array, Array]:
    values = tuple(windows)
    if not values or any(
        not isinstance(value, SmoothCompressibleRolloutWindow) for value in values
    ):
        raise ValueError("A rollout objective requires prepared rollout windows.")
    horizon_ = _positive_integer(horizon, "horizon")
    if any(value.f_targets.shape[0] < horizon_ for value in values):
        raise ValueError("Every rollout window must contain the complete horizon.")
    identities = {
        (value.schema_id, value.window_plan_id, value.split) for value in values
    }
    shapes = {
        (
            value.f_history.shape,
            value.g_history.shape,
            value.f_targets.shape,
            value.g_targets.shape,
            value.U_targets.shape,
        )
        for value in values
    }
    if len(identities) != 1 or len(shapes) != 1:
        raise ValueError(
            "Vmapped rollout windows must have one schema and shape contract."
        )
    return (
        jnp.stack(tuple(value.f_history[-1] for value in values), axis=0),
        jnp.stack(tuple(value.g_history[-1] for value in values), axis=0),
        jnp.stack(tuple(value.g_targets[:horizon_] for value in values), axis=0),
        jnp.stack(tuple(value.U_targets[:horizon_] for value in values), axis=0),
    )


def kinetic_rollout_objective(
    model: AbstractArrayModel,
    plan: KineticRolloutTrainingPlan,
    statistics: SmoothCompressibleRolloutStatistics,
    windows: tuple[SmoothCompressibleRolloutWindow, ...],
    horizon: int,
    /,
) -> tuple[Array, KineticRolloutObjectiveEvidence]:
    """Vmap complete atomic trajectory scans without materializing state histories."""

    if (
        not isinstance(model, AbstractArrayModel)
        or model.in_size != 4
        or model.out_size != 2
    ):
        raise TypeError("The rollout model must implement the explicit 4→2 array ABI.")
    if not isinstance(plan, KineticRolloutTrainingPlan):
        raise TypeError("plan must be KineticRolloutTrainingPlan.")
    if not isinstance(statistics, SmoothCompressibleRolloutStatistics):
        raise TypeError("statistics must be SmoothCompressibleRolloutStatistics.")
    horizon_ = _positive_integer(horizon, "horizon")
    if horizon_ not in plan.curriculum_horizons:
        raise ValueError("The objective horizon must be a curriculum stage.")
    values = tuple(windows)
    if any(value.schema_id != statistics.schema_id for value in values):
        raise ValueError("Rollout windows and training statistics schemas differ.")
    f_initial, g_initial, g_targets, U_targets = _window_arrays(values, horizon_)
    total, compact = jax.vmap(
        lambda f0, g0, gt, Ut: _single_trajectory_objective(
            model,
            plan,
            statistics,
            f0,
            g0,
            gt,
            Ut,
            horizon_,
        )
    )(f_initial, g_initial, g_targets, U_targets)
    (
        macro,
        energy_population,
        equilibrium,
        successful,
        maximum_energy,
        maximum_particle_stress,
        maximum_flux,
        maximum_scaled_flux,
        maximum_conservation,
        minimum_particle,
        minimum_energy,
        minimum_support,
    ) = compact
    return jnp.mean(total), KineticRolloutObjectiveEvidence(
        total_loss=total,
        macro_loss=macro,
        energy_population_loss=energy_population,
        equilibrium_loss=equilibrium,
        successful=successful,
        maximum_energy_residual=maximum_energy,
        maximum_particle_stress_residual=maximum_particle_stress,
        maximum_flux_residual=maximum_flux,
        maximum_scaled_flux_error=maximum_scaled_flux,
        maximum_conservation_residual=maximum_conservation,
        minimum_particle_population=minimum_particle,
        minimum_energy_population=minimum_energy,
        minimum_support_margin=minimum_support,
        horizon=horizon_,
    )


def _validate_dataset(
    plan: KineticRolloutTrainingPlan,
    dataset: PreparedSmoothCompressibleRolloutDataset,
    /,
) -> None:
    if not isinstance(dataset, PreparedSmoothCompressibleRolloutDataset):
        raise TypeError("dataset must be PreparedSmoothCompressibleRolloutDataset.")
    if dataset.window_plan.horizon_steps < plan.curriculum_horizons[-1]:
        raise ValueError(
            "The prepared rollout dataset must contain the full 25-step horizon."
        )
    schema = dataset.trajectories[0].schema
    binding = plan.binding_plan
    dynamics = plan.dynamics
    expected_parent = binding.parent_artifact_id
    if expected_parent is None:
        raise ValueError("Stage-two training requires an identified stage-one artifact.")
    expected_boundary = None if dynamics.boundary is None else dynamics.boundary.plan_id
    expected_force = None if dynamics.forcing is None else dynamics.forcing.plan_id
    if (
        schema.runtime_id != dynamics.prepared_id
        or schema.topology_id != dynamics.transport.plan_id
        or schema.method_id != dynamics.method.method_id
        or schema.quadrature_id != dynamics.method.quadrature.quadrature_id
        or schema.material_id != dynamics.method.material.material_id
        or schema.oracle_id != dynamics.energy_plan.plan_id
        or schema.stage_one_artifact_id != expected_parent
        or schema.support_id != binding.support.support_id
        or schema.boundary_schedule_id != expected_boundary
        or schema.force_schedule_id != expected_force
        or schema.dt != dynamics.required_step_size
        or dataset.schema_id != schema.schema_id
        or binding.training_preparation_id != dataset.preparation_id
    ):
        raise ValueError(
            "Rollout dataset, stage-one binding, support, and physical runtime identity mismatch."
        )


class KineticRolloutTrainingState(StrictModule):
    """One exact transactional optimizer boundary and curriculum cursor."""

    model: AbstractArrayModel
    optimizer_state: Any
    best_model: AbstractArrayModel
    key: Array
    best_loss: Array
    attempt_count: int = eqx.field(static=True)
    accepted_update_count: int = eqx.field(static=True)
    rejection_count: int = eqx.field(static=True)
    curriculum_index: int = eqx.field(static=True)
    accepted_in_curriculum: int = eqx.field(static=True)
    training_cursor: int = eqx.field(static=True)
    guard_cursor: int = eqx.field(static=True)
    last_update_accepted: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    dataset_id: str = eqx.field(static=True)
    model_structure_id: str = eqx.field(static=True)
    state_id: str = eqx.field(static=True)

    def __init__(
        self,
        model: AbstractArrayModel,
        optimizer_state: Any,
        best_model: AbstractArrayModel,
        key: Array,
        best_loss: Array,
        /,
        *,
        attempt_count: int,
        accepted_update_count: int,
        rejection_count: int,
        curriculum_index: int,
        accepted_in_curriculum: int,
        training_cursor: int,
        guard_cursor: int,
        last_update_accepted: bool,
        plan_id: str,
        dataset_id: str,
        model_structure_id: str,
    ):
        if not isinstance(model, AbstractArrayModel) or not isinstance(
            best_model, AbstractArrayModel
        ):
            raise TypeError("Training state model snapshots must be array models.")
        structure = _model_structure_id(model)
        if (
            structure != _model_structure_id(best_model)
            or structure != model_structure_id
        ):
            raise ValueError("Current and best model structures must match exactly.")
        key_ = jnp.asarray(key)
        if key_.shape != (2,) or key_.dtype != jnp.uint32:
            raise ValueError("Training checkpoint keys use exact uint32[2] key data.")
        best_loss_ = jnp.asarray(best_loss)
        if best_loss_.shape != () or not jnp.issubdtype(best_loss_.dtype, jnp.floating):
            raise ValueError("best_loss must be one floating scalar array.")
        counters = tuple(
            (
                attempt_count,
                accepted_update_count,
                rejection_count,
                curriculum_index,
                accepted_in_curriculum,
                training_cursor,
                guard_cursor,
            )
        )
        if any(value < 0 for value in counters):
            raise ValueError("Training progress and cursors must be non-negative.")
        if counters[1] + counters[2] != counters[0]:
            raise ValueError(
                "Every optimizer attempt must be accepted or rejected exactly once."
            )
        if not isinstance(last_update_accepted, bool):
            raise TypeError("last_update_accepted must be boolean.")
        plan_identity = str(plan_id).strip()
        dataset_identity = str(dataset_id).strip()
        if not plan_identity or not dataset_identity:
            raise ValueError("Training state identities must be non-empty.")
        self.model = model
        self.optimizer_state = optimizer_state
        self.best_model = best_model
        self.key = key_
        self.best_loss = best_loss_
        (
            self.attempt_count,
            self.accepted_update_count,
            self.rejection_count,
            self.curriculum_index,
            self.accepted_in_curriculum,
            self.training_cursor,
            self.guard_cursor,
        ) = counters
        self.last_update_accepted = last_update_accepted
        self.plan_id = plan_identity
        self.dataset_id = dataset_identity
        self.model_structure_id = structure
        self.state_id = canonical_fingerprint(
            {
                "kind": "kinetic-rollout-training-state",
                "plan": plan_identity,
                "dataset": dataset_identity,
                "model_structure": structure,
                "progress": {
                    "attempt_count": self.attempt_count,
                    "accepted_update_count": self.accepted_update_count,
                    "rejection_count": self.rejection_count,
                    "curriculum_index": self.curriculum_index,
                    "accepted_in_curriculum": self.accepted_in_curriculum,
                    "training_cursor": self.training_cursor,
                    "guard_cursor": self.guard_cursor,
                    "last_update_accepted": last_update_accepted,
                },
                "arrays": array_tree_fingerprint(
                    {
                        "model": model,
                        "optimizer_state": optimizer_state,
                        "best_model": best_model,
                        "key": key_,
                        "best_loss": best_loss_,
                    }
                ),
            }
        )


class KineticRolloutUpdateResult(StrictModule):
    state: KineticRolloutTrainingState
    training_evidence: KineticRolloutObjectiveEvidence
    guard_evidence: KineticRolloutObjectiveEvidence
    selection_evidence: KineticRolloutObjectiveEvidence
    training_loss: Array
    guard_loss: Array
    selection_loss: Array
    gradient_finite: Array
    proposal_finite: Array
    accepted: bool = eqx.field(static=True)
    horizon: int = eqx.field(static=True)


class KineticRolloutTrainingResult(StrictModule):
    state: KineticRolloutTrainingState
    attempts: tuple[KineticRolloutUpdateResult, ...]
    termination: KineticRolloutTermination = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        state: KineticRolloutTrainingState,
        attempts: tuple[KineticRolloutUpdateResult, ...],
        termination: KineticRolloutTermination,
        /,
    ):
        if not isinstance(state, KineticRolloutTrainingState):
            raise TypeError("state must be KineticRolloutTrainingState.")
        attempts_ = tuple(attempts)
        if any(not isinstance(value, KineticRolloutUpdateResult) for value in attempts_):
            raise TypeError("attempts must contain KineticRolloutUpdateResult values.")
        if termination not in ("maximum_attempts", "curriculum_complete"):
            raise ValueError("Unknown kinetic-rollout termination.")
        self.state = state
        self.attempts = attempts_
        self.termination = termination
        self.result_id = canonical_fingerprint(
            {
                "kind": "kinetic-rollout-training-result",
                "state": state.state_id,
                "attempt_state_ids": [value.state.state_id for value in attempts_],
                "termination": termination,
            }
        )


def _cyclic_batch(
    windows: tuple[SmoothCompressibleRolloutWindow, ...], size: int, cursor: int, /
) -> tuple[tuple[SmoothCompressibleRolloutWindow, ...], int]:
    count = len(windows)
    if count == 0:
        raise ValueError("A deterministic rollout batch cannot be empty.")
    selected_count = min(size, count)
    start = cursor % count
    indices = tuple((start + offset) % count for offset in range(selected_count))
    return tuple(windows[index] for index in indices), (start + selected_count) % count


def _guard_windows(
    dataset: PreparedSmoothCompressibleRolloutDataset, /
) -> tuple[SmoothCompressibleRolloutWindow, ...]:
    if dataset.validation_windows:
        return dataset.validation_windows
    return dataset.train_windows


def initialize_kinetic_rollout_training(
    model: AbstractArrayModel,
    plan: KineticRolloutTrainingPlan,
    dataset: PreparedSmoothCompressibleRolloutDataset,
    /,
    *,
    key: Key[Array, ""],
) -> KineticRolloutTrainingState:
    """Initialize an exact optimizer boundary and immutable baseline selection."""

    if (
        not isinstance(model, AbstractArrayModel)
        or model.in_size != 4
        or model.out_size != 2
    ):
        raise TypeError("Stage-two training requires an explicit 4→2 array model.")
    if not isinstance(plan, KineticRolloutTrainingPlan):
        raise TypeError("plan must be KineticRolloutTrainingPlan.")
    _validate_dataset(plan, dataset)
    raw_key = jnp.asarray(jr.key_data(key), dtype=jnp.uint32)
    if raw_key.shape != (2,):
        raise ValueError("Stage-two training requires one scalar PRNG key.")
    trainable = eqx.filter(model, eqx.is_inexact_array)
    optimizer_state = plan.optimizer().init(trainable)
    selection_windows = _guard_windows(dataset)
    initial_loss, evidence = kinetic_rollout_objective(
        model,
        plan,
        dataset.statistics,
        selection_windows,
        plan.curriculum_horizons[-1],
    )
    successful = bool(
        np.asarray(jnp.all(evidence.successful) & jnp.isfinite(initial_loss))
    )
    best_loss = jnp.where(
        successful, initial_loss, jnp.asarray(jnp.inf, initial_loss.dtype)
    )
    return KineticRolloutTrainingState(
        model,
        optimizer_state,
        model,
        raw_key,
        best_loss,
        attempt_count=0,
        accepted_update_count=0,
        rejection_count=0,
        curriculum_index=0,
        accepted_in_curriculum=0,
        training_cursor=0,
        guard_cursor=0,
        last_update_accepted=False,
        plan_id=plan.plan_id,
        dataset_id=dataset.preparation_id,
        model_structure_id=_model_structure_id(model),
    )


def _validate_state_binding(
    state: KineticRolloutTrainingState,
    plan: KineticRolloutTrainingPlan,
    dataset: PreparedSmoothCompressibleRolloutDataset,
    /,
) -> None:
    _validate_dataset(plan, dataset)
    if (
        not isinstance(state, KineticRolloutTrainingState)
        or state.plan_id != plan.plan_id
        or state.dataset_id != dataset.preparation_id
        or state.model_structure_id != _model_structure_id(state.model)
        or state.curriculum_index >= len(plan.curriculum_horizons)
        or (
            state.curriculum_index < len(plan.curriculum_horizons)
            and state.accepted_in_curriculum
            > plan.accepted_updates_per_horizon[state.curriculum_index]
        )
        or state.training_cursor >= len(dataset.train_windows)
        or state.guard_cursor >= len(_guard_windows(dataset))
    ):
        raise ValueError("Training state identities or progress do not match this run.")


def attempt_kinetic_rollout_update(
    state: KineticRolloutTrainingState,
    plan: KineticRolloutTrainingPlan,
    dataset: PreparedSmoothCompressibleRolloutDataset,
    /,
) -> KineticRolloutUpdateResult:
    """Attempt one proposal and atomically commit model and optimizer on guard success."""

    _validate_state_binding(state, plan, dataset)
    horizon = plan.curriculum_horizons[state.curriculum_index]
    training_key, guard_key, next_key = jr.split(jr.wrap_key_data(state.key), num=3)
    training_start = (
        state.training_cursor
        + int(np.asarray(jr.randint(training_key, (), 0, len(dataset.train_windows))))
    ) % len(dataset.train_windows)
    training_batch, next_training_cursor = _cyclic_batch(
        dataset.train_windows, plan.batch_size, training_start
    )
    guard_pool = _guard_windows(dataset)
    guard_start = (
        state.guard_cursor
        + int(np.asarray(jr.randint(guard_key, (), 0, len(guard_pool))))
    ) % len(guard_pool)
    guard_batch, next_guard_cursor = _cyclic_batch(
        guard_pool, plan.guard_batch_size, guard_start
    )

    def objective(candidate: AbstractArrayModel):
        return kinetic_rollout_objective(
            candidate, plan, dataset.statistics, training_batch, horizon
        )

    (training_loss, training_evidence), gradient = eqx.filter_value_and_grad(
        objective, has_aux=True
    )(state.model)
    gradient_finite = _tree_finite(gradient) & jnp.isfinite(training_loss)
    updates, candidate_optimizer_state = plan.optimizer().update(
        gradient,
        state.optimizer_state,
        params=eqx.filter(state.model, eqx.is_inexact_array),
    )
    candidate_model = eqx.apply_updates(state.model, updates)
    proposal_finite = (
        gradient_finite
        & _tree_finite(updates)
        & _tree_finite(candidate_model)
        & _tree_finite(candidate_optimizer_state)
    )
    guard_loss, guard_evidence = kinetic_rollout_objective(
        candidate_model, plan, dataset.statistics, guard_batch, horizon
    )
    accepted = bool(
        np.asarray(
            proposal_finite
            & jnp.isfinite(guard_loss)
            & jnp.all(guard_evidence.successful)
        )
    )
    model = candidate_model if accepted else state.model
    optimizer_state = candidate_optimizer_state if accepted else state.optimizer_state
    accepted_count = state.accepted_update_count + int(accepted)
    rejection_count = state.rejection_count + int(not accepted)
    accepted_in_curriculum = state.accepted_in_curriculum + int(accepted)
    curriculum_index = state.curriculum_index
    if (
        accepted
        and accepted_in_curriculum == plan.accepted_updates_per_horizon[curriculum_index]
        and curriculum_index + 1 < len(plan.curriculum_horizons)
    ):
        curriculum_index += 1
        accepted_in_curriculum = 0
    elif accepted_in_curriculum > plan.accepted_updates_per_horizon[curriculum_index]:
        raise ValueError("Training state exceeded its curriculum acceptance gate.")

    selection_model = candidate_model if accepted else state.model
    selection_loss, selection_evidence = kinetic_rollout_objective(
        selection_model,
        plan,
        dataset.statistics,
        guard_pool,
        plan.curriculum_horizons[-1],
    )
    improves = accepted and bool(
        np.asarray(
            jnp.isfinite(selection_loss)
            & jnp.all(selection_evidence.successful)
            & (selection_loss < state.best_loss)
        )
    )
    best_model = candidate_model if improves else state.best_model
    best_loss = selection_loss if improves else state.best_loss
    next_state = KineticRolloutTrainingState(
        model,
        optimizer_state,
        best_model,
        jr.key_data(next_key),
        best_loss,
        attempt_count=state.attempt_count + 1,
        accepted_update_count=accepted_count,
        rejection_count=rejection_count,
        curriculum_index=curriculum_index,
        accepted_in_curriculum=accepted_in_curriculum,
        training_cursor=next_training_cursor,
        guard_cursor=next_guard_cursor,
        last_update_accepted=accepted,
        plan_id=state.plan_id,
        dataset_id=state.dataset_id,
        model_structure_id=state.model_structure_id,
    )
    return KineticRolloutUpdateResult(
        state=next_state,
        training_evidence=training_evidence,
        guard_evidence=guard_evidence,
        selection_evidence=selection_evidence,
        training_loss=training_loss,
        guard_loss=guard_loss,
        selection_loss=selection_loss,
        gradient_finite=gradient_finite,
        proposal_finite=proposal_finite,
        accepted=accepted,
        horizon=horizon,
    )


def curriculum_complete(
    state: KineticRolloutTrainingState, plan: KineticRolloutTrainingPlan, /
) -> bool:
    if state.curriculum_index != len(plan.curriculum_horizons) - 1:
        return False
    return (
        state.accepted_in_curriculum
        >= plan.accepted_updates_per_horizon[state.curriculum_index]
    )


def train_kinetic_rollout(
    state: KineticRolloutTrainingState,
    plan: KineticRolloutTrainingPlan,
    dataset: PreparedSmoothCompressibleRolloutDataset,
    maximum_attempts: int,
    /,
) -> KineticRolloutTrainingResult:
    """Run bounded transactional attempts without skipping any curriculum horizon."""

    attempts_limit = _positive_integer(maximum_attempts, "maximum_attempts")
    current = state
    attempts: list[KineticRolloutUpdateResult] = []
    for _ in range(attempts_limit):
        if curriculum_complete(current, plan):
            break
        update = attempt_kinetic_rollout_update(current, plan, dataset)
        attempts.append(update)
        current = update.state
    termination: KineticRolloutTermination = (
        "curriculum_complete"
        if curriculum_complete(current, plan)
        else "maximum_attempts"
    )
    return KineticRolloutTrainingResult(current, tuple(attempts), termination)


__all__ = [
    "attempt_kinetic_rollout_update",
    "curriculum_complete",
    "initialize_kinetic_rollout_training",
    "kinetic_rollout_objective",
    "KineticRolloutObjectiveEvidence",
    "KineticRolloutReplayMode",
    "KineticRolloutTermination",
    "KineticRolloutTrainingPlan",
    "KineticRolloutTrainingResult",
    "KineticRolloutTrainingState",
    "KineticRolloutUpdateResult",
    "train_kinetic_rollout",
]
