#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

import phydrax as phx
import phydrax.closure_data as closure_data
import phydrax.discretization.discrete_velocity as discrete_velocity
import phydrax.equations as equations
from benchmarks._comparison import compare_performance, PerformancePolicy
from benchmarks._io import write_json_atomic
from benchmarks._runtime import (
    capture_environment,
    DurationDistribution,
    measure_lower_and_compile,
    measure_synchronized,
)


MODEL_SEED = 17_031
TRAINING_SEED = 91_727
MAXIMUM_UPDATES = 2_000
VALIDATION_INTERVAL = 50
PATIENCE = 10
TRAINING_BATCH_SIZE = 128
LEARNING_RATE = 3.0e-3
MINIMUM_VALIDATION_IMPROVEMENT = 1.0e-7
BENCHMARK_WARMUPS = 2
BENCHMARK_REPETITIONS = 10

THRESHOLDS = {
    "oracle_maximum_root_residual": 1.0e-9,
    "oracle_maximum_relative_energy_error": 1.0e-12,
    "oracle_minimum_hull_margin": 1.0e-6,
    "validation_mean_relative_population_error": 2.0e-2,
    "test_mean_relative_population_error": 2.0e-2,
    "test_maximum_relative_population_error": 6.0e-2,
    "test_mean_scaled_flux_error": 2.0e-2,
    "test_maximum_scaled_flux_error": 6.0e-2,
    "test_maximum_relative_energy_error": 1.0e-12,
    "gradient_relative_discrepancy": 5.0e-4,
    "collision_maximum_conservation_residual": 1.0e-10,
    "batch_256_minimum_speedup": 1.25,
}


def _physical_targets(
    conserved: jax.Array, material: equations.IdealGasMaterial
) -> tuple[jax.Array, jax.Array, jax.Array]:
    density = conserved[..., 0]
    momentum = conserved[..., 1:3]
    total_energy = conserved[..., 3]
    velocity = momentum / density[..., None]
    kinetic_energy = 0.5 * jnp.sum(momentum * velocity, axis=-1)
    specific_internal_energy = (total_energy - kinetic_energy) / density
    pressure = material.pressure(density, specific_internal_energy)
    target_flux = (total_energy + pressure)[..., None] * velocity
    return total_energy, pressure, target_flux


def _corpus(
    material: equations.IdealGasMaterial,
) -> tuple[jax.Array, jax.Array, tuple[tuple[str, int, str], ...]]:
    density_levels = np.linspace(0.8, 1.2, 5, dtype=np.float64)
    velocity_levels = np.linspace(-0.15, 0.15, 5, dtype=np.float64)
    temperature_levels = np.linspace(0.4, 0.6, 5, dtype=np.float64)
    primitive_rows: list[tuple[float, float, float, float]] = []
    metadata: list[tuple[str, int, str]] = []

    for density_index, density in enumerate(density_levels):
        for temperature_index, temperature in enumerate(temperature_levels):
            case_id = f"rho-{density_index}-temperature-{temperature_index}"
            latin_class = (density_index + 2 * temperature_index) % 5
            split = (
                "test"
                if latin_class == 0
                else "validation"
                if latin_class == 1
                else "train"
            )
            time_index = 0
            for velocity_x in velocity_levels:
                for velocity_y in velocity_levels:
                    primitive_rows.append(
                        (
                            float(density),
                            float(velocity_x),
                            float(velocity_y),
                            float(temperature),
                        )
                    )
                    metadata.append((case_id, time_index, split))
                    time_index += 1

    primitive = jnp.asarray(primitive_rows, dtype=jnp.float64)
    density = primitive[:, 0]
    velocity = primitive[:, 1:3]
    temperature = primitive[:, 3]
    pressure = density * material.gas_constant * temperature
    specific_internal_energy = material.specific_internal_energy(density, pressure)
    total_energy = density * specific_internal_energy + 0.5 * density * jnp.sum(
        velocity * velocity, axis=-1
    )
    conserved = jnp.concatenate(
        (density[:, None], density[:, None] * velocity, total_energy[:, None]), axis=-1
    )
    return primitive, conserved, tuple(metadata)


def _flow_schema() -> closure_data.FlowStateSchema:
    return closure_data.FlowStateSchema(
        ("rho", "rho_u_x", "rho_u_y", "total_energy_density"),
        ("1", "1", "1", "1"),
        (1.0, 1.0, 1.0, 1.0),
        density_name="rho",
        total_energy_name="total_energy_density",
    )


def _prepare_dataset(
    conserved: jax.Array,
    oracle_dual: jax.Array,
    metadata: tuple[tuple[str, int, str], ...],
    schema: closure_data.FlowStateSchema,
    equilibrium_plan: discrete_velocity.PositiveEnergyEquilibriumPlan,
    material: equations.IdealGasMaterial,
) -> tuple[
    closure_data.PreparedEnergyEquilibriumDataset,
    tuple[closure_data.ClosureSample, ...],
]:
    keys = tuple(
        closure_data.ClosureSampleKey(
            case_id=case_id,
            trajectory_id="interior-grid",
            realization_id="deterministic",
            time_block_id="velocity-grid",
            time_index=time_index,
        )
        for case_id, time_index, _ in metadata
    )
    conserved_samples = tuple(
        closure_data.ClosureSample(values, key, schema_id=schema.schema_id)
        for values, key in zip(conserved, keys, strict=True)
    )
    oracle_samples = tuple(
        closure_data.ClosureSample(values, key, schema_id=schema.schema_id)
        for values, key in zip(oracle_dual, keys, strict=True)
    )
    pairs = tuple(
        closure_data.EnergyEquilibriumTrainingPair(
            state,
            dual,
            quadrature_id=equilibrium_plan.quadrature.quadrature_id,
            material_id=material.material_id,
            oracle_plan_id=equilibrium_plan.plan_id,
        )
        for state, dual in zip(conserved_samples, oracle_samples, strict=True)
    )

    partition_plan = closure_data.LeakageSafePartitionPlan(
        "case",
        train_fraction=0.6,
        validation_fraction=0.2,
        test_fraction=0.2,
        salt="deterministic-latin-rho-temperature",
    )
    assignments = tuple(
        closure_data.PartitionAssignment(
            sample_id=sample.sample_id,
            group_key=sample.key.group_key("case"),
            split=record[2],
        )
        for sample, record in zip(conserved_samples, metadata, strict=True)
    )
    partition = closure_data.LeakageSafePartition(partition_plan, assignments)
    normalizer = closure_data.TrainOnlyNormalizer.fit(
        conserved_samples,
        partition,
        feature_name="native-conserved-state",
        epsilon=1.0e-12,
    )

    extents: list[closure_data.DatasetExtent] = []
    chunks: list[closure_data.ClosureDatasetChunk] = []
    case_ids = tuple(dict.fromkeys(record[0] for record in metadata))
    conserved_host = np.asarray(conserved)
    oracle_host = np.asarray(oracle_dual)
    for case_id in case_ids:
        indices = tuple(
            index for index, record in enumerate(metadata) if record[0] == case_id
        )
        extent = closure_data.DatasetExtent(
            case_id=case_id,
            trajectory_id="interior-grid",
            realization_id="deterministic",
            time_block_id="velocity-grid",
            sample_count=len(indices),
        )
        payload = (
            np.concatenate(
                (
                    conserved_host[np.asarray(indices)].reshape(-1),
                    oracle_host[np.asarray(indices)].reshape(-1),
                )
            )
            .astype("<f8", copy=False)
            .tobytes(order="C")
        )
        chunk = closure_data.ClosureDatasetChunk.from_payload(
            payload,
            extent_id=extent.extent_id,
            logical_name=f"aligned-{case_id}",
            chunk_index=0,
            sample_start=0,
            sample_stop=len(indices),
            byte_offset=0,
        )
        extents.append(extent)
        chunks.append(chunk)

    manifest = closure_data.ChunkedClosureDatasetManifest(
        dataset_id="deterministic-interior-d2v17-energy-equilibrium",
        schema_id=schema.schema_id,
        analysis_dag_id="ideal-gas-to-positive-energy-oracle",
        extents=tuple(extents),
        chunks=tuple(chunks),
    )
    dataset = closure_data.prepare_energy_equilibrium_dataset(
        pairs, manifest, partition, normalizer
    )
    return dataset, conserved_samples


def _split_arrays(
    pairs: tuple[closure_data.EnergyEquilibriumTrainingPair, ...],
    material: equations.IdealGasMaterial,
    equilibrium_plan: discrete_velocity.PositiveEnergyEquilibriumPlan,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    conserved = jnp.stack(tuple(pair.conserved.values for pair in pairs), axis=0)
    dual = jnp.stack(tuple(pair.oracle_dual.values for pair in pairs), axis=0)
    total_energy, _, target_flux = _physical_targets(conserved, material)
    populations = equilibrium_plan.evaluate(total_energy, target_flux, dual).populations
    return conserved, total_energy, target_flux, populations


def _relative_population_errors(
    populations: jax.Array, reference: jax.Array
) -> jax.Array:
    numerator = jnp.sqrt(jnp.sum((populations - reference) ** 2, axis=-1))
    denominator = jnp.maximum(
        jnp.sqrt(jnp.sum(reference**2, axis=-1)),
        jnp.finfo(reference.dtype).tiny,
    )
    return numerator / denominator


def _train(
    dataset: closure_data.PreparedEnergyEquilibriumDataset,
    equilibrium_plan: discrete_velocity.PositiveEnergyEquilibriumPlan,
    material: equations.IdealGasMaterial,
) -> tuple[Any, dict[str, Any]]:
    train_state, train_energy, train_flux, train_population = _split_arrays(
        dataset.train_pairs, material, equilibrium_plan
    )
    validation_state, validation_energy, validation_flux, validation_population = (
        _split_arrays(dataset.validation_pairs, material, equilibrium_plan)
    )
    model = phx.nn.models.MLP(
        in_size=4,
        out_size=2,
        hidden_sizes=(32, 32),
        activation=jax.nn.tanh,
        key=jr.key(MODEL_SEED),
    )
    optimizer = optax.adam(LEARNING_RATE)
    optimizer_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    def loss(
        candidate: Any,
        states: jax.Array,
        energies: jax.Array,
        fluxes: jax.Array,
        reference: jax.Array,
    ) -> jax.Array:
        normalized = dataset.normalizer.normalize(states)
        dual = jax.vmap(lambda point: candidate(point, key=None))(normalized)
        populations = equilibrium_plan.evaluate(energies, fluxes, dual).populations
        return jnp.mean(_relative_population_errors(populations, reference))

    loss_and_gradient = eqx.filter_value_and_grad(loss)

    @eqx.filter_jit
    def update(
        candidate: Any,
        state: optax.OptState,
        states: jax.Array,
        energies: jax.Array,
        fluxes: jax.Array,
        reference: jax.Array,
    ) -> tuple[Any, optax.OptState, jax.Array]:
        value, gradient = loss_and_gradient(
            candidate, states, energies, fluxes, reference
        )
        updates, next_state = optimizer.update(gradient, state, candidate)
        return eqx.apply_updates(candidate, updates), next_state, value

    evaluate_loss = eqx.filter_jit(loss)
    best_model = model
    best_validation = float(
        evaluate_loss(
            model,
            validation_state,
            validation_energy,
            validation_flux,
            validation_population,
        )
    )
    validation_history: list[dict[str, float | int]] = [
        {"update": 0, "mean_relative_population_error": best_validation}
    ]
    stale_validations = 0
    updates_completed = 0
    stop_reason = "maximum_updates"
    training_key = jr.key(TRAINING_SEED)
    batch_size = min(TRAINING_BATCH_SIZE, int(train_state.shape[0]))

    for update_index in range(1, MAXIMUM_UPDATES + 1):
        batch_indices = jr.choice(
            jr.fold_in(training_key, update_index),
            train_state.shape[0],
            shape=(batch_size,),
            replace=False,
        )
        model, optimizer_state, _ = update(
            model,
            optimizer_state,
            train_state[batch_indices],
            train_energy[batch_indices],
            train_flux[batch_indices],
            train_population[batch_indices],
        )
        updates_completed = update_index
        if update_index % VALIDATION_INTERVAL != 0:
            continue
        validation_loss = float(
            evaluate_loss(
                model,
                validation_state,
                validation_energy,
                validation_flux,
                validation_population,
            )
        )
        validation_history.append(
            {
                "update": update_index,
                "mean_relative_population_error": validation_loss,
            }
        )
        if validation_loss < best_validation - MINIMUM_VALIDATION_IMPROVEMENT:
            best_model = model
            best_validation = validation_loss
            stale_validations = 0
        else:
            stale_validations += 1
        if stale_validations >= PATIENCE:
            stop_reason = "validation_patience"
            break

    best_training = float(
        evaluate_loss(
            best_model,
            train_state,
            train_energy,
            train_flux,
            train_population,
        )
    )
    return best_model, {
        "model_seed": MODEL_SEED,
        "training_seed": TRAINING_SEED,
        "optimizer": "optax.adam",
        "learning_rate": LEARNING_RATE,
        "batch_size": batch_size,
        "maximum_updates": MAXIMUM_UPDATES,
        "updates_completed": updates_completed,
        "validation_interval": VALIDATION_INTERVAL,
        "patience": PATIENCE,
        "minimum_validation_improvement": MINIMUM_VALIDATION_IMPROVEMENT,
        "stop_reason": stop_reason,
        "best_model_selection": "immutable Equinox model snapshot at lowest validation loss",
        "best_training_mean_relative_population_error": best_training,
        "best_validation_mean_relative_population_error": best_validation,
        "validation_history": validation_history,
    }


def _evaluate_test(
    binding: closure_data.PreparedLearnedEnergyEquilibriumBinding,
    dataset: closure_data.PreparedEnergyEquilibriumDataset,
    equilibrium_plan: discrete_velocity.PositiveEnergyEquilibriumPlan,
    material: equations.IdealGasMaterial,
) -> tuple[dict[str, Any], tuple[jax.Array, jax.Array, jax.Array]]:
    states, total_energy, target_flux, reference = _split_arrays(
        dataset.test_pairs, material, equilibrium_plan
    )
    learned = binding.evaluate(total_energy, target_flux, states)
    population_error = _relative_population_errors(learned.populations, reference)
    scaled_flux_error = learned.evidence.flux_error_norm / total_energy
    relative_energy_error = jnp.abs(learned.evidence.total_energy_residual) / total_energy
    result = {
        "sample_count": int(states.shape[0]),
        "all_finite": bool(jnp.all(learned.evidence.finite)),
        "all_status_success": bool(jnp.all(learned.evidence.successful)),
        "mean_relative_population_error": float(jnp.mean(population_error)),
        "maximum_relative_population_error": float(jnp.max(population_error)),
        "mean_scaled_flux_error": float(jnp.mean(scaled_flux_error)),
        "maximum_scaled_flux_error": float(jnp.max(scaled_flux_error)),
        "flux_error_scale": "total_energy_density",
        "maximum_absolute_energy_error": float(
            jnp.max(jnp.abs(learned.evidence.total_energy_residual))
        ),
        "maximum_relative_energy_error": float(jnp.max(relative_energy_error)),
        "minimum_population": float(jnp.min(learned.populations)),
        "all_populations_positive": bool(jnp.all(learned.populations > 0.0)),
        "energy_and_flux_interpretation": (
            "energy is exact by quadrature-weighted normalization; flux is a separately "
            "measured learned constitutive quantity"
        ),
    }
    return result, (states, total_energy, target_flux)


def _gradient_check(
    binding: closure_data.PreparedLearnedEnergyEquilibriumBinding,
    material: equations.IdealGasMaterial,
    state: jax.Array,
) -> dict[str, Any]:
    direction = jnp.asarray((0.31, -0.47, 0.23, 0.79), dtype=state.dtype)
    direction = direction / jnp.sqrt(jnp.sum(direction**2))
    step = 1.0e-5

    def response(conserved: jax.Array) -> jax.Array:
        total_energy, _, target_flux = _physical_targets(conserved, material)
        result = binding.evaluate(total_energy, target_flux, conserved)
        return result.evidence.recovered_flux[0] / total_energy

    automatic = jnp.vdot(jax.grad(response)(state), direction)
    finite_difference = (
        response(state + step * direction) - response(state - step * direction)
    ) / (2.0 * step)
    denominator = jnp.maximum(
        jnp.maximum(jnp.abs(automatic), jnp.abs(finite_difference)), 1.0e-8
    )
    discrepancy = jnp.abs(automatic - finite_difference) / denominator
    return {
        "quantity": "learned normalized x-directed total-energy flux",
        "finite_difference_step": step,
        "automatic_directional_derivative": float(automatic),
        "finite_difference_directional_derivative": float(finite_difference),
        "relative_discrepancy": float(discrepancy),
        "finite": bool(jnp.isfinite(automatic) & jnp.isfinite(finite_difference)),
    }


def _collision_checks(
    binding: closure_data.PreparedLearnedEnergyEquilibriumBinding,
    equilibrium_plan: discrete_velocity.PositiveEnergyEquilibriumPlan,
    material: equations.IdealGasMaterial,
    test_states: jax.Array,
) -> dict[str, Any]:
    velocity = test_states[:, 1:3] / test_states[:, 0, None]
    local_index = int(jnp.argmin(jnp.sum(velocity**2, axis=-1)))
    conserved = test_states[local_index]
    dual = binding.predict_dual(conserved)
    method = equations.smooth_compressible_d2v17_method(
        material, equations.ConstantTransport(0.03, 0.04), dtype=jnp.float64
    )
    equilibrium, equilibrium_evidence = method.equilibrium_from_energy_dual_with_evidence(
        conserved, dual, equilibrium_plan
    )
    minimum_population = jnp.minimum(
        jnp.min(equilibrium.particle_populations),
        jnp.min(equilibrium.total_energy_populations),
    )
    amplitude = 0.01 * minimum_population
    particle_perturbation = amplitude * method.particle_nullspace_projector[0]
    energy_perturbation = (
        jnp.zeros_like(equilibrium.total_energy_populations)
        .at[0]
        .set(amplitude)
        .at[1]
        .set(-amplitude)
    )
    state = equations.SmoothCompressibleKineticState(
        equilibrium.particle_populations + particle_perturbation,
        equilibrium.total_energy_populations + energy_perturbation,
    )
    collision = method.collide_with_energy_dual_with_evidence(
        state, jnp.asarray(0.01, dtype=jnp.float64), dual, equilibrium_plan
    )
    accepted_change = jnp.sqrt(
        jnp.sum(
            (collision.accepted_state.particle_populations - state.particle_populations)
            ** 2
        )
        + jnp.sum(
            (
                collision.accepted_state.total_energy_populations
                - state.total_energy_populations
            )
            ** 2
        )
    )

    invalid = method.collide_with_energy_dual_with_evidence(
        state,
        jnp.asarray(0.01, dtype=jnp.float64),
        jnp.full((2,), jnp.nan, dtype=jnp.float64),
        equilibrium_plan,
    )
    rollback_exact = jnp.array_equal(
        invalid.accepted_state.particle_populations, state.particle_populations
    ) & jnp.array_equal(
        invalid.accepted_state.total_energy_populations,
        state.total_energy_populations,
    )
    refusal_observed = ~invalid.successful & invalid.rollback_applied & rollback_exact
    return {
        "local_learned_collision": {
            "equilibrium_successful": bool(equilibrium_evidence.successful),
            "collision_successful": bool(collision.successful),
            "rollback_applied": bool(collision.rollback_applied),
            "maximum_conservation_residual": float(
                collision.collision_evidence.maximum_absolute_residual
            ),
            "post_collision_realizable": bool(
                collision.collision_evidence.post_collision_realizability.realizable
            ),
            "accepted_state_change_norm": float(accepted_change),
        },
        "invalid_dual_refusal": {
            "expected": True,
            "input": "nonfinite learned dual",
            "status": int(invalid.equilibrium_evidence.energy.status),
            "successful": bool(invalid.successful),
            "rollback_applied": bool(invalid.rollback_applied),
            "accepted_state_matches_input_exactly": bool(rollback_exact),
            "observed_as_expected": bool(refusal_observed),
            "interpretation": "expected refusal evidence, not a tool error",
        },
    }


def _compilation_record(timing: Any) -> dict[str, float]:
    return {
        "lowering_seconds": float(timing.lowering_seconds),
        "compilation_seconds": float(timing.compilation_seconds),
    }


def _measure_paired_runtime(
    oracle: Callable[[], Any],
    learned: Callable[[], Any],
    /,
) -> tuple[Any, DurationDistribution, Any, DurationDistribution]:
    for index in range(BENCHMARK_WARMUPS):
        first, second = (oracle, learned) if index % 2 == 0 else (learned, oracle)
        measure_synchronized(first)
        measure_synchronized(second)

    oracle_samples: list[float] = []
    learned_samples: list[float] = []
    oracle_result = None
    learned_result = None
    for index in range(BENCHMARK_REPETITIONS):
        if index % 2 == 0:
            oracle_result, oracle_seconds = measure_synchronized(oracle)
            learned_result, learned_seconds = measure_synchronized(learned)
        else:
            learned_result, learned_seconds = measure_synchronized(learned)
            oracle_result, oracle_seconds = measure_synchronized(oracle)
        oracle_samples.append(oracle_seconds)
        learned_samples.append(learned_seconds)
    if oracle_result is None or learned_result is None:
        raise RuntimeError("Paired runtime measurement produced no samples.")
    return (
        oracle_result,
        DurationDistribution(tuple(oracle_samples)),
        learned_result,
        DurationDistribution(tuple(learned_samples)),
    )


def _runtime_benchmarks(
    binding: closure_data.PreparedLearnedEnergyEquilibriumBinding,
    equilibrium_plan: discrete_velocity.PositiveEnergyEquilibriumPlan,
    conserved_corpus: jax.Array,
    material: equations.IdealGasMaterial,
) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    policy = PerformancePolicy(
        objective="minimize",
        relative_tolerance=0.0,
        confidence=0.95,
        bootstrap_resamples=2_000,
        minimum_samples=BENCHMARK_REPETITIONS,
    )
    pair_ids = tuple(
        f"paired-repetition-{index:02d}" for index in range(BENCHMARK_REPETITIONS)
    )

    for batch_size in (1, 256):
        indices = jnp.arange(batch_size) % conserved_corpus.shape[0]
        states = conserved_corpus[indices]
        total_energy, _, target_flux = _physical_targets(states, material)
        oracle_jit = jax.jit(lambda energy, flux: equilibrium_plan.solve(energy, flux))
        learned_jit = jax.jit(
            lambda energy, flux, conserved: binding.evaluate(energy, flux, conserved)
        )
        oracle_compiled, oracle_compilation = measure_lower_and_compile(
            lambda: oracle_jit.lower(total_energy, target_flux),
            lambda lowered: lowered.compile(),
        )
        learned_compiled, learned_compilation = measure_lower_and_compile(
            lambda: learned_jit.lower(total_energy, target_flux, states),
            lambda lowered: lowered.compile(),
        )
        (
            oracle_result,
            oracle_duration,
            learned_result,
            learned_duration,
        ) = _measure_paired_runtime(
            lambda: oracle_compiled(total_energy, target_flux),
            lambda: learned_compiled(total_energy, target_flux, states),
        )
        comparable = bool(
            oracle_result.populations.shape == learned_result.populations.shape
            and oracle_result.dual.shape == learned_result.dual.shape
            and oracle_result.plan_id == learned_result.plan_id
            and oracle_result.populations.dtype == learned_result.populations.dtype
        )
        comparison = compare_performance(
            oracle_duration,
            learned_duration,
            policy,
            comparison_id=f"d2v17-oracle-vs-learned-batch-{batch_size}",
            baseline_pair_ids=pair_ids,
            candidate_pair_ids=pair_ids,
        )
        oracle_median_value = oracle_duration.median_seconds
        learned_median_value = learned_duration.median_seconds
        if (
            oracle_median_value is None
            or learned_median_value is None
            or learned_median_value <= 0.0
        ):
            raise RuntimeError("Runtime distributions require positive finite medians.")
        oracle_median = float(oracle_median_value)
        learned_median = float(learned_median_value)
        speedup = oracle_median / learned_median
        performance_claim = bool(
            batch_size == 256
            and comparable
            and comparison.sufficient_samples
            and speedup >= THRESHOLDS["batch_256_minimum_speedup"]
        )
        rows[str(batch_size)] = {
            "batch_size": batch_size,
            "workload": "same interior conserved states and total-energy flux targets",
            "dtype": str(states.dtype),
            "collection": "interleaved oracle/learned pairs with alternating order",
            "device": str(states.device),
            "paired": comparison.paired,
            "comparable": comparable,
            "warmups": BENCHMARK_WARMUPS,
            "repetitions": BENCHMARK_REPETITIONS,
            "oracle": {
                "operation": "JIT PositiveEnergyEquilibriumPlan.solve",
                "compilation": _compilation_record(oracle_compilation),
                "steady_state": oracle_duration.to_seconds_dict(),
            },
            "learned": {
                "operation": "JIT bound-model predict_dual plus evaluate",
                "compilation": _compilation_record(learned_compilation),
                "steady_state": learned_duration.to_seconds_dict(),
            },
            "comparison": comparison.to_dict(),
            "median_speedup_oracle_over_learned": speedup,
            "performance_claim": {
                "made": performance_claim,
                "statement": (
                    "bound learned evaluation is at least 1.25x faster than the oracle "
                    "at batch size 256"
                    if performance_claim
                    else "no performance claim"
                ),
            },
        }
    return rows


def _status_counts(status: jax.Array) -> dict[str, int]:
    host = np.asarray(status).reshape(-1)
    return {
        member.name: int(np.count_nonzero(host == int(member)))
        for member in discrete_velocity.EnergyEquilibriumStatus
    }


def _qualification() -> dict[str, Any]:
    material = equations.IdealGasMaterial(1.4, 1.0)
    quadrature = discrete_velocity.d2v17_quadrature(dtype=jnp.float64)
    equilibrium_plan = discrete_velocity.PositiveEnergyEquilibriumPlan(quadrature)
    schema = _flow_schema()
    primitive, conserved, metadata = _corpus(material)
    total_energy, _, target_flux = _physical_targets(conserved, material)
    corpus_oracle = jax.jit(lambda energy, flux: equilibrium_plan.solve(energy, flux))
    oracle, oracle_generation_seconds = measure_synchronized(
        lambda: corpus_oracle(total_energy, target_flux)
    )
    dataset, conserved_samples = _prepare_dataset(
        conserved,
        oracle.dual,
        metadata,
        schema,
        equilibrium_plan,
        material,
    )

    (training_result, training_seconds) = measure_synchronized(
        lambda: _train(dataset, equilibrium_plan, material)
    )
    best_model, training = training_result
    training["wall_seconds"] = training_seconds

    semantic = phx.SemanticProvenance(
        {
            "kind": "learned-positive-total-energy-equilibrium-dual",
            "architecture": {
                "kind": "MLP",
                "sizes": [4, 32, 32, 2],
                "activation": "tanh",
            },
            "input": "train-only-normalized native conserved state",
            "output": "two-component positive-energy exponential-family dual",
            "objective": "mean relative energy-population L2 error",
            "training": {
                "optimizer": "Adam",
                "maximum_updates": MAXIMUM_UPDATES,
                "validation_interval": VALIDATION_INTERVAL,
                "patience": PATIENCE,
                "model_seed": MODEL_SEED,
                "training_seed": TRAINING_SEED,
            },
        },
        resource_ids={
            "dataset_preparation": dataset.preparation_id,
            "material": material.material_id,
            "normalizer": dataset.normalizer.normalizer_id,
            "oracle_plan": equilibrium_plan.plan_id,
            "quadrature": quadrature.quadrature_id,
        },
    )
    numeric_revision = closure_data.energy_equilibrium_numeric_revision(
        semantic, best_model
    )
    binding_plan = closure_data.LearnedEnergyEquilibriumBindingPlan(
        equilibrium_plan,
        schema,
        dataset,
        semantic,
        input_component_names=schema.component_names,
        material_id=material.material_id,
    )
    binding = binding_plan.prepare(best_model, numeric_revision, dataset.normalizer)

    test, test_inputs = _evaluate_test(binding, dataset, equilibrium_plan, material)
    gradient = _gradient_check(binding, material, test_inputs[0][0])
    collision = _collision_checks(binding, equilibrium_plan, material, test_inputs[0])
    runtime = _runtime_benchmarks(binding, equilibrium_plan, conserved, material)

    oracle_relative_energy_error = (
        jnp.abs(oracle.evidence.total_energy_residual) / total_energy
    )
    oracle_record = {
        "generation_wall_seconds_including_first_jit": oracle_generation_seconds,
        "sample_count": int(conserved.shape[0]),
        "status_counts": _status_counts(oracle.status),
        "all_successful": bool(jnp.all(oracle.successful)),
        "all_converged": bool(jnp.all(oracle.evidence.converged)),
        "maximum_root_residual": float(jnp.max(oracle.evidence.residual_norm)),
        "maximum_iterations": int(jnp.max(oracle.evidence.iterations)),
        "minimum_hull_margin": float(jnp.min(oracle.evidence.interior_margin)),
        "all_targets_in_hull_interior": bool(
            jnp.all(oracle.evidence.interior_margin > equilibrium_plan.interior_tolerance)
        ),
        "maximum_relative_energy_error": float(jnp.max(oracle_relative_energy_error)),
        "minimum_population": float(jnp.min(oracle.populations)),
        "all_populations_positive": bool(jnp.all(oracle.populations > 0.0)),
    }

    split_groups = {
        "train": sorted({pair.conserved.key.case_id for pair in dataset.train_pairs}),
        "validation": sorted(
            {pair.conserved.key.case_id for pair in dataset.validation_pairs}
        ),
        "test": sorted({pair.conserved.key.case_id for pair in dataset.test_pairs}),
    }
    training_ids = set(dataset.normalizer.provenance.training_sample_ids)
    holdout_ids = {
        pair.conserved.sample_id
        for pair in (*dataset.validation_pairs, *dataset.test_pairs)
    }
    data_record = {
        "construction": "deterministic Cartesian interior grid",
        "primitive_component_order": ["rho", "u_x", "u_y", "temperature"],
        "primitive_bounds": {
            "rho": [float(jnp.min(primitive[:, 0])), float(jnp.max(primitive[:, 0]))],
            "u_x": [float(jnp.min(primitive[:, 1])), float(jnp.max(primitive[:, 1]))],
            "u_y": [float(jnp.min(primitive[:, 2])), float(jnp.max(primitive[:, 2]))],
            "temperature": [
                float(jnp.min(primitive[:, 3])),
                float(jnp.max(primitive[:, 3])),
            ],
        },
        "levels_per_primitive_component": 5,
        "sample_count": len(conserved_samples),
        "thermodynamic_case_count": 25,
        "samples_per_case": 25,
        "partition_level": dataset.partition.plan.level,
        "assignment_rule": (
            "case-stratified Latin classes over density and temperature; no case "
            "crosses a split"
        ),
        "split_sample_counts": {
            "train": len(dataset.train_pairs),
            "validation": len(dataset.validation_pairs),
            "test": len(dataset.test_pairs),
        },
        "split_group_counts": {
            split: len(groups) for split, groups in split_groups.items()
        },
        "groups_pairwise_disjoint": bool(
            set(split_groups["train"]).isdisjoint(split_groups["validation"])
            and set(split_groups["train"]).isdisjoint(split_groups["test"])
            and set(split_groups["validation"]).isdisjoint(split_groups["test"])
        ),
        "aligned_pair_count": len(dataset.pairs),
        "manifest_extent_count": len(dataset.manifest.extents),
        "manifest_chunk_count": len(dataset.manifest.chunks),
        "manifest_exactly_covers_pairs": len(dataset.pairs) == len(conserved_samples),
        "normalizer_training_sample_count": len(training_ids),
        "normalizer_excludes_all_holdout_samples": training_ids.isdisjoint(holdout_ids),
        "normalizer_mean": [float(value) for value in dataset.normalizer.mean],
        "normalizer_scale": [float(value) for value in dataset.normalizer.scale],
    }

    gates = {
        "dataset_alignment_and_leakage_safety": bool(
            data_record["groups_pairwise_disjoint"]
            and data_record["manifest_exactly_covers_pairs"]
            and data_record["normalizer_excludes_all_holdout_samples"]
            and all(
                data_record["split_sample_counts"][split] > 0
                for split in ("train", "validation", "test")
            )
        ),
        "oracle_root_and_hull": bool(
            oracle_record["all_successful"]
            and oracle_record["all_converged"]
            and oracle_record["all_targets_in_hull_interior"]
            and oracle_record["maximum_root_residual"]
            <= THRESHOLDS["oracle_maximum_root_residual"]
            and oracle_record["minimum_hull_margin"]
            >= THRESHOLDS["oracle_minimum_hull_margin"]
            and oracle_record["maximum_relative_energy_error"]
            <= THRESHOLDS["oracle_maximum_relative_energy_error"]
            and oracle_record["all_populations_positive"]
        ),
        "training_validation": bool(
            math.isfinite(training["best_validation_mean_relative_population_error"])
            and training["updates_completed"] <= MAXIMUM_UPDATES
            and training["best_validation_mean_relative_population_error"]
            <= THRESHOLDS["validation_mean_relative_population_error"]
        ),
        "test_population_accuracy": bool(
            test["all_finite"]
            and test["all_status_success"]
            and test["mean_relative_population_error"]
            <= THRESHOLDS["test_mean_relative_population_error"]
            and test["maximum_relative_population_error"]
            <= THRESHOLDS["test_maximum_relative_population_error"]
        ),
        "test_exact_energy": bool(
            test["maximum_relative_energy_error"]
            <= THRESHOLDS["test_maximum_relative_energy_error"]
        ),
        "test_learned_constitutive_flux": bool(
            test["mean_scaled_flux_error"] <= THRESHOLDS["test_mean_scaled_flux_error"]
            and test["maximum_scaled_flux_error"]
            <= THRESHOLDS["test_maximum_scaled_flux_error"]
        ),
        "test_positivity": bool(test["all_populations_positive"]),
        "finite_gradient": bool(
            gradient["finite"]
            and gradient["relative_discrepancy"]
            <= THRESHOLDS["gradient_relative_discrepancy"]
        ),
        "local_learned_collision": bool(
            collision["local_learned_collision"]["equilibrium_successful"]
            and collision["local_learned_collision"]["collision_successful"]
            and not collision["local_learned_collision"]["rollback_applied"]
            and collision["local_learned_collision"]["post_collision_realizable"]
            and collision["local_learned_collision"]["accepted_state_change_norm"] > 0.0
            and collision["local_learned_collision"]["maximum_conservation_residual"]
            <= THRESHOLDS["collision_maximum_conservation_residual"]
        ),
        "invalid_dual_refusal_and_rollback": bool(
            collision["invalid_dual_refusal"]["observed_as_expected"]
        ),
        "runtime_batch_256": bool(runtime["256"]["performance_claim"]["made"]),
    }

    report = {
        "tool": "learned_energy_equilibrium_qualification",
        "scope": {
            "stage": "deterministic stage-1 scientific qualification",
            "included": [
                "native D2V17 local total-energy equilibrium",
                "oracle-dual supervised model training",
                "local learned equilibrium and collision",
                "paired compiled runtime measurements",
            ],
            "excluded": [
                "transport",
                "boundary conditions",
                "forcing",
                "shock qualification",
                "stage-2 qualification",
                "entropy qualification",
                "production-readiness claims",
            ],
            "claim": "local equilibrium/collision qualification only",
        },
        "contracts": {
            "native_state_order": list(schema.component_names),
            "energy_population_identity": "sum(g) = total_energy_density",
            "energy_population_convention": equilibrium_plan.population_convention,
            "model_output": "two-component dual",
            "normalization": "quadrature-weighted exponential-family normalization",
            "exact_energy_separate_from_learned_flux": True,
        },
        "identities": {
            "quadrature": quadrature.quadrature_id,
            "material": material.material_id,
            "flow_schema": schema.schema_id,
            "oracle_plan": equilibrium_plan.plan_id,
            "dataset_manifest": dataset.manifest.manifest_id,
            "dataset_preparation": dataset.preparation_id,
            "partition": dataset.partition.partition_id,
            "normalizer": dataset.normalizer.normalizer_id,
            "normalizer_provenance": dataset.normalizer.provenance.provenance_id,
            "semantic_provenance": semantic.semantic_id,
            "numeric_revision": numeric_revision.revision_id,
            "binding_plan": binding_plan.plan_id,
            "prepared_binding": binding.prepared_id,
        },
        "environment": capture_environment().to_dict(),
        "thresholds": THRESHOLDS,
        "data": data_record,
        "oracle": oracle_record,
        "training": training,
        "test": test,
        "gradient_check": gradient,
        "collision": collision,
        "runtime": {
            "training_time_reported_separately": True,
            "compilation_reported_separately": True,
            "rows": runtime,
        },
        "gates": gates,
    }
    report["passed"] = all(gates.values())
    return report


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Qualify and benchmark the learned native D2V17 total-energy equilibrium."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/learned_energy_equilibrium.json"),
        help="atomic JSON output path",
    )
    return parser.parse_args()


def main() -> int:
    arguments = _parse_arguments()
    with jax.enable_x64(True):
        report = _qualification()
    write_json_atomic(arguments.output, report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
