# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Deterministic physical twin: this is NOT Earth-climate validation.

Run: JAX_ENABLE_X64=1 PYTHONPATH=. python examples/interactive_column_inference.py
All grey optical coefficients and synthetic instrument errors below are declared
experimental choices, not measurements or claimed Earth parameter estimates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import jax
import jax.numpy as jnp
import numpy as np

from phydrax.applications.atmosphere._interactive_column import InteractiveMoistColumnPlan
from phydrax.applications.atmosphere._radiation import (
    ColumnOpticalProperties,
    ColumnRadiationPlan,
)
from phydrax.applications.atmosphere._surface import BulkSurfaceExchangePlan
from phydrax.applications.geophysics._inference import (
    column_gradient_audit,
    ColumnCalibrationProblem,
    ColumnDesignCandidate,
    ColumnExperiment,
    ColumnIntervention,
    ColumnObservationBinding,
    ColumnObservationData,
    ColumnParameterSpace,
    design_column_intervention,
    load_column_inference,
    save_column_inference,
    score_column_holdout,
)
from phydrax.applications.geophysics._observations import (
    prepare_geophysical_observations,
    prepare_tensor_observation_operator,
)
from phydrax.applications.geophysics._quantities import GeophysicalQuantity
from phydrax.applications.geophysics._time import GeophysicalTimeSpec
from phydrax.discretization import DiscreteFieldSpace, TensorDofLayout
from phydrax.linalg import ArraySpace
from phydrax.optim import OptimizationTermination
from phydrax.units import derived_unit, JOULE, KELVIN, KILOGRAM, METER, ONE, SECOND
from phydrax.uq import DenseCovariance


SYNTHETIC_PROVENANCE = "deterministic-interactive-column-physical-twin; synthetic-grey-optics; no-Earth-validation"
FLUX = derived_unit("W/m2", ((JOULE, 1), (SECOND, -1), (METER, -2)))
WATER = derived_unit("kg/m2", ((KILOGRAM, 1), (METER, -2)))
SIGNALS = {
    "toa_net_upward_flux": ("radiative_flux", FLUX, "upward", "absolute", 0.2),
    "surface_net_downward_flux": ("radiative_flux", FLUX, "downward", "absolute", 0.2),
    "temperature": ("temperature", KELVIN, "positive", "absolute", 0.005),
    "specific_humidity": ("specific_humidity", ONE, "positive", "absolute", 1e-6),
    "surface_temperature": ("temperature", KELVIN, "positive", "absolute", 1e-4),
    "surface_water_mass": ("water_mass_per_area", WATER, "positive", "absolute", 1e-4),
    "precipitated_water": (
        "precipitation_amount",
        WATER,
        "positive",
        "experiment-initial-state",
        1e-5,
    ),
}


def twin_model():
    optics = ColumnOpticalProperties(
        shortwave_absorption=(0.0004, 0.02, 0.4, 0.2),
        shortwave_scattering=(0.0001, 0.0, 0.5, 0.3),
        longwave_absorption=(0.003, 0.1, 0.2, 0.2),
        reference_id=SYNTHETIC_PROVENANCE,
    )
    plan = InteractiveMoistColumnPlan(
        radiation=ColumnRadiationPlan(optics),
        surface_exchange=BulkSurfaceExchangePlan(stability="neutral"),
        mixing_length=0.0,
        background_diffusivity=0.2,
    )
    initial = plan.initialize(
        [100.0, 110.0],
        [0.2, 0.4],
        [285.0, 290.0],
        [100.0, 100.0],
        rain_mass=[0.2, 0.25],
        surface_temperature=295.0,
        surface_water_mass=1000.0,
    )
    space = ColumnParameterSpace(
        (
            "longwave_absorption_scale",
            "shortwave_absorption_scale",
            "heat_transfer_coefficient",
            "moisture_transfer_coefficient",
            "background_diffusivity",
            "rain_evaporation_timescale",
        ),
        [1.0, 1.0, 0.0012, 0.0012, 0.2, 120.0],
        [0.5, 0.5, 0.0006, 0.0006, 0.05, 80.0],
        [1.5, 1.5, 0.002, 0.002, 0.4, 180.0],
    )
    truth = jnp.asarray([1.1, 0.9, 0.0014, 0.0011, 0.25, 135.0])
    return plan, initial, space, truth


def twin_experiment(
    initial,
    *,
    steps=12,
    dt=1.0,
    signals=None,
    times=None,
    intervention=None,
    source_id="synthetic-control",
):
    clock = GeophysicalTimeSpec()
    signals = tuple(SIGNALS) if signals is None else tuple(signals)
    times = (
        tuple(float(t) for t in (0.0, steps * dt / 2, steps * dt))
        if times is None
        else tuple(times)
    )
    bindings = []
    for signal in signals:
        kind, unit, sign, reference, _ = SIGNALS[signal]
        n = initial.dry_mass.size if signal in ("temperature", "specific_humidity") else 1
        quantity = GeophysicalQuantity(
            signal, kind, unit, sign_convention=sign, reference_configuration=reference
        )
        source = DiscreteFieldSpace(
            signal,
            f"synthetic-top-to-bottom-column-{n}",
            TensorDofLayout(("layer_index",), (n,)),
            ArraySpace((n,), dtype=initial.time.dtype),
            representation="point_value",
        )
        operator = prepare_tensor_observation_operator(
            source,
            {"layer_index": np.arange(n)},
            {"layer_index": np.arange(n)},
            quantity,
            source_support_id=source.support_id,
            time=clock,
            kind="profile",
        )
        bindings.append(ColumnObservationBinding(signal, operator, times))
    return ColumnExperiment(
        initial,
        bindings,
        time=clock,
        dt=dt,
        steps=steps,
        source_id=source_id,
        intervention=intervention,
    )


def twin_data(experiment, truth_plan, *, role, noise_seed=None):
    prediction = experiment.predict(truth_plan)
    if not bool(prediction.successful):
        raise ValueError(
            "Synthetic truth was physically rejected; do not manufacture observations."
        )
    # Default is an exact mean twin. Explicit covariance still defines the
    # hypothetical sensor likelihood; this is not an empirical error estimate.
    products, cursor = [], 0
    for i, binding in enumerate(experiment.bindings):
        shape = (
            len(binding.times),
        ) + binding.operator.transfer.target.vector_space.shape
        size = int(np.prod(shape))
        values = prediction.values[cursor : cursor + size].reshape(shape)
        std = SIGNALS[binding.signal][-1]
        if noise_seed is not None:
            values = values + std * jax.random.normal(
                jax.random.fold_in(jax.random.key(noise_seed), i),
                shape,
                dtype=values.dtype,
            )
        cursor += size
        products.append(
            prepare_geophysical_observations(
                binding.operator,
                binding.times,
                values,
                std,
                quantity=binding.operator.quantity,
                time=experiment.time,
                target_support_id=binding.operator.transfer.target.support_id,
                case_id=f"{SYNTHETIC_PROVENANCE}:{experiment.experiment_id}",
            )
        )
    return ColumnObservationData(
        experiment,
        products,
        provenance=SYNTHETIC_PROVENANCE
        + (
            ";exact-mean-observations"
            if noise_seed is None
            else f";Gaussian-noise-seed={noise_seed}"
        ),
        role=role,
    )


def twin_candidate(experiment):
    variances = np.concatenate(
        [
            np.full(
                len(b.times) * b.operator.transfer.target.vector_space.size,
                SIGNALS[b.signal][-1] ** 2,
            )
            for b in experiment.bindings
        ]
    )
    return ColumnDesignCandidate(
        experiment,
        DenseCovariance(jnp.diag(jnp.asarray(variances))),
        "declared-synthetic-instrument-errors;independent;not-estimated-from-heldout-responses",
    )


def run_twin(*, steps=12, maximum_steps=24, audit=True):
    if not jax.config.x64_enabled:
        raise ValueError(
            "This qualification requires JAX_ENABLE_X64=1 for small physical response differences."
        )
    plan, initial, space, truth = twin_model()
    truth_plan = space.apply(plan, truth)
    control = twin_experiment(initial, steps=steps)
    data = twin_data(control, truth_plan, role="calibration")
    problem = ColumnCalibrationProblem(plan, space, (data,))
    confounded_experiment = twin_experiment(
        initial,
        steps=steps,
        signals=("toa_net_upward_flux",),
        times=(0.0,),
        source_id="synthetic-single-TOA",
    )
    confounded = ColumnCalibrationProblem(
        plan, space, (twin_data(confounded_experiment, truth_plan, role="calibration"),)
    )
    confounding = confounded.information(truth)
    # Measured float64 diagnostics near this exact twin's optimum: whitened
    # objective 1.27e-20, max residual 1.14e-10, gradient norm 2.42e-8.
    # Use a declared 1e-7 gradient tolerance, not a success-status override.
    termination = OptimizationTermination(
        maximum_steps=maximum_steps, absolute_optimality=1e-7, relative_optimality=1e-8
    )
    result = problem.calibrate(space.scales, termination=termination)
    greenhouse = twin_experiment(
        initial,
        steps=steps,
        intervention=ColumnIntervention("greenhouse-opacity", longwave_multiplier=1.2),
        source_id="withheld-greenhouse",
    )
    solar = twin_experiment(
        initial,
        steps=steps,
        intervention=ColumnIntervention("solar", solar_multiplier=1.1),
        source_id="withheld-solar",
    )
    capacity = twin_experiment(
        initial,
        steps=steps,
        intervention=ColumnIntervention(
            "half-slab-capacity", slab_capacity_multiplier=0.5
        ),
        source_id="prospective-slab",
    )
    targeted = twin_experiment(
        initial,
        steps=steps,
        signals=("temperature", "specific_humidity", "precipitated_water"),
        source_id="prospective-targeted-profile",
    )
    # Design sees only experiments and stipulated covariance. NO truth responses.
    design = design_column_intervention(
        problem,
        result,
        tuple(twin_candidate(e) for e in (greenhouse, solar, capacity, targeted)),
        reference_covariance=np.eye(len(space.names)),
    )
    # Only now generate hidden outcomes. They are never passed to calibrate/design.
    heldouts = {
        "greenhouse": score_column_holdout(
            problem, result, twin_data(greenhouse, truth_plan, role="holdout")
        ),
        "solar": score_column_holdout(
            problem, result, twin_data(solar, truth_plan, role="holdout")
        ),
    }
    gradients = (
        column_gradient_audit(
            problem, truth, jnp.asarray([0.2, -0.1, 0.15, -0.1, 0.1, 0.1])
        )
        if audit
        else None
    )
    continuation_error = None
    if bool(result.successful):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "inference.npz"
            save_column_inference(path, problem, result)
            restored = load_column_inference(path, problem)
            fitted = space.apply(plan, result.parameters)
            native_state = control.predict(fitted).final_state
            expected = fitted.step(native_state, control.dt, **control.forcing)
            resumed = restored["plan"].step(
                restored["state"], control.dt, **control.forcing
            )
            continuation_error = max(
                float(jnp.max(jnp.abs(a - b)))
                for a, b in zip(
                    jax.tree.leaves(expected.state), jax.tree.leaves(resumed.state)
                )
            )
            if not bool(expected.successful & resumed.successful):
                raise ValueError("Physical continuation was rejected.")
    serial = lambda x: np.asarray(x).tolist()
    summary = {
        "provenance": SYNTHETIC_PROVENANCE,
        "parameters": list(space.names),
        "truth": serial(truth),
        "estimate": serial(result.parameters),
        "successful": bool(result.successful),
        "optimizer_status": int(result.optimizer.status),
        "optimizer_objective": float(result.optimizer.objective),
        "optimizer_initial_optimality": float(
            result.optimizer.diagnostics.initial_optimality_norm
        ),
        "optimizer_final_optimality": float(
            result.optimizer.diagnostics.final_optimality_norm
        ),
        "optimizer_final_step": float(result.optimizer.diagnostics.final_step_norm),
        "optimizer_optimality_threshold": float(
            termination.optimality_threshold(
                result.optimizer.diagnostics.initial_optimality_norm
            )
        ),
        "optimizer_termination_policy": {
            "absolute_optimality": termination.absolute_optimality,
            "relative_optimality": termination.relative_optimality,
            "absolute_step": termination.absolute_step,
            "relative_step": termination.relative_step,
            "maximum_steps": termination.maximum_steps,
            "maximum_evaluations": termination.maximum_evaluations,
        },
        "confounded_rank": int(confounding.rank),
        "complementary_rank": int(result.information.rank),
        "singular_values": serial(result.information.singular_values),
        "identifiable_combinations": serial(result.information.combinations),
        "bounds_respected": bool(
            jnp.all(
                (result.parameters >= space.lower) & (result.parameters <= space.upper)
            )
        ),
        "derivative_valid": bool(result.information.derivative_valid),
        "approximation": result.information.approximation,
        "candidate_names": [
            e.intervention.label for e in (greenhouse, solar, capacity, targeted)
        ],
        "chosen_candidate": design.chosen,
        "expected_information_gain": serial(design.expected_information_gain),
        "design_valid": serial(design.valid),
        "continuation_max_error": continuation_error,
        "inference_id": result.inference_id,
        "design_id": design.design_id,
        "heldout": {
            name: {k: (v if isinstance(v, str) else serial(v)) for k, v in score.items()}
            for name, score in heldouts.items()
        },
        "gradient_audit": None
        if gradients is None
        else {k: serial(v) for k, v in gradients.items()},
    }
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--maximum-steps", type=int, default=24)
    parser.add_argument("--skip-gradient-audit", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            run_twin(
                steps=args.steps,
                maximum_steps=args.maximum_steps,
                audit=not args.skip_gradient_audit,
            ),
            indent=2,
        )
    )
