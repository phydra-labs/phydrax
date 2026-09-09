#!/usr/bin/env python3
#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Runtime and derivative evidence for periodic solver-interleaved learned stress."""

from __future__ import annotations

import argparse
import json
from math import prod
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp

import phydrax as phx
from benchmarks._runtime import (
    measure_lower_and_compile,
    measure_repeated,
    measure_synchronized,
)
from phydrax._model import AbstractArrayModel
from phydrax.equations._learned_stress import (
    LEARNED_STRESS_FEATURE_NAME,
    LEARNED_STRESS_VELOCITY_GRADIENT_COMPONENTS,
    LEARNED_STRESS_VELOCITY_GRADIENT_UNITS,
)


class _ViscosityStressModel(AbstractArrayModel):
    coefficient: jax.Array
    physical_shape: tuple[int, ...] = eqx.field(static=True)
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(self, coefficient, physical_shape, dtype):
        self.coefficient = jnp.asarray(coefficient, dtype=dtype)
        self.physical_shape = tuple(physical_shape)
        self.in_size = prod(self.physical_shape + (9,))
        self.out_size = prod(self.physical_shape + (3, 3))

    def __call__(self, values, /, *, key=None):
        del key
        gradient = jnp.asarray(values).reshape(self.physical_shape + (3, 3))
        strain = 0.5 * (gradient + jnp.swapaxes(gradient, -1, -2))
        trace = jnp.trace(strain, axis1=-2, axis2=-1)
        deviatoric = strain - (trace / 3.0)[..., None, None] * jnp.eye(
            3, dtype=strain.dtype
        )
        return (-2.0 * self.coefficient * deviatoric).reshape((self.out_size,))


def _case(count):
    space = phx.discretization.TensorSpectralPlan(
        tuple(phx.discretization.FourierBasisPlan(count) for _ in range(3)),
        axis_names=("x", "y", "z"),
        field_name="velocity",
    ).prepare(
        tuple(phx.discretization.AxisDomain.periodic(0.0, 2.0 * jnp.pi) for _ in range(3))
    )
    dtype = jnp.dtype(space.plan.precision.physical_dtype)
    resolved_filter = phx.equations.ResolvedLESFilter(
        "retained Fourier grid",
        family="sharp-fourier-projection",
        axis_names=("x", "y", "z"),
        topology="tensor-product",
        boundary_class="periodic",
        scale_rule="cutoff-equivalent",
        commutation_status="commuting",
        repeated_filter_semantics="idempotent",
    )
    feature_schema = phx.closure_data.LearnedStressFeatureSchema(
        name=LEARNED_STRESS_FEATURE_NAME,
        component_names=LEARNED_STRESS_VELOCITY_GRADIENT_COMPONENTS,
        component_units=LEARNED_STRESS_VELOCITY_GRADIENT_UNITS,
        shape=space.physical_shape + (9,),
        dtype=dtype,
        flow_schema_id="benchmark-periodic-flow",
    )
    output = phx.closure_data.LearnedStressOutputContract(
        shape=space.physical_shape + (3, 3),
        dtype=dtype,
        units="(m/s)^2",
        target_id="benchmark-periodic-stress",
        filter_id=resolved_filter.filter_id,
        discretization_id=space.prepared_id,
        regime="three-dimensional-periodic-unit-density",
        symmetry_tolerance=2e-6,
        trace_tolerance=2e-6,
    )
    normalizer = phx.closure_data.TrainOnlyNormalizer(
        jnp.zeros((9,), dtype=dtype),
        jnp.ones((9,), dtype=dtype),
        phx.closure_data.NormalizerProvenance(
            partition_id="benchmark-train",
            training_assignment_ids=("assignment",),
            training_sample_ids=("sample",),
            feature_name=LEARNED_STRESS_FEATURE_NAME,
            schema_id="benchmark-periodic-flow",
        ),
        epsilon=1e-12,
    )
    plan = phx.closure_data.LearnedStressBindingPlan(
        feature_schema,
        output,
        resolved_filter,
        phx.equations.LESParameterProvenance(
            resolved_filter,
            space.prepared_id,
            "three-dimensional-periodic-unit-density",
            source_kind="user",
            evidence_ids=(),
        ),
        model_artifact_id="benchmark-stress-model",
        normalizer_id=normalizer.normalizer_id,
    )

    def zero_predictor(features, args):
        del args
        return jnp.zeros(features.shape[:-1] + (3, 3), dtype=features.dtype)

    binding = plan.prepare(
        zero_predictor,
        normalizer,
        model_artifact_id="benchmark-stress-model",
        target_id=output.target_id,
        output_units=output.units,
    )
    projector = phx.discretization.PeriodicLerayProjector(space)
    prepared = phx.equations.PeriodicLearnedStressPlan(binding).prepare(space, projector)
    coordinates = phx.discretization.HermitianSpectralCoordinates(
        space,
        component_shape=(3,),
    )
    x, y, z = jnp.meshgrid(
        space.axes[0].nodes,
        space.axes[1].nodes,
        space.axes[2].nodes,
        indexing="ij",
    )
    velocity = jnp.stack(
        (
            jnp.sin(y) * jnp.cos(z),
            jnp.sin(z) * jnp.cos(x),
            jnp.sin(x) * jnp.cos(y),
        ),
        axis=-1,
    )
    state = coordinates.to_real_coordinates(projector.project(space.project(velocity)))

    def zero_base_rate(time, modal, inputs):
        del time, inputs
        return jnp.zeros_like(modal)

    transition = (
        phx.applications.incompressible_flow.PeriodicLearnedStressRolloutTransition(
            prepared,
            coordinates,
            zero_base_rate,
            base_rate_id="benchmark-zero-base-rate",
            state_layout=phx.dynamics.StateLayout((coordinates.coordinate_size,)),
            step_size=0.01,
            step_rtol=0.0,
            step_atol=0.0,
        )
    )
    context = phx.dynamics.DiscreteStepContext(
        jnp.asarray(0.0),
        jnp.asarray(0.01),
        jnp.asarray(0, dtype=jnp.int32),
    )
    model = _ViscosityStressModel(0.1, space.physical_shape, dtype)
    return transition, context, state, model


def run(*, quick: bool) -> dict:
    count = 3 if quick else 5
    repetitions = 2 if quick else 5
    transition, context, state, model = _case(count)

    def advance(candidate, current):
        return transition.evaluate(
            candidate,
            context,
            current,
            None,
            key=jax.random.key(0),
            iteration=jnp.asarray(0),
        )

    jitted = eqx.filter_jit(advance)
    compiled, compilation = measure_lower_and_compile(
        lambda: jitted.lower(model, state),
        lambda lowered: lowered.compile(),
    )
    result, first = measure_synchronized(lambda: compiled(model, state))
    _, steady = measure_repeated(
        lambda: compiled(model, state),
        warmup=0,
        repeats=repetitions,
    )

    def objective(candidate):
        value = advance(candidate, state).accepted_state
        return jnp.sum(jnp.square(value - state))

    objective_value, gradient = eqx.filter_value_and_grad(objective)(model)
    return {
        "benchmark": "phydrax-learned-stress-hybrid",
        "backend": jax.default_backend(),
        "quick": quick,
        "grid_count": count,
        "coordinate_size": int(state.size),
        "lowering_seconds": compilation.lowering_seconds,
        "compilation_seconds": compilation.compilation_seconds,
        "first_execution_seconds": first,
        "steady": steady.to_seconds_dict(),
        "successful": bool(result.training_usable),
        "ssprk_stages": int(result.iterations),
        "state_displacement_squared": float(objective_value),
        "coefficient_gradient": float(gradient.coefficient),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args()
    report = run(quick=arguments.quick)
    payload = json.dumps(report, indent=2, sort_keys=True)
    if arguments.output is None:
        print(payload)
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(payload + "\n")
        print(arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
