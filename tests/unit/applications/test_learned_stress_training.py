from __future__ import annotations

from math import prod

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
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


def _periodic_prepared():
    space = phx.discretization.TensorSpectralPlan(
        tuple(phx.discretization.FourierBasisPlan(3) for _ in range(3)),
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
    schema = phx.closure_data.LearnedStressFeatureSchema(
        name=LEARNED_STRESS_FEATURE_NAME,
        component_names=LEARNED_STRESS_VELOCITY_GRADIENT_COMPONENTS,
        component_units=LEARNED_STRESS_VELOCITY_GRADIENT_UNITS,
        shape=space.physical_shape + (9,),
        dtype=dtype,
        flow_schema_id="periodic-flow",
    )
    output = phx.closure_data.LearnedStressOutputContract(
        shape=space.physical_shape + (3, 3),
        dtype=dtype,
        units="(m/s)^2",
        target_id="periodic-stress",
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
            partition_id="train",
            training_assignment_ids=("assignment",),
            training_sample_ids=("sample",),
            feature_name=LEARNED_STRESS_FEATURE_NAME,
            schema_id="periodic-flow",
        ),
        epsilon=1e-12,
    )
    plan = phx.closure_data.LearnedStressBindingPlan(
        schema,
        output,
        resolved_filter,
        phx.equations.LESParameterProvenance(
            resolved_filter,
            space.prepared_id,
            "three-dimensional-periodic-unit-density",
            source_kind="user",
            evidence_ids=(),
        ),
        model_artifact_id="stress-model",
        normalizer_id=normalizer.normalizer_id,
    )

    def dummy_predictor(features, args):
        del args
        return jnp.zeros(features.shape[:-1] + (3, 3), dtype=features.dtype)

    binding = plan.prepare(
        dummy_predictor,
        normalizer,
        model_artifact_id="stress-model",
        target_id=output.target_id,
        output_units=output.units,
    )
    projector = phx.discretization.PeriodicLerayProjector(space)
    prepared = phx.equations.PeriodicLearnedStressPlan(binding).prepare(
        space,
        projector,
    )
    coordinates = phx.discretization.HermitianSpectralCoordinates(
        space,
        component_shape=(3,),
    )
    return space, prepared, coordinates


def _initial_state(space, prepared, coordinates):
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
    modal = prepared.projector.project(space.project(velocity))
    return coordinates.to_real_coordinates(modal)


def test_periodic_learned_stress_is_evaluated_inside_every_ssprk_stage():
    space, prepared, coordinates = _periodic_prepared()
    state = _initial_state(space, prepared, coordinates)
    layout = phx.dynamics.StateLayout((coordinates.coordinate_size,))

    def zero_base_rate(time, modal, inputs):
        del time, inputs
        return jnp.zeros_like(modal)

    transition = (
        phx.applications.incompressible_flow.PeriodicLearnedStressRolloutTransition(
            prepared,
            coordinates,
            zero_base_rate,
            base_rate_id="zero-periodic-base-rate",
            state_layout=layout,
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
    zero = _ViscosityStressModel(
        0.0,
        space.physical_shape,
        jnp.dtype(space.plan.precision.physical_dtype),
    )
    unchanged = transition.evaluate(
        zero,
        context,
        state,
        None,
        key=jax.random.key(0),
        iteration=jnp.asarray(0),
    )
    np.testing.assert_allclose(unchanged.accepted_state, state, atol=1e-6)
    assert bool(unchanged.training_usable)
    assert int(unchanged.iterations) == 3

    model = _ViscosityStressModel(
        0.1,
        space.physical_shape,
        jnp.dtype(space.plan.precision.physical_dtype),
    )

    def displacement(candidate):
        result = transition.evaluate(
            candidate,
            context,
            state,
            None,
            key=jax.random.key(1),
            iteration=jnp.asarray(0),
        )
        return jnp.sum(jnp.square(result.accepted_state - state))

    value, gradient = eqx.filter_value_and_grad(displacement)(model)
    assert float(value) > 0.0
    assert bool(jnp.isfinite(gradient.coefficient))
    assert float(jnp.abs(gradient.coefficient)) > 0.0

    system = transition.bind(model, system_id="periodic-learned-stress")
    deployed = system.evaluate(context, state)
    np.testing.assert_allclose(
        deployed,
        transition.evaluate(
            model,
            context,
            state,
            None,
            key=None,
            iteration=context.step_index,
        ).accepted_state,
    )
