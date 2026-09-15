#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr

from phydrax.applications.sign_problem import (
    canonical_fugacity_transform,
    CanonicalFugacityPlan,
    complex_langevin_one_variable_controls,
    ComplexLangevinPlan,
    ComplexLangevinStatus,
    deform_holomorphic_quadrature,
    evaluate_fugacity_expansion,
    evaluate_imaginary_chemical_potential_reference,
    GaugeCoolingPlan,
    HolomorphicFlowQuadraturePlan,
    HolomorphicFlowStatus,
    integrate_holomorphic_flow_quadrature,
    prepare_canonical_fugacity,
    prepare_complex_langevin,
    prepare_gauge_cooling,
    prepare_holomorphic_flow_quadrature,
    sample_complex_langevin,
)


def test_canonical_fugacity_round_trip_from_imaginary_chemical_potential():
    sectors = jnp.asarray([0.25 + 0.0j, 1.0 + 0.0j, 0.25 + 0.0j])
    prepared = prepare_canonical_fugacity(
        CanonicalFugacityPlan(-1, 1, node_count=5, temperature=2.0)
    )

    def partition(mu):
        fugacity = jnp.exp(mu / 2.0)
        return sectors[0] / fugacity + sectors[1] + sectors[2] * fugacity

    reference = evaluate_imaginary_chemical_potential_reference(
        prepared, partition, grand_partition_id="symmetric-three-sector-reference"
    )
    result = canonical_fugacity_transform(prepared, reference)
    evaluated = evaluate_fugacity_expansion(
        prepared, result.canonical_sectors, jnp.asarray(0.3)
    )

    assert result.valid
    assert jnp.allclose(result.canonical_sectors, sectors, atol=2e-6)
    assert result.maximum_reconstruction_residual < 2e-6
    assert reference.conjugation_residual < 2e-6
    assert reference.periodicity_residual < 2e-6
    assert jnp.allclose(evaluated.grand_partition, partition(0.3), atol=2e-6)


def test_complex_langevin_gaussian_one_variable_controls():
    runtime = prepare_complex_langevin(
        ComplexLangevinPlan(
            num_steps=7000,
            step_size=0.004,
            burn_in=1000,
            thinning=3,
            tail_window=512,
            drift_tail_threshold=8.0,
            maximum_tail_probability=0.02,
        ),
        lambda z: 0.5 * jnp.sum(z**2),
        configuration_shape=(1,),
        action_id="unit-complex-gaussian",
    )
    result = sample_complex_langevin(runtime, jnp.asarray([0.0 + 0.0j]), key=jr.key(17))
    controls = complex_langevin_one_variable_controls(runtime, result, maximum_order=1)
    second_moment = jnp.mean(result.samples[:, 0] ** 2)

    assert result.diagnostics.status == int(ComplexLangevinStatus.SUCCESS)
    assert result.successful
    assert jnp.abs(second_moment - 1.0) < 0.18
    assert controls.finite
    assert controls.maximum_residual < 0.1


def test_complex_langevin_rejects_heavy_drift_tail():
    runtime = prepare_complex_langevin(
        ComplexLangevinPlan(
            num_steps=32,
            step_size=0.001,
            tail_window=16,
            drift_tail_threshold=0.1,
            maximum_tail_probability=0.0,
        ),
        lambda z: 25.0 * jnp.sum(z**2),
        configuration_shape=(1,),
        action_id="large-drift-control",
    )
    result = sample_complex_langevin(runtime, jnp.asarray([1.0 + 0.0j]), key=jr.key(21))

    assert result.diagnostics.status == int(ComplexLangevinStatus.DRIFT_TAIL_REJECTED)
    assert result.diagnostics.drift_tail_probability > 0.0
    assert not result.successful


def test_complex_langevin_applies_monotone_gauge_cooling():
    cooling = prepare_gauge_cooling(
        GaugeCoolingPlan(iterations=2, step_size=0.5),
        lambda z: 1j * jnp.imag(z),
        lambda z, delta: z + delta,
        lambda z: jnp.sum(jnp.imag(z) ** 2),
        configuration_shape=(1,),
        cooling_id="scalar-imaginary-gauge-orbit-control",
    )
    runtime = prepare_complex_langevin(
        ComplexLangevinPlan(
            num_steps=8,
            step_size=0.001,
            tail_window=4,
            drift_tail_threshold=10.0,
            maximum_tail_probability=1.0,
        ),
        lambda z: 0.5 * jnp.sum(z**2),
        configuration_shape=(1,),
        action_id="cooled-complex-gaussian",
        cooling=cooling,
    )
    result = sample_complex_langevin(runtime, jnp.asarray([0.0 + 1.0j]), key=jr.key(31))

    assert result.diagnostics.status == int(ComplexLangevinStatus.SUCCESS)
    assert result.diagnostics.cooling_accepted_updates == 16
    assert result.diagnostics.cooling_rejected_updates == 0
    assert jnp.all(
        result.diagnostics.cooling_final_norm <= result.diagnostics.cooling_initial_norm
    )


def test_holomorphic_flow_jacobian_and_residual_phase_are_consistent():
    flow_time = 0.2
    prepared = prepare_holomorphic_flow_quadrature(
        HolomorphicFlowQuadraturePlan(
            flow_time=flow_time,
            flow_steps=8,
            minimum_average_residual_phase=0.5,
            maximum_imaginary_action_drift=1e-5,
        ),
        lambda z: 0.5 * jnp.sum(z**2),
        jnp.asarray([[-1.0], [0.0], [1.0]]),
        jnp.ones((3,)),
        action_id="one-dimensional-gaussian-thimble",
    )
    geometry = deform_holomorphic_quadrature(prepared)
    result = integrate_holomorphic_flow_quadrature(
        prepared, geometry, lambda z: jnp.asarray(1.0 + 0.0j)
    )

    assert jnp.allclose(geometry.jacobians[:, 0, 0], jnp.exp(flow_time), rtol=2e-6)
    assert jnp.allclose(geometry.log_abs_determinant, flow_time, rtol=2e-6)
    assert geometry.maximum_imaginary_action_drift < 1e-6
    assert result.diagnostics.status == int(HolomorphicFlowStatus.SUCCESS)
    assert jnp.allclose(result.value, 1.0 + 0.0j)


def test_holomorphic_flow_abstains_on_residual_phase_cancellation():
    prepared = prepare_holomorphic_flow_quadrature(
        HolomorphicFlowQuadraturePlan(
            flow_time=0.0,
            flow_steps=1,
            minimum_average_residual_phase=0.5,
        ),
        lambda z: 10.0j * jnp.sum(z),
        jnp.asarray([[0.0], [jnp.pi / 10.0]]),
        jnp.ones((2,)),
        action_id="phase-cancellation-control",
    )
    geometry = deform_holomorphic_quadrature(prepared)
    result = integrate_holomorphic_flow_quadrature(
        prepared, geometry, lambda z: jnp.asarray(1.0 + 0.0j)
    )

    assert result.diagnostics.status == int(
        HolomorphicFlowStatus.INSUFFICIENT_RESIDUAL_PHASE
    )
    assert result.diagnostics.abstained
    assert not result.successful
    assert jnp.isnan(jnp.real(result.value))
