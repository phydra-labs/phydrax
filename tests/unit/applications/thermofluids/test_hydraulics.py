#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _reduced_pressure_drop(flow):
    return 10.0 * flow + 5.0 * flow**2


def _fluid():
    return phx.applications.thermofluids.HydraulicFluidProperties(
        density=1000.0,
        dynamic_viscosity=1.0e-3,
        vapor_pressure=2000.0,
        temperature=300.0,
        provenance="synthetic-water-like-test-fluid",
    )


def _initialize(process, initial_values):
    compilation = phx.dynamics.compile_acausal_dae(
        process.source, phx.dynamics.DAEStructuralPolicy(1, 0, tearing="none")
    )
    initial = jnp.asarray(
        tuple(
            initial_values.get(name, 0.0) for name in compilation.analysis.variable_names
        )
    )
    problem = phx.solver.DifferentialAlgebraicProblem(
        compilation.system,
        initial,
        initialization=phx.solver.DAEInitializationSpec.from_masks(
            compilation.fixed_state_mask, compilation.fixed_rate_mask
        ),
        problem_id=process.process_model_id,
    )
    dimension = compilation.system.state_shape[0]
    method = phx.nonlinear.NewtonKrylov(
        linear_policy=phx.linalg.LinearSolvePolicy(
            phx.linalg.GMRES(restart=dimension),
            tolerance=phx.linalg.TolerancePolicy(
                relative=1.0e-12, absolute=1.0e-13, max_steps=4 * dimension
            ),
        ),
        forcing_policy=phx.nonlinear.NewtonForcingPolicy("constant"),
    )
    result = phx.solver.initialize_dae(
        problem,
        0.0,
        policy=phx.solver.DAESolvePolicy(initialization_method=method),
    )
    assert bool(result.valid)
    return compilation.reconstruction(result.state, result.state_rate)


def test_circular_hydraulic_network_solves_pressure_drop_and_directed_flow():
    tf = phx.applications.thermofluids
    fluid = _fluid()
    law = tf.HydraulicChannelPlan.circular(
        fluid, radius=1.0e-3, length=0.1, maximum_reynolds_number=1000.0
    )
    source = tf.hydraulic_pressure_boundary_component("source", fluid, pressure=101425.0)
    channel = tf.hydraulic_channel_component("channel", law)
    sink = tf.hydraulic_pressure_boundary_component("sink", fluid, pressure=101325.0)
    process = tf.ThermofluidProcessPlan(
        (source, channel, sink),
        (
            tf.ThermofluidConnection("source", "hydraulic", "channel", "left"),
            tf.ThermofluidConnection("channel", "right", "sink", "hydraulic"),
        ),
    )
    expected = 100.0 / law.resistance
    jet = _initialize(
        process,
        {
            "source.pressure": 101425.0,
            "source.volume_flow": -expected,
            "channel.left_pressure": 101425.0,
            "channel.left_volume_flow": expected,
            "channel.right_pressure": 101325.0,
            "channel.right_volume_flow": -expected,
            "sink.pressure": 101325.0,
            "sink.volume_flow": expected,
        },
    )

    np.testing.assert_allclose(jet.value("channel.left_volume_flow"), expected)
    np.testing.assert_allclose(jet.value("channel.right_volume_flow"), -expected)
    np.testing.assert_allclose(jet.value("source.volume_flow"), -expected)
    evidence = law.evaluate(expected, 101425.0, 101325.0)
    assert bool(evidence.header.globally_eligible)
    unsupported = law.evaluate(1.0, 101425.0, 101325.0)
    assert not bool(unsupported.header.globally_eligible)
    assert bool(jnp.isnan(unsupported.pressure_drop))


def test_rectangular_resistance_is_orientation_symmetric_and_bounded():
    tf = phx.applications.thermofluids
    first = tf.HydraulicChannelPlan.rectangular(
        _fluid(), width=2.0e-3, height=1.0e-3, length=0.1, series_terms=32
    )
    second = tf.HydraulicChannelPlan.rectangular(
        _fluid(), width=1.0e-3, height=2.0e-3, length=0.1, series_terms=32
    )

    np.testing.assert_allclose(first.resistance, second.resistance)
    assert first.truncation_error_bound <= 1.0e-8


def test_compliance_and_inertance_encode_storage_and_momentum_equations():
    tf = phx.applications.thermofluids
    compliance = tf.hydraulic_compliance_component(
        "compliance",
        _fluid(),
        compliance=2.0,
        reference_pressure=101325.0,
    )
    compliance_jet = phx.dynamics.DAEJet(
        ("pressure", "volume_flow", "volume"),
        (
            (jnp.asarray(101325.0), jnp.asarray(3.0)),
            (jnp.asarray(6.0),),
            (jnp.asarray(0.0),),
        ),
    )
    np.testing.assert_allclose(
        compliance.dae_component.equations[0].residual(0.0, compliance_jet, None), 0.0
    )
    np.testing.assert_allclose(
        compliance.dae_component.equations[1].residual(0.0, compliance_jet, None),
        0.0,
    )

    inertance = tf.hydraulic_inertance_component("inertance", _fluid(), inertance=4.0)
    inertance_jet = phx.dynamics.DAEJet(
        (
            "left_pressure",
            "left_volume_flow",
            "right_pressure",
            "right_volume_flow",
        ),
        (
            (jnp.asarray(12.0),),
            (jnp.asarray(2.0), jnp.asarray(2.0)),
            (jnp.asarray(4.0),),
            (jnp.asarray(-2.0),),
        ),
    )
    np.testing.assert_allclose(
        inertance.dae_component.equations[0].residual(0.0, inertance_jet, None),
        0.0,
    )
    np.testing.assert_allclose(
        inertance.dae_component.equations[1].residual(0.0, inertance_jet, None),
        0.0,
    )


def test_calibrated_hydraulic_response_never_extrapolates():
    tf = phx.applications.thermofluids
    response = tf.MonotoneHydraulicResponsePlan(
        _fluid(),
        jnp.asarray((0.0, 1.0, 2.0)),
        jnp.asarray((0.0, 10.0, 30.0)),
        reverse_symmetric=True,
    )
    forward = response.evaluate(1.5, 101325.0, 101300.0)
    reverse = response.evaluate(-1.5, 101300.0, 101325.0)
    outside = response.evaluate(3.0, 101325.0, 101300.0)

    assert bool(forward.header.globally_eligible)
    np.testing.assert_allclose(reverse.pressure_drop, -forward.pressure_drop)
    assert not bool(outside.header.globally_eligible)
    assert bool(jnp.isnan(outside.pressure_drop))


def test_fixed_reduced_hydraulic_response_has_no_truth_fallback():
    tf = phx.applications.thermofluids
    response = tf.HydraulicReducedResponsePlan(
        _fluid(),
        _reduced_pressure_drop,
        minimum_flow=0.0,
        maximum_flow=2.0,
        artifact_id="fixed-reduced-law-artifact",
        reverse_symmetric=True,
    )
    inside = response.evaluate(1.5, 101325.0, 101300.0)
    outside = response.evaluate(2.5, 101325.0, 101300.0)

    assert bool(inside.header.globally_eligible)
    np.testing.assert_allclose(inside.pressure_drop, _reduced_pressure_drop(1.5))
    assert not bool(outside.header.globally_eligible)
    assert bool(jnp.isnan(outside.pressure_drop))
