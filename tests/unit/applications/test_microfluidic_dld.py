#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _velocity(time, position, args):
    del time, args
    return jnp.broadcast_to(jnp.asarray((1.0, 0.0)), position.shape)


def _geometry():
    mf = phx.applications.microfluidics
    topology = mf.DLDTopology(
        row_count=4,
        column_count=1,
        period_rows=4,
        outlet_count=2,
        particle_capacity=2,
    )
    design = mf.DLDDesign(
        post_radius=0.1,
        axial_pitch=1.0,
        lateral_pitch=1.0,
        row_shift=0.25,
        channel_lower=0.0,
        channel_upper=3.0,
        first_row_x=1.0,
        outlet_x=5.0,
        depth=0.1,
        length_unit_id="mm",
    )
    return mf.DLDGeometryPlan(topology, design)


def _workflow():
    mf = phx.applications.microfluidics
    geometry = _geometry()
    particles = phx.discretization.ParticleSetPlan(
        jnp.asarray((0, 1)),
        jnp.asarray((1.0, 1.0)),
        ambient_dimension=2,
    ).prepare()
    population = phx.discretization.ParticlePopulationPlan(particles)
    properties = phx.discretization.FiniteParticleProperties(
        jnp.asarray((0.05, 0.05)),
        jnp.ones((2,)),
        jnp.ones((2,)),
        jnp.zeros((2, 2, 2)),
    )
    units = phx.discretization.FiniteParticleTransportUnits(
        length_unit_id="mm",
        time_unit_id="s",
        mass_unit_id="kg",
        temperature_unit_id="K",
        frame="device",
    )
    field = phx.discretization.FiniteParticleVelocityFieldPlan(
        _velocity,
        provider_id="synthetic-dld-flow",
        velocity_unit_id="mm/s",
        frame="device",
    )
    transport = phx.solver.FiniteParticleTransportPlan(
        population,
        properties,
        units,
        field,
        geometry,
        motion=phx.discretization.FiniteParticleMotionKind.OVERDAMPED_STOKES,
    )
    outlets = mf.DLDOutletPlan(5.0, jnp.asarray((0.0, 1.5, 3.0)))
    metrics = mf.DLDMetricPlan(2, 2, jnp.asarray((0, 1)))
    screen = mf.DLDEmpiricalScreenPlan(
        coefficient=1.4,
        exponent=0.48,
        minimum_shift_fraction=0.1,
        maximum_shift_fraction=0.4,
        source_id="primary-equation-test-fixture",
    )
    workflow = mf.DLDWorkflowPlan(
        geometry,
        transport,
        outlets,
        metrics,
        jnp.asarray((0, 1)),
        step_count=6,
        step_size=1.0,
        flow_model_id="synthetic-dld-flow",
        screening=screen,
    )
    state = transport.initialize(
        population.initialize(),
        jnp.asarray(((0.0, 1.2), (0.0, 2.0))),
        jnp.zeros((2, 2)),
        jax.random.key(3),
    )
    flow = phx.AdmissibilityHeader(
        jnp.asarray(1.0),
        jnp.asarray(0, dtype=jnp.uint32),
        "synthetic-dld-flow",
        "synthetic-flow-evidence",
    )
    return workflow, state, flow


def test_dld_workflow_rejects_classifier_on_another_outlet_plane():
    workflow, _, _ = _workflow()
    misplaced = phx.applications.microfluidics.DLDOutletPlan(
        4.5, workflow.outlets.transverse_edges
    )
    with pytest.raises(ValueError, match="mismatch"):
        phx.applications.microfluidics.DLDWorkflowPlan(
            workflow.geometry,
            workflow.transport,
            misplaced,
            workflow.metrics,
            workflow.particle_classes,
            step_count=workflow.step_count,
            step_size=workflow.step_size,
            flow_model_id=workflow.flow_model_id,
            screening=workflow.screening,
        )


def test_dld_geometry_has_exact_post_wall_clearance_and_periodic_shift():
    geometry = _geometry()
    np.testing.assert_allclose(
        geometry.post_centers,
        ((1.0, 0.1), (2.0, 0.35), (3.0, 0.6), (4.0, 0.85)),
    )
    evaluation = geometry.evaluate(
        jnp.asarray(((1.0, 0.4), (0.0, 1.4))),
        jnp.asarray((0.05, 0.05)),
    )
    np.testing.assert_allclose(evaluation.clearance, (0.15, 1.35))
    assert bool(evaluation.header.globally_eligible)


def test_dld_workflow_runs_particles_to_disjoint_outlets_and_reports_metrics():
    workflow, state, flow = _workflow()
    result = workflow.run(
        state, flow, volume_flow=jnp.asarray(2.0), pressure_drop=jnp.asarray(4.0)
    )

    assert bool(result.successful)
    np.testing.assert_array_equal(result.final_state.terminal_code, (0, 1))
    np.testing.assert_allclose(result.metrics.transfer_matrix, np.eye(2))
    np.testing.assert_allclose(result.metrics.purity, 1.0)
    np.testing.assert_allclose(result.metrics.recovery, 1.0)
    np.testing.assert_allclose(result.metrics.throughput_proxy, 2.0)
    np.testing.assert_allclose(result.metrics.hydraulic_resistance, 2.0)
    assert bool(result.screening.header.globally_eligible)
    assert result.screening.critical_diameter > 0.0


def test_dld_robustness_retains_invalid_samples_and_refuses_claim():
    plan = phx.applications.microfluidics.DLDRobustnessPlan(
        tail_fraction=0.5, required_value=0.8, maximize=True
    )
    result = plan.evaluate(
        jnp.asarray((0.9, 0.7, 0.95)),
        jnp.asarray((True, False, True)),
    )

    assert int(result.invalid_sample_count) == 1
    assert not bool(result.header.globally_eligible)
    assert result.nominal == 0.9


def test_dld_lbm_adapter_runs_bound_flow_and_checks_operating_envelope():
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(4, periodic=True),
            phx.discretization.UniformCellAxisSpec(4, periodic=True),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray(((0.0, 0.0), (1.0, 1.0))))
    lattice = phx.discretization.D2Q9()
    precision = phx.discretization.LatticeBoltzmannPrecisionPolicy()
    discretization = phx.discretization.LatticeBoltzmannPlan(
        grid, lattice, precision=precision
    ).prepare()
    collision = phx.discretization.BGKCollisionPlan()
    compiled = phx.equations.compile_lattice_boltzmann_problem(
        phx.equations.LatticeBoltzmannProblem("dld-flow-test", 2),
        discretization,
        phx.discretization.LatticeBoltzmannMethodPlan(collision),
        phx.discretization.LatticeBoltzmannBoundaryPlan(),
        time_step=0.01,
    )
    hardware = phx.discretization.LatticeBoltzmannHardwareTarget(
        "cpu",
        "test",
        "test",
        maximum_device_bytes=1024**3,
    )
    envelope = phx.discretization.LatticeBoltzmannOperatingEnvelopePlan(
        lattice,
        collision,
        None,
        precision,
        hardware,
        physics_model="athermal-single-phase",
        boundary_model="periodic-test",
        relaxation_rate_limits=(0.01, 1.99),
        maximum_mach_number=0.2,
        maximum_knudsen_number=0.1,
        density_limits=(0.5, 1.5),
        maximum_density_ratio=3.0,
        maximum_force_number=0.1,
        minimum_wall_resolution_cells=0.0,
        maximum_relative_mass_drift=1.0e-10,
    )
    parameters = phx.discretization.LatticeBoltzmannRuntimeParameters(0.01)
    initial = compiled.initialize_state(1.0, jnp.asarray((0.01, 0.0)), parameters)
    result = phx.applications.microfluidics.DLDLatticeBoltzmannFlowPlan(
        compiled.dynamics,
        envelope,
        maximum_steps=2,
        steady_tolerance=1.0e-12,
        knudsen_number=0.0,
        force_number=0.0,
        wall_resolution_cells=1.0,
        geometry_id="periodic-flow-test",
    ).solve(initial, parameters)

    assert bool(result.header.globally_eligible)
    np.testing.assert_allclose(result.velocity[..., 0], 0.01, atol=1.0e-12)
    np.testing.assert_allclose(result.velocity[..., 1], 0.0, atol=1.0e-12)
