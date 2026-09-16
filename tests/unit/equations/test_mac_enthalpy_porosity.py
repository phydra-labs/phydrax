#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _compiled(*, count=4, resistance=1.0e4):
    grid = phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformCellAxisSpec(count, periodic=True) for _ in range(2)
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))
    finite_volume = phx.discretization.FiniteVolumePlan(grid).prepare()
    operators = phx.discretization.MACOperatorPlan(finite_volume).prepare()
    momentum = phx.discretization.MACMomentumPlan(operators).prepare()
    projection = phx.solver.MACPressureProjectionPlan(
        operators, solve_method="transform", tolerance=1.0e-9
    )
    material = phx.equations.SolidLiquidEnthalpyPlan(
        1000.0,
        273.15,
        300.0,
        302.0,
        2000.0,
        2200.0,
        2.0e5,
        2.0,
        0.5,
        0.01,
        thermal_expansion=2.0e-4,
        mushy_resistance_coefficient=resistance,
    )
    boundaries = phx.discretization.MACThermalBoundarySet(operators)
    transport = phx.discretization.MACEnthalpyTransportPlan(
        operators, boundaries, advection="centered"
    ).prepare()
    problem = phx.equations.MACEnthalpyPorosityProblem(
        material, jnp.asarray([0.0, -9.81])
    )
    dynamics = phx.equations.compile_mac_enthalpy_porosity(
        phx.equations.IncompressibleFlowProblem(2, 0.01),
        problem,
        momentum,
        projection,
        transport,
    )
    return finite_volume, dynamics


def _taylor_green(discretization):
    x_faces = discretization.face_centers[0]
    y_faces = discretization.face_centers[1]
    return (
        jnp.sin(2.0 * jnp.pi * x_faces[..., 0]) * jnp.cos(2.0 * jnp.pi * x_faces[..., 1]),
        -jnp.cos(2.0 * jnp.pi * y_faces[..., 0])
        * jnp.sin(2.0 * jnp.pi * y_faces[..., 1]),
    )


def test_mac_enthalpy_uniform_liquid_closes_energy_and_projection_ledgers():
    finite_volume, dynamics = _compiled()
    velocity = _taylor_green(finite_volume)
    enthalpy = dynamics.problem.material.enthalpy_from_temperature(
        jnp.full(finite_volume.cell_shape, 310.0)
    )
    state = dynamics.pack_state(velocity, enthalpy)
    stage = dynamics.stage(0.0, state)
    diagnostics = dynamics.diagnostics_from_stage(stage)
    restriction = dynamics.step_restriction(0.0, state)

    assert stage.successful
    assert diagnostics.successful
    assert diagnostics.projection_converged
    assert diagnostics.enthalpy.successful
    assert jnp.abs(diagnostics.enthalpy.balance_defect) < 1.0e-8
    assert jnp.max(jnp.abs(stage.enthalpy_rate)) < 1.0e-6
    assert restriction.successful
    assert restriction.selected > 0.0


def test_mac_enthalpy_imex_resistance_damps_solid_and_rolls_state_atomically():
    finite_volume, dynamics = _compiled(resistance=1.0e3)
    velocity = _taylor_green(finite_volume)
    enthalpy = dynamics.problem.material.enthalpy_from_temperature(
        jnp.full(finite_volume.cell_shape, 290.0)
    )
    state = dynamics.pack_state(velocity, enthalpy)
    method = phx.solver.MACEnthalpyPorosityIMEXEulerMethod(
        dynamics, fixed_step_size=1.0e-4, maximum_iterations=100
    )
    result = method.step(0.0, state)

    assert result.accepted
    before = dynamics.momentum.operators.velocity_space.inner(velocity, velocity)
    after = dynamics.momentum.operators.velocity_space.inner(
        result.velocity, result.velocity
    )
    assert after < before
    np.testing.assert_allclose(result.enthalpy, enthalpy, rtol=1.0e-6, atol=1.0e-6)

    sbdf = phx.solver.MACEnthalpyPorositySBDF2Method(
        dynamics, 1.0e-4, maximum_iterations=100
    )
    startup = sbdf.initialize(0.0, state)
    assert startup.accepted
    second = sbdf.step(startup.state)
    assert second.accepted
    assert second.state.time > startup.state.time
    assert jnp.all(jnp.isfinite(second.state.state))


def test_mac_binary_alloy_uniform_solute_closes_conservative_transport():
    finite_volume, base = _compiled()
    scalar_problem = phx.discretization.MACScalarProblem(
        (
            phx.discretization.MACScalarTransport(
                "total_solute", 0.0, advection="centered"
            ),
        )
    )
    solute_transport = scalar_problem.prepare(base.momentum.operators)
    phase_diagram = phx.equations.BinaryAlloyPhaseDiagramPlan(
        1000.0,
        273.15,
        330.0,
        -50.0,
        0.3,
        1.5,
        2000.0,
        2.0e5,
        2.0,
        0.5,
        0.01,
        1.0e-10,
        1.0e-8,
        mushy_resistance_coefficient=1.0e3,
    )
    dynamics = phx.equations.compile_mac_binary_alloy(
        base, phase_diagram, solute_transport
    )
    velocity = tuple(jnp.zeros(layout.shape) for layout in finite_volume.face_layouts)
    concentration = jnp.full(finite_volume.cell_shape, 0.05)
    liquidus = (
        phase_diagram.melting_temperature + phase_diagram.liquidus_slope * concentration
    )
    temperature = liquidus + 2.0
    fraction = 0.5 * (
        1.0 + jnp.tanh((temperature - liquidus) / phase_diagram.smoothing_width)
    )
    enthalpy = (
        phase_diagram.reference_density
        * phase_diagram.heat_capacity
        * (temperature - phase_diagram.reference_temperature)
        + phase_diagram.reference_density * phase_diagram.latent_heat * fraction
    )
    state = dynamics.pack_state(velocity, enthalpy, concentration)
    stage = dynamics.stage(0.0, state)
    diagnostics = dynamics.diagnostics_from_stage(stage)

    assert stage.successful
    assert diagnostics.successful
    assert jnp.max(jnp.abs(stage.solute_rate)) < 1.0e-12
    assert jnp.abs(diagnostics.solute_balance_defect) < 1.0e-12
