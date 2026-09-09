#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.applications.thermofluids._topology_design import (
    ThermofluidMaterial,
    ThermofluidTopologyDesign,
)


def _drive(_time, velocity, _args):
    return jnp.ones_like(velocity[0]), jnp.zeros_like(velocity[1])


@pytest.fixture(scope="module")
def channel():
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(4, periodic=True),
            phx.discretization.UniformCellAxisSpec(4, periodic=False),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))
    finite_volume = phx.discretization.FiniteVolumePlan(grid).prepare()
    operators = phx.discretization.MACOperatorPlan(finite_volume).prepare()
    momentum = phx.discretization.MACMomentumPlan(operators).prepare()
    projection = phx.solver.MACPressureProjectionPlan(
        operators,
        solve_method="transform",
        tolerance=1e-9,
    )
    scalars = phx.discretization.MACScalarProblem(
        (phx.discretization.MACScalarTransport("temperature", 0.01),)
    )
    layout = phx.discretization.MACScalarLayout(operators, ("temperature",))
    cold = phx.discretization.MACScalarBoundaryCondition("dirichlet", 0.0)
    boundaries = phx.discretization.MACScalarBoundarySet(
        layout,
        walls={"temperature": {"y": (cold, cold)}},
    )
    dynamics = phx.equations.compile_mac_scalar_buoyancy(
        phx.equations.IncompressibleFlowProblem(
            2, 0.02, forcing=_drive, forcing_id="test-channel-drive"
        ),
        momentum,
        projection,
        scalars,
        scalars.prepare(operators, boundaries=boundaries),
        phx.equations.MACBuoyancyLaw([0.0, -1.0], {"temperature": -0.05}),
    )
    velocity = tuple(jnp.zeros(face.shape) for face in finite_volume.face_layouts)
    state = dynamics.pack_state(velocity, {"temperature": jnp.zeros((4, 4))})
    transform = phx.optim.DensityTransformPlan(
        phx.optim.ConicDensityFilterPlan(
            finite_volume.cell_centers.reshape((-1, 2)),
            0.28,
            jnp.ones((16,), dtype=bool),
            jnp.zeros((16,)),
            finite_volume.cell_volumes.reshape((-1,)),
        ),
        phx.optim.TanhDensityProjectionPlan(0.5),
    ).prepare()
    points = finite_volume.cell_centers
    source = 1.0 + jnp.exp(
        -40.0 * ((points[..., 0] - 0.3) ** 2 + (points[..., 1] - 0.5) ** 2)
    )
    material = ThermofluidMaterial(
        solid_resistance=12.0,
        fluid_conductivity=0.01,
        solid_conductivity=0.1,
        heat_capacity=2.0,
    )
    workflow = ThermofluidTopologyDesign(
        dynamics,
        transform,
        material,
        state,
        heat_source=source,
        final_time=0.1,
        step_size=0.01,
        maximum_solid_fraction=0.5,
    )
    return workflow, dynamics


def test_spatial_material_transport_conserves_heat_and_brinkman_work(channel):
    workflow, _ = channel
    fluid_density = jnp.zeros((4, 4))
    solid_density = jnp.ones((4, 4))
    fluid = workflow.integrate_density(fluid_density)
    solid = workflow.integrate_density(solid_density)
    fluid_evidence = workflow.evidence(fluid.final_state, fluid_density)
    solid_evidence = workflow.evidence(solid.final_state, solid_density)
    assert fluid.successful & solid.successful
    # Material substitution must not remove volumetric heating in the solid.
    expected_heating = jnp.sum(
        workflow.heat_source
        * workflow.dynamics.momentum.operators.discretization.cell_volumes
    )
    for evidence in (fluid_evidence, solid_evidence):
        assert evidence.successful
        assert float(evidence.heat_source_power) == pytest.approx(
            float(expected_heating), rel=1e-12
        )
        assert float(
            evidence.heat_content_rate + evidence.boundary_heat_outflow
        ) == pytest.approx(float(expected_heating), abs=1e-9)
        assert abs(float(evidence.thermal_balance_defect)) < 1e-9
        assert float(evidence.divergence_norm) < 1e-8
        assert abs(float(evidence.resistance_work_defect)) < 1e-10
        assert abs(float(evidence.kinetic_balance_defect)) < 1e-8
    assert float(solid_evidence.resistance_power) > 0.0
    assert float(fluid_evidence.resistance_power) == 0.0
    assert float(solid_evidence.kinetic_energy) < float(fluid_evidence.kinetic_energy)
    assert float(solid_evidence.boundary_heat_outflow) > float(
        fluid_evidence.boundary_heat_outflow
    )
    assert float(solid_evidence.heat_content) < float(fluid_evidence.heat_content)


def test_fixed_algorithm_gradient_and_stale_realization_rejection(channel):
    workflow, _ = channel
    design = jnp.linspace(0.2, 0.4, 16)
    direction = jnp.cos(jnp.arange(16, dtype=design.dtype))
    direction = direction / jnp.sqrt(jnp.sum(direction * direction))

    def objective(current):
        return workflow.objective(workflow.integrate(current).final_state, current)

    _, derivative = jax.jvp(objective, (design,), (direction,))
    step = 1e-4
    finite_difference = (
        objective(design + step * direction) - objective(design - step * direction)
    ) / (2.0 * step)
    assert abs(float(derivative)) > 1e-8
    assert float(derivative) == pytest.approx(
        float(finite_difference), rel=2e-3, abs=1e-8
    )
    problem = workflow.state_design_problem()
    accepted = problem.solve_state(design, workflow.initial_state)
    assert accepted.successful
    changed = jnp.full((16,), 0.8)
    stale_residual = problem.residual(accepted.state, changed)
    stale = problem.state_evidence(
        accepted.state,
        changed,
        stale_residual,
        phx.optim.OptimizationStatus.SUCCESS,
        reference_norm=1.0,
    )
    assert not stale.accepted
    binary = workflow.binary_reanalysis(design)
    direct = workflow.integrate_density(binary.density)
    np.testing.assert_allclose(binary.state, direct.final_state, atol=1e-12)
    assert bool(jnp.all((binary.density == 0.0) | (binary.density == 1.0)))


def test_invalid_material_and_unstable_realization_are_rejected(channel):
    workflow, dynamics = channel
    with pytest.raises(ValueError, match="positive"):
        ThermofluidMaterial(
            solid_resistance=10.0,
            fluid_conductivity=0.01,
            solid_conductivity=0.0,
            heat_capacity=1.0,
        )
    with pytest.raises(ValueError, match="stability bound"):
        ThermofluidTopologyDesign(
            dynamics,
            workflow.transform,
            workflow.material,
            workflow.initial_state,
            heat_source=workflow.heat_source,
            final_time=1.0,
            step_size=1.0,
            maximum_solid_fraction=0.5,
        )
    strong_dynamics = phx.equations.compile_mac_scalar_buoyancy(
        dynamics.flow_problem,
        dynamics.momentum,
        dynamics.projection,
        dynamics.scalar_problem,
        dynamics.transport,
        phx.equations.MACBuoyancyLaw(
            [0.0, -1.0],
            {"temperature": -1.0e8},
        ),
    )
    velocity, _ = strong_dynamics.unpack_state(workflow.initial_state)
    temperature = jnp.broadcast_to(jnp.linspace(0.0, 1.0, 4)[None, :], (4, 4))
    stratified_state = strong_dynamics.pack_state(
        velocity,
        {"temperature": temperature},
    )
    stratified = ThermofluidTopologyDesign(
        strong_dynamics,
        workflow.transform,
        workflow.material,
        stratified_state,
        heat_source=workflow.heat_source,
        final_time=0.1,
        step_size=0.01,
        maximum_solid_fraction=0.5,
    )
    with pytest.raises(RuntimeError, match="stability bound"):
        stratified.integrate(jnp.zeros((16,)))
