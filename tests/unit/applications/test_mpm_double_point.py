#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.applications.solid_mechanics._mpm_double_point import (
    cell_to_faces,
    MixedMPMPressureOperator,
    MPMToMACFaceTransferPlan,
)
from phydrax.discretization.finite_volume import FaceVelocity
from phydrax.discretization.flip import FLIPParticleState, FLIPRuntimeState
from phydrax.equations import MaterialPointArguments
from phydrax.solver._mpm_double_point import (
    advance_double_point_mpm_step,
    DoublePointMPMPlan,
    DoublePointMPMState,
    DoublePointMPMStatus,
    project_double_point_velocity,
)


@dataclass(frozen=True)
class _Case:
    plan: DoublePointMPMPlan
    state: DoublePointMPMState
    arguments: MaterialPointArguments


def _case(
    *,
    solid_fraction: float = 0.6,
    maximum_iterations: int = 128,
    datum: bool = True,
    nonuniform_velocity: bool = False,
) -> _Case:
    bounds = jnp.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=jnp.float64)
    nodal = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformAxisSpec(4, periodic=True, endpoint=False),) * 2,
        axis_names=("x", "y"),
    ).prepare(bounds)
    cells = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(4, periodic=True),) * 2,
        axis_names=("x", "y"),
    ).prepare(bounds)
    mac = phx.discretization.MACOperatorPlan(
        phx.discretization.FiniteVolumePlan(cells).prepare()
    ).prepare()
    position = mac.discretization.cell_centers.reshape((-1, 2))
    volume = jnp.full((16,), 1.0 / 16.0, dtype=jnp.float64)
    solid_particles = phx.discretization.ParticleSetPlan(
        jnp.arange(16, dtype=jnp.int32), solid_fraction * volume, ambient_dimension=2
    ).prepare()
    splat = phx.discretization.ParticleGridSplatPlan(
        nodal, assignment=phx.discretization.TensorBSplineSplatAssignment(1)
    ).prepare(solid_particles)
    compiled = phx.equations.compile_material_point_problem(
        phx.equations.MaterialPointProblemIR(
            "double-point-test",
            phx.applications.solid_mechanics.NeoHookeanMPMConstitutivePlan(2),
        ),
        solid_particles,
        splat,
        phx.discretization.ExplicitMPMMethodPlan(),
        phx.discretization.MPMParticleDomainPlan(
            bounds, periodic=(True, True), support_margin=0.0
        ),
    )
    arguments = MaterialPointArguments(
        phx.applications.solid_mechanics.NeoHookeanParameters.from_shear_bulk(1.0, 4.0)
    )
    solid_velocity = jnp.broadcast_to(
        jnp.asarray([0.02, -0.01], dtype=jnp.float64), position.shape
    )
    fluid_velocity = jnp.broadcast_to(
        jnp.asarray([-0.01, 0.005], dtype=jnp.float64), position.shape
    )
    if nonuniform_velocity:
        solid_velocity = solid_velocity + jnp.stack(
            (
                0.01 * jnp.sin(2.0 * jnp.pi * position[:, 0]),
                0.007 * jnp.cos(4.0 * jnp.pi * position[:, 1]),
            ),
            axis=-1,
        )
        fluid_velocity = fluid_velocity + jnp.stack(
            (
                0.006 * jnp.cos(2.0 * jnp.pi * (position[:, 0] + position[:, 1])),
                0.004 * jnp.sin(2.0 * jnp.pi * position[:, 1]),
            ),
            axis=-1,
        )
    solid = compiled.initialize_state(position, solid_velocity, volume, arguments)
    fluid_particles = phx.discretization.ParticleSetPlan(
        jnp.arange(16, dtype=jnp.int32), 0.4 * volume, ambient_dimension=2
    ).prepare()
    fluid_transfer = phx.discretization.flip.FLIPParticleTransferPlan(mac).prepare(
        fluid_particles
    )
    transfer = MPMToMACFaceTransferPlan(
        compiled.dynamics, mac, maximum_routes=128
    ).prepare()
    pressure_mask = np.zeros((4, 4), dtype=np.bool_)
    pressure_mask[0, 0] = datum
    plan = DoublePointMPMPlan(
        compiled.dynamics,
        fluid_transfer,
        transfer,
        phx.equations.InterphaseDragLawPlan("stokes"),
        solid_density=1.0,
        fluid_density=1.0,
        fluid_viscosity=1.0,
        grain_diameter=1.0,
        pressure_mask=pressure_mask,
        tolerance=1.0e-9,
        maximum_iterations=maximum_iterations,
    )
    zero = jnp.asarray(0, dtype=jnp.int32)
    fluid = FLIPRuntimeState(
        FLIPParticleState(position, fluid_velocity),
        jnp.zeros((4, 4), dtype=jnp.float64),
        jnp.asarray(0.0, dtype=jnp.float64),
        zero,
        zero,
        zero,
        "",
    )
    return _Case(plan, plan.initialize(solid, fluid), arguments)


def _mixed(case: _Case, *, porosity: float = 0.4) -> MixedMPMPressureOperator:
    transfer = case.plan.solid_to_mac
    mac = transfer.operators
    n = cell_to_faces(mac, jnp.full((4, 4), porosity, dtype=jnp.float64))
    s = tuple(1.0 - value for value in n)
    solid_mass = jnp.full(
        transfer.solid.nodal_shape, (1.0 - porosity) / 16.0, dtype=jnp.float64
    )
    fluid_mass = tuple(
        value * measure for value, measure in zip(n, mac.face_dual_measures, strict=True)
    )
    return MixedMPMPressureOperator(transfer, s, n, solid_mass, fluid_mass)


def _assert_rollback(
    original: DoublePointMPMState, accepted: DoublePointMPMState
) -> None:
    before = jax.tree_util.tree_leaves(original)
    after = jax.tree_util.tree_leaves(accepted)
    assert len(before) == len(after)
    for committed, retained in zip(before, after, strict=True):
        np.testing.assert_array_equal(retained, committed)


def test_mixed_divergence_and_each_phase_force_are_exact_adjoints() -> None:
    case = _case()
    transfer = case.plan.solid_to_mac
    mac = transfer.operators
    pressure = jnp.arange(16, dtype=jnp.float64).reshape((4, 4)) / 7.0
    n = cell_to_faces(mac, 0.2 + 0.5 * pressure / jnp.max(pressure))
    solid_mass = jnp.linspace(0.01, 0.03, 16, dtype=jnp.float64).reshape((4, 4))
    solid_mass = solid_mass.at[1, 2].set(0.0)
    fluid_mass = tuple(
        0.1 + 0.01 * jnp.arange(value.size, dtype=jnp.float64).reshape(value.shape)
        for value in n
    )
    solid_mask = jnp.ones((4, 4, 2), dtype=jnp.bool_).at[0, :, 0].set(False)
    fluid_mask = tuple(
        jnp.ones(value.shape, dtype=jnp.bool_).at[2, 1].set(False) for value in n
    )
    mixed = MixedMPMPressureOperator(
        transfer,
        tuple(1.0 - value for value in n),
        n,
        solid_mass,
        fluid_mass,
        solid_active=solid_mask,
        fluid_active=fluid_mask,
    )
    solid = jnp.sin(jnp.arange(32, dtype=jnp.float64)).reshape((4, 4, 2))
    fluid = tuple(
        jnp.cos(jnp.arange(value.size, dtype=jnp.float64)).reshape(value.shape)
        for value in n
    )
    assert mixed.evidence.successful
    np.testing.assert_allclose(
        jnp.vdot(pressure, mixed.solid_divergence(solid)),
        jnp.vdot(mixed.solid_transpose(pressure), solid),
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        jnp.vdot(pressure, mixed.fluid_divergence(fluid)),
        sum(
            jnp.vdot(force, velocity)
            for force, velocity in zip(
                mixed.fluid_transpose(pressure), fluid, strict=True
            )
        ),
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    force_s = mixed.solid_transpose(pressure)
    force_f = mixed.fluid_transpose(pressure)
    energy = jnp.vdot(force_s, mixed.inverse_mass_s(force_s)) + sum(
        jnp.vdot(force, acceleration)
        for force, acceleration in zip(
            force_f, mixed.inverse_mass_f(force_f), strict=True
        )
    )
    np.testing.assert_allclose(
        jnp.vdot(pressure, mixed.pressure_action(pressure)), energy, rtol=2.0e-13
    )
    np.testing.assert_array_equal(force_s[0, :, 0], 0.0)
    np.testing.assert_array_equal(force_s[1, 2], 0.0)


def test_native_mixed_projection_corrects_both_phases_and_continuity() -> None:
    case = _case()
    mixed = _mixed(case)
    solid = jnp.sin(jnp.arange(32, dtype=jnp.float64)).reshape((4, 4, 2))
    fluid = tuple(
        jnp.cos(jnp.arange(value.size, dtype=jnp.float64)).reshape(value.shape)
        for value in mixed.fluid_mass
    )
    result = project_double_point_velocity(
        case.plan,
        mixed,
        solid,
        fluid,
        case.state.fluid.pressure,
        jnp.asarray(0.01, dtype=jnp.float64),
    )
    assert result.successful
    assert result.iterations > 0
    divergence = mixed.divergence(result.solid_velocity, result.fluid_velocity)
    scale = jnp.max(jnp.abs(mixed.divergence(solid, fluid)))
    np.testing.assert_allclose(
        jnp.where(case.plan.pressure_mask, 0.0, divergence),
        0.0,
        atol=2.0e-9 * float(scale),
    )
    assert not jnp.array_equal(result.solid_velocity, solid)
    assert any(
        not jnp.array_equal(after, before)
        for after, before in zip(result.fluid_velocity, fluid, strict=True)
    )
    np.testing.assert_array_equal(result.pressure[case.plan.pressure_mask], 0.0)


def test_coupled_stage_conserves_drag_momentum_and_uses_fluid_flip_increment() -> None:
    case = _case()
    dt = 0.001
    result = advance_double_point_mpm_step(
        case.plan, case.state, dt, arguments=case.arguments
    )
    assert result.successful
    assert result.status == DoublePointMPMStatus.SUCCESS
    assert result.projection.successful
    assert all(jnp.all(value.successful) for value in result.drag)
    np.testing.assert_allclose(result.drag_momentum_defect, 0.0, atol=2.0e-12)
    np.testing.assert_allclose(result.divergence_identity_defect, 0.0, atol=2.0e-12)
    np.testing.assert_allclose(
        jnp.where(case.plan.pressure_mask, 0.0, result.mixture_divergence),
        0.0,
        atol=2.0e-11,
    )
    vs = np.asarray([0.02, -0.01], dtype=np.float64)
    vf = np.asarray([-0.01, 0.005], dtype=np.float64)
    beta = 18.0 * 0.6
    slip = (vf - vs) / (1.0 + dt * beta * (1.0 / 0.6 + 1.0 / 0.4))
    barycenter = 0.6 * vs + 0.4 * vf
    expected_s = barycenter - 0.4 * slip
    expected_f = barycenter + 0.6 * slip
    np.testing.assert_allclose(
        result.accepted_state.solid.particles.velocity,
        np.broadcast_to(expected_s, (16, 2)),
        rtol=2.0e-9,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        result.accepted_state.fluid.particles.velocity,
        np.broadcast_to(expected_f, (16, 2)),
        rtol=2.0e-9,
        atol=2.0e-12,
    )
    assert result.accepted_state.solid.accepted_step == 1
    assert result.accepted_state.fluid.accepted_step == 1


def test_uniform_pore_pressure_is_not_committed_as_skeleton_material_stress() -> None:
    case = _case()
    pressure = jnp.full((4, 4), 2.0, dtype=jnp.float64)
    plan = eqx.tree_at(lambda value: value.pressure_values, case.plan, pressure)
    state = eqx.tree_at(
        lambda value: (
            value.solid.particles.velocity,
            value.fluid.particles.velocity,
            value.fluid.pressure,
        ),
        case.state,
        (
            jnp.zeros_like(case.state.solid.particles.velocity),
            jnp.zeros_like(case.state.fluid.particles.velocity),
            pressure,
        ),
    )
    result = advance_double_point_mpm_step(plan, state, 0.001, arguments=case.arguments)
    assert result.successful
    np.testing.assert_array_equal(result.accepted_state.fluid.pressure, pressure)
    np.testing.assert_allclose(result.effective_stress, 0.0, atol=1.0e-13)
    np.testing.assert_allclose(
        result.accepted_state.solid.particles.first_piola, 0.0, atol=1.0e-13
    )
    np.testing.assert_allclose(
        result.total_stress,
        jnp.broadcast_to(-2.0 * jnp.eye(2, dtype=jnp.float64), (16, 2, 2)),
        atol=1.0e-13,
    )


def test_porosity_uses_solid_assignment_and_conserves_intrinsic_volume() -> None:
    case = _case()
    source = case.plan.solid
    splat = phx.discretization.ParticleGridSplatPlan(
        source.splat.plan.target,
        assignment=phx.discretization.TensorBSplineSplatAssignment(2),
    ).prepare(source.particles)
    compiled = phx.equations.compile_material_point_problem(
        phx.equations.MaterialPointProblemIR("quadratic-solid-porosity", source.material),
        source.particles,
        splat,
        source.method,
        source.particle_domain,
    )
    mac = case.plan.fluid_transfer.plan.operators
    transfer = MPMToMACFaceTransferPlan(
        compiled.dynamics, mac, maximum_routes=128
    ).prepare()
    plan = DoublePointMPMPlan(
        compiled.dynamics,
        case.plan.fluid_transfer,
        transfer,
        case.plan.drag_law,
        solid_density=1.0,
        fluid_density=1.0,
        fluid_viscosity=1.0,
        grain_diameter=1.0,
        pressure_mask=case.plan.pressure_mask,
    )
    position = case.state.solid.particles.position.at[0, 0].add(0.07)
    solid = compiled.initialize_state(
        position,
        jnp.zeros_like(position),
        jnp.full((16,), 1.0 / 16.0, dtype=jnp.float64),
        case.arguments,
    )
    fluid = eqx.tree_at(
        lambda value: value.particles.velocity,
        case.state.fluid,
        jnp.zeros_like(case.state.fluid.particles.velocity),
    )
    result = advance_double_point_mpm_step(
        plan, plan.initialize(solid, fluid), 0.001, arguments=case.arguments
    )
    assert result.successful
    # Independent cardinal quadratic B-spline, centered on periodic MAC cells.
    centers = np.asarray(mac.discretization.cell_centers).reshape((-1, 2))
    distance = (np.asarray(position)[:, None, :] - centers[None, :, :]) / 0.25
    radius = np.abs((distance + 2.0) % 4.0 - 2.0)
    weight = np.where(
        radius < 0.5,
        0.75 - radius**2,
        np.where(radius < 1.5, 0.5 * (1.5 - radius) ** 2, 0.0),
    )
    expected_solid_fraction = 0.6 * np.sum(np.prod(weight, axis=-1), axis=0).reshape(
        (4, 4)
    )
    np.testing.assert_allclose(
        result.porosity, 1.0 - expected_solid_fraction, rtol=2e-14, atol=2e-14
    )
    np.testing.assert_allclose(
        jnp.sum(mac.discretization.cell_volumes * (1.0 - result.porosity)),
        0.6,
        rtol=2e-14,
    )


@pytest.mark.parametrize("porosity", [0.0, 1.0], ids=["fluid-absent", "solid-absent"])
def test_phase_absent_zero_mass_dofs_are_masked_without_mass_floors(
    porosity: float,
) -> None:
    case = _case()
    mixed = _mixed(case, porosity=porosity)
    pressure = jnp.arange(16, dtype=jnp.float64).reshape((4, 4))
    solid = jnp.ones((4, 4, 2), dtype=jnp.float64)
    fluid: FaceVelocity = tuple(jnp.ones_like(value) for value in mixed.fluid_mass)
    assert mixed.evidence.successful
    assert jnp.all(jnp.isfinite(mixed.pressure_action(pressure)))
    if porosity == 1.0:
        assert not jnp.any(mixed.solid_active)
        np.testing.assert_array_equal(mixed.solid_divergence(solid), 0.0)
        np.testing.assert_array_equal(mixed.solid_transpose(pressure), 0.0)
        np.testing.assert_array_equal(mixed.inverse_mass_s(solid), 0.0)
    else:
        assert all(not jnp.any(mask) for mask in mixed.fluid_active)
        np.testing.assert_array_equal(mixed.fluid_divergence(fluid), 0.0)
        for force, acceleration in zip(
            mixed.fluid_transpose(pressure), mixed.inverse_mass_f(fluid), strict=True
        ):
            np.testing.assert_array_equal(force, 0.0)
            np.testing.assert_array_equal(acceleration, 0.0)


def test_missing_physical_pressure_datum_is_refused() -> None:
    with pytest.raises(ValueError, match="pressure datum"):
        _case(datum=False)


def test_overfilled_solid_fraction_refuses_and_rolls_back_both_phases() -> None:
    case = _case(solid_fraction=1.2)
    result = advance_double_point_mpm_step(
        case.plan, case.state, 0.001, arguments=case.arguments
    )
    assert not result.successful
    assert result.status == DoublePointMPMStatus.INVALID_FRACTION
    assert jnp.any(result.porosity < 0.0)
    _assert_rollback(case.state, result.accepted_state)


def test_projection_iteration_exhaustion_rolls_back_both_phases() -> None:
    case = _case(maximum_iterations=1, nonuniform_velocity=True)
    # Exhaust only the pressure solve: drag must reach its actual native root.
    drag_policy = phx.linalg.LinearSolvePolicy(
        phx.linalg.PCG(),
        tolerance=phx.linalg.TolerancePolicy(
            relative=1.0e-9, absolute=0.0, max_steps=128
        ),
    )
    plan = eqx.tree_at(
        lambda value: value.prepared_drag,
        case.plan,
        phx.linalg.prepare(case.plan.prepared_drag.problem, drag_policy),
    )
    result = advance_double_point_mpm_step(
        plan, case.state, 0.001, arguments=case.arguments
    )
    assert not result.successful
    assert not result.projection.successful
    assert result.status == DoublePointMPMStatus.PROJECTION_FAILED
    _assert_rollback(case.state, result.accepted_state)


def test_corrected_material_deformation_refusal_rolls_back_both_phases() -> None:
    bounds = jnp.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=jnp.float64)
    nodal = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformAxisSpec(4, periodic=True, endpoint=False),) * 3,
        axis_names=("x", "y", "z"),
    ).prepare(bounds)
    cells = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(4, periodic=True),) * 3,
        axis_names=("x", "y", "z"),
    ).prepare(bounds)
    mac = phx.discretization.MACOperatorPlan(
        phx.discretization.FiniteVolumePlan(cells).prepare()
    ).prepare()
    position = mac.discretization.cell_centers.reshape((-1, 3))
    volume = jnp.full((64,), 1.0 / 64.0, dtype=jnp.float64)
    solid_particles = phx.discretization.ParticleSetPlan(
        jnp.arange(64, dtype=jnp.int32), 0.6 * volume, ambient_dimension=3
    ).prepare()
    splat = phx.discretization.ParticleGridSplatPlan(
        nodal, assignment=phx.discretization.TensorBSplineSplatAssignment(1)
    ).prepare(solid_particles)
    material = phx.applications.solid_mechanics.FiniteStrainJ2MPMConstitutivePlan()
    compiled = phx.equations.compile_material_point_problem(
        phx.equations.MaterialPointProblemIR("double-point-material-domain", material),
        solid_particles,
        splat,
        phx.discretization.ExplicitMPMMethodPlan(),
        phx.discretization.MPMParticleDomainPlan(
            bounds, periodic=(True, True, True), support_margin=0.0
        ),
    )
    arguments = MaterialPointArguments(
        phx.applications.solid_mechanics.FiniteStrainJ2Parameters(10.0, 30.0, 0.15, 2.0)
    )
    solid = compiled.initialize_state(
        position, jnp.zeros_like(position), volume, arguments
    )
    fluid_particles = phx.discretization.ParticleSetPlan(
        jnp.arange(64, dtype=jnp.int32), 0.4 * volume, ambient_dimension=3
    ).prepare()
    fluid_transfer = phx.discretization.flip.FLIPParticleTransferPlan(mac).prepare(
        fluid_particles
    )
    transfer = MPMToMACFaceTransferPlan(
        compiled.dynamics, mac, maximum_routes=768
    ).prepare()
    pressure_mask = np.zeros((4, 4, 4), dtype=np.bool_)
    pressure_mask[0, 0, 0] = True
    plan = DoublePointMPMPlan(
        compiled.dynamics,
        fluid_transfer,
        transfer,
        phx.equations.InterphaseDragLawPlan("stokes"),
        solid_density=1.0,
        fluid_density=1.0,
        fluid_viscosity=1.0,
        grain_diameter=1.0,
        pressure_mask=pressure_mask,
        tolerance=1.0e-9,
        maximum_iterations=128,
    )
    zero = jnp.asarray(0, dtype=jnp.int32)
    fluid = FLIPRuntimeState(
        FLIPParticleState(position, jnp.zeros_like(position)),
        jnp.zeros((4, 4, 4), dtype=jnp.float64),
        jnp.asarray(0.0, dtype=jnp.float64),
        zero,
        zero,
        zero,
        "",
    )
    valid = plan.initialize(solid, fluid)
    # A finite but non-SPD plastic metric is a real constitutive-domain error.
    # Saved accepted stress, positive-J deformation, positions and velocities
    # stay valid, so the predictor is admitted before the USL material update.
    invalid_history = eqx.tree_at(
        lambda state: state.solid.particles.material_state,
        valid,
        valid.solid.particles.material_state.at[0, 0].set(-1.0),
    )
    dt = 0.001
    predictor = plan.solid.predict_grid(invalid_history.solid, dt, arguments)
    assert predictor.successful
    result = advance_double_point_mpm_step(plan, invalid_history, dt, arguments=arguments)
    assert not result.successful
    assert result.projection.successful
    assert result.status == DoublePointMPMStatus.MATERIAL_FAILED
    assert not result.solid_step.successful
    _assert_rollback(invalid_history, result.accepted_state)


def test_sealed_pressure_component_without_datum_is_rejected() -> None:
    case = _case()
    mixed = _mixed(case, porosity=1.0)
    masks = (
        jnp.ones_like(mixed.fluid_mass[0], dtype=jnp.bool_)
        .at[2, 2]
        .set(False)
        .at[3, 2]
        .set(False),
        jnp.ones_like(mixed.fluid_mass[1], dtype=jnp.bool_)
        .at[2, 2]
        .set(False)
        .at[2, 3]
        .set(False),
    )
    mixed = MixedMPMPressureOperator(
        mixed.transfer,
        mixed.solid_fraction,
        mixed.fluid_fraction,
        mixed.solid_mass,
        mixed.fluid_mass,
        solid_active=jnp.zeros_like(mixed.solid_active),
        fluid_active=masks,
    )
    result = project_double_point_velocity(
        case.plan,
        mixed,
        jnp.zeros_like(mixed.solid_active, dtype=jnp.float64),
        tuple(jnp.zeros_like(value) for value in mixed.fluid_mass),
        case.state.fluid.pressure,
        jnp.asarray(0.001, dtype=jnp.float64),
    )
    assert not result.successful


def test_stability_predictor_rejection_is_not_material_failure() -> None:
    case = _case()
    result = advance_double_point_mpm_step(
        case.plan, case.state, 1.0, arguments=case.arguments
    )
    assert not result.successful
    assert result.status == int(DoublePointMPMStatus.PREDICTOR_FAILED)
    _assert_rollback(case.state, result.accepted_state)
