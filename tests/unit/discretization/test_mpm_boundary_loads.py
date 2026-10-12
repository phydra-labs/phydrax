#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Canonical MPM loads: independent weak equilibria and dynamic impedance.

The column oracle is constant-stress equilibrium, not relaxation to a fitted
answer. The cylinder oracle is Lamé's tension-positive thick-wall solution.
The wave oracle measures separated incident/reflected pulses, rather than
checking a dashpot coefficient or merely observing declining total energy.
"""

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax as phx
from phydrax.applications.solid_mechanics import (
    AbstractMechanicalLoad,
    CurrentSurfaceTraction,
    GeneralFollowerLoad,
    MechanicalLoadState,
    ReferenceDeadTraction,
)
from phydrax.discretization.mpm import (
    build_apic_route_payload,
    LysmerKuhlemeyerDashpot,
    MPMBoundaryState,
    MPMVolumetricStabilizationKind,
    PreparedMPMBoundaryLoad,
)
from phydrax.integration import DeformedMeasurePlan, DeformedMeasureState


def _pressure(
    reference: Array,
    current: Array,
    measure: DeformedMeasureState,
    state: MechanicalLoadState,
    args: Array | None,
) -> Array:
    del reference, current, args
    return -state.parameter * measure.normal("current")


def _position_traction(position: Array, time: Array, args: Array | None) -> Array:
    del time
    scale = jnp.asarray(1.0, dtype=position.dtype) if args is None else args
    return scale * jnp.stack((0.3 + position[:, 1], -0.2 * position[:, 0]), axis=-1)


def _column_traction(position: Array, time: Array, args: Array | None) -> Array:
    del time, args
    return jnp.stack(
        (jnp.zeros_like(position[:, 0]), jnp.where(position[:, 1] > 0.75, 2.0, -2.0)),
        axis=-1,
    )


def _periodic_problem(
    *,
    sparse: bool = False,
    multifield: bool = False,
    kind: MPMVolumetricStabilizationKind = "none",
    load: AbstractMechanicalLoad | None = None,
    damping: bool = False,
    include_boundary: bool = True,
) -> tuple[
    phx.equations.CompiledMaterialPointProblem,
    phx.equations.MaterialPointArguments,
    Array,
]:
    bounds = jnp.asarray(((0.0, 0.0), (1.0, 1.0)), dtype=jnp.float64)
    grid_plan = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformAxisSpec(8, periodic=True, endpoint=False),) * 2,
        axis_names=("x", "y"),
    )
    grid = grid_plan.prepare(bounds)
    position = jnp.asarray(
        ((0.31, 0.36), (0.42, 0.48), (0.31, 0.36), (0.42, 0.48)), dtype=jnp.float64
    )
    volume = jnp.full((4,), 0.02, dtype=jnp.float64)
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(4, dtype=jnp.int32), volume, ambient_dimension=2
    ).prepare()
    splat = phx.discretization.ParticleGridSplatPlan(
        grid, assignment=phx.discretization.TensorBSplineSplatAssignment(2)
    ).prepare(particles)
    boundary = PreparedMPMBoundaryLoad(
        splat,
        position[:2],
        jnp.asarray(((0.0, 0.07), (0.0, 0.11)), dtype=jnp.float64),
        ReferenceDeadTraction(jnp.asarray((0.12, -0.08), dtype=jnp.float64))
        if load is None
        else load,
        field_slot=1 if multifield else 0,
    )
    area = jnp.zeros(grid.vertices().shape + (2,), dtype=jnp.float64)
    area = area.at[3, 3, 0].set(0.04).at[3, 4, 0].set(0.03)
    dashpots = (
        (LysmerKuhlemeyerDashpot(area, 1.0, 3.0, 1.0, field_slot=1 if multifield else 0),)
        if damping
        else ()
    )
    storage = None
    if sparse:
        index = grid_plan.prepare_index_space(bounds)
        storage = phx.discretization.BlockSparseMPMNodalStoragePlan(
            phx.discretization.SparseBlockTopologyPlan(
                index, (4, 4), 4, layout=index.vertices()
            )
        )
    fields = (
        phx.discretization.MPMNodalFieldPlan(
            ("unloaded", "loaded"), jnp.asarray((0, 0, 1, 1), dtype=jnp.int32)
        )
        if multifield
        else None
    )
    compiled = phx.equations.compile_material_point_problem(
        phx.equations.MaterialPointProblemIR(
            "boundary-acceptance",
            phx.applications.solid_mechanics.NeoHookeanMPMConstitutivePlan(2),
            boundary_loads=(boundary,) if include_boundary else (),
            dashpots=dashpots,
        ),
        particles,
        splat,
        phx.discretization.ExplicitMPMMethodPlan(
            volumetric_stabilization=phx.discretization.MPMVolumetricStabilizationPlan(
                kind
            )
        ),
        phx.discretization.MPMParticleDomainPlan(
            bounds, periodic=(True, True), support_margin=0.0
        ),
        nodal_storage=storage,
        nodal_fields=fields,
    )
    arguments = phx.equations.MaterialPointArguments(
        phx.applications.solid_mechanics.NeoHookeanParameters.from_shear_bulk(2.0, 8.0),
        mechanical_load_states=(MechanicalLoadState(1.4),) if include_boundary else (),
    )
    return compiled, arguments, position


@pytest.mark.parametrize(
    "current_measure", (False, True), ids=("dead-reference", "current-surface")
)
@pytest.mark.strict_jax
def test_canonical_evaluation_force_impulse_work_and_advected_boundary(
    current_measure: bool,
) -> None:
    load = (
        CurrentSurfaceTraction(jnp.asarray((0.4, -0.7), dtype=jnp.float64))
        if current_measure
        else ReferenceDeadTraction(jnp.asarray((0.4, -0.7), dtype=jnp.float64))
    )
    compiled, arguments, _ = _periodic_problem(load=load)
    adapter = compiled.dynamics.boundary_loads[0]
    initial = adapter.initialize_state()
    deformation = jnp.broadcast_to(
        jnp.asarray(((1.2, 0.1), (0.0, 0.8)), dtype=jnp.float64),
        initial.deformation_gradient.shape,
    )
    state = MPMBoundaryState(initial.position + 0.03, deformation)
    routes = adapter.build_routes(state)
    velocity = jnp.broadcast_to(
        jnp.asarray((0.06, -0.02), dtype=jnp.float64), (compiled.dynamics.grid_count, 2)
    )
    dt = jnp.asarray(0.002, dtype=jnp.float64)
    result = adapter.evaluate(
        state, routes, velocity, dt, arguments.mechanical_load_states[0]
    )
    area = jnp.asarray((0.07, 0.11), dtype=jnp.float64)
    measure = DeformedMeasurePlan(
        "surface", area, reference_normal=jnp.asarray((0.0, 1.0), dtype=jnp.float64)
    ).evaluate(deformation)
    canonical = load.evaluate(
        initial.position, state.position, measure, arguments.mechanical_load_states[0]
    )
    expected_area = np.asarray((0.07, 0.11)) * (1.2 if current_measure else 1.0)
    expected_force = expected_area.sum() * 1.4 * np.asarray((0.4, -0.7))

    assert bool(result.successful) and bool(canonical.valid)
    np.testing.assert_allclose(
        result.evaluation.total_force_density,
        canonical.total_force_density,
        rtol=2e-13,
        atol=2e-13,
    )
    np.testing.assert_allclose(
        np.sum(result.force, axis=0), expected_force, rtol=2e-13, atol=2e-13
    )
    np.testing.assert_allclose(result.impulse, dt * result.force, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(
        result.work,
        dt * np.dot(expected_force, np.asarray((0.06, -0.02))),
        rtol=2e-13,
        atol=2e-15,
    )
    np.testing.assert_allclose(
        result.candidate_state.position,
        state.position + dt * jnp.asarray(((0.06, -0.02),), dtype=jnp.float64),
        rtol=2e-13,
        atol=2e-13,
    )
    np.testing.assert_allclose(
        result.candidate_state.deformation_gradient, deformation, rtol=2e-13, atol=2e-13
    )
    assert float(result.dissipation) == 0.0
    if current_measure:
        assert result.potential is None
    else:
        assert result.potential is not None
        expected_potential = -np.dot(
            expected_force, np.asarray((0.03 + 0.002 * 0.06, 0.03 - 0.002 * 0.02))
        )
        np.testing.assert_allclose(
            result.potential, expected_potential, rtol=3e-12, atol=3e-15
        )


@pytest.mark.strict_jax
def test_follower_pressure_uses_nanson_normal_and_advances_affine_boundary_geometry() -> (
    None
):
    load = GeneralFollowerLoad(
        _pressure, support="boundary", measure_frame="current", load_id="tilted-pressure"
    )
    compiled, arguments, _ = _periodic_problem(load=load)
    adapter = compiled.dynamics.boundary_loads[0]
    initial = adapter.initialize_state()
    deformation = jnp.asarray(((1.2, 0.1), (0.12, 0.8)), dtype=jnp.float64)
    state = MPMBoundaryState(
        initial.position,
        jnp.broadcast_to(deformation, initial.deformation_gradient.shape),
    )
    gradient = jnp.asarray(((0.03, -0.02), (0.01, 0.04)), dtype=jnp.float64)
    offset = jnp.asarray((0.05, -0.01), dtype=jnp.float64)
    velocity = compiled.dynamics.grid_coordinates @ gradient.T + offset[None, :]
    dt = jnp.asarray(0.003, dtype=jnp.float64)
    result = adapter.evaluate(
        state,
        adapter.build_routes(state),
        velocity,
        dt,
        arguments.mechanical_load_states[0],
    )
    # For A0=(0,A), cof(F) A0=A*(-F_21,F_11), independently
    # of the unit-normal normalization and current-area calculation.
    trial_deformation = (jnp.eye(2, dtype=jnp.float64) + dt * gradient) @ deformation
    expected_force = (
        -1.4 * 0.18 * np.asarray((-trial_deformation[1, 0], trial_deformation[0, 0]))
    )

    assert bool(result.successful)
    np.testing.assert_allclose(
        jnp.sum(result.force, axis=0), expected_force, rtol=3e-12, atol=3e-13
    )
    np.testing.assert_allclose(
        result.candidate_state.position,
        state.position + dt * (state.position @ gradient.T + offset[None, :]),
        rtol=3e-12,
        atol=3e-13,
    )
    np.testing.assert_allclose(
        result.candidate_state.deformation_gradient,
        jnp.broadcast_to(trial_deformation, initial.deformation_gradient.shape),
        rtol=3e-12,
        atol=3e-13,
    )


@pytest.mark.strict_jax
def test_explicit_dashpot_midpoint_work_equals_grid_kinetic_energy_change() -> None:
    compiled, arguments, position = _periodic_problem(
        damping=True, include_boundary=False
    )
    velocity = jnp.broadcast_to(
        jnp.asarray((0.04, -0.02), dtype=jnp.float64), position.shape
    )
    state = compiled.initialize_state(
        position, velocity, jnp.full((4,), 0.02, dtype=jnp.float64), arguments
    )
    result = compiled.dynamics.step_detailed(state, 0.002, arguments)

    assert bool(result.successful)
    ledger = result.diagnostics.dashpots[0]
    change = (
        result.diagnostics.energy.grid_kinetic_after
        - result.diagnostics.energy.grid_kinetic_before
    )
    assert float(ledger.dissipation) > 0.0
    np.testing.assert_allclose(ledger.work, change, rtol=3e-11, atol=3e-15)
    np.testing.assert_allclose(ledger.dissipation, -ledger.work, rtol=3e-12, atol=3e-15)


@pytest.mark.strict_jax
def test_surface_particle_and_explicit_quadrature_preparation_are_equivalent() -> None:
    compiled, arguments, position = _periodic_problem()
    original = compiled.dynamics.boundary_loads[0]
    particles = PreparedMPMBoundaryLoad.from_surface_particles(
        compiled.dynamics.splat,
        jnp.asarray((0, 1), dtype=jnp.int32),
        position,
        jnp.asarray(((0.0, 0.07), (0.0, 0.11)), dtype=jnp.float64),
        original.load,
    )
    velocity = (
        jnp.arange(compiled.dynamics.grid_count * 2, dtype=jnp.float64).reshape(-1, 2)
        * 0.0001
    )
    left_state, right_state = original.initialize_state(), particles.initialize_state()
    left = original.evaluate(
        left_state,
        original.build_routes(left_state),
        velocity,
        0.001,
        arguments.mechanical_load_states[0],
    )
    right = particles.evaluate(
        right_state,
        particles.build_routes(right_state),
        velocity,
        0.001,
        arguments.mechanical_load_states[0],
    )

    assert bool(left.successful & right.successful)
    for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        np.testing.assert_allclose(a, b, rtol=2e-13, atol=2e-13)


@pytest.mark.strict_jax
def test_column_constant_stress_balances_surface_traction_at_every_grid_node() -> None:
    """σ_yy = T = 2, σ_xx = σ_xy = 0; Q1 midpoint integration is exact."""
    h = 0.125
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformAxisSpec(9, periodic=False, endpoint=True),
            phx.discretization.UniformAxisSpec(13, periodic=False, endpoint=True),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray(((0.0, 0.0), (1.0, 1.5)), dtype=jnp.float64))
    x, y = np.meshgrid(
        np.arange(0.25 + h / 2, 0.75, h), np.arange(0.25 + h / 2, 1.25, h), indexing="ij"
    )
    position = jnp.asarray(np.stack((x.ravel(), y.ravel()), axis=-1), dtype=jnp.float64)
    volume = jnp.full((position.shape[0],), h * h, dtype=jnp.float64)
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(position.shape[0], dtype=jnp.int32), volume, ambient_dimension=2
    ).prepare()
    splat = phx.discretization.ParticleGridSplatPlan(
        grid,
        assignment=phx.discretization.TensorBSplineSplatAssignment(1),
        boundary="reject",
    ).prepare(particles)
    boundary_position = jnp.asarray(
        np.concatenate(
            (
                np.stack((x[:, 0], np.full(x.shape[0], 1.25)), axis=-1),
                np.stack((x[:, 0], np.full(x.shape[0], 0.25)), axis=-1),
            )
        ),
        dtype=jnp.float64,
    )
    area = jnp.asarray(
        np.concatenate(
            (np.tile((0.0, h), (x.shape[0], 1)), np.tile((0.0, -h), (x.shape[0], 1)))
        ),
        dtype=jnp.float64,
    )
    adapter = PreparedMPMBoundaryLoad(
        splat,
        boundary_position,
        area,
        ReferenceDeadTraction(_column_traction, load_id="column-analytic-traction"),
    )
    state = adapter.initialize_state()
    stress = jnp.broadcast_to(
        jnp.asarray(((0.0, 0.0), (0.0, 2.0)), dtype=jnp.float64),
        (position.shape[0], 2, 2),
    )
    payload = build_apic_route_payload(
        splat.build(position),
        volume,
        jnp.zeros_like(position),
        jnp.zeros_like(stress),
        volume,
        stress,
        jnp.broadcast_to(jnp.eye(2, dtype=jnp.float64), stress.shape),
        jnp.zeros_like(position),
        jnp.ones((position.shape[0],), dtype=jnp.bool_),
    )
    internal = (
        splat.scatter_route_payload(splat.build(position), payload)
        .values[..., 2:4]
        .reshape(-1, 2)
    )
    result = adapter.evaluate(
        state,
        adapter.build_routes(state),
        jnp.zeros_like(internal),
        0.001,
        MechanicalLoadState(),
    )

    assert bool(result.successful)
    assert float(jnp.linalg.norm(result.force)) > 0.1
    np.testing.assert_allclose(internal + result.force, 0.0, rtol=0.0, atol=3e-13)
    top = np.asarray(grid.points)[:, 1] == 1.25
    np.testing.assert_allclose(
        np.sum(np.asarray(result.force)[top, 1]) / 0.5, 2.0, rtol=2e-13
    )


@pytest.mark.parametrize("compact", (False, True), ids=("dense", "logical-node-subset"))
@pytest.mark.strict_jax
def test_dashpot_matches_independent_normal_tangential_impedance_and_work(
    compact: bool,
) -> None:
    area = jnp.asarray(((0.6, 0.8), (-0.3, 0.4), (0.0, 0.0)), dtype=jnp.float64)
    velocity = jnp.asarray(((0.2, -0.1), (-0.3, 0.4), (2.0, -1.0)), dtype=jnp.float64)
    dashpot = LysmerKuhlemeyerDashpot(area, 2.0, 3.0, 1.0)
    ids = jnp.asarray((1, 0, 2), dtype=jnp.int32) if compact else None
    valid = jnp.asarray((True, True, False), dtype=jnp.bool_) if compact else None
    result = dashpot.evaluate(velocity, 0.004, logical_node_ids=ids, node_valid=valid)
    selected = np.asarray(area)[np.asarray(ids)] if ids is not None else np.asarray(area)
    magnitude = np.linalg.norm(selected, axis=-1)
    normal = np.zeros_like(selected)
    np.divide(selected, magnitude[:, None], out=normal, where=magnitude[:, None] > 0.0)
    normal_velocity = np.sum(np.asarray(velocity) * normal, axis=-1)[:, None] * normal
    expected = (
        -2.0
        * magnitude[:, None]
        * (3.0 * normal_velocity + np.asarray(velocity) - normal_velocity)
    )

    assert bool(result.successful)
    np.testing.assert_allclose(result.force, expected, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(result.impulse, 0.004 * expected, rtol=2e-13, atol=2e-13)
    expected_work = 0.004 * np.sum(expected * np.asarray(velocity))
    np.testing.assert_allclose(result.work, expected_work, rtol=2e-13, atol=2e-15)
    np.testing.assert_allclose(result.dissipation, -expected_work, rtol=2e-13, atol=2e-15)
    assert float(result.work) < 0.0


@pytest.mark.parametrize(
    "kind", ("none", "b_bar", "f_bar"), ids=("ordinary", "b-bar", "f-bar")
)
@pytest.mark.parametrize("multifield", (False, True), ids=("single-field", "two-fields"))
def test_loaded_dense_sparse_steps_agree_and_preserve_field_ownership(
    kind: MPMVolumetricStabilizationKind, multifield: bool
) -> None:
    dense, arguments, position = _periodic_problem(
        kind=kind, multifield=multifield, damping=True
    )
    sparse, sparse_arguments, _ = _periodic_problem(
        kind=kind, multifield=multifield, sparse=True, damping=True
    )
    volume = jnp.full((4,), 0.02, dtype=jnp.float64)
    initial = dense.initialize_state(
        position, jnp.zeros_like(position), volume, arguments
    )
    sparse_initial = sparse.initialize_state(
        position, jnp.zeros_like(position), volume, sparse_arguments
    )
    left = dense.dynamics.step_detailed(initial, 0.001, arguments)
    right = sparse.dynamics.step_detailed(sparse_initial, 0.001, sparse_arguments)

    assert bool(left.successful & right.successful)
    for a, b in zip(
        jax.tree.leaves(left.accepted_state.particles),
        jax.tree.leaves(right.accepted_state.particles),
        strict=True,
    ):
        np.testing.assert_allclose(a, b, rtol=2e-11, atol=2e-13)
    np.testing.assert_allclose(
        left.accepted_state.boundary_states[0].position,
        right.accepted_state.boundary_states[0].position,
        rtol=2e-11,
        atol=2e-13,
    )
    np.testing.assert_allclose(
        np.sum(left.diagnostics.boundary_loads[0].impulse.reshape(-1, 2), axis=0),
        0.001 * 1.4 * 0.18 * np.asarray((0.12, -0.08)),
        rtol=2e-12,
        atol=2e-14,
    )
    assert float(left.diagnostics.dashpots[0].dissipation) >= 0.0
    if multifield:
        np.testing.assert_array_equal(
            left.accepted_state.particles.position[:2], position[:2]
        )
        np.testing.assert_array_equal(
            left.accepted_state.particles.velocity[:2], jnp.zeros_like(position[:2])
        )
        assert float(jnp.linalg.norm(left.accepted_state.particles.velocity[2:])) > 1e-5


@pytest.mark.parametrize(
    "kind", ("none", "b_bar", "f_bar"), ids=("ordinary", "b-bar", "f-bar")
)
@pytest.mark.parametrize("sparse", (False, True), ids=("dense", "block-sparse"))
@pytest.mark.parametrize(
    "load_kind", ("current", "follower"), ids=("current-traction", "follower-pressure")
)
def test_implicit_load_and_damping_residual_directional_central_difference(
    kind: MPMVolumetricStabilizationKind,
    sparse: bool,
    load_kind: Literal["current", "follower"],
) -> None:
    load = (
        CurrentSurfaceTraction(_position_traction, load_id="position-dependent-traction")
        if load_kind == "current"
        else GeneralFollowerLoad(
            _pressure,
            support="boundary",
            measure_frame="current",
            load_id="oriented-follower-pressure",
        )
    )
    compiled, arguments, position = _periodic_problem(
        kind=kind, sparse=sparse, load=load, damping=True
    )
    state = compiled.initialize_state(
        position,
        jnp.zeros_like(position),
        jnp.full((4,), 0.02, dtype=jnp.float64),
        arguments,
    )
    prepared = phx.solver.PreparedImplicitMPMDynamics(compiled.dynamics).prepare_residual(
        state, 0.007, arguments
    )
    phase = jnp.arange(prepared.initial_velocity.size, dtype=jnp.float64).reshape(
        prepared.initial_velocity.shape
    )
    velocity = prepared.initial_velocity + 0.03 * jnp.sin(0.31 * phase)
    direction = jnp.cos(0.17 * phase)
    value, tangent = jax.jvp(lambda v: prepared.evaluate(v)[0], (velocity,), (direction,))
    epsilon = jnp.asarray(1e-5, dtype=jnp.float64)
    plus, plus_valid = prepared.evaluate(velocity + epsilon * direction)
    minus, minus_valid = prepared.evaluate(velocity - epsilon * direction)

    assert bool(plus_valid & minus_valid)
    assert bool(jnp.all(jnp.isfinite(value)) & jnp.all(jnp.isfinite(tangent)))
    # O(epsilon^2) central FD; this is relative to the directional residual,
    # not an absolute stress-unit-dependent material tolerance.
    np.testing.assert_allclose(
        tangent, (plus - minus) / (2.0 * epsilon), rtol=3e-6, atol=3e-9
    )
    # A residual that silently drops both contributions can still have a
    # perfectly consistent derivative. Compare with the genuinely unloaded
    # problem to make this directional check sensitive to that wiring fault.
    unloaded, unloaded_arguments, _ = _periodic_problem(
        kind=kind, sparse=sparse, include_boundary=False
    )
    unloaded_state = unloaded.initialize_state(
        position,
        jnp.zeros_like(position),
        jnp.full((4,), 0.02, dtype=jnp.float64),
        unloaded_arguments,
    )
    unloaded_residual = phx.solver.PreparedImplicitMPMDynamics(
        unloaded.dynamics
    ).prepare_residual(unloaded_state, 0.007, unloaded_arguments)
    _, unloaded_tangent = jax.jvp(
        lambda v: unloaded_residual.evaluate(v)[0], (velocity,), (direction,)
    )
    assert float(jnp.linalg.norm(tangent - unloaded_tangent)) > 1e-6


@pytest.mark.parametrize("implicit", (False, True), ids=("explicit", "implicit"))
@pytest.mark.parametrize(
    "failure", ("load", "dashpot"), ids=("invalid-load", "invalid-dashpot")
)
def test_failed_boundary_contribution_preserves_particle_and_boundary_commits(
    implicit: bool, failure: Literal["load", "dashpot"]
) -> None:
    compiled, arguments, position = _periodic_problem(
        load=CurrentSurfaceTraction(
            _position_traction, load_id="dynamic-argument-traction"
        ),
        damping=True,
    )
    state = compiled.initialize_state(
        position,
        jnp.full_like(position, 0.01),
        jnp.full((4,), 0.02, dtype=jnp.float64),
        arguments,
    )
    dynamics = compiled.dynamics
    if failure == "load":
        arguments = eqx.tree_at(
            lambda a: a.boundary_arguments,
            arguments,
            jnp.asarray(jnp.nan, dtype=jnp.float64),
            is_leaf=lambda value: value is None,
        )
    else:
        dynamics = eqx.tree_at(
            lambda d: d.dashpots[0].density,
            dynamics,
            jnp.full_like(dynamics.dashpots[0].density, jnp.nan),
        )
    result = (
        phx.solver.PreparedImplicitMPMDynamics(dynamics).step_detailed(
            state, 0.001, arguments
        )
        if implicit
        else dynamics.step_detailed(state, 0.001, arguments)
    )

    assert not bool(result.successful)
    for before, after in zip(
        jax.tree.leaves(state.particles),
        jax.tree.leaves(result.accepted_state.particles),
        strict=True,
    ):
        np.testing.assert_array_equal(after, before)
    for before, after in zip(
        jax.tree.leaves(state.boundary_states),
        jax.tree.leaves(result.accepted_state.boundary_states),
        strict=True,
    ):
        np.testing.assert_array_equal(after, before)
    np.testing.assert_array_equal(result.accepted_state.time, state.time)
    np.testing.assert_array_equal(
        result.accepted_state.accepted_step, state.accepted_step
    )


def test_axisymmetric_follower_pressure_balances_independent_lame_stress() -> None:
    """Lamé a=1, b=2, p_i=.1, p_o=0; ring-integrated Q1 weak balance."""
    a, b, pressure, h = 1.0, 2.0, 0.1, 0.125
    gauss, weights = np.polynomial.legendre.leggauss(3)
    r = (np.arange(a, b, h)[:, None] + h * (gauss[None, :] + 1.0) / 2).ravel()
    rw = np.tile(h * weights / 2, 8)
    z = (np.arange(0.0, 1.0, 0.25)[:, None] + 0.25 * (gauss[None, :] + 1.0) / 2).ravel()
    zw = np.tile(0.25 * weights / 2, 4)
    radial, axial = np.meshgrid(r, z, indexing="ij")
    position = jnp.asarray(
        np.stack((radial.ravel(), axial.ravel()), axis=-1), dtype=jnp.float64
    )
    volume = jnp.asarray(
        (2 * np.pi * radial * rw[:, None] * zw[None, :]).ravel(), dtype=jnp.float64
    )
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformAxisSpec(19, periodic=False, endpoint=True),
            phx.discretization.UniformAxisSpec(4, periodic=True, endpoint=False),
        ),
        axis_names=("r", "z"),
    ).prepare(jnp.asarray(((0.0, 0.0), (2.25, 1.0)), dtype=jnp.float64))
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(position.shape[0], dtype=jnp.int32), volume, ambient_dimension=2
    ).prepare()
    splat = phx.discretization.ParticleGridSplatPlan(
        grid,
        assignment=phx.discretization.TensorBSplineSplatAssignment(1),
        boundary="reject",
    ).prepare(particles)
    geometry = phx.discretization.MPMGeometryPlan("axisymmetric", spatial_dimension=2)
    load = GeneralFollowerLoad(
        _pressure,
        support="boundary",
        measure_frame="current",
        load_id="lame-inner-pressure",
    )
    adapter = PreparedMPMBoundaryLoad(
        splat,
        jnp.asarray(np.stack((np.full_like(z, a), z), axis=-1), dtype=jnp.float64),
        jnp.asarray(
            np.stack((-2 * np.pi * a * zw, np.zeros_like(zw)), axis=-1), dtype=jnp.float64
        ),
        load,
        geometry=geometry,
    )
    material = phx.applications.solid_mechanics.FiniteStrainJ2MPMConstitutivePlan()
    parameters = phx.applications.solid_mechanics.FiniteStrainJ2Parameters(
        100.0, 200.0, 1e6, 0.0
    )
    arguments = phx.equations.MaterialPointArguments(
        parameters, mechanical_load_states=(MechanicalLoadState(pressure),)
    )
    compiled = phx.equations.compile_material_point_problem(
        phx.equations.MaterialPointProblemIR(
            "lame-canonical-pressure", material, boundary_loads=(adapter,)
        ),
        particles,
        splat,
        phx.discretization.ExplicitMPMMethodPlan(geometry=geometry),
        phx.discretization.MPMParticleDomainPlan(
            jnp.asarray(((0.125, 0.0), (2.125, 1.0)), dtype=jnp.float64),
            periodic=(False, True),
            support_margin=(0.125, 0.0),
        ),
    )
    constant_a = pressure * a * a / (b * b - a * a)
    constant_b = pressure * a * a * b * b / (b * b - a * a)
    stress = jnp.zeros((position.shape[0], 3, 3), dtype=jnp.float64)
    stress = stress.at[:, 0, 0].set(constant_a - constant_b / position[:, 0] ** 2)
    stress = stress.at[:, 2, 2].set(constant_a + constant_b / position[:, 0] ** 2)
    prestress = material.initialize_state_from_stress(stress, parameters)
    assert bool(jnp.all(prestress.admissible))
    state = compiled.initialize_state(
        position,
        jnp.zeros_like(position),
        volume,
        arguments,
        material_state=prestress.state,
    )
    prepared = phx.solver.PreparedImplicitMPMDynamics(compiled.dynamics).prepare_residual(
        state, 0.001, arguments
    )
    residual, valid = prepared.evaluate(prepared.initial_velocity)
    boundary_state = state.boundary_states[0]
    evaluated = adapter.evaluate(
        boundary_state,
        adapter.build_routes(boundary_state),
        jnp.zeros((compiled.dynamics.grid_count, 2), dtype=jnp.float64),
        0.001,
        arguments.mechanical_load_states[0],
    )

    assert bool(valid & evaluated.successful)
    np.testing.assert_allclose(
        jnp.sum(evaluated.force, axis=0),
        (2 * np.pi * a * pressure, 0.0),
        rtol=3e-12,
        atol=3e-13,
    )
    # Three-point Gauss integration of 1/r terms on h=.125 cells; scale by
    # physical pressure impulse, so Pa/kPa rescaling leaves this oracle intact.
    relative_defect = float(jnp.linalg.norm(residual)) / (
        0.001 * 2 * np.pi * a * pressure
    )
    assert relative_defect < 2e-7


def _wave_history(matched: bool) -> tuple[Array, Array, Array]:
    """1-D longitudinal MPM; c=sqrt(lambda+2mu)=1, rho=area=1."""
    cells, particles_per_cell = 64, 4
    h = 1.0 / cells
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformAxisSpec(cells + 3, periodic=False, endpoint=True),),
        axis_names=("x",),
    ).prepare(jnp.asarray(((-h,), (1.0 + h,)), dtype=jnp.float64))
    position = jnp.asarray(
        ((np.arange(cells * particles_per_cell) + 0.5) / (cells * particles_per_cell))[
            :, None
        ],
        dtype=jnp.float64,
    )
    volume = jnp.full((position.shape[0],), h / particles_per_cell, dtype=jnp.float64)
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(position.shape[0], dtype=jnp.int32), volume, ambient_dimension=1
    ).prepare()
    splat = phx.discretization.ParticleGridSplatPlan(
        grid,
        assignment=phx.discretization.TensorBSplineSplatAssignment(1),
        boundary="reject",
    ).prepare(particles)
    area = jnp.zeros((cells + 3, 1), dtype=jnp.float64).at[cells + 1, 0].set(1.0)
    fixed_mask = jnp.zeros((cells + 3, 1), dtype=jnp.bool_).at[1, 0].set(True)
    if not matched:
        fixed_mask = fixed_mask.at[cells + 1, 0].set(True)
    compiled = phx.equations.compile_material_point_problem(
        phx.equations.MaterialPointProblemIR(
            "p-wave-reflection",
            phx.applications.solid_mechanics.NeoHookeanMPMConstitutivePlan(1),
            dashpots=(LysmerKuhlemeyerDashpot(area, 1.0, 1.0, 0.5),) if matched else (),
        ),
        particles,
        splat,
        phx.discretization.ExplicitMPMMethodPlan(phx.discretization.FLIPTransferPlan()),
        phx.discretization.MPMParticleDomainPlan(
            jnp.asarray(((0.0,), (1.0,)), dtype=jnp.float64),
            periodic=(False,),
            support_margin=h,
        ),
        boundary=phx.discretization.PrescribedGridVelocityPlan(fixed_mask),
    )
    arguments = phx.equations.MaterialPointArguments(
        phx.applications.solid_mechanics.NeoHookeanParameters.from_shear_bulk(
            0.25, 2.0 / 3.0
        )
    )
    amplitude, center, width = 1e-5, 0.30, 0.055
    velocity = amplitude * jnp.exp(-(((position - center) / width) ** 2))
    # Right-going solution u=f(x-ct): strain=-v/c. Tiny amplitude makes the
    # hyperelastic correction O(1e-5), independent of the reflection tolerance.
    state = compiled.initialize_state(
        position,
        velocity,
        volume,
        arguments,
        deformation_gradient=(1.0 - velocity).reshape(-1, 1, 1),
    )
    dt = jnp.asarray(0.2 * h, dtype=jnp.float64)
    probe = jnp.asarray(np.abs(np.asarray(position[:, 0]) - 0.60) < h, dtype=jnp.bool_)

    def advance(
        committed: phx.discretization.MPMRuntimeState, unused: None
    ) -> tuple[phx.discretization.MPMRuntimeState, tuple[Array, Array]]:
        del unused
        detail = compiled.dynamics.step_detailed(committed, dt, arguments)
        measured = jnp.sum(
            jnp.where(probe, detail.accepted_state.particles.velocity[:, 0], 0.0)
        ) / jnp.sum(probe.astype(jnp.float64))
        return detail.accepted_state, (measured, detail.successful)

    _, (history, successful) = jax.jit(
        lambda initial: jax.lax.scan(advance, initial, None, length=430)
    )(state)
    time = dt * jnp.arange(1, history.size + 1, dtype=jnp.float64)
    return time, history, successful


def test_normally_incident_p_wave_matched_dashpot_has_small_measured_reflection() -> None:
    time, absorbed, absorbed_ok = _wave_history(True)
    fixed_time, fixed, fixed_ok = _wave_history(False)
    assert bool(jnp.all(absorbed_ok) & jnp.all(fixed_ok))
    np.testing.assert_array_equal(time, fixed_time)
    # Probe x=.60: incident arrival .30; reflected arrival 1.10. Windows
    # exclude both the initial tail and any return from the fixed left wall.
    incident = (np.asarray(time) > 0.15) & (np.asarray(time) < 0.45)
    reflected = (np.asarray(time) > 0.90) & (np.asarray(time) < 1.30)
    incident_amplitude = np.max(np.abs(np.asarray(absorbed)[incident]))
    fixed_incident = np.max(np.abs(np.asarray(fixed)[incident]))
    matched_coefficient = (
        np.max(np.abs(np.asarray(absorbed)[reflected])) / incident_amplitude
    )
    fixed_coefficient = np.max(np.abs(np.asarray(fixed)[reflected])) / fixed_incident
    print(
        f"P-wave reflection: matched={matched_coefficient:.8g}, "
        f"fixed={fixed_coefficient:.8g}"
    )
    assert incident_amplitude > 0.7e-5
    # The small grid resolves a Gaussian over ~7 cells. Ten-percent reflection
    # allows discretization/transfer dispersion, not an impedance mismatch;
    # the fixed-wall positive control must retain a substantial reflected pulse.
    assert matched_coefficient < 0.10
    assert fixed_coefficient > 0.65
    assert matched_coefficient < 0.15 * fixed_coefficient


@pytest.mark.parametrize(
    "kind", ("none", "b_bar", "f_bar"), ids=("ordinary", "b-bar", "f-bar")
)
def test_homogeneous_column_dynamics_retains_analytic_traction_stress_equilibrium(
    kind: MPMVolumetricStabilizationKind,
) -> None:
    """Independent 1-D Neo-Hookean equilibrium with prescribed end tractions.

    P=mu*(s-1/s)+lambda*log(s)/s; in 1-D Cauchy stress is P.
    The small strain 1e-5 approaches sigma=(lambda+2mu)*epsilon.
    Current quadrature volumes are divided by s to supply reference volumes.
    """
    stretch, shear, bulk = 1.0 + 1e-5, 2.0, 8.0
    lame = bulk - 2.0 * shear / 3.0
    analytic_stress = shear * (stretch - 1.0 / stretch) + lame * np.log(stretch) / stretch
    h, particle_count = 0.125, 32
    position = jnp.asarray(
        (0.25 + (np.arange(particle_count) + 0.5) / particle_count)[:, None],
        dtype=jnp.float64,
    )
    reference_volume = jnp.full(
        (particle_count,), 1.0 / (particle_count * stretch), dtype=jnp.float64
    )
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformAxisSpec(13, periodic=False, endpoint=True),),
        axis_names=("x",),
    ).prepare(jnp.asarray(((0.0,), (1.5,)), dtype=jnp.float64))
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(particle_count, dtype=jnp.int32), reference_volume, ambient_dimension=1
    ).prepare()
    splat = phx.discretization.ParticleGridSplatPlan(
        grid,
        assignment=phx.discretization.TensorBSplineSplatAssignment(1),
        boundary="reject",
    ).prepare(particles)
    left = PreparedMPMBoundaryLoad(
        splat,
        jnp.asarray(((0.25,),), dtype=jnp.float64),
        jnp.asarray(((-1.0,),), dtype=jnp.float64),
        ReferenceDeadTraction(jnp.asarray((-analytic_stress,), dtype=jnp.float64)),
    )
    right = PreparedMPMBoundaryLoad(
        splat,
        jnp.asarray(((1.25,),), dtype=jnp.float64),
        jnp.asarray(((1.0,),), dtype=jnp.float64),
        ReferenceDeadTraction(jnp.asarray((analytic_stress,), dtype=jnp.float64)),
    )
    compiled = phx.equations.compile_material_point_problem(
        phx.equations.MaterialPointProblemIR(
            "analytic-column-equilibrium",
            phx.applications.solid_mechanics.NeoHookeanMPMConstitutivePlan(1),
            boundary_loads=(left, right),
        ),
        particles,
        splat,
        phx.discretization.ExplicitMPMMethodPlan(
            volumetric_stabilization=phx.discretization.MPMVolumetricStabilizationPlan(
                kind
            )
        ),
        phx.discretization.MPMParticleDomainPlan(
            jnp.asarray(((h,), (1.5 - h,)), dtype=jnp.float64),
            periodic=(False,),
            support_margin=h,
        ),
    )
    arguments = phx.equations.MaterialPointArguments(
        phx.applications.solid_mechanics.NeoHookeanParameters.from_shear_bulk(
            shear, bulk
        ),
        mechanical_load_states=(MechanicalLoadState(), MechanicalLoadState()),
    )
    deformation = jnp.full((particle_count, 1, 1), stretch, dtype=jnp.float64)
    state = compiled.initialize_state(
        position,
        jnp.zeros_like(position),
        reference_volume,
        arguments,
        deformation_gradient=deformation,
    )
    result = compiled.dynamics.step_detailed(state, 0.001, arguments)

    assert bool(result.successful)
    np.testing.assert_allclose(
        result.accepted_state.particles.first_piola[:, 0, 0],
        analytic_stress,
        rtol=2e-11,
        atol=2e-15,
    )
    np.testing.assert_allclose(
        result.accepted_state.particles.deformation_gradient,
        deformation,
        rtol=0.0,
        atol=2e-14,
    )
    np.testing.assert_allclose(
        result.accepted_state.particles.velocity, 0.0, rtol=0.0, atol=2e-14
    )
    np.testing.assert_allclose(
        result.accepted_state.particles.position, position, rtol=0.0, atol=2e-14
    )
    np.testing.assert_allclose(
        analytic_stress, (lame + 2.0 * shear) * (stretch - 1.0), rtol=2e-5, atol=0.0
    )
