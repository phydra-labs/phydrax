#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Bounded geometry and measure-adjoint VF-IBM consumer contracts."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax as phx
from phydrax.discretization.finite_volume._incompressible import (
    FaceVelocity,
    PreparedMACOperators,
)
from phydrax.discretization.particle._rigid_body import RigidBodyKinematics
from phydrax.equations._mac_volume_fraction_ib_cfd_dem import (
    MACSDFCutFractionPlan,
    MACVolumeFractionIBCFDEMCouplingPlan,
)
from phydrax.solver._mac_volume_fraction_ib_cfd_dem import MACVolumeFractionIBState


pytestmark = pytest.mark.strict_jax


def _operators(count: int = 6) -> PreparedMACOperators:
    grid = phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformCellAxisSpec(count, periodic=True) for _ in range(3)
        ),
        axis_names=("x", "y", "z"),
    ).prepare(np.asarray(((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), dtype=np.float64))
    return phx.discretization.MACOperatorPlan(
        phx.discretization.FiniteVolumePlan(grid).prepare()
    ).prepare()


def _pose(position: Array) -> RigidBodyKinematics:
    count = position.shape[0]
    return RigidBodyKinematics(
        position,
        jnp.zeros_like(position),
        jnp.broadcast_to(
            jnp.asarray((1.0, 0.0, 0.0, 0.0), dtype=jnp.float64), (count, 4)
        ),
        jnp.zeros((count, 3), dtype=jnp.float64),
    )


def _case(
    *,
    count: int = 6,
    bodies: int = 1,
    unresolved: float = 1.0,
    maximum_queries: int | None = None,
    projection_iterations: int = 200,
) -> tuple[MACVolumeFractionIBCFDEMCouplingPlan, MACVolumeFractionIBState]:
    operators = _operators(count)
    radius = 0.23
    particles = phx.discretization.ParticleSetPlan(
        np.arange(bodies, dtype=np.int64),
        np.full(bodies, 2.5 * 4.0 * np.pi * radius**3 / 3.0, dtype=np.float64),
        ambient_dimension=3,
    ).prepare()
    shapes = phx.discretization.RigidSphereSetPlan(
        np.full(bodies, radius, dtype=np.float64), np.zeros(bodies, dtype=np.int32)
    )
    dem = phx.equations.compile_discrete_element_problem(
        phx.equations.DiscreteElementProblemIR(
            "vf-ib-unit",
            phx.equations.DEMMaterialTable(
                np.asarray((1.0e5,), dtype=np.float64),
                np.asarray((0.25,), dtype=np.float64),
                np.asarray(((0.9,),), dtype=np.float64),
                np.asarray(((0.0,),), dtype=np.float64),
            ),
            gravity=jnp.zeros((3,), dtype=jnp.float64),
        ),
        particles,
        shapes,
        phx.discretization.CompliantDEMMethodPlan(
            phx.discretization.DEMContactModelPlan(
                phx.discretization.LinearSpringDashpotNormalPlan(100.0)
            )
        ),
        neighborhood=phx.discretization.DenseParticleNeighborhoodPlan(max(1, bodies**2)),
    )
    initial_position = jnp.full((bodies, 3), 0.5, dtype=jnp.float64)
    if bodies > 1:
        initial_position = initial_position.at[:, 0].set(
            jnp.linspace(0.25, 0.75, bodies, dtype=jnp.float64)
        )
    dem_state = dem.initialize_state(
        0.0,
        initial_position,
        jnp.zeros((bodies, 3), dtype=jnp.float64),
        orientation=jnp.broadcast_to(
            jnp.asarray((1.0, 0.0, 0.0, 0.0), dtype=jnp.float64), (bodies, 4)
        ),
    )
    projection = phx.solver.MACVariableDensityProjectionPlan(
        operators, tolerance=1.0e-10, maximum_iterations=projection_iterations
    )
    fluid = phx.equations.compile_mac_variable_density_flow(
        phx.equations.MACVariableDensityFlowProblem(3, 0.01),
        phx.discretization.MACVariableDensityPlan(
            phx.discretization.MACMomentumPlan(operators).prepare()
        ).prepare(),
        projection,
    )
    cut = MACSDFCutFractionPlan(
        operators,
        dem.dynamics.shapes,
        subdivisions=4,
        maximum_unresolved_fraction=unresolved,
        maximum_queries=maximum_queries,
    )
    coupling = MACVolumeFractionIBCFDEMCouplingPlan(
        fluid, dem.dynamics, cut, projection, fluid_density=1.0, solid_density=2.5
    )
    zero = tuple(
        jnp.zeros(layout.shape, dtype=jnp.float64)
        for layout in operators.discretization.face_layouts
    )
    state = MACVolumeFractionIBState.initialize(
        coupling,
        dem_state,
        fluid.pack_state(
            jnp.ones(operators.discretization.cell_shape, dtype=jnp.float64), zero
        ),
    )
    return coupling, state


def test_sphere_analytic_volume_is_enclosed_and_estimate_is_bounded() -> None:
    coupling, state = _case()
    result = coupling.cut_fraction.evaluate(state.dem_state.kinematics, 0.0)
    volumes = coupling.cut_fraction.operators.discretization.cell_volumes
    exact = 4.0 * np.pi * 0.23**3 / 3.0
    assert result.successful
    assert float(jnp.sum(result.fraction_lower * volumes)) <= exact
    assert exact <= float(jnp.sum(result.fraction_upper * volumes))
    assert jnp.all(result.fraction_lower <= result.solid_fraction)
    assert jnp.all(result.solid_fraction <= result.fraction_upper)
    np.testing.assert_allclose(result.fluid_fraction + result.solid_fraction, 1.0)
    np.testing.assert_allclose(
        result.unresolved_fraction, result.fraction_upper - result.fraction_lower
    )
    assert result.query_reliable and not result.query_overflow


def test_independent_plane_slab_fraction_is_enclosed_in_every_cell() -> None:
    operators = _operators(4)
    # A finite canonical convex box whose other five faces are outside the grid.
    # Inside the grid its indicator is exactly x < 0.37; no package VOF oracle.
    vertices = np.asarray(
        (
            (-2.0, -2.0, -2.0),
            (0.37, -2.0, -2.0),
            (0.37, 2.0, -2.0),
            (-2.0, 2.0, -2.0),
            (-2.0, -2.0, 2.0),
            (0.37, -2.0, 2.0),
            (0.37, 2.0, 2.0),
            (-2.0, 2.0, 2.0),
        ),
        dtype=np.float64,
    )
    triangles = np.asarray(
        (
            (0, 2, 1),
            (0, 3, 2),
            (4, 5, 6),
            (4, 6, 7),
            (0, 1, 5),
            (0, 5, 4),
            (3, 7, 6),
            (3, 6, 2),
            (0, 4, 7),
            (0, 7, 3),
            (1, 2, 6),
            (1, 6, 5),
        ),
        dtype=np.int32,
    )
    particles = phx.discretization.ParticleSetPlan(
        np.asarray((0,), dtype=np.int64),
        np.ones(1, dtype=np.float64),
        ambient_dimension=3,
    ).prepare()
    shape = phx.discretization.ConvexShapeSetPlan(
        (phx.discretization.ConvexShapePlan(vertices, triangles, 0),),
        np.zeros(1, dtype=np.int32),
        np.zeros(1, dtype=np.int32),
    ).prepare(particles)
    # The convex owner recenters its template at the material COM.
    center = np.asarray(((-2.0 + 0.37) / 2.0, 0.0, 0.0), dtype=np.float64)
    result = MACSDFCutFractionPlan(
        operators, shape, subdivisions=8, maximum_unresolved_fraction=1.0
    ).evaluate(_pose(jnp.asarray(center[None, :])), 0.0)
    exact_x = np.clip((0.37 - np.arange(4, dtype=np.float64) / 4.0) * 4.0, 0.0, 1.0)
    exact = np.broadcast_to(exact_x[:, None, None], (4, 4, 4))
    assert result.query_reliable
    assert not result.domain_supported
    assert not result.successful
    assert np.all(np.asarray(result.fraction_lower) <= exact + 1e-14)
    assert np.all(np.asarray(result.fraction_upper) >= exact - 1e-14)


def test_subbox_refinement_contracts_sphere_volume_enclosure() -> None:
    coupling, state = _case(count=4)
    operators = coupling.cut_fraction.operators
    results = tuple(
        MACSDFCutFractionPlan(
            operators,
            coupling.dynamics.shapes,
            subdivisions=n,
            maximum_unresolved_fraction=1.0,
        ).evaluate(state.dem_state.kinematics, 0.0)
        for n in (2, 4, 8)
    )
    widths = tuple(
        float(jnp.sum(item.unresolved_fraction * operators.discretization.cell_volumes))
        for item in results
    )
    assert widths[2] < widths[1] < widths[0]
    assert jnp.all(results[2].fraction_lower >= results[0].fraction_lower)
    assert jnp.all(results[2].fraction_upper <= results[0].fraction_upper)


def test_overlapping_identical_bodies_use_union_not_fraction_sum() -> None:
    coupling, state = _case(bodies=2)
    pose = eqx.tree_at(
        lambda value: value.position,
        state.dem_state.kinematics,
        jnp.full((2, 3), 0.5, dtype=jnp.float64),
    )
    result = coupling.cut_fraction.evaluate(pose, 0.0)
    assert result.successful
    np.testing.assert_allclose(result.solid_fraction, result.solid_fraction_by_body[0])
    np.testing.assert_allclose(
        result.solid_fraction_by_body[0], result.solid_fraction_by_body[1]
    )
    assert jnp.all((result.solid_fraction >= 0.0) & (result.solid_fraction <= 1.0))
    assert jnp.max(jnp.sum(result.solid_fraction_by_body, axis=0)) > 1.0


def test_unresolved_budget_is_per_cell_not_domain_average() -> None:
    coupling, state = _case(unresolved=0.2)
    result = coupling.cut_fraction.evaluate(state.dem_state.kinematics, 0.0)
    assert jnp.mean(result.unresolved_fraction) < 0.2
    assert jnp.max(result.unresolved_fraction) > 0.2
    assert not result.successful
    assert result.query_reliable
    assert jnp.min(result.geometry_margin) < 0.0


def test_query_capacity_overflow_is_failed_bounded_evidence() -> None:
    coupling, state = _case(maximum_queries=1)
    result = coupling.cut_fraction.evaluate(state.dem_state.kinematics, 0.0)
    assert result.query_overflow
    assert not result.successful
    assert (
        result.solid_fraction.shape
        == coupling.cut_fraction.operators.discretization.cell_shape
    )
    assert jnp.all(result.fraction_lower <= result.fraction_upper)


def test_unreliable_native_pose_queries_fail_closed() -> None:
    coupling, state = _case()
    bad = eqx.tree_at(
        lambda item: item.orientation,
        state.dem_state.kinematics,
        2.0 * state.dem_state.kinematics.orientation,
    )
    result = coupling.cut_fraction.evaluate(bad, 0.0)
    assert not result.query_reliable
    assert not result.successful


def test_inactive_native_shape_lanes_do_not_poison_geometry() -> None:
    operators = _operators(4)
    particles = phx.discretization.ParticleSetPlan(
        np.arange(2, dtype=np.int64),
        np.ones(2, dtype=np.float64),
        ambient_dimension=3,
        active_mask=np.asarray((True, False), dtype=np.bool_),
    ).prepare()
    shape = phx.discretization.RigidSphereSetPlan(
        np.full(2, 0.23, dtype=np.float64), np.zeros(2, dtype=np.int32)
    ).prepare(particles)
    result = MACSDFCutFractionPlan(
        operators, shape, subdivisions=4, maximum_unresolved_fraction=1.0
    ).evaluate(_pose(jnp.full((2, 3), 0.5, dtype=jnp.float64)), 0.0)
    assert result.successful and result.query_reliable
    np.testing.assert_array_equal(result.solid_fraction_by_body[1], 0.0)


def test_superquadric_is_refused_before_cut_fraction_preparation() -> None:
    operators = _operators(4)
    particles = phx.discretization.ParticleSetPlan(
        np.asarray((0,), dtype=np.int64),
        np.ones(1, dtype=np.float64),
        ambient_dimension=3,
    ).prepare()
    shape = phx.discretization.SuperquadricSetPlan(
        np.ones((1, 3), dtype=np.float64),
        np.full(1, 2.0, dtype=np.float64),
        np.full(1, 2.0, dtype=np.float64),
        np.zeros(1, dtype=np.int32),
    ).prepare(particles)
    with pytest.raises(ValueError, match="[Ss]uperquadric|bound"):
        MACSDFCutFractionPlan(operators, shape)


def test_mixture_mass_and_mass_fraction_follow_intrinsic_density_formula() -> None:
    coupling, state = _case()
    geometry = coupling.cut_fraction.evaluate(state.dem_state.kinematics, 0.0)
    zero = coupling.fluid.physical_state(0.0, state.fluid_state).velocity
    result = phx.equations.evaluate_mac_volume_fraction_ib_cfd_dem(
        coupling, state.dem_state.kinematics, geometry, zero, zero, jnp.asarray(0.01)
    )
    expected = 1.0 + 1.5 * geometry.solid_fraction
    assert result.successful
    np.testing.assert_allclose(result.mixture_density, expected)
    np.testing.assert_allclose(
        result.solid_mass_fraction, 2.5 * geometry.solid_fraction / expected
    )
    volumes = coupling.cut_fraction.operators.discretization.cell_volumes
    np.testing.assert_allclose(
        jnp.sum(result.mixture_density * volumes),
        jnp.sum((geometry.fluid_fraction + 2.5 * geometry.solid_fraction) * volumes),
    )


def test_rigid_rotation_force_torque_and_work_are_measure_adjoints() -> None:
    coupling, state = _case()
    pose = eqx.tree_at(
        lambda item: (item.velocity, item.angular_velocity),
        state.dem_state.kinematics,
        (
            jnp.asarray(((0.3, -0.2, 0.1),), dtype=jnp.float64),
            jnp.asarray(((0.7, -0.4, 0.9),), dtype=jnp.float64),
        ),
    )
    geometry = coupling.cut_fraction.evaluate(pose, 0.0)
    operators = coupling.cut_fraction.operators
    zero: FaceVelocity = tuple(
        jnp.zeros(layout.shape, dtype=jnp.float64)
        for layout in operators.discretization.face_layouts
    )
    result = phx.equations.evaluate_mac_volume_fraction_ib_cfd_dem(
        coupling, pose, geometry, zero, zero, jnp.asarray(0.01)
    )
    assert result.successful
    np.testing.assert_allclose(result.impulse_residual, np.zeros(3), atol=1e-12)
    np.testing.assert_allclose(result.torque_residual, np.zeros(3), atol=1e-12)
    np.testing.assert_allclose(result.work_adjoint_residual, 0.0, atol=1e-12)
    np.testing.assert_allclose(
        result.fluid_external_impulse + jnp.sum(result.body_direct_impulse, axis=0),
        np.zeros(3),
        atol=1e-12,
    )
    # Independent physical face quadrature catches using cell centers for torque.
    torque = np.zeros(3, dtype=np.float64)
    power = 0.0
    for axis, (force, density, measure, points) in enumerate(
        zip(
            result.fluid_ib_force_density,
            result.face_mixture_density,
            operators.face_dual_measures,
            operators.discretization.face_centers,
            strict=True,
        )
    ):
        # psi_face follows rho_face = rhof + phi_face*(rhos-rhof).
        phi = (np.asarray(density) - 1.0) / 1.5
        psi = 2.5 * phi / np.asarray(density)
        external = (1.0 - psi) * np.asarray(force) * np.asarray(measure)
        vector = np.zeros((*external.shape, 3), dtype=np.float64)
        vector[..., axis] = external
        torque += np.sum(np.cross(np.asarray(points), vector), axis=(0, 1, 2))
        lever = np.asarray(points) - np.asarray(pose.position[0])
        rigid = np.asarray(pose.velocity[0]) + np.cross(
            np.asarray(pose.angular_velocity[0]), lever
        )
        power += float(np.sum(external * rigid[..., axis]))
    np.testing.assert_allclose(
        0.01 * torque
        + 0.01
        * (
            np.asarray(result.body_torque[0])
            + np.cross(np.asarray(pose.position[0]), np.asarray(result.body_force[0]))
        ),
        np.zeros(3),
        atol=1e-12,
    )
    body_power = float(
        jnp.sum(result.body_force * pose.velocity)
        + jnp.sum(result.body_torque * pose.angular_velocity)
    )
    np.testing.assert_allclose(power + body_power, 0.0, atol=1e-12)


def test_centered_translating_sphere_has_no_spurious_torque() -> None:
    coupling, state = _case()
    pose = eqx.tree_at(
        lambda item: item.velocity,
        state.dem_state.kinematics,
        jnp.asarray(((0.2, 0.0, 0.0),), dtype=jnp.float64),
    )
    geometry = coupling.cut_fraction.evaluate(pose, 0.0)
    zero = coupling.fluid.physical_state(0.0, state.fluid_state).velocity
    result = phx.equations.evaluate_mac_volume_fraction_ib_cfd_dem(
        coupling, pose, geometry, zero, zero, jnp.asarray(0.01)
    )
    assert result.successful
    assert result.body_force[0, 0] < 0.0
    np.testing.assert_allclose(result.body_force[0, 1:], 0.0, atol=1e-12)
    np.testing.assert_allclose(result.body_torque, 0.0, atol=1e-12)


def test_pressure_viscous_load_is_separate_from_direct_reaction() -> None:
    coupling, state = _case()
    geometry = coupling.cut_fraction.evaluate(state.dem_state.kinematics, 0.0)
    operators = coupling.cut_fraction.operators
    zero = coupling.fluid.physical_state(0.0, state.fluid_state).velocity
    stress = tuple(
        jnp.full(layout.shape, axis + 1.0, dtype=jnp.float64)
        for axis, layout in enumerate(operators.discretization.face_layouts)
    )
    result = phx.equations.evaluate_mac_volume_fraction_ib_cfd_dem(
        coupling, state.dem_state.kinematics, geometry, zero, stress, jnp.asarray(0.01)
    )
    assert result.successful
    np.testing.assert_allclose(result.body_force, result.pressure_viscous_load)
    np.testing.assert_allclose(result.body_torque, result.pressure_viscous_torque)
    assert jnp.linalg.norm(result.pressure_viscous_load) > 0.0
    np.testing.assert_allclose(result.body_direct_impulse, 0.0)


def _assert_state_equal(
    actual: MACVolumeFractionIBState, expected: MACVolumeFractionIBState
) -> None:
    for left, right in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize(
    "failure", ("geometry", "projection"), ids=("geometry-budget", "projection-budget")
)
def test_failed_attempt_atomically_retains_entire_accepted_state(failure: str) -> None:
    coupling, state = _case(
        unresolved=0.02 if failure == "geometry" else 1.0,
        projection_iterations=1 if failure == "projection" else 200,
    )
    if failure == "projection":
        pose = eqx.tree_at(
            lambda item: item.velocity,
            state.dem_state.kinematics,
            jnp.asarray(((0.2, -0.1, 0.3),), dtype=jnp.float64),
        )
        state = eqx.tree_at(lambda item: item.dem_state.kinematics, state, pose)
    result = phx.solver.advance_mac_volume_fraction_ib_cfd_dem(
        coupling, state, 0.0, 0.001
    )
    assert not result.successful
    _assert_state_equal(result.accepted_state, state)
    assert result.accepted_state.accepted_steps == 0
    if failure == "projection":
        assert not result.pressure_projection.successful


def test_successful_step_commits_geometry_density_and_one_step_counter() -> None:
    coupling, state = _case()
    result = phx.solver.advance_mac_volume_fraction_ib_cfd_dem(
        coupling, state, 0.0, 0.001
    )
    assert result.successful
    assert result.pressure_projection.successful
    assert result.accepted_state.accepted_steps == 1
    physical = coupling.fluid.physical_state(0.001, result.accepted_state.fluid_state)
    np.testing.assert_allclose(
        physical.density, 1.0 + 1.5 * result.candidate_cut_fraction.solid_fraction
    )
    assert (
        jnp.linalg.norm(coupling.cut_fraction.operators.divergence(physical.velocity))
        < 1e-8
    )


def test_volume_fraction_regime_is_distinct_from_marker_penalty_portfolio() -> None:
    coupling, _ = _case()
    regime = phx.applications.incompressible_flow.ImmersedBodyRegimePlan(
        coupling,
        marker_set_id="no-markers-volume-fraction",
        geometry_id=coupling.cut_fraction.shape_query.shape_id,
        route_id=coupling.cut_fraction.plan_id,
        topology_epoch_id="vf-fixed-body-capacity",
        geometry_epoch=0,
        moving=True,
        motion_epoch_id="vf-accepted-pose",
    )
    assert regime.regime == "volume-fraction-cfd-dem"
    assert regime.support_tuple.support_tuple_id == (
        phx.applications.incompressible_flow.VOLUME_FRACTION_CFD_DEM_SUPPORT_TUPLE.support_tuple_id
    )
    assert regime.support_tuple.support_tuple_id != (
        phx.applications.incompressible_flow.RESOLVED_CFD_DEM_SUPPORT_TUPLE.support_tuple_id
    )


def test_candidate_pose_geometry_failure_rolls_back_fluid_and_dem_together() -> None:
    coupling, state = _case()
    pose = eqx.tree_at(
        lambda item: (item.position, item.velocity),
        state.dem_state.kinematics,
        (
            jnp.asarray(((0.75, 0.5, 0.5),), dtype=jnp.float64),
            jnp.asarray(((10.0, 0.0, 0.0),), dtype=jnp.float64),
        ),
    )
    state = eqx.tree_at(lambda item: item.dem_state.kinematics, state, pose)
    operators = coupling.cut_fraction.operators
    # Uniform co-translation avoids an imposed slip force: the candidate body
    # actually crosses the admitted domain limit during the native DEM step.
    velocity = tuple(
        jnp.full(layout.shape, 10.0 if axis == 0 else 0.0, dtype=jnp.float64)
        for axis, layout in enumerate(operators.discretization.face_layouts)
    )
    packed = coupling.fluid.pack_state(
        jnp.ones(operators.discretization.cell_shape, dtype=jnp.float64), velocity
    )
    state = eqx.tree_at(lambda item: item.fluid_state, state, packed)
    assert coupling.cut_fraction.evaluate(pose, 0.0).successful
    result = phx.solver.advance_mac_volume_fraction_ib_cfd_dem(
        coupling, state, 0.0, 0.003
    )
    assert result.pressure_projection.successful
    assert result.dem_step.successful
    assert not result.candidate_cut_fraction.domain_supported
    assert not result.successful
    _assert_state_equal(result.accepted_state, state)


def test_external_impulse_excludes_internal_mixture_acceleration() -> None:
    coupling, state = _case()
    pose = eqx.tree_at(
        lambda item: item.velocity,
        state.dem_state.kinematics,
        jnp.asarray(((0.2, 0.0, 0.0),), dtype=jnp.float64),
    )
    geometry = coupling.cut_fraction.evaluate(pose, 0.0)
    zero = coupling.fluid.physical_state(0.0, state.fluid_state).velocity
    result = phx.equations.evaluate_mac_volume_fraction_ib_cfd_dem(
        coupling, pose, geometry, zero, zero, 0.01
    )
    measures = coupling.cut_fraction.operators.face_dual_measures
    total = 0.01 * jnp.stack(
        tuple(
            jnp.sum(w * f)
            for w, f in zip(measures, result.fluid_ib_force_density, strict=True)
        )
    )
    assert result.successful
    np.testing.assert_allclose(
        total, result.fluid_external_impulse + result.fluid_internal_impulse
    )
    assert result.fluid_internal_impulse[0] > 0.0
    assert jnp.linalg.norm(total + jnp.sum(result.body_direct_impulse, axis=0)) > 1e-4
    np.testing.assert_allclose(result.impulse_residual, 0.0, atol=1e-12)


def test_overlapping_bodies_share_union_load_without_double_counting() -> None:
    coupling, state = _case(bodies=2)
    pose = eqx.tree_at(
        lambda item: (item.position, item.velocity),
        state.dem_state.kinematics,
        (
            jnp.full((2, 3), 0.5, dtype=jnp.float64),
            jnp.asarray(((0.2, 0.0, 0.0), (0.2, 0.0, 0.0)), dtype=jnp.float64),
        ),
    )
    geometry = coupling.cut_fraction.evaluate(pose, 0.0)
    zero = coupling.fluid.physical_state(0.0, state.fluid_state).velocity
    result = phx.equations.evaluate_mac_volume_fraction_ib_cfd_dem(
        coupling, pose, geometry, zero, zero, 0.01
    )
    assert result.successful
    np.testing.assert_allclose(result.body_force[0], result.body_force[1])
    np.testing.assert_allclose(
        result.fluid_external_impulse + jnp.sum(result.body_direct_impulse, axis=0),
        0.0,
        atol=1e-12,
    )
    single, single_state = _case()
    single_pose = eqx.tree_at(
        lambda item: item.velocity, single_state.dem_state.kinematics, pose.velocity[:1]
    )
    single_geometry = single.cut_fraction.evaluate(single_pose, 0.0)
    reference = phx.equations.evaluate_mac_volume_fraction_ib_cfd_dem(
        single, single_pose, single_geometry, zero, zero, 0.01
    )
    np.testing.assert_allclose(
        jnp.sum(result.body_force, axis=0), reference.body_force[0]
    )


@pytest.mark.parametrize(
    "density", (0.0, -1.0, np.inf), ids=("zero", "negative", "nonfinite")
)
def test_intrinsic_density_admission_rejects_invalid_values(density: float) -> None:
    coupling, _ = _case()
    with pytest.raises(ValueError, match="densit"):
        MACVolumeFractionIBCFDEMCouplingPlan(
            coupling.fluid,
            coupling.dynamics,
            coupling.cut_fraction,
            coupling.projection,
            fluid_density=density,
            solid_density=2.5,
        )


def test_intrinsic_solid_density_must_match_native_dem_mass_and_volume() -> None:
    coupling, _ = _case()
    with pytest.raises(ValueError, match="mass|volume|solid_density"):
        MACVolumeFractionIBCFDEMCouplingPlan(
            coupling.fluid,
            coupling.dynamics,
            coupling.cut_fraction,
            coupling.projection,
            fluid_density=1.0,
            solid_density=3.0,
        )


def test_coupling_rejects_mismatched_operator_identity() -> None:
    coupling, _ = _case()
    other = phx.solver.MACVariableDensityProjectionPlan(_operators(4))
    with pytest.raises(ValueError, match="operator"):
        MACVolumeFractionIBCFDEMCouplingPlan(
            coupling.fluid,
            coupling.dynamics,
            coupling.cut_fraction,
            other,
            fluid_density=1.0,
            solid_density=2.5,
        )


def test_native_voxel_evaluation_error_is_retained_in_cut_enclosure() -> None:
    operators = _operators(4)
    depth = 3
    resolution = 2**depth
    coordinates = np.stack(
        np.meshgrid(
            *(np.arange(resolution, dtype=np.int32) for _ in range(3)), indexing="ij"
        ),
        axis=-1,
    ).reshape(-1, 3)
    voxel = phx.discretization.SparseVoxelGridPlan(
        phx.discretization.MortonAddressPlan((-0.75,) * 3, (0.75,) * 3, depth),
        brick_size=2,
        brick_capacity=(resolution // 2) ** 3,
    ).prepare(coordinates)
    analytic = phx.geometry.Sphere(
        (0.0, 0.0, 0.0), 0.23, feature_id="vf-voxel-error-sphere"
    ).compile()
    samples = phx.geometry.VoxelGeometrySamplingPlan(
        voxel,
        enclosure=phx.geometry.ExactSDFEnclosureCertificate(analytic.field_certificate),
        enclosure_geometry=analytic,
    ).sample_level_set(analytic)
    radius = 0.23
    nodes = radius * np.concatenate(
        (np.eye(3, dtype=np.float64), -np.eye(3, dtype=np.float64))
    )
    template = phx.discretization.LevelSetShapeTemplatePlan(
        samples,
        np.arange(6, dtype=np.int64),
        nodes,
        np.full(6, 4.0 * np.pi * radius**2 / 6.0, dtype=np.float64),
        0,
        exact_volume=4.0 * np.pi * radius**3 / 3.0,
        exact_center_of_mass=np.zeros(3, dtype=np.float64),
        exact_inertia_per_mass=0.4 * radius**2 * np.eye(3, dtype=np.float64),
    )
    particles = phx.discretization.ParticleSetPlan(
        np.asarray((0,), dtype=np.int64),
        np.ones(1, dtype=np.float64),
        ambient_dimension=3,
    ).prepare()
    shape = phx.discretization.LevelSetRigidShapeSetPlan(
        (template,),
        np.zeros(1, dtype=np.int32),
        np.zeros(1, dtype=np.int32),
        maximum_contact_routes=8,
        maximum_candidate_slots=8,
    ).prepare(particles)
    result = MACSDFCutFractionPlan(
        operators,
        shape,
        subdivisions=4,
        maximum_unresolved_fraction=1.0,
    ).evaluate(_pose(jnp.full((1, 3), 0.5, dtype=jnp.float64)), 0.0)
    assert template.evaluation_error > radius
    assert result.query_reliable
    # The deliberately coarse certified error exceeds the sphere's maximum
    # interior depth: no subbox may be certified solid even at its center.
    np.testing.assert_array_equal(result.fraction_lower, 0.0)
    exact = 4.0 * np.pi * radius**3 / 3.0
    volumes = operators.discretization.cell_volumes
    assert float(jnp.sum(result.fraction_upper * volumes)) >= exact


@pytest.mark.parametrize(
    "step", (0.0, -0.001, np.nan), ids=("zero", "negative", "nonfinite")
)
def test_invalid_step_size_retains_the_committed_transaction(step: float) -> None:
    coupling, state = _case()
    result = phx.solver.advance_mac_volume_fraction_ib_cfd_dem(coupling, state, 0.0, step)
    assert not result.successful
    assert result.status != 0
    _assert_state_equal(result.accepted_state, state)


@pytest.mark.parametrize(
    "mutation",
    ("scale", "mass", "inverse"),
    ids=("scaled-volume", "population-mass", "inverse-mass"),
)
def test_runtime_solid_mass_mismatch_refuses_initialize_and_rolls_back(
    mutation: str,
) -> None:
    coupling, state = _case(count=4)
    properties = state.dem_state.body_properties
    if mutation == "scale":
        properties = eqx.tree_at(
            lambda item: item.shape_scale, properties, properties.shape_scale * 1.1
        )
    elif mutation == "mass":
        properties = eqx.tree_at(
            lambda item: item.population.mass, properties, properties.masses * 1.1
        )
    else:
        properties = eqx.tree_at(
            lambda item: item.inverse_masses, properties, properties.inverse_masses * 1.1
        )
    dem = eqx.tree_at(lambda item: item.body_properties, state.dem_state, properties)
    with pytest.raises(ValueError, match="runtime mass"):
        MACVolumeFractionIBState.initialize(coupling, dem, state.fluid_state)
    invalid = eqx.tree_at(lambda item: item.dem_state, state, dem)
    result = eqx.filter_jit(phx.solver.advance_mac_volume_fraction_ib_cfd_dem)(
        coupling, invalid, 0.0, 0.001
    )
    assert not result.successful
    assert result.status & int(phx.solver.MACVolumeFractionIBStepStatus.GEOMETRY_FAILED)
    _assert_state_equal(result.accepted_state, invalid)


def test_scaled_shape_with_consistent_runtime_mass_is_admitted() -> None:
    coupling, state = _case(count=4)
    properties = state.dem_state.body_properties
    scale = 0.9
    properties = eqx.tree_at(
        lambda item: (item.shape_scale, item.population.mass, item.inverse_masses),
        properties,
        (
            properties.shape_scale * scale,
            properties.masses * scale**3,
            properties.inverse_masses / scale**3,
        ),
    )
    dem = eqx.tree_at(lambda item: item.body_properties, state.dem_state, properties)
    initialized = MACVolumeFractionIBState.initialize(coupling, dem, state.fluid_state)
    result = phx.solver.advance_mac_volume_fraction_ib_cfd_dem(
        coupling, initialized, 0.0, 0.001
    )
    assert result.successful
