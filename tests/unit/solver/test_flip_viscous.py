#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax as phx
from phydrax._fingerprint import canonical_fingerprint
from phydrax._sharp_measures import exact_sharp_geometry
from phydrax.discretization.flip._viscous import viscous_flip_stage


D = phx.discretization


def _momentum(cells: int = 32) -> D.PreparedMACMomentumOperators:
    grid = D.TensorGridPlan(
        (D.UniformCellAxisSpec(4, periodic=True), D.UniformCellAxisSpec(cells)),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=jnp.float64))
    mac = D.MACOperatorPlan(D.FiniteVolumePlan(grid).prepare()).prepare()
    boundaries = D.MACBoundaryPlan(
        mac,
        (
            D.MACBoundarySide("y", "lower", "no-slip"),
            D.MACBoundarySide("y", "upper", "no-slip"),
        ),
    ).prepare()
    return D.MACMomentumPlan(mac, boundaries=boundaries).prepare()


def _profile_stage(
    law: phx.rheology.GeneralizedNewtonianLaw,
    profile: Array,
    momentum: D.PreparedMACMomentumOperators,
    *,
    maximum_iterations: int = 100,
) -> tuple[tuple[Array, ...], D.FLIPViscousEvidence]:
    mac = momentum.operators
    dt = jnp.asarray(0.1, dtype=jnp.float64)
    forced = (
        jnp.broadcast_to(profile[None, :] + dt, mac.discretization.face_layouts[0].shape),
        jnp.zeros(mac.discretization.face_layouts[1].shape, dtype=jnp.float64),
    )
    liquid = jnp.ones(mac.discretization.cell_shape, dtype=jnp.bool_)
    measures = D.finite_volume.MACFreeSurfaceViscousMeasurePlan(
        mac, 1.0
    ).evaluate_cell_fraction(liquid.astype(jnp.float64), 0.0)
    return viscous_flip_stage(
        phx.solver.MACVariationalViscosityPlan(momentum, tolerance=1.0e-12),
        law,
        None,
        D.FLIPMethodPlan(
            0.0, maximum_picard_iterations=maximum_iterations, viscous_tolerance=1.0e-8
        ),
        measures,
        forced,
        jnp.zeros_like(profile)[None, :].repeat(4, axis=0),
        liquid,
        dt,
        momentum.boundaries.evaluate(0.0),
        jnp.asarray(True),
    )


def test_newtonian_poiseuille_profile_and_native_solve_evidence() -> None:
    momentum = _momentum()
    y = momentum.operators.discretization.grid.structured_axes[1].interval_centers
    viscosity = 0.2
    # rho*g=1, no slip at y=0,1: eta*u''=-1.
    exact = y * (1.0 - y) / (2.0 * viscosity)
    velocity, evidence = _profile_stage(
        phx.rheology.GeneralizedNewtonianLaw(
            "power-law", viscosity, consistency_pa_s_n=viscosity, power_index=1.0
        ),
        exact,
        momentum,
    )
    assert evidence.successful
    assert evidence.converged
    assert evidence.linear.linear_status == int(phx.linalg.LinearSolveStatus.SUCCESS)
    assert evidence.linear.dissipation > 0.0
    assert evidence.relative_residual <= 1.0e-8
    # The half-cell wall stencil has an O(dy^2) steady-profile error.
    np.testing.assert_allclose(
        velocity[0],
        jnp.broadcast_to(exact, velocity[0].shape),
        rtol=0.0,
        atol=1.0 / (8.0 * viscosity * 32**2),
    )


def test_regularized_bingham_plug_width_matches_yield_surface() -> None:
    momentum = _momentum(48)
    y = momentum.operators.discretization.grid.structured_axes[1].interval_centers
    yield_stress, plastic_viscosity = 0.2, 0.1
    distance = jnp.abs(y - 0.5)
    yield_distance = yield_stress  # |tau|=rho*g*|y-1/2|, rho*g=1.
    moving = jnp.maximum(distance - yield_distance, 0.0)
    exact = ((0.5 - yield_distance) ** 2 - moving**2) / (2.0 * plastic_viscosity)
    velocity, evidence = _profile_stage(
        phx.rheology.GeneralizedNewtonianLaw(
            "bingham",
            plastic_viscosity,
            consistency_pa_s_n=plastic_viscosity,
            yield_stress_pa=yield_stress,
            regularization_s=1000.0,
        ),
        exact,
        momentum,
        maximum_iterations=200,
    )
    assert evidence.successful, {
        "nonlinear_status": int(evidence.nonlinear.status),
        "fixed_point_residual": float(evidence.nonlinear.diagnostics.final_residual_norm),
        "momentum_residual": float(evidence.relative_residual),
        "linear_status": int(evidence.linear.linear_status),
        "iterations": int(evidence.iterations),
    }
    # The finite regularization has a small creep rate in the nominal plug;
    # identify its width using the independent analytic edge stress threshold.
    derivative = np.diff(np.asarray(velocity[0][0])) * 48
    edge_y = np.arange(1, 48, dtype=np.float64) / 48
    plug = np.abs(derivative) < 0.03
    width = float(np.sum(plug) / 48)
    analytic_plug = np.abs(edge_y - 0.5) < yield_distance
    discrete_reference_width = float(np.sum(analytic_plug) / 48)
    # Two cells cover the two staggered yield surfaces and small creep band.
    # The analytic edge-count oracle has its own continuum-width quantization.
    assert abs(width - discrete_reference_width) <= 2.0 / 48
    assert np.all(np.abs(edge_y[plug] - 0.5) <= yield_distance + 1.0 / 48)
    np.testing.assert_allclose(velocity[0][0], exact, rtol=0.0, atol=0.025)


def _compiled(
    law: phx.equations.FLIPViscousLaw | None,
    *,
    parameters: phx.rheology.MuIParameters | None = None,
    iterations: int = 50,
) -> tuple[phx.equations.CompiledFLIPProblem, Array]:
    momentum = _momentum(8)
    mac = momentum.operators
    position = jnp.asarray(
        [[0.2, 0.2], [0.35, 0.2], [0.2, 0.35], [0.35, 0.35]], dtype=jnp.float64
    )
    particles = D.ParticleSetPlan(
        jnp.arange(4), jnp.full((4,), 0.01), ambient_dimension=2
    ).prepare()
    transfer = D.FLIPParticleTransferPlan(mac).prepare(particles)
    projection = phx.solver.MACFreeSurfaceProjectionPlan(
        mac,
        boundaries=momentum.boundaries,
        tolerance=1.0e-8,
    )
    compiled = phx.equations.compile_flip_problem(
        phx.equations.FLIPProblemIR(
            "viscous-test",
            1.0,
            jnp.asarray([0.1, 0.0]),
            viscous_law=law,
            mu_i_parameters=parameters,
        ),
        transfer,
        projection,
        D.FLIPMethodPlan(
            0.0, liquid_fraction_threshold=0.01, maximum_picard_iterations=iterations
        ),
    )
    return compiled, position


@pytest.mark.parametrize("law_kind", ["newtonian", "mu-i"])
def test_embedded_solid_viscosity_is_refused_but_inviscid_compiles(
    law_kind: str,
) -> None:
    compiled, _ = _compiled(None)
    mac = compiled.projection.operators
    finite_volume = mac.discretization
    cell_fraction = jnp.ones(finite_volume.cell_shape).at[:, 0].set(0.0)
    x_aperture = jnp.ones(finite_volume.face_layouts[0].shape).at[:, 0].set(0.0)
    y_aperture = (
        jnp.ones(finite_volume.face_layouts[1].shape).at[:, 0].set(0.0).at[:, 1].set(0.0)
    )
    source_id = "static-viscous-flip-plane"
    geometry = exact_sharp_geometry(
        finite_volume.cell_volumes * cell_fraction,
        finite_volume.cell_volumes,
        (
            finite_volume.face_measures[0] * x_aperture,
            finite_volume.face_measures[1] * y_aperture,
        ),
        finite_volume.face_measures,
        source_id=source_id,
        source_fidelity="exact-polytope",
        measure_evidence_id="aligned-plane-cell-clipping",
        support_id=finite_volume.support.support_id,
        cell_field_id=finite_volume.cell_space.field_space_id,
        face_field_ids=tuple(space.field_space_id for space in finite_volume.face_spaces),
        operator_id=mac.prepared_id,
        pairing_id=canonical_fingerprint(
            {
                "pressure": mac.pressure_space.space_id,
                "velocity": mac.velocity_space.space_id,
            }
        ),
    )

    def plane(points: Array, time: Array, args: None) -> Array:
        del time, args
        return points[..., 1] - 0.125

    def wall(points: Array, time: Array, args: None) -> Array:
        del time, args
        return jnp.zeros_like(points)

    collision = D.FLIPSolidBoundaryPlan(plane, wall, no_slip=False, field_id=source_id)
    inviscid_problem = phx.equations.FLIPProblemIR(
        "embedded-solid",
        1.0,
        jnp.zeros((2,), dtype=jnp.float64),
        solid_geometry_source_id=source_id,
    )
    inviscid = phx.equations.compile_flip_problem(
        inviscid_problem,
        compiled.transfer,
        compiled.projection,
        compiled.method,
        geometry=geometry,
        solid_boundary=collision,
    )
    assert inviscid.geometry is geometry
    assert inviscid.viscosity is None
    law = (
        phx.rheology.GeneralizedNewtonianLaw(
            "power-law", 0.1, consistency_pa_s_n=0.1, power_index=1.0
        )
        if law_kind == "newtonian"
        else phx.rheology.MuILaw()
    )
    parameters = (
        None
        if law_kind == "newtonian"
        else phx.rheology.MuIParameters(0.38, 0.64, 0.279, 0.01, 2500.0)
    )
    viscous_problem = phx.equations.FLIPProblemIR(
        "embedded-solid-viscous",
        1.0,
        inviscid_problem.acceleration,
        solid_geometry_source_id=source_id,
        viscous_law=law,
        mu_i_parameters=parameters,
    )
    with pytest.raises(
        ValueError,
        match="embedded-wall viscous stage.*cut-cell essential stress affine offsets",
    ):
        phx.equations.compile_flip_problem(
            viscous_problem,
            compiled.transfer,
            compiled.projection,
            compiled.method,
            geometry=geometry,
            solid_boundary=collision,
        )


def test_picard_nonconvergence_rolls_back_particles_pressure_and_clock() -> None:
    law = phx.rheology.GeneralizedNewtonianLaw(
        "carreau-yasuda",
        100.0,
        infinite_shear_viscosity_pa_s=0.01,
        power_index=0.2,
        time_constant_s=10.0,
    )
    compiled, position = _compiled(law, iterations=1)
    velocity = jnp.stack((0.1 * position[:, 1], jnp.zeros(position.shape[0])), axis=-1)
    state = compiled.initialize_state(position, velocity)
    result = eqx.filter_jit(compiled.step_detailed)(state, jnp.asarray(1.0e-3))
    assert result.viscous is not None
    assert not result.viscous.converged
    assert not result.successful
    assert result.viscous.iterations <= 1
    assert result.candidate_state.status == int(D.FLIPRunStatus.VISCOSITY_FAILED)
    assert result.diagnostics.rejection_reason & int(D.FLIPRejectionReason.VISCOSITY)
    for before, after in zip(
        jax.tree.leaves(state), jax.tree.leaves(result.accepted_state), strict=True
    ):
        np.testing.assert_array_equal(before, after)


def test_undefined_yield_viscosity_is_a_rejected_closure() -> None:
    law = phx.rheology.GeneralizedNewtonianLaw(
        "bingham",
        0.1,
        yield_stress_pa=1.0,
        regularization_s=0.0,
    )
    compiled, position = _compiled(law)
    compiled = eqx.tree_at(lambda c: c.problem.acceleration, compiled, jnp.zeros((2,)))
    state = compiled.initialize_state(position, jnp.zeros_like(position))
    result = compiled.step_detailed(state, 1.0e-3)
    assert result.viscous is not None
    assert not result.viscous.closure_successful
    assert not result.successful
    assert result.accepted_state.time == state.time
    assert result.diagnostics.rejection_reason & int(
        D.FLIPRejectionReason.VISCOSITY_CLOSURE
    )


def test_mu_i_requires_initial_physical_pressure_and_lags_accepted_projection() -> None:
    compiled, position = _compiled(
        phx.rheology.MuILaw(),
        parameters=phx.rheology.MuIParameters(0.38, 0.64, 0.279, 0.01, 2500.0),
    )
    compiled = eqx.tree_at(
        lambda c: c.problem.acceleration, compiled, jnp.asarray([0.0, -0.1])
    )
    position = position.at[:, 1].add(-0.1)
    with pytest.raises(ValueError, match="initial_pressure"):
        compiled.initialize_state(position, jnp.zeros_like(position))
    # Zero is an explicitly supplied physical field, not an implicit default;
    # this separated state develops projection pressure on its first step.
    pressure = jnp.zeros(compiled.projection.operators.discretization.cell_shape)
    state = compiled.initialize_state(
        position, jnp.zeros_like(position), initial_pressure=pressure
    )
    result = compiled.step_detailed(state, 1.0e-4)
    assert result.viscous is not None
    assert result.successful
    np.testing.assert_array_equal(result.viscous.lagged_pressure_pa, state.pressure)
    assert jnp.all(result.viscous.viscosity_pa_s == 0.0)
    assert jnp.any(result.accepted_state.pressure != 0.0)
    second = compiled.step_detailed(result.accepted_state, 1.0e-4)
    assert second.viscous is not None
    np.testing.assert_array_equal(
        second.viscous.lagged_pressure_pa, result.accepted_state.pressure
    )


def test_mu_i_all_liquid_gauge_is_not_physical_pressure() -> None:
    compiled, _ = _compiled(
        phx.rheology.MuILaw(),
        parameters=phx.rheology.MuIParameters(0.38, 0.64, 0.279, 0.01, 2500.0),
    )
    mac = compiled.projection.operators
    x, y = (axis.interval_centers for axis in mac.discretization.grid.structured_axes)
    coordinates = jnp.meshgrid(x, y, indexing="ij")
    position = jnp.stack(coordinates, axis=-1).reshape((-1, 2))
    particles = D.ParticleSetPlan(
        jnp.arange(position.shape[0]),
        jnp.full((position.shape[0],), 1.0),
        ambient_dimension=2,
    ).prepare()
    transfer = D.FLIPParticleTransferPlan(mac).prepare(particles)
    full = phx.equations.compile_flip_problem(
        compiled.problem, transfer, compiled.projection, compiled.method
    )
    with pytest.raises(eqx.EquinoxRuntimeError, match="free surface"):
        full.initialize_state(
            position, jnp.zeros_like(position), initial_pressure=jnp.zeros((4, 8))
        )


def test_no_law_preserves_exact_inviscid_results() -> None:
    compiled, position = _compiled(None)
    state = compiled.initialize_state(position, jnp.zeros_like(position))
    first = compiled.step_detailed(state, 1.0e-4)
    second = eqx.filter_jit(compiled.step_detailed)(state, jnp.asarray(1.0e-4))
    assert first.successful and second.successful
    assert first.viscous is None and second.viscous is None
    for before, after in zip(
        jax.tree.leaves(first.accepted_state),
        jax.tree.leaves(second.accepted_state),
        strict=True,
    ):
        np.testing.assert_array_equal(before, after)


def test_compact_viscosity_is_explicitly_refused_without_dense_fallback() -> None:
    grid = D.TensorGridPlan(
        (
            D.UniformCellAxisSpec(8, periodic=True),
            D.UniformCellAxisSpec(8, periodic=True),
        ),
        axis_names=("x", "y"),
    )
    index = grid.prepare_index_space(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))
    particles = D.ParticleSetPlan(
        jnp.arange(2), jnp.ones((2,)), ambient_dimension=2
    ).prepare()
    cell = D.SparseBlockTopologyPlan(index, (2, 2), 8, layout=index.cells())
    faces = tuple(
        D.SparseBlockTopologyPlan(index, (2, 2), 8, layout=index.faces(axis))
        for axis in index.axis_names
    )
    transfer = D.SparseFLIPParticleTransferPlan(index, cell, faces).prepare(particles)
    projection = phx.solver.SparseMACFreeSurfaceProjectionPlan(transfer)
    problem = phx.equations.FLIPProblemIR(
        "compact-viscous",
        1.0,
        jnp.zeros((2,)),
        viscous_law=phx.rheology.GeneralizedNewtonianLaw("power-law", 1.0),
    )
    with pytest.raises(ValueError, match="Compact FLIP does not support viscous_law"):
        phx.equations.compile_sparse_flip_problem(
            problem, transfer, projection, D.FLIPMethodPlan(0.0)
        )


def test_mu_i_projection_density_must_match_physical_pressure_units() -> None:
    compiled, _ = _compiled(
        phx.rheology.MuILaw(),
        parameters=phx.rheology.MuIParameters(0.38, 0.64, 0.279, 0.01, 2500.0),
    )
    incompatible = phx.solver.MACFreeSurfaceProjectionPlan(
        compiled.projection.operators,
        boundaries=compiled.projection.boundaries,
        density=10.0,
    )
    with pytest.raises(ValueError, match="matching projection density"):
        phx.equations.compile_flip_problem(
            compiled.problem, compiled.transfer, incompatible, compiled.method
        )


def test_viscous_preparation_refuses_resource_policy_overflow() -> None:
    compiled, _ = _compiled(None)
    problem = phx.equations.FLIPProblemIR(
        "resource-viscous",
        1.0,
        jnp.zeros((2,)),
        viscous_law=phx.rheology.GeneralizedNewtonianLaw("power-law", 1.0),
    )
    method = D.FLIPMethodPlan(
        0.0, resource_policy=D.FLIPResourcePolicy(maximum_workspace_bytes=1)
    )
    with pytest.raises(ValueError, match="resource policy"):
        phx.equations.compile_flip_problem(
            problem, compiled.transfer, compiled.projection, method
        )


def test_liquid_air_surface_is_natural_traction_free_not_zero_velocity() -> None:
    momentum = _momentum()
    mac = momentum.operators
    y = mac.discretization.grid.structured_axes[1].interval_centers
    liquid = jnp.broadcast_to(y < 0.5, mac.discretization.cell_shape)
    measures = D.finite_volume.MACFreeSurfaceViscousMeasurePlan(
        mac, 1.0
    ).evaluate_cell_fraction(liquid.astype(jnp.float64), 0.0)
    forced = (
        jnp.full(mac.discretization.face_layouts[0].shape, 0.1),
        jnp.zeros(mac.discretization.face_layouts[1].shape),
    )
    velocity, evidence = viscous_flip_stage(
        phx.solver.MACVariationalViscosityPlan(momentum, tolerance=1.0e-11),
        phx.rheology.GeneralizedNewtonianLaw("power-law", 0.2, consistency_pa_s_n=0.2),
        None,
        D.FLIPMethodPlan(0.0),
        measures,
        forced,
        jnp.zeros_like(y)[None, :].repeat(4, axis=0),
        liquid,
        jnp.asarray(0.1),
        momentum.boundaries.evaluate(0.0),
        jnp.asarray(True),
    )
    assert evidence.successful
    assert evidence.linear.decoupled_face_count > 0
    assert jnp.all(velocity[0][:, 16] > 0.0)
    np.testing.assert_allclose(velocity[0][:, 15], velocity[0][:, 16], atol=1.0e-8)


@pytest.mark.parametrize("viscosity", [1.0e-3, 1.0e3], ids=["low-eta", "high-eta"])
def test_viscous_hydrostatic_rest_preserves_discrete_pressure(viscosity: float) -> None:
    """Accepted pressure supports gravity independently of implicit viscosity."""
    nx, ny, air = 4, 8, 4
    length, height, density, gravity = 0.08, 0.08, 1500.0, 9.81
    dx, dy = length / nx, height / ny
    grid = D.TensorGridPlan(
        (D.UniformCellAxisSpec(nx, periodic=True), D.UniformCellAxisSpec(ny + air)),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [length, height + air * dy]], dtype=jnp.float64))
    mac = D.MACOperatorPlan(D.FiniteVolumePlan(grid).prepare()).prepare()
    boundaries = D.MACBoundaryPlan(
        mac,
        (
            D.MACBoundarySide("y", "lower", "no-slip"),
            D.MACBoundarySide("y", "upper", "free-slip"),
        ),
    ).prepare()
    x = (np.arange(2 * nx, dtype=np.float64) + 0.5) * dx / 2
    y = (np.arange(ny, dtype=np.float64) + 0.75) * dy
    xx, yy = np.meshgrid(x, y, indexing="ij")
    position = jnp.asarray(
        np.stack((xx, yy), axis=-1).reshape((-1, 2)), dtype=jnp.float64
    )
    particles = D.ParticleSetPlan(
        jnp.arange(position.shape[0], dtype=jnp.int32),
        jnp.full((position.shape[0],), density * dx * dy / 2, dtype=jnp.float64),
        ambient_dimension=2,
    ).prepare()
    transfer = D.FLIPParticleTransferPlan(mac).prepare(particles)
    projection = phx.solver.MACFreeSurfaceProjectionPlan(
        mac,
        boundaries=boundaries,
        density=density,
        tolerance=1.0e-12,
        maximum_iterations=500,
    )
    compiled = phx.equations.compile_flip_problem(
        phx.equations.FLIPProblemIR(
            "incremental-hydrostatic",
            density,
            jnp.asarray([0.0, -gravity], dtype=jnp.float64),
            viscous_law=phx.rheology.GeneralizedNewtonianLaw(
                "power-law", viscosity, consistency_pa_s_n=viscosity
            ),
        ),
        transfer,
        projection,
        D.FLIPMethodPlan(
            0.0,
            liquid_fraction_threshold=0.5,
            maximum_picard_iterations=100,
            viscous_tolerance=1.0e-8,
            maximum_viscous_linear_iterations=500,
        ),
    )
    # The atmospheric datum is the first AIR CELL, not the geometric top face.
    # dp/dy=-rho*g on every liquid face gives p_j=rho*g*dy*(N-j).
    reference = density * gravity * dy * (ny - np.arange(ny, dtype=np.float64))
    pressure = jnp.broadcast_to(
        jnp.asarray(np.concatenate((reference, np.zeros(air))), dtype=jnp.float64),
        mac.discretization.cell_shape,
    )
    state = compiled.initialize_state(
        position, jnp.zeros_like(position), initial_pressure=pressure
    )
    advance = eqx.filter_jit(compiled.step_detailed)
    for _ in range(2):
        result = advance(state, jnp.asarray(0.01, dtype=jnp.float64))
        assert result.viscous is not None
        assert result.successful, {
            "rejection": int(result.diagnostics.rejection_reason),
            "relative_viscous_residual": float(result.viscous.relative_residual),
            "viscous_residual": float(result.viscous.residual_norm),
            "nonlinear_status": int(result.viscous.nonlinear.status),
            "linear_status": int(result.viscous.linear.linear_status),
            "projection": bool(result.diagnostics.details.successful),
        }
        assert result.viscous.converged
        observed = np.asarray(result.accepted_state.pressure[:, :ny])
        relative_error = np.linalg.norm(observed - reference[None, :]) / (
            np.sqrt(nx) * np.linalg.norm(reference)
        )
        assert relative_error <= 1.0e-10, f"Hydrostatic pressure gap: {relative_error}"
        np.testing.assert_allclose(
            result.post_grid_velocity[1][:, : ny + 1], 0.0, rtol=0.0, atol=1.0e-12
        )
        np.testing.assert_allclose(
            result.accepted_state.particles.velocity, 0.0, rtol=0.0, atol=1.0e-12
        )
        np.testing.assert_allclose(
            result.accepted_state.particles.position, position, rtol=0.0, atol=1.0e-14
        )
        projection_evidence = result.diagnostics.details
        np.testing.assert_allclose(
            projection_evidence.pressure_increment,
            projection_evidence.pressure - state.pressure,
            rtol=0.0,
            atol=1.0e-12,
        )
        state = result.accepted_state
