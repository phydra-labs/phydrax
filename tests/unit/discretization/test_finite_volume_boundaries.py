#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import math

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _bounded_grid(count=8):
    return phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(count),), axis_names=("x",)
    ).prepare(jnp.asarray([[0.0], [1.0]]))


def _scalar_system():
    return phx.equations.ScalarConservationSystem(
        1,
        lambda state, axis, args: state,
        lambda left, right, axis, args: jnp.ones(left.shape[:-1]),
        system_id="unit-linear-advection",
    )


def test_piecewise_constant_reconstruction_uses_explicit_exterior_states():
    state = jnp.arange(4.0)[:, None]
    left, right = phx.discretization.PiecewiseConstantReconstruction().reconstruct_axis(
        state,
        0,
        periodic=False,
        lower_exterior=jnp.asarray([-2.0]),
        upper_exterior=jnp.asarray([7.0]),
    )

    np.testing.assert_allclose(left[:, 0], [-2.0, 0.0, 1.0, 2.0, 3.0])
    np.testing.assert_allclose(right[:, 0], [0.0, 1.0, 2.0, 3.0, 7.0])


def test_muscl_reconstructs_linear_cell_average_at_interior_faces():
    grid = _bounded_grid(12)
    centers = grid.structured_axes[0].interval_centers
    state = (1.5 + 2.0 * centers)[:, None]
    lower = jnp.asarray([1.5 - 2.0 * 0.5 / 12.0])
    upper = jnp.asarray([3.5 + 2.0 * 0.5 / 12.0])
    left, right = phx.discretization.MUSCLReconstruction(
        phx.discretization.MCLimiter()
    ).reconstruct_axis(
        state,
        0,
        periodic=False,
        lower_exterior=lower,
        upper_exterior=upper,
        cell_widths=grid.structured_axes[0].interval_widths,
    )
    exact = 1.5 + 2.0 * grid.structured_axes[0].point_coordinates

    np.testing.assert_allclose(left[1:-1, 0], exact[1:-1], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(right[1:-1, 0], exact[1:-1], rtol=1e-12, atol=1e-12)


def test_rusanov_flux_is_consistent_and_orientation_reversing():
    system = _scalar_system()
    solver = phx.discretization.RusanovFluxPlan()
    state = jnp.asarray([[0.2], [0.7], [-0.3]])

    equal = solver.face_flux(system, state, state, 0)
    positive = solver.normal_face_flux(system, state, state, jnp.ones((3, 1)))
    negative = solver.normal_face_flux(system, state, state, -jnp.ones((3, 1)))

    np.testing.assert_allclose(equal.normal_flux, state)
    np.testing.assert_allclose(positive.normal_flux, -negative.normal_flux)
    assert jnp.all(equal.max_speed >= 0.0)


def test_hllc_preserves_stationary_euler_contact_flux():
    system = phx.equations.EulerSystem()
    primitive_left = jnp.asarray([[1.0, 0.0, 1.0]])
    primitive_right = jnp.asarray([[0.3, 0.0, 1.0]])
    result = phx.discretization.HLLCFluxPlan().face_flux(
        system,
        system.primitive_to_conserved(primitive_left),
        system.primitive_to_conserved(primitive_right),
        0,
    )

    np.testing.assert_allclose(result.normal_flux, [[0.0, 1.0, 0.0]], atol=1e-12)


def test_bounded_compilation_requires_complete_boundary_pairs():
    grid = _bounded_grid()
    discretization = phx.discretization.FiniteVolumePlan(grid).prepare()
    system = _scalar_system()
    boundaries = phx.discretization.FiniteVolumeBoundarySet(("x",), (None,))
    method = phx.discretization.FiniteVolumeMethodPlan(
        phx.discretization.PiecewiseConstantReconstruction(),
        phx.discretization.RusanovFluxPlan(),
    )
    problem = phx.equations.ConservationProblemIR(
        "advection", "state", system, boundaries
    )

    with pytest.raises(ValueError, match="boundary pair"):
        phx.equations.compile_conservation_problem(problem, discretization, method)


def test_prescribed_outward_flux_controls_global_balance():
    grid = _bounded_grid(10)
    discretization = phx.discretization.FiniteVolumePlan(grid).prepare()
    system = _scalar_system()
    lower = phx.discretization.PrescribedNormalFluxBoundary(
        lambda time, interior, coordinates, normal, args: jnp.asarray([2.0]),
        boundary_id="lower-flux",
    )
    upper = phx.discretization.PrescribedNormalFluxBoundary(
        lambda time, interior, coordinates, normal, args: jnp.asarray([5.0]),
        boundary_id="upper-flux",
    )
    boundaries = phx.discretization.FiniteVolumeBoundarySet(
        ("x",), (phx.discretization.FiniteVolumeBoundaryPair(lower, upper),)
    )
    method = phx.discretization.FiniteVolumeMethodPlan(
        phx.discretization.PiecewiseConstantReconstruction(),
        phx.discretization.RusanovFluxPlan(),
    )
    problem = phx.equations.ConservationProblemIR(
        "forced-flux", "state", system, boundaries
    )
    compiled = phx.equations.compile_conservation_problem(problem, discretization, method)
    residual, diagnostics = compiled.residual_with_diagnostics(
        0.0, jnp.ones(discretization.state_shape)
    )
    balance_terms = np.asarray(discretization.cell_volumes[..., None] * residual)
    expected_defect = np.asarray(
        [
            math.fsum(
                balance_terms[:, index].tolist()
                + [
                    float(diagnostics.boundary_outward_flux[index]),
                    -float(diagnostics.source_integral[index]),
                ]
            )
            for index in range(balance_terms.shape[1])
        ]
    )
    np.testing.assert_array_equal(diagnostics.conservation_defect, expected_defect)

    np.testing.assert_allclose(
        jnp.sum(discretization.cell_volumes[..., None] * residual, axis=0),
        [-7.0],
        atol=1e-12,
    )
    np.testing.assert_allclose(diagnostics.conservation_defect, [0.0], atol=1e-12)


def test_typed_boundary_trace_distinguishes_state_and_direct_flux():
    system = _scalar_system()
    interior = jnp.asarray(((0.2,), (0.7,)))
    points = jnp.asarray(((0.0,), (1.0,)))
    normal = jnp.ones((2, 1))
    state_trace = phx.discretization.evaluate_conservation_boundary(
        phx.discretization.ExtrapolationBoundary(),
        system,
        jnp.asarray(0.0),
        interior,
        points,
        normal,
        0,
        None,
    )
    np.testing.assert_allclose(state_trace.exterior_state, interior)
    assert state_trace.direct_normal_flux is None
    assert jnp.all(state_trace.admissible)

    direct = phx.discretization.evaluate_conservation_boundary(
        phx.discretization.PrescribedNormalFluxBoundary(
            lambda time, state, coordinates, outward_normal, args: 2.0 * state,
            boundary_id="direct-flux",
        ),
        system,
        jnp.asarray(0.0),
        interior,
        points,
        normal,
        0,
        None,
    )
    assert direct.exterior_state is None
    np.testing.assert_allclose(direct.direct_normal_flux, 2.0 * interior)
