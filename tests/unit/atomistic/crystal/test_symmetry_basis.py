#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Sparse symmetry tangents, adjoints, retractions and residuals.

Independent controls: central finite differences of ``expand o retract``
against the operator action, the pairing identity ``<T v, w> = <v, T^T w>``,
the Cholesky-chart derivative against finite differences of the native cell,
exact preservation of special-site compatibility after finite steps, and a
deliberately broken atom detected by the symmetry residual.  The rigid
retraction is differentiated by AD at the anchor (``F = I``, repeated singular
values of the frame change), at nearby admissible cells and to second order,
against the tangent operator, central differences and the JVP/VJP pairing.
The route count is checked against the exact per-atom block structure, far
below a dense ``(3N + 9) x K`` basis.
"""

from collections.abc import Callable
from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from phydrax.atomistic.crystal import (
    AsymmetricUnitPlan,
    AsymmetricUnitSite,
    AsymmetricUnitState,
    atomic_state_from_expansion,
    atomic_tangent,
    AtomicSymmetryPlan,
    expand_asymmetric_unit,
    expand_symmetric_atoms,
    metric_from_cell_parameters,
    retract_atomic,
    RigidSymmetryLayout,
    site_of_point,
    standard_space_group_setting,
    strain_generators,
    symmetry_residual,
    symmetry_residuals,
)
from phydrax.sparse import SparseCoordinateOperator
from tests._support.crystal_geometry_molecules import (
    argon,
    chlorofluorobromomethane,
    nontrivial_automorphism,
    trans_dichloroethene,
)


_HALF = float(np.sqrt(0.5))
_STEP = 1.0e-6


def _plan_and_state() -> tuple[AsymmetricUnitPlan, AsymmetricUnitState]:
    setting = standard_space_group_setting(3)
    planar = trans_dichloroethene()
    swap = nontrivial_automorphism(planar)
    general = site_of_point(setting, (Fraction(1, 5), Fraction(1, 3), Fraction(1, 7)))
    plan = AsymmetricUnitPlan(
        setting,
        [
            AsymmetricUnitSite(
                planar,
                site_of_point(setting, (0, 0, 0)),
                site_permutations=[tuple(range(6)), swap],
            ),
            AsymmetricUnitSite(chlorofluorobromomethane(), general),
            AsymmetricUnitSite(argon(), general),
        ],
        chirality="enantiopure",
    )
    quaternion = np.asarray((0.6, 0.3, -0.5, 0.55))
    quaternion /= np.linalg.norm(quaternion)
    metric = metric_from_cell_parameters(
        jnp.asarray((6.4, 7.3, 8.1), dtype=jnp.float64),
        jnp.asarray((90.0, 104.0, 90.0), dtype=jnp.float64),
    )
    state = AsymmetricUnitState(
        jnp.asarray([(0.0, 0.37, 0.0), (0.21, 0.13, 0.64), (0.55, 0.81, 0.29)]),
        jnp.asarray([(_HALF, -_HALF, 0.0, 0.0), quaternion, (1.0, 0.0, 0.0, 0.0)]),
        plan.chart.coefficients(metric),
        jnp.zeros((3,), dtype=jnp.int32),
    )
    return plan, state


def _flat(positions: Array, vectors: Array) -> np.ndarray:
    return np.concatenate(
        (np.asarray(positions).reshape(-1), np.asarray(vectors).reshape(-1))
    )


def test_rigid_layout_is_sparse_per_atom_blocks() -> None:
    plan, state = _plan_and_state()
    layout = RigidSymmetryLayout(plan)
    assert layout.center_dimension == (1, 3, 3)
    assert layout.orientation_dimension == (1, 3, 0)
    assert layout.parameter_count == 2 + 6 + 3 + plan.chart.dimension == 15
    operator, _ = layout.tangent(state)
    assert isinstance(operator.site_operator, SparseCoordinateOperator)
    assert plan.atom_count == 18
    assert operator.route_count == 3 * (6 * 2 + 10 * 6 + 2 * 3)
    assert operator.route_count < (3 * plan.atom_count + 9) * layout.parameter_count


def test_rigid_tangent_matches_finite_differences_of_the_retraction() -> None:
    plan, state = _plan_and_state()
    layout = RigidSymmetryLayout(plan)
    operator, _ = layout.tangent(state)
    direction = np.random.default_rng(3).normal(size=layout.parameter_count)

    def evaluate(scale: float) -> np.ndarray:
        moved = layout.retract(state, jnp.asarray(scale * direction))
        expansion = expand_asymmetric_unit(plan, moved)
        return _flat(expansion.positions, expansion.cell.vectors)

    finite = (evaluate(_STEP) - evaluate(-_STEP)) / (2.0 * _STEP)
    np.testing.assert_allclose(
        np.asarray(operator.mv(jnp.asarray(direction))), finite, atol=2e-7
    )


def test_rigid_tangent_adjoint_identity() -> None:
    plan, state = _plan_and_state()
    operator, _ = RigidSymmetryLayout(plan).tangent(state)
    rng = np.random.default_rng(5)
    v = jnp.asarray(rng.normal(size=operator.source.size))
    w = jnp.asarray(rng.normal(size=operator.target.size))
    left = float(jnp.dot(operator.mv(v), w))
    right = float(jnp.dot(v, operator.transpose_mv(w)))
    assert left == pytest.approx(right, rel=1e-12, abs=1e-12)
    np.testing.assert_allclose(
        np.asarray(operator.adjoint_mv(w)),
        np.asarray(operator.transpose_mv(w)),
        rtol=1e-14,
    )


def test_finite_rigid_steps_keep_special_site_compatibility() -> None:
    plan, state = _plan_and_state()
    layout = RigidSymmetryLayout(plan)
    step = 0.05 * np.random.default_rng(7).normal(size=layout.parameter_count)
    moved = layout.retract(state, jnp.asarray(step))
    expansion = expand_asymmetric_unit(plan, moved)
    assert bool(expansion.cell.successful)
    assert float(expansion.orientation_residual[0]) < 1e-10
    assert float(expansion.center_residual[0]) < 1e-12
    assert np.allclose(np.linalg.norm(np.asarray(moved.orientations), axis=1), 1.0)


def test_refused_frame_change_reports_failed_expansion_without_raising() -> None:
    plan, state = _plan_and_state()
    layout = RigidSymmetryLayout(plan)
    step = np.zeros((layout.parameter_count,))
    # A non-positive metric: the moved cell and its frame change are refused.
    step[layout.strain_start :] = -2.0 * np.asarray(state.cell_coefficients)
    moved = layout.retract(state, jnp.asarray(step))
    assert np.all(np.isnan(np.asarray(moved.orientations)))
    expansion = expand_asymmetric_unit(plan, moved)
    assert not bool(expansion.cell.successful)
    assert not bool(expansion.offset_solve_successful)


def test_strain_generators_differentiate_the_cholesky_chart() -> None:
    plan, state = _plan_and_state()
    vectors = plan.chart.cell(state.cell_coefficients).vectors
    generators = np.asarray(strain_generators(plan.chart, vectors))
    for k in range(plan.chart.dimension):
        offset = np.zeros(plan.chart.dimension)
        offset[k] = _STEP
        plus = np.asarray(plan.chart.cell(state.cell_coefficients + offset).vectors)
        minus = np.asarray(plan.chart.cell(state.cell_coefficients - offset).vectors)
        np.testing.assert_allclose(
            np.asarray(vectors) @ generators[k].T,
            (plus - minus) / (2.0 * _STEP),
            atol=1e-7,
        )


def _realization(
    layout: RigidSymmetryLayout, state: AsymmetricUnitState
) -> Callable[[Array], Array]:
    def realize(step: Array) -> Array:
        expansion = expand_asymmetric_unit(layout.unit, layout.retract(state, step))
        return jnp.concatenate(
            (expansion.positions.reshape(-1), expansion.cell.vectors.reshape(-1))
        )

    return realize


def _central_columns(
    realize: Callable[[Array], Array], point: np.ndarray, columns: range
) -> np.ndarray:
    finite = []
    for k in columns:
        offset = np.zeros_like(point)
        offset[k] = _STEP
        plus = np.asarray(realize(jnp.asarray(point + offset)))
        minus = np.asarray(realize(jnp.asarray(point - offset)))
        finite.append((plus - minus) / (2.0 * _STEP))
    return np.stack(finite, axis=1)


def test_rigid_retraction_has_finite_exact_strain_derivatives_at_the_anchor() -> None:
    plan, state = _plan_and_state()
    layout = RigidSymmetryLayout(plan)
    realize = _realization(layout, state)
    zero = np.zeros((layout.parameter_count,))
    forward = np.asarray(jax.jacfwd(realize)(jnp.asarray(zero)))
    reverse = np.asarray(jax.jacrev(realize)(jnp.asarray(zero)))
    assert np.all(np.isfinite(forward)) and np.all(np.isfinite(reverse))
    np.testing.assert_allclose(reverse, forward, rtol=0.0, atol=1e-11)
    operator, _ = layout.tangent(state)
    columns = np.asarray(
        jax.vmap(operator.mv, out_axes=1)(jnp.eye(layout.parameter_count))
    )
    np.testing.assert_allclose(forward, columns, atol=1e-10)
    strain = range(layout.strain_start, layout.parameter_count)
    np.testing.assert_allclose(
        forward[:, layout.strain_start :],
        _central_columns(realize, zero, strain),
        atol=2e-7,
    )


def test_rigid_retraction_derivatives_at_nearby_admissible_cells() -> None:
    plan, state = _plan_and_state()
    layout = RigidSymmetryLayout(plan)
    rng = np.random.default_rng(23)
    nearby = layout.retract(
        state, jnp.asarray(0.03 * rng.normal(size=layout.parameter_count))
    )
    assert bool(expand_asymmetric_unit(plan, nearby).cell.successful)
    point = 0.02 * rng.normal(size=layout.parameter_count)
    for anchor, step in ((nearby, np.zeros_like(point)), (state, point)):
        realize = _realization(layout, anchor)
        forward = np.asarray(jax.jacfwd(realize)(jnp.asarray(step)))
        assert np.all(np.isfinite(forward))
        np.testing.assert_allclose(
            forward,
            _central_columns(realize, step, range(layout.parameter_count)),
            atol=2e-7,
        )
        v = jnp.asarray(rng.normal(size=layout.parameter_count))
        w = jnp.asarray(rng.normal(size=forward.shape[0]))
        value, tangent = jax.jvp(realize, (jnp.asarray(step),), (v,))
        _, pullback = jax.vjp(realize, jnp.asarray(step))
        (cotangent,) = pullback(w)
        assert float(jnp.dot(tangent, w)) == pytest.approx(
            float(jnp.dot(v, cotangent)), rel=1e-12
        )
        assert np.all(np.isfinite(np.asarray(value)))


def test_rigid_retraction_second_derivative_matches_gradient_differences() -> None:
    plan, state = _plan_and_state()
    layout = RigidSymmetryLayout(plan)
    realize = _realization(layout, state)
    rng = np.random.default_rng(29)
    weights = jnp.asarray(rng.normal(size=3 * plan.atom_count + 9))
    direction = np.zeros((layout.parameter_count,))
    direction[layout.strain_start :] = rng.normal(
        size=layout.parameter_count - layout.strain_start
    )

    def gradient(step: Array) -> Array:
        return jax.grad(lambda value: jnp.dot(weights, realize(value)))(step)

    zero = jnp.zeros((layout.parameter_count,))
    _, curvature = jax.jvp(gradient, (zero,), (jnp.asarray(direction),))
    curvature = np.asarray(curvature)
    assert np.all(np.isfinite(curvature))
    step = 1.0e-5
    finite = (
        np.asarray(gradient(jnp.asarray(step * direction)))
        - np.asarray(gradient(jnp.asarray(-step * direction)))
    ) / (2.0 * step)
    np.testing.assert_allclose(
        curvature, finite, rtol=1e-5, atol=1e-5 * float(np.max(np.abs(finite)))
    )


def test_atomic_chart_reproduces_expansion_and_its_tangent() -> None:
    plan, state = _plan_and_state()
    layout = AtomicSymmetryPlan(plan)
    assert layout.representative_count == 3 + 5 + 1
    assert layout.representative_dimension == (3,) * 9
    expansion = expand_asymmetric_unit(plan, state)
    atomic = atomic_state_from_expansion(layout, state, expansion)
    np.testing.assert_allclose(
        np.asarray(expand_symmetric_atoms(layout, atomic).positions),
        np.asarray(expansion.positions),
        atol=1e-10,
    )
    operator, _ = atomic_tangent(layout, atomic)
    direction = np.random.default_rng(11).normal(size=layout.parameter_count)

    def evaluate(scale: float) -> np.ndarray:
        moved = expand_symmetric_atoms(
            layout, retract_atomic(layout, atomic, jnp.asarray(scale * direction))
        )
        return _flat(moved.positions, moved.cell.vectors)

    finite = (evaluate(_STEP) - evaluate(-_STEP)) / (2.0 * _STEP)
    np.testing.assert_allclose(
        np.asarray(operator.mv(jnp.asarray(direction))), finite, atol=2e-7
    )
    rng = np.random.default_rng(13)
    v = jnp.asarray(rng.normal(size=operator.source.size))
    w = jnp.asarray(rng.normal(size=operator.target.size))
    assert float(jnp.dot(operator.mv(v), w)) == pytest.approx(
        float(jnp.dot(v, operator.transpose_mv(w))), rel=1e-12
    )


def test_symmetry_residual_detects_broken_symmetry_and_reports_both_residuals() -> None:
    plan, state = _plan_and_state()
    layout = AtomicSymmetryPlan(plan)
    expansion = expand_asymmetric_unit(plan, state)
    vectors = expansion.cell.vectors
    intact = np.asarray(symmetry_residual(layout, expansion.positions, vectors))
    assert np.max(intact) < 1e-10
    broken_positions = np.asarray(expansion.positions).copy()
    broken_positions[1] += np.asarray((0.03, -0.04, 0.0))
    broken = np.asarray(symmetry_residual(layout, broken_positions, vectors))
    assert broken[1] == pytest.approx(0.05, rel=1e-8)
    assert np.max(np.delete(broken, 1)) < 1e-10
    operator, _ = RigidSymmetryLayout(plan).tangent(state)
    rng = np.random.default_rng(17)
    forces = jnp.asarray(rng.normal(size=(plan.atom_count, 3)))
    cell = jnp.asarray(rng.normal(size=(3, 3)))
    report = symmetry_residuals(operator, forces, cell)
    np.testing.assert_allclose(
        np.asarray(report.projected),
        np.asarray(operator.transpose_mv(operator.join(forces, cell))),
    )
    assert float(report.full_force) == pytest.approx(
        float(np.max(np.linalg.norm(forces, axis=1)))
    )
    assert float(report.full_cell_force) == pytest.approx(float(np.linalg.norm(cell)))
