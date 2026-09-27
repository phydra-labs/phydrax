#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


import itertools
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax._meshcore import meshcore_available
from phydrax.geometry import (
    CommonRefinementCoverage,
    CommonRefinementPolicy,
    prepare_common_refinement,
)


pytestmark = pytest.mark.skipif(
    not meshcore_available(), reason="Common refinement requires meshcore."
)

_KUHN = tuple(itertools.permutations(range(3)))


def _triangles(
    n: Any, *, lower: Any = 0.0, upper: Any = 1.0, perturb: Any = 0.0, seed: Any = 0
) -> Any:
    axis = np.linspace(lower, upper, n + 1)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    points = np.stack((x.ravel(), y.ravel()), axis=-1)
    interior = np.all((points > lower) & (points < upper), axis=1)
    offsets = np.random.default_rng(seed).uniform(-1.0, 1.0, (interior.sum(), 2))
    points[interior] += perturb * (upper - lower) / n * offsets
    corner = (np.arange(n)[:, None] * (n + 1) + np.arange(n)[None, :]).ravel()
    cells = np.concatenate(
        (
            np.stack((corner, corner + n + 1, corner + n + 2), axis=-1),
            np.stack((corner, corner + n + 2, corner + 1), axis=-1),
        )
    )
    return phx.discretization.CellMesh.from_triangles(
        jnp.asarray(points), jnp.asarray(cells, dtype=jnp.int32)
    )


def _tetrahedra(n: Any) -> Any:
    axis = np.linspace(0.0, 1.0, n + 1)
    points = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3)
    stride = np.asarray(((n + 1) ** 2, n + 1, 1))
    origins = np.stack(np.meshgrid(*(np.arange(n),) * 3, indexing="ij"), -1).reshape(
        -1, 3
    )
    cells = []
    for order in _KUHN:
        path = [np.zeros(3, dtype=np.int64)]
        for axis_index in order:
            path.append(path[-1] + np.eye(3, dtype=np.int64)[axis_index])
        cells.append(np.stack([(origins + step) @ stride for step in path], axis=-1))
    cells = np.concatenate(cells)
    corners = points[cells]
    negative = np.linalg.det(corners[:, 1:] - corners[:, :1]) < 0.0
    cells[negative] = cells[negative][:, (0, 1, 3, 2)]
    return phx.discretization.CellMesh.from_tetrahedra(
        jnp.asarray(points), jnp.asarray(cells, dtype=jnp.int32)
    )


def _space(mesh: Any, kind: Any, degree: Any = 1, *, discontinuous: Any = False) -> Any:
    element = (
        phx.discretization.discontinuous_element(kind, degree)
        if discontinuous
        else phx.discretization.lagrange_element(kind, degree)
    )
    return phx.discretization.FiniteElementPlan(
        mesh, phx.discretization.FiniteElementFieldSpec("u", element)
    ).prepare()


def _refinement(source: Any, target: Any, **policy: Any) -> Any:
    return prepare_common_refinement(
        source,
        target,
        policy=CommonRefinementPolicy(overlap_simplices=True, **policy),
    )


def _dofs(space: Any) -> Any:
    return np.asarray(space.dof_maps[0].dof_coordinates)


def _integrals(space: Any) -> Any:
    return np.asarray(space.mass.mv(jnp.ones((space.dof_maps[0].global_dof_count,))))


def _smooth(points: Any) -> Any:
    return np.sin(3.0 * points[:, 0]) * np.exp(points[:, 1]) + points[:, 0] ** 2


def _transfer(source: Any, target: Any, refinement: Any) -> Any:
    prepared = phx.discretization.prepare_l2_projection_target(target, field_name="u")
    return phx.discretization.prepare_l2_projection_transfer(
        source, prepared, refinement, field_name="u"
    )


@pytest.fixture(scope="module")
def planar() -> Any:
    source_mesh = _triangles(7, perturb=0.3, seed=1)
    target_mesh = _triangles(5, perturb=0.3, seed=2)
    source = _space(source_mesh, "triangle")
    target = _space(target_mesh, "triangle")
    transfer = _transfer(source, target, _refinement(source_mesh, target_mesh))
    return source, target, transfer


def test_projection_onto_the_same_space_is_the_identity() -> None:
    mesh = _triangles(4, perturb=0.25, seed=3)
    space = _space(mesh, "triangle", 2)
    transfer = _transfer(space, space, _refinement(mesh, mesh))
    values = jnp.asarray(_smooth(_dofs(space)))
    np.testing.assert_allclose(transfer.apply(values), values, atol=1e-12)


def test_target_space_fields_are_reproduced_and_claims_are_certified(planar: Any) -> None:
    source, target, transfer = planar
    assert transfer.preserves_constants and transfer.preserves_linear
    assert transfer.conservative and not transfer.positivity_preserving
    linear = lambda x: 0.5 - 2.0 * x[:, 0] + 3.0 * x[:, 1]
    projected = transfer.apply(jnp.asarray(linear(_dofs(source))))
    np.testing.assert_allclose(projected, linear(_dofs(target)), atol=1e-12)


def test_linear_source_is_reproduced_by_a_higher_degree_target() -> None:
    source_mesh = _triangles(5, perturb=0.3, seed=4)
    target_mesh = _triangles(3, perturb=0.3, seed=5)
    source = _space(source_mesh, "triangle")
    target = _space(target_mesh, "triangle", 2)
    transfer = _transfer(source, target, _refinement(source_mesh, target_mesh))
    linear = lambda x: 1.0 + x[:, 0] - 0.25 * x[:, 1]
    projected = transfer.apply(jnp.asarray(linear(_dofs(source))))
    np.testing.assert_allclose(projected, linear(_dofs(target)), atol=1e-12)


def test_projection_error_is_orthogonal_to_the_target_space(planar: Any) -> None:
    source, target, transfer = planar
    values = jnp.asarray(_smooth(_dofs(source)))
    projected = transfer.apply(values)
    projection = transfer.primal
    residual = projection.mixed_mass.mv(values) - target.mass.mv(projected)
    assert float(jnp.max(jnp.abs(residual))) < 1e-13
    # The mixed mass integrates the source field exactly against target constants.
    np.testing.assert_allclose(
        float(jnp.sum(projection.mixed_mass.mv(values))),
        float(_integrals(source) @ np.asarray(values)),
        rtol=1e-13,
    )


def test_projection_conserves_the_integral_and_pullback_is_the_transpose(
    planar: Any,
) -> None:
    source, target, transfer = planar
    rng = np.random.default_rng(6)
    values = jnp.asarray(rng.normal(size=(source.dof_maps[0].global_dof_count, 2)))
    dual = jnp.asarray(rng.normal(size=(target.dof_maps[0].global_dof_count, 2)))
    projected = transfer.apply(values)
    assert projected.shape == (target.dof_maps[0].global_dof_count, 2)
    np.testing.assert_allclose(
        _integrals(target) @ np.asarray(projected),
        _integrals(source) @ np.asarray(values),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        float(jnp.vdot(projected, dual)),
        float(jnp.vdot(values, transfer.pullback(dual))),
        rtol=1e-12,
    )


def test_projection_reports_its_target_mass_solve_evidence(planar: Any) -> None:
    projection = planar[2].primal
    assert isinstance(projection, phx.discretization.FiniteElementL2Projection)
    prepared = projection.prepared_target
    assert isinstance(prepared, phx.discretization.PreparedL2ProjectionTarget)
    assert int(prepared.factorization.status) == 0
    assert float(prepared.factorization.diagnostics.minimum_pivot) > 0.0
    assert 1.0 <= float(prepared.mass_condition.value) < 100.0
    np.testing.assert_allclose(
        prepared.dof_measures, _integrals(planar[1]), rtol=1e-13, atol=1e-16
    )
    with pytest.raises(TypeError, match="prepare_l2_projection_target"):
        phx.discretization.PreparedL2ProjectionTarget(
            prepared.discretization,
            prepared.mass,
            prepared.factorization,
            prepared.mass_condition,
            prepared.dof_measures,
            field_name=prepared.field_name,
        )
    with pytest.raises(TypeError, match="prepare_l2_projection_transfer"):
        phx.discretization.FiniteElementL2Projection(
            projection.mixed_mass, prepared, operator_id="forged"
        )


def test_one_prepared_target_serves_every_source_and_payload_block() -> None:
    target_mesh = _triangles(4, perturb=0.3, seed=10)
    target = _space(target_mesh, "triangle")
    prepared = phx.discretization.prepare_l2_projection_target(target, field_name="u")
    linear = lambda x: 0.25 + x[:, 0] - 1.5 * x[:, 1]
    for n, degree, seed in ((6, 1, 11), (3, 2, 12)):
        source_mesh = _triangles(n, perturb=0.3, seed=seed)
        source = _space(source_mesh, "triangle", degree)
        transfer = phx.discretization.prepare_l2_projection_transfer(
            source, prepared, _refinement(source_mesh, target_mesh), field_name="u"
        )
        points = _dofs(source)
        block = jnp.stack(
            (linear(points), _smooth(points), jnp.ones(points.shape[0])), -1
        )
        projected = transfer.apply(block)
        np.testing.assert_allclose(projected[:, 0], linear(_dofs(target)), atol=1e-12)
        np.testing.assert_allclose(projected[:, 2], 1.0, atol=1e-12)
        np.testing.assert_allclose(
            projected[:, 1], transfer.apply(block[:, 1]), rtol=1e-13, atol=1e-15
        )
        np.testing.assert_allclose(
            _integrals(target) @ np.asarray(projected),
            _integrals(source) @ np.asarray(block),
            rtol=1e-12,
            atol=1e-14,
        )


def test_refreshed_target_matches_a_cold_preparation_of_the_moved_geometry() -> None:
    source_mesh = _triangles(6, perturb=0.3, seed=13)
    target_mesh = _triangles(4, perturb=0.3, seed=14)
    moved_mesh = _triangles(4, perturb=0.3, seed=15)
    source = _space(source_mesh, "triangle")
    target, moved = _space(target_mesh, "triangle"), _space(moved_mesh, "triangle")
    prepared = phx.discretization.prepare_l2_projection_target(target, field_name="u")
    refreshed = phx.discretization.refresh_l2_projection_target(prepared, moved)
    cold = phx.discretization.prepare_l2_projection_target(moved, field_name="u")
    assert refreshed.structure_id == prepared.structure_id == cold.structure_id
    assert refreshed.target_id == cold.target_id != prepared.target_id
    np.testing.assert_array_equal(
        refreshed.factorization.factor_values, cold.factorization.factor_values
    )
    refinement = _refinement(source_mesh, moved_mesh)
    transfer = phx.discretization.prepare_l2_projection_transfer(
        source, refreshed, refinement, field_name="u"
    )
    assert transfer.preserves_linear and transfer.conservative
    values = jnp.asarray(_smooth(_dofs(source)))
    np.testing.assert_array_equal(
        transfer.apply(values),
        phx.discretization.prepare_l2_projection_transfer(
            source, cold, refinement, field_name="u"
        ).apply(values),
    )
    linear = lambda x: 1.0 - x[:, 0] + 2.0 * x[:, 1]
    np.testing.assert_allclose(
        transfer.apply(jnp.asarray(linear(_dofs(source)))),
        linear(_dofs(moved)),
        atol=1e-12,
    )
    # The refreshed target no longer joins refinements of the old geometry.
    with pytest.raises(ValueError, match="does not join"):
        phx.discretization.prepare_l2_projection_transfer(
            source, refreshed, _refinement(source_mesh, target_mesh), field_name="u"
        )


def test_refresh_rejects_a_changed_target_structure() -> None:
    mesh = _triangles(3, perturb=0.2, seed=16)
    prepared = phx.discretization.prepare_l2_projection_target(
        _space(mesh, "triangle"), field_name="u"
    )
    refresh = phx.discretization.refresh_l2_projection_target
    with pytest.raises(ValueError, match="unchanged DOF structure"):
        refresh(prepared, _space(mesh, "triangle", 2))
    with pytest.raises(ValueError, match="unchanged DOF structure"):
        refresh(prepared, _space(_triangles(4, perturb=0.2, seed=16), "triangle"))
    with pytest.raises(TypeError, match="PreparedL2ProjectionTarget"):
        refresh(_space(mesh, "triangle"), _space(mesh, "triangle"))


def test_target_coverage_preserves_constants_without_claiming_conservation() -> None:
    source_mesh = _triangles(6)
    target_mesh = _triangles(3, lower=0.25, upper=0.75)
    source = _space(source_mesh, "triangle")
    target = _space(target_mesh, "triangle", discontinuous=True)
    transfer = _transfer(
        source,
        target,
        _refinement(source_mesh, target_mesh, coverage=CommonRefinementCoverage.TARGET),
    )
    assert transfer.preserves_constants and transfer.preserves_linear
    assert not transfer.conservative
    linear = lambda x: 2.0 * x[:, 0] + x[:, 1]
    np.testing.assert_allclose(
        transfer.apply(jnp.asarray(linear(_dofs(source)))),
        linear(_dofs(target)),
        atol=1e-12,
    )


def test_tetrahedral_projection_reproduces_linears_and_conserves() -> None:
    source_mesh, target_mesh = _tetrahedra(3), _tetrahedra(2)
    source = _space(source_mesh, "tetrahedron")
    target = _space(target_mesh, "tetrahedron")
    transfer = _transfer(source, target, _refinement(source_mesh, target_mesh))
    assert transfer.preserves_linear and transfer.conservative
    linear = lambda x: 1.0 + x[:, 0] - 2.0 * x[:, 1] + 0.5 * x[:, 2]
    np.testing.assert_allclose(
        transfer.apply(jnp.asarray(linear(_dofs(source)))),
        linear(_dofs(target)),
        atol=1e-12,
    )
    values = jnp.asarray(_smooth(_dofs(source)) * _dofs(source)[:, 2])
    np.testing.assert_allclose(
        _integrals(target) @ np.asarray(transfer.apply(values)),
        _integrals(source) @ np.asarray(values),
        rtol=1e-12,
    )


def test_invalid_refinements_and_elements_are_rejected() -> None:
    first, second, third = (
        _triangles(3),
        _triangles(2, perturb=0.2, seed=7),
        _triangles(4, perturb=0.2, seed=8),
    )
    source = _space(first, "triangle")
    target = phx.discretization.prepare_l2_projection_target(
        _space(second, "triangle"), field_name="u"
    )
    prepare = phx.discretization.prepare_l2_projection_transfer
    with pytest.raises(ValueError, match="does not join"):
        prepare(source, target, _refinement(first, third), field_name="u")
    with pytest.raises(ValueError, match="overlap_simplices"):
        prepare(
            source,
            target,
            prepare_common_refinement(first, second, policy=CommonRefinementPolicy()),
            field_name="u",
        )
    shrunk = _triangles(2, lower=0.1, upper=0.9)
    with pytest.raises(ValueError, match="successful common refinement"):
        prepare(
            source,
            phx.discretization.prepare_l2_projection_target(
                _space(shrunk, "triangle"), field_name="u"
            ),
            _refinement(first, shrunk),
            field_name="u",
        )
    with pytest.raises(TypeError, match="PreparedL2ProjectionTarget"):
        prepare(
            source, _space(second, "triangle"), _refinement(first, second), field_name="u"
        )
    vector = phx.discretization.FiniteElementPlan(
        first,
        phx.discretization.FiniteElementFieldSpec(
            "u", phx.discretization.raviart_thomas_element("triangle")
        ),
    ).prepare()
    with pytest.raises(ValueError, match="scalar Lagrange elements"):
        prepare(vector, target, _refinement(first, second), field_name="u")
    with pytest.raises(ValueError, match="scalar Lagrange elements"):
        phx.discretization.prepare_l2_projection_target(vector, field_name="u")
