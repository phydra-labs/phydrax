#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._meshcore import meshcore_available, MeshcoreUnavailableError
from phydrax.discretization import (
    prepare_unstructured_conservative_remap,
    UnstructuredFiniteVolumePlan,
    UnstructuredRemapLimiter,
    UnstructuredSecondOrderRemapPlan,
)
from phydrax.geometry import CommonRefinementPolicy, CommonRefinementStatus


requires_meshcore = pytest.mark.skipif(
    not meshcore_available(), reason="phydrax-meshcore unavailable"
)


def _quad(vertices, cells, *, ids=None):
    return UnstructuredFiniteVolumePlan(
        np.asarray(vertices, dtype=np.float64),
        quadrilaterals=np.asarray(cells, dtype=np.int32),
        cell_global_ids=None if ids is None else np.asarray(ids, dtype=np.int64),
    ).prepare()


def _unit_quad():
    return _quad(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)), ((0, 1, 2, 3),))


def _grid_points(n):
    x = np.linspace(0.0, 1.0, n + 1)
    points = np.stack(np.meshgrid(x, x, indexing="ij"), axis=-1).reshape(-1, 2)
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    corner = (i * (n + 1) + j).reshape(-1)
    return points, corner


def _quad_grid(n):
    points, a = _grid_points(n)
    return _quad(points, np.stack((a, a + n + 1, a + n + 2, a + 1), axis=-1))


def _perturbed_triangles(n, *, seed=0):
    points, a = _grid_points(n)
    interior = np.all((points > 0.0) & (points < 1.0), axis=1)
    rng = np.random.default_rng(seed)
    points[interior] += rng.uniform(-0.25, 0.25, (np.sum(interior), 2)) / n
    b, c, d = a + n + 1, a + n + 2, a + 1
    triangles = np.concatenate((np.stack((a, b, c), -1), np.stack((a, c, d), -1)))
    return UnstructuredFiniteVolumePlan(
        points, triangles=triangles.astype(np.int32)
    ).prepare()


def _cell_averages(geometry, function):
    integrals = jnp.sum(
        geometry.cell_quadrature_weights * function(geometry.cell_quadrature_points),
        axis=1,
    )
    return integrals / geometry.cell_volumes


def _second_order(remap, source, limiter=UnstructuredRemapLimiter.BARTH_JESPERSEN):
    return UnstructuredSecondOrderRemapPlan(
        remap.plan, remap.refinement, source, limiter=limiter
    )


@pytest.mark.meshcore
@requires_meshcore
def test_identical_meshes_remap_as_exact_jittable_identity():
    source = _quad_grid(4)
    remap = prepare_unstructured_conservative_remap(source, source, provenance="identity")
    assert remap.status is CommonRefinementStatus.SUCCESS
    assert remap.succeeded
    np.testing.assert_array_equal(
        remap.plan.target_offsets, np.arange(source.cell_count + 1)
    )
    np.testing.assert_array_equal(remap.plan.source_indices, np.arange(source.cell_count))
    np.testing.assert_allclose(
        remap.plan.intersection_measures, source.cell_volumes, rtol=1e-14
    )
    values = _cell_averages(source, lambda x: jnp.sin(3.0 * x[..., 0]) + x[..., 1] ** 2)
    np.testing.assert_allclose(
        eqx.filter_jit(remap.plan.apply)(values), values, rtol=1e-13
    )
    gradient = jax.grad(lambda value: jnp.sum(remap.plan.apply(value) ** 2))(values)
    np.testing.assert_allclose(gradient, 2.0 * values, rtol=1e-13)
    second = eqx.filter_jit(_second_order(remap, source).apply)(values)
    np.testing.assert_allclose(second.values, values, rtol=1e-13, atol=1e-14)
    assert int(second.limited_count) == 0


@pytest.mark.meshcore
@requires_meshcore
def test_containment_and_mixed_cells_cover_both_ledgers():
    source = _quad(
        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (2.0, 0.0), (2.0, 1.0)),
        ((0, 1, 2, 3), (1, 4, 5, 2)),
        ids=(20, 10),
    )
    target = _quad(
        ((0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)), ((0, 1, 2, 3),), ids=(99,)
    )
    remap = prepare_unstructured_conservative_remap(
        source, target, provenance="containment"
    )
    assert remap.succeeded
    np.testing.assert_allclose(remap.plan.intersection_measures, (1.0, 1.0))
    np.testing.assert_allclose(remap.evidence.source_coverage_defects, 0.0, atol=1e-14)
    np.testing.assert_allclose(remap.evidence.target_coverage_defects, 0.0, atol=1e-14)
    values = jnp.asarray([[1.0], [3.0]])
    mapped = remap.plan.apply(values)
    np.testing.assert_allclose(mapped, [[2.0]])
    np.testing.assert_allclose(
        remap.plan.conservation_defect(values, mapped), 0.0, atol=1e-14
    )

    mixed_source = UnstructuredFiniteVolumePlan(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (2.0, 0.0), (2.0, 1.0))),
        triangles=np.asarray(((0, 1, 2),), dtype=np.int32),
        quadrilaterals=np.asarray(((1, 3, 4, 2),), dtype=np.int32),
        cell_global_ids=np.asarray((31, 32), dtype=np.int64),
    ).prepare()
    mixed = prepare_unstructured_conservative_remap(
        mixed_source, target, provenance="mixed"
    )
    assert mixed.succeeded
    np.testing.assert_allclose(np.sum(mixed.plan.intersection_measures), 2.0)


@pytest.mark.meshcore
@requires_meshcore
def test_tetrahedron_identity_and_conservation():
    geometry = UnstructuredFiniteVolumePlan(
        np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
        tetrahedra=np.asarray(((0, 1, 2, 3),), dtype=np.int32),
        cell_global_ids=np.asarray((5,), dtype=np.int64),
    ).prepare()
    remap = prepare_unstructured_conservative_remap(
        geometry, geometry, provenance="tetra"
    )
    assert remap.succeeded
    np.testing.assert_allclose(remap.plan.intersection_measures, geometry.cell_volumes)
    values = jnp.asarray([[4.0]])
    np.testing.assert_allclose(remap.plan.apply(values), values)
    np.testing.assert_allclose(
        remap.plan.conservation_defect(values, remap.plan.apply(values)), 0.0
    )


@pytest.mark.meshcore
@requires_meshcore
def test_coincident_faces_give_exact_complete_coverage():
    points = ((0.0, 0.0), (0.5, 0.0), (1.0, 0.0), (0.0, 1.0), (0.5, 1.0), (1.0, 1.0))
    source = _quad(points, ((0, 1, 4, 3), (1, 2, 5, 4)))
    target = UnstructuredFiniteVolumePlan(
        np.asarray(points),
        triangles=np.asarray(((0, 1, 4), (0, 4, 3), (1, 2, 5), (1, 5, 4)), np.int32),
    ).prepare()
    remap = prepare_unstructured_conservative_remap(
        source, target, provenance="coincident"
    )
    assert remap.succeeded
    evidence = remap.evidence
    assert evidence.source_gap_count == evidence.target_gap_count == 0
    assert evidence.source_double_count == evidence.target_double_count == 0
    np.testing.assert_array_equal(remap.plan.target_offsets, (0, 1, 2, 3, 4))
    np.testing.assert_array_equal(remap.plan.source_indices, (0, 0, 1, 1))
    np.testing.assert_allclose(
        remap.plan.intersection_measures, target.cell_volumes, rtol=1e-15
    )
    np.testing.assert_allclose(
        remap.plan.apply(jnp.asarray((-2.0, 5.0))), (-2.0, -2.0, 5.0, 5.0), rtol=1e-15
    )


@pytest.mark.meshcore
@requires_meshcore
def test_constants_are_conserved_by_first_and_second_order_remap():
    source = _quad_grid(6)
    target = _perturbed_triangles(5)
    remap = prepare_unstructured_conservative_remap(source, target, provenance="constant")
    assert remap.succeeded
    constant = jnp.full((source.cell_count, 2), jnp.asarray((3.7, -1.25)))
    np.testing.assert_allclose(
        remap.plan.apply(constant),
        jnp.broadcast_to(constant[:1], (target.cell_count, 2)),
        rtol=1e-12,
    )
    for limiter in UnstructuredRemapLimiter:
        result = _second_order(remap, source, limiter).apply(constant)
        np.testing.assert_allclose(result.values[:, 0], 3.7, rtol=1e-12)
        np.testing.assert_allclose(result.values[:, 1], -1.25, rtol=1e-12)
        np.testing.assert_allclose(result.conservation_residual_after, 0.0, atol=1e-13)
        assert result.values.shape == (target.cell_count, 2)


@pytest.mark.meshcore
@requires_meshcore
def test_second_order_remap_reproduces_linear_fields_on_nonmatching_meshes():
    source = _quad_grid(6)
    target = _perturbed_triangles(5, seed=3)
    remap = prepare_unstructured_conservative_remap(source, target, provenance="linear")

    def linear(x):
        return 0.5 + 2.0 * x[..., 0] - 3.0 * x[..., 1]

    values = _cell_averages(source, linear)
    exact = _cell_averages(target, linear)
    second = _second_order(remap, source, UnstructuredRemapLimiter.NONE).apply(values)
    np.testing.assert_allclose(second.values, exact, rtol=1e-12, atol=1e-12)
    assert np.max(np.abs(remap.plan.apply(values) - exact)) > 1e-3


@pytest.mark.meshcore
@requires_meshcore
def test_limiter_keeps_discontinuous_fields_in_bounds_and_conserves():
    source = _quad_grid(8)
    target = _perturbed_triangles(7, seed=5)
    remap = prepare_unstructured_conservative_remap(source, target, provenance="step")
    centers = source.cell_centers
    step = jnp.where(centers[:, 0] + 0.3 * centers[:, 1] > 0.55, 2.0, -1.0)
    unlimited = _second_order(remap, source, UnstructuredRemapLimiter.NONE).apply(step)
    assert float(jnp.max(unlimited.values)) > 2.0 + 1e-6
    limited = _second_order(remap, source).apply(step)
    assert float(jnp.min(limited.values)) >= -1.0 - 1e-12
    assert float(jnp.max(limited.values)) <= 2.0 + 1e-12
    assert int(limited.limited_count) > 0
    assert 0.0 < float(limited.limited_fraction) < 1.0
    assert float(limited.minimum_limiter_factor) < 1.0
    assert bool(limited.restored)
    total = float(jnp.sum(source.cell_volumes * jnp.abs(step)))
    assert abs(float(limited.conservation_residual_after)) <= 1e-14 * total
    np.testing.assert_allclose(
        remap.plan.conservation_defect(step, limited.values), 0.0, atol=1e-14 * total
    )


@pytest.mark.meshcore
@requires_meshcore
def test_uncovered_and_overcovered_targets_fail_closed_without_plan():
    source = _unit_quad()
    larger_target = _quad(
        ((-0.5, -0.5), (1.5, -0.5), (1.5, 1.5), (-0.5, 1.5)), ((0, 1, 2, 3),)
    )
    under = prepare_unstructured_conservative_remap(
        source, larger_target, provenance="under"
    )
    assert under.status is CommonRefinementStatus.COVERAGE_GAP
    assert under.plan is None
    assert not under.succeeded
    assert under.evidence.target_gap_count == 1

    overlapping_source = _quad(
        (
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0),
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0),
        ),
        ((0, 1, 2, 3), (4, 5, 6, 7)),
    )
    over = prepare_unstructured_conservative_remap(
        overlapping_source, source, provenance="over"
    )
    assert over.status is CommonRefinementStatus.DOUBLE_COVERAGE
    assert over.plan is None


@pytest.mark.meshcore
@requires_meshcore
def test_first_order_remap_is_dual_to_the_reverse_remap():
    quads = _quad_grid(5)
    triangles = _perturbed_triangles(4, seed=7)
    forward = prepare_unstructured_conservative_remap(
        quads, triangles, provenance="forward"
    )
    reverse = prepare_unstructured_conservative_remap(
        triangles, quads, provenance="reverse"
    )
    rng = np.random.default_rng(11)
    u = jnp.asarray(rng.normal(size=quads.cell_count))
    w = jnp.asarray(rng.normal(size=triangles.cell_count))
    target_pairing = jnp.sum(triangles.cell_volumes * forward.plan.apply(u) * w)
    source_pairing = jnp.sum(quads.cell_volumes * u * reverse.plan.apply(w))
    np.testing.assert_allclose(target_pairing, source_pairing, rtol=1e-13)


@pytest.mark.meshcore
@requires_meshcore
def test_candidate_pair_limit_is_a_resource_refusal():
    geometry = _quad_grid(2)
    limited = prepare_unstructured_conservative_remap(
        geometry,
        geometry,
        provenance="limits",
        policy=CommonRefinementPolicy(maximum_candidate_pairs=1),
    )
    assert limited.status is CommonRefinementStatus.RESOURCE_LIMIT
    assert limited.plan is None
    assert limited.reason.startswith("RESOURCE_LIMIT")


@pytest.mark.meshcore
@requires_meshcore
def test_second_order_remap_differentiates_through_values():
    source = _quad_grid(6)
    target = _perturbed_triangles(5, seed=2)
    remap = prepare_unstructured_conservative_remap(source, target, provenance="grad")
    rng = np.random.default_rng(4)
    weights = jnp.asarray(rng.normal(size=target.cell_count))
    direction = jnp.asarray(rng.normal(size=source.cell_count))
    values = _cell_averages(source, lambda x: jnp.sin(4.0 * x[..., 0]) * x[..., 1])

    unlimited = _second_order(remap, source, UnstructuredRemapLimiter.NONE)
    gradient = jax.grad(lambda u: jnp.sum(weights * unlimited.apply(u).values))(values)
    # The unlimited remap is linear, so its gradient is the exact transpose action.
    np.testing.assert_allclose(
        jnp.sum(gradient * direction),
        jnp.sum(weights * unlimited.apply(direction).values),
        rtol=1e-12,
    )
    limited = _second_order(remap, source)
    limited_gradient = eqx.filter_jit(
        jax.grad(lambda u: jnp.sum(weights * limited.apply(u).values))
    )(values)
    assert bool(jnp.all(jnp.isfinite(limited_gradient)))


@pytest.mark.meshcore
@requires_meshcore
def test_second_order_remap_requires_full_rank_gradient_stencils():
    source = _unit_quad()
    remap = prepare_unstructured_conservative_remap(source, source, provenance="one")
    with pytest.raises(ValueError, match="rank deficient"):
        _second_order(remap, source)


def test_remap_requires_finite_volume_geometry_and_meshcore(monkeypatch, tmp_path):
    with pytest.raises(TypeError, match="unstructured FV"):
        prepare_unstructured_conservative_remap(object(), object(), provenance="kind")
    geometry = _unit_quad()
    monkeypatch.setenv("PHYDRAX_MESHCORE_LIBRARY", str(tmp_path / "missing-meshcore"))
    with pytest.raises(MeshcoreUnavailableError):
        prepare_unstructured_conservative_remap(geometry, geometry, provenance="missing")
