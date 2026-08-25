#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from phydrax.discretization.finite_volume._automatic_remap import (
    build_unstructured_conservative_remap,
    UnstructuredConservativeRemapStatus,
)
from phydrax.discretization.finite_volume._unstructured import (
    UnstructuredFiniteVolumePlan,
)


def _quad(vertices, cells, *, ids=None):
    return UnstructuredFiniteVolumePlan(
        np.asarray(vertices, dtype=float),
        quadrilaterals=np.asarray(cells, dtype=np.int32),
        cell_global_ids=None if ids is None else np.asarray(ids, dtype=np.int64),
    ).prepare()


def _unit_quad(*, offset=(0.0, 0.0), ids=None):
    x, y = offset
    return _quad(
        ((x, y), (x + 1.0, y), (x + 1.0, y + 1.0), (x, y + 1.0)),
        ((0, 1, 2, 3),),
        ids=ids,
    )


def test_identity_and_translated_geometry_are_exact_and_jittable():
    source = _unit_quad(ids=(11,))
    target = _unit_quad(ids=(22,))
    result = build_unstructured_conservative_remap(
        source, target, tolerance=1e-10, provenance="identity"
    )
    assert result.status is UnstructuredConservativeRemapStatus.SUCCESS
    assert result.plan is not None
    np.testing.assert_array_equal(result.target_offsets, (0, 1))
    np.testing.assert_allclose(result.intersection_measures, (1.0,))

    translated = _unit_quad(offset=(3.0, -2.0), ids=(33,))
    translated_result = build_unstructured_conservative_remap(
        translated, translated, tolerance=1e-10, provenance="translation"
    )
    assert translated_result.status is UnstructuredConservativeRemapStatus.SUCCESS
    values = jnp.asarray([[2.0, -1.0]])
    transferred = eqx.filter_jit(translated_result.plan.apply)(values)
    np.testing.assert_allclose(transferred, values)
    gradient = jax.grad(lambda value: jnp.sum(translated_result.plan.apply(value) ** 2))(
        values
    )
    np.testing.assert_allclose(gradient, 2.0 * values)


def test_containment_and_mixed_triangle_quad_refinement_cover_both_ledgers():
    source = _quad(
        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (2.0, 0.0), (2.0, 1.0)),
        ((0, 1, 2, 3), (1, 4, 5, 2)),
        ids=(20, 10),
    )
    target = _unit_quad(offset=(0.0, 0.0), ids=(99,))
    # The target is intentionally made the full 2-by-1 containment cell.
    target = _quad(
        ((0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)), ((0, 1, 2, 3),), ids=(99,)
    )
    result = build_unstructured_conservative_remap(
        source, target, tolerance=1e-10, provenance="containment"
    )
    assert result.status is UnstructuredConservativeRemapStatus.SUCCESS
    np.testing.assert_allclose(result.intersection_measures, (1.0, 1.0))
    np.testing.assert_allclose(result.source_coverage_defects, 0.0, atol=1e-12)
    np.testing.assert_allclose(result.target_coverage_defects, 0.0, atol=1e-12)
    assert result.pair_ids == ((99, 10), (99, 20))

    mixed_source = UnstructuredFiniteVolumePlan(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (2.0, 0.0), (2.0, 1.0))),
        triangles=np.asarray(((0, 1, 2),)),
        quadrilaterals=np.asarray(((1, 3, 4, 2),)),
        cell_global_ids=np.asarray((31, 32)),
    ).prepare()
    mixed_result = build_unstructured_conservative_remap(
        mixed_source, target, tolerance=1e-10, provenance="mixed"
    )
    assert mixed_result.status is UnstructuredConservativeRemapStatus.SUCCESS
    np.testing.assert_allclose(np.sum(mixed_result.intersection_measures), 2.0)


def test_tetrahedron_identity_and_conservation():
    plan = UnstructuredFiniteVolumePlan(
        np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
        tetrahedra=np.asarray(((0, 1, 2, 3),)),
        cell_global_ids=np.asarray((5,)),
    ).prepare()
    result = build_unstructured_conservative_remap(
        plan, plan, tolerance=1e-10, provenance="tetra"
    )
    assert result.status is UnstructuredConservativeRemapStatus.SUCCESS
    np.testing.assert_allclose(result.intersection_measures, plan.cell_volumes)
    values = jnp.asarray([[4.0]])
    np.testing.assert_allclose(result.plan.apply(values), values)
    np.testing.assert_allclose(
        result.plan.conservation_defect(values, result.plan.apply(values)), 0.0
    )


def test_stable_ids_and_complete_source_target_conservation():
    source = _quad(
        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (2.0, 0.0), (2.0, 1.0)),
        ((0, 1, 2, 3), (1, 4, 5, 2)),
        ids=(20, 10),
    )
    target = _quad(
        ((0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)), ((0, 1, 2, 3),), ids=(99,)
    )
    result = build_unstructured_conservative_remap(
        source,
        target,
        tolerance=1e-10,
        provenance="stable-ids",
        source_global_ids=np.asarray((20, 10)),
        target_global_ids=np.asarray((99,)),
    )
    assert result.status is UnstructuredConservativeRemapStatus.SUCCESS
    assert result.candidate_count == 2
    assert result.accepted_count == 2
    assert result.pair_ids == ((99, 10), (99, 20))
    values = jnp.asarray([[1.0], [3.0]])
    mapped = result.plan.apply(values)
    np.testing.assert_allclose(mapped, [[2.0]])
    np.testing.assert_allclose(
        result.plan.conservation_defect(values, mapped), 0.0, atol=1e-14
    )


def test_under_and_over_coverage_fail_closed_without_repair():
    source = _unit_quad()
    larger_target = _quad(
        ((-0.5, -0.5), (1.5, -0.5), (1.5, 1.5), (-0.5, 1.5)), ((0, 1, 2, 3),)
    )
    under = build_unstructured_conservative_remap(
        source, larger_target, tolerance=1e-10, provenance="under"
    )
    assert under.status is UnstructuredConservativeRemapStatus.COVERAGE_FAILURE
    assert under.plan is None

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
    over = build_unstructured_conservative_remap(
        overlapping_source, source, tolerance=1e-10, provenance="over"
    )
    assert over.status is UnstructuredConservativeRemapStatus.COVERAGE_FAILURE
    assert over.plan is None


def test_predicate_uncertainty_and_resource_limits_are_typed_failures():
    source = _unit_quad()
    shifted = _quad(
        ((1.0 - 1e-12, 0.0), (2.0 - 1e-12, 0.0), (2.0 - 1e-12, 1.0), (1.0 - 1e-12, 1.0)),
        ((0, 1, 2, 3),),
    )
    uncertain = build_unstructured_conservative_remap(
        source, shifted, tolerance=1e-8, provenance="predicate"
    )
    assert uncertain.status is UnstructuredConservativeRemapStatus.PREDICATE_UNCERTAIN
    assert uncertain.evidence.predicate_uncertain_count >= 1

    tiled_source = _quad(
        (
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0),
            (1.0, 0.0),
            (2.0, 0.0),
            (2.0, 1.0),
            (1.0, 1.0),
        ),
        ((0, 1, 2, 3), (4, 5, 6, 7)),
    )
    limited = build_unstructured_conservative_remap(
        tiled_source,
        tiled_source,
        tolerance=1e-10,
        limits={"max_candidate_pairs": 1},
        provenance="limits",
    )
    assert limited.status is UnstructuredConservativeRemapStatus.RESOURCE_LIMIT
    assert limited.plan is None


def test_explicit_unsupported_geometry_returns_failure_artifact():
    result = build_unstructured_conservative_remap(
        object(), object(), tolerance=1e-10, provenance="unsupported"
    )
    assert result.status is UnstructuredConservativeRemapStatus.UNSUPPORTED_GEOMETRY
    assert result.plan is None
    assert not result.passed
