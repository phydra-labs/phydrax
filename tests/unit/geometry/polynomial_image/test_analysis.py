#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from collections.abc import Sequence

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from jax.typing import ArrayLike

from phydrax.algebraic._system import SparsePolynomialSystem
from phydrax.geometry.polynomial_image import (
    EvidenceDisposition,
    PolynomialImageAnalysisPlan,
    PolynomialImageAnalysisPolicy,
    PolynomialImageAnalysisStatus,
    prepare_polynomial_image_analysis,
    SparsePolynomialMap,
    TargetMonomialSupport,
)


def _map(
    variable_labels: Sequence[str],
    equation_labels: Sequence[str],
    equation_indices: Sequence[int],
    exponents: Sequence[tuple[int, ...]],
    coefficients: ArrayLike | Sequence[float],
) -> SparsePolynomialMap:
    system = SparsePolynomialSystem.from_coo(
        variable_labels,
        equation_labels,
        equation_indices,
        exponents,
        jnp.asarray(coefficients, dtype=jnp.float64),
    )
    return SparsePolynomialMap(system)


def _discovery_policy(
    *,
    rank_relative_tolerance: float = 1.0e-5,
    heldout_absolute_tolerance: float = 2.0e-5,
    heldout_relative_tolerance: float = 2.0e-5,
    maximum_design_entries: int = 20_000_000,
) -> PolynomialImageAnalysisPolicy:
    # Empirical discovery/heldout thresholds retained from these fixed polynomial
    # fixtures at d2dfcb941; they label numerical discovery, not exact containment.
    return PolynomialImageAnalysisPolicy(
        rank_relative_tolerance=rank_relative_tolerance,
        rank_ambiguity_factor=8.0,
        heldout_absolute_tolerance=heldout_absolute_tolerance,
        heldout_relative_tolerance=heldout_relative_tolerance,
        maximum_design_entries=maximum_design_entries,
    )


def _total_degree_support(
    labels: tuple[str, ...], maximum_degree: int
) -> TargetMonomialSupport:
    if len(labels) == 2:
        exponents = [
            (first, second)
            for first in range(maximum_degree + 1)
            for second in range(maximum_degree + 1 - first)
        ]
    elif len(labels) == 3:
        exponents = [
            (first, second, third)
            for first in range(maximum_degree + 1)
            for second in range(maximum_degree + 1 - first)
            for third in range(maximum_degree + 1 - first - second)
        ]
    else:
        raise ValueError("Test support helper only covers two or three variables.")
    return TargetMonomialSupport(labels, np.asarray(exponents, dtype=np.int64))


def test_cubic_curve_discovers_replayable_relations_and_rank() -> None:
    polynomial_map = _map(
        ("t",),
        ("x", "y", "z"),
        (0, 1, 2),
        ((1,), (2,), (3,)),
        (1.0, 1.0, 1.0),
    )
    support = _total_degree_support(("x", "y", "z"), 2)
    plan = PolynomialImageAnalysisPlan(
        polynomial_map,
        support,
        discovery_sample_count=32,
        heldout_sample_count=12,
        key=jr.key(37, impl="threefry2x32"),
        policy=_discovery_policy(),
    )

    prepared = prepare_polynomial_image_analysis(plan)
    result = prepared.result
    replay = prepare_polynomial_image_analysis(plan).result

    assert result.status is PolynomialImageAnalysisStatus.DISCOVERED
    assert result.jacobian_rank.resolved
    assert result.jacobian_rank.dimension_lower_bound == 1
    assert result.jacobian_rank.dimension_upper_bound == 1
    assert result.relations.relation_count == 3
    assert jnp.array_equal(
        result.relations.candidate_coefficients,
        replay.relations.candidate_coefficients,
    )
    active = result.relations.candidate_coefficients[: result.relations.relation_count]
    assert jnp.allclose(jnp.max(jnp.abs(active), axis=1), 1.0)
    assert result.relations.validation_accepted
    assert result.claims.numerical_discovery is EvidenceDisposition.SUPPORTED
    assert result.claims.exact_containment is EvidenceDisposition.NOT_ASSESSED
    assert result.claims.ideal_equality is EvidenceDisposition.NOT_ASSESSED
    assert result.claims.real_geometry is EvidenceDisposition.NOT_ASSESSED
    assert result.claims.topology is EvidenceDisposition.NOT_ASSESSED


def test_quadratic_surface_discovers_one_relation_and_rank_two() -> None:
    polynomial_map = _map(
        ("s", "t"),
        ("x", "y", "z"),
        (0, 1, 2),
        ((2, 0), (1, 1), (0, 2)),
        (1.0, 1.0, 1.0),
    )
    support = TargetMonomialSupport(
        ("x", "y", "z"),
        # ty: ignore[invalid-argument-type]
        (
            (2, 0, 0),
            (1, 1, 0),
            (1, 0, 1),
            (0, 2, 0),
            (0, 1, 1),
            (0, 0, 2),
        ),
    )
    plan = PolynomialImageAnalysisPlan(
        polynomial_map,
        support,
        discovery_sample_count=32,
        heldout_sample_count=12,
        key=jr.key(91, impl="threefry2x32"),
        policy=_discovery_policy(),
    )

    result = plan.prepare().result

    assert result.status is PolynomialImageAnalysisStatus.DISCOVERED
    assert result.jacobian_rank.dimension_lower_bound == 2
    assert result.jacobian_rank.dimension_upper_bound == 2
    assert result.relations.relation_count == 1


def test_jacobian_rank_separates_resolved_and_ambiguous_singular_values() -> None:
    deficient = _map(
        ("u", "v"),
        ("x", "y"),
        (0, 1),
        ((1, 0), (1, 0)),
        (1.0, 1.0),
    )
    # ty: ignore[invalid-argument-type]
    support = TargetMonomialSupport(("x", "y"), ((0, 0), (1, 0), (0, 1)))
    resolved = (
        PolynomialImageAnalysisPlan(
            deficient,
            support,
            discovery_sample_count=12,
            heldout_sample_count=4,
            key=jr.key(4, impl="threefry2x32"),
            policy=_discovery_policy(),
        )
        .prepare()
        .result
    )

    nearly_deficient = _map(
        ("u", "v"),
        ("x", "y"),
        (0, 1),
        ((1, 0), (0, 1)),
        (1.0, 1.0e-4),
    )
    ambiguous = (
        PolynomialImageAnalysisPlan(
            nearly_deficient,
            support,
            discovery_sample_count=12,
            heldout_sample_count=4,
            key=jr.key(4, impl="threefry2x32"),
            policy=_discovery_policy(rank_relative_tolerance=1.0e-4),
        )
        .prepare()
        .result
    )

    assert resolved.jacobian_rank.resolved
    assert resolved.jacobian_rank.dimension_lower_bound == 1
    assert resolved.jacobian_rank.dimension_upper_bound == 1
    assert not ambiguous.jacobian_rank.resolved
    assert ambiguous.jacobian_rank.dimension_lower_bound == 1
    assert ambiguous.jacobian_rank.dimension_upper_bound == 2
    assert ambiguous.status is PolynomialImageAnalysisStatus.JACOBIAN_RANK_AMBIGUOUS


def test_sample_and_design_admission_preserve_refusal_categories() -> None:
    polynomial_map = _map(
        ("t",),
        ("x",),
        (0,),
        ((1,),),
        (1.0,),
    )
    # ty: ignore[invalid-argument-type]
    support = TargetMonomialSupport(("x",), ((0,), (1,), (2,)))
    insufficient = (
        PolynomialImageAnalysisPlan(
            polynomial_map,
            support,
            discovery_sample_count=2,
            heldout_sample_count=1,
            key=jr.key(0, impl="threefry2x32"),
            policy=_discovery_policy(),
        )
        .prepare()
        .result
    )
    limited = (
        PolynomialImageAnalysisPlan(
            polynomial_map,
            support,
            discovery_sample_count=3,
            heldout_sample_count=1,
            policy=_discovery_policy(maximum_design_entries=4),
            key=jr.key(0, impl="threefry2x32"),
        )
        .prepare()
        .result
    )

    assert (
        insufficient.status
        is PolynomialImageAnalysisStatus.INSUFFICIENT_DISCOVERY_SAMPLES
    )
    assert limited.status is PolynomialImageAnalysisStatus.RESOURCE_LIMIT
    assert not limited.resources.within_budget
    assert limited.resources.limiting_resource == "maximum_design_entries"


def test_independent_heldout_samples_reject_a_spurious_relation() -> None:
    polynomial_map = _map(
        ("t",),
        ("y",),
        (0,),
        ((2,),),
        (1.0,),
    )
    # ty: ignore[invalid-argument-type]
    support = TargetMonomialSupport(("y",), ((0,), (1,)))
    plan = PolynomialImageAnalysisPlan(
        polynomial_map,
        support,
        source_samples=jnp.asarray([[-1.0], [1.0]]),
        heldout_source_samples=jnp.asarray([[2.0]]),
        policy=_discovery_policy(
            heldout_absolute_tolerance=1.0e-8,
            heldout_relative_tolerance=1.0e-8,
        ),
    )

    result = plan.prepare().result

    assert result.status is PolynomialImageAnalysisStatus.HELDOUT_REJECTED
    assert result.relations.relation_count == 1
    assert not result.relations.validation_accepted
    assert result.claims.numerical_discovery is EvidenceDisposition.REJECTED


def test_seeded_samples_replay_and_refresh_preserve_prepared_identity() -> None:
    polynomial_map = _map(
        ("s", "t"),
        ("x", "y"),
        (0, 1),
        ((1, 0), (0, 1)),
        (1.0, 1.0),
    )
    # ty: ignore[invalid-argument-type]
    support = TargetMonomialSupport(("x", "y"), ((0, 0), (1, 0), (0, 1)))
    plan = PolynomialImageAnalysisPlan(
        polynomial_map,
        support,
        discovery_sample_count=8,
        heldout_sample_count=3,
        key=jr.key(2026, impl="threefry2x32"),
        policy=_discovery_policy(),
    )

    first = plan.prepare()
    replay = plan.prepare()
    changed_key = PolynomialImageAnalysisPlan(
        polynomial_map,
        support,
        discovery_sample_count=8,
        heldout_sample_count=3,
        key=jr.key(2027, impl="threefry2x32"),
        policy=_discovery_policy(),
    ).prepare()
    refreshed = first.refresh(polynomial_map)

    assert jnp.array_equal(first.source_points, replay.source_points)
    assert jnp.array_equal(first.heldout_source_points, replay.heldout_source_points)
    assert first.prepared_id == replay.prepared_id
    assert not jnp.array_equal(first.source_points, changed_key.source_points)
    assert jnp.array_equal(first.source_points, refreshed.source_points)
    assert int(refreshed.numeric_version) == int(first.numeric_version) + 1


def test_sampling_refuses_legacy_unsigned_key_arrays() -> None:
    mapping = _map(("t",), ("x",), (0,), ((1,),), (1.0,))
    support = TargetMonomialSupport(("x",), np.asarray(((0,), (1,)), dtype=np.int64))
    with pytest.raises(TypeError, match="key"):
        PolynomialImageAnalysisPlan(
            mapping, support, key=jnp.asarray((0, 37), dtype=jnp.uint32)
        )


def test_sampling_refuses_a_different_key_algorithm() -> None:
    mapping = _map(("t",), ("x",), (0,), ((1,),), (1.0,))
    support = TargetMonomialSupport(("x",), np.asarray(((0,), (1,)), dtype=np.int64))
    with pytest.raises(ValueError, match="PolynomialImageAnalysisPlan.key"):
        PolynomialImageAnalysisPlan(mapping, support, key=jr.key(37, impl="rbg"))
