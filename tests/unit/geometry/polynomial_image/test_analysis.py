#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr

from phydrax.algebraic._system import SparsePolynomialSystem
from phydrax.geometry.polynomial_image import (
    EvidenceDisposition,
    plan_polynomial_image_analysis,
    PolynomialImageAnalysisPolicy,
    PolynomialImageAnalysisStatus,
    prepare_polynomial_image_analysis,
    SparsePolynomialMap,
    TargetMonomialSupport,
)


def _map(
    variable_labels,
    equation_labels,
    equation_indices,
    exponents,
    coefficients,
):
    system = SparsePolynomialSystem.from_coo(
        variable_labels,
        equation_labels,
        equation_indices,
        exponents,
        jnp.asarray(coefficients, dtype=float),
    )
    return SparsePolynomialMap(system)


def _discovery_policy(**overrides):
    values = {
        "rank_relative_tolerance": 1.0e-5,
        "rank_ambiguity_factor": 8.0,
        "heldout_absolute_tolerance": 2.0e-5,
        "heldout_relative_tolerance": 2.0e-5,
    }
    values.update(overrides)
    return PolynomialImageAnalysisPolicy(**values)


def _total_degree_support(labels, maximum_degree):
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
    return TargetMonomialSupport(labels, exponents)


def test_twisted_cubic_discovers_three_quadratic_relations_and_dimension_one():
    polynomial_map = _map(
        ("t",),
        ("x", "y", "z"),
        (0, 1, 2),
        ((1,), (2,), (3,)),
        (1.0, 1.0, 1.0),
    )
    support = _total_degree_support(("x", "y", "z"), 2)
    plan = plan_polynomial_image_analysis(
        polynomial_map,
        support,
        discovery_sample_count=32,
        heldout_sample_count=12,
        key=jr.PRNGKey(37),
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


def test_veronese_discovers_quadratic_relation_and_generic_dimension_two():
    polynomial_map = _map(
        ("s", "t"),
        ("x", "y", "z"),
        (0, 1, 2),
        ((2, 0), (1, 1), (0, 2)),
        (1.0, 1.0, 1.0),
    )
    support = TargetMonomialSupport(
        ("x", "y", "z"),
        (
            (2, 0, 0),
            (1, 1, 0),
            (1, 0, 1),
            (0, 2, 0),
            (0, 1, 1),
            (0, 0, 2),
        ),
    )
    plan = plan_polynomial_image_analysis(
        polynomial_map,
        support,
        discovery_sample_count=32,
        heldout_sample_count=12,
        key=jr.PRNGKey(91),
        policy=_discovery_policy(),
    )

    result = plan.prepare().result

    assert result.status is PolynomialImageAnalysisStatus.DISCOVERED
    assert result.jacobian_rank.dimension_lower_bound == 2
    assert result.jacobian_rank.dimension_upper_bound == 2
    assert result.relations.relation_count == 1


def test_rank_deficiency_is_resolved_but_near_cutoff_rank_is_ambiguous():
    deficient = _map(
        ("u", "v"),
        ("x", "y"),
        (0, 1),
        ((1, 0), (1, 0)),
        (1.0, 1.0),
    )
    support = TargetMonomialSupport(("x", "y"), ((0, 0), (1, 0), (0, 1)))
    resolved = (
        plan_polynomial_image_analysis(
            deficient,
            support,
            discovery_sample_count=12,
            heldout_sample_count=4,
            key=jr.PRNGKey(4),
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
        plan_polynomial_image_analysis(
            nearly_deficient,
            support,
            discovery_sample_count=12,
            heldout_sample_count=4,
            key=jr.PRNGKey(4),
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


def test_insufficient_samples_and_resource_limits_return_typed_results():
    polynomial_map = _map(
        ("t",),
        ("x",),
        (0,),
        ((1,),),
        (1.0,),
    )
    support = TargetMonomialSupport(("x",), ((0,), (1,), (2,)))
    insufficient = (
        plan_polynomial_image_analysis(
            polynomial_map,
            support,
            discovery_sample_count=2,
            heldout_sample_count=1,
            key=jr.PRNGKey(0),
            policy=_discovery_policy(),
        )
        .prepare()
        .result
    )
    limited = (
        plan_polynomial_image_analysis(
            polynomial_map,
            support,
            discovery_sample_count=3,
            heldout_sample_count=1,
            policy=_discovery_policy(maximum_design_entries=4),
            key=jr.PRNGKey(0),
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


def test_heldout_samples_reject_relation_created_by_nonrepresentative_discovery_points():
    polynomial_map = _map(
        ("t",),
        ("y",),
        (0,),
        ((2,),),
        (1.0,),
    )
    support = TargetMonomialSupport(("y",), ((0,), (1,)))
    plan = plan_polynomial_image_analysis(
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


def test_affine_sampling_is_replayable_and_refresh_preserves_samples():
    polynomial_map = _map(
        ("s", "t"),
        ("x", "y"),
        (0, 1),
        ((1, 0), (0, 1)),
        (1.0, 1.0),
    )
    support = TargetMonomialSupport(("x", "y"), ((0, 0), (1, 0), (0, 1)))
    plan = plan_polynomial_image_analysis(
        polynomial_map,
        support,
        discovery_sample_count=8,
        heldout_sample_count=3,
        key=jr.PRNGKey(2026),
        policy=_discovery_policy(),
    )

    first = plan.prepare()
    replay = plan.prepare()
    changed_key = plan_polynomial_image_analysis(
        polynomial_map,
        support,
        discovery_sample_count=8,
        heldout_sample_count=3,
        key=jr.PRNGKey(2027),
        policy=_discovery_policy(),
    ).prepare()
    refreshed = first.refresh(polynomial_map)

    assert jnp.array_equal(first.source_points, replay.source_points)
    assert jnp.array_equal(first.heldout_source_points, replay.heldout_source_points)
    assert first.prepared_id == replay.prepared_id
    assert not jnp.array_equal(first.source_points, changed_key.source_points)
    assert jnp.array_equal(first.source_points, refreshed.source_points)
    assert int(refreshed.numeric_version) == int(first.numeric_version) + 1
