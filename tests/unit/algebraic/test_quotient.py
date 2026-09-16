#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

from phydrax.algebraic._quotient import (
    plan_quotient_roots,
    prepare_quotient_root_solver,
    QuotientRankPolicy,
    QuotientRootPolicy,
    QuotientRootResourcePolicy,
    QuotientRootStatus,
    refresh_quotient_root_solver,
    solve_quotient_roots,
)
from phydrax.algebraic._system import SparsePolynomialSystem


def _univariate_quadratic(constant: float) -> SparsePolynomialSystem:
    return SparsePolynomialSystem.from_coo(
        ("x",),
        ("f",),
        (0, 0),
        ((2,), (0,)),
        jnp.asarray((1.0, constant)),
    )


def test_x_squared_minus_one_recovers_both_roots_and_replays_original_system():
    system = _univariate_quadratic(-1.0)

    result = solve_quotient_roots(system, polish=True)

    assert int(result.status) == int(QuotientRootStatus.SUCCESS)
    assert bool(result.successful)
    np.testing.assert_allclose(
        np.asarray(result.roots[:, 0]),
        np.asarray((-1.0, 1.0)),
        atol=2e-6,
    )
    replayed = result.replay(system)
    np.testing.assert_allclose(replayed, result.original_residuals)
    np.testing.assert_allclose(replayed, 0.0, atol=2e-6)
    assert bool(jnp.all(result.polished))


def test_small_bivariate_finite_system_has_jointly_consistent_roots():
    system = SparsePolynomialSystem.from_coo(
        ("x", "y"),
        ("circle", "diagonal"),
        (0, 0, 1, 1),
        ((2, 0), (0, 0), (0, 1), (1, 0)),
        jnp.asarray((1.0, -1.0, 1.0, -1.0)),
    )

    result = solve_quotient_roots(system)

    assert int(result.status) == int(QuotientRootStatus.SUCCESS)
    np.testing.assert_allclose(
        np.asarray(result.roots),
        np.asarray(((-1.0, -1.0), (1.0, 1.0))),
        atol=3e-6,
    )
    np.testing.assert_allclose(result.original_residuals, 0.0, atol=3e-6)
    assert result.evidence.cluster_count == 2


def test_repeated_root_is_returned_as_an_explicit_ambiguity_not_a_success():
    system = _univariate_quadratic(0.0)

    result = solve_quotient_roots(system)

    assert int(result.status) == int(QuotientRootStatus.REPEATED_ROOT_AMBIGUITY)
    assert not bool(result.successful)
    np.testing.assert_allclose(result.roots, 0.0, atol=1e-7)
    np.testing.assert_array_equal(result.cluster_multiplicities, (2, 2))
    np.testing.assert_allclose(result.replay(system), 0.0, atol=1e-7)


def test_rank_policy_ambiguity_stops_before_quotient_basis_selection():
    system = _univariate_quadratic(-1.0)
    policy = QuotientRootPolicy(
        rank=QuotientRankPolicy(
            relative_tolerance=0.75,
            ambiguity_factor=2.0,
        )
    )

    prepared = prepare_quotient_root_solver(system, policy)
    result = solve_quotient_roots(prepared)

    assert prepared.status == int(QuotientRootStatus.RANK_AMBIGUOUS)
    assert int(prepared.preparation.definite_rank) == 0
    assert int(prepared.preparation.possible_rank) == 1
    assert int(result.status) == int(QuotientRootStatus.RANK_AMBIGUOUS)
    assert result.root_count == 0


def test_resource_policy_rejects_before_monomial_allocation():
    system = _univariate_quadratic(-1.0)
    policy = QuotientRootPolicy(resources=QuotientRootResourcePolicy(maximum_monomials=2))

    plan = plan_quotient_roots(system.support, policy)
    result = solve_quotient_roots(system, plan=plan)

    assert not plan.accepted
    assert plan.rejection_reasons == ("monomial-count",)
    assert plan.monomials.shape == (0, 1)
    assert int(result.status) == int(QuotientRootStatus.RESOURCE_REJECTED)
    assert result.root_count == 0


def test_coefficient_refresh_preserves_support_plan_and_changes_root_replay():
    first = _univariate_quadratic(-1.0)
    second = first.with_coefficients(jnp.asarray((-4.0, 1.0)))
    plan = plan_quotient_roots(first.support)
    prepared = prepare_quotient_root_solver(first, plan)

    refreshed = refresh_quotient_root_solver(prepared, second)
    result = solve_quotient_roots(refreshed)

    assert refreshed.prepared_id == prepared.prepared_id
    assert int(refreshed.numeric_version) == 1
    assert int(refreshed.refresh_count) == 1
    assert int(result.status) == int(QuotientRootStatus.SUCCESS)
    np.testing.assert_allclose(result.roots[:, 0], (-2.0, 2.0), atol=3e-6)
    np.testing.assert_allclose(result.replay(second), result.original_residuals)
    np.testing.assert_allclose(result.original_residuals, 0.0, atol=3e-6)
