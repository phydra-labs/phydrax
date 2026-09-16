#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.algebraic import SparsePolynomialSystem
from phydrax.optim._programming._cones import ProductCone, ZeroCone
from phydrax.optim._programming._psd_cone import PositiveSemidefiniteCone
from phydrax.optim._programming._quadratic import ConvexProgramResult
from phydrax.optim._programming._types import (
    ConvexProgramCertificate,
    ConvexProgramProvenance,
    ConvexProgramStatus,
)
from phydrax.optim.polynomial import (
    AtomExtractionStatus,
    audit_polynomial_candidate,
    audit_polynomial_relaxation,
    plan_polynomial_relaxation,
    PolynomialOptimizationProblem,
    PolynomialRelaxationResources,
    PolynomialRelaxationStatus,
    prepare_polynomial_relaxation,
    refresh_polynomial_relaxation,
)


def _system(variable_labels, equation_labels, equation_indices, exponents, coefficients):
    return SparsePolynomialSystem.from_coo(
        variable_labels,
        equation_labels,
        equation_indices,
        exponents,
        coefficients,
    )


def _provider_result(prepared, moments, *, accepted=True, cone_dual=None):
    program = prepared.program
    primal = jnp.asarray(moments, dtype=program.linear.dtype)
    slack = program.constraint_rhs - program.constraint_matrix @ primal
    dual = (
        jnp.zeros_like(slack)
        if cone_dual is None
        else jnp.asarray(cone_dual, dtype=primal.dtype)
    )
    empty = jnp.empty((0,), dtype=primal.dtype)
    zero = jnp.asarray(0.0, dtype=primal.dtype)
    certificate = ConvexProgramCertificate(
        primal_ray=jnp.zeros_like(primal),
        equality_dual_ray=empty,
        inequality_dual_ray=empty,
        lower_bound_dual_ray=jnp.zeros_like(primal),
        upper_bound_dual_ray=jnp.zeros_like(primal),
        primal_ray_residual_norm=zero,
        dual_ray_residual_norm=zero,
        primal_ray_objective=zero,
        dual_ray_objective=zero,
        primal_ray_valid=False,
        dual_ray_valid=False,
    )
    provenance = ConvexProgramProvenance(
        numeric_version=0,
        problem_id=program.problem_id,
        structure_id=program.structure_id,
        policy_id="polynomial-test-policy",
        method_id="polynomial-test-provider",
        backend="test",
        backend_version="test",
        convexity_evidence=program.convexity_evidence,
        regularization=0.0,
        numeric_binding_id="polynomial-test-binding",
    )
    return ConvexProgramResult(
        primal=primal,
        equality_dual=empty,
        inequality_dual=empty,
        inequality_slack=empty,
        cone_slack=slack,
        cone_dual=dual,
        cone_primal_residual=jnp.zeros_like(slack),
        cone_violation=jnp.zeros_like(slack),
        cone_dual_violation=jnp.zeros_like(slack),
        cone_complementarity=empty,
        lower_bound_dual=jnp.zeros_like(primal),
        upper_bound_dual=jnp.zeros_like(primal),
        objective=jnp.sum(program.linear * primal),
        stationarity_residual=jnp.zeros_like(primal),
        solver_stationarity_residual=jnp.zeros_like(primal),
        equality_residual=empty,
        inequality_residual=empty,
        inequality_violation=empty,
        complementarity_residual=empty,
        primal_residual_norm=zero,
        dual_residual_norm=zero,
        solver_dual_residual_norm=zero,
        complementarity_gap=zero,
        kkt_residual_norm=zero,
        iterations=jnp.asarray(0, dtype=jnp.int32),
        backend_converged=jnp.asarray(accepted),
        valid=jnp.asarray(accepted),
        status=jnp.asarray(
            int(
                ConvexProgramStatus.OPTIMAL
                if accepted
                else ConvexProgramStatus.NUMERICAL_FAILURE
            ),
            dtype=jnp.int32,
        ),
        certificate=certificate,
        provenance=provenance,
        batch_shape=(),
        method="polynomial-test-provider",
        backend="test",
        regularization=0.0,
        tolerance=1e-8,
        max_iterations=0,
    )


def _dual_certificate(
    prepared,
    lower_bound,
    *,
    equality_blocks=(),
    moment_matrix=None,
    localizing_matrices=(),
):
    dual = jnp.zeros(
        (prepared.program.num_constraints,), dtype=prepared.program.linear.dtype
    )
    dual = dual.at[0].set(-lower_bound)
    for row_slice, values in zip(prepared.template.equality_row_slices, equality_blocks):
        dual = dual.at[row_slice].set(jnp.asarray(values, dtype=dual.dtype))
    if moment_matrix is not None:
        cone = PositiveSemidefiniteCone(prepared.template.moment_basis.matrix_size)
        dual = dual.at[prepared.template.moment_psd_slice].set(cone.pack(moment_matrix))
    for basis, row_slice, matrix in zip(
        prepared.template.localizing_bases,
        prepared.template.localizing_psd_slices,
        localizing_matrices,
    ):
        dual = dual.at[row_slice].set(
            PositiveSemidefiniteCone(basis.matrix_size).pack(matrix)
        )
    return dual


def _unit_interval_problem(objective_coefficients=(0.0, 0.0, 1.0)):
    objective = _system(
        ("x",),
        ("objective",),
        (0, 0, 0),
        ((0,), (1,), (2,)),
        objective_coefficients,
    )
    inequalities = _system(
        ("x",),
        ("unit interval",),
        (0, 0),
        ((0,), (2,)),
        (1.0, -1.0),
    )
    return PolynomialOptimizationProblem(objective, inequalities=inequalities)


def test_compact_univariate_known_optimum_has_matching_bounds_and_rank_one_atom():
    problem = _unit_interval_problem()
    prepared = prepare_polynomial_relaxation(problem, 1)

    assert isinstance(prepared.program.cone, ProductCone)
    assert isinstance(prepared.program.cone.cones[0], ZeroCone)
    assert all(
        isinstance(cone, PositiveSemidefiniteCone)
        for cone in prepared.program.cone.cones[1:]
    )

    dual = _dual_certificate(
        prepared,
        0.0,
        moment_matrix=jnp.asarray([[0.0, 0.0], [0.0, 1.0]]),
    )

    result = audit_polynomial_relaxation(
        prepared,
        _provider_result(prepared, (1.0, 0.0, 0.0), cone_dual=dual),
        candidate=jnp.asarray([0.0]),
    )

    np.testing.assert_allclose(result.lower_bound, 0.0, atol=1e-7)
    np.testing.assert_allclose(result.upper_bound, 0.0, atol=1e-7)
    assert bool(result.original_candidate_accepted)
    assert bool(result.flatness.flat)
    assert bool(result.exactness_evidenced)
    assert bool(result.dual.accepted)
    assert result.atom_extraction.status == AtomExtractionStatus.EXTRACTED
    np.testing.assert_allclose(result.atom_extraction.atoms, [[0.0]], atol=1e-7)


def test_compact_bivariate_known_optimum_uses_dense_moment_and_localizing_bases():
    objective = _system(
        ("x", "y"),
        ("radius squared",),
        (0, 0),
        ((2, 0), (0, 2)),
        (1.0, 1.0),
    )
    inequalities = _system(
        ("x", "y"),
        ("unit disk",),
        (0, 0, 0),
        ((0, 0), (2, 0), (0, 2)),
        (1.0, -1.0, -1.0),
    )
    problem = PolynomialOptimizationProblem(objective, inequalities=inequalities)
    prepared = prepare_polynomial_relaxation(problem, 1)

    assert prepared.template.moment_basis.moment_count == 6
    assert prepared.template.moment_basis.matrix_size == 3
    assert prepared.template.localizing_bases[0].matrix_size == 1

    dual = _dual_certificate(
        prepared,
        0.0,
        moment_matrix=jnp.diag(jnp.asarray([0.0, 1.0, 1.0])),
    )

    result = audit_polynomial_relaxation(
        prepared,
        _provider_result(
            prepared,
            (1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            cone_dual=dual,
        ),
        candidate=jnp.asarray([0.0, 0.0]),
    )
    np.testing.assert_allclose(result.lower_bound, 0.0, atol=1e-7)
    np.testing.assert_allclose(result.upper_bound, 0.0, atol=1e-7)
    assert bool(result.psd.accepted)
    assert bool(result.exactness_evidenced)
    assert bool(result.dual.accepted)


def test_equalities_and_inequalities_are_compiled_and_replayed_in_original_domain():
    objective = _system(("x",), ("x",), (0,), ((1,),), (1.0,))
    equalities = _system(
        ("x",),
        ("binary",),
        (0, 0),
        ((0,), (2,)),
        (-1.0, 1.0),
    )
    inequalities = _system(("x",), ("positive",), (0,), ((1,),), (1.0,))
    problem = PolynomialOptimizationProblem(
        objective,
        equalities=equalities,
        inequalities=inequalities,
    )
    prepared = prepare_polynomial_relaxation(problem, 2)

    dual = _dual_certificate(
        prepared,
        1.0,
        equality_blocks=((-1.0, 0.5, 0.0),),
        localizing_matrices=(jnp.asarray([[0.5, -0.5], [-0.5, 0.5]]),),
    )

    result = audit_polynomial_relaxation(
        prepared,
        _provider_result(
            prepared,
            (1.0, 1.0, 1.0, 1.0, 1.0),
            cone_dual=dual,
        ),
        candidate=jnp.asarray([1.0]),
    )

    assert prepared.plan.estimate.zero_rows == 4
    np.testing.assert_allclose(result.candidate.equality_values, [0.0], atol=1e-7)
    np.testing.assert_allclose(result.candidate.inequality_values, [1.0], atol=1e-7)
    np.testing.assert_allclose(result.lower_bound, 1.0, atol=1e-7)
    np.testing.assert_allclose(result.upper_bound, 1.0, atol=1e-7)
    assert bool(result.dual.accepted)


def test_low_but_compilable_order_keeps_nonzero_gap_and_no_exactness_evidence():
    objective = _system(("x",), ("x",), (0,), ((1,),), (1.0,))
    equalities = _system(
        ("x",),
        ("binary",),
        (0, 0),
        ((0,), (2,)),
        (-1.0, 1.0),
    )
    inequalities = _system(("x",), ("positive",), (0,), ((1,),), (1.0,))
    problem = PolynomialOptimizationProblem(
        objective,
        equalities=equalities,
        inequalities=inequalities,
    )
    prepared = prepare_polynomial_relaxation(problem, 1)
    dual = _dual_certificate(
        prepared,
        0.0,
        localizing_matrices=(jnp.ones((1, 1)),),
    )

    result = audit_polynomial_relaxation(
        prepared,
        _provider_result(prepared, (1.0, 0.0, 1.0), cone_dual=dual),
        candidate=jnp.asarray([1.0]),
    )

    np.testing.assert_allclose(result.lower_bound, 0.0, atol=1e-7)
    np.testing.assert_allclose(result.upper_bound, 1.0, atol=1e-7)
    assert not bool(result.bound_gap_closed)
    assert not bool(result.flatness.flat)
    assert not bool(result.exactness_evidenced)
    assert result.atom_extraction.status == AtomExtractionStatus.NOT_FLAT


def test_plan_reports_constant_infeasibility_and_resource_rejection_before_allocation():
    objective = _system(("x",), ("objective",), (0,), ((0,),), (0.0,))
    impossible = _system(("x",), ("impossible",), (0,), ((0,),), (-1.0,))
    infeasible = PolynomialOptimizationProblem(objective, inequalities=impossible)
    infeasible_plan = plan_polynomial_relaxation(infeasible, 1)
    assert infeasible_plan.status == PolynomialRelaxationStatus.STRUCTURALLY_INFEASIBLE
    with pytest.raises(ValueError, match="STRUCTURALLY_INFEASIBLE"):
        prepare_polynomial_relaxation(infeasible_plan)

    constrained = _unit_interval_problem()
    resources = PolynomialRelaxationResources(max_moments=2)
    rejected = plan_polynomial_relaxation(constrained, 1, resources=resources)
    assert rejected.status == PolynomialRelaxationStatus.RESOURCE_REJECTED
    assert rejected.estimate.moment_count == 3
    with pytest.raises(ValueError, match="RESOURCE_REJECTED"):
        prepare_polynomial_relaxation(rejected)


def test_insufficient_order_is_not_compiled_or_reported_as_exactness():
    quartic = _system(("x",), ("quartic",), (0,), ((4,),), (1.0,))
    problem = PolynomialOptimizationProblem(quartic)
    plan = plan_polynomial_relaxation(problem, 1)

    assert plan.required_order == 2
    assert plan.status == PolynomialRelaxationStatus.INSUFFICIENT_ORDER
    assert not plan.ready
    with pytest.raises(ValueError, match="INSUFFICIENT_ORDER"):
        prepare_polynomial_relaxation(plan)


def test_nonflat_moments_do_not_fabricate_atoms_or_exactness():
    problem = _unit_interval_problem()
    prepared = prepare_polynomial_relaxation(problem, 2)
    nonflat_moments = (1.0, 0.0, 2.0 / 3.0, 0.0, 2.0 / 3.0)

    result = audit_polynomial_relaxation(
        prepared,
        _provider_result(prepared, nonflat_moments),
    )

    assert bool(result.psd.accepted)
    assert not bool(result.flatness.flat)
    assert not bool(result.exactness_evidenced)
    assert result.atom_extraction.status == AtomExtractionStatus.NOT_FLAT
    assert result.atom_extraction.atoms.shape == (0, 1)
    assert result.atom_extraction.weights.shape == (0,)


def test_original_candidate_replay_controls_upper_bound_independently_of_provider_path():
    problem = _unit_interval_problem()
    feasible = audit_polynomial_candidate(problem, jnp.asarray([0.5]))
    infeasible = audit_polynomial_candidate(problem, jnp.asarray([2.0]))

    assert bool(feasible.feasible)
    np.testing.assert_allclose(feasible.objective, 0.25, atol=1e-7)
    assert not bool(infeasible.feasible)
    assert float(infeasible.inequality_violation) > 0.0

    prepared = prepare_polynomial_relaxation(problem, 1)
    rejected_provider = _provider_result(prepared, (1.0, 0.0, 0.0), accepted=False)
    result = audit_polynomial_relaxation(
        prepared,
        rejected_provider,
        candidate=jnp.asarray([0.5]),
    )
    assert not bool(result.provider_path_accepted)
    assert bool(result.original_candidate_accepted)
    assert not bool(result.lower_bound_available)
    assert bool(result.upper_bound_available)
    np.testing.assert_allclose(result.upper_bound, 0.25, atol=1e-7)


def test_numeric_refresh_reuses_topology_and_updates_only_coefficient_binding():
    original = _unit_interval_problem((0.0, 0.0, 1.0))
    refreshed_problem = _unit_interval_problem((1.0, 0.0, 2.0))
    prepared = prepare_polynomial_relaxation(original, 1)
    refreshed = refresh_polynomial_relaxation(prepared, refreshed_problem)

    assert refreshed.template is prepared.template
    assert refreshed.program.structure_id == prepared.program.structure_id
    assert refreshed.numeric_version == 1
    assert refreshed.binding_id != prepared.binding_id
    np.testing.assert_allclose(refreshed.program.linear, [1.0, 0.0, 2.0])
