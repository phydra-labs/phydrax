#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import numpy as np

from benchmarks.advanced_solvers.campaign import build_cases, CampaignConfig
from benchmarks.advanced_solvers.certificates import independent_certificate
from benchmarks.advanced_solvers.problems import (
    default_problems,
    general_eigenproblem,
    l1_composite_optimization,
    maratos_constrained_optimization,
    nonlinear_root,
    quadratic_fold,
    rosenbrock_optimization,
    semilinear_poisson_root,
    sparse_block_linear,
    sparse_scalar_linear,
    variational_inequality,
)


def test_generators_are_seed_deterministic_and_fingerprints_cover_values():
    first = default_problems(size=12, seed=17)
    second = default_problems(size=12, seed=17)
    changed = default_problems(size=12, seed=18)

    assert first.keys() == second.keys()
    assert {
        name: problem.identity()["fingerprint"] for name, problem in first.items()
    } == {name: problem.identity()["fingerprint"] for name, problem in second.items()}
    assert (
        first["linear-scalar"].identity()["fingerprint"]
        != changed["linear-scalar"].identity()["fingerprint"]
    )


def test_sparse_scalar_and_block_systems_are_symmetric_positive_definite():
    scalar = sparse_scalar_linear(size=10, right_hand_sides=1, seed=2)
    block = sparse_block_linear(
        block_count=6,
        block_size=2,
        right_hand_sides=2,
        seed=3,
    )

    for problem in (scalar, block):
        np.testing.assert_allclose(problem.matrix, problem.matrix.T, atol=0.0)
        assert np.min(np.linalg.eigvalsh(problem.matrix)) > 0.0
        solution = np.linalg.solve(problem.matrix, problem.rhs)
        certificate = independent_certificate(problem, solution, {})
        assert certificate["relative_residual"] < 1e-12
        assert certificate["backward_error"] < 1e-12

    assert block.block_size == 2
    assert block.rhs.shape == (12, 2)


def test_nonlinear_root_and_vi_certificates_use_matched_problem_relations():
    root = nonlinear_root(size=8, seed=5)
    root_solution = np.sqrt(root.target)
    root_certificate = independent_certificate(root, root_solution, {})
    assert root_certificate["kind"] == "nonlinear-root"
    assert root_certificate["residual_norm"] < 1e-12

    vi = variational_inequality(size=8, seed=6)
    vi_solution = np.maximum(vi.target / vi.diagonal, 0.0)
    vi_certificate = independent_certificate(vi, vi_solution, {})
    assert vi_certificate["kind"] == "variational-inequality-natural-map"
    assert vi_certificate["residual_norm"] < 1e-12
    assert 0 < vi_certificate["details"]["active_lower_count"] < vi.initial.size


def test_semilinear_poisson_root_has_sparse_spd_jacobian_and_exact_solution():
    problem = semilinear_poisson_root(size=12, seed=9)
    coordinates = np.arange(1, 13, dtype=np.float64) / 13.0
    exact = np.sin(np.pi * coordinates)
    jacobian = problem.jacobian(exact)

    assert problem.root_kind == "semilinear-poisson-1d"
    assert problem.sizes()["nnz"] == 34
    assert np.linalg.norm(problem.residual(exact)) < 1e-12
    np.testing.assert_allclose(jacobian, jacobian.T, atol=0.0)
    assert np.min(np.linalg.eigvalsh(jacobian)) > 0.0


def test_campaign_exposes_matched_root_modes_without_changing_default_cases():
    cases = build_cases(CampaignConfig(seed=17, size=8, warmup=0, repeats=1))

    assert cases["nonlinear-root-dense"].solver_mode == "dense"
    assert cases["nonlinear-root-matrix-free"].solver_mode == "matrix-free"
    assert cases["nonlinear-root-sparse-pde"].solver_mode == "sparse"
    assert (
        cases["nonlinear-root-dense"].problem.identity()["fingerprint"]
        == cases["nonlinear-root-matrix-free"].problem.identity()["fingerprint"]
    )
    assert cases["nonlinear-root-sparse-pde"].problem.root_kind == (
        "semilinear-poisson-1d"
    )


def test_general_eigen_and_continuation_certificates_are_independent_relations():
    eigen = general_eigenproblem(size=10, eigenpairs=3, seed=7)
    eigenvalues, eigenvectors = np.linalg.eig(eigen.matrix)
    selected = np.argsort(np.abs(eigenvalues))[-eigen.eigenpairs :]
    eigen_certificate = independent_certificate(
        eigen,
        eigenvectors[:, selected],
        {"eigenvalues": eigenvalues[selected]},
    )
    assert eigen_certificate["kind"] == "eigenpair-relation"
    assert eigen_certificate["relative_residual"] < 1e-12

    fold = quadratic_fold(seed=8)
    states = np.linspace(1.0, -1.0, 21)
    coordinates = states * states
    branch_certificate = independent_certificate(
        fold,
        states,
        {
            "coordinates": coordinates,
            "branch_successful": True,
            "residual_tolerance": 1e-10,
        },
    )
    assert branch_certificate["kind"] == "continuation-branch-residual"
    assert branch_certificate["residual_norm"] < 1e-12
    assert branch_certificate["details"]["state_sign_change"] is True
    assert branch_certificate["details"]["tangent_coordinate_sign_change"] is True
    assert branch_certificate["details"]["fold_bracket"] is True
    assert branch_certificate["details"]["successful_fold_traversal"] is True


def test_optimization_certificates_use_independent_stationarity_and_kkt_relations():
    unconstrained = rosenbrock_optimization(size=6, seed=19)
    unconstrained_certificate = independent_certificate(
        unconstrained,
        unconstrained.optimum,
        {},
    )
    assert unconstrained_certificate["kind"] == "optimization-stationarity"
    assert unconstrained_certificate["residual_norm"] == 0.0

    constrained = maratos_constrained_optimization(seed=20)
    constrained_certificate = independent_certificate(
        constrained,
        constrained.optimum,
        {},
    )
    assert constrained_certificate["kind"] == "optimization-kkt"
    assert constrained_certificate["residual_norm"] == 0.0
    assert constrained_certificate["details"]["equality_violation"] == 0.0
    assert constrained_certificate["details"]["inequality_violation"] == 0.0

    proximal = l1_composite_optimization(size=7, seed=21)
    proximal_certificate = independent_certificate(
        proximal,
        proximal.optimum,
        {},
    )
    assert proximal_certificate["kind"] == "optimization-proximal-stationarity"
    assert proximal_certificate["residual_norm"] < 1e-15


def test_continuation_certificate_rejects_residual_only_nonfold_branch():
    fold = quadratic_fold(seed=9)
    states = np.linspace(1.0, 0.5, 8)
    certificate = independent_certificate(
        fold,
        states,
        {
            "coordinates": states * states,
            "branch_successful": True,
            "residual_tolerance": 1e-10,
        },
    )

    assert certificate["residual_norm"] < 1e-12
    assert certificate["details"]["fold_bracket"] is False
    assert certificate["details"]["successful_fold_traversal"] is False
