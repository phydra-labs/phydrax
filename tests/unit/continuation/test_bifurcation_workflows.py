#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


ct = phx.continuation
nl = phx.nonlinear


def _geometry(problem, state, parameter, state_space):
    residual = problem.residual(state, parameter)
    return ct.ContinuationGeometry.resolve(
        state,
        residual,
        state_space=state_space,
        residual_space=state_space,
    )


def _nullspace_analyzer(
    right,
    left,
    singular_values,
    *,
    full_spectrum=True,
    source_success=True,
    analyzer_id="deterministic-nullspace",
):
    def analyze(problem, state, parameter, geometry, args):
        return ct.evaluate_nullspace(
            problem,
            state,
            parameter,
            geometry,
            right,
            left,
            jnp.asarray(singular_values),
            source_success=source_success,
            full_spectrum=full_spectrum,
            analyzer_id=analyzer_id,
            args=args,
        )

    return ct.CallableNullspaceAnalyzer(analyze, analyzer_id=analyzer_id)


def _successful_linear_solve(solution_function, *, condition=1.0):
    def solve(action, right_hand_side, system_id):
        solution = solution_function(action, right_hand_side, system_id)
        difference = jax.tree.map(
            lambda left, right: left - right,
            action(solution),
            right_hand_side,
        )
        squared_norms = [
            jnp.real(jnp.vdot(value, value)) for value in jax.tree.leaves(difference)
        ]
        residual = jnp.sqrt(sum(squared_norms, start=jnp.asarray(0.0)))
        return ct.NormalFormLinearSolveResult(
            solution=solution,
            residual_norm=residual,
            condition_estimate=condition,
            iterations=1,
            successful=True,
            source_status=0,
            solver_id="deterministic-linear-solve",
        )

    return ct.CallableNormalFormLinearSolver(
        solve,
        solver_id="deterministic-linear-solve",
    )


def test_fold_extended_system_exposes_blocks_and_requires_certificate():
    dtype = jnp.float32
    state_space = phx.linalg.PyTreeSpace({"x": jnp.asarray(0.0, dtype=dtype)})
    problem = ct.ParameterContinuationProblem(
        lambda state, parameter, args: {"x": state["x"] ** 2 - parameter},
        problem_id="quadratic-fold",
    )
    fold = ct.FoldProblem(
        problem,
        state_space,
        {"x": jnp.asarray(1.0, dtype=dtype)},
    )
    initial = ct.FoldState(
        {"x": jnp.asarray(0.0, dtype=dtype)},
        jnp.asarray(0.0, dtype=dtype),
        {"x": jnp.asarray(1.0, dtype=dtype)},
    )
    result = ct.FoldMethod(
        nl.NewtonKrylov(),
        residual_tolerance=1e-6,
    ).solve(fold, initial)

    assert bool(result.candidate_converged)
    np.testing.assert_allclose(np.asarray(result.convergence.block_norms), 0.0)
    assert set(result.residual_blocks.equilibrium) == {"x"}
    analyzer = _nullspace_analyzer(
        {"x": jnp.asarray(1.0, dtype=dtype)},
        {"x": jnp.asarray(1.0, dtype=dtype)},
        [0.0],
    )
    certificate = ct.certify_fold(
        fold,
        result,
        analyzer,
        ct.FoldAssumptions(
            smoothness_order=2,
            scalar_parameter_verified=True,
            local_fredholm_index_zero_verified=True,
        ),
    )
    normal_form = ct.fold_normal_form(
        problem,
        result.state.physical_state,
        result.state.parameter,
        certificate.geometry,
        certificate.evidence.nullspace,
    )

    assert bool(certificate.certified)
    assert bool(normal_form.successful)
    np.testing.assert_allclose(float(normal_form.coefficient), 1.0, rtol=1e-6)


def test_hopf_extended_system_and_spectral_certificate_are_distinct():
    dtype = jnp.float32
    state_space = phx.linalg.ArraySpace((2,), dtype=dtype)

    def vector_field(state, parameter, args):
        x, y = state
        radius_squared = x**2 + y**2
        return jnp.asarray(
            [
                parameter * x - y - x * radius_squared,
                x + parameter * y - y * radius_squared,
            ],
            dtype=dtype,
        )

    problem = ct.ParameterContinuationProblem(vector_field, problem_id="hopf-flow")
    scale = jnp.asarray(1.0 / jnp.sqrt(2.0), dtype=dtype)
    mode_real = jnp.asarray([scale, 0.0], dtype=dtype)
    mode_imaginary = jnp.asarray([0.0, -scale], dtype=dtype)
    hopf = ct.HopfProblem(
        problem,
        state_space,
        mode_real,
        mode_imaginary,
    )
    initial = ct.HopfState(
        jnp.zeros((2,), dtype=dtype),
        jnp.asarray(0.0, dtype=dtype),
        mode_real,
        mode_imaginary,
        jnp.asarray(1.0, dtype=dtype),
    )
    result = ct.HopfMethod(
        nl.NewtonKrylov(),
        residual_tolerance=1e-6,
    ).solve(hopf, initial)

    assert bool(result.candidate_converged)
    assert float(jnp.max(result.convergence.block_norms)) <= 1e-6

    def spectral_analysis(problem, candidate, state_space, args):
        return ct.HopfEigenEvidence(
            eigenvalues=jnp.asarray([1j, -1j], dtype=jnp.complex64),
            critical_pair_residual=0.0,
            adjoint_pair_residual=0.0,
            crossing_speed=1.0,
            pair_condition=1.0,
            source_status=0,
            source_success=True,
            analyzer_id="complete-hopf-spectrum",
            full_spectrum=True,
        )

    analyzer = ct.CallableHopfAnalyzer(
        spectral_analysis,
        analyzer_id="complete-hopf-spectrum",
    )
    certificate = ct.certify_hopf(
        hopf,
        result,
        analyzer,
        ct.HopfAssumptions(
            smoothness_order=3,
            autonomous_flow_verified=True,
            scalar_parameter_verified=True,
        ),
    )
    assert bool(certificate.certified)

    matrix = jnp.asarray([[0.0, -1.0], [1.0, 0.0]], dtype=jnp.complex64)

    def harmonic_solution(action, right_hand_side, system_id):
        operator = matrix if "zero-harmonic" in system_id else 2j * jnp.eye(2) - matrix
        return jnp.linalg.solve(operator, right_hand_side)

    normal_form = ct.hopf_first_lyapunov(
        problem,
        result.state,
        _geometry(
            problem,
            result.state.physical_state,
            result.state.parameter,
            state_space,
        ),
        mode_real,
        mode_imaginary,
        _successful_linear_solve(harmonic_solution),
    )
    assert bool(normal_form.successful)
    assert float(normal_form.first_lyapunov_coefficient) < 0.0


def test_pitchfork_certificate_drives_two_automatic_switches():
    dtype = jnp.float32
    state_space = phx.linalg.ArraySpace((), dtype=dtype)
    problem = ct.ParameterContinuationProblem(
        lambda state, parameter, args: parameter * state - state**3,
        problem_id="symmetric-pitchfork",
    )
    state = jnp.asarray(0.0, dtype=dtype)
    parameter = jnp.asarray(0.0, dtype=dtype)
    mode = jnp.asarray(1.0, dtype=dtype)
    geometry = _geometry(problem, state, parameter, state_space)
    analyzer = _nullspace_analyzer(mode, mode, [0.0])
    branch = ct.certify_branch_point(
        problem,
        state,
        parameter,
        geometry,
        analyzer,
        ct.BranchPointAssumptions(
            smoothness_order=3,
            scalar_parameter_verified=True,
            reference_branch_verified=True,
            local_fredholm_index_zero_verified=True,
        ),
    )
    assert bool(branch.certified)

    normal_form = ct.pitchfork_normal_form(
        problem,
        state,
        parameter,
        geometry,
        branch.evidence.nullspace,
        _successful_linear_solve(
            lambda action, right_hand_side, system_id: jnp.zeros_like(right_hand_side)
        ),
    )
    pitchfork = ct.certify_pitchfork(
        branch,
        problem,
        lambda value: -value,
        ct.PitchforkAssumptions(
            smoothness_order=3,
            symmetry_is_linear=True,
            symmetry_is_involutive=True,
            equation_equivariance_verified=True,
            reference_branch_symmetric=True,
            critical_mode_is_odd=True,
        ),
        quadratic_coefficient=normal_form.quadratic_coefficient,
        cubic_coefficient=normal_form.cubic_coefficient,
        normal_form_solve_residual=jnp.max(normal_form.diagnostics.linear_residuals),
        normal_form_condition=jnp.max(normal_form.diagnostics.linear_condition_estimates),
        normal_form_success=normal_form.successful,
    )
    seeds = ct.switch_branches_from_nullspace(pitchfork, amplitude=0.05)

    assert bool(normal_form.successful)
    np.testing.assert_allclose(float(normal_form.quadratic_coefficient), 0.0)
    np.testing.assert_allclose(float(normal_form.cubic_coefficient), -1.0, rtol=1e-6)
    assert bool(pitchfork.certified)
    np.testing.assert_allclose(float(seeds[0][0]), 0.05, rtol=1e-6)
    np.testing.assert_allclose(float(seeds[1][0]), -0.05, rtol=1e-6)


def test_incomplete_or_ill_conditioned_evidence_never_certifies():
    dtype = jnp.float32
    state_space = phx.linalg.ArraySpace((), dtype=dtype)
    problem = ct.ParameterContinuationProblem(
        lambda state, parameter, args: parameter * state - state**3,
        problem_id="insufficient-pitchfork",
    )
    state = jnp.asarray(0.0, dtype=dtype)
    mode = jnp.asarray(1.0, dtype=dtype)
    geometry = _geometry(
        problem,
        state,
        jnp.asarray(0.0, dtype=dtype),
        state_space,
    )
    incomplete = _nullspace_analyzer(
        mode,
        mode,
        [0.0],
        full_spectrum=False,
    )
    certificate = ct.certify_branch_point(
        problem,
        state,
        jnp.asarray(0.0, dtype=dtype),
        geometry,
        incomplete,
        ct.BranchPointAssumptions(
            smoothness_order=3,
            scalar_parameter_verified=True,
            reference_branch_verified=True,
            local_fredholm_index_zero_verified=True,
        ),
    )

    assert not bool(certificate.certified)
    assert int(certificate.status) == int(
        ct.BifurcationStatus.INSUFFICIENT_SPECTRAL_EVIDENCE
    )
    with pytest.raises(ValueError, match="certified nullspace"):
        ct.switch_branches_from_nullspace(certificate, amplitude=0.1)

    complete = _nullspace_analyzer(mode, mode, [0.0]).analyze(
        problem,
        state,
        jnp.asarray(0.0, dtype=dtype),
        geometry,
    )
    ill_conditioned = ct.pitchfork_normal_form(
        problem,
        state,
        jnp.asarray(0.0, dtype=dtype),
        geometry,
        complete,
        _successful_linear_solve(
            lambda action, right_hand_side, system_id: jnp.zeros_like(right_hand_side),
            condition=1e12,
        ),
    )
    assert not bool(ill_conditioned.successful)
    assert int(ill_conditioned.status) == int(ct.NormalFormStatus.ILL_CONDITIONED)


def test_linear_and_parameter_homotopies_have_exact_endpoints():
    dtype = jnp.float32
    start = nl.NonlinearSystemProblem(
        lambda state, args: state - 1.0,
        problem_id="easy-root",
    )
    target = nl.NonlinearSystemProblem(
        lambda state, args: state**2 - 4.0,
        problem_id="target-root",
    )
    homotopy = ct.linear_homotopy(start, target)
    endpoints = homotopy.verify_endpoints(
        jnp.asarray(1.0, dtype=dtype),
        jnp.asarray(2.0, dtype=dtype),
        tolerance=1e-6,
    )

    assert bool(endpoints.successful)
    np.testing.assert_allclose(
        float(homotopy.residual(jnp.asarray(3.0, dtype=dtype), 0.0)),
        2.0,
    )
    np.testing.assert_allclose(
        float(homotopy.residual(jnp.asarray(3.0, dtype=dtype), 1.0)),
        5.0,
    )

    parameter_problem = ct.ParameterContinuationProblem(
        lambda state, parameter, args: state - parameter,
        parameter_lower=-2.0,
        parameter_upper=3.0,
        problem_id="affine-parameter",
    )
    parameter_path = ct.parameter_homotopy(parameter_problem, -1.0, 2.0)
    np.testing.assert_allclose(float(parameter_path.physical_parameter(0.25)), -0.25)
    np.testing.assert_allclose(
        float(parameter_path.residual(jnp.asarray(-0.25, dtype=dtype), 0.25)),
        0.0,
        atol=1e-7,
    )


def test_metric_deflation_rejects_known_root_and_preserves_other_root():
    dtype = jnp.float32
    problem = nl.NonlinearSystemProblem(
        lambda state, args: state * (state - 1.0),
        problem_id="two-roots",
    )
    metric = ct.CallableDeflationMetric(
        lambda left, right: 2.0 * jnp.abs(left - right),
        metric_id="scaled-distance",
    )
    deflation = ct.RootDeflation(
        problem,
        [jnp.asarray(0.0, dtype=dtype)],
        metric=metric,
        policy=ct.DeflationPolicy(
            distance_floor=1e-3,
            known_root_tolerance=1e-5,
            original_residual_tolerance=1e-6,
        ),
    )
    np.testing.assert_allclose(
        float(deflation.residual(jnp.asarray(1.0, dtype=dtype))),
        0.0,
        atol=0.0,
    )
    termination = nl.NonlinearTermination(
        absolute_residual=1e-7,
        relative_residual=0.0,
        maximum_steps=4,
    )
    known = ct.solve_deflated(
        deflation,
        nl.NewtonKrylov(),
        jnp.asarray(0.0, dtype=dtype),
        termination=termination,
    )
    other = ct.solve_deflated(
        deflation,
        nl.NewtonKrylov(),
        jnp.asarray(1.0, dtype=dtype),
        termination=termination,
    )

    assert int(known.status) == int(ct.DeflatedRootStatus.KNOWN_ROOT_REJECTED)
    assert not bool(known.successful)
    assert bool(other.successful)
    np.testing.assert_allclose(float(other.state), 1.0, atol=0.0)
    assert float(other.minimum_known_root_distance) == pytest.approx(2.0)


def test_public_continuation_namespace_owns_workflows():
    names = (
        "FoldProblem",
        "HopfProblem",
        "certify_pitchfork",
        "switch_branches_from_nullspace",
        "hopf_first_lyapunov",
        "linear_homotopy",
        "RootDeflation",
    )
    assert all(hasattr(phx.continuation, name) for name in names)
    assert not hasattr(phx.dynamics.analysis, "ContinuationProblem")
    assert not hasattr(phx.dynamics.analysis, "branch_switch_seed")


def test_nullspace_evidence_respects_distinct_state_and_residual_spaces():
    state_space = phx.linalg.PyTreeSpace(
        {"x": jnp.zeros((1,), dtype=jnp.float64)},
        space_id="nullspace-state",
    )
    residual_space = phx.linalg.PyTreeSpace(
        {"f": jnp.zeros((1,), dtype=jnp.float64)},
        space_id="nullspace-residual",
    )
    problem = ct.ParameterContinuationProblem(
        lambda state, parameter, args: {"f": parameter * state["x"]},
        state_space=state_space,
        residual_space=residual_space,
        problem_id="distinct-nullspace-spaces",
    )
    state = {"x": jnp.zeros((1,), dtype=jnp.float64)}
    parameter = jnp.asarray(0.0, dtype=jnp.float64)
    geometry = ct.ContinuationGeometry.resolve(
        state,
        problem.residual(state, parameter),
        state_space=state_space,
        residual_space=residual_space,
    )
    evidence = ct.evaluate_nullspace(
        problem,
        state,
        parameter,
        geometry,
        {"x": jnp.ones((1,), dtype=jnp.float64)},
        {"f": jnp.ones((1,), dtype=jnp.float64)},
        jnp.asarray([0.0]),
        source_success=True,
        full_spectrum=True,
        analyzer_id="distinct-nullspace",
    )

    assert float(evidence.right_residual_norm) == 0.0
    assert float(evidence.left_residual_norm) == 0.0
    assert float(evidence.right_norm) == 1.0
    assert float(evidence.left_norm) == 1.0
