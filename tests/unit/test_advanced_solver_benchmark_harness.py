#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import json
from dataclasses import replace
from importlib import import_module
from types import SimpleNamespace

import numpy as np

from benchmarks.advanced_solvers.adapters import adapter_names, load_adapter
from benchmarks.advanced_solvers.adapters.base import (
    Availability,
    BenchmarkAdapter,
    CaseSpec,
    Implementation,
    RefreshEvidence,
    SolveResult,
    Tolerances,
)
from benchmarks.advanced_solvers.adapters.optimistix import _root_solver
from benchmarks.advanced_solvers.certificates import independent_certificate
from benchmarks.advanced_solvers.cli import main as benchmark_main
from benchmarks.advanced_solvers.harness import execute_case
from benchmarks.advanced_solvers.problems import (
    general_eigenproblem,
    nonlinear_root,
    semilinear_poisson_root,
    sparse_scalar_linear,
    variational_inequality,
)
from benchmarks.advanced_solvers.schema import validate_row


class _ExactAdapter(BenchmarkAdapter):
    name = "exact"
    dependency = "numpy"
    capabilities = frozenset({"linear.scalar"})

    def __init__(self):
        self.events = []

    def availability(self, capability, /):
        return Availability(True, capability, "numpy", np.__version__, None)

    def implementation(self, spec, /):
        return Implementation(
            "exact", "numpy-cpu", "solve", "none", {"numpy": np.__version__}
        )

    def setup(self, spec, /):
        self.events.append("setup")
        return {
            "spec": spec,
            "matrix": spec.problem.matrix,
            "certificate_problem": spec.problem,
        }

    def compilation_applicable(self, setup_state, /):
        return True

    def compile(self, setup_state, /):
        self.events.append("compilation")
        return setup_state

    def preparation_applicable(self, compiled_state, /):
        return True

    def prepare(self, compiled_state, /):
        self.events.append("preparation")
        return compiled_state

    def solve(self, prepared_state, /):
        self.events.append("solve")
        problem = prepared_state["spec"].problem
        return SolveResult(
            solution=np.linalg.solve(prepared_state["matrix"], problem.rhs),
            auxiliary={},
            converged=True,
            message="exact solve",
            operations={
                "iterations": 0,
                "matvecs": 0,
                "preconditioner_applications": 0,
                "linear_solves": 1,
                "nonlinear_evaluations": 0,
                "jacobian_evaluations": 0,
            },
        )

    def differentiation_applicable(self, prepared_state, /):
        return True

    def compile_differentiation(self, prepared_state, /):
        self.events.append("differentiation_compilation")
        return prepared_state

    def differentiate(self, prepared_state, /):
        self.events.append("differentiation")
        return np.linalg.inv(prepared_state["matrix"])

    def refresh_applicable(self, prepared_state, /):
        return True

    def refresh(self, prepared_state, /):
        self.events.append("refresh")
        problem = prepared_state["spec"].problem
        refreshed_problem = replace(
            problem,
            coefficients=problem.coefficients * 1.01,
        )
        prepared_state["matrix"] = refreshed_problem.matrix
        prepared_state["certificate_problem"] = refreshed_problem
        return prepared_state, RefreshEvidence(
            applicable=True,
            symbolic_reused=True,
            numeric_refreshed=True,
            symbolic_refresh_count=0,
            numeric_refresh_count=1,
            evidence="deterministic coefficient-scale refresh",
        )

    def certificate_problem(self, prepared_state, /):
        return prepared_state["certificate_problem"]

    def memory(self, prepared_state, result, /):
        matrix_bytes = prepared_state["matrix"].nbytes
        return {
            "matrix_bytes": matrix_bytes,
            "setup_bytes": 0,
            "peak_estimate_bytes": matrix_bytes + np.asarray(result.solution).nbytes,
            "evidence": "exact NumPy arrays",
        }


class _MissingAdapter(_ExactAdapter):
    def availability(self, capability, /):
        return Availability(
            False,
            capability,
            "missing-solver",
            None,
            "required module 'missing_solver' is not installed for adapter 'exact'",
        )


def test_harness_separates_phases_excludes_warmup_and_verifies_independently():
    problem = sparse_scalar_linear(size=8, right_hand_sides=1, seed=3)
    spec = CaseSpec("linear-scalar", problem, Tolerances(max_steps=20))
    adapter = _ExactAdapter()

    row = execute_case(
        adapter,
        spec,
        environment=_environment(),
        warmup=2,
        repeats=3,
    )

    assert adapter.events == [
        "setup",
        "compilation",
        "preparation",
        "solve",
        "solve",
        "solve",
        "solve",
        "solve",
        "differentiation_compilation",
        "differentiation",
        "differentiation",
        "differentiation",
        "refresh",
        "solve",
    ]
    assert row["timing"]["solve"]["count"] == 3
    assert row["timing"]["setup"]["count"] == 1
    assert row["timing"]["compilation"]["count"] == 1
    assert row["timing"]["preparation"]["count"] == 1
    assert row["timing"]["differentiation_compilation"]["count"] == 1
    assert row["timing"]["differentiation"]["count"] == 3
    assert row["timing"]["refresh"]["count"] == 1
    assert row["timing"]["refreshed_solve"]["count"] == 1
    assert row["timing"]["refreshed_verification"]["count"] == 1
    assert row["timing"]["verification"]["count"] == 1
    assert row["certificate"]["independently_computed"] is True
    assert row["problem"]["fingerprint"] == problem.identity()["fingerprint"]
    assert (
        row["refresh"]["certificate_problem_fingerprint"]
        != problem.identity()["fingerprint"]
    )
    assert row["certificate"]["relative_residual"] < 1e-12
    assert row["refresh"]["symbolic_reused"] is True
    assert row["refresh"]["independently_certified"] is True
    assert row["refresh"]["certificate_converged"] is True
    assert row["refresh"]["certificate_relative_residual"] < 1e-12
    assert row["transfers"]["host_to_device_bytes"] == 0
    assert row["transfers"]["device_to_host_bytes"] == 0
    validate_row(row)
    json.dumps(row, allow_nan=False)


def test_unavailable_adapter_emits_precise_skip_without_executing_setup():
    problem = sparse_scalar_linear(size=8, right_hand_sides=1, seed=4)
    adapter = _MissingAdapter()

    row = execute_case(
        adapter,
        CaseSpec("linear-scalar", problem),
        environment=_environment(),
        warmup=1,
        repeats=2,
    )

    assert adapter.events == []
    assert row["outcome"]["status"] == "skipped"
    assert row["outcome"]["skip_reason"] == (
        "required module 'missing_solver' is not installed for adapter 'exact'"
    )
    assert row["certificate"]["relative_residual"] is None
    assert row["transfers"]["host_to_device_bytes"] is None
    assert row["transfers"]["device_to_host_bytes"] is None
    assert row["timing"]["solve"]["count"] == 0
    validate_row(row)


def test_optional_adapters_load_without_importing_optional_dependencies_eagerly():
    assert adapter_names() == (
        "phydrax",
        "jax",
        "lineax",
        "optimistix",
        "scipy",
        "pyamg",
        "amgcl",
        "amgx",
        "petsc",
        "slepc",
    )
    for name in adapter_names():
        adapter = load_adapter(name)
        assert adapter.name == name

    unsupported = load_adapter("slepc").availability("linear.scalar")
    assert unsupported.available is False
    assert unsupported.reason == (
        "adapter 'slepc' does not implement capability 'linear.scalar'"
    )


def test_public_phydrax_adapter_declares_every_representative_family():
    adapter = load_adapter("phydrax")

    assert adapter.capabilities == frozenset(
        {
            "linear.scalar",
            "linear.block",
            "nonlinear.root",
            "nonlinear.vi",
            "eigen.general",
            "continuation.fold",
            "optimization.unconstrained",
            "optimization.constrained",
            "optimization.proximal",
        }
    )


def test_capabilities_cli_emits_all_common_solver_families(capsys):
    benchmark_main(["capabilities", "--adapter", "phydrax"])
    payload = json.loads(capsys.readouterr().out)

    assert set(payload["phydrax"]) == {
        "linear.scalar",
        "linear.block",
        "nonlinear.root",
        "nonlinear.vi",
        "eigen.general",
        "continuation.fold",
        "optimization.unconstrained",
        "optimization.constrained",
        "optimization.proximal",
    }


def _environment():
    return {
        "fingerprint": "test-environment",
        "python_version": "3.12.0",
        "platform": "test-platform",
        "machine": "test-machine",
        "processor": "test-processor",
        "logical_cpus": 1,
        "numpy_version": np.__version__,
        "jax": {"version": "test", "backend": "cpu", "devices": []},
        "thread_environment": {},
    }


def test_phydrax_sparse_linear_adapter_runs_public_canonical_operator_contract():
    problem = sparse_scalar_linear(size=8, right_hand_sides=1, seed=21)
    row = execute_case(
        load_adapter("phydrax"),
        CaseSpec("linear-scalar", problem),
        environment=_environment(),
        warmup=0,
        repeats=1,
    )

    assert row["outcome"]["status"] == "success"
    assert row["certificate"]["relative_residual"] < 1e-12
    validate_row(row)


def test_phydrax_nonlinear_adapter_uses_prepared_refresh_lifecycle():
    problem = nonlinear_root(size=4, seed=22)
    row = execute_case(
        load_adapter("phydrax"),
        CaseSpec("nonlinear-root", problem),
        environment=_environment(),
        warmup=0,
        repeats=1,
    )

    assert row["outcome"]["status"] == "success"
    assert row["certificate"]["relative_residual"] < 1e-8
    assert row["timing"]["compilation"]["count"] == 1
    assert row["timing"]["preparation"]["count"] == 1
    assert row["timing"]["differentiation_compilation"]["count"] == 1
    assert row["timing"]["differentiation"]["count"] == 1
    assert row["timing"]["refresh"]["count"] == 1
    assert row["timing"]["refreshed_solve"]["count"] == 1
    assert row["timing"]["refreshed_verification"]["count"] == 1
    assert row["refresh"]["symbolic_reused"] is True
    assert row["refresh"]["numeric_refreshed"] is True
    assert row["refresh"]["independently_certified"] is True
    assert row["refresh"]["certificate_converged"] is True
    assert row["refresh"]["certificate_relative_residual"] < 1e-8
    validate_row(row)


def test_matched_dense_and_matrix_free_root_modes_run_end_to_end():
    problem = nonlinear_root(size=4, seed=23)
    expected_methods = {
        ("phydrax", "dense"): "newton+dense-lu",
        ("phydrax", "matrix-free"): "newton+matrix-free-gmres",
        ("optimistix", "dense"): "optimistix-newton+dense-lu",
        (
            "optimistix",
            "matrix-free",
        ): "optimistix-newton+matrix-free-gmres",
    }

    for (adapter_name, solver_mode), method in expected_methods.items():
        row = execute_case(
            load_adapter(adapter_name),
            CaseSpec(
                f"nonlinear-root-{solver_mode}",
                problem,
                Tolerances(relative=1e-8, absolute=1e-10, max_steps=20),
                solver_mode=solver_mode,
            ),
            environment=_environment(),
            warmup=0,
            repeats=1,
        )

        assert row["implementation"]["method"] == method
        assert row["outcome"]["status"] == "success"
        assert row["certificate"]["relative_residual"] < 1e-8
        assert row["timing"]["differentiation_compilation"]["count"] == 1
        assert row["timing"]["differentiation"]["count"] == 1
        validate_row(row)


def test_phydrax_sparse_root_runs_prepared_numeric_refresh_lifecycle():
    problem = semilinear_poisson_root(size=8, seed=24)
    adapter = load_adapter("phydrax")
    spec = CaseSpec(
        "nonlinear-root-sparse-pde",
        problem,
        Tolerances(relative=1e-8, absolute=1e-10, max_steps=50),
        solver_mode="sparse",
    )
    setup_state = adapter.setup(spec)
    assert setup_state.method.linear_policy.method.name == "pcg"
    row = execute_case(
        adapter,
        spec,
        environment=_environment(),
        warmup=0,
        repeats=1,
    )

    assert row["implementation"]["method"] == "newton+sparse-pcg"
    assert row["implementation"]["preconditioner"] == "jacobi"
    assert row["outcome"]["status"] == "success"
    assert row["certificate"]["relative_residual"] < 1e-8
    assert row["timing"]["differentiation"]["count"] == 0
    assert row["timing"]["refresh"]["count"] == 1
    assert row["refresh"]["symbolic_reused"] is True
    assert row["refresh"]["numeric_refreshed"] is True
    assert row["refresh"]["certificate_converged"] is True
    validate_row(row)


def test_optimistix_sparse_reference_uses_dimension_scaled_linear_budget():
    problem = semilinear_poisson_root(size=128, seed=24)
    spec = CaseSpec(
        "nonlinear-root-sparse-pde",
        problem,
        Tolerances(relative=1e-8, absolute=1e-10, max_steps=50),
        solver_mode="sparse",
    )
    state = load_adapter("optimistix").setup(spec)
    solver = _root_solver(
        state,
        import_module("optimistix"),
        import_module("lineax"),
    )
    linear_solver = solver.linear_solver

    assert linear_solver.restart == 16
    assert linear_solver.max_steps * linear_solver.restart == problem.initial.size


def test_root_differentiation_matches_analytic_implicit_sensitivity():
    problem = nonlinear_root(size=2, seed=31)
    spec = CaseSpec(
        "nonlinear-root",
        problem,
        Tolerances(relative=1e-8, absolute=1e-10, max_steps=50),
    )
    expected = np.diag(0.5 / np.sqrt(problem.target))

    for adapter_name in ("phydrax", "optimistix"):
        adapter = load_adapter(adapter_name)
        assert adapter.availability(spec.capability).available
        state = adapter.setup(spec)
        compilation_applicable = adapter.compilation_applicable(state)
        compile_after_preparation = (
            compilation_applicable and adapter.compilation_after_preparation(state)
        )
        if compilation_applicable and not compile_after_preparation:
            state = adapter.compile(state)
        if adapter.preparation_applicable(state):
            state = adapter.prepare(state)
        if compile_after_preparation:
            state = adapter.compile(state)
        assert adapter.differentiation_applicable(state)
        state = adapter.compile_differentiation(state)
        derivative = np.asarray(adapter.differentiate(state))

        assert derivative.shape == expected.shape
        assert derivative.dtype == expected.dtype
        assert np.all(np.isfinite(derivative))
        assert np.allclose(derivative, expected, rtol=1e-8, atol=1e-10)


def test_scipy_vi_memory_counts_initial_target_bounds_and_diagonal():
    problem = variational_inequality(size=8, seed=12)
    adapter = load_adapter("scipy")
    state = adapter.setup(CaseSpec("nonlinear-vi", problem))
    result = SolveResult(
        solution=problem.initial,
        auxiliary={},
        converged=False,
        message="not solved",
        operations={},
    )

    memory = adapter.memory(state, result)
    assert problem.lower is not None
    assert problem.upper is not None
    assert problem.diagonal is not None

    expected = sum(
        array.nbytes
        for array in (
            problem.initial,
            problem.target,
            problem.lower,
            problem.upper,
            problem.diagonal,
        )
    )
    assert memory["matrix_bytes"] == expected


def test_phydrax_implementation_records_source_fingerprint():
    problem = sparse_scalar_linear(size=8, right_hand_sides=1, seed=13)
    implementation = load_adapter("phydrax").implementation(
        CaseSpec("linear-scalar", problem)
    )

    fingerprint = implementation.versions["phydrax_source_sha256"]
    assert len(fingerprint) == 64
    int(fingerprint, 16)


def test_phydrax_source_fingerprint_covers_non_init_modules(tmp_path, monkeypatch):
    from benchmarks.advanced_solvers.adapters import phydrax as phydrax_adapter

    package_root = tmp_path / "phydrax"
    package_root.mkdir()
    package_init = package_root / "__init__.py"
    implementation = package_root / "_implementation.py"
    package_init.write_text("version = 1\n", encoding="utf-8")
    implementation.write_text("algorithm = 1\n", encoding="utf-8")
    monkeypatch.setattr(
        phydrax_adapter.importlib.util,
        "find_spec",
        lambda name: SimpleNamespace(origin=str(package_init)),
    )

    first = phydrax_adapter._source_fingerprint()
    implementation.write_text("algorithm = 2\n", encoding="utf-8")
    second = phydrax_adapter._source_fingerprint()

    assert first != second


def test_scipy_arpack_no_pair_returns_honest_nonconverged_candidate(monkeypatch):
    from benchmarks.advanced_solvers.adapters import scipy as scipy_adapter

    problem = general_eigenproblem(size=8, eigenpairs=2, seed=14)

    class NoConvergence(Exception):
        def __init__(self):
            self.eigenvalues = np.asarray([], dtype=np.complex128)
            self.eigenvectors = np.empty(
                (problem.matrix.shape[0], 0), dtype=np.complex128
            )

    def fail_eigs(*args, **kwargs):
        del args, kwargs
        raise NoConvergence()

    fake_sparse_linalg = SimpleNamespace(
        eigs=fail_eigs,
        ArpackNoConvergence=NoConvergence,
    )
    monkeypatch.setattr(
        scipy_adapter,
        "import_module",
        lambda name: fake_sparse_linalg,
    )
    spec = CaseSpec("general-eigen", problem)
    state = scipy_adapter._ScipyState(spec=spec, matrix=problem.matrix)

    result = scipy_adapter.ScipyAdapter().solve(state)
    certificate = independent_certificate(
        problem,
        result.solution,
        result.auxiliary,
    )

    assert result.converged is False
    assert result.solution.shape == (problem.matrix.shape[0], 1)
    assert np.isfinite(certificate["relative_residual"])


def test_slepc_initial_space_is_seed_deterministic():
    from benchmarks.advanced_solvers.adapters.slepc import (
        _deterministic_initial_vector,
    )

    first = general_eigenproblem(size=8, eigenpairs=2, seed=15)
    same = general_eigenproblem(size=8, eigenpairs=2, seed=15)
    changed = general_eigenproblem(size=8, eigenpairs=2, seed=16)

    np.testing.assert_array_equal(
        _deterministic_initial_vector(first),
        _deterministic_initial_vector(same),
    )
    assert not np.array_equal(
        _deterministic_initial_vector(first),
        _deterministic_initial_vector(changed),
    )
