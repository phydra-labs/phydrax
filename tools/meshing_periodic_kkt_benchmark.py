#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import final, Protocol, runtime_checkable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from phydrax import StrictModule
from phydrax.linalg import SparseFactorizationPlan
from phydrax.meshing.providers import _native_periodic_size as size
from phydrax.optim import MinimizationProblem
from phydrax.optim._primal_dual import PrimalDualNewtonKrylov


class _CompilerMemory(Protocol):
    temp_size_in_bytes: int
    output_size_in_bytes: int
    argument_size_in_bytes: int
    generated_code_size_in_bytes: int


class _Executable(Protocol):
    def memory_analysis(self) -> _CompilerMemory: ...


class _CompiledPolish(Protocol):
    compiled: _Executable

    def __call__(
        self,
        parameters: jax.Array,
        args: size._PolishArgs,
        problem: MinimizationProblem,
        method: PrimalDualNewtonKrylov,
        /,
    ) -> size._SolveRecord: ...


class _LoweredPolish(Protocol):
    def compile(self) -> _CompiledPolish: ...


@runtime_checkable
class _PolishCompilerEntry(Protocol):
    """The public lowering capability of the existing Equinox JIT entry."""

    def lower(
        self,
        parameters: jax.Array,
        args: size._PolishArgs,
        problem: MinimizationProblem,
        method: PrimalDualNewtonKrylov,
        /,
    ) -> _LoweredPolish: ...


@final
class _CapturedPolishInput(StrictModule):
    parameters: jax.Array
    args: size._PolishArgs
    problem: MinimizationProblem
    lower: np.ndarray
    upper: np.ndarray
    mutable: np.ndarray
    active: np.ndarray
    roots: np.ndarray
    edge_keys: np.ndarray
    source_binding: str
    maximum_work: int
    maximum_bytes: int
    input_preparation: float


@final
class _PreparedPolishBenchmark(StrictModule):
    captured: _CapturedPolishInput
    method: PrimalDualNewtonKrylov
    setup: size._PeriodicSparseKKTSetup
    factor_plan: SparseFactorizationPlan
    coefficient_dtype: np.dtype
    rhs_dtype: np.dtype
    symbolic_work_upper: int
    hard_work_phases: tuple[tuple[str, int], ...]
    symbolic_preparation: float


@final
class _PolishCompilation(StrictModule):
    compiled: _CompiledPolish
    lowering: float
    compilation: float


@final
class _PolishExecution(StrictModule):
    cold: size._SolveRecord
    cold_seconds: float
    warm_seconds: list[float]


def _prepare_captured_polish_input(snapshot: Path) -> _CapturedPolishInput:
    """Load and bind the captured constraints, scientific identity, and rank witness."""
    started = perf_counter()
    with np.load(snapshot, allow_pickle=False) as bank:
        parameters = jnp.asarray(bank["parameters"])
        source = size._SizeArgs(
            *(jnp.asarray(bank[name]) for name in size._SizeArgs._fields)
        )
        args = size._PolishArgs(
            source,
            jnp.asarray(bank["active"]),
            jnp.asarray(bank["goals"]),
            jnp.asarray(bank["mutable_edges"], dtype=jnp.int32),
        )
        lower = np.array(bank["edge_lower"], copy=True)
        upper = np.array(bank["edge_upper"], copy=True)
    # Rebuild through the scientific owner and refuse a changed snapshot contract.
    mutable = np.asarray(args.mutable_edges)
    active = np.asarray(args.active)
    unequal = lower[lower != upper]
    minimum = float(unequal[0]) if unequal.size else 0.0
    problem = size._polish_problem(mutable, active, minimum)
    if not (
        np.array_equal(np.asarray(problem.constraints[0].lower), lower)
        and np.array_equal(np.asarray(problem.constraints[0].upper), upper)
    ):
        raise ValueError("Snapshot does not match the current original hard constraints.")
    suffix = ".polish-input.npz"
    if not snapshot.name.endswith(suffix):
        raise ValueError(
            "Captured hard-solve input must retain its original artifact name."
        )
    stem = snapshot.name.removesuffix(suffix)
    identity = json.loads(snapshot.with_name(f"{stem}.evidence.json").read_text())
    if not isinstance(identity, dict):
        raise TypeError("Captured scientific identity must be a metadata mapping.")
    source_binding = identity.get("source_binding_id")
    limits = identity.get("original_limits")
    if not isinstance(source_binding, str) or not isinstance(limits, dict):
        raise TypeError(
            "Captured hard solve must retain its real scientific source and limits."
        )
    maximum_work = limits.get("maximum_work_units")
    maximum_bytes = limits.get("maximum_scratch_bytes")
    if not isinstance(maximum_work, int) or not isinstance(maximum_bytes, int):
        raise TypeError("Captured original work/storage allowances must be integers.")
    with np.load(
        snapshot.with_name(f"{stem}.rank-witness.npz"), allow_pickle=False
    ) as witness:
        roots = np.array(witness["free"], copy=True)
        edge_keys = np.array(witness["native_quotient_edge_keys"], copy=True)
        if not np.array_equal(roots, np.asarray(args.size.free)):
            raise ValueError(
                "Captured free-root ownership differs from the actual hard solve."
            )
        if not np.array_equal(edge_keys[:, :2], np.asarray(args.size.edges)):
            raise ValueError(
                "Captured native edge keys differ from the actual hard solve."
            )
        if not np.array_equal(witness["kept"], active):
            raise ValueError(
                "Captured equality-row ownership differs from the actual hard solve."
            )
    input_preparation = perf_counter() - started
    return _CapturedPolishInput(
        parameters=parameters,
        args=args,
        problem=problem,
        lower=lower,
        upper=upper,
        mutable=mutable,
        active=active,
        roots=roots,
        edge_keys=edge_keys,
        source_binding=source_binding,
        maximum_work=maximum_work,
        maximum_bytes=maximum_bytes,
        input_preparation=input_preparation,
    )


def _prepare_polish_benchmark(
    captured: _CapturedPolishInput,
) -> _PreparedPolishBenchmark:
    """Prepare the sparse symbolic owners and synchronize their numerical leaves."""
    parameters, args = captured.parameters, captured.args
    started = perf_counter()
    method, symbolic_work_upper = size._prepare_polish_method(
        captured.source_binding,
        captured.roots,
        captured.edge_keys,
        captured.active,
        np.asarray(args.size.mutable),
        captured.maximum_work,
        captured.maximum_bytes,
        args,
    )
    hard_work_phases = size._polish_work_bounds(
        args.size.origins.size,
        args.size.edges.shape[0],
        args.size.cells.shape[0],
        parameters.size,
        captured.mutable.size,
        int(np.count_nonzero(captured.active)),
        method,
    )
    setup, factor_plan = size._polish_sparse_owners(method)
    coefficient_dtype = setup.kkt_assembly.template.coefficients.dtype
    rhs_dtype = setup.primal_space.dtype
    jax.block_until_ready((parameters, args, method))
    symbolic_preparation = perf_counter() - started
    return _PreparedPolishBenchmark(
        captured=captured,
        method=method,
        setup=setup,
        factor_plan=factor_plan,
        coefficient_dtype=coefficient_dtype,
        rhs_dtype=rhs_dtype,
        symbolic_work_upper=symbolic_work_upper,
        hard_work_phases=hard_work_phases,
        symbolic_preparation=symbolic_preparation,
    )


def _measure_polish_compilation(
    preparation: _PreparedPolishBenchmark,
) -> _PolishCompilation:
    """Lower and compile the unchanged hard-polish entry with its prepared owners."""
    captured = preparation.captured
    parameters, args, problem, method = (
        captured.parameters,
        captured.args,
        captured.problem,
        preparation.method,
    )
    entry = size._solve_polish
    if not isinstance(entry, _PolishCompilerEntry):
        raise TypeError("The owning hard-polish entry must support lowering.")
    started = perf_counter()
    lowered = entry.lower(parameters, args, problem, method)
    lowering = perf_counter() - started
    started = perf_counter()
    compiled = lowered.compile()
    compilation = perf_counter() - started
    return _PolishCompilation(
        compiled=compiled, lowering=lowering, compilation=compilation
    )


def _execute_polish_benchmark(
    preparation: _PreparedPolishBenchmark,
    compilation: _PolishCompilation,
    repeats: int,
) -> _PolishExecution:
    """Measure cold and repeated executions, requiring leafwise exact reproduction."""
    captured = preparation.captured
    parameters, args, problem, method = (
        captured.parameters,
        captured.args,
        captured.problem,
        preparation.method,
    )
    compiled = compilation.compiled
    started = perf_counter()
    cold = jax.block_until_ready(compiled(parameters, args, problem, method))
    cold_seconds = perf_counter() - started
    warm_seconds: list[float] = []
    for _ in range(repeats):
        started = perf_counter()
        warm = jax.block_until_ready(compiled(parameters, args, problem, method))
        warm_seconds.append(perf_counter() - started)
        for first, second in zip(
            jax.tree.leaves(cold), jax.tree.leaves(warm), strict=True
        ):
            np.testing.assert_array_equal(np.asarray(first), np.asarray(second))
    return _PolishExecution(
        cold=cold, cold_seconds=cold_seconds, warm_seconds=warm_seconds
    )


def _polish_derivative_report(
    preparation: _PreparedPolishBenchmark,
) -> dict[str, object]:
    """Report prepared exact derivative support and logical operation bounds."""
    setup, coefficient_dtype = preparation.setup, preparation.coefficient_dtype
    return {
        "scope": "Current true Lagrangian Hessian and original canonical bound Jacobian.",
        "representation": "Prepared exact local edge norm blocks and oriented cell cross-product minors.",
        "local_derivative_id": setup.local_derivative_id,
        "original_mutable_edge_blocks": setup.edge_positions.shape[0],
        "original_source_cell_blocks": setup.cell_positions.shape[0],
        "edge_gradient_shape": [setup.edge_positions.shape[0], 2],
        "edge_hessian_shape": [setup.edge_positions.shape[0], 2, 2],
        "cell_gradient_shape": [setup.cell_positions.shape[0], 3, 2],
        "cell_hessian": "Exact constant oriented2x2 minors, not a dense6x6 bank.",
        "sparse_saddle_contributions": setup.kkt_assembly.template.relation.capacity,
        "preconditioner": "Native unit-L/signed-U-magnitude paired congruence, not LDL or abs(H).",
        "kkt_representation": "Prepared true local Hessian+barrier rank-one blocks, fused saddle action.",
        "canonical_equality_rows": setup.derivative_source.equality_indices.size,
        "canonical_lower_rows": setup.derivative_source.lower_indices.size,
        "canonical_upper_rows": setup.derivative_source.upper_indices.size,
        "one_derivative_preparation_work_upper": setup.derivative_preparation_work_upper(),
        "one_sparse_saddle_setup_work_upper": setup.sparse_setup_work_upper(),
        "one_kkt_action_work_upper": setup.kkt_action_work_upper(),
        "one_kkt_preparation_work_upper": setup.kkt_preparation_work_upper(),
        "logical_workspace_bytes_upper": setup.derivative_workspace_bytes_upper(
            coefficient_dtype.itemsize
        ),
    }


def _polish_substitution_report(
    preparation: _PreparedPolishBenchmark,
) -> dict[str, object]:
    """Report native substitution support and dtype-dependent logical bounds."""
    factor_plan = preparation.factor_plan
    coefficient_dtype, rhs_dtype = (
        preparation.coefficient_dtype,
        preparation.rhs_dtype,
    )
    return {
        "scope": "Prepared logical upper bounds, not measured host/device memory.",
        "coefficient_dtype": str(coefficient_dtype),
        "rhs_dtype": str(rhs_dtype),
        "symbolic_factor_nnz": factor_plan.factor_nnz,
        "lower_symbolic_nnz": factor_plan.lower_positions.size,
        "refresh_preparation_work_upper": (
            factor_plan.numeric_substitution_preparation_work_units_upper
            + factor_plan.lu_congruence_preparation_work_units_upper
        ),
        "one_rhs_work_upper": factor_plan.lu_congruence_solve_work_units_upper_for(
            coefficient_dtype, rhs_dtype
        ),
        "promoted_rhs_work_upper": factor_plan.lu_congruence_solve_work_units_upper_for(
            coefficient_dtype, np.dtype("complex128")
        ),
        "lower_scheduled_coefficient_slots": factor_plan.lower_analysis.schedule_positions.size,
        "upper_scheduled_coefficient_slots": factor_plan.lower_analysis.transpose_schedule_positions.size,
        "lower_original_reduction_width": factor_plan.lower_analysis.row_width,
        "upper_original_reduction_width": factor_plan.lower_analysis.transpose_row_width,
        "lower_analysis_pattern_id": factor_plan.lower_analysis.pattern_id,
        "upper_analysis_pattern_id": factor_plan.lower_analysis.pattern_id,
        "cache_logical_storage_bytes_upper": (
            factor_plan.numeric_substitution_storage_bytes_upper(
                coefficient_dtype.itemsize
            )
            + factor_plan.lu_congruence_storage_bytes_upper(coefficient_dtype.itemsize)
        ),
        "refresh_logical_workspace_bytes_upper": (
            factor_plan.numeric_substitution_refresh_workspace_bytes_upper(
                coefficient_dtype.itemsize
            )
            + factor_plan.lu_congruence_refresh_workspace_bytes_upper(
                coefficient_dtype.itemsize
            )
        ),
    }


def _polish_benchmark_report(
    snapshot: Path,
    preparation: _PreparedPolishBenchmark,
    compilation: _PolishCompilation,
    execution: _PolishExecution,
) -> dict[str, object]:
    """Join execution measurements and prepared diagnostics without changing identities."""
    captured = preparation.captured
    parameters, args, problem, method = (
        captured.parameters,
        captured.args,
        captured.problem,
        preparation.method,
    )
    leaves = {
        id(leaf): leaf
        for leaf in jax.tree.leaves(
            eqx.filter((parameters, args, problem, method), eqx.is_array)
        )
    }
    memory = compilation.compiled.compiled.memory_analysis()
    return {
        "scope": "Captured original hard solve only; no placement, publication or archive claim.",
        "snapshot": str(snapshot),
        "method_id": size._POLISH_METHOD.method_id,
        "problem_id": problem.problem_id,
        "dimension": parameters.size,
        "kept_equalities": int(np.count_nonzero(captured.lower == captured.upper)),
        "retained_input_bytes": sum(leaf.nbytes for leaf in leaves.values()),
        "stages_seconds": {
            "input_preparation": captured.input_preparation,
            "symbolic_preparation": preparation.symbolic_preparation,
            "lowering": compilation.lowering,
            "compilation": compilation.compilation,
            "cold_execution": execution.cold_seconds,
            "warm_execution": execution.warm_seconds,
        },
        "compiler_memory": {
            "temporary_bytes": memory.temp_size_in_bytes,
            "output_bytes": memory.output_size_in_bytes,
            "argument_bytes": memory.argument_size_in_bytes,
            "generated_code_bytes": memory.generated_code_size_in_bytes,
        },
        "symbolic_preparation_work_upper": preparation.symbolic_work_upper,
        "hard_solve_work_upper": sum(work for _, work in preparation.hard_work_phases),
        "hard_solve_work_upper_by_phase": dict(preparation.hard_work_phases),
        "prepared_exact_derivatives": _polish_derivative_report(preparation),
        "native_numeric_substitution": _polish_substitution_report(preparation),
        "source_binding_id": captured.source_binding,
        "hard_solve": size._polish_record(execution.cold, method)._asdict(),
    }


def benchmark(snapshot: Path, repeats: int) -> dict[str, object]:
    """Time the unchanged captured hard problem, not a new meshing qualification."""
    if repeats < 1:
        raise ValueError("repeats must be positive.")
    captured = _prepare_captured_polish_input(snapshot)
    preparation = _prepare_polish_benchmark(captured)
    compilation = _measure_polish_compilation(preparation)
    execution = _execute_polish_benchmark(preparation, compilation, repeats)
    return _polish_benchmark_report(snapshot, preparation, compilation, execution)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(benchmark(args.snapshot, args.repeats), allow_nan=True))


if __name__ == "__main__":
    main()
