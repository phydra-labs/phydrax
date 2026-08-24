#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import platform
import time
from collections.abc import Callable
from typing import Any

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp

import phydrax as phx


def _timed(call: Callable[[], Any], repeats: int, /) -> tuple[Any, float]:
    value = call()
    jax.block_until_ready(value)
    started = time.perf_counter()
    for _ in range(repeats):
        value = call()
    jax.block_until_ready(value)
    return value, (time.perf_counter() - started) / repeats


def _fd_case(size: int, repeats: int, /) -> dict[str, Any]:
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformAxisSpec(
                size,
                endpoint=False,
                periodic=True,
            ),
        ),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    coordinate = jnp.arange(size, dtype=jnp.float64) / size
    exact = 2.0 * jnp.pi * jnp.cos(2.0 * jnp.pi * coordinate)
    configurations = {
        "float64": phx.discretization.FDExecutionPrecisionPolicy(),
        "field32_accum64": phx.discretization.FDExecutionPrecisionPolicy(
            coefficient_dtype="float32",
            field_dtype="float32",
            accumulation_dtype="float64",
            certification_dtype="float64",
        ),
    }
    results = {}
    for name, precision in configurations.items():
        discretization = phx.discretization.periodic_finite_difference(
            grid,
            accuracy_order=4,
            precision=precision,
        )
        operator = discretization.operator("d_x_1")
        state = jnp.sin(2.0 * jnp.pi * coordinate).astype(precision.field_dtype)
        apply = eqx.filter_jit(operator.mv)
        value, seconds = _timed(
            lambda apply=apply, state=state: apply(state),
            repeats,
        )
        estimate = phx.discretization.FDExecutionPreflightPlan(
            grid,
            field_count=1,
            operators=(operator,),
            precision=precision,
        ).estimate()
        results[name] = {
            "seconds": seconds,
            "maximum_absolute_error": float(
                jnp.max(jnp.abs(value.astype(jnp.float64) - exact))
            ),
            "state_bytes": estimate.state_bytes,
            "stencil_metadata_bytes": estimate.stencil_metadata_bytes,
            "output_dtype": value.dtype.name,
            "precision_evidence_id": precision.evidence().evidence_id,
        }
    return results


def _integration_case(order: int, repeats: int, /) -> dict[str, Any]:
    domain = phx.domain.ScalarInterval(0.0, 1.0, label="x")
    function = domain.Function("x")(lambda value: value**4)
    target = phx.integration.over(domain.component())
    plan = phx.integration.FixedQuadraturePlan(phx.integration.GaussLegendreRule(order))
    configurations = {
        "float64": phx.integration.IntegrationPrecisionPolicy(
            evaluation_dtype="float64",
            accumulation_dtype="float64",
            decision_dtype="float64",
            output_dtype="float64",
        ),
        "evaluate32_accum64": phx.integration.IntegrationPrecisionPolicy(
            evaluation_dtype="float32",
            accumulation_dtype="float64",
            decision_dtype="float64",
            output_dtype="float32",
        ),
    }
    results = {}
    for name, precision in configurations.items():
        realization = phx.integration.materialize(
            target,
            plan,
            precision=precision,
        )
        reduce = eqx.filter_jit(
            lambda realization=realization: (
                phx.integration.reduce(
                    function,
                    realization,
                ).value.data
            )
        )
        value, seconds = _timed(reduce, repeats)
        estimate = phx.integration.reduce(function, realization)
        results[name] = {
            "seconds": seconds,
            "absolute_error": float(jnp.abs(value.astype(jnp.float64) - 0.2)),
            "output_dtype": value.dtype.name,
            "precision_evidence_id": estimate.precision_evidence.evidence_id,
        }
    return results


def _linalg_case(size: int, repeats: int, /) -> dict[str, Any]:
    diagonal = jnp.linspace(2.0, 4.0, size, dtype=jnp.float64)
    off_diagonal = jnp.full((size - 1,), -0.25, dtype=jnp.float64)
    matrix = jnp.diag(diagonal)
    matrix = matrix + jnp.diag(off_diagonal, 1) + jnp.diag(off_diagonal, -1)
    problem = phx.linalg.LinearSystem(phx.linalg.DenseLinearOperator(matrix))
    right_hand_side = jnp.ones((size,), dtype=jnp.float64)
    configurations = {
        "float64": phx.linalg.LinearSolvePolicy(
            phx.linalg.GMRES(restart=min(size, 16)),
            tolerance=phx.linalg.TolerancePolicy(relative=1e-10, max_steps=64),
            preconditioning=phx.linalg.PreconditioningPolicy(
                phx.linalg.JacobiPreconditionerBuilder()
            ),
            differentiation=phx.linalg.DifferentiationPolicy("none"),
        ),
        "preconditioner32_basis32": phx.linalg.LinearSolvePolicy(
            phx.linalg.GMRES(restart=min(size, 16)),
            tolerance=phx.linalg.TolerancePolicy(relative=1e-10, max_steps=64),
            preconditioning=phx.linalg.PreconditioningPolicy(
                phx.linalg.JacobiPreconditionerBuilder()
            ),
            precision=phx.linalg.MixedPrecisionPolicy(
                preconditioner_dtype=jnp.float32,
                krylov_dtype=jnp.float32,
            ),
            differentiation=phx.linalg.DifferentiationPolicy("none"),
        ),
    }
    results = {}
    for name, policy in configurations.items():
        prepared = phx.linalg.prepare(problem, policy)
        execute = eqx.filter_jit(
            lambda prepared=prepared: (
                phx.linalg.solve(
                    prepared,
                    right_hand_side,
                ).value
            )
        )
        value, seconds = _timed(execute, repeats)
        solve_result = phx.linalg.solve(prepared, right_hand_side)
        estimate = phx.linalg.plan(problem, policy).candidates[-1]
        results[name] = {
            "seconds": seconds,
            "relative_residual": float(
                jnp.linalg.norm(matrix @ value - right_hand_side)
                / jnp.linalg.norm(right_hand_side)
            ),
            "krylov_basis_bytes_per_rhs": estimate.krylov_basis_bytes_per_rhs,
            "preconditioner_storage_bytes": estimate.preconditioner_storage_bytes,
            "output_dtype": value.dtype.name,
            "effective_precision": (
                None
                if solve_result.provenance.effective_precision is None
                else {
                    "preconditioner": (
                        solve_result.provenance.effective_precision.preconditioner_dtype
                    ),
                    "krylov": solve_result.provenance.effective_precision.krylov_dtype,
                    "residual": (
                        solve_result.provenance.effective_precision.residual_dtype
                    ),
                }
            ),
        }
    return results


def _predictive_case(draws: int, width: int, repeats: int, /) -> dict[str, Any]:
    source = jnp.linspace(0.0, 1.0, draws * width, dtype=jnp.float64).reshape(
        (draws, width)
    )
    configurations = {
        "float64": phx.uq.PredictivePrecisionPolicy(
            storage_dtype="float64",
            summary_dtype="float64",
        ),
        "storage32_summary64": phx.uq.PredictivePrecisionPolicy(
            storage_dtype="float32",
            summary_dtype="float64",
        ),
    }
    results = {}
    for name, precision in configurations.items():
        predictive = phx.uq.PredictiveField(
            cx.Field(source, dims=("draw", "x")),
            (phx.uq.SampleAxis("draw", "epistemic"),),
            precision=precision,
        )
        summarize = eqx.filter_jit(lambda predictive=predictive: predictive.mean().data)
        value, seconds = _timed(summarize, repeats)
        results[name] = {
            "seconds": seconds,
            "storage_bytes": int(
                predictive.samples.data.size * predictive.samples.data.dtype.itemsize
            ),
            "maximum_mean_error": float(
                jnp.max(jnp.abs(value - jnp.mean(source, axis=0)))
            ),
            "summary_dtype": value.dtype.name,
            "precision_evidence_id": predictive.precision_evidence.evidence_id,
        }
    return results


def _nonlinear_case(size: int, repeats: int, /) -> dict[str, Any]:
    configurations = {
        "float64": (
            jnp.float64,
            phx.nonlinear.NonlinearPrecisionPolicy(
                state_dtype="float64",
                residual_dtype="float64",
                accumulation_dtype="float64",
                decision_dtype="float64",
                output_dtype="float64",
            ),
            1e-10,
        ),
        "state32_accum64": (
            jnp.float32,
            phx.nonlinear.NonlinearPrecisionPolicy(
                state_dtype="float32",
                residual_dtype="float32",
                accumulation_dtype="float64",
                decision_dtype="float64",
                output_dtype="float32",
            ),
            1e-5,
        ),
    }
    results = {}
    for name, (dtype, precision, tolerance) in configurations.items():
        space = phx.linalg.ArraySpace((size,), dtype=dtype)
        target = jnp.full((size,), 2.0, dtype=dtype)
        problem = phx.nonlinear.NonlinearSystemProblem(
            lambda state, _: state**2 - target,
            state_space=space,
            residual_space=space,
            problem_id=f"precision-benchmark:{name}",
        )
        initial = jnp.full((size,), 1.5, dtype=dtype)
        termination = phx.nonlinear.NonlinearTermination(
            absolute_residual=tolerance,
            relative_residual=0.0,
            maximum_steps=12,
        )
        execute = eqx.filter_jit(
            lambda problem=problem, initial=initial, termination=termination, precision=precision: (
                phx.nonlinear.root(
                    problem,
                    initial,
                    termination=termination,
                    precision=precision,
                ).state
            )
        )
        value, seconds = _timed(execute, repeats)
        result = phx.nonlinear.root(
            problem,
            initial,
            termination=termination,
            precision=precision,
        )
        precision_evidence = result.precision_evidence
        assert precision_evidence is not None
        results[name] = {
            "seconds": seconds,
            "maximum_absolute_error": float(
                jnp.max(jnp.abs(value.astype(jnp.float64) - jnp.sqrt(2.0)))
            ),
            "final_residual_norm": float(result.diagnostics.final_residual_norm),
            "output_dtype": jnp.asarray(value).dtype.name,
            "precision_evidence_id": precision_evidence.evidence_id,
        }
    return results


def _temporal_case(steps: int, repeats: int, /) -> dict[str, Any]:
    configurations = {
        "float64": (
            jnp.float64,
            phx.solver.TemporalPrecisionPolicy(
                state_dtype="float64",
                stage_dtype="float64",
                accumulation_dtype="float64",
                decision_dtype="float64",
                output_dtype="float64",
            ),
        ),
        "state32_accum64": (
            jnp.float32,
            phx.solver.TemporalPrecisionPolicy(
                state_dtype="float32",
                stage_dtype="float32",
                accumulation_dtype="float64",
                decision_dtype="float64",
                checkpoint_dtype="float32",
                output_dtype="float32",
            ),
        ),
    }
    results = {}
    save_times = jnp.asarray([0.0, 1.0], dtype=jnp.float64)
    for name, (dtype, precision) in configurations.items():
        problem = phx.solver.DifferentialProblem(
            lambda time, state, _: -state,
            jnp.ones((1,), dtype=dtype),
            t0=0.0,
            t1=1.0,
            problem_id=f"precision-benchmark:{name}",
        )
        solver = phx.solver.SSPRK33(precision=precision)
        solve = eqx.filter_jit(
            lambda problem=problem, solver=solver: phx.solver.solve_diffrax(
                problem,
                save_times=save_times,
                solver=solver,
                dt0=1.0 / steps,
                max_steps=steps + 1,
            )
        )
        value, seconds = _timed(lambda solve=solve: solve().states[-1], repeats)
        solution = solve()
        temporal_evidence = solution.temporal_evidence
        assert temporal_evidence is not None
        evidence = temporal_evidence.precision_evidence
        assert evidence is not None
        results[name] = {
            "seconds": seconds,
            "maximum_absolute_error": float(
                jnp.max(jnp.abs(value.astype(jnp.float64) - jnp.exp(-1.0)))
            ),
            "output_dtype": value.dtype.name,
            "precision_evidence_id": evidence.evidence_id,
        }
    return results


def _geometry_case(size: int, repeats: int, /) -> dict[str, Any]:
    reference_state = jnp.linspace(-1.0, 1.0, size, dtype=jnp.float64)
    reference_target = jnp.sin(reference_state)
    reference_prediction = reference_target + 1e-3 * jnp.cos(reference_state)
    reference = jnp.sum((reference_prediction - reference_target) ** 2)
    configurations = {
        "float64": (
            jnp.float64,
            phx.metrix.GeometryPrecisionPolicy(
                coordinate_dtype="float64",
                compute_dtype="float64",
                accumulation_dtype="float64",
                decision_dtype="float64",
            ),
        ),
        "coordinate32_accum64": (
            jnp.float32,
            phx.metrix.GeometryPrecisionPolicy(
                coordinate_dtype="float32",
                compute_dtype="float32",
                accumulation_dtype="float64",
                decision_dtype="float64",
            ),
        ),
    }
    results = {}
    for name, (dtype, precision) in configurations.items():
        state = reference_state.astype(dtype)
        target = reference_target.astype(dtype)
        prediction = reference_prediction.astype(dtype)
        metric = phx.terms.EuclideanFlowMatchingMetric(precision=precision)
        execute = eqx.filter_jit(
            lambda metric=metric, state=state, prediction=prediction, target=target: (
                metric(state, prediction, target)
            )
        )
        value, seconds = _timed(execute, repeats)
        results[name] = {
            "seconds": seconds,
            "absolute_reduction_error": float(
                jnp.abs(value.astype(jnp.float64) - reference)
            ),
            "output_dtype": value.dtype.name,
            "precision_evidence_id": precision.evidence_for(state).evidence_id,
        }
    return results


def _finite_volume_case(cells: int, repeats: int, /) -> dict[str, Any]:
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(cells, periodic=True),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    system = phx.equations.EulerSystem()
    discretization = phx.discretization.FiniteVolumePlan(
        grid,
        component_names=system.component_names,
    ).prepare()
    problem = phx.equations.ConservationProblemIR(
        "precision-benchmark-fv",
        "state",
        system,
        phx.discretization.FiniteVolumeBoundarySet.periodic(("x",)),
    )
    method = phx.discretization.FiniteVolumeMethodPlan(
        phx.discretization.PiecewiseConstantReconstruction(),
        phx.discretization.HLLCFluxPlan(),
    )
    configurations = {
        "float64": (
            jnp.float64,
            phx.discretization.FiniteVolumePrecisionPolicy(),
        ),
        "storage32_reduction64": (
            jnp.float32,
            phx.discretization.FiniteVolumePrecisionPolicy(
                "float32",
                reconstruction_dtype="float32",
                flux_dtype="float32",
                reduction_dtype="float64",
                output_dtype="float32",
                checkpoint_dtype="float32",
            ),
        ),
    }
    results = {}
    coordinate = jnp.arange(cells, dtype=jnp.float64) / cells
    for name, (dtype, precision) in configurations.items():
        compiled = phx.equations.compile_conservation_problem(
            problem,
            discretization,
            method,
            precision=precision,
        )
        runtime = phx.solver.PreparedFiniteVolumeRuntime(
            compiled.dynamics,
            phx.discretization.FluxPositivityPlan(),
        )
        primitive = jnp.stack(
            (
                1.0 + 0.01 * jnp.sin(2.0 * jnp.pi * coordinate),
                jnp.full((cells,), 0.1),
                jnp.ones((cells,)),
            ),
            axis=-1,
        ).astype(dtype)
        state = precision.storage(system.primitive_to_conserved(primitive))
        initial = phx.solver.FiniteVolumeRuntimeState(
            state,
            precision.decision(0.0),
            precision.decision(1e-4),
        )
        execute = eqx.filter_jit(
            lambda runtime=runtime, initial=initial: (
                runtime.advance(initial).runtime_state.conservative_state
            )
        )
        value, seconds = _timed(execute, repeats)
        advanced = runtime.advance(initial)
        _, diagnostics = compiled.dynamics.residual_with_diagnostics(
            precision.decision(0.0),
            state,
        )
        results[name] = {
            "seconds": seconds,
            "state_bytes": int(value.size * value.dtype.itemsize),
            "state_dtype": value.dtype.name,
            "conservation_defect": float(
                jnp.max(jnp.abs(diagnostics.conservation_defect))
            ),
            "precision_evidence_id": advanced.precision_evidence.evidence_id,
        }
    return results


def _hermitian_case(repeats: int, /) -> dict[str, Any]:
    matrix = jnp.asarray(
        [[2.0, 0.25 + 0.1j], [0.25 - 0.1j, 1.5]],
        dtype=jnp.complex128,
    )
    configurations = {
        "complex128": phx.linalg.HermitianPrecisionPolicy(
            compute_dtype="float64",
            factorization_dtype="float64",
            accumulation_dtype="float64",
            decision_dtype="float64",
            output_dtype="float64",
        ),
        "factorization64_output32": phx.linalg.HermitianPrecisionPolicy(
            compute_dtype="float64",
            factorization_dtype="float64",
            accumulation_dtype="float64",
            decision_dtype="float64",
            output_dtype="float32",
        ),
    }
    results = {}
    for name, precision in configurations.items():
        execute = eqx.filter_jit(
            lambda precision=precision: (
                phx.linalg.hermitian_sqrt(
                    matrix,
                    precision=precision,
                ).value
            )
        )
        value, seconds = _timed(execute, repeats)
        result = phx.linalg.hermitian_sqrt(matrix, precision=precision)
        reconstruction = value.astype(jnp.complex128) @ value.astype(jnp.complex128)
        results[name] = {
            "seconds": seconds,
            "reconstruction_error": float(jnp.linalg.norm(reconstruction - matrix)),
            "output_dtype": value.dtype.name,
            "precision_evidence_id": result.spectrum.precision_evidence.evidence_id,
        }
    return results


def _optimization_case(dimension: int, repeats: int, /) -> dict[str, Any]:
    results = {}
    configurations = {
        "float64": (
            jnp.float64,
            phx.nonlinear.NonlinearPrecisionPolicy(
                state_dtype="float64",
                residual_dtype="float64",
                direction_dtype="float64",
                accumulation_dtype="float64",
                decision_dtype="float64",
                output_dtype="float64",
            ),
        ),
        "model32_accum64": (
            jnp.float32,
            phx.nonlinear.NonlinearPrecisionPolicy(
                state_dtype="float32",
                residual_dtype="float32",
                direction_dtype="float32",
                accumulation_dtype="float64",
                decision_dtype="float64",
                output_dtype="float32",
            ),
        ),
    }
    for name, (dtype, precision) in configurations.items():
        target = jnp.linspace(0.5, 1.5, dimension, dtype=dtype)
        problem = phx.optim.NonlinearLeastSquaresProblem(
            lambda parameters, _: parameters - target,
            problem_id=f"precision-benchmark:{name}",
        )
        initial = jnp.zeros((dimension,), dtype=dtype)
        method = phx.optim.POUNDERS(
            initial_radius=0.5,
            maximum_dimension=max(dimension, 2),
            precision=precision,
        )
        termination = phx.optim.OptimizationTermination(
            absolute_optimality=1e-5 if dtype == jnp.float32 else 1e-9,
            relative_optimality=0.0,
            maximum_steps=8,
        )
        execute = lambda: method.solve(
            problem,
            initial,
            termination=termination,
            args=None,
        )
        value, seconds = _timed(lambda: execute().parameters, repeats)
        result = execute()
        evidence = result.precision_evidence
        assert evidence is not None
        results[name] = {
            "seconds": seconds,
            "maximum_absolute_error": float(
                jnp.max(
                    jnp.abs(
                        jnp.asarray(value, dtype=jnp.float64) - target.astype(jnp.float64)
                    )
                )
            ),
            "final_optimality_norm": float(result.diagnostics.final_optimality_norm),
            "output_dtype": jnp.asarray(value).dtype.name,
            "precision_evidence_id": evidence.evidence_id,
        }
    return results


def _open_system_case(steps: int, repeats: int, /) -> dict[str, Any]:
    results = {}
    configurations = {
        "complex128": (
            jnp.complex128,
            phx.metrix.GeometryPrecisionPolicy(
                coordinate_dtype="complex128",
                compute_dtype="complex128",
                accumulation_dtype="complex128",
                decision_dtype="float64",
                output_dtype="complex128",
            ),
            phx.solver.TemporalPrecisionPolicy(
                state_dtype="complex128",
                stage_dtype="complex128",
                accumulation_dtype="complex128",
                decision_dtype="float64",
                output_dtype="complex128",
            ),
        ),
        "state64_accum128": (
            jnp.complex64,
            phx.metrix.GeometryPrecisionPolicy(
                coordinate_dtype="complex64",
                compute_dtype="complex64",
                accumulation_dtype="complex128",
                decision_dtype="float64",
                output_dtype="complex64",
            ),
            phx.solver.TemporalPrecisionPolicy(
                state_dtype="complex64",
                stage_dtype="complex64",
                accumulation_dtype="complex128",
                decision_dtype="float64",
                output_dtype="complex64",
            ),
        ),
    }
    for name, (dtype, geometry, temporal) in configurations.items():
        lowering = jnp.asarray(
            [[0.0, 1.0], [0.0, 0.0]],
            dtype=dtype,
        )
        problem = phx.solver.QuantumJumpProblem(
            phx.solver.StateVectorOperator.from_matrix(
                jnp.zeros((2, 2), dtype=dtype),
                operator_id="zero",
            ),
            (
                phx.solver.StateVectorOperator.from_matrix(
                    jnp.sqrt(0.2) * lowering,
                    operator_id="decay",
                ),
            ),
            jnp.asarray([0.0, 1.0], dtype=dtype),
            geometry_precision=geometry,
            problem_id=f"precision-benchmark:{name}",
        )
        execute = lambda: phx.solver.solve_quantum_jump_ensemble(
            problem,
            jax.random.key(0),
            step_size=jnp.asarray(0.01, dtype=jnp.float32),
            steps=steps,
            trajectory_count=16,
            temporal_precision=temporal,
            geometry_precision=geometry,
        )
        value, seconds = _timed(lambda: execute().states, repeats)
        result = execute()
        evidence = result.precision_evidence
        assert evidence is not None
        results[name] = {
            "seconds": seconds,
            "trajectory_bytes": int(value.size * value.dtype.itemsize),
            "output_dtype": value.dtype.name,
            "statistical_error": float(result.approximation.statistical_error),
            "precision_evidence_id": evidence.evidence_id,
            "approximation_policy_ids": list(result.approximation.precision_policy_ids),
        }
    return results


def _tensor_network_case(sites: int, repeats: int, /) -> dict[str, Any]:
    if sites < 2:
        raise ValueError("tensor_sites must be at least two.")
    results = {}
    configurations = {
        "complex128": (
            jnp.complex128,
            phx.tensor_network.TensorNetworkPrecisionPolicy(
                storage_dtype="complex128",
                contraction_dtype="complex128",
                factorization_dtype="complex128",
                accumulation_dtype="complex128",
                decision_dtype="float64",
                output_dtype="complex128",
            ),
        ),
        "storage64_factor128": (
            jnp.complex64,
            phx.tensor_network.TensorNetworkPrecisionPolicy(
                storage_dtype="complex64",
                contraction_dtype="complex64",
                factorization_dtype="complex128",
                accumulation_dtype="complex128",
                decision_dtype="float64",
                output_dtype="complex64",
            ),
        ),
    }
    for name, (dtype, precision) in configurations.items():
        local_states = jnp.zeros((sites, 2), dtype=dtype).at[:, 0].set(1.0)
        state = phx.tensor_network.product_mps(
            local_states,
            precision=precision,
        )
        gate = jnp.eye(4, dtype=dtype).reshape((2, 2, 2, 2))
        execute = lambda: phx.tensor_network.apply_two_site_gate(
            state,
            sites // 2 - 1,
            gate,
            maximum_bond_dimension=2,
        )
        value, seconds = _timed(lambda: execute()[0].to_dense(), repeats)
        _, truncation = execute()
        results[name] = {
            "seconds": seconds,
            "state_bytes": int(
                sum(tensor.size * tensor.dtype.itemsize for tensor in state.tensors)
            ),
            "dense_norm_error": float(
                jnp.abs(jnp.linalg.norm(value.astype(jnp.complex128)) - 1.0)
            ),
            "discarded_weight": float(truncation.discarded_weight),
            "output_dtype": value.dtype.name,
            "precision_evidence_id": truncation.precision_evidence.evidence_id,
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--fd-size", type=int, default=4096)
    parser.add_argument("--quadrature-order", type=int, default=64)
    parser.add_argument("--linear-size", type=int, default=128)
    parser.add_argument("--draws", type=int, default=4096)
    parser.add_argument("--nonlinear-size", type=int, default=128)
    parser.add_argument("--temporal-steps", type=int, default=256)
    parser.add_argument("--geometry-size", type=int, default=4096)
    parser.add_argument("--fv-cells", type=int, default=4096)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--optimization-dimension", type=int, default=4)
    parser.add_argument("--open-system-steps", type=int, default=32)
    parser.add_argument("--tensor-sites", type=int, default=8)
    arguments = parser.parse_args()
    if arguments.repeats < 1:
        raise ValueError("repeats must be positive.")
    payload = {
        "environment": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "backend": jax.default_backend(),
            "device": str(jax.devices()[0]),
        },
        "finite_difference": _fd_case(arguments.fd_size, arguments.repeats),
        "integration": _integration_case(
            arguments.quadrature_order,
            arguments.repeats,
        ),
        "linear_algebra": _linalg_case(arguments.linear_size, arguments.repeats),
        "nonlinear": _nonlinear_case(arguments.nonlinear_size, arguments.repeats),
        "optimization": _optimization_case(
            arguments.optimization_dimension,
            arguments.repeats,
        ),
        "temporal": _temporal_case(arguments.temporal_steps, arguments.repeats),
        "geometry": _geometry_case(arguments.geometry_size, arguments.repeats),
        "finite_volume": _finite_volume_case(
            arguments.fv_cells,
            arguments.repeats,
        ),
        "hermitian_spectral": _hermitian_case(arguments.repeats),
        "open_system": _open_system_case(
            arguments.open_system_steps,
            arguments.repeats,
        ),
        "tensor_network": _tensor_network_case(
            arguments.tensor_sites,
            arguments.repeats,
        ),
        "predictive_uq": _predictive_case(
            arguments.draws,
            arguments.width,
            arguments.repeats,
        ),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
