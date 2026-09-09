#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from benchmarks._io import write_json_atomic
from benchmarks._runtime import (
    capture_environment,
    measure_repeated,
    measure_synchronized,
)
from phydrax import ein


class _MatrixCorrectionOperator(phx.nn.operator.AbstractOperatorModel):
    matrix: jax.Array
    in_size: str = eqx.field(static=True)
    out_size: str = eqx.field(static=True)

    def __init__(self, matrix):
        self.matrix = jnp.asarray(matrix, dtype=jnp.float64)
        self.in_size = "scalar"
        self.out_size = "scalar"

    @property
    def operator_contract(self):
        return phx.nn.operator.operator_architecture_contract("FNO")

    def __call_operator_batch__(self, batch, /, *, key=None):
        del key
        residual = batch.input("residual").values
        assert residual is not None
        return ein.contract("ij,j->i", self.matrix, residual)

    def __call__(self, batch, /, *, key=None):
        return self.__call_operator_batch__(batch, key=key)


def _field_space(name, support_id, vector_space):
    return phx.discretization.DiscreteFieldSpace(
        name,
        support_id,
        phx.discretization.TensorDofLayout(("x",), vector_space.shape),
        vector_space,
        representation="point_value",
        field_space_id=f"quick-{name}-field-space",
    )


def _identity_transfer(source, target, transfer_id):
    return phx.discretization.FieldTransfer(
        source,
        target,
        phx.linalg.DenseLinearOperator(
            jnp.eye(source.vector_space.size, dtype=jnp.float64),
            source=source.vector_space,
            target=target.vector_space,
        ),
        transfer_id=transfer_id,
    )


def _operator_task():
    return phx.nn.operator.OperatorTask(
        "quick-poisson-correction",
        fields=(
            phx.nn.operator.OperatorFieldSpec(
                "residual", role="source", source_name="residual"
            ),
            phx.nn.operator.OperatorFieldSpec(
                "correction", role="target", query_name="query"
            ),
        ),
        queries=(
            phx.nn.operator.OperatorQuerySpec(
                "query",
                geometry_kind="tensor_grid",
                coordinate_components=("x",),
                fixed_geometry=True,
            ),
        ),
        problem=phx.nn.operator.OperatorProblemSpec(
            source_query_relation="coincident",
            query_is_fixed=True,
        ),
    )


def _solve_record(problem, right_hand_side, policy, *, warmup, repeats):
    prepared, preparation_seconds = measure_synchronized(
        lambda: phx.linalg.prepare(problem, policy)
    )
    result, timing = measure_repeated(
        lambda: phx.linalg.solve(prepared, right_hand_side),
        warmup=warmup,
        repeats=repeats,
    )
    residual = right_hand_side - problem.operator.mv(result.value)
    residual_norm = float(
        np.asarray(jnp.sqrt(jnp.real(problem.operator.source.inner(residual, residual))))
    )
    right_hand_side_norm = float(
        np.asarray(
            jnp.sqrt(
                jnp.real(
                    problem.operator.target.inner(
                        right_hand_side,
                        right_hand_side,
                    )
                )
            )
        )
    )
    relative_residual = residual_norm / max(right_hand_side_norm, np.finfo(float).tiny)
    return {
        "successful": bool(np.asarray(result.successful)),
        "status": int(np.asarray(result.status)),
        "iterations": int(np.asarray(result.diagnostics.iterations)),
        "original_residual_norm": residual_norm,
        "original_relative_residual": relative_residual,
        "false_success": bool(np.asarray(result.successful)) and relative_residual > 1e-7,
        "preparation_seconds": preparation_seconds,
        "steady_solve": timing.to_dict(unit="milliseconds"),
        "method": result.provenance.method,
        "backend": result.provenance.backend,
    }


def run_quick(*, size=31, modes=4, warmup=1, repeats=3):
    size_ = int(size)
    modes_ = int(modes)
    if size_ < 5 or not 1 <= modes_ < size_:
        raise ValueError("Quick benchmark requires size >= 5 and 1 <= modes < size.")
    diagonal = 2.0 * jnp.ones((size_,), dtype=jnp.float64)
    matrix = jnp.diag(diagonal)
    matrix = matrix + jnp.diag(-jnp.ones((size_ - 1,), dtype=jnp.float64), 1)
    matrix = matrix + jnp.diag(-jnp.ones((size_ - 1,), dtype=jnp.float64), -1)
    eigenvalues, eigenvectors = jnp.linalg.eigh(matrix)
    coordinate = jnp.linspace(0.0, 1.0, size_)
    axis = phx.nn.operator.OperatorAxis(
        "x",
        coordinate,
        quadrature_weights=jnp.full((size_,), 1.0 / size_, dtype=jnp.float64),
    )
    basis_samples = phx.nn.operator.FunctionSamples(
        values=eigenvectors[:, :modes_],
        axes=(axis,),
    )
    vector_space = phx.linalg.ArraySpace((size_,), dtype=jnp.float64)
    support_id = basis_samples.support_id
    solver_space = _field_space("solver", support_id, vector_space)
    model_residual_space = _field_space("model-residual", support_id, vector_space)
    model_correction_space = _field_space("model-correction", support_id, vector_space)
    basis_transfer = _identity_transfer(
        model_correction_space, solver_space, "quick-basis-transfer"
    )
    learned_subspace, subspace_preparation_seconds = measure_synchronized(
        lambda: phx.nn.operator.prepare_operator_subspace_correction(
            basis_samples,
            model_correction_space,
            solver_space,
            basis_transfer,
            phx.linalg.DenseInversePreconditionerBuilder(),
        )
    )
    identity = phx.linalg.IdentityLinearOperator(vector_space)
    smoother_term = phx.linalg.SubspaceCorrectionTerm(
        identity,
        identity,
        phx.linalg.JacobiPreconditionerBuilder(),
    )
    subspace_builder = phx.linalg.AdditiveSubspaceCorrectionBuilder(
        (smoother_term, learned_subspace.term)
    )

    low_basis = eigenvectors[:, :modes_]
    low_inverse = ein.contract(
        "ik,k,jk->ij", low_basis, 1.0 / eigenvalues[:modes_], low_basis
    )
    jacobi_inverse = 0.5 * jnp.eye(size_, dtype=jnp.float64)
    low_jacobi = ein.contract("ik,jk->ij", low_basis, low_basis) * 0.5
    direct_matrix = jacobi_inverse + low_inverse - low_jacobi
    template = phx.nn.operator.OperatorBatch(
        inputs={
            "residual": phx.nn.operator.FunctionSamples(
                values=jnp.zeros((size_,), dtype=jnp.float64), axes=(axis,)
            )
        },
        queries={"query": phx.nn.operator.FunctionSamples(values=None, axes=(axis,))},
    )
    trained = phx.nn.operator.training.TrainedOperator(
        _MatrixCorrectionOperator(direct_matrix),
        _operator_task(),
        training_evidence=phx.nn.operator.OperatorTrainingEvidence("task_specific"),
        output_field_map={"output": "correction"},
        fixed_query_fingerprints={
            "query": template.query("query").geometry_fingerprint()
        },
        dtype_policy=phx.nn.operator.training.OperatorDTypePolicy(
            parameter_dtype="float64",
            compute_dtype="float64",
            reduction_dtype="float64",
        ),
        artifact_id="quick-deterministic-correction-artifact",
    )
    binding = phx.nn.operator.OperatorCorrectionBinding(
        trained,
        template,
        solver_space,
        model_residual_space,
        model_correction_space,
        _identity_transfer(solver_space, model_residual_space, "quick-residual-transfer"),
        _identity_transfer(
            model_correction_space, solver_space, "quick-correction-transfer"
        ),
        residual_source_name="residual",
        correction_field_name="correction",
        condition_ids=(
            "boundary:homogeneous-dirichlet",
            "operator:one-dimensional-poisson",
        ),
    )
    direct_builder = phx.nn.operator.TrainedOperatorPreconditionerBuilder(
        binding,
        phx.nn.operator.OperatorCorrectionCost(
            preparation_workspace_bytes=int(direct_matrix.nbytes),
            inference_workspace_bytes_per_rhs=int(direct_matrix.nbytes),
        ),
    )
    operator = phx.linalg.DenseLinearOperator(
        matrix,
        source=vector_space,
        target=vector_space,
        properties=phx.linalg.OperatorProperties(
            self_adjoint=True,
            positive_definite=True,
            evidence={
                "self_adjoint": "construction",
                "positive_definite": "construction",
                "positive_semidefinite": "construction",
            },
        ),
    )
    problem = phx.linalg.LinearSystem(operator)
    right_hand_side = jnp.sin(jnp.pi * coordinate) + 0.2 * jnp.sin(
        11.0 * jnp.pi * coordinate
    )

    def policy(builder):
        return phx.linalg.LinearSolvePolicy(
            phx.linalg.FGMRES(restart=min(size_, 32)),
            preconditioning=phx.linalg.PreconditioningPolicy(builder, side="right"),
            differentiation=phx.linalg.DifferentiationPolicy("none"),
        )

    records = {
        "native_jacobi": _solve_record(
            problem,
            right_hand_side,
            policy(phx.linalg.JacobiPreconditionerBuilder()),
            warmup=warmup,
            repeats=repeats,
        ),
        "operator_subspace": _solve_record(
            problem,
            right_hand_side,
            policy(subspace_builder),
            warmup=warmup,
            repeats=repeats,
        ),
        "direct_operator": _solve_record(
            problem,
            right_hand_side,
            policy(direct_builder),
            warmup=warmup,
            repeats=repeats,
        ),
    }
    return {
        "benchmark": "operator-correction-quick",
        "environment": capture_environment().to_dict(),
        "problem": {
            "kind": "one-dimensional-dirichlet-poisson",
            "size": size_,
            "learned_mode_capacity": modes_,
            "right_hand_side": "mixed-low-high-sine",
        },
        "identities": {
            "basis_preparation_id": learned_subspace.preparation_id,
            "basis_transfer_id": learned_subspace.transfer_id,
            "direct_binding_id": binding.binding_id,
            "direct_artifact_id": trained.artifact_id,
        },
        "basis_preparation_seconds": subspace_preparation_seconds,
        "solvers": records,
        "qualification": {
            "false_success_count": sum(
                int(record["false_success"]) for record in records.values()
            ),
            "all_original_residuals_finite": all(
                np.isfinite(record["original_residual_norm"])
                for record in records.values()
            ),
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run the deterministic learned numerical-correction benchmark."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=31)
    parser.add_argument("--modes", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    arguments = parser.parse_args()
    report = run_quick(
        size=arguments.size,
        modes=arguments.modes,
        warmup=arguments.warmup,
        repeats=arguments.repeats,
    )
    path = write_json_atomic(arguments.output, report)
    print(path)


if __name__ == "__main__":
    main()
