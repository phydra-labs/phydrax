#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from math import isfinite
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key, PyTree

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..linalg import (
    eigen,
    estimate_diagonal,
    FunctionLinearOperator,
    JacobianLinearOperator,
    LinearizationPolicy,
    MaterializationPolicy,
    materialize,
    OperatorProperties,
    prepare_linearization,
    PreparedLinearization,
    PyTreeSpace,
    stochastic_trace,
)


TangentKernelKind = Literal["euclidean", "parameter_metric"]


class PreparedEmpiricalNTK(StrictModule):
    """Reusable finite-width empirical tangent-kernel actions at one point."""

    linearization: PreparedLinearization
    jacobian: JacobianLinearOperator
    kernel: FunctionLinearOperator
    parameter_gram: FunctionLinearOperator | None
    parameter_geometry: Any
    kind: TangentKernelKind = eqx.field(static=True)
    ntk_id: str = eqx.field(static=True)

    def __init__(
        self,
        linearization: PreparedLinearization,
        /,
        *,
        parameter_geometry: Any = None,
        ntk_id: str | None = None,
    ):
        if not isinstance(linearization, PreparedLinearization):
            raise TypeError("linearization must be a PreparedLinearization.")
        jacobian = JacobianLinearOperator(
            linearization,
            operator_id=f"{linearization.linearization_id}:jacobian",
        )
        kind: TangentKernelKind = (
            "euclidean" if parameter_geometry is None else "parameter_metric"
        )
        if parameter_geometry is not None:
            from ..optim._riemannian import ParameterGeometry

            if not isinstance(parameter_geometry, ParameterGeometry):
                raise TypeError("parameter_geometry must be a ParameterGeometry or None.")
            parameter_geometry.validate(linearization.point)

        def inverse_metric(cotangent):
            if parameter_geometry is None:
                return cotangent
            return parameter_geometry.egrad_to_rgrad(
                linearization.point, cotangent
            )

        def kernel_action(cotangent):
            parameter_cotangent = linearization.vjp(cotangent)
            return linearization.jvp(inverse_metric(parameter_cotangent))

        kernel = FunctionLinearOperator(
            kernel_action,
            source=linearization.target,
            target=linearization.target,
            transpose_action=kernel_action,
            properties=OperatorProperties(
                self_adjoint=True,
                positive_semidefinite=True,
                evidence={
                    "self_adjoint": "construction",
                    "positive_semidefinite": "construction",
                },
            ),
            operator_id=f"{linearization.linearization_id}:{kind}-ntk",
            closure_convert=False,
        )
        parameter_gram = None
        if parameter_geometry is None:

            def gram_action(tangent):
                return linearization.vjp(linearization.jvp(tangent))

            parameter_gram = FunctionLinearOperator(
                gram_action,
                source=linearization.source,
                target=linearization.source,
                transpose_action=gram_action,
                properties=OperatorProperties(
                    self_adjoint=True,
                    positive_semidefinite=True,
                    evidence={
                        "self_adjoint": "construction",
                        "positive_semidefinite": "construction",
                    },
                ),
                operator_id=f"{linearization.linearization_id}:parameter-gram",
                closure_convert=False,
            )
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "prepared-empirical-ntk",
                    "linearization": linearization.linearization_id,
                    "kernel_kind": kind,
                    "parameter_space": linearization.source.space_id,
                    "output_space": linearization.target.space_id,
                }
            )
            if ntk_id is None
            else str(ntk_id)
        )
        if not identifier:
            raise ValueError("ntk_id must be non-empty.")
        self.linearization = linearization
        self.jacobian = jacobian
        self.kernel = kernel
        self.parameter_gram = parameter_gram
        self.parameter_geometry = parameter_geometry
        self.kind = kind
        self.ntk_id = identifier

    @property
    def output(self) -> PyTree[Array]:
        return self.linearization.primal

    @property
    def parameter_space(self):
        return self.linearization.source

    @property
    def output_space(self):
        return self.linearization.target

    def jvp(self, tangent: PyTree[Any], /) -> PyTree[Array]:
        return self.linearization.jvp(tangent)

    def vjp(self, cotangent: PyTree[Any], /) -> PyTree[Array]:
        return self.linearization.vjp(cotangent)

    def cross_kernel(self, other: PreparedEmpiricalNTK, /) -> FunctionLinearOperator:
        """Return ``J_self G⁻¹ J_other*`` for a shared parameter point."""
        if not isinstance(other, PreparedEmpiricalNTK):
            raise TypeError("other must be a PreparedEmpiricalNTK.")
        if self.kind != other.kind or not self.parameter_space.compatible(
            other.parameter_space
        ):
            raise ValueError("Cross kernels require one parameter space and metric kind.")
        equal = eqx.tree_equal(self.linearization.point, other.linearization.point)
        if equal is not True and not bool(jax.device_get(equal)):
            raise ValueError("Cross kernels require an identical parameter point.")
        if self.parameter_geometry is not other.parameter_geometry:
            raise ValueError("Metric cross kernels require the same geometry object.")

        def inverse_metric(cotangent):
            if self.parameter_geometry is None:
                return cotangent
            return self.parameter_geometry.egrad_to_rgrad(
                self.linearization.point, cotangent
            )

        def action(cotangent):
            return self.linearization.jvp(
                inverse_metric(other.linearization.vjp(cotangent))
            )

        def transpose_action(cotangent):
            return other.linearization.jvp(
                inverse_metric(self.linearization.vjp(cotangent))
            )

        return FunctionLinearOperator(
            action,
            source=other.output_space,
            target=self.output_space,
            transpose_action=transpose_action,
            operator_id=f"{self.ntk_id}:cross:{other.ntk_id}",
            closure_convert=False,
        )


def prepare_empirical_ntk(
    function: Callable[[PyTree[Any]], PyTree[Any]],
    parameters: PyTree[Any],
    /,
    *,
    parameter_space: Any = None,
    output_space: Any = None,
    linearization: LinearizationPolicy | None = None,
    parameter_geometry: Any = None,
    ntk_id: str | None = None,
) -> PreparedEmpiricalNTK:
    """Prepare matrix-free empirical NTK and parameter-Gram actions."""
    if not callable(function):
        raise TypeError("function must be callable.")
    source = PyTreeSpace(parameters) if parameter_space is None else parameter_space
    prepared = prepare_linearization(
        function,
        parameters,
        source=source,
        target=output_space,
        policy=linearization,
        linearization_id=(None if ntk_id is None else f"{ntk_id}:linearization"),
    )
    return PreparedEmpiricalNTK(
        prepared,
        parameter_geometry=parameter_geometry,
        ntk_id=ntk_id,
    )


class NTKDiagnosticsPolicy(StrictModule):
    """Resource-bounded empirical NTK diagonal, trace, and spectrum policy."""

    dense_max_dimension: int = eqx.field(static=True)
    num_probes: int = eqx.field(static=True)
    eigenvalue_count: int = eqx.field(static=True)
    max_krylov_steps: int = eqx.field(static=True)
    rank_tolerance: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        dense_max_dimension: int = 512,
        num_probes: int = 16,
        eigenvalue_count: int = 8,
        max_krylov_steps: int = 64,
        rank_tolerance: float = 1e-8,
    ):
        integers = tuple(
            int(value)
            for value in (
                dense_max_dimension,
                num_probes,
                eigenvalue_count,
                max_krylov_steps,
            )
        )
        tolerance = float(rank_tolerance)
        if any(value < 1 for value in integers) or not isfinite(tolerance) or tolerance < 0:
            raise ValueError("NTK diagnostic capacities or rank tolerance are invalid.")
        (
            self.dense_max_dimension,
            self.num_probes,
            self.eigenvalue_count,
            self.max_krylov_steps,
        ) = integers
        self.rank_tolerance = tolerance


class NTKDiagnostics(StrictModule):
    diagonal: Array
    diagonal_standard_error: Array
    trace: Array
    trace_standard_error: Array
    trace_square: Array
    largest_eigenvalue: Array
    leading_eigenvalues: Array
    stable_rank: Array
    effective_rank: Array
    numerical_rank: Array
    nullity: Array
    active_condition_number: Array
    finite: Array
    converged: Array
    dense: bool = eqx.field(static=True)
    method: str = eqx.field(static=True)


def _dense_diagnostics(
    prepared: PreparedEmpiricalNTK,
    policy: NTKDiagnosticsPolicy,
) -> NTKDiagnostics:
    matrix = materialize(
        prepared.kernel,
        MaterializationPolicy(
            max_entries=prepared.output_space.size**2,
            max_bytes=2**63 - 1,
        ),
    )
    matrix = 0.5 * (matrix + jnp.conj(matrix.T))
    diagonal = jnp.real(jnp.diag(matrix))
    values = jnp.real(jnp.linalg.eigvalsh(matrix))
    descending = values[::-1]
    count = min(policy.eigenvalue_count, int(values.size))
    leading = descending[:count]
    trace = jnp.sum(values)
    trace_square = jnp.sum(values**2)
    largest = descending[0]
    threshold = policy.rank_tolerance * jnp.maximum(
        jnp.abs(largest),
        jnp.finfo(values.dtype).tiny,
    )
    positive = values > threshold
    rank = jnp.sum(positive, dtype=jnp.int32)
    nullity = jnp.asarray(values.size, dtype=jnp.int32) - rank
    minimum_positive = jnp.min(jnp.where(positive, values, jnp.inf))
    condition = jnp.where(rank > 0, largest / minimum_positive, jnp.inf)
    tiny = jnp.finfo(values.dtype).tiny
    stable_rank = trace / jnp.maximum(largest, tiny)
    effective_rank = trace**2 / jnp.maximum(trace_square, tiny)
    finite = jnp.all(jnp.isfinite(matrix)) & jnp.all(jnp.isfinite(values))
    return NTKDiagnostics(
        diagonal,
        jnp.zeros_like(diagonal),
        trace,
        jnp.asarray(0.0, dtype=trace.dtype),
        trace_square,
        largest,
        leading,
        stable_rank,
        effective_rank,
        rank,
        nullity,
        condition,
        finite,
        finite,
        True,
        "dense-eigh",
    )


def _matrix_free_diagnostics(
    prepared: PreparedEmpiricalNTK,
    policy: NTKDiagnosticsPolicy,
    key: Key[Array, ""],
) -> NTKDiagnostics:
    diagonal_key, trace_key, square_key, eigen_key = jr.split(key, 4)
    diagonal = estimate_diagonal(
        prepared.kernel,
        key=diagonal_key,
        num_probes=policy.num_probes,
    )
    trace = stochastic_trace(
        prepared.kernel,
        key=trace_key,
        num_probes=policy.num_probes,
        max_dimension=1,
    )
    trace_square = stochastic_trace(
        prepared.kernel,
        lambda value: value**2,
        key=square_key,
        num_probes=policy.num_probes,
        max_dimension=min(2, prepared.output_space.size),
    )
    count = min(policy.eigenvalue_count, max(prepared.output_space.size - 1, 1))
    eigen_result = eigen.eigensolve(
        eigen.Eigenproblem(prepared.kernel),
        policy=eigen.EigenSolvePolicy(
            eigen.RestartedLanczos(),
            count=count,
            which="largest-algebraic",
            max_steps=max(policy.max_krylov_steps, 2 * count),
            key=eigen_key,
        ),
    )
    leading = jnp.asarray(eigen_result.eigenvalues).reshape((-1,))
    largest = leading[0]
    tiny = jnp.finfo(jnp.asarray(trace.estimate).real.dtype).tiny
    stable_rank = trace.estimate / jnp.maximum(largest, tiny)
    effective_rank = trace.estimate**2 / jnp.maximum(trace_square.estimate, tiny)
    finite = (
        diagonal.finite
        & trace.finite
        & trace_square.finite
        & jnp.all(jnp.isfinite(leading))
    )
    converged = finite & eigen_result.converged
    return NTKDiagnostics(
        jnp.real(diagonal.estimate),
        jnp.real(diagonal.standard_error),
        jnp.real(trace.estimate),
        jnp.real(trace.standard_error),
        jnp.real(trace_square.estimate),
        jnp.real(largest),
        jnp.real(leading),
        jnp.real(stable_rank),
        jnp.real(effective_rank),
        jnp.asarray(-1, dtype=jnp.int32),
        jnp.asarray(-1, dtype=jnp.int32),
        jnp.asarray(jnp.nan),
        finite,
        converged,
        False,
        "hutchinson-lanczos",
    )


def analyze_ntk(
    prepared: PreparedEmpiricalNTK,
    /,
    *,
    policy: NTKDiagnosticsPolicy | None = None,
    key: Key[Array, ""] | None = None,
) -> NTKDiagnostics:
    """Measure a prepared NTK without silently materializing large kernels."""
    if not isinstance(prepared, PreparedEmpiricalNTK):
        raise TypeError("prepared must be a PreparedEmpiricalNTK.")
    policy_ = NTKDiagnosticsPolicy() if policy is None else policy
    if not isinstance(policy_, NTKDiagnosticsPolicy):
        raise TypeError("policy must be an NTKDiagnosticsPolicy or None.")
    if prepared.output_space.size <= policy_.dense_max_dimension:
        return _dense_diagnostics(prepared, policy_)
    if key is None:
        raise ValueError("Matrix-free NTK diagnostics require a PRNG key.")
    return _matrix_free_diagnostics(prepared, policy_, key)


__all__ = [
    "NTKDiagnostics",
    "NTKDiagnosticsPolicy",
    "PreparedEmpiricalNTK",
    "TangentKernelKind",
    "analyze_ntk",
    "prepare_empirical_ntk",
]
