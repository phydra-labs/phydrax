#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.fem import (
    PreparedMaxwellMortarInterfaceTrace3D,
    PreparedScalarMortarInterfaceTrace3D,
)
from ..linalg import (
    AbstractLinearOperator,
    ArraySpace,
    coordinate_space,
    DenseLinearOperator,
    DualSpace,
    FailurePolicy,
    FGMRES,
    FunctionLinearOperator,
    LinearSolvePolicy,
    LinearSolveResult,
    LinearSystem,
    prepare,
    PreparedLinearSolve,
    solve,
    TolerancePolicy,
)
from ..linalg._spaces import _coordinate_dtype
from ..operators.integral.layer_potential._maxwell_bc3d import PreparedMaxwellBCEFIE3D
from ..operators.integral.layer_potential._periodic_maxwell_boundary3d import (
    PreparedPeriodicMaxwellBoundary3D,
)


class CoupledFEMBEMResult3D(StrictModule):
    interior: Array
    boundary: Array
    linear_result: LinearSolveResult
    interface_residual: Array
    interior_residual: Array
    relative_block_residual: Array
    successful: Array
    evidence_ids: tuple[str, ...] = eqx.field(static=True)
    continuum_certified: bool = eqx.field(static=True)


class PreparedNonmatchingFEMBEM3D(StrictModule, NonTrainableState):
    """Prepared matrix-free weak block [A, N*; T, -V].

    N is the qualified magnetic conormal, not an inferred copy of T. Both
    equations, including their loads and boundary action, are reported at solve.
    """

    operator: AbstractLinearOperator
    prepared_linear: PreparedLinearSolve
    interior_operator: AbstractLinearOperator
    boundary_operator: AbstractLinearOperator
    coupling: AbstractLinearOperator
    conormal: AbstractLinearOperator
    interior_size: int = eqx.field(static=True)
    boundary_size: int = eqx.field(static=True)
    family: str = eqx.field(static=True)
    evidence_ids: tuple[str, ...] = eqx.field(static=True)
    residual_tolerance: float = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    boundary_provider: PreparedMaxwellBCEFIE3D | PreparedPeriodicMaxwellBoundary3D | None

    def solve(
        self, interior_load: ArrayLike, boundary_load: ArrayLike, /
    ) -> CoupledFEMBEMResult3D:
        dtype = _coordinate_dtype(self.operator.source)
        interior = jnp.asarray(interior_load, dtype=dtype)
        boundary = jnp.asarray(boundary_load, dtype=dtype)
        if interior.shape != (self.interior_size,) or boundary.shape != (
            self.boundary_size,
        ):
            raise ValueError("Coupled FEM-BEM loads have incompatible shapes.")
        rhs = jnp.concatenate((interior, boundary))
        result = solve(self.prepared_linear, rhs)
        u = result.value[: self.interior_size]
        q = result.value[self.interior_size :]
        residual = self.operator.mv(result.value) - rhs
        lower = residual[self.interior_size :]
        upper = residual[: self.interior_size]
        scale = jnp.maximum(jnp.linalg.norm(rhs), jnp.finfo(rhs.real.dtype).tiny)
        relative = jnp.linalg.norm(residual) / scale
        accepted = (
            result.successful
            & result.diagnostics.finite
            & jnp.isfinite(relative)
            & (relative <= self.residual_tolerance)
        )
        return CoupledFEMBEMResult3D(
            u, q, result, lower, upper, relative, accepted, self.evidence_ids, False
        )


def _coordinate_action(operator: AbstractLinearOperator, vector: Array, /) -> Array:
    return operator.target.flatten(operator.mv(operator.source.unflatten(vector)))


def _coordinate_transpose(operator: AbstractLinearOperator, vector: Array, /) -> Array:
    return operator.source.flatten(
        operator.transpose_mv(operator.target.unflatten(vector))
    )


def _prepare(
    interior_operator: AbstractLinearOperator,
    boundary_operator: AbstractLinearOperator,
    trace: AbstractLinearOperator,
    conormal: AbstractLinearOperator,
    family: str,
    evidence_ids: tuple[str, ...],
    policy: LinearSolvePolicy | None,
    residual_tolerance: float,
    boundary_provider: PreparedMaxwellBCEFIE3D
    | PreparedPeriodicMaxwellBoundary3D
    | None = None,
) -> PreparedNonmatchingFEMBEM3D:
    n, m = interior_operator.source.size, boundary_operator.source.size
    if (
        interior_operator.target.size != n
        or boundary_operator.target.size != m
        or trace.source.size != n
        or trace.target.size != m
        or conormal.source.size != n
        or conormal.target.size != m
    ):
        raise ValueError("Coupled FEM-BEM operator shapes are incompatible.")
    if not np.isfinite(residual_tolerance) or residual_tolerance <= 0.0:
        raise ValueError("residual_tolerance must be finite and positive.")
    operators = (interior_operator, boundary_operator, trace, conormal)
    if any(not item.capabilities.transpose for item in operators):
        raise ValueError("Coupled FEM-BEM operators require coordinate transposes.")
    dtype = np.dtype(
        jnp.result_type(*(_coordinate_dtype(item.source) for item in operators))
    )
    if any(
        _coordinate_dtype(item.source) != dtype or _coordinate_dtype(item.target) != dtype
        for item in operators
    ):
        raise TypeError("Coupled FEM-BEM operators must share one coordinate dtype.")
    space = ArraySpace((n + m,), dtype=dtype)

    def action(value: Array) -> Array:
        u, q = value[:n], value[n:]
        upper = _coordinate_action(interior_operator, u) + jnp.conj(
            _coordinate_transpose(conormal, jnp.conj(q))
        )
        lower = _coordinate_action(trace, u) - _coordinate_action(boundary_operator, q)
        return jnp.concatenate((upper, lower))

    def transpose_action(value: Array) -> Array:
        u, q = value[:n], value[n:]
        upper = _coordinate_transpose(interior_operator, u) + _coordinate_transpose(
            trace, q
        )
        lower = jnp.conj(
            _coordinate_action(conormal, jnp.conj(u))
        ) - _coordinate_transpose(boundary_operator, q)
        return jnp.concatenate((upper, lower))

    operator_id = canonical_fingerprint(
        {
            "kind": "coupled-fem-bem-3d",
            "family": family,
            "operators": tuple(item.operator_id for item in operators),
            "evidence": evidence_ids,
        }
    )
    operator = FunctionLinearOperator(
        action,
        source=space,
        target=space,
        transpose_action=transpose_action,
        operator_id=operator_id,
    )
    selected = (
        LinearSolvePolicy(
            FGMRES(restart=min(n + m, 50)),
            tolerance=TolerancePolicy(relative=residual_tolerance * 0.1, max_steps=1000),
            failure=FailurePolicy("status"),
        )
        if policy is None
        else policy
    )
    prepared_linear = prepare(LinearSystem(operator, problem_id=operator_id), selected)
    return PreparedNonmatchingFEMBEM3D(
        operator,
        prepared_linear,
        interior_operator,
        boundary_operator,
        trace,
        conormal,
        n,
        m,
        family,
        evidence_ids,
        residual_tolerance,
        canonical_fingerprint(
            {"kind": "prepared-coupled-fem-bem-3d", "operator": operator_id}
        ),
        boundary_provider,
    )


def prepare_scalar_nonmatching_fem_bem_3d(
    interior_matrix: ArrayLike,
    boundary_matrix: ArrayLike,
    mortar: PreparedScalarMortarInterfaceTrace3D,
    /,
    *,
    maximum_dense_entries: int = 4_000_000,
) -> PreparedNonmatchingFEMBEM3D:
    if not isinstance(mortar, PreparedScalarMortarInterfaceTrace3D):
        raise TypeError("mortar must be PreparedScalarMortarInterfaceTrace3D.")
    interior, boundary = jnp.asarray(interior_matrix), jnp.asarray(boundary_matrix)
    if interior.size + boundary.size > maximum_dense_entries:
        raise ValueError("Caller block storage exceeds maximum_dense_entries.")
    return _prepare(
        DenseLinearOperator(interior),
        DenseLinearOperator(boundary),
        mortar.trace,
        mortar.trace,
        "scalar",
        (mortar.evidence.evidence_id,),
        None,
        1e-5,
    )


def prepare_maxwell_fem_bem_3d(
    interior_operator: AbstractLinearOperator,
    boundary: PreparedMaxwellBCEFIE3D | PreparedPeriodicMaxwellBoundary3D,
    mortar: PreparedMaxwellMortarInterfaceTrace3D,
    /,
    *,
    policy: LinearSolvePolicy | None = None,
    residual_tolerance: float = 1e-5,
) -> PreparedNonmatchingFEMBEM3D:
    """Solve caller-qualified nonmatching Maxwell blocks with the actual conormal.

    The periodic alternative is a caller-built mortar/block route over bounded
    images, not automatic periodic volume-to-surface matching.
    """
    if not isinstance(interior_operator, AbstractLinearOperator):
        raise TypeError("interior_operator must be an AbstractLinearOperator.")
    if not isinstance(
        boundary, (PreparedMaxwellBCEFIE3D, PreparedPeriodicMaxwellBoundary3D)
    ):
        raise TypeError("boundary must be a prepared genuine Maxwell boundary operator.")
    if not isinstance(mortar, PreparedMaxwellMortarInterfaceTrace3D):
        raise TypeError("mortar must be PreparedMaxwellMortarInterfaceTrace3D.")
    volume = mortar.volume_complex.hilbert_complex().space(1)
    if not (
        interior_operator.source.compatible(volume)
        or interior_operator.source.compatible(coordinate_space(volume))
    ):
        raise ValueError(
            "Interior operator must act on the mortar's declared FE edge space."
        )
    if not mortar.tangential_trace.target.compatible(DualSpace(boundary.operator.source)):
        raise ValueError(
            "Mortar and boundary operators must share the declared boundary coefficient space."
        )
    if isinstance(boundary, PreparedMaxwellBCEFIE3D):
        if not bool(boundary.assembly_report.discrete_accuracy_supported):
            raise ValueError(
                "Maxwell boundary quadrature is outside its supported envelope."
            )
        evidence = boundary.assembly_report.report_id
    else:
        evidence = boundary.evidence.evidence_id
    return _prepare(
        interior_operator,
        boundary.operator,
        mortar.tangential_trace,
        mortar.magnetic_conormal,
        "maxwell",
        (mortar.evidence.evidence_id, evidence),
        policy,
        residual_tolerance,
        boundary,
    )


__all__ = [
    "CoupledFEMBEMResult3D",
    "PreparedNonmatchingFEMBEM3D",
    "prepare_maxwell_fem_bem_3d",
    "prepare_scalar_nonmatching_fem_bem_3d",
]
