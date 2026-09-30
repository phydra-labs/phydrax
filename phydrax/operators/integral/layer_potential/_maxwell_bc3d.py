# Copyright © 2026 PHYDRA, Inc. All rights reserved.

from __future__ import annotations

from typing import final

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....discretization.bem._bc_dual import BuffaChristiansenDualSpace3D
from ....discretization.bem._rwg import rwg_gram_entries
from ....linalg import (
    DenseLinearOperator,
    DualSpace,
    FailurePolicy,
    FunctionLinearOperator,
    LinearSolvePolicy,
    LinearSystem,
    OperatorProperties,
    PCG,
    prepare,
    PreparedLinearSolve,
    solve,
    TolerancePolicy,
)
from ._maxwell3d import (
    MaxwellEFIEAssemblyReport3D,
    MaxwellEFIEPolicy3D,
    prepare_maxwell_efie_3d,
    PreparedMaxwellEFIE3D,
)
from ._maxwell_magnetic3d import central_magnetic_matrix


@final
class PreparedMaxwellBCEFIE3D(StrictModule, NonTrainableState):
    """BC electric single layer and its genuine exterior magnetic trace.

    Weak matrices are B^T V_refined B and B^T (G_refined/2+K_refined) B,
    not primal matrices relabeled as BC. The magnetic coefficient trace uses
    the exact BC Gram solve. Singular electric quadrature, centroid products,
    electrical-size/resource refusals and noncertification remain visible.
    """

    current_space: BuffaChristiansenDualSpace3D
    operator: DenseLinearOperator
    magnetic_operator: DenseLinearOperator
    gram_operator: DenseLinearOperator
    magnetic_trace: FunctionLinearOperator
    gram_solve: PreparedLinearSolve
    refined_efie: PreparedMaxwellEFIE3D
    assembly_report: MaxwellEFIEAssemblyReport3D
    condition_number: Array
    prepared_id: str = eqx.field(static=True)


def prepare_maxwell_bc_efie_3d(
    current_space: BuffaChristiansenDualSpace3D,
    wavenumber: ArrayLike,
    /,
    *,
    wave_impedance: ArrayLike = 1.0,
    policy: MaxwellEFIEPolicy3D | None = None,
) -> PreparedMaxwellBCEFIE3D:
    if not isinstance(current_space, BuffaChristiansenDualSpace3D):
        raise TypeError("current_space must be BuffaChristiansenDualSpace3D.")
    selected = MaxwellEFIEPolicy3D() if policy is None else policy
    refined = prepare_maxwell_efie_3d(
        current_space.barycentric_rwg,
        wavenumber,
        wave_impedance=wave_impedance,
        policy=selected,
    )
    transform = current_space.barycentric_transform.astype(refined.operator.matrix.dtype)
    electric = transform.T @ refined.operator.matrix @ transform
    condition = jnp.linalg.cond(electric)
    if not bool(jnp.all(jnp.isfinite(electric)) & jnp.isfinite(condition)):
        raise ValueError(
            "BC Maxwell boundary assembly or condition evidence is nonfinite."
        )
    if float(condition) > selected.max_condition_number:
        raise ValueError("BC Maxwell boundary condition number exceeds its policy.")
    fine = current_space.barycentric_rwg
    rows, columns, values = rwg_gram_entries(fine.surface)
    # Preparation applies the sparse refined Gram to B without materializing it.
    host_transform = np.asarray(current_space.barycentric_transform)
    applied = np.zeros_like(host_transform)
    np.add.at(applied, rows, values[:, None] * host_transform[columns])
    gram = jnp.asarray(host_transform.T @ applied, dtype=electric.dtype)
    central = jnp.asarray(
        central_magnetic_matrix(fine, float(refined.wavenumber)), dtype=electric.dtype
    )
    magnetic = 0.5 * gram + transform.T @ central @ transform
    if not bool(jnp.all(jnp.isfinite(magnetic))):
        raise ValueError("BC Maxwell magnetic boundary action is nonfinite.")
    identifier = canonical_fingerprint(
        {
            "kind": "maxwell-bc-boundary-3d",
            "bc": current_space.space_id,
            "refined_efie": refined.prepared_id,
        }
    )
    space = current_space.vector_space
    operator = DenseLinearOperator(
        electric,
        source=space,
        target=DualSpace(space),
        operator_id=f"{identifier}:electric",
    )
    magnetic_operator = DenseLinearOperator(
        magnetic,
        source=space,
        target=DualSpace(space),
        operator_id=f"{identifier}:magnetic",
    )
    gram_operator = DenseLinearOperator(
        gram,
        source=space,
        target=space,
        properties=OperatorProperties(
            self_adjoint=True,
            positive_definite=True,
            evidence={
                "self_adjoint": "construction",
                "positive_definite": "construction",
            },
        ),
        operator_id=f"{identifier}:gram",
    )
    gram_tolerance = max(1e-12, 32.0 * np.finfo(np.dtype(gram.real.dtype)).eps)
    gram_solve = prepare(
        LinearSystem(gram_operator, problem_id=f"{identifier}:gram-system"),
        LinearSolvePolicy(
            PCG(),
            tolerance=TolerancePolicy(
                relative=gram_tolerance, max_steps=max(1, 4 * space.size)
            ),
            failure=FailurePolicy("status"),
        ),
    )

    def inverse_gram(value: Array) -> Array:
        result = solve(gram_solve, value)
        accepted = result.successful & result.diagnostics.finite
        return jnp.where(accepted, result.value, jnp.full_like(result.value, jnp.nan))

    def magnetic_action(value: Array) -> Array:
        return inverse_gram(magnetic @ value)

    def magnetic_transpose(value: Array) -> Array:
        return magnetic.T @ inverse_gram(value)

    magnetic_trace = FunctionLinearOperator(
        magnetic_action,
        source=space,
        target=space,
        transpose_action=magnetic_transpose,
        operator_id=f"{identifier}:magnetic-trace",
    )
    return PreparedMaxwellBCEFIE3D(
        current_space,
        operator,
        magnetic_operator,
        gram_operator,
        magnetic_trace,
        gram_solve,
        refined,
        refined.assembly_report,
        condition,
        identifier,
    )


__all__ = ["PreparedMaxwellBCEFIE3D", "prepare_maxwell_bc_efie_3d"]
