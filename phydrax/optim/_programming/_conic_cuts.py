#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax import ein

from ..._cone import AbstractConvexCone
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._cones import NonnegativeCone, ProductCone, ZeroCone
from ._problem import (
    _conic_matrix_mv,
    _conic_matrix_transpose_mv,
    ConicProgram,
)
from ._quadratic import _max_abs


ConicCutSource: TypeAlias = Literal[
    "projection",
    "continuous-dual",
    "fixed-discrete-dual",
    "infeasibility-ray",
]


class ConicCut(StrictModule, NonTrainableState):
    """One affine inequality derived from a dual-cone direction."""

    row: Array
    rhs: Array
    dual: Array
    source_primal: Array | None
    source_kind: ConicCutSource = eqx.field(static=True)
    cone_block: int = eqx.field(static=True)
    cone_id: str = eqx.field(static=True)
    structure_id: str = eqx.field(static=True)
    binding_id: str = eqx.field(static=True)
    cut_id: str = eqx.field(static=True)


class ConicCutAudit(StrictModule, NonTrainableState):
    dual_residual: Array
    row_residual: Array
    rhs_residual: Array
    source_violation: Array
    finite: Array
    global_valid: Array
    source_violated: Array
    cut_id: str = eqx.field(static=True)


class ConicSeparationResult(StrictModule, NonTrainableState):
    cuts: tuple[ConicCut, ...]
    audits: tuple[ConicCutAudit, ...]
    cone_residual: Array
    violated_blocks: Array
    valid: Array
    binding_id: str = eqx.field(static=True)


def _cone_blocks(
    cone: AbstractConvexCone, /
) -> tuple[tuple[int, slice, AbstractConvexCone], ...]:
    if isinstance(cone, ProductCone):
        return tuple(
            (index, block, child)
            for index, (block, child) in enumerate(
                zip(cone.slices, cone.cones, strict=True)
            )
        )
    return ((0, slice(0, cone.dimension), cone),)


def conic_cut_from_dual(
    program: ConicProgram,
    dual: ArrayLike,
    /,
    *,
    source_kind: ConicCutSource,
    cone_block: int,
    cone_id: str,
    binding_id: str,
    source_primal: ArrayLike | None = None,
) -> ConicCut:
    if not isinstance(program, ConicProgram):
        raise TypeError("program must be a ConicProgram.")
    dual_ = jnp.asarray(dual, dtype=program.linear.dtype)
    if dual_.shape != (program.num_constraints,):
        raise ValueError(f"dual must have shape ({program.num_constraints},).")
    source_ = (
        None
        if source_primal is None
        else jnp.asarray(source_primal, dtype=program.linear.dtype)
    )
    if source_ is not None and source_.shape != (program.num_variables,):
        raise ValueError(f"source_primal must have shape ({program.num_variables},).")
    row = _conic_matrix_transpose_mv(program.constraint_matrix, dual_)
    rhs = ein.contract("i,i->", program.constraint_rhs, dual_)
    scale = jnp.maximum(
        1.0,
        jnp.maximum(_max_abs(row), jnp.abs(rhs)),
    )
    row = row / scale
    rhs = rhs / scale
    dual_ = dual_ / scale
    identifier = canonical_fingerprint(
        {
            "kind": "conic-outer-cut",
            "structure": program.structure_id,
            "binding": str(binding_id),
            "source_kind": source_kind,
            "cone_block": int(cone_block),
            "cone": str(cone_id),
            "geometry": array_tree_fingerprint((row, rhs)),
        }
    )
    return ConicCut(
        row,
        rhs,
        dual_,
        source_,
        source_kind,
        int(cone_block),
        str(cone_id),
        program.structure_id,
        str(binding_id),
        identifier,
    )


def audit_conic_cut(
    program: ConicProgram,
    cut: ConicCut,
    /,
    *,
    tolerance: float,
) -> ConicCutAudit:
    if not isinstance(program, ConicProgram):
        raise TypeError("program must be a ConicProgram.")
    if not isinstance(cut, ConicCut):
        raise TypeError("cut must be a ConicCut.")
    tolerance_ = float(tolerance)
    if not np.isfinite(tolerance_) or tolerance_ < 0.0:
        raise ValueError("tolerance must be finite and nonnegative.")
    if cut.structure_id != program.structure_id:
        raise ValueError("Cut structure does not match the conic program.")
    expected_row = _conic_matrix_transpose_mv(
        program.constraint_matrix,
        cut.dual,
    )
    expected_rhs = ein.contract("i,i->", program.constraint_rhs, cut.dual)
    row_residual = _max_abs(cut.row - expected_row)
    rhs_residual = jnp.abs(cut.rhs - expected_rhs)
    dual_residual = jnp.asarray(program.cone.dual_residual(cut.dual))
    if cut.source_primal is None:
        source_violation = jnp.asarray(jnp.nan, dtype=cut.row.dtype)
        source_violated = jnp.asarray(False)
        source_finite = jnp.asarray(True)
    else:
        source_violation = ein.contract("i,i->", cut.row, cut.source_primal) - cut.rhs
        source_violated = source_violation > tolerance_
        source_finite = jnp.all(jnp.isfinite(cut.source_primal))
    finite = (
        jnp.all(jnp.isfinite(cut.row))
        & jnp.isfinite(cut.rhs)
        & jnp.all(jnp.isfinite(cut.dual))
        & source_finite
    )
    scale = jnp.maximum(1.0, jnp.maximum(_max_abs(cut.row), jnp.abs(cut.rhs)))
    global_valid = (
        finite
        & (dual_residual <= tolerance_)
        & (row_residual <= tolerance_ * scale)
        & (rhs_residual <= tolerance_ * scale)
    )
    return ConicCutAudit(
        dual_residual,
        row_residual,
        rhs_residual,
        source_violation,
        finite,
        global_valid,
        source_violated,
        cut.cut_id,
    )


def separate_conic_point(
    program: ConicProgram,
    primal: ArrayLike,
    /,
    *,
    tolerance: float,
    binding_id: str,
) -> ConicSeparationResult:
    if not isinstance(program, ConicProgram):
        raise TypeError("program must be a ConicProgram.")
    primal_ = jnp.asarray(primal, dtype=program.linear.dtype)
    if primal_.shape != (program.num_variables,):
        raise ValueError(f"primal must have shape ({program.num_variables},).")
    tolerance_ = float(tolerance)
    if not np.isfinite(tolerance_) or tolerance_ < 0.0:
        raise ValueError("tolerance must be finite and nonnegative.")
    slack = program.constraint_rhs - _conic_matrix_mv(
        program.constraint_matrix,
        primal_,
    )
    cuts: list[ConicCut] = []
    audits: list[ConicCutAudit] = []
    violated = 0
    valid = bool(np.asarray(jnp.all(jnp.isfinite(slack))))
    for block_index, block, cone in _cone_blocks(program.cone):
        block_slack = slack[block]
        residual = float(np.asarray(cone.residual(block_slack)))
        if not np.isfinite(residual):
            valid = False
            continue
        if residual <= tolerance_:
            continue
        violated += 1
        block_dual = cone.project_dual(-block_slack)
        dual = jnp.zeros_like(slack).at[block].set(block_dual)
        cut = conic_cut_from_dual(
            program,
            dual,
            source_kind="projection",
            cone_block=block_index,
            cone_id=cone.cone_id,
            binding_id=binding_id,
            source_primal=primal_,
        )
        audit = audit_conic_cut(program, cut, tolerance=tolerance_)
        cut_valid = bool(np.asarray(audit.global_valid & audit.source_violated))
        valid &= cut_valid
        if cut_valid:
            cuts.append(cut)
            audits.append(audit)
    return ConicSeparationResult(
        tuple(cuts),
        tuple(audits),
        jnp.asarray(program.cone.residual(slack)),
        jnp.asarray(violated, dtype=jnp.int32),
        jnp.asarray(valid),
        str(binding_id),
    )


def polyhedral_conic_rows(
    program: ConicProgram,
    /,
) -> tuple[Array, Array, Array, Array]:
    """Return exact zero-cone equalities and orthant inequalities."""
    if not isinstance(program, ConicProgram):
        raise TypeError("program must be a ConicProgram.")
    dtype = program.linear.dtype
    equalities: list[Array] = []
    equality_rhs: list[Array] = []
    inequalities: list[Array] = []
    inequality_rhs: list[Array] = []
    for _, block, cone in _cone_blocks(program.cone):
        if not isinstance(cone, (ZeroCone, NonnegativeCone)):
            continue
        start = 0 if block.start is None else int(block.start)
        stop = program.num_constraints if block.stop is None else int(block.stop)
        for coordinate in range(start, stop):
            basis = (
                jnp.zeros((program.num_constraints,), dtype=dtype).at[coordinate].set(1.0)
            )
            row = _conic_matrix_transpose_mv(
                program.constraint_matrix,
                basis,
            )
            rhs = program.constraint_rhs[coordinate]
            if isinstance(cone, ZeroCone):
                equalities.append(row)
                equality_rhs.append(rhs)
            else:
                inequalities.append(row)
                inequality_rhs.append(rhs)
    equality_matrix = (
        jnp.stack(equalities)
        if equalities
        else jnp.empty((0, program.num_variables), dtype=dtype)
    )
    equality_vector = (
        jnp.stack(equality_rhs) if equality_rhs else jnp.empty((0,), dtype=dtype)
    )
    inequality_matrix = (
        jnp.stack(inequalities)
        if inequalities
        else jnp.empty((0, program.num_variables), dtype=dtype)
    )
    inequality_vector = (
        jnp.stack(inequality_rhs) if inequality_rhs else jnp.empty((0,), dtype=dtype)
    )
    return (
        equality_matrix,
        equality_vector,
        inequality_matrix,
        inequality_vector,
    )


__all__ = [
    "ConicCut",
    "ConicCutAudit",
    "ConicCutSource",
    "ConicSeparationResult",
    "audit_conic_cut",
    "conic_cut_from_dual",
    "polyhedral_conic_rows",
    "separate_conic_point",
]
