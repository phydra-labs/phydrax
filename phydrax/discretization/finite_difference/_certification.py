#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import factorial
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._precision import complex_precision_dtype, real_precision_dtype_name
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import DiagonalPairing, EuclideanPairing
from ._stencil import BoundaryStencilSet


FDEvidenceKind: TypeAlias = Literal["analytic", "algebraic", "numerical", "unknown"]


class FDConsistencyReport(StrictModule, NonTrainableState):
    """Polynomial consistency and conditioning evidence for one stencil bank."""

    derivative_order: int = eqx.field(static=True)
    requested_accuracy_order: int = eqx.field(static=True)
    minimum_accuracy_order: int = eqx.field(static=True)
    minimum_valid_width: int = eqx.field(static=True)
    maximum_moment_residual: float = eqx.field(static=True)
    maximum_condition_estimate: float = eqx.field(static=True)
    failed_rows: tuple[int, ...] = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    evidence: FDEvidenceKind = eqx.field(static=True)
    passed: bool = eqx.field(static=True)
    report_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        derivative_order: int,
        requested_accuracy_order: int,
        minimum_accuracy_order: int,
        minimum_valid_width: int,
        maximum_moment_residual: float,
        maximum_condition_estimate: float,
        failed_rows: tuple[int, ...],
        tolerance: float,
        stencil_id: str,
    ):
        self.derivative_order = int(derivative_order)
        self.requested_accuracy_order = int(requested_accuracy_order)
        self.minimum_accuracy_order = int(minimum_accuracy_order)
        self.minimum_valid_width = int(minimum_valid_width)
        self.maximum_moment_residual = float(maximum_moment_residual)
        self.maximum_condition_estimate = float(maximum_condition_estimate)
        self.failed_rows = tuple(int(row) for row in failed_rows)
        self.tolerance = float(tolerance)
        self.evidence = "algebraic"
        self.passed = not self.failed_rows
        self.report_id = canonical_fingerprint(
            {
                "kind": "fd-consistency-report",
                "stencil": stencil_id,
                "derivative_order": self.derivative_order,
                "requested_accuracy_order": self.requested_accuracy_order,
                "minimum_accuracy_order": self.minimum_accuracy_order,
                "minimum_valid_width": self.minimum_valid_width,
                "maximum_moment_residual": self.maximum_moment_residual,
                "maximum_condition_estimate": self.maximum_condition_estimate,
                "failed_rows": list(self.failed_rows),
                "tolerance": self.tolerance,
            }
        )


class FDAdjointReport(StrictModule, NonTrainableState):
    """Coordinate-transpose and pairing-adjoint probe evidence."""

    coordinate_transpose_residual: float = eqx.field(static=True)
    pairing_adjoint_residual: float = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    evidence: FDEvidenceKind = eqx.field(static=True)
    passed: bool = eqx.field(static=True)
    report_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        coordinate_transpose_residual: float,
        pairing_adjoint_residual: float,
        tolerance: float,
        operator_id: str,
    ):
        coordinate = float(coordinate_transpose_residual)
        pairing = float(pairing_adjoint_residual)
        tolerance_ = float(tolerance)
        self.coordinate_transpose_residual = coordinate
        self.pairing_adjoint_residual = pairing
        self.tolerance = tolerance_
        self.evidence = "numerical"
        self.passed = max(coordinate, pairing) <= tolerance_
        self.report_id = canonical_fingerprint(
            {
                "kind": "fd-adjoint-report",
                "operator": operator_id,
                "coordinate_transpose_residual": coordinate,
                "pairing_adjoint_residual": pairing,
                "tolerance": tolerance_,
            }
        )


class FDConservationReport(StrictModule, NonTrainableState):
    """Constant preservation and applicable global flux-balance evidence."""

    constant_state_residual: float = eqx.field(static=True)
    global_balance_residual: float | None = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    evidence: FDEvidenceKind = eqx.field(static=True)
    conservative: bool | None = eqx.field(static=True)
    report_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        constant_state_residual: float,
        global_balance_residual: float | None,
        tolerance: float,
        operator_id: str,
    ):
        constant = float(constant_state_residual)
        balance = (
            None if global_balance_residual is None else float(global_balance_residual)
        )
        tolerance_ = float(tolerance)
        self.constant_state_residual = constant
        self.global_balance_residual = balance
        self.tolerance = tolerance_
        self.evidence = "numerical"
        self.conservative = (
            None if balance is None else max(constant, balance) <= tolerance_
        )
        self.report_id = canonical_fingerprint(
            {
                "kind": "fd-conservation-report",
                "operator": operator_id,
                "constant_state_residual": constant,
                "global_balance_residual": balance,
                "tolerance": tolerance_,
            }
        )


class FDStabilityReport(StrictModule, NonTrainableState):
    """Explicit evidence and assumptions for one discrete stability claim."""

    property_name: str = eqx.field(static=True)
    residual: float | None = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    assumptions: tuple[str, ...] = eqx.field(static=True)
    evidence: FDEvidenceKind = eqx.field(static=True)
    passed: bool | None = eqx.field(static=True)
    report_id: str = eqx.field(static=True)

    def __init__(
        self,
        property_name: str,
        /,
        *,
        residual: float | None,
        tolerance: float,
        assumptions: tuple[str, ...],
        evidence: FDEvidenceKind,
        subject_id: str,
    ):
        name = str(property_name)
        if not name or evidence not in ("analytic", "algebraic", "numerical", "unknown"):
            raise ValueError("Stability property name/evidence is invalid.")
        residual_ = None if residual is None else float(residual)
        tolerance_ = float(tolerance)
        if not np.isfinite(tolerance_) or tolerance_ <= 0.0:
            raise ValueError("Stability tolerance must be finite and positive.")
        if residual_ is not None and not np.isfinite(residual_):
            raise ValueError("Stability residual must be finite when supplied.")
        if evidence == "unknown" and residual_ is not None:
            raise ValueError("Unknown evidence cannot carry a certified residual.")
        self.property_name = name
        self.residual = residual_
        self.tolerance = tolerance_
        self.assumptions = tuple(str(value) for value in assumptions)
        self.evidence = evidence
        self.passed = None if residual_ is None else residual_ <= tolerance_
        self.report_id = canonical_fingerprint(
            {
                "kind": "fd-stability-report",
                "subject": subject_id,
                "property_name": name,
                "residual": residual_,
                "tolerance": tolerance_,
                "assumptions": list(self.assumptions),
                "evidence": evidence,
            }
        )


def certify_stencil_consistency(
    stencil_set: BoundaryStencilSet,
    /,
    *,
    tolerance: float = 1e-9,
) -> FDConsistencyReport:
    """Aggregate row-level moment evidence without assembling an operator matrix."""
    if not isinstance(stencil_set, BoundaryStencilSet):
        raise TypeError("stencil_set must be a BoundaryStencilSet.")
    tolerance_ = float(tolerance)
    if not np.isfinite(tolerance_) or tolerance_ <= 0.0:
        raise ValueError("Consistency tolerance must be finite and positive.")
    plans = {plan.plan_id: plan for plan in stencil_set.stencil.coefficient_plans}
    failed_rows = []
    maximum_residual = 0.0
    maximum_condition = 0.0
    minimum_width = min(report.valid_width for report in stencil_set.stencil.row_reports)
    for row, report in enumerate(stencil_set.stencil.row_reports):
        plan = plans[report.coefficient_plan_id]
        required_degree = report.derivative_order + report.achieved_accuracy_order - 1
        residual = float(
            np.max(np.abs(np.asarray(plan.moment_residuals)[: required_degree + 1]))
        )
        scale = max(1.0, float(factorial(report.derivative_order)))
        maximum_residual = max(maximum_residual, residual / scale)
        maximum_condition = max(maximum_condition, plan.condition_estimate)
        if residual > tolerance_ * scale:
            failed_rows.append(row)
    request = stencil_set.stencil.request
    return FDConsistencyReport(
        derivative_order=request.derivative_order,
        requested_accuracy_order=request.accuracy_order,
        minimum_accuracy_order=stencil_set.minimum_accuracy_order,
        minimum_valid_width=minimum_width,
        maximum_moment_residual=maximum_residual,
        maximum_condition_estimate=maximum_condition,
        failed_rows=tuple(failed_rows),
        tolerance=tolerance_,
        stencil_id=stencil_set.stencil.stencil_id,
    )


def _certification_value(value: Array, dtype: Any | None, /) -> Array:
    array = jnp.asarray(value)
    if dtype is None:
        return array
    real_dtype = real_precision_dtype_name(dtype)
    target = (
        complex_precision_dtype(real_dtype)
        if jnp.issubdtype(array.dtype, jnp.complexfloating)
        else real_dtype
    )
    return array.astype(target)


def _pairing_inner(
    space: Any,
    left: Array,
    right: Array,
    /,
    *,
    certification_dtype: Any | None,
) -> Array:
    left_ = _certification_value(left, certification_dtype)
    right_ = _certification_value(right, certification_dtype)
    pairing = space.pairing
    if isinstance(pairing, DiagonalPairing):
        weights = _certification_value(pairing.weights, certification_dtype)
        return jnp.sum(jnp.conj(left_) * weights * right_)
    if isinstance(pairing, EuclideanPairing):
        return jnp.sum(jnp.conj(left_) * right_)
    raise ValueError("FD certification requires Euclidean or diagonal pairings.")


def certify_operator_adjoint(
    operator: Any,
    /,
    *,
    tolerance: float = 1e-10,
    certification_dtype: Any | None = None,
) -> FDAdjointReport:
    """Probe transpose and weighted-adjoint identities matrix-free."""
    tolerance_ = float(tolerance)
    if not np.isfinite(tolerance_) or tolerance_ <= 0.0:
        raise ValueError("Adjoint tolerance must be finite and positive.")
    source_index = jnp.arange(operator.source.size, dtype=operator.source.dtype)
    target_index = jnp.arange(operator.target.size, dtype=operator.target.dtype)
    source = (jnp.sin(0.37 * source_index) + 0.25 * jnp.cos(0.11 * source_index)).reshape(
        operator.source.shape
    )
    target = (jnp.cos(0.29 * target_index) - 0.2 * jnp.sin(0.17 * target_index)).reshape(
        operator.target.shape
    )
    action = operator.mv(source)
    transpose = operator.transpose_mv(target)
    target_cert = _certification_value(target, certification_dtype)
    action_cert = _certification_value(action, certification_dtype)
    transpose_cert = _certification_value(transpose, certification_dtype)
    source_cert = _certification_value(source, certification_dtype)
    left_coordinate = jnp.sum(target_cert * action_cert)
    right_coordinate = jnp.sum(transpose_cert * source_cert)
    coordinate_scale = jnp.maximum(
        1.0,
        jnp.maximum(jnp.abs(left_coordinate), jnp.abs(right_coordinate)),
    )
    coordinate_residual = float(
        np.asarray(jnp.abs(left_coordinate - right_coordinate) / coordinate_scale)
    )
    adjoint = operator.adjoint_mv(target)
    left_pairing = _pairing_inner(
        operator.target,
        target,
        action,
        certification_dtype=certification_dtype,
    )
    right_pairing = _pairing_inner(
        operator.source,
        adjoint,
        source,
        certification_dtype=certification_dtype,
    )
    pairing_scale = jnp.maximum(
        1.0,
        jnp.maximum(jnp.abs(left_pairing), jnp.abs(right_pairing)),
    )
    pairing_residual = float(
        np.asarray(jnp.abs(left_pairing - right_pairing) / pairing_scale)
    )
    return FDAdjointReport(
        coordinate_transpose_residual=coordinate_residual,
        pairing_adjoint_residual=pairing_residual,
        tolerance=tolerance_,
        operator_id=operator.operator_id,
    )


def certify_operator_conservation(
    operator: Any,
    /,
    *,
    periodic: bool,
    tolerance: float = 1e-10,
    certification_dtype: Any | None = None,
) -> FDConservationReport:
    """Probe constant annihilation and periodic global derivative balance."""
    tolerance_ = float(tolerance)
    constant = jnp.ones(operator.source.shape, dtype=operator.source.dtype)
    constant_action = _certification_value(
        operator.mv(constant),
        certification_dtype,
    )
    constant_residual = float(np.asarray(jnp.max(jnp.abs(constant_action))))
    balance_residual = None
    if periodic:
        index = jnp.arange(operator.source.size, dtype=operator.source.dtype)
        probe = (jnp.sin(0.31 * index) + 0.1 * jnp.cos(0.07 * index)).reshape(
            operator.source.shape
        )
        action = _certification_value(operator.mv(probe), certification_dtype)
        pairing = operator.target.pairing
        if isinstance(pairing, DiagonalPairing):
            weights = _certification_value(
                pairing.weights,
                certification_dtype,
            )
            balance = jnp.sum(weights * action)
            balance_scale = jnp.maximum(1.0, jnp.sum(jnp.abs(weights * action)))
        elif isinstance(pairing, EuclideanPairing):
            balance = jnp.sum(action)
            balance_scale = jnp.maximum(1.0, jnp.sum(jnp.abs(action)))
        else:
            raise ValueError("FD conservation requires Euclidean or diagonal pairing.")
        balance_residual = float(np.asarray(jnp.abs(balance) / balance_scale))
    return FDConservationReport(
        constant_state_residual=constant_residual,
        global_balance_residual=balance_residual,
        tolerance=tolerance_,
        operator_id=operator.operator_id,
    )


__all__ = [
    "certify_operator_adjoint",
    "certify_operator_conservation",
    "certify_stencil_consistency",
    "FDAdjointReport",
    "FDConsistencyReport",
    "FDConservationReport",
    "FDEvidenceKind",
    "FDStabilityReport",
]
