#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite, prod
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._interpolation._bspline import bspline_stencil
from ..._interpolation._bspline_grid import BSplineGrid
from ..._interpolation._bspline_projection import BSplineGridTransfer
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...lifecycle import NumericRevision
from ...linalg import AbstractLinearOperator, FunctionLinearOperator, transpose
from .._core import DiscretizationCapability, PreparationReport
from .._spaces import DiscreteFieldSpace, TensorDofLayout
from .._transfer import FieldTransfer, TransferProperties
from ._basis import SplineAxisPlan, TensorSplineBasisSpec


TransferClass: TypeAlias = Literal["exact", "projected"]


class TransferEvidence(StrictModule, NonTrainableState):
    """Auditable norm, quadrature, conditioning, and preservation evidence."""

    transfer_class: TransferClass = eqx.field(static=True)
    operator_norm: float = eqx.field(static=True)
    quadrature_id: str | None = eqx.field(static=True)
    condition_estimate: float = eqx.field(static=True)
    maximum_condition: float = eqx.field(static=True)
    projection_error_bound: float = eqx.field(static=True)
    pointwise_residual: float = eqx.field(static=True)
    constant_residual: float = eqx.field(static=True)
    duality_residual: float = eqx.field(static=True)
    preserved: tuple[str, ...] = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        transfer_class: TransferClass,
        /,
        *,
        operator_norm: float,
        quadrature_id: str | None,
        condition_estimate: float,
        maximum_condition: float,
        projection_error_bound: float,
        pointwise_residual: float,
        constant_residual: float,
        duality_residual: float,
        preserved: Sequence[str] = (),
    ):
        class_ = str(transfer_class)
        norm = float(operator_norm)
        quadrature = None if quadrature_id is None else str(quadrature_id)
        condition = float(condition_estimate)
        condition_limit = float(maximum_condition)
        error = float(projection_error_bound)
        pointwise = float(pointwise_residual)
        constant = float(constant_residual)
        duality = float(duality_residual)
        preserved_ = tuple(str(value) for value in preserved)
        if class_ not in ("exact", "projected"):
            raise ValueError("transfer_class must be 'exact' or 'projected'.")
        if quadrature is not None and not quadrature:
            raise ValueError("quadrature_id must be non-empty or None.")
        if class_ == "projected" and quadrature is None:
            raise ValueError("Projected IGA transfers require quadrature evidence.")
        if (
            not isfinite(norm)
            or norm < 0.0
            or not isfinite(condition)
            or condition < 1.0
            or not isfinite(condition_limit)
            or condition_limit <= 1.0
            or condition > condition_limit
            or any(
                not isfinite(value) or value < 0.0
                for value in (error, pointwise, constant, duality)
            )
        ):
            raise ValueError("IGA transfer numeric evidence is invalid or unqualified.")
        if class_ == "exact" and error != 0.0:
            raise ValueError("Exact IGA transfers must have zero projection error bound.")
        if any(not value for value in preserved_) or len(set(preserved_)) != len(
            preserved_
        ):
            raise ValueError("preserved entries must be unique non-empty strings.")
        self.transfer_class = class_  # type: ignore[assignment]
        self.operator_norm = norm
        self.quadrature_id = quadrature
        self.condition_estimate = condition
        self.maximum_condition = condition_limit
        self.projection_error_bound = error
        self.pointwise_residual = pointwise
        self.constant_residual = constant
        self.duality_residual = duality
        self.preserved = preserved_
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "iga-transfer-evidence",
                "transfer_class": class_,
                "operator_norm": norm,
                "quadrature": quadrature,
                "condition_estimate": condition,
                "maximum_condition": condition_limit,
                "projection_error_bound": error,
                "pointwise_residual": pointwise,
                "constant_residual": constant,
                "duality_residual": duality,
                "preserved": list(preserved_),
            }
        )


class TransferPlan(StrictModule, NonTrainableState):
    """Revision-bound IGA transfer exposing P, P^T, and an optional R."""

    field_transfer: FieldTransfer
    restriction_operator: AbstractLinearOperator | None
    evidence: TransferEvidence
    archive_names: tuple[str, ...] = eqx.field(static=True)
    archive_arrays: tuple[Array, ...]
    source_plan_id: str = eqx.field(static=True)
    target_plan_id: str = eqx.field(static=True)
    source_layout_id: str = eqx.field(static=True)
    target_layout_id: str = eqx.field(static=True)
    source_revision_id: str = eqx.field(static=True)
    target_revision_id: str = eqx.field(static=True)
    source_content_digest: str = eqx.field(static=True)
    target_content_digest: str = eqx.field(static=True)
    composition: tuple[str, ...] = eqx.field(static=True)
    invalidation_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        field_transfer: FieldTransfer,
        evidence: TransferEvidence,
        /,
        *,
        source_plan_id: str,
        target_plan_id: str,
        source_layout_id: str,
        target_layout_id: str,
        source_revision: NumericRevision,
        target_revision: NumericRevision,
        restriction_operator: AbstractLinearOperator | None = None,
        composition: Sequence[str] = (),
        archive_arrays: Mapping[str, ArrayLike] | Sequence[tuple[str, ArrayLike]] = (),
    ):
        if not isinstance(field_transfer, FieldTransfer):
            raise TypeError("field_transfer must be the generic FieldTransfer.")
        if not isinstance(evidence, TransferEvidence):
            raise TypeError("evidence must be TransferEvidence.")
        if not isinstance(source_revision, NumericRevision) or not isinstance(
            target_revision, NumericRevision
        ):
            raise TypeError(
                "source_revision and target_revision must be NumericRevision."
            )
        source_plan = str(source_plan_id)
        target_plan = str(target_plan_id)
        source_layout = str(source_layout_id)
        target_layout = str(target_layout_id)
        if any(
            not value
            for value in (source_plan, target_plan, source_layout, target_layout)
        ):
            raise ValueError("IGA transfer plan and layout IDs must be non-empty.")
        if field_transfer.source.layout.layout_id != source_layout:
            raise ValueError("source_layout_id does not match the FieldTransfer source.")
        if field_transfer.target.layout.layout_id != target_layout:
            raise ValueError("target_layout_id does not match the FieldTransfer target.")
        if field_transfer.adjoint_operator is None:
            raise ValueError("IGA FieldTransfer must provide the P^T operator.")
        if restriction_operator is not None:
            if not isinstance(restriction_operator, AbstractLinearOperator):
                raise TypeError(
                    "restriction_operator must be an AbstractLinearOperator or None."
                )
            if not restriction_operator.source.compatible(
                field_transfer.target.vector_space
            ) or not restriction_operator.target.compatible(
                field_transfer.source.vector_space
            ):
                raise ValueError("R must map the target field space back to the source.")
        composition_ = tuple(str(value) for value in composition)
        if any(not value for value in composition_) or len(set(composition_)) != len(
            composition_
        ):
            raise ValueError("composition entries must be unique non-empty strings.")
        items = (
            tuple(archive_arrays.items())
            if isinstance(archive_arrays, Mapping)
            else tuple(archive_arrays)
        )
        names = tuple(str(name) for name, _ in items)
        arrays = tuple(jnp.asarray(value) for _, value in items)
        if (
            any(not name for name in names)
            or len(set(names)) != len(names)
            or any(array.size == 0 for array in arrays)
            or any(
                not np.all(np.isfinite(np.asarray(array)))
                for array in arrays
                if np.issubdtype(np.asarray(array).dtype, np.inexact)
            )
        ):
            raise ValueError("IGA transition archive arrays are invalid.")
        invalidation = canonical_fingerprint(
            {
                "kind": "iga-transfer-invalidation",
                "source_plan": source_plan,
                "target_plan": target_plan,
                "source_layout": source_layout,
                "target_layout": target_layout,
                "source_revision": source_revision.revision_id,
                "target_revision": target_revision.revision_id,
            }
        )
        self.field_transfer = field_transfer
        self.restriction_operator = restriction_operator
        self.evidence = evidence
        self.archive_names = names
        self.archive_arrays = arrays
        self.source_plan_id = source_plan
        self.target_plan_id = target_plan
        self.source_layout_id = source_layout
        self.target_layout_id = target_layout
        self.source_revision_id = source_revision.revision_id
        self.target_revision_id = target_revision.revision_id
        self.source_content_digest = source_revision.content_digest
        self.target_content_digest = target_revision.content_digest
        self.composition = composition_
        self.invalidation_id = invalidation
        self.plan_id = canonical_fingerprint(
            {
                "kind": "iga-transfer-plan",
                "field_transfer": field_transfer.transfer_id,
                "restriction": None
                if restriction_operator is None
                else restriction_operator.operator_id,
                "evidence": evidence.evidence_id,
                "invalidation": invalidation,
                "composition": list(composition_),
                "archive_arrays": [
                    [name, array_tree_fingerprint(array)]
                    for name, array in zip(names, arrays, strict=True)
                ],
            }
        )

    @property
    def P(self) -> AbstractLinearOperator:
        """Primal coefficient map from source to target."""
        return self.field_transfer.operator

    @property
    def PT(self) -> AbstractLinearOperator:
        """Algebraic transpose P^T, used for raw dual pullback."""
        operator = self.field_transfer.adjoint_operator
        if operator is None:
            raise RuntimeError("IGA transfer construction omitted P^T.")
        return operator

    @property
    def R(self) -> AbstractLinearOperator | None:
        """Qualified target-to-source projection, when explicitly prepared."""
        return self.restriction_operator

    def apply(self, coefficients: Any, /) -> Any:
        return self.P.mv(coefficients)

    def pullback(self, dual_coefficients: Any, /) -> Any:
        return self.PT.mv(dual_coefficients)

    def restrict(self, coefficients: Any, /) -> Any:
        if self.R is None:
            raise ValueError("This IGA transfer has no qualified restriction R.")
        return self.R.mv(coefficients)

    def apply_payload(self, coefficients: ArrayLike, /) -> Array:
        """Apply P independently to trailing payload components."""
        source = self.P.source
        target = self.P.target
        values = jnp.asarray(coefficients)
        if not isinstance(source, type(target)) or not hasattr(source, "shape"):
            raise TypeError(
                "Payload transfer requires array-valued source/target spaces."
            )
        source_shape = source.shape
        target_shape = target.shape
        if values.ndim < len(source_shape) or tuple(
            values.shape[: len(source_shape)]
        ) != (*source_shape,):
            raise ValueError("Payload coefficients do not begin with the source shape.")
        payload_shape = tuple(int(size) for size in values.shape[len(source_shape) :])
        if not payload_shape:
            return jnp.asarray(self.apply(values))
        columns = values.reshape((source.size, prod(payload_shape)))
        transferred = self.P.mv_block(columns)
        return transferred.reshape(target_shape + payload_shape)

    def is_valid_for(
        self,
        *,
        source_plan_id: str,
        target_plan_id: str,
        source_layout_id: str,
        target_layout_id: str,
        source_revision: NumericRevision,
        target_revision: NumericRevision,
    ) -> bool:
        """Return whether every structure and numeric identity remains bound."""
        if not isinstance(source_revision, NumericRevision) or not isinstance(
            target_revision, NumericRevision
        ):
            raise TypeError(
                "source_revision and target_revision must be NumericRevision."
            )
        return (
            str(source_plan_id) == self.source_plan_id
            and str(target_plan_id) == self.target_plan_id
            and str(source_layout_id) == self.source_layout_id
            and str(target_layout_id) == self.target_layout_id
            and source_revision.revision_id == self.source_revision_id
            and target_revision.revision_id == self.target_revision_id
        )

    def require_valid_for(
        self,
        *,
        source_plan_id: str,
        target_plan_id: str,
        source_layout_id: str,
        target_layout_id: str,
        source_revision: NumericRevision,
        target_revision: NumericRevision,
    ) -> None:
        if not self.is_valid_for(
            source_plan_id=source_plan_id,
            target_plan_id=target_plan_id,
            source_layout_id=source_layout_id,
            target_layout_id=target_layout_id,
            source_revision=source_revision,
            target_revision=target_revision,
        ):
            raise ValueError(
                "IGA transfer was invalidated by a plan, layout, or revision change."
            )

    def transition_archive_payload(self, /) -> tuple[dict[str, Any], dict[str, Array]]:
        """Return deterministic metadata and arrays for a lifecycle archive shard."""
        manifest: dict[str, Any] = {
            "kind": "iga-transfer-transition",
            "plan_id": self.plan_id,
            "field_transfer_id": self.field_transfer.transfer_id,
            "source_plan_id": self.source_plan_id,
            "target_plan_id": self.target_plan_id,
            "source_layout_id": self.source_layout_id,
            "target_layout_id": self.target_layout_id,
            "source_revision_id": self.source_revision_id,
            "target_revision_id": self.target_revision_id,
            "source_content_digest": self.source_content_digest,
            "target_content_digest": self.target_content_digest,
            "transfer_class": self.evidence.transfer_class,
            "evidence_id": self.evidence.evidence_id,
            "composition": list(self.composition),
            "restriction_operator_id": None if self.R is None else self.R.operator_id,
        }
        return manifest, dict(zip(self.archive_names, self.archive_arrays, strict=True))


def _apply_axis_matrices(
    coefficients: Array, matrices: tuple[Array, ...], /, *, transpose_action: bool
) -> Array:
    result = coefficients
    for axis, matrix in enumerate(matrices):
        moved = jnp.moveaxis(result, axis, 0)
        if transpose_action:
            moved = oe.contract("ij,i...->j...", matrix, moved)
        else:
            moved = oe.contract("ji,i...->j...", matrix, moved)
        result = jnp.moveaxis(moved, 0, axis)
    return result


def _tensor_operator(
    matrices: tuple[Array, ...],
    source: DiscreteFieldSpace,
    target: DiscreteFieldSpace,
    /,
    *,
    operator_id: str,
) -> FunctionLinearOperator:
    return FunctionLinearOperator(
        lambda value: _apply_axis_matrices(value, matrices, transpose_action=False),
        source=source.vector_space,
        target=target.vector_space,
        transpose_action=lambda value: _apply_axis_matrices(
            value, matrices, transpose_action=True
        ),
        operator_id=operator_id,
    )


def _axis_grid(axis: SplineAxisPlan, /) -> BSplineGrid:
    return BSplineGrid(axis.knots, axis.degree)


def _knot_multiplicities(knots: np.ndarray, /) -> dict[float, int]:
    unique, counts = np.unique(knots, return_counts=True)
    return {float(knot): int(count) for knot, count in zip(unique, counts, strict=True)}


def _is_nested_axis(source: SplineAxisPlan, target: SplineAxisPlan, /) -> bool:
    if (
        source.name != target.name
        or source.parameter_interval != target.parameter_interval
    ):
        return False
    degree_increase = target.degree - source.degree
    if degree_increase < 0:
        return False
    source_multiplicity = _knot_multiplicities(np.asarray(source.knots))
    target_multiplicity = _knot_multiplicities(np.asarray(target.knots))
    return all(
        target_multiplicity.get(knot, 0) >= multiplicity + degree_increase
        for knot, multiplicity in source_multiplicity.items()
    )


def _basis_matrix(axis: SplineAxisPlan, points: np.ndarray, /) -> np.ndarray:
    stencil = bspline_stencil(axis.knots, points, degree=axis.degree)
    indices = np.asarray(stencil.source_indices)
    weights = np.asarray(stencil.weights)
    matrix = np.zeros((points.size, axis.control_count), dtype=weights.dtype)
    rows = np.arange(points.size)[:, None]
    matrix[rows, indices] = weights
    return matrix


def _verification_points(source: SplineAxisPlan, target: SplineAxisPlan, /) -> np.ndarray:
    breakpoints = np.unique(
        np.concatenate((np.asarray(source.knots), np.asarray(target.knots)))
    )
    lower, upper = source.parameter_interval
    breakpoints = breakpoints[(breakpoints >= lower) & (breakpoints <= upper)]
    nodes, _ = np.polynomial.legendre.leggauss(max(source.degree, target.degree) + 2)
    pieces = []
    for left, right in zip(breakpoints[:-1], breakpoints[1:], strict=True):
        if right > left:
            pieces.append(0.5 * ((right - left) * nodes + right + left))
    return np.unique(np.concatenate((np.asarray([lower, upper]), *pieces)))


def _exact_axis_matrix(
    source: SplineAxisPlan,
    target: SplineAxisPlan,
    /,
    *,
    maximum_condition: float,
) -> tuple[np.ndarray, float, float, float]:
    if not _is_nested_axis(source, target):
        raise ValueError(
            "Exact IGA transfer requires a structurally nested target spline space."
        )
    source_grid = _axis_grid(source)
    target_grid = _axis_grid(target)
    if source.degree == target.degree:
        matrix = np.asarray(
            BSplineGridTransfer(
                source_grid,
                target_grid,
                method="exact",
                maximum_condition=maximum_condition,
            ).matrix
        )
        solve_condition = 1.0
    else:
        collocation_points = np.asarray(target_grid.greville_abscissae)
        target_collocation = _basis_matrix(target, collocation_points)
        source_collocation = _basis_matrix(source, collocation_points)
        solve_condition = float(np.linalg.cond(target_collocation))
        if not isfinite(solve_condition) or solve_condition > maximum_condition:
            raise ValueError(
                "Exact IGA degree elevation collocation is ill-conditioned: "
                f"condition estimate {solve_condition:.6g}."
            )
        matrix = np.linalg.solve(target_collocation, source_collocation)
    points = _verification_points(source, target)
    residual = _basis_matrix(target, points) @ matrix - _basis_matrix(source, points)
    scale = max(1.0, float(np.max(np.abs(_basis_matrix(source, points)))))
    pointwise = float(np.max(np.abs(residual)) / scale)
    constant = float(np.max(np.abs(matrix @ np.ones(source.control_count) - 1.0)))
    matrix_condition = float(np.linalg.cond(matrix))
    condition = max(1.0, solve_condition, matrix_condition)
    return matrix, condition, pointwise, constant


def _projected_axis_matrix(
    source: SplineAxisPlan,
    target: SplineAxisPlan,
    /,
    *,
    maximum_condition: float,
) -> tuple[np.ndarray, float, float, float, float]:
    transfer = BSplineGridTransfer(
        _axis_grid(source),
        _axis_grid(target),
        method="l2",
        maximum_condition=maximum_condition,
    )
    matrix = np.asarray(transfer.matrix)
    points = _verification_points(source, target)
    residual = _basis_matrix(target, points) @ matrix - _basis_matrix(source, points)
    scale = max(1.0, float(np.max(np.abs(_basis_matrix(source, points)))))
    pointwise = float(np.max(np.abs(residual)) / scale)
    constant = float(np.max(np.abs(matrix @ np.ones(source.control_count) - 1.0)))
    condition = max(1.0, float(transfer.condition_estimate))
    return (
        matrix,
        condition,
        float(transfer.projection_error_bound),
        pointwise,
        constant,
    )


def prepare_tensor_transfer(
    source_basis: TensorSplineBasisSpec,
    target_basis: TensorSplineBasisSpec,
    source_field: DiscreteFieldSpace,
    target_field: DiscreteFieldSpace,
    /,
    *,
    source_plan_id: str,
    target_plan_id: str,
    source_revision: NumericRevision,
    target_revision: NumericRevision,
    transfer_class: TransferClass,
    quadrature_id: str | None = None,
    maximum_condition: float = 1.0e12,
    include_restriction: bool = False,
    composition: Sequence[str] = (),
) -> TransferPlan:
    """Prepare a qualified tensor transfer without materializing a global matrix."""
    if not isinstance(source_basis, TensorSplineBasisSpec) or not isinstance(
        target_basis, TensorSplineBasisSpec
    ):
        raise TypeError("source_basis and target_basis must be TensorSplineBasisSpec.")
    if not isinstance(source_field, DiscreteFieldSpace) or not isinstance(
        target_field, DiscreteFieldSpace
    ):
        raise TypeError("source_field and target_field must be DiscreteFieldSpace.")
    if not isinstance(source_field.layout, TensorDofLayout) or not isinstance(
        target_field.layout, TensorDofLayout
    ):
        raise TypeError("IGA tensor transfers require TensorDofLayout field spaces.")
    if (
        source_basis.axis_names != target_basis.axis_names
        or source_field.layout.axis_names != source_basis.axis_names
        or target_field.layout.axis_names != target_basis.axis_names
        or source_field.layout.axis_shape != source_basis.control_shape
        or target_field.layout.axis_shape != target_basis.control_shape
    ):
        raise ValueError("IGA transfer basis axes and field layouts must match exactly.")
    class_ = str(transfer_class)
    if class_ not in ("exact", "projected"):
        raise ValueError("transfer_class must be 'exact' or 'projected'.")
    limit = float(maximum_condition)
    if not isfinite(limit) or limit <= 1.0:
        raise ValueError("maximum_condition must be finite and greater than one.")
    if class_ == "projected" and (quadrature_id is None or not str(quadrature_id)):
        raise ValueError("Projected IGA transfers require an explicit quadrature_id.")

    factors: list[Array] = []
    conditions: list[float] = []
    projection_errors: list[float] = []
    pointwise_residuals: list[float] = []
    constant_residuals: list[float] = []
    for source_axis, target_axis in zip(
        source_basis.axes, target_basis.axes, strict=True
    ):
        if class_ == "exact":
            matrix, condition, pointwise, constant = _exact_axis_matrix(
                source_axis,
                target_axis,
                maximum_condition=limit,
            )
            error = 0.0
        else:
            matrix, condition, error, pointwise, constant = _projected_axis_matrix(
                source_axis,
                target_axis,
                maximum_condition=limit,
            )
        factors.append(jnp.asarray(matrix))
        conditions.append(condition)
        projection_errors.append(error)
        pointwise_residuals.append(pointwise)
        constant_residuals.append(constant)

    factor_tuple = tuple(factors)
    operator_id = canonical_fingerprint(
        {
            "kind": "iga-tensor-prolongation",
            "source_basis": source_basis.basis_id,
            "target_basis": target_basis.basis_id,
            "factors": [array_tree_fingerprint(matrix) for matrix in factor_tuple],
        }
    )
    operator = _tensor_operator(
        factor_tuple,
        source_field,
        target_field,
        operator_id=operator_id,
    )
    transpose_operator = transpose(operator)
    constant_residual = max(constant_residuals, default=0.0)
    preservation_tolerance = (
        256.0
        * np.finfo(float).eps
        * max(source_basis.control_shape + target_basis.control_shape)
    )
    exact_on = (
        ("constants", "spline-space", "homogeneous-geometry")
        if class_ == "exact"
        else (("constants",) if constant_residual <= preservation_tolerance else ())
    )
    preparation = PreparationReport(
        capabilities=(DiscretizationCapability.FIELD_TRANSFER,),
        diagnostics=(
            f"class:{class_}",
            f"condition:{prod(conditions):.6e}",
            f"pointwise_residual:{max(pointwise_residuals, default=0.0):.6e}",
        ),
        resource_counts={
            "axis_factor_entries": sum(int(matrix.size) for matrix in factor_tuple),
        },
    )
    field_transfer = FieldTransfer(
        source_field,
        target_field,
        operator,
        adjoint_operator=transpose_operator,
        properties=TransferProperties(
            constant_preserving=constant_residual <= preservation_tolerance,
            nested=class_ == "exact",
            adjoint_paired=True,
            differentiable_geometry=True,
            exact_on=exact_on,
        ),
        preparation=preparation,
    )

    restriction_operator = None
    restriction_factors: tuple[Array, ...] = ()
    if include_restriction:
        if quadrature_id is None or not str(quadrature_id):
            raise ValueError(
                "Qualified restriction R requires an explicit quadrature_id."
            )
        reverse_matrices = []
        for source_axis, target_axis in zip(
            source_basis.axes, target_basis.axes, strict=True
        ):
            reverse, condition, _, _, _ = _projected_axis_matrix(
                target_axis,
                source_axis,
                maximum_condition=limit,
            )
            conditions.append(condition)
            reverse_matrices.append(jnp.asarray(reverse))
        restriction_factors = tuple(reverse_matrices)
        restriction_operator = _tensor_operator(
            restriction_factors,
            target_field,
            source_field,
            operator_id=canonical_fingerprint(
                {
                    "kind": "iga-tensor-restriction",
                    "source_basis": target_basis.basis_id,
                    "target_basis": source_basis.basis_id,
                    "factors": [
                        array_tree_fingerprint(matrix) for matrix in restriction_factors
                    ],
                }
            ),
        )

    condition_estimate = prod(conditions)
    if not isfinite(condition_estimate) or condition_estimate > limit:
        raise ValueError(
            "IGA tensor transfer is ill-conditioned: "
            f"condition estimate {condition_estimate:.6g}."
        )
    operator_norm = prod(
        float(np.linalg.norm(np.asarray(matrix), ord=2)) for matrix in factor_tuple
    )
    evidence = TransferEvidence(
        class_,  # type: ignore[arg-type]
        operator_norm=operator_norm,
        quadrature_id=quadrature_id,
        condition_estimate=max(1.0, condition_estimate),
        maximum_condition=limit,
        projection_error_bound=0.0
        if class_ == "exact"
        else float(np.sqrt(np.sum(np.square(projection_errors)))),
        pointwise_residual=max(pointwise_residuals, default=0.0),
        constant_residual=constant_residual,
        duality_residual=0.0,
        preserved=exact_on,
    )
    archive: dict[str, Array] = {
        f"P_axis_{axis:03d}": matrix for axis, matrix in enumerate(factor_tuple)
    }
    archive.update(
        {f"R_axis_{axis:03d}": matrix for axis, matrix in enumerate(restriction_factors)}
    )
    return TransferPlan(
        field_transfer,
        evidence,
        source_plan_id=source_plan_id,
        target_plan_id=target_plan_id,
        source_layout_id=source_basis.layout_id,
        target_layout_id=target_basis.layout_id,
        source_revision=source_revision,
        target_revision=target_revision,
        restriction_operator=restriction_operator,
        composition=composition,
        archive_arrays=archive,
    )


__all__ = [
    "TransferClass",
    "TransferEvidence",
    "TransferPlan",
    "prepare_tensor_transfer",
]
