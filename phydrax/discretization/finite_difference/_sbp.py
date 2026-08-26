#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import (
    AbstractLinearOperator,
    ArraySpace,
    DiagonalPairing,
    OperatorCapabilities,
    OperatorProperties,
)
from .._spaces import DiscreteFieldSpace
from .._tensor_support import PreparedTensorGrid
from ._certification import FDStabilityReport
from ._coefficients import StencilCoefficientPlan
from ._operators import PreparedStencilOperator
from ._request import DerivativeRequest
from ._stencil import BoundaryStencilSet, LinearStencil, StencilFootprint


SBPInteriorOrder: TypeAlias = Literal[2, 4, 6, 8]
SATConditionKind: TypeAlias = Literal["none", "dirichlet", "neumann", "robin"]
SATInterfaceFlux: TypeAlias = Literal["central", "upwind"]


_NORM_BOUNDARY_WEIGHTS = {
    2: (1.0 / 2.0,),
    4: (17.0 / 48.0, 59.0 / 48.0, 43.0 / 48.0, 49.0 / 48.0),
    6: (
        13649.0 / 43200.0,
        12013.0 / 8640.0,
        2711.0 / 4320.0,
        5359.0 / 4320.0,
        7877.0 / 8640.0,
        43801.0 / 43200.0,
    ),
    8: (
        1498139.0 / 5080320.0,
        1107307.0 / 725760.0,
        20761.0 / 80640.0,
        1304999.0 / 725760.0,
        299527.0 / 725760.0,
        103097.0 / 80640.0,
        670091.0 / 725760.0,
        5127739.0 / 5080320.0,
    ),
}

_INTERIOR_FIRST_DERIVATIVE = {
    2: (-1.0 / 2.0, 0.0, 1.0 / 2.0),
    4: (1.0 / 12.0, -2.0 / 3.0, 0.0, 2.0 / 3.0, -1.0 / 12.0),
    6: (
        -1.0 / 60.0,
        3.0 / 20.0,
        -3.0 / 4.0,
        0.0,
        3.0 / 4.0,
        -3.0 / 20.0,
        1.0 / 60.0,
    ),
    8: (
        1.0 / 280.0,
        -4.0 / 105.0,
        1.0 / 5.0,
        -4.0 / 5.0,
        0.0,
        4.0 / 5.0,
        -1.0 / 5.0,
        4.0 / 105.0,
        -1.0 / 280.0,
    ),
}


class SBPFamily(StrictModule, NonTrainableState):
    """One diagonal-norm first-derivative family with verified construction data."""

    interior_order: SBPInteriorOrder = eqx.field(static=True)
    closure_order: int = eqx.field(static=True)
    boundary_width: int = eqx.field(static=True)
    norm_boundary_weights: tuple[float, ...] = eqx.field(static=True)
    family_id: str = eqx.field(static=True)

    def __init__(self, interior_order: SBPInteriorOrder, /):
        order = int(interior_order)
        if order not in (2, 4, 6, 8):
            raise ValueError("Diagonal-norm SBP interior order must be 2, 4, 6, or 8.")
        norm = _NORM_BOUNDARY_WEIGHTS[order]
        self.interior_order = order
        self.closure_order = order // 2
        self.boundary_width = len(norm)
        self.norm_boundary_weights = norm
        self.family_id = canonical_fingerprint(
            {
                "kind": "diagonal-norm-sbp-family",
                "interior_order": order,
                "closure_order": order // 2,
                "norm_boundary_weights": list(norm),
            }
        )


def _normalized_sbp_matrix(
    family: SBPFamily,
    count: int,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    order = family.interior_order
    half_order = order // 2
    boundary_width = family.boundary_width
    minimum_count = 2 * boundary_width + 2 * half_order + 1
    if count < minimum_count:
        raise ValueError(f"SBP order {order} requires at least {minimum_count} nodes.")
    norm = np.ones((count,), dtype=float)
    boundary_norm = np.asarray(family.norm_boundary_weights)
    norm[:boundary_width] = boundary_norm
    norm[-boundary_width:] = boundary_norm[::-1]
    q_matrix = np.zeros((count, count), dtype=float)
    interior = np.asarray(_INTERIOR_FIRST_DERIVATIVE[order])
    for row in range(boundary_width, count - boundary_width):
        q_matrix[row, row - half_order : row + half_order + 1] = interior
    boundary_indices = tuple(range(boundary_width)) + tuple(
        range(count - boundary_width, count)
    )
    for row in range(boundary_width, count - boundary_width):
        for column in boundary_indices:
            q_matrix[column, row] = -q_matrix[row, column]
    q_matrix[0, 0] = -0.5
    q_matrix[-1, -1] = 0.5
    pairs = tuple(
        (left, right)
        for left in range(boundary_width)
        for right in range(left + 1, boundary_width)
    )
    if pairs:
        equations = []
        targets = []
        coordinates = np.arange(count, dtype=float)
        for row in range(boundary_width):
            for degree in range(family.closure_order + 1):
                monomial = coordinates**degree
                known = float(q_matrix[row] @ monomial)
                target = norm[row] * (
                    0.0 if degree == 0 else degree * coordinates[row] ** (degree - 1)
                )
                equations.append(
                    [
                        monomial[right]
                        if row == left
                        else -monomial[left]
                        if row == right
                        else 0.0
                        for left, right in pairs
                    ]
                )
                targets.append(target - known)
        system = np.asarray(equations)
        right_hand_side = np.asarray(targets)
        solution, _, _, _ = np.linalg.lstsq(system, right_hand_side, rcond=None)
        residual = np.max(np.abs(system @ solution - right_hand_side))
        if residual > 5e-10:
            raise RuntimeError(
                "SBP boundary closure construction failed its constraints."
            )
        for value, (left, right) in zip(solution, pairs, strict=True):
            q_matrix[left, right] = value
            q_matrix[right, left] = -value
    for row in range(boundary_width):
        for column in range(boundary_width):
            q_matrix[count - 1 - row, count - 1 - column] = -q_matrix[row, column]
    derivative = q_matrix / norm[:, None]
    boundary = np.zeros((count, count), dtype=float)
    boundary[0, 0] = -1.0
    boundary[-1, -1] = 1.0
    identity_residual = np.max(
        np.abs(np.diag(norm) @ derivative + derivative.T @ np.diag(norm) - boundary)
    )
    if identity_residual > 5e-12:
        raise RuntimeError("Constructed SBP derivative violates its norm identity.")
    return derivative, norm

def _normalized_periodic_sbp_matrix(
    family: SBPFamily,
    count: int,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    order = family.interior_order
    half_order = order // 2
    minimum_count = 2 * half_order + 1
    if count < minimum_count:
        raise ValueError(
            f"Periodic SBP order {order} requires at least {minimum_count} nodes."
        )
    relative = np.arange(-half_order, half_order + 1, dtype=np.int32)
    coefficients = np.asarray(_INTERIOR_FIRST_DERIVATIVE[order])
    derivative = np.zeros((count, count), dtype=float)
    for row in range(count):
        derivative[row, (row + relative) % count] = coefficients
    residual = np.max(np.abs(derivative + derivative.T))
    if residual > 5e-14:
        raise RuntimeError("Periodic SBP derivative is not skew-symmetric.")
    return derivative, np.ones((count,), dtype=float)


def _tensor_norm_weights(
    grid: PreparedTensorGrid,
    axis_index: int,
    axis_norm: Array,
    /,
) -> Array:
    entities = grid.primary_entity_layout.axis_entities
    result = jnp.ones(grid.shape)
    for index, (structured_axis, entity) in enumerate(
        zip(grid.structured_axes, entities, strict=True)
    ):
        weights = axis_norm if index == axis_index else structured_axis.measure(entity)
        reshape = [1] * len(grid.shape)
        reshape[index] = int(weights.size)
        result = result * weights.reshape(reshape)
    return result


class SBPDerivativePlan(StrictModule, NonTrainableState):
    """Preparation contract for one tensor-axis diagonal-norm SBP derivative."""

    grid: PreparedTensorGrid
    axis: str = eqx.field(static=True)
    family: SBPFamily
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        grid: PreparedTensorGrid,
        axis: str,
        /,
        *,
        interior_order: SBPInteriorOrder = 2,
    ):
        if not isinstance(grid, PreparedTensorGrid):
            raise TypeError("grid must be a PreparedTensorGrid.")
        axis_ = str(axis)
        if axis_ not in grid.axis_names:
            raise ValueError("SBP axis must belong to the prepared tensor grid.")
        axis_index = grid.axis_names.index(axis_)
        structured_axis = grid.structured_axes[axis_index]
        if structured_axis.primary_entity != "point":
            raise ValueError("SBP derivatives require a point-primary axis.")
        family = SBPFamily(interior_order)
        self.grid = grid
        self.axis = axis_
        self.family = family
        self.plan_id = canonical_fingerprint(
            {
                "kind": "sbp-derivative-plan-v2",
                "grid": grid.prepared_id,
                "axis": axis_,
                "family": family.family_id,
                "periodic": bool(structured_axis.periodic),
            }
        )

    def prepare(self, /) -> "PreparedSBPOperator":
        return PreparedSBPOperator(self)


class CompatibleSBPSecondDerivative(AbstractLinearOperator):
    """Matrix-free compatible second derivative using the first-derivative norm."""

    source: ArraySpace
    target: ArraySpace

    first_operator: PreparedStencilOperator
    norm_weights: Array
    axis_index: int = eqx.field(static=True)
    coefficient: Array

    def __init__(
        self,
        first_derivative: "PreparedSBPOperator",
        /,
        *,
        coefficient: ArrayLike = 1.0,
    ):
        if not isinstance(first_derivative, PreparedSBPOperator):
            raise TypeError("first_derivative must be a PreparedSBPOperator.")
        coefficient_ = jnp.broadcast_to(
            jnp.asarray(coefficient, dtype=first_derivative.operator.source.dtype),
            first_derivative.grid.shape,
        )
        coefficient_ = eqx.error_if(
            coefficient_,
            jnp.any(~jnp.isfinite(coefficient_)) | jnp.any(coefficient_ <= 0.0),
            "Compatible SBP coefficient must be finite and positive.",
        )
        self.source = first_derivative.operator.source
        self.target = first_derivative.operator.target
        self.properties = OperatorProperties(evidence={})
        self.capabilities = OperatorCapabilities(
            transpose=True,
            adjoint=True,
            materialize=True,
            diagonal_assembly=False,
        )
        self.batch_shape = ()
        self.operator_id = canonical_fingerprint(
            {
                "kind": "compatible-sbp-second-derivative",
                "first_derivative": first_derivative.prepared_id,
                "coefficient_shape": list(coefficient_.shape),
            }
        )
        self.first_operator = first_derivative.operator
        self.norm_weights = first_derivative.norm_weights
        self.axis_index = first_derivative.axis_index
        self.coefficient = coefficient_

    def _boundary_flux(self, gradient: Array, /) -> Array:
        axis = self.axis_index
        flux = self.coefficient * gradient
        correction = jnp.zeros_like(flux)
        lower_index: list[slice | int] = [slice(None)] * flux.ndim
        upper_index: list[slice | int] = [slice(None)] * flux.ndim
        lower_index[axis] = 0
        upper_index[axis] = flux.shape[axis] - 1
        correction = correction.at[tuple(lower_index)].set(-flux[tuple(lower_index)])
        return correction.at[tuple(upper_index)].set(flux[tuple(upper_index)])

    def mv(self, vector: ArrayLike, /) -> Array:
        value = self.source.validate(jnp.asarray(vector))
        first = self.first_operator
        gradient = first.mv(value)
        weighted_gradient = self.norm_weights * self.coefficient * gradient
        volume = -first.transpose_mv(weighted_gradient)
        return (volume + self._boundary_flux(gradient)) / self.norm_weights

    def transpose_mv(self, vector: ArrayLike, /) -> Array:
        value = self.target.validate(jnp.asarray(vector))
        first = self.first_operator
        scaled = value / self.norm_weights
        derivative = first.mv(scaled)
        boundary = jnp.zeros_like(scaled)
        axis = self.axis_index
        lower_index: list[slice | int] = [slice(None)] * scaled.ndim
        upper_index: list[slice | int] = [slice(None)] * scaled.ndim
        lower_index[axis] = 0
        upper_index[axis] = scaled.shape[axis] - 1
        boundary = boundary.at[tuple(lower_index)].set(-scaled[tuple(lower_index)])
        boundary = boundary.at[tuple(upper_index)].set(scaled[tuple(upper_index)])
        covector = self.coefficient * (-self.norm_weights * derivative + boundary)
        return first.transpose_mv(covector)

    def adjoint_mv(self, vector: ArrayLike, /) -> Array:
        value = self.target.validate(jnp.asarray(vector))
        weighted = self.norm_weights * value
        transposed = self.transpose_mv(weighted)
        return transposed / self.norm_weights

    def _materialize(self, /) -> Array:
        if self.source.size * self.target.size > 4096**2:
            raise ValueError("SBP second-derivative materialization exceeds size budget.")
        identity = jnp.eye(self.source.size, dtype=self.source.dtype).reshape(
            (self.source.size,) + self.source.shape
        )
        columns = jax.vmap(self.mv)(identity).reshape((self.source.size, -1))
        return columns.T


class PreparedSBPOperator(StrictModule, NonTrainableState):
    """Prepared SBP first/second derivatives, norm, and algebraic evidence."""

    plan: SBPDerivativePlan
    grid: PreparedTensorGrid
    axis: str = eqx.field(static=True)
    axis_index: int = eqx.field(static=True)
    family: SBPFamily
    operator: PreparedStencilOperator
    second_derivative: CompatibleSBPSecondDerivative
    axis_norm_weights: Array
    norm_weights: Array
    boundary_diagonal: Array
    stability_report: FDStabilityReport
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: SBPDerivativePlan, /):
        if not isinstance(plan, SBPDerivativePlan):
            raise TypeError("plan must be an SBPDerivativePlan.")
        grid = plan.grid
        axis_index = grid.axis_names.index(plan.axis)
        structured_axis = grid.structured_axes[axis_index]
        nodes = np.asarray(structured_axis.point_coordinates, dtype=float)
        spacing = np.diff(nodes)
        if not np.allclose(spacing, spacing[0], rtol=1e-10, atol=1e-12):
            raise ValueError("Diagonal-norm SBP derivatives require uniform spacing.")
        delta = float(spacing[0])
        periodic = bool(structured_axis.periodic)
        if periodic:
            normalized, normalized_norm = _normalized_periodic_sbp_matrix(
                plan.family, nodes.size
            )
        else:
            normalized, normalized_norm = _normalized_sbp_matrix(
                plan.family, nodes.size
            )
        matrix = normalized / delta
        axis_norm = jnp.asarray(delta * normalized_norm)
        row_plans = []
        row_kinds = []
        if periodic:
            half_order = plan.family.interior_order // 2
            relative = np.arange(-half_order, half_order + 1, dtype=np.int32)
            coefficients = np.asarray(
                _INTERIOR_FIRST_DERIVATIVE[plan.family.interior_order]
            ) / delta
            maximum_width = relative.size
            indices = np.zeros((nodes.size, maximum_width), dtype=np.int32)
            weights = np.zeros((nodes.size, maximum_width), dtype=float)
            valid = np.ones((nodes.size, maximum_width), dtype=bool)
            for row in range(nodes.size):
                indices[row] = (row + relative) % nodes.size
                weights[row] = coefficients
                row_plans.append(
                    StencilCoefficientPlan(
                        relative.astype(float) * delta,
                        0.0,
                        1,
                        plan.family.interior_order,
                        weights=coefficients,
                        residual_tolerance=2e-7,
                    )
                )
                row_kinds.append("interior")
            footprint_lower = half_order
            footprint_upper = half_order
            closure_order = plan.family.interior_order
            boundary_kind = "periodic"
            boundary_diagonal = jnp.zeros((nodes.size,))
        else:
            maximum_width = 0
            active_rows = []
            threshold = 5e-14 / delta
            for row in range(nodes.size):
                active = np.flatnonzero(np.abs(matrix[row]) > threshold)
                active_rows.append(active)
                maximum_width = max(maximum_width, int(active.size))
            indices = np.zeros((nodes.size, maximum_width), dtype=np.int32)
            weights = np.full((nodes.size, maximum_width), np.nan)
            valid = np.zeros((nodes.size, maximum_width), dtype=bool)
            boundary_width = plan.family.boundary_width
            for row, active in enumerate(active_rows):
                row_accuracy = (
                    plan.family.closure_order
                    if row < boundary_width or row >= nodes.size - boundary_width
                    else plan.family.interior_order
                )
                indices[row, : active.size] = active
                weights[row, : active.size] = matrix[row, active]
                valid[row, : active.size] = True
                row_plans.append(
                    StencilCoefficientPlan(
                        nodes[active],
                        nodes[row],
                        1,
                        row_accuracy,
                        weights=matrix[row, active],
                        residual_tolerance=2e-7,
                    )
                )
                row_kinds.append(
                    "lower_closure"
                    if row < boundary_width
                    else "upper_closure"
                    if row >= nodes.size - boundary_width
                    else "interior"
                )
            footprint_lower = max(
                row - int(np.min(active)) for row, active in enumerate(active_rows)
            )
            footprint_upper = max(
                int(np.max(active)) - row for row, active in enumerate(active_rows)
            )
            closure_order = plan.family.closure_order
            boundary_kind = "one_sided"
            boundary_diagonal = (
                jnp.zeros((nodes.size,)).at[0].set(-1.0).at[-1].set(1.0)
            )
        request = DerivativeRequest(
            f"sbp_d_{plan.axis}_{plan.family.interior_order}",
            grid,
            plan.axis,
            derivative_order=1,
            accuracy_order=plan.family.interior_order,
            boundary=boundary_kind,
        )
        lower = [0] * len(grid.shape)
        upper = [0] * len(grid.shape)
        lower[axis_index] = footprint_lower
        upper[axis_index] = footprint_upper
        stencil = LinearStencil(
            request,
            axis_index,
            indices,
            weights,
            row_plans,
            StencilFootprint(grid.axis_names, lower, upper),
            valid=valid,
            row_kinds=row_kinds,
        )
        stencil_set = BoundaryStencilSet(
            stencil,
            kind=boundary_kind,
            interior_accuracy_order=plan.family.interior_order,
            closure_accuracy_order=closure_order,
        )
        norm = _tensor_norm_weights(grid, axis_index, axis_norm)
        base_field = grid.field_space("sbp_state")
        if not isinstance(base_field.vector_space, ArraySpace):
            raise TypeError("SBP tensor grid requires an ArraySpace field.")
        vector_space = ArraySpace(
            grid.shape,
            dtype=base_field.vector_space.dtype,
            pairing=DiagonalPairing(norm),
        )
        field = DiscreteFieldSpace(
            "sbp_state",
            base_field.support_id,
            base_field.layout,
            vector_space,
            representation=base_field.representation,
            conformity=base_field.conformity,
        )
        operator = PreparedStencilOperator(stencil_set, field, field)
        residual = float(
            np.max(
                np.abs(
                    np.diag(np.asarray(axis_norm)) @ matrix
                    + matrix.T @ np.diag(np.asarray(axis_norm))
                    - np.diag(np.asarray(boundary_diagonal))
                )
            )
        )
        prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-sbp-operator-v2",
                "plan": plan.plan_id,
                "operator": operator.operator_id,
                "periodic": periodic,
            }
        )
        self.plan = plan
        self.grid = grid
        self.axis = plan.axis
        self.axis_index = axis_index
        self.family = plan.family
        self.operator = operator
        self.axis_norm_weights = axis_norm
        self.norm_weights = norm
        self.boundary_diagonal = boundary_diagonal
        self.stability_report = FDStabilityReport(
            "sbp_norm_identity",
            residual=residual,
            tolerance=5e-11,
            assumptions=(
                "uniform point-primary axis",
                "diagonal positive norm",
                "periodic skew derivative" if periodic else "bounded SBP closure",
            ),
            evidence="algebraic",
            subject_id=prepared_id,
        )
        self.prepared_id = prepared_id
        self.second_derivative = CompatibleSBPSecondDerivative(self)

    def identity_residual(self, /) -> Array:
        if len(self.grid.shape) != 1:
            raise ValueError(
                "Explicit SBP identity residual is a one-dimensional diagnostic."
            )
        matrix = self.operator._materialize()
        norm = jnp.diag(self.axis_norm_weights)
        return norm @ matrix + matrix.T @ norm - jnp.diag(self.boundary_diagonal)


class SATBoundaryPlan(StrictModule, NonTrainableState):
    """Explicit SAT residual and penalty realization for one SBP axis."""

    sbp: PreparedSBPOperator
    lower_kind: SATConditionKind = eqx.field(static=True)
    upper_kind: SATConditionKind = eqx.field(static=True)
    lower_alpha: float = eqx.field(static=True)
    lower_beta: float = eqx.field(static=True)
    upper_alpha: float = eqx.field(static=True)
    upper_beta: float = eqx.field(static=True)
    lower_penalty: float = eqx.field(static=True)
    upper_penalty: float = eqx.field(static=True)
    stability_report: FDStabilityReport
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        sbp: PreparedSBPOperator,
        lower_kind: SATConditionKind,
        upper_kind: SATConditionKind,
        /,
        *,
        lower_alpha: float = 1.0,
        lower_beta: float = 1.0,
        upper_alpha: float = 1.0,
        upper_beta: float = 1.0,
        lower_penalty: float = 0.0,
        upper_penalty: float = 0.0,
        stability_report: FDStabilityReport | None = None,
    ):
        if not isinstance(sbp, PreparedSBPOperator):
            raise TypeError("sbp must be a PreparedSBPOperator.")
        if lower_kind not in (
            "none",
            "dirichlet",
            "neumann",
            "robin",
        ) or upper_kind not in (
            "none",
            "dirichlet",
            "neumann",
            "robin",
        ):
            raise ValueError("Unknown SAT boundary condition kind.")
        coefficients = tuple(
            float(value) for value in (lower_alpha, lower_beta, upper_alpha, upper_beta)
        )
        penalties = (float(lower_penalty), float(upper_penalty))
        if any(not np.isfinite(value) for value in coefficients + penalties):
            raise ValueError("SAT coefficients and penalties must be finite.")
        identifier = canonical_fingerprint(
            {
                "kind": "sat-boundary-plan",
                "sbp": sbp.prepared_id,
                "conditions": [lower_kind, upper_kind],
                "coefficients": list(coefficients),
                "penalties": list(penalties),
            }
        )
        self.sbp = sbp
        self.lower_kind = lower_kind
        self.upper_kind = upper_kind
        self.lower_alpha = coefficients[0]
        self.lower_beta = coefficients[1]
        self.upper_alpha = coefficients[2]
        self.upper_beta = coefficients[3]
        self.lower_penalty = penalties[0]
        self.upper_penalty = penalties[1]
        self.stability_report = (
            FDStabilityReport(
                "sat_stability",
                residual=None,
                tolerance=1e-12,
                assumptions=("user-supplied SAT penalties",),
                evidence="unknown",
                subject_id=identifier,
            )
            if stability_report is None
            else stability_report
        )
        self.plan_id = identifier

    @classmethod
    def advection_inflow(
        cls,
        sbp: PreparedSBPOperator,
        speed: float,
        /,
    ) -> "SATBoundaryPlan":
        speed_ = float(speed)
        if not np.isfinite(speed_) or speed_ == 0.0:
            raise ValueError("Advection inflow SAT requires finite nonzero speed.")
        identifier = canonical_fingerprint(
            {
                "kind": "advection-inflow-sat",
                "sbp": sbp.prepared_id,
                "speed": speed_,
            }
        )
        report = FDStabilityReport(
            "linear_advection_energy",
            residual=0.0,
            tolerance=1e-12,
            assumptions=("constant scalar speed", "homogeneous inflow data"),
            evidence="analytic",
            subject_id=identifier,
        )
        return cls(
            sbp,
            "dirichlet" if speed_ > 0.0 else "none",
            "none" if speed_ > 0.0 else "dirichlet",
            lower_penalty=-speed_ if speed_ > 0.0 else 0.0,
            upper_penalty=speed_ if speed_ < 0.0 else 0.0,
            stability_report=report,
        )

    def _residual(
        self,
        kind: SATConditionKind,
        state: Array,
        derivative: Array,
        target: ArrayLike,
        side: Literal["lower", "upper"],
        /,
    ) -> Array:
        axis = self.sbp.axis_index
        index = 0 if side == "lower" else state.shape[axis] - 1
        state_trace = jnp.take(state, index, axis=axis)
        derivative_trace = jnp.take(derivative, index, axis=axis)
        alpha = self.lower_alpha if side == "lower" else self.upper_alpha
        beta = self.lower_beta if side == "lower" else self.upper_beta
        if kind == "none":
            return jnp.zeros_like(state_trace)
        if kind == "dirichlet":
            return alpha * state_trace - jnp.asarray(target)
        if kind == "neumann":
            return beta * derivative_trace - jnp.asarray(target)
        return alpha * state_trace + beta * derivative_trace - jnp.asarray(target)

    def correction(
        self,
        state: ArrayLike,
        lower_target: ArrayLike = 0.0,
        upper_target: ArrayLike = 0.0,
        /,
    ) -> Array:
        value = self.sbp.operator.source.validate(jnp.asarray(state))
        derivative = self.sbp.operator.mv(value)
        lower_residual = self._residual(
            self.lower_kind,
            value,
            derivative,
            lower_target,
            "lower",
        )
        upper_residual = self._residual(
            self.upper_kind,
            value,
            derivative,
            upper_target,
            "upper",
        )
        correction = jnp.zeros_like(value)
        lower_index: list[slice | int] = [slice(None)] * value.ndim
        upper_index: list[slice | int] = [slice(None)] * value.ndim
        lower_index[self.sbp.axis_index] = 0
        upper_index[self.sbp.axis_index] = value.shape[self.sbp.axis_index] - 1
        correction = correction.at[tuple(lower_index)].set(
            self.lower_penalty * lower_residual
        )
        correction = correction.at[tuple(upper_index)].set(
            self.upper_penalty * upper_residual
        )
        return correction / self.sbp.norm_weights


class SATInterfacePlan(StrictModule, NonTrainableState):
    """Conforming scalar-advection SAT coupling between two SBP blocks."""

    left: PreparedSBPOperator
    right: PreparedSBPOperator
    speed: float = eqx.field(static=True)
    flux: SATInterfaceFlux = eqx.field(static=True)
    stability_report: FDStabilityReport
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        left: PreparedSBPOperator,
        right: PreparedSBPOperator,
        speed: float,
        /,
        *,
        flux: SATInterfaceFlux = "central",
    ):
        if not isinstance(left, PreparedSBPOperator) or not isinstance(
            right, PreparedSBPOperator
        ):
            raise TypeError("SAT interface requires two prepared SBP operators.")
        speed_ = float(speed)
        if (
            not np.isfinite(speed_)
            or speed_ == 0.0
            or flux
            not in (
                "central",
                "upwind",
            )
        ):
            raise ValueError("SAT interface speed/flux is invalid.")
        left_trace_shape = (
            left.grid.shape[: left.axis_index] + left.grid.shape[left.axis_index + 1 :]
        )
        right_trace_shape = (
            right.grid.shape[: right.axis_index]
            + right.grid.shape[right.axis_index + 1 :]
        )
        if left_trace_shape != right_trace_shape:
            raise ValueError("Conforming SAT interface trace shapes must agree.")
        identifier = canonical_fingerprint(
            {
                "kind": "sat-interface-plan",
                "left": left.prepared_id,
                "right": right.prepared_id,
                "speed": speed_,
                "flux": flux,
            }
        )
        self.left = left
        self.right = right
        self.speed = speed_
        self.flux = flux
        self.stability_report = FDStabilityReport(
            "sat_interface_energy",
            residual=0.0,
            tolerance=1e-12,
            assumptions=("conforming traces", "constant scalar advection speed"),
            evidence="analytic",
            subject_id=identifier,
        )
        self.plan_id = identifier

    def corrections(
        self,
        left_state: ArrayLike,
        right_state: ArrayLike,
        /,
    ) -> tuple[Array, Array]:
        left = self.left.operator.source.validate(jnp.asarray(left_state))
        right = self.right.operator.source.validate(jnp.asarray(right_state))
        left_trace = jnp.take(
            left,
            left.shape[self.left.axis_index] - 1,
            axis=self.left.axis_index,
        )
        right_trace = jnp.take(right, 0, axis=self.right.axis_index)
        numerical_flux = (
            0.5 * self.speed * (left_trace + right_trace)
            if self.flux == "central"
            else self.speed * left_trace
            if self.speed > 0.0
            else self.speed * right_trace
        )
        left_residual = self.speed * left_trace - numerical_flux
        right_residual = numerical_flux - self.speed * right_trace
        left_correction = jnp.zeros_like(left)
        right_correction = jnp.zeros_like(right)
        left_index: list[slice | int] = [slice(None)] * left.ndim
        right_index: list[slice | int] = [slice(None)] * right.ndim
        left_index[self.left.axis_index] = left.shape[self.left.axis_index] - 1
        right_index[self.right.axis_index] = 0
        left_correction = left_correction.at[tuple(left_index)].set(left_residual)
        right_correction = right_correction.at[tuple(right_index)].set(right_residual)
        return (
            left_correction / self.left.norm_weights,
            right_correction / self.right.norm_weights,
        )


__all__ = [
    "CompatibleSBPSecondDerivative",
    "PreparedSBPOperator",
    "SATBoundaryPlan",
    "SATConditionKind",
    "SATInterfaceFlux",
    "SATInterfacePlan",
    "SBPDerivativePlan",
    "SBPFamily",
    "SBPInteriorOrder",
]
