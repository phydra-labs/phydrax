#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import core as jax_core
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import (
    AbstractLinearOperator,
    ArraySpace,
    DiagonalPairing,
    OperatorCapabilities,
    OperatorProperties,
)
from .._tensor_entities import AxisEntityKind, TensorEntityLayout
from .._tensor_support import PreparedTensorGrid
from ._certification import FDConservationReport, FDStabilityReport
from ._sbp import PreparedSBPOperator, SBPDerivativePlan, SBPInteriorOrder


MappedMetricMode: TypeAlias = Literal["discrete_curl"]


class MappedMetricIdentityReport(StrictModule, NonTrainableState):
    """Jacobian validity, metric identity, map consistency, and free-stream evidence."""

    minimum_jacobian: float = eqx.field(static=True)
    metric_identity_residual: float = eqx.field(static=True)
    map_derivative_residual: float = eqx.field(static=True)
    free_stream_residual: float = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    passed: bool = eqx.field(static=True)
    report_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        minimum_jacobian: float,
        metric_identity_residual: float,
        map_derivative_residual: float,
        free_stream_residual: float,
        tolerance: float,
        mapped_id: str,
    ):
        minimum = float(minimum_jacobian)
        metric = float(metric_identity_residual)
        map_residual = float(map_derivative_residual)
        free_stream = float(free_stream_residual)
        tolerance_ = float(tolerance)
        if (
            not all(
                np.isfinite(value)
                for value in (minimum, metric, map_residual, free_stream, tolerance_)
            )
            or tolerance_ <= 0.0
        ):
            raise ValueError(
                "Mapped metric diagnostics must be finite with positive tolerance."
            )
        self.minimum_jacobian = minimum
        self.metric_identity_residual = metric
        self.map_derivative_residual = map_residual
        self.free_stream_residual = free_stream
        self.tolerance = tolerance_
        self.passed = minimum > 0.0 and max(metric, free_stream) <= tolerance_
        self.report_id = canonical_fingerprint(
            {
                "kind": "mapped-metric-identity-report",
                "mapped": mapped_id,
                "minimum_jacobian": minimum,
                "metric_identity_residual": metric,
                "map_derivative_residual": map_residual,
                "free_stream_residual": free_stream,
                "tolerance": tolerance_,
            }
        )


class MappedTensorGridPlan(StrictModule):
    """Stationary physical embedding of a point-primary reference tensor grid."""

    reference_grid: PreparedTensorGrid
    coordinate_map: Callable[[Array], ArrayLike]
    metric_mode: MappedMetricMode = eqx.field(static=True)
    sbp_order: SBPInteriorOrder = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        reference_grid: PreparedTensorGrid,
        coordinate_map: Callable[[Array], ArrayLike],
        /,
        *,
        sbp_order: SBPInteriorOrder = 4,
        metric_mode: MappedMetricMode = "discrete_curl",
    ):
        if not isinstance(reference_grid, PreparedTensorGrid) or not callable(
            coordinate_map
        ):
            raise TypeError("Mapped tensor plan requires a grid and coordinate map.")
        dimension = len(reference_grid.shape)
        if dimension not in (1, 2, 3):
            raise ValueError("Mapped tensor grids support one, two, or three dimensions.")
        if any(
            axis.primary_entity != "point" or axis.periodic
            for axis in reference_grid.structured_axes
        ):
            raise ValueError(
                "Initial mapped tensor grids require bounded point-primary axes."
            )
        if metric_mode != "discrete_curl":
            raise ValueError("Unknown mapped metric mode.")
        self.reference_grid = reference_grid
        self.coordinate_map = coordinate_map
        self.metric_mode = metric_mode
        self.sbp_order = sbp_order
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mapped-tensor-grid-plan",
                "coordinate_map_type": type(coordinate_map).__name__,
                "coordinate_map_arrays": array_tree_fingerprint(coordinate_map),
                "reference_grid": reference_grid.prepared_id,
                "coordinate_map": repr(coordinate_map),
                "metric_mode": metric_mode,
                "sbp_order": int(sbp_order),
            }
        )

    def prepare(self, /) -> "PreparedMappedTensorGrid":
        return PreparedMappedTensorGrid(self)


def _differentiate(
    derivative: PreparedSBPOperator,
    value: Array,
    /,
) -> Array:
    return derivative.operator.mv(value)


def _discrete_cofactor(
    physical: Array,
    derivatives: tuple[PreparedSBPOperator, ...],
    /,
) -> tuple[Array, Array]:
    dimension = physical.shape[-1]
    deformation = jnp.stack(
        [
            jnp.stack(
                [
                    _differentiate(derivative, physical[..., component])
                    for derivative in derivatives
                ],
                axis=-1,
            )
            for component in range(dimension)
        ],
        axis=-2,
    )
    if dimension == 1:
        cofactor = jnp.ones_like(deformation)
    elif dimension == 2:
        cofactor = jnp.stack(
            (
                jnp.stack((deformation[..., 1, 1], -deformation[..., 1, 0]), axis=-1),
                jnp.stack((-deformation[..., 0, 1], deformation[..., 0, 0]), axis=-1),
            ),
            axis=-2,
        )
    else:
        cofactor_components = []
        physical_cycles = ((0, 1, 2), (1, 2, 0), (2, 0, 1))
        reference_cycles = ((0, 1, 2), (1, 2, 0), (2, 0, 1))
        for _, first_physical, second_physical in physical_cycles:
            row = []
            for _, first_reference, second_reference in reference_cycles:
                first_potential = (
                    physical[..., first_physical]
                    * deformation[..., second_physical, second_reference]
                    - physical[..., second_physical]
                    * deformation[..., first_physical, second_reference]
                )
                second_potential = (
                    physical[..., first_physical]
                    * deformation[..., second_physical, first_reference]
                    - physical[..., second_physical]
                    * deformation[..., first_physical, first_reference]
                )
                row.append(
                    0.5
                    * (
                        _differentiate(
                            derivatives[first_reference],
                            first_potential,
                        )
                        - _differentiate(
                            derivatives[second_reference],
                            second_potential,
                        )
                    )
                )
            cofactor_components.append(jnp.stack(row, axis=-1))
        cofactor = jnp.stack(cofactor_components, axis=-2)
    return deformation, cofactor


def _interpolate_nodal_to_faces(
    grid: PreparedTensorGrid,
    values: Array,
    axis_index: int,
    /,
) -> Array:
    axis = grid.structured_axes[axis_index]
    if axis.periodic:
        return 0.5 * (values + jnp.roll(values, 1, axis=axis_index))
    left = jnp.take(
        values,
        jnp.arange(values.shape[axis_index] - 1),
        axis=axis_index,
    )
    right = jnp.take(
        values,
        jnp.arange(1, values.shape[axis_index]),
        axis=axis_index,
    )
    return 0.5 * (left + right)


def _dual_face_layout(
    grid: PreparedTensorGrid,
    axis_index: int,
    /,
) -> TensorEntityLayout:
    entities: list[AxisEntityKind] = ["point"] * len(grid.shape)
    entities[axis_index] = "interval"
    return grid.entity_layout(entities)


def _reference_face_measure(
    grid: PreparedTensorGrid,
    axis_index: int,
    /,
) -> Array:
    layout = _dual_face_layout(grid, axis_index)
    result = jnp.ones(layout.shape)
    for index, (axis, entity) in enumerate(
        zip(grid.structured_axes, layout.axis_entities, strict=True)
    ):
        weights = (
            jnp.ones((axis.count(entity),))
            if index == axis_index
            else axis.measure(entity)
        )
        reshape = [1] * len(layout.shape)
        reshape[index] = int(weights.size)
        result = result * weights.reshape(reshape)
    return result


def evaluate_mapped_metrics(
    reference_grid: PreparedTensorGrid,
    coordinate_map: Callable[[Array], ArrayLike],
    /,
    *,
    sbp_order: SBPInteriorOrder = 4,
) -> tuple[Array, Array, Array, Array]:
    """Pure differentiable mapped coordinates, deformation, cofactor, and Jacobian."""
    if not isinstance(reference_grid, PreparedTensorGrid) or not callable(coordinate_map):
        raise TypeError("Mapped metric evaluation requires a grid and coordinate map.")
    dimension = len(reference_grid.shape)
    physical = jax.vmap(coordinate_map)(reference_grid.points)
    physical = jnp.asarray(physical)
    if physical.shape != (reference_grid.size, dimension):
        raise ValueError("Coordinate map must return one physical vector per point.")
    physical = physical.reshape(reference_grid.shape + (dimension,))
    derivatives = tuple(
        SBPDerivativePlan(
            reference_grid,
            axis,
            interior_order=sbp_order,
        ).prepare()
        for axis in reference_grid.axis_names
    )
    deformation, cofactor = _discrete_cofactor(physical, derivatives)
    jacobian = jnp.sum(deformation * cofactor, axis=(-2, -1)) / float(dimension)
    return physical, deformation, cofactor, jacobian


class PreparedMappedTensorGrid(StrictModule, NonTrainableState):
    """Stationary mapped geometry with discrete metric identities and operators."""

    plan: MappedTensorGridPlan
    reference_grid: PreparedTensorGrid
    derivatives: tuple[PreparedSBPOperator, ...]
    physical_coordinates: Array
    deformation_gradient: Array
    cofactor: Array
    jacobian: Array
    inverse_deformation: Array
    physical_measure: Array
    dual_face_layouts: tuple[TensorEntityLayout, ...]
    face_normals: tuple[Array, ...]
    face_measures: tuple[Array, ...]
    metric_report: MappedMetricIdentityReport
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: MappedTensorGridPlan, /):
        if not isinstance(plan, MappedTensorGridPlan):
            raise TypeError("plan must be a MappedTensorGridPlan.")
        grid = plan.reference_grid
        dimension = len(grid.shape)
        physical_flat = jax.vmap(plan.coordinate_map)(grid.points)
        physical_flat = jnp.asarray(physical_flat)
        if physical_flat.shape != (grid.size, dimension):
            raise ValueError("Coordinate map must return one physical vector per point.")
        physical = physical_flat.reshape(grid.shape + (dimension,))
        derivatives = tuple(
            SBPDerivativePlan(
                grid,
                axis,
                interior_order=plan.sbp_order,
            ).prepare()
            for axis in grid.axis_names
        )
        deformation, cofactor = _discrete_cofactor(physical, derivatives)
        jacobian = jnp.sum(deformation * cofactor, axis=(-2, -1)) / float(dimension)
        invalid_jacobian = jnp.any(~jnp.isfinite(jacobian)) | jnp.any(jacobian <= 0.0)
        if not isinstance(invalid_jacobian, jax_core.Tracer) and bool(invalid_jacobian):
            raise eqx.EquinoxRuntimeError(
                "Mapped tensor Jacobian must be finite and positive."
            )
        jacobian = eqx.error_if(
            jacobian,
            invalid_jacobian,
            "Mapped tensor Jacobian must be finite and positive.",
        )
        inverse = jnp.swapaxes(cofactor, -1, -2) / jacobian[..., None, None]
        physical_measure = grid.quadrature_weights * jacobian
        metric_divergence = []
        for physical_axis in range(dimension):
            residual = jnp.zeros(grid.shape)
            for reference_axis, derivative in enumerate(derivatives):
                residual = residual + derivative.operator.mv(
                    cofactor[..., physical_axis, reference_axis]
                )
            metric_divergence.append(residual)
        metric_residual = jnp.max(jnp.abs(jnp.stack(metric_divergence, axis=-1)))
        free_stream = jnp.max(
            jnp.abs(
                jnp.stack(
                    [
                        sum(
                            cofactor[..., physical_axis, reference_axis]
                            * derivative.operator.mv(jnp.ones(grid.shape))
                            for reference_axis, derivative in enumerate(derivatives)
                        )
                        / jacobian
                        for physical_axis in range(dimension)
                    ],
                    axis=-1,
                )
            )
        )
        analytic_deformation = jax.vmap(jax.jacfwd(plan.coordinate_map))(grid.points)
        analytic_deformation = jnp.asarray(analytic_deformation).reshape(
            grid.shape + (dimension, dimension)
        )
        map_scale = jnp.maximum(1.0, jnp.max(jnp.abs(analytic_deformation)))
        map_residual = jnp.max(jnp.abs(deformation - analytic_deformation)) / map_scale
        dual_face_layouts = tuple(
            _dual_face_layout(grid, axis_index) for axis_index in range(dimension)
        )
        face_normals = []
        face_measures = []
        for axis_index in range(dimension):
            cofactor_face = _interpolate_nodal_to_faces(
                grid,
                cofactor[..., :, axis_index],
                axis_index,
            )
            magnitude = jnp.linalg.norm(cofactor_face, axis=-1)
            magnitude = eqx.error_if(
                magnitude,
                jnp.any(~jnp.isfinite(magnitude)) | jnp.any(magnitude <= 0.0),
                "Mapped face metric must be finite and positive.",
            )
            face_normals.append(cofactor_face / magnitude[..., None])
            face_measures.append(_reference_face_measure(grid, axis_index) * magnitude)
        prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-mapped-tensor-grid",
                "plan": plan.plan_id,
                "reference_derivatives": [value.prepared_id for value in derivatives],
                "shape": list(grid.shape),
            }
        )
        report = MappedMetricIdentityReport(
            minimum_jacobian=float(np.min(np.asarray(jacobian))),
            metric_identity_residual=float(np.asarray(metric_residual)),
            map_derivative_residual=float(np.asarray(map_residual)),
            free_stream_residual=float(np.asarray(free_stream)),
            tolerance=5e-10,
            mapped_id=prepared_id,
        )
        self.plan = plan
        self.reference_grid = grid
        self.derivatives = derivatives
        self.physical_coordinates = physical
        self.deformation_gradient = deformation
        self.cofactor = cofactor
        self.jacobian = jacobian
        self.inverse_deformation = inverse
        self.physical_measure = physical_measure
        self.dual_face_layouts = dual_face_layouts
        self.face_normals = tuple(face_normals)
        self.face_measures = tuple(face_measures)
        self.metric_report = report
        self.prepared_id = prepared_id

    @property
    def shape(self) -> tuple[int, ...]:
        return self.reference_grid.shape

    def gradient(self, scalar: ArrayLike, /) -> Array:
        value = jnp.asarray(scalar)
        if value.shape != self.shape:
            raise ValueError("Mapped scalar field must match the reference grid shape.")
        reference_gradient = jnp.stack(
            [derivative.operator.mv(value) for derivative in self.derivatives],
            axis=-1,
        )
        return (
            ein.contract(
                "...ij,...j->...i",
                self.cofactor,
                reference_gradient,
            )
            / self.jacobian[..., None]
        )

    def divergence(self, vector: ArrayLike, /) -> Array:
        value = jnp.asarray(vector)
        dimension = len(self.shape)
        if value.shape != self.shape + (dimension,):
            raise ValueError("Mapped vector must have one trailing physical component.")
        contravariant_flux = ein.contract("...ij,...i->...j", self.cofactor, value)
        result = jnp.zeros(self.shape, dtype=value.dtype)
        for axis, derivative in enumerate(self.derivatives):
            result = result + derivative.operator.mv(contravariant_flux[..., axis])
        return result / self.jacobian

    def laplacian(self, scalar: ArrayLike, /) -> Array:
        return self.divergence(self.gradient(scalar))

    def integral(self, scalar: ArrayLike, /) -> Array:
        value = jnp.asarray(scalar)
        if value.shape != self.shape:
            raise ValueError("Mapped integral field must match the grid shape.")
        return jnp.sum(self.physical_measure * value)

    def diffusion(self, coefficient: ArrayLike = 1.0, /) -> "MappedDiffusionOperator":
        return MappedDiffusionOperator(self, coefficient)


class MappedDiffusionOperator(AbstractLinearOperator):
    """Physical tensor diffusion in conservative mapped divergence form."""

    source: ArraySpace
    target: ArraySpace
    mapped_grid: PreparedMappedTensorGrid
    coefficient: Array
    conservation_report: FDConservationReport
    stability_report: FDStabilityReport

    def __init__(
        self,
        mapped_grid: PreparedMappedTensorGrid,
        coefficient: ArrayLike = 1.0,
        /,
    ):
        if not isinstance(mapped_grid, PreparedMappedTensorGrid):
            raise TypeError("mapped_grid must be PreparedMappedTensorGrid.")
        dimension = len(mapped_grid.shape)
        value = jnp.asarray(coefficient)
        if value.shape == () or value.shape == mapped_grid.shape:
            scalar = jnp.broadcast_to(value, mapped_grid.shape)
            coefficient_ = scalar[..., None, None] * jnp.eye(
                dimension,
                dtype=scalar.dtype,
            )
        elif value.shape == (dimension, dimension):
            coefficient_ = jnp.broadcast_to(
                value,
                mapped_grid.shape + (dimension, dimension),
            )
        elif value.shape == mapped_grid.shape + (dimension, dimension):
            coefficient_ = value
        else:
            raise ValueError(
                "Mapped coefficient must be scalar, cell scalar, or physical tensor."
            )
        host = np.asarray(coefficient_)
        if np.any(~np.isfinite(host)):
            raise ValueError("Mapped diffusion coefficient must be finite.")
        base = mapped_grid.reference_grid.field_space("mapped_diffusion")
        if not isinstance(base.vector_space, ArraySpace):
            raise TypeError("Mapped diffusion requires an ArraySpace field.")
        space = ArraySpace(
            mapped_grid.shape,
            dtype=base.vector_space.dtype,
            pairing=DiagonalPairing(mapped_grid.physical_measure),
        )
        self.source = space
        self.target = space
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
                "kind": "mapped-diffusion-operator",
                "mapped_grid": mapped_grid.prepared_id,
                "coefficient_shape": list(coefficient_.shape),
            }
        )
        self.mapped_grid = mapped_grid
        self.coefficient = coefficient_
        constant_residual = float(
            np.max(np.abs(np.asarray(self.mv(jnp.ones(mapped_grid.shape)))))
        )
        self.conservation_report = FDConservationReport(
            constant_state_residual=constant_residual,
            global_balance_residual=None,
            tolerance=1e-10,
            operator_id=self.operator_id,
        )
        self.stability_report = FDStabilityReport(
            "mapped_diffusion_stability",
            residual=None,
            tolerance=1e-10,
            assumptions=("boundary SAT supplied separately",),
            evidence="unknown",
            subject_id=self.operator_id,
        )

    def mv(self, vector: ArrayLike, /) -> Array:
        value = self.source.validate(jnp.asarray(vector))
        gradient = self.mapped_grid.gradient(value)
        flux = ein.contract("...ij,...j->...i", self.coefficient, gradient)
        return self.mapped_grid.divergence(flux)

    def transpose_mv(self, vector: ArrayLike, /) -> Array:
        value = self.target.validate(jnp.asarray(vector))
        zero = jnp.zeros(self.source.shape, dtype=self.source.dtype)
        return jax.linear_transpose(self.mv, zero)(value)[0]

    def adjoint_mv(self, vector: ArrayLike, /) -> Array:
        value = self.target.validate(jnp.asarray(vector))
        target_pairing = self.target.pairing
        source_pairing = self.source.pairing
        if not isinstance(target_pairing, DiagonalPairing) or not isinstance(
            source_pairing, DiagonalPairing
        ):
            raise RuntimeError("Mapped diffusion lost its physical diagonal pairing.")
        weighted = target_pairing.weights * value
        transposed = self.transpose_mv(weighted)
        return transposed / source_pairing.weights

    def _materialize(self, /) -> Array:
        if self.source.size * self.target.size > 4096**2:
            raise ValueError("Mapped diffusion materialization exceeds size budget.")
        identity = jnp.eye(self.source.size, dtype=self.source.dtype).reshape(
            (self.source.size,) + self.source.shape
        )
        columns = jax.vmap(self.mv)(identity).reshape((self.source.size, -1))
        return columns.T


__all__ = [
    "evaluate_mapped_metrics",
    "MappedDiffusionOperator",
    "MappedMetricIdentityReport",
    "MappedMetricMode",
    "MappedTensorGridPlan",
    "PreparedMappedTensorGrid",
]
