#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import prod

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._fingerprint import canonical_fingerprint
from ..._interpolation._rational_spline import RationalSplineJet
from ..._interpolation._tensor_bspline import TensorBSplineJetPlan
from ...linalg import inverse_small_linear, SmallLinearSolvePlan
from .._local_variational import (
    LocalGeometryActions,
    LocalMetricResult,
    LocalReferenceActions,
)
from ._geometry import (
    IsogeometricGeometryEvidence,
    IsogeometricH1QualificationPolicy,
    IsogeometricRuntimeData,
)


def _query_view(
    values: Array,
    permutation: tuple[int, ...],
    entity_shape: tuple[int, ...],
    point_shape: tuple[int, ...],
    tail_rank: int,
    /,
) -> Array:
    query_rank = values.ndim - tail_rank
    tail_axes = tuple(range(query_rank, values.ndim))
    reordered = jnp.transpose(values, permutation + tail_axes)
    tail_shape = values.shape[query_rank:]
    return reordered.reshape((prod(entity_shape), prod(point_shape)) + tail_shape)


def _runtime(
    runtime: object,
    /,
    *,
    topology_id: str,
    geometry_layout_id: str,
) -> IsogeometricRuntimeData:
    if not isinstance(runtime, IsogeometricRuntimeData):
        raise TypeError("IGA local actions require IsogeometricRuntimeData.")
    if (
        runtime.topology_id != topology_id
        or runtime.geometry_layout_id != geometry_layout_id
    ):
        raise ValueError("IGA runtime does not match the prepared topology and layout.")
    return runtime


class IsogeometricReferenceActions(LocalReferenceActions):
    """Runtime-rational tensor-spline interpolation with exact local transposes."""

    tensor_plan: TensorBSplineJetPlan
    field_weights: Array | None
    field_weights_from_geometry: bool = eqx.field(static=True)
    entity_rows: Array
    query_permutation: tuple[int, ...] = eqx.field(static=True)
    entity_shape: tuple[int, ...] = eqx.field(static=True)
    point_shape: tuple[int, ...] = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    geometry_layout_id: str = eqx.field(static=True)
    action_id: str = eqx.field(static=True)
    realization_id: str = eqx.field(static=True)
    local_width: int = eqx.field(static=True)
    point_count: int = eqx.field(static=True)
    maximum_derivative_order: int = eqx.field(static=True)
    kernel_modes: tuple[str, ...] = eqx.field(static=True)
    is_trace: bool = eqx.field(static=True)

    def __init__(
        self,
        tensor_plan: TensorBSplineJetPlan,
        field_weights: ArrayLike | None,
        entity_rows: ArrayLike,
        query_permutation: tuple[int, ...],
        entity_shape: tuple[int, ...],
        point_shape: tuple[int, ...],
        /,
        *,
        topology_id: str,
        geometry_layout_id: str,
        maximum_derivative_order: int,
        structural_id: str,
        is_trace: bool,
        field_weights_from_geometry: bool = False,
    ):
        if not isinstance(tensor_plan, TensorBSplineJetPlan):
            raise TypeError("tensor_plan must be a TensorBSplineJetPlan.")
        rows = jnp.asarray(entity_rows, dtype=jnp.int32)
        weights = None if field_weights is None else jnp.asarray(field_weights)
        dynamic_weights = bool(field_weights_from_geometry)
        if dynamic_weights and weights is not None:
            raise ValueError(
                "Geometry-owned IGA field weights cannot also be supplied statically."
            )
        if weights is not None and weights.shape != tensor_plan.source_shape:
            raise ValueError("IGA field rational weights do not match its tensor basis.")
        entity_count = prod(entity_shape)
        if rows.ndim != 1 or jnp.issubdtype(rows.dtype, jnp.bool_):
            raise ValueError("IGA reference entity rows must be rank-1 integer routes.")
        order = int(maximum_derivative_order)
        if order < 0 or order > 2:
            raise ValueError(
                "IGA reference actions support derivative orders zero through two."
            )
        local_width = tensor_plan.local_size
        self.tensor_plan = tensor_plan
        self.field_weights = weights
        self.field_weights_from_geometry = dynamic_weights
        self.entity_rows = rows
        self.query_permutation = tuple(int(value) for value in query_permutation)
        self.entity_shape = tuple(int(value) for value in entity_shape)
        self.point_shape = tuple(int(value) for value in point_shape)
        self.topology_id = str(topology_id)
        self.realization_id = "isogeometric-direct-tensor"
        self.geometry_layout_id = str(geometry_layout_id)
        self.local_width = local_width
        self.point_count = prod(self.point_shape)
        self.maximum_derivative_order = order
        self.kernel_modes = ("sum_factorized",)
        self.is_trace = bool(is_trace)
        self.action_id = canonical_fingerprint(
            {
                "kind": "isogeometric-reference-actions",
                "structural": str(structural_id),
                "field_weight_kind": (
                    "geometry"
                    if dynamic_weights
                    else "polynomial"
                    if weights is None
                    else "rational"
                ),
                "field_weights": (
                    None
                    if weights is None
                    else canonical_fingerprint({"weights": weights.tolist()})
                ),
                "tensor_source_shape": list(tensor_plan.source_shape),
                "tensor_query_shape": list(tensor_plan.query_shape),
                "tensor_multi_indices": [
                    list(value) for value in tensor_plan.multi_indices
                ],
                "topology": self.topology_id,
                "geometry_layout": self.geometry_layout_id,
                "maximum_derivative_order": order,
                "entity_count": entity_count,
                "entity_rows": tuple(int(value) for value in rows.tolist()),
                "trace": self.is_trace,
            }
        )

    def _rational(self, runtime: object, /) -> RationalSplineJet:
        runtime_ = _runtime(
            runtime,
            topology_id=self.topology_id,
            geometry_layout_id=self.geometry_layout_id,
        )
        if self.field_weights_from_geometry:
            if runtime_.weights.shape != self.tensor_plan.source_shape:
                raise ValueError(
                    "Dynamic geometry weights do not match the field tensor basis."
                )
            weights = runtime_.weights
        else:
            weights = (
                jnp.ones(self.tensor_plan.source_shape)
                if self.field_weights is None
                else self.field_weights
            )
        return RationalSplineJet(self.tensor_plan, weights)

    def _values(self, runtime: object, /) -> Array:
        rational = self._rational(runtime)
        values = _query_view(
            rational.values,
            self.query_permutation,
            self.entity_shape,
            self.point_shape,
            1,
        )
        return values[self.entity_rows]

    def _gradients(self, runtime: object, /) -> Array:
        rational = self._rational(runtime)
        gradients = _query_view(
            rational.gradients,
            self.query_permutation,
            self.entity_shape,
            self.point_shape,
            2,
        )
        return gradients[self.entity_rows]

    def _hessians(self, runtime: object, /) -> Array:
        rational = self._rational(runtime)
        hessians = _query_view(
            rational.hessians,
            self.query_permutation,
            self.entity_shape,
            self.point_shape,
            3,
        )
        return hessians[self.entity_rows]

    def realize_reference_actions(
        self, runtime: object, /
    ) -> IsogeometricReferenceActions:
        self._rational(runtime)
        return self

    def interpolate(self, runtime: object, local_coefficients: ArrayLike, /) -> Array:
        coefficients = jnp.asarray(local_coefficients)
        if coefficients.shape[:2] != (self.entity_rows.size, self.local_width):
            raise ValueError("IGA local coefficients do not match reference actions.")
        return ein.contract("eql,el...->eq...", self._values(runtime), coefficients)

    def interpolate_transpose(self, runtime: object, values: ArrayLike, /) -> Array:
        values_ = jnp.asarray(values)
        if values_.shape[:2] != (self.entity_rows.size, self.point_count):
            raise ValueError("IGA point values do not match reference actions.")
        return ein.contract("eql,eq...->el...", self._values(runtime), values_)

    def reference_gradient(
        self, runtime: object, local_coefficients: ArrayLike, /
    ) -> Array:
        coefficients = jnp.asarray(local_coefficients)
        if coefficients.shape[:2] != (self.entity_rows.size, self.local_width):
            raise ValueError("IGA local coefficients do not match reference actions.")
        return ein.contract("eqlr,el...->eq...r", self._gradients(runtime), coefficients)

    def reference_gradient_transpose(
        self, runtime: object, gradients: ArrayLike, /
    ) -> Array:
        gradients_ = jnp.asarray(gradients)
        if (
            gradients_.shape[:2] != (self.entity_rows.size, self.point_count)
            or gradients_.shape[-1] != self.tensor_plan.dimension
        ):
            raise ValueError("IGA point gradients do not match reference actions.")
        return ein.contract("eqlr,eq...r->el...", self._gradients(runtime), gradients_)

    def reference_hessian(
        self, runtime: object, local_coefficients: ArrayLike, /
    ) -> Array:
        coefficients = jnp.asarray(local_coefficients)
        if coefficients.shape[:2] != (self.entity_rows.size, self.local_width):
            raise ValueError("IGA local coefficients do not match reference actions.")
        return ein.contract("eqlrs,el...->eq...rs", self._hessians(runtime), coefficients)

    def reference_hessian_transpose(
        self, runtime: object, hessians: ArrayLike, /
    ) -> Array:
        hessians_ = jnp.asarray(hessians)
        expected = (self.tensor_plan.dimension, self.tensor_plan.dimension)
        if (
            hessians_.shape[:2] != (self.entity_rows.size, self.point_count)
            or hessians_.shape[-2:] != expected
        ):
            raise ValueError("IGA point Hessians do not match reference actions.")
        return ein.contract("eqlrs,eq...rs->el...", self._hessians(runtime), hessians_)

    def trace(self, runtime: object, local_coefficients: ArrayLike, /) -> Array:
        if not self.is_trace:
            raise ValueError("Trace actions require an exterior-facet IGA region.")
        return self.interpolate(runtime, local_coefficients)

    def trace_transpose(self, runtime: object, values: ArrayLike, /) -> Array:
        if not self.is_trace:
            raise ValueError("Trace actions require an exterior-facet IGA region.")
        return self.interpolate_transpose(runtime, values)


class IsogeometricGeometryActions(LocalGeometryActions):
    """Runtime NURBS geometry and metric realization for one aligned region."""

    tensor_plan: TensorBSplineJetPlan
    entity_rows: Array
    reference_weights: Array
    query_permutation: tuple[int, ...] = eqx.field(static=True)
    entity_shape: tuple[int, ...] = eqx.field(static=True)
    point_shape: tuple[int, ...] = eqx.field(static=True)
    parameter_scales: Array
    qualification_policy: IsogeometricH1QualificationPolicy
    topology_id: str = eqx.field(static=True)
    runtime_layout_id: str = eqx.field(static=True)
    action_id: str = eqx.field(static=True)
    entity_count: int = eqx.field(static=True)
    domain_kind: str = eqx.field(static=True)
    facet_axis: int = eqx.field(static=True)
    facet_side: int = eqx.field(static=True)

    def __init__(
        self,
        tensor_plan: TensorBSplineJetPlan,
        entity_rows: ArrayLike,
        reference_weights: ArrayLike,
        query_permutation: tuple[int, ...],
        entity_shape: tuple[int, ...],
        point_shape: tuple[int, ...],
        parameter_scales: ArrayLike,
        qualification_policy: IsogeometricH1QualificationPolicy,
        /,
        *,
        topology_id: str,
        runtime_layout_id: str,
        domain_kind: str,
        structural_id: str,
        facet_axis: int = -1,
        facet_side: int = 0,
    ):
        if not isinstance(tensor_plan, TensorBSplineJetPlan):
            raise TypeError("tensor_plan must be a TensorBSplineJetPlan.")
        if not isinstance(qualification_policy, IsogeometricH1QualificationPolicy):
            raise TypeError("qualification_policy is invalid.")
        kind = str(domain_kind)
        if kind not in ("cell", "exterior_facet"):
            raise ValueError("S1 IGA geometry supports only cell and exterior facets.")
        rows = jnp.asarray(entity_rows, dtype=jnp.int32)
        weights = jnp.asarray(reference_weights)
        if weights.shape != (rows.size, prod(point_shape)):
            raise ValueError("IGA reference weights do not match region entities.")
        self.tensor_plan = tensor_plan
        self.entity_rows = rows
        self.reference_weights = weights
        self.query_permutation = tuple(int(value) for value in query_permutation)
        self.entity_shape = tuple(int(value) for value in entity_shape)
        self.point_shape = tuple(int(value) for value in point_shape)
        self.parameter_scales = jnp.asarray(parameter_scales)
        self.qualification_policy = qualification_policy
        self.topology_id = str(topology_id)
        self.runtime_layout_id = str(runtime_layout_id)
        self.entity_count = int(rows.size)
        self.domain_kind = kind
        self.facet_axis = int(facet_axis)
        self.facet_side = int(facet_side)
        self.action_id = canonical_fingerprint(
            {
                "kind": "isogeometric-geometry-actions",
                "structural": str(structural_id),
                "tensor_source_shape": list(tensor_plan.source_shape),
                "tensor_query_shape": list(tensor_plan.query_shape),
                "tensor_multi_indices": [
                    list(value) for value in tensor_plan.multi_indices
                ],
                "topology": self.topology_id,
                "runtime_layout": self.runtime_layout_id,
                "domain_kind": kind,
                "facet_axis": self.facet_axis,
                "facet_side": self.facet_side,
                "qualification": qualification_policy.policy_id,
                "entity_rows": tuple(int(value) for value in rows.tolist()),
            }
        )

    def _rational_data(
        self, runtime: IsogeometricRuntimeData, /
    ) -> tuple[Array, Array, Array, Array]:
        rational = RationalSplineJet(self.tensor_plan, runtime.weights)
        values = _query_view(
            rational.values,
            self.query_permutation,
            self.entity_shape,
            self.point_shape,
            1,
        )[self.entity_rows]
        gradients = _query_view(
            rational.gradients,
            self.query_permutation,
            self.entity_shape,
            self.point_shape,
            2,
        )[self.entity_rows]
        hessians = _query_view(
            rational.hessians,
            self.query_permutation,
            self.entity_shape,
            self.point_shape,
            3,
        )[self.entity_rows]
        indices = _query_view(
            rational.indices,
            self.query_permutation,
            self.entity_shape,
            self.point_shape,
            1,
        )[self.entity_rows]
        return values, gradients, hessians, indices

    def _metric(self, runtime: object, /) -> tuple[LocalMetricResult, Array]:
        runtime_ = _runtime(
            runtime,
            topology_id=self.topology_id,
            geometry_layout_id=self.runtime_layout_id,
        )
        values, gradients, hessians, indices = self._rational_data(runtime_)
        local_points = runtime_.control_points.reshape(
            (-1, runtime_.control_points.shape[-1])
        )[indices]
        points = ein.contract("eql,eqld->eqd", values, local_points)
        jacobian = ein.contract("eqlr,eqld->eqdr", gradients, local_points)
        mapping_hessian = ein.contract("eqlrs,eqld->eqdrs", hessians, local_points)
        metric = ein.contract("eqdi,eqdj->eqij", jacobian, jacobian)
        inverse_result = inverse_small_linear(
            SmallLinearSolvePlan(metric.shape[-1]),
            metric,
        )
        inverse_metric = eqx.error_if(
            inverse_result.value,
            jnp.any(~inverse_result.successful),
            "Isogeometric metric inversion failed.",
        )
        inverse_jacobian = ein.contract("eqij,eqdj->eqid", inverse_metric, jacobian)
        determinant = inverse_result.determinant
        volume_measure = jnp.sqrt(jnp.maximum(determinant, 0.0))
        normals = None
        physical_weights = self.reference_weights * volume_measure
        if self.domain_kind == "exterior_facet":
            covector = (
                jnp.zeros((jacobian.shape[-1],), dtype=jacobian.dtype)
                .at[self.facet_axis]
                .set(float(self.facet_side))
            )
            normal_vector = ein.contract(
                "eqdr,eqrs,s->eqd", jacobian, inverse_metric, covector
            )
            normal_scale = jnp.linalg.norm(normal_vector, axis=-1)
            safe_normal_scale = jnp.where(
                normal_scale > 0.0, normal_scale, jnp.ones_like(normal_scale)
            )
            normals = normal_vector / safe_normal_scale[..., None]
            physical_weights = physical_weights * normal_scale
        inverse_hessian = None
        if jacobian.shape[-2] == jacobian.shape[-1]:
            inverse_hessian = -ein.contract(
                "eqrd,eqdst,eqsa,eqtb->eqrab",
                inverse_jacobian,
                mapping_hessian,
                inverse_jacobian,
                inverse_jacobian,
            )
        result = LocalMetricResult(
            points,
            physical_weights,
            jacobian,
            inverse_jacobian,
            inverse_hessian=inverse_hessian,
            normals=normals,
        )
        zero = (0,) * self.tensor_plan.dimension
        polynomial_values = self.tensor_plan.basis(zero)
        polynomial_indices = self.tensor_plan.tensor_indices
        local_weights = runtime_.weights.reshape((-1,))[polynomial_indices]
        weight_sum = ein.contract("...l,...l->...", polynomial_values, local_weights)
        weight_sum = _query_view(
            weight_sum,
            self.query_permutation,
            self.entity_shape,
            self.point_shape,
            0,
        )[self.entity_rows]
        return result, weight_sum

    def evidence(self, runtime: object, /) -> IsogeometricGeometryEvidence:
        runtime_ = _runtime(
            runtime,
            topology_id=self.topology_id,
            geometry_layout_id=self.runtime_layout_id,
        )
        metric, weight_sum = self._metric(runtime_)
        points = runtime_.control_points.reshape((-1, runtime_.control_points.shape[-1]))
        center = jnp.mean(points, axis=0)
        coordinate_scale = 2.0 * jnp.max(jnp.linalg.norm(points - center, axis=-1))
        tiny = jnp.finfo(points.dtype).tiny
        safe_scale = jnp.maximum(coordinate_scale, tiny)
        scaled_jacobian = metric.jacobian * self.parameter_scales
        gram = ein.contract("eqdi,eqdj->eqij", scaled_jacobian, scaled_jacobian)
        eigenvalues = jnp.linalg.eigvalsh(gram)
        minimum_rank_ratio = jnp.min(eigenvalues[..., 0]) / (safe_scale * safe_scale)
        maximum_weight = jnp.maximum(jnp.max(jnp.abs(weight_sum)), tiny)
        minimum_weight_ratio = jnp.min(weight_sum) / maximum_weight
        ambient = int(metric.jacobian.shape[-2])
        parametric = int(metric.jacobian.shape[-1])
        if ambient == parametric:
            determinant = jnp.linalg.det(scaled_jacobian)
            absolute = jnp.sqrt(jnp.maximum(jnp.linalg.det(gram), 0.0))
            safe_absolute = jnp.where(absolute > 0.0, absolute, jnp.ones_like(absolute))
            minimum_orientation_ratio = jnp.min(determinant / safe_absolute)
        else:
            minimum_orientation_ratio = jnp.ones((), dtype=points.dtype)
        return IsogeometricGeometryEvidence(
            coordinate_scale,
            minimum_weight_ratio,
            minimum_rank_ratio,
            minimum_orientation_ratio,
            ambient_dimension=ambient,
            parametric_dimension=parametric,
            evidence_id=canonical_fingerprint(
                {
                    "kind": "isogeometric-geometry-evidence",
                    "actions": self.action_id,
                    "runtime_layout": runtime_.geometry_layout_id,
                    "numeric_version": runtime_.numeric_version,
                }
            ),
        )

    def realize(self, runtime: object, /) -> LocalMetricResult:
        metric, _ = self._metric(runtime)
        self.qualification_policy.check(self.evidence(runtime))
        return metric


__all__ = [
    "IsogeometricGeometryActions",
    "IsogeometricReferenceActions",
]
