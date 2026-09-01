#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import Enum

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from .._atlas import AbstractBoundaryMap, BoundaryAtlas
from ..brep._patches import BSplineSurfacePatch
from ._model import SurfaceModel


class HighOrderSurfaceSource(str, Enum):
    """Concrete high-order coordinate providers."""

    BSPLINE_PATCH = "bspline_patch"
    BOUNDARY_ATLAS = "boundary_atlas"
    ISOPARAMETRIC_TRIANGLE = "isoparametric_triangle"


class HighOrderDifferentiationError(ValueError):
    """Raised when a caller requests an unimplemented derivative order."""


class HighOrderResourceLimitError(ValueError):
    """Raised before realization/evaluation exceeds a fixed host capacity."""


class HighOrderGeometryMismatchError(ValueError):
    """Raised when high-order corner geometry disagrees with the CellMesh."""


class HighOrderSurfacePolicy(StrictModule, NonTrainableState):
    """Fixed capacities and geometry tolerances for high-order realization."""

    maximum_order: int = eqx.field(static=True)
    maximum_cells: int = eqx.field(static=True)
    maximum_nodes_per_cell: int = eqx.field(static=True)
    maximum_evaluation_points: int = eqx.field(static=True)
    corner_tolerance: float = eqx.field(static=True)
    minimum_jacobian: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_order: int = 8,
        maximum_cells: int = 1_000_000,
        maximum_nodes_per_cell: int = 45,
        maximum_evaluation_points: int = 1_000_000,
        corner_tolerance: float = 1.0e-10,
        minimum_jacobian: float = 0.0,
    ):
        capacities = (
            int(maximum_order),
            int(maximum_cells),
            int(maximum_nodes_per_cell),
            int(maximum_evaluation_points),
        )
        if any(value <= 0 for value in capacities):
            raise ValueError("High-order capacities and maximum_order must be positive.")
        tolerance = float(corner_tolerance)
        minimum = float(minimum_jacobian)
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("corner_tolerance must be finite and non-negative.")
        if not np.isfinite(minimum) or minimum < 0.0:
            raise ValueError("minimum_jacobian must be finite and non-negative.")
        (
            self.maximum_order,
            self.maximum_cells,
            self.maximum_nodes_per_cell,
            self.maximum_evaluation_points,
        ) = capacities
        self.corner_tolerance = tolerance
        self.minimum_jacobian = minimum
        self.policy_id = canonical_fingerprint(
            {
                "kind": "bounded-high-order-surface-policy",
                "capacities": capacities,
                "corner_tolerance": tolerance,
                "minimum_jacobian": minimum,
            }
        )


class HighOrderSurfaceReport(StrictModule, NonTrainableState):
    """Immutable binding, units, capacity, and differentiation evidence."""

    source: HighOrderSurfaceSource = eqx.field(static=True)
    order: int = eqx.field(static=True)
    cell_count: int = eqx.field(static=True)
    nodes_per_cell: int = eqx.field(static=True)
    maximum_parametric_derivative_order: int = eqx.field(static=True)
    length_unit: str = eqx.field(static=True)
    metric_unit: str = eqx.field(static=True)
    jacobian_unit: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    model_id: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)
    corner_maximum_error: float = eqx.field(static=True)
    report_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        source: HighOrderSurfaceSource,
        order: int,
        cell_count: int,
        nodes_per_cell: int,
        length_unit: str,
        topology_id: str,
        model_id: str,
        policy_id: str,
        corner_maximum_error: float,
    ):
        if not isinstance(source, HighOrderSurfaceSource):
            raise TypeError("source must be HighOrderSurfaceSource.")
        unit = str(length_unit)
        if not unit:
            raise ValueError("High-order realization requires a length unit.")
        self.source = source
        self.order = int(order)
        self.cell_count = int(cell_count)
        self.nodes_per_cell = int(nodes_per_cell)
        self.maximum_parametric_derivative_order = 1
        self.length_unit = unit
        self.metric_unit = f"{unit}^2"
        self.jacobian_unit = f"{unit}^2"
        self.topology_id = str(topology_id)
        self.model_id = str(model_id)
        self.policy_id = str(policy_id)
        self.corner_maximum_error = float(corner_maximum_error)
        self.report_id = canonical_fingerprint(
            {
                "kind": "high-order-surface-report",
                "source": source.value,
                "order": int(order),
                "cell_count": int(cell_count),
                "nodes_per_cell": int(nodes_per_cell),
                "maximum_parametric_derivative_order": 1,
                "length_unit": unit,
                "topology_id": topology_id,
                "model_id": model_id,
                "policy_id": policy_id,
                "corner_maximum_error": float(corner_maximum_error),
            }
        )


class HighOrderSurfaceFrameEvidence(StrictModule, NonTrainableState):
    """Pointwise metric, oriented normal, and surface-Jacobian evidence."""

    chart_indices: Array
    cell_global_ids: Array
    reference_coordinates: Array
    physical_coordinates: Array
    differential: Array
    metric: Array
    normal: Array
    jacobian: Array
    finite: Array
    nondegenerate: Array
    valid: Array
    report_id: str = eqx.field(static=True)
    maximum_parametric_derivative_order: int = eqx.field(static=True)
    metric_unit: str = eqx.field(static=True)
    jacobian_unit: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        chart_indices: Array,
        cell_global_ids: Array,
        reference_coordinates: Array,
        physical_coordinates: Array,
        differential: Array,
        metric: Array,
        normal: Array,
        jacobian: Array,
        finite: Array,
        nondegenerate: Array,
        report: HighOrderSurfaceReport,
    ):
        finite_ = jnp.asarray(finite, dtype=bool)
        nondegenerate_ = jnp.asarray(nondegenerate, dtype=bool)
        self.chart_indices = jnp.asarray(chart_indices, dtype=jnp.int32)
        self.cell_global_ids = jnp.asarray(cell_global_ids, dtype=jnp.int64)
        self.reference_coordinates = jnp.asarray(reference_coordinates)
        self.physical_coordinates = jnp.asarray(physical_coordinates)
        self.differential = jnp.asarray(differential)
        self.metric = jnp.asarray(metric)
        self.normal = jnp.asarray(normal)
        self.jacobian = jnp.asarray(jacobian)
        self.finite = finite_
        self.nondegenerate = nondegenerate_
        self.valid = finite_ & nondegenerate_
        self.report_id = report.report_id
        self.maximum_parametric_derivative_order = 1
        self.metric_unit = report.metric_unit
        self.jacobian_unit = report.jacobian_unit


def _duffy_triangle_coordinates(vertices: Array, reference: Array, /) -> Array:
    first = reference[..., :1]
    second = reference[..., 1:2]
    return (
        vertices[..., 0, :]
        + first * (vertices[..., 1, :] - vertices[..., 0, :])
        + (1.0 - first) * second * (vertices[..., 2, :] - vertices[..., 0, :])
    )


def _batched_differential(
    mapping: AbstractBoundaryMap, indices: Array, reference: Array, /
) -> Array:
    leading = indices.shape
    values = jax.vmap(
        lambda index, coordinate: jax.jacfwd(lambda value: mapping.map(index, value))(
            coordinate
        )
    )(indices.reshape((-1,)), reference.reshape((-1, 2)))
    return values.reshape((*leading, 3, 2))


class _PatchTriangleMap(AbstractBoundaryMap):
    patch: BSplineSurfacePatch
    cell_parameters: Array

    def __init__(self, patch: BSplineSurfacePatch, cell_parameters: ArrayLike, /):
        self.patch = patch
        self.cell_parameters = jnp.asarray(cell_parameters, dtype=float)

    @property
    def num_charts(self) -> int:
        return int(self.cell_parameters.shape[0])

    @property
    def reference_dimension(self) -> int:
        return 2

    @property
    def ambient_dimension(self) -> int:
        return 3

    def map(self, chart_indices: Array, reference: Array, /) -> Array:
        indices = jnp.asarray(chart_indices, dtype=jnp.int32)
        coordinates = jnp.asarray(reference, dtype=self.cell_parameters.dtype)
        parameters = _duffy_triangle_coordinates(
            self.cell_parameters[indices], coordinates
        )
        return self.patch.evaluate(parameters)

    def jacobian(self, chart_indices: Array, reference: Array, /) -> Array:
        indices = jnp.asarray(chart_indices, dtype=jnp.int32)
        coordinates = jnp.asarray(reference, dtype=self.cell_parameters.dtype)
        differential = _batched_differential(self, indices, coordinates)
        return jnp.linalg.norm(
            jnp.cross(differential[..., :, 0], differential[..., :, 1]), axis=-1
        )


class _AtlasTriangleMap(AbstractBoundaryMap):
    atlas: BoundaryAtlas
    source_chart_indices: Array
    cell_parameters: Array

    def __init__(
        self,
        atlas: BoundaryAtlas,
        source_chart_indices: ArrayLike,
        cell_parameters: ArrayLike,
        /,
    ):
        self.atlas = atlas
        self.source_chart_indices = jnp.asarray(source_chart_indices, dtype=jnp.int32)
        self.cell_parameters = jnp.asarray(cell_parameters, dtype=float)

    @property
    def num_charts(self) -> int:
        return int(self.cell_parameters.shape[0])

    @property
    def reference_dimension(self) -> int:
        return 2

    @property
    def ambient_dimension(self) -> int:
        return 3

    def map(self, chart_indices: Array, reference: Array, /) -> Array:
        indices = jnp.asarray(chart_indices, dtype=jnp.int32)
        coordinates = jnp.asarray(reference, dtype=self.cell_parameters.dtype)
        parameters = _duffy_triangle_coordinates(
            self.cell_parameters[indices], coordinates
        )
        return self.atlas.mapping.map(self.source_chart_indices[indices], parameters)

    def jacobian(self, chart_indices: Array, reference: Array, /) -> Array:
        indices = jnp.asarray(chart_indices, dtype=jnp.int32)
        coordinates = jnp.asarray(reference, dtype=self.cell_parameters.dtype)
        differential = _batched_differential(self, indices, coordinates)
        return jnp.linalg.norm(
            jnp.cross(differential[..., :, 0], differential[..., :, 1]), axis=-1
        )


class _IsoparametricTriangleMap(AbstractBoundaryMap):
    coordinate_nodes: Array
    coefficients: Array
    exponents: Array
    order: int = eqx.field(static=True)

    def __init__(self, coordinate_nodes, coefficients, exponents, order: int, /):
        self.coordinate_nodes = jnp.asarray(coordinate_nodes, dtype=float)
        self.coefficients = jnp.asarray(coefficients, dtype=float)
        self.exponents = jnp.asarray(exponents, dtype=jnp.int32)
        self.order = int(order)

    @property
    def num_charts(self) -> int:
        return int(self.coordinate_nodes.shape[0])

    @property
    def reference_dimension(self) -> int:
        return 2

    @property
    def ambient_dimension(self) -> int:
        return 3

    def _map_one(self, chart_index: Array, reference: Array, /) -> Array:
        simplex = reference
        monomials = (
            simplex[0] ** self.exponents[:, 0] * simplex[1] ** self.exponents[:, 1]
        )
        return monomials @ self.coefficients[chart_index]

    def map(self, chart_indices: Array, reference: Array, /) -> Array:
        indices = jnp.asarray(chart_indices, dtype=jnp.int32)
        coordinates = jnp.asarray(reference, dtype=self.coordinate_nodes.dtype)
        leading = indices.shape
        values = jax.vmap(self._map_one)(
            indices.reshape((-1,)), coordinates.reshape((-1, 2))
        )
        return values.reshape((*leading, 3))

    def jacobian(self, chart_indices: Array, reference: Array, /) -> Array:
        indices = jnp.asarray(chart_indices, dtype=jnp.int32)
        coordinates = jnp.asarray(reference, dtype=self.coordinate_nodes.dtype)
        differential = _batched_differential(self, indices, coordinates)
        return jnp.linalg.norm(
            jnp.cross(differential[..., :, 0], differential[..., :, 1]), axis=-1
        )


HighOrderMapping = _PatchTriangleMap | _AtlasTriangleMap | _IsoparametricTriangleMap


class HighOrderSurfaceRealization(StrictModule, NonTrainableState):
    """Bounded high-order coordinates bound to an authoritative SurfaceModel."""

    model: SurfaceModel
    mapping: HighOrderMapping
    orientation: Array
    cell_global_ids: Array
    policy: HighOrderSurfacePolicy
    report: HighOrderSurfaceReport
    realization_id: str = eqx.field(static=True)

    def __init__(
        self,
        model: SurfaceModel,
        mapping: HighOrderMapping,
        orientation: ArrayLike,
        policy: HighOrderSurfacePolicy,
        report: HighOrderSurfaceReport,
        /,
    ):
        if mapping.num_charts != int(model.mesh.connectivity.cell_count):
            raise ValueError(
                "High-order chart count must equal authoritative cell count."
            )
        signs = np.asarray(orientation, dtype=float)
        if signs.shape != (mapping.num_charts,) or np.any(
            (signs != 1.0) & (signs != -1.0)
        ):
            raise ValueError("High-order orientation requires one +1/-1 sign per cell.")
        if (
            report.model_id != model.model_id
            or report.topology_id != model.mesh.topology_id
        ):
            raise ValueError("High-order report is not bound to its authoritative model.")
        cell_ids = np.asarray(model.mesh.entity_set(2).entity_ids, dtype=np.int64)
        self.model = model
        self.mapping = mapping
        self.orientation = jnp.asarray(signs)
        self.cell_global_ids = jnp.asarray(cell_ids, dtype=jnp.int64)
        self.policy = policy
        self.report = report
        self.realization_id = canonical_fingerprint(
            {
                "kind": "bounded-high-order-surface-realization",
                "model_id": model.model_id,
                "report_id": report.report_id,
                "cell_global_ids": array_tree_fingerprint(cell_ids),
            }
        )

    def _inputs(self, chart_indices: ArrayLike, reference: ArrayLike, /):
        indices = jnp.asarray(chart_indices, dtype=jnp.int32)
        coordinates = jnp.asarray(reference, dtype=float)
        if coordinates.ndim < 1 or coordinates.shape[-1] != 2:
            raise ValueError("reference coordinates require trailing dimension 2.")
        if indices.shape != coordinates.shape[:-1]:
            raise ValueError("chart_indices must match reference leading dimensions.")
        if indices.size > self.policy.maximum_evaluation_points:
            raise HighOrderResourceLimitError(
                "Evaluation exceeds maximum_evaluation_points."
            )
        return indices, coordinates

    def evaluate(
        self,
        chart_indices: ArrayLike,
        reference: ArrayLike,
        /,
        *,
        derivative_order: int = 0,
    ) -> Array | HighOrderSurfaceFrameEvidence:
        order = int(derivative_order)
        if order == 1:
            return self.frame(chart_indices, reference)
        if order != 0:
            raise HighOrderDifferentiationError(
                "High-order realization supports parameter derivatives only through order 1."
            )
        indices, coordinates = self._inputs(chart_indices, reference)
        return self.mapping.map(indices, coordinates)

    def frame(self, chart_indices: ArrayLike, reference: ArrayLike, /):
        indices, coordinates = self._inputs(chart_indices, reference)
        points = self.mapping.map(indices, coordinates)
        differential = _batched_differential(self.mapping, indices, coordinates)
        metric = jnp.swapaxes(differential, -1, -2) @ differential
        raw_normal = jnp.cross(differential[..., :, 0], differential[..., :, 1])
        jacobian = jnp.linalg.norm(raw_normal, axis=-1)
        safe = jnp.where(jacobian > 0.0, jacobian, 1.0)
        normal = raw_normal / safe[..., None] * self.orientation[indices][..., None]
        finite = (
            jnp.all(jnp.isfinite(points), axis=-1)
            & jnp.all(jnp.isfinite(differential), axis=(-2, -1))
            & jnp.all(jnp.isfinite(metric), axis=(-2, -1))
            & jnp.all(jnp.isfinite(normal), axis=-1)
            & jnp.isfinite(jacobian)
        )
        return HighOrderSurfaceFrameEvidence(
            chart_indices=indices,
            cell_global_ids=self.cell_global_ids[indices],
            reference_coordinates=coordinates,
            physical_coordinates=points,
            differential=differential,
            metric=metric,
            normal=normal,
            jacobian=jacobian,
            finite=finite,
            nondegenerate=jacobian > self.policy.minimum_jacobian,
            report=self.report,
        )


def _policy(value: HighOrderSurfacePolicy | None, /) -> HighOrderSurfacePolicy:
    policy = HighOrderSurfacePolicy() if value is None else value
    if not isinstance(policy, HighOrderSurfacePolicy):
        raise TypeError("policy must be HighOrderSurfacePolicy or None.")
    return policy


def _authoritative_corners(model: SurfaceModel, /) -> np.ndarray:
    faces = np.asarray(model.mesh.connectivity.cell_vertices, dtype=np.int32)[:, :3]
    return np.asarray(model.mesh.coordinates, dtype=float)[faces]


def _validate_parameter_triangles(value: ArrayLike, cell_count: int, /) -> np.ndarray:
    parameters = np.asarray(value, dtype=float)
    if parameters.shape != (cell_count, 3, 2) or not np.all(np.isfinite(parameters)):
        raise ValueError(
            "cell_parameters must have shape (cell_count, 3, 2) and be finite."
        )
    first = parameters[:, 1] - parameters[:, 0]
    second = parameters[:, 2] - parameters[:, 0]
    signed_area = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
    if np.any(signed_area == 0.0):
        raise ValueError("High-order parameter triangles must be nondegenerate.")
    return parameters


def _corner_error(expected, actual, tolerance: float, /) -> float:
    if actual.shape != expected.shape or not np.all(np.isfinite(actual)):
        raise HighOrderGeometryMismatchError("High-order corner evaluation is malformed.")
    maximum = float(np.max(np.linalg.norm(actual - expected, axis=-1)))
    if maximum > tolerance:
        raise HighOrderGeometryMismatchError(
            f"High-order corners differ from CellMesh by {maximum}; limit is {tolerance}."
        )
    return maximum


def _preflight(model, policy, nodes_per_cell: int, /) -> int:
    count = int(model.mesh.connectivity.cell_count)
    if count > policy.maximum_cells or nodes_per_cell > policy.maximum_nodes_per_cell:
        raise HighOrderResourceLimitError(
            "High-order realization exceeds fixed capacities."
        )
    return count


def _report(model, policy, source, order, nodes, error, /):
    return HighOrderSurfaceReport(
        source=source,
        order=order,
        cell_count=int(model.mesh.connectivity.cell_count),
        nodes_per_cell=nodes,
        length_unit=model.metadata.length_unit,
        topology_id=model.mesh.topology_id,
        model_id=model.model_id,
        policy_id=policy.policy_id,
        corner_maximum_error=error,
    )


def realize_bspline_surface(
    model: SurfaceModel,
    patch: BSplineSurfacePatch,
    cell_parameters: ArrayLike,
    /,
    *,
    policy: HighOrderSurfacePolicy | None = None,
) -> HighOrderSurfaceRealization:
    if not isinstance(model, SurfaceModel) or not isinstance(patch, BSplineSurfacePatch):
        raise TypeError("Requires SurfaceModel and BSplineSurfacePatch.")
    policy_ = _policy(policy)
    order = max(patch.u_degree, patch.v_degree)
    if order > policy_.maximum_order:
        raise HighOrderResourceLimitError("B-spline degree exceeds maximum_order.")
    count = _preflight(model, policy_, 3)
    parameters = _validate_parameter_triangles(cell_parameters, count)
    mapping = _PatchTriangleMap(patch, parameters)
    actual = np.asarray(patch.evaluate(jnp.asarray(parameters.reshape((-1, 2))))).reshape(
        (count, 3, 3)
    )
    error = _corner_error(_authoritative_corners(model), actual, policy_.corner_tolerance)
    report = _report(
        model, policy_, HighOrderSurfaceSource.BSPLINE_PATCH, order, 3, error
    )
    return HighOrderSurfaceRealization(model, mapping, np.ones(count), policy_, report)


def realize_boundary_atlas(
    model: SurfaceModel,
    atlas: BoundaryAtlas,
    source_chart_indices: ArrayLike,
    cell_parameters: ArrayLike,
    /,
    *,
    policy: HighOrderSurfacePolicy | None = None,
) -> HighOrderSurfaceRealization:
    if not isinstance(model, SurfaceModel) or not isinstance(atlas, BoundaryAtlas):
        raise TypeError("Requires SurfaceModel and BoundaryAtlas.")
    policy_ = _policy(policy)
    count = _preflight(model, policy_, 3)
    charts = np.asarray(source_chart_indices, dtype=np.int32)
    if (
        charts.shape != (count,)
        or np.any(charts < 0)
        or np.any(charts >= atlas.num_charts)
    ):
        raise ValueError("Select one existing atlas chart per cell.")
    if atlas.reference_dimension != 2 or atlas.ambient_dimension != 3:
        raise ValueError("Surface atlas must map 2D charts into 3D.")
    parameters = _validate_parameter_triangles(cell_parameters, count)
    actual = np.asarray(
        atlas.mapping.map(
            jnp.asarray(np.repeat(charts, 3)), jnp.asarray(parameters.reshape((-1, 2)))
        )
    ).reshape((count, 3, 3))
    error = _corner_error(_authoritative_corners(model), actual, policy_.corner_tolerance)
    mapping = _AtlasTriangleMap(atlas, charts, parameters)
    report = _report(model, policy_, HighOrderSurfaceSource.BOUNDARY_ATLAS, 1, 3, error)
    return HighOrderSurfaceRealization(
        model, mapping, np.asarray(atlas.orientation)[charts], policy_, report
    )


def isoparametric_triangle_reference_nodes(order: int, /) -> np.ndarray:
    degree = int(order)
    if degree < 1:
        raise ValueError("Isoparametric triangle order must be positive.")
    return np.asarray(
        [
            (first / degree, second / degree)
            for first in range(degree + 1)
            for second in range(degree + 1 - first)
        ],
        dtype=float,
    )


def realize_isoparametric_triangles(
    model: SurfaceModel,
    coordinate_nodes: ArrayLike,
    order: int,
    /,
    *,
    policy: HighOrderSurfacePolicy | None = None,
) -> HighOrderSurfaceRealization:
    if not isinstance(model, SurfaceModel):
        raise TypeError("model must be SurfaceModel.")
    policy_ = _policy(policy)
    degree = int(order)
    if degree < 1 or degree > policy_.maximum_order:
        raise HighOrderResourceLimitError("Isoparametric order exceeds policy.")
    reference = isoparametric_triangle_reference_nodes(degree)
    nodes_per_cell = int(reference.shape[0])
    count = _preflight(model, policy_, nodes_per_cell)
    nodes = np.asarray(coordinate_nodes, dtype=float)
    if nodes.shape != (count, nodes_per_cell, 3) or not np.all(np.isfinite(nodes)):
        raise ValueError("coordinate_nodes have incompatible shape or non-finite data.")
    exponents = np.asarray(
        [(a, b) for a in range(degree + 1) for b in range(degree + 1 - a)],
        dtype=np.int32,
    )
    vandermonde = np.column_stack(
        tuple(reference[:, 0] ** int(a) * reference[:, 1] ** int(b) for a, b in exponents)
    )
    coefficients = np.matmul(np.linalg.inv(vandermonde)[None, :, :], nodes)
    corners = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
    corner_indices = tuple(
        int(np.flatnonzero(np.all(reference == location, axis=1))[0])
        for location in corners
    )
    error = _corner_error(
        _authoritative_corners(model),
        nodes[:, corner_indices, :],
        policy_.corner_tolerance,
    )
    mapping = _IsoparametricTriangleMap(nodes, coefficients, exponents, degree)
    report = _report(
        model,
        policy_,
        HighOrderSurfaceSource.ISOPARAMETRIC_TRIANGLE,
        degree,
        nodes_per_cell,
        error,
    )
    return HighOrderSurfaceRealization(model, mapping, np.ones(count), policy_, report)


__all__ = [
    "HighOrderDifferentiationError",
    "HighOrderGeometryMismatchError",
    "HighOrderResourceLimitError",
    "HighOrderSurfaceFrameEvidence",
    "HighOrderSurfacePolicy",
    "HighOrderSurfaceRealization",
    "HighOrderSurfaceReport",
    "HighOrderSurfaceSource",
    "isoparametric_triangle_reference_nodes",
    "realize_boundary_atlas",
    "realize_bspline_surface",
    "realize_isoparametric_triangles",
]
