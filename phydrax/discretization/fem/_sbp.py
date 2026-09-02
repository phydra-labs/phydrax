#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._interpolation import barycentric_differentiation_matrix
from ..._polynomial._orthogonal import legendre_rule_data
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


def _gll_rule(order: int, /) -> tuple[np.ndarray, np.ndarray]:
    if order < 1:
        raise ValueError("A GLL SBP rule requires polynomial order >= 1.")
    data = legendre_rule_data(order + 1, "lobatto")
    return (
        0.5 * (np.asarray(data.nodes) + 1.0),
        0.5 * np.asarray(data.weights),
    )


def _nodal_derivative(nodes: np.ndarray, /) -> np.ndarray:
    return np.asarray(
        barycentric_differentiation_matrix(jnp.asarray(nodes)),
        dtype=nodes.dtype,
    )


class ElementLocalSBPReport(StrictModule, NonTrainableState):
    """Numerical evidence for every defining identity of one nodal SBP rule."""

    positive_norm_margin: Array
    constant_derivative_defect: Array
    sbp_identity_defect: Array
    boundary_extraction_defect: Array
    tolerance: float = eqx.field(static=True)
    passed: bool = eqx.field(static=True)
    report_id: str = eqx.field(static=True)

    def __init__(
        self,
        positive_norm_margin: ArrayLike,
        constant_derivative_defect: ArrayLike,
        sbp_identity_defect: ArrayLike,
        boundary_extraction_defect: ArrayLike,
        /,
        *,
        tolerance: float,
        data_id: str,
    ):
        margin = jnp.asarray(positive_norm_margin)
        constant = jnp.asarray(constant_derivative_defect)
        identity = jnp.asarray(sbp_identity_defect)
        extraction = jnp.asarray(boundary_extraction_defect)
        tolerance_ = float(tolerance)
        if not np.isfinite(tolerance_) or tolerance_ <= 0.0:
            raise ValueError("SBP evidence tolerance must be positive and finite.")
        passed = bool(
            float(np.asarray(margin)) > 0.0
            and float(np.max(np.abs(np.asarray(constant)), initial=0.0)) <= tolerance_
            and float(np.max(np.abs(np.asarray(identity)), initial=0.0)) <= tolerance_
            and float(np.max(np.abs(np.asarray(extraction)), initial=0.0)) <= tolerance_
        )
        self.positive_norm_margin = margin
        self.constant_derivative_defect = constant
        self.sbp_identity_defect = identity
        self.boundary_extraction_defect = extraction
        self.tolerance = tolerance_
        self.passed = passed
        self.report_id = canonical_fingerprint(
            {
                "kind": "element-local-sbp-report",
                "data": str(data_id),
                "tolerance": tolerance_,
                "passed": passed,
            }
        )


class ElementLocalSBPData(StrictModule, NonTrainableState):
    """Immutable one-dimensional GLL diagonal-norm SBP data on ``[0, 1]``."""

    order: int = eqx.field(static=True)
    nodes: Array
    norm_weights: Array
    norm_matrix: Array
    derivative_matrix: Array
    restriction: Array
    boundary_weights: Array
    boundary_normals: Array
    report: ElementLocalSBPReport
    data_id: str = eqx.field(static=True)

    def __init__(self, order: int, /, *, tolerance: float | None = None):
        order_ = int(order)
        nodes, weights = _gll_rule(order_)
        runtime_dtype = np.asarray(jnp.asarray(nodes)).dtype
        nodes = nodes.astype(runtime_dtype)
        weights = weights.astype(runtime_dtype)
        derivative = _nodal_derivative(nodes)
        norm = np.diag(weights)
        restriction = np.zeros((2, order_ + 1), dtype=nodes.dtype)
        restriction[0, 0] = 1.0
        restriction[1, -1] = 1.0
        boundary_weights = np.eye(2, dtype=nodes.dtype)
        boundary_normals = np.diag(np.asarray((-1.0, 1.0), dtype=nodes.dtype))
        boundary_operator = (
            restriction.T @ boundary_weights @ boundary_normals @ restriction
        )
        constant_defect = derivative @ np.ones((order_ + 1,), dtype=nodes.dtype)
        identity_defect = norm @ derivative + derivative.T @ norm - boundary_operator
        expected_restriction = np.eye(order_ + 1, dtype=nodes.dtype)[[0, -1]]
        extraction_defect = restriction - expected_restriction
        tolerance_ = (
            128.0 * np.finfo(nodes.dtype).eps * max(1, order_) ** 2
            if tolerance is None
            else float(tolerance)
        )
        identifier = canonical_fingerprint(
            {
                "kind": "element-local-tensor-gll-sbp",
                "order": order_,
                "nodes": array_tree_fingerprint(nodes),
                "weights": array_tree_fingerprint(weights),
                "derivative": array_tree_fingerprint(derivative),
            }
        )
        report = ElementLocalSBPReport(
            np.min(weights),
            constant_defect,
            identity_defect,
            extraction_defect,
            tolerance=tolerance_,
            data_id=identifier,
        )
        if not report.passed:
            raise RuntimeError(
                "Constructed GLL rule failed its element-local SBP identities."
            )
        self.order = order_
        self.nodes = jnp.asarray(nodes)
        self.norm_weights = jnp.asarray(weights)
        self.norm_matrix = jnp.asarray(norm)
        self.derivative_matrix = jnp.asarray(derivative)
        self.restriction = jnp.asarray(restriction)
        self.boundary_weights = jnp.asarray(boundary_weights)
        self.boundary_normals = jnp.asarray(boundary_normals)
        self.report = report
        self.data_id = identifier

    @property
    def node_count(self) -> int:
        return self.order + 1


class TensorGLLSBPPlan(StrictModule, NonTrainableState):
    """Preparation plan for one reusable element-local GLL SBP rule."""

    order: int = eqx.field(static=True)
    tolerance: float | None = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, order: int, /, *, tolerance: float | None = None):
        order_ = int(order)
        if order_ < 1:
            raise ValueError("Tensor GLL polynomial order must be >= 1.")
        tolerance_ = None if tolerance is None else float(tolerance)
        if tolerance_ is not None and (not np.isfinite(tolerance_) or tolerance_ <= 0.0):
            raise ValueError("SBP tolerance must be positive and finite.")
        self.order = order_
        self.tolerance = tolerance_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "tensor-gll-sbp-plan",
                "order": order_,
                "tolerance": tolerance_,
            }
        )

    def prepare(self, /) -> ElementLocalSBPData:
        return ElementLocalSBPData(self.order, tolerance=self.tolerance)


class MetricFacePair(StrictModule, NonTrainableState):
    """Two local tensor faces expected to be watertight or periodically translated."""

    owner_cell: int = eqx.field(static=True)
    owner_axis: int = eqx.field(static=True)
    owner_side: int = eqx.field(static=True)
    neighbour_cell: int = eqx.field(static=True)
    neighbour_axis: int = eqx.field(static=True)
    neighbour_side: int = eqx.field(static=True)
    periodic_translation: bool = eqx.field(static=True)

    def __init__(
        self,
        owner_cell: int,
        owner_axis: int,
        owner_side: int,
        neighbour_cell: int,
        neighbour_axis: int,
        neighbour_side: int,
        /,
        *,
        periodic_translation: bool = False,
    ):
        values = tuple(
            int(value)
            for value in (
                owner_cell,
                owner_axis,
                owner_side,
                neighbour_cell,
                neighbour_axis,
                neighbour_side,
            )
        )
        if values[0] < 0 or values[3] < 0:
            raise ValueError("Metric face-pair cell indices must be non-negative.")
        if values[1] < 0 or values[4] < 0:
            raise ValueError("Metric face-pair axes must be non-negative.")
        if values[2] not in (0, 1) or values[5] not in (0, 1):
            raise ValueError("Metric face-pair sides must be zero or one.")
        self.owner_cell = values[0]
        self.owner_axis = values[1]
        self.owner_side = values[2]
        self.neighbour_cell = values[3]
        self.neighbour_axis = values[4]
        self.neighbour_side = values[5]
        self.periodic_translation = bool(periodic_translation)


class MappedTensorMetricReport(StrictModule, NonTrainableState):
    """Stationary mapped-metric, discrete GCL, and face-matching evidence."""

    determinant_margin: Array
    metric_identity_defect: Array
    free_stream_residual: Array
    watertight_face_position_defect: Array
    periodic_face_translation_defect: Array
    opposite_scaled_normal_defect: Array
    tolerance: float = eqx.field(static=True)
    passed: bool = eqx.field(static=True)
    report_id: str = eqx.field(static=True)

    def __init__(
        self,
        determinant_margin: ArrayLike,
        metric_identity_defect: ArrayLike,
        free_stream_residual: ArrayLike,
        watertight_face_position_defect: ArrayLike,
        periodic_face_translation_defect: ArrayLike,
        opposite_scaled_normal_defect: ArrayLike,
        /,
        *,
        tolerance: float,
        metrics_id: str,
    ):
        determinant = jnp.asarray(determinant_margin)
        identity = jnp.asarray(metric_identity_defect)
        free_stream = jnp.asarray(free_stream_residual)
        watertight = jnp.asarray(watertight_face_position_defect)
        periodic = jnp.asarray(periodic_face_translation_defect)
        opposite = jnp.asarray(opposite_scaled_normal_defect)
        tolerance_ = float(tolerance)
        if not np.isfinite(tolerance_) or tolerance_ <= 0.0:
            raise ValueError("Metric evidence tolerance must be positive and finite.")
        maxima = tuple(
            float(np.max(np.abs(np.asarray(value)), initial=0.0))
            for value in (identity, free_stream, watertight, periodic, opposite)
        )
        passed = bool(float(np.asarray(determinant)) > 0.0 and max(maxima) <= tolerance_)
        self.determinant_margin = determinant
        self.metric_identity_defect = identity
        self.free_stream_residual = free_stream
        self.watertight_face_position_defect = watertight
        self.periodic_face_translation_defect = periodic
        self.opposite_scaled_normal_defect = opposite
        self.tolerance = tolerance_
        self.passed = passed
        self.report_id = canonical_fingerprint(
            {
                "kind": "mapped-tensor-metric-report",
                "metrics": str(metrics_id),
                "tolerance": tolerance_,
                "passed": passed,
            }
        )


class MappedTensorMetrics(StrictModule, NonTrainableState):
    """Compatible stationary quad/hex determinant, cofactors, and face geometry."""

    dimension: int = eqx.field(static=True)
    coordinates: Array
    determinant: Array
    contravariant_cofactors: Array
    face_coordinates: tuple[Array, ...]
    face_scaled_normals: tuple[Array, ...]
    report: MappedTensorMetricReport
    metrics_id: str = eqx.field(static=True)

    @property
    def cell_count(self) -> int:
        return int(self.coordinates.shape[0])

    def face_pair_evidence(self, pair: MetricFacePair, /) -> tuple[Array, Array, Array]:
        """Return neighbour permutation, point defect, and opposite-normal defect."""

        if not isinstance(pair, MetricFacePair):
            raise TypeError("pair must be MetricFacePair.")
        if (
            pair.owner_axis >= self.dimension
            or pair.neighbour_axis >= self.dimension
            or pair.owner_cell >= self.cell_count
            or pair.neighbour_cell >= self.cell_count
        ):
            raise ValueError("Metric face-pair route is out of bounds.")
        owner_points = np.asarray(self.face_coordinates[pair.owner_axis])[
            pair.owner_cell, pair.owner_side
        ].reshape((-1, self.dimension))
        neighbour_points = np.asarray(self.face_coordinates[pair.neighbour_axis])[
            pair.neighbour_cell, pair.neighbour_side
        ].reshape((-1, self.dimension))
        face_node_count = owner_points.shape[0]
        node_count = (
            face_node_count
            if self.dimension == 2
            else int(round(np.sqrt(face_node_count)))
        )
        permutation, _translation, position_defect = _match_face_permutation(
            owner_points,
            neighbour_points,
            allow_translation=pair.periodic_translation,
            node_count=node_count,
            dimension=self.dimension,
        )
        owner_normals = np.asarray(self.face_scaled_normals[pair.owner_axis])[
            pair.owner_cell, pair.owner_side
        ].reshape((-1, self.dimension))
        neighbour_normals = np.asarray(self.face_scaled_normals[pair.neighbour_axis])[
            pair.neighbour_cell, pair.neighbour_side
        ].reshape((-1, self.dimension))[permutation]
        normal_defect = np.max(np.abs(owner_normals + neighbour_normals), initial=0.0)
        return (
            jnp.asarray(permutation, dtype=jnp.int32),
            jnp.asarray(position_defect),
            jnp.asarray(normal_defect),
        )


def _differentiate(value: Array, derivative: Array, axis: int, /) -> Array:
    moved = jnp.moveaxis(value, axis + 1, 1)
    differentiated = ein.contract("ij,cj...->ci...", derivative, moved, backend="jax")
    return jnp.moveaxis(differentiated, 1, axis + 1)


def _two_dimensional_cofactors(coordinates: Array, derivative: Array, /) -> Array:
    xi = _differentiate(coordinates, derivative, 0)
    eta = _differentiate(coordinates, derivative, 1)
    first = jnp.stack((eta[..., 1], -eta[..., 0]), axis=-1)
    second = jnp.stack((-xi[..., 1], xi[..., 0]), axis=-1)
    return jnp.stack((first, second), axis=-2)


def _three_dimensional_curl_cofactors(coordinates: Array, derivative: Array, /) -> Array:
    derivatives = tuple(
        _differentiate(coordinates, derivative, axis) for axis in range(3)
    )
    physical_components = []
    for first, second in ((1, 2), (2, 0), (0, 1)):
        product = tuple(
            coordinates[..., first] * derivatives[axis][..., second] for axis in range(3)
        )
        component = (
            _differentiate(product[2], derivative, 1)
            - _differentiate(product[1], derivative, 2),
            _differentiate(product[0], derivative, 2)
            - _differentiate(product[2], derivative, 0),
            _differentiate(product[1], derivative, 0)
            - _differentiate(product[0], derivative, 1),
        )
        physical_components.append(jnp.stack(component, axis=-1))
    return jnp.stack(physical_components, axis=-1)


def _face_symmetry_permutations(
    node_count: int, dimension: int, /
) -> tuple[np.ndarray, ...]:
    if dimension == 2:
        identity = np.arange(node_count, dtype=np.int32)
        return identity, identity[::-1]
    grid = np.arange(node_count**2, dtype=np.int32).reshape((node_count, node_count))
    candidates = []
    for rotation in range(4):
        rotated = np.rot90(grid, rotation)
        candidates.append(rotated.reshape((-1,)))
        candidates.append(np.flip(rotated, axis=0).reshape((-1,)))
    unique = []
    for candidate in candidates:
        if not any(np.array_equal(candidate, prior) for prior in unique):
            unique.append(candidate)
    return tuple(unique)


def _match_face_permutation(
    owner: np.ndarray,
    neighbour: np.ndarray,
    /,
    *,
    allow_translation: bool,
    node_count: int,
    dimension: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    permutations = _face_symmetry_permutations(node_count, dimension)
    best = None
    for permutation in permutations:
        candidate = neighbour[permutation]
        translation = (
            np.mean(candidate - owner, axis=0)
            if allow_translation
            else np.zeros((owner.shape[-1],), dtype=owner.dtype)
        )
        defect = float(np.max(np.abs(candidate - translation - owner), initial=0.0))
        if best is None or defect < best[2]:
            best = permutation, translation, defect
    if best is None:
        raise RuntimeError("Tensor face matching produced no admissible symmetry.")
    return best


class MappedTensorMetricPlan(StrictModule, NonTrainableState):
    """Compatible stationary mapped quad/hex metric and discrete-GCL plan."""

    sbp: ElementLocalSBPData
    dimension: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        sbp: ElementLocalSBPData,
        dimension: int,
        /,
        *,
        tolerance: float | None = None,
    ):
        if not isinstance(sbp, ElementLocalSBPData):
            raise TypeError("sbp must be ElementLocalSBPData.")
        dimension_ = int(dimension)
        if dimension_ not in (2, 3):
            raise ValueError("Mapped tensor metrics support dimensions two and three.")
        dtype = np.asarray(sbp.nodes).dtype
        tolerance_ = float(
            512.0 * np.finfo(dtype).eps * max(1, sbp.order) ** 3
            if tolerance is None
            else tolerance
        )
        if not np.isfinite(tolerance_) or tolerance_ <= 0.0:
            raise ValueError("Metric evidence tolerance must be positive and finite.")
        self.sbp = sbp
        self.dimension = dimension_
        self.tolerance = tolerance_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mapped-tensor-metric-plan",
                "sbp": sbp.data_id,
                "dimension": dimension_,
                "tolerance": tolerance_,
                "motion": "stationary",
            }
        )

    def prepare(
        self,
        coordinate_values: ArrayLike,
        /,
        *,
        face_pairs: Sequence[MetricFacePair] = (),
    ) -> MappedTensorMetrics:
        coordinates = jnp.asarray(coordinate_values)
        node_count = self.sbp.node_count
        tensor_shape = (node_count,) * self.dimension
        if coordinates.ndim == 3 and coordinates.shape[1:] == (
            node_count**self.dimension,
            self.dimension,
        ):
            coordinates = coordinates.reshape(
                (coordinates.shape[0],) + tensor_shape + (self.dimension,)
            )
        expected_tail = tensor_shape + (self.dimension,)
        if (
            coordinates.ndim != self.dimension + 2
            or coordinates.shape[1:] != expected_tail
        ):
            raise ValueError(
                "Mapped tensor coordinates must have shape "
                f"(cell, {', '.join(map(str, tensor_shape))}, {self.dimension}) "
                "or the equivalent flattened nodal shape."
            )
        derivative = self.sbp.derivative_matrix
        if self.dimension == 2:
            cofactors = _two_dimensional_cofactors(coordinates, derivative)
        else:
            cofactors = _three_dimensional_curl_cofactors(coordinates, derivative)
        coordinate_derivatives = tuple(
            _differentiate(coordinates, derivative, axis)
            for axis in range(self.dimension)
        )
        determinant = sum(
            ein.contract(
                "c...d,c...d->c...",
                coordinate_derivatives[axis],
                cofactors[..., axis, :],
                backend="jax",
            )
            for axis in range(self.dimension)
        ) / float(self.dimension)
        identity = sum(
            _differentiate(cofactors[..., axis, :], derivative, axis)
            for axis in range(self.dimension)
        )
        face_coordinates = []
        face_normals = []
        for axis in range(self.dimension):
            axis_coordinates = []
            axis_normals = []
            for side, index in enumerate((0, -1)):
                axis_coordinates.append(jnp.take(coordinates, index, axis=axis + 1))
                cofactor = jnp.take(cofactors[..., axis, :], index, axis=axis + 1)
                axis_normals.append((-1.0 if side == 0 else 1.0) * cofactor)
            face_coordinates.append(jnp.stack(tuple(axis_coordinates), axis=1))
            face_normals.append(jnp.stack(tuple(axis_normals), axis=1))

        watertight_position_defects = []
        periodic_translation_defects = []
        normal_defects = []
        coordinate_arrays = tuple(np.asarray(value) for value in face_coordinates)
        normal_arrays = tuple(np.asarray(value) for value in face_normals)
        for pair in tuple(face_pairs):
            if not isinstance(pair, MetricFacePair):
                raise TypeError("face_pairs must contain MetricFacePair values.")
            if (
                pair.owner_axis >= self.dimension
                or pair.neighbour_axis >= self.dimension
                or pair.owner_cell >= coordinates.shape[0]
                or pair.neighbour_cell >= coordinates.shape[0]
            ):
                raise ValueError("Metric face-pair route is out of bounds.")
            owner_points = coordinate_arrays[pair.owner_axis][
                pair.owner_cell, pair.owner_side
            ].reshape((-1, self.dimension))
            neighbour_points = coordinate_arrays[pair.neighbour_axis][
                pair.neighbour_cell, pair.neighbour_side
            ].reshape((-1, self.dimension))
            permutation, _translation, defect = _match_face_permutation(
                owner_points,
                neighbour_points,
                allow_translation=pair.periodic_translation,
                node_count=node_count,
                dimension=self.dimension,
            )
            owner_normals = normal_arrays[pair.owner_axis][
                pair.owner_cell, pair.owner_side
            ].reshape((-1, self.dimension))
            neighbour_normals = normal_arrays[pair.neighbour_axis][
                pair.neighbour_cell, pair.neighbour_side
            ].reshape((-1, self.dimension))[permutation]
            if pair.periodic_translation:
                periodic_translation_defects.append(defect)
            else:
                watertight_position_defects.append(defect)
            normal_defects.append(
                np.max(np.abs(owner_normals + neighbour_normals), initial=0.0)
            )
        watertight_evidence = np.asarray(
            watertight_position_defects, dtype=np.asarray(coordinates).dtype
        )
        periodic_evidence = np.asarray(
            periodic_translation_defects, dtype=np.asarray(coordinates).dtype
        )
        normal_evidence = np.asarray(normal_defects, dtype=np.asarray(coordinates).dtype)
        identifier = canonical_fingerprint(
            {
                "kind": "mapped-tensor-metrics",
                "plan": self.plan_id,
                "coordinate_shape": list(coordinates.shape),
                "face_pairs": [
                    [
                        pair.owner_cell,
                        pair.owner_axis,
                        pair.owner_side,
                        pair.neighbour_cell,
                        pair.neighbour_axis,
                        pair.neighbour_side,
                        pair.periodic_translation,
                    ]
                    for pair in face_pairs
                ],
            }
        )
        report = MappedTensorMetricReport(
            jnp.min(determinant),
            identity,
            identity,
            watertight_evidence,
            periodic_evidence,
            normal_evidence,
            tolerance=self.tolerance,
            metrics_id=identifier,
        )
        if not report.passed:
            raise ValueError(
                "Mapped tensor metric/GCL evidence failed: nonpositive determinant, "
                "metric identity defect, or incompatible paired faces."
            )
        return MappedTensorMetrics(
            dimension=self.dimension,
            coordinates=coordinates,
            determinant=determinant,
            contravariant_cofactors=cofactors,
            face_coordinates=tuple(face_coordinates),
            face_scaled_normals=tuple(face_normals),
            report=report,
            metrics_id=identifier,
        )


__all__ = [
    "ElementLocalSBPData",
    "ElementLocalSBPReport",
    "MappedTensorMetricPlan",
    "MappedTensorMetricReport",
    "MappedTensorMetrics",
    "MetricFacePair",
    "TensorGLLSBPPlan",
]
