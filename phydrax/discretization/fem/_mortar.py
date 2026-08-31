#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from itertools import product

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import (
    ArraySpace,
    DenseLinearOperator,
    FactorizationPolicy,
    factorize,
    OperatorProperties,
)


def _point_array(
    value: ArrayLike,
    name: str,
    /,
    *,
    allowed_dimensions: tuple[int, ...] = (1, 2),
) -> np.ndarray:
    points = np.asarray(value)
    if points.ndim == 1:
        points = points[:, None]
    expected = ", ".join(str(value) for value in allowed_dimensions)
    if (
        points.ndim != 2
        or points.shape[0] == 0
        or points.shape[1] not in allowed_dimensions
    ):
        raise ValueError(f"{name} must have one of dimensions ({expected}).")
    if not np.issubdtype(points.dtype, np.inexact) or np.any(~np.isfinite(points)):
        raise ValueError(f"{name} must contain finite inexact values.")
    return points


def _permutation(value: ArrayLike | None, size: int, name: str, /) -> np.ndarray:
    permutation = (
        np.arange(size, dtype=np.int32)
        if value is None
        else np.asarray(value, dtype=np.int32)
    )
    if permutation.shape != (size,) or not np.array_equal(
        np.sort(permutation), np.arange(size, dtype=np.int32)
    ):
        raise ValueError(f"{name} must be one permutation of the trace nodes.")
    return permutation


def _tensor_tabulation(nodes: np.ndarray, points: np.ndarray, /) -> np.ndarray:
    if nodes.shape[1] != points.shape[1]:
        raise ValueError("Tensor nodes and evaluation points have different dimensions.")
    axes = tuple(np.unique(nodes[:, axis]) for axis in range(nodes.shape[1]))
    if np.prod([axis.size for axis in axes], dtype=np.int64) != nodes.shape[0]:
        raise ValueError("Trace and mortar nodes must form complete tensor grids.")
    tensor_indices = np.stack(
        tuple(
            np.searchsorted(axis_values, nodes[:, axis]).astype(np.int32)
            for axis, axis_values in enumerate(axes)
        ),
        axis=-1,
    )
    if np.unique(tensor_indices, axis=0).shape[0] != nodes.shape[0]:
        raise ValueError("Tensor nodes must not contain duplicates.")

    axis_values = []
    for axis_nodes, evaluation in zip(axes, points.T, strict=True):
        values = np.ones((points.shape[0], axis_nodes.size), dtype=nodes.dtype)
        for column, node in enumerate(axis_nodes):
            for other_column, other in enumerate(axis_nodes):
                if column != other_column:
                    values[:, column] *= (evaluation - other) / (node - other)
        axis_values.append(values)
    result = np.ones((points.shape[0], nodes.shape[0]), dtype=nodes.dtype)
    for axis, values in enumerate(axis_values):
        result *= values[:, tensor_indices[:, axis]]
    return result


def _mass_solve(matrix: Array, right_hand_side: Array, name: str, /) -> Array:
    space = ArraySpace((matrix.shape[0],), dtype=matrix.dtype)
    operator = DenseLinearOperator(
        matrix,
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
    )
    result = factorize(operator, FactorizationPolicy("cholesky")).solve(right_hand_side)
    return eqx.error_if(
        result.value,
        ~jnp.all(result.successful),
        f"{name} factorization failed.",
    )


class FiniteElementMortarMetricData(StrictModule):
    """Physical coordinates, weights, and opposite scaled normals for one mortar."""

    physical_coordinates: Array
    physical_weights: Array
    owner_scaled_normals: Array
    neighbour_scaled_normals: Array
    metric_id: str = eqx.field(static=True)

    def __init__(
        self,
        physical_coordinates: ArrayLike,
        physical_weights: ArrayLike,
        owner_scaled_normals: ArrayLike,
        neighbour_scaled_normals: ArrayLike,
        /,
        *,
        metric_id: str | None = None,
    ):
        coordinates = jnp.asarray(physical_coordinates)
        weights = jnp.asarray(physical_weights)
        owner = jnp.asarray(owner_scaled_normals)
        neighbour = jnp.asarray(neighbour_scaled_normals)
        if (
            coordinates.ndim != 2
            or weights.shape != (coordinates.shape[0],)
            or owner.shape != coordinates.shape
            or neighbour.shape != coordinates.shape
            or not jnp.issubdtype(coordinates.dtype, jnp.inexact)
            or not jnp.issubdtype(weights.dtype, jnp.inexact)
        ):
            raise ValueError("Mortar metric arrays have incompatible shapes.")
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "finite-element-mortar-metric",
                    "coordinates": array_tree_fingerprint(np.asarray(coordinates)),
                    "weights": array_tree_fingerprint(np.asarray(weights)),
                    "owner_normals": array_tree_fingerprint(np.asarray(owner)),
                    "neighbour_normals": array_tree_fingerprint(np.asarray(neighbour)),
                }
            )
            if metric_id is None
            else str(metric_id)
        )
        if not identifier:
            raise ValueError("metric_id must be non-empty.")
        self.physical_coordinates = coordinates
        self.physical_weights = weights
        self.owner_scaled_normals = owner
        self.neighbour_scaled_normals = neighbour
        self.metric_id = identifier

    @property
    def opposite_normal_error(self) -> Array:
        return jnp.max(jnp.abs(self.owner_scaled_normals + self.neighbour_scaled_normals))


class FiniteElementMortarEvidence(StrictModule, NonTrainableState):
    """Inspectable reproduction, coordinate, and conservation evidence."""

    left_polynomial_error: Array
    right_polynomial_error: Array
    mortar_polynomial_error: Array
    monomial_degrees: Array
    coordinate_error: Array
    declared_degree: tuple[int, ...] = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    coordinate_tolerance: float = eqx.field(static=True)
    constant_reproduced: bool = eqx.field(static=True)
    declared_polynomials_reproduced: bool = eqx.field(static=True)
    coordinates_compatible: bool = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        left_polynomial_error: ArrayLike,
        right_polynomial_error: ArrayLike,
        mortar_polynomial_error: ArrayLike,
        monomial_degrees: ArrayLike,
        coordinate_error: ArrayLike,
        declared_degree: tuple[int, ...],
        tolerance: float,
        coordinate_tolerance: float,
        /,
    ):
        left = np.asarray(left_polynomial_error)
        right = np.asarray(right_polynomial_error)
        mortar = np.asarray(mortar_polynomial_error)
        monomials = np.asarray(monomial_degrees, dtype=np.int32)
        coordinate = np.asarray(coordinate_error)
        degree = tuple(int(value) for value in declared_degree)
        tolerance_ = float(tolerance)
        coordinate_tolerance_ = float(coordinate_tolerance)
        if (
            left.ndim != 1
            or right.shape != left.shape
            or mortar.shape != left.shape
            or monomials.shape != (left.size, len(degree))
            or coordinate.shape != ()
            or tolerance_ <= 0.0
            or coordinate_tolerance_ <= 0.0
        ):
            raise ValueError("Mortar reproduction evidence is inconsistent.")
        constant_index = int(np.flatnonzero(np.all(monomials == 0, axis=1))[0])
        constant = bool(
            max(left[constant_index], right[constant_index], mortar[constant_index])
            <= tolerance_
        )
        reproduced = bool(
            max(
                float(np.max(left, initial=0.0)),
                float(np.max(right, initial=0.0)),
                float(np.max(mortar, initial=0.0)),
            )
            <= tolerance_
        )
        compatible = bool(float(coordinate) <= coordinate_tolerance_)
        self.left_polynomial_error = jnp.asarray(left)
        self.right_polynomial_error = jnp.asarray(right)
        self.mortar_polynomial_error = jnp.asarray(mortar)
        self.monomial_degrees = jnp.asarray(monomials)
        self.coordinate_error = jnp.asarray(coordinate)
        self.declared_degree = degree
        self.tolerance = tolerance_
        self.coordinate_tolerance = coordinate_tolerance_
        self.constant_reproduced = constant
        self.declared_polynomials_reproduced = reproduced
        self.coordinates_compatible = compatible
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "finite-element-mortar-evidence",
                "left": array_tree_fingerprint(left),
                "right": array_tree_fingerprint(right),
                "mortar": array_tree_fingerprint(mortar),
                "monomials": array_tree_fingerprint(monomials),
                "coordinate_error": array_tree_fingerprint(coordinate),
                "degree": list(degree),
                "tolerance": tolerance_,
                "coordinate_tolerance": coordinate_tolerance_,
            }
        )


class FiniteElementMortarPlan(StrictModule, NonTrainableState):
    """One serial tensor-product mortar patch with explicit transfer roles."""

    left_interpolation: Array
    right_interpolation: Array
    mortar_interpolation: Array
    left_orientation: Array
    right_orientation: Array
    quadrature_points: Array
    physical_coordinates: Array
    physical_weights: Array
    mortar_mass: Array
    left_mass: Array
    right_mass: Array
    left_raw_dual_pullback: Array
    right_raw_dual_pullback: Array
    left_weighted_pairing_pullback: Array
    right_weighted_pairing_pullback: Array
    left_pairing_adjoint: Array
    right_pairing_adjoint: Array
    left_mass_projection: Array
    right_mass_projection: Array
    evidence: FiniteElementMortarEvidence
    interface_id: str = eqx.field(static=True)
    child_index: int = eqx.field(static=True)
    child_count: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        left_interpolation: ArrayLike,
        right_interpolation: ArrayLike,
        mortar_interpolation: ArrayLike,
        left_orientation: ArrayLike,
        right_orientation: ArrayLike,
        quadrature_points: ArrayLike,
        physical_coordinates: ArrayLike,
        physical_weights: ArrayLike,
        mortar_mass: ArrayLike,
        left_mass: ArrayLike,
        right_mass: ArrayLike,
        left_pairing_adjoint: ArrayLike,
        right_pairing_adjoint: ArrayLike,
        left_mass_projection: ArrayLike,
        right_mass_projection: ArrayLike,
        evidence: FiniteElementMortarEvidence,
        interface_id: str,
        child_index: int,
        child_count: int,
        /,
    ):
        left = jnp.asarray(left_interpolation)
        right = jnp.asarray(right_interpolation)
        mortar = jnp.asarray(mortar_interpolation)
        if left.ndim != 2 or right.ndim != 2 or mortar.ndim != 2:
            raise ValueError("Mortar interpolation arrays must be rank-2.")
        left_orientation_ = _permutation(
            left_orientation, left.shape[1], "left_orientation"
        )
        right_orientation_ = _permutation(
            right_orientation, right.shape[1], "right_orientation"
        )
        quadrature = _point_array(quadrature_points, "quadrature_points")
        physical = _point_array(
            physical_coordinates,
            "physical_coordinates",
            allowed_dimensions=(1, 2, 3),
        )
        weights_array = np.asarray(physical_weights)
        weights = jnp.asarray(weights_array)
        mortar_mass_ = jnp.asarray(mortar_mass)
        left_mass_ = jnp.asarray(left_mass)
        right_mass_ = jnp.asarray(right_mass)
        left_adjoint = jnp.asarray(left_pairing_adjoint)
        right_adjoint = jnp.asarray(right_pairing_adjoint)
        left_projection = jnp.asarray(left_mass_projection)
        right_projection = jnp.asarray(right_mass_projection)
        identifier = str(interface_id)
        child = int(child_index)
        children = int(child_count)
        if (
            left.ndim != 2
            or right.ndim != 2
            or mortar.ndim != 2
            or not jnp.issubdtype(left.dtype, jnp.inexact)
            or not jnp.issubdtype(right.dtype, jnp.inexact)
            or not jnp.issubdtype(mortar.dtype, jnp.inexact)
            or left.shape[0] != right.shape[0]
            or left.shape[0] != mortar.shape[0]
            or quadrature.shape[0] != left.shape[0]
            or physical.shape[0] != left.shape[0]
            or weights.shape != (left.shape[0],)
            or not np.issubdtype(weights_array.dtype, np.inexact)
            or np.any(~np.isfinite(weights_array))
            or np.any(weights_array <= 0.0)
            or mortar_mass_.shape != (mortar.shape[1], mortar.shape[1])
            or left_mass_.shape != (left.shape[1], left.shape[1])
            or right_mass_.shape != (right.shape[1], right.shape[1])
            or left_adjoint.shape != (left.shape[1], mortar.shape[1])
            or right_adjoint.shape != (right.shape[1], mortar.shape[1])
            or left_projection.shape != (mortar.shape[1], left.shape[1])
            or right_projection.shape != (mortar.shape[1], right.shape[1])
        ):
            raise ValueError("Mortar transfer matrices have incompatible shapes.")
        if (
            not isinstance(evidence, FiniteElementMortarEvidence)
            or not evidence.constant_reproduced
            or not evidence.declared_polynomials_reproduced
            or not evidence.coordinates_compatible
        ):
            raise ValueError("Mortar construction evidence did not satisfy its contract.")
        if not identifier or children <= 0 or child < 0 or child >= children:
            raise ValueError("Mortar interface/child identity is invalid.")
        self.left_interpolation = left
        self.right_interpolation = right
        self.mortar_interpolation = mortar
        self.left_orientation = jnp.asarray(left_orientation_)
        self.right_orientation = jnp.asarray(right_orientation_)
        self.quadrature_points = jnp.asarray(quadrature)
        self.physical_coordinates = jnp.asarray(physical)
        self.physical_weights = weights
        self.mortar_mass = mortar_mass_
        self.left_mass = left_mass_
        self.right_mass = right_mass_
        self.left_raw_dual_pullback = left.T
        self.right_raw_dual_pullback = right.T
        self.left_weighted_pairing_pullback = left.T @ (weights[:, None] * mortar)
        self.right_weighted_pairing_pullback = right.T @ (weights[:, None] * mortar)
        self.left_pairing_adjoint = left_adjoint
        self.right_pairing_adjoint = right_adjoint
        self.left_mass_projection = left_projection
        self.right_mass_projection = right_projection
        self.evidence = evidence
        self.interface_id = identifier
        self.child_index = child
        self.child_count = children
        self.plan_id = canonical_fingerprint(
            {
                "kind": "finite-element-serial-mortar-plan",
                "interface": identifier,
                "child": child,
                "child_count": children,
                "left": array_tree_fingerprint(np.asarray(left)),
                "right": array_tree_fingerprint(np.asarray(right)),
                "mortar": array_tree_fingerprint(np.asarray(mortar)),
                "weights": array_tree_fingerprint(np.asarray(weights)),
                "quadrature": array_tree_fingerprint(quadrature),
                "physical_coordinates": array_tree_fingerprint(physical),
                "mortar_mass": array_tree_fingerprint(np.asarray(mortar_mass_)),
                "left_mass": array_tree_fingerprint(np.asarray(left_mass_)),
                "right_mass": array_tree_fingerprint(np.asarray(right_mass_)),
                "left_orientation": array_tree_fingerprint(
                    np.asarray(self.left_orientation)
                ),
                "right_orientation": array_tree_fingerprint(
                    np.asarray(self.right_orientation)
                ),
                "left_adjoint": array_tree_fingerprint(np.asarray(left_adjoint)),
                "right_adjoint": array_tree_fingerprint(np.asarray(right_adjoint)),
                "left_projection": array_tree_fingerprint(np.asarray(left_projection)),
                "right_projection": array_tree_fingerprint(np.asarray(right_projection)),
                "evidence": evidence.evidence_id,
            }
        )

    def interpolate_left(self, trace_values: ArrayLike, /) -> Array:
        values = jnp.asarray(trace_values)
        if values.shape[0] != self.left_interpolation.shape[1]:
            raise ValueError("Left trace values have incompatible shape.")
        return oe.contract("qi,i...->q...", self.left_interpolation, values)

    def interpolate_right(self, trace_values: ArrayLike, /) -> Array:
        values = jnp.asarray(trace_values)
        if values.shape[0] != self.right_interpolation.shape[1]:
            raise ValueError("Right trace values have incompatible shape.")
        return oe.contract("qi,i...->q...", self.right_interpolation, values)

    def pullback_left_raw(self, quadrature_dual: ArrayLike, /) -> Array:
        value = jnp.asarray(quadrature_dual)
        if value.shape[0] != self.left_interpolation.shape[0]:
            raise ValueError("Mortar quadrature dual has incompatible shape.")
        return oe.contract("iq,q...->i...", self.left_raw_dual_pullback, value)

    def pullback_right_raw(self, quadrature_dual: ArrayLike, /) -> Array:
        value = jnp.asarray(quadrature_dual)
        if value.shape[0] != self.right_interpolation.shape[0]:
            raise ValueError("Mortar quadrature dual has incompatible shape.")
        return oe.contract("iq,q...->i...", self.right_raw_dual_pullback, value)

    def mass_project_left(self, trace_values: ArrayLike, /) -> Array:
        values = jnp.asarray(trace_values)
        if values.shape[0] != self.left_mass_projection.shape[1]:
            raise ValueError("Left trace values have incompatible shape.")
        return oe.contract("mi,i...->m...", self.left_mass_projection, values)

    def mass_project_right(self, trace_values: ArrayLike, /) -> Array:
        values = jnp.asarray(trace_values)
        if values.shape[0] != self.right_mass_projection.shape[1]:
            raise ValueError("Right trace values have incompatible shape.")
        return oe.contract("mi,i...->m...", self.right_mass_projection, values)

    def pairing_adjoint_to_left(self, mortar_values: ArrayLike, /) -> Array:
        values = jnp.asarray(mortar_values)
        if values.shape[0] != self.left_pairing_adjoint.shape[1]:
            raise ValueError("Mortar values have incompatible shape.")
        return oe.contract("im,m...->i...", self.left_pairing_adjoint, values)

    def pairing_adjoint_to_right(self, mortar_values: ArrayLike, /) -> Array:
        values = jnp.asarray(mortar_values)
        if values.shape[0] != self.right_pairing_adjoint.shape[1]:
            raise ValueError("Mortar values have incompatible shape.")
        return oe.contract("im,m...->i...", self.right_pairing_adjoint, values)

    def integrated_flux(
        self,
        flux: ArrayLike,
        metric: FiniteElementMortarMetricData | None = None,
        /,
    ) -> Array:
        value = jnp.asarray(flux)
        physical_weights = (
            self.physical_weights if metric is None else metric.physical_weights
        )
        if value.shape[0] != physical_weights.shape[0]:
            raise ValueError("Mortar flux has incompatible quadrature shape.")
        weights = physical_weights.reshape(
            physical_weights.shape + (1,) * (value.ndim - 1)
        )
        return jnp.sum(weights * value, axis=0)

    def conservative_flux_contributions(
        self,
        flux: ArrayLike,
        metric: FiniteElementMortarMetricData | None = None,
        /,
    ) -> tuple[Array, Array]:
        value = jnp.asarray(flux)
        physical_weights = (
            self.physical_weights if metric is None else metric.physical_weights
        )
        if value.shape[0] != physical_weights.shape[0]:
            raise ValueError("Mortar flux has incompatible quadrature shape.")
        weights = physical_weights.reshape(
            physical_weights.shape + (1,) * (value.ndim - 1)
        )
        weighted = weights * value
        return self.pullback_left_raw(weighted), -self.pullback_right_raw(weighted)

    def conservation_residual(
        self,
        flux: ArrayLike,
        metric: FiniteElementMortarMetricData | None = None,
        /,
    ) -> Array:
        left, right = self.conservative_flux_contributions(flux, metric)
        return jnp.sum(left, axis=0) + jnp.sum(right, axis=0)


def serial_finite_element_mortar_plan(
    left_trace_nodes: ArrayLike,
    right_trace_nodes: ArrayLike,
    mortar_nodes: ArrayLike,
    quadrature_points: ArrayLike,
    quadrature_weights: ArrayLike,
    /,
    *,
    left_evaluation_points: ArrayLike | None = None,
    right_evaluation_points: ArrayLike | None = None,
    left_orientation: ArrayLike | None = None,
    right_orientation: ArrayLike | None = None,
    left_polynomial_coordinates: ArrayLike | None = None,
    right_polynomial_coordinates: ArrayLike | None = None,
    mortar_polynomial_coordinates: ArrayLike | None = None,
    polynomial_evaluation_points: ArrayLike | None = None,
    declared_reproduction_degree: int | tuple[int, ...] = 0,
    left_physical_coordinates: ArrayLike | None = None,
    right_physical_coordinates: ArrayLike | None = None,
    coordinate_measure: ArrayLike | None = None,
    reproduction_tolerance: float = 1.0e-10,
    coordinate_tolerance: float = 1.0e-10,
    interface_id: str = "mortar-interface",
    child_index: int = 0,
    child_count: int = 1,
) -> FiniteElementMortarPlan:
    """Build one serial mortar; side evaluation points encode child mappings."""

    left_nodes = _point_array(left_trace_nodes, "left_trace_nodes")
    right_nodes = _point_array(right_trace_nodes, "right_trace_nodes")
    mortar_nodes_ = _point_array(mortar_nodes, "mortar_nodes")
    quadrature = _point_array(quadrature_points, "quadrature_points")
    dimension = quadrature.shape[1]
    if any(
        points.shape[1] != dimension
        for points in (left_nodes, right_nodes, mortar_nodes_)
    ):
        raise ValueError("Mortar trace and quadrature dimensions must agree.")
    weights = np.asarray(quadrature_weights)
    if (
        weights.shape != (quadrature.shape[0],)
        or not np.issubdtype(weights.dtype, np.inexact)
        or np.any(~np.isfinite(weights))
        or np.any(weights <= 0.0)
    ):
        raise ValueError("Mortar quadrature weights must be positive and finite.")
    left_points = (
        quadrature
        if left_evaluation_points is None
        else _point_array(left_evaluation_points, "left_evaluation_points")
    )
    right_points = (
        quadrature
        if right_evaluation_points is None
        else _point_array(right_evaluation_points, "right_evaluation_points")
    )
    if left_points.shape != quadrature.shape or right_points.shape != quadrature.shape:
        raise ValueError("Side evaluation points must match mortar quadrature points.")
    left_permutation = _permutation(
        left_orientation, left_nodes.shape[0], "left_orientation"
    )
    right_permutation = _permutation(
        right_orientation, right_nodes.shape[0], "right_orientation"
    )
    left_base = _tensor_tabulation(left_nodes, left_points)
    right_base = _tensor_tabulation(right_nodes, right_points)
    mortar_basis = _tensor_tabulation(mortar_nodes_, quadrature)
    left_matrix = left_base @ np.eye(left_nodes.shape[0])[left_permutation]
    right_matrix = right_base @ np.eye(right_nodes.shape[0])[right_permutation]

    if (left_physical_coordinates is None) != (right_physical_coordinates is None):
        raise ValueError(
            "Curved-coordinate compatibility requires both side coordinates."
        )
    if left_physical_coordinates is None:
        physical = quadrature
        coordinate_error = np.asarray(0.0)
    else:
        if right_physical_coordinates is None:
            raise RuntimeError("Validated mortar coordinate pairing became inconsistent.")
        left_physical = _point_array(
            left_physical_coordinates,
            "left_physical_coordinates",
            allowed_dimensions=(1, 2, 3),
        )
        right_physical = _point_array(
            right_physical_coordinates,
            "right_physical_coordinates",
            allowed_dimensions=(1, 2, 3),
        )
        if (
            left_physical.shape != right_physical.shape
            or left_physical.shape[0] != quadrature.shape[0]
        ):
            raise ValueError("Curved mortar coordinates have incompatible shapes.")
        coordinate_error = np.asarray(np.max(np.abs(left_physical - right_physical)))
        physical = 0.5 * (left_physical + right_physical)
    measure = (
        np.ones(weights.shape, dtype=weights.dtype)
        if coordinate_measure is None
        else np.asarray(coordinate_measure)
    )
    if (
        measure.shape != weights.shape
        or not np.issubdtype(measure.dtype, np.inexact)
        or np.any(~np.isfinite(measure))
        or np.any(measure <= 0.0)
    ):
        raise ValueError("Mortar coordinate measure must be positive and finite.")
    physical_weights = weights * measure

    degree = (
        (int(declared_reproduction_degree),) * dimension
        if isinstance(declared_reproduction_degree, (int, np.integer))
        else tuple(int(value) for value in declared_reproduction_degree)
    )
    if len(degree) != dimension or any(value < 0 for value in degree):
        raise ValueError("Declared mortar reproduction degree is invalid.")
    left_common = (
        left_nodes
        if left_polynomial_coordinates is None
        else _point_array(left_polynomial_coordinates, "left_polynomial_coordinates")
    )
    right_common = (
        right_nodes
        if right_polynomial_coordinates is None
        else _point_array(right_polynomial_coordinates, "right_polynomial_coordinates")
    )
    mortar_common = (
        mortar_nodes_
        if mortar_polynomial_coordinates is None
        else _point_array(mortar_polynomial_coordinates, "mortar_polynomial_coordinates")
    )
    evaluation_common = (
        quadrature
        if polynomial_evaluation_points is None
        else _point_array(polynomial_evaluation_points, "polynomial_evaluation_points")
    )
    if (
        left_common.shape != left_nodes.shape
        or right_common.shape != right_nodes.shape
        or mortar_common.shape != mortar_nodes_.shape
        or evaluation_common.shape != quadrature.shape
    ):
        raise ValueError(
            "Polynomial coordinates must match their trace/mortar point arrays."
        )
    monomials = np.asarray(tuple(product(*(range(value + 1) for value in degree))))
    left_errors = []
    right_errors = []
    mortar_errors = []
    inverse_left = np.argsort(left_permutation)
    inverse_right = np.argsort(right_permutation)
    for powers in monomials:
        target = np.prod(evaluation_common ** powers[None, :], axis=1)
        left_canonical = np.prod(left_common ** powers[None, :], axis=1)
        right_canonical = np.prod(right_common ** powers[None, :], axis=1)
        mortar_values = np.prod(mortar_common ** powers[None, :], axis=1)
        left_local = left_canonical[inverse_left]
        right_local = right_canonical[inverse_right]
        left_errors.append(float(np.max(np.abs(left_matrix @ left_local - target))))
        right_errors.append(float(np.max(np.abs(right_matrix @ right_local - target))))
        mortar_errors.append(float(np.max(np.abs(mortar_basis @ mortar_values - target))))
    evidence = FiniteElementMortarEvidence(
        np.asarray(left_errors),
        np.asarray(right_errors),
        np.asarray(mortar_errors),
        monomials,
        coordinate_error,
        degree,
        reproduction_tolerance,
        coordinate_tolerance,
    )

    left_jax = jnp.asarray(left_matrix)
    right_jax = jnp.asarray(right_matrix)
    mortar_jax = jnp.asarray(mortar_basis)
    physical_weights_jax = jnp.asarray(physical_weights)
    weighted_left = physical_weights_jax[:, None] * left_jax
    weighted_right = physical_weights_jax[:, None] * right_jax
    weighted_mortar = physical_weights_jax[:, None] * mortar_jax
    left_mass = left_jax.T @ weighted_left
    right_mass = right_jax.T @ weighted_right
    mortar_mass = mortar_jax.T @ weighted_mortar
    left_pairing = left_jax.T @ weighted_mortar
    right_pairing = right_jax.T @ weighted_mortar
    left_adjoint = _mass_solve(left_mass, left_pairing, "left trace mass")
    right_adjoint = _mass_solve(right_mass, right_pairing, "right trace mass")
    left_projection = _mass_solve(
        mortar_mass, mortar_jax.T @ weighted_left, "mortar mass"
    )
    right_projection = _mass_solve(
        mortar_mass, mortar_jax.T @ weighted_right, "mortar mass"
    )
    return FiniteElementMortarPlan(
        left_jax,
        right_jax,
        mortar_jax,
        left_permutation,
        right_permutation,
        quadrature,
        physical,
        physical_weights_jax,
        mortar_mass,
        left_mass,
        right_mass,
        left_adjoint,
        right_adjoint,
        left_projection,
        right_projection,
        evidence,
        interface_id,
        child_index,
        child_count,
    )


__all__ = [
    "FiniteElementMortarEvidence",
    "FiniteElementMortarPlan",
    "FiniteElementMortarMetricData",
    "serial_finite_element_mortar_plan",
]
