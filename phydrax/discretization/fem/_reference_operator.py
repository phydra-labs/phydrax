#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._high_order import ReferenceNodalFamily, TensorProductTabulation
from ._precision import FiniteElementPrecisionPolicy
from ._reference import FiniteElementSpec
from ._reference_topology import reference_cell_topology


ReferenceRule: TypeAlias = Any
ReferenceCellData: TypeAlias = Any


ReferenceAction: TypeAlias = Literal[
    "interpolate",
    "interpolate_transpose",
    "gradient",
    "gradient_transpose",
    "trace",
    "trace_transpose",
]
_ACTION_ORDER: tuple[ReferenceAction, ...] = (
    "interpolate",
    "interpolate_transpose",
    "gradient",
    "gradient_transpose",
    "trace",
    "trace_transpose",
)


def _canonical_actions(
    actions: tuple[ReferenceAction, ...], /
) -> tuple[ReferenceAction, ...]:
    if not isinstance(actions, tuple) or not actions:
        raise ValueError("Reference actions must be a nonempty tuple.")
    unknown = tuple(action for action in actions if action not in _ACTION_ORDER)
    if unknown:
        raise ValueError(f"Unknown reference action {unknown[0]!r}.")
    return tuple(action for action in _ACTION_ORDER if action in actions)


def _rule_id(rule: ReferenceRule, data: ReferenceCellData, /) -> str:
    return canonical_fingerprint(
        {
            "kind": "finite-element-reference-rule",
            "rule_type": type(rule).__qualname__,
            "cell": data.cell,
            "points": array_tree_fingerprint(np.asarray(data.points)),
            "weights": array_tree_fingerprint(np.asarray(data.weights)),
            "embedded_weights": (
                None
                if data.embedded_weights is None
                else array_tree_fingerprint(np.asarray(data.embedded_weights))
            ),
        }
    )


def _exact_degree(rule: ReferenceRule, /) -> int | None:
    from ...integration._rules import (
        CubatureRule,
        interval_rule_data,
        ReferenceHexahedronRule,
        ReferenceIntervalRule,
        ReferenceQuadrilateralRule,
    )

    if isinstance(rule, CubatureRule):
        return rule.exact_degree
    if isinstance(rule, ReferenceIntervalRule):
        return interval_rule_data(rule.rule).degree
    if isinstance(
        rule,
        (ReferenceQuadrilateralRule, ReferenceHexahedronRule),
    ):
        return interval_rule_data(rule.rule).degree
    return None


def _map_edge_rule(
    cell_kind: str,
    facet_index: int,
    data: ReferenceCellData,
    /,
) -> tuple[Array, Array, Array]:
    if data.cell != "interval":
        raise ValueError("Two-dimensional facets require interval reference rules.")
    topology = reference_cell_topology(cell_kind)
    vertex_ids = topology.entities[1][facet_index]
    vertices = jnp.asarray(tuple(topology.vertices[index] for index in vertex_ids))
    parameter = jnp.asarray(data.points)[:, 0]
    tangent = vertices[1] - vertices[0]
    scale = jnp.linalg.norm(tangent)
    points = (1.0 - parameter[:, None]) * vertices[0] + parameter[:, None] * vertices[1]
    normal = jnp.asarray((tangent[1], -tangent[0])) / scale
    normals = jnp.broadcast_to(normal, points.shape)
    return points, jnp.asarray(data.weights) * scale, normals


def _map_face_rule(
    cell_kind: str,
    facet_index: int,
    data: ReferenceCellData,
    /,
) -> tuple[Array, Array, Array]:
    topology = reference_cell_topology(cell_kind)
    vertex_ids = topology.entities[2][facet_index]
    corners = jnp.asarray(tuple(topology.vertices[index] for index in vertex_ids))
    parameter = jnp.asarray(data.points)
    if len(vertex_ids) == 3:
        if data.cell != "triangle":
            raise ValueError("Triangular faces require triangle reference rules.")
        first = parameter[:, 0]
        second = parameter[:, 1]
        points = (
            (1.0 - first - second)[:, None] * corners[0]
            + first[:, None] * corners[1]
            + second[:, None] * corners[2]
        )
        normal_vector = jnp.cross(corners[1] - corners[0], corners[2] - corners[0])
        scale = jnp.linalg.norm(normal_vector)
        normal = normal_vector / scale
        normals = jnp.broadcast_to(normal, points.shape)
        return points, jnp.asarray(data.weights) * scale, normals
    if len(vertex_ids) != 4 or data.cell != "quadrilateral":
        raise ValueError("Quadrilateral faces require quadrilateral reference rules.")
    u = parameter[:, 0]
    v = parameter[:, 1]
    points = (
        ((1.0 - u) * (1.0 - v))[:, None] * corners[0]
        + (u * (1.0 - v))[:, None] * corners[1]
        + (u * v)[:, None] * corners[2]
        + ((1.0 - u) * v)[:, None] * corners[3]
    )
    tangent_u = (
        -(1.0 - v)[:, None] * corners[0]
        + (1.0 - v)[:, None] * corners[1]
        + v[:, None] * corners[2]
        - v[:, None] * corners[3]
    )
    tangent_v = (
        -(1.0 - u)[:, None] * corners[0]
        - u[:, None] * corners[1]
        + u[:, None] * corners[2]
        + (1.0 - u)[:, None] * corners[3]
    )
    normal_vectors = jnp.cross(tangent_u, tangent_v)
    scales = jnp.linalg.norm(normal_vectors, axis=-1)
    normals = normal_vectors / scales[:, None]
    return points, jnp.asarray(data.weights) * scales, normals


def _dense_interpolate(basis: Array, coefficients: ArrayLike, /) -> Array:
    values = jnp.asarray(coefficients)
    if values.ndim < 1 or values.shape[-1] != basis.shape[1]:
        raise ValueError("Reference coefficients have an incompatible DOF axis.")
    return oe.contract("qd,...d->...q", basis, values)


def _dense_interpolate_transpose(basis: Array, values: ArrayLike, /) -> Array:
    evaluated = jnp.asarray(values)
    if evaluated.ndim < 1 or evaluated.shape[-1] != basis.shape[0]:
        raise ValueError("Reference values have an incompatible point axis.")
    return oe.contract("qd,...q->...d", basis, evaluated)


def _dense_gradient(gradients: Array, coefficients: ArrayLike, /) -> Array:
    values = jnp.asarray(coefficients)
    if values.ndim < 1 or values.shape[-1] != gradients.shape[1]:
        raise ValueError("Reference coefficients have an incompatible DOF axis.")
    return oe.contract("qdk,...d->...qk", gradients, values)


def _dense_gradient_transpose(gradients: Array, values: ArrayLike, /) -> Array:
    evaluated = jnp.asarray(values)
    if evaluated.ndim < 2 or evaluated.shape[-2:] != (
        gradients.shape[0],
        gradients.shape[2],
    ):
        raise ValueError("Reference gradients have incompatible point/component axes.")
    return oe.contract("qdk,...qk->...d", gradients, evaluated)


class FiniteElementFacetReference(StrictModule, NonTrainableState):
    """Prepared values, gradients, weights, and normals on one oriented facet."""

    facet_index: int = eqx.field(static=True)
    rule_id: str = eqx.field(static=True)
    points: Array
    weights: Array
    normals: Array
    basis_values: Array
    basis_gradients: Array
    facet_id: str = eqx.field(static=True)

    def __init__(
        self,
        facet_index: int,
        rule_id: str,
        points: ArrayLike,
        weights: ArrayLike,
        normals: ArrayLike,
        basis_values: ArrayLike,
        basis_gradients: ArrayLike,
        /,
    ):
        index = int(facet_index)
        points_ = jnp.asarray(points)
        weights_ = jnp.asarray(weights)
        normals_ = jnp.asarray(normals)
        basis_ = jnp.asarray(basis_values)
        gradients_ = jnp.asarray(basis_gradients)
        if points_.ndim != 2 or weights_.shape != (points_.shape[0],):
            raise ValueError("Facet points and weights have incompatible shapes.")
        if normals_.shape != points_.shape:
            raise ValueError("Facet normals must match facet point coordinates.")
        if basis_.shape[0] != points_.shape[0] or gradients_.shape != (
            points_.shape[0],
            basis_.shape[1],
            points_.shape[1],
        ):
            raise ValueError("Facet basis tabulation has incompatible axes.")
        self.facet_index = index
        self.rule_id = str(rule_id)
        self.points = points_
        self.weights = weights_
        self.normals = normals_
        self.basis_values = basis_
        self.basis_gradients = gradients_
        self.facet_id = canonical_fingerprint(
            {
                "kind": "finite-element-facet-reference",
                "facet_index": index,
                "rule_id": self.rule_id,
                "points": array_tree_fingerprint(np.asarray(points_)),
                "weights": array_tree_fingerprint(np.asarray(weights_)),
                "normals": array_tree_fingerprint(np.asarray(normals_)),
                "basis_values": array_tree_fingerprint(np.asarray(basis_)),
                "basis_gradients": array_tree_fingerprint(np.asarray(gradients_)),
            }
        )

    def interpolate(self, coefficients: ArrayLike, /) -> Array:
        return _dense_interpolate(self.basis_values, coefficients)

    def interpolate_transpose(self, values: ArrayLike, /) -> Array:
        return _dense_interpolate_transpose(self.basis_values, values)

    def gradient(self, coefficients: ArrayLike, /) -> Array:
        return _dense_gradient(self.basis_gradients, coefficients)

    def gradient_transpose(self, values: ArrayLike, /) -> Array:
        return _dense_gradient_transpose(self.basis_gradients, values)


class FiniteElementReferenceReport(StrictModule, NonTrainableState):
    """Inspectible identity and shape evidence for one prepared reference."""

    element_id: str = eqx.field(static=True)
    volume_rule_id: str = eqx.field(static=True)
    facet_rule_ids: tuple[str, ...] = eqx.field(static=True)
    actions: tuple[ReferenceAction, ...] = eqx.field(static=True)
    precision_id: str = eqx.field(static=True)
    volume_exact_degree: int | None = eqx.field(static=True)
    facet_exact_degrees: tuple[int | None, ...] = eqx.field(static=True)
    point_count: int = eqx.field(static=True)
    facet_point_counts: tuple[int, ...] = eqx.field(static=True)
    tensor_factorized: bool = eqx.field(static=True)
    report_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        element_id: str,
        volume_rule_id: str,
        facet_rule_ids: tuple[str, ...],
        actions: tuple[ReferenceAction, ...],
        precision_id: str,
        volume_exact_degree: int | None,
        facet_exact_degrees: tuple[int | None, ...],
        point_count: int,
        facet_point_counts: tuple[int, ...],
        tensor_factorized: bool,
    ):
        content = {
            "kind": "finite-element-reference-report",
            "element_id": element_id,
            "volume_rule_id": volume_rule_id,
            "facet_rule_ids": facet_rule_ids,
            "actions": actions,
            "precision_id": precision_id,
            "volume_exact_degree": volume_exact_degree,
            "facet_exact_degrees": facet_exact_degrees,
            "point_count": int(point_count),
            "facet_point_counts": facet_point_counts,
            "tensor_factorized": bool(tensor_factorized),
        }
        self.element_id = str(element_id)
        self.volume_rule_id = str(volume_rule_id)
        self.facet_rule_ids = tuple(str(value) for value in facet_rule_ids)
        self.actions = actions
        self.precision_id = str(precision_id)
        self.volume_exact_degree = volume_exact_degree
        self.facet_exact_degrees = facet_exact_degrees
        self.point_count = int(point_count)
        self.facet_point_counts = tuple(int(value) for value in facet_point_counts)
        self.tensor_factorized = bool(tensor_factorized)
        self.report_id = canonical_fingerprint(content)


class PreparedFiniteElementReference(StrictModule, NonTrainableState):
    """Dense reference actions and optional tensor factors bound to explicit rules."""

    element: FiniteElementSpec
    volume_rule: ReferenceCellData
    facet_rules: tuple[ReferenceCellData, ...]
    actions: tuple[ReferenceAction, ...] = eqx.field(static=True)
    precision: FiniteElementPrecisionPolicy
    basis_values: Array
    basis_gradients: Array
    weights: Array
    facets: tuple[FiniteElementFacetReference, ...]
    tensor_tabulation: TensorProductTabulation | None
    tensor_weights_by_axis: tuple[Array, ...] | None
    report: FiniteElementReferenceReport
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        element: FiniteElementSpec,
        volume_rule: ReferenceRule,
        facet_rules: tuple[ReferenceRule, ...],
        actions: tuple[ReferenceAction, ...],
        precision: FiniteElementPrecisionPolicy,
        /,
        *,
        tensor_family: ReferenceNodalFamily | None = None,
    ):
        from ...integration._rules import (
            interval_rule_data,
            reference_rule_data,
            ReferenceCellData,
            ReferenceHexahedronRule,
            ReferenceQuadrilateralRule,
        )

        if not isinstance(element, FiniteElementSpec):
            raise TypeError("element must be FiniteElementSpec.")
        if element.cell_kind not in (
            "triangle",
            "quadrilateral",
            "tetrahedron",
            "hexahedron",
            "prism",
            "pyramid",
        ):
            raise ValueError("Prepared references require a supported FE cell.")
        if element.representation != "point_value":
            raise ValueError("Reference nodal actions require point-value coefficients.")
        if element.value_shape:
            raise ValueError("Reference nodal actions require a scalar finite element.")
        if not isinstance(precision, FiniteElementPrecisionPolicy):
            raise TypeError("precision must be FiniteElementPrecisionPolicy.")
        action_set = _canonical_actions(actions)
        topology = reference_cell_topology(element.cell_kind)
        expected_facets = len(topology.entities[element.topological_dimension - 1])
        if not isinstance(facet_rules, tuple) or len(facet_rules) != expected_facets:
            raise ValueError("One explicit reference rule is required for every facet.")

        volume_data_raw = reference_rule_data(volume_rule)
        if volume_data_raw.cell != element.cell_kind:
            raise ValueError(
                "The volume rule must match the finite-element reference cell."
            )
        volume_rule_id = _rule_id(volume_rule, volume_data_raw)
        points = precision.geometry(volume_data_raw.points)
        weights = precision.accumulation(volume_data_raw.weights)
        basis_values, basis_gradients = element.tabulate(points)
        if basis_values.shape != (points.shape[0], element.local_dof_count) or (
            basis_gradients.shape
            != (
                points.shape[0],
                element.local_dof_count,
                element.topological_dimension,
            )
        ):
            raise ValueError("Volume basis tabulation has incompatible axes.")
        basis_values = precision.evaluation(basis_values)
        basis_gradients = precision.evaluation(basis_gradients)
        volume_data = ReferenceCellData(
            points,
            weights,
            None
            if volume_data_raw.embedded_weights is None
            else precision.accumulation(volume_data_raw.embedded_weights),
            volume_data_raw.cell,
        )

        prepared_facets = []
        prepared_facet_rules = []
        facet_rule_ids = []
        for facet_index, rule in enumerate(facet_rules):
            data_raw = reference_rule_data(rule)
            rule_id = _rule_id(rule, data_raw)
            facet_points, facet_weights, facet_normals = (
                _map_edge_rule(element.cell_kind, facet_index, data_raw)
                if element.topological_dimension == 2
                else _map_face_rule(element.cell_kind, facet_index, data_raw)
            )
            facet_points = precision.geometry(facet_points)
            facet_weights = precision.accumulation(facet_weights)
            facet_normals = precision.geometry(facet_normals)
            facet_values, facet_gradients = element.tabulate(facet_points)
            facet_values = precision.evaluation(facet_values)
            facet_gradients = precision.evaluation(facet_gradients)
            prepared_facets.append(
                FiniteElementFacetReference(
                    facet_index,
                    rule_id,
                    facet_points,
                    facet_weights,
                    facet_normals,
                    facet_values,
                    facet_gradients,
                )
            )
            prepared_facet_rules.append(
                ReferenceCellData(
                    precision.geometry(data_raw.points),
                    precision.accumulation(data_raw.weights),
                    None
                    if data_raw.embedded_weights is None
                    else precision.accumulation(data_raw.embedded_weights),
                    data_raw.cell,
                )
            )
            facet_rule_ids.append(rule_id)

        tensor_tabulation = None
        tensor_weights = None
        if tensor_family is not None:
            if not isinstance(tensor_family, ReferenceNodalFamily):
                raise TypeError("tensor_family must be ReferenceNodalFamily or None.")
            tensor_element = tensor_family.finite_element()
            compatible_element = tensor_element.element_id == element.element_id
            compatible_discontinuous = (
                element.family == "DiscontinuousLagrange"
                and element.tabulator_id == f"discontinuous:{tensor_element.element_id}"
            )
            if not compatible_element and not compatible_discontinuous:
                raise ValueError(
                    "tensor_family must construct the prepared nodal element."
                )
            tensor_rule_types = {
                "quadrilateral": ReferenceQuadrilateralRule,
                "hexahedron": ReferenceHexahedronRule,
            }
            expected_type = tensor_rule_types[element.cell_kind]
            if not isinstance(volume_rule, expected_type):
                raise ValueError(
                    "Tensor factors require an explicit tensor-product volume rule."
                )
            axis_data = interval_rule_data(volume_rule.rule)
            axis_points = precision.geometry(0.5 * (axis_data.nodes + 1.0))
            axis_weights = precision.accumulation(0.5 * axis_data.weights)
            points_by_axis = (axis_points,) * element.topological_dimension
            tensor_tabulation = TensorProductTabulation(tensor_family, points_by_axis)
            tensor_weights = (axis_weights,) * element.topological_dimension

        facets = tuple(prepared_facets)
        facet_data = tuple(prepared_facet_rules)
        facet_ids = tuple(facet_rule_ids)
        report = FiniteElementReferenceReport(
            element_id=element.element_id,
            volume_rule_id=volume_rule_id,
            facet_rule_ids=facet_ids,
            actions=action_set,
            precision_id=precision.policy_id,
            volume_exact_degree=_exact_degree(volume_rule),
            facet_exact_degrees=tuple(_exact_degree(rule) for rule in facet_rules),
            point_count=int(points.shape[0]),
            facet_point_counts=tuple(int(facet.points.shape[0]) for facet in facets),
            tensor_factorized=tensor_tabulation is not None,
        )
        self.element = element
        self.volume_rule = volume_data
        self.facet_rules = facet_data
        self.actions = action_set
        self.precision = precision
        self.basis_values = basis_values
        self.basis_gradients = basis_gradients
        self.weights = weights
        self.facets = facets
        self.tensor_tabulation = tensor_tabulation
        self.tensor_weights_by_axis = tensor_weights
        self.report = report
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-finite-element-reference",
                "report_id": report.report_id,
                "basis_values": array_tree_fingerprint(np.asarray(basis_values)),
                "basis_gradients": array_tree_fingerprint(np.asarray(basis_gradients)),
                "weights": array_tree_fingerprint(np.asarray(weights)),
                "facets": tuple(facet.facet_id for facet in facets),
                "tensor_tabulation": (
                    None if tensor_tabulation is None else tensor_tabulation.tabulation_id
                ),
                "tensor_weights_by_axis": (
                    None
                    if tensor_weights is None
                    else tuple(
                        array_tree_fingerprint(np.asarray(value))
                        for value in tensor_weights
                    )
                ),
            }
        )

    def interpolate(self, coefficients: ArrayLike, /) -> Array:
        return _dense_interpolate(self.basis_values, coefficients)

    def interpolate_transpose(self, values: ArrayLike, /) -> Array:
        return _dense_interpolate_transpose(self.basis_values, values)

    def gradient(self, coefficients: ArrayLike, /) -> Array:
        return _dense_gradient(self.basis_gradients, coefficients)

    def gradient_transpose(self, values: ArrayLike, /) -> Array:
        return _dense_gradient_transpose(self.basis_gradients, values)

    def trace(self, facet_index: int, coefficients: ArrayLike, /) -> Array:
        return self.facets[int(facet_index)].interpolate(coefficients)

    def trace_transpose(self, facet_index: int, values: ArrayLike, /) -> Array:
        return self.facets[int(facet_index)].interpolate_transpose(values)


__all__ = [
    "FiniteElementFacetReference",
    "FiniteElementReferenceReport",
    "PreparedFiniteElementReference",
    "ReferenceAction",
]
