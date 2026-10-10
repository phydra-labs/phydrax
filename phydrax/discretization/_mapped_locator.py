#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Whole-source enclosures and the shared bounded inverse for mapped FE cells."""

from __future__ import annotations

from fractions import Fraction

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._bvh import BVHBuildPolicy, prepare_bvh
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..typing import checked
from ._cell_geometry import (
    _require_scalar_coordinate_element,
    CellGeometryElement,
    LayerColumnCellGeometryElement,
    PolynomialComposedCellGeometryElement,
    RationalComposedCellGeometryElement,
    RestrictedCellGeometryElement,
    SplineCellGeometryElement,
)
from ._coordinate_enclosure import (
    coordinate_expressions,
    expression_bounds,
    expression_node_count,
)
from ._reference_cell import reference_cell_topology
from ._simplicial_locator import (
    _AbstractPreparedNewtonCellLocator,
    _cell_map_vertices,
    _is_affine_simplex_element,
    _vertex_star_routes,
    SimplicialLocationPolicy,
)
from .fem._cell_map import PreparedFiniteElementCellMap
from .fem._reference import FiniteElementSpec


_REFERENCE_DOMAINS = {
    "interval": "box",
    "triangle": "simplex",
    "tetrahedron": "simplex",
    "quadrilateral": "box",
    "hexahedron": "box",
    "prism": "prism",
    # Coordinate source expressions use the collapsed cube; execution uses
    # physical pyramid coordinates (x,y,z), not that cube.
    "pyramid": "box",
}


def _reference_domain(cell_kind: str, /) -> str:
    reference_cell_topology(cell_kind)
    if cell_kind.startswith("simplex:"):
        return "simplex"
    if cell_kind.startswith("tensor:"):
        return "box"
    try:
        return _REFERENCE_DOMAINS[cell_kind]
    except KeyError:
        raise ValueError(f"Unsupported mapped reference cell {cell_kind!r}.") from None


def reference_margin(cell_kind: str, reference: Array, /) -> Array:
    """Signed inward chart margin, explicitly not a physical distance."""
    if cell_kind in ("triangle", "tetrahedron") or cell_kind.startswith("simplex:"):
        return jnp.minimum(jnp.min(reference, axis=-1), 1.0 - jnp.sum(reference, axis=-1))
    if cell_kind == "prism":
        triangle = jnp.minimum(
            jnp.min(reference[..., :2], axis=-1),
            1.0 - jnp.sum(reference[..., :2], axis=-1),
        )
        height = jnp.minimum(reference[..., 2], 1.0 - reference[..., 2])
        return jnp.minimum(triangle, height)
    if cell_kind == "pyramid":
        height = reference[..., 2]
        lower = 0.5 * height
        upper = 1.0 - lower
        return jnp.minimum(
            jnp.minimum(height, 1.0 - height),
            jnp.minimum(
                jnp.min(reference[..., :2] - lower[..., None], axis=-1),
                jnp.min(upper[..., None] - reference[..., :2], axis=-1),
            ),
        )
    if cell_kind in ("interval", "quadrilateral", "hexahedron") or cell_kind.startswith(
        "tensor:"
    ):
        return jnp.min(jnp.minimum(reference, 1.0 - reference), axis=-1)
    raise ValueError(f"Unsupported mapped reference cell {cell_kind!r}.")


def _exact_inside(kind: str, point: tuple[Fraction, ...], /) -> bool:
    if kind in ("triangle", "tetrahedron") or kind.startswith("simplex:"):
        return min(point) >= 0 and sum(point) <= 1
    if kind == "prism":
        return min(point[:2]) >= 0 and sum(point[:2]) <= 1 and 0 <= point[2] <= 1
    if kind == "pyramid":
        return 0 <= point[2] <= 1 and all(
            point[2] / 2 <= value <= 1 - point[2] / 2 for value in point[:2]
        )
    return all(0 <= value <= 1 for value in point)


def _validate_restrictions(
    element: CellGeometryElement, /
) -> (
    FiniteElementSpec
    | LayerColumnCellGeometryElement
    | PolynomialComposedCellGeometryElement
    | RationalComposedCellGeometryElement
    | SplineCellGeometryElement
):
    """An ancestor enclosure is valid only for a true source-domain restriction."""
    while isinstance(element, RestrictedCellGeometryElement):
        matrix = tuple(
            tuple(Fraction(float(value)) for value in row)
            for row in np.asarray(element.matrix)
        )
        offset = tuple(Fraction(float(value)) for value in np.asarray(element.offset))
        for vertex in reference_cell_topology(element.cell_kind).vertices:
            point = tuple(Fraction(value) for value in vertex)
            source = tuple(
                start
                + sum(
                    value * coordinate
                    for value, coordinate in zip(row, point, strict=True)
                )
                for start, row in zip(offset, matrix, strict=True)
            )
            if not _exact_inside(element.source_element.cell_kind, source):
                raise ValueError(
                    "Restricted coordinate map leaves its canonical source domain."
                )
        element = element.source_element
    element = _require_scalar_coordinate_element(element, "Mapped coordinate sources")
    if isinstance(element, RestrictedCellGeometryElement):
        raise RuntimeError(
            "Reference source validation left an unresolved affine restriction."
        )
    if not isinstance(
        element,
        (
            FiniteElementSpec,
            LayerColumnCellGeometryElement,
            PolynomialComposedCellGeometryElement,
            RationalComposedCellGeometryElement,
            SplineCellGeometryElement,
        ),
    ):
        raise TypeError("Mapped coordinate sources require a supported scalar element.")
    return element


def _bound_budget(cell_map: PreparedFiniteElementCellMap, capacity: int, /) -> None:
    """Bound source expansion before gathering local arrays or making polynomials."""
    if capacity < 1:
        raise ValueError("maximum_bound_coefficients must be positive.")
    element = _validate_restrictions(cell_map.coordinate_element)
    dimension = element.topological_dimension
    degree = max(1, element.degree)
    # Tensor-to-affine composition has total degree at most dimension*degree.
    # Pyramid collapse/removable-denominator work needs two further orders.
    order = (dimension + (2 if element.cell_kind == "pyramid" else 0)) * degree
    terms = (order + 1) ** dimension
    required = terms * (
        element.local_dof_count + cell_map.cell_count * cell_map.ambient_dimension
    )
    required += cell_map.cell_count * element.local_dof_count * cell_map.ambient_dimension
    if required > capacity:
        raise ValueError(
            f"Mapped locator source-bound coefficient capacity exceeded: {required} > {capacity}."
        )


def _mapped_cell_bounds(
    cell_map: PreparedFiniteElementCellMap, coordinates: np.ndarray, capacity: int, /
) -> tuple[np.ndarray, np.ndarray]:
    _bound_budget(cell_map, capacity)
    shape = (cell_map.cell_count, cell_map.ambient_dimension)
    lower = np.empty(shape, dtype=np.float64)
    upper = np.empty(shape, dtype=np.float64)
    routes = np.asarray(cell_map.coordinate_dofs)
    source_coordinates = cell_map.source_coordinates(coordinates)
    for cell, route in enumerate(routes):
        local = tuple(source_coordinates[index] for index in route)
        element: CellGeometryElement = cell_map.coordinate_element
        polynomials = coordinate_expressions(element, local)
        if polynomials is None:
            raise ValueError(
                "Mapped locator requires a canonical coordinate source enclosure; this source is unsupported."
            )
        domain = _reference_domain(element.cell_kind)
        dimension = reference_cell_topology(element.cell_kind).dimension
        for axis, polynomial in enumerate(polynomials):
            count = expression_node_count(polynomial, domain, dimension)
            if count > capacity:
                raise ValueError(
                    "Mapped locator source-bound coefficient capacity exceeded."
                )
            lower[cell, axis], upper[cell, axis] = expression_bounds(
                polynomial, domain, dimension
            )
    if not np.all(np.isfinite(lower) & np.isfinite(upper)):
        raise ValueError("Mapped locator source enclosures must be finite.")
    return lower, upper


class PreparedMappedCellLocator(_AbstractPreparedNewtonCellLocator):
    """Actual prepared coordinate-map inverse on simplex, tensor, hybrid or child cells.

    The Newton iteration and bounded candidate assembly are shared with the
    simplicial locator. Only source enclosure and reference-domain operations
    differ. Preparation refuses unsupported sources before claiming support;
    singular geometry and bounded inverse/candidate failure remain query data.
    """

    cell_lower: Array
    cell_upper: Array
    source_binding_id: str = eqx.field(static=True)
    maximum_bound_coefficients: int = eqx.field(static=True)

    @checked
    def __init__(
        self,
        cell_map: PreparedFiniteElementCellMap,
        coordinates: ArrayLike,
        policy: SimplicialLocationPolicy,
        /,
        *,
        maximum_bound_coefficients: int = 2_000_000,
    ) -> None:
        _reference_domain(cell_map.coordinate_element.cell_kind)
        values = jnp.asarray(coordinates)
        if values.shape != (cell_map.coordinate_count, cell_map.ambient_dimension):
            raise ValueError("Locator coordinates do not match the prepared cell map.")
        host = np.asarray(values, dtype=np.float64)
        if not np.all(np.isfinite(host)):
            raise ValueError("Mapped locator coordinate source must be finite.")
        capacity = int(maximum_bound_coefficients)
        lower, upper = _mapped_cell_bounds(cell_map, host, capacity)
        # Keep the enclosure outward when a lower precision execution is used.
        lower_jax = jnp.nextafter(jnp.asarray(lower, dtype=values.dtype), -jnp.inf)
        upper_jax = jnp.nextafter(jnp.asarray(upper, dtype=values.dtype), jnp.inf)
        self.cell_map = cell_map
        self.coordinates = values
        self.cells = cell_map.coordinate_dofs
        if _is_affine_simplex_element(cell_map.coordinate_element):
            star_cells, star_valid = _vertex_star_routes(_cell_map_vertices(cell_map))
        else:
            star_cells = np.zeros((cell_map.cell_count, 1), dtype=np.int32)
            star_valid = np.zeros(star_cells.shape, dtype=np.bool_)
        self.vertex_star_cells = jnp.asarray(star_cells)
        self.vertex_star_valid = jnp.asarray(star_valid)
        center = self._reference_seeds(values.dtype)[0]
        self.centroids = cell_map.evaluate(
            values,
            jnp.arange(cell_map.cell_count),
            jnp.broadcast_to(center, (cell_map.cell_count, cell_map.reference_dimension)),
        ).physical_points
        self.cell_lower, self.cell_upper = lower_jax, upper_jax
        self.bvh = prepare_bvh(
            np.asarray(lower_jax),
            np.asarray(upper_jax),
            policy=BVHBuildPolicy(leaf_size=min(16, cell_map.cell_count)),
            dtype=values.dtype,
        )
        self.policy = policy
        self.maximum_bound_coefficients = capacity
        self.source_binding_id = canonical_fingerprint(
            {
                "kind": "whole-mapped-cell-source-binding",
                "cell_map": cell_map.cell_map_id,
                "coordinates": array_tree_fingerprint(values),
                "lower": array_tree_fingerprint(lower_jax),
                "upper": array_tree_fingerprint(upper_jax),
            }
        )
        self.locator_id = canonical_fingerprint(
            {
                "kind": "prepared-mapped-cell-locator",
                "source": self.source_binding_id,
                "policy": policy.policy_id,
            }
        )

    def _reference_seeds(self, dtype: jnp.dtype, /) -> Array:
        kind = self.cell_map.coordinate_element.cell_kind
        topology = reference_cell_topology(kind)
        vertices = jnp.asarray(topology.vertices, dtype=dtype)
        if kind in ("triangle", "tetrahedron") or kind.startswith("simplex:"):
            center = jnp.full((self.dimension,), 1.0 / (self.dimension + 1), dtype=dtype)
        elif kind == "prism":
            center = jnp.asarray((1.0 / 3.0, 1.0 / 3.0, 0.5), dtype=dtype)
        elif kind == "pyramid":
            center = jnp.asarray((0.5, 0.5, 0.25), dtype=dtype)
        else:
            center = jnp.full((self.dimension,), 0.5, dtype=dtype)
        # Target-reference vertices exist even when source coefficient routes
        # are parent DOFs and a restricted element has no reference_nodes.
        return jnp.concatenate((center[None], vertices), axis=0)

    def _inside_reference(self, reference: Array, /) -> Array:
        return (
            reference_margin(self.cell_map.coordinate_element.cell_kind, reference)
            >= -self.policy.reference_tolerance
        )

    def _reference_weights(self, reference: Array, /) -> Array:
        kind = self.cell_map.coordinate_element.cell_kind
        if kind in ("triangle", "tetrahedron") or kind.startswith("simplex:"):
            return super()._reference_weights(reference)
        if kind == "prism":
            triangle = jnp.concatenate(
                ((1.0 - jnp.sum(reference[:, :2], axis=-1))[:, None], reference[:, :2]),
                axis=-1,
            )
            return jnp.concatenate(
                (
                    triangle * (1.0 - reference[:, 2, None]),
                    triangle * reference[:, 2, None],
                ),
                axis=-1,
            )
        vertices = jnp.asarray(
            reference_cell_topology(kind).vertices, dtype=reference.dtype
        )
        if kind == "pyramid":
            height = reference[:, 2]
            collapse = 1.0 - height
            safe = jnp.where(collapse != 0.0, collapse, 1.0)
            chart = (reference[:, :2] - 0.5 * height[:, None]) / safe[:, None]
            base = (
                jnp.prod(
                    jnp.where(
                        vertices[None, :4, :2] == 1.0,
                        chart[:, None, :],
                        1.0 - chart[:, None, :],
                    ),
                    axis=-1,
                )
                * collapse[:, None]
            )
            return jnp.concatenate((base, height[:, None]), axis=-1)
        return jnp.prod(
            jnp.where(
                vertices[None] == 1.0, reference[:, None], 1.0 - reference[:, None]
            ),
            axis=-1,
        )


__all__ = ["PreparedMappedCellLocator", "reference_margin"]
