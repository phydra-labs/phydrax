#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from fractions import Fraction
from functools import cache
from itertools import product
from typing import final, Protocol, runtime_checkable, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, core as jax_core
from jax.sharding import NamedSharding, PartitionSpec
from jax.typing import ArrayLike

from .._fingerprint import (
    array_tree_fingerprint,
    canonical_fingerprint,
    logical_array_value_collection_digest,
)
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..typing import checked, ConvertibleToArray
from ._cell_mesh import CellBlock, CellMesh, CellMeshStorage, PolyhedralBlock
from ._exact_plc_geometry import (
    ExactPlcCellGeometryConvexSource,
    ExactPlcCellGeometrySource,
)
from ._exact_power_geometry import (
    ExactPowerCellGeometryLinearActionSource,
    ExactPowerCellGeometryRestrictionSource,
    ExactPowerCellGeometrySource,
)
from ._periodic_geometry import PeriodicCellGeometrySource


# The closed set of owning exact coordinate sources. Each kind keeps its own
# scientific identity; consumers dispatch nominally and never reinterpret one
# kind through another's construction.
type ExactCellGeometrySource = (
    ExactPowerCellGeometrySource
    | ExactPowerCellGeometryRestrictionSource
    | ExactPowerCellGeometryLinearActionSource
    | ExactPlcCellGeometrySource
    | ExactPlcCellGeometryConvexSource
)

if TYPE_CHECKING:
    from ._coordinate_enclosure import CoordinateSourceBank, Expression
    from .fem._reference import FiniteElementSpec


@runtime_checkable
class CellGeometryElement(Protocol):
    """Common layout metadata, including nonpolynomial vertex descriptors.

    Scalar tabulation and source order belong to validated concrete coordinate
    elements, not to every layout descriptor. Source order alone does not
    establish polynomiality; certificates inspect the exact source expressions.
    """

    @property
    def cell_kind(self) -> str: ...

    @property
    def conformity(self) -> str: ...

    @property
    def local_dof_count(self) -> int: ...

    @property
    def element_id(self) -> str: ...


class _CoordinateTabulator(StrictModule, NonTrainableState):
    """Numerical evaluation of the canonical coordinate source definition.

    The source is exact nodal interpolation on the rational equispaced lattice,
    in the authoritative polynomial space or collapsed rational pyramid space.
    Floating evaluation approximates this source; it does not define a new
    rational map from rounded barycentric weights or a rounded change of basis.
    Certificates extract the exact source expressions, never sampled values.
    """

    cell_kind: str = eqx.field(static=True)
    degree: int = eqx.field(static=True)
    source_basis_semantics: str = eqx.field(
        static=True, default="exact-rational-equispaced-reference-interpolation"
    )

    def __call__(self, points: ArrayLike, /) -> tuple[Array, Array]:
        from ._coordinate_enclosure import evaluate_coordinate_lattice_basis

        return evaluate_coordinate_lattice_basis(self.cell_kind, self.degree, points)


@cache
def coordinate_lagrange_element(cell_kind: str, degree: int, /) -> FiniteElementSpec:
    """Conforming equispaced coordinates in the authoritative reference space.

    Solution nodes may be warped or Lobatto nodes. Coordinate traces instead
    share one lattice across simplex, tensor, prism and rational pyramid cells.
    Exact source expressions belong to the shared coordinate enclosure owner,
    including the rational pyramid, not a polynomial surrogate.
    """
    from ..exterior._form_type import FormType, FormValueSpec
    from ._coordinate_enclosure import evaluate_coordinate_lattice_basis
    from ._reference_cell import reference_cell_topology
    from .fem._reference import FiniteElementSpec

    if isinstance(degree, bool) or not isinstance(degree, (int, np.integer)):
        raise TypeError("degree must be an integer.")
    if degree < 1 or degree > 10:
        raise ValueError("Coordinate elements require degrees from 1 through 10.")
    topology = reference_cell_topology(cell_kind)
    dimension = topology.dimension
    if degree == 1:
        nodes = np.asarray(topology.vertices, dtype=np.float64)
    elif cell_kind == "pyramid":
        nodes = np.asarray(
            [
                ((i + 0.5 * k) / degree, (j + 0.5 * k) / degree, k / degree)
                for k in range(degree + 1)
                for i in range(degree - k + 1)
                for j in range(degree - k + 1)
            ],
            dtype=np.float64,
        )
    else:
        indices = tuple(product(range(degree + 1), repeat=dimension))
        if cell_kind in ("triangle", "tetrahedron") or cell_kind.startswith("simplex:"):
            indices = tuple(row for row in indices if sum(row) <= degree)
        elif cell_kind == "prism":
            indices = tuple(row for row in indices if row[0] + row[1] <= degree)
        elif cell_kind not in (
            "interval",
            "quadrilateral",
            "hexahedron",
        ) and not cell_kind.startswith("tensor:"):
            raise ValueError("Unsupported coordinate cell kind.")
        nodes = np.asarray(indices, dtype=np.float64) / degree
    weights = np.asarray(
        evaluate_coordinate_lattice_basis(cell_kind, 1, nodes)[0], dtype=np.float64
    )
    entity_sets = [
        [frozenset(vertices) for vertices in entities] for entities in topology.entities
    ]
    entity_dofs = [[[] for _ in entities] for entities in topology.entities]
    for dof, row in enumerate(weights):
        support = frozenset(np.flatnonzero(row > 1.0e-10).tolist())
        matches = [
            (dim, entities.index(support))
            for dim, entities in enumerate(entity_sets)
            if support in entities
        ]
        if len(matches) != 1:
            raise ValueError("Coordinate node must own exactly one reference entity.")
        dim, entity = matches[0]
        entity_dofs[dim][entity].append(dof)
    tabulator = _CoordinateTabulator(cell_kind, degree)
    family = (
        "HybridLagrange"
        if cell_kind in ("prism", "pyramid")
        else "TensorProductLagrange"
        if cell_kind in ("interval", "quadrilateral", "hexahedron")
        or cell_kind.startswith("tensor:")
        else "SimplexLagrange"
    )
    return FiniteElementSpec(
        family,
        cell_kind,
        degree,
        nodes,
        tuple(tuple(tuple(dofs) for dofs in entities) for entities in entity_dofs),
        value_spec=FormValueSpec(
            FormType(dimension, 0, twist="untwisted"), proxy="scalar"
        ),
        tabulator=tabulator,
        tabulator_id=canonical_fingerprint(
            {
                "kind": "conforming-coordinate-lattice",
                "cell_kind": cell_kind,
                "degree": degree,
                "source_basis_semantics": tabulator.source_basis_semantics,
                "nodes": array_tree_fingerprint(nodes),
            }
        ),
    )


class _SweptCoordinateTabulator(StrictModule, NonTrainableState):
    """Exact tensor lift of an authored profile basis, linear in station space."""

    source: FiniteElementSpec

    def __call__(self, points: ArrayLike, /) -> tuple[Array, Array]:
        reference = jnp.asarray(points, dtype=jnp.float64)
        if reference.ndim != 2 or reference.shape[1] != 3:
            raise ValueError("Swept coordinate points require three reference axes.")
        values, gradients = self.source.tabulate(reference[:, :2])
        w = reference[:, 2:3]
        lower = jnp.concatenate(
            (gradients * (1.0 - w[..., None]), -values[..., None]), axis=-1
        )
        upper = jnp.concatenate((gradients * w[..., None], values[..., None]), axis=-1)
        return jnp.concatenate(
            (values * (1.0 - w), values * w), axis=-1
        ), jnp.concatenate((lower, upper), axis=1)


def swept_coordinate_element(source: FiniteElementSpec, /) -> FiniteElementSpec:
    """Retain the full polynomial profile source without nodal reconstruction.

    Only two endpoint banks are required, irrespective of profile order. The
    declared space is anisotropic: the source profile space times P1 in the
    station axis. Absolute frames interpolate affinely between authored stations.
    """
    from ..exterior._form_type import FormType, FormValueSpec
    from ._coordinate_enclosure import source_basis
    from ._reference_cell import reference_cell_topology
    from .fem._reference import FiniteElementSpec

    if not isinstance(source, FiniteElementSpec):
        raise TypeError("Swept coordinates require an owning FiniteElementSpec profile.")
    if (
        source.cell_kind not in ("triangle", "quadrilateral")
        or source.conformity != "H1"
        or source.degree < 1
    ):
        raise ValueError(
            "Swept coordinate sources require H1 triangle or quadrilateral profiles."
        )
    if source.mapping != "identity" or source.value_shape or source_basis(source) is None:
        raise ValueError(
            "Swept coordinate sources require an actual canonical scalar polynomial basis."
        )
    kind = "prism" if source.cell_kind == "triangle" else "hexahedron"
    topology = reference_cell_topology(kind)
    source_topology = reference_cell_topology(source.cell_kind)
    node_count = source.local_dof_count
    corner_count = len(source_topology.vertices)
    nodes = np.concatenate(
        tuple(
            np.column_stack(
                (
                    np.asarray(source.reference_nodes),
                    np.full(node_count, side, dtype=np.float64),
                )
            )
            for side in (0.0, 1.0)
        )
    )
    entity_sets = tuple(
        tuple(frozenset(vertices) for vertices in level) for level in topology.entities
    )
    entity_dofs = [[[] for _ in level] for level in topology.entities]
    for side in range(2):
        for dimension, level in enumerate(source.entity_dofs):
            for entity, dofs in enumerate(level):
                vertices = frozenset(
                    vertex + side * corner_count
                    for vertex in source_topology.entities[dimension][entity]
                )
                target = entity_sets[dimension].index(vertices)
                entity_dofs[dimension][target].extend(
                    dof + side * node_count for dof in dofs
                )
    tabulator = _SweptCoordinateTabulator(source)
    return FiniteElementSpec(
        "SweptCoordinate",
        kind,
        source.degree,
        nodes,
        tuple(tuple(tuple(dofs) for dofs in level) for level in entity_dofs),
        value_spec=FormValueSpec(FormType(3, 0, twist="untwisted"), proxy="scalar"),
        representation=source.representation,
        tabulator=tabulator,
        tabulator_id=canonical_fingerprint(
            {
                "kind": "exact-swept-coordinate-source",
                "profile": source.element_id,
                "station_axis": 2,
                "station_degree": 1,
                "endpoint_order": (0, 1),
            }
        ),
    )


class SplineCellGeometryElement(StrictModule, NonTrainableState):
    """One exact knot-span chart of an original rational B-spline surface.

    Geometry coefficients are the original control points, in row-major order.
    Knots and rational weights remain numerical leaves of the original source;
    no fitted Bernstein control net or sampled triangle defines this map.
    """

    u_knots: Array
    v_knots: Array
    weights: Array
    u_degree: int = eqx.field(static=True)
    v_degree: int = eqx.field(static=True)
    span_indices: tuple[int, int] = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    source_shape: tuple[int, int] = eqx.field(static=True)
    cell_kind: str = eqx.field(static=True)
    conformity: str = eqx.field(static=True)
    local_dof_count: int = eqx.field(static=True)
    degree: int = eqx.field(static=True)
    topological_dimension: int = eqx.field(static=True)
    element_id: str = eqx.field(static=True)

    def __init__(
        self,
        u_knots: ArrayLike,
        v_knots: ArrayLike,
        weights: ArrayLike,
        u_degree: int,
        v_degree: int,
        span_indices: tuple[int, int],
        source_id: str,
        source_revision: str,
        /,
    ) -> None:
        knots = (
            np.asarray(u_knots, dtype=np.float64),
            np.asarray(v_knots, dtype=np.float64),
        )
        weights_ = np.asarray(weights, dtype=np.float64)
        if (
            weights_.ndim != 2
            or not np.all(np.isfinite(weights_))
            or np.all(weights_ == 0.0)
        ):
            raise ValueError(
                "Spline coordinates require a finite, nonzero rational-weight grid."
            )
        degrees = (u_degree, v_degree)
        if len(span_indices) != 2:
            raise ValueError("A spline surface chart requires two knot-span indices.")
        for axis, (vector, degree, span) in enumerate(
            zip(knots, degrees, span_indices, strict=True)
        ):
            if isinstance(degree, bool) or not isinstance(degree, (int, np.integer)):
                raise TypeError("Spline coordinate degrees must be integers.")
            if isinstance(span, bool) or not isinstance(span, (int, np.integer)):
                raise TypeError("Spline coordinate spans must be integer indices.")
            count = weights_.shape[axis]
            if degree < 1 or degree >= count:
                raise ValueError(
                    "Spline coordinate degrees require more source controls than their order."
                )
            if vector.ndim != 1 or vector.size != count + degree + 1:
                raise ValueError(
                    "Spline coordinate knot vectors differ from the original control grid."
                )
            if not np.all(np.isfinite(vector)) or np.any(np.diff(vector) < 0.0):
                raise ValueError(
                    "Spline coordinate knot vectors must be finite and nondecreasing."
                )
            if not degree <= span < count or not vector[span] < vector[span + 1]:
                raise ValueError(
                    "A spline coordinate chart requires a nonempty active knot span."
                )
        if not isinstance(source_id, str) or not isinstance(source_revision, str):
            raise TypeError(
                "Spline coordinates require explicit original source identity and revision."
            )
        if not source_id or not source_revision:
            raise ValueError(
                "Spline coordinate source identity and revision must be nonempty."
            )
        self.u_knots, self.v_knots = (jnp.asarray(vector) for vector in knots)
        self.weights = jnp.asarray(weights_)
        self.u_degree, self.v_degree = (int(value) for value in degrees)
        self.span_indices = (int(span_indices[0]), int(span_indices[1]))
        self.source_id, self.source_revision = source_id, source_revision
        self.source_shape = (weights_.shape[0], weights_.shape[1])
        self.cell_kind, self.conformity = "quadrilateral", "H1"
        self.local_dof_count = weights_.size
        self.degree, self.topological_dimension = max(self.u_degree, self.v_degree), 2
        self.element_id = canonical_fingerprint(
            {
                "kind": "original-rational-spline-coordinate-span",
                "source": source_id,
                "revision": source_revision,
                "knots": tuple(array_tree_fingerprint(vector) for vector in knots),
                "weights": array_tree_fingerprint(weights_),
                "degrees": degrees,
                "spans": self.span_indices,
            }
        )

    def tabulate(self, points: ArrayLike, /) -> tuple[Array, Array]:
        from .._interpolation import (
            bspline_jet_stencil,
            RationalSplineJet,
            TensorBSplineJetPlan,
        )

        reference = jnp.asarray(points, dtype=jnp.float64)
        if reference.ndim != 2 or reference.shape[1] != 2:
            raise ValueError("Spline chart points require two reference coordinates.")
        widths = jnp.stack(
            tuple(
                vector[span + 1] - vector[span]
                for vector, span in zip(
                    (self.u_knots, self.v_knots), self.span_indices, strict=True
                )
            )
        )
        origins = jnp.stack(
            tuple(
                vector[span]
                for vector, span in zip(
                    (self.u_knots, self.v_knots), self.span_indices, strict=True
                )
            )
        )

        def evaluate(point: Array) -> tuple[Array, Array]:
            parameters = origins + widths * point
            stencils = tuple(
                bspline_jet_stencil(
                    vector,
                    parameters[axis],
                    degree=degree,
                    maximum_order=1,
                    spans=jnp.asarray(span, dtype=jnp.int32),
                    bounds="extrapolate",
                )
                for axis, (vector, degree, span) in enumerate(
                    zip(
                        (self.u_knots, self.v_knots),
                        (self.u_degree, self.v_degree),
                        self.span_indices,
                        strict=True,
                    )
                )
            )
            rational = RationalSplineJet(
                TensorBSplineJetPlan(stencils, maximum_order=1), self.weights
            )
            values = (
                jnp.zeros((self.local_dof_count,), dtype=jnp.float64)
                .at[rational.indices]
                .set(
                    rational.derivative((0, 0)),
                )
            )
            gradients = jnp.zeros((self.local_dof_count, 2), dtype=jnp.float64)
            gradients = gradients.at[rational.indices, 0].set(
                rational.derivative((1, 0)) * widths[0]
            )
            gradients = gradients.at[rational.indices, 1].set(
                rational.derivative((0, 1)) * widths[1]
            )
            return values, gradients

        return jax.vmap(evaluate)(reference)


@final
class LayerColumnCellGeometryElement(StrictModule, NonTrainableState):
    """Endpoint profile sources plus exact authored corner corrections.

    ``fiber_graph`` declares a separately validated map that preserves the
    transverse affine carrier and adds one station-independent displacement
    along the final physical coordinate. General column maps leave it false.

    Scalar controls are bottom/top profile banks, actual bottom/top corner
    banks, then original bottom/top profile-corner banks. Corrections keep the
    positive and negative physical banks separate, never a rounded difference.
    """

    wall_element: CellGeometryElement
    corner_element: CellGeometryElement
    cell_kind: str = eqx.field(static=True)
    conformity: str = eqx.field(static=True)
    local_dof_count: int = eqx.field(static=True)
    degree: int = eqx.field(static=True)
    topological_dimension: int = eqx.field(static=True)
    station_axis: int = eqx.field(static=True)
    element_id: str = eqx.field(static=True)
    fiber_graph: bool = eqx.field(static=True)

    def __init__(
        self, wall_element: CellGeometryElement, /, *, fiber_graph: bool = False
    ) -> None:
        from ._coordinate_enclosure import source_basis
        from .fem._reference import FiniteElementSpec

        if not isinstance(wall_element, FiniteElementSpec):
            raise TypeError(
                "Layer columns require an original scalar finite-element profile source."
            )
        if (
            wall_element.cell_kind not in ("triangle", "quadrilateral")
            or wall_element.conformity != "H1"
            or wall_element.mapping != "identity"
            or wall_element.value_shape
            or wall_element.representation != "point_value"
            or source_basis(wall_element) is None
        ):
            raise ValueError(
                "Layer columns require a canonical scalar H1 polynomial profile."
            )
        if not isinstance(fiber_graph, bool):
            raise TypeError("fiber_graph must be a bool.")
        corners = coordinate_lagrange_element(wall_element.cell_kind, 1)
        self.wall_element, self.corner_element = wall_element, corners
        self.cell_kind = "prism" if wall_element.cell_kind == "triangle" else "hexahedron"
        self.conformity, self.topological_dimension = "H1", 3
        self.local_dof_count = (
            2 * wall_element.local_dof_count + 4 * corners.local_dof_count
        )
        self.degree = max(wall_element.degree, 1)
        self.station_axis = 2
        self.fiber_graph = fiber_graph
        self.element_id = canonical_fingerprint(
            {
                "kind": "layer-column-coordinate-source",
                "profile": wall_element.element_id,
                "corners": corners.element_id,
                "profile_axes": (0, 1),
                "station_axis": self.station_axis,
                "control_banks": (
                    "bottom_profile",
                    "top_profile",
                    "actual_bottom",
                    "actual_top",
                    "profile_bottom_corners",
                    "profile_top_corners",
                ),
                "fiber_graph": fiber_graph,
            }
        )

    def fiber_reference_expressions(
        self, local: CoordinateSourceBank, /
    ) -> tuple[Expression, ...] | None:
        """Return the exact affine fiber carrier for a declared graph map."""
        if not self.fiber_graph:
            return None
        from ._coordinate_enclosure import (
            coordinate_expressions,
            expression_derivative,
            expression_scale,
            expression_sum,
        )

        if len(local) != self.local_dof_count:
            raise ValueError(
                "Layer fiber-graph validation requires its complete source bank."
            )
        physical = coordinate_expressions(self, local)
        profile_count = self.wall_element.local_dof_count
        corner_count = self.corner_element.local_dof_count
        affine = coordinate_expressions(
            coordinate_lagrange_element(self.cell_kind, 1),
            local[2 * profile_count : 2 * profile_count + 2 * corner_count],
        )
        if physical is None or affine is None:
            raise ValueError(
                "Layer fiber graphs require exact physical and affine source expressions."
            )
        displacement = expression_sum((physical[-1], expression_scale(affine[-1], -1)))
        if physical[:-1] != affine[:-1] or expression_derivative(
            displacement, self.station_axis
        ):
            raise ValueError(
                "A declared layer fiber graph must preserve every transverse "
                "coordinate and one station-independent fiber displacement."
            )
        return affine

    def tabulate(self, points: ConvertibleToArray, /) -> tuple[Array, Array]:
        from .fem._reference import FiniteElementSpec

        if not isinstance(self.wall_element, FiniteElementSpec) or not isinstance(
            self.corner_element, FiniteElementSpec
        ):
            raise TypeError(
                "Layer column tabulation requires its canonical profile and corner owners."
            )
        reference = jnp.asarray(points, dtype=jnp.float64)
        if reference.ndim != 2 or reference.shape[1] != 3:
            raise ValueError("Layer column points must have three reference coordinates.")
        station = reference[:, 2:3]

        def lift(element: FiniteElementSpec) -> tuple[Array, Array, Array, Array]:
            values, gradients = element.tabulate(reference[:, :2])
            lower, upper = values * (1.0 - station), values * station
            lower_gradient = jnp.concatenate(
                (gradients * (1.0 - station)[:, None, :], -values[..., None]), axis=-1
            )
            upper_gradient = jnp.concatenate(
                (gradients * station[:, None, :], values[..., None]), axis=-1
            )
            return lower, upper, lower_gradient, upper_gradient

        bottom_profile, top_profile, bottom_gradient, top_gradient = lift(
            self.wall_element
        )
        bottom_corner, top_corner, bottom_corner_gradient, top_corner_gradient = lift(
            self.corner_element
        )
        return (
            jnp.concatenate(
                (
                    bottom_profile,
                    top_profile,
                    bottom_corner,
                    top_corner,
                    -bottom_corner,
                    -top_corner,
                ),
                axis=1,
            ),
            jnp.concatenate(
                (
                    bottom_gradient,
                    top_gradient,
                    bottom_corner_gradient,
                    top_corner_gradient,
                    -bottom_corner_gradient,
                    -top_corner_gradient,
                ),
                axis=1,
            ),
        )


def _require_p1_cardinal_source(
    source_element: CellGeometryElement, /
) -> FiniteElementSpec:
    from ._coordinate_enclosure import coordinate_source_signature
    from .fem._reference import FiniteElementSpec

    if not isinstance(source_element, FiniteElementSpec):
        raise TypeError(
            "Full barycentric actions require the actual canonical P1 source element."
        )
    if source_element.cell_kind not in ("triangle", "tetrahedron"):
        raise ValueError(
            "Full barycentric coefficient actions require an original simplex P1 source."
        )
    canonical = coordinate_lagrange_element(source_element.cell_kind, 1)
    if coordinate_source_signature(source_element) != coordinate_source_signature(
        canonical
    ):
        raise ValueError(
            "The source does not declare the canonical P1 cardinal vertex coefficient bank."
        )
    return source_element


@final
class BarycentricCellGeometryElement(StrictModule, NonTrainableState):
    """Retained full coefficient actions over one canonical affine P1 source."""

    source_element: FiniteElementSpec | BarycentricCellGeometryElement
    barycentric_weights: Array
    cell_kind: str = eqx.field(static=True)
    conformity: str = eqx.field(static=True)
    local_dof_count: int = eqx.field(static=True)
    degree: int = eqx.field(static=True)
    topological_dimension: int = eqx.field(static=True)
    element_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_element: FiniteElementSpec | BarycentricCellGeometryElement,
        barycentric_weights: ArrayLike,
        /,
    ) -> None:
        root = source_element
        while isinstance(root, BarycentricCellGeometryElement):
            root = root.source_element
        _require_p1_cardinal_source(root)
        weights = np.asarray(barycentric_weights, dtype=np.float64)
        width = source_element.local_dof_count
        if weights.shape != (width, width) or not np.all(np.isfinite(weights)):
            raise ValueError(
                "Full barycentric weights must bind every target and source vertex."
            )
        self.source_element = source_element
        self.barycentric_weights = jnp.asarray(weights)
        self.cell_kind = source_element.cell_kind
        self.conformity = "H1"
        self.local_dof_count = width
        self.degree = 1
        self.topological_dimension = source_element.topological_dimension
        self.element_id = canonical_fingerprint(
            {
                "kind": "full-barycentric-cell-geometry",
                "source": source_element.element_id,
                "weights": array_tree_fingerprint(weights),
            }
        )

    def tabulate(self, points: ArrayLike, /) -> tuple[Array, Array]:
        root = self.source_element
        while isinstance(root, BarycentricCellGeometryElement):
            root = root.source_element
        values, gradients = root.tabulate(points)
        current: FiniteElementSpec | BarycentricCellGeometryElement = self
        while isinstance(current, BarycentricCellGeometryElement):
            values = values @ current.barycentric_weights
            gradients = jnp.swapaxes(
                jnp.swapaxes(gradients, 1, 2) @ current.barycentric_weights,
                1,
                2,
            )
            current = current.source_element
        return values, gradients


def _require_full_p1_source(
    source_element: CellGeometryElement, /
) -> FiniteElementSpec | BarycentricCellGeometryElement:
    """Validate the actual retained action chain without replacing its owner."""
    if not isinstance(source_element, BarycentricCellGeometryElement):
        return _require_p1_cardinal_source(source_element)
    root = source_element.source_element
    while isinstance(root, BarycentricCellGeometryElement):
        root = root.source_element
    _require_p1_cardinal_source(root)
    return source_element


class RestrictedCellGeometryElement(StrictModule, NonTrainableState):
    """Exact source basis restricted by ``source_xi = A @ target_xi + b``.

    Coefficients remain source coefficients, not interpolated child nodes.
    Such maps require separate validity evidence; rational restrictions must
    never be silently relabeled as standard polynomial Lagrange elements.
    """

    source_element: (
        FiniteElementSpec
        | BarycentricCellGeometryElement
        | RestrictedCellGeometryElement
        | PolynomialComposedCellGeometryElement
        | RationalComposedCellGeometryElement
        | SplineCellGeometryElement
        | LayerColumnCellGeometryElement
    )
    cell_kind: str = eqx.field(static=True)
    matrix: Array
    offset: Array
    conformity: str = eqx.field(static=True)
    local_dof_count: int = eqx.field(static=True)
    degree: int = eqx.field(static=True)
    topological_dimension: int = eqx.field(static=True)
    element_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_element: FiniteElementSpec
        | BarycentricCellGeometryElement
        | RestrictedCellGeometryElement
        | PolynomialComposedCellGeometryElement
        | RationalComposedCellGeometryElement
        | SplineCellGeometryElement
        | LayerColumnCellGeometryElement,
        cell_kind: str,
        matrix: ArrayLike,
        offset: ArrayLike,
        /,
    ) -> None:
        from ._reference_cell import reference_cell_topology
        from .fem._reference import FiniteElementSpec

        if not isinstance(
            source_element,
            (
                FiniteElementSpec,
                BarycentricCellGeometryElement,
                RestrictedCellGeometryElement,
                PolynomialComposedCellGeometryElement,
                RationalComposedCellGeometryElement,
                SplineCellGeometryElement,
                LayerColumnCellGeometryElement,
            ),
        ):
            raise TypeError("source_element must support scalar coordinate tabulation.")
        if source_element.conformity != "H1":
            raise ValueError("Restricted coordinate elements must be H1 conforming.")
        if isinstance(source_element, FiniteElementSpec) and (
            source_element.value_shape or source_element.mapping != "identity"
        ):
            raise ValueError("Restrictions require scalar identity-mapped coordinates.")
        dimension = reference_cell_topology(cell_kind).dimension
        matrix_ = np.asarray(matrix, dtype=np.float64)
        offset_ = np.asarray(offset, dtype=np.float64)
        if matrix_.shape != (source_element.topological_dimension, dimension):
            raise ValueError("Restriction matrix dimensions must match both references.")
        if offset_.shape != (source_element.topological_dimension,):
            raise ValueError("Restriction offset must match the source reference.")
        if not np.all(np.isfinite(matrix_)) or not np.all(np.isfinite(offset_)):
            raise ValueError("Restriction map must be finite.")
        if np.linalg.matrix_rank(matrix_) != dimension:
            raise ValueError("Restriction map must have full target rank.")
        self.source_element = source_element
        self.cell_kind = cell_kind
        self.matrix = jnp.asarray(matrix_)
        self.offset = jnp.asarray(offset_)
        self.conformity = "H1"
        self.local_dof_count = source_element.local_dof_count
        self.degree = source_element.degree
        self.topological_dimension = dimension
        self.element_id = canonical_fingerprint(
            {
                "kind": "restricted-cell-geometry-element",
                "source": source_element.element_id,
                "target": cell_kind,
                "matrix": array_tree_fingerprint(matrix_),
                "offset": array_tree_fingerprint(offset_),
            }
        )

    def tabulate(self, points: ArrayLike, /) -> tuple[Array, Array]:
        reference = jnp.asarray(points, dtype=jnp.float64)
        if reference.ndim != 2 or reference.shape[1] != self.topological_dimension:
            raise ValueError("Restriction points must match the target reference.")
        values, gradients = self.source_element.tabulate(
            reference @ self.matrix.T + self.offset
        )
        return values, gradients @ self.matrix


class PolynomialComposedCellGeometryElement(StrictModule, NonTrainableState):
    """Original coordinate source pulled back through an exact polynomial chart.

    Source coefficients remain in their original basis. The chart coefficients
    are exact rational numbers; numerical tabulation rounds their evaluation,
    while certificates compose the original source expressions exactly.
    """

    source_element: (
        FiniteElementSpec
        | BarycentricCellGeometryElement
        | RestrictedCellGeometryElement
        | PolynomialComposedCellGeometryElement
        | RationalComposedCellGeometryElement
        | SplineCellGeometryElement
        | LayerColumnCellGeometryElement
    )
    chart_element: FiniteElementSpec
    chart_coordinates: Array
    chart_coefficients: tuple[tuple[tuple[int, int], ...], ...] = eqx.field(static=True)
    cell_kind: str = eqx.field(static=True)
    conformity: str = eqx.field(static=True)
    local_dof_count: int = eqx.field(static=True)
    degree: int = eqx.field(static=True)
    topological_dimension: int = eqx.field(static=True)
    element_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_element: FiniteElementSpec
        | BarycentricCellGeometryElement
        | RestrictedCellGeometryElement
        | PolynomialComposedCellGeometryElement
        | RationalComposedCellGeometryElement
        | SplineCellGeometryElement
        | LayerColumnCellGeometryElement,
        chart_element: FiniteElementSpec,
        chart_numerators: ConvertibleToArray,
        chart_denominators: ConvertibleToArray,
        /,
    ) -> None:
        from ._coordinate_enclosure import (
            _prepare_reference_chart_arguments,
            expression_parts,
            source_basis,
            source_expressions,
        )
        from .fem._reference import FiniteElementSpec

        source = _require_scalar_coordinate_element(source_element, "Composed source")
        chart = _require_scalar_coordinate_element(chart_element, "Reference chart")
        if not isinstance(chart, FiniteElementSpec):
            raise TypeError(
                "A reference chart requires an original finite-element basis."
            )
        if source.cell_kind == "pyramid" or chart.cell_kind == "pyramid":
            raise ValueError(
                "Polynomial composition cannot replace a collapsed rational pyramid chart."
            )
        source_basis_, chart_basis_ = source_expressions(source), source_basis(chart)
        if source_basis_ is None or chart_basis_ is None:
            raise ValueError(
                "Polynomial reference composition requires actual source expressions and a polynomial chart."
            )
        numerators, denominators = (
            np.asarray(chart_numerators),
            np.asarray(chart_denominators),
        )
        for bank in (numerators, denominators):
            if bank.dtype == object:
                if any(type(value) is not int for value in bank.flat):
                    raise TypeError(
                        "Exact reference chart object banks must contain only Python integers."
                    )
            elif bank.dtype.kind not in "iu":
                raise TypeError(
                    "Reference chart numerator and denominator arrays must contain integers."
                )
        shape = (chart.local_dof_count, source.topological_dimension)
        if numerators.shape != shape or denominators.shape != shape:
            raise ValueError(
                "Reference chart coefficients must match chart DOFs and source dimension."
            )
        if np.any(denominators == 0):
            raise ValueError("Reference chart coefficients require nonzero denominators.")
        coefficients = tuple(
            tuple(
                (value.numerator, value.denominator)
                for value in (
                    Fraction(int(a), int(b)) for a, b in zip(left, right, strict=True)
                )
            )
            for left, right in zip(
                numerators.tolist(), denominators.tolist(), strict=True
            )
        )
        self.source_element, self.chart_element = source, chart
        self.chart_coefficients = coefficients
        self.chart_coordinates = jnp.asarray(
            [[a / b for a, b in row] for row in coefficients],
            dtype=jnp.float64,
        )
        self.cell_kind, self.conformity = chart.cell_kind, "H1"
        self.local_dof_count = source.local_dof_count
        arguments = _prepare_reference_chart_arguments(
            chart, coefficients, source.topological_dimension
        )
        if chart.cell_kind in (
            "triangle",
            "tetrahedron",
            "interval",
        ) or chart.cell_kind.startswith("simplex:"):
            axis_groups = (tuple(range(chart.topological_dimension)),)
        elif chart.cell_kind == "prism":
            axis_groups = ((0, 1), (2,))
        else:
            axis_groups = tuple((axis,) for axis in range(chart.topological_dimension))
        chart_degrees = tuple(
            tuple(
                max((sum(index[axis] for axis in group) for index in argument), default=0)
                for argument in arguments
            )
            for group in axis_groups
        )
        # Count exact source-space powers per target axis. A diagonal/permuted
        # tensor chart therefore preserves its authored per-axis order, while
        # genuine mixing raises the bound without inspecting physical controls.
        self.degree = max(
            (
                sum(power * degree for power, degree in zip(index, degrees, strict=True))
                for degrees in chart_degrees
                for term in source_basis_
                for polynomial in expression_parts(term, source.topological_dimension)
                for index in polynomial
            ),
            default=0,
        )
        self.topological_dimension = chart.topological_dimension
        self.element_id = canonical_fingerprint(
            {
                "kind": "polynomial-composed-cell-geometry",
                "source": source.element_id,
                "chart": chart.element_id,
                "chart_coefficients": coefficients,
            }
        )

    def tabulate(self, points: ArrayLike, /) -> tuple[Array, Array]:
        reference = jnp.asarray(points, dtype=jnp.float64)
        if reference.ndim != 2 or reference.shape[1] != self.topological_dimension:
            raise ValueError("Composition points must match the target reference.")
        chart_values, chart_gradients = self.chart_element.tabulate(reference)
        source_points = chart_values @ self.chart_coordinates
        chart_jacobian = jnp.swapaxes(
            jnp.swapaxes(chart_gradients, -1, -2) @ self.chart_coordinates,
            -1,
            -2,
        )
        values, gradients = self.source_element.tabulate(source_points)
        return values, gradients @ chart_jacobian


class RationalComposedCellGeometryElement(StrictModule, NonTrainableState):
    """Original P1 simplex law pulled back through a true collapsed pyramid.

    The exact chart is stored at the five canonical rational pyramid nodes.
    Certificates use its exact collapsed polynomial representation; execution
    uses the canonical rational pyramid basis and its continuous apex value.
    No polynomial physical-reference surrogate or fitted coefficient bank exists.
    The returned apex gradient is the canonical centered-chart limit, not a
    Fréchet derivative when the base has a genuine bilinear cross term.
    ``reference_derivative_status`` reports that distinction explicitly;
    derivatives with respect to original source coefficients remain linear.
    """

    source_element: FiniteElementSpec | BarycentricCellGeometryElement
    chart_element: FiniteElementSpec
    chart_coordinates: Array
    chart_coefficients: tuple[tuple[tuple[int, int], ...], ...] = eqx.field(static=True)
    cell_kind: str = eqx.field(static=True)
    conformity: str = eqx.field(static=True)
    local_dof_count: int = eqx.field(static=True)
    degree: int = eqx.field(static=True)
    topological_dimension: int = eqx.field(static=True)
    element_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_element: FiniteElementSpec | BarycentricCellGeometryElement,
        chart_element: FiniteElementSpec,
        chart_numerators: ConvertibleToArray,
        chart_denominators: ConvertibleToArray,
        /,
    ) -> None:
        from .fem._reference import FiniteElementSpec

        source = _require_full_p1_source(source_element)
        if source.cell_kind != "tetrahedron":
            raise ValueError(
                "Rational simplex pullbacks require their original P1 tetrahedral source."
            )
        if (
            not isinstance(chart_element, FiniteElementSpec)
            or chart_element.element_id
            != coordinate_lagrange_element("pyramid", 1).element_id
        ):
            raise ValueError(
                "A rational reference composition requires the canonical collapsed pyramid chart."
            )
        numerators, denominators = (
            np.asarray(chart_numerators),
            np.asarray(chart_denominators),
        )
        for bank in (numerators, denominators):
            if bank.dtype == object:
                if any(type(value) is not int for value in bank.flat):
                    raise TypeError(
                        "Exact rational chart banks require Python integer entries."
                    )
            elif bank.dtype.kind not in "iu":
                raise TypeError("Exact rational chart banks require integer entries.")
        if (
            numerators.shape != (5, 3)
            or denominators.shape != (5, 3)
            or np.any(denominators == 0)
        ):
            raise ValueError(
                "Rational pyramid charts require five exact source-reference triples."
            )
        exact = tuple(
            tuple(Fraction(int(a), int(b)) for a, b in zip(left, right, strict=True))
            for left, right in zip(numerators, denominators, strict=True)
        )
        if any(min(row) < 0 or sum(row, Fraction(0)) > 1 for row in exact):
            raise ValueError(
                "Rational pyramid chart corners must lie in the original source simplex."
            )
        self.source_element, self.chart_element = source, chart_element
        self.chart_coefficients = tuple(
            tuple((value.numerator, value.denominator) for value in row) for row in exact
        )
        self.chart_coordinates = jnp.asarray(
            tuple(tuple(float(value) for value in row) for row in exact),
            dtype=jnp.float64,
        )
        self.cell_kind, self.conformity = "pyramid", "H1"
        self.local_dof_count, self.degree, self.topological_dimension = (
            source.local_dof_count,
            1,
            3,
        )
        self.element_id = canonical_fingerprint(
            {
                "kind": "rational-collapsed-pyramid-source-composition",
                "source": source.element_id,
                "chart": chart_element.element_id,
                "chart_coefficients": self.chart_coefficients,
            }
        )

    def reference_derivative_status(self, points: ArrayLike, /) -> Array:
        """Admit the reference derivative only where the actual source law has one."""
        reference = jnp.asarray(points, dtype=jnp.float64)
        if reference.ndim != 2 or reference.shape[1] != 3:
            raise ValueError(
                "Rational derivative status requires pyramid reference points."
            )
        affine = all(
            Fraction(*self.chart_coefficients[0][axis])
            - Fraction(*self.chart_coefficients[1][axis])
            + Fraction(*self.chart_coefficients[2][axis])
            - Fraction(*self.chart_coefficients[3][axis])
            == 0
            for axis in range(3)
        )
        apex = jnp.all(reference == jnp.asarray([0.5, 0.5, 1.0]), axis=1)
        return ~apex | affine

    def tabulate(self, points: ArrayLike, /) -> tuple[Array, Array]:
        reference = jnp.asarray(points, dtype=jnp.float64)
        if reference.ndim != 2 or reference.shape[1] != 3:
            raise ValueError(
                "Rational composition points must match the pyramid reference."
            )
        chart_values, chart_gradients = self.chart_element.tabulate(reference)
        source_points = chart_values @ self.chart_coordinates
        chart_jacobian = jnp.swapaxes(
            jnp.swapaxes(chart_gradients, -1, -2) @ self.chart_coordinates, -1, -2
        )
        values, gradients = self.source_element.tabulate(source_points)
        return values, gradients @ chart_jacobian


def _require_scalar_coordinate_element(
    element: CellGeometryElement, role: str, /
) -> (
    FiniteElementSpec
    | BarycentricCellGeometryElement
    | RestrictedCellGeometryElement
    | PolynomialComposedCellGeometryElement
    | RationalComposedCellGeometryElement
    | SplineCellGeometryElement
    | LayerColumnCellGeometryElement
):
    """Narrow the actual scalar coordinate capability, never a claimed family name."""
    from .fem._reference import FiniteElementSpec

    if not isinstance(
        element,
        (
            FiniteElementSpec,
            BarycentricCellGeometryElement,
            RestrictedCellGeometryElement,
            PolynomialComposedCellGeometryElement,
            RationalComposedCellGeometryElement,
            SplineCellGeometryElement,
            LayerColumnCellGeometryElement,
        ),
    ):
        raise TypeError(
            f"{role} requires a canonical tabulated scalar coordinate element."
        )
    if element.conformity != "H1":
        raise ValueError(f"{role} requires H1 coordinate elements.")
    if isinstance(element, FiniteElementSpec) and (
        element.value_shape or element.mapping != "identity"
    ):
        raise ValueError(f"{role} requires scalar identity-mapped coordinate values.")
    return element


class CellVertexGeometryElement(StrictModule, NonTrainableState):
    """Vertex-coordinate geometry descriptor for a variable-topology cell block."""

    cell_kind: str = eqx.field(static=True)
    conformity: str = eqx.field(static=True)
    local_dof_count: int = eqx.field(static=True)
    element_id: str = eqx.field(static=True)

    def __init__(self, cell_kind: str, local_dof_count: int, /) -> None:
        kind = str(cell_kind)
        count = int(local_dof_count)
        if kind not in ("polygon", "polyhedron"):
            raise ValueError(
                "CellVertexGeometryElement supports polygon or polyhedron blocks."
            )
        if count < (3 if kind == "polygon" else 4):
            raise ValueError("Variable-topology geometry requires enough vertices.")
        self.cell_kind = kind
        self.conformity = "H1"
        self.local_dof_count = count
        self.element_id = canonical_fingerprint(
            {
                "kind": "cell-vertex-geometry-element",
                "cell_kind": kind,
                "local_dof_count": count,
            }
        )


class CellGeometryRestrictionSource(StrictModule, NonTrainableState):
    """Scientific parent identities for exact restricted coordinate blocks.

    Ordered source corner IDs describe parent reference orientation; neither
    coefficient equality nor coordinate coincidence establishes this identity.
    ``block_source_blocks`` optionally retains, per restricted block, the one
    authored source block that owns every parent cell of that block. It is the
    explicit presentation owner of restored roots; block display names never are.
    """

    source_geometry_id: str = eqx.field(static=True)
    source_topology_id: str = eqx.field(static=True)
    block_names: tuple[str, ...] = eqx.field(static=True)
    parent_cell_ids: tuple[Array, ...]
    parent_vertex_ids: tuple[Array, ...]
    source_block_names: tuple[str, ...] | None = eqx.field(static=True)
    restriction_source_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_geometry_id: str,
        source_topology_id: str,
        block_parent_cell_ids: Mapping[str, ArrayLike],
        block_parent_vertex_ids: Mapping[str, ArrayLike],
        /,
        *,
        block_source_blocks: Mapping[str, str] | None = None,
    ) -> None:
        if not isinstance(source_geometry_id, str) or not isinstance(
            source_topology_id, str
        ):
            raise TypeError("Restriction source identities must be strings.")
        if not source_geometry_id or not source_topology_id:
            raise ValueError("Restriction source identities must be nonempty.")
        names = tuple(sorted(block_parent_cell_ids))
        if (
            not names
            or any(not name for name in names)
            or set(names) != set(block_parent_vertex_ids)
        ):
            raise ValueError(
                "Restriction source mappings must name the same nonempty blocks."
            )
        if block_source_blocks is not None and (
            set(block_source_blocks) != set(names)
            or any(
                not isinstance(value, str) or not value
                for value in block_source_blocks.values()
            )
        ):
            raise ValueError(
                "Restriction source blocks must name one authored owner per block."
            )
        cells = []
        vertices = []
        for name in names:
            cell = np.asarray(block_parent_cell_ids[name], dtype=np.int64)
            vertex = np.asarray(block_parent_vertex_ids[name], dtype=np.int64)
            if cell.ndim != 1 or vertex.ndim != 2 or vertex.shape[0] != cell.size:
                raise ValueError("Restriction parent rows must align with target cells.")
            if vertex.shape[1] < 2 or np.any(cell < 0) or np.any(vertex < -1):
                raise ValueError("Restriction parent identities are invalid.")
            for row in vertex:
                valid = row[row >= 0]
                if valid.size < 2 or np.unique(valid).size != valid.size:
                    raise ValueError(
                        "Restriction source corners must be distinct identities."
                    )
                if np.any(row[: valid.size] < 0) or np.any(row[valid.size :] != -1):
                    raise ValueError("Restriction corner padding must be trailing -1.")
            cells.append(jnp.asarray(cell))
            vertices.append(jnp.asarray(vertex))
        self.source_geometry_id = source_geometry_id
        self.source_topology_id = source_topology_id
        self.block_names = names
        self.parent_cell_ids = tuple(cells)
        self.parent_vertex_ids = tuple(vertices)
        self.source_block_names = (
            None
            if block_source_blocks is None
            else tuple(block_source_blocks[name] for name in names)
        )
        payload: dict[str, object] = {
            "kind": "cell-geometry-restriction-source",
            "source_geometry": source_geometry_id,
            "source_topology": source_topology_id,
            "blocks": [
                (name, array_tree_fingerprint(cell), array_tree_fingerprint(vertex))
                for name, cell, vertex in zip(names, cells, vertices, strict=True)
            ],
        }
        if self.source_block_names is not None:
            payload["source_blocks"] = list(self.source_block_names)
        self.restriction_source_id = canonical_fingerprint(payload)

    @property
    def block_parent_cell_ids(self) -> Mapping[str, Array]:
        return dict(zip(self.block_names, self.parent_cell_ids, strict=True))

    @property
    def block_parent_vertex_ids(self) -> Mapping[str, Array]:
        return dict(zip(self.block_names, self.parent_vertex_ids, strict=True))

    @property
    def block_source_blocks(self) -> Mapping[str, str] | None:
        if self.source_block_names is None:
            return None
        return dict(zip(self.block_names, self.source_block_names, strict=True))


class CellGeometryStorageProjection(StrictModule, NonTrainableState):
    """Actual fixed-capacity collective bank joins, lowered only on their owner.

    The source-content digest is computed from actual scientific bank values.
    Cold serial restoration validates equal content across independently restored
    array instances. Distributed restoration rebuilds the projection collectively
    before owner-local guards; those guards never initiate rank-local collectives.
    """

    source_arrays: tuple[tuple[str, Array], ...]
    projected_arrays: tuple[tuple[str, Array], ...]
    source_elements: tuple[tuple[str, CellGeometryElement], ...]
    global_cell_count: int = eqx.field(static=True)
    logical_coordinate_geometry_id: str = eqx.field(static=True)
    global_coordinate_count: int = eqx.field(static=True)
    partition_count: int = eqx.field(static=True)
    source_content_id: str = eqx.field(static=True)

    def __init__(
        self,
        logical_arrays: Sequence[tuple[str, Array]],
        logical_coordinate_geometry_id: str,
        global_coordinate_count: int,
        /,
        *,
        source_elements: Mapping[str, CellGeometryElement] | None = None,
    ) -> None:
        entries = tuple(logical_arrays)
        from ._coordinate_enclosure import coordinate_source_signature

        names = tuple(name for name, _ in entries)
        if names != tuple(sorted(names)) or len(set(names)) != len(names):
            raise ValueError(
                "Collective geometry source arrays require unique canonical names."
            )
        if any(not isinstance(value, Array) for _, value in entries):
            raise TypeError(
                "Collective geometry projection requires actual logical JAX arrays."
            )
        if (
            not isinstance(logical_coordinate_geometry_id, str)
            or len(logical_coordinate_geometry_id) != 64
            or any(
                value not in "0123456789abcdef"
                for value in logical_coordinate_geometry_id
            )
        ):
            raise ValueError(
                "Collective geometry projection requires its scientific coordinate-map digest."
            )
        arrays = dict(entries)
        queries = arrays["closure/cell_ids"]
        valid = arrays["closure/cell_valid"]
        if (
            queries.ndim != 2
            or queries.dtype != jnp.int64
            or valid.shape != queries.shape
            or valid.dtype != jnp.bool_
            or isinstance(global_coordinate_count, bool)
            or global_coordinate_count <= 0
        ):
            raise ValueError(
                "Collective geometry projection requires fixed-capacity scientific cell queries."
            )
        source = {
            name: value for name, value in arrays.items() if name.startswith("geometry/")
        }
        banks = tuple(
            sorted(
                name.removeprefix("geometry/cell_ids/")
                for name in source
                if name.startswith("geometry/cell_ids/")
            )
        )
        if not banks:
            raise ValueError(
                "Collective geometry projection requires actual scientific source banks."
            )
        coordinate_ids = source["geometry/coordinate_ids"]
        coordinates = source["geometry/coordinates"]
        owners = source["geometry/coordinate_owners"]
        if (
            coordinate_ids.ndim != 1
            or coordinate_ids.dtype != jnp.int64
            or coordinate_ids.shape[0] < global_coordinate_count
            or coordinates.ndim != 2
            or coordinates.dtype != jnp.float64
            or coordinates.shape[0] != coordinate_ids.shape[0]
            or owners.shape != coordinate_ids.shape
            or owners.dtype != jnp.int32
        ):
            raise ValueError(
                "Collective source coordinate banks have inconsistent scientific axes."
            )
        coordinate_table = jnp.where(
            jnp.arange(coordinate_ids.shape[0]) < global_coordinate_count,
            coordinate_ids,
            jnp.iinfo(jnp.int64).max,
        )
        coordinate_active = jnp.arange(coordinate_ids.shape[0]) < global_coordinate_count
        if not bool(
            jax.device_get(
                jnp.all(
                    (
                        (coordinate_ids >= 0)
                        & (owners >= 0)
                        & (owners < queries.shape[0])
                        & jnp.all(jnp.isfinite(coordinates), axis=1)
                    )
                    | ~coordinate_active
                )
                & jnp.all(
                    (coordinate_table[1:] > coordinate_table[:-1])
                    | ~coordinate_active[1:]
                )
            )
        ):
            raise ValueError(
                "Scientific source coefficients require unique ordered IDs, finite values, and valid owners."
            )
        declared = {} if source_elements is None else dict(source_elements)
        if declared and set(declared) != set(banks):
            raise ValueError(
                "Typed scientific source elements must name every logical root bank."
            )
        if any(not isinstance(value, CellGeometryElement) for value in declared.values()):
            raise TypeError(
                "Typed scientific source layouts require actual CellGeometryElement values."
            )
        source_cell_count = jnp.asarray(0, dtype=jnp.int64)
        coverage = jnp.zeros(queries.shape, dtype=jnp.int32)
        projected: dict[str, Array] = {}

        def placed(value: Array) -> Array:
            sharding = queries.sharding
            if isinstance(sharding, NamedSharding):
                sharding = NamedSharding(
                    sharding.mesh,
                    PartitionSpec(sharding.spec[0] if sharding.spec else None),
                )
            return jax.device_put(value, sharding)

        cell_owner = arrays.get("closure/cell_owner")
        owned = None
        owner_keys = None
        if cell_owner is not None:
            if cell_owner.shape != queries.shape or cell_owner.dtype != jnp.int32:
                raise ValueError(
                    "Scientific closure ownership must align with the global query layout."
                )
            if not bool(
                jax.device_get(
                    jnp.all(
                        ((cell_owner >= 0) & (cell_owner < queries.shape[0])) | ~valid
                    )
                )
            ):
                raise ValueError("Actual closure cells require valid scientific owners.")
            owned = valid & (
                cell_owner == jnp.arange(queries.shape[0], dtype=jnp.int32)[:, None]
            )
            owner_keys = jnp.sort(
                jnp.where(owned, queries, jnp.iinfo(jnp.int64).max).reshape(-1)
            )
            if owner_keys.shape[0] == 0:
                raise ValueError(
                    "A globally nonempty source requires actual owned closure cells."
                )
            projected["geometry/owned_cell_count"] = placed(
                jnp.sum(owned, axis=1, dtype=jnp.int64)
            )

        for bank in banks:
            cell_ids = source[f"geometry/cell_ids/{bank}"]
            if (
                cell_ids.ndim != 1
                or cell_ids.dtype != jnp.int64
                or cell_ids.shape[0] == 0
            ):
                raise ValueError(
                    "Scientific source cell banks require nonzero-capacity int64 identities."
                )
            source_active = cell_ids >= 0
            source_cell_count = source_cell_count + jnp.count_nonzero(source_active)
            source_routes = source[f"geometry/routes/{bank}"]
            if (
                source_routes.ndim != 2
                or source_routes.shape[0] != cell_ids.shape[0]
                or source_routes.dtype != jnp.int64
            ):
                raise ValueError(
                    "Scientific source coefficient routes must align with actual global cells."
                )
            source_positions = jnp.searchsorted(coordinate_table, source_routes)
            source_safe = jnp.minimum(source_positions, global_coordinate_count - 1)
            if not bool(
                jax.device_get(
                    jnp.all(
                        (
                            (source_positions < global_coordinate_count)
                            & (coordinate_table[source_safe] == source_routes)
                        )
                        | ~source_active[:, None]
                    )
                )
            ):
                raise ValueError(
                    "Actual global source cells reference absent scientific coordinate DOFs."
                )
            action_name = f"geometry/action_weights/{bank}"
            count_name = f"geometry/action_counts/{bank}"
            if (action_name in source) != (count_name in source):
                raise ValueError(
                    "Coefficient action stacks require their actual row counts."
                )
            if action_name in source:
                actions, counts = source[action_name], source[count_name]
                width = source_routes.shape[1]
                if (
                    actions.ndim != 4
                    or actions.shape[0] != cell_ids.shape[0]
                    or actions.shape[2:] != (width, width)
                    or actions.dtype != jnp.float64
                    or counts.shape != cell_ids.shape
                    or counts.dtype != jnp.int32
                ):
                    raise ValueError(
                        "Coefficient action stacks must bind their scientific cell and coefficient axes."
                    )
                selected = jnp.arange(actions.shape[1])[None, :] < counts[:, None]
                if not bool(
                    jax.device_get(
                        jnp.all(
                            ((counts >= 0) & (counts <= actions.shape[1]))
                            | ~source_active
                        )
                        & jnp.all(
                            jnp.all(jnp.isfinite(actions), axis=(-2, -1))
                            | ~selected
                            | ~source_active[:, None]
                        )
                    )
                ):
                    raise ValueError(
                        "Active coefficient actions require finite weights and valid stack counts."
                    )
            if bank in declared:
                element = declared[bank]
                if (
                    element.local_dof_count != source_routes.shape[1]
                    or element.conformity != "H1"
                ):
                    raise ValueError(
                        "Scientific source routes differ from their declared coordinate basis."
                    )
                signature = canonical_fingerprint(
                    {
                        "signature": coordinate_source_signature(element),
                        "arrays": array_tree_fingerprint(element),
                    }
                )
                basis = source[f"geometry/source_basis/{bank}"]
                digest = jnp.asarray(
                    np.frombuffer(bytes.fromhex(signature), dtype=np.uint8)
                )
                if basis.shape != (cell_ids.shape[0], 32) or not bool(
                    jax.device_get(
                        jnp.all(jnp.all(basis == digest, axis=1) | ~source_active)
                    )
                ):
                    raise ValueError(
                        "Global scientific coordinate basis differs from its actual source element."
                    )
            if owner_keys is not None:
                position = jnp.searchsorted(owner_keys, cell_ids)
                safe_owner = jnp.minimum(position, owner_keys.shape[0] - 1)
                successor = jnp.minimum(position + 1, owner_keys.shape[0] - 1)
                unique_owner = (
                    (position < owner_keys.shape[0])
                    & (owner_keys[safe_owner] == cell_ids)
                    & (
                        (position + 1 >= owner_keys.shape[0])
                        | (owner_keys[successor] != cell_ids)
                    )
                )
                if not bool(jax.device_get(jnp.all(unique_owner | ~source_active))):
                    raise ValueError(
                        "Actual global source cells require exactly one owned closure occurrence."
                    )
            table = jnp.where(cell_ids >= 0, cell_ids, jnp.iinfo(jnp.int64).max)
            positions = jnp.searchsorted(table, queries)
            safe = jnp.minimum(positions, cell_ids.shape[0] - 1)
            matched = (
                valid & (positions < cell_ids.shape[0]) & (cell_ids[safe] == queries)
            )
            coverage = coverage + matched.astype(jnp.int32)
            projected[f"geometry/cell_ids/{bank}"] = placed(
                jnp.where(matched, queries, -1)
            )
            for field in (
                "routes",
                "matrix",
                "offset",
                "action_weights",
                "action_counts",
                "source_basis",
                "parent_cell_ids",
                "parent_vertex_ids",
                "reference_ancestry",
            ):
                name = f"geometry/{field}/{bank}"
                if name in source:
                    projected[name] = placed(source[name][safe])
            routes = projected[f"geometry/routes/{bank}"]
            coefficient_rows = jnp.searchsorted(coordinate_table, routes)
            coefficient_safe = jnp.minimum(coefficient_rows, global_coordinate_count - 1)
            found = (coefficient_rows < global_coordinate_count) & (
                coordinate_table[coefficient_safe] == routes
            )
            if not bool(jax.device_get(jnp.all(found | ~matched[..., None]))):
                raise ValueError(
                    "Collective source routes reference absent scientific coordinate DOFs."
                )
            projected[f"geometry/coefficient_values/{bank}"] = placed(
                coordinates[coefficient_safe]
            )
            projected[f"geometry/coefficient_owners/{bank}"] = placed(
                owners[coefficient_safe]
            )
        if not bool(jax.device_get(jnp.all(coverage == valid.astype(jnp.int32)))):
            raise ValueError(
                "Every closure cell requires one unique scientific source chart."
            )
        if owned is not None and not bool(
            jax.device_get(jnp.sum(owned, dtype=jnp.int64) == source_cell_count)
        ):
            raise ValueError(
                "Owned closure occurrences must completely cover the actual global source cells."
            )
        for field in ("source_geometry_id", "source_topology_id"):
            name = f"geometry/{field}"
            if name in source:
                projected[name] = placed(
                    jnp.broadcast_to(source[name], (queries.shape[0], 32))
                )
        self.source_arrays = tuple(sorted(source.items()))
        self.projected_arrays = tuple(sorted(projected.items()))
        self.logical_coordinate_geometry_id = logical_coordinate_geometry_id
        self.global_coordinate_count = global_coordinate_count
        self.partition_count = queries.shape[0]
        self.source_content_id = logical_array_value_collection_digest(source)
        self.source_elements = tuple(sorted(declared.items()))
        self.global_cell_count = int(jax.device_get(source_cell_count))

    def require_source(
        self,
        logical_arrays: Sequence[tuple[str, Array]],
        geometry_id: str,
        /,
    ) -> None:
        """Validate scientific content after cold restore without rank-local collectives."""
        actual = {
            name: value for name, value in logical_arrays if name.startswith("geometry/")
        }
        if geometry_id != self.logical_coordinate_geometry_id or set(actual) != {
            name for name, _ in self.source_arrays
        }:
            raise ValueError(
                "Geometry projection does not belong to these accepted scientific source banks."
            )
        if all(actual[name] is value for name, value in self.source_arrays):
            return
        if jax.process_count() != 1 or any(
            not value.is_fully_addressable for value in actual.values()
        ):
            raise ValueError(
                "Cold distributed source banks require collective geometry projection reconstruction."
            )
        if logical_array_value_collection_digest(actual) != self.source_content_id:
            raise ValueError(
                "Geometry projection does not belong to these accepted scientific source banks."
            )

    def addressable_arrays(
        self, partition_index: int, /
    ) -> tuple[tuple[str, Array], ...]:
        """Return source-bound owner receipts without reading any remote shard."""
        if not 0 <= partition_index < self.partition_count:
            raise ValueError(
                "Geometry projection partition is outside its scientific placement."
            )

        def local(value: Array) -> np.ndarray:
            for shard in value.addressable_shards:
                selection = shard.index[0]
                if isinstance(selection, slice):
                    start = 0 if selection.start is None else selection.start
                    stop = value.shape[0] if selection.stop is None else selection.stop
                    if start <= partition_index < stop:
                        return np.asarray(
                            jax.device_get(shard.data[partition_index - start])
                        )
            raise ValueError("Geometry projection partition is not process-addressable.")

        packets = {name: local(value) for name, value in self.projected_arrays}
        result: dict[str, np.ndarray] = {}
        coefficient_ids = []
        coefficient_values = []
        coefficient_owners = []
        for name, values in packets.items():
            if not name.startswith("geometry/cell_ids/"):
                continue
            bank = name.removeprefix("geometry/cell_ids/")
            selected = np.flatnonzero(values >= 0)
            selected = selected[np.argsort(values[selected], kind="stable")]
            for field in (
                "cell_ids",
                "routes",
                "matrix",
                "offset",
                "action_weights",
                "action_counts",
                "source_basis",
                "parent_cell_ids",
                "parent_vertex_ids",
                "reference_ancestry",
            ):
                key = f"geometry/{field}/{bank}"
                if key in packets:
                    result[key] = packets[key][selected]
            routes = result[f"geometry/routes/{bank}"]
            coefficient_ids.append(routes.reshape(-1))
            coefficient_values.append(
                packets[f"geometry/coefficient_values/{bank}"][selected].reshape(
                    (-1, packets[f"geometry/coefficient_values/{bank}"].shape[-1])
                )
            )
            coefficient_owners.append(
                packets[f"geometry/coefficient_owners/{bank}"][selected].reshape(-1)
            )
        identifiers = np.concatenate(coefficient_ids)
        values = np.concatenate(coefficient_values)
        owners = np.concatenate(coefficient_owners)
        unique, first, inverse = np.unique(
            identifiers, return_index=True, return_inverse=True
        )
        if not np.array_equal(values, values[first][inverse]) or not np.array_equal(
            owners, owners[first][inverse]
        ):
            raise ValueError(
                "Collective coordinate receipts disagree on original coefficients or ownership."
            )
        result["geometry/coordinate_ids"] = unique
        result["geometry/coordinates"] = values[first]
        result["geometry/coordinate_owners"] = owners[first]
        for field in ("source_geometry_id", "source_topology_id"):
            name = f"geometry/{field}"
            if name in packets:
                result[name] = packets[name]
        if "geometry/owned_cell_count" in packets:
            result["geometry/owned_cell_count"] = packets["geometry/owned_cell_count"]
        return tuple((name, jnp.asarray(value)) for name, value in sorted(result.items()))


class CellGeometrySpec(StrictModule, NonTrainableState):
    """Per-block coordinate elements, geometry routes, and coordinate values."""

    block_names: tuple[str, ...] = eqx.field(static=True)
    elements: tuple[CellGeometryElement, ...]
    geometry_dofs: tuple[Array, ...]
    coordinates: Array
    geometry_layout_id: str = eqx.field(static=True)
    storage_id: str | None = eqx.field(static=True)
    logical_geometry_id: str | None = eqx.field(static=True)
    restriction_source: CellGeometryRestrictionSource | None
    exact_source: ExactCellGeometrySource | None
    periodic_source: PeriodicCellGeometrySource | None
    storage: CellMeshStorage | None

    @checked
    def __init__(
        self,
        elements: Mapping[str, CellGeometryElement],
        geometry_dofs: Mapping[str, ArrayLike],
        coordinates: ArrayLike,
        /,
        *,
        storage: CellMeshStorage | None = None,
        restriction_source: CellGeometryRestrictionSource | None = None,
        exact_source: ExactCellGeometrySource | None = None,
        periodic_source: PeriodicCellGeometrySource | None = None,
    ) -> None:
        if isinstance(coordinates, Array) and not coordinates.is_fully_addressable:
            raise ValueError(
                "Geometry coordinates require an explicit locally addressable view."
            )
        if any(
            isinstance(value, Array) and not value.is_fully_addressable
            for value in geometry_dofs.values()
        ):
            raise ValueError(
                "Geometry routes require an explicit locally addressable lowering."
            )
        items = tuple(sorted((str(name), element) for name, element in elements.items()))
        if exact_source is not None:
            _require_exact_source_layout(
                exact_source, tuple(element for _, element in items)
            )
            local_exact_power = (
                isinstance(exact_source, ExactPowerCellGeometryRestrictionSource)
                and exact_source.planes.shape[0] == 0
                and np.all(np.asarray(exact_source.vertex_plane_ids) == -1)
                and np.all(np.asarray(exact_source.vertex_parents)[:, 1] == -1)
            )
            if (storage is not None and not local_exact_power) or (
                restriction_source is not None
                and not isinstance(exact_source, ExactPlcCellGeometrySource)
                and not local_exact_power
            ):
                raise ValueError(
                    "Exact source geometry requires its direct source binding or authentic parent-bank compaction."
                )
        routes = {
            str(name): np.asarray(value, dtype=np.int32)
            for name, value in geometry_dofs.items()
        }
        points = np.asarray(coordinates, dtype=np.float64)
        if periodic_source is not None:
            if exact_source is not None or points.shape != (
                len(periodic_source.active_indices),
                periodic_source.representative_coordinates.shape[1],
            ):
                raise ValueError(
                    "A quotient source must bind its actual coordinate carrier view."
                )
        if exact_source is not None:
            exact_source.prepare(points)
        if any(not name for name, _ in items) or (
            not items and (storage is None or storage.local_blocks)
        ):
            raise ValueError(
                "Coordinate element mapping may be empty only for a bound zero-resident mesh."
            )
        if not all(isinstance(element, CellGeometryElement) for _, element in items):
            raise TypeError("Coordinate elements must satisfy CellGeometryElement.")
        if any(element.conformity != "H1" for _, element in items):
            raise ValueError("Coordinate elements must be H1-conforming.")
        if set(routes) != {name for name, _ in items}:
            raise ValueError(
                "Coordinate DOF routes must match coordinate element blocks."
            )
        if points.ndim != 2 or not np.all(np.isfinite(points)):
            raise ValueError("Coordinate values must be one finite rank-2 array.")
        if storage is not None:
            _require_storage_geometry(
                storage,
                tuple(name for name, _ in items),
                tuple(element for _, element in items),
                routes,
                points,
                restriction_source,
                exact_source,
                periodic_source,
            )
        normalized_routes = []
        for name, element in items:
            route = routes[name]
            if route.ndim != 2 or route.shape[1] != element.local_dof_count:
                raise ValueError(
                    "Coordinate DOF route width must match its coordinate element."
                )
            if np.any(route < 0) or np.any(route >= points.shape[0]):
                raise ValueError("Coordinate DOF routes index undeclared coordinates.")
            normalized_routes.append(jnp.asarray(route))
        if restriction_source is not None:
            if restriction_source.block_names != tuple(name for name, _ in items):
                raise ValueError(
                    "Restriction source identities must name every geometry block."
                )
            for route, parents in zip(
                normalized_routes, restriction_source.parent_cell_ids, strict=True
            ):
                if route.shape[0] != parents.shape[0]:
                    raise ValueError(
                        "Restriction source identities require one row per target cell."
                    )
        self.restriction_source = restriction_source
        self.exact_source = exact_source
        self.periodic_source = periodic_source
        self.storage = storage
        self.block_names = tuple(name for name, _ in items)
        self.elements = tuple(element for _, element in items)
        self.geometry_dofs = tuple(normalized_routes)
        self.coordinates = jnp.asarray(points)
        self.storage_id = None if storage is None else storage.storage_id
        self.logical_geometry_id = (
            None if storage is None else storage.logical_coordinate_geometry_id
        )
        self.geometry_layout_id = canonical_fingerprint(
            {
                "kind": "cell-geometry-spec",
                "blocks": [[name, element.element_id] for name, element in items],
                "geometry_dofs": [
                    array_tree_fingerprint(np.asarray(value))
                    for value in normalized_routes
                ],
                "coordinate_shape": list(points.shape),
                "restriction_source": (
                    None
                    if restriction_source is None
                    else restriction_source.restriction_source_id
                ),
                "exact_source": None if exact_source is None else exact_source.source_id,
                "periodic_source": None
                if periodic_source is None
                else periodic_source.source_layout_id,
            }
        )
        if storage is not None:
            self.geometry_layout_id = canonical_fingerprint(
                {
                    "kind": "owner-local-cell-geometry-layout",
                    "topology": storage.logical_topology_id,
                    "coordinate_geometry": storage.logical_coordinate_geometry_id,
                    "coordinate_count": storage.global_coordinate_count,
                    "ambient_dimension": points.shape[1],
                }
            )

    def source_coordinates(
        self, indices: ArrayLike | None = None, /
    ) -> CoordinateSourceBank:
        """Prepare current exact coefficients on the host, never in a kernel.

        This immutable preparation is not another coordinate authority. Numeric
        arrays remain the dynamic execution carrier; exact power or PLC coefficients
        are renewed only from their owning source and checked against that RNE
        carrier. A supplied one-dimensional route selects coefficient rows.
        """
        if any(
            isinstance(value, jax_core.Tracer)
            for value in jax.tree_util.tree_leaves(self)
        ):
            raise TypeError("Exact coordinate source preparation is host-only.")
        from ._coordinate_enclosure import _COORDINATE_BUDGET

        budget = _COORDINATE_BUDGET.get()
        if budget is not None:
            coefficient_bytes = (
                0
                if isinstance(
                    self.exact_source,
                    (ExactPlcCellGeometrySource, ExactPlcCellGeometryConvexSource),
                )
                else 512
                if self.exact_source is None
                else 128 + 2 * (32 + 4 * ((self.exact_source.maximum_bits + 29) // 30))
            )
            budget.reserve(
                self.coordinates.size, self.coordinates.size * coefficient_bytes
            )
        if self.periodic_source is not None:
            bank: CoordinateSourceBank = self.periodic_source.source_coordinates()
        elif self.exact_source is not None:
            bank = self.exact_source.prepare(self.coordinates).vertices
        else:
            bank = tuple(
                tuple(Fraction(float(value)) for value in row)
                for row in np.asarray(self.coordinates, dtype=np.float64)
            )
        if indices is None:
            return bank
        route = np.asarray(indices)
        if route.ndim != 1 or not np.issubdtype(route.dtype, np.integer):
            raise ValueError(
                "Exact source coefficient routes require one-dimensional integer indices."
            )
        if np.any(route < 0) or np.any(route >= len(bank)):
            raise ValueError(
                "Exact source coefficient route references an absent coefficient."
            )
        return tuple(bank[int(index)] for index in route)

    def source_execution_error(self, mesh: CellMesh, /) -> float:
        """Continuous physical bound for the RNE coefficient-image error.

        This host proof uses the actual source basis, including rational
        restrictions. It adds no epsilon tolerance to a seam equality.
        """
        if self.periodic_source is None and self.exact_source is None:
            return 0.0
        import math

        from ._coordinate_enclosure import (
            coordinate_expressions,
            expression_bernstein_coefficients,
        )

        elements, routes, runtime = self._resolve(mesh, exact_source_prepared=True)
        numeric = np.asarray(runtime, dtype=np.float64)
        source = self.source_coordinates()
        squared = Fraction(0)
        for block, element, route in zip(mesh.blocks, elements, routes, strict=True):
            domain = (
                "simplex"
                if block.cell_kind in ("interval", "triangle", "tetrahedron")
                else "prism"
                if block.cell_kind == "prism"
                else "box"
            )
            for row in np.asarray(route):
                errors = tuple(
                    tuple(
                        Fraction(float(value)) - exact
                        for value, exact in zip(
                            numeric[int(index)], source[int(index)], strict=True
                        )
                    )
                    for index in row
                    if index >= 0
                )
                if not any(value for point in errors for value in point):
                    continue
                if isinstance(element, CellVertexGeometryElement):
                    squared = max(
                        squared,
                        *(
                            sum((value * value for value in point), Fraction(0))
                            for point in errors
                        ),
                    )
                    continue
                expressions = coordinate_expressions(element, errors)
                if expressions is None:
                    raise ValueError(
                        "Execution error requires the actual coordinate source expressions."
                    )
                radii = tuple(
                    max(
                        abs(value)
                        for value in expression_bernstein_coefficients(
                            expression, domain, mesh.topological_dimension
                        )
                    )
                    for expression in expressions
                )
                squared = max(
                    squared, sum((value * value for value in radii), Fraction(0))
                )
        if not squared:
            return 0.0
        result = math.sqrt(float(squared))
        if not math.isfinite(result):
            return math.inf
        if Fraction(result) * Fraction(result) < squared:
            result = float(np.nextafter(result, np.inf))
        return result

    @classmethod
    def affine(cls, mesh: CellMesh, /) -> CellGeometrySpec:
        elements: dict[str, CellGeometryElement] = {}
        for block in mesh.blocks:
            elements[block.name] = (
                CellVertexGeometryElement(block.cell_kind, block.arity)
                if block.cell_kind in ("polygon", "polyhedron")
                else coordinate_lagrange_element(block.cell_kind, 1)
            )
        geometry = cls(
            elements,
            {block.name: block.vertices for block in mesh.blocks},
            mesh.coordinates,
            storage=mesh.storage,
        )
        return (
            geometry.with_periodic_source(mesh)
            if mesh.periodic_topology is not None
            and mesh.storage is None
            and all(
                not isinstance(element, CellVertexGeometryElement)
                for element in geometry.elements
            )
            else geometry
        )

    def with_periodic_source(self, mesh: CellMesh, /) -> CellGeometrySpec:
        """Compile the authored quotient interpretation through the FE owner."""
        if self.periodic_source is not None:
            return self
        if self.exact_source is not None:
            raise ValueError(
                "An exact coordinate source has no authored quotient interpretation."
            )
        source = PeriodicCellGeometrySource(mesh, self)
        return CellGeometrySpec(
            dict(zip(self.block_names, self.elements, strict=True)),
            dict(zip(self.block_names, self.geometry_dofs, strict=True)),
            self.coordinates,
            restriction_source=self.restriction_source,
            periodic_source=source,
            storage=self.storage,
        )

    def with_coordinates(self, coordinates: ArrayLike, /) -> CellGeometrySpec:
        """Rebind dynamic coefficients without dropping their source graph."""
        source = (
            None
            if self.periodic_source is None
            else self.periodic_source.rebound(coordinates)
        )
        return CellGeometrySpec(
            dict(zip(self.block_names, self.elements, strict=True)),
            dict(zip(self.block_names, self.geometry_dofs, strict=True)),
            coordinates,
            restriction_source=self.restriction_source,
            exact_source=self.exact_source,
            periodic_source=source,
            storage=self.storage,
        )

    @classmethod
    def power(
        cls,
        mesh: CellMesh,
        source: ExactPowerCellGeometrySource
        | ExactPowerCellGeometryRestrictionSource
        | ExactPowerCellGeometryLinearActionSource,
        /,
    ) -> CellGeometrySpec:
        """Bind an ideal power source to its independently verified RNE carrier."""
        if not isinstance(
            source,
            (
                ExactPowerCellGeometrySource,
                ExactPowerCellGeometryRestrictionSource,
                ExactPowerCellGeometryLinearActionSource,
            ),
        ):
            raise TypeError("Power geometry requires an owning exact power construction.")
        if any(
            block.cell_kind not in ("polyhedron", "hexahedron") for block in mesh.blocks
        ):
            raise ValueError(
                "Exact power geometry requires polyhedral or exact convex-parent Q1 hexahedral blocks."
            )
        return cls(
            {
                block.name: CellVertexGeometryElement("polyhedron", block.arity)
                if block.cell_kind == "polyhedron"
                else coordinate_lagrange_element("hexahedron", 1)
                for block in mesh.blocks
            },
            {block.name: block.vertices for block in mesh.blocks},
            mesh.coordinates,
            exact_source=source,
        )

    @classmethod
    def plc(
        cls,
        mesh: CellMesh,
        source: ExactPlcCellGeometrySource | ExactPlcCellGeometryConvexSource,
        /,
    ) -> CellGeometrySpec:
        """Bind an owning original PLC source or its exact convex target bank."""
        if not isinstance(
            source, (ExactPlcCellGeometrySource, ExactPlcCellGeometryConvexSource)
        ):
            raise TypeError(
                "PLC geometry requires an owning exact PLC source construction."
            )
        kinds = (
            ("tetrahedron", "hexahedron", "pyramid")
            if isinstance(source, ExactPlcCellGeometryConvexSource)
            else ("tetrahedron",)
        )
        if (
            mesh.storage is not None
            or mesh.periodic_topology is not None
            or any(block.cell_kind not in kinds for block in mesh.blocks)
        ):
            raise ValueError(
                "Exact PLC geometry requires its direct global canonical carrier."
            )
        geometry = cls(
            {
                block.name: coordinate_lagrange_element(block.cell_kind, 1)
                for block in mesh.blocks
            },
            {block.name: block.vertices for block in mesh.blocks},
            mesh.coordinates,
            exact_source=source,
        )
        geometry.resolve(mesh)
        return geometry

    def resolve(
        self,
        mesh: CellMesh,
        /,
    ) -> tuple[tuple[CellGeometryElement, ...], tuple[Array, ...], Array]:
        return self._resolve(mesh, exact_source_prepared=False)

    def _resolve(
        self,
        mesh: CellMesh,
        /,
        *,
        exact_source_prepared: bool,
    ) -> tuple[tuple[CellGeometryElement, ...], tuple[Array, ...], Array]:
        if mesh.storage is None:
            if self.storage_id is not None:
                raise ValueError(
                    "Owner-local coordinate geometry requires its bound mesh storage."
                )
        elif (
            self.storage_id != mesh.storage.storage_id
            or self.logical_geometry_id != mesh.storage.logical_coordinate_geometry_id
        ):
            raise ValueError(
                "Owner-local coordinate geometry differs from its logical storage binding."
            )
        mapping = dict(zip(self.block_names, self.elements, strict=True))
        routes = dict(zip(self.block_names, self.geometry_dofs, strict=True))
        if set(mapping) != {block.name for block in mesh.blocks}:
            raise ValueError("Coordinate element assignments must match mesh blocks.")
        resolved = tuple(mapping[block.name] for block in mesh.blocks)
        resolved_routes = tuple(routes[block.name] for block in mesh.blocks)
        for block, element, route in zip(
            mesh.blocks,
            resolved,
            resolved_routes,
            strict=True,
        ):
            if block.cell_kind != element.cell_kind:
                raise ValueError("Coordinate element cell kind does not match its block.")
            if route.shape[0] != block.cell_count:
                raise ValueError("Coordinate DOF routes require one row per cell.")
        if isinstance(self.exact_source, ExactPlcCellGeometryConvexSource):
            source = self.exact_source
            source.prepare(self.coordinates)
            target = source.target_mesh
            if not isinstance(target, CellMesh):
                raise TypeError(
                    "Exact PLC convex geometry requires its canonical target mesh."
                )
            if mesh.topology_id != target.topology_id:
                raise ValueError(
                    "Exact PLC convex geometry requires its authenticated target topology."
                )
            if mesh.storage is not None or mesh.periodic_topology is not None:
                raise ValueError(
                    "Exact PLC convex geometry requires its direct global carrier."
                )
            if not np.array_equal(
                np.asarray(mesh.coordinates).view(np.uint64),
                np.asarray(self.coordinates).view(np.uint64),
            ):
                raise ValueError(
                    "Exact source numerical carrier differs from the bound mesh."
                )
            target_blocks = target.blocks
            if len(mesh.blocks) != len(target_blocks) or not np.array_equal(
                np.asarray(mesh.vertex_global_ids),
                np.asarray(target.vertex_global_ids),
            ):
                raise ValueError(
                    "Exact PLC convex geometry differs from its original target identities."
                )
            for block, original, route in zip(
                mesh.blocks, target_blocks, resolved_routes, strict=True
            ):
                if (
                    block.name != original.name
                    or block.cell_kind != original.cell_kind
                    or not np.array_equal(
                        np.asarray(block.global_ids), np.asarray(original.global_ids)
                    )
                    or not np.array_equal(
                        np.asarray(block.vertices), np.asarray(original.vertices)
                    )
                    or not np.array_equal(np.asarray(route), np.asarray(block.vertices))
                ):
                    raise ValueError(
                        "Exact PLC convex geometry requires its original target scientific cells and vertex routes."
                    )
        elif isinstance(self.exact_source, ExactPlcCellGeometrySource):
            _require_exact_plc_carrier(self, mesh, resolved, resolved_routes)
        elif self.exact_source is not None:
            if not exact_source_prepared:
                self.exact_source.prepare(self.coordinates)
            if not np.array_equal(
                np.asarray(mesh.coordinates).view(np.uint64),
                np.asarray(self.coordinates).view(np.uint64),
            ):
                raise ValueError(
                    "Exact source numerical carrier differs from the bound mesh."
                )
        if self.periodic_source is not None:
            from ._periodic_topology import _identification_id

            origin = self.periodic_source.source_mesh
            if mesh.periodic_topology is None or origin.periodic_topology is None:
                raise ValueError(
                    "A quotient coordinate source requires its authored periodic carrier."
                )
            if _identification_id(mesh.periodic_topology.cell) != _identification_id(
                origin.periodic_topology.cell
            ):
                raise ValueError(
                    "Coordinate coefficients are not bound to this authored periodic identification."
                )
            if mesh.topology_id != origin.topology_id and (
                self.restriction_source is None
                or self.restriction_source.source_topology_id != origin.topology_id
            ):
                raise ValueError(
                    "A quotient source view requires explicit original-root restriction identity."
                )
        return (
            resolved,
            resolved_routes,
            self.coordinates
            if self.periodic_source is None
            else self.periodic_source.runtime_coordinates(),
        )


def _require_exact_source_layout(
    source: ExactCellGeometrySource,
    elements: tuple[CellGeometryElement, ...],
    /,
) -> None:
    """Admit only the coordinate layout family each exact source constructs."""
    match source:
        case (
            ExactPowerCellGeometrySource()
            | ExactPowerCellGeometryRestrictionSource()
            | ExactPowerCellGeometryLinearActionSource()
        ):
            q1_source = source
            while isinstance(q1_source, ExactPowerCellGeometryRestrictionSource):
                q1_source = q1_source.parent
            q1 = coordinate_lagrange_element("hexahedron", 1).element_id
            for element in elements:
                if (
                    isinstance(element, CellVertexGeometryElement)
                    and element.cell_kind == "polyhedron"
                ):
                    continue
                if (
                    isinstance(q1_source, ExactPowerCellGeometryLinearActionSource)
                    and q1_source.periodic_preparation is None
                    and element.element_id == q1
                ):
                    continue
                raise ValueError(
                    "Exact power sources require polyhedral layouts or authentic nonperiodic convex-parent Q1 hexahedra."
                )
        case ExactPlcCellGeometryConvexSource():
            if not elements or any(
                element.cell_kind not in ("tetrahedron", "hexahedron", "pyramid")
                or element.element_id
                != coordinate_lagrange_element(element.cell_kind, 1).element_id
                for element in elements
            ):
                raise ValueError(
                    "Exact PLC convex sources require canonical degree-one tetrahedral, Q1 hexahedral, or rational pyramid elements."
                )
        case ExactPlcCellGeometrySource():
            affine = coordinate_lagrange_element("tetrahedron", 1).element_id
            if not elements:
                raise ValueError(
                    "Exact PLC sources require affine tetrahedral coordinate elements."
                )
            for element in elements:
                if isinstance(
                    element,
                    (
                        PolynomialComposedCellGeometryElement,
                        RationalComposedCellGeometryElement,
                    ),
                ):
                    original = _require_full_p1_source(element.source_element)
                    kind = element.chart_element.cell_kind
                    expected = (
                        ("pyramid",)
                        if isinstance(element, RationalComposedCellGeometryElement)
                        else ("tetrahedron", "hexahedron")
                    )
                    if (
                        original.cell_kind != "tetrahedron"
                        or kind not in expected
                        or element.chart_element.element_id
                        != coordinate_lagrange_element(kind, 1).element_id
                    ):
                        raise ValueError(
                            "Exact PLC composed charts require original P1 tetrahedra and canonical exact target charts."
                        )
                    corners = tuple(
                        tuple(Fraction(a, b) for a, b in row)
                        for row in element.chart_coefficients
                    )
                    if any(
                        min(corner) < 0 or sum(corner, Fraction(0)) > 1
                        for corner in corners
                    ):
                        raise ValueError(
                            "Exact PLC Q1 chart corners must lie in their original source simplex."
                        )
                    continue
                if isinstance(element, BarycentricCellGeometryElement):
                    _require_full_p1_source(element)
                    if element.cell_kind != "tetrahedron":
                        raise ValueError(
                            "Exact PLC sources require affine tetrahedral coordinate elements."
                        )
                elif element.element_id != affine:
                    raise ValueError(
                        "Exact PLC sources require affine tetrahedral coordinate elements."
                    )
        case _:
            raise TypeError(
                "exact_source must be an owning exact power or PLC source construction or None."
            )


def _require_exact_plc_carrier(
    geometry: CellGeometrySpec,
    mesh: CellMesh,
    elements: tuple[CellGeometryElement, ...],
    routes: tuple[Array, ...],
    /,
) -> None:
    """Bind each carrier corner to the exact retained P1 coefficient law."""
    from ._coordinate_enclosure import (
        coordinate_corner_images,
        prepared_coordinate_source_bank,
        rounded_point,
    )

    bank = prepared_coordinate_source_bank(geometry)
    carrier = np.asarray(mesh.coordinates, dtype=np.float64)
    for block, element, route in zip(mesh.blocks, elements, routes, strict=True):
        for vertices, coefficients in zip(
            np.asarray(block.vertices), np.asarray(route), strict=True
        ):
            images = coordinate_corner_images(
                element,
                tuple(bank[int(index)] for index in coefficients),
            )
            if images is None:
                raise ValueError(
                    "Exact PLC coordinate actions require their complete exact P1 source law."
                )
            for vertex, image in zip(vertices, images, strict=True):
                if not np.array_equal(
                    carrier[vertex].view(np.uint64),
                    np.asarray(rounded_point(image), dtype=np.float64).view(np.uint64),
                ):
                    raise ValueError(
                        "Exact source numerical carrier differs from the bound mesh."
                    )


def _logical_geometry_rows(
    identifiers: Array,
    queries: ArrayLike,
    /,
) -> Array:
    """Locate only requested scientific rows without a host-global conversion."""
    query = jnp.asarray(queries, dtype=jnp.int64)
    if identifiers.ndim != 1 or identifiers.dtype != jnp.int64:
        raise ValueError("Logical scientific identities must be int64 vectors.")
    if query.shape[0] == 0:
        return jnp.empty(query.shape, dtype=jnp.int64)
    table = jnp.where(identifiers >= 0, identifiers, jnp.iinfo(jnp.int64).max)
    positions = jnp.searchsorted(table, query)
    safe = jnp.minimum(positions, max(identifiers.shape[0] - 1, 0))
    if identifiers.shape[0] == 0 or not bool(
        jnp.all((positions < identifiers.shape[0]) & (identifiers[safe] == query))
    ):
        raise ValueError(
            "Local coordinate or cell identity is absent from logical geometry."
        )
    return safe


def _require_logical_geometry_lowering(
    geometry: CellGeometrySpec,
    blocks: Sequence[CellBlock | PolyhedralBlock],
    logical_arrays: Sequence[tuple[str, Array]],
    global_coordinate_count: int,
    coordinate_ids: Array,
    coordinate_owners: Array,
    source_blocks: Mapping[str, str],
    /,
) -> None:
    """Prove the lowered source coefficients and exact ancestry against logical rows."""

    arrays = dict(logical_arrays)

    def required(name: str) -> Array:
        if name not in arrays:
            raise ValueError(f"Mapped storage lacks scientific logical array {name!r}.")
        return arrays[name]

    def equal(actual: ArrayLike, expected: ArrayLike, name: str) -> None:
        if not np.array_equal(np.asarray(actual), np.asarray(expected)):
            raise ValueError(f"Local geometry differs from logical {name}.")

    logical_ids = required("geometry/coordinate_ids")
    if (
        logical_ids.ndim != 1
        or logical_ids.shape[0] < global_coordinate_count
        or logical_ids.dtype != jnp.int64
    ):
        raise ValueError(
            "Logical coordinate identities must match the declared scientific DOF count."
        )
    rows = _logical_geometry_rows(logical_ids[:global_coordinate_count], coordinate_ids)
    logical_points = required("geometry/coordinates")
    if (
        logical_points.shape != (logical_ids.shape[0], geometry.coordinates.shape[1])
        or logical_points.dtype != jnp.float64
        or required("geometry/coordinate_owners").shape != logical_ids.shape
    ):
        raise ValueError(
            "Logical source coefficients differ from their coordinate layout."
        )
    equal(logical_points[rows], geometry.coordinates, "original source coefficients")
    equal(
        required("geometry/coordinate_owners")[rows],
        coordinate_owners,
        "coordinate ownership",
    )
    if set(geometry.block_names) != {block.name for block in blocks}:
        raise ValueError("Mapped source basis must name every local cell block.")
    elements = dict(zip(geometry.block_names, geometry.elements, strict=True))
    routes = dict(zip(geometry.block_names, geometry.geometry_dofs, strict=True))
    origin = geometry.restriction_source
    if origin is not None:
        for name, identity in (
            ("source_geometry_id", origin.source_geometry_id),
            ("source_topology_id", origin.source_topology_id),
        ):
            equal(
                required(f"geometry/{name}"),
                np.frombuffer(
                    bytes.fromhex(canonical_fingerprint(identity)), dtype=np.uint8
                ),
                name,
            )
    for block in blocks:
        name = block.name
        bank = source_blocks[name]
        cell_rows = _logical_geometry_rows(
            required(f"geometry/cell_ids/{bank}"), block.global_ids
        )
        route = routes[name]
        if route.shape[0] != block.cell_count:
            raise ValueError("Mapped source routes require one row per local cell.")
        equal(
            required(f"geometry/routes/{bank}")[cell_rows],
            coordinate_ids[route],
            "source coefficient routes",
        )
        element = elements[name]
        if element.cell_kind != block.cell_kind:
            raise ValueError(
                "Mapped source element must match its target reference kind."
            )
        root, matrix, offset = _logical_source_basis(
            element,
            required(f"geometry/source_basis/{bank}")[cell_rows],
            block.topological_dimension,
        )
        action_name = f"geometry/action_weights/{bank}"
        count_name = f"geometry/action_counts/{bank}"
        if (action_name in arrays) != (count_name in arrays):
            raise ValueError("Full coefficient action stacks require their exact counts.")
        if action_name in arrays:
            actions: list[np.ndarray] = []
            current = element
            while current is not root:
                if not isinstance(current, BarycentricCellGeometryElement):
                    raise ValueError(
                        "A Cartesian restriction cannot impersonate a full coefficient action."
                    )
                actions.append(np.asarray(current.barycentric_weights))
                current = current.source_element
            stored = required(action_name)[cell_rows]
            equal(
                required(count_name)[cell_rows],
                np.full((block.cell_count,), len(actions), dtype=np.int32),
                "authored coefficient action depth",
            )
            if stored.ndim != 4 or stored.shape[1] < len(actions):
                raise ValueError(
                    "Full coefficient actions exceed their actual stored stack."
                )
            if actions:
                expected = np.stack(actions)
                equal(
                    stored[:, : len(actions)],
                    np.broadcast_to(expected, (block.cell_count,) + expected.shape),
                    "full authored coefficient action chain",
                )
        elif not isinstance(element, CellVertexGeometryElement):
            if matrix is None or offset is None:
                raise ValueError(
                    "Full coefficient actions require their complete barycentric witnesses."
                )
            matrix_name, offset_name = (
                f"geometry/matrix/{bank}",
                f"geometry/offset/{bank}",
            )
            if matrix_name in arrays or offset_name in arrays:
                equal(
                    required(matrix_name)[cell_rows],
                    np.broadcast_to(matrix, (block.cell_count,) + matrix.shape),
                    "exact reference matrix",
                )
                equal(
                    required(offset_name)[cell_rows],
                    np.broadcast_to(offset, (block.cell_count,) + offset.shape),
                    "exact reference offset",
                )
            elif root is not element:
                raise ValueError(
                    "Restricted source charts require exact reference matrix and offset witnesses."
                )
            ancestry = _nonrepresentable_reference_ancestry(
                element, matrix, offset, source_basis=root
            )
            if ancestry is not None:
                exact_digest = np.frombuffer(bytes.fromhex(ancestry), dtype=np.uint8)
                equal(
                    required(f"geometry/reference_ancestry/{bank}")[cell_rows],
                    np.broadcast_to(exact_digest, (block.cell_count, 32)),
                    "exact composed ancestry",
                )
        if (
            isinstance(
                element, (BarycentricCellGeometryElement, RestrictedCellGeometryElement)
            )
            and origin is None
        ):
            raise ValueError(
                "Exact restricted storage requires scientific root ancestry."
            )
        if origin is not None:
            equal(
                required(f"geometry/parent_cell_ids/{bank}")[cell_rows],
                origin.block_parent_cell_ids[name],
                "scientific root cell identities",
            )
            equal(
                required(f"geometry/parent_vertex_ids/{bank}")[cell_rows],
                origin.block_parent_vertex_ids[name],
                "ordered scientific root corners",
            )


def _restore_authored_storage_geometry(
    blocks: Sequence[CellBlock | PolyhedralBlock],
    source_blocks: Mapping[str, str],
    projection: CellGeometryStorageProjection,
    partition_index: int,
    coordinate_ids: ArrayLike,
    /,
) -> CellGeometrySpec:
    """Restore an authored source chart from actual coefficient routes, not corners."""
    packets = dict(projection.addressable_arrays(partition_index))
    declared = dict(projection.source_elements)
    identifiers = np.asarray(coordinate_ids, dtype=np.int64)
    positions = np.asarray(
        _logical_geometry_rows(packets["geometry/coordinate_ids"], identifiers)
    )
    coordinates = np.asarray(packets["geometry/coordinates"])[positions]
    elements: dict[str, CellGeometryElement] = {}
    routes: dict[str, np.ndarray] = {}
    slots = {int(identifier): slot for slot, identifier in enumerate(identifiers)}
    for block in blocks:
        bank = source_blocks[block.name]
        if bank not in declared:
            raise ValueError(
                "Authored geometry requires its actual typed scientific source element."
            )
        element = declared[bank]
        if isinstance(element, RestrictedCellGeometryElement):
            raise ValueError(
                "Restricted source geometry requires its explicit ancestry specification."
            )
        rows = np.asarray(
            _logical_geometry_rows(packets[f"geometry/cell_ids/{bank}"], block.global_ids)
        )
        action_name = f"geometry/action_weights/{bank}"
        count_name = f"geometry/action_counts/{bank}"
        if (action_name in packets) != (count_name in packets):
            raise ValueError(
                "Authored coefficient stacks require their actual row counts."
            )
        if action_name in packets:
            counts = np.asarray(packets[count_name])[rows]
            if np.any(counts != 0):
                raise ValueError(
                    "Nonidentity coefficient actions require their actual barycentric geometry owner."
                )
        if f"geometry/matrix/{bank}" in packets:
            matrix = np.asarray(packets[f"geometry/matrix/{bank}"])[rows]
            offset = np.asarray(packets[f"geometry/offset/{bank}"])[rows]
            if not np.array_equal(
                matrix,
                np.broadcast_to(
                    np.eye(block.topological_dimension, dtype=np.float64), matrix.shape
                ),
            ) or np.any(offset):
                raise ValueError(
                    "Nonidentity source charts require their exact restricted geometry specification."
                )
        scientific_routes = np.asarray(packets[f"geometry/routes/{bank}"])[rows]
        if any(int(value) not in slots for value in scientific_routes.reshape(-1)):
            raise ValueError(
                "Authored source routes require their original local coordinate DOFs."
            )
        elements[block.name] = element
        routes[block.name] = np.asarray(
            [[slots[int(value)] for value in row] for row in scientific_routes],
            dtype=np.int32,
        )
    return CellGeometrySpec(elements, routes, coordinates)


def _logical_source_basis(
    element: CellGeometryElement,
    logical_basis: Array,
    dimension: int,
    /,
) -> tuple[CellGeometryElement, np.ndarray | None, np.ndarray | None]:
    """Resolve the declared scientific basis through the existing restriction chain."""
    from ._coordinate_enclosure import coordinate_source_signature

    matrix = np.eye(dimension, dtype=np.float64)
    offset = np.zeros((dimension,), dtype=np.float64)
    current = element
    basis = np.asarray(logical_basis)
    while True:
        signature = canonical_fingerprint(
            {
                "signature": coordinate_source_signature(current),
                "arrays": array_tree_fingerprint(current),
            }
        )
        digest = np.frombuffer(bytes.fromhex(signature), dtype=np.uint8)
        if np.array_equal(basis, np.broadcast_to(digest, (basis.shape[0], digest.size))):
            return current, matrix, offset
        if isinstance(current, BarycentricCellGeometryElement):
            if current is not element:
                raise ValueError(
                    "Full coefficient action lost its exact original cardinal basis."
                )
            source, _, _ = _logical_source_basis(
                current.source_element, logical_basis, dimension
            )
            return source, None, None
        if not isinstance(current, RestrictedCellGeometryElement):
            raise ValueError("Local geometry differs from logical original source basis.")
        source_matrix = np.asarray(current.matrix)
        matrix = source_matrix @ matrix
        offset = source_matrix @ offset + np.asarray(current.offset)
        current = current.source_element


def _nonrepresentable_reference_ancestry(
    element: CellGeometryElement,
    rounded_matrix: np.ndarray,
    rounded_offset: np.ndarray,
    /,
    *,
    source_basis: CellGeometryElement | None = None,
) -> str | None:
    """Bind exact composition when binary64 flattening cannot retain its source map."""
    dimension = rounded_matrix.shape[1]
    matrix = [
        [Fraction(row == column) for column in range(dimension)]
        for row in range(dimension)
    ]
    offset = [Fraction(0) for _ in range(dimension)]
    current = element
    while (
        isinstance(current, RestrictedCellGeometryElement) and current is not source_basis
    ):
        source_matrix = [
            [Fraction(float(value)) for value in row]
            for row in np.asarray(current.matrix)
        ]
        source_offset = [Fraction(float(value)) for value in np.asarray(current.offset)]
        matrix = [
            [
                sum(
                    (value * matrix[index][column] for index, value in enumerate(row)),
                    Fraction(0),
                )
                for column in range(dimension)
            ]
            for row in source_matrix
        ]
        offset = [
            sum(
                (value * offset[index] for index, value in enumerate(row)),
                source_offset[axis],
            )
            for axis, row in enumerate(source_matrix)
        ]
        current = current.source_element
    exact_matrix = tuple(
        tuple((value.numerator, value.denominator) for value in row) for row in matrix
    )
    exact_offset = tuple((value.numerator, value.denominator) for value in offset)
    if all(
        value == Fraction(float(rounded_matrix[row, column]))
        for row, values in enumerate(matrix)
        for column, value in enumerate(values)
    ) and all(
        value == Fraction(float(rounded_offset[row])) for row, value in enumerate(offset)
    ):
        return None
    return canonical_fingerprint(
        {
            "kind": "exact-composed-reference-ancestry",
            "matrix": exact_matrix,
            "offset": exact_offset,
            "source_kind": current.cell_kind,
            "target_kind": element.cell_kind,
        }
    )


def _require_storage_geometry(
    storage: CellMeshStorage,
    names: tuple[str, ...],
    elements: tuple[CellGeometryElement, ...],
    routes: Mapping[str, ArrayLike],
    points: ArrayLike,
    restriction_source: CellGeometryRestrictionSource | None,
    exact_source: ExactCellGeometrySource | None,
    periodic_source: PeriodicCellGeometrySource | None,
    /,
) -> None:
    """Reject mutated local maps independently of their cached layout identities."""
    from ._cell_geometry_validity import cell_geometry_id
    from ._coordinate_enclosure import coordinate_source_signature

    if not storage.local_blocks:
        projection = storage.geometry_projection
        if projection is None or not projection.source_elements:
            raise ValueError(
                "Zero-resident geometry requires its actual global source basis projection."
            )
        projection.require_source(
            storage.logical_arrays, storage.logical_coordinate_geometry_id
        )
        packets = dict(projection.addressable_arrays(storage.partition_index))
        values = np.asarray(points, dtype=np.float64)
        if (
            names
            or elements
            or routes
            or restriction_source is not None
            or values.shape != np.asarray(packets["geometry/coordinates"]).shape
            or values.shape != (0, storage.local_coordinates.shape[1])
            or storage.coordinate_global_ids.shape != (0,)
            or storage.coordinate_owner.shape != (0,)
            or any(value.shape != (0,) for value in storage.entity_global_ids)
            or projection.global_cell_count != storage.global_entity_counts[-1]
            or projection.global_coordinate_count != storage.global_coordinate_count
            or not np.array_equal(values, np.asarray(packets["geometry/coordinates"]))
            or "geometry/owned_cell_count" not in packets
            or int(np.asarray(packets["geometry/owned_cell_count"])) != 0
        ):
            raise ValueError(
                "Zero-resident geometry differs from its actual scientific owner receipt."
            )
        return

    witness = storage.local_geometry
    if witness is not None:
        candidate = CellGeometrySpec(
            dict(zip(names, elements, strict=True)),
            routes,
            points,
            restriction_source=restriction_source,
            exact_source=exact_source,
            periodic_source=periodic_source,
        )
        if cell_geometry_id(candidate) != storage.local_geometry_source_id:
            raise ValueError(
                "Owner-local geometry differs from its original source coefficient binding."
            )
        witness_arrays = storage.logical_arrays
        witness_count = storage.global_coordinate_count
        projection = storage.geometry_projection
        if projection is not None:
            projection.require_source(
                storage.logical_arrays, storage.logical_coordinate_geometry_id
            )
            witness_arrays = projection.addressable_arrays(storage.partition_index)
            witness_count = dict(witness_arrays)["geometry/coordinate_ids"].shape[0]
        _require_logical_geometry_lowering(
            candidate,
            storage.local_blocks,
            witness_arrays,
            witness_count,
            storage.coordinate_global_ids,
            storage.coordinate_owner,
            dict(storage.geometry_source_blocks),
        )
        return
    if (
        restriction_source is not None
        or storage.local_coordinate_id
        != canonical_fingerprint(
            array_tree_fingerprint(np.asarray(points, dtype=np.float64))
        )
    ):
        raise ValueError(
            "Owner-local geometry differs from its authoritative coordinate view."
        )
    blocks = {block.name: block for block in storage.local_blocks}
    if set(blocks) != set(names):
        raise ValueError(
            "Owner-local coordinate blocks differ from their storage witness."
        )
    for name, element in zip(names, elements, strict=True):
        block = blocks[name]
        expected = (
            CellVertexGeometryElement(block.cell_kind, block.arity)
            if block.cell_kind in ("polygon", "polyhedron")
            else coordinate_lagrange_element(block.cell_kind, 1)
        )
        if (
            coordinate_source_signature(element) != coordinate_source_signature(expected)
            or array_tree_fingerprint(element) != array_tree_fingerprint(expected)
            or not np.array_equal(np.asarray(routes[name]), np.asarray(block.vertices))
        ):
            raise ValueError(
                "Owner-local geometry differs from its authoritative coordinate routes."
            )


__all__ = [
    "CellGeometryElement",
    "CellGeometrySpec",
    "CellGeometryStorageProjection",
    "CellVertexGeometryElement",
    "RestrictedCellGeometryElement",
    "BarycentricCellGeometryElement",
    "PolynomialComposedCellGeometryElement",
    "RationalComposedCellGeometryElement",
    "SplineCellGeometryElement",
    "coordinate_lagrange_element",
    "swept_coordinate_element",
    "CellGeometryRestrictionSource",
]
