#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from fractions import Fraction
from itertools import product
from typing import assert_never, cast, final, Literal, NamedTuple, Protocol, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, core as jax_core
from jax.sharding import NamedSharding, PartitionSpec
from jax.typing import ArrayLike

import phydrax.discretization._periodic_exact as periodic_exact_owner
import phydrax.discretization._reference_cell_types as reference_cell_types_owner
import phydrax.discretization.fem._reference_types as reference_types_owner
from phydrax.discretization.fem._reference_types import FiniteElementSpec

from .. import _interpolation as interpolation_owner
from .._fingerprint import (
    array_tree_fingerprint,
    canonical_fingerprint,
    logical_array_value_collection_digest,
)
from .._host_storage import retained_host_storage_profile
from .._host_table_cache import host_table_cache, logical_table_bytes
from .._meshcore import is_native_host_buffer
from .._resource_ticket import HostBatchTicket
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import positive_integer
from ..exterior import _form_type as form_type_owner
from ..geometry._rational import fraction_endpoint
from ..linalg import _hermitian_spectral as hermitian_spectral_owner
from ..typing import checked, ConvertibleToArray, parse
from . import (
    _cell_mesh as cell_mesh_owner,
    _coordinate_enclosure as coordinate_enclosure_owner,
    _exact_algebraic_geometry as exact_algebraic_geometry_owner,
    _exact_plc_geometry as exact_plc_geometry_owner,
    _exact_power_geometry as exact_power_geometry_owner,
    _exact_rational_geometry as exact_rational_geometry_owner,
    _periodic_geometry as periodic_geometry_owner,
    _reference_cell as reference_cell_owner,
)
from ._coordinate_budget import CoordinateEnclosureBudget
from ._coordinate_enclosure import CoordinateSourceBank, Expression
from ._coordinate_source_snapshot import BarycentricSourceSnapshot, SplineSourceSnapshot
from ._reference_cell_types import CellGeometryElement


# Coordinate source kinds retain distinct scientific identities.
# Consumers dispatch nominally and never reinterpret one
# kind through another's construction.
type ExactCellGeometrySource = (
    exact_power_geometry_owner.ExactPowerCellGeometrySource
    | exact_power_geometry_owner.ExactPowerCellGeometryRestrictionSource
    | exact_power_geometry_owner.ExactPowerCellGeometryLinearActionSource
    | exact_plc_geometry_owner.ExactPlcCellGeometrySource
    | exact_plc_geometry_owner.ExactPlcCellGeometryConvexSource
    | exact_rational_geometry_owner.ExactRationalCellGeometrySource
    | exact_algebraic_geometry_owner.ExactAlgebraicCellGeometrySource
)


class ScalarCoordinateTabulation(Protocol):
    """Numerical scalar coordinate basis evaluation without source identity."""

    def tabulate(self, points: ArrayLike, /) -> tuple[Array, Array]: ...


class CoordinateTabulator(StrictModule, NonTrainableState):
    """Numerical evaluation of the coordinate source definition.

    The source is nodal interpolation on the rational equispaced lattice,
    in the polynomial space or collapsed rational pyramid space.
    Floating evaluation approximates this source; it does not define a new
    rational map from rounded barycentric weights or a rounded change of basis.
    Certificates extract the source expressions, never sampled values.
    """

    cell_kind: str = eqx.field(static=True)
    degree: int = eqx.field(static=True)
    source_basis_semantics: str = eqx.field(
        static=True, default="exact-rational-equispaced-reference-interpolation"
    )
    preparation: coordinate_enclosure_owner.PreparedCoordinateLatticeExecution | None = (
        eqx.field(static=True, default=None)
    )

    @classmethod
    @checked
    def from_stored_fields(cls, value: CoordinateTabulator, /) -> CoordinateTabulator:
        # The coordinate factory validates the lattice and constructs this
        # nominal tabulator; the FE callable field erases that constructor type.
        return cast(
            CoordinateTabulator,
            coordinate_lagrange_element(value.cell_kind, value.degree).tabulator,
        )

    def __call__(self, points: ArrayLike, /) -> tuple[Array, Array]:

        return coordinate_enclosure_owner.evaluate_coordinate_lattice_basis(
            self.cell_kind, self.degree, points, preparation=self.preparation
        )


class _CoordinateLagrangePreparation(NamedTuple):
    family: str
    nodes: np.ndarray
    exact_nodes: tuple[tuple[Fraction, ...], ...]
    entity_dofs: tuple[tuple[tuple[int, ...], ...], ...]
    tabulator: CoordinateTabulator
    tabulator_id: str


def _coordinate_lagrange_bytes(value: _CoordinateLagrangePreparation, /) -> int:
    tabulator = value.tabulator
    size = logical_table_bytes(
        (
            value.family,
            value.nodes,
            tuple(
                tuple((value.numerator, value.denominator) for value in row)
                for row in value.exact_nodes
            ),
            value.entity_dofs,
            value.tabulator_id,
            tabulator.cell_kind,
            tabulator.degree,
            tabulator.source_basis_semantics,
        )
    )
    preparation = tabulator.preparation
    if preparation is not None:
        size += logical_table_bytes((preparation.cell_kind, preparation.degree))
        polynomials = (
            *preparation.basis,
            *(p for row in preparation.gradients for p in row),
        )
        size += sum(
            logical_table_bytes(
                (exponents, coefficient.numerator, coefficient.denominator)
            )
            for polynomial in polynomials
            for exponents, raw_coefficient in polynomial
            for coefficient in (
                coordinate_enclosure_owner.require_rational_reference_scalar(
                    raw_coefficient
                ),
            )
        )
    return size


def coordinate_lagrange_element(
    cell_kind: str, degree: int, /
) -> reference_types_owner.FiniteElementSpec:
    """Bind cached immutable host lattice preparation to a fresh device element."""
    prepared = _coordinate_lagrange_preparation(cell_kind, degree)
    dimension = reference_cell_owner.reference_cell_topology(cell_kind).dimension
    return reference_types_owner.FiniteElementSpec(
        prepared.family,
        cell_kind,
        degree,
        prepared.nodes,
        prepared.entity_dofs,
        value_spec=form_type_owner.FormValueSpec(
            form_type_owner.FormType(dimension, 0, twist="untwisted"), proxy="scalar"
        ),
        tabulator=prepared.tabulator,
        tabulator_id=prepared.tabulator_id,
    )


@host_table_cache(max_entries=256, max_bytes=1048576, weight=_coordinate_lagrange_bytes)
def _coordinate_lagrange_preparation(
    cell_kind: str, degree: int, /
) -> _CoordinateLagrangePreparation:
    """Conforming equispaced coordinates in the reference space.

    Solution nodes may be warped or Lobatto nodes. Coordinate traces instead
    share one lattice across simplex, tensor, prism and rational pyramid cells.
    source expressions belong to the shared coordinate enclosure owner,
    including the rational pyramid, not a polynomial surrogate.
    """

    positive_integer(degree, "degree")
    if degree > 10:
        raise ValueError(
            "coordinate_lagrange_element.inputs must provide degrees from 1 through 10; got incompatible inputs."
        )
    topology = reference_cell_owner.reference_cell_topology(cell_kind)
    dimension = topology.dimension
    if degree == 1:
        exact_nodes = tuple(
            tuple(Fraction(float(value)) for value in row) for row in topology.vertices
        )
    elif cell_kind == "pyramid":
        exact_nodes = tuple(
            (
                Fraction(2 * i + k, 2 * degree),
                Fraction(2 * j + k, 2 * degree),
                Fraction(k, degree),
            )
            for k in range(degree + 1)
            for i in range(degree - k + 1)
            for j in range(degree - k + 1)
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
            raise ValueError(
                "coordinate_lagrange_element.inputs must satisfy the condition: Unsupported coordinate cell kind; got incompatible inputs."
            )
        exact_nodes = tuple(
            tuple(Fraction(value, degree) for value in row) for row in indices
        )
    nodes = np.asarray(
        [[float(value) for value in row] for row in exact_nodes], dtype=np.float64
    )
    preparation = None
    if cell_kind not in (
        "interval",
        "quadrilateral",
        "hexahedron",
    ) and not cell_kind.startswith("tensor:"):
        estimate = coordinate_enclosure_owner.coordinate_lattice_estimate(cell_kind, 1)
        with owned_coordinate_preparation(
            estimate.work_upper, estimate.bytes_upper
        ) as budget:
            preparation = coordinate_enclosure_owner.prepare_coordinate_lattice_basis(
                cell_kind, 1, budget=budget
            )
    weights = np.asarray(
        coordinate_enclosure_owner.evaluate_coordinate_lattice_basis(
            cell_kind, 1, nodes, preparation=preparation
        )[0],
        dtype=np.float64,
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
            raise ValueError(
                "coordinate_lagrange_element.inputs must own exactly one reference entity; got incompatible inputs."
            )
        dim, entity = matches[0]
        entity_dofs[dim][entity].append(dof)
    tabulator = CoordinateTabulator(
        cell_kind, degree, preparation=preparation if degree == 1 else None
    )
    family = (
        "HybridLagrange"
        if cell_kind in ("prism", "pyramid")
        else "TensorProductLagrange"
        if cell_kind in ("interval", "quadrilateral", "hexahedron")
        or cell_kind.startswith("tensor:")
        else "SimplexLagrange"
    )
    nodes.flags.writeable = False
    return _CoordinateLagrangePreparation(
        family,
        nodes,
        exact_nodes,
        tuple(tuple(tuple(dofs) for dofs in entities) for entities in entity_dofs),
        tabulator,
        canonical_fingerprint(
            {
                "kind": "conforming-coordinate-lattice",
                "cell_kind": cell_kind,
                "degree": degree,
                "source_basis_semantics": tabulator.source_basis_semantics,
                "nodes": array_tree_fingerprint(nodes),
            }
        ),
    )


def coordinate_lagrange_reference_nodes(
    cell_kind: str, degree: int, /
) -> tuple[tuple[Fraction, ...], ...]:
    """Return the rational lattice in the coordinate element's declared node order."""
    return _coordinate_lagrange_preparation(cell_kind, degree).exact_nodes


class SweptCoordinateTabulator(
    reference_types_owner.AbstractStoredFiniteElementTabulator
):
    """tensor lift of an authored profile basis, linear in station space."""

    source: FiniteElementSpec

    @classmethod
    @checked
    def from_stored_fields(
        cls, value: SweptCoordinateTabulator, /
    ) -> SweptCoordinateTabulator:
        return cls(value.source)

    def finite_element_from_stored_fields(
        self, /
    ) -> reference_types_owner.FiniteElementSpec:
        estimate = coordinate_enclosure_owner.source_basis_estimate(self.source)
        if estimate is None:
            raise ValueError(
                "SweptCoordinateTabulator.finite_element_from_stored_fields.inputs must provide a scalar polynomial basis; got incompatible inputs."
            )
        with owned_coordinate_preparation(
            estimate.work_upper, estimate.bytes_upper
        ) as budget:
            return swept_coordinate_element(self.source, budget=budget)

    def __call__(self, points: ArrayLike, /) -> tuple[Array, Array]:
        reference = jnp.asarray(points, dtype=jnp.float64)
        if reference.ndim != 2 or reference.shape[1] != 3:
            raise ValueError(
                "SweptCoordinateTabulator.call.inputs must provide three reference axes; got incompatible inputs."
            )
        values, gradients = self.source.tabulate(reference[:, :2])
        w = reference[:, 2:3]
        lower = jnp.concatenate(
            (gradients * (1.0 - w[..., None]), -values[..., None]), axis=-1
        )
        upper = jnp.concatenate((gradients * w[..., None], values[..., None]), axis=-1)
        return jnp.concatenate(
            (values * (1.0 - w), values * w), axis=-1
        ), jnp.concatenate((lower, upper), axis=1)


@checked
def swept_coordinate_element(
    source: reference_types_owner.FiniteElementSpec,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> reference_types_owner.FiniteElementSpec:
    """Retain the full polynomial profile source without nodal reconstruction.

    Only two endpoint banks are required, irrespective of profile order. The
    declared space is anisotropic: the source profile space times P1 in the
    station axis. Absolute frames interpolate affinely between authored stations.
    """

    if (
        source.cell_kind not in ("triangle", "quadrilateral")
        or source.conformity != "H1"
        or source.degree < 1
    ):
        raise ValueError(
            "swept_coordinate_element.inputs must provide H1 triangle or quadrilateral profiles; got incompatible inputs."
        )
    if (
        source.mapping != "identity"
        or source.value_shape
        or coordinate_enclosure_owner.source_basis(source, budget=budget) is None
    ):
        raise ValueError(
            "SweptCoordinateTabulator.finite_element_from_stored_fields.inputs must provide a scalar polynomial basis; got incompatible inputs."
        )
    kind = "prism" if source.cell_kind == "triangle" else "hexahedron"
    topology = reference_cell_owner.reference_cell_topology(kind)
    source_topology = reference_cell_owner.reference_cell_topology(source.cell_kind)
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
    tabulator = SweptCoordinateTabulator(source)
    return reference_types_owner.FiniteElementSpec(
        "SweptCoordinate",
        kind,
        source.degree,
        nodes,
        tuple(tuple(tuple(dofs) for dofs in level) for level in entity_dofs),
        value_spec=form_type_owner.FormValueSpec(
            form_type_owner.FormType(3, 0, twist="untwisted"), proxy="scalar"
        ),
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
    """One knot-span chart of an rational B-spline surface.

    Geometry coefficients are the control points, in row-major order.
    Knots and rational weights remain numerical leaves of the source;
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
                "SplineCellGeometryElement.inputs must provide a finite, nonzero rational-weight grid; got incompatible inputs."
            )
        degrees = (u_degree, v_degree)
        if len(span_indices) != 2:
            raise ValueError(
                "SplineCellGeometryElement.inputs must provide two knot-span indices; got incompatible inputs."
            )
        for axis, (vector, degree, span) in enumerate(
            zip(knots, degrees, span_indices, strict=True)
        ):
            if isinstance(degree, bool) or not isinstance(degree, (int, np.integer)):
                raise TypeError(
                    "SplineCellGeometryElement.inputs must be integers; got incompatible inputs."
                )
            if isinstance(span, bool) or not isinstance(span, (int, np.integer)):
                raise TypeError(
                    "SplineCellGeometryElement.inputs must be integer indices; got incompatible inputs."
                )
            count = weights_.shape[axis]
            if degree < 1 or degree >= count:
                raise ValueError(
                    "SplineCellGeometryElement.inputs must provide more source controls than their order; got incompatible inputs."
                )
            if vector.ndim != 1 or vector.size != count + degree + 1:
                raise ValueError(
                    "SplineCellGeometryElement.inputs must match the control grid; got incompatible inputs."
                )
            if not np.all(np.isfinite(vector)) or np.any(np.diff(vector) < 0.0):
                raise ValueError(
                    "SplineCellGeometryElement.inputs must be finite and nondecreasing; got incompatible inputs."
                )
            if not degree <= span < count or not vector[span] < vector[span + 1]:
                raise ValueError(
                    "SplineCellGeometryElement.inputs must provide a nonempty active knot span; got incompatible inputs."
                )
        if not isinstance(source_id, str) or not isinstance(source_revision, str):
            raise TypeError(
                "SplineCellGeometryElement.inputs must provide explicit source identity and revision; got incompatible inputs."
            )
        if not source_id or not source_revision:
            raise ValueError(
                "SplineCellGeometryElement.inputs must be nonempty; got incompatible inputs."
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

        reference = jnp.asarray(points, dtype=jnp.float64)
        if reference.ndim != 2 or reference.shape[1] != 2:
            raise ValueError(
                "SplineCellGeometryElement.tabulate.inputs must provide two reference coordinates; got incompatible inputs."
            )
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
                interpolation_owner.bspline_jet_stencil(
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
            rational = interpolation_owner.RationalSplineJet(
                interpolation_owner.TensorBSplineJetPlan(stencils, maximum_order=1),
                self.weights,
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
    """Endpoint profile sources plus authored corner corrections.

    ``fiber_graph`` declares a separately validated map that preserves the
    transverse affine carrier and adds one station-independent displacement
    along the final physical coordinate. General column maps leave it false.

    Scalar controls are bottom/top profile banks, bottom/top corner
    banks, then bottom/top profile-corner banks. Corrections keep the
    positive and negative physical banks separate, never a rounded difference.
    """

    wall_element: reference_types_owner.FiniteElementSpec
    corner_element: reference_types_owner.FiniteElementSpec
    cell_kind: str = eqx.field(static=True)
    conformity: str = eqx.field(static=True)
    local_dof_count: int = eqx.field(static=True)
    degree: int = eqx.field(static=True)
    topological_dimension: int = eqx.field(static=True)
    station_axis: int = eqx.field(static=True)
    element_id: str = eqx.field(static=True)
    fiber_graph: bool = eqx.field(static=True)

    @classmethod
    @checked
    def from_stored_fields(
        cls, value: LayerColumnCellGeometryElement, /
    ) -> LayerColumnCellGeometryElement:
        estimate = coordinate_enclosure_owner.source_basis_estimate(value.wall_element)
        if estimate is None:
            raise ValueError(
                "LayerColumnCellGeometryElement.from_stored_fields.inputs must provide a scalar H1 polynomial profile; got incompatible inputs."
            )
        with owned_coordinate_preparation(
            estimate.work_upper, estimate.bytes_upper
        ) as budget:
            fresh = cls(value.wall_element, fiber_graph=value.fiber_graph, budget=budget)
        for name in (
            "cell_kind",
            "conformity",
            "local_dof_count",
            "degree",
            "topological_dimension",
            "station_axis",
            "fiber_graph",
            "element_id",
        ):
            observed = object.__getattribute__(value, name)
            expected = object.__getattribute__(fresh, name)
            if type(observed) is not type(expected) or observed != expected:
                field_name = f"LayerColumnCellGeometryElement.{name}"
                raise ValueError(
                    f"{field_name} must match reconstructed profile metadata; got {observed!r}."
                )
        if type(value.corner_element) is not type(fresh.corner_element) or (
            coordinate_enclosure_owner.coordinate_source_signature(value.corner_element)
            != coordinate_enclosure_owner.coordinate_source_signature(
                fresh.corner_element
            )
        ):
            raise ValueError(
                "LayerColumnCellGeometryElement.corner_element must match its profile corner source; got a different restored basis."
            )
        return fresh

    @checked
    def __init__(
        self,
        wall_element: reference_types_owner.FiniteElementSpec,
        /,
        *,
        fiber_graph: bool = False,
        budget: CoordinateEnclosureBudget,
    ) -> None:

        if (
            wall_element.cell_kind not in ("triangle", "quadrilateral")
            or wall_element.conformity != "H1"
            or wall_element.mapping != "identity"
            or wall_element.value_shape
            or wall_element.representation != "point_value"
            or coordinate_enclosure_owner.source_basis(wall_element, budget=budget)
            is None
        ):
            raise ValueError(
                "LayerColumnCellGeometryElement.from_stored_fields.inputs must provide a scalar H1 polynomial profile; got incompatible inputs."
            )
        if not isinstance(fiber_graph, bool):
            raise TypeError(
                "LayerColumnCellGeometryElement.inputs must be a bool; got incompatible inputs."
            )
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

    @checked
    def fiber_reference_expressions(
        self,
        local: CoordinateSourceBank,
        /,
        *,
        budget: CoordinateEnclosureBudget,
    ) -> tuple[Expression, ...] | None:
        """Return the affine fiber carrier for a declared graph map."""
        if not self.fiber_graph:
            return None

        if len(local) != self.local_dof_count:
            raise ValueError(
                "LayerColumnCellGeometryElement.fiber_reference_expressions.inputs must provide its source bank; got incompatible inputs."
            )
        physical = coordinate_enclosure_owner.coordinate_expressions(
            self, local, budget=budget
        )
        profile_count = self.wall_element.local_dof_count
        corner_count = self.corner_element.local_dof_count
        affine = coordinate_enclosure_owner.coordinate_expressions(
            coordinate_lagrange_element(self.cell_kind, 1),
            local[2 * profile_count : 2 * profile_count + 2 * corner_count],
            budget=budget,
        )
        if physical is None or affine is None:
            raise ValueError(
                "LayerColumnCellGeometryElement.fiber_reference_expressions.inputs must provide physical and affine source expressions; got incompatible inputs."
            )
        displacement = coordinate_enclosure_owner.expression_sum(
            (
                physical[-1],
                coordinate_enclosure_owner.expression_scale(
                    affine[-1], -1, budget=budget
                ),
            ),
            budget=budget,
        )
        if not all(
            coordinate_enclosure_owner.expression_equal(first, second, budget=budget)
            for first, second in zip(physical[:-1], affine[:-1], strict=True)
        ) or coordinate_enclosure_owner.expression_derivative(
            displacement,
            self.station_axis,
            budget=budget,
        ):
            raise ValueError(
                "LayerColumnCellGeometryElement.fiber_graph must preserve every transverse coordinate and one station-independent fiber displacement; got a different coordinate action."
            )
        return affine

    def tabulate(self, points: ConvertibleToArray, /) -> tuple[Array, Array]:

        reference = jnp.asarray(points, dtype=jnp.float64)
        if reference.ndim != 2 or reference.shape[1] != 3:
            raise ValueError(
                "LayerColumnCellGeometryElement.tabulate.inputs must have three reference coordinates; got incompatible inputs."
            )
        station = reference[:, 2:3]

        def lift(
            element: reference_types_owner.FiniteElementSpec,
        ) -> tuple[Array, Array, Array, Array]:
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


def require_p1_cardinal_source(
    source_element: CellGeometryElement, /
) -> reference_types_owner.FiniteElementSpec:
    match source_element:
        case reference_types_owner.FiniteElementSpec():
            pass
        case _:
            raise TypeError(
                "require_p1_cardinal_source.source_element must provide the P1 source element; got another element kind."
            )
    if source_element.cell_kind not in ("triangle", "tetrahedron"):
        raise ValueError(
            "require_p1_cardinal_source.inputs must provide an simplex P1 source; got incompatible inputs."
        )
    canonical = coordinate_lagrange_element(source_element.cell_kind, 1)
    if coordinate_enclosure_owner.coordinate_source_signature(
        source_element
    ) != coordinate_enclosure_owner.coordinate_source_signature(canonical):
        raise ValueError(
            "require_p1_cardinal_source.inputs must satisfy the condition: The source does not declare the P1 cardinal vertex coefficient bank; got incompatible inputs."
        )
    return source_element


@final
class BarycentricCellGeometryElement(StrictModule, NonTrainableState):
    """Retained full coefficient actions over one affine P1 source."""

    source_element: FiniteElementSpec | BarycentricCellGeometryElement
    barycentric_weights: Array
    cell_kind: str = eqx.field(static=True)
    conformity: str = eqx.field(static=True)
    local_dof_count: int = eqx.field(static=True)
    degree: int = eqx.field(static=True)
    topological_dimension: int = eqx.field(static=True)
    element_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        source_element: FiniteElementSpec | BarycentricCellGeometryElement,
        barycentric_weights: ArrayLike,
        /,
    ) -> None:
        root = source_element
        while isinstance(root, BarycentricCellGeometryElement):
            root = root.source_element
        require_p1_cardinal_source(root)
        weights = np.asarray(barycentric_weights, dtype=np.float64)
        width = source_element.local_dof_count
        if weights.shape != (width, width) or not np.all(np.isfinite(weights)):
            raise ValueError(
                "BarycentricCellGeometryElement.inputs must bind every target and source vertex; got incompatible inputs."
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
        actions: list[Array] = []
        current: FiniteElementSpec | BarycentricCellGeometryElement = self
        while isinstance(current, BarycentricCellGeometryElement):
            actions.append(current.barycentric_weights)
            current = current.source_element
        return _apply_barycentric_actions(values, gradients, tuple(actions))


def _apply_barycentric_actions(
    values: Array, gradients: Array, actions: tuple[Array, ...], /
) -> tuple[Array, Array]:
    """Apply retained coefficient actions in outer-to-inner source order."""
    for action in actions:
        values = values @ action
        gradients = jnp.swapaxes(
            jnp.swapaxes(gradients, 1, 2) @ action,
            1,
            2,
        )
    return values, gradients


@final
class _BarycentricCoordinateTabulation(StrictModule, NonTrainableState):
    source_tabulator: Callable[[Array], tuple[ArrayLike, ArrayLike]]
    topological_dimension: int = eqx.field(static=True)
    local_dof_count: int = eqx.field(static=True)
    actions: tuple[Array, ...]

    def tabulate(self, points: ArrayLike, /) -> tuple[Array, Array]:
        locations = jnp.asarray(points)
        if locations.ndim != 2 or locations.shape[1] != self.topological_dimension:
            raise ValueError(
                "FiniteElementSpec.points must have shape (point_count, dimension); got incompatible shape"
            )
        values, gradients = self.source_tabulator(locations)
        values_, gradients_ = jnp.asarray(values), jnp.asarray(gradients)
        if values_.shape[:2] != (locations.shape[0], self.local_dof_count):
            raise ValueError(
                "FiniteElementSpec.tabulator.values must have leading axes (point_count, local_dof_count); got incompatible axes"
            )
        if gradients_.shape[:2] != values_.shape[:2]:
            raise ValueError(
                "FiniteElementSpec.tabulator.gradients must match the value leading axes; got incompatible axes"
            )
        return _apply_barycentric_actions(values_, gradients_, self.actions)


def require_full_p1_source(
    source_element: CellGeometryElement, /
) -> FiniteElementSpec | BarycentricCellGeometryElement:
    """Validate the retained action chain without replacing its owner."""
    match source_element:
        case BarycentricCellGeometryElement():
            root = source_element.source_element
            while isinstance(root, BarycentricCellGeometryElement):
                root = root.source_element
            require_p1_cardinal_source(root)
            return source_element
        case _:
            return require_p1_cardinal_source(source_element)


class RestrictedCellGeometryElement(StrictModule, NonTrainableState):
    """source basis restricted by ``source_xi = A @ target_xi + b``.

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

    @checked
    def __init__(
        self,
        source_element: reference_types_owner.FiniteElementSpec
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

        if source_element.conformity != "H1":
            raise ValueError(
                "RestrictedCellGeometryElement.inputs must be H1 conforming; got incompatible inputs."
            )
        if isinstance(source_element, reference_types_owner.FiniteElementSpec) and (
            source_element.value_shape or source_element.mapping != "identity"
        ):
            raise ValueError(
                "RestrictedCellGeometryElement.inputs must provide scalar identity-mapped coordinates; got incompatible inputs."
            )
        dimension = reference_cell_owner.reference_cell_topology(cell_kind).dimension
        matrix_ = np.asarray(matrix, dtype=np.float64)
        offset_ = np.asarray(offset, dtype=np.float64)
        if matrix_.shape != (source_element.topological_dimension, dimension):
            raise ValueError(
                "RestrictedCellGeometryElement.inputs must match both references; got incompatible inputs."
            )
        if offset_.shape != (source_element.topological_dimension,):
            raise ValueError(
                "RestrictedCellGeometryElement.inputs must match the source reference; got incompatible inputs."
            )
        if not np.all(np.isfinite(matrix_)) or not np.all(np.isfinite(offset_)):
            raise ValueError(
                "RestrictedCellGeometryElement.inputs must be finite; got incompatible inputs."
            )
        if np.linalg.matrix_rank(matrix_) != dimension:
            raise ValueError(
                "RestrictedCellGeometryElement.inputs must have full target rank; got incompatible inputs."
            )
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
            raise ValueError(
                "RestrictedCellGeometryElement.tabulate.inputs must match the target reference; got incompatible inputs."
            )
        values, gradients = self.source_element.tabulate(
            reference @ self.matrix.T + self.offset[None, :]
        )
        return values, gradients @ self.matrix


class PolynomialComposedCellGeometryElement(StrictModule, NonTrainableState):
    """coordinate source pulled back through an polynomial chart.

    Source coefficients remain in their basis. The chart coefficients
    are exact rational numbers; numerical tabulation rounds their evaluation,
    while certificates compose the source expressions exactly.
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

    @classmethod
    @checked
    def from_stored_fields(
        cls, value: PolynomialComposedCellGeometryElement, /
    ) -> PolynomialComposedCellGeometryElement:
        numerators = np.asarray(
            [[a for a, _ in row] for row in value.chart_coefficients], dtype=object
        )
        denominators = np.asarray(
            [[b for _, b in row] for row in value.chart_coefficients], dtype=object
        )
        local = tuple(
            tuple(Fraction(a, b) for a, b in row) for row in value.chart_coefficients
        )
        source_estimate = coordinate_enclosure_owner.source_basis_estimate(
            value.source_element
        )
        chart_estimate = coordinate_enclosure_owner.coordinate_polynomials_estimate(
            value.chart_element, local
        )
        if source_estimate is None or chart_estimate is None:
            raise ValueError(
                "PolynomialComposedCellGeometryElement.from_stored_fields.inputs must provide source expressions and a polynomial chart; got incompatible inputs."
            )
        with owned_coordinate_preparation(
            source_estimate.work_upper + chart_estimate.work_upper,
            source_estimate.bytes_upper + chart_estimate.bytes_upper,
        ) as budget:
            return cls(
                value.source_element,
                value.chart_element,
                numerators,
                denominators,
                budget=budget,
            )

    @checked
    def __init__(
        self,
        source_element: reference_types_owner.FiniteElementSpec
        | BarycentricCellGeometryElement
        | RestrictedCellGeometryElement
        | PolynomialComposedCellGeometryElement
        | RationalComposedCellGeometryElement
        | SplineCellGeometryElement
        | LayerColumnCellGeometryElement,
        chart_element: reference_types_owner.FiniteElementSpec,
        chart_numerators: ConvertibleToArray,
        chart_denominators: ConvertibleToArray,
        /,
        *,
        budget: CoordinateEnclosureBudget,
    ) -> None:

        source = require_scalar_coordinate_element(source_element, "Composed source")
        require_scalar_coordinate_element(chart_element, "Reference chart")
        chart = chart_element
        if source.cell_kind == "pyramid" or chart.cell_kind == "pyramid":
            raise ValueError(
                "PolynomialComposedCellGeometryElement.inputs must satisfy the condition: Polynomial composition cannot replace a collapsed rational pyramid chart; got incompatible inputs."
            )
        source_basis_, chart_basis_ = (
            coordinate_enclosure_owner.source_expressions(source, budget=budget),
            coordinate_enclosure_owner.source_basis(chart, budget=budget),
        )
        if source_basis_ is None or chart_basis_ is None:
            raise ValueError(
                "PolynomialComposedCellGeometryElement.from_stored_fields.inputs must provide source expressions and a polynomial chart; got incompatible inputs."
            )
        numerators, denominators = (
            np.asarray(chart_numerators),
            np.asarray(chart_denominators),
        )
        for bank in (numerators, denominators):
            if bank.dtype == object:
                if any(type(value) is not int for value in bank.flat):
                    raise TypeError(
                        "PolynomialComposedCellGeometryElement.inputs must contain only Python integers; got incompatible inputs."
                    )
            elif bank.dtype.kind not in "iu":
                raise TypeError(
                    "PolynomialComposedCellGeometryElement.inputs must contain integers; got incompatible inputs."
                )
        shape = (chart.local_dof_count, source.topological_dimension)
        if numerators.shape != shape or denominators.shape != shape:
            raise ValueError(
                "PolynomialComposedCellGeometryElement.inputs must match chart DOFs and source dimension; got incompatible inputs."
            )
        if np.any(denominators == 0):
            raise ValueError(
                "PolynomialComposedCellGeometryElement.inputs must provide nonzero denominators; got incompatible inputs."
            )
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
        arguments = coordinate_enclosure_owner.prepare_reference_chart_arguments(
            chart,
            coefficients,
            source.topological_dimension,
            budget=budget,
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
        # Count source-space powers per target axis. A diagonal/permuted
        # tensor chart therefore preserves its authored per-axis order, while
        # mixed-axis composition raises the bound without inspecting physical controls.
        self.degree = max(
            (
                sum(power * degree for power, degree in zip(index, degrees, strict=True))
                for degrees in chart_degrees
                for term in source_basis_
                for polynomial in coordinate_enclosure_owner.expression_parts(
                    term, source.topological_dimension, budget=budget
                )
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
            raise ValueError(
                "PolynomialComposedCellGeometryElement.tabulate.inputs must match the target reference; got incompatible inputs."
            )
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
    """P1 simplex law pulled back through a true collapsed pyramid.

    The chart is stored at the five rational pyramid nodes.
    Certificates use its collapsed polynomial representation; execution
    uses the rational pyramid basis and its continuous apex value.
    No polynomial physical-reference surrogate or fitted coefficient bank exists.
    The returned apex gradient is the centered-chart limit, not a
    Fréchet derivative when the base has a bilinear cross term.
    ``reference_derivative_status`` reports that distinction explicitly;
    derivatives with respect to source coefficients remain linear.
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

    @classmethod
    @checked
    def from_stored_fields(
        cls, value: RationalComposedCellGeometryElement, /
    ) -> RationalComposedCellGeometryElement:
        numerators = np.asarray(
            [[a for a, _ in row] for row in value.chart_coefficients], dtype=object
        )
        denominators = np.asarray(
            [[b for _, b in row] for row in value.chart_coefficients], dtype=object
        )
        return cls(value.source_element, value.chart_element, numerators, denominators)

    @checked
    def __init__(
        self,
        source_element: reference_types_owner.FiniteElementSpec
        | BarycentricCellGeometryElement,
        chart_element: reference_types_owner.FiniteElementSpec,
        chart_numerators: ConvertibleToArray,
        chart_denominators: ConvertibleToArray,
        /,
    ) -> None:

        source = require_full_p1_source(source_element)
        if source.cell_kind != "tetrahedron":
            raise ValueError(
                "RationalComposedCellGeometryElement.inputs must provide their P1 tetrahedral source; got incompatible inputs."
            )
        if (
            chart_element.element_id
            != coordinate_lagrange_element("pyramid", 1).element_id
        ):
            raise ValueError(
                "RationalComposedCellGeometryElement.inputs must provide the collapsed pyramid chart; got incompatible inputs."
            )
        numerators, denominators = (
            np.asarray(chart_numerators),
            np.asarray(chart_denominators),
        )
        for bank in (numerators, denominators):
            if bank.dtype == object:
                if any(type(value) is not int for value in bank.flat):
                    raise TypeError(
                        "RationalComposedCellGeometryElement.inputs must provide Python integer entries; got incompatible inputs."
                    )
            elif bank.dtype.kind not in "iu":
                raise TypeError(
                    "RationalComposedCellGeometryElement.inputs must provide integer entries; got incompatible inputs."
                )
        if (
            numerators.shape != (5, 3)
            or denominators.shape != (5, 3)
            or np.any(denominators == 0)
        ):
            raise ValueError(
                "RationalComposedCellGeometryElement.inputs must provide five source-reference triples; got incompatible inputs."
            )
        exact = tuple(
            tuple(Fraction(int(a), int(b)) for a, b in zip(left, right, strict=True))
            for left, right in zip(numerators, denominators, strict=True)
        )
        if any(min(row) < 0 or sum(row, Fraction(0)) > 1 for row in exact):
            raise ValueError(
                "RationalComposedCellGeometryElement.inputs must lie in the source simplex; got incompatible inputs."
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
        """Admit the reference derivative only where the source law has one."""
        reference = jnp.asarray(points, dtype=jnp.float64)
        if reference.ndim != 2 or reference.shape[1] != 3:
            raise ValueError(
                "RationalComposedCellGeometryElement.reference_derivative_status.inputs must provide pyramid reference points; got incompatible inputs."
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
                "RationalComposedCellGeometryElement.tabulate.inputs must match the pyramid reference; got incompatible inputs."
            )
        chart_values, chart_gradients = self.chart_element.tabulate(reference)
        source_points = chart_values @ self.chart_coordinates
        chart_jacobian = jnp.swapaxes(
            jnp.swapaxes(chart_gradients, -1, -2) @ self.chart_coordinates, -1, -2
        )
        values, gradients = self.source_element.tabulate(source_points)
        return values, gradients @ chart_jacobian


type ScalarCoordinateElement = (
    reference_types_owner.FiniteElementSpec
    | BarycentricCellGeometryElement
    | RestrictedCellGeometryElement
    | PolynomialComposedCellGeometryElement
    | RationalComposedCellGeometryElement
    | SplineCellGeometryElement
    | LayerColumnCellGeometryElement
)


@checked
def coordinate_tabulation(
    element: ScalarCoordinateElement, /
) -> ScalarCoordinateTabulation:
    """Project full P1 actions to dynamic arrays without their scientific IDs."""
    match element:
        case BarycentricCellGeometryElement():
            actions: list[Array] = []
            current: FiniteElementSpec | BarycentricCellGeometryElement = element
            while isinstance(current, BarycentricCellGeometryElement):
                actions.append(current.barycentric_weights)
                current = current.source_element
            tabulator = current.tabulator
            if tabulator is None:
                raise RuntimeError(
                    "coordinate_tabulation.source.tabulator must retain its admitted coordinate callback; got None"
                )
            return _BarycentricCoordinateTabulation(
                tabulator,
                current.topological_dimension,
                current.local_dof_count,
                tuple(actions),
            )
        case _:
            return element


def require_scalar_coordinate_element(
    element: CellGeometryElement, role: str, /
) -> ScalarCoordinateElement:
    """Narrow the scalar coordinate capability, never a claimed family name."""

    match element:
        case (
            reference_types_owner.FiniteElementSpec()
            | BarycentricCellGeometryElement()
            | RestrictedCellGeometryElement()
            | PolynomialComposedCellGeometryElement()
            | RationalComposedCellGeometryElement()
            | SplineCellGeometryElement()
            | LayerColumnCellGeometryElement()
        ):
            pass
        case _:
            raise TypeError(
                f"require_scalar_coordinate_element.inputs must satisfy the condition: {role} requires a tabulated scalar coordinate element.; got incompatible inputs."
            )
    if element.conformity != "H1":
        raise ValueError(
            f"require_scalar_coordinate_element.inputs must satisfy the condition: {role} requires H1 coordinate elements.; got incompatible inputs."
        )
    if isinstance(element, reference_types_owner.FiniteElementSpec) and (
        element.value_shape or element.mapping != "identity"
    ):
        raise ValueError(
            f"require_scalar_coordinate_element.inputs must satisfy the condition: {role} requires scalar identity-mapped coordinate values.; got incompatible inputs."
        )
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
                "CellVertexGeometryElement.inputs must satisfy the condition: CellVertexGeometryElement supports polygon or polyhedron blocks; got incompatible inputs."
            )
        if count < (3 if kind == "polygon" else 4):
            raise ValueError(
                "CellVertexGeometryElement.inputs must provide enough vertices; got incompatible inputs."
            )
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


CellCarrierPublication: TypeAlias = Literal["source_image", "published_affine_roots"]


class CellGeometryRestrictionSource(StrictModule, NonTrainableState):
    """Scientific parent identities for restricted coordinate blocks.

    Ordered source corner IDs describe parent reference orientation; neither
    coefficient equality nor coordinate coincidence establishes this identity.
    ``block_source_blocks`` optionally retains, per restricted block, the one
    authored source block that owns every parent cell of that block. It is the
    explicit presentation owner of restored roots; block display names never are.

    ``source_image`` publishes source-map corner images. ``published_affine_roots``
    retains a separate ``carrier_geometry`` seeded from the accepted root carrier:
    each new vertex rounds its ideal rational affine image once, while retained
    root vertices keep their input bytes. The source map remains the input to
    source certification. Non-affine maps use ``source_image``.
    """

    source_geometry_id: str = eqx.field(static=True)
    source_topology_id: str = eqx.field(static=True)
    block_names: tuple[str, ...] = eqx.field(static=True)
    parent_cell_ids: tuple[Array, ...]
    parent_vertex_ids: tuple[Array, ...]
    source_block_names: tuple[str, ...] | None = eqx.field(static=True)
    carrier_publication: CellCarrierPublication = eqx.field(static=True)
    carrier_geometry: CellGeometrySpec | None
    restriction_source_id: str = eqx.field(static=True)

    @classmethod
    @checked
    def from_stored_fields(
        cls, value: CellGeometryRestrictionSource, /
    ) -> CellGeometryRestrictionSource:
        """Rebuild parent identity banks through constructor validation."""
        return cls(
            value.source_geometry_id,
            value.source_topology_id,
            value.block_parent_cell_ids,
            value.block_parent_vertex_ids,
            block_source_blocks=value.block_source_blocks,
            carrier_publication=value.carrier_publication,
            carrier_geometry=value.carrier_geometry,
        )

    @checked
    def __init__(
        self,
        source_geometry_id: str,
        source_topology_id: str,
        block_parent_cell_ids: Mapping[str, ArrayLike],
        block_parent_vertex_ids: Mapping[str, ArrayLike],
        /,
        *,
        block_source_blocks: Mapping[str, str] | None = None,
        carrier_publication: CellCarrierPublication = "source_image",
        carrier_geometry: CellGeometrySpec | None = None,
    ) -> None:
        carrier_publication = parse(
            carrier_publication,
            CellCarrierPublication,
            "CellGeometryRestrictionSource.carrier_publication",
        )
        if (carrier_geometry is None) != (carrier_publication == "source_image"):
            raise ValueError(
                "CellGeometryRestrictionSource.carrier_geometry must accompany published_affine_roots only; got inconsistent publication law"
            )
        if not isinstance(source_geometry_id, str) or not isinstance(
            source_topology_id, str
        ):
            raise TypeError(
                "CellGeometryRestrictionSource.inputs must be strings; got incompatible inputs."
            )
        if not source_geometry_id or not source_topology_id:
            raise ValueError(
                "CellGeometryRestrictionSource.inputs must be nonempty; got incompatible inputs."
            )
        names = tuple(sorted(block_parent_cell_ids))
        if (
            not names
            or any(not name for name in names)
            or set(names) != set(block_parent_vertex_ids)
        ):
            raise ValueError(
                "CellGeometryRestrictionSource.inputs must name the same nonempty blocks; got incompatible inputs."
            )
        if block_source_blocks is not None and (
            set(block_source_blocks) != set(names)
            or any(
                not isinstance(value, str) or not value
                for value in block_source_blocks.values()
            )
        ):
            raise ValueError(
                "CellGeometryRestrictionSource.inputs must name one authored owner per block; got incompatible inputs."
            )
        cells = []
        vertices = []
        for name in names:
            cell = np.asarray(block_parent_cell_ids[name], dtype=np.int64)
            vertex = np.asarray(block_parent_vertex_ids[name], dtype=np.int64)
            if cell.ndim != 1 or vertex.ndim != 2 or vertex.shape[0] != cell.size:
                raise ValueError(
                    "CellGeometryRestrictionSource.inputs must align with target cells; got incompatible inputs."
                )
            if vertex.shape[1] < 2 or np.any(cell < 0) or np.any(vertex < -1):
                raise ValueError(
                    "CellGeometryRestrictionSource.inputs must satisfy the condition: Restriction parent identities are invalid; got incompatible inputs."
                )
            for row in vertex:
                valid = row[row >= 0]
                if valid.size < 2 or np.unique(valid).size != valid.size:
                    raise ValueError(
                        "CellGeometryRestrictionSource.inputs must be distinct identities; got incompatible inputs."
                    )
                if np.any(row[: valid.size] < 0) or np.any(row[valid.size :] != -1):
                    raise ValueError(
                        "CellGeometryRestrictionSource.inputs must be trailing -1; got incompatible inputs."
                    )
            cell_bank, vertex_bank = jax.device_put((cell, vertex))
            cells.append(cell_bank)
            vertices.append(vertex_bank)
        self.source_geometry_id = source_geometry_id
        self.source_topology_id = source_topology_id
        self.block_names = names
        self.parent_cell_ids = tuple(cells)
        self.parent_vertex_ids = tuple(vertices)
        self.carrier_publication = carrier_publication
        self.carrier_geometry = carrier_geometry
        self.source_block_names = (
            None
            if block_source_blocks is None
            else tuple(block_source_blocks[name] for name in names)
        )
        payload: dict[str, object] = {
            "kind": "cell-geometry-restriction-source",
            "source_geometry": source_geometry_id,
            "source_topology": source_topology_id,
            "carrier_publication": carrier_publication,
            "blocks": [
                (name, array_tree_fingerprint(cell), array_tree_fingerprint(vertex))
                for name, cell, vertex in zip(names, cells, vertices, strict=True)
            ],
        }
        if carrier_geometry is not None:
            from ..lifecycle._meshing_source_records import stored_record_content_id

            payload["carrier_geometry"] = stored_record_content_id(
                carrier_geometry, excluded=""
            )
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


class _ProjectionInputs(StrictModule):
    arrays: dict[str, Array]
    queries: Array
    valid: Array
    source: dict[str, Array]
    banks: tuple[str, ...]
    coordinate_table: Array
    coordinates: Array
    owners: Array
    declared: dict[str, CellGeometryElement]


def _prepare_projection_inputs(
    logical_arrays: Sequence[tuple[str, Array]],
    logical_coordinate_geometry_id: str,
    global_coordinate_count: int,
    source_elements: Mapping[str, CellGeometryElement] | None,
) -> _ProjectionInputs:
    entries = tuple(logical_arrays)

    names = tuple(name for name, _ in entries)
    if names != tuple(sorted(names)) or len(set(names)) != len(names):
        raise ValueError(
            "prepare_projection_inputs.inputs must provide unique names; got incompatible inputs."
        )
    if any(not isinstance(value, Array) for _, value in entries):
        raise TypeError(
            "prepare_projection_inputs.inputs must provide logical JAX arrays; got incompatible inputs."
        )
    if (
        not isinstance(logical_coordinate_geometry_id, str)
        or len(logical_coordinate_geometry_id) != 64
        or any(
            value not in "0123456789abcdef" for value in logical_coordinate_geometry_id
        )
    ):
        raise ValueError(
            "prepare_projection_inputs.inputs must provide its scientific coordinate-map digest; got incompatible inputs."
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
            "prepare_projection_inputs.inputs must provide fixed-capacity scientific cell queries; got incompatible inputs."
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
            "prepare_projection_inputs.inputs must provide scientific source banks; got incompatible inputs."
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
            "prepare_projection_inputs.inputs must satisfy the condition: Collective source coordinate banks have inconsistent scientific axes; got incompatible inputs."
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
                (coordinate_table[1:] > coordinate_table[:-1]) | ~coordinate_active[1:]
            )
        )
    ):
        raise ValueError(
            "prepare_projection_inputs.inputs must provide unique ordered IDs, finite values, and valid owners; got incompatible inputs."
        )
    declared = {} if source_elements is None else dict(source_elements)
    if declared and set(declared) != set(banks):
        raise ValueError(
            "prepare_projection_inputs.inputs must name every logical root bank; got incompatible inputs."
        )
    if any(not isinstance(value, CellGeometryElement) for value in declared.values()):
        raise TypeError(
            "prepare_projection_inputs.inputs must provide CellGeometryElement values; got incompatible inputs."
        )
    return _ProjectionInputs(
        arrays,
        queries,
        valid,
        source,
        banks,
        coordinate_table,
        coordinates,
        owners,
        declared,
    )


class _ProjectedBank(StrictModule):
    source_cell_count: Array
    coverage: Array


def _require_projection_actions(
    source: Mapping[str, Array],
    bank: str,
    cell_ids: Array,
    source_routes: Array,
    source_active: Array,
) -> None:
    action_name = f"geometry/action_weights/{bank}"
    count_name = f"geometry/action_counts/{bank}"
    if (action_name in source) != (count_name in source):
        raise ValueError(
            "require_projection_actions.inputs must provide their row counts; got incompatible inputs."
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
                "require_projection_actions.inputs must bind their scientific cell and coefficient axes; got incompatible inputs."
            )
        selected = jnp.arange(actions.shape[1])[None, :] < counts[:, None]
        if not bool(
            jax.device_get(
                jnp.all(((counts >= 0) & (counts <= actions.shape[1])) | ~source_active)
                & jnp.all(
                    jnp.all(jnp.isfinite(actions), axis=(-2, -1))
                    | ~selected
                    | ~source_active[:, None]
                )
            )
        ):
            raise ValueError(
                "require_projection_actions.inputs must provide finite weights and valid stack counts; got incompatible inputs."
            )


@jax.jit
def _projection_source_values(
    cell_ids: Array,
    routes: Array,
    coordinate_table: Array,
    coordinate_count: int,
) -> tuple[Array, Array, Array]:
    active = cell_ids >= 0
    positions = jnp.searchsorted(coordinate_table, routes)
    safe = jnp.minimum(positions, coordinate_count - 1)
    found = jnp.all(
        ((positions < coordinate_count) & (coordinate_table[safe] == routes))
        | ~active[:, None]
    )
    return active, jnp.count_nonzero(active), found


@jax.jit
def _projection_basis_valid(basis: Array, digest: Array, active: Array) -> Array:
    return jnp.all(jnp.all(basis == digest, axis=1) | ~active)


@jax.jit
def _projection_owners_valid(owner_keys: Array, cell_ids: Array, active: Array) -> Array:
    positions = jnp.searchsorted(owner_keys, cell_ids)
    safe = jnp.minimum(positions, owner_keys.shape[0] - 1)
    successor = jnp.minimum(positions + 1, owner_keys.shape[0] - 1)
    unique = (
        (positions < owner_keys.shape[0])
        & (owner_keys[safe] == cell_ids)
        & ((positions + 1 >= owner_keys.shape[0]) | (owner_keys[successor] != cell_ids))
    )
    return jnp.all(unique | ~active)


@jax.jit
def _projection_bank_values(
    cell_ids: Array,
    source_routes: Array,
    queries: Array,
    valid: Array,
    coordinate_table: Array,
    coordinate_count: int,
    coordinates: Array,
    owners: Array,
    fields: tuple[Array, ...],
) -> tuple[Array, Array, tuple[Array, ...], Array, Array, Array]:
    table = jnp.where(cell_ids >= 0, cell_ids, jnp.iinfo(jnp.int64).max)
    positions = jnp.searchsorted(table, queries)
    safe = jnp.minimum(positions, cell_ids.shape[0] - 1)
    matched = valid & (positions < cell_ids.shape[0]) & (cell_ids[safe] == queries)
    routes = source_routes[safe]
    coefficient_rows = jnp.searchsorted(coordinate_table, routes)
    coefficient_safe = jnp.minimum(coefficient_rows, coordinate_count - 1)
    found = jnp.all(
        (
            (coefficient_rows < coordinate_count)
            & (coordinate_table[coefficient_safe] == routes)
        )
        | ~matched[..., None]
    )
    return (
        jnp.where(matched, queries, -1),
        matched.astype(jnp.int32),
        tuple(field[safe] for field in fields),
        found,
        coordinates[coefficient_safe],
        owners[coefficient_safe],
    )


def _project_geometry_bank(
    prepared: _ProjectionInputs,
    bank: str,
    global_coordinate_count: int,
    owner_keys: Array | None,
    projected: dict[str, Array],
    placed: Callable[[Array], Array],
) -> _ProjectedBank:
    source, queries, valid = prepared.source, prepared.queries, prepared.valid
    coordinate_table, declared = prepared.coordinate_table, prepared.declared
    coordinates, owners = prepared.coordinates, prepared.owners
    cell_ids = source[f"geometry/cell_ids/{bank}"]
    if cell_ids.ndim != 1 or cell_ids.dtype != jnp.int64 or cell_ids.shape[0] == 0:
        raise ValueError(
            "project_geometry_bank.inputs must provide nonzero-capacity int64 identities; got incompatible inputs."
        )
    source_routes = source[f"geometry/routes/{bank}"]
    if (
        source_routes.ndim != 2
        or source_routes.shape[0] != cell_ids.shape[0]
        or source_routes.dtype != jnp.int64
    ):
        raise ValueError(
            "project_geometry_bank.inputs must align with global cells; got incompatible inputs."
        )
    source_active, source_cell_count, source_found = _projection_source_values(
        cell_ids, source_routes, coordinate_table, global_coordinate_count
    )
    if not bool(jax.device_get(source_found)):
        raise ValueError(
            "project_geometry_bank.inputs must satisfy the condition: global source cells reference absent scientific coordinate DOFs; got incompatible inputs."
        )
    _require_projection_actions(source, bank, cell_ids, source_routes, source_active)
    if bank in declared:
        element = declared[bank]
        if (
            element.local_dof_count != source_routes.shape[1]
            or element.conformity != "H1"
        ):
            raise ValueError(
                "project_geometry_bank.inputs must match their declared coordinate basis; got incompatible inputs."
            )
        signature = canonical_fingerprint(
            {
                "signature": coordinate_enclosure_owner.coordinate_source_signature(
                    element
                ),
                "arrays": array_tree_fingerprint(element),
            }
        )
        basis = source[f"geometry/source_basis/{bank}"]
        digest = jnp.asarray(np.frombuffer(bytes.fromhex(signature), dtype=np.uint8))
        if basis.shape != (cell_ids.shape[0], 32) or not bool(
            jax.device_get(_projection_basis_valid(basis, digest, source_active))
        ):
            raise ValueError(
                "project_geometry_bank.inputs must match its source element; got incompatible inputs."
            )
    if owner_keys is not None:
        if not bool(
            jax.device_get(_projection_owners_valid(owner_keys, cell_ids, source_active))
        ):
            raise ValueError(
                "project_geometry_bank.inputs must provide exactly one owned closure occurrence; got incompatible inputs."
            )
    field_names = tuple(
        f"geometry/{field}/{bank}"
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
        )
        if f"geometry/{field}/{bank}" in source
    )
    cell_values, coverage, fields, found, coefficient_values, coefficient_owners = (
        _projection_bank_values(
            cell_ids,
            source_routes,
            queries,
            valid,
            coordinate_table,
            global_coordinate_count,
            coordinates,
            owners,
            tuple(source[name] for name in field_names),
        )
    )
    projected[f"geometry/cell_ids/{bank}"] = placed(cell_values)
    for name, value in zip(field_names, fields, strict=True):
        projected[name] = placed(value)
    if not bool(jax.device_get(found)):
        raise ValueError(
            "project_geometry_bank.inputs must satisfy the condition: Collective source routes reference absent scientific coordinate DOFs; got incompatible inputs."
        )
    projected[f"geometry/coefficient_values/{bank}"] = placed(coefficient_values)
    projected[f"geometry/coefficient_owners/{bank}"] = placed(coefficient_owners)
    return _ProjectedBank(source_cell_count, coverage)


class CellGeometryStorageProjection(StrictModule, NonTrainableState):
    """fixed-capacity collective bank joins, lowered only on their owner.

    The source-content digest is computed from scientific bank values.
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
        prepared = _prepare_projection_inputs(
            logical_arrays,
            logical_coordinate_geometry_id,
            global_coordinate_count,
            source_elements,
        )
        arrays, queries, valid = prepared.arrays, prepared.queries, prepared.valid
        source, banks = prepared.source, prepared.banks
        declared = prepared.declared
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
                    "CellGeometryStorageProjection.inputs must align with the global query layout; got incompatible inputs."
                )
            if not bool(
                jax.device_get(
                    jnp.all(
                        ((cell_owner >= 0) & (cell_owner < queries.shape[0])) | ~valid
                    )
                )
            ):
                raise ValueError(
                    "CellGeometryStorageProjection.inputs must provide valid scientific owners; got incompatible inputs."
                )
            owned = valid & (
                cell_owner == jnp.arange(queries.shape[0], dtype=jnp.int32)[:, None]
            )
            owner_keys = jnp.sort(
                jnp.where(owned, queries, jnp.iinfo(jnp.int64).max).reshape(-1)
            )
            if owner_keys.shape[0] == 0:
                raise ValueError(
                    "CellGeometryStorageProjection.inputs must provide owned closure cells; got incompatible inputs."
                )
            projected["geometry/owned_cell_count"] = placed(
                jnp.sum(owned, axis=1, dtype=jnp.int64)
            )

        for bank in banks:
            projected_bank = _project_geometry_bank(
                prepared,
                bank,
                global_coordinate_count,
                owner_keys,
                projected,
                placed,
            )
            source_cell_count = source_cell_count + projected_bank.source_cell_count
            coverage = coverage + projected_bank.coverage
        if not bool(jax.device_get(jnp.all(coverage == valid.astype(jnp.int32)))):
            raise ValueError(
                "CellGeometryStorageProjection.inputs must provide one unique scientific source chart; got incompatible inputs."
            )
        if owned is not None and not bool(
            jax.device_get(jnp.sum(owned, dtype=jnp.int64) == source_cell_count)
        ):
            raise ValueError(
                "CellGeometryStorageProjection.inputs must completely cover the global source cells; got incompatible inputs."
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
                f"CellGeometryStorageProjection.source_arrays must match the bound logical bank names and geometry ID; got names={tuple(sorted(actual))}, geometry_id={geometry_id!r}."
            )
        if all(actual[name] is value for name, value in self.source_arrays):
            return
        if jax.process_count() != 1 or any(
            not value.is_fully_addressable for value in actual.values()
        ):
            raise ValueError(
                "CellGeometryStorageProjection.require_source.inputs must provide collective geometry projection reconstruction; got incompatible inputs."
            )
        if logical_array_value_collection_digest(actual) != self.source_content_id:
            raise ValueError(
                "CellGeometryStorageProjection.source_content_id must match the scientific bank digest; got different coefficient values."
            )

    def addressable_arrays(
        self, partition_index: int, /
    ) -> tuple[tuple[str, Array], ...]:
        """Return source-bound owner receipts without reading any remote shard."""
        if not 0 <= partition_index < self.partition_count:
            raise ValueError(
                "CellGeometryStorageProjection.addressable_arrays.inputs must satisfy the condition: Geometry projection partition is outside its scientific placement; got incompatible inputs."
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
            raise ValueError(
                "CellGeometryStorageProjection.addressable_arrays.local.inputs must satisfy the condition: Geometry projection partition is not process-addressable; got incompatible inputs."
            )

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
                "CellGeometryStorageProjection.addressable_arrays.inputs must satisfy the condition: Collective coordinate receipts disagree on coefficients or ownership; got incompatible inputs."
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


@contextmanager
def owned_coordinate_preparation(
    maximum_work_units: int,
    maximum_memory_bytes: int,
    /,
    *,
    maximum_geometry_queries: int = 0,
) -> Iterator[CoordinateEnclosureBudget]:
    """Own one host coefficient reservation with explicit domain maxima."""
    ticket = HostBatchTicket(
        maximum_work_units,
        queries_upper=maximum_geometry_queries,
        bytes_upper=maximum_memory_bytes,
    )
    budget = CoordinateEnclosureBudget(
        ticket,
        maximum_work_units=maximum_work_units,
        maximum_memory_bytes=maximum_memory_bytes,
    )
    with ticket, budget.activate():
        yield budget


@checked
def exact_source_preparation_bounds(
    source: ExactCellGeometrySource,
    coordinate_count: int,
    dimension: int,
    /,
) -> tuple[int, int]:
    """Bound one source proof, its linear actions, and retained-bank metadata."""
    if isinstance(
        source,
        (
            exact_rational_geometry_owner.ExactRationalCellGeometrySource,
            exact_algebraic_geometry_owner.ExactAlgebraicCellGeometrySource,
        ),
    ):
        return source.preparation_bounds()
    bits = max(source.maximum_bits, 1075)
    coefficient_bytes = 512 + 2 * ((bits + 7) // 8)
    queries = source.preparation_query_upper()
    rows = source.maximum_work + coordinate_count + queries
    # Source solvers charge 4*n**3 scientific visits, n<=4 and one
    # action column. The solver's elimination plus two bank scans
    # is at most twice that; scientific source limits remain enforced by
    # _ExactPlcBudget/_ExactPowerBudget independently of metadata visits.
    arithmetic = 3 * source.maximum_work
    # One bank tuple, point tuples, booleans, coefficient Fractions and their
    # two integer children, plus the fixed key/prepared-record shells.
    metadata = 8 + 2 * rows + 3 * rows * dimension

    # PLC preparation admits an eight-times-input-bit arithmetic
    # window before constructing support proofs. Convex proofs also retain the
    # parent bank and their support/corner-reference bank concurrently.
    plc_storage = 0
    match source:
        case exact_plc_geometry_owner.ExactPlcCellGeometrySource():
            plc_storage = sum(
                exact_plc_geometry_owner.plc_storage_bounds(
                    source.vertex_strata.shape[0], bits
                )
            )
        case exact_plc_geometry_owner.ExactPlcCellGeometryConvexSource():
            _, parent_geometry, _ = source.owners()
            plc_storage = sum(
                exact_plc_geometry_owner.plc_storage_bounds(
                    parent_geometry.coordinates.shape[0], bits
                )
            )
            plc_storage += sum(
                exact_plc_geometry_owner.plc_storage_bounds(
                    source.support_vertices.shape[0]
                    + source.cell_parent_ids.shape[0] * 8,
                    bits,
                )
            )
        case (
            exact_power_geometry_owner.ExactPowerCellGeometrySource()
            | exact_power_geometry_owner.ExactPowerCellGeometryRestrictionSource()
            | exact_power_geometry_owner.ExactPowerCellGeometryLinearActionSource()
        ):
            pass
        case invalid:
            assert_never(invalid)
    # A source action has n<=4, one RHS; a PLC inverse has n=3, three RHS.
    # Both fit 20 authored entries and 176 live arithmetic entries. A
    # difference of two input Fractions needs at most twice the declared bits.
    terms = 4 * (4 + 1)
    height = 2 * bits * (terms + 1) + 1
    linear_storage = hermitian_spectral_owner.fraction_storage_upper(
        4, 2 * terms * bits + 1
    )
    linear_storage += hermitian_spectral_owner.fraction_storage_upper(
        8 * 4 * (4 + 1) + 16, 16 * 4 * height + 64
    )
    memory = (rows * dimension + 8) * coefficient_bytes + plc_storage + linear_storage
    return arithmetic + metadata, memory


def exact_source_field_degree(source: ExactCellGeometrySource | None, /) -> int:
    """Return the generated multiquadratic coefficient-field degree."""
    match source:
        case (
            exact_algebraic_geometry_owner.ExactAlgebraicCellGeometrySource()
            | exact_rational_geometry_owner.ExactRationalCellGeometrySource()
        ):
            return source.field_degree
        case _:
            return 1


def coordinate_source_snapshot_preparation_bounds(
    element: CellGeometryElement, /
) -> tuple[int, int]:
    """Bound host snapshot admission and retained source-owner coexistence."""
    work, memory = 0, 0
    pending = [element]
    while pending:
        current = pending.pop()
        match current:
            case BarycentricCellGeometryElement():
                arrays = (current.barycentric_weights,)
                pending.append(current.source_element)
            case SplineCellGeometryElement():
                arrays = (current.u_knots, current.v_knots, current.weights)
            case RestrictedCellGeometryElement():
                pending.append(current.source_element)
                continue
            case (
                PolynomialComposedCellGeometryElement()
                | RationalComposedCellGeometryElement()
            ):
                pending.extend((current.source_element, current.chart_element))
                continue
            case LayerColumnCellGeometryElement():
                pending.append(current.wall_element)
                continue
            case _:
                continue
        _, storage, logical, visits = retained_host_storage_profile(
            current, {}, is_native_host_buffer
        )
        work += visits + 128
        memory += storage + logical + 3 * sum(value.nbytes for value in arrays) + 16384
    return work, memory


def _coordinate_snapshot_identity(
    element: CellGeometryElement, budget: CoordinateEnclosureBudget
) -> str:
    identifier = id(element)
    if identifier not in budget.source_snapshot_identity_cache:
        budget.retain_owner(element)
        identity = canonical_fingerprint(array_tree_fingerprint(element))
        budget.source_snapshot_identity_cache[identifier] = identity
        return identity
    return budget.source_snapshot_identity_cache[identifier]


@checked
def prepare_barycentric_source_snapshot(
    element: BarycentricCellGeometryElement,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> BarycentricSourceSnapshot:
    """Admit and retain one host coefficient action per source owner and budget."""
    key = _coordinate_snapshot_identity(element, budget)
    if key not in budget.barycentric_source_snapshot_cache:
        with budget.temporary_scope():
            # The device-get view and one host-owned copy coexist. Retained
            # snapshots must not keep a backend capsule as their storage owner.
            budget.reserve(0, 2 * element.barycentric_weights.nbytes + 4096)
            weights = np.array(
                jax.device_get(element.barycentric_weights), dtype=np.float64, copy=True
            )
            weights.setflags(write=False)
            snapshot = BarycentricSourceSnapshot(weights)
            budget.retain_owner(snapshot)
            budget.barycentric_source_snapshot_cache[key] = snapshot
    return budget.barycentric_source_snapshot_cache[key]


@checked
def prepare_spline_source_snapshot(
    element: SplineCellGeometryElement,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> SplineSourceSnapshot:
    """Admit one host knot/weight bundle without changing dynamic execution leaves."""
    key = _coordinate_snapshot_identity(element, budget)
    if key not in budget.spline_source_snapshot_cache:
        sources = (element.u_knots, element.v_knots, element.weights)
        with budget.temporary_scope():
            budget.reserve(0, 2 * sum(value.nbytes for value in sources) + 4096)
            u, v, weights = (
                np.array(value, dtype=np.float64, copy=True)
                for value in jax.device_get(sources)
            )
            for value in (u, v, weights):
                value.setflags(write=False)
            snapshot = SplineSourceSnapshot(u, v, weights)
            budget.retain_owner(snapshot)
            budget.spline_source_snapshot_cache[key] = snapshot
    return budget.spline_source_snapshot_cache[key]


@checked
def cell_geometry_preparation_bounds(
    geometry: CellGeometrySpec,
    /,
) -> tuple[int, int]:
    """Bound banks and cell-map/corner preparation from their source recipes."""
    host_coordinates, host_routes = jax.device_get(
        (geometry.coordinates, geometry.geometry_dofs)
    )
    values = np.asarray(host_coordinates, dtype=np.float64)
    work = geometry.coordinates.size
    memory = geometry.coordinates.size * 512
    source = geometry.exact_source
    coefficient_bits = None if source is None else max(source.maximum_bits, 1075)
    if geometry.periodic_source is not None:
        periodic_work, periodic_memory, periodic_bits = (
            geometry.periodic_source.preparation_bounds()
        )
        work += periodic_work
        memory += periodic_memory
        coefficient_bits = periodic_bits
    if source is not None:
        source_work, source_memory = exact_source_preparation_bounds(
            source, values.shape[0], values.shape[1]
        )
        work += source_work
        memory += source_memory
    source_work_upper, source_memory_upper = work, memory
    estimates: dict[
        tuple[str, tuple[int, ...], int],
        coordinate_enclosure_owner.CoordinateEnclosureEstimate,
    ] = {}
    snapshot_sources: set[str] = set()
    for element, routes in zip(geometry.elements, host_routes, strict=True):
        if isinstance(element, CellVertexGeometryElement):
            continue
        signature = canonical_fingerprint(
            coordinate_enclosure_owner.coordinate_source_signature(element)
        )
        if signature not in snapshot_sources:
            snapshot_work, snapshot_memory = (
                coordinate_source_snapshot_preparation_bounds(element)
            )
            work += snapshot_work
            memory += snapshot_memory
            snapshot_sources.add(signature)
        for route in np.asarray(routes):
            local = values[route]
            # Binary64 carrier coefficients share the 1075-bit bound. Rational
            # banks retain the source's declared profile instead.
            key = (
                signature,
                local.shape,
                1075 if coefficient_bits is None else coefficient_bits,
            )
            if key not in estimates:
                estimate = coordinate_enclosure_owner.coordinate_polynomials_estimate(
                    element, local, coefficient_bits_upper=coefficient_bits
                )
                if estimate is None:
                    estimate = coordinate_enclosure_owner.coordinate_preparation_metadata_estimate(
                        element, local, coefficient_bits_upper=coefficient_bits
                    )
                estimates[key] = estimate
            estimate = estimates[key]
            work += estimate.work_upper
            memory += estimate.bytes_upper
    algebraic_work, algebraic_memory = (
        coordinate_enclosure_owner.coordinate_algebraic_preparation_bounds(
            work - source_work_upper,
            memory - source_memory_upper,
            field_degree_upper=exact_source_field_degree(source),
            polynomial_degree_upper=max(
                (
                    scalar.degree * scalar.topological_dimension
                    for element in geometry.elements
                    if not isinstance(element, CellVertexGeometryElement)
                    for scalar in (
                        require_scalar_coordinate_element(
                            element, "coordinate preparation"
                        ),
                    )
                ),
                default=1,
            ),
        )
    )
    return source_work_upper + algebraic_work, source_memory_upper + algebraic_memory


@checked
def cell_geometry_preparation_query_upper(geometry: CellGeometrySpec, /) -> int:
    """Bound one source-coordinate preparation; caller repetitions remain explicit."""
    if geometry.periodic_source is not None:
        return 0
    source = geometry.exact_source
    return 0 if source is None else source.preparation_query_upper()


def _require_geometry_source_binding(
    exact_source: ExactCellGeometrySource | None,
    items: tuple[tuple[str, CellGeometryElement], ...],
    storage: cell_mesh_owner.CellMeshStorage | None,
    restriction_source: CellGeometryRestrictionSource | None,
) -> None:
    if exact_source is not None:
        _require_exact_source_layout(exact_source, tuple(element for _, element in items))
        local_exact_power = (
            isinstance(
                exact_source,
                exact_power_geometry_owner.ExactPowerCellGeometryRestrictionSource,
            )
            and exact_source.planes.shape[0] == 0
            and np.all(np.asarray(exact_source.vertex_plane_ids) == -1)
            and np.all(np.asarray(exact_source.vertex_parents)[:, 1] == -1)
        )
        match exact_source:
            case (
                exact_plc_geometry_owner.ExactPlcCellGeometrySource()
                | exact_rational_geometry_owner.ExactRationalCellGeometrySource()
                | exact_algebraic_geometry_owner.ExactAlgebraicCellGeometrySource()
            ):
                local_exact_owner = True
            case _:
                local_exact_owner = local_exact_power
        if (storage is not None and not local_exact_owner) or (
            restriction_source is not None
            and not isinstance(
                exact_source,
                (
                    exact_plc_geometry_owner.ExactPlcCellGeometrySource,
                    exact_rational_geometry_owner.ExactRationalCellGeometrySource,
                    exact_algebraic_geometry_owner.ExactAlgebraicCellGeometrySource,
                ),
            )
            and not local_exact_power
        ):
            raise ValueError(
                "require_geometry_source_binding.inputs must provide its direct source binding or parent-bank compaction; got incompatible inputs."
            )


class _GeometryRoutes(StrictModule):
    routes: tuple[Array, ...]


def _prepare_geometry_routes(
    items: tuple[tuple[str, CellGeometryElement], ...],
    routes: Mapping[str, np.ndarray],
    points: np.ndarray,
    storage: cell_mesh_owner.CellMeshStorage | None,
    restriction_source: CellGeometryRestrictionSource | None,
    exact_source: ExactCellGeometrySource | None,
    periodic_source: periodic_geometry_owner.PeriodicCellGeometrySource | None,
) -> _GeometryRoutes:
    if any(not name for name, _ in items) or (
        not items and (storage is None or storage.local_blocks)
    ):
        raise ValueError(
            "prepare_geometry_routes.inputs must satisfy the condition: Coordinate element mapping may be empty only for a bound zero-resident mesh; got incompatible inputs."
        )
    if not all(isinstance(element, CellGeometryElement) for _, element in items):
        raise TypeError(
            "prepare_geometry_routes.inputs must satisfy CellGeometryElement; got incompatible inputs."
        )
    if any(element.conformity != "H1" for _, element in items):
        raise ValueError(
            "prepare_geometry_routes.inputs must be H1-conforming; got incompatible inputs."
        )
    if set(routes) != {name for name, _ in items}:
        raise ValueError(
            "prepare_geometry_routes.inputs must match coordinate element blocks; got incompatible inputs."
        )
    if points.ndim != 2 or not np.all(np.isfinite(points)):
        raise ValueError(
            "prepare_geometry_routes.inputs must be one finite rank-2 array; got incompatible inputs."
        )
    if storage is not None:
        require_storage_geometry(
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
                "prepare_geometry_routes.inputs must match its coordinate element; got incompatible inputs."
            )
        if np.any(route < 0) or np.any(route >= points.shape[0]):
            raise ValueError(
                "prepare_geometry_routes.inputs must satisfy the condition: Coordinate DOF routes index undeclared coordinates; got incompatible inputs."
            )
        normalized_routes.append(jax.device_put(route))
    if restriction_source is not None:
        if restriction_source.block_names != tuple(name for name, _ in items):
            raise ValueError(
                "prepare_geometry_routes.inputs must name every geometry block; got incompatible inputs."
            )
        for route, parents in zip(
            normalized_routes, restriction_source.parent_cell_ids, strict=True
        ):
            if route.shape[0] != parents.shape[0]:
                raise ValueError(
                    "prepare_geometry_routes.inputs must provide one row per target cell; got incompatible inputs."
                )
    return _GeometryRoutes(tuple(normalized_routes))


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
    periodic_source: periodic_geometry_owner.PeriodicCellGeometrySource | None
    storage: cell_mesh_owner.CellMeshStorage | None

    @classmethod
    @checked
    def from_stored_fields(cls, value: CellGeometrySpec, /) -> CellGeometrySpec:
        return cls(
            dict(zip(value.block_names, value.elements, strict=True)),
            dict(zip(value.block_names, value.geometry_dofs, strict=True)),
            value.coordinates,
            storage=value.storage,
            restriction_source=value.restriction_source,
            exact_source=value.exact_source,
            periodic_source=value.periodic_source,
        )

    @checked
    def __init__(
        self,
        elements: Mapping[str, CellGeometryElement],
        geometry_dofs: Mapping[str, ArrayLike],
        coordinates: ArrayLike,
        /,
        *,
        storage: cell_mesh_owner.CellMeshStorage | None = None,
        restriction_source: CellGeometryRestrictionSource | None = None,
        exact_source: ExactCellGeometrySource | None = None,
        periodic_source: periodic_geometry_owner.PeriodicCellGeometrySource | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> None:
        if isinstance(coordinates, Array) and not coordinates.is_fully_addressable:
            raise ValueError(
                "CellGeometrySpec.inputs must provide an explicit locally addressable view; got incompatible inputs."
            )
        if any(
            isinstance(value, Array) and not value.is_fully_addressable
            for value in geometry_dofs.values()
        ):
            raise ValueError(
                "CellGeometrySpec.inputs must provide an explicit locally addressable lowering; got incompatible inputs."
            )
        items = tuple(sorted((str(name), element) for name, element in elements.items()))
        _require_geometry_source_binding(exact_source, items, storage, restriction_source)
        host_coordinates, host_routes = jax.device_get(
            (coordinates, tuple(geometry_dofs.values()))
        )
        routes = {
            str(name): np.asarray(value, dtype=np.int32)
            for name, value in zip(geometry_dofs, host_routes, strict=True)
        }
        points = np.asarray(host_coordinates, dtype=np.float64)
        if periodic_source is not None:
            if exact_source is not None or points.shape != (
                len(periodic_source.active_indices),
                periodic_source.representative_coordinates.shape[1],
            ):
                raise ValueError(
                    "CellGeometrySpec.inputs must bind its coordinate carrier view; got incompatible inputs."
                )
        if exact_source is not None:
            if isinstance(
                exact_source,
                (
                    exact_rational_geometry_owner.ExactRationalCellGeometrySource,
                    exact_algebraic_geometry_owner.ExactAlgebraicCellGeometrySource,
                ),
            ):
                if (
                    exact_rational_geometry_owner.rational_source_layouts(
                        dict(items), routes
                    )
                    != exact_source.layouts
                ):
                    raise ValueError(
                        "CellGeometrySpec.exact_source must bind its nominal elements and ordered coefficient routes; got a mismatched layout"
                    )
            if budget is None:
                work_upper, bytes_upper = exact_source_preparation_bounds(
                    exact_source, points.shape[0], points.shape[1]
                )
                with owned_coordinate_preparation(
                    work_upper,
                    bytes_upper,
                    maximum_geometry_queries=exact_source.preparation_query_upper(),
                ) as budget:
                    exact_source.prepare(points, budget=budget)
            else:
                exact_source.prepare(points, budget=budget)
        normalized_routes = _prepare_geometry_routes(
            items,
            routes,
            points,
            storage,
            restriction_source,
            exact_source,
            periodic_source,
        ).routes
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
        if storage is None:
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
                    "exact_source": None
                    if exact_source is None
                    else exact_source.source_id,
                    "periodic_source": None
                    if periodic_source is None
                    else periodic_source.source_layout_id,
                }
            )
        else:
            self.geometry_layout_id = canonical_fingerprint(
                {
                    "kind": "owner-local-cell-geometry-layout",
                    "topology": storage.logical_topology_id,
                    "coordinate_geometry": storage.logical_coordinate_geometry_id,
                    "coordinate_count": storage.global_coordinate_count,
                    "ambient_dimension": points.shape[1],
                }
            )

    @checked
    def source_coordinates(
        self,
        indices: ArrayLike | None = None,
        /,
        *,
        budget: CoordinateEnclosureBudget,
    ) -> CoordinateSourceBank:
        """Prepare current coefficients on the host, never in a kernel.
        Return the host coefficient bank for the numerical execution carrier.
        Source owners authenticate field-valued coefficients and their authorized
        numerical representatives. Optional one-dimensional indices select
        coefficient rows.
        """
        if any(
            isinstance(value, jax_core.Tracer)
            for value in jax.tree_util.tree_leaves(self)
        ):
            raise TypeError(
                "CellGeometrySpec.source_coordinates.inputs must satisfy the condition: coordinate source preparation is host-only; got incompatible inputs."
            )

        match self.exact_source:
            case (
                exact_plc_geometry_owner.ExactPlcCellGeometrySource()
                | exact_rational_geometry_owner.ExactRationalCellGeometrySource()
                | exact_algebraic_geometry_owner.ExactAlgebraicCellGeometrySource()
                | exact_plc_geometry_owner.ExactPlcCellGeometryConvexSource() as source
            ):
                bank_upper, scratch_upper = exact_plc_geometry_owner.plc_storage_bounds(
                    self.coordinates.shape[0], source.maximum_bits
                )
                budget.admit_work_bound(
                    self.coordinates.size, storage_upper=bank_upper + scratch_upper
                )
                coefficient_bytes = 0
            case None:
                coefficient_bytes = 512
            case (
                exact_power_geometry_owner.ExactPowerCellGeometrySource()
                | exact_power_geometry_owner.ExactPowerCellGeometryRestrictionSource()
                | exact_power_geometry_owner.ExactPowerCellGeometryLinearActionSource()
            ) as source:
                coefficient_bytes = 128 + 2 * (
                    32 + 4 * ((source.maximum_bits + 29) // 30)
                )
            case invalid:
                assert_never(invalid)
        budget.reserve(self.coordinates.size, self.coordinates.size * coefficient_bytes)
        if self.periodic_source is not None:
            bank: CoordinateSourceBank = self.periodic_source.source_coordinates(
                budget=budget
            )
        elif self.exact_source is not None:
            bank = self.exact_source.prepare(self.coordinates, budget=budget).vertices
        else:
            bank = tuple(
                tuple(Fraction(float(value)) for value in row)
                for row in np.asarray(jax.device_get(self.coordinates), dtype=np.float64)
            )
        if indices is None:
            return bank
        route = np.asarray(indices)
        if route.ndim != 1 or not np.issubdtype(route.dtype, np.integer):
            raise ValueError(
                "CellGeometrySpec.source_coordinates.inputs must provide one-dimensional integer indices; got incompatible inputs."
            )
        if np.any(route < 0) or np.any(route >= len(bank)):
            raise ValueError(
                "CellGeometrySpec.source_coordinates.inputs must satisfy the condition: source coefficient route references an absent coefficient; got incompatible inputs."
            )
        return tuple(bank[int(index)] for index in route)

    @checked
    def rational_source_coordinates(
        self,
        indices: ArrayLike | None = None,
        /,
        *,
        budget: CoordinateEnclosureBudget,
    ) -> tuple[tuple[Fraction, ...], ...]:
        """Return authenticated coefficient rows in the rational subfield."""
        bank = self.source_coordinates(indices, budget=budget)
        budget.reserve(0, 64 * len(bank) + 8 * sum(len(row) for row in bank))
        return tuple(
            tuple(
                exact_algebraic_geometry_owner.rational_coefficient(
                    value,
                    role="CellGeometrySpec.rational_source_coordinates.coefficients",
                )
                for value in row
            )
            for row in bank
        )

    @checked
    def source_execution_error(
        self, mesh: cell_mesh_owner.CellMesh, /, *, budget: CoordinateEnclosureBudget
    ) -> float:
        """Continuous physical bound for the representative coefficient-image error.

        This host proof uses the source basis, including rational
        restrictions. It adds no epsilon tolerance to a seam equality.
        """
        if self.periodic_source is None and self.exact_source is None:
            return 0.0

        elements, routes, runtime = self._resolve(
            mesh, exact_source_prepared=True, budget=budget
        )
        numeric = np.asarray(runtime, dtype=np.float64)
        source = self.source_coordinates(budget=budget)
        squared: coordinate_enclosure_owner.ExactScalar = Fraction(0)
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
                expressions = coordinate_enclosure_owner.coordinate_expressions(
                    element, errors, budget=budget
                )
                if expressions is None:
                    raise ValueError(
                        "CellGeometrySpec.source_execution_error.inputs must provide the coordinate source expressions; got incompatible inputs."
                    )
                radii = tuple(
                    max(
                        abs(value)
                        for value in coordinate_enclosure_owner.expression_bernstein_coefficients(
                            expression,
                            domain,
                            mesh.topological_dimension,
                            budget=budget,
                        )
                    )
                    for expression in expressions
                )
                squared = max(
                    squared, sum((value * value for value in radii), Fraction(0))
                )
        if not squared:
            return 0.0
        precision = 80
        squared_lower, squared_upper = (
            exact_algebraic_geometry_owner.coefficient_interval(
                squared, bits=precision, budget=budget
            )
        )
        while squared_lower <= 0 or squared_upper - squared_lower > squared_lower / (
            1 << 64
        ):
            precision *= 2
            squared_lower, squared_upper = (
                exact_algebraic_geometry_owner.coefficient_interval(
                    squared, bits=precision, budget=budget
                )
            )
        upper = hermitian_spectral_owner.fraction_sqrt_interval(
            squared_upper, budget=budget
        )[1]
        return fraction_endpoint(upper, upper=True, saturate=True)

    @classmethod
    @checked
    def affine(
        cls,
        mesh: cell_mesh_owner.CellMesh,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> CellGeometrySpec:
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
            budget=budget,
        )
        return (
            geometry.with_periodic_source(mesh, budget=budget)
            if mesh.periodic_topology is not None
            and mesh.storage is None
            and all(
                not isinstance(element, CellVertexGeometryElement)
                for element in geometry.elements
            )
            else geometry
        )

    @checked
    def with_periodic_source(
        self,
        mesh: cell_mesh_owner.CellMesh,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> CellGeometrySpec:
        """Compile the authored quotient interpretation through the FE owner."""
        if self.periodic_source is not None:
            return self
        if self.exact_source is not None:
            raise ValueError(
                "CellGeometrySpec.with_periodic_source.inputs must satisfy the condition: An coordinate source has no authored quotient interpretation; got incompatible inputs."
            )
        source = periodic_geometry_owner.PeriodicCellGeometrySource(
            mesh, self, budget=budget
        )
        return CellGeometrySpec(
            dict(zip(self.block_names, self.elements, strict=True)),
            dict(zip(self.block_names, self.geometry_dofs, strict=True)),
            self.coordinates,
            restriction_source=self.restriction_source,
            periodic_source=source,
            storage=self.storage,
            budget=budget,
        )

    @checked
    def with_coordinates(
        self,
        coordinates: ArrayLike,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> CellGeometrySpec:
        """Rebind dynamic coefficients without dropping their source graph."""
        source = (
            None
            if self.periodic_source is None
            else self.periodic_source.rebound(coordinates, budget=budget)
        )
        return CellGeometrySpec(
            dict(zip(self.block_names, self.elements, strict=True)),
            dict(zip(self.block_names, self.geometry_dofs, strict=True)),
            coordinates,
            restriction_source=self.restriction_source,
            exact_source=self.exact_source,
            periodic_source=source,
            storage=self.storage,
            budget=budget,
        )

    @classmethod
    @checked
    def power(
        cls,
        mesh: cell_mesh_owner.CellMesh,
        source: exact_power_geometry_owner.ExactPowerCellGeometrySource
        | exact_power_geometry_owner.ExactPowerCellGeometryRestrictionSource
        | exact_power_geometry_owner.ExactPowerCellGeometryLinearActionSource,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> CellGeometrySpec:
        """Bind an ideal power source to its independently verified RNE carrier."""
        if any(
            block.cell_kind not in ("polyhedron", "hexahedron") for block in mesh.blocks
        ):
            raise ValueError(
                "CellGeometrySpec.power.inputs must provide polyhedral or convex-parent Q1 hexahedral blocks; got incompatible inputs."
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
            budget=budget,
        )

    @classmethod
    @checked
    def plc(
        cls,
        mesh: cell_mesh_owner.CellMesh,
        source: exact_plc_geometry_owner.ExactPlcCellGeometrySource
        | exact_plc_geometry_owner.ExactPlcCellGeometryConvexSource,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> CellGeometrySpec:
        """Bind an PLC source or its convex target bank."""
        match source:
            case exact_plc_geometry_owner.ExactPlcCellGeometryConvexSource():
                kinds = ("tetrahedron", "hexahedron", "pyramid")
            case exact_plc_geometry_owner.ExactPlcCellGeometrySource():
                kinds = (
                    ("triangle",)
                    if source.source_points.shape[1] == 2
                    else ("tetrahedron",)
                )
            case invalid:
                assert_never(invalid)
        if (
            mesh.storage is not None
            or mesh.periodic_topology is not None
            or any(block.cell_kind not in kinds for block in mesh.blocks)
        ):
            raise ValueError(
                "CellGeometrySpec.plc.inputs must provide its direct global carrier; got incompatible inputs."
            )
        geometry = cls(
            {
                block.name: coordinate_lagrange_element(block.cell_kind, 1)
                for block in mesh.blocks
            },
            {block.name: block.vertices for block in mesh.blocks},
            mesh.coordinates,
            exact_source=source,
            budget=budget,
        )
        geometry.resolve(mesh, budget=budget)
        return geometry

    @checked
    def resolve(
        self,
        mesh: cell_mesh_owner.CellMesh,
        /,
        *,
        exact_source_prepared: bool = False,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> tuple[tuple[CellGeometryElement, ...], tuple[Array, ...], Array]:
        """Resolve block coordinate maps against the mesh.

        ``exact_source_prepared`` means the source coefficient bank has already
        been prepared within the supplied coordinate budget.
        """
        if budget is not None:
            return self._resolve(
                mesh, exact_source_prepared=exact_source_prepared, budget=budget
            )
        if self.exact_source is None and (
            self.restriction_source is None
            or self.restriction_source.carrier_publication == "source_image"
        ):
            return self._resolve_layout(mesh)
        work_upper, bytes_upper = cell_geometry_preparation_bounds(self)
        with owned_coordinate_preparation(
            work_upper,
            bytes_upper,
            maximum_geometry_queries=cell_geometry_preparation_query_upper(self),
        ) as budget:
            return self._resolve(
                mesh, exact_source_prepared=exact_source_prepared, budget=budget
            )

    def _resolve_layout(
        self, mesh: cell_mesh_owner.CellMesh, /
    ) -> tuple[tuple[CellGeometryElement, ...], tuple[Array, ...], Array]:
        if mesh.storage is None:
            if self.storage_id is not None:
                raise ValueError(
                    "CellGeometrySpec.resolve_layout.inputs must provide its bound mesh storage; got incompatible inputs."
                )
        elif (
            self.storage_id != mesh.storage.storage_id
            or self.logical_geometry_id != mesh.storage.logical_coordinate_geometry_id
        ):
            raise ValueError(
                "CellGeometrySpec.resolve_layout.inputs must match its logical storage binding; got incompatible inputs."
            )
        mapping = dict(zip(self.block_names, self.elements, strict=True))
        routes = dict(zip(self.block_names, self.geometry_dofs, strict=True))
        if set(mapping) != {block.name for block in mesh.blocks}:
            raise ValueError(
                "CellGeometrySpec.resolve_layout.inputs must match mesh blocks; got incompatible inputs."
            )
        resolved = tuple(mapping[block.name] for block in mesh.blocks)
        resolved_routes = tuple(routes[block.name] for block in mesh.blocks)
        for block, element, route in zip(
            mesh.blocks,
            resolved,
            resolved_routes,
            strict=True,
        ):
            if block.cell_kind != element.cell_kind:
                raise ValueError(
                    "CellGeometrySpec.resolve_layout.inputs must satisfy the condition: Coordinate element cell kind does not match its block; got incompatible inputs."
                )
            if route.shape[0] != block.cell_count:
                raise ValueError(
                    "CellGeometrySpec.resolve_layout.inputs must provide one row per cell; got incompatible inputs."
                )
        if self.periodic_source is not None:
            origin = self.periodic_source.source_mesh
            if mesh.periodic_topology is None or origin.periodic_topology is None:
                raise ValueError(
                    "CellGeometrySpec.resolve_layout.inputs must provide its authored periodic carrier; got incompatible inputs."
                )
            if periodic_exact_owner.identification_id(
                mesh.periodic_topology.cell
            ) != periodic_exact_owner.identification_id(origin.periodic_topology.cell):
                raise ValueError(
                    "CellGeometrySpec.resolve_layout.inputs must satisfy the condition: Coordinate coefficients are not bound to this authored periodic identification; got incompatible inputs."
                )
            if mesh.topology_id != origin.topology_id and (
                self.restriction_source is None
                or self.restriction_source.source_topology_id != origin.topology_id
            ):
                raise ValueError(
                    "CellGeometrySpec.resolve_layout.inputs must provide source-root restriction identity; got incompatible inputs."
                )
        return (
            resolved,
            resolved_routes,
            self.coordinates
            if self.periodic_source is None
            else self.periodic_source.runtime_coordinates(),
        )

    def _resolve(
        self,
        mesh: cell_mesh_owner.CellMesh,
        /,
        *,
        exact_source_prepared: bool,
        budget: CoordinateEnclosureBudget,
    ) -> tuple[tuple[CellGeometryElement, ...], tuple[Array, ...], Array]:
        resolved, resolved_routes, runtime = self._resolve_layout(mesh)
        if (
            self.restriction_source is not None
            and self.restriction_source.carrier_publication == "published_affine_roots"
            and self.exact_source is None
        ):
            require_published_affine_carrier(self, mesh, budget=budget)
        match self.exact_source:
            case exact_plc_geometry_owner.ExactPlcCellGeometryConvexSource() as source:
                source.prepare(self.coordinates, budget=budget)
                target = source.target_mesh
                if mesh.topology_id != target.topology_id:
                    raise ValueError(
                        "CellGeometrySpec.resolve.inputs must provide its target topology; got incompatible inputs."
                    )
                if mesh.storage is not None or mesh.periodic_topology is not None:
                    raise ValueError(
                        "CellGeometrySpec.resolve.inputs must provide its direct global carrier; got incompatible inputs."
                    )
                if not np.array_equal(
                    np.asarray(mesh.coordinates).view(np.uint64),
                    np.asarray(self.coordinates).view(np.uint64),
                ):
                    raise ValueError(
                        "CellGeometrySpec.resolve.inputs must match the bound mesh; got incompatible inputs."
                    )
                target_blocks = target.blocks
                if len(mesh.blocks) != len(target_blocks) or not np.array_equal(
                    np.asarray(mesh.vertex_global_ids),
                    np.asarray(target.vertex_global_ids),
                ):
                    raise ValueError(
                        "CellGeometrySpec.resolve.inputs must match its target identities; got incompatible inputs."
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
                        or not np.array_equal(
                            np.asarray(route), np.asarray(block.vertices)
                        )
                    ):
                        raise ValueError(
                            "CellGeometrySpec.resolve.inputs must provide its target scientific cells and vertex routes; got incompatible inputs."
                        )
            case exact_plc_geometry_owner.ExactPlcCellGeometrySource():
                _require_exact_plc_carrier(
                    self, mesh, resolved, resolved_routes, budget=budget
                )
            case (
                exact_rational_geometry_owner.ExactRationalCellGeometrySource()
                | exact_algebraic_geometry_owner.ExactAlgebraicCellGeometrySource()
            ) as source:
                if not exact_source_prepared:
                    source.prepare(self.coordinates, budget=budget)
            case None:
                pass
            case (
                exact_power_geometry_owner.ExactPowerCellGeometrySource()
                | exact_power_geometry_owner.ExactPowerCellGeometryRestrictionSource()
                | exact_power_geometry_owner.ExactPowerCellGeometryLinearActionSource()
            ) as source:
                if not exact_source_prepared:
                    source.prepare(self.coordinates, budget=budget)
                if not np.array_equal(
                    np.asarray(mesh.coordinates).view(np.uint64),
                    np.asarray(self.coordinates).view(np.uint64),
                ):
                    raise ValueError(
                        "CellGeometrySpec.resolve.inputs must match the bound mesh; got incompatible inputs."
                    )
            case invalid:
                assert_never(invalid)
        return resolved, resolved_routes, runtime


def _require_plc_composed_chart(
    element: PolynomialComposedCellGeometryElement | RationalComposedCellGeometryElement,
    source_kind: str,
) -> None:
    original = require_full_p1_source(element.source_element)
    kind = element.chart_element.cell_kind
    if source_kind == "triangle":
        expected = ("triangle", "quadrilateral")
    else:
        match element:
            case RationalComposedCellGeometryElement():
                expected = ("pyramid",)
            case PolynomialComposedCellGeometryElement():
                expected = ("tetrahedron", "hexahedron")
            case invalid:
                assert_never(invalid)
    if (
        original.cell_kind != source_kind
        or kind not in expected
        or element.chart_element.element_id
        != coordinate_lagrange_element(kind, 1).element_id
    ):
        raise ValueError(
            "require_plc_source_layout.inputs must provide P1 source simplices and target charts; got incompatible inputs."
        )
    corners = tuple(
        tuple(Fraction(a, b) for a, b in row) for row in element.chart_coefficients
    )
    if any(min(corner) < 0 or sum(corner, Fraction(0)) > 1 for corner in corners):
        raise ValueError(
            "require_plc_source_layout.inputs must lie in their source simplex; got incompatible inputs."
        )


def _require_plc_restricted_chart(
    source: exact_plc_geometry_owner.ExactPlcCellGeometrySource,
    element: RestrictedCellGeometryElement,
) -> None:
    """Require an admitted parent map and an affine action inside its reference."""
    _require_plc_source_layout(source, (element.source_element,))
    matrix = tuple(
        tuple(Fraction(float(value)) for value in row)
        for row in np.asarray(element.matrix)
    )
    offset = tuple(Fraction(float(value)) for value in np.asarray(element.offset))
    vertices = reference_cell_owner.reference_cell_topology(element.cell_kind).vertices
    for vertex in vertices:
        point = tuple(Fraction(float(value)) for value in vertex)
        image = tuple(
            shift + sum((value * xi for value, xi in zip(row, point, strict=True)))
            for row, shift in zip(matrix, offset, strict=True)
        )
        match element.source_element.cell_kind:
            case "triangle" | "tetrahedron":
                contained = min(image) >= 0 and sum(image) <= 1
            case "quadrilateral" | "hexahedron":
                contained = min(image) >= 0 and max(image) <= 1
            case "pyramid":
                x, y, z = image
                contained = (
                    0 <= z <= 1 and z / 2 <= x <= 1 - z / 2 and z / 2 <= y <= 1 - z / 2
                )
            case _:
                raise ValueError(
                    "require_plc_source_layout.restriction must have an admitted parent reference; got unsupported cell kind."
                )
        if not contained:
            raise ValueError(
                "require_plc_source_layout.restriction must lie in its parent reference; got an outside affine corner."
            )


def _require_plc_source_layout(
    source: exact_plc_geometry_owner.ExactPlcCellGeometrySource,
    elements: tuple[CellGeometryElement, ...],
) -> None:
    source_kind = "triangle" if source.source_points.shape[1] == 2 else "tetrahedron"
    affine = coordinate_lagrange_element(source_kind, 1).element_id
    required = (
        "CellGeometrySpec.elements must provide affine triangular coordinate elements; got another source layout."
        if source_kind == "triangle"
        else "CellGeometrySpec.elements must provide affine tetrahedral coordinate elements; got another source layout."
    )
    if not elements:
        raise ValueError(required)
    for element in elements:
        match element:
            case (
                PolynomialComposedCellGeometryElement()
                | RationalComposedCellGeometryElement()
            ):
                _require_plc_composed_chart(element, source_kind)
            case RestrictedCellGeometryElement():
                _require_plc_restricted_chart(source, element)
            case BarycentricCellGeometryElement():
                require_full_p1_source(element)
                if element.cell_kind != source_kind:
                    raise ValueError(required)
            case _:
                if element.element_id != affine:
                    raise ValueError(required)


def equivalent_coordinate_root(
    element: CellGeometryElement,
    controls: CoordinateSourceBank,
    root_element: CellGeometryElement,
    root_controls: CoordinateSourceBank,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> CellGeometryElement | None:
    """Prove a retained root law in the same explicitly identified reference cell.

    Callers must first bind scientific root cell and ordered vertex identities.
    This theorem changes only the nodal representation of that root, never its
    reference coordinates, nominal degree, or source identity.
    """
    current = element
    while current.element_id != root_element.element_id and isinstance(
        current,
        (
            RestrictedCellGeometryElement,
            PolynomialComposedCellGeometryElement,
            RationalComposedCellGeometryElement,
        ),
    ):
        current = current.source_element
    if current.element_id == root_element.element_id and controls == root_controls:
        return root_element
    if current.cell_kind != root_element.cell_kind:
        return None
    budget.reserve(
        sum(len(point) for point in controls) + sum(len(point) for point in root_controls)
    )
    actual = coordinate_enclosure_owner.coordinate_expressions(
        current, controls, budget=budget
    )
    expected = coordinate_enclosure_owner.coordinate_expressions(
        root_element, root_controls, budget=budget
    )
    if (
        actual is None
        or expected is None
        or not coordinate_enclosure_owner.expression_sequence_equal(
            actual, expected, budget=budget
        )
    ):
        return None
    return current


def _require_exact_source_layout(
    source: ExactCellGeometrySource,
    elements: tuple[CellGeometryElement, ...],
    /,
) -> None:
    """Admit only the coordinate layout family each source constructs."""
    match source:
        case (
            exact_rational_geometry_owner.ExactRationalCellGeometrySource()
            | exact_algebraic_geometry_owner.ExactAlgebraicCellGeometrySource()
        ):
            declared = tuple(
                (kind, degree, identity)
                for _, kind, degree, identity, _ in source.layouts
            )
            actual = tuple(
                (
                    element.cell_kind,
                    require_scalar_coordinate_element(
                        element, "exact rational source"
                    ).degree,
                    element.element_id,
                )
                for element in elements
            )
            if actual != declared:
                raise ValueError(
                    "require_exact_source_layout.inputs must match the declared interpolation layout; "
                    f"got declared={declared!r}, constructed={actual!r}"
                )
            for element in elements:
                root = element
                while isinstance(
                    root,
                    (
                        BarycentricCellGeometryElement,
                        RestrictedCellGeometryElement,
                        PolynomialComposedCellGeometryElement,
                        RationalComposedCellGeometryElement,
                    ),
                ):
                    root = root.source_element
                if (
                    not isinstance(root, reference_types_owner.FiniteElementSpec)
                    or root.element_id
                    != coordinate_lagrange_element(root.cell_kind, root.degree).element_id
                ):
                    raise ValueError(
                        "require_exact_source_layout.inputs must retain nodal roots through reference restrictions; got another source basis"
                    )
        case (
            exact_power_geometry_owner.ExactPowerCellGeometrySource()
            | exact_power_geometry_owner.ExactPowerCellGeometryRestrictionSource()
            | exact_power_geometry_owner.ExactPowerCellGeometryLinearActionSource()
        ):
            q1_source = source
            while isinstance(
                q1_source,
                exact_power_geometry_owner.ExactPowerCellGeometryRestrictionSource,
            ):
                q1_source = q1_source.parent
            q1 = coordinate_lagrange_element("hexahedron", 1).element_id
            for element in elements:
                if (
                    isinstance(element, CellVertexGeometryElement)
                    and element.cell_kind == "polyhedron"
                ):
                    continue
                if (
                    isinstance(
                        q1_source,
                        exact_power_geometry_owner.ExactPowerCellGeometryLinearActionSource,
                    )
                    and q1_source.periodic_preparation is None
                    and element.element_id == q1
                ):
                    continue
                raise ValueError(
                    "require_exact_source_layout.inputs must provide polyhedral layouts or nonperiodic convex-parent Q1 hexahedra; got incompatible inputs."
                )
        case exact_plc_geometry_owner.ExactPlcCellGeometryConvexSource():
            if not elements or any(
                element.cell_kind not in ("tetrahedron", "hexahedron", "pyramid")
                or element.element_id
                != coordinate_lagrange_element(element.cell_kind, 1).element_id
                for element in elements
            ):
                raise ValueError(
                    "require_exact_source_layout.inputs must provide degree-one tetrahedral, Q1 hexahedral, or rational pyramid elements; got incompatible inputs."
                )
        case exact_plc_geometry_owner.ExactPlcCellGeometrySource():
            _require_plc_source_layout(source, elements)
        case _:
            raise TypeError(
                "require_exact_source_layout.inputs must be an power or PLC source construction or None; got incompatible inputs."
            )


@checked
def prepared_published_affine_root_bank(
    geometry: CellGeometrySpec, /, *, budget: CoordinateEnclosureBudget
) -> coordinate_enclosure_owner.CoordinateSourceBank:
    """Retain the dyadic bank of the published root coefficient carriers."""
    key = (
        "published-affine-root-carriers",
        canonical_fingerprint(array_tree_fingerprint(geometry.coordinates)),
    )
    earned = budget.source_bank_cache.get(key)
    if earned is not None:
        return earned
    values = np.asarray(geometry.coordinates, dtype=np.float64)
    budget.reserve(values.size)
    bank = tuple(
        tuple(Fraction(float(value)) for value in values[row])
        for row in range(values.shape[0])
    )
    budget.retain_basis((key, bank))
    budget.source_bank_cache[key] = bank
    return bank


def _require_affine_root_publication(
    elements: tuple[CellGeometryElement, ...], /
) -> None:
    for element in elements:
        match element:
            case FiniteElementSpec() | BarycentricCellGeometryElement() as simplex:
                require_full_p1_source(simplex)
            case (
                PolynomialComposedCellGeometryElement()
                | RationalComposedCellGeometryElement() as simplex
            ):
                require_full_p1_source(simplex.source_element)
            case _:
                raise ValueError(
                    "CellGeometryRestrictionSource.carrier_publication must use affine simplex maps for published_affine_roots; got non-affine source cells"
                )
        if simplex.cell_kind not in ("triangle", "tetrahedron") or simplex.degree != 1:
            raise ValueError(
                "CellGeometryRestrictionSource.carrier_publication must use affine simplex maps for published_affine_roots; got non-affine source cells"
            )


@checked
def require_published_affine_carrier(
    geometry: CellGeometrySpec,
    mesh: cell_mesh_owner.CellMesh,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> None:
    """Verify retained root bytes and once-rounded ideal affine descendant images."""
    restriction = geometry.restriction_source
    if restriction is None or restriction.carrier_geometry is None:
        raise ValueError(
            "CellGeometrySpec.restriction_source must retain its affine carrier geometry; got missing publication evidence"
        )
    publication = restriction.carrier_geometry
    origin = publication.restriction_source
    if origin is None:
        raise ValueError(
            "CellGeometrySpec.carrier_geometry.restriction_source must bind published root identities; got None"
        )
    elements, routes, _ = publication.resolve(mesh, budget=budget)
    _require_affine_root_publication(elements)
    bank = prepared_published_affine_root_bank(publication, budget=budget)
    root_values = np.asarray(publication.coordinates, dtype=np.float64)
    carrier = np.asarray(mesh.coordinates, dtype=np.float64)
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    for block, element, route in zip(mesh.blocks, elements, routes, strict=True):
        width = element.local_dof_count
        weights = coordinate_enclosure_owner.coordinate_corner_images(
            element,
            tuple(
                tuple(Fraction(int(row == column)) for column in range(width))
                for row in range(width)
            ),
            budget=budget,
        )
        if weights is None:
            raise ValueError(
                "CellGeometrySpec.carrier_geometry.elements must expose their rational affine corner actions; got unsupported basis"
            )
        root_ids = np.asarray(origin.block_parent_vertex_ids[block.name], dtype=np.int64)
        for row, (vertices, coefficients) in enumerate(
            zip(np.asarray(block.vertices), np.asarray(route), strict=True)
        ):
            images = coordinate_enclosure_owner.coordinate_corner_images(
                element, tuple(bank[int(index)] for index in coefficients), budget=budget
            )
            if images is None:
                raise ValueError(
                    "CellGeometrySpec.carrier_geometry.elements must expose their rational affine corner images; got unsupported basis"
                )
            for corner, (vertex, image) in enumerate(zip(vertices, images, strict=True)):
                support = tuple(
                    index for index, weight in enumerate(weights[corner]) if weight
                )
                root = (
                    len(support) == 1
                    and weights[corner][support[0]] == 1
                    and vertex_ids[vertex] == root_ids[row, support[0]]
                )
                expected = (
                    root_values[coefficients[support[0]]]
                    if root
                    else np.asarray(
                        coordinate_enclosure_owner.rounded_point(image), dtype=np.float64
                    )
                )
                if not np.array_equal(
                    carrier[vertex].view(np.uint64), expected.view(np.uint64)
                ):
                    raise ValueError(
                        f"CellGeometrySpec.carrier_geometry must reproduce retained root bytes and rounded ideal descendants; got vertex={int(vertex_ids[vertex])}"
                    )


def _require_exact_plc_carrier(
    geometry: CellGeometrySpec,
    mesh: cell_mesh_owner.CellMesh,
    elements: tuple[CellGeometryElement, ...],
    routes: tuple[Array, ...],
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> None:
    """Bind each carrier corner to the retained P1 coefficient law."""
    restriction = geometry.restriction_source
    if (
        restriction is not None
        and restriction.carrier_publication == "published_affine_roots"
    ):
        coordinate_enclosure_owner.prepared_coordinate_source_bank(
            geometry, budget=budget
        )
        require_published_affine_carrier(geometry, mesh, budget=budget)
        return

    bank = coordinate_enclosure_owner.prepared_coordinate_source_bank(
        geometry, budget=budget
    )
    carrier = np.asarray(mesh.coordinates, dtype=np.float64)
    approved_carrier = np.asarray(geometry.coordinates, dtype=np.float64)
    publication_bank = bank
    for block, element, route in zip(mesh.blocks, elements, routes, strict=True):
        for vertices, coefficients in zip(
            np.asarray(block.vertices), np.asarray(route), strict=True
        ):
            images = coordinate_enclosure_owner.coordinate_corner_images(
                element,
                tuple(publication_bank[int(index)] for index in coefficients),
                budget=budget,
            )
            if images is None:
                raise ValueError(
                    "require_exact_plc_carrier.inputs must provide their P1 source law; got incompatible inputs."
                )
            for vertex, image in zip(vertices, images, strict=True):
                if (
                    vertex < len(bank)
                    and image == bank[int(vertex)]
                    and np.array_equal(
                        carrier[vertex].view(np.uint64),
                        approved_carrier[vertex].view(np.uint64),
                    )
                ):
                    continue
                if not np.array_equal(
                    carrier[vertex].view(np.uint64),
                    np.asarray(
                        coordinate_enclosure_owner.rounded_point(image), dtype=np.float64
                    ).view(np.uint64),
                ):
                    raise ValueError(
                        "CellGeometrySpec.resolve.inputs must match the bound mesh; got incompatible inputs."
                    )


@jax.jit
def _logical_geometry_row_values(
    identifiers: Array, queries: Array
) -> tuple[Array, Array]:
    table = jnp.where(identifiers >= 0, identifiers, jnp.iinfo(jnp.int64).max)
    positions = jnp.searchsorted(table, queries)
    safe = jnp.minimum(positions, identifiers.shape[0] - 1)
    return safe, jnp.all(
        (positions < identifiers.shape[0]) & (identifiers[safe] == queries)
    )


def _logical_values_equal(actual: Array, expected: Array) -> Array:
    """Retain NumPy equality, including signed zero, NaNs and subnormal values."""
    if actual.shape != expected.shape:
        return jnp.asarray(False)
    if actual.dtype == jnp.float64:
        first = jax.lax.bitcast_convert_type(actual, jnp.uint64)
        second = jax.lax.bitcast_convert_type(expected, jnp.uint64)
        magnitude = jnp.uint64(0x7FFFFFFFFFFFFFFF)
        infinity = jnp.uint64(0x7FF0000000000000)
        first_magnitude, second_magnitude = first & magnitude, second & magnitude
        equal = (first == second) | ((first_magnitude == 0) & (second_magnitude == 0))
        return jnp.all(
            equal & (first_magnitude <= infinity) & (second_magnitude <= infinity)
        )
    return jnp.all(actual == expected)


@jax.jit
def _logical_selected_values_equal(table: Array, rows: Array, expected: Array) -> Array:
    return _logical_values_equal(table[rows], expected)


@jax.jit
def _logical_routed_values_equal(
    table: Array, rows: Array, identifiers: Array, routes: Array
) -> Array:
    return _logical_values_equal(table[rows], identifiers[routes])


def _require_logical_selection_equal(
    table: Array, rows: Array, expected: ArrayLike, name: str
) -> None:
    # Keep heterogeneous/other numerical dtypes at the existing NumPy boundary.
    if getattr(expected, "dtype", None) != table.dtype or (
        table.dtype != jnp.float64 and table.dtype.kind not in "biu"
    ):
        _require_logical_equal(table[rows], expected, name)
    elif not bool(_logical_selected_values_equal(table, rows, jnp.asarray(expected))):
        raise ValueError(
            f"CellGeometrySpec.logical_values must match logical {name}; got different local values."
        )


def _require_logical_routes_equal(
    table: Array, rows: Array, identifiers: Array, routes: Array
) -> None:
    name = "source coefficient routes"
    if table.dtype != identifiers.dtype or table.dtype.kind not in "biu":
        _require_logical_equal(table[rows], identifiers[routes], name)
    elif not bool(_logical_routed_values_equal(table, rows, identifiers, routes)):
        raise ValueError(
            f"CellGeometrySpec.logical_values must match logical {name}; got different local values."
        )


def _logical_geometry_rows(
    identifiers: Array,
    queries: ArrayLike,
    /,
) -> Array:
    """Locate only requested scientific rows without a host-global conversion."""
    query = jnp.asarray(queries, dtype=jnp.int64)
    if identifiers.ndim != 1 or identifiers.dtype != jnp.int64:
        raise ValueError(
            "logical_geometry_rows.inputs must be int64 vectors; got incompatible inputs."
        )
    if query.shape[0] == 0:
        return jnp.empty(query.shape, dtype=jnp.int64)
    if identifiers.shape[0] == 0:
        raise ValueError(
            "logical_geometry_rows.inputs must satisfy the condition: Local coordinate or cell identity is absent from logical geometry; got incompatible inputs."
        )
    safe, present = _logical_geometry_row_values(identifiers, query)
    if not bool(present):
        raise ValueError(
            "logical_geometry_rows.inputs must satisfy the condition: Local coordinate or cell identity is absent from logical geometry; got incompatible inputs."
        )
    return safe


def _require_logical_equal(actual: ArrayLike, expected: ArrayLike, name: str) -> None:
    if not np.array_equal(np.asarray(actual), np.asarray(expected)):
        raise ValueError(
            f"CellGeometrySpec.logical_values must match logical {name}; got different local values."
        )


def _require_uniform_chart_action(
    element: PolynomialComposedCellGeometryElement,
    stack: np.ndarray,
    width: int,
) -> None:
    exact_rows: list[list[Fraction]] = []
    for row in stack[-1]:
        support = np.flatnonzero(row)
        if support.size == 0 or not np.all(row[support] == 1.0 / support.size):
            raise ValueError(
                "geometry.uniform_action must contain uniformly weighted subset rows; "
                "got nonuniform source support"
            )
        exact_rows.append(
            [
                Fraction(1, support.size) if column in support else Fraction(0)
                for column in range(width)
            ]
        )
    exact = np.asarray(exact_rows, dtype=object)
    for action in reversed(stack[:-1]):
        dyadic = np.asarray(
            [[Fraction(float(value)) for value in row] for row in action],
            dtype=object,
        )
        exact = dyadic @ exact
    coefficients = tuple(
        tuple((value.numerator, value.denominator) for value in row[1:]) for row in exact
    )
    if element.chart_coefficients != coefficients:
        raise ValueError(
            "geometry.chart_coefficients must equal the rational composed source path; "
            "got different uniform or binary coefficients"
        )
    _require_logical_equal(
        element.chart_coordinates,
        np.asarray(
            [[float(value) for value in row[1:]] for row in exact],
            dtype=np.float64,
        ),
        "rounded exact reference chart",
    )


def _require_logical_actions(
    element: CellGeometryElement,
    root: CellGeometryElement,
    stored: Array,
    counts: Array,
    cell_count: int,
    /,
) -> None:
    """Authenticate one composed chart against every retained raw source path."""
    weights, depths = np.asarray(stored), np.asarray(counts)
    width = root.local_dof_count
    if (
        weights.ndim != 4
        or weights.shape[0] != cell_count
        or weights.shape[2:] != (width, width)
        or weights.dtype != np.float64
        or depths.shape != (cell_count,)
        or depths.dtype != np.int32
        or np.any(depths < 0)
        or np.any(depths > weights.shape[1])
    ):
        raise ValueError(
            "geometry.action_stack must retain its cell, source, and depth axes; "
            "got incompatible weights or counts"
        )
    if element is root:
        _require_logical_equal(
            depths, np.zeros((cell_count,), dtype=np.int32), "identity action depth"
        )
        return
    match element:
        case BarycentricCellGeometryElement():
            packed = False
            if element.source_element is not root:
                raise ValueError(
                    "geometry.action_source must be the declared coefficient basis; "
                    "got a nested presentation action"
                )
        case PolynomialComposedCellGeometryElement():
            packed = True
            if (
                element.source_element is not root
                or width != root.topological_dimension + 1
                or element.chart_element.element_id
                != coordinate_lagrange_element(element.cell_kind, 1).element_id
            ):
                raise ValueError(
                    "geometry.action_source must bind one degree-one chart over its simplex source; "
                    "got an incompatible coefficient source or chart basis"
                )
        case _:
            raise ValueError(
                "geometry.action_element must carry the composed source action; "
                "got an incompatible coordinate element"
            )
    for cell in range(cell_count):
        stack = weights[cell, : depths[cell]]
        if stack.shape[0] == 0 or not np.all(np.isfinite(stack)):
            raise ValueError(
                "geometry.action_stack must contain finite nonempty source paths; "
                "got an empty or nonfinite composed action"
            )
        binary = stack[:-1] if packed else stack
        if np.any((binary != 0.0) & (binary != 0.5) & (binary != 1.0)) or np.any(
            np.sum(binary, axis=-1) != 1.0
        ):
            raise ValueError(
                "geometry.binary_action must retain dyadic simplex midpoint rows; "
                "got different weights or incomplete source support"
            )
        match element:
            case BarycentricCellGeometryElement():
                composed = np.eye(width, dtype=np.float64)
                for action in reversed(stack):
                    composed = action @ composed
                _require_logical_equal(
                    element.barycentric_weights,
                    composed,
                    "composed binary coefficient action",
                )
            case PolynomialComposedCellGeometryElement():
                _require_uniform_chart_action(element, stack, width)
            case _:
                raise ValueError(
                    "geometry.action_element must retain its admitted chart kind; "
                    "got an incompatible coordinate element"
                )


def _require_logical_geometry_block(
    block: reference_cell_types_owner.CellBlock
    | reference_cell_types_owner.PolyhedralBlock,
    geometry: CellGeometrySpec,
    arrays: Mapping[str, Array],
    coordinate_ids: Array,
    source_blocks: Mapping[str, str],
    elements: Mapping[str, CellGeometryElement],
    routes: Mapping[str, Array],
    origin: CellGeometryRestrictionSource | None,
    required: Callable[[str], Array],
) -> None:
    name = block.name
    bank = source_blocks[name]
    cell_rows = _logical_geometry_rows(
        required(f"geometry/cell_ids/{bank}"), block.global_ids
    )
    route = routes[name]
    if route.shape[0] != block.cell_count:
        raise ValueError(
            "require_logical_geometry_block.inputs must provide one row per local cell; got incompatible inputs."
        )
    _require_logical_routes_equal(
        required(f"geometry/routes/{bank}"), cell_rows, coordinate_ids, route
    )
    element = elements[name]
    if element.cell_kind != block.cell_kind:
        raise ValueError(
            "require_logical_geometry_block.inputs must match its target reference kind; got incompatible inputs."
        )
    root, matrix, offset = _logical_source_basis(
        element,
        required(f"geometry/source_basis/{bank}")[cell_rows],
        block.topological_dimension,
    )
    action_name = f"geometry/action_weights/{bank}"
    count_name = f"geometry/action_counts/{bank}"
    if (action_name in arrays) != (count_name in arrays):
        raise ValueError(
            "require_logical_geometry_block.inputs must provide their counts; got incompatible inputs."
        )
    match element:
        case CellGeometryElement() if action_name in arrays:
            _require_logical_actions(
                element,
                root,
                required(action_name)[cell_rows],
                required(count_name)[cell_rows],
                block.cell_count,
            )
        case CellVertexGeometryElement():
            pass
        case CellGeometryElement():
            if matrix is None or offset is None:
                raise ValueError(
                    "require_logical_geometry_block.inputs must provide their barycentric witnesses; got incompatible inputs."
                )
            matrix_name, offset_name = (
                f"geometry/matrix/{bank}",
                f"geometry/offset/{bank}",
            )
            if matrix_name in arrays or offset_name in arrays:
                _require_logical_selection_equal(
                    required(matrix_name),
                    cell_rows,
                    np.broadcast_to(matrix, (block.cell_count,) + matrix.shape),
                    "exact reference matrix",
                )
                _require_logical_selection_equal(
                    required(offset_name),
                    cell_rows,
                    np.broadcast_to(offset, (block.cell_count,) + offset.shape),
                    "exact reference offset",
                )
            elif root is not element:
                raise ValueError(
                    "require_logical_geometry_block.inputs must provide reference matrix and offset witnesses; got incompatible inputs."
                )
            ancestry = nonrepresentable_reference_ancestry(
                element, matrix, offset, source_basis=root
            )
            if ancestry is not None:
                exact_digest = np.frombuffer(bytes.fromhex(ancestry), dtype=np.uint8)
                _require_logical_selection_equal(
                    required(f"geometry/reference_ancestry/{bank}"),
                    cell_rows,
                    np.broadcast_to(exact_digest, (block.cell_count, 32)),
                    "exact composed ancestry",
                )
        case _:
            raise TypeError(
                "CellGeometrySpec.elements must contain coordinate source elements; got an unsupported stored element kind."
            )
    if (
        isinstance(
            element,
            (
                BarycentricCellGeometryElement,
                RestrictedCellGeometryElement,
                PolynomialComposedCellGeometryElement,
            ),
        )
        and origin is None
    ):
        raise ValueError(
            "require_logical_geometry_block.inputs must provide scientific root ancestry; got incompatible inputs."
        )
    if origin is not None:
        _require_logical_selection_equal(
            required(f"geometry/parent_cell_ids/{bank}"),
            cell_rows,
            origin.block_parent_cell_ids[name],
            "scientific root cell identities",
        )
        _require_logical_selection_equal(
            required(f"geometry/parent_vertex_ids/{bank}"),
            cell_rows,
            origin.block_parent_vertex_ids[name],
            "ordered scientific root corners",
        )


@checked
def require_logical_geometry_lowering(
    geometry: CellGeometrySpec,
    blocks: Sequence[
        reference_cell_types_owner.CellBlock | reference_cell_types_owner.PolyhedralBlock
    ],
    logical_arrays: Sequence[tuple[str, Array]],
    global_coordinate_count: int,
    coordinate_ids: Array,
    coordinate_owners: Array,
    source_blocks: Mapping[str, str],
    /,
) -> None:
    """Prove lowered source coefficients, composed charts, and retained raw paths.

    Raw action depth belongs to the epoch history, not the presentation element.
    Each local cell's path is checked at this output-size host boundary.
    """

    arrays = dict(logical_arrays)

    def required(name: str) -> Array:
        if name not in arrays:
            raise ValueError(
                f"require_logical_geometry_lowering.required.inputs must satisfy the condition: Mapped storage lacks scientific logical array{name!r}.; got incompatible inputs."
            )
        return arrays[name]

    logical_ids = required("geometry/coordinate_ids")
    if (
        logical_ids.ndim != 1
        or logical_ids.shape[0] < global_coordinate_count
        or logical_ids.dtype != jnp.int64
    ):
        raise ValueError(
            "require_logical_geometry_lowering.inputs must match the declared scientific DOF count; got incompatible inputs."
        )
    rows = _logical_geometry_rows(logical_ids[:global_coordinate_count], coordinate_ids)
    logical_points = required("geometry/coordinates")
    if (
        logical_points.shape != (logical_ids.shape[0], geometry.coordinates.shape[1])
        or logical_points.dtype != jnp.float64
        or required("geometry/coordinate_owners").shape != logical_ids.shape
    ):
        raise ValueError(
            "require_logical_geometry_lowering.inputs must match their coordinate layout; got incompatible inputs."
        )
    _require_logical_selection_equal(
        logical_points, rows, geometry.coordinates, "original source coefficients"
    )
    _require_logical_selection_equal(
        required("geometry/coordinate_owners"),
        rows,
        coordinate_owners,
        "coordinate ownership",
    )
    if set(geometry.block_names) != {block.name for block in blocks}:
        raise ValueError(
            "require_logical_geometry_lowering.inputs must name every local cell block; got incompatible inputs."
        )
    elements = dict(zip(geometry.block_names, geometry.elements, strict=True))
    routes = dict(zip(geometry.block_names, geometry.geometry_dofs, strict=True))
    origin = geometry.restriction_source
    if origin is not None:
        for name, identity in (
            ("source_geometry_id", origin.source_geometry_id),
            ("source_topology_id", origin.source_topology_id),
        ):
            _require_logical_equal(
                required(f"geometry/{name}"),
                np.frombuffer(
                    bytes.fromhex(canonical_fingerprint(identity)), dtype=np.uint8
                ),
                name,
            )
    for block in blocks:
        _require_logical_geometry_block(
            block,
            geometry,
            arrays,
            coordinate_ids,
            source_blocks,
            elements,
            routes,
            origin,
            required,
        )


@checked
def restore_authored_storage_geometry(
    blocks: Sequence[
        reference_cell_types_owner.CellBlock | reference_cell_types_owner.PolyhedralBlock
    ],
    source_blocks: Mapping[str, str],
    projection: CellGeometryStorageProjection,
    partition_index: int,
    coordinate_ids: ArrayLike,
    /,
) -> CellGeometrySpec:
    """Restore an authored source chart from coefficient routes, not corners."""
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
                "restore_authored_storage_geometry.inputs must provide its typed scientific source element; got incompatible inputs."
            )
        element = declared[bank]
        if isinstance(element, RestrictedCellGeometryElement):
            raise ValueError(
                "restore_authored_storage_geometry.inputs must provide its explicit ancestry specification; got incompatible inputs."
            )
        rows = np.asarray(
            _logical_geometry_rows(packets[f"geometry/cell_ids/{bank}"], block.global_ids)
        )
        action_name = f"geometry/action_weights/{bank}"
        count_name = f"geometry/action_counts/{bank}"
        if (action_name in packets) != (count_name in packets):
            raise ValueError(
                "restore_authored_storage_geometry.inputs must provide their row counts; got incompatible inputs."
            )
        if action_name in packets:
            counts = np.asarray(packets[count_name])[rows]
            if np.any(counts != 0):
                raise ValueError(
                    "restore_authored_storage_geometry.inputs must provide their barycentric geometry owner; got incompatible inputs."
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
                    "restore_authored_storage_geometry.inputs must provide their restricted geometry specification; got incompatible inputs."
                )
        scientific_routes = np.asarray(packets[f"geometry/routes/{bank}"])[rows]
        if any(int(value) not in slots for value in scientific_routes.reshape(-1)):
            raise ValueError(
                "restore_authored_storage_geometry.inputs must provide their local coordinate DOFs; got incompatible inputs."
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

    matrix = np.eye(dimension, dtype=np.float64)
    offset = np.zeros((dimension,), dtype=np.float64)
    current = element
    basis = np.asarray(logical_basis)
    while True:
        signature = canonical_fingerprint(
            {
                "signature": coordinate_enclosure_owner.coordinate_source_signature(
                    current
                ),
                "arrays": array_tree_fingerprint(current),
            }
        )
        digest = np.frombuffer(bytes.fromhex(signature), dtype=np.uint8)
        if np.array_equal(basis, np.broadcast_to(digest, (basis.shape[0], digest.size))):
            return current, matrix, offset
        if isinstance(
            current,
            (BarycentricCellGeometryElement, PolynomialComposedCellGeometryElement),
        ):
            if current is not element:
                raise ValueError(
                    "logical_source_basis.inputs must satisfy the condition: Full coefficient action lost its cardinal basis; got incompatible inputs."
                )
            source, _, _ = _logical_source_basis(
                current.source_element, logical_basis, dimension
            )
            return source, None, None
        if not isinstance(current, RestrictedCellGeometryElement):
            raise ValueError(
                "logical_source_basis.inputs must match logical source basis; got incompatible inputs."
            )
        source_matrix = np.asarray(current.matrix)
        matrix = source_matrix @ matrix
        offset = source_matrix @ offset + np.asarray(current.offset)
        current = current.source_element


def nonrepresentable_reference_ancestry(
    element: CellGeometryElement,
    rounded_matrix: np.ndarray,
    rounded_offset: np.ndarray,
    /,
    *,
    source_basis: CellGeometryElement | None = None,
) -> str | None:
    """Bind composition when binary64 flattening cannot retain its source map."""
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


@checked
def require_storage_geometry(
    storage: cell_mesh_owner.CellMeshStorage,
    names: tuple[str, ...],
    elements: tuple[CellGeometryElement, ...],
    routes: Mapping[str, ArrayLike],
    points: ArrayLike,
    restriction_source: CellGeometryRestrictionSource | None,
    exact_source: ExactCellGeometrySource | None,
    periodic_source: periodic_geometry_owner.PeriodicCellGeometrySource | None,
    /,
) -> None:
    """Reject mutated local maps independently of their cached layout identities."""
    # Storage replay validation consumes the coordinate-spec owner defined in this module.
    from . import _cell_geometry_validity as cell_geometry_validity_owner

    if not storage.local_blocks:
        projection = storage.geometry_projection
        if projection is None or not projection.source_elements:
            raise ValueError(
                "require_storage_geometry.inputs must provide its global source basis projection; got incompatible inputs."
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
                "require_storage_geometry.inputs must match its zero-resident source projection; got incompatible inputs."
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
        if (
            cell_geometry_validity_owner.cell_geometry_id(candidate)
            != storage.local_geometry_source_id
        ):
            raise ValueError(
                "CellGeometrySpec.coordinates must match the source coefficient binding; got different local values."
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
        require_logical_geometry_lowering(
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
            "require_storage_geometry.inputs must match its coordinate view; got incompatible inputs."
        )
    blocks = {block.name: block for block in storage.local_blocks}
    if set(blocks) != set(names):
        raise ValueError(
            "require_storage_geometry.inputs must match the stored block names; got incompatible inputs."
        )
    for name, element in zip(names, elements, strict=True):
        block = blocks[name]
        expected = (
            CellVertexGeometryElement(block.cell_kind, block.arity)
            if block.cell_kind in ("polygon", "polyhedron")
            else coordinate_lagrange_element(block.cell_kind, 1)
        )
        if (
            coordinate_enclosure_owner.coordinate_source_signature(element)
            != coordinate_enclosure_owner.coordinate_source_signature(expected)
            or array_tree_fingerprint(element) != array_tree_fingerprint(expected)
            or not np.array_equal(np.asarray(routes[name]), np.asarray(block.vertices))
        ):
            raise ValueError(
                "require_storage_geometry.inputs must match its coordinate routes; got incompatible inputs."
            )


__all__ = [
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
