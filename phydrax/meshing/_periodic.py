#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Periodic construction orbits, quotient publication and orbit-preserving refinement.

Translational periodic constraints compile into one ``PeriodicCell``; seed and
feature points compile into lattice orbits with one representative each and
the lattice shift of every input copy. Construction triangulates the orbits on
the flat torus (``PeriodicDelaunayTriangulation``) and publishes one lifted
simplex per quotient orbit as a ``CellMesh`` with ``PeriodicMeshTopology``.

Refinement edits the quotient complex, never a displayed fundamental cell: a
quotient edge is keyed by its representative endpoints and their relative
lattice shift, and bisecting it splits every cell of its orbit star at the one
new quotient vertex, each in its own local lift. All seam copies of the edge
are therefore refined together and the result is conforming on the quotient.

The closed affine quotient route combines full-source positive-cell validity,
the topology owner's oppositely oriented two-sided facet admission, and provably
interior-disjoint group lifts. The local-star degree is a positive integer:
distinct preimages of a closed contact would each cover an open neighborhood
and contradict interior disjointness. Nonclosed quotients retain closed mapped
contact predicates. See Lipman, *Bijective Mappings of Meshes with Boundary and
the Degree in Mesh Processing*, Section 3 (https://arxiv.org/abs/1310.0955).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Generator, Iterator, Mapping, Sequence
from contextlib import closing, contextmanager, nullcontext
from fractions import Fraction
from itertools import combinations
from types import MappingProxyType
from typing import assert_never, Literal, NamedTuple

import equinox as eqx
import jax
import numpy as np
from jax.typing import ArrayLike
from numpy.typing import NDArray

import phydrax.discretization._cell_mesh as cell_mesh_owner
import phydrax.discretization._periodic_cell as periodic_cell_owner
import phydrax.discretization._periodic_topology as periodic_topology_owner
import phydrax.discretization._reference_cell_types as reference_cell_types_owner
import phydrax.meshing._host_banks as _leaf_phydrax_meshing__host_banks
from phydrax.discretization._periodic_exact import (
    exact_periodic_element,
    exact_periodic_generators,
    identification_id,
    identification_orders,
    PeriodicIsometryGroup,
    reduce_periodic_exponents,
)
from phydrax.discretization._sphere_chart_records import PreparedSphereChartDeformation
from phydrax.discretization._surface_chart_records import PreparedSurfaceChartDeformation
from phydrax.geometry._embedding_state import (
    coordinate_certificate_request,
    MeshCertificateLimits,
)
from phydrax.geometry._rational import (
    dyadic_integers,
    fraction_endpoint,
)
from phydrax.linalg._small_batched import cofactor_determinant_2, cofactor_determinant_3
from phydrax.meshing._association_state import GeometryAssociation
from phydrax.meshing._contracts import MeshingLimits
from phydrax.meshing._failures import MeshingFailure, MeshingFailureCategory

from .._bvh import bvh_overlap_pair_blocks, BVHBuildPolicy, PackedBVH, prepare_bvh
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._geometry_predicates import bigint_bytes
from .._meshcore import NativeExecutionScope
from .._model._structure import model_array_bytes
from .._resource_ticket import HostBatchTicket
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import _periodic_measures as periodic_measures_owner
from ..discretization._cell_complex import (
    PolygonalConnectivity,
    PolyhedralConnectivity,
    TetrahedralConnectivity,
)
from ..discretization._cell_geometry import (
    cell_geometry_preparation_bounds,
    cell_geometry_preparation_query_upper,
    CellGeometrySpec,
    coordinate_lagrange_element,
)
from ..discretization._cell_geometry_transition_types import (
    CellGeometryTransition,
    NestedReferenceWitnesses,
)
from ..discretization._cell_geometry_validity import (
    cell_geometry_id,
    CellValidityCertificate,
    CellValidityPolicy,
    certify_cell_geometry_validity,
)
from ..discretization._cell_mesh import SimplicialConnectivity
from ..discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateEnclosureResourceError,
    CoordinateLiveStorage,
)
from ..discretization._coordinate_enclosure import (
    coordinate_corner_images,
    coordinate_expressions,
    CoordinateSourceBank,
    prepared_coordinate_source_bank,
    rounded_point,
)
from ..discretization._exact_algebraic_geometry import (
    coefficient_interval,
    ExactAlgebraicCellGeometrySource,
    RadicalElement,
    reserve_algebraic_work,
)
from ..discretization._exact_power_geometry import (
    ExactPowerCellGeometryLinearActionSource,
    ExactPowerCellGeometryRestrictionSource,
    ExactPowerCellGeometrySource,
)
from ..discretization._periodic_topology import periodic_preparation_bounds
from ..geometry import _affine_mapped_contact as affine_mapped_contact_owner
from ..geometry._global_embedding import (
    certify_global_embedding,
    GlobalEmbeddingCertificate,
)
from ..geometry._periodic_image_frame import (
    periodic_exact_product,
    PeriodicEmbeddingResourceError,
    PeriodicImage,
    PeriodicImageFrame,
    prepare_periodic_image_frame,
    reserve_periodic_exact_terms,
)
from ..geometry._triangulation import (
    PeriodicDelaunayTriangulation,
    PeriodicTriangulationEvidence,
)
from ..typing import checked, Dim, HostInt64, parse
from . import _adaptation as adaptation_owner, _layer_core as layer_core_owner
from ._association import SurfaceAssociationTransfer
from ._bisection import BisectionEvidence
from ._entity_keys import entity_keys, key_rows
from ._lineage_records import (
    EntityLineageKind,
    MeshLineage,
    MeshTransitionKind,
    VertexInterpolationStencil,
)
from ._organization_records import MeshZoneRole
from ._periodic_constraint import PeriodicConstraint
from ._result import CellMeshingResult
from ._scope import resolve_mesh_scope
from ._topology_edit import (
    assemble_topology_edit,
    build_mesh,
    CellTopologyEdit,
    entity_lineage,
    EntityRelations,
    PeriodicEntityIdentityBank,
    PeriodicNonnestedGeometryAuthority,
    PeriodicVertexOrbitWitness,
    PrescribedEntityIds,
    require_periodic_nonnested_geometry,
    require_periodic_nonnested_source,
    TopologyEditBlock,
)
from .providers._native_route_checks import edge_growth_evidence, published_entity_limits


@checked
def feature_entity_mask(
    source: cell_mesh_owner.CellMesh,
    target: cell_mesh_owner.CellMesh,
    identifiers: np.ndarray,
    degree: int,
    *,
    scope: NativeExecutionScope,
    budget: CoordinateEnclosureBudget,
) -> np.ndarray:
    """Recognize affine source-edge/face subdivisions on every lifted copy."""
    original = require_periodic_topology(source)
    orbit = np.asarray(original.orbits(degree)[0])
    source_rows = simplex_entity_corners(source, degree)
    target_rows = simplex_entity_corners(target, degree)
    segments = np.asarray(source.coordinates)[source_rows[np.isin(orbit, identifiers)]]
    corners = np.asarray(target.coordinates)[target_rows]
    result = np.zeros(corners.shape[0], dtype=np.bool_)
    item_work = corners.size + segments.shape[1] * source.ambient_dimension
    active = not budget.ticket.settled
    work_upper = segments.shape[0] * item_work
    bytes_upper = 1024 * item_work if segments.shape[0] else 0
    if active:
        if work_upper > budget.ticket.remaining_work_units:
            budget.ticket.refuse(
                budget.ticket.work_upper - budget.ticket.remaining_work_units + work_upper
            )
    owner = budget.ticket if active else scope
    with (
        budget.temporary_scope() if active else nullcontext(),
        owner.batch(work_upper, bytes_upper=0 if active else bytes_upper) as ticket,
    ):
        if active:
            budget.reserve(0, bytes_upper)
        for segment in segments:
            ticket.consume(item_work, managed_bytes=0 if active else bytes_upper)
            # Refinement retains each parent's continuous lift. Check every source
            # copy directly; nearest-image shifts lose long winding features.
            packed = np.concatenate(
                (corners.reshape((-1, source.ambient_dimension)), segment)
            )
            integers, _ = dyadic_integers(packed)
            corner_count = corners.shape[0] * corners.shape[1]
            test = integers[:corner_count].reshape(corners.shape)
            original_corners = integers[corner_count : corner_count + segment.shape[0]]
            result |= source_entity_contains(test, original_corners)
    return result


def source_entity_contains(corners: np.ndarray, source: np.ndarray, /) -> np.ndarray:
    """Return affine source support using exact arithmetic on the supplied integer bank."""
    if source.shape[0] == 1:
        return np.all(corners == source[0], axis=(1, 2))
    start, stop = source[:2]
    direction, delta = stop - start, corners - start
    if source.shape[0] == 2:
        squared = direction @ direction
        if squared == 0:
            return np.zeros(corners.shape[0], dtype=np.bool_)
        if source.shape[1] == 2:
            collinear = delta[..., 0] * direction[1] == delta[..., 1] * direction[0]
        else:
            collinear = np.all(
                np.stack(
                    (
                        delta[..., 1] * direction[2] - delta[..., 2] * direction[1],
                        delta[..., 2] * direction[0] - delta[..., 0] * direction[2],
                        delta[..., 0] * direction[1] - delta[..., 1] * direction[0],
                    ),
                    axis=-1,
                )
                == 0,
                axis=-1,
            )
        supported = np.all(collinear, axis=1)
        result = np.zeros(corners.shape[0], dtype=np.bool_)
        if np.any(supported):
            parameter = delta[supported] @ direction
            result[supported] = np.all((parameter >= 0) & (parameter <= squared), axis=1)
        return result
    second = source[2] - start
    normal = np.asarray(
        (
            direction[1] * second[2] - direction[2] * second[1],
            direction[2] * second[0] - direction[0] * second[2],
            direction[0] * second[1] - direction[1] * second[0],
        ),
        dtype=object,
    )
    aa, ab, bb = direction @ direction, direction @ second, second @ second
    denominator = aa * bb - ab * ab
    if denominator == 0:
        return np.zeros(corners.shape[0], dtype=np.bool_)
    supported = np.all(delta @ normal == 0, axis=1)
    result = np.zeros(corners.shape[0], dtype=np.bool_)
    if np.any(supported):
        pa, pb = delta[supported] @ direction, delta[supported] @ second
        first_parameter, second_parameter = pa * bb - pb * ab, pb * aa - pa * ab
        result[supported] = np.all(
            (first_parameter >= 0)
            & (second_parameter >= 0)
            & (first_parameter + second_parameter <= denominator),
            axis=1,
        )
    return result


def require_periodic_topology(
    mesh: cell_mesh_owner.CellMesh, /
) -> periodic_topology_owner.PeriodicMeshTopology:
    periodic = mesh.periodic_topology
    if periodic is None:
        raise ValueError(
            "require_periodic_topology.mesh must retain a periodic quotient descriptor; got The operation requires a bound periodic mesh topology."
        )
    return periodic


type ConstructionPointKey = tuple[tuple[int, Fraction], ...]
type PeriodicConstructionPointKey = tuple[tuple[int, tuple[int, ...], Fraction], ...]


class PeriodicConstructionOrbits:
    """Rational source-incidence point orbits, bound to one source numeric frame."""

    def __init__(self, mesh: cell_mesh_owner.CellMesh) -> None:

        periodic = require_periodic_topology(mesh)
        ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
        roots = np.asarray(periodic.vertex_representatives, dtype=np.int64)
        shifts = np.asarray(periodic.vertex_shifts, dtype=np.int64)
        fixed = np.asarray(periodic.vertex_fixed_generators, dtype=np.bool_)
        self.vertices: dict[int, tuple[int, NDArray[np.int64], NDArray[np.bool_]]] = {
            int(identifier): (
                int(ids[root]),
                np.asarray(shift, dtype=np.int64),
                np.asarray(invariant, dtype=np.bool_),
            )
            for identifier, root, shift, invariant in zip(
                ids, roots, shifts, fixed, strict=True
            )
        }
        self.orders = identification_orders(periodic.cell)
        self.rank = periodic.cell.rank
        self.source_topology_id = mesh.topology_id
        self.source_periodic_topology_id = periodic.periodic_topology_id
        self.source_numeric_version = mesh.numeric_version
        self.source_frame_id = array_tree_fingerprint(mesh.coordinates)

    def require_source(self, mesh: cell_mesh_owner.CellMesh, /) -> None:
        if (
            mesh.topology_id != self.source_topology_id
            or require_periodic_topology(mesh).periodic_topology_id
            != self.source_periodic_topology_id
            or mesh.numeric_version != self.source_numeric_version
            or array_tree_fingerprint(mesh.coordinates) != self.source_frame_id
        ):
            raise ValueError(
                "require_source.mesh must retain the bound source topology and numeric frame; got Periodic construction orbits require their rational source topology and numeric frame."
            )

    def reduced(self, shifts: np.ndarray) -> np.ndarray:

        return reduce_periodic_exponents(shifts, self.orders)

    def point(
        self,
        key: ConstructionPointKey,
        anchor: np.ndarray | None = None,
    ) -> tuple[PeriodicConstructionPointKey, np.ndarray]:
        if not key or any(identifier not in self.vertices for identifier, _ in key):
            raise ValueError(
                "point.key must retain normalized source support and a group-image anchor; got A periodic construction point must name its source vertex support."
            )
        support = [self.vertices[identifier] for identifier, _ in key]
        invariant = np.all(np.stack([fixed for _, _, fixed in support]), axis=0)
        anchors = (
            [anchor]
            if anchor is not None
            else [
                np.where(invariant, 0, shift)
                for _, shift, fixed in support
                if np.all(~fixed | invariant)
            ]
        )
        if not anchors:
            raise ValueError(
                "point.key must retain normalized source support and a group-image anchor; got Construction support has no group-image anchor."
            )
        candidates = []
        for origin in anchors:
            combined: dict[tuple[int, tuple[int, ...]], Fraction] = {}
            for (_, weight), (root, shift, fixed) in zip(key, support, strict=True):
                relative = tuple(
                    int(value)
                    for value in self.reduced(np.where(fixed, 0, shift - origin))
                )
                label = (root, relative)
                combined[label] = combined.get(label, Fraction(0)) + weight
            canonical = tuple(
                (root, shift, weight)
                for (root, shift), weight in sorted(combined.items())
                if weight
            )
            candidates.append((canonical, origin))
        return min(candidates, key=lambda value: value[0])

    def face(
        self, vertices: np.ndarray
    ) -> tuple[PeriodicConstructionPointKey, np.ndarray]:
        return self.point(
            tuple((int(identifier), Fraction(1)) for identifier in vertices)
        )

    def witness(
        self,
        source: cell_mesh_owner.CellMesh,
        target: cell_mesh_owner.CellMesh,
        stencils: dict[int, ConstructionPointKey],
        retained: tuple[PeriodicEntityIdentityBank, ...],
        *,
        allocation_prior: tuple[cell_mesh_owner.CellMesh, PeriodicVertexOrbitWitness]
        | None = None,
        budget: CoordinateEnclosureBudget,
    ) -> PeriodicVertexOrbitWitness:
        self.require_source(source)
        ids = np.asarray(target.vertex_global_ids, dtype=np.int64)
        groups: dict[PeriodicConstructionPointKey, list[tuple[int, np.ndarray]]] = {}
        for identifier in ids:
            key, anchor = self.point(stencils[int(identifier)])
            groups.setdefault(key, []).append((int(identifier), anchor))
        orbit = {}
        for members in groups.values():
            old = [
                (identifier, anchor)
                for identifier, anchor in members
                if identifier in self.vertices
            ]
            root = (
                self.vertices[old[0][0]][0]
                if old
                else min(identifier for identifier, _ in members)
            )
            reference = next(
                (anchor for identifier, anchor in members if identifier == root), None
            )
            if reference is None:
                raise ValueError(
                    "PeriodicVertexOrbitWitness.source must retain representatives of every surviving periodic orbit; got a removed retained representative"
                )
            for identifier, anchor in members:
                orbit[identifier] = (
                    root,
                    self.vertices[identifier][1]
                    if identifier in self.vertices
                    else self.reduced(anchor - reference),
                )
        representatives = np.asarray(
            [orbit[int(identifier)][0] for identifier in ids], dtype=np.int64
        )
        shifts = np.stack([orbit[int(identifier)][1] for identifier in ids]).astype(
            np.int64
        )
        return periodic_vertex_orbit_witness(
            source,
            target,
            representatives,
            shifts,
            None,
            retained_quotient_entities=retained,
            allocation_prior=allocation_prior,
            budget=budget,
        )


def simplex_entity_corners(mesh: cell_mesh_owner.CellMesh, degree: int, /) -> np.ndarray:
    connectivity = mesh.connectivity
    if isinstance(connectivity, SimplicialConnectivity):
        if 0 < degree < connectivity.dimension:
            return np.asarray(connectivity.entities[degree])
        raise ValueError(
            "simplex_entity_corners.mesh must retain simplex connectivity for the requested degree; got The periodic simplex mesh has no entities of that intermediate degree."
        )
    if isinstance(connectivity, PolyhedralConnectivity):
        if degree == 1:
            return np.asarray(connectivity.edges)
        if degree == 2:
            offsets = np.asarray(connectivity.face_vertex_offsets)
            if np.any(np.diff(offsets) != 3):
                raise TypeError(
                    "simplex_entity_corners.mesh must retain simplex connectivity for the requested degree; got Periodic simplex faces require triangular packed incidence."
                )
            return np.asarray(connectivity.face_vertex_values).reshape((-1, 3))
        raise ValueError(
            "simplex_entity_corners.mesh must retain simplex connectivity for the requested degree; got The periodic simplex mesh has no entities of that intermediate degree."
        )
    if not isinstance(connectivity, (PolygonalConnectivity, TetrahedralConnectivity)):
        raise TypeError(
            "simplex_entity_corners.mesh must retain simplex connectivity for the requested degree; got Periodic simplex operations require deterministic simplex connectivity."
        )
    if degree == 1:
        return np.asarray(connectivity.edges)
    if degree == 2 and isinstance(connectivity, TetrahedralConnectivity):
        return np.asarray(connectivity.faces)
    raise ValueError(
        "simplex_entity_corners.mesh must retain simplex connectivity for the requested degree; got The periodic simplex mesh has no entities of that intermediate degree."
    )


def periodic_exact_coordinate_bank(
    mesh: cell_mesh_owner.CellMesh,
    geometry: CellGeometrySpec | None = None,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> CoordinateSourceBank:
    """Return scientific vertex images, never lifts of rounded declared sources."""
    topology = require_periodic_topology(mesh)
    if geometry is None:
        coordinates = np.asarray(mesh.coordinates)
        roots = np.asarray(topology.vertex_representatives)
        shifts = np.asarray(topology.vertex_shifts)
        matrices, orders = exact_periodic_generators(topology.cell, budget=budget)
        reserve_periodic_exact_terms(
            coordinates.size, 512 * coordinates.size, budget=budget
        )
        result = []
        for vertex, root in enumerate(roots):
            original = np.asarray(
                [Fraction(float(value)) for value in coordinates[int(root)]],
                dtype=object,
            )
            action = np.asarray(
                exact_periodic_element(
                    matrices,
                    orders,
                    tuple(int(value) for value in shifts[vertex]),
                    budget=budget,
                ),
                dtype=object,
            )
            result.append(
                tuple(
                    periodic_exact_product(action[:-1, :-1], original, budget=budget)
                    + action[:-1, -1]
                )
            )
        return tuple(result)
    elements, routes, _ = geometry.resolve(mesh, budget=budget)
    source = prepared_coordinate_source_bank(geometry, budget=budget)
    budget.reserve(mesh.coordinates.size, 128 * mesh.coordinates.size)
    images: list[tuple[Fraction | RadicalElement, ...] | None] = [
        None
    ] * mesh.coordinates.shape[0]
    for block, element, route in zip(mesh.blocks, elements, routes, strict=True):
        for vertices, row in zip(
            np.asarray(block.vertices), np.asarray(route), strict=True
        ):
            corners = coordinate_corner_images(
                element,
                tuple(source[int(index)] for index in row if index >= 0),
                budget=budget,
            )
            if corners is None:
                raise ValueError(
                    "periodic_exact_coordinate_bank.geometry must expose reference-corner images; got no corner capability"
                )
            for vertex, point in zip(
                (int(value) for value in vertices if value >= 0), corners, strict=True
            ):
                previous = images[vertex]
                if previous is not None and previous != point:
                    raise ValueError(
                        "periodic_exact_coordinate_bank.geometry must have identical shared vertex images; got conflicting images"
                    )
                images[vertex] = point
    return tuple(
        tuple(Fraction(float(value)) for value in np.asarray(mesh.coordinates)[vertex])
        if point is None
        else point
        for vertex, point in enumerate(images)
    )


def periodic_edge_squared_lengths(
    mesh: cell_mesh_owner.CellMesh,
    /,
    *,
    geometry: CellGeometrySpec | None = None,
    budget: CoordinateEnclosureBudget | None = None,
) -> np.ndarray:
    """Return field-valued squared lengths in scientific quotient-edge order."""
    if budget is None:
        with coordinate_certificate_request(MeshCertificateLimits()) as owned:
            return periodic_edge_squared_lengths(mesh, geometry=geometry, budget=owned)
    topology = require_periodic_topology(mesh)
    edges = simplex_entity_corners(mesh, 1)[np.asarray(topology.orbit_representatives(1))]
    bank = periodic_exact_coordinate_bank(mesh, geometry, budget=budget)
    values = np.asarray(bank, dtype=object)[edges]
    reserve_algebraic_work(
        tuple(values.flat), 4 * values.size, 3 * values.size, 2, budget=budget
    )
    delta = values[:, 1] - values[:, 0]
    return np.sum(delta * delta, axis=1)


def _root_interpolation_compare(
    first: Fraction | RadicalElement,
    second: Fraction | RadicalElement,
    weight: Fraction,
    target: Fraction,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> int:
    """Compare a nonnegative weighted root sum without extending its number field."""
    reserve_algebraic_work((first, second, weight, target), 40, 20, 4, budget=budget)
    if target < 0:
        return 1
    left_weight = 1 - weight
    remainder = (
        target * target - left_weight * left_weight * first - weight * weight * second
    )
    cross_squared = 4 * left_weight * left_weight * weight * weight * first * second
    if remainder < 0:
        return 1
    difference = cross_squared - remainder * remainder
    return (difference > 0) - (difference < 0)


def _rounded_root_interpolation(
    first: Fraction | RadicalElement,
    second: Fraction | RadicalElement,
    weight: Fraction,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> float:
    """Round the algebraic root sum to nearest binary64, with ties to even."""
    maximum = float(np.finfo(np.float64).max)

    def compare(target: Fraction) -> int:
        return _root_interpolation_compare(first, second, weight, target, budget=budget)

    def value(bits: int) -> float:
        return float(np.asarray(bits, dtype=np.uint64).view(np.float64))

    def midpoint(lower: int, upper: int) -> Fraction:
        if upper == 0x7FF0000000000000:
            return Fraction(maximum) + Fraction(2) ** 970
        return (Fraction(value(lower)) + Fraction(value(upper))) / 2

    reserve_algebraic_work((first, second), 8, 8, 2, budget=budget)

    # Ordinary finite edges start within a few ulps. The ordered-bit fallback
    # also handles underflow, squared-norm overflow, and cancellation in a radical.
    def approximate_root(squared: Fraction | RadicalElement) -> float:
        if squared > Fraction(maximum):
            return math.sqrt(maximum)
        nominal = (
            squared.rounded(budget=budget)[0]
            if isinstance(squared, RadicalElement)
            else float(squared)
        )
        return math.sqrt(nominal)

    approximate = (1 - float(weight)) * approximate_root(first) + float(
        weight
    ) * approximate_root(second)
    candidate = int(np.asarray(approximate, dtype=np.float64).view(np.uint64))
    if candidate < 0x7FF0000000000000:
        low_comparison = (
            1 if candidate == 0 else compare(midpoint(candidate - 1, candidate))
        )
        high_comparison = compare(midpoint(candidate, candidate + 1))
        even = candidate % 2 == 0
        if (low_comparison > 0 or low_comparison == 0 and even) and (
            high_comparison < 0 or high_comparison == 0 and even
        ):
            return approximate
    lower, upper = 0, 0x7FF0000000000000
    while upper - lower > 1:
        middle = (lower + upper) // 2
        if compare(Fraction(value(middle))) >= 0:
            lower = middle
        else:
            upper = middle
    decision = compare(midpoint(lower, upper))
    return value(lower if decision < 0 or decision == 0 and lower % 2 == 0 else upper)


def _edge_quantile_roots(
    squared: np.ndarray, quantile: float, /, *, budget: CoordinateEnclosureBudget
) -> tuple[Fraction | RadicalElement, Fraction | RadicalElement, Fraction]:
    if not math.isfinite(quantile) or not 0 <= quantile <= 1:
        raise ValueError(
            f"periodic_edge_size_quantile.quantile must lie in [0, 1]; got {quantile!r}"
        )
    if not squared.size:
        raise ValueError(
            "periodic_edge_size_quantile.edges must be nonempty; got an empty bank"
        )
    reserve_algebraic_work(
        tuple(squared),
        squared.size * max(1, squared.size.bit_length()),
        squared.size,
        1,
        budget=budget,
    )
    ordered = sorted(squared)
    rank = float(np.float64(quantile) * np.float64(len(ordered) - 1))
    lower = math.floor(rank)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower], ordered[upper], Fraction(rank - lower)


def periodic_edge_size_quantile_compare(
    mesh: cell_mesh_owner.CellMesh,
    quantile: float,
    target: Fraction | float,
    /,
    *,
    geometry: CellGeometrySpec | None = None,
    budget: CoordinateEnclosureBudget | None = None,
) -> int:
    """Compare the algebraic linear quantile with an authored bound, without tolerance."""
    if budget is None:
        with coordinate_certificate_request(MeshCertificateLimits()) as owned:
            return periodic_edge_size_quantile_compare(
                mesh, quantile, target, geometry=geometry, budget=owned
            )
    squared = periodic_edge_squared_lengths(mesh, geometry=geometry, budget=budget)
    first, second, weight = _edge_quantile_roots(squared, quantile, budget=budget)
    return _root_interpolation_compare(
        first,
        second,
        weight,
        target if isinstance(target, Fraction) else Fraction(float(target)),
        budget=budget,
    )


def periodic_edge_size_quantile(
    mesh: cell_mesh_owner.CellMesh,
    quantiles: ArrayLike,
    /,
    *,
    geometry: CellGeometrySpec | None = None,
    budget: CoordinateEnclosureBudget | None = None,
) -> np.ndarray:
    """Return correctly rounded algebraic linear quantiles of declared edge lengths."""
    if geometry is None:
        return np.asarray(
            np.quantile(periodic_edge_size_evidence(mesh)[0], np.asarray(quantiles))
        )
    if budget is None:
        with coordinate_certificate_request(MeshCertificateLimits()) as owned:
            return periodic_edge_size_quantile(
                mesh, quantiles, geometry=geometry, budget=owned
            )
    squared = periodic_edge_squared_lengths(mesh, geometry=geometry, budget=budget)
    requested = np.asarray(quantiles, dtype=np.float64)
    result = np.empty(requested.shape, dtype=np.float64)
    for index in np.ndindex(requested.shape):
        first, second, weight = _edge_quantile_roots(
            squared, float(requested[index]), budget=budget
        )
        result[index] = _rounded_root_interpolation(first, second, weight, budget=budget)
    return result


class ExactPeriodicEdgeSizeEvidence(StrictModule, NonTrainableState):
    """Algebraic positive-root edge lengths with their binary64 execution views."""

    squared_lengths: tuple[Fraction | RadicalElement, ...] = eqx.field(static=True)
    representative_lengths: np.ndarray
    length_rounding_error_bounds: tuple[Fraction, ...] = eqx.field(static=True)
    maximum_local_edge_ratio_squared: Fraction | RadicalElement = eqx.field(static=True)
    maximum_local_edge_ratio: float = eqx.field(static=True)
    maximum_local_edge_ratio_rounding_error_bound: Fraction = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    periodic_topology_id: str = eqx.field(static=True)

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[ExactPeriodicEdgeSizeEvidence],
        value: ExactPeriodicEdgeSizeEvidence,
        /,
        *,
        mesh: cell_mesh_owner.CellMesh,
        geometry: CellGeometrySpec,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> ExactPeriodicEdgeSizeEvidence:
        """Recompute algebraic roots and rounding bounds against the retained source."""
        rebuilt = periodic_exact_edge_size_evidence(mesh, geometry, budget=budget)
        if not eqx.tree_equal(rebuilt, value, typematch=True):
            raise ValueError(
                "ExactPeriodicEdgeSizeEvidence.fields must agree with the retained source geometry; got inconsistent evidence"
            )
        return rebuilt


def _root_rounding_error_bound(
    squared: Fraction | RadicalElement,
    representative: float,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> Fraction:
    reserve_algebraic_work((squared,), 4, 4, 2, budget=budget)
    if not math.isfinite(representative):
        raise ValueError(
            f"ExactPeriodicEdgeSizeEvidence.representative_lengths must be finite; got {representative!r}"
        )
    nominal = Fraction(representative)
    if squared == nominal * nominal:
        return Fraction(0)
    previous = Fraction(float(np.nextafter(representative, -np.inf)))
    following_float = float(np.nextafter(representative, np.inf))
    spacing = nominal - previous
    if math.isfinite(following_float):
        spacing = max(spacing, Fraction(following_float) - nominal)
    return spacing / 2


def periodic_exact_edge_size_evidence(
    mesh: cell_mesh_owner.CellMesh,
    geometry: CellGeometrySpec,
    /,
    *,
    budget: CoordinateEnclosureBudget | None = None,
) -> ExactPeriodicEdgeSizeEvidence:
    """Retain field-valued squared edges and certified rounding of their positive roots."""
    if budget is None:
        with coordinate_certificate_request(MeshCertificateLimits()) as owned:
            return periodic_exact_edge_size_evidence(mesh, geometry, budget=owned)
    squared = periodic_edge_squared_lengths(mesh, geometry=geometry, budget=budget)
    lengths = np.asarray(
        [
            _rounded_root_interpolation(value, value, Fraction(0), budget=budget)
            for value in squared
        ],
        dtype=np.float64,
    )
    topology = require_periodic_topology(mesh)
    edges = simplex_entity_corners(mesh, 1)[np.asarray(topology.orbit_representatives(1))]
    orbits = np.asarray(topology.orbits(0)[0])[edges]
    reserve_algebraic_work(
        tuple(squared), 8 * squared.size, 4 * squared.size, 2, budget=budget
    )
    incident: dict[int, list[Fraction | RadicalElement]] = {}
    for edge, value in zip(orbits, squared, strict=True):
        if value <= 0:
            raise ValueError(
                "periodic_exact_edge_size_evidence.geometry must have positive quotient edge lengths; got a nonpositive squared length"
            )
        for vertex in edge:
            incident.setdefault(int(vertex), []).append(value)
    ratio = max(
        (max(values) / min(values) for values in incident.values()),
        default=Fraction(1),
    )
    growth = _rounded_root_interpolation(ratio, ratio, Fraction(0), budget=budget)
    lengths.setflags(write=False)
    return ExactPeriodicEdgeSizeEvidence(
        squared_lengths=tuple(squared),
        representative_lengths=lengths,
        length_rounding_error_bounds=tuple(
            _root_rounding_error_bound(value, float(length), budget=budget)
            for value, length in zip(squared, lengths, strict=True)
        ),
        maximum_local_edge_ratio_squared=ratio,
        maximum_local_edge_ratio=growth,
        maximum_local_edge_ratio_rounding_error_bound=_root_rounding_error_bound(
            ratio, growth, budget=budget
        ),
        geometry_id=cell_geometry_id(geometry),
        periodic_topology_id=topology.periodic_topology_id,
    )


def periodic_edge_size_evidence(
    mesh: cell_mesh_owner.CellMesh, /
) -> tuple[np.ndarray, float]:
    """Physical length/growth of each quotient edge exactly once."""
    topology = require_periodic_topology(mesh)
    representatives = np.asarray(topology.orbit_representatives(1))
    edges = simplex_entity_corners(mesh, 1)[representatives]
    roots = np.asarray(topology.vertex_representatives)
    shifts = np.asarray(topology.vertex_shifts)[edges]
    points = np.asarray(mesh.coordinates)[roots[edges]]
    if isinstance(topology.cell, periodic_cell_owner.PeriodicCell):
        delta = (
            points[:, 1]
            - points[:, 0]
            + (shifts[:, 1] - shifts[:, 0]) @ np.asarray(topology.cell.vectors)
        )
    else:
        imaged = topology.cell.apply(
            points.reshape((-1, mesh.ambient_dimension)),
            shifts.reshape((-1, topology.cell.rank)),
        ).reshape(points.shape)
        delta = imaged[:, 1] - imaged[:, 0]
    lengths = np.linalg.norm(delta, axis=1)
    vertex_orbits = np.asarray(topology.orbits(0)[0])[edges]
    growth = edge_growth_evidence(
        lengths, vertex_orbits, topology.quotient.entities(0).count
    )
    return lengths, growth


class PeriodicEmbeddingEvidence(StrictModule, NonTrainableState):
    """Interior-disjoint scientific group lifts plus their stored-coordinate error.

    ``maximum_coordinate_residual`` is an outward-rounded infinity-norm bound;
    finite isometries additionally check their declared Euclidean tolerance.
    """

    periodic_topology_id: str = eqx.field(static=True)
    image_count: int = eqx.field(static=True)
    candidate_pair_count: int = eqx.field(static=True)
    coordinate_scope: str = eqx.field(static=True)
    maximum_coordinate_residual: float = eqx.field(static=True)
    coordinate_residual_bound: float = eqx.field(static=True)
    certificate_id: str = eqx.field(static=True)
    global_embedding: GlobalEmbeddingCertificate | None = None


def _periodic_charge(
    work: int = 0,
    /,
    *,
    geometry_queries: int = 0,
    cavity_cells: int | None = None,
    budget: CoordinateEnclosureBudget,
) -> None:

    ticket = budget.ticket
    if (
        work > ticket.remaining_work_units
        or geometry_queries > ticket.remaining_geometry_queries
        or (cavity_cells is not None and cavity_cells > ticket.cavity_upper)
    ):
        ticket.refuse(
            ticket.work_units + work,
            queries_upper=ticket.geometry_queries + geometry_queries,
            bytes_upper=ticket.managed_bytes,
            cavity_upper=ticket.cavity_cells if cavity_cells is None else cavity_cells,
        )
    ticket.consume(work, geometry_queries=geometry_queries, cavity_cells=cavity_cells)


@contextmanager
def _periodic_phase(
    scope: NativeExecutionScope | None,
    budget: CoordinateEnclosureBudget,
    maximum_work: int,
    maximum_bytes: int,
    /,
) -> Iterator[CoordinateEnclosureBudget]:
    if not budget.ticket.settled:
        with budget.bound_stage(maximum_work, maximum_bytes):
            yield budget
        return
    if scope is None:
        ticket = HostBatchTicket(
            maximum_work,
            queries_upper=maximum_work,
            bytes_upper=maximum_bytes,
        )
        with budget.bind_ticket(ticket), budget.bound_stage(maximum_work, maximum_bytes):
            yield budget
        return
    remaining = scope.remaining()
    ticket = scope.batch(
        min(maximum_work, remaining.remaining_work_units),
        queries_upper=remaining.remaining_geometry_queries,
        bytes_upper=min(maximum_bytes, remaining.remaining_managed_bytes),
        cavity_upper=remaining.remaining_cavity_cells,
    )
    with (
        budget.bind_ticket(ticket),
        ticket.native_target(),
        budget.bound_stage(maximum_work, maximum_bytes),
    ):
        yield budget


def affine_periodic_source_support_boxes(
    coordinates: np.ndarray,
    roots: np.ndarray,
    shifts: np.ndarray,
    cell: PeriodicIsometryGroup,
    /,
    *,
    exact_coordinates: CoordinateSourceBank | None = None,
    budget: CoordinateEnclosureBudget,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Whole affine-cell support from coefficient-field SCI lifts, not displayed corners."""

    matrices, orders = exact_periodic_generators(cell, budget=budget)
    lower, upper = None, None
    for vertex, root in enumerate(roots):
        reserve_periodic_exact_terms(3 * cell.ambient_dimension, budget=budget)
        original = np.asarray(
            [Fraction(float(value)) for value in coordinates[int(root)]]
            if exact_coordinates is None
            else exact_coordinates[int(root)],
            dtype=object,
        )
        matrix = np.asarray(
            exact_periodic_element(
                matrices,
                orders,
                tuple(int(value) for value in shifts[vertex]),
                budget=budget,
            ),
            dtype=object,
        )
        point = (
            periodic_exact_product(matrix[:-1, :-1], original, budget=budget)
            + matrix[:-1, -1]
        )
        if lower is None:
            lower, upper = point.copy(), point.copy()
        else:
            if upper is None:
                raise RuntimeError(
                    "affine_periodic_source_support_boxes.coordinates must retain a nonempty source and paired affine support bounds; got Periodic affine support lost its paired upper bound."
                )
            lower, upper = np.minimum(lower, point), np.maximum(upper, point)
    if lower is None or upper is None:
        raise ValueError(
            "affine_periodic_source_support_boxes.coordinates must retain a nonempty source and paired affine support bounds; got Periodic affine support requires a nonempty scientific source."
        )
    if exact_coordinates is not None:
        reserve_algebraic_work(
            tuple(lower) + tuple(upper),
            4 * len(lower),
            2 * len(lower),
            1,
            budget=budget,
        )
        lower = np.asarray(
            [
                value.interval(budget=budget)[0]
                if isinstance(value, RadicalElement)
                else value
                for value in lower
            ],
            dtype=object,
        )
        upper = np.asarray(
            [
                value.interval(budget=budget)[1]
                if isinstance(value, RadicalElement)
                else value
                for value in upper
            ],
            dtype=object,
        )
    return ((lower, upper),)


@contextmanager
def _embedding_storage(
    bound: int, /, *, budget: CoordinateEnclosureBudget
) -> Iterator[CoordinateLiveStorage]:
    with budget.live_storage() as owned:
        owned.set_bound(bound)
        yield owned


def integer_cell_bounds(
    corners: np.ndarray,
    scale_bits: int,
    /,
    *,
    budget: CoordinateEnclosureBudget | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return outward float64 index boxes for coefficient-field predicates."""
    algebraic = any(
        isinstance(value, (Fraction, RadicalElement)) for value in corners.flat
    )
    if algebraic and budget is not None:
        reserve_algebraic_work(
            tuple(corners.flat), 4 * corners.size, 2 * corners.size, 1, budget=budget
        )
    lower, upper = np.min(corners, axis=1), np.max(corners, axis=1)
    denominator = 1 << scale_bits
    lower_float = np.empty(lower.shape, dtype=np.float64)
    upper_float = np.empty(upper.shape, dtype=np.float64)
    for row in range(lower.shape[0]):
        for axis in range(lower.shape[1]):
            if algebraic:
                low, high = lower[row, axis], upper[row, axis]
                low = (
                    low.interval(budget=budget)[0]
                    if isinstance(low, RadicalElement)
                    else Fraction(low)
                )
                high = (
                    high.interval(budget=budget)[1]
                    if isinstance(high, RadicalElement)
                    else Fraction(high)
                )
                lower_float[row, axis] = fraction_endpoint(
                    low / denominator, upper=False, saturate=True
                )
                upper_float[row, axis] = fraction_endpoint(
                    high / denominator, upper=True, saturate=True
                )
                continue
            lower_float[row, axis] = np.nextafter(
                float(Fraction(int(lower[row, axis]), denominator)),
                -np.inf,
            )
            upper_float[row, axis] = np.nextafter(
                float(Fraction(int(upper[row, axis]), denominator)),
                np.inf,
            )
    return lower_float, upper_float


def _image_scale_bits(
    frame: PeriodicImageFrame,
    dimension: int,
    /,
    *,
    budget: CoordinateEnclosureBudget | None = None,
) -> int:
    """One monotone power-of-two index scale covers every field-valued group image."""
    if any(isinstance(value, (Fraction, RadicalElement)) for value in frame.points.flat):

        def magnitude(values: np.ndarray) -> Fraction:
            if budget is not None:
                reserve_algebraic_work(
                    tuple(values.flat), 4 * values.size, 2 * values.size, 1, budget=budget
                )
            return max(
                (
                    max(abs(lo), abs(hi))
                    for value in values.flat
                    for lo, hi in (coefficient_interval(value, budget=budget),)
                ),
                default=Fraction(0),
            )

        point_bound = magnitude(frame.points)
        if frame.lattice is not None:
            shifts = max(
                (max(abs(value.start), abs(value.stop - 1)) for value in frame.ranges),
                default=0,
            )
            bound = point_bound + dimension * shifts * magnitude(frame.lattice)
        elif frame.group_matrices is not None:
            bank_bound = magnitude(frame.group_matrices)
            bound = max(
                point_bound,
                dimension * magnitude(frame.base_points) * bank_bound + bank_bound,
            )
        else:
            raise RuntimeError(
                "image_scale_bits.frame must retain group operands; got no lattice or group matrix"
            )
        return max(
            0, bound.numerator.bit_length() - bound.denominator.bit_length() + 2 - 500
        )
    image_bank = frame.lattice if frame.lattice is not None else frame.group_matrices
    if image_bank is None:
        raise RuntimeError(
            "image_scale_bits.frame must retain the image frame group operands; got The rational image frame must retain its group operands."
        )
    point_bits = max(
        (abs(int(value)).bit_length() for value in frame.points.flat),
        default=0,
    )
    bank_bits = max(
        (abs(int(value)).bit_length() for value in image_bank.flat),
        default=0,
    )
    if frame.lattice is not None:
        shift_bits = max(
            (
                max(abs(values.start), abs(values.stop - 1)).bit_length()
                for values in frame.ranges
            ),
            default=0,
        )
        image_bits = max(point_bits, bank_bits + shift_bits + dimension.bit_length()) + 1
    else:
        base_bits = max(
            (abs(int(value)).bit_length() for value in frame.base_points.flat),
            default=0,
        )
        image_bits = (
            max(
                point_bits,
                base_bits + bank_bits + dimension.bit_length(),
                bank_bits - 2 * frame.exponent,
            )
            + 1
        )
    return max(0, image_bits - 500)


def _embedding_image_pairs(
    frame: PeriodicImageFrame,
    connectivity: np.ndarray,
    first_bvh: PackedBVH,
    lower: np.ndarray,
    upper: np.ndarray,
    scale_bits: int,
    image_storage: int,
    tree_scratch: int,
    visit: Callable[[int], None],
    prepare_image: Callable[[], None],
    record_storage: Callable[[int], None],
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> Generator[tuple[int, int, np.ndarray, PeriodicImage], None, None]:
    for shift in frame.images():
        with _embedding_storage(image_storage, budget=budget) as owner:
            prepare_image()
            imaged = frame.image_points(shift, budget=budget)[connectivity]
            image_lower, image_upper = np.min(imaged, axis=1), np.max(imaged, axis=1)
            identity = shift == 0 if isinstance(shift, int) else not any(shift)
            second_bvh = prepare_bvh(
                *integer_cell_bounds(imaged, scale_bits, budget=budget),
                policy=BVHBuildPolicy(leaf_size=1),
                dtype=np.float64,
            )
            owner.set_bound(owner.bound - tree_scratch)
            record_storage(model_array_bytes(second_bvh))

            def retain_host(value: object) -> None:
                if not isinstance(value, tuple):
                    raise TypeError(
                        "retain_host.value must retain a tuple of NumPy host banks; got The deterministic BVH host bank must be a tuple."
                    )
                for array in value:
                    if not isinstance(array, np.ndarray):
                        raise TypeError(
                            "retain_host.value must retain a tuple of NumPy host banks; got The deterministic BVH host bank must contain NumPy arrays."
                        )
                # Payload copies fit the image scratch bound; account array metadata locally.
                owner.set_bound(owner.bound + 64 + 128 * len(value))
                record_storage(model_array_bytes(second_bvh))

            for first_rows, second_rows in bvh_overlap_pair_blocks(
                first_bvh,
                second_bvh,
                include_touching=True,
                visit=visit,
                maximum_block_pairs=256,
                retain_owner=retain_host,
            ):
                for first, second in zip(first_rows, second_rows, strict=True):
                    if identity and second <= first:
                        continue
                    visit(1)
                    if np.all(
                        (upper[first] > image_lower[second])
                        & (image_upper[second] > lower[first])
                    ):
                        yield int(first), int(second), imaged[second], shift


def _affine_periodic_embedding(
    mesh: cell_mesh_owner.CellMesh,
    topology: periodic_topology_owner.PeriodicMeshTopology,
    maximum_images: int,
    maximum_pairs: int,
    record_work: Callable[[int, int, int], None] | None,
    record_unmanaged: Callable[[int], None] | None,
    /,
    *,
    geometry: CellGeometrySpec | None = None,
    budget: CoordinateEnclosureBudget,
) -> PeriodicEmbeddingEvidence:
    dimension = mesh.ambient_dimension
    vertices = mesh.coordinates.shape[0]
    cells = sum(block.cell_count for block in mesh.blocks)
    group = topology.cell
    matrix_count = (
        1
        if isinstance(group, periodic_cell_owner.PeriodicCell)
        else int(np.prod(group.orders, dtype=object))
    )
    if matrix_count > maximum_images:
        raise PeriodicEmbeddingResourceError(
            "affine_periodic_embedding.mesh must retain disjoint lifted simplex interiors within its image and pair allowances; got "
            + (
                f"Periodic embedding requires {matrix_count} images, exceeding {maximum_images}."
            )
        )
    # IEEE float dyadic lifting multiplies at most three integer factors. 1024
    # bytes cover each resulting Python integer; indexed corners share owners.
    depth = max(1, cells.bit_length())
    tree_scratch = 4096 * cells * (dimension + 1)
    query_scratch = 256 * (depth + 1) * (512 + 128 * dimension)
    frame_storage = 1024 * (
        8 * vertices * dimension + matrix_count * (dimension + 1) ** 2
    )
    connectivity_storage = 8 * cells * (dimension + 2) if len(mesh.blocks) > 1 else 0
    storage_upper = frame_storage + tree_scratch + query_scratch + connectivity_storage
    setup_bound = 80 * (vertices * dimension + matrix_count * (dimension + 1) ** 2)
    aabb_tests = exact_tests = 0

    def visit(count: int) -> None:
        nonlocal aabb_tests
        _periodic_charge(count, geometry_queries=count, budget=budget)
        aabb_tests += count
        if record_work is not None:
            record_work(setup_bound, aabb_tests, exact_tests)

    with _embedding_storage(storage_upper, budget=budget) as source_owner:
        _periodic_charge(setup_bound, geometry_queries=0, budget=budget)
        if record_work is not None:
            record_work(setup_bound, aabb_tests, exact_tests)
        coordinates = np.asarray(mesh.coordinates)
        if len(mesh.blocks) == 1:
            connectivity = np.asarray(mesh.blocks[0].vertices)
            cell_ids = np.asarray(mesh.blocks[0].global_ids)
        else:
            copy_work = cells * (dimension + 2)
            _periodic_charge(copy_work, geometry_queries=0, budget=budget)
            setup_bound += copy_work
            connectivity, cell_ids = _periodic_simplex_rows(mesh)
        exact_coordinates = (
            None
            if geometry is None
            else periodic_exact_coordinate_bank(mesh, geometry, budget=budget)
        )
        frame = prepare_periodic_image_frame(
            coordinates,
            np.asarray(topology.vertex_representatives),
            np.asarray(topology.vertex_shifts),
            group,
            maximum_images,
            budget=budget,
            exact_coordinates=exact_coordinates,
        )
        corners = frame.points[connectivity]
        lower, upper = np.min(corners, axis=1), np.max(corners, axis=1)
        scale_bits = _image_scale_bits(
            frame, dimension, budget=budget if geometry is not None else None
        )
        policy = BVHBuildPolicy(leaf_size=1)
        build_work = 64 * cells * (dimension + depth)
        _periodic_charge(build_work, geometry_queries=0, budget=budget)
        setup_bound += build_work
        first_bvh = prepare_bvh(
            *integer_cell_bounds(
                corners, scale_bits, budget=budget if geometry is not None else None
            ),
            policy=policy,
            dtype=np.float64,
        )
        source_owner.set_bound(source_owner.bound - tree_scratch - query_scratch)
        unmanaged_peak = 0

        def record_storage(image_bytes: int) -> None:
            nonlocal unmanaged_peak
            source_bytes = model_array_bytes(first_bvh)
            unmanaged_peak = max(unmanaged_peak, source_bytes + image_bytes)
            if record_unmanaged is not None:
                record_unmanaged(unmanaged_peak)

        def prepare_image() -> None:
            nonlocal setup_bound
            _periodic_charge(build_work, geometry_queries=0, budget=budget)
            setup_bound += build_work
            if record_work is not None:
                record_work(setup_bound, aabb_tests, exact_tests)

        record_storage(0)
        candidates = 0
        pairs = _embedding_image_pairs(
            frame,
            connectivity,
            first_bvh,
            lower,
            upper,
            scale_bits,
            1024 * 4 * vertices * dimension + tree_scratch + query_scratch,
            tree_scratch,
            visit,
            prepare_image,
            record_storage,
            budget=budget,
        )
        with closing(pairs):
            for first, second, imaged_cell, shift in pairs:
                if candidates == maximum_pairs:
                    raise PeriodicEmbeddingResourceError(
                        "affine_periodic_embedding.mesh must retain disjoint lifted simplex interiors within its image and pair allowances; got Periodic embedding exhausted its candidate-pair budget."
                    )
                candidates += 1
                _periodic_charge(1, geometry_queries=1, budget=budget)
                exact_tests += 1
                if record_work is not None:
                    record_work(setup_bound, aabb_tests, exact_tests)
                if affine_mapped_contact_owner.interiors_overlap(
                    corners[first],
                    imaged_cell,
                    budget=budget if geometry is not None else None,
                ):
                    raise ValueError(
                        "affine_periodic_embedding.mesh must retain disjoint lifted simplex interiors within its image and pair allowances; got "
                        + (
                            f"Quotient cells {first} and {second} (scientific IDs {int(cell_ids[first])}, {int(cell_ids[second])}) overlap under image {shift}."
                        )
                    )
        if record_work is not None:
            record_work(setup_bound, aabb_tests, exact_tests)
        coordinate_scope = (
            "exact_dyadic_group_lift"
            if geometry is None
            else "exact_algebraic_group_lift"
        )
        return PeriodicEmbeddingEvidence(
            periodic_topology_id=topology.periodic_topology_id,
            image_count=frame.image_count,
            candidate_pair_count=candidates,
            coordinate_scope=coordinate_scope,
            maximum_coordinate_residual=frame.residual_float,
            coordinate_residual_bound=float(frame.residual_bound),
            certificate_id=canonical_fingerprint(
                {
                    "kind": "periodic-simplex-embedding-bvh",
                    "topology": topology.periodic_topology_id,
                    "coordinates": array_tree_fingerprint(coordinates),
                    "cell_ids": array_tree_fingerprint(cell_ids),
                    "images": frame.image_count,
                    "pairs": candidates,
                    "coordinate_scope": coordinate_scope,
                    **(
                        {}
                        if geometry is None
                        else {"geometry": cell_geometry_id(geometry)}
                    ),
                    # Preserve the deterministic bound payload separately from host scalar metadata.
                    "coordinate_residual": frame.residual_float,
                    "coordinate_residual_bound": frame.residual_bound,
                }
            ),
        )


class _MappedEmbeddingPreparation(StrictModule):
    geometry: CellGeometrySpec
    validity: CellValidityCertificate
    limits: MeshCertificateLimits


def _prepare_mapped_embedding(
    mesh: cell_mesh_owner.CellMesh,
    geometry: CellGeometrySpec | None,
    limits: MeshCertificateLimits | None,
    validity: CellValidityCertificate | None,
    validity_policy: CellValidityPolicy | None,
    maximum_images: int,
    maximum_pairs: int,
    budget: CoordinateEnclosureBudget,
) -> _MappedEmbeddingPreparation:
    spec = CellGeometrySpec.affine(mesh) if geometry is None else geometry
    limits_ = (
        MeshCertificateLimits(
            maximum_candidate_pairs=maximum_pairs,
            maximum_periodic_images=maximum_images,
        )
        if limits is None
        else limits
    )
    if (
        maximum_images < limits_.maximum_periodic_images
        or maximum_pairs < limits_.maximum_candidate_pairs
    ):
        limits_ = MeshCertificateLimits(
            maximum_candidate_pairs=min(maximum_pairs, limits_.maximum_candidate_pairs),
            maximum_periodic_images=min(maximum_images, limits_.maximum_periodic_images),
            maximum_ray_tests=limits_.maximum_ray_tests,
            maximum_source_samples=limits_.maximum_source_samples,
            maximum_distance_evaluations=limits_.maximum_distance_evaluations,
            maximum_subdivision_depth=limits_.maximum_subdivision_depth,
            maximum_subdivision_pieces=limits_.maximum_subdivision_pieces,
            maximum_bernstein_nodes=limits_.maximum_bernstein_nodes,
            maximum_work_units=limits_.maximum_work_units,
            maximum_scratch_bytes=limits_.maximum_scratch_bytes,
        )
    if validity is not None:
        validity.require_bound(spec, mesh=mesh)
        if (
            validity_policy is not None
            and validity.policy_id != validity_policy.policy_id
        ):
            raise ValueError(
                "prepare_mapped_embedding.mesh must retain certified validity under the authored policy; got Periodic mapped validity changes its authored policy."
            )
        if not validity.all_certified:
            raise ValueError(
                "prepare_mapped_embedding.mesh must retain certified validity under the authored policy; got Periodic mapped validity is unresolved."
            )
    else:
        validity = certify_cell_geometry_validity(
            spec, mesh=mesh, policy=validity_policy, budget=budget
        )
    return _MappedEmbeddingPreparation(spec, validity, limits_)


def _require_mapped_embedding(
    certificate: GlobalEmbeddingCertificate,
) -> GlobalEmbeddingCertificate:
    if certificate.status != "certified":
        resource_findings = tuple(
            finding for finding in certificate.findings if finding.resource is not None
        )
        if resource_findings:
            requested: list[tuple[str, int]] = []
            achieved: list[tuple[str, int]] = []
            for finding in resource_findings:
                prefix = (
                    f"certificate:{certificate.certificate_id}:finding:{finding.finding_id}"
                    f":{finding.check}:resource:{finding.resource}"
                )
                requested.extend(
                    (f"{prefix}:{key}", value) for key, value in finding.requested
                )
                achieved.extend(
                    (f"{prefix}:{key}", value) for key, value in finding.achieved
                )
            raise MeshingFailure(
                MeshingFailureCategory.RESOURCE_EXHAUSTED,
                "require_mapped_embedding.certificate must certify disjoint quotient images within its resource allowance; got Periodic mapped embedding exhausted its coordinate proof allowance.",
                stage="periodic_embedding",
                entity_ids=tuple(
                    sorted(
                        {
                            identifier
                            for finding in resource_findings
                            for identifier in finding.entity_ids
                        }
                    )
                ),
                requested=tuple(requested),
                achieved=tuple(achieved),
            )
        raise ValueError(
            "require_mapped_embedding.certificate must certify disjoint quotient images within its resource allowance; got "
            + (
                f"Periodic mapped embedding {certificate.status}: "
                + ", ".join((value.check for value in certificate.findings))
            )
        )
    return certificate


def _closed_affine_quotient_embedding(
    mesh: cell_mesh_owner.CellMesh,
    topology: periodic_topology_owner.PeriodicMeshTopology,
    preparation: _MappedEmbeddingPreparation,
    maximum_images: int,
    maximum_pairs: int,
    budget: CoordinateEnclosureBudget,
) -> PeriodicEmbeddingEvidence | None:
    """Combine source validity, closed quotient incidence, and field-arithmetic open contacts."""
    if (
        not preparation.validity.all_certified
        or any(
            block.cell_kind != _SIMPLEX_KINDS.get(mesh.ambient_dimension)
            for block in mesh.blocks
        )
        or (
            isinstance(topology.cell, PeriodicIsometryGroup)
            and any(order == 0 for order in topology.cell.orders)
        )
    ):
        return None
    algebraic = isinstance(
        preparation.geometry.exact_source, ExactAlgebraicCellGeometrySource
    )
    if algebraic:
        spec = preparation.geometry
        elements, routes, _ = spec.resolve(mesh, budget=budget)
        bank = prepared_coordinate_source_bank(spec, budget=budget)
        for element, route in zip(elements, routes, strict=True):
            for row in np.asarray(route):
                expressions = coordinate_expressions(
                    element,
                    tuple(bank[int(index)] for index in row if index >= 0),
                    budget=budget,
                )
                if (
                    expressions is None
                    or not affine_mapped_contact_owner.affine_expressions(expressions)
                ):
                    return None
    facets = topology.quotient.entities(mesh.topological_dimension - 1)
    with budget.temporary_scope():
        budget.reserve(facets.count, facets.count)
        if np.any(np.asarray(facets.subset("boundary").mask)):
            return None
    # PeriodicMeshTopology has proved at most two sides and opposite incidence
    # orientation on each quotient facet. With no boundary, a positive affine
    # local star has positive integer degree. Nonincident closed contact would
    # force open-interior overlap in a neighborhood of that contact.
    affine = _affine_periodic_embedding(
        mesh,
        topology,
        min(maximum_images, preparation.limits.maximum_periodic_images),
        min(maximum_pairs, preparation.limits.maximum_candidate_pairs),
        None,
        None,
        budget=budget,
        geometry=preparation.geometry if algebraic else None,
    )
    return PeriodicEmbeddingEvidence(
        periodic_topology_id=affine.periodic_topology_id,
        image_count=affine.image_count,
        candidate_pair_count=affine.candidate_pair_count,
        coordinate_scope=(
            "exact_algebraic_closed_quotient_lift"
            if algebraic
            else "exact_dyadic_closed_quotient_lift"
        ),
        maximum_coordinate_residual=affine.maximum_coordinate_residual,
        coordinate_residual_bound=affine.coordinate_residual_bound,
        certificate_id=canonical_fingerprint(
            {
                "kind": "periodic-closed-affine-quotient-embedding",
                "interior_embedding": affine.certificate_id,
                "source_validity": preparation.validity.certificate_id,
                "closed_oriented_quotient": topology.periodic_topology_id,
            }
        ),
    )


def _mapped_periodic_embedding(
    mesh: cell_mesh_owner.CellMesh,
    topology: periodic_topology_owner.PeriodicMeshTopology,
    geometry: CellGeometrySpec | None,
    limits: MeshCertificateLimits | None,
    validity: CellValidityCertificate | None,
    validity_policy: CellValidityPolicy | None,
    maximum_images: int,
    maximum_pairs: int,
    budget: CoordinateEnclosureBudget,
) -> PeriodicEmbeddingEvidence:
    request_limits = MeshCertificateLimits() if limits is None else limits
    try:
        with (
            budget.measurement() as measurement,
            budget.activate(),
            budget.bound_stage(
                request_limits.maximum_work_units,
                request_limits.maximum_scratch_bytes,
            ),
        ):
            preparation = _prepare_mapped_embedding(
                mesh,
                geometry,
                limits,
                validity,
                validity_policy,
                maximum_images,
                maximum_pairs,
                budget,
            )
            if (
                geometry is None
                and len(mesh.blocks) > 1
                or (
                    geometry is not None
                    and isinstance(
                        geometry.exact_source, ExactAlgebraicCellGeometrySource
                    )
                )
            ):
                closed_affine = _closed_affine_quotient_embedding(
                    mesh, topology, preparation, maximum_images, maximum_pairs, budget
                )
                if closed_affine is not None:
                    return closed_affine
            spec = preparation.geometry
            certificate = certify_global_embedding(
                mesh, spec, preparation.validity, limits=preparation.limits, budget=budget
            )
            _require_mapped_embedding(certificate)
            _, _, executed = spec.resolve(mesh, budget=budget)
            source_bank = prepared_coordinate_source_bank(spec, budget=budget)
            if isinstance(spec.exact_source, ExactAlgebraicCellGeometrySource):
                reserve_algebraic_work(
                    source_bank,
                    4 * np.asarray(executed).size,
                    2 * np.asarray(executed).size,
                    1,
                    budget=budget,
                )
            coefficient_error = max(
                (
                    abs(Fraction(float(actual)) - exact)
                    for row, source_row in zip(
                        np.asarray(executed), source_bank, strict=True
                    )
                    for actual, exact in zip(row, source_row, strict=True)
                ),
                default=Fraction(0),
            )
            if isinstance(coefficient_error, RadicalElement):
                coefficient_error = coefficient_error.interval(budget=budget)[1]
            residual = fraction_endpoint(coefficient_error, upper=True, saturate=True)
            return PeriodicEmbeddingEvidence(
                periodic_topology_id=topology.periodic_topology_id,
                image_count=certificate.periodic_image_count,
                candidate_pair_count=certificate.candidate_pair_count,
                coordinate_scope="exact_source_group_images",
                maximum_coordinate_residual=residual,
                coordinate_residual_bound=residual,
                certificate_id=certificate.certificate_id,
                global_embedding=certificate,
            )
    except CoordinateEnclosureResourceError as error:
        prefix = f"periodic:coordinate_source:resource:{error.resource}"
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "mapped_periodic_embedding.mesh must fit the coordinate-source preparation allowance; got Periodic source preparation exhausted its coordinate proof allowance.",
            stage="periodic_embedding",
            requested=(
                (f"{prefix}:limit", error.limit),
                (f"{prefix}:requested", error.requested),
            ),
            achieved=(
                (f"{prefix}:completed", error.completed),
                (f"{prefix}:source_expression_work_units", measurement.work_units),
                (f"{prefix}:source_expression_peak_bytes", measurement.peak_bytes_upper),
                (
                    f"{prefix}:source_expression_native_charged_work_units",
                    measurement.native_charged_work_units,
                ),
                (
                    f"{prefix}:source_expression_retained_basis_bytes_at_catch",
                    budget.retained_basis_bytes,
                ),
                (
                    f"{prefix}:source_expression_temporary_bytes_upper_at_catch",
                    budget.temporary_bytes_upper,
                ),
            ),
        ) from error


def _certify_periodic_embedding(
    mesh: cell_mesh_owner.CellMesh,
    /,
    *,
    geometry: CellGeometrySpec | None = None,
    limits: MeshCertificateLimits | None = None,
    validity: CellValidityCertificate | None = None,
    validity_policy: CellValidityPolicy | None = None,
    maximum_images: int = 100000,
    maximum_pairs: int = 5000000,
    record_work: Callable[[int, int, int], None] | None = None,
    record_unmanaged: Callable[[int], None] | None = None,
    budget: CoordinateEnclosureBudget,
) -> PeriodicEmbeddingEvidence:
    """Certify group-lift interior disjointness or the recorded closed-contact proof.

    Mapped/mixed cells use the continuous global certificate. Default-affine
    multi-block simplices reuse coefficient-field group-lift predicates only after proving
    full-source positive validity and closed oriented quotient incidence.
    Nonclosed multi-block quotients retain the mapped closed-contact proof.
    The default-affine single-block control records interior-disjoint lifts.
    """

    topology = require_periodic_topology(mesh)
    if (
        geometry is not None
        or len(mesh.blocks) != 1
        or mesh.blocks[0].cell_kind != _SIMPLEX_KINDS.get(mesh.ambient_dimension)
        or (
            isinstance(topology.cell, PeriodicIsometryGroup)
            and any(order == 0 for order in topology.cell.orders)
        )
    ):
        return _mapped_periodic_embedding(
            mesh,
            topology,
            geometry,
            limits,
            validity,
            validity_policy,
            maximum_images,
            maximum_pairs,
            budget,
        )
    if maximum_images <= 0 or maximum_pairs <= 0:
        raise ValueError(
            "certify_periodic_embedding.mesh must retain full-dimensional simplices and positive image and pair allowances; got Image and candidate-pair budgets must be positive."
        )
    return _affine_periodic_embedding(
        mesh,
        topology,
        maximum_images,
        maximum_pairs,
        record_work,
        record_unmanaged,
        budget=budget,
    )


@checked
def certify_periodic_embedding(
    mesh: cell_mesh_owner.CellMesh,
    /,
    *,
    geometry: CellGeometrySpec | None = None,
    limits: MeshCertificateLimits | None = None,
    validity: CellValidityCertificate | None = None,
    validity_policy: CellValidityPolicy | None = None,
    maximum_images: int = 100_000,
    maximum_pairs: int = 5_000_000,
    record_work: Callable[[int, int, int], None] | None = None,
    record_unmanaged: Callable[[int], None] | None = None,
    scope: NativeExecutionScope | None = None,
    budget: CoordinateEnclosureBudget | None = None,
) -> PeriodicEmbeddingEvidence:
    """Certify quotient embedding under the certificate request allowance."""
    request_limits = MeshCertificateLimits() if limits is None else limits
    if budget is None:
        initial = HostBatchTicket(0)
        budget = CoordinateEnclosureBudget(
            initial,
            maximum_work_units=request_limits.maximum_work_units,
            maximum_memory_bytes=request_limits.maximum_scratch_bytes,
        )
        initial.settle()
    with _periodic_phase(
        scope,
        budget,
        request_limits.maximum_work_units,
        request_limits.maximum_scratch_bytes,
    ):
        return _certify_periodic_embedding(
            mesh,
            geometry=geometry,
            limits=limits,
            validity=validity,
            validity_policy=validity_policy,
            maximum_images=maximum_images,
            maximum_pairs=maximum_pairs,
            record_work=record_work,
            record_unmanaged=record_unmanaged,
            budget=budget,
        )


_SIMPLEX_KINDS = {2: "triangle", 3: "tetrahedron"}
_REFINEMENT_CELL_WORK = 64


def _periodic_simplex_rows(
    mesh: cell_mesh_owner.CellMesh, /
) -> tuple[np.ndarray, np.ndarray]:
    """cell rows and SCI IDs in the same all required block order as geometry."""
    kind = _SIMPLEX_KINDS.get(mesh.topological_dimension)
    if kind is None or any(block.cell_kind != kind for block in mesh.blocks):
        raise ValueError(
            "periodic_simplex_rows.mesh must retain all simplex blocks in scientific row order; got Periodic simplex transactions require all blocks of one simplex family."
        )
    return (
        np.concatenate(tuple(np.asarray(block.vertices) for block in mesh.blocks)),
        np.concatenate(tuple(np.asarray(block.global_ids) for block in mesh.blocks)),
    )


def periodic_cell_from_constraints(
    constraints: Sequence[PeriodicConstraint],
    /,
    *,
    origin: ArrayLike | None = None,
) -> periodic_cell_owner.PeriodicCell:
    """Compile translational periodic constraints into their lattice.

    Every constraint must be a pure translation (identity linear part within
    its declared tolerance); the translations become the lattice vectors, one
    per ambient axis, in the given order. Rotational or reflecting pairings are
    boundary isometries, not translational torus topology, and are refused.
    """

    values = tuple(constraints)
    if not values or not all(isinstance(item, PeriodicConstraint) for item in values):
        raise TypeError(
            "periodic_cell_from_constraints.constraints must provide translational identifications spanning each ambient axis; got constraints must be a nonempty sequence of PeriodicConstraint."
        )
    vectors = []
    for index, constraint in enumerate(values):
        transform = np.asarray(constraint.transform, dtype=np.float64)
        dimension = transform.shape[0] - 1
        deviation = np.max(np.abs(transform[:dimension, :dimension] - np.eye(dimension)))
        if deviation > constraint.tolerance:
            raise ValueError(
                "periodic_cell_from_constraints.constraints must provide translational identifications spanning each ambient axis; got "
                + (
                    f"constraints[{index}] is not a translation; boundary isometries do not define a translational periodic lattice."
                )
            )
        vectors.append(transform[:dimension, dimension])
    matrix = np.stack(vectors)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            "periodic_cell_from_constraints.constraints must provide translational identifications spanning each ambient axis; got A periodic lattice needs one translational constraint per ambient axis."
        )
    return periodic_cell_owner.PeriodicCell(matrix, origin=origin)


class _Orbits:
    """Union-find over input points with the lattice shift to each root."""

    def __init__(self, count: int, dimension: int, /) -> None:
        self.parent = np.arange(count, dtype=np.int64)
        self.shift = np.zeros((count, dimension), dtype=np.int64)

    def find(self, index: int, /) -> tuple[int, np.ndarray]:
        shift = np.zeros_like(self.shift[0])
        while self.parent[index] != index:
            shift = shift + self.shift[index]
            index = int(self.parent[index])
        return index, shift


class PeriodicPointOrbits(StrictModule, NonTrainableState):
    """Lattice orbits of seed and feature points on a flat torus.

    Input points that coincide modulo the lattice within ``tolerance`` (for
    example the copies of one feature on opposite seams) form one orbit.
    ``points`` retains the authored input; ``representatives`` holds the first
    input point of every orbit, in order of first appearance; input point ``i``
    equals ``representatives[orbits[i]] + shifts[i] @ cell.vectors`` up to
    ``tolerance``. Copies farther apart than ``tolerance`` from their orbit
    representative are refused as ambiguous.
    """

    cell: periodic_cell_owner.PeriodicCell
    points: np.ndarray
    representatives: np.ndarray
    orbits: np.ndarray
    shifts: np.ndarray
    tolerance: float = eqx.field(static=True)
    orbits_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        points: ArrayLike,
        cell: periodic_cell_owner.PeriodicCell,
        /,
        *,
        tolerance: float = 0.0,
    ) -> None:
        array = np.asarray(points, dtype=np.float64)
        if (
            array.ndim != 2
            or array.shape[0] == 0
            or array.shape[1] != cell.ambient_dimension
        ):
            raise ValueError(
                "PeriodicPointOrbits.points must use finite ambient-dimension points, nonnegative tolerance, and a full-rank periodic lattice; got points must have shape (n > 0, ambient dimension)."
            )
        if not np.all(np.isfinite(array)):
            raise ValueError(
                "PeriodicPointOrbits.points must use finite ambient-dimension points, nonnegative tolerance, and a full-rank periodic lattice; got points must be finite."
            )
        threshold = float(tolerance)
        if not np.isfinite(threshold) or threshold < 0.0:
            raise ValueError(
                "PeriodicPointOrbits.tolerance must use finite ambient-dimension points, nonnegative tolerance, and a full-rank periodic lattice; got tolerance must be finite and nonnegative."
            )
        if cell.rank != cell.ambient_dimension or not cell.fully_periodic:
            raise ValueError(
                "PeriodicPointOrbits.cell must use finite ambient-dimension points, nonnegative tolerance, and a full-rank periodic lattice; got Point orbits require a fully periodic full-rank lattice."
            )
        roots, shifts = _point_orbits(array, cell, threshold)
        _, first, orbit = np.unique(roots, return_index=True, return_inverse=True)
        # Orbits are numbered by first appearance; that input point represents
        # its orbit and every shift is taken relative to it.
        order = np.argsort(first, kind="stable")
        rank = np.empty_like(order)
        rank[order] = np.arange(order.size)
        leaders = first[orbit]
        self.cell = cell
        self.points = _frozen(array.copy())
        self.representatives = _frozen(array[first[order]])
        self.orbits = _frozen(rank[orbit].astype(np.int32))
        self.shifts = _frozen((shifts - shifts[leaders]).astype(np.int32))
        self.tolerance = threshold
        self.orbits_id = canonical_fingerprint(
            {
                "kind": "periodic-point-orbits",
                "cell": cell.cell_id,
                "tolerance": threshold,
                "arrays": array_tree_fingerprint(
                    {"points": array, "orbits": self.orbits, "shifts": self.shifts}
                ),
            }
        )

    @property
    def orbit_count(self) -> int:
        return self.representatives.shape[0]


def _frozen(array: np.ndarray, /) -> np.ndarray:
    result = np.ascontiguousarray(array)
    result.setflags(write=False)
    return result


def _point_orbits(
    points: np.ndarray, cell: periodic_cell_owner.PeriodicCell, tolerance: float, /
) -> tuple[np.ndarray, np.ndarray]:
    """Return each point's orbit root and its lattice shift from that root."""

    vectors = np.asarray(cell.vectors, dtype=np.float64)
    inverse = np.asarray(cell.inverse_vectors, dtype=np.float64)
    count, dimension = points.shape
    fractional = (points - np.asarray(cell.origin, dtype=np.float64)) @ inverse
    wrapped = fractional - np.floor(fractional)
    # Bins at least as wide as the tolerance in lattice units, so equivalent
    # points fall into the same or a neighboring (periodically wrapped) bin.
    reach = tolerance * float(np.max(np.linalg.norm(inverse, axis=0)))
    bins = int(min(1024, max(1, np.floor(1.0 / max(2.0 * reach, 2.0**-20)))))
    keys = np.minimum(np.floor(wrapped * bins).astype(np.int64), bins - 1)
    table: dict[tuple[int, ...], list[int]] = {}
    for index, key in enumerate(map(tuple, keys.tolist())):
        table.setdefault(key, []).append(index)
    offsets = np.stack(
        np.meshgrid(*([np.arange(-1, 2)] * dimension), indexing="ij"), axis=-1
    ).reshape((-1, dimension))
    forest = _Orbits(count, dimension)
    for index in range(count):
        neighbors = {
            tuple(((keys[index] + offset) % bins).tolist()) for offset in offsets
        }
        for key in sorted(neighbors):
            for other in table.get(key, ()):
                if other >= index:
                    continue
                shift = np.rint(fractional[index] - fractional[other]).astype(np.int64)
                residual = points[index] - points[other] - shift @ vectors
                if np.linalg.norm(residual) > tolerance:
                    continue
                root, root_shift = forest.find(other)
                own_root, own_shift = forest.find(index)
                if root == own_root:
                    continue
                # index = other + shift; other = root + root_shift.
                forest.parent[own_root] = root
                forest.shift[own_root] = root_shift + shift - own_shift
    roots = np.empty((count,), dtype=np.int64)
    shifts = np.empty((count, dimension), dtype=np.int64)
    for index in range(count):
        root, shift = forest.find(index)
        roots[index] = root
        shifts[index] = shift
    deviation = np.linalg.norm(points - points[roots] - shifts @ vectors, axis=1)
    if np.any(deviation > tolerance):
        raise ValueError(
            "point_orbits.points must place each copy within tolerance of its orbit representative; got Periodic point orbits are ambiguous: a copy lies farther than the tolerance from its orbit representative."
        )
    return roots, shifts


def _lifted_exact_point_bank(
    coordinates: CoordinateSourceBank,
    lifted: np.ndarray,
    cell: periodic_cell_owner.PeriodicCell,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> CoordinateSourceBank:
    lattice = np.asarray(
        [[Fraction(float(value)) for value in row] for row in np.asarray(cell.vectors)],
        dtype=object,
    )
    bank = np.asarray(coordinates, dtype=object)
    reserve_algebraic_work(
        coordinates, 4 * lifted.size, 4 * lifted.size, 2, budget=budget
    )
    images = bank[lifted[:, 0]] + periodic_exact_product(
        lifted[:, 1:].astype(object), lattice, budget=budget
    )
    return tuple(tuple(row) for row in images)


def periodic_lifted_exact_coordinates(
    coordinates: CoordinateSourceBank,
    simplices: ArrayLike,
    shifts: ArrayLike,
    cell: periodic_cell_owner.PeriodicCell,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> CoordinateSourceBank:
    """Return field-valued vertices in the publication's declared lifted-row order."""
    cells = np.asarray(simplices, dtype=np.int64)
    actions = np.asarray(shifts, dtype=np.int64)
    if (
        cells.ndim != 2
        or actions.shape != (*cells.shape, cell.rank)
        or any(len(row) != cell.ambient_dimension for row in coordinates)
    ):
        raise ValueError(
            "periodic_lifted_exact_coordinates.inputs must align source lifts and ambient dimensions; got incompatible banks"
        )
    reserve_periodic_exact_terms(
        cells.size * (cell.rank + 1),
        32 * cells.size * (cell.rank + 1),
        budget=budget,
    )
    lifted = np.unique(
        np.concatenate(
            (cells.reshape((-1, 1)), actions.reshape((-1, cell.rank))), axis=1
        ),
        axis=0,
    )
    return _lifted_exact_point_bank(coordinates, lifted, cell, budget=budget)


def _publish_periodic_simplices(
    points: ArrayLike,
    simplices: ArrayLike,
    shifts: ArrayLike,
    cell: periodic_cell_owner.PeriodicCell,
    /,
    *,
    block_name: str = "periodic",
    exact_coordinates: CoordinateSourceBank | None = None,
    budget: CoordinateEnclosureBudget,
) -> cell_mesh_owner.CellMesh:
    """Publish quotient simplices as a lifted ``CellMesh`` with its quotient topology.

    ``simplices`` (T, d + 1) name representative ``points`` and ``shifts``
    (T, d + 1, d) the lattice image of each corner in the simplex's local lift.
    Every distinct (representative, shift) corner becomes one lifted vertex;
    each representative's quotient representative is its zero-shift image when
    present, else its lexicographically smallest image.
    """

    point_array = np.asarray(points, dtype=np.float64)
    cells = np.asarray(simplices, dtype=np.int64)
    corner_shifts = np.asarray(shifts, dtype=np.int64)
    dimension = point_array.shape[1]
    if (
        dimension not in _SIMPLEX_KINDS
        or cells.ndim != 2
        or cells.shape[1] != dimension + 1
        or corner_shifts.shape != (*cells.shape, dimension)
    ):
        raise ValueError(
            "publish_periodic_simplices.points must retain dimension-aligned simplices, shifts, and used representatives; got Periodic simplices must be (T, d + 1) with (T, d + 1, d) shifts."
        )
    corners = np.concatenate(
        (cells.reshape((-1, 1)), corner_shifts.reshape((-1, dimension))), axis=1
    )
    lifted, inverse = np.unique(corners, axis=0, return_inverse=True)
    representatives = lifted[:, 0]
    lifted_shifts = lifted[:, 1:]
    if np.unique(representatives).size != point_array.shape[0]:
        raise ValueError(
            "publish_periodic_simplices.points must retain dimension-aligned simplices, shifts, and used representatives; got Every representative point must be a simplex corner."
        )
    zero = np.all(lifted_shifts == 0, axis=1)
    # np.unique sorts rows, so the first row per representative is its
    # smallest shift; a zero-shift image takes precedence.
    first = np.searchsorted(representatives, np.arange(point_array.shape[0]))
    zero_rows = np.flatnonzero(zero)
    reference = first.copy()
    reference[representatives[zero_rows]] = zero_rows
    vertex_representatives = reference[representatives]
    vertex_shifts = lifted_shifts - lifted_shifts[vertex_representatives]
    if exact_coordinates is None:
        coordinates = point_array[representatives] + lifted_shifts.astype(
            np.float64
        ) @ np.asarray(cell.vectors, dtype=np.float64)
    else:
        if len(exact_coordinates) != point_array.shape[0] or any(
            len(row) != dimension for row in exact_coordinates
        ):
            raise ValueError(
                "publish_periodic_simplices.exact_coordinates must match the representative bank; got incompatible dimensions"
            )
        reserve_algebraic_work(
            exact_coordinates, 2 * point_array.size, point_array.size, 1, budget=budget
        )
        if any(
            (
                exact.rounded(budget=budget)[0]
                if isinstance(exact, RadicalElement)
                else float(exact)
            )
            != point_array[vertex, axis]
            for vertex, row in enumerate(exact_coordinates)
            for axis, exact in enumerate(row)
        ):
            raise ValueError(
                "publish_periodic_simplices.points must correctly round source coordinates; got differing representative bytes"
            )
        exact_lifts = _lifted_exact_point_bank(
            exact_coordinates, lifted, cell, budget=budget
        )
        coordinates = np.asarray(
            [
                [
                    value.rounded(budget=budget)[0]
                    if isinstance(value, RadicalElement)
                    else float(value)
                    for value in row
                ]
                for row in exact_lifts
            ],
            dtype=np.float64,
        )
    block = reference_cell_types_owner.CellBlock(
        block_name,
        _SIMPLEX_KINDS[dimension],
        inverse.reshape(cells.shape).astype(np.int32),
    )
    carrier = cell_mesh_owner.CellMesh(coordinates, (block,))
    topology = periodic_topology_owner.PeriodicMeshTopology(
        carrier, cell, vertex_representatives, vertex_shifts, budget=budget
    )
    return cell_mesh_owner.CellMesh(coordinates, (block,), periodic_topology=topology)


def publish_periodic_simplices(
    points: ArrayLike,
    simplices: ArrayLike,
    shifts: ArrayLike,
    cell: periodic_cell_owner.PeriodicCell,
    /,
    *,
    block_name: str = "periodic",
    scope: NativeExecutionScope | None = None,
    budget: CoordinateEnclosureBudget | None = None,
    exact_coordinates: CoordinateSourceBank | None = None,
) -> cell_mesh_owner.CellMesh:
    """Publish quotient simplices using explicit or input-derived rational preparation."""
    if budget is None and exact_coordinates is not None:
        with coordinate_certificate_request(MeshCertificateLimits()) as owned:
            return publish_periodic_simplices(
                points,
                simplices,
                shifts,
                cell,
                block_name=block_name,
                scope=scope,
                budget=owned,
                exact_coordinates=exact_coordinates,
            )
    if budget is None:
        cells = np.asarray(simplices, dtype=np.int64)
        actions = np.asarray(shifts, dtype=np.int64)
        work_upper, bytes_upper, queries_upper = periodic_preparation_bounds(
            cell, cells.size, actions, stabilizers=True
        )
        ticket = HostBatchTicket(
            work_upper, queries_upper=queries_upper, bytes_upper=bytes_upper
        )
        budget = CoordinateEnclosureBudget(
            ticket,
            maximum_work_units=work_upper,
            maximum_memory_bytes=bytes_upper,
        )
        if scope is None:
            with ticket, budget.activate():
                return _publish_periodic_simplices(
                    points,
                    cells,
                    actions,
                    cell,
                    block_name=block_name,
                    budget=budget,
                    exact_coordinates=exact_coordinates,
                )
        ticket.settle()
    with _periodic_phase(
        scope, budget, budget.maximum_work_units, budget.maximum_memory_bytes
    ):
        return _publish_periodic_simplices(
            points,
            simplices,
            shifts,
            cell,
            block_name=block_name,
            budget=budget,
            exact_coordinates=exact_coordinates,
        )


class PeriodicQuotientEvidence(StrictModule, NonTrainableState):
    """Quotient acceptance of a published periodic mesh.

    Counts are quotient entities by degree; the total measure integrates each
    quotient top cell once in its local lift and is compared with the lattice
    cell measure.
    """

    quotient_counts: tuple[int, ...] = eqx.field(static=True)
    euler_characteristic: int = eqx.field(static=True)
    boundary_facet_count: int = eqx.field(static=True)
    total_measure: float = eqx.field(static=True)
    lattice_measure: float | None = eqx.field(static=True)
    relative_coverage_defect: float | None = eqx.field(static=True)
    all_cells_valid: bool = eqx.field(static=True)
    periodic_topology_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        mesh: cell_mesh_owner.CellMesh,
        /,
        *,
        geometry: CellGeometrySpec | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> None:
        periodic = mesh.periodic_topology
        if periodic is None:
            raise ValueError(
                "PeriodicQuotientEvidence.mesh must retain a periodic mesh descriptor; got Quotient evidence requires a periodic CellMesh."
            )
        report = periodic_measures_owner.periodic_orbit_measures(
            mesh, geometry=geometry, budget=budget
        )
        defect = report.relative_coverage_defect
        facets = periodic.quotient.entities(mesh.topological_dimension - 1)
        self.quotient_counts = tuple(
            entities.count for entities in periodic.quotient.entity_sets
        )
        self.euler_characteristic = periodic.euler_characteristic
        self.boundary_facet_count = int(
            np.count_nonzero(np.asarray(facets.subset("boundary").mask))
        )
        self.total_measure = float(report.total_measure)
        self.lattice_measure = report.lattice_measure
        self.relative_coverage_defect = defect
        self.all_cells_valid = bool(np.all(np.asarray(report.valid)))
        self.periodic_topology_id = periodic.periodic_topology_id

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[PeriodicQuotientEvidence],
        value: PeriodicQuotientEvidence,
        /,
        *,
        mesh: cell_mesh_owner.CellMesh,
        geometry: CellGeometrySpec | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> PeriodicQuotientEvidence:
        """Recompute stored quotient evidence against its retained mesh."""
        topology = require_periodic_topology(mesh)
        if value.periodic_topology_id != topology.periodic_topology_id:
            raise ValueError(
                f"PeriodicQuotientEvidence.periodic_topology_id must bind the retained mesh; got {value.periodic_topology_id!r}."
            )
        restored = cls(mesh, geometry=geometry, budget=budget)
        if not eqx.tree_equal(restored, value, typematch=True):
            raise ValueError(
                "PeriodicQuotientEvidence.fields must agree with recomputed quotient evidence; "
                "got inconsistent stored fields."
            )
        return restored


class PeriodicMeshConstruction(StrictModule, NonTrainableState):
    """Periodic Delaunay mesh of point orbits with its certificates."""

    mesh: cell_mesh_owner.CellMesh
    orbits: PeriodicPointOrbits
    triangulation: PeriodicTriangulationEvidence
    quotient: PeriodicQuotientEvidence


@checked
def periodic_delaunay_mesh(
    orbits: PeriodicPointOrbits,
    /,
    *,
    initial_margin: float | None = None,
    maximum_images: int | None = None,
    maximum_simplices: int | None = None,
    block_name: str = "periodic",
    scope: NativeExecutionScope | None = None,
    budget: CoordinateEnclosureBudget | None = None,
) -> PeriodicMeshConstruction:
    """Construct and publish the periodic Delaunay mesh of point orbits.

    Budget refusals raise ``PeriodicImageBudgetError`` with the refused round's
    evidence; the published mesh carries its quotient topology and the
    construction reports the certified image neighborhood and quotient
    acceptance.
    """

    context = (
        nullcontext()
        if budget is None
        else _periodic_phase(
            scope, budget, budget.maximum_work_units, budget.maximum_memory_bytes
        )
    )
    with context:
        triangulation = PeriodicDelaunayTriangulation(
            orbits.representatives,
            orbits.cell,
            initial_margin=initial_margin,
            maximum_images=maximum_images,
            maximum_simplices=maximum_simplices,
        )
        mesh = publish_periodic_simplices(
            triangulation.points,
            triangulation.simplices,
            triangulation.simplex_shifts,
            orbits.cell,
            block_name=block_name,
            scope=scope,
            budget=budget,
        )
        return PeriodicMeshConstruction(
            mesh=mesh,
            orbits=orbits,
            triangulation=triangulation.evidence,
            quotient=PeriodicQuotientEvidence(mesh),
        )


def quotient_simplices(
    mesh: cell_mesh_owner.CellMesh, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray, periodic_cell_owner.PeriodicCell, str]:
    """Representative points, quotient simplices and corner shifts of a mesh."""

    periodic = mesh.periodic_topology
    if periodic is None:
        raise ValueError(
            "quotient_simplices.mesh must retain a translational periodic quotient; got Periodic refinement requires a CellMesh with periodic topology."
        )
    vertices, _ = _periodic_simplex_rows(mesh)
    if not isinstance(periodic.cell, periodic_cell_owner.PeriodicCell):
        raise TypeError(
            "quotient_simplices.mesh must retain a translational periodic quotient; got Translational quotient simplices require a PeriodicCell."
        )
    representatives = np.asarray(periodic.vertex_representatives, dtype=np.int64)
    quotient, local = np.unique(representatives, return_inverse=True)
    vertices = np.asarray(vertices, dtype=np.int64)
    points = np.asarray(mesh.coordinates, dtype=np.float64)[quotient]
    shifts = np.asarray(periodic.vertex_shifts, dtype=np.int64)[vertices]
    return (
        points,
        local[vertices],
        shifts,
        periodic.cell,
        "|".join(block.name for block in mesh.blocks),
    )


def edge_key(
    first: int, first_shift: np.ndarray, second: int, second_shift: np.ndarray, /
) -> tuple[tuple[int, ...], bool]:
    """Translation-invariant quotient edge key and whether ``first`` leads it."""

    leads = (first, *first_shift.tolist()) <= (second, *second_shift.tolist())
    if leads:
        return (first, second, *(second_shift - first_shift).tolist()), True
    return (second, first, *(first_shift - second_shift).tolist()), False


class _QuotientBisection:
    """Mutable quotient simplices with an edge-orbit to cell index."""

    def __init__(
        self,
        points: np.ndarray,
        cells: np.ndarray,
        shifts: np.ndarray,
        vectors: np.ndarray,
        /,
    ) -> None:
        self.points = [row for row in points]
        self.cells = [row.copy() for row in cells]
        self.shifts = [row.copy() for row in shifts]
        self.parents = list(range(len(cells)))
        self.barycentric = [np.eye(cells.shape[1]) for _ in cells]
        self.generations = [0 for _ in cells]
        self.vectors = vectors
        self.star: dict[tuple[int, ...], set[int]] = {}
        for cell in range(len(self.cells)):
            self._register(cell)

    def _edges(self, cell: int, /) -> list[tuple[tuple[int, ...], int, int]]:
        """Quotient edge keys of a cell with their local corner pairs."""

        vertices = self.cells[cell]
        shifts = self.shifts[cell]
        arity = vertices.shape[0]
        edges = [
            (
                edge_key(int(vertices[i]), shifts[i], int(vertices[j]), shifts[j])[0],
                i,
                j,
            )
            for i in range(arity)
            for j in range(i + 1, arity)
        ]
        if len({key for key, _, _ in edges}) != len(edges):
            raise ValueError(
                "edges.cell must retain distinct quotient edge orbits within each cell; got A cell contains two members of one quotient edge orbit; orbit bisection requires distinct quotient edges per cell."
            )
        return edges

    def _register(self, cell: int, /) -> None:
        for key, _, _ in self._edges(cell):
            self.star.setdefault(key, set()).add(cell)

    def _unregister(self, cell: int, /) -> None:
        for key, _, _ in self._edges(cell):
            members = self.star[key]
            members.discard(cell)
            if not members:
                del self.star[key]

    def length(self, key: tuple[int, ...], /) -> float:
        first, second = key[0], key[1]
        relative = np.asarray(key[2:], dtype=np.float64)
        return float(
            np.linalg.norm(
                self.points[second] + relative @ self.vectors - self.points[first]
            )
        )

    def longest_edge(self, cell: int, /) -> tuple[int, ...]:
        # Ties are broken by the deterministic key, so every lift decides alike.
        return min(
            (key for key, _, _ in self._edges(cell)),
            key=lambda key: (-self.length(key), key),
        )

    def bisect(self, key: tuple[int, ...], /) -> None:
        """Split every cell of the quotient edge's star at its one midpoint."""

        first, second = key[0], key[1]
        relative = np.asarray(key[2:], dtype=np.float64)
        midpoint = len(self.points)
        self.points.append(
            0.5 * (self.points[first] + self.points[second] + relative @ self.vectors)
        )
        for cell in sorted(self.star[key]):
            ((i, j),) = [(i, j) for edge, i, j in self._edges(cell) if edge == key]
            self._unregister(cell)
            vertices = self.cells[cell]
            shifts = self.shifts[cell]
            parent_barycentric = self.barycentric[cell]
            midpoint_barycentric = 0.5 * (parent_barycentric[i] + parent_barycentric[j])
            _, leads = edge_key(int(vertices[i]), shifts[i], int(vertices[j]), shifts[j])
            # The midpoint sits at the leading corner's image in this lift;
            # replacing either endpoint by it preserves the orientation.
            image = shifts[i] if leads else shifts[j]
            children = []
            child_barycentric = []
            for replaced in (j, i):
                child_vertices = vertices.copy()
                child_shifts = shifts.copy()
                child_vertices[replaced] = midpoint
                child_shifts[replaced] = image
                children.append((child_vertices, child_shifts))
                barycentric = parent_barycentric.copy()
                barycentric[replaced] = midpoint_barycentric
                child_barycentric.append(barycentric)
            self.cells[cell], self.shifts[cell] = children[0]
            self.cells.append(children[1][0])
            self.shifts.append(children[1][1])
            self.parents.append(self.parents[cell])
            self.barycentric[cell] = child_barycentric[0]
            self.barycentric.append(child_barycentric[1])
            self.generations[cell] += 1
            self.generations.append(self.generations[cell])
            self._register(cell)
            self._register(len(self.cells) - 1)


class _RefinementSource(StrictModule):
    dimension: int = eqx.field(static=True)
    cell_count: int = eqx.field(static=True)
    source_cell_count: int = eqx.field(static=True)


class _RefinementBanks(StrictModule):
    parents: np.ndarray
    reference: np.ndarray
    generations: np.ndarray


def _prepare_refinement_source(
    mesh: cell_mesh_owner.CellMesh,
    source: cell_mesh_owner.CellMesh,
    source_geometry: CellGeometrySpec,
    budget: CoordinateEnclosureBudget,
) -> _RefinementSource:
    bank = prepared_coordinate_source_bank(source_geometry, budget=budget)
    elements, routes, _ = source_geometry.resolve(
        source, exact_source_prepared=True, budget=budget
    )
    reserve_periodic_exact_terms(
        _REFINEMENT_CELL_WORK * sum(block.cell_count for block in source.blocks),
        budget=budget,
    )
    host_coordinates, host_vertices = jax.device_get(
        (source.coordinates, tuple(block.vertices for block in source.blocks))
    )
    for element, rows, vertices in zip(elements, routes, host_vertices, strict=True):
        carrier = np.asarray(host_coordinates)[np.asarray(vertices)]
        for row, expected in zip(rows, carrier, strict=True):
            corners = coordinate_corner_images(
                element, tuple(bank[int(index)] for index in row), budget=budget
            )
            if corners is None or not np.array_equal(
                np.asarray(tuple(rounded_point(point) for point in corners)).view(
                    np.uint64
                ),
                np.asarray(expected, dtype=np.float64).view(np.uint64),
            ):
                raise ValueError(
                    "prepare_refinement_source.mesh must match the source geometry, simplex dimension, and group action; got Periodic history source geometry differs from its source carrier."
                )
    dimension = source.topological_dimension
    if mesh.topological_dimension != dimension or dimension not in (2, 3):
        raise ValueError(
            "prepare_refinement_source.mesh must match the source geometry, simplex dimension, and group action; got Periodic history requires matching simplex dimensions."
        )
    source_topology = require_periodic_topology(source)
    target_topology = require_periodic_topology(mesh)

    if identification_id(source_topology.cell) != identification_id(target_topology.cell):
        raise ValueError(
            "prepare_refinement_source.mesh must match the source geometry, simplex dimension, and group action; got Periodic history changes the group action."
        )
    if any(
        block.cell_kind != _SIMPLEX_KINDS[dimension]
        for block in (*source.blocks, *mesh.blocks)
    ):
        raise ValueError(
            "prepare_refinement_source.mesh must match the source geometry, simplex dimension, and group action; got Periodic histories require simplex carriers."
        )
    return _RefinementSource(
        dimension,
        sum(block.cell_count for block in mesh.blocks),
        sum(block.cell_count for block in source.blocks),
    )


def _prepare_refinement_banks(
    prepared: _RefinementSource,
    mesh: cell_mesh_owner.CellMesh,
    parent_cells: np.ndarray,
    parent_reference_vertices: np.ndarray,
    cell_generations: np.ndarray,
    bisected_edges: int,
    quotient: PeriodicQuotientEvidence,
) -> _RefinementBanks:
    dimension, count, source_count = (
        prepared.dimension,
        prepared.cell_count,
        prepared.source_cell_count,
    )
    target_topology = require_periodic_topology(mesh)
    parents = np.asarray(parent_cells)
    reference = np.asarray(parent_reference_vertices)
    generations = np.asarray(cell_generations)
    if (
        parents.shape != (count,)
        or parents.dtype.kind not in "iu"
        or generations.shape != (count,)
        or generations.dtype.kind not in "iu"
        or reference.shape != (count, dimension + 1, dimension)
        or reference.dtype != np.dtype(np.float64)
        or np.any(parents < 0)
        or np.any(parents >= source_count)
        or np.any(generations < 0)
        or not np.all(np.isfinite(reference))
        or type(bisected_edges) is not int
        or bisected_edges < 0
    ):
        raise ValueError(
            "prepare_refinement_banks.prepared must retain valid parent, reference, generation, and quotient banks; got Periodic history has invalid retained parent, reference or generation banks."
        )
    if (
        quotient.periodic_topology_id != target_topology.periodic_topology_id
        or not quotient.all_cells_valid
    ):
        raise ValueError(
            "prepare_refinement_banks.prepared must retain valid parent, reference, generation, and quotient banks; got Periodic history requires its target quotient evidence."
        )
    return _RefinementBanks(parents, reference, generations)


def _certify_refinement_partition(
    prepared: _RefinementSource,
    banks: _RefinementBanks,
    previous: PeriodicRefinement | None,
    retired: PeriodicRefinement | None,
    budget: CoordinateEnclosureBudget,
) -> _RefinementBanks:
    dimension, count, source_count = (
        prepared.dimension,
        prepared.cell_count,
        prepared.source_cell_count,
    )
    parents, reference = banks.parents, banks.reference
    active, visited = set(), set()

    def visit(node: PeriodicRefinement | None) -> None:
        if node is None or id(node) in visited:
            return
        if id(node) in active:
            raise ValueError(
                "visit.node must retain an acyclic predecessor and retirement history; got Periodic history contains a cyclic predecessor or retirement."
            )
        active.add(id(node))
        visit(node.previous)
        visit(node.retired)
        active.remove(id(node))
        visited.add(id(node))

    visit(previous)
    visit(retired)
    reserve_periodic_exact_terms(_REFINEMENT_CELL_WORK * count, budget=budget)
    measures = [Fraction(0) for _ in range(source_count)]
    charts = set()
    for parent, row in zip(parents, reference, strict=True):
        vertices = tuple(
            tuple(Fraction(float(value)) for value in vertex) for vertex in row
        )
        if any(
            any(value < 0 for value in vertex) or sum(vertex) > 1 for vertex in vertices
        ):
            raise ValueError(
                "certify_refinement_partition.prepared must partition each source simplex through an acyclic retained history; got Periodic child reference vertices leave their parent simplex."
            )
        chart = (int(parent), tuple(sorted(vertices)))
        if chart in charts:
            raise ValueError(
                "certify_refinement_partition.prepared must partition each source simplex through an acyclic retained history; got Periodic history duplicates a retained sibling simplex."
            )
        charts.add(chart)
        edges = np.asarray(
            [
                [
                    value - origin
                    for value, origin in zip(vertex, vertices[0], strict=True)
                ]
                for vertex in vertices[1:]
            ],
            dtype=object,
        )
        measure = abs(
            cofactor_determinant_2(*edges)
            if dimension == 2
            else cofactor_determinant_3(*edges)
        )
        if measure == 0:
            raise ValueError(
                "certify_refinement_partition.prepared must partition each source simplex through an acyclic retained history; got Periodic history contains a degenerate sibling reference simplex."
            )
        measures[int(parent)] += measure
    if any(measure != 1 for measure in measures):
        raise ValueError(
            "certify_refinement_partition.prepared must partition each source simplex through an acyclic retained history; got Periodic sibling reference simplices do not conserve every source cell."
        )
    return banks


class PeriodicRefinement(StrictModule, NonTrainableState):
    """Orbit-preserving bisection of a periodic mesh.

    ``parent_cells`` names, per published child cell, the input cell (in input
    block order) it refines. ``bisected_edges`` counts quotient edge orbits
    split, each at one new quotient vertex shared by every seam copy.
    """

    mesh: cell_mesh_owner.CellMesh
    source: cell_mesh_owner.CellMesh
    source_geometry: CellGeometrySpec
    parent_reference_vertices: np.ndarray
    parent_cells: np.ndarray
    cell_generations: np.ndarray
    bisected_edges: int = eqx.field(static=True)
    quotient: PeriodicQuotientEvidence
    refinement_id: str = eqx.field(static=True)
    previous: PeriodicRefinement | None = None
    retired: PeriodicRefinement | None = None

    @checked
    def __init__(
        self,
        *,
        mesh: cell_mesh_owner.CellMesh,
        source: cell_mesh_owner.CellMesh,
        source_geometry: CellGeometrySpec,
        parent_reference_vertices: np.ndarray,
        parent_cells: np.ndarray,
        cell_generations: np.ndarray,
        bisected_edges: int,
        quotient: PeriodicQuotientEvidence,
        previous: PeriodicRefinement | None = None,
        retired: PeriodicRefinement | None = None,
        budget: CoordinateEnclosureBudget,
    ) -> None:

        prepared = _prepare_refinement_source(mesh, source, source_geometry, budget)
        banks = _prepare_refinement_banks(
            prepared,
            mesh,
            parent_cells,
            parent_reference_vertices,
            cell_generations,
            bisected_edges,
            quotient,
        )
        banks = _certify_refinement_partition(prepared, banks, previous, retired, budget)
        parents, reference, generations = (
            banks.parents,
            banks.reference,
            banks.generations,
        )
        self.mesh, self.source = mesh, source
        self.parent_cells = _frozen(parents)
        self.parent_reference_vertices = _frozen(reference)
        self.cell_generations = _frozen(generations)
        self.bisected_edges, self.quotient = bisected_edges, quotient
        self.previous, self.retired = previous, retired
        self.source_geometry = source_geometry
        self.refinement_id = canonical_fingerprint(
            {
                "kind": "periodic-refinement",
                "source": source.mesh_id,
                "target": mesh.mesh_id,
                "source_geometry": cell_geometry_id(source_geometry),
                "source_geometry_arrays": array_tree_fingerprint(source_geometry),
                "parents": array_tree_fingerprint(self.parent_cells),
                "reference": array_tree_fingerprint(self.parent_reference_vertices),
                "generations": array_tree_fingerprint(self.cell_generations),
                "bisected_edges": bisected_edges,
                "quotient": quotient.periodic_topology_id,
                "previous": None if previous is None else previous.refinement_id,
                "retired": None if retired is None else retired.refinement_id,
            }
        )

    def reconstruction_preparation_bounds(self) -> tuple[int, int, int]:
        """Bound replay of the source coefficient bank and both cell-visit charges."""
        work, storage = cell_geometry_preparation_bounds(self.source_geometry)
        cells = sum(
            block.cell_count for mesh in (self.source, self.mesh) for block in mesh.blocks
        )
        return (
            work + _REFINEMENT_CELL_WORK * cells,
            storage,
            cell_geometry_preparation_query_upper(self.source_geometry),
        )


def _periodic_child_blocks(
    source: cell_mesh_owner.CellMesh, vertices: np.ndarray, parents: np.ndarray
) -> tuple[tuple[reference_cell_types_owner.CellBlock, ...], np.ndarray]:
    """Keep source block metadata and return the child publication order."""
    blocks = []
    rows = []
    start = 0
    offset = 0
    for block in source.blocks:
        selected = np.flatnonzero(
            (parents >= start) & (parents < start + block.cell_count)
        )
        rows.append(selected)
        blocks.append(
            reference_cell_types_owner.CellBlock(
                block.name,
                block.cell_kind,
                vertices[selected],
                global_ids=np.arange(offset, offset + selected.size, dtype=np.int64),
            )
        )
        offset += selected.size
        start += block.cell_count
    return tuple(blocks), np.concatenate(rows)


def _refine_isometry_mesh(
    mesh: cell_mesh_owner.CellMesh,
    cells: ArrayLike | None,
    edge_orbits: ArrayLike | None,
    source_geometry: CellGeometrySpec,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> PeriodicRefinement:
    """Bisect all lifted copies of each selected quotient edge together."""

    topology = require_periodic_topology(mesh)
    if topology is None or not isinstance(topology.cell, PeriodicIsometryGroup):
        raise ValueError(
            "refine_isometry_mesh.mesh must retain an isometry topology and valid marked source cells; got Boundary-isometry refinement requires an isometry topology."
        )
    source_vertices, _ = _periodic_simplex_rows(mesh)
    points = np.asarray(mesh.coordinates).tolist()
    vertices = source_vertices.tolist()
    parents = list(range(len(vertices)))
    generations = [0 for _ in vertices]
    representatives = np.asarray(topology.vertex_representatives).tolist()
    barycentric = [np.eye(source_vertices.shape[1]) for _ in vertices]
    shifts = np.asarray(topology.vertex_shifts).tolist()
    edges = simplex_entity_corners(mesh, 1)
    orbit, _, anchors = (np.asarray(value) for value in topology.orbits(1))
    leaders = np.asarray(topology.orbit_representatives(1))
    lengths = np.linalg.norm(
        np.asarray(points)[edges[:, 1]] - np.asarray(points)[edges[:, 0]], axis=1
    )
    selected = set(orbit.tolist())
    if cells is not None:
        marked = np.asarray(cells)
        if marked.ndim != 1 or not np.issubdtype(marked.dtype, np.integer):
            raise TypeError(
                "refine_isometry_mesh.mesh must retain an isometry topology and valid marked source cells; got cells must be a one-dimensional integer array."
            )
        if np.any(marked < 0) or np.any(marked >= len(source_vertices)):
            raise ValueError(
                "refine_isometry_mesh.mesh must retain an isometry topology and valid marked source cells; got cells must index cells of the periodic mesh."
            )
        by_pair = {tuple(sorted(edge)): row for row, edge in enumerate(edges.tolist())}
        selected = set()
        for cell in np.unique(marked):
            corners = vertices[int(cell)]
            local_edges = [
                by_pair[tuple(sorted((first, second)))]
                for position, first in enumerate(corners)
                for second in corners[position + 1 :]
            ]
            chosen = max(local_edges, key=lambda edge: (lengths[edge], -edge))
            selected.add(int(orbit[chosen]))
    if edge_orbits is not None:
        selected = set(np.asarray(edge_orbits, dtype=np.int64).tolist())
    queue = sorted(selected, key=lambda key: (-lengths[leaders[key]], key))
    for key in queue:
        copies = np.flatnonzero(orbit == key)
        leader = int(leaders[key])
        first_midpoint = len(points)
        # Publish the orbit leader first, so every shift refers to its zero image.
        copies = np.concatenate(([leader], copies[copies != leader]))
        for edge in copies:
            first, second = edges[edge].tolist()
            midpoint = len(points)
            points.append(
                ((np.asarray(points[first]) + np.asarray(points[second])) * 0.5).tolist()
            )
            representatives.append(first_midpoint)
            shifts.append((anchors[edge] - anchors[leader]).tolist())
            old_count = len(vertices)
            for cell in range(old_count):
                corners = vertices[cell]
                if first not in corners or second not in corners:
                    continue
                parent_barycentric = barycentric[cell]
                midpoint_barycentric = 0.5 * (
                    parent_barycentric[corners.index(first)]
                    + parent_barycentric[corners.index(second)]
                )
                child_barycentric = parent_barycentric.copy()
                child_barycentric[corners.index(first)] = midpoint_barycentric
                parent_barycentric[corners.index(second)] = midpoint_barycentric
                barycentric.append(child_barycentric)
                child = corners.copy()
                child[corners.index(first)] = midpoint
                corners[corners.index(second)] = midpoint
                vertices.append(child)
                parents.append(parents[cell])
                generations[cell] += 1
                generations.append(generations[cell])
    coordinates = np.asarray(points, dtype=np.float64)
    successor_blocks, publication_order = _periodic_child_blocks(
        mesh, np.asarray(vertices, dtype=np.int32), np.asarray(parents)
    )
    lifted = cell_mesh_owner.CellMesh(coordinates, successor_blocks)
    quotient = periodic_topology_owner.PeriodicMeshTopology(
        lifted,
        topology.cell,
        np.asarray(representatives),
        np.asarray(shifts),
        budget=budget,
    )
    successor = cell_mesh_owner.CellMesh(
        coordinates, successor_blocks, periodic_topology=quotient
    )
    ancestry = _frozen(np.asarray(parents, dtype=np.int32)[publication_order])
    return PeriodicRefinement(
        source=mesh,
        source_geometry=source_geometry,
        parent_reference_vertices=_frozen(
            np.stack(barycentric)[publication_order, :, 1:]
        ),
        cell_generations=_frozen(
            np.asarray(generations, dtype=np.int32)[publication_order]
        ),
        mesh=successor,
        parent_cells=ancestry,
        bisected_edges=len(queue),
        quotient=PeriodicQuotientEvidence(successor),
        budget=budget,
    )


def _bare_periodic_affine_source(mesh: cell_mesh_owner.CellMesh, /) -> CellGeometrySpec:
    """Bind rational stored source rows for the bare topological refinement API."""
    return CellGeometrySpec(
        {
            block.name: coordinate_lagrange_element(block.cell_kind, 1)
            for block in mesh.blocks
        },
        {block.name: block.vertices for block in mesh.blocks},
        mesh.coordinates,
        storage=mesh.storage,
    )


def _refine_periodic_mesh(
    mesh: cell_mesh_owner.CellMesh,
    /,
    *,
    cells: ArrayLike | None = None,
    edge_orbits: ArrayLike | None = None,
    source_geometry: CellGeometrySpec | None = None,
    budget: CoordinateEnclosureBudget,
) -> PeriodicRefinement:
    """Refine a periodic simplex mesh by quotient edge bisection.

    With ``cells=None`` every quotient edge of the input is bisected once,
    longest first (the four-triangle longest-edge partition in 2D, eight
    children per tetrahedron in 3D). Otherwise the longest quotient edge of
    each listed cell is bisected together with its whole orbit star, so
    conformity and periodicity hold without closure refinement.
    """
    if source_geometry is None:
        source_geometry = require_periodic_topology(mesh).actual_geometry
        if source_geometry is None:
            source_geometry = _bare_periodic_affine_source(mesh)
    if cells is not None and edge_orbits is not None:
        raise ValueError(
            "refine_periodic_mesh.mesh must select valid source cells or quotient edge orbits exclusively; got Select cells or quotient edge orbits, not both."
        )
    if edge_orbits is not None:
        marked_edges = np.asarray(edge_orbits)
        count = require_periodic_topology(mesh).quotient.entities(1).count
        if marked_edges.ndim != 1 or not np.issubdtype(marked_edges.dtype, np.integer):
            raise TypeError(
                "refine_periodic_mesh.mesh must select valid source cells or quotient edge orbits exclusively; got edge_orbits must be a one-dimensional integer array."
            )
        if np.any(marked_edges < 0) or np.any(marked_edges >= count):
            raise ValueError(
                "refine_periodic_mesh.mesh must select valid source cells or quotient edge orbits exclusively; got edge_orbits must name quotient edges of the source."
            )

    if mesh.periodic_topology is not None and isinstance(
        require_periodic_topology(mesh).cell, PeriodicIsometryGroup
    ):
        return _refine_isometry_mesh(
            mesh, cells, edge_orbits, source_geometry, budget=budget
        )
    points, simplices, shifts, cell, block_name = quotient_simplices(mesh)
    vectors = np.asarray(cell.vectors, dtype=np.float64)
    state = _QuotientBisection(points, simplices, shifts, vectors)
    if edge_orbits is not None:
        topology = require_periodic_topology(mesh)
        leaders = np.asarray(topology.orbit_representatives(1))[np.asarray(edge_orbits)]
        endpoints = simplex_entity_corners(mesh, 1)[leaders]
        roots = np.asarray(topology.vertex_representatives)
        unique = np.unique(roots)
        local = np.searchsorted(unique, roots)
        vertex_shifts = np.asarray(topology.vertex_shifts)
        keys = {
            edge_key(
                int(local[first]),
                vertex_shifts[first],
                int(local[second]),
                vertex_shifts[second],
            )[0]
            for first, second in endpoints
        }
        queue = sorted(keys, key=lambda key: (-state.length(key), key))
    elif cells is None:
        queue = sorted(state.star, key=lambda key: (-state.length(key), key))
    else:
        marked = np.asarray(cells)
        if not np.issubdtype(marked.dtype, np.integer) or marked.ndim != 1:
            raise TypeError(
                "refine_periodic_mesh.mesh must select valid source cells or quotient edge orbits exclusively; got cells must be a one-dimensional integer array."
            )
        if np.any(marked < 0) or np.any(marked >= simplices.shape[0]):
            raise ValueError(
                "refine_periodic_mesh.mesh must select valid source cells or quotient edge orbits exclusively; got cells must index cells of the periodic mesh."
            )
        queue = sorted(
            {state.longest_edge(int(index)) for index in np.unique(marked)},
            key=lambda key: (-state.length(key), key),
        )
    bisected = 0
    for key in queue:
        if key in state.star:
            state.bisect(key)
            bisected += 1
    refined = _publish_periodic_simplices(
        np.stack(state.points),
        np.stack(state.cells),
        np.stack(state.shifts),
        cell,
        block_name=block_name,
        budget=budget,
    )
    parents = np.asarray(state.parents, dtype=np.int32)
    refined_blocks, publication_order = _periodic_child_blocks(
        mesh, _periodic_simplex_rows(refined)[0], parents
    )
    lifted = cell_mesh_owner.CellMesh(np.asarray(refined.coordinates), refined_blocks)
    topology = require_periodic_topology(refined)
    refined = cell_mesh_owner.CellMesh(
        np.asarray(refined.coordinates),
        refined_blocks,
        periodic_topology=periodic_topology_owner.PeriodicMeshTopology(
            lifted,
            cell,
            topology.vertex_representatives,
            topology.vertex_shifts,
            budget=budget,
        ),
    )
    return PeriodicRefinement(
        mesh=refined,
        source=mesh,
        source_geometry=source_geometry,
        parent_reference_vertices=_frozen(
            np.stack(state.barycentric)[publication_order, :, 1:]
        ),
        cell_generations=_frozen(
            np.asarray(state.generations, dtype=np.int32)[publication_order]
        ),
        parent_cells=_frozen(parents[publication_order]),
        bisected_edges=bisected,
        quotient=PeriodicQuotientEvidence(refined),
        budget=budget,
    )


def refine_periodic_mesh(
    mesh: cell_mesh_owner.CellMesh,
    /,
    *,
    cells: ArrayLike | None = None,
    edge_orbits: ArrayLike | None = None,
    source_geometry: CellGeometrySpec | None = None,
    scope: NativeExecutionScope | None = None,
    budget: CoordinateEnclosureBudget | None = None,
) -> PeriodicRefinement:
    """Refine quotient edge orbits under explicit or source-derived preparation bounds."""
    if budget is None:
        topology = require_periodic_topology(mesh)
        if source_geometry is None:
            source_geometry = topology.actual_geometry
            if source_geometry is None:
                source_geometry = _bare_periodic_affine_source(mesh)
        source_work, source_bytes = cell_geometry_preparation_bounds(source_geometry)
        source_queries = cell_geometry_preparation_query_upper(source_geometry)
        source_cells = sum(block.cell_count for block in mesh.blocks)
        dimension = mesh.topological_dimension
        # Each cell can branch twice at each of its edge keys.
        # No newly created edge is added to the refinement queue.
        edges_per_cell = dimension * (dimension + 1) // 2
        child_cells_upper = source_cells * (1 << edges_per_cell)
        vertices_upper = mesh.coordinates.shape[0] + child_cells_upper * (dimension + 1)
        shift_bits = np.full(
            (1, topology.cell.rank), np.iinfo(np.int64).min, dtype=np.int64
        )
        target_work, target_bytes, target_queries = periodic_preparation_bounds(
            topology.cell, vertices_upper, shift_bits, stabilizers=True
        )
        input_work, input_bytes, input_queries = periodic_preparation_bounds(
            topology.cell,
            mesh.coordinates.shape[0],
            np.asarray(topology.vertex_shifts),
            actual_geometry=source_geometry,
            stabilizers=True,
        )
        # Publication and block-preserving reconstruction each build a topology;
        # the source evidence retains the constructor's declared cell visits.
        work_upper = (
            source_work
            + input_work
            + 2 * target_work
            + _REFINEMENT_CELL_WORK * source_cells
        )
        bytes_upper = source_bytes + input_bytes + 2 * target_bytes
        queries_upper = source_queries + input_queries + 2 * target_queries
        ticket = HostBatchTicket(
            work_upper, queries_upper=queries_upper, bytes_upper=bytes_upper
        )
        budget = CoordinateEnclosureBudget(
            ticket,
            maximum_work_units=work_upper,
            maximum_memory_bytes=bytes_upper,
        )
        if scope is None:
            with ticket, budget.activate():
                return _refine_periodic_mesh(
                    mesh,
                    cells=cells,
                    edge_orbits=edge_orbits,
                    source_geometry=source_geometry,
                    budget=budget,
                )
        ticket.settle()
    with _periodic_phase(
        scope, budget, budget.maximum_work_units, budget.maximum_memory_bytes
    ):
        return _refine_periodic_mesh(
            mesh,
            cells=cells,
            edge_orbits=edge_orbits,
            source_geometry=source_geometry,
            budget=budget,
        )


def _periodic_history_nodes(
    history: PeriodicRefinement | None,
) -> tuple[PeriodicRefinement, ...]:
    """Visit the retained epoch DAG once; retirement never reuses scientific IDs."""

    result = []
    queue = [] if history is None else [history]
    visited = set()
    for node in queue:
        if id(node) in visited:
            continue
        visited.add(id(node))
        result.append(node)
        if node.previous is not None:
            queue.append(node.previous)
        if node.retired is not None:
            queue.append(node.retired)
    return tuple(result)


def _periodic_history_meshes(
    history: PeriodicRefinement | None,
) -> tuple[cell_mesh_owner.CellMesh, ...]:
    meshes = {}
    for node in _periodic_history_nodes(history):
        for mesh in (node.source, node.mesh):
            meshes.setdefault(mesh.mesh_id, mesh)
    return tuple(meshes.values())


class _ScientificOwners(StrictModule):
    meshes: tuple[cell_mesh_owner.CellMesh, ...]


class _ScientificBlockRows(StrictModule):
    rows: NDArray[np.int32]


def _prepare_scientific_owners(
    history: PeriodicRefinement,
    source_topology_id: str,
    admit: Callable[[int, int], None],
) -> _ScientificOwners:
    admit(1, 256)
    queue, visited, active, owners = [(history, False)], set(), set(), []
    while queue:
        node, finished = queue.pop()
        if finished:
            active.remove(id(node))
            visited.add(id(node))
            continue
        if type(node) is not PeriodicRefinement or id(node) in active:
            raise ValueError(
                "prepare_scientific_owners.history must retain an acyclic history with its declared source owner; got Periodic scientific block ownership has an invalid or cyclic history."
            )
        if id(node) in visited:
            continue
        admit(3, 768)
        active.add(id(node))
        queue.append((node, True))
        for mesh in (node.source, node.mesh):
            if mesh.topology_id == source_topology_id:
                owners.append(mesh)
        queue.extend(
            (child, False) for child in (node.previous, node.retired) if child is not None
        )
    if not owners:
        raise ValueError(
            "prepare_scientific_owners.history must retain an acyclic history with its declared source owner; got Periodic restriction declares a root with no true retained source owner."
        )
    return _ScientificOwners(tuple(owners))


def _bind_scientific_rows(
    source: CellMeshingResult,
    root: cell_mesh_owner.CellMesh,
    root_bank: tuple[np.ndarray, np.ndarray, tuple[tuple[np.ndarray, np.ndarray], ...]],
    parent_banks: tuple[tuple[np.ndarray, np.ndarray], ...],
    count: int,
) -> _ScientificBlockRows:
    root_cells = {}
    root_vertices = np.asarray(root_bank[1])
    for block_index, (_, (vertices, identifiers)) in enumerate(
        zip(root.blocks, root_bank[2], strict=True)
    ):
        for cell, corners in zip(identifiers, vertices, strict=True):
            identifier = int(cell)
            if identifier in root_cells:
                raise ValueError(
                    "bind_scientific_rows.source must bind each presentation row to a unique retained source cell; got Periodic scientific root duplicates an cell ID."
                )
            root_cells[identifier] = (
                block_index,
                tuple(int(value) for value in root_vertices[corners]),
            )
    result = np.empty((count,), dtype=np.int32)
    offset = 0
    for block, (parents, corners) in zip(source.mesh.blocks, parent_banks, strict=True):
        if parents.shape != (block.cell_count,) or corners.shape[0] != block.cell_count:
            raise ValueError(
                "bind_scientific_rows.source must bind each presentation row to a unique retained source cell; got Periodic presentation rows lack all required SCI parent banks."
            )
        for parent, vertices in zip(parents, corners, strict=True):
            owner = root_cells.get(int(parent))
            if (
                owner is None
                or tuple(int(value) for value in vertices if value >= 0) != owner[1]
            ):
                raise ValueError(
                    "bind_scientific_rows.source must bind each presentation row to a unique retained source cell; got Periodic presentation has a missing, foreign or changed SCI cell."
                )
            result[offset] = owner[0]
            offset += 1
    return _ScientificBlockRows(result)


@checked
def periodic_scientific_block_rows(
    source: CellMeshingResult,
    history: PeriodicRefinement,
    /,
    *,
    limits: MeshingLimits,
    scope: NativeExecutionScope,
    budget: CoordinateEnclosureBudget,
) -> NDArray[np.int32]:
    """Bind presentation rows to their real retained scientific blocks."""
    with _periodic_phase(
        scope, budget, budget.maximum_work_units, budget.maximum_memory_bytes
    ):
        origin = source.geometry.restriction_source
        if origin is None:
            raise ValueError(
                "periodic_scientific_block_rows.source must bind the current history and declared scientific source root; got Periodic scientific block lookup requires its declared restriction root."
            )
        if (
            history.mesh.mesh_id != source.mesh.mesh_id
            or history.mesh.numeric_version != source.mesh.numeric_version
        ):
            raise ValueError(
                "periodic_scientific_block_rows.source must bind the current history and declared scientific source root; got Periodic scientific blocks require their current retained history."
            )
        work, storage = 0, 0

        def admit(amount: int, byte_count: int) -> None:
            nonlocal work, storage
            if (
                work + amount > limits.maximum_work_units
                or storage + byte_count > limits.maximum_scratch_bytes
            ):
                raise MeshingFailure(
                    MeshingFailureCategory.RESOURCE_EXHAUSTED,
                    "admit.amount must fit the declared work and scratch allowances; got Periodic scientific block preparation exceeds its allowance.",
                    stage="scientific-class-preparation",
                )
            _periodic_charge(amount, budget=budget)
            budget.reserve(0, byte_count)
            work += amount
            storage += byte_count

        owners = _prepare_scientific_owners(
            history, origin.source_topology_id, admit
        ).meshes
        candidates = (source.mesh, history.mesh, *owners)
        count = sum(block.cell_count for block in source.mesh.blocks)
        bank_rows = sum(
            sum(block.cell_count for block in mesh.blocks) + mesh.coordinates.shape[0]
            for mesh in candidates
        )
        admit(bank_rows + count, (bank_rows + count) * 256)
        # All source leaves are fetched together, before any per-row iteration.
        banks = jax.device_get(
            tuple(
                (
                    mesh.coordinates,
                    mesh.vertex_global_ids,
                    tuple((block.vertices, block.global_ids) for block in mesh.blocks),
                )
                for mesh in candidates
            )
            + (
                tuple(
                    (
                        origin.block_parent_cell_ids[block.name],
                        origin.block_parent_vertex_ids[block.name],
                    )
                    for block in source.mesh.blocks
                ),
            )
        )

        def identity(
            mesh: cell_mesh_owner.CellMesh,
            bank: tuple[
                np.ndarray,
                np.ndarray,
                tuple[tuple[np.ndarray, np.ndarray], ...],
            ],
            /,
        ) -> str:
            return canonical_fingerprint(
                {
                    "topology": mesh.topology_id,
                    "numeric_version": mesh.numeric_version,
                    "blocks": tuple(
                        (block.name, block.cell_kind) for block in mesh.blocks
                    ),
                    "arrays": array_tree_fingerprint(bank),
                }
            )

        if identity(source.mesh, banks[0]) != identity(history.mesh, banks[1]):
            raise ValueError(
                "periodic_scientific_block_rows.source must bind the current history and declared scientific source root; got Periodic scientific block history has changed current mesh banks."
            )
        root = owners[0]
        root_bank = banks[2]
        expected = identity(root, root_bank)
        if any(
            identity(mesh, bank) != expected
            for mesh, bank in zip(owners[1:], banks[3:-1], strict=True)
        ):
            raise ValueError(
                "periodic_scientific_block_rows.source must bind the current history and declared scientific source root; got Periodic restriction has conflicting retained scientific owners."
            )
        return _bind_scientific_rows(source, root, root_bank, banks[-1], count).rows


def periodic_vertex_stencils(
    history: PeriodicRefinement,
) -> tuple[np.ndarray, np.ndarray]:
    """Rational dyadic construction stencils, reconciled on shared lifted vertices."""

    source = history.source
    target = history.mesh
    width = source.topological_dimension + 1
    sources = np.full((target.coordinates.shape[0], width), -1, dtype=np.int64)
    weights = np.zeros_like(sources, dtype=np.float64)
    source_rows, _ = _periodic_simplex_rows(source)
    target_rows, _ = _periodic_simplex_rows(target)
    original = np.asarray(source.vertex_global_ids)[source_rows]
    reference = np.asarray(history.parent_reference_vertices)
    barycentric = np.concatenate(
        (1.0 - np.sum(reference, axis=-1, keepdims=True), reference), axis=-1
    )
    for cell, corners in enumerate(target_rows):
        parent = int(history.parent_cells[cell])
        for corner, vertex in enumerate(corners):
            active = barycentric[cell, corner] > 0
            ids = original[parent, active]
            coefficients = barycentric[cell, corner, active]
            order = np.argsort(ids, kind="stable")
            row = np.full(width, -1, dtype=np.int64)
            values = np.zeros(width)
            row[: ids.size], values[: ids.size] = ids[order], coefficients[order]
            if sources[vertex, 0] >= 0 and (
                not np.array_equal(sources[vertex], row)
                or not np.array_equal(weights[vertex], values)
            ):
                raise ValueError(
                    "periodic_vertex_stencils.history must retain consistent shared-vertex construction and use every lifted vertex; got Periodic cell lifts disagree on a shared vertex's construction evidence."
                )
            sources[vertex], weights[vertex] = row, values
    if np.any(sources[:, 0] < 0):
        raise ValueError(
            "periodic_vertex_stencils.history must retain consistent shared-vertex construction and use every lifted vertex; got A periodic refinement contains an unused lifted vertex."
        )
    return sources, weights


def _periodic_entity_relations(
    source: cell_mesh_owner.CellMesh,
    target: cell_mesh_owner.CellMesh,
    vertex_sources: np.ndarray,
    *,
    coarsening: bool = False,
) -> list[EntityRelations]:

    records = []
    fine, coarse = (source, target) if coarsening else (target, source)
    for degree in range(1, source.topological_dimension):
        fine_rows = simplex_entity_corners(fine, degree)
        coarse_keys = entity_keys(coarse, degree)
        lookup = {tuple(row): row for row in coarse_keys}
        fine_keys = entity_keys(fine, degree)
        first, second, kinds = [], [], []
        for index, corners in enumerate(fine_rows):
            support = np.unique(vertex_sources[corners][vertex_sources[corners] >= 0])
            direct = tuple(fine_keys[index])
            key = direct if coarsening and direct in lookup else tuple(support.tolist())
            if key not in lookup:
                continue
            coarse_key = lookup[key]
            first.append(fine_keys[index] if coarsening else coarse_key)
            second.append(coarse_key if coarsening else fine_keys[index])
            kinds.append(
                int(
                    EntityLineageKind.PRESERVED
                    if coarsening and direct == key
                    else EntityLineageKind.COARSENED_INTO
                    if coarsening
                    else EntityLineageKind.REFINED_FROM
                )
            )
        width = coarse_keys.shape[1]
        records.append(
            EntityRelations(
                degree,
                np.asarray(first, dtype=np.int64).reshape((-1, width)),
                np.asarray(second, dtype=np.int64).reshape((-1, width)),
                np.asarray(kinds, dtype=np.int32),
            )
        )
    return records


def _periodic_prescribed_ids(
    source: cell_mesh_owner.CellMesh,
    target: cell_mesh_owner.CellMesh,
    prior: PeriodicRefinement | None,
) -> tuple[PrescribedEntityIds, ...]:

    result = []
    for degree in range(1, source.topological_dimension):
        current = dict(
            zip(
                map(tuple, entity_keys(source, degree)),
                np.asarray(source.entity_set(degree).entity_ids).tolist(),
                strict=True,
            )
        )
        known = current.copy()
        maximum = max(current.values(), default=-1)
        for old in _periodic_history_meshes(prior):
            for key, identifier in zip(
                entity_keys(old, degree),
                np.asarray(old.entity_set(degree).entity_ids).tolist(),
                strict=True,
            ):
                known.setdefault(tuple(key), identifier)
                maximum = max(maximum, identifier)
        keys, ids = [], []
        for key in sorted(map(tuple, entity_keys(target, degree))):
            if key in current:
                continue
            if key not in known:
                maximum += 1
                if maximum > np.iinfo(np.int64).max:
                    raise ValueError(
                        "periodic_prescribed_ids.source must allocate scientific entity IDs within the integer identity space; got Periodic scientific entity identity space is exhausted."
                    )
                known[key] = maximum
            keys.append(key)
            ids.append(known[key])
        width = entity_keys(target, degree).shape[1]
        result.append(
            PrescribedEntityIds(
                degree,
                np.asarray(keys, dtype=np.int64).reshape((-1, width)),
                np.asarray(ids, dtype=np.int64),
            )
        )
    return tuple(result)


def _periodic_refinement_edit(
    history: PeriodicRefinement,
    prior: PeriodicRefinement | None,
    *,
    budget: CoordinateEnclosureBudget,
) -> CellTopologyEdit:

    source, target = history.source, history.mesh
    sources, weights = periodic_vertex_stencils(history)
    count = sources.shape[0]
    vertex_ids = np.empty(count, dtype=np.int64)
    old_meshes = (source, *_periodic_history_meshes(prior))
    next_id = (
        max(int(np.max(np.asarray(old.vertex_global_ids))) for old in old_meshes) + 1
    )
    retired_vertices = {}
    for node in _periodic_history_nodes(prior):
        old_sources, old_weights = periodic_vertex_stencils(node)
        for vertex, identifier in enumerate(np.asarray(node.mesh.vertex_global_ids)):
            valid = old_sources[vertex] >= 0
            key = (
                tuple(
                    zip(
                        old_sources[vertex, valid].tolist(),
                        old_weights[vertex, valid].tolist(),
                        strict=True,
                    )
                ),
                tuple(np.asarray(node.mesh.coordinates)[vertex]),
            )
            retired_vertices.setdefault(key, int(identifier))
    new = []
    for vertex in range(count):
        active = sources[vertex] >= 0
        if np.count_nonzero(active) == 1 and weights[vertex, 0] == 1.0:
            vertex_ids[vertex] = sources[vertex, 0]
        else:
            new.append(vertex)
    for vertex in sorted(new, key=lambda row: (tuple(sources[row]), tuple(weights[row]))):
        valid = sources[vertex] >= 0
        key = (
            tuple(
                zip(
                    sources[vertex, valid].tolist(),
                    weights[vertex, valid].tolist(),
                    strict=True,
                )
            ),
            tuple(np.asarray(target.coordinates)[vertex]),
        )
        if key in retired_vertices:
            vertex_ids[vertex] = retired_vertices[key]
        else:
            if next_id > np.iinfo(np.int64).max:
                raise ValueError(
                    "periodic_refinement_edit.history must allocate scientific vertex and cell IDs within the integer identity space; got Periodic scientific vertex identity space is exhausted."
                )
            vertex_ids[vertex] = next_id
            next_id += 1
    order = np.argsort(vertex_ids, kind="stable")
    inverse = np.empty_like(order)
    inverse[order] = np.arange(count)
    target_rows, _ = _periodic_simplex_rows(target)
    cells = inverse[target_rows]
    _, source_cell_ids = _periodic_simplex_rows(source)
    parents = np.asarray(history.parent_cells)
    next_cell = max(int(np.max(_periodic_simplex_rows(old)[1])) for old in old_meshes) + 1
    retired_cells = {}
    for old in old_meshes:
        for vertices, identifier in zip(
            *_periodic_simplex_rows(old),
            strict=True,
        ):
            key = tuple(sorted(np.asarray(old.vertex_global_ids)[vertices].tolist()))
            retired_cells.setdefault(key, int(identifier))
    fine_ids = np.empty(parents.size, dtype=np.int64)
    identity = np.concatenate(
        (
            np.zeros((1, source.topological_dimension)),
            np.eye(source.topological_dimension),
        )
    )
    for cell in range(parents.size):
        if np.array_equal(history.parent_reference_vertices[cell], identity):
            fine_ids[cell] = source_cell_ids[parents[cell]]
        else:
            key = tuple(sorted(vertex_ids[target_rows[cell]].tolist()))
            if key in retired_cells:
                fine_ids[cell] = retired_cells[key]
            else:
                if next_cell > np.iinfo(np.int64).max:
                    raise ValueError(
                        "periodic_refinement_edit.history must allocate scientific vertex and cell IDs within the integer identity space; got Periodic scientific cell identity space is exhausted."
                    )
                fine_ids[cell] = next_cell
                next_cell += 1
    active = sources >= 0
    vertex_source = sources[active][:, None]
    vertex_target = np.repeat(vertex_ids, np.sum(active, axis=1))[:, None]
    vertex_relations = EntityRelations(
        0,
        vertex_source,
        vertex_target,
        np.where(
            vertex_source[:, 0] == vertex_target[:, 0],
            int(EntityLineageKind.PRESERVED),
            int(EntityLineageKind.REFINED_FROM),
        ).astype(np.int32),
    )
    relations = [vertex_relations, *_periodic_entity_relations(source, target, sources)]
    relations.append(
        EntityRelations(
            source.topological_dimension,
            source_cell_ids[parents, None],
            fine_ids[:, None],
            np.where(
                source_cell_ids[parents] == fine_ids,
                int(EntityLineageKind.PRESERVED),
                int(EntityLineageKind.REFINED_FROM),
            ).astype(np.int32),
        )
    )
    # Intermediate keys above must use the newly issued target scientific IDs.
    original_ids = np.asarray(target.vertex_global_ids)
    relabeled = dict(zip(original_ids.tolist(), vertex_ids.tolist(), strict=True))
    for degree in range(1, source.topological_dimension):
        record = relations[degree]
        changed = np.asarray(
            [[relabeled[int(value)] for value in row] for row in record.target_keys],
            dtype=np.int64,
        ).reshape(record.target_keys.shape)
        mapped = np.sort(changed, axis=1)
        relations[degree] = record._replace(
            target_keys=mapped,
            kinds=np.where(
                np.all(record.source_keys == mapped, axis=1),
                int(EntityLineageKind.PRESERVED),
                record.kinds,
            ).astype(np.int32),
        )
    labeled_blocks = []
    edit_blocks = []
    start = 0
    for block in target.blocks:
        stop = start + block.cell_count
        labeled_blocks.append(
            reference_cell_types_owner.CellBlock(
                block.name,
                block.cell_kind,
                cells[start:stop],
                global_ids=fine_ids[start:stop],
            )
        )
        edit_blocks.append(
            TopologyEditBlock(
                block.name,
                block.cell_kind,
                block.cell_kind,
                cells[start:stop],
                fine_ids[start:stop],
            )
        )
        start = stop
    labeled_target = cell_mesh_owner.CellMesh(
        np.asarray(target.coordinates)[order],
        tuple(labeled_blocks),
        vertex_global_ids=vertex_ids[order],
    )
    prescribed = _periodic_prescribed_ids(source, labeled_target, prior)
    return CellTopologyEdit(
        "nested_refinement",
        np.asarray(target.coordinates)[order],
        vertex_ids[order],
        tuple(edit_blocks),
        sources[order],
        weights[order],
        active[order],
        tuple(relations),
        prescribed_entity_ids=prescribed,
        periodic_orbits=_periodic_orbit_witness(
            source, history.mesh, labeled_target, prior, order, budget=budget
        ),
        refinement=NestedReferenceWitnesses(
            fine_ids,
            source_cell_ids[parents],
            np.asarray(history.parent_reference_vertices),
        ),
    )


class PeriodicCoarseningCellDim(Dim):
    """Scientific cells admitted for one periodic sibling coarsening request."""


@checked
def periodic_coarsening_cell_ids(
    prepared: adaptation_owner.PreparedMeshAdaptation, /
) -> HostInt64[PeriodicCoarseningCellDim]:
    """Return ascending requested cell IDs admitted by periodic sibling closure.

    The execution owner uses the same metered selection. This is a selection
    query, not a topology edit or a certificate for a later modified request.
    """
    request = prepared.request
    if not isinstance(request, adaptation_owner.MarkedMeshAdaptation) or not isinstance(
        request.hierarchy, PeriodicRefinement
    ):
        raise ValueError(
            "periodic_coarsening_cell_ids.prepared must retain marked periodic history; got no periodic sibling request"
        )
    if np.asarray(request.refine_cell_ids).size:
        raise ValueError(
            "periodic_coarsening_cell_ids.prepared must contain only coarsening marks; got refinement marks"
        )
    history = request.hierarchy
    limits = prepared.policy.limits
    with NativeExecutionScope(limits.native_limits()) as scope:
        initial = scope.batch(0)
        budget = CoordinateEnclosureBudget(
            initial,
            maximum_work_units=limits.maximum_work_units,
            maximum_memory_bytes=limits.maximum_scratch_bytes,
        )
        initial.settle()
        with _periodic_phase(
            scope, budget, limits.maximum_work_units, limits.maximum_scratch_bytes
        ):
            validate_periodic_bisection(prepared, budget=budget)
            parents = _periodic_coarsening_parents(prepared, history, budget=budget)
            _, cell_ids = _periodic_simplex_rows(history.mesh)
            selected = np.sort(cell_ids[np.isin(history.parent_cells, parents)])
    return parse(
        selected,
        HostInt64[PeriodicCoarseningCellDim],
        "periodic_coarsening_cell_ids",
    )


def _periodic_coarsening_parents(
    prepared: adaptation_owner.PreparedMeshAdaptation,
    history: PeriodicRefinement,
    *,
    budget: CoordinateEnclosureBudget,
) -> np.ndarray:
    """Meter the common compatible sibling/edge-star selection."""
    fine_count = sum(block.cell_count for block in history.mesh.blocks)
    coarse_count = sum(block.cell_count for block in history.source.blocks)
    _periodic_charge(
        8 * fine_count * coarse_count + 64 * (fine_count + coarse_count),
        budget=budget,
    )
    with budget.temporary_scope():
        budget.reserve(0, 512 * (fine_count + coarse_count))
        return _select_periodic_coarsening_parents(prepared, history)


def _select_periodic_coarsening_parents(
    prepared: adaptation_owner.PreparedMeshAdaptation, history: PeriodicRefinement
) -> np.ndarray:
    """All required compatible sibling families, closed across every split edge orbit."""

    request = prepared.request
    if not isinstance(request, adaptation_owner.MarkedMeshAdaptation):
        raise TypeError(
            "periodic_coarsening_parents.prepared must retain marked sibling families with scientific constraint rows; got Periodic sibling coarsening requires a marked request."
        )
    fine, coarse = history.mesh, history.source
    fine_cells, fine_ids = _periodic_simplex_rows(fine)
    coarse_cells, _ = _periodic_simplex_rows(coarse)
    parents = np.asarray(history.parent_cells)
    marked = np.isin(fine_ids, np.asarray(request.coarsen_cell_ids))
    class_rows = key_rows(
        entity_keys(fine, fine.topological_dimension), fine_ids[:, None]
    )
    if np.any(class_rows < 0):
        raise ValueError(
            "periodic_coarsening_parents.prepared must retain marked sibling families with scientific constraint rows; got Periodic sibling cells lack their scientific constraint rows."
        )
    cell_classes = np.asarray(prepared.constraints.cell_classes)[class_rows]
    families = [
        np.flatnonzero(parents == parent) for parent in range(coarse_cells.shape[0])
    ]
    selected = np.asarray(
        [
            rows.size > 1
            and np.all(marked[rows])
            and np.unique(cell_classes[rows]).size == 1
            for rows in families
        ],
        dtype=np.bool_,
    )
    blocked_vertices = np.isin(
        np.asarray(fine.vertex_global_ids), prepared.constraints.protected_vertex_ids
    ) & ~np.isin(np.asarray(fine.vertex_global_ids), np.asarray(coarse.vertex_global_ids))
    blocked_cells = np.any(blocked_vertices[fine_cells], axis=1)
    selected[np.unique(parents[blocked_cells])] = False
    source_edge_keys = set(map(tuple, entity_keys(coarse, 1)))
    edge_keys = entity_keys(fine, 1)
    for edge in np.flatnonzero(prepared.constraints.protected_edge_mask):
        if tuple(edge_keys[edge]) in source_edge_keys:
            continue
        endpoints = simplex_entity_corners(fine, 1)[edge]
        # Explicit endpoint containment avoids treating a merely adjacent vertex
        # as an edge-protection conflict.
        touched = np.asarray(
            [np.all(np.isin(endpoints, corners)) for corners in fine_cells]
        )
        selected[np.unique(parents[touched])] = False
    # A label/material trace may not be merged across inconsistent child classes.
    degree = coarse.topological_dimension - 1
    construction, _ = periodic_vertex_stencils(history)
    fine_facets = simplex_entity_corners(fine, degree)
    coarse_keys = entity_keys(coarse, degree)
    lookup = {tuple(key): index for index, key in enumerate(coarse_keys)}
    children = {}
    for facet, corners in enumerate(fine_facets):
        support = np.unique(construction[corners][construction[corners] >= 0])
        key = tuple(support.tolist())
        if key in lookup:
            children.setdefault(lookup[key], []).append(facet)
    original_vertices = np.asarray(coarse.vertex_global_ids)[coarse_cells]
    for facet, descendants in children.items():
        if np.unique(prepared.constraints.facet_classes[descendants]).size > 1:
            incident = np.asarray(
                [
                    np.all(np.isin(coarse_keys[facet], corners))
                    for corners in original_vertices
                ]
            )
            selected[incident] = False
    by_pair = {
        tuple(sorted(edge)): row
        for row, edge in enumerate(simplex_entity_corners(coarse, 1).tolist())
    }
    edge_orbit = np.asarray(require_periodic_topology(coarse).orbits(1)[0])
    stars = {}
    for parent, corners in enumerate(coarse_cells):
        for pair in combinations(corners.tolist(), 2):
            orbit = int(edge_orbit[by_pair[tuple(sorted(pair))]])
            stars.setdefault(orbit, set()).add(parent)
    present = set(require_periodic_topology(fine).entity_keys(1))
    split = [
        index
        for index, key in enumerate(require_periodic_topology(coarse).entity_keys(1))
        if key not in present
    ]
    changed = True
    while changed:
        changed = False
        for orbit in split:
            rows = np.asarray(sorted(stars[orbit]), dtype=np.int64)
            if np.any(selected[rows]) and not np.all(selected[rows]):
                selected[rows] = False
                changed = True
    return np.flatnonzero(selected)


def _periodic_coarsening_edit(
    history: PeriodicRefinement,
    restored_parents: np.ndarray,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[CellTopologyEdit, np.ndarray, np.ndarray, np.ndarray]:

    fine, original = history.mesh, history.source
    construction_sources, _ = periodic_vertex_stencils(history)
    fine_cells, fine_ids = _periodic_simplex_rows(fine)
    original_cells, original_ids = _periodic_simplex_rows(original)
    parents = np.asarray(history.parent_cells)
    replaced = np.isin(parents, restored_parents)
    kept = np.flatnonzero(~replaced)
    original_vertex_ids = np.asarray(original.vertex_global_ids)
    fine_vertex_ids = np.asarray(fine.vertex_global_ids)
    lookup = {int(identifier): row for row, identifier in enumerate(fine_vertex_ids)}
    restored_cells = np.asarray(
        [
            [lookup[int(identifier)] for identifier in original_vertex_ids[corners]]
            for corners in original_cells[restored_parents]
        ],
        dtype=np.int32,
    ).reshape((-1, fine.topological_dimension + 1))
    cells = np.concatenate((fine_cells[kept], restored_cells))
    cell_ids = np.concatenate((fine_ids[kept], original_ids[restored_parents]))
    used = np.unique(cells)
    used = used[np.argsort(fine_vertex_ids[used], kind="stable")]
    local = np.full(fine_vertex_ids.size, -1, dtype=np.int32)
    local[used] = np.arange(used.size, dtype=np.int32)
    ids = fine_vertex_ids[used]
    sources = ids[:, None]
    target_parents = np.concatenate((parents[kept], restored_parents))
    blocks = []
    output_rows = []
    source_kinds = {block.name: block.cell_kind for block in fine.blocks}
    offset = 0
    for original_block in original.blocks:
        rows = np.flatnonzero(
            (target_parents >= offset)
            & (target_parents < offset + original_block.cell_count)
        )
        offset += original_block.cell_count
        if not rows.size:
            continue
        output_rows.append(rows)
        blocks.append(
            TopologyEditBlock(
                original_block.name,
                original_block.cell_kind,
                source_kinds.get(original_block.name),
                local[cells[rows]],
                cell_ids[rows],
            )
        )
    output_order = np.concatenate(output_rows)
    target = cell_mesh_owner.CellMesh(
        np.asarray(fine.coordinates)[used],
        tuple(
            reference_cell_types_owner.CellBlock(
                block.name, block.cell_kind, block.cells, global_ids=block.cell_ids
            )
            for block in blocks
        ),
        vertex_global_ids=ids,
    )
    target_ids = np.where(replaced, original_ids[parents], fine_ids)
    identity = np.concatenate(
        (np.zeros((1, fine.topological_dimension)), np.eye(fine.topological_dimension))
    )
    reference = np.where(
        replaced[:, None, None],
        np.asarray(history.parent_reference_vertices),
        identity[None],
    )
    relations = [
        EntityRelations(
            0,
            sources,
            sources,
            np.full(ids.size, int(EntityLineageKind.PRESERVED), dtype=np.int32),
        )
    ]
    relations.extend(
        _periodic_entity_relations(fine, target, construction_sources, coarsening=True)
    )
    relations.append(
        EntityRelations(
            fine.topological_dimension,
            fine_ids[:, None],
            target_ids[:, None],
            np.where(
                replaced,
                int(EntityLineageKind.COARSENED_INTO),
                int(EntityLineageKind.PRESERVED),
            ).astype(np.int32),
        )
    )
    edit = CellTopologyEdit(
        "nested_coarsening",
        np.asarray(target.coordinates),
        ids,
        tuple(blocks),
        sources,
        np.ones(sources.shape),
        np.ones(sources.shape, dtype=np.bool_),
        tuple(relations),
        prescribed_entity_ids=_periodic_prescribed_ids(fine, target, history),
        periodic_orbits=_periodic_orbit_witness(
            fine, fine, target, history, used, budget=budget
        ),
        coarsening=NestedReferenceWitnesses(fine_ids, target_ids, reference),
    )
    target_reference = np.concatenate(
        (
            np.asarray(history.parent_reference_vertices)[kept],
            np.broadcast_to(identity, (restored_parents.size, *identity.shape)),
        )
    )
    base_generations = np.zeros(original_ids.size, dtype=np.int32)
    if history.previous is not None:
        _, previous_ids = _periodic_simplex_rows(history.previous.mesh)
        by_id = dict(
            zip(
                previous_ids.tolist(),
                np.asarray(history.previous.cell_generations).tolist(),
                strict=True,
            )
        )
        base_generations = np.asarray(
            [by_id[int(identifier)] for identifier in original_ids], dtype=np.int32
        )
    generations = np.concatenate(
        (np.asarray(history.cell_generations)[kept], base_generations[restored_parents])
    )
    return (
        edit,
        target_parents[output_order],
        target_reference[output_order],
        generations[output_order],
    )


def _periodic_identity_banks(
    source: cell_mesh_owner.CellMesh, history: PeriodicRefinement | None
) -> tuple[PeriodicEntityIdentityBank, ...]:
    """Read keyed allocation history from scientific owners, never live row order."""

    topology = require_periodic_topology(source)
    identification = identification_id(topology.cell)
    if history is not None and (
        history.mesh.topology_id != source.topology_id
        or not np.array_equal(history.mesh.coordinates, source.coordinates)
    ):
        raise ValueError(
            "periodic_identity_banks.source must retain source-epoch bindings, identification, and unique scientific entity IDs; got Periodic allocation history does not bind the source epoch."
        )
    owners = (source, *_periodic_history_meshes(history))
    result = []
    for degree in range(1, source.topological_dimension):
        known: dict[tuple[int, ...], int] = {}
        reverse: dict[int, tuple[int, ...]] = {}
        cursor = 0
        for owner in owners:
            periodic = require_periodic_topology(owner)
            if identification_id(periodic.cell) != identification:
                raise ValueError(
                    "periodic_identity_banks.source must retain source-epoch bindings, identification, and unique scientific entity IDs; got Periodic allocation history changes identification."
                )
            recorded = periodic.allocator_next_ids[degree]
            if recorded < 0:
                raise ValueError(
                    "periodic_identity_banks.source must retain source-epoch bindings, identification, and unique scientific entity IDs; got Periodic scientific allocation history is unresolved."
                )
            cursor = max(cursor, recorded)
            for key, identifier in zip(
                periodic.entity_keys(degree),
                np.asarray(periodic.quotient.entities(degree).entity_ids).tolist(),
                strict=True,
            ):
                if (
                    key in known
                    and known[key] != identifier
                    or identifier in reverse
                    and reverse[identifier] != key
                ):
                    raise ValueError(
                        "periodic_identity_banks.source must retain source-epoch bindings, identification, and unique scientific entity IDs; got Periodic history reuses a scientific entity identity."
                    )
                known[key] = int(identifier)
                reverse[int(identifier)] = key
        keys = sorted(known)
        result.append(
            PeriodicEntityIdentityBank(
                degree, tuple(keys), tuple(known[key] for key in keys), cursor
            )
        )
    return tuple(result)


@checked
def require_periodic_identity_bank(bank: PeriodicEntityIdentityBank, /) -> None:
    """Validate the immutable variable-arity identity payload without coercion."""

    if (
        not isinstance(bank.entity_keys, tuple)
        or not isinstance(bank.entity_global_ids, tuple)
        or isinstance(bank.degree, bool)
        or not isinstance(bank.degree, (int, np.integer))
        or bank.degree < 1
        or isinstance(bank.allocator_next_id, bool)
        or not isinstance(bank.allocator_next_id, (int, np.integer))
        or bank.allocator_next_id < 0
        or any(
            not isinstance(key, tuple)
            or not key
            or any(
                isinstance(value, bool) or not isinstance(value, (int, np.integer))
                for value in key
            )
            for key in bank.entity_keys
        )
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or value < 0
            or value >= bank.allocator_next_id
            for value in bank.entity_global_ids
        )
        or len(bank.entity_keys) != len(bank.entity_global_ids)
        or len(set(bank.entity_keys)) != len(bank.entity_keys)
        or len(set(bank.entity_global_ids)) != len(bank.entity_global_ids)
    ):
        raise ValueError(
            "require_periodic_identity_bank.bank must retain integer keys, IDs, and a valid allocation cursor; got Periodic scientific identity keys, IDs or allocation cursor are invalid."
        )


def _periodic_target_identity_banks(
    source: cell_mesh_owner.CellMesh,
    target: periodic_topology_owner.PeriodicMeshTopology,
    history: PeriodicRefinement | None,
    retained: tuple[PeriodicEntityIdentityBank, ...] = (),
    allocation_prior: tuple[cell_mesh_owner.CellMesh, PeriodicVertexOrbitWitness]
    | None = None,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[PeriodicEntityIdentityBank, ...]:
    current = _periodic_identity_banks(source, history)
    if allocation_prior is not None:
        current = _validated_periodic_allocation_prior(
            source, allocation_prior, budget=budget
        )
        if retained != current:
            raise ValueError(
                "periodic_target_identity_banks.source must retain the validated prior allocation bank; got Retained periodic identities must equal the validated prior allocation bank."
            )
    return _allocate_periodic_identity_banks(target, current, retained)


def _allocate_periodic_identity_banks(
    target: periodic_topology_owner.PeriodicMeshTopology,
    current: tuple[PeriodicEntityIdentityBank, ...],
    retained: tuple[PeriodicEntityIdentityBank, ...],
    /,
) -> tuple[PeriodicEntityIdentityBank, ...]:
    """Allocate only beyond the validated base's keyed high-water."""

    result = []
    if retained and tuple(bank.degree for bank in retained) != tuple(
        bank.degree for bank in current
    ):
        raise ValueError(
            "allocate_periodic_identity_banks.target must retain all entity degrees, scientific IDs, and allocation cursors; got Retained periodic identities omit an entity dimension."
        )
    for index, bank in enumerate(current):
        known = dict(zip(bank.entity_keys, bank.entity_global_ids, strict=True))
        cursor = bank.allocator_next_id
        if retained:
            prior = retained[index]
            require_periodic_identity_bank(prior)
            if prior.allocator_next_id != cursor:
                raise ValueError(
                    "allocate_periodic_identity_banks.target must retain all entity degrees, scientific IDs, and allocation cursors; got Retained periodic identities do not bind the current allocation cursor."
                )
            historical = dict(
                zip(prior.entity_keys, prior.entity_global_ids, strict=True)
            )
            if len(historical) != len(prior.entity_keys) or any(
                historical.get(key) != identifier for key, identifier in known.items()
            ):
                raise ValueError(
                    "allocate_periodic_identity_banks.target must retain all entity degrees, scientific IDs, and allocation cursors; got Retained periodic identities do not include the current scientific epoch."
                )
            known = historical
            cursor = prior.allocator_next_id
        for key in sorted(target.entity_keys(bank.degree)):
            if key not in known:
                if cursor > np.iinfo(np.int64).max:
                    raise ValueError(
                        "allocate_periodic_identity_banks.target must retain all entity degrees, scientific IDs, and allocation cursors; got Periodic scientific identity allocation is exhausted."
                    )
                known[key] = cursor
                cursor += 1
        keys = sorted(known)
        result.append(
            PeriodicEntityIdentityBank(
                bank.degree, tuple(keys), tuple(known[key] for key in keys), cursor
            )
        )
    return tuple(result)


def _validated_periodic_allocation_prior(
    source: cell_mesh_owner.CellMesh,
    prior: tuple[cell_mesh_owner.CellMesh, PeriodicVertexOrbitWitness],
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[PeriodicEntityIdentityBank, ...]:
    """Validate the all required prior chain iteratively, without a private depth cap."""

    chain: list[tuple[cell_mesh_owner.CellMesh, PeriodicVertexOrbitWitness]] = []
    seen: set[int] = set()
    cursor: tuple[cell_mesh_owner.CellMesh, PeriodicVertexOrbitWitness] | None = prior
    while cursor is not None:
        if not isinstance(cursor, tuple) or len(cursor) != 2:
            raise TypeError(
                "validated_periodic_allocation_prior.source must retain a bound, acyclic prior scientific allocation chain; got A periodic allocation prior requires its bound target and deterministic evidence."
            )
        previous, witness = cursor
        if not isinstance(previous, cell_mesh_owner.CellMesh) or not isinstance(
            witness, PeriodicVertexOrbitWitness
        ):
            raise TypeError(
                "validated_periodic_allocation_prior.source must retain a bound, acyclic prior scientific allocation chain; got A periodic allocation prior requires its bound target and deterministic evidence."
            )
        identity = id(witness)
        if identity in seen:
            raise ValueError(
                "validated_periodic_allocation_prior.source must retain a bound, acyclic prior scientific allocation chain; got A periodic allocation prior contains a cyclic evidence chain."
            )
        seen.add(identity)
        chain.append((previous, witness))
        cursor = witness.allocation_prior
    current: tuple[PeriodicEntityIdentityBank, ...] | None = None
    for previous, witness in reversed(chain):
        periodic = require_periodic_topology(previous)
        validated = _bind_periodic_target(
            source, previous, witness, current, budget=budget
        )
        bound = require_periodic_topology(validated)
        if (
            validated.topology_id != previous.topology_id
            or bound.allocator_next_ids != periodic.allocator_next_ids
            or any(
                not np.array_equal(
                    bound.quotient.entities(degree).entity_ids,
                    periodic.quotient.entities(degree).entity_ids,
                )
                for degree in range(1, source.topological_dimension)
            )
        ):
            raise ValueError(
                "validated_periodic_allocation_prior.source must retain a bound, acyclic prior scientific allocation chain; got A periodic allocation prior does not match its bound target identities."
            )
        current = witness.quotient_entities
    if current is None:
        raise ValueError(
            "validated_periodic_allocation_prior.source must retain a bound, acyclic prior scientific allocation chain; got A periodic allocation prior omits its bound target."
        )
    return current


def _periodic_replay_identity_storage_upper(
    witness: PeriodicVertexOrbitWitness, /
) -> int:
    """Bound keyed identity tuples, allocation dictionaries and orbit-array metadata."""
    storage = witness.vertex_representative_ids.nbytes + witness.vertex_shifts.nbytes
    banks = (*witness.quotient_entities, *witness.retained_quotient_entities)
    storage += 64 + 8 * len(banks)
    for bank in banks:
        storage += 64 + 8 * len(bank)
        storage += bigint_bytes(abs(int(bank.degree)).bit_length())
        storage += bigint_bytes(abs(int(bank.allocator_next_id)).bit_length())
        storage += 2 * (64 + 8 * len(bank.entity_keys))
        for key, identifier in zip(bank.entity_keys, bank.entity_global_ids, strict=True):
            # Key/ID and reverse-ID dictionaries coexist with sorted key lists.
            storage += 2 * (64 + 8 * len(key)) + 128
            storage += bigint_bytes(abs(int(identifier)).bit_length())
            storage += sum(bigint_bytes(abs(int(value)).bit_length()) for value in key)
    return storage


def _periodic_bind_stage_preparation_bounds(
    source: cell_mesh_owner.CellMesh,
    point_count: int,
    shifts: np.ndarray,
    target_geometry: CellGeometrySpec | None,
    witness: PeriodicVertexOrbitWitness,
    /,
) -> tuple[int, int, int]:
    """Bound the two descriptor constructors and their source-provenance preparation."""
    topology = require_periodic_topology(source)
    topology_work, topology_bytes, topology_queries = periodic_preparation_bounds(
        topology.cell,
        point_count,
        shifts,
        actual_geometry=target_geometry,
        stabilizers=True,
    )
    work, storage, queries = (2 * topology_work, 2 * topology_bytes, 2 * topology_queries)
    storage += _periodic_replay_identity_storage_upper(witness)
    # The target source is prepared by each constructor; the source lift
    # check or nonnested provenance also prepares its source banks.
    geometries: list[CellGeometrySpec] = []
    if target_geometry is not None:
        geometries.extend((target_geometry, target_geometry))
    authority = witness.nonnested_geometry
    if authority is not None:
        geometries.extend((authority.source_geometry, authority.target_geometry))
    elif topology.actual_geometry is not None:
        geometries.append(topology.actual_geometry)
    for geometry in geometries:
        geometry_work, geometry_bytes = cell_geometry_preparation_bounds(geometry)
        work += geometry_work
        storage += geometry_bytes
        queries += cell_geometry_preparation_query_upper(geometry)
    return work, storage, queries


def periodic_topology_bind_preparation_bounds(
    source: cell_mesh_owner.CellMesh,
    target: cell_mesh_owner.CellMesh,
    witness: PeriodicVertexOrbitWitness,
    /,
) -> tuple[int, int, int]:
    """Return host work/storage/query bounds for a all required periodic evidence replay.

    Each allocation-prior stage is replayed once before the current target.
    The bound reads only existing source, shift and descriptor metadata."""
    work = storage = queries = 0
    current_target, current_witness = target, witness
    seen: set[int] = set()
    while True:
        if id(current_witness) in seen:
            raise ValueError(
                "periodic_topology_bind_preparation_bounds.source must retain an acyclic prior allocation chain; got A periodic allocation prior contains a cyclic evidence chain."
            )
        seen.add(id(current_witness))
        descriptor = current_target.periodic_topology
        storage += model_array_bytes(current_target)
        geometry = None if descriptor is None else descriptor.actual_geometry
        stage_work, stage_bytes, stage_queries = _periodic_bind_stage_preparation_bounds(
            source,
            current_target.coordinates.shape[0],
            np.asarray(current_witness.vertex_shifts),
            geometry,
            current_witness,
        )
        work += stage_work
        storage += stage_bytes
        queries += stage_queries
        if current_witness.allocation_prior is None:
            return work, storage, queries
        current_target, current_witness = current_witness.allocation_prior


def topology_edit_assembly_preparation_bounds(
    source: cell_mesh_owner.CellMesh, edit: CellTopologyEdit, /
) -> tuple[int, int, int]:
    """Return host preparation bounds for target construction and periodic binding.

    Target preparation builds twice. Each attached descriptor rebuild renews
    its source and runs two constructors; evidence binding then replays the
    allocation-prior chain and two constructors for the current target.
    Nonperiodic assembly has no charged coordinate preparation."""
    work = storage = queries = 0
    storage += model_array_bytes(edit)
    descriptor = edit.target_periodic_topology
    geometry = None if descriptor is None else descriptor.actual_geometry
    if descriptor is not None:
        rebuild_work, rebuild_bytes, rebuild_queries = periodic_preparation_bounds(
            descriptor.cell,
            edit.coordinates.shape[0],
            np.asarray(descriptor.vertex_shifts),
            actual_geometry=geometry,
            stabilizers=True,
        )
        # Each rebuild has a source renewal and two constructor passes.
        work += 2 * 3 * rebuild_work
        storage += 2 * 3 * rebuild_bytes
        queries += 2 * 3 * rebuild_queries
        if geometry is not None:
            geometry_work, geometry_bytes = cell_geometry_preparation_bounds(geometry)
            work += 2 * 3 * geometry_work
            storage += 2 * 3 * geometry_bytes
            queries += 2 * 3 * cell_geometry_preparation_query_upper(geometry)
    witness = edit.periodic_orbits
    if witness is not None:
        stage_work, stage_bytes, stage_queries = _periodic_bind_stage_preparation_bounds(
            source,
            edit.coordinates.shape[0],
            np.asarray(witness.vertex_shifts),
            geometry,
            witness,
        )
        work += stage_work
        storage += stage_bytes
        queries += stage_queries
        if witness.allocation_prior is not None:
            previous, previous_witness = witness.allocation_prior
            prior_work, prior_bytes, prior_queries = (
                periodic_topology_bind_preparation_bounds(
                    source,
                    previous,
                    previous_witness,
                )
            )
            work += prior_work
            storage += prior_bytes
            queries += prior_queries
    return work, storage, queries


@checked
def bind_periodic_topology_edit(
    source: cell_mesh_owner.CellMesh,
    target: cell_mesh_owner.CellMesh,
    witness: PeriodicVertexOrbitWitness,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> cell_mesh_owner.CellMesh:
    """Validate explicit target-global-ID orbits and keyed scientific identities."""

    current = (
        None
        if witness.allocation_prior is None
        else _validated_periodic_allocation_prior(
            source, witness.allocation_prior, budget=budget
        )
    )
    return _bind_periodic_target(source, target, witness, current, budget=budget)


def prepare_periodic_edit_target(
    source: cell_mesh_owner.CellMesh,
    target: cell_mesh_owner.CellMesh,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> cell_mesh_owner.CellMesh:
    """Stage deterministic quotient IDs before preparing the overlap certificate."""

    original = require_periodic_topology(source)
    descriptor = require_periodic_topology(target)
    if identification_id(original.cell) != identification_id(descriptor.cell):
        raise ValueError(
            "prepare_periodic_edit_target.source must preserve the source identification group; got A staged periodic edit changes its identification group."
        )
    banks = _periodic_target_identity_banks(source, descriptor, None, budget=budget)
    ids = {}
    cursors = {}
    for bank in banks:
        known = dict(zip(bank.entity_keys, bank.entity_global_ids, strict=True))
        ids[bank.degree] = np.asarray(
            [known[key] for key in descriptor.entity_keys(bank.degree)], dtype=np.int64
        )
        cursors[bank.degree] = bank.allocator_next_id
    plain = cell_mesh_owner.CellMesh(
        target.coordinates,
        target.blocks,
        vertex_global_ids=target.vertex_global_ids,
        entity_global_ids={
            degree: target.entity_set(degree).entity_ids
            for degree in range(1, target.topological_dimension)
        },
        polyhedral_connectivity=target.connectivity
        if isinstance(target.connectivity, PolyhedralConnectivity)
        else None,
        numeric_version=target.numeric_version,
    )
    staged = periodic_topology_owner.PeriodicMeshTopology(
        plain,
        descriptor.cell,
        descriptor.vertex_representatives,
        descriptor.vertex_shifts,
        actual_geometry=descriptor.actual_geometry,
        entity_global_ids=ids,
        entity_allocator_next_ids=cursors,
        budget=budget,
    )
    return cell_mesh_owner.CellMesh(
        plain.coordinates,
        plain.blocks,
        vertex_global_ids=plain.vertex_global_ids,
        entity_global_ids={
            degree: plain.entity_set(degree).entity_ids
            for degree in range(1, plain.topological_dimension)
        },
        polyhedral_connectivity=plain.connectivity
        if isinstance(plain.connectivity, PolyhedralConnectivity)
        else None,
        periodic_topology=staged,
        numeric_version=plain.numeric_version,
    )


def _periodic_target_lift_authority(
    source: periodic_topology_owner.PeriodicMeshTopology,
    target: cell_mesh_owner.CellMesh,
    nonnested_geometry: PeriodicNonnestedGeometryAuthority | None = None,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[cell_mesh_owner.CellMesh, CellGeometrySpec | None]:
    """Retain the producer's real successor source before stripping its descriptor."""

    descriptor = target.periodic_topology
    geometry = descriptor.actual_geometry if descriptor is not None else None
    if nonnested_geometry is not None:
        require_periodic_nonnested_geometry(
            source, target, nonnested_geometry, budget=budget
        )
    if source.actual_geometry is not None and nonnested_geometry is None:
        if geometry is None:
            raise ValueError(
                "periodic_target_lift_authority.source must retain target geometry provenance and acyclic predecessor sources; got A rational periodic edit requires its target geometry source authority."
            )
        original = source.actual_geometry.exact_source
        current = geometry.exact_source
        source_types = (
            ExactPowerCellGeometrySource,
            ExactPowerCellGeometryRestrictionSource,
            ExactPowerCellGeometryLinearActionSource,
        )
        if not isinstance(original, source_types) or not isinstance(
            current, source_types
        ):
            raise TypeError(
                "periodic_target_lift_authority.source must retain target geometry provenance and acyclic predecessor sources; got Rational periodic edits require deterministic vertex source owners."
            )
        source.actual_geometry.source_coordinates(budget=budget)
        visited: set[int] = set()
        while current.source_id != original.source_id:
            if id(current) in visited:
                raise ValueError(
                    "periodic_target_lift_authority.source must retain target geometry provenance and acyclic predecessor sources; got Rational periodic successor source ancestry contains a cycle."
                )
            visited.add(id(current))
            if isinstance(current, ExactPowerCellGeometrySource):
                raise ValueError(
                    "periodic_target_lift_authority.source must retain target geometry provenance and acyclic predecessor sources; got Rational periodic successor does not retain its predecessor source."
                )
            parent = current.parent
            if not isinstance(parent, source_types):
                raise TypeError(
                    "periodic_target_lift_authority.source must retain target geometry provenance and acyclic predecessor sources; got Rational periodic successor ancestry omits a deterministic source owner."
                )
            current = parent
    if descriptor is not None:
        target = cell_mesh_owner.CellMesh(
            target.coordinates,
            target.blocks,
            vertex_global_ids=target.vertex_global_ids,
            entity_global_ids={
                degree: target.entity_set(degree).entity_ids
                for degree in range(1, target.topological_dimension)
            },
            numeric_version=target.numeric_version,
            polyhedral_connectivity=target.connectivity
            if isinstance(target.connectivity, PolyhedralConnectivity)
            else None,
        )
    return target, geometry


def _bind_periodic_target(
    source: cell_mesh_owner.CellMesh,
    target: cell_mesh_owner.CellMesh,
    witness: PeriodicVertexOrbitWitness,
    allocation_base: tuple[PeriodicEntityIdentityBank, ...] | None,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> cell_mesh_owner.CellMesh:
    """Bind one target after its caller validates the prior allocation chain."""

    if witness.nonnested_geometry is not None:
        require_periodic_nonnested_source(source, witness.nonnested_geometry)
    topology = require_periodic_topology(source)
    if (
        witness.source_topology_id != source.topology_id
        or witness.identification_id != identification_id(topology.cell)
    ):
        raise ValueError(
            "bind_periodic_target.source must retain scientific vertex orbits and all entity allocation banks; got Periodic orbit evidence does not bind source topology/identification."
        )
    identifiers = np.asarray(target.vertex_global_ids)
    representative_ids = np.asarray(witness.vertex_representative_ids)
    shifts = np.asarray(witness.vertex_shifts)
    if representative_ids.shape != identifiers.shape or not np.issubdtype(
        representative_ids.dtype, np.integer
    ):
        raise ValueError(
            "bind_periodic_target.source must retain scientific vertex orbits and all entity allocation banks; got Periodic representatives must be global integer IDs per target vertex row."
        )
    if shifts.shape != (identifiers.size, topology.cell.rank) or not np.issubdtype(
        shifts.dtype, np.integer
    ):
        raise ValueError(
            "bind_periodic_target.source must retain scientific vertex orbits and all entity allocation banks; got Periodic image exponents must be integer target-row lattice shifts."
        )
    lookup = {int(identifier): row for row, identifier in enumerate(identifiers)}
    if any(int(identifier) not in lookup for identifier in representative_ids):
        raise ValueError(
            "bind_periodic_target.source must retain scientific vertex orbits and all entity allocation banks; got A periodic representative is not a global target vertex."
        )
    rows = np.asarray(
        [lookup[int(identifier)] for identifier in representative_ids], dtype=np.int64
    )
    original_ids = np.asarray(source.vertex_global_ids)
    old_roots = original_ids[np.asarray(topology.vertex_representatives)]
    original_lookup = dict(zip(original_ids.tolist(), old_roots.tolist(), strict=True))
    for vertex, identifier in enumerate(identifiers):
        if (
            int(identifier) in original_lookup
            and representative_ids[vertex] != original_lookup[int(identifier)]
        ):
            raise ValueError(
                "bind_periodic_target.source must retain scientific vertex orbits and all entity allocation banks; got A periodic edit changes an old vertex's scientific quotient membership."
            )
    old_representatives = set(old_roots.tolist())
    for representative in np.unique(representative_ids):
        members = identifiers[representative_ids == representative]
        if representative not in old_representatives and representative != np.min(
            members
        ):
            raise ValueError(
                "bind_periodic_target.source must retain scientific vertex orbits and all entity allocation banks; got A born quotient vertex must use its deterministic global representative."
            )
    target, actual_geometry = _periodic_target_lift_authority(
        topology, target, witness.nonnested_geometry, budget=budget
    )
    probe = periodic_topology_owner.PeriodicMeshTopology(
        target,
        topology.cell,
        rows,
        shifts,
        actual_geometry=actual_geometry,
        budget=budget,
    )
    if witness.allocation_prior is None:
        current = _periodic_identity_banks(source, witness.history)
    else:
        if (
            allocation_base is None
            or witness.retained_quotient_entities != allocation_base
        ):
            raise ValueError(
                "bind_periodic_target.source must retain scientific vertex orbits and all entity allocation banks; got Retained periodic identities must equal the validated prior all required allocation bank."
            )
        current = allocation_base
    expected = _allocate_periodic_identity_banks(
        probe, current, witness.retained_quotient_entities
    )
    if tuple(bank.degree for bank in witness.quotient_entities) != tuple(
        bank.degree for bank in expected
    ):
        raise ValueError(
            "bind_periodic_target.source must retain scientific vertex orbits and all entity allocation banks; got A periodic evidence omits quotient scientific identity banks."
        )
    entity_ids, cursors = {}, {}
    for actual, bank in zip(witness.quotient_entities, expected, strict=True):
        require_periodic_identity_bank(actual)
        if (
            actual.allocator_next_id != bank.allocator_next_id
            or actual.entity_keys != bank.entity_keys
            or actual.entity_global_ids != bank.entity_global_ids
        ):
            raise ValueError(
                "bind_periodic_target.source must retain scientific vertex orbits and all entity allocation banks; got Periodic live/restored/retired identities or allocator high-water disagree."
            )
        known = dict(zip(bank.entity_keys, bank.entity_global_ids, strict=True))
        entity_ids[bank.degree] = np.asarray(
            [known[key] for key in probe.entity_keys(bank.degree)], dtype=np.int64
        )
        cursors[bank.degree] = bank.allocator_next_id
    periodic = periodic_topology_owner.PeriodicMeshTopology(
        target,
        topology.cell,
        rows,
        shifts,
        entity_global_ids=entity_ids,
        entity_allocator_next_ids=cursors,
        actual_geometry=actual_geometry,
        budget=budget,
    )

    return cell_mesh_owner.CellMesh(
        target.coordinates,
        target.blocks,
        vertex_global_ids=target.vertex_global_ids,
        entity_global_ids={
            degree: target.entity_set(degree).entity_ids
            for degree in range(1, target.topological_dimension)
        },
        numeric_version=target.numeric_version,
        periodic_topology=periodic,
        polyhedral_connectivity=target.connectivity
        if isinstance(target.connectivity, PolyhedralConnectivity)
        else None,
    )


def periodic_vertex_orbit_witness(
    source: cell_mesh_owner.CellMesh,
    target: cell_mesh_owner.CellMesh,
    representative_ids: np.ndarray,
    exponents: np.ndarray,
    history: PeriodicRefinement | None,
    /,
    *,
    retained_quotient_entities: tuple[PeriodicEntityIdentityBank, ...] = (),
    allocation_prior: tuple[cell_mesh_owner.CellMesh, PeriodicVertexOrbitWitness]
    | None = None,
    nonnested_geometry: PeriodicNonnestedGeometryAuthority | None = None,
    budget: CoordinateEnclosureBudget,
) -> PeriodicVertexOrbitWitness:
    """Bind a producer's explicit target orbits to scientific identity banks.

    ``representative_ids`` names the global representative vertex of every
    target vertex row of the unbound ``target`` carrier; ``exponents`` holds its
    integer group exponents relative to that representative.
    Advanced retained banks require ``allocation_prior``: the real bound prior
    target and its deterministic evidence, recursively validated against ``source``."""

    if nonnested_geometry is not None:
        require_periodic_nonnested_source(source, nonnested_geometry)
    topology = require_periodic_topology(source)
    target, actual_geometry = _periodic_target_lift_authority(
        topology, target, nonnested_geometry, budget=budget
    )
    ids = np.asarray(target.vertex_global_ids)
    representatives = np.asarray(representative_ids)
    shifts = np.asarray(exponents)
    if representatives.shape != ids.shape or not np.issubdtype(
        representatives.dtype, np.integer
    ):
        raise ValueError(
            "periodic_vertex_orbit_witness.source must bind global representatives and integer group exponents; got Periodic representatives must be global integer IDs per target vertex row."
        )
    if shifts.shape != (ids.size, topology.cell.rank) or not np.issubdtype(
        shifts.dtype, np.integer
    ):
        raise ValueError(
            "periodic_vertex_orbit_witness.source must bind global representatives and integer group exponents; got Periodic image exponents must be integer target-row group exponents."
        )
    indices = {int(identifier): row for row, identifier in enumerate(ids)}
    if any(int(identifier) not in indices for identifier in representatives):
        raise ValueError(
            "periodic_vertex_orbit_witness.source must bind global representatives and integer group exponents; got A periodic representative is not a global target vertex."
        )
    local_roots = np.asarray(
        [indices[int(identifier)] for identifier in representatives], dtype=np.int64
    )
    probe = periodic_topology_owner.PeriodicMeshTopology(
        target,
        topology.cell,
        local_roots,
        shifts,
        actual_geometry=actual_geometry,
        budget=budget,
    )
    banks = _periodic_target_identity_banks(
        source,
        probe,
        history,
        retained_quotient_entities,
        allocation_prior,
        budget=budget,
    )
    return PeriodicVertexOrbitWitness(
        source.topology_id,
        identification_id(topology.cell),
        representatives.astype(np.int64),
        shifts.astype(np.int64),
        banks,
        history,
        retained_quotient_entities,
        allocation_prior,
        nonnested_geometry,
    )


def _periodic_orbit_witness(
    source: cell_mesh_owner.CellMesh,
    geometric: cell_mesh_owner.CellMesh,
    target: cell_mesh_owner.CellMesh,
    history: PeriodicRefinement | None,
    source_rows: np.ndarray,
    *,
    budget: CoordinateEnclosureBudget,
) -> PeriodicVertexOrbitWitness:
    """Lower a real construction's orbit map onto global target identities."""

    topology = require_periodic_topology(source)
    geometry = require_periodic_topology(geometric)
    orbit = np.asarray(geometry.orbits(0)[0])[source_rows]
    shifts = np.asarray(geometry.vertex_shifts)[source_rows]
    ids = np.asarray(target.vertex_global_ids)
    old_ids = np.asarray(source.vertex_global_ids)[
        np.asarray(topology.orbit_representatives(0))
    ]
    representatives = np.empty(ids.size, dtype=np.int64)
    exponents = np.empty(shifts.shape, dtype=np.int64)
    for group in np.unique(orbit):
        members = np.flatnonzero(orbit == group)
        retained = members[np.isin(ids[members], old_ids)]
        root = (
            int(retained[0]) if retained.size else int(members[np.argmin(ids[members])])
        )
        representatives[members] = ids[root]
        exponents[members] = reduce_periodic_exponents(
            shifts[members] - shifts[root], identification_orders(topology.cell)
        )
    return periodic_vertex_orbit_witness(
        source, target, representatives, exponents, history, budget=budget
    )


class PeriodicTopologyEditStage(NamedTuple):
    """Immutable all required candidate; accepted state is never mutated by staging."""

    mesh: cell_mesh_owner.CellMesh
    lineage: MeshLineage
    stencil: VertexInterpolationStencil | None
    embedding: PeriodicEmbeddingEvidence
    quotient: PeriodicQuotientEvidence


def _require_complete_periodic_entity_changes(
    source: cell_mesh_owner.CellMesh,
    target: cell_mesh_owner.CellMesh,
    edit: CellTopologyEdit,
    /,
    *,
    scope: NativeExecutionScope,
    budget: CoordinateEnclosureBudget,
) -> None:

    periodic = require_periodic_topology(source)
    source_ids = np.asarray(source.vertex_global_ids)
    target_ids = np.asarray(target.vertex_global_ids)
    source_coordinates, target_coordinates = (
        np.asarray(source.coordinates),
        np.asarray(target.coordinates),
    )
    by_id = {int(identifier): row for row, identifier in enumerate(target_ids)}
    changed = np.asarray(
        [
            int(identifier) not in by_id
            or not np.array_equal(
                source_coordinates[row], target_coordinates[by_id[int(identifier)]]
            )
            for row, identifier in enumerate(source_ids)
        ],
        dtype=np.bool_,
    )
    for degree in range(source.topological_dimension):
        if degree:
            present = set(map(tuple, entity_keys(target, degree)))
            changed = np.asarray(
                [tuple(key) not in present for key in entity_keys(source, degree)],
                dtype=np.bool_,
            )
        orbit = np.asarray(periodic.orbits(degree)[0])
        for group in np.unique(orbit[changed]):
            if not np.all(changed[orbit == group]):
                raise ValueError(
                    "require_complete_periodic_entity_changes.source must preserve whole entity orbits and physical-boundary coverage; got A periodic transaction changes only part of a scientific entity orbit."
                )
    degree = source.topological_dimension - 1
    before = np.asarray(periodic.quotient.entities(degree).subset("boundary").mask)
    source_boundary = before[np.asarray(periodic.orbits(degree)[0])]
    source_keys = entity_keys(source, degree)
    allowed = set(map(tuple, source_keys[source_boundary]))
    relation = edit.relations[degree]
    for old, new in zip(relation.source_keys, relation.target_keys, strict=True):
        if tuple(old) in allowed:
            allowed.add(tuple(new))
    successor = require_periodic_topology(target)
    after = np.asarray(successor.quotient.entities(degree).subset("boundary").mask)
    target_boundary = after[np.asarray(successor.orbits(degree)[0])]
    if any(
        tuple(key) not in allowed for key in entity_keys(target, degree)[target_boundary]
    ):
        raise ValueError(
            "require_complete_periodic_entity_changes.source must preserve whole entity orbits and physical-boundary coverage; got A periodic transaction creates an unbound physical boundary or partial seam."
        )
    if np.any(before):
        source_corners = simplex_entity_corners(source, degree)
        target_corners = simplex_entity_corners(target, degree)
        source_points = source_coordinates[source_corners]
        target_points = target_coordinates[target_corners]
        if degree == 1:
            source_measures = np.linalg.norm(
                source_points[:, 1] - source_points[:, 0], axis=1
            )
            target_measures = np.linalg.norm(
                target_points[:, 1] - target_points[:, 0], axis=1
            )
        else:
            source_measures = 0.5 * np.linalg.norm(
                np.cross(
                    source_points[:, 1] - source_points[:, 0],
                    source_points[:, 2] - source_points[:, 0],
                ),
                axis=1,
            )
            target_measures = 0.5 * np.linalg.norm(
                np.cross(
                    target_points[:, 1] - target_points[:, 0],
                    target_points[:, 2] - target_points[:, 0],
                ),
                axis=1,
            )
        old_leaders = np.asarray(periodic.orbit_representatives(degree))
        new_leaders = np.asarray(successor.orbit_representatives(degree))
        covered = np.zeros(target_boundary.shape, dtype=np.bool_)
        for group in np.flatnonzero(before):
            matched = feature_entity_mask(
                source,
                target,
                np.asarray([group], dtype=np.int64),
                degree,
                scope=scope,
                budget=budget,
            )
            covered |= matched
            measured = float(
                np.sum(target_measures[new_leaders[matched[new_leaders] & after]])
            )
            expected = float(source_measures[old_leaders[group]])
            if abs(measured - expected) > 1.0e-10 * max(1.0, expected):
                raise ValueError(
                    "require_complete_periodic_entity_changes.source must preserve whole entity orbits and physical-boundary coverage; got Periodic physical-boundary coverage changed in the staged transaction."
                )
        if np.any(target_boundary & ~covered):
            raise ValueError(
                "require_complete_periodic_entity_changes.source must preserve whole entity orbits and physical-boundary coverage; got Periodic physical boundary moved away from its represented source."
            )


def stage_periodic_topology_edit(
    source: CellMeshingResult,
    edit: CellTopologyEdit,
    /,
    *,
    numeric_version: str,
    maximum_images: int,
    maximum_pairs: int,
    budget: CoordinateEnclosureBudget,
    scope: NativeExecutionScope,
) -> PeriodicTopologyEditStage:
    """Certify a all required private orbit candidate before any accepted/native commit."""

    with _periodic_phase(
        scope, budget, budget.maximum_work_units, budget.maximum_memory_bytes
    ):
        adaptation_owner.require_affine_geometry(source, budget=budget)
        if edit.periodic_orbits is None:
            raise ValueError(
                "stage_periodic_topology_edit.source must preserve entity orbits and source material coverage; got A periodic candidate omits its all required scientific orbit evidence."
            )
        target, lineage, stencil = assemble_topology_edit(
            source.mesh, edit, numeric_version=numeric_version, budget=budget
        )
        _require_complete_periodic_entity_changes(
            source.mesh, target, edit, scope=scope, budget=budget
        )
        embedding = certify_periodic_embedding(
            target,
            maximum_images=maximum_images,
            maximum_pairs=maximum_pairs,
            budget=budget,
            scope=scope,
        )
        old = periodic_measures_owner.periodic_orbit_measures(source.mesh)
        new = periodic_measures_owner.periodic_orbit_measures(target)
        if abs(float(old.total_measure) - float(new.total_measure)) > 1.0e-10 * float(
            old.total_measure
        ):
            raise ValueError(
                "stage_periodic_topology_edit.source must preserve entity orbits and source material coverage; got A periodic staged transaction changes one-orbit domain coverage."
            )
        source_ids = np.asarray(
            source.mesh.entity_set(source.mesh.topological_dimension).entity_ids
        )
        target_ids = np.asarray(
            target.entity_set(target.topological_dimension).entity_ids
        )
        relation = edit.relations[source.mesh.topological_dimension]
        origins: dict[int, set[int]] = {}
        for before, after in zip(
            relation.source_keys[:, 0], relation.target_keys[:, 0], strict=True
        ):
            origins.setdefault(int(after), set()).add(int(before))
        source_id_set = set(source_ids.tolist())
        for identifier in target_ids:
            if int(identifier) in source_id_set:
                origins.setdefault(int(identifier), set()).add(int(identifier))
        regions = [zone for zone in source.zones if zone.role is MeshZoneRole.REGION]
        if regions:
            assignment: dict[int, int] = {}
            for region, zone in enumerate(regions):
                mask = np.asarray(resolve_mesh_scope(source.mesh, zone.scope).mask)
                for identifier in source_ids[mask]:
                    if int(identifier) in assignment:
                        raise ValueError(
                            "stage_periodic_topology_edit.source must preserve entity orbits and source material coverage; got Source periodic material regions are not exclusive."
                        )
                    assignment[int(identifier)] = region
            target_assignment = []
            for identifier in target_ids:
                materials = {
                    assignment[parent]
                    for parent in origins.get(int(identifier), ())
                    if parent in assignment
                }
                if len(materials) != 1:
                    raise ValueError(
                        "stage_periodic_topology_edit.source must preserve entity orbits and source material coverage; got A periodic candidate has unresolved or contradictory source material ancestry."
                    )
                target_assignment.append(next(iter(materials)))
            dimension = source.mesh.topological_dimension
            source_orbits = np.asarray(
                require_periodic_topology(source.mesh).orbits(dimension)[0],
                dtype=np.int64,
            )
            target_orbits = np.asarray(
                require_periodic_topology(target).orbits(dimension)[0], dtype=np.int64
            )
            for region in range(len(regions)):
                source_mask = np.asarray(
                    [
                        assignment.get(int(identifier), -1) == region
                        for identifier in source_ids
                    ],
                    dtype=np.bool_,
                )
                target_mask = np.asarray(target_assignment, dtype=np.int64) == region
                # Each quotient cell is measured once, whatever its lifted multiplicity.
                before = float(
                    np.sum(
                        np.asarray(old.orbit_measures)[
                            np.unique(source_orbits[source_mask])
                        ]
                    )
                )
                after = float(
                    np.sum(
                        np.asarray(new.orbit_measures)[
                            np.unique(target_orbits[target_mask])
                        ]
                    )
                )
                if abs(before - after) > 1.0e-10 * max(1.0, before):
                    raise ValueError(
                        "stage_periodic_topology_edit.source must preserve entity orbits and source material coverage; got A periodic staged transaction changes source material coverage."
                    )
        return PeriodicTopologyEditStage(
            target, lineage, stencil, embedding, PeriodicQuotientEvidence(target)
        )


def validate_periodic_bisection(
    prepared: adaptation_owner.PreparedMeshAdaptation,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> None:
    """Admit rational affine orbit refinement and retained-epoch coarsening."""

    request, source = prepared.request, prepared.source.mesh
    if (
        not isinstance(request, adaptation_owner.MarkedMeshAdaptation)
        or source.periodic_topology is None
    ):
        raise TypeError(
            "validate_periodic_bisection.prepared must retain marked simplex sources and current coarsening history; got Periodic bisection requires a marked periodic source."
        )
    adaptation_owner.require_affine_geometry(prepared.source, budget=budget)
    _leaf_phydrax_meshing__host_banks.check_marked_source(prepared.source, request)
    if any(
        block.cell_kind != _SIMPLEX_KINDS.get(source.topological_dimension)
        for block in source.blocks
    ):
        raise ValueError(
            "validate_periodic_bisection.prepared must retain marked simplex sources and current coarsening history; got Periodic orbit bisection requires a all required simplex family."
        )
    if request.hierarchy is not None:
        history = request.hierarchy
        if not isinstance(history, PeriodicRefinement):
            raise TypeError(
                "validate_periodic_bisection.prepared must retain marked simplex sources and current coarsening history; got Periodic bisection history must be a PeriodicRefinement."
            )
        if history.mesh.topology_id != source.topology_id or not np.array_equal(
            history.mesh.coordinates, source.coordinates
        ):
            raise ValueError(
                "validate_periodic_bisection.prepared must retain marked simplex sources and current coarsening history; got Periodic coarsening requires its current retained orbit-refinement history."
            )
    if np.asarray(request.coarsen_cell_ids).size and request.hierarchy is None:
        raise ValueError(
            "validate_periodic_bisection.prepared must retain marked simplex sources and current coarsening history; got Periodic coarsening needs the retained source-epoch history."
        )


class _BisectionSelection(StrictModule):
    source_ids: np.ndarray
    refine_ids: np.ndarray
    coarsen_ids: np.ndarray
    history: PeriodicRefinement | None
    restored_parents: np.ndarray
    coarsening: bool = eqx.field(static=True)
    rejected_coarsening: np.ndarray
    edges: np.ndarray
    selected: frozenset[int] = eqx.field(static=True)
    rejected_refinement: tuple[int, ...] = eqx.field(static=True)


class _BisectionEvidencePhase(StrictModule):
    evidence: BisectionEvidence
    partial: bool = eqx.field(static=True)


def _prepare_periodic_bisection(
    prepared: adaptation_owner.PreparedMeshAdaptation, budget: CoordinateEnclosureBudget
) -> _BisectionSelection:
    validate_periodic_bisection(prepared, budget=budget)
    source, request = prepared.source.mesh, prepared.request
    if not isinstance(request, adaptation_owner.MarkedMeshAdaptation):
        raise TypeError(
            "prepare_periodic_bisection.prepared must retain a marked source and current refinement history; got Periodic bisection requires a marked request."
        )
    source_cells, source_ids = _periodic_simplex_rows(source)
    refine_ids = np.asarray(request.refine_cell_ids)
    coarsen_ids = np.asarray(request.coarsen_cell_ids)
    history = request.hierarchy
    if history is not None and not isinstance(history, PeriodicRefinement):
        raise TypeError(
            "prepare_periodic_bisection.prepared must retain a marked source and current refinement history; got Periodic bisection history must be a PeriodicRefinement."
        )
    restored_parents = np.empty(0, dtype=np.int64)
    if coarsen_ids.size:
        if history is None:
            raise ValueError(
                "prepare_periodic_bisection.prepared must retain a marked source and current refinement history; got Periodic coarsening has no retained source epoch."
            )
        restored_parents = _periodic_coarsening_parents(prepared, history, budget=budget)
    coarsening = restored_parents.size > 0
    accepted_coarse = (
        source_ids[np.isin(history.parent_cells, restored_parents)]
        if history is not None
        else np.empty(0, dtype=np.int64)
    )
    rejected_coarsening = coarsen_ids[~np.isin(coarsen_ids, accepted_coarse)]
    protected = set(
        np.asarray(require_periodic_topology(source).orbits(1)[0])[
            prepared.constraints.protected_edge_mask
        ].tolist()
    )
    edges = simplex_entity_corners(source, 1)
    orbit = np.asarray(require_periodic_topology(source).orbits(1)[0])
    lengths = np.linalg.norm(
        np.asarray(source.coordinates)[edges[:, 1]]
        - np.asarray(source.coordinates)[edges[:, 0]],
        axis=1,
    )
    by_pair = {tuple(sorted(edge)): row for row, edge in enumerate(edges.tolist())}
    quotient_keys = require_periodic_topology(source).entity_keys(1)
    selected, rejected_refinement = set(), []
    for row in np.flatnonzero(np.isin(source_ids, refine_ids)):
        corners = source_cells[row]
        local = [
            by_pair[tuple(sorted(pair))] for pair in combinations(corners.tolist(), 2)
        ]
        edge = min(
            local, key=lambda index: (-lengths[index], quotient_keys[orbit[index]])
        )
        key = int(orbit[edge])
        if key in protected:
            rejected_refinement.append(int(source_ids[row]))
        else:
            selected.add(key)
    return _BisectionSelection(
        source_ids,
        refine_ids,
        coarsen_ids,
        history,
        restored_parents,
        coarsening,
        rejected_coarsening,
        edges,
        frozenset(selected),
        tuple(rejected_refinement),
    )


def _periodic_bisection_evidence(
    source: cell_mesh_owner.CellMesh,
    selection: _BisectionSelection,
    refinement: PeriodicRefinement | None,
    edit: CellTopologyEdit | None,
    generations: np.ndarray,
    coarsening: bool,
    rejected_coarsening: np.ndarray,
) -> _BisectionEvidencePhase:
    source_ids, refine_ids, coarsen_ids = (
        selection.source_ids,
        selection.refine_ids,
        selection.coarsen_ids,
    )
    rejected_refinement, selected = selection.rejected_refinement, selection.selected
    restored_parents = selection.restored_parents
    evidence = BisectionEvidence(
        requested_refinements=refine_ids.size,
        accepted_refinements=refine_ids.size - len(rejected_refinement),
        rejected_refinement_ids=np.asarray(rejected_refinement, dtype=np.int64),
        admissibility_tests=refine_ids.size,
        bisections=0
        if refinement is None
        else sum(block.cell_count for block in refinement.mesh.blocks) - source_ids.size,
        closure_iterations=int(bool(selected)),
        created_vertices=0
        if refinement is None
        else refinement.mesh.coordinates.shape[0] - source.coordinates.shape[0],
        maximum_generation=int(np.max(generations, initial=0)),
        initially_compatible=None,
        incompatible_facets=0,
        uniform_refinement_applied=False,
        requested_coarsenings=coarsen_ids.size,
        coarsened_vertices=source.coordinates.shape[0] - edit.coordinates.shape[0]
        if coarsening and edit is not None
        else 0,
        coarsening_passes=int(coarsening),
        restored_cells=restored_parents.size if coarsening else 0,
        rejected_coarsening_ids=rejected_coarsening,
    )
    partial = bool(rejected_refinement) or bool(rejected_coarsening.size)
    return _BisectionEvidencePhase(evidence, partial)


class _BisectionCoarseBanks(StrictModule):
    parents: np.ndarray
    reference: np.ndarray
    generations: np.ndarray


class _BisectionHistoryPhase(StrictModule):
    history: PeriodicRefinement | None


def _publish_bisection_history(
    prepared: adaptation_owner.PreparedMeshAdaptation,
    edit: CellTopologyEdit,
    target: cell_mesh_owner.CellMesh,
    history: PeriodicRefinement | None,
    refinement: PeriodicRefinement | None,
    generations: np.ndarray,
    coarse: _BisectionCoarseBanks | None,
    budget: CoordinateEnclosureBudget,
) -> _BisectionHistoryPhase:
    source = prepared.source.mesh
    next_history = None
    edit_ids = np.concatenate(tuple(np.asarray(block.cell_ids) for block in edit.blocks))
    target_ids = np.concatenate(
        tuple(np.asarray(block.global_ids) for block in target.blocks)
    )
    rows_by_id = {int(identifier): row for row, identifier in enumerate(edit_ids)}
    if (
        len(rows_by_id) != edit_ids.size
        or len(set(target_ids.tolist())) != target_ids.size
        or set(rows_by_id) != set(target_ids.tolist())
    ):
        raise ValueError(
            "publish_bisection_history.prepared must preserve the retained child scientific cell bank; got Periodic publication changed the retained child scientific cell bank."
        )
    order = np.fromiter(
        (rows_by_id[int(identifier)] for identifier in target_ids),
        dtype=np.int64,
        count=target_ids.size,
    )
    if coarse is not None and history is not None:
        source_keys = set(require_periodic_topology(history.source).entity_keys(1))
        present_keys = set(require_periodic_topology(target).entity_keys(1))
        next_history = PeriodicRefinement(
            mesh=target,
            source=history.source,
            source_geometry=history.source_geometry,
            parent_cells=coarse.parents[order],
            parent_reference_vertices=coarse.reference[order],
            cell_generations=coarse.generations[order],
            bisected_edges=len(source_keys - present_keys),
            quotient=PeriodicQuotientEvidence(target),
            previous=history.previous,
            retired=history,
            budget=budget,
        )
    if refinement is not None:
        next_history = PeriodicRefinement(
            mesh=target,
            source=source,
            source_geometry=prepared.source.geometry,
            parent_cells=np.asarray(refinement.parent_cells)[order],
            parent_reference_vertices=np.asarray(refinement.parent_reference_vertices)[
                order
            ],
            cell_generations=generations[order],
            bisected_edges=refinement.bisected_edges,
            quotient=PeriodicQuotientEvidence(target),
            previous=history,
            budget=budget,
        )
    return _BisectionHistoryPhase(next_history)


@checked
def execute_periodic_bisection_route(
    prepared: adaptation_owner.PreparedMeshAdaptation,
    /,
    *,
    budget: CoordinateEnclosureBudget,
    scope: NativeExecutionScope,
) -> adaptation_owner.RouteOutcome:

    with _periodic_phase(
        scope, budget, budget.maximum_work_units, budget.maximum_memory_bytes
    ):
        selection = _prepare_periodic_bisection(prepared, budget)
        source, limits = prepared.source.mesh, prepared.policy.limits
        source_ids, coarsen_ids = selection.source_ids, selection.coarsen_ids
        history, restored_parents = selection.history, selection.restored_parents
        coarsening, rejected_coarsening = (
            selection.coarsening,
            selection.rejected_coarsening,
        )
        edges, selected = selection.edges, selection.selected
        refinement = None
        if selected and coarsening:
            rejected_coarsening = coarsen_ids
            coarsening = False
        if selected:
            upper_cells = source_ids.size * (
                4 if source.topological_dimension == 2 else 64
            )
            upper_vertices = source.coordinates.shape[0] + edges.shape[0] * (
                source.topological_dimension + 1
            )
            if (
                upper_cells > limits.maximum_cells
                or upper_vertices > limits.maximum_vertices
                or upper_cells * 128 + upper_vertices * 64 > limits.maximum_scratch_bytes
                or upper_cells > limits.maximum_work_units
            ):
                raise ValueError(
                    "execute_periodic_bisection_route.prepared must fit construction limits and preserve one-orbit measure; got Periodic orbit closure exceeds its declared construction budget."
                )
            refinement = refine_periodic_mesh(
                source,
                edge_orbits=np.asarray(sorted(selected), dtype=np.int64),
                source_geometry=prepared.source.geometry,
                budget=budget,
                scope=scope,
            )
            edit = _periodic_refinement_edit(refinement, history, budget=budget)
            kind = MeshTransitionKind.REFINE
        elif coarsening and history is not None:
            edit, coarse_parents, coarse_reference, coarse_generations = (
                _periodic_coarsening_edit(history, restored_parents, budget=budget)
            )
            kind = MeshTransitionKind.COARSEN
        else:
            edit = None
        generations = (
            np.asarray(refinement.cell_generations)
            + (
                np.asarray(history.cell_generations)[refinement.parent_cells]
                if history is not None
                else 0
            )
            if refinement is not None
            else (
                coarse_generations
                if coarsening
                else np.zeros(source_ids.size, dtype=np.int32)
            )
        )
        result_evidence = _periodic_bisection_evidence(
            source,
            selection,
            refinement,
            edit,
            generations,
            coarsening,
            rejected_coarsening,
        )
        evidence, partial = result_evidence.evidence, result_evidence.partial
        if edit is None:
            return adaptation_owner.unchanged(
                prepared,
                evidence,
                history,
                status=adaptation_owner.MeshAdaptationStatus.PARTIAL
                if partial
                else adaptation_owner.MeshAdaptationStatus.UNCHANGED,
            )
        probe, _, _ = assemble_topology_edit(
            source,
            edit,
            numeric_version=f"adaptation:{prepared.prepared_id}",
            budget=budget,
        )
        certify_periodic_embedding(
            probe,
            maximum_images=limits.maximum_scratch_bytes
            // max(1, probe.coordinates.size * 8),
            maximum_pairs=limits.maximum_work_units,
            budget=budget,
            scope=scope,
        )
        old_measure, new_measure = (
            periodic_measures_owner.periodic_orbit_measures(source).total_measure,
            periodic_measures_owner.periodic_orbit_measures(probe).total_measure,
        )
        if abs(float(old_measure) - float(new_measure)) > 1.0e-10 * float(old_measure):
            raise ValueError(
                "execute_periodic_bisection_route.prepared must fit construction limits and preserve one-orbit measure; got Periodic bisection changed the quotient's one-orbit measure."
            )
        native = adaptation_owner.finalize_native(
            prepared, edit, kind, conservative=not coarsening, scope=scope, budget=budget
        )
        coarse = (
            _BisectionCoarseBanks(coarse_parents, coarse_reference, coarse_generations)
            if coarsening
            else None
        )
        next_history = _publish_bisection_history(
            prepared,
            edit,
            native.target.mesh,
            history,
            refinement,
            generations,
            coarse,
            budget,
        ).history
        return adaptation_owner.RouteOutcome(
            adaptation_owner.MeshAdaptationStatus.PARTIAL
            if partial
            else adaptation_owner.MeshAdaptationStatus.COMPLETE,
            native.target,
            native.transition,
            native.lineage,
            native.stencil,
            native.transfer,
            None,
            evidence,
            next_history,
        )


def periodic_coarsening_witnesses(
    adaptation: adaptation_owner.MeshAdaptationResult, /
) -> NestedReferenceWitnesses | None:
    """Return fine-to-coarse charts in concatenated fine-cell block order.

    Retained source cells use the reference identity. Restored sibling families
    use their recorded dyadic construction charts. This joins scientific cell
    IDs only; it never intersects or nearest-matches independently lifted ghosts."""

    transition = adaptation.transition
    history = adaptation.hierarchy
    if transition is None or transition.transition_kind is not MeshTransitionKind.COARSEN:
        return None
    if not isinstance(history, PeriodicRefinement) or history.retired is None:
        return None
    retired = history.retired
    fine, target = retired.mesh, adaptation.target.mesh
    if fine.topology_id != transition.source_topology_id:
        raise ValueError(
            "periodic_coarsening_witnesses.adaptation must bind reference charts to the source epoch and cover all target cells; got Periodic coarsening witnesses are stale for the source epoch."
        )
    _, fine_ids = _periodic_simplex_rows(fine)
    _, source_ids = _periodic_simplex_rows(retired.source)
    original_ids = source_ids[np.asarray(retired.parent_cells, dtype=np.int64)]
    _, target_ids = _periodic_simplex_rows(target)
    restored = np.isin(original_ids, target_ids)
    coarse_ids = np.where(restored, original_ids, fine_ids)
    if not np.all(np.isin(coarse_ids, target_ids)) or set(coarse_ids.tolist()) != set(
        target_ids.tolist()
    ):
        raise ValueError(
            "periodic_coarsening_witnesses.adaptation must bind reference charts to the source epoch and cover all target cells; got Periodic coarsening reference charts do not cover the target cells."
        )
    dimension = fine.topological_dimension
    identity = np.concatenate((np.zeros((1, dimension)), np.eye(dimension)))
    reference = np.where(
        restored[:, None, None],
        np.asarray(retired.parent_reference_vertices),
        identity[None],
    )
    return NestedReferenceWitnesses(fine_ids, coarse_ids, reference)


__all__ = [
    "ExactPeriodicEdgeSizeEvidence",
    "periodic_exact_edge_size_evidence",
    "PeriodicEmbeddingEvidence",
    "certify_periodic_embedding",
    "PeriodicMeshConstruction",
    "PeriodicPointOrbits",
    "PeriodicQuotientEvidence",
    "PeriodicRefinement",
    "periodic_cell_from_constraints",
    "periodic_delaunay_mesh",
    "publish_periodic_simplices",
    "refine_periodic_mesh",
    "periodic_coarsening_cell_ids",
    "periodic_coarsening_witnesses",
]


# Source-aware metric edits select lifted carriers. Concrete
# Surface/Sphere owners construct every local operation on one private state.


type PeriodicMetricOperation = Literal["split", "collapse", "flip", "relocate"]


class _MetricControllerWork:
    """Attempted controller entity visits, separate from primitive work."""

    def __init__(self, ledger: CoordinateEnclosureBudget) -> None:
        self.ledger = ledger
        self.visits = 0
        self.charged = 0
        self.remaining_work = ledger.maximum_work_units - ledger.work_units

    def visit(self, count: int = 1) -> None:
        if self.visits - self.charged + count > self.remaining_work:
            raise ValueError(
                "visit.count must fit the remaining native controller work allowance; got Periodic controller visits exceed the native work remainder."
            )
        self.ledger.reserve(count)
        self.visits += count

    def flush(self) -> None:

        self.charged = self.visits
        self.remaining_work = self.ledger.maximum_work_units - self.ledger.work_units

    @contextmanager
    def activate(self) -> Iterator[None]:
        try:
            yield
        finally:
            self.flush()


class PeriodicMetricOrbitCarrier(NamedTuple):
    """One source incidence, with rational action from the carrier."""

    entity_global_id: int
    vertex_global_ids: tuple[int, ...]
    incident_cell_global_ids: tuple[int, ...]
    incident_cell_classes: tuple[int, ...]
    group_shift: tuple[int, ...]
    vertex_permutation: tuple[int, ...]
    protected_vertex_global_ids: tuple[int, ...]
    protected_entity: bool


class PeriodicMetricOrbitSelection(NamedTuple):
    """All required winding-sensitive operation support in the immutable source frame."""

    operation: PeriodicMetricOperation
    original_result_id: str
    source_topology_id: str
    source_numeric_version: str
    source_frame_id: str
    source_geometry_id: str
    coordinate_contract_id: str
    original_entity_dimension: int
    original_entity_global_id: int
    original_vertex_global_ids: tuple[int, ...]
    required_orbit_ids: tuple[int, ...]
    quotient_entity_key: tuple[int, ...]
    carriers: tuple[PeriodicMetricOrbitCarrier, ...]


class PeriodicMetricCandidateStage(NamedTuple):
    """aggregate edit returned after every required operation."""

    edit: CellTopologyEdit
    exact_stencils: Mapping[int, ConstructionPointKey]
    executed_entity_ids: tuple[int, ...]


class PeriodicMetricGeometryStage(NamedTuple):
    """full-map successor and its all required source-owned correspondence."""

    target_mesh: cell_mesh_owner.CellMesh
    target_geometry: CellGeometrySpec
    source_membership: tuple[GeometryAssociation, ...]
    source_correspondence: (
        PreparedSurfaceChartDeformation | PreparedSphereChartDeformation
    )
    geometry_transition: CellGeometryTransition


class PeriodicMetricOrbitEvidence(NamedTuple):
    """Host preparation evidence; native ended-scope evidence is attached later."""

    operation: PeriodicMetricOperation
    status: Literal["complete"]
    orbit_copy_count: int
    cavity_cells: int
    controller_work_units: int
    host_storage_upper_bytes: int
    embedding: PeriodicEmbeddingEvidence
    quotient: PeriodicQuotientEvidence


class PeriodicMetricOrbitOutcome(NamedTuple):
    edit: CellTopologyEdit
    target_mesh: cell_mesh_owner.CellMesh
    periodic_witness: PeriodicVertexOrbitWitness
    geometry_stage: PeriodicMetricGeometryStage
    exact_stencils: Mapping[int, ConstructionPointKey]
    source_result_id: str
    source_topology_id: str
    source_numeric_version: str
    source_frame_id: str
    coordinate_contract_id: str
    source_geometry_id: str
    target_numeric_version: str
    target_frame_id: str
    target_geometry_id: str
    policy_id: str
    certificate_limits_id: str
    coordinate_work_units: int
    coordinate_peak_storage_upper_bytes: int
    evidence: PeriodicMetricOrbitEvidence


class _MetricOrbitCell(NamedTuple):
    name: str
    kind: str
    identifier: int
    vertices: tuple[int, ...]


def _metric_orbit_cells(
    mesh: cell_mesh_owner.CellMesh, /, *, controller_work: _MetricControllerWork
) -> tuple[_MetricOrbitCell, ...]:
    ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    cells = []
    for block in mesh.blocks:
        for identifier, vertices in zip(
            np.asarray(block.global_ids), np.asarray(block.vertices), strict=True
        ):
            controller_work.visit()
            cells.append(
                _MetricOrbitCell(
                    block.name,
                    block.cell_kind,
                    int(identifier),
                    tuple(int(value) for value in ids[vertices]),
                )
            )
    return tuple(cells)


def _metric_candidate_cells(
    edit: CellTopologyEdit, /, *, controller_work: _MetricControllerWork
) -> tuple[_MetricOrbitCell, ...]:
    cells = []
    for block in edit.blocks:
        if not isinstance(block, TopologyEditBlock):
            raise ValueError(
                "metric_candidate_cells.edit must retain fixed-family topology edit blocks; got A local metric rewrite requires its fixed-family candidate."
            )
        for identifier, vertices in zip(block.cell_ids, block.cells, strict=True):
            controller_work.visit()
            cells.append(
                _MetricOrbitCell(
                    block.name,
                    block.cell_kind,
                    int(identifier),
                    tuple(int(value) for value in edit.vertex_global_ids[vertices]),
                )
            )
    return tuple(cells)


def _require_metric_stencils(
    source: cell_mesh_owner.CellMesh,
    edit: CellTopologyEdit,
    stencils: Mapping[int, ConstructionPointKey],
    /,
    *,
    controller_work: _MetricControllerWork,
) -> dict[int, ConstructionPointKey]:
    source_ids = set(np.asarray(source.vertex_global_ids).tolist())
    ids = np.asarray(edit.vertex_global_ids)
    if (
        ids.ndim != 1
        or not np.issubdtype(ids.dtype, np.integer)
        or len(set(ids.tolist())) != ids.size
    ):
        raise ValueError(
            "require_metric_stencils.source must retain normalized positive source supports aligned with target IDs; got A metric candidate requires unique integer scientific vertex IDs."
        )
    if set(stencils) != set(ids.tolist()):
        raise ValueError(
            "require_metric_stencils.source must retain normalized positive source supports aligned with target IDs; got Rational metric stencils must cover exactly the private target vertices."
        )
    if (
        edit.stencil_sources.shape != edit.stencil_weights.shape
        or edit.stencil_valid.shape != edit.stencil_sources.shape
        or edit.stencil_sources.shape[0] != ids.size
    ):
        raise ValueError(
            "require_metric_stencils.source must retain normalized positive source supports aligned with target IDs; got Metric candidate interpolation rows do not align with target IDs."
        )
    result = {}
    for row, identifier in enumerate(ids):
        controller_work.visit()
        key = stencils[int(identifier)]
        if not isinstance(key, tuple) or not key:
            raise ValueError(
                "require_metric_stencils.source must retain normalized positive source supports aligned with target IDs; got Rational construction stencils need deterministic positive source supports."
            )
        for vertex, weight in key:
            controller_work.visit()
            if (
                not isinstance(weight, Fraction)
                or weight <= 0
                or vertex not in source_ids
            ):
                raise ValueError(
                    "require_metric_stencils.source must retain normalized positive source supports aligned with target IDs; got Rational construction stencils need deterministic positive source supports."
                )
        if tuple(sorted(key)) != key or len({vertex for vertex, _ in key}) != len(key):
            raise ValueError(
                "require_metric_stencils.source must retain normalized positive source supports aligned with target IDs; got Rational construction stencils need deterministic positive source supports."
            )
        if sum((weight for _, weight in key), Fraction(0)) != 1:
            raise ValueError(
                "require_metric_stencils.source must retain normalized positive source supports aligned with target IDs; got A rational metric construction stencil must preserve constants."
            )
        actual = tuple(
            sorted(
                (int(vertex), float(weight))
                for vertex, weight, valid in zip(
                    edit.stencil_sources[row],
                    edit.stencil_weights[row],
                    edit.stencil_valid[row],
                    strict=True,
                )
                if valid
            )
        )
        if actual != tuple((vertex, float(weight)) for vertex, weight in key):
            raise ValueError(
                "require_metric_stencils.source must retain normalized positive source supports aligned with target IDs; got Rational construction support disagrees with the candidate stencil."
            )
        result[int(identifier)] = key
    return result


class _MetricCavity(StrictModule):
    source_cells: tuple[_MetricOrbitCell, ...]
    target_cells: tuple[_MetricOrbitCell, ...]
    support: frozenset[int] = eqx.field(static=True)


def _metric_orbit_cavity(
    source: cell_mesh_owner.CellMesh,
    edit: CellTopologyEdit,
    /,
    *,
    controller_work: _MetricControllerWork,
) -> _MetricCavity:
    before = _metric_orbit_cells(source, controller_work=controller_work)
    after = _metric_candidate_cells(edit, controller_work=controller_work)
    targets = {cell.identifier: cell for cell in after}
    old_ids = np.asarray(source.vertex_global_ids)
    new_rows = {
        int(identifier): row for row, identifier in enumerate(edit.vertex_global_ids)
    }
    old_points = np.asarray(source.coordinates)
    moved = set()
    for row, identifier in enumerate(old_ids):
        controller_work.visit()
        if int(identifier) not in new_rows or not np.array_equal(
            old_points[row], edit.coordinates[new_rows[int(identifier)]]
        ):
            moved.add(int(identifier))
    selected = []
    for cell in before:
        controller_work.visit()
        if (
            cell.identifier not in targets
            or targets[cell.identifier] != cell
            or moved.intersection(cell.vertices)
        ):
            selected.append(cell)
    cavity = tuple(selected)
    if not cavity:
        raise ValueError(
            "metric_orbit_cavity.source must retain nonempty changed source and target cavities; got A periodic metric candidate must contain a real local edit."
        )
    originals = {cell.identifier: cell for cell in before}
    replacements = []
    for cell in after:
        controller_work.visit()
        if (
            cell.identifier not in originals
            or originals[cell.identifier] != cell
            or moved.intersection(cell.vertices)
        ):
            replacements.append(cell)
    if not replacements:
        raise ValueError(
            "metric_orbit_cavity.source must retain nonempty changed source and target cavities; got A local metric edit has no all required target cavity."
        )
    support = {vertex for cell in cavity for vertex in cell.vertices}
    return _MetricCavity(cavity, tuple(replacements), frozenset(support))


def _metric_protected_rows(
    source: CellMeshingResult,
    policy: adaptation_owner.MeshAdaptationPolicy,
    /,
    *,
    controller_work: _MetricControllerWork,
) -> tuple[set[int], ...]:

    mesh = source.mesh
    periodic = require_periodic_topology(mesh)
    protected: tuple[set[int], ...] = tuple(
        set() for _ in range(mesh.topological_dimension + 1)
    )
    for scope in policy.protected_scopes:
        controller_work.visit()
        selected = _leaf_phydrax_meshing__host_banks.selected_rows(mesh, scope)
        for degree in range(scope.entity_dimension + 1):
            rows = _leaf_phydrax_meshing__host_banks.closure_rows(
                mesh, scope.entity_dimension, selected, degree
            )
            orbit = np.asarray(periodic.orbits(degree)[0])
            controller_work.visit(orbit.size)
            protected[degree].update(np.flatnonzero(np.isin(orbit, orbit[rows])).tolist())
    # A scientific corner or ambiguous membership is fixed under every copy,
    # not just the lifted row on which the source happened to publish it.
    for association in source.associations:
        controller_work.visit()
        if association.target_entity_set_id != mesh.entity_set(0).entity_set_id:
            continue
        rows = association.target_rows(np.asarray(mesh.vertex_global_ids))
        fixed = np.asarray(association.ambiguous)[rows] | (
            np.asarray(association.source_dimensions)[rows] == 0
        )
        orbit = np.asarray(periodic.orbits(0)[0])
        controller_work.visit(orbit.size)
        protected[0].update(np.flatnonzero(np.isin(orbit, orbit[fixed])).tolist())
    fixed = np.any(
        _leaf_phydrax_meshing__host_banks.membership(
            mesh, _leaf_phydrax_meshing__host_banks.organization_scopes(source), 0
        ),
        axis=1,
    )
    orbit = np.asarray(periodic.orbits(0)[0])
    controller_work.visit(orbit.size)
    protected[0].update(np.flatnonzero(np.isin(orbit, orbit[fixed])).tolist())
    transfer = policy.association_transfer
    if isinstance(transfer, SurfaceAssociationTransfer):
        classes = transfer.classes(source)
        fixed = classes[0].dimensions == 0
        controller_work.visit(orbit.size)
        protected[0].update(np.flatnonzero(np.isin(orbit, orbit[fixed])).tolist())
        edge_mask = (
            transfer.protected_edges(
                source,
                midpoint_required=policy.route
                in (
                    adaptation_owner.MeshAdaptationRoute.NATIVE_BISECTION,
                    adaptation_owner.MeshAdaptationRoute.DEVICE_BISECTION,
                    adaptation_owner.MeshAdaptationRoute.NATIVE_MIXED,
                ),
            )
            | ~classes[1].resolved
        )
        edge_orbit = np.asarray(periodic.orbits(1)[0])
        controller_work.visit(edge_orbit.size)
        protected[1].update(
            np.flatnonzero(np.isin(edge_orbit, edge_orbit[edge_mask])).tolist()
        )
    return protected


class _MetricSupport(StrictModule):
    dimension: int = eqx.field(static=True)
    entity_id: int = eqx.field(static=True)
    vertices: tuple[int, ...] = eqx.field(static=True)


class _MetricSupportVertices(StrictModule):
    vertices: tuple[int, ...] = eqx.field(static=True)


def _metric_relocation_support(
    mesh: cell_mesh_owner.CellMesh,
    candidate: CellTopologyEdit,
    old_ids: set[int],
    new_ids: set[int],
    controller_work: _MetricControllerWork,
) -> _MetricSupport:
    coordinates = np.asarray(jax.device_get(mesh.coordinates))
    identifiers = np.asarray(jax.device_get(mesh.vertex_global_ids))
    rows = {
        int(identifier): row for row, identifier in enumerate(candidate.vertex_global_ids)
    }
    changed = []
    for row, identifier in enumerate(identifiers):
        controller_work.visit()
        if int(identifier) in rows and not np.array_equal(
            coordinates[row],
            candidate.coordinates[rows[int(identifier)]],
        ):
            changed.append(int(identifier))
    moved = tuple(changed)
    if len(moved) != 1 or old_ids != new_ids:
        raise ValueError(
            "metric_relocation_support.mesh must move one retained source vertex; got A periodic relocation seed must name one retained source vertex."
        )
    return _MetricSupport(0, moved[0], moved)


def _metric_collapse_support(
    mesh: cell_mesh_owner.CellMesh,
    candidate: CellTopologyEdit,
    old_ids: set[int],
    new_ids: set[int],
    controller_work: _MetricControllerWork,
) -> _MetricSupportVertices:
    removed = old_ids - new_ids
    if len(removed) != 1 or new_ids - old_ids:
        raise ValueError(
            "metric_collapse_support.mesh must identify a unique source edge contraction; got A periodic collapse seed must remove exactly one source vertex."
        )
    vertex = next(iter(removed))
    before = _metric_orbit_cells(mesh, controller_work=controller_work)
    after = _metric_candidate_cells(candidate, controller_work=controller_work)
    target_keys = {(cell.name, cell.kind, tuple(sorted(cell.vertices))) for cell in after}
    neighbors = {
        value for cell in before if vertex in cell.vertices for value in cell.vertices
    }
    successors = set()
    for kept in sorted((neighbors - {vertex}) & new_ids):
        substituted = set()
        for cell in before:
            controller_work.visit()
            values = tuple(kept if value == vertex else value for value in cell.vertices)
            if len(set(values)) == len(values):
                substituted.add((cell.name, cell.kind, tuple(sorted(values))))
        if substituted == target_keys:
            successors.add(kept)
    if len(successors) != 1:
        raise ValueError(
            "metric_collapse_support.mesh must identify a unique source edge contraction; got A collapse seed has no unique source edge contraction evidence."
        )
    support = (vertex, next(iter(successors)))
    return _MetricSupportVertices(support)


def _metric_removed_edge_support(
    mesh: cell_mesh_owner.CellMesh,
    candidate: CellTopologyEdit,
    operation: PeriodicMetricOperation,
    old_ids: set[int],
    new_ids: set[int],
    controller_work: _MetricControllerWork,
) -> _MetricSupportVertices:
    probe = build_mesh(
        candidate, mesh.numeric_version, None, budget=controller_work.ledger
    )
    target_keys = set(map(tuple, entity_keys(probe, 1)))
    removed = []
    for key in entity_keys(mesh, 1):
        controller_work.visit()
        if tuple(key) not in target_keys:
            removed.append(tuple(int(value) for value in key))
    removed_edges = tuple(removed)
    if len(removed_edges) != 1:
        raise ValueError(
            "metric_removed_edge_support.mesh must remove one source edge with the declared vertex birth or retention; got A split or flip seed must remove one source edge."
        )
    if operation == "split" and (len(new_ids - old_ids) != 1 or old_ids - new_ids):
        raise ValueError(
            "metric_removed_edge_support.mesh must remove one source edge with the declared vertex birth or retention; got A split seed must construct one born vertex."
        )
    if operation == "flip" and old_ids != new_ids:
        raise ValueError(
            "metric_removed_edge_support.mesh must remove one source edge with the declared vertex birth or retention; got A flip seed cannot construct or remove source vertices."
        )
    support = removed_edges[0]
    return _MetricSupportVertices(support)


def _metric_original_support(
    mesh: cell_mesh_owner.CellMesh,
    candidate: CellTopologyEdit,
    operation: PeriodicMetricOperation,
    /,
    *,
    controller_work: _MetricControllerWork,
) -> _MetricSupport:
    old_ids = set(np.asarray(mesh.vertex_global_ids).tolist())
    new_ids = set(candidate.vertex_global_ids.tolist())
    match operation:
        case "relocate":
            return _metric_relocation_support(
                mesh, candidate, old_ids, new_ids, controller_work
            )
        case "collapse":
            selected = _metric_collapse_support(
                mesh, candidate, old_ids, new_ids, controller_work
            )
        case "split" | "flip":
            selected = _metric_removed_edge_support(
                mesh, candidate, operation, old_ids, new_ids, controller_work
            )
        case invalid:
            assert_never(invalid)
    support = selected.vertices
    keys = entity_keys(mesh, 1)
    rows = [row for row, key in enumerate(keys) if tuple(key) == tuple(sorted(support))]
    if len(rows) != 1:
        raise ValueError(
            "metric_original_support.mesh must identify one lifted source edge; got The operation support is not one lifted source edge."
        )
    identifier = int(np.asarray(mesh.entity_set(1).entity_ids)[rows[0]])
    return _MetricSupport(1, identifier, support)


def _metric_source_classes(
    source: CellMeshingResult,
    policy: adaptation_owner.MeshAdaptationPolicy,
    /,
    *,
    controller_work: _MetricControllerWork,
) -> dict[int, int]:

    mesh = source.mesh
    classes = _leaf_phydrax_meshing__host_banks.mesh_cell_classes(
        mesh,
        _leaf_phydrax_meshing__host_banks.organization_scopes(source),
        family_identity=True,
    )
    controller_work.visit(classes.size)
    if source.attributes:
        rows = key_rows(
            _leaf_phydrax_meshing__host_banks.mesh_cell_ids(mesh)[:, None],
            entity_keys(mesh, mesh.topological_dimension),
        )
        classes = _leaf_phydrax_meshing__host_banks.row_classes(
            np.column_stack(
                (classes, layer_core_owner.layer_interval_classes(source)[rows])
            )
        )
        controller_work.visit(classes.size)
    if isinstance(policy.association_transfer, SurfaceAssociationTransfer):
        source_classes = policy.association_transfer.classes(source)
        classes = _leaf_phydrax_meshing__host_banks.row_classes(
            np.column_stack((classes, source_classes[mesh.topological_dimension].codes))
        )
        controller_work.visit(classes.size)
    return dict(
        zip(
            entity_keys(mesh, mesh.topological_dimension)[:, 0].tolist(),
            classes.tolist(),
            strict=True,
        )
    )


def _metric_current_classes(
    source: CellMeshingResult,
    mesh: cell_mesh_owner.CellMesh,
    prior: PeriodicMetricOrbitOutcome | None,
    policy: adaptation_owner.MeshAdaptationPolicy,
    /,
    *,
    controller_work: _MetricControllerWork,
) -> dict[int, int]:
    originals = _metric_source_classes(source, policy, controller_work=controller_work)
    if prior is None:
        return originals
    inherited: dict[int, set[int]] = {}
    relations = prior.edit.relations[source.mesh.topological_dimension]
    for old, new in zip(relations.source_keys, relations.target_keys, strict=True):
        controller_work.visit()
        if int(old[0]) not in originals:
            raise ValueError(
                "metric_current_classes.source must retain compatible source cell ancestry and organization; got The private prior stage carries stale cell ancestry."
            )
        inherited.setdefault(int(new[0]), set()).add(originals[int(old[0])])
    result = {}
    for identifier in entity_keys(mesh, mesh.topological_dimension)[:, 0]:
        controller_work.visit()
        value = int(identifier)
        values = inherited.get(value, {originals[value]} if value in originals else set())
        if len(values) != 1:
            raise ValueError(
                "metric_current_classes.source must retain compatible source cell ancestry and organization; got The private current carrier has incomplete or incompatible cell classes."
            )
        result[value] = next(iter(values))
    return result


def _metric_orbit_selection(
    source: CellMeshingResult,
    candidate: CellTopologyEdit,
    operation: PeriodicMetricOperation,
    policy: adaptation_owner.MeshAdaptationPolicy,
    orbits: PeriodicConstructionOrbits,
    mesh: cell_mesh_owner.CellMesh,
    geometry: CellGeometrySpec,
    prior: PeriodicMetricOrbitOutcome | None,
    /,
    *,
    controller_work: _MetricControllerWork,
) -> PeriodicMetricOrbitSelection:
    periodic = require_periodic_topology(mesh)
    support = _metric_original_support(
        mesh, candidate, operation, controller_work=controller_work
    )
    degree, identifier, original = support.dimension, support.entity_id, support.vertices
    ids = np.asarray(mesh.entity_set(degree).entity_ids)
    original_row = int(np.flatnonzero(ids == identifier)[0])
    orbit, _, shifts = (np.asarray(value) for value in periodic.orbits(degree))
    group = int(orbit[original_row])
    rows = np.flatnonzero(orbit == group)
    vertex_ids = np.asarray(mesh.vertex_global_ids)
    corners = (
        np.arange(vertex_ids.size, dtype=np.int32)[:, None]
        if degree == 0
        else simplex_entity_corners(mesh, degree)
    )
    permutations = (
        (np.zeros((1,), dtype=np.int32),) * vertex_ids.size
        if degree == 0
        else periodic.entity_vertex_permutations(mesh, degree)
    )
    original_corners = tuple(int(value) for value in vertex_ids[corners[original_row]])
    original_positions = tuple(original.index(value) for value in original_corners)
    canonical_to_original = {
        int(canonical): original_positions[slot]
        for slot, canonical in enumerate(permutations[original_row])
    }
    original_protected = _metric_protected_rows(
        source, policy, controller_work=controller_work
    )
    protected: tuple[set[int], ...] = tuple(
        set() for _ in range(mesh.topological_dimension + 1)
    )
    for protected_degree, old_rows in enumerate(original_protected):
        old_keys = entity_keys(source.mesh, protected_degree)
        keys = {tuple(old_keys[row]) for row in old_rows}
        current_keys = entity_keys(mesh, protected_degree)
        current_orbits = np.asarray(periodic.orbits(protected_degree)[0])
        selected_rows = [
            row for row, key in enumerate(current_keys) if tuple(key) in keys
        ]
        controller_work.visit(current_orbits.size)
        protected[protected_degree].update(
            np.flatnonzero(
                np.isin(current_orbits, current_orbits[selected_rows])
            ).tolist()
        )
    cells = _metric_orbit_cells(mesh, controller_work=controller_work)
    class_by_id = _metric_current_classes(
        source, mesh, prior, policy, controller_work=controller_work
    )
    carriers = []
    for row in rows:
        controller_work.visit()
        vertices = tuple(int(value) for value in vertex_ids[corners[row]])
        incident = []
        for cell in cells:
            controller_work.visit()
            if (
                bool(set(vertices).intersection(cell.vertices))
                if operation == "collapse"
                else set(vertices) <= set(cell.vertices)
            ):
                incident.append(cell.identifier)
        incidence = tuple(sorted(incident))
        if not incidence:
            raise ValueError(
                "metric_orbit_selection.source must retain operation support within its periodic source incidence; got An periodic carrier has no source incidence."
            )
        carriers.append(
            PeriodicMetricOrbitCarrier(
                int(ids[row]),
                vertices,
                incidence,
                tuple(class_by_id[value] for value in incidence),
                tuple(
                    int(value)
                    for value in orbits.reduced(shifts[row] - shifts[original_row])
                ),
                tuple(canonical_to_original[int(value)] for value in permutations[row]),
                tuple(
                    sorted(
                        int(vertex_ids[value])
                        for value in corners[row]
                        if int(value) in protected[0]
                    )
                ),
                int(row) in protected[degree],
            )
        )
    carriers.sort(
        key=lambda carrier: (
            carrier.entity_global_id != identifier,
            carrier.group_shift,
            carrier.entity_global_id,
        )
    )
    cavity = _metric_orbit_cavity(
        mesh, candidate, controller_work=controller_work
    ).source_cells
    selected_incidence = set(
        next(
            value.incident_cell_global_ids
            for value in carriers
            if value.entity_global_id == identifier
        )
    )
    if not {cell.identifier for cell in cavity} <= selected_incidence:
        raise ValueError(
            "metric_orbit_selection.source must retain operation support within its periodic source incidence; got A metric seed changes cells outside its source operation incidence."
        )
    return PeriodicMetricOrbitSelection(
        operation,
        source.result_id,
        mesh.topology_id,
        mesh.numeric_version,
        canonical_fingerprint(orbits.source_frame_id),
        cell_geometry_id(geometry),
        source.coordinate_contract.spatial_id,
        degree,
        identifier,
        original,
        (int(np.asarray(periodic.quotient.entities(degree).entity_ids)[group]),),
        periodic.entity_keys(degree)[group],
        tuple(carriers),
    )


class _MetricCandidateValidation(StrictModule):
    exact: dict[int, ConstructionPointKey]
    cavity_count: int = eqx.field(static=True)


@checked
def _require_metric_candidate_stage(
    source: CellMeshingResult,
    current: cell_mesh_owner.CellMesh,
    candidate: CellTopologyEdit,
    stage: PeriodicMetricCandidateStage,
    selection: PeriodicMetricOrbitSelection,
    seed_exact: Mapping[int, ConstructionPointKey],
    /,
    *,
    controller_work: _MetricControllerWork,
) -> _MetricCandidateValidation:
    if not isinstance(stage.edit, CellTopologyEdit):
        raise TypeError(
            "require_metric_candidate_stage.source must execute each required carrier operation with compatible ancestry and coordinates; got The operation owner must return PeriodicMetricCandidateStage."
        )
    required = {carrier.entity_global_id for carrier in selection.carriers}
    if (
        len(stage.executed_entity_ids) != len(required)
        or set(stage.executed_entity_ids) != required
    ):
        raise ValueError(
            "require_metric_candidate_stage.source must execute each required carrier operation with compatible ancestry and coordinates; got The candidate omits or duplicates periodic operations."
        )
    edit = stage.edit
    if (
        edit.operation != candidate.operation
        or edit.refinement is not None
        or edit.coarsening is not None
    ):
        raise ValueError(
            "require_metric_candidate_stage.source must execute each required carrier operation with compatible ancestry and coordinates; got The callback changed the operation or substituted a family template."
        )
    if edit.shared_faces is not None or edit.prescribed_entity_ids:
        raise ValueError(
            "require_metric_candidate_stage.source must execute each required carrier operation with compatible ancestry and coordinates; got The metric callback cannot substitute restored family witnesses."
        )
    if tuple(record.dimension for record in edit.relations) != tuple(
        range(source.mesh.topological_dimension + 1)
    ):
        raise ValueError(
            "require_metric_candidate_stage.source must execute each required carrier operation with compatible ancestry and coordinates; got The aggregate periodic candidate omits all required entity ancestry."
        )
    exact = _require_metric_stencils(
        source.mesh, edit, stage.exact_stencils, controller_work=controller_work
    )
    cavity = _metric_orbit_cavity(
        current, edit, controller_work=controller_work
    ).source_cells
    changed = {cell.identifier for cell in cavity}
    allowed = {
        identifier
        for carrier in selection.carriers
        for identifier in carrier.incident_cell_global_ids
    }
    if not changed <= allowed:
        raise ValueError(
            "require_metric_candidate_stage.source must execute each required carrier operation with compatible ancestry and coordinates; got The aggregate operation changes cells outside its all required incidence."
        )
    for carrier in selection.carriers:
        controller_work.visit()
        if not changed.intersection(carrier.incident_cell_global_ids):
            raise ValueError(
                "require_metric_candidate_stage.source must execute each required carrier operation with compatible ancestry and coordinates; got A purported periodic operation left an carrier unchanged."
            )
    # Compatible neighboring operations may further subdivide seed children.
    # The seed's constructions and moved coordinates remain .
    seed_cavity = _metric_orbit_cavity(
        current, candidate, controller_work=controller_work
    ).source_cells
    seed_ids = {cell.identifier for cell in seed_cavity}
    if not seed_ids <= changed:
        raise ValueError(
            "require_metric_candidate_stage.source must execute each required carrier operation with compatible ancestry and coordinates; got The aggregate operation lost its private source cavity."
        )
    rows = {int(identifier): row for row, identifier in enumerate(edit.vertex_global_ids)}
    original_rows = {
        int(identifier): row for row, identifier in enumerate(current.vertex_global_ids)
    }
    for row, identifier in enumerate(candidate.vertex_global_ids):
        controller_work.visit()
        value = int(identifier)
        constructed = value not in original_rows or not np.array_equal(
            candidate.coordinates[row],
            np.asarray(current.coordinates)[original_rows[value]],
        )
        if constructed and (
            value not in rows
            or exact[value] != seed_exact[value]
            or not np.array_equal(
                candidate.coordinates[row], edit.coordinates[rows[value]]
            )
        ):
            raise ValueError(
                "require_metric_candidate_stage.source must execute each required carrier operation with compatible ancestry and coordinates; got Overlapping periodic operations prescribe incompatible seed constructions or coordinates."
            )
    return _MetricCandidateValidation(exact, len(changed))


def _require_metric_operation_support(
    current: cell_mesh_owner.CellMesh,
    target: cell_mesh_owner.CellMesh,
    selection: PeriodicMetricOrbitSelection,
    /,
    *,
    controller_work: _MetricControllerWork,
) -> None:
    target_ids = set(np.asarray(target.vertex_global_ids).tolist())
    old_rows = {
        int(identifier): row for row, identifier in enumerate(current.vertex_global_ids)
    }
    new_rows = {
        int(identifier): row for row, identifier in enumerate(target.vertex_global_ids)
    }
    edges = set(map(tuple, entity_keys(target, 1)))
    for carrier in selection.carriers:
        controller_work.visit()
        ordered = tuple(
            carrier.vertex_global_ids[carrier.vertex_permutation.index(slot)]
            for slot in range(len(carrier.vertex_global_ids))
        )
        match selection.operation:
            case "split" | "flip":
                if tuple(sorted(ordered)) in edges:
                    raise ValueError(
                        "require_metric_operation_support.current must apply the declared operation to every selected carrier; got A synchronized edge operation retains an unedited carrier."
                    )
            case "collapse":
                if ordered[0] in target_ids or ordered[1] not in target_ids:
                    raise ValueError(
                        "require_metric_operation_support.current must apply the declared operation to every selected carrier; got A synchronized collapse loses its rational oriented contraction evidence."
                    )
            case "relocate":
                vertex = ordered[0]
                if vertex not in new_rows or np.array_equal(
                    np.asarray(current.coordinates)[old_rows[vertex]],
                    np.asarray(target.coordinates)[new_rows[vertex]],
                ):
                    raise ValueError(
                        "require_metric_operation_support.current must apply the declared operation to every selected carrier; got A synchronized relocation leaves an source copy unchanged."
                    )
            case invalid:
                assert_never(invalid)


def _require_metric_protected_scopes(
    source: CellMeshingResult,
    target: cell_mesh_owner.CellMesh,
    policy: adaptation_owner.MeshAdaptationPolicy,
    /,
    *,
    controller_work: _MetricControllerWork,
) -> None:
    protected = _metric_protected_rows(source, policy, controller_work=controller_work)
    old_ids = np.asarray(source.mesh.vertex_global_ids)
    new_rows = {
        int(identifier): row for row, identifier in enumerate(target.vertex_global_ids)
    }
    for row in protected[0]:
        controller_work.visit()
        identifier = int(old_ids[row])
        if identifier not in new_rows or not np.array_equal(
            np.asarray(source.mesh.coordinates)[row],
            np.asarray(target.coordinates)[new_rows[identifier]],
        ):
            raise ValueError(
                "require_metric_protected_scopes.source must preserve protected source entities and their orbit copies; got Periodic metric closure changes a protected source vertex or orbit copy."
            )
    for degree in range(1, source.mesh.topological_dimension + 1):
        controller_work.visit()
        present = set(map(tuple, entity_keys(target, degree)))
        keys = entity_keys(source.mesh, degree)
        if any(tuple(keys[row]) not in present for row in protected[degree]):
            raise ValueError(
                "require_metric_protected_scopes.source must preserve protected source entities and their orbit copies; got Periodic metric closure splits or removes a protected source entity or orbit copy."
            )


def _require_metric_organization(
    source: CellMeshingResult,
    target: cell_mesh_owner.CellMesh,
    edit: CellTopologyEdit,
    policy: adaptation_owner.MeshAdaptationPolicy,
    /,
    *,
    controller_work: _MetricControllerWork,
) -> None:
    dimension = source.mesh.topological_dimension
    source_classes = _metric_source_classes(
        source, policy, controller_work=controller_work
    )
    origins: dict[int, set[int]] = {}
    for old, new in zip(
        edit.relations[dimension].source_keys[:, 0],
        edit.relations[dimension].target_keys[:, 0],
        strict=True,
    ):
        controller_work.visit()
        if int(old) not in source_classes:
            raise ValueError(
                "require_metric_organization.source must preserve scientific source cell classes; got Periodic metric cell ancestry names a stale or absent scientific cell."
            )
        origins.setdefault(int(new), set()).add(source_classes[int(old)])
    for identifier in entity_keys(target, dimension)[:, 0]:
        controller_work.visit()
        if int(identifier) in source_classes:
            origins.setdefault(int(identifier), set()).add(
                source_classes[int(identifier)]
            )
        if len(origins.get(int(identifier), ())) != 1:
            raise ValueError(
                "require_metric_organization.source must preserve scientific source cell classes; got Periodic metric closure merges or loses scientific source cell classes."
            )


def metric_target_lineage(
    source: cell_mesh_owner.CellMesh,
    edit: CellTopologyEdit,
    target: cell_mesh_owner.CellMesh,
    /,
    *,
    budget: CoordinateEnclosureBudget,
    scope: NativeExecutionScope,
) -> MeshLineage:
    """Bind ancestry to the reconstructed carrier by scientific identity."""

    staged, _, _ = assemble_topology_edit(
        source, edit, numeric_version=target.numeric_version, budget=budget
    )
    if staged.topological_dimension != target.topological_dimension:
        raise ValueError(
            "metric_target_lineage.source must preserve staged dimensions, scientific IDs, incidence, and identification; got Reconstruction changes the staged carrier dimension."
        )
    for degree in range(source.topological_dimension + 1):
        staged_ids = np.asarray(staged.entity_set(degree).entity_ids)
        target_ids = np.asarray(target.entity_set(degree).entity_ids)
        rows = key_rows(staged_ids[:, None], target_ids[:, None])
        if staged_ids.size != target_ids.size or np.any(rows < 0):
            raise ValueError(
                "metric_target_lineage.source must preserve staged dimensions, scientific IDs, incidence, and identification; got Reconstruction changes scientific carrier entity identities."
            )
        if not np.array_equal(
            entity_keys(staged, degree)[rows], entity_keys(target, degree)
        ):
            raise ValueError(
                "metric_target_lineage.source must preserve staged dimensions, scientific IDs, incidence, and identification; got Reconstruction changes scientific carrier incidence."
            )
    if (staged.periodic_topology is None) != (target.periodic_topology is None):
        raise ValueError(
            "metric_target_lineage.source must preserve staged dimensions, scientific IDs, incidence, and identification; got Reconstruction changes the periodic quotient identity."
        )
    if staged.periodic_topology is not None:
        if target.periodic_topology is None:
            raise RuntimeError(
                "metric_target_lineage.source must preserve staged dimensions, scientific IDs, incidence, and identification; got Matched periodic reconstruction lost its target quotient."
            )
        if identification_id(staged.periodic_topology.cell) != identification_id(
            target.periodic_topology.cell
        ):
            raise ValueError(
                "metric_target_lineage.source must preserve staged dimensions, scientific IDs, incidence, and identification; got Reconstruction changes the periodic identification."
            )
        _require_complete_periodic_entity_changes(
            source, target, edit, scope=scope, budget=budget
        )
    return MeshLineage(
        source.topology_id,
        target.topology_id,
        tuple(entity_lineage(source, target, relation) for relation in edit.relations),
    )


class _MetricMembership(StrictModule):
    dimensions: dict[int, GeometryAssociation]


def _prepare_metric_membership(
    source: CellMeshingResult,
    target: cell_mesh_owner.CellMesh,
    stage: PeriodicMetricGeometryStage,
    controller_work: _MetricControllerWork,
) -> _MetricMembership:
    if not isinstance(stage.source_membership, tuple) or not stage.source_membership:
        raise ValueError(
            "prepare_metric_membership.source must bind source membership to every target entity dimension; got The periodic full-map builder must return all required source membership."
        )
    declared = {
        (association.source_id, association.source_revision)
        for association in source.associations
    }
    dimensions = {}
    for association in stage.source_membership:
        controller_work.visit()
        if not isinstance(association, GeometryAssociation):
            raise TypeError(
                "prepare_metric_membership.source must bind source membership to every target entity dimension; got Periodic source membership requires deterministic GeometryAssociation records."
            )
        if (
            association.source_id,
            association.source_revision,
        ) not in declared or not association.complete:
            raise ValueError(
                "prepare_metric_membership.source must bind source membership to every target entity dimension; got Periodic successor membership loses its source/version or is undecided."
            )
        matches = [
            degree
            for degree in range(target.topological_dimension + 1)
            if target.entity_set(degree).entity_set_id == association.target_entity_set_id
        ]
        if len(matches) != 1 or matches[0] in dimensions:
            raise ValueError(
                "prepare_metric_membership.source must bind source membership to every target entity dimension; got Periodic successor membership is stale or duplicates an entity dimension."
            )
        degree = matches[0]
        association.validate_target(target.entity_set(degree))
        association.target_rows(np.asarray(target.entity_set(degree).entity_ids))
        dimensions[degree] = association
    if set(dimensions) != set(range(target.topological_dimension + 1)):
        raise ValueError(
            "prepare_metric_membership.source must bind source membership to every target entity dimension; got Periodic successor membership omits an mesh entity dimension."
        )
    return _MetricMembership(dimensions)


def _certify_metric_membership(
    source: CellMeshingResult,
    target: cell_mesh_owner.CellMesh,
    membership: _MetricMembership,
    controller_work: _MetricControllerWork,
) -> _MetricMembership:
    dimensions = membership.dimensions
    for association in source.associations:
        controller_work.visit()
        matches = [
            degree
            for degree in range(source.mesh.topological_dimension + 1)
            if source.mesh.entity_set(degree).entity_set_id
            == association.target_entity_set_id
        ]
        if len(matches) != 1:
            raise ValueError(
                "certify_metric_membership.source must preserve retained scientific source strata and fixed corners; got Source geometry membership does not bind its scientific mesh epoch."
            )
        degree = matches[0]
        successor = dimensions[degree]
        old_keys = entity_keys(source.mesh, degree)
        old_ids = np.asarray(source.mesh.entity_set(degree).entity_ids)
        source_rows = association.target_rows(old_ids)
        new_ids = np.asarray(target.entity_set(degree).entity_ids)
        target_rows = successor.target_rows(new_ids)
        new_key_rows = {
            tuple(key): row for row, key in enumerate(entity_keys(target, degree))
        }
        for old_row, key in enumerate(old_keys):
            controller_work.visit()
            row = int(source_rows[old_row])
            ambiguous = bool(np.asarray(association.ambiguous)[row])
            corner = (
                degree == 0 and int(np.asarray(association.source_dimensions)[row]) == 0
            )
            if ambiguous or corner:
                if tuple(key) not in new_key_rows:
                    raise ValueError(
                        "certify_metric_membership.source must preserve retained scientific source strata and fixed corners; got Periodic metric closure removes a fixed or ambiguous source class."
                    )
                if degree == 0 and not np.array_equal(
                    np.asarray(source.mesh.coordinates)[old_row],
                    np.asarray(target.coordinates)[new_key_rows[tuple(key)]],
                ):
                    raise ValueError(
                        "certify_metric_membership.source must preserve retained scientific source strata and fixed corners; got Periodic metric closure relocates a scientific source corner."
                    )
            if tuple(key) in new_key_rows:
                after = int(target_rows[new_key_rows[tuple(key)]])
                if (
                    association.source_entities.entity_id(row),
                    association.source_entities.occurrence_path(row),
                ) != (
                    successor.source_entities.entity_id(after),
                    successor.source_entities.occurrence_path(after),
                ):
                    raise ValueError(
                        "certify_metric_membership.source must preserve retained scientific source strata and fixed corners; got Periodic metric closure changes a retained scientific source stratum."
                    )
    return membership


@checked
def _require_metric_geometry_stage(
    source: CellMeshingResult,
    target: cell_mesh_owner.CellMesh,
    edit: CellTopologyEdit,
    stage: PeriodicMetricGeometryStage,
    policy: adaptation_owner.MeshAdaptationPolicy,
    certificate_limits: MeshCertificateLimits,
    /,
    *,
    controller_work: _MetricControllerWork,
    budget: CoordinateEnclosureBudget,
) -> None:
    if stage.target_mesh is not target:
        raise ValueError(
            "require_metric_geometry_stage.source must bind its target, correspondence, transition, and authored certificate policies; got The periodic geometry stage must retain its target carrier."
        )
    if not isinstance(stage.target_geometry, CellGeometrySpec) or not isinstance(
        stage.geometry_transition, CellGeometryTransition
    ):
        raise TypeError(
            "require_metric_geometry_stage.source must bind its target, correspondence, transition, and authored certificate policies; got A periodic geometry stage requires all required geometry and transition owners."
        )
    correspondence = stage.source_correspondence
    match correspondence:
        case PreparedSphereChartDeformation():
            stage.target_geometry.resolve(target, budget=budget)
            correspondence.require_bound(
                source.mesh, source.geometry, target, stage.target_geometry, budget=budget
            )
        case PreparedSurfaceChartDeformation():
            stage.target_geometry.resolve(target, budget=budget)
            correspondence.require_bound(
                source.mesh, source.geometry, target, stage.target_geometry
            )
        case _:
            raise TypeError(
                "require_metric_geometry_stage.source must bind its target, correspondence, transition, and authored certificate policies; got Periodic source correspondence requires its Surface or Sphere owner."
            )
    if (
        correspondence.target_validity.policy_id
        != policy.audit_policy.validity_policy.policy_id
    ):
        raise ValueError(
            "require_metric_geometry_stage.source must bind its target, correspondence, transition, and authored certificate policies; got The periodic full-map builder renewed or changed the validity policy."
        )
    if correspondence.target_embedding.binding.limits_id != certificate_limits.limits_id:
        raise ValueError(
            "require_metric_geometry_stage.source must bind its target, correspondence, transition, and authored certificate policies; got The periodic full-map builder changed the authored certificate limits."
        )
    transition = stage.geometry_transition
    if (
        transition.source_topology_id,
        transition.target_topology_id,
        transition.source_geometry_id,
        transition.target_geometry_id,
    ) != (
        source.mesh.topology_id,
        target.topology_id,
        cell_geometry_id(source.geometry),
        cell_geometry_id(stage.target_geometry),
    ) or cell_geometry_id(transition.geometry) != cell_geometry_id(stage.target_geometry):
        raise ValueError(
            "require_metric_geometry_stage.source must bind its target, correspondence, transition, and authored certificate policies; got The periodic full-map transition is stale for its source or target."
        )
    if not np.array_equal(transition.vertex_coordinates, target.coordinates):
        raise ValueError(
            "require_metric_geometry_stage.source must bind its target, correspondence, transition, and authored certificate policies; got The periodic full-map transition does not place the target vertices."
        )
    membership = _prepare_metric_membership(source, target, stage, controller_work)
    _certify_metric_membership(source, target, membership, controller_work)


def _require_metric_output_limits(
    mesh: cell_mesh_owner.CellMesh,
    stage: PeriodicMetricGeometryStage | None,
    edit: CellTopologyEdit,
    limits: MeshingLimits,
    /,
    *,
    controller_work: _MetricControllerWork,
) -> None:

    controller_work.visit(mesh.topological_dimension + 1 + len(mesh.topology.incidences))
    published_entity_limits(mesh, limits)
    controller_work.visit()
    if model_array_bytes((mesh, stage, edit)) > limits.maximum_data_bytes:
        raise ValueError(
            "require_metric_output_limits.mesh must fit the authored publication and retained-data limits; got Periodic metric full-map stage exceeds its all required retained-data capacity."
        )


def _metric_identity_storage(
    banks: tuple[PeriodicEntityIdentityBank, ...],
    witness: PeriodicVertexOrbitWitness | None,
    ledger: CoordinateEnclosureBudget,
    /,
    *,
    controller_work: _MetricControllerWork,
) -> int:
    """Bound retained bank DAG metadata, separately from numerical buffers."""

    pending_banks: list[tuple[PeriodicEntityIdentityBank, ...]] = [banks]
    pending_witnesses = [] if witness is None else [witness]
    seen_witnesses: set[int] = set()
    storage = 0
    while pending_witnesses:
        owner = pending_witnesses.pop()
        controller_work.visit()
        if id(owner) in seen_witnesses:
            continue
        seen_witnesses.add(id(owner))
        # Bank-reference lists and evidence dedup tables are included before grow.
        ledger.reserve(0, 1024)
        storage += 1024
        pending_banks.extend((owner.quotient_entities, owner.retained_quotient_entities))
        if owner.allocation_prior is not None:
            pending_witnesses.append(owner.allocation_prior[1])
    seen: set[int] = set()
    pending: list[object] = list(pending_banks)
    while pending:
        value = pending.pop()
        controller_work.visit()
        if id(value) in seen:
            continue
        seen.add(id(value))
        # immutable tuple/int payload plus conservative stack/set overhead.
        match value:
            case tuple():
                upper = 64 + 8 * len(value) + 512
            case int():
                upper = bigint_bytes(abs(value).bit_length()) + 512
            case _:
                raise TypeError(
                    "metric_identity_storage.banks must contain retained tuple and integer identity metadata; got Periodic identity storage must contain tuple/int metadata."
                )
        ledger.reserve(0, upper)
        storage += upper
        if isinstance(value, tuple):
            pending.extend(value)
    return storage


class _MetricAdmission(StrictModule):
    work_upper: int = eqx.field(static=True)
    host_upper: int = eqx.field(static=True)


def _metric_controller_admission(
    source: CellMeshingResult,
    candidate: CellTopologyEdit,
    exact_stencils: Mapping[int, ConstructionPointKey],
    limits: MeshingLimits,
    budget: CoordinateEnclosureBudget,
    /,
    *,
    scope: NativeExecutionScope,
) -> _MetricAdmission:

    # The support has at most one carrier per source entity.
    # Each carrier can inspect the source incidence bank; dictionaries and rational
    # deterministic keys have a separately proved CPython bound (not native RSS).
    # Fraction numerator/denominator growth is bounded by all supplied input bits.
    vertices = source.mesh.coordinates.shape[0]
    cells = sum(block.cell_count for block in source.mesh.blocks)
    candidate_cells = sum(block.cell_ids.size for block in candidate.blocks)
    support_count = sum(len(key) for key in exact_stencils.values())
    bit_bound = sum(
        abs(weight.numerator).bit_length() + weight.denominator.bit_length() + 1
        for key in exact_stencils.values()
        for _, weight in key
    )
    coefficient_upper = (
        512 + 32 * source.mesh.ambient_dimension + 2 * bigint_bytes(max(1, bit_bound))
    )
    candidate_vertices = candidate.vertex_global_ids.size
    relation_rows = sum(record.kinds.size for record in candidate.relations)
    arity = max(
        block.cells.shape[1]
        for block in candidate.blocks
        if isinstance(block, TopologyEditBlock)
    )
    copies_upper = max(
        1,
        vertices,
        candidate_vertices,
        source.mesh.entity_set(1).count,
        arity * candidate_cells,
    )
    # Stored incidence is sparse: each source cell belongs to at most its
    # arity-squared edge endpoint stars, even though selection scans all cells.
    records_upper = 8 * (
        vertices
        + candidate_vertices
        + cells
        + candidate_cells
        + support_count
        + relation_rows
    ) + arity * arity * (cells + candidate_cells)
    host_upper = 4096 + records_upper * (coefficient_upper + 1024 + arity * 128)
    work_upper = (
        records_upper * copies_upper * (arity + source.mesh.topological_dimension + 4)
    )
    allowance = scope.remaining()
    if host_upper > min(limits.maximum_scratch_bytes, budget.maximum_memory_bytes):
        raise ValueError(
            "metric_controller_admission.source must fit the source request work, wall, and scratch allowances; got Periodic metric host-object preparation exceeds its proven scratch upper bound."
        )
    if work_upper > limits.maximum_work_units or allowance.remaining_seconds <= 0:
        raise ValueError(
            "metric_controller_admission.source must fit the source request work, wall, and scratch allowances; got Periodic metric preparation exceeds the work or wall allowance."
        )
    budget.admit_work_bound(work_upper)
    budget.reserve(0, host_upper)
    return _MetricAdmission(work_upper, host_upper)


class PeriodicMetricEditRequest(StrictModule, NonTrainableState):
    """Bind one periodic metric attempt to its source, owners, and resource ledger."""

    source: CellMeshingResult
    candidate: CellTopologyEdit
    exact_stencils: Mapping[int, ConstructionPointKey]
    scope: NativeExecutionScope
    operation: PeriodicMetricOperation
    limits: MeshingLimits
    policy: adaptation_owner.MeshAdaptationPolicy
    budget: CoordinateEnclosureBudget
    certificate_limits: MeshCertificateLimits
    prior_stage: PeriodicMetricOrbitOutcome | None
    build_candidate: Callable[
        [
            CellTopologyEdit,
            Mapping[int, ConstructionPointKey],
            PeriodicMetricOrbitSelection,
        ],
        PeriodicMetricCandidateStage,
    ]
    build_geometry: Callable[
        [CellTopologyEdit, cell_mesh_owner.CellMesh], PeriodicMetricGeometryStage
    ]
    retained_quotient_entities: tuple[PeriodicEntityIdentityBank, ...]

    @checked
    def __init__(
        self,
        source: CellMeshingResult,
        candidate: CellTopologyEdit,
        exact_stencils: Mapping[int, ConstructionPointKey],
        /,
        *,
        scope: NativeExecutionScope,
        operation: PeriodicMetricOperation,
        limits: MeshingLimits,
        policy: adaptation_owner.MeshAdaptationPolicy,
        budget: CoordinateEnclosureBudget,
        certificate_limits: MeshCertificateLimits,
        prior_stage: PeriodicMetricOrbitOutcome | None,
        build_candidate: Callable[
            [
                CellTopologyEdit,
                Mapping[int, ConstructionPointKey],
                PeriodicMetricOrbitSelection,
            ],
            PeriodicMetricCandidateStage,
        ],
        build_geometry: Callable[
            [CellTopologyEdit, cell_mesh_owner.CellMesh], PeriodicMetricGeometryStage
        ],
        retained_quotient_entities: tuple[PeriodicEntityIdentityBank, ...],
    ) -> None:
        self.source = source
        self.candidate = candidate
        self.exact_stencils = exact_stencils
        self.scope = scope
        self.operation = operation
        self.limits = limits
        self.policy = policy
        self.budget = budget
        self.certificate_limits = certificate_limits
        self.prior_stage = prior_stage
        self.build_candidate = build_candidate
        self.build_geometry = build_geometry
        self.retained_quotient_entities = retained_quotient_entities


class _MetricPreparation(StrictModule):
    operation: PeriodicMetricOperation = eqx.field(static=True)
    orbits: PeriodicConstructionOrbits
    original_geometry_id: str = eqx.field(static=True)


class _MetricContinuation(StrictModule):
    mesh: cell_mesh_owner.CellMesh
    geometry: CellGeometrySpec
    retained: tuple[PeriodicEntityIdentityBank, ...]
    allocation_prior: tuple[cell_mesh_owner.CellMesh, PeriodicVertexOrbitWitness] | None
    host_upper: int = eqx.field(static=True)


class _MetricCandidate(StrictModule):
    edit: CellTopologyEdit
    target: cell_mesh_owner.CellMesh
    exact: dict[int, ConstructionPointKey]
    selection: PeriodicMetricOrbitSelection
    current_orbits: PeriodicConstructionOrbits
    cavity_count: int = eqx.field(static=True)
    host_upper: int = eqx.field(static=True)


class _MetricGeometryExecution(StrictModule):
    stage: PeriodicMetricGeometryStage


class _MetricCertification(StrictModule):
    stage: PeriodicMetricGeometryStage
    evidence: PeriodicMetricOrbitEvidence


def _prepare_metric_edit(request: PeriodicMetricEditRequest) -> _MetricPreparation:
    source = request.source
    candidate = request.candidate
    operation = request.operation
    policy = request.policy
    limits = request.limits
    budget = request.budget
    if not bool(eqx.tree_equal(limits, policy.limits, typematch=True)):
        raise ValueError(
            "prepare_metric_edit.request must retain compatible operation ancestry and an unchanged Cartesian source frame; got Periodic metric synchronization cannot reset the policy limits."
        )
    selected = parse(operation, PeriodicMetricOperation, "operation")
    if selected == "relocate" and not policy.relocation:
        raise ValueError(
            "prepare_metric_edit.request must retain compatible operation ancestry and an unchanged Cartesian source frame; got The adaptation policy forbids periodic relocation."
        )
    if candidate.operation not in ("local_reconnection", "relocation"):
        raise ValueError(
            "prepare_metric_edit.request must retain compatible operation ancestry and an unchanged Cartesian source frame; got Nested family constructions must use their concrete periodic template owner."
        )
    if (selected == "relocate") != (candidate.operation == "relocation"):
        raise ValueError(
            "prepare_metric_edit.request must retain compatible operation ancestry and an unchanged Cartesian source frame; got The periodic operation does not describe the private metric candidate."
        )
    if candidate.refinement is not None or candidate.coarsening is not None:
        raise ValueError(
            "prepare_metric_edit.request must retain compatible operation ancestry and an unchanged Cartesian source frame; got Local metric candidates cannot substitute nested family geometry witnesses."
        )
    if candidate.shared_faces is not None or candidate.prescribed_entity_ids:
        raise ValueError(
            "prepare_metric_edit.request must retain compatible operation ancestry and an unchanged Cartesian source frame; got A concrete restored/template candidate must be synchronized by its family."
        )
    dimension = source.mesh.topological_dimension
    if tuple(record.dimension for record in candidate.relations) != tuple(
        range(dimension + 1)
    ):
        raise ValueError(
            "prepare_metric_edit.request must retain compatible operation ancestry and an unchanged Cartesian source frame; got The private metric candidate omits all required entity ancestry."
        )
    if not source.coordinate_contract.is_orthonormal_cartesian:
        raise ValueError(
            "prepare_metric_edit.request must retain compatible operation ancestry and an unchanged Cartesian source frame; got Periodic metric group actions require the Cartesian source frame."
        )
    source.geometry.resolve(source.mesh, budget=budget)
    orbits = PeriodicConstructionOrbits(source.mesh)
    original_geometry_id = cell_geometry_id(source.geometry)
    if (
        candidate.periodic_orbits is not None
        and candidate.periodic_orbits.source_topology_id != source.mesh.topology_id
    ):
        raise ValueError(
            "prepare_metric_edit.request must retain compatible operation ancestry and an unchanged Cartesian source frame; got The private metric candidate carries a stale scientific source topology."
        )
    return _MetricPreparation(selected, orbits, original_geometry_id)


def _prepare_metric_continuation(
    request: PeriodicMetricEditRequest,
    preparation: _MetricPreparation,
    controller_work: _MetricControllerWork,
) -> _MetricContinuation:
    source = request.source
    candidate = request.candidate
    exact_stencils = request.exact_stencils
    limits = request.limits
    budget = request.budget
    scope = request.scope
    prior_stage = request.prior_stage
    policy = request.policy
    certificate_limits = request.certificate_limits
    retained_quotient_entities = request.retained_quotient_entities
    orbits = preparation.orbits
    current = source.mesh
    current_geometry = source.geometry
    retained = retained_quotient_entities
    allocation_prior = None
    admission = _metric_controller_admission(
        source, candidate, exact_stencils, limits, budget, scope=scope
    )
    host_upper = admission.host_upper
    if prior_stage is not None:
        if not isinstance(prior_stage, PeriodicMetricOrbitOutcome):
            raise TypeError(
                "prepare_metric_continuation.request must retain source bindings, prior evidence, and the same coordinate ledger; got A continued periodic edit requires the prior private stage."
            )
        if (
            budget.work_units < prior_stage.coordinate_work_units
            or budget.peak_bytes_upper < prior_stage.coordinate_peak_storage_upper_bytes
        ):
            raise ValueError(
                "prepare_metric_continuation.request must retain source bindings, prior evidence, and the same coordinate ledger; got A private periodic continuation cannot renew its coordinate ledger."
            )
        if (
            prior_stage.source_result_id,
            prior_stage.source_topology_id,
            prior_stage.source_numeric_version,
            prior_stage.source_frame_id,
            prior_stage.source_geometry_id,
            prior_stage.coordinate_contract_id,
            prior_stage.policy_id,
            prior_stage.certificate_limits_id,
        ) != (
            source.result_id,
            source.mesh.topology_id,
            source.mesh.numeric_version,
            canonical_fingerprint(orbits.source_frame_id),
            cell_geometry_id(source.geometry),
            source.coordinate_contract.spatial_id,
            policy.policy_id,
            certificate_limits.limits_id,
        ):
            raise ValueError(
                "prepare_metric_continuation.request must retain source bindings, prior evidence, and the same coordinate ledger; got The private periodic continuation is stale or renews the source policy."
            )
        current = prior_stage.target_mesh
        current_geometry = prior_stage.geometry_stage.target_geometry
        if (
            current.numeric_version,
            canonical_fingerprint(array_tree_fingerprint(current.coordinates)),
            cell_geometry_id(current_geometry),
        ) != (
            prior_stage.target_numeric_version,
            prior_stage.target_frame_id,
            prior_stage.target_geometry_id,
        ):
            raise ValueError(
                "prepare_metric_continuation.request must retain source bindings, prior evidence, and the same coordinate ledger; got The private periodic continuation changed its current coordinate frame or map."
            )
        if prior_stage.periodic_witness is not prior_stage.edit.periodic_orbits:
            raise ValueError(
                "prepare_metric_continuation.request must retain source bindings, prior evidence, and the same coordinate ledger; got The private periodic continuation lost its rational orbit evidence."
            )
        current_periodic = require_periodic_topology(current)
        if (
            prior_stage.evidence.embedding.periodic_topology_id
            != current_periodic.periodic_topology_id
            or prior_stage.evidence.quotient.periodic_topology_id
            != current_periodic.periodic_topology_id
            or prior_stage.evidence.status != "complete"
        ):
            raise ValueError(
                "prepare_metric_continuation.request must retain source bindings, prior evidence, and the same coordinate ledger; got The private periodic continuation lost its all required quotient embedding evidence."
            )
        _require_metric_stencils(
            source.mesh,
            prior_stage.edit,
            prior_stage.exact_stencils,
            controller_work=controller_work,
        )
        _require_metric_geometry_stage(
            source,
            current,
            prior_stage.edit,
            prior_stage.geometry_stage,
            policy,
            certificate_limits,
            controller_work=controller_work,
            budget=budget,
        )
        _require_metric_protected_scopes(
            source, current, policy, controller_work=controller_work
        )
        _require_metric_organization(
            source,
            current,
            prior_stage.edit,
            policy,
            controller_work=controller_work,
        )
        retained = prior_stage.periodic_witness.quotient_entities
        allocation_prior = (current, prior_stage.periodic_witness)
    host_upper += _metric_identity_storage(
        retained,
        None if allocation_prior is None else allocation_prior[1],
        budget,
        controller_work=controller_work,
    )
    return _MetricContinuation(
        current, current_geometry, retained, allocation_prior, host_upper
    )


def _execute_metric_candidate(
    request: PeriodicMetricEditRequest,
    preparation: _MetricPreparation,
    continuation: _MetricContinuation,
    controller_work: _MetricControllerWork,
) -> _MetricCandidate:
    source = request.source
    candidate = request.candidate
    exact_stencils = request.exact_stencils
    policy = request.policy
    budget = request.budget
    limits = request.limits
    scope = request.scope
    prior_stage = request.prior_stage
    build_candidate = request.build_candidate
    selected = preparation.operation
    orbits = preparation.orbits
    current = continuation.mesh
    current_geometry = continuation.geometry
    retained = continuation.retained
    allocation_prior = continuation.allocation_prior
    host_upper = continuation.host_upper
    current_orbits = PeriodicConstructionOrbits(current)
    stencils = _require_metric_stencils(
        source.mesh, candidate, exact_stencils, controller_work=controller_work
    )
    selection = _metric_orbit_selection(
        source,
        candidate,
        selected,
        policy,
        current_orbits,
        current,
        current_geometry,
        prior_stage,
        controller_work=controller_work,
    )
    required_incidence = {
        identifier
        for carrier in selection.carriers
        for identifier in carrier.incident_cell_global_ids
    }
    _periodic_charge(cavity_cells=len(required_incidence), budget=budget)
    controller_work.flush()
    candidate_stage = build_candidate(candidate, stencils, selection)
    controller_work.flush()
    orbits.require_source(source.mesh)
    current_orbits.require_source(current)
    if cell_geometry_id(current_geometry) != selection.source_geometry_id:
        raise ValueError(
            "execute_metric_candidate.request must leave the retained source map and construction frame unchanged; got The callback changed the current source map."
        )
    validated = _require_metric_candidate_stage(
        source,
        current,
        candidate,
        candidate_stage,
        selection,
        stencils,
        controller_work=controller_work,
    )
    exact, cavity_count = validated.exact, validated.cavity_count
    edit = candidate_stage.edit
    admission = _metric_controller_admission(
        source, edit, exact, limits, budget, scope=scope
    )
    host_upper += admission.host_upper

    probe = build_mesh(edit, source.mesh.numeric_version, None, budget=budget)
    _require_metric_operation_support(
        current, probe, selection, controller_work=controller_work
    )
    controller_work.visit(probe.coordinates.shape[0])
    witness = orbits.witness(
        source.mesh,
        probe,
        exact,
        retained,
        allocation_prior=allocation_prior,
        budget=budget,
    )
    host_upper += _metric_identity_storage(
        witness.quotient_entities,
        witness,
        budget,
        controller_work=controller_work,
    )
    edit = edit._replace(periodic_orbits=witness)
    version = canonical_fingerprint(
        {
            "kind": "periodic-source-metric-frame",
            "source": source.mesh.numeric_version,
            "operation": selected,
            "coordinates": array_tree_fingerprint(edit.coordinates),
        }
    )
    target, _, _ = assemble_topology_edit(
        source.mesh, edit, numeric_version=version, budget=budget
    )
    _require_metric_output_limits(
        target, None, edit, limits, controller_work=controller_work
    )
    _require_complete_periodic_entity_changes(
        source.mesh, target, edit, scope=scope, budget=budget
    )
    _require_metric_protected_scopes(
        source, target, policy, controller_work=controller_work
    )
    _require_metric_organization(
        source, target, edit, policy, controller_work=controller_work
    )
    return _MetricCandidate(
        edit, target, exact, selection, current_orbits, cavity_count, host_upper
    )


def _execute_metric_geometry(
    request: PeriodicMetricEditRequest,
    preparation: _MetricPreparation,
    continuation: _MetricContinuation,
    candidate: _MetricCandidate,
    controller_work: _MetricControllerWork,
) -> _MetricGeometryExecution:
    source = request.source
    policy = request.policy
    budget = request.budget
    scope = request.scope
    certificate_limits = request.certificate_limits
    build_geometry = request.build_geometry
    limits = request.limits
    edit, target, selection = candidate.edit, candidate.target, candidate.selection
    orbits = preparation.orbits
    current = continuation.mesh
    current_geometry = continuation.geometry
    current_orbits = candidate.current_orbits
    orbits.require_source(source.mesh)
    controller_work.flush()
    stage = build_geometry(edit, target)
    controller_work.flush()
    if not isinstance(stage, PeriodicMetricGeometryStage) or not isinstance(
        stage.target_mesh, cell_mesh_owner.CellMesh
    ):
        raise TypeError(
            "execute_metric_geometry.request must retain the reconstructed target and leave the source map unchanged; got The geometry builder must retain its target CellMesh."
        )
    target = stage.target_mesh
    metric_target_lineage(source.mesh, edit, target, budget=budget, scope=scope)
    _require_metric_protected_scopes(
        source, target, policy, controller_work=controller_work
    )
    _require_metric_organization(
        source, target, edit, policy, controller_work=controller_work
    )
    orbits.require_source(source.mesh)
    current_orbits.require_source(current)
    if cell_geometry_id(current_geometry) != selection.source_geometry_id:
        raise ValueError(
            "execute_metric_geometry.request must retain the reconstructed target and leave the source map unchanged; got The geometry builder changed the current source map."
        )
    _require_metric_geometry_stage(
        source,
        target,
        edit,
        stage,
        policy,
        certificate_limits,
        controller_work=controller_work,
        budget=budget,
    )
    _require_metric_output_limits(
        target, stage, edit, limits, controller_work=controller_work
    )
    return _MetricGeometryExecution(stage)


def _certify_metric_edit(
    request: PeriodicMetricEditRequest,
    preparation: _MetricPreparation,
    continuation: _MetricContinuation,
    candidate: _MetricCandidate,
    stage: PeriodicMetricGeometryStage,
    controller_work: _MetricControllerWork,
) -> _MetricCertification:
    source = request.source
    budget = request.budget
    scope = request.scope
    certificate_limits = request.certificate_limits
    policy = request.policy
    execution = scope
    target = stage.target_mesh
    selected = preparation.operation
    orbits = preparation.orbits
    original_geometry_id = preparation.original_geometry_id
    current = continuation.mesh
    current_geometry = continuation.geometry
    current_orbits = candidate.current_orbits
    edit, selection = candidate.edit, candidate.selection
    cavity_count, host_upper = candidate.cavity_count, candidate.host_upper
    controller_work.flush()
    remaining = execution.remaining()
    if remaining.remaining_work_units <= 0 or remaining.remaining_managed_bytes <= 0:
        raise ValueError(
            "certify_metric_edit.request must retain the source map and fit the remaining quotient-certificate allowance; got The periodic certificate work or storage allowance is exhausted."
        )
    bounded_certificate_limits = MeshCertificateLimits(
        maximum_candidate_pairs=min(
            certificate_limits.maximum_candidate_pairs,
            remaining.remaining_work_units,
        ),
        maximum_ray_tests=certificate_limits.maximum_ray_tests,
        maximum_source_samples=certificate_limits.maximum_source_samples,
        maximum_distance_evaluations=certificate_limits.maximum_distance_evaluations,
        maximum_subdivision_depth=certificate_limits.maximum_subdivision_depth,
        maximum_subdivision_pieces=certificate_limits.maximum_subdivision_pieces,
        maximum_bernstein_nodes=certificate_limits.maximum_bernstein_nodes,
        maximum_periodic_images=min(
            certificate_limits.maximum_periodic_images,
            remaining.remaining_managed_bytes // max(1, target.coordinates.size * 8),
        ),
        maximum_work_units=min(
            certificate_limits.maximum_work_units, remaining.remaining_work_units
        ),
        maximum_scratch_bytes=min(
            certificate_limits.maximum_scratch_bytes,
            remaining.remaining_managed_bytes,
        ),
    )
    embedding = certify_periodic_embedding(
        target,
        geometry=stage.target_geometry,
        validity=stage.source_correspondence.target_validity,
        validity_policy=policy.audit_policy.validity_policy,
        limits=bounded_certificate_limits,
        maximum_images=bounded_certificate_limits.maximum_periodic_images,
        maximum_pairs=bounded_certificate_limits.maximum_candidate_pairs,
        budget=budget,
        scope=scope,
    )
    orbits.require_source(source.mesh)
    current_orbits.require_source(current)
    if cell_geometry_id(current_geometry) != selection.source_geometry_id:
        raise ValueError(
            "certify_metric_edit.request must retain the source map and fit the remaining quotient-certificate allowance; got The quotient certificate changed the current source map."
        )
    if cell_geometry_id(source.geometry) != original_geometry_id:
        raise ValueError(
            "certify_metric_edit.request must retain the source map and fit the remaining quotient-certificate allowance; got The periodic operation changed its immutable full source map."
        )
    budget.ticket.checkpoint()
    witness = edit.periodic_orbits
    if witness is None:
        raise RuntimeError(
            "certify_metric_edit.request must retain the source map and fit the remaining quotient-certificate allowance; got A synchronized metric edit lost its deterministic periodic evidence."
        )
    evidence = PeriodicMetricOrbitEvidence(
        selected,
        "complete",
        len(selection.carriers),
        cavity_count,
        controller_work.visits,
        host_upper,
        embedding,
        PeriodicQuotientEvidence(target),
    )
    return _MetricCertification(stage, evidence)


@checked
def synchronize_periodic_metric_edit(
    request: PeriodicMetricEditRequest, /
) -> PeriodicMetricOrbitOutcome:
    """Stage, certify, and return one orbit-preserving metric successor."""
    source, scope, budget = request.source, request.scope, request.budget
    policy, certificate_limits = request.policy, request.certificate_limits
    with _periodic_phase(
        scope, budget, budget.maximum_work_units, budget.maximum_memory_bytes
    ):
        preparation = _prepare_metric_edit(request)
        controller_work = _MetricControllerWork(budget)
        with budget.activate(), budget.temporary_scope(), controller_work.activate():
            continuation = _prepare_metric_continuation(
                request, preparation, controller_work
            )
            candidate = _execute_metric_candidate(
                request, preparation, continuation, controller_work
            )
            stage = _execute_metric_geometry(
                request, preparation, continuation, candidate, controller_work
            ).stage
            certification = _certify_metric_edit(
                request, preparation, continuation, candidate, stage, controller_work
            )
            edit, exact = candidate.edit, candidate.exact
            target = certification.stage.target_mesh
            stage, evidence = certification.stage, certification.evidence
            orbits = preparation.orbits
            witness = edit.periodic_orbits
            if witness is None:
                raise ValueError(
                    "PeriodicMetricEditRequest.candidate must retain its periodic orbit evidence; got None"
                )

            return PeriodicMetricOrbitOutcome(
                edit,
                target,
                witness,
                stage,
                MappingProxyType(exact),
                source.result_id,
                source.mesh.topology_id,
                source.mesh.numeric_version,
                canonical_fingerprint(orbits.source_frame_id),
                source.coordinate_contract.spatial_id,
                cell_geometry_id(source.geometry),
                target.numeric_version,
                canonical_fingerprint(array_tree_fingerprint(target.coordinates)),
                cell_geometry_id(stage.target_geometry),
                policy.policy_id,
                certificate_limits.limits_id,
                budget.work_units,
                budget.peak_bytes_upper,
                evidence,
            )
