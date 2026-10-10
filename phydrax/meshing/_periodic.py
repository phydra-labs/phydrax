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
"""

from __future__ import annotations

from collections.abc import Callable, Generator, Iterator, Sequence
from contextlib import closing, contextmanager
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations, product
from typing import NamedTuple, TYPE_CHECKING

import equinox as eqx
import numpy as np
from jax.typing import ArrayLike
from numpy.typing import NDArray

from .._bvh import bvh_overlap_pair_blocks, BVHBuildPolicy, PackedBVH, prepare_bvh
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._meshcore import NativeHostStorageWorkspace
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import (
    CellBlock,
    CellMesh,
    periodic_orbit_measures,
    PeriodicCell,
    PeriodicMeshTopology,
)
from ..discretization._cell_complex import (
    PolygonalConnectivity,
    PolyhedralConnectivity,
    TetrahedralConnectivity,
)
from ..discretization._cell_geometry import (
    CellGeometrySpec,
    coordinate_lagrange_element,
)
from ..discretization._cell_mesh import SimplicialConnectivity
from ..discretization._periodic_topology import PeriodicIsometryGroup
from ..geometry._mesh_certificates import (
    _det2,
    _det3,
    _dyadic_integers,
    GlobalEmbeddingCertificate,
    MeshCertificateLimits,
)
from ..geometry._triangulation import (
    PeriodicDelaunayTriangulation,
    PeriodicTriangulationEvidence,
)


if TYPE_CHECKING:
    from ..discretization._cell_geometry_transfer import NestedReferenceWitnesses
    from ..discretization._cell_geometry_validity import (
        CellValidityCertificate,
        CellValidityPolicy,
    )
    from ._adaptation import _RouteOutcome, MeshAdaptationResult, PreparedMeshAdaptation
    from ._lineage import MeshLineage, VertexInterpolationStencil
    from ._result import CellMeshingResult
    from ._topology_edit import (
        CellTopologyEdit,
        EntityRelations,
        PeriodicEntityIdentityBank,
        PeriodicNonnestedGeometryAuthority,
        PeriodicVertexOrbitWitness,
        PrescribedEntityIds,
    )

from ._controls import PeriodicConstraint


def _require_periodic_topology(mesh: CellMesh, /) -> PeriodicMeshTopology:
    periodic = mesh.periodic_topology
    if periodic is None:
        raise ValueError("The operation requires a bound periodic mesh topology.")
    return periodic


type ConstructionPointKey = tuple[tuple[int, Fraction], ...]
type PeriodicConstructionPointKey = tuple[tuple[int, tuple[int, ...], Fraction], ...]


class PeriodicConstructionOrbits:
    """Exact source-incidence point orbits, bound to one source numeric frame."""

    def __init__(self, mesh: CellMesh) -> None:
        from ..discretization._periodic_topology import (
            _identification_orders,
        )

        periodic = _require_periodic_topology(mesh)
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
        self.orders = _identification_orders(periodic.cell)
        self.rank = periodic.cell.rank
        self.source_topology_id = mesh.topology_id
        self.source_periodic_topology_id = periodic.periodic_topology_id
        self.source_numeric_version = mesh.numeric_version
        self.source_frame_id = array_tree_fingerprint(mesh.coordinates)

    def require_source(self, mesh: CellMesh, /) -> None:
        if (
            mesh.topology_id != self.source_topology_id
            or _require_periodic_topology(mesh).periodic_topology_id
            != self.source_periodic_topology_id
            or mesh.numeric_version != self.source_numeric_version
            or array_tree_fingerprint(mesh.coordinates) != self.source_frame_id
        ):
            raise ValueError(
                "Periodic construction orbits require their exact source topology and numeric frame."
            )

    def reduced(self, shifts: np.ndarray) -> np.ndarray:
        from ..discretization._periodic_topology import _reduced

        return _reduced(shifts, self.orders)

    def point(
        self,
        key: ConstructionPointKey,
        anchor: np.ndarray | None = None,
    ) -> tuple[PeriodicConstructionPointKey, np.ndarray]:
        if not key or any(identifier not in self.vertices for identifier, _ in key):
            raise ValueError(
                "A periodic construction point must name its actual source vertex support."
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
                "Construction support has no authoritative group-image anchor."
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
        source: CellMesh,
        target: CellMesh,
        stencils: dict[int, ConstructionPointKey],
        retained: tuple[PeriodicEntityIdentityBank, ...],
        *,
        allocation_prior: tuple[CellMesh, PeriodicVertexOrbitWitness] | None = None,
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
                    "Construction removes a representative of a retained periodic orbit."
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
        )


def _simplex_entity_corners(mesh: CellMesh, degree: int, /) -> np.ndarray:
    connectivity = mesh.connectivity
    if isinstance(connectivity, SimplicialConnectivity):
        if 0 < degree < connectivity.dimension:
            return np.asarray(connectivity.entities[degree])
        raise ValueError(
            "The periodic simplex mesh has no entities of that intermediate degree."
        )
    if isinstance(connectivity, PolyhedralConnectivity):
        if degree == 1:
            return np.asarray(connectivity.edges)
        if degree == 2:
            offsets = np.asarray(connectivity.face_vertex_offsets)
            if np.any(np.diff(offsets) != 3):
                raise TypeError(
                    "Periodic simplex faces require triangular packed incidence."
                )
            return np.asarray(connectivity.face_vertex_values).reshape((-1, 3))
        raise ValueError(
            "The periodic simplex mesh has no entities of that intermediate degree."
        )
    if not isinstance(connectivity, (PolygonalConnectivity, TetrahedralConnectivity)):
        raise TypeError(
            "Periodic simplex operations require canonical simplex connectivity."
        )
    if degree == 1:
        return np.asarray(connectivity.edges)
    if degree == 2 and isinstance(connectivity, TetrahedralConnectivity):
        return np.asarray(connectivity.faces)
    raise ValueError(
        "The periodic simplex mesh has no entities of that intermediate degree."
    )


def periodic_edge_size_evidence(mesh: CellMesh, /) -> tuple[np.ndarray, float]:
    """Physical length/growth of each authoritative quotient edge exactly once."""

    from .providers._native_publication import edge_growth_evidence

    topology = _require_periodic_topology(mesh)
    representatives = np.asarray(topology.orbit_representatives(1))
    edges = _simplex_entity_corners(mesh, 1)[representatives]
    roots = np.asarray(topology.vertex_representatives)
    shifts = np.asarray(topology.vertex_shifts)[edges]
    points = np.asarray(mesh.coordinates)[roots[edges]]
    if isinstance(topology.cell, PeriodicCell):
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
    This is not an exact-construction claim for independently rounded images.
    """

    periodic_topology_id: str = eqx.field(static=True)
    image_count: int = eqx.field(static=True)
    candidate_pair_count: int = eqx.field(static=True)
    coordinate_scope: str = eqx.field(static=True)
    maximum_coordinate_residual: float = eqx.field(static=True)
    coordinate_residual_bound: float = eqx.field(static=True)
    certificate_id: str = eqx.field(static=True)
    global_embedding: GlobalEmbeddingCertificate | None = None


def _cross(first: np.ndarray, second: np.ndarray, /) -> np.ndarray:
    return np.asarray(
        (
            first[1] * second[2] - first[2] * second[1],
            first[2] * second[0] - first[0] * second[2],
            first[0] * second[1] - first[1] * second[0],
        ),
        dtype=object,
    )


def _interiors_overlap(first: np.ndarray, second: np.ndarray, /) -> bool:
    """Separating-axis theorem for exact integer triangles/tetrahedra."""

    dimension = first.shape[1]
    if dimension == 2:
        edges = [
            corners[b] - corners[a]
            for corners in (first, second)
            for a, b in ((0, 1), (1, 2), (2, 0))
        ]
        axes = [np.asarray((-edge[1], edge[0]), dtype=object) for edge in edges]
    else:
        pairs = tuple(combinations(range(4), 2))
        first_edges = [first[b] - first[a] for a, b in pairs]
        second_edges = [second[b] - second[a] for a, b in pairs]
        axes = [
            _cross(corners[b] - corners[a], corners[c] - corners[a])
            for corners in (first, second)
            for a, b, c in combinations(range(4), 3)
        ]
        axes.extend(_cross(a, b) for a in first_edges for b in second_edges)
    for axis in axes:
        if not np.any(axis):
            continue
        left, right = first @ axis, second @ axis
        if max(left) <= min(right) or max(right) <= min(left):
            return False
    return True


def _lattice_overlap_images(
    points: np.ndarray, vectors: np.ndarray, /
) -> tuple[range, ...]:
    """Exact fractional bounding box of all potentially intersecting images."""

    dimension = vectors.shape[0]
    determinant = (
        _det2(vectors[0], vectors[1])
        if dimension == 2
        else _det3(vectors[0], vectors[1], vectors[2])
    )
    fractions = []
    for point in points:
        row = []
        for axis in range(dimension):
            replaced = vectors.copy()
            replaced[axis] = point
            numerator = (
                _det2(replaced[0], replaced[1])
                if dimension == 2
                else _det3(replaced[0], replaced[1], replaced[2])
            )
            row.append(Fraction(int(numerator), int(determinant)))
        fractions.append(row)
    ranges = []
    for axis in range(dimension):
        values = [row[axis] for row in fractions]
        diameter = max(values) - min(values)
        limit = diameter.numerator // diameter.denominator
        ranges.append(range(-limit, limit + 1))
    return tuple(ranges)


type _PeriodicImage = int | tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _PeriodicImageFrame:
    """Exact scientific lifts and their sufficient, bounded group image bank."""

    points: np.ndarray
    base_points: np.ndarray
    lattice: np.ndarray | None
    group_matrices: np.ndarray | None
    ranges: tuple[range, ...]
    exponent: int
    image_count: int
    residual: Fraction
    residual_float: float
    residual_bound: float
    image_exponents: np.ndarray | None = None

    def images(self) -> Iterator[_PeriodicImage]:
        if self.lattice is not None:
            return product(*self.ranges)
        return iter(range(self.image_count))

    def image_points(
        self, image: _PeriodicImage, rows: slice = slice(None), /
    ) -> np.ndarray:
        if self.lattice is not None:
            if not isinstance(image, tuple):
                raise TypeError("A lattice image requires its integer exponent tuple.")
            _reserve_periodic_exact_terms(self.points[rows].size)
            return self.points[rows] + _periodic_exact_product(
                np.asarray(image, dtype=object), self.lattice
            )
        matrices = self.group_matrices
        if matrices is None or not isinstance(image, int):
            raise TypeError("A finite isometry image requires its group-bank index.")
        matrix = matrices[image]
        _reserve_periodic_exact_terms(
            self.base_points[rows].size + self.base_points.shape[1]
        )
        return _periodic_exact_product(
            self.base_points[rows], matrix[:-1, :-1].T
        ) + matrix[:-1, -1] * (1 << (-2 * self.exponent))


def _reserve_periodic_exact_terms(work: int, storage_upper: int = 0, /) -> None:
    """Reserve actual scalar/term visits on the existing scientific ledger."""
    from ..discretization._coordinate_enclosure import _COORDINATE_BUDGET

    budget = _COORDINATE_BUDGET.get()
    if budget is not None:
        budget.admit_work_bound(work)
        budget.reserve(work, storage_upper)


def _periodic_exact_product(left: np.ndarray, right: np.ndarray, /) -> np.ndarray:
    """Object matmul with its actual dense dot-product term bank admitted."""
    rows = left.shape[0] if left.ndim == 2 else 1
    columns = right.shape[1] if right.ndim == 2 else 1
    _reserve_periodic_exact_terms(rows * left.shape[-1] * columns)
    return left @ right


def _exact_periodic_group_element(
    cell: PeriodicIsometryGroup,
    exponents: np.ndarray,
    /,
    *,
    prepared_generators: tuple[
        tuple[tuple[tuple[Fraction, ...], ...], ...], tuple[int, ...]
    ]
    | None = None,
) -> np.ndarray:
    """Compose authored dyadic isometries over Q, without matrix roundoff."""
    from ..discretization._periodic_topology import (
        _exact_periodic_element,
        _exact_periodic_generators,
    )

    matrices, orders = (
        _exact_periodic_generators(cell)
        if prepared_generators is None
        else prepared_generators
    )
    return np.asarray(
        _exact_periodic_element(
            matrices, orders, tuple(int(value) for value in exponents)
        ),
        dtype=object,
    )


def _exact_periodic_dyadic_bank(values: np.ndarray, /) -> tuple[np.ndarray, int]:
    """Pack exact composed rational dyadics in one immutable integer frame."""
    _reserve_periodic_exact_terms(3 * values.size)
    fractions = [
        value if isinstance(value, Fraction) else Fraction(float(value))
        for value in values.reshape(-1)
    ]
    exponent = min(
        (-value.denominator.bit_length() + 1 for value in fractions), default=0
    )
    integers = np.asarray(
        [
            value.numerator * ((1 << -exponent) // value.denominator)
            for value in fractions
        ],
        dtype=object,
    ).reshape(values.shape)
    return integers, exponent


def _affine_periodic_source_support_boxes(
    coordinates: np.ndarray,
    roots: np.ndarray,
    shifts: np.ndarray,
    cell: PeriodicIsometryGroup,
    /,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Whole affine-cell support from exact SCI lifts, not displayed corners."""
    from ..discretization._periodic_topology import (
        _exact_periodic_element,
        _exact_periodic_generators,
    )

    matrices, orders = _exact_periodic_generators(cell)
    lower, upper = None, None
    for vertex, root in enumerate(roots):
        _reserve_periodic_exact_terms(3 * cell.ambient_dimension)
        original = np.asarray(
            [Fraction(float(value)) for value in coordinates[int(root)]],
            dtype=object,
        )
        matrix = np.asarray(
            _exact_periodic_element(
                matrices,
                orders,
                tuple(int(value) for value in shifts[vertex]),
            ),
            dtype=object,
        )
        point = _periodic_exact_product(matrix[:-1, :-1], original) + matrix[:-1, -1]
        if lower is None:
            lower, upper = point.copy(), point.copy()
        else:
            if upper is None:
                raise RuntimeError("Periodic affine support lost its paired upper bound.")
            lower, upper = np.minimum(lower, point), np.maximum(upper, point)
    if lower is None or upper is None:
        raise ValueError("Periodic affine support requires a nonempty scientific source.")
    return ((lower, upper),)


def _periodic_overlap_image_exponents(
    cell: PeriodicIsometryGroup,
    source_boxes: tuple[tuple[np.ndarray, np.ndarray], ...],
    maximum_images: int,
    /,
) -> np.ndarray:
    """Complete SCI action bank from whole-source support, not power radii."""
    from ..discretization._coordinate_enclosure import outward
    from ..discretization._periodic_topology import (
        _exact_isometry_power,
        _exact_periodic_element,
        _exact_periodic_generators,
    )
    from ..geometry._periodic_embedding import _translation_ranges

    matrices, orders = _exact_periodic_generators(cell)
    finite_count = int(np.prod(cell.linear_orders, dtype=object))
    if finite_count > maximum_images:
        raise _PeriodicEmbeddingResourceError(
            f"Periodic overlap requires {finite_count} images, exceeding {maximum_images}."
        )
    if not source_boxes:
        raise ValueError("Periodic overlap requires complete nonempty source support.")
    # Explicit conservative CPython tuple/list + array bounds, not an RSS or
    # maximum-images proxy. These remain live until the caller's source scope
    # releases the resulting bank.
    _reserve_periodic_exact_terms(
        finite_count * len(orders),
        128
        + finite_count * (40 + 48 * len(orders))
        + finite_count * len(source_boxes) * (512 + 96 * cell.ambient_dimension),
    )
    finite_actions = tuple(product(*(range(order) for order in cell.linear_orders)))
    transformed_boxes = []
    for action in finite_actions:
        matrix = _exact_periodic_element(matrices, orders, action)
        for lower, upper in source_boxes:
            if (
                len(lower) != cell.ambient_dimension
                or len(upper) != cell.ambient_dimension
            ):
                raise ValueError(
                    "Periodic source support must have one bound per ambient axis."
                )
            if any(
                Fraction(lo) > Fraction(hi) for lo, hi in zip(lower, upper, strict=True)
            ):
                raise ValueError("Periodic whole-source support bounds are reversed.")
            lows, highs = [], []
            for row in matrix[:-1]:
                _reserve_periodic_exact_terms(2 * cell.ambient_dimension + 2)
                low = row[-1] + sum(
                    (
                        weight * Fraction(lo if weight >= 0 else hi)
                        for weight, lo, hi in zip(row[:-1], lower, upper, strict=True)
                    ),
                    Fraction(0),
                )
                high = row[-1] + sum(
                    (
                        weight * Fraction(hi if weight >= 0 else lo)
                        for weight, lo, hi in zip(row[:-1], lower, upper, strict=True)
                    ),
                    Fraction(0),
                )
                lows.append(outward(low, -np.inf))
                highs.append(outward(high, np.inf))
            if not np.all(np.isfinite(lows)) or not np.all(np.isfinite(highs)):
                raise ValueError(
                    "Periodic whole-source support exceeds finite enclosure range."
                )
            transformed_boxes.append((np.asarray(lows), np.asarray(highs)))
    translations = []
    for matrix, period in zip(matrices, cell.translation_periods, strict=True):
        if period:
            translation = _exact_isometry_power(matrix, period)
            translations.append(tuple(row[-1] for row in translation[:-1]))
    ranges = _translation_ranges(transformed_boxes, tuple(translations))
    image_count = finite_count * int(
        np.prod([len(interval) for interval in ranges], dtype=object)
    )
    if image_count > maximum_images:
        raise _PeriodicEmbeddingResourceError(
            f"Periodic overlap requires {image_count} images, exceeding {maximum_images}."
        )
    _reserve_periodic_exact_terms(
        image_count * len(orders),
        128 + image_count * (64 + 56 * len(orders)),
    )
    actions = []
    for finite in finite_actions:
        for translated in product(*ranges):
            iterator = iter(translated)
            actions.append(
                tuple(
                    value if order else value + period * next(iterator)
                    for value, order, period in zip(
                        finite, orders, cell.translation_periods, strict=True
                    )
                )
            )
    result = np.asarray(actions, dtype=np.int64).reshape((image_count, len(orders)))
    result.setflags(write=False)
    return result


def _prepare_periodic_image_frame(
    coordinates: np.ndarray,
    roots: np.ndarray,
    vertex_shifts: np.ndarray,
    cell: PeriodicCell | PeriodicIsometryGroup,
    maximum_images: int,
    /,
    *,
    image_exponents: np.ndarray | None = None,
    source_support_boxes: tuple[tuple[np.ndarray, np.ndarray], ...] | None = None,
) -> _PeriodicImageFrame:
    """Prepare the same exact image arithmetic for embedding and source support."""
    dimension = coordinates.shape[1]
    lattice, group_matrices = None, None
    if (
        isinstance(cell, PeriodicIsometryGroup)
        and any(order == 0 for order in cell.orders)
        and image_exponents is None
        and source_support_boxes is not None
    ):
        image_exponents = _periodic_overlap_image_exponents(
            cell,
            source_support_boxes,
            maximum_images,
        )
    ranges: tuple[range, ...] = ()
    if isinstance(cell, PeriodicCell):
        _reserve_periodic_exact_terms(3 * (coordinates.size + cell.vectors.size))
        values, exponent = _dyadic_integers(
            np.concatenate((coordinates, np.asarray(cell.vectors)))
        )
        actual, lattice = values[: coordinates.shape[0]], values[coordinates.shape[0] :]
        _reserve_periodic_exact_terms(actual.size)
        points = actual[roots] + _periodic_exact_product(
            vertex_shifts.astype(object), lattice
        )
        base_points = points
        _reserve_periodic_exact_terms(3 * actual.size + 2)
        residual_integer = max(np.abs(actual - points).reshape(-1), default=0)
        residual = Fraction(int(residual_integer)) * Fraction(2) ** exponent
        scale = max(
            1.0,
            float(np.max(np.abs(coordinates))),
            float(np.max(np.abs(vertex_shifts @ np.asarray(cell.vectors)), initial=0.0)),
        )
        residual_bound = 64.0 * np.finfo(np.float64).eps * scale
        if image_exponents is None:
            ranges = _lattice_overlap_images(points, lattice)
            ranges = tuple(
                value if periodic else range(1)
                for value, periodic in zip(ranges, cell.periodic_axes, strict=True)
            )
            image_count = int(np.prod([len(value) for value in ranges], dtype=object))
        else:
            actions = np.asarray(image_exponents)
            if (
                actions.ndim != 2
                or actions.shape[1] != cell.rank
                or not np.issubdtype(actions.dtype, np.integer)
                or len(actions) == 0
            ):
                raise ValueError(
                    "Explicit lattice images require nonempty integer action rows."
                )
            _reserve_periodic_exact_terms(3 * actions.size)
            lower = [int(value) for value in np.min(actions, axis=0)]
            upper = [int(value) for value in np.max(actions, axis=0)]
            image_count = int(
                np.prod(
                    [high - low + 1 for low, high in zip(lower, upper, strict=True)],
                    dtype=object,
                )
            )
            if image_count > maximum_images:
                raise _PeriodicEmbeddingResourceError(
                    f"Periodic embedding requires {image_count} images, exceeding {maximum_images}."
                )
            if (
                image_count != len(actions)
                or len(np.unique(actions, axis=0)) != image_count
            ):
                raise ValueError(
                    "Explicit lattice actions must form one complete rectangular image bank."
                )
            ranges = tuple(
                range(low, high + 1) for low, high in zip(lower, upper, strict=True)
            )
    else:
        if image_exponents is None:
            if any(order == 0 for order in cell.orders):
                raise ValueError(
                    "Mixed rotational/translational embedding needs a bounded fundamental domain."
                )
            image_count = int(np.prod(cell.orders, dtype=object))
            if image_count > maximum_images:
                raise _PeriodicEmbeddingResourceError(
                    f"Periodic embedding requires {image_count} images, exceeding {maximum_images}."
                )
            actions = np.asarray(
                tuple(product(*(range(order) for order in cell.orders))),
                dtype=np.int64,
            ).reshape((image_count, cell.rank))
        else:
            actions = np.asarray(image_exponents)
            if (
                actions.ndim != 2
                or actions.shape[1] != cell.rank
                or not np.issubdtype(actions.dtype, np.integer)
            ):
                raise ValueError(
                    "Explicit image actions require one integer exponent per generator."
                )
            image_count = len(actions)
            if image_count > maximum_images:
                raise _PeriodicEmbeddingResourceError(
                    f"Periodic embedding requires {image_count} images, exceeding {maximum_images}."
                )
            if image_count == 0:
                raise ValueError("The explicit image action bank must not be empty.")
        from ..discretization._periodic_topology import _exact_periodic_generators

        prepared_generators = _exact_periodic_generators(cell)
        matrices = [
            _exact_periodic_group_element(
                cell, action, prepared_generators=prepared_generators
            )
            for action in actions
        ]
        vertex_matrices = np.asarray(
            [
                _exact_periodic_group_element(
                    cell, shift, prepared_generators=prepared_generators
                )
                for shift in vertex_shifts
            ],
            dtype=object,
        )
        actions = np.asarray(actions, dtype=np.int64)
        actions.setflags(write=False)
        packed, exponent = _exact_periodic_dyadic_bank(
            np.concatenate(
                (
                    coordinates.reshape(-1),
                    np.asarray(matrices).reshape(-1),
                    vertex_matrices.reshape(-1),
                )
            )
        )
        coordinate_count = coordinates.size
        matrix_count = image_count * (dimension + 1) ** 2
        actual = packed[:coordinate_count].reshape(coordinates.shape)
        group_matrices = packed[
            coordinate_count : coordinate_count + matrix_count
        ].reshape((image_count, dimension + 1, dimension + 1))
        vertex_matrices = packed[coordinate_count + matrix_count :].reshape(
            vertex_matrices.shape
        )
        base_points = np.empty_like(actual)
        for vertex, matrix in enumerate(vertex_matrices):
            _reserve_periodic_exact_terms(2 * dimension)
            base_points[vertex] = _periodic_exact_product(
                actual[roots[vertex]], matrix[:-1, :-1].T
            ) + matrix[:-1, -1] * (1 << -exponent)
        _reserve_periodic_exact_terms(5 * actual.size)
        points = base_points * (1 << -exponent)
        difference = actual * (1 << (-2 * exponent)) - points
        residual_integer = max(np.abs(difference).reshape(-1), default=0)
        residual = Fraction(int(residual_integer)) * Fraction(2) ** (3 * exponent)
        residual_bound = cell.tolerance
        squared = max(
            (sum(int(value) ** 2 for value in row) for row in difference), default=0
        )
        if (
            Fraction(squared) * Fraction(2) ** (6 * exponent)
            > Fraction(cell.tolerance) ** 2
        ):
            raise ValueError(
                "The scientific isometry lift exceeds its declared Euclidean construction tolerance."
            )
    if image_count > maximum_images:
        raise _PeriodicEmbeddingResourceError(
            f"Periodic embedding requires {image_count} images, exceeding {maximum_images}."
        )
    residual_float = float(residual)
    if Fraction(residual_float) < residual:
        residual_float = float(np.nextafter(residual_float, np.inf))
    if residual > Fraction(float(residual_bound)):
        raise ValueError(
            "The exact scientific group lift exceeds its declared coordinate construction bound."
        )
    for values in (points, base_points, lattice, group_matrices):
        if values is not None:
            values.setflags(write=False)
    return _PeriodicImageFrame(
        points,
        base_points,
        lattice,
        group_matrices,
        ranges,
        exponent,
        image_count,
        residual,
        residual_float,
        residual_bound,
        actions if isinstance(cell, PeriodicIsometryGroup) else None,
    )


class _PeriodicEmbeddingResourceError(ValueError):
    """An original image/candidate allowance refused before scientific publication."""


@contextmanager
def _embedding_storage(bound: int, /) -> Iterator[NativeHostStorageWorkspace | None]:
    budget = current_native_execution_budget()
    if budget is None:
        yield None
    else:
        # Each index lifetime borrows the same original managed pool.
        with NativeHostStorageWorkspace(budget) as owned:
            owned.set_bound(bound)
            yield owned


def _integer_cell_bounds(
    corners: np.ndarray, scale_bits: int, /
) -> tuple[np.ndarray, np.ndarray]:
    """Outward float64 index boxes; exact integer predicates remain authoritative."""
    lower, upper = np.min(corners, axis=1), np.max(corners, axis=1)
    denominator = 1 << scale_bits
    lower_float = np.empty(lower.shape, dtype=np.float64)
    upper_float = np.empty(upper.shape, dtype=np.float64)
    for row in range(lower.shape[0]):
        for axis in range(lower.shape[1]):
            lower_float[row, axis] = np.nextafter(
                float(Fraction(int(lower[row, axis]), denominator)),
                -np.inf,
            )
            upper_float[row, axis] = np.nextafter(
                float(Fraction(int(upper[row, axis]), denominator)),
                np.inf,
            )
    return lower_float, upper_float


def _image_scale_bits(frame: _PeriodicImageFrame, dimension: int, /) -> int:
    """One monotone power-of-two index scale covers every exact group image."""
    image_bank = frame.lattice if frame.lattice is not None else frame.group_matrices
    if image_bank is None:
        raise RuntimeError("The exact image frame must retain its group operands.")
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
    frame: _PeriodicImageFrame,
    connectivity: np.ndarray,
    first_bvh: PackedBVH,
    lower: np.ndarray,
    upper: np.ndarray,
    scale_bits: int,
    image_storage: int,
    tree_scratch: int,
    visit: Callable[[int], None],
    prepare_image: Callable[[], None],
    record_storage: Callable[[NativeHostStorageWorkspace | None], None],
    /,
) -> Generator[tuple[int, int, np.ndarray, _PeriodicImage], None, None]:
    for shift in frame.images():
        with _embedding_storage(image_storage) as owner:
            prepare_image()
            imaged = frame.image_points(shift)[connectivity]
            image_lower, image_upper = np.min(imaged, axis=1), np.max(imaged, axis=1)
            identity = shift == 0 if isinstance(shift, int) else not any(shift)
            second_bvh = prepare_bvh(
                *_integer_cell_bounds(imaged, scale_bits),
                policy=BVHBuildPolicy(leaf_size=1),
                dtype=np.float64,
            )
            if owner is not None:
                owner.retain_owner(second_bvh)
                owner.set_bound(owner.bound - tree_scratch)
            record_storage(owner)

            def retain_host(value: object) -> None:
                if owner is None:
                    raise RuntimeError(
                        "Host-copy retention requires its actual child owner."
                    )
                if not isinstance(value, tuple):
                    raise TypeError("The canonical BVH host bank must be a tuple.")
                copied_bytes = 0
                for array in value:
                    if not isinstance(array, np.ndarray):
                        raise TypeError(
                            "The canonical BVH host bank must contain NumPy arrays."
                        )
                    copied_bytes += array.nbytes
                owner.retain_owner(value)
                # Actual retained copy owners replace their admitted raw bytes.
                owner.set_bound(owner.bound - copied_bytes)
                record_storage(owner)

            for first_rows, second_rows in bvh_overlap_pair_blocks(
                first_bvh,
                second_bvh,
                include_touching=True,
                visit=visit,
                maximum_block_pairs=256,
                retain_owner=None if owner is None else retain_host,
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
    mesh: CellMesh,
    topology: PeriodicMeshTopology,
    maximum_images: int,
    maximum_pairs: int,
    record_work: Callable[[int, int, int], None] | None,
    record_unmanaged: Callable[[int], None] | None,
    /,
) -> PeriodicEmbeddingEvidence:
    block = mesh.blocks[0]
    dimension = mesh.ambient_dimension
    vertices, cells = mesh.coordinates.shape[0], block.cell_count
    group = topology.cell
    matrix_count = (
        1 if isinstance(group, PeriodicCell) else int(np.prod(group.orders, dtype=object))
    )
    if matrix_count > maximum_images:
        raise _PeriodicEmbeddingResourceError(
            f"Periodic embedding requires {matrix_count} images, exceeding {maximum_images}."
        )
    # IEEE float dyadic lifting multiplies at most three integer factors. 1024
    # bytes cover each resulting Python integer; indexed corners share owners.
    depth = max(1, cells.bit_length())
    tree_scratch = 4096 * cells * (dimension + 1)
    query_scratch = 256 * (depth + 1) * (512 + 128 * dimension)
    frame_storage = 1024 * (
        8 * vertices * dimension + matrix_count * (dimension + 1) ** 2
    )
    storage_upper = frame_storage + tree_scratch + query_scratch
    setup_bound = 80 * (vertices * dimension + matrix_count * (dimension + 1) ** 2)
    aabb_tests = exact_tests = 0
    budget = current_native_execution_budget()

    def visit(count: int) -> None:
        nonlocal aabb_tests
        if budget is not None:
            budget.charge(work=count, geometry_queries=count)
        aabb_tests += count
        if record_work is not None:
            record_work(setup_bound, aabb_tests, exact_tests)

    with _embedding_storage(storage_upper) as source_owner:
        if budget is not None:
            budget.admit_work_bound(setup_bound)
            budget.charge(work=setup_bound)
        if record_work is not None:
            record_work(setup_bound, aabb_tests, exact_tests)
        coordinates = np.asarray(mesh.coordinates)
        connectivity = np.asarray(block.vertices)
        cell_ids = np.asarray(block.global_ids)
        frame = _prepare_periodic_image_frame(
            coordinates,
            np.asarray(topology.vertex_representatives),
            np.asarray(topology.vertex_shifts),
            group,
            maximum_images,
        )
        corners = frame.points[connectivity]
        lower, upper = np.min(corners, axis=1), np.max(corners, axis=1)
        scale_bits = _image_scale_bits(frame, dimension)
        policy = BVHBuildPolicy(leaf_size=1)
        build_work = 64 * cells * (dimension + depth)
        if budget is not None:
            budget.admit_work_bound(build_work)
            budget.charge(work=build_work)
        setup_bound += build_work
        first_bvh = prepare_bvh(
            *_integer_cell_bounds(corners, scale_bits), policy=policy, dtype=np.float64
        )
        if source_owner is not None:
            source_owner.retain_owner(first_bvh)
            source_owner.set_bound(source_owner.bound - tree_scratch - query_scratch)
        unmanaged_peak = 0

        def record_storage(image_owner: NativeHostStorageWorkspace | None) -> None:
            nonlocal unmanaged_peak
            source_bytes = (
                0 if source_owner is None else source_owner.logical_unmanaged_bytes_upper
            )
            image_bytes = (
                0 if image_owner is None else image_owner.logical_unmanaged_bytes_upper
            )
            unmanaged_peak = max(unmanaged_peak, source_bytes + image_bytes)
            if record_unmanaged is not None:
                record_unmanaged(unmanaged_peak)

        def prepare_image() -> None:
            nonlocal setup_bound
            if budget is not None:
                budget.admit_work_bound(build_work)
                budget.charge(work=build_work)
            setup_bound += build_work
            if record_work is not None:
                record_work(setup_bound, aabb_tests, exact_tests)

        record_storage(None)
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
        )
        with closing(pairs):
            for first, second, imaged_cell, shift in pairs:
                if candidates == maximum_pairs:
                    raise _PeriodicEmbeddingResourceError(
                        "Periodic embedding exhausted its candidate-pair budget."
                    )
                candidates += 1
                if budget is not None:
                    budget.charge(work=1, geometry_queries=1)
                exact_tests += 1
                if record_work is not None:
                    record_work(setup_bound, aabb_tests, exact_tests)
                if _interiors_overlap(corners[first], imaged_cell):
                    raise ValueError(
                        f"Quotient cells {first} and {second} "
                        f"(scientific IDs {int(cell_ids[first])}, {int(cell_ids[second])}) "
                        f"overlap under image {shift}."
                    )
        if record_work is not None:
            record_work(setup_bound, aabb_tests, exact_tests)
        return PeriodicEmbeddingEvidence(
            periodic_topology_id=topology.periodic_topology_id,
            image_count=frame.image_count,
            candidate_pair_count=candidates,
            coordinate_scope="exact_dyadic_group_lift",
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
                    "coordinate_scope": "exact_dyadic_group_lift",
                    # Preserve the canonical bound payload separately from host scalar metadata.
                    "coordinate_residual": frame.residual_float,
                    "coordinate_residual_bound": frame.residual_bound,
                }
            ),
        )


def certify_periodic_embedding(
    mesh: CellMesh,
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
) -> PeriodicEmbeddingEvidence:
    """Refuse full source-map quotient overlaps, including nontrivial self images.

    Mapped/mixed cells use the canonical continuous global certificate. The
    affine simplex control retains its exact dyadic separating-axis algorithm.
    Every image neighborhood has a full physical-hull sufficiency proof.
    """

    topology = _require_periodic_topology(mesh)
    if (
        geometry is not None
        or len(mesh.blocks) != 1
        or mesh.blocks[0].cell_kind != _SIMPLEX_KINDS.get(mesh.ambient_dimension)
        or (
            isinstance(topology.cell, PeriodicIsometryGroup)
            and any(order == 0 for order in topology.cell.orders)
        )
    ):
        from ..discretization._coordinate_enclosure import (
            coordinate_enclosure_budget,
            CoordinateEnclosureResourceError,
        )

        request_limits = MeshCertificateLimits() if limits is None else limits
        ledger = coordinate_enclosure_budget(
            request_limits.maximum_work_units, request_limits.maximum_scratch_bytes
        )
        try:
            with (
                ledger.activate(),
                ledger.bound_stage(
                    request_limits.maximum_work_units,
                    request_limits.maximum_scratch_bytes,
                ),
            ):
                from ..discretization._cell_geometry_validity import (
                    CellValidityCertificate,
                    CellValidityPolicy,
                    certify_cell_geometry_validity,
                )
                from ..geometry._mesh_certificates import certify_global_embedding

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
                        maximum_candidate_pairs=min(
                            maximum_pairs, limits_.maximum_candidate_pairs
                        ),
                        maximum_periodic_images=min(
                            maximum_images, limits_.maximum_periodic_images
                        ),
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
                    if not isinstance(validity, CellValidityCertificate):
                        raise TypeError(
                            "Periodic mapped validity must be its actual owning certificate."
                        )
                    validity.require_bound(spec, mesh=mesh)
                    if (
                        validity_policy is not None
                        and validity.policy_id != validity_policy.policy_id
                    ):
                        raise ValueError(
                            "Periodic mapped validity changes its authored original policy."
                        )
                    if not validity.all_certified:
                        raise ValueError("Periodic mapped validity is unresolved.")
                else:
                    if validity_policy is not None and not isinstance(
                        validity_policy, CellValidityPolicy
                    ):
                        raise TypeError(
                            "Periodic mapped validity policy must be CellValidityPolicy."
                        )
                    validity = certify_cell_geometry_validity(
                        spec, mesh=mesh, policy=validity_policy
                    )
                certificate = certify_global_embedding(
                    mesh, spec, validity, limits=limits_
                )
                if certificate.status != "certified":
                    resource_findings = tuple(
                        finding
                        for finding in certificate.findings
                        if finding.resource is not None
                    )
                    if resource_findings:
                        from ._contracts import MeshingFailure, MeshingFailureCategory

                        requested: list[tuple[str, int]] = []
                        achieved: list[tuple[str, int]] = []
                        for finding in resource_findings:
                            prefix = (
                                f"certificate:{certificate.certificate_id}:finding:{finding.finding_id}"
                                f":{finding.check}:resource:{finding.resource}"
                            )
                            requested.extend(
                                (f"{prefix}:{key}", value)
                                for key, value in finding.requested
                            )
                            achieved.extend(
                                (f"{prefix}:{key}", value)
                                for key, value in finding.achieved
                            )
                        raise MeshingFailure(
                            MeshingFailureCategory.RESOURCE_EXHAUSTED,
                            "Periodic mapped embedding exhausted its original coordinate proof allowance.",
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
                        f"Periodic mapped embedding {certificate.status}: "
                        + ", ".join(value.check for value in certificate.findings)
                    )
                from ..discretization._coordinate_enclosure import (
                    outward,
                    prepared_coordinate_source_bank,
                )

                _, _, executed = spec.resolve(mesh)
                source_bank = prepared_coordinate_source_bank(spec)
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
                residual = outward(coefficient_error, np.inf)
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
            from ._contracts import MeshingFailure, MeshingFailureCategory

            prefix = f"periodic:coordinate_source:resource:{error.resource}"
            raise MeshingFailure(
                MeshingFailureCategory.RESOURCE_EXHAUSTED,
                "Periodic source preparation exhausted its original coordinate proof allowance.",
                stage="periodic_embedding",
                requested=(
                    (f"{prefix}:limit", error.limit),
                    (f"{prefix}:requested", error.requested),
                ),
                achieved=(
                    (f"{prefix}:completed", error.completed),
                    (f"{prefix}:source_expression_work_units", ledger.work_units),
                    (f"{prefix}:source_expression_peak_bytes", ledger.peak_bytes_upper),
                    (
                        f"{prefix}:source_expression_native_charged_work_units",
                        ledger.native_charged_work_units,
                    ),
                    (
                        f"{prefix}:source_expression_retained_basis_bytes_at_catch",
                        ledger.retained_basis_bytes,
                    ),
                    (
                        f"{prefix}:source_expression_temporary_bytes_upper_at_catch",
                        ledger.temporary_bytes_upper,
                    ),
                ),
            ) from error
    block = mesh.blocks[0]
    dimension = mesh.ambient_dimension
    if block.cell_kind != _SIMPLEX_KINDS.get(dimension):
        raise ValueError("Periodic embedding requires full-dimensional simplices.")
    if maximum_images <= 0 or maximum_pairs <= 0:
        raise ValueError("Image and candidate-pair budgets must be positive.")
    return _affine_periodic_embedding(
        mesh,
        topology,
        maximum_images,
        maximum_pairs,
        record_work,
        record_unmanaged,
    )


_SIMPLEX_KINDS = {2: "triangle", 3: "tetrahedron"}


def _periodic_simplex_rows(mesh: CellMesh, /) -> tuple[np.ndarray, np.ndarray]:
    """Actual cell rows and SCI IDs in the same complete block order as geometry."""
    kind = _SIMPLEX_KINDS.get(mesh.topological_dimension)
    if kind is None or any(block.cell_kind != kind for block in mesh.blocks):
        raise ValueError(
            "Periodic simplex transactions require one complete simplex family."
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
) -> PeriodicCell:
    """Compile translational periodic constraints into their lattice.

    Every constraint must be a pure translation (identity linear part within
    its declared tolerance); the translations become the lattice vectors, one
    per ambient axis, in the given order. Rotational or reflecting pairings are
    boundary isometries, not translational torus topology, and are refused.
    """

    values = tuple(constraints)
    if not values or not all(isinstance(item, PeriodicConstraint) for item in values):
        raise TypeError("constraints must be a nonempty sequence of PeriodicConstraint.")
    vectors = []
    for index, constraint in enumerate(values):
        transform = np.asarray(constraint.transform, dtype=np.float64)
        dimension = transform.shape[0] - 1
        deviation = np.max(np.abs(transform[:dimension, :dimension] - np.eye(dimension)))
        if deviation > constraint.tolerance:
            raise ValueError(
                f"constraints[{index}] is not a translation; boundary isometries "
                "do not define a translational periodic lattice."
            )
        vectors.append(transform[:dimension, dimension])
    matrix = np.stack(vectors)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            "A periodic lattice needs one translational constraint per ambient axis."
        )
    return PeriodicCell(matrix, origin=origin)


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

    cell: PeriodicCell
    points: np.ndarray
    representatives: np.ndarray
    orbits: np.ndarray
    shifts: np.ndarray
    tolerance: float = eqx.field(static=True)
    orbits_id: str = eqx.field(static=True)

    def __init__(
        self, points: ArrayLike, cell: PeriodicCell, /, *, tolerance: float = 0.0
    ) -> None:
        if not isinstance(cell, PeriodicCell):
            raise TypeError("cell must be a PeriodicCell.")
        array = np.asarray(points, dtype=np.float64)
        if (
            array.ndim != 2
            or array.shape[0] == 0
            or array.shape[1] != cell.ambient_dimension
        ):
            raise ValueError("points must have shape (n > 0, ambient dimension).")
        if not np.all(np.isfinite(array)):
            raise ValueError("points must be finite.")
        threshold = float(tolerance)
        if not np.isfinite(threshold) or threshold < 0.0:
            raise ValueError("tolerance must be finite and nonnegative.")
        if cell.rank != cell.ambient_dimension or not cell.fully_periodic:
            raise ValueError("Point orbits require a fully periodic full-rank lattice.")
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
    points: np.ndarray, cell: PeriodicCell, tolerance: float, /
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
            "Periodic point orbits are ambiguous: a copy lies farther than the "
            "tolerance from its orbit representative."
        )
    return roots, shifts


def publish_periodic_simplices(
    points: ArrayLike,
    simplices: ArrayLike,
    shifts: ArrayLike,
    cell: PeriodicCell,
    /,
    *,
    block_name: str = "periodic",
) -> CellMesh:
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
            "Periodic simplices must be (T, d + 1) with (T, d + 1, d) shifts."
        )
    corners = np.concatenate(
        (cells.reshape((-1, 1)), corner_shifts.reshape((-1, dimension))), axis=1
    )
    lifted, inverse = np.unique(corners, axis=0, return_inverse=True)
    representatives = lifted[:, 0]
    lifted_shifts = lifted[:, 1:]
    if np.unique(representatives).size != point_array.shape[0]:
        raise ValueError("Every representative point must be a simplex corner.")
    zero = np.all(lifted_shifts == 0, axis=1)
    # np.unique sorts rows, so the first row per representative is its
    # smallest shift; a zero-shift image takes precedence.
    first = np.searchsorted(representatives, np.arange(point_array.shape[0]))
    zero_rows = np.flatnonzero(zero)
    reference = first.copy()
    reference[representatives[zero_rows]] = zero_rows
    vertex_representatives = reference[representatives]
    vertex_shifts = lifted_shifts - lifted_shifts[vertex_representatives]
    coordinates = point_array[representatives] + lifted_shifts.astype(
        np.float64
    ) @ np.asarray(cell.vectors, dtype=np.float64)
    block = CellBlock(
        block_name,
        _SIMPLEX_KINDS[dimension],
        inverse.reshape(cells.shape).astype(np.int32),
    )
    carrier = CellMesh(coordinates, (block,))
    topology = PeriodicMeshTopology(carrier, cell, vertex_representatives, vertex_shifts)
    return CellMesh(coordinates, (block,), periodic_topology=topology)


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

    def __init__(self, mesh: CellMesh, /) -> None:
        periodic = mesh.periodic_topology
        if periodic is None:
            raise ValueError("Quotient evidence requires a periodic CellMesh.")
        report = periodic_orbit_measures(mesh)
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


class PeriodicMeshConstruction(StrictModule, NonTrainableState):
    """Periodic Delaunay mesh of point orbits with its certificates."""

    mesh: CellMesh
    orbits: PeriodicPointOrbits
    triangulation: PeriodicTriangulationEvidence
    quotient: PeriodicQuotientEvidence


def periodic_delaunay_mesh(
    orbits: PeriodicPointOrbits,
    /,
    *,
    initial_margin: float | None = None,
    maximum_images: int | None = None,
    maximum_simplices: int | None = None,
    block_name: str = "periodic",
) -> PeriodicMeshConstruction:
    """Construct and publish the periodic Delaunay mesh of point orbits.

    Budget refusals raise ``PeriodicImageBudgetError`` with the refused round's
    evidence; the published mesh carries its quotient topology and the
    construction reports the certified image neighborhood and quotient
    acceptance.
    """

    if not isinstance(orbits, PeriodicPointOrbits):
        raise TypeError("orbits must be PeriodicPointOrbits.")
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
    )
    return PeriodicMeshConstruction(
        mesh=mesh,
        orbits=orbits,
        triangulation=triangulation.evidence,
        quotient=PeriodicQuotientEvidence(mesh),
    )


def _quotient_simplices(
    mesh: CellMesh, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray, PeriodicCell, str]:
    """Representative points, quotient simplices and corner shifts of a mesh."""

    periodic = mesh.periodic_topology
    if periodic is None:
        raise ValueError(
            "Periodic refinement requires a CellMesh with periodic topology."
        )
    vertices, _ = _periodic_simplex_rows(mesh)
    if not isinstance(periodic.cell, PeriodicCell):
        raise TypeError("Translational quotient simplices require a PeriodicCell.")
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


def _edge_key(
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
                _edge_key(int(vertices[i]), shifts[i], int(vertices[j]), shifts[j])[0],
                i,
                j,
            )
            for i in range(arity)
            for j in range(i + 1, arity)
        ]
        if len({key for key, _, _ in edges}) != len(edges):
            raise ValueError(
                "A cell contains two members of one quotient edge orbit; orbit "
                "bisection requires distinct quotient edges per cell."
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
        # Ties are broken by the canonical key, so every lift decides alike.
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
            _, leads = _edge_key(int(vertices[i]), shifts[i], int(vertices[j]), shifts[j])
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


class PeriodicRefinement(StrictModule, NonTrainableState):
    """Orbit-preserving bisection of a periodic mesh.

    ``parent_cells`` names, per published child cell, the input cell (in input
    block order) it refines. ``bisected_edges`` counts quotient edge orbits
    split, each at one new quotient vertex shared by every seam copy.
    """

    mesh: CellMesh
    source: CellMesh
    source_geometry: CellGeometrySpec
    parent_reference_vertices: np.ndarray
    parent_cells: np.ndarray
    cell_generations: np.ndarray
    bisected_edges: int = eqx.field(static=True)
    quotient: PeriodicQuotientEvidence
    refinement_id: str = eqx.field(static=True)
    previous: PeriodicRefinement | None = None
    retired: PeriodicRefinement | None = None

    def __init__(
        self,
        *,
        mesh: CellMesh,
        source: CellMesh,
        source_geometry: CellGeometrySpec,
        parent_reference_vertices: np.ndarray,
        parent_cells: np.ndarray,
        cell_generations: np.ndarray,
        bisected_edges: int,
        quotient: PeriodicQuotientEvidence,
        previous: PeriodicRefinement | None = None,
        retired: PeriodicRefinement | None = None,
    ) -> None:
        from ..discretization._cell_geometry_validity import cell_geometry_id
        from ..discretization._coordinate_enclosure import (
            coordinate_corner_images,
            prepared_coordinate_source_bank,
            rounded_point,
        )

        if type(source_geometry) is not CellGeometrySpec:
            raise TypeError(
                "Periodic history requires its actual owning source coordinate specification."
            )
        bank = prepared_coordinate_source_bank(source_geometry)
        elements, routes, _ = source_geometry._resolve(source, exact_source_prepared=True)
        _reserve_periodic_exact_terms(
            64 * sum(block.cell_count for block in source.blocks)
        )
        for block, element, rows in zip(source.blocks, elements, routes, strict=True):
            carrier = np.asarray(source.coordinates)[np.asarray(block.vertices)]
            for row, expected in zip(rows, carrier, strict=True):
                corners = coordinate_corner_images(
                    element, tuple(bank[int(index)] for index in row)
                )
                if corners is None or not np.array_equal(
                    np.asarray(tuple(rounded_point(point) for point in corners)).view(
                        np.uint64
                    ),
                    np.asarray(expected, dtype=np.float64).view(np.uint64),
                ):
                    raise ValueError(
                        "Periodic history source geometry differs from its actual source carrier."
                    )
        dimension = source.topological_dimension
        if mesh.topological_dimension != dimension or dimension not in (2, 3):
            raise ValueError("Periodic history requires matching simplex dimensions.")
        source_topology = _require_periodic_topology(source)
        target_topology = _require_periodic_topology(mesh)
        if source_topology is None or target_topology is None:
            raise ValueError(
                "Periodic history requires actual source and target quotient descriptors."
            )
        from ..discretization._periodic_topology import _identification_id

        if _identification_id(source_topology.cell) != _identification_id(
            target_topology.cell
        ):
            raise ValueError("Periodic history changes the original group action.")
        if any(
            block.cell_kind != _SIMPLEX_KINDS[dimension]
            for block in (*source.blocks, *mesh.blocks)
        ):
            raise ValueError("Periodic histories require actual simplex carriers.")
        count = sum(block.cell_count for block in mesh.blocks)
        source_count = sum(block.cell_count for block in source.blocks)
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
                "Periodic history has invalid retained parent, reference or generation banks."
            )
        if (
            type(quotient) is not PeriodicQuotientEvidence
            or quotient.periodic_topology_id != target_topology.periodic_topology_id
            or not quotient.all_cells_valid
        ):
            raise ValueError(
                "Periodic history requires its actual target quotient evidence."
            )
        if any(
            node is not None and type(node) is not PeriodicRefinement
            for node in (previous, retired)
        ):
            raise TypeError(
                "Periodic predecessor and retirement require actual owning histories."
            )
        active, visited = set(), set()

        def visit(node: PeriodicRefinement | None) -> None:
            if node is None or id(node) in visited:
                return
            if id(node) in active:
                raise ValueError(
                    "Periodic history contains a cyclic predecessor or retirement."
                )
            active.add(id(node))
            visit(node.previous)
            visit(node.retired)
            active.remove(id(node))
            visited.add(id(node))

        visit(previous)
        visit(retired)
        _reserve_periodic_exact_terms(64 * count)
        measures = [Fraction(0) for _ in range(source_count)]
        charts = set()
        for parent, row in zip(parents, reference, strict=True):
            vertices = tuple(
                tuple(Fraction(float(value)) for value in vertex) for vertex in row
            )
            if any(
                any(value < 0 for value in vertex) or sum(vertex) > 1
                for vertex in vertices
            ):
                raise ValueError(
                    "Periodic child reference vertices leave their actual parent simplex."
                )
            chart = (int(parent), tuple(sorted(vertices)))
            if chart in charts:
                raise ValueError(
                    "Periodic history duplicates a retained sibling simplex."
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
            measure = abs(_det2(*edges) if dimension == 2 else _det3(*edges))
            if measure == 0:
                raise ValueError(
                    "Periodic history contains a degenerate sibling reference simplex."
                )
            measures[int(parent)] += measure
        if any(measure != 1 for measure in measures):
            raise ValueError(
                "Periodic sibling reference simplices do not conserve every actual source cell."
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


def _periodic_child_blocks(
    source: CellMesh, vertices: np.ndarray, parents: np.ndarray
) -> tuple[tuple[CellBlock, ...], np.ndarray]:
    """Keep source block metadata and return the actual child publication order."""
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
            CellBlock(
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
    mesh: CellMesh,
    cells: ArrayLike | None,
    edge_orbits: ArrayLike | None,
    source_geometry: CellGeometrySpec,
    /,
) -> PeriodicRefinement:
    """Bisect all lifted copies of each selected quotient edge together."""

    topology = _require_periodic_topology(mesh)
    if topology is None or not isinstance(topology.cell, PeriodicIsometryGroup):
        raise ValueError("Boundary-isometry refinement requires an isometry topology.")
    source_vertices, _ = _periodic_simplex_rows(mesh)
    points = np.asarray(mesh.coordinates).tolist()
    vertices = source_vertices.tolist()
    parents = list(range(len(vertices)))
    generations = [0 for _ in vertices]
    representatives = np.asarray(topology.vertex_representatives).tolist()
    barycentric = [np.eye(source_vertices.shape[1]) for _ in vertices]
    shifts = np.asarray(topology.vertex_shifts).tolist()
    edges = _simplex_entity_corners(mesh, 1)
    orbit, _, anchors = (np.asarray(value) for value in topology.orbits(1))
    leaders = np.asarray(topology.orbit_representatives(1))
    lengths = np.linalg.norm(
        np.asarray(points)[edges[:, 1]] - np.asarray(points)[edges[:, 0]], axis=1
    )
    selected = set(orbit.tolist())
    if cells is not None:
        marked = np.asarray(cells)
        if marked.ndim != 1 or not np.issubdtype(marked.dtype, np.integer):
            raise TypeError("cells must be a one-dimensional integer array.")
        if np.any(marked < 0) or np.any(marked >= len(source_vertices)):
            raise ValueError("cells must index cells of the periodic mesh.")
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
    lifted = CellMesh(coordinates, successor_blocks)
    quotient = PeriodicMeshTopology(
        lifted, topology.cell, np.asarray(representatives), np.asarray(shifts)
    )
    successor = CellMesh(coordinates, successor_blocks, periodic_topology=quotient)
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
    )


def _bare_periodic_affine_source(mesh: CellMesh, /) -> CellGeometrySpec:
    """Bind exact stored source rows for the bare topological refinement API."""
    return CellGeometrySpec(
        {
            block.name: coordinate_lagrange_element(block.cell_kind, 1)
            for block in mesh.blocks
        },
        {block.name: block.vertices for block in mesh.blocks},
        mesh.coordinates,
        storage=mesh.storage,
    )


def refine_periodic_mesh(
    mesh: CellMesh,
    /,
    *,
    cells: ArrayLike | None = None,
    edge_orbits: ArrayLike | None = None,
    source_geometry: CellGeometrySpec | None = None,
) -> PeriodicRefinement:
    """Refine a periodic simplex mesh by quotient edge bisection.

    With ``cells=None`` every quotient edge of the input is bisected once,
    longest first (the four-triangle longest-edge partition in 2D, eight
    children per tetrahedron in 3D). Otherwise the longest quotient edge of
    each listed cell is bisected together with its whole orbit star, so
    conformity and periodicity hold without closure refinement.
    """
    if source_geometry is None:
        source_geometry = _require_periodic_topology(mesh).actual_geometry
        if source_geometry is None:
            source_geometry = _bare_periodic_affine_source(mesh)
    if cells is not None and edge_orbits is not None:
        raise ValueError("Select cells or quotient edge orbits, not both.")
    if edge_orbits is not None:
        marked_edges = np.asarray(edge_orbits)
        count = _require_periodic_topology(mesh).quotient.entities(1).count
        if marked_edges.ndim != 1 or not np.issubdtype(marked_edges.dtype, np.integer):
            raise TypeError("edge_orbits must be a one-dimensional integer array.")
        if np.any(marked_edges < 0) or np.any(marked_edges >= count):
            raise ValueError("edge_orbits must name quotient edges of the source.")

    if mesh.periodic_topology is not None and isinstance(
        _require_periodic_topology(mesh).cell, PeriodicIsometryGroup
    ):
        return _refine_isometry_mesh(mesh, cells, edge_orbits, source_geometry)
    points, simplices, shifts, cell, block_name = _quotient_simplices(mesh)
    vectors = np.asarray(cell.vectors, dtype=np.float64)
    state = _QuotientBisection(points, simplices, shifts, vectors)
    if edge_orbits is not None:
        topology = _require_periodic_topology(mesh)
        leaders = np.asarray(topology.orbit_representatives(1))[np.asarray(edge_orbits)]
        endpoints = _simplex_entity_corners(mesh, 1)[leaders]
        roots = np.asarray(topology.vertex_representatives)
        unique = np.unique(roots)
        local = np.searchsorted(unique, roots)
        vertex_shifts = np.asarray(topology.vertex_shifts)
        keys = {
            _edge_key(
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
            raise TypeError("cells must be a one-dimensional integer array.")
        if np.any(marked < 0) or np.any(marked >= simplices.shape[0]):
            raise ValueError("cells must index cells of the periodic mesh.")
        queue = sorted(
            {state.longest_edge(int(index)) for index in np.unique(marked)},
            key=lambda key: (-state.length(key), key),
        )
    bisected = 0
    for key in queue:
        if key in state.star:
            state.bisect(key)
            bisected += 1
    refined = publish_periodic_simplices(
        np.stack(state.points),
        np.stack(state.cells),
        np.stack(state.shifts),
        cell,
        block_name=block_name,
    )
    parents = np.asarray(state.parents, dtype=np.int32)
    refined_blocks, publication_order = _periodic_child_blocks(
        mesh, _periodic_simplex_rows(refined)[0], parents
    )
    lifted = CellMesh(np.asarray(refined.coordinates), refined_blocks)
    topology = _require_periodic_topology(refined)
    refined = CellMesh(
        np.asarray(refined.coordinates),
        refined_blocks,
        periodic_topology=PeriodicMeshTopology(
            lifted, cell, topology.vertex_representatives, topology.vertex_shifts
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


def _periodic_history_meshes(history: PeriodicRefinement | None) -> tuple[CellMesh, ...]:
    meshes = {}
    for node in _periodic_history_nodes(history):
        for mesh in (node.source, node.mesh):
            meshes.setdefault(mesh.mesh_id, mesh)
    return tuple(meshes.values())


def _periodic_scientific_block_rows(
    source: CellMeshingResult,
    history: PeriodicRefinement,
    /,
    *,
    limits: MeshingLimits,
) -> NDArray[np.int32]:
    """Bind presentation rows to their real retained original scientific blocks."""
    import jax

    from .._meshcore import current_native_execution_budget
    from ._bisection import _uniform_charge
    from ._contracts import MeshingFailure, MeshingFailureCategory

    origin = source.geometry.restriction_source
    if origin is None:
        raise ValueError(
            "Periodic scientific block lookup requires its declared restriction root."
        )
    if (
        history.mesh.mesh_id != source.mesh.mesh_id
        or history.mesh.numeric_version != source.mesh.numeric_version
    ):
        raise ValueError(
            "Periodic scientific blocks require their actual current retained history."
        )
    budget = current_native_execution_budget()
    work, storage = 0, 0

    def admit(amount: int, byte_count: int) -> None:
        nonlocal work, storage
        if (
            work + amount > limits.maximum_work_units
            or storage + byte_count > limits.maximum_scratch_bytes
        ):
            raise MeshingFailure(
                MeshingFailureCategory.RESOURCE_EXHAUSTED,
                "Periodic scientific block preparation exceeds its original allowance.",
                stage="scientific-class-preparation",
            )
        if budget is not None:
            budget.admit_work_bound(amount)
            _uniform_charge(amount, byte_count)
        work += amount
        storage += byte_count

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
                "Periodic scientific block ownership has an invalid or cyclic history."
            )
        if id(node) in visited:
            continue
        admit(3, 768)
        active.add(id(node))
        queue.append((node, True))
        for mesh in (node.source, node.mesh):
            if mesh.topology_id == origin.source_topology_id:
                owners.append(mesh)
        queue.extend(
            (child, False) for child in (node.previous, node.retired) if child is not None
        )
    if not owners:
        raise ValueError(
            "Periodic restriction declares a root with no true retained source owner."
        )
    candidates = (source.mesh, history.mesh, *owners)
    count = sum(block.cell_count for block in source.mesh.blocks)
    bank_rows = sum(
        sum(block.cell_count for block in mesh.blocks) + mesh.coordinates.shape[0]
        for mesh in candidates
    )
    admit(bank_rows + count, (bank_rows + count) * 256)
    # All genuine source leaves are fetched together, before any per-row iteration.
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
        mesh: CellMesh,
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
                "blocks": tuple((block.name, block.cell_kind) for block in mesh.blocks),
                "arrays": array_tree_fingerprint(bank),
            }
        )

    if identity(source.mesh, banks[0]) != identity(history.mesh, banks[1]):
        raise ValueError(
            "Periodic scientific block history has changed actual current mesh banks."
        )
    root = owners[0]
    root_bank = banks[2]
    expected = identity(root, root_bank)
    if any(
        identity(mesh, bank) != expected
        for mesh, bank in zip(owners[1:], banks[3:-1], strict=True)
    ):
        raise ValueError(
            "Periodic restriction has conflicting retained original scientific owners."
        )
    root_cells = {}
    root_vertices = np.asarray(root_bank[1])
    for block_index, (_, (vertices, identifiers)) in enumerate(
        zip(root.blocks, root_bank[2], strict=True)
    ):
        for cell, corners in zip(identifiers, vertices, strict=True):
            identifier = int(cell)
            if identifier in root_cells:
                raise ValueError(
                    "Periodic scientific root duplicates an original cell ID."
                )
            root_cells[identifier] = (
                block_index,
                tuple(int(value) for value in root_vertices[corners]),
            )
    result = np.empty((count,), dtype=np.int32)
    offset = 0
    for block, (parents, corners) in zip(source.mesh.blocks, banks[-1], strict=True):
        if parents.shape != (block.cell_count,) or corners.shape[0] != block.cell_count:
            raise ValueError(
                "Periodic presentation rows lack complete original SCI parent banks."
            )
        for parent, vertices in zip(parents, corners, strict=True):
            owner = root_cells.get(int(parent))
            if (
                owner is None
                or tuple(int(value) for value in vertices if value >= 0) != owner[1]
            ):
                raise ValueError(
                    "Periodic presentation has a missing, foreign or changed original SCI cell."
                )
            result[offset] = owner[0]
            offset += 1
    return result


def _periodic_vertex_stencils(
    history: PeriodicRefinement,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact dyadic construction stencils, reconciled on shared lifted vertices."""

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
                    "Periodic cell lifts disagree on a shared vertex's construction witness."
                )
            sources[vertex], weights[vertex] = row, values
    if np.any(sources[:, 0] < 0):
        raise ValueError("A periodic refinement contains an unused lifted vertex.")
    return sources, weights


def _periodic_entity_relations(
    source: CellMesh,
    target: CellMesh,
    vertex_sources: np.ndarray,
    *,
    coarsening: bool = False,
) -> list[EntityRelations]:
    from ._lineage import EntityLineageKind
    from ._topology_edit import entity_keys, EntityRelations

    records = []
    fine, coarse = (source, target) if coarsening else (target, source)
    for degree in range(1, source.topological_dimension):
        fine_rows = _simplex_entity_corners(fine, degree)
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
    source: CellMesh, target: CellMesh, prior: PeriodicRefinement | None
) -> tuple[PrescribedEntityIds, ...]:
    from ._topology_edit import entity_keys, PrescribedEntityIds

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
                        "Periodic scientific entity identity space is exhausted."
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
    history: PeriodicRefinement, prior: PeriodicRefinement | None
) -> CellTopologyEdit:
    from ..discretization._cell_geometry_transfer import NestedReferenceWitnesses
    from ._lineage import EntityLineageKind
    from ._topology_edit import CellTopologyEdit, EntityRelations, TopologyEditBlock

    source, target = history.source, history.mesh
    sources, weights = _periodic_vertex_stencils(history)
    count = sources.shape[0]
    vertex_ids = np.empty(count, dtype=np.int64)
    old_meshes = (source, *_periodic_history_meshes(prior))
    next_id = (
        max(int(np.max(np.asarray(old.vertex_global_ids))) for old in old_meshes) + 1
    )
    retired_vertices = {}
    for node in _periodic_history_nodes(prior):
        old_sources, old_weights = _periodic_vertex_stencils(node)
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
                    "Periodic scientific vertex identity space is exhausted."
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
                        "Periodic scientific cell identity space is exhausted."
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
            CellBlock(
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
    labeled_target = CellMesh(
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
            source, history.mesh, labeled_target, prior, order
        ),
        refinement=NestedReferenceWitnesses(
            fine_ids,
            source_cell_ids[parents],
            np.asarray(history.parent_reference_vertices),
        ),
    )


def _periodic_coarsening_parents(
    prepared: PreparedMeshAdaptation, history: PeriodicRefinement
) -> np.ndarray:
    """Complete compatible sibling families, closed across every split edge orbit."""

    from ._adaptation import MarkedMeshAdaptation
    from ._topology_edit import entity_keys, key_rows

    request = prepared.request
    if not isinstance(request, MarkedMeshAdaptation):
        raise TypeError("Periodic sibling coarsening requires a marked request.")
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
            "Periodic sibling cells lack their actual scientific constraint rows."
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
        endpoints = _simplex_entity_corners(fine, 1)[edge]
        # Explicit endpoint containment avoids treating a merely adjacent vertex
        # as an edge-protection conflict.
        touched = np.asarray(
            [np.all(np.isin(endpoints, corners)) for corners in fine_cells]
        )
        selected[np.unique(parents[touched])] = False
    # A label/material trace may not be merged across inconsistent child classes.
    degree = coarse.topological_dimension - 1
    construction, _ = _periodic_vertex_stencils(history)
    fine_facets = _simplex_entity_corners(fine, degree)
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
        for row, edge in enumerate(_simplex_entity_corners(coarse, 1).tolist())
    }
    edge_orbit = np.asarray(_require_periodic_topology(coarse).orbits(1)[0])
    stars = {}
    for parent, corners in enumerate(coarse_cells):
        for pair in combinations(corners.tolist(), 2):
            orbit = int(edge_orbit[by_pair[tuple(sorted(pair))]])
            stars.setdefault(orbit, set()).add(parent)
    present = set(_require_periodic_topology(fine).entity_keys(1))
    split = [
        index
        for index, key in enumerate(_require_periodic_topology(coarse).entity_keys(1))
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
    history: PeriodicRefinement, restored_parents: np.ndarray
) -> tuple[CellTopologyEdit, np.ndarray, np.ndarray, np.ndarray]:
    from ..discretization._cell_geometry_transfer import NestedReferenceWitnesses
    from ._lineage import EntityLineageKind
    from ._topology_edit import CellTopologyEdit, EntityRelations, TopologyEditBlock

    fine, original = history.mesh, history.source
    construction_sources, _ = _periodic_vertex_stencils(history)
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
    target = CellMesh(
        np.asarray(fine.coordinates)[used],
        tuple(
            CellBlock(block.name, block.cell_kind, block.cells, global_ids=block.cell_ids)
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
        periodic_orbits=_periodic_orbit_witness(fine, fine, target, history, used),
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
    source: CellMesh, history: PeriodicRefinement | None
) -> tuple[PeriodicEntityIdentityBank, ...]:
    """Read keyed allocation history from scientific owners, never live row order."""

    from ..discretization._periodic_topology import _identification_id
    from ._topology_edit import PeriodicEntityIdentityBank

    topology = _require_periodic_topology(source)
    identification = _identification_id(topology.cell)
    if history is not None and (
        history.mesh.topology_id != source.topology_id
        or not np.array_equal(history.mesh.coordinates, source.coordinates)
    ):
        raise ValueError("Periodic allocation history does not bind the source epoch.")
    owners = (source, *_periodic_history_meshes(history))
    result = []
    for degree in range(1, source.topological_dimension):
        known: dict[tuple[int, ...], int] = {}
        reverse: dict[int, tuple[int, ...]] = {}
        cursor = 0
        for owner in owners:
            periodic = _require_periodic_topology(owner)
            if _identification_id(periodic.cell) != identification:
                raise ValueError("Periodic allocation history changes identification.")
            recorded = periodic.allocator_next_ids[degree]
            if recorded < 0:
                raise ValueError("Periodic scientific allocation history is unresolved.")
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
                        "Periodic history reuses a scientific entity identity."
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


def _require_periodic_identity_bank(bank: PeriodicEntityIdentityBank, /) -> None:
    """Validate the immutable variable-arity identity payload without coercion."""
    from ._topology_edit import PeriodicEntityIdentityBank

    if not isinstance(bank, PeriodicEntityIdentityBank):
        raise TypeError("Periodic scientific identities require their canonical bank.")
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
            "Periodic scientific identity keys, IDs or allocation cursor are invalid."
        )


def _periodic_target_identity_banks(
    source: CellMesh,
    target: PeriodicMeshTopology,
    history: PeriodicRefinement | None,
    retained: tuple[PeriodicEntityIdentityBank, ...] = (),
    allocation_prior: tuple[CellMesh, PeriodicVertexOrbitWitness] | None = None,
) -> tuple[PeriodicEntityIdentityBank, ...]:
    current = _periodic_identity_banks(source, history)
    if allocation_prior is not None:
        current = _validated_periodic_allocation_prior(source, allocation_prior)
        if retained != current:
            raise ValueError(
                "Retained periodic identities must equal the validated prior complete allocation bank."
            )
    return _allocate_periodic_identity_banks(target, current, retained)


def _allocate_periodic_identity_banks(
    target: PeriodicMeshTopology,
    current: tuple[PeriodicEntityIdentityBank, ...],
    retained: tuple[PeriodicEntityIdentityBank, ...],
    /,
) -> tuple[PeriodicEntityIdentityBank, ...]:
    """Allocate only beyond the validated base's keyed high-water."""
    from ._topology_edit import PeriodicEntityIdentityBank

    result = []
    if retained and tuple(bank.degree for bank in retained) != tuple(
        bank.degree for bank in current
    ):
        raise ValueError("Retained periodic identities omit an entity dimension.")
    for index, bank in enumerate(current):
        known = dict(zip(bank.entity_keys, bank.entity_global_ids, strict=True))
        cursor = bank.allocator_next_id
        if retained:
            prior = retained[index]
            _require_periodic_identity_bank(prior)
            if prior.allocator_next_id != cursor:
                raise ValueError(
                    "Retained periodic identities do not bind the current allocation cursor."
                )
            historical = dict(
                zip(prior.entity_keys, prior.entity_global_ids, strict=True)
            )
            if len(historical) != len(prior.entity_keys) or any(
                historical.get(key) != identifier for key, identifier in known.items()
            ):
                raise ValueError(
                    "Retained periodic identities do not include the current scientific epoch."
                )
            known = historical
            cursor = prior.allocator_next_id
        for key in sorted(target.entity_keys(bank.degree)):
            if key not in known:
                if cursor > np.iinfo(np.int64).max:
                    raise ValueError(
                        "Periodic scientific identity allocation is exhausted."
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
    source: CellMesh,
    prior: tuple[CellMesh, PeriodicVertexOrbitWitness],
    /,
) -> tuple[PeriodicEntityIdentityBank, ...]:
    """Validate the complete prior chain iteratively, without a private depth cap."""
    from ._topology_edit import PeriodicVertexOrbitWitness

    chain: list[tuple[CellMesh, PeriodicVertexOrbitWitness]] = []
    seen: set[int] = set()
    cursor: tuple[CellMesh, PeriodicVertexOrbitWitness] | None = prior
    while cursor is not None:
        if not isinstance(cursor, tuple) or len(cursor) != 2:
            raise TypeError(
                "A periodic allocation prior requires its bound target and canonical witness."
            )
        previous, witness = cursor
        if not isinstance(previous, CellMesh) or not isinstance(
            witness, PeriodicVertexOrbitWitness
        ):
            raise TypeError(
                "A periodic allocation prior requires its bound target and canonical witness."
            )
        identity = id(witness)
        if identity in seen:
            raise ValueError(
                "A periodic allocation prior contains a cyclic witness chain."
            )
        seen.add(identity)
        chain.append((previous, witness))
        cursor = witness.allocation_prior
    current: tuple[PeriodicEntityIdentityBank, ...] | None = None
    for previous, witness in reversed(chain):
        periodic = _require_periodic_topology(previous)
        validated = _bind_periodic_target(source, previous, witness, current)
        bound = _require_periodic_topology(validated)
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
                "A periodic allocation prior does not match its actual bound target identities."
            )
        current = witness.quotient_entities
    if current is None:
        raise ValueError("A periodic allocation prior omits its actual bound target.")
    return current


def bind_periodic_topology_edit(
    source: CellMesh, target: CellMesh, witness: PeriodicVertexOrbitWitness, /
) -> CellMesh:
    """Validate explicit target-global-ID orbits and keyed scientific identities."""

    from ._topology_edit import PeriodicVertexOrbitWitness

    if not isinstance(witness, PeriodicVertexOrbitWitness):
        raise TypeError("A periodic edit requires its explicit vertex-orbit witness.")
    current = (
        None
        if witness.allocation_prior is None
        else _validated_periodic_allocation_prior(source, witness.allocation_prior)
    )
    return _bind_periodic_target(source, target, witness, current)


def prepare_periodic_edit_target(source: CellMesh, target: CellMesh, /) -> CellMesh:
    """Stage canonical quotient IDs before preparing the actual overlap certificate."""
    from ..discretization._cell_complex import PolyhedralConnectivity
    from ..discretization._periodic_topology import _identification_id

    original = _require_periodic_topology(source)
    descriptor = _require_periodic_topology(target)
    if _identification_id(original.cell) != _identification_id(descriptor.cell):
        raise ValueError(
            "A staged periodic edit changes its original identification group."
        )
    banks = _periodic_target_identity_banks(source, descriptor, None)
    ids = {}
    cursors = {}
    for bank in banks:
        known = dict(zip(bank.entity_keys, bank.entity_global_ids, strict=True))
        ids[bank.degree] = np.asarray(
            [known[key] for key in descriptor.entity_keys(bank.degree)], dtype=np.int64
        )
        cursors[bank.degree] = bank.allocator_next_id
    plain = CellMesh(
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
    staged = PeriodicMeshTopology(
        plain,
        descriptor.cell,
        descriptor.vertex_representatives,
        descriptor.vertex_shifts,
        actual_geometry=descriptor.actual_geometry,
        entity_global_ids=ids,
        entity_allocator_next_ids=cursors,
    )
    return CellMesh(
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
    source: PeriodicMeshTopology,
    target: CellMesh,
    nonnested_geometry: PeriodicNonnestedGeometryAuthority | None = None,
    /,
) -> tuple[CellMesh, CellGeometrySpec | None]:
    """Retain the producer's real successor source before stripping its descriptor."""
    from ..discretization._exact_power_geometry import (
        ExactPowerCellGeometryLinearActionSource,
        ExactPowerCellGeometryRestrictionSource,
        ExactPowerCellGeometrySource,
    )

    descriptor = target.periodic_topology
    geometry = descriptor.actual_geometry if descriptor is not None else None
    if nonnested_geometry is not None:
        from ._topology_edit import require_periodic_nonnested_geometry

        require_periodic_nonnested_geometry(source, target, nonnested_geometry)
    if source.actual_geometry is not None and nonnested_geometry is None:
        if geometry is None:
            raise ValueError(
                "An exact periodic edit requires its actual target geometry source authority."
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
                "Exact periodic edits require canonical vertex source owners."
            )
        source.actual_geometry.source_coordinates()
        visited: set[int] = set()
        while current.source_id != original.source_id:
            if id(current) in visited:
                raise ValueError(
                    "Exact periodic successor source ancestry contains a cycle."
                )
            visited.add(id(current))
            if isinstance(current, ExactPowerCellGeometrySource):
                raise ValueError(
                    "Exact periodic successor does not retain its actual predecessor source."
                )
            parent = current.parent
            if not isinstance(parent, source_types):
                raise TypeError(
                    "Exact periodic successor ancestry omits a canonical source owner."
                )
            current = parent
    if descriptor is not None:
        from ..discretization._cell_complex import PolyhedralConnectivity

        target = CellMesh(
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
    source: CellMesh,
    target: CellMesh,
    witness: PeriodicVertexOrbitWitness,
    allocation_base: tuple[PeriodicEntityIdentityBank, ...] | None,
    /,
) -> CellMesh:
    """Bind one target after its owning caller validates the prior allocation chain."""
    from ..discretization._periodic_topology import _identification_id

    if witness.nonnested_geometry is not None:
        from ._topology_edit import require_periodic_nonnested_source

        require_periodic_nonnested_source(source, witness.nonnested_geometry)
    topology = _require_periodic_topology(source)
    if (
        witness.source_topology_id != source.topology_id
        or witness.identification_id != _identification_id(topology.cell)
    ):
        raise ValueError(
            "Periodic orbit witness does not bind source topology/identification."
        )
    identifiers = np.asarray(target.vertex_global_ids)
    representative_ids = np.asarray(witness.vertex_representative_ids)
    shifts = np.asarray(witness.vertex_shifts)
    if representative_ids.shape != identifiers.shape or not np.issubdtype(
        representative_ids.dtype, np.integer
    ):
        raise ValueError(
            "Periodic representatives must be global integer IDs per target vertex row."
        )
    if shifts.shape != (identifiers.size, topology.cell.rank) or not np.issubdtype(
        shifts.dtype, np.integer
    ):
        raise ValueError(
            "Periodic image exponents must be integer target-row lattice shifts."
        )
    lookup = {int(identifier): row for row, identifier in enumerate(identifiers)}
    try:
        rows = np.asarray(
            [lookup[int(identifier)] for identifier in representative_ids], dtype=np.int64
        )
    except KeyError as error:
        raise ValueError(
            "A periodic representative is not a global target vertex."
        ) from error
    original_ids = np.asarray(source.vertex_global_ids)
    old_roots = original_ids[np.asarray(topology.vertex_representatives)]
    original_lookup = dict(zip(original_ids.tolist(), old_roots.tolist(), strict=True))
    for vertex, identifier in enumerate(identifiers):
        if (
            int(identifier) in original_lookup
            and representative_ids[vertex] != original_lookup[int(identifier)]
        ):
            raise ValueError(
                "A periodic edit changes an old vertex's scientific quotient membership."
            )
    old_representatives = set(old_roots.tolist())
    for representative in np.unique(representative_ids):
        members = identifiers[representative_ids == representative]
        if representative not in old_representatives and representative != np.min(
            members
        ):
            raise ValueError(
                "A born quotient vertex must use its canonical global representative."
            )
    target, actual_geometry = _periodic_target_lift_authority(
        topology, target, witness.nonnested_geometry
    )
    probe = PeriodicMeshTopology(
        target, topology.cell, rows, shifts, actual_geometry=actual_geometry
    )
    if witness.allocation_prior is None:
        current = _periodic_identity_banks(source, witness.history)
    else:
        if (
            allocation_base is None
            or witness.retained_quotient_entities != allocation_base
        ):
            raise ValueError(
                "Retained periodic identities must equal the validated prior complete allocation bank."
            )
        current = allocation_base
    expected = _allocate_periodic_identity_banks(
        probe, current, witness.retained_quotient_entities
    )
    if tuple(bank.degree for bank in witness.quotient_entities) != tuple(
        bank.degree for bank in expected
    ):
        raise ValueError("A periodic witness omits quotient scientific identity banks.")
    entity_ids, cursors = {}, {}
    for actual, bank in zip(witness.quotient_entities, expected, strict=True):
        _require_periodic_identity_bank(actual)
        if (
            actual.allocator_next_id != bank.allocator_next_id
            or actual.entity_keys != bank.entity_keys
            or actual.entity_global_ids != bank.entity_global_ids
        ):
            raise ValueError(
                "Periodic live/restored/retired identities or allocator high-water disagree."
            )
        known = dict(zip(bank.entity_keys, bank.entity_global_ids, strict=True))
        entity_ids[bank.degree] = np.asarray(
            [known[key] for key in probe.entity_keys(bank.degree)], dtype=np.int64
        )
        cursors[bank.degree] = bank.allocator_next_id
    periodic = PeriodicMeshTopology(
        target,
        topology.cell,
        rows,
        shifts,
        entity_global_ids=entity_ids,
        entity_allocator_next_ids=cursors,
        actual_geometry=actual_geometry,
    )
    from ..discretization._cell_complex import PolyhedralConnectivity

    return CellMesh(
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
    source: CellMesh,
    target: CellMesh,
    representative_ids: np.ndarray,
    exponents: np.ndarray,
    history: PeriodicRefinement | None,
    /,
    *,
    retained_quotient_entities: tuple[PeriodicEntityIdentityBank, ...] = (),
    allocation_prior: tuple[CellMesh, PeriodicVertexOrbitWitness] | None = None,
    nonnested_geometry: PeriodicNonnestedGeometryAuthority | None = None,
) -> PeriodicVertexOrbitWitness:
    """Witness a producer's explicit target orbits with keyed scientific identity banks.

    ``representative_ids`` names the global representative vertex of every
    target vertex row of the unbound ``target`` carrier; ``exponents`` holds its
    integer group exponents relative to that representative.
    Advanced retained banks require ``allocation_prior``: the real bound prior
    target and its canonical witness, recursively validated against ``source``.
    """

    from ..discretization._periodic_topology import _identification_id
    from ._topology_edit import PeriodicVertexOrbitWitness

    if nonnested_geometry is not None:
        from ._topology_edit import require_periodic_nonnested_source

        require_periodic_nonnested_source(source, nonnested_geometry)
    topology = _require_periodic_topology(source)
    target, actual_geometry = _periodic_target_lift_authority(
        topology, target, nonnested_geometry
    )
    ids = np.asarray(target.vertex_global_ids)
    representatives = np.asarray(representative_ids)
    shifts = np.asarray(exponents)
    if representatives.shape != ids.shape or not np.issubdtype(
        representatives.dtype, np.integer
    ):
        raise ValueError(
            "Periodic representatives must be global integer IDs per target vertex row."
        )
    if shifts.shape != (ids.size, topology.cell.rank) or not np.issubdtype(
        shifts.dtype, np.integer
    ):
        raise ValueError(
            "Periodic image exponents must be integer target-row group exponents."
        )
    indices = {int(identifier): row for row, identifier in enumerate(ids)}
    try:
        local_roots = np.asarray(
            [indices[int(identifier)] for identifier in representatives], dtype=np.int64
        )
    except KeyError as error:
        raise ValueError(
            "A periodic representative is not a global target vertex."
        ) from error
    probe = PeriodicMeshTopology(
        target,
        topology.cell,
        local_roots,
        shifts,
        actual_geometry=actual_geometry,
    )
    banks = _periodic_target_identity_banks(
        source, probe, history, retained_quotient_entities, allocation_prior
    )
    return PeriodicVertexOrbitWitness(
        source.topology_id,
        _identification_id(topology.cell),
        representatives.astype(np.int64),
        shifts.astype(np.int64),
        banks,
        history,
        retained_quotient_entities,
        allocation_prior,
        nonnested_geometry,
    )


def _periodic_orbit_witness(
    source: CellMesh,
    geometric: CellMesh,
    target: CellMesh,
    history: PeriodicRefinement | None,
    source_rows: np.ndarray,
) -> PeriodicVertexOrbitWitness:
    """Lower a real construction's orbit map onto global target identities."""

    from ..discretization._periodic_topology import _identification_orders, _reduced

    topology = _require_periodic_topology(source)
    geometry = _require_periodic_topology(geometric)
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
        exponents[members] = _reduced(
            shifts[members] - shifts[root], _identification_orders(topology.cell)
        )
    return periodic_vertex_orbit_witness(
        source, target, representatives, exponents, history
    )


class PeriodicTopologyEditStage(NamedTuple):
    """Immutable complete candidate; accepted state is never mutated by staging."""

    mesh: CellMesh
    lineage: MeshLineage
    stencil: VertexInterpolationStencil | None
    embedding: PeriodicEmbeddingEvidence
    quotient: PeriodicQuotientEvidence


def _require_complete_periodic_entity_changes(
    source: CellMesh, target: CellMesh, edit: CellTopologyEdit, /
) -> None:
    from ._topology_edit import entity_keys

    periodic = _require_periodic_topology(source)
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
                    "A periodic transaction changes only part of a scientific entity orbit."
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
    successor = _require_periodic_topology(target)
    after = np.asarray(successor.quotient.entities(degree).subset("boundary").mask)
    target_boundary = after[np.asarray(successor.orbits(degree)[0])]
    if any(
        tuple(key) not in allowed for key in entity_keys(target, degree)[target_boundary]
    ):
        raise ValueError(
            "A periodic transaction creates an unbound physical boundary or partial seam."
        )
    if np.any(before):
        from .providers._native_periodic import _feature_entity_mask

        source_corners = _simplex_entity_corners(source, degree)
        target_corners = _simplex_entity_corners(target, degree)
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
            matched = _feature_entity_mask(
                source, target, np.asarray([group], dtype=np.int64), degree
            )
            covered |= matched
            measured = float(
                np.sum(target_measures[new_leaders[matched[new_leaders] & after]])
            )
            expected = float(source_measures[old_leaders[group]])
            if abs(measured - expected) > 1.0e-10 * max(1.0, expected):
                raise ValueError(
                    "Periodic physical-boundary coverage changed in the staged transaction."
                )
        if np.any(target_boundary & ~covered):
            raise ValueError(
                "Periodic physical boundary moved away from its represented source."
            )


def stage_periodic_topology_edit(
    source: CellMeshingResult,
    edit: CellTopologyEdit,
    /,
    *,
    numeric_version: str,
    maximum_images: int,
    maximum_pairs: int,
) -> PeriodicTopologyEditStage:
    """Certify a complete private orbit candidate before any accepted/native commit."""

    from ._adaptation import _require_affine_geometry
    from ._organization import MeshZoneRole
    from ._scope import resolve_mesh_scope
    from ._topology_edit import assemble_topology_edit

    _require_affine_geometry(source)
    if edit.periodic_orbits is None:
        raise ValueError(
            "A periodic candidate omits its complete scientific orbit witness."
        )
    target, lineage, stencil = assemble_topology_edit(
        source.mesh, edit, numeric_version=numeric_version
    )
    _require_complete_periodic_entity_changes(source.mesh, target, edit)
    embedding = certify_periodic_embedding(
        target, maximum_images=maximum_images, maximum_pairs=maximum_pairs
    )
    old = periodic_orbit_measures(source.mesh)
    new = periodic_orbit_measures(target)
    if abs(float(old.total_measure) - float(new.total_measure)) > 1.0e-10 * float(
        old.total_measure
    ):
        raise ValueError(
            "A periodic staged transaction changes one-orbit domain coverage."
        )
    source_ids = np.asarray(
        source.mesh.entity_set(source.mesh.topological_dimension).entity_ids
    )
    target_ids = np.asarray(target.entity_set(target.topological_dimension).entity_ids)
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
                        "Source periodic material regions are not exclusive."
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
                    "A periodic candidate has unresolved or contradictory source material ancestry."
                )
            target_assignment.append(next(iter(materials)))
        dimension = source.mesh.topological_dimension
        source_orbits = np.asarray(
            _require_periodic_topology(source.mesh).orbits(dimension)[0], dtype=np.int64
        )
        target_orbits = np.asarray(
            _require_periodic_topology(target).orbits(dimension)[0], dtype=np.int64
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
                    np.asarray(old.orbit_measures)[np.unique(source_orbits[source_mask])]
                )
            )
            after = float(
                np.sum(
                    np.asarray(new.orbit_measures)[np.unique(target_orbits[target_mask])]
                )
            )
            if abs(before - after) > 1.0e-10 * max(1.0, before):
                raise ValueError(
                    "A periodic staged transaction changes source material coverage."
                )
    return PeriodicTopologyEditStage(
        target, lineage, stencil, embedding, PeriodicQuotientEvidence(target)
    )


def validate_periodic_bisection(prepared: PreparedMeshAdaptation, /) -> None:
    """Admit exact affine orbit refinement and retained-epoch coarsening."""

    from ._adaptation import (
        _check_marked_source,
        _require_affine_geometry,
        MarkedMeshAdaptation,
    )

    request, source = prepared.request, prepared.source.mesh
    if not isinstance(request, MarkedMeshAdaptation) or source.periodic_topology is None:
        raise TypeError("Periodic bisection requires a marked periodic source.")
    _require_affine_geometry(prepared.source)
    _check_marked_source(prepared.source, request)
    if any(
        block.cell_kind != _SIMPLEX_KINDS.get(source.topological_dimension)
        for block in source.blocks
    ):
        raise ValueError("Periodic orbit bisection requires a complete simplex family.")
    if request.hierarchy is not None:
        history = request.hierarchy
        if not isinstance(history, PeriodicRefinement):
            raise TypeError("Periodic bisection history must be a PeriodicRefinement.")
        if history.mesh.topology_id != source.topology_id or not np.array_equal(
            history.mesh.coordinates, source.coordinates
        ):
            raise ValueError(
                "Periodic coarsening requires its current retained orbit-refinement history."
            )
    if np.asarray(request.coarsen_cell_ids).size and request.hierarchy is None:
        raise ValueError("Periodic coarsening needs the retained source-epoch history.")


def _execute_periodic_bisection_route(
    prepared: PreparedMeshAdaptation, /
) -> _RouteOutcome:
    from ._adaptation import (
        _finalize_native,
        _RouteOutcome,
        _unchanged,
        MarkedMeshAdaptation,
        MeshAdaptationStatus,
    )
    from ._bisection import BisectionEvidence
    from ._lineage import MeshTransitionKind
    from ._topology_edit import assemble_topology_edit

    validate_periodic_bisection(prepared)
    source, request, limits = (
        prepared.source.mesh,
        prepared.request,
        prepared.policy.limits,
    )
    if not isinstance(request, MarkedMeshAdaptation):
        raise TypeError("Periodic bisection requires a marked request.")
    source_cells, source_ids = _periodic_simplex_rows(source)
    refine_ids = np.asarray(request.refine_cell_ids)
    coarsen_ids = np.asarray(request.coarsen_cell_ids)
    history = request.hierarchy
    if history is not None and not isinstance(history, PeriodicRefinement):
        raise TypeError("Periodic bisection history must be a PeriodicRefinement.")
    restored_parents = np.empty(0, dtype=np.int64)
    if coarsen_ids.size:
        if history is None:
            raise ValueError("Periodic coarsening has no retained source epoch.")
        restored_parents = _periodic_coarsening_parents(prepared, history)
    coarsening = restored_parents.size > 0
    accepted_coarse = (
        source_ids[np.isin(history.parent_cells, restored_parents)]
        if history is not None
        else np.empty(0, dtype=np.int64)
    )
    rejected_coarsening = coarsen_ids[~np.isin(coarsen_ids, accepted_coarse)]
    protected = set(
        np.asarray(_require_periodic_topology(source).orbits(1)[0])[
            prepared.constraints.protected_edge_mask
        ].tolist()
    )
    edges = _simplex_entity_corners(source, 1)
    orbit = np.asarray(_require_periodic_topology(source).orbits(1)[0])
    lengths = np.linalg.norm(
        np.asarray(source.coordinates)[edges[:, 1]]
        - np.asarray(source.coordinates)[edges[:, 0]],
        axis=1,
    )
    by_pair = {tuple(sorted(edge)): row for row, edge in enumerate(edges.tolist())}
    quotient_keys = _require_periodic_topology(source).entity_keys(1)
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
    refinement = None
    if selected and coarsening:
        rejected_coarsening = coarsen_ids
        coarsening = False
    if selected:
        upper_cells = source_ids.size * (4 if source.topological_dimension == 2 else 64)
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
                "Periodic orbit closure exceeds its declared construction budget."
            )
        refinement = refine_periodic_mesh(
            source,
            edge_orbits=np.asarray(sorted(selected), dtype=np.int64),
            source_geometry=prepared.source.geometry,
        )
        edit = _periodic_refinement_edit(refinement, history)
        kind = MeshTransitionKind.REFINE
    elif coarsening and history is not None:
        edit, coarse_parents, coarse_reference, coarse_generations = (
            _periodic_coarsening_edit(history, restored_parents)
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
    if edit is None:
        return _unchanged(
            prepared,
            evidence,
            history,
            status=MeshAdaptationStatus.PARTIAL
            if partial
            else MeshAdaptationStatus.UNCHANGED,
        )
    probe, _, _ = assemble_topology_edit(
        source, edit, numeric_version=f"adaptation:{prepared.prepared_id}"
    )
    certify_periodic_embedding(
        probe,
        maximum_images=limits.maximum_scratch_bytes // max(1, probe.coordinates.size * 8),
        maximum_pairs=limits.maximum_work_units,
    )
    old_measure, new_measure = (
        periodic_orbit_measures(source).total_measure,
        periodic_orbit_measures(probe).total_measure,
    )
    if abs(float(old_measure) - float(new_measure)) > 1.0e-10 * float(old_measure):
        raise ValueError("Periodic bisection changed the quotient's one-orbit measure.")
    native = _finalize_native(prepared, edit, kind, conservative=not coarsening)
    next_history = None
    edit_ids = np.concatenate(tuple(np.asarray(block.cell_ids) for block in edit.blocks))
    target_ids = np.concatenate(
        tuple(np.asarray(block.global_ids) for block in native.target.mesh.blocks)
    )
    rows_by_id = {int(identifier): row for row, identifier in enumerate(edit_ids)}
    if (
        len(rows_by_id) != edit_ids.size
        or len(set(target_ids.tolist())) != target_ids.size
        or set(rows_by_id) != set(target_ids.tolist())
    ):
        raise ValueError(
            "Periodic publication changed the actual retained child scientific cell bank."
        )
    order = np.fromiter(
        (rows_by_id[int(identifier)] for identifier in target_ids),
        dtype=np.int64,
        count=target_ids.size,
    )
    if coarsening and history is not None:
        source_keys = set(_require_periodic_topology(history.source).entity_keys(1))
        present_keys = set(_require_periodic_topology(native.target.mesh).entity_keys(1))
        next_history = PeriodicRefinement(
            mesh=native.target.mesh,
            source=history.source,
            source_geometry=history.source_geometry,
            parent_cells=coarse_parents[order],
            parent_reference_vertices=coarse_reference[order],
            cell_generations=coarse_generations[order],
            bisected_edges=len(source_keys - present_keys),
            quotient=PeriodicQuotientEvidence(native.target.mesh),
            previous=history.previous,
            retired=history,
        )
    if refinement is not None:
        next_history = PeriodicRefinement(
            mesh=native.target.mesh,
            source=source,
            source_geometry=prepared.source.geometry,
            parent_cells=np.asarray(refinement.parent_cells)[order],
            parent_reference_vertices=np.asarray(refinement.parent_reference_vertices)[
                order
            ],
            cell_generations=generations[order],
            bisected_edges=refinement.bisected_edges,
            quotient=PeriodicQuotientEvidence(native.target.mesh),
            previous=history,
        )
    return _RouteOutcome(
        MeshAdaptationStatus.PARTIAL if partial else MeshAdaptationStatus.COMPLETE,
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
    adaptation: MeshAdaptationResult, /
) -> NestedReferenceWitnesses | None:
    """Authoritative fine-to-coarse reference charts of a periodic coarsening.

    Retained source cells use the reference identity. Restored sibling families
    use their recorded dyadic construction charts. This joins scientific cell
    IDs only; it never intersects or nearest-matches independently lifted ghosts.
    """

    from ..discretization._cell_geometry_transfer import NestedReferenceWitnesses
    from ._lineage import MeshTransitionKind

    transition = adaptation.transition
    history = adaptation.hierarchy
    if transition is None or transition.transition_kind is not MeshTransitionKind.COARSEN:
        return None
    if not isinstance(history, PeriodicRefinement) or history.retired is None:
        return None
    retired = history.retired
    fine, target = retired.mesh, adaptation.target.mesh
    if fine.topology_id != transition.source_topology_id:
        raise ValueError("Periodic coarsening witnesses are stale for the source epoch.")
    fine_ids = np.asarray(fine.blocks[0].global_ids, dtype=np.int64)
    original_ids = np.asarray(retired.source.blocks[0].global_ids, dtype=np.int64)[
        np.asarray(retired.parent_cells, dtype=np.int64)
    ]
    target_ids = np.asarray(target.blocks[0].global_ids, dtype=np.int64)
    restored = np.isin(original_ids, target_ids)
    coarse_ids = np.where(restored, original_ids, fine_ids)
    if not np.all(np.isin(coarse_ids, target_ids)) or set(coarse_ids.tolist()) != set(
        target_ids.tolist()
    ):
        raise ValueError(
            "Periodic coarsening reference charts do not cover the target cells."
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
    "periodic_coarsening_witnesses",
]


# Source-aware metric edits select authoritative lifted carriers. Concrete
# Surface/Sphere owners construct every local operation on one private state.
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import assert_never, Literal, TypeAlias

from .._meshcore import current_native_execution_budget, NativeExecutionBudget
from ..discretization._cell_geometry_transfer import CellGeometryTransition
from ..discretization._cell_geometry_validity import cell_geometry_id
from ..discretization._coordinate_enclosure import CoordinateEnclosureBudget
from ..discretization._sphere_chart_deformation import PreparedSphereChartDeformation
from ..discretization._surface_chart_deformation import PreparedSurfaceChartDeformation
from ..typing import parse
from ._association import GeometryAssociation
from ._contracts import MeshingLimits
from ._result import CellMeshingResult
from ._topology_edit import (
    assemble_topology_edit,
    CellTopologyEdit,
    entity_keys,
    EntityRelations,
    PeriodicEntityIdentityBank,
    PeriodicVertexOrbitWitness,
    TopologyEditBlock,
)


if TYPE_CHECKING:
    from ._adaptation import MeshAdaptationPolicy


PeriodicMetricOperation: TypeAlias = Literal["split", "collapse", "flip", "relocate"]


class _MetricControllerWork:
    """Attempted controller entity visits, separate from owning primitive work."""

    def __init__(
        self, ledger: CoordinateEnclosureBudget, execution: NativeExecutionBudget
    ) -> None:
        self.ledger = ledger
        self.execution = execution
        self.visits = 0
        self.charged = 0
        self.remaining_work = execution.remaining().remaining_work_units

    def visit(self, count: int = 1) -> None:
        if self.visits - self.charged + count > self.remaining_work:
            raise ValueError(
                "Periodic controller visits exceed the actual original native work remainder."
            )
        self.ledger.reserve(count)
        self.visits += count

    def flush(self) -> None:
        if current_native_execution_budget() is not self.execution:
            raise RuntimeError(
                "The periodic controller cannot replace its original native execution scope."
            )
        self.ledger.charge_native_work(self.visits - self.charged)
        self.charged = self.visits
        self.remaining_work = self.execution.remaining().remaining_work_units

    @contextmanager
    def activate(self) -> Iterator[None]:
        token = _METRIC_CONTROLLER_WORK.set(self)
        try:
            yield
        finally:
            _METRIC_CONTROLLER_WORK.reset(token)
            self.flush()


_METRIC_CONTROLLER_WORK: ContextVar[_MetricControllerWork | None] = ContextVar(
    "periodic_metric_controller_work", default=None
)


def _metric_visit(count: int = 1, /) -> None:
    work = _METRIC_CONTROLLER_WORK.get()
    if work is None:
        raise RuntimeError(
            "Periodic metric controller visits require the original active ledger."
        )
    work.visit(count)


class PeriodicMetricOrbitCarrier(NamedTuple):
    """One actual source incidence, with exact action from the original carrier."""

    entity_global_id: int
    vertex_global_ids: tuple[int, ...]
    incident_cell_global_ids: tuple[int, ...]
    incident_cell_classes: tuple[int, ...]
    group_shift: tuple[int, ...]
    vertex_permutation: tuple[int, ...]
    protected_vertex_global_ids: tuple[int, ...]
    protected_entity: bool


class PeriodicMetricOrbitSelection(NamedTuple):
    """Complete winding-sensitive operation support in the immutable source frame."""

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
    """Actual aggregate edit returned after every required owning operation."""

    edit: CellTopologyEdit
    exact_stencils: Mapping[int, ConstructionPointKey]
    executed_entity_ids: tuple[int, ...]


class PeriodicMetricGeometryStage(NamedTuple):
    """Actual full-map successor and its complete source-owned correspondence."""

    target_mesh: CellMesh
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
    target_mesh: CellMesh
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


def _metric_orbit_cells(mesh: CellMesh, /) -> tuple[_MetricOrbitCell, ...]:
    ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    cells = []
    for block in mesh.blocks:
        for identifier, vertices in zip(
            np.asarray(block.global_ids), np.asarray(block.vertices), strict=True
        ):
            _metric_visit()
            cells.append(
                _MetricOrbitCell(
                    block.name,
                    block.cell_kind,
                    int(identifier),
                    tuple(int(value) for value in ids[vertices]),
                )
            )
    return tuple(cells)


def _metric_candidate_cells(edit: CellTopologyEdit, /) -> tuple[_MetricOrbitCell, ...]:
    cells = []
    for block in edit.blocks:
        if not isinstance(block, TopologyEditBlock):
            raise ValueError(
                "A local metric rewrite requires its owning fixed-family candidate."
            )
        for identifier, vertices in zip(block.cell_ids, block.cells, strict=True):
            _metric_visit()
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
    source: CellMesh,
    edit: CellTopologyEdit,
    stencils: Mapping[int, ConstructionPointKey],
    /,
) -> dict[int, ConstructionPointKey]:
    source_ids = set(np.asarray(source.vertex_global_ids).tolist())
    ids = np.asarray(edit.vertex_global_ids)
    if (
        ids.ndim != 1
        or not np.issubdtype(ids.dtype, np.integer)
        or len(set(ids.tolist())) != ids.size
    ):
        raise ValueError(
            "A metric candidate requires unique integer scientific vertex IDs."
        )
    if set(stencils) != set(ids.tolist()):
        raise ValueError(
            "Exact metric stencils must cover exactly the private target vertices."
        )
    if (
        edit.stencil_sources.shape != edit.stencil_weights.shape
        or edit.stencil_valid.shape != edit.stencil_sources.shape
        or edit.stencil_sources.shape[0] != ids.size
    ):
        raise ValueError(
            "Metric candidate interpolation rows do not align with target IDs."
        )
    result = {}
    for row, identifier in enumerate(ids):
        _metric_visit()
        key = stencils[int(identifier)]
        if not isinstance(key, tuple) or not key:
            raise ValueError(
                "Exact construction stencils need canonical positive source supports."
            )
        for vertex, weight in key:
            _metric_visit()
            if (
                not isinstance(weight, Fraction)
                or weight <= 0
                or vertex not in source_ids
            ):
                raise ValueError(
                    "Exact construction stencils need canonical positive source supports."
                )
        if tuple(sorted(key)) != key or len({vertex for vertex, _ in key}) != len(key):
            raise ValueError(
                "Exact construction stencils need canonical positive source supports."
            )
        if sum((weight for _, weight in key), Fraction(0)) != 1:
            raise ValueError(
                "An exact metric construction stencil must preserve constants."
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
                "Exact construction support disagrees with the actual candidate stencil."
            )
        result[int(identifier)] = key
    return result


def _metric_orbit_cavity(
    source: CellMesh, edit: CellTopologyEdit, /
) -> tuple[tuple[_MetricOrbitCell, ...], tuple[_MetricOrbitCell, ...], set[int]]:
    before = _metric_orbit_cells(source)
    after = _metric_candidate_cells(edit)
    targets = {cell.identifier: cell for cell in after}
    old_ids = np.asarray(source.vertex_global_ids)
    new_rows = {
        int(identifier): row for row, identifier in enumerate(edit.vertex_global_ids)
    }
    old_points = np.asarray(source.coordinates)
    moved = set()
    for row, identifier in enumerate(old_ids):
        _metric_visit()
        if int(identifier) not in new_rows or not np.array_equal(
            old_points[row], edit.coordinates[new_rows[int(identifier)]]
        ):
            moved.add(int(identifier))
    selected = []
    for cell in before:
        _metric_visit()
        if (
            cell.identifier not in targets
            or targets[cell.identifier] != cell
            or moved.intersection(cell.vertices)
        ):
            selected.append(cell)
    cavity = tuple(selected)
    if not cavity:
        raise ValueError("A periodic metric candidate must contain a real local edit.")
    originals = {cell.identifier: cell for cell in before}
    replacements = []
    for cell in after:
        _metric_visit()
        if (
            cell.identifier not in originals
            or originals[cell.identifier] != cell
            or moved.intersection(cell.vertices)
        ):
            replacements.append(cell)
    if not replacements:
        raise ValueError("A local metric edit has no complete target cavity.")
    support = {vertex for cell in cavity for vertex in cell.vertices}
    return cavity, tuple(replacements), support


def _metric_protected_rows(
    source: CellMeshingResult, policy: MeshAdaptationPolicy, /
) -> tuple[set[int], ...]:
    from ._adaptation import (
        _closure_rows,
        _membership,
        _organization_scopes,
        _selected_rows,
        MeshAdaptationRoute,
    )
    from ._association import SurfaceAssociationTransfer

    mesh = source.mesh
    periodic = _require_periodic_topology(mesh)
    protected: tuple[set[int], ...] = tuple(
        set() for _ in range(mesh.topological_dimension + 1)
    )
    for scope in policy.protected_scopes:
        _metric_visit()
        selected = _selected_rows(mesh, scope)
        for degree in range(scope.entity_dimension + 1):
            rows = _closure_rows(mesh, scope.entity_dimension, selected, degree)
            orbit = np.asarray(periodic.orbits(degree)[0])
            _metric_visit(orbit.size)
            protected[degree].update(np.flatnonzero(np.isin(orbit, orbit[rows])).tolist())
    # A scientific corner or ambiguous membership is fixed under every copy,
    # not just the lifted row on which the source happened to publish it.
    for association in source.associations:
        _metric_visit()
        if association.target_entity_set_id != mesh.entity_set(0).entity_set_id:
            continue
        rows = association.target_rows(np.asarray(mesh.vertex_global_ids))
        fixed = np.asarray(association.ambiguous)[rows] | (
            np.asarray(association.source_dimensions)[rows] == 0
        )
        orbit = np.asarray(periodic.orbits(0)[0])
        _metric_visit(orbit.size)
        protected[0].update(np.flatnonzero(np.isin(orbit, orbit[fixed])).tolist())
    fixed = np.any(_membership(mesh, _organization_scopes(source), 0), axis=1)
    orbit = np.asarray(periodic.orbits(0)[0])
    _metric_visit(orbit.size)
    protected[0].update(np.flatnonzero(np.isin(orbit, orbit[fixed])).tolist())
    transfer = policy.association_transfer
    if isinstance(transfer, SurfaceAssociationTransfer):
        classes = transfer.classes(source)
        fixed = classes[0].dimensions == 0
        _metric_visit(orbit.size)
        protected[0].update(np.flatnonzero(np.isin(orbit, orbit[fixed])).tolist())
        edge_mask = (
            transfer.protected_edges(
                source,
                midpoint_required=policy.route
                in (
                    MeshAdaptationRoute.NATIVE_BISECTION,
                    MeshAdaptationRoute.DEVICE_BISECTION,
                    MeshAdaptationRoute.NATIVE_MIXED,
                ),
            )
            | ~classes[1].resolved
        )
        edge_orbit = np.asarray(periodic.orbits(1)[0])
        _metric_visit(edge_orbit.size)
        protected[1].update(
            np.flatnonzero(np.isin(edge_orbit, edge_orbit[edge_mask])).tolist()
        )
    return protected


def _metric_original_support(
    mesh: CellMesh, candidate: CellTopologyEdit, operation: PeriodicMetricOperation, /
) -> tuple[int, int, tuple[int, ...]]:
    from ._topology_edit import _build_mesh

    old_ids = set(np.asarray(mesh.vertex_global_ids).tolist())
    new_ids = set(candidate.vertex_global_ids.tolist())
    if operation == "relocate":
        rows = {
            int(identifier): row
            for row, identifier in enumerate(candidate.vertex_global_ids)
        }
        changed = []
        for row, identifier in enumerate(mesh.vertex_global_ids):
            _metric_visit()
            if int(identifier) in rows and not np.array_equal(
                np.asarray(mesh.coordinates)[row],
                candidate.coordinates[rows[int(identifier)]],
            ):
                changed.append(int(identifier))
        moved = tuple(changed)
        if len(moved) != 1 or old_ids != new_ids:
            raise ValueError(
                "A periodic relocation seed must name one actual retained source vertex."
            )
        return 0, moved[0], moved
    if operation == "collapse":
        removed = old_ids - new_ids
        if len(removed) != 1 or new_ids - old_ids:
            raise ValueError(
                "A periodic collapse seed must remove exactly one source vertex."
            )
        vertex = next(iter(removed))
        before = _metric_orbit_cells(mesh)
        after = _metric_candidate_cells(candidate)
        target_keys = {
            (cell.name, cell.kind, tuple(sorted(cell.vertices))) for cell in after
        }
        neighbors = {
            value for cell in before if vertex in cell.vertices for value in cell.vertices
        }
        successors = set()
        for kept in sorted((neighbors - {vertex}) & new_ids):
            substituted = set()
            for cell in before:
                _metric_visit()
                values = tuple(
                    kept if value == vertex else value for value in cell.vertices
                )
                if len(set(values)) == len(values):
                    substituted.add((cell.name, cell.kind, tuple(sorted(values))))
            if substituted == target_keys:
                successors.add(kept)
        if len(successors) != 1:
            raise ValueError(
                "A collapse seed has no unique actual source edge contraction witness."
            )
        support = (vertex, next(iter(successors)))
    else:
        probe = _build_mesh(candidate, mesh.numeric_version, None)
        target_keys = set(map(tuple, entity_keys(probe, 1)))
        removed = []
        for key in entity_keys(mesh, 1):
            _metric_visit()
            if tuple(key) not in target_keys:
                removed.append(tuple(int(value) for value in key))
        removed_edges = tuple(removed)
        if len(removed_edges) != 1:
            raise ValueError(
                "A split or flip seed must remove one authoritative source edge."
            )
        if operation == "split" and (len(new_ids - old_ids) != 1 or old_ids - new_ids):
            raise ValueError("A split seed must construct one actual born vertex.")
        if operation == "flip" and old_ids != new_ids:
            raise ValueError("A flip seed cannot construct or remove source vertices.")
        support = removed_edges[0]
    keys = entity_keys(mesh, 1)
    rows = [row for row, key in enumerate(keys) if tuple(key) == tuple(sorted(support))]
    if len(rows) != 1:
        raise ValueError("The operation support is not one actual lifted source edge.")
    identifier = int(np.asarray(mesh.entity_set(1).entity_ids)[rows[0]])
    return 1, identifier, support


def _metric_source_classes(
    source: CellMeshingResult, policy: MeshAdaptationPolicy, /
) -> dict[int, int]:
    from ._adaptation import _cell_classes, _cell_ids, _organization_scopes, _row_classes
    from ._association import SurfaceAssociationTransfer
    from ._topology_edit import key_rows

    mesh = source.mesh
    classes = _cell_classes(mesh, _organization_scopes(source), family_identity=True)
    _metric_visit(classes.size)
    if source.attributes:
        from ._layer_core import layer_interval_classes

        rows = key_rows(
            _cell_ids(mesh)[:, None], entity_keys(mesh, mesh.topological_dimension)
        )
        classes = _row_classes(
            np.column_stack((classes, layer_interval_classes(source)[rows]))
        )
        _metric_visit(classes.size)
    if isinstance(policy.association_transfer, SurfaceAssociationTransfer):
        source_classes = policy.association_transfer.classes(source)
        classes = _row_classes(
            np.column_stack((classes, source_classes[mesh.topological_dimension].codes))
        )
        _metric_visit(classes.size)
    return dict(
        zip(
            entity_keys(mesh, mesh.topological_dimension)[:, 0].tolist(),
            classes.tolist(),
            strict=True,
        )
    )


def _metric_current_classes(
    source: CellMeshingResult,
    mesh: CellMesh,
    prior: PeriodicMetricOrbitOutcome | None,
    policy: MeshAdaptationPolicy,
    /,
) -> dict[int, int]:
    originals = _metric_source_classes(source, policy)
    if prior is None:
        return originals
    inherited: dict[int, set[int]] = {}
    relations = prior.edit.relations[source.mesh.topological_dimension]
    for old, new in zip(relations.source_keys, relations.target_keys, strict=True):
        _metric_visit()
        if int(old[0]) not in originals:
            raise ValueError(
                "The private prior stage carries stale original cell ancestry."
            )
        inherited.setdefault(int(new[0]), set()).add(originals[int(old[0])])
    result = {}
    for identifier in entity_keys(mesh, mesh.topological_dimension)[:, 0]:
        _metric_visit()
        value = int(identifier)
        values = inherited.get(value, {originals[value]} if value in originals else set())
        if len(values) != 1:
            raise ValueError(
                "The private current carrier has incomplete or incompatible original cell classes."
            )
        result[value] = next(iter(values))
    return result


def _metric_orbit_selection(
    source: CellMeshingResult,
    candidate: CellTopologyEdit,
    operation: PeriodicMetricOperation,
    policy: MeshAdaptationPolicy,
    orbits: PeriodicConstructionOrbits,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    prior: PeriodicMetricOrbitOutcome | None,
    /,
) -> PeriodicMetricOrbitSelection:
    periodic = _require_periodic_topology(mesh)
    degree, identifier, original = _metric_original_support(mesh, candidate, operation)
    ids = np.asarray(mesh.entity_set(degree).entity_ids)
    original_row = int(np.flatnonzero(ids == identifier)[0])
    orbit, _, shifts = (np.asarray(value) for value in periodic.orbits(degree))
    group = int(orbit[original_row])
    rows = np.flatnonzero(orbit == group)
    vertex_ids = np.asarray(mesh.vertex_global_ids)
    corners = (
        np.arange(vertex_ids.size, dtype=np.int32)[:, None]
        if degree == 0
        else _simplex_entity_corners(mesh, degree)
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
    original_protected = _metric_protected_rows(source, policy)
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
        _metric_visit(current_orbits.size)
        protected[protected_degree].update(
            np.flatnonzero(
                np.isin(current_orbits, current_orbits[selected_rows])
            ).tolist()
        )
    cells = _metric_orbit_cells(mesh)
    class_by_id = _metric_current_classes(source, mesh, prior, policy)
    carriers = []
    for row in rows:
        _metric_visit()
        vertices = tuple(int(value) for value in vertex_ids[corners[row]])
        incident = []
        for cell in cells:
            _metric_visit()
            if (
                bool(set(vertices).intersection(cell.vertices))
                if operation == "collapse"
                else set(vertices) <= set(cell.vertices)
            ):
                incident.append(cell.identifier)
        incidence = tuple(sorted(incident))
        if not incidence:
            raise ValueError(
                "An authoritative periodic carrier has no actual source incidence."
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
    cavity, _, _ = _metric_orbit_cavity(mesh, candidate)
    selected_incidence = set(
        next(
            value.incident_cell_global_ids
            for value in carriers
            if value.entity_global_id == identifier
        )
    )
    if not {cell.identifier for cell in cavity} <= selected_incidence:
        raise ValueError(
            "A metric seed changes cells outside its actual source operation incidence."
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


def _require_metric_candidate_stage(
    source: CellMeshingResult,
    current: CellMesh,
    candidate: CellTopologyEdit,
    stage: PeriodicMetricCandidateStage,
    selection: PeriodicMetricOrbitSelection,
    seed_exact: Mapping[int, ConstructionPointKey],
    /,
) -> tuple[dict[int, ConstructionPointKey], int]:
    if not isinstance(stage, PeriodicMetricCandidateStage) or not isinstance(
        stage.edit, CellTopologyEdit
    ):
        raise TypeError(
            "The actual operation owner must return PeriodicMetricCandidateStage."
        )
    required = {carrier.entity_global_id for carrier in selection.carriers}
    if (
        len(stage.executed_entity_ids) != len(required)
        or set(stage.executed_entity_ids) != required
    ):
        raise ValueError(
            "The owning candidate omits or duplicates authoritative periodic operations."
        )
    edit = stage.edit
    if (
        edit.operation != candidate.operation
        or edit.refinement is not None
        or edit.coarsening is not None
    ):
        raise ValueError(
            "The owning callback changed the actual operation or substituted a family template."
        )
    if edit.shared_faces is not None or edit.prescribed_entity_ids:
        raise ValueError(
            "The owning metric callback cannot substitute restored family witnesses."
        )
    if tuple(record.dimension for record in edit.relations) != tuple(
        range(source.mesh.topological_dimension + 1)
    ):
        raise ValueError(
            "The aggregate periodic candidate omits complete entity ancestry."
        )
    exact = _require_metric_stencils(source.mesh, edit, stage.exact_stencils)
    cavity, _, _ = _metric_orbit_cavity(current, edit)
    changed = {cell.identifier for cell in cavity}
    allowed = {
        identifier
        for carrier in selection.carriers
        for identifier in carrier.incident_cell_global_ids
    }
    if not changed <= allowed:
        raise ValueError(
            "The aggregate operation changes cells outside its complete authoritative incidence."
        )
    for carrier in selection.carriers:
        _metric_visit()
        if not changed.intersection(carrier.incident_cell_global_ids):
            raise ValueError(
                "A purported periodic operation left an authoritative carrier unchanged."
            )
    # Compatible neighboring operations may further subdivide seed children.
    # The actual seed's constructions and moved coordinates remain authoritative.
    seed_cavity, _, _ = _metric_orbit_cavity(current, candidate)
    seed_ids = {cell.identifier for cell in seed_cavity}
    if not seed_ids <= changed:
        raise ValueError(
            "The aggregate operation lost its original private source cavity."
        )
    rows = {int(identifier): row for row, identifier in enumerate(edit.vertex_global_ids)}
    original_rows = {
        int(identifier): row for row, identifier in enumerate(current.vertex_global_ids)
    }
    for row, identifier in enumerate(candidate.vertex_global_ids):
        _metric_visit()
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
                "Overlapping periodic operations prescribe incompatible seed constructions or coordinates."
            )
    return exact, len(changed)


def _require_metric_operation_support(
    current: CellMesh, target: CellMesh, selection: PeriodicMetricOrbitSelection, /
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
        _metric_visit()
        ordered = tuple(
            carrier.vertex_global_ids[carrier.vertex_permutation.index(slot)]
            for slot in range(len(carrier.vertex_global_ids))
        )
        match selection.operation:
            case "split" | "flip":
                if tuple(sorted(ordered)) in edges:
                    raise ValueError(
                        "A synchronized edge operation retains an unedited authoritative carrier."
                    )
            case "collapse":
                if ordered[0] in target_ids or ordered[1] not in target_ids:
                    raise ValueError(
                        "A synchronized collapse loses its exact oriented contraction witness."
                    )
            case "relocate":
                vertex = ordered[0]
                if vertex not in new_rows or np.array_equal(
                    np.asarray(current.coordinates)[old_rows[vertex]],
                    np.asarray(target.coordinates)[new_rows[vertex]],
                ):
                    raise ValueError(
                        "A synchronized relocation leaves an authoritative source copy unchanged."
                    )
            case invalid:
                assert_never(invalid)


def _require_metric_protected_scopes(
    source: CellMeshingResult, target: CellMesh, policy: MeshAdaptationPolicy, /
) -> None:
    protected = _metric_protected_rows(source, policy)
    old_ids = np.asarray(source.mesh.vertex_global_ids)
    new_rows = {
        int(identifier): row for row, identifier in enumerate(target.vertex_global_ids)
    }
    for row in protected[0]:
        _metric_visit()
        identifier = int(old_ids[row])
        if identifier not in new_rows or not np.array_equal(
            np.asarray(source.mesh.coordinates)[row],
            np.asarray(target.coordinates)[new_rows[identifier]],
        ):
            raise ValueError(
                "Periodic metric closure changes a protected source vertex or orbit copy."
            )
    for degree in range(1, source.mesh.topological_dimension + 1):
        _metric_visit()
        present = set(map(tuple, entity_keys(target, degree)))
        keys = entity_keys(source.mesh, degree)
        if any(tuple(keys[row]) not in present for row in protected[degree]):
            raise ValueError(
                "Periodic metric closure splits or removes a protected source entity or orbit copy."
            )


def _require_metric_organization(
    source: CellMeshingResult,
    target: CellMesh,
    edit: CellTopologyEdit,
    policy: MeshAdaptationPolicy,
    /,
) -> None:
    dimension = source.mesh.topological_dimension
    source_classes = _metric_source_classes(source, policy)
    origins: dict[int, set[int]] = {}
    for old, new in zip(
        edit.relations[dimension].source_keys[:, 0],
        edit.relations[dimension].target_keys[:, 0],
        strict=True,
    ):
        _metric_visit()
        if int(old) not in source_classes:
            raise ValueError(
                "Periodic metric cell ancestry names a stale or absent original scientific cell."
            )
        origins.setdefault(int(new), set()).add(source_classes[int(old)])
    for identifier in entity_keys(target, dimension)[:, 0]:
        _metric_visit()
        if int(identifier) in source_classes:
            origins.setdefault(int(identifier), set()).add(
                source_classes[int(identifier)]
            )
        if len(origins.get(int(identifier), ())) != 1:
            raise ValueError(
                "Periodic metric closure merges or loses scientific source cell classes."
            )


def metric_target_lineage(
    source: CellMesh, edit: CellTopologyEdit, target: CellMesh, /
) -> MeshLineage:
    """Bind ancestry to the actual reconstructed carrier by scientific identity."""
    from ._lineage import MeshLineage
    from ._topology_edit import _entity_lineage, key_rows

    staged, _, _ = assemble_topology_edit(
        source, edit, numeric_version=target.numeric_version
    )
    if staged.topological_dimension != target.topological_dimension:
        raise ValueError("Reconstruction changes the staged carrier dimension.")
    for degree in range(source.topological_dimension + 1):
        staged_ids = np.asarray(staged.entity_set(degree).entity_ids)
        target_ids = np.asarray(target.entity_set(degree).entity_ids)
        rows = key_rows(staged_ids[:, None], target_ids[:, None])
        if staged_ids.size != target_ids.size or np.any(rows < 0):
            raise ValueError(
                "Reconstruction changes scientific carrier entity identities."
            )
        if not np.array_equal(
            entity_keys(staged, degree)[rows], entity_keys(target, degree)
        ):
            raise ValueError("Reconstruction changes scientific carrier incidence.")
    if (staged.periodic_topology is None) != (target.periodic_topology is None):
        raise ValueError("Reconstruction changes the periodic quotient identity.")
    if staged.periodic_topology is not None:
        from ..discretization._periodic_topology import _identification_id

        if target.periodic_topology is None:
            raise RuntimeError(
                "Matched periodic reconstruction lost its target quotient."
            )
        if _identification_id(staged.periodic_topology.cell) != _identification_id(
            target.periodic_topology.cell
        ):
            raise ValueError(
                "Reconstruction changes the original periodic identification."
            )
        _require_complete_periodic_entity_changes(source, target, edit)
    return MeshLineage(
        source.topology_id,
        target.topology_id,
        tuple(_entity_lineage(source, target, relation) for relation in edit.relations),
    )


def _require_metric_geometry_stage(
    source: CellMeshingResult,
    target: CellMesh,
    edit: CellTopologyEdit,
    stage: PeriodicMetricGeometryStage,
    policy: MeshAdaptationPolicy,
    certificate_limits: MeshCertificateLimits,
    /,
) -> None:
    if not isinstance(stage, PeriodicMetricGeometryStage):
        raise TypeError(
            "The owning geometry builder must return PeriodicMetricGeometryStage."
        )
    if stage.target_mesh is not target:
        raise ValueError(
            "The periodic geometry stage must retain its actual target carrier."
        )
    if not isinstance(stage.target_geometry, CellGeometrySpec) or not isinstance(
        stage.geometry_transition, CellGeometryTransition
    ):
        raise TypeError(
            "A periodic geometry stage requires actual complete geometry and transition owners."
        )
    correspondence = stage.source_correspondence
    if not isinstance(
        correspondence, (PreparedSurfaceChartDeformation, PreparedSphereChartDeformation)
    ):
        raise TypeError(
            "Periodic source correspondence requires its actual Surface or Sphere owner."
        )
    stage.target_geometry.resolve(target)
    correspondence.require_bound(
        source.mesh, source.geometry, target, stage.target_geometry
    )
    if (
        correspondence.target_validity.policy_id
        != policy.audit_policy.validity_policy.policy_id
    ):
        raise ValueError(
            "The periodic full-map builder renewed or changed the original validity policy."
        )
    if correspondence.target_embedding.binding.limits_id != certificate_limits.limits_id:
        raise ValueError(
            "The periodic full-map builder changed the original authored certificate limits."
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
            "The periodic full-map transition is stale for its actual source or target."
        )
    if not np.array_equal(transition.vertex_coordinates, target.coordinates):
        raise ValueError(
            "The periodic full-map transition does not place the actual target vertices."
        )
    if not isinstance(stage.source_membership, tuple) or not stage.source_membership:
        raise ValueError(
            "The periodic full-map builder must return complete actual source membership."
        )
    declared = {
        (association.source_id, association.source_revision)
        for association in source.associations
    }
    dimensions = {}
    for association in stage.source_membership:
        _metric_visit()
        if not isinstance(association, GeometryAssociation):
            raise TypeError(
                "Periodic source membership requires canonical GeometryAssociation records."
            )
        if (
            association.source_id,
            association.source_revision,
        ) not in declared or not association.complete:
            raise ValueError(
                "Periodic successor membership loses its original source/version or is undecided."
            )
        matches = [
            degree
            for degree in range(target.topological_dimension + 1)
            if target.entity_set(degree).entity_set_id == association.target_entity_set_id
        ]
        if len(matches) != 1 or matches[0] in dimensions:
            raise ValueError(
                "Periodic successor membership is stale or duplicates an entity dimension."
            )
        degree = matches[0]
        association.validate_target(target.entity_set(degree))
        association.target_rows(np.asarray(target.entity_set(degree).entity_ids))
        dimensions[degree] = association
    if set(dimensions) != set(range(target.topological_dimension + 1)):
        raise ValueError(
            "Periodic successor membership omits an actual mesh entity dimension."
        )
    for association in source.associations:
        _metric_visit()
        matches = [
            degree
            for degree in range(source.mesh.topological_dimension + 1)
            if source.mesh.entity_set(degree).entity_set_id
            == association.target_entity_set_id
        ]
        if len(matches) != 1:
            raise ValueError(
                "Source geometry membership does not bind its scientific mesh epoch."
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
            _metric_visit()
            row = int(source_rows[old_row])
            ambiguous = bool(np.asarray(association.ambiguous)[row])
            corner = (
                degree == 0 and int(np.asarray(association.source_dimensions)[row]) == 0
            )
            if ambiguous or corner:
                if tuple(key) not in new_key_rows:
                    raise ValueError(
                        "Periodic metric closure removes a fixed or ambiguous source class."
                    )
                if degree == 0 and not np.array_equal(
                    np.asarray(source.mesh.coordinates)[old_row],
                    np.asarray(target.coordinates)[new_key_rows[tuple(key)]],
                ):
                    raise ValueError(
                        "Periodic metric closure relocates a scientific source corner."
                    )
            if tuple(key) in new_key_rows:
                after = int(target_rows[new_key_rows[tuple(key)]])
                if (
                    association.source_entity_ids[row],
                    association.source_occurrence_paths[row],
                ) != (
                    successor.source_entity_ids[after],
                    successor.source_occurrence_paths[after],
                ):
                    raise ValueError(
                        "Periodic metric closure changes a retained scientific source stratum."
                    )


def _require_metric_output_limits(
    mesh: CellMesh,
    stage: PeriodicMetricGeometryStage | None,
    edit: CellTopologyEdit,
    limits: MeshingLimits,
    /,
) -> None:
    from .._model._structure import model_array_bytes
    from .providers._native_publication import _published_entity_limits

    _metric_visit(mesh.topological_dimension + 1 + len(mesh.topology.incidences))
    _published_entity_limits(mesh, limits)
    _metric_visit()
    if model_array_bytes((mesh, stage, edit)) > limits.maximum_data_bytes:
        raise ValueError(
            "Periodic metric full-map stage exceeds its complete original retained-data capacity."
        )


def _metric_identity_storage(
    banks: tuple[PeriodicEntityIdentityBank, ...],
    witness: PeriodicVertexOrbitWitness | None,
    ledger: CoordinateEnclosureBudget,
    /,
) -> int:
    """Bound actual retained bank DAG metadata, separately from numerical buffers."""
    import sys

    pending_banks: list[tuple[PeriodicEntityIdentityBank, ...]] = [banks]
    pending_witnesses = [] if witness is None else [witness]
    seen_witnesses: set[int] = set()
    storage = 0
    while pending_witnesses:
        owner = pending_witnesses.pop()
        _metric_visit()
        if id(owner) in seen_witnesses:
            continue
        seen_witnesses.add(id(owner))
        # Bank-reference lists and witness dedup tables are included before grow.
        ledger.reserve(0, 1024)
        storage += 1024
        pending_banks.extend((owner.quotient_entities, owner.retained_quotient_entities))
        if owner.allocation_prior is not None:
            pending_witnesses.append(owner.allocation_prior[1])
    seen: set[int] = set()
    pending: list[object] = list(pending_banks)
    while pending:
        value = pending.pop()
        _metric_visit()
        if id(value) in seen:
            continue
        seen.add(id(value))
        # Actual immutable tuple/int payload plus conservative stack/set overhead.
        upper = sys.getsizeof(value) + 512
        ledger.reserve(0, upper)
        storage += upper
        if isinstance(value, tuple):
            pending.extend(value)
    return storage


def _metric_controller_admission(
    source: CellMeshingResult,
    candidate: CellTopologyEdit,
    exact_stencils: Mapping[int, ConstructionPointKey],
    limits: MeshingLimits,
    coordinate_budget: CoordinateEnclosureBudget,
    /,
) -> tuple[int, int]:
    import sys

    # The authoritative support has at most one carrier per source entity.
    # Each carrier can inspect the source incidence bank; dictionaries and exact
    # canonical keys have a separately proved CPython bound (not native RSS).
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
    digits = (
        max(1, bit_bound) + sys.int_info.bits_per_digit - 1
    ) // sys.int_info.bits_per_digit
    coefficient_upper = (
        512
        + 32 * source.mesh.ambient_dimension
        + 2 * (sys.getsizeof(0) + digits * sys.int_info.sizeof_digit)
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
    execution = current_native_execution_budget()
    if execution is None:
        raise ValueError(
            "Periodic metric synchronization requires the actual original native execution scope."
        )
    allowance = execution.remaining()
    if host_upper > min(
        limits.maximum_scratch_bytes, coordinate_budget.maximum_memory_bytes
    ):
        raise ValueError(
            "Periodic metric host-object preparation exceeds its proven original scratch upper bound."
        )
    if work_upper > limits.maximum_work_units or allowance.remaining_wall_seconds <= 0:
        raise ValueError(
            "Periodic metric preparation exceeds the original work or wall allowance."
        )
    execution.admit_work_bound(work_upper)
    coordinate_budget.reserve(0, host_upper)
    return work_upper, host_upper


def synchronize_periodic_metric_edit(
    source: CellMeshingResult,
    candidate: CellTopologyEdit,
    exact_stencils: Mapping[int, ConstructionPointKey],
    /,
    *,
    operation: PeriodicMetricOperation,
    limits: MeshingLimits,
    policy: MeshAdaptationPolicy,
    coordinate_budget: CoordinateEnclosureBudget,
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
    build_geometry: Callable[[CellTopologyEdit, CellMesh], PeriodicMetricGeometryStage],
    retained_quotient_entities: tuple[PeriodicEntityIdentityBank, ...],
) -> PeriodicMetricOrbitOutcome:
    """Stage complete authoritative carrier operations, full maps and correspondence.

    Nothing mutates the accepted source. Any invalid orbit, incompatible cavity,
    undecided source geometry or original resource refusal raises before a
    successor is returned. The caller's geometry builder owns reconstruction,
    source queries and its original fidelity/quality admission.
    """
    from ._adaptation import MeshAdaptationPolicy

    if not isinstance(source, CellMeshingResult) or not isinstance(
        candidate, CellTopologyEdit
    ):
        raise TypeError(
            "Periodic metric synchronization requires actual source and private edit records."
        )
    if not isinstance(policy, MeshAdaptationPolicy) or not isinstance(
        limits, MeshingLimits
    ):
        raise TypeError(
            "Periodic metric synchronization requires the authored adaptation policy and limits."
        )
    if not bool(eqx.tree_equal(limits, policy.limits, typematch=True)):
        raise ValueError(
            "Periodic metric synchronization cannot reset the original policy limits."
        )
    if (
        not isinstance(coordinate_budget, CoordinateEnclosureBudget)
        or not callable(build_geometry)
        or not callable(build_candidate)
    ):
        raise TypeError(
            "Periodic synchronization requires its original ledger and actual candidate/geometry owners."
        )
    if not isinstance(certificate_limits, MeshCertificateLimits):
        raise TypeError(
            "Periodic synchronization requires the original authored certificate limits."
        )
    selected = parse(operation, PeriodicMetricOperation, "operation")
    if selected == "relocate" and not policy.relocation:
        raise ValueError("The original adaptation policy forbids periodic relocation.")
    if candidate.operation not in ("local_reconnection", "relocation"):
        raise ValueError(
            "Nested family constructions must use their concrete periodic template owner."
        )
    if (selected == "relocate") != (candidate.operation == "relocation"):
        raise ValueError(
            "The periodic operation does not describe the actual private metric candidate."
        )
    if candidate.refinement is not None or candidate.coarsening is not None:
        raise ValueError(
            "Local metric candidates cannot substitute nested family geometry witnesses."
        )
    if candidate.shared_faces is not None or candidate.prescribed_entity_ids:
        raise ValueError(
            "A concrete restored/template candidate must be synchronized by its owning family."
        )
    dimension = source.mesh.topological_dimension
    if tuple(record.dimension for record in candidate.relations) != tuple(
        range(dimension + 1)
    ):
        raise ValueError("The private metric candidate omits complete entity ancestry.")
    if not source.coordinate_contract.is_orthonormal_cartesian:
        raise ValueError(
            "Periodic metric group actions require the actual Cartesian source frame."
        )
    source.geometry.resolve(source.mesh)
    orbits = PeriodicConstructionOrbits(source.mesh)
    original_geometry_id = cell_geometry_id(source.geometry)
    current = source.mesh
    current_geometry = source.geometry
    retained = retained_quotient_entities
    allocation_prior = None
    if (
        candidate.periodic_orbits is not None
        and candidate.periodic_orbits.source_topology_id != source.mesh.topology_id
    ):
        raise ValueError(
            "The private metric candidate carries a stale scientific source topology."
        )
    execution = current_native_execution_budget()
    if execution is None:
        raise ValueError(
            "Periodic synchronization requires the original native execution scope."
        )
    controller_work = _MetricControllerWork(coordinate_budget, execution)
    with (
        coordinate_budget.activate(),
        coordinate_budget.temporary_scope(),
        controller_work.activate(),
    ):
        _, host_upper = _metric_controller_admission(
            source, candidate, exact_stencils, limits, coordinate_budget
        )
        if prior_stage is not None:
            if not isinstance(prior_stage, PeriodicMetricOrbitOutcome):
                raise TypeError(
                    "A continued periodic edit requires the actual prior private stage."
                )
            if (
                coordinate_budget.work_units < prior_stage.coordinate_work_units
                or coordinate_budget.peak_bytes_upper
                < prior_stage.coordinate_peak_storage_upper_bytes
            ):
                raise ValueError(
                    "A private periodic continuation cannot renew its original coordinate ledger."
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
                    "The private periodic continuation is stale or renews the original source policy."
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
                    "The private periodic continuation changed its actual current coordinate frame or map."
                )
            if prior_stage.periodic_witness is not prior_stage.edit.periodic_orbits:
                raise ValueError(
                    "The private periodic continuation lost its exact owning orbit witness."
                )
            current_periodic = _require_periodic_topology(current)
            if (
                prior_stage.evidence.embedding.periodic_topology_id
                != current_periodic.periodic_topology_id
                or prior_stage.evidence.quotient.periodic_topology_id
                != current_periodic.periodic_topology_id
                or prior_stage.evidence.status != "complete"
            ):
                raise ValueError(
                    "The private periodic continuation lost its complete actual quotient embedding evidence."
                )
            _require_metric_stencils(
                source.mesh, prior_stage.edit, prior_stage.exact_stencils
            )
            _require_metric_geometry_stage(
                source,
                current,
                prior_stage.edit,
                prior_stage.geometry_stage,
                policy,
                certificate_limits,
            )
            _require_metric_protected_scopes(source, current, policy)
            _require_metric_organization(source, current, prior_stage.edit, policy)
            retained = prior_stage.periodic_witness.quotient_entities
            allocation_prior = (current, prior_stage.periodic_witness)
        host_upper += _metric_identity_storage(
            retained,
            None if allocation_prior is None else allocation_prior[1],
            coordinate_budget,
        )
        current_orbits = PeriodicConstructionOrbits(current)
        stencils = _require_metric_stencils(source.mesh, candidate, exact_stencils)
        selection = _metric_orbit_selection(
            source,
            candidate,
            selected,
            policy,
            current_orbits,
            current,
            current_geometry,
            prior_stage,
        )
        required_incidence = {
            identifier
            for carrier in selection.carriers
            for identifier in carrier.incident_cell_global_ids
        }
        execution.admit_cavity(len(required_incidence))
        controller_work.flush()
        candidate_stage = build_candidate(candidate, stencils, selection)
        controller_work.flush()
        orbits.require_source(source.mesh)
        current_orbits.require_source(current)
        if cell_geometry_id(current_geometry) != selection.source_geometry_id:
            raise ValueError("The owning callback changed the actual current source map.")
        exact, cavity_count = _require_metric_candidate_stage(
            source, current, candidate, candidate_stage, selection, stencils
        )
        edit = candidate_stage.edit
        _, additional_host_upper = _metric_controller_admission(
            source, edit, exact, limits, coordinate_budget
        )
        host_upper += additional_host_upper
        from ._topology_edit import _build_mesh

        probe = _build_mesh(edit, source.mesh.numeric_version, None)
        _require_metric_operation_support(current, probe, selection)
        _metric_visit(probe.coordinates.shape[0])
        witness = orbits.witness(
            source.mesh, probe, exact, retained, allocation_prior=allocation_prior
        )
        host_upper += _metric_identity_storage(
            witness.quotient_entities, witness, coordinate_budget
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
        target, _, _ = assemble_topology_edit(source.mesh, edit, numeric_version=version)
        _require_metric_output_limits(target, None, edit, limits)
        _require_complete_periodic_entity_changes(source.mesh, target, edit)
        _require_metric_protected_scopes(source, target, policy)
        _require_metric_organization(source, target, edit, policy)
        orbits.require_source(source.mesh)
        controller_work.flush()
        stage = build_geometry(edit, target)
        controller_work.flush()
        if not isinstance(stage, PeriodicMetricGeometryStage) or not isinstance(
            stage.target_mesh, CellMesh
        ):
            raise TypeError(
                "The owning geometry builder must retain its actual target CellMesh."
            )
        target = stage.target_mesh
        metric_target_lineage(source.mesh, edit, target)
        _require_metric_protected_scopes(source, target, policy)
        _require_metric_organization(source, target, edit, policy)
        orbits.require_source(source.mesh)
        current_orbits.require_source(current)
        if cell_geometry_id(current_geometry) != selection.source_geometry_id:
            raise ValueError(
                "The geometry builder changed the actual current source map."
            )
        _require_metric_geometry_stage(
            source, target, edit, stage, policy, certificate_limits
        )
        _require_metric_output_limits(target, stage, edit, limits)
        controller_work.flush()
        remaining = execution.remaining()
        if remaining.remaining_work_units <= 0 or remaining.remaining_scratch_bytes <= 0:
            raise ValueError(
                "The original periodic certificate work or storage allowance is exhausted."
            )
        bounded_certificate_limits = MeshCertificateLimits(
            maximum_candidate_pairs=min(
                certificate_limits.maximum_candidate_pairs, remaining.remaining_work_units
            ),
            maximum_ray_tests=certificate_limits.maximum_ray_tests,
            maximum_source_samples=certificate_limits.maximum_source_samples,
            maximum_distance_evaluations=certificate_limits.maximum_distance_evaluations,
            maximum_subdivision_depth=certificate_limits.maximum_subdivision_depth,
            maximum_subdivision_pieces=certificate_limits.maximum_subdivision_pieces,
            maximum_bernstein_nodes=certificate_limits.maximum_bernstein_nodes,
            maximum_periodic_images=min(
                certificate_limits.maximum_periodic_images,
                remaining.remaining_scratch_bytes // max(1, target.coordinates.size * 8),
            ),
            maximum_work_units=min(
                certificate_limits.maximum_work_units, remaining.remaining_work_units
            ),
            maximum_scratch_bytes=min(
                certificate_limits.maximum_scratch_bytes,
                remaining.remaining_scratch_bytes,
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
        )
        orbits.require_source(source.mesh)
        current_orbits.require_source(current)
        if cell_geometry_id(current_geometry) != selection.source_geometry_id:
            raise ValueError(
                "The quotient certificate changed the actual current source map."
            )
        if cell_geometry_id(source.geometry) != original_geometry_id:
            raise ValueError(
                "The periodic operation changed its immutable original full source map."
            )
        execution.charge(work=0)
        witness = edit.periodic_orbits
        if witness is None:
            raise RuntimeError(
                "A synchronized metric edit lost its canonical periodic witness."
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
        from types import MappingProxyType

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
            coordinate_budget.work_units,
            coordinate_budget.peak_bytes_upper,
            evidence,
        )
