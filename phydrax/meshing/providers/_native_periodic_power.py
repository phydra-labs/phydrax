#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from fractions import Fraction
from itertools import pairwise, permutations, product
from math import ceil, floor, isfinite, ldexp, nextafter, sqrt
from typing import Any, TYPE_CHECKING

import numpy as np
from jax.typing import ArrayLike

import phydrax._meshcore as meshcore
from phydrax._fingerprint import canonical_fingerprint
from phydrax.discretization._cell_complex import PolyhedralConnectivity
from phydrax.discretization._cell_geometry import CellGeometrySpec
from phydrax.discretization._cell_mesh import CellBlock, CellMesh
from phydrax.discretization._exact_power_geometry import (
    _prepare_power_site_bank,
    _vertex_power_point,
    ExactPeriodicPowerSourceContext,
    ExactPowerCellGeometrySource,
)
from phydrax.discretization._periodic_exact import (
    exact_periodic_element,
    identification_orders,
    reduce_periodic_exponents,
)
from phydrax.discretization._periodic_topology import PeriodicMeshTopology
from phydrax.meshing._periodic import PeriodicMeshConstruction, PeriodicPointOrbits
from phydrax.meshing._plc_source import _numeric_entity_ids


if TYPE_CHECKING:
    from phydrax._execution_runtime import ExecutionGroup
    from phydrax.meshing.providers._native_periodic import NativePeriodicSource


type _PowerSiteSignature = tuple[tuple[int, tuple[int, ...]], ...]
type _PowerPointSignature = tuple[
    _PowerSiteSignature, tuple[tuple[int, int], ...]
]
type _PowerVertexKey = tuple[_PowerPointSignature, tuple[int, ...]]
type _CarrierProof = tuple[tuple[int, tuple[int, int]], ...]


def _convex_power_face_loop(
    loop: np.ndarray,
    exact_points: dict[int, tuple[Fraction, Fraction, Fraction]],
    /,
) -> np.ndarray:
    """Remove carrier-only collinear subdivisions from an exact power face."""
    rows = tuple(int(value) for value in loop)
    points = tuple(exact_points[row] for row in rows)
    if len(set(points)) != len(points) or len(points) < 3:
        raise ValueError(
            "Exact periodic power face has duplicate or insufficient vertices."
        )
    origin = points[0]
    normal = None
    for first in range(1, len(points)):
        left = tuple(points[first][axis] - origin[axis] for axis in range(3))
        for second in range(first + 1, len(points)):
            right = tuple(points[second][axis] - origin[axis] for axis in range(3))
            candidate = (
                left[1] * right[2] - left[2] * right[1],
                left[2] * right[0] - left[0] * right[2],
                left[0] * right[1] - left[1] * right[0],
            )
            if any(candidate):
                normal = candidate
                break
        if normal is not None:
            break
    if normal is None:
        raise ValueError("Exact periodic power face vertices are collinear.")
    if any(
        sum(
            (normal[axis] * (point[axis] - origin[axis]) for axis in range(3)),
            Fraction(0),
        )
        for point in points
    ):
        raise ValueError("Exact periodic power face vertices are not coplanar.")
    dropped_axis = max(range(3), key=lambda axis: abs(normal[axis]))
    projected_axes = tuple(axis for axis in range(3) if axis != dropped_axis)
    projected_by_point: dict[tuple[Fraction, Fraction], int] = {}
    for row, point in zip(rows, points, strict=True):
        projected = (point[projected_axes[0]], point[projected_axes[1]])
        if projected in projected_by_point:
            raise ValueError("Exact periodic power face projects distinct rows to one point.")
        projected_by_point[projected] = row

    projected = tuple(sorted(projected_by_point))

    def cross(
        first: tuple[Fraction, Fraction],
        second: tuple[Fraction, Fraction],
        third: tuple[Fraction, Fraction],
        /,
    ) -> Fraction:
        return (second[0] - first[0]) * (third[1] - first[1]) - (
            second[1] - first[1]
        ) * (third[0] - first[0])

    lower: list[tuple[Fraction, Fraction]] = []
    for point in projected:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[Fraction, Fraction]] = []
    for point in reversed(projected):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        raise ValueError("Exact periodic power face has no convex polygon boundary.")

    def twice_area(
        points_: tuple[tuple[Fraction, Fraction], ...]
        | list[tuple[Fraction, Fraction]],
        /,
    ) -> Fraction:
        return sum(
            (
                points_[index][0] * points_[(index + 1) % len(points_)][1]
                - points_[index][1] * points_[(index + 1) % len(points_)][0]
                for index in range(len(points_))
            ),
            Fraction(0),
        )

    loop_projected = tuple(
        (point[projected_axes[0]], point[projected_axes[1]]) for point in points
    )
    source_area = twice_area(loop_projected)
    hull_area = twice_area(hull)
    if source_area == 0 or abs(source_area) != abs(hull_area):
        raise ValueError(
            "Exact periodic power face is not its convex carrier-clipped polygon."
        )
    if source_area * hull_area < 0:
        hull.reverse()
    return np.asarray([projected_by_point[point] for point in hull], dtype=np.int64)


def periodic_power_mesh(
    source: PeriodicPointOrbits,
    site_weights: ArrayLike | None = None,
    /,
    *,
    source_id: str | None = None,
    source_revision: str | None = None,
) -> PeriodicMeshConstruction:
    """Construct weighted polyhedral power cells in the periodic quotient.

    Exact restricted-power cells are clipped to a growing carrier that contains
    every base-image cell. Image coordinates are exact expansions of authored
    points plus integer lattice translations. Exact source-bound intersections
    and deck actions identify quotient vertices; rounded coordinates do not.
    """
    if not isinstance(source, PeriodicPointOrbits):
        raise TypeError("source must be PeriodicPointOrbits.")
    if source.ambient_dimension != 3:
        raise ValueError("Periodic power cells require a three-dimensional periodic lattice.")
    if source_id is not None and not isinstance(source_id, str):
        raise TypeError("source_id must be a string when supplied.")
    if source_revision is not None and not isinstance(source_revision, str):
        raise TypeError("source_revision must be a string when supplied.")
    geometry_source_id = source.orbits_id if source_id is None else source_id.strip()
    geometry_source_revision = (
        source.orbits_id if source_revision is None else source_revision.strip()
    )
    if not geometry_source_id or not geometry_source_revision:
        raise ValueError("Periodic power geometry source identity must be nonempty.")
    identifiers = np.asarray(source.point_ids, dtype=np.int64)
    if identifiers.size == 0:
        raise ValueError("Global periodic power requires at least one authored site.")
    order = np.argsort(identifiers, kind="stable")
    point_ids = identifiers[order]
    points = np.ascontiguousarray(np.asarray(source.points, dtype=np.float64)[order])
    if site_weights is None:
        authored_weights = np.zeros((identifiers.size,), dtype=np.float64)
    else:
        raw_weights = np.asarray(site_weights)
        if raw_weights.shape != identifiers.shape or raw_weights.dtype.kind not in "iuf":
            raise TypeError("site_weights must be one real weight per authored point ID.")
        authored_weights = np.asarray(raw_weights, dtype=np.float64)
        if np.any(~np.isfinite(authored_weights)):
            raise ValueError("Periodic power site weights must be finite.")
    weights = authored_weights[order]
    lattice = np.ascontiguousarray(np.asarray(source.cell.vectors, dtype=np.float64))
    inverse = _exact_lattice_inverse(lattice)
    radius = _lattice_covering_radius(lattice)
    weight_fractions = tuple(Fraction.from_float(float(value)) for value in weights)
    influence_radius = _upper_sqrt(
        Fraction.from_float(radius) ** 2
        + max(weight_fractions)
        - min(weight_fractions)
    )
    preferred_owner = int(np.argmax(weights))
    lower = np.min(points, axis=0)
    upper = np.max(points, axis=0)
    margin = radius
    while True:
        carrier_lower = _outward_bound(lower - margin, -np.inf)
        carrier_upper = _outward_bound(upper + margin, np.inf)
        image_owners, image_windings = _power_images(
            points,
            point_ids,
            inverse,
            carrier_lower,
            carrier_upper,
            influence_radius,
            preferred_owner,
        )
        if image_owners.size >= np.iinfo(np.int32).max:
            raise OverflowError("Periodic power image bank exceeds native index capacity.")
        coordinate_offsets, coordinate_components = _exact_image_coordinates(
            points, lattice, image_owners, image_windings
        )
        neighbor_offsets, neighbors = _image_radius_neighbor_csr(
            image_owners,
            image_windings,
            coordinate_offsets,
            coordinate_components,
            influence_radius,
        )
        domain_points, domain_tets, domain_face_facets = _box_tetrahedra(
            carrier_lower, carrier_upper
        )
        restricted = meshcore.restricted_power_cells_exact(
            points,
            weights,
            image_owners,
            coordinate_offsets,
            coordinate_components,
            neighbor_offsets,
            neighbors,
            np.zeros((image_owners.size,), dtype=np.int8),
            domain_points,
            domain_tets,
            np.zeros((domain_tets.shape[0],), dtype=np.int32),
            domain_face_facets,
        )
        base_images = {
            int(point_row): _image_row(image_owners, image_windings, point_row)
            for point_row in range(point_ids.size)
        }
        cell_rows = _base_image_cell_rows(restricted, base_images, point_ids.size)
        if _base_cells_are_unclipped(restricted, cell_rows):
            (
                lifted,
                exact_vertex_bank,
                carrier_barycentric_bank,
                output_vertex_rows,
                exact_image_points,
                _,
                _,
                _,
                vertex_representatives,
                vertex_shifts,
            ) = _polyhedral_quotient_from_power(
                source,
                geometry_source_id,
                geometry_source_revision,
                point_ids,
                np.zeros((point_ids.size,), dtype=np.int32),
                weights,
                image_owners,
                image_windings,
                coordinate_offsets,
                coordinate_components,
                domain_points,
                restricted,
                cell_rows,
            )
            carrier_mesh = CellMesh(
                domain_points,
                (
                    CellBlock(
                        "periodic-power-carrier", "tetrahedron", domain_tets
                    ),
                ),
            )
            periodic_source_context = ExactPeriodicPowerSourceContext(
                source,
                authored_weights,
                image_owners,
                image_windings,
                weights_present=site_weights is not None,
            )
            source_map = ExactPowerCellGeometrySource(
                np.asarray(exact_image_points, dtype=np.float64),
                np.asarray(weights[image_owners], dtype=np.float64),
                carrier_mesh,
                restricted,
                source_id=geometry_source_id,
                source_revision=geometry_source_revision,
                periodic_source_context=periodic_source_context,
                exact_site_points=exact_image_points,
                precomputed_exact_vertices=exact_vertex_bank,
                precomputed_carrier_barycentrics=carrier_barycentric_bank,
                output_vertex_indices=output_vertex_rows,
            )
            if source_map.exact_vertices != exact_vertex_bank:
                raise ValueError(
                    "Exact periodic power geometry differs from quotient vertex ancestry."
                )
            mesh = _bind_power_periodic_topology(
                lifted,
                source,
                vertex_representatives,
                vertex_shifts,
                source_map,
                point_ids,
            )
            source_map.require_periodic_mesh(mesh)
            _validate_periodic_power_measure(
                mesh, restricted, cell_rows, point_ids.size, lattice
            )
            geometry = CellGeometrySpec.power(mesh, source_map)
            return PeriodicMeshConstruction(mesh, geometry=geometry)
        if not isfinite(margin * 2.0):
            raise RuntimeError("Periodic power carrier expansion did not resolve all base cells.")
        margin = _outward_scalar(margin * 2.0)


def _upper_sqrt(value: Fraction, /) -> float:
    if value <= 0:
        raise ValueError("Periodic power radius squared must be positive.")
    numerator, denominator = value.numerator, value.denominator
    exponent = numerator.bit_length() - denominator.bit_length()
    if exponent >= 0:
        if numerator < denominator << exponent:
            exponent -= 1
    elif numerator << -exponent < denominator:
        exponent -= 1
    half_exponent = exponent // 2
    scale = -2 * half_exponent
    scaled = (
        Fraction(numerator << scale, denominator)
        if scale >= 0
        else Fraction(numerator, denominator << -scale)
    )
    try:
        radius = ldexp(sqrt(float(scaled)), half_exponent)
    except OverflowError as error:
        raise OverflowError("Periodic power radius exceeds binary64 range.") from error
    if not isfinite(radius):
        raise OverflowError("Periodic power radius exceeds finite binary64 range.")
    if radius <= 0.0:
        radius = nextafter(0.0, np.inf)
    while Fraction.from_float(radius) ** 2 < value:
        radius = nextafter(radius, np.inf)
        if not isfinite(radius):
            raise OverflowError("Periodic power radius exceeds finite binary64 range.")
    return radius


def _lattice_covering_radius(lattice: np.ndarray, /) -> float:
    column_sums = tuple(
        sum(
            (abs(Fraction.from_float(float(lattice[row, column]))) for row in range(3)),
            Fraction(0),
        )
        for column in range(3)
    )
    squared = sum(((value / 2) ** 2 for value in column_sums), Fraction(0))
    return _upper_sqrt(squared)


def _outward_bound(values: np.ndarray, direction: float, /) -> np.ndarray:
    result = np.nextafter(np.asarray(values, dtype=np.float64), direction)
    if np.any(~np.isfinite(result)):
        raise OverflowError("Periodic power carrier bounds exceed finite binary64 storage.")
    return result


def _outward_scalar(value: float, /) -> float:
    result = nextafter(value, np.inf)
    if not isfinite(result):
        raise OverflowError("Periodic power carrier growth exceeds finite binary64 storage.")
    return result


def _exact_lattice_inverse(
    lattice: np.ndarray, /
) -> tuple[tuple[Fraction, ...], ...]:
    matrix = np.asarray(lattice, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Exact periodic lattice inversion requires a square matrix.")
    dimension = matrix.shape[0]
    augmented = [
        [
            *(Fraction.from_float(float(matrix[row, column])) for column in range(dimension)),
            *(Fraction(int(row == column)) for column in range(dimension)),
        ]
        for row in range(dimension)
    ]
    for column in range(dimension):
        pivot = next(
            (row for row in range(column, dimension) if augmented[row][column]), None
        )
        if pivot is None:
            raise ValueError("Periodic lattice is singular in exact binary arithmetic.")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(dimension):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor:
                augmented[row] = [
                    value - factor * pivot_value
                    for value, pivot_value in zip(
                        augmented[row], augmented[column], strict=True
                    )
                ]
    return tuple(tuple(row[dimension:]) for row in augmented)


def _power_images(
    points: np.ndarray,
    point_ids: np.ndarray,
    inverse: tuple[tuple[Fraction, ...], ...],
    carrier_lower: np.ndarray,
    carrier_upper: np.ndarray,
    influence_radius: float,
    preferred_owner: int,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    image_lower = _outward_bound(carrier_lower - influence_radius, -np.inf)
    image_upper = _outward_bound(carrier_upper + influence_radius, np.inf)
    corners = tuple(
        product(*zip(image_lower.tolist(), image_upper.tolist(), strict=True))
    )
    owners: list[int] = []
    windings: list[tuple[int, int, int]] = []
    limit = int(np.iinfo(np.int32).max) - 1
    for point_row, point in enumerate(points):
        point_exact = tuple(Fraction.from_float(float(value)) for value in point)
        fractional_values: list[list[Fraction]] = [[], [], []]
        for corner in corners:
            delta = tuple(
                Fraction.from_float(float(corner[axis])) - point_exact[axis]
                for axis in range(3)
            )
            for axis in range(3):
                fractional_values[axis].append(
                    sum(
                        (delta[component] * inverse[component][axis] for component in range(3)),
                        Fraction(0),
                    )
                )
        first = tuple(floor(min(values)) - 1 for values in fractional_values)
        last = tuple(ceil(max(values)) + 1 for values in fractional_values)
        ranges = tuple(range(first[axis], last[axis] + 1) for axis in range(3))
        count = 1
        for values in ranges:
            count *= len(values)
        if count > limit - len(owners):
            raise OverflowError("Periodic power image bank exceeds native index capacity.")
        for winding in product(*ranges):
            if any(abs(int(value)) > np.iinfo(np.int32).max for value in winding):
                raise OverflowError("Periodic power image winding exceeds topology index capacity.")
            owners.append(point_row)
            windings.append((int(winding[0]), int(winding[1]), int(winding[2])))
    identities = {
        (int(owner), tuple(int(value) for value in winding))
        for owner, winding in zip(owners, windings, strict=True)
    }
    if len(identities) != len(owners):
        raise ValueError("Periodic power image identities are not unique.")
    owner_array = np.asarray(owners, dtype=np.int32)
    winding_array = np.asarray(windings, dtype=np.int64).reshape((-1, 3))
    base = np.all(winding_array == 0, axis=1)
    preferred = base & (owner_array == preferred_owner)
    priority = np.where(preferred, 0, np.where(base, 1, 2))
    stable = np.lexsort(
        (
            winding_array[:, 2],
            winding_array[:, 1],
            winding_array[:, 0],
            point_ids[owner_array],
            priority,
        )
    )
    return owner_array[stable], winding_array[stable]


def _image_row(
    owners: np.ndarray, windings: np.ndarray, point_row: int, /
) -> int:
    matches = np.flatnonzero(
        (owners == point_row) & np.all(windings == 0, axis=1)
    )
    if matches.size != 1:
        raise ValueError("Periodic power image bank must contain each authored base site exactly once.")
    return int(matches[0])


def _exact_image_coordinates(
    points: np.ndarray,
    lattice: np.ndarray,
    owners: np.ndarray,
    windings: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    offsets = [0]
    components: list[float] = []
    for owner, winding in zip(owners.tolist(), windings, strict=True):
        for axis in range(3):
            value = Fraction.from_float(float(points[int(owner), axis]))
            for generator in range(3):
                value += int(winding[generator]) * Fraction.from_float(
                    float(lattice[generator, axis])
                )
            expansion = _float_expansion(value)
            components.extend(expansion)
            offsets.append(len(components))
    return (
        np.asarray(offsets, dtype=np.int64),
        np.asarray(components, dtype=np.float64),
    )


def _float_expansion(value: Fraction, /) -> tuple[float, ...]:
    values: list[float] = []
    remainder = value
    while remainder:
        try:
            term = float(remainder)
        except OverflowError as error:
            raise OverflowError("Periodic exact image coordinate exceeds binary64 range.") from error
        if not isfinite(term) or term == 0.0:
            raise OverflowError("Periodic exact image coordinate has no finite dyadic expansion.")
        values.append(term)
        remainder -= Fraction.from_float(term)
    values.sort(key=abs)
    for previous, current in pairwise(values):
        gap = abs(current) - nextafter(abs(current), 0.0)
        if abs(previous) > 0.5 * gap:
            raise OverflowError("Periodic exact image coordinate lacks a nonoverlapping expansion.")
    return tuple(values)


def _image_radius_neighbor_csr(
    image_owners: np.ndarray,
    image_windings: np.ndarray,
    coordinate_offsets: np.ndarray,
    coordinate_components: np.ndarray,
    influence_radius: float,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a complete local superset of exact power neighbors in the carrier.

    A site's own periodic copies bound every same-site cell by the lattice
    covering radius. The site-weight range expands the bound for any competing
    image. Any two cells sharing a power face inside the carrier are therefore
    within ``2 * influence_radius``. Exact dyadic buckets enumerate that radius
    neighborhood without a global image-pair gather.
    """
    owners = np.asarray(image_owners, dtype=np.int64)
    windings = np.asarray(image_windings, dtype=np.int64)
    offsets = np.asarray(coordinate_offsets, dtype=np.int64)
    components = np.asarray(coordinate_components, dtype=np.float64)
    if (
        owners.ndim != 1
        or windings.shape != (owners.size, 3)
        or offsets.shape != (3 * owners.size + 1,)
        or offsets[0] != 0
        or offsets[-1] != components.size
        or np.any(np.diff(offsets) < 0)
        or np.any(~np.isfinite(components))
    ):
        raise ValueError("Periodic image coordinates do not match their owner/winding rows.")
    diameter = 2 * Fraction.from_float(float(influence_radius))
    if diameter <= 0:
        raise ValueError("Periodic power neighbor radius must be positive.")
    coordinates: list[tuple[Fraction, Fraction, Fraction]] = []
    buckets: dict[tuple[int, int, int], list[int]] = {}
    for image in range(owners.size):
        coordinate = []
        for axis in range(3):
            position = 3 * image + axis
            first, last = map(int, offsets[position : position + 2])
            coordinate.append(
                sum(
                    (Fraction.from_float(float(value)) for value in components[first:last]),
                    Fraction(0),
                )
            )
        point = (coordinate[0], coordinate[1], coordinate[2])
        key = (
            floor(point[0] / diameter),
            floor(point[1] / diameter),
            floor(point[2] / diameter),
        )
        coordinates.append(point)
        buckets.setdefault(key, []).append(image)

    pairs: set[tuple[int, int]] = set()
    diameter_squared = diameter * diameter
    neighbor_offsets = tuple(product((-1, 0, 1), repeat=3))
    for first, point in enumerate(coordinates):
        bucket = (
            floor(point[0] / diameter),
            floor(point[1] / diameter),
            floor(point[2] / diameter),
        )
        for neighbor_offset in neighbor_offsets:
            nearby = tuple(
                bucket[axis] + neighbor_offset[axis] for axis in range(3)
            )
            for second in buckets.get(nearby, ()):
                if second <= first:
                    continue
                distance_squared = sum(
                    (
                        (point[axis] - coordinates[second][axis]) ** 2
                        for axis in range(3)
                    ),
                    Fraction(0),
                )
                if distance_squared <= diameter_squared:
                    pairs.add((first, second))
    if not pairs:
        raise ValueError("Periodic image radius search found no exact power-neighbor edges.")
    undirected = np.asarray(sorted(pairs), dtype=np.int64).reshape((-1, 2))
    directed = np.concatenate((undirected, undirected[:, ::-1]), axis=0)
    directed = directed[np.lexsort((directed[:, 1], directed[:, 0]))]
    counts = np.bincount(directed[:, 0], minlength=owners.size)
    offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
    return offsets, directed[:, 1].astype(np.int32)


def _box_tetrahedra(
    lower: np.ndarray, upper: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(
        tuple(
            (upper[0] if x else lower[0], upper[1] if y else lower[1], upper[2] if z else lower[2])
            for x, y, z in product((0, 1), repeat=3)
        ),
        dtype=np.float64,
    )
    vertex = {bits: row for row, bits in enumerate(product((0, 1), repeat=3))}
    tetrahedra = []
    for axes in permutations(range(3)):
        bits = [0, 0, 0]
        path = [vertex[tuple(bits)]]
        for axis in axes:
            bits[axis] = 1
            path.append(vertex[tuple(bits)])
        inversions = sum(axes[i] > axes[j] for i in range(3) for j in range(i + 1, 3))
        if inversions % 2:
            path[1], path[2] = path[2], path[1]
        tetrahedra.append(path)
    cells = np.asarray(tetrahedra, dtype=np.int32)
    occurrences: dict[tuple[int, ...], list[tuple[int, int]]] = {}
    for cell, vertices in enumerate(cells):
        for opposite in range(4):
            face = tuple(
                sorted(int(vertices[corner]) for corner in range(4) if corner != opposite)
            )
            occurrences.setdefault(face, []).append((cell, opposite))
    facets = np.full(cells.shape, -1, dtype=np.int32)
    boundary = []
    for face, adjacent in occurrences.items():
        if len(adjacent) == 1:
            boundary.append((face, adjacent[0]))
        elif len(adjacent) != 2:
            raise ValueError("Periodic power carrier has a nonmanifold tetrahedral facet.")
    for facet, (_, (cell, opposite)) in enumerate(sorted(boundary)):
        facets[cell, opposite] = facet
    return points, cells, facets


def _base_image_cell_rows(
    restricted: meshcore.RestrictedPowerCells,
    base_images: dict[int, int],
    site_count: int,
    /,
) -> dict[int, int]:
    cell_sites = np.asarray(restricted.cell_sites, dtype=np.int64)
    result: dict[int, int] = {}
    for point_row in range(site_count):
        rows = np.flatnonzero(cell_sites == base_images[point_row])
        if rows.size != 1:
            raise ValueError("Every authored periodic site requires one connected full power cell.")
        result[point_row] = int(rows[0])
    return result


def _base_cells_are_unclipped(
    restricted: meshcore.RestrictedPowerCells, cell_rows: dict[int, int], /
) -> bool:
    owners = np.asarray(restricted.face_cells, dtype=np.int64)
    faces_by_cell: list[list[int]] = [
        [] for _ in range(np.asarray(restricted.cell_sites).size)
    ]
    for face in range(owners.shape[0]):
        first = int(owners[face, 0])
        second = int(owners[face, 1])
        if first >= 0:
            faces_by_cell[first].append(face)
        if second >= 0:
            faces_by_cell[second].append(face)
    return all(
        all(
            owners[face, 0] >= 0 and owners[face, 1] >= 0
            for face in faces_by_cell[row]
        )
        for row in cell_rows.values()
    )


def _polyhedral_quotient_from_power(
    source: PeriodicPointOrbits,
    source_id: str,
    source_revision: str,
    point_ids: np.ndarray,
    point_owner_ranks: np.ndarray,
    site_weights: np.ndarray,
    image_owners: np.ndarray,
    image_windings: np.ndarray,
    image_coordinate_offsets: np.ndarray,
    image_coordinate_components: np.ndarray,
    carrier_points: np.ndarray,
    restricted: meshcore.RestrictedPowerCells,
    cell_rows: dict[int, int],
    /,
) -> tuple[
    CellMesh,
    tuple[tuple[Fraction, Fraction, Fraction], ...],
    tuple[tuple[Fraction, Fraction, Fraction, Fraction], ...],
    np.ndarray,
    tuple[tuple[Fraction, Fraction, Fraction], ...],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    points = np.asarray(restricted.vertices, dtype=np.float64)
    offsets = np.asarray(restricted.face_offsets, dtype=np.int64)
    face_vertices = np.asarray(restricted.face_vertices, dtype=np.int64)
    face_cells = np.asarray(restricted.face_cells, dtype=np.int64)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or offsets.shape != (face_cells.shape[0] + 1,)
        or offsets[0] != 0
        or offsets[-1] != face_vertices.size
        or face_cells.shape != (np.asarray(restricted.face_facets).size, 2)
        or np.any((face_vertices < 0) | (face_vertices >= points.shape[0]))
    ):
        raise ValueError("Exact periodic power output has malformed face topology.")
    incident: list[list[int]] = [[] for _ in range(np.asarray(restricted.cell_sites).size)]
    for face in range(face_cells.shape[0]):
        first = int(face_cells[face, 0])
        second = int(face_cells[face, 1])
        if first >= 0:
            incident[first].append(face)
        if second >= 0:
            incident[second].append(face)
    loops_by_site: dict[int, tuple[np.ndarray, ...]] = {}
    used_vertices: set[int] = set()
    for point_row, cell_row in cell_rows.items():
        loops = []
        for face in incident[cell_row]:
            first, second = map(int, face_cells[face])
            if first == cell_row:
                loop = face_vertices[offsets[face] : offsets[face + 1]]
            elif second == cell_row:
                loop = face_vertices[offsets[face] : offsets[face + 1]][::-1]
            else:
                raise ValueError("Periodic power face incidence lost its owning cell.")
            if loop.size < 3 or np.unique(loop).size != loop.size:
                raise ValueError("Periodic power output contains a degenerate cell face.")
            loops.append(np.asarray(loop, dtype=np.int64))
            used_vertices.update(map(int, loop))
        if not loops:
            raise ValueError("An authored periodic power cell has no faces.")
        loops_by_site[point_row] = tuple(loops)

    orders = identification_orders(source.cell)
    coordinate_offsets = np.asarray(image_coordinate_offsets, dtype=np.int64)
    coordinate_components = np.asarray(
        image_coordinate_components, dtype=np.float64
    )
    exact_image_points: list[tuple[Fraction, Fraction, Fraction]] = []
    for image in range(image_owners.size):
        image_coordinates = []
        for axis in range(3):
            coordinate = 3 * image + axis
            start, stop = map(int, coordinate_offsets[coordinate : coordinate + 2])
            image_coordinates.append(
                sum(
                    (
                        Fraction.from_float(float(value))
                        for value in coordinate_components[start:stop]
                    ),
                    Fraction(0),
                )
            )
        exact_point_row: tuple[Fraction, Fraction, Fraction] = (
            image_coordinates[0],
            image_coordinates[1],
            image_coordinates[2],
        )
        exact_image_points.append(exact_point_row)
    exact_image_point_bank: tuple[
        tuple[Fraction, Fraction, Fraction], ...
    ] = tuple(exact_image_points)
    exact_image_weights = tuple(
        Fraction.from_float(float(site_weights[int(owner)]))
        for owner in image_owners
    )
    prepared_exact_image_bank = _prepare_power_site_bank(
        exact_image_point_bank, exact_image_weights
    )
    exact_carrier_points: tuple[
        tuple[Fraction, Fraction, Fraction], ...
    ] = tuple(
        (
            Fraction.from_float(float(row[0])),
            Fraction.from_float(float(row[1])),
            Fraction.from_float(float(row[2])),
        )
        for row in carrier_points
    )
    carrier_bank = np.asarray(restricted.vertex_carriers, dtype=np.int64)
    identity_by_vertex: dict[int, _PowerVertexKey] = {}
    exact_point_by_vertex: dict[int, tuple[Fraction, Fraction, Fraction]] = {}
    carrier_proof_by_vertex: dict[int, _CarrierProof] = {}
    carrier_barycentric_by_vertex: dict[
        int, tuple[Fraction, Fraction, Fraction, Fraction]
    ] = {}
    # Authored sites whose unshifted (owned) cell meets each native vertex.
    base_site_ids_by_vertex: dict[int, frozenset[int]] = {}
    for vertex in sorted(used_vertices):
        start, stop = map(int, np.asarray(restricted.vertex_site_offsets)[vertex : vertex + 2])
        witnesses = np.asarray(restricted.vertex_sites, dtype=np.int64)[start:stop]
        if witnesses.size == 0 or np.any((witnesses < 0) | (witnesses >= image_owners.size)):
            raise ValueError("Periodic power vertex lacks exact active image-site ancestry.")
        equal_images = tuple(int(image) for image in witnesses)
        if tuple(sorted(set(equal_images))) != equal_images:
            raise ValueError("Periodic power image-site witnesses must be sorted and unique.")
        carrier_ids = tuple(int(value) for value in carrier_bank[vertex] if value >= 0)
        if not carrier_ids or any(index >= len(exact_carrier_points) for index in carrier_ids):
            raise ValueError("Periodic power vertex lacks exact carrier-point ancestry.")
        carrier_coordinates = tuple(
            exact_carrier_points[index] for index in carrier_ids
        )
        try:
            exact_point, barycentric = _vertex_power_point(
                points[vertex],
                exact_image_point_bank,
                exact_image_weights,
                equal_images,
                carrier_coordinates,
                prepared_site_bank=prepared_exact_image_bank,
            )
        except ValueError as error:
            active_site_ids = tuple(
                int(point_ids[int(image_owners[image])]) for image in equal_images
            )
            active_windings = tuple(
                tuple(int(value) for value in image_windings[image])
                for image in equal_images
            )
            active_points = tuple(
                exact_image_point_bank[image] for image in equal_images
            )
            active_weights = tuple(
                exact_image_weights[image] for image in equal_images
            )
            raise ValueError(
                "Exact periodic power carrier intersection failed for one native vertex: "
                f"source_orbits={source.orbits_id}; cell={source.cell.cell_id}; "
                f"native_row={vertex}; rounded={points[vertex].tolist()!r}; "
                f"image_rows={equal_images!r}; source_site_ids={active_site_ids!r}; "
                f"image_windings={active_windings!r}; "
                f"exact_image_points={active_points!r}; exact_weights={active_weights!r}; "
                f"carrier_ids={carrier_ids!r}; carrier_points={carrier_coordinates!r}; "
                f"helper_failure={error!r}"
            ) from error
        exact_point_by_vertex[vertex] = exact_point
        carrier_barycentric_by_vertex[vertex] = barycentric
        carrier_proof: _CarrierProof = tuple(
            (
                carrier_id,
                (
                    barycentric[index].numerator,
                    barycentric[index].denominator,
                ),
            )
            for index, carrier_id in enumerate(carrier_ids)
        )
        carrier_proof_by_vertex[vertex] = carrier_proof
        candidates = []
        for anchor_row in witnesses:
            anchor = image_windings[int(anchor_row)]
            relative = reduce_periodic_exponents(
                image_windings[witnesses] - anchor[None, :], orders
            )
            site_signature = tuple(
                sorted(
                    (
                        int(point_ids[image_owners[int(image)]]),
                        tuple(int(value) for value in winding),
                    )
                    for image, winding in zip(witnesses, relative, strict=True)
                )
            )
            inverse = exact_periodic_element(
                source.cell, -np.asarray(anchor, dtype=np.int64)
            )
            normalized_point = tuple(
                sum(
                    (
                        inverse[axis][component] * exact_point[component]
                        for component in range(3)
                    ),
                    Fraction(0),
                )
                + inverse[axis][3]
                for axis in range(3)
            )
            coordinate_signature = tuple(
                (value.numerator, value.denominator) for value in normalized_point
            )
            candidates.append(
                (
                    (site_signature, coordinate_signature),
                    tuple(int(value) for value in anchor),
                )
            )
        key = min(candidates)
        identity_by_vertex[vertex] = key
        base_site_ids_by_vertex[vertex] = frozenset(
            int(point_ids[image_owners[int(image)]])
            for image in witnesses
            if not np.any(image_windings[int(image)])
        )


    loops_by_site = {
        point_row: tuple(
            _convex_power_face_loop(loop, exact_point_by_vertex) for loop in loops
        )
        for point_row, loops in loops_by_site.items()
    }
    used_vertices = {
        int(vertex) for loops in loops_by_site.values() for loop in loops for vertex in loop
    }
    identity_by_vertex = {
        vertex: key for vertex, key in identity_by_vertex.items() if vertex in used_vertices
    }

    vertex_keys = tuple(sorted(set(identity_by_vertex.values())))
    vertex_row_by_key = {key: row for row, key in enumerate(vertex_keys)}
    vertex_coordinates = np.empty((len(vertex_keys), 3), dtype=np.float64)
    raw_coordinate_by_key: dict[_PowerVertexKey, tuple[int, np.ndarray]] = {}

    def native_witness(vertex: int, /) -> tuple[object, ...]:
        offsets = np.asarray(restricted.vertex_site_offsets, dtype=np.int64)
        start, stop = map(int, offsets[vertex : vertex + 2])
        image_rows = np.asarray(restricted.vertex_sites, dtype=np.int64)[start:stop]
        site_ids = tuple(
            int(point_ids[int(image_owners[int(image)])]) for image in image_rows
        )
        windings = tuple(
            tuple(int(value) for value in image_windings[int(image)])
            for image in image_rows
        )
        carriers = tuple(
            int(value)
            for value in np.asarray(restricted.vertex_carriers, dtype=np.int64)[vertex]
            if int(value) >= 0
        )
        return tuple(int(image) for image in image_rows), site_ids, windings, carriers

    for vertex, key in identity_by_vertex.items():
        coordinate = points[vertex]
        previous = raw_coordinate_by_key.get(key)
        if previous is not None and not np.array_equal(previous[1], coordinate):
            previous_vertex, previous_coordinate = previous
            raise ValueError(
                "One exact periodic power vertex key has inconsistent native coordinates: "
                f"key={key!r}; rows=({previous_vertex}, {vertex}); "
                f"coordinates=({previous_coordinate.tolist()!r}, {coordinate.tolist()!r}); "
                f"witnesses=({native_witness(previous_vertex)!r}, {native_witness(vertex)!r})."
            )
        raw_coordinate_by_key[key] = (vertex, coordinate)
    for key, row in vertex_row_by_key.items():
        vertex_coordinates[row] = raw_coordinate_by_key[key][1]

    representative_key: dict[
        _PowerPointSignature, tuple[tuple[int, ...], _PowerVertexKey]
    ] = {}
    for key in vertex_keys:
        signature, winding = key
        previous = representative_key.get(signature)
        if previous is None or winding < previous[0]:
            representative_key[signature] = (winding, key)
    representatives = np.empty((len(vertex_keys),), dtype=np.int32)
    shifts = np.empty((len(vertex_keys), 3), dtype=np.int64)
    for key, row in vertex_row_by_key.items():
        winding, representative = representative_key[key[0]]
        representatives[row] = vertex_row_by_key[representative]
        shifts[row] = reduce_periodic_exponents(
            np.asarray(key[1], dtype=np.int64) - np.asarray(winding, dtype=np.int64),
            orders,
        )

    root_signatures = tuple(sorted({signature for signature, _ in vertex_keys}))
    root_identity_text = tuple(
        canonical_fingerprint(
            {
                "source_id": source_id,
                "source_revision": source_revision,
                "source_cell": source.cell.cell_id,
                "kind": "periodic-power-quotient-vertex",
                "signature": signature,
            }
        )
        for signature in root_signatures
    )
    root_ids = _numeric_entity_ids(
        "periodic-power-quotient-vertex",
        root_identity_text,
        tuple(() for _ in root_identity_text),
    )
    root_id_by_signature = dict(zip(root_signatures, root_ids, strict=True))
    owner_by_site_id = {
        int(identifier): int(owner)
        for identifier, owner in zip(
            point_ids, point_owner_ranks, strict=True
        )
    }
    vertex_source_ids = np.asarray(
        [root_id_by_signature[signature] for signature, _ in vertex_keys],
        dtype=np.int64,
    )
    # A lifted vertex image is owned by the owner of the smallest authored site
    # whose unshifted cell contains that exact image. Every rank resident on the
    # image reaches it through one of its own unshifted cells, and the chosen
    # owner's unshifted cell holds the same lifted ID, so owner directories can
    # resolve every resident image without a global vertex table.
    base_site_ids_by_key: dict[_PowerVertexKey, frozenset[int]] = {}
    for vertex, key in identity_by_vertex.items():
        base_sites = base_site_ids_by_vertex[vertex]
        previous_sites = base_site_ids_by_key.setdefault(key, base_sites)
        if previous_sites != base_sites or not base_sites:
            raise ValueError(
                "A lifted periodic power vertex lacks one consistent unshifted incident cell."
            )
    vertex_owner_ranks = np.asarray(
        [owner_by_site_id[min(base_site_ids_by_key[key])] for key in vertex_keys],
        dtype=np.int32,
    )
    vertex_absolute_windings = np.asarray(
        [winding for _, winding in vertex_keys], dtype=np.int64
    ).reshape((-1, 3))
    vertex_identity_text = tuple(
        canonical_fingerprint(
            {
                "source_id": source_id,
                "source_revision": source_revision,
                "source_cell": source.cell.cell_id,
                "kind": "periodic-power-vertex",
                "signature": signature,
                "winding": winding,
            }
        )
        for signature, winding in vertex_keys
    )
    vertex_ids = np.asarray(
        _numeric_entity_ids(
            "periodic-power-lifted-vertex",
            vertex_identity_text,
            tuple(() for _ in vertex_identity_text),
        ),
        dtype=np.int64,
    )
    selected_point_rows = tuple(sorted(cell_rows))
    polyhedral_cells: list[tuple[np.ndarray, ...]] = []
    for point_row in selected_point_rows:
        loops = []
        for loop in loops_by_site[point_row]:
            loops.append(
                np.asarray(
                    [
                        vertex_row_by_key[identity_by_vertex[int(vertex)]]
                        for vertex in loop
                    ],
                    dtype=np.int32,
                )
            )
        polyhedral_cells.append(tuple(loops))
    selected_cell_ids = np.asarray(
        [point_ids[point_row] for point_row in selected_point_rows],
        dtype=np.int64,
    )
    lifted = CellMesh.from_polyhedra(
        vertex_coordinates,
        tuple(polyhedral_cells),
        block_name="periodic-power",
        vertex_global_ids=vertex_ids,
        cell_global_ids=selected_cell_ids,
    )
    cell_ids = np.asarray(lifted.entity_set(3).entity_ids, dtype=np.int64)
    if not np.array_equal(np.sort(cell_ids), np.sort(selected_cell_ids)):
        raise ValueError("Periodic power quotient cells do not retain authored point identities.")
    selected_vertex_rows = np.asarray(
        [raw_coordinate_by_key[key][0] for key in vertex_keys], dtype=np.int64
    )
    carrier_barycentric_bank = tuple(
        carrier_barycentric_by_vertex[int(vertex)] for vertex in selected_vertex_rows
    )
    exact_vertex_bank = tuple(
        exact_point_by_vertex[int(vertex)] for vertex in selected_vertex_rows
    )
    return (
        lifted,
        exact_vertex_bank,
        carrier_barycentric_bank,
        selected_vertex_rows,
        exact_image_point_bank,
        vertex_source_ids,
        vertex_owner_ranks,
        vertex_absolute_windings,
        representatives,
        shifts,
    )

def _bind_power_periodic_topology(
    lifted: CellMesh,
    cell: PeriodicPointOrbits,
    representatives: np.ndarray,
    shifts: np.ndarray,
    exact_coordinate_source: ExactPowerCellGeometrySource,
    selected_cell_ids: np.ndarray,
    /,
) -> CellMesh:
    connectivity = lifted.connectivity
    if not isinstance(connectivity, PolyhedralConnectivity):
        raise TypeError("Periodic power quotient construction requires polyhedral connectivity.")
    topology = PeriodicMeshTopology(
        lifted,
        cell.cell,
        representatives,
        shifts,
        exact_coordinate_source=exact_coordinate_source,
    )
    entity_ids = {
        degree: np.asarray(lifted.entity_set(degree).entity_ids, dtype=np.int64)
        for degree in range(4)
    }
    bound = CellMesh(
        lifted.coordinates,
        lifted.blocks,
        vertex_global_ids=lifted.vertex_global_ids,
        entity_global_ids=entity_ids,
        polyhedral_connectivity=connectivity,
        numeric_version=lifted.numeric_version,
        periodic_topology=topology,
    )
    topology.require_lifted(bound.topology, bound.lifted_topology_id, bound.blocks)
    topology.require_coordinates(bound.coordinates)
    cell_ids = np.asarray(bound.entity_set(3).entity_ids, dtype=np.int64)
    quotient_cell_ids = np.asarray(
        topology.quotient.entity_sets[3].entity_ids, dtype=np.int64
    )
    expected_cell_ids = np.asarray(selected_cell_ids, dtype=np.int64)
    if (
        not np.array_equal(np.sort(cell_ids), np.sort(expected_cell_ids))
        or not np.array_equal(np.sort(quotient_cell_ids), np.sort(expected_cell_ids))
    ):
        raise ValueError("Periodic power quotient cells do not retain authored point identities.")
    return bound


def _validate_periodic_power_measure(
    mesh: CellMesh,
    restricted: meshcore.RestrictedPowerCells,
    cell_rows: dict[int, int],
    site_count: int,
    lattice: np.ndarray,
    /,
) -> None:
    rows = np.asarray(tuple(cell_rows[index] for index in range(site_count)), dtype=np.int64)
    volumes = np.asarray(restricted.cell_volumes, dtype=np.float64)[rows]
    total = float(np.sum(volumes, dtype=np.float64))
    expected = abs(float(np.linalg.det(lattice)))
    tolerance = 1.0e-9 * expected
    if (
        volumes.shape != (site_count,)
        or np.any(~np.isfinite(volumes))
        or np.any(volumes <= 0.0)
        or not isfinite(expected)
        or expected <= 0.0
        or abs(total - expected) > tolerance
        or mesh.entity_set(3).entity_ids.size != site_count
    ):
        raise ValueError("Periodic power cells do not conserve one fundamental-cell volume.")



def _owner_local_power_source_bank(
    source: NativePeriodicSource,
    rank: int,
    execution_group: ExecutionGroup,
    source_manifest_id: str,
    axis_name: str,
    /,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    str,
    int,
    bool,
]:
    from phydrax.meshing._distribution import (
        _owner_exchange_message_banks,
        _owner_group_fail_if_any,
    )
    from phydrax.meshing.providers._native_periodic import (
        _owner_local_source_bank,
        NativePeriodicSource,
    )

    if (
        not isinstance(source, NativePeriodicSource)
        or source.point_owner_ranks is None
        or not isinstance(source.domain, PeriodicPointOrbits)
    ):
        raise TypeError("Owner-local power requires typed periodic point ownership.")
    points, point_ids, point_owners, bucket_ids, bucket_owners, directory_id, total = (
        _owner_local_source_bank(
            source.domain.points,
            source.domain.point_ids,
            source.point_owner_ranks,
            np.asarray(source.point_owner_ranks, dtype=np.int32) == rank,
            rank,
            execution_group,
            source_manifest_id,
            axis_name,
        )
    )
    weights_present = source.site_weights is not None
    if weights_present:
        raw_weights = np.asarray(source.site_weights, dtype=np.float64)
        if (
            raw_weights.shape != np.asarray(source.domain.point_ids).shape
            or np.any(~np.isfinite(raw_weights))
        ):
            raise ValueError("Owner-local power weights do not align with authored sites.")
        weight_by_id = {
            int(identifier): float(weight)
            for identifier, weight in zip(
                np.asarray(source.domain.point_ids, dtype=np.int64),
                raw_weights,
                strict=True,
            )
        }
        weights = np.asarray(
            [weight_by_id[int(identifier)] for identifier in point_ids],
            dtype=np.float64,
        )
    else:
        weights = np.zeros((point_ids.size,), dtype=np.float64)

    parts = len(execution_group.spec.process_indices)
    owned = point_owners == rank
    outgoing_requests = [
        point_ids[(point_owners == owner) & (owner != rank), None]
        if np.any((point_owners == owner) & (owner != rank))
        else np.zeros((0, 1), dtype=np.int64)
        for owner in range(parts)
    ]
    incoming_requests = _owner_exchange_message_banks(
        tuple(outgoing_requests), rank, execution_group, axis_name
    )
    owned_rows = {
        int(identifier): row
        for row, identifier in enumerate(point_ids)
        if owned[row]
    }
    outgoing_weights = [np.zeros((0, 1), dtype=np.float64) for _ in range(parts)]
    local_error = None
    for requester, requests in incoming_requests.items():
        identifiers = np.asarray(requests, dtype=np.int64)
        if identifiers.ndim != 2 or identifiers.shape[1] != 1:
            local_error = "Owner-local power weight requests are malformed."
            continue
        positions = [owned_rows.get(int(identifier), -1) for identifier in identifiers[:, 0]]
        if any(position < 0 for position in positions):
            local_error = "A power source owner lacks a requested authored weight."
            continue
        outgoing_weights[requester] = weights[
            np.asarray(positions, dtype=np.int64), None
        ]
    _owner_group_fail_if_any(
        local_error, execution_group, axis_name, "periodic power weight request"
    )
    incoming_weights = _owner_exchange_message_banks(
        tuple(outgoing_weights), rank, execution_group, axis_name
    )
    local_rows = {
        int(identifier): row for row, identifier in enumerate(point_ids)
    }
    local_error = None
    for owner, values in incoming_weights.items():
        owner_weights = np.asarray(values, dtype=np.float64)
        requested_ids = np.asarray(outgoing_requests[owner], dtype=np.int64)[:, 0]
        if owner_weights.shape != (requested_ids.size, 1):
            local_error = "Periodic power weight replies do not match their requests."
            continue
        for identifier, weight in zip(
            requested_ids, owner_weights[:, 0], strict=True
        ):
            row = local_rows[int(identifier)]
            if weights[row] != weight:
                local_error = "A periodic power ghost weight differs from its exact owner."
                continue
            weights[row] = weight
    _owner_group_fail_if_any(
        local_error, execution_group, axis_name, "periodic power weights"
    )
    return (
        points,
        point_ids,
        point_owners,
        owned,
        weights,
        bucket_ids,
        bucket_owners,
        directory_id,
        total,
        weights_present,
    )


def _owner_local_power_bounds(
    points: np.ndarray,
    weights: np.ndarray,
    owned: np.ndarray,
    origin: np.ndarray,
    lattice: np.ndarray,
    execution_group: ExecutionGroup,
    axis_name: str,
    /,
) -> tuple[
    tuple[tuple[np.ndarray, np.ndarray] | None, ...],
    np.ndarray,
    np.ndarray,
    float,
    float,
]:
    from phydrax.meshing._distribution import (
        _owner_group_allgather,
        _periodic_fractional_point_bounds,
    )

    dimension = origin.size
    rows = np.flatnonzero(owned)
    local_record = np.zeros((3 + 4 * dimension,), dtype=np.float64)
    if rows.size:
        point_values = points[rows]
        point_lower, point_upper = _periodic_fractional_point_bounds(
            point_values, origin, lattice
        )
        local_record[0] = 1.0
        local_record[1] = float(np.min(weights[rows]))
        local_record[2] = float(np.max(weights[rows]))
        local_record[3 : 3 + dimension] = np.min(point_lower, axis=0)
        local_record[3 + dimension : 3 + 2 * dimension] = np.max(
            point_upper, axis=0
        )
        local_record[3 + 2 * dimension : 3 + 3 * dimension] = np.min(
            point_values, axis=0
        )
        local_record[3 + 3 * dimension :] = np.max(point_values, axis=0)
    records = _owner_group_allgather(local_record, execution_group, axis_name)
    bounds: list[tuple[np.ndarray, np.ndarray] | None] = []
    nonempty = []
    for row in records:
        flag = row[0]
        if flag not in (0.0, 1.0):
            raise ValueError("Collective power owner-bound flags are malformed.")
        if flag == 0.0:
            bounds.append(None)
            continue
        lower = np.asarray(row[3 : 3 + dimension], dtype=np.float64)
        upper = np.asarray(
            row[3 + dimension : 3 + 2 * dimension], dtype=np.float64
        )
        if (
            np.any(~np.isfinite(lower) | ~np.isfinite(upper))
            or np.any(lower > upper)
        ):
            raise ValueError("Collective power owner fractional bounds are malformed.")
        bounds.append((lower, upper))
        nonempty.append(row)
    if not nonempty:
        raise ValueError("Owner-local power generation has no authored source sites.")
    global_lower = np.min(
        np.asarray(
            [row[3 + 2 * dimension : 3 + 3 * dimension] for row in nonempty],
            dtype=np.float64,
        ),
        axis=0,
    )
    global_upper = np.max(
        np.asarray(
            [row[3 + 3 * dimension :] for row in nonempty], dtype=np.float64
        ),
        axis=0,
    )
    global_weight_min = min(float(row[1]) for row in nonempty)
    global_weight_max = max(float(row[2]) for row in nonempty)
    return (
        tuple(bounds),
        global_lower,
        global_upper,
        global_weight_min,
        global_weight_max,
    )


def _owner_local_power_support_exchange(
    points: np.ndarray,
    point_ids: np.ndarray,
    point_owners: np.ndarray,
    weights: np.ndarray,
    owned: np.ndarray,
    origin: np.ndarray,
    lattice: np.ndarray,
    support_lower: np.ndarray,
    support_upper: np.ndarray,
    owner_bounds: tuple[tuple[np.ndarray, np.ndarray] | None, ...],
    bucket_ids: np.ndarray,
    bucket_owners: np.ndarray,
    rank: int,
    execution_group: ExecutionGroup,
    source_manifest_id: str,
    axis_name: str,
    /,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[str, ...],
]:
    from phydrax.meshing._distribution import (
        _owner_directory_lookup,
        _owner_exchange_message_banks,
        _owner_group_allgather,
        _owner_group_fail_if_any,
        _periodic_fractional_point_bounds,
        _periodic_support_owner_routes,
        _periodic_support_point_rows,
    )

    dimension = points.shape[1]
    parts = len(owner_bounds)
    routes = _periodic_support_owner_routes(
        support_lower, support_upper, owner_bounds, (True,) * dimension
    )
    outgoing_boxes = [
        np.zeros((0, 2 * dimension), dtype=np.float64) for _ in range(parts)
    ]
    for owner, selected in enumerate(routes):
        if owner == rank or selected.size == 0:
            continue
        outgoing_boxes[owner] = np.concatenate(
            (support_lower[selected], support_upper[selected]), axis=1
        )
    incoming_boxes = _owner_exchange_message_banks(
        tuple(outgoing_boxes), rank, execution_group, axis_name
    )
    owned_rows = np.flatnonzero(owned)
    owned_points = points[owned_rows]
    point_lower, point_upper = _periodic_fractional_point_bounds(
        owned_points, origin, lattice
    )
    response_ids = [np.zeros((0, 1), dtype=np.int64) for _ in range(parts)]
    response_points = [
        np.zeros((0, dimension), dtype=np.float64) for _ in range(parts)
    ]
    response_weights = [np.zeros((0, 1), dtype=np.float64) for _ in range(parts)]
    local_error = None
    for requester, values in incoming_boxes.items():
        boxes = np.asarray(values, dtype=np.float64)
        if boxes.ndim != 2 or boxes.shape[1] != 2 * dimension:
            local_error = "Owner-local power support boxes are malformed."
            continue
        lower = boxes[:, :dimension]
        upper = boxes[:, dimension:]
        selected_rows = _periodic_support_point_rows(
            lower,
            upper,
            point_lower,
            point_upper,
            (True,) * dimension,
        )
        source_rows = owned_rows[selected_rows]
        order = np.argsort(point_ids[source_rows], kind="stable")
        source_rows = source_rows[order]
        response_ids[requester] = point_ids[source_rows, None]
        response_points[requester] = points[source_rows]
        response_weights[requester] = weights[source_rows, None]
    _owner_group_fail_if_any(
        local_error, execution_group, axis_name, "periodic power support query"
    )
    incoming_ids = _owner_exchange_message_banks(
        tuple(response_ids), rank, execution_group, axis_name
    )
    incoming_points = _owner_exchange_message_banks(
        tuple(response_points), rank, execution_group, axis_name
    )
    incoming_weights = _owner_exchange_message_banks(
        tuple(response_weights), rank, execution_group, axis_name
    )
    local_error = (
        None
        if set(incoming_ids) == set(incoming_points) == set(incoming_weights)
        else "Periodic power support reply channels disagree."
    )
    _owner_group_fail_if_any(
        local_error, execution_group, axis_name, "periodic power support channels"
    )
    offers = tuple(
        (
            owner,
            np.asarray(incoming_ids[owner], dtype=np.int64).reshape(-1),
            np.asarray(incoming_points[owner], dtype=np.float64),
            np.asarray(incoming_weights[owner], dtype=np.float64).reshape(-1),
        )
        for owner in sorted(incoming_ids)
    )
    offered_ids = (
        np.concatenate(tuple(ids for _, ids, _, _ in offers))
        if offers
        else np.zeros((0,), dtype=np.int64)
    )
    offered_owners = (
        np.concatenate(
            tuple(np.full((ids.size,), owner, dtype=np.int32) for owner, ids, _, _ in offers)
        )
        if offers
        else np.zeros((0,), dtype=np.int32)
    )
    resolved = _owner_directory_lookup(
        offered_ids,
        bucket_ids,
        bucket_owners,
        rank,
        parts,
        execution_group,
        axis_name,
        "source-points",
    )
    local_error = (
        None
        if np.array_equal(resolved, offered_owners)
        else "Periodic power support replies disagree with source ownership."
    )
    _owner_group_fail_if_any(
        local_error, execution_group, axis_name, "periodic power support ownership"
    )
    local_rows = {int(identifier): row for row, identifier in enumerate(point_ids)}
    new_records: dict[int, tuple[int, np.ndarray, float]] = {}
    local_error = None
    for owner, identifiers, coordinates, weights_in in offers:
        if (
            coordinates.shape != (identifiers.size, dimension)
            or weights_in.shape != identifiers.shape
        ):
            local_error = "Periodic power support replies are misaligned."
            continue
        for identifier, coordinate, weight in zip(
            identifiers, coordinates, weights_in, strict=True
        ):
            coordinate_array = np.asarray(coordinate, dtype=np.float64).reshape(
                (dimension,)
            )
            row = local_rows.get(int(identifier))
            if row is not None:
                if (
                    point_owners[row] != owner
                    or not np.array_equal(
                        points[row].view(np.uint64),
                        coordinate_array.view(np.uint64),
                    )
                    or not np.array_equal(
                        np.asarray((weights[row],), dtype=np.float64).view(np.uint64),
                        np.asarray((weight,), dtype=np.float64).view(np.uint64),
                    )
                ):
                    local_error = (
                        "Periodic power support data differs from its exact owner."
                    )
            else:
                previous = new_records.get(int(identifier))
                if previous is not None and (
                    previous[0] != owner
                    or not np.array_equal(
                        previous[1].view(np.uint64),
                        coordinate_array.view(np.uint64),
                    )
                    or not np.array_equal(
                        np.asarray((previous[2],), dtype=np.float64).view(np.uint64),
                        np.asarray((weight,), dtype=np.float64).view(np.uint64),
                    )
                ):
                    local_error = "Periodic power support returned conflicting source rows."
                else:
                    new_records[int(identifier)] = (
                        owner,
                        coordinate_array.copy(),
                        float(weight),
                    )
    _owner_group_fail_if_any(
        local_error, execution_group, axis_name, "periodic power support merge"
    )
    new_ids = np.asarray(sorted(new_records), dtype=np.int64)
    if new_ids.size:
        points = np.concatenate(
            (points, np.asarray([new_records[int(identifier)][1] for identifier in new_ids])),
            axis=0,
        )
        point_owners = np.concatenate(
            (
                point_owners,
                np.asarray(
                    [new_records[int(identifier)][0] for identifier in new_ids],
                    dtype=np.int32,
                ),
            )
        )
        weights = np.concatenate(
            (
                weights,
                np.asarray(
                    [new_records[int(identifier)][2] for identifier in new_ids],
                    dtype=np.float64,
                ),
            )
        )
        point_ids = np.concatenate((point_ids, new_ids))
        order = np.argsort(point_ids, kind="stable")
        points, point_ids, point_owners, weights = (
            points[order],
            point_ids[order],
            point_owners[order],
            weights[order],
        )
        owned = point_owners == rank
    local_digest = canonical_fingerprint(
        {
            "source_manifest": source_manifest_id,
            "rank": rank,
            "support_boxes": (support_lower, support_upper),
            "received_ids": new_ids,
        }
    )
    round_records = _owner_group_allgather(
        np.frombuffer(bytes.fromhex(local_digest), dtype=np.uint8),
        execution_group,
        axis_name,
    )
    support_round_id = canonical_fingerprint(
        {
            "kind": "periodic-power-support-closure",
            "source_manifest": source_manifest_id,
            "rank_records": tuple(
                np.asarray(record, dtype=np.uint8).tobytes().hex()
                for record in round_records
            ),
        }
    )
    return (
        points,
        point_ids,
        point_owners,
        weights,
        owned,
        (support_round_id,),
    )

def _owner_local_periodic_site_weights_for_support(
    source: NativePeriodicSource,
    support_point_ids: ArrayLike,
    support_owner_ranks: ArrayLike,
    rank: int,
    execution_group: ExecutionGroup,
    /,
    *,
    axis_name: str,
) -> np.ndarray | None:
    import jax

    from phydrax._execution_runtime import ExecutionGroup
    from phydrax.meshing._distribution import (
        _owner_exchange_message_banks,
        _owner_group_allgather,
        _owner_group_fail_if_any,
    )
    from phydrax.meshing._periodic import PeriodicPointOrbits
    from phydrax.meshing.providers._native_periodic import NativePeriodicSource

    if not isinstance(source, NativePeriodicSource) or not isinstance(
        source.domain, PeriodicPointOrbits
    ):
        raise TypeError("Site-weight exchange requires a typed periodic source.")
    if not isinstance(execution_group, ExecutionGroup):
        raise TypeError("execution_group must be ExecutionGroup.")
    members = execution_group.spec.process_indices
    if (
        not execution_group.is_member
        or isinstance(rank, bool)
        or not isinstance(rank, (int, np.integer))
        or not 0 <= rank < len(members)
        or members[rank] != int(jax.process_index())
        or not isinstance(axis_name, str)
        or not axis_name.strip()
    ):
        raise ValueError("Site-weight exchange rank or group is invalid.")
    weights_present = source.site_weights is not None
    header = canonical_fingerprint(
        (
            source.source_id,
            source.source_revision,
            source.domain.cell.cell_id,
            source.domain.ambient_dimension,
            weights_present,
        )
    )
    header_rows = _owner_group_allgather(
        np.frombuffer(bytes.fromhex(header), dtype=np.uint8),
        execution_group,
        f"{axis_name}_schema",
    )
    if any(
        np.asarray(row, dtype=np.uint8).tobytes().hex() != header
        for row in header_rows
    ):
        raise ValueError("Periodic source weight schemas differ across ranks.")
    if not weights_present:
        return None

    local_error = None
    support_ids = np.empty((0,), dtype=np.int64)
    support_owners = np.empty((0,), dtype=np.int32)
    owned_weight_bits: dict[int, int] = {}
    local_weight_bits: dict[int, int] = {}
    try:
        raw_ids = np.asarray(source.domain.point_ids)
        raw_owners = np.asarray(source.point_owner_ranks)
        raw_weights = np.asarray(source.site_weights, dtype=np.float64)
        raw_support_ids = np.asarray(support_point_ids)
        raw_support_owners = np.asarray(support_owner_ranks)
        if (
            raw_ids.dtype.kind not in "iu"
            or raw_ids.ndim != 1
            or raw_owners.dtype.kind not in "iu"
            or raw_owners.shape != raw_ids.shape
            or raw_weights.shape != raw_ids.shape
            or np.any(~np.isfinite(raw_weights))
            or raw_support_ids.dtype.kind not in "iu"
            or raw_support_ids.ndim != 1
            or raw_support_owners.dtype.kind not in "iu"
            or raw_support_owners.shape != raw_support_ids.shape
            or np.any(raw_owners < 0)
            or np.any(raw_owners >= len(members))
            or np.any(raw_support_owners < 0)
            or np.any(raw_support_owners >= len(members))
            or (
                raw_ids.dtype.kind == "u"
                and raw_ids.size
                and int(np.max(raw_ids)) > np.iinfo(np.int64).max
            )
            or (
                raw_support_ids.dtype.kind == "u"
                and raw_support_ids.size
                and int(np.max(raw_support_ids)) > np.iinfo(np.int64).max
            )
        ):
            raise ValueError("Periodic source weights and support IDs are misaligned.")
        source_ids = np.asarray(raw_ids, dtype=np.int64)
        source_owners = np.asarray(raw_owners, dtype=np.int32)
        support_ids = np.asarray(raw_support_ids, dtype=np.int64)
        support_owners = np.asarray(raw_support_owners, dtype=np.int32)
        weight_bits = np.ascontiguousarray(raw_weights).view(np.uint64)
        if (
            np.unique(source_ids).size != source_ids.size
            or np.any(source_ids < 0)
            or np.unique(support_ids).size != support_ids.size
            or np.any(support_ids < 0)
        ):
            raise ValueError("Periodic source weight ownership rows are invalid.")
        local_weight_bits = {
            int(identifier): int(weight_bits[row])
            for row, identifier in enumerate(source_ids.tolist())
        }
        owned_weight_bits = {
            int(identifier): int(weight_bits[row])
            for row, identifier in enumerate(source_ids.tolist())
            if source_owners[row] == rank
        }
        if np.unique(
            np.asarray(list(owned_weight_bits), dtype=np.int64)
        ).size != len(owned_weight_bits):
            raise ValueError("A periodic source owner duplicated a weighted site.")
    except Exception as error:
        local_error = f"{type(error).__name__}: {error}"
    _owner_group_fail_if_any(
        local_error,
        execution_group,
        f"{axis_name}_input",
        "periodic source weight support",
    )

    request_ids_by_owner: list[list[int]] = [[] for _ in members]
    request_rows_by_owner: list[list[int]] = [[] for _ in members]
    result_bits = np.zeros((support_ids.size,), dtype=np.uint64)
    result_valid = np.zeros((support_ids.size,), dtype=np.bool_)
    local_error = None
    try:
        for row, (identifier_value, owner_value) in enumerate(
            zip(support_ids.tolist(), support_owners.tolist(), strict=True)
        ):
            identifier = int(identifier_value)
            owner = int(owner_value)
            if owner == rank:
                bit_value = owned_weight_bits.get(identifier)
                if bit_value is None:
                    raise ValueError("A periodic source owner lacks its authored site weight.")
                result_bits[row] = np.uint64(bit_value)
                result_valid[row] = True
            else:
                request_ids_by_owner[owner].append(identifier)
                request_rows_by_owner[owner].append(row)
        request_banks = tuple(
            np.asarray(ids, dtype=np.int64).reshape((-1, 1))
            for ids in request_ids_by_owner
        )
    except Exception as error:
        local_error = f"{type(error).__name__}: {error}"
        request_banks = tuple(
            np.zeros((0, 1), dtype=np.int64) for _ in members
        )
    _owner_group_fail_if_any(
        local_error,
        execution_group,
        f"{axis_name}_request_encode",
        "periodic site weight requests",
    )
    incoming_requests = _owner_exchange_message_banks(
        request_banks, rank, execution_group, f"{axis_name}_requests"
    )
    reply_banks = [
        np.zeros((0, 2), dtype=np.int64) for _ in members
    ]
    local_error = None
    try:
        for requester, request_values in incoming_requests.items():
            request_array = np.asarray(request_values, dtype=np.int64)
            if request_array.ndim != 2 or request_array.shape[1] != 1:
                raise ValueError("Periodic site weight request layout is invalid.")
            reply = np.empty((request_array.shape[0], 2), dtype=np.int64)
            for row, identifier_value in enumerate(request_array[:, 0].tolist()):
                identifier = int(identifier_value)
                if identifier not in owned_weight_bits:
                    raise ValueError("A site weight request reached a non-owner.")
                reply[row, 0] = identifier
                reply[row, 1] = np.asarray(
                    [owned_weight_bits[identifier]], dtype=np.uint64
                ).view(np.int64)[0]
            reply_banks[int(requester)] = reply
    except Exception as error:
        local_error = f"{type(error).__name__}: {error}"
    _owner_group_fail_if_any(
        local_error,
        execution_group,
        f"{axis_name}_request_service",
        "periodic site weight owner lookup",
    )
    incoming_replies = _owner_exchange_message_banks(
        tuple(reply_banks), rank, execution_group, f"{axis_name}_replies"
    )
    local_error = None
    try:
        for owner, reply_values in incoming_replies.items():
            replies = np.asarray(reply_values, dtype=np.int64)
            requested_ids = np.asarray(
                request_ids_by_owner[int(owner)], dtype=np.int64
            )
            if (
                replies.shape != (requested_ids.size, 2)
                or not np.array_equal(replies[:, 0], requested_ids)
            ):
                raise ValueError("Periodic site weight replies do not match their requests.")
            rows = np.asarray(request_rows_by_owner[int(owner)], dtype=np.int64)
            result_bits[rows] = replies[:, 1].view(np.uint64)
            result_valid[rows] = True
        if not np.all(result_valid):
            raise ValueError("Periodic site weight support is incomplete.")
        for row, identifier_value in enumerate(support_ids.tolist()):
            identifier = int(identifier_value)
            local_bit = local_weight_bits.get(identifier)
            if local_bit is not None and local_bit != int(result_bits[row]):
                raise ValueError(
                    "A periodic site weight differs from its exact source owner."
                )
    except Exception as error:
        local_error = f"{type(error).__name__}: {error}"
    _owner_group_fail_if_any(
        local_error,
        execution_group,
        f"{axis_name}_reply_decode",
        "periodic site weight support coverage",
    )
    return result_bits.view(np.float64)


def _owner_local_periodic_power_candidate(
    source: NativePeriodicSource,
    rank: int,
    execution_group: ExecutionGroup,
    source_manifest_id: str,
    /,
    *,
    axis_name: str = "periodic_owner_power",
) -> dict[str, Any]:
    from phydrax.meshing._distribution import (
        _owner_group_allgather,
        _owner_group_fail_if_any,
        _periodic_power_support_boxes,
    )
    from phydrax.meshing.providers._native_periodic import (
        NativePeriodicSource,
    )

    if (
        not isinstance(source, NativePeriodicSource)
        or source.ambient_dimension != 3
    ):
        raise TypeError("Owner-local power requires a 3D typed periodic point source.")
    domain = source.domain
    if not isinstance(domain, PeriodicPointOrbits):
        raise TypeError("Owner-local power requires authored periodic point orbits.")
    point_owner_ranks = source.point_owner_ranks
    if point_owner_ranks is None:
        raise ValueError("Owner-local power requires point owner ranks.")
    from phydrax.meshing.providers._native_periodic import (
        _owner_local_source_bank,
    )

    points, point_ids, point_owners, bucket_ids, bucket_owners, directory_id, total = (
        _owner_local_source_bank(
            domain.points,
            domain.point_ids,
            point_owner_ranks,
            np.asarray(point_owner_ranks, dtype=np.int32) == rank,
            rank,
            execution_group,
            source_manifest_id,
            axis_name,
        )
    )
    weights = _owner_local_periodic_site_weights_for_support(
        source,
        point_ids,
        point_owners,
        rank,
        execution_group,
        axis_name=f"{axis_name}_weights",
    )
    weights_present = weights is not None
    if weights is None:
        weights = np.zeros((point_ids.size,), dtype=np.float64)
    owned = point_owners == rank
    origin = np.asarray(domain.cell.origin, dtype=np.float64)
    lattice = np.ascontiguousarray(np.asarray(domain.cell.vectors, dtype=np.float64))
    owner_bounds, global_lower, global_upper, weight_min, weight_max = (
        _owner_local_power_bounds(
            points,
            weights,
            owned,
            origin,
            lattice,
            execution_group,
            axis_name,
        )
    )
    radius = _lattice_covering_radius(lattice)
    influence_radius = _upper_sqrt(
        Fraction.from_float(radius) ** 2
        + Fraction.from_float(weight_max)
        - Fraction.from_float(weight_min)
    )
    support_lower, support_upper = _periodic_power_support_boxes(
        points[owned],
        origin,
        lattice,
        2.0 * influence_radius,
    )
    (
        points,
        point_ids,
        point_owners,
        weights,
        owned,
        support_round_ids,
    ) = _owner_local_power_support_exchange(
        points,
        point_ids,
        point_owners,
        weights,
        owned,
        origin,
        lattice,
        support_lower,
        support_upper,
        owner_bounds,
        bucket_ids,
        bucket_owners,
        rank,
        execution_group,
        source_manifest_id,
        axis_name,
    )
    source = NativePeriodicSource(
        PeriodicPointOrbits(domain.cell, points, point_ids),
        source.source_id,
        source.source_revision,
        site_weights=weights if weights_present else None,
        point_owner_ranks=point_owners,
    )
    dimension = source.ambient_dimension
    inverse = _exact_lattice_inverse(lattice)
    margin = radius
    preferred_owner = (
        int(np.argmax(weights))
        if weights.size
        else 0
    )
    restricted = None
    cell_rows: dict[int, int] = {}
    image_owners = np.zeros((0,), dtype=np.int32)
    image_windings = np.zeros((0, dimension), dtype=np.int64)
    coordinate_offsets = np.zeros((0,), dtype=np.int64)
    coordinate_components = np.zeros((0,), dtype=np.float64)
    domain_points = np.zeros((0, dimension), dtype=np.float64)
    domain_tets = np.zeros((0, 4), dtype=np.int32)
    domain_face_facets = np.zeros((0, 4), dtype=np.int32)
    while True:
        local_unclipped = True
        if np.any(owned):
            carrier_lower = _outward_bound(global_lower - margin, -np.inf)
            carrier_upper = _outward_bound(global_upper + margin, np.inf)
            image_owners, image_windings = _power_images(
                points,
                point_ids,
                inverse,
                carrier_lower,
                carrier_upper,
                influence_radius,
                preferred_owner,
            )
            coordinate_offsets, coordinate_components = _exact_image_coordinates(
                points, lattice, image_owners, image_windings
            )
            neighbor_offsets, neighbors = _image_radius_neighbor_csr(
                image_owners,
                image_windings,
                coordinate_offsets,
                coordinate_components,
                influence_radius,
            )
            domain_points, domain_tets, domain_face_facets = _box_tetrahedra(
                carrier_lower, carrier_upper
            )
            try:
                restricted = meshcore.restricted_power_cells_exact(
                    points,
                    weights,
                    image_owners,
                    coordinate_offsets,
                    coordinate_components,
                    neighbor_offsets,
                    neighbors,
                    np.zeros((image_owners.size,), dtype=np.int8),
                    domain_points,
                    domain_tets,
                    np.zeros((domain_tets.shape[0],), dtype=np.int32),
                    domain_face_facets,
                )
                base_images = {
                    int(point_row): _image_row(
                        image_owners, image_windings, point_row
                    )
                    for point_row in range(point_ids.size)
                }
                all_cell_rows = _base_image_cell_rows(
                    restricted, base_images, point_ids.size
                )
                cell_rows = {
                    point_row: all_cell_rows[point_row]
                    for point_row in range(point_ids.size)
                    if owned[point_row]
                }
                if not cell_rows:
                    raise ValueError("Owned power sites have no restricted cells.")
                local_unclipped = _base_cells_are_unclipped(
                    restricted, cell_rows
                )
            except Exception as error:
                local_error = str(error)
            else:
                local_error = None
        else:
            local_error = None
        _owner_group_fail_if_any(
            local_error, execution_group, axis_name, "restricted owner power cells"
        )
        flags = _owner_group_allgather(
            np.asarray([local_unclipped], dtype=np.bool_),
            execution_group,
            axis_name,
        )
        if np.all(np.asarray(flags, dtype=np.bool_)):
            break
        margin = _outward_scalar(margin * 2.0)

    source_map = None
    geometry = None
    mesh = None
    exact_vertex_bank: tuple[tuple[Fraction, Fraction, Fraction], ...] = ()
    carrier_barycentric_bank: tuple[
        tuple[Fraction, Fraction, Fraction, Fraction], ...
    ] = ()
    output_vertex_rows = np.zeros((0,), dtype=np.int64)
    vertex_source_ids = np.zeros((0,), dtype=np.int64)
    vertex_owner_ranks = np.zeros((0,), dtype=np.int32)
    vertex_absolute_windings = np.zeros((0, dimension), dtype=np.int64)
    # Every owner joins the geometry failure collective, including owners
    # without cells; a conditional collective would desynchronize the group.
    local_error = None
    if cell_rows:
        try:
            if restricted is None:
                raise ValueError("Owned periodic cells require exact restricted power data.")
            (
                lifted,
                exact_vertex_bank,
                carrier_barycentric_bank,
                output_vertex_rows,
                exact_image_points,
                vertex_source_ids,
                vertex_owner_ranks,
                vertex_absolute_windings,
                vertex_representatives,
                vertex_shifts,
            ) = _polyhedral_quotient_from_power(
                domain,
                source.source_id,
                source.source_revision,
                point_ids,
                point_owners,
                weights,
                image_owners,
                image_windings,
                coordinate_offsets,
                coordinate_components,
                domain_points,
                restricted,
                cell_rows,
            )
            carrier_mesh = CellMesh(
                domain_points,
                (
                    CellBlock(
                        "periodic-power-carrier",
                        "tetrahedron",
                        domain_tets,
                    ),
                ),
            )
            context = ExactPeriodicPowerSourceContext(
                source.domain,
                weights,
                image_owners,
                image_windings,
                weights_present=weights_present,
            )
            source_map = ExactPowerCellGeometrySource(
                np.asarray(exact_image_points, dtype=np.float64),
                np.asarray(weights[image_owners], dtype=np.float64),
                carrier_mesh,
                restricted,
                source_id=source.source_id,
                source_revision=source.source_revision,
                periodic_source_context=context,
                exact_site_points=exact_image_points,
                precomputed_exact_vertices=exact_vertex_bank,
                precomputed_carrier_barycentrics=carrier_barycentric_bank,
                output_vertex_indices=output_vertex_rows,
            )
            mesh = _bind_power_periodic_topology(
                lifted,
                domain,
                vertex_representatives,
                vertex_shifts,
                source_map,
                np.asarray(
                    [point_ids[point_row] for point_row in sorted(cell_rows)],
                    dtype=np.int64,
                ),
            )
            source_map.require_periodic_mesh(mesh)
            geometry = CellGeometrySpec.power(mesh, source_map)
        except Exception as error:
            local_error = str(error)
    _owner_group_fail_if_any(
        local_error, execution_group, axis_name, "exact owner power geometry"
    )
    candidate_revision_id = canonical_fingerprint(
        {
            "kind": "owner-local-periodic-power-candidate",
            "source_manifest": source_manifest_id,
            "source_directory": directory_id,
            "periodic_cell": domain.cell.cell_id,
            "support_rounds": support_round_ids,
            "influence_radius": (
                Fraction.from_float(influence_radius).numerator,
                Fraction.from_float(influence_radius).denominator,
            ),
            "carrier_margin": (
                Fraction.from_float(margin).numerator,
                Fraction.from_float(margin).denominator,
            ),
        }
    )
    shell_certificate = {
        "source_manifest_id": source_manifest_id,
        "source_directory_id": directory_id,
        "periodic_cell_id": domain.cell.cell_id,
        "candidate_revision_id": candidate_revision_id,
        "support_rounds": support_round_ids,
        "owner_fractional_bounds": tuple(
            None if bounds is None else (bounds[0].copy(), bounds[1].copy())
            for bounds in owner_bounds
        ),
        "global_weight_bounds": (weight_min, weight_max),
        "influence_radius": influence_radius,
        "carrier_margin": margin,
    }
    return {
        "candidate": {
            "point_ids": point_ids,
            "point_owner_ranks": point_owners,
            "source_owned": owned,
            "source_count": total,
            "source_directory_id": directory_id,
            "source_directory_bucket_ids": bucket_ids,
            "source_directory_bucket_owners": bucket_owners,
            "candidate_revision_id": candidate_revision_id,
            "certified_periodic_margin": margin,
            "periodic_shell_certificate": shell_certificate,
        },
        "source": source,
        "mesh": mesh,
        "geometry": geometry,
        "source_map": source_map,
        "exact_vertex_bank": exact_vertex_bank,
        "vertex_source_ids": vertex_source_ids,
        "vertex_owner_ranks": vertex_owner_ranks,
        "vertex_absolute_windings": vertex_absolute_windings,
        "cell_owner_ranks": (
            np.full((mesh.entity_set(3).entity_ids.size,), rank, dtype=np.int32)
            if mesh is not None
            else np.zeros((0,), dtype=np.int32)
        ),
        "origin": origin,
        "lattice": lattice,
    }

def _owner_local_periodic_power_mesh_data(
    power: dict[str, Any],
    rank: int,
    /,
) -> dict[str, Any]:
    from phydrax.discretization._periodic_topology import _entity_vertex_rows
    from phydrax.geometry._rational import fraction_down, fraction_up

    mesh = power["mesh"]
    dimension = 3
    layout_id = canonical_fingerprint(
        {
            "kind": "owner-local-periodic-power-layout",
            "dimension": dimension,
            "cell_kind": "polyhedron",
            "coordinate_basis": "exact-cell-vertices",
        }
    )
    if mesh is None:
        cell_ids = np.zeros((0,), dtype=np.int64)
        vertex_ids = np.zeros((0,), dtype=np.int64)
        vertex_coordinates = np.zeros((0, dimension), dtype=np.float64)
        exact_vertex_bank: tuple[tuple[Fraction, ...], ...] = ()
        cell_rows = np.zeros((0,), dtype=np.int64)
        vertex_rows = np.zeros((0,), dtype=np.int64)
        coordinate_rows = np.zeros((0,), dtype=np.int64)
        empty_keys = np.zeros((0,), dtype=np.int64)
        entity_ids = {
            degree: np.zeros((0,), dtype=np.int64) for degree in range(4)
        }
        entity_offsets = {
            degree: np.zeros((1,), dtype=np.int64) for degree in range(4)
        }
        entity_values = {
            degree: np.zeros((0,), dtype=np.int32) for degree in range(4)
        }
        return {
            "cell_rows": cell_rows,
            "cell_ids": cell_ids,
            "owned": np.zeros((0,), dtype=np.bool_),
            "cell_measures": np.zeros((0,), dtype=np.float64),
            "exact_cell_measures": (),
            "cell_owner_ranks": np.zeros((0,), dtype=np.int32),
            "local_mesh": None,
            "support_cell_mask": np.zeros((0,), dtype=np.bool_),
            "vertex_valid": np.zeros((0, 0), dtype=np.bool_),
            "vertex_rows": vertex_rows,
            "vertex_ids": vertex_ids,
            "vertex_coordinates": vertex_coordinates,
            "exact_vertex_coordinates": exact_vertex_bank,
            "vertex_source_ids": np.zeros((0,), dtype=np.int64),
            "vertex_owner_ranks": np.zeros((0,), dtype=np.int32),
            "vertex_absolute_windings": np.zeros((0, dimension), dtype=np.int64),
            "coordinate_rows": coordinate_rows,
            "coordinates": vertex_coordinates.copy(),
            "exact_coordinates": (),
            "local_entity_ids": entity_ids,
            "entity_vertex_offsets": entity_offsets,
            "entity_vertex_values": entity_values,
            "cell_key_offsets": np.zeros((1,), dtype=np.int64),
            "cell_key_source_ids": empty_keys,
            "cell_key_relative_windings": np.zeros(
                (0, dimension), dtype=np.int64
            ),
            "periodic_cell_ids": cell_ids.copy(),
            "vertex_representative_ids": np.zeros((0,), dtype=np.int64),
            "vertex_windings": np.zeros((0, dimension), dtype=np.int64),
            "cell_windings": np.zeros((0, dimension), dtype=np.int64),
            "support_fractional_lower": np.zeros((0, dimension), dtype=np.float64),
            "support_fractional_upper": np.zeros((0, dimension), dtype=np.float64),
            "geometry_layout_id": layout_id,
            "blocks": (),
            "cell_kind": "polyhedron",
            "vertices": np.zeros((0, 0), dtype=np.int32),
            "cell_vertex_shifts": np.zeros((0, 0, dimension), dtype=np.int64),
            "source_cell_ids": cell_ids.copy(),
        }

    source_map = power["source_map"]
    geometry = power["geometry"]
    if (
        not isinstance(source_map, ExactPowerCellGeometrySource)
        or not isinstance(geometry, CellGeometrySpec)
        or mesh.periodic_topology is None
        or not isinstance(mesh.connectivity, PolyhedralConnectivity)
    ):
        raise TypeError("Owner-local power output lost its exact polyhedral source map.")
    connectivity = mesh.connectivity
    blocks = tuple(mesh.blocks)
    cell_ids = np.asarray(mesh.entity_set(3).entity_ids, dtype=np.int64)
    cell_rows = np.arange(cell_ids.size, dtype=np.int64)
    cell_row_by_id = {
        int(identifier): row for row, identifier in enumerate(cell_ids.tolist())
    }
    block_vertices_by_index = tuple(
        np.asarray(block.vertices, dtype=np.int32) for block in blocks
    )
    block_valid_by_index = tuple(
        np.asarray(block.vertex_valid, dtype=np.bool_) for block in blocks
    )
    block_rows_by_cell_id: dict[int, tuple[int, int]] = {}
    for block_index, block in enumerate(blocks):
        for local_row, identifier in enumerate(
            np.asarray(block.global_ids, dtype=np.int64).tolist()
        ):
            cell_id = int(identifier)
            if cell_id in block_rows_by_cell_id:
                raise ValueError("Periodic power cell ID occurs in multiple blocks.")
            block_rows_by_cell_id[cell_id] = (block_index, local_row)
    if set(block_rows_by_cell_id) != set(cell_row_by_id):
        raise ValueError("Periodic power blocks do not cover the entity cell bank.")
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    vertex_rows = np.arange(vertex_ids.size, dtype=np.int64)
    coordinate_rows = vertex_rows.copy()
    vertex_coordinates = np.asarray(mesh.coordinates, dtype=np.float64)
    exact_vertex_bank = tuple(
        tuple(value for value in row) for row in power["exact_vertex_bank"]
    )
    if len(exact_vertex_bank) != vertex_ids.size:
        raise ValueError("Exact power vertex bank differs from local mesh vertices.")
    vertex_source_ids = np.asarray(power["vertex_source_ids"], dtype=np.int64)
    vertex_owner_ranks = np.asarray(power["vertex_owner_ranks"], dtype=np.int32)
    vertex_absolute_windings = np.asarray(
        power["vertex_absolute_windings"], dtype=np.int64
    )
    if (
        vertex_source_ids.shape != vertex_ids.shape
        or vertex_owner_ranks.shape != vertex_ids.shape
        or vertex_absolute_windings.shape != (vertex_ids.size, dimension)
    ):
        raise ValueError("Exact power quotient vertex identities are malformed.")

    cell_key_offsets = np.zeros((cell_ids.size + 1,), dtype=np.int64)
    key_sources: list[np.ndarray] = []
    key_windings: list[np.ndarray] = []
    orders = identification_orders(power["source"].domain.cell)
    for row, cell_id in enumerate(cell_ids):
        block_index, block_row = block_rows_by_cell_id[int(cell_id)]
        local_vertices = block_vertices_by_index[block_index][block_row][
            block_valid_by_index[block_index][block_row]
        ]
        source_ids_for_cell = vertex_source_ids[local_vertices]
        windings_for_cell = vertex_absolute_windings[local_vertices]
        anchor = min(
            range(local_vertices.size),
            key=lambda local: (
                int(source_ids_for_cell[local]),
                tuple(int(value) for value in windings_for_cell[local]),
            ),
        )
        relative = reduce_periodic_exponents(
            windings_for_cell - windings_for_cell[anchor][None, :], orders
        )
        ordered_keys = tuple(
            sorted(
                (
                    int(source_id),
                    tuple(int(value) for value in winding),
                )
                for source_id, winding in zip(
                    source_ids_for_cell, relative, strict=True
                )
            )
        )
        key_sources.append(
            np.asarray([source_id for source_id, _ in ordered_keys], dtype=np.int64)
        )
        key_windings.append(
            np.asarray(
                [winding for _, winding in ordered_keys], dtype=np.int64
            ).reshape((-1, dimension))
        )
        cell_key_offsets[row + 1] = (
            cell_key_offsets[row] + len(ordered_keys)
        )
    cell_key_source_ids = (
        np.concatenate(key_sources).astype(np.int64, copy=False)
        if key_sources
        else np.zeros((0,), dtype=np.int64)
    )
    cell_key_relative_windings = (
        np.concatenate(key_windings, axis=0).astype(np.int64, copy=False)
        if key_windings
        else np.zeros((0, dimension), dtype=np.int64)
    )

    face_vertex_offsets = np.asarray(
        connectivity.face_vertex_offsets, dtype=np.int64
    )
    face_vertex_values = np.asarray(
        connectivity.face_vertex_values, dtype=np.int32
    )
    cell_face_offsets = np.asarray(connectivity.cell_face_offsets, dtype=np.int64)
    cell_face_values = np.asarray(connectivity.cell_face_values, dtype=np.int64)
    cell_face_signs = np.asarray(
        connectivity.cell_face_sign_values, dtype=np.int8
    )
    cell_measures = np.empty((cell_ids.size,), dtype=np.float64)
    exact_cell_measures: list[Fraction] = []
    for cell in range(cell_ids.size):
        volume_six = Fraction(0)
        for incidence in range(
            int(cell_face_offsets[cell]), int(cell_face_offsets[cell + 1])
        ):
            face = int(cell_face_values[incidence])
            start, stop = map(int, face_vertex_offsets[face : face + 2])
            loop = face_vertex_values[start:stop]
            if cell_face_signs[incidence] < 0:
                loop = loop[::-1]
            if loop.size < 3:
                raise ValueError("Exact periodic power cell has a degenerate face.")
            origin = exact_vertex_bank[int(loop[0])]
            for index in range(1, loop.size - 1):
                left = exact_vertex_bank[int(loop[index])]
                right = exact_vertex_bank[int(loop[index + 1])]
                cross = (
                    left[1] * right[2] - left[2] * right[1],
                    left[2] * right[0] - left[0] * right[2],
                    left[0] * right[1] - left[1] * right[0],
                )
                volume_six += sum(
                    (origin[axis] * cross[axis] for axis in range(3)),
                    Fraction(0),
                )
        measure = volume_six / 6
        if measure <= 0:
            raise ValueError("Exact periodic power cell has nonpositive volume.")
        exact_cell_measures.append(measure)
        cell_measures[cell] = float(measure)

    entity_vertices = _entity_vertex_rows(mesh)
    local_entity_ids: dict[int, np.ndarray] = {}
    entity_vertex_offsets: dict[int, np.ndarray] = {}
    entity_vertex_values: dict[int, np.ndarray] = {}
    for degree, rows_at_degree in enumerate(entity_vertices):
        entity_ids_at_degree = np.asarray(
            mesh.entity_set(degree).entity_ids, dtype=np.int64
        )
        ordered_rows = (
            np.arange(entity_ids_at_degree.size, dtype=np.int64)
            if degree in (0, dimension)
            else np.argsort(entity_ids_at_degree, kind="stable")
        )
        local_entity_ids[degree] = entity_ids_at_degree[ordered_rows]
        offsets = np.zeros((ordered_rows.size + 1,), dtype=np.int64)
        loops: list[np.ndarray] = []
        for local_row, entity_row in enumerate(ordered_rows):
            vertices_ = np.asarray(rows_at_degree[int(entity_row)], dtype=np.int64)
            loops.append(vertices_.astype(np.int32, copy=False))
            offsets[local_row + 1] = offsets[local_row] + vertices_.size
        entity_vertex_offsets[degree] = offsets
        entity_vertex_values[degree] = (
            np.concatenate(loops).astype(np.int32, copy=False)
            if loops
            else np.zeros((0,), dtype=np.int32)
        )
    element_by_name = dict(
        zip(geometry.block_names, geometry.elements, strict=True)
    )
    face_ids_global = np.asarray(connectivity.face_global_ids, dtype=np.int64)
    block_cell_face_offsets_global = np.asarray(
        connectivity.cell_face_offsets, dtype=np.int64
    )
    block_cell_face_values = np.asarray(
        connectivity.cell_face_values, dtype=np.int64
    )
    block_cell_face_signs = np.asarray(
        connectivity.cell_face_sign_values, dtype=np.int8
    )
    face_vertex_offsets_global = np.asarray(
        connectivity.face_vertex_offsets, dtype=np.int64
    )
    face_vertex_values_global = np.asarray(
        connectivity.face_vertex_values, dtype=np.int32
    )
    maximum_arity = max(vertices.shape[1] for vertices in block_vertices_by_index)
    combined_vertices = np.full(
        (cell_ids.size, maximum_arity), -1, dtype=np.int32
    )
    combined_vertex_valid = np.zeros(
        (cell_ids.size, maximum_arity), dtype=np.bool_
    )
    combined_cell_vertex_shifts = np.zeros(
        (cell_ids.size, maximum_arity, dimension), dtype=np.int64
    )
    block_records: list[dict[str, Any]] = []
    for block_index, block in enumerate(blocks):
        block_vertices = block_vertices_by_index[block_index]
        block_valid = block_valid_by_index[block_index]
        block_cell_ids = np.asarray(block.global_ids, dtype=np.int64)
        global_rows = np.asarray(
            [cell_row_by_id[int(identifier)] for identifier in block_cell_ids],
            dtype=np.int64,
        )
        arity = block_vertices.shape[1]
        combined_vertices[global_rows, :arity] = block_vertices
        combined_vertex_valid[global_rows, :arity] = block_valid
        block_shifts = np.zeros(
            (block_vertices.shape[0], arity, dimension), dtype=np.int64
        )
        block_shifts[block_valid] = vertex_absolute_windings[
            block_vertices[block_valid]
        ]
        combined_cell_vertex_shifts[global_rows, :arity] = block_shifts
        cell_face_offsets_local = np.zeros(
            (block_cell_ids.size + 1,), dtype=np.int64
        )
        used_face_rows: list[int] = []
        cell_face_signs_local: list[int] = []
        for local_row, global_row in enumerate(global_rows):
            start, stop = block_cell_face_offsets_global[
                global_row : global_row + 2
            ]
            for incidence in range(int(start), int(stop)):
                used_face_rows.append(int(block_cell_face_values[incidence]))
                cell_face_signs_local.append(int(block_cell_face_signs[incidence]))
            cell_face_offsets_local[local_row + 1] = len(used_face_rows)
        local_face_rows = tuple(
            sorted(
                set(used_face_rows),
                key=lambda face: int(face_ids_global[face]),
            )
        )
        local_face_by_row = {
            face_row: local_row for local_row, face_row in enumerate(local_face_rows)
        }
        cell_face_values_local = np.asarray(
            [local_face_by_row[face] for face in used_face_rows],
            dtype=np.int64,
        )
        face_vertex_offsets_local = np.zeros(
            (len(local_face_rows) + 1,), dtype=np.int64
        )
        face_vertex_values_local: list[int] = []
        for local_row, face_row in enumerate(local_face_rows):
            start, stop = face_vertex_offsets_global[face_row : face_row + 2]
            face_vertex_values_local.extend(
                int(value) for value in face_vertex_values_global[int(start) : int(stop)]
            )
            face_vertex_offsets_local[local_row + 1] = len(
                face_vertex_values_local
            )
        block_records.append(
            {
                "name": block.name,
                "cell_kind": block.cell_kind,
                "cell_rows": global_rows,
                "cell_ids": block_cell_ids,
                "vertices": block_vertices,
                "vertex_valid": block_valid,
                "coordinate_routes": block_vertices,
                "element": element_by_name[block.name],
                "element_rows": np.arange(
                    block_cell_ids.size, dtype=np.int32
                ),
                "source_cell_ids": block_cell_ids,
                "chart_arguments": None,
                "cell_measures": cell_measures[global_rows],
                "face_ids": face_ids_global[np.asarray(local_face_rows, dtype=np.int64)],
                "cell_face_offsets": cell_face_offsets_local,
                "cell_face_values": cell_face_values_local,
                "cell_face_signs": np.asarray(
                    cell_face_signs_local, dtype=np.int8
                ),
                "face_vertex_offsets": face_vertex_offsets_local,
                "face_vertex_values": np.asarray(
                    face_vertex_values_local, dtype=np.int32
                ),
                "cell_vertex_shifts": block_shifts,
            }
        )

    origin_exact = tuple(
        Fraction.from_float(float(value)) for value in power["origin"]
    )
    inverse = _exact_lattice_inverse(np.asarray(power["lattice"], dtype=np.float64))
    local_cell_vertices = _entity_vertex_rows(mesh)[dimension]
    support_lower = np.empty((cell_ids.size, dimension), dtype=np.float64)
    support_upper = np.empty_like(support_lower)
    for row, vertex_rows_ in enumerate(local_cell_vertices):
        fractional_points = []
        for vertex in np.asarray(vertex_rows_, dtype=np.int64):
            point = exact_vertex_bank[int(vertex)]
            fractional_points.append(
                tuple(
                    sum(
                        (
                            (point[cart] - origin_exact[cart])
                            * inverse[cart][axis]
                            for cart in range(dimension)
                        ),
                        Fraction(0),
                    )
                    for axis in range(dimension)
                )
            )
        for axis in range(dimension):
            support_lower[row, axis] = fraction_down(
                min(point[axis] for point in fractional_points)
            )
            support_upper[row, axis] = fraction_up(
                max(point[axis] for point in fractional_points)
            )
    return {
        "cell_rows": cell_rows,
        "cell_ids": cell_ids,
        "owned": np.ones(cell_ids.shape, dtype=np.bool_),
        "cell_measures": cell_measures,
        "exact_cell_measures": tuple(exact_cell_measures),
        "cell_owner_ranks": np.full(cell_ids.shape, rank, dtype=np.int32),
        "support_cell_mask": np.ones(cell_ids.shape, dtype=np.bool_),
        "vertex_valid": combined_vertex_valid,
        "vertex_rows": vertex_rows,
        "vertex_ids": vertex_ids,
        "vertex_coordinates": vertex_coordinates,
        "exact_vertex_coordinates": exact_vertex_bank,
        "vertex_source_ids": vertex_source_ids,
        "vertex_owner_ranks": vertex_owner_ranks,
        "vertex_absolute_windings": vertex_absolute_windings,
        "coordinate_rows": coordinate_rows,
        "coordinates": vertex_coordinates.copy(),
        "exact_coordinates": exact_vertex_bank,
        "local_entity_ids": local_entity_ids,
        "entity_vertex_offsets": entity_vertex_offsets,
        "entity_vertex_values": entity_vertex_values,
        "cell_key_offsets": cell_key_offsets,
        "cell_key_source_ids": cell_key_source_ids,
        "cell_key_relative_windings": cell_key_relative_windings,
        "periodic_cell_ids": cell_ids.copy(),
        "vertex_representative_ids": vertex_source_ids.copy(),
        "vertex_windings": vertex_absolute_windings.copy(),
        "cell_windings": np.zeros((cell_ids.size, dimension), dtype=np.int64),
        "support_fractional_lower": support_lower,
        "support_fractional_upper": support_upper,
        "geometry_layout_id": layout_id,
        "blocks": tuple(block_records),
        "local_mesh": mesh,
        "cell_kind": "polyhedron",
        "vertices": combined_vertices,
        "cell_vertex_shifts": combined_cell_vertex_shifts,
        "source_cell_ids": cell_ids.copy(),
    }
__all__ = ["periodic_power_mesh"]
