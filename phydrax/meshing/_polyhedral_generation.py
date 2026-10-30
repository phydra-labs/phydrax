from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from itertools import combinations
from math import isfinite
from typing import final

import equinox as eqx
import numpy as np

import phydrax._meshcore as meshcore
from phydrax._fingerprint import array_tree_fingerprint, canonical_fingerprint
from phydrax._strict import StrictModule
from phydrax._trainable import NonTrainableState
from phydrax._validation import (
    nonnegative_integer,
    normalized_identifier,
    positive_finite_float,
)
from phydrax.discretization._cell_complex import (
    PolyhedralConnectivity,
)
from phydrax.discretization._cell_geometry import CellGeometrySpec
from phydrax.discretization._cell_geometry_validity import cell_geometry_id
from phydrax.discretization._cell_mesh import CellMesh
from phydrax.discretization._exact_power_geometry import (
    ExactPowerCellGeometrySource,
    prepare_exact_power_geometry_source,
)
from phydrax.discretization._periodic_cell import PeriodicCell
from phydrax.discretization._periodic_exact import (
    exact_periodic_element,
    identification_orders,
    PeriodicIsometryGroup,
)
from phydrax.discretization._periodic_topology import PeriodicMeshTopology
from phydrax.discretization._reference_cell import reference_cell_topology
from phydrax.meshing._contracts import VolumeMeshingSpec
from phydrax.meshing._controls import PeriodicConstraint
from phydrax.meshing._sizing import SizeControlStrength, UniformSizeControl
from phydrax.meshing._volume_generation import (
    generate_plc_volume,
    NativeVolumeSchedule,
    VolumeConstruction,
)
from phydrax.meshing.providers._native_sources import (
    NativePlcSource,
    NativePolyhedralSource,
)


type _ExactAffineTransform = tuple[tuple[Fraction, ...], ...]
type _PeriodicControlBinding = tuple[int, int, _ExactAffineTransform]


@final
class NativePolyhedralSchedule(StrictModule, NonTrainableState):
    """Scientific controls for restricted-power site refinement and ownership."""

    maximum_cell_diameter: float | None = eqx.field(static=True)
    target_site_volumes: tuple[tuple[str, float], ...] = eqx.field(static=True)
    relative_volume_tolerance: float = eqx.field(static=True)
    feature_tolerance: float = eqx.field(static=True)
    protected_site_ids: tuple[str, ...] = eqx.field(static=True)
    require_every_site: bool = eqx.field(static=True)
    lloyd_steps: int = eqx.field(static=True)
    schedule_id: str = eqx.field(static=True)

    def __init__(
        self,
        maximum_cell_diameter: float | None = None,
        target_site_volumes: Sequence[tuple[str, float]] = (),
        relative_volume_tolerance: float = 1e-6,
        feature_tolerance: float = 1e-10,
        protected_site_ids: Sequence[str] = (),
        require_every_site: bool = False,
        lloyd_steps: int = 0,
    ) -> None:
        diameter = (
            None
            if maximum_cell_diameter is None
            else positive_finite_float(maximum_cell_diameter, "maximum_cell_diameter")
        )
        volumes = tuple(
            sorted(
                (
                    normalized_identifier(site_id, "target_site_volume.site_id"),
                    positive_finite_float(volume, "target_site_volume.value"),
                )
                for site_id, volume in target_site_volumes
            )
        )
        if len({site_id for site_id, _ in volumes}) != len(volumes):
            raise ValueError("target_site_volumes must identify unique power sites.")
        tolerance = positive_finite_float(
            relative_volume_tolerance, "relative_volume_tolerance"
        )
        feature = positive_finite_float(feature_tolerance, "feature_tolerance")
        protected = tuple(
            sorted(
                normalized_identifier(site_id, "protected_site_id")
                for site_id in protected_site_ids
            )
        )
        if len(set(protected)) != len(protected):
            raise ValueError("protected_site_ids must be unique.")
        if not isinstance(require_every_site, bool):
            raise TypeError("require_every_site must be bool.")
        relocation_steps = nonnegative_integer(lloyd_steps, "lloyd_steps")
        self.maximum_cell_diameter = diameter
        self.target_site_volumes = volumes
        self.relative_volume_tolerance = tolerance
        self.feature_tolerance = feature
        self.protected_site_ids = protected
        self.require_every_site = require_every_site
        self.lloyd_steps = relocation_steps
        self.schedule_id = canonical_fingerprint(
            {
                "kind": "native-polyhedral-schedule",
                "maximum_cell_diameter": diameter,
                "target_site_volumes": volumes,
                "relative_volume_tolerance": tolerance,
                "feature_tolerance": feature,
                "protected_site_ids": protected,
                "require_every_site": require_every_site,
                "lloyd_steps": relocation_steps,
            }
        )


def effective_polyhedral_schedule(
    specification: VolumeMeshingSpec,
    supplied: NativePolyhedralSchedule,
    /,
) -> NativePolyhedralSchedule:
    """Tighten the supplied cell-diameter bound by every hard uniform size control."""
    hard_bounds = []
    policy = specification.size_compliance
    for control in specification.size_controls:
        if not isinstance(control, UniformSizeControl):
            continue
        if control.strength is not SizeControlStrength.HARD:
            continue
        tolerance = policy.absolute_tolerance + policy.relative_tolerance * abs(
            control.target_size
        )
        bound = (
            control.maximum_size
            if control.maximum_size is not None
            else control.target_size + tolerance
        )
        hard_bounds.append(bound)
    diameter = supplied.maximum_cell_diameter
    if hard_bounds:
        physical_bound = min(hard_bounds)
        diameter = physical_bound if diameter is None else min(diameter, physical_bound)
    return NativePolyhedralSchedule(
        maximum_cell_diameter=diameter,
        target_site_volumes=supplied.target_site_volumes,
        relative_volume_tolerance=supplied.relative_volume_tolerance,
        feature_tolerance=supplied.feature_tolerance,
        protected_site_ids=supplied.protected_site_ids,
        require_every_site=supplied.require_every_site,
        lloyd_steps=supplied.lloyd_steps,
    )


def _carrier_tetrahedralization(
    carrier: VolumeConstruction, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align constrained recovered PLC faces with canonical carrier tet slots."""
    mesh = carrier.mesh
    if (
        len(mesh.blocks) != 1
        or reference_cell_topology(mesh.blocks[0].cell_kind).family != "tetrahedron"
    ):
        raise TypeError(
            "Restricted-power construction requires one tetrahedral-family carrier block."
        )
    tetrahedra = np.asarray(mesh.blocks[0].vertices, dtype=np.int32)
    boundary_faces = np.asarray(carrier.boundary_faces, dtype=np.int64)
    face_sources = np.asarray(carrier.face_sources)
    if (
        tetrahedra.ndim != 2
        or tetrahedra.shape[1] != 4
        or boundary_faces.ndim != 2
        or boundary_faces.shape[1] != 3
        or face_sources.shape != (boundary_faces.shape[0],)
    ):
        raise ValueError("Recovered PLC tetrahedra and constrained faces must align.")

    constraint_by_vertices: dict[tuple[int, int, int], int] = {}
    for constraint in range(boundary_faces.shape[0]):
        sorted_vertices = sorted(
            (
                int(boundary_faces[constraint, 0]),
                int(boundary_faces[constraint, 1]),
                int(boundary_faces[constraint, 2]),
            )
        )
        key = (sorted_vertices[0], sorted_vertices[1], sorted_vertices[2])
        if key in constraint_by_vertices:
            raise ValueError(
                "Recovered PLC constrained faces must have unique vertex triples."
            )
        constraint_by_vertices[key] = constraint

    # Native restricted-power slot ``4 * cell + k`` is the face opposite vertex k.
    tet_face_facets = np.full(tetrahedra.shape, -1, dtype=np.int32)
    matched = np.zeros(boundary_faces.shape[0], dtype=np.bool_)
    for cell in range(tetrahedra.shape[0]):
        for local_face in range(4):
            sorted_vertices = sorted(
                int(tetrahedra[cell, vertex])
                for vertex in range(4)
                if vertex != local_face
            )
            key = (sorted_vertices[0], sorted_vertices[1], sorted_vertices[2])
            constraint = constraint_by_vertices.get(key)
            if constraint is not None:
                tet_face_facets[cell, local_face] = constraint
                matched[constraint] = True
    if not np.all(matched):
        raise ValueError(
            "Every recovered PLC constraint face must bound its tetrahedral carrier."
        )
    return tetrahedra, tet_face_facets, face_sources


def same_region_component_ids(
    cell_sites: np.ndarray,
    cell_regions: np.ndarray,
    face_cells: np.ndarray,
    face_facets: np.ndarray,
    split_sites: np.ndarray,
    /,
) -> np.ndarray:
    """Label same-site connected components without crossing PLC strata.

    Native pieces may join only through reciprocal interior faces that are
    unconstrained, belong to the same source region, and have an unsplit site.
    """
    sites = np.asarray(cell_sites)
    regions = np.asarray(cell_regions)
    sides = np.asarray(face_cells)
    facets = np.asarray(face_facets)
    split = np.asarray(split_sites)
    if (
        sites.ndim != 1
        or not np.issubdtype(sites.dtype, np.integer)
        or regions.shape != sites.shape
        or not np.issubdtype(regions.dtype, np.integer)
        or np.any(sites < 0)
        or np.any(regions < 0)
    ):
        raise ValueError(
            "Polyhedral cell site and region rows must be nonnegative integer vectors."
        )
    if (
        sides.ndim != 2
        or sides.shape[1] != 2
        or not np.issubdtype(sides.dtype, np.integer)
        or facets.shape != (sides.shape[0],)
        or not np.issubdtype(facets.dtype, np.integer)
        or np.any(facets < -1)
        or np.any(sides[:, 0] < 0)
        or np.any(sides[:, 0] >= sites.size)
        or np.any(sides[:, 1] >= sites.size)
    ):
        raise ValueError(
            "Polyhedral reciprocal face owners and source facets must align."
        )
    if (
        split.ndim != 1
        or not np.issubdtype(split.dtype, np.integer)
        or np.any((split != 0) & (split != 1))
        or (sites.size and int(np.max(sites)) >= split.size)
    ):
        raise ValueError(
            "Polyhedral split-site flags must cover every nonnegative site ID."
        )

    parent = np.arange(sites.size, dtype=np.int64)

    def root(cell: int, /) -> int:
        current = cell
        while parent[current] != current:
            parent[current] = parent[parent[current]]
            current = int(parent[current])
        return current

    for face, (left_value, right_value) in enumerate(sides):
        left, right = int(left_value), int(right_value)
        if (
            right >= 0
            and facets[face] < 0
            and sites[left] == sites[right]
            and regions[left] == regions[right]
            and split[sites[left]] == 0
        ):
            left_root, right_root = root(left), root(right)
            if left_root != right_root:
                parent[max(left_root, right_root)] = min(left_root, right_root)

    result = np.empty(sites.shape, dtype=np.int64)
    for site_value in np.unique(sites):
        rows = np.flatnonzero(sites == site_value)
        roots = np.fromiter((root(int(row)) for row in rows), dtype=np.int64)
        result[rows] = np.searchsorted(np.unique(roots), roots)
    return result


def _spans_three_space(points: np.ndarray, /) -> bool:
    """Test full affine dimension with exact represented-coordinate predicates."""
    if points.shape[0] < 4:
        return False
    origin = points[0]
    distinct = np.flatnonzero(np.any(points != origin, axis=1))
    if not distinct.size:
        return False
    second = points[distinct[0]]
    collinear = np.ones(points.shape[0], dtype=np.bool_)
    for axes in ((0, 1), (1, 2), (2, 0)):
        selected = np.asarray(axes, dtype=np.int64)
        signs = meshcore.exact_orient2d(
            origin[selected], second[selected], points[:, selected]
        )
        collinear &= signs == 0
    off_line = np.flatnonzero(~collinear)
    if not off_line.size:
        return False
    third = points[off_line[0]]
    signs = meshcore.exact_orient3d(origin, second, third, points)
    return bool(np.any(signs != 0))


def _spanning_guard_sites(
    sites: np.ndarray,
    weights: np.ndarray,
    domain: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Add four exactly power-dominated sites when rank is deficient."""
    if sites.shape[0] == 1 or _spans_three_space(sites):
        return sites, weights
    lower = np.min(domain, axis=0)
    upper = np.max(domain, axis=0)
    extents = upper - lower
    if not np.all(np.isfinite(extents)):
        raise ValueError("Polyhedral carrier extent is not representable in binary64.")
    scale = max(1.0, float(np.max(extents)))
    base = lower - 4.0 * scale
    guards = base + 8.0 * scale * np.asarray(
        ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)),
        dtype=np.float64,
    )
    if not np.all(np.isfinite(guards)) or not _spans_three_space(guards):
        raise ValueError("Polyhedral carrier has no finite spanning guard coordinates.")

    reference = tuple(Fraction(float(value)) for value in sites[0])
    reference_weight = Fraction(float(weights[0]))
    corners = tuple(
        tuple(Fraction(float(value)) for value in corner)
        for corner in np.asarray(
            tuple(
                (x, y, z)
                for x in (lower[0], upper[0])
                for y in (lower[1], upper[1])
                for z in (lower[2], upper[2])
            ),
            dtype=np.float64,
        )
    )
    guard_weights = np.empty((guards.shape[0],), dtype=np.float64)
    for row, guard in enumerate(guards):
        exact_guard = tuple(Fraction(float(value)) for value in guard)
        minimum = min(
            sum(
                (site_axis - x_axis) ** 2 - (reference_axis - x_axis) ** 2
                for site_axis, reference_axis, x_axis in zip(
                    exact_guard, reference, corner, strict=True
                )
            )
            + reference_weight
            for corner in corners
        )
        weight = float(minimum - max(Fraction(1), abs(minimum)) - 1)
        weight = float(np.nextafter(weight, -np.inf))
        if not np.isfinite(weight) or Fraction(weight) >= minimum:
            raise ValueError("Polyhedral guard power cannot be strictly dominated.")
        guard_weights[row] = weight
    return (
        np.concatenate((sites, guards), axis=0),
        np.concatenate((weights, guard_weights), axis=0),
    )


def _regular_neighbor_csr(
    sites: np.ndarray,
    weights: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Build complete site adjacency from the native regular tetrahedralization."""
    if sites.shape[0] == 1:
        return np.zeros((2,), dtype=np.int64), np.zeros((0,), dtype=np.int32)
    tetrahedra, _ = meshcore.regular_3d(sites, weights)
    if tetrahedra.ndim != 2 or tetrahedra.shape[1] != 4:
        raise RuntimeError("Native regular triangulation returned malformed tetrahedra.")
    if not tetrahedra.shape[0]:
        raise RuntimeError(
            "Native regular triangulation returned no spanning tetrahedra."
        )
    edge_rows = np.concatenate(
        tuple(np.sort(tetrahedra[:, pair], axis=1) for pair in combinations(range(4), 2)),
        axis=0,
    )
    edges = np.unique(edge_rows, axis=0)
    directed = np.concatenate((edges, edges[:, ::-1]), axis=0)
    directed = directed[np.lexsort((directed[:, 1], directed[:, 0]))]
    counts = np.bincount(directed[:, 0], minlength=sites.shape[0])
    offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
    return offsets, directed[:, 1].astype(np.int32)


def _restricted_power_cells(
    sites: np.ndarray,
    weights: np.ndarray,
    split_sites: np.ndarray,
    carrier_points: np.ndarray,
    carrier_tetrahedra: np.ndarray,
    carrier_regions: np.ndarray,
    tet_face_facets: np.ndarray,
    /,
) -> tuple[
    meshcore.RestrictedPowerCells,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Restrict the actual regular-triangulation cells to recovered PLC tets."""
    regular_sites, regular_weights = _spanning_guard_sites(sites, weights, carrier_points)
    source_site_ids = np.concatenate(
        (
            np.arange(sites.shape[0], dtype=np.int64),
            np.full(regular_sites.shape[0] - sites.shape[0], -1, dtype=np.int64),
        )
    )
    neighbor_offsets, neighbors = _regular_neighbor_csr(regular_sites, regular_weights)
    regular_split = np.concatenate(
        (
            np.asarray(split_sites, dtype=np.int8),
            np.zeros(regular_sites.shape[0] - sites.shape[0], dtype=np.int8),
        )
    )
    restricted = meshcore.restricted_power_cells(
        regular_sites,
        regular_weights,
        neighbor_offsets,
        neighbors,
        regular_split,
        carrier_points,
        carrier_tetrahedra,
        carrier_regions,
        tet_face_facets,
    )
    if np.any(restricted.cell_sites >= sites.shape[0]) or np.any(
        restricted.vertex_sites >= sites.shape[0]
    ):
        raise RuntimeError(
            "Strictly hidden spanning guards must not enter restricted output."
        )
    return (
        restricted,
        source_site_ids,
        regular_sites,
        regular_weights,
        neighbor_offsets,
        neighbors,
    )


class PolyhedralGenerationError(RuntimeError):
    """A numerical restricted-power request could not progress honestly."""

    def __init__(
        self,
        reason: str,
        /,
        *,
        entities: tuple[int, ...] = (),
    ) -> None:
        self.reason = reason
        self.entities = entities
        super().__init__(
            f"Polyhedral generation could not satisfy {reason}; entities={entities}."
        )


def _point_power(
    point: tuple[Fraction, Fraction, Fraction],
    site: np.ndarray,
    weight: float,
    /,
) -> Fraction:
    exact_site = (
        Fraction(float(site[0])),
        Fraction(float(site[1])),
        Fraction(float(site[2])),
    )
    return sum(
        ((x - y) ** 2 for x, y in zip(point, exact_site, strict=True)),
        Fraction(0),
    ) - Fraction(float(weight))


def _strictly_inside_tetrahedron(
    point: np.ndarray,
    tetrahedron: np.ndarray,
    carrier_points: np.ndarray,
    /,
) -> bool:
    for opposite in range(4):
        face = tuple(index for index in range(4) if index != opposite)
        a, b, c = carrier_points[tetrahedron[np.asarray(face, dtype=np.int64)]]
        expected = int(
            meshcore.exact_orient3d(a, b, c, carrier_points[tetrahedron[opposite]])
        )
        actual = int(meshcore.exact_orient3d(a, b, c, point))
        if expected == 0 or actual * expected <= 0:
            return False
    return True


def _strictly_inside_power_cell(
    point: np.ndarray,
    site: int,
    sites: np.ndarray,
    weights: np.ndarray,
    neighbor_offsets: np.ndarray,
    neighbors: np.ndarray,
    /,
) -> bool:
    exact_point: tuple[Fraction, Fraction, Fraction] = (
        Fraction(float(point[0])),
        Fraction(float(point[1])),
        Fraction(float(point[2])),
    )
    own = _point_power(exact_point, sites[site], weights[site])
    for neighbor in neighbors[neighbor_offsets[site] : neighbor_offsets[site + 1]]:
        if own >= _point_power(exact_point, sites[int(neighbor)], weights[int(neighbor)]):
            return False
    return True


def _piece_centroid(
    restricted: meshcore.RestrictedPowerCells,
    piece: int,
    carrier_points: np.ndarray,
    carrier_tetrahedra: np.ndarray,
    regular_sites: np.ndarray,
    regular_weights: np.ndarray,
    neighbor_offsets: np.ndarray,
    neighbors: np.ndarray,
    /,
) -> np.ndarray | None:
    volume = float(restricted.piece_volumes[piece])
    if not isfinite(volume) or volume <= 0.0:
        return None
    point = np.asarray(restricted.piece_moments[piece], dtype=np.float64) / volume
    if not np.all(np.isfinite(point)):
        return None
    tet = carrier_tetrahedra[int(restricted.piece_tets[piece])]
    site = int(restricted.piece_sites[piece])
    if not _strictly_inside_tetrahedron(point, tet, carrier_points):
        return None
    if not _strictly_inside_power_cell(
        point, site, regular_sites, regular_weights, neighbor_offsets, neighbors
    ):
        return None
    return point


def _in_tetrahedron_closure(
    point: np.ndarray,
    tetrahedron: np.ndarray,
    carrier_points: np.ndarray,
    /,
) -> bool:
    for opposite in range(4):
        face = tuple(index for index in range(4) if index != opposite)
        a, b, c = carrier_points[tetrahedron[np.asarray(face, dtype=np.int64)]]
        expected = int(
            meshcore.exact_orient3d(a, b, c, carrier_points[tetrahedron[opposite]])
        )
        actual = int(meshcore.exact_orient3d(a, b, c, point))
        if expected == 0 or actual * expected < 0:
            return False
    return True


def _lloyd_relocation(
    restricted: meshcore.RestrictedPowerCells,
    sites: np.ndarray,
    site_ids: tuple[str, ...],
    site_parent_ids: tuple[str | None, ...],
    split_sites: np.ndarray,
    protected_site_ids: tuple[str, ...],
    carrier_points: np.ndarray,
    carrier_tetrahedra: np.ndarray,
    carrier_regions: np.ndarray,
    regular_sites: np.ndarray,
    regular_weights: np.ndarray,
    neighbor_offsets: np.ndarray,
    neighbors: np.ndarray,
    /,
) -> tuple[np.ndarray, bool]:
    """Relocate only single-component unsplit sites to verified cell centroids."""
    cell_sites = np.asarray(restricted.cell_sites, dtype=np.int64)
    cell_regions = np.asarray(restricted.cell_regions, dtype=np.int64)
    volumes = np.asarray(restricted.cell_volumes, dtype=np.float64)
    moments = np.asarray(restricted.cell_moments, dtype=np.float64)
    multiplicity = np.bincount(cell_sites, minlength=sites.shape[0])
    protected = frozenset(protected_site_ids)
    result = np.array(sites, dtype=np.float64, copy=True)
    for cell, site_value in enumerate(cell_sites):
        site = int(site_value)
        parent_id = site_parent_ids[site]
        if (
            multiplicity[site] != 1
            or split_sites[site]
            or site_ids[site] in protected
            or parent_id in protected
            or volumes[cell] <= 0.0
        ):
            continue
        candidate = moments[cell] / volumes[cell]
        if not np.all(np.isfinite(candidate)) or not _strictly_inside_power_cell(
            candidate, site, regular_sites, regular_weights, neighbor_offsets, neighbors
        ):
            continue
        region = cell_regions[cell]
        if any(
            carrier_regions[tet] == region
            and _in_tetrahedron_closure(
                candidate, carrier_tetrahedra[tet], carrier_points
            )
            for tet in range(carrier_tetrahedra.shape[0])
        ):
            result[site] = candidate
    return result, not np.array_equal(result, sites)


def _refinement_candidates(
    restricted: meshcore.RestrictedPowerCells,
    cell_order: np.ndarray,
    oversized_cells: np.ndarray,
    carrier_points: np.ndarray,
    carrier_tetrahedra: np.ndarray,
    regular_sites: np.ndarray,
    regular_weights: np.ndarray,
    neighbor_offsets: np.ndarray,
    neighbors: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points: list[np.ndarray] = []
    parents: list[int] = []
    parent_cells: list[int] = []
    piece_cells = np.asarray(restricted.piece_cells, dtype=np.int64)
    piece_volumes = np.asarray(restricted.piece_volumes, dtype=np.float64)
    for cell in oversized_cells:
        native_cell = int(cell_order[int(cell)])
        pieces = np.flatnonzero(piece_cells == native_cell)
        ordered = pieces[np.lexsort((pieces, -piece_volumes[pieces]))]
        candidate = None
        for piece in ordered:
            candidate = _piece_centroid(
                restricted,
                int(piece),
                carrier_points,
                carrier_tetrahedra,
                regular_sites,
                regular_weights,
                neighbor_offsets,
                neighbors,
            )
            if candidate is not None:
                break
        if candidate is None:
            raise PolyhedralGenerationError(
                "piece_geometry_unresolved", entities=(int(cell),)
            )
        points.append(candidate)
        parents.append(int(restricted.cell_sites[native_cell]))
        parent_cells.append(native_cell)
    return (
        np.asarray(points, dtype=np.float64),
        np.asarray(parents, dtype=np.int64),
        np.asarray(parent_cells, dtype=np.int64),
    )


def _walk_power_owner(
    point: np.ndarray,
    sites: np.ndarray,
    weights: np.ndarray,
    neighbor_offsets: np.ndarray,
    neighbors: np.ndarray,
    /,
) -> tuple[int, Fraction]:
    """Find the global power minimum by strict descent on regular neighbors."""
    exact_point: tuple[Fraction, Fraction, Fraction] = (
        Fraction(float(point[0])),
        Fraction(float(point[1])),
        Fraction(float(point[2])),
    )
    if sites.shape[0] == 1:
        return 0, _point_power(exact_point, sites[0], weights[0])
    active = np.flatnonzero(neighbor_offsets[1:] > neighbor_offsets[:-1])
    if not active.size:
        raise PolyhedralGenerationError("weight_jacobian_unresolved")
    owner = int(active[0])
    while True:
        best_site = owner
        best_power = _point_power(exact_point, sites[owner], weights[owner])
        for neighbor_value in neighbors[
            neighbor_offsets[owner] : neighbor_offsets[owner + 1]
        ]:
            neighbor = int(neighbor_value)
            power = _point_power(exact_point, sites[neighbor], weights[neighbor])
            if power < best_power or (power == best_power and neighbor < best_site):
                best_site, best_power = neighbor, power
        if best_site == owner:
            return owner, best_power
        owner = best_site


def _revive_hidden_site(
    site: int,
    sites: np.ndarray,
    weights: np.ndarray,
    carrier_points: np.ndarray,
    carrier_tetrahedra: np.ndarray,
    neighbor_offsets: np.ndarray,
    neighbors: np.ndarray,
    /,
) -> np.ndarray:
    """Raise one requested hidden site's weight until it owns an interior point."""
    point = np.mean(carrier_points[carrier_tetrahedra[:1]], axis=1)[0]
    _, best_power = _walk_power_owner(point, sites, weights, neighbor_offsets, neighbors)
    exact_point = tuple(Fraction(float(value)) for value in point)
    required = (
        sum(
            (
                (x - Fraction(float(y))) ** 2
                for x, y in zip(exact_point, sites[site], strict=True)
            ),
            Fraction(0),
        )
        - best_power
    )
    candidate = float(np.nextafter(float(required), np.inf))
    while isfinite(candidate) and Fraction(candidate) <= required:
        candidate = float(np.nextafter(candidate, np.inf))
    if not isfinite(candidate) or candidate <= weights[site]:
        raise PolyhedralGenerationError("weight_jacobian_unresolved", entities=(site,))
    renewed = np.array(weights, dtype=np.float64, copy=True)
    renewed[site] = candidate
    renewed -= renewed[0]
    return renewed


def _exact_polygon_area_upper(
    coordinate_source: ExactPowerCellGeometrySource,
    vertex_rows: np.ndarray,
    /,
) -> float:
    points = coordinate_source.source_coordinates(vertex_rows)
    if len(points) < 3:
        return 0.0
    origin = points[0]
    vector = [Fraction(0), Fraction(0), Fraction(0)]
    for index in range(1, len(points) - 1):
        first = tuple(points[index][axis] - origin[axis] for axis in range(3))
        second = tuple(points[index + 1][axis] - origin[axis] for axis in range(3))
        cross = (
            first[1] * second[2] - first[2] * second[1],
            first[2] * second[0] - first[0] * second[2],
            first[0] * second[1] - first[1] * second[0],
        )
        vector = [left + right for left, right in zip(vector, cross, strict=True)]
    squared = sum((value * value for value in vector), Fraction(0))
    return 0.5 * _sqrt_fraction_upper(squared)


def _site_volume_residual(
    restricted: meshcore.RestrictedPowerCells,
    targets: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray, float]:
    achieved = np.bincount(
        np.asarray(restricted.cell_sites, dtype=np.int64),
        weights=np.asarray(restricted.cell_volumes, dtype=np.float64),
        minlength=targets.size,
    )
    empty = np.flatnonzero(achieved == 0)
    residual = float(np.max(np.abs(achieved - targets) / targets))
    total_error = abs(float(np.sum(achieved) - np.sum(targets))) / float(np.sum(targets))
    if total_error > 0:
        residual = max(residual, total_error)
    return achieved, empty, residual


def _update_site_weights(
    restricted: meshcore.RestrictedPowerCells,
    coordinate_source: ExactPowerCellGeometrySource,
    sites: np.ndarray,
    weights: np.ndarray,
    targets: np.ndarray,
    achieved: np.ndarray,
    empty: np.ndarray,
    carrier_points: np.ndarray,
    carrier_tetrahedra: np.ndarray,
    neighbor_offsets: np.ndarray,
    neighbors: np.ndarray,
    /,
) -> np.ndarray:
    if empty.size:
        return _revive_hidden_site(
            int(empty[0]),
            sites,
            weights,
            carrier_points,
            carrier_tetrahedra,
            neighbor_offsets,
            neighbors,
        )
    diagonal = np.zeros(weights.shape, dtype=np.float64)
    offsets = np.asarray(restricted.face_offsets, dtype=np.int64)
    face_vertices = np.asarray(restricted.face_vertices, dtype=np.int64)
    sides = np.asarray(restricted.face_cells, dtype=np.int64)
    facets = np.asarray(restricted.face_facets, dtype=np.int64)
    cell_sites = np.asarray(restricted.cell_sites, dtype=np.int64)
    if sides.ndim != 2 or sides.shape[1] != 2 or facets.shape != (sides.shape[0],):
        raise ValueError("Restricted power face owners and facet IDs must align.")
    for face in range(sides.shape[0]):
        left, right = int(sides[face, 0]), int(sides[face, 1])
        if right < 0 or facets[face] >= 0:
            continue
        first, second = int(cell_sites[left]), int(cell_sites[right])
        if first == second:
            continue
        start, stop = offsets[face : face + 2]
        area = _exact_polygon_area_upper(coordinate_source, face_vertices[start:stop])
        delta = tuple(
            Fraction(float(a)) - Fraction(float(b))
            for a, b in zip(sites[first], sites[second], strict=True)
        )
        distance = _sqrt_fraction_upper(
            sum((value * value for value in delta), Fraction(0))
        )
        if distance == 0.0:
            raise PolyhedralGenerationError(
                "coincident_positive_volume_sites", entities=(first, second)
            )
        derivative = area / (2.0 * distance)
        diagonal[first] += derivative
        diagonal[second] += derivative
    unresolved = np.flatnonzero((diagonal <= 0.0) & (targets > 0.0))
    if unresolved.size:
        raise PolyhedralGenerationError(
            "weight_jacobian_unresolved", entities=tuple(map(int, unresolved))
        )
    update = 0.5 * (targets - achieved) / diagonal
    renewed = np.asarray(weights + update, dtype=np.float64)
    if not np.all(np.isfinite(renewed)) or np.array_equal(renewed, weights):
        raise PolyhedralGenerationError("weight_jacobian_unresolved")
    renewed -= renewed[0]
    return renewed


def _publish_power_topology(
    restricted: meshcore.RestrictedPowerCells, /
) -> tuple[CellMesh, np.ndarray, np.ndarray, np.ndarray]:
    """Publish reciprocal native face loops and retain both canonical orders."""
    site_count = len(restricted.cell_sites)
    offsets = np.asarray(restricted.face_offsets)
    vertices = np.asarray(restricted.face_vertices)
    sides = np.asarray(restricted.face_cells)
    if (
        site_count == 0
        or sides.ndim != 2
        or sides.shape[1] != 2
        or not np.issubdtype(sides.dtype, np.integer)
        or offsets.shape != (sides.shape[0] + 1,)
        or offsets[0] != 0
        or offsets[-1] != vertices.size
        or np.any(offsets[1:] < offsets[:-1])
        or np.any(sides[:, 0] < 0)
        or np.any(sides[:, 0] >= site_count)
        or np.any(sides[:, 1] >= site_count)
    ):
        raise ValueError(
            "Restricted-power reciprocal faces must align with their cell owners."
        )
    cells: list[list[np.ndarray]] = [[] for _ in range(site_count)]
    for face, (left_value, right_value) in enumerate(sides):
        left, right = int(left_value), int(right_value)
        loop = vertices[offsets[face] : offsets[face + 1]]
        if loop.size < 3:
            raise ValueError("Restricted-power faces must have at least three vertices.")
        cells[left].append(loop)
        if right >= 0:
            if right == left:
                raise ValueError("A restricted-power face cannot neighbor its own cell.")
            cells[right].append(loop[::-1])

    mesh = CellMesh.from_polyhedra(
        restricted.vertices,
        cells,
        block_name="restricted-power",
    )
    connectivity = mesh.connectivity
    if not isinstance(connectivity, PolyhedralConnectivity):
        raise TypeError(
            "Restricted-power publication must produce polyhedral connectivity."
        )
    cell_order = np.asarray(connectivity.cell_global_ids, dtype=np.int64)
    if cell_order.shape != (site_count,) or np.unique(cell_order).size != site_count:
        raise RuntimeError(
            "Polyhedral publication must retain every native cell identity."
        )
    inverse_cell_order = np.empty((site_count,), dtype=np.int64)
    inverse_cell_order[cell_order] = np.arange(site_count, dtype=np.int64)

    native_face_by_vertices: dict[tuple[int, ...], int] = {}
    for face, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
        key = tuple(sorted(map(int, vertices[start:stop])))
        if key in native_face_by_vertices:
            raise RuntimeError(
                "Restricted-power kernel emitted a duplicate reciprocal face."
            )
        native_face_by_vertices[key] = face
    native_face_order = np.empty((connectivity.face_count,), dtype=np.int64)
    canonical_offsets = np.asarray(connectivity.face_vertex_offsets)
    canonical_vertices = np.asarray(connectivity.face_vertex_values)
    for face, (start, stop) in enumerate(
        zip(canonical_offsets[:-1], canonical_offsets[1:], strict=True)
    ):
        key = tuple(sorted(map(int, canonical_vertices[start:stop])))
        try:
            native_face_order[face] = native_face_by_vertices[key]
        except KeyError as error:
            raise RuntimeError(
                "Canonical polyhedral faces must retain native ancestry."
            ) from error
    return mesh, cell_order, inverse_cell_order, native_face_order


@dataclass(frozen=True, slots=True)
class PolyhedralConstruction:
    """Restricted-power candidate plus its carrier and source ancestry."""

    source: NativePlcSource | NativePolyhedralSource
    carrier: VolumeConstruction
    mesh: CellMesh
    geometry: CellGeometrySpec
    coordinate_source: ExactPowerCellGeometrySource
    restricted_cells: meshcore.RestrictedPowerCells
    site_points: np.ndarray
    site_weights: np.ndarray
    site_ids: tuple[str, ...]
    site_parents: np.ndarray
    site_parent_ids: tuple[str | None, ...]
    split_sites: np.ndarray
    cell_sites: np.ndarray
    cell_regions: np.ndarray
    component_ids: np.ndarray
    face_recovery_faces: np.ndarray
    face_source_facets: np.ndarray
    face_source_ids: tuple[str | None, ...]
    face_source_occurrence_paths: tuple[tuple[str, ...], ...]
    interface_faces: np.ndarray
    sheet_faces: np.ndarray
    piece_sites: np.ndarray
    piece_tetrahedra: np.ndarray
    piece_cells: np.ndarray
    piece_volumes: np.ndarray
    piece_moments: np.ndarray
    cell_diameter_upper: np.ndarray
    refinement_steps: int
    construction_id: str


def _source_sites(
    source: NativePolyhedralSource, /
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[str, ...],
    tuple[str | None, ...],
]:
    if not isinstance(source, NativePolyhedralSource):
        raise TypeError("Restricted-power generation requires authored power sites.")
    sites = np.asarray(source.sites, dtype=np.float64)
    weights = np.asarray(source.weights, dtype=np.float64)
    site_ids = source.site_ids
    parents = np.arange(sites.shape[0], dtype=np.int64)
    parent_ids = tuple(None for _ in site_ids)
    return sites, weights, parents, site_ids, parent_ids


def _refined_site_id(
    parent_site_id: str, parent_cell_id: int, refinement_step: int, /
) -> str:
    if not parent_site_id or parent_cell_id < 0 or refinement_step < 0:
        raise ValueError(
            "Refined site ancestry requires a parent ID and stable cell/event IDs."
        )
    return canonical_fingerprint(
        {
            "kind": "native-polyhedral-refined-site",
            "parent_site_id": parent_site_id,
            "parent_cell_id": int(parent_cell_id),
            "refinement_step": int(refinement_step),
        }
    )


def _source_split_flags(
    source: NativePlcSource | NativePolyhedralSource,
    specification: VolumeMeshingSpec,
    site_count: int,
    /,
) -> np.ndarray:
    """Split sites only for features that same-site joining would swallow.

    Native joining merges a site's pieces across unconstrained carrier faces
    only, so every facet-polygon edge and vertex stays on a constrained face.
    Segments or protected vertices off every facet polygon lie in the volume
    and survive only as edges or vertices of unjoined per-tetrahedron pieces.
    """
    complex_ = source.complex
    offsets = np.asarray(complex_.polygon_offsets, dtype=np.int64).tolist()
    loops = np.asarray(complex_.polygon_vertices, dtype=np.int64).tolist()
    polygon_edges: set[tuple[int, int]] = set()
    for start, stop in zip(offsets[:-1], offsets[1:], strict=True):
        loop = loops[start:stop]
        for first, second in zip(loop, (*loop[1:], loop[0]), strict=True):
            polygon_edges.add((min(first, second), max(first, second)))
    floating_segment = any(
        (min(first, second), max(first, second)) not in polygon_edges
        for first, second in np.asarray(complex_.segments, dtype=np.int64).tolist()
    )
    vertex_bank = np.asarray(source.scope_entity_ids(0), dtype=np.int64)
    polygon_vertices = set(loops)
    floating_vertex = any(
        int(row) not in polygon_vertices
        for feature in specification.protected_features
        if feature.scope.entity_dimension == 0
        for row in np.flatnonzero(
            np.isin(vertex_bank, np.asarray(feature.scope.entity_ids, dtype=np.int64))
        )
    )
    return np.full((site_count,), floating_segment or floating_vertex, dtype=np.int8)


def _mapped_face_ancestry(
    carrier: VolumeConstruction,
    source: NativePolyhedralSource,
    restricted: meshcore.RestrictedPowerCells,
    native_face_order: np.ndarray,
    /,
) -> tuple[
    np.ndarray,
    tuple[str | None, ...],
    tuple[tuple[str, ...], ...],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    recovery_faces = np.asarray(restricted.face_facets, dtype=np.int64)[native_face_order]
    source_facets = np.full(recovery_faces.shape, -1, dtype=np.int64)
    source_ids: list[str | None] = []
    source_paths: list[tuple[str, ...]] = []
    for recovered in recovery_faces.tolist():
        if recovered < 0:
            source_ids.append(None)
            source_paths.append(())
            continue
        source_facets[len(source_ids)] = carrier.face_source_facets[recovered]
        source_ids.append(str(carrier.face_source_ids[recovered]))
        source_paths.append(tuple(carrier.face_source_occurrence_paths[recovered]))
    regions = np.asarray(source.complex.facet_regions, dtype=np.int64)
    constrained = source_facets >= 0
    pairs = np.full((source_facets.size, 2), -1, dtype=np.int64)
    pairs[constrained] = regions[source_facets[constrained]]
    interfaces = np.flatnonzero(
        constrained
        & (pairs[:, 0] >= 0)
        & (pairs[:, 1] >= 0)
        & (pairs[:, 0] != pairs[:, 1])
    )
    sheets = np.flatnonzero(constrained & (pairs[:, 0] == pairs[:, 1]))
    return (
        recovery_faces,
        tuple(source_ids),
        tuple(source_paths),
        interfaces,
        sheets,
        source_facets,
    )


def _sqrt_fraction_upper(value: Fraction, /) -> float:
    """Outward-round the square root of one nonnegative exact measure."""
    if value < 0:
        raise ValueError("Exact squared length must be nonnegative.")
    if value == 0:
        return 0.0
    represented = float(value)
    if represented == 0.0 or not isfinite(represented):
        raise ValueError("Exact cell diameter is not representable in binary64.")
    result = float(np.nextafter(np.sqrt(represented), np.inf))
    if not isfinite(result):
        raise ValueError("Exact cell diameter is not representable in binary64.")
    while Fraction(result) ** 2 < value:
        result = float(np.nextafter(result, np.inf))
        if not isfinite(result):
            raise ValueError("Exact cell diameter is not representable in binary64.")
    return result


def _cell_diameter_upper_bounds(
    mesh: CellMesh,
    coordinate_source: ExactPowerCellGeometrySource,
    /,
) -> np.ndarray:
    """Bound cell diameters by the diagonals of exact coordinate boxes."""
    connectivity = mesh.connectivity
    if not isinstance(connectivity, PolyhedralConnectivity):
        raise TypeError("Power-cell size measurement requires polyhedral connectivity.")
    coordinates = coordinate_source.source_coordinates()
    if len(coordinates) != mesh.coordinates.shape[0]:
        raise ValueError(
            "Exact power coordinates must align with published mesh vertices."
        )
    offsets = np.asarray(connectivity.cell_vertex_offsets, dtype=np.int64)
    vertices = np.asarray(connectivity.cell_vertex_values, dtype=np.int64)
    result = np.empty((connectivity.cell_count,), dtype=np.float64)
    for cell, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
        rows = vertices[start:stop]
        lower = tuple(
            min(coordinates[int(row)][axis] for row in rows) for axis in range(3)
        )
        upper = tuple(
            max(coordinates[int(row)][axis] for row in rows) for axis in range(3)
        )
        squared = sum(
            ((right - left) ** 2 for left, right in zip(lower, upper, strict=True)),
            Fraction(0),
        )
        result[cell] = _sqrt_fraction_upper(squared)
    return result


def _construct_polyhedral(
    source: NativePolyhedralSource,
    carrier: VolumeConstruction,
    schedule: NativePolyhedralSchedule,
    sites: np.ndarray,
    weights: np.ndarray,
    site_ids: tuple[str, ...],
    site_parents: np.ndarray,
    site_parent_ids: tuple[str | None, ...],
    split_sites: np.ndarray,
    restricted: meshcore.RestrictedPowerCells,
    source_site_ids: np.ndarray,
    refinement_steps: int,
    /,
) -> PolyhedralConstruction:
    mesh, cell_order, inverse_cell_order, native_face_order = _publish_power_topology(
        restricted
    )
    native_cells = np.asarray(restricted.cell_sites, dtype=np.int64)
    cell_sites = source_site_ids[native_cells[cell_order]]
    if np.any(cell_sites < 0):
        raise RuntimeError("Hidden guard sites must not own a published polyhedron.")
    cell_regions = np.asarray(restricted.cell_regions, dtype=np.int32)[cell_order]
    component_ids = same_region_component_ids(
        native_cells,
        np.asarray(restricted.cell_regions),
        np.asarray(restricted.face_cells),
        np.asarray(restricted.face_facets),
        split_sites,
    )[cell_order]
    (
        face_recovery_faces,
        face_source_ids,
        face_source_occurrence_paths,
        interface_faces,
        sheet_faces,
        face_source_facets,
    ) = _mapped_face_ancestry(carrier, source, restricted, native_face_order)
    coordinate_source = prepare_exact_power_geometry_source(
        sites,
        weights,
        carrier.mesh,
        restricted,
        source_id=source.source_id,
        source_revision=source.source_revision,
        domain_source=(
            source.complex if source.mapped_domain is None else source.mapped_domain
        ),
    )
    geometry = CellGeometrySpec.power(mesh, coordinate_source)
    cell_diameters = _cell_diameter_upper_bounds(mesh, coordinate_source)
    piece_sites = source_site_ids[np.asarray(restricted.piece_sites, dtype=np.int64)]
    if np.any(piece_sites < 0):
        raise RuntimeError("Hidden guard sites must not own a restricted convex piece.")
    piece_cells = inverse_cell_order[np.asarray(restricted.piece_cells, dtype=np.int64)]
    identity = canonical_fingerprint(
        {
            "kind": "native-polyhedral-construction",
            "source": source.binding_id,
            "carrier": carrier.complex_id,
            "schedule": schedule.schedule_id,
            "site_ids": site_ids,
            "site_parents": site_parent_ids,
            "sites": array_tree_fingerprint((sites, weights)),
            "mesh": mesh.mesh_id,
            "geometry": cell_geometry_id(geometry),
            "coordinate_source": coordinate_source.source_id,
        }
    )
    return PolyhedralConstruction(
        source=source,
        carrier=carrier,
        mesh=mesh,
        geometry=geometry,
        coordinate_source=coordinate_source,
        restricted_cells=restricted,
        site_points=sites,
        site_weights=weights,
        site_ids=site_ids,
        site_parent_ids=site_parent_ids,
        site_parents=site_parents,
        split_sites=split_sites,
        cell_sites=cell_sites,
        cell_regions=cell_regions,
        component_ids=component_ids,
        face_recovery_faces=face_recovery_faces,
        face_source_facets=face_source_facets,
        face_source_ids=face_source_ids,
        face_source_occurrence_paths=face_source_occurrence_paths,
        interface_faces=interface_faces,
        sheet_faces=sheet_faces,
        piece_sites=piece_sites,
        piece_tetrahedra=np.asarray(restricted.piece_tets, dtype=np.int64),
        piece_cells=piece_cells,
        piece_volumes=np.asarray(restricted.piece_volumes, dtype=np.float64),
        piece_moments=np.asarray(restricted.piece_moments, dtype=np.float64),
        cell_diameter_upper=cell_diameters,
        refinement_steps=refinement_steps,
        construction_id=identity,
    )


def _periodic_transform_matrix(
    constraint: PeriodicConstraint, /
) -> _ExactAffineTransform:
    transform = np.asarray(constraint.transform, dtype=np.float64)
    if transform.shape != (4, 4) or not np.array_equal(
        transform[3], np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float64)
    ):
        raise ValueError("Periodic PLC constraints require exact 3D affine maps.")
    return tuple(tuple(Fraction(float(value)) for value in row) for row in transform)


def _periodic_identification(
    specification: VolumeMeshingSpec, /
) -> tuple[PeriodicCell | PeriodicIsometryGroup, tuple[_PeriodicControlBinding, ...]]:
    """Resolve authored isometries into one canonical periodic identification."""
    constraints = specification.periodic_constraints
    if not constraints:
        raise ValueError("A periodic quotient requires authored periodic constraints.")
    exact_transforms = tuple(
        _periodic_transform_matrix(constraint) for constraint in constraints
    )
    identity = tuple(
        tuple(Fraction(int(row == column)) for column in range(3)) for row in range(3)
    )
    translations_only = all(
        tuple(tuple(transform[row][column] for column in range(3)) for row in range(3))
        == identity
        for transform in exact_transforms
    )
    if translations_only:
        normalized_translations: list[tuple[Fraction, Fraction, Fraction]] = []
        signs: list[int] = []
        for transform in exact_transforms:
            translation = (transform[0][3], transform[1][3], transform[2][3])
            first_nonzero = next((value for value in translation if value), Fraction(0))
            if not first_nonzero:
                raise ValueError("Periodic PLC translations must be nonzero.")
            sign = 1 if first_nonzero > 0 else -1
            normalized_translations.append(
                (sign * translation[0], sign * translation[1], sign * translation[2])
            )
            signs.append(sign)
        lattice_vectors = tuple(sorted(set(normalized_translations)))
        vector_rows = np.asarray(
            [[float(value) for value in vector] for vector in lattice_vectors],
            dtype=np.float64,
        )
        identification: PeriodicCell | PeriodicIsometryGroup = PeriodicCell(
            vector_rows, periodic_axes=(True,) * len(lattice_vectors)
        )
        generator_rows = {vector: row for row, vector in enumerate(lattice_vectors)}
        bindings = tuple(
            (
                generator_rows[normalized],
                sign,
                transform,
            )
            for normalized, sign, transform in zip(
                normalized_translations, signs, exact_transforms, strict=True
            )
        )
    else:
        matrices = {
            transform: np.asarray(constraint.transform, dtype=np.float64)
            for constraint, transform in zip(constraints, exact_transforms, strict=True)
        }
        actions: dict[_ExactAffineTransform, tuple[int, int]] = {}
        generator_matrices: list[np.ndarray] = []
        for transform in sorted(matrices):
            if transform in actions:
                continue
            generator = matrices[transform]
            generator_index = len(generator_matrices)
            actions[transform] = (generator_index, 1)
            single_generator = PeriodicIsometryGroup(generator[None, :, :])
            inverse = exact_periodic_element(
                single_generator, np.asarray((-1,), dtype=np.int64)
            )
            actions.setdefault(inverse, (generator_index, -1))
            generator_matrices.append(generator)
        identification = PeriodicIsometryGroup(np.stack(generator_matrices, axis=0))
        bindings = tuple(
            (actions[transform][0], actions[transform][1], transform)
            for transform in exact_transforms
        )
    facet_generators = {
        generator
        for constraint, (generator, _, _) in zip(constraints, bindings, strict=True)
        if constraint.source_scope.entity_dimension == 2
    }
    if facet_generators != set(range(identification.rank)):
        raise ValueError("Every periodic identification generator must pair a PLC facet.")
    return identification, bindings


def _periodic_facet_rows(
    source: NativePolyhedralSource, constraint: PeriodicConstraint, *, target: bool
) -> set[int]:
    scope = constraint.target_scope if target else constraint.source_scope
    if (
        scope.entity_dimension != 2
        or scope.source_id != source.source_id
        or scope.source_revision != source.source_revision
        or scope.entity_set_id != source.scope_entity_set_id(2)
    ):
        raise ValueError("Periodic face scopes must bind this source's PLC facet set.")
    facet_rows = {
        (identifier, tuple(path)): row
        for row, (identifier, path) in enumerate(
            zip(
                source.complex.facet_ids,
                source.complex.facet_occurrence_paths,
                strict=True,
            )
        )
    }
    rows: set[int] = set()
    for identifier in np.asarray(scope.entity_ids, dtype=np.int64).tolist():
        key = source.source_entity_key_for_scope(2, int(identifier))
        try:
            rows.add(facet_rows[key])
        except KeyError as error:
            raise ValueError(
                "Periodic face scopes must identify authored PLC facets."
            ) from error
    return rows


def _periodic_face_vertex_sets(
    construction: PolyhedralConstruction, facet_rows: set[int], /
) -> tuple[frozenset[int], ...]:
    connectivity = construction.mesh.connectivity
    if not isinstance(connectivity, PolyhedralConnectivity):
        raise TypeError("Periodic PLC generation requires polyhedral connectivity.")
    recovered = np.asarray(construction.face_source_facets, dtype=np.int64)
    selected_faces = np.flatnonzero(np.isin(recovered, tuple(sorted(facet_rows))))
    if selected_faces.size == 0:
        raise ValueError("A periodic PLC facet scope has no output boundary faces.")
    offsets = np.asarray(connectivity.face_vertex_offsets, dtype=np.int64).tolist()
    vertices = np.asarray(connectivity.face_vertex_values, dtype=np.int64).tolist()
    return tuple(
        frozenset(vertices[offsets[face] : offsets[face + 1]])
        for face in selected_faces.tolist()
    )


def _apply_periodic_transform(
    transform: _ExactAffineTransform,
    point: tuple[Fraction, ...],
    /,
) -> tuple[Fraction, Fraction, Fraction]:
    homogeneous = (*point, Fraction(1))

    def component(axis: int, /) -> Fraction:
        return sum(
            (
                transform[axis][coordinate] * homogeneous[coordinate]
                for coordinate in range(4)
            ),
            Fraction(0),
        )

    return component(0), component(1), component(2)


def _periodic_vertex_relations(
    source: NativePolyhedralSource,
    specification: VolumeMeshingSpec,
    construction: PolyhedralConstruction,
    bindings: tuple[_PeriodicControlBinding, ...],
    /,
) -> tuple[tuple[int, int, int, int], ...]:
    exact_points = construction.coordinate_source.source_coordinates()
    relations: list[tuple[int, int, int, int]] = []
    for constraint, (generator, sign, transform) in zip(
        specification.periodic_constraints, bindings, strict=True
    ):
        if constraint.source_scope.entity_dimension != 2:
            continue
        source_faces = _periodic_face_vertex_sets(
            construction, _periodic_facet_rows(source, constraint, target=False)
        )
        target_faces = _periodic_face_vertex_sets(
            construction, _periodic_facet_rows(source, constraint, target=True)
        )
        source_rows = sorted(set().union(*source_faces))
        target_rows = sorted(set().union(*target_faces))
        if len(source_rows) != len(target_rows) or len(source_faces) != len(target_faces):
            raise ValueError(
                "Periodic PLC facets require matching output vertex subdivisions."
            )
        target_by_coordinate: dict[tuple[Fraction, ...], int] = {}
        for row in target_rows:
            point = exact_points[row]
            if point in target_by_coordinate:
                raise ValueError(
                    "Periodic PLC target facets have ambiguous coincident output vertices."
                )
            target_by_coordinate[point] = row
        paired: dict[int, int] = {}
        paired_targets: set[int] = set()
        for source_row in source_rows:
            source_point = exact_points[source_row]
            target_point = _apply_periodic_transform(transform, source_point)
            try:
                target_row = target_by_coordinate[target_point]
            except KeyError as error:
                raise ValueError(
                    "Periodic PLC facets do not have an exact transformed output subdivision."
                ) from error
            if target_row in paired_targets:
                raise ValueError("Periodic PLC vertex pairs must be bijective.")
            paired[source_row] = target_row
            paired_targets.add(target_row)
            if target_row != source_row:
                relations.append((source_row, target_row, generator, sign))
        if len(paired) != len(target_rows):
            raise ValueError("Periodic PLC vertex pairs must cover both facet scopes.")
        target_face_set = set(target_faces)
        if any(
            frozenset(paired[row] for row in face) not in target_face_set
            for face in source_faces
        ):
            raise ValueError(
                "Periodic PLC facets do not have an exact transformed output face subdivision."
            )
    if not relations:
        raise ValueError("Periodic quotient generation requires a paired PLC facet.")
    return tuple(relations)


def _periodic_vertex_orbits(
    mesh: CellMesh,
    relations: tuple[tuple[int, int, int, int], ...],
    identification: PeriodicCell | PeriodicIsometryGroup,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    orders = identification_orders(identification)
    rank = orders.size

    def reduce_potential(values: tuple[int, ...], /) -> tuple[int, ...]:
        return tuple(
            value if int(order) == 0 else value % int(order)
            for value, order in zip(values, orders, strict=True)
        )

    adjacency: list[list[tuple[int, tuple[int, ...]]]] = [
        [] for _ in range(mesh.coordinates.shape[0])
    ]
    for source_row, target_row, generator, sign in relations:
        step = tuple(sign if axis == generator else 0 for axis in range(rank))
        reverse = tuple(-value for value in step)
        adjacency[source_row].append((target_row, step))
        adjacency[target_row].append((source_row, reverse))
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    representatives = np.empty((vertex_ids.size,), dtype=np.int32)
    shifts = np.zeros((vertex_ids.size, rank), dtype=np.int64)
    assigned = np.zeros((vertex_ids.size,), dtype=np.bool_)
    for start in np.argsort(vertex_ids, kind="stable").tolist():
        if assigned[start]:
            continue
        potentials: dict[int, tuple[int, ...]] = {int(start): (0,) * rank}
        component: list[int] = []
        pending = [int(start)]
        while pending:
            row = pending.pop()
            component.append(row)
            for neighbor, step in adjacency[row]:
                candidate = reduce_potential(
                    tuple(potentials[row][axis] + step[axis] for axis in range(rank))
                )
                existing = potentials.get(neighbor)
                if existing is None:
                    potentials[neighbor] = candidate
                    pending.append(neighbor)
                elif existing != candidate:
                    raise ValueError(
                        "Periodic PLC vertex pairs have inconsistent quotient windings."
                    )
        root = min(component, key=lambda row: int(vertex_ids[row]))
        root_potential = potentials[root]
        for row in component:
            relative = reduce_potential(
                tuple(
                    potentials[row][axis] - root_potential[axis] for axis in range(rank)
                )
            )
            if any(
                value < np.iinfo(np.int32).min or value > np.iinfo(np.int32).max
                for value in relative
            ):
                raise ValueError(
                    "Periodic PLC winding lies outside the mesh index range."
                )
            representatives[row] = root
            shifts[row] = relative
            assigned[row] = True
    return representatives, shifts.astype(np.int32)


def _bind_periodic_polyhedral_quotient(
    source: NativePolyhedralSource,
    specification: VolumeMeshingSpec,
    construction: PolyhedralConstruction,
    /,
) -> PolyhedralConstruction:
    if not specification.periodic_constraints:
        return construction
    identification, bindings = _periodic_identification(specification)
    relations = _periodic_vertex_relations(source, specification, construction, bindings)
    representatives, shifts = _periodic_vertex_orbits(
        construction.mesh, relations, identification
    )
    lifted = construction.mesh
    connectivity = lifted.connectivity
    if not isinstance(connectivity, PolyhedralConnectivity):
        raise TypeError("Periodic quotient binding requires polyhedral connectivity.")
    topology = PeriodicMeshTopology(lifted, identification, representatives, shifts)
    entity_ids = {
        degree: np.asarray(lifted.entity_set(degree).entity_ids, dtype=np.int64)
        for degree in range(4)
    }
    mesh = CellMesh(
        lifted.coordinates,
        lifted.blocks,
        vertex_global_ids=lifted.vertex_global_ids,
        entity_global_ids=entity_ids,
        polyhedral_connectivity=connectivity,
        numeric_version=lifted.numeric_version,
        periodic_topology=topology,
    )
    topology.require_lifted(mesh.topology, mesh.lifted_topology_id, mesh.blocks)
    topology.require_coordinates(mesh.coordinates)
    geometry = CellGeometrySpec.power(mesh, construction.coordinate_source)
    construction_id = canonical_fingerprint(
        {
            "kind": "native-polyhedral-periodic-construction",
            "construction": construction.construction_id,
            "periodic_topology": topology.periodic_topology_id,
            "geometry": cell_geometry_id(geometry),
        }
    )
    return replace(
        construction,
        mesh=mesh,
        geometry=geometry,
        construction_id=construction_id,
    )


def generate_polyhedral_volume(
    source: NativePlcSource | NativePolyhedralSource,
    specification: VolumeMeshingSpec,
    schedule: NativePolyhedralSchedule,
    /,
    *,
    frame_id: str,
    volume_schedule: NativeVolumeSchedule | None = None,
) -> PolyhedralConstruction:
    """Construct nonconvex PLC-restricted power polyhedra for canonical acceptance."""
    if specification.layer_controls:
        raise ValueError(
            "Layer controls must be composed by the layer/core owner before restricted-power core fill."
        )
    mapped_geometry = (
        None if source.mapped_domain is None else source.mapped_domain.source_geometry
    )
    source_map_id = (
        ""
        if mapped_geometry is None or mapped_geometry.source_map_id is None
        else mapped_geometry.source_map_id
    )
    carrier = generate_plc_volume(
        source.complex,
        specification,
        NativeVolumeSchedule() if volume_schedule is None else volume_schedule,
        source_id=source.source_id,
        source_revision=source.source_revision,
        source_map_id=source_map_id,
        frame_id=frame_id,
        source_domain=source.mapped_domain,
    )
    if volume_schedule is not None:
        if carrier.improvement_status is not meshcore.MeshcoreStatus.OK:
            raise meshcore.MeshcoreError(
                carrier.improvement_status,
                "The explicitly scheduled fixed-boundary PLC carrier did not satisfy its joint quality request.",
            )
        if carrier.improvement_unmet:
            raise meshcore.MeshcoreError(
                meshcore.MeshcoreStatus.UNRESOLVED,
                "The explicitly scheduled fixed-boundary PLC carrier retains per-cell quality residuals.",
            )
    tetrahedra, tet_face_facets, _ = _carrier_tetrahedralization(carrier)
    carrier_points = np.asarray(carrier.mesh.coordinates, dtype=np.float64)
    carrier_regions = np.asarray(carrier.cell_regions, dtype=np.int32)
    if not isinstance(source, NativePolyhedralSource):
        raise TypeError("Restricted-power generation requires authored power sites.")
    sites, weights, site_parents, site_ids, site_parent_ids = _source_sites(source)
    if set(schedule.protected_site_ids) - set(site_ids):
        raise ValueError(
            "protected_site_ids must identify an authored source power site."
        )
    targets_by_id = dict(schedule.target_site_volumes)
    if targets_by_id and set(targets_by_id) != set(site_ids):
        raise ValueError(
            "target_site_volumes must identify every authored source power site."
        )
    targets = (
        None
        if not targets_by_id
        else np.asarray(
            [targets_by_id[site_id] for site_id in site_ids], dtype=np.float64
        )
    )
    split_sites = _source_split_flags(source, specification, len(sites))

    previous_maximum: float | None = None
    previous_volume_residual: float | None = None
    previous_volume_site_count = len(sites)
    refinement_steps = 0
    lloyd_remaining = schedule.lloyd_steps
    while True:
        (
            restricted,
            source_site_ids,
            regular_sites,
            regular_weights,
            neighbor_offsets,
            neighbors,
        ) = _restricted_power_cells(
            sites,
            weights,
            split_sites,
            carrier_points,
            tetrahedra,
            carrier_regions,
            tet_face_facets,
        )
        construction = _construct_polyhedral(
            source,
            carrier,
            schedule,
            sites,
            weights,
            site_ids,
            site_parents,
            site_parent_ids,
            split_sites,
            restricted,
            source_site_ids,
            refinement_steps,
        )

        if targets is not None:
            achieved, empty, residual = _site_volume_residual(restricted, targets)
            total_error = abs(float(np.sum(achieved) - np.sum(targets))) / float(
                np.sum(targets)
            )
            if total_error > schedule.relative_volume_tolerance:
                raise PolyhedralGenerationError("target_site_volume_total_mismatch")
            if residual > schedule.relative_volume_tolerance:
                if (
                    previous_volume_site_count == len(sites)
                    and previous_volume_residual is not None
                    and residual >= previous_volume_residual
                ):
                    raise PolyhedralGenerationError("weight_jacobian_unresolved")
                weights = _update_site_weights(
                    restricted,
                    construction.coordinate_source,
                    sites,
                    weights,
                    targets,
                    achieved,
                    empty,
                    carrier_points,
                    tetrahedra,
                    neighbor_offsets,
                    neighbors,
                )
                previous_volume_residual = residual
                previous_volume_site_count = len(sites)
                previous_maximum = None
                continue
            previous_volume_residual = None

        if schedule.require_every_site:
            counts = np.bincount(
                np.asarray(restricted.cell_sites, dtype=np.int64),
                minlength=len(sites),
            )
            empty_sites = np.flatnonzero(counts == 0)
            if empty_sites.size:
                raise PolyhedralGenerationError(
                    "required_empty_sites",
                    entities=tuple(map(int, empty_sites)),
                )

        if lloyd_remaining:
            relocated, changed = _lloyd_relocation(
                restricted,
                sites,
                site_ids,
                site_parent_ids,
                split_sites,
                schedule.protected_site_ids,
                carrier_points,
                tetrahedra,
                carrier_regions,
                regular_sites,
                regular_weights,
                neighbor_offsets,
                neighbors,
            )
            lloyd_remaining -= 1
            if changed:
                sites = relocated
                previous_volume_residual = None
                previous_maximum = None
                continue
            if lloyd_remaining:
                continue

        requested = schedule.maximum_cell_diameter
        if requested is None:
            return _bind_periodic_polyhedral_quotient(source, specification, construction)
        oversized = np.flatnonzero(construction.cell_diameter_upper > requested)
        if not oversized.size:
            return _bind_periodic_polyhedral_quotient(source, specification, construction)
        maximum = float(np.max(construction.cell_diameter_upper))
        if previous_maximum is not None and maximum >= previous_maximum:
            raise PolyhedralGenerationError(
                "site_insertion_stalled",
                entities=tuple(map(int, oversized)),
            )
        previous_maximum = maximum
        connectivity = construction.mesh.connectivity
        if isinstance(connectivity, PolyhedralConnectivity):
            cell_order = np.asarray(connectivity.cell_global_ids, dtype=np.int64)
        else:
            raise TypeError(
                "Polyhedral publication must produce polyhedral connectivity."
            )
        candidates, parents, parent_cells = _refinement_candidates(
            restricted,
            cell_order,
            oversized,
            carrier_points,
            tetrahedra,
            regular_sites,
            regular_weights,
            neighbor_offsets,
            neighbors,
        )
        combined_sites = np.concatenate((sites, candidates), axis=0)
        if np.unique(combined_sites, axis=0).shape[0] != combined_sites.shape[0]:
            raise PolyhedralGenerationError(
                "site_insertion_stalled", entities=tuple(map(int, oversized))
            )
        new_site_ids = tuple(
            _refined_site_id(site_ids[int(parent)], int(parent_cell), refinement_steps)
            for parent, parent_cell in zip(parents, parent_cells, strict=True)
        )
        if len(set((*site_ids, *new_site_ids))) != len(site_ids) + len(new_site_ids):
            raise PolyhedralGenerationError(
                "site_insertion_stalled", entities=tuple(map(int, oversized))
            )
        if targets is not None:
            new_counts = np.bincount(parents, minlength=len(sites))
            child_targets = np.asarray(
                [targets[parent] / (new_counts[parent] + 1) for parent in parents],
                dtype=np.float64,
            )
            targets = np.concatenate(
                (
                    targets / (new_counts + 1),
                    child_targets,
                )
            )
            if np.any(targets <= 0.0):
                raise PolyhedralGenerationError("site_volume_refinement_unresolved")
        sites = combined_sites
        weights = np.concatenate((weights, weights[parents]))
        site_parents = np.concatenate((site_parents, parents))
        site_parent_ids = (
            *site_parent_ids,
            *(site_ids[int(parent)] for parent in parents),
        )
        site_ids = (*site_ids, *new_site_ids)
        split_sites = np.concatenate((split_sites, split_sites[parents]))
        previous_volume_residual = None
        previous_volume_site_count = len(sites)
        refinement_steps += 1
