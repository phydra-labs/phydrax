#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Revalidated restricted-Delaunay coverage for established analytic regions."""

from __future__ import annotations

import math
from collections.abc import Iterable
from fractions import Fraction
from itertools import combinations
from typing import final, Literal, NamedTuple, TYPE_CHECKING

import equinox as eqx
import numpy as np
from jax.typing import ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._validation import canonical_identifier
from ...discretization._cell_complex import TetrahedralConnectivity
from ...discretization._cell_geometry import CellGeometrySpec
from ...discretization._cell_geometry_transition_types import CellGeometryTransition
from ...discretization._cell_geometry_validity import cell_geometry_id
from ...discretization._cell_mesh import CellMesh
from ...typing import Identifier
from .._planar_coverage import FacetOverlap
from .._source_boundary_records import SourceBoundaryDistanceWitness
from ._analytic_profile import AnalyticImplicitProfile


if TYPE_CHECKING:
    from .._global_embedding import GlobalEmbeddingCertificate


type ReferencePoint3 = tuple[Fraction, Fraction, Fraction]
type ReferenceTetrahedron = tuple[
    ReferencePoint3,
    ReferencePoint3,
    ReferencePoint3,
    ReferencePoint3,
]


class RestrictedDelaunayCoverageUnavailable(ValueError):
    """A retained restricted-Delaunay premise cannot establish source coverage."""

    def __init__(
        self,
        check: str,
        detail: str,
        /,
        *,
        status: Literal["violated", "unresolved"],
    ) -> None:
        super().__init__(detail)
        self.check = check
        self.status = status


@final
class RestrictedDelaunayBoundaryWitness(StrictModule, NonTrainableState):
    """Input binding for the independently replayed analytic boundary proof.

    The separate distance witness binds a complete certified source-cover
    query. This witness stores only identities binding the target mesh,
    analytic profile, and distance proof; boundary topology, projection, and
    degree premises are recomputed from those inputs.
    """

    __strict_contract__ = True

    mesh_id: Identifier = eqx.field(static=True)
    geometry_id: Identifier = eqx.field(static=True)
    topology_id: Identifier = eqx.field(static=True)
    profile_id: Identifier = eqx.field(static=True)
    source_id: Identifier = eqx.field(static=True)
    source_revision: Identifier = eqx.field(static=True)
    domain_id: Identifier = eqx.field(static=True)
    distance_witness_id: Identifier = eqx.field(static=True)
    witness_id: Identifier = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        mesh_id: str,
        geometry_id: str,
        topology_id: str,
        profile_id: str,
        source_id: str,
        source_revision: str,
        domain_id: str,
        distance_witness_id: str,
    ) -> None:
        identities = tuple(
            canonical_identifier(value, name)
            for value, name in (
                (mesh_id, "mesh_id"),
                (geometry_id, "geometry_id"),
                (topology_id, "topology_id"),
                (profile_id, "profile_id"),
                (source_id, "source_id"),
                (source_revision, "source_revision"),
                (domain_id, "domain_id"),
                (distance_witness_id, "distance_witness_id"),
            )
        )
        (
            self.mesh_id,
            self.geometry_id,
            self.topology_id,
            self.profile_id,
            self.source_id,
            self.source_revision,
            self.domain_id,
            self.distance_witness_id,
        ) = identities
        self.witness_id = canonical_fingerprint(
            {
                "kind": "restricted-delaunay-boundary-witness",
                "identities": identities,
            }
        )


@final
class RestrictedDelaunayCoverageTransportWitness(StrictModule, NonTrainableState):
    """Flattened exact transition chain preserving a restricted source degree."""

    __strict_contract__ = True

    root_witness: RestrictedDelaunayBoundaryWitness
    root_distance_witness: SourceBoundaryDistanceWitness
    transitions: tuple[CellGeometryTransition, ...]
    stage_cell_regions: tuple[np.ndarray, ...]
    target_distance_witness_id: Identifier = eqx.field(static=True)
    target_mesh_id: Identifier = eqx.field(static=True)
    target_geometry_id: Identifier = eqx.field(static=True)
    target_topology_id: Identifier = eqx.field(static=True)
    witness_id: Identifier = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        root_witness: RestrictedDelaunayBoundaryWitness,
        root_distance_witness: SourceBoundaryDistanceWitness,
        transitions: tuple[CellGeometryTransition, ...],
        stage_cell_regions: tuple[ArrayLike, ...],
        target_distance_witness_id: str,
    ) -> None:
        if not isinstance(root_witness, RestrictedDelaunayBoundaryWitness):
            raise TypeError("root_witness must be RestrictedDelaunayBoundaryWitness.")
        if not isinstance(root_distance_witness, SourceBoundaryDistanceWitness):
            raise TypeError(
                "root_distance_witness must be SourceBoundaryDistanceWitness."
            )
        transition_chain = tuple(transitions)
        if not transition_chain or any(
            not isinstance(value, CellGeometryTransition) for value in transition_chain
        ):
            raise TypeError(
                "transitions must be a nonempty tuple of CellGeometryTransition values."
            )
        distance_id = canonical_identifier(
            target_distance_witness_id, "target_distance_witness_id"
        )
        if root_witness.distance_witness_id != root_distance_witness.witness_id:
            raise ValueError(
                "Root restricted and distance witnesses bind different source-distance records."
            )
        first = transition_chain[0]
        if (
            first.source_mesh.mesh_id != root_witness.mesh_id
            or first.source_mesh.topology_id != root_witness.topology_id
            or first.source_geometry_id != root_witness.geometry_id
        ):
            raise ValueError(
                "First transition source endpoint differs from the retained direct source witness."
            )
        for previous, current in zip(
            transition_chain, transition_chain[1:], strict=False
        ):
            if (
                previous.target_mesh.mesh_id != current.source_mesh.mesh_id
                or previous.target_topology_id != current.source_topology_id
                or previous.target_geometry_id != current.source_geometry_id
            ):
                raise ValueError("Transition chain endpoints are not exactly contiguous.")
        endpoint_meshes = (
            first.source_mesh,
            *(value.target_mesh for value in transition_chain),
        )
        raw_regions = tuple(stage_cell_regions)
        if len(raw_regions) != len(endpoint_meshes):
            raise ValueError(
                "stage_cell_regions must retain one dense assignment for every transition endpoint."
            )
        immutable_regions: list[np.ndarray] = []
        for index, (regions, mesh) in enumerate(
            zip(raw_regions, endpoint_meshes, strict=True)
        ):
            raw = np.asarray(regions)
            cell_count = sum(block.cell_count for block in mesh.blocks)
            if (
                raw.shape != (cell_count,)
                or raw.dtype.kind not in "iu"
                or np.any(raw != 0)
            ):
                raise ValueError(
                    f"Transition stage {index} must retain the single interior material assignment."
                )
            copied = np.array(raw, dtype=np.int64, copy=True)
            copied.setflags(write=False)
            immutable_regions.append(copied)
        self.root_witness = root_witness
        self.root_distance_witness = root_distance_witness
        self.transitions = transition_chain
        self.stage_cell_regions = tuple(immutable_regions)
        self.target_distance_witness_id = distance_id
        last = transition_chain[-1]
        self.target_mesh_id = last.target_mesh.mesh_id
        self.target_geometry_id = last.target_geometry_id
        self.target_topology_id = last.target_topology_id
        self.witness_id = canonical_fingerprint(
            {
                "kind": "restricted-delaunay-coverage-transport-witness",
                "root": root_witness.witness_id,
                "root_distance": root_distance_witness.witness_id,
                "transitions": tuple(value.transition_id for value in transition_chain),
                "stage_regions": tuple(
                    array_tree_fingerprint(value) for value in immutable_regions
                ),
                "target_distance": distance_id,
            }
        )


type RestrictedDelaunayCoverageWitness = (
    RestrictedDelaunayBoundaryWitness | RestrictedDelaunayCoverageTransportWitness
)


class RestrictedDelaunayCoverageProof(NamedTuple):
    witness_id: str
    degree_id: str
    source_facet_count: int
    target_facet_count: int
    covered_source_facet_count: int
    region_ids: tuple[str, ...]
    facet_overlaps: tuple[FacetOverlap, ...] = ()


def _unavailable(
    check: str,
    detail: str,
    /,
    *,
    status: Literal["violated", "unresolved"] = "unresolved",
) -> RestrictedDelaunayCoverageUnavailable:
    return RestrictedDelaunayCoverageUnavailable(check, detail, status=status)


def _fraction_det(
    a: tuple[Fraction, Fraction, Fraction],
    b: tuple[Fraction, Fraction, Fraction],
    c: tuple[Fraction, Fraction, Fraction],
) -> Fraction:
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def _contains_point(tetra: np.ndarray, points: np.ndarray, point: np.ndarray) -> bool:
    vertices = [
        tuple(Fraction(float(value)) for value in points[int(row)]) for row in tetra
    ]
    query = tuple(Fraction(float(value)) for value in point)
    origin = vertices[0]
    subtract = lambda a, b: tuple(x - y for x, y in zip(a, b, strict=True))
    first, second, third = (subtract(value, origin) for value in vertices[1:])
    relative = subtract(query, origin)
    denominator = _fraction_det(first, second, third)
    if denominator == 0:
        raise _unavailable(
            "restricted_target_degeneracy",
            "A target tetrahedron is exactly degenerate.",
            status="violated",
        )
    coordinates = (
        _fraction_det(relative, second, third) / denominator,
        _fraction_det(first, relative, third) / denominator,
        _fraction_det(first, second, relative) / denominator,
    )
    weights = (Fraction(1) - sum(coordinates), *coordinates)
    return all(value >= 0 for value in weights)


def _outward_add(
    first: tuple[float, float], second: tuple[float, float]
) -> tuple[float, float]:
    if first == (0.0, 0.0):
        return second
    if second == (0.0, 0.0):
        return first
    return (
        float(np.nextafter(first[0] + second[0], -np.inf)),
        float(np.nextafter(first[1] + second[1], np.inf)),
    )


def _outward_sub(
    first: tuple[float, float], second: tuple[float, float]
) -> tuple[float, float]:
    if second == (0.0, 0.0):
        return first
    lower = float(np.nextafter(first[0] - second[1], -np.inf))
    upper = float(np.nextafter(first[1] - second[0], np.inf))
    return lower, upper


def _outward_square(interval: tuple[float, float], /) -> tuple[float, float]:
    lower, upper = interval
    if lower <= 0.0 <= upper:
        minimum = 0.0
        maximum = max(lower * lower, upper * upper)
    else:
        first, second = lower * lower, upper * upper
        minimum, maximum = min(first, second), max(first, second)
    outward_lower = 0.0 if minimum == 0.0 else float(np.nextafter(minimum, -np.inf))
    outward_upper = float(np.nextafter(maximum, np.inf))
    return outward_lower, outward_upper


def _outward_sqrt(interval: tuple[float, float], /) -> tuple[float, float]:
    lower, upper = interval
    if upper < 0.0:
        return (-np.inf, np.inf)
    lower = max(0.0, lower)
    upper = max(0.0, upper)
    root_lower = math.sqrt(lower)
    root_upper = math.sqrt(upper)
    outward_lower = 0.0 if lower == 0.0 else float(np.nextafter(root_lower, -np.inf))
    outward_upper = (
        np.inf if math.isinf(root_upper) else float(np.nextafter(root_upper, np.inf))
    )
    return outward_lower, outward_upper


def _outward_reciprocal(interval: tuple[float, float], /) -> tuple[float, float]:
    lower, upper = interval
    if lower <= 0.0 <= upper:
        return (-np.inf, np.inf)
    reciprocal_lower = 1.0 / upper
    reciprocal_upper = 1.0 / lower
    return (
        float(np.nextafter(reciprocal_lower, -np.inf)),
        float(np.nextafter(reciprocal_upper, np.inf)),
    )


def _outward_divide(
    numerator: tuple[float, float], denominator: tuple[float, float], /
) -> tuple[float, float]:
    return _outward_mul(numerator, _outward_reciprocal(denominator))


def _ring_torus_gradient_intervals(
    profile: AnalyticImplicitProfile,
    lower: np.ndarray,
    upper: np.ndarray,
    /,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    center = tuple(float(value) for value in profile.center)
    relative_x = _outward_sub((float(lower[0]), float(upper[0])), (center[0], center[0]))
    relative_y = _outward_sub((float(lower[1]), float(upper[1])), (center[1], center[1]))
    relative_z = _outward_sub((float(lower[2]), float(upper[2])), (center[2], center[2]))
    radial = _outward_sqrt(
        _outward_add(
            _outward_square(relative_x),
            _outward_square(relative_y),
        )
    )
    radial_offset = _outward_sub(
        radial, (float(profile.major_radius), float(profile.major_radius))
    )
    tube_distance = _outward_sqrt(
        _outward_add(
            _outward_square(radial_offset),
            _outward_square(relative_z),
        )
    )
    if radial[0] <= 0.0 or tube_distance[0] <= 0.0:
        return ((-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf))
    radial_gradient = _outward_divide(radial_offset, tube_distance)
    axial_gradient = _outward_divide(relative_z, tube_distance)
    radial_x = _outward_divide(relative_x, radial)
    radial_y = _outward_divide(relative_y, radial)
    return (
        _outward_mul(radial_gradient, radial_x),
        _outward_mul(radial_gradient, radial_y),
        axial_gradient,
    )


def _outward_mul(
    first: tuple[float, float], second: tuple[float, float]
) -> tuple[float, float]:
    if first == (0.0, 0.0) or second == (0.0, 0.0):
        return (0.0, 0.0)
    products = (
        first[0] * second[0],
        first[0] * second[1],
        first[1] * second[0],
        first[1] * second[1],
    )
    if not all(math.isfinite(value) for value in products):
        return (-np.inf, np.inf)
    return (
        float(np.nextafter(min(products), -np.inf)),
        float(np.nextafter(max(products), np.inf)),
    )


def _fraction_float_lower(value: Fraction, /) -> float:
    try:
        rounded = float(value)
    except OverflowError:
        return float(np.finfo(np.float64).max) if value > 0 else -np.inf
    if rounded == np.inf:
        return float(np.finfo(np.float64).max)
    if rounded == -np.inf:
        return -np.inf
    return (
        float(np.nextafter(rounded, -np.inf))
        if Fraction.from_float(rounded) > value
        else rounded
    )


def _fraction_float_upper(value: Fraction, /) -> float:
    try:
        rounded = float(value)
    except OverflowError:
        return np.inf if value > 0 else -float(np.finfo(np.float64).max)
    if rounded == np.inf:
        return np.inf
    if rounded == -np.inf:
        return -float(np.finfo(np.float64).max)
    return (
        float(np.nextafter(rounded, np.inf))
        if Fraction.from_float(rounded) < value
        else rounded
    )


def _fraction_point(point: ArrayLike, /) -> ReferencePoint3:
    values = np.asarray(point, dtype=np.float64)
    if values.shape != (3,) or not np.all(np.isfinite(values)):
        raise _unavailable(
            "restricted_target_geometry",
            "A target boundary vertex is not a finite physical point.",
            status="violated",
        )
    return (
        Fraction.from_float(float(values[0])),
        Fraction.from_float(float(values[1])),
        Fraction.from_float(float(values[2])),
    )


def _subtract3(first: ReferencePoint3, second: ReferencePoint3, /) -> ReferencePoint3:
    return (
        first[0] - second[0],
        first[1] - second[1],
        first[2] - second[2],
    )


def _cross3(first: ReferencePoint3, second: ReferencePoint3, /) -> ReferencePoint3:
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


def _dot3(first: ReferencePoint3, second: ReferencePoint3, /) -> Fraction:
    return sum(
        (left * right for left, right in zip(first, second, strict=True)), Fraction(0)
    )


def _canonical_analytic_profile(
    profile: AnalyticImplicitProfile, /
) -> AnalyticImplicitProfile:
    try:
        profile.validate_source_integrity()
        renewed = AnalyticImplicitProfile(
            profile.geometry,
            profile.coordinate_contract,
            tube_radius=profile.tube_radius,
            cover_radius=profile.cover_radius,
            source_id=profile.source_id,
            source_revision=profile.source_revision,
        )
    except (TypeError, ValueError, IndexError, ArithmeticError) as error:
        raise _unavailable(
            "analytic_profile_integrity",
            f"Analytic profile does not reconstruct from its nominal source: {error}",
            status="violated",
        ) from error
    if renewed.profile_id != profile.profile_id or renewed.domain_id != profile.domain_id:
        raise _unavailable(
            "analytic_profile_integrity",
            "Analytic profile fields differ from their canonical source reconstruction.",
            status="violated",
        )
    return renewed


def _canonical_cell_mesh(mesh: CellMesh, /) -> CellMesh:
    try:
        entity_ids = {
            dimension: np.asarray(entity.entity_ids, dtype=np.int64)
            for dimension, entity in enumerate(mesh.topology.entity_sets)
            if 0 < dimension < mesh.topological_dimension
        }
        renewed = CellMesh(
            mesh.coordinates,
            mesh.blocks,
            vertex_global_ids=mesh.vertex_global_ids,
            entity_global_ids=entity_ids,
            numeric_version=mesh.numeric_version,
            periodic_topology=mesh.periodic_topology,
        )
    except (TypeError, ValueError, IndexError, ArithmeticError) as error:
        raise _unavailable(
            "restricted_mesh_identity",
            f"Target mesh does not reconstruct from its canonical topology and coordinates: {error}",
            status="violated",
        ) from error
    if (
        renewed.mesh_id != mesh.mesh_id
        or renewed.topology_id != mesh.topology_id
        or renewed.geometry_id != mesh.geometry_id
    ):
        raise _unavailable(
            "restricted_mesh_identity",
            "Target mesh content differs from its retained mesh, topology, or geometry identity.",
            status="violated",
        )
    return renewed


def _triangle_box(
    triangle: tuple[ReferencePoint3, ReferencePoint3, ReferencePoint3], /
) -> tuple[np.ndarray, np.ndarray]:
    lower = np.asarray(
        [
            _fraction_float_lower(min(point[axis] for point in triangle))
            for axis in range(3)
        ],
        dtype=np.float64,
    )
    upper = np.asarray(
        [
            _fraction_float_upper(max(point[axis] for point in triangle))
            for axis in range(3)
        ],
        dtype=np.float64,
    )
    return lower, upper


def _triangle_children(
    triangle: tuple[ReferencePoint3, ReferencePoint3, ReferencePoint3], /
) -> tuple[tuple[ReferencePoint3, ReferencePoint3, ReferencePoint3], ...]:
    first, second, third = triangle
    midpoint = lambda left, right: tuple(
        (a + b) / 2 for a, b in zip(left, right, strict=True)
    )
    first_second = midpoint(first, second)
    second_third = midpoint(second, third)
    third_first = midpoint(third, first)
    return (
        (first, first_second, third_first),
        (first_second, second, second_third),
        (third_first, second_third, third),
        (first_second, second_third, third_first),
    )


def _face_normal_interval(
    profile: AnalyticImplicitProfile,
    triangle: tuple[ReferencePoint3, ReferencePoint3, ReferencePoint3],
    outward_normal: ReferencePoint3,
    /,
) -> tuple[float, float]:
    if profile.family == "sphere":
        center: ReferencePoint3 = (
            Fraction.from_float(float(profile.center[0])),
            Fraction.from_float(float(profile.center[1])),
            Fraction.from_float(float(profile.center[2])),
        )
        projections = tuple(
            _dot3(outward_normal, _subtract3(point, center)) for point in triangle
        )
        if any(value <= 0 for value in projections):
            raise _unavailable(
                "restricted_projection_orientation",
                "A complete spherical target face is tangent to or reverses the analytic source normal.",
                status="violated",
            )
        maximum_squared_distance = max(
            sum(
                ((point[axis] - center[axis]) ** 2 for axis in range(3)),
                Fraction(0),
            )
            for point in triangle
        )
        distance_upper = float(
            np.nextafter(
                math.sqrt(_fraction_float_upper(maximum_squared_distance)),
                np.inf,
            )
        )
        if not math.isfinite(distance_upper) or distance_upper <= 0.0:
            raise _unavailable(
                "restricted_projection_transversality",
                "Exact spherical face-normal separation is outside binary64 resolution.",
            )
        lower = _fraction_float_lower(
            min(projections) / Fraction.from_float(distance_upper)
        )
        if lower <= 0.0:
            raise _unavailable(
                "restricted_projection_transversality",
                "Exact spherical face-normal separation is below binary64 interval resolution.",
            )
        return lower, np.inf
    if profile.family != "ring-torus":
        raise _unavailable(
            "analytic_projection_gradient",
            "Whole-face projection bounds are supported only for sphere and full ring-torus profiles.",
        )
    if not profile.field_bounds.gradient_rigorous:
        raise _unavailable(
            "analytic_projection_gradient",
            "A rigorous analytic gradient enclosure is required on every target boundary face.",
        )
    lower, upper = _triangle_box(triangle)
    gradient_intervals = _ring_torus_gradient_intervals(profile, lower, upper)
    if any(
        math.isnan(component_lower)
        or math.isnan(component_upper)
        or component_lower > component_upper
        for component_lower, component_upper in gradient_intervals
    ):
        raise _unavailable(
            "analytic_projection_gradient",
            "Whole-face analytic torus gradient interval is malformed.",
        )
    total = (0.0, 0.0)
    for axis, gradient_interval in enumerate(gradient_intervals):
        normal_interval = (
            _fraction_float_lower(outward_normal[axis]),
            _fraction_float_upper(outward_normal[axis]),
        )
        term = _outward_mul(gradient_interval, normal_interval)
        total = _outward_add(total, term)
    return total


def _prove_face_projection_orientation(
    profile: AnalyticImplicitProfile,
    triangle: tuple[ReferencePoint3, ReferencePoint3, ReferencePoint3],
    outward_normal: ReferencePoint3,
    face_id: int,
    /,
) -> tuple[int, ...]:
    pending = [triangle]
    leaf_count = 0
    while pending:
        current = pending.pop()
        derivative = _face_normal_interval(profile, current, outward_normal)
        if derivative[0] > 0.0:
            leaf_count += 1
            continue
        if derivative[1] <= 0.0:
            raise _unavailable(
                "restricted_projection_orientation",
                f"Target boundary face {face_id} is tangent to or reverses the analytic source normal.",
                status="violated",
            )
        parent_lower, parent_upper = _triangle_box(current)
        children = _triangle_children(current)
        if any(
            len(set(child)) != 3
            or all(child[index] == current[index] for index in range(3))
            for child in children
        ):
            raise _unavailable(
                "restricted_projection_transversality",
                f"Target boundary face {face_id} remains unresolved at exact subdivision resolution.",
            )
        child_boxes = tuple(_triangle_box(child) for child in children)
        if all(
            np.array_equal(child_lower, parent_lower)
            and np.array_equal(child_upper, parent_upper)
            for child_lower, child_upper in child_boxes
        ):
            raise _unavailable(
                "restricted_projection_transversality",
                f"Target boundary face {face_id} has no further representable interval subdivision.",
            )
        pending.extend(children)
    return (leaf_count,)


def _triangle_barycentric_line(
    point: ReferencePoint3,
    direction: ReferencePoint3,
    triangle: tuple[ReferencePoint3, ReferencePoint3, ReferencePoint3],
    /,
) -> tuple[
    tuple[Fraction, Fraction], tuple[Fraction, Fraction], tuple[Fraction, Fraction]
]:
    first, second, third = triangle
    edge_first = _subtract3(second, first)
    edge_second = _subtract3(third, first)
    coordinate_pair = next(
        (
            (left, right)
            for left, right in combinations(range(3), 2)
            if edge_first[left] * edge_second[right]
            - edge_first[right] * edge_second[left]
        ),
        None,
    )
    if coordinate_pair is None:
        raise _unavailable(
            "restricted_target_boundary",
            "A target boundary triangle has zero exact area.",
            status="violated",
        )
    left, right = coordinate_pair
    determinant = (
        edge_first[left] * edge_second[right] - edge_first[right] * edge_second[left]
    )
    relative = _subtract3(point, first)

    def coordinates(value: ReferencePoint3) -> tuple[Fraction, Fraction]:
        return (
            (value[left] * edge_second[right] - value[right] * edge_second[left])
            / determinant,
            (edge_first[left] * value[right] - edge_first[right] * value[left])
            / determinant,
        )

    alpha, beta = coordinates(relative)
    alpha_rate, beta_rate = coordinates(direction)
    return (
        (Fraction(1) - alpha - beta, -alpha_rate - beta_rate),
        (alpha, alpha_rate),
        (beta, beta_rate),
    )


def _line_triangle_intersection(
    point: ReferencePoint3,
    direction: ReferencePoint3,
    triangle: tuple[ReferencePoint3, ReferencePoint3, ReferencePoint3],
    tube_radius: Fraction,
    /,
) -> tuple[bool, int]:
    first, second, third = triangle
    normal = _cross3(_subtract3(second, first), _subtract3(third, first))
    if normal == (Fraction(0), Fraction(0), Fraction(0)):
        raise _unavailable(
            "restricted_target_boundary",
            "A target boundary triangle has zero exact area.",
            status="violated",
        )
    denominator = _dot3(normal, direction)
    numerator = _dot3(normal, _subtract3(first, point))
    if denominator == 0:
        if numerator != 0:
            return False, 0
        lower, upper = -tube_radius, tube_radius
        for intercept, slope in _triangle_barycentric_line(point, direction, triangle):
            if slope == 0:
                if intercept < 0:
                    return False, 0
            else:
                crossing = -intercept / slope
                if slope > 0:
                    lower = max(lower, crossing)
                else:
                    upper = min(upper, crossing)
        if lower > upper:
            return False, 0
        if lower == upper and abs(lower) == tube_radius:
            raise _unavailable(
                "analytic_projection_tube",
                "A target face touches the endpoint of the strict normal-tube segment.",
                status="violated",
            )
        return True, 0
    parameter = numerator / denominator
    if parameter < -tube_radius or parameter > tube_radius:
        return False, 0
    barycentric = tuple(
        value[0] + parameter * value[1]
        for value in _triangle_barycentric_line(point, direction, triangle)
    )
    if any(value < 0 for value in barycentric):
        return False, 0
    if abs(parameter) == tube_radius:
        raise _unavailable(
            "analytic_projection_tube",
            "A target face touches the endpoint of the strict normal-tube segment.",
            status="violated",
        )
    if any(value == 0 for value in barycentric):
        return True, 0
    if sum(barycentric, Fraction(0)) != 1:
        raise _unavailable(
            "restricted_projection_fiber",
            "Exact normal-fiber intersection arithmetic failed.",
            status="violated",
        )
    return False, 1


def _sphere_or_torus_anchors(
    profile: AnalyticImplicitProfile, /
) -> Iterable[tuple[ReferencePoint3, ReferencePoint3, tuple[Fraction, Fraction]]]:
    center = tuple(Fraction.from_float(float(value)) for value in profile.center)
    radius = Fraction.from_float(profile.radius)
    major = Fraction.from_float(profile.major_radius)
    denominator = 1
    while True:
        rational_grid = tuple(
            Fraction(numerator, denominator)
            for numerator in range(-denominator, denominator + 1)
        )
        for first_parameter in rational_grid:
            for second_parameter in rational_grid:
                if profile.family == "sphere":
                    divisor = 1 + first_parameter**2 + second_parameter**2
                    normal = (
                        2 * first_parameter / divisor,
                        2 * second_parameter / divisor,
                        (1 - first_parameter**2 - second_parameter**2) / divisor,
                    )
                    point = tuple(
                        center[axis] + radius * normal[axis] for axis in range(3)
                    )
                else:
                    first_divisor = 1 + first_parameter**2
                    second_divisor = 1 + second_parameter**2
                    cosine_longitude = (1 - first_parameter**2) / first_divisor
                    sine_longitude = 2 * first_parameter / first_divisor
                    cosine_latitude = (1 - second_parameter**2) / second_divisor
                    sine_latitude = 2 * second_parameter / second_divisor
                    radial = major + radius * cosine_latitude
                    normal = (
                        cosine_latitude * cosine_longitude,
                        cosine_latitude * sine_longitude,
                        sine_latitude,
                    )
                    point = (
                        center[0] + radial * cosine_longitude,
                        center[1] + radial * sine_longitude,
                        center[2] + radius * sine_latitude,
                    )
                if _dot3(normal, normal) != 1:
                    raise _unavailable(
                        "analytic_projection_anchor",
                        "Canonical rational source normal is not exactly unit length.",
                        status="violated",
                    )
                yield point, normal, (first_parameter, second_parameter)
        denominator += 1


def _regular_normal_fiber_count(
    faces: tuple[tuple[ReferencePoint3, ReferencePoint3, ReferencePoint3], ...],
    anchor: ReferencePoint3,
    normal: ReferencePoint3,
    tube_radius: Fraction,
    /,
) -> tuple[bool, int]:
    """Count the entire closed conservative normal-tube segment exactly.

    Edge, vertex, coplanar, or tube-endpoint contacts mark the candidate
    source value nonregular; they are never treated as a regular preimage.
    """
    count = 0
    for triangle in faces:
        degenerate, intersections = _line_triangle_intersection(
            anchor, normal, triangle, tube_radius
        )
        if degenerate:
            return True, 0
        count += intersections
    return False, count


def _revalidate_oriented_surface_topology(
    faces: tuple[tuple[int, int, int], ...],
    expected_euler: int,
    /,
) -> tuple[set[int], dict[tuple[int, int], list[int]], int, int]:
    if not faces:
        raise _unavailable(
            "restricted_target_boundary",
            "An analytic target projection needs a nonempty oriented boundary.",
            status="violated",
        )
    surface_vertices: set[int] = set()
    edge_faces: dict[tuple[int, int], list[int]] = {}
    edge_signs: dict[tuple[int, int], list[int]] = {}
    vertex_links: dict[int, list[tuple[int, int]]] = {}
    seen_faces: set[tuple[int, int, int]] = set()
    for face_index, face in enumerate(faces):
        if len(set(face)) != 3:
            raise _unavailable(
                "restricted_target_boundary",
                "A target boundary triangle repeats a vertex row.",
                status="violated",
            )
        ordered_face = sorted(face)
        key_face = (ordered_face[0], ordered_face[1], ordered_face[2])
        if key_face in seen_faces:
            raise _unavailable(
                "restricted_target_boundary",
                "Target boundary contains duplicate vertex faces.",
                status="violated",
            )
        seen_faces.add(key_face)
        surface_vertices.update(face)
        for edge_position in range(3):
            first = face[edge_position]
            second = face[(edge_position + 1) % 3]
            key = (min(first, second), max(first, second))
            edge_faces.setdefault(key, []).append(face_index)
            edge_signs.setdefault(key, []).append(1 if first < second else -1)
        for vertex_position, vertex in enumerate(face):
            others = tuple(face[index] for index in range(3) if index != vertex_position)
            vertex_links.setdefault(vertex, []).append((others[0], others[1]))

    if any(
        len(incident) != 2 or sum(edge_signs[edge]) != 0
        for edge, incident in edge_faces.items()
    ):
        raise _unavailable(
            "restricted_boundary_orientation",
            "Target boundary edges must have exactly two oppositely oriented incident faces.",
            status="violated",
        )
    for vertex, link_edges in vertex_links.items():
        link: dict[int, set[int]] = {}
        for first, second in link_edges:
            if first == second or second in link.get(first, set()):
                raise _unavailable(
                    "restricted_boundary_vertex_link",
                    f"Target boundary vertex {vertex} has a repeated or degenerate link edge.",
                    status="violated",
                )
            link.setdefault(first, set()).add(second)
            link.setdefault(second, set()).add(first)
        if not link or any(len(neighbors) != 2 for neighbors in link.values()):
            raise _unavailable(
                "restricted_boundary_vertex_link",
                f"Target boundary vertex {vertex} does not have a cyclic 2-manifold link.",
                status="violated",
            )
        reached = {next(iter(link))}
        pending_vertices = list(reached)
        while pending_vertices:
            current = pending_vertices.pop()
            for neighbor in link[current] - reached:
                reached.add(neighbor)
                pending_vertices.append(neighbor)
        if reached != set(link):
            raise _unavailable(
                "restricted_boundary_vertex_link",
                f"Target boundary vertex {vertex} has multiple disconnected link cycles.",
                status="violated",
            )

    face_neighbors: list[set[int]] = [set() for _ in faces]
    for incident in edge_faces.values():
        first, second = incident
        face_neighbors[first].add(second)
        face_neighbors[second].add(first)
    remaining = set(range(len(faces)))
    components = 0
    while remaining:
        components += 1
        pending_faces = [remaining.pop()]
        while pending_faces:
            current = pending_faces.pop()
            for neighbor in face_neighbors[current] & remaining:
                remaining.remove(neighbor)
                pending_faces.append(neighbor)
    euler = len(surface_vertices) - len(edge_faces) + len(faces)
    if components != 1 or euler != expected_euler:
        raise _unavailable(
            "restricted_boundary_topology",
            "Target boundary components and Euler characteristic differ from the connected analytic sphere or torus.",
            status="violated",
        )
    return surface_vertices, edge_faces, components, euler


def _prove_degree_one_normal_fiber(
    profile: AnalyticImplicitProfile,
    faces: tuple[tuple[ReferencePoint3, ReferencePoint3, ReferencePoint3], ...],
    tube_radius: Fraction,
    /,
) -> tuple[ReferencePoint3, ReferencePoint3, tuple[Fraction, Fraction]]:
    # The rational parameter grid is dense in an open source patch. The finite
    # projected edge/vertex set cannot cover that patch, so exact degeneracy
    # checks terminate without a numerical search budget or arbitrary cap.
    for anchor, normal, parameters in _sphere_or_torus_anchors(profile):
        degenerate, fiber_count = _regular_normal_fiber_count(
            faces, anchor, normal, tube_radius
        )
        if degenerate:
            continue
        if fiber_count != 1:
            raise _unavailable(
                "restricted_projection_degree",
                f"A regular complete normal-tube fiber has {fiber_count} target preimages, not degree one.",
                status="violated",
            )
        return anchor, normal, parameters
    raise _unavailable(
        "restricted_projection_regular_fiber",
        "No exact regular normal fiber was found in the dense rational source-anchor set.",
    )


def validate_restricted_delaunay_boundary_witness(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    profile: AnalyticImplicitProfile,
    cell_regions: ArrayLike,
    distance_witness: SourceBoundaryDistanceWitness,
    witness: RestrictedDelaunayBoundaryWitness,
    /,
    *,
    embedding: GlobalEmbeddingCertificate | None = None,
) -> RestrictedDelaunayCoverageProof:
    """Prove degree-one source coverage by exact tubular projection.

    The nominal profile contract admits only one connected signed-distance
    sphere or full ring torus. In its strict reach tube, nearest-point
    projection to the analytic zero set is smooth. The replay proves that the
    target is an embedded oriented closed 2-manifold, every complete outward
    face is strictly transverse with positive projection orientation, every
    edge and vertex link has the exact manifold incidence, and a regular
    source normal fiber has exactly one preimage. Face/edge/vertex positivity
    makes this a proper positive branched covering; degree one removes every
    branch and duplicate/missing component. The retained witness stores only
    input identities; all topology and degree evidence is recomputed. A
    prepared ``embedding`` computed for exactly this mesh, coordinate map and
    local-validity proof (no junctions) replaces recomputing that theorem.
    """
    from ...discretization._cell_geometry_validity import (
        certify_cell_geometry_validity,
    )
    from .._embedding_cells import _iter_embedding_cells, UnsupportedEmbeddingMap
    from .._global_embedding import (
        certify_global_embedding,
        GlobalEmbeddingCertificate as EmbeddingCertificate,
    )
    from .._source_fidelity import _analytic_profile_distance_evidence

    if not isinstance(mesh, CellMesh) or not isinstance(geometry, CellGeometrySpec):
        raise TypeError("Restricted coverage requires CellMesh and CellGeometrySpec.")
    if not isinstance(profile, AnalyticImplicitProfile):
        raise TypeError(
            "Restricted coverage requires AnalyticImplicitProfile source authority."
        )
    if not isinstance(distance_witness, SourceBoundaryDistanceWitness):
        raise TypeError("distance_witness must be SourceBoundaryDistanceWitness.")
    if not isinstance(witness, RestrictedDelaunayBoundaryWitness):
        raise TypeError("witness must be RestrictedDelaunayBoundaryWitness.")
    if embedding is not None and not isinstance(embedding, EmbeddingCertificate):
        raise TypeError("embedding must be GlobalEmbeddingCertificate or None.")

    profile = _canonical_analytic_profile(profile)
    mesh = _canonical_cell_mesh(mesh)
    identities = (
        mesh.mesh_id,
        cell_geometry_id(geometry),
        mesh.topology_id,
        profile.profile_id,
        profile.source_id,
        profile.source_revision,
        profile.domain_id,
        distance_witness.witness_id,
    )
    retained = (
        witness.mesh_id,
        witness.geometry_id,
        witness.topology_id,
        witness.profile_id,
        witness.source_id,
        witness.source_revision,
        witness.domain_id,
        witness.distance_witness_id,
    )
    expected_witness_id = canonical_fingerprint(
        {
            "kind": "restricted-delaunay-boundary-witness",
            "identities": retained,
        }
    )
    if identities != retained or witness.witness_id != expected_witness_id:
        raise _unavailable(
            "restricted_witness_binding",
            "Restricted witness does not authenticate the current mesh, profile, domain, and separate distance proof.",
            status="violated",
        )
    if (
        distance_witness.source_id,
        distance_witness.source_revision,
        distance_witness.source_query_id,
    ) != (profile.source_id, profile.source_revision, profile.profile_id):
        raise _unavailable(
            "analytic_distance_binding",
            "Distance witness binds another analytic source revision.",
            status="violated",
        )

    # The admitted exact parameter domains S^2 and T^2 are connected source
    # boundaries; no generic analytic profile inherits this theorem.
    if profile.family == "sphere":
        expected_genus, expected_euler = 0, 2
    elif profile.family == "ring-torus":
        expected_genus, expected_euler = 1, 0
    else:
        raise _unavailable(
            "analytic_material_definition",
            "Projection-degree coverage is proved only for the nominal signed-distance sphere and full ring torus.",
        )
    if (
        profile.region_ids != ("interior",)
        or profile.component_count != 1
        or profile.genus != expected_genus
        or profile.boundary_euler_characteristic != expected_euler
    ):
        raise _unavailable(
            "analytic_material_definition",
            "The retained analytic profile does not establish its one connected sphere or ring-torus interior.",
            status="violated",
        )
    if (
        mesh.topological_dimension != 3
        or mesh.ambient_dimension != 3
        or not isinstance(mesh.connectivity, TetrahedralConnectivity)
        or mesh.periodic_topology is not None
        or any(block.cell_kind != "tetrahedron" for block in mesh.blocks)
    ):
        raise _unavailable(
            "restricted_target_topology",
            "Analytic projection coverage requires a nonperiodic full-dimensional tetrahedral target.",
        )

    target_cell_count = sum(block.cell_count for block in mesh.blocks)
    regions = np.asarray(cell_regions)
    if (
        regions.shape != (target_cell_count,)
        or regions.dtype.kind not in "iu"
        or np.any(regions != 0)
    ):
        raise _unavailable(
            "analytic_material_assignment",
            "Every target tetrahedron must retain the one declared interior material assignment.",
            status="violated",
        )

    requested_radius = float(distance_witness.cover_radius)
    if not math.isfinite(requested_radius) or requested_radius <= 0.0:
        raise _unavailable(
            "analytic_source_cover",
            "Distance witness must retain a positive physical source-cover radius.",
            status="violated",
        )
    try:
        mesh_to_source_lower, mesh_to_source_upper, source_to_mesh_upper = (
            _analytic_profile_distance_evidence(mesh, geometry, profile, distance_witness)
        )
    except (TypeError, ValueError, IndexError, ArithmeticError, OverflowError) as error:
        raise _unavailable(
            "analytic_distance_premise",
            f"Analytic source distance and complete-cover evidence does not revalidate: {error}",
            status="violated",
        ) from error
    del mesh_to_source_lower
    tube_radius = Fraction.from_float(profile.reach_lower)
    distance_upper = max(mesh_to_source_upper, source_to_mesh_upper)
    if distance_upper >= tube_radius:
        raise _unavailable(
            "analytic_projection_tube",
            "The independently revalidated source-to-target distance does not lie strictly inside the established normal tube.",
        )

    try:
        validity = certify_cell_geometry_validity(geometry, mesh=mesh)
        if not validity.all_certified:
            raise _unavailable(
                "restricted_target_map",
                "Every target tetrahedron needs a certified positive affine P1 map.",
            )
        if embedding is None:
            embedding = certify_global_embedding(mesh, geometry, validity)
        else:
            embedding.require_inputs(mesh, geometry, validity)
    except RestrictedDelaunayCoverageUnavailable:
        raise
    except (
        TypeError,
        ValueError,
        IndexError,
        ArithmeticError,
        FloatingPointError,
    ) as error:
        raise _unavailable(
            "restricted_target_embedding",
            f"Target map or exact global embedding could not be revalidated: {error}",
        ) from error
    if embedding.status != "certified":
        status = "violated" if embedding.status == "violated" else "unresolved"
        detail = "; ".join(value.detail for value in embedding.findings)
        raise _unavailable(
            "restricted_target_embedding",
            detail
            or "Target cell contacts and outward boundary shells are not certified.",
            status=status,
        )

    if cell_geometry_id(geometry) != witness.geometry_id:
        raise _unavailable(
            "restricted_target_geometry",
            "Target geometry content changed while revalidating the exact P1 maps.",
            status="violated",
        )
    output_cell_rows = np.concatenate(
        tuple(np.asarray(block.vertices, dtype=np.int64) for block in mesh.blocks),
        axis=0,
    )
    if output_cell_rows.shape != (target_cell_count, 4):
        raise _unavailable(
            "restricted_target_geometry",
            "Target tetrahedron vertex rows do not align with all target cells.",
            status="violated",
        )
    target_coordinates = np.asarray(mesh.coordinates, dtype=np.float64)
    exact_vertices = tuple(_fraction_point(point) for point in target_coordinates)
    validated_cell_count = 0
    try:
        for row, cell in enumerate(_iter_embedding_cells(mesh, geometry)):
            if not cell.affine:
                raise _unavailable(
                    "restricted_target_map",
                    f"Target tetrahedron {row} has unsupported non-affine or unavailable coordinate-map evidence.",
                )
            expected = tuple(
                exact_vertices[int(vertex)] for vertex in output_cell_rows[row]
            )
            if cell.corner_images != expected:
                raise _unavailable(
                    "restricted_target_geometry",
                    f"Target tetrahedron {row} does not realize its exact mesh-coordinate vertex map.",
                    status="violated",
                )
            validated_cell_count += 1
    except UnsupportedEmbeddingMap as error:
        raise _unavailable("restricted_target_map", str(error)) from error
    if validated_cell_count != target_cell_count:
        raise _unavailable(
            "restricted_target_geometry",
            "Target physical cell maps do not align with tetrahedron ancestry.",
            status="violated",
        )

    interior_lower, interior_upper = profile.field_bounds.point_values(
        np.asarray(profile.interior_witness, dtype=np.float64).reshape((1, 3))
    )
    if (
        np.asarray(interior_lower).shape != (1,)
        or np.asarray(interior_upper).shape != (1,)
        or not np.isfinite(np.asarray(interior_lower)[0])
        or not np.isfinite(np.asarray(interior_upper)[0])
        or float(np.asarray(interior_upper)[0]) >= 0.0
    ):
        raise _unavailable(
            "analytic_interior_witness",
            "The analytic interior witness lacks a strict rigorous negative-field enclosure.",
            status="violated",
        )
    points_for_containment = np.asarray(mesh.coordinates, dtype=np.float64)
    if not any(
        _contains_point(cell, points_for_containment, profile.interior_witness)
        for cell in output_cell_rows
    ):
        raise _unavailable(
            "restricted_interior_degree",
            "The established analytic interior witness is absent from the complete target tetrahedral union.",
            status="violated",
        )

    connectivity = mesh.connectivity
    face_vertices = np.asarray(connectivity.faces, dtype=np.int64)
    boundary_mask = np.asarray(connectivity.boundary_faces, dtype=np.bool_)
    face_counts = np.asarray(connectivity.face_cell_counts, dtype=np.int64)
    cell_faces = np.asarray(connectivity.cell_faces, dtype=np.int64)
    if (
        face_vertices.ndim != 2
        or face_vertices.shape[1] != 3
        or boundary_mask.shape != (face_vertices.shape[0],)
        or face_counts.shape != boundary_mask.shape
        or cell_faces.shape != (target_cell_count, 4)
        or not np.array_equal(boundary_mask, face_counts == 1)
        or np.any(face_vertices < 0)
        or np.any(face_vertices >= target_coordinates.shape[0])
    ):
        raise _unavailable(
            "restricted_target_boundary",
            "Recomputed tetrahedral boundary incidence has inconsistent exact row shapes.",
            status="violated",
        )
    face_entity_set = mesh.entity_set(2)
    entity_ids = np.asarray(face_entity_set.entity_ids, dtype=np.int64)
    entity_boundary_mask = np.asarray(
        face_entity_set.subset("boundary").mask, dtype=np.bool_
    )
    if (
        entity_ids.shape != boundary_mask.shape
        or entity_boundary_mask.shape != boundary_mask.shape
        or not np.array_equal(entity_boundary_mask, boundary_mask)
    ):
        raise _unavailable(
            "restricted_target_boundary",
            "Target boundary face IDs do not align with the actual tetrahedral boundary subset.",
            status="violated",
        )
    boundary_rows = np.flatnonzero(boundary_mask)
    boundary_ids = entity_ids[boundary_rows]
    if (
        boundary_rows.size == 0
        or np.unique(boundary_ids).size != boundary_ids.size
        or np.any(boundary_ids < 0)
    ):
        raise _unavailable(
            "restricted_target_boundary",
            "Target analytic coverage requires a nonempty boundary with unique canonical face IDs.",
            status="violated",
        )

    owners: list[list[int]] = [[] for _ in range(face_vertices.shape[0])]
    for cell_row in range(target_cell_count):
        for local_face in range(4):
            face_index = int(cell_faces[cell_row, local_face])
            if face_index < 0 or face_index >= len(owners):
                raise _unavailable(
                    "restricted_target_boundary",
                    "A tetrahedron names an out-of-range face row.",
                    status="violated",
                )
            owners[face_index].append(cell_row)
    if any(
        len(face_owners) != (1 if boundary_mask[face_row] else 2)
        for face_row, face_owners in enumerate(owners)
    ):
        raise _unavailable(
            "restricted_target_boundary",
            "Every target face must have exactly one boundary owner or two interior owners.",
            status="violated",
        )

    surface_faces: list[tuple[int, int, int]] = []
    surface_triangles: list[tuple[ReferencePoint3, ReferencePoint3, ReferencePoint3]] = []
    face_leaf_counts: list[int] = []
    for boundary_position, face_row_raw in enumerate(boundary_rows):
        face_row = int(face_row_raw)
        raw_vertices: tuple[int, int, int] = (
            int(face_vertices[face_row, 0]),
            int(face_vertices[face_row, 1]),
            int(face_vertices[face_row, 2]),
        )
        if len(set(raw_vertices)) != 3:
            raise _unavailable(
                "restricted_target_boundary",
                "A target boundary triangle repeats a vertex row.",
                status="violated",
            )
        owner_cell = owners[face_row][0]
        triangle: tuple[ReferencePoint3, ReferencePoint3, ReferencePoint3] = (
            exact_vertices[raw_vertices[0]],
            exact_vertices[raw_vertices[1]],
            exact_vertices[raw_vertices[2]],
        )
        first_edge = _subtract3(triangle[1], triangle[0])
        second_edge = _subtract3(triangle[2], triangle[0])
        outward_normal = _cross3(first_edge, second_edge)
        if outward_normal == (Fraction(0), Fraction(0), Fraction(0)):
            raise _unavailable(
                "restricted_target_boundary",
                f"Target boundary face {int(boundary_ids[boundary_position])} is exactly degenerate.",
                status="violated",
            )
        cell_points = tuple(
            exact_vertices[int(vertex)] for vertex in output_cell_rows[owner_cell]
        )
        cell_center: ReferencePoint3 = (
            sum((cell_points[index][0] for index in range(4)), Fraction(0)) / 4,
            sum((cell_points[index][1] for index in range(4)), Fraction(0)) / 4,
            sum((cell_points[index][2] for index in range(4)), Fraction(0)) / 4,
        )
        face_center: ReferencePoint3 = (
            sum((triangle[index][0] for index in range(3)), Fraction(0)) / 3,
            sum((triangle[index][1] for index in range(3)), Fraction(0)) / 3,
            sum((triangle[index][2] for index in range(3)), Fraction(0)) / 3,
        )
        inward_measure = _dot3(outward_normal, _subtract3(cell_center, face_center))
        if inward_measure == 0:
            raise _unavailable(
                "restricted_target_boundary",
                "An exact target cell centroid is coplanar with its boundary face.",
                status="violated",
            )
        oriented_vertices = raw_vertices
        if inward_measure > 0:
            oriented_vertices = (
                raw_vertices[0],
                raw_vertices[2],
                raw_vertices[1],
            )
            triangle = (triangle[0], triangle[2], triangle[1])
            outward_normal = (
                -outward_normal[0],
                -outward_normal[1],
                -outward_normal[2],
            )
        face_id = int(boundary_ids[boundary_position])
        (leaf_count,) = _prove_face_projection_orientation(
            profile, triangle, outward_normal, face_id
        )
        face_leaf_counts.append(leaf_count)
        surface_faces.append(oriented_vertices)
        surface_triangles.append(triangle)
    surface_vertices, edge_faces, target_components, target_euler = (
        _revalidate_oriented_surface_topology(
            tuple(surface_faces), profile.boundary_euler_characteristic
        )
    )

    # The exact global-embedding replay excludes nonshared cell contacts;
    # together with each single-cycle combinatorial vertex link, this makes
    # every physical link an embedded simple loop in the unit sphere about its
    # vertex. Strict face transversality excludes both projection-kernel poles
    # from that link. Its face sectors map with positive orientation, so the
    # winding around the two poles is positive; a simple loop in the punctured
    # sphere has winding only 0 or +/-1, hence this winding is exactly +1.
    # Opposite induced edge orientations likewise put adjacent face images on
    # opposite sides of each regular projected edge. Thus projection is a
    # local homeomorphism at faces, edges, and vertices, and compactness makes
    # it a proper covering. The exact regular-fiber count proves its degree.
    selected_anchor, selected_normal, selected_parameters = (
        _prove_degree_one_normal_fiber(profile, tuple(surface_triangles), tube_radius)
    )
    degree_id = canonical_fingerprint(
        {
            "kind": "analytic-normal-projection-degree-one",
            "domain": profile.domain_id,
            "source": (profile.source_id, profile.source_revision),
            "profile_state": (profile.state_id, profile.schema_id, profile.kernel_id),
            "witness": witness.witness_id,
            "distance_witness": distance_witness.witness_id,
            "distance_upper": str(distance_upper),
            "source_family": (profile.family, profile.component_count, profile.genus),
            "target_boundary": (
                tuple(sorted(int(value) for value in boundary_ids)),
                len(surface_vertices),
                len(edge_faces),
                len(surface_faces),
                target_components,
                target_euler,
            ),
            "projection_leaves": tuple(face_leaf_counts),
            "regular_fiber": (
                tuple(str(value) for value in selected_parameters),
                tuple(str(value) for value in selected_anchor),
                tuple(str(value) for value in selected_normal),
                1,
            ),
            "embedding": embedding.certificate_id,
            "degree": 1,
        }
    )
    return RestrictedDelaunayCoverageProof(
        witness.witness_id,
        degree_id,
        0,
        len(boundary_rows),
        0,
        profile.region_ids,
    )


def _transition_mesh_ids(mesh: CellMesh, /) -> tuple[int, ...]:
    return tuple(
        int(identifier)
        for block in mesh.blocks
        for identifier in np.asarray(block.global_ids, dtype=np.int64)
    )


def _transition_cell_kinds(mesh: CellMesh, /) -> dict[int, str]:
    return {
        int(identifier): block.cell_kind
        for block in mesh.blocks
        for identifier in np.asarray(block.global_ids, dtype=np.int64)
    }


def _reference_tetrahedron(
    vertices: tuple[tuple[Fraction, ...], ...], /
) -> ReferenceTetrahedron:
    if len(vertices) != 4 or any(len(vertex) != 3 for vertex in vertices):
        raise _unavailable(
            "nested_reference_partition",
            "Nested root transport requires exact three-dimensional tetrahedron corners.",
        )
    return (
        (vertices[0][0], vertices[0][1], vertices[0][2]),
        (vertices[1][0], vertices[1][1], vertices[1][2]),
        (vertices[2][0], vertices[2][1], vertices[2][2]),
        (vertices[3][0], vertices[3][1], vertices[3][2]),
    )


def _reference_tetra_volume(vertices: tuple[tuple[Fraction, ...], ...], /) -> Fraction:
    tetrahedron = _reference_tetrahedron(vertices)
    origin = tetrahedron[0]
    vectors: tuple[ReferencePoint3, ReferencePoint3, ReferencePoint3] = (
        (
            tetrahedron[1][0] - origin[0],
            tetrahedron[1][1] - origin[1],
            tetrahedron[1][2] - origin[2],
        ),
        (
            tetrahedron[2][0] - origin[0],
            tetrahedron[2][1] - origin[1],
            tetrahedron[2][2] - origin[2],
        ),
        (
            tetrahedron[3][0] - origin[0],
            tetrahedron[3][1] - origin[1],
            tetrahedron[3][2] - origin[2],
        ),
    )
    signed = _fraction_det(vectors[0], vectors[1], vectors[2])
    if signed <= 0:
        raise _unavailable(
            "nested_reference_orientation",
            "A nested tetrahedron does not preserve positive reference orientation.",
            status="violated",
        )
    return signed / 6


def _require_nested_reference_partition(
    transition: CellGeometryTransition,
    /,
) -> None:
    from ...discretization._nested_reference_types import (
        exact_reference_cell_vertices,
        exact_reference_vertices,
        ReferenceCellMapWitnesses,
        validate_nested_reference_witnesses,
        validate_reference_cell_map_witnesses,
    )

    kind = transition.evidence.kind
    if kind == "nested_restriction":
        nested = transition.refinement
        source_mesh, target_mesh = transition.source_mesh, transition.target_mesh
        fine_ids_expected = set(_transition_mesh_ids(target_mesh))
        coarse_ids_expected = set(_transition_mesh_ids(source_mesh))
    elif kind == "coarsening_interpolation":
        nested = transition.coarsening
        source_mesh, target_mesh = transition.source_mesh, transition.target_mesh
        fine_ids_expected = set(_transition_mesh_ids(source_mesh))
        coarse_ids_expected = set(_transition_mesh_ids(target_mesh))
    else:
        raise _unavailable(
            "nested_reference_partition",
            f"Transition kind {kind!r} has no exact nested reference partition.",
        )

    source_kinds = _transition_cell_kinds(source_mesh)
    target_kinds = _transition_cell_kinds(target_mesh)
    if any(
        value != "tetrahedron"
        for value in (*source_kinds.values(), *target_kinds.values())
    ):
        raise _unavailable(
            "nested_reference_cell_kind",
            "Restricted analytic transport currently supports tetrahedral whole-root cells only.",
        )

    # Rows are always oriented as source-fine cell -> target/coarse cell.
    rows_by_fine: dict[int, tuple[int, ReferenceTetrahedron]] = {}
    if nested is not None:
        try:
            validate_nested_reference_witnesses(nested, dimension=3)
            nested_vertices = exact_reference_vertices(nested)
        except (TypeError, ValueError, ArithmeticError) as error:
            raise _unavailable(
                "nested_reference_partition",
                f"Nested whole-root reference witness does not revalidate: {error}",
                status="violated",
            ) from error
        for fine_id, coarse_id, vertices in zip(
            np.asarray(nested.fine_cell_ids, dtype=np.int64),
            np.asarray(nested.coarse_cell_ids, dtype=np.int64),
            nested_vertices,
            strict=True,
        ):
            fine, coarse = int(fine_id), int(coarse_id)
            if fine not in fine_ids_expected or coarse not in coarse_ids_expected:
                raise _unavailable(
                    "nested_reference_partition",
                    "Nested reference rows name cells outside their transition endpoints.",
                    status="violated",
                )
            rows_by_fine[fine] = (coarse, _reference_tetrahedron(vertices))

    reference_maps = transition.reference_map
    if isinstance(reference_maps, ReferenceCellMapWitnesses):
        map_groups = (reference_maps,)
    elif isinstance(reference_maps, tuple) and all(
        isinstance(value, ReferenceCellMapWitnesses) for value in reference_maps
    ):
        map_groups = reference_maps
    elif reference_maps is None:
        map_groups = ()
    else:
        raise _unavailable(
            "nested_reference_partition",
            "Transition reference maps have an unsupported retained representation.",
            status="violated",
        )

    def inverse_tetra_map(
        target_vertices: tuple[tuple[Fraction, ...], ...], /
    ) -> ReferenceTetrahedron:
        target = _reference_tetrahedron(target_vertices)
        origin = target[0]
        columns: tuple[ReferencePoint3, ReferencePoint3, ReferencePoint3] = (
            (
                target[1][0] - origin[0],
                target[1][1] - origin[1],
                target[1][2] - origin[2],
            ),
            (
                target[2][0] - origin[0],
                target[2][1] - origin[1],
                target[2][2] - origin[2],
            ),
            (
                target[3][0] - origin[0],
                target[3][1] - origin[1],
                target[3][2] - origin[2],
            ),
        )
        determinant = _fraction_det(columns[0], columns[1], columns[2])
        if determinant == 0:
            raise _unavailable(
                "nested_reference_partition",
                "A preserved tetrahedral reference map is singular.",
                status="violated",
            )
        source_corners: ReferenceTetrahedron = (
            (Fraction(0), Fraction(0), Fraction(0)),
            (Fraction(1), Fraction(0), Fraction(0)),
            (Fraction(0), Fraction(1), Fraction(0)),
            (Fraction(0), Fraction(0), Fraction(1)),
        )
        result: list[ReferencePoint3] = []
        for corner in source_corners:
            relative: ReferencePoint3 = (
                corner[0] - origin[0],
                corner[1] - origin[1],
                corner[2] - origin[2],
            )
            result.append(
                (
                    _fraction_det(relative, columns[1], columns[2]) / determinant,
                    _fraction_det(columns[0], relative, columns[2]) / determinant,
                    _fraction_det(columns[0], columns[1], relative) / determinant,
                )
            )
        return (result[0], result[1], result[2], result[3])

    for group in map_groups:
        try:
            validate_reference_cell_map_witnesses(group)
            exact_maps = exact_reference_cell_vertices(group)
        except (TypeError, ValueError, ArithmeticError) as error:
            raise _unavailable(
                "nested_reference_partition",
                f"Direct preserved-cell reference map does not revalidate: {error}",
                status="violated",
            ) from error
        if (
            group.source_cell_kind != "tetrahedron"
            or group.target_cell_kind != "tetrahedron"
        ):
            raise _unavailable(
                "nested_reference_cell_kind",
                "Restricted analytic transport currently supports tetrahedral preserved cells only.",
            )
        for target_id_raw, source_id_raw, target_in_source in zip(
            np.asarray(group.target_cell_ids, dtype=np.int64),
            np.asarray(group.source_cell_ids, dtype=np.int64),
            exact_maps,
            strict=True,
        ):
            target_id, source_id = int(target_id_raw), int(source_id_raw)
            if kind == "nested_restriction":
                fine, coarse = target_id, source_id
                vertices = _reference_tetrahedron(target_in_source)
            else:
                fine, coarse = source_id, target_id
                vertices = inverse_tetra_map(target_in_source)
            if fine not in fine_ids_expected or coarse not in coarse_ids_expected:
                raise _unavailable(
                    "nested_reference_partition",
                    "Direct reference maps name cells outside their transition endpoints.",
                    status="violated",
                )
            existing = rows_by_fine.get(fine)
            value = (coarse, vertices)
            if existing is not None and existing != value:
                raise _unavailable(
                    "nested_reference_partition",
                    "Nested and direct reference maps disagree on a retained cell ancestry.",
                    status="violated",
                )
            rows_by_fine[fine] = value

    if set(rows_by_fine) != fine_ids_expected:
        raise _unavailable(
            "nested_reference_partition",
            "Nested and direct reference maps do not cover every source or target fine cell.",
            status="violated",
        )
    coarse_ids = {coarse for coarse, _ in rows_by_fine.values()}
    if coarse_ids != coarse_ids_expected:
        raise _unavailable(
            "nested_reference_partition",
            "Nested and direct reference maps do not cover every target or source parent cell.",
            status="violated",
        )

    volume_by_parent: dict[int, Fraction] = {}
    for coarse_id, vertices in rows_by_fine.values():
        volume_by_parent[coarse_id] = volume_by_parent.get(
            coarse_id, Fraction(0)
        ) + _reference_tetra_volume(vertices)
    if set(volume_by_parent) != coarse_ids_expected or any(
        value != Fraction(1, 6) for value in volume_by_parent.values()
    ):
        raise _unavailable(
            "nested_reference_partition",
            "Fine tetrahedra do not exactly partition every whole-root reference tetrahedron.",
            status="violated",
        )


def validate_restricted_delaunay_coverage_transport_witness(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    profile: AnalyticImplicitProfile,
    cell_regions: ArrayLike,
    distance_witness: SourceBoundaryDistanceWitness,
    witness: RestrictedDelaunayCoverageTransportWitness,
    /,
    *,
    embedding: GlobalEmbeddingCertificate | None = None,
) -> RestrictedDelaunayCoverageProof:
    """Revalidate every exact transport stage and its analytic projection degree.

    A prepared ``embedding`` of the current target is reused by the final stage.
    """
    from ...discretization._cell_geometry_validity import cell_geometry_id

    if not isinstance(witness, RestrictedDelaunayCoverageTransportWitness):
        raise TypeError("witness must be RestrictedDelaunayCoverageTransportWitness.")
    if not isinstance(profile, AnalyticImplicitProfile) or not isinstance(
        distance_witness, SourceBoundaryDistanceWitness
    ):
        raise TypeError(
            "Transported analytic coverage requires an analytic profile and current distance witness."
        )
    profile = _canonical_analytic_profile(profile)
    mesh = _canonical_cell_mesh(mesh)
    expected_witness_id = canonical_fingerprint(
        {
            "kind": "restricted-delaunay-coverage-transport-witness",
            "root": witness.root_witness.witness_id,
            "root_distance": witness.root_distance_witness.witness_id,
            "transitions": tuple(value.transition_id for value in witness.transitions),
            "stage_regions": tuple(
                array_tree_fingerprint(value) for value in witness.stage_cell_regions
            ),
            "target_distance": witness.target_distance_witness_id,
        }
    )
    if witness.witness_id != expected_witness_id:
        raise _unavailable(
            "transport_witness_binding",
            "Transport witness content does not reproduce its retained identity.",
            status="violated",
        )
    transitions = witness.transitions
    if not transitions or len(witness.stage_cell_regions) != len(transitions) + 1:
        raise _unavailable(
            "transport_geometry_transition",
            "Transport witness must retain one material assignment per exact chain endpoint.",
            status="violated",
        )
    for previous, current in zip(transitions, transitions[1:], strict=False):
        if (
            previous.target_mesh.mesh_id != current.source_mesh.mesh_id
            or previous.target_topology_id != current.source_topology_id
            or previous.target_geometry_id != current.source_geometry_id
        ):
            raise _unavailable(
                "transport_geometry_transition",
                "Retained transition endpoints are not exactly contiguous.",
                status="violated",
            )
    if (
        profile.profile_id != witness.root_witness.profile_id
        or profile.domain_id != witness.root_witness.domain_id
        or profile.source_id != witness.root_witness.source_id
        or profile.source_revision != witness.root_witness.source_revision
    ):
        raise _unavailable(
            "transport_source_identity",
            "Transported root witness binds another analytic source domain.",
            status="violated",
        )
    if distance_witness.witness_id != witness.target_distance_witness_id:
        raise _unavailable(
            "transport_distance_binding",
            "Transport witness binds another current target distance witness.",
            status="violated",
        )
    if (
        mesh.mesh_id != witness.target_mesh_id
        or mesh.topology_id != witness.target_topology_id
        or cell_geometry_id(geometry) != witness.target_geometry_id
    ):
        raise _unavailable(
            "transport_target_binding",
            "Transported coverage witness binds another current target mesh/map.",
            status="violated",
        )
    current_regions = np.asarray(cell_regions)
    if not np.array_equal(current_regions, witness.stage_cell_regions[-1]):
        raise _unavailable(
            "transport_region_assignment",
            "Current cell-region assignments differ from the transported proof.",
            status="violated",
        )

    root_mesh = _canonical_cell_mesh(transitions[0].source_mesh)
    root_geometry = transitions[0].source_geometry
    root_regions = witness.stage_cell_regions[0]
    root_proof = validate_restricted_delaunay_boundary_witness(
        root_mesh,
        root_geometry,
        profile,
        root_regions,
        witness.root_distance_witness,
        witness.root_witness,
    )
    stage_degree_ids = [root_proof.degree_id]
    final_proof: RestrictedDelaunayCoverageProof | None = None

    for index, transition in enumerate(transitions):
        try:
            rehydrated = CellGeometryTransition.from_stored_fields(transition)
            source_mesh = _canonical_cell_mesh(transition.source_mesh)
            target_mesh = _canonical_cell_mesh(transition.target_mesh)
            source_geometry_id = cell_geometry_id(transition.source_geometry)
            target_geometry_id = cell_geometry_id(transition.target_geometry)
        except (TypeError, ValueError, IndexError, ArithmeticError) as error:
            raise _unavailable(
                "transport_geometry_transition",
                f"Retained exact geometry transition does not revalidate: {error}",
                status="violated",
            ) from error
        if (
            rehydrated.transition_id != transition.transition_id
            or source_geometry_id != transition.source_geometry_id
            or target_geometry_id != transition.target_geometry_id
        ):
            raise _unavailable(
                "transport_geometry_transition",
                "Exact geometry transition identity changed during reconstruction.",
                status="violated",
            )
        source_cells = _transition_mesh_ids(source_mesh)
        target_cells = _transition_mesh_ids(target_mesh)
        if not np.array_equal(
            witness.stage_cell_regions[index],
            np.zeros((len(source_cells),), dtype=np.int64),
        ) or not np.array_equal(
            witness.stage_cell_regions[index + 1],
            np.zeros((len(target_cells),), dtype=np.int64),
        ):
            raise _unavailable(
                "transport_region_assignment",
                "Every transition endpoint must preserve the sole interior material region.",
                status="violated",
            )
        if transition.evidence.kind in (
            "nested_restriction",
            "coarsening_interpolation",
        ):
            _require_nested_reference_partition(transition)
        elif transition.evidence.kind == "vertex_displacement":
            if (
                source_mesh.topology_id != target_mesh.topology_id
                or source_cells != target_cells
                or not np.array_equal(
                    np.asarray(source_mesh.vertex_global_ids, dtype=np.int64),
                    np.asarray(target_mesh.vertex_global_ids, dtype=np.int64),
                )
            ):
                raise _unavailable(
                    "transport_boundary_motion",
                    "Boundary-preserving vertex motion changed target topology or stable identities.",
                    status="violated",
                )
        else:
            raise _unavailable(
                "transport_kind",
                f"Transition kind {transition.evidence.kind!r} has no source-coverage transport proof.",
            )

        stage_regions = witness.stage_cell_regions[index + 1]
        if index + 1 == len(transitions):
            if (
                target_mesh.mesh_id != mesh.mesh_id
                or target_geometry_id != cell_geometry_id(geometry)
                or not np.array_equal(stage_regions, current_regions)
            ):
                raise _unavailable(
                    "transport_target_binding",
                    "Final exact transition endpoint differs from the current accepted mesh/map.",
                    status="violated",
                )
            stage_distance = distance_witness
        else:
            stage_face_set = target_mesh.entity_set(2)
            stage_boundary_mask = np.asarray(
                stage_face_set.subset("boundary").mask, dtype=np.bool_
            )
            stage_face_ids = np.asarray(stage_face_set.entity_ids, dtype=np.int64)
            if stage_boundary_mask.shape != stage_face_ids.shape:
                raise _unavailable(
                    "transport_stage_boundary",
                    "Intermediate target boundary IDs do not align with their boundary subset.",
                    status="violated",
                )
            stage_distance = SourceBoundaryDistanceWitness(
                mesh_id=target_mesh.mesh_id,
                geometry_id=target_geometry_id,
                topology_id=target_mesh.topology_id,
                source_id=profile.source_id,
                source_revision=profile.source_revision,
                source_query_id=profile.profile_id,
                sample_count=witness.root_distance_witness.sample_count,
                cover_radius=witness.root_distance_witness.cover_radius,
                target_facet_global_ids=stage_face_ids[stage_boundary_mask],
            )
        stage_witness = RestrictedDelaunayBoundaryWitness(
            mesh_id=target_mesh.mesh_id,
            geometry_id=target_geometry_id,
            topology_id=target_mesh.topology_id,
            profile_id=profile.profile_id,
            source_id=profile.source_id,
            source_revision=profile.source_revision,
            domain_id=profile.domain_id,
            distance_witness_id=stage_distance.witness_id,
        )
        stage_proof = validate_restricted_delaunay_boundary_witness(
            target_mesh,
            transition.target_geometry,
            profile,
            stage_regions,
            stage_distance,
            stage_witness,
            embedding=embedding if index + 1 == len(transitions) else None,
        )
        stage_degree_ids.append(stage_proof.degree_id)
        if index + 1 == len(transitions):
            final_proof = stage_proof

    if final_proof is None:
        raise _unavailable(
            "transport_geometry_transition",
            "Transported proof has no final degree-validated endpoint.",
            status="violated",
        )
    degree_id = canonical_fingerprint(
        {
            "kind": "analytic-normal-projection-transported-degree-one",
            "root_degree": root_proof.degree_id,
            "stage_degrees": tuple(stage_degree_ids),
            "transitions": tuple(value.transition_id for value in transitions),
            "stage_regions": tuple(
                array_tree_fingerprint(value) for value in witness.stage_cell_regions
            ),
            "target_distance": distance_witness.witness_id,
            "degree": 1,
        }
    )
    return RestrictedDelaunayCoverageProof(
        witness.witness_id,
        degree_id,
        0,
        final_proof.target_facet_count,
        0,
        profile.region_ids,
    )


def validate_restricted_delaunay_coverage_witness(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    profile: AnalyticImplicitProfile,
    cell_regions: ArrayLike,
    distance_witness: SourceBoundaryDistanceWitness,
    witness: RestrictedDelaunayCoverageWitness,
    /,
    *,
    embedding: GlobalEmbeddingCertificate | None = None,
) -> RestrictedDelaunayCoverageProof:
    """Validate the direct restricted proof or its exact transported continuation.

    ``embedding`` is an optional prepared global-embedding certificate of the
    current target; it is checked against these inputs, then reused.
    """
    if isinstance(witness, RestrictedDelaunayBoundaryWitness):
        return validate_restricted_delaunay_boundary_witness(
            mesh,
            geometry,
            profile,
            cell_regions,
            distance_witness,
            witness,
            embedding=embedding,
        )
    if isinstance(witness, RestrictedDelaunayCoverageTransportWitness):
        return validate_restricted_delaunay_coverage_transport_witness(
            mesh,
            geometry,
            profile,
            cell_regions,
            distance_witness,
            witness,
            embedding=embedding,
        )
    raise TypeError(
        "witness must be a direct or transported restricted-Delaunay coverage witness."
    )


__all__ = [
    "RestrictedDelaunayBoundaryWitness",
    "RestrictedDelaunayCoverageProof",
    "RestrictedDelaunayCoverageTransportWitness",
    "RestrictedDelaunayCoverageWitness",
    "RestrictedDelaunayCoverageUnavailable",
    "validate_restricted_delaunay_coverage_transport_witness",
    "validate_restricted_delaunay_coverage_witness",
    "validate_restricted_delaunay_boundary_witness",
]
