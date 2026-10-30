from __future__ import annotations

from fractions import Fraction
from itertools import combinations, islice
from types import SimpleNamespace
from typing import Any, NamedTuple

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..geometry._interval_enclosure import (
    interval_add,
    interval_square,
    interval_subtract,
)
from ..geometry._rational import fraction_down, fraction_up
from ..typing import checked
from ._cell_mesh import CellBlock, CellMesh
from ._exact_algebraic_geometry import coefficient_payload
from ._periodic_cell import PeriodicCell


ExactPoint3 = tuple[Fraction, Fraction, Fraction]
CoordinateSourceBank = tuple[tuple[Fraction, ...], ...]
_UNSET_DOMAIN_SOURCE = object()

_POWER_SITE_FILTER_CHUNK = 256


class _PreparedPowerSiteBank(NamedTuple):
    site_points: tuple[ExactPoint3, ...]
    site_weights: tuple[Fraction, ...]
    power_offsets: tuple[Fraction, ...]
    coordinate_lower: np.ndarray
    coordinate_upper: np.ndarray
    weight_offset_lower: np.ndarray
    weight_offset_upper: np.ndarray


def _prepare_power_site_bank(
    site_points: tuple[ExactPoint3, ...],
    site_weights: tuple[Fraction, ...],
    /,
) -> _PreparedPowerSiteBank:
    if not site_points or len(site_points) != len(site_weights):
        raise ValueError(
            "Exact power sites and weights must be nonempty aligned banks."
        )
    offsets = tuple(
        _dot(point, point) - weight
        for point, weight in zip(site_points, site_weights, strict=True)
    )
    coordinate_lower = np.asarray(
        [[fraction_down(value) for value in point] for point in site_points],
        dtype=np.float64,
    )
    coordinate_upper = np.asarray(
        [[fraction_up(value) for value in point] for point in site_points],
        dtype=np.float64,
    )
    weight_offset_lower = np.asarray(
        [fraction_down(-weight) for weight in site_weights], dtype=np.float64
    )
    weight_offset_upper = np.asarray(
        [fraction_up(-weight) for weight in site_weights], dtype=np.float64
    )
    for values in (
        coordinate_lower,
        coordinate_upper,
        weight_offset_lower,
        weight_offset_upper,
    ):
        values.setflags(write=False)
    return _PreparedPowerSiteBank(
        site_points,
        site_weights,
        offsets,
        coordinate_lower,
        coordinate_upper,
        weight_offset_lower,
        weight_offset_upper,
    )


def _site_power_value(
    point: ExactPoint3,
    point_norm: Fraction,
    site_row: int,
    bank: _PreparedPowerSiteBank,
    /,
) -> Fraction:
    return (
        point_norm
        - 2 * _dot(point, bank.site_points[site_row])
        + bank.power_offsets[site_row]
    )


def _candidate_is_global_minimum(
    point: ExactPoint3,
    equal_sites: tuple[int, ...],
    bank: _PreparedPowerSiteBank,
    /,
) -> bool:
    anchor = equal_sites[0]
    point_norm = _dot(point, point)
    anchor_power = _site_power_value(point, point_norm, anchor, bank)
    if any(
        _site_power_value(point, point_norm, site, bank) != anchor_power
        for site in equal_sites[1:]
    ):
        return False
    point_lower = np.asarray(
        [fraction_down(value) for value in point], dtype=np.float64
    )
    point_upper = np.asarray(
        [fraction_up(value) for value in point], dtype=np.float64
    )
    anchor_upper = fraction_up(anchor_power)
    active = set(equal_sites)
    for first in range(0, len(bank.site_points), _POWER_SITE_FILTER_CHUNK):
        stop = min(first + _POWER_SITE_FILTER_CHUNK, len(bank.site_points))
        site_intervals = (
            bank.coordinate_lower[first:stop],
            bank.coordinate_upper[first:stop],
        )
        delta = interval_subtract(
            (point_lower[None, :], point_upper[None, :]),
            site_intervals,
        )
        squared = interval_square(delta)
        squared_lower = np.asarray(squared[0], dtype=np.float64)
        squared_upper = np.asarray(squared[1], dtype=np.float64)
        distance_xy = interval_add(
            (squared_lower[:, 0], squared_upper[:, 0]),
            (squared_lower[:, 1], squared_upper[:, 1]),
        )
        distance = interval_add(
            distance_xy,
            (squared_lower[:, 2], squared_upper[:, 2]),
        )
        powers = interval_add(
            distance,
            (
                bank.weight_offset_lower[first:stop],
                bank.weight_offset_upper[first:stop],
            ),
        )
        lower = np.asarray(powers[0], dtype=np.float64).reshape(-1)
        for local_row, lower_bound in enumerate(lower):
            site_row = first + local_row
            if site_row in active or float(lower_bound) > anchor_upper:
                continue
            if _site_power_value(point, point_norm, site_row, bank) <= anchor_power:
                return False
    return True


def _carrier_replay_identity(carrier_mesh: CellMesh, /) -> tuple[object, ...]:
    periodic = carrier_mesh.periodic_topology
    return (
        carrier_mesh.numeric_version,
        array_tree_fingerprint(carrier_mesh.vertex_global_ids),
        tuple(
            (
                block.name,
                block.cell_kind,
                block.block_id,
                array_tree_fingerprint(block.vertices),
                array_tree_fingerprint(block.vertex_valid),
                array_tree_fingerprint(block.global_ids),
            )
            for block in carrier_mesh.blocks
        ),
        tuple(
            (
                entity_set.name,
                entity_set.intrinsic_dimension,
                entity_set.entity_set_id,
                array_tree_fingerprint(entity_set.entity_ids),
                array_tree_fingerprint(entity_set.active_mask),
                tuple(
                    (
                        subset.name,
                        subset.subset_id,
                        array_tree_fingerprint(subset.mask),
                    )
                    for subset in entity_set.subsets
                ),
            )
            for entity_set in carrier_mesh.topology.entity_sets
        ),
        None if periodic is None else periodic.periodic_topology_id,
    )


def _fraction_point(point: np.ndarray, /) -> ExactPoint3:
    return (
        Fraction(float(point[0])),
        Fraction(float(point[1])),
        Fraction(float(point[2])),
    )


class ExactPeriodicPowerSourceContext(StrictModule, NonTrainableState):
    """Retain authored periodic sites and their exact native image rows."""

    periodic_source: Any
    source_weights: jnp.ndarray
    image_owners: jnp.ndarray
    image_windings: jnp.ndarray
    weights_present: bool = eqx.field(static=True)
    exact_source_weights: tuple[Fraction, ...] = eqx.field(static=True)
    exact_image_weights: tuple[Fraction, ...] = eqx.field(static=True)
    context_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        periodic_source: Any,
        source_weights: Any,
        image_owners: Any,
        image_windings: Any,
        /,
        *,
        weights_present: bool,
    ) -> None:
        from ..meshing._periodic import PeriodicPointOrbits

        if not isinstance(periodic_source, PeriodicPointOrbits):
            raise TypeError("periodic_source must be PeriodicPointOrbits.")
        cell = periodic_source.cell
        if (
            cell.ambient_dimension != 3
            or cell.rank != 3
            or not cell.fully_periodic
        ):
            raise ValueError(
                "Exact periodic power source context requires a full 3-D lattice."
            )
        identifiers = np.asarray(periodic_source.point_ids, dtype=np.int64)
        order = np.argsort(identifiers, kind="stable")
        point_ids = identifiers[order]
        points = np.asarray(periodic_source.points, dtype=np.float64)[order]
        raw_weights = np.asarray(source_weights)
        authored_weights = np.asarray(raw_weights, dtype=np.float64)
        weights = authored_weights[order]
        owners_raw = np.asarray(image_owners)
        windings_raw = np.asarray(image_windings)
        if (
            type(weights_present) is not bool
            or raw_weights.shape != (point_ids.size,)
            or raw_weights.dtype.kind not in "iuf"
            or np.any(~np.isfinite(authored_weights))
            or owners_raw.ndim != 1
            or owners_raw.dtype.kind not in "iu"
            or windings_raw.shape != (owners_raw.size, 3)
            or windings_raw.dtype.kind not in "iu"
            or owners_raw.size == 0
            or np.any(owners_raw < 0)
            or np.any(owners_raw >= point_ids.size)
        ):
            raise ValueError("Periodic power source/image identity arrays are misaligned.")
        owners = owners_raw.astype(np.int64, copy=False)
        windings = windings_raw.astype(np.int64, copy=False)
        image_identities = tuple(
            (
                int(owners[image]),
                tuple(int(windings[image, axis]) for axis in range(3)),
            )
            for image in range(owners.size)
        )
        if len(set(image_identities)) != len(image_identities):
            raise ValueError("Periodic power image owner/winding identities must be unique.")
        exact_points = tuple(_fraction_point(row) for row in points)
        exact_weights = tuple(Fraction.from_float(float(value)) for value in weights)
        exact_lattice = tuple(
            _fraction_point(row)
            for row in np.asarray(cell.vectors, dtype=np.float64)
        )
        exact_image_points = tuple(
            tuple(
                exact_points[int(owners[image])][axis]
                + sum(
                    (
                        int(windings[image, generator])
                        * exact_lattice[generator][axis]
                        for generator in range(3)
                    ),
                    Fraction(0),
                )
                for axis in range(3)
            )
            for image in range(owners.size)
        )
        exact_image_weights = tuple(
            exact_weights[int(owners[image])] for image in range(owners.size)
        )
        expected_orbits_id = canonical_fingerprint(
            {
                "kind": "periodic-point-orbits",
                "cell": cell.cell_id,
                "point_ids": array_tree_fingerprint(point_ids),
                "points": array_tree_fingerprint(points),
            }
        )
        if expected_orbits_id != periodic_source.orbits_id:
            raise ValueError("Periodic power source identity does not reproduce its orbit bank.")
        identity = canonical_fingerprint(
            {
                "kind": "exact-periodic-power-source-context",
                "source_orbits": periodic_source.orbits_id,
                "periodic_cell": cell.cell_id,
                "point_ids": array_tree_fingerprint(point_ids),
                "source_points": array_tree_fingerprint(points),
                "weights_present": weights_present,
                "source_weights": array_tree_fingerprint(weights),
                "image_owners": array_tree_fingerprint(owners),
                "image_windings": array_tree_fingerprint(windings),
                "exact_image_points": tuple(
                    tuple(coefficient_payload(value) for value in point)
                    for point in exact_image_points
                ),
                "exact_image_weights": tuple(
                    coefficient_payload(value) for value in exact_image_weights
                ),
            }
        )
        self.periodic_source = periodic_source
        self.source_weights = jnp.asarray(authored_weights)
        self.image_owners = jnp.asarray(owners, dtype=jnp.int64)
        self.image_windings = jnp.asarray(windings, dtype=jnp.int64)
        self.weights_present = weights_present
        self.exact_source_weights = exact_weights
        self.exact_image_weights = exact_image_weights
        self.context_id = identity

    def require_image_bank(
        self,
        exact_image_points: tuple[ExactPoint3, ...],
        image_points: ArrayLike,
        image_weights: ArrayLike,
        /,
    ) -> None:
        source = self.periodic_source
        point_ids = np.asarray(source.point_ids, dtype=np.int64)
        order = np.argsort(point_ids, kind="stable")
        exact_source_points = tuple(
            _fraction_point(row)
            for row in np.asarray(source.points, dtype=np.float64)[order]
        )
        exact_lattice = tuple(
            _fraction_point(row)
            for row in np.asarray(source.cell.vectors, dtype=np.float64)
        )
        owners = np.asarray(self.image_owners, dtype=np.int64)
        windings = np.asarray(self.image_windings, dtype=np.int64)
        expected_points = tuple(
            tuple(
                exact_source_points[int(owners[image])][axis]
                + sum(
                    (
                        int(windings[image, generator])
                        * exact_lattice[generator][axis]
                        for generator in range(3)
                    ),
                    Fraction(0),
                )
                for axis in range(3)
            )
            for image in range(owners.size)
        )
        points = np.asarray(image_points, dtype=np.float64)
        weights = np.asarray(image_weights, dtype=np.float64)
        if (
            tuple(exact_image_points) != expected_points
            or not np.array_equal(
                np.asarray(expected_points, dtype=np.float64), points
            )
            or tuple(Fraction.from_float(float(value)) for value in weights)
            != self.exact_image_weights
        ):
            raise ValueError(
                "Periodic power image bank does not reproduce its authored source context."
            )

    def require_periodic_cell(self, cell: PeriodicCell, /) -> None:
        """Require the exact authored lattice used by this source context."""
        if not isinstance(cell, PeriodicCell):
            raise TypeError("cell must be PeriodicCell.")
        source_cell = self.periodic_source.cell
        if (
            cell.cell_id != source_cell.cell_id
            or not np.array_equal(
                np.asarray(cell.vectors), np.asarray(source_cell.vectors)
            )
            or not np.array_equal(
                np.asarray(cell.origin), np.asarray(source_cell.origin)
            )
            or cell.periodic_axes != source_cell.periodic_axes
        ):
            raise ValueError(
                "Periodic power geometry source context does not match the target lattice."
            )

    def require_periodic_mesh(self, mesh: CellMesh, /) -> None:
        topology = mesh.periodic_topology
        if topology is None or not isinstance(topology.cell, PeriodicCell):
            raise ValueError("Periodic power target mesh lacks its periodic quotient.")
        self.require_periodic_cell(topology.cell)

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[ExactPeriodicPowerSourceContext],
        value: ExactPeriodicPowerSourceContext,
        /,
    ) -> ExactPeriodicPowerSourceContext:
        from ..meshing._periodic import PeriodicPointOrbits

        if not isinstance(value, cls) or not isinstance(
            value.periodic_source, PeriodicPointOrbits
        ):
            raise TypeError("value must retain an exact periodic power source context.")
        source = PeriodicPointOrbits.from_stored_fields(value.periodic_source)
        renewed = cls(
            source,
            value.source_weights,
            value.image_owners,
            value.image_windings,
            weights_present=value.weights_present,
        )
        if (
            renewed.context_id != value.context_id
            or renewed.exact_source_weights != value.exact_source_weights
            or renewed.exact_image_weights != value.exact_image_weights
            or not np.array_equal(
                np.asarray(renewed.image_owners), np.asarray(value.image_owners)
            )
            or not np.array_equal(
                np.asarray(renewed.image_windings), np.asarray(value.image_windings)
            )
            or not np.array_equal(
                np.asarray(renewed.source_weights), np.asarray(value.source_weights)
            )
        ):
            raise ValueError("Stored periodic power source context does not reproduce its operands.")
        return renewed


def _sub(first: ExactPoint3, second: ExactPoint3, /) -> ExactPoint3:
    return (
        first[0] - second[0],
        first[1] - second[1],
        first[2] - second[2],
    )

def _dot(first: ExactPoint3, second: ExactPoint3, /) -> Fraction:
    return sum((a * b for a, b in zip(first, second, strict=True)), Fraction(0))


def _cross(first: ExactPoint3, second: ExactPoint3, /) -> ExactPoint3:
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


def _plane(points: tuple[ExactPoint3, ExactPoint3, ExactPoint3], /) -> tuple[ExactPoint3, Fraction] | None:
    normal = _cross(_sub(points[1], points[0]), _sub(points[2], points[0]))
    if not any(normal):
        return None
    return normal, _dot(normal, points[0])


def _power_plane(
    points: tuple[ExactPoint3, ...],
    weights: tuple[Fraction, ...],
    anchor: int,
    other: int,
    /,
) -> tuple[ExactPoint3, Fraction] | None:
    first, second = points[anchor], points[other]
    normal: ExactPoint3 = (
        2 * (second[0] - first[0]),
        2 * (second[1] - first[1]),
        2 * (second[2] - first[2]),
    )
    right = _dot(second, second) - _dot(first, first) - weights[other] + weights[anchor]
    if not any(normal):
        if right:
            raise ValueError("Equal-site witness has inconsistent represented power values.")
        return None
    return normal, right


def _solve_three(
    planes: tuple[tuple[ExactPoint3, Fraction], ...], /
) -> ExactPoint3 | None:
    matrix = [list(normal) for normal, _ in planes]
    right = [[rhs] for _, rhs in planes]
    from ..linalg._exact import solve_exact

    try:
        solved = solve_exact(matrix, right)
    except ValueError:
        return None
    return (solved[0][0], solved[1][0], solved[2][0])


def _barycentric(
    point: ExactPoint3, carrier: tuple[ExactPoint3, ...], /
) -> tuple[Fraction, Fraction, Fraction, Fraction] | None:
    count = len(carrier)
    if count == 1:
        if point != carrier[0]:
            return None
        return Fraction(1), Fraction(0), Fraction(0), Fraction(0)
    if count == 2:
        direction = _sub(carrier[1], carrier[0])
        axes = [axis for axis, value in enumerate(direction) if value]
        if not axes:
            return None
        parameter = (point[axes[0]] - carrier[0][axes[0]]) / direction[axes[0]]
        if any(
            point[axis] != carrier[0][axis] + parameter * direction[axis]
            for axis in range(3)
        ):
            return None
        return Fraction(1) - parameter, parameter, Fraction(0), Fraction(0)
    if count == 3:
        first, second = _sub(carrier[1], carrier[0]), _sub(carrier[2], carrier[0])
        delta = _sub(point, carrier[0])
        for axis_a, axis_b in combinations(range(3), 2):
            determinant = first[axis_a] * second[axis_b] - first[axis_b] * second[axis_a]
            if not determinant:
                continue
            u = (
                delta[axis_a] * second[axis_b]
                - delta[axis_b] * second[axis_a]
            ) / determinant
            v = (
                first[axis_a] * delta[axis_b]
                - first[axis_b] * delta[axis_a]
            ) / determinant
            if any(
                point[axis]
                != carrier[0][axis] + u * first[axis] + v * second[axis]
                for axis in range(3)
            ):
                return None
            return Fraction(1) - u - v, u, v, Fraction(0)
        return None
    if count != 4:
        raise ValueError("Carrier simplices must have one through four vertices.")
    matrix = [
        [carrier[column + 1][axis] - carrier[0][axis] for column in range(3)]
        for axis in range(3)
    ]
    right = [[point[axis] - carrier[0][axis]] for axis in range(3)]
    from ..linalg._exact import solve_exact

    try:
        coordinates = solve_exact(matrix, right)
    except ValueError:
        return None
    values: ExactPoint3 = (
        coordinates[0][0],
        coordinates[1][0],
        coordinates[2][0],
    )
    return (
        Fraction(1) - sum(values),
        values[0],
        values[1],
        values[2],
    )


def _carrier_planes(
    carrier: tuple[ExactPoint3, ...], /
) -> tuple[tuple[ExactPoint3, Fraction], ...]:
    if len(carrier) == 1:
        return ()
    zero: ExactPoint3 = (Fraction(0), Fraction(0), Fraction(0))
    if len(carrier) == 2:
        first, second = carrier
        direction = _sub(second, first)
        normals: list[ExactPoint3] = []
        for axis in range(3):
            basis: ExactPoint3 = (
                Fraction(int(axis == 0)),
                Fraction(int(axis == 1)),
                Fraction(int(axis == 2)),
            )
            normal = _cross(direction, basis)
            if any(normal) and all(_cross(normal, old) != zero for old in normals):
                normals.append(normal)
            if len(normals) == 2:
                break
        if len(normals) != 2:
            raise ValueError("Line carrier witnesses must have two distinct endpoints.")
        return (
            (normals[0], _dot(normals[0], first)),
            (normals[1], _dot(normals[1], first)),
            (direction, _dot(direction, first)),
            (direction, _dot(direction, second)),
        )
    if len(carrier) == 3:
        first, second, third = carrier
        triangle_points = (first, second, third)
        triangle = _plane(triangle_points)
        if triangle is None:
            raise ValueError("Triangle carrier witnesses must have full affine rank.")
        normal, _ = triangle
        edges: tuple[tuple[ExactPoint3, ExactPoint3], ...] = (
            (first, second),
            (second, third),
            (third, first),
        )
        result: list[tuple[ExactPoint3, Fraction]] = [triangle]
        for start, end in edges:
            edge_normal = _cross(_sub(end, start), normal)
            if not any(edge_normal):
                raise ValueError("Triangle carrier edge is degenerate.")
            result.append((edge_normal, _dot(edge_normal, start)))
        return tuple(result)
    if len(carrier) == 4:
        first, second, third, fourth = carrier
        faces: tuple[
            tuple[ExactPoint3, ExactPoint3, ExactPoint3],
            tuple[ExactPoint3, ExactPoint3, ExactPoint3],
            tuple[ExactPoint3, ExactPoint3, ExactPoint3],
            tuple[ExactPoint3, ExactPoint3, ExactPoint3],
        ] = (
            (second, third, fourth),
            (first, third, fourth),
            (first, second, fourth),
            (first, second, third),
        )
        result: list[tuple[ExactPoint3, Fraction]] = []
        for face in faces:
            plane = _plane(face)
            if plane is not None:
                result.append(plane)
        return tuple(result)
    raise ValueError("Carrier simplices must have one through four vertices.")


def _power_candidate_diagnostic(
    candidates: dict[
        ExactPoint3, tuple[Fraction, Fraction, Fraction, Fraction]
    ],
    /,
) -> str:
    """Format a bounded failure-only exact candidate sample."""
    sample = tuple(islice(candidates.items(), 4))
    return (
        f"exact_candidate_count={len(candidates)}, "
        f"candidate_rows={sample!r}, truncated={len(candidates) > len(sample)}"
    )


def _vertex_power_point(
    rounded: np.ndarray,
    site_points: tuple[ExactPoint3, ...],
    site_weights: tuple[Fraction, ...],
    equal_sites: tuple[int, ...],
    carrier: tuple[ExactPoint3, ...],
    /,
    *,
    prepared_site_bank: _PreparedPowerSiteBank | None = None,
) -> tuple[ExactPoint3, tuple[Fraction, Fraction, Fraction, Fraction]]:
    if not equal_sites or tuple(sorted(set(equal_sites))) != equal_sites:
        raise ValueError("Equal-power site witnesses must be sorted unique site IDs.")
    if any(site < 0 or site >= len(site_points) for site in equal_sites):
        raise ValueError("Equal-power site witness references an undeclared site.")
    if not 1 <= len(carrier) <= 4:
        raise ValueError("Carrier simplex witnesses need one to four valid point IDs.")
    if prepared_site_bank is None:
        prepared_site_bank = _prepare_power_site_bank(site_points, site_weights)
    elif (
        prepared_site_bank.site_points is not site_points
        or prepared_site_bank.site_weights is not site_weights
    ):
        raise ValueError("Prepared exact power site bank does not match solver inputs.")
    anchor = equal_sites[0]
    power_constraints = []
    for site in equal_sites[1:]:
        plane = _power_plane(site_points, site_weights, anchor, site)
        if plane is not None:
            power_constraints.append(plane)
    constraints = (*power_constraints, *_carrier_planes(carrier))
    exact_candidates: dict[
        ExactPoint3, tuple[Fraction, Fraction, Fraction, Fraction]
    ] = {}
    if len(carrier) == 1:
        candidate_points = (carrier[0],)
    else:
        candidate_points = tuple(
            point
            for selected in combinations(constraints, 3)
            if (point := _solve_three(selected)) is not None
        )
    for point in candidate_points:
        barycentric = _barycentric(point, carrier)
        if barycentric is None or any(value < 0 for value in barycentric):
            continue
        if not _candidate_is_global_minimum(point, equal_sites, prepared_site_bank):
            continue
        if any(
            sum((normal[i] * point[i] for i in range(3)), Fraction(0)) != rhs
            for normal, rhs in power_constraints
        ):
            continue
        exact_candidates[point] = barycentric
    if not exact_candidates:
        raise ValueError("Power vertex witness has no exact feasible carrier/power intersection.")
    # Exact rational coordinates carry no signed-zero bit; compare values without
    # rewriting the native rounded array.
    rounded_candidates = {
        point: barycentric
        for point, barycentric in exact_candidates.items()
        if np.array_equal(
            np.asarray([float(value) for value in point], dtype=np.float64),
            rounded,
        )
    }
    if not rounded_candidates:
        raise ValueError(
            "Exact carrier/power intersection does not match the rounded binary64 vertex; "
            + _power_candidate_diagnostic(exact_candidates)
        )
    if len(rounded_candidates) != 1:
        raise ValueError(
            "Power vertex witness must identify exactly one rational carrier/power "
            "intersection; "
            + _power_candidate_diagnostic(rounded_candidates)
        )
    return next(iter(rounded_candidates.items()))


class ExactPowerCellGeometrySource(StrictModule, NonTrainableState):
    """Exact rational power vertices bound to source sites and PLC carrier witnesses."""

    site_points: jnp.ndarray
    site_weights: jnp.ndarray
    exact_site_points: tuple[ExactPoint3, ...] = eqx.field(static=True)
    carrier_mesh: CellMesh
    carrier_points: jnp.ndarray
    carrier_tets: jnp.ndarray
    vertex_site_offsets: jnp.ndarray
    vertex_sites: jnp.ndarray
    vertex_carriers: jnp.ndarray
    runtime_vertices: jnp.ndarray
    output_vertex_indices: jnp.ndarray
    cell_sites: jnp.ndarray
    cell_regions: jnp.ndarray
    exact_vertices: tuple[ExactPoint3, ...] = eqx.field(static=True)
    carrier_barycentrics: tuple[tuple[Fraction, Fraction, Fraction, Fraction], ...] = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    periodic_source_context: ExactPeriodicPowerSourceContext | None
    source_map_id: str = eqx.field(static=True)
    carrier_mesh_id: str = eqx.field(static=True)
    carrier_topology_id: str = eqx.field(static=True)
    domain_source: Any

    @checked
    def __init__(
        self,
        site_points: Any,
        site_weights: Any,
        carrier_mesh: CellMesh,
        restricted_cells: Any,
        /,
        *,
        source_id: str,
        source_revision: str,
        periodic_source_context: ExactPeriodicPowerSourceContext | None = None,
        domain_source: Any = None,
        exact_site_points: tuple[ExactPoint3, ...] | None = None,
        output_vertex_indices: Any = None,
        precomputed_exact_vertices: tuple[ExactPoint3, ...] | None = None,
        precomputed_carrier_barycentrics: tuple[
            tuple[Fraction, Fraction, Fraction, Fraction], ...
        ]
        | None = None,
    ) -> None:
        if not isinstance(carrier_mesh, CellMesh) or carrier_mesh.topological_dimension != 3:
            raise TypeError("carrier_mesh must be a three-dimensional CellMesh.")
        if (
            not isinstance(source_id, str)
            or not source_id.strip()
            or not isinstance(source_revision, str)
            or not source_revision.strip()
        ):
            raise ValueError("Power source identity and revision must be nonempty strings.")
        normalized_source_id = source_id.strip()
        normalized_source_revision = source_revision.strip()
        points = np.asarray(site_points, dtype=np.float64)
        weights = np.asarray(site_weights, dtype=np.float64)
        rounded = np.asarray(restricted_cells.vertices, dtype=np.float64)
        offsets = np.asarray(restricted_cells.vertex_site_offsets)
        sites = np.asarray(restricted_cells.vertex_sites)
        carriers = np.asarray(restricted_cells.vertex_carriers)
        cell_sites = np.asarray(restricted_cells.cell_sites)
        cell_regions = np.asarray(restricted_cells.cell_regions)
        if (
            points.ndim != 2
            or points.shape[1] != 3
            or not points.shape[0]
            or weights.shape != (points.shape[0],)
            or rounded.ndim != 2
            or rounded.shape[1] != 3
            or not np.all(np.isfinite(rounded))
            or not np.all(np.isfinite(points))
            or not np.all(np.isfinite(weights))
        ):
            raise ValueError("Power source sites, weights, and output vertices must be finite 3-D arrays.")
        if (
            offsets.shape != (rounded.shape[0] + 1,)
            or offsets.dtype.kind not in "iu"
            or offsets[0] != 0
            or offsets[-1] != sites.size
            or np.any(np.diff(offsets) < 1)
            or sites.ndim != 1
            or sites.dtype.kind not in "iu"
            or np.any(sites < 0)
            or np.any(sites >= points.shape[0])
            or carriers.shape != (rounded.shape[0], 4)
            or carriers.dtype.kind not in "iu"
            or np.any(carriers < -1)
            or cell_sites.dtype.kind not in "iu"
            or cell_regions.dtype.kind not in "iu"
            or np.any(cell_sites < 0)
            or np.any(cell_sites >= points.shape[0])
            or cell_regions.shape != cell_sites.shape
        ):
            raise ValueError("Power output site and carrier witness arrays are misaligned.")
        if exact_site_points is None:
            exact_sites = tuple(_fraction_point(row) for row in points)
        elif (
            not isinstance(exact_site_points, tuple)
            or len(exact_site_points) != points.shape[0]
            or any(
                not isinstance(point, tuple)
                or len(point) != 3
                or any(not isinstance(value, Fraction) for value in point)
                for point in exact_site_points
            )
        ):
            raise TypeError("Exact power site coordinates must be immutable Fraction triples.")
        else:
            exact_sites = exact_site_points
        if not np.array_equal(np.asarray(exact_sites, dtype=np.float64), points):
            raise ValueError(
                "Exact power site coordinates do not match their execution shadows."
            )
        if periodic_source_context is not None:
            if not isinstance(
                periodic_source_context, ExactPeriodicPowerSourceContext
            ):
                raise TypeError(
                    "periodic_source_context must be ExactPeriodicPowerSourceContext."
                )
            periodic_source_context.require_image_bank(exact_sites, points, weights)
        if output_vertex_indices is None:
            output_rows = np.arange(rounded.shape[0], dtype=np.int64)
        else:
            raw_output_rows = np.asarray(output_vertex_indices)
            if raw_output_rows.ndim != 1 or raw_output_rows.dtype.kind not in "iu":
                raise TypeError("output_vertex_indices must be an integer row vector.")
            output_rows = raw_output_rows.astype(np.int64, copy=False)
            if (
                output_rows.size == 0
                or np.any(output_rows < 0)
                or np.any(output_rows >= rounded.shape[0])
                or np.unique(output_rows).size != output_rows.size
            ):
                raise ValueError(
                    "output_vertex_indices must select unique in-range power vertices."
                )
        if output_rows.size == 0:
            raise ValueError("output_vertex_indices must select a nonempty power vertex bank.")
        if precomputed_exact_vertices is None:
            if precomputed_carrier_barycentrics is not None:
                raise ValueError(
                    "Precomputed power roots require both exact coordinates and carrier proofs."
                )
        elif precomputed_carrier_barycentrics is None:
            raise ValueError(
                "Precomputed power roots require both exact coordinates and carrier proofs."
            )
        elif (
            len(precomputed_exact_vertices) != output_rows.size
            or len(precomputed_carrier_barycentrics) != output_rows.size
            or any(
                not isinstance(point, tuple)
                or len(point) != 3
                or any(not isinstance(value, Fraction) for value in point)
                for point in precomputed_exact_vertices
            )
            or any(
                not isinstance(row, tuple)
                or len(row) != 4
                or any(not isinstance(value, Fraction) for value in row)
                for row in precomputed_carrier_barycentrics
            )
        ):
            raise ValueError("Precomputed exact power root proofs are malformed.")

        offsets = offsets.astype(np.int64, copy=False)
        sites = sites.astype(np.int64, copy=False)
        carriers = carriers.astype(np.int64, copy=False)
        cell_sites = cell_sites.astype(np.int64, copy=False)
        cell_regions = cell_regions.astype(np.int64, copy=False)
        for vertex in range(rounded.shape[0]):
            row = sites[offsets[vertex] : offsets[vertex + 1]]
            if np.any(np.diff(row) <= 0):
                raise ValueError("Equal-power site IDs must be sorted and unique per vertex.")
        carrier_mask = carriers >= 0
        carrier_counts = np.sum(carrier_mask, axis=1)
        if (
            np.any(carrier_counts < 1)
            or np.any(carrier_counts > 4)
            or np.any(
                carrier_mask
                != (np.arange(4)[None, :] < carrier_counts[:, None])
            )
            or np.any(carriers[carrier_mask] >= carrier_mesh.coordinates.shape[0])
            or any(
                np.unique(row[row >= 0]).size != np.count_nonzero(row >= 0)
                for row in carriers
            )
        ):
            raise ValueError("Carrier point IDs must form unique valid-prefix rows.")
        carrier_tets: list[np.ndarray] = []
        for block in carrier_mesh.blocks:
            if block.cell_kind not in ("tetrahedron", "simplex:3"):
                raise ValueError("Power source carrier mesh must contain tetrahedral blocks only.")
            carrier_tets.append(np.asarray(block.vertices, dtype=np.int64))
        tets = np.concatenate(carrier_tets, axis=0)
        if any(int(index) >= carrier_mesh.coordinates.shape[0] for index in tets.flat):
            raise ValueError("Carrier tetrahedra reference absent mesh vertices.")
        carrier_coordinates = np.asarray(carrier_mesh.coordinates, dtype=np.float64)
        exact_weights = tuple(Fraction(float(value)) for value in weights)
        prepared_site_bank = _prepare_power_site_bank(exact_sites, exact_weights)
        exact_carrier = tuple(_fraction_point(row) for row in carrier_coordinates)
        exact_vertices: list[ExactPoint3] = []
        barycentric: list[tuple[Fraction, Fraction, Fraction, Fraction]] = []
        for output_row, vertex in enumerate(output_rows.tolist()):
            carrier_row = carriers[vertex]
            carrier_ids = tuple(map(int, carrier_row[carrier_row >= 0]))
            if not any(set(carrier_ids) <= set(map(int, tet)) for tet in tets):
                raise ValueError(
                    "A power vertex carrier must name a recovered source simplex or face."
                )
            carrier = tuple(exact_carrier[index] for index in carrier_ids)
            equal = tuple(
                map(int, sites[int(offsets[vertex]) : int(offsets[vertex + 1])])
            )
            if precomputed_exact_vertices is None:
                exact, weights_in_carrier = _vertex_power_point(
                    rounded[vertex],
                    exact_sites,
                    exact_weights,
                    equal,
                    carrier,
                    prepared_site_bank=prepared_site_bank,
                )
            else:
                if precomputed_carrier_barycentrics is None:
                    raise ValueError(
                        "Precomputed power roots require exact carrier barycentrics."
                    )
                exact = precomputed_exact_vertices[output_row]
                weights_in_carrier = precomputed_carrier_barycentrics[output_row]
                carrier_count = len(carrier)
                if (
                    any(value < 0 for value in weights_in_carrier[:carrier_count])
                    or sum(
                        weights_in_carrier[:carrier_count], Fraction(0)
                    )
                    != 1
                    or any(weights_in_carrier[carrier_count:])
                ):
                    raise ValueError("Precomputed power carrier barycentrics are invalid.")
                reconstructed = tuple(
                    sum(
                        (
                            weights_in_carrier[index] * carrier[index][axis]
                            for index in range(carrier_count)
                        ),
                        Fraction(0),
                    )
                    for axis in range(3)
                )
                point_norm = _dot(exact, exact)
                reference_power = _site_power_value(
                    exact, point_norm, equal[0], prepared_site_bank
                )
                if (
                    reconstructed != exact
                    or not np.array_equal(
                        np.asarray(exact, dtype=np.float64), rounded[vertex]
                    )
                    or any(
                        _site_power_value(exact, point_norm, site, prepared_site_bank)
                        != reference_power
                        for site in equal[1:]
                    )
                ):
                    raise ValueError(
                        "Precomputed power root disagrees with its exact carrier or site proof."
                    )
            exact_vertices.append(exact)
            barycentric.append(weights_in_carrier)
        identity = canonical_fingerprint(
            {
                "kind": "exact-power-cell-geometry-source",
                "source_id": normalized_source_id,
                "source_revision": normalized_source_revision,
                "periodic_source_context": None
                if periodic_source_context is None
                else periodic_source_context.context_id,
                "site_points": array_tree_fingerprint(points),
                "site_weights": array_tree_fingerprint(weights),
                "carrier_mesh": carrier_mesh.mesh_id,
                "carrier_topology": carrier_mesh.topology_id,
                "carrier_canonical_identity": _carrier_replay_identity(carrier_mesh),
                "carrier_tets": array_tree_fingerprint(tets),
                "restricted_cells": array_tree_fingerprint(
                    (
                        rounded,
                        offsets,
                        sites,
                        carriers,
                        cell_sites,
                        cell_regions,
                    )
                ),
                "exact_vertices": tuple(
                    tuple(coefficient_payload(value) for value in point)
                    for point in exact_vertices
                ),
                "carrier_barycentrics": tuple(
                    tuple(coefficient_payload(value) for value in row)
                    for row in barycentric
                ),
                "exact_site_points": tuple(
                    tuple(coefficient_payload(value) for value in point)
                    for point in exact_sites
                ),
                "output_vertex_indices": array_tree_fingerprint(output_rows),
            }
        )
        self.exact_site_points = exact_sites
        self.site_points = jnp.asarray(points)
        self.site_weights = jnp.asarray(weights)
        self.carrier_mesh = carrier_mesh
        self.carrier_points = carrier_mesh.coordinates
        self.carrier_tets = jnp.asarray(tets, dtype=jnp.int64)
        self.vertex_site_offsets = jnp.asarray(offsets, dtype=jnp.int64)
        self.vertex_sites = jnp.asarray(sites, dtype=jnp.int64)
        self.vertex_carriers = jnp.asarray(carriers, dtype=jnp.int64)
        self.runtime_vertices = jnp.asarray(rounded)
        self.output_vertex_indices = jnp.asarray(output_rows, dtype=jnp.int64)
        self.cell_sites = jnp.asarray(cell_sites, dtype=jnp.int64)
        self.cell_regions = jnp.asarray(cell_regions, dtype=jnp.int64)
        self.exact_vertices = tuple(exact_vertices)
        self.carrier_barycentrics = tuple(barycentric)
        self.source_id = normalized_source_id
        self.source_revision = normalized_source_revision
        self.periodic_source_context = periodic_source_context
        self.source_map_id = identity
        self.carrier_mesh_id = carrier_mesh.mesh_id
        self.carrier_topology_id = carrier_mesh.topology_id
        self.domain_source = domain_source

    @property
    def rounded_vertices(self) -> np.ndarray:
        rows = np.asarray(self.output_vertex_indices, dtype=np.int64)
        return np.asarray(self.runtime_vertices)[rows]

    def source_coordinates(
        self, indices: Any = None, /
    ) -> CoordinateSourceBank:
        """Return exact Fraction coordinates for selected output vertex rows."""
        if indices is None:
            selected = range(len(self.exact_vertices))
        else:
            rows = np.asarray(indices)
            if rows.dtype.kind not in "iu" or rows.ndim != 1:
                raise TypeError("Power coordinate selection must be an integer vector.")
            if np.any(rows < 0) or np.any(rows >= len(self.exact_vertices)):
                raise IndexError("Power coordinate selection is outside the source vertex bank.")
            selected = rows.astype(np.int64).tolist()
        return tuple(self.exact_vertices[int(index)] for index in selected)


    def require_periodic_lift(
        self,
        coordinates: ArrayLike,
        cell: PeriodicCell,
        representatives: ArrayLike,
        shifts: ArrayLike,
        /,
    ) -> None:
        """Prove exact deck action and its rounded execution-coordinate shadow."""
        from ._periodic_exact import exact_periodic_element, identification_orders

        if self.periodic_source_context is None:
            raise ValueError("Exact power geometry has no periodic source context.")
        self.periodic_source_context.require_periodic_cell(cell)
        points = np.asarray(coordinates, dtype=np.float64)
        exact_points = self.source_coordinates()
        if (
            points.shape != (len(exact_points), 3)
            or not np.array_equal(points, np.asarray(self.rounded_vertices))
            or not np.array_equal(points, np.asarray(exact_points, dtype=np.float64))
        ):
            raise ValueError(
                "Exact periodic power coordinates do not match their exact source-map shadows."
            )
        roots = np.asarray(representatives)
        exponents = np.asarray(shifts)
        rank = len(identification_orders(cell))
        if (
            roots.shape != (points.shape[0],)
            or roots.dtype.kind not in "iu"
            or exponents.shape != (points.shape[0], rank)
            or exponents.dtype.kind not in "iu"
        ):
            raise ValueError("Exact periodic power orbit rows are malformed.")
        roots = roots.astype(np.int64, copy=False)
        exponents = exponents.astype(np.int64, copy=False)
        transforms: dict[tuple[int, ...], tuple[tuple[Fraction, ...], ...]] = {}
        for row, root in enumerate(roots):
            key = tuple(int(value) for value in exponents[row])
            matrix = transforms.get(key)
            if matrix is None:
                matrix = exact_periodic_element(cell, exponents[row])
                transforms[key] = matrix
            expected = tuple(
                sum(
                    (
                        matrix[axis][component] * exact_points[int(root)][component]
                        for component in range(3)
                    ),
                    Fraction(0),
                )
                + matrix[axis][3]
                for axis in range(3)
            )
            if expected != exact_points[row]:
                raise ValueError(
                    "Exact power source coordinates violate their declared periodic deck action."
                )


    def require_periodic_mesh(self, mesh: CellMesh, /) -> None:
        """Require the retained exact power source to bind the mesh quotient."""
        if self.periodic_source_context is None:
            raise ValueError("Exact power geometry has no periodic source context.")
        topology = mesh.periodic_topology
        if topology is None or not isinstance(topology.cell, PeriodicCell):
            raise ValueError("Periodic power target mesh lacks its periodic quotient.")
        self.periodic_source_context.require_periodic_cell(topology.cell)
        if topology.coordinate_source_map_id != self.source_map_id:
            raise ValueError(
                "Periodic power topology is bound to another exact coordinate source."
            )
        topology.require_coordinates(mesh.coordinates)

    def require_carrier(self, carrier_mesh: CellMesh, /) -> None:
        if (
            carrier_mesh.mesh_id != self.carrier_mesh_id
            or carrier_mesh.topology_id != self.carrier_topology_id
        ):
            raise ValueError("Exact power source is bound to a different carrier mesh epoch.")

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[ExactPowerCellGeometrySource],
        value: ExactPowerCellGeometrySource,
        /,
        *,
        domain_source: Any = _UNSET_DOMAIN_SOURCE,
        target_periodic_cell: PeriodicCell | None = None,
    ) -> ExactPowerCellGeometrySource:
        """Rebuild an exact power map against its current target periodic cell."""
        if not isinstance(value, cls):
            raise TypeError("value must be ExactPowerCellGeometrySource.")
        if target_periodic_cell is not None and not isinstance(
            target_periodic_cell, PeriodicCell
        ):
            raise TypeError("target_periodic_cell must be PeriodicCell or None.")
        if value.periodic_source_context is None:
            periodic_source_context = None
        elif type(value.periodic_source_context) is ExactPeriodicPowerSourceContext:
            periodic_source_context = ExactPeriodicPowerSourceContext.from_stored_fields(
                value.periodic_source_context
            )
        else:
            raise TypeError(
                "Stored periodic power source context has an unexpected concrete type."
            )
        stored_context_id = (
            None
            if value.periodic_source_context is None
            else value.periodic_source_context.context_id
        )
        stored_carrier = value.carrier_mesh
        stored_carrier_identity = _carrier_replay_identity(stored_carrier)
        blocks = tuple(
            CellBlock(
                block.name,
                block.cell_kind,
                block.vertices,
                vertex_valid=block.vertex_valid,
                global_ids=block.global_ids,
            )
            for block in stored_carrier.blocks
        )
        entity_global_ids = {
            entity_set.intrinsic_dimension: np.asarray(entity_set.entity_ids)
            for entity_set in stored_carrier.topology.entity_sets
        }
        carrier_mesh = CellMesh(
            stored_carrier.coordinates,
            blocks,
            vertex_global_ids=stored_carrier.vertex_global_ids,
            entity_global_ids=entity_global_ids,
            numeric_version=stored_carrier.numeric_version,
            periodic_topology=stored_carrier.periodic_topology,
        )
        restricted_cells = SimpleNamespace(
            vertices=value.runtime_vertices,
            vertex_site_offsets=value.vertex_site_offsets,
            vertex_sites=value.vertex_sites,
            vertex_carriers=value.vertex_carriers,
            cell_sites=value.cell_sites,
            cell_regions=value.cell_regions,
        )
        restored_domain_source = (
            value.domain_source
            if domain_source is _UNSET_DOMAIN_SOURCE
            else domain_source
        )
        renewed = cls(
            value.site_points,
            value.site_weights,
            carrier_mesh,
            restricted_cells,
            source_id=value.source_id,
            source_revision=value.source_revision,
            periodic_source_context=periodic_source_context,
            domain_source=restored_domain_source,
            exact_site_points=value.exact_site_points,
            output_vertex_indices=value.output_vertex_indices,
        )
        array_fields = (
            "site_points",
            "site_weights",
            "carrier_points",
            "carrier_tets",
            "vertex_site_offsets",
            "vertex_sites",
            "vertex_carriers",
            "runtime_vertices",
            "output_vertex_indices",
            "cell_sites",
            "cell_regions",
        )
        if (
            renewed.source_map_id != value.source_map_id
            or renewed.source_id != value.source_id
            or renewed.source_revision != value.source_revision
            or (
                None
                if renewed.periodic_source_context is None
                else renewed.periodic_source_context.context_id
            )
            != stored_context_id
            or renewed.carrier_mesh_id != value.carrier_mesh_id
            or renewed.carrier_topology_id != value.carrier_topology_id
            or carrier_mesh.numeric_version != stored_carrier.numeric_version
            or _carrier_replay_identity(carrier_mesh) != stored_carrier_identity
            or len(carrier_mesh.blocks) != len(stored_carrier.blocks)
            or tuple(
                (block.name, block.cell_kind, block.block_id)
                for block in carrier_mesh.blocks
            )
            != tuple(
                (block.name, block.cell_kind, block.block_id)
                for block in stored_carrier.blocks
            )
            or any(
                not np.array_equal(
                    np.asarray(rebuilt_block.global_ids),
                    np.asarray(stored_block.global_ids),
                )
                for rebuilt_block, stored_block in zip(
                    carrier_mesh.blocks, stored_carrier.blocks, strict=True
                )
            )
            or carrier_mesh.mesh_id != stored_carrier.mesh_id
            or carrier_mesh.topology_id != stored_carrier.topology_id
            or not np.array_equal(
                np.asarray(carrier_mesh.vertex_global_ids),
                np.asarray(stored_carrier.vertex_global_ids),
            )
            or renewed.exact_vertices != value.exact_vertices
            or renewed.exact_site_points != value.exact_site_points
            or renewed.carrier_barycentrics != value.carrier_barycentrics
            or any(
                not np.array_equal(
                    np.asarray(getattr(renewed, name)),
                    np.asarray(getattr(value, name)),
                )
                for name in array_fields
            )
        ):
            raise ValueError("Stored exact power geometry source does not reproduce its exact map.")
        if renewed.periodic_source_context is not None:
            if target_periodic_cell is None:
                raise ValueError(
                    "Periodic exact power restoration requires its target PeriodicCell."
                )
            renewed.periodic_source_context.require_periodic_cell(
                target_periodic_cell
            )
        elif target_periodic_cell is not None:
            raise ValueError(
                "A nonperiodic exact power map cannot bind a target PeriodicCell."
            )
        return renewed



@checked
def prepare_exact_power_geometry_source(
    site_points: Any,
    site_weights: Any,
    carrier_mesh: CellMesh,
    restricted_cells: Any,
    /,
    *,
    source_id: str,
    source_revision: str,
    domain_source: Any = None,
) -> ExactPowerCellGeometrySource:
    return ExactPowerCellGeometrySource(
        site_points,
        site_weights,
        carrier_mesh,
        restricted_cells,
        source_id=source_id,
        source_revision=source_revision,
        domain_source=domain_source,
    )


__all__ = [
    "ExactPowerCellGeometrySource",
    "prepare_exact_power_geometry_source",
]
