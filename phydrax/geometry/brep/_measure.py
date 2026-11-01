#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Source-native enclosed B-Rep volume measures.

The source intervals, not tessellation triangles, define every measure. Direct
smoke recipe::

    from phydrax._physical import SpatialCoordinateContract
    from phydrax.geometry.brep._constructors import brep_box
    from phydrax.geometry.brep._measure import brep_mass_properties

    model = brep_box(
        (0, 0, 0),
        (1, 2, 3),
        coordinate_contract=SpatialCoordinateContract.si(),
    )
    measure = brep_mass_properties(model, subdivisions=8)
    assert measure.status == ("certified",)
    assert measure.volume_enclosures[0, 0] <= 6 <= measure.volume_enclosures[0, 1]
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from numbers import Integral
from typing import Literal, final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from ..._jax_config import x64_enabled
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._validation import positive_integer
from ...typing import Bool, Dim, Float64, Scope, Size, as_array, checked, parse
from .._atlas import TrimDomain
from .._interval_enclosure import (
    Interval,
    IntervalExtensionError,
    PreparedIntervalFunction,
    interval_add,
    interval_divide,
    interval_multiply,
    interval_subtract,
    prepare_interval_function,
)
from .._rational import (
    fraction_down,
    fraction_up,
    rational_cross,
    rational_dot,
    rational_vector,
)
from ._assembly import BRepAssemblyOccurrence
from ._boolean_arrangement import _shell_volume_sign
from ._carrier_pieces import child_coefficient_enclosures
from ._exact import rational_determinant
from ._model import BRepEntityId, BRepModel
from ._patches import AbstractSurfacePatch, ExactLineCurve, LineCurve, PlanePatch

type BRepMassMeasureStatus = Literal["certified", "unsupported", "unresolved"]
type _ScalarInterval = tuple[np.float64, np.float64]


@dataclass(slots=True, eq=False)
class _VolumeCell:
    face: int
    domain: TrimDomain
    function: PreparedIntervalFunction
    box: np.ndarray
    contribution: _ScalarInterval
    splittable: bool = True


class _MeasureOccurrenceDim(Dim, minimum=1):
    """Solid occurrences represented by one mass measure."""


class _MeasureFailure(Exception):
    def __init__(self, status: BRepMassMeasureStatus, reason: str, /) -> None:
        self.status = status
        self.reason = reason
        super().__init__(reason)


@final
class BRepMassProperties(StrictModule, NonTrainableState):
    """Certified source-volume intervals in assembly occurrence order.

    ``volume`` is the interval midpoint for convenient numerical consumers;
    scientific decisions must use ``volume_enclosures`` and ``status``.
    Unsupported and unresolved occurrences carry ``[-inf, inf]`` and NaN,
    never a sampled finite estimate. ``solid_ids`` retain source owner,
    revision, and assembly occurrence identity.
    """

    __strict_contract__ = True

    model_id: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    solid_indices: tuple[int, ...] = eqx.field(static=True)
    occurrence_paths: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    solid_ids: tuple[BRepEntityId, ...] = eqx.field(static=True)
    status: tuple[BRepMassMeasureStatus, ...] = eqx.field(static=True)
    reasons: tuple[str, ...] = eqx.field(static=True)
    volume: Float64[_MeasureOccurrenceDim]
    volume_enclosures: Float64[_MeasureOccurrenceDim, Literal[2]]

    def __init__(
        self,
        *,
        model_id: str,
        source_id: str,
        solid_indices: tuple[int, ...],
        occurrence_paths: tuple[tuple[str, ...], ...],
        solid_ids: tuple[BRepEntityId, ...],
        status: tuple[BRepMassMeasureStatus, ...],
        reasons: tuple[str, ...],
        volume_enclosures: np.ndarray,
    ) -> None:
        count = len(solid_indices)
        scope = Scope()
        parse(
            count,
            Size[_MeasureOccurrenceDim],
            "BRepMassProperties.occurrence_count",
            scope=scope,
        )
        status_ = tuple(
            parse(value, BRepMassMeasureStatus, "BRepMassProperties.status")
            for value in status
        )
        reasons_ = tuple(reasons)
        bounds = np.asarray(volume_enclosures, dtype=np.float64)
        if (
            len(occurrence_paths) != count
            or len(solid_ids) != count
            or len(status_) != count
            or len(reasons_) != count
            or bounds.shape != (count, 2)
            or any(bounds[index, 0] > bounds[index, 1] for index in range(count))
        ):
            raise ValueError("BRepMassProperties occurrence data must align.")
        if any(
            (state == "certified")
            != (np.all(np.isfinite(bounds[index])) and not reasons_[index])
            for index, state in enumerate(status_)
        ):
            raise ValueError(
                "BRepMassProperties finite enclosures and failure reasons must match status."
            )
        midpoint = np.asarray(
            [
                bounds[index, 0] * 0.5 + bounds[index, 1] * 0.5
                if state == "certified"
                else np.nan
                for index, state in enumerate(status_)
            ],
            dtype=np.float64,
        )
        self.model_id = model_id
        self.source_id = source_id
        self.solid_indices = solid_indices
        self.occurrence_paths = occurrence_paths
        self.solid_ids = solid_ids
        self.status = status_
        self.reasons = reasons_
        self.volume = as_array(
            midpoint,
            Float64[_MeasureOccurrenceDim],
            "BRepMassProperties.volume",
            scope=scope,
        )
        self.volume_enclosures = as_array(
            bounds,
            Float64[_MeasureOccurrenceDim, Literal[2]],
            "BRepMassProperties.volume_enclosures",
            scope=scope,
        )

    @property
    def successful(self) -> Bool[_MeasureOccurrenceDim]:
        return jnp.asarray(
            tuple(value == "certified" for value in self.status), dtype=jnp.bool_
        )


def _fraction_uv(value: object, /) -> tuple[Fraction, Fraction] | None:
    if not isinstance(value, tuple) or len(value) != 2:
        return None
    first, second = value
    if not isinstance(first, Fraction) or not isinstance(second, Fraction):
        return None
    return first, second


def _exact_planar_face_volume(model: BRepModel, face: int, /) -> Fraction | None:
    """Integrate a plane face exactly when all source trims are affine lines."""
    patch = model.patches[face]
    geometry = model.geometry
    if not isinstance(patch, PlanePatch) or not geometry.face_loops[face]:
        return None
    origin = rational_vector(np.asarray(patch.origin))
    first = rational_vector(np.asarray(patch.first_axis))
    second = rational_vector(np.asarray(patch.second_axis))
    normal = rational_cross(first, second)
    exact_offset = rational_dot(origin, normal)
    if not isinstance(exact_offset, Fraction):
        return None
    offset = Fraction(int(model.orientation[face])) * exact_offset / 3
    ranges = np.asarray(geometry.edge_ranges, dtype=np.float64)
    parameter_area = Fraction(0)
    for loop in geometry.face_loops[face]:
        loop_area = Fraction(0)
        for coedge in loop:
            pcurve = geometry.pcurves[coedge]
            if not isinstance(pcurve, (LineCurve, ExactLineCurve)):
                return None
            edge = geometry.coedge_edges[coedge]
            start, end = (Fraction.from_float(float(value)) for value in ranges[edge])
            first_point = _fraction_uv(pcurve.evaluate_exact(start))
            last_point = _fraction_uv(pcurve.evaluate_exact(end))
            if first_point is None or last_point is None:
                return None
            if geometry.coedge_senses[coedge] < 0:
                first_point, last_point = last_point, first_point
            loop_area += (
                first_point[0] * last_point[1] - last_point[0] * first_point[1]
            ) / 2
        parameter_area += loop_area
    volume = offset * parameter_area
    return volume if isinstance(volume, Fraction) else None


def _volume_function(
    patch: AbstractSurfacePatch, orientation: float, /
) -> PreparedIntervalFunction:
    def integrand(parameters: Array, /) -> Array:
        point = patch.evaluate(parameters)
        differential = jax.jacfwd(patch.evaluate)(parameters)
        flux = jnp.dot(point, jnp.cross(differential[:, 0], differential[:, 1]))
        return orientation * flux / 3.0

    return prepare_interval_function(
        integrand,
        2,
        batch_capacity=1,
        constant_bounds=child_coefficient_enclosures(patch),
    )


def _finite_interval(bounds: tuple[object, object], /) -> _ScalarInterval:
    lower = float(np.asarray(bounds[0]))
    upper = float(np.asarray(bounds[1]))
    if not np.isfinite(lower) or not np.isfinite(upper) or lower > upper:
        raise _MeasureFailure(
            "unresolved", "source integration produced no finite ordered enclosure"
        )
    return np.float64(lower), np.float64(upper)


def _point_interval(value: float, /) -> _ScalarInterval:
    point = np.float64(value)
    return point, point




def _cell_area(box: np.ndarray, /) -> _ScalarInterval:
    width_u = interval_subtract(
        _point_interval(box[1, 0]), _point_interval(box[0, 0])
    )
    width_v = interval_subtract(
        _point_interval(box[1, 1]), _point_interval(box[0, 1])
    )
    return _finite_interval(interval_multiply(width_u, width_v))


def _classification_box(
    model: BRepModel, face: int, domain: TrimDomain, box: np.ndarray, /
) -> np.ndarray:
    if domain.curve_loops:
        return box
    bounds = np.asarray(model.parameter_bounds[face], dtype=np.float64)
    span = interval_subtract((bounds[1], bounds[1]), (bounds[0], bounds[0]))
    normalized = interval_divide(
        interval_subtract((box[0], box[0]), (bounds[0], bounds[0])), span
    )
    return np.stack((normalized[0], normalized[1]))


def _prepare_face_integrator(
    model: BRepModel, face: int, /
) -> tuple[TrimDomain, PreparedIntervalFunction]:
    domain = model.trim_domains[face]
    if domain is None or domain.numeric_only:
        raise _MeasureFailure(
            "unsupported",
            f"source face {face} has no source-authoritative trim domain",
        )
    try:
        function = _volume_function(model.patches[face], float(model.orientation[face]))
    except IntervalExtensionError as error:
        raise _MeasureFailure(
            "unsupported",
            f"source face {face} has no sound interval integral: {error}",
        ) from error
    except NotImplementedError as error:
        raise _MeasureFailure(
            "unsupported",
            f"source face {face} does not implement interval integration: {error}",
        ) from error
    except ValueError as error:
        raise _MeasureFailure(
            "unsupported",
            f"source face {face} cannot prepare an interval integral: {error}",
        ) from error
    return domain, function


def _volume_cell_contribution(
    model: BRepModel,
    face: int,
    domain: TrimDomain,
    function: PreparedIntervalFunction,
    box: np.ndarray,
    /,
) -> _ScalarInterval:
    try:
        classification_box = _classification_box(model, face, domain, box)
        classification = domain.classify_boxes(classification_box[None, :, :])
        lower, upper = function.evaluate(box[0][None, :], box[1][None, :])
    except IntervalExtensionError as error:
        raise _MeasureFailure(
            "unsupported",
            f"source face {face} has no sound interval integral: {error}",
        ) from error
    except NotImplementedError as error:
        raise _MeasureFailure(
            "unsupported",
            f"source face {face} does not implement interval integration: {error}",
        ) from error
    except ValueError as error:
        raise _MeasureFailure(
            "unresolved",
            f"source face {face} interval evaluation did not resolve: {error}",
        ) from error
    contribution = _finite_interval((lower[0], upper[0]))
    resolved = bool(np.asarray(classification.resolved[0]))
    inside = bool(np.asarray(classification.inside[0]))
    if resolved and not inside:
        contribution = _point_interval(0.0)
    elif not resolved or bool(np.asarray(classification.boundary[0])):
        contribution = _finite_interval(
            (min(0.0, float(contribution[0])), max(0.0, float(contribution[1])))
        )
    return _finite_interval(interval_multiply(contribution, _cell_area(box)))


def _initial_face_cells(
    model: BRepModel, face: int, subdivisions: int, /
) -> list[_VolumeCell]:
    domain, function = _prepare_face_integrator(model, face)
    parameter_bounds = np.asarray(model.parameter_bounds[face], dtype=np.float64)
    cells = []
    for u_index in range(subdivisions):
        u0 = parameter_bounds[0, 0] + (
            parameter_bounds[1, 0] - parameter_bounds[0, 0]
        ) * (u_index / subdivisions)
        u1 = (
            parameter_bounds[1, 0]
            if u_index + 1 == subdivisions
            else parameter_bounds[0, 0]
            + (parameter_bounds[1, 0] - parameter_bounds[0, 0])
            * ((u_index + 1) / subdivisions)
        )
        if not u0 < u1:
            raise _MeasureFailure(
                "unresolved",
                f"source face {face} reaches parameter representability",
            )
        for v_index in range(subdivisions):
            v0 = parameter_bounds[0, 1] + (
                parameter_bounds[1, 1] - parameter_bounds[0, 1]
            ) * (v_index / subdivisions)
            v1 = (
                parameter_bounds[1, 1]
                if v_index + 1 == subdivisions
                else parameter_bounds[0, 1]
                + (parameter_bounds[1, 1] - parameter_bounds[0, 1])
                * ((v_index + 1) / subdivisions)
            )
            if not v0 < v1:
                raise _MeasureFailure(
                    "unresolved",
                    f"source face {face} reaches parameter representability",
                )
            box = np.asarray(((u0, v0), (u1, v1)), dtype=np.float64)
            cells.append(
                _VolumeCell(
                    face,
                    domain,
                    function,
                    box,
                    _volume_cell_contribution(model, face, domain, function, box),
                )
            )
    return cells


def _closed_solid(model: BRepModel, solid: int, /) -> str | None:
    geometry = model.geometry
    shell_indices = geometry.solid_shells[solid]
    if not shell_indices:
        return f"solid {solid} has no source boundary shells"
    for shell in shell_indices:
        counts = [0] * geometry.topology.num_edges
        balance = [0] * geometry.topology.num_edges
        for face, shell_sign in zip(
            geometry.shell_faces[shell],
            geometry.shell_orientations[shell],
            strict=True,
        ):
            face_sign = int(model.orientation[face]) * shell_sign
            for loop in geometry.face_loops[face]:
                for coedge in loop:
                    edge = geometry.coedge_edges[coedge]
                    if geometry.edge_curves[edge] < 0:
                        continue
                    counts[edge] += 1
                    balance[edge] += geometry.coedge_senses[coedge] * face_sign
        for edge, curve_index in enumerate(geometry.edge_curves):
            if (
                counts[edge]
                and curve_index >= 0
                and (counts[edge] != 2 or balance[edge] != 0)
            ):
                return f"solid {solid} shell {shell} is not closed at source edge {edge}"
    return None


def _split_volume_cell(
    model: BRepModel, cell: _VolumeCell, /
) -> tuple[_VolumeCell, _VolumeCell] | None:
    extents = cell.box[1] - cell.box[0]
    for axis in sorted(range(2), key=lambda value: extents[value], reverse=True):
        first, last = cell.box[:, axis]
        middle = 0.5 * first + 0.5 * last
        if not first < middle < last:
            continue
        first_box, second_box = cell.box.copy(), cell.box.copy()
        first_box[1, axis] = middle
        second_box[0, axis] = middle
        return (
            _VolumeCell(
                cell.face,
                cell.domain,
                cell.function,
                first_box,
                _volume_cell_contribution(
                    model, cell.face, cell.domain, cell.function, first_box
                ),
            ),
            _VolumeCell(
                cell.face,
                cell.domain,
                cell.function,
                second_box,
                _volume_cell_contribution(
                    model, cell.face, cell.domain, cell.function, second_box
                ),
            ),
        )
    return None


def _solid_volume_enclosure(
    model: BRepModel, solid: int, subdivisions: int, /
) -> _ScalarInterval:
    reason = _closed_solid(model, solid)
    if reason is not None:
        raise _MeasureFailure("unresolved", reason)
    faces, signs = (
        model.topology.solid_faces[solid],
        model.topology.solid_face_orientations[solid],
    )
    if not faces:
        raise _MeasureFailure(
            "unresolved", f"solid {solid} has no source faces to integrate"
        )

    exact_total = Fraction(0)
    face_signs: dict[int, int] = {}
    cells: list[_VolumeCell] = []
    for face, sign in zip(faces, signs, strict=True):
        face_signs[face] = sign
        exact = _exact_planar_face_volume(model, face)
        if exact is None:
            cells.extend(_initial_face_cells(model, face, subdivisions))
        else:
            exact_total += sign * exact

    while True:
        total = _finite_interval((fraction_down(exact_total), fraction_up(exact_total)))
        for cell in cells:
            total = _finite_interval(
                interval_add(
                    total,
                    interval_multiply(
                        cell.contribution,
                        _point_interval(float(face_signs[cell.face])),
                    ),
                )
            )
        if total[0] > 0.0:
            return total
        if total[1] <= 0.0:
            raise _MeasureFailure(
                "unresolved",
                f"solid {solid} volume enclosure proves nonpositive material volume",
            )
        refinable = tuple(cell for cell in cells if cell.splittable)
        if not refinable:
            raise _MeasureFailure(
                "unresolved",
                f"solid {solid} volume enclosure remains unresolved at parameter representability",
            )
        worst = max(
            refinable,
            key=lambda cell: cell.contribution[1] - cell.contribution[0],
        )
        split = _split_volume_cell(model, worst)
        if split is None:
            worst.splittable = False
            continue
        cells.remove(worst)
        cells.extend(split)


def _placement_volume_factor(
    occurrence: BRepAssemblyOccurrence, /
) -> _ScalarInterval:
    matrix = occurrence.placement.matrix
    rows = tuple(tuple(matrix[row][column] for column in range(3)) for row in range(3))
    determinant = rational_determinant(*rows)
    scale = abs(determinant)
    return _finite_interval((fraction_down(scale), fraction_up(scale)))


@checked
def brep_mass_properties(
    model: BRepModel,
    /,
    *,
    subdivisions: int,
) -> BRepMassProperties:
    """Enclose the positive material volume of every source solid occurrence.

    ``subdivisions`` is an initial interval-integration grid density, not a
    claim of requested accuracy. Planar faces with affine line trims use their
    exact rational boundary integral. Other supported trimmed faces use the
    canonical JAX interval extension on source parameter cells, refining the
    largest unresolved contribution until material positivity is proved or no
    distinct parameter midpoint remains.
    Unsupported trims, interval operations, or unresolved orientation/closure return an
    explicit per-occurrence status and an infinite interval; no tessellation or
    sample-derived finite estimate is published.
    """
    if not x64_enabled():
        raise ValueError(
            "BRepMassProperties.precision must retain binary64 enclosures; got JAX x64 disabled."
        )
    if isinstance(subdivisions, bool) or not isinstance(subdivisions, Integral):
        raise TypeError("subdivisions must be a positive integer.")
    subdivisions_ = positive_integer(int(subdivisions), "subdivisions")
    if model.topology.num_solids == 0:
        raise ValueError("BRepMassProperties requires at least one closed source solid.")
    occurrences = tuple(model.geometry.iter_occurrences())
    if not occurrences:
        raise ValueError("BRepMassProperties requires at least one solid occurrence.")

    bounds = np.tile(
        np.asarray((-np.inf, np.inf), dtype=np.float64), (len(occurrences), 1)
    )
    statuses: list[BRepMassMeasureStatus] = []
    reasons: list[str] = []
    solid_cache: dict[int, tuple[BRepMassMeasureStatus, _ScalarInterval, str]] = {}
    indices = tuple(int(occurrence.solid_index) for occurrence in occurrences)
    paths = tuple(occurrence.path for occurrence in occurrences)
    ids = tuple(
        BRepEntityId(
            model.source_id,
            model.source_revision,
            "solid",
            index,
            occurrence_path=path,
        )
        for index, path in zip(indices, paths, strict=True)
    )
    unbounded: _ScalarInterval = (np.float64(-np.inf), np.float64(np.inf))
    for row, occurrence in enumerate(occurrences):
        solid = indices[row]
        if solid not in solid_cache:
            entry: tuple[BRepMassMeasureStatus, _ScalarInterval, str]
            try:
                measured = _solid_volume_enclosure(model, solid, subdivisions_)
                entry = ("certified", measured, "")
            except _MeasureFailure as error:
                entry = (error.status, unbounded, error.reason)
            except IntervalExtensionError as error:
                entry = (
                    "unsupported",
                    unbounded,
                    f"source interval extension is unavailable: {error}",
                )
            except (FloatingPointError, OverflowError) as error:
                entry = (
                    "unresolved",
                    unbounded,
                    f"source integration could not be represented: {error}",
                )
            solid_cache[solid] = entry
        status, local, reason = solid_cache[solid]
        if status == "certified":
            try:
                placed = _finite_interval(
                    interval_multiply(local, _placement_volume_factor(occurrence))
                )
                if placed[0] <= 0.0:
                    raise _MeasureFailure(
                        "unresolved",
                        "placed solid volume enclosure does not prove positive material volume",
                    )
                bounds[row] = placed
            except _MeasureFailure as error:
                status, reason = error.status, error.reason
        statuses.append(status)
        reasons.append(reason)

    return BRepMassProperties(
        model_id=model.model_id,
        source_id=model.source_id,
        solid_indices=indices,
        occurrence_paths=paths,
        solid_ids=ids,
        status=tuple(statuses),
        reasons=tuple(reasons),
        volume_enclosures=bounds,
    )


def brep_shell_volume_sign(model: BRepModel, shell: int, /) -> int:
    """Return a certified oriented-volume sign for one exact source shell.

    This is intentionally distinct from ``brep_mass_properties``: it determines
    one shell's sign without requiring a positive solid-volume enclosure.
    Unsupported trim or interval extensions remain explicit failures.
    """
    if not isinstance(model, BRepModel):
        raise TypeError("model must be a BRepModel.")
    if isinstance(shell, bool) or not isinstance(shell, Integral):
        raise TypeError("shell must be an integer source shell index.")
    geometry = model.geometry
    shell_index = int(shell)
    if shell_index < 0 or shell_index >= len(geometry.shell_faces):
        raise ValueError("shell must select an existing source shell.")
    faces = geometry.shell_faces[shell_index]
    if not faces:
        raise ValueError("A signed shell volume requires at least one source face.")
    signs = [float(value) for value in np.asarray(model.orientation)]
    for face, shell_orientation in zip(
        faces, geometry.shell_orientations[shell_index], strict=True
    ):
        signs[face] = float(model.orientation[face]) * shell_orientation
    return _shell_volume_sign(
        geometry,
        model.patches,
        np.asarray(model.parameter_bounds, dtype=np.float64),
        model.trim_domains,
        np.asarray(signs, dtype=np.float64),
        faces,
    )


__all__ = [
    "BRepMassMeasureStatus",
    "BRepMassProperties",
    "brep_mass_properties",
    "brep_shell_volume_sign",
]
