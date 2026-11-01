#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Prepare exact source-preserving periodic power-image banks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from typing import assert_never, final

import equinox as eqx
import numpy as np

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._exact_algebraic_geometry import (
    ExactScalar,
    RadicalElement,
    rational_coefficient,
)
from ..discretization._periodic_cell import PeriodicCell
from ..discretization._periodic_exact import (
    exact_periodic_element,
    ExactIsometry,
    identification_id,
    PeriodicIsometryGroup,
    PeriodicMeshIdentification,
)
from ..linalg._exact import solve_exact
from ..typing import checked, ConvertibleToArray
from ._periodic_image_frame import prepare_periodic_image_frame


def _frozen_array(value: ConvertibleToArray, /) -> np.ndarray:
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


def _point_bank(value: ConvertibleToArray, name: str, /) -> np.ndarray:
    try:
        points = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as failure:
        raise TypeError(f"{name} must be convertible to binary64 points.") from failure
    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] != 3:
        raise ValueError(
            f"{name} must have nonempty shape (count, 3); got {points.shape}."
        )
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{name} must contain only finite points.")
    return np.array(points, dtype=np.float64, copy=True)


def _validate_group(value: PeriodicMeshIdentification, /) -> PeriodicMeshIdentification:
    if isinstance(value, PeriodicCell):
        return PeriodicCell.from_stored_fields(value)
    if isinstance(value, PeriodicIsometryGroup):
        value.validate_restored()
        return value
    raise TypeError("periodic_group must be PeriodicCell or PeriodicIsometryGroup.")


def _exact_point(
    matrix: ExactIsometry, point: tuple[Fraction, ...], /
) -> tuple[Fraction, ...]:
    dimension = len(matrix) - 1
    if len(point) != dimension:
        raise ValueError("Periodic power action and source point dimensions differ.")
    return tuple(
        sum(
            (matrix[row][column] * point[column] for column in range(dimension)),
            matrix[row][dimension],
        )
        for row in range(dimension)
    )


def _translation_dual(
    vectors: tuple[tuple[Fraction, ...], ...], /
) -> tuple[tuple[Fraction, ...], ...]:
    rank = len(vectors)
    if rank == 0:
        return ()
    dimension = len(vectors[0])
    if any(len(vector) != dimension for vector in vectors):
        raise ValueError("Periodic power translations must share one ambient dimension.")
    gram = tuple(
        tuple(
            sum(
                (left[axis] * right[axis] for axis in range(dimension)),
                Fraction(0),
            )
            for right in vectors
        )
        for left in vectors
    )
    inverse = solve_exact(
        gram,
        tuple(
            tuple(Fraction(int(row == column)) for column in range(rank))
            for row in range(rank)
        ),
        singular_message="Periodic power translation generators must be independent.",
    )
    return tuple(
        tuple(
            sum(
                (
                    inverse[axis][index] * vectors[index][coordinate]
                    for index in range(rank)
                ),
                Fraction(0),
            )
            for axis in range(rank)
        )
        for coordinate in range(dimension)
    )


def _integer_rows(rows: tuple[tuple[int, ...], ...], rank: int, /) -> np.ndarray:
    lower, upper = np.iinfo(np.int64).min, np.iinfo(np.int64).max
    if any(len(row) != rank for row in rows):
        raise ValueError("Periodic power image actions must align with group rank.")
    if any(value < lower or value > upper for row in rows for value in row):
        raise ValueError("Periodic power image exponents are not representable by int64.")
    if len(rows) > np.iinfo(np.intp).max:
        raise ValueError(
            "Periodic power action count is not representable by a host array."
        )
    return np.asarray(rows, dtype=np.int64).reshape((len(rows), rank))


@dataclass(frozen=True, slots=True)
class _PowerSource:
    sites: np.ndarray
    domain: np.ndarray
    values: np.ndarray


@dataclass(frozen=True, slots=True)
class _FinitePowerOrbits:
    generators: np.ndarray
    orders: tuple[int, ...]
    linear_orders: tuple[int, ...]
    finite_sites: np.ndarray


@dataclass(frozen=True, slots=True)
class _PowerImageActions:
    rows: np.ndarray
    count: int


@dataclass(frozen=True, slots=True)
class _PowerCoordinateBank:
    offsets: np.ndarray
    components: np.ndarray


def _power_source(
    points: ConvertibleToArray,
    weights: ConvertibleToArray,
    domain_points: ConvertibleToArray,
    periodic_group: PeriodicMeshIdentification,
    /,
) -> _PowerSource:
    sites = _point_bank(points, "points")
    domain = _point_bank(domain_points, "domain_points")
    try:
        values = np.asarray(weights, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as failure:
        raise TypeError("weights must be convertible to binary64 values.") from failure
    if values.shape != (len(sites),) or not np.all(np.isfinite(values)):
        raise ValueError(
            "weights must be finite with one value per authored periodic power site."
        )
    if periodic_group.ambient_dimension != 3:
        raise ValueError("periodic_group must have ambient dimension three.")
    if isinstance(periodic_group, PeriodicCell) and not periodic_group.fully_periodic:
        raise ValueError(
            "Periodic power preparation requires every lattice axis periodic."
        )
    return _PowerSource(sites, domain, np.array(values, dtype=np.float64, copy=True))


def _finite_power_orbits(
    source: _PowerSource,
    periodic_group: PeriodicMeshIdentification,
    /,
) -> _FinitePowerOrbits:
    original = np.asarray(
        [[Fraction(float(value)) for value in row] for row in source.sites],
        dtype=object,
    )
    match periodic_group:
        case PeriodicCell():
            vectors = np.asarray(periodic_group.vectors, dtype=np.float64)
            generators = np.repeat(
                np.eye(4, dtype=np.float64)[None, :, :], periodic_group.rank, axis=0
            )
            generators[:, :3, 3] = vectors
            orders = (0,) * periodic_group.rank
            linear_orders = (1,) * periodic_group.rank
            finite_sites = original
        case PeriodicIsometryGroup():
            generators = np.asarray(periodic_group.generators, dtype=np.float64)
            orders = periodic_group.orders
            linear_orders = periodic_group.linear_orders
            if any(order <= 0 for order in linear_orders):
                raise ValueError(
                    "Periodic power preparation requires finite linear generator orders."
                )
            finite_count = math.prod(linear_orders)
            if finite_count * len(source.sites) > np.iinfo(np.intp).max:
                raise ValueError(
                    "Periodic power finite-orbit bank is not representable by a host array."
                )
            orbit_parts = []
            for action in product(*(range(order) for order in linear_orders)):
                matrix = exact_periodic_element(
                    periodic_group, np.asarray(action, dtype=np.int64)
                )
                orbit_parts.append(
                    np.asarray(
                        [_exact_point(matrix, tuple(row)) for row in original],
                        dtype=object,
                    )
                )
            finite_sites = np.concatenate(orbit_parts, axis=0)
        case unreachable:
            assert_never(unreachable)
    return _FinitePowerOrbits(
        np.asarray(generators, dtype=np.float64),
        tuple(int(value) for value in orders),
        tuple(int(value) for value in linear_orders),
        finite_sites,
    )


def _power_image_actions(
    source: _PowerSource,
    orbits: _FinitePowerOrbits,
    periodic_group: PeriodicMeshIdentification,
    /,
) -> _PowerImageActions:
    generators, orders, linear_orders, finite_sites = (
        orbits.generators,
        orbits.orders,
        orbits.linear_orders,
        orbits.finite_sites,
    )
    translations = tuple(axis for axis, order in enumerate(orders) if order == 0)
    ranges = [range(order) for order in linear_orders]
    if translations:
        match periodic_group:
            case PeriodicCell():
                vectors = tuple(
                    tuple(Fraction(float(value)) for value in generators[axis, :3, 3])
                    for axis in translations
                )
            case PeriodicIsometryGroup():
                translation_vectors = []
                for axis in translations:
                    exponent = [0] * periodic_group.rank
                    exponent[axis] = linear_orders[axis]
                    matrix = exact_periodic_element(
                        periodic_group, np.asarray(exponent, dtype=np.int64)
                    )
                    translation_vectors.append(
                        tuple(matrix[coordinate][3] for coordinate in range(3))
                    )
                vectors = tuple(translation_vectors)
            case unreachable:
                assert_never(unreachable)
        dual = _translation_dual(vectors)
        lower = tuple(Fraction(float(value)) for value in np.min(source.domain, axis=0))
        upper = tuple(Fraction(float(value)) for value in np.max(source.domain, axis=0))
        corners = tuple(product(*zip(lower, upper, strict=True)))
        reference = tuple(Fraction(float(value)) for value in source.sites[0])
        ceiling = (
            max(
                (
                    sum(
                        ((corner[axis] - reference[axis]) ** 2 for axis in range(3)),
                        Fraction(0),
                    )
                    for corner in corners
                ),
                default=Fraction(0),
            )
            - Fraction(float(source.values[0]))
            + max(Fraction(float(value)) for value in source.values)
        )
        radius = math.isqrt(max(0, ceiling.numerator // ceiling.denominator))
        if Fraction(radius * radius) < ceiling:
            radius += 1
        for column, axis in enumerate(translations):
            coefficients = tuple(dual[coordinate][column] for coordinate in range(3))
            carrier_low = sum(
                (
                    coefficient * (lower[index] if coefficient >= 0 else upper[index])
                    for index, coefficient in enumerate(coefficients)
                ),
                Fraction(0),
            )
            carrier_high = sum(
                (
                    coefficient * (upper[index] if coefficient >= 0 else lower[index])
                    for index, coefficient in enumerate(coefficients)
                ),
                Fraction(0),
            )
            site_axis = tuple(
                sum(
                    (row[index] * coefficients[index] for index in range(3)),
                    Fraction(0),
                )
                for row in finite_sites
            )
            reach = radius * sum((abs(value) for value in coefficients), Fraction(0))
            low = carrier_low - max(site_axis) - reach
            high = carrier_high - min(site_axis) + reach
            period = linear_orders[axis]
            ranges[axis] = range(period * math.ceil(low), period * (math.floor(high) + 1))
    action_count = math.prod(len(interval) for interval in ranges)
    if action_count == 0:
        raise ValueError("Periodic power bounds produced no group image actions.")
    if action_count > np.iinfo(np.intp).max:
        raise ValueError(
            "Periodic power action count is not representable by a host array."
        )
    action_rows = _integer_rows(tuple(product(*ranges)), periodic_group.rank)
    return _PowerImageActions(action_rows, action_count)


def _finite_dyadic_expansion(value: ExactScalar, /) -> tuple[float, ...]:
    rational = rational_coefficient(value, role="periodic power image coordinate")
    if rational.denominator & (rational.denominator - 1):
        raise ValueError(
            "Periodic power image coordinates have no finite binary64 expansion."
        )
    components: list[float] = []
    remaining = rational
    while remaining:
        try:
            component = float(remaining)
        except OverflowError as failure:
            raise ValueError(
                "Periodic power image coordinate is outside finite binary64 expansion range."
            ) from failure
        if not math.isfinite(component) or component == 0.0:
            raise ValueError(
                "Periodic power image coordinate is outside finite binary64 expansion range."
            )
        components.append(component)
        remaining -= Fraction(component)
    return tuple(reversed(components))


def _power_coordinate_bank(
    source: _PowerSource,
    actions: _PowerImageActions,
    periodic_group: PeriodicMeshIdentification,
    /,
) -> _PowerCoordinateBank:
    site_count = len(source.sites)
    coordinate_count = actions.count * site_count * 3
    if coordinate_count + 1 > np.iinfo(np.intp).max:
        raise ValueError(
            "Periodic power coordinate-offset bank is not host-representable."
        )
    if actions.count * site_count > np.iinfo(np.intp).max:
        raise ValueError(
            "Periodic power image-site bank is not representable by a host array."
        )
    roots = np.arange(site_count, dtype=np.int64)
    vertex_exponents = np.zeros((site_count, periodic_group.rank), dtype=np.int64)
    frame = prepare_periodic_image_frame(
        source.sites,
        roots,
        vertex_exponents,
        periodic_group,
        image_exponents=actions.rows,
    )
    scale = Fraction(2) ** (
        frame.exponent if frame.lattice is not None else 3 * frame.exponent
    )
    components: list[float] = []
    offsets = [0]
    for index, action in enumerate(actions.rows):
        image: int | tuple[int, ...] = (
            tuple(int(value) for value in action) if frame.lattice is not None else index
        )
        exact_coordinates = frame.image_points(image)
        for value in exact_coordinates.reshape(-1):
            exact_value: ExactScalar
            if isinstance(value, RadicalElement):
                exact_value = value
            elif isinstance(value, Fraction):
                exact_value = value
            elif isinstance(value, (int, np.integer)) and not isinstance(
                value, (bool, np.bool_)
            ):
                exact_value = Fraction(int(value))
            else:
                raise TypeError(
                    "Periodic image frame returned a non-exact coordinate scalar."
                )
            coordinate = _finite_dyadic_expansion(exact_value * scale)
            components.extend(coordinate)
            if len(components) > np.iinfo(np.intp).max:
                raise ValueError(
                    "Periodic power coordinate component offsets are not host-representable."
                )
            offsets.append(len(components))
    return _PowerCoordinateBank(
        np.asarray(offsets, dtype=np.int64),
        np.asarray(components, dtype=np.float64),
    )


@final
class PeriodicPowerPreparation(StrictModule, NonTrainableState):
    """Host-prepared power images with exact authored-site and action identity.

    ``image_sites`` indexes authored source sites rather than expanded points.
    ``image_exponents`` retains the exact group exponent row for every image
    site. Coordinates are represented by per-component offsets into an exact
    finite binary64 expansion bank; nonrepresentable expansions are refused.
    """

    points: np.ndarray
    weights: np.ndarray
    domain_points: np.ndarray
    periodic_group: PeriodicMeshIdentification
    generators: np.ndarray
    image_sites: np.ndarray
    image_exponents: np.ndarray
    image_coordinate_offsets: np.ndarray
    image_coordinate_components: np.ndarray
    orders: tuple[int, ...] = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    carrier_id: str = eqx.field(static=True)
    identification_id: str = eqx.field(static=True)
    preparation_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        points: ConvertibleToArray,
        weights: ConvertibleToArray,
        domain_points: ConvertibleToArray,
        periodic_group: PeriodicMeshIdentification,
        /,
    ) -> None:
        group = _validate_group(periodic_group)
        source = _power_source(points, weights, domain_points, group)
        orbits = _finite_power_orbits(source, group)
        actions = _power_image_actions(source, orbits, group)
        coordinates = _power_coordinate_bank(source, actions, group)
        sites, values, domain = source.sites, source.values, source.domain
        image_sites = np.tile(np.arange(len(sites), dtype=np.int64), actions.count)
        image_exponents = np.repeat(actions.rows, len(sites), axis=0)
        if len(image_sites) != len(image_exponents):
            raise ValueError("Periodic power source and exponent axes are misaligned.")
        self.points = _frozen_array(sites)
        self.weights = _frozen_array(values)
        self.domain_points = _frozen_array(domain)
        self.periodic_group = group
        self.generators = _frozen_array(orbits.generators)
        self.image_sites = _frozen_array(image_sites)
        self.image_exponents = _frozen_array(image_exponents)
        self.image_coordinate_offsets = _frozen_array(coordinates.offsets)
        self.image_coordinate_components = _frozen_array(coordinates.components)
        self.orders = orbits.orders
        self.identification_id = identification_id(group)
        self.carrier_id = canonical_fingerprint(
            array_tree_fingerprint(self.domain_points)
        )
        self.source_id = canonical_fingerprint(
            {
                "kind": "periodic-power-source",
                "points": array_tree_fingerprint(self.points),
                "weights": array_tree_fingerprint(self.weights),
                "generators": array_tree_fingerprint(self.generators),
                "orders": self.orders,
                "identification": self.identification_id,
            }
        )
        self.preparation_id = canonical_fingerprint(
            {
                "kind": "periodic-power-preparation",
                "source": self.source_id,
                "carrier": array_tree_fingerprint(self.domain_points),
                "site_axis": array_tree_fingerprint(self.image_sites),
                "action_axis": array_tree_fingerprint(self.image_exponents),
                "coordinate_offsets": array_tree_fingerprint(
                    self.image_coordinate_offsets
                ),
                "coordinate_components": array_tree_fingerprint(
                    self.image_coordinate_components
                ),
            }
        )

    @property
    def image_count(self) -> int:
        return len(self.image_sites)

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[PeriodicPowerPreparation],
        value: PeriodicPowerPreparation,
        /,
    ) -> PeriodicPowerPreparation:
        """Authenticate stored inputs and preserve the retained preparation."""
        value.validate_restored()
        return value

    def validate_restored(self, /) -> None:
        """Rebuild the source law and authenticate every retained host bank."""
        group = _validate_group(self.periodic_group)
        replay = PeriodicPowerPreparation(
            self.points,
            self.weights,
            self.domain_points,
            group,
        )
        arrays = (
            ("points", self.points, replay.points),
            ("weights", self.weights, replay.weights),
            ("domain_points", self.domain_points, replay.domain_points),
            ("generators", self.generators, replay.generators),
            ("image_sites", self.image_sites, replay.image_sites),
            ("image_exponents", self.image_exponents, replay.image_exponents),
            (
                "image_coordinate_offsets",
                self.image_coordinate_offsets,
                replay.image_coordinate_offsets,
            ),
            (
                "image_coordinate_components",
                self.image_coordinate_components,
                replay.image_coordinate_components,
            ),
        )
        for name, value, rebuilt in arrays:
            if not isinstance(value, np.ndarray):
                raise TypeError(
                    f"PeriodicPowerPreparation.{name} must be a host ndarray."
                )
            if array_tree_fingerprint(value) != array_tree_fingerprint(rebuilt):
                raise ValueError(
                    f"PeriodicPowerPreparation.{name} does not match its retained source law."
                )
            value.setflags(write=False)
        metadata = (
            ("orders", self.orders, replay.orders),
            ("source_id", self.source_id, replay.source_id),
            ("carrier_id", self.carrier_id, replay.carrier_id),
            ("identification_id", self.identification_id, replay.identification_id),
            ("preparation_id", self.preparation_id, replay.preparation_id),
        )
        for name, value, rebuilt in metadata:
            if value != rebuilt:
                raise ValueError(
                    f"PeriodicPowerPreparation.{name} does not match its source identity."
                )
