#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Prepare source-preserving periodic power images in bounded host arithmetic."""

from __future__ import annotations

import math
from collections.abc import Callable
from contextlib import nullcontext
from fractions import Fraction
from itertools import product
from typing import assert_never

import equinox as eqx
import numpy as np

import phydrax.discretization._periodic_exact as periodic_exact_owner
import phydrax.geometry._periodic_image_frame as periodic_owner

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._meshcore import MeshcoreError, MeshcoreStatus
from .._resource_ticket import HostBatchTicket
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import positive_integer
from ..discretization._coordinate_budget import CoordinateEnclosureBudget
from ..discretization._periodic_cell import PeriodicCell
from ..discretization._periodic_exact import (
    identification_id,
    PeriodicIsometryGroup,
    PeriodicMeshIdentification,
)
from ..linalg._small_batched import prepare_exact_small_linear_actions
from ..typing import checked, ConvertibleToArray
from ._triangulation_host import frozen_array, host_point_array


def _exact_power_translation_dual(
    vectors: np.ndarray, /, *, budget: CoordinateEnclosureBudget
) -> np.ndarray:
    """Solve the Gram system for the translation dual, including partial rank."""
    gram = periodic_owner.periodic_exact_product(vectors, vectors.T, budget=budget)
    actions = prepare_exact_small_linear_actions(
        tuple(tuple(value for value in row) for row in gram),
        tuple(tuple(value for value in row) for row in vectors),
        budget=budget,
    )
    if actions.actions is None:
        raise ValueError(
            "PeriodicPowerPreparation.translation_generators must be linearly independent over the rationals; got a singular Gram system."
        )
    return np.asarray(actions.actions, dtype=object).T


class PeriodicPowerImageCapacityRefusal(MeshcoreError):
    """Retain source data and image counts for pre-materialization refusal."""

    @checked
    def __init__(
        self,
        points: np.ndarray,
        weights: np.ndarray,
        domain: np.ndarray,
        periodic_group: PeriodicMeshIdentification | None,
        requested: int,
        maximum: int,
        stage: str,
        /,
    ) -> None:
        super().__init__(
            MeshcoreStatus.CAPACITY_EXCEEDED,
            f"PeriodicPowerPreparation.{stage} must fit maximum_images; got {requested} images against {maximum}.",
        )
        self.points = frozen_array(points.copy())
        self.weights = frozen_array(weights.copy())
        self.domain_points = frozen_array(domain.copy())
        self.periodic_group = periodic_group
        self.requested_images = requested
        self.completed_images = 0
        self.maximum_images = maximum
        self.stage = stage


class _PowerSource(StrictModule):
    sites: np.ndarray
    domain: np.ndarray
    values: np.ndarray
    limit: int = eqx.field(static=True)


class _FinitePowerOrbits(StrictModule):
    generators: np.ndarray
    orders: tuple[int, ...] = eqx.field(static=True)
    linear_orders: tuple[int, ...] = eqx.field(static=True)
    finite_sites: np.ndarray
    prepared_generators: (
        tuple[tuple[tuple[tuple[Fraction, ...], ...], ...], tuple[int, ...]] | None
    )


class _PowerImageActions(StrictModule):
    rows: np.ndarray
    count: int = eqx.field(static=True)


class _PowerCoordinateBank(StrictModule):
    offsets: np.ndarray
    components: np.ndarray


def _power_source(
    points: ConvertibleToArray,
    weights: ConvertibleToArray,
    domain_points: ConvertibleToArray,
    periodic_group: PeriodicMeshIdentification,
    maximum_images: int,
    *,
    budget: CoordinateEnclosureBudget,
) -> _PowerSource:
    sites = host_point_array(points, "points", (3,))
    domain = host_point_array(domain_points, "domain_points", (3,))
    values = np.asarray(weights, dtype=np.float64)
    periodic_owner.reserve_periodic_exact_terms(
        sites.size + domain.size + values.size, budget=budget
    )
    if not len(sites) or not len(domain):
        raise ValueError(
            "PeriodicPowerPreparation.source must contain sites and carrier points; got an empty bank."
        )
    if values.shape != (len(sites),) or not np.all(np.isfinite(values)):
        raise ValueError(
            f"PeriodicPowerPreparation.weights must be finite with one value per authored site; got shape {values.shape}."
        )
    limit = positive_integer(maximum_images, "maximum_images")
    if periodic_group.ambient_dimension != 3:
        raise ValueError(
            f"PeriodicPowerPreparation.periodic_group must have ambient dimension three; got {periodic_group.ambient_dimension}."
        )
    if len(sites) > limit:
        raise PeriodicPowerImageCapacityRefusal(
            sites,
            values,
            domain,
            periodic_group,
            len(sites),
            limit,
            "original-sites",
        )
    return _PowerSource(sites, domain, values, limit)


def _finite_power_orbits(
    source: _PowerSource,
    periodic_group: PeriodicMeshIdentification,
    *,
    budget: CoordinateEnclosureBudget,
) -> _FinitePowerOrbits:
    sites, values, domain, limit = (
        source.sites,
        source.values,
        source.domain,
        source.limit,
    )
    prepared_generators = None
    match periodic_group:
        case PeriodicCell():
            if not periodic_group.fully_periodic:
                raise ValueError(
                    "PeriodicPowerPreparation.periodic_group.periodic_axes must all be periodic; got a nonperiodic axis."
                )
            vectors = np.asarray(periodic_group.vectors, dtype=np.float64)
            generators = np.repeat(
                np.eye(4, dtype=np.float64)[None], periodic_group.rank, axis=0
            )
            generators[:, :3, 3] = vectors
            orders = (0,) * periodic_group.rank
            linear_orders = (1,) * periodic_group.rank
            periodic_owner.reserve_periodic_exact_terms(sites.size, budget=budget)
            finite_sites = np.asarray(
                [[Fraction(float(x)) for x in site] for site in sites],
                dtype=object,
            )
        case PeriodicIsometryGroup():
            generators = np.asarray(periodic_group.generators, dtype=np.float64)
            orders = periodic_group.orders
            linear_orders = periodic_group.linear_orders
            finite_count = math.prod(linear_orders)
            if finite_count * len(sites) > limit:
                raise PeriodicPowerImageCapacityRefusal(
                    sites,
                    values,
                    domain,
                    periodic_group,
                    finite_count * len(sites),
                    limit,
                    "finite-orbits",
                )
            finite_sites_parts = []
            periodic_owner.reserve_periodic_exact_terms(sites.size, budget=budget)
            original = np.asarray(
                [[Fraction(float(x)) for x in site] for site in sites],
                dtype=object,
            )
            prepared_generators = periodic_exact_owner.exact_periodic_generators(
                periodic_group, budget=budget
            )
            for finite_action in product(*(range(order) for order in linear_orders)):
                matrix = periodic_owner.exact_periodic_group_element(
                    periodic_group,
                    np.asarray(finite_action, dtype=np.int64),
                    prepared_generators=prepared_generators,
                    budget=budget,
                )
                periodic_owner.reserve_periodic_exact_terms(len(sites) * 3, budget=budget)
                finite_sites_parts.append(
                    periodic_owner.periodic_exact_product(
                        original, matrix[:3, :3].T, budget=budget
                    )
                    + matrix[:3, 3]
                )
            finite_sites = np.concatenate(finite_sites_parts)
        case unreachable:
            assert_never(unreachable)
    return _FinitePowerOrbits(
        generators, orders, linear_orders, finite_sites, prepared_generators
    )


def _power_image_actions(
    source: _PowerSource,
    orbits: _FinitePowerOrbits,
    periodic_group: PeriodicMeshIdentification,
    *,
    budget: CoordinateEnclosureBudget,
) -> _PowerImageActions:
    sites, values, domain, limit = (
        source.sites,
        source.values,
        source.domain,
        source.limit,
    )
    generators, orders, linear_orders, finite_sites = (
        orbits.generators,
        orbits.orders,
        orbits.linear_orders,
        orbits.finite_sites,
    )
    translations = [axis for axis, order in enumerate(orders) if order == 0]
    ranges = [range(order) for order in linear_orders]
    if translations:
        periodic_owner.reserve_periodic_exact_terms(len(translations) * 3, budget=budget)
        match periodic_group:
            case PeriodicCell():
                vectors = np.asarray(
                    [
                        [Fraction(float(x)) for x in generators[axis, :3, 3]]
                        for axis in translations
                    ],
                    dtype=object,
                )
            case PeriodicIsometryGroup():
                vectors = np.asarray(
                    [
                        periodic_owner.exact_periodic_group_element(
                            periodic_group,
                            np.asarray(
                                [
                                    linear_orders[index] if index == axis else 0
                                    for index in range(periodic_group.rank)
                                ],
                                dtype=np.int64,
                            ),
                            prepared_generators=orbits.prepared_generators,
                            budget=budget,
                        )[:3, 3]
                        for axis in translations
                    ],
                    dtype=object,
                )
            case unreachable:
                assert_never(unreachable)
        dual = _exact_power_translation_dual(vectors, budget=budget)
        lower, upper = np.min(domain, axis=0), np.max(domain, axis=0)
        corners = tuple(product(*zip(lower, upper, strict=True)))
        reference = [Fraction(float(x)) for x in sites[0]]
        periodic_owner.reserve_periodic_exact_terms(
            len(corners) * 2 * 3 + len(values) + 2, budget=budget
        )
        ceiling = (
            max(
                sum(
                    (Fraction(float(x)) - y) ** 2
                    for x, y in zip(corner, reference, strict=True)
                )
                for corner in corners
            )
            - Fraction(float(values[0]))
            + max(Fraction(float(w)) for w in values)
        )
        radius = math.isqrt(max(0, ceiling.numerator // ceiling.denominator))
        if Fraction(radius * radius) < ceiling:
            radius += 1
        for column, axis in enumerate(translations):
            coefficients = dual[:, column]
            periodic_owner.reserve_periodic_exact_terms(3 * 3, budget=budget)
            carrier_low = sum(
                a * Fraction(float(lower[i] if a >= 0 else upper[i]))
                for i, a in enumerate(coefficients)
            )
            carrier_high = sum(
                a * Fraction(float(upper[i] if a >= 0 else lower[i]))
                for i, a in enumerate(coefficients)
            )
            site_axis = periodic_owner.periodic_exact_product(
                finite_sites, coefficients, budget=budget
            )
            periodic_owner.reserve_periodic_exact_terms(
                2 * len(site_axis) + 5, budget=budget
            )
            reach = radius * sum(abs(a) for a in coefficients)
            low = carrier_low - max(site_axis) - reach
            high = carrier_high - min(site_axis) + reach
            period = linear_orders[axis]
            ranges[axis] = range(
                period * math.ceil(low),
                period * (math.floor(high) + 1),
            )
    actions = product(*ranges)
    action_count = math.prod(len(value) for value in ranges)
    required = action_count * len(sites)
    if required > limit:
        raise PeriodicPowerImageCapacityRefusal(
            sites,
            values,
            domain,
            periodic_group,
            required,
            limit,
            "power-images",
        )
    # Use the authored group frame shared by source support and embedding.
    # Its overlap bank is not used as the power completeness bound above.
    action_rows = np.asarray(tuple(actions), dtype=np.int64).reshape(
        (action_count, periodic_group.rank)
    )
    return _PowerImageActions(action_rows, action_count)


def _power_coordinate_bank(
    source: _PowerSource,
    actions: _PowerImageActions,
    periodic_group: PeriodicMeshIdentification,
    *,
    budget: CoordinateEnclosureBudget,
) -> _PowerCoordinateBank:
    sites, limit, action_rows = source.sites, source.limit, actions.rows
    frame = periodic_owner.prepare_periodic_image_frame(
        sites,
        np.arange(len(sites), dtype=np.int64),
        np.zeros((len(sites), periodic_group.rank), dtype=np.int64),
        periodic_group,
        limit,
        image_exponents=action_rows,
        budget=budget,
    )
    components: list[float] = []
    coordinate_offsets = [0]
    scale = Fraction(2) ** (
        frame.exponent if frame.lattice is not None else 3 * frame.exponent
    )
    for index, action in enumerate(action_rows):
        image = tuple(int(x) for x in action) if frame.lattice is not None else index
        exact_coordinates = frame.image_points(image, budget=budget)
        for integer in exact_coordinates.reshape(-1):
            periodic_owner.reserve_periodic_exact_terms(1, budget=budget)
            remaining = Fraction(int(integer)) * scale
            coordinate: list[float] = []
            while remaining:
                periodic_owner.reserve_periodic_exact_terms(2, budget=budget)
                component = float(remaining)
                if not math.isfinite(component) or component == 0.0:
                    raise ValueError(
                        "PeriodicPowerPreparation.image_coordinates must fit binary64 expansion range; got an unrepresentable component."
                    )
                coordinate.append(component)
                remaining -= Fraction(component)
            components.extend(reversed(coordinate))
            coordinate_offsets.append(len(components))
    return _PowerCoordinateBank(
        np.asarray(coordinate_offsets, dtype=np.int64),
        np.asarray(components, dtype=np.float64),
    )


class PeriodicPowerPreparation(StrictModule, NonTrainableState):
    """Prepare power-relevant images while retaining the authored site and action axes.

    ``image_sites`` indexes authored sites, never materialized coordinates.
    ``image_exponents`` specifies the authored group action before coordinate
    rounding; the numerical image coordinates retain expansions of that action.
    The bank includes every image capable of winning anywhere in the carrier
    bounding box, including weighted sites outside that box.

    Translation duals use ``prepare_exact_small_linear_actions``. ``spent_work``
    includes the coefficient visits and solver work charged by this request.
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
    spent_work: int = eqx.field(static=True)
    native_charged_work: int = eqx.field(static=True)
    maximum_images: int = eqx.field(static=True)
    maximum_work_units: int = eqx.field(static=True)

    @checked
    def __init__(
        self,
        points: ConvertibleToArray,
        weights: ConvertibleToArray,
        domain_points: ConvertibleToArray,
        periodic_group: PeriodicMeshIdentification,
        /,
        *,
        maximum_images: int,
        maximum_work_units: int = 1 << 26,
        budget: CoordinateEnclosureBudget | None = None,
        admit_host_output: Callable[[int], None] | None = None,
    ) -> None:
        work_limit = positive_integer(maximum_work_units, "maximum_work_units")
        image_limit = positive_integer(
            maximum_images, "PeriodicPowerPreparation.maximum_images"
        )
        memory_limit = (
            int(np.iinfo(np.intp).max) if budget is None else budget.maximum_memory_bytes
        )
        if budget is None:
            owned_ticket = HostBatchTicket(work_limit, bytes_upper=memory_limit)
            ledger = CoordinateEnclosureBudget(
                owned_ticket,
                maximum_work_units=work_limit,
                maximum_memory_bytes=memory_limit,
            )
        else:
            owned_ticket = None
            ledger = budget
        starting_work = ledger.work_units
        starting_native = ledger.native_charged_work_units
        with (
            nullcontext() if owned_ticket is None else owned_ticket,
            ledger.activate(),
            ledger.bound_stage(
                work_limit, memory_limit, starting_work_units=starting_work
            ),
            ledger.temporary_scope(),
        ):
            self._prepare(
                points,
                weights,
                domain_points,
                periodic_group,
                maximum_images=image_limit,
                budget=ledger,
                admit_host_output=admit_host_output,
            )
        self.maximum_images = image_limit
        self.maximum_work_units = work_limit
        self.spent_work = ledger.work_units - starting_work
        self.native_charged_work = ledger.native_charged_work_units - starting_native

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[PeriodicPowerPreparation],
        value: PeriodicPowerPreparation,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> PeriodicPowerPreparation:
        """Authenticate stored source fields and preserve the preparation instance."""
        value.validate_restored(budget=budget)
        return value

    @checked
    def validate_restored(
        self, *, budget: CoordinateEnclosureBudget | None = None
    ) -> None:
        """Authenticate restored source, controls, and image banks.

        Revalidation charges current work to the request coefficient
        ledger and leaves archived work counters unchanged.
        """

        match self.periodic_group:
            case PeriodicCell() | PeriodicIsometryGroup():
                self.periodic_group.validate_restored()
            case _:
                raise TypeError(
                    "PeriodicPowerPreparation.periodic_group must be a periodic identification; got a different restored type."
                )
        for value in (self.spent_work, self.native_charged_work):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or value < 0
            ):
                raise ValueError(
                    "PeriodicPowerPreparation.work_counters must be nonnegative integers; got invalid restored counters."
                )
        if (
            self.native_charged_work > self.spent_work
            or self.spent_work > self.maximum_work_units
        ):
            raise ValueError(
                "PeriodicPowerPreparation.work_counters must fit the preparation allowance; got excessive restored work."
            )
        replay = PeriodicPowerPreparation(
            self.points,
            self.weights,
            self.domain_points,
            self.periodic_group,
            maximum_images=self.maximum_images,
            maximum_work_units=self.maximum_work_units,
            budget=budget,
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
            if array_tree_fingerprint(value) != array_tree_fingerprint(rebuilt):
                raise ValueError(
                    f"PeriodicPowerPreparation.{name} must match the retained source law; got different content."
                )
        metadata = (
            ("orders", self.orders, replay.orders),
            ("source_id", self.source_id, replay.source_id),
            ("carrier_id", self.carrier_id, replay.carrier_id),
            ("identification_id", self.identification_id, replay.identification_id),
            ("preparation_id", self.preparation_id, replay.preparation_id),
            ("maximum_images", self.maximum_images, replay.maximum_images),
            ("maximum_work_units", self.maximum_work_units, replay.maximum_work_units),
        )
        for name, value, rebuilt in metadata:
            if value != rebuilt:
                raise ValueError(
                    f"PeriodicPowerPreparation.{name} must match its source identity; got different metadata."
                )
        for name, value, _ in arrays:
            match value:
                case np.ndarray():
                    value.setflags(write=False)
                case _:
                    raise TypeError(
                        f"PeriodicPowerPreparation.{name} must be a host ndarray; got {type(value).__name__}."
                    )

    def _prepare(
        self,
        points: ConvertibleToArray,
        weights: ConvertibleToArray,
        domain_points: ConvertibleToArray,
        periodic_group: PeriodicMeshIdentification,
        /,
        *,
        maximum_images: int,
        budget: CoordinateEnclosureBudget,
        admit_host_output: Callable[[int], None] | None = None,
    ) -> None:

        source = _power_source(
            points, weights, domain_points, periodic_group, maximum_images, budget=budget
        )
        orbits = _finite_power_orbits(source, periodic_group, budget=budget)
        actions = _power_image_actions(source, orbits, periodic_group, budget=budget)
        bank = _power_coordinate_bank(source, actions, periodic_group, budget=budget)
        sites, values, domain = source.sites, source.values, source.domain
        generators, orders = orbits.generators, orbits.orders
        action_rows, action_count = actions.rows, actions.count
        if admit_host_output is not None:
            # Concrete action/image counts, never maximum_images: source and
            # generator copies, site-axis arange/tile, exponent repeat, retained
            # coordinate banks and the largest fingerprint byte serialization.
            # The eight array headers follow the declared 256-byte convention.
            image_count = len(sites) * action_count
            payload = (
                sites.nbytes + values.nbytes + domain.nbytes + generators.nbytes
                + image_count * (4 + action_rows.shape[1] * 8)
                + bank.offsets.nbytes + bank.components.nbytes
            )
            largest_bank = max(
                sites.nbytes, domain.nbytes, generators.nbytes,
                image_count * action_rows.shape[1] * 8,
                bank.offsets.nbytes, bank.components.nbytes,
            )
            admit_host_output(
                8 * 256 + payload + 256 + len(sites) * 4
                + 256 + largest_bank + 16 * 256
            )
        self.points = frozen_array(sites.copy())
        self.weights = frozen_array(values.copy())
        self.domain_points = frozen_array(domain.copy())
        self.periodic_group = periodic_group
        self.generators = frozen_array(generators.copy())
        self.image_sites = frozen_array(
            np.tile(np.arange(len(sites), dtype=np.int32), action_count)
        )
        self.image_exponents = frozen_array(np.repeat(action_rows, len(sites), axis=0))
        self.image_coordinate_offsets = frozen_array(bank.offsets)
        self.image_coordinate_components = frozen_array(bank.components)
        self.orders = orders
        self.carrier_id = canonical_fingerprint(array_tree_fingerprint(domain))
        self.identification_id = identification_id(periodic_group)
        self.source_id = canonical_fingerprint(
            {
                "kind": "periodic-power-source",
                "points": array_tree_fingerprint(self.points),
                "weights": array_tree_fingerprint(self.weights),
                "generators": array_tree_fingerprint(self.generators),
                "orders": orders,
                "identification": self.identification_id,
            }
        )
        self.preparation_id = canonical_fingerprint(
            {
                "kind": "periodic-power-preparation",
                "source": self.source_id,
                "carrier": array_tree_fingerprint(domain),
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
