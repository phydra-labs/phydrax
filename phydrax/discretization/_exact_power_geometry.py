#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from typing import final, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, core
from jax.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


if TYPE_CHECKING:
    from ..geometry._triangulation import PeriodicPowerPreparation
    from ..meshing._controls import PeriodicConstraint
    from ..meshing._volume_generation import PiecewiseLinearComplex


type ExactPoint = tuple[Fraction, Fraction, Fraction]


@dataclass(frozen=True, slots=True)
class PreparedExactPowerGeometry:
    """Bounded host proof of the current source, never an alternate authority."""

    vertices: tuple[ExactPoint, ...]
    barycentric: tuple[tuple[Fraction, ...], ...]
    source_id: str
    operation_count: int
    maximum_integer_bits: int
    maximum_rounding_error: Fraction
    condition_upper_bounds: tuple[Fraction, ...]

    @property
    def rounded_vertices(self) -> np.ndarray:
        return np.asarray(self.vertices, dtype=np.float64)


@final
class ExactPowerCellGeometrySource(StrictModule, NonTrainableState):
    """Ideal radical-plane intersections with independently verified RNE carriers.

    Dynamic binary64 leaves define the source. CSR site witnesses contain all
    equal-power sites; each carrier row names a simplex of the authored tetrahedral
    decomposition. Exact rational preparation is explicitly host-only. It must be
    repeated after changing source leaves; fixed-topology derivatives are not
    claimed by this preparation API.
    """

    site_points: Array
    site_weights: Array
    carrier_points: Array
    carrier_tets: Array
    vertex_site_offsets: Array
    vertex_sites: Array
    vertex_carriers: Array
    periodic_preparation: object | None
    periodic_source_binding: str | None = eqx.field(static=True)
    domain_source: object | None
    periodic_constraints: tuple[object, ...]
    authority_binding: str | None = eqx.field(static=True)
    maximum_work: int = eqx.field(static=True)
    maximum_bits: int = eqx.field(static=True)
    maximum_condition: float = eqx.field(static=True)

    def __init__(
        self,
        site_points: ArrayLike,
        site_weights: ArrayLike,
        carrier_points: ArrayLike,
        carrier_tets: ArrayLike,
        vertex_site_offsets: ArrayLike,
        vertex_sites: ArrayLike,
        vertex_carriers: ArrayLike,
        /,
        *,
        maximum_work: int = 10_000_000,
        maximum_bits: int = 16384,
        maximum_condition: float = 1e12,
        periodic_preparation: object | None = None,
        domain_source: object | None = None,
        periodic_constraints: tuple[object, ...] = (),
    ) -> None:
        points, weights, carrier = (
            np.asarray(site_points, dtype=np.float64),
            np.asarray(site_weights, dtype=np.float64),
            np.asarray(carrier_points, dtype=np.float64),
        )
        raw_indices = tuple(
            np.asarray(value)
            for value in (
                carrier_tets,
                vertex_site_offsets,
                vertex_sites,
                vertex_carriers,
            )
        )
        if any(value.dtype.kind not in "iu" for value in raw_indices):
            raise TypeError("Exact power witnesses require integer arrays.")
        tets, offsets, sites, simplices = tuple(
            np.asarray(value, dtype=np.int64) for value in raw_indices
        )
        if (
            points.ndim != 2
            or points.shape[1] != 3
            or weights.shape != (points.shape[0],)
            or not points.shape[0]
        ):
            raise ValueError(
                "Exact power sources require nonempty 3-D sites and aligned weights."
            )
        if (
            carrier.ndim != 2
            or carrier.shape[1] != 3
            or tets.ndim != 2
            or tets.shape[1] != 4
        ):
            raise ValueError("Exact power carriers require 3-D points and tetrahedra.")
        if not all(np.all(np.isfinite(value)) for value in (points, weights, carrier)):
            raise ValueError("Exact power source numerical leaves must be finite.")
        if (
            np.any(tets < 0)
            or np.any(tets >= carrier.shape[0])
            or any(len(set(row)) != 4 for row in tets.tolist())
        ):
            raise ValueError(
                "Exact power carrier tetrahedra contain invalid vertex indices."
            )
        if (
            simplices.ndim != 2
            or simplices.shape[1] != 4
            or offsets.shape != (simplices.shape[0] + 1,)
        ):
            raise ValueError(
                "Exact power vertex witnesses require aligned CSR and padded simplex rows."
            )
        if (
            sites.ndim != 1
            or offsets[0] != 0
            or offsets[-1] != sites.size
            or np.any(np.diff(offsets) < 1)
        ):
            raise ValueError(
                "Exact power site witnesses require a nonempty CSR row per vertex."
            )
        if periodic_preparation is not None:
            from ..geometry._triangulation import PeriodicPowerPreparation

            if not isinstance(periodic_preparation, PeriodicPowerPreparation):
                raise TypeError(
                    "Periodic exact power requires its retained PeriodicPowerPreparation."
                )
            if (
                array_tree_fingerprint(points)
                != array_tree_fingerprint(periodic_preparation.points)
                or array_tree_fingerprint(weights)
                != array_tree_fingerprint(periodic_preparation.weights)
                or canonical_fingerprint(array_tree_fingerprint(carrier))
                != periodic_preparation.carrier_id
            ):
                raise ValueError(
                    "Periodic exact power requires original source and carrier bytes."
                )
        site_count = (
            points.shape[0]
            if periodic_preparation is None
            else len(periodic_preparation.image_sites)
        )
        if np.any(sites < 0) or np.any(sites >= site_count):
            raise ValueError("Exact power witnesses index undeclared sites or images.")
        for vertex in range(simplices.shape[0]):
            start, stop = offsets[vertex], offsets[vertex + 1]
            row = simplices[vertex : vertex + 1].reshape((4,))
            equal = sites[start:stop]
            valid = row[row >= 0]
            if (
                not valid.size
                or np.any(row < -1)
                or np.any(valid >= carrier.shape[0])
                or not np.array_equal(row >= 0, np.arange(4) < valid.size)
            ):
                raise ValueError(
                    "Exact power carrier witnesses require valid-prefix simplex indices."
                )
            if np.any(np.diff(equal) <= 0) or np.any(np.diff(valid) <= 0):
                raise ValueError(
                    "Exact power witness indices must be canonical, sorted and unique."
                )
        for name, value in (
            ("maximum_work", maximum_work),
            ("maximum_bits", maximum_bits),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if not math.isfinite(maximum_condition) or maximum_condition < 1:
            raise ValueError("maximum_condition must be finite and at least one.")
        from ..meshing._controls import PeriodicConstraint
        from ..meshing._volume_generation import PiecewiseLinearComplex

        if domain_source is not None and not isinstance(
            domain_source, PiecewiseLinearComplex
        ):
            raise TypeError(
                "Exact power domain authority must be the original PiecewiseLinearComplex."
            )
        if not isinstance(periodic_constraints, tuple) or any(
            not isinstance(value, PeriodicConstraint) for value in periodic_constraints
        ):
            raise TypeError(
                "Exact power periodic controls require their original "
                "PeriodicConstraint tuple."
            )
        controls = tuple(
            value
            for value in periodic_constraints
            if isinstance(value, PeriodicConstraint)
        )
        if controls and (domain_source is None or periodic_preparation is None):
            raise ValueError(
                "Original periodic controls require their original domain "
                "and preparation authority."
            )
        self.site_points, self.site_weights, self.carrier_points = tuple(
            jnp.asarray(value) for value in (points, weights, carrier)
        )
        (
            self.carrier_tets,
            self.vertex_site_offsets,
            self.vertex_sites,
            self.vertex_carriers,
        ) = tuple(jnp.asarray(value) for value in (tets, offsets, sites, simplices))
        self.maximum_work, self.maximum_bits = maximum_work, maximum_bits
        self.maximum_condition = float(maximum_condition)
        self.periodic_preparation = periodic_preparation
        self.periodic_source_binding = (
            None
            if periodic_preparation is None
            else _power_preparation_binding(periodic_preparation)
        )
        self.domain_source, self.periodic_constraints = domain_source, controls
        self.authority_binding = (
            None
            if domain_source is None
            else _power_domain_binding(domain_source, controls)
        )

    def _periodic_owner(self) -> PeriodicPowerPreparation | None:
        from ..geometry._triangulation import PeriodicPowerPreparation

        preparation = self.periodic_preparation
        if preparation is not None and not isinstance(
            preparation, PeriodicPowerPreparation
        ):
            raise RuntimeError("Exact power periodic preparation changed type.")
        return preparation

    def _domain_owner(
        self,
    ) -> tuple[PiecewiseLinearComplex | None, tuple[PeriodicConstraint, ...]]:
        from ..meshing._controls import PeriodicConstraint
        from ..meshing._volume_generation import PiecewiseLinearComplex

        domain = self.domain_source
        constraints = tuple(
            value
            for value in self.periodic_constraints
            if isinstance(value, PeriodicConstraint)
        )
        if domain is not None and not isinstance(domain, PiecewiseLinearComplex):
            raise RuntimeError("Exact power domain owner changed type.")
        if len(constraints) != len(self.periodic_constraints):
            raise RuntimeError("Exact power periodic controls changed type.")
        return domain, constraints

    @property
    def source_id(self) -> str:
        preparation = self._periodic_owner()
        domain, constraints = self._domain_owner()
        return canonical_fingerprint(
            {
                "kind": "exact-power-cell-source",
                "source_arrays": array_tree_fingerprint(
                    (
                        self.site_points,
                        self.site_weights,
                        self.carrier_points,
                        self.carrier_tets,
                        self.vertex_site_offsets,
                        self.vertex_sites,
                        self.vertex_carriers,
                    )
                ),
                "maximum_work": self.maximum_work,
                "maximum_bits": self.maximum_bits,
                "maximum_condition": self.maximum_condition,
                **(
                    {}
                    if preparation is None
                    else {"periodic_preparation": _power_preparation_binding(preparation)}
                ),
                **(
                    {}
                    if domain is None
                    else {"domain_authority": _power_domain_binding(domain, constraints)}
                ),
            }
        )

    def prepare(
        self, rounded_vertices: ArrayLike | None = None, /
    ) -> PreparedExactPowerGeometry:
        if any(
            isinstance(value, core.Tracer) for value in jax.tree_util.tree_leaves(self)
        ):
            raise TypeError(
                "Exact power preparation is host-only; no source derivative is claimed."
            )
        domain, constraints = self._domain_owner()
        if (
            domain is not None
            and _power_domain_binding(domain, constraints) != self.authority_binding
        ):
            raise ValueError(
                "Exact power original domain/control authority no longer "
                "matches its immutable source binding."
            )
        rounded = (
            None
            if rounded_vertices is None
            else np.asarray(rounded_vertices, dtype=np.float64)
        )
        if rounded is not None and (
            rounded.shape != (self.vertex_carriers.shape[0], 3)
            or not np.all(np.isfinite(rounded))
        ):
            raise ValueError(
                "Exact power RNE carrier coordinates must align with source vertex witnesses."
            )
        budget = _ExactPowerBudget(self.maximum_work, self.maximum_bits)
        points = tuple(
            tuple(Fraction(float(value)) for value in row)
            for row in np.asarray(self.site_points)
        )
        weights = tuple(Fraction(float(value)) for value in np.asarray(self.site_weights))
        carrier = tuple(
            tuple(Fraction(float(value)) for value in row)
            for row in np.asarray(self.carrier_points)
        )
        budget.charge(
            0, (*weights, *(value for row in (*points, *carrier) for value in row))
        )
        preparation = self._periodic_owner()
        if preparation is not None:
            if _power_preparation_binding(preparation) != self.periodic_source_binding:
                raise ValueError(
                    "Periodic exact power preparation no longer matches its immutable source binding."
                )
            if (
                array_tree_fingerprint(self.site_points)
                != array_tree_fingerprint(preparation.points)
                or array_tree_fingerprint(self.site_weights)
                != array_tree_fingerprint(preparation.weights)
                or canonical_fingerprint(array_tree_fingerprint(self.carrier_points))
                != preparation.carrier_id
            ):
                raise ValueError(
                    "Periodic exact power source leaves no longer match the retained original preparation."
                )
            original_points, original_weights = points, weights
            matrices = tuple(
                tuple(tuple(Fraction(float(value)) for value in row) for row in matrix)
                for matrix in preparation.generators
            )

            images = []
            image_weights = []
            for owner, exponents in zip(
                preparation.image_sites, preparation.image_exponents, strict=True
            ):
                _charge_power_source_query()
                action = _prepare_power_action(
                    matrices,
                    preparation.orders,
                    tuple(int(value) for value in exponents),
                    budget,
                )
                point = original_points[int(owner)]
                image = tuple(
                    sum(
                        (action[axis][column] * point[column] for column in range(3)),
                        action[axis][3],
                    )
                    for axis in range(3)
                )
                budget.charge(24, image)
                images.append(image)
                image_weights.append(original_weights[int(owner)])
            points, weights = tuple(images), tuple(image_weights)
        from ..linalg._small_batched import prepare_exact_small_linear_actions

        for row in np.asarray(self.carrier_tets).tolist():
            matrix = tuple(
                tuple(
                    carrier[row[column + 1]][axis] - carrier[row[0]][axis]
                    for column in range(3)
                )
                for axis in range(3)
            )
            budget.charge(4 * 3**3)
            rank = prepare_exact_small_linear_actions(matrix, ((Fraction(0),),) * 3)
            budget.charge(0, (rank.determinant,))
            if not rank.successful:
                raise ValueError(
                    "Exact power carrier tetrahedron has deficient source rank."
                )
        tets = tuple(frozenset(row) for row in np.asarray(self.carrier_tets).tolist())
        offsets, sites = (
            np.asarray(self.vertex_site_offsets),
            np.asarray(self.vertex_sites),
        )
        vertices, barycentric = [], []
        conditions = []
        rounding_error = Fraction(0)
        for vertex, simplex in enumerate(np.asarray(self.vertex_carriers)):
            _charge_power_source_query()
            equal = tuple(sites[offsets[vertex] : offsets[vertex + 1]].tolist())
            indices = tuple(simplex[simplex >= 0].tolist())
            budget.charge(len(tets))
            if not any(frozenset(indices) <= tet for tet in tets):
                raise ValueError(
                    "Exact power carrier witness is not an authored tetrahedral simplex."
                )
            coordinates, coefficients, condition = _prepare_vertex(
                points, weights, carrier, equal, indices, budget
            )
            if condition > Fraction(self.maximum_condition):
                raise ValueError(
                    "Exact power vertex construction exceeds its certified condition upper bound."
                )
            expected = np.asarray(
                tuple(float(value) for value in coordinates), dtype=np.float64
            )
            if rounded is not None and not np.array_equal(
                expected.view(np.uint64), rounded[vertex].view(np.uint64)
            ):
                raise ValueError(
                    "Power-cell numerical vertices are not the RNE of their exact source intersections."
                )
            rounding_error = max(
                rounding_error,
                *(
                    abs(value - Fraction(float(rne)))
                    for value, rne in zip(coordinates, expected, strict=True)
                ),
            )
            vertices.append(coordinates)
            barycentric.append(coefficients)
            conditions.append(condition)
        return PreparedExactPowerGeometry(
            tuple(vertices),
            tuple(barycentric),
            self.source_id,
            budget.work,
            budget.bits,
            rounding_error,
            tuple(conditions),
        )


class _ExactPowerBudget:
    def __init__(self, maximum_work: int, maximum_bits: int, /) -> None:
        self.maximum_work, self.maximum_bits = maximum_work, maximum_bits
        self.work, self.bits = 0, 0

    def charge(self, work: int, values: tuple[Fraction, ...] = (), /) -> None:
        self.work += work
        self.bits = (
            max(
                self.bits,
                *(
                    max(value.numerator.bit_length(), value.denominator.bit_length())
                    for value in values
                ),
            )
            if values
            else self.bits
        )
        if self.work > self.maximum_work:
            raise ValueError("Exact power construction work budget exhausted.")
        if self.bits > self.maximum_bits:
            raise ValueError("Exact power construction integer-bit budget exhausted.")
        from ._coordinate_enclosure import _COORDINATE_BUDGET

        ledger = _COORDINATE_BUDGET.get()
        if ledger is not None:
            ledger.reserve(work)
            ledger.charge_native_work(work)
            return
        from .._meshcore import current_native_execution_budget

        execution = current_native_execution_budget()
        if execution is not None:
            execution.charge(work=work)


def _prepare_vertex(
    points: tuple[tuple[Fraction, ...], ...],
    weights: tuple[Fraction, ...],
    carrier: tuple[tuple[Fraction, ...], ...],
    equal: tuple[int, ...],
    indices: tuple[int, ...],
    budget: _ExactPowerBudget,
    /,
) -> tuple[ExactPoint, tuple[Fraction, ...], Fraction]:
    from ..linalg._small_batched import prepare_exact_small_linear_actions

    anchor = equal[0]
    planes = tuple(
        (
            tuple(2 * (q - p) for p, q in zip(points[anchor], points[site], strict=True)),
            sum(
                (
                    q * q - p * p
                    for p, q in zip(points[anchor], points[site], strict=True)
                ),
                Fraction(0),
            )
            + weights[anchor]
            - weights[site],
        )
        for site in equal[1:]
    )
    size = len(indices)
    equations = tuple(
        tuple(
            sum((normal[axis] * carrier[index][axis] for axis in range(3)), Fraction(0))
            for index in indices
        )
        for normal, _ in planes
    )
    budget.charge(
        6 * len(planes) * size, tuple(value for row in equations for value in row)
    )
    coefficients = None
    condition = Fraction(0)
    for selected in combinations(range(len(planes)), size - 1):
        matrix = ((Fraction(1),) * size, *(equations[index] for index in selected))
        right = ((Fraction(1),), *((planes[index][1],) for index in selected))
        budget.charge(4 * size**3)
        solved = prepare_exact_small_linear_actions(matrix, right)
        if solved.actions is not None:
            coefficients = tuple(row[0] for row in solved.actions)
            row_scale = math.prod(max(abs(value) for value in row) for row in matrix)
            condition = (
                Fraction(size * size * math.factorial(size - 1))
                * row_scale
                / abs(solved.determinant)
            )
            budget.charge(0, coefficients)
            break
    if coefficients is None:
        raise ValueError("Exact power vertex witnesses have deficient intersection rank.")
    if min(coefficients) < 0:
        raise ValueError("Exact power intersection is outside its carrier simplex.")
    coordinates = tuple(
        sum(
            (
                coefficient * carrier[index][axis]
                for coefficient, index in zip(coefficients, indices, strict=True)
            ),
            Fraction(0),
        )
        for axis in range(3)
    )
    budget.charge(6 * size, coordinates)
    powers = tuple(
        sum(
            (
                (coordinate - point) ** 2
                for coordinate, point in zip(coordinates, site, strict=True)
            ),
            Fraction(0),
        )
        - weight
        for site, weight in zip(points, weights, strict=True)
    )
    budget.charge(12 * len(points), powers)
    minimum = powers[anchor]
    actual_equal = tuple(index for index, value in enumerate(powers) if value == minimum)
    if actual_equal != equal or any(value < minimum for value in powers):
        raise ValueError(
            "Exact power vertex equal-site witness contradicts the original site power relations."
        )
    return (coordinates[0], coordinates[1], coordinates[2]), coefficients, condition


def _power_domain_binding(
    domain: PiecewiseLinearComplex,
    constraints: tuple[PeriodicConstraint, ...],
) -> str:
    return canonical_fingerprint(
        {
            "kind": "original-power-domain-controls",
            "domain_id": domain.complex_id,
            "domain_arrays": array_tree_fingerprint(domain),
            "regions": domain.region_ids,
            "boundary": domain.boundary,
            "constraints": tuple(
                {
                    "constraint_id": value.constraint_id,
                    "arrays": array_tree_fingerprint(value),
                    "scopes": tuple(
                        (
                            scope.scope_id,
                            scope.source_id,
                            scope.source_revision,
                            str(scope.entity_kind),
                            scope.entity_dimension,
                            scope.entity_set_id,
                        )
                        for scope in (value.source_scope, value.target_scope)
                    ),
                    "tolerance": value.tolerance,
                    "conforming_required": value.conforming_required,
                    "orientation_preserving": value.orientation_preserving,
                }
                for value in constraints
            ),
        }
    )


def _charge_power_source_query() -> None:
    from .._meshcore import charge_native_geometry_queries

    charge_native_geometry_queries(1, work_units=0)


def _power_preparation_binding(
    preparation: PeriodicPowerPreparation,
) -> str:
    return canonical_fingerprint(
        {
            "preparation_id": preparation.preparation_id,
            "arrays": array_tree_fingerprint(
                (
                    preparation.points,
                    preparation.weights,
                    preparation.domain_points,
                    preparation.periodic_group,
                    preparation.generators,
                    preparation.image_sites,
                    preparation.image_exponents,
                    preparation.image_coordinate_offsets,
                    preparation.image_coordinate_components,
                )
            ),
            "orders": preparation.orders,
            "carrier_id": preparation.carrier_id,
            "identification_id": preparation.identification_id,
            "maximum_images": preparation.maximum_images,
            "maximum_work_units": preparation.maximum_work_units,
        }
    )


def _prepare_power_action(
    matrices: tuple,
    orders: tuple[int, ...],
    exponents: tuple[int, ...],
    budget: _ExactPowerBudget,
) -> tuple:
    """Charge canonical action term visits, including refusal prefixes."""
    import sys

    from ._coordinate_enclosure import coordinate_enclosure_budget
    from ._periodic_topology import _exact_periodic_element

    ledger = coordinate_enclosure_budget(
        max(0, budget.maximum_work - budget.work), sys.maxsize
    )
    start = ledger.work_units
    visits = 16
    for order, exponent in zip(orders, exponents, strict=True):
        if order:
            exponent %= order
            visits += (
                16 + 64 * (exponent.bit_count() + max(0, exponent.bit_length() - 1)) + 64
            )
        else:
            visits += 16 + 3 + 64
    try:
        with (
            ledger.activate(),
            ledger.bound_stage(
                max(0, budget.maximum_work - budget.work),
                ledger.maximum_memory_bytes,
                starting_work_units=start,
            ),
            ledger.temporary_scope(),
        ):
            ledger.admit_work_bound(visits)
            ledger.reserve(0, 48 * (128 + 2 * ((budget.maximum_bits + 7) // 8)))
            action = _exact_periodic_element(matrices, orders, exponents)
    finally:
        work = ledger.work_units - start
        budget.work += work
        ledger.charge_native_work(work)
    budget.charge(0, tuple(value for row in action for value in row))
    return action


@final
class ExactPowerCellGeometryLinearActionSource(StrictModule, NonTrainableState):
    """Original-parent convex coefficients, optionally followed by a retained group action."""

    parent: object
    periodic_preparation: object | None
    vertex_parents: Array
    vertex_actions: Array
    vertex_coefficients: tuple[tuple[Fraction, ...], ...] = eqx.field(static=True)
    parent_source_id: str = eqx.field(static=True)
    construction_id: str = eqx.field(static=True)
    maximum_work: int = eqx.field(static=True)
    maximum_bits: int = eqx.field(static=True)
    maximum_condition: float = eqx.field(static=True)

    def __init__(
        self,
        parent: object,
        vertex_parents: ArrayLike,
        vertex_coefficients: tuple[tuple[Fraction, ...], ...],
        vertex_actions: ArrayLike,
        /,
        *,
        periodic_preparation: object | None = None,
    ) -> None:
        from ..geometry._triangulation import PeriodicPowerPreparation

        if not isinstance(
            parent,
            (
                ExactPowerCellGeometrySource,
                ExactPowerCellGeometryRestrictionSource,
                ExactPowerCellGeometryLinearActionSource,
            ),
        ):
            raise TypeError("Exact linear actions require an owning exact power parent.")
        if periodic_preparation is not None and not isinstance(
            periodic_preparation, PeriodicPowerPreparation
        ):
            raise TypeError(
                "Exact group actions require the retained periodic preparation."
            )
        root_owner: object = parent
        while not isinstance(root_owner, ExactPowerCellGeometrySource):
            if isinstance(root_owner, ExactPowerCellGeometryRestrictionSource):
                root_owner = root_owner.parent
            elif isinstance(root_owner, ExactPowerCellGeometryLinearActionSource):
                root_owner = root_owner.parent
            else:
                raise RuntimeError("Exact power parent chain changed type.")
        root = root_owner
        root_preparation = root._periodic_owner()
        if periodic_preparation is None:
            if root_preparation is not None:
                raise ValueError(
                    "A periodic original source cannot discard its group authority."
                )
        elif root_preparation is None or _power_preparation_binding(
            root_preparation
        ) != _power_preparation_binding(periodic_preparation):
            raise ValueError(
                "Exact linear actions require their original periodic source binding."
            )
        indices, actions = np.asarray(vertex_parents), np.asarray(vertex_actions)
        if indices.dtype.kind not in "iu" or actions.dtype.kind not in "iu":
            raise TypeError(
                "Exact linear-action parent indices and exponents must be integers."
            )
        if (
            indices.ndim != 2
            or not indices.shape[1]
            or actions.shape
            != (
                len(indices),
                0 if periodic_preparation is None else len(periodic_preparation.orders),
            )
        ):
            raise ValueError(
                "Exact linear-action witnesses require aligned parent and generator rows."
            )
        size = (
            parent.vertex_carriers.shape[0]
            if isinstance(parent, ExactPowerCellGeometrySource)
            else parent.vertex_parents.shape[0]
        )
        if np.any(indices < -1) or np.any(indices >= size):
            raise ValueError(
                "Exact linear-action witnesses index undeclared parent vertices."
            )
        coefficients = tuple(tuple(row) for row in vertex_coefficients)
        if len(coefficients) != len(indices):
            raise ValueError(
                "Exact linear-action coefficients must align with parent rows."
            )
        for row, values in zip(indices, coefficients, strict=True):
            if len(values) != len(row) or any(
                not isinstance(value, Fraction) for value in values
            ):
                raise TypeError(
                    "Exact linear-action coefficients must be original exact Fractions."
                )
            active = row >= 0
            if not np.any(active) or not np.array_equal(
                active, np.arange(len(row)) < np.count_nonzero(active)
            ):
                raise ValueError(
                    "Exact linear-action parents require a nonempty valid prefix and -1 padding."
                )
            if (
                min(values) < 0
                or sum(values, Fraction(0)) != 1
                or any(
                    value != 0
                    for value, valid in zip(values, active, strict=True)
                    if not valid
                )
                or len(set(row[active].tolist())) != np.count_nonzero(active)
            ):
                raise ValueError(
                    "Exact linear-action witnesses require distinct active parents and convex unit coefficients."
                )
        for axis, order in enumerate(
            () if periodic_preparation is None else periodic_preparation.orders
        ):
            if order and (
                np.any(actions[:, axis] < 0) or np.any(actions[:, axis] >= order)
            ):
                raise ValueError("Finite exact group exponents must be canonical.")
        self.parent, self.periodic_preparation = parent, periodic_preparation
        self.vertex_parents, self.vertex_actions = (
            jnp.asarray(indices),
            jnp.asarray(actions),
        )
        self.vertex_coefficients, self.parent_source_id = coefficients, parent.source_id
        self.maximum_work, self.maximum_bits, self.maximum_condition = (
            parent.maximum_work,
            parent.maximum_bits,
            parent.maximum_condition,
        )
        self.construction_id = self.source_id

    def _owners(
        self,
    ) -> tuple[
        ExactPowerCellGeometrySource
        | ExactPowerCellGeometryRestrictionSource
        | ExactPowerCellGeometryLinearActionSource,
        PeriodicPowerPreparation | None,
    ]:
        from ..geometry._triangulation import PeriodicPowerPreparation

        parent = self.parent
        preparation = self.periodic_preparation
        if not isinstance(
            parent,
            (
                ExactPowerCellGeometrySource,
                ExactPowerCellGeometryRestrictionSource,
                ExactPowerCellGeometryLinearActionSource,
            ),
        ):
            raise RuntimeError("Exact power parent owner changed type.")
        if preparation is not None and not isinstance(
            preparation, PeriodicPowerPreparation
        ):
            raise RuntimeError("Exact power periodic owner changed type.")
        return parent, preparation

    @property
    def source_id(self) -> str:
        parent, preparation = self._owners()
        return canonical_fingerprint(
            {
                "kind": (
                    "exact-power-cell-convex-parent"
                    if preparation is None
                    else "exact-power-cell-linear-action"
                ),
                "parent": parent.source_id,
                "preparation": (
                    None
                    if preparation is None
                    else _power_preparation_binding(preparation)
                ),
                "indices_actions": array_tree_fingerprint(
                    (self.vertex_parents, self.vertex_actions)
                ),
                "coefficients": tuple(
                    tuple((value.numerator, value.denominator) for value in row)
                    for row in self.vertex_coefficients
                ),
                "maximum_work": self.maximum_work,
                "maximum_bits": self.maximum_bits,
                "maximum_condition": self.maximum_condition,
            }
        )

    def prepare(
        self, rounded_vertices: ArrayLike | None = None, /
    ) -> PreparedExactPowerGeometry:
        if any(
            isinstance(value, core.Tracer) for value in jax.tree_util.tree_leaves(self)
        ):
            raise TypeError("Exact linear-action preparation is host-only.")
        owner, preparation = self._owners()
        if owner.source_id != self.parent_source_id:
            raise ValueError(
                "Exact linear-action original parent coefficients are bound "
                "to an immutable source."
            )
        if self.source_id != self.construction_id:
            raise ValueError(
                "Exact linear-action witnesses are bound to an immutable "
                "source construction."
            )
        parent = owner.prepare()
        budget = _ExactPowerBudget(self.maximum_work, self.maximum_bits)
        budget.work, budget.bits = parent.operation_count, parent.maximum_integer_bits
        rounded = (
            None
            if rounded_vertices is None
            else np.asarray(rounded_vertices, dtype=np.float64)
        )
        if rounded is not None and (
            rounded.shape != (len(self.vertex_parents), 3)
            or not np.all(np.isfinite(rounded))
        ):
            raise ValueError(
                "Exact linear-action RNE carriers must align with source witnesses."
            )
        matrices = (
            ()
            if preparation is None
            else tuple(
                tuple(tuple(Fraction(float(value)) for value in row) for row in matrix)
                for matrix in preparation.generators
            )
        )
        result, conditions = [], []
        error = Fraction(0)
        for vertex, (indices, coefficients, exponents) in enumerate(
            zip(
                np.asarray(self.vertex_parents),
                self.vertex_coefficients,
                np.asarray(self.vertex_actions),
                strict=True,
            )
        ):
            _charge_power_source_query()
            active = indices >= 0
            coefficients = tuple(
                value for value, valid in zip(coefficients, active, strict=True) if valid
            )
            indices = indices[active]
            original = tuple(
                sum(
                    (
                        coefficient * parent.vertices[int(index)][axis]
                        for index, coefficient in zip(indices, coefficients, strict=True)
                    ),
                    Fraction(0),
                )
                for axis in range(3)
            )
            budget.charge(6 * len(indices), (*original, *coefficients))
            if preparation is None:
                point = original
            else:
                action = _prepare_power_action(
                    matrices,
                    preparation.orders,
                    tuple(int(value) for value in exponents),
                    budget,
                )
                point = tuple(
                    sum(
                        (action[axis][column] * original[column] for column in range(3)),
                        action[axis][3],
                    )
                    for axis in range(3)
                )
                budget.charge(24, point)
            expected = np.asarray(
                tuple(float(value) for value in point), dtype=np.float64
            )
            if rounded is not None and not np.array_equal(
                expected.view(np.uint64), rounded[vertex].view(np.uint64)
            ):
                raise ValueError(
                    "Linear-action carrier is not the RNE of its original source construction."
                )
            error = max(
                error,
                *(
                    abs(value - Fraction(float(rne)))
                    for value, rne in zip(point, expected, strict=True)
                ),
            )
            result.append(point)
            conditions.append(
                max(parent.condition_upper_bounds[int(index)] for index in indices)
            )
        return PreparedExactPowerGeometry(
            tuple(result),
            self.vertex_coefficients,
            self.source_id,
            budget.work,
            budget.bits,
            error,
            tuple(conditions),
        )


@final
class ExactPowerCellGeometryRestrictionSource(StrictModule, NonTrainableState):
    """Exact authored-plane restriction or vertex-bank compaction of a source.

    Each vertex either retains one parent source vertex or intersects one
    parent source edge with one original binary64 plane. No rounded coordinate
    bank defines these intersections; all parent definitions remain dynamic.
    """

    parent: (
        ExactPowerCellGeometrySource
        | ExactPowerCellGeometryRestrictionSource
        | ExactPowerCellGeometryLinearActionSource
    )
    planes: Array
    vertex_parents: Array
    vertex_plane_ids: Array
    maximum_work: int = eqx.field(static=True)
    maximum_bits: int = eqx.field(static=True)
    maximum_condition: float = eqx.field(static=True)

    def __init__(
        self,
        parent: ExactPowerCellGeometrySource
        | ExactPowerCellGeometryRestrictionSource
        | ExactPowerCellGeometryLinearActionSource,
        planes: ArrayLike,
        vertex_parents: ArrayLike,
        vertex_plane_ids: ArrayLike,
        /,
    ) -> None:
        if not isinstance(
            parent,
            (
                ExactPowerCellGeometrySource,
                ExactPowerCellGeometryRestrictionSource,
                ExactPowerCellGeometryLinearActionSource,
            ),
        ):
            raise TypeError(
                "Exact power restrictions require their owning parent construction."
            )
        p = np.asarray(planes, dtype=np.float64)
        raw_parents, raw_ids = np.asarray(vertex_parents), np.asarray(vertex_plane_ids)
        if raw_parents.dtype.kind not in "iu" or raw_ids.dtype.kind not in "iu":
            raise TypeError("Exact power restriction witnesses require integer arrays.")
        vertices, identifiers = (
            np.asarray(raw_parents, dtype=np.int64),
            np.asarray(raw_ids, dtype=np.int64),
        )
        if p.ndim != 2 or p.shape[1] != 4 or not np.all(np.isfinite(p)):
            raise ValueError(
                "Exact power restriction planes require finite raw normal/offset rows."
            )
        if np.any(np.all(p[:, :3] == 0, axis=1)):
            raise ValueError("Exact power restriction planes require nonzero normals.")
        if (
            vertices.ndim != 2
            or vertices.shape[1] != 2
            or identifiers.shape != (vertices.shape[0],)
        ):
            raise ValueError(
                "Exact power restriction witnesses require one parent pair and plane ID per vertex."
            )
        size = (
            parent.vertex_carriers.shape[0]
            if isinstance(parent, ExactPowerCellGeometrySource)
            else parent.vertex_parents.shape[0]
        )
        if (
            np.any(vertices[:, 0] < 0)
            or np.any(vertices[:, 0] >= size)
            or np.any(vertices[:, 1] < -1)
            or np.any(vertices[:, 1] >= size)
        ):
            raise ValueError("Exact power restrictions index undeclared parent vertices.")
        retained = vertices[:, 1] == -1
        if (
            np.any(identifiers[retained] != -1)
            or np.any(identifiers[~retained] < 0)
            or np.any(identifiers[~retained] >= p.shape[0])
            or np.any(vertices[~retained, 0] >= vertices[~retained, 1])
        ):
            raise ValueError(
                "Exact power restriction witnesses must be canonical retained vertices or sorted intersected edges."
            )
        self.parent = parent
        self.planes, self.vertex_parents, self.vertex_plane_ids = (
            jnp.asarray(p),
            jnp.asarray(vertices),
            jnp.asarray(identifiers),
        )
        self.maximum_work, self.maximum_bits, self.maximum_condition = (
            parent.maximum_work,
            parent.maximum_bits,
            parent.maximum_condition,
        )

    @property
    def source_id(self) -> str:
        return canonical_fingerprint(
            {
                "kind": "exact-power-cell-plane-restriction",
                "parent": self.parent.source_id,
                "witness_arrays": array_tree_fingerprint(
                    (self.planes, self.vertex_parents, self.vertex_plane_ids)
                ),
                "maximum_work": self.maximum_work,
                "maximum_bits": self.maximum_bits,
                "maximum_condition": self.maximum_condition,
            }
        )

    def prepare(
        self, rounded_vertices: ArrayLike | None = None, /
    ) -> PreparedExactPowerGeometry:
        from ..linalg._small_batched import prepare_exact_small_linear_actions

        if any(
            isinstance(value, core.Tracer) for value in jax.tree_util.tree_leaves(self)
        ):
            raise TypeError(
                "Exact power restriction preparation is host-only; no source derivative is claimed."
            )
        parent = self.parent.prepare()
        budget = _ExactPowerBudget(self.maximum_work, self.maximum_bits)
        # Parent preparation already charged an active owning coefficient ledger.
        budget.work, budget.bits = parent.operation_count, parent.maximum_integer_bits
        rounded = (
            None
            if rounded_vertices is None
            else np.asarray(rounded_vertices, dtype=np.float64)
        )
        if rounded is not None and (
            rounded.shape != (self.vertex_parents.shape[0], 3)
            or not np.all(np.isfinite(rounded))
        ):
            raise ValueError(
                "Exact power restriction RNE carrier must align with its vertex witnesses."
            )
        planes = tuple(
            tuple(Fraction(float(value)) for value in row)
            for row in np.asarray(self.planes)
        )
        result, barycentric, conditions = [], [], []
        error = Fraction(0)
        for vertex, (pair, plane_id) in enumerate(
            zip(
                np.asarray(self.vertex_parents),
                np.asarray(self.vertex_plane_ids),
                strict=True,
            )
        ):
            _charge_power_source_query()
            first = parent.vertices[pair[0]]
            if pair[1] < 0:
                point, coefficients, condition = (
                    first,
                    (Fraction(1),),
                    parent.condition_upper_bounds[pair[0]],
                )
            else:
                second, plane = parent.vertices[pair[1]], planes[plane_id]
                direction = tuple(b - a for a, b in zip(first, second, strict=True))
                divisor = sum(
                    (plane[axis] * direction[axis] for axis in range(3)), Fraction(0)
                )
                right = plane[3] - sum(
                    (plane[axis] * first[axis] for axis in range(3)), Fraction(0)
                )
                budget.charge(30, (divisor, right))
                solve = prepare_exact_small_linear_actions(((divisor,),), ((right,),))
                if solve.actions is None:
                    raise ValueError(
                        "Exact power plane-edge restriction has deficient intersection rank."
                    )
                parameter = solve.actions[0][0]
                if not 0 < parameter < 1:
                    raise ValueError(
                        "Exact power restriction witness must intersect the parent edge interior."
                    )
                point = tuple(
                    a + parameter * delta
                    for a, delta in zip(first, direction, strict=True)
                )
                if (
                    sum((plane[axis] * point[axis] for axis in range(3)), Fraction(0))
                    != plane[3]
                ):
                    raise ValueError(
                        "Exact power restriction fails its original plane residual."
                    )
                coefficients, condition = (
                    (1 - parameter, parameter),
                    max(
                        parent.condition_upper_bounds[pair[0]],
                        parent.condition_upper_bounds[pair[1]],
                    ),
                )
                budget.charge(30, (*point, *coefficients))
            expected = np.asarray(
                tuple(float(value) for value in point), dtype=np.float64
            )
            if rounded is not None and not np.array_equal(
                expected.view(np.uint64), rounded[vertex].view(np.uint64)
            ):
                raise ValueError(
                    "Restricted power-cell carrier is not the RNE of its exact source construction."
                )
            error = max(
                error,
                *(
                    abs(value - Fraction(float(value_)))
                    for value, value_ in zip(point, expected, strict=True)
                ),
            )
            result.append(point)
            barycentric.append(coefficients)
            conditions.append(condition)
        return PreparedExactPowerGeometry(
            tuple(result),
            tuple(barycentric),
            self.source_id,
            budget.work,
            budget.bits,
            error,
            tuple(conditions),
        )


__all__ = [
    "ExactPowerCellGeometrySource",
    "ExactPowerCellGeometryRestrictionSource",
    "ExactPowerCellGeometryLinearActionSource",
    "PreparedExactPowerGeometry",
]
