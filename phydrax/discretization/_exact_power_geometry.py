#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from typing import assert_never, final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, core
from jax.typing import ArrayLike

import phydrax.discretization._periodic_exact as periodic_exact_owner

# Constraint → scope → cell geometry → power source: resolve the class at invocation.
import phydrax.meshing._periodic_constraint as periodic_constraint_owner

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..geometry._periodic_power_preparation import PeriodicPowerPreparation
from ..geometry._rational import rational_vector
from ..linalg import _small_batched as imported_linalg__small_batched
from ..meshing._plc_source import PiecewiseLinearComplex
from ..typing import checked
from ._coordinate_budget import CoordinateEnclosureBudget


class _PowerArrays(StrictModule):
    """Validated host source arrays before device publication."""

    points: np.ndarray
    weights: np.ndarray
    carrier: np.ndarray
    tets: np.ndarray
    offsets: np.ndarray
    sites: np.ndarray
    simplices: np.ndarray


def _prepare_power_arrays(
    site_points: ArrayLike,
    site_weights: ArrayLike,
    carrier_points: ArrayLike,
    carrier_tets: ArrayLike,
    vertex_site_offsets: ArrayLike,
    vertex_sites: ArrayLike,
    vertex_carriers: ArrayLike,
    /,
) -> _PowerArrays:
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
        raise TypeError(
            "ExactPowerCellGeometrySource.indices must be integer arrays; got noninteger dtype"
        )
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
            f"ExactPowerCellGeometrySource.site_points must be nonempty 3-D sites with aligned weights; got {points.shape=} and {weights.shape=}"
        )
    if carrier.ndim != 2 or carrier.shape[1] != 3 or tets.ndim != 2 or tets.shape[1] != 4:
        raise ValueError(
            f"ExactPowerCellGeometrySource.carrier_points must be 3-D points with tetrahedra; got {carrier.shape=} and {tets.shape=}"
        )
    if not all(np.all(np.isfinite(value)) for value in (points, weights, carrier)):
        raise ValueError(
            "ExactPowerCellGeometrySource.coordinates must be finite; got nonfinite numerical leaves"
        )
    if (
        np.any(tets < 0)
        or np.any(tets >= carrier.shape[0])
        or any(len(set(row)) != 4 for row in tets.tolist())
    ):
        raise ValueError(
            "ExactPowerCellGeometrySource.carrier_tets must index four distinct carrier vertices; got invalid vertex indices"
        )
    if (
        simplices.ndim != 2
        or simplices.shape[1] != 4
        or offsets.shape != (simplices.shape[0] + 1,)
    ):
        raise ValueError(
            f"ExactPowerCellGeometrySource.vertex_carriers must align CSR offsets and padded simplex rows; got {simplices.shape=} and {offsets.shape=}"
        )
    if (
        sites.ndim != 1
        or offsets[0] != 0
        or offsets[-1] != sites.size
        or np.any(np.diff(offsets) < 1)
    ):
        raise ValueError(
            "ExactPowerCellGeometrySource.vertex_sites must have one nonempty CSR row per vertex; got invalid offsets or site rank"
        )
    return _PowerArrays(points, weights, carrier, tets, offsets, sites, simplices)


def _validate_power_witness_rows(arrays: _PowerArrays, /) -> None:
    for vertex in range(arrays.simplices.shape[0]):
        start, stop = arrays.offsets[vertex], arrays.offsets[vertex + 1]
        row = arrays.simplices[vertex : vertex + 1].reshape((4,))
        equal = arrays.sites[start:stop]
        valid = row[row >= 0]
        if (
            not valid.size
            or np.any(row < -1)
            or np.any(valid >= arrays.carrier.shape[0])
            or not np.array_equal(row >= 0, np.arange(4) < valid.size)
        ):
            raise ValueError(
                "ExactPowerCellGeometrySource.vertex_carriers must contain valid-prefix simplex indices; got invalid carrier row"
            )
        if np.any(np.diff(equal) <= 0) or np.any(np.diff(valid) <= 0):
            raise ValueError(
                "ExactPowerCellGeometrySource.indices must be sorted and unique; got nonincreasing source indices"
            )


type ExactPoint = tuple[Fraction, Fraction, Fraction]


@dataclass(frozen=True, slots=True)
class PreparedExactPowerGeometry:
    """Return rational source vertices with bounded work and rounding evidence."""

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
    """Construct radical-plane intersections with independently verified RNE carriers.

    Dynamic binary64 leaves define the source. CSR rows contain all equal-power
    sites; each carrier row names a simplex of the authored tetrahedral
    decomposition. Rational preparation is host-only and must be repeated after
    changing source leaves.
    """

    site_points: Array
    site_weights: Array
    carrier_points: Array
    carrier_tets: Array
    vertex_site_offsets: Array
    vertex_sites: Array
    vertex_carriers: Array
    periodic_preparation: PeriodicPowerPreparation | None
    periodic_source_binding: str | None = eqx.field(static=True)
    domain_source: PiecewiseLinearComplex | None
    periodic_constraints: tuple[periodic_constraint_owner.PeriodicConstraint, ...]
    authority_binding: str | None = eqx.field(static=True)
    maximum_work: int = eqx.field(static=True)
    maximum_bits: int = eqx.field(static=True)
    maximum_condition: float = eqx.field(static=True)

    @checked
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
        periodic_preparation: PeriodicPowerPreparation | None = None,
        domain_source: PiecewiseLinearComplex | None = None,
        periodic_constraints: tuple[
            periodic_constraint_owner.PeriodicConstraint, ...
        ] = (),
    ) -> None:
        arrays = _prepare_power_arrays(
            site_points,
            site_weights,
            carrier_points,
            carrier_tets,
            vertex_site_offsets,
            vertex_sites,
            vertex_carriers,
        )
        points, weights, carrier = arrays.points, arrays.weights, arrays.carrier
        tets, offsets, sites, simplices = (
            arrays.tets,
            arrays.offsets,
            arrays.sites,
            arrays.simplices,
        )
        if periodic_preparation is not None:
            if (
                array_tree_fingerprint(points)
                != array_tree_fingerprint(periodic_preparation.points)
                or array_tree_fingerprint(weights)
                != array_tree_fingerprint(periodic_preparation.weights)
                or canonical_fingerprint(array_tree_fingerprint(carrier))
                != periodic_preparation.carrier_id
            ):
                raise ValueError(
                    "ExactPowerCellGeometrySource.periodic_preparation must match source and carrier bytes; got different source arrays"
                )
        site_count = (
            points.shape[0]
            if periodic_preparation is None
            else len(periodic_preparation.image_sites)
        )
        if np.any(sites < 0) or np.any(sites >= site_count):
            raise ValueError(
                "ExactPowerCellGeometrySource.vertex_sites must index declared sites or images; got undeclared sites or images"
            )
        _validate_power_witness_rows(arrays)
        for name, value in (
            ("maximum_work", maximum_work),
            ("maximum_bits", maximum_bits),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(
                    f"ExactPowerCellGeometrySource.{name} must be a positive integer; got {value!r}"
                )
        if not math.isfinite(maximum_condition) or maximum_condition < 1:
            raise ValueError(
                f"ExactPowerCellGeometrySource.maximum_condition must be finite and at least one; got {maximum_condition!r}"
            )

        controls = periodic_constraints
        if not all(
            isinstance(control, periodic_constraint_owner.PeriodicConstraint)
            for control in controls
        ):
            raise TypeError(
                "ExactPowerCellGeometrySource.periodic_constraints must contain PeriodicConstraint owners; got an unsupported control"
            )
        if controls and (domain_source is None or periodic_preparation is None):
            raise ValueError(
                "ExactPowerCellGeometrySource.periodic_constraints must retain domain_source and periodic_preparation; got a missing source"
            )
        self.site_points, self.site_weights, self.carrier_points = jax.device_put(
            (points, weights, carrier)
        )
        (
            self.carrier_tets,
            self.vertex_site_offsets,
            self.vertex_sites,
            self.vertex_carriers,
        ) = jax.device_put((tets, offsets, sites, simplices))
        self.maximum_work, self.maximum_bits = maximum_work, maximum_bits
        self.maximum_condition = float(maximum_condition)
        self.periodic_preparation = periodic_preparation
        self.periodic_source_binding = (
            None
            if periodic_preparation is None
            else power_preparation_binding(periodic_preparation)
        )
        self.domain_source, self.periodic_constraints = domain_source, controls
        self.authority_binding = (
            None
            if domain_source is None
            else power_domain_binding(domain_source, controls)
        )

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[ExactPowerCellGeometrySource],
        value: ExactPowerCellGeometrySource,
        /,
    ) -> ExactPowerCellGeometrySource:
        """Rebuild source arrays and authenticate retained source bindings."""
        renewed = cls(
            value.site_points,
            value.site_weights,
            value.carrier_points,
            value.carrier_tets,
            value.vertex_site_offsets,
            value.vertex_sites,
            value.vertex_carriers,
            maximum_work=value.maximum_work,
            maximum_bits=value.maximum_bits,
            maximum_condition=value.maximum_condition,
            periodic_preparation=value.periodic_preparation,
            domain_source=value.domain_source,
            periodic_constraints=value.periodic_constraints,
        )
        if renewed.periodic_source_binding != value.periodic_source_binding:
            raise ValueError(
                "ExactPowerCellGeometrySource.periodic_source_binding must match restored preparation; "
                f"got {value.periodic_source_binding!r}"
            )
        if renewed.authority_binding != value.authority_binding:
            raise ValueError(
                "ExactPowerCellGeometrySource.authority_binding must match restored domain and controls; "
                f"got {value.authority_binding!r}"
            )
        return renewed

    @property
    def source_id(self) -> str:
        preparation = self.periodic_preparation
        domain, constraints = (self.domain_source, self.periodic_constraints)
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
                    else {"periodic_preparation": power_preparation_binding(preparation)}
                ),
                **(
                    {}
                    if domain is None
                    else {"domain_authority": power_domain_binding(domain, constraints)}
                ),
            }
        )

    def preparation_query_upper(self) -> int:
        """Bound the preparation's site-image and vertex queries."""
        preparation = self.periodic_preparation
        images = 0 if preparation is None else preparation.image_sites.shape[0]
        return images + self.vertex_carriers.shape[0]

    @checked
    def prepare(
        self,
        rounded_vertices: ArrayLike | None = None,
        /,
        *,
        budget: CoordinateEnclosureBudget,
    ) -> PreparedExactPowerGeometry:
        """Reuse earned source intersections within the caller's request ledger."""
        if any(
            isinstance(value, core.Tracer) for value in jax.tree_util.tree_leaves(self)
        ):
            raise TypeError(
                "ExactPowerCellGeometrySource.prepare must receive host source leaves; got a JAX tracer"
            )
        domain, constraints = (self.domain_source, self.periodic_constraints)
        if (
            domain is not None
            and power_domain_binding(domain, constraints) != self.authority_binding
        ):
            raise ValueError(
                "ExactPowerCellGeometrySource.authority_binding must match domain and control content; got a changed binding"
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
                f"ExactPowerCellGeometrySource.rounded_vertices must align with vertex rows and be finite; got {rounded.shape}"
            )
        preparation = self.periodic_preparation
        if preparation is not None:
            if power_preparation_binding(preparation) != self.periodic_source_binding:
                raise ValueError(
                    "ExactPowerCellGeometrySource.periodic_source_binding must match retained preparation; got a changed binding"
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
                    "ExactPowerCellGeometrySource.periodic_preparation must match current source leaves; got different source bytes"
                )
        identity = self.source_id
        cached = _reuse_power_witness(
            identity, rounded, "ExactPowerCellGeometrySource", budget=budget
        )
        if cached is not None:
            return cached
        exact_work = _ExactPowerBudget(
            self.maximum_work, self.maximum_bits, budget=budget
        )
        points = tuple(rational_vector(row) for row in np.asarray(self.site_points))
        weights = rational_vector(np.asarray(self.site_weights))
        carrier = tuple(rational_vector(row) for row in np.asarray(self.carrier_points))
        exact_work.charge(
            0, (*weights, *(value for row in (*points, *carrier) for value in row))
        )
        if preparation is not None:
            images = _prepare_power_images(
                preparation, points, weights, exact_work, budget=budget
            )
            points, weights = images.points, images.weights

        # Full carrier rank is exactly a nonzero edge determinant; no solution
        # is consumed, so the exact cofactor substrate charges nine differences
        # and its fourteen operations instead of an elimination.
        tet_rows = np.asarray(self.carrier_tets, dtype=np.int64)
        for row in tet_rows.tolist():
            determinant = imported_linalg__small_batched.exact_cofactor_determinant_3(
                *(
                    tuple(
                        carrier[row[column + 1]][axis] - carrier[row[0]][axis]
                        for axis in range(3)
                    )
                    for column in range(3)
                )
            )
            exact_work.charge(9 + 14, (determinant,))
            if determinant == 0:
                raise ValueError(
                    "ExactPowerCellGeometrySource.carrier_tets must have full source rank; got deficient rank"
                )
        # Index carrier tetrahedra by corner once; a vertex simplex is authored
        # exactly when a tetrahedron incident to its first corner contains it.
        corners = tet_rows.reshape(-1)
        exact_work.charge(corners.size)
        budget.reserve(0, 8 * (3 * corners.size + len(carrier) + 1))
        incident_order = np.argsort(corners, kind="stable")
        incident_tets = incident_order // 4
        incident_offsets = np.searchsorted(
            corners[incident_order], np.arange(len(carrier) + 1)
        )

        offsets, sites = (
            np.asarray(self.vertex_site_offsets),
            np.asarray(self.vertex_sites),
        )
        vertices, barycentric = [], []
        conditions = []
        rounding_error = Fraction(0)
        for vertex, simplex in enumerate(np.asarray(self.vertex_carriers)):
            budget.reserve_geometry_queries(1)
            equal = tuple(sites[offsets[vertex] : offsets[vertex + 1]].tolist())
            indices = tuple(simplex[simplex >= 0].tolist())
            first = indices[0] if indices and 0 <= indices[0] < len(carrier) else None
            incident = (
                incident_tets[:0]
                if first is None
                else incident_tets[incident_offsets[first] : incident_offsets[first + 1]]
            )
            exact_work.charge(len(incident) * len(indices))
            if not any(set(indices) <= set(tet_rows[tet].tolist()) for tet in incident):
                raise ValueError(
                    "ExactPowerCellGeometrySource.vertex_carriers must name an authored tetrahedral simplex; got a foreign simplex"
                )
            coordinates, coefficients, condition = _prepare_vertex(
                points,
                weights,
                carrier,
                equal,
                indices,
                exact_work,
                budget=budget,
            )
            if condition > Fraction(self.maximum_condition):
                raise ValueError(
                    f"ExactPowerCellGeometrySource.maximum_condition must bound vertex construction; got condition {condition}"
                )
            expected = np.asarray(
                tuple(float(value) for value in coordinates), dtype=np.float64
            )
            if rounded is not None and not np.array_equal(
                expected.view(np.uint64), rounded[vertex].view(np.uint64)
            ):
                raise ValueError(
                    "ExactPowerCellGeometrySource.rounded_vertices must equal RNE source intersections; got different coordinate bits"
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
        prepared = PreparedExactPowerGeometry(
            tuple(vertices),
            tuple(barycentric),
            identity,
            exact_work.work,
            exact_work.bits,
            rounding_error,
            tuple(conditions),
        )
        return _retain_power_witness(prepared, budget=budget)


def _reuse_power_witness(
    identity: str,
    rounded: np.ndarray | None,
    owner: str,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> PreparedExactPowerGeometry | None:
    """Return this request's earned construction for the full source identity.

    The identity covers every source array, parent identity and control, so the
    retained proof is the same construction; a caller's rounded carrier is
    still checked bitwise against it on every reuse.
    """
    cached = budget.power_witness_cache.get(identity)
    if cached is None:
        return None
    budget.reserve(0 if rounded is None else 6 * len(cached.vertices))
    if rounded is not None and not np.array_equal(
        cached.rounded_vertices.view(np.uint64), rounded.view(np.uint64)
    ):
        raise ValueError(
            f"{owner}.rounded_vertices must equal RNE source construction; got different coordinate bits"
        )
    return cached


def _power_witness_operand_basis(
    prepared: PreparedExactPowerGeometry, /,
) -> tuple[object, ...]:
    """Expose only the exact construction's immutable numerical inputs."""
    return (
        prepared.source_id, prepared.vertices, prepared.barycentric,
        prepared.operation_count, prepared.maximum_integer_bits,
        prepared.maximum_rounding_error, prepared.condition_upper_bounds,
    )


def _retain_power_witness(
    prepared: PreparedExactPowerGeometry, /, *, budget: CoordinateEnclosureBudget
) -> PreparedExactPowerGeometry:
    if budget.capture_coordinate_operands:
        # Earn the numeric tuple's profile before the enclosing result retains
        # its children. No membership, coverage, or certification is captured.
        budget.retain_basis(
            _power_witness_operand_basis(prepared), operand=True,
            _kernel_operands=True,
        )
    budget.retain_basis((prepared.source_id, prepared))
    budget.power_witness_cache[prepared.source_id] = prepared
    return prepared


class _ExactPowerBudget:
    def __init__(
        self,
        maximum_work: int,
        maximum_bits: int,
        /,
        *,
        budget: CoordinateEnclosureBudget,
    ) -> None:
        self.maximum_work, self.maximum_bits = maximum_work, maximum_bits
        self.work, self.bits = 0, 0
        self.enclosure = budget

    def admit(self, work: int, /) -> None:
        """Admit an owner-charged upper bound without counting it as completed."""
        if self.work + work > self.maximum_work:
            raise ValueError(
                f"ExactPowerCellGeometrySource.maximum_work must admit construction work budget; got {self.work + work} > {self.maximum_work}"
            )
        self.enclosure.admit_work_bound(work)

    def completed(self, work: int, values: tuple[Fraction, ...] = (), /) -> None:
        """Count work its owning substrate already charged to the enclosure."""
        self._observe(work, values)

    def charge(self, work: int, values: tuple[Fraction, ...] = (), /) -> None:
        self._observe(work, values)
        self.enclosure.reserve(work)

    def _observe(self, work: int, values: tuple[Fraction, ...], /) -> None:
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
            raise ValueError(
                f"ExactPowerCellGeometrySource.maximum_work must admit construction work budget; got {self.work} > {self.maximum_work}"
            )
        if self.bits > self.maximum_bits:
            raise ValueError(
                f"ExactPowerCellGeometrySource.maximum_bits must admit construction integer-bit width; got {self.bits} > {self.maximum_bits}"
            )


def _prepare_vertex(
    points: tuple[tuple[Fraction, ...], ...],
    weights: tuple[Fraction, ...],
    carrier: tuple[tuple[Fraction, ...], ...],
    equal: tuple[int, ...],
    indices: tuple[int, ...],
    exact_work: _ExactPowerBudget,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[ExactPoint, tuple[Fraction, ...], Fraction]:

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
    exact_work.charge(
        6 * len(planes) * size, tuple(value for row in equations for value in row)
    )
    coefficients = None
    condition = Fraction(0)
    for selected in combinations(range(len(planes)), size - 1):
        matrix = ((Fraction(1),) * size, *(equations[index] for index in selected))
        right = ((Fraction(1),), *((planes[index][1],) for index in selected))
        exact_work.admit(4 * size**3)
        completed = budget.work_units
        solved = imported_linalg__small_batched.prepare_exact_small_linear_actions(
            matrix, right, budget=budget
        )
        exact_work.completed(budget.work_units - completed)
        if solved.actions is not None:
            coefficients = tuple(row[0] for row in solved.actions)
            row_scale = math.prod(max(abs(value) for value in row) for row in matrix)
            condition = (
                Fraction(size * size * math.factorial(size - 1))
                * row_scale
                / abs(solved.determinant)
            )
            exact_work.charge(0, coefficients)
            break
    if coefficients is None:
        raise ValueError(
            "ExactPowerCellGeometrySource.vertex_carriers must have full intersection rank; got deficient rank"
        )
    if min(coefficients) < 0:
        raise ValueError(
            "ExactPowerCellGeometrySource.vertex_carriers must contain the intersection; got an exterior simplex point"
        )
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
    exact_work.charge(6 * size, coordinates)
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
    exact_work.charge(12 * len(points), powers)
    minimum = powers[anchor]
    actual_equal = tuple(index for index, value in enumerate(powers) if value == minimum)
    if actual_equal != equal or any(value < minimum for value in powers):
        raise ValueError(
            "ExactPowerCellGeometrySource.vertex_sites must match equal-site power relations; got contradictory indices"
        )
    return (coordinates[0], coordinates[1], coordinates[2]), coefficients, condition


@checked
def power_domain_binding(
    domain: PiecewiseLinearComplex,
    constraints: tuple[periodic_constraint_owner.PeriodicConstraint, ...],
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


@checked
def power_preparation_binding(
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


class _PowerSites(StrictModule):
    """Rational image sites paired with their inherited weights."""

    points: tuple[tuple[Fraction, ...], ...] = eqx.field(static=True)
    weights: tuple[Fraction, ...] = eqx.field(static=True)


def _prepare_power_images(
    preparation: PeriodicPowerPreparation,
    points: tuple[tuple[Fraction, ...], ...],
    weights: tuple[Fraction, ...],
    exact_work: _ExactPowerBudget,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> _PowerSites:
    actions = _prepare_power_actions(
        preparation, preparation.image_exponents, exact_work, budget=budget
    )
    images = []
    image_weights = []
    for owner, action in zip(preparation.image_sites, actions, strict=True):
        budget.reserve_geometry_queries(1)
        point = points[int(owner)]
        image = tuple(
            sum(
                (action[axis][column] * point[column] for column in range(3)),
                action[axis][3],
            )
            for axis in range(3)
        )
        exact_work.charge(24, image)
        images.append(image)
        image_weights.append(weights[int(owner)])
    return _PowerSites(tuple(images), tuple(image_weights))


def _prepare_power_actions(
    preparation: PeriodicPowerPreparation,
    exponents: ArrayLike,
    exact_work: _ExactPowerBudget,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[periodic_exact_owner.ExactIsometry, ...]:
    """Evaluate each distinct retained group word once, charging refusal prefixes."""
    lattice = periodic_exact_owner.ExactPeriodicActionLattice(
        tuple(
            tuple(rational_vector(row) for row in matrix)
            for matrix in preparation.generators
        ),
        preparation.orders,
    )
    start = budget.work_units
    try:
        with (
            budget.activate(),
            budget.bound_stage(
                max(0, exact_work.maximum_work - exact_work.work),
                budget.maximum_memory_bytes,
                starting_work_units=start,
            ),
            budget.temporary_scope(),
        ):
            budget.reserve(0, 48 * (128 + 2 * ((exact_work.maximum_bits + 7) // 8)))
            actions = lattice.prepare(
                np.asarray(exponents, dtype=np.int64).tolist(), budget=budget
            )
    finally:
        exact_work.work += budget.work_units - start
    exact_work.charge(
        0,
        tuple(
            value
            for action in lattice.elements.values()
            for row in action
            for value in row
        ),
    )
    return actions


class _RestrictedPowerVertex(StrictModule):
    """A parent-edge intersection and its inherited conditioning evidence."""

    point: tuple[Fraction, ...] = eqx.field(static=True)
    coefficients: tuple[Fraction, ...] = eqx.field(static=True)
    condition: Fraction = eqx.field(static=True)


def _restrict_power_edge(
    parent: PreparedExactPowerGeometry,
    pair: np.ndarray,
    plane: tuple[Fraction, ...],
    exact_work: _ExactPowerBudget,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> _RestrictedPowerVertex:
    first, second = parent.vertices[pair[0]], parent.vertices[pair[1]]
    direction = tuple(b - a for a, b in zip(first, second, strict=True))
    divisor = sum((plane[axis] * direction[axis] for axis in range(3)), Fraction(0))
    right = plane[3] - sum((plane[axis] * first[axis] for axis in range(3)), Fraction(0))
    exact_work.charge(30, (divisor, right))
    solve = imported_linalg__small_batched.prepare_exact_small_linear_actions(
        ((divisor,),), ((right,),), budget=budget
    )
    if solve.actions is None:
        raise ValueError(
            "ExactPowerCellGeometryRestrictionSource.planes must have full plane-edge intersection rank; got deficient rank"
        )
    parameter = solve.actions[0][0]
    if not 0 < parameter < 1:
        raise ValueError(
            f"ExactPowerCellGeometryRestrictionSource.vertex_parents must intersect the parent edge interior; got parameter {parameter}"
        )
    point = tuple(
        a + parameter * delta for a, delta in zip(first, direction, strict=True)
    )
    if sum((plane[axis] * point[axis] for axis in range(3)), Fraction(0)) != plane[3]:
        raise ValueError(
            "ExactPowerCellGeometryRestrictionSource.planes must contain the constructed point; got nonzero rational plane residual"
        )
    coefficients, condition = (
        (1 - parameter, parameter),
        max(
            parent.condition_upper_bounds[pair[0]],
            parent.condition_upper_bounds[pair[1]],
        ),
    )
    exact_work.charge(30, (*point, *coefficients))
    return _RestrictedPowerVertex(point, coefficients, condition)


@final
class ExactPowerCellGeometryLinearActionSource(StrictModule, NonTrainableState):
    """Apply convex parent coefficients followed by an optional retained group action."""

    parent: (
        ExactPowerCellGeometrySource
        | ExactPowerCellGeometryRestrictionSource
        | ExactPowerCellGeometryLinearActionSource
    )
    periodic_preparation: PeriodicPowerPreparation | None
    vertex_parents: Array
    vertex_actions: Array
    vertex_coefficients: tuple[tuple[Fraction, ...], ...] = eqx.field(static=True)
    parent_source_id: str = eqx.field(static=True)
    construction_id: str = eqx.field(static=True)
    maximum_work: int = eqx.field(static=True)
    maximum_bits: int = eqx.field(static=True)
    maximum_condition: float = eqx.field(static=True)

    @checked
    def __init__(
        self,
        parent: ExactPowerCellGeometrySource
        | ExactPowerCellGeometryRestrictionSource
        | ExactPowerCellGeometryLinearActionSource,
        vertex_parents: ArrayLike,
        vertex_coefficients: Sequence[Sequence[Fraction]],
        vertex_actions: ArrayLike,
        /,
        *,
        periodic_preparation: PeriodicPowerPreparation | None = None,
    ) -> None:

        root = parent
        while True:
            match root:
                case ExactPowerCellGeometrySource():
                    break
                case (
                    ExactPowerCellGeometryRestrictionSource()
                    | ExactPowerCellGeometryLinearActionSource()
                ):
                    root = root.parent
                case invalid:
                    assert_never(invalid)
        root_preparation = root.periodic_preparation
        if periodic_preparation is None:
            if root_preparation is not None:
                raise ValueError(
                    "ExactPowerCellGeometryLinearActionSource.periodic_preparation must retain the parent group; got None"
                )
        elif root_preparation is None or power_preparation_binding(
            root_preparation
        ) != power_preparation_binding(periodic_preparation):
            raise ValueError(
                "ExactPowerCellGeometryLinearActionSource.periodic_preparation must match the parent binding; got a different preparation"
            )
        indices, actions = np.asarray(vertex_parents), np.asarray(vertex_actions)
        if indices.dtype.kind not in "iu" or actions.dtype.kind not in "iu":
            raise TypeError(
                "ExactPowerCellGeometryLinearActionSource.indices must be integers; got noninteger parent or action dtype"
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
                f"ExactPowerCellGeometryLinearActionSource.vertex_actions must have aligned parent and generator rows; got {indices.shape=} and {actions.shape=}"
            )
        size = _power_vertex_count(parent)
        if np.any(indices < -1) or np.any(indices >= size):
            raise ValueError(
                "ExactPowerCellGeometryLinearActionSource.vertex_parents must index declared parent vertices; got an undeclared parent"
            )
        coefficients = tuple(tuple(row) for row in vertex_coefficients)
        if len(coefficients) != len(indices):
            raise ValueError(
                "ExactPowerCellGeometryLinearActionSource.vertex_coefficients must align with parent rows; got a different row count"
            )
        for row, values in zip(indices, coefficients, strict=True):
            if len(values) != len(row) or any(
                not isinstance(value, Fraction) for value in values
            ):
                raise TypeError(
                    "ExactPowerCellGeometryLinearActionSource.vertex_coefficients must have one Fraction per parent slot; got a different row length or object kind"
                )
            active = row >= 0
            if not np.any(active) or not np.array_equal(
                active, np.arange(len(row)) < np.count_nonzero(active)
            ):
                raise ValueError(
                    "ExactPowerCellGeometryLinearActionSource.vertex_parents must contain a nonempty valid prefix and -1 padding; got invalid padding"
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
                    "ExactPowerCellGeometryLinearActionSource.vertex_coefficients must provide distinct parents and convex unit coefficients; got invalid convex coefficients"
                )
        for axis, order in enumerate(
            () if periodic_preparation is None else periodic_preparation.orders
        ):
            if order and (
                np.any(actions[:, axis] < 0) or np.any(actions[:, axis] >= order)
            ):
                raise ValueError(
                    "ExactPowerCellGeometryLinearActionSource.vertex_actions must use reduced finite group exponents; got an exponent outside its order"
                )
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

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[ExactPowerCellGeometryLinearActionSource],
        value: ExactPowerCellGeometryLinearActionSource,
        /,
    ) -> ExactPowerCellGeometryLinearActionSource:
        """Rebuild a parent action and authenticate its retained construction."""
        renewed = cls(
            value.parent,
            value.vertex_parents,
            value.vertex_coefficients,
            value.vertex_actions,
            periodic_preparation=value.periodic_preparation,
        )
        if renewed.parent_source_id != value.parent_source_id:
            raise ValueError(
                "ExactPowerCellGeometryLinearActionSource.parent_source_id must match restored parent content; "
                f"got {value.parent_source_id!r}"
            )
        if renewed.construction_id != value.construction_id:
            raise ValueError(
                "ExactPowerCellGeometryLinearActionSource.construction_id must match restored action content; "
                f"got {value.construction_id!r}"
            )
        if (
            value.maximum_work,
            value.maximum_bits,
            value.maximum_condition,
        ) != (
            renewed.maximum_work,
            renewed.maximum_bits,
            renewed.maximum_condition,
        ):
            raise ValueError(
                "ExactPowerCellGeometryLinearActionSource.limits must match restored parent limits; "
                "got different construction limits"
            )
        return renewed

    @property
    def source_id(self) -> str:
        parent, preparation = (self.parent, self.periodic_preparation)
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
                    else power_preparation_binding(preparation)
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

    def preparation_query_upper(self) -> int:
        """Bound one parent preparation and every output-vertex query."""
        parent, _ = (self.parent, self.periodic_preparation)
        return parent.preparation_query_upper() + self.vertex_parents.shape[0]

    @checked
    def prepare(
        self,
        rounded_vertices: ArrayLike | None = None,
        /,
        *,
        budget: CoordinateEnclosureBudget,
    ) -> PreparedExactPowerGeometry:
        if any(
            isinstance(value, core.Tracer) for value in jax.tree_util.tree_leaves(self)
        ):
            raise TypeError(
                "ExactPowerCellGeometryLinearActionSource.prepare must receive host source leaves; got a JAX tracer"
            )
        owner, preparation = (self.parent, self.periodic_preparation)
        if owner.source_id != self.parent_source_id:
            raise ValueError(
                "ExactPowerCellGeometryLinearActionSource.parent_source_id must retain immutable source content; got a changed parent binding"
            )
        if self.source_id != self.construction_id:
            raise ValueError(
                "ExactPowerCellGeometryLinearActionSource.construction_id must match its source construction; got a changed binding"
            )
        parent = owner.prepare(budget=budget)
        exact_work = _ExactPowerBudget(
            self.maximum_work, self.maximum_bits, budget=budget
        )
        exact_work.work, exact_work.bits = (
            parent.operation_count,
            parent.maximum_integer_bits,
        )
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
                f"ExactPowerCellGeometryLinearActionSource.rounded_vertices must align with source rows and be finite; got {rounded.shape}"
            )
        cached = _reuse_power_witness(
            self.construction_id,
            rounded,
            "ExactPowerCellGeometryLinearActionSource",
            budget=budget,
        )
        if cached is not None:
            return cached
        actions = (
            None
            if preparation is None
            else _prepare_power_actions(
                preparation, self.vertex_actions, exact_work, budget=budget
            )
        )
        result, conditions = [], []
        error = Fraction(0)
        for vertex, (indices, coefficients) in enumerate(
            zip(np.asarray(self.vertex_parents), self.vertex_coefficients, strict=True)
        ):
            budget.reserve_geometry_queries(1)
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
            exact_work.charge(6 * len(indices), (*original, *coefficients))
            if actions is None:
                point = original
            else:
                action = actions[vertex]
                point = tuple(
                    sum(
                        (action[axis][column] * original[column] for column in range(3)),
                        action[axis][3],
                    )
                    for axis in range(3)
                )
                exact_work.charge(24, point)
            expected = np.asarray(
                tuple(float(value) for value in point), dtype=np.float64
            )
            if rounded is not None and not np.array_equal(
                expected.view(np.uint64), rounded[vertex].view(np.uint64)
            ):
                raise ValueError(
                    "ExactPowerCellGeometryLinearActionSource.rounded_vertices must equal RNE source construction; got different coordinate bits"
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
        prepared = PreparedExactPowerGeometry(
            tuple(result),
            self.vertex_coefficients,
            self.construction_id,
            exact_work.work,
            exact_work.bits,
            error,
            tuple(conditions),
        )
        return _retain_power_witness(prepared, budget=budget)


@final
class ExactPowerCellGeometryRestrictionSource(StrictModule, NonTrainableState):
    """Restrict source edges to authored planes or compact the vertex bank.

    Each vertex retains one parent vertex or intersects a parent edge with a
    binary64 plane. Parent definitions remain dynamic.
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

    @checked
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
        p = np.asarray(planes, dtype=np.float64)
        raw_parents, raw_ids = np.asarray(vertex_parents), np.asarray(vertex_plane_ids)
        if raw_parents.dtype.kind not in "iu" or raw_ids.dtype.kind not in "iu":
            raise TypeError(
                "ExactPowerCellGeometryRestrictionSource.indices must be integer arrays; got noninteger dtype"
            )
        vertices, identifiers = (
            np.asarray(raw_parents, dtype=np.int64),
            np.asarray(raw_ids, dtype=np.int64),
        )
        if p.ndim != 2 or p.shape[1] != 4 or not np.all(np.isfinite(p)):
            raise ValueError(
                f"ExactPowerCellGeometryRestrictionSource.planes must contain finite normal/offset rows; got {p.shape}"
            )
        if np.any(np.all(p[:, :3] == 0, axis=1)):
            raise ValueError(
                "ExactPowerCellGeometryRestrictionSource.planes must have nonzero normals; got a zero normal"
            )
        if (
            vertices.ndim != 2
            or vertices.shape[1] != 2
            or identifiers.shape != (vertices.shape[0],)
        ):
            raise ValueError(
                f"ExactPowerCellGeometryRestrictionSource.vertex_parents must have one pair and plane ID per vertex; got {vertices.shape=} and {identifiers.shape=}"
            )
        size = _power_vertex_count(parent)
        if (
            np.any(vertices[:, 0] < 0)
            or np.any(vertices[:, 0] >= size)
            or np.any(vertices[:, 1] < -1)
            or np.any(vertices[:, 1] >= size)
        ):
            raise ValueError(
                "ExactPowerCellGeometryRestrictionSource.vertex_parents must index declared parent vertices; got undeclared indices"
            )
        retained = vertices[:, 1] == -1
        if (
            np.any(identifiers[retained] != -1)
            or np.any(identifiers[~retained] < 0)
            or np.any(identifiers[~retained] >= p.shape[0])
            or np.any(vertices[~retained, 0] >= vertices[~retained, 1])
        ):
            raise ValueError(
                "ExactPowerCellGeometryRestrictionSource.vertex_parents must name retained vertices or sorted intersected edges; got invalid indices"
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

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[ExactPowerCellGeometryRestrictionSource],
        value: ExactPowerCellGeometryRestrictionSource,
        /,
    ) -> ExactPowerCellGeometryRestrictionSource:
        """Rebuild a restriction through its parent-edge and plane admission."""
        renewed = cls(
            value.parent, value.planes, value.vertex_parents, value.vertex_plane_ids
        )
        if (
            value.maximum_work,
            value.maximum_bits,
            value.maximum_condition,
        ) != (
            renewed.maximum_work,
            renewed.maximum_bits,
            renewed.maximum_condition,
        ):
            raise ValueError(
                "ExactPowerCellGeometryRestrictionSource.limits must match restored parent limits; "
                "got different construction limits"
            )
        return renewed

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

    def preparation_query_upper(self) -> int:
        """Bound one parent preparation and every retained/intersected vertex query."""
        return self.parent.preparation_query_upper() + self.vertex_parents.shape[0]

    @checked
    def prepare(
        self,
        rounded_vertices: ArrayLike | None = None,
        /,
        *,
        budget: CoordinateEnclosureBudget,
    ) -> PreparedExactPowerGeometry:

        if any(
            isinstance(value, core.Tracer) for value in jax.tree_util.tree_leaves(self)
        ):
            raise TypeError(
                "ExactPowerCellGeometryRestrictionSource.prepare must receive host source leaves; got a JAX tracer"
            )
        parent = self.parent.prepare(budget=budget)
        exact_work = _ExactPowerBudget(
            self.maximum_work, self.maximum_bits, budget=budget
        )
        # Parent preparation has already charged the request's coefficient ledger.
        exact_work.work, exact_work.bits = (
            parent.operation_count,
            parent.maximum_integer_bits,
        )
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
                f"ExactPowerCellGeometryRestrictionSource.rounded_vertices must align with vertex rows and be finite; got {rounded.shape}"
            )
        identity = self.source_id
        cached = _reuse_power_witness(
            identity, rounded, "ExactPowerCellGeometryRestrictionSource", budget=budget
        )
        if cached is not None:
            return cached
        planes = tuple(rational_vector(row) for row in np.asarray(self.planes))
        result, barycentric, conditions = [], [], []
        error = Fraction(0)
        for vertex, (pair, plane_id) in enumerate(
            zip(
                np.asarray(self.vertex_parents),
                np.asarray(self.vertex_plane_ids),
                strict=True,
            )
        ):
            budget.reserve_geometry_queries(1)
            first = parent.vertices[pair[0]]
            if pair[1] < 0:
                point, coefficients, condition = (
                    first,
                    (Fraction(1),),
                    parent.condition_upper_bounds[pair[0]],
                )
            else:
                restriction = _restrict_power_edge(
                    parent, pair, planes[plane_id], exact_work, budget=budget
                )
                point, coefficients, condition = (
                    restriction.point,
                    restriction.coefficients,
                    restriction.condition,
                )
            expected = np.asarray(
                tuple(float(value) for value in point), dtype=np.float64
            )
            if rounded is not None and not np.array_equal(
                expected.view(np.uint64), rounded[vertex].view(np.uint64)
            ):
                raise ValueError(
                    "ExactPowerCellGeometryRestrictionSource.rounded_vertices must equal RNE source construction; got different coordinate bits"
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
        prepared = PreparedExactPowerGeometry(
            tuple(result),
            tuple(barycentric),
            identity,
            exact_work.work,
            exact_work.bits,
            error,
            tuple(conditions),
        )
        return _retain_power_witness(prepared, budget=budget)


def _power_vertex_count(
    source: ExactPowerCellGeometrySource
    | ExactPowerCellGeometryRestrictionSource
    | ExactPowerCellGeometryLinearActionSource,
    /,
) -> int:
    match source:
        case ExactPowerCellGeometrySource():
            return source.vertex_carriers.shape[0]
        case (
            ExactPowerCellGeometryRestrictionSource()
            | ExactPowerCellGeometryLinearActionSource()
        ):
            return source.vertex_parents.shape[0]
        case invalid:
            assert_never(invalid)


__all__ = [
    "ExactPowerCellGeometrySource",
    "ExactPowerCellGeometryRestrictionSource",
    "ExactPowerCellGeometryLinearActionSource",
    "PreparedExactPowerGeometry",
]
