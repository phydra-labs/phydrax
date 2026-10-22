# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Durable rational interpolation actions on a parent coordinate bank.

Exact affine images use the same authenticated coefficient actions as
restricted cell geometry. Image/material charts and paired-cell material maps
therefore share one rational coordinate convention and correctly rounded
execution representatives, without replacing the authored coefficients.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from fractions import Fraction
from typing import final

import equinox as eqx
import jax
import numpy as np
from jax import Array, core
from jax.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import positive_integer
from ..typing import checked
from . import (
    _cell_geometry as geometry_owner,
    _coordinate_enclosure as algebra,
    _exact_algebraic_geometry as exact_algebraic_geometry_owner,
    _exact_plc_geometry as exact_plc_geometry_owner,
)
from ._coordinate_budget import CoordinateEnclosureBudget
from ._reference_cell_types import CellGeometryElement
from .fem._reference_types import FiniteElementSpec


type RationalAction = tuple[tuple[int, int, int], ...]
type RationalLayouts = tuple[tuple[str, str, int, str, str], ...]


def _nodal_element(element: CellGeometryElement) -> FiniteElementSpec:
    match element:
        case FiniteElementSpec():
            return element
        case _:
            raise TypeError(
                f"ExactRationalCellGeometrySource.layouts must contain nodal elements; got {type(element).__name__}"
            )


def _parent_binding(parent: geometry_owner.CellGeometrySpec) -> str:
    return canonical_fingerprint(
        {
            "layout": parent.geometry_layout_id,
            "arrays": array_tree_fingerprint(parent),
            "source": None
            if parent.exact_source is None
            else parent.exact_source.source_id,
        }
    )


def rational_source_layouts(
    elements: Mapping[str, CellGeometryElement],
    routes: Mapping[str, ArrayLike],
) -> RationalLayouts:
    """Bind nominal elements and ordered coefficient routes, not array shapes."""
    return tuple(
        (
            name,
            element.cell_kind,
            geometry_owner.require_scalar_coordinate_element(
                element, "exact rational layout"
            ).degree,
            element.element_id,
            canonical_fingerprint(
                array_tree_fingerprint(np.asarray(routes[name], dtype=np.int32))
            ),
        )
        for name, element in sorted(elements.items())
    )


def _require_rational_actions(
    actions: tuple[RationalAction, ...],
    source_count: int,
    maximum_work: int,
    maximum_bits: int,
    *,
    budget: CoordinateEnclosureBudget,
) -> None:
    exact = exact_plc_geometry_owner.ExactPlcBudget(
        maximum_work, maximum_bits, budget=budget
    )
    for row in actions:
        if not 1 <= len(row) <= 8:
            raise ValueError(
                f"ExactRationalCellGeometrySource.actions must use degree-one interpolation arity from one through eight; got {len(row)}"
            )
        previous = -1
        total = Fraction(0)
        for index, numerator, denominator in row:
            if any(type(value) is not int for value in (index, numerator, denominator)):
                raise TypeError(
                    f"ExactRationalCellGeometrySource.actions must contain integer triples; got {(index, numerator, denominator)!r}"
                )
            if not previous < index < source_count or denominator <= 0 or numerator == 0:
                raise ValueError(
                    f"ExactRationalCellGeometrySource.actions must use sorted unique source indices and nonzero reduced ratios; got {(index, numerator, denominator)!r} after {previous}"
                )
            value = Fraction(numerator, denominator)
            if (value.numerator, value.denominator) != (numerator, denominator):
                raise ValueError(
                    f"ExactRationalCellGeometrySource.actions must contain reduced ratios; got {(numerator, denominator)!r}"
                )
            previous = index
            total += value
            exact.charge(2, (value, total))
        if total != 1:
            raise ValueError(
                f"ExactRationalCellGeometrySource.actions must sum to one; got {total}"
            )


def _representative_row(
    point: tuple[Fraction | exact_algebraic_geometry_owner.RadicalElement, ...],
    retained_index: int | None,
    parent: np.ndarray,
    budget: CoordinateEnclosureBudget,
) -> tuple[float, ...]:
    if retained_index is not None:
        return tuple(
            float(parent[retained_index, axis]) for axis in range(parent.shape[1])
        )
    return tuple(
        value.rounded(budget=budget)[0]
        if isinstance(value, exact_algebraic_geometry_owner.RadicalElement)
        else float(value)
        for value in point
    )


@final
class ExactRationalCellGeometrySource(StrictModule, NonTrainableState):
    """Store rational CSR interpolation actions on an authenticated parent bank.

    Integer CSR routes are nontrainable dynamic leaves. The sorted Fraction bank
    is immutable host metadata, never a device precision array. Unit actions
    preserve parent representative bytes and their bounded error contract.
    Newly generated coefficients use correctly rounded binary64 representatives.
    """

    parent: geometry_owner.CellGeometrySpec
    action_offsets: Array
    action_sources: Array
    action_weights: Array
    coefficients: tuple[Fraction, ...] = eqx.field(static=True)
    layouts: RationalLayouts = eqx.field(static=True)
    parent_binding: str = eqx.field(static=True)
    authority_binding: str = eqx.field(static=True)
    maximum_work: int = eqx.field(static=True)
    maximum_bits: int = eqx.field(static=True)

    @checked
    def __init__(
        self,
        parent: geometry_owner.CellGeometrySpec,
        actions: tuple[RationalAction, ...],
        layouts: RationalLayouts,
        /,
        *,
        maximum_work: int,
        maximum_bits: int,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> None:
        maximum_work = positive_integer(maximum_work, "maximum_work")
        maximum_bits = positive_integer(maximum_bits, "maximum_bits")
        if (
            not layouts
            or tuple(sorted(layouts)) != layouts
            or len({row[0] for row in layouts}) != len(layouts)
        ):
            raise ValueError(
                f"ExactRationalCellGeometrySource.layouts must be nonempty, unique and sorted; got {layouts!r}"
            )
        for _, kind, degree, identity, _ in layouts:
            if kind not in (
                "interval",
                "triangle",
                "quadrilateral",
                "tetrahedron",
                "hexahedron",
                "prism",
                "pyramid",
            ):
                raise ValueError(
                    f"ExactRationalCellGeometrySource.layouts must use supported coordinate families; got {kind!r}"
                )
            # The geometry owner authenticates each element or restricted
            # descendant against this explicit layout identity.
            geometry_owner.coordinate_lagrange_element(kind, degree)
            if not identity:
                raise ValueError(
                    f"ExactRationalCellGeometrySource.layouts must retain explicit element identities; got {identity!r}"
                )
        if not actions:
            raise ValueError(
                "ExactRationalCellGeometrySource.actions must contain a nonempty nodal bank; got no rows"
            )
        entries = sum(len(row) for row in actions)
        bank, scratch = exact_plc_geometry_owner.plc_storage_bounds(
            8 * len(actions), maximum_bits
        )
        with (
            nullcontext(budget)
            if budget is not None
            else geometry_owner.owned_coordinate_preparation(
                128 * (entries + len(actions) + len(layouts) + 1), bank + scratch
            )
        ) as admission:
            admission.admit_work_bound(2 * entries, storage_upper=bank + scratch)
            _require_rational_actions(
                actions,
                parent.coordinates.shape[0],
                maximum_work,
                maximum_bits,
                budget=admission,
            )
            coefficients = tuple(
                sorted({Fraction(n, d) for row in actions for _, n, d in row})
            )
            positions = {value: index for index, value in enumerate(coefficients)}
            offsets, sources, weights = [0], [], []
            for row in actions:
                for index, numerator, denominator in row:
                    sources.append(index)
                    weights.append(positions[Fraction(numerator, denominator)])
                offsets.append(len(sources))
            admission.reserve(4 * entries)
            routing = jax.device_put(
                tuple(
                    np.asarray(value, dtype=np.int64)
                    for value in (offsets, sources, weights)
                )
            )
            admission.retain_basis((*routing, coefficients, layouts))
        self.parent, self.layouts = parent, layouts
        self.action_offsets, self.action_sources, self.action_weights = routing
        self.coefficients = coefficients
        self.maximum_work, self.maximum_bits = maximum_work, maximum_bits
        self.parent_binding = _parent_binding(parent)
        self.authority_binding = self._binding()

    def _binding(self) -> str:
        return canonical_fingerprint(
            {
                "kind": "exact-rational-cell-source",
                "parent": self.parent_binding,
                "actions": array_tree_fingerprint(
                    (self.action_offsets, self.action_sources, self.action_weights)
                ),
                "coefficients": tuple(
                    (value.numerator, value.denominator) for value in self.coefficients
                ),
                "layouts": self.layouts,
                "maximum_work": self.maximum_work,
                "maximum_bits": self.maximum_bits,
            }
        )

    @property
    def source_id(self) -> str:
        return self.authority_binding

    @property
    def field_degree(self) -> int:
        match self.parent.exact_source:
            case (
                exact_algebraic_geometry_owner.ExactAlgebraicCellGeometrySource()
                | ExactRationalCellGeometrySource()
            ) as source:
                return source.field_degree
            case _:
                return 1

    def preparation_query_upper(self) -> int:
        return geometry_owner.cell_geometry_preparation_query_upper(self.parent)

    def preparation_bounds(self) -> tuple[int, int]:
        parent_work, parent_memory = geometry_owner.cell_geometry_preparation_bounds(
            self.parent
        )
        entries = len(self.action_sources)
        count = len(self.action_offsets) - 1
        bank, scratch = exact_plc_geometry_owner.plc_storage_bounds(
            count, self.maximum_bits
        )
        work = 8 * entries * self.parent.coordinates.shape[1] + count
        degree = self.field_degree
        if degree > 1:
            precision = (
                degree * ((degree + 2) * self.maximum_bits + 2102 + degree.bit_length())
                + self.maximum_bits
                + degree.bit_length()
                + 2
            )
            refinements = precision.bit_length()
            scalars = count * self.parent.coordinates.shape[1]
            work = work * degree**3 + 12 * scalars * degree * refinements
            scratch += 128 * scalars * (degree + 1) * ((precision + 7) // 8)
        return parent_work + work, parent_memory + bank + scratch

    def action_rows(self) -> tuple[RationalAction, ...]:
        """Return action rows after validating restored CSR routing and its bank."""
        offsets, sources, weights = self._host_actions()
        if not isinstance(self.coefficients, tuple):
            raise TypeError(
                f"ExactRationalCellGeometrySource.coefficients must use an immutable Fraction bank; got {type(self.coefficients).__name__}"
            )
        if not self.coefficients or any(
            type(value) is not Fraction or value == 0 for value in self.coefficients
        ):
            raise ValueError(
                "ExactRationalCellGeometrySource.coefficients must contain nonzero Fractions; got an empty bank or another coefficient"
            )
        if tuple(sorted(set(self.coefficients))) != self.coefficients:
            raise ValueError(
                "ExactRationalCellGeometrySource.coefficients must be sorted and unique; got an unordered or repeated coefficient"
            )
        if (
            len(offsets) < 2
            or offsets[0] != 0
            or offsets[-1] != len(sources)
            or len(weights) != len(sources)
        ):
            raise ValueError(
                f"ExactRationalCellGeometrySource.actions must align CSR offsets, sources and weights; got lengths {(len(offsets), len(sources), len(weights))!r}"
            )
        if np.any(offsets < 0) or np.any(offsets > len(sources)):
            raise ValueError(
                "ExactRationalCellGeometrySource.actions must bound every CSR offset by its source entries; got an out-of-range offset"
            )
        if any(
            not 1 <= int(offsets[row + 1]) - int(offsets[row]) <= 8
            for row in range(len(offsets) - 1)
        ):
            raise ValueError(
                "ExactRationalCellGeometrySource.actions must use CSR row arity from one through eight; got an empty, decreasing or oversized row"
            )
        if np.any(sources < 0) or np.any(sources >= self.parent.coordinates.shape[0]):
            raise ValueError(
                "ExactRationalCellGeometrySource.actions must index the declared parent coordinate bank; got an out-of-range source"
            )
        if any(value < 0 or value >= len(self.coefficients) for value in weights) or set(
            weights.tolist()
        ) != set(range(len(self.coefficients))):
            raise ValueError(
                "ExactRationalCellGeometrySource.actions must index every declared coefficient; got a missing or out-of-range coefficient"
            )
        rows = []
        for row in range(len(offsets) - 1):
            start, stop = int(offsets[row]), int(offsets[row + 1])
            if np.any(sources[start + 1 : stop] <= sources[start : stop - 1]):
                raise ValueError(
                    f"ExactRationalCellGeometrySource.actions must use sorted unique source indices; got a repeated or unordered entry in row {row}"
                )
            values = tuple(
                self.coefficients[int(weights[index])] for index in range(start, stop)
            )
            if sum(values, Fraction(0)) != 1:
                raise ValueError(
                    f"ExactRationalCellGeometrySource.actions must sum to one; got a nonunit sum in row {row}"
                )
            rows.append(
                tuple(
                    (int(sources[index]), value.numerator, value.denominator)
                    for index, value in zip(range(start, stop), values, strict=True)
                )
            )
        return tuple(rows)

    def _host_actions(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        values = tuple(
            np.asarray(value)
            for value in (self.action_offsets, self.action_sources, self.action_weights)
        )
        if any(value.ndim != 1 or value.dtype != np.dtype(np.int64) for value in values):
            raise ValueError(
                "ExactRationalCellGeometrySource.actions must retain int64 CSR vectors; got another dtype or rank"
            )
        return values[0], values[1], values[2]

    @staticmethod
    def _retained_index(row: int, offsets: np.ndarray, sources: np.ndarray) -> int | None:
        start, stop = int(offsets[row]), int(offsets[row + 1])
        return int(sources[start]) if stop - start == 1 else None

    @checked
    def reindexed(
        self,
        indices: ArrayLike,
        elements: Mapping[str, CellGeometryElement],
        routes: Mapping[str, ArrayLike],
        /,
        *,
        budget: CoordinateEnclosureBudget,
    ) -> ExactRationalCellGeometrySource:
        """Compact coefficient actions while retaining their parent binding."""
        selected = np.asarray(indices)
        if selected.ndim != 1 or selected.dtype.kind not in "iu":
            raise TypeError(
                f"ExactRationalCellGeometrySource.indices must be a one-dimensional integer route; got shape {selected.shape!r} and dtype {selected.dtype}"
            )
        if (
            np.any(selected < 0)
            or np.any(selected >= len(self.action_offsets) - 1)
            or np.any(selected[1:] <= selected[:-1])
        ):
            raise ValueError(
                "ExactRationalCellGeometrySource.indices must be sorted unique declared coefficient rows; got an out-of-range, repeated or unordered row"
            )
        if (
            _parent_binding(self.parent) != self.parent_binding
            or self._binding() != self.authority_binding
        ):
            raise ValueError(
                "ExactRationalCellGeometrySource.authority_binding must authenticate coefficients and source bank; got a mismatched binding"
            )
        actions = self.action_rows()
        return ExactRationalCellGeometrySource(
            self.parent,
            tuple(actions[int(index)] for index in selected),
            rational_source_layouts(elements, routes),
            maximum_work=self.maximum_work,
            maximum_bits=self.maximum_bits,
            budget=budget,
        )

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[ExactRationalCellGeometrySource],
        value: ExactRationalCellGeometrySource,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> ExactRationalCellGeometrySource:
        renewed = cls(
            value.parent,
            value.action_rows(),
            value.layouts,
            maximum_work=value.maximum_work,
            maximum_bits=value.maximum_bits,
            budget=budget,
        )
        if (
            renewed.parent_binding != value.parent_binding
            or renewed.authority_binding != value.authority_binding
        ):
            raise ValueError(
                "ExactRationalCellGeometrySource.authority_binding must authenticate restored coefficients and source bank; got a mismatched binding"
            )
        return renewed

    def prepare(
        self, rounded_vertices: ArrayLike, /, *, budget: CoordinateEnclosureBudget
    ) -> (
        exact_plc_geometry_owner.PreparedExactPlcGeometry
        | exact_algebraic_geometry_owner.PreparedExactAlgebraicGeometry
    ):
        if any(
            isinstance(value, core.Tracer) for value in jax.tree_util.tree_leaves(self)
        ):
            raise TypeError(
                "ExactRationalCellGeometrySource.prepare must run on the host; got a traced source"
            )
        if (
            _parent_binding(self.parent) != self.parent_binding
            or self._binding() != self.authority_binding
        ):
            raise ValueError(
                "ExactRationalCellGeometrySource.authority_binding must authenticate coefficients and source bank; got a mismatched binding"
            )
        rounded = np.asarray(rounded_vertices, dtype=np.float64)
        if rounded.shape != (
            len(self.action_offsets) - 1,
            self.parent.coordinates.shape[1],
        ) or not np.all(np.isfinite(rounded)):
            raise ValueError(
                f"ExactRationalCellGeometrySource.representatives must align with the finite nodal bank; got shape {rounded.shape!r} or a nonfinite coordinate"
            )
        key = (self.source_id, canonical_fingerprint(array_tree_fingerprint(rounded)))
        cached = budget.plc_witness_cache.get(key)
        if cached is None:
            cached = budget.algebraic_witness_cache.get(key)
        if cached is not None:
            return cached
        prepared = self._apply(budget=budget)
        representatives = self._rounded_representatives(prepared.vertices, budget=budget)
        if not np.array_equal(representatives.view(np.uint64), rounded.view(np.uint64)):
            raise ValueError(
                "ExactRationalCellGeometrySource.representatives must preserve inherited bytes and correctly round generated coefficients; got differing binary64 bytes"
            )
        budget.retain_basis((key, prepared))
        if isinstance(
            prepared, exact_algebraic_geometry_owner.PreparedExactAlgebraicGeometry
        ):
            budget.algebraic_witness_cache[key] = prepared
        else:
            budget.plc_witness_cache[key] = prepared
        return prepared

    def _apply(
        self, *, budget: CoordinateEnclosureBudget
    ) -> (
        exact_plc_geometry_owner.PreparedExactPlcGeometry
        | exact_algebraic_geometry_owner.PreparedExactAlgebraicGeometry
    ):
        if (
            _parent_binding(self.parent) != self.parent_binding
            or self._binding() != self.authority_binding
        ):
            raise ValueError(
                "ExactRationalCellGeometrySource.authority_binding must authenticate coefficients and source bank; got a mismatched binding"
            )
        key = ("exact-rational-actions", self.source_id)
        cached = budget.plc_witness_cache.get(key)
        if cached is None:
            cached = budget.algebraic_witness_cache.get(key)
        if cached is not None:
            return cached
        entries = len(self.action_sources)
        count_rows = len(self.action_offsets) - 1
        rounding_work = count_rows * self.parent.coordinates.shape[1]
        count = 2 * entries * self.parent.coordinates.shape[1] + rounding_work
        if count > self.maximum_work:
            from ._coordinate_budget import CoordinateEnclosureResourceError

            raise CoordinateEnclosureResourceError(
                "coefficient_work", self.maximum_work, count, 0
            )
        bank, scratch = exact_plc_geometry_owner.plc_storage_bounds(
            count_rows, self.maximum_bits
        )
        budget.admit_work_bound(count, storage_upper=bank + scratch)
        exact = exact_plc_geometry_owner.ExactPlcBudget(
            self.maximum_work, self.maximum_bits, budget=budget
        )
        parent = self.parent.source_coordinates(budget=budget)
        budget.admit_work_bound(count, storage_upper=bank + scratch)
        with budget.temporary_scope():
            budget.reserve(0, bank + scratch)
            vertices = []
            offsets, sources, weights = self._host_actions()
            for row in range(count_rows):
                point = []
                for axis in range(self.parent.coordinates.shape[1]):
                    value: Fraction | exact_algebraic_geometry_owner.RadicalElement = (
                        Fraction(0)
                    )
                    for index in range(int(offsets[row]), int(offsets[row + 1])):
                        operand = parent[int(sources[index])][axis]
                        if isinstance(
                            operand, exact_algebraic_geometry_owner.RadicalElement
                        ):
                            exact_algebraic_geometry_owner.reserve_algebraic_work(
                                (value, operand), 2, 2, 1, budget=budget
                            )
                        value += self.coefficients[int(weights[index])] * operand
                        exact.charge(
                            2,
                            tuple(coefficient for radicand, coefficient in value.terms)
                            if isinstance(
                                value, exact_algebraic_geometry_owner.RadicalElement
                            )
                            else (value,),
                        )
                    point.append(value)
                vertices.append(tuple(point))
            result = tuple(vertices)
            budget.reserve(rounding_work)
            carrier = np.asarray(self.parent.coordinates, dtype=np.float64)
            error = max(
                (
                    exact_algebraic_geometry_owner.coefficient_rounding_error(
                        value, rounded, budget=budget
                    )
                    for row, point in enumerate(result)
                    for value, rounded in zip(
                        point,
                        _representative_row(
                            point,
                            self._retained_index(row, offsets, sources),
                            carrier,
                            budget,
                        ),
                        strict=True,
                    )
                ),
                default=Fraction(0),
            )
            if any(
                isinstance(value, exact_algebraic_geometry_owner.RadicalElement)
                for point in result
                for value in point
            ):
                algebraic_result = tuple(
                    tuple(
                        value
                        if isinstance(
                            value, exact_algebraic_geometry_owner.RadicalElement
                        )
                        else exact_algebraic_geometry_owner.RadicalElement(value)
                        for value in point
                    )
                    for point in result
                )
                prepared = exact_algebraic_geometry_owner.PreparedExactAlgebraicGeometry(
                    algebraic_result,
                    (False,) * len(result),
                    self.source_id,
                    exact.work + rounding_work,
                    exact.bits,
                    error,
                )
            else:
                rational_result: list[tuple[Fraction, ...]] = []
                for point in result:
                    rational_point: list[Fraction] = []
                    for coordinate in point:
                        match coordinate:
                            case Fraction():
                                rational_point.append(coordinate)
                            case _:
                                raise RuntimeError(
                                    "ExactRationalCellGeometrySource.rational_result must contain rational coordinates on its rational branch; got an algebraic coordinate"
                                )
                    rational_result.append(tuple(rational_point))
                prepared = exact_plc_geometry_owner.PreparedExactPlcGeometry(
                    tuple(rational_result),
                    (False,) * len(result),
                    self.source_id,
                    exact.work + rounding_work,
                    exact.bits,
                    error,
                )
            result = prepared.vertices
            # Retain the prepared host record's coefficient graph explicitly;
            # it is not a registered numerical owner.
            budget.retain_basis(
                (
                    key,
                    result,
                    prepared.on_entity,
                    prepared.maximum_rounding_error,
                    prepared,
                )
            )
            budget.source_bank_cache[key] = result
            if isinstance(
                prepared, exact_algebraic_geometry_owner.PreparedExactAlgebraicGeometry
            ):
                budget.algebraic_witness_cache[key] = prepared
            else:
                budget.plc_witness_cache[key] = prepared
        return prepared

    def _rounded_representatives(
        self,
        vertices: algebra.CoordinateSourceBank,
        *,
        budget: CoordinateEnclosureBudget,
    ) -> np.ndarray:
        """Preserve unit-row bytes; generated exact zeros round canonically to +0."""
        with budget.temporary_scope():
            budget.reserve(
                0,
                (len(vertices) + self.parent.coordinates.shape[0])
                * self.parent.coordinates.shape[1]
                * 8,
            )
            parent = np.asarray(self.parent.coordinates, dtype=np.float64)
            offsets, sources, _ = self._host_actions()
            numeric = np.asarray(
                [
                    _representative_row(
                        point, self._retained_index(row, offsets, sources), parent, budget
                    )
                    for row, point in enumerate(vertices)
                ],
                dtype=np.float64,
            )
            budget.retain_basis((numeric,))
        return numeric

    def representatives(self, *, budget: CoordinateEnclosureBudget) -> np.ndarray:
        """Round after applying rational actions to the authenticated parent bank."""
        return self._rounded_representatives(
            self._apply(budget=budget).vertices, budget=budget
        )


def rational_geometry_lift_preparation_bounds(
    parent: geometry_owner.CellGeometrySpec,
    elements: Mapping[str, CellGeometryElement],
    count: int,
    /,
) -> tuple[int, int]:
    """Bound standalone lifting from the admitted node count and element degree."""
    work, memory = geometry_owner.cell_geometry_preparation_bounds(parent)
    parent_bits = max(
        1075, 1 if parent.exact_source is None else parent.exact_source.maximum_bits
    )
    degree = max(_nodal_element(element).degree for element in elements.values())
    # P1 tensor weights have at most eight terms and three affine factors.
    # Collapsed pyramid arguments add at most one division per coordinate;
    # six degree-sized factors bound both numerator and denominator heights.
    bits = 8 * (parent_bits + 6 * (degree.bit_length() + 1)) + 5
    bank, scratch = exact_plc_geometry_owner.plc_storage_bounds(8 * count, bits)
    entries = 8 * count
    reference_nodes = sum(element.local_dof_count for element in elements.values())
    # Each P1 basis has at most eight polynomials of eight multilinear terms.
    # Evaluation, action scans and retained-object traversal fit the
    # same per-entry metadata bound used by source reconstruction.
    work += 128 * (entries + count + len(elements) + 1)
    work += 512 * reference_nodes + 32 * count * parent.coordinates.shape[1]
    return work, memory + bank + scratch


def rational_geometry_lift(
    parent: geometry_owner.CellGeometrySpec,
    elements: Mapping[str, CellGeometryElement],
    routes: Mapping[str, ArrayLike],
    count: int,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> ExactRationalCellGeometrySource:
    """Build rational degree-one actions in the high-order element's node order."""
    if tuple(sorted(elements)) != parent.block_names:
        raise ValueError(
            f"rational_geometry_lift.blocks must match the parent source; got {tuple(sorted(elements))!r} for {parent.block_names!r}"
        )
    actions: list[RationalAction | None] = [None] * count
    for (name, element), original, parent_route in zip(
        sorted(elements.items()), parent.elements, parent.geometry_dofs, strict=True
    ):
        if (
            original.element_id
            != geometry_owner.coordinate_lagrange_element(element.cell_kind, 1).element_id
        ):
            raise ValueError(
                f"rational_geometry_lift.parent must provide degree-one nodal source elements; got {original.element_id!r}"
            )
        basis = algebra.source_basis(original, budget=budget)
        if basis is None:
            raise ValueError(
                f"rational_geometry_lift.parent must resolve a rational basis; got no basis for {original.element_id!r}"
            )
        nodes = geometry_owner.coordinate_lagrange_reference_nodes(
            element.cell_kind, _nodal_element(element).degree
        )
        if len(nodes) != element.local_dof_count:
            raise ValueError(
                f"rational_geometry_lift.nodes must match element support; got {len(nodes)} nodes for {element.local_dof_count} degrees of freedom"
            )
        weights = []
        for point in nodes:
            if element.cell_kind == "pyramid":
                x, y, z = point
                point = (
                    (Fraction(0), Fraction(0), z)
                    if z == 1
                    else ((x - z / 2) / (1 - z), (y - z / 2) / (1 - z), z)
                )
            weights.append(
                tuple(
                    algebra.require_rational_reference_scalar(
                        algebra.expression_evaluate(term, point, budget=budget)
                    )
                    for term in basis
                )
            )
        target = np.asarray(routes[name], dtype=np.int64)
        source = np.asarray(parent_route, dtype=np.int64)
        if target.ndim != 2 or source.ndim != 2 or target.shape[0] != source.shape[0]:
            raise ValueError(
                f"rational_geometry_lift.routes must retain source cell ordering; got shapes {target.shape!r} and {source.shape!r}"
            )
        for cell in range(target.shape[0]):
            for dof, row in enumerate(weights):
                index = int(target[cell, dof])
                action = tuple(
                    sorted(
                        (int(source[cell, corner]), value.numerator, value.denominator)
                        for corner, value in enumerate(row)
                        if value
                    )
                )
                previous = actions[index]
                if previous is not None and previous != action:
                    raise ValueError(
                        f"rational_geometry_lift.actions must agree on shared nodes; got different actions for node {index}"
                    )
                actions[index] = action
    complete = tuple(row for row in actions if row is not None)
    if len(complete) != count:
        raise ValueError(
            f"rational_geometry_lift.actions must cover every target coordinate; got {len(complete)} of {count} rows"
        )
    parent_bits = max(
        1075, 1 if parent.exact_source is None else parent.exact_source.maximum_bits
    )
    # Products and partial sums of n rational operands have denominator at
    # most the product of the n denominators; the numerator adds log2(n)
    # bits. This also covers odd degree denominators at subnormal coordinates.
    bits = max(
        len(row)
        * (parent_bits + max(max(abs(n).bit_length(), d.bit_length()) for _, n, d in row))
        + len(row).bit_length()
        + 1
        for row in complete
    )
    return ExactRationalCellGeometrySource(
        parent,
        complete,
        rational_source_layouts(elements, routes),
        maximum_work=(2 * sum(len(row) for row in complete) + count)
        * parent.coordinates.shape[1],
        maximum_bits=bits,
        budget=budget,
    )


@checked
def affine_image_geometry_from_routes(
    affine: np.ndarray,
    local_coordinates: tuple[tuple[Fraction, ...], ...],
    elements: Mapping[str, CellGeometryElement],
    routes: Mapping[str, np.ndarray],
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> geometry_owner.CellGeometrySpec:
    """Bind exact affine coefficients to actual coordinate-chart routes.

    A material interface collection may have triple junctions and therefore
    need not be a manifold surface mesh. Its charts still share one exact
    coefficient row per source vertex. The physical volume domain is declared
    and certified independently; no artificial mesh is used as its authority.
    """
    matrix = np.asarray(affine, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(
            f"affine_image_geometry.affine must be a finite (4, 4) matrix; got {matrix.shape}"
        )
    if not np.array_equal(matrix[3], np.asarray((0.0, 0.0, 0.0, 1.0))):
        raise ValueError(
            "affine_image_geometry.affine must retain the affine homogeneous row; got another last row"
        )
    frame = np.concatenate(
        (matrix[:3, 3][None], matrix[:3, :3].T, np.zeros((1, 3), dtype=np.float64))
    )
    return _affine_image_geometry_from_frame(
        frame, local_coordinates, elements, routes, budget=budget
    )


def _affine_image_geometry_from_frame(
    frame: np.ndarray,
    local_coordinates: tuple[tuple[Fraction, ...], ...],
    elements: Mapping[str, CellGeometryElement],
    routes: Mapping[str, np.ndarray],
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> geometry_owner.CellGeometrySpec:
    from ._cell_mesh import CellMesh

    if not local_coordinates or any(
        len(row) != 3 or any(not isinstance(value, Fraction) for value in row)
        for row in local_coordinates
    ):
        raise ValueError(
            "affine_image_geometry.local_coordinates must contain three exact rational coordinates per coefficient row; got an incompatible bank"
        )
    actions = []
    for local in local_coordinates:
        weights = (Fraction(1), *local, -sum(local, Fraction(0)))
        actions.append(
            tuple(
                (position, value.numerator, value.denominator)
                for position, value in enumerate(weights)
                if value
            )
        )
    parent = geometry_owner.CellGeometrySpec.affine(
        CellMesh.from_tetrahedra(
            frame, np.asarray(((0, 1, 2, 3),), dtype=np.int32)
        ),
        budget=budget,
    )
    bits = 4 * (
        1078
        + max(
            max(abs(numerator).bit_length(), denominator.bit_length())
            for row in actions
            for _, numerator, denominator in row
        )
    )
    source = ExactRationalCellGeometrySource(
        parent,
        tuple(actions),
        rational_source_layouts(elements, routes),
        maximum_work=budget.maximum_work_units,
        maximum_bits=bits,
        budget=budget,
    )
    return geometry_owner.CellGeometrySpec(
        elements,
        routes,
        source.representatives(budget=budget),
        exact_source=source,
        budget=budget,
    )
