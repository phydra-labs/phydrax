#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from fractions import Fraction
from itertools import product
from math import comb
from typing import Any, cast, Literal, Protocol, runtime_checkable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, core as jax_core
from jax.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..exterior._form_type import FormType, FormValueSpec
from ..typing import ConvertibleToArray
from ._cell_mesh import CellMesh
from ._coordinate_enclosure import (
    add,
    axes,
    compose_expression,
    constant,
    CoordinateSourceBank,
    derivative,
    evaluate_expression,
    Expression,
    ExpressionInput,
    linear_combinations,
    multiply,
    normalize_pyramid_reference_derivative,
    normalize_pyramid_reference_expression,
    Polynomial,
    power as polynomial_power,
    RationalPolynomial,
    scale,
    sum_polynomials,
)
from ._exact_algebraic_geometry import coefficient_payload, ExactScalar, RadicalElement
from ._nested_reference_types import (
    exact_reference_cell_vertices,
    freeze_reference_chart_arguments,
    reference_chart_arguments_have_nonvanishing_denominators,
    reference_chart_arguments_identity,
    ReferenceCellMapWitnesses,
    validate_reference_cell_map_witnesses,
)
from ._reference_cell import reference_cell_topology
from .fem._reference import FiniteElementSpec


class SweptCoordinateTabulator:
    """Polynomial profile basis tensored with a linear station coordinate."""

    source: FiniteElementSpec

    def __init__(self, source: FiniteElementSpec, /) -> None:
        self.source = source

    def __call__(self, points: ArrayLike, /) -> tuple[Array, Array]:
        reference = jnp.asarray(points, dtype=jnp.float64)
        if reference.ndim != 2 or reference.shape[1] != 3:
            raise ValueError("Swept coordinate points must have three reference axes.")
        profile, gradients = self.source.tabulate(reference[:, :2])
        station = reference[:, 2:3]
        lower_gradients = jnp.concatenate(
            (gradients * (1.0 - station[..., None]), -profile[..., None]),
            axis=-1,
        )
        upper_gradients = jnp.concatenate(
            (gradients * station[..., None], profile[..., None]),
            axis=-1,
        )
        return (
            jnp.concatenate((profile * (1.0 - station), profile * station), axis=-1),
            jnp.concatenate((lower_gradients, upper_gradients), axis=1),
        )


def swept_coordinate_element(source: FiniteElementSpec, /) -> FiniteElementSpec:
    """Lift an H1 scalar triangle/quad map exactly through a linear station axis."""

    if not isinstance(source, FiniteElementSpec):
        raise TypeError("source must be FiniteElementSpec.")
    if (
        source.cell_kind not in ("triangle", "quadrilateral")
        or source.conformity != "H1"
        or source.degree < 1
        or source.mapping != "identity"
        or source.value_shape
    ):
        raise ValueError(
            "Swept coordinate maps require a positive-order scalar H1 profile element."
        )
    source_topology = reference_cell_topology(source.cell_kind)
    target_kind = "prism" if source.cell_kind == "triangle" else "hexahedron"
    target_topology = reference_cell_topology(target_kind)
    dof_count = source.local_dof_count
    corner_count = len(source_topology.vertices)
    nodes = np.concatenate(
        tuple(
            np.column_stack(
                (
                    np.asarray(source.reference_nodes),
                    np.full(dof_count, station, dtype=np.float64),
                )
            )
            for station in (0.0, 1.0)
        )
    )
    target_entities = tuple(
        tuple(frozenset(vertices) for vertices in level)
        for level in target_topology.entities
    )
    entity_dofs: list[list[list[int]]] = [
        [[] for _ in level] for level in target_topology.entities
    ]
    for side in range(2):
        for dimension, level in enumerate(source.entity_dofs):
            if dimension >= len(target_entities):
                continue
            for entity, dofs in enumerate(level):
                vertices = frozenset(
                    vertex + side * corner_count
                    for vertex in source_topology.entities[dimension][entity]
                )
                target = target_entities[dimension].index(vertices)
                entity_dofs[dimension][target].extend(
                    int(dof) + side * dof_count for dof in dofs
                )
    tabulator = SweptCoordinateTabulator(source)
    return FiniteElementSpec(
        "SweptCoordinate",
        target_kind,
        source.degree,
        nodes,
        tuple(tuple(tuple(dofs) for dofs in level) for level in entity_dofs),
        value_spec=FormValueSpec(FormType(3, 0), proxy="scalar"),
        representation=source.representation,
        tabulator=tabulator,
        tabulator_id=canonical_fingerprint(
            {
                "kind": "exact-swept-coordinate-source",
                "profile": source.element_id,
                "station_axis": 2,
                "station_degree": 1,
                "endpoint_order": (0, 1),
            }
        ),
    )


@runtime_checkable
class CellGeometryElement(Protocol):
    """Reference element contract required by a cell geometry layout."""

    @property
    def cell_kind(self) -> str: ...

    @property
    def conformity(self) -> str: ...

    @property
    def local_dof_count(self) -> int: ...

    @property
    def element_id(self) -> str: ...


@runtime_checkable
class CellGeometryCoordinateMap(Protocol):
    """Validated physical-coordinate source bound to one exact source revision."""

    @property
    def source_id(self) -> str: ...

    @property
    def source_revision(self) -> str: ...

    @property
    def source_map_id(self) -> str: ...


def _capture_exact_coordinate_bank(
    source_map: CellGeometryCoordinateMap, coordinates: np.ndarray, /
) -> CoordinateSourceBank | None:
    prepare = getattr(source_map, "source_coordinates", None)
    if prepare is None:
        return None
    if not callable(prepare):
        raise TypeError("A source coordinate bank provider must be callable.")
    raw = prepare()
    if not isinstance(raw, tuple) or any(
        not isinstance(row, tuple)
        or any(not isinstance(value, (Fraction, RadicalElement)) for value in row)
        for row in raw
    ):
        raise TypeError(
            "Exact coordinate banks require immutable tuple rows of canonical exact scalars."
        )
    # The boundary above owns the nested runtime kind check.
    bank = cast(CoordinateSourceBank, raw)
    if len(bank) != coordinates.shape[0] or any(
        len(row) != coordinates.shape[1] for row in bank
    ):
        raise ValueError(
            "Exact source coordinates must match the coefficient bank shape."
        )
    if bank:
        try:
            rounded = np.asarray(bank, dtype=np.float64)
        except OverflowError as error:
            raise ValueError(
                "Exact source coordinates require a finite binary64 realization."
            ) from error
        if not np.array_equal(rounded, coordinates):
            raise ValueError(
                "Exact source coordinates do not match the execution coefficient shadows."
            )
    return bank


class CellGeometryRestrictionSource(StrictModule, NonTrainableState):
    """Source-ID-bound parent cells and oriented vertices of a restriction."""

    source_geometry_id: str = eqx.field(static=True)
    source_topology_id: str = eqx.field(static=True)
    block_names: tuple[str, ...] = eqx.field(static=True)
    parent_cell_ids: tuple[Array, ...]
    parent_vertex_ids: tuple[Array, ...]
    parent_reference_maps: tuple[ReferenceCellMapWitnesses | None, ...]
    source_block_names: tuple[str, ...] | None = eqx.field(static=True)
    carrier_publication: Literal["source_image", "published_affine_roots"] = eqx.field(
        static=True
    )
    carrier_geometry: CellGeometrySpec | None
    restriction_source_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_geometry_id: str,
        source_topology_id: str,
        block_parent_cell_ids: Mapping[str, ArrayLike],
        block_parent_vertex_ids: Mapping[str, ArrayLike],
        /,
        *,
        block_source_blocks: Mapping[str, str] | None = None,
        carrier_publication: Literal[
            "source_image", "published_affine_roots"
        ] = "source_image",
        block_parent_reference_maps: Mapping[str, ReferenceCellMapWitnesses | None]
        | None = None,
        carrier_geometry: CellGeometrySpec | None = None,
    ) -> None:
        source_geometry = str(source_geometry_id).strip()
        source_topology = str(source_topology_id).strip()
        if not source_geometry or not source_topology:
            raise ValueError(
                "Restriction sources require source geometry and topology IDs."
            )
        if carrier_publication not in ("source_image", "published_affine_roots"):
            raise ValueError("Unsupported coordinate carrier publication law.")
        if (carrier_publication == "source_image") != (carrier_geometry is None):
            raise ValueError(
                "A retained carrier geometry is required exactly for published affine roots."
            )
        names = tuple(sorted(str(name) for name in block_parent_cell_ids))
        if (
            not names
            or any(not name for name in names)
            or set(names) != set(block_parent_vertex_ids)
        ):
            raise ValueError(
                "Restriction parent tables must name the same target blocks."
            )
        if block_source_blocks is not None and (
            set(block_source_blocks) != set(names)
            or any(
                not isinstance(name, str) or not name
                for name in block_source_blocks.values()
            )
        ):
            raise ValueError(
                "Restriction source block names must align with target blocks."
            )
        if block_parent_reference_maps is not None and set(
            block_parent_reference_maps
        ) != set(names):
            raise ValueError("Reference-map ancestry must cover every target block.")
        cells = []
        vertices = []
        reference_maps = []
        for name in names:
            cell_ids = np.asarray(block_parent_cell_ids[name], dtype=np.int64)
            vertex_ids = np.asarray(block_parent_vertex_ids[name], dtype=np.int64)
            if (
                cell_ids.ndim != 1
                or not cell_ids.size
                or np.any(cell_ids < 0)
                or vertex_ids.ndim != 2
                or vertex_ids.shape[0] != cell_ids.size
                or vertex_ids.shape[1] < 2
                or np.any(vertex_ids < -1)
            ):
                raise ValueError("Restriction parent IDs must align with target cells.")
            reference_map = (
                None
                if block_parent_reference_maps is None
                else block_parent_reference_maps[name]
            )
            for row in vertex_ids:
                valid = row[row >= 0]
                if (
                    (
                        valid.size < 2
                        and not (valid.size == 0 and reference_map is not None)
                    )
                    or np.unique(valid).size != valid.size
                    or np.any(row[: valid.size] < 0)
                    or np.any(row[valid.size :] != -1)
                ):
                    raise ValueError(
                        "Parent vertex IDs must be unique with trailing -1 padding; empty support requires exact reference-map ancestry."
                    )
            if reference_map is not None:
                chart_arguments = (
                    None
                    if reference_map.source_reference_chart_arguments is None
                    else freeze_reference_chart_arguments(
                        reference_map.source_reference_chart_arguments
                    )
                )
                retained_map = ReferenceCellMapWitnesses(
                    reference_map.target_cell_ids,
                    reference_map.source_cell_ids,
                    reference_map.target_cell_kind,
                    reference_map.source_cell_kind,
                    reference_map.source_reference_vertices,
                    reference_map.source_reference_exact_vertices,
                    chart_arguments,
                )
                validate_reference_cell_map_witnesses(retained_map)
                target_ids = np.asarray(retained_map.target_cell_ids, dtype=np.int64)
                source_ids = np.asarray(retained_map.source_cell_ids, dtype=np.int64)
                if not np.array_equal(
                    target_ids, np.arange(cell_ids.size)
                ) or not np.array_equal(source_ids, cell_ids):
                    raise ValueError(
                        "Reference-map ancestry must align local targets and parent-cell IDs."
                    )
                exact = exact_reference_cell_vertices(retained_map)
                target_copy = target_ids.copy()
                source_copy = source_ids.copy()
                point_copy = np.asarray(
                    retained_map.source_reference_vertices, dtype=np.float64
                ).copy()
                target_copy.setflags(write=False)
                source_copy.setflags(write=False)
                point_copy.setflags(write=False)
                reference_map = ReferenceCellMapWitnesses(
                    target_copy,
                    source_copy,
                    retained_map.target_cell_kind,
                    retained_map.source_cell_kind,
                    point_copy,
                    exact,
                    chart_arguments,
                )
            cells.append(jnp.asarray(cell_ids))
            vertices.append(jnp.asarray(vertex_ids))
            reference_maps.append(reference_map)
        reference_map_identity = []
        for reference_map in reference_maps:
            if reference_map is None:
                reference_map_identity.append(None)
                continue
            exact = exact_reference_cell_vertices(reference_map)
            exact_payload = tuple(
                tuple(
                    tuple((value.numerator, value.denominator) for value in point)
                    for point in cell
                )
                for cell in exact
            )
            reference_map_identity.append(
                {
                    "target_cell_kind": reference_map.target_cell_kind,
                    "source_cell_kind": reference_map.source_cell_kind,
                    "target_cell_ids": array_tree_fingerprint(
                        np.asarray(reference_map.target_cell_ids)
                    ),
                    "source_cell_ids": array_tree_fingerprint(
                        np.asarray(reference_map.source_cell_ids)
                    ),
                    "source_reference_vertices": array_tree_fingerprint(
                        np.asarray(reference_map.source_reference_vertices)
                    ),
                    "exact_reference_vertices": exact_payload,
                    "source_reference_chart_arguments": reference_chart_arguments_identity(
                        reference_map.source_reference_chart_arguments
                    ),
                }
            )
        self.source_geometry_id = source_geometry
        self.source_topology_id = source_topology
        self.block_names = names
        self.parent_cell_ids = tuple(cells)
        self.parent_vertex_ids = tuple(vertices)
        self.parent_reference_maps = tuple(reference_maps)
        self.source_block_names = (
            None
            if block_source_blocks is None
            else tuple(block_source_blocks[name] for name in names)
        )
        self.carrier_publication = carrier_publication
        self.carrier_geometry = carrier_geometry
        self.restriction_source_id = canonical_fingerprint(
            {
                "kind": "cell-geometry-restriction-source",
                "source_geometry": source_geometry,
                "source_topology": source_topology,
                "blocks": names,
                "parent_cells": [array_tree_fingerprint(value) for value in cells],
                "parent_vertices": [array_tree_fingerprint(value) for value in vertices],
                "parent_reference_maps": reference_map_identity,
                "source_blocks": self.source_block_names,
                "carrier_publication": carrier_publication,
                "carrier_geometry": (
                    None
                    if carrier_geometry is None
                    else carrier_geometry.geometry_layout_id
                ),
            }
        )

    @property
    def block_parent_cell_ids(self) -> dict[str, Array]:
        return dict(zip(self.block_names, self.parent_cell_ids, strict=True))

    @property
    def block_parent_vertex_ids(self) -> dict[str, Array]:
        return dict(zip(self.block_names, self.parent_vertex_ids, strict=True))

    @property
    def block_parent_reference_maps(
        self,
    ) -> dict[str, ReferenceCellMapWitnesses | None]:
        return dict(zip(self.block_names, self.parent_reference_maps, strict=True))

    @property
    def block_source_blocks(self) -> dict[str, str] | None:
        if self.source_block_names is None:
            return None
        return dict(zip(self.block_names, self.source_block_names, strict=True))

    @classmethod
    def from_stored_fields(
        cls, value: CellGeometryRestrictionSource, /
    ) -> CellGeometryRestrictionSource:
        return cls(
            value.source_geometry_id,
            value.source_topology_id,
            value.block_parent_cell_ids,
            value.block_parent_vertex_ids,
            block_source_blocks=value.block_source_blocks,
            carrier_publication=value.carrier_publication,
            block_parent_reference_maps=value.block_parent_reference_maps,
            carrier_geometry=value.carrier_geometry,
        )


def _canonical_chart_arguments(
    values: tuple[Mapping[tuple[int, ...], Any], ...],
    target_dimension: int,
    /,
) -> tuple[tuple[tuple[tuple[int, ...], Any], ...], ...]:
    result = []
    for polynomial in values:
        terms = []
        for exponent, coefficient in polynomial.items():
            powers = tuple(int(value) for value in exponent)
            if (
                len(powers) != target_dimension
                or any(value < 0 for value in powers)
                or not coefficient
            ):
                if not coefficient:
                    continue
                raise ValueError(
                    "Chart polynomial terms must use nonnegative target exponents."
                )
            terms.append((powers, coefficient))
        result.append(tuple(sorted(terms)))
    return tuple(result)


def _chart_polynomials(
    terms: tuple[tuple[tuple[tuple[int, ...], Any], ...], ...], /
) -> tuple[dict[tuple[int, ...], Any], ...]:
    return tuple(dict(polynomial) for polynomial in terms)


type _CoordinateCoefficientIdentity = (
    int | float | tuple[str, int, int] | tuple[str, tuple[tuple[int, int, int], ...]]
)

type _ExactPolynomialTerms = tuple[tuple[tuple[int, ...], ExactScalar], ...]
type _ExactRationalChartTerms = tuple[_ExactPolynomialTerms, _ExactPolynomialTerms]


def _coordinate_coefficient_identity(
    value: ExactScalar | int | float, /
) -> _CoordinateCoefficientIdentity:
    """Retain exact rational/radical values without rounding them for identity."""
    if isinstance(value, Fraction):
        return ("rational", value.numerator, value.denominator)
    if isinstance(value, RadicalElement):
        return (
            "radical",
            tuple(
                (radicand, coefficient.numerator, coefficient.denominator)
                for radicand, coefficient in value.terms
            ),
        )
    return value


def _canonical_rational_chart_arguments(
    values: Sequence[ExpressionInput], target_dimension: int, /
) -> tuple[_ExactRationalChartTerms, ...]:
    zero = (0,) * target_dimension
    unit = {zero: Fraction(1)}
    result = []
    for value in values:
        if isinstance(value, RationalPolynomial):
            rational = value
        elif isinstance(value, Mapping):
            rational = RationalPolynomial(value, unit)
        else:
            raise TypeError(
                "Per-cell chart arguments must be exact coordinate expressions."
            )
        numerator = _canonical_chart_arguments((rational.numerator,), target_dimension)[0]
        denominator = _canonical_chart_arguments(
            (rational.denominator,), target_dimension
        )[0]
        result.append((numerator, denominator))
    return tuple(result)


def _rational_chart_arguments_from_terms(
    chart: Sequence[_ExactRationalChartTerms], /
) -> tuple[RationalPolynomial, ...]:
    return tuple(
        RationalPolynomial(dict(numerator), dict(denominator))
        for numerator, denominator in chart
    )


def _per_cell_chart_arguments_from_terms(
    chart: Sequence[_ExactRationalChartTerms], /
) -> tuple[Expression, ...]:
    result: list[Expression] = []
    for numerator, denominator in chart:
        zero = (0,) * len(denominator[0][0]) if denominator else ()
        if (
            len(denominator) == 1
            and denominator[0][0] == zero
            and denominator[0][1] == Fraction(1)
        ):
            result.append(dict(numerator))
        else:
            result.append(RationalPolynomial(dict(numerator), dict(denominator)))
    return tuple(result)


def _canonical_cell_id_tuple(
    values: Sequence[int] | np.ndarray, name: str, /
) -> tuple[int, ...]:
    identifiers = np.asarray(values)
    if identifiers.ndim != 1 or identifiers.dtype.kind not in "iu":
        raise TypeError(f"{name} must be a one-dimensional integer vector.")
    if identifiers.dtype.kind == "u" and np.any(identifiers > np.iinfo(np.int64).max):
        raise ValueError(f"{name} must fit the canonical signed 64-bit ID range.")
    if np.any(identifiers < 0):
        raise ValueError(f"{name} must be nonnegative.")
    return tuple(int(value) for value in identifiers.tolist())


def _evaluate_polynomial_values(polynomial: Polynomial, points: Array, /) -> Array:
    result = jnp.zeros((points.shape[0],), dtype=jnp.float64)
    for exponent, coefficient in polynomial.items():
        term = jnp.full((points.shape[0],), float(coefficient), dtype=jnp.float64)
        for axis, power in enumerate(exponent):
            if power:
                term = term * points[:, axis] ** power
        result = result + term
    return result


def _evaluate_polynomial(polynomial: Polynomial, points: Array, /) -> tuple[Array, Array]:
    gradients: list[Array] = []
    for axis in range(points.shape[1]):
        differentiated = derivative(polynomial, axis)
        if isinstance(differentiated, RationalPolynomial):
            raise RuntimeError("A polynomial derivative must remain polynomial.")
        gradients.append(_evaluate_polynomial_values(differentiated, points))
    return _evaluate_polynomial_values(polynomial, points), jnp.stack(gradients, axis=-1)


def _evaluate_expression_values(
    expression: Expression, points: Array, /
) -> tuple[Array, Array]:
    if isinstance(expression, RationalPolynomial):
        numerator = _evaluate_polynomial_values(dict(expression.numerator), points)
        denominator = _evaluate_polynomial_values(dict(expression.denominator), points)
        failed = denominator == 0.0
        safe_denominator = jnp.where(failed, 1.0, denominator)
        return numerator / safe_denominator, failed
    values = _evaluate_polynomial_values(expression, points)
    return values, jnp.zeros((points.shape[0],), dtype=jnp.bool_)


def _evaluate_pyramid_rational_values(
    value: RationalPolynomial, points: Array, /
) -> tuple[Array, Array]:
    box_points, apex, invalid_apex_coordinates = _pyramid_box_evaluation_points(points)
    normalized = normalize_pyramid_reference_expression(value)
    values, failed = _evaluate_expression_values(normalized.expression, box_points)
    if normalized.apex_value is None:
        failed = failed | apex
        apex_value = 0.0
    elif not reference_chart_arguments_have_nonvanishing_denominators(
        "hexahedron", (normalized.expression,)
    ):
        failed = failed | apex
        apex_value = 0.0
    else:
        apex_value = float(normalized.apex_value)
    values = jnp.where(apex, apex_value, values)
    failed = failed | invalid_apex_coordinates
    return values, jnp.any(failed)


def _evaluate_rational_values(
    value: RationalPolynomial, points: Array, /, *, pyramid_target: bool
) -> tuple[Array, Array]:
    if (
        pyramid_target
        and evaluate_expression(
            value.denominator, (Fraction(1, 2), Fraction(1, 2), Fraction(1))
        )
        == 0
    ):
        return _evaluate_pyramid_rational_values(value, points)
    values, failed = _evaluate_expression_values(value, points)
    return values, jnp.any(failed)


def _composed_values(
    source_element: ScalarCoordinateElement,
    chart_arguments: tuple[Expression, ...],
    points: ArrayLike,
    /,
    *,
    target_cell_kind: str,
) -> Array:
    reference = jnp.asarray(points, dtype=jnp.float64)
    if reference.ndim != 2:
        raise ValueError("Coordinate tabulation points must be a rank-two array.")
    target_is_pyramid = reference_cell_topology(target_cell_kind).name == "pyramid"
    mapped: list[Array] = []
    invalid: Array | None = None
    for argument in chart_arguments:
        if isinstance(argument, RationalPolynomial):
            values, failed = _evaluate_rational_values(
                argument, reference, pyramid_target=target_is_pyramid
            )
            invalid = failed if invalid is None else invalid | failed
        else:
            values, failed = _evaluate_expression_values(argument, reference)
            failed = jnp.any(failed)
            invalid = failed if invalid is None else invalid | failed
        mapped.append(values)
    source_points = jnp.stack(mapped, axis=-1)
    if invalid is not None:
        message = (
            "Coordinate chart denominator vanishes or pyramid-apex value "
            "continuation is undefined at an evaluation point."
        )
        if isinstance(invalid, jax_core.Tracer):
            source_points = eqx.error_if(source_points, invalid, message)
        elif bool(invalid):
            raise ValueError(message)
    return _source_element_values(source_element, source_points)


def _source_element_values(
    source_element: ScalarCoordinateElement, points: Array, /
) -> Array:
    if isinstance(source_element, RationalComposedCellGeometryElement):
        return _composed_values(
            source_element.source_element,
            source_element.chart_arguments,
            points,
            target_cell_kind=source_element.cell_kind,
        )
    if isinstance(source_element, PolynomialComposedCellGeometryElement):
        return _composed_values(
            source_element.source_element,
            source_element.chart_arguments,
            points,
            target_cell_kind=source_element.cell_kind,
        )
    if isinstance(source_element, BarycentricCellGeometryElement):
        return (
            _source_element_values(source_element.source_element, points)
            @ source_element.barycentric_weights
        )
    values, _ = source_element.tabulate(points)
    return values


def _pyramid_box_evaluation_points(points: Array, /) -> tuple[Array, Array, Array]:
    apex = points[:, 2] == 1.0
    width = jnp.where(apex, 1.0, 1.0 - points[:, 2])
    u = (points[:, 0] - 0.5 * points[:, 2]) / width
    v = (points[:, 1] - 0.5 * points[:, 2]) / width
    box_points = jnp.stack((u, v, points[:, 2]), axis=-1)
    invalid_apex_coordinates = apex & ((points[:, 0] != 0.5) | (points[:, 1] != 0.5))
    return box_points, apex, invalid_apex_coordinates


def _evaluate_pyramid_rational_polynomial(
    value: RationalPolynomial, points: Array, /
) -> tuple[Array, Array, Array]:
    box_points, apex, invalid_apex_coordinates = _pyramid_box_evaluation_points(points)
    normalized = normalize_pyramid_reference_expression(value)
    values, failed = _evaluate_expression_values(normalized.expression, box_points)
    if normalized.apex_value is None:
        failed = failed | apex
        apex_value = 0.0
    elif not reference_chart_arguments_have_nonvanishing_denominators(
        "hexahedron", (normalized.expression,)
    ):
        failed = failed | apex
        apex_value = 0.0
    else:
        apex_value = float(normalized.apex_value)
    values = jnp.where(apex, apex_value, values)
    failed = failed | invalid_apex_coordinates

    gradients: list[Array] = []
    for axis in range(points.shape[1]):
        normalized_derivative = normalize_pyramid_reference_derivative(value, axis)
        derivative_values, derivative_failed = _evaluate_expression_values(
            normalized_derivative.expression, box_points
        )
        if normalized_derivative.apex_value is None:
            derivative_failed = derivative_failed | apex
            derivative_apex_value = 0.0
        elif not reference_chart_arguments_have_nonvanishing_denominators(
            "hexahedron", (normalized_derivative.expression,)
        ):
            derivative_failed = derivative_failed | apex
            derivative_apex_value = 0.0
        else:
            derivative_apex_value = float(normalized_derivative.apex_value)
        gradients.append(jnp.where(apex, derivative_apex_value, derivative_values))
        failed = failed | derivative_failed
    return values, jnp.stack(gradients, axis=-1), jnp.any(failed)


def _evaluate_rational_polynomial(
    value: RationalPolynomial, points: Array, /, *, pyramid_target: bool = False
) -> tuple[Array, Array, Array]:
    if (
        pyramid_target
        and evaluate_expression(
            value.denominator, (Fraction(1, 2), Fraction(1, 2), Fraction(1))
        )
        == 0
    ):
        return _evaluate_pyramid_rational_polynomial(value, points)
    numerator, numerator_gradient = _evaluate_polynomial(dict(value.numerator), points)
    denominator, denominator_gradient = _evaluate_polynomial(
        dict(value.denominator), points
    )
    failed = denominator == 0.0
    safe_denominator = jnp.where(failed, 1.0, denominator)
    gradient = (
        numerator_gradient * safe_denominator[:, None]
        - denominator_gradient * numerator[:, None]
    ) / (safe_denominator[:, None] ** 2)
    return numerator / safe_denominator, gradient, jnp.any(failed)


@jax.custom_jvp
def _chart_values_with_exact_jacobian(
    points: Array, values: Array, jacobian: Array, /
) -> Array:
    del points, jacobian
    return values


@_chart_values_with_exact_jacobian.defjvp
def _chart_values_with_exact_jacobian_jvp(
    primals: tuple[Array, Array, Array], tangents: tuple[Array, Array, Array]
) -> tuple[Array, Array]:
    _, values, jacobian = primals
    point_tangent, _, _ = tangents
    return values, (jacobian @ point_tangent[..., None])[..., 0]


def _composed_tabulation(
    source_element: ScalarCoordinateElement,
    chart_arguments: tuple[Expression, ...],
    points: ArrayLike,
    /,
    *,
    target_cell_kind: str | None = None,
) -> tuple[Array, Array]:
    reference = jnp.asarray(points, dtype=jnp.float64)
    if reference.ndim != 2:
        raise ValueError("Coordinate tabulation points must be a rank-two array.")
    target_is_pyramid = (
        target_cell_kind is not None
        and reference_cell_topology(target_cell_kind).name == "pyramid"
    )
    has_rational_chart = target_is_pyramid and any(
        isinstance(argument, RationalPolynomial) for argument in chart_arguments
    )
    mapped: list[Array] = []
    chart_gradients: list[Array] = []
    invalid: Array | None = None
    for argument in chart_arguments:
        if isinstance(argument, RationalPolynomial):
            values, gradients, failed = _evaluate_rational_polynomial(
                argument, reference, pyramid_target=target_is_pyramid
            )
            invalid = failed if invalid is None else invalid | failed
        else:
            values, gradients = _evaluate_polynomial(argument, reference)
        mapped.append(values)
        chart_gradients.append(gradients)
    source_points = jnp.stack(mapped, axis=-1)
    jacobian = jnp.stack(chart_gradients, axis=1)
    if invalid is not None:
        message = (
            "Coordinate chart denominator vanishes or pyramid-apex continuation "
            "is undefined at an evaluation point."
        )
        if isinstance(invalid, jax_core.Tracer):
            source_points, jacobian = eqx.error_if(
                (source_points, jacobian), invalid, message
            )
        elif bool(invalid):
            raise ValueError(message)
    if has_rational_chart:
        source_points = _chart_values_with_exact_jacobian(
            reference, source_points, jacobian
        )
    values, gradients = source_element.tabulate(source_points)
    return values, gradients @ jacobian


class BarycentricCellGeometryElement(StrictModule, NonTrainableState):
    """Full coefficient action over a retained source coordinate basis."""

    source_element: ScalarCoordinateElement
    barycentric_weights: Array
    exact_barycentric_weights: tuple[tuple[Fraction, ...], ...] = eqx.field(static=True)
    cell_kind: str = eqx.field(static=True)
    conformity: str = eqx.field(static=True)
    local_dof_count: int = eqx.field(static=True)
    degree: int = eqx.field(static=True)
    topological_dimension: int = eqx.field(static=True)
    element_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_element: ScalarCoordinateElement,
        barycentric_weights: ArrayLike,
        /,
    ) -> None:
        if not isinstance(source_element, ScalarCoordinateElement):
            raise TypeError("source_element must be a scalar coordinate element.")
        weights = np.asarray(barycentric_weights, dtype=np.float64)
        width = source_element.local_dof_count
        if (
            weights.ndim != 2
            or weights.shape[0] != width
            or not np.all(np.isfinite(weights))
        ):
            raise ValueError("Barycentric coordinate action must align with source DOFs.")
        self.source_element = source_element
        self.barycentric_weights = jnp.asarray(weights)
        self.exact_barycentric_weights = tuple(
            tuple(Fraction(float(value)) for value in row) for row in weights
        )
        self.cell_kind = source_element.cell_kind
        self.conformity = "H1"
        self.local_dof_count = weights.shape[1]
        self.degree = source_element.degree
        self.topological_dimension = reference_cell_topology(self.cell_kind).dimension
        self.element_id = canonical_fingerprint(
            {
                "kind": "barycentric-cell-geometry-element",
                "source": source_element.element_id,
                "weights": array_tree_fingerprint(weights),
            }
        )

    def tabulate(self, points: ArrayLike, /) -> tuple[Array, Array]:
        values, gradients = self.source_element.tabulate(points)
        action = self.barycentric_weights
        return (
            values @ action,
            jnp.swapaxes(jnp.swapaxes(gradients, 1, 2) @ action, 1, 2),
        )


class RestrictedCellGeometryElement(StrictModule, NonTrainableState):
    """Source basis restricted by an exact affine reference-coordinate map."""

    source_element: ScalarCoordinateElement
    matrix: tuple[tuple[Fraction, ...], ...] = eqx.field(static=True)
    offset: tuple[Fraction, ...] = eqx.field(static=True)
    cell_kind: str = eqx.field(static=True)
    conformity: str = eqx.field(static=True)
    local_dof_count: int = eqx.field(static=True)
    degree: int = eqx.field(static=True)
    topological_dimension: int = eqx.field(static=True)
    element_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_element: ScalarCoordinateElement,
        cell_kind: str,
        matrix: ArrayLike,
        offset: ArrayLike,
        /,
    ) -> None:
        if not isinstance(source_element, ScalarCoordinateElement):
            raise TypeError("source_element must be a scalar coordinate element.")
        matrix_ = np.asarray(matrix, dtype=np.float64)
        offset_ = np.asarray(offset, dtype=np.float64)
        target_dimension = reference_cell_topology(cell_kind).dimension
        source_dimension = reference_cell_topology(source_element.cell_kind).dimension
        if (
            matrix_.shape != (source_dimension, target_dimension)
            or offset_.shape != (source_dimension,)
            or not np.all(np.isfinite(matrix_))
            or not np.all(np.isfinite(offset_))
        ):
            raise ValueError(
                "Reference restriction matrix and offset have incompatible dimensions."
            )
        self.source_element = source_element
        self.matrix = tuple(
            tuple(Fraction(float(value)) for value in row) for row in matrix_
        )
        self.offset = tuple(Fraction(float(value)) for value in offset_)
        self.cell_kind = str(cell_kind)
        self.conformity = "H1"
        self.local_dof_count = source_element.local_dof_count
        self.degree = source_element.degree
        self.topological_dimension = target_dimension
        self.element_id = canonical_fingerprint(
            {
                "kind": "restricted-cell-geometry-element",
                "source": source_element.element_id,
                "target": cell_kind,
                "matrix": tuple(
                    tuple(str(value) for value in row) for row in self.matrix
                ),
                "offset": tuple(str(value) for value in self.offset),
            }
        )

    def tabulate(self, points: ArrayLike, /) -> tuple[Array, Array]:
        from ._coordinate_enclosure import axes, constant, scale, sum_polynomials

        dimension = self.topological_dimension
        variables = axes(dimension)
        arguments = tuple(
            sum_polynomials(
                (
                    constant(self.offset[row], dimension),
                    *(
                        scale(variables[column], self.matrix[row][column])
                        for column in range(dimension)
                    ),
                ),
                dimension,
            )
            for row in range(len(self.offset))
        )
        return _composed_tabulation(
            self.source_element, arguments, points, target_cell_kind=self.cell_kind
        )


class PolynomialComposedCellGeometryElement(StrictModule, NonTrainableState):
    """Retain a source basis pulled back through an exact polynomial chart."""

    source_element: ScalarCoordinateElement
    chart_coefficients: tuple[tuple[tuple[tuple[int, ...], Any], ...], ...] = eqx.field(
        static=True
    )
    cell_kind: str = eqx.field(static=True)
    conformity: str = eqx.field(static=True)
    local_dof_count: int = eqx.field(static=True)
    degree: int = eqx.field(static=True)
    topological_dimension: int = eqx.field(static=True)
    element_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_element: ScalarCoordinateElement,
        target_cell_kind: str,
        chart_arguments: tuple[Mapping[tuple[int, ...], Any], ...],
        /,
    ) -> None:
        if not isinstance(source_element, ScalarCoordinateElement):
            raise TypeError("source_element must be a scalar coordinate element.")
        target_dimension = reference_cell_topology(target_cell_kind).dimension
        source_dimension = reference_cell_topology(source_element.cell_kind).dimension
        chart = _canonical_chart_arguments(chart_arguments, target_dimension)
        if len(chart) != source_dimension:
            raise ValueError(
                "Polynomial chart outputs must match source reference dimension."
            )
        self.source_element = source_element
        self.chart_coefficients = chart
        self.cell_kind = str(target_cell_kind)
        self.conformity = "H1"
        self.local_dof_count = source_element.local_dof_count
        self.degree = source_element.degree * max(
            (sum(exponent) for polynomial in chart for exponent, _ in polynomial),
            default=1,
        )
        self.topological_dimension = target_dimension
        self.element_id = canonical_fingerprint(
            {
                "kind": "polynomial-composed-cell-geometry-element",
                "source": source_element.element_id,
                "target": self.cell_kind,
                "chart": tuple(
                    tuple(
                        (exponent, _coordinate_coefficient_identity(coefficient))
                        for exponent, coefficient in polynomial
                    )
                    for polynomial in chart
                ),
            }
        )

    @property
    def chart_arguments(self) -> tuple[dict[tuple[int, ...], Any], ...]:
        return _chart_polynomials(self.chart_coefficients)

    def tabulate(self, points: ArrayLike, /) -> tuple[Array, Array]:
        return _composed_tabulation(
            self.source_element,
            self.chart_arguments,
            points,
            target_cell_kind=self.cell_kind,
        )


class RationalComposedCellGeometryElement(StrictModule, NonTrainableState):
    """Retain a source basis pulled back through exact rational chart expressions."""

    source_element: ScalarCoordinateElement
    chart_coefficients: tuple[_ExactRationalChartTerms, ...] = eqx.field(static=True)
    cell_kind: str = eqx.field(static=True)
    conformity: str = eqx.field(static=True)
    local_dof_count: int = eqx.field(static=True)
    degree: int = eqx.field(static=True)
    topological_dimension: int = eqx.field(static=True)
    element_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_element: ScalarCoordinateElement,
        target_cell_kind: str,
        chart_arguments: tuple[RationalPolynomial, ...],
        /,
    ) -> None:
        if not isinstance(source_element, ScalarCoordinateElement):
            raise TypeError("source_element must be a scalar coordinate element.")
        target_dimension = reference_cell_topology(target_cell_kind).dimension
        source_dimension = reference_cell_topology(source_element.cell_kind).dimension
        if len(chart_arguments) != source_dimension:
            raise ValueError(
                "Rational chart outputs must match source reference dimension."
            )
        if not all(isinstance(value, RationalPolynomial) for value in chart_arguments):
            raise TypeError("Rational chart arguments must be RationalPolynomial values.")
        if any(
            len(exponent) != target_dimension
            for value in chart_arguments
            for polynomial in (value.numerator, value.denominator)
            for exponent in polynomial
        ):
            raise ValueError(
                "Rational chart terms must use the target reference dimension."
            )
        self.source_element = source_element
        self.chart_coefficients = tuple(
            (
                tuple(sorted(value.numerator.items())),
                tuple(sorted(value.denominator.items())),
            )
            for value in chart_arguments
        )
        self.cell_kind = str(target_cell_kind)
        self.conformity = "H1"
        self.local_dof_count = source_element.local_dof_count
        self.degree = source_element.degree
        self.topological_dimension = target_dimension
        self.element_id = canonical_fingerprint(
            {
                "kind": "rational-composed-cell-geometry-element",
                "source": source_element.element_id,
                "target": self.cell_kind,
                "chart_coefficients": tuple(
                    (
                        tuple(
                            (exponent, _coordinate_coefficient_identity(coefficient))
                            for exponent, coefficient in sorted(value.numerator.items())
                        ),
                        tuple(
                            (exponent, _coordinate_coefficient_identity(coefficient))
                            for exponent, coefficient in sorted(value.denominator.items())
                        ),
                    )
                    for value in chart_arguments
                ),
            }
        )

    @property
    def chart_arguments(self) -> tuple[RationalPolynomial, ...]:
        return _rational_chart_arguments_from_terms(self.chart_coefficients)

    def tabulate(self, points: ArrayLike, /) -> tuple[Array, Array]:
        return _composed_tabulation(
            self.source_element,
            self.chart_arguments,
            points,
            target_cell_kind=self.cell_kind,
        )


class SplineCellGeometryElement(StrictModule, NonTrainableState):
    """One shared-spline knot span represented by exact Bernstein extraction."""

    anchor_ids: tuple[int, ...] = eqx.field(static=True)
    degrees: tuple[int, int] = eqx.field(static=True)
    parameter_bounds: tuple[Fraction, Fraction, Fraction, Fraction] = eqx.field(
        static=True
    )
    exact_extraction_operator: tuple[tuple[Fraction, ...], ...] = eqx.field(static=True)
    weights: Array
    exact_weights: tuple[ExactScalar, ...] = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    cell_kind: str = eqx.field(static=True)
    conformity: str = eqx.field(static=True)
    local_dof_count: int = eqx.field(static=True)
    degree: int = eqx.field(static=True)
    topological_dimension: int = eqx.field(static=True)
    element_id: str = eqx.field(static=True)

    def __init__(
        self,
        anchor_ids: Sequence[int],
        parameter_bounds: tuple[Fraction, Fraction, Fraction, Fraction],
        exact_extraction_operator: Sequence[Sequence[Fraction]],
        weights: ConvertibleToArray,
        source_id: str,
        source_revision: str,
        /,
        *,
        degrees: tuple[int, int],
        exact_weights: Sequence[ExactScalar] | None = None,
    ) -> None:
        anchors = tuple(int(value) for value in anchor_ids)
        bounds = tuple(parameter_bounds)
        extraction = tuple(tuple(row) for row in exact_extraction_operator)
        weight_values = np.asarray(weights, dtype=np.float64)
        degree_pair = tuple(degrees)
        if len(degree_pair) != 2 or any(
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or int(value) < 1
            for value in degree_pair
        ):
            raise ValueError("Spline tensor degrees must be two positive integers.")
        degree_pair = int(degree_pair[0]), int(degree_pair[1])
        if (
            not anchors
            or len(set(anchors)) != len(anchors)
            or any(value < 0 for value in anchors)
            or weight_values.shape != (len(anchors),)
            or not np.all(np.isfinite(weight_values))
            or np.any(weight_values <= 0.0)
        ):
            raise ValueError("Spline span anchors and positive weights must align.")
        if exact_weights is None:
            exact_weight_values: tuple[ExactScalar, ...] = tuple(
                Fraction.from_float(float(value)) for value in weight_values
            )
        else:
            exact_values: list[ExactScalar] = []
            for value in exact_weights:
                if isinstance(value, Fraction):
                    exact_values.append(value)
                elif isinstance(value, RadicalElement):
                    exact_values.append(RadicalElement.from_stored_fields(value))
                else:
                    raise TypeError(
                        "exact_weights must contain Fraction or RadicalElement values."
                    )
            exact_weight_values = tuple(exact_values)
        if len(exact_weight_values) != len(anchors) or any(
            value <= 0 for value in exact_weight_values
        ):
            raise ValueError(
                "Exact spline weights must be positive and align with anchors."
            )
        try:
            exact_weight_shadows = np.asarray(
                [float(value) for value in exact_weight_values], dtype=np.float64
            )
        except OverflowError as error:
            raise ValueError(
                "Exact spline weights must have finite binary64 execution shadows."
            ) from error
        if not np.all(np.isfinite(exact_weight_shadows)) or not np.array_equal(
            exact_weight_shadows, weight_values
        ):
            raise ValueError(
                "Exact spline weights must match their binary64 execution shadows."
            )
        if (
            len(bounds) != 4
            or any(not isinstance(value, Fraction) for value in bounds)
            or not bounds[0] < bounds[1]
            or not bounds[2] < bounds[3]
        ):
            raise ValueError(
                "Spline parameter bounds must be exact nonempty Fraction intervals."
            )
        extraction_width = (degree_pair[0] + 1) * (degree_pair[1] + 1)
        if len(extraction) != len(anchors) or any(
            len(row) != extraction_width
            or any(not isinstance(value, Fraction) for value in row)
            for row in extraction
        ):
            raise ValueError(
                "Spline extraction width must match its exact tensor degree pair."
            )
        if not source_id or not source_revision:
            raise ValueError("Spline geometry elements require source ID and revision.")
        exact_weight_payload = tuple(
            ("rational", value.numerator, value.denominator)
            if isinstance(value, Fraction)
            else (
                "radical",
                tuple(
                    (radicand, coefficient.numerator, coefficient.denominator)
                    for radicand, coefficient in value.terms
                ),
            )
            for value in exact_weight_values
        )
        self.anchor_ids = anchors
        self.degrees = degree_pair
        self.parameter_bounds = bounds
        self.exact_extraction_operator = extraction
        self.exact_weights = exact_weight_values
        self.weights = jnp.asarray(weight_values)
        self.source_id = str(source_id)
        self.source_revision = str(source_revision)
        self.cell_kind = "quadrilateral"
        self.conformity = "H1"
        self.local_dof_count = len(anchors)
        self.degree = max(degree_pair)
        self.topological_dimension = 2
        self.element_id = canonical_fingerprint(
            {
                "kind": "shared-spline-cell-geometry-element",
                "source_id": self.source_id,
                "source_revision": self.source_revision,
                "degrees": degree_pair,
                "anchors": anchors,
                "parameter_bounds": tuple(str(value) for value in bounds),
                "extraction": tuple(
                    tuple(str(value) for value in row) for row in extraction
                ),
                "exact_weights": exact_weight_payload,
            }
        )

    def tabulate(self, points: ArrayLike, /) -> tuple[Array, Array]:
        reference = np.asarray(points, dtype=np.float64)
        if (
            reference.ndim != 2
            or reference.shape[1] != 2
            or not np.all(np.isfinite(reference))
        ):
            raise ValueError(
                "Spline chart points must have two finite reference coordinates."
            )
        u, v = reference[:, 0], reference[:, 1]

        def bernstein(value: np.ndarray, degree: int, /) -> tuple[np.ndarray, np.ndarray]:
            one_minus = 1.0 - value
            basis = np.stack(
                tuple(
                    comb(degree, index) * value**index * one_minus ** (degree - index)
                    for index in range(degree + 1)
                ),
                axis=1,
            )
            if degree == 0:
                return basis, np.zeros_like(basis)
            lower = np.stack(
                tuple(
                    comb(degree - 1, index)
                    * value**index
                    * one_minus ** (degree - 1 - index)
                    for index in range(degree)
                ),
                axis=1,
            )
            derivative = np.zeros_like(basis)
            for index in range(degree + 1):
                if index:
                    derivative[:, index] += degree * lower[:, index - 1]
                if index < degree:
                    derivative[:, index] -= degree * lower[:, index]
            return basis, derivative

        degree_u, degree_v = self.degrees
        bu, du = bernstein(u, degree_u)
        bv, dv = bernstein(v, degree_v)
        basis = (bu[:, :, None] * bv[:, None, :]).reshape(
            (reference.shape[0], (degree_u + 1) * (degree_v + 1))
        )
        derivative_u = (du[:, :, None] * bv[:, None, :]).reshape(basis.shape)
        derivative_v = (bu[:, :, None] * dv[:, None, :]).reshape(basis.shape)
        extraction = np.asarray(self.exact_extraction_operator, dtype=np.float64)
        polynomial = basis @ extraction.T
        gradients = np.stack(
            (derivative_u @ extraction.T, derivative_v @ extraction.T), axis=-1
        )
        weighted = polynomial * np.asarray(self.weights)[None, :]
        weighted_gradients = gradients * np.asarray(self.weights)[None, :, None]
        denominator = np.sum(weighted, axis=1)
        if np.any(denominator <= 0.0):
            raise ValueError("Spline rational denominator must be positive on its span.")
        denominator_gradient = np.sum(weighted_gradients, axis=1)
        values = weighted / denominator[:, None]
        rational_gradients = (
            weighted_gradients * denominator[:, None, None]
            - weighted[:, :, None] * denominator_gradient[:, None, :]
        ) / denominator[:, None, None] ** 2
        return jnp.asarray(values), jnp.asarray(rational_gradients)


class LayerColumnCellGeometryElement(StrictModule, NonTrainableState):
    """Exact profile-times-P1-station basis with authored corner corrections."""

    wall_element: FiniteElementSpec
    corner_element: FiniteElementSpec
    cell_kind: str = eqx.field(static=True)
    conformity: str = eqx.field(static=True)
    local_dof_count: int = eqx.field(static=True)
    degree: int = eqx.field(static=True)
    topological_dimension: int = eqx.field(static=True)
    station_axis: int = eqx.field(static=True)
    fiber_graph: bool = eqx.field(static=True)
    element_id: str = eqx.field(static=True)

    def __init__(
        self,
        wall_element: FiniteElementSpec,
        /,
        *,
        fiber_graph: bool = False,
    ) -> None:
        if (
            not isinstance(wall_element, FiniteElementSpec)
            or wall_element.cell_kind not in ("triangle", "quadrilateral")
            or wall_element.conformity != "H1"
            or wall_element.mapping != "identity"
            or wall_element.value_shape
        ):
            raise ValueError(
                "Layer columns require a scalar H1 triangle/quad profile element."
            )
        if not isinstance(fiber_graph, bool):
            raise TypeError("fiber_graph must be bool.")
        from .fem._reference import lagrange_element

        corners = lagrange_element(wall_element.cell_kind, 1)
        self.wall_element = wall_element
        self.corner_element = corners
        self.cell_kind = "prism" if wall_element.cell_kind == "triangle" else "hexahedron"
        self.conformity = "H1"
        self.local_dof_count = (
            2 * wall_element.local_dof_count + 4 * corners.local_dof_count
        )
        self.degree = max(wall_element.degree, 1)
        self.topological_dimension = 3
        self.station_axis = 2
        self.fiber_graph = fiber_graph
        self.element_id = canonical_fingerprint(
            {
                "kind": "layer-column-coordinate-element",
                "profile": wall_element.element_id,
                "corners": corners.element_id,
                "station_axis": 2,
                "fiber_graph": fiber_graph,
                "coefficient_banks": (
                    "bottom_profile",
                    "top_profile",
                    "actual_bottom_corners",
                    "actual_top_corners",
                    "bottom_corner_corrections",
                    "top_corner_corrections",
                ),
            }
        )

    def tabulate(self, points: ArrayLike, /) -> tuple[Array, Array]:
        reference = jnp.asarray(points, dtype=jnp.float64)
        if reference.ndim != 2 or reference.shape[1] != 3:
            raise ValueError(
                "Layer reference points must provide profile and station axes."
            )
        station = reference[:, 2:3]

        def lift(element: FiniteElementSpec) -> tuple[Array, Array, Array, Array]:
            values, gradients = element.tabulate(reference[:, :2])
            lower = values * (1.0 - station)
            upper = values * station
            lower_gradient = jnp.concatenate(
                (gradients * (1.0 - station[..., None]), -values[..., None]),
                axis=-1,
            )
            upper_gradient = jnp.concatenate(
                (gradients * station[..., None], values[..., None]),
                axis=-1,
            )
            return lower, upper, lower_gradient, upper_gradient

        bottom, top, bottom_grad, top_grad = lift(self.wall_element)
        corner_bottom, corner_top, corner_bottom_grad, corner_top_grad = lift(
            self.corner_element
        )
        return (
            jnp.concatenate(
                (
                    bottom,
                    top,
                    corner_bottom,
                    corner_top,
                    -corner_bottom,
                    -corner_top,
                ),
                axis=1,
            ),
            jnp.concatenate(
                (
                    bottom_grad,
                    top_grad,
                    corner_bottom_grad,
                    corner_top_grad,
                    -corner_bottom_grad,
                    -corner_top_grad,
                ),
                axis=1,
            ),
        )


class PerCellRestrictedCellGeometryElement(StrictModule, NonTrainableState):
    """Compose a basis through exact per-cell polynomial or rational charts.

    Rational denominators are proved nonzero on target cells. This element is
    a coordinate-function definition, not a source-cell inclusion/ancestry
    witness; those claims require validated ReferenceCellMapWitnesses.
    """

    source_element: ScalarCoordinateElement
    target_cell_ids: tuple[int, ...] = eqx.field(static=True)
    source_cell_ids: tuple[int, ...] = eqx.field(static=True)
    chart_coefficients: tuple[tuple[_ExactRationalChartTerms, ...], ...] = eqx.field(
        static=True
    )
    cell_kind: str = eqx.field(static=True)
    conformity: str = eqx.field(static=True)
    local_dof_count: int = eqx.field(static=True)
    degree: int = eqx.field(static=True)
    topological_dimension: int = eqx.field(static=True)
    element_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_element: ScalarCoordinateElement,
        target_cell_kind: str,
        target_cell_ids: Sequence[int] | np.ndarray,
        source_cell_ids: Sequence[int] | np.ndarray,
        chart_arguments: Sequence[Sequence[ExpressionInput]],
        /,
    ) -> None:
        if not isinstance(source_element, ScalarCoordinateElement):
            raise TypeError("source_element must be a scalar coordinate element.")
        targets = _canonical_cell_id_tuple(target_cell_ids, "target_cell_ids")
        sources = _canonical_cell_id_tuple(source_cell_ids, "source_cell_ids")
        target_dimension = reference_cell_topology(target_cell_kind).dimension
        source_dimension = reference_cell_topology(source_element.cell_kind).dimension
        charts = tuple(
            _canonical_rational_chart_arguments(value, target_dimension)
            for value in chart_arguments
        )
        if (
            not targets
            or len(set(targets)) != len(targets)
            or len(sources) != len(targets)
            or len(charts) != len(targets)
            or any(len(chart) != source_dimension for chart in charts)
        ):
            raise ValueError("Per-cell restriction IDs and exact charts must align.")
        zero_exponent = (0,) * target_dimension
        checked_denominators = set()
        for chart in charts:
            for _, denominator in chart:
                if (
                    len(denominator) == 1
                    and denominator[0][0] == zero_exponent
                    and denominator[0][1] == Fraction(1)
                ):
                    continue
                identity = tuple(
                    (exponent, _coordinate_coefficient_identity(coefficient))
                    for exponent, coefficient in denominator
                )
                if identity in checked_denominators:
                    continue
                checked_denominators.add(identity)
                probe = RationalPolynomial(
                    {zero_exponent: Fraction(1)}, dict(denominator)
                )
                if not reference_chart_arguments_have_nonvanishing_denominators(
                    target_cell_kind, (probe,)
                ):
                    raise ValueError(
                        "A per-cell rational chart denominator is not proved nonzero over its target cell."
                    )
        self.source_element = source_element
        self.target_cell_ids = targets
        self.source_cell_ids = sources
        self.chart_coefficients = charts
        self.cell_kind = str(target_cell_kind)
        self.conformity = "H1"
        self.local_dof_count = source_element.local_dof_count
        self.degree = int(getattr(source_element, "degree", 1))
        self.topological_dimension = target_dimension
        self.element_id = canonical_fingerprint(
            {
                "kind": "per-cell-restricted-coordinate-element",
                "source": source_element.element_id,
                "target_kind": self.cell_kind,
                "target_cells": targets,
                "source_cells": sources,
                "charts": tuple(
                    tuple(
                        (
                            tuple(
                                (exponent, _coordinate_coefficient_identity(coefficient))
                                for exponent, coefficient in numerator
                            ),
                            tuple(
                                (exponent, _coordinate_coefficient_identity(coefficient))
                                for exponent, coefficient in denominator
                            ),
                        )
                        for numerator, denominator in chart
                    )
                    for chart in charts
                ),
            }
        )

    def chart_arguments(self, cell_row: int, /) -> tuple[Expression, ...]:
        row = int(cell_row)
        if row < 0 or row >= len(self.chart_coefficients):
            raise IndexError("Per-cell restriction row is outside the target block.")
        return _per_cell_chart_arguments_from_terms(self.chart_coefficients[row])

    def tabulate_cell(self, cell_row: int, points: ArrayLike, /) -> tuple[Array, Array]:
        return _composed_tabulation(
            self.source_element,
            self.chart_arguments(cell_row),
            points,
            target_cell_kind=self.cell_kind,
        )

    def tabulate(self, points: ArrayLike, /) -> tuple[Array, Array]:
        raise ValueError(
            "Per-cell coordinate elements require tabulate_cell(cell_row, points)."
        )

    @property
    def source_element_id(self) -> str:
        return self.source_element.element_id


ScalarCoordinateElement = (
    FiniteElementSpec
    | BarycentricCellGeometryElement
    | RestrictedCellGeometryElement
    | PolynomialComposedCellGeometryElement
    | RationalComposedCellGeometryElement
    | SplineCellGeometryElement
    | LayerColumnCellGeometryElement
    | PerCellRestrictedCellGeometryElement
)


def coordinate_reference_arguments(
    element: ScalarCoordinateElement, /, *, cell_row: int | None = None
) -> tuple[Expression, ...] | None:
    if isinstance(element, PerCellRestrictedCellGeometryElement):
        if cell_row is None:
            if len(element.target_cell_ids) != 1:
                raise ValueError(
                    "Per-cell reference arguments require a local target row."
                )
            cell_row = 0
        return element.chart_arguments(cell_row)
    if isinstance(element, PolynomialComposedCellGeometryElement):
        return element.chart_arguments
    if isinstance(element, RationalComposedCellGeometryElement):
        return element.chart_arguments
    if isinstance(element, SplineCellGeometryElement):
        from ._coordinate_enclosure import axes, constant, scale, sum_polynomials

        variables = axes(2)
        u0, u1, v0, v1 = element.parameter_bounds
        return (
            sum_polynomials((constant(u0, 2), scale(variables[0], u1 - u0)), 2),
            sum_polynomials((constant(v0, 2), scale(variables[1], v1 - v0)), 2),
        )
    if isinstance(element, RestrictedCellGeometryElement):
        from ._coordinate_enclosure import axes, constant, scale, sum_polynomials

        variables = axes(element.topological_dimension)
        return tuple(
            sum_polynomials(
                (
                    constant(element.offset[row], element.topological_dimension),
                    *(
                        scale(variables[column], element.matrix[row][column])
                        for column in range(element.topological_dimension)
                    ),
                ),
                element.topological_dimension,
            )
            for row in range(len(element.offset))
        )
    if isinstance(element, BarycentricCellGeometryElement):
        from ._coordinate_enclosure import axes

        return axes(element.topological_dimension)
    return None


def coordinate_reference_chain(
    element: ScalarCoordinateElement,
    /,
    *,
    ancestor: ScalarCoordinateElement | None = None,
) -> tuple[ScalarCoordinateElement, ...]:
    chain = []
    seen: set[int] = set()
    current: ScalarCoordinateElement | None = element
    while current is not None:
        if id(current) in seen:
            raise ValueError("Coordinate source ancestry must be acyclic.")
        seen.add(id(current))
        chain.append(current)
        if ancestor is not None and current is ancestor:
            break
        current = getattr(current, "source_element", None)
    if ancestor is not None and chain[-1] is not ancestor:
        raise ValueError(
            "Requested coordinate ancestor is not in the retained source chain."
        )
    return tuple(reversed(chain))


def coordinate_tabulation(
    element: ScalarCoordinateElement,
    points: ArrayLike,
    /,
    *,
    cell_row: int | None = None,
) -> tuple[Array, Array]:
    if isinstance(element, PerCellRestrictedCellGeometryElement):
        if cell_row is None:
            if len(element.target_cell_ids) != 1:
                raise ValueError("Per-cell tabulation requires a local target row.")
            cell_row = 0
        return element.tabulate_cell(cell_row, points)
    tabulate = getattr(element, "tabulate", None)
    if not callable(tabulate):
        raise TypeError(f"{type(element).__name__} has no scalar coordinate tabulator.")
    return tabulate(points)


def _finite_element_hybrid_owner(element: FiniteElementSpec, /) -> Any | None:
    from .fem._spectral_hp_completion import HybridReferenceFamily

    family_owner = getattr(element.tabulator, "__self__", None)
    if isinstance(family_owner, FiniteElementSpec):
        family_owner = getattr(family_owner.tabulator, "__self__", None)
    if not isinstance(family_owner, HybridReferenceFamily):
        return None
    if (
        element.family != "HybridLagrange"
        or element.cell_kind != family_owner.cell_kind
        or element.degree != family_owner.degree
        or element.tabulator_id != family_owner.family_id
        or getattr(element.tabulator, "__self__", None) is not family_owner
        or element.representation != "point_value"
        or element.form_basis is not None
        or element.value_shape
        or element.mapping != "identity"
        or element.conformity != "H1"
        or element.value_spec.proxy != "scalar"
        or element.value_spec.form_type.degree != 0
        or element.local_dof_count != np.asarray(family_owner.nodes).shape[0]
        or not np.array_equal(
            np.asarray(element.reference_nodes), np.asarray(family_owner.nodes)
        )
    ):
        raise ValueError(
            "Exact Hybrid source basis requires its admitted scalar nodal family owner."
        )
    if family_owner.cell_kind == "pyramid":
        first_order, second_order = family_owner.orders
        expected_modes = tuple(
            (first, second, height)
            for first in range(first_order + 1)
            for second in range(second_order + 1)
            for height in range(family_owner.degree - max(first, second) + 1)
        )
        if (
            first_order != second_order
            or first_order != family_owner.degree
            or tuple(family_owner.modal_indices) != expected_modes
            or family_owner.coefficients.shape
            != (len(expected_modes), element.local_dof_count)
        ):
            raise ValueError(
                "Exact rational pyramid basis requires the canonical modal ordering."
            )
    return family_owner


def _hybrid_shifted_jacobi_polynomial(
    degree: int, alpha: int, /
) -> dict[tuple[int, ...], Fraction]:
    coefficients: dict[tuple[int, ...], Fraction] = {}
    for index in range(degree + 1):
        leading = Fraction(comb(degree + alpha, index) * comb(degree, degree - index))
        remaining = degree - index
        for power_ in range(remaining + 1):
            exponent = index + power_
            coefficient = leading * comb(remaining, power_) * (-1) ** (remaining - power_)
            term = (exponent,)
            coefficients[term] = coefficients.get(term, Fraction(0)) + coefficient
    return {term: value for term, value in coefficients.items() if value}


def _hybrid_pyramid_modal_basis(family_owner: Any, /) -> tuple[RationalPolynomial, ...]:
    from ._coordinate_enclosure import (
        add,
        axes,
        constant,
        multiply,
        power,
        scale,
        sum_polynomials,
    )

    x, y, z = axes(3)
    one = constant(1, 3)
    width = add(one, scale(z, Fraction(-1)))
    shifted_x = add(x, scale(z, Fraction(-1, 2)))
    shifted_y = add(y, scale(z, Fraction(-1, 2)))
    common_denominator = power(width, family_owner.degree, dimension=3)
    if isinstance(common_denominator, RationalPolynomial):
        raise RuntimeError("A positive pyramid width power must be polynomial.")

    modal_basis: list[RationalPolynomial] = []
    for first, second, height in family_owner.modal_indices:
        maximum = max(first, second)
        minimum = min(first, second)
        first_polynomial = _hybrid_shifted_jacobi_polynomial(first, 0)
        second_polynomial = _hybrid_shifted_jacobi_polynomial(second, 0)
        terms = []
        for (first_power,), first_coefficient in first_polynomial.items():
            for (second_power,), second_coefficient in second_polynomial.items():
                width_power = maximum + minimum - first_power - second_power
                if width_power < 0:
                    raise RuntimeError("Pyramid modal width exponent became negative.")
                term = multiply(
                    multiply(
                        power(shifted_x, first_power),
                        power(shifted_y, second_power),
                    ),
                    power(width, width_power),
                )
                terms.append(scale(term, first_coefficient * second_coefficient))
        numerator = sum_polynomials(tuple(terms), 3)
        height_polynomial: Polynomial = {
            (0, 0, exponent[0]): coefficient
            for exponent, coefficient in _hybrid_shifted_jacobi_polynomial(
                height, 2 * maximum + 2
            ).items()
        }
        numerator = multiply(numerator, height_polynomial)
        if isinstance(numerator, RationalPolynomial):
            raise RuntimeError("An exact pyramid modal numerator must be polynomial.")
        if family_owner.degree != minimum:
            lifted = multiply(
                numerator, power(width, family_owner.degree - minimum, dimension=3)
            )
            if isinstance(lifted, RationalPolynomial):
                raise RuntimeError("An exact pyramid modal lift must be polynomial.")
            numerator = lifted
        modal_basis.append(RationalPolynomial(numerator, common_denominator))
    return tuple(modal_basis)


def _exact_nodal_reference_points(
    element: FiniteElementSpec, /
) -> tuple[tuple[ExactScalar, ...], ...]:
    from .fem._reference import lagrange_reference_node_data

    data = lagrange_reference_node_data(element)
    if any(node.exact_point is None for node in data):
        raise ValueError(
            "Exact coordinate source basis requires reference nodes in the retained exact scalar field."
        )
    return tuple(node.exact_point for node in data if node.exact_point is not None)


def _finite_element_rational_pyramid_source_basis(
    element: FiniteElementSpec, family_owner: Any, /
) -> tuple[Expression, ...]:
    from ..linalg._exact import solve_exact
    from ._coordinate_enclosure import linear_combinations

    modal_basis = _hybrid_pyramid_modal_basis(family_owner)
    nodes = _exact_nodal_reference_points(element)
    if len(nodes) != len(modal_basis) or len(set(nodes)) != len(nodes):
        raise ValueError("Hybrid pyramid nodes and exact modal basis do not align.")
    apex = (Fraction(1, 2), Fraction(1, 2), Fraction(1))
    modal_values = []
    for point in nodes:
        row = []
        for expression in modal_basis:
            if point == apex:
                normalized = normalize_pyramid_reference_expression(expression)
                if normalized.apex_value is None:
                    raise ValueError(
                        "Canonical rational pyramid mode has no unique apex value."
                    )
                if not reference_chart_arguments_have_nonvanishing_denominators(
                    "hexahedron", (normalized.expression,)
                ):
                    raise ValueError(
                        "Canonical rational pyramid mode has an unproved residual denominator."
                    )
                row.append(normalized.apex_value)
            else:
                row.append(evaluate_expression(expression, point))
        modal_values.append(row)
    identity = [
        [Fraction(int(row == column)) for column in range(len(nodes))]
        for row in range(len(nodes))
    ]
    coefficients = solve_exact(modal_values, identity)
    nodal_coefficients = tuple(
        tuple(coefficients[mode][basis] for mode in range(len(modal_basis)))
        for basis in range(len(nodes))
    )
    return linear_combinations(modal_basis, nodal_coefficients)


def _finite_element_monomials(
    element: FiniteElementSpec, /
) -> tuple[tuple[int, ...], ...]:
    from .fem._high_order import ReferenceNodalFamily, SimplexNodalFamily

    dimension = element.topological_dimension
    hybrid_owner = _finite_element_hybrid_owner(element)
    if hybrid_owner is not None:
        if hybrid_owner.cell_kind != "prism":
            raise ValueError(
                "Rational pyramid source basis requires its canonical modal owner."
            )
        triangle_degree, station_degree = tuple(
            int(value) for value in hybrid_owner.orders
        )
        return tuple(
            (i, j, k)
            for i, j in product(range(triangle_degree + 1), repeat=2)
            if i + j <= triangle_degree
            for k in range(station_degree + 1)
        )
    family_owner = getattr(element.tabulator, "__self__", None)
    if isinstance(family_owner, FiniteElementSpec):
        family_owner = getattr(family_owner.tabulator, "__self__", None)
    if isinstance(family_owner, ReferenceNodalFamily):
        orders = tuple(int(value) for value in family_owner.orders)
        return tuple(
            tuple(int(value) for value in exponent)
            for exponent in product(*(range(order + 1) for order in orders))
        )
    # Preserve FEEC polynomial-span exponents; source coefficients come from its
    # exact FormBasis bank.
    form_basis = element.form_basis
    if form_basis is not None:
        from .fem._form_elements import _ProxyTabulator, FormBasis

        tabulator = element.tabulator
        value_form = element.value_spec.form_type
        if (
            not isinstance(form_basis, FormBasis)
            or not isinstance(tabulator, _ProxyTabulator)
            or tabulator.basis.basis_id != form_basis.basis_id
            or element.tabulator_id != form_basis.basis_id
            or tabulator.value_spec.form_type != value_form
            or tabulator.value_spec.proxy != element.value_spec.proxy
            or form_basis.dimension != dimension
            or form_basis.form_degree != 0
            or form_basis.order != element.degree
            or form_basis.family != element.family
            or form_basis.twist != value_form.twist
            or value_form.degree != 0
            or element.value_spec.proxy != "scalar"
            or element.representation != "polynomial_moment"
            or not np.array_equal(
                np.asarray(element.reference_nodes),
                np.asarray(form_basis.reference_nodes),
            )
        ):
            raise ValueError(
                "Scalar coordinate basis requires its canonical zero-form FormBasis."
            )
        exponents = tuple(
            tuple(int(value) for value in exponent) for exponent in form_basis.exponents
        )
        if (
            len(exponents) != element.local_dof_count
            or len(set(exponents)) != len(exponents)
            or any(
                len(exponent) != dimension or any(value < 0 for value in exponent)
                for exponent in exponents
            )
        ):
            raise ValueError(
                "Canonical zero-form FormBasis does not expose a complete scalar monomial basis."
            )
        return exponents
    if element.family not in (
        "Lagrange",
        "DiscontinuousLagrange",
        "SimplexLagrange",
        "TensorProductLagrange",
    ):
        raise ValueError(
            f"Finite-element family {element.family!r} has no canonical nodal polynomial basis."
        )
    if element.tabulator is not None and not isinstance(
        family_owner, (SimplexNodalFamily, ReferenceNodalFamily)
    ):
        if not (
            element.family == "DiscontinuousLagrange"
            and isinstance(
                getattr(element.tabulator, "__self__", None), FiniteElementSpec
            )
            and str(element.tabulator_id).startswith("discontinuous:")
        ):
            raise ValueError(
                "Opaque/custom coordinate tabulators have no exact source basis."
            )
    cell = element.cell_kind
    if cell in ("interval", "triangle", "tetrahedron") or cell.startswith("simplex:"):
        return tuple(
            tuple(int(value) for value in exponent)
            for exponent in product(
                *(range(element.degree + 1) for _ in range(dimension))
            )
            if sum(exponent) <= element.degree
        )
    if cell in ("quadrilateral", "hexahedron") or cell.startswith("tensor:"):
        return tuple(
            tuple(int(value) for value in exponent)
            for exponent in product(
                *(range(element.degree + 1) for _ in range(dimension))
            )
        )
    raise ValueError(f"Exact nodal source basis is unsupported on {cell!r}.")


def _finite_element_form_basis_source_basis(
    element: FiniteElementSpec, /
) -> tuple[Expression, ...]:
    exponents = _finite_element_monomials(element)
    form_basis = element.form_basis
    if form_basis is None:
        raise ValueError("Exact FormBasis source construction requires its owner.")
    coefficients = form_basis.exact_coefficients
    if (
        len(coefficients) != len(exponents)
        or any(len(row) != 1 for row in coefficients)
        or any(len(row[0]) != element.local_dof_count for row in coefficients)
    ):
        raise ValueError("Exact scalar FormBasis coefficients have incompatible extents.")
    return tuple(
        {
            exponent: coefficients[monomial][0][dof]
            for monomial, exponent in enumerate(exponents)
            if coefficients[monomial][0][dof]
        }
        for dof in range(element.local_dof_count)
    )


def _finite_element_source_basis(element: FiniteElementSpec, /) -> tuple[Expression, ...]:
    if element.form_basis is not None:
        return _finite_element_form_basis_source_basis(element)
    if element.representation != "point_value":
        raise ValueError(
            "Exact coordinate source basis requires scalar nodal or canonical FormBasis functionals."
        )
    family_owner = _finite_element_hybrid_owner(element)
    if family_owner is not None and family_owner.cell_kind == "pyramid":
        return _finite_element_rational_pyramid_source_basis(element, family_owner)
    exponents = _finite_element_monomials(element)
    if len(exponents) != element.local_dof_count:
        raise ValueError(
            "Declared nodal element does not have a complete polynomial basis."
        )
    nodes = _exact_nodal_reference_points(element)

    def monomial(point: tuple[ExactScalar, ...], powers: tuple[int, ...]) -> ExactScalar:
        value: ExactScalar = Fraction(1)
        for coordinate, exponent in zip(point, powers, strict=True):
            value *= coordinate**exponent
        return value

    vandermonde = [
        [monomial(point, exponent) for exponent in exponents] for point in nodes
    ]
    identity = [
        [Fraction(int(row == column)) for column in range(len(nodes))]
        for row in range(len(nodes))
    ]
    from ..linalg._exact import solve_exact

    coefficients = solve_exact(vandermonde, identity)
    return tuple(
        {
            exponent: coefficients[monomial_index][basis_index]
            for monomial_index, exponent in enumerate(exponents)
            if coefficients[monomial_index][basis_index]
        }
        for basis_index in range(len(nodes))
    )


# Root source bases keyed by element type and content identity; each is formed
# once per process.
_SOURCE_BASES: dict[tuple[str, str], tuple[Any, ...]] = {}


def coordinate_source_basis(
    element: ScalarCoordinateElement, /, *, cell_row: int | None = None
) -> tuple[Any, ...]:
    """Return exact root basis expressions without applying reference charts/actions.

    The basis depends only on the element's content identity, so it is formed
    once per element; every caller receives its own polynomial dictionaries.
    """
    if isinstance(element, PerCellRestrictedCellGeometryElement):
        if cell_row is not None and not 0 <= int(cell_row) < len(element.target_cell_ids):
            raise IndexError("Per-cell source basis row is outside the target block.")
        return coordinate_source_basis(element.source_element)
    if isinstance(
        element,
        BarycentricCellGeometryElement
        | RestrictedCellGeometryElement
        | PolynomialComposedCellGeometryElement
        | RationalComposedCellGeometryElement,
    ):
        return coordinate_source_basis(element.source_element)
    identity = getattr(element, "element_id", None)
    if not isinstance(identity, str):
        return _root_source_basis(element)
    key = (type(element).__name__, identity)
    basis = _SOURCE_BASES.get(key)
    if basis is None:
        basis = _root_source_basis(element)
        _SOURCE_BASES[key] = basis
    return tuple(dict(value) if isinstance(value, dict) else value for value in basis)


def _root_source_basis(element: ScalarCoordinateElement, /) -> tuple[Any, ...]:
    from ._coordinate_enclosure import (
        add,
        axes,
        constant,
        linear_combinations,
        multiply,
        power,
        scale,
        sum_polynomials,
    )

    if isinstance(element, SplineCellGeometryElement):
        variables = axes(2)
        one = constant(Fraction(1), 2)
        basis_1d = []
        for axis, degree in zip(variables, element.degrees, strict=True):
            complement = add(one, scale(axis, -1))
            basis_1d.append(
                tuple(
                    scale(
                        multiply(power(axis, index), power(complement, degree - index)),
                        Fraction(comb(degree, index)),
                    )
                    for index in range(degree + 1)
                )
            )
        degree_u, degree_v = element.degrees
        bernstein = tuple(
            multiply(basis_1d[0][u], basis_1d[1][v])
            for u in range(degree_u + 1)
            for v in range(degree_v + 1)
        )
        extracted = linear_combinations(bernstein, element.exact_extraction_operator)
        weights = element.exact_weights
        denominator = sum_polynomials(
            tuple(
                scale(polynomial, weight)
                for polynomial, weight in zip(extracted, weights, strict=True)
            ),
            2,
        )
        return tuple(
            multiply(scale(polynomial, weight), power(denominator, -1))
            for polynomial, weight in zip(extracted, weights, strict=True)
        )
    if isinstance(element, LayerColumnCellGeometryElement):
        variables = axes(3)
        profile = tuple(
            compose_expression(value, variables[:2])
            for value in coordinate_source_basis(element.wall_element)
        )
        corner = tuple(
            compose_expression(value, variables[:2])
            for value in coordinate_source_basis(element.corner_element)
        )
        one = constant(Fraction(1), 3)
        lower = add(one, scale(variables[2], -1))
        upper = variables[2]
        return tuple(
            [*(multiply(value, lower) for value in profile)]
            + [*(multiply(value, upper) for value in profile)]
            + [*(multiply(value, lower) for value in corner)]
            + [*(multiply(value, upper) for value in corner)]
            + [*(scale(multiply(value, lower), -1) for value in corner)]
            + [*(scale(multiply(value, upper), -1) for value in corner)]
        )
    if not isinstance(element, FiniteElementSpec):
        raise TypeError("Coordinate source basis does not recognize this scalar element.")
    if element.value_shape or element.mapping != "identity" or element.conformity != "H1":
        raise ValueError(
            "Coordinate source basis requires scalar identity-mapped H1 elements."
        )
    if isinstance(element.tabulator, SweptCoordinateTabulator):
        variables = axes(element.topological_dimension)
        profile = tuple(
            compose_expression(value, variables[:2])
            for value in coordinate_source_basis(element.tabulator.source)
        )
        one = constant(Fraction(1), element.topological_dimension)
        lower = add(one, scale(variables[2], -1))
        upper = variables[2]
        return tuple(
            [*(multiply(polynomial, lower) for polynomial in profile)]
            + [*(multiply(polynomial, upper) for polynomial in profile)]
        )
    return _finite_element_source_basis(element)


def coordinate_effective_basis(
    element: ScalarCoordinateElement, /, *, cell_row: int | None = None
) -> tuple[Any, ...]:
    """Compose exact reference charts and coefficient actions into the root basis."""
    from ._coordinate_enclosure import linear_combinations

    if isinstance(element, BarycentricCellGeometryElement):
        basis = coordinate_effective_basis(element.source_element)
        coefficients = tuple(
            tuple(
                element.exact_barycentric_weights[source_row][target_column]
                for source_row in range(len(basis))
            )
            for target_column in range(element.local_dof_count)
        )
        return linear_combinations(basis, coefficients)
    if isinstance(element, PerCellRestrictedCellGeometryElement):
        if cell_row is None:
            if len(element.target_cell_ids) != 1:
                raise ValueError(
                    "Per-cell basis composition requires a local target row."
                )
            cell_row = 0
        basis = coordinate_effective_basis(element.source_element)
        arguments = element.chart_arguments(cell_row)
        return tuple(compose_expression(value, arguments) for value in basis)
    if isinstance(
        element,
        RestrictedCellGeometryElement
        | PolynomialComposedCellGeometryElement
        | RationalComposedCellGeometryElement,
    ):
        basis = coordinate_effective_basis(element.source_element)
        arguments = coordinate_reference_arguments(element, cell_row=cell_row)
        if arguments is None:
            raise ValueError("Composed coordinate element must retain its exact chart.")
        return tuple(compose_expression(value, arguments) for value in basis)
    return coordinate_source_basis(element)


def _binary64_shadow(value: Any, /) -> Fraction:
    """Represent the runtime binary64 coefficient as an exact rational."""
    try:
        rounded = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(
            "Element coefficients require finite binary64 shadows."
        ) from error
    if not np.isfinite(rounded):
        raise ValueError("Element coefficients require finite binary64 shadows.")
    return Fraction.from_float(rounded)


def _binary64_polynomial(value: Mapping[tuple[int, ...], ExactScalar], /) -> Polynomial:
    return {
        exponent: shadow
        for exponent, coefficient in value.items()
        if (shadow := _binary64_shadow(coefficient))
    }


def _binary64_expression(value: ExpressionInput, /) -> Expression:
    if not isinstance(value, RationalPolynomial):
        return _binary64_polynomial(value)
    numerator = _binary64_polynomial(value.numerator)
    denominator = _binary64_polynomial(value.denominator)
    if not denominator:
        raise ValueError("Execution chart denominator has no finite coefficients.")
    return RationalPolynomial(numerator, denominator)


def _constant_denominator_polynomial(
    value: Expression, dimension: int, /
) -> Polynomial | None:
    if not isinstance(value, RationalPolynomial):
        return value
    zero = (0,) * dimension
    if set(value.denominator) != {zero}:
        return None
    coefficient = value.denominator[zero]
    if not coefficient:
        raise ValueError("Execution expression has a zero constant denominator.")
    return {
        exponent: term / coefficient for exponent, term in value.numerator.items() if term
    }


def _execution_lagrange_axis(
    nodes: ArrayLike, weights: ArrayLike, /
) -> tuple[Expression, ...]:
    from ._coordinate_enclosure import axes

    nodes_ = tuple(_binary64_shadow(value) for value in np.asarray(nodes).reshape((-1,)))
    weights_ = tuple(
        _binary64_shadow(value) for value in np.asarray(weights).reshape((-1,))
    )
    if not nodes_ or len(nodes_) != len(weights_):
        raise ValueError("Native nodal basis nodes and weights must align.")
    axis = axes(1)[0]
    one = constant(1, 1)
    factors: list[Polynomial] = []
    for omitted in range(len(nodes_)):
        factor: Expression = one
        for index, node in enumerate(nodes_):
            if index != omitted:
                factor = multiply(factor, add(axis, constant(-node, 1)))
        if isinstance(factor, RationalPolynomial):
            raise RuntimeError("A nodal product must be polynomial.")
        factors.append(factor)
    denominator = sum_polynomials(
        tuple(
            scale(factor, weight)
            for factor, weight in zip(factors, weights_, strict=True)
        ),
        1,
    )
    if isinstance(denominator, RationalPolynomial) or not denominator:
        raise ValueError("Native nodal basis has an identically zero denominator.")
    result: list[Expression] = []
    for factor, weight in zip(factors, weights_, strict=True):
        numerator = scale(factor, weight)
        expression = RationalPolynomial(numerator, denominator)
        polynomial = _constant_denominator_polynomial(expression, 1)
        result.append(expression if polynomial is None else polynomial)
    return tuple(result)


def _reference_nodal_execution_basis(owner: Any, /) -> tuple[Expression, ...]:
    from ._coordinate_enclosure import axes

    dimension = len(owner.orders)
    variables = axes(dimension)
    factors = tuple(
        _execution_lagrange_axis(nodes, weights)
        for nodes, weights in zip(
            owner.nodes_by_axis,
            owner.barycentric_weights_by_axis,
            strict=True,
        )
    )
    axis_factors = tuple(
        tuple(compose_expression(value, (variables[axis],)) for value in factors[axis])
        for axis in range(dimension)
    )
    basis: list[Expression] = []
    for index in product(*(range(order + 1) for order in owner.orders)):
        value: Expression = constant(1, dimension)
        for axis in range(dimension):
            value = multiply(value, axis_factors[axis][index[axis]])
        basis.append(value)
    return tuple(basis)


def _simplex_nodal_execution_basis(owner: Any, /) -> tuple[Expression, ...]:
    from ._coordinate_enclosure import axes

    dimension = len(owner.multiindices[0]) - 1
    variables = axes(dimension)
    barycentric = (
        add(constant(1, dimension), scale(sum_polynomials(variables, dimension), -1)),
        *variables,
    )
    from math import factorial

    modal: list[Polynomial] = []
    for indices in owner.multiindices:
        multinomial = _binary64_shadow(
            factorial(owner.order)
            / np.prod(tuple(factorial(int(index)) for index in indices))
        )
        value: Expression = scale(constant(1, dimension), multinomial)
        for coordinate, exponent in zip(barycentric, indices, strict=True):
            value = multiply(value, polynomial_power(coordinate, int(exponent)))
        if isinstance(value, RationalPolynomial):
            raise RuntimeError("A simplex modal basis must be polynomial.")
        modal.append(value)
    coefficients = np.asarray(owner.coefficients, dtype=np.float64)
    if coefficients.shape != (len(modal), len(modal)):
        raise ValueError("Native simplex coefficient table has incompatible dimensions.")
    rows = tuple(
        tuple(_binary64_shadow(coefficients[mode, dof]) for mode in range(len(modal)))
        for dof in range(len(modal))
    )
    return linear_combinations(modal, rows)


def _form_tensor_execution_basis(form_basis: Any, /) -> tuple[Expression, ...]:
    from ._coordinate_enclosure import axes

    factors = form_basis.tensor_factors
    if factors is None or form_basis.form_degree != 0:
        raise ValueError(
            "Tensor coordinate elements require the scalar native FormBasis."
        )
    variables = axes(form_basis.dimension)
    row_count = factors.zero.shape[0]
    degree = row_count - 1
    zero_coefficients = np.asarray(factors.zero, dtype=np.float64)

    def legendre(order: int) -> Polynomial:
        if order == 0:
            return constant(1, 1)
        x = axes(1)[0]
        previous = constant(1, 1)
        current = add(scale(x, 2), constant(-1, 1))
        for level in range(2, order + 1):
            following = scale(
                add(
                    scale(
                        multiply(add(scale(x, 2), constant(-1, 1)), current),
                        2 * level - 1,
                    ),
                    scale(previous, -(level - 1)),
                ),
                Fraction(1, level),
            )
            if isinstance(following, RationalPolynomial):
                raise RuntimeError("Legendre recurrence must remain polynomial.")
            previous, current = current, following
        return current

    zero_factors: list[tuple[Polynomial, ...]] = []
    for axis in range(form_basis.dimension):
        x = axes(1)[0]
        complement = add(constant(1, 1), scale(x, -1))
        bubble = multiply(x, complement)
        modes = tuple(legendre(index) for index in range(max(degree - 1, 0)))
        local: list[Polynomial] = []
        for dof in range(degree + 1):
            if dof == 0:
                value: Expression = add(constant(1, 1), scale(x, -1))
            elif dof == degree:
                value = x
            else:
                value = constant(0, 1)
            for mode, polynomial in enumerate(modes):
                value = add(
                    value,
                    scale(
                        multiply(bubble, polynomial),
                        _binary64_shadow(zero_coefficients[mode + 2, dof]),
                    ),
                )
            if isinstance(value, RationalPolynomial):
                raise RuntimeError("A tensor modal factor must be polynomial.")
            local.append(value)
        zero_factors.append(tuple(local))
    axis_factors = tuple(
        tuple(
            compose_expression(value, (variables[axis],)) for value in zero_factors[axis]
        )
        for axis in range(form_basis.dimension)
    )
    basis: list[Expression] = []
    for _, index, _ in form_basis.dof_labels:
        value: Expression = constant(1, form_basis.dimension)
        for axis, dof in enumerate(index):
            value = multiply(value, axis_factors[axis][dof])
        basis.append(value)
    return tuple(basis)


def _finite_element_execution_basis(
    element: FiniteElementSpec, /
) -> tuple[Expression, ...]:
    if element.value_shape or element.mapping != "identity" or element.conformity != "H1":
        raise ValueError("Execution basis requires a scalar identity-mapped H1 element.")
    tabulator = element.tabulator
    if isinstance(tabulator, SweptCoordinateTabulator):
        variables = axes(element.topological_dimension)
        profile = tuple(
            compose_expression(value, variables[:2])
            for value in coordinate_execution_basis(tabulator.source)
        )
        one = constant(1, element.topological_dimension)
        lower = add(one, scale(variables[2], -1))
        upper = variables[2]
        return tuple(
            [*(multiply(value, lower) for value in profile)]
            + [*(multiply(value, upper) for value in profile)]
        )
    hybrid_owner = _finite_element_hybrid_owner(element)
    if hybrid_owner is not None:
        if hybrid_owner.cell_kind == "pyramid":
            modal = _hybrid_pyramid_modal_basis(hybrid_owner)
            coefficients = np.asarray(hybrid_owner.coefficients, dtype=np.float64)
            if coefficients.shape != (len(modal), element.local_dof_count):
                raise ValueError(
                    "Native pyramid coefficient table has incompatible dimensions."
                )
            rows = tuple(
                tuple(
                    _binary64_shadow(coefficients[mode, dof])
                    for mode in range(len(modal))
                )
                for dof in range(element.local_dof_count)
            )
            return linear_combinations(modal, rows)
        from .._polynomial._orthogonal import legendre_rule_data
        from .fem._high_order import (
            _default_barycentric_weights,
            SimplexNodalFamily,
        )

        triangle_degree, axial_degree = hybrid_owner.orders
        triangle = SimplexNodalFamily("triangle", triangle_degree).finite_element()
        horizontal = coordinate_execution_basis(triangle)
        rule = legendre_rule_data(axial_degree + 1, "lobatto")
        z_nodes = 0.5 * (jnp.asarray(rule.nodes) + 1.0)
        z_weights = _default_barycentric_weights(z_nodes)
        vertical = _execution_lagrange_axis(z_nodes, z_weights)
        variables = axes(3)
        vertical = tuple(compose_expression(value, (variables[2],)) for value in vertical)
        product_basis = tuple(
            multiply(
                compose_expression(value, variables[:2]),
                vertical[axial],
            )
            for value in horizontal
            for axial in range(len(vertical))
        )
        return tuple(product_basis[index] for index in hybrid_owner.basis_permutation)
    if element.form_basis is not None:
        _finite_element_monomials(element)
        form_basis = element.form_basis
        if form_basis.tensor_factors is not None:
            basis = _form_tensor_execution_basis(form_basis)
        else:
            monomials: list[Polynomial] = []
            for exponent in form_basis.exponents:
                value: Expression = constant(1, form_basis.dimension)
                for variable, power_ in zip(
                    axes(form_basis.dimension), exponent, strict=True
                ):
                    value = multiply(value, polynomial_power(variable, int(power_)))
                if isinstance(value, RationalPolynomial):
                    raise RuntimeError("Finite-element monomials must be polynomial.")
                monomials.append(value)
            coefficients = np.asarray(form_basis.coefficients, dtype=np.float64)
            if coefficients.shape != (
                len(monomials),
                1,
                element.local_dof_count,
            ):
                raise ValueError(
                    "Native scalar FormBasis coefficient table is incompatible."
                )
            rows = tuple(
                tuple(
                    _binary64_shadow(coefficients[mode, 0, dof])
                    for mode in range(len(monomials))
                )
                for dof in range(element.local_dof_count)
            )
            basis = linear_combinations(monomials, rows)
        return basis
    owner = getattr(tabulator, "__self__", None)
    if tabulator is not None and isinstance(owner, FiniteElementSpec):
        if (
            element.family == "DiscontinuousLagrange"
            and element.tabulator_id == f"discontinuous:{owner.element_id}"
        ):
            return coordinate_execution_basis(owner)
    if tabulator is not None:
        from .fem._high_order import ReferenceNodalFamily, SimplexNodalFamily

        if isinstance(owner, ReferenceNodalFamily):
            _finite_element_monomials(element)
            return _reference_nodal_execution_basis(owner)
        if isinstance(owner, SimplexNodalFamily):
            _finite_element_monomials(element)
            return _simplex_nodal_execution_basis(owner)
        raise ValueError("Finite-element tabulator has no canonical coefficient owner.")
    if tabulator is None:
        if element.family == "DiscontinuousLagrange" and element.degree == 0:
            return (constant(1, element.topological_dimension),)
        if (
            element.cell_kind,
            element.degree,
        ) in (
            ("triangle", 1),
            ("triangle", 2),
            ("quadrilateral", 1),
            ("tetrahedron", 1),
            ("hexahedron", 1),
        ):
            from .fem._reference import lagrange_element

            return coordinate_source_basis(
                lagrange_element(element.cell_kind, element.degree)
            )
        raise ValueError("Finite-element spec has no implemented native tabulator.")


def coordinate_execution_source_basis(
    element: ScalarCoordinateElement, /, *, cell_row: int | None = None
) -> tuple[Expression, ...]:
    """Return the native coefficient-shadow basis before reference composition."""
    if isinstance(
        element,
        PerCellRestrictedCellGeometryElement
        | BarycentricCellGeometryElement
        | RestrictedCellGeometryElement
        | PolynomialComposedCellGeometryElement
        | RationalComposedCellGeometryElement,
    ):
        return coordinate_execution_source_basis(element.source_element)
    if isinstance(element, SplineCellGeometryElement):
        variables = axes(2)
        one = constant(1, 2)
        basis_1d = []
        for axis, degree in zip(variables, element.degrees, strict=True):
            complement = add(one, scale(axis, -1))
            basis_1d.append(
                tuple(
                    scale(
                        multiply(
                            polynomial_power(axis, index),
                            polynomial_power(complement, degree - index),
                        ),
                        Fraction(comb(degree, index)),
                    )
                    for index in range(degree + 1)
                )
            )
        degree_u, degree_v = element.degrees
        bernstein = tuple(
            multiply(basis_1d[0][u], basis_1d[1][v])
            for u in range(degree_u + 1)
            for v in range(degree_v + 1)
        )
        extraction = tuple(
            tuple(_binary64_shadow(value) for value in row)
            for row in element.exact_extraction_operator
        )
        extracted = linear_combinations(bernstein, extraction)
        weights = tuple(_binary64_shadow(value) for value in np.asarray(element.weights))
        denominator = sum_polynomials(
            tuple(
                scale(polynomial, weight)
                for polynomial, weight in zip(extracted, weights, strict=True)
            ),
            2,
        )
        return tuple(
            multiply(
                scale(polynomial, weight),
                polynomial_power(denominator, -1, dimension=2),
            )
            for polynomial, weight in zip(extracted, weights, strict=True)
        )
    if isinstance(element, LayerColumnCellGeometryElement):
        variables = axes(3)
        profile = tuple(
            compose_expression(value, variables[:2])
            for value in coordinate_execution_basis(element.wall_element)
        )
        corner = tuple(
            compose_expression(value, variables[:2])
            for value in coordinate_execution_basis(element.corner_element)
        )
        one = constant(1, 3)
        lower = add(one, scale(variables[2], -1))
        upper = variables[2]
        return tuple(
            [*(multiply(value, lower) for value in profile)]
            + [*(multiply(value, upper) for value in profile)]
            + [*(multiply(value, lower) for value in corner)]
            + [*(multiply(value, upper) for value in corner)]
            + [*(scale(multiply(value, lower), -1) for value in corner)]
            + [*(scale(multiply(value, upper), -1) for value in corner)]
        )
    if isinstance(element, FiniteElementSpec):
        return _finite_element_execution_basis(element)
    raise TypeError("Coordinate execution basis does not recognize this scalar element.")


def _execution_reference_arguments(
    element: ScalarCoordinateElement, /, *, cell_row: int | None = None
) -> tuple[Expression, ...] | None:
    arguments = coordinate_reference_arguments(element, cell_row=cell_row)
    if arguments is None:
        return None
    return tuple(_binary64_expression(value) for value in arguments)


def coordinate_execution_basis(
    element: ScalarCoordinateElement, /, *, cell_row: int | None = None
) -> tuple[Expression, ...]:
    """Compose runtime binary64 coefficient shadows through the effective map."""
    if isinstance(element, BarycentricCellGeometryElement):
        basis = coordinate_execution_basis(element.source_element)
        weights = np.asarray(element.barycentric_weights, dtype=np.float64)
        coefficients = tuple(
            tuple(
                _binary64_shadow(weights[source, target]) for source in range(len(basis))
            )
            for target in range(element.local_dof_count)
        )
        return linear_combinations(basis, coefficients)
    if isinstance(element, PerCellRestrictedCellGeometryElement):
        if cell_row is None:
            if len(element.target_cell_ids) != 1:
                raise ValueError("Per-cell execution basis requires a local target row.")
            cell_row = 0
        basis = coordinate_execution_basis(element.source_element)
        arguments = _execution_reference_arguments(element, cell_row=cell_row)
        if arguments is None:
            raise ValueError("Per-cell coordinate element must retain its exact chart.")
        return tuple(compose_expression(value, arguments) for value in basis)
    if isinstance(
        element,
        RestrictedCellGeometryElement
        | PolynomialComposedCellGeometryElement
        | RationalComposedCellGeometryElement,
    ):
        basis = coordinate_execution_basis(element.source_element)
        arguments = _execution_reference_arguments(element, cell_row=cell_row)
        if arguments is None:
            raise ValueError("Composed coordinate element must retain its exact chart.")
        return tuple(compose_expression(value, arguments) for value in basis)
    return coordinate_execution_source_basis(element, cell_row=cell_row)


def restriction_chart_arguments(
    element: ScalarCoordinateElement,
    /,
    *,
    ancestor: ScalarCoordinateElement | None = None,
    cell_row: int | None = None,
) -> tuple[Expression, ...] | None:
    chain = coordinate_reference_chain(element, ancestor=ancestor)
    arguments: tuple[Expression, ...] | None = None
    for step in chain[1:]:
        local = coordinate_reference_arguments(step, cell_row=cell_row)
        if local is None:
            continue
        if arguments is None:
            arguments = local
        else:
            arguments = tuple(
                compose_expression(argument, local) for argument in arguments
            )
    return arguments


class CellVertexGeometryElement(StrictModule, NonTrainableState):
    """Vertex-coordinate geometry descriptor for a variable-topology cell block."""

    cell_kind: str = eqx.field(static=True)
    conformity: str = eqx.field(static=True)
    local_dof_count: int = eqx.field(static=True)
    element_id: str = eqx.field(static=True)

    def __init__(self, cell_kind: str, local_dof_count: int, /) -> None:
        kind = str(cell_kind)
        count = int(local_dof_count)
        if kind not in ("polygon", "polyhedron"):
            raise ValueError(
                "CellVertexGeometryElement supports polygon or polyhedron blocks."
            )
        if count < (3 if kind == "polygon" else 4):
            raise ValueError("Variable-topology geometry requires enough vertices.")
        self.cell_kind = kind
        self.conformity = "H1"
        self.local_dof_count = count
        self.element_id = canonical_fingerprint(
            {
                "kind": "cell-vertex-geometry-element",
                "cell_kind": kind,
                "local_dof_count": count,
            }
        )


class CellGeometrySpec(StrictModule, NonTrainableState):
    """Per-block coordinate elements, routes, values, and source-map identity."""

    block_names: tuple[str, ...] = eqx.field(static=True)
    elements: tuple[CellGeometryElement, ...]
    geometry_dofs: tuple[Array, ...]
    coordinates: Array
    exact_coordinate_bank: CoordinateSourceBank | None = eqx.field(static=True)
    source_map: CellGeometryCoordinateMap | None
    restriction_source: CellGeometryRestrictionSource | None
    source_mesh_id: str | None = eqx.field(static=True)
    source_topology_id: str | None = eqx.field(static=True)
    source_map_id: str | None = eqx.field(static=True)
    geometry_layout_id: str = eqx.field(static=True)

    def __init__(
        self,
        elements: Mapping[str, CellGeometryElement],
        geometry_dofs: Mapping[str, ArrayLike],
        coordinates: ArrayLike,
        /,
        *,
        source_map: CellGeometryCoordinateMap | None = None,
        source_mesh: CellMesh | None = None,
        restriction_source: CellGeometryRestrictionSource | None = None,
    ) -> None:
        items = tuple(sorted((str(name), element) for name, element in elements.items()))
        routes = {
            str(name): np.asarray(value, dtype=np.int32)
            for name, value in geometry_dofs.items()
        }
        points = np.asarray(coordinates, dtype=np.float64)
        if not items or any(not name for name, _ in items):
            raise ValueError("Coordinate element mapping must be non-empty.")
        if not all(isinstance(element, CellGeometryElement) for _, element in items):
            raise TypeError("Coordinate elements must satisfy CellGeometryElement.")
        if any(element.conformity != "H1" for _, element in items):
            raise ValueError("Coordinate elements must be H1-conforming.")
        if set(routes) != {name for name, _ in items}:
            raise ValueError(
                "Coordinate DOF routes must match coordinate element blocks."
            )
        if points.ndim != 2 or not np.all(np.isfinite(points)):
            raise ValueError("Coordinate values must be one finite rank-2 array.")
        normalized_routes = []
        for name, element in items:
            route = routes[name]
            if route.ndim != 2 or route.shape[1] != element.local_dof_count:
                raise ValueError(
                    "Coordinate DOF route width must match its coordinate element."
                )
            if np.any(route < 0) or np.any(route >= points.shape[0]):
                raise ValueError("Coordinate DOF routes index undeclared coordinates.")
            normalized_routes.append(jnp.asarray(route))
        exact_coordinate_bank = None
        if source_map is None:
            if source_mesh is not None:
                raise ValueError(
                    "A source mesh binding requires a source coordinate map."
                )
            source_mesh_id = source_topology_id = source_map_id = None
        else:
            if not isinstance(source_map, CellGeometryCoordinateMap):
                raise TypeError("source_map must satisfy CellGeometryCoordinateMap.")
            if source_mesh is None or not isinstance(source_mesh, CellMesh):
                raise ValueError(
                    "A source coordinate map must bind an exact target mesh."
                )
            if not all(
                isinstance(value, str) and value
                for value in (
                    source_map.source_id,
                    source_map.source_revision,
                    source_map.source_map_id,
                )
            ):
                raise ValueError("Source coordinate maps require full source identities.")
            rounded_source_vertices = getattr(source_map, "rounded_vertices", None)
            if rounded_source_vertices is not None:
                rounded = np.asarray(rounded_source_vertices, dtype=np.float64)
                target_coordinates = np.asarray(source_mesh.coordinates)
                if rounded.shape != target_coordinates.shape or not np.array_equal(
                    rounded, target_coordinates
                ):
                    raise ValueError(
                        "Source map vertex view must match its bound target mesh."
                    )
            source_mesh_id = source_mesh.mesh_id
            source_topology_id = source_mesh.topology_id
            source_map_id = source_map.source_map_id
            exact_coordinate_bank = _capture_exact_coordinate_bank(source_map, points)
        if restriction_source is not None:
            if not isinstance(restriction_source, CellGeometryRestrictionSource):
                raise TypeError(
                    "restriction_source must be CellGeometryRestrictionSource."
                )
            if restriction_source.block_names != tuple(name for name, _ in items):
                raise ValueError("Restriction source blocks must match geometry blocks.")
            element_lookup = dict(items)
            for name, route, reference_map in zip(
                restriction_source.block_names,
                restriction_source.parent_cell_ids,
                restriction_source.parent_reference_maps,
                strict=True,
            ):
                if np.asarray(route).shape != (routes[name].shape[0],):
                    raise ValueError(
                        "Restriction parent cells must align with target block rows."
                    )
                if reference_map is not None and (
                    reference_map.target_cell_kind != element_lookup[name].cell_kind
                    or np.asarray(reference_map.target_cell_ids).shape
                    != (routes[name].shape[0],)
                ):
                    raise ValueError(
                        "Restriction reference maps must align with their target coordinate block."
                    )
        restriction_source_id = (
            None
            if restriction_source is None
            else restriction_source.restriction_source_id
        )
        self.block_names = tuple(name for name, _ in items)
        self.elements = tuple(element for _, element in items)
        self.geometry_dofs = tuple(normalized_routes)
        self.coordinates = jnp.asarray(points)
        self.exact_coordinate_bank = exact_coordinate_bank
        self.source_map = source_map
        self.restriction_source = restriction_source
        self.source_mesh_id = source_mesh_id
        self.source_topology_id = source_topology_id
        self.source_map_id = source_map_id
        self.geometry_layout_id = canonical_fingerprint(
            {
                "kind": "cell-geometry-spec",
                "blocks": [[name, element.element_id] for name, element in items],
                "geometry_dofs": [
                    array_tree_fingerprint(np.asarray(value))
                    for value in normalized_routes
                ],
                "coordinate_shape": list(points.shape),
                "source_map": source_map_id,
                "restriction_source": restriction_source_id,
            }
        )

    @classmethod
    def from_stored_fields(
        cls,
        value: CellGeometrySpec,
        /,
        *,
        source_mesh: CellMesh | None,
        source_map: CellGeometryCoordinateMap | None,
        restriction_source: CellGeometryRestrictionSource | None,
        elements: Mapping[str, Any],
    ) -> CellGeometrySpec:
        """Renew source owners and decode canonical coordinate-element recipes."""
        from .._model._structure import (
            _FiniteElementSpecArchiveProjection,
            model_archive_projection,
        )
        from ..geometry._mesh_certificates import _same_tree_content
        from ._exact_power_geometry import ExactPowerCellGeometrySource

        if type(value) is not cls:
            raise TypeError("value must be CellGeometrySpec.")
        if not isinstance(elements, Mapping) or set(elements) != set(value.block_names):
            raise ValueError(
                "Restored coordinate elements must match stored block names."
            )
        if len(value.geometry_dofs) != len(value.block_names):
            raise ValueError("Stored coordinate routes must match stored block names.")
        restored_elements: dict[str, CellGeometryElement] = {}
        for name in value.block_names:
            element = elements[name]
            if type(element) is _FiniteElementSpecArchiveProjection:
                element = element.rebuild()
            if not isinstance(element, CellGeometryElement):
                raise TypeError(
                    "Stored coordinate element must satisfy CellGeometryElement."
                )
            restored_elements[name] = element
        rebuilt = cls(
            restored_elements,
            dict(zip(value.block_names, value.geometry_dofs, strict=True)),
            value.coordinates,
            source_map=source_map,
            source_mesh=source_mesh,
            restriction_source=restriction_source,
        )
        if (
            type(source_map) is ExactPowerCellGeometrySource
            and source_map.periodic_source_context is not None
        ):
            if not isinstance(source_mesh, CellMesh):
                raise ValueError(
                    "Periodic exact power geometry requires its current target mesh."
                )
            source_map.require_periodic_mesh(source_mesh)

        def bank_payload(bank: CoordinateSourceBank | None, /) -> object:
            if bank is None:
                return None
            return tuple(
                tuple(coefficient_payload(coefficient) for coefficient in row)
                for row in bank
            )

        if bank_payload(rebuilt.exact_coordinate_bank) != bank_payload(
            value.exact_coordinate_bank
        ):
            raise ValueError(
                "Stored exact coordinate snapshot differs from its renewed source map."
            )
        if (
            rebuilt.block_names != value.block_names
            or rebuilt.geometry_layout_id != value.geometry_layout_id
            or rebuilt.source_mesh_id != value.source_mesh_id
            or rebuilt.source_topology_id != value.source_topology_id
            or rebuilt.source_map_id != value.source_map_id
            or not _same_tree_content(
                model_archive_projection(value), model_archive_projection(rebuilt)
            )
        ):
            raise ValueError("Stored CellGeometrySpec differs from reconstructed owners.")
        return rebuilt

    @classmethod
    def affine(cls, mesh: CellMesh, /) -> CellGeometrySpec:
        from .fem._form_elements import form_element
        from .fem._reference import lagrange_element

        elements: dict[str, CellGeometryElement] = {}
        for block in mesh.blocks:
            if block.cell_kind in ("polygon", "polyhedron"):
                element = CellVertexGeometryElement(block.cell_kind, block.arity)
            elif ":" in block.cell_kind:
                family = (
                    "tensor-trimmed"
                    if block.cell_kind.startswith("tensor:")
                    else "trimmed"
                )
                element = form_element(
                    block.cell_kind,
                    0,
                    1,
                    family=family,
                    twist="untwisted",
                    proxy="scalar",
                )
            else:
                element = lagrange_element(block.cell_kind, 1)
            elements[block.name] = element
        return cls(
            elements,
            {block.name: block.vertices for block in mesh.blocks},
            mesh.coordinates,
        )

    @classmethod
    def power(
        cls,
        mesh: CellMesh,
        source_map: CellGeometryCoordinateMap,
        /,
    ) -> CellGeometrySpec:
        """Bind an exact power-cell map to its rounded mesh realization."""
        if not isinstance(mesh, CellMesh):
            raise TypeError("mesh must be CellMesh.")
        if not isinstance(source_map, CellGeometryCoordinateMap):
            raise TypeError("source_map must satisfy CellGeometryCoordinateMap.")
        from ._exact_power_geometry import ExactPowerCellGeometrySource

        if (
            type(source_map) is ExactPowerCellGeometrySource
            and source_map.periodic_source_context is not None
        ):
            source_map.require_periodic_mesh(mesh)
        rounded = getattr(source_map, "rounded_vertices", None)
        if rounded is None or not np.array_equal(
            np.asarray(rounded, dtype=np.float64), np.asarray(mesh.coordinates)
        ):
            raise ValueError(
                "Power source-map vertices must exactly match the mesh execution view."
            )
        base = cls.affine(mesh)
        return cls(
            dict(zip(base.block_names, base.elements, strict=True)),
            {
                name: np.asarray(route)
                for name, route in zip(base.block_names, base.geometry_dofs, strict=True)
            },
            mesh.coordinates,
            source_map=source_map,
            source_mesh=mesh,
        )

    def source_coordinates(
        self, indices: ArrayLike | None = None, /
    ) -> CoordinateSourceBank:
        """Index the admitted immutable exact bank, or lift ordinary execution values."""
        values = np.asarray(self.coordinates, dtype=np.float64)
        if indices is None:
            rows = np.arange(values.shape[0], dtype=np.int64)
        else:
            selection = np.asarray(indices)
            if selection.dtype.kind not in "iu" or selection.ndim != 1:
                raise TypeError("Coordinate source indices must be an integer vector.")
            rows = selection.astype(np.int64)
            if np.any(rows < 0) or np.any(rows >= values.shape[0]):
                raise IndexError(
                    "Coordinate source index lies outside the coefficient bank."
                )
        if self.exact_coordinate_bank is not None:
            bank = self.exact_coordinate_bank
            selected = bank if indices is None else tuple(bank[row] for row in rows)
            expected = values if indices is None else values[rows]
            if selected and not np.array_equal(
                np.asarray(selected, dtype=np.float64), expected
            ):
                raise ValueError(
                    "Exact source coordinates no longer match their execution shadows."
                )
            return selected
        return tuple(
            tuple(Fraction(float(value)) for value in values[row]) for row in rows
        )

    def rational_source_coordinates(
        self, indices: ArrayLike | None = None, /
    ) -> tuple[tuple[Fraction, ...], ...]:
        from ._exact_algebraic_geometry import rational_coefficient

        return tuple(
            tuple(
                rational_coefficient(
                    value, role="CellGeometrySpec.rational_source_coordinates"
                )
                for value in row
            )
            for row in self.source_coordinates(indices)
        )

    def evaluate_exact(
        self,
        mesh: CellMesh,
        cell_global_ids: ArrayLike,
        reference_points: tuple[tuple[tuple[ExactScalar, ...], ...], ...],
        /,
    ) -> tuple[tuple[tuple[ExactScalar, ...], ...], ...]:
        """Evaluate exact polynomial/rational source maps at exact reference points."""
        from ._coordinate_enclosure import (
            linear_combinations,
            RationalPolynomial,
        )
        from ._exact_algebraic_geometry import RadicalElement

        elements, routes, _ = self.resolve(mesh)
        identifiers = np.asarray(cell_global_ids)
        if identifiers.ndim != 1 or identifiers.dtype.kind not in "iu":
            raise TypeError("cell_global_ids must be an integer vector.")
        if len(reference_points) != identifiers.size:
            raise ValueError("Exact reference point rows must align with cell IDs.")
        source_bank = self.source_coordinates()
        row_by_cell: dict[int, tuple[int, int]] = {}
        for block_index, block in enumerate(mesh.blocks):
            for local_row, identifier in enumerate(
                np.asarray(block.global_ids, dtype=np.int64)
            ):
                row_by_cell[int(identifier)] = (block_index, local_row)

        def evaluate(
            expression: Expression,
            point: tuple[ExactScalar, ...],
            /,
            *,
            pyramid_target: bool = False,
        ) -> ExactScalar:
            def polynomial_value(
                polynomial: Mapping[tuple[int, ...], ExactScalar],
            ) -> ExactScalar:
                total: ExactScalar = Fraction(0)
                for exponent, coefficient in polynomial.items():
                    term: ExactScalar = coefficient
                    for coordinate, power in zip(point, exponent, strict=True):
                        term *= coordinate**power
                    total = total + term
                return total

            if isinstance(expression, RationalPolynomial):
                denominator = polynomial_value(expression.denominator)
                if not denominator:
                    apex = (Fraction(1, 2), Fraction(1, 2), Fraction(1))
                    if not pyramid_target or point != apex:
                        raise ValueError("Exact coordinate map denominator vanishes.")
                    normalized = normalize_pyramid_reference_expression(expression)
                    if normalized.apex_value is None:
                        raise ValueError(
                            "Exact coordinate map has no unique pyramid apex value."
                        )
                    if not reference_chart_arguments_have_nonvanishing_denominators(
                        "hexahedron", (normalized.expression,)
                    ):
                        raise ValueError(
                            "Exact coordinate map has an unproved residual pyramid denominator."
                        )
                    return normalized.apex_value
                return polynomial_value(expression.numerator) / denominator
            return polynomial_value(expression)

        results = []
        for identifier, point_rows in zip(
            identifiers.tolist(), reference_points, strict=True
        ):
            owner = row_by_cell.get(int(identifier))
            if owner is None:
                raise ValueError(
                    f"Exact coordinate map references absent cell ID {identifier}."
                )
            block_index, local_row = owner
            element = elements[block_index]
            if not isinstance(element, ScalarCoordinateElement):
                raise ValueError(
                    "Exact reference evaluation requires a scalar coordinate map."
                )
            pyramid_target = reference_cell_topology(element.cell_kind).name == "pyramid"
            route = np.asarray(routes[block_index], dtype=np.int64)[local_row]
            coefficients = tuple(source_bank[int(index)] for index in route)
            basis = coordinate_effective_basis(element, cell_row=local_row)
            if len(basis) != len(coefficients):
                raise ValueError(
                    "Exact coordinate basis and local coefficient bank disagree."
                )
            coordinate_expressions = linear_combinations(
                basis,
                tuple(
                    tuple(coefficients[dof][axis] for dof in range(len(coefficients)))
                    for axis in range(len(coefficients[0]))
                ),
            )
            cell_results = []
            for point in point_rows:
                if any(
                    not isinstance(value, (Fraction, RadicalElement)) for value in point
                ):
                    raise TypeError(
                        "Exact reference points must use Fraction or RadicalElement scalars."
                    )
                exact_point = tuple(point)
                cell_results.append(
                    tuple(
                        evaluate(expression, exact_point, pyramid_target=pyramid_target)
                        for expression in coordinate_expressions
                    )
                )
            results.append(tuple(cell_results))
        return tuple(results)

    def resolve(
        self,
        mesh: CellMesh,
        /,
    ) -> tuple[tuple[CellGeometryElement, ...], tuple[Array, ...], Array]:
        if self.source_mesh_id is not None and (
            mesh.mesh_id != self.source_mesh_id
            or mesh.topology_id != self.source_topology_id
        ):
            raise ValueError("Coordinate source map is bound to a different mesh epoch.")
        mapping = dict(zip(self.block_names, self.elements, strict=True))
        routes = dict(zip(self.block_names, self.geometry_dofs, strict=True))
        if set(mapping) != {block.name for block in mesh.blocks}:
            raise ValueError("Coordinate element assignments must match mesh blocks.")
        resolved = tuple(mapping[block.name] for block in mesh.blocks)
        resolved_routes = tuple(routes[block.name] for block in mesh.blocks)
        for block, element, route in zip(
            mesh.blocks,
            resolved,
            resolved_routes,
            strict=True,
        ):
            if block.cell_kind != element.cell_kind:
                raise ValueError("Coordinate element cell kind does not match its block.")
            if route.shape[0] != block.cell_count:
                raise ValueError("Coordinate DOF routes require one row per cell.")
            if isinstance(element, PerCellRestrictedCellGeometryElement) and (
                tuple(np.asarray(block.global_ids, dtype=np.int64).tolist())
                != element.target_cell_ids
            ):
                raise ValueError(
                    "Per-cell charts must align with target cell global-ID order."
                )
        return resolved, resolved_routes, self.coordinates

    def evaluate(
        self,
        mesh: CellMesh,
        cell_global_ids: ArrayLike,
        reference_points: ArrayLike,
        /,
    ) -> np.ndarray:
        """Evaluate the represented physical map at explicit cell/reference pairs."""
        elements, routes, coordinates = self.resolve(mesh)
        identifiers = np.asarray(cell_global_ids)
        references = np.asarray(reference_points, dtype=np.float64)
        if identifiers.ndim != 1 or identifiers.dtype.kind not in "iu":
            raise TypeError("cell_global_ids must be an integer vector.")
        if references.ndim == 2:
            references = references[:, None, :]
        if (
            references.ndim != 3
            or references.shape[0] != identifiers.size
            or not np.all(np.isfinite(references))
        ):
            raise ValueError("reference_points must align with requested cells.")
        cell_rows: dict[int, tuple[int, int]] = {}
        for block_index, block in enumerate(mesh.blocks):
            for row, identifier in enumerate(
                np.asarray(block.global_ids, dtype=np.int64)
            ):
                cell_rows[int(identifier)] = (block_index, row)
        result = np.empty(
            (identifiers.size, references.shape[1], mesh.ambient_dimension),
            dtype=np.float64,
        )
        point_values = np.asarray(coordinates, dtype=np.float64)
        for output_row, identifier in enumerate(identifiers.tolist()):
            owner = cell_rows.get(int(identifier))
            if owner is None:
                raise ValueError(
                    f"Coordinate map references absent cell ID {identifier}."
                )
            block_index, cell_row = owner
            element = elements[block_index]
            if isinstance(element, RationalComposedCellGeometryElement):
                values = _composed_values(
                    element.source_element,
                    element.chart_arguments,
                    references[output_row],
                    target_cell_kind=element.cell_kind,
                )
            elif isinstance(element, PerCellRestrictedCellGeometryElement):
                values = _composed_values(
                    element.source_element,
                    element.chart_arguments(cell_row),
                    references[output_row],
                    target_cell_kind=element.cell_kind,
                )
            else:
                tabulate_cell = getattr(element, "tabulate_cell", None)
                if callable(tabulate_cell):
                    values, _ = tabulate_cell(cell_row, references[output_row])
                else:
                    tabulate = getattr(element, "tabulate", None)
                    if not callable(tabulate):
                        raise ValueError(
                            f"Coordinate element {type(element).__name__} has no reference evaluator."
                        )
                    values, _ = tabulate(references[output_row])
            basis = np.asarray(values, dtype=np.float64)
            dofs = np.asarray(routes[block_index], dtype=np.int64)[cell_row]
            if basis.shape != (references.shape[1], dofs.size):
                raise ValueError(
                    "Coordinate basis evaluation returned incompatible shape."
                )
            result[output_row] = basis @ point_values[dofs]
        if not np.all(np.isfinite(result)):
            raise ValueError(
                "Coordinate map evaluation returned nonfinite physical points."
            )
        return result


__all__ = [
    "BarycentricCellGeometryElement",
    "CellGeometryCoordinateMap",
    "CellGeometryElement",
    "CellGeometryRestrictionSource",
    "CellGeometrySpec",
    "CellVertexGeometryElement",
    "LayerColumnCellGeometryElement",
    "SplineCellGeometryElement",
    "PolynomialComposedCellGeometryElement",
    "RationalComposedCellGeometryElement",
    "PerCellRestrictedCellGeometryElement",
    "RestrictedCellGeometryElement",
    "ScalarCoordinateElement",
    "SweptCoordinateTabulator",
    "coordinate_reference_arguments",
    "coordinate_reference_chain",
    "coordinate_source_basis",
    "coordinate_effective_basis",
    "coordinate_execution_basis",
    "coordinate_execution_source_basis",
    "coordinate_tabulation",
    "restriction_chart_arguments",
    "swept_coordinate_element",
]
