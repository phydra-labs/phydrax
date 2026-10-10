#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Scalar field actions on a certified, occurrence-local exact material deformation.

Exact material charts express correspondence, not physical area. H1 applies the
old field's point functionals through the exact material reference relation; DG
integrates paired exact physical-reference pieces against old or new geometry.
No vector/Piola compatibility is implied by this route.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from fractions import Fraction
from typing import Literal, TYPE_CHECKING, TypedDict

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike
from numpy.typing import NDArray

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import (
    ArraySpace,
    DenseLinearOperator,
    FactorizationPolicy,
    factorize,
    OperatorProperties,
)
from ...sparse import RowRelation, SparseLinearMap
from .._cell_geometry import CellGeometrySpec
from .._cell_geometry_transfer import (
    _embedded_squared_density,
    _mapped_geometry_cells,
    SourceGeometryRealization,
)
from .._cell_geometry_validity import cell_geometry_id
from .._cell_mesh import CellMesh
from .._coordinate_enclosure import (
    axes,
    compose,
    constant,
    Expression,
    expression_compose,
    expression_scale,
    multiply,
    Polynomial,
    scale,
    source_basis,
    sum_polynomials,
)
from .._transfer import TransferGeometryBinding
from ._generic import FiniteElementDiscretization
from ._reference import FiniteElementSpec
from ._topology_transfer import (
    _bound_coordinate_spec,
    _coalesced_map,
    _mapped_dg_cells,
    _MappedDgCell,
    _measure_defect_bound,
    _owned_rows,
    _tabulated,
    FiniteElementFieldTransfer,
    FiniteElementTopologyTransfer,
    FiniteElementTransferEvidence,
)


if TYPE_CHECKING:
    from .._sphere_chart_deformation import PreparedSphereChartDeformation
    from .._surface_chart_deformation import (
        PreparedSurfaceChartDeformation,
        PreparedSurfaceChartOccurrence,
    )
    from ..finite_volume._unstructured import UnstructuredFiniteVolumeDiscretization

type _ChartEndpoint = FiniteElementDiscretization | UnstructuredFiniteVolumeDiscretization
type _ChartDeformation = PreparedSurfaceChartDeformation | PreparedSphereChartDeformation
type _FloatArray = NDArray[np.float64]
type _IntArray = NDArray[np.int64]
type _ExactArray = NDArray[np.object_]


class _IntegrationBudgets(TypedDict):
    absolute_tolerance: float
    relative_tolerance: float
    maximum_work: int
    maximum_subcells: int
    maximum_binomial_terms: int


def _exact_material_reference_points(
    corners: _ExactArray,
    points: _ExactArray,
) -> _ExactArray:
    """Invert one exact triangle chart with a bounded serial 2x2 action."""
    from ...linalg._hermitian_spectral import _reserve_fraction_work
    from .._coordinate_enclosure import _COORDINATE_BUDGET

    if corners.shape != (3, 2) or points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Exact material reference inversion requires triangle points.")
    inputs = tuple(
        value if isinstance(value, Fraction) else Fraction(value)
        for value in (*corners.flat, *points.flat)
    )
    bits = max(
        max(abs(value.numerator).bit_length(), value.denominator.bit_length())
        for value in inputs
    )
    count = points.shape[0]
    _reserve_fraction_work(
        _COORDINATE_BUDGET.get(),
        3 + 8 * count,
        13 + 2 * count,
        8 * bits + 8,
    )
    origin_x, origin_y = inputs[0], inputs[1]
    a = inputs[2] - origin_x
    c = inputs[3] - origin_y
    b = inputs[4] - origin_x
    d = inputs[5] - origin_y
    determinant = a * d - b * c
    if determinant == 0:
        raise ValueError("An exact material chart has a singular reference action.")
    result = []
    for row in range(count):
        x = inputs[6 + 2 * row] - origin_x
        y = inputs[7 + 2 * row] - origin_y
        result.append(((d * x - b * y) / determinant, (a * y - c * x) / determinant))
    return np.asarray(result, dtype=object)


def _inside_material(corners: _ExactArray, point: _ExactArray) -> bool:
    def orientation(
        first: _ExactArray, second: _ExactArray, third: _ExactArray
    ) -> Fraction:
        return (second[0] - first[0]) * (third[1] - first[1]) - (second[1] - first[1]) * (
            third[0] - first[0]
        )

    sign = orientation(*corners)
    if sign == 0:
        raise ValueError("An exact material chart has a degenerate triangle.")
    return all(
        orientation(corners[index], corners[(index + 1) % 3], point) * sign >= 0
        for index in range(3)
    )


def _bound_surface_endpoint(
    discretization: _ChartEndpoint,
    geometry: CellGeometrySpec,
    role: Literal["source", "target"],
    prepared: _ChartDeformation,
) -> None:
    if isinstance(discretization, FiniteElementDiscretization):
        _bound_coordinate_spec(discretization, geometry, role)
        return
    from ..finite_volume._unstructured import UnstructuredFiniteVolumeDiscretization

    if not isinstance(discretization, UnstructuredFiniteVolumeDiscretization):
        raise TypeError(
            "Surface chart endpoints require actual prepared FE or unstructured FV discretizations."
        )
    retained = discretization.cell_geometry
    if retained is None or cell_geometry_id(retained) != cell_geometry_id(geometry):
        raise ValueError(
            f"The {role} chart geometry differs from the actual prepared FV map."
        )
    ids = np.concatenate(
        [
            np.asarray(block.global_ids, dtype=np.int64)
            for block in discretization.mesh.blocks
        ]
    )
    if not np.array_equal(np.asarray(discretization.cell_global_ids), ids):
        raise ValueError(
            f"The {role} FV state rows do not bind the mesh's scientific cells."
        )
    volumes = np.asarray(discretization.cell_volumes, dtype=np.float64)
    errors = np.asarray(discretization.cell_volume_error_bounds, dtype=np.float64)
    if (
        volumes.shape != ids.shape
        or errors.shape != ids.shape
        or not np.all(np.isfinite(volumes))
        or not np.all(np.isfinite(errors))
        or np.any(errors < 0)
        or np.any(volumes <= errors)
    ):
        raise ValueError(f"The {role} FV physical cell-measure certificate is invalid.")
    original_mesh = prepared.source_mesh if role == "source" else prepared.target_mesh
    measures = (
        prepared.source_cell_measures
        if role == "source"
        else prepared.target_cell_measures
    )
    measure_errors = (
        prepared.source_measure_errors
        if role == "source"
        else prepared.target_measure_errors
    )
    order = _physical_rows(discretization, original_mesh)
    expected = np.asarray(measures, dtype=np.float64)[np.argsort(order)]
    expected_errors = np.asarray(measure_errors, dtype=np.float64)[np.argsort(order)]
    # Independent certified integrations may publish different midpoints. Their
    # exact dyadic intervals must overlap; the FV's own measures remain authoritative
    # for its cell-average pairing and inventory ledger.
    if any(
        abs(Fraction(float(actual)) - Fraction(float(reference)))
        > Fraction(float(error)) + Fraction(float(reference_error))
        for actual, reference, error, reference_error in zip(
            volumes, expected, errors, expected_errors, strict=True
        )
    ):
        raise ValueError(
            f"The {role} FV cell measures disagree with the certified chart geometry."
        )


def _endpoint_cell_measures(
    discretization: _ChartEndpoint,
    prepared: _ChartDeformation,
    order: _IntArray,
    role: Literal["source", "target"],
) -> tuple[_FloatArray, _FloatArray]:
    if isinstance(discretization, FiniteElementDiscretization):
        measures = (
            prepared.source_cell_measures
            if role == "source"
            else prepared.target_cell_measures
        )
        errors = (
            prepared.source_measure_errors
            if role == "source"
            else prepared.target_measure_errors
        )
        return (
            np.asarray(measures, dtype=np.float64)[np.argsort(order)],
            np.asarray(errors, dtype=np.float64)[np.argsort(order)],
        )
    return (
        np.asarray(discretization.cell_volumes, dtype=np.float64),
        np.asarray(discretization.cell_volume_error_bounds, dtype=np.float64),
    )


def _validate(
    source: _ChartEndpoint,
    target: _ChartEndpoint,
    prepared: PreparedSurfaceChartDeformation,
    source_geometry: CellGeometrySpec,
    target_geometry: CellGeometrySpec,
) -> tuple[_IntArray, _IntArray]:
    from .._surface_chart_deformation import PreparedSurfaceChartDeformation

    if not isinstance(prepared, PreparedSurfaceChartDeformation):
        raise TypeError("A prepared surface chart deformation is required.")
    _bound_surface_endpoint(source, source_geometry, "source", prepared)
    _bound_surface_endpoint(target, target_geometry, "target", prepared)
    prepared.require_bound(source.mesh, source_geometry, target.mesh, target_geometry)
    if (
        source.mesh.topology_id != prepared.source_topology_id
        or target.mesh.topology_id != prepared.target_topology_id
        or cell_geometry_id(source_geometry) != prepared.source_geometry_id
        or cell_geometry_id(target_geometry) != prepared.target_geometry_id
    ):
        raise ValueError(
            "Surface chart transfer has stale physical geometry/topology endpoints."
        )
    for discretization, witness in (
        (source, prepared.source_witness),
        (target, prepared.target_witness),
    ):
        ids = np.concatenate(
            [
                np.asarray(block.global_ids, dtype=np.int64)
                for block in discretization.mesh.blocks
            ]
        )
        # Scientific identities select physical rows; the witness order is explicit.
        if len(set(ids.tolist())) != ids.size or set(ids.tolist()) != set(
            np.asarray(witness.cell_global_ids, dtype=np.int64).tolist()
        ):
            raise ValueError(
                "Surface chart witness does not bind the current scientific cells."
            )
    return (
        _physical_rows(source, prepared.source_mesh),
        _physical_rows(target, prepared.target_mesh),
    )


def _physical_rows(discretization: _ChartEndpoint, original_mesh: CellMesh) -> _IntArray:
    ids = np.concatenate(
        [
            np.asarray(block.global_ids, dtype=np.int64)
            for block in discretization.mesh.blocks
        ]
    )
    index = {int(value): row for row, value in enumerate(ids)}
    original_ids = np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in original_mesh.blocks]
    )
    return np.asarray([index[int(value)] for value in original_ids], dtype=np.int64)


def _field_cells(
    discretization: FiniteElementDiscretization, field_index: int
) -> tuple[tuple[FiniteElementSpec, _IntArray], ...]:
    dofs = discretization.dof_maps[field_index]
    cells = []
    for element, routes, signs in zip(
        discretization.elements[field_index],
        dofs.cell_dofs,
        dofs.orientations,
        strict=True,
    ):
        if (
            element.cell_kind != "triangle"
            or element.value_shape
            or element.mapping != "identity"
            or element.representation != "point_value"
        ):
            raise ValueError(
                "Surface chart actions require scalar triangle point-value fields."
            )
        for route, sign in zip(np.asarray(routes), np.asarray(signs), strict=True):
            if np.any(sign != 1):
                raise ValueError("Scalar surface fields require canonical orientation.")
            cells.append((element, np.asarray(route, dtype=np.int64)))
    return tuple(cells)


def _binding(
    prepared: _ChartDeformation | SourceGeometryRealization,
) -> TransferGeometryBinding:
    if isinstance(prepared, SourceGeometryRealization):
        transition = prepared.transition
        return TransferGeometryBinding(
            transition.source_geometry_id,
            transition.target_geometry_id,
            "source-realization",
            source_topology_id=transition.source_topology_id,
            target_topology_id=transition.target_topology_id,
            coverage_defect=None,
        )
    return TransferGeometryBinding(
        prepared.source_geometry_id,
        prepared.target_geometry_id,
        "bounded-reconstruction",
        source_topology_id=prepared.source_topology_id,
        target_topology_id=prepared.target_topology_id,
    )


def _prepare_h1(
    source: FiniteElementDiscretization,
    target: FiniteElementDiscretization,
    prepared: PreparedSurfaceChartDeformation,
    source_order: _IntArray,
    target_order: _IntArray,
    name: str,
) -> FiniteElementFieldTransfer:
    si, ti = source._field_index(name), target._field_index(name)
    sc, tc = _field_cells(source, si), _field_cells(target, ti)
    if any(
        element.conformity != "H1" or source_basis(element) is None
        for element, _ in sc + tc
    ):
        raise ValueError(
            "Surface chart H1 requires actual canonical scalar nodal sources."
        )
    candidates, columns, values = [], [], []
    for occurrence in prepared.occurrences:
        source_triangles = np.asarray(
            occurrence.exact_source_material_charts, dtype=object
        )
        target_triangles = np.asarray(
            occurrence.exact_target_material_charts, dtype=object
        )
        original_ids = np.concatenate(
            [
                np.asarray(block.global_ids, dtype=np.int64)
                for block in prepared.source_mesh.blocks
            ]
        )
        source_ids = original_ids[np.asarray(occurrence.source_rows)]
        source_scan = np.argsort(source_ids, kind="stable")
        for uv_row, physical in enumerate(np.asarray(occurrence.target_rows)):
            element, route = tc[int(target_order[int(physical)])]
            local = np.asarray(
                [
                    [Fraction(float(value)) for value in node]
                    for node in np.asarray(element.reference_nodes)
                ],
                dtype=object,
            )
            points = target_triangles[uv_row, 0] + local @ (
                target_triangles[uv_row, 1:] - target_triangles[uv_row, 0]
            )
            for node, point in enumerate(points):
                found = False
                for old_row in source_scan:
                    if not _inside_material(source_triangles[old_row], point):
                        continue
                    old_physical = int(
                        source_order[int(np.asarray(occurrence.source_rows)[old_row])]
                    )
                    old_element, old_route = sc[old_physical]
                    reference = _exact_material_reference_points(
                        source_triangles[old_row], point[None]
                    )
                    basis, _ = _tabulated(
                        old_element, np.asarray(reference, dtype=np.float64)
                    )
                    candidates.append(int(route[node]))
                    columns.append(old_route)
                    values.append(basis[0])
                    found = True
                    # Retain every boundary candidate: shared trace rows must agree.
                if not found:
                    raise ValueError(
                        "Target H1 node is outside the complete exact material cover."
                    )
    width = max(element.local_dof_count for element, _ in sc)
    indices = np.zeros((len(candidates), width), dtype=np.int64)
    coefficients = np.zeros(indices.shape, dtype=np.float64)
    for row, (route, value) in enumerate(zip(columns, values, strict=True)):
        indices[row, : route.size], coefficients[row, : route.size] = route, value
    ss, ts = source.dof_maps[si].global_dof_count, target.dof_maps[ti].global_dof_count
    rows, coefficients, continuity = _owned_rows(
        np.asarray(candidates, dtype=np.int64)[:, None],
        indices,
        coefficients[:, None],
        ts,
        ss,
    )
    amplification = max(1.0, width * float(np.max(np.sum(np.abs(coefficients), axis=1))))
    tolerance = 64 * np.finfo(np.float64).eps * amplification
    evidence = FiniteElementTransferEvidence(
        {
            "continuity": continuity,
            "constants": float(np.max(np.abs(coefficients.sum(axis=1) - 1))),
        },
        tolerance,
    )
    if not evidence.passed:
        raise ValueError("Surface chart H1 nodal/trace certificate failed.")
    primal = SparseLinearMap(
        RowRelation(rows.astype(np.int32), source_size=ss),
        coefficients,
        operator_id=canonical_fingerprint(
            {
                "kind": "surface-chart-h1",
                "field": name,
                "source": source.prepared_id,
                "target": target.prepared_id,
                "deformation": prepared.deformation_id,
                "source_geometry": prepared.source_geometry_id,
                "target_geometry": prepared.target_geometry_id,
                "routes": array_tree_fingerprint(rows),
                "values": array_tree_fingerprint(coefficients),
            }
        ),
    )
    transfer = FiniteElementTopologyTransfer(
        primal,
        prepared.source_topology_id,
        prepared.target_topology_id,
        preserves_constants=True,
        positivity_preserving=bool(np.all(coefficients >= 0)),
        semantics="interpolation",
        action_condition=amplification,
    )
    return FiniteElementFieldTransfer(
        transfer,
        source.field_spaces[si],
        target.field_spaces[ti],
        _binding(prepared),
        evidence,
    )


def _integral(
    gram: Expression, weight: Expression, budgets: _IntegrationBudgets
) -> tuple[float, float]:
    from .._cell_geometry_transfer import _certified_sqrt_polynomial_integral

    return _certified_sqrt_polynomial_integral(gram, weight, "triangle", **budgets)


def _piece_maps(
    occurrence: PreparedSurfaceChartOccurrence,
) -> Iterator[tuple[int, int, int, _ExactArray, _ExactArray]]:
    for entry, piece in enumerate(occurrence.pieces):
        yield (
            entry,
            piece.source_row,
            piece.target_row,
            np.asarray(piece.exact_source_reference_vertices, dtype=object),
            np.asarray(piece.exact_target_reference_vertices, dtype=object),
        )


def _restricted(
    gram: Expression, reference: _ExactArray
) -> tuple[Expression, tuple[Polynomial, ...]]:
    matrix = (reference[1:] - reference[0]).T
    determinant = matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]
    if determinant == 0:
        raise ValueError(
            "An exact material piece has a degenerate physical-reference map."
        )
    variables = axes(2)
    arguments = tuple(
        sum_polynomials(
            (constant(origin, 2),)
            + tuple(
                scale(variable, value)
                for variable, value in zip(variables, row, strict=True)
            )
        )
        for origin, row in zip(reference[0], matrix, strict=True)
    )
    return expression_scale(
        expression_compose(gram, arguments), determinant**2
    ), arguments


def _column_content(
    rows: _IntArray,
    columns: _IntArray,
    coefficients: _FloatArray,
    source_measures: _FloatArray,
    target_measures: _FloatArray,
) -> list[Fraction]:
    result = [-Fraction(float(value)) for value in source_measures]
    for row, column, value in zip(rows, columns, coefficients, strict=True):
        result[int(column)] += Fraction(float(value)) * Fraction(
            float(target_measures[int(row)])
        )
    return result


def _prepare_dg(
    source: FiniteElementDiscretization,
    target: FiniteElementDiscretization,
    prepared: PreparedSurfaceChartDeformation,
    source_geometry: CellGeometrySpec,
    target_geometry: CellGeometrySpec,
    source_order: _IntArray,
    target_order: _IntArray,
    name: str,
    semantics: Literal["intensive", "conservative-density"],
    budgets: _IntegrationBudgets,
) -> FiniteElementFieldTransfer:
    si, ti = source._field_index(name), target._field_index(name)
    sc, tc = _mapped_dg_cells(source, si), _mapped_dg_cells(target, ti)
    sg, tg = (
        _mapped_geometry_cells(source.mesh, source_geometry),
        _mapped_geometry_cells(target.mesh, target_geometry),
    )
    grams = (
        tuple(_embedded_squared_density(*cell) for cell in sg),
        tuple(_embedded_squared_density(*cell) for cell in tg),
    )
    ss, ts = source.dof_maps[si].global_dof_count, target.dof_maps[ti].global_dof_count
    piece_count = sum(len(occurrence.pieces) for occurrence in prepared.occurrences)
    form_count = (
        ss
        + ts
        + sum(len(basis) ** 2 for _, _, basis in tc)
        + piece_count
        * max(len(basis) for _, _, basis in sc)
        * max(len(basis) for _, _, basis in tc)
    )
    local_budgets: _IntegrationBudgets = {
        **budgets,
        "absolute_tolerance": budgets["absolute_tolerance"] / form_count,
        "relative_tolerance": budgets["relative_tolerance"] / form_count,
    }
    sm, tm, se, te = np.zeros(ss), np.zeros(ts), np.zeros(ss), np.zeros(ts)
    masses: list[_FloatArray] = []
    mixed: list[dict[int, _FloatArray]] = [dict() for _ in tc]
    errors = 0.0
    for cells, density, measures, uncertainty, is_target in (
        (sc, grams[0], sm, se, False),
        (tc, grams[1], tm, te, True),
    ):
        for cell, (element, route, basis) in enumerate(cells):
            if element.cell_kind != "triangle":
                raise ValueError(
                    "Surface chart DG requires canonical scalar triangle elements."
                )
            for dof, term in zip(route, basis, strict=True):
                measures[dof], uncertainty[dof] = _integral(
                    density[cell], term, local_budgets
                )
            if is_target:
                matrix = np.empty((len(basis), len(basis)), dtype=np.float64)
                for i, first in enumerate(basis):
                    for j, second in enumerate(basis):
                        matrix[i, j], error = _integral(
                            density[cell], multiply(first, second), local_budgets
                        )
                        errors += error
                masses.append(matrix)
    for occurrence in prepared.occurrences:
        for _, old_row, new_row, old_ref, new_ref in _piece_maps(occurrence):
            old_cell = int(source_order[int(np.asarray(occurrence.source_rows)[old_row])])
            new_cell = int(target_order[int(np.asarray(occurrence.target_rows)[new_row])])
            old_gram, old_arguments = _restricted(grams[0][old_cell], old_ref)
            new_gram, new_arguments = _restricted(grams[1][new_cell], new_ref)
            gram = old_gram if semantics == "conservative-density" else new_gram
            old_basis = tuple(compose(term, old_arguments) for term in sc[old_cell][2])
            new_basis = tuple(compose(term, new_arguments) for term in tc[new_cell][2])
            form = np.empty((len(new_basis), len(old_basis)), dtype=np.float64)
            for i, first in enumerate(new_basis):
                for j, second in enumerate(old_basis):
                    form[i, j], error = _integral(
                        gram, multiply(first, second), local_budgets
                    )
                    errors += error
            previous = mixed[new_cell].get(old_cell)
            mixed[new_cell][old_cell] = form if previous is None else previous + form
    return _finish_material_dg_projection(
        source,
        target,
        prepared,
        field_name=name,
        source_cells=sc,
        target_cells=tc,
        source_measures=sm,
        target_measures=tm,
        source_errors=se,
        target_errors=te,
        masses=masses,
        mixed=mixed,
        integration_error=errors,
        semantics=semantics,
        budgets=budgets,
    )


def _finish_material_dg_projection(
    source: FiniteElementDiscretization,
    target: FiniteElementDiscretization,
    prepared: _ChartDeformation | SourceGeometryRealization,
    /,
    *,
    field_name: str,
    source_cells: tuple[_MappedDgCell, ...],
    target_cells: tuple[_MappedDgCell, ...],
    source_measures: _FloatArray,
    target_measures: _FloatArray,
    source_errors: _FloatArray,
    target_errors: _FloatArray,
    masses: Sequence[_FloatArray],
    mixed: Sequence[dict[int, _FloatArray]],
    integration_error: float,
    semantics: Literal["intensive", "conservative-density"],
    budgets: _IntegrationBudgets,
) -> FiniteElementFieldTransfer:
    """Native scalar mass projection under its declared physical functional."""
    binding = _binding(prepared)
    correspondence_id = (
        prepared.realization_id
        if isinstance(prepared, SourceGeometryRealization)
        else prepared.deformation_id
    )
    si, ti = source._field_index(field_name), target._field_index(field_name)
    sc, tc, name = source_cells, target_cells, field_name
    sm, tm, se, te = source_measures, target_measures, source_errors, target_errors
    ss, ts, errors = sm.size, tm.size, integration_error
    row_batches: list[_IntArray] = []
    column_batches: list[_IntArray] = []
    coefficient_batches: list[_FloatArray] = []
    lifts: list[_FloatArray] = []
    denominator = np.zeros(ss, dtype=np.float64)
    condition, solve_defect, correction_bound = 1.0, 0.0, 0.0
    conservative = semantics == "conservative-density"
    for cell, matrix in enumerate(masses):
        parents = sorted(mixed[cell])
        if not parents:
            raise ValueError("Surface chart DG target has no complete overlap.")
        rhs = np.concatenate([mixed[cell][parent] for parent in parents], axis=1)
        old_routes = np.concatenate([sc[parent][1] for parent in parents])
        route = tc[cell][1]
        # The exact projection has a physical inventory constraint for density,
        # and a partition-of-unity constraint for intensive scalars. Independently
        # enclosed forms do not share identical publication errors; enforce these
        # actual functionals using the same native mass inverse, not relaxed claims.
        extra = tm[route] if conservative else matrix.sum(axis=1) - rhs.sum(axis=1)
        full_rhs = np.column_stack((rhs, extra))
        factor = factorize(
            DenseLinearOperator(
                matrix,
                operator_id=canonical_fingerprint(
                    {
                        "kind": "surface-chart-dg-local-mass",
                        "values": array_tree_fingerprint(matrix),
                    }
                ),
            ),
            FactorizationPolicy("svd"),
        )
        if int(np.asarray(factor.rank())) != matrix.shape[0]:
            raise ValueError("Surface chart DG target physical mass is rank deficient.")
        spectrum = np.asarray(factor.singular_values())
        condition = max(condition, float(spectrum[0] / spectrum[-1]))
        solved = factor.solve(jnp.asarray(full_rhs, dtype=jnp.float64))
        if not bool(np.all(np.asarray(solved.successful))):
            raise ValueError("Surface chart DG native mass solve failed.")
        full_value = np.asarray(solved.value, dtype=np.float64)
        value, lift = full_value[:, :-1], full_value[:, -1]
        if not np.all(np.isfinite(full_value)):
            raise ValueError(
                "Surface chart DG mass solve returned nonfinite coefficients."
            )
        solve_defect = max(
            solve_defect, float(np.max(np.abs(matrix @ full_value - full_rhs)))
        )
        if conservative:
            local_denominator = float(np.dot(tm[route], lift))
            if not np.isfinite(local_denominator) or local_denominator <= 0:
                raise ValueError(
                    "Surface chart DG physical inventory lift is not positive."
                )
            np.add.at(denominator, old_routes, local_denominator)
            lifts.append(np.repeat(lift, old_routes.size))
        else:
            physical_mean = sm[old_routes]
            total_mean = float(np.sum(physical_mean))
            if not np.isfinite(total_mean) or total_mean <= 0:
                raise ValueError(
                    "Surface chart DG source support has no positive physical measure."
                )
            correction = lift[:, None] * (physical_mean / total_mean)[None, :]
            correction_bound = max(
                correction_bound,
                float(np.max(np.abs(correction)))
                + np.finfo(np.float64).eps * float(np.max(np.abs(value))),
            )
            value = value + correction
        row_batches.append(np.repeat(route, old_routes.size))
        column_batches.append(np.tile(old_routes, route.size))
        coefficient_batches.append(value.reshape(-1))
    rows = np.concatenate(row_batches)
    columns = np.concatenate(column_batches)
    coefficients = np.concatenate(coefficient_batches)
    if conservative:
        if np.any(denominator <= 0):
            raise ValueError(
                "Surface chart DG inventory constraint lacks source support."
            )
        residual = _column_content(rows, columns, coefficients, sm, tm)
        correction = (
            np.concatenate(lifts)
            * (
                np.asarray([float(value) for value in residual], dtype=np.float64)
                / denominator
            )[columns]
        )
        correction_bound = float(np.max(np.abs(correction))) + np.finfo(
            np.float64
        ).eps * float(np.max(np.abs(coefficients)))
        coefficients -= correction
    amplification = max(1.0, condition * max(matrix.shape[0] ** 2 for matrix in masses))
    primal = _coalesced_map(
        rows,
        columns,
        coefficients,
        target_size=ts,
        source_size=ss,
        properties=OperatorProperties(),
        operator_id=canonical_fingerprint(
            {
                "kind": "source-realization-dg"
                if isinstance(prepared, SourceGeometryRealization)
                else "surface-chart-dg",
                "field": name,
                "semantics": semantics,
                "source": source.prepared_id,
                "target": target.prepared_id,
                "deformation": correspondence_id,
                "source_geometry": binding.source_geometry_id,
                "target_geometry": binding.target_geometry_id,
                "values": array_tree_fingerprint(coefficients),
            }
        ),
    )
    scale_ = max(float(np.sum(np.abs(sm))), float(np.sum(np.abs(tm))), 1.0)
    propagated = np.zeros(ss, dtype=np.float64)
    np.add.at(propagated, columns, np.abs(coefficients) * te[rows])
    exact_content = _column_content(rows, columns, coefficients, sm, tm)
    from .._coordinate_enclosure import outward

    content_defect = max(outward(abs(value), np.inf) for value in exact_content)
    publication_roundoff = np.finfo(np.float64).eps * (coefficients.size + 2)
    content_bound = float(
        np.nextafter(
            content_defect
            + np.max(se + propagated) / (1 - publication_roundoff)
            + _measure_defect_bound(np.dtype(np.float64), amplification, sm, tm),
            np.inf,
        )
    )
    tolerance = max(
        64 * np.finfo(np.float64).eps * amplification,
        budgets["absolute_tolerance"] + budgets["relative_tolerance"],
    )
    defects = {"mass_solve": solve_defect / scale_}
    if conservative:
        defects["content"] = content_defect / scale_
    else:
        defects["constants"] = float(
            np.max(np.abs(np.asarray(primal.mv(jnp.ones(ss))) - 1))
        )
    evidence = FiniteElementTransferEvidence(
        defects,
        tolerance,
        bounds={
            "integration": float(np.nextafter(errors + np.sum(se) + np.sum(te), np.inf)),
            "projection_correction": float(np.nextafter(correction_bound, np.inf)),
            **({"content": content_bound} if conservative else {}),
        },
        estimates={"mass_condition": condition},
    )
    if not evidence.passed:
        raise ValueError("Surface chart DG physical projection certificate failed.")
    transfer = FiniteElementTopologyTransfer(
        primal,
        binding.source_topology_id,
        binding.target_topology_id,
        conservative=conservative,
        preserves_constants=not conservative,
        semantics="l2-projection",
        positivity_preserving=bool(np.all(coefficients >= 0)),
        action_condition=amplification,
        source_measures=sm if conservative else None,
        target_measures=tm if conservative else None,
    )
    return FiniteElementFieldTransfer(
        transfer,
        source.field_spaces[si],
        target.field_spaces[ti],
        binding,
        evidence,
        source_measures=sm if conservative else None,
        target_measures=tm if conservative else None,
    )


def prepare_surface_chart_field_transfer(
    source: FiniteElementDiscretization,
    target: FiniteElementDiscretization,
    prepared: PreparedSurfaceChartDeformation,
    /,
    *,
    field_name: str,
    source_geometry: CellGeometrySpec,
    target_geometry: CellGeometrySpec,
    semantics: Literal["intensive", "conservative-density"] = "intensive",
    absolute_tolerance: float = 1e-12,
    relative_tolerance: float = 1e-12,
    maximum_work: int = 100_000_000,
    maximum_subcells: int = 10000,
    maximum_binomial_terms: int = 32,
) -> FiniteElementFieldTransfer:
    """Prepare actual scalar material correspondence and its algebraic transpose.

    Both supplied coordinate specifications must realize the actual prepared
    endpoint elements, routes, layout and coordinate values; a valid external
    chart witness cannot authorize a differently prepared field map.

    H1 supports intensive point functionals. DG intensive projection uses target
    density; conservative-density projection uses old physical density and target
    physical mass. A constant conservative density need not remain constant when
    the surface area changes. H(curl), H(div), and other all-field dispositions
    require a separate declared differential/geometric transport and are refused.

    Enclosed mass and mixed forms are projected under their exact declared
    functional: old physical inventory for conservative density, or
    partition-of-unity for intensive scalars. Inventory uses a native
    mass-minimal lift on the true overlap support. Intensive correction uses
    the old support's physical mean functional. The coefficient perturbation
    from independent integration/publication errors is exposed by
    ``evidence.bound("projection_correction")``; conservation/constant claims
    retain the canonical roundoff-only checks, never an inflated tolerance.

    Error tolerances are allocated across all integrated forms; work/subcell
    limits apply to each enclosed integral. Physical inventory errors are exposed
    by ``evidence.bound("content")`` for conservative-density fields. Epoch
    transport uses the canonical field-transfer disposition: positive DOF
    measures admit a topology content ledger; other scalar bases retain a
    ``FieldEpochTransition`` with the physical content evidence on this artifact.
    """
    if semantics not in ("intensive", "conservative-density"):
        raise ValueError("Unknown surface chart field semantics.")
    source_order, target_order = _validate(
        source, target, prepared, source_geometry, target_geometry
    )
    si, ti = source._field_index(field_name), target._field_index(field_name)
    source_space = source.field_spaces[si].vector_space
    target_space = target.field_spaces[ti].vector_space
    if not isinstance(source_space, ArraySpace) or not isinstance(
        target_space, ArraySpace
    ):
        raise TypeError("Surface chart transfer requires array coefficient spaces.")
    if source_space.shape[1:] != target_space.shape[1:]:
        raise ValueError("Surface chart field payload layouts differ.")
    conformities = {
        element.conformity for element in source.elements[si] + target.elements[ti]
    }
    if conformities == {"H1"} and semantics == "intensive":
        return _prepare_h1(
            source, target, prepared, source_order, target_order, field_name
        )
    if conformities != {"L2"}:
        raise ValueError(
            "Surface chart all-field contract is open for non-scalar-DG density or compatible/Piola fields."
        )
    budgets = _IntegrationBudgets(
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
        maximum_work=maximum_work,
        maximum_subcells=maximum_subcells,
        maximum_binomial_terms=maximum_binomial_terms,
    )
    return _prepare_dg(
        source,
        target,
        prepared,
        source_geometry,
        target_geometry,
        source_order,
        target_order,
        field_name,
        semantics,
        budgets,
    )


class PreparedSurfaceChartFiniteVolumeContents(StrictModule, NonTrainableState):
    """Old physical density contents on exact material pieces, not chart areas.

    Rows refer to current concatenated physical cell rows. Dividing these contents
    by target volumes gives conservative-density transfer; dividing by summed old
    contents gives intensive cell averages. The latter is not inventory preserving
    under an area-changing geometry deformation. Errors are absolute physical
    measure enclosures, suitable for the owner's quantitative content ledger.
    """

    source_rows: Array
    target_rows: Array
    source_contents: Array
    source_content_errors: Array
    source_volumes: Array
    source_volume_errors: Array
    source_content_defect_bounds: Array
    target_volumes: Array
    target_volume_errors: Array
    source_geometry_id: str = eqx.field(static=True)
    target_geometry_id: str = eqx.field(static=True)
    source_topology_id: str = eqx.field(static=True)
    target_topology_id: str = eqx.field(static=True)
    deformation_id: str = eqx.field(static=True)
    contents_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_rows: ArrayLike,
        target_rows: ArrayLike,
        contents: ArrayLike,
        errors: ArrayLike,
        volumes: ArrayLike,
        volume_errors: ArrayLike,
        source_volumes: ArrayLike,
        source_volume_errors: ArrayLike,
        content_bounds: ArrayLike,
        prepared: _ChartDeformation,
    ) -> None:
        self.source_rows = jnp.asarray(source_rows, dtype=jnp.int64)
        self.target_rows = jnp.asarray(target_rows, dtype=jnp.int64)
        self.source_contents = jnp.asarray(contents, dtype=jnp.float64)
        self.source_content_errors = jnp.asarray(errors, dtype=jnp.float64)
        self.target_volumes = jnp.asarray(volumes, dtype=jnp.float64)
        self.target_volume_errors = jnp.asarray(volume_errors, dtype=jnp.float64)
        self.source_volumes = jnp.asarray(source_volumes, dtype=jnp.float64)
        self.source_volume_errors = jnp.asarray(source_volume_errors, dtype=jnp.float64)
        self.source_content_defect_bounds = jnp.asarray(content_bounds, dtype=jnp.float64)
        self.source_geometry_id, self.target_geometry_id = (
            prepared.source_geometry_id,
            prepared.target_geometry_id,
        )
        self.source_topology_id, self.target_topology_id = (
            prepared.source_topology_id,
            prepared.target_topology_id,
        )
        self.deformation_id = prepared.deformation_id
        self.contents_id = canonical_fingerprint(
            {
                "kind": "surface-chart-finite-volume-contents",
                "deformation": prepared.deformation_id,
                "source_rows": array_tree_fingerprint(self.source_rows),
                "target_rows": array_tree_fingerprint(self.target_rows),
                "contents": array_tree_fingerprint(self.source_contents),
                "errors": array_tree_fingerprint(self.source_content_errors),
                "target_volumes": array_tree_fingerprint(self.target_volumes),
                "target_errors": array_tree_fingerprint(self.target_volume_errors),
                "source_volumes": array_tree_fingerprint(self.source_volumes),
                "source_errors": array_tree_fingerprint(self.source_volume_errors),
                "content_bounds": array_tree_fingerprint(
                    self.source_content_defect_bounds
                ),
            }
        )


def prepare_surface_chart_finite_volume_contents(
    source: _ChartEndpoint,
    target: _ChartEndpoint,
    prepared: PreparedSurfaceChartDeformation,
    /,
    *,
    source_geometry: CellGeometrySpec,
    target_geometry: CellGeometrySpec,
    absolute_tolerance: float = 1e-12,
    relative_tolerance: float = 1e-12,
    maximum_work: int = 100_000_000,
    maximum_subcells: int = 10000,
    maximum_binomial_terms: int = 32,
) -> PreparedSurfaceChartFiniteVolumeContents:
    """Prepare occurrence-local physical overlaps for the finite-volume owner.

    Actual unstructured FV endpoints bind their retained coordinate map and
    certified physical volumes/errors. Those endpoint arrays, rather than
    independent core integration midpoints, define the returned inventory
    pairings. Their exact dyadic measure intervals must agree with the core's
    scientific-cell certificate. FE endpoints remain supported for geometry
    handoff and must bind their actual prepared coordinate specification.
    """
    source_order, target_order = _validate(
        source, target, prepared, source_geometry, target_geometry
    )
    rows, targets, source_references = [], [], []
    exact_whole_targets = True
    for occurrence in prepared.occurrences:
        displacement = np.asarray(occurrence.displacement_bounds, dtype=np.float64)
        if displacement.shape != (len(occurrence.pieces),):
            raise ValueError(
                "Material displacement bounds do not align with their exact pieces."
            )
        for entry, old_row, new_row, old_ref, _ in _piece_maps(occurrence):
            old = int(source_order[int(np.asarray(occurrence.source_rows)[old_row])])
            new = int(target_order[int(np.asarray(occurrence.target_rows)[new_row])])
            rows.append(old)
            targets.append(new)
            source_references.append(old_ref)
            exact_whole_targets &= displacement[entry] == 0.0 and occurrence.pieces[
                entry
            ].target_reference_area == Fraction(1, 2)
    volumes, volume_errors = _endpoint_cell_measures(
        target, prepared, target_order, "target"
    )
    old_volumes, old_errors = _endpoint_cell_measures(
        source, prepared, source_order, "source"
    )
    exact_whole_targets &= len(targets) == volumes.size and np.array_equal(
        np.sort(np.asarray(targets, dtype=np.int64)),
        np.arange(volumes.size, dtype=np.int64),
    )
    if exact_whole_targets:
        target_rows = np.asarray(targets, dtype=np.int64)
        values = volumes[target_rows].tolist()
        errors = volume_errors[target_rows].tolist()
    else:
        geometry = _mapped_geometry_cells(source.mesh, source_geometry)
        grams = tuple(
            _embedded_squared_density(*cell, denominators_certified=True)
            for cell in geometry
        )
        budgets = _IntegrationBudgets(
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
            maximum_work=maximum_work,
            maximum_subcells=maximum_subcells,
            maximum_binomial_terms=maximum_binomial_terms,
        )
        piece_count = len(rows)
        budgets["absolute_tolerance"] /= piece_count
        budgets["relative_tolerance"] /= piece_count
        values, errors = [], []
        for old, old_ref in zip(rows, source_references, strict=True):
            gram, _ = _restricted(grams[old], old_ref)
            value, error = _integral(gram, constant(1, 2), budgets)
            values.append(value)
            errors.append(error)
    sums, uncertainties = np.zeros_like(old_volumes), np.zeros_like(old_volumes)
    np.add.at(sums, rows, values)
    np.add.at(uncertainties, rows, errors)
    roundoff = np.finfo(np.float64).eps * (len(values) + 2)
    content_bounds = np.nextafter(
        (
            np.abs(sums - old_volumes)
            + uncertainties
            + old_errors
            + roundoff * (sums + old_volumes)
        )
        / (1 - roundoff),
        np.inf,
    )
    return PreparedSurfaceChartFiniteVolumeContents(
        np.asarray(rows, dtype=np.int64),
        np.asarray(targets, dtype=np.int64),
        np.asarray(values, dtype=np.float64),
        np.asarray(errors, dtype=np.float64),
        volumes,
        volume_errors,
        old_volumes,
        old_errors,
        content_bounds,
        prepared,
    )


__all__ = [
    "PreparedSurfaceChartFiniteVolumeContents",
    "prepare_surface_chart_field_transfer",
    "prepare_surface_chart_finite_volume_contents",
]
