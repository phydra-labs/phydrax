#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"Prepare commuting forms between independent material-reference charts."

from __future__ import annotations

from collections.abc import Callable, Sequence
from fractions import Fraction
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from phydrax.discretization._sphere_chart_records import PreparedSphereChartDeformation
from phydrax.discretization._surface_chart_records import PreparedSurfaceChartDeformation
from phydrax.geometry._rational import fraction_endpoint

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._resource_ticket import HostBatchTicket
from ..._strict import StrictModule
from ...geometry._rational import rational_vector
from ...linalg import (
    _small_batched as imported_linalg__small_batched,
    ArraySpace,
    OperatorProperties,
)
from ...linalg._hermitian_spectral import reserve_fraction_work
from ...sparse import RowRelation, SparseLinearMap
from ...typing import checked
from .. import _coordinate_enclosure as algebra
from .._cell_geometry import CellGeometrySpec
from .._chart_geometry import orientation
from .._coordinate_budget import CoordinateEnclosureBudget
from .._reference_cell import reference_cell_topology
from ._exact_form_moments import ExactFormMoments, MomentIntegralMatrix
from ._form_basis_types import AbstractReferenceFormBasis
from ._form_elements import FormBasis
from ._generic import FiniteElementDiscretization
from ._mapped_form_transfer import (
    correct_commuting_moments,
    derivative_cell,
    embed_columns,
    form_cells,
    FormCell,
    PreparationWork,
    transform_integrals,
)
from ._surface_chart_transfer import deformation_binding, piece_maps, validate
from ._topology_transfer import (
    CLAIM_ULPS,
    coalesced_map,
    field_transfer_query_upper,
    FiniteElementFieldTransfer,
    FiniteElementTopologyTransfer,
    FiniteElementTransferEvidence,
    owned_rows,
)


type _Point = tuple[Fraction, Fraction]
type MaterialCompatiblePiece = tuple[
    int, int, tuple[algebra.Expression, ...], tuple[algebra.Expression, ...]
]


class PreparedSurfaceChartCompatibleTransfer(NamedTuple):
    field_transfer: FiniteElementFieldTransfer
    source_differential: SparseLinearMap
    target_differential: SparseLinearMap
    differential_companion: SparseLinearMap


def _material_entity_coverage(
    entity: tuple[int, ...],
    keys: Sequence[tuple[tuple[Fraction, ...], ...]],
) -> None:
    vertices = tuple(
        rational_vector(point) for point in reference_cell_topology("triangle").vertices
    )
    dimension = len(entity) - 1
    if dimension == 0:
        if set(keys) != {(vertices[entity[0]],)}:
            raise ValueError(
                "surface_chart.material_entity_coverage.pieces must satisfy the material reference contract; got Material correspondence does not support its target vertex functional."
            )
    elif dimension == 1:
        first, last = vertices[entity[0]], vertices[entity[1]]
        axis = next(axis for axis in range(2) if first[axis] != last[axis])
        intervals = sorted(
            tuple(
                sorted(
                    (point[axis] - first[axis]) / (last[axis] - first[axis])
                    for point in key
                )
            )
            for key in keys
        )
        previous = Fraction(0)
        for lower, upper in intervals:
            if lower != previous or upper <= lower:
                raise ValueError(
                    "surface_chart.material_entity_coverage.pieces must satisfy the material reference contract; got Material pieces leave a gap or overlapping target edge functional."
                )
            previous = upper
        if previous != 1:
            raise ValueError(
                "surface_chart.material_entity_coverage.pieces must satisfy the material reference contract; got Material pieces do not cover the entire target edge functional."
            )
    elif dimension == 2:
        area = sum(
            (
                abs(
                    orientation(
                        (key[0][0], key[0][1]),
                        (key[1][0], key[1][1]),
                        (key[2][0], key[2][1]),
                    )
                )
                / 2
                for key in keys
            ),
            Fraction(0),
        )
        if area != Fraction(1, 2):
            raise ValueError(
                "surface_chart.material_entity_coverage.pieces must satisfy the material reference contract; got Material pieces do not cover the entire target cell functional."
            )
    else:
        raise ValueError(
            "surface_chart.material_entity_coverage.binding must satisfy the declared geometry and cell identities; got Material forms require triangular reference entities."
        )


def _material_common_moments(
    sources: tuple[FormCell, ...],
    target: AbstractReferenceFormBasis,
    pieces: tuple[MaterialCompatiblePiece, ...],
    columns: NDArray[np.int64],
    work: PreparationWork,
    exact: ExactFormMoments,
) -> tuple[MomentIntegralMatrix, float]:
    work.charge(0, 2 * target.local_dof_count * columns.size)
    values = np.zeros((target.local_dof_count, columns.size), dtype=np.float64)
    errors = np.zeros_like(values)
    continuity = 0.0
    topology = reference_cell_topology("triangle")
    for dimension, entities in enumerate(target.entity_vertices):
        for entity in entities:
            if not any(label[0] == entity for label in target.dof_labels):
                continue
            seen: dict[tuple[tuple[Fraction, ...], ...], MomentIntegralMatrix] = {}
            for source_row, _, source_arguments, target_arguments in pieces:
                source = sources[source_row]
                for native_entity in topology.entities[dimension]:
                    work.charge(
                        target.local_dof_count
                        * source.basis.local_dof_count
                        * source.routes.size,
                        2
                        * target.local_dof_count
                        * (source.basis.local_dof_count + source.routes.size),
                    )
                    with (
                        work.enclosure.temporary_scope(),
                        work.enclosure.polynomial_preparation_scope(),
                    ):
                        result = exact.common_entity(
                            source.basis,
                            target,
                            entity,
                            "triangle",
                            native_entity,
                            source_arguments,
                            target_arguments,
                        )
                        if result is None:
                            continue
                        raw, key = result
                        transformed = transform_integrals(raw, source.transform)
                        contribution = MomentIntegralMatrix(
                            embed_columns(
                                transformed.value, source.routes, columns, work
                            ),
                            embed_columns(
                                transformed.error, source.routes, columns, work
                            ),
                        )
                    previous = seen.get(key)
                    if previous is not None:
                        scale_ = max(
                            float(np.max(np.abs(previous.value), initial=0)),
                            float(np.max(np.abs(contribution.value), initial=0)),
                            1.0,
                        )
                        continuity = max(
                            continuity,
                            float(
                                np.max(
                                    np.abs(previous.value - contribution.value)
                                    + previous.error
                                    + contribution.error,
                                    initial=0,
                                )
                            )
                            / scale_,
                        )
                    else:
                        work.enclosure.retain_basis((key,))
                        values += contribution.value
                        errors += contribution.error
                        seen[key] = contribution
            _material_entity_coverage(entity, tuple(seen))
    gamma = np.finfo(np.float64).eps * max(len(pieces), 1)
    return MomentIntegralMatrix(
        values, np.nextafter(errors + gamma * np.abs(values) / (1 - gamma), np.inf)
    ), continuity


def _material_sparse_blocks(
    blocks: Sequence[tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]],
    target_size: int,
    source_size: int,
    name: str,
) -> SparseLinearMap:
    rows, columns, values = [], [], []
    for targets, sources, coefficients in blocks:
        rows.append(np.broadcast_to(targets[:, None], coefficients.shape).reshape((-1,)))
        columns.append(np.broadcast_to(sources[None], coefficients.shape).reshape((-1,)))
        values.append(coefficients.reshape((-1,)))
    return coalesced_map(
        np.concatenate(rows),
        np.concatenate(columns),
        np.concatenate(values),
        target_size=target_size,
        source_size=source_size,
        properties=OperatorProperties(),
        operator_id=canonical_fingerprint(
            {
                "kind": name,
                "rows": array_tree_fingerprint(tuple(rows)),
                "columns": array_tree_fingerprint(tuple(columns)),
                "values": array_tree_fingerprint(tuple(values)),
            }
        ),
    )


class _MaterialSourceDifferentials(StrictModule):
    old_offset: int


def _prepare_material_source_differentials(
    sources: tuple[FormCell, ...],
    derivative_cache: dict[int, tuple[FormBasis, NDArray[np.float64]]],
    work: PreparationWork,
    budget: CoordinateEnclosureBudget,
    source_differentials: list[
        tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]
    ],
    companion_owners: dict[int, tuple[int, int]],
    derivative_sources: list[FormCell],
    companion_sources: list[FormCell],
    old_offset: int,
) -> _MaterialSourceDifferentials:
    "Prepare source exterior derivatives and companion coefficient routes."
    for cell in sources:
        work.observe_basis(cell.basis)
        following = derivative_cell(cell, derivative_cache, budget=budget)
        work.observe_basis(following.basis)
        routes = np.arange(
            old_offset, old_offset + following.basis.local_dof_count, dtype=np.int64
        )
        work.charge(
            following.transform.size,
            routes.size + following.basis.local_dof_count**2 + following.transform.size,
        )
        source_differentials.append((routes, cell.routes, following.transform))
        companion_owners.update(
            (int(identifier), (len(source_differentials) - 1, index))
            for index, identifier in enumerate(routes)
        )
        derivative_sources.append(following)
        companion_sources.append(
            FormCell(
                following.basis,
                routes,
                np.eye(following.basis.local_dof_count, dtype=np.float64),
            )
        )
        old_offset += routes.size
    return _MaterialSourceDifferentials(old_offset=old_offset)


class _MaterialTargetMoments(StrictModule):
    new_offset: int
    continuity: float


def _prepare_material_target_moments(
    sources: tuple[FormCell, ...],
    targets: tuple[FormCell, ...],
    charts: list[list[MaterialCompatiblePiece]],
    derivative_cache: dict[int, tuple[FormBasis, NDArray[np.float64]]],
    work: PreparationWork,
    exact: ExactFormMoments,
    budget: CoordinateEnclosureBudget,
    derivative_sources: list[FormCell],
    companion_sources: list[FormCell],
    candidate_rows: list[int],
    candidate_columns: list[NDArray[np.int64]],
    candidate_values: list[NDArray[np.float64]],
    target_differentials: list[
        tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]
    ],
    companion_blocks: list[
        tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]
    ],
    published_checks: list[
        tuple[
            NDArray[np.int64],
            NDArray[np.float64],
            NDArray[np.int64],
            MomentIntegralMatrix,
            NDArray[np.int64],
            MomentIntegralMatrix,
        ]
    ],
    new_offset: int,
    continuity: float,
) -> _MaterialTargetMoments:
    "Integrate target moments and assemble their differential companion."
    for row, cell in enumerate(targets):
        work.observe_basis(cell.basis)
        pieces = tuple(charts[row])
        if not pieces:
            raise ValueError(
                "surface_chart.prepare_material_target_moments.pieces must satisfy the material reference contract; got Material correspondence leaves a target compatible cell unsupported."
            )
        count = sum(sources[old].routes.size for old, _, _, _ in pieces)
        work.charge(count, count)
        columns = np.unique(
            np.concatenate([sources[old].routes for old, _, _, _ in pieces])
        )
        raw, defect = _material_common_moments(
            sources, cell.basis, pieces, columns, work, exact
        )
        continuity = max(continuity, defect)
        following = derivative_cell(
            FormCell(
                cell.basis,
                cell.routes,
                np.eye(cell.basis.local_dof_count, dtype=np.float64),
            ),
            derivative_cache,
            budget=budget,
        )
        work.observe_basis(following.basis)
        expected, defect = _material_common_moments(
            tuple(derivative_sources), following.basis, pieces, columns, work, exact
        )
        continuity = max(continuity, defect)
        corrected = correct_commuting_moments(
            cell.basis, raw, expected, following.transform, columns, work
        )
        local = work.orientation_solve(cell.transform, corrected.value)
        spectrum = np.asarray(work.factor_matrix(cell.transform).singular_values())
        work.integration_error = max(
            work.integration_error,
            float(np.max(corrected.error, initial=0))
            * np.sqrt(cell.basis.local_dof_count)
            / float(spectrum[-1]),
        )
        candidate_rows.extend(cell.routes.tolist())
        candidate_columns.extend(np.broadcast_to(columns, local.shape))
        candidate_values.extend(local)
        new_routes = np.arange(
            new_offset, new_offset + following.basis.local_dof_count, dtype=np.int64
        )
        work.charge(
            following.transform.size * cell.transform.shape[1],
            following.transform.shape[0] * cell.transform.shape[1] + new_routes.size,
        )
        differential = following.transform @ cell.transform
        target_differentials.append((new_routes, cell.routes, differential))
        companion_columns = np.unique(
            np.concatenate([companion_sources[old].routes for old, _, _, _ in pieces])
        )
        companion, defect = _material_common_moments(
            tuple(companion_sources),
            following.basis,
            pieces,
            companion_columns,
            work,
            exact,
        )
        continuity = max(continuity, defect)
        companion_blocks.append((new_routes, companion_columns, companion.value))
        published_checks.append(
            (
                cell.routes,
                differential,
                columns,
                expected,
                companion_columns,
                companion,
            )
        )
        new_offset += new_routes.size
    return _MaterialTargetMoments(new_offset=new_offset, continuity=continuity)


def _check_published_material_commuting(
    published_checks: list[
        tuple[
            NDArray[np.int64],
            NDArray[np.float64],
            NDArray[np.int64],
            MomentIntegralMatrix,
            NDArray[np.int64],
            MomentIntegralMatrix,
        ]
    ],
    routes: NDArray[np.int64],
    coefficients: NDArray[np.float64],
    companion_owners: dict[int, tuple[int, int]],
    source_differentials: list[
        tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]
    ],
    work: PreparationWork,
) -> None:
    "Bound commuting defects using the published field coefficient rows."
    for (
        dofs,
        differential,
        expected_columns,
        expected,
        companion_columns,
        companion,
    ) in published_checks:
        columns = np.unique(
            np.concatenate((expected_columns, routes[dofs].reshape((-1,))))
        )
        work.charge(
            differential.size * columns.size,
            dofs.size * columns.size + 4 * differential.shape[0] * columns.size,
        )
        values = np.zeros((dofs.size, columns.size), dtype=np.float64)
        np.add.at(
            values,
            (np.arange(dofs.size)[:, None], np.searchsorted(columns, routes[dofs])),
            coefficients[dofs],
        )
        actual = differential @ values
        desired = np.zeros((differential.shape[0], columns.size), dtype=np.float64)
        error = np.zeros_like(desired)
        for index, identifier in enumerate(companion_columns):
            owner, local_row = companion_owners[int(identifier)]
            _, source_columns, source_differential = source_differentials[owner]
            differential_row = source_differential[local_row]
            work.charge(
                differential.shape[0] * source_columns.size,
                2 * differential.shape[0] * source_columns.size,
            )
            contribution = companion.value[:, index, None] * differential_row[None]
            uncertainty = companion.error[:, index, None] * np.abs(differential_row[None])
            uncertainty += np.finfo(np.float64).eps * np.abs(contribution)
            selected = np.searchsorted(columns, source_columns)
            np.add.at(
                desired,
                (np.arange(differential.shape[0])[:, None], selected[None]),
                contribution,
            )
            np.add.at(
                error,
                (np.arange(differential.shape[0])[:, None], selected[None]),
                uncertainty,
            )
        gamma = np.finfo(np.float64).eps * max(
            companion_columns.size, differential.shape[1]
        )
        error = np.nextafter(
            error
            + gamma
            * (np.abs(desired) + np.abs(differential) @ np.abs(values))
            / (1 - gamma),
            np.inf,
        )
        independently_integrated = embed_columns(
            expected.value, expected_columns, columns, work
        )
        independent_error = embed_columns(expected.error, expected_columns, columns, work)
        scale_ = max(
            float(np.max(np.abs(desired), initial=0)),
            float(np.max(np.abs(independently_integrated), initial=0)),
            1.0,
        )
        work.commuting_defect = max(
            work.commuting_defect,
            float(
                np.max(
                    np.abs(desired - independently_integrated)
                    + error
                    + independent_error,
                    initial=0,
                )
            )
            / scale_,
        )
        scale_ = max(
            float(np.max(np.abs(actual), initial=0)),
            float(np.max(np.abs(desired), initial=0)),
            1.0,
        )
        work.commuting_defect = max(
            work.commuting_defect,
            float(np.max(np.abs(actual - desired) + error, initial=0)) / scale_,
        )


def _material_owned_rows(
    candidate_rows: list[int],
    candidate_columns: list[NDArray[np.int64]],
    candidate_values: list[NDArray[np.float64]],
    target_size: int,
    source_size: int,
    work: PreparationWork,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Charge padded storage before reconciling shared material coefficient rows."""
    width = max(len(columns) for columns in candidate_columns)
    work.charge(0, 2 * len(candidate_rows) * width)
    padded_columns = np.zeros((len(candidate_rows), width), dtype=np.int64)
    padded_values = np.zeros_like(padded_columns, dtype=np.float64)
    for row, (columns, values) in enumerate(
        zip(candidate_columns, candidate_values, strict=True)
    ):
        padded_columns[row, : columns.size], padded_values[row, : values.size] = (
            columns,
            values,
        )
    routes, coefficients, shared = owned_rows(
        np.asarray(candidate_rows, dtype=np.int64)[:, None],
        padded_columns,
        padded_values[:, None],
        target_size,
        source_size,
    )
    return routes, coefficients, shared, width


@checked
def prepare_material_chart_compatible_transfer(
    source: FiniteElementDiscretization,
    target: FiniteElementDiscretization,
    prepared: PreparedSurfaceChartDeformation | PreparedSphereChartDeformation,
    /,
    *,
    field_name: str,
    prepare_pieces: Callable[[PreparationWork], tuple[MaterialCompatiblePiece, ...]],
    maximum_work: int,
    maximum_storage_bytes: int,
    budget: CoordinateEnclosureBudget,
) -> PreparedSurfaceChartCompatibleTransfer:
    "Publish independent-chart moments and their whole-column companion.\n\n producers supply whole partitions, not rounded corner fits.\n Interior-only commuting correction preserves all target boundary moments.\n The returned companion is checked against the published field map."
    si, ti = source.field_index(field_name), target.field_index(field_name)
    conformities = {
        element.conformity for element in source.elements[si] + target.elements[ti]
    }
    if conformities not in ({"Hcurl"}, {"Hdiv"}):
        raise ValueError(
            "surface_chart.prepare_material_chart_compatible_transfer.pieces must satisfy the material reference contract; got Material compatible transport must retain H(curl) or H(div) conformity."
        )
    identities = {
        (
            element.value_spec.value_spec_id,
            element.degree,
            element.representation,
            element.family,
        )
        for endpoint, index in ((source, si), (target, ti))
        for element in endpoint.elements[index]
    }
    if len(identities) != 1:
        raise ValueError(
            "surface_chart.prepare_material_chart_compatible_transfer.pieces must satisfy the material reference contract; got Material forms cannot change degree, twist, proxy, order, or family."
        )
    sources, targets = form_cells(source, si), form_cells(target, ti)
    if any(
        cell.basis.dimension != 2 or cell.basis.form_degree != 1
        for cell in sources + targets
    ):
        raise ValueError(
            "surface_chart.prepare_material_chart_compatible_transfer.pieces must satisfy the material reference contract; got Material compatible fields require triangular one-form bases."
        )
    ss, ts = source.dof_maps[si].global_dof_count, target.dof_maps[ti].global_dof_count
    old_space, new_space = (
        source.field_spaces[si].vector_space,
        target.field_spaces[ti].vector_space,
    )
    match (old_space, new_space):
        case (ArraySpace() as old_space, ArraySpace() as new_space) if not (
            old_space.shape != (ss,)
            or new_space.shape != (ts,)
            or old_space.dtype != new_space.dtype
        ):
            pass
        case _:
            raise ValueError(
                "surface_chart.prepare_material_chart_compatible_transfer.pieces must satisfy the material reference contract; got Material compatible coefficient shape/precision contracts differ."
            )
    work = PreparationWork(maximum_work, maximum_storage_bytes, budget=budget)
    exact = ExactFormMoments(work.enclosure)
    derivative_cache: dict[int, tuple[FormBasis, NDArray[np.float64]]] = {}
    candidate_rows, candidate_columns, candidate_values = [], [], []
    source_differentials, target_differentials, companion_blocks = [], [], []
    companion_sources: list[FormCell] = []
    derivative_sources: list[FormCell] = []
    companion_owners: dict[int, tuple[int, int]] = {}
    old_offset = 0
    continuity = 0.0
    published_checks = []
    with work.enclosure.activate():
        _phase = _prepare_material_source_differentials(
            sources=sources,
            derivative_cache=derivative_cache,
            work=work,
            budget=budget,
            source_differentials=source_differentials,
            companion_owners=companion_owners,
            derivative_sources=derivative_sources,
            companion_sources=companion_sources,
            old_offset=old_offset,
        )
        old_offset = _phase.old_offset
        charts: list[list[MaterialCompatiblePiece]] = [[] for _ in targets]
        with work.enclosure.polynomial_preparation_scope():
            for old, new, arguments, images in prepare_pieces(work):
                if not 0 <= old < len(sources) or not 0 <= new < len(targets):
                    raise ValueError(
                        "surface_chart.prepare_material_chart_compatible_transfer.pieces must satisfy the material reference contract; got Material pieces do not bind the source and target cell rows."
                    )
                work.enclosure.retain_basis((arguments, images))
                charts[new].append((old, new, arguments, images))
        new_offset = 0
        _phase = _prepare_material_target_moments(
            sources=sources,
            targets=targets,
            charts=charts,
            derivative_cache=derivative_cache,
            work=work,
            exact=exact,
            budget=budget,
            derivative_sources=derivative_sources,
            companion_sources=companion_sources,
            candidate_rows=candidate_rows,
            candidate_columns=candidate_columns,
            candidate_values=candidate_values,
            target_differentials=target_differentials,
            companion_blocks=companion_blocks,
            published_checks=published_checks,
            new_offset=new_offset,
            continuity=continuity,
        )
        new_offset = _phase.new_offset
        continuity = _phase.continuity
        routes, coefficients, shared, width = _material_owned_rows(
            candidate_rows, candidate_columns, candidate_values, ts, ss, work
        )
        _check_published_material_commuting(
            published_checks,
            routes,
            coefficients,
            companion_owners,
            source_differentials,
            work,
        )
    condition = max(work.condition, 1.0) * width
    evidence = FiniteElementTransferEvidence(
        {
            "commuting": work.commuting_defect,
            "patch_trace": continuity,
            "continuity": shared,
            "moment_solve": work.solve_defect,
            "basis_duality": work.basis_solve_defect,
            "integration": work.integration_error,
        },
        CLAIM_ULPS * np.finfo(np.float64).eps * condition,
        bounds={
            "moments": work.integration_error,
            "geometry-displacement": prepared.maximum_displacement_bound,
        },
        estimates={
            "moment_condition": work.condition,
            "minimum_local_rank": work.minimum_rank,
            "projection_moment_change": work.projection_change,
            "preparation_work": float(work.preparation_work),
            "preparation_storage_bytes": float(work.preparation_storage_upper),
        },
    )
    if not evidence.passed:
        raise ValueError(
            f"surface_chart.prepare_material_chart_compatible_transfer.pieces must satisfy the material reference contract; got Material compatible moments failed their certificate:{evidence.defects}.."
        )
    primal = SparseLinearMap(
        RowRelation(routes.astype(np.int32), source_size=ss),
        coefficients,
        operator_id=canonical_fingerprint(
            {
                "kind": "material-chart-compatible-moments",
                "source": source.prepared_id,
                "target": target.prepared_id,
                "field": field_name,
                "deformation": prepared.deformation_id,
                "routes": array_tree_fingerprint(routes),
                "values": array_tree_fingerprint(coefficients),
            }
        ),
    )
    transfer = FiniteElementTopologyTransfer(
        primal,
        prepared.source_topology_id,
        prepared.target_topology_id,
        action_condition=condition,
        semantics="covariant-piola"
        if conformities == {"Hcurl"}
        else "contravariant-piola",
    )
    field = FiniteElementFieldTransfer(
        transfer,
        source.field_spaces[si],
        target.field_spaces[ti],
        deformation_binding(prepared),
        evidence,
    )
    return PreparedSurfaceChartCompatibleTransfer(
        field,
        _material_sparse_blocks(
            source_differentials, old_offset, ss, "material-chart-source-differential"
        ),
        _material_sparse_blocks(
            target_differentials, new_offset, ts, "material-chart-target-differential"
        ),
        _material_sparse_blocks(
            companion_blocks, new_offset, old_offset, "material-chart-two-form-companion"
        ),
    )


def _admit_fraction_action(
    work: PreparationWork,
    values: tuple[Fraction, ...],
    operations: int,
    coefficients: int,
) -> None:
    bits = (
        4
        * sum(
            abs(value.numerator).bit_length() + value.denominator.bit_length()
            for value in values
        )
        + 8
    )
    reserve_fraction_work(
        work.enclosure, operations + 2 * len(values), coefficients, bits
    )


def _material_affine_arguments(
    vertices: tuple[_Point, ...],
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[algebra.Expression, ...]:
    variables = algebra.axes(2, budget=budget)
    return tuple(
        algebra.add(
            algebra.constant(vertices[0][axis], 2, budget=budget),
            algebra.sum_polynomials(
                tuple(
                    algebra.scale(
                        variable,
                        vertices[column + 1][axis] - vertices[0][axis],
                        budget=budget,
                    )
                    for column, variable in enumerate(variables)
                ),
                budget=budget,
            ),
            budget=budget,
        )
        for axis in range(2)
    )


def _observe_material_affine_condition(
    source: tuple[_Point, ...],
    target: tuple[_Point, ...],
    work: PreparationWork,
    *,
    budget: CoordinateEnclosureBudget,
) -> None:

    _admit_fraction_action(
        work, tuple(value for point in (*source, *target) for value in point), 8, 8
    )
    source_frame = tuple(
        tuple(source[column + 1][axis] - source[0][axis] for axis in range(2))
        for column in range(2)
    )
    target_frame = tuple(
        tuple(target[column + 1][axis] - target[0][axis] for axis in range(2))
        for column in range(2)
    )
    prepared = imported_linalg__small_batched.prepare_exact_small_linear_actions(
        source_frame, target_frame, budget=work.enclosure
    )
    if prepared.actions is None:
        raise ValueError(
            "surface_chart.observe_material_affine_condition.reference_map must satisfy the regular reference-map contract; got Material references lack an regular affine action."
        )
    entries = tuple(value for row in prepared.actions for value in row)
    _admit_fraction_action(work, entries, 12, 12)
    determinant = abs(orientation(*target) / orientation(*source))
    upper = fraction_endpoint(
        sum((value * value for value in entries), Fraction(0)) / determinant,
        upper=True,
        saturate=True,
    )
    if not np.isfinite(upper) or upper > 1e12:
        raise ValueError(
            "surface_chart.observe_material_affine_condition.reference_map must satisfy the regular reference-map contract; got Material reference action exceeds its source whole-piece condition bound."
        )
    work.condition = max(work.condition, upper)


class _SurfaceMaterialPiecePreparation(StrictModule):
    """Prepare paired material actions with visible endpoint arrays."""

    prepared: PreparedSurfaceChartDeformation
    source_order: NDArray[np.int64]
    target_order: NDArray[np.int64]

    def __call__(self, work: PreparationWork) -> tuple[MaterialCompatiblePiece, ...]:
        pieces: list[MaterialCompatiblePiece] = []
        source_coverage = [Fraction(0) for _ in self.source_order]
        target_coverage = [Fraction(0) for _ in self.target_order]
        for occurrence in self.prepared.occurrences:
            old_rows = self.source_order[
                np.asarray(occurrence.source_rows, dtype=np.int64)
            ]
            new_rows = self.target_order[
                np.asarray(occurrence.target_rows, dtype=np.int64)
            ]
            for _, old_row, new_row, old_reference, new_reference in piece_maps(
                occurrence
            ):
                old = int(old_rows[old_row])
                new = int(new_rows[new_row])
                old_triangle = tuple(
                    (tuple((value for value in point)) for point in old_reference)
                )
                new_triangle = tuple(
                    (tuple((value for value in point)) for point in new_reference)
                )
                _admit_fraction_action(
                    work,
                    tuple((value for point in old_triangle for value in point)),
                    7,
                    7,
                )
                source_orientation = orientation(*old_triangle)
                if source_orientation == 0:
                    raise ValueError(
                        "surface_chart.prepare_pieces.reference_map must satisfy the regular reference-map contract; got A material piece has a singular source reference action."
                    )
                if source_orientation < 0:
                    old_triangle = (old_triangle[0], old_triangle[2], old_triangle[1])
                    new_triangle = (new_triangle[0], new_triangle[2], new_triangle[1])
                    source_orientation = -source_orientation
                _admit_fraction_action(
                    work,
                    tuple((value for point in new_triangle for value in point)),
                    7,
                    7,
                )
                image_orientation = orientation(*new_triangle)
                if image_orientation == 0:
                    raise ValueError(
                        "surface_chart.prepare_pieces.reference_map must satisfy the regular reference-map contract; got A material piece has a singular target reference action."
                    )
                source_coverage[old] += source_orientation / 2
                target_coverage[new] += abs(image_orientation) / 2
                arguments = _material_affine_arguments(
                    old_triangle, budget=work.enclosure
                )
                images = _material_affine_arguments(new_triangle, budget=work.enclosure)
                _observe_material_affine_condition(
                    old_triangle, new_triangle, work, budget=work.enclosure
                )
                work.enclosure.retain_basis((arguments, images))
                pieces.append((old, new, arguments, images))
        if any((area != Fraction(1, 2) for area in (*source_coverage, *target_coverage))):
            raise ValueError(
                "surface_chart.prepare_pieces.pieces must satisfy the material reference contract; got material pieces do not partition the full source and target physical references."
            )
        return tuple(pieces)


def _prepare_surface_chart_compatible_transfer(
    source: FiniteElementDiscretization,
    target: FiniteElementDiscretization,
    prepared: PreparedSurfaceChartDeformation,
    /,
    *,
    field_name: str,
    source_geometry: CellGeometrySpec,
    target_geometry: CellGeometrySpec,
    maximum_work: int = 100_000_000,
    maximum_storage_bytes: int = 256_000_000,
    budget: CoordinateEnclosureBudget,
) -> PreparedSurfaceChartCompatibleTransfer:
    "Integrate same-order one-forms on ordered material pieces.\n\n Physical element reference triangles and their paired affine actions\n are retained by the deformation owner; no UV reconstruction is performed."
    source_order, target_order = validate(
        source, target, prepared, source_geometry, target_geometry, budget=budget
    )

    prepare_pieces = _SurfaceMaterialPiecePreparation(
        prepared=prepared, source_order=source_order, target_order=target_order
    )

    return prepare_material_chart_compatible_transfer(
        source,
        target,
        prepared,
        field_name=field_name,
        prepare_pieces=prepare_pieces,
        maximum_work=maximum_work,
        maximum_storage_bytes=maximum_storage_bytes,
        budget=budget,
    )


__all__ = [
    "PreparedSurfaceChartCompatibleTransfer",
    "prepare_surface_chart_compatible_transfer",
]


@checked
def prepare_surface_chart_compatible_transfer(
    source: FiniteElementDiscretization,
    target: FiniteElementDiscretization,
    prepared: PreparedSurfaceChartDeformation,
    /,
    *,
    field_name: str,
    source_geometry: CellGeometrySpec,
    target_geometry: CellGeometrySpec,
    maximum_work: int = 100_000_000,
    maximum_storage_bytes: int = 256_000_000,
    budget: CoordinateEnclosureBudget | None = None,
) -> PreparedSurfaceChartCompatibleTransfer:
    "Prepare the request with one explicitly bounded host -work ticket."
    if budget is not None:
        return _prepare_surface_chart_compatible_transfer(
            source,
            target,
            prepared,
            field_name=field_name,
            source_geometry=source_geometry,
            target_geometry=target_geometry,
            maximum_work=maximum_work,
            maximum_storage_bytes=maximum_storage_bytes,
            budget=budget,
        )
    with HostBatchTicket(
        maximum_work,
        queries_upper=field_transfer_query_upper(
            maximum_work, source_geometry, target_geometry
        ),
        bytes_upper=maximum_storage_bytes,
    ) as ticket:
        budget = CoordinateEnclosureBudget(
            ticket,
            maximum_work_units=maximum_work,
            maximum_memory_bytes=maximum_storage_bytes,
        )
        with budget.activate():
            return _prepare_surface_chart_compatible_transfer(
                source,
                target,
                prepared,
                field_name=field_name,
                source_geometry=source_geometry,
                target_geometry=target_geometry,
                maximum_work=maximum_work,
                maximum_storage_bytes=maximum_storage_bytes,
                budget=budget,
            )
