# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Entity-functional actions on certified complete reference patches.

All calculations use exterior components, independently of the physical proxy.
The caller owns geometry/source certification; no physical point matching occurs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import lu_factor, lu_solve

from ...linalg import LinearSolveStatus
from .. import _coordinate_enclosure as algebra
from .._nested_reference import _NestedReferencePair, _PolynomialReferencePair
from .._reference_cell import reference_cell_topology
from ._exact_form_moments import (
    _identity_reference_pair,
    ExactFormMoments,
    MomentIntegralMatrix,
)
from ._form_elements import FormBasis
from ._generic import (
    FiniteElementDiscretization,
    FiniteElementTransferDiscretization,
)


type _NestedFiniteElementDiscretization = (
    FiniteElementDiscretization | FiniteElementTransferDiscretization
)


@dataclass(frozen=True)
class _FormCell:
    basis: FormBasis
    routes: NDArray[np.int64]
    transform: NDArray[np.float64]


def _form_cells(
    discretization: _NestedFiniteElementDiscretization,
    field_index: int,
) -> tuple[_FormCell, ...]:
    result = []
    dofs = discretization.dof_maps[field_index]
    for element, routes, transformations in zip(
        discretization.elements[field_index],
        dofs.cell_dofs,
        dofs.cell_transforms,
        strict=True,
    ):
        if element.form_basis is None:
            raise ValueError(
                "Nested compatible transfer requires canonical form moments."
            )
        for route, transformation in zip(
            np.asarray(routes), np.asarray(transformations), strict=True
        ):
            result.append(
                _FormCell(element.form_basis, route.astype(np.int64), transformation)
            )
    return tuple(result)


@dataclass(frozen=True)
class _HostMatrixFactor:
    matrix: NDArray[np.float64]
    spectrum: NDArray[np.float64]
    lu: NDArray[np.float64] | None
    pivots: NDArray[np.int32] | None

    def solve(self, values: NDArray[np.float64], /) -> NDArray[np.float64]:
        if self.lu is None or self.pivots is None:
            return np.linalg.solve(self.matrix, values)
        return np.asarray(lu_solve((self.lu, self.pivots), values), dtype=np.float64)

    def singular_values(self, /) -> NDArray[np.float64]:
        return self.spectrum


@dataclass
class _PreparationWork:
    maximum_work: int
    maximum_storage_bytes: int
    work: int = 0
    storage_bytes: int = 0
    condition: float = 1.0
    solve_defect: float = 0.0
    minimum_rank: float = float("inf")
    basis_solve_defect: float = 0.0
    integration_error: float = 0.0
    commuting_defect: float = 0.0
    projection_change: float = 0.0
    enclosure: algebra.CoordinateEnclosureBudget = field(init=False)
    matrix_factors: dict[
        tuple[tuple[int, ...], bytes], tuple[_HostMatrixFactor, float, float]
    ] = field(default_factory=dict)
    coordinate_budget: algebra.CoordinateEnclosureBudget | None = None
    _initial_coordinate_work: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        for value in (self.maximum_work, self.maximum_storage_bytes):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TypeError(
                    "Compatible preparation resource bounds must be integers."
                )
            if value < 1:
                raise ValueError(
                    "Compatible preparation requires positive resource budgets."
                )
        self.enclosure = (
            algebra.CoordinateEnclosureBudget(
                self.maximum_work, self.maximum_storage_bytes
            )
            if self.coordinate_budget is None
            else self.coordinate_budget
        )
        self._initial_coordinate_work = self.enclosure.work_units

    def charge(self, operations: int, entries: int = 0) -> None:
        if self.coordinate_budget is not None:
            self.enclosure.reserve(operations, entries * np.dtype(np.float64).itemsize)
            self.work += operations
            self.storage_bytes += entries * np.dtype(np.float64).itemsize
            return
        self.work += operations
        self.storage_bytes += entries * np.dtype(np.float64).itemsize
        if (
            self.work > self.maximum_work
            or self.storage_bytes > self.maximum_storage_bytes
        ):
            raise ValueError(
                "Compatible reference moment preparation exhausted its cumulative budget."
            )
        self.enclosure.maximum_work_units = self.maximum_work - self.work
        self.enclosure.maximum_memory_bytes = (
            self.maximum_storage_bytes - self.storage_bytes
        )
        if (
            self.enclosure.work_units > self.enclosure.maximum_work_units
            or self.enclosure.peak_bytes_upper > self.enclosure.maximum_memory_bytes
        ):
            raise ValueError(
                "Compatible reference moments exhausted the combined numerical/coefficient budget."
            )

    @property
    def preparation_work(self) -> int:
        if self.coordinate_budget is not None:
            return self.enclosure.work_units - self._initial_coordinate_work
        return self.work + self.enclosure.work_units

    @property
    def preparation_storage_upper(self) -> int:
        if self.coordinate_budget is not None:
            return self.enclosure.peak_bytes_upper
        return self.storage_bytes + self.enclosure.peak_bytes_upper

    def observe_basis(self, basis: FormBasis) -> None:
        factors = basis.hybrid_factors
        if factors is None:
            return
        rank = float(np.min(np.asarray(factors.dual_rank)))
        condition = float(np.max(np.asarray(factors.dual_condition)))
        if (
            rank != basis.local_dof_count
            or not np.all(np.asarray(factors.dual_status) == LinearSolveStatus.SUCCESS)
            or not np.isfinite(condition)
            or condition > 1e12
        ):
            raise ValueError(
                "Compatible basis has failed unisolvence or conditioning evidence."
            )
        self.condition = max(self.condition, condition)
        self.minimum_rank = min(self.minimum_rank, rank)
        self.basis_solve_defect = max(
            self.basis_solve_defect, float(np.max(np.asarray(factors.dual_solve_error)))
        )

    def factor_matrix(self, matrix: NDArray[np.float64]) -> _HostMatrixFactor:
        key = (matrix.shape, matrix.tobytes())
        prepared = self.matrix_factors.get(key)
        if prepared is None:
            width = matrix.shape[-1]
            self.charge(width * matrix.size, 3 * matrix.size)
            spectrum = np.linalg.svd(matrix, compute_uv=False)
            threshold = (
                spectrum[..., :1] * np.finfo(np.float64).eps * max(matrix.shape[-2:])
            )
            ranks = np.count_nonzero(spectrum > threshold, axis=-1)
            if np.any(ranks != width) or np.any(spectrum[..., -1] <= 0):
                raise ValueError("Compatible chart/entity transformation is singular.")
            condition = float(np.max(spectrum[..., 0] / spectrum[..., -1]))
            if not np.isfinite(condition) or condition > 1e12:
                raise ValueError(
                    "Compatible chart/entity transformation is ill conditioned."
                )
            rank = float(np.min(ranks))
            if matrix.ndim == 2:
                lu, pivots = lu_factor(matrix)
                factor = _HostMatrixFactor(
                    matrix,
                    spectrum,
                    np.asarray(lu, dtype=np.float64),
                    np.asarray(pivots, dtype=np.int32),
                )
            else:
                factor = _HostMatrixFactor(matrix, spectrum, None, None)
            prepared = factor, rank, condition
            self.matrix_factors[key] = prepared
        factor, rank, condition = prepared
        self.condition = max(self.condition, condition)
        self.minimum_rank = min(self.minimum_rank, rank)
        return factor

    def orientation_solve(
        self,
        transform: NDArray[np.float64],
        moments: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        self.charge(transform.size * moments.shape[1], moments.size)
        factor = self.factor_matrix(transform)
        values = np.asarray(factor.solve(moments), dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError("Compatible entity transformation solve failed.")
        scale = max(float(np.max(np.abs(moments), initial=0)), 1.0)
        self.solve_defect = max(
            self.solve_defect,
            float(np.max(np.abs(transform @ values - moments), initial=0)) / scale,
        )
        return values


def _derivative_cell(
    cell: _FormCell,
    prepared: dict[int, tuple[FormBasis, NDArray[np.float64]]],
) -> _FormCell:
    basis = cell.basis
    key = id(basis)
    if key not in prepared:
        order = max(basis.order - 1, 0) if basis.family == "full" else basis.order
        following = FormBasis(
            basis.dimension, basis.form_degree + 1, order, basis.family, basis.twist
        )
        differential = np.asarray(
            basis.exterior_derivative_matrix(following), dtype=np.float64
        )
        prepared[key] = following, differential
    following, differential = prepared[key]
    return _FormCell(following, cell.routes, differential @ cell.transform)


def _embed_columns(
    values: NDArray[np.float64],
    routes: NDArray[np.int64],
    columns: NDArray[np.int64],
    work: _PreparationWork,
) -> NDArray[np.float64]:
    work.charge(values.size, values.shape[0] * columns.size)
    result = np.zeros((values.shape[0], columns.size), dtype=np.float64)
    np.add.at(
        result,
        (np.arange(values.shape[0])[:, None], np.searchsorted(columns, routes)[None]),
        values,
    )
    return result


def _refined_moments(
    source: _FormCell,
    target: FormBasis,
    pair: _NestedReferencePair | _PolynomialReferencePair,
    work: _PreparationWork,
    exact: ExactFormMoments,
) -> MomentIntegralMatrix:
    work.charge(
        target.local_dof_count * source.basis.local_dof_count * source.routes.size,
        2 * target.local_dof_count * (source.basis.local_dof_count + source.routes.size),
    )
    with work.enclosure.temporary_scope():
        integral = exact.refine(source.basis, target, pair)
    return _transform_integrals(integral, source.transform)


def _coarsened_moments(
    sources: tuple[_FormCell, ...],
    target: FormBasis,
    pairs: tuple[_NestedReferencePair | _PolynomialReferencePair, ...],
    columns: NDArray[np.int64],
    work: _PreparationWork,
    exact: ExactFormMoments,
) -> tuple[MomentIntegralMatrix, float]:
    work.charge(0, 2 * target.local_dof_count * columns.size)
    result = np.zeros((target.local_dof_count, columns.size), dtype=np.float64)
    errors = np.zeros_like(result)
    continuity = 0.0
    for dimension, entities in enumerate(target.entity_vertices):
        for entity in entities:
            if not any(label[0] == entity for label in target.dof_labels):
                continue
            seen: dict[tuple[tuple[Fraction, ...], ...], MomentIntegralMatrix] = {}
            for pair in pairs:
                source = sources[pair.source_cell]
                for fine_entity in source.basis.entity_vertices[dimension]:
                    with work.enclosure.temporary_scope():
                        value = exact.coarse_entity(
                            source.basis, target, entity, fine_entity, pair
                        )
                        if value is None:
                            continue
                        integral, key = value
                        transformed = _transform_integrals(integral, source.transform)
                        contribution = MomentIntegralMatrix(
                            _embed_columns(
                                transformed.value, source.routes, columns, work
                            ),
                            _embed_columns(
                                transformed.error, source.routes, columns, work
                            ),
                        )
                        if key in seen:
                            scale = max(
                                float(np.max(np.abs(contribution.value))),
                                float(np.max(np.abs(seen[key].value))),
                                1.0,
                            )
                            difference = (
                                np.abs(contribution.value - seen[key].value)
                                + contribution.error
                                + seen[key].error
                            )
                            continuity = max(
                                continuity, float(np.max(difference)) / scale
                            )
                        else:
                            result += contribution.value
                            errors += contribution.error
                            seen[key] = contribution
            if not seen:
                raise ValueError(
                    "A complete-source patch leaves a coarse form entity unsupported."
                )
    gamma = np.finfo(np.float64).eps * max(len(pairs), 1)
    errors = np.nextafter(errors + gamma * np.abs(result) / (1 - gamma), np.inf)
    return MomentIntegralMatrix(result, errors), continuity


def _transform_integrals(
    integral: MomentIntegralMatrix,
    transform: NDArray[np.float64],
) -> MomentIntegralMatrix:
    gamma = np.finfo(np.float64).eps * transform.shape[0]
    absolute = np.abs(integral.value) @ np.abs(transform)
    error = integral.error @ np.abs(transform) + gamma * absolute / (1 - gamma)
    return MomentIntegralMatrix(integral.value @ transform, np.nextafter(error, np.inf))


def _certify_chart_condition(
    pair: _NestedReferencePair | _PolynomialReferencePair,
    work: _PreparationWork,
) -> None:
    if isinstance(pair, _NestedReferencePair):
        work.factor_matrix(pair.matrix)
        return
    dimension = len(pair.arguments)
    jacobian = tuple(
        tuple(algebra.derivative(value, axis) for axis in range(dimension))
        for value in pair.arguments
    )
    kind = pair.source_kind if pair.fine_is_source else pair.target_kind
    domain = "box" if kind == "quadrilateral" else "simplex"
    entry_bounds = [
        [
            max(
                abs(value)
                for value in algebra.bernstein_coefficients(entry, domain, dimension)
            )
            for entry in row
        ]
        for row in jacobian
    ]
    norm = max(sum(row, Fraction(0)) for row in entry_bounds)
    inverse_norm = Fraction(0)
    for column in range(dimension):
        total = Fraction(0)
        for row in range(dimension):
            minor = algebra.expression_determinant(
                tuple(
                    tuple(jacobian[i][j] for j in range(dimension) if j != column)
                    for i in range(dimension)
                    if i != row
                ),
                variable_dimension=dimension,
            )
            total += max(
                abs(value)
                for value in algebra.expression_bernstein_coefficients(
                    minor, domain, dimension
                )
            )
        inverse_norm = max(inverse_norm, total)
    if pair.jacobian_bounds[0] <= 0:
        raise ValueError(
            "Compatible polynomial chart has no positive whole-cell Jacobian bound."
        )
    condition = algebra.outward(norm * inverse_norm / pair.jacobian_bounds[0], np.inf)
    if not np.isfinite(condition) or condition > 1e12:
        raise ValueError(
            "Compatible polynomial chart exceeds its whole-cell condition bound."
        )
    work.condition = max(work.condition, condition)
    vertices = reference_cell_topology(kind).vertices
    matrices = np.asarray(
        [
            [
                [
                    float(
                        algebra.evaluate(entry, tuple(Fraction(float(x)) for x in vertex))
                    )
                    for entry in row
                ]
                for row in jacobian
            ]
            for vertex in vertices
        ],
        dtype=np.float64,
    )
    work.factor_matrix(matrices)


def _derivative_sources(
    sources: tuple[_FormCell, ...],
    prepared: dict[int, tuple[FormBasis, NDArray[np.float64]]],
    cache: dict[int, tuple[_FormCell, ...]],
) -> tuple[_FormCell, ...]:
    key = id(sources)
    if key not in cache:
        cache[key] = tuple(_derivative_cell(value, prepared) for value in sources)
    return cache[key]


def _commuting_projection(
    sources: tuple[_FormCell, ...],
    target: FormBasis,
    pairs: tuple[_NestedReferencePair | _PolynomialReferencePair, ...],
    columns: NDArray[np.int64],
    work: _PreparationWork,
    exact: ExactFormMoments,
    prepared: dict[int, tuple[FormBasis, NDArray[np.float64]]],
    source_derivatives: dict[int, tuple[_FormCell, ...]],
) -> tuple[MomentIntegralMatrix, float]:
    integral, continuity = _coarsened_moments(
        sources, target, pairs, columns, work, exact
    )
    if target.form_degree == target.dimension:
        return integral, continuity
    following = _derivative_cell(
        _FormCell(
            target,
            np.zeros(target.local_dof_count, dtype=np.int64),
            np.eye(target.local_dof_count, dtype=np.float64),
        ),
        prepared,
    )
    work.observe_basis(following.basis)
    derivative_sources = _derivative_sources(sources, prepared, source_derivatives)
    expected, defect = _commuting_projection(
        derivative_sources,
        following.basis,
        pairs,
        columns,
        work,
        exact,
        prepared,
        source_derivatives,
    )
    continuity = max(continuity, defect)
    return _correct_commuting_moments(
        target, integral, expected, following.transform, columns, work
    ), continuity


def _correct_commuting_moments(
    target: FormBasis,
    integral: MomentIntegralMatrix,
    expected: MomentIntegralMatrix,
    differential: NDArray[np.float64],
    columns: NDArray[np.int64],
    work: _PreparationWork,
) -> MomentIntegralMatrix:
    """Preserve actual boundary moments while correcting admitted interior residuals."""
    from ._topology_transfer import _CLAIM_ULPS

    # Both actions already use the target chart and canonical source columns.
    # A column sign choice here would compare different global fields. In
    # particular, retwisting an H(div) face trace erases its boundary-incidence
    # sign and exposes an O(1) Stokes residual (one observed mesh gives 1/4),
    # rather than a commuting defect that may be aligned after assembly.
    expected_value = expected.value
    residual = expected_value - differential @ integral.value
    interior = [
        row
        for row, label in enumerate(target.dof_labels)
        if label[0] == target.entity_vertices[-1][0]
    ]
    corrected = integral.value.copy()
    errors = integral.error.copy()
    residual_error = expected.error + np.abs(differential) @ integral.error
    residual_error += (
        np.finfo(np.float64).eps
        * differential.shape[1]
        * (np.abs(differential) @ np.abs(integral.value))
    )
    scale = max(
        float(np.max(np.abs(expected_value), initial=0)),
        float(np.max(np.abs(integral.value), initial=0)),
        1.0,
    )
    roundoff_allowance = (
        _CLAIM_ULPS * np.finfo(np.float64).eps * work.condition * columns.size * scale
    )
    # Avoid amplifying a constraint residual already admitted by the original
    # roundoff policy; its actual residual and integral enclosure remain checked.
    if interior and np.any(
        np.abs(residual) > np.maximum(residual_error, roundoff_allowance)
    ):
        constraint = differential[:, interior]
        work.charge(
            min(constraint.shape) * constraint.size + constraint.size * columns.size,
            3 * constraint.size + 2 * len(interior) * columns.size,
        )
        left, spectrum, right = np.linalg.svd(constraint, full_matrices=False)
        rank = int(
            np.sum(spectrum > (0.0 if not spectrum.size else spectrum[0] * 1.0e-12))
        )
        if rank:
            condition = float(spectrum[0] / spectrum[rank - 1])
            if not np.isfinite(condition) or condition > 1e12:
                raise ValueError(
                    "Compatible interior commuting projection is ill conditioned."
                )
            correction = right[:rank].T @ (
                (left[:, :rank].T @ residual) / spectrum[:rank, None]
            )
            if not np.all(np.isfinite(correction)):
                raise ValueError("Compatible interior commuting projection solve failed.")
            corrected[interior] += correction
            uncertainty = (
                np.sqrt(len(interior) * residual_error.shape[0])
                * np.max(residual_error, axis=0)
                / spectrum[rank - 1]
            )
            errors[interior] += uncertainty[None]
            work.condition = max(work.condition, condition)
            work.minimum_rank = min(work.minimum_rank, float(rank))
            work.projection_change = max(
                work.projection_change, float(np.max(np.abs(correction), initial=0))
            )
    actual = differential @ corrected
    roundoff = (
        np.finfo(np.float64).eps
        * differential.shape[1]
        * (np.abs(differential) @ np.abs(corrected))
    )
    uncertainty = np.abs(differential) @ errors + expected.error + roundoff
    scale = max(
        float(np.max(np.abs(expected_value), initial=0)),
        float(np.max(np.abs(actual), initial=0)),
        1.0,
    )
    work.commuting_defect = max(
        work.commuting_defect,
        float(np.max(np.abs(actual - expected_value) + uncertainty, initial=0)) / scale,
    )
    return MomentIntegralMatrix(corrected, np.nextafter(errors, np.inf))


def prepare_form_patch_actions(
    source: _NestedFiniteElementDiscretization,
    target: _NestedFiniteElementDiscretization,
    source_index: int,
    target_index: int,
    pairs: tuple[_NestedReferencePair | _PolynomialReferencePair, ...],
    /,
    *,
    maximum_work: int = 100_000_000,
    maximum_storage_bytes: int = 256_000_000,
) -> tuple[
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.float64],
    dict[str, float],
    dict[str, float],
]:
    """Return sparse candidate rows, full functional/commuting defects and work."""
    work = _PreparationWork(maximum_work, maximum_storage_bytes)
    sources, targets = (
        _form_cells(source, source_index),
        _form_cells(target, target_index),
    )
    identities = {
        (
            element.value_spec.value_spec_id,
            element.degree,
            element.representation,
            "full" if element.family == "full" else "trimmed",
        )
        for space, index in ((source, source_index), (target, target_index))
        for element in space.elements[index]
    }
    if len(identities) != 1:
        raise ValueError(
            "Compatible transfer changes form degree, twist, proxy, polynomial order, or family."
        )
    work.charge(len(sources) + len(targets), 1)
    exact = ExactFormMoments(work.enclosure)
    with work.enclosure.activate():
        for cell in sources + targets:
            work.observe_basis(cell.basis)
        for pair in pairs:
            _certify_chart_condition(pair, work)
        return _patch_rows(sources, targets, pairs, work, exact)


def _patch_rows(
    sources: tuple[_FormCell, ...],
    targets: tuple[_FormCell, ...],
    pairs: tuple[_NestedReferencePair | _PolynomialReferencePair, ...],
    work: _PreparationWork,
    exact: ExactFormMoments,
) -> tuple[
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.float64],
    dict[str, float],
    dict[str, float],
]:
    rows, routes, coefficients = [], [], []
    continuity = 0.0
    prepared: dict[int, tuple[FormBasis, NDArray[np.float64]]] = {}
    source_derivatives: dict[int, tuple[_FormCell, ...]] = {}
    for cell_index, cell in enumerate(targets):
        patch = tuple(pair for pair in pairs if pair.target_cell == cell_index)
        identity = len(patch) == 1 and _identity_reference_pair(patch[0])
        refinement = len(patch) == 1 and (not patch[0].fine_is_source or identity)
        if not patch or (
            not refinement and not all(pair.fine_is_source for pair in patch)
        ):
            raise ValueError(
                "Compatible patch must be one nested parent or complete source siblings."
            )
        columns = np.unique(
            np.concatenate([sources[pair.source_cell].routes for pair in patch])
        )
        if refinement:
            pair = patch[0]
            local = _refined_moments(
                sources[pair.source_cell], cell.basis, pair, work, exact
            )
            raw = MomentIntegralMatrix(
                _embed_columns(
                    local.value, sources[pair.source_cell].routes, columns, work
                ),
                _embed_columns(
                    local.error, sources[pair.source_cell].routes, columns, work
                ),
            )
            if cell.basis.form_degree < cell.basis.dimension:
                following = _derivative_cell(
                    _FormCell(
                        cell.basis,
                        cell.routes,
                        np.eye(cell.basis.local_dof_count, dtype=np.float64),
                    ),
                    prepared,
                )
                work.observe_basis(following.basis)
                derivative_sources = _derivative_sources(
                    sources, prepared, source_derivatives
                )
                expected = _refined_moments(
                    derivative_sources[pair.source_cell],
                    following.basis,
                    pair,
                    work,
                    exact,
                )
                expected = MomentIntegralMatrix(
                    _embed_columns(
                        expected.value, sources[pair.source_cell].routes, columns, work
                    ),
                    _embed_columns(
                        expected.error, sources[pair.source_cell].routes, columns, work
                    ),
                )
                # The source/target moment bases already share one global
                # orientation; commuting is the direct Stokes equality.
                actual = following.transform @ raw.value
                expected_value = expected.value
                uncertainty = np.abs(following.transform) @ raw.error + expected.error
                scale = max(
                    float(np.max(np.abs(expected_value), initial=0)),
                    float(np.max(np.abs(actual), initial=0)),
                    1.0,
                )
                work.commuting_defect = max(
                    work.commuting_defect,
                    float(
                        np.max(np.abs(actual - expected_value) + uncertainty, initial=0)
                    )
                    / scale,
                )
        else:
            raw, defect = _commuting_projection(
                sources,
                cell.basis,
                patch,
                columns,
                work,
                exact,
                prepared,
                source_derivatives,
            )
            continuity = max(continuity, defect)
        local = work.orientation_solve(cell.transform, raw.value)
        spectrum = np.asarray(work.factor_matrix(cell.transform).singular_values())
        work.integration_error = max(
            work.integration_error,
            float(np.max(raw.error, initial=0))
            * np.sqrt(cell.basis.local_dof_count)
            / float(spectrum[-1]),
        )
        rows.extend(cell.routes.tolist())
        routes.extend(np.broadcast_to(columns, local.shape))
        coefficients.extend(local)
    width = max(len(row) for row in routes)
    work.charge(0, 2 * len(rows) * width)
    padded_routes = np.zeros((len(rows), width), dtype=np.int64)
    padded_values = np.zeros((len(rows), width), dtype=np.float64)
    for row, (indices, values) in enumerate(zip(routes, coefficients, strict=True)):
        padded_routes[row, : indices.size], padded_values[row, : values.size] = (
            indices,
            values,
        )
    return (
        np.asarray(rows, dtype=np.int64),
        padded_routes,
        padded_values,
        {
            "commuting": work.commuting_defect,
            "patch_trace": continuity,
            "moment_solve": work.solve_defect,
            "basis_duality": work.basis_solve_defect,
            "integration": work.integration_error,
        },
        {
            "moment_condition": work.condition,
            "minimum_local_rank": work.minimum_rank,
            "projection_moment_change": work.projection_change,
            "preparation_work": float(work.preparation_work),
            "preparation_storage_bytes": float(work.preparation_storage_upper),
        },
    )
