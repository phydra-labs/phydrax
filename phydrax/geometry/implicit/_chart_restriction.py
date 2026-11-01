#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Implicit root-chart fidelity of an adapted mesh from its certified parent.

A parent certificate proved by the implicit root-chart prover retains its
per-row chart evidence (:class:`ImplicitChartRows`), keyed by the exact inputs
of each row's proof: the scaffold root cell, the root-edge triangle, the exact
source-parameter triangle and the exact target face polynomial in its stored
term order. A target row with an equal key has an equal proof, so the target
certificate reuses it by identity and proves only the other rows; the result
is the certificate a full proof produces.

Theorem (chart restriction). Let a certified exact
:class:`CellGeometryTransition` bind the parent ``(S, F_S)`` to the target
``(T, F_T)`` by affine reference maps ``A_t`` with ``F_T|_t = F_S|_{p(t)} o
A_t``, and let the parent fidelity certificate be certified for the same
source, tolerance and normal limit with per-row upper bounds ``U_r`` (target to
root distance, already at least the reverse bound) and normal bounds ``N_r``.
Suppose every target chart row ``t`` (cell ``c``, reference triangle ``R_t``)
has a parent row ``r`` of ``p(c)`` with the same scaffold root cell,
root-edge triangle and source chart such that

1. the barycentric matrix ``B_t`` of ``A_c(R_t)`` in ``R_r`` has nonnegative
   entries and positive determinant;
2. the source-parameter triangle of ``t`` is ``B_t`` applied to that of ``r``;
3. per parent row the determinants sum to one, and every oriented child edge
   not contained in an edge of ``R_r`` cancels against an opposite one.

By (3) the children of ``r`` form an oriented chain whose boundary lies in
``partial R_r``; its degree is constant on the open triangle and zero outside,
so (1) and the area identity make it one: the children tile ``R_r``. Hence
``F_T(R_t) = F_S(A_c(R_t))`` is contained in ``F_S(R_r)`` and the target
image of all rows equals the parent image. Every quantity the parent bounds
is a function of the physical point and its scaffold root cell (the gradient
line lift, its distance, the source normal and, by (2) with ``det B_t > 0``,
the oriented target normal), so ``U_r`` and ``N_r`` bound row ``t``; the
parent's directed distance and normal bounds bound the target, whose image
and lifts are the parent's. Restrictions of injective charts are injective.
Lower bounds of a row are kept only where ``B_t`` is the identity.

The certifier answers ``None`` when a premise does not hold; the caller then
proves the rows, so the verdict is never weakened. Continuity of the target
map across cells is the embedding certificate's claim, not this one's.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

import numpy as np

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ...discretization._cell_geometry import CellGeometrySpec
from ...discretization._cell_geometry_transition_types import (
    _nested_reference_groups,
    _reference_map_groups,
    CellGeometryTransition,
)
from ...discretization._cell_geometry_validity import cell_geometry_id
from ...discretization._cell_mesh import CellMesh
from ...discretization._nested_reference_types import exact_reference_cell_vertices
from ...discretization._reference_cell import reference_cell_topology
from .._nested_restriction_embedding import _unit_simplex
from .._source_boundary_records import SourceBoundaryChartWitness, SourceBoundaryQuery


if TYPE_CHECKING:
    from .._source_fidelity_certificate import SourceFidelityCertificate


CHART_RESTRICTION_ROUTE = "nested-chart-restriction"
_ZERO = Fraction(0)
_ONE = Fraction(1)
# Columns of ``ImplicitChartRows.values``.
CHART_ROW_COLUMNS = (
    "field_residual_bound",
    "target_to_root_lower_bound",
    "target_to_root_bound",
    "source_normal_deviation_lower",
    "source_normal_deviation_upper",
    "chart_jacobian_lower",
    "chart_injectivity_margin",
    "source_gradient_lower",
)


class ImplicitChartRows(StrictModule):
    """Per-row certified chart evidence of one certificate, keyed by proof inputs.

    ``keys[i]`` is the 16-byte digest of row ``i``'s proof inputs and
    ``reusable[i]`` marks rows proved certified with such a key; ``values``
    holds their evidence in :data:`CHART_ROW_COLUMNS` order.
    """

    keys: np.ndarray
    reusable: np.ndarray
    values: np.ndarray
    rows_id: str

    def __init__(
        self, keys: np.ndarray, reusable: np.ndarray, values: np.ndarray, /
    ) -> None:
        keys_ = np.array(keys, dtype=np.uint8, order="C", copy=True)
        reusable_ = np.array(reusable, dtype=np.bool_, order="C", copy=True)
        values_ = np.array(values, dtype=np.float64, order="C", copy=True)
        count = reusable_.size
        if (
            keys_.shape != (count, 16)
            or reusable_.ndim != 1
            or values_.shape != (count, len(CHART_ROW_COLUMNS))
        ):
            raise ValueError("Implicit chart rows must align with their keys.")
        for array in (keys_, reusable_, values_):
            array.setflags(write=False)
        self.keys = keys_
        self.reusable = reusable_
        self.values = values_
        self.rows_id = self._identity(keys_, reusable_, values_)

    @staticmethod
    def _identity(keys: np.ndarray, reusable: np.ndarray, values: np.ndarray, /) -> str:
        return canonical_fingerprint(
            {
                "kind": "implicit-chart-rows",
                "rows": array_tree_fingerprint((keys, reusable, values)),
            }
        )

    def require_current(self, /) -> None:
        if self.rows_id != self._identity(self.keys, self.reusable, self.values):
            raise ValueError("Retained implicit chart rows differ from their identity.")

    def lookup(self, /) -> dict[bytes, int]:
        """Reusable row per key; the first row of a repeated key wins."""
        result: dict[bytes, int] = {}
        for row in np.flatnonzero(self.reusable).tolist():
            result.setdefault(self.keys[row].tobytes(), row)
        return result


@dataclass(frozen=True, slots=True)
class ValidatedImplicitChartParent:
    """Host-only mathematical replay bound to the exact parent operands."""

    mesh: CellMesh
    geometry: CellGeometrySpec
    source: SourceBoundaryQuery
    certificate: SourceFidelityCertificate

    def __post_init__(self) -> None:
        self.certificate.revalidate(self.mesh, self.geometry, self.source)


class ImplicitChartPremise(StrictModule):
    """A parent fidelity certificate and, optionally, the exact transition to the target."""

    parent_mesh: CellMesh
    parent_geometry: CellGeometrySpec
    parent_certificate: SourceFidelityCertificate
    transition: CellGeometryTransition | None

    def __init__(
        self,
        parent_mesh: CellMesh,
        parent_geometry: CellGeometrySpec,
        parent_certificate: SourceFidelityCertificate,
        transition: CellGeometryTransition | None,
        /,
    ) -> None:
        from .._source_fidelity_certificate import SourceFidelityCertificate

        if not isinstance(parent_mesh, CellMesh):
            raise TypeError("parent_mesh must be CellMesh.")
        if not isinstance(parent_geometry, CellGeometrySpec):
            raise TypeError("parent_geometry must be CellGeometrySpec.")
        if type(parent_certificate) is not SourceFidelityCertificate:
            raise TypeError("parent_certificate must be SourceFidelityCertificate.")
        if transition is not None and not isinstance(transition, CellGeometryTransition):
            raise TypeError("transition must be CellGeometryTransition or None.")
        self.parent_mesh = parent_mesh
        self.parent_geometry = parent_geometry
        self.parent_certificate = parent_certificate
        self.transition = transition

    def certified_parent(
        self,
        source: SourceBoundaryQuery,
        tolerance: float,
        maximum_normal_deviation: float | None,
        *,
        validated: ValidatedImplicitChartParent | None = None,
    ) -> SourceFidelityCertificate | None:
        """The parent certificate when it answers this source and these controls."""
        parent = self.parent_certificate
        if validated is None:
            parent.revalidate(self.parent_mesh, self.parent_geometry, source)
        else:
            if (
                validated.mesh is not self.parent_mesh
                or validated.geometry is not self.parent_geometry
                or validated.source is not source
                or validated.certificate is not parent
            ):
                raise ValueError("Validated chart parent binds different exact operands.")
            parent.require_current(self.parent_mesh, self.parent_geometry, source)
        normal = (
            None if maximum_normal_deviation is None else float(maximum_normal_deviation)
        )
        if (
            parent.tolerance != float(tolerance)
            or parent.maximum_normal_deviation != normal
            or not isinstance(parent.witness, SourceBoundaryChartWitness)
            or parent.witness.chart_dimension != 2
        ):
            return None
        return parent


@dataclass(frozen=True, slots=True)
class ImplicitChartRestrictionEvidence:
    """Exact per-parent-row tiling evidence of one chart restriction."""

    route: str
    mesh_id: str
    topology_id: str
    geometry_id: str
    witness_id: str
    parent_certificate_id: str
    transition_id: str
    parent_row_count: int
    target_row_count: int
    identity_row_count: int
    evidence_id: str


@dataclass(frozen=True, slots=True)
class ImplicitChartRestriction:
    """The proved restriction: evidence and the parent row of every target row."""

    evidence: ImplicitChartRestrictionEvidence
    parent_rows: np.ndarray
    identity_rows: np.ndarray


def _determinant3(rows: tuple[tuple[Fraction, ...], ...], /) -> Fraction:
    (a, b, c), (d, e, f), (g, h, i) = rows
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def _barycentric_solver(
    corners: tuple[tuple[Fraction, ...], ...], /
) -> tuple[tuple[int, int], tuple[Fraction, Fraction, Fraction, Fraction]] | None:
    """Two coordinate axes with a nonzero minor of the triangle and its inverse."""
    origin = corners[0]
    first = tuple(value - base for value, base in zip(corners[1], origin, strict=True))
    second = tuple(value - base for value, base in zip(corners[2], origin, strict=True))
    dimension = len(origin)
    for axis_a in range(dimension):
        for axis_b in range(axis_a + 1, dimension):
            minor = first[axis_a] * second[axis_b] - second[axis_a] * first[axis_b]
            if minor:
                return (axis_a, axis_b), (
                    second[axis_b] / minor,
                    -second[axis_a] / minor,
                    -first[axis_b] / minor,
                    first[axis_a] / minor,
                )
    return None


def _barycentric(
    point: tuple[Fraction, ...],
    corners: tuple[tuple[Fraction, ...], ...],
    solver: tuple[tuple[int, int], tuple[Fraction, Fraction, Fraction, Fraction]],
    /,
) -> tuple[Fraction, Fraction, Fraction] | None:
    """Exact barycentric coordinates of ``point`` in the triangle's plane, else ``None``."""
    (axis_a, axis_b), (m00, m01, m10, m11) = solver
    origin = corners[0]
    delta_a = point[axis_a] - origin[axis_a]
    delta_b = point[axis_b] - origin[axis_b]
    second = m00 * delta_a + m01 * delta_b
    third = m10 * delta_a + m11 * delta_b
    for axis in range(len(origin)):
        if axis in (axis_a, axis_b):
            continue
        value = (
            origin[axis]
            + second * (corners[1][axis] - origin[axis])
            + third * (corners[2][axis] - origin[axis])
        )
        if value != point[axis]:
            return None
    return (_ONE - second - third, second, third)


def _chart_points(
    tokens: tuple[tuple[tuple[int, int], ...], ...], /
) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(
        tuple(Fraction(numerator, denominator) for numerator, denominator in point)
        for point in tokens
    )


def _row_identity(witness: SourceBoundaryChartWitness, row: int, /) -> tuple[object, ...]:
    return (
        np.asarray(witness.implicit_cell_keys[row]).tobytes(),
        np.asarray(witness.implicit_root_edge_keys[row]).tobytes(),
        witness.source_chart_ids[row],
        witness.source_entity_ids[row],
        witness.occurrence_paths[row],
    )


def _transition_maps(
    transition: CellGeometryTransition,
    parent_mesh: CellMesh,
    mesh: CellMesh,
    /,
) -> dict[int, tuple[int, tuple[tuple[Fraction, ...], ...]]] | None:
    """Target cell ID -> (parent cell ID, exact child corners in parent reference)."""
    if (
        transition.evidence.kind != "nested_restriction"
        or (transition.reference_map is None and transition.refinement is None)
        or transition.coarsening is not None
    ):
        return None
    if transition.reference_map is not None:
        groups = _reference_map_groups(transition.reference_map)
    elif transition.refinement is not None:
        groups = _nested_reference_groups(transition.refinement, parent_mesh, mesh)
    else:
        return None
    mapped: dict[int, tuple[int, tuple[tuple[Fraction, ...], ...]]] = {}
    for group in groups:
        if (
            group.target_cell_kind != group.source_cell_kind
            or group.target_cell_kind not in ("triangle", "tetrahedron")
            or group.source_reference_chart_arguments is not None
        ):
            return None
        exact = group.source_reference_exact_vertices
        if exact is None:
            exact = exact_reference_cell_vertices(group)
        for target_id, source_id, corners in zip(
            np.asarray(group.target_cell_ids, dtype=np.int64).tolist(),
            np.asarray(group.source_cell_ids, dtype=np.int64).tolist(),
            exact,
            strict=True,
        ):
            if target_id in mapped:
                return None
            mapped[target_id] = (source_id, tuple(tuple(point) for point in corners))
    return mapped


def certify_implicit_chart_restriction(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    witness: SourceBoundaryChartWitness,
    parent: SourceFidelityCertificate,
    premise: ImplicitChartPremise,
    /,
) -> ImplicitChartRestriction | None:
    """Prove the module theorem's premises exactly, or ``None`` if one fails.

    ``parent`` is ``premise``'s certificate, already checked against the
    request by :meth:`ImplicitChartPremise.certified_parent`.
    """
    transition = premise.transition
    parent_witness = parent.witness
    if (
        transition is None
        or parent.status != "certified"
        or parent.findings
        or not isinstance(parent_witness, SourceBoundaryChartWitness)
        or witness.chart_dimension != 2
        or witness.reference_dimension != parent_witness.reference_dimension
        or parent.normal_deviation_bounds is None
    ):
        return None
    transition.require_bound(premise.parent_mesh, premise.parent_geometry, mesh, geometry)
    if any(
        not _unit_simplex(
            block.cell_kind, reference_cell_topology(block.cell_kind).dimension
        )
        for block in (*mesh.blocks, *premise.parent_mesh.blocks)
    ):
        return None
    mapped = _transition_maps(transition, premise.parent_mesh, mesh)
    if mapped is None:
        return None
    parent_rows_by_cell: dict[int, list[int]] = {}
    for row, cell in enumerate(
        np.asarray(parent_witness.cell_global_ids, dtype=np.int64).tolist()
    ):
        parent_rows_by_cell.setdefault(cell, []).append(row)
    parent_identity = [
        _row_identity(parent_witness, row)
        for row in range(len(parent_witness.source_chart_ids))
    ]
    parent_reference_tokens = parent_witness.target_reference_rational_tokens
    parent_parameter_tokens = parent_witness.source_parameter_rational_tokens
    parent_corners: dict[int, tuple[tuple[Fraction, ...], ...]] = {}
    parent_solvers: dict[
        int,
        tuple[tuple[int, int], tuple[Fraction, Fraction, Fraction, Fraction]] | None,
    ] = {}
    parent_parameters: dict[int, tuple[tuple[Fraction, ...], ...]] = {}
    # Exact barycentric matrices shared by children of one shape.
    matrices: dict[
        tuple[object, ...],
        tuple[tuple[tuple[Fraction, ...], ...], Fraction] | None,
    ] = {}
    area: dict[int, Fraction] = {}
    edges: dict[int, dict[tuple[tuple[Fraction, ...], ...], int]] = {}
    target_cells = np.asarray(witness.cell_global_ids, dtype=np.int64).tolist()
    count = len(target_cells)
    parent_rows = np.empty((count,), dtype=np.int64)
    identity_rows = np.zeros((count,), dtype=np.bool_)
    identity_matrix = ((_ONE, _ZERO, _ZERO), (_ZERO, _ONE, _ZERO), (_ZERO, _ZERO, _ONE))
    for row, cell in enumerate(target_cells):
        owner = mapped.get(cell)
        if owner is None:
            return None
        parent_cell, child_corners = owner
        identity = _row_identity(witness, row)
        reference_tokens = witness.target_reference_rational_tokens[row]
        selected: tuple[int, tuple[tuple[Fraction, ...], ...], Fraction] | None = None
        for candidate in parent_rows_by_cell.get(parent_cell, ()):
            if parent_identity[candidate] != identity:
                continue
            key = (parent_reference_tokens[candidate], child_corners, reference_tokens)
            if key not in matrices:
                corners = parent_corners.get(candidate)
                if corners is None:
                    corners = parent_corners[candidate] = _chart_points(
                        parent_reference_tokens[candidate]
                    )
                solver = parent_solvers.get(candidate)
                if solver is None:
                    solver = parent_solvers[candidate] = _barycentric_solver(corners)
                matrix = None
                if solver is not None:
                    origin = child_corners[0]
                    rows: list[tuple[Fraction, ...]] = []
                    for point in _chart_points(reference_tokens):
                        mapped_point = tuple(
                            origin[axis]
                            + sum(
                                (
                                    point[local]
                                    * (child_corners[local + 1][axis] - origin[axis])
                                    for local in range(len(point))
                                ),
                                _ZERO,
                            )
                            for axis in range(len(origin))
                        )
                        weights = _barycentric(mapped_point, corners, solver)
                        if weights is None or any(value < 0 for value in weights):
                            break
                        rows.append(weights)
                    else:
                        matrix_rows = tuple(rows)
                        determinant = _determinant3(matrix_rows)
                        if determinant > 0:
                            matrix = (matrix_rows, determinant)
                matrices[key] = matrix
            found = matrices[key]
            if found is not None:
                if selected is not None:
                    return None
                selected = (candidate, *found)
        if selected is None:
            return None
        parent_row, matrix_rows, determinant = selected
        # (2) Source parameters follow the same barycentric map.
        parameters = parent_parameters.get(parent_row)
        if parameters is None:
            parameters = parent_parameters[parent_row] = _chart_points(
                parent_parameter_tokens[parent_row]
            )
        child_parameters = _chart_points(witness.source_parameter_rational_tokens[row])
        for weights, point in zip(matrix_rows, child_parameters, strict=True):
            for axis in range(len(point)):
                if point[axis] != sum(
                    (weights[corner] * parameters[corner][axis] for corner in range(3)),
                    _ZERO,
                ):
                    return None
        # (3) Area and oriented edges in the parent row's barycentric plane.
        area[parent_row] = area.get(parent_row, _ZERO) + determinant
        row_edges = edges.setdefault(parent_row, {})
        for first, second in ((0, 1), (1, 2), (2, 0)):
            start, stop = matrix_rows[first], matrix_rows[second]
            if any(start[slot] == 0 and stop[slot] == 0 for slot in range(3)):
                continue
            if start < stop:
                edge, sign = (start, stop), 1
            else:
                edge, sign = (stop, start), -1
            row_edges[edge] = row_edges.get(edge, 0) + sign
        parent_rows[row] = parent_row
        identity_rows[row] = matrix_rows == identity_matrix
    parent_count = len(parent_witness.source_chart_ids)
    if len(area) != parent_count or any(value != _ONE for value in area.values()):
        return None
    if any(any(row_edges.values()) for row_edges in edges.values()):
        return None
    payload = {
        "kind": "implicit-chart-restriction-evidence",
        "route": CHART_RESTRICTION_ROUTE,
        "mesh": (mesh.mesh_id, mesh.topology_id, cell_geometry_id(geometry)),
        "witness": witness.witness_id,
        "parent_certificate": parent.certificate_id,
        "transition": transition.transition_id,
        "parent_rows": array_tree_fingerprint(parent_rows),
        "identity_rows": array_tree_fingerprint(identity_rows),
    }
    parent_rows.setflags(write=False)
    identity_rows.setflags(write=False)
    return ImplicitChartRestriction(
        ImplicitChartRestrictionEvidence(
            CHART_RESTRICTION_ROUTE,
            mesh.mesh_id,
            mesh.topology_id,
            cell_geometry_id(geometry),
            witness.witness_id,
            parent.certificate_id,
            transition.transition_id,
            parent_count,
            count,
            int(np.count_nonzero(identity_rows)),
            canonical_fingerprint(payload),
        ),
        parent_rows,
        identity_rows,
    )


__all__ = [
    "CHART_RESTRICTION_ROUTE",
    "CHART_ROW_COLUMNS",
    "ImplicitChartPremise",
    "ImplicitChartRestriction",
    "ImplicitChartRestrictionEvidence",
    "ImplicitChartRows",
    "certify_implicit_chart_restriction",
]
