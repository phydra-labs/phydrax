"""Commuting polynomial-form transfer on exact surface material pieces."""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import Array

from ..._fingerprint import canonical_fingerprint
from ...exterior._form_type import FormTwist
from ...linalg import AbstractLinearOperator, ComplexMap
from ...sparse import SparseCoordinateOperator
from ...typing import parse
from .._cell_mesh import CellMesh
from .._surface_chart_records import PreparedSurfaceChartDeformation
from ._de_rham import (
    _periodic_matrix,
    _rechart_reference_piece,
    _reference_chart_form_action,
    _ReferenceChartPiece,
    FiniteElementDeRhamComplex,
)


class PreparedSurfaceChartCompatibleTransfer(NamedTuple):
    """A material pullback chain map and its observed commuting defects.

    ``complex_map.map(k)`` is the primal coefficient action for degree ``k``.
    Its sparse operator's transpose is the algebraic dual action; the
    ``complex_map.adjoint()`` is the Hilbert adjoint in the endpoint Hodge
    pairings. These actions are intentionally distinct.
    """

    complex_map: ComplexMap
    commuting_defects: Array
    deformation_id: str
    form_twist: FormTwist


def _validate_chart_topology(
    chart_mesh: CellMesh,
    physical_mesh: CellMesh,
    role: str,
    /,
) -> None:
    if (
        chart_mesh.topological_dimension != physical_mesh.topological_dimension
        or not np.array_equal(
            np.asarray(chart_mesh.vertex_global_ids),
            np.asarray(physical_mesh.vertex_global_ids),
        )
        or len(chart_mesh.blocks) != len(physical_mesh.blocks)
    ):
        raise ValueError(f"Surface chart {role} complex has different mesh topology.")
    chart_vertex_ids = np.asarray(chart_mesh.vertex_global_ids, dtype=np.int64)
    physical_vertex_ids = np.asarray(physical_mesh.vertex_global_ids, dtype=np.int64)
    for chart_block, physical_block in zip(
        chart_mesh.blocks, physical_mesh.blocks, strict=True
    ):
        chart_cell_ids = np.asarray(chart_block.global_ids, dtype=np.int64)
        physical_cell_ids = np.asarray(physical_block.global_ids, dtype=np.int64)
        chart_vertices = np.sort(
            chart_vertex_ids[np.asarray(chart_block.vertices, dtype=np.int64)], axis=1
        )
        physical_vertices = np.sort(
            physical_vertex_ids[np.asarray(physical_block.vertices, dtype=np.int64)],
            axis=1,
        )
        if (
            chart_block.name != physical_block.name
            or chart_block.cell_kind != physical_block.cell_kind
            or not np.array_equal(chart_cell_ids, physical_cell_ids)
            or not np.array_equal(chart_vertices, physical_vertices)
        ):
            raise ValueError(f"Surface chart {role} complex changes cell identity.")


def _validate_endpoints(
    source: FiniteElementDeRhamComplex,
    target: FiniteElementDeRhamComplex,
    prepared: PreparedSurfaceChartDeformation,
    /,
) -> None:
    _validate_chart_topology(source.mesh, prepared.source_mesh, "source")
    _validate_chart_topology(target.mesh, prepared.target_mesh, "target")
    prepared.require_bound(
        prepared.source_mesh,
        prepared.source_geometry,
        prepared.target_mesh,
        prepared.target_geometry,
    )


def _commuting_defects(
    source: FiniteElementDeRhamComplex,
    target: FiniteElementDeRhamComplex,
    maps: Sequence[AbstractLinearOperator],
    /,
) -> np.ndarray:
    defects = np.empty((source.dimension,), dtype=np.float64)
    for degree in range(source.dimension):
        source_map = source._absolute.differentials[degree]
        target_map = target._absolute.differentials[degree]
        current_map, following_map = maps[degree : degree + 2]
        if not all(
            isinstance(value, SparseCoordinateOperator)
            for value in (source_map, target_map, current_map, following_map)
        ):
            raise TypeError("Chart form actions require sparse de Rham coordinates.")
        source_differential = _periodic_matrix(source_map)
        target_differential = _periodic_matrix(target_map)
        current = _periodic_matrix(current_map)
        following = _periodic_matrix(following_map)
        left = target_differential @ current
        right = following @ source_differential
        residual = (left - right).tocsr()
        residual.eliminate_zeros()
        defects[degree] = float(np.max(np.abs(residual.data), initial=0.0))
    return defects


def prepare_surface_chart_compatible_transfer(
    source: FiniteElementDeRhamComplex,
    target: FiniteElementDeRhamComplex,
    prepared: PreparedSurfaceChartDeformation,
    /,
    *,
    commuting_tolerance: float = 2.0e-8,
    form_twist: FormTwist | None = None,
) -> PreparedSurfaceChartCompatibleTransfer:
    """Prepare all H1, H(curl), and surface H(div) form actions.

    Exact piece correspondences are integrated against the target's complete
    canonical form functionals. Affine material maps use exact reference
    geometry; orientation and twisted-form actions use the canonical form
    basis and exterior pullback. The result is admitted only if its observed
    chain residual is within ``commuting_tolerance``.
    """
    if not isinstance(source, FiniteElementDeRhamComplex) or not isinstance(
        target, FiniteElementDeRhamComplex
    ):
        raise TypeError("Surface chart form transfer requires de Rham complexes.")
    if not isinstance(prepared, PreparedSurfaceChartDeformation):
        raise TypeError("prepared must be PreparedSurfaceChartDeformation.")
    tolerance = float(commuting_tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("commuting_tolerance must be finite and nonnegative.")
    selected_twist = (
        source.primal_twist
        if form_twist is None
        else parse(form_twist, FormTwist, "form_twist")
    )
    _validate_endpoints(source, target, prepared)
    pieces = tuple(
        _rechart_reference_piece(
            _ReferenceChartPiece(
                piece.source_row,
                piece.target_row,
                piece.exact_target_reference_vertices,
                piece.exact_source_reference_vertices,
                None,
            ),
            source.mesh,
            target.mesh,
            prepared.source_mesh,
            prepared.target_mesh,
        )
        for occurrence in prepared.occurrences
        for piece in occurrence.pieces
    )
    maps = tuple(
        _reference_chart_form_action(
            source,
            target,
            degree,
            pieces,
            f"{prepared.deformation_id}:surface-form:{degree}:{selected_twist}",
            form_twist=selected_twist,
        )
        for degree in range(source.dimension + 1)
    )
    map_id = canonical_fingerprint(
        {
            "kind": "surface-chart-form-transfer",
            "source": source.realization_id,
            "target": target.realization_id,
            "deformation": prepared.deformation_id,
            "form_twist": selected_twist,
            "operators": tuple(value.operator_id for value in maps),
        }
    )
    complex_map = ComplexMap(source._absolute, target._absolute, maps, map_id=map_id)
    defects = _commuting_defects(source, target, maps)
    scale = max(1.0, float(np.max(defects, initial=0.0)))
    if np.any(defects > tolerance * scale):
        raise ValueError(
            "Surface chart form transfer does not commute within its numerical tolerance."
        )
    return PreparedSurfaceChartCompatibleTransfer(
        complex_map,
        jnp.asarray(defects, dtype=jnp.float64),
        prepared.deformation_id,
        selected_twist,
    )


__all__ = [
    "PreparedSurfaceChartCompatibleTransfer",
    "prepare_surface_chart_compatible_transfer",
]
