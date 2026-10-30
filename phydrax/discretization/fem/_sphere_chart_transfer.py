"""Projective sphere-chart H1, DG, and compatible form transfer."""

from __future__ import annotations

from fractions import Fraction
from typing import assert_never, NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import Array

from ..._fingerprint import canonical_fingerprint
from ...exterior._form_type import FormTwist
from ...linalg import AbstractLinearOperator, ComplexMap
from ...linalg._exact import solve_exact
from ...sparse import SparseCoordinateOperator
from ...typing import parse
from .._sphere_chart_records import PreparedSphereChartDeformation
from ..finite_volume._unstructured import UnstructuredFiniteVolumeDiscretization
from ._de_rham import (
    _periodic_matrix,
    _rechart_reference_piece,
    _reference_chart_form_action,
    _ReferenceChartPiece,
    FiniteElementDeRhamComplex,
)
from ._surface_chart_compatible import _validate_chart_topology
from ._surface_chart_transfer import (
    _prepare_finite_volume_transfer,
    PreparedSurfaceChartFieldTransfer,
    PreparedSurfaceChartFiniteVolumeContents,
    SurfaceChartFieldSemantics,
)


type _ProjectiveMatrix = tuple[tuple[Fraction, ...], ...]

_PROJECTIVE_IDENTITY: _ProjectiveMatrix = (
    (Fraction(1), Fraction(0), Fraction(0)),
    (Fraction(0), Fraction(1), Fraction(0)),
    (Fraction(0), Fraction(0), Fraction(1)),
)


def _inverse_projective_coefficients(
    coefficients: _ProjectiveMatrix, /
) -> _ProjectiveMatrix:
    solved = solve_exact(coefficients, _PROJECTIVE_IDENTITY)
    return (
        (solved[0][0], solved[0][1], solved[0][2]),
        (solved[1][0], solved[1][1], solved[1][2]),
        (solved[2][0], solved[2][1], solved[2][2]),
    )


class PreparedSphereChartCompatibleTransfer(NamedTuple):
    """A projective reference-map chain action and observed commuting defects."""

    complex_map: ComplexMap
    commuting_defects: Array
    deformation_id: str
    form_twist: FormTwist


def _validate_endpoints(
    source: FiniteElementDeRhamComplex,
    target: FiniteElementDeRhamComplex,
    prepared: PreparedSphereChartDeformation,
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
    maps: tuple[AbstractLinearOperator, ...],
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
            raise TypeError("Sphere chart actions require sparse de Rham coordinates.")
        left = _periodic_matrix(target_map) @ _periodic_matrix(current_map)
        right = _periodic_matrix(following_map) @ _periodic_matrix(source_map)
        residual = (left - right).tocsr()
        residual.eliminate_zeros()
        defects[degree] = float(np.max(np.abs(residual.data), initial=0.0))
    return defects


def prepare_sphere_chart_compatible_transfer(
    source: FiniteElementDeRhamComplex,
    target: FiniteElementDeRhamComplex,
    prepared: PreparedSphereChartDeformation,
    /,
    *,
    commuting_tolerance: float = 2.0e-8,
    form_twist: FormTwist | None = None,
) -> PreparedSphereChartCompatibleTransfer:
    """Prepare projective Piola actions for every surface form degree.

    The exact homogeneous projective matrix and its nonconstant Jacobian are
    evaluated on each target overlap piece. All target canonical functionals,
    local orientation actions, and twisted-form signs are retained by the
    general-order de Rham basis.
    """
    if not isinstance(source, FiniteElementDeRhamComplex) or not isinstance(
        target, FiniteElementDeRhamComplex
    ):
        raise TypeError("Sphere chart form transfer requires de Rham complexes.")
    if not isinstance(prepared, PreparedSphereChartDeformation):
        raise TypeError("prepared must be PreparedSphereChartDeformation.")
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
                _inverse_projective_coefficients(piece.projective_map.exact_coefficients),
            ),
            source.mesh,
            target.mesh,
            prepared.source_mesh,
            prepared.target_mesh,
        )
        for piece in prepared.pieces
    )
    maps = tuple(
        _reference_chart_form_action(
            source,
            target,
            degree,
            pieces,
            f"{prepared.deformation_id}:sphere-form:{degree}:{selected_twist}",
            form_twist=selected_twist,
        )
        for degree in range(source.dimension + 1)
    )
    map_id = canonical_fingerprint(
        {
            "kind": "sphere-projective-form-transfer",
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
            "Sphere chart form transfer does not commute within its numerical tolerance."
        )
    return PreparedSphereChartCompatibleTransfer(
        complex_map,
        jnp.asarray(defects, dtype=jnp.float64),
        prepared.deformation_id,
        selected_twist,
    )


def prepare_sphere_chart_field_transfer(
    source: FiniteElementDeRhamComplex,
    target: FiniteElementDeRhamComplex,
    prepared: PreparedSphereChartDeformation,
    /,
    *,
    semantics: SurfaceChartFieldSemantics,
    commuting_tolerance: float = 2.0e-8,
) -> PreparedSurfaceChartFieldTransfer:
    selected = parse(semantics, SurfaceChartFieldSemantics, "semantics")
    match selected:
        case "h1":
            degree, transfer_twist = 0, source.primal_twist
        case "hcurl":
            degree, transfer_twist = 1, "untwisted"
        case "hdiv":
            degree, transfer_twist = 1, "twisted"
        case "dg":
            degree, transfer_twist = source.dimension, source.primal_twist
        case other:
            assert_never(other)
    transfer = prepare_sphere_chart_compatible_transfer(
        source,
        target,
        prepared,
        commuting_tolerance=commuting_tolerance,
        form_twist=transfer_twist,
    )
    return PreparedSurfaceChartFieldTransfer(
        transfer.complex_map,
        degree,
        transfer.commuting_defects,
        transfer.form_twist,
        transfer.deformation_id,
    )


def prepare_sphere_chart_finite_volume_contents(
    source: UnstructuredFiniteVolumeDiscretization,
    target: UnstructuredFiniteVolumeDiscretization,
    prepared: PreparedSphereChartDeformation,
    /,
) -> PreparedSurfaceChartFiniteVolumeContents:
    if source.mesh is None or target.mesh is None:
        raise ValueError("Sphere chart FV transfer requires resident endpoint meshes.")
    prepared.require_bound(
        source.mesh,
        prepared.source_geometry,
        target.mesh,
        prepared.target_geometry,
    )
    pieces = tuple(
        _ReferenceChartPiece(
            piece.source_row,
            piece.target_row,
            piece.exact_target_reference_vertices,
            piece.exact_source_reference_vertices,
            piece.projective_map.exact_coefficients,
        )
        for piece in prepared.pieces
    )
    return _prepare_finite_volume_transfer(source, target, prepared, pieces)


__all__ = [
    "PreparedSphereChartCompatibleTransfer",
    "prepare_sphere_chart_compatible_transfer",
    "prepare_sphere_chart_field_transfer",
    "prepare_sphere_chart_finite_volume_contents",
]
