#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Certified analytic edge blends with explicit unsupported decisions."""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from typing import final, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array

from ..._strict import StrictModule
from ..._validation import positive_finite_float
from ...typing import checked, parse
from ._assembly import BRepAffinePlacement
from .._rational import rational_vector
from ...discretization._exact_algebraic_geometry import ExactScalar
from ._constructors import BRepTessellationPolicy
from ._differentiable import FixedTopologyBRepSource
from ._edge_classification import brep_edge_classification
from ._model import BRepEntityId, BRepModel
from ._parameter_range import (
    attach_parameter_range,
    CertifiedParameterRange,
    unknown_parameter_range,
)
from ._patches import (
    ConePatch,
    CylinderPatch,
    ExactLineCurve,
    LineCurve,
    PlanePatch,
    RevolutionSurface,
    RevolutionQuadricWitness,
    SpherePatch,
)


BRepBlendStatus: TypeAlias = Literal["certified", "unsupported", "unresolved"]
BRepBlendSensitivityStatus: TypeAlias = Literal[
    "available", "unsupported", "not_applicable"
]
_BlendFamily: TypeAlias = Literal[
    "planes", "closed_seam", "parallel_cylinders", "unsupported"
]
type _AnalyticFaceFamily = Literal["plane", "cylinder", "cone", "sphere"]
type _ExactVector = tuple[ExactScalar, ...]


class BRepBlendUnsupported(ValueError):
    """An analytic blend family or its unchanged-topology certificate is unavailable."""


class BRepBlendUnresolved(ValueError):
    """A supported blend proof remains undecided at mathematical resolution."""


@final
class BRepBlendResult(StrictModule):
    """A rebuilt solid or an explicit refusal keyed by source edge identities.

    Producers omit ``parameter_range`` on success unless requested. Refusals
    retain a certified interval or explicit unknown evidence. Radius intervals
    have coordinate length units; chamfer ``setback_scale`` is dimensionless
    and multiplies both authored distances without changing their ratio.
    """

    status: BRepBlendStatus = eqx.field(static=True)
    reason: str = eqx.field(static=True)
    source_model_id: str = eqx.field(static=True)
    source_edges: tuple[BRepEntityId, ...] = eqx.field(static=True)
    model: BRepModel | None
    g1_angle_bounds: Array
    source: FixedTopologyBRepSource | None
    sensitivity_status: BRepBlendSensitivityStatus = eqx.field(static=True)
    sensitivity_reason: str = eqx.field(static=True)
    parameter_range: CertifiedParameterRange | None = eqx.field(static=True)

    def __init__(
        self,
        *,
        status: BRepBlendStatus,
        reason: str,
        source_model_id: str,
        source_edges: tuple[BRepEntityId, ...],
        model: BRepModel | None,
        g1_angle_bounds: Array,
        source: FixedTopologyBRepSource | None = None,
        sensitivity_status: BRepBlendSensitivityStatus = "not_applicable",
        sensitivity_reason: str = "",
        parameter_range: CertifiedParameterRange | None = None,
    ) -> None:
        self.status = parse(status, BRepBlendStatus, "blend.status")
        if (self.status == "certified") != (model is not None):
            raise ValueError(
                "BRepBlendResult.model must be present only for a certified result; got inconsistent status and model"
            )
        self.reason = reason
        self.source_model_id = source_model_id
        self.source_edges = source_edges
        self.model = model
        self.g1_angle_bounds = jnp.asarray(g1_angle_bounds, dtype=jnp.float64).reshape(
            (-1, 2)
        )
        self.source = source
        self.sensitivity_status = parse(
            sensitivity_status, BRepBlendSensitivityStatus, "blend.sensitivity_status"
        )
        if (self.sensitivity_status == "available") != (source is not None):
            raise ValueError(
                "BRepBlendResult.source must accompany available sensitivity; got inconsistent source and status"
            )
        self.sensitivity_reason = sensitivity_reason
        self.parameter_range = parameter_range


def _refusal(
    model: BRepModel,
    edges: tuple[BRepEntityId, ...],
    status: BRepBlendStatus,
    reason: str,
    /,
    *,
    parameter: str,
    parameter_range: CertifiedParameterRange | None = None,
) -> BRepBlendResult:
    return BRepBlendResult(
        status=status,
        reason=reason,
        source_model_id=model.model_id,
        source_edges=edges,
        model=None,
        g1_angle_bounds=jnp.empty((0, 2), dtype=jnp.float64),
        parameter_range=parameter_range or unknown_parameter_range(parameter, reason),
    )


@checked
def brep_fillet(
    model: BRepModel,
    edges: Sequence[BRepEntityId],
    radius: float,
    /,
    *,
    tessellation: BRepTessellationPolicy | None = None,
    source_id: str = "native-fillet",
    include_parameter_range: bool = False,
) -> BRepBlendResult:
    """Blend selected analytic edges with a constant positive rolling radius.

    The side is selected by the whole-edge BX-11 certificate. A corner requiring
    multiple blended incident edges refuses rather than inventing a vertex patch.
    Requested ranges certify an open radius interval; all refusals retain evidence.
    """
    try:
        radius_ = positive_finite_float(radius, "radius")
        return _blend(
            model,
            tuple(edges),
            radius_,
            None,
            tessellation=tessellation,
            source_id=source_id,
            include_parameter_range=include_parameter_range,
        )
    except ValueError as error:
        if "parameter_range" not in error.__dict__:
            attach_parameter_range(error, unknown_parameter_range("radius", str(error)))
        raise


@checked
def brep_chamfer(
    model: BRepModel,
    edges: Sequence[BRepEntityId],
    setbacks: tuple[float, float],
    /,
    *,
    tessellation: BRepTessellationPolicy | None = None,
    source_id: str = "native-chamfer",
    include_parameter_range: bool = False,
) -> BRepBlendResult:
    """Replace an edge by an analytic face with two explicit face setback distances.

    A requested ``setback_scale`` range multiplies both authored distances by one
    positive scale, keeping their ratio fixed; this request is at scale one.
    All refusals retain range evidence, even when no success range is requested.
    """
    try:
        if len(setbacks) != 2:
            raise ValueError(
                "brep_chamfer.setbacks must contain two face distances; got a different count"
            )
        setbacks_ = tuple(positive_finite_float(value, "setback") for value in setbacks)
        return _blend(
            model,
            tuple(edges),
            None,
            (setbacks_[0], setbacks_[1]),
            tessellation=tessellation,
            source_id=source_id,
            include_parameter_range=include_parameter_range,
        )
    except ValueError as error:
        if "parameter_range" not in error.__dict__:
            attach_parameter_range(
                error, unknown_parameter_range("setback_scale", str(error))
            )
        raise


def _blend(
    model: BRepModel,
    edges: tuple[BRepEntityId, ...],
    radius: float | None,
    setbacks: tuple[float, float] | None,
    /,
    *,
    tessellation: BRepTessellationPolicy | None,
    source_id: str,
    include_parameter_range: bool,
) -> BRepBlendResult:
    parameter = "radius" if radius is not None else "setback_scale"
    if not edges or len(set(edges)) != len(edges):
        raise ValueError(
            "brep_fillet.edges must contain distinct nonempty source identities; got an empty or repeated selection"
        )
    source_edges = model.edge_ids
    if any(edge not in source_edges for edge in edges):
        raise ValueError(
            "brep_fillet.edges must belong to the source model revision; got a foreign identity"
        )
    geometry = model.geometry
    indices = tuple(source_edges.index(edge) for edge in edges)
    if len(indices) > 1:
        vertices = tuple(set(geometry.edge_vertices[index]) for index in indices)
        if any(
            first.intersection(last)
            for position, first in enumerate(vertices)
            for last in vertices[position + 1 :]
        ):
            return _refusal(
                model,
                edges,
                "unsupported",
                "vertex blend with multiple blended incident edges requires a corner patch",
                parameter=parameter,
            )
        return _refusal(
            model,
            edges,
            "unsupported",
            "multiple independently selected edges require an exhaustive blend lineage",
            parameter=parameter,
        )
    edge_index = indices[0]
    incident = tuple(
        face
        for face, loops in enumerate(geometry.face_loops)
        if any(
            geometry.coedge_edges[coedge] == edge_index
            for loop in loops
            for coedge in loop
        )
    )
    if len(incident) != 2 or incident[0] == incident[1]:
        return _refusal(
            model,
            edges,
            "unsupported",
            "blend requires exactly two distinct incident faces",
            parameter=parameter,
        )
    family = _blend_family(model, (incident[0], incident[1]))
    if family == "unsupported":
        return _refusal(
            model,
            edges,
            "unsupported",
            "brep_fillet.adjacency must belong to a supported analytic pair; got an unsupported analytic face adjacency",
            parameter=parameter,
        )
    classification = brep_edge_classification(model, geometry)[edge_index]
    if classification.status in ("unresolved", "sign_change"):
        return _refusal(
            model,
            edges,
            "unresolved",
            f"whole-edge convexity certificate is {classification.status}",
            parameter=parameter,
        )
    try:
        rebuilt = _dispatch(
            model,
            edge_index,
            family,
            radius,
            setbacks,
            tessellation=tessellation,
            source_id=source_id,
        )
    except BRepBlendUnresolved as unresolved:
        parameter_range = _blend_parameter_range(
            model, edge_index, family, radius, setbacks
        )
        attach_parameter_range(unresolved, parameter_range)
        return _refusal(
            model,
            edges,
            "unresolved",
            str(unresolved),
            parameter=parameter,
            parameter_range=parameter_range,
        )
    except BRepBlendUnsupported as refusal:
        parameter_range = _blend_parameter_range(
            model, edge_index, family, radius, setbacks
        )
        attach_parameter_range(refusal, parameter_range)
        return _refusal(
            model,
            edges,
            "unsupported",
            str(refusal),
            parameter=parameter,
            parameter_range=parameter_range,
        )
    except ValueError as error:
        attach_parameter_range(
            error,
            _blend_parameter_range(model, edge_index, family, radius, setbacks),
        )
        raise
    parameter_range = (
        _blend_parameter_range(model, edge_index, family, radius, setbacks)
        if include_parameter_range
        else None
    )
    bounds = jnp.empty((0, 2), dtype=jnp.float64)
    source = None
    sensitivity_status: BRepBlendSensitivityStatus = "not_applicable"
    sensitivity_reason = ""
    if radius is not None:
        try:
            bounds = _g1_bounds(model, rebuilt, edge_index, family, radius)
        except BRepBlendUnresolved as failure:
            range_evidence = parameter_range or _blend_parameter_range(
                model, edge_index, family, radius, setbacks
            )
            attach_parameter_range(failure, range_evidence)
            return _refusal(
                model,
                edges,
                "unresolved",
                str(failure),
                parameter=parameter,
                parameter_range=range_evidence,
            )
        except BRepBlendUnsupported as refusal:
            range_evidence = parameter_range or _blend_parameter_range(
                model, edge_index, family, radius, setbacks
            )
            return _refusal(
                model,
                edges,
                "unsupported",
                str(refusal),
                parameter=parameter,
                parameter_range=range_evidence,
            )
        try:
            source = _radius_source(model, rebuilt, edge_index, family, radius)
            sensitivity_status = "available"
        except BRepBlendUnsupported as refusal:
            sensitivity_status = "unsupported"
            sensitivity_reason = str(refusal)
    return BRepBlendResult(
        status="certified",
        reason="certified native analytic blend with exact source correspondence",
        source_model_id=model.model_id,
        source_edges=edges,
        model=rebuilt,
        g1_angle_bounds=bounds,
        source=source,
        sensitivity_status=sensitivity_status,
        sensitivity_reason=sensitivity_reason,
        parameter_range=parameter_range if include_parameter_range else None,
    )


def _blend_parameter_range(
    model: BRepModel,
    edge_index: int,
    family: _BlendFamily,
    radius: float | None,
    setbacks: tuple[float, float] | None,
    /,
) -> CertifiedParameterRange:
    parameter = "radius" if setbacks is None else "setback_scale"
    try:
        if family == "planes":
            if _perpendicular_plane_ends(model, edge_index):
                from ._planar_blend import planar_blend_parameter_range

                return planar_blend_parameter_range(model, edge_index, setbacks)
            if setbacks is not None:
                from ._oblique_chamfer import oblique_chamfer_parameter_range

                return oblique_chamfer_parameter_range(model, edge_index, setbacks)
            from ._oblique_fillet import oblique_fillet_parameter_range

            return oblique_fillet_parameter_range(model, edge_index)
        if family == "closed_seam":
            from ._closed_seam_blend import closed_seam_parameter_range

            if not _closed_seam_range_source(model):
                return unknown_parameter_range(
                    parameter,
                    "closed seam range: remote source faces or shells lack a whole-interval collision exclusion",
                )
            return closed_seam_parameter_range(model, edge_index, setbacks)
        if family == "parallel_cylinders" and radius is not None:
            from ._parallel_cylinder_blend import parallel_cylinder_parameter_range

            return parallel_cylinder_parameter_range(model, edge_index, radius)
    except ValueError as error:
        return unknown_parameter_range(parameter, str(error))
    return unknown_parameter_range(
        parameter,
        f"No certified feasible interval for the {family} blend family and its source trims.",
    )


def _closed_seam_range_source(model: BRepModel, /) -> bool:
    """Admit only one exact quadric with its coaxial circular plane caps."""
    geometry = model.geometry
    families = tuple(
        _face_analytic_family(model, face) for face in range(len(model.patches))
    )
    curved_faces = tuple(
        face
        for face, family in enumerate(families)
        if family in ("cylinder", "cone", "sphere")
    )
    plane_faces = tuple(
        face for face, family in enumerate(families) if family == "plane"
    )
    if (
        len(curved_faces) != 1
        or not plane_faces
        or len(model.patches) not in (2, 3)
        or any(family is None for family in families)
        or len(geometry.shell_faces) != 1
        or set(geometry.shell_faces[0]) != set(range(len(model.patches)))
        or geometry.solid_shells != ((0,),)
        or geometry.assembly_graph.solid_count != 1
        or geometry.assembly_graph.definitions
        or len(geometry.assembly_graph.roots) != 1
        or geometry.assembly_graph.roots[0].solid_index != 0
        or geometry.assembly_graph.roots[0].placement != BRepAffinePlacement.identity()
    ):
        return False
    curved_frame = _exact_surface_frame(model, curved_faces[0])
    if curved_frame is None:
        return False
    _, curved_origin, axis, radial_axis, second_radial_axis = curved_frame
    axis_squared = _vector_dot_exact(axis, axis)
    if axis_squared == 0:
        return False
    for face in plane_faces:
        if (
            len(geometry.face_loops[face]) != 1
            or len(geometry.face_loops[face][0]) != 1
        ):
            return False
        plane_frame = _exact_surface_frame(model, face)
        if plane_frame is None:
            return False
        _, plane_origin, _, plane_radial, plane_second_radial = plane_frame
        if plane_radial != radial_axis or plane_second_radial != second_radial_axis:
            return False
        shift = tuple(
            plane_origin[index] - curved_origin[index] for index in range(3)
        )
        height = _vector_dot_exact(shift, axis) / axis_squared
        if shift != tuple(height * value for value in axis):
            return False
    return True


def _vector_dot_exact(first: _ExactVector, second: _ExactVector, /) -> ExactScalar:
    return sum(
        (left * right for left, right in zip(first, second, strict=True)),
        Fraction(0),
    )


def _vector_cross_exact(
    first: _ExactVector, second: _ExactVector, /
) -> _ExactVector:
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


def _revolution_quadric_witness(
    model: BRepModel, face: int, /
) -> RevolutionQuadricWitness | None:
    patch = model.patches[face]
    if not isinstance(patch, RevolutionSurface):
        return None
    bounds = np.asarray(model.parameter_bounds[face], dtype=np.float64)
    return patch.exact_quadric_witness(
        (float(bounds[0, 1]), float(bounds[1, 1]))
    )


def _face_analytic_family(
    model: BRepModel, face: int, /
) -> _AnalyticFaceFamily | None:
    patch = model.patches[face]
    if isinstance(patch, PlanePatch):
        return "plane"
    if isinstance(patch, CylinderPatch):
        return "cylinder"
    if isinstance(patch, ConePatch):
        return "cone"
    if isinstance(patch, SpherePatch):
        return "sphere"
    witness = _revolution_quadric_witness(model, face)
    return None if witness is None else witness.family


def _exact_surface_frame(
    model: BRepModel, face: int, /
) -> tuple[
    _AnalyticFaceFamily, _ExactVector, _ExactVector, _ExactVector, _ExactVector
] | None:
    family = _face_analytic_family(model, face)
    if family is None:
        return None
    patch = model.patches[face]
    if isinstance(patch, RevolutionSurface):
        witness = _revolution_quadric_witness(model, face)
        if witness is None:
            return None
        if family == "sphere":
            if witness.center_exact is None:
                return None
            origin = witness.center_exact
        else:
            if witness.origin_exact is None or witness.axial_origin_exact is None:
                return None
            origin = tuple(
                witness.origin_exact[index]
                + witness.axial_origin_exact * witness.axis_exact[index]
                for index in range(3)
            )
        return (
            family,
            tuple(origin),
            witness.axis_exact,
            witness.radial_axis_exact,
            witness.second_radial_axis_exact,
        )
    if isinstance(patch, CylinderPatch):
        return (
            family,
            patch.origin_exact,
            patch.axis_exact,
            patch.first_axis_exact,
            patch.second_axis_exact,
        )
    if isinstance(patch, PlanePatch):
        origin = tuple(rational_vector(np.asarray(patch.origin)))
        first = tuple(rational_vector(np.asarray(patch.first_axis)))
        second = tuple(rational_vector(np.asarray(patch.second_axis)))
        return family, origin, _vector_cross_exact(first, second), first, second
    if isinstance(patch, (ConePatch, SpherePatch)):
        origin_source = patch.center if isinstance(patch, SpherePatch) else patch.origin
        return (
            family,
            tuple(rational_vector(np.asarray(origin_source))),
            tuple(rational_vector(np.asarray(patch.axis))),
            tuple(rational_vector(np.asarray(patch.first_axis))),
            tuple(rational_vector(np.asarray(patch.second_axis))),
        )
    return None


def _blend_family(model: BRepModel, faces: tuple[int, int], /) -> _BlendFamily:
    patches = tuple(model.patches[face] for face in faces)
    if all(isinstance(patch, PlanePatch) for patch in patches):
        return "planes"
    if all(isinstance(patch, CylinderPatch) for patch in patches):
        return "parallel_cylinders"
    families = tuple(_face_analytic_family(model, face) for face in faces)
    curved_families = ("cylinder", "cone", "sphere")
    if (
        families[0] == "plane" and families[1] in curved_families
    ) or (families[1] == "plane" and families[0] in curved_families):
        return "closed_seam"
    return "unsupported"


def _dispatch(
    model: BRepModel,
    edge_index: int,
    family: _BlendFamily,
    radius: float | None,
    setbacks: tuple[float, float] | None,
    /,
    *,
    tessellation: BRepTessellationPolicy | None,
    source_id: str,
) -> BRepModel:
    match family:
        case "planes":
            if _perpendicular_plane_ends(model, edge_index):
                from ._planar_blend import planar_blend

                return planar_blend(
                    model,
                    edge_index,
                    radius,
                    setbacks,
                    tessellation=tessellation,
                    source_id=source_id,
                )
            if radius is not None:
                from ._oblique_fillet import oblique_fillet

                return oblique_fillet(
                    model,
                    edge_index,
                    radius,
                    tessellation=tessellation,
                    source_id=source_id,
                )
            if setbacks is None:
                raise ValueError(
                    "brep_chamfer.setbacks must declare two face distances; got no distances"
                )
            from ._oblique_chamfer import oblique_chamfer

            return oblique_chamfer(
                model,
                edge_index,
                setbacks,
                tessellation=tessellation,
                source_id=source_id,
            )
        case "closed_seam":
            from ._closed_seam_blend import closed_seam_blend

            return closed_seam_blend(
                model,
                edge_index,
                radius,
                setbacks,
                tessellation=tessellation,
                source_id=source_id,
            )
        case "parallel_cylinders":
            from ._parallel_cylinder_blend import blend_parallel_cylinders

            return blend_parallel_cylinders(
                model,
                edge_index,
                radius,
                setbacks,
                tessellation=tessellation,
                source_id=source_id,
            )
        case "unsupported":
            raise BRepBlendUnsupported(
                "brep_fillet.adjacency must belong to a supported analytic pair; got an unsupported analytic face adjacency"
            )
        case _:
            raise ValueError(
                "brep_fillet.family must select a supported analytic pair; got an invalid family"
            )


def _g1_bounds(
    source: BRepModel,
    result: BRepModel,
    edge_index: int,
    family: _BlendFamily,
    radius: float,
    /,
) -> Array:
    match family:
        case "planes":
            return _planar_g1_bounds(result)
        case "parallel_cylinders":
            from ._parallel_cylinder_blend import parallel_cylinder_g1_bounds

            return jnp.asarray(
                parallel_cylinder_g1_bounds(source, result, edge_index),
                dtype=jnp.float64,
            )
        case "closed_seam":
            from ._closed_seam_blend import closed_seam_tangency_certificate

            certificate = closed_seam_tangency_certificate(source, edge_index, radius)
            differences = np.asarray(certificate.normal_difference_bounds)
            magnitudes = np.max(np.abs(differences), axis=-1)
            lengths = np.nextafter(
                np.sqrt(np.sum(np.nextafter(magnitudes**2, np.inf), axis=-1)), np.inf
            )
            upper = np.nextafter(2 * np.arcsin(np.minimum(1.0, lengths / 2)), np.inf)
            return jnp.asarray(
                np.stack((np.zeros_like(upper), upper), axis=-1), dtype=jnp.float64
            )
        case "unsupported":
            raise BRepBlendUnsupported(
                "brep_fillet.family must support tangency certification; got an unsupported family"
            )
        case _:
            raise ValueError(
                "brep_fillet.family must select a supported analytic pair; got an invalid family"
            )


def _planar_g1_bounds(result: BRepModel, /) -> Array:
    geometry = result.geometry
    records = brep_edge_classification(result, geometry)
    blend_faces = {
        index
        for index, patch in enumerate(result.patches)
        if isinstance(patch, CylinderPatch)
    }
    boundary_edges = {
        geometry.coedge_edges[coedge]
        for face in blend_faces
        for loop in geometry.face_loops[face]
        for coedge in loop
        if isinstance(
            geometry.curves[geometry.edge_curves[geometry.coedge_edges[coedge]]],
            (ExactLineCurve, LineCurve),
        )
    }
    if len(boundary_edges) != 2:
        raise RuntimeError(
            "brep_fillet.boundaries must contain two straight G1 edges; got different incidence"
        )
    return jnp.stack(
        tuple(records[edge].dihedral_range for edge in sorted(boundary_edges))
    )

def _radius_source(
    source: BRepModel,
    result: BRepModel,
    edge_index: int,
    family: _BlendFamily,
    radius: float,
    /,
) -> FixedTopologyBRepSource:
    if result.tessellation is not None:
        raise BRepBlendUnsupported(
            "fillet radius sensitivity cannot retain a tessellation fidelity certificate under parameter motion"
        )
    match family:
        case "planes":
            if not _perpendicular_plane_ends(source, edge_index):
                raise BRepBlendUnsupported(
                    "brep_fillet.radius_sensitivity must support linked intersection-cap trims; got unsupported oblique-cap sensitivity"
                )
            from ._fillet_radius_map import PlanarFilletRadiusMap

            parameter_map = PlanarFilletRadiusMap(source, edge_index, radius, result)
        case "parallel_cylinders":
            from ..design._schema import ParameterId
            from ._parallel_cylinder_blend import ParallelCylinderRadiusMap

            parameter_map = ParallelCylinderRadiusMap(
                source,
                edge_index,
                radius,
                result,
                parameter_id=ParameterId(
                    f"{source.source_revision}:edge:{edge_index}:fillet", "radius"
                ),
            )
        case "closed_seam":
            from ._closed_seam_blend import prepare_closed_seam_radius_map

            parameter_map = prepare_closed_seam_radius_map(
                source, edge_index, radius, result
            )
        case "unsupported":
            raise BRepBlendUnsupported(
                "brep_fillet.family must support linked radius realization; got an unsupported family"
            )
        case _:
            raise ValueError(
                "brep_fillet.family must select a supported analytic pair; got an invalid family"
            )
    return FixedTopologyBRepSource(result, parameter_map=parameter_map)


def _perpendicular_plane_ends(model: BRepModel, edge_index: int, /) -> bool:
    geometry = model.geometry
    first, last = geometry.edge_vertices[edge_index]
    first_point = geometry.vertex_exact_points[first]
    if first_point is None:
        first_point = rational_vector(np.asarray(geometry.vertex_points[first]))
    last_point = geometry.vertex_exact_points[last]
    if last_point is None:
        last_point = rational_vector(np.asarray(geometry.vertex_points[last]))
    direction = tuple(
        b - a for a, b in zip(first_point, last_point, strict=True)
    )
    for face, loops in enumerate(geometry.face_loops):
        edges = {geometry.coedge_edges[coedge] for loop in loops for coedge in loop}
        if edge_index in edges:
            continue
        vertices = {vertex for edge in edges for vertex in geometry.edge_vertices[edge]}
        if first not in vertices and last not in vertices:
            continue
        patch = model.patches[face]
        if not isinstance(patch, PlanePatch):
            return False
        for axis in (patch.first_axis, patch.second_axis):
            if (
                sum(
                    (
                        a * b
                        for a, b in zip(
                            direction, rational_vector(np.asarray(axis)), strict=True
                        )
                    ),
                    Fraction(),
                )
                != 0
            ):
                return False
    return True