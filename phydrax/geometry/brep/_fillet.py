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
from ...discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateEnclosureResourceError,
)
from ...typing import checked, parse
from .._rational import rational_vector
from ._constructors import BRepTessellationPolicy
from ._coordinate_request import coordinate_source_request
from ._differentiable import FixedTopologyBRepSource
from ._edge_classification import brep_edge_classification
from ._model import BRepEntityId, BRepModel
from ._parameter_range import (
    attach_parameter_range,
    CertifiedParameterRange,
    unknown_parameter_range,
)
from ._patches import (
    AbstractSurfacePatch,
    ConePatch,
    CylinderPatch,
    LineCurve,
    PlanePatch,
    SpherePatch,
)
from ._request_preparations import TrimRequestPreparations


BRepBlendStatus: TypeAlias = Literal["certified", "unsupported", "unresolved"]
BRepBlendSensitivityStatus: TypeAlias = Literal[
    "available", "unsupported", "not_applicable"
]
_BlendFamily: TypeAlias = Literal[
    "planes", "closed_seam", "parallel_cylinders", "unsupported"
]


class BRepBlendUnsupported(ValueError):
    """An analytic blend family or its unchanged-topology certificate is unavailable."""


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
    maximum_cells: int = 64,
    tessellation: BRepTessellationPolicy | None = None,
    source_id: str = "native-fillet",
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget | None = None,
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
            maximum_cells=maximum_cells,
            tessellation=tessellation,
            source_id=source_id,
            preparations=preparations,
            budget=budget,
            include_parameter_range=include_parameter_range,
        )
    except CoordinateEnclosureResourceError:
        raise
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
    maximum_cells: int = 64,
    tessellation: BRepTessellationPolicy | None = None,
    source_id: str = "native-chamfer",
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget | None = None,
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
            maximum_cells=maximum_cells,
            tessellation=tessellation,
            source_id=source_id,
            preparations=preparations,
            budget=budget,
            include_parameter_range=include_parameter_range,
        )
    except CoordinateEnclosureResourceError:
        raise
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
    maximum_cells: int,
    tessellation: BRepTessellationPolicy | None,
    source_id: str,
    preparations: TrimRequestPreparations | None,
    budget: CoordinateEnclosureBudget | None,
    include_parameter_range: bool,
) -> BRepBlendResult:
    parameter = "radius" if radius is not None else "setback_scale"
    if not edges or len(set(edges)) != len(edges):
        raise ValueError(
            "brep_fillet.edges must contain distinct nonempty source identities; got an empty or repeated selection"
        )
    if any(edge not in model.edge_ids for edge in edges):
        raise ValueError(
            "brep_fillet.edges must belong to the source model revision; got a foreign identity"
        )
    geometry = model.require_geometry()
    indices = tuple(model.edge_ids.index(edge) for edge in edges)
    if len(indices) > 1:
        vertices = [set(geometry.edge_vertices[index]) for index in indices]
        if any(a & b for i, a in enumerate(vertices) for b in vertices[i + 1 :]):
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
    if preparations is None:
        preparations = TrimRequestPreparations()
    if budget is None:
        with coordinate_source_request(
            (model, edges, radius, setbacks)
        ) as request_budget:
            return _blend(
                model,
                edges,
                radius,
                setbacks,
                maximum_cells=maximum_cells,
                tessellation=tessellation,
                source_id=source_id,
                preparations=preparations,
                budget=request_budget,
                include_parameter_range=include_parameter_range,
            )
    incident = [
        face
        for face, loops in enumerate(geometry.face_loops)
        if any(
            geometry.coedge_edges[coedge] == indices[0]
            for loop in loops
            for coedge in loop
        )
    ]
    if len(incident) != 2:
        return _refusal(
            model,
            edges,
            "unsupported",
            "blend requires exactly two distinct incident faces",
            parameter=parameter,
        )
    patches = tuple(model.patches[face] for face in incident)
    family = _blend_family(patches)
    if family == "unsupported":
        return _refusal(
            model,
            edges,
            "unsupported",
            "brep_fillet.adjacency must belong to a supported analytic pair; got an unsupported analytic face adjacency",
            parameter=parameter,
        )
    classification = brep_edge_classification(
        model, geometry, maximum_cells=maximum_cells, coordinate_budget=budget
    )[indices[0]]
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
            indices[0],
            family,
            radius,
            setbacks,
            tessellation=tessellation,
            source_id=source_id,
            preparations=preparations,
            budget=budget,
        )
    except BRepBlendUnsupported as refusal:
        parameter_range = _blend_parameter_range(
            model, indices[0], family, radius, setbacks, budget=budget
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
    except CoordinateEnclosureResourceError:
        raise
    except ValueError as error:
        attach_parameter_range(
            error,
            _blend_parameter_range(
                model, indices[0], family, radius, setbacks, budget=budget
            ),
        )
        raise
    parameter_range = (
        _blend_parameter_range(model, indices[0], family, radius, setbacks, budget=budget)
        if include_parameter_range
        else None
    )
    bounds = jnp.empty((0, 2), dtype=jnp.float64)
    source = None
    sensitivity_status: BRepBlendSensitivityStatus = "not_applicable"
    sensitivity_reason = ""
    if radius is not None:
        bounds = _g1_bounds(
            model,
            rebuilt,
            indices[0],
            family,
            radius,
            preparations=preparations,
            budget=budget,
        )
        try:
            source = _radius_source(model, rebuilt, indices[0], family, radius)
            sensitivity_status = "available"
        except BRepBlendUnsupported as refusal:
            sensitivity_status = "unsupported"
            sensitivity_reason = str(refusal)
    return BRepBlendResult(
        status="certified",
        reason="exact analytic pair with unchanged trim incidence",
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
    *,
    budget: CoordinateEnclosureBudget,
) -> CertifiedParameterRange:
    parameter = "radius" if setbacks is None else "setback_scale"
    try:
        if family == "planes":
            if _perpendicular_plane_ends(model, edge_index):
                from ._planar_blend import planar_blend_parameter_range

                return planar_blend_parameter_range(
                    model, edge_index, setbacks, budget=budget
                )
            if setbacks is not None:
                from ._oblique_chamfer import oblique_chamfer_parameter_range

                return oblique_chamfer_parameter_range(
                    model, edge_index, setbacks, budget=budget
                )
            from ._oblique_fillet import oblique_fillet_parameter_range

            return oblique_fillet_parameter_range(model, edge_index, budget=budget)
        if family == "closed_seam":
            from ._closed_seam_blend import closed_seam_parameter_range

            budget.reserve(128, 1024)
            if not _closed_seam_range_source(model):
                return unknown_parameter_range(
                    parameter,
                    "closed seam range: remote source faces or shells lack a whole-interval collision exclusion",
                )
            return closed_seam_parameter_range(model, edge_index, setbacks, budget=budget)
        if family == "parallel_cylinders" and radius is not None:
            from ._parallel_cylinder_blend import parallel_cylinder_parameter_range

            return parallel_cylinder_parameter_range(
                model, edge_index, radius, budget=budget
            )
    except CoordinateEnclosureResourceError:
        raise
    except ValueError as error:
        # Proof refusal must not replace the construction decision.
        return unknown_parameter_range(parameter, str(error))
    return unknown_parameter_range(
        parameter,
        f"No certified feasible interval for the {family} blend family and its source trims.",
    )


def _closed_seam_range_source(model: BRepModel, /) -> bool:
    """Restrict local meridian proofs to a whole coaxial capped quadric boundary."""
    geometry = model.require_geometry()
    curved = tuple(
        patch
        for patch in model.patches
        if isinstance(patch, (CylinderPatch, ConePatch, SpherePatch))
    )
    if (
        len(curved) != 1
        or len(model.patches) not in (2, 3)
        or len(geometry.shell_faces) != 1
        or set(geometry.shell_faces[0]) != set(range(len(model.patches)))
        or geometry.solid_shells != ((0,),)
    ):
        return False
    carrier = curved[0]
    center = carrier.center if isinstance(carrier, SpherePatch) else carrier.origin
    axis = rational_vector(np.asarray(carrier.axis))
    for face, patch in enumerate(model.patches):
        if patch is carrier:
            continue
        if (
            not isinstance(patch, PlanePatch)
            or len(geometry.face_loops[face]) != 1
            or len(geometry.face_loops[face][0]) != 1
            or not np.array_equal(patch.first_axis, carrier.first_axis)
            or not np.array_equal(patch.second_axis, carrier.second_axis)
        ):
            return False
        shift = tuple(
            a - b
            for a, b in zip(
                rational_vector(np.asarray(patch.origin)),
                rational_vector(np.asarray(center)),
                strict=True,
            )
        )
        height = sum((a * b for a, b in zip(shift, axis, strict=True)), Fraction())
        if shift != tuple(height * value for value in axis):
            return False
    return True


def _blend_family(patches: tuple[AbstractSurfacePatch, ...], /) -> _BlendFamily:
    match patches:
        case (PlanePatch(), PlanePatch()):
            return "planes"
        case (PlanePatch(), CylinderPatch() | ConePatch() | SpherePatch()) | (
            CylinderPatch() | ConePatch() | SpherePatch(),
            PlanePatch(),
        ):
            return "closed_seam"
        case (CylinderPatch(), CylinderPatch()):
            return "parallel_cylinders"
        case _:
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
    preparations: TrimRequestPreparations,
    budget: CoordinateEnclosureBudget,
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
                    preparations=preparations,
                    budget=budget,
                )
            if radius is not None:
                from ._oblique_fillet import oblique_fillet

                return oblique_fillet(
                    model,
                    edge_index,
                    radius,
                    tessellation=tessellation,
                    source_id=source_id,
                    preparations=preparations,
                    budget=budget,
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
                preparations=preparations,
                budget=budget,
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
                preparations=preparations,
                budget=budget,
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
                preparations=preparations,
                budget=budget,
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
    *,
    preparations: TrimRequestPreparations,
    budget: CoordinateEnclosureBudget,
) -> Array:
    match family:
        case "planes":
            return _planar_g1_bounds(result, budget=budget)
        case "parallel_cylinders":
            from ._parallel_cylinder_blend import parallel_cylinder_g1_bounds

            return jnp.asarray(
                parallel_cylinder_g1_bounds(source, result, edge_index, budget=budget),
                dtype=jnp.float64,
            )
        case "closed_seam":
            from ._closed_seam_blend import closed_seam_tangency_certificate

            certificate = closed_seam_tangency_certificate(
                source, edge_index, radius, preparations=preparations, budget=budget
            )
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


def _planar_g1_bounds(
    result: BRepModel, /, *, budget: CoordinateEnclosureBudget
) -> Array:
    geometry = result.require_geometry()
    records = brep_edge_classification(
        result, geometry, maximum_cells=1, coordinate_budget=budget
    )
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
            LineCurve,
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
    match family:
        case "planes":
            if not _perpendicular_plane_ends(source, edge_index):
                raise BRepBlendUnsupported(
                    "brep_fillet.radius_sensitivity must support linked intersection-cap trims; got unsupported oblique-cap sensitivity"
                )
            from ._fillet_radius_map import PlanarFilletRadiusMap

            parameter_map = PlanarFilletRadiusMap(source, edge_index, radius)
        case "parallel_cylinders":
            from ..design._schema import ParameterId
            from ._parallel_cylinder_blend import ParallelCylinderRadiusMap

            parameter_map = ParallelCylinderRadiusMap(
                source,
                edge_index,
                radius,
                result,
                parameter_id=ParameterId(
                    f"{source.model_id}:edge:{edge_index}:fillet", "radius"
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
    geometry = model.require_geometry()
    first, last = geometry.edge_vertices[edge_index]
    direction = tuple(
        b - a
        for a, b in zip(
            rational_vector(np.asarray(geometry.vertex_points[first])),
            rational_vector(np.asarray(geometry.vertex_points[last])),
            strict=True,
        )
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
