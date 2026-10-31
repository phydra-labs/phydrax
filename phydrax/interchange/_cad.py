#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Shared native CAD interchange policies, coverage and definition-preserving staging.

Format readers (STEP, IGES, external BRep text) lower source entities into a
`CadStage`: vertices, carriers, oriented edges, coedges and trimmed faces in
the requested coordinate contract. Missing p-curves are recovered only from
exact algebraic source maps; unsupported maps, topology and carrier structures
are refused with their dependency chain. Staging retains source assembly
definitions and uses as a shared graph without expanding occurrences.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from math import atan2, isfinite, pi
from pathlib import Path
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from .._fingerprint import canonical_fingerprint
from .._physical import SpatialCoordinateContract
from .._strict import StrictModule
from .._validation import finite_real_scalar, positive_finite_float
from ..discretization._exact_algebraic_geometry import (
    ExactScalar,
    RadicalElement,
    rational_coefficient,
)
from ..geometry._atlas import TrimDomain
from ..geometry._interval_enclosure import prepare_interval_function
from ..geometry.brep._assembly import BRepAssemblyGraph
from ..geometry.brep._boolean_core import AffinePCurve, PeriodicPCurve
from ..geometry.brep._carrier_pieces import carrier_payload
from ..geometry.brep._conic_gaps import _affine_coefficients
from ..geometry.brep._constructors import (
    assemble_brep_model,
    BRepTessellationPolicy,
)
from ..geometry.brep._correspondence import (
    _affine_bspline_pcurve,
    _affine_surface_frame,
    _plane_parameter_vector,
    exact_source_lift,
    exact_sphere_pole,
    native_edge_pcurve,
    torus_circle_pcurve,
)
from ..geometry.brep._intersection import (
    closes_native_turn,
    CurveSurfaceIntersectionRoot,
    endpoint_root_payload,
    NativePeriodEndpoint,
    RootEndpoint,
    TrimIntersectionRoot,
    TrimRootEndpoint,
)
from ..geometry.brep._intersection_curve import (
    CurveTrimSegment,
    exact_curve_parameter_point,
    IntersectionCurve,
    IntersectionPCurve,
    IntersectionSide,
    SurfaceRegion,
)
from ..geometry.brep._model import (
    BRepEntityId,
    BRepGeometry,
    BRepModel,
)
from ..geometry.brep._patches import (
    AbstractCurve,
    AbstractSurfacePatch,
    BSplineCurve,
    BSplineSurfacePatch,
    CircleCurve,
    ConePatch,
    CylinderPatch,
    EllipseCurve,
    ExactLineCurve,
    ExtrusionSurface,
    HyperbolaCurve,
    LineCurve,
    OffsetCurve,
    OffsetSurface,
    ParabolaCurve,
    PlanePatch,
    RevolutionSurface,
    RuledSurface,
    SpherePatch,
    surface_differential,
    SurfaceIsoparametricCurve,
    TorusPatch,
)
from ..geometry.brep._placed import PlacedSurface
from ..geometry.brep._root_bindings import (
    BRepCurveSurfaceLift,
    BRepRootSupport,
    BRepVertexRoot,
)
from ..geometry.brep._source_correspondence import (
    _affine_curve_coefficients,
    _harmonic_coefficients,
    norm_upper,
)
from ..geometry.brep._trim_helpers import period_offset, PeriodOffset
from ..typing import checked, parse
from ._report import (
    AdapterError,
    AdapterFormatProfile,
    AdapterLoss,
    AdapterReport,
    AdapterStatus,
    AdapterWaiver,
)
from ._step_metadata import StepEntityMetadata


_TWO_PI = 2.0 * pi
# Exchange formats print reals with finitely many decimal digits; an authored
# value this close (relative to its scale) to a symbolic source value is it.
_AUTHORED_DECIMAL_RELATIVE = 1.0e-12
_ZERO_UV = (Fraction(0), Fraction(0))

CadRefusalReason: TypeAlias = Literal[
    "unsupported-entity",
    "malformed",
    "dangling-reference",
    "cyclic-reference",
    "limit",
    "units",
    "inconsistent-geometry",
    "inverted-shell",
    "non-manifold",
    "inexact-export",
]


# ------------------------------------------------------------------ policies


@dataclass(frozen=True, slots=True)
class CadCurveFitPolicy:
    """Error targets for continuous physical and chart-coordinate fit bounds."""

    tolerance: float = 1.0e-10
    parameter_tolerance: float | None = None
    policy_id: str = field(init=False)

    def __post_init__(self) -> None:
        tolerance = positive_finite_float(self.tolerance, "CadCurveFitPolicy.tolerance")
        parameter_tolerance = (
            tolerance
            if self.parameter_tolerance is None
            else positive_finite_float(
                self.parameter_tolerance, "CadCurveFitPolicy.parameter_tolerance"
            )
        )
        object.__setattr__(self, "tolerance", tolerance)
        object.__setattr__(self, "parameter_tolerance", parameter_tolerance)
        object.__setattr__(
            self,
            "policy_id",
            canonical_fingerprint(
                {
                    "kind": "cad-curve-fit-policy",
                    "tolerance": tolerance,
                    "parameter_tolerance": parameter_tolerance,
                }
            ),
        )

    @property
    def resolved_parameter_tolerance(self) -> float:
        return (
            self.tolerance
            if self.parameter_tolerance is None
            else self.parameter_tolerance
        )


@dataclass(frozen=True, slots=True)
class CadImportPolicy:
    """Target coordinates, physical tolerances and occurrence placement policy.

    ``maximum_placement_deviation`` is an outward point-displacement limit in
    target length units, certified per assembly occurrence using its local solid
    extent. Zero retains exact placement admission.
    """

    coordinate_contract: SpatialCoordinateContract
    tessellation: BRepTessellationPolicy | None = None
    pcurve_fit: CadCurveFitPolicy | None = None
    relative_geometric_tolerance: float = 1.0e-8
    maximum_placement_deviation: float = 0.0
    policy_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.coordinate_contract, SpatialCoordinateContract):
            raise TypeError(
                f"CadImportPolicy.coordinate_contract must be SpatialCoordinateContract; got {type(self.coordinate_contract).__name__}"
            )
        if self.coordinate_contract.coordinate_system != "cartesian":
            raise ValueError(
                f"CadImportPolicy.coordinate_contract.coordinate_system must be cartesian; got {self.coordinate_contract.coordinate_system!r}"
            )
        if self.tessellation is not None and not isinstance(
            self.tessellation, BRepTessellationPolicy
        ):
            raise TypeError(
                f"CadImportPolicy.tessellation must be BRepTessellationPolicy or None; got {type(self.tessellation).__name__}"
            )
        if self.pcurve_fit is not None and not isinstance(
            self.pcurve_fit, CadCurveFitPolicy
        ):
            raise TypeError(
                f"CadImportPolicy.pcurve_fit must be CadCurveFitPolicy or None; got {type(self.pcurve_fit).__name__}"
            )
        tolerance = float(self.relative_geometric_tolerance)
        if (
            isinstance(self.relative_geometric_tolerance, (bool, np.bool_))
            or not isfinite(tolerance)
            or tolerance < 0.0
        ):
            raise ValueError(
                f"CadImportPolicy.relative_geometric_tolerance must be finite and nonnegative; got {self.relative_geometric_tolerance!r}"
            )
        placement_limit = float(self.maximum_placement_deviation)
        if (
            isinstance(self.maximum_placement_deviation, (bool, np.bool_))
            or not isfinite(placement_limit)
            or placement_limit < 0.0
        ):
            raise ValueError(
                "CadImportPolicy.maximum_placement_deviation must be finite and nonnegative; got an invalid limit."
            )
        object.__setattr__(self, "maximum_placement_deviation", placement_limit)
        object.__setattr__(self, "relative_geometric_tolerance", tolerance)
        object.__setattr__(
            self,
            "policy_id",
            canonical_fingerprint(
                {
                    "kind": "cad-import-policy",
                    "spatial_id": self.coordinate_contract.spatial_id,
                    "tessellation": (
                        None if self.tessellation is None else self.tessellation.policy_id
                    ),
                    "pcurve_fit": None
                    if self.pcurve_fit is None
                    else self.pcurve_fit.policy_id,
                    "relative_geometric_tolerance": tolerance,
                    **(
                        {"maximum_placement_deviation": placement_limit}
                        if placement_limit > 0.0
                        else {}
                    ),
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class CadExportPolicy:
    """Export exact carriers or an explicitly requested certified intersection fit."""

    intersection_approximation: CadCurveFitPolicy | None = None
    policy_id: str = field(init=False)

    def __post_init__(self) -> None:
        if self.intersection_approximation is not None and not isinstance(
            self.intersection_approximation, CadCurveFitPolicy
        ):
            raise TypeError(
                f"CadExportPolicy.intersection_approximation must be CadCurveFitPolicy or None; got {type(self.intersection_approximation).__name__}"
            )
        object.__setattr__(
            self,
            "policy_id",
            canonical_fingerprint(
                {
                    "kind": "cad-export-policy",
                    "intersection_approximation": None
                    if self.intersection_approximation is None
                    else self.intersection_approximation.policy_id,
                }
            ),
        )


@checked
def export_loss_waivers(
    losses: Sequence[AdapterLoss],
    policy: CadExportPolicy,
    format_name: str,
    /,
) -> tuple[AdapterWaiver, ...]:
    """Bind declared format losses to the caller's explicit export policy."""
    return tuple(
        AdapterWaiver(
            loss,
            f"Explicit {format_name} export under CAD policy {policy.policy_id} "
            "accepts this declared representation change.",
        )
        for loss in losses
    )


# ------------------------------------------------------ refusals and evidence


@dataclass(frozen=True, slots=True)
class CadRefusal:
    """Why one external entity was refused, with its dependency chain.

    ``chain`` lists external entity labels from the admitted root down to the
    refused ``entity`` (for example ``('#15 MANIFOLD_SOLID_BREP', ...,
    '#40 OFFSET_SURFACE')``). Labels are provenance only.
    """

    reason: CadRefusalReason
    entity: str
    chain: tuple[str, ...]
    message: str

    def __post_init__(self) -> None:
        parse(self.reason, CadRefusalReason, "CadRefusal.reason")
        if not isinstance(self.message, str) or not self.message:
            raise ValueError(
                f"CadRefusal.message must be a nonempty string; got {self.message!r}"
            )
        object.__setattr__(self, "chain", tuple(str(item) for item in self.chain))


def _refusal_status(reason: CadRefusalReason, /) -> AdapterStatus:
    match reason:
        case "unsupported-entity" | "inexact-export":
            return AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC
        case "malformed" | "dangling-reference" | "cyclic-reference" | "limit":
            return AdapterStatus.MALFORMED_SOURCE
        case "units" | "inconsistent-geometry" | "non-manifold" | "inverted-shell":
            return AdapterStatus.INCONSISTENT_SOURCE
        case _:
            raise ValueError(
                f"CadRefusal.reason must be a supported refusal selector; got {reason!r}"
            )


class CadCodecInvariantError(RuntimeError):
    """Indicate an inconsistent prepared CAD codec state."""


class CadInterchangeError(AdapterError):
    """Fail-closed CAD import/export refusal carrying its dependency chain."""

    refusal: CadRefusal

    def __init__(self, refusal: CadRefusal, /) -> None:
        self.refusal = refusal
        chain = " -> ".join(refusal.chain)
        detail = f"{refusal.message} [{refusal.entity}]"
        super().__init__(
            _refusal_status(refusal.reason),
            detail if not chain else f"{detail} via {chain}",
        )


def refuse(
    reason: CadRefusalReason, entity: str, message: str, /, chain: Sequence[str] = ()
) -> CadInterchangeError:
    """Build the typed refusal for one external entity (raised by the caller)."""
    return CadInterchangeError(CadRefusal(reason, entity, tuple(chain), message))


def _rational_affine_pcurve(
    curve: AbstractCurve,
    matrix: Sequence[Sequence[ExactScalar]],
    offset: Sequence[ExactScalar],
    label: str,
    /,
) -> AffinePCurve:
    try:
        exact_matrix = tuple(
            tuple(rational_coefficient(value, role=label) for value in row)
            for row in matrix
        )
        exact_offset = tuple(rational_coefficient(value, role=label) for value in offset)
    except ValueError as error:
        raise refuse(
            "unsupported-entity",
            label,
            "A native p-curve affine map requires exact rational coefficients.",
        ) from error
    return AffinePCurve(
        curve,
        np.asarray(exact_matrix, dtype=object),
        np.asarray(exact_offset, dtype=object),
    )


@dataclass(frozen=True, slots=True)
class CadCoverage:
    """Admitted entity counts and exact or source-supplied p-curve coverage."""

    entity_counts: tuple[tuple[str, int], ...]
    pcurves_honored: int
    pcurves_exact: int
    degenerate_edges: int
    natural_boundaries: int

    def __post_init__(self) -> None:
        if not isinstance(self.entity_counts, tuple) or any(
            not isinstance(pair, tuple) or len(pair) != 2 for pair in self.entity_counts
        ):
            raise ValueError(
                f"CadCoverage.entity_counts must contain immutable name/count pairs; got {self.entity_counts!r}"
            )
        names: set[str] = set()
        for name, count in self.entity_counts:
            if not isinstance(name, str) or not name or name in names:
                raise ValueError(
                    f"CadCoverage.entity_counts must have unique nonempty entity names; got {name!r}"
                )
            names.add(name)
            if (
                isinstance(count, (bool, np.bool_))
                or not isinstance(count, int)
                or count < 0
            ):
                raise ValueError(
                    f"CadCoverage.entity_counts must contain nonnegative integer counts; got {count!r}"
                )
        for name, value in (
            ("pcurves_honored", self.pcurves_honored),
            ("pcurves_exact", self.pcurves_exact),
            ("degenerate_edges", self.degenerate_edges),
            ("natural_boundaries", self.natural_boundaries),
        ):
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(
                    f"CadCoverage.{name} must be a nonnegative integer; got {value!r}"
                )


@dataclass(frozen=True, slots=True)
class _ChartClosureRepair:
    """One clamped p-curve end pole moved onto its predecessor's chart end."""

    coedge: int
    label: str
    pole: int
    previous: tuple[Fraction, ...]
    moved: tuple[Fraction, ...]

    @property
    def displacement(self) -> Fraction:
        """The exact max-norm chart distance the pole moved."""
        return max(
            abs(moved - previous)
            for moved, previous in zip(self.moved, self.previous, strict=True)
        )

    def loss(self, gap_bound: float, tolerance: float, /) -> AdapterLoss:
        """Declare the repair with the face-gap proof that admitted it."""
        previous = ", ".join(float(value).hex() for value in self.previous)
        moved = ", ".join(float(value).hex() for value in self.moved)
        return AdapterLoss(
            f"geometry.chart-closure:{self.label}",
            "import",
            "transformed",
            f"coedge={self.coedge}, pole={self.pole}, previous=({previous}), "
            f"moved=({moved}), chart-displacement={float(self.displacement):.17g}, "
            f"certified-face-gap={gap_bound:.17g}, import-tolerance={tolerance:.17g}; "
            "a clamped p-curve end pole was moved onto its predecessor's exact "
            "chart end so the trim loop closes, and the edge's face gap was "
            "re-certified against the import tolerance.",
            changes_interpretation=False,
        )


def _clamped_pole_moved_to(
    curve: AbstractCurve,
    parameter: float,
    target: tuple[ExactScalar, ...],
    /,
) -> tuple[BSplineCurve, int, tuple[Fraction, ...], tuple[Fraction, ...]] | None:
    """The clamped B-spline with its end pole at ``parameter`` moved onto ``target``.

    Returns the curve, the moved pole index and its previous and moved exact
    coordinates. ``None`` unless ``parameter`` is a clamped end of ``curve``
    and both poles have distinct binary64-representable rational coordinates.
    """
    if not isinstance(curve, BSplineCurve) or len(target) != curve.ambient_dimension:
        return None
    knots = curve.exact_knots
    controls = curve.exact_control_points
    degree, count = curve.degree, len(controls)
    exact = Fraction.from_float(parameter)
    if all(knot == exact for knot in knots[: degree + 1]):
        index = 0
    elif all(knot == exact for knot in knots[count:]):
        index = count - 1
    else:
        return None
    pole = tuple(value for value in target if isinstance(value, Fraction))
    previous = tuple(value for value in controls[index] if isinstance(value, Fraction))
    if (
        len(pole) != len(target)
        or len(previous) != len(target)
        or any(Fraction(float(value)) != value for value in pole)
        or pole == previous
    ):
        return None
    exact_controls = (*controls[:index], pole, *controls[index + 1 :])
    numeric = np.array(curve.control_points, dtype=np.float64)
    numeric[index] = [float(value) for value in pole]
    closed = BSplineCurve(
        numeric,
        curve.weights,
        curve.knots,
        degree,
        exact_control_points=exact_controls,
        exact_weights=curve.exact_weights,
        exact_knots=knots,
    )
    return closed, index, previous, pole


def _bind_stage_face_gap_bounds(
    geometry: BRepGeometry,
    patches: tuple[AbstractSurfacePatch, ...],
    parameter_bounds: np.ndarray,
    maximum_gap: float,
    /,
) -> BRepGeometry:
    """Publish only independently proved gaps within the import requirement."""
    from ..geometry.brep._model import (
        _certify_edge_face_gap_bounds,
        _validate_degenerate_coedge_poles,
    )

    if not isfinite(maximum_gap) or maximum_gap < 0.0:
        raise ValueError("The import gap requirement must be finite and nonnegative.")
    _validate_degenerate_coedge_poles(geometry, patches)
    bounds = _certify_edge_face_gap_bounds(
        geometry,
        patches,
        parameter_bounds=parameter_bounds,
        exact_unsupported=True,
        certified_degenerate_coedges=True,
    )
    if bounds is None:
        raise CadCodecInvariantError("CAD publication omitted its source face-gap proof.")
    if any(bound > maximum_gap for bound in bounds):
        raise ValueError("Source face gaps exceed the declared import tolerance.")
    if not any(bounds):
        return geometry
    return BRepGeometry(
        vertex_points=geometry.vertex_points,
        vertex_exact_points=geometry.vertex_exact_points,
        vertex_roots=geometry.vertex_roots,
        edge_endpoint_roots=geometry.edge_endpoint_roots,
        coedge_endpoint_roots=geometry.coedge_endpoint_roots,
        edge_vertex_gap_bounds=geometry.edge_vertex_gap_bounds,
        edge_face_gap_bounds=bounds,
        curves=geometry.curves,
        edge_curves=geometry.edge_curves,
        edge_ranges=geometry.edge_ranges,
        edge_vertices=geometry.edge_vertices,
        pcurves=geometry.pcurves,
        coedge_edges=geometry.coedge_edges,
        coedge_senses=geometry.coedge_senses,
        face_loops=geometry.face_loops,
        shell_faces=geometry.shell_faces,
        shell_orientations=geometry.shell_orientations,
        solid_shells=geometry.solid_shells,
        assembly_graph=geometry.assembly_graph,
    )


def _gap_entity_bounds(
    model: BRepModel,
    patches: Sequence[AbstractSurfacePatch] | None = None,
    /,
) -> dict[BRepEntityId, float]:
    """Map certified source gaps using the model or an exact exported surface bank."""
    model.require_current()
    geometry = model.geometry
    patches_ = model.patches if patches is None else tuple(patches)
    if len(patches_) != len(model.patches):
        raise CadCodecInvariantError(
            "CAD gap certification requires one surface carrier per source face."
        )
    bounds: dict[BRepEntityId, float] = {}

    def admit(kind: str, index: int, value: float, /) -> None:
        entity = BRepEntityId(model.source_id, model.source_revision, kind, index)
        bounds[entity] = max(bounds.get(entity, 0.0), float(value))

    vertex_errors = np.asarray(geometry.vertex_evaluation_bounds, dtype=np.float64)
    for vertex, value in enumerate(vertex_errors):
        admit("vertex", vertex, float(value))

    endpoint_gaps = geometry.edge_vertex_gap_bounds
    if endpoint_gaps is None:
        endpoint_gaps = ((0.0, 0.0),) * len(geometry.edge_curves)
    coedge_faces = [-1] * len(geometry.coedge_edges)
    for face, loops in enumerate(geometry.face_loops):
        for loop in loops:
            for coedge in loop:
                if coedge_faces[coedge] not in (-1, face):
                    raise ValueError("A source coedge must belong to exactly one face.")
                coedge_faces[coedge] = face
    edge_coedges: list[list[int]] = [[] for _ in range(len(geometry.edge_curves))]
    for coedge, edge in enumerate(geometry.coedge_edges):
        if coedge_faces[coedge] < 0:
            raise ValueError("A source coedge must belong to a face.")
        edge_coedges[edge].append(coedge)

    from ..geometry.brep._model import (
        _certify_edge_face_gap_bounds,
        _validate_degenerate_coedge_poles,
    )

    if patches is not None:
        from ..geometry.brep._patches import _validate_surface_source_shadows

        for patch in patches_:
            _validate_surface_source_shadows(patch)
        _validate_degenerate_coedge_poles(geometry, patches_)

    face_gaps = _certify_edge_face_gap_bounds(
        geometry,
        patches_,
        parameter_bounds=model.parameter_bounds,
        declared_bounds=geometry.edge_face_gap_bounds,
        certified_degenerate_coedges=True,
    )
    face_bounds: list[float] | None = None
    edge_face_bounds: list[float] | None = None
    if face_gaps is not None:
        face_bounds = [0.0] * len(patches_)
        edge_face_bounds = [0.0] * len(geometry.edge_curves)
        for coedge, value in enumerate(face_gaps):
            face = coedge_faces[coedge]
            edge = geometry.coedge_edges[coedge]
            face_bounds[face] = max(face_bounds[face], float(value))
            edge_face_bounds[edge] = max(edge_face_bounds[edge], float(value))
    for edge, endpoint_pair in enumerate(endpoint_gaps):
        endpoint_bound = max(float(endpoint_pair[0]), float(endpoint_pair[1]))
        for vertex, value in zip(
            geometry.edge_vertices[edge], endpoint_pair, strict=True
        ):
            admit("vertex", vertex, float(value))
        face_bound = None if edge_face_bounds is None else edge_face_bounds[edge]
        if face_bound is None and edge_coedges[edge]:
            continue
        admit(
            "edge",
            edge,
            max(endpoint_bound, 0.0 if face_bound is None else face_bound),
        )
    if face_bounds is not None:
        for face, value in enumerate(face_bounds):
            admit("face", face, value)
    return bounds


def _tolerance_envelope(
    certified: float | None,
    declared: float | None,
) -> float | None:
    """Return the available bound maximum; None denotes absent evidence."""
    if certified is None:
        return declared
    if declared is None:
        return certified
    return max(certified, declared)


class _CadDeclaredTolerances(StrictModule):
    source_bounds: tuple[float | None, ...]
    source_tolerance: float | None
    losses: tuple[AdapterLoss, ...]


def _prepare_declared_tolerances(
    entities: tuple[BRepEntityId, ...],
    declared: Sequence[tuple[BRepEntityId | None, float]],
    limit: float,
) -> _CadDeclaredTolerances:
    represented = set(entities)
    source: dict[BRepEntityId, float] = {}
    global_source: float | None = None
    failed: dict[str, list[str]] = {}
    for entity, value in declared:
        value = finite_real_scalar(value, "build_cad_import_tolerance_evidence.declared")
        if value < 0.0:
            raise ValueError(
                f"build_cad_import_tolerance_evidence.declared must contain nonnegative tolerances; got {value!r}"
            )
        carried = entity is None or entity in represented
        if entity is None:
            global_source = value if global_source is None else max(global_source, value)
        elif carried:
            source[entity] = max(source.get(entity, value), value)
        if not carried or value > limit:
            kind = "global" if entity is None else entity.kind
            label = (
                "global"
                if entity is None
                else f"{entity.kind}:{entity.index}@{entity.source_revision}/{entity.occurrence_path!r}"
            )
            reason = "unrepresented-entity" if not carried else "declared-limit-exceeded"
            failed.setdefault(kind, []).append(
                f"entity={label}, declared={value:.17g}, limit={limit:.17g}, reason={reason}"
            )
    return _CadDeclaredTolerances(
        source_bounds=tuple(
            _tolerance_envelope(source.get(entity), global_source) for entity in entities
        ),
        source_tolerance=global_source,
        losses=tuple(
            AdapterLoss(
                f"source-tolerance:{kind}",
                "import",
                "transformed",
                "Source tolerance exceeds the imported model declared limit or cannot be carried; "
                + "; ".join(sorted(details)),
                changes_interpretation=False,
            )
            for kind, details in sorted(failed.items())
        ),
    )


def unrepresentable_cad_tolerance_loss(
    entity: str, declared: float, source_unit: str, limit: float, /
) -> AdapterLoss:
    """Record a valid source declaration whose converted bound is unavailable."""
    return AdapterLoss(
        f"source-tolerance:unrepresentable:{entity}",
        "import",
        "dropped",
        f"entity={entity}, declared={declared:.17g}, source-unit={source_unit}, "
        f"limit={limit:.17g}, reason=unrepresentable-target-bound",
        changes_interpretation=False,
    )


class CadImportToleranceEvidence(StrictModule):
    """Record native gap bounds and separately carried source uncertainty envelopes.

    ``certified_bounds`` are unchanged native gap bounds. ``source_bounds`` retain
    declarations in target length units, including global declarations applied to
    each represented entity. ``upper_bounds`` take their maximum; they do not
    replace native gap evidence. None denotes an absent bound.
    """

    entities: tuple[BRepEntityId, ...] = eqx.field(static=True)
    certified_bounds: tuple[float | None, ...] = eqx.field(static=True)
    source_bounds: tuple[float | None, ...] = eqx.field(static=True)
    upper_bounds: tuple[float | None, ...] = eqx.field(static=True)
    source_tolerance: float | None = eqx.field(static=True)
    declared_limit: float = eqx.field(static=True)
    uncertainty: float | None = eqx.field(static=True)
    losses: tuple[AdapterLoss, ...] = eqx.field(static=True)
    model_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        model: BRepModel,
        declared: Sequence[tuple[BRepEntityId | None, float]],
        /,
        *,
        unrepresentable_losses: Sequence[AdapterLoss] = (),
    ) -> None:
        entities = tuple(sorted((*model.vertex_ids, *model.edge_ids, *model.face_ids)))
        prepared = _prepare_declared_tolerances(
            entities, declared, model.report.curve_surface_tolerance
        )
        certified = _gap_entity_bounds(model)
        self.entities = entities
        self.certified_bounds = tuple(certified.get(entity) for entity in entities)
        self.source_bounds = prepared.source_bounds
        self.upper_bounds = tuple(
            _tolerance_envelope(native, source)
            for native, source in zip(
                self.certified_bounds, self.source_bounds, strict=True
            )
        )
        self.source_tolerance = prepared.source_tolerance
        self.declared_limit = model.report.curve_surface_tolerance
        available = tuple(value for value in self.upper_bounds if value is not None)
        self.uncertainty = max(available, default=self.source_tolerance)
        self.losses = (*prepared.losses, *unrepresentable_losses)
        self.model_id = model.model_id
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "cad-import-tolerance-evidence",
                "model": self.model_id,
                "limit": self.declared_limit,
                "entities": [
                    [
                        entity.source_revision,
                        entity.kind,
                        entity.index,
                        list(entity.occurrence_path),
                    ]
                    for entity in entities
                ],
                "certified": [
                    None if bound is None else float(bound).hex()
                    for bound in self.certified_bounds
                ],
                "source": [
                    None if bound is None else float(bound).hex()
                    for bound in self.source_bounds
                ],
                "upper": [
                    None if bound is None else float(bound).hex()
                    for bound in self.upper_bounds
                ],
                "source-tolerance": self.source_tolerance,
                "losses": [loss.loss_id for loss in self.losses],
            }
        )


@checked
def build_cad_import_tolerance_evidence(
    model: BRepModel,
    declared: Sequence[tuple[BRepEntityId | None, float]],
    /,
    *,
    unrepresentable_losses: Sequence[AdapterLoss] = (),
) -> CadImportToleranceEvidence:
    """Carry raw source tolerances beside the model's certified source gaps."""
    return CadImportToleranceEvidence(
        model,
        declared,
        unrepresentable_losses=unrepresentable_losses,
    )


def _canonical_source_tolerances(
    declarations: tuple[tuple[str, float], ...],
) -> tuple[tuple[str, float], ...]:
    """Normalize raw source-length declarations by their external field references."""
    if not isinstance(declarations, tuple):
        raise TypeError(
            f"CadImportResult.source_tolerances must be an immutable tuple; got {type(declarations).__name__}"
        )
    rows: list[tuple[str, float]] = []
    references: set[str] = set()
    for declaration in declarations:
        if not isinstance(declaration, tuple) or len(declaration) != 2:
            raise ValueError(
                f"CadImportResult.source_tolerances must contain reference/value pairs; got {declaration!r}"
            )
        reference, value = declaration
        if not isinstance(reference, str):
            raise TypeError(
                f"CadImportResult.source_tolerances.reference must be a string; got {type(reference).__name__}"
            )
        if not reference or not reference.strip() or reference in references:
            raise ValueError(
                f"CadImportResult.source_tolerances.reference must be nonempty and unique; got {reference!r}"
            )
        references.add(reference)
        tolerance = finite_real_scalar(
            value, f"CadImportResult.source_tolerances.{reference}"
        )
        if tolerance < 0.0:
            raise ValueError(
                f"CadImportResult.source_tolerances.{reference} must be nonnegative; got {tolerance!r}"
            )
        rows.append((reference, tolerance))
    return tuple(sorted(rows))


@dataclass(frozen=True, slots=True)
class CadImportResult:
    """Return native B-Rep geometry, source identity and codec evidence."""

    model: BRepModel
    format: str
    schema: str
    source_digest: str
    source_length_unit_meters: float
    source_angle_unit_radians: float
    report: AdapterReport
    coverage: CadCoverage
    provenance: tuple[tuple[str, str], ...]
    entity_metadata: StepEntityMetadata | None
    result_id: str = field(init=False)
    source_tolerance: float | None = None
    source_tolerances: tuple[tuple[str, float], ...] = ()
    tolerance_evidence: CadImportToleranceEvidence | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.model, BRepModel):
            raise TypeError(
                f"CadImportResult.model must be BRepModel; got {type(self.model).__name__}"
            )
        if (
            not isinstance(self.source_digest, str)
            or len(self.source_digest) != 64
            or self.source_digest != self.source_digest.lower()
            or any(
                character not in "0123456789abcdef" for character in self.source_digest
            )
        ):
            raise ValueError(
                "CadImportResult.source_digest must be a lowercase SHA-256 digest."
            )
        if not isinstance(self.report, AdapterReport) or not self.report.valid:
            raise ValueError(
                f"CadImportResult.report must be a valid AdapterReport; got type={type(self.report).__name__}, valid={self.report.valid if isinstance(self.report, AdapterReport) else 'not-AdapterReport'}"
            )
        if not isinstance(self.coverage, CadCoverage):
            raise TypeError(
                f"CadImportResult.coverage must be CadCoverage; got {type(self.coverage).__name__}"
            )
        if self.entity_metadata is not None and not isinstance(
            self.entity_metadata, StepEntityMetadata
        ):
            raise TypeError(
                f"CadImportResult.entity_metadata must be StepEntityMetadata or None; got {type(self.entity_metadata).__name__}"
            )
        if self.entity_metadata is not None and (
            self.format != "step" or self.entity_metadata.model_id != self.model.model_id
        ):
            raise ValueError(
                f"CadImportResult.entity_metadata must belong to the imported STEP model; got format={self.format!r}, metadata_model={self.entity_metadata.model_id!r}, model={self.model.model_id!r}"
            )
        if self.source_tolerance is not None:
            tolerance = finite_real_scalar(
                self.source_tolerance, "CadImportResult.source_tolerance"
            )
            if tolerance < 0.0:
                raise ValueError(
                    f"CadImportResult.source_tolerance must be nonnegative; got {tolerance!r}"
                )
            object.__setattr__(self, "source_tolerance", tolerance)
        object.__setattr__(
            self,
            "source_tolerances",
            _canonical_source_tolerances(self.source_tolerances),
        )
        if self.tolerance_evidence is not None:
            if not isinstance(self.tolerance_evidence, CadImportToleranceEvidence):
                raise TypeError(
                    f"CadImportResult.tolerance_evidence must be CadImportToleranceEvidence or None; got {type(self.tolerance_evidence).__name__}"
                )
            if self.tolerance_evidence.model_id != self.model.model_id:
                raise ValueError(
                    f"CadImportResult.tolerance_evidence.model_id must match the imported model; got {self.tolerance_evidence.model_id!r}, expected={self.model.model_id!r}"
                )
        positive_finite_float(
            self.source_length_unit_meters, "CadImportResult.source_length_unit_meters"
        )
        positive_finite_float(
            self.source_angle_unit_radians, "CadImportResult.source_angle_unit_radians"
        )
        object.__setattr__(
            self,
            "result_id",
            canonical_fingerprint(
                {
                    "kind": "cad-import-result",
                    "model": self.model.model_id,
                    "format": self.format,
                    "schema": self.schema,
                    "source_digest": self.source_digest,
                    "units": [
                        self.source_length_unit_meters,
                        self.source_angle_unit_radians,
                    ],
                    "source_tolerance": self.source_tolerance,
                    "source_tolerances": [list(row) for row in self.source_tolerances],
                    "tolerance_evidence": None
                    if self.tolerance_evidence is None
                    else self.tolerance_evidence.evidence_id,
                    "report": self.report.report_id,
                    "coverage": {
                        "entity_counts": [
                            list(pair) for pair in self.coverage.entity_counts
                        ],
                        "pcurves_honored": self.coverage.pcurves_honored,
                        "pcurves_exact": self.coverage.pcurves_exact,
                        "degenerate_edges": self.coverage.degenerate_edges,
                        "natural_boundaries": self.coverage.natural_boundaries,
                    },
                    "provenance": [list(pair) for pair in self.provenance],
                    **(
                        {}
                        if self.entity_metadata is None
                        else {"entity_metadata": self.entity_metadata.metadata_id}
                    ),
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class CadExportResult:
    """Published CAD path, entity inventory, provenance and codec evidence."""

    path: Path
    format: str
    schema: str
    report: AdapterReport
    entity_counts: tuple[tuple[str, int], ...]
    provenance: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            raise TypeError("CadExportResult.path must be a pathlib.Path.")
        if not isinstance(self.report, AdapterReport) or not self.report.valid:
            raise ValueError(
                "CadExportResult.report must be a valid AdapterReport; got invalid report"
            )
        if not isinstance(self.entity_counts, tuple) or any(
            not isinstance(pair, tuple)
            or len(pair) != 2
            or not isinstance(pair[0], str)
            or not pair[0]
            or isinstance(pair[1], bool)
            or not isinstance(pair[1], int)
            or pair[1] < 0
            for pair in self.entity_counts
        ):
            raise ValueError(
                "CadExportResult.entity_counts must contain named nonnegative counts."
            )
        if not isinstance(self.provenance, tuple) or any(
            not isinstance(pair, tuple)
            or len(pair) != 2
            or any(not isinstance(value, str) or not value for value in pair)
            for pair in self.provenance
        ):
            raise ValueError(
                f"CadExportResult.provenance must contain nonempty source-key/external-reference pairs; got {self.provenance!r}"
            )


@checked
def build_cad_export_report(
    model: BRepModel,
    target_digest: str,
    policy: CadExportPolicy,
    /,
    *,
    format_name: str,
    qualifiers: Mapping[str, str],
    preserved_fields: Sequence[str],
    coordinate_mapping: Sequence[str] = (),
    assumptions: Sequence[str] = (),
    target_id: str | None = None,
    stage: str = "host-export",
    source_profile: AdapterFormatProfile | None = None,
    extra_losses: Sequence[AdapterLoss] = (),
    waiver_format_name: str | None = None,
) -> AdapterReport:
    """Build format and physical-loss evidence for an exported native model."""
    losses = tuple(extra_losses)
    return AdapterReport(
        AdapterStatus.DECLARED_LOSS if losses else AdapterStatus.LOSSLESS,
        "phydrax-brep",
        format_name,
        source_id=model.model_id,
        target_id=target_digest if target_id is None else target_id,
        stage=stage,
        source_profile=source_profile,
        target_profile=AdapterFormatProfile(
            format_name,
            qualifiers=dict(qualifiers),
        ),
        preserved_fields=tuple(preserved_fields),
        coordinate_mapping=tuple(coordinate_mapping),
        assumptions=tuple(assumptions),
        losses=losses,
        waivers=export_loss_waivers(
            losses,
            policy,
            format_name.upper() if waiver_format_name is None else waiver_format_name,
        ),
    )


class CadExportTolerances(StrictModule):
    """Certified physical-length bounds rounded upward for 17-digit output."""

    vertex_bounds: tuple[float, ...]
    edge_bounds: tuple[float, ...]
    face_bounds: tuple[float, ...]
    uncertainty: float


def _written_upper_bound(value: float) -> float:
    if not isfinite(value) or value < 0.0:
        raise refuse(
            "inexact-export",
            "tolerance",
            f"CadExportTolerances.uncertainty must have a finite nonnegative certified bound; got {value!r}",
        )
    # Decimal output must not round below the binary certified endpoint.
    endpoint = Fraction(value)
    while Fraction(format(value, ".17g")) < endpoint:
        if value == np.finfo(np.float64).max:
            raise refuse(
                "inexact-export",
                "tolerance",
                f"CadExportTolerances.uncertainty must have finite upward 17-digit output; got {value!r}",
            )
        value = float(np.nextafter(value, np.inf))
    return value


@checked
def _certified_cad_export_tolerances(
    model: BRepModel,
    patches: Sequence[AbstractSurfacePatch],
    /,
) -> CadExportTolerances:
    geometry = model.geometry
    if geometry is None:
        raise refuse(
            "inexact-export",
            model.source_id,
            "BRepModel.geometry must be present for CAD export; got None",
        )
    bounds = _gap_entity_bounds(model, patches)
    rows: list[tuple[float, ...]] = []
    for kind, count in (
        ("vertex", len(geometry.vertex_roots)),
        ("edge", len(geometry.edge_curves)),
        ("face", len(geometry.face_loops)),
    ):
        row: list[float] = []
        for index in range(count):
            entity = BRepEntityId(model.source_id, model.source_revision, kind, index)
            if entity not in bounds:
                raise refuse(
                    "inexact-export",
                    f"{kind}:{index}",
                    f"CadExportTolerances.{kind + '_bounds'} must have certified gap evidence; got absent entity={entity}",
                )
            row.append(_written_upper_bound(bounds[entity]))
        rows.append(tuple(row))
    return CadExportTolerances(
        vertex_bounds=rows[0],
        edge_bounds=rows[1],
        face_bounds=rows[2],
        uncertainty=max((value for row in rows for value in row), default=0.0),
    )


@checked
def certified_cad_export_tolerances(
    model: BRepModel,
    /,
) -> CadExportTolerances:
    """Return entity tolerances from the model's certified source gap bounds.

    Bounds use model length units and are rounded outward for decimal output.
    Missing or unbounded entity evidence refuses export.
    """
    return _certified_cad_export_tolerances(model, model.patches)


# --------------------------------------------------------------------- units


_SI_PREFIX_EXPONENTS: dict[str, int] = {
    "EXA": 18,
    "PETA": 15,
    "TERA": 12,
    "GIGA": 9,
    "MEGA": 6,
    "KILO": 3,
    "HECTO": 2,
    "DECA": 1,
    "DECI": -1,
    "CENTI": -2,
    "MILLI": -3,
    "MICRO": -6,
    "NANO": -9,
    "PICO": -12,
    "FEMTO": -15,
    "ATTO": -18,
}


def si_prefix_scale(prefix: str | None, /) -> Fraction:
    """Return the rational SI prefix factor (``None`` is the unprefixed unit)."""
    if prefix is None:
        return Fraction(1)
    if prefix not in _SI_PREFIX_EXPONENTS:
        raise ValueError(
            f"SIPrefix.prefix must be a supported prefix name; got {prefix!r}"
        )
    return Fraction(10) ** _SI_PREFIX_EXPONENTS[prefix]


def si_prefix_for(scale: Fraction, /) -> str | None:
    """The SI prefix whose factor is exactly ``scale`` (``""`` for one), else None."""
    if scale == 1:
        return ""
    for prefix, exponent in _SI_PREFIX_EXPONENTS.items():
        if Fraction(10) ** exponent == scale:
            return prefix
    return None


@checked
def length_factor(
    source_meters: Fraction, contract: SpatialCoordinateContract, /
) -> float:
    """Return the source-to-target length factor from a rational unit ratio."""
    return float(source_meters / contract.length_unit.scale_to_reference)


# -------------------------------------------------------------- 2D carriers


@checked
def transform_pcurve(
    curve: AbstractCurve,
    axis_factors: tuple[float, float],
    parameter_factor: float,
    offset: tuple[float, float] = (0.0, 0.0),
    /,
) -> AbstractCurve:
    """Map a p-curve by ``uv -> axis_factors * uv + offset`` with ``t -> t * parameter_factor``.

    Lines scale origin/direction and B-splines scale control points and knots.
    Conics retain their angle parameter only when the mapped Fourier axes remain
    orthogonal; a map lacking a parameter-preserving native carrier is refused.
    """
    factors = np.asarray(axis_factors, dtype=np.float64)
    shift = np.asarray(offset, dtype=np.float64)
    identity = bool(np.all(factors == 1.0) and np.all(shift == 0.0))
    if identity and parameter_factor == 1.0:
        return curve
    match curve:
        case LineCurve():
            return LineCurve(
                np.asarray(curve.origin) * factors + shift,
                np.asarray(curve.direction) * factors / parameter_factor,
            )
        case BSplineCurve():
            return BSplineCurve(
                np.asarray(curve.control_points) * factors + shift,
                curve.weights,
                np.asarray(curve.knots) * parameter_factor,
                curve.degree,
            )
        case CircleCurve() if factors[0] == factors[1] and parameter_factor == 1.0:
            return CircleCurve(
                np.asarray(curve.center) * factors + shift,
                curve.first_axis,
                curve.second_axis,
                float(curve.radius) * float(factors[0]),
            )
        case EllipseCurve() if factors[0] == factors[1] and parameter_factor == 1.0:
            return EllipseCurve(
                np.asarray(curve.center) * factors + shift,
                curve.first_axis,
                curve.second_axis,
                float(curve.first_radius) * float(factors[0]),
                float(curve.second_radius) * float(factors[0]),
            )
        case ParabolaCurve() if (
            factors[0] == factors[1] == parameter_factor and parameter_factor > 0.0
        ):
            return ParabolaCurve(
                np.asarray(curve.vertex) * factors + shift,
                curve.first_axis,
                curve.second_axis,
                float(curve.focal_length) * parameter_factor,
            )
        case HyperbolaCurve() if (
            factors[0] == factors[1] and factors[0] > 0.0 and parameter_factor == 1.0
        ):
            return HyperbolaCurve(
                np.asarray(curve.center) * factors + shift,
                curve.first_axis,
                curve.second_axis,
                float(curve.first_radius) * factors[0],
                float(curve.second_radius) * factors[0],
            )
        case OffsetCurve() if factors[0] == factors[1] and factors[0] > 0.0:
            return OffsetCurve(
                transform_pcurve(curve.base, axis_factors, parameter_factor, offset),
                curve.distance_exact * Fraction.from_float(float(factors[0])),
                curve.direction,
            )
        case CircleCurve() | EllipseCurve() if parameter_factor == 1.0:
            first_radius = float(
                curve.radius if isinstance(curve, CircleCurve) else curve.first_radius
            )
            second_radius = float(
                curve.radius if isinstance(curve, CircleCurve) else curve.second_radius
            )
            first = factors * np.asarray(curve.first_axis) * first_radius
            second = factors * np.asarray(curve.second_axis) * second_radius
            radii = (float(np.linalg.norm(first)), float(np.linalg.norm(second)))
            if min(radii) > 0.0 and abs(first @ second) <= 1.0e-12 * radii[0] * radii[1]:
                return EllipseCurve(
                    np.asarray(curve.center) * factors + shift,
                    first / radii[0],
                    second / radii[1],
                    radii[0],
                    radii[1],
                )
            raise ValueError(
                f"transform_pcurve.axis_factors must preserve orthogonal conic Fourier axes; got factors={axis_factors!r}, parameter_factor={parameter_factor!r}"
            )
        case _:
            raise ValueError(
                f"transform_pcurve.parameter_factor must have a parameter-preserving native carrier; got {type(curve).__name__}"
            )


_EXPLICIT_CARRIERS = (
    LineCurve,
    CircleCurve,
    EllipseCurve,
    ParabolaCurve,
    HyperbolaCurve,
    OffsetCurve,
    BSplineCurve,
)


@checked
def explicit_carrier(curve: AbstractCurve, /) -> AbstractCurve:
    """Line, conic or B-spline carrier with the same parameterization as ``curve``.

    Format writers emit only these families. A periodic p-curve sheet is its
    source translated by the binary representative of its symbolic
    native-period multiple; a diagonal affine UV transport maps the source
    coefficientwise; analytic source isolines become the line or circle they
    parameterize. Nothing is sampled or refitted. Curves without such a rule
    are returned unchanged for the writer to refuse.
    """
    match curve:
        case ExactLineCurve() as line:
            direction_exact = tuple(
                last - first
                for first, last in zip(line.first_exact, line.last_exact, strict=True)
            )
            try:
                return LineCurve(
                    np.asarray(line.first_exact, dtype=np.float64),
                    np.asarray(direction_exact, dtype=np.float64),
                    origin_exact=line.first_exact,
                    direction_exact=direction_exact,
                )
            except (OverflowError, ValueError):
                return curve
        case PeriodicPCurve() | AffinePCurve():
            inner = (
                curve.source_curve if isinstance(curve, PeriodicPCurve) else curve.curve
            )
            source = (
                explicit_carrier(inner) if isinstance(inner, AbstractCurve) else inner
            )
            if not isinstance(source, _EXPLICIT_CARRIERS):
                return curve
            if isinstance(curve, PeriodicPCurve):
                factors = (1.0, 1.0)
            else:
                (a, b), (c, d) = curve.matrix
                if b or c:
                    return curve
                factors = (float(a), float(d))
            offset = np.asarray(curve.offset_value)
            try:
                return transform_pcurve(
                    source, factors, 1.0, (float(offset[0]), float(offset[1]))
                )
            except ValueError:
                return curve
        case SurfaceIsoparametricCurve(surface=CylinderPatch() as cylinder):
            first = np.asarray(cylinder.first_axis)
            second = np.asarray(cylinder.second_axis)
            axis = np.asarray(cylinder.axis)
            origin = np.asarray(cylinder.origin)
            if curve.fixed_axis == 0:
                radial = (
                    np.cos(curve.fixed_value) * first + np.sin(curve.fixed_value) * second
                )
                return LineCurve(origin + float(cylinder.radius) * radial, axis)
            return CircleCurve(
                origin + curve.fixed_value * axis,
                first,
                second,
                cylinder.radius,
            )
        case SurfaceIsoparametricCurve(surface=SpherePatch() as sphere) if (
            curve.fixed_axis == 0
        ):
            first, second = np.asarray(sphere.first_axis), np.asarray(sphere.second_axis)
            meridian = (
                np.cos(curve.fixed_value) * first + np.sin(curve.fixed_value) * second
            )
            return CircleCurve(
                np.asarray(sphere.center),
                meridian,
                np.asarray(sphere.axis),
                sphere.radius,
            )
        case SurfaceIsoparametricCurve(surface=TorusPatch() as torus):
            first, second = np.asarray(torus.first_axis), np.asarray(torus.second_axis)
            axis, center = np.asarray(torus.axis), np.asarray(torus.center)
            major, minor = float(torus.major_radius), float(torus.minor_radius)
            angle = curve.fixed_value
            if curve.fixed_axis == 0:
                radial = np.cos(angle) * first + np.sin(angle) * second
                return CircleCurve(center + major * radial, radial, axis, minor)
            return CircleCurve(
                center + minor * np.sin(angle) * axis,
                first,
                second,
                major + minor * np.cos(angle),
            )
        case _:
            return curve


# ------------------------------------------------------------ curve inversion


@checked
def curve_parameter(curve: AbstractCurve, point: np.ndarray, /) -> tuple[float, float]:
    """Return a nearest carrier parameter and its physical endpoint distance.

    Lines, exact segments and conics have direct inverses. Bounded splines and
    offsets use a source-scaled proposal and refinement; `edge_range` admits an
    endpoint only within the declared physical tolerance.
    """
    target = np.asarray(point, dtype=np.float64)
    match curve:
        case OffsetCurve():
            base = curve.base
            while isinstance(base, OffsetCurve):
                base = base.base
            if isinstance(base, (LineCurve, ExactLineCurve)):
                origin = np.asarray(
                    curve.evaluate(jnp.asarray(0.0, dtype=jnp.float64)),
                    dtype=np.float64,
                )
                direction = np.asarray(
                    jax.jacfwd(curve.evaluate)(jnp.asarray(0.0, dtype=jnp.float64)),
                    dtype=np.float64,
                )
                parameter = float((target - origin) @ direction / (direction @ direction))
                if isinstance(base, ExactLineCurve):
                    parameter = min(max(parameter, 0.0), 1.0)
            else:
                parameter = _sampled_curve_parameter(curve, target)
        case ExactLineCurve():
            origin = np.asarray(curve.first)
            direction = np.asarray(curve.last) - origin
            parameter = float((target - origin) @ direction / (direction @ direction))
            parameter = min(max(parameter, 0.0), 1.0)
        case LineCurve():
            origin = np.asarray(curve.origin)
            direction = np.asarray(curve.direction)
            parameter = float((target - origin) @ direction / (direction @ direction))
        case CircleCurve() | EllipseCurve():
            relative = target - np.asarray(curve.center)
            first = float(relative @ np.asarray(curve.first_axis))
            second = float(relative @ np.asarray(curve.second_axis))
            if isinstance(curve, EllipseCurve):
                first /= float(curve.first_radius)
                second /= float(curve.second_radius)
            parameter = atan2(second, first)
        case BSplineCurve():
            parameter = _sampled_curve_parameter(curve, target)
        case _:
            raise TypeError(
                f"curve_parameter.curve must support parameter inversion; got {type(curve).__name__}"
            )
    distance = float(
        np.linalg.norm(
            np.asarray(curve.evaluate(jnp.asarray(parameter, dtype=jnp.float64))) - target
        )
    )
    return parameter, distance


def _sampled_curve_parameter(curve: AbstractCurve, target: np.ndarray, /) -> float:
    domain = curve.parameter_domain
    if domain is None:
        period = curve.period
        if period is None or not isfinite(period) or period <= 0.0:
            raise ValueError(
                f"{type(curve).__name__}.parameter_domain must be finite or periodic for sampled inversion."
            )
        domain = (0.0, period)
    if not all(isfinite(value) for value in domain) or not domain[0] < domain[1]:
        raise ValueError(
            f"{type(curve).__name__}.parameter_domain must be finite and increasing."
        )
    base = curve
    while isinstance(base, OffsetCurve):
        base = base.base
    sample_count = (
        8 * base.control_points.shape[0] + 1 if isinstance(base, BSplineCurve) else 65
    )
    samples = np.linspace(domain[0], domain[1], sample_count)
    values = np.asarray(curve.evaluate(jnp.asarray(samples)), dtype=np.float64)
    if values.shape != (sample_count, curve.ambient_dimension) or not np.all(
        np.isfinite(values)
    ):
        raise ValueError(
            f"{type(curve).__name__}.evaluate must return finite source points during inversion."
        )
    parameter = float(samples[int(np.argmin(np.linalg.norm(values - target, axis=1)))])
    derivative = jax.jacfwd(curve.evaluate)
    second = jax.jacfwd(derivative)

    def value_jets(row: jax.Array) -> jax.Array:
        parameter_ = row[0]
        return jnp.stack(
            (curve.evaluate(parameter_), derivative(parameter_), second(parameter_)),
            axis=-1,
        )

    prepared = prepare_interval_function(value_jets, 1, batch_capacity=1)
    while True:
        jets = prepared.evaluate_host_points(np.asarray([[parameter]], dtype=np.float64))[
            0
        ]
        value, first_, second_ = jets[:, 0], jets[:, 1], jets[:, 2]
        residual = value - target
        distance = float(np.linalg.norm(residual))
        gradient = float(residual @ first_)
        curvature = float(first_ @ first_ + residual @ second_)
        if not np.isfinite(curvature) or curvature <= 0.0:
            return parameter
        step = gradient / curvature
        candidate = min(max(parameter - step, domain[0]), domain[1])
        if not np.isfinite(candidate) or candidate == parameter:
            return parameter
        candidate_value = np.asarray(
            curve.evaluate(jnp.asarray(candidate, dtype=jnp.float64)),
            dtype=np.float64,
        )
        candidate_distance = float(np.linalg.norm(candidate_value - target))
        if not np.isfinite(candidate_distance) or candidate_distance >= distance:
            return parameter
        parameter = candidate


def _authored_turn_end(first: float, last: float, /) -> float:
    """Return the binary64 end of one authored turn written as a rounded decimal.

    A closed edge on a ``2 pi``-periodic carrier (or a pole edge spanning the
    chart period) covers a whole number of turns; exchange formats print that
    end with finitely many decimal digits. A range within decimal rounding of
    one turn is the turn itself; every other range is returned unchanged.
    """
    turn = float(first) + _TWO_PI
    last = float(last)
    if last != turn and abs(last - turn) <= _AUTHORED_DECIMAL_RELATIVE * max(
        1.0, abs(turn)
    ):
        return turn
    return last


@checked
def edge_range(
    curve: AbstractCurve,
    start: np.ndarray,
    end: np.ndarray,
    closed: bool,
    tolerance: float,
    /,
) -> tuple[float, float]:
    """Increasing carrier range from ``start`` to ``end`` within physical tolerance."""
    if not isfinite(tolerance) or tolerance < 0.0:
        raise ValueError(
            f"edge_range.tolerance must be finite and nonnegative; got {tolerance!r}"
        )
    period = curve.period
    if period is not None:
        period = float(period)
        if not isfinite(period) or period <= 0.0:
            raise ValueError(
                f"edge_range.curve.period must be finite and positive; got {period!r}"
            )
    if closed:
        if period is not None:
            first, gap = curve_parameter(curve, start)
            if not isfinite(first) or not isfinite(gap) or gap > tolerance:
                raise ValueError(
                    f"edge_range.start must lie on the periodic carrier within tolerance={tolerance}; got parameter={first!r}, deviation={gap!r}"
                )
            try:
                last = float(Fraction.from_float(first) + Fraction.from_float(period))
            except OverflowError as error:
                raise ValueError(
                    "edge_range.periodic range exceeds finite parameters."
                ) from error
            if not isfinite(last) or not last > first:
                raise ValueError("edge_range.periodic range is not representable.")
            return first, last
        domain = curve.parameter_domain
        if domain is None:
            raise ValueError(
                "edge_range.curve.parameter_domain must be finite for a nonperiodic closed edge; got None"
            )
        # A bounded closed edge uses its whole domain. Endpoint admission
        # proves that range directly; an inverse at the shared vertex cannot
        # select another range and introduces an unnecessary global root solve.
        ends = np.asarray(curve.evaluate(jnp.asarray(domain)), dtype=np.float64)
        gaps = np.linalg.norm(ends - start, axis=1)
        if not np.all(np.isfinite(gaps)) or np.max(gaps) > tolerance:
            raise ValueError(
                f"edge_range.start must lie on both bounded carrier endpoints within tolerance={tolerance}; got distances={gaps!r}"
            )
        return domain
    first, first_gap = curve_parameter(curve, start)
    last, last_gap = curve_parameter(curve, end)
    if (
        not (isfinite(first) and isfinite(last))
        or not (isfinite(first_gap) and isfinite(last_gap))
        or max(first_gap, last_gap) > tolerance
    ):
        raise ValueError(
            f"edge_range endpoints must lie within carrier tolerance={tolerance}; got parameters=({first!r},{last!r}), deviations=({first_gap!r},{last_gap!r})"
        )
    if period is not None:
        difference = Fraction.from_float(last) - Fraction.from_float(first)
        exact_period = Fraction.from_float(period)
        if difference <= 0:
            _, remainder = divmod(-difference, exact_period)
            width = exact_period - remainder if remainder else exact_period
        elif difference > exact_period:
            _, remainder = divmod(difference, exact_period)
            width = exact_period if remainder == 0 else remainder
        else:
            width = difference
        try:
            last = float(Fraction.from_float(first) + width)
        except OverflowError as error:
            raise ValueError(
                "edge_range.periodic range exceeds finite parameters."
            ) from error
        if (
            not isfinite(last)
            or not last > first
            or Fraction.from_float(last) - Fraction.from_float(first) > exact_period
        ):
            raise ValueError(
                "edge_range.periodic range cannot be represented by finite binary parameters."
            )
    if not last > first:
        raise ValueError(
            f"edge_range.end must increase from start to end; got first={first}, last={last}"
        )
    return first, last


def _window_root_endpoint(
    root: TrimIntersectionRoot, operand: IntersectionSide, window: CurveTrimSegment, /
) -> TrimRootEndpoint:
    """Bind a window-local trim root to its coedge's edge parameter exactly.

    The root operand is the incidence window's normalized chart; the endpoint
    carries the window's exact affine map back to the source parameter.
    """
    scale = window.last - window.first
    if window.reversed or Fraction(scale) != Fraction(window.last) - Fraction(
        window.first
    ):
        raise ValueError(
            f"CadStage incidence window must have an exact forward affine chart; got first={window.first!r}, last={window.last!r}, reversed={window.reversed!r}"
        )
    return TrimRootEndpoint(
        root,
        operand,
        affine_parameter_offset=window.first,
        affine_parameter_scale=scale,
    )


# -------------------------------------------------------- surface inversion


def _angle(value: np.ndarray, /) -> np.ndarray:
    return np.arctan2(value[..., 1], value[..., 0])


@checked
def natural_box(patch: AbstractSurfacePatch, /) -> np.ndarray:
    """Finite parameter domain limits of a carrier (infinite where unbounded)."""
    match patch:
        case OffsetSurface():
            return natural_box(patch.base)
        case PlacedSurface():
            return natural_box(patch.definition)
        case _:
            pass
    box = np.asarray(((-np.inf, -np.inf), (np.inf, np.inf)), dtype=np.float64)
    match patch:
        case SpherePatch():
            box[:, 1] = (-0.5 * pi, 0.5 * pi)
        case BSplineSurfacePatch():
            for axis, (knots, degree) in enumerate(
                ((patch.u_knots, patch.u_degree), (patch.v_knots, patch.v_degree))
            ):
                host = np.asarray(knots)
                box[:, axis] = (host[degree], host[-degree - 1])
        case RuledSurface():
            box[:, 1] = (0.0, 1.0)
        case _:
            pass
    for axis in (0, 1):
        domain = _sweep_domain(patch, axis)
        if domain is not None:
            box[:, axis] = domain
    return box


def _aligned_period_axis(
    patch: AbstractSurfacePatch, pcurve: AbstractCurve, /
) -> Literal[0, 1] | None:
    """Declared 2pi axis of ``patch`` along which an affine p-curve source moves."""
    source = pcurve.source_curve if isinstance(pcurve, PeriodicPCurve) else pcurve
    if not isinstance(source, AbstractCurve):
        return None
    affine = _affine_curve_coefficients(explicit_carrier(source), _ZERO_UV)
    if affine is None:
        return None
    direction = affine[1]
    for axis, period in enumerate(patch.native_period_symbols()):
        unit = tuple(Fraction(int(component == axis)) for component in range(2))
        if period == "two_pi" and (
            direction == unit or direction == tuple(-value for value in unit)
        ):
            return 0 if axis == 0 else 1
    return None


def _coedge_period_binding(
    face_patch: AbstractSurfacePatch,
    patch: AbstractSurfacePatch | None,
    axis: int | None,
    pcurve: AbstractCurve,
    /,
) -> tuple[AbstractSurfacePatch | None, int | None] | None:
    """Bind a native-turn coedge endpoint to the period of its own face.

    The edge owner's period when the face carries that patch, else a declared
    period of the face patch aligned with the use, else the p-curve's own native
    turn. ``None`` means the use has no symbolic support: a curve-owned full turn
    does not fabricate it for an externally rounded p-curve parameterization.
    """
    if patch is not None and carrier_payload(face_patch) == carrier_payload(patch):
        return patch, axis
    aligned = _aligned_period_axis(face_patch, pcurve)
    if aligned is not None:
        return face_patch, aligned
    if pcurve.native_period_symbol() == "two_pi":
        return None, None
    return None


def _sweep_domain(
    patch: AbstractSurfacePatch, axis: int, /
) -> tuple[float, float] | None:
    match patch:
        case OffsetSurface():
            return _sweep_domain(patch.base, axis)
        case PlacedSurface():
            return _sweep_domain(patch.definition, axis)
        case ExtrusionSurface() if axis == 0:
            return patch.curve.parameter_domain
        case RevolutionSurface() if axis == 1:
            return patch.curve.parameter_domain
        case RuledSurface() if axis == 0:
            first, second = patch.first.parameter_domain, patch.second.parameter_domain
            if first is None or second is None:
                return first or second
            return max(first[0], second[0]), min(first[1], second[1])
        case _:
            return None


def _representative_pcurve_box(
    patch: AbstractSurfacePatch, pcurve: AbstractCurve, first: float, last: float, /
) -> np.ndarray:
    """UV box of a p-curve in the native representative chart.

    A whole-turn sheet p-curve denotes its source shifted by symbolic native
    turns; its representative is the source translated by the binary period,
    the convention under which a full-turn source region is exactly one binary
    period wide.
    """
    if isinstance(pcurve, PeriodicPCurve) and pcurve.patch is patch:
        symbols = patch.native_period_symbols()
        periods = patch.periods
        offset = np.zeros(2, dtype=np.float64)
        for axis, shift in enumerate(pcurve.period_shifts):
            period = periods[axis]
            if not shift:
                continue
            if symbols[axis] != "two_pi" or period is None:
                return np.asarray(pcurve.bounding_box(first, last), dtype=np.float64)
            offset[axis] = shift * float(period)
        source = np.asarray(
            pcurve.source_curve.bounding_box(first, last), dtype=np.float64
        )
        return source + offset
    return np.asarray(pcurve.bounding_box(first, last), dtype=np.float64)


# ------------------------------------------------------------ p-curve recovery


def _fit_basis(knots: np.ndarray, count: int, parameters: np.ndarray, /) -> np.ndarray:
    # The rational evaluator with identity controls yields the basis.
    basis = BSplineCurve(np.eye(count), np.ones(count), knots, 3)
    return np.asarray(
        basis.evaluate(jnp.asarray(parameters, dtype=jnp.float64)),
        dtype=np.float64,
    )


def _segment_knots(breakpoints: np.ndarray, spans: int, /) -> np.ndarray:
    """Cubic knots: ``spans`` uniform spans per segment, C0 at interior breakpoints."""
    knots = [np.full(4, breakpoints[0])]
    for index, (lower, upper) in enumerate(
        zip(breakpoints[:-1], breakpoints[1:], strict=True)
    ):
        knots.append(np.linspace(lower, upper, spans + 1)[1:-1])
        last = index == breakpoints.shape[0] - 2
        knots.append(np.full(4 if last else 3, upper))
    return np.concatenate(knots)


@checked
def recover_pcurve(
    patch: AbstractSurfacePatch,
    curve: AbstractCurve,
    policy: CadCurveFitPolicy | None,
    /,
) -> AbstractCurve:
    """Recover only a p-curve whose complete source map is algebraically known.

    A circle in a plane exactly parallel to a planar face maps to its exact
    orthogonal chart projection; its normal offset is not hidden, it is the
    face gap that publication certifies against the import tolerance.
    """
    candidate = native_edge_pcurve(curve, patch)
    if candidate is None and isinstance(curve, CircleCurve):
        candidate = _parallel_plane_circle_pcurve(patch, curve)
    if (
        candidate is None
        and isinstance(curve, CircleCurve)
        and isinstance(patch, TorusPatch)
    ):
        # Rounded torus angles: publication certifies the face gap.
        candidate = torus_circle_pcurve(curve, patch)
    if candidate is not None:
        return candidate
    if policy is not None:
        raise ValueError(
            f"CadCurveFitPolicy.tolerance={policy.tolerance!r} requests bounded p-curve fitting, but no continuous source/topology certificate exists for surface={type(patch).__name__}, curve={type(curve).__name__}."
        )
    raise ValueError(
        f"No exact p-curve source map exists for surface={type(patch).__name__}, curve={type(curve).__name__}."
    )


def _parallel_plane_circle_pcurve(
    patch: AbstractSurfacePatch, curve: CircleCurve, /
) -> CircleCurve | None:
    """Chart circle of a 3D circle whose radial harmonics lie exactly in a plane chart."""
    frame = _affine_surface_frame(patch)
    if frame is None or curve.ambient_dimension != 3:
        return None
    origin, first, second = frame
    center, cosine, sine = _harmonic_coefficients(curve)
    chart = (
        _plane_parameter_vector(
            frame, tuple(value - base for value, base in zip(center, origin, strict=True))
        ),
        _plane_parameter_vector(frame, cosine),
        _plane_parameter_vector(frame, sine),
    )
    chart_center, chart_cosine, chart_sine = chart
    if chart_center is None or chart_cosine is None or chart_sine is None:
        return None
    for vector, target in ((chart_cosine, cosine), (chart_sine, sine)):
        image = tuple(first[i] * vector[0] + second[i] * vector[1] for i in range(3))
        if image != target:
            return None
    radius = curve.radius_exact

    def dot(
        left: tuple[ExactScalar, ...], right: tuple[ExactScalar, ...], /
    ) -> ExactScalar:
        return sum((a * b for a, b in zip(left, right, strict=True)), Fraction(0))

    squared = radius * radius
    if (
        dot(chart_cosine, chart_cosine) != squared
        or dot(chart_sine, chart_sine) != squared
        or dot(chart_cosine, chart_sine) != 0
    ):
        return None
    inverse = radius.reciprocal() if isinstance(radius, RadicalElement) else 1 / radius
    first_axis = tuple(value * inverse for value in chart_cosine)
    second_axis = tuple(value * inverse for value in chart_sine)
    return CircleCurve(
        tuple(float(value) for value in chart_center),
        tuple(float(value) for value in first_axis),
        tuple(float(value) for value in second_axis),
        float(radius),
        center_exact=chart_center,
        first_axis_exact=first_axis,
        second_axis_exact=second_axis,
        radius_exact=radius,
    )


def _validate_curve_parameter_range(
    curve: AbstractCurve,
    first: float,
    last: float,
    label: str,
    /,
) -> None:
    """Validate an authored interval against the carrier's source domain."""
    if not np.isfinite(first) or not np.isfinite(last) or not first < last:
        raise ValueError(
            f"{label}.parameter_range must be finite and increasing; got {(first, last)!r}."
        )
    domain = curve.parameter_domain
    if domain is not None and (first < domain[0] or last > domain[1]):
        raise ValueError(
            f"{label}.parameter_range must lie in the source parameter_domain; "
            f"got range={(first, last)!r}, domain={domain!r}."
        )


# ------------------------------------------------------------------- staging


@dataclass(slots=True)
class StagedCoedge:
    """One oriented edge use with an optional file p-curve (on the edge parameter)."""

    edge: int
    sense: int
    pcurve: AbstractCurve | None
    label: str


@dataclass(slots=True)
class StagedFace:
    patch: AbstractSurfacePatch
    loops: list[list[StagedCoedge]]
    orientation: int
    tag: str
    label: str
    outer_known: bool
    loops_by_containment: bool = False


class _IncidencePlan(StrictModule):
    junctions: dict[int, list[tuple[int, int, int, int, int]]]
    needed: tuple[int, ...]


class _IncidenceVertex(StrictModule):
    definition: BRepVertexRoot
    bindings: tuple[tuple[int, int, TrimRootEndpoint], ...]


@dataclass(slots=True)
class CadStage:
    """Mutable host staging of one definition-preserving decoded B-Rep (never published).

    Readers add vertices, carriers, edges, faces, shells, solids and exact
    assembly definitions. `CadStage.publish` recovers only exact source
    p-curves, seam representatives, pole edges and parameter boxes before
    publishing one validated `BRepModel`.
    Winding conic bands use source-proved chart cuts: source-ring subdivisions
    propagate through every incident face, authored vertices remain present,
    and each derived chart seam cancels between its two opposite uses.
    Toleranced authored incidences are not promoted to zero-gap source proofs.
    """

    scale: float
    fit_policy: CadCurveFitPolicy | None
    relative_geometric_tolerance: float = 1.0e-8
    vertices: list[np.ndarray] = field(default_factory=list)
    curves: list[AbstractCurve] = field(default_factory=list)
    edge_curves: list[int] = field(default_factory=list)
    edge_ranges: list[tuple[float, float]] = field(default_factory=list)
    edge_vertices: list[tuple[int, int]] = field(default_factory=list)
    faces: list[StagedFace] = field(default_factory=list)
    shells: list[tuple[list[int], bool]] = field(default_factory=list)
    shell_orientations: dict[int, list[int]] = field(default_factory=dict)
    solids: list[list[int]] = field(default_factory=list)
    assembly_graph: BRepAssemblyGraph | None = None
    provenance: list[tuple[str, str]] = field(default_factory=list)
    pcurves_honored: int = 0
    pcurves_exact: int = 0
    degenerate_edges: int = 0
    natural_boundaries: int = 0
    native_phases: dict[
        int,
        tuple[AbstractSurfacePatch | None, Literal[0, 1] | None, Fraction],
    ] = field(default_factory=dict)
    split_endpoint_phases: dict[int, tuple[Fraction | None, Fraction | None]] = field(
        default_factory=dict
    )
    split_period_surfaces: dict[int, AbstractSurfacePatch] = field(default_factory=dict)
    chart_closure_repairs: list[AdapterLoss] = field(default_factory=list)

    @property
    def tolerance(self) -> float:
        return 1.0e-9 * self.scale

    def vertex(self, point: np.ndarray, label: str, /) -> int:
        self.vertices.append(np.asarray(point, dtype=np.float64).reshape(3))
        self.provenance.append((f"vertex:{len(self.vertices) - 1}", label))
        return len(self.vertices) - 1

    def edge(
        self,
        curve: AbstractCurve,
        start: int,
        end: int,
        label: str,
        /,
        *,
        parameter_range: tuple[float, float] | None = None,
    ) -> int:
        """Stage an edge from ``start`` to ``end`` along the increasing carrier.

        Without ``parameter_range`` the range comes from vertex inversion; a
        given range (from formats that store it) is validated against the
        vertices and the carrier period instead.
        """
        if parameter_range is None:
            first, last = edge_range(
                curve,
                self.vertices[start],
                self.vertices[end],
                start == end,
                self.tolerance,
            )
        else:
            first, last = (float(parameter_range[0]), float(parameter_range[1]))
            if start == end and isinstance(curve, (CircleCurve, EllipseCurve)):
                last = _authored_turn_end(first, last)
            _validate_curve_parameter_range(curve, first, last, label)
            ends = np.asarray(
                curve.evaluate(jnp.asarray((first, last), dtype=jnp.float64))
            )
            gaps = (
                np.linalg.norm(ends[0] - self.vertices[start]),
                np.linalg.norm(ends[1] - self.vertices[end]),
            )
            if max(gaps) > self.tolerance:
                raise ValueError(
                    f"CadStage.edge.parameter_range must end at its vertices within tolerance={self.tolerance}; got deviation={max(gaps):.3e}"
                )
        self.curves.append(curve)
        self.edge_curves.append(len(self.curves) - 1)
        self.edge_ranges.append((first, last))
        self.edge_vertices.append((start, end))
        edge = len(self.edge_curves) - 1
        self.provenance.append((f"edge:{edge}", label))
        if (
            start == end
            and isinstance(curve, (CircleCurve, EllipseCurve))
            and last == first + _TWO_PI
        ):
            # A closed conic entity over one whole native parameter interval
            # authors one mathematical turn; binary64 tau is only its carrier.
            self.native_phase(edge, None, None, rational=Fraction(first))
        return edge

    def degenerate(self, vertex: int, first: float, last: float, /) -> int:
        self.edge_curves.append(-1)
        self.edge_ranges.append((float(first), _authored_turn_end(first, last)))
        self.edge_vertices.append((vertex, vertex))
        self.degenerate_edges += 1
        return len(self.edge_curves) - 1

    def native_phase(
        self,
        edge: int,
        patch: AbstractSurfacePatch | None,
        axis: Literal[0, 1] | None,
        /,
        *,
        rational: Fraction = Fraction(),
    ) -> None:
        """Author one full native turn on an edge and every coedge use.

        The published endpoint roots retain the symbolic expression ``rational + turns * 2 pi``.
        A surface and axis prove a chart period when supplied; otherwise the
        curve family defines the period.
        """
        if self.edge_ranges[edge] != (float(rational), float(rational) + _TWO_PI):
            raise ValueError(
                f"CadStage.edge_ranges must span one authored turn; got {self.edge_ranges[edge]!r}"
            )
        self.native_phases[edge] = (patch, axis, rational)

    def face(
        self,
        patch: AbstractSurfacePatch,
        loops: list[list[StagedCoedge]],
        orientation: int,
        tag: str,
        label: str,
        /,
        *,
        outer_known: bool,
        loops_by_containment: bool = False,
    ) -> int:
        if loops_by_containment and not outer_known:
            raise ValueError(
                f"StagedFace.outer_known must be true for containment-bounded loops; got {outer_known!r}"
            )
        self.faces.append(
            StagedFace(
                patch, loops, orientation, tag, label, outer_known, loops_by_containment
            )
        )
        self.provenance.append((f"face:{len(self.faces) - 1}", label))
        return len(self.faces) - 1

    # -------------------------------------------------------- natural bounds

    @staticmethod
    def _sphere_chart_radius(patch: AbstractSurfacePatch, /) -> Fraction | None:
        source = patch
        offsets: list[OffsetSurface] = []
        while isinstance(source, (OffsetSurface, PlacedSurface)):
            if isinstance(source, OffsetSurface):
                offsets.append(source)
                source = source.base
            else:
                source = source.definition
        if not isinstance(source, SpherePatch):
            return None
        radius_value = float(source.radius)
        if not isfinite(radius_value):
            return None
        radius = Fraction.from_float(radius_value)
        for offset in reversed(offsets):
            if offset.frame_sign == 0:
                return None
            radius += offset.frame_sign * offset.distance_exact
        return radius

    def _sphere_pole_points(
        self,
        pcurves: list[AbstractCurve],
        coedge_edges: list[int],
        face_loops: list[tuple[tuple[int, ...], ...]],
        /,
    ) -> dict[int, tuple[Fraction, Fraction, Fraction]]:
        """Exact poles of authored vertices collapsing sphere-chart pole coedges.

        A pole coedge on a sphere chart's declared latitude limit collapses to
        ``center ± radius * axis`` exactly; the file prints its vertex through
        rounded ``cos(pi / 2)``. A vertex within decimal rounding of that pole is
        the pole; any other vertex keeps its printed coordinates.
        """
        poles: dict[int, tuple[Fraction, Fraction, Fraction]] = {}
        for face, loops in enumerate(face_loops):
            patch = self.faces[face].patch
            if not isinstance(patch, SpherePatch):
                continue
            for loop in loops:
                for coedge in loop:
                    edge = coedge_edges[coedge]
                    if self.edge_curves[edge] >= 0:
                        continue
                    affine = _affine_curve_coefficients(pcurves[coedge], _ZERO_UV)
                    if affine is None or affine[1][1] != 0:
                        continue
                    pole = exact_sphere_pole(patch, affine[0][1])
                    if pole is None:
                        continue
                    vertex = self.edge_vertices[edge][0]
                    rounded = np.asarray(
                        tuple(float(value) for value in pole), dtype=np.float64
                    )
                    authored = np.asarray(self.vertices[vertex], dtype=np.float64)
                    scale = max(1.0, float(np.max(np.abs(rounded))))
                    if (
                        float(np.max(np.abs(authored - rounded)))
                        <= _AUTHORED_DECIMAL_RELATIVE * scale
                    ):
                        poles[vertex] = pole
        return poles

    def _snap_sphere_meridian_ranges(self) -> None:
        """Return authored meridian ends at a sphere chart's pole latitude to it.

        Exchange formats print a meridian's edge range in decimal, so its p-curve
        end misses the declared latitude limit ``±0.5 * pi`` by decimal rounding.
        An end within that rounding becomes the representable edge parameter
        whose exact p-curve image is the limit; other ends are unchanged.
        """
        limit = Fraction.from_float(0.5 * pi)
        for face in self.faces:
            if not isinstance(face.patch, SpherePatch):
                continue
            for loop in face.loops:
                for coedge in loop:
                    if coedge.pcurve is None or self.edge_curves[coedge.edge] < 0:
                        continue
                    pcurve = coedge.pcurve
                    if isinstance(pcurve, PeriodicPCurve):
                        if pcurve.period_shifts[1] != 0:
                            continue
                        pcurve = pcurve.source_curve
                    affine = _affine_curve_coefficients(
                        explicit_carrier(pcurve), _ZERO_UV
                    )
                    if affine is None:
                        continue
                    offset, direction = affine
                    if (
                        direction[0] != 0
                        or direction[1] == 0
                        or not isinstance(offset[1], Fraction)
                        or not isinstance(direction[1], Fraction)
                    ):
                        continue
                    ends = list(self.edge_ranges[coedge.edge])
                    for index, parameter in enumerate(ends):
                        latitude = offset[1] + direction[1] * Fraction.from_float(
                            float(parameter)
                        )
                        for target in (limit, -limit):
                            snapped = (target - offset[1]) / direction[1]
                            if (
                                latitude != target
                                and abs(latitude - target)
                                <= _AUTHORED_DECIMAL_RELATIVE * max(1, abs(target))
                                and Fraction.from_float(float(snapped)) == snapped
                            ):
                                ends[index] = float(snapped)
                    self.edge_ranges[coedge.edge] = (float(ends[0]), float(ends[1]))

    @staticmethod
    def _native_pcurve(
        curve: AbstractCurve,
        patch: AbstractSurfacePatch,
        label: str,
        /,
    ) -> AbstractCurve:
        pcurve = native_edge_pcurve(curve, patch)
        if pcurve is None:
            raise refuse(
                "unsupported-entity",
                label,
                f"{label}.pcurve must have an exact native source map.",
            )
        return pcurve

    def natural_loop(
        self,
        patch: AbstractSurfacePatch,
        label: str,
        /,
        *,
        boundary_vertices: tuple[int, ...] = (),
    ) -> list[StagedCoedge]:
        """Bind declared point boundaries to the source's seam/pole topology."""
        source = patch
        while isinstance(source, (OffsetSurface, PlacedSurface)):
            source = (
                source.base if isinstance(source, OffsetSurface) else source.definition
            )
        if not isinstance(source, (SpherePatch, TorusPatch)):
            raise refuse(
                "unsupported-entity",
                label,
                f"CadStage.natural_loop.patch must declare a finite natural boundary when edges are absent; got {type(patch).__name__}",
            )
        sphere = isinstance(source, SpherePatch)
        lower, upper = (-0.5 * pi, 0.5 * pi) if sphere else (0.0, _TWO_PI)
        box = np.asarray(((0.0, lower), (_TWO_PI, upper)), dtype=np.float64)
        patch.validate_parameter_box(box)
        parameters = np.asarray(((0.0, lower), (0.0, upper)), dtype=np.float64)
        points = np.asarray(
            patch.evaluate(jnp.asarray(parameters, dtype=jnp.float64)),
            dtype=np.float64,
        )
        if not np.all(np.isfinite(points)):
            raise refuse(
                "inconsistent-geometry",
                label,
                f"CadStage.natural_loop.patch must have finite values at its natural seams and poles; got points={points!r}",
            )
        if sphere:
            radius = self._sphere_chart_radius(patch)
            if radius is None or radius == 0:
                raise refuse(
                    "unsupported-entity",
                    label,
                    "CadStage.natural_loop.patch requires an exact nonzero sphere-pole radius.",
                )
        # A closed torus seam is one native tube period by construction; only
        # the open sphere meridian carries an authored latitude interval.
        seam_curve = SurfaceIsoparametricCurve(
            patch, 0, 0.0, parameter_range=(lower, upper) if sphere else None
        )
        seam_source = self._native_pcurve(seam_curve, patch, label)
        left = PeriodicPCurve(seam_source, patch, (0, 0))
        right = PeriodicPCurve(seam_source, patch, (1, 0))
        # Only explicitly declared face-bound vertices can own these corners.
        # The geometric check validates that incidence; it does not weld an
        # unrelated staged vertex merely because its coordinates are nearby.
        corners: list[int | None] = [None, None] if sphere else [None]
        for vertex in boundary_vertices:
            if vertex < 0 or vertex >= len(self.vertices):
                raise ValueError(
                    f"CadStage.natural_loop.boundary_vertices must reference a staged vertex; got vertex={vertex!r}, count={len(self.vertices)}"
                )
            candidates = [
                side
                for side in range(len(corners))
                if np.linalg.norm(self.vertices[vertex] - points[side]) <= self.tolerance
            ]
            if len(candidates) != 1:
                raise refuse(
                    "inconsistent-geometry",
                    label,
                    f"CadStage.natural_loop.boundary_vertices must identify one natural corner per declared vertex; got vertex={vertex}, candidates={candidates!r}",
                )
            side = candidates[0]
            if corners[side] is not None and corners[side] != vertex:
                raise refuse(
                    "inconsistent-geometry",
                    label,
                    f"CadStage.natural_loop.boundary_vertices must identify distinct natural corners; got vertex={vertex}, corner={side}, previous_vertex={corners[side]}",
                )
            corners[side] = vertex
        corner = corners[0]
        if corner is None:
            corner = self.vertex(points[0], label)
        other = corners[1] if sphere else corner
        if other is None:
            other = self.vertex(points[1], label)
        seam = self.edge(seam_curve, corner, other, label, parameter_range=(lower, upper))
        if sphere:
            bottom = self.degenerate(corner, 0.0, _TWO_PI)
            top = self.degenerate(other, 0.0, _TWO_PI)
            # Pole sides traverse one native longitude turn and meet both
            # seam sheets at symbolic 0 and 2 pi endpoints rather than binary tau.
            self.native_phase(bottom, patch, 0)
            self.native_phase(top, patch, 0)
            bottom_curve = LineCurve((0.0, lower), (1.0, 0.0))
            top_curve = LineCurve((0.0, upper), (1.0, 0.0))
        else:
            ring_curve = SurfaceIsoparametricCurve(patch, 1, 0.0)
            bottom = self.edge(
                ring_curve, corner, corner, label, parameter_range=(0.0, _TWO_PI)
            )
            top = bottom
            # The closed ring and seam each traverse one native turn; their
            # sheets meet at symbolic 0 and 2 pi on both proved periods.
            self.native_phase(bottom, patch, 0)
            self.native_phase(seam, patch, 1)
            ring_source = self._native_pcurve(ring_curve, patch, label)
            bottom_curve = PeriodicPCurve(ring_source, patch, (0, 0))
            top_curve = PeriodicPCurve(ring_source, patch, (0, 1))
        self.natural_boundaries += 1
        return [
            StagedCoedge(bottom, 1, bottom_curve, label),
            StagedCoedge(seam, 1, right, label),
            StagedCoedge(top, -1, top_curve, label),
            StagedCoedge(seam, -1, left, label),
        ]

    # ------------------------------------------------------------ resolution

    @staticmethod
    def _native_period_turns(
        patch: AbstractSurfacePatch, shift: np.ndarray, /
    ) -> tuple[int, int] | None:
        turns: list[int] = []
        for delta, period, symbol in zip(
            shift, patch.periods, patch.native_period_symbols(), strict=True
        ):
            if delta == 0.0:
                turns.append(0)
                continue
            if period is None or symbol != "two_pi":
                return None
            turn = round(float(delta) / period)
            if turn == 0 or float(delta) != turn * period:
                return None
            turns.append(turn)
        return turns[0], turns[1]

    @classmethod
    def _shift_pcurve(
        cls,
        pcurve: AbstractCurve,
        patch: AbstractSurfacePatch,
        shift: np.ndarray,
        /,
    ) -> AbstractCurve:
        turns = cls._native_period_turns(patch, shift)
        if turns is not None:
            return PeriodicPCurve(pcurve, patch, turns)
        return transform_pcurve(
            pcurve, (1.0, 1.0), 1.0, (float(shift[0]), float(shift[1]))
        )

    @classmethod
    def _canonical_native_sheet(
        cls,
        pcurve: AbstractCurve,
        patch: AbstractSurfacePatch,
        parameter_range: tuple[float, float],
        /,
    ) -> AbstractCurve:
        if isinstance(pcurve, PeriodicPCurve):
            return pcurve
        periods = patch.periods
        symbols = patch.native_period_symbols()
        if not any(symbol is not None for symbol in symbols):
            return pcurve
        if isinstance(pcurve, BSplineCurve):
            # A spline p-curve is a native sheet only when it is exactly the
            # affine isoline over the authored edge range (formats without a
            # line p-curve entity write seams this way).
            affine = _affine_bspline_pcurve(pcurve, *parameter_range)
            if affine is None:
                return pcurve
            sheet = cls._canonical_native_sheet(affine, patch, parameter_range)
            return sheet if isinstance(sheet, PeriodicPCurve) else pcurve
        carrier = explicit_carrier(pcurve)
        affine = _affine_curve_coefficients(carrier, _ZERO_UV)
        if affine is None:
            return pcurve
        constant, direction = affine
        source_constant = list(constant)
        turns = [0, 0]
        native_sheet = False
        for axis, (value, period, symbol) in enumerate(
            zip(constant, periods, symbols, strict=True)
        ):
            if period is None or symbol != "two_pi":
                continue
            turn = round(float(value) / period)
            if direction[axis] and turn == 0:
                continue
            if float(value) == turn * period:
                source_constant[axis] = Fraction()
                turns[axis] = turn
                native_sheet = True
        if native_sheet:
            source = LineCurve(
                tuple(float(value) for value in source_constant),
                tuple(float(value) for value in direction),
                origin_exact=tuple(source_constant),
                direction_exact=direction,
            )
            return PeriodicPCurve(source, patch, (turns[0], turns[1]))
        if isinstance(pcurve, LineCurve):
            return pcurve
        if any(
            symbol == "two_pi" and derivative
            for derivative, symbol in zip(direction, symbols, strict=True)
        ):
            source = LineCurve(
                (0.0, 0.0),
                tuple(float(value) for value in direction),
                origin_exact=(Fraction(), Fraction()),
                direction_exact=direction,
            )
            return _rational_affine_pcurve(
                source,
                ((Fraction(1), Fraction()), (Fraction(), Fraction(1))),
                (constant[0], constant[1]),
                "native periodic p-curve",
            )
        return pcurve

    def _coedge_pcurve(self, face: StagedFace, coedge: StagedCoedge, /) -> AbstractCurve:
        edge = coedge.edge
        curve_index = self.edge_curves[edge]
        curve = None if curve_index == -1 else self.curves[curve_index]
        if coedge.pcurve is not None:
            self.pcurves_honored += 1
            return coedge.pcurve
        if curve is None:
            raise refuse(
                "inconsistent-geometry",
                coedge.label,
                f"StagedCoedge.pcurve must be present for a degenerate edge; got edge={edge}, curve_index={curve_index}, pcurve={coedge.pcurve!r}",
                (face.label,),
            )
        try:
            recovered = recover_pcurve(face.patch, curve, self.fit_policy)
        except ValueError as error:
            raise refuse(
                "inconsistent-geometry", coedge.label, str(error), (face.label,)
            ) from error
        self.pcurves_exact += 1
        return recovered

    def _ends(self, coedge: StagedCoedge, pcurve: AbstractCurve, /) -> np.ndarray:
        first, last = self.edge_ranges[coedge.edge]
        ends = np.asarray(pcurve.evaluate(jnp.asarray((first, last), dtype=jnp.float64)))
        return ends if coedge.sense > 0 else ends[::-1]

    def _resolve_loop(
        self, face: StagedFace, loop: list[StagedCoedge], /
    ) -> list[tuple[StagedCoedge, AbstractCurve]]:
        """Close p-curve loops with exact seam representatives and pole edges.

        A collapsed isoline is repaired only when its endpoints are exact
        Q-valued source images on one declared native period; otherwise import
        refuses rather than promoting a tolerance-coincident point. Seam uses
        on the same trace are separated by exact whole periods. Any other gap
        is refused.
        """
        periods = face.patch.periods
        resolved: list[tuple[StagedCoedge, AbstractCurve]] = []
        previous_end: np.ndarray | None = None
        previous_source: tuple[StagedCoedge, AbstractCurve] | None = None
        traces: dict[int, AbstractCurve] = {}
        for coedge in loop:
            pcurve = self._coedge_pcurve(face, coedge)
            pcurve = self._canonical_native_sheet(
                pcurve, face.patch, self.edge_ranges[coedge.edge]
            )
            source_trace = pcurve
            ends = self._ends(coedge, pcurve)
            if coedge.edge in traces:
                shift = self._seam_shift(
                    coedge, traces[coedge.edge], pcurve, ends, periods
                )
                if np.any(shift != 0.0):
                    pcurve = self._shift_pcurve(pcurve, face.patch, shift)
                    ends = ends + shift
            traces.setdefault(coedge.edge, source_trace)
            if previous_end is not None and not self._connected(previous_end, ends[0]):
                if previous_source is None:
                    raise CadCodecInvariantError(
                        "CadStage._resolve_loop.previous_source must accompany a resolved endpoint; got None"
                    )
                if not self._pole_gap(face, previous_end, ends[0]):
                    shift = self._period_shift(previous_end, ends[0], periods)
                    pcurve = self._shift_pcurve(pcurve, face.patch, shift)
                    ends = ends + shift
                if not self._connected(previous_end, ends[0]):
                    resolved.append(
                        self._pole_coedge(
                            face,
                            coedge,
                            previous_end,
                            ends[0],
                            previous_source,
                            (coedge, pcurve),
                        )
                    )
            resolved.append((coedge, pcurve))
            previous_end = ends[1]
            previous_source = coedge, pcurve
        if previous_end is None or previous_source is None:
            raise CadCodecInvariantError(
                f"StagedFace.loops must contain nonempty coedge loops; got loop_size={len(loop)}"
            )
        first_source = resolved[0]
        start = self._ends(*first_source)[0]
        if not self._connected(previous_end, start):
            resolved.insert(
                0,
                self._pole_coedge(
                    face,
                    loop[0],
                    previous_end,
                    start,
                    previous_source,
                    first_source,
                ),
            )
        return self._canonical_degenerate_coedges(face, resolved)

    def _seam_shift(
        self,
        coedge: StagedCoedge,
        first_use: AbstractCurve,
        pcurve: AbstractCurve,
        ends: np.ndarray,
        periods: tuple[float | None, float | None],
    ) -> np.ndarray:
        """Period offset separating a second seam use from a coincident first use.

        A counterclockwise loop runs up (+v) along its larger-u seam side and
        right (+u) along its smaller-v side.
        """
        first, last = self.edge_ranges[coedge.edge]
        middle = np.asarray(0.5 * (first + last))
        shift = np.zeros(2)
        if not self._connected(
            np.asarray(
                first_use.evaluate(jnp.asarray(middle, dtype=jnp.float64)),
                dtype=np.float64,
            ),
            np.asarray(
                pcurve.evaluate(jnp.asarray(middle, dtype=jnp.float64)),
                dtype=np.float64,
            ),
        ):
            return shift
        moving = int(np.argmax(np.abs(ends[1] - ends[0])))
        fixed = 1 - moving
        period = periods[fixed]
        if period is None:
            return shift
        increasing = ends[1][moving] > ends[0][moving]
        shift[fixed] = period if increasing == (moving == 1) else -period
        return shift

    def _connected(self, head: np.ndarray, tail: np.ndarray, /) -> bool:
        return bool(
            np.max(np.abs(tail - head))
            <= self._parameter_slack(np.concatenate((head, tail)))
        )

    @staticmethod
    def _parameter_slack(point: np.ndarray, /) -> float:
        return 1.0e-9 * max(1.0, float(np.max(np.abs(point))))

    def _period_shift(
        self,
        target: np.ndarray,
        start: np.ndarray,
        periods: tuple[float | None, float | None],
    ) -> np.ndarray:
        shift = np.zeros(2)
        for axis, period in enumerate(periods):
            if period is not None:
                shift[axis] = period * round((target[axis] - start[axis]) / period)
        return shift

    def _pole_gap(self, face: StagedFace, head: np.ndarray, tail: np.ndarray, /) -> bool:
        """Whether the straight parameter segment ``head -> tail`` collapses to a point."""
        moving = int(np.argmax(np.abs(tail - head)))
        slack = self._parameter_slack(np.concatenate((head, tail)))
        if abs(tail[1 - moving] - head[1 - moving]) > slack or moving != 0:
            return False
        patch = face.patch
        sphere_radius = self._sphere_chart_radius(patch)
        if sphere_radius is not None:
            if sphere_radius == 0:
                return False
            return any(
                abs(head[1] - value) <= slack and abs(tail[1] - value) <= slack
                for value in (-0.5 * pi, 0.5 * pi)
            )
        source = patch
        while isinstance(source, PlacedSurface):
            source = source.definition
        if isinstance(source, OffsetSurface):
            box = np.stack((np.minimum(head, tail), np.maximum(head, tail)))
            source.validate_parameter_box(box)
            source = source.analytic_equivalent()
            if source is None:
                return False
        v = float(head[1])
        match source:
            case SpherePatch():
                return False
            case ConePatch():
                radius = float(source.reference_radius) + v * np.tan(
                    float(source.semi_angle)
                )
            case TorusPatch():
                radius = float(source.major_radius) + float(source.minor_radius) * np.cos(
                    v
                )
            case RevolutionSurface():
                point = np.asarray(
                    source.curve.evaluate(jnp.asarray(v, dtype=jnp.float64))
                ) - np.asarray(source.axis_origin)
                axis = np.asarray(source.axis_direction)
                radius = float(np.linalg.norm(point - (point @ axis) * axis))
            case _:
                return False
        return abs(radius) <= self.tolerance

    def _exact_coedge_endpoint(
        self,
        source: tuple[StagedCoedge, AbstractCurve],
        patch: AbstractSurfacePatch,
        axis: int,
        start: bool,
        /,
    ) -> tuple[tuple[ExactScalar, ...], Fraction | PeriodOffset] | None:
        coedge, pcurve = source
        first, last = self.edge_ranges[coedge.edge]
        parameter = first if (coedge.sense > 0) == start else last
        periods = patch.periods
        symbols = patch.native_period_symbols()
        period = periods[axis]
        if symbols[axis] != "two_pi" or period is None:
            return None
        seam = self._exact_seam_endpoint(pcurve, axis, parameter)
        if seam is not None:
            return seam
        affine = _affine_curve_coefficients(explicit_carrier(pcurve), _ZERO_UV)
        if affine is None:
            return None
        offset, direction = (np.asarray(values, dtype=np.float64) for values in affine)
        expected = np.zeros(2, dtype=np.float64)
        expected[axis] = 1.0
        if offset[axis] != 0.0 or not np.array_equal(direction, expected):
            return None
        domain = (
            (patch.parameter_domain[0][axis], patch.parameter_domain[1][axis])
            if patch.parameter_domain is not None
            else (0.0, period)
        )
        lower, upper = domain
        try:
            if parameter == lower:
                endpoint = NativePeriodEndpoint(
                    pcurve,
                    patch,
                    axis,
                    rational=Fraction.from_float(lower),
                    turns=0,
                )
            elif parameter == upper and float(upper) - float(lower) == period:
                endpoint = NativePeriodEndpoint(
                    pcurve,
                    patch,
                    axis,
                    rational=Fraction.from_float(lower),
                    turns=1,
                )
            else:
                return None
        except ValueError:
            return None
        point = exact_curve_parameter_point(pcurve, endpoint.exact_parameter)
        if point is None:
            return None
        return tuple(point), endpoint.exact_parameter

    @staticmethod
    def _exact_seam_endpoint(
        pcurve: AbstractCurve, axis: int, parameter: float, /
    ) -> tuple[tuple[ExactScalar, ...], Fraction | PeriodOffset] | None:
        """Exact chart endpoint of a seam isoline on a declared native-period sheet.

        A seam use is an exact affine isoline of the periodic coordinate whose
        sheet is an integer number of native turns, so its periodic coordinate
        is ``offset + turns * period`` with no rounding; the transverse
        coordinate is the exact affine image of the authored edge parameter.
        """
        if not isinstance(pcurve, PeriodicPCurve):
            return None
        fixed = 1 - axis
        if pcurve.period_shifts[fixed] != 0:
            return None
        affine = _affine_curve_coefficients(
            explicit_carrier(pcurve.source_curve), _ZERO_UV
        )
        if affine is None:
            return None
        offset, direction = affine
        if direction[axis] != 0 or direction[fixed] == 0:
            return None
        point: list[ExactScalar] = [Fraction(), Fraction()]
        point[axis] = offset[axis]
        point[fixed] = offset[fixed] + direction[fixed] * Fraction.from_float(parameter)
        rational = offset[axis]
        if not isinstance(rational, Fraction):
            return None
        return tuple(point), period_offset(rational, Fraction(pcurve.period_shifts[axis]))

    def _pole_curve(
        self,
        face: StagedFace,
        label: str,
        head: np.ndarray,
        tail: np.ndarray,
        head_source: tuple[StagedCoedge, AbstractCurve],
        tail_source: tuple[StagedCoedge, AbstractCurve],
        /,
    ) -> tuple[AbstractCurve, int, int, bool]:
        """Synthesize a pole p-curve only from exact source-period endpoint evidence."""
        if not self._pole_gap(face, head, tail):
            raise refuse(
                "inconsistent-geometry",
                label,
                f"StagedFace.loops must connect consecutive p-curve endpoints in the parameter plane; got head={head!r}, tail={tail!r}",
                (face.label,),
            )
        moving = int(np.argmax(np.abs(tail - head)))
        fixed = 1 - moving
        head_evidence = self._exact_coedge_endpoint(
            head_source, face.patch, moving, False
        )
        tail_evidence = self._exact_coedge_endpoint(tail_source, face.patch, moving, True)
        if head_evidence is None or tail_evidence is None:
            raise refuse(
                "unsupported-entity",
                label,
                "A synthesized pole edge requires exact Q-valued source endpoint images and a source-declared native period; rounded coordinates are not exact evidence.",
                (face.label,),
            )
        head_point, head_parameter = head_evidence
        tail_point, tail_parameter = tail_evidence

        def parts(value: Fraction | PeriodOffset, /) -> tuple[Fraction, Fraction]:
            if isinstance(value, PeriodOffset):
                return value.rational, value.turns
            return value, Fraction()

        head_rational, head_turns = parts(head_parameter)
        tail_rational, tail_turns = parts(tail_parameter)
        if (
            head_rational != tail_rational
            or abs(head_turns - tail_turns) != 1
            or head_turns.denominator != 1
            or tail_turns.denominator != 1
            or head_point[fixed] != tail_point[fixed]
            or head_point[moving] != tail_point[moving]
        ):
            raise refuse(
                "unsupported-entity",
                label,
                "A synthesized pole edge requires exact endpoint images at the same chart point and one source-declared native turn.",
                (face.label,),
            )
        direction_exact = [Fraction(), Fraction()]
        direction_exact[moving] = Fraction(1)
        offset = [Fraction(), Fraction()]
        offset[moving] = head_point[moving]
        offset[fixed] = head_point[fixed]
        pcurve: AbstractCurve = LineCurve(
            np.asarray(offset, dtype=np.float64),
            np.asarray(direction_exact, dtype=np.float64),
            origin_exact=tuple(offset),
            direction_exact=tuple(direction_exact),
        )
        shifts = [0, 0]
        shifts[moving] = int(min(head_turns, tail_turns))
        if any(shifts):
            pcurve = PeriodicPCurve(pcurve, face.patch, (shifts[0], shifts[1]))
        sense = 1 if tail_turns > head_turns else -1
        return pcurve, sense, moving, True

    def _pole_coedge(
        self,
        face: StagedFace,
        neighbor: StagedCoedge,
        head: np.ndarray,
        tail: np.ndarray,
        head_source: tuple[StagedCoedge, AbstractCurve],
        tail_source: tuple[StagedCoedge, AbstractCurve],
        /,
    ) -> tuple[StagedCoedge, AbstractCurve]:
        """Insert one missing pole edge without inventing a tolerance root."""
        pcurve, sense, moving, native_period = self._pole_curve(
            face,
            neighbor.label,
            head,
            tail,
            head_source,
            tail_source,
        )
        # The missing pole coedge ends at the neighboring authored start vertex.
        start, end = self.edge_vertices[neighbor.edge]
        vertex = start if neighbor.sense > 0 else end
        edge = self.degenerate(vertex, 0.0, _TWO_PI if native_period else 1.0)
        if native_period:
            self.native_phase(edge, face.patch, 0 if moving == 0 else 1)
        return StagedCoedge(edge, sense, None, neighbor.label), pcurve

    def _canonical_degenerate_coedges(
        self,
        face: StagedFace,
        resolved: list[tuple[StagedCoedge, AbstractCurve]],
        /,
    ) -> list[tuple[StagedCoedge, AbstractCurve]]:
        """Bind pole edges only from exact source-axis period evidence."""
        updates: list[tuple[int, int, StagedCoedge, AbstractCurve, int, bool]] = []
        for position, (coedge, pcurve) in enumerate(resolved):
            edge = coedge.edge
            if self.edge_curves[edge] >= 0:
                continue
            previous = resolved[position - 1]
            following = resolved[(position + 1) % len(resolved)]
            head, tail = self._ends(*previous)[1], self._ends(*following)[0]
            own = self._ends(coedge, pcurve)
            if (
                not self._connected(head, own[0])
                or not self._connected(own[1], tail)
                or not self._pole_gap(face, own[0], own[1])
            ):
                raise refuse(
                    "inconsistent-geometry",
                    coedge.label,
                    f"StagedCoedge.pcurve must span an exact source-mapped one-turn collapsed chart side; got coedge={coedge.label!r}, face={face.label!r}",
                    (face.label,),
                )
            canonical, sense, moving, native_period = self._pole_curve(
                face,
                coedge.label,
                head,
                tail,
                previous,
                following,
            )
            updates.append(
                (
                    position,
                    edge,
                    StagedCoedge(edge, sense, None, coedge.label),
                    canonical,
                    moving,
                    native_period,
                )
            )
        result = list(resolved)
        for position, edge, coedge, pcurve, moving, native_period in updates:
            self.edge_ranges[edge] = (0.0, _TWO_PI if native_period else 1.0)
            if native_period:
                self.native_phase(edge, face.patch, 0 if moving == 0 else 1)
            result[position] = coedge, pcurve
        return result

    @staticmethod
    def _signed_area(uv: np.ndarray, /) -> float:
        x, y = uv[:, 0], uv[:, 1]
        return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))

    def _loop_polygon(
        self, resolved: list[tuple[StagedCoedge, AbstractCurve]], /
    ) -> np.ndarray:
        points = []
        for coedge, pcurve in resolved:
            first, last = self.edge_ranges[coedge.edge]
            samples = np.linspace(first, last, 17)
            if coedge.sense < 0:
                samples = samples[::-1]
            points.append(
                np.asarray(pcurve.evaluate(jnp.asarray(samples[:-1])), dtype=np.float64)
            )
        return np.concatenate(points)

    def _box(
        self,
        patch: AbstractSurfacePatch,
        loops: list[list[tuple[StagedCoedge, AbstractCurve]]],
    ) -> np.ndarray:
        boxes = []
        for loop in loops:
            for coedge, pcurve in loop:
                first, last = self.edge_ranges[coedge.edge]
                boxes.append(_representative_pcurve_box(patch, pcurve, first, last))
        stacked = np.stack(boxes)
        box = np.stack((np.min(stacked[:, 0], axis=0), np.max(stacked[:, 1], axis=0)))
        natural = natural_box(patch)
        box = np.stack((np.maximum(box[0], natural[0]), np.minimum(box[1], natural[1])))
        # Whole-turn sheet p-curves use the binary period representative, so a
        # full native turn is exactly one binary period wide; other symbolic
        # gauges keep outward bounds, and only a source-declared domain clips.
        return box

    @staticmethod
    def _ring_frame_phase(
        patch: AbstractSurfacePatch, curve: AbstractCurve, /
    ) -> tuple[Fraction, int] | None:
        """Prove a native quarter-turn from frame coefficients, never an angle fit."""
        if not isinstance(patch, (CylinderPatch, TorusPatch)) or not isinstance(
            curve, CircleCurve
        ):
            return None
        x, y = np.asarray(patch.first_axis), np.asarray(patch.second_axis)
        frames = ((x, y), (y, -x), (-x, -y), (-y, x))
        for quarter, (first, second) in enumerate(frames):
            if np.array_equal(curve.first_axis, first):
                for sense in (1, -1):
                    if np.array_equal(curve.second_axis, sense * second):
                        return Fraction(quarter if quarter < 3 else -1, 4), sense
        return None

    def _ring_pcurve(self, face: StagedFace, coedge: StagedCoedge, /) -> AbstractCurve:
        curve = self.curves[self.edge_curves[coedge.edge]]
        pcurve = self._coedge_pcurve(face, coedge)
        frame = self._ring_frame_phase(face.patch, curve)
        if (
            coedge.pcurve is not None
            or frame is None
            or not isinstance(pcurve, LineCurve)
        ):
            return pcurve
        phase, sense = frame
        if not phase:
            return pcurve
        # A rational affine map of a native period keeps its irrational atom.
        # The source frame, not the rounded atan2 recovery, owns this phase.
        source = LineCurve(
            (0.0, float(pcurve.origin[1])), (float(Fraction(sense) / phase), 0.0)
        )
        return _rational_affine_pcurve(
            PeriodicPCurve(source, face.patch, (1, 0)),
            ((phase, Fraction()), (Fraction(), Fraction(1))),
            (Fraction(), Fraction()),
            "ring p-curve frame",
        )

    def _split_periodic_bands(self) -> None:
        """Refine all incident uses before cutting winding faces into chart loops.

        Original curve definitions and authored vertices remain in their banks.
        Only topology is subdivided, with explicit source-entity ancestry. The
        canonical source interval is one native turn; every original endpoint
        remains a split point rather than being discarded by seam rebasing.
        """
        plans = []
        cuts: dict[int, set[Fraction]] = {}
        owners: dict[int, AbstractSurfacePatch] = {}
        for face in self.faces:
            if len(face.loops) != 2 or any(len(loop) != 1 for loop in face.loops):
                continue
            rings = [loop[0] for loop in face.loops]
            frames = []
            for ring in rings:
                edge = ring.edge
                first, last = self.edge_ranges[edge]
                index = self.edge_curves[edge]
                frame = (
                    None
                    if index < 0
                    else self._ring_frame_phase(face.patch, self.curves[index])
                )
                if (
                    frame is None
                    or self.edge_vertices[edge][0] != self.edge_vertices[edge][1]
                    or last != first + _TWO_PI
                ):
                    break
                frames.append(frame)
            if len(frames) != 2:
                continue
            if frames[0][1] * rings[0].sense != -frames[1][1] * rings[1].sense:
                continue
            phases = [(-phase * sense) % 1 for phase, sense in frames]
            plans.append((face, phases))
            for ring, phase in zip(rings, phases, strict=True):
                cuts.setdefault(ring.edge, {Fraction()}).add(phase)
                owners.setdefault(ring.edge, face.patch)
        if not plans:
            return
        replacements: dict[int, list[int]] = {}
        cut_vertices: dict[tuple[int, Fraction], int] = {}
        labels = dict(self.provenance)
        for edge, phases in sorted(cuts.items()):
            curve_index = self.edge_curves[edge]
            curve = self.curves[curve_index]
            assert isinstance(curve, CircleCurve)
            original_vertex = self.edge_vertices[edge][0]
            original_parameter = self.edge_ranges[edge][0]
            axes = (
                np.asarray(curve.first_axis),
                np.asarray(curve.second_axis),
                -np.asarray(curve.first_axis),
                -np.asarray(curve.second_axis),
            )
            radial_exact = tuple(
                Fraction(float(value)) - Fraction(float(center))
                for value, center in zip(
                    self.vertices[original_vertex], np.asarray(curve.center), strict=True
                )
            )
            original_phase = None
            for quarter, axis in enumerate(axes):
                direction = tuple(Fraction(float(value)) for value in axis)
                if sum(
                    a * b for a, b in zip(radial_exact, direction, strict=True)
                ) > 0 and all(
                    radial_exact[i] * direction[j] == radial_exact[j] * direction[i]
                    for i, j in ((0, 1), (0, 2), (1, 2))
                ):
                    original_phase = Fraction(quarter, 4)
                    phases.add(original_phase)
                    break
            if (
                original_parameter == 0.0
                and original_phase == 0
                and phases == {Fraction()}
            ):
                # No subdivision: retain the authored edge identity verbatim.
                replacements[edge] = [edge]
                cut_vertices[edge, Fraction()] = original_vertex
                continue
            parameter = original_parameter % _TWO_PI
            source_label = labels[f"edge:{edge}"]
            self.provenance = [
                (f"curve:{curve_index}", value) if key == f"edge:{edge}" else (key, value)
                for key, value in self.provenance
            ]
            points: dict[float, tuple[int, Fraction | None]] = {}
            for phase in sorted(phases):
                value = float(phase) * _TWO_PI
                if phase == original_phase:
                    vertex = original_vertex
                else:
                    quarter = int(4 * phase)
                    vertex = self.vertex(
                        np.asarray(curve.center) + float(curve.radius) * axes[quarter],
                        f"{source_label}:DERIVED:chart-cut:{phase}",
                    )
                points[value] = (vertex, phase)
                cut_vertices[edge, phase] = vertex
            if original_phase is None:
                # The original literal vertex selects a ray, not a snapped
                # native angle. Isolate its circle/meridian-plane intersection.
                radial = self.vertices[original_vertex] - np.asarray(curve.center)
                normal = np.cross(curve.first_axis, curve.second_axis)
                plane = PlanePatch(curve.center, radial, normal)
                center = np.asarray((parameter, 1.0, 0.0))
                radius = 1.0e-6 * np.maximum(1.0, np.abs(center))
                region = SurfaceRegion(plane, np.asarray(((0.5, -0.5), (1.5, 0.5))))
                root = CurveSurfaceIntersectionRoot(
                    curve,
                    region,
                    parameter_lower=center - radius,
                    parameter_upper=center + radius,
                )
                endpoint = TrimRootEndpoint(root, "first")
                lower, upper = endpoint.parameter_enclosure()
                parameter = 0.5 * (lower + upper)
                if parameter in points:
                    # Coincident numerical representatives are not authority
                    # to discard the authored vertex or weld it to a new cut.
                    raise refuse(
                        "unsupported-entity",
                        source_label,
                        "Authored vertex and chart cut require a certified shared root before subdivision.",
                    )
                # This is still the authored tolerance vertex. The isolated
                # ray root selects its scalar representative; it does not
                # authorize an exact vertex/surface correspondence on every
                # incident (possibly independently rounded) source surface.
                points[parameter] = (original_vertex, None)
            zero = points[0.0]
            points[_TWO_PI] = (zero[0], Fraction(1))
            ordered = sorted(points.items())
            segments = []
            self.native_phases.pop(edge, None)
            for position, ((first, start), (last, end)) in enumerate(
                zip(ordered[:-1], ordered[1:], strict=True)
            ):
                child = edge if position == 0 else len(self.edge_curves)
                if position:
                    self.edge_curves.append(curve_index)
                    self.edge_ranges.append((first, last))
                    self.edge_vertices.append((start[0], end[0]))
                else:
                    self.edge_ranges[child] = (first, last)
                    self.edge_vertices[child] = (start[0], end[0])
                self.provenance.append(
                    (f"edge:{child}", f"{source_label}:DERIVED:subedge:{position}")
                )
                self.split_endpoint_phases[child] = start[1], end[1]
                self.split_period_surfaces[child] = owners[edge]
                segments.append(child)
            replacements[edge] = segments
        # Propagate the same source subdivision through EVERY incident face,
        # including planar caps and other periodic charts.
        for face in self.faces:
            loops = []
            for loop in face.loops:
                refined = []
                for coedge in loop:
                    children = replacements.get(coedge.edge)
                    if children is None:
                        refined.append(coedge)
                        continue
                    pcurve = self._ring_pcurve(face, coedge)
                    for child in children if coedge.sense > 0 else reversed(children):
                        refined.append(
                            StagedCoedge(child, coedge.sense, pcurve, coedge.label)
                        )
                loops.append(refined)
            face.loops = loops
        for face, phases in plans:
            rings = face.loops
            starts = []
            heights = []
            for ring, phase in zip(rings, phases, strict=True):
                # Each original source edge's children retain its curve bank.
                source_edge = next(
                    edge
                    for edge, children in replacements.items()
                    if ring[0].edge in children
                )
                vertex = cut_vertices[source_edge, phase]
                position = next(
                    index
                    for index, use in enumerate(ring)
                    if self.edge_vertices[use.edge][0 if use.sense > 0 else 1] == vertex
                )
                ring[:] = ring[position:] + ring[:position]
                starts.append(vertex)
                assert ring[0].pcurve is not None
                ends = self._ends(ring[0], ring[0].pcurve)
                heights.append(float(ends[0, 1]))
                target = 0.0 if ends[1, 0] > ends[0, 0] else _TWO_PI
                shift = round((target - ends[0, 0]) / _TWO_PI)
                for use in ring:
                    assert use.pcurve is not None
                    use.pcurve = PeriodicPCurve(use.pcurve, face.patch, (shift, 0))
            lower, upper = sorted(heights)
            label = f"{face.label}:DERIVED:periodic-chart-identification"
            seam_curve = SurfaceIsoparametricCurve(
                face.patch, 0, 0.0, parameter_range=(lower, upper)
            )
            increasing = heights[0] < heights[1]
            seam = self.edge(
                seam_curve,
                starts[0] if increasing else starts[1],
                starts[1] if increasing else starts[0],
                label,
                parameter_range=(lower, upper),
            )
            seam_source = self._native_pcurve(seam_curve, face.patch, label)
            sense = 1 if increasing else -1
            face.loops = [
                [
                    *rings[0],
                    StagedCoedge(
                        seam,
                        sense,
                        PeriodicPCurve(seam_source, face.patch, (1, 0)),
                        label,
                    ),
                    *rings[1],
                    StagedCoedge(
                        seam,
                        -sense,
                        PeriodicPCurve(seam_source, face.patch, (0, 0)),
                        label,
                    ),
                ]
            ]

    def _periodic_band(
        self, face: StagedFace, /
    ) -> list[tuple[StagedCoedge, AbstractCurve]] | None:
        """Cut two authored winding rings along their common chart isoline.

        The derived edge is a chart identification, used twice with opposite
        senses, not another physical boundary. Full-turn source roots and exact
        affine coefficients establish the winding; proximity to a period or to
        a common phase is deliberately insufficient.
        """
        if (
            len(face.loops) != 2
            or any(len(loop) != 1 for loop in face.loops)
            or "two_pi" not in face.patch.native_period_symbols()
        ):
            return None
        rings = [loop[0] for loop in face.loops]
        for ring in rings:
            edge = ring.edge
            if (
                edge not in self.native_phases
                or self.edge_ranges[edge] != (0.0, _TWO_PI)
                or self.edge_vertices[edge][0] != self.edge_vertices[edge][1]
            ):
                return None
        curves = [self._coedge_pcurve(face, ring) for ring in rings]
        coefficients = [
            _affine_curve_coefficients(explicit_carrier(curve), _ZERO_UV)
            for curve in curves
        ]
        if any(coefficient is None for coefficient in coefficients):
            return None
        first, second = coefficients
        assert first is not None and second is not None
        constant, direction = first
        other_constant, other_direction = second
        for moving in (0, 1):
            fixed = 1 - moving
            if (
                face.patch.native_period_symbols()[moving] != "two_pi"
                or face.patch.periods[moving] != _TWO_PI
                or direction[fixed] != 0
                or other_direction[fixed] != 0
                or abs(direction[moving]) != 1
                or abs(other_direction[moving]) != 1
                or direction[moving] * rings[0].sense
                != -other_direction[moving] * rings[1].sense
                or constant[moving] != other_constant[moving]
                or constant[fixed] == other_constant[fixed]
            ):
                continue
            # Both carriers start at the authored common phase. A decreasing
            # carrier uses the adjacent sheet without changing its winding.
            resolved = []
            for ring, curve, derivative in zip(
                rings, curves, (direction, other_direction), strict=True
            ):
                turns = [0, 0]
                turns[moving] = int(derivative[moving] < 0)
                resolved.append(
                    (ring, PeriodicPCurve(curve, face.patch, (turns[0], turns[1])))
                )
            lower, upper = sorted((float(constant[fixed]), float(other_constant[fixed])))
            increasing = constant[fixed] < other_constant[fixed]
            vertices = tuple(self.edge_vertices[ring.edge][0] for ring in rings)
            label = f"{face.label}:DERIVED:periodic-chart-identification"
            axis: Literal[0, 1] = 0 if moving == 0 else 1
            seam_curve = SurfaceIsoparametricCurve(
                face.patch,
                axis,
                float(constant[moving]),
                parameter_range=(lower, upper),
            )
            seam = self.edge(
                seam_curve,
                vertices[0] if increasing else vertices[1],
                vertices[1] if increasing else vertices[0],
                label,
                parameter_range=(lower, upper),
            )
            seam_source = self._native_pcurve(seam_curve, face.patch, label)
            first_turns = [0, 0]
            second_turns = [0, 0]
            forward = direction[moving] * rings[0].sense > 0
            first_turns[moving] = int(forward)
            second_turns[moving] = int(not forward)
            sense = 1 if increasing else -1
            return [
                resolved[0],
                (
                    StagedCoedge(seam, sense, None, label),
                    PeriodicPCurve(
                        seam_source, face.patch, (first_turns[0], first_turns[1])
                    ),
                ),
                resolved[1],
                (
                    StagedCoedge(seam, -sense, None, label),
                    PeriodicPCurve(
                        seam_source, face.patch, (second_turns[0], second_turns[1])
                    ),
                ),
            ]
        return None

    def _resolve_face(
        self, face: StagedFace, /
    ) -> tuple[list[list[tuple[StagedCoedge, AbstractCurve]]], np.ndarray]:
        band = self._periodic_band(face)
        loops = (
            [band]
            if band is not None
            else [self._resolve_loop(face, loop) for loop in face.loops]
        )
        areas = [self._signed_area(self._loop_polygon(loop)) for loop in loops]
        if face.loops_by_containment:
            # The format bounds the region by containment, not by curve
            # direction: orient the outer loop CCW and holes CW.
            for index, area in enumerate(areas):
                if area * (1.0 if index == 0 else -1.0) < 0.0:
                    loops[index] = [
                        (StagedCoedge(c.edge, -c.sense, c.pcurve, c.label), p)
                        for c, p in reversed(loops[index])
                    ]
                    areas[index] = -area
        if face.outer_known:
            if areas[0] <= 0.0 or any(area >= 0.0 for area in areas[1:]):
                raise refuse(
                    "inconsistent-geometry",
                    face.label,
                    f"StagedFace.loops must orient the outer loop counterclockwise and holes clockwise about the face normal; got signed_areas={areas!r}",
                )
        else:
            outer = [index for index, area in enumerate(areas) if area > 0.0]
            if len(outer) != 1 or any(
                area >= 0.0 for index, area in enumerate(areas) if index != outer[0]
            ):
                raise refuse(
                    "inconsistent-geometry",
                    face.label,
                    f"StagedFace.loops must determine one counterclockwise outer loop about the face normal; got signed_areas={areas!r}",
                )
            loops.insert(0, loops.pop(outer[0]))
        return loops, self._box(face.patch, loops)

    def _native_phase_surface_candidates(
        self,
        edge: int,
        resolved: list[tuple[list[list[tuple[StagedCoedge, AbstractCurve]]], np.ndarray]],
    ) -> list[tuple[int, Literal[0, 1], AbstractSurfacePatch]]:
        """Find periodic surface axes aligned with the source affine p-curve."""
        candidates: list[tuple[int, Literal[0, 1], AbstractSurfacePatch]] = []
        for face, ((loops, _), staged) in enumerate(
            zip(resolved, self.faces, strict=True)
        ):
            for loop in loops:
                for coedge, pcurve in loop:
                    if coedge.edge != edge:
                        continue
                    axis = _aligned_period_axis(staged.patch, pcurve)
                    if axis is not None:
                        candidates.append((face, axis, staged.patch))
        return candidates

    def _bind_native_phase_surfaces(
        self,
        resolved: list[tuple[list[list[tuple[StagedCoedge, AbstractCurve]]], np.ndarray]],
        /,
    ) -> None:
        """Bind curve-native turns to a symbolic-period surface definition when needed."""
        discovered: set[int] = set()
        for edge, (curve, parameter_range) in enumerate(
            zip(self.edge_curves, self.edge_ranges, strict=True)
        ):
            if (
                edge not in self.native_phases
                and curve < 0
                and parameter_range == (0.0, _TWO_PI)
            ):
                # A format-declared degenerate pole edge has no 3D carrier; its
                # symbolic one-turn support comes from the periodic face p-curve.
                self.native_phases[edge] = (None, None, Fraction())
                discovered.add(edge)
        for edge, (patch, _, rational) in tuple(self.native_phases.items()):
            if patch is not None:
                continue
            candidates = self._native_phase_surface_candidates(edge, resolved)
            if candidates:
                _, axis, owner = min(candidates, key=lambda value: value[:2])
                self.native_phases[edge] = (owner, axis, rational)
            elif edge in discovered:
                self.native_phases.pop(edge)

    def _native_phase_roots(
        self,
        pcurves: Sequence[AbstractCurve],
        coedge_edges: Sequence[int],
        face_loops: Sequence[tuple[tuple[int, ...], ...]],
        /,
    ) -> tuple[
        tuple[tuple[RootEndpoint | None, RootEndpoint | None], ...] | None,
        tuple[tuple[RootEndpoint | None, RootEndpoint | None], ...] | None,
    ]:
        """Bind symbolic native-turn endpoints to each edge carrier and coedge p-curve.

        Every coedge use binds to its own face: the edge owner's period when
        the face carries that patch, else a declared period of the face patch
        aligned with the use, else the p-curve's own native turn.
        """
        if not self.native_phases and not self.split_endpoint_phases:
            return None, None
        edge_roots: list[tuple[RootEndpoint | None, RootEndpoint | None]] = [
            (None, None)
        ] * len(self.edge_curves)
        coedge_roots: list[tuple[RootEndpoint | None, RootEndpoint | None]] = [
            (None, None)
        ] * len(coedge_edges)
        coedge_faces = [-1] * len(coedge_edges)
        for face, loops in enumerate(face_loops):
            for loop in loops:
                for coedge in loop:
                    coedge_faces[coedge] = face
        for edge, (patch, axis, rational) in sorted(self.native_phases.items()):
            uses = [
                coedge for coedge, source in enumerate(coedge_edges) if source == edge
            ]
            if not uses:
                raise ValueError(
                    f"CadStage.native_phases must have an incident coedge use; got edge={edge}, coedge_count=0"
                )
            index = self.edge_curves[edge]
            carrier = pcurves[uses[0]] if index < 0 else self.curves[index]
            edge_roots[edge] = (
                NativePeriodEndpoint(
                    carrier,
                    patch,
                    axis,
                    rational=rational,
                    turns=0,
                ),
                NativePeriodEndpoint(
                    carrier,
                    patch,
                    axis,
                    rational=rational,
                    turns=1,
                ),
            )
            for coedge in uses:
                pcurve = pcurves[coedge]
                binding = _coedge_period_binding(
                    self.faces[coedge_faces[coedge]].patch, patch, axis, pcurve
                )
                if binding is None:
                    continue
                coedge_roots[coedge] = (
                    NativePeriodEndpoint(pcurve, *binding, rational=rational, turns=0),
                    NativePeriodEndpoint(pcurve, *binding, rational=rational, turns=1),
                )
        for edge, phases in self.split_endpoint_phases.items():
            carrier = self.curves[self.edge_curves[edge]]
            patch = self.split_period_surfaces[edge]
            roots: list[RootEndpoint | None] = [None, None]
            for endpoint, phase in enumerate(phases):
                if phase is not None:
                    roots[endpoint] = NativePeriodEndpoint(carrier, turns=phase)
            edge_roots[edge] = roots[0], roots[1]
            for coedge, source in enumerate(coedge_edges):
                if source != edge:
                    continue
                binding = _coedge_period_binding(
                    self.faces[coedge_faces[coedge]].patch, patch, 0, pcurves[coedge]
                )
                if binding is None:
                    continue
                endpoints = [
                    None
                    if phase is None
                    else NativePeriodEndpoint(pcurves[coedge], *binding, turns=phase)
                    for phase in phases
                ]
                coedge_roots[coedge] = endpoints[0], endpoints[1]
        return tuple(edge_roots), tuple(coedge_roots)

    def _junction_window(
        self, patch: AbstractSurfacePatch, pcurve: AbstractCurve, parameter: float, /
    ) -> CurveTrimSegment:
        """A bounded source-incidence window, not a snapped scalar endpoint."""
        parameter_ = jnp.asarray(parameter, dtype=jnp.float64)
        uv = np.asarray(pcurve.evaluate(parameter_), dtype=np.float64)
        curve_tangent = np.asarray(jax.jacfwd(pcurve.evaluate)(parameter_))
        surface_map = np.asarray(
            surface_differential(patch, jnp.asarray(uv, dtype=jnp.float64))
        )
        tangent = surface_map @ curve_tangent
        speed = float(np.linalg.norm(tangent))
        if not isfinite(speed) or speed == 0.0:
            raise ValueError(
                f"CadStage._junction_window.pcurve must have finite positive lifted speed; got {speed!r}"
            )
        radius = self.relative_geometric_tolerance * self.scale / speed
        radius = max(radius, 32.0 * abs(float(np.spacing(parameter))))
        lower, upper = parameter - radius, parameter + radius
        if pcurve.parameter_domain is not None:
            domain_lower, domain_upper = pcurve.parameter_domain
            lower, upper = max(lower, domain_lower), min(upper, domain_upper)
        return CurveTrimSegment(pcurve, lower, upper)

    def _prepare_incidence_junctions(
        self,
        pcurves: Sequence[AbstractCurve],
        coedge_edges: Sequence[int],
        coedge_senses: Sequence[int],
        face_loops: Sequence[tuple[tuple[int, ...], ...]],
        uses: list[tuple[RootEndpoint | None, RootEndpoint | None]],
    ) -> _IncidencePlan:
        """Collect source junctions and reject collapsed/regular incidence conflicts."""
        junctions: dict[int, list[tuple[int, int, int, int, int]]] = {}
        collapsed_vertices: set[int] = set()
        needed: set[int] = set()
        for face, loops in enumerate(face_loops):
            for loop in loops:
                for first, second in zip(loop, (*loop[1:], loop[0]), strict=True):
                    end = 1 if coedge_senses[first] > 0 else 0
                    start = 0 if coedge_senses[second] > 0 else 1
                    vertex = self.edge_vertices[coedge_edges[first]][end]
                    if vertex != self.edge_vertices[coedge_edges[second]][start]:
                        raise refuse(
                            "inconsistent-geometry",
                            self.faces[face].label,
                            f"StagedFace.loops must connect adjacent coedges at the same authored vertex; got first_vertex={vertex}, second_vertex={self.edge_vertices[coedge_edges[second]][start]}",
                        )
                    junctions.setdefault(vertex, []).append(
                        (face, first, end, second, start)
                    )
                    first_root = uses[first][end]
                    second_root = uses[second][start]
                    shares_source_root = (
                        first_root is not None
                        and second_root is not None
                        and endpoint_root_payload(first_root)
                        == endpoint_root_payload(second_root)
                    ) or (
                        # A self-closing whole native turn is one point.
                        first == second and closes_native_turn(first_root, second_root)
                    )
                    if not shares_source_root:
                        if any(
                            self.edge_curves[coedge_edges[coedge]] < 0
                            for coedge in (first, second)
                        ):
                            # A declared pole edge already owns this topology.
                            # Its bounded whole-edge correspondence is honest
                            # external-tolerance evidence; it is never promoted
                            # to a regular unique UV root at the chart collapse.
                            collapsed_vertices.add(vertex)
                        else:
                            needed.add(vertex)
        ambiguous = needed & collapsed_vertices
        if ambiguous:
            vertex = min(ambiguous)
            raise refuse(
                "inconsistent-geometry",
                f"vertex:{vertex}",
                f"CadStage.vertex_roots must not mix collapsed and regular incidence at one vertex; got conflicting_vertex={vertex}",
            )
        return _IncidencePlan(junctions=junctions, needed=tuple(sorted(needed)))

    def _prepare_incidence_vertex(
        self,
        junctions: list[tuple[int, int, int, int, int]],
        pcurves: Sequence[AbstractCurve],
        coedge_edges: Sequence[int],
        resolved: Sequence[
            tuple[list[list[tuple[StagedCoedge, AbstractCurve]]], np.ndarray]
        ],
    ) -> _IncidenceVertex:
        """Certify the joined source roots and their local and whole-edge lifts."""
        supports: list[BRepRootSupport] = []
        bindings: list[tuple[int, int, TrimRootEndpoint]] = []
        lifts: list[BRepCurveSurfaceLift] = []
        for face, first, end, second, start in junctions:
            patch = self.faces[face].patch
            source_region = SurfaceRegion(patch, resolved[face][1])
            operands = []
            for coedge, endpoint in ((first, end), (second, start)):
                edge = coedge_edges[coedge]
                parameter = self.edge_ranges[edge][endpoint]
                window = self._junction_window(patch, pcurves[coedge], parameter)
                carrier_index = self.edge_curves[edge]
                if carrier_index < 0:
                    raise ValueError(
                        f"CadStage.edge_curves must identify a carrier for a regular incidence lift; got edge={edge}, carrier_index={carrier_index}"
                    )
                carrier = self.curves[carrier_index]
                if carrier.parameter_domain is not None:
                    lower, upper = carrier.parameter_domain
                    window = CurveTrimSegment(
                        pcurves[coedge],
                        max(window.first, lower),
                        min(window.last, upper),
                    )
                operands.append(window)
                # Whole-edge and local incidence proofs use separate
                # charts; extending a full-period chart would exceed
                # the surface's authored one-period domain.
                lifts.append(
                    BRepCurveSurfaceLift(
                        carrier,
                        patch,
                        pcurves[coedge],
                        *self.edge_ranges[edge],
                        source_region=source_region,
                    )
                )
                lifts.append(
                    BRepCurveSurfaceLift(
                        carrier,
                        patch,
                        pcurves[coedge],
                        window.first,
                        window.last,
                        source_region=source_region,
                    )
                )
            root = TrimIntersectionRoot(
                operands[0],
                operands[1],
                parameter_lower=np.zeros(2, dtype=np.float64),
                parameter_upper=np.ones(2, dtype=np.float64),
            )
            supports.append(BRepRootSupport(patch, root))
            bindings.extend(
                (
                    (first, end, _window_root_endpoint(root, "first", operands[0])),
                    (second, start, _window_root_endpoint(root, "second", operands[1])),
                )
            )
        primary = next(
            (support for support in supports if isinstance(support.patch, PlanePatch)),
            supports[0],
        )
        definition = BRepVertexRoot(
            primary,
            aliases=tuple(support for support in supports if support is not primary),
            source_edge_lifts=tuple(lifts),
        )
        return _IncidenceVertex(definition=definition, bindings=tuple(bindings))

    def _exact_affine_junction(
        self,
        vertex: int,
        junctions: list[tuple[int, int, int, int, int]],
        pcurves: Sequence[AbstractCurve],
        coedge_edges: Sequence[int],
        resolved: Sequence[
            tuple[list[list[tuple[StagedCoedge, AbstractCurve]]], np.ndarray]
        ],
        /,
    ) -> bool:
        """Prove an authored vertex is exactly every incident affine carrier endpoint.

        The junction root's binary64 enclosure is outward rounded, so an exact
        source (zero declared tolerance) needs the algebraic proof instead: each
        incident carrier is an exact line whose endpoint is the authored vertex
        in rational arithmetic, and whose p-curve lift onto the face is exact.
        The authored vertex then is the junction root itself.
        """
        authored = tuple(
            Fraction.from_float(float(value)) for value in self.vertices[vertex]
        )
        for face, first, end, second, start in junctions:
            region = SurfaceRegion(self.faces[face].patch, resolved[face][1])
            for coedge, endpoint in ((first, end), (second, start)):
                edge = coedge_edges[coedge]
                carrier = self.curves[self.edge_curves[edge]]
                coefficients = _affine_coefficients(carrier)
                if coefficients is None:
                    return False
                lower, upper = self.edge_ranges[edge]
                if not exact_source_lift(carrier, pcurves[coedge], region, lower, upper):
                    return False
                parameter = Fraction.from_float(float(self.edge_ranges[edge][endpoint]))
                origin, direction = coefficients
                if any(
                    point + parameter * slope != value
                    for point, slope, value in zip(
                        origin, direction, authored, strict=True
                    )
                ):
                    return False
        return True

    def _bind_incidence_roots(
        self,
        pcurves: Sequence[AbstractCurve],
        coedge_edges: Sequence[int],
        coedge_senses: Sequence[int],
        face_loops: Sequence[tuple[tuple[int, ...], ...]],
        resolved: Sequence[
            tuple[list[list[tuple[StagedCoedge, AbstractCurve]]], np.ndarray]
        ],
        edge_roots: tuple[tuple[RootEndpoint | None, RootEndpoint | None], ...] | None,
        coedge_roots: tuple[tuple[RootEndpoint | None, RootEndpoint | None], ...] | None,
        /,
    ) -> tuple[
        tuple[BRepVertexRoot | None, ...],
        tuple[tuple[RootEndpoint | None, RootEndpoint | None], ...],
        tuple[tuple[RootEndpoint | None, RootEndpoint | None], ...],
    ]:
        """Certify nonalgebraic external joins and their spatial root aliases."""
        edges = list(edge_roots or ((None, None),) * len(self.edge_curves))
        uses = list(coedge_roots or ((None, None),) * len(coedge_edges))
        vertices: list[BRepVertexRoot | None] = [None] * len(self.vertices)
        plan = self._prepare_incidence_junctions(
            pcurves, coedge_edges, coedge_senses, face_loops, uses
        )
        for vertex in plan.needed:
            label = f"vertex:{vertex}"
            committed = False
            try:
                prepared = self._prepare_incidence_vertex(
                    plan.junctions[vertex], pcurves, coedge_edges, resolved
                )
                definition, bindings = prepared.definition, prepared.bindings
                committed = True
                for coedge, endpoint, root in bindings:
                    pair = list(uses[coedge])
                    pair[endpoint] = root
                    uses[coedge] = pair[0], pair[1]
                for edge, endpoints in enumerate(self.edge_vertices):
                    for endpoint, source_vertex in enumerate(endpoints):
                        if source_vertex != vertex:
                            continue
                        candidates = [
                            uses[coedge][endpoint]
                            for coedge, source_edge in enumerate(coedge_edges)
                            if source_edge == edge
                        ]
                        if not candidates or any(root is None for root in candidates):
                            raise ValueError(
                                f"BRepGeometry.coedge_endpoint_roots must bind every incident coedge; got edge={edge}, endpoint={endpoint}, bindings={candidates!r}"
                            )
                        intervals = [
                            root.parameter_enclosure()
                            for root in candidates
                            if root is not None
                        ]
                        lower, upper = (
                            max(a for a, _ in intervals),
                            min(b for _, b in intervals),
                        )
                        if lower > upper:
                            raise ValueError(
                                f"BRepGeometry.edge_endpoint_roots must have overlapping parameter enclosures; got edge={edge}, lower={lower}, upper={upper}"
                            )
                        pair = list(edges[edge])
                        pair[endpoint] = candidates[0]
                        edges[edge] = pair[0], pair[1]
                        parameters = list(self.edge_ranges[edge])
                        # Any representative inside the certified enclosure is
                        # sound; the authored parameter keeps the file's chart
                        # (exact polygon corners) when the root admits it.
                        if not lower <= parameters[endpoint] <= upper:
                            parameters[endpoint] = 0.5 * (lower + upper)
                        self.edge_ranges[edge] = parameters[0], parameters[1]
                box = definition.point_enclosure()
                displacement = np.nextafter(
                    box - self.vertices[vertex],
                    np.where(box < self.vertices[vertex], -np.inf, np.inf),
                )
                incidence_bound = norm_upper(np.max(np.abs(displacement), axis=0))
                if incidence_bound <= self.relative_geometric_tolerance * self.scale:
                    self.vertices[vertex] = np.mean(box, axis=0)
                elif not self._exact_affine_junction(
                    vertex, plan.junctions[vertex], pcurves, coedge_edges, resolved
                ):
                    raise ValueError(
                        f"BRepGeometry.vertex_roots must remain within the declared vertex incidence tolerance; got vertex={vertex}, bound={incidence_bound}, limit={self.relative_geometric_tolerance * self.scale}"
                    )
                vertices[vertex] = definition
            except ValueError as error:
                if not committed:
                    # A bounded external tolerance vertex remains a literal
                    # topology vertex unless every source-root isolation and correspondence condition
                    # succeeds. Never publish a partial or borrowed root.
                    continue
                raise refuse("inconsistent-geometry", label, str(error)) from error
        return tuple(vertices), tuple(edges), tuple(uses)

    def _close_literal_chart_junctions(
        self,
        pcurves: list[AbstractCurve],
        coedge_edges: Sequence[int],
        coedge_senses: Sequence[int],
        coedge_labels: Sequence[str],
        face_loops: Sequence[tuple[tuple[int, ...], ...]],
        coedge_roots: tuple[tuple[RootEndpoint | None, RootEndpoint | None], ...],
        /,
    ) -> tuple[_ChartClosureRepair, ...]:
        """Close binary64 chart misses at literal tolerance vertices.

        A junction without shared source roots keeps its authored vertex, and
        its two p-curve ends may miss each other by binary64 rounding. When the
        successor is a clamped B-spline whose junction end is its end pole,
        that pole moves onto the predecessor's exact chart endpoint so the trim
        closes exactly. The edge carrier is unchanged; the stage's face-gap
        proof then certifies the modified chart against it, so the move is
        admitted only within the declared import tolerance. Every move is
        returned so publication declares it as import repair evidence.
        """
        repairs: list[_ChartClosureRepair] = []
        for loops in face_loops:
            for loop in loops:
                for first, second in zip(loop, (*loop[1:], loop[0]), strict=True):
                    end = 1 if coedge_senses[first] > 0 else 0
                    start = 0 if coedge_senses[second] > 0 else 1
                    if (
                        coedge_roots[first][end] is not None
                        and coedge_roots[second][start] is not None
                    ):
                        continue
                    target = exact_curve_parameter_point(
                        pcurves[first],
                        Fraction.from_float(
                            float(self.edge_ranges[coedge_edges[first]][end])
                        ),
                    )
                    if target is None:
                        continue
                    closed = _clamped_pole_moved_to(
                        pcurves[second],
                        float(self.edge_ranges[coedge_edges[second]][start]),
                        target,
                    )
                    if closed is None:
                        continue
                    pcurves[second], pole, previous, moved = closed
                    repairs.append(
                        _ChartClosureRepair(
                            second, coedge_labels[second], pole, previous, moved
                        )
                    )
        return tuple(repairs)

    def publish(
        self,
        *,
        coordinate_contract: SpatialCoordinateContract,
        source_id: str,
        source_format: str,
        source_digest: str,
        import_policy_id: str,
        tessellation: BRepTessellationPolicy | None,
    ) -> BRepModel:
        boxes = [
            self.curves[index].bounding_box(*self.edge_ranges[edge])
            for edge, index in enumerate(self.edge_curves)
            if index >= 0
        ]
        if self.vertices:
            points = np.asarray(self.vertices)
            boxes.append(np.stack((np.min(points, axis=0), np.max(points, axis=0))))
        if boxes:
            box_array = np.asarray(boxes)
            self.scale = float(
                np.max(np.max(box_array[:, 1], axis=0) - np.min(box_array[:, 0], axis=0))
            )
        else:
            self.scale = 0.0
        if not isfinite(self.scale) or self.scale < 0.0:
            raise refuse(
                "inconsistent-geometry",
                source_id,
                f"CadStage.scale must be finite and positive for the physical source enclosure; got {self.scale!r}",
            )
        self._snap_sphere_meridian_ranges()
        self._split_periodic_bands()
        resolved = [self._resolve_face(face) for face in self.faces]
        pcurves: list[AbstractCurve] = []
        coedge_edges: list[int] = []
        coedge_senses: list[int] = []
        face_loops: list[tuple[tuple[int, ...], ...]] = []
        coedge_labels: list[str] = []
        for loops, _ in resolved:
            indexed = []
            for loop in loops:
                ids = []
                for coedge, pcurve in loop:
                    pcurves.append(pcurve)
                    coedge_edges.append(coedge.edge)
                    coedge_senses.append(coedge.sense)
                    coedge_labels.append(coedge.label)
                    ids.append(len(pcurves) - 1)
                indexed.append(tuple(ids))
            face_loops.append(tuple(indexed))
        self._bind_native_phase_surfaces(resolved)
        edge_roots, coedge_roots = self._native_phase_roots(
            pcurves, coedge_edges, face_loops
        )
        vertex_roots, edge_roots, coedge_roots = self._bind_incidence_roots(
            pcurves,
            coedge_edges,
            coedge_senses,
            face_loops,
            resolved,
            edge_roots,
            coedge_roots,
        )
        repairs = self._close_literal_chart_junctions(
            pcurves,
            coedge_edges,
            coedge_senses,
            coedge_labels,
            face_loops,
            coedge_roots,
        )
        vertex_points = np.asarray(self.vertices, dtype=np.float64).reshape(-1, 3)
        # A collapsed pole edge is proved against its authored vertex; the
        # binary file coordinates are that vertex's exact source value, except
        # at a sphere-chart pole, which the file prints through rounded trig.
        pole_vertices = {
            vertex
            for edge, curve in enumerate(self.edge_curves)
            if curve < 0
            for vertex in self.edge_vertices[edge]
        }
        sphere_poles = self._sphere_pole_points(pcurves, coedge_edges, face_loops)
        vertex_exact_points: list[tuple[Fraction, Fraction, Fraction] | None] = []
        for vertex, (x, y, z) in enumerate(vertex_points):
            pole = sphere_poles.get(vertex)
            if pole is not None:
                vertex_points[vertex] = tuple(float(value) for value in pole)
                vertex_exact_points.append(pole)
                continue
            vertex_exact_points.append(
                (
                    Fraction.from_float(float(x)),
                    Fraction.from_float(float(y)),
                    Fraction.from_float(float(z)),
                )
                if vertex in pole_vertices
                else None
            )
        geometry = BRepGeometry(
            vertex_points=vertex_points,
            vertex_exact_points=tuple(vertex_exact_points),
            curves=tuple(self.curves),
            edge_curves=tuple(self.edge_curves),
            edge_ranges=np.asarray(self.edge_ranges, dtype=np.float64).reshape(-1, 2),
            edge_vertices=tuple(self.edge_vertices),
            pcurves=tuple(pcurves),
            coedge_edges=tuple(coedge_edges),
            coedge_senses=tuple(coedge_senses),
            face_loops=tuple(face_loops),
            shell_faces=tuple(tuple(faces) for faces, _ in self.shells),
            shell_orientations=tuple(
                tuple(self.shell_orientations.get(index, [1] * len(faces)))
                for index, (faces, _) in enumerate(self.shells)
            ),
            solid_shells=tuple(tuple(shells) for shells in self.solids),
            assembly_graph=self.assembly_graph,
            vertex_roots=vertex_roots,
            edge_endpoint_roots=edge_roots,
            coedge_endpoint_roots=coedge_roots,
        )
        try:
            patches = tuple(face.patch for face in self.faces)
            parameter_bounds = np.asarray(
                [box for _, box in resolved], dtype=np.float64
            ).reshape(-1, 2, 2)
            tolerance = self.relative_geometric_tolerance * self.scale
            geometry = _bind_stage_face_gap_bounds(
                geometry, patches, parameter_bounds, tolerance
            )
            gap_bounds = geometry.edge_face_gap_bounds
            self.chart_closure_repairs = [
                repair.loss(
                    0.0 if gap_bounds is None else gap_bounds[repair.coedge], tolerance
                )
                for repair in repairs
            ]
            return assemble_brep_model(
                geometry,
                patches,
                parameter_bounds,
                np.asarray([face.orientation for face in self.faces], dtype=np.float64),
                tuple(face.tag for face in self.faces),
                coordinate_contract=coordinate_contract,
                source_id=source_id,
                source_format=source_format,
                source_digest=source_digest,
                import_policy_id=import_policy_id,
                converted_surface_count=0,
                tessellation=tessellation,
                curve_surface_tolerance=self.relative_geometric_tolerance * self.scale,
                curve_surface_scale=self.scale,
            )
        except ValueError as error:
            raise refuse(
                "inconsistent-geometry",
                source_id,
                str(error),
            ) from error

    def coverage(self, entity_counts: dict[str, int], /) -> CadCoverage:
        return CadCoverage(
            tuple(sorted(entity_counts.items())),
            self.pcurves_honored,
            self.pcurves_exact,
            self.degenerate_edges,
            self.natural_boundaries,
        )


def cad_import_stage_losses(stage: CadStage, /) -> tuple[AdapterLoss, ...]:
    """Report source semantics staging synthesized or repaired.

    Natural face boundaries are reconstructed from source surface domains;
    chart-closure repairs carry the moved pole and its certified face gap.
    """
    synthesized = (
        (
            AdapterLoss(
                "geometry.natural-boundaries",
                "import",
                "synthesized",
                f"{stage.natural_boundaries} natural closed-surface boundaries were derived from source surface domains.",
                changes_interpretation=False,
            ),
        )
        if stage.natural_boundaries
        else ()
    )
    return (*synthesized, *stage.chart_closure_repairs)


# ------------------------------------------------------ trim validation


def _validate_export_coedge_trim(
    trim: CurveTrimSegment,
    coedge: int,
    geometry: BRepGeometry,
    ranges: np.ndarray,
    format_name: str,
    chain: tuple[str, str],
) -> None:
    """Require a source trim segment to retain its coedge carrier and interval."""
    label = f"coedge:{coedge}"
    if not isinstance(trim, CurveTrimSegment):
        raise refuse(
            "inexact-export",
            label,
            f"TrimDomain.curve_loops must contain CurveTrimSegment values; got {type(trim).__name__}",
            chain,
        )
    edge = geometry.coedge_edges[coedge]
    carrier = trim.curve
    if not (
        isinstance(carrier, IntersectionPCurve)
        or (
            isinstance(carrier, AbstractCurve)
            and isinstance(explicit_carrier(carrier), _EXPLICIT_CARRIERS)
        )
    ):
        raise refuse(
            "inexact-export",
            label,
            f"CurveTrimSegment.curve must have a parameter-preserving carrier in format={format_name}; got carrier_type={type(carrier).__name__}",
            chain,
        )
    source = geometry.pcurves[coedge]
    if isinstance(carrier, IntersectionPCurve) or isinstance(source, IntersectionPCurve):
        same_carrier = (
            isinstance(carrier, IntersectionPCurve)
            and isinstance(source, IntersectionPCurve)
            and carrier.curve.branch_id == source.curve.branch_id
            and (carrier.side, carrier.first, carrier.last, carrier.reversed)
            == (source.side, source.first, source.last, source.reversed)
        )
    else:
        same_carrier = canonical_fingerprint(
            carrier_payload(carrier)
        ) == canonical_fingerprint(carrier_payload(source))
    if (
        trim.first != ranges[edge, 0]
        or trim.last != ranges[edge, 1]
        or trim.reversed != (geometry.coedge_senses[coedge] < 0)
        or not same_carrier
    ):
        raise refuse(
            "inexact-export",
            label,
            f"BRepGeometry.pcurves must match the corresponding trim carrier, chart and orientation; got mismatched coedge={coedge}",
            chain,
        )


@checked
def validate_export_trims(
    model: BRepModel,
    geometry: BRepGeometry,
    format_name: str,
    /,
    *,
    trim_domains: Sequence[TrimDomain | None] | None = None,
) -> None:
    """Require face trims to match the external codec's coedges."""
    bounds = np.asarray(model.parameter_bounds)
    ranges = np.asarray(geometry.edge_ranges)
    domains = model.trim_domains if trim_domains is None else trim_domains
    if len(domains) != len(geometry.face_loops):
        raise refuse(
            "inexact-export",
            "geometry",
            f"BRepModel.trim_domains must align with BRepGeometry.face_loops; got domains={len(domains)}, faces={len(geometry.face_loops)}",
        )
    for face, (domain, loops) in enumerate(
        zip(domains, geometry.face_loops, strict=True)
    ):
        if domain is None:
            raise refuse(
                "inexact-export",
                f"face:{face}",
                f"BRepModel.trim_domains must have bounded source closure for face={face}; got None",
                (f"face:{face}",),
            )
        if domain.curve_loops:
            if domain.outer is not None or domain.holes:
                raise refuse(
                    "inexact-export",
                    f"face:{face}",
                    "A source curve trim cannot also claim a polygonal surrogate.",
                    (f"face:{face}",),
                )
            if len(domain.curve_loops) != len(loops):
                raise refuse(
                    "inexact-export",
                    f"face:{face}",
                    f"TrimDomain.curve_loops must align with the face coedge loops; got trim_loops={len(domain.curve_loops)}, coedge_loops={len(loops)}",
                    (f"face:{face}",),
                )
            for loop_index, (segments, coedges) in enumerate(
                zip(domain.curve_loops, loops, strict=True)
            ):
                chain = (f"face:{face}", f"loop:{loop_index}")
                if len(segments) != len(coedges):
                    raise refuse(
                        "inexact-export",
                        f"face:{face}",
                        f"TrimDomain.curve_loops must retain one source segment per coedge; got segments={len(segments)}, coedges={len(coedges)}",
                        chain,
                    )
                for segment, coedge in zip(segments, coedges, strict=True):
                    if not isinstance(segment, CurveTrimSegment):
                        raise refuse(
                            "inexact-export",
                            f"face:{face}",
                            "TrimDomain.curve_loops must contain CurveTrimSegment source carriers.",
                            chain,
                        )
                    _validate_export_coedge_trim(
                        segment, coedge, geometry, ranges, format_name, chain
                    )
            continue
        if domain.outer is None or len(domain.holes) + 1 != len(loops):
            raise refuse(
                "inexact-export",
                f"face:{face}",
                f"TrimDomain polygon loops must align with the face coedge loops; got polygon_loops={len(domain.holes) + 1}, coedge_loops={len(loops)}",
                (f"face:{face}",),
            )
        polygon_loops = (domain.outer, *domain.holes)
        for loop_index, (vertices, coedges) in enumerate(
            zip(polygon_loops, loops, strict=True)
        ):
            chain = (f"face:{face}", f"loop:{loop_index}")
            if not all(
                isinstance(geometry.pcurves[coedge], (LineCurve, ExactLineCurve))
                for coedge in coedges
            ):
                raise refuse(
                    "inexact-export",
                    f"face:{face}",
                    "A polygon trim must have line-valued source p-curves.",
                    chain,
                )
            endpoints = []
            for coedge in coedges:
                edge = geometry.coedge_edges[coedge]
                parameter = ranges[edge, 0 if geometry.coedge_senses[coedge] > 0 else 1]
                point = np.asarray(
                    geometry.pcurves[coedge].evaluate(
                        jnp.asarray(parameter, dtype=jnp.float64)
                    ),
                    dtype=np.float64,
                )
                endpoints.append(
                    (point - bounds[face, 0]) / (bounds[face, 1] - bounds[face, 0])
                )
            polygon = np.asarray(endpoints)
            vertices_ = np.asarray(vertices, dtype=np.float64)
            if polygon.shape == vertices_.shape and any(
                np.max(np.abs(np.roll(polygon, offset, axis=0) - vertices_)) <= 1.0e-12
                for offset in range(len(endpoints))
            ):
                continue
            raise refuse(
                "inexact-export",
                f"face:{face}",
                f"TrimDomain polygon must match the oriented line coedge closure; got loop={loop_index}",
                chain,
            )


def _copy_geometry_with_approximation(
    geometry: BRepGeometry,
    curves: Sequence[AbstractCurve],
    pcurves: Sequence[AbstractCurve],
    fitted_edges: set[int],
    fitted_coedges: set[int],
    /,
) -> BRepGeometry:
    # A fitted carrier only interpolates its endpoint points: the vertices it
    # touches stop being exact source roots, so every root-supported endpoint
    # binding at them is released too. Native period turns need no vertex root.
    released = {
        vertex for edge in fitted_edges for vertex in geometry.edge_vertices[edge]
    }

    def release(
        edge: int, roots: tuple[RootEndpoint | None, RootEndpoint | None], /
    ) -> tuple[RootEndpoint | None, RootEndpoint | None]:
        if edge in fitted_edges:
            return None, None
        first, last = (
            None
            if geometry.edge_vertices[edge][endpoint] in released
            and not isinstance(root, NativePeriodEndpoint)
            else root
            for endpoint, root in enumerate(roots)
        )
        return first, last

    edge_roots = [
        release(edge, roots) for edge, roots in enumerate(geometry.edge_endpoint_roots)
    ]
    coedge_roots = [
        release(geometry.coedge_edges[coedge], roots)
        for coedge, roots in enumerate(geometry.coedge_endpoint_roots)
    ]
    return BRepGeometry(
        vertex_points=geometry.vertex_points,
        vertex_exact_points=geometry.vertex_exact_points,
        vertex_roots=tuple(
            None if vertex in released else root
            for vertex, root in enumerate(geometry.vertex_roots)
        ),
        edge_endpoint_roots=tuple(edge_roots),
        coedge_endpoint_roots=tuple(coedge_roots),
        edge_vertex_gap_bounds=geometry.edge_vertex_gap_bounds,
        edge_face_gap_bounds=None,
        curves=tuple(curves),
        edge_curves=geometry.edge_curves,
        edge_ranges=geometry.edge_ranges,
        edge_vertices=geometry.edge_vertices,
        pcurves=tuple(pcurves),
        coedge_edges=geometry.coedge_edges,
        coedge_senses=geometry.coedge_senses,
        face_loops=geometry.face_loops,
        shell_faces=geometry.shell_faces,
        shell_orientations=geometry.shell_orientations,
        solid_shells=geometry.solid_shells,
        assembly_graph=geometry.assembly_graph,
    )


def _fit_intersection_edges(
    model: BRepModel,
    geometry: BRepGeometry,
    policy: CadCurveFitPolicy,
    format_name: str,
    /,
) -> tuple[BRepGeometry, tuple[AdapterLoss, ...]]:
    from ..geometry.brep._curve_approximation_bounds import (
        ApproximationCertificationError,
        certify_coupled_approximation,
        propose_intersection_approximation,
    )
    from ..geometry.brep._curve_approximation_topology import (
        BranchTopologyCertificationError,
        certify_branch_approximation_topology,
        certify_branch_trim_separation,
    )

    curves = list(geometry.curves)
    pcurves = list(geometry.pcurves)
    replacements: dict[int, tuple[BSplineCurve, BSplineCurve, BSplineCurve]] = {}
    losses: list[AdapterLoss] = []
    fitted_edges: set[int] = set()
    fitted_coedges: set[int] = set()
    ranges = np.asarray(geometry.edge_ranges, dtype=np.float64)
    vertices = geometry.edge_vertices
    for edge, curve_index in enumerate(geometry.edge_curves):
        if curve_index < 0:
            continue
        source = geometry.curves[curve_index]
        if not isinstance(source, IntersectionCurve):
            continue
        if not source.fully_certified:
            raise refuse(
                "inexact-export",
                f"intersection-curve {source.branch_id}",
                f"{format_name} requires a fully certified source branch before fitting.",
                (f"edge:{edge}",),
            )
        fitted = replacements.get(curve_index)
        if fitted is None:
            fitted = propose_intersection_approximation(
                source,
                endpoint_points=(
                    np.asarray(
                        geometry.vertex_points[vertices[edge][0]], dtype=np.float64
                    ),
                    np.asarray(
                        geometry.vertex_points[vertices[edge][1]], dtype=np.float64
                    ),
                ),
            )
            try:
                bounds = certify_coupled_approximation(
                    source,
                    *fitted,
                    distance_tolerance=policy.tolerance,
                    parameter_tolerance=policy.resolved_parameter_tolerance,
                )
                topology = certify_branch_approximation_topology(source, *fitted, bounds)
            except (
                ApproximationCertificationError,
                BranchTopologyCertificationError,
            ) as error:
                raise refuse(
                    "inexact-export",
                    f"intersection-curve {source.branch_id}",
                    f"{format_name} continuous intersection approximation was not certified: {error}",
                    (f"edge:{edge}",),
                ) from error
            replacements[curve_index] = fitted
            curves[curve_index] = fitted[0]
        else:
            bounds = certify_coupled_approximation(
                source,
                *fitted,
                distance_tolerance=policy.tolerance,
                parameter_tolerance=policy.resolved_parameter_tolerance,
            )
            topology = certify_branch_approximation_topology(source, *fitted, bounds)
        edge_coedges = tuple(
            coedge
            for loops in geometry.face_loops
            for loop in loops
            for coedge in loop
            if geometry.coedge_edges[coedge] == edge
        )
        if not edge_coedges:
            raise refuse(
                "inexact-export",
                f"edge:{edge}",
                f"{format_name} requires both coupled source p-curves for an intersection branch.",
            )
        pcurve_sides: dict[int, IntersectionSide] = {}
        for coedge in edge_coedges:
            source_pcurve = geometry.pcurves[coedge]
            if (
                not isinstance(source_pcurve, IntersectionPCurve)
                or source_pcurve.curve.branch_id != source.branch_id
                or source_pcurve.reversed
            ):
                raise refuse(
                    "inexact-export",
                    f"edge:{edge}",
                    f"{format_name} approximation requires forward coupled source p-curves.",
                )
            side = source_pcurve.side
            pcurves[coedge] = fitted[1] if side == "first" else fitted[2]
            pcurve_sides[coedge] = side
        fitted_edges.add(edge)
        fitted_coedges.update(edge_coedges)
        if set(pcurve_sides.values()) != {"first", "second"}:
            raise refuse(
                "inexact-export",
                f"edge:{edge}",
                f"{format_name} approximation requires both source-surface p-curves.",
            )
        for face in geometry.topology.edge_faces[edge]:
            branch_coedges = tuple(
                coedge
                for loop in geometry.face_loops[face]
                for coedge in loop
                if geometry.coedge_edges[coedge] == edge
            )
            for branch_coedge in branch_coedges:
                side = pcurve_sides[branch_coedge]
                for loop in geometry.face_loops[face]:
                    for trim_coedge in loop:
                        trim_edge = geometry.coedge_edges[trim_coedge]
                        if trim_edge == edge or set(vertices[edge]) & set(
                            vertices[trim_edge]
                        ):
                            continue
                        trim = geometry.pcurves[trim_coedge]
                        trim_first, trim_last = ranges[trim_edge]
                        try:
                            certify_branch_trim_separation(
                                topology,
                                trim,
                                side,
                                first=float(trim_first),
                                last=float(trim_last),
                            )
                        except BranchTopologyCertificationError as error:
                            raise refuse(
                                "inexact-export",
                                f"edge:{edge}",
                                f"{format_name} intersection approximation lacks trim-separation evidence: {error}",
                                (f"edge:{edge}", f"edge:{trim_edge}", f"face:{face}"),
                            ) from error
        losses.append(
            AdapterLoss(
                f"edge:{edge}.intersection-curve",
                "export",
                "transformed",
                f"{format_name} substituted a fitted rational B-spline after continuous coupled error "
                f"and branch topology certification (distance={bounds.distance_bound:.17g}, "
                f"parameter={bounds.parameter_bound:.17g}, topology={topology.evidence_id}).",
                changes_interpretation=True,
            )
        )
    if not losses:
        return geometry, ()
    return (
        _copy_geometry_with_approximation(
            geometry, curves, pcurves, fitted_edges, fitted_coedges
        ),
        tuple(losses),
    )


@checked
def prepare_cad_export_geometry(
    model: BRepModel,
    geometry: BRepGeometry,
    policy: CadExportPolicy,
    /,
    *,
    format_name: str = "native CAD",
) -> tuple[BRepGeometry, tuple[TrimDomain | None, ...], tuple[AdapterLoss, ...]]:
    """Validate and, when requested, continuously certify fitted branch carriers."""
    if model.geometry is None or model.geometry is not geometry:
        raise refuse(
            "inexact-export",
            model.source_id,
            "CAD export must use the exact geometry owned by its source model.",
        )
    validate_export_trims(model, geometry, format_name)
    has_intersections = any(
        curve_index >= 0 and isinstance(geometry.curves[curve_index], IntersectionCurve)
        for curve_index in geometry.edge_curves
    )
    if not has_intersections:
        return geometry, model.trim_domains, ()
    if policy.intersection_approximation is None:
        raise refuse(
            "inexact-export",
            model.source_id,
            f"{format_name} export has no exact carrier for a native intersection branch and no approximation was requested.",
        )
    prepared, losses = _fit_intersection_edges(
        model, geometry, policy.intersection_approximation, format_name
    )
    return prepared, model.trim_domains, losses


def _tspline_surface_carrier(
    patch: AbstractSurfacePatch, label: str, format_name: str, /
) -> tuple[AbstractSurfacePatch | None, AdapterLoss | None]:
    from ..geometry.brep._tspline import TSplineSurfacePatch

    if isinstance(patch, OffsetSurface):
        base, loss = _tspline_surface_carrier(patch.base, label, format_name)
        if base is None:
            return None, None
        return OffsetSurface(
            base, patch.distance_exact, frame_sign=patch.frame_sign
        ), loss
    if not isinstance(patch, TSplineSurfacePatch):
        return None, None
    patch.validate_exact_transfer_chain()
    extractions = patch.realization.extractions
    if (
        len(extractions) != 1
        or extractions[0].parameter_bounds != patch.mesh.parameter_domain
    ):
        raise refuse(
            "unsupported-entity",
            label,
            f"{format_name} cannot represent this multi-piece T-spline face as one exact surface; no approximation is performed.",
        )
    extraction = patch.realization.extraction_piece(extractions[0].piece_id)
    homogeneous_controls = patch.realization.exact_homogeneous_piece(
        extraction.piece_id, patch.exact_target_homogeneous
    )
    if len(homogeneous_controls) != 16 or any(
        len(row) != 4 for row in homogeneous_controls
    ):
        raise CadCodecInvariantError(
            f"{label} T-spline exact extraction is not one 4-by-4 homogeneous net."
        )
    exact_weights = tuple(value[3] for value in homogeneous_controls)
    if any(value <= 0 for value in exact_weights):
        raise refuse(
            "unsupported-entity",
            label,
            f"{format_name} cannot represent a T-spline extraction with nonpositive rational surface weights.",
        )
    exact_points = tuple(
        tuple(
            tuple(
                homogeneous_controls[4 * u + v][axis] / exact_weights[4 * u + v]
                for axis in range(3)
            )
            for v in range(4)
        )
        for u in range(4)
    )
    exact_weights_2d = tuple(
        tuple(exact_weights[4 * u + v] for v in range(4)) for u in range(4)
    )
    u0, u1, v0, v1 = extraction.parameter_bounds
    exact_u_knots = (u0,) * 4 + (u1,) * 4
    exact_v_knots = (v0,) * 4 + (v1,) * 4
    points = np.asarray(
        [[[float(value) for value in point] for point in row] for row in exact_points],
        dtype=np.float64,
    )
    weights = np.asarray(
        [[float(value) for value in row] for row in exact_weights_2d],
        dtype=np.float64,
    )
    if (
        not np.all(np.isfinite(points))
        or not np.all(np.isfinite(weights))
        or np.any(weights <= 0.0)
    ):
        raise refuse(
            "unsupported-entity",
            label,
            f"{format_name} cannot represent the exact T-spline extraction in finite binary64 surface coefficients.",
        )
    carrier = BSplineSurfacePatch(
        points,
        weights,
        np.asarray([float(value) for value in exact_u_knots]),
        np.asarray([float(value) for value in exact_v_knots]),
        3,
        3,
        exact_control_points=exact_points,
        exact_weights=exact_weights_2d,
        exact_u_knots=exact_u_knots,
        exact_v_knots=exact_v_knots,
    )
    loss = AdapterLoss(
        f"{label}.tspline-refinement",
        "export",
        "transformed",
        f"{format_name} represents the exact single-piece T-spline geometry as a rational B-spline surface; the source T-mesh refinement and transfer structure are not encoded.",
        changes_interpretation=True,
    )
    return carrier, loss


def prepare_cad_export_surfaces(
    patches: Sequence[AbstractSurfacePatch], format_name: str, /
) -> tuple[tuple[AbstractSurfacePatch, ...], tuple[AdapterLoss, ...]]:
    """Prepare exact external surface carriers and report T-spline structure loss."""
    carriers: list[AbstractSurfacePatch] = []
    losses: list[AdapterLoss] = []
    for face, patch in enumerate(patches):
        carrier, loss = _tspline_surface_carrier(patch, f"face:{face}", format_name)
        carriers.append(patch if carrier is None else carrier)
        if loss is not None:
            losses.append(loss)
    return tuple(carriers), tuple(losses)


def _exact_affine_plane_surface(
    patch: AbstractSurfacePatch, /
) -> PlanePatch | OffsetSurface | None:
    if isinstance(patch, OffsetSurface):
        base = _exact_affine_plane_surface(patch.base)
        if base is None:
            return None
        return OffsetSurface(base, patch.distance_exact, frame_sign=patch.frame_sign)
    if not isinstance(patch, BSplineSurfacePatch):
        return None
    degree_u, degree_v = patch.u_degree, patch.v_degree
    count_u, count_v = degree_u + 1, degree_v + 1
    if (
        degree_u < 1
        or degree_v < 1
        or len(patch.exact_control_points) != count_u
        or any(len(row) != count_v for row in patch.exact_control_points)
        or len(patch.exact_weights) != count_u
        or any(len(row) != count_v for row in patch.exact_weights)
        or len(patch.exact_u_knots) != 2 * count_u
        or len(patch.exact_v_knots) != 2 * count_v
    ):
        return None
    try:
        points = tuple(
            tuple(
                tuple(
                    rational_coefficient(value, role="exported B-spline control point")
                    for value in point
                )
                for point in row
            )
            for row in patch.exact_control_points
        )
        weights = tuple(
            tuple(
                rational_coefficient(value, role="exported B-spline weight")
                for value in row
            )
            for row in patch.exact_weights
        )
    except ValueError:
        return None
    first_weight = weights[0][0]
    if any(value != first_weight for row in weights for value in row):
        return None
    u0, u1 = patch.exact_u_knots[degree_u], patch.exact_u_knots[count_u]
    v0, v1 = patch.exact_v_knots[degree_v], patch.exact_v_knots[count_v]
    if (
        u0 >= u1
        or v0 >= v1
        or patch.exact_u_knots[:count_u] != (u0,) * count_u
        or patch.exact_u_knots[count_u:] != (u1,) * count_u
        or patch.exact_v_knots[:count_v] != (v0,) * count_v
        or patch.exact_v_knots[count_v:] != (v1,) * count_v
    ):
        return None
    corner = points[0][0]
    first_axis = tuple(
        (points[-1][0][axis] - corner[axis]) / (u1 - u0) for axis in range(3)
    )
    second_axis = tuple(
        (points[0][-1][axis] - corner[axis]) / (v1 - v0) for axis in range(3)
    )
    origin = tuple(
        corner[axis] - u0 * first_axis[axis] - v0 * second_axis[axis] for axis in range(3)
    )
    if any(
        points[u][v][axis]
        != origin[axis]
        + (u0 + (u1 - u0) * Fraction(u, degree_u)) * first_axis[axis]
        + (v0 + (v1 - v0) * Fraction(v, degree_v)) * second_axis[axis]
        for u in range(count_u)
        for v in range(count_v)
        for axis in range(3)
    ):
        return None
    exact_frame = (*origin, *first_axis, *second_axis)
    numeric_frame = tuple(float(value) for value in exact_frame)
    if any(
        not isfinite(value) or Fraction.from_float(value) != exact
        for value, exact in zip(numeric_frame, exact_frame, strict=True)
    ):
        return None
    return PlanePatch(
        np.asarray(numeric_frame[:3], dtype=np.float64),
        np.asarray(numeric_frame[3:6], dtype=np.float64),
        np.asarray(numeric_frame[6:], dtype=np.float64),
    )


def _cad_export_gap_surfaces(
    patches: Sequence[AbstractSurfacePatch], /
) -> tuple[AbstractSurfacePatch, ...]:
    """Use only exact affine-plane equivalents to certify exported surface gaps."""
    return tuple(
        exact_plane
        if (exact_plane := _exact_affine_plane_surface(patch)) is not None
        else patch
        for patch in patches
    )


@checked
def cad_export_physical_tag_losses(
    model: BRepModel,
    target_patches: Sequence[AbstractSurfacePatch],
    format_name: str,
    /,
) -> tuple[AdapterLoss, ...]:
    """Report source physical-family tags that an external carrier cannot retain."""
    if len(target_patches) != len(model.physical_tags):
        raise CadCodecInvariantError(
            "Physical-tag loss accounting requires one target carrier per source face."
        )
    from ._cad_carriers import surface_tag

    return tuple(
        AdapterLoss(
            f"face:{face}.physical-tag",
            "export",
            "transformed",
            f"{format_name} does not encode BRepModel.physical_tags; the source tag {source_tag!r} is not embedded and re-import derives the carrier-family tag {surface_tag(patch)!r}.",
            changes_interpretation=True,
        )
        for face, (source_tag, patch) in enumerate(
            zip(model.physical_tags, target_patches, strict=True)
        )
        if source_tag != surface_tag(patch)
    )


@checked
def cad_export_assembly_identifier_losses(
    graph: BRepAssemblyGraph | None, format_name: str, /
) -> tuple[AdapterLoss, ...]:
    """Report format-level renumbering of native assembly definition/use IDs."""
    if graph is None:
        return ()
    references = len(graph.roots) + sum(len(item.uses) for item in graph.definitions)
    identifiers = references + len(graph.definitions)
    if identifiers == 0:
        return ()
    return (
        AdapterLoss(
            "geometry.assembly-source-identifiers",
            "export",
            "transformed",
            f"{format_name} regenerates external references for {identifiers} native assembly definition/use identifiers; exact assembly sharing and placements remain represented, but the original source identifiers are not embedded in the file.",
            changes_interpretation=True,
        ),
    )


__all__ = [
    "CadCodecInvariantError",
    "CadExportTolerances",
    "CadImportToleranceEvidence",
    "build_cad_import_tolerance_evidence",
    "cad_export_assembly_identifier_losses",
    "cad_export_physical_tag_losses",
    "prepare_cad_export_surfaces",
    "build_cad_export_report",
    "certified_cad_export_tolerances",
    "CadCoverage",
    "CadCurveFitPolicy",
    "CadExportPolicy",
    "CadExportResult",
    "CadImportPolicy",
    "CadImportResult",
    "CadInterchangeError",
    "CadRefusal",
    "CadRefusalReason",
]
