#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Shared native CAD interchange policies, refusals, coverage and definition-preserving staging.

Format readers (STEP, IGES, external BRep text) decode their entity graphs into
a `CadStage`: source vertices, 3D carriers, oriented edges, coedges with p-curves
and trimmed faces in the target coordinate contract. The stage owns the
format-independent invariants every reader shares: vertex inversion onto edge
carriers, algebraic p-curve recovery for analytic surfaces (with bounded fitting and
evidence otherwise), periodic seam representatives, pole (degenerate) edges and
face parameter boxes. Any geometry the stage cannot represent exactly or within
the explicit fitting policy raises a `CadInterchangeError` naming the external
dependency chain.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from fractions import Fraction
from math import atan2, isfinite, pi
from typing import assert_never, Literal, Never, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from .._external_resource import ResourceLimits, ResourceManifest, ResourceReadError
from .._fingerprint import array_bytes_digest, canonical_fingerprint
from .._physical import SpatialCoordinateContract
from .._publication import PublicationReceipt
from .._strict import StrictModule
from .._validation import finite_real_scalar, positive_finite_float, positive_integer
from ..geometry._atlas import (
    AbstractTrimCurve,
    CurveTrimLoop,
    DynamicCurveTrimLoop,
    PolygonTrimLoop,
    TrimDomain,
)
from ..geometry._interval_enclosure import prepare_interval_function
from ..geometry.brep._constructors import (
    assemble_brep_model,
    brep_trim_domain,
    BRepTessellationPolicy,
    NormalizedTrimCurve,
)
from ..geometry.brep._coordinate_request import coordinate_source_request
from ..geometry.brep._correspondence import (
    exact_parameter_parts,
    ExactParameter,
)
from ..geometry.brep._curve_approximation_bounds import (
    ApproximationDeviationError,
    ApproximationResourceError,
    certify_coupled_approximation,
    CoupledApproximationEvidence,
)
from ..geometry.brep._curve_approximation_topology import (
    BranchApproximationTopologyEvidence,
    BranchApproximationTopologyResourceError,
    BranchTrimHomotopyTube,
    BranchTrimSeparationEvidence,
    certify_branch_approximation_topology,
    certify_branch_trim_separation,
)
from ..geometry.brep._gap_evidence import BRepGapEvidence, certify_brep_gaps
from ..geometry.brep._intersection import (
    NativePeriodEndpoint,
    RootEndpoint,
    TrimIntersectionRoot,
    TrimRootEndpoint,
)
from ..geometry.brep._intersection_curve import (
    affine_curve_coefficients,
    AffinePCurve,
    CurveTrimSegment,
    encode_geometry,
    exact_curve_parameter_point,
    exact_line_pcurve,
    IntersectionCurve,
    IntersectionPCurve,
    PeriodicPCurve,
    SurfaceRegion,
)
from ..geometry.brep._model import (
    brep_physical_scale,
    BRepAssemblyContainer,
    BRepEntityId,
    BRepGeometry,
    BRepModel,
    BRepOccurrence,
    BRepValidationError,
)
from ..geometry.brep._patches import (
    AbstractCurve,
    AbstractSurfacePatch,
    BSplineCurve,
    BSplineSurfacePatch,
    CircleCurve,
    ConePatch,
    curve_evaluate_host,
    curve_value_and_derivative_host,
    CylinderPatch,
    EllipseCurve,
    ExtrusionSurface,
    HyperbolaCurve,
    LineCurve,
    OffsetCurve,
    OffsetSurface,
    ParabolaCurve,
    PlanePatch,
    prepare_surface_host,
    RevolutionSurface,
    RuledSurface,
    SpherePatch,
    surface_evaluate_host,
    surface_value_and_differential_host,
    SurfaceIsoparametricCurve,
    TorusPatch,
)
from ..geometry.brep._placed import PlacedSurface
from ..geometry.brep._root_bindings import (
    BRepCurveSurfaceLift,
    BRepRootSupport,
    BRepVertexRoot,
)
from ..geometry.brep._source_correspondence import norm_upper
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
    """Explicit proposal controls and continuous export-proof resource bounds.

    ``tolerance`` is relative to the physical source extent for 3D curves and
    to the native chart extent for p-curves. ``maximum_control_points`` and
    ``check_samples`` bound cubic least-squares proposals; samples never certify
    an exported intersection. Export acceptance additionally requires continuous
    coupled error, correspondence and tube/trim topology evidence within the
    certificate cell, depth and byte budgets. Exhaustion is a refusal, not a
    tolerance relaxation. Imported fitted p-curves still undergo the
    whole-edge curve/surface correspondence certificate before publication.
    """

    tolerance: float = 1.0e-10
    maximum_control_points: int = 256
    check_samples: int = 513
    maximum_certificate_cells: int = 65536
    maximum_certificate_depth: int = 32
    maximum_certificate_bytes: int = 1 << 26
    policy_id: str = field(init=False)

    def __post_init__(self) -> None:
        tolerance = positive_finite_float(self.tolerance, "CadCurveFitPolicy.tolerance")
        maximum = positive_integer(
            self.maximum_control_points, "CadCurveFitPolicy.maximum_control_points"
        )
        samples = positive_integer(self.check_samples, "CadCurveFitPolicy.check_samples")
        cells = positive_integer(
            self.maximum_certificate_cells, "CadCurveFitPolicy.maximum_certificate_cells"
        )
        depth = positive_integer(
            self.maximum_certificate_depth, "CadCurveFitPolicy.maximum_certificate_depth"
        )
        proof_bytes = positive_integer(
            self.maximum_certificate_bytes, "CadCurveFitPolicy.maximum_certificate_bytes"
        )
        if maximum < 4:
            raise ValueError(
                f"CadCurveFitPolicy.maximum_control_points must be at least 4; got {maximum}"
            )
        if samples < 2 * maximum:
            raise ValueError(
                f"CadCurveFitPolicy.check_samples must be at least twice maximum_control_points={maximum}; got {samples}"
            )
        object.__setattr__(self, "tolerance", tolerance)
        object.__setattr__(self, "maximum_control_points", maximum)
        object.__setattr__(self, "check_samples", samples)
        object.__setattr__(self, "maximum_certificate_cells", cells)
        object.__setattr__(self, "maximum_certificate_depth", depth)
        object.__setattr__(self, "maximum_certificate_bytes", proof_bytes)
        object.__setattr__(
            self,
            "policy_id",
            canonical_fingerprint(
                {
                    "kind": "cad-curve-fit-policy",
                    "tolerance": tolerance,
                    "maximum_control_points": maximum,
                    "check_samples": samples,
                    "maximum_certificate_cells": cells,
                    "maximum_certificate_depth": depth,
                    "maximum_certificate_bytes": proof_bytes,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class CadImportPolicy:
    """Target coordinates, resource bounds and derived-tessellation policy.

    ``limits.max_bytes`` bounds the file, ``limits.max_nodes`` the entity
    instances, ``limits.max_attributes`` the total parameters, and
    ``limits.max_depth`` both the list nesting and the reference depth of the
    entity graph. ``maximum_occurrences`` bounds expanded assembly
    occurrences. ``pcurve_fit`` explicitly permits sampled-error approximations
    when algebraic recovery is unavailable; ``None`` admits definition-preserving p-curves only.
    ``relative_geometric_tolerance`` is a dimensionless whole-edge
    correspondence allowance, converted to target length coordinates using
    the declared translation-invariant physical model scale. It is separate
    from the finite-sample residual requested by ``pcurve_fit``.
    ``maximum_placement_deviation`` is a certified point-displacement limit in
    target length coordinates; zero preserves zero-defect STEP placement admission.
    """

    coordinate_contract: SpatialCoordinateContract
    limits: ResourceLimits
    tessellation: BRepTessellationPolicy | None = None
    pcurve_fit: CadCurveFitPolicy | None = None
    maximum_occurrences: int = 4096
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
        if not isinstance(self.limits, ResourceLimits):
            raise TypeError(
                f"CadImportPolicy.limits must be ResourceLimits; got {type(self.limits).__name__}"
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
        occurrences = positive_integer(
            self.maximum_occurrences, "CadImportPolicy.maximum_occurrences"
        )
        object.__setattr__(self, "maximum_occurrences", occurrences)
        object.__setattr__(self, "relative_geometric_tolerance", tolerance)
        tessellation = (
            BRepTessellationPolicy() if self.tessellation is None else self.tessellation
        )
        object.__setattr__(
            self,
            "policy_id",
            canonical_fingerprint(
                {
                    "kind": "cad-import-policy",
                    "spatial_id": self.coordinate_contract.spatial_id,
                    "limits": [
                        self.limits.max_bytes,
                        self.limits.max_depth,
                        self.limits.max_nodes,
                        self.limits.max_attributes,
                        self.limits.max_losses,
                    ],
                    "tessellation": tessellation.policy_id,
                    "pcurve_fit": None
                    if self.pcurve_fit is None
                    else self.pcurve_fit.policy_id,
                    "maximum_occurrences": occurrences,
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
    """Output bounds and the optional explicit intersection-curve approximation.

    Without ``intersection_approximation`` an unrepresentable native branch is
    refused, never tessellated into CAD. With it, coupled B-spline proposals
    require continuous native-interval deviation/surface-lift bounds and
    topology-preserving straight-homotopy/used-trim evidence. Certificates
    retain interval arithmetic assumptions and resource limits;
    publication remains declared approximation, not lossless native archive.
    """

    intersection_approximation: CadCurveFitPolicy | None = None
    maximum_bytes: int = 1 << 30
    maximum_entities: int = 10_000_000
    policy_id: str = field(init=False)

    def __post_init__(self) -> None:
        if self.intersection_approximation is not None and not isinstance(
            self.intersection_approximation, CadCurveFitPolicy
        ):
            raise TypeError(
                f"CadExportPolicy.intersection_approximation must be CadCurveFitPolicy or None; got {type(self.intersection_approximation).__name__}"
            )
        maximum_bytes = positive_integer(
            self.maximum_bytes, "CadExportPolicy.maximum_bytes"
        )
        maximum_entities = positive_integer(
            self.maximum_entities, "CadExportPolicy.maximum_entities"
        )
        object.__setattr__(self, "maximum_bytes", maximum_bytes)
        object.__setattr__(self, "maximum_entities", maximum_entities)
        object.__setattr__(
            self,
            "policy_id",
            canonical_fingerprint(
                {
                    "kind": "cad-export-policy",
                    "intersection_approximation": None
                    if self.intersection_approximation is None
                    else self.intersection_approximation.policy_id,
                    "maximum_bytes": maximum_bytes,
                    "maximum_entities": maximum_entities,
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
    resource_manifest: ResourceManifest | None

    def __init__(
        self, refusal: CadRefusal, /, *, resource_manifest: ResourceManifest | None = None
    ) -> None:
        self.refusal = refusal
        self.resource_manifest = resource_manifest
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


@checked
def resource_refusal(error: ResourceReadError, /) -> Never:
    """Re-raise a bounded-resource admission failure as a typed CAD refusal."""
    reason: CadRefusalReason = "limit" if error.reason == "limit" else "malformed"
    raise refuse(reason, "resource", str(error)) from error


@dataclass(frozen=True, slots=True)
class CadCoverage:
    """Admitted external entity types and how every p-curve was obtained.

    ``pcurves_honored`` came from the file; ``pcurves_exact`` were recovered by
    algebraic inversion; ``pcurves_fitted`` are explicitly requested fits whose largest
    sampled relative residual is ``maximum_fit_error`` (not a continuous bound).
    ``degenerate_edges`` and ``natural_boundaries`` count synthesized pole edges
    and closed-surface boundaries absent from the external representation.
    """

    entity_counts: tuple[tuple[str, int], ...]
    pcurves_honored: int
    pcurves_exact: int
    pcurves_fitted: int
    maximum_fit_error: float
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
            ("pcurves_fitted", self.pcurves_fitted),
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
        if (
            isinstance(self.maximum_fit_error, (bool, np.bool_))
            or not isfinite(self.maximum_fit_error)
            or self.maximum_fit_error < 0.0
        ):
            raise ValueError(
                f"CadCoverage.maximum_fit_error must be finite and nonnegative; got {self.maximum_fit_error!r}"
            )
        if self.pcurves_fitted == 0 and self.maximum_fit_error != 0.0:
            raise ValueError(
                f"CadCoverage.maximum_fit_error must be zero without fitted p-curves; got {self.maximum_fit_error!r}"
            )


def _gap_entity_bounds(evidence: BRepGapEvidence) -> dict[BRepEntityId, float]:
    bounds: dict[BRepEntityId, float] = {}
    for entity, partner, value in zip(
        evidence.entities, evidence.partners, evidence.upper_bounds, strict=True
    ):
        for item in (entity, partner):
            bounds[item] = max(bounds.get(item, 0.0), float(value))
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
        gaps: BRepGapEvidence,
        declared: Sequence[tuple[BRepEntityId | None, float]],
        /,
        *,
        unrepresentable_losses: Sequence[AdapterLoss] = (),
    ) -> None:
        if gaps.model_id != model.model_id:
            raise ValueError(
                f"CadImportToleranceEvidence.model_id must match native gap evidence; got model={model.model_id!r}, gaps={gaps.model_id!r}"
            )
        entities = tuple(sorted((*model.vertex_ids, *model.edge_ids, *model.face_ids)))
        prepared = _prepare_declared_tolerances(
            entities, declared, model.report.curve_surface_tolerance
        )
        certified = _gap_entity_bounds(gaps)
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
    """Carry source tolerances beside certified gaps under the model declared limit."""
    return CadImportToleranceEvidence(
        model,
        certify_brep_gaps(model),
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
    """Return native B-Rep geometry and retained source metadata.

    ``source_tolerance`` is a target-coordinate-unit summary. ``source_tolerances``
    retains every declared source tolerance, including unused declarations, by
    external field reference in raw source-length units.
    ``tolerance_evidence`` records certified bounds, declarations and their carried
    envelopes separately under the model's declared physical tolerance limit.
    """

    model: BRepModel
    format: str
    schema: str
    source_digest: str
    source_length_unit_meters: float
    source_angle_unit_radians: float
    resource_manifest: ResourceManifest
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
        if self.source_digest != self.resource_manifest.content_sha256:
            raise ValueError(
                f"CadImportResult.source_digest must equal resource_manifest.content_sha256; got {self.source_digest!r}, expected={self.resource_manifest.content_sha256!r}"
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
                    "resource_manifest": self.resource_manifest.manifest_id,
                    "report": self.report.report_id,
                    "coverage": {
                        "entity_counts": [
                            list(pair) for pair in self.coverage.entity_counts
                        ],
                        "pcurves_honored": self.coverage.pcurves_honored,
                        "pcurves_exact": self.coverage.pcurves_exact,
                        "pcurves_fitted": self.coverage.pcurves_fitted,
                        "maximum_fit_error": self.coverage.maximum_fit_error,
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
class CadCurveApproximation:
    """Explicitly bounded coupled substitution of a source native branch.

    ``continuous_evidence`` bounds whole-domain 3D/UV deviation and both native
    surface lifts. ``topology_evidence`` certifies the source/fit straight
    homotopy and identifies open native-period lifts without claiming quotient
    closure. Sampled residuals are proposal diagnostics only. Source
    roots and enclosures remain attached to the substitution; ``approximation_id`` identifies the new
    emitted carrier definition, never the source branch definition.
    """

    branch_id: str
    curve: BSplineCurve
    first_pcurve: BSplineCurve
    second_pcurve: BSplineCurve
    sampled_distance_error: float
    sampled_parameter_error: float
    samples: int
    policy_id: str
    distance_scale: float
    parameter_scale: float
    continuous_evidence: CoupledApproximationEvidence
    topology_evidence: BranchApproximationTopologyEvidence
    certification_cells: int
    continuous_jet_evaluations: int
    certification_peak_bytes: int
    approximation_id: str = field(init=False)
    trim_separation_evidence: tuple[BranchTrimSeparationEvidence, ...] = ()
    source_geometry_id: str | None = None
    export_geometry_id: str | None = None
    source_vertex_roots: tuple[tuple[int, BRepVertexRoot], ...] = ()
    source_vertex_enclosures: tuple[tuple[int, np.ndarray], ...] = ()
    source_vertex_error_bounds: tuple[tuple[int, float], ...] = ()
    source_edge_endpoint_roots: tuple[
        tuple[int, tuple[RootEndpoint | None, RootEndpoint | None]], ...
    ] = ()
    source_coedge_endpoint_roots: tuple[
        tuple[int, tuple[RootEndpoint | None, RootEndpoint | None]], ...
    ] = ()

    def __post_init__(self) -> None:
        if self.topology_evidence.source_branch_id != self.branch_id:
            raise ValueError(
                f"CadCurveApproximation.topology_evidence.source_branch_id must equal branch_id; got {self.topology_evidence.source_branch_id!r}, expected={self.branch_id!r}"
            )
        if (self.source_geometry_id is None) != (self.export_geometry_id is None):
            raise ValueError(
                f"CadCurveApproximation.source_geometry_id must be present exactly when export_geometry_id is present; got source={self.source_geometry_id!r}, export={self.export_geometry_id!r}"
            )
        object.__setattr__(
            self,
            "approximation_id",
            canonical_fingerprint(
                {
                    "kind": "coupled-branch-approximation",
                    "source-branch-id": self.branch_id,
                    "policy-id": self.policy_id,
                    "curve": encode_geometry(self.curve),
                    "first-pcurve": encode_geometry(self.first_pcurve),
                    "second-pcurve": encode_geometry(self.second_pcurve),
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class CadExportResult:
    """Published external CAD file, its report and approximation evidence."""

    receipt: PublicationReceipt
    format: str
    schema: str
    report: AdapterReport
    entity_counts: tuple[tuple[str, int], ...]
    approximations: tuple[CadCurveApproximation, ...]
    provenance: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.receipt, PublicationReceipt):
            raise TypeError(
                f"CadExportResult.receipt must be PublicationReceipt; got {type(self.receipt).__name__}"
            )
        if not isinstance(self.report, AdapterReport) or not self.report.valid:
            raise ValueError(
                "CadExportResult.report must be a valid AdapterReport; got invalid report"
            )
        if not isinstance(self.approximations, tuple) or any(
            not isinstance(item, CadCurveApproximation) for item in self.approximations
        ):
            raise TypeError(
                "CadExportResult.approximations must contain immutable coupled certificates; got invalid approximation inventory"
            )
        if self.approximations and self.report.status != AdapterStatus.DECLARED_LOSS:
            raise ValueError(
                f"CadExportResult.report.status must be DECLARED_LOSS for approximated exports; got {self.report.status.name}"
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


def approximation_export_qualifiers(
    approximations: Sequence[CadCurveApproximation],
    /,
) -> dict[str, str]:
    """Publish source/target identity and continuous certificate summaries."""
    if not approximations:
        return {}
    qualifiers = {
        "approximation-policy": ",".join(
            sorted({item.policy_id for item in approximations})
        ),
        "source-branch-identities": ",".join(item.branch_id for item in approximations),
        "approximation-identities": ",".join(
            item.approximation_id for item in approximations
        ),
        "continuous-distance-bound-max": format(
            max(item.continuous_evidence.distance_bound for item in approximations),
            ".17g",
        ),
        "continuous-parameter-bound-max": format(
            max(item.continuous_evidence.parameter_bound for item in approximations),
            ".17g",
        ),
        "continuous-correspondence-bound-max": format(
            max(
                max(
                    item.continuous_evidence.first_correspondence_bound,
                    item.continuous_evidence.second_correspondence_bound,
                )
                for item in approximations
            ),
            ".17g",
        ),
        "certificate-cells": str(
            sum(item.certification_cells for item in approximations)
        ),
        "continuous-jet-evaluations": str(
            sum(item.continuous_jet_evaluations for item in approximations)
        ),
        "certificate-peak-numeric-bytes": str(
            max(item.certification_peak_bytes for item in approximations)
        ),
        "approximation-evidence": "continuous coupled deviation/correspondence; embedded source-fit straight homotopy; exported trim separation",
        "arithmetic-premise": "canonical outward IEEE interval arithmetic; conditional four-ulp transcendental enclosures",
    }
    source_ids = {
        item.source_geometry_id
        for item in approximations
        if item.source_geometry_id is not None
    }
    export_ids = {
        item.export_geometry_id
        for item in approximations
        if item.export_geometry_id is not None
    }
    if source_ids:
        qualifiers["source-geometry-identities"] = ",".join(sorted(source_ids))
        qualifiers["approximate-geometry-identities"] = ",".join(sorted(export_ids))
    return qualifiers


@checked
def build_cad_export_report(
    model: BRepModel,
    receipt: PublicationReceipt,
    policy: CadExportPolicy,
    approximations: Sequence[CadCurveApproximation],
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
    loss_field_prefix: str = "geometry.intersection-curve.",
    extra_losses: Sequence[AdapterLoss] = (),
    waiver_format_name: str | None = None,
) -> AdapterReport:
    """Build export loss, qualification and waiver evidence for fitted branches."""
    losses = (
        *extra_losses,
        *tuple(
            AdapterLoss(
                f"{loss_field_prefix}{approximation.branch_id}",
                "export",
                "transformed",
                f"New approximation {approximation.approximation_id}; continuous 3D "
                f"deviation bound {approximation.continuous_evidence.distance_bound:.17g}, "
                f"UV deviation bound {approximation.continuous_evidence.parameter_bound:.17g}, "
                f"surface-lift bounds ({approximation.continuous_evidence.first_correspondence_bound:.17g}, "
                f"{approximation.continuous_evidence.second_correspondence_bound:.17g}); "
                "continuous embedded source-fit homotopy and exported trim separation.",
                changes_interpretation=True,
            )
            for approximation in approximations
        ),
    )
    return AdapterReport(
        AdapterStatus.DECLARED_LOSS if losses else AdapterStatus.LOSSLESS,
        "phydrax-brep",
        format_name,
        source_id=model.model_id,
        target_id=receipt.content_sha256 if target_id is None else target_id,
        stage=stage,
        source_profile=source_profile,
        target_profile=AdapterFormatProfile(
            format_name,
            qualifiers=dict(qualifiers) | approximation_export_qualifiers(approximations),
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


def _cad_export_substitution_bounds(
    model: BRepModel,
    geometry: BRepGeometry,
    approximations: Sequence[CadCurveApproximation],
) -> dict[BRepEntityId, float]:
    """Compose branch substitution and numeric-vertex bounds by incident entity."""
    substitutions: dict[BRepEntityId, float] = {}
    for approximation in approximations:
        if approximation.source_geometry_id is not None and (
            approximation.source_geometry_id != geometry.geometry_id
        ):
            raise refuse(
                "inexact-export",
                f"intersection-curve {approximation.branch_id}",
                f"CadCurveApproximation.source_geometry_id must match the source model geometry; got {approximation.source_geometry_id!r}, expected={geometry.geometry_id!r}",
            )
        proof = approximation.continuous_evidence
        deviation_bounds = (
            proof.distance_bound,
            proof.first_correspondence_bound,
            proof.second_correspondence_bound,
        )
        if any(not isfinite(value) or value < 0.0 for value in deviation_bounds):
            raise refuse(
                "inexact-export",
                f"intersection-curve {approximation.branch_id}",
                f"CadCurveApproximation.continuous_evidence must have finite nonnegative distance and correspondence bounds; got {deviation_bounds!r}",
            )
        deviation = max(deviation_bounds)
        affected: set[BRepEntityId] = set()
        for edge, curve_index in enumerate(geometry.edge_curves):
            if curve_index < 0:
                continue
            curve = geometry.curves[curve_index]
            if (
                isinstance(curve, IntersectionCurve)
                and curve.branch_id == approximation.branch_id
            ):
                affected.add(BRepEntityId(model.source_revision, "edge", edge))
                affected.update(
                    BRepEntityId(model.source_revision, "vertex", vertex)
                    for vertex in geometry.edge_vertices[edge]
                )
                affected.update(
                    BRepEntityId(
                        model.source_revision, "face", geometry.coedge_faces[coedge]
                    )
                    for coedge, used_edge in enumerate(geometry.coedge_edges)
                    if used_edge == edge
                )
        if not affected:
            raise refuse(
                "inexact-export",
                f"intersection-curve {approximation.branch_id}",
                f"CadCurveApproximation.branch_id must bind an exported source branch; got {approximation.branch_id!r} with no matching edges",
            )
        vertex_errors = dict(approximation.source_vertex_error_bounds)
        for entity in affected:
            entity_deviation = deviation
            if entity.kind == "vertex" and entity.index in vertex_errors:
                error = vertex_errors[entity.index]
                if not isfinite(error) or error < 0.0:
                    raise refuse(
                        "inexact-export",
                        f"vertex:{entity.index}",
                        f"CadCurveApproximation.source_vertex_error_bounds must be finite and nonnegative; got vertex={entity.index}, bound={error!r}",
                    )
                entity_deviation = float(np.nextafter(deviation + error, np.inf))
            substitutions[entity] = max(substitutions.get(entity, 0.0), entity_deviation)
    return substitutions


@checked
def certified_cad_export_tolerances(
    model: BRepModel,
    approximations: Sequence[CadCurveApproximation] = (),
    /,
) -> CadExportTolerances:
    """Return entity tolerances and uncertainty from gap and substitution proofs.

    Bounds have model length units. Missing or unbounded entity evidence refuses.
    Substitution deviation is added outward to the source incidence bound.
    """
    geometry = model.geometry
    if geometry is None:
        raise refuse(
            "inexact-export",
            model.source_id,
            "BRepModel.geometry must be present for CAD export; got None",
        )
    evidence = certify_brep_gaps(model)
    bounds = _gap_entity_bounds(evidence)
    substitutions = _cad_export_substitution_bounds(model, geometry, approximations)
    for entity, deviation in substitutions.items():
        if entity in bounds:
            bounds[entity] = float(np.nextafter(bounds[entity] + deviation, np.inf))
    rows: list[tuple[float, ...]] = []
    for kind, count in (
        ("vertex", len(geometry.vertex_roots)),
        ("edge", len(geometry.edge_curves)),
        ("face", len(geometry.face_loops)),
    ):
        row: list[float] = []
        for index in range(count):
            entity = BRepEntityId(model.source_revision, kind, index)
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
                float(curve.distance) * factors[0],
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
    coefficientwise; an isoline of a source sphere or torus is the closed-form
    circle it parameterizes. Nothing is sampled or refitted. Curves without
    such a rule are returned unchanged for the writer to refuse.
    """
    match curve:
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
    """Parameter of the carrier point nearest ``point`` and its distance.

    Lines and conics invert in closed form (conic angles in ``(-pi, pi]``);
    B-splines use bounded sampling plus Newton refinement on the native
    evaluator.
    """
    target = np.asarray(point, dtype=np.float64)
    match curve:
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
            parameter = _spline_parameter(curve, target)
        case _:
            raise TypeError(
                f"curve_parameter.curve must support parameter inversion; got {type(curve).__name__}"
            )
    distance = float(
        np.linalg.norm(curve_evaluate_host(curve, np.asarray(parameter)) - target)
    )
    return parameter, distance


def _spline_parameter(curve: BSplineCurve, target: np.ndarray, /) -> float:
    domain = curve.parameter_domain
    if domain is None:
        raise CadCodecInvariantError(
            "BSplineCurve.parameter_domain must be finite; got None"
        )
    samples = np.linspace(domain[0], domain[1], 8 * curve.control_points.shape[0] + 1)
    values = curve_evaluate_host(curve, samples)
    parameter = float(samples[int(np.argmin(np.linalg.norm(values - target, axis=1)))])
    derivative = jax.jacfwd(curve.evaluate)
    second = jax.jacfwd(derivative)

    def value_jets(row: jax.Array) -> jax.Array:
        parameter = row[0]
        return jnp.stack(
            (curve.evaluate(parameter), derivative(parameter), second(parameter)),
            axis=-1,
        )

    prepared = prepare_interval_function(value_jets, 1, batch_capacity=1)
    for _ in range(32):
        jets = prepared.evaluate_host_points(np.asarray([[parameter]], dtype=np.float64))[
            0
        ]
        value, first_, second_ = jets[:, 0], jets[:, 1], jets[:, 2]
        gradient = float((value - target) @ first_)
        curvature = float(first_ @ first_ + (value - target) @ second_)
        if curvature <= 0.0:
            break
        step = gradient / curvature
        parameter = min(max(parameter - step, domain[0]), domain[1])
        if abs(step) <= 1.0e-15 * max(1.0, abs(parameter)):
            break
    return parameter


@checked
def edge_range(
    curve: AbstractCurve,
    start: np.ndarray,
    end: np.ndarray,
    closed: bool,
    tolerance: float,
    /,
) -> tuple[float, float]:
    """Increasing carrier range from ``start`` to ``end`` (a full period if closed).

    Raises ``ValueError`` when a vertex does not lie on the carrier within
    ``tolerance``.
    """
    period = curve.period
    if closed:
        if period is not None:
            first, _ = curve_parameter(curve, start)
            return first, first + period
        domain = curve.parameter_domain
        if domain is None:
            raise ValueError(
                "edge_range.curve.parameter_domain must be finite for a nonperiodic closed edge; got None"
            )
        # A bounded closed edge uses its whole domain. Endpoint admission
        # proves that range directly; an inverse at the shared vertex cannot
        # select another range and introduces an unnecessary global root solve.
        ends = curve_evaluate_host(curve, np.asarray(domain))
        if max(np.linalg.norm(ends - start, axis=1)) > tolerance:
            raise ValueError(
                f"edge_range.start must lie on both bounded carrier endpoints within tolerance={tolerance}; got distances={np.linalg.norm(ends - start, axis=1)!r}"
            )
        return domain
    first, first_gap = curve_parameter(curve, start)
    last, last_gap = curve_parameter(curve, end)
    if max(first_gap, last_gap) > tolerance:
        raise ValueError(
            f"edge_range.start must lie within carrier tolerance={tolerance}; got deviation={max(first_gap, last_gap):.3e}"
        )
    if period is not None:
        while last <= first:
            last += period
        while last - first > period:
            last -= period
    if not last > first:
        raise ValueError(
            f"edge_range.end must increase from start to end; got first={first}, last={last}"
        )
    return first, last


# -------------------------------------------------------- surface inversion


def _angle(value: np.ndarray, /) -> np.ndarray:
    return np.arctan2(value[..., 1], value[..., 0])


@checked
def surface_parameters(
    patch: AbstractSurfacePatch, points: np.ndarray, /
) -> np.ndarray | None:
    """Closed-form ``(u, v)`` of points on analytic surfaces (angles in ``(-pi, pi]``).

    Returns ``None`` for families without a closed-form inverse (splines,
    sweeps), which use bounded numerical projection instead.
    """
    points_ = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    match patch:
        case OffsetSurface():
            equivalent = patch.analytic_equivalent()
            return None if equivalent is None else surface_parameters(equivalent, points_)
        case PlanePatch():
            relative = points_ - np.asarray(patch.origin)
            basis = np.stack(
                (np.asarray(patch.first_axis), np.asarray(patch.second_axis))
            )
            # Host-side 2x2 normal equations of the (possibly non-orthogonal) frame.
            return np.linalg.solve(basis @ basis.T, basis @ relative.T).T
        case CylinderPatch() | ConePatch():
            relative = points_ - np.asarray(patch.origin)
            planar = np.stack(
                (
                    relative @ np.asarray(patch.first_axis),
                    relative @ np.asarray(patch.second_axis),
                ),
                axis=-1,
            )
            return np.stack((_angle(planar), relative @ np.asarray(patch.axis)), axis=-1)
        case SpherePatch():
            relative = points_ - np.asarray(patch.center)
            x = relative @ np.asarray(patch.first_axis)
            y = relative @ np.asarray(patch.second_axis)
            z = relative @ np.asarray(patch.axis)
            return np.stack((np.arctan2(y, x), np.arctan2(z, np.hypot(x, y))), axis=-1)
        case TorusPatch():
            relative = points_ - np.asarray(patch.center)
            x = relative @ np.asarray(patch.first_axis)
            y = relative @ np.asarray(patch.second_axis)
            z = relative @ np.asarray(patch.axis)
            ring = np.hypot(x, y) - float(patch.major_radius)
            return np.stack((np.arctan2(y, x), np.arctan2(z, ring)), axis=-1)
        case (
            BSplineSurfacePatch()
            | ExtrusionSurface()
            | RevolutionSurface()
            | RuledSurface()
        ):
            return None
        case _:
            raise TypeError(
                f"surface_parameters.patch must support parameter inversion; got {type(patch).__name__}"
            )


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


@checked
def project_to_surface(
    patch: AbstractSurfacePatch,
    points: np.ndarray,
    guess: np.ndarray | None,
    /,
) -> np.ndarray:
    """Surface parameters of points by bounded grid search plus Gauss-Newton.

    Used only where no closed form exists; callers verify the reproduced
    points against their tolerance.
    """
    exact = surface_parameters(patch, points)
    if exact is not None:
        return exact
    natural = natural_box(patch)
    finite = np.where(
        np.isfinite(natural), natural, np.asarray(((-1.0, -1.0), (1.0, 1.0)))
    )
    grid = np.stack(
        np.meshgrid(
            np.linspace(finite[0, 0], finite[1, 0], 33),
            np.linspace(finite[0, 1], finite[1, 1], 33),
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 2)
    values = surface_evaluate_host(patch, grid)
    prepared = prepare_surface_host(patch, with_differential=True, batch_capacity=1)
    result = []
    for index, point in enumerate(np.asarray(points, dtype=np.float64).reshape(-1, 3)):
        uv = (
            grid[int(np.argmin(np.linalg.norm(values - point, axis=1)))]
            if guess is None
            else np.asarray(guess[index], dtype=np.float64)
        )
        for _ in range(40):
            jets = prepared.evaluate_host_points(uv.reshape(1, 2))[0]
            residual = jets[:, 0] - point
            differential = jets[:, 1:]
            # Host-side 3x2 least-squares Gauss-Newton step of one projection.
            step = np.linalg.lstsq(differential, -residual, rcond=None)[0]
            uv = np.clip(uv + step, natural[0], natural[1])
            if np.max(np.abs(step)) <= 1.0e-15 * max(1.0, float(np.max(np.abs(uv)))):
                break
        result.append(uv)
    return np.asarray(result, dtype=np.float64).reshape(-1, 2)


# ------------------------------------------------------------ p-curve recovery


def _unwrap(values: np.ndarray, period: float | None, /) -> np.ndarray:
    if period is None:
        return values
    return np.unwrap(values, period=period)


def _affine_line(
    patch: AbstractSurfacePatch,
    curve: AbstractCurve,
    first: float,
    last: float,
    tolerance: float,
    /,
) -> LineCurve | None:
    """Algebraic isoparametric recovery, never recognition from sampled agreement."""
    del first, last
    if isinstance(curve, LineCurve) and isinstance(patch, (CylinderPatch, ConePatch)):
        origin = np.asarray(curve.origin)
        direction = np.asarray(curve.direction)
        axis = np.asarray(patch.axis)
        parameters = surface_parameters(patch, origin[None, :])
        if parameters is None:
            return None
        uv = parameters[0]
        radial = np.cos(uv[0]) * np.asarray(patch.first_axis) + np.sin(
            uv[0]
        ) * np.asarray(patch.second_axis)
        ruling = axis + (
            np.tan(float(patch.semi_angle)) * radial
            if isinstance(patch, ConePatch)
            else 0.0
        )
        speed = float(direction @ ruling / (ruling @ ruling))
        if (
            np.linalg.norm(direction - speed * ruling) <= tolerance
            and np.linalg.norm(surface_evaluate_host(patch, uv) - origin) <= tolerance
        ):
            return LineCurve(uv, (0.0, speed))
        return None
    if not isinstance(curve, CircleCurve) or not isinstance(
        patch, (CylinderPatch, ConePatch, SpherePatch, TorusPatch)
    ):
        return None
    axis = np.asarray(patch.axis)
    x, y = np.asarray(patch.first_axis), np.asarray(patch.second_axis)
    base = np.asarray(
        patch.center if isinstance(patch, (SpherePatch, TorusPatch)) else patch.origin
    )
    center = np.asarray(curve.center)
    a = float(curve.radius) * np.asarray(curve.first_axis)
    b = float(curve.radius) * np.asarray(curve.second_axis)

    def agrees(c: np.ndarray, cosine: np.ndarray, sine: np.ndarray) -> bool:
        # Equality of constant/cosine/sine coefficients proves the whole carrier.
        return (
            max(
                np.linalg.norm(center - c),
                np.linalg.norm(a - cosine),
                np.linalg.norm(b - sine),
            )
            <= tolerance
        )

    phase = float(np.arctan2(a @ y, a @ x))
    radial = np.cos(phase) * x + np.sin(phase) * y
    tangent = -np.sin(phase) * x + np.cos(phase) * y
    height = float((center - base) @ axis)
    radius = float(curve.radius)
    match patch:
        case CylinderPatch():
            v, ring_radius = height, float(patch.radius)
        case ConePatch():
            v = height
            ring_radius = float(patch.reference_radius) + height * np.tan(
                float(patch.semi_angle)
            )
        case SpherePatch():
            v = float(np.arcsin(np.clip(height / float(patch.radius), -1.0, 1.0)))
            ring_radius = float(patch.radius) * np.cos(v)
        case TorusPatch():
            v = float(np.arctan2(height, radius - float(patch.major_radius)))
            ring_radius = float(patch.major_radius) + float(patch.minor_radius) * np.cos(
                v
            )
            height = float(patch.minor_radius) * np.sin(v)
        case _:
            raise CadCodecInvariantError(
                f"_affine_line.patch must be an analytic ring carrier; got {type(patch).__name__}"
            )
    for sense in (1.0, -1.0):
        if agrees(
            base + height * axis, ring_radius * radial, sense * ring_radius * tangent
        ):
            return LineCurve((phase, v), (sense, 0.0))
    if not isinstance(patch, (SpherePatch, TorusPatch)):
        return None
    horizontal = a - float(a @ axis) * axis
    if np.linalg.norm(horizontal) <= tolerance:
        horizontal = b - float(b @ axis) * axis
    if np.linalg.norm(horizontal) <= tolerance:
        return None
    radial = horizontal / np.linalg.norm(horizontal)
    longitude = float(np.arctan2(radial @ y, radial @ x))
    tube_radius = float(
        patch.radius if isinstance(patch, SpherePatch) else patch.minor_radius
    )
    tube_center = (
        base
        if isinstance(patch, SpherePatch)
        else base + float(patch.major_radius) * radial
    )
    phase = float(np.arctan2(a @ axis, a @ radial))
    cosine = tube_radius * (np.cos(phase) * radial + np.sin(phase) * axis)
    sine = tube_radius * (-np.sin(phase) * radial + np.cos(phase) * axis)
    for sense in (1.0, -1.0):
        if agrees(tube_center, cosine, sense * sine):
            return LineCurve((longitude, phase), (0.0, sense))
    return None


def _plane_pcurve(patch: PlanePatch, curve: AbstractCurve, /) -> AbstractCurve | None:
    """Return the parameter-preserving planar image of a conic or spline in an orthonormal frame."""
    first_axis = np.asarray(patch.first_axis)
    second_axis = np.asarray(patch.second_axis)
    basis = np.stack((first_axis, second_axis))
    if np.max(np.abs(basis @ basis.T - np.eye(2))) > 1.0e-14:
        return None
    origin = np.asarray(patch.origin)
    normal = np.cross(first_axis, second_axis)

    def in_plane(points: np.ndarray) -> bool:
        return bool(np.max(np.abs((points - origin) @ normal)) <= 1.0e-9)

    match curve:
        case LineCurve():
            if not in_plane(
                np.stack(
                    (
                        np.asarray(curve.origin),
                        np.asarray(curve.origin) + np.asarray(curve.direction),
                    )
                )
            ):
                return None
            return LineCurve(
                basis @ (np.asarray(curve.origin) - origin),
                basis @ np.asarray(curve.direction),
            )
        case CircleCurve() | EllipseCurve():
            if not in_plane(np.asarray(curve.center)[None, :]):
                return None
            center = basis @ (np.asarray(curve.center) - origin)
            axes = (
                basis @ np.asarray(curve.first_axis),
                basis @ np.asarray(curve.second_axis),
            )
            if (
                abs(np.linalg.norm(axes[0]) - 1.0) > 1.0e-12
                or abs(np.linalg.norm(axes[1]) - 1.0) > 1.0e-12
            ):
                return None
            if isinstance(curve, CircleCurve):
                return CircleCurve(center, axes[0], axes[1], curve.radius)
            return EllipseCurve(
                center, axes[0], axes[1], curve.first_radius, curve.second_radius
            )
        case BSplineCurve():
            if not in_plane(np.asarray(curve.control_points)):
                return None
            controls = (np.asarray(curve.control_points) - origin) @ basis.T
            return BSplineCurve(controls, curve.weights, curve.knots, curve.degree)
        case _:
            return None


def _sweep_pcurve(
    patch: AbstractSurfacePatch, curve: AbstractCurve, first: float, tolerance: float, /
) -> LineCurve | None:
    """Sweep isolines verified by carrier coefficients, not sampled agreement."""
    match patch:
        case ExtrusionSurface() | RevolutionSurface():
            basis = patch.curve
        case _:
            return None

    def equal(transformed: AbstractCurve) -> bool:
        match curve, transformed:
            case LineCurve(), LineCurve():
                pairs = (
                    (curve.origin, transformed.origin),
                    (curve.direction, transformed.direction),
                )
            case CircleCurve(), CircleCurve():
                ra, rb = float(curve.radius), float(transformed.radius)
                pairs = (
                    (curve.center, transformed.center),
                    (
                        ra * np.asarray(curve.first_axis),
                        rb * np.asarray(transformed.first_axis),
                    ),
                    (
                        ra * np.asarray(curve.second_axis),
                        rb * np.asarray(transformed.second_axis),
                    ),
                )
            case EllipseCurve(), EllipseCurve():
                pairs = (
                    (curve.center, transformed.center),
                    (
                        float(curve.first_radius) * np.asarray(curve.first_axis),
                        float(transformed.first_radius)
                        * np.asarray(transformed.first_axis),
                    ),
                    (
                        float(curve.second_radius) * np.asarray(curve.second_axis),
                        float(transformed.second_radius)
                        * np.asarray(transformed.second_axis),
                    ),
                )
            case BSplineCurve(), BSplineCurve():
                if (
                    curve.degree != transformed.degree
                    or not np.array_equal(curve.knots, transformed.knots)
                    or not np.array_equal(curve.weights, transformed.weights)
                ):
                    return False
                pairs = ((curve.control_points, transformed.control_points),)
            case _:
                return False
        return all(
            np.asarray(a).shape == np.asarray(b).shape
            and np.max(np.abs(np.asarray(a) - np.asarray(b))) <= tolerance
            for a, b in pairs
        )

    # Local import avoids the shared carrier module's dependency on this module.
    from ._cad_carriers import place_curve, RigidPlacement

    match patch:
        case ExtrusionSurface() if type(basis) is type(curve):
            match curve, basis:
                case LineCurve(), LineCurve():
                    difference = np.asarray(curve.origin) - np.asarray(basis.origin)
                case BSplineCurve(), BSplineCurve():
                    difference = (
                        np.asarray(curve.control_points)
                        - np.asarray(basis.control_points)
                    ).reshape(-1, 3)[0]
                case CircleCurve(), CircleCurve():
                    difference = np.asarray(curve.center) - np.asarray(basis.center)
                case EllipseCurve(), EllipseCurve():
                    difference = np.asarray(curve.center) - np.asarray(basis.center)
                case _:
                    raise TypeError(
                        f"_sweep_pcurve.curve must have a supported coefficient anchor; got {type(curve).__name__}"
                    )
            direction = np.asarray(patch.direction)
            height = float(difference @ direction / (direction @ direction))
            shifted = place_curve(
                basis, RigidPlacement(np.eye(3), height * direction), "sweep"
            )
            if equal(shifted):
                return LineCurve((0.0, height), (1.0, 0.0))
        case _:
            pass
    uv = project_to_surface(
        patch, curve_evaluate_host(curve, np.asarray((first,))), None
    )[0]
    match patch, curve:
        case ExtrusionSurface(), LineCurve():
            direction = np.asarray(patch.direction)
            speed = float(
                np.asarray(curve.direction) @ direction / (direction @ direction)
            )
            if (
                np.linalg.norm(np.asarray(curve.direction) - speed * direction)
                <= tolerance
                and np.linalg.norm(
                    surface_evaluate_host(patch, uv)
                    - curve_evaluate_host(curve, np.asarray(first))
                )
                <= tolerance
            ):
                return LineCurve((uv[0], uv[1] - speed * first), (0.0, speed))
        case _:
            pass
    match patch:
        case RevolutionSurface():
            pass
        case _:
            return None
    axis = np.asarray(patch.axis_direction)
    origin = np.asarray(patch.axis_origin)
    angle = float(uv[0])
    cross = np.asarray(
        ((0.0, -axis[2], axis[1]), (axis[2], 0.0, -axis[0]), (-axis[1], axis[0], 0.0))
    )
    rotation = (
        np.cos(angle) * np.eye(3)
        + (1.0 - np.cos(angle)) * np.outer(axis, axis)
        + np.sin(angle) * cross
    )
    rotated = place_curve(
        basis, RigidPlacement(rotation, origin - rotation @ origin), "sweep"
    )
    if equal(rotated):
        return LineCurve((angle, 0.0), (0.0, 1.0))
    if isinstance(curve, CircleCurve):
        profile = curve_evaluate_host(basis, np.asarray(float(uv[1]))) - origin
        center = origin + float(profile @ axis) * axis
        radial = profile - float(profile @ axis) * axis
        a = float(curve.radius) * np.asarray(curve.first_axis)
        b = float(curve.radius) * np.asarray(curve.second_axis)
        phase = float(np.arctan2(a @ np.cross(axis, radial), a @ radial))
        cosine = np.cos(phase) * radial + np.sin(phase) * np.cross(axis, radial)
        sine = np.cross(axis, cosine)
        for sense in (1.0, -1.0):
            if (
                max(
                    np.linalg.norm(center - np.asarray(curve.center)),
                    np.linalg.norm(a - cosine),
                    np.linalg.norm(b - sense * sine),
                )
                <= tolerance
            ):
                return LineCurve((phase, uv[1]), (sense, 0.0))
    return None


def _fit_basis(knots: np.ndarray, count: int, parameters: np.ndarray, /) -> np.ndarray:
    # The rational evaluator with identity controls yields the basis.
    basis = BSplineCurve(np.eye(count), np.ones(count), knots, 3)
    return curve_evaluate_host(basis, parameters)


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
def fit_bspline(
    sampler: Callable[[np.ndarray], np.ndarray],
    breakpoints: np.ndarray,
    error: Callable[[BSplineCurve, np.ndarray], float],
    tolerance: float,
    policy: CadCurveFitPolicy,
    /,
    *,
    accept: Callable[[BSplineCurve], bool] | None = None,
) -> tuple[BSplineCurve, float] | None:
    """Propose endpoint/chart-node constrained cubic least-squares fits.

    Every breakpoint has one shared C0 control point. Uniform spans double
    until sampled residuals propose a candidate and ``accept`` certifies it,
    or the control-point budget is exhausted. Without ``accept`` the returned
    curve is a proposal only; publication checks its acceptance.
    """
    bounds = np.asarray(breakpoints, dtype=np.float64)
    checks = np.unique(
        np.concatenate((np.linspace(bounds[0], bounds[-1], policy.check_samples), bounds))
    )
    spans = 1
    while True:
        knots = _segment_knots(bounds, spans)
        count = knots.shape[0] - 4
        if count > policy.maximum_control_points:
            return None
        parameters = np.unique(
            np.concatenate(
                [
                    np.linspace(lower, upper, 4 * spans + 1)
                    for lower, upper in zip(bounds[:-1], bounds[1:], strict=True)
                ]
            )
        )
        design = _fit_basis(knots, count, parameters)
        fixed = np.arange(bounds.size, dtype=np.int64) * (spans + 2)
        values = sampler(parameters)
        controls = np.empty((count, values.shape[1]), dtype=np.float64)
        controls[fixed] = sampler(bounds)
        free = np.ones(count, dtype=np.bool_)
        free[fixed] = False
        residual_values = values - design[:, fixed] @ controls[fixed]
        # Host-side bounded least squares preserves authored breakpoint interpolation values.
        controls[free] = np.linalg.lstsq(design[:, free], residual_values, rcond=None)[0]
        fitted = BSplineCurve(controls, np.ones(count), knots, 3)
        residual = error(fitted, checks)
        if residual <= tolerance and (accept is None or accept(fitted)):
            return fitted, residual
        spans *= 2


@dataclass(frozen=True, slots=True)
class RecoveredPCurve:
    """Return a p-curve with provenance kind and relative fitting error."""

    curve: AbstractCurve
    kind: Literal["honored", "exact", "fitted"]
    error: float


@checked
def recover_pcurve(
    patch: AbstractSurfacePatch,
    curve: AbstractCurve,
    first: float,
    last: float,
    scale: float,
    policy: CadCurveFitPolicy | None,
    /,
) -> RecoveredPCurve:
    """Recover an algebraic p-curve, or an explicitly requested sampled-error fit.

    Raises ``ValueError`` when algebraic recovery is unavailable without a fitting
    policy, or when the requested sampled tolerance cannot be reached.
    """
    tolerance = 1.0e-9 * scale
    if isinstance(patch, OffsetSurface):
        equivalent = patch.analytic_equivalent()
        if equivalent is not None:
            # This proof preserves the entire parameter map exactly. Only
            # recovery uses it; the published surface retains its source tree.
            return recover_pcurve(equivalent, curve, first, last, scale, policy)
    if isinstance(patch, PlanePatch):
        planar = _plane_pcurve(patch, curve)
        if planar is not None:
            return RecoveredPCurve(planar, "exact", 0.0)
    line = _affine_line(patch, curve, first, last, tolerance)
    if line is not None:
        return RecoveredPCurve(line, "exact", 0.0)
    sweep = _sweep_pcurve(patch, curve, first, tolerance)
    if sweep is not None:
        return RecoveredPCurve(sweep, "exact", 0.0)
    if policy is None:
        raise ValueError(
            f"recover_pcurve.policy must be present when algebraic recovery is unsupported; got None for surface={type(patch).__name__}, curve={type(curve).__name__}"
        )
    periods = patch.periods

    def sampler(parameters: np.ndarray) -> np.ndarray:
        uv = project_to_surface(patch, curve_evaluate_host(curve, parameters), None)
        return np.stack([_unwrap(uv[:, axis], periods[axis]) for axis in (0, 1)], axis=1)

    def error(fitted: BSplineCurve, parameters: np.ndarray) -> float:
        on_surface = surface_evaluate_host(patch, curve_evaluate_host(fitted, parameters))
        return float(
            np.max(
                np.linalg.norm(
                    on_surface - curve_evaluate_host(curve, parameters), axis=1
                )
            )
        )

    fitted = fit_bspline(
        sampler, np.asarray((first, last)), error, policy.tolerance * scale, policy
    )
    if fitted is None:
        raise ValueError(
            f"CadCurveFitPolicy.tolerance must admit a surface-consistent p-curve fit; got no accepted proposal at tolerance={policy.tolerance}, controls={policy.maximum_control_points}"
        )
    return RecoveredPCurve(fitted[0], "fitted", fitted[1] / scale)


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

    Readers add vertices, carriers, edges (with vertex-inverted ranges), faces
    (loops of coedges in the orientation of the parametric normal), shells,
    solids and occurrences; `CadStage.publish` recovers missing p-curves,
    seam representatives, pole edges and parameter boxes and publishes one
    validated `BRepModel`.
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
    occurrences: list[BRepOccurrence] = field(default_factory=list)
    assembly_containers: list[BRepAssemblyContainer] = field(default_factory=list)
    provenance: list[tuple[str, str]] = field(default_factory=list)
    pcurves_honored: int = 0
    pcurves_exact: int = 0
    pcurves_fitted: int = 0
    maximum_fit_error: float = 0.0
    degenerate_edges: int = 0
    natural_boundaries: int = 0
    native_phases: dict[
        int,
        tuple[AbstractSurfacePatch | None, Literal[0, 1] | None, Fraction],
    ] = field(default_factory=dict)

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
            if not last > first:
                raise ValueError(
                    f"CadStage.edge.parameter_range must increase; got first={first}, last={last}"
                )
            curve.validate_range(first, last)
            ends = curve_evaluate_host(curve, np.asarray((first, last)))
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
            and (first, last) == (0.0, _TWO_PI)
        ):
            # A closed conic entity over its whole native parameter interval
            # authors one mathematical turn; binary64 tau is only its carrier.
            self.native_phase(edge, None, None)
        return edge

    def degenerate(self, vertex: int, first: float, last: float, /) -> int:
        self.edge_curves.append(-1)
        self.edge_ranges.append((float(first), float(last)))
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
        points = surface_evaluate_host(patch, parameters)
        if not np.all(np.isfinite(points)):
            raise refuse(
                "inconsistent-geometry",
                label,
                f"CadStage.natural_loop.patch must have finite values at its natural seams and poles; got points={points!r}",
            )
        if sphere:
            poles = patch.degenerate_isolines(box)
            if (1, lower) not in poles or (1, upper) not in poles:
                raise refuse(
                    "inconsistent-geometry",
                    label,
                    f"CadStage.natural_loop.patch must declare both sphere pole isolines; got {poles!r}",
                )
        # A closed torus seam is one native tube period by construction; only
        # the open sphere meridian carries an authored latitude interval.
        seam_curve = SurfaceIsoparametricCurve(
            patch, 0, 0.0, parameter_range=(lower, upper) if sphere else None
        )
        seam_source = seam_curve.p_curve()
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
            ring_source = ring_curve.p_curve()
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

    @staticmethod
    def _canonical_native_sheet(
        pcurve: AbstractCurve, patch: AbstractSurfacePatch, /
    ) -> AbstractCurve:
        if isinstance(pcurve, PeriodicPCurve):
            return pcurve
        periods = patch.periods
        symbols = patch.native_period_symbols()
        if not any(symbol is not None for symbol in symbols):
            return pcurve
        carrier = explicit_carrier(pcurve)
        affine = affine_curve_coefficients(carrier, np.empty((2,), dtype=np.float64))
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
                np.asarray(
                    tuple(float(value) for value in source_constant),
                    dtype=np.float64,
                ),
                np.asarray(
                    tuple(float(value) for value in direction),
                    dtype=np.float64,
                ),
            )
            return PeriodicPCurve(source, patch, (turns[0], turns[1]))
        if isinstance(pcurve, LineCurve):
            return pcurve
        if any(
            symbol == "two_pi" and derivative
            for derivative, symbol in zip(direction, symbols, strict=True)
        ):
            source = LineCurve(
                np.zeros((2,), dtype=np.float64),
                np.asarray(
                    tuple(float(value) for value in direction),
                    dtype=np.float64,
                ),
            )
            return AffinePCurve(
                source,
                ((Fraction(1), Fraction()), (Fraction(), Fraction(1))),
                (constant[0], constant[1]),
            )
        return pcurve

    def _coedge_pcurve(self, face: StagedFace, coedge: StagedCoedge, /) -> AbstractCurve:
        edge = coedge.edge
        first, last = self.edge_ranges[edge]
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
            recovered = recover_pcurve(
                face.patch, curve, first, last, self.scale, self.fit_policy
            )
        except ValueError as error:
            raise refuse(
                "inconsistent-geometry", coedge.label, str(error), (face.label,)
            ) from error
        match recovered.kind:
            case "exact":
                self.pcurves_exact += 1
            case "fitted":
                self.pcurves_fitted += 1
                self.maximum_fit_error = max(self.maximum_fit_error, recovered.error)
            case "honored":
                self.pcurves_honored += 1
        return recovered.curve

    def _ends(self, coedge: StagedCoedge, pcurve: AbstractCurve, /) -> np.ndarray:
        first, last = self.edge_ranges[coedge.edge]
        ends = curve_evaluate_host(pcurve, np.asarray((first, last)))
        return ends if coedge.sense > 0 else ends[::-1]

    def _resolve_loop(
        self, face: StagedFace, loop: list[StagedCoedge], /
    ) -> list[tuple[StagedCoedge, AbstractCurve]]:
        """Seam representatives and pole edges making the loop closed in (u, v).

        A gap between consecutive p-curves along a collapsed isoline (a pole or
        apex) is closed by a degenerate edge, inserted in place (at the loop
        start for the closing gap); otherwise the next p-curve is shifted by
        whole periods to connect (the other seam representative). The two uses
        of a seam edge that land on the same trace (for example both recovered
        by inversion) are separated by one period so the loop turns
        counterclockwise. Any other gap is refused.
        """
        periods = face.patch.periods
        resolved: list[tuple[StagedCoedge, AbstractCurve]] = []
        previous_end: np.ndarray | None = None
        previous_source: tuple[StagedCoedge, AbstractCurve] | None = None
        traces: dict[int, AbstractCurve] = {}
        for coedge in loop:
            pcurve = self._coedge_pcurve(face, coedge)
            pcurve = self._canonical_native_sheet(pcurve, face.patch)
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
            curve_evaluate_host(first_use, middle), curve_evaluate_host(pcurve, middle)
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
        if abs(tail[1 - moving] - head[1 - moving]) > slack:
            return False
        if moving != 0:
            return False
        v = float(head[1])
        patch = face.patch
        match patch:
            case SpherePatch():
                return any(
                    abs(head[1] - value) <= slack and abs(tail[1] - value) <= slack
                    for value in (-0.5 * pi, 0.5 * pi)
                )
            case OffsetSurface() | PlacedSurface():
                box = np.stack((np.minimum(head, tail), np.maximum(head, tail)))
                patch.validate_parameter_box(box)
                return any(
                    axis == 1 and head[1] == value and tail[1] == value
                    for axis, value in patch.degenerate_isolines(box)
                )
            case ConePatch():
                radius = float(patch.reference_radius) + v * np.tan(
                    float(patch.semi_angle)
                )
            case TorusPatch():
                radius = float(patch.major_radius) + float(patch.minor_radius) * np.cos(v)
            case RevolutionSurface():
                point = curve_evaluate_host(patch.curve, np.asarray(v)) - np.asarray(
                    patch.axis_origin
                )
                axis = np.asarray(patch.axis_direction)
                radius = float(np.linalg.norm(point - (point @ axis) * axis))
            case _:
                return False
        # The longitude coefficient vanishes throughout this isoline.
        return abs(radius) <= self.tolerance

    def _exact_coedge_endpoint(
        self,
        source: tuple[StagedCoedge, AbstractCurve],
        start: bool,
        /,
    ) -> tuple[ExactParameter, ...] | None:
        coedge, pcurve = source
        first, last = self.edge_ranges[coedge.edge]
        parameter = first if (coedge.sense > 0) == start else last

        with coordinate_source_request((pcurve, parameter)) as budget:
            return exact_curve_parameter_point(
                pcurve, Fraction(parameter), coordinate_budget=budget
            )

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
        """Return a pole p-curve, sense, periodic axis and symbolic-period status."""
        if not self._pole_gap(face, head, tail):
            raise refuse(
                "inconsistent-geometry",
                label,
                f"StagedFace.loops must connect consecutive p-curve endpoints in the parameter plane; got head={head!r}, tail={tail!r}",
                (face.label,),
            )
        moving = int(np.argmax(np.abs(tail - head)))
        fixed = 1 - moving

        def bounded_tolerance_curve() -> tuple[AbstractCurve, int, int, bool]:
            # The external topology owns this collapsed side, while its
            # endpoints are only tolerance-coincident. Retain that literal
            # bounded chart segment without claiming a symbolic native-period root.
            return LineCurve(head, tail - head), 1, moving, False

        head_exact = self._exact_coedge_endpoint(head_source, False)
        tail_exact = self._exact_coedge_endpoint(tail_source, True)
        if head_exact is None or tail_exact is None:
            return bounded_tolerance_curve()
        head_rational, head_turns = exact_parameter_parts(head_exact[moving])
        tail_rational, tail_turns = exact_parameter_parts(tail_exact[moving])
        head_fixed, head_fixed_turns = exact_parameter_parts(head_exact[fixed])
        tail_fixed, tail_fixed_turns = exact_parameter_parts(tail_exact[fixed])
        if (
            head_rational != tail_rational
            or abs(head_turns - tail_turns) != 1
            or head_turns.denominator != 1
            or tail_turns.denominator != 1
            or head_fixed != tail_fixed
            or head_fixed_turns
            or tail_fixed_turns
        ):
            return bounded_tolerance_curve()
        direction = [0.0, 0.0]
        direction[moving] = 1.0
        offset = [Fraction(), Fraction()]
        offset[moving] = head_rational
        offset[fixed] = head_fixed
        pcurve: AbstractCurve = exact_line_pcurve(
            (offset[0], offset[1]), (direction[0], direction[1])
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
        """Bind pole edges with symbolic periods or bounded literal chart sides."""
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
                    f"StagedCoedge.pcurve must span the bounded collapsed chart side between its neighbors; got coedge={coedge.label!r}, face={face.label!r}",
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
            points.append(curve_evaluate_host(pcurve, samples[:-1]))
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
                boxes.append(np.asarray(pcurve.bounding_box(first, last)))
        stacked = np.stack(boxes)
        box = np.stack((np.min(stacked[:, 0], axis=0), np.max(stacked[:, 1], axis=0)))
        natural = natural_box(patch)
        box = np.stack((np.maximum(box[0], natural[0]), np.minimum(box[1], natural[1])))
        # Symbolic native-period gauges have outward UV bounds wider than their
        # numerical period representative. Re-centering to that width clips
        # the represented source; only a source-declared domain may clip here.
        return box

    def _resolve_face(
        self, face: StagedFace, /
    ) -> tuple[list[list[tuple[StagedCoedge, AbstractCurve]]], np.ndarray]:
        loops = [self._resolve_loop(face, loop) for loop in face.loops]
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
            periods = staged.patch.native_period_symbols()
            for loop in loops:
                for coedge, pcurve in loop:
                    if coedge.edge != edge:
                        continue
                    source = (
                        pcurve.source_curve
                        if isinstance(pcurve, PeriodicPCurve)
                        else pcurve
                    )
                    if not isinstance(source, AbstractCurve):
                        continue
                    carrier = explicit_carrier(source)
                    affine = affine_curve_coefficients(
                        carrier, np.empty((2,), dtype=np.float64)
                    )
                    if affine is None:
                        continue
                    direction = affine[1]
                    for axis, period in enumerate(periods):
                        unit = tuple(
                            Fraction(int(component == axis)) for component in range(2)
                        )
                        if period == "two_pi" and (
                            direction == unit
                            or direction == tuple(-value for value in unit)
                        ):
                            axis_: Literal[0, 1] = 0 if axis == 0 else 1
                            candidates.append((face, axis_, staged.patch))
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
        /,
    ) -> tuple[
        tuple[tuple[RootEndpoint | None, RootEndpoint | None], ...] | None,
        tuple[tuple[RootEndpoint | None, RootEndpoint | None], ...] | None,
    ]:
        """Bind symbolic native-turn endpoints to each edge carrier and coedge p-curve."""
        if not self.native_phases:
            return None, None
        edge_roots: list[tuple[RootEndpoint | None, RootEndpoint | None]] = [
            (None, None)
        ] * len(self.edge_curves)
        coedge_roots: list[tuple[RootEndpoint | None, RootEndpoint | None]] = [
            (None, None)
        ] * len(coedge_edges)
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
                NativePeriodEndpoint(carrier, patch, axis, rational=rational, turns=0),
                NativePeriodEndpoint(carrier, patch, axis, rational=rational, turns=1),
            )
            for coedge in uses:
                pcurve = pcurves[coedge]
                if patch is None and pcurve.native_period_symbol() != "two_pi":
                    # A curve-owned full turn does not fabricate the same
                    # symbolic support for an externally rounded p-curve parameterization.
                    continue
                coedge_roots[coedge] = (
                    NativePeriodEndpoint(
                        pcurve,
                        patch,
                        axis,
                        rational=rational,
                        turns=0,
                    ),
                    NativePeriodEndpoint(
                        pcurve,
                        patch,
                        axis,
                        rational=rational,
                        turns=1,
                    ),
                )
        return tuple(edge_roots), tuple(coedge_roots)

    def _junction_window(
        self, patch: AbstractSurfacePatch, pcurve: AbstractCurve, parameter: float, /
    ) -> CurveTrimSegment:
        """A bounded source-incidence window, not a snapped scalar endpoint."""
        uv, curve_tangent = curve_value_and_derivative_host(pcurve, np.asarray(parameter))
        _, surface_differential = surface_value_and_differential_host(patch, uv)
        tangent = surface_differential @ curve_tangent
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
                    segments = [
                        CurveTrimSegment(
                            pcurves[coedge],
                            *self.edge_ranges[coedge_edges[coedge]],
                            reversed=coedge_senses[coedge] < 0,
                            first_root=uses[coedge][0],
                            last_root=uses[coedge][1],
                        )
                        for coedge in (first, second)
                    ]
                    if not segments[0].shares_endpoint(segments[1]):
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
                        pcurves[coedge],
                        SurfaceRegion(patch, resolved[face][1]),
                        *self.edge_ranges[edge],
                    )
                )
                lifts.append(
                    BRepCurveSurfaceLift(
                        carrier,
                        pcurves[coedge],
                        SurfaceRegion(
                            patch,
                            pcurves[coedge].bounding_box(window.first, window.last),
                        ),
                        window.first,
                        window.last,
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
                    (first, end, TrimRootEndpoint(root, "first")),
                    (second, start, TrimRootEndpoint(root, "second")),
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
                        parameters[endpoint] = 0.5 * (lower + upper)
                        self.edge_ranges[edge] = parameters[0], parameters[1]
                box = definition.point_enclosure()
                displacement = np.nextafter(
                    box - self.vertices[vertex],
                    np.where(box < self.vertices[vertex], -np.inf, np.inf),
                )
                incidence_bound = norm_upper(np.max(np.abs(displacement), axis=0))
                if incidence_bound > self.relative_geometric_tolerance * self.scale:
                    raise ValueError(
                        f"BRepGeometry.vertex_roots must remain within the declared vertex incidence tolerance; got vertex={vertex}, bound={incidence_bound}, limit={self.relative_geometric_tolerance * self.scale}"
                    )
                self.vertices[vertex] = np.mean(box, axis=0)
                vertices[vertex] = definition
            except ValueError as error:
                if not committed:
                    # A bounded external tolerance vertex remains a literal
                    # topology vertex unless every source-root isolation and correspondence condition
                    # succeeds. Never publish a partial or borrowed root.
                    continue
                raise refuse("inconsistent-geometry", label, str(error)) from error
        return tuple(vertices), tuple(edges), tuple(uses)

    def publish(
        self,
        *,
        coordinate_contract: SpatialCoordinateContract,
        source_id: str,
        source_format: str,
        source_digest: str,
        import_policy_id: str,
        tessellation: BRepTessellationPolicy | None,
        default_occurrences: bool,
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
        edge_roots, coedge_roots = self._native_phase_roots(pcurves, coedge_edges)
        vertex_roots, edge_roots, coedge_roots = self._bind_incidence_roots(
            pcurves,
            coedge_edges,
            coedge_senses,
            face_loops,
            resolved,
            edge_roots,
            coedge_roots,
        )
        geometry = BRepGeometry(
            vertex_points=np.asarray(self.vertices, dtype=np.float64).reshape(-1, 3),
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
            occurrences=None if default_occurrences else tuple(self.occurrences),
            assembly_containers=None
            if default_occurrences
            else tuple(self.assembly_containers),
            vertex_roots=vertex_roots,
            edge_endpoint_roots=edge_roots,
            coedge_endpoint_roots=coedge_roots,
        )
        self.scale = brep_physical_scale(geometry)
        try:
            return assemble_brep_model(
                geometry,
                tuple(face.patch for face in self.faces),
                np.asarray([box for _, box in resolved], dtype=np.float64).reshape(
                    -1, 2, 2
                ),
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
            )
        except BRepValidationError as error:
            finding = error.report.require_coedge_failure()
            reason: CadRefusalReason = (
                "limit"
                if finding.unresolved and finding.intervals_processed >= 65536
                else "unsupported-entity"
                if finding.unresolved
                else "inconsistent-geometry"
            )
            raise refuse(
                reason,
                coedge_labels[finding.coedge],
                str(error),
                (self.faces[finding.face].label, coedge_labels[finding.coedge]),
            ) from error

    def coverage(self, entity_counts: dict[str, int], /) -> CadCoverage:
        return CadCoverage(
            tuple(sorted(entity_counts.items())),
            self.pcurves_honored,
            self.pcurves_exact,
            self.pcurves_fitted,
            self.maximum_fit_error,
            self.degenerate_edges,
            self.natural_boundaries,
        )


# ------------------------------------------------- intersection approximation


def _validate_export_coedge_trim(
    trim: AbstractTrimCurve,
    coedge: int,
    geometry: BRepGeometry,
    bounds: np.ndarray,
    ranges: np.ndarray,
    face: int,
    format_name: str,
    chain: tuple[str, str],
) -> None:
    """Require chart, interval, orientation and carrier identity of one trim use."""
    label = f"coedge:{coedge}"
    edge = geometry.coedge_edges[coedge]
    if isinstance(trim, IntersectionPCurve) and np.array_equal(
        bounds[face], [[0.0, 0.0], [1.0, 1.0]]
    ):
        source = geometry.pcurves[coedge]
        if (
            isinstance(source, IntersectionPCurve)
            and trim.curve.branch_id == source.curve.branch_id
            and trim.side == source.side
            and trim.first == ranges[edge, 0]
            and trim.last == ranges[edge, 1]
            and not source.reversed
            and trim.reversed == (geometry.coedge_senses[coedge] < 0)
        ):
            return
    if not isinstance(trim, NormalizedTrimCurve) or not (
        np.array_equal(trim.lower, bounds[face, 0])
        and np.array_equal(trim.extent, bounds[face, 1] - bounds[face, 0])
    ):
        raise refuse(
            "inexact-export",
            label,
            f"NormalizedTrimCurve.lower must normalize the face parameter bounds together with extent; got trim_type={type(trim).__name__}",
            chain,
        )
    segment = trim.curve
    edge = geometry.coedge_edges[coedge]
    if not isinstance(segment, CurveTrimSegment) or not (
        isinstance(segment.curve, IntersectionPCurve)
        or (
            isinstance(segment.curve, AbstractCurve)
            and isinstance(explicit_carrier(segment.curve), _EXPLICIT_CARRIERS)
        )
    ):
        raise refuse(
            "inexact-export",
            label,
            f"CurveTrimSegment.curve must have a parameter-preserving carrier in format={format_name}; got segment_type={type(segment).__name__}",
            chain,
        )
    source = geometry.pcurves[coedge]
    if isinstance(segment.curve, IntersectionPCurve) or isinstance(
        source, IntersectionPCurve
    ):
        same_carrier = (
            isinstance(segment.curve, IntersectionPCurve)
            and isinstance(source, IntersectionPCurve)
            and segment.curve.curve.branch_id == source.curve.branch_id
            and (
                segment.curve.side,
                segment.curve.first,
                segment.curve.last,
                segment.curve.reversed,
            )
            == (source.side, source.first, source.last, source.reversed)
        )
    else:
        same_carrier = encode_geometry(segment.curve) == encode_geometry(source)
    if (
        segment.first != ranges[edge, 0]
        or segment.last != ranges[edge, 1]
        or segment.reversed != (geometry.coedge_senses[coedge] < 0)
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
    if any(
        occurrence.placement_evidence is not None
        and occurrence.placement_evidence.point_deviation_bound > 0.0
        for occurrence in geometry.occurrences
    ):
        raise refuse(
            "inexact-export",
            "placement",
            f"{format_name}.occurrences must preserve certified placement evidence; got an unsupported placement-gap export",
        )
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
        zip(
            domains,
            geometry.face_loops,
            strict=True,
        )
    ):
        if domain is None:
            raise refuse(
                "inexact-export",
                f"face:{face}",
                f"BRepModel.trim_domains must have bounded source closure for face={face}; got None",
                (f"face:{face}",),
            )
        if len(domain.loops) != len(loops):
            raise refuse(
                "inexact-export",
                f"face:{face}",
                f"TrimDomain.loops must align with the face coedge loops; got face={face}, trim_loops={len(domain.loops)}, coedge_loops={len(loops)}",
            )
        for loop_index, (trim_loop, coedges) in enumerate(
            zip(domain.loops, loops, strict=True)
        ):
            chain = (f"face:{face}", f"loop:{loop_index}")
            if isinstance(trim_loop, PolygonTrimLoop) and all(
                isinstance(geometry.pcurves[c], LineCurve) for c in coedges
            ):
                endpoints = []
                for coedge in coedges:
                    edge = geometry.coedge_edges[coedge]
                    parameter = ranges[
                        edge, 0 if geometry.coedge_senses[coedge] > 0 else 1
                    ]
                    point = curve_evaluate_host(
                        geometry.pcurves[coedge], np.asarray(parameter)
                    )
                    endpoints.append(
                        (point - bounds[face, 0]) / (bounds[face, 1] - bounds[face, 0])
                    )
                polygon = np.asarray(endpoints)
                if polygon.shape == trim_loop.vertices.shape and any(
                    np.max(np.abs(np.roll(polygon, offset, axis=0) - trim_loop.vertices))
                    <= 1.0e-12
                    for offset in range(len(endpoints))
                ):
                    continue
            if not isinstance(trim_loop, CurveTrimLoop) or len(trim_loop.curves) != len(
                coedges
            ):
                raise refuse(
                    "inexact-export",
                    f"face:{face}",
                    f"TrimDomain.loops must have a definition-preserving coedge closure in format={format_name}; got trim_loop_type={type(trim_loop).__name__}, coedges={len(coedges)}",
                    chain,
                )
            for trim, coedge in zip(trim_loop.curves, coedges, strict=True):
                _validate_export_coedge_trim(
                    trim, coedge, geometry, bounds, ranges, face, format_name, chain
                )


class _ApproximationAccumulator(StrictModule):
    """Cumulative proof consumption across accepted and rejected proposals."""

    cells: int = 0
    jets: int = 0
    peak_bytes: int = 0
    depth: int = 0
    accepted: (
        tuple[
            tuple[BSplineCurve, BSplineCurve, BSplineCurve],
            CoupledApproximationEvidence,
            BranchApproximationTopologyEvidence,
        ]
        | None
    ) = None

    def consume(
        self,
        cells: int,
        jets: int,
        peak_bytes: int,
        depth: int,
    ) -> _ApproximationAccumulator:
        return _ApproximationAccumulator(
            cells=self.cells + cells,
            jets=self.jets + jets,
            peak_bytes=max(self.peak_bytes, peak_bytes),
            depth=max(self.depth, depth),
            accepted=self.accepted,
        )


def _certify_approximation_candidate(
    curve: IntersectionCurve,
    spline: BSplineCurve,
    policy: CadCurveFitPolicy,
    scale: float,
    parameter_scale: float,
    state: _ApproximationAccumulator,
) -> _ApproximationAccumulator:
    if state.cells >= policy.maximum_certificate_cells:
        raise ApproximationResourceError(
            "maximum_cells",
            f"CadCurveFitPolicy.maximum_certificate_cells must exceed consumed continuous-proof cells; got consumed={state.cells}, limit={policy.maximum_certificate_cells}",
            cells=state.cells,
            depth=state.depth,
            bytes_used=0,
            jet_evaluations=state.jets,
            peak_bytes=state.peak_bytes,
        )
    controls = np.asarray(spline.control_points)
    carriers = (
        BSplineCurve(controls[:, :3], spline.weights, spline.knots, 3),
        BSplineCurve(controls[:, 3:5], spline.weights, spline.knots, 3),
        BSplineCurve(controls[:, 5:7], spline.weights, spline.knots, 3),
    )
    try:
        continuous = certify_coupled_approximation(
            curve,
            *carriers,
            distance_tolerance=policy.tolerance * scale,
            parameter_tolerance=policy.tolerance * parameter_scale,
            maximum_cells=policy.maximum_certificate_cells - state.cells,
            maximum_depth=policy.maximum_certificate_depth,
            maximum_bytes=policy.maximum_certificate_bytes,
        )
    except ApproximationResourceError as exhausted:
        raise ApproximationResourceError(
            exhausted.resource,
            exhausted.cause,
            cells=state.cells + exhausted.cells,
            depth=max(state.depth, exhausted.maximum_depth_used),
            bytes_used=exhausted.bytes_used,
            jet_evaluations=state.jets + exhausted.jet_evaluations,
            peak_bytes=max(state.peak_bytes, exhausted.peak_bytes),
        ) from exhausted
    except ApproximationDeviationError as mismatch:
        return state.consume(
            mismatch.cells,
            mismatch.jet_evaluations,
            mismatch.peak_bytes,
            mismatch.maximum_depth_used,
        )
    state = state.consume(
        continuous.cells,
        continuous.jet_evaluations,
        continuous.peak_bytes,
        continuous.maximum_depth_used,
    )
    if state.cells >= policy.maximum_certificate_cells:
        raise ApproximationResourceError(
            "maximum_cells",
            f"CadCurveFitPolicy.maximum_certificate_cells must leave cells for topology proof; got consumed={state.cells}, limit={policy.maximum_certificate_cells}",
            cells=state.cells,
            depth=state.depth,
            bytes_used=continuous.bytes_used,
            jet_evaluations=state.jets,
            peak_bytes=state.peak_bytes,
        )
    remaining_bytes = policy.maximum_certificate_bytes - continuous.bytes_used
    if remaining_bytes <= 0:
        raise ApproximationResourceError(
            "maximum_bytes",
            f"CadCurveFitPolicy.maximum_certificate_bytes must leave topology workspace after retaining error evidence; got retained={continuous.bytes_used}, limit={policy.maximum_certificate_bytes}",
            cells=state.cells,
            depth=state.depth,
            bytes_used=continuous.bytes_used,
            jet_evaluations=state.jets,
            peak_bytes=state.peak_bytes,
        )
    try:
        topology = certify_branch_approximation_topology(
            curve,
            *carriers,
            maximum_cells=policy.maximum_certificate_cells - state.cells,
            maximum_depth=policy.maximum_certificate_depth,
            maximum_bytes=remaining_bytes,
        )
    except BranchApproximationTopologyResourceError as exhausted:
        raise ApproximationResourceError(
            "topology",
            exhausted.cause,
            cells=state.cells + exhausted.cells,
            depth=max(state.depth, exhausted.maximum_depth_used),
            bytes_used=continuous.bytes_used + exhausted.bytes_used,
            jet_evaluations=state.jets,
            peak_bytes=max(
                state.peak_bytes, continuous.bytes_used + exhausted.bytes_used
            ),
        ) from exhausted
    return _ApproximationAccumulator(
        cells=state.cells + topology.cells,
        jets=state.jets,
        peak_bytes=max(state.peak_bytes, continuous.bytes_used + topology.bytes_used),
        depth=max(state.depth, topology.maximum_depth_used),
        accepted=(carriers, continuous, topology),
    )


@checked
def approximate_intersection_curve(
    curve: IntersectionCurve,
    policy: CadCurveFitPolicy,
    /,
    *,
    required_parameters: Sequence[float] = (),
) -> CadCurveApproximation:
    """Fit and continuously certify a coupled source-branch substitution.

    Samples propose constrained splines. Acceptance requires native interval
    source/fit difference and surface-lift bounds, plus a continuous embedded
    straight homotopy in XYZ and both UV lifts. Failed geometric candidates
    refine within the source control budget; unresolved proof resources
    refuse immediately and never reset or relax the policy.
    """
    if not curve.fully_certified:
        raise ValueError(
            f"IntersectionCurve.fully_certified must be true before approximation; got {curve.fully_certified}"
        )
    first, last = curve.parameter_interval
    required = np.asarray(required_parameters, dtype=np.float64)
    if (
        required.ndim != 1
        or not np.all(np.isfinite(required))
        or np.any(required < first)
        or np.any(required > last)
    ):
        raise ValueError(
            f"approximate_intersection_curve.required_parameters must be finite and within the branch interval; got parameters={required!r}, interval=({first}, {last})"
        )
    cache: dict[str, np.ndarray] = {}
    anchor_result = jax.device_get(
        curve.evaluate(jnp.asarray((first,), dtype=jnp.float64))
    )
    anchor = np.concatenate(
        (
            np.asarray(anchor_result.point)[0],
            np.asarray(anchor_result.first_parameters)[0],
            np.asarray(anchor_result.second_parameters)[0],
        )
    )
    closure_axes = np.r_[
        np.ones(3, dtype=np.bool_),
        np.asarray(curve.node_period_shifts[-1])
        == np.asarray(curve.node_period_shifts[0]),
    ]

    def coupled(parameters: np.ndarray) -> np.ndarray:
        key = array_bytes_digest(np.asarray(parameters, dtype=np.float64))
        if key not in cache:
            result = jax.device_get(
                curve.evaluate(jnp.asarray(parameters, dtype=jnp.float64))
            )
            cache[key] = np.concatenate(
                (
                    np.asarray(result.point),
                    np.asarray(result.first_parameters),
                    np.asarray(result.second_parameters),
                ),
                axis=1,
            )
            if curve.closed:
                terminal = np.flatnonzero(np.asarray(parameters) == last)
                cache[key][np.ix_(terminal, np.flatnonzero(closure_axes))] = anchor[
                    closure_axes
                ]
        return cache[key]

    exact_checks = coupled(np.linspace(first, last, policy.check_samples))
    source_box = np.asarray(curve.bounding_box(first, last))
    parameter_boxes = np.asarray(curve.parameter_enclosures(first, last))
    scale = max(
        1.0, float(np.nextafter(0.5 * np.max(source_box[1] - source_box[0]), np.inf))
    )
    parameter_scale = max(1.0, float(np.max(np.abs(parameter_boxes))))
    weights = np.concatenate((np.full(3, 1.0 / scale), np.full(4, 1.0 / parameter_scale)))

    def error(fitted: BSplineCurve, parameters: np.ndarray) -> float:
        deviation = (
            curve_evaluate_host(fitted, parameters) - coupled(parameters)
        ) * weights
        return float(
            max(
                np.max(np.linalg.norm(deviation[:, :3], axis=1)),
                np.max(np.abs(deviation[:, 3:])),
            )
        )

    state = _ApproximationAccumulator()

    def accept(spline: BSplineCurve) -> bool:
        nonlocal state
        state = _certify_approximation_candidate(
            curve, spline, policy, scale, parameter_scale, state
        )
        return state.accepted is not None

    breakpoints = np.unique(
        np.r_[np.arange(curve.num_charts + 1, dtype=np.float64), required]
    )
    if 3 * (breakpoints.size - 1) + 1 > policy.maximum_control_points:
        raise ValueError(
            f"CadCurveFitPolicy.maximum_control_points must admit all required interpolation points; got required={3 * (breakpoints.size - 1) + 1}, limit={policy.maximum_control_points}"
        )
    fitted = fit_bspline(
        coupled,
        breakpoints,
        error,
        policy.tolerance,
        policy,
        accept=accept,
    )
    if fitted is None:
        raise ValueError(
            f"CadCurveFitPolicy.tolerance must admit a coupled branch spline within the control budget; got no accepted proposal at tolerance={policy.tolerance}, controls={policy.maximum_control_points}"
        )
    spline = fitted[0]
    if state.accepted is None:
        raise ValueError(
            "_ApproximationAccumulator.accepted must contain continuous and topology evidence for the spline proposal; got None"
        )
    carriers, continuous, topology = state.accepted
    checks = np.linspace(first, last, policy.check_samples)
    deviation = curve_evaluate_host(spline, checks) - exact_checks
    return CadCurveApproximation(
        curve.branch_id,
        *carriers,
        float(np.max(np.linalg.norm(deviation[:, :3], axis=1))),
        float(np.max(np.abs(deviation[:, 3:]))),
        policy.check_samples,
        policy.policy_id,
        scale,
        parameter_scale,
        continuous,
        topology,
        state.cells,
        state.jets,
        state.peak_bytes,
    )


def _certify_export_quotient_closure(
    geometry: BRepGeometry,
    source: IntersectionPCurve,
    approximation: CadCurveApproximation,
    loops: tuple[tuple[int, ...], ...],
    ranges: np.ndarray,
    face: int,
    coedge: int,
) -> None:
    """Require quotient closure when the emitted trims cover a closed native branch."""
    branch_id = source.curve.branch_id
    side_index = 0 if source.side == "first" else 1
    for loop in loops:
        members = [geometry.pcurves[item] for item in loop]
        if source.curve.closed and all(
            isinstance(item, IntersectionPCurve)
            and item.curve.branch_id == branch_id
            and item.side == source.side
            for item in members
        ):
            intervals = sorted(
                tuple(ranges[geometry.coedge_edges[item]]) for item in loop
            )
            cursor = source.curve.parameter_interval[0]
            for lower, upper in intervals:
                if lower > cursor:
                    break
                cursor = max(cursor, float(upper))
            else:
                if cursor == source.curve.parameter_interval[1] and (
                    approximation.topology_evidence.trim_closure_kinds[side_index]
                    != "closed"
                ):
                    raise refuse(
                        "inexact-export",
                        f"coedge:{coedge}",
                        f"BranchApproximationTopologyEvidence.trim_closure_kinds must certify closed quotient coverage for the emitted branch; got side={source.side}, closure={approximation.topology_evidence.trim_closure_kinds[side_index]!r}",
                        (f"face:{face}", f"intersection-curve {branch_id}"),
                    )


def _certify_export_trim_pair(
    geometry: BRepGeometry,
    source: IntersectionPCurve,
    approximation: CadCurveApproximation,
    fitted: dict[str, CadCurveApproximation],
    policy: CadCurveFitPolicy,
    ranges: np.ndarray,
    face: int,
    coedge: int,
    other: int,
) -> CadCurveApproximation:
    """Certify one exported branch against another trim within retained proof budgets."""
    other_source = geometry.pcurves[other]
    other_edge = geometry.coedge_edges[other]
    first, last = map(float, ranges[other_edge])
    if (
        isinstance(other_source, IntersectionPCurve)
        and other_source.curve.branch_id in fitted
    ):
        other_approximation = fitted[other_source.curve.branch_id]
        trim = BranchTrimHomotopyTube(
            other_approximation.topology_evidence,
            other_source.side,
            (first, last),
        )
        borrowed_bytes = other_approximation.topology_evidence.bytes_used
    else:
        roots = geometry.coedge_endpoint_roots[other]
        trim = CurveTrimSegment(
            other_source,
            first,
            last,
            first_root=roots[0],
            last_root=roots[1],
        )
        borrowed_bytes = 0
    remaining_cells = policy.maximum_certificate_cells - approximation.certification_cells
    retained_bytes = (
        approximation.continuous_evidence.bytes_used
        + approximation.topology_evidence.bytes_used
        + sum(item.bytes_used for item in approximation.trim_separation_evidence)
        + borrowed_bytes
    )
    remaining_bytes = policy.maximum_certificate_bytes - retained_bytes
    if remaining_cells <= 0 or remaining_bytes <= 0:
        raise refuse(
            "limit",
            f"coedge:{other}",
            f"CadCurveFitPolicy.maximum_certificate_cells must leave trim-separation work and maximum_certificate_bytes must leave workspace; got remaining_cells={remaining_cells}, remaining_bytes={remaining_bytes}, retained_bytes={retained_bytes}",
            (f"face:{face}", f"coedge:{coedge}"),
        )
    try:
        separation = certify_branch_trim_separation(
            approximation.topology_evidence,
            trim,
            side=source.side,
            maximum_cells=remaining_cells,
            maximum_depth=policy.maximum_certificate_depth,
            maximum_bytes=remaining_bytes,
            first=first,
            last=last,
        )
    except BranchApproximationTopologyResourceError as error:
        raise refuse(
            "limit",
            f"coedge:{other}",
            str(error),
            (f"face:{face}", f"coedge:{coedge}"),
        ) from error
    except (TypeError, ValueError) as error:
        raise refuse(
            "inexact-export",
            f"coedge:{other}",
            str(error),
            (f"face:{face}", f"coedge:{coedge}"),
        ) from error
    approximation = replace(
        approximation,
        trim_separation_evidence=(
            *approximation.trim_separation_evidence,
            separation,
        ),
        certification_cells=approximation.certification_cells + separation.cells,
        certification_peak_bytes=max(
            approximation.certification_peak_bytes,
            retained_bytes + separation.bytes_used,
        ),
    )
    return approximation


def _certify_export_face_trims(
    geometry: BRepGeometry,
    fitted: dict[str, CadCurveApproximation],
    policy: CadCurveFitPolicy,
    /,
) -> dict[str, CadCurveApproximation]:
    """Require continuous trim separation on each emitted face."""

    ranges = np.asarray(geometry.edge_ranges)
    checked: set[tuple[int, str, str, int]] = set()
    result = dict(fitted)
    for face, loops in enumerate(geometry.face_loops):
        coedges = tuple(coedge for loop in loops for coedge in loop)
        for coedge in coedges:
            source = geometry.pcurves[coedge]
            if (
                not isinstance(source, IntersectionPCurve)
                or source.curve.branch_id not in fitted
            ):
                continue
            branch_id = source.curve.branch_id
            approximation = result[branch_id]
            _certify_export_quotient_closure(
                geometry, source, approximation, loops, ranges, face, coedge
            )
            for other in coedges:
                other_source = geometry.pcurves[other]
                if (
                    isinstance(other_source, IntersectionPCurve)
                    and other_source.curve.branch_id == branch_id
                ):
                    continue  # The full branch homotopy already proves these pairs.
                key = (face, branch_id, source.side, other)
                if key in checked:
                    continue
                checked.add(key)
                approximation = _certify_export_trim_pair(
                    geometry,
                    source,
                    approximation,
                    fitted,
                    policy,
                    ranges,
                    face,
                    coedge,
                    other,
                )
                result[branch_id] = approximation
    return result


def _retain_export_source_bindings(
    geometry: BRepGeometry,
    branches: dict[str, tuple[IntersectionCurve, list[int]]],
    fitted: dict[str, CadCurveApproximation],
) -> None:
    """Retain source root definitions and physical enclosures before substitution."""
    source_error_bounds = np.asarray(geometry.vertex_evaluation_bounds)
    for branch_id, (_, edges) in branches.items():
        vertices = sorted(
            {vertex for edge in edges for vertex in geometry.edge_vertices[edge]}
        )
        rooted: list[tuple[int, BRepVertexRoot]] = []
        for vertex in vertices:
            root = geometry.vertex_roots[vertex]
            if root is not None:
                rooted.append((vertex, root))
        boxes = []
        errors = []
        for vertex, root in rooted:
            box = np.asarray(root.point_enclosure(), dtype=np.float64).copy()
            error = float(source_error_bounds[vertex])
            if (
                box.shape != (2, 3)
                or not np.all(np.isfinite(box))
                or not np.isfinite(error)
                or error < 0.0
            ):
                raise refuse(
                    "inexact-export",
                    f"vertex:{vertex}",
                    f"CadCurveApproximation.source_vertex_enclosures must be finite with nonnegative evaluation errors; got vertex={vertex}, shape={box.shape}, error={error!r}",
                    (f"intersection-curve {branch_id}",),
                )
            box.setflags(write=False)
            boxes.append((vertex, box))
            errors.append((vertex, error))
        incident_edges = {
            edge
            for edge, endpoints in enumerate(geometry.edge_vertices)
            if any(vertex in vertices for vertex in endpoints)
        }
        fitted[branch_id] = replace(
            fitted[branch_id],
            source_vertex_roots=tuple(rooted),
            source_vertex_enclosures=tuple(boxes),
            source_vertex_error_bounds=tuple(errors),
            source_edge_endpoint_roots=tuple(
                (edge, geometry.edge_endpoint_roots[edge])
                for edge in sorted(incident_edges)
            ),
            source_coedge_endpoint_roots=tuple(
                (coedge, geometry.coedge_endpoint_roots[coedge])
                for coedge, edge in enumerate(geometry.coedge_edges)
                if edge in incident_edges
            ),
        )


class _ExportGeometry(StrictModule):
    geometry: BRepGeometry
    trim_domains: tuple[TrimDomain | None, ...]


def _substitute_export_geometry(
    model: BRepModel,
    geometry: BRepGeometry,
    curves: tuple[AbstractCurve | IntersectionCurve, ...],
    pcurves: list[AbstractCurve | IntersectionPCurve],
    affected_vertices: set[int],
    format_name: str,
) -> _ExportGeometry:
    """Clear substituted root bindings and construct the export trim closure."""
    vertex_roots = tuple(
        None if vertex in affected_vertices else root
        for vertex, root in enumerate(geometry.vertex_roots)
    )
    edge_roots = tuple(
        (
            None if vertices[0] in affected_vertices else roots[0],
            None if vertices[1] in affected_vertices else roots[1],
        )
        for vertices, roots in zip(
            geometry.edge_vertices, geometry.edge_endpoint_roots, strict=True
        )
    )
    coedge_roots = tuple(
        (
            None if geometry.edge_vertices[edge][0] in affected_vertices else roots[0],
            None if geometry.edge_vertices[edge][1] in affected_vertices else roots[1],
        )
        for edge, roots in zip(
            geometry.coedge_edges, geometry.coedge_endpoint_roots, strict=True
        )
    )
    try:
        substituted = replace(
            geometry,
            curves=curves,
            pcurves=tuple(pcurves),
            vertex_roots=vertex_roots,
            edge_endpoint_roots=edge_roots,
            coedge_endpoint_roots=coedge_roots,
        )
        bounds = np.asarray(model.parameter_bounds)
        domains = []
        for face in range(len(substituted.face_loops)):
            source_domain = model.trim_domains[face]
            if source_domain is None:
                raise refuse(
                    "inexact-export",
                    f"face:{face}",
                    f"BRepModel.trim_domains must have bounded source closure for face={face}; got None",
                    (f"face:{face}",),
                )
            curve_tolerances: list[float] = []
            for loop in source_domain.loops:
                match loop:
                    case CurveTrimLoop():
                        curve_tolerances.append(loop.tolerance)
                    case PolygonTrimLoop():
                        pass
                    case DynamicCurveTrimLoop():
                        raise refuse(
                            "inexact-export",
                            f"face:{face}",
                            "Dynamic trim covers require source-owned static export closure.",
                        )
                    case _:
                        assert_never(loop)
            if not curve_tolerances:
                # Polygon trims have no chord-approximation policy to reconstruct.
                domains.append(source_domain)
                continue
            tolerance = max(curve_tolerances)
            domains.append(
                brep_trim_domain(
                    substituted,
                    face,
                    model.patches[face],
                    tolerance=tolerance,
                    parameter_bounds=bounds[face],
                )
            )
    except (TypeError, ValueError) as error:
        raise refuse("inexact-export", "branch topology", str(error)) from error
    local_domains = tuple(domains)
    validate_export_trims(model, substituted, format_name, trim_domains=local_domains)
    return _ExportGeometry(geometry=substituted, trim_domains=local_domains)


@checked
def prepare_cad_export_geometry(
    model: BRepModel,
    geometry: BRepGeometry,
    policy: CadExportPolicy,
    /,
    *,
    format_name: str = "native CAD",
) -> tuple[
    BRepGeometry, tuple[TrimDomain | None, ...], tuple[CadCurveApproximation, ...]
]:
    """Validate source closure and prepare an explicitly approximated export view.

    The source model remains unchanged. A separate approximate geometry retains
    incidence, placement and assembly records; continuous
    source/lift error, embedded homotopy and used-face trim evidence precede
    publication. Native archive preserves source definitions without approximation.
    """

    validate_export_trims(model, geometry, format_name)
    ranges = np.asarray(geometry.edge_ranges)
    branches: dict[str, tuple[IntersectionCurve, list[int]]] = {}
    for edge, index in enumerate(geometry.edge_curves):
        if index < 0:
            continue
        curve = geometry.curves[index]
        if isinstance(curve, IntersectionCurve):
            branches.setdefault(curve.branch_id, (curve, []))[1].append(edge)
    if not branches:
        return geometry, model.trim_domains, ()
    if policy.intersection_approximation is None:
        branch_id, (_, edges) = next(iter(branches.items()))
        raise refuse(
            "inexact-export",
            f"intersection-curve {branch_id}",
            f"CadExportPolicy.intersection_approximation must be present for intersection edge export; got None for branch={branch_id}, edges={edges!r}",
            (f"edge:{edges[0]}",),
        )
    fitted: dict[str, CadCurveApproximation] = {}
    for branch_id, (curve, edges) in branches.items():
        parameters = sorted({float(value) for edge in edges for value in ranges[edge]})
        try:
            fitted[branch_id] = approximate_intersection_curve(
                curve,
                policy.intersection_approximation,
                required_parameters=parameters,
            )
        except (
            ApproximationResourceError,
            BranchApproximationTopologyResourceError,
        ) as error:
            raise refuse(
                "limit",
                f"intersection-curve {branch_id}",
                str(error),
                tuple(f"edge:{edge}" for edge in edges),
            ) from error
        except ValueError as error:
            raise refuse(
                "inexact-export",
                f"intersection-curve {branch_id}",
                str(error),
                tuple(f"edge:{edge}" for edge in edges),
            ) from error
    curves = tuple(
        fitted[curve.branch_id].curve
        if isinstance(curve, IntersectionCurve) and curve.branch_id in fitted
        else curve
        for curve in geometry.curves
    )
    pcurves = list(geometry.pcurves)
    for coedge, edge in enumerate(geometry.coedge_edges):
        index = geometry.edge_curves[edge]
        if index < 0:
            continue
        branch = geometry.curves[index]
        if not isinstance(branch, IntersectionCurve):
            continue
        pcurve = geometry.pcurves[coedge]
        if not isinstance(pcurve, IntersectionPCurve) or (
            pcurve.curve.branch_id != branch.branch_id or pcurve.reversed
        ):
            raise refuse(
                "inexact-export",
                f"coedge:{coedge}",
                f"BRepGeometry.pcurves must retain the coupled same-parameter intersection trace; got coedge={coedge}, branch={branch.branch_id}, pcurve_type={type(pcurve).__name__}",
                (f"face:{geometry.coedge_faces[coedge]}", f"edge:{edge}"),
            )
        approximation = fitted[branch.branch_id]
        pcurves[coedge] = (
            approximation.first_pcurve
            if pcurve.side == "first"
            else approximation.second_pcurve
        )
    fitted = _certify_export_face_trims(
        geometry, fitted, policy.intersection_approximation
    )
    affected_edges = {edge for _, edges in branches.values() for edge in edges}
    affected_vertices = {
        vertex for edge in affected_edges for vertex in geometry.edge_vertices[edge]
    }

    _retain_export_source_bindings(geometry, branches, fitted)
    view = _substitute_export_geometry(
        model, geometry, curves, pcurves, affected_vertices, format_name
    )
    substituted, local_domains = view.geometry, view.trim_domains
    fitted = {
        branch_id: replace(
            approximation,
            source_geometry_id=geometry.geometry_id,
            export_geometry_id=substituted.geometry_id,
        )
        for branch_id, approximation in fitted.items()
    }
    return substituted, local_domains, tuple(fitted.values())


__all__ = [
    "CadCodecInvariantError",
    "CadExportTolerances",
    "CadImportToleranceEvidence",
    "build_cad_import_tolerance_evidence",
    "unrepresentable_cad_tolerance_loss",
    "build_cad_export_report",
    "certified_cad_export_tolerances",
    "CadCoverage",
    "CadCurveApproximation",
    "CadCurveFitPolicy",
    "CadExportPolicy",
    "CadExportResult",
    "CadImportPolicy",
    "CadImportResult",
    "CadInterchangeError",
    "CadRefusal",
    "CadRefusalReason",
    "approximate_intersection_curve",
]
