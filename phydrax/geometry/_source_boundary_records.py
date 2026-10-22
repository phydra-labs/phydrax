#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Immutable source-boundary distance, sample, and chart-cover evidence."""

from __future__ import annotations

import math
from typing import final, get_args, Literal, Protocol, runtime_checkable, TypeAlias

import equinox as eqx
import numpy as np
from jax.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._host_storage import immutable_host_array
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import canonical_identifier, positive_integer
from ..discretization._coordinate_budget import CoordinateEnclosureBudget
from ..typing import checked, Dim, HostFloat64, parse, Scope
from ._embedding_state import MeshCertificateFinding
from ._mesh_certificates import MeshCertificateBinding


SourceBoundSemantics: TypeAlias = Literal["certified", "sampled"]


_AffineChainCompletedMetric: TypeAlias = Literal[
    "total_work",
    "required_work",
    "coefficient_work",
    "coefficient_bytes_upper",
    "retained_coefficient_bytes",
    "native_peak_bytes",
    "identity_rows",
    "identity_key_bytes_upper",
    "embedding_obligations",
    "embedding_target",
    "embedding_source",
    "embedding_components",
    "embedding_holes",
    "embedding_boundary_edges",
    "plane_comparisons",
    "plane_mismatches",
    "zero_area_pairs",
    "plane_index_bytes",
    "exact_clip_calls",
    "native_clip_work",
    "native_primitive_counter_available",
]
_AffineChainRefusedMetric: TypeAlias = Literal[
    "refused_total_work",
    "refused_native_peak_bytes",
    "refused_coefficient_work",
    "refused_polynomial_storage",
    "refused_retained_basis",
]
AffineChainResourceMetric: TypeAlias = _AffineChainCompletedMetric | _AffineChainRefusedMetric
AffineChainResourceRole: TypeAlias = Literal["source", "target"]
_AFFINE_CHAIN_COMPLETED_METRICS = frozenset(get_args(_AffineChainCompletedMetric))
_AFFINE_CHAIN_REFUSED_METRICS = frozenset(get_args(_AffineChainRefusedMetric))


@final
class AffineChainResourceEvidence(StrictModule, NonTrainableState):
    """One exact counter owned by a chart-chain proof input and its role."""

    __strict_contract__ = True

    metric: AffineChainResourceMetric = eqx.field(static=True)
    role: AffineChainResourceRole = eqx.field(static=True)
    owner_id: str = eqx.field(static=True)
    used: int = eqx.field(static=True)
    maximum: int = eqx.field(static=True)

    def __init__(
        self,
        metric: AffineChainResourceMetric,
        role: AffineChainResourceRole,
        owner_id: str,
        used: int,
        maximum: int,
    ) -> None:
        self.metric = parse(metric, AffineChainResourceMetric, "metric")
        self.role = parse(role, AffineChainResourceRole, "role")
        self.owner_id = canonical_identifier(owner_id, "affine_chain_owner_id")
        if (
            not isinstance(used, (int, np.integer))
            or not isinstance(maximum, (int, np.integer))
            or used < 0
            or maximum < 0
        ):
            raise ValueError("AffineChainResourceEvidence counts must be nonnegative integers.")
        self.used = int(used)
        self.maximum = int(maximum)

    def as_source(self) -> AffineChainResourceEvidence:
        """Retain this exact owned counter as predecessor evidence."""
        if self.role == "source":
            return self
        return AffineChainResourceEvidence(
            self.metric, "source", self.owner_id, self.used, self.maximum,
        )


class _SourcePointDim(Dim):
    """Query or sample points of one geometry source."""


class SourceBoundaryCellDim(Dim):
    """Globally identified mesh cells carrying authored boundary sides."""


class SourceBoundarySideDim(Dim, minimum=3):
    """Three oriented local sides of a triangle."""


class SourceBoundaryOccurrenceDim(Dim, minimum=3):
    """Loop, use ordinal, and authored traversal orientation."""


class SourceAmbientDim(Dim, minimum=1):
    """Ambient dimension of geometry source points."""


@final
class SourceBoundaryDistance(StrictModule, NonTrainableState):
    """Lower/upper bounds of the distance from query points to a source boundary."""

    __strict_contract__ = True

    lower: HostFloat64[_SourcePointDim]
    upper: HostFloat64[_SourcePointDim]
    semantics: SourceBoundSemantics = eqx.field(static=True)

    def __init__(
        self, lower: ArrayLike, upper: ArrayLike, semantics: SourceBoundSemantics, /
    ) -> None:
        scope = Scope()
        low = parse(
            np.asarray(lower, dtype=np.float64),
            HostFloat64[_SourcePointDim],
            "lower",
            scope=scope,
        )
        high = parse(
            np.asarray(upper, dtype=np.float64),
            HostFloat64[_SourcePointDim],
            "upper",
            scope=scope,
        )
        if np.any(np.isnan(low)) or np.any(np.isnan(high)) or np.any(low > high):
            raise ValueError(
                f"SourceBoundaryDistance.bounds must contain non-NaN numbers with lower <= upper; got {(low, high)!r}."
            )
        self.lower = low
        self.upper = high
        self.semantics = parse(semantics, SourceBoundSemantics, "semantics")


@final
class SourceBoundarySamples(StrictModule, NonTrainableState):
    """Points covering a source boundary.

    Every source boundary point lies within ``covering_radius[k]`` of some
    sample ``points[k]``, and every sample lies within ``point_error[k]`` of the
    boundary. The coverage flag is false when the sample budget prevented coverage.
    """

    __strict_contract__ = True

    points: HostFloat64[_SourcePointDim, SourceAmbientDim]
    covering_radius: HostFloat64[_SourcePointDim]
    point_error: HostFloat64[_SourcePointDim]
    semantics: SourceBoundSemantics = eqx.field(static=True)
    complete: bool = eqx.field(static=True)

    def __init__(
        self,
        points: ArrayLike,
        covering_radius: ArrayLike,
        point_error: ArrayLike,
        semantics: SourceBoundSemantics,
        /,
        *,
        complete: bool,
    ) -> None:
        scope = Scope()
        values = parse(
            np.asarray(points, dtype=np.float64),
            HostFloat64[_SourcePointDim, SourceAmbientDim],
            "points",
            scope=scope,
        )
        radius = parse(
            np.asarray(covering_radius, dtype=np.float64),
            HostFloat64[_SourcePointDim],
            "covering_radius",
            scope=scope,
        )
        error = parse(
            np.asarray(point_error, dtype=np.float64),
            HostFloat64[_SourcePointDim],
            "point_error",
            scope=scope,
        )
        if (
            not np.all(np.isfinite(values))
            or np.any(~(radius >= 0.0))
            or np.any(~(error >= 0.0))
        ):
            raise ValueError(
                f"SourceBoundarySamples.values must have finite points and nonnegative covering_radius and point_error; got {(values, radius, error)!r}."
            )
        self.points = values
        self.covering_radius = radius
        self.point_error = error
        self.semantics = parse(semantics, SourceBoundSemantics, "semantics")
        self.complete = bool(complete)


class _SourceChartCornerDim(Dim, minimum=3):
    """Three corners of an affine source chart proxy."""


@final
class SourceBoundaryChartCover(StrictModule, NonTrainableState):
    """Independent continuous source cover by affine chart image triangles.

    Certified semantics require every oriented chart/trim chain component and a
    two-sided pointwise interpolation bound, not sampled nodal residuals.
    Collapsed triangles are retained: their source wedges still need a bound.
    """

    __strict_contract__ = True

    simplices: HostFloat64[_SourcePointDim, _SourceChartCornerDim, SourceAmbientDim]
    deviation_bounds: HostFloat64[_SourcePointDim]
    semantics: SourceBoundSemantics = eqx.field(static=True)
    complete: bool = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    findings: tuple[MeshCertificateFinding, ...]
    resource_counts: tuple[tuple[str, int | np.integer, int | np.integer], ...] = (
        eqx.field(static=True)
    )
    affine_chain_resources: tuple[AffineChainResourceEvidence, ...]
    restriction_ids: tuple[str, ...] = eqx.field(static=True)
    cover_id: str = eqx.field(static=True)

    def __init__(
        self,
        simplices: ArrayLike,
        deviation_bounds: ArrayLike,
        semantics: SourceBoundSemantics,
        complete: bool,
        source_id: str,
        source_revision: str,
        *,
        findings: tuple[MeshCertificateFinding, ...] = (),
        resource_counts: tuple[tuple[str, int | np.integer, int | np.integer], ...] = (),
        affine_chain_resources: tuple[AffineChainResourceEvidence, ...] = (),
        restriction_ids: tuple[str, ...] = (),
    ) -> None:
        scope = Scope()
        values = parse(
            np.asarray(simplices, dtype=np.float64),
            HostFloat64[_SourcePointDim, _SourceChartCornerDim, SourceAmbientDim],
            "simplices",
            scope=scope,
        )
        bounds = parse(
            np.asarray(deviation_bounds, dtype=np.float64),
            HostFloat64[_SourcePointDim],
            "deviation_bounds",
            scope=scope,
        )
        if values.shape[1] != 3:
            raise ValueError(
                f"SourceBoundaryChartCover.simplices.shape must have three corners per simplex; got {values.shape!r}."
            )
        if not np.all(np.isfinite(values)) or np.any(~(bounds >= 0.0)):
            raise ValueError(
                f"SourceBoundaryChartCover.values must have finite simplices and nonnegative deviation_bounds; got {(values, bounds)!r}."
            )
        self.semantics = parse(semantics, SourceBoundSemantics, "semantics")
        self.complete = bool(complete)
        self.source_id = canonical_identifier(source_id, "source_id")
        self.source_revision = canonical_identifier(source_revision, "source_revision")
        if any(not isinstance(finding, MeshCertificateFinding) for finding in findings):
            raise TypeError(
                f"SourceBoundaryChartCover.findings must contain MeshCertificateFinding records; got {tuple(type(finding).__name__ for finding in findings)!r}."
            )
        if any(used < 0 or maximum < 0 for _, used, maximum in resource_counts):
            raise ValueError(
                f"SourceBoundaryChartCover.resource_counts must contain nonnegative used work and limits; got {resource_counts!r}."
            )
        if any(name.startswith("affine_chain_") for name, _, _ in resource_counts):
            raise ValueError("Affine chain metrics require structured affine_chain_resources.")
        resources = tuple(
            sorted(
                (canonical_identifier(name, "source_resource"), used, maximum)
                for name, used, maximum in resource_counts
            )
        )
        if len({name for name, _, _ in resources}) != len(resources):
            raise ValueError(
                f"SourceBoundaryChartCover.resource_counts.names must be unique; got {tuple(name for name, _, _ in resources)!r}."
            )
        for row in affine_chain_resources:
            if not isinstance(row, AffineChainResourceEvidence):
                raise TypeError("affine_chain_resources must contain AffineChainResourceEvidence.")
            AffineChainResourceEvidence(
                row.metric, row.role, row.owner_id, row.used, row.maximum,
            )
        chain_resources = tuple(
            sorted(
                affine_chain_resources,
                key=lambda row: (row.role, row.owner_id, row.metric),
            )
        )
        keys = tuple((row.role, row.owner_id, row.metric) for row in chain_resources)
        if len(set(keys)) != len(keys):
            raise ValueError("SourceBoundaryChartCover.affine_chain_resources keys must be unique.")
        if len({row.owner_id for row in chain_resources if row.role == "target"}) > 1:
            raise ValueError("SourceBoundaryChartCover target resources must have one proof owner.")
        groups: dict[tuple[AffineChainResourceRole, str], set[AffineChainResourceMetric]] = {}
        for row in chain_resources:
            groups.setdefault((row.role, row.owner_id), set()).add(row.metric)
        if any(
            metrics != _AFFINE_CHAIN_COMPLETED_METRICS
            and not metrics <= _AFFINE_CHAIN_REFUSED_METRICS
            for metrics in groups.values()
        ):
            raise ValueError(
                "Affine chain resource owners require exhaustive completed metrics "
                "or refusal metrics in one role."
            )
        if len({owner for _, owner in groups}) != len(groups):
            raise ValueError(
                "Affine chain resource owners cannot occupy both source and target roles."
            )
        self.affine_chain_resources = chain_resources
        self.findings = tuple(findings)
        self.resource_counts = resources
        restrictions = tuple(
            canonical_identifier(value, "chart_restriction_id")
            for value in restriction_ids
        )
        if len(set(restrictions)) != len(restrictions):
            raise ValueError(
                f"SourceBoundaryChartCover.restriction_ids must be unique; got {restrictions!r}."
            )
        self.restriction_ids = restrictions
        self.simplices = immutable_host_array(values)
        self.deviation_bounds = immutable_host_array(bounds)
        self.cover_id = canonical_fingerprint(
            {
                "kind": "source-boundary-chart-cover",
                "source": self.source_id,
                "revision": self.source_revision,
                "arrays": array_tree_fingerprint((self.simplices, self.deviation_bounds)),
                "semantics": self.semantics,
                "complete": self.complete,
                "findings": tuple(finding.finding_id for finding in self.findings),
                "resources": resources,
                "affine_chain_resources": tuple(
                    (row.role, row.owner_id, row.metric, row.used, row.maximum)
                    for row in chain_resources
                ),
                "restrictions": restrictions,
            }
        )


@runtime_checkable
class SourceBoundaryChartQuery(Protocol):
    """A source's independently certified chart chain and Taylor enclosures."""

    def boundary_chart_cover(
        self,
        maximum_patches: int,
        /,
        *,
        budget: CoordinateEnclosureBudget,
    ) -> SourceBoundaryChartCover: ...


@runtime_checkable
class SourceBoundaryQuery(Protocol):
    """Geometry-owned boundary queries of one source revision."""

    @property
    def source_id(self) -> str: ...

    @property
    def source_revision(self) -> str: ...

    @property
    def ambient_dimension(self) -> int: ...

    def boundary_distance(self, points: np.ndarray, /) -> SourceBoundaryDistance: ...

    def boundary_samples(self, maximum_samples: int, /) -> SourceBoundarySamples: ...


@final
class CadBoundaryClosureEvidence(StrictModule, NonTrainableState):
    """Retain a bounded CAD closure without replacing its authored source.

    Each row binds an original patch/chart cell to a target facet and its
    certified original-face/edge-to-representative displacement. Distinct
    target restrictions may legitimately repeat an original chart row.
    ``original_gap_allowance`` is in the binding's physical length units.
    The original source, closed representative and target homotopy have
    separate identities; none is an alias for an exact zero-gap assertion.
    ``homotopy_distance_upper`` and ``homotopy_piece_count`` are the concrete
    outputs of the complete interval proof. This record is a serializable
    report, not a capability: only its independent issuer can authorize reuse,
    and restoring/reconstructing an equal report does not issue that proof.
    """

    binding: MeshCertificateBinding
    authority_id: str = eqx.field(static=True)
    source_scope_id: str = eqx.field(static=True)
    representative_id: str = eqx.field(static=True)
    homotopy_id: str = eqx.field(static=True)
    homotopy_distance_upper: float = eqx.field(static=True)
    homotopy_piece_count: int = eqx.field(static=True)
    chart_indices: np.ndarray
    target_facet_ids: np.ndarray
    gap_bounds: np.ndarray
    original_gap_allowance: np.ndarray
    evidence_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        binding: MeshCertificateBinding,
        *,
        authority_id: str,
        source_scope_id: str,
        representative_id: str,
        homotopy_id: str,
        homotopy_distance_upper: float,
        homotopy_piece_count: int,
        chart_indices: ArrayLike,
        target_facet_ids: ArrayLike,
        gap_bounds: ArrayLike,
        original_gap_allowance: ArrayLike,
    ) -> None:
        charts, facets = np.asarray(chart_indices), np.asarray(target_facet_ids)
        gaps = np.asarray(gap_bounds, dtype=np.float64)
        allowance = np.asarray(original_gap_allowance, dtype=np.float64)
        if (
            charts.ndim != 2
            or charts.shape[1] != 2
            or charts.dtype.kind not in "iu"
            or facets.ndim != 1
            or facets.dtype.kind not in "iu"
            or not facets.size
            or charts.shape[0] != facets.size
            or gaps.shape != facets.shape
            or np.any(charts < 0)
            or np.any(facets < 0)
            or np.any(charts > np.iinfo(np.int64).max)
            or np.any(facets > np.iinfo(np.int64).max)
            or np.unique(facets).size != facets.size
        ):
            raise ValueError(
                "CadBoundaryClosureEvidence.rows must bind nonnegative chart pairs "
                "to distinct target facets with aligned gap bounds"
            )
        if (
            not np.all(np.isfinite(gaps))
            or np.any(gaps < 0)
            or allowance.shape != ()
            or not np.isfinite(allowance)
            or allowance < 0
            or not binding.source_id
            or not binding.source_revision
        ):
            raise ValueError(
                "CadBoundaryClosureEvidence.bounds must be finite nonnegative "
                "physical bounds on an identified source revision"
            )
        if np.any(gaps > allowance):
            raise ValueError(
                "CadBoundaryClosureEvidence.gap_bounds must not exceed the "
                "original physical gap allowance"
            )
        self.binding = binding
        self.authority_id = canonical_identifier(authority_id, "authority_id")
        self.source_scope_id = canonical_identifier(source_scope_id, "source_scope_id")
        self.representative_id = canonical_identifier(representative_id, "representative_id")
        self.homotopy_id = canonical_identifier(homotopy_id, "homotopy_id")
        distance = float(homotopy_distance_upper)
        pieces = positive_integer(homotopy_piece_count, "homotopy_piece_count")
        if not math.isfinite(distance) or distance < 0 or distance < float(np.max(gaps)):
            raise ValueError(
                "CadBoundaryClosureEvidence.homotopy_distance_upper must retain a "
                "finite whole-boundary bound including the authored gap"
            )
        self.homotopy_distance_upper = distance
        self.homotopy_piece_count = pieces
        if self.representative_id in (
            binding.source_id, binding.source_revision, self.authority_id,
            self.source_scope_id,
        ):
            raise ValueError(
                "CadBoundaryClosureEvidence.representative_id must identify the "
                "derived representative separately from the authored source"
            )
        self.chart_indices = np.asarray(charts, dtype=np.int64)
        self.target_facet_ids = np.asarray(facets, dtype=np.int64)
        self.gap_bounds = gaps
        self.original_gap_allowance = allowance
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "cad-boundary-closure-evidence",
                "binding": binding.binding_id,
                "authority": self.authority_id,
                "source_scope": self.source_scope_id,
                "representative": self.representative_id,
                "homotopy": self.homotopy_id,
                "homotopy_distance_upper": distance,
                "homotopy_piece_count": pieces,
                "arrays": array_tree_fingerprint(
                    (
                        self.chart_indices, self.target_facet_ids,
                        self.gap_bounds, self.original_gap_allowance,
                    )
                ),
            }
        )

    @property
    def gap_upper(self) -> float:
        return float(np.max(self.gap_bounds))

    def to_record(self) -> dict[str, object]:
        """Expose every retained numerical row with its source/target identities."""
        value = type(self).from_stored_fields(self)
        return {
            "evidence_id": value.evidence_id,
            "binding_id": value.binding.binding_id,
            "source_id": value.binding.source_id,
            "source_revision": value.binding.source_revision,
            "authority_id": value.authority_id,
            "source_scope_id": value.source_scope_id,
            "representative_id": value.representative_id,
            "homotopy_id": value.homotopy_id,
            "homotopy_distance_upper": value.homotopy_distance_upper,
            "homotopy_piece_count": value.homotopy_piece_count,
            "gap_upper": value.gap_upper,
            "original_gap_allowance": float(value.original_gap_allowance),
            "chart_rows": len(value.gap_bounds),
            "chart_indices": value.chart_indices.tolist(),
            "target_facet_ids": value.target_facet_ids.tolist(),
            "gap_bounds": value.gap_bounds.tolist(),
        }

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[CadBoundaryClosureEvidence], value: CadBoundaryClosureEvidence, /
    ) -> CadBoundaryClosureEvidence:
        renewed = cls(
            value.binding,
            authority_id=value.authority_id,
            source_scope_id=value.source_scope_id,
            representative_id=value.representative_id,
            homotopy_id=value.homotopy_id,
            homotopy_distance_upper=value.homotopy_distance_upper,
            homotopy_piece_count=value.homotopy_piece_count,
            chart_indices=value.chart_indices,
            target_facet_ids=value.target_facet_ids,
            gap_bounds=value.gap_bounds,
            original_gap_allowance=value.original_gap_allowance,
        )
        if renewed.evidence_id != value.evidence_id:
            raise ValueError(
                "CadBoundaryClosureEvidence.evidence_id must bind all retained "
                "source, target, representative and physical gap fields"
            )
        return renewed
