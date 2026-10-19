#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Immutable source-boundary distance, sample, and chart-cover evidence."""

from __future__ import annotations

from typing import final, Literal, Protocol, runtime_checkable, TypeAlias

import equinox as eqx
import numpy as np
from jax.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import canonical_identifier
from ..discretization._coordinate_budget import CoordinateEnclosureBudget
from ..typing import Dim, HostFloat64, parse, Scope
from ._embedding_state import MeshCertificateFinding


SourceBoundSemantics: TypeAlias = Literal["certified", "sampled"]


class _SourcePointDim(Dim):
    """Query or sample points of one geometry source."""


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
        self.simplices = values
        self.deviation_bounds = bounds
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
