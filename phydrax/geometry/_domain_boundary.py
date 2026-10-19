#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Boundary query preparation for parametric and embedded planar CAD domains.

Mesh certificate checks
-----------------------
[
  {
    "checks": [
      "source_chart_cover_capacity"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "The cumulative triangle and boundary-arc count does not exceed maximum_patches before querying the next source patch."
      }
    ],
    "conclusion": "Exceeding the requested source-chart work allowance makes the chart cover incomplete and unresolved.",
    "citation": {
      "kind": "definition",
      "reference": "Source boundary chart-cover allowance",
      "locator": "phydrax.geometry._domain_boundary.MeshingDomainBoundarySource.boundary_chart_cover",
      "url": ""
    },
    "failure_statuses": [
      "unresolved"
    ]
  },
  {
    "checks": [
      "source_trim_coverage_premise"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "Each selected patch occurs once; chart triangles have positive orientation, their interior edge chains cancel, and their remaining edge chain equals the simple outer trim and clockwise source holes."
      },
      {
        "tag": "N",
        "statement": "Continuous source/chord trim ribbons are bounded and disjoint between distinct loops."
      },
      {
        "tag": "T",
        "statement": "Local positive degrees sum to the boundary degree, which is unchanged under a homotopy avoiding the tested complement point."
      }
    ],
    "conclusion": "A missing, repeated, or unproved source chart chain leaves the selected patch cover incomplete and unresolved.",
    "citation": {
      "kind": "mathematical",
      "reference": "Allen Hatcher, Algebraic Topology (2002)",
      "locator": "Section 2.2, Degree, pp. 134-135 and Proposition 2.30 (local degree sum)",
      "url": "https://pi.math.cornell.edu/~hatcher/AT/AT.pdf"
    },
    "failure_statuses": [
      "unresolved"
    ]
  }
]
End mesh certificate checks
"""

from __future__ import annotations

from typing import Callable, final

import equinox as eqx
import numpy as np

import phydrax.geometry._domain_chart_chain as domain_chart_chain_owner
import phydrax.geometry._domain_numerics as domain_numerics_owner
import phydrax.geometry._embedding_state as embedding_state_owner
import phydrax.geometry._meshing_domain as domain_owner
import phydrax.geometry._source_boundary_records as source_boundary_records_owner

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._validation import nonnegative_integer
from ..discretization._coordinate_budget import CoordinateEnclosureBudget
from ..typing import checked
from ._atlas import TrimDomain
from ._chart_restriction import validate_chart_restrictions
from .brep._coordinate_request import coordinate_source_request
from .brep._patches import (
    BernsteinRestrictionBank,
    MeridianOverlaps,
    shared_meridian_jets,
    source_bernstein_restriction_scope,
)
from .brep._request_preparations import TrimRequestPreparations


_SEED_BLOCK = 256
from ._source_boundary_records import SourceBoundaryDistance, SourceBoundarySamples


@final
class _PreparedBoundaryDistanceGrid(StrictModule):
    """One immutable sampled query grid shared by a whole certification."""

    source: MeshingDomainBoundarySource
    rows: np.ndarray
    charts: np.ndarray
    samples: np.ndarray
    radius: float
    trims: tuple[TrimDomain, ...]

    @property
    def source_id(self) -> str:
        return self.source.source_id

    @property
    def source_revision(self) -> str:
        return self.source.source_revision

    @property
    def ambient_dimension(self) -> int:
        return self.source.ambient_dimension

    def boundary_distance(
        self, points: np.ndarray, /, *, budget: CoordinateEnclosureBudget | None = None
    ) -> source_boundary_records_owner.SourceBoundaryDistance:

        queries = np.asarray(points, dtype=np.float64).reshape((-1, 3))
        nearest = np.concatenate(
            [
                np.argmin(
                    np.linalg.norm(
                        queries[start : start + _SEED_BLOCK, None] - self.samples[None],
                        axis=-1,
                    ),
                    axis=1,
                )
                for start in range(0, queries.shape[0], _SEED_BLOCK)
            ]
            + [np.zeros((0,), dtype=np.int64)]
        )
        seed = np.linalg.norm(queries - self.samples[nearest], axis=1)
        projection = self.source.domain.project(
            queries, self.rows[nearest], self.charts[nearest], budget=budget
        )
        inside = np.zeros((queries.shape[0],), dtype=np.bool_)
        for patch, trim in zip(self.source.patches, self.trims, strict=True):
            selected = (self.rows[nearest] == patch) & projection.converged
            if np.any(selected):
                classification = trim.classify(projection.parameters[selected])
                inside[selected] = classification.inside & classification.resolved
        distance = np.where(
            projection.converged & inside,
            np.minimum(projection.distances, seed),
            seed,
        )
        return source_boundary_records_owner.SourceBoundaryDistance(
            distance, distance, "sampled"
        )

    def boundary_samples(
        self, maximum_samples: int, /, *, budget: CoordinateEnclosureBudget | None = None
    ) -> source_boundary_records_owner.SourceBoundarySamples:

        count = self.source._sampling_capacity()
        if count > maximum_samples:
            empty = np.empty((0,), dtype=np.float64)
            return source_boundary_records_owner.SourceBoundarySamples(
                np.empty((0, 3), dtype=np.float64),
                empty,
                empty,
                "sampled",
                complete=False,
            )
        size = self.samples.shape[0]
        return source_boundary_records_owner.SourceBoundarySamples(
            self.samples,
            np.full((size,), self.radius, dtype=np.float64),
            np.zeros((size,), dtype=np.float64),
            "sampled",
            complete=True,
        )


type SourceChartTriangulation = tuple[
    int,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]


@final
class _PhysicalChartBounds(StrictModule):
    """Cell images and continuous source deviation after trim-chain admission."""

    images: np.ndarray
    deviations: np.ndarray


@final
class MeshingDomainBoundarySource(StrictModule):
    """Sampled boundary queries of the selected surfaces of one meshing domain.

    Each patch retains authored trim curves and samples its boundary chains as
    well as a ``resolution x resolution`` interior chart grid. native trim
    classification excludes holes and points outside curved boundaries.
    Distances use a converged source projection only inside the authored trim;
    otherwise they retain the nearest sampled boundary/interior point. The
    covering radius is a sampled cell/chain half-diagonal estimate. Distances
    and radii remain ``sampled`` estimates, not certified bounds.

    With an explicit chart chain, ``boundary_chart_cover`` instead verifies
    oriented coverage and returns continuous source enclosures. A whole
    sphere may use convex radial-shell/plane bounds only after an independent
    degree-one radial-chain check; a same-UV interpolation residual is
    neither necessary nor substituted for the requested Euclidean deviation.
    """

    domain: domain_owner.MeshingDomain
    patches: tuple[int, ...] = eqx.field(static=True)
    resolution: int = eqx.field(static=True)
    chart_triangulations: tuple[
        tuple[
            int,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ],
        ...,
    ]

    @checked
    def __init__(
        self,
        domain: domain_owner.MeshingDomain,
        patches: tuple[int, ...],
        /,
        *,
        resolution: int = 48,
        chart_triangulations: tuple[
            tuple[
                int,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
            ],
            ...,
        ] = (),
    ) -> None:
        selected = tuple(
            sorted({nonnegative_integer(patch, "patch") for patch in patches})
        )
        if not selected or selected[-1] >= len(domain.patches):
            raise ValueError(
                "MeshingDomainBoundarySource.patches must select present domain surfaces and be nonempty; got empty or out-of-range patch indices"
            )
        count = nonnegative_integer(resolution, "resolution")
        if count < 2:
            raise ValueError(
                f"MeshingDomainBoundarySource.resolution must be at least two; got {count}"
            )
        self.domain = domain
        self.patches = selected
        self.resolution = count
        self.chart_triangulations = chart_triangulations

    @property
    def source_id(self) -> str:
        return self.domain.source_id

    @property
    def source_revision(self) -> str:
        return self.domain.source_revision

    @property
    def source_scope_id(self) -> str:
        """Bind the source patch subset, including occurrence-qualified IDs."""
        indices = self.domain.scope_indices(2)
        return canonical_fingerprint(
            {
                "kind": "meshing-domain-boundary-source-scope",
                "domain": self.domain.domain_id,
                "entity_set": self.domain.entity_set_id(2),
                "entity_ids": tuple(indices[patch] for patch in self.patches),
            }
        )

    @property
    def ambient_dimension(self) -> int:
        return 3

    def _sampling_capacity(self) -> int:
        """Bound grid and authored boundary candidates before source evaluation."""
        return len(self.patches) * self.resolution * self.resolution + max(
            3, self.resolution - 1
        ) * sum(
            len(loop)
            for patch in self.patches
            for loop in self.domain.patches[patch].loops
        )

    def _grid(
        self,
        *,
        budget: CoordinateEnclosureBudget,
        preparations: TrimRequestPreparations | None = None,
        restriction_bank: BernsteinRestrictionBank | None = None,
        meridian_jets: list[MeridianOverlaps] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, tuple[TrimDomain, ...]]:
        """Build inside-chart and boundary samples with trim membership."""
        rows: list[np.ndarray] = []
        charts: list[np.ndarray] = []
        images: list[np.ndarray] = []
        trims: list[TrimDomain] = []
        radius = 0.0
        intervals = max(3, self.resolution - 1)
        for patch in self.patches:
            source = self.domain.patches[patch]
            trim = domain_chart_chain_owner.sampling_trim(
                source,
                intervals,
                self.domain.tolerance,
                budget=budget,
                preparations=preparations,
                restriction_bank=restriction_bank,
                meridian_jets=meridian_jets,
            )
            trims.append(trim)
            lower = np.min(trim.outer.chords, axis=0)
            upper = np.max(trim.outer.chords, axis=0)
            axes = [
                np.linspace(lower[k], upper[k], self.resolution, dtype=np.float64)
                for k in range(2)
            ]
            grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
            flat = grid.reshape((-1, 2))
            grid_points = self.domain.evaluate(
                np.full((flat.shape[0],), patch, dtype=np.int64), flat, budget=budget
            ).reshape(grid.shape[:2] + (3,))
            diagonal = np.linalg.norm(
                grid_points[1:, 1:] - grid_points[:-1, :-1], axis=-1
            )
            radius = max(radius, 0.5 * float(np.max(diagonal)))
            classification = trim.classify(flat)
            inside = classification.inside & classification.resolved
            rows.append(np.full((np.count_nonzero(inside),), patch, dtype=np.int64))
            charts.append(flat[inside])
            images.append(grid_points.reshape((-1, 3))[inside])
            for loop in source.loops:
                boundary_charts = []
                for use in loop:
                    first, last = (
                        (use.first, use.last)
                        if isinstance(use, domain_owner.PatchCurveUse)
                        else (0.0, 1.0)
                    )
                    boundary_charts.append(
                        use.charts(
                            np.linspace(first, last, intervals + 1, dtype=np.float64)
                        )[:-1]
                    )
                boundary = np.concatenate(boundary_charts, axis=0)
                boundary_rows = np.full((boundary.shape[0],), patch, dtype=np.int64)
                points = self.domain.evaluate(boundary_rows, boundary, budget=budget)
                radius = max(
                    radius,
                    0.5
                    * float(
                        np.max(
                            np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
                        )
                    ),
                )
                rows.append(boundary_rows)
                charts.append(boundary)
                images.append(points)
        return (
            np.concatenate(rows),
            np.concatenate(charts, axis=0),
            np.concatenate(images, axis=0),
            radius,
            tuple(trims),
        )

    def prepare_boundary_queries(
        self,
        *,
        budget: CoordinateEnclosureBudget | None = None,
        preparations: TrimRequestPreparations | None = None,
        restriction_bank: BernsteinRestrictionBank | None = None,
        meridian_jets: list[MeridianOverlaps] | None = None,
    ) -> _PreparedBoundaryDistanceGrid:
        """Prepare one immutable distance/sample artifact for a certification."""
        with (
            coordinate_source_request(
                (self.domain, self.chart_triangulations, self.resolution),
                rows=self._sampling_capacity(),
                queries_upper=2 * (self._sampling_capacity()),
                budget=budget,
            ) as budget,
            source_bernstein_restriction_scope(
                budget=budget, restriction_bank=restriction_bank
            ) as restriction_bank,
            shared_meridian_jets(meridian_jets=meridian_jets) as meridian_jets,
        ):
            rows, charts, samples, radius, trims = self._grid(
                budget=budget,
                preparations=preparations,
                restriction_bank=restriction_bank,
                meridian_jets=meridian_jets,
            )
            return _PreparedBoundaryDistanceGrid(
                self, rows, charts, samples, radius, trims
            )

    def boundary_distance(
        self,
        points: np.ndarray,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
        preparations: TrimRequestPreparations | None = None,
        restriction_bank: BernsteinRestrictionBank | None = None,
        meridian_jets: list[MeridianOverlaps] | None = None,
    ) -> SourceBoundaryDistance:
        with (
            coordinate_source_request(
                (self.domain, self.chart_triangulations, self.resolution),
                rows=self._sampling_capacity() + points.shape[0],
                queries_upper=2 * (self._sampling_capacity() + points.shape[0]),
                budget=budget,
            ) as budget,
            source_bernstein_restriction_scope(
                budget=budget, restriction_bank=restriction_bank
            ) as restriction_bank,
            shared_meridian_jets(meridian_jets=meridian_jets) as meridian_jets,
        ):
            return self.prepare_boundary_queries(
                budget=budget,
                preparations=preparations,
                restriction_bank=restriction_bank,
                meridian_jets=meridian_jets,
            ).boundary_distance(points, budget=budget)

    def boundary_samples(
        self,
        maximum_samples: int,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
        preparations: TrimRequestPreparations | None = None,
        restriction_bank: BernsteinRestrictionBank | None = None,
        meridian_jets: list[MeridianOverlaps] | None = None,
    ) -> SourceBoundarySamples:
        with (
            coordinate_source_request(
                (self.domain, self.chart_triangulations, self.resolution),
                rows=self._sampling_capacity() + maximum_samples,
                queries_upper=2 * (self._sampling_capacity() + maximum_samples),
                budget=budget,
            ) as budget,
            source_bernstein_restriction_scope(
                budget=budget, restriction_bank=restriction_bank
            ) as restriction_bank,
            shared_meridian_jets(meridian_jets=meridian_jets) as meridian_jets,
        ):
            return self.prepare_boundary_queries(
                budget=budget,
                preparations=preparations,
                restriction_bank=restriction_bank,
                meridian_jets=meridian_jets,
            ).boundary_samples(maximum_samples, budget=budget)

    def boundary_chart_cover(
        self,
        maximum_patches: int,
        /,
        *,
        reserve_queries: Callable[[int], None] | None = None,
        _topology_only: bool = False,
        budget: CoordinateEnclosureBudget | None = None,
        preparations: TrimRequestPreparations | None = None,
        restriction_bank: BernsteinRestrictionBank | None = None,
        meridian_jets: list[MeridianOverlaps] | None = None,
    ) -> source_boundary_records_owner.SourceBoundaryChartCover:
        """Independently verify a whole oriented chart chain and trims.

        This does not trust generation's deviation or coverage flags. Interior
        edges cancel exactly, chart triangles have positive orientation,
        and the remaining chain must equal the declared, simple trim boundary.
        General source trims include interval-bounded chord homotopy ribbons.
        Collapsed pole charts remain as degenerate physical simplices.
        """
        with (
            coordinate_source_request(
                (self.domain, self.chart_triangulations, self.resolution),
                rows=maximum_patches,
                queries_upper=maximum_patches * (200_000 + 1),
                budget=budget,
            ) as budget,
            source_bernstein_restriction_scope(
                budget=budget, restriction_bank=restriction_bank
            ) as restriction_bank,
            shared_meridian_jets(meridian_jets=meridian_jets) as meridian_jets,
        ):
            if not isinstance(_topology_only, bool):
                raise TypeError(
                    f"MeshingDomainBoundarySource.topology_only must be boolean; got {type(_topology_only).__name__}"
                )

            images, bounds = [], []
            findings: list[embedding_state_owner.MeshCertificateFinding] = []
            resource_counts: list[tuple[str, int, int]] = []
            restriction_ids: list[str] = []
            complete = bool(self.chart_triangulations)
            seen = []
            work = 0
            for data in self.chart_triangulations:
                (
                    patch,
                    charts,
                    points,
                    cells,
                    boundary,
                    provenance,
                    restriction_required,
                    restriction_vertices,
                    restriction_edges,
                    restriction_parameters,
                ) = data
                if patch not in self.patches:
                    continue
                work += cells.shape[0] + boundary.shape[0]
                if work > maximum_patches or patch in seen:
                    complete = False
                    findings.append(
                        embedding_state_owner.MeshCertificateFinding(
                            "source_chart_cover_capacity"
                            if work > maximum_patches
                            else "source_trim_coverage_premise",
                            "unresolved",
                            "source_facet",
                            (patch,),
                        )
                    )
                    break
                seen.append(patch)
                valid, ribbon_images, ribbon_bounds, patch_restriction_ids = (
                    domain_chart_chain_owner.verify_chart_chain(
                        self.domain,
                        patch,
                        charts,
                        points,
                        cells,
                        boundary,
                        provenance,
                        restriction_required,
                        restriction_vertices,
                        restriction_edges,
                        restriction_parameters,
                        findings,
                        resource_counts,
                        topology_only=_topology_only,
                        budget=budget,
                        preparations=preparations,
                        restriction_bank=restriction_bank,
                        meridian_jets=meridian_jets,
                    )
                )
                restriction_ids.extend(patch_restriction_ids)
                complete &= valid
                if not valid:
                    findings.append(
                        embedding_state_owner.MeshCertificateFinding(
                            "source_trim_coverage_premise",
                            "unresolved",
                            "source_facet",
                            (patch,),
                        )
                    )
                    continue
                if _topology_only:
                    images.append(points[cells])
                    bounds.append(np.zeros((cells.shape[0],), dtype=np.float64))
                    continue
                physical = _physical_chart_bounds(
                    self,
                    data,
                    ribbon_bounds,
                    reserve_queries,
                    budget=budget,
                    preparations=preparations,
                    restriction_bank=restriction_bank,
                    meridian_jets=meridian_jets,
                )
                deviation = physical.deviations
                complete &= bool(np.all(np.isfinite(deviation)))
                # Curved trim uncertainty is already included in every physical
                # source simplex. Only authored collapsed pole wedges remain as
                # measure-zero source support; ordinary chord placeholders are not
                # independent facets.
                images.append(physical.images)
                bounds.append(deviation)
                if any(
                    isinstance(use, domain_owner.PatchPoleUse)
                    for loop in self.domain.patches[patch].loops
                    for use in loop
                ):
                    images.append(ribbon_images)
                    bounds.append(ribbon_bounds)
            complete &= sorted(seen) == list(self.patches)
            return source_boundary_records_owner.SourceBoundaryChartCover(
                np.concatenate(images) if images else np.empty((0, 3, 3)),
                np.concatenate(bounds) if bounds else np.empty((0,)),
                "certified" if complete else "sampled",
                complete=complete,
                source_id=self.source_id,
                source_revision=self.source_revision,
                findings=tuple(findings),
                resource_counts=tuple(resource_counts),
                restriction_ids=tuple(restriction_ids),
            )


def _physical_chart_bounds(
    source: MeshingDomainBoundarySource,
    data: SourceChartTriangulation,
    ribbon_bounds: np.ndarray,
    reserve_queries: Callable[[int], None] | None,
    *,
    budget: CoordinateEnclosureBudget,
    preparations: TrimRequestPreparations | None,
    restriction_bank: BernsteinRestrictionBank | None,
    meridian_jets: list[MeridianOverlaps] | None,
) -> _PhysicalChartBounds:
    """Bound nodal, rational-chart, interpolation, and trim-ribbon source error."""
    (
        patch,
        charts,
        points,
        cells,
        boundary,
        _,
        restriction_required,
        restriction_vertices,
        restriction_edges,
        restriction_parameters,
    ) = data
    coordinates = charts[cells]
    source_nodes = source.domain.evaluate(
        np.full((coordinates.size // 2,), patch),
        coordinates.reshape((-1, 2)),
        budget=budget,
    ).reshape((-1, 3, 3))
    _, _, coordinate_errors = validate_chart_restrictions(
        charts,
        restriction_required,
        restriction_vertices,
        restriction_edges,
        charts[restriction_edges]
        if restriction_vertices.size
        else np.empty((0, 2, 2), dtype=np.float64),
        restriction_parameters,
        source.source_revision,
        patch,
    )
    restriction_error = np.zeros((charts.shape[0],), dtype=np.float64)
    rows = np.flatnonzero(restriction_required)
    if rows.size:
        boxes = np.stack(
            (
                charts[rows] - coordinate_errors[rows],
                charts[rows] + coordinate_errors[rows],
            ),
            axis=1,
        )
        lower, upper = source.domain.patches[patch].surface.derivative_bounds_batch(
            boxes,
            order=1,
            budget=budget,
            preparations=preparations,
            restriction_bank=restriction_bank,
            meridian_jets=meridian_jets,
        )
        jacobian = np.linalg.norm(np.maximum(np.abs(lower), np.abs(upper)), axis=1)
        restriction_error[rows] = np.nextafter(
            np.sum(jacobian * coordinate_errors[rows], axis=1),
            np.inf,
        )
    restriction_cell_error = np.max(restriction_error[cells], axis=1)
    nodal = (
        np.max(np.linalg.norm(source_nodes - points[cells], axis=2), axis=1)
        + restriction_cell_error
    )

    deviation = (
        source.domain.interpolation_bounds(
            patch,
            coordinates,
            budget=budget,
            reserve_queries=reserve_queries,
            preparations=preparations,
            restriction_bank=restriction_bank,
            meridian_jets=meridian_jets,
        )
        + nodal
    )
    trim_cell_bound = domain_chart_chain_owner.boundary_cell_ribbon_bounds(
        cells, boundary, ribbon_bounds
    )
    deviation += trim_cell_bound
    deviation += 256 * np.finfo(np.float64).eps * source.domain.scale
    sphere_frame = domain_numerics_owner.full_sphere_frame(
        source.domain, patch, budget=budget
    )
    if sphere_frame is not None and domain_numerics_owner.sphere_radial_degree_one(
        sphere_frame, points, cells
    ):
        analytic = (
            domain_numerics_owner.sphere_triangle_distance_bounds(
                sphere_frame, points[cells]
            )
            + restriction_cell_error
            + trim_cell_bound
        )
        deviation = np.minimum(deviation, analytic)
    return _PhysicalChartBounds(points[cells], deviation)
