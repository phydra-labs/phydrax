#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Geometry-owned global-embedding, coverage and source-fidelity certificates.

These certificates are computed independently of any generator and are bound
to the certified mesh topology, the actual coordinate arrays and coordinate
element layout, the geometry source revision and the resource limits.

Global embedding (affine coordinate maps). Let every cell be certified
positively oriented (``CellValidityCertificate``), every interior facet be
shared by exactly two cells with opposite induced orientations, and the image
of the boundary facets be embedded (no contact beyond combinatorially shared
vertices, decided with exact predicates). Then the mapping degree at a point
off the boundary image equals the number of covering cells and is the winding
number of the oriented boundary. Every bounded complement component touches
some boundary shell, and across an embedded shell the degree jumps by exactly
one, so requiring degree zero on the exterior side of every shell forces degree
at most one everywhere: cell interiors are pairwise disjoint. The exterior
degree of shell ``i`` is the winding number of all other shells at a vertex
exclusive to ``i`` (exact ray parity with symbolic perturbation) plus ``-1``
when the shell's exact signed measure is negative (a cavity) and ``0``
otherwise. Surfaces and curves are certified by exact pairwise contact tests of
all cells. Canonical mapped cells use exact source-level Bernstein enclosures.
Exact coordinate sources are embedded on their authoritative rational points,
never on the correctly rounded binary64 carrier: an exact PLC source runs the
affine boundary-contact and exterior-degree proof above with exact rational
facet triangulation and the same predicate algorithms on its integer scaling,
every step charged to the request's coordinate ledger before it runs; the
source's integer-bit budget bounds the scaling before it is formed.
Local injectivity requires a fixed projected Jacobian with strictly positive
symmetric part on the convex reference domain, not determinant positivity alone.
Mapped volume meshes first run the same degree argument on the curved map
(``MappedBoundaryDegreeEvidence``): closed-cell orientation, exact identity of
every shared vertex/edge/facet restriction, injectivity of the curved boundary
facets (subdivision separation, and certified tangent-cone contacts at shared
edges and vertices) and zero exterior shell degree replace all interior cell
pair checks. An unproven premise falls back to the pairwise route below, never
to acceptance. Continuous image separation, strict adjacency-aware halfspace
bounds and a complete convex reference atlas establish global embedding there.
Interval-Newton inclusions witness overlaps and contained components. Missing
injectivity, trace continuity, denominator or resource premises remain
explicit UNRESOLVED.

Domain and interface coverage bind an independently declared source revision.
Affine planar facets use exact rational clipping against authoritative source
fragments; mapped planar facets use source-polynomial plane identities and
joint Bernstein-image hull containment in a nonoverlapping source union.
Exact projected integrals establish coverage of every original source row.
MappedReferenceDomain adds independently declared physical root maps: exact
source identity, coefficient equality, reference-chart partitions and source
physical embedding establish coverage without claiming CAD equivalence.
Physical region measures are exact coordinate-Jacobian integrals.

Source fidelity bounds the two-sided boundary deviation from a geometry source
query. A bound is ``certified`` only when the source certifies its distance
bounds (for example an exact signed distance with established evaluation
error); otherwise it is reported as ``sampled`` and cannot certify fidelity.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from contextlib import AbstractContextManager, ExitStack, nullcontext
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from typing import (
    assert_never,
    final,
    Literal,
    Protocol,
    runtime_checkable,
    TYPE_CHECKING,
    TypeAlias,
)

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

import phydrax.ein as ein

from .._bvh import bvh_host_minima, bvh_overlap_pair_blocks, prepare_bvh
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._geometry_predicates import (
    bigint_bytes,
    exact_bits,
    exact_charge,
    exact_reserve,
    orient2d,
    orient3d,
    PredicateMode,
    PredicateResult,
    resolve_host_predicate_mode,
    segment_intersections_2d,
    SegmentIntersectionStatus,
)
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import (
    canonical_identifier,
    nonnegative_integer,
    positive_integer,
    unique_identifiers,
)
from ..discretization._cell_complex import (
    PolygonalConnectivity,
    PolyhedralConnectivity,
    TetrahedralConnectivity,
)
from ..discretization._cell_geometry import (
    CellGeometrySpec,
    CellVertexGeometryElement,
    ExactCellGeometrySource,
)
from ..discretization._cell_geometry_validity import (
    cell_geometry_id,
    CellValidityCertificate,
    CellValidityStatus,
)
from ..discretization._cell_mesh import CellMesh
from ..discretization._coordinate_enclosure import (
    coordinate_enclosure_budget,
    CoordinateCoefficients,
    CoordinateEnclosureBudget,
    CoordinateEnclosureResource,
    CoordinateEnclosureResourceError,
    CoordinateSourceBank,
    Expression,
    Polynomial,
)
from ..discretization._hexahedral import HexahedralConnectivity
from ..discretization._reference_cell import reference_cell_topology
from ..typing import Dim, HostFloat64, HostInt64, parse, Scope
from ._certificate import DistanceSemantics, SignReliability, ZeroSetAccuracy
from ._certified_implicit import implicit_state_id
from ._contracts import CompiledGeometry
from .brep._patches import AbstractCurve


if TYPE_CHECKING:
    from ..discretization._cell_geometry import CellGeometryElement
    from ._mapped_reference_domain import MappedReferenceDomain


MeshCertificateStatus: TypeAlias = Literal["certified", "violated", "unresolved"]
MeshFindingStatus: TypeAlias = Literal["violated", "unresolved"]
CoordinateMapScope: TypeAlias = Literal["affine", "mapped"]
MappedBoundaryDegreeStatus: TypeAlias = Literal[
    "embedded", "violated", "premise_unproven"
]
MeshCertificateResource: TypeAlias = (
    CoordinateEnclosureResource | Literal["distance_evaluations"]
)
MeshCertificateEntityKind: TypeAlias = Literal[
    "mesh",
    "cell",
    "facet",
    "source_cell",
    "source_facet",
    "source_region",
    "source_sample",
]
SourceBoundSemantics: TypeAlias = Literal["certified", "sampled"]
CoverageOverlapSemantics: TypeAlias = Literal["exact", "candidate"]
CoverageCoordinateSpace: TypeAlias = Literal["physical", "reference"]
type _CoverageOverlapKey = tuple[
    int, int, tuple[int, ...], CoverageOverlapSemantics, CoverageCoordinateSpace
]
type _CoverageOverlap = tuple[
    int,
    int,
    tuple[int, ...],
    float,
    float,
    CoverageOverlapSemantics,
    CoverageCoordinateSpace,
]

_EPSILON = float(np.finfo(np.float64).eps)
_DISTANCE_WORKING_ENTRIES = 1 << 22


class _DomainVertexDim(Dim, minimum=1):
    """Vertices of one declared piecewise-linear domain."""


class _DomainFacetDim(Dim, minimum=1):
    """Oriented facets of one declared piecewise-linear domain."""


class _DomainAmbientDim(Dim, minimum=2):
    """Ambient dimension of a declared domain (facet arity)."""


class _SourcePointDim(Dim):
    """Query or sample points of one geometry source."""


class _SourceAmbientDim(Dim, minimum=1):
    """Ambient dimension of geometry source points."""


class _MeshCellDim(Dim, minimum=1):
    """Cells of one certified mesh."""


# Records -------------------------------------------------------------------------


@final
class MeshCertificateLimits(StrictModule, NonTrainableState):
    """Work budgets of the mesh certificates; exhaustion is reported UNRESOLVED."""

    maximum_candidate_pairs: int = eqx.field(static=True)
    maximum_ray_tests: int = eqx.field(static=True)
    maximum_source_samples: int = eqx.field(static=True)
    maximum_distance_evaluations: int = eqx.field(static=True)
    maximum_subdivision_depth: int = eqx.field(static=True)
    maximum_subdivision_pieces: int = eqx.field(static=True)
    maximum_bernstein_nodes: int = eqx.field(static=True)
    maximum_periodic_images: int = eqx.field(static=True)
    maximum_work_units: int = eqx.field(static=True)
    maximum_scratch_bytes: int = eqx.field(static=True)
    limits_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_candidate_pairs: int = 5_000_000,
        maximum_ray_tests: int = 50_000_000,
        maximum_source_samples: int = 2_000_000,
        maximum_distance_evaluations: int = 200_000_000,
        maximum_subdivision_depth: int = 12,
        maximum_subdivision_pieces: int = 1_000_000,
        maximum_bernstein_nodes: int = 4096,
        maximum_periodic_images: int = 100_000,
        maximum_work_units: int = 200_000_000,
        maximum_scratch_bytes: int = 512 * 1024**2,
    ) -> None:
        values = {
            name: positive_integer(value, name)
            for name, value in (
                ("maximum_candidate_pairs", maximum_candidate_pairs),
                ("maximum_ray_tests", maximum_ray_tests),
                ("maximum_source_samples", maximum_source_samples),
                ("maximum_distance_evaluations", maximum_distance_evaluations),
                ("maximum_subdivision_pieces", maximum_subdivision_pieces),
                ("maximum_bernstein_nodes", maximum_bernstein_nodes),
                ("maximum_periodic_images", maximum_periodic_images),
                ("maximum_work_units", maximum_work_units),
                ("maximum_scratch_bytes", maximum_scratch_bytes),
            )
        }
        if isinstance(maximum_subdivision_depth, bool) or not isinstance(
            maximum_subdivision_depth, (int, np.integer)
        ):
            raise TypeError("maximum_subdivision_depth must be an integer.")
        if maximum_subdivision_depth < 0:
            raise ValueError("maximum_subdivision_depth must be nonnegative.")
        values["maximum_subdivision_depth"] = int(maximum_subdivision_depth)
        self.maximum_candidate_pairs = values["maximum_candidate_pairs"]
        self.maximum_ray_tests = values["maximum_ray_tests"]
        self.maximum_source_samples = values["maximum_source_samples"]
        self.maximum_distance_evaluations = values["maximum_distance_evaluations"]
        self.maximum_subdivision_depth = values["maximum_subdivision_depth"]
        self.maximum_subdivision_pieces = values["maximum_subdivision_pieces"]
        self.maximum_bernstein_nodes = values["maximum_bernstein_nodes"]
        self.maximum_periodic_images = values["maximum_periodic_images"]
        self.maximum_work_units = values["maximum_work_units"]
        self.maximum_scratch_bytes = values["maximum_scratch_bytes"]
        self.limits_id = canonical_fingerprint(
            {"kind": "mesh-certificate-limits", **values}
        )


@final
class MeshCertificateFinding(StrictModule, NonTrainableState):
    """One violated or unresolved check with the exact entities it concerns.

    ``entity_ids`` are mesh global ids for ``cell``/``facet`` entities and
    declared-source row indices for ``source_*`` entities.

    Coordinate and distance-query resource refusals retain their exact integer ``limit``,
    ``requested``, and ``completed`` counts plus the explicit owning resource.
    Optional achieved expression quantities come from the actual refusing
    ledger, not native execution counters or a recreated default budget.
    ``*_at_catch`` storage quantities describe the ledger after scope unwinding;
    its peak remains the measured storage upper bound, not process RSS.
    """

    check: str = eqx.field(static=True)
    status: MeshFindingStatus = eqx.field(static=True)
    entity_kind: MeshCertificateEntityKind = eqx.field(static=True)
    entity_ids: tuple[int, ...] = eqx.field(static=True)
    resource: MeshCertificateResource | None = eqx.field(static=True)
    requested: tuple[tuple[str, int], ...] = eqx.field(static=True)
    achieved: tuple[tuple[str, int], ...] = eqx.field(static=True)
    observations: tuple[tuple[str, int], ...] = eqx.field(static=True)
    finding_id: str = eqx.field(static=True)

    def __init__(
        self,
        check: str,
        status: MeshFindingStatus,
        entity_kind: MeshCertificateEntityKind,
        entity_ids: tuple[int, ...] = (),
        /,
        *,
        resource: MeshCertificateResource | None = None,
        requested: tuple[tuple[str, int], ...] = (),
        achieved: tuple[tuple[str, int], ...] = (),
        observations: tuple[tuple[str, int], ...] = (),
    ) -> None:
        name = canonical_identifier(check, "check")
        status_ = parse(status, MeshFindingStatus, "status")
        kind = parse(entity_kind, MeshCertificateEntityKind, "entity_kind")
        identifiers = tuple(sorted({int(value) for value in entity_ids}))

        def quantities(
            values: tuple[tuple[str, int], ...], label: str, /
        ) -> tuple[tuple[str, int], ...]:
            if not isinstance(values, tuple):
                raise TypeError(f"{label} must be an immutable quantity tuple.")
            rows: list[tuple[str, int]] = []
            for item in values:
                if not isinstance(item, tuple) or len(item) != 2:
                    raise TypeError(f"{label} must contain named integer quantities.")
                key = canonical_identifier(item[0], f"{label} quantity")
                rows.append((key, nonnegative_integer(item[1], f"{label}:{key}")))
            if len({key for key, _ in rows}) != len(rows):
                raise ValueError(f"{label} quantity keys must be unique.")
            return tuple(sorted(rows))

        resource_ = (
            None
            if resource is None
            else parse(resource, MeshCertificateResource, "resource")
        )
        requested_ = quantities(requested, "requested")
        achieved_ = quantities(achieved, "achieved")
        observations_ = quantities(observations, "observations")
        if resource_ is None:
            if requested_ or achieved_:
                raise ValueError(
                    "Resource quantities require their explicit owning resource."
                )
        else:
            wanted, completed = dict(requested_), dict(achieved_)
            if set(wanted) != {"limit", "requested"} or "completed" not in completed:
                raise ValueError(
                    "Resource refusals require their exact limit, requested, and completed quantities."
                )
            if (
                wanted["requested"] <= wanted["limit"]
                or completed["completed"] > wanted["limit"]
            ):
                raise ValueError(
                    "Refusal quantities must describe an actual pre-operation resource refusal."
                )
        self.check = name
        self.status = status_
        self.entity_kind = kind
        self.entity_ids = identifiers
        self.resource = resource_
        self.requested = requested_
        self.achieved = achieved_
        self.observations = observations_
        self.finding_id = canonical_fingerprint(
            {
                "kind": "mesh-certificate-finding",
                "check": name,
                "status": status_,
                "entity_kind": kind,
                "entity_ids": identifiers,
                "observations": observations_,
                **(
                    {}
                    if resource_ is None
                    else {
                        "resource": resource_,
                        "requested": requested_,
                        "achieved": achieved_,
                    }
                ),
            }
        )


@final
class MeshCertificateBinding(StrictModule, NonTrainableState):
    """Identity of everything a mesh certificate was computed from."""

    mesh_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    geometry_layout_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    coordinate_scope: CoordinateMapScope = eqx.field(static=True)
    source_id: str | None = eqx.field(static=True)
    source_revision: str | None = eqx.field(static=True)
    junction_vertices: tuple[int, ...] = eqx.field(static=True)
    limits_id: str = eqx.field(static=True)
    binding_id: str = eqx.field(static=True)

    def __init__(
        self,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        coordinate_scope: CoordinateMapScope,
        limits: MeshCertificateLimits,
        /,
        *,
        source_id: str | None = None,
        source_revision: str | None = None,
        junction_vertices: tuple[int, ...] = (),
    ) -> None:
        scope = parse(coordinate_scope, CoordinateMapScope, "coordinate_scope")
        self.mesh_id = mesh.mesh_id
        self.topology_id = mesh.topology_id
        self.geometry_layout_id = geometry.geometry_layout_id
        self.geometry_id = cell_geometry_id(geometry)
        self.coordinate_scope = scope
        self.source_id = source_id
        self.source_revision = source_revision
        self.junction_vertices = junction_vertices
        self.limits_id = limits.limits_id
        self.binding_id = canonical_fingerprint(
            {
                "kind": "mesh-certificate-binding",
                "mesh": mesh.mesh_id,
                "topology": mesh.topology_id,
                "geometry_layout": self.geometry_layout_id,
                "geometry": self.geometry_id,
                "coordinate_scope": scope,
                "source": source_id,
                "source_revision": source_revision,
                "junction_vertices": junction_vertices,
                "limits": limits.limits_id,
            }
        )

    def require(self, mesh: CellMesh, geometry: CellGeometrySpec, /) -> None:
        """Refuse reuse for another mesh topology, coordinate array or layout."""

        if (
            mesh.mesh_id != self.mesh_id
            or mesh.topology_id != self.topology_id
            or geometry.geometry_layout_id != self.geometry_layout_id
            or cell_geometry_id(geometry) != self.geometry_id
        ):
            raise ValueError("Mesh certificate is not bound to this mesh and geometry.")


def _certificate_status(
    findings: tuple[MeshCertificateFinding, ...], /
) -> MeshCertificateStatus:
    if any(finding.status == "violated" for finding in findings):
        return "violated"
    if findings:
        return "unresolved"
    return "certified"


@final
class MappedBoundaryDegreeEvidence(StrictModule, NonTrainableState):
    """Premises and outcome of the boundary-degree theorem for mapped volume meshes.

    Theorem. Let ``M`` be a mesh of triangles/quadrilaterals in ``R^2`` or of
    tetrahedra/prisms/hexahedra in ``R^3`` whose interior facets pair exactly
    two cells with opposite induced orientations, and let ``F`` be its piecewise
    polynomial coordinate map. Suppose

    (a) ``det DF_c > 0`` on every closed reference cell (the bound validity
        certificate) and every cell map is injective on its closed cell (the
        fixed-projection local injectivity proof);
    (b) cells that share a vertex, an edge or a facet have identical exact
        restricted expressions on it, so ``F`` is well defined on ``|M|``;
    (c) ``F`` is injective on the union of boundary facets;
    (d) every boundary shell ``j`` (a ridge-connected, balanced boundary
        component) has exterior degree ``e_j = a_j + c_j - 1 = 0``, where
        ``a_j`` is the winding number of all other shells at a point of
        shell ``j`` and ``c_j`` is ``1`` for a positively and ``0`` for a
        negatively oriented shell.

    Then ``F`` is injective on ``|M|``: cell interiors are pairwise disjoint
    and closed cells meet only on combinatorially shared entities.

    Proof outline. Off the image of the facet skeleton, the preimage count is
    the sum of per-cell degrees, which is the winding number of the oriented
    boundary cycle because identical interior facet traces cancel. Every
    complement component of the embedded boundary borders some shell, the
    winding number jumps by exactly one into the side covered by the adjacent
    cell, and that side carries ``a_j + c_j = e_j + 1``; so (d) bounds the
    preimage count by one almost everywhere. A local-degree argument at
    interior points of ``|M|`` and (c) at boundary points turn this into
    injectivity. A single shell has ``a_j = 0`` and its covered side already
    carries degree at least one, so (d) holds without further work. Several
    shells take ``a_j`` from the exact winding number of an affine proxy shell
    (vertex images, triangulated faces) joined to the curved shell by a
    straight homotopy whose certified deviation stays strictly away from the
    query point, and ``c_j`` from the sign of the exact signed shell measure.

    ``status`` is ``embedded`` when every premise is proved, ``violated`` when
    proved premises imply a nonconforming map, a boundary self-contact or a
    degree above one (``failed_premise`` names the deciding check), and
    ``premise_unproven`` when the route falls back to pairwise cell contacts.
    """

    status: MappedBoundaryDegreeStatus = eqx.field(static=True)
    failed_premise: str | None = eqx.field(static=True)
    oriented_cell_count: int = eqx.field(static=True)
    conforming_entity_count: int = eqx.field(static=True)
    boundary_facet_count: int = eqx.field(static=True)
    boundary_pair_count: int = eqx.field(static=True)
    tangent_cone_contact_count: int = eqx.field(static=True)
    shell_count: int = eqx.field(static=True)
    shell_exterior_degrees: tuple[int, ...] = eqx.field(static=True)
    maximum_degree: int | None = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        status: MappedBoundaryDegreeStatus,
        failed_premise: str | None,
        /,
        *,
        oriented_cell_count: int,
        conforming_entity_count: int,
        boundary_facet_count: int,
        boundary_pair_count: int,
        tangent_cone_contact_count: int,
        shell_count: int,
        shell_exterior_degrees: tuple[int, ...],
        maximum_degree: int | None,
    ) -> None:
        status_ = parse(status, MappedBoundaryDegreeStatus, "status")
        premise = (
            None
            if failed_premise is None
            else canonical_identifier(failed_premise, "failed_premise")
        )
        counts = {
            name: nonnegative_integer(value, name)
            for name, value in (
                ("oriented_cell_count", oriented_cell_count),
                ("conforming_entity_count", conforming_entity_count),
                ("boundary_facet_count", boundary_facet_count),
                ("boundary_pair_count", boundary_pair_count),
                ("tangent_cone_contact_count", tangent_cone_contact_count),
                ("shell_count", shell_count),
            )
        }
        if not isinstance(shell_exterior_degrees, tuple) or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in shell_exterior_degrees
        ):
            raise TypeError("shell_exterior_degrees must be a tuple of integers.")
        degree = (
            None
            if maximum_degree is None
            else nonnegative_integer(maximum_degree, "maximum_degree")
        )
        if (status_ == "embedded") != (premise is None):
            raise ValueError("Only an embedded outcome has no deciding premise.")
        if status_ == "embedded" and (
            degree != 1
            or len(shell_exterior_degrees) != counts["shell_count"]
            or any(shell_exterior_degrees)
        ):
            raise ValueError(
                "An embedded outcome requires zero exterior degree on every shell."
            )
        self.status = status_
        self.failed_premise = premise
        self.oriented_cell_count = counts["oriented_cell_count"]
        self.conforming_entity_count = counts["conforming_entity_count"]
        self.boundary_facet_count = counts["boundary_facet_count"]
        self.boundary_pair_count = counts["boundary_pair_count"]
        self.tangent_cone_contact_count = counts["tangent_cone_contact_count"]
        self.shell_count = counts["shell_count"]
        self.shell_exterior_degrees = shell_exterior_degrees
        self.maximum_degree = degree
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "mapped-boundary-degree-evidence",
                "status": status_,
                "failed_premise": premise,
                "counts": tuple(counts.items()),
                "shell_exterior_degrees": shell_exterior_degrees,
                "maximum_degree": degree,
            }
        )


@final
class GlobalEmbeddingCertificate(StrictModule, NonTrainableState):
    """Evidence that cell interiors are pairwise disjoint and contacts are shared.

    ``evaluated_checks`` lists the premises actually decided. ``status`` is
    ``certified`` only when every premise holds; findings expose the failing
    or undecided entities. ``boundary_degree`` records the premises of the
    mapped boundary-degree theorem whenever a mapped volume proof attempted it.
    """

    binding: MeshCertificateBinding
    validity_certificate_id: str = eqx.field(static=True)
    status: MeshCertificateStatus = eqx.field(static=True)
    findings: tuple[MeshCertificateFinding, ...]
    evaluated_checks: tuple[str, ...] = eqx.field(static=True)
    cell_count: int = eqx.field(static=True)
    boundary_facet_count: int = eqx.field(static=True)
    shell_count: int = eqx.field(static=True)
    candidate_pair_count: int = eqx.field(static=True)
    ray_test_count: int = eqx.field(static=True)
    subdivision_piece_count: int = eqx.field(static=True)
    periodic_image_count: int = eqx.field(static=True)
    source_expression_work_units: int = eqx.field(static=True)
    source_expression_peak_bytes: int = eqx.field(static=True)
    boundary_degree: MappedBoundaryDegreeEvidence | None
    certificate_id: str = eqx.field(static=True)

    def __init__(
        self,
        binding: MeshCertificateBinding,
        validity_certificate_id: str,
        findings: tuple[MeshCertificateFinding, ...],
        evaluated_checks: tuple[str, ...],
        /,
        *,
        cell_count: int,
        boundary_facet_count: int,
        shell_count: int,
        candidate_pair_count: int,
        ray_test_count: int,
        subdivision_piece_count: int,
        periodic_image_count: int = 0,
        source_expression_work_units: int = 0,
        source_expression_peak_bytes: int = 0,
        boundary_degree: MappedBoundaryDegreeEvidence | None = None,
    ) -> None:
        if not isinstance(binding, MeshCertificateBinding):
            raise TypeError("binding must be MeshCertificateBinding.")
        if not all(isinstance(value, MeshCertificateFinding) for value in findings):
            raise TypeError("findings must contain MeshCertificateFinding values.")
        if boundary_degree is not None and not isinstance(
            boundary_degree, MappedBoundaryDegreeEvidence
        ):
            raise TypeError("boundary_degree must be MappedBoundaryDegreeEvidence.")
        self.binding = binding
        self.validity_certificate_id = str(validity_certificate_id)
        self.status = _certificate_status(findings)
        self.findings = tuple(findings)
        self.evaluated_checks = tuple(evaluated_checks)
        self.cell_count = cell_count
        self.boundary_facet_count = boundary_facet_count
        self.shell_count = shell_count
        self.candidate_pair_count = candidate_pair_count
        self.ray_test_count = ray_test_count
        self.subdivision_piece_count = subdivision_piece_count
        self.periodic_image_count = periodic_image_count
        self.source_expression_work_units = source_expression_work_units
        self.source_expression_peak_bytes = source_expression_peak_bytes
        self.boundary_degree = boundary_degree
        self.certificate_id = canonical_fingerprint(
            {
                "kind": "global-embedding-certificate",
                "binding": binding.binding_id,
                "validity": self.validity_certificate_id,
                "status": self.status,
                "findings": [value.finding_id for value in findings],
                "checks": self.evaluated_checks,
                "counts": (
                    cell_count,
                    boundary_facet_count,
                    shell_count,
                    candidate_pair_count,
                    ray_test_count,
                    subdivision_piece_count,
                    periodic_image_count,
                    source_expression_work_units,
                    source_expression_peak_bytes,
                ),
                **(
                    {}
                    if boundary_degree is None
                    else {"boundary_degree": boundary_degree.evidence_id}
                ),
            }
        )


@final
class PiecewiseLinearDomain(StrictModule, NonTrainableState):
    """Declared affine domain: oriented boundary and interface facets.

    ``facets`` are segments (2-D) or triangles (3-D) whose induced normal points
    from region ``facet_regions[:, 0]`` into region ``facet_regions[:, 1]``;
    ``-1`` denotes the exterior. Region measures follow from the divergence
    theorem over these facets.
    """

    __strict_contract__ = True

    vertices: HostFloat64[_DomainVertexDim, _DomainAmbientDim]
    facets: HostInt64[_DomainFacetDim, _DomainAmbientDim]
    facet_regions: HostInt64[_DomainFacetDim, Literal[2]]
    region_ids: tuple[str, ...] = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    domain_id: str = eqx.field(static=True)

    def __init__(
        self,
        vertices: ArrayLike,
        facets: ArrayLike,
        facet_regions: ArrayLike,
        region_ids: tuple[str, ...],
        /,
        *,
        source_id: str,
    ) -> None:
        scope = Scope()
        points = parse(
            np.asarray(vertices, dtype=np.float64),
            HostFloat64[_DomainVertexDim, _DomainAmbientDim],
            "vertices",
            scope=scope,
        )
        rows = parse(
            np.asarray(facets, dtype=np.int64),
            HostInt64[_DomainFacetDim, _DomainAmbientDim],
            "facets",
            scope=scope,
        )
        pairs = parse(
            np.asarray(facet_regions, dtype=np.int64),
            HostInt64[_DomainFacetDim, Literal[2]],
            "facet_regions",
            scope=scope,
        )
        ids = unique_identifiers(region_ids, "region_ids")
        source = canonical_identifier(source_id, "source_id")
        if points.shape[1] not in (2, 3):
            raise ValueError("Declared domains are two- or three-dimensional.")
        if not np.all(np.isfinite(points)):
            raise ValueError("Declared domain vertices must be finite.")
        if np.any(rows < 0) or np.any(rows >= points.shape[0]):
            raise ValueError("Declared domain facets index undeclared vertices.")
        if (
            np.any(pairs < -1)
            or np.any(pairs >= len(ids))
            or np.any(pairs[:, 0] == pairs[:, 1])
        ):
            raise ValueError("Declared facet regions must be distinct regions or -1.")
        if np.any(_degenerate_simplices(points, rows)):
            raise ValueError("Declared domain facets must be nondegenerate.")
        self.vertices = points
        self.facets = rows
        self.facet_regions = pairs
        self.region_ids = ids
        self.source_id = source
        self.source_revision = canonical_fingerprint(
            {
                "vertices": array_tree_fingerprint(points),
                "facets": array_tree_fingerprint(rows),
                "facet_regions": array_tree_fingerprint(pairs),
                "region_ids": ids,
            }
        )
        self.domain_id = canonical_fingerprint(
            {
                "kind": "piecewise-linear-domain",
                "source": source,
                "revision": self.source_revision,
            }
        )

    @property
    def ambient_dimension(self) -> int:
        return self.vertices.shape[1]


@final
class DomainCoverageCertificate(StrictModule, NonTrainableState):
    """Exact coverage of a declared domain, its boundary, interfaces and regions.

    Region measures are exact integrals of the source coordinate expressions.
    Reporting bounds include binary64 rounding; integration errors are zero
    for exact integration. An absent integral (unresolved or refused source
    expression) reports ``None`` for its achieved measure, achieved bounds and
    integration error: ``None`` means unknown/unbounded, never an exact zero.

    ``facet_source_overlaps`` retains authoritative source facet row identities.
    Rows contain target facet global ID, source row ID, projection axes, lower
    and upper projected overlap measures, exact/candidate semantics and the
    coordinate space. Physical rows describe planar source overlap; reference
    rows describe the independent parameter charts of a mapped source.
    Candidate bounds are conservative relations, not asserted actual overlaps.
    ``premise_certificate_ids`` binds independently computed source embedding
    and reference/root partition evidence for mapped reference domains.
    ``candidate_pair_count`` counts this owner's evaluated source-source and
    target-source closed-box candidates; nested premise work is not guessed
    or added to this direct counter. ``subdivision_piece_count`` and
    ``maximum_subdivision_depth_reached`` are this owner's actually charged
    subdivision pieces and deepest examined level, including a refused
    over-budget piece; zero means no subdivision was performed.
    ``source_expression_work_units`` is the actual charged work of this owner's
    exact coordinate-expression ledger. ``source_expression_required_work_units``
    is its largest exact dry-admission requirement and therefore the minimum
    work limit that admits the same invocation. ``source_expression_peak_bytes``
    is the charged storage upper bound. Refused requests retain completed and
    required work. Zero means no coordinate expression; ``None`` means unmeasured.
    """

    binding: MeshCertificateBinding
    embedding_certificate_id: str = eqx.field(static=True)
    domain_id: str = eqx.field(static=True)
    status: MeshCertificateStatus = eqx.field(static=True)
    findings: tuple[MeshCertificateFinding, ...]
    region_ids: tuple[str, ...] = eqx.field(static=True)
    requested_region_measures: tuple[float, ...] = eqx.field(static=True)
    achieved_region_measures: tuple[float | None, ...] = eqx.field(static=True)
    requested_region_measure_bounds: tuple[tuple[float, float], ...] = eqx.field(
        static=True
    )
    achieved_region_measure_bounds: tuple[tuple[float, float] | None, ...] = eqx.field(
        static=True
    )
    integration_error_bounds: tuple[float | None, ...] = eqx.field(static=True)
    facet_source_overlaps: tuple[_CoverageOverlap, ...] = eqx.field(static=True)
    premise_certificate_ids: tuple[str, ...] = eqx.field(static=True)
    source_facet_count: int = eqx.field(static=True)
    covered_source_facet_count: int = eqx.field(static=True)
    candidate_pair_count: int = eqx.field(static=True)
    subdivision_piece_count: int = eqx.field(static=True)
    maximum_subdivision_depth_reached: int = eqx.field(static=True)
    source_expression_work_units: int | None = eqx.field(static=True)
    source_expression_required_work_units: int | None = eqx.field(static=True)
    source_expression_peak_bytes: int | None = eqx.field(static=True)
    certificate_id: str = eqx.field(static=True)

    def __init__(
        self,
        binding: MeshCertificateBinding,
        embedding_certificate_id: str,
        domain: PiecewiseLinearDomain | MappedReferenceDomain,
        findings: tuple[MeshCertificateFinding, ...],
        /,
        *,
        requested_region_measures: tuple[float, ...],
        achieved_region_measures: tuple[float | None, ...],
        requested_region_measure_bounds: tuple[tuple[float, float], ...],
        achieved_region_measure_bounds: tuple[tuple[float, float] | None, ...],
        integration_error_bounds: tuple[float | None, ...],
        covered_source_facet_count: int,
        facet_source_overlaps: tuple[_CoverageOverlap, ...],
        premise_certificate_ids: tuple[str, ...],
        candidate_pair_count: int,
        subdivision_piece_count: int,
        maximum_subdivision_depth_reached: int,
        source_expression_work_units: int | None,
        source_expression_required_work_units: int | None,
        source_expression_peak_bytes: int | None,
    ) -> None:
        if not isinstance(binding, MeshCertificateBinding):
            raise TypeError("binding must be MeshCertificateBinding.")
        for name, count in (
            ("candidate_pair_count", candidate_pair_count),
            ("subdivision_piece_count", subdivision_piece_count),
            ("maximum_subdivision_depth_reached", maximum_subdivision_depth_reached),
        ):
            if isinstance(count, bool) or not isinstance(count, (int, np.integer)):
                raise TypeError(f"{name} must be an integer.")
            if count < 0:
                raise ValueError(f"{name} must be nonnegative.")
        if not (
            (source_expression_work_units is None)
            == (source_expression_required_work_units is None)
            == (source_expression_peak_bytes is None)
        ):
            raise ValueError(
                "Source-expression charged work, requirement, and peak are "
                "measured together or not at all."
            )
        work_units = (
            None
            if source_expression_work_units is None
            else nonnegative_integer(
                source_expression_work_units, "source_expression_work_units"
            )
        )
        required_work_units = (
            None
            if source_expression_required_work_units is None
            else nonnegative_integer(
                source_expression_required_work_units,
                "source_expression_required_work_units",
            )
        )
        if (
            work_units is not None
            and required_work_units is not None
            and work_units > required_work_units
        ):
            raise ValueError(
                "Source-expression charged work cannot exceed its admitted requirement."
            )
        peak_bytes = (
            None
            if source_expression_peak_bytes is None
            else nonnegative_integer(
                source_expression_peak_bytes, "source_expression_peak_bytes"
            )
        )
        if not all(isinstance(value, MeshCertificateFinding) for value in findings):
            raise TypeError("findings must contain MeshCertificateFinding values.")
        if not (
            len(requested_region_measures)
            == len(achieved_region_measures)
            == len(requested_region_measure_bounds)
            == len(achieved_region_measure_bounds)
            == len(integration_error_bounds)
            == len(domain.region_ids)
        ):
            raise ValueError("Region measures must follow the declared regions.")
        for requested, achieved, requested_bounds, achieved_bounds, error in zip(
            requested_region_measures,
            achieved_region_measures,
            requested_region_measure_bounds,
            achieved_region_measure_bounds,
            integration_error_bounds,
            strict=True,
        ):
            if not (
                math.isfinite(requested)
                and math.isfinite(requested_bounds[0])
                and math.isfinite(requested_bounds[1])
                and requested_bounds[0] <= requested <= requested_bounds[1]
            ):
                raise ValueError(
                    "Requested region measure enclosures must contain their reports."
                )
            if achieved is None or achieved_bounds is None or error is None:
                if not (achieved is None and achieved_bounds is None and error is None):
                    raise ValueError(
                        "An absent region integral reports no achieved bound or error."
                    )
            elif not (
                math.isfinite(achieved)
                and math.isfinite(error)
                and error >= 0.0
                and math.isfinite(achieved_bounds[0])
                and math.isfinite(achieved_bounds[1])
                and achieved_bounds[0] <= achieved <= achieved_bounds[1]
            ):
                raise ValueError(
                    "Achieved measure reports must lie in their finite enclosures."
                )
        if (
            _certificate_status(findings) == "certified"
            and None in achieved_region_measures
        ):
            raise ValueError(
                "Certified coverage requires every achieved region integral."
            )
        parsed_overlaps: list[_CoverageOverlap] = []
        for facet, source, axes, lower, upper, semantics, space in facet_source_overlaps:
            semantics = parse(semantics, CoverageOverlapSemantics, "overlap semantics")
            space = parse(space, CoverageCoordinateSpace, "overlap coordinate space")
            if (
                source < 0
                or source >= domain.facets.shape[0]
                or len(axes) != domain.ambient_dimension - 1
                or len(set(axes)) != len(axes)
                or any(axis < 0 or axis >= domain.ambient_dimension for axis in axes)
                or not 0.0 <= lower <= upper
            ):
                raise ValueError("Facet-source overlap evidence must be well formed.")
            parsed_overlaps.append((facet, source, axes, lower, upper, semantics, space))
        facet_source_overlaps = tuple(parsed_overlaps)
        if tuple(sorted(facet_source_overlaps)) != facet_source_overlaps:
            raise ValueError("Facet-source overlap evidence must be canonically ordered.")
        self.binding = binding
        self.embedding_certificate_id = str(embedding_certificate_id)
        self.domain_id = domain.domain_id
        self.status = _certificate_status(findings)
        self.findings = tuple(findings)
        self.region_ids = domain.region_ids
        self.requested_region_measures = tuple(requested_region_measures)
        self.achieved_region_measures = tuple(achieved_region_measures)
        self.requested_region_measure_bounds = tuple(requested_region_measure_bounds)
        self.achieved_region_measure_bounds = tuple(achieved_region_measure_bounds)
        self.integration_error_bounds = tuple(integration_error_bounds)
        self.source_facet_count = domain.facets.shape[0]
        self.covered_source_facet_count = covered_source_facet_count
        self.facet_source_overlaps = tuple(facet_source_overlaps)
        self.premise_certificate_ids = tuple(premise_certificate_ids)
        self.candidate_pair_count = int(candidate_pair_count)
        self.subdivision_piece_count = int(subdivision_piece_count)
        self.maximum_subdivision_depth_reached = int(maximum_subdivision_depth_reached)
        self.source_expression_work_units = work_units
        self.source_expression_required_work_units = required_work_units
        self.source_expression_peak_bytes = peak_bytes
        self.certificate_id = canonical_fingerprint(
            {
                "kind": "domain-coverage-certificate",
                "binding": binding.binding_id,
                "embedding": self.embedding_certificate_id,
                "domain": domain.domain_id,
                "status": self.status,
                "findings": [value.finding_id for value in findings],
                "requested": self.requested_region_measures,
                "achieved": self.achieved_region_measures,
                "requested_bounds": self.requested_region_measure_bounds,
                "achieved_bounds": self.achieved_region_measure_bounds,
                "integration_errors": self.integration_error_bounds,
                "facet_source_overlaps": self.facet_source_overlaps,
                "premises": self.premise_certificate_ids,
                "covered": covered_source_facet_count,
                "candidate_pairs": self.candidate_pair_count,
                "subdivision_pieces": self.subdivision_piece_count,
                "maximum_subdivision_depth": self.maximum_subdivision_depth_reached,
                "source_expression_work": (
                    work_units,
                    required_work_units,
                    peak_bytes,
                ),
            }
        )


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
            raise ValueError("Source distance bounds must be ordered numbers.")
        self.lower = low
        self.upper = high
        self.semantics = parse(semantics, SourceBoundSemantics, "semantics")


@final
class SourceBoundarySamples(StrictModule, NonTrainableState):
    """Points covering a source boundary.

    Every source boundary point lies within ``covering_radius[k]`` of some
    sample ``points[k]``, and every sample lies within ``point_error[k]`` of the
    boundary. ``complete`` is false when the sample budget prevented coverage.
    """

    __strict_contract__ = True

    points: HostFloat64[_SourcePointDim, _SourceAmbientDim]
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
            HostFloat64[_SourcePointDim, _SourceAmbientDim],
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
            raise ValueError("Source samples need finite points and nonnegative radii.")
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

    Certified semantics require a complete oriented chart/trim chain and a
    two-sided pointwise interpolation bound, not sampled nodal residuals.
    Collapsed triangles are retained: their source wedges still need a bound.
    """

    __strict_contract__ = True

    simplices: HostFloat64[_SourcePointDim, _SourceChartCornerDim, _SourceAmbientDim]
    deviation_bounds: HostFloat64[_SourcePointDim]
    semantics: SourceBoundSemantics = eqx.field(static=True)
    complete: bool = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    findings: tuple[MeshCertificateFinding, ...]
    resource_counts: tuple[tuple[str, int, int], ...] = eqx.field(static=True)
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
        resource_counts: tuple[tuple[str, int, int], ...] = (),
        restriction_ids: tuple[str, ...] = (),
    ) -> None:
        scope = Scope()
        values = parse(
            np.asarray(simplices, dtype=np.float64),
            HostFloat64[_SourcePointDim, _SourceChartCornerDim, _SourceAmbientDim],
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
            raise ValueError("Source chart simplices must have three corners.")
        if not np.all(np.isfinite(values)) or np.any(~(bounds >= 0.0)):
            raise ValueError(
                "Source chart images must be finite with nonnegative bounds."
            )
        self.simplices = values
        self.deviation_bounds = bounds
        self.semantics = parse(semantics, SourceBoundSemantics, "semantics")
        self.complete = bool(complete)
        self.source_id = canonical_identifier(source_id, "source_id")
        self.source_revision = canonical_identifier(source_revision, "source_revision")
        if any(not isinstance(finding, MeshCertificateFinding) for finding in findings):
            raise TypeError(
                "Source chart findings must be MeshCertificateFinding records."
            )
        if any(used < 0 or maximum < 0 for _, used, maximum in resource_counts):
            raise ValueError("Source chart resource work and limits must be nonnegative.")
        resources = tuple(
            sorted(
                (canonical_identifier(name, "source_resource"), used, maximum)
                for name, used, maximum in resource_counts
            )
        )
        if len({name for name, _, _ in resources}) != len(resources):
            raise ValueError("Source chart resource names must be unique.")
        self.findings = tuple(findings)
        self.resource_counts = resources
        restrictions = tuple(
            canonical_identifier(value, "chart_restriction_id")
            for value in restriction_ids
        )
        if len(set(restrictions)) != len(restrictions):
            raise ValueError("Source chart restriction identities must be unique.")
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
        self, maximum_patches: int, /
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


class _CurveSourceDim(Dim, minimum=1):
    """Independently declared trimmed source carriers."""


class _SourceBoxCornerDim(Dim, minimum=2):
    """Lower and upper corners of a source enclosure."""


def _source_box_radius(box: np.ndarray, point: np.ndarray, /) -> float:
    squared = sum(
        max(
            abs(Fraction(float(low)) - Fraction(float(value))),
            abs(Fraction(float(high)) - Fraction(float(value))),
        )
        ** 2
        for low, high, value in zip(box[0], box[1], point, strict=True)
    )
    return float(
        np.nextafter(math.sqrt(float(np.nextafter(float(squared), math.inf))), math.inf)
    )


def _enclosed_source_distance(
    points: np.ndarray, boxes: np.ndarray, samples: SourceBoundarySamples, /
) -> SourceBoundaryDistance:
    """Whole-source box lower bounds and actual source-point witness upper bounds."""
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != boxes.shape[2]:
        raise ValueError("Source distance points must match the source dimension.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Source distance points must be finite.")
    lower = np.zeros((values.shape[0],), dtype=np.float64)
    upper = np.full((values.shape[0],), math.inf, dtype=np.float64)
    chunk = max(
        1, _DISTANCE_WORKING_ENTRIES // max(boxes.shape[0], samples.points.shape[0], 1)
    )
    source_scale = max(
        float(np.max(np.abs(boxes))),
        float(np.max(np.abs(samples.points), initial=0.0)),
        1.0,
    )
    for start in range(0, values.shape[0], chunk):
        selected = values[start : start + chunk]
        scale = max(float(np.max(np.abs(selected), initial=0.0)), source_scale)
        slack = 64 * _EPSILON * scale
        offset = np.maximum(
            np.maximum(
                boxes[None, :, 0] - selected[:, None],
                selected[:, None] - boxes[None, :, 1],
            ),
            0.0,
        )
        lower[start : start + chunk] = np.maximum(
            np.min(np.linalg.norm(offset, axis=-1), axis=1) - slack, 0.0
        )
        if samples.points.shape[0]:
            distance = np.linalg.norm(selected[:, None] - samples.points[None], axis=-1)
            upper[start : start + chunk] = np.nextafter(
                np.min(distance + samples.point_error, axis=1) + slack, math.inf
            )
    return SourceBoundaryDistance(lower, upper, "certified")


@final
class ParametricCurveBoundarySource(StrictModule, NonTrainableState):
    """Continuous boundary queries of independently declared trimmed curves.

    Carrier boxes cover whole parameter intervals. Evaluated anchors are only
    representatives; outward parameter brackets bound their evaluation error.
    Distance bounds use source-point witnesses and full source boxes, never a
    sampled chord or a nearest sampled parameter as an exact distance.
    """

    __strict_contract__ = True

    curves: tuple[AbstractCurve, ...]
    parameter_ranges: tuple[tuple[float, float], ...] = eqx.field(static=True)
    boxes: HostFloat64[_CurveSourceDim, _SourceBoxCornerDim, _SourceAmbientDim]
    samples: SourceBoundarySamples
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    covering_radius: float = eqx.field(static=True)
    maximum_samples: int = eqx.field(static=True)

    def __init__(
        self,
        curves: tuple[AbstractCurve, ...],
        parameter_ranges: tuple[tuple[float, float], ...],
        /,
        *,
        source_id: str,
        source_revision: str,
        covering_radius: float = 0.01,
        maximum_samples: int = 4096,
    ) -> None:
        if (
            not isinstance(curves, tuple)
            or not curves
            or any(not isinstance(curve, AbstractCurve) for curve in curves)
        ):
            raise TypeError("curves must be a nonempty tuple of AbstractCurve carriers.")
        if len(curves) != len(parameter_ranges):
            raise ValueError("Each source carrier needs one explicit trimmed range.")
        dimension = curves[0].ambient_dimension
        if any(curve.ambient_dimension != dimension for curve in curves):
            raise ValueError("Source curve ambient dimensions must agree.")
        radius = float(covering_radius)
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError("covering_radius must be finite and positive.")
        budget = positive_integer(maximum_samples, "maximum_samples")
        ranges = tuple(
            curve.validate_range(*interval)
            for curve, interval in zip(curves, parameter_ranges, strict=True)
        )
        boxes = np.asarray(
            [
                curve.bounding_box(*interval)
                for curve, interval in zip(curves, ranges, strict=True)
            ],
            dtype=np.float64,
        )
        if boxes.shape != (len(curves), 2, dimension) or not np.all(np.isfinite(boxes)):
            raise ValueError(
                "Trimmed curve enclosures must be finite (2, ambient) boxes."
            )
        if np.any(boxes[:, 0] > boxes[:, 1]):
            raise ValueError("Trimmed curve enclosure bounds must be ordered.")
        points, radii, errors = [], [], []
        pending = [(i, first, last) for i, (first, last) in enumerate(ranges)]
        complete = True
        evaluations = 0
        while pending:
            if evaluations >= budget:
                complete = False
                break
            i, first, last = pending.pop()
            middle = float((Fraction(first) + Fraction(last)) / 2)
            point = np.asarray(
                curves[i].evaluate(jnp.asarray(middle, dtype=jnp.float64)),
                dtype=np.float64,
            )
            box = np.asarray(curves[i].bounding_box(first, last), dtype=np.float64)
            anchor_first = max(first, float(np.nextafter(middle, -math.inf)))
            anchor_last = min(last, float(np.nextafter(middle, math.inf)))
            error_box = np.asarray(
                curves[i].bounding_box(anchor_first, anchor_last), dtype=np.float64
            )
            error = _source_box_radius(error_box, point)
            bound = _source_box_radius(box, point)
            evaluations += 1
            if bound > radius and first < middle < last:
                pending.extend(((i, first, middle), (i, middle, last)))
                continue
            complete = complete and bound <= radius
            points.append(point)
            radii.append(bound)
            errors.append(error)
        self.curves = curves
        self.parameter_ranges = ranges
        self.boxes = parse(
            boxes,
            HostFloat64[_CurveSourceDim, _SourceBoxCornerDim, _SourceAmbientDim],
            "boxes",
        )
        self.samples = SourceBoundarySamples(
            np.asarray(points, dtype=np.float64).reshape(-1, dimension),
            np.asarray(radii, dtype=np.float64),
            np.asarray(errors, dtype=np.float64),
            "certified",
            complete=complete,
        )
        self.source_id = canonical_identifier(source_id, "source_id")
        self.source_revision = canonical_identifier(source_revision, "source_revision")
        self.covering_radius = radius
        self.maximum_samples = budget

    @property
    def ambient_dimension(self) -> int:
        return self.curves[0].ambient_dimension

    def boundary_distance(self, points: np.ndarray, /) -> SourceBoundaryDistance:
        return _enclosed_source_distance(points, self.boxes, self.samples)

    def boundary_samples(self, maximum_samples: int, /) -> SourceBoundarySamples:
        budget = positive_integer(maximum_samples, "maximum_samples")
        count = self.samples.points.shape[0]
        return SourceBoundarySamples(
            self.samples.points[:budget],
            self.samples.covering_radius[:budget],
            self.samples.point_error[:budget],
            "certified",
            complete=self.samples.complete and count <= budget,
        )


class _MappedBoundaryDomain(Protocol):
    """Authoritative fields of the nominal mapped-reference domain owner."""

    @property
    def source_id(self) -> str: ...

    @property
    def source_revision(self) -> str: ...

    @property
    def ambient_dimension(self) -> int: ...

    @property
    def reference_mesh(self) -> CellMesh: ...

    @property
    def source_geometry(self) -> CellGeometrySpec: ...


@final
class MappedDomainBoundarySource(StrictModule, NonTrainableState):
    """Queries of declared mapped source images, not their original CAD sources.

    Exact boundary equality requires the owning independent reference coverage,
    source/target embedding and complete source restriction partition proofs.
    Point queries separately use genuine exact-expression image covers.
    """

    __strict_contract__ = True

    domain: _MappedBoundaryDomain
    cell_regions: HostInt64[_MeshCellDim]
    limits: MeshCertificateLimits
    covering_radius: float = eqx.field(static=True)

    def __init__(
        self,
        domain: _MappedBoundaryDomain,
        cell_regions: ArrayLike,
        /,
        *,
        limits: MeshCertificateLimits | None = None,
        covering_radius: float = 0.01,
    ) -> None:
        from ._mapped_reference_domain import MappedReferenceDomain

        if not isinstance(domain, MappedReferenceDomain):
            raise TypeError(
                "domain must be an independently declared MappedReferenceDomain."
            )
        limits_ = MeshCertificateLimits() if limits is None else limits
        if not isinstance(limits_, MeshCertificateLimits):
            raise TypeError("limits must be MeshCertificateLimits or None.")
        radius = float(covering_radius)
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError("covering_radius must be finite and positive.")
        regions = parse(
            np.asarray(cell_regions, dtype=np.int64),
            HostInt64[_MeshCellDim],
            "cell_regions",
        )
        if np.any(regions < 0) or np.any(regions >= len(domain.region_ids)):
            raise ValueError(
                "Target cell regions must index the declared source regions."
            )
        self.domain = domain
        self.cell_regions = regions
        self.limits = limits_
        self.covering_radius = radius

    def for_target_regions(self, regions: ArrayLike, /) -> MappedDomainBoundarySource:
        """Renew an explicit target assignment without changing the source domain."""
        return MappedDomainBoundarySource(
            self.domain,
            regions,
            limits=self.limits,
            covering_radius=self.covering_radius,
        )

    @property
    def source_id(self) -> str:
        return self.domain.source_id

    @property
    def source_revision(self) -> str:
        return self.domain.source_revision

    @property
    def ambient_dimension(self) -> int:
        return self.domain.ambient_dimension

    def boundary_samples(self, maximum_samples: int, /) -> SourceBoundarySamples:
        maps, _ = _mapped_domain_source_maps(self)
        return _mapped_domain_source_samples(
            maps,
            self.ambient_dimension,
            self.covering_radius,
            positive_integer(maximum_samples, "maximum_samples"),
            self.limits,
        )

    def boundary_distance(self, points: np.ndarray, /) -> SourceBoundaryDistance:
        maps, boxes = _mapped_domain_source_maps(self)
        samples = _mapped_domain_source_samples(
            maps,
            self.ambient_dimension,
            self.covering_radius,
            self.limits.maximum_source_samples,
            self.limits,
        )
        return _enclosed_source_distance(points, boxes, samples)

    def boundary_coverage(
        self,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        /,
        *,
        limits: MeshCertificateLimits,
    ) -> DomainCoverageCertificate:
        from ..discretization._cell_geometry_validity import (
            certify_cell_geometry_validity,
        )
        from ._mapped_reference_domain import MappedReferenceDomain

        domain = self.domain
        if not isinstance(domain, MappedReferenceDomain):
            raise TypeError("The source no longer contains its nominal mapped domain.")
        validity = certify_cell_geometry_validity(geometry, mesh=mesh)
        embedding = certify_global_embedding(mesh, geometry, validity, limits=limits)
        return certify_domain_coverage(
            mesh,
            geometry,
            domain,
            self.cell_regions,
            limits=limits,
            embedding=embedding,
        )


class _ProjectionFacetDim(Dim):
    """Actual facets of an independently checked closed source projection."""


class _ProjectionPieceDim(Dim):
    """Continuously enclosed facet subpieces."""


class _ProjectionFiberCandidateDim(Dim):
    """Actual candidate facets tested by an exact source normal fiber."""


ProjectionFiberCrossingKind: TypeAlias = Literal[
    "disjoint", "interior", "edge", "vertex", "coplanar"
]


@final
class ImplicitProjectionCoverageEvidence(StrictModule, NonTrainableState):
    """Raw source-profile, tube, normal and exact degree-one fiber evidence.

    A creator's status is not a source-cover premise. Source fidelity computes
    this evidence again from the actual geometry, topology and runtime limits.
    Exact fiber coordinates and crossing parameters retain rational pairs;
    rounded source anchors cannot establish nearest-projection fibers.
    """

    __strict_contract__ = True

    binding: MeshCertificateBinding
    profile_id: str = eqx.field(static=True)
    source_state_id: str = eqx.field(static=True)
    source_schema_id: str = eqx.field(static=True)
    source_kernel_id: str = eqx.field(static=True)
    source_physical_id: str = eqx.field(static=True)
    source_family: str = eqx.field(static=True)
    source_center: HostFloat64[Literal[3]]
    source_radius: float = eqx.field(static=True)
    source_major_radius: float = eqx.field(static=True)
    source_reach_lower: float = eqx.field(static=True)
    source_tube_radius: float = eqx.field(static=True)
    source_hessian_upper: float = eqx.field(static=True)
    source_normal_lipschitz_upper: float = eqx.field(static=True)
    source_component_count: int = eqx.field(static=True)
    source_genus: int = eqx.field(static=True)
    status: MeshCertificateStatus = eqx.field(static=True)
    findings: tuple[MeshCertificateFinding, ...]
    facets: HostFloat64[_ProjectionFacetDim, Literal[3], Literal[3]]
    facet_vertices: HostInt64[_ProjectionFacetDim, Literal[3]]
    facet_ids: HostInt64[_ProjectionFacetDim]
    facet_components: HostInt64[_ProjectionFacetDim]
    piece_triangles: HostFloat64[_ProjectionPieceDim, Literal[3], Literal[3]]
    piece_barycentric_vertices: HostFloat64[_ProjectionPieceDim, Literal[3], Literal[3]]
    piece_coordinate_bounds: HostFloat64[_ProjectionPieceDim, Literal[2], Literal[3]]
    piece_owners: HostInt64[_ProjectionPieceDim]
    field_bounds: HostFloat64[_ProjectionPieceDim, Literal[2]]
    gradient_bounds: HostFloat64[_ProjectionPieceDim, Literal[2], Literal[3]]
    normal_bounds: HostFloat64[_ProjectionPieceDim, Literal[2]]
    fiber_endpoints: tuple[tuple[tuple[int, int], ...], ...] = eqx.field(static=True)
    fiber_value_bounds: HostFloat64[Literal[2], Literal[2]]
    fiber_gradient_bounds: HostFloat64[Literal[2]]
    fiber_candidate_facets: HostInt64[_ProjectionFiberCandidateDim]
    fiber_crossing_classes: tuple[ProjectionFiberCrossingKind, ...] = eqx.field(
        static=True
    )
    fiber_crossing_parameters: tuple[tuple[int, int] | None, ...] = eqx.field(static=True)
    forward_upper: float = eqx.field(static=True)
    forward_lower: float = eqx.field(static=True)
    query_count: int = eqx.field(static=True)
    ray_test_count: int = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        binding: MeshCertificateBinding,
        profile: SourceBoundaryQuery,
        findings: tuple[MeshCertificateFinding, ...],
        /,
        *,
        facets: np.ndarray,
        facet_vertices: np.ndarray,
        facet_ids: np.ndarray,
        facet_components: np.ndarray,
        piece_triangles: np.ndarray,
        piece_barycentric_vertices: np.ndarray,
        piece_coordinate_bounds: np.ndarray,
        piece_owners: np.ndarray,
        field_bounds: np.ndarray,
        gradient_bounds: np.ndarray,
        normal_bounds: np.ndarray,
        fiber_endpoints: tuple[tuple[tuple[int, int], ...], ...],
        fiber_value_bounds: np.ndarray,
        fiber_gradient_bounds: np.ndarray,
        fiber_candidate_facets: np.ndarray,
        fiber_crossing_classes: tuple[ProjectionFiberCrossingKind, ...],
        fiber_crossing_parameters: tuple[tuple[int, int] | None, ...],
        forward_upper: float,
        forward_lower: float,
        query_count: int,
        ray_test_count: int,
    ) -> None:
        from .implicit._analytic_profile import AnalyticImplicitProfile

        if not isinstance(profile, AnalyticImplicitProfile):
            raise TypeError(
                "Projection evidence requires an established AnalyticImplicitProfile."
            )
        profile.require_bound(profile.geometry, profile.coordinate_contract)
        if not isinstance(binding, MeshCertificateBinding):
            raise TypeError("binding must be MeshCertificateBinding.")
        if (binding.source_id, binding.source_revision) != (
            profile.source_id,
            profile.source_revision,
        ):
            raise ValueError(
                "Projection source facts are not bound to the source revision."
            )
        if not (float(forward_upper) >= float(forward_lower) >= 0.0):
            raise ValueError(
                "Projection distance bounds must be nonnegative and ordered."
            )
        if query_count < 0 or ray_test_count < 0:
            raise ValueError("Projection work counts must be nonnegative.")
        if fiber_endpoints and (
            len(fiber_endpoints) != 2
            or any(len(point) != 3 for point in fiber_endpoints)
            or any(
                denominator <= 0 for point in fiber_endpoints for _, denominator in point
            )
        ):
            raise ValueError(
                "An exact source fiber requires two rational three-dimensional endpoints."
            )
        count = fiber_candidate_facets.shape[0]
        if (
            len(fiber_crossing_classes) != count
            or len(fiber_crossing_parameters) != count
        ):
            raise ValueError(
                "Fiber crossing evidence must match its actual candidate facets."
            )
        classes = tuple(
            parse(kind, ProjectionFiberCrossingKind, "fiber_crossing_class")
            for kind in fiber_crossing_classes
        )
        scope = Scope()
        self.binding = binding
        self.profile_id = profile.profile_id
        self.source_state_id = profile.state_id
        self.source_schema_id = profile.schema_id
        self.source_kernel_id = profile.kernel_id
        self.source_physical_id = profile.coordinate_contract.spatial_id
        self.source_family = profile.family
        self.source_center = parse(
            profile.center, HostFloat64[Literal[3]], "source_center"
        )
        self.source_radius = profile.radius
        self.source_major_radius = profile.major_radius
        self.source_reach_lower = profile.reach_lower
        self.source_tube_radius = profile.tube_radius
        self.source_hessian_upper = profile.hessian_upper
        self.source_normal_lipschitz_upper = profile.normal_lipschitz_upper
        self.source_component_count = profile.component_count
        self.source_genus = profile.genus
        self.findings = tuple(findings)
        self.status = _certificate_status(self.findings)
        self.facets = parse(
            facets,
            HostFloat64[_ProjectionFacetDim, Literal[3], Literal[3]],
            "facets",
            scope=scope,
        )
        self.facet_vertices = parse(
            facet_vertices,
            HostInt64[_ProjectionFacetDim, Literal[3]],
            "facet_vertices",
            scope=scope,
        )
        self.facet_ids = parse(
            facet_ids, HostInt64[_ProjectionFacetDim], "facet_ids", scope=scope
        )
        self.facet_components = parse(
            facet_components,
            HostInt64[_ProjectionFacetDim],
            "facet_components",
            scope=scope,
        )
        self.piece_triangles = parse(
            piece_triangles,
            HostFloat64[_ProjectionPieceDim, Literal[3], Literal[3]],
            "piece_triangles",
            scope=scope,
        )
        self.piece_barycentric_vertices = parse(
            piece_barycentric_vertices,
            HostFloat64[_ProjectionPieceDim, Literal[3], Literal[3]],
            "piece_barycentric_vertices",
            scope=scope,
        )
        self.piece_coordinate_bounds = parse(
            piece_coordinate_bounds,
            HostFloat64[_ProjectionPieceDim, Literal[2], Literal[3]],
            "piece_coordinate_bounds",
            scope=scope,
        )
        self.piece_owners = parse(
            piece_owners, HostInt64[_ProjectionPieceDim], "piece_owners", scope=scope
        )
        self.field_bounds = parse(
            field_bounds,
            HostFloat64[_ProjectionPieceDim, Literal[2]],
            "field_bounds",
            scope=scope,
        )
        self.gradient_bounds = parse(
            gradient_bounds,
            HostFloat64[_ProjectionPieceDim, Literal[2], Literal[3]],
            "gradient_bounds",
            scope=scope,
        )
        self.normal_bounds = parse(
            normal_bounds,
            HostFloat64[_ProjectionPieceDim, Literal[2]],
            "normal_bounds",
            scope=scope,
        )
        self.fiber_endpoints = fiber_endpoints
        self.fiber_value_bounds = parse(
            fiber_value_bounds, HostFloat64[Literal[2], Literal[2]], "fiber_value_bounds"
        )
        self.fiber_gradient_bounds = parse(
            fiber_gradient_bounds, HostFloat64[Literal[2]], "fiber_gradient_bounds"
        )
        self.fiber_candidate_facets = parse(
            fiber_candidate_facets,
            HostInt64[_ProjectionFiberCandidateDim],
            "fiber_candidate_facets",
        )
        self.fiber_crossing_classes = classes
        self.fiber_crossing_parameters = fiber_crossing_parameters
        self.forward_upper, self.forward_lower = (
            float(forward_upper),
            float(forward_lower),
        )
        self.query_count, self.ray_test_count = query_count, ray_test_count
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "implicit-projection-coverage-evidence",
                "binding": binding.binding_id,
                "profile": self.profile_id,
                "source_facts": (
                    self.source_state_id,
                    self.source_schema_id,
                    self.source_kernel_id,
                    self.source_physical_id,
                    self.source_family,
                    self.source_radius,
                    self.source_major_radius,
                    self.source_reach_lower,
                    self.source_tube_radius,
                    self.source_hessian_upper,
                    self.source_normal_lipschitz_upper,
                    self.source_component_count,
                    self.source_genus,
                ),
                "findings": tuple(finding.finding_id for finding in self.findings),
                "arrays": array_tree_fingerprint(
                    (
                        self.facets,
                        self.facet_vertices,
                        self.facet_ids,
                        self.facet_components,
                        self.source_center,
                        self.piece_barycentric_vertices,
                        self.piece_coordinate_bounds,
                        self.piece_triangles,
                        self.piece_owners,
                        self.field_bounds,
                        self.gradient_bounds,
                        self.normal_bounds,
                        self.fiber_value_bounds,
                        self.fiber_gradient_bounds,
                        self.fiber_candidate_facets,
                    )
                ),
                "fiber": fiber_endpoints,
                "crossing_classes": classes,
                "crossing_parameters": fiber_crossing_parameters,
                "forward": (repr(self.forward_upper), repr(self.forward_lower)),
                "work": (query_count, ray_test_count),
            }
        )


@final
class ImplicitProjectionBoundarySource(StrictModule, NonTrainableState):
    """Nominal analytic source with independently recomputed projection coverage."""

    profile: SourceBoundaryQuery

    def __init__(self, profile: SourceBoundaryQuery, /) -> None:
        from .implicit._analytic_profile import AnalyticImplicitProfile

        if not isinstance(profile, AnalyticImplicitProfile):
            raise TypeError("profile must be an established AnalyticImplicitProfile.")
        profile.require_bound(profile.geometry, profile.coordinate_contract)
        self.profile = profile

    @property
    def source_id(self) -> str:
        return self.profile.source_id

    @property
    def source_revision(self) -> str:
        return self.profile.source_revision

    @property
    def ambient_dimension(self) -> int:
        return self.profile.ambient_dimension

    def boundary_distance(self, points: np.ndarray, /) -> SourceBoundaryDistance:
        return self.profile.boundary_distance(points)

    def boundary_samples(self, maximum_samples: int, /) -> SourceBoundarySamples:
        return self.profile.boundary_samples(maximum_samples)

    def projection_coverage(
        self,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        /,
        *,
        tolerance: float,
        limits: MeshCertificateLimits,
    ) -> ImplicitProjectionCoverageEvidence:
        from .implicit._analytic_profile import AnalyticImplicitProfile
        from .implicit._projection_cover import compute_implicit_projection_cover

        profile = self.profile
        if not isinstance(profile, AnalyticImplicitProfile):
            raise TypeError(
                "The projection source no longer contains its nominal profile."
            )
        profile.require_bound(profile.geometry, profile.coordinate_contract)
        return compute_implicit_projection_cover(
            mesh,
            geometry,
            profile,
            tolerance=tolerance,
            limits=limits,
        )


@final
class ImplicitBoundarySource(StrictModule):
    """Boundary queries of a compiled implicit geometry at one design state.

    Distance bounds are ``certified`` only for a globally reliable exact signed
    distance whose evaluation error is established by its owner; the boundary
    is then sampled on a grid of ``spacing`` over the kernel bounds, which are
    the declared enclosure of the boundary. Other fields report ``sampled``
    first-order distance estimates ``|phi| / |grad phi|``.
    """

    geometry: CompiledGeometry
    spacing: float = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)

    def __init__(
        self, geometry: CompiledGeometry, /, *, source_id: str, spacing: float
    ) -> None:
        if not isinstance(geometry, CompiledGeometry):
            raise TypeError("geometry must be CompiledGeometry.")
        step = float(spacing)
        if not math.isfinite(step) or step <= 0.0:
            raise ValueError("spacing must be finite and positive.")
        self.geometry = geometry
        self.spacing = step
        self.source_id = canonical_identifier(source_id, "source_id")
        self.source_revision = implicit_state_id(geometry)

    @property
    def ambient_dimension(self) -> int:
        return self.geometry.ambient_dimension

    @property
    def certified_distance(self) -> bool:
        certificate = self.geometry.field_certificate
        return (
            certificate.distance_semantics is DistanceSemantics.EXACT
            and certificate.sign_reliability is SignReliability.RELIABLE
            and certificate.zero_set_accuracy is ZeroSetAccuracy.EXACT
            and certificate.validity_region == "all_space"
            and certificate.evaluation_error is not None
            and certificate.bound_origin == "established"
        )

    def _field(self, points: np.ndarray, /) -> tuple[np.ndarray, np.ndarray]:
        values = jnp.asarray(points, dtype=jnp.float64)

        def scalar(point: jax.Array) -> jax.Array:
            return self.geometry.boundary_field(point[None])[0]

        field = np.asarray(self.geometry.boundary_field(values), dtype=np.float64)
        gradient = np.asarray(jax.vmap(jax.grad(scalar))(values), dtype=np.float64)
        return field.reshape(-1), gradient

    def boundary_distance(self, points: np.ndarray, /) -> SourceBoundaryDistance:
        field, gradient = self._field(points)
        if self.certified_distance:
            error = float(self.geometry.field_certificate.evaluation_error or 0.0)
            magnitude = np.abs(field)
            return SourceBoundaryDistance(
                np.maximum(magnitude - error, 0.0), magnitude + error, "certified"
            )
        norm = np.maximum(np.linalg.norm(gradient, axis=-1), np.finfo(np.float64).tiny)
        estimate = np.abs(field) / norm
        return SourceBoundaryDistance(estimate, estimate, "sampled")

    def boundary_samples(self, maximum_samples: int, /) -> SourceBoundarySamples:
        bounds = np.asarray(self.geometry.bounds, dtype=np.float64).reshape(2, -1)
        counts = np.floor((bounds[1] - bounds[0]) / self.spacing).astype(np.int64) + 2
        dimension = bounds.shape[1]
        if math.prod(int(value) for value in counts) > maximum_samples:
            empty = np.empty((0,), dtype=np.float64)
            return SourceBoundarySamples(
                np.empty((0, dimension)), empty, empty, "sampled", complete=False
            )
        axes = tuple(
            bounds[0, axis] + self.spacing * np.arange(counts[axis], dtype=np.float64)
            for axis in range(dimension)
        )
        grid = np.stack(
            [value.reshape(-1) for value in np.meshgrid(*axes, indexing="ij")], axis=-1
        )
        field, gradient = self._field(grid)
        certificate = self.geometry.field_certificate
        certified = self.certified_distance
        error = float(certificate.evaluation_error or 0.0) if certified else 0.0
        lipschitz = certificate.lipschitz_upper_bound
        half_diagonal = 0.5 * self.spacing * math.sqrt(dimension)
        # A boundary point y has a nearest node x with |x - y| <= half_diagonal,
        # so |phi(x)| <= L half_diagonal + error keeps every such node.
        band = (lipschitz if lipschitz is not None else 2.0) * half_diagonal + error
        keep = np.abs(field) <= band * (1.0 + 16.0 * _EPSILON)
        squared = np.maximum(
            np.sum(gradient[keep] ** 2, axis=-1), np.finfo(np.float64).tiny
        )
        projected = grid[keep] - (field[keep] / squared)[:, None] * gradient[keep]
        step = np.linalg.norm(grid[keep] - projected, axis=-1)
        residual, _ = self._field(projected)
        return SourceBoundarySamples(
            projected,
            (half_diagonal + step) * (1.0 + 16.0 * _EPSILON),
            np.abs(residual) + error,
            "certified" if certified else "sampled",
            complete=True,
        )


@final
class SourceFidelityCertificate(StrictModule, NonTrainableState):
    """Two-sided bounded deviation between the mesh boundary and a source.

    ``mesh_to_source_upper`` bounds the distance of every mesh boundary point
    from the source boundary; ``source_to_mesh_upper`` bounds the distance of
    every source boundary point from the mesh boundary. ``*_lower`` values are
    witnessed deviations. Each direction states whether its bound is certified
    or only sampled; ``status`` is certified only for certified bounds within
    the requested ``tolerance``.
    """

    binding: MeshCertificateBinding
    projection_coverage: ImplicitProjectionCoverageEvidence | None
    domain_coverage: DomainCoverageCertificate | None
    chart_coverage: SourceBoundaryChartCover | None
    status: MeshCertificateStatus = eqx.field(static=True)
    findings: tuple[MeshCertificateFinding, ...]
    tolerance: float = eqx.field(static=True)
    mesh_to_source_semantics: SourceBoundSemantics = eqx.field(static=True)
    source_to_mesh_semantics: SourceBoundSemantics = eqx.field(static=True)
    mesh_to_source_upper: float = eqx.field(static=True)
    mesh_to_source_lower: float = eqx.field(static=True)
    source_to_mesh_upper: float = eqx.field(static=True)
    source_to_mesh_lower: float = eqx.field(static=True)
    sample_order: int = eqx.field(static=True)
    mesh_sample_count: int = eqx.field(static=True)
    source_sample_count: int = eqx.field(static=True)
    certificate_id: str = eqx.field(static=True)
    target_facet_ids: tuple[int, ...] | None = eqx.field(static=True)
    source_scope_id: str | None = eqx.field(static=True)

    def __init__(
        self,
        binding: MeshCertificateBinding,
        findings: tuple[MeshCertificateFinding, ...],
        /,
        *,
        tolerance: float,
        semantics: tuple[SourceBoundSemantics, SourceBoundSemantics],
        mesh_to_source: tuple[float, float],
        source_to_mesh: tuple[float, float],
        sample_order: int,
        sample_counts: tuple[int, int],
        projection_coverage: ImplicitProjectionCoverageEvidence | None = None,
        domain_coverage: DomainCoverageCertificate | None = None,
        chart_coverage: SourceBoundaryChartCover | None = None,
        target_facet_ids: tuple[int, ...] | None = None,
        source_scope_id: str | None = None,
    ) -> None:
        if not isinstance(binding, MeshCertificateBinding):
            raise TypeError("binding must be MeshCertificateBinding.")
        if target_facet_ids is not None:
            if (
                not target_facet_ids
                or tuple(sorted(set(target_facet_ids))) != target_facet_ids
            ):
                raise ValueError(
                    "Scoped fidelity requires unique, increasing target facet identifiers."
                )
            if not isinstance(source_scope_id, str) or not source_scope_id:
                raise ValueError(
                    "Scoped fidelity requires its actual original source scope identity."
                )
        elif source_scope_id is not None:
            raise ValueError(
                "A source scope identity requires an explicit target facet scope."
            )
        if projection_coverage is not None and (
            not isinstance(projection_coverage, ImplicitProjectionCoverageEvidence)
            or projection_coverage.binding.binding_id != binding.binding_id
        ):
            raise ValueError(
                "Projection evidence is not bound to this fidelity certificate."
            )
        if domain_coverage is not None and (
            not isinstance(domain_coverage, DomainCoverageCertificate)
            or domain_coverage.binding.binding_id != binding.binding_id
        ):
            raise ValueError(
                "Mapped domain evidence is not bound to this fidelity certificate."
            )
        if chart_coverage is not None and (
            not isinstance(chart_coverage, SourceBoundaryChartCover)
            or (chart_coverage.source_id, chart_coverage.source_revision)
            != (binding.source_id, binding.source_revision)
        ):
            raise ValueError("Chart cover is not bound to this fidelity source revision.")
        forward = parse(semantics[0], SourceBoundSemantics, "semantics")
        backward = parse(semantics[1], SourceBoundSemantics, "semantics")
        findings_ = tuple(findings)
        if (forward, backward) != ("certified", "certified") and not findings_:
            findings_ = (
                MeshCertificateFinding("sampled_fidelity", "unresolved", "mesh"),
            )
        self.binding = binding
        self.projection_coverage = projection_coverage
        self.domain_coverage = domain_coverage
        self.chart_coverage = chart_coverage
        self.status = _certificate_status(findings_)
        self.findings = findings_
        self.tolerance = float(tolerance)
        self.mesh_to_source_semantics = forward
        self.source_to_mesh_semantics = backward
        self.mesh_to_source_lower, self.mesh_to_source_upper = (
            float(mesh_to_source[1]),
            float(mesh_to_source[0]),
        )
        self.source_to_mesh_lower, self.source_to_mesh_upper = (
            float(source_to_mesh[1]),
            float(source_to_mesh[0]),
        )
        self.sample_order = sample_order
        self.mesh_sample_count, self.source_sample_count = sample_counts
        self.target_facet_ids = target_facet_ids
        self.source_scope_id = source_scope_id
        self.certificate_id = canonical_fingerprint(
            {
                "kind": "source-fidelity-certificate",
                "binding": binding.binding_id,
                "status": self.status,
                "findings": [value.finding_id for value in findings_],
                "tolerance": self.tolerance,
                "semantics": (forward, backward),
                # Unresolved directions are unbounded (inf), which JSON cannot
                # carry; the exact float repr is the canonical identity.
                "mesh_to_source": tuple(repr(float(value)) for value in mesh_to_source),
                "source_to_mesh": tuple(repr(float(value)) for value in source_to_mesh),
                "sample_order": sample_order,
                "sample_counts": tuple(sample_counts),
                "projection_coverage": None
                if projection_coverage is None
                else projection_coverage.evidence_id,
                "domain_coverage": None
                if domain_coverage is None
                else domain_coverage.certificate_id,
                "chart_coverage": None
                if chart_coverage is None
                else chart_coverage.cover_id,
                **(
                    {}
                    if target_facet_ids is None
                    else {
                        "target_facets": target_facet_ids,
                        "source_scope": source_scope_id,
                    }
                ),
            }
        )


# Exact arithmetic ------------------------------------------------------------------


def _dyadic_integers(values: np.ndarray, /) -> tuple[np.ndarray, int]:
    """Integers ``n`` and one exponent ``e`` with ``values == n * 2**e`` exactly."""

    array = np.asarray(values, dtype=np.float64)
    mantissa, exponent = np.frexp(array)
    significand = np.ldexp(mantissa, 53).astype(np.int64)
    shift = exponent.astype(np.int64) - 53
    nonzero = significand != 0
    base = int(np.min(shift[nonzero])) if np.any(nonzero) else 0
    shift = np.where(nonzero, shift - base, 0)
    integers = np.asarray(
        [
            int(value) << int(amount)
            for value, amount in zip(
                significand.reshape(-1).tolist(), shift.reshape(-1).tolist(), strict=True
            )
        ],
        dtype=object,
    ).reshape(array.shape)
    return integers, base


def _det2(u: np.ndarray, v: np.ndarray, /) -> np.ndarray:
    return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]


def _det3(u: np.ndarray, v: np.ndarray, w: np.ndarray, /) -> np.ndarray:
    return (
        u[..., 0] * (v[..., 1] * w[..., 2] - v[..., 2] * w[..., 1])
        - u[..., 1] * (v[..., 0] * w[..., 2] - v[..., 2] * w[..., 0])
        + u[..., 2] * (v[..., 0] * w[..., 1] - v[..., 1] * w[..., 0])
    )


def _scaled_float(value: int, exponent: int, denominator: int, /) -> float:
    return float(Fraction(value) * Fraction(2) ** exponent / denominator)


def _signs(result: PredicateResult, /) -> tuple[np.ndarray, np.ndarray]:
    signs = np.asarray(result.signs, dtype=np.int16)
    certain = np.asarray(result.certain, dtype=np.bool_)
    return signs, certain


def _exact_step(
    exact: bool, work: int, storage: int = 0, /
) -> AbstractContextManager[None]:
    """Charge an exact-source step to the request ledger; binary64 steps are unmetered."""
    return exact_charge(work, storage) if exact else nullcontext()


def _orient2d(a: np.ndarray, b: np.ndarray, c: np.ndarray, /) -> tuple:
    mode = resolve_host_predicate_mode(PredicateMode.EXACT)
    return _signs(orient2d(a, b, c, mode=mode))


def _orient3d(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, /) -> tuple:
    mode = resolve_host_predicate_mode(PredicateMode.EXACT)
    return _signs(orient3d(a, b, c, d, mode=mode))


def _degenerate_simplices(points: np.ndarray, rows: np.ndarray, /) -> np.ndarray:
    """Exactly degenerate segments (2-D) or collinear triangles (3-D)."""

    corners = points[rows]
    if rows.shape[1] == 2:
        return np.all(corners[:, 0] == corners[:, 1], axis=-1)
    degenerate = np.ones((rows.shape[0],), dtype=np.bool_)
    for axes in ((0, 1), (0, 2), (1, 2)):
        projected = corners[:, :, axes]
        sign, _ = _orient2d(projected[:, 0], projected[:, 1], projected[:, 2])
        degenerate &= sign == 0
    return degenerate


# Mesh preparation ------------------------------------------------------------------


@dataclass(frozen=True)
class _Facets:
    """Oriented facet occurrences (outward from their cell) and their pairing."""

    rows: np.ndarray
    signs: np.ndarray
    cells: np.ndarray
    group: np.ndarray
    counts: np.ndarray
    order: np.ndarray
    groups: tuple[np.ndarray, ...]

    @property
    def boundary(self) -> np.ndarray:
        return self.counts[self.group] == 1


def _cell_global_ids(mesh: CellMesh, /) -> np.ndarray:
    return np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks]
    )


def _coordinate_scope(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    budget: CoordinateEnclosureBudget,
    /,
) -> tuple[CoordinateMapScope, np.ndarray, CoordinateSourceBank]:
    """Prepare scope inside the owning request ledger."""
    with budget.activate():
        return _coordinate_scope_active(mesh, geometry)


def _coordinate_scope_active(
    mesh: CellMesh, geometry: CellGeometrySpec, /
) -> tuple[CoordinateMapScope, np.ndarray, CoordinateSourceBank]:
    """Affine scope with exact vertex binding, or the cells of mapped blocks."""

    from contextlib import nullcontext

    from ..discretization._coordinate_enclosure import (
        _COORDINATE_BUDGET,
        coordinate_corner_images,
        coordinate_polynomials,
        coordinate_scope_key,
        evaluate,
        multi_affine_coordinates,
        prepared_coordinate_source_bank,
        restrict_coordinates,
    )
    from ..discretization.fem._reference import FiniteElementSpec

    ledger = _COORDINATE_BUDGET.get()
    scope_key = None
    if ledger is not None:
        scope_key = coordinate_scope_key(mesh, geometry)
        cached = ledger.scope_cache.get(scope_key)
        if cached is not None:
            ledger.reserve(0)
            has_mapped, cells, source_values = cached
            return ("mapped" if has_mapped else "affine"), cells, source_values

    if geometry.exact_source is None:
        elements, routes, coordinates = geometry.resolve(mesh)
    else:
        elements, routes, coordinates = geometry._resolve(
            mesh, exact_source_prepared=True
        )
    source_values = prepared_coordinate_source_bank(geometry)
    values = np.asarray(coordinates, dtype=np.float64)
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    mapped = []
    target_corners: dict[int, tuple[Fraction, ...]] = {}
    cursor = 0
    for block, element, route in zip(mesh.blocks, elements, routes, strict=True):
        count = block.cell_count
        vertices = np.asarray(block.vertices, dtype=np.int64)
        if isinstance(element, CellVertexGeometryElement):
            valid = np.asarray(block.vertex_valid, dtype=np.bool_)
            if route.shape[1] != vertices.shape[1]:
                raise ValueError(
                    "Vertex coordinate routes must match the actual cell corner layout."
                )
            if not np.array_equal(
                values[np.asarray(route, dtype=np.int64)][valid], points[vertices][valid]
            ):
                raise ValueError(
                    "Vertex-level coordinate geometry must equal the mesh coordinates."
                )
        elif isinstance(element, FiniteElementSpec) and element.degree == 1:
            reference = tuple(
                tuple(Fraction(float(value)) for value in point)
                for point in reference_cell_topology(block.cell_kind).vertices
            )
            local_values = tuple(
                tuple(source_values[index] for index in row)
                for row in np.asarray(route, dtype=np.int64)
            )
            for cell, local in enumerate(local_values):
                # Complete source maps and corners stay in the same ledger for
                # downstream proofs. Only discarded classification workspace
                # is released between cells; carrier binding is checked anew.
                with nullcontext() if ledger is None else ledger.temporary_scope():
                    simplex_images = (
                        coordinate_corner_images(element, local)
                        if block.cell_kind in ("interval", "triangle", "tetrahedron")
                        else None
                    )
                    prepared = None
                    polynomials: tuple[Polynomial, ...] = ()
                    if simplex_images is None:
                        prepared = multi_affine_coordinates(element, local)
                        if prepared is None:
                            coordinate_values = coordinate_polynomials(element, local)
                            if coordinate_values is None:
                                mapped.append(
                                    np.asarray((cursor + cell,), dtype=np.int64)
                                )
                                continue
                            polynomials = coordinate_values
                        else:
                            polynomials = prepared[0]
                    nonbinary_corner = False
                    for corner, (vertex, point) in enumerate(
                        zip(vertices[cell], reference, strict=True)
                    ):
                        parameter = (
                            (Fraction(1, 2), Fraction(1, 2), Fraction(1))
                            if block.cell_kind == "pyramid" and point[2] == 1
                            else point
                        )
                        if simplex_images is not None:
                            image = simplex_images[corner]
                        elif prepared is not None:
                            image = prepared[1][corner]
                        else:
                            image = tuple(
                                evaluate(value, parameter) for value in polynomials
                            )
                        if ledger is not None:
                            ledger.reserve(len(image))
                        if not np.array_equal(
                            np.asarray(
                                tuple(float(value) for value in image), dtype=np.float64
                            ).view(np.uint64),
                            points[vertex].view(np.uint64),
                        ):
                            raise ValueError(
                                "Vertex-level coordinate carrier must be the correctly rounded source image."
                            )
                        prior = target_corners.get(int(vertex))
                        if prior is not None and prior != image:
                            raise ValueError(
                                "Coordinate charts disagree on an exact shared vertex image."
                            )
                        if prior is None and ledger is not None:
                            ledger.reserve(1, 256 + 128 * len(image))
                        target_corners[int(vertex)] = image
                        nonbinary_corner |= any(
                            Fraction(float(value)) != value for value in image
                        )
                    if simplex_images is not None:
                        if nonbinary_corner:
                            mapped.append(np.asarray((cursor + cell,), dtype=np.int64))
                        continue
                    physical = (
                        restrict_coordinates(
                            polynomials,
                            "pyramid",
                            np.zeros((3,), dtype=np.float64),
                            np.eye(3, dtype=np.float64),
                        )
                        if block.cell_kind == "pyramid"
                        else polynomials
                    )
                    if ledger is not None and physical is not None:
                        ledger.reserve(sum(len(polynomial) for polynomial in physical))
                    if (
                        nonbinary_corner
                        or physical is None
                        or any(
                            sum(index) > 1
                            for polynomial in physical
                            for index in polynomial
                        )
                    ):
                        mapped.append(np.asarray((cursor + cell,), dtype=np.int64))
        else:
            mapped.append(np.arange(cursor, cursor + count))
        cursor += count
    if ledger is not None:
        ledger.reserve(0, 128 + 8 * sum(value.size for value in mapped))
    cells = np.concatenate(mapped) if mapped else np.empty((0,), dtype=np.int64)
    if not cells.size and target_corners and geometry.exact_source is not None:
        if len(target_corners) != points.shape[0]:
            raise ValueError(
                "Affine exact coordinate charts must bind every target vertex."
            )
        source_values = tuple(target_corners[index] for index in range(points.shape[0]))
        if ledger is not None:
            ledger.reserve(points.size, 128 + 8 * points.size)
            ledger.retain_basis(source_values)
    if ledger is not None and scope_key is not None:
        cells.flags.writeable = False
        prepared_scope = (bool(cells.size), cells, source_values)
        ledger.retain_basis((scope_key, prepared_scope))
        ledger.scope_cache[scope_key] = prepared_scope
    return ("mapped" if cells.size else "affine"), cells, source_values


def _csr_rows(offsets: np.ndarray, values: np.ndarray, /) -> np.ndarray:
    sizes = np.diff(offsets)
    rows = np.repeat(np.arange(sizes.size), sizes)
    rank = np.arange(values.size) - offsets[rows]
    result = np.full((sizes.size, int(np.max(sizes))), -1, dtype=np.int64)
    result[rows, rank] = values
    return result


def _padded(blocks: list[np.ndarray], /) -> np.ndarray:
    width = max(value.shape[1] for value in blocks)
    return np.concatenate(
        [
            np.pad(value, ((0, 0), (0, width - value.shape[1])), constant_values=-1)
            for value in blocks
        ]
    )


def _polygon_edges(rows: np.ndarray, valid: np.ndarray, /) -> tuple:
    """Directed edges of counterclockwise loops with valid-prefix padding."""

    lengths = np.sum(valid, axis=1)
    column = np.arange(rows.shape[1])
    following = rows[np.arange(rows.shape[0])[:, None], (column + 1) % lengths[:, None]]
    slots = column[None, :] < lengths[:, None]
    owners = np.broadcast_to(np.arange(rows.shape[0])[:, None], rows.shape)
    return np.stack((rows[slots], following[slots]), axis=1), owners[slots]


def _polyhedral_facets(
    connectivity: PolyhedralConnectivity, owners: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    faces = np.asarray(connectivity.cell_face_values, dtype=np.int64)
    offsets = np.asarray(connectivity.cell_face_offsets, dtype=np.int64)
    signs = np.asarray(connectivity.cell_face_sign_values, dtype=np.float64)
    incidence = np.repeat(np.arange(offsets.size - 1), np.diff(offsets))
    chosen = np.isin(incidence, owners)
    loops = _csr_rows(
        np.asarray(connectivity.face_vertex_offsets, dtype=np.int64),
        np.asarray(connectivity.face_vertex_values, dtype=np.int64),
    )[faces[chosen]]
    lengths = np.sum(loops >= 0, axis=1)
    reverse = signs[chosen] < 0.0
    for length in np.unique(lengths[reverse]):
        rows = reverse & (lengths == length)
        loops[rows, :length] = loops[rows, :length][:, ::-1]
    return loops, incidence[chosen]


def _mesh_facets(
    mesh: CellMesh,
    /,
    *,
    geometry: CellGeometrySpec | None = None,
) -> _Facets:
    dimension = mesh.topological_dimension
    from ..discretization import _coordinate_enclosure as algebra

    ledger = algebra._COORDINATE_BUDGET.get()
    source_key = None
    if ledger is not None and geometry is not None:
        ledger.reserve(1)
        source_key = algebra.coordinate_scope_key(mesh, geometry)
        cached = ledger.facet_cache.get(source_key)
        if cached is not None:
            return cached
        if any(block.cell_kind == "polyhedron" for block in mesh.blocks):
            connectivity = mesh.connectivity
            if not isinstance(connectivity, PolyhedralConnectivity):
                raise TypeError("Polyhedral blocks require polyhedral connectivity.")
            occurrences = connectivity.cell_face_values.size
        else:
            occurrences = sum(
                block.vertices.size
                if dimension <= 2
                else block.cell_count
                * len(reference_cell_topology(block.cell_kind).entities[2])
                for block in mesh.blocks
            )
        ledger.reserve(0, 1024 + 256 * occurrences)
    rows = []
    signs = []
    cells = []
    cursor = 0
    for block in mesh.blocks:
        count = block.cell_count
        owners = np.arange(cursor, cursor + count)
        vertices = np.asarray(block.vertices, dtype=np.int64)
        if block.cell_kind == "polyhedron":
            connectivity = mesh.connectivity
            if not isinstance(connectivity, PolyhedralConnectivity):
                raise TypeError("Polyhedral blocks require polyhedral connectivity.")
            loops, incidence = _polyhedral_facets(connectivity, owners)
            rows.append(loops)
            cells.append(incidence)
            signs.append(np.zeros((incidence.size,), dtype=np.int8))
        elif dimension == 1:
            rows.append(vertices.reshape(-1, 1))
            cells.append(np.repeat(owners, 2))
            signs.append(np.tile(np.asarray((-1, 1), dtype=np.int8), count))
        elif dimension == 2:
            edges, local = _polygon_edges(
                vertices, np.asarray(block.vertex_valid, dtype=np.bool_)
            )
            rows.append(edges)
            cells.append(owners[local])
            signs.append(np.zeros((local.size,), dtype=np.int8))
        else:
            local_facets = reference_cell_topology(block.cell_kind).entities[2]
            for face in local_facets:
                values = np.full((count, 4), -1, dtype=np.int64)
                values[:, : len(face)] = vertices[:, np.asarray(face)]
                rows.append(values)
                cells.append(owners)
                signs.append(np.zeros((count,), dtype=np.int8))
        cursor += count
    facet_rows = _padded(rows)
    keys = np.sort(np.where(facet_rows < 0, np.iinfo(np.int64).max, facet_rows), axis=1)
    _, group, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    incidence = group.reshape(-1)
    if ledger is not None and source_key is not None:
        ledger.reserve(incidence.size, 256 + 32 * incidence.size)
    order = np.argsort(incidence, kind="stable")
    groups = tuple(np.split(order, np.flatnonzero(np.diff(incidence[order])) + 1))
    prepared = _Facets(
        facet_rows,
        np.concatenate(signs),
        np.concatenate(cells),
        incidence,
        counts,
        order,
        groups,
    )
    if ledger is not None and source_key is not None:
        arrays = (
            prepared.rows,
            prepared.signs,
            prepared.cells,
            prepared.group,
            prepared.counts,
            prepared.order,
        )
        for array in (*arrays, *prepared.groups):
            array.flags.writeable = False
        ledger.retain_basis((source_key, prepared, arrays, prepared.groups))
        ledger.facet_cache[source_key] = prepared
    return prepared


def _sorted_keys(rows: np.ndarray, width: int, /) -> np.ndarray:
    padded = np.full((rows.shape[0], width), -1, dtype=np.int64)
    padded[:, : rows.shape[1]] = rows
    return np.sort(np.where(padded < 0, np.iinfo(np.int64).max, padded), axis=1)


def _facet_entity_ids(mesh: CellMesh, rows: np.ndarray, /) -> np.ndarray:
    """Mesh facet entity global ids of facet vertex rows."""

    dimension = mesh.topological_dimension
    if dimension == 1:
        return np.asarray(mesh.vertex_global_ids, dtype=np.int64)[rows[:, 0]]
    connectivity = mesh.connectivity
    if isinstance(connectivity, PolyhedralConnectivity):
        table = _csr_rows(
            np.asarray(connectivity.face_vertex_offsets, dtype=np.int64),
            np.asarray(connectivity.face_vertex_values, dtype=np.int64),
        )
    elif isinstance(connectivity, PolygonalConnectivity):
        table = np.asarray(connectivity.edges, dtype=np.int64)
    elif isinstance(connectivity, (TetrahedralConnectivity, HexahedralConnectivity)):
        table = np.asarray(connectivity.faces, dtype=np.int64)
    else:
        raise TypeError("Unsupported CellMesh connectivity for facet identities.")
    width = max(table.shape[1], rows.shape[1])
    known = _sorted_keys(table, width)
    query = _sorted_keys(rows, width)
    view = np.dtype((np.void, known.dtype.itemsize * width))
    known_view = np.ascontiguousarray(known).view(view).reshape(-1)
    query_view = np.ascontiguousarray(query).view(view).reshape(-1)
    order = np.argsort(known_view, kind="stable")
    position = order[np.searchsorted(known_view[order], query_view)]
    if not np.array_equal(known[position], query):
        raise RuntimeError("Mesh facets are missing from the mesh topology.")
    return np.asarray(mesh.entity_set(dimension - 1).entity_ids, dtype=np.int64)[position]


def _planar_loops(
    points: np.ndarray, loops: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    """Exact coplanarity of padded 3-D loops: ``(nonplanar, undecided)``."""

    lengths = np.sum(loops >= 0, axis=1)
    nonplanar = np.zeros((loops.shape[0],), dtype=np.bool_)
    undecided = np.zeros((loops.shape[0],), dtype=np.bool_)
    for length in np.unique(lengths[lengths >= 4]):
        selected = np.flatnonzero(lengths == length)
        corners = points[loops[selected, :length]]
        # All tetrahedra with apex at corner zero are flat iff the loop is planar.
        for first, second, third in combinations(range(1, length), 3):
            sign, certain = _orient3d(
                corners[:, 0], corners[:, first], corners[:, second], corners[:, third]
            )
            nonplanar[selected] |= certain & (sign != 0)
            undecided[selected] |= ~certain
    return nonplanar, undecided & ~nonplanar


def _loop_triangles(
    points: np.ndarray, loops: np.ndarray, capacity: int, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Exact triangles of planar 3-D loops with owners and failing loops.

    Returns ``(triangles, owners, nonplanar, undecided)``; nonplanar loops have a
    curved (mapped) boundary and self-intersecting loops count as nonplanar
    faces with no admissible affine triangulation.
    """

    # Geometry is importable before meshing; the exact polygon triangulation
    # shares the audit owner's predicates rather than a second convention.
    from ..meshing._audit_topology import _polygon_loop_triangles

    if points.dtype == object:
        from ._exact_polyhedral_geometry import triangulate_loop, triangulation_charge

        exact_triangles, exact_owners = [], []
        failed = np.zeros(loops.shape[0], dtype=np.bool_)
        exhausted = np.zeros_like(failed)
        used = 0
        for owner, row in enumerate(loops):
            indices = tuple(row[row >= 0].tolist())
            used += len(indices) ** 3
            if used > capacity:
                exhausted[owner:] = True
                break
            if len(indices) == 3:
                # A nondegenerate triangle already is its oriented
                # triangulation; retain the exact plane/rank check without
                # repeating it or preparing projected polygon ear clipping.
                from ._planar_coverage import plane_key

                corners = tuple(
                    tuple(value for value in points[index]) for index in indices
                )
                if plane_key(corners) is None:
                    failed[owner] = True
                    continue
                exact_triangles.append(indices)
                exact_owners.append(owner)
                continue
            # Charged outside the refusal handler: exhausting the request
            # ledger is a resource decision, never a failed facet.
            with exact_charge(*triangulation_charge(points, indices)):
                try:
                    pieces = triangulate_loop(points, indices)
                except ValueError:
                    failed[owner] = True
                    continue
            exact_triangles.extend(pieces)
            exact_owners.extend((owner,) * len(pieces))
        return (
            np.asarray(exact_triangles, dtype=np.int64).reshape((-1, 3)),
            np.asarray(exact_owners, dtype=np.int64),
            failed,
            exhausted,
        )
    lengths = np.sum(loops >= 0, axis=1)
    nonplanar, undecided = _planar_loops(points, loops)
    triangles = [np.empty((0, 3), dtype=np.int64)]
    owners = [np.empty((0,), dtype=np.int64)]
    for length in np.unique(lengths):
        selected = np.flatnonzero((lengths == length) & ~nonplanar & ~undecided)
        rows = loops[selected, :length]
        if length == 3:
            triangles.append(rows)
            owners.append(selected)
            continue
        if not selected.size:
            continue
        clipped, local, crossing, unknown, _, exceeded = _polygon_loop_triangles(
            points, rows, capacity
        )
        triangles.append(clipped)
        owners.append(selected[local])
        nonplanar[selected[crossing]] = True
        undecided[selected[unknown]] = True
        if exceeded:
            undecided[selected] = True
    return np.concatenate(triangles), np.concatenate(owners), nonplanar, undecided


def _candidate_pairs(
    lower: np.ndarray,
    upper: np.ndarray,
    capacity: int,
    /,
    *,
    other: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Closed-box candidate pairs (``first < second`` for self queries)."""

    empty = np.empty((0,), dtype=np.int64)
    if lower.shape[0] == 0 or (other is not None and other[0].shape[0] == 0):
        return empty, empty, False
    first_bvh = prepare_bvh(lower, upper, dtype=np.float64)
    second_bvh = first_bvh if other is None else prepare_bvh(*other, dtype=np.float64)
    firsts = [empty]
    seconds = [empty]
    remaining = capacity
    for first, second in bvh_overlap_pair_blocks(
        first_bvh, second_bvh, include_touching=True
    ):
        if other is None:
            keep = first < second
            first = first[keep]
            second = second[keep]
        if first.size > remaining:
            return np.concatenate(firsts), np.concatenate(seconds), True
        remaining -= first.size
        firsts.append(first)
        seconds.append(second)
    return np.concatenate(firsts), np.concatenate(seconds), False


def _boxes(points: np.ndarray, rows: np.ndarray, /) -> tuple[np.ndarray, np.ndarray]:
    corners = points[rows]
    if points.dtype == object:
        lower, upper = (
            np.asarray(np.min(corners, axis=1), dtype=np.float64),
            np.asarray(np.max(corners, axis=1), dtype=np.float64),
        )
        return np.nextafter(lower, -np.inf), np.nextafter(upper, np.inf)
    return np.min(corners, axis=1), np.max(corners, axis=1)


# Global embedding ------------------------------------------------------------------


@dataclass
class _EmbeddingState:
    findings: list[MeshCertificateFinding]
    checks: list[str]
    candidate_pairs: int = 0
    ray_tests: int = 0
    subdivision_pieces: int = 0
    subdivision_depth: int = 0
    boundary_facets: int = 0
    shells: int = 0
    periodic_images: int = 0
    source_expression_work_units: int = 0
    source_expression_peak_bytes: int = 0
    # Premise (a) of the boundary-degree theorem: the bound validity
    # certificate proves det DF > 0 on every closed reference cell.
    closed_cell_orientation: bool = False
    boundary_degree: MappedBoundaryDegreeEvidence | None = None

    def add(
        self,
        check: str,
        status: MeshFindingStatus,
        kind: MeshCertificateEntityKind,
        identifiers: np.ndarray | tuple[int, ...] = (),
        /,
        *,
        resource_error: CoordinateEnclosureResourceError | None = None,
        expression_budget: CoordinateEnclosureBudget | None = None,
    ) -> None:
        values = tuple(int(value) for value in np.asarray(identifiers).reshape(-1))
        achieved: tuple[tuple[str, int], ...] = ()
        if resource_error is not None:
            achieved = (("completed", resource_error.completed),)
            if expression_budget is not None:
                achieved += (
                    ("source_expression_work_units", expression_budget.work_units),
                    ("source_expression_peak_bytes", expression_budget.peak_bytes_upper),
                    (
                        "source_expression_native_charged_work_units",
                        expression_budget.native_charged_work_units,
                    ),
                    (
                        "source_expression_retained_basis_bytes_at_catch",
                        expression_budget.retained_basis_bytes,
                    ),
                    (
                        "source_expression_temporary_bytes_upper_at_catch",
                        expression_budget.temporary_bytes_upper,
                    ),
                )
        self.findings.append(
            MeshCertificateFinding(
                check,
                status,
                kind,
                values,
                resource=None if resource_error is None else resource_error.resource,
                requested=()
                if resource_error is None
                else (
                    ("limit", resource_error.limit),
                    ("requested", resource_error.requested),
                ),
                achieved=achieved,
            )
        )

    @property
    def clean(self) -> bool:
        return not self.findings


def _validity_findings(
    state: _EmbeddingState, validity: CellValidityCertificate, cell_ids: np.ndarray, /
) -> None:
    state.checks.append("cell_validity")
    state.closed_cell_orientation = validity.all_certified
    status = np.asarray(validity.status)
    invalid = status == CellValidityStatus.INVALID
    unresolved = status == CellValidityStatus.UNRESOLVED
    if np.any(invalid):
        state.add("invalid_cell", "violated", "cell", cell_ids[invalid])
    if np.any(unresolved):
        state.add("unresolved_cell_validity", "unresolved", "cell", cell_ids[unresolved])


def _pairing_findings(
    state: _EmbeddingState, mesh: CellMesh, facets: _Facets, junctions: np.ndarray, /
) -> None:
    state.checks.append("facet_pairing")
    order = facets.order
    same = facets.group[order[1:]] == facets.group[order[:-1]]
    first = order[:-1][same]
    second = order[1:][same]
    # Declared curve-network junctions are the only vertices where more than two
    # intervals may meet; valence-two junctions keep the orientation check.
    declared = (
        np.isin(
            np.asarray(mesh.vertex_global_ids, dtype=np.int64)[facets.rows[:, 0]],
            junctions,
        )
        if junctions.size
        else np.zeros(facets.rows.shape[0], dtype=np.bool_)
    )
    nonmanifold = (facets.counts[facets.group] > 2) & ~declared
    twice = facets.counts[facets.group[first]] == 2
    first = first[twice]
    second = second[twice]
    dimension = mesh.topological_dimension
    if dimension == 1:
        consistent = facets.signs[first] != facets.signs[second]
    elif dimension == 2:
        consistent = (facets.rows[first, 0] == facets.rows[second, 1]) & (
            facets.rows[first, 1] == facets.rows[second, 0]
        )
    else:
        rows = facets.rows
        lengths = np.sum(rows >= 0, axis=1)
        position = np.argmin(np.where(rows >= 0, rows, np.iinfo(np.int64).max), axis=1)
        index = np.arange(rows.shape[0])
        successor = rows[index, (position + 1) % lengths]
        predecessor = rows[index, (position - 1) % lengths]
        consistent = successor[first] == predecessor[second]
    if np.any(nonmanifold):
        state.add(
            "nonmanifold_facet",
            "violated",
            "facet",
            _facet_entity_ids(mesh, facets.rows[nonmanifold]),
        )
    if not np.all(consistent):
        state.add(
            "inconsistent_facet_orientation",
            "violated",
            "facet",
            _facet_entity_ids(mesh, facets.rows[first[~consistent]]),
        )


def _segment_contacts(
    points: np.ndarray, first: np.ndarray, second: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    """Exact planar or spatial segment contacts beyond shared mesh vertices."""

    if points.shape[1] == 3:
        sign, certain = _orient3d(
            points[first[:, 0]],
            points[first[:, 1]],
            points[second[:, 0]],
            points[second[:, 1]],
        )
        hit = np.zeros(first.shape[0], dtype=np.bool_)
        planar = np.flatnonzero(certain & (sign == 0))
        if planar.size:
            import sys

            from ..discretization._coordinate_enclosure import _COORDINATE_BUDGET

            ledger = _COORDINATE_BUDGET.get()
            if ledger is not None:
                ledger.reserve(
                    certain.size + planar.size, certain.nbytes + sys.getsizeof(certain)
                )
            # Native/JAX predicate results are immutable borrowed evidence. Only
            # this derived aggregate mask is owned and changed by the proof.
            certain = certain.copy()
            projected_hit = np.ones(planar.size, dtype=np.bool_)
            projected_certain = np.ones(planar.size, dtype=np.bool_)
            # A coplanar pair has an injective coordinate-plane projection on
            # its affine hull (also for collinear pairs). Thus contact beyond
            # the shared vertices must hold in all three projections. Skew
            # pairs are already disjoint by the exact four-point orientation.
            # These slices are views, not three copies of the vertex bank.
            for projection in (points[:, :2], points[:, 1:], points[:, ::2]):
                contact, resolved = _segment_contacts(
                    projection, first[planar], second[planar]
                )
                # A projection constant on both segments cannot distinguish
                # overlap from their shared endpoint; it imposes no restriction
                # when both constant images are the same exact point.
                collapsed = (
                    np.all(
                        projection[first[planar, 0]] == projection[first[planar, 1]],
                        axis=1,
                    )
                    & np.all(
                        projection[second[planar, 0]] == projection[second[planar, 1]],
                        axis=1,
                    )
                    & np.all(
                        projection[first[planar, 0]] == projection[second[planar, 0]],
                        axis=1,
                    )
                )
                contact |= collapsed
                resolved |= collapsed
                projected_hit &= contact
                projected_certain &= resolved
            hit[planar] = projected_hit
            certain[planar] &= projected_certain
        return hit & certain, certain

    mode = resolve_host_predicate_mode(PredicateMode.EXACT)
    status = np.asarray(
        segment_intersections_2d(
            points[first[:, 0]],
            points[first[:, 1]],
            points[second[:, 0]],
            points[second[:, 1]],
            mode=mode,
        ).status
    )
    shared = np.sum(first[:, :, None] == second[:, None, :], axis=(1, 2))
    certain = status != SegmentIntersectionStatus.UNCERTAIN
    hit = np.where(
        shared == 0,
        status != SegmentIntersectionStatus.DISJOINT,
        np.where(
            shared == 1, status == SegmentIntersectionStatus.COLLINEAR_OVERLAP, True
        ),
    )
    return hit & certain, certain


# Exact contact pairs per batch: the integer predicate kernels hold one batch
# of bigint temporaries at a time, which bounds the transient working set.
_EXACT_CONTACT_BATCH = 4096


def _exact_box_charge(points: np.ndarray, count: int, /) -> tuple[int, int]:
    """Work and transient storage of ``_boxes`` over ``count`` exact items.

    Per item twelve rational comparisons (each cross-multiplies two integers
    of at most ``2 b + 1`` bits, ``b`` the source bits) and six outward float
    conversions, with sixteen gathered slots; the bit scan charges itself.
    """
    return 18 * count, 2 * bigint_bytes(2 * exact_bits(points) + 1) + 8 * 16 * count


def _exact_shared_edge_disjoint(
    points: np.ndarray, first: np.ndarray, second: np.ndarray, /
) -> np.ndarray:
    """Prove edge-adjacent affine triangles meet only on their shared edge."""
    result = np.zeros((first.shape[0],), dtype=np.bool_)
    for row, (left, right) in enumerate(zip(first, second, strict=True)):
        shared = tuple(sorted(set(left.tolist()).intersection(right.tolist())))
        if len(shared) != 2:
            continue
        free_left = tuple(set(left.tolist()).difference(shared))
        free_right = tuple(set(right.tolist()).difference(shared))
        if len(free_left) != 1 or len(free_right) != 1:
            continue
        with _exact_step(True, 15):
            a, b = points[list(shared)]
            c, d = points[free_left[0]], points[free_right[0]]
            u = tuple(y - x for x, y in zip(a, b, strict=True))
            v = tuple(y - x for x, y in zip(a, c, strict=True))
            normal = (
                u[1] * v[2] - u[2] * v[1],
                u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0],
            )
            distance = sum(
                (
                    value * (target - source)
                    for value, source, target in zip(normal, a, d, strict=True)
                ),
                Fraction(0),
            )
        if distance:
            result[row] = True
            continue
        with _exact_step(True, 6):
            axis = max(range(3), key=lambda index: abs(normal[index]))
            axes = tuple(index for index in range(3) if index != axis)
            side_c = (b[axes[0]] - a[axes[0]]) * (c[axes[1]] - a[axes[1]]) - (
                b[axes[1]] - a[axes[1]]
            ) * (c[axes[0]] - a[axes[0]])
            side_d = (b[axes[0]] - a[axes[0]]) * (d[axes[1]] - a[axes[1]]) - (
                b[axes[1]] - a[axes[1]]
            ) * (d[axes[0]] - a[axes[0]])
            result[row] = side_c * side_d < 0
    return result


def _exact_shared_vertex_axis_disjoint(
    points: np.ndarray, first: np.ndarray, second: np.ndarray, /
) -> np.ndarray:
    """Prove vertex-adjacent triangles occupy opposite open coordinate halfspaces."""
    result = np.zeros((first.shape[0],), dtype=np.bool_)
    for row, (left, right) in enumerate(zip(first, second, strict=True)):
        shared = tuple(set(left.tolist()).intersection(right.tolist()))
        if len(shared) != 1:
            continue
        vertex = points[shared[0]]
        free_left = points[list(set(left.tolist()).difference(shared))]
        free_right = points[list(set(right.tolist()).difference(shared))]
        for axis in range(3):
            with _exact_step(True, 4):
                left_delta = tuple(point[axis] - vertex[axis] for point in free_left)
                right_delta = tuple(point[axis] - vertex[axis] for point in free_right)
            if (
                min(left_delta) > 0
                and max(right_delta) < 0
                or max(left_delta) < 0
                and min(right_delta) > 0
            ):
                result[row] = True
                break
    return result


def _exact_triangle_axis_disjoint(
    points: np.ndarray, first: np.ndarray, second: np.ndarray, /
) -> np.ndarray:
    """Prove disjoint affine triangles by a strict exact separating axis."""
    result = np.zeros((first.shape[0],), dtype=np.bool_)
    for row, (left, right) in enumerate(zip(first, second, strict=True)):
        a, b = points[left], points[right]
        with _exact_step(True, 18):
            edges_a = tuple(
                tuple(y - x for x, y in zip(a[index], a[(index + 1) % 3], strict=True))
                for index in range(3)
            )
            edges_b = tuple(
                tuple(y - x for x, y in zip(b[index], b[(index + 1) % 3], strict=True))
                for index in range(3)
            )
            normals = (
                (
                    edges_a[0][1] * edges_a[1][2] - edges_a[0][2] * edges_a[1][1],
                    edges_a[0][2] * edges_a[1][0] - edges_a[0][0] * edges_a[1][2],
                    edges_a[0][0] * edges_a[1][1] - edges_a[0][1] * edges_a[1][0],
                ),
                (
                    edges_b[0][1] * edges_b[1][2] - edges_b[0][2] * edges_b[1][1],
                    edges_b[0][2] * edges_b[1][0] - edges_b[0][0] * edges_b[1][2],
                    edges_b[0][0] * edges_b[1][1] - edges_b[0][1] * edges_b[1][0],
                ),
            )
        axes = [*normals]
        axes.extend(
            (
                left_edge[1] * right_edge[2] - left_edge[2] * right_edge[1],
                left_edge[2] * right_edge[0] - left_edge[0] * right_edge[2],
                left_edge[0] * right_edge[1] - left_edge[1] * right_edge[0],
            )
            for left_edge in edges_a
            for right_edge in edges_b
        )
        for axis in axes:
            if not any(axis):
                continue
            with _exact_step(True, 18):
                projection_a = tuple(
                    sum(
                        (
                            weight * value
                            for weight, value in zip(axis, point, strict=True)
                        ),
                        Fraction(0),
                    )
                    for point in a
                )
                projection_b = tuple(
                    sum(
                        (
                            weight * value
                            for weight, value in zip(axis, point, strict=True)
                        ),
                        Fraction(0),
                    )
                    for point in b
                )
            if max(projection_a) < min(projection_b) or max(projection_b) < min(
                projection_a
            ):
                result[row] = True
                break
    return result


def _exact_pair_contacts(
    points: np.ndarray, first: np.ndarray, second: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    """The binary64 contact algorithm on the integer scaling of exact points.

    One positive common denominator preserves every orientation sign, so the
    canonical predicate algorithm decides exact contact with certain signs that
    charge themselves. Per triangle pair it forms at most two projection
    normals (twenty visits, eighteen integers of at most ``2 b + 3`` bits for
    bank bits ``b``) and gathers at most 128 slots; a segment pair gathers
    eight slots.
    """

    from ..meshing._audit_topology import _triangle_pairs_intersect
    from ._exact_polyhedral_geometry import coordinate_integers

    with exact_charge(0):
        integers, _ = coordinate_integers(points)
        if first.shape[1] == 2:
            with exact_charge(0, 8 * 8 * first.shape[0]):
                return _segment_contacts(integers, first, second)
        bits = exact_bits(integers)
        hit = np.zeros((first.shape[0],), dtype=np.bool_)
        certain = np.ones((first.shape[0],), dtype=np.bool_)
        for start in range(0, first.shape[0], _EXACT_CONTACT_BATCH):
            stop = min(start + _EXACT_CONTACT_BATCH, first.shape[0])
            count = stop - start
            left, right = first[start:stop], second[start:stop]
            shared = np.sum(left[:, :, None] == right[:, None, :], axis=(1, 2))
            # A welded vertex removes two plane-side inputs and four irrelevant
            # edge/triangle preparations; a welded edge leaves only its fold test.
            work = int(np.sum(np.where(shared == 2, 20, np.where(shared == 1, 28, 40))))
            with exact_charge(work, count * (36 * bigint_bytes(2 * bits + 3) + 8 * 128)):
                hit[start:stop], certain[start:stop] = _triangle_pairs_intersect(
                    integers, left, right
                )
        return hit, certain


def _pairwise_contacts(
    state: _EmbeddingState,
    points: np.ndarray,
    items: np.ndarray,
    item_entities: np.ndarray,
    kind: MeshCertificateEntityKind,
    check: str,
    limits: MeshCertificateLimits,
    /,
) -> None:
    """Exact contacts of segment/triangle items beyond combinatorial sharing."""

    from ..meshing._audit_topology import _triangle_pairs_intersect

    state.checks.append(check)
    exact = points.dtype == object
    with _exact_step(
        exact, *(_exact_box_charge(points, items.shape[0]) if exact else (0, 0))
    ):
        lower, upper = _boxes(points, items)
    first, second, exceeded = _candidate_pairs(
        lower, upper, limits.maximum_candidate_pairs - state.candidate_pairs
    )
    state.candidate_pairs += first.size
    if exceeded:
        state.add(f"{check}_capacity", "unresolved", "mesh")
    if not first.size:
        return
    if exact:
        hit = np.zeros(first.shape, dtype=np.bool_)
        certain = np.ones(first.shape, dtype=np.bool_)
        if items.shape[1] == 3:
            selected_first, selected_second = items[first], items[second]
            shared = np.sum(
                selected_first[:, :, None] == selected_second[:, None, :],
                axis=(1, 2),
            )
            adjacent = shared == 2
            if np.any(adjacent):
                proven = _exact_shared_edge_disjoint(
                    points,
                    selected_first[adjacent],
                    selected_second[adjacent],
                )
                adjacent_rows = np.flatnonzero(adjacent)
                certain[adjacent_rows[proven]] = True
                pending = np.ones(first.shape, dtype=np.bool_)
                pending[adjacent_rows[proven]] = False
            else:
                pending = np.ones(first.shape, dtype=np.bool_)
            vertex_adjacent = (shared == 1) & pending
            if np.any(vertex_adjacent):
                proven = _exact_shared_vertex_axis_disjoint(
                    points,
                    selected_first[vertex_adjacent],
                    selected_second[vertex_adjacent],
                )
                vertex_rows = np.flatnonzero(vertex_adjacent)
                pending[vertex_rows[proven]] = False
            separate = (shared == 0) & pending
            if np.any(separate):
                proven = _exact_triangle_axis_disjoint(
                    points,
                    selected_first[separate],
                    selected_second[separate],
                )
                separate_rows = np.flatnonzero(separate)
                pending[separate_rows[proven]] = False
        else:
            pending = np.ones(first.shape, dtype=np.bool_)
        if np.any(pending):
            hit[pending], certain[pending] = _exact_pair_contacts(
                points,
                items[first[pending]],
                items[second[pending]],
            )
    elif items.shape[1] == 2:
        hit, certain = _segment_contacts(points, items[first], items[second])
    else:
        hit, certain = _triangle_pairs_intersect(points, items[first], items[second])
    # Pieces of one facet or cell touch along their own diagonals.
    distinct = item_entities[first] != item_entities[second]
    hit &= certain & distinct
    uncertain = ~certain & distinct
    if np.any(hit):
        state.add(
            check,
            "violated",
            kind,
            np.concatenate((item_entities[first[hit]], item_entities[second[hit]])),
        )
    if np.any(uncertain):
        state.add(
            f"{check}_predicates",
            "unresolved",
            kind,
            np.concatenate(
                (item_entities[first[uncertain]], item_entities[second[uncertain]])
            ),
        )


def _interval_embedding(
    state: _EmbeddingState, mesh: CellMesh, cell_ids: np.ndarray, /
) -> None:
    """Exact 1-D interval disjointness beyond shared vertices."""

    state.checks.append("interval_overlap")
    rows = np.concatenate(
        [np.asarray(block.vertices, dtype=np.int64) for block in mesh.blocks]
    )
    coordinates = np.asarray(mesh.coordinates, dtype=np.float64)[:, 0]
    left = np.minimum(coordinates[rows[:, 0]], coordinates[rows[:, 1]])
    right = np.maximum(coordinates[rows[:, 0]], coordinates[rows[:, 1]])
    left_vertex = np.where(coordinates[rows[:, 0]] <= coordinates[rows[:, 1]], 0, 1)
    order = np.lexsort((right, left))
    left, right = left[order], right[order]
    start = rows[order, left_vertex[order]]
    stop = rows[order, 1 - left_vertex[order]]
    running = np.maximum.accumulate(right)
    owner = np.maximum.accumulate(np.where(right == running, np.arange(right.size), 0))
    previous = running[:-1]
    previous_owner = owner[:-1]
    overlap = left[1:] < previous
    contact = (left[1:] == previous) & (start[1:] != stop[previous_owner])
    bad = overlap | contact
    if np.any(bad):
        state.add(
            "interval_overlap",
            "violated",
            "cell",
            np.concatenate(
                (cell_ids[order[1:][bad]], cell_ids[order[previous_owner[bad]]])
            ),
        )


def _cell_loops(mesh: CellMesh, /) -> np.ndarray:
    """Counterclockwise vertex loops of 2-D cells, padded with ``-1``."""

    loops = []
    for block in mesh.blocks:
        vertices = np.asarray(block.vertices, dtype=np.int64)
        valid = np.asarray(block.vertex_valid, dtype=np.bool_)
        loops.append(np.where(valid, vertices, -1))
    return _padded(loops)


def _surface_embedding(
    state: _EmbeddingState,
    mesh: CellMesh,
    points: np.ndarray,
    cell_ids: np.ndarray,
    limits: MeshCertificateLimits,
    /,
) -> None:
    loops = _cell_loops(mesh)
    triangles, owners, nonplanar, undecided = _loop_triangles(
        points, loops, limits.maximum_candidate_pairs
    )
    if np.any(nonplanar):
        state.add("nonplanar_cell", "unresolved", "cell", cell_ids[nonplanar])
    if np.any(undecided):
        state.add(
            "undecided_cell_triangulation", "unresolved", "cell", cell_ids[undecided]
        )
    _pairwise_contacts(
        state,
        points,
        triangles,
        cell_ids[owners],
        "cell",
        "cell_contact",
        limits,
    )


@dataclass(frozen=True)
class _Boundary:
    """Oriented boundary facets, their affine pieces and their entity ids."""

    rows: np.ndarray
    entities: np.ndarray
    pieces: np.ndarray
    piece_owners: np.ndarray


def _boundary_pieces(
    state: _EmbeddingState,
    mesh: CellMesh,
    points: np.ndarray,
    facets: _Facets,
    limits: MeshCertificateLimits,
    /,
) -> _Boundary:
    rows = facets.rows[facets.boundary]
    rows = rows[:, : int(np.max(np.sum(rows >= 0, axis=1), initial=1))]
    entities = (
        _facet_entity_ids(mesh, rows) if rows.size else np.empty((0,), dtype=np.int64)
    )
    state.boundary_facets = rows.shape[0]
    if mesh.topological_dimension == 2:
        return _Boundary(rows, entities, rows, np.arange(rows.shape[0]))
    pieces, owners, nonplanar, undecided = _loop_triangles(
        points, rows, limits.maximum_candidate_pairs
    )
    if np.any(nonplanar):
        state.add("curved_boundary_facet", "unresolved", "facet", entities[nonplanar])
    if np.any(undecided):
        state.add("undecided_boundary_facet", "unresolved", "facet", entities[undecided])
    return _Boundary(rows, entities, pieces, owners)


def _shells(boundary: _Boundary, dimension: int, /) -> tuple[np.ndarray, np.ndarray]:
    """Boundary shells linked through manifold ridges, and unbalanced shells."""

    rows = boundary.rows
    count = rows.shape[0]
    if dimension == 2:
        tails = rows[:, 0]
        heads = rows[:, 1]
        facet = np.arange(count)
    else:
        lengths = np.sum(rows >= 0, axis=1)
        column = np.arange(rows.shape[1])
        following = rows[np.arange(count)[:, None], (column + 1) % lengths[:, None]]
        slots = column[None, :] < lengths[:, None]
        tails = rows[slots]
        heads = following[slots]
        facet = np.broadcast_to(np.arange(count)[:, None], rows.shape)[slots]
    if dimension == 2:
        # Ridges are vertices: one outgoing and one incoming facet link.
        vertex = np.concatenate((tails, heads))
        direction = np.concatenate((np.ones_like(tails), -np.ones_like(heads)))
        owners = np.concatenate((facet, facet))
    else:
        vertex = np.stack((np.minimum(tails, heads), np.maximum(tails, heads)), axis=1)
        direction = np.where(tails < heads, 1, -1)
        owners = facet
    keys = vertex.reshape(vertex.shape[0], -1)
    _, ridge, ridge_counts = np.unique(
        keys, axis=0, return_inverse=True, return_counts=True
    )
    ridge = ridge.reshape(-1)
    balance = np.zeros((ridge_counts.size,), dtype=np.int64)
    np.add.at(balance, ridge, direction)
    order = np.argsort(ridge, kind="stable")
    same = ridge[order[1:]] == ridge[order[:-1]]
    manifold = (ridge_counts[ridge[order[:-1]]] == 2) & (balance[ridge[order[:-1]]] == 0)
    link = same & manifold
    graph = coo_matrix(
        (
            np.ones(np.count_nonzero(link), dtype=np.int8),
            (owners[order[:-1][link]], owners[order[1:][link]]),
        ),
        shape=(count, count),
    )
    _, labels = connected_components(graph, directed=False)
    shell_keys = np.concatenate((labels[owners][:, None], keys), axis=1)
    _, shell_ridge = np.unique(shell_keys, axis=0, return_inverse=True)
    shell_balance = np.zeros((int(np.max(shell_ridge, initial=-1)) + 1,), np.int64)
    np.add.at(shell_balance, shell_ridge.reshape(-1), direction)
    unbalanced = np.zeros((int(np.max(labels, initial=-1)) + 1,), dtype=np.bool_)
    np.logical_or.at(
        unbalanced, labels[owners], shell_balance[shell_ridge.reshape(-1)] != 0
    )
    return labels, unbalanced


def _shell_representatives(
    rows: np.ndarray, labels: np.ndarray, shell_count: int, /
) -> np.ndarray:
    """Smallest vertex of each shell that belongs to no other shell (``-1``)."""

    slots = rows >= 0
    vertex = rows[slots]
    shell = np.broadcast_to(labels[:, None], rows.shape)[slots]
    pairs = np.unique(np.stack((vertex, shell), axis=1), axis=0)
    vertices, multiplicity = np.unique(pairs[:, 0], return_counts=True)
    exclusive = np.isin(pairs[:, 0], vertices[multiplicity == 1])
    representative = np.full((shell_count,), -1, dtype=np.int64)
    chosen = pairs[exclusive]
    # Rows are sorted by vertex, so the first occurrence per shell is minimal.
    shells, first = np.unique(chosen[:, 1], return_index=True)
    representative[shells] = chosen[first, 0]
    return representative


def _shell_measures(
    integers: np.ndarray, boundary: _Boundary, labels: np.ndarray, shells: int, /
) -> list[int]:
    """Exact (scaled) signed measures of every shell."""

    pieces = integers[boundary.pieces]
    if boundary.pieces.shape[1] == 2:
        values = _det2(pieces[:, 0], pieces[:, 1])
    else:
        values = _det3(pieces[:, 0], pieces[:, 1], pieces[:, 2])
    totals = [0] * shells
    for shell, value in zip(
        labels[boundary.piece_owners].tolist(), values.tolist(), strict=True
    ):
        totals[shell] += value
    return totals


def _ray_crossings_2d(point: np.ndarray, segments: np.ndarray, /) -> tuple[int, bool]:
    """Winding of oriented segments at ``point`` from a perturbed +x ray."""

    a = segments[:, 0]
    b = segments[:, 1]
    direction = np.sign(b[:, 1] - a[:, 1]).astype(np.int16)
    upward = (a[:, 1] <= point[1]) & (point[1] < b[:, 1])
    downward = (b[:, 1] <= point[1]) & (point[1] < a[:, 1])
    inside = np.where(direction > 0, upward, downward) & (direction != 0)
    if not np.any(inside):
        return 0, True
    sigma, certain = _orient2d(
        a[inside], b[inside], np.broadcast_to(point, a[inside].shape)
    )
    hit = sigma * direction[inside] > 0
    decided = bool(np.all(certain) and np.all(sigma != 0))
    return int(np.sum(direction[inside][hit])), decided


def _ray_crossings_3d(point: np.ndarray, triangles: np.ndarray, /) -> tuple[int, bool]:
    """Winding of oriented triangles at ``point`` from a perturbed +x ray.

    The ray starts at ``point + (0, e, e**2)`` for infinitesimal ``e``; a zero
    projected edge orientation is resolved by the first nonvanishing derivative,
    ``sign(u_z - v_z)`` then ``sign(v_y - u_y)``.
    """

    projected = triangles[:, :, 1:]
    target = np.broadcast_to(point[1:], projected[:, 0].shape)
    orientation, certain = _orient2d(projected[:, 0], projected[:, 1], projected[:, 2])
    decided = bool(np.all(certain))
    inside = orientation != 0
    for start, stop in ((0, 1), (1, 2), (2, 0)):
        u = projected[:, start]
        v = projected[:, stop]
        sign, known = _orient2d(u, v, target)
        perturbed = np.where(
            sign != 0,
            sign,
            np.where(
                u[:, 1] != v[:, 1],
                np.sign(u[:, 1] - v[:, 1]),
                np.sign(v[:, 0] - u[:, 0]),
            ),
        )
        inside &= perturbed == orientation
        decided &= bool(np.all(known))
    if not np.any(inside):
        return 0, decided
    selected = triangles[inside]
    sigma, known = _orient3d(
        selected[:, 0],
        selected[:, 1],
        selected[:, 2],
        np.broadcast_to(point, selected[:, 0].shape),
    )
    hit = sigma * orientation[inside] < 0
    decided &= bool(np.all(known) and np.all(sigma != 0))
    return int(np.sum(orientation[inside][hit])), decided


def _exterior_degrees(
    state: _EmbeddingState,
    points: np.ndarray,
    boundary: _Boundary,
    dimension: int,
    limits: MeshCertificateLimits,
    /,
) -> None:
    state.checks.append("exterior_degree")
    labels, unbalanced = _shells(boundary, dimension)
    shells = unbalanced.size
    state.shells = shells
    if np.any(unbalanced):
        state.add(
            "nonmanifold_boundary_ridge",
            "unresolved",
            "facet",
            boundary.entities[unbalanced[labels]],
        )
        return
    representative = _shell_representatives(boundary.rows, labels, shells)
    exact = points.dtype == object
    pieces = boundary.pieces.shape[0]
    if exact:
        from ._exact_polyhedral_geometry import coordinate_integers

        # One positive common denominator preserves every orientation, measure
        # sign and the perturbed ray: exact source shells use integer scaling.
        # Per piece the shell measure is one 14-visit integer determinant
        # below 2**(3 b + 3); boxes take twelve comparisons and fifteen slots;
        # a ray test takes five box comparisons per piece and, per crossed
        # candidate, fifteen perturbation visits with six differences below
        # 2**(b + 1) and forty slots. Orientation signs charge themselves.
        integers, _ = coordinate_integers(points)
        bits = exact_bits(integers)
        coordinates = integers
        exact_reserve(
            12 * pieces,
            8 * 15 * pieces + shells * bigint_bytes(3 * bits + 3 + pieces.bit_length()),
        )
    else:
        integers, _ = _dyadic_integers(points)
        coordinates = points
        bits = 0
    with _exact_step(exact, 14 * pieces, 14 * pieces * bigint_bytes(3 * bits + 3)):
        measures = _shell_measures(integers, boundary, labels, shells)
    piece_shell = labels[boundary.piece_owners]
    corners = coordinates[boundary.pieces]
    lower = np.min(corners, axis=1)
    upper = np.max(corners, axis=1)
    if shells * boundary.pieces.shape[0] > limits.maximum_ray_tests:
        state.add("exterior_degree_capacity", "unresolved", "mesh")
        return
    # One exact ray query per shell; the shell count is bounded by the ray budget.
    for shell in range(shells):
        facets = boundary.entities[labels == shell]
        if representative[shell] < 0 or measures[shell] == 0:
            state.add("degenerate_shell", "unresolved", "facet", facets)
            continue
        origin = coordinates[representative[shell]]
        # Closed transverse boxes keep every piece the perturbed ray can cross.
        with _exact_step(exact, 5 * pieces):
            others = corners[
                (piece_shell != shell)
                & np.all(lower[:, 1:] <= origin[1:], axis=1)
                & np.all(upper[:, 1:] >= origin[1:], axis=1)
                & (upper[:, 0] >= origin[0])
            ]
        state.ray_tests += boundary.pieces.shape[0]
        rows = others.shape[0]
        with _exact_step(exact, 15 * rows, rows * (6 * bigint_bytes(bits + 1) + 8 * 40)):
            if dimension == 3:
                winding, decided = _ray_crossings_3d(origin, others)
            else:
                winding, decided = _ray_crossings_2d(origin, others)
        if not decided:
            state.add("exterior_degree_predicates", "unresolved", "facet", facets)
            continue
        degree = winding + (0 if measures[shell] > 0 else -1)
        if degree != 0:
            state.add("nested_component", "violated", "facet", facets)


def _volume_embedding(
    state: _EmbeddingState,
    mesh: CellMesh,
    points: np.ndarray,
    facets: _Facets,
    limits: MeshCertificateLimits,
    /,
) -> None:
    boundary = _boundary_pieces(state, mesh, points, facets, limits)
    _pairwise_contacts(
        state,
        points,
        boundary.pieces,
        boundary.entities[boundary.piece_owners],
        "facet",
        "boundary_contact",
        limits,
    )
    if not state.clean:
        # Degree theory needs every other premise; report the missing decision.
        state.add("exterior_degree_premise", "unresolved", "mesh")
        return
    _exterior_degrees(state, points, boundary, mesh.topological_dimension, limits)


def _exact_source_predicate_view(points: np.ndarray) -> tuple[np.ndarray, bool]:
    """Export binary64 only when it represents every authenticated source value.

    The canonical binary predicates use exact adaptive/dyadic signs. This is an
    equality-certified execution view of the source bank, not its mesh carrier.
    A nonrepresentable value keeps the original rational predicate path.
    """
    from ..discretization._coordinate_enclosure import _COORDINATE_BUDGET
    from ..meshing._quad_generation import _family_host_array

    ledger = _COORDINATE_BUDGET.get()
    for value in points.flat:
        if ledger is not None:
            ledger.reserve(2)
        try:
            numeric = float(value)
        except OverflowError:
            return points, False
        if not math.isfinite(numeric) or Fraction(numeric) != value:
            return points, False
    if ledger is not None:
        ledger.reserve(points.size, 128 + points.size * np.dtype(np.float64).itemsize)
    exported = _family_host_array(points.shape, np.float64)
    for index, value in enumerate(points.flat):
        exported.flat[index] = float(value)
    return exported, True


def _exact_source_embedding(
    state: _EmbeddingState,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    source: ExactCellGeometrySource,
    cell_ids: np.ndarray,
    junctions: np.ndarray,
    limits: MeshCertificateLimits,
    scope: CoordinateMapScope,
    budget: CoordinateEnclosureBudget,
    exact_points: CoordinateSourceBank,
    /,
) -> None:
    """Embed an exact coordinate source on its authoritative points.

    The binary64 carrier is only the correctly rounded image of the source;
    contacts, shell measures and exterior degrees are decided on the source.
    The PLC constructor admits only direct tetrahedral carriers in three
    dimensions, so the source runs the volume proof.
    """

    from ..discretization._cell_geometry import (
        BarycentricCellGeometryElement,
        PolynomialComposedCellGeometryElement,
        RationalComposedCellGeometryElement,
    )
    from ..discretization._exact_plc_geometry import (
        ExactPlcCellGeometryConvexSource,
        ExactPlcCellGeometrySource,
    )
    from ..discretization._exact_power_geometry import (
        ExactPowerCellGeometryLinearActionSource,
        ExactPowerCellGeometryRestrictionSource,
        ExactPowerCellGeometrySource,
    )
    from ._exact_polyhedral_geometry import coordinate_integer_profile

    facets = _mesh_facets(mesh)
    _pairing_findings(state, mesh, facets, junctions)
    match source:
        case ExactPlcCellGeometryConvexSource() | ExactPlcCellGeometrySource() if (
            scope == "affine"
        ):
            state.checks.append("exact_plc_source_coordinates")
            # Scope preparation proved every complete map affine and retained
            # the exact target vertex bank, so no cell expression is rebuilt.
            points = np.asarray(exact_points, dtype=object)
            if coordinate_integer_profile(points, source.maximum_bits) is None:
                state.add("exact_source_bit_budget", "unresolved", "mesh")
            else:
                predicate_points, binary = _exact_source_predicate_view(points)
                if binary:
                    state.checks.append("exact_source_binary64_value_equivalence")
                _volume_embedding(state, mesh, predicate_points, facets, limits)
            state.source_expression_work_units = budget.work_units
            state.source_expression_peak_bytes = budget.peak_bytes_upper
        case ExactPlcCellGeometryConvexSource():
            from ._mapped_embedding import certify_mapped_embedding

            certify_mapped_embedding(state, mesh, geometry, cell_ids, limits)
            state.source_expression_work_units = budget.work_units
            state.source_expression_peak_bytes = budget.peak_bytes_upper
        case (
            ExactPowerCellGeometryLinearActionSource()
            | ExactPowerCellGeometryRestrictionSource()
            | ExactPlcCellGeometrySource()
        ) if all(block.cell_kind == "hexahedron" for block in mesh.blocks) or (
            isinstance(source, ExactPlcCellGeometrySource)
            and any(
                isinstance(
                    element,
                    (
                        BarycentricCellGeometryElement,
                        PolynomialComposedCellGeometryElement,
                        RationalComposedCellGeometryElement,
                    ),
                )
                for element in geometry.elements
            )
        ):
            from ._mapped_embedding import certify_mapped_embedding

            certify_mapped_embedding(state, mesh, geometry, cell_ids, limits)
            state.source_expression_work_units = budget.work_units
            state.source_expression_peak_bytes = budget.peak_bytes_upper
        case (
            ExactPowerCellGeometrySource()
            | ExactPowerCellGeometryRestrictionSource()
            | ExactPowerCellGeometryLinearActionSource()
        ):
            from ._exact_power_certificates import certify_exact_power_embedding

            certify_exact_power_embedding(state, mesh, geometry, facets, cell_ids, limits)
        case ExactPlcCellGeometrySource():
            state.checks.append("exact_plc_source_coordinates")
            # The request's already-active ledger owns source preparation and
            # every exact predicate; no stage renews its allowance.
            points = np.asarray(exact_points, dtype=object)
            if coordinate_integer_profile(points, source.maximum_bits) is None:
                state.add("exact_source_bit_budget", "unresolved", "mesh")
            else:
                predicate_points, binary = _exact_source_predicate_view(points)
                if binary:
                    state.checks.append("exact_source_binary64_value_equivalence")
                _volume_embedding(state, mesh, predicate_points, facets, limits)
            state.source_expression_work_units = budget.work_units
            state.source_expression_peak_bytes = budget.peak_bytes_upper
        case invalid:
            assert_never(invalid)


def _junction_vertices(mesh: CellMesh, values: ArrayLike | None, /) -> np.ndarray:
    """Sorted declared junction vertex global ids of an interval mesh."""

    if values is None:
        return np.zeros((0,), dtype=np.int64)
    if mesh.topological_dimension != 1:
        raise ValueError("junction_vertices are declared for interval meshes only.")
    junctions = np.unique(np.asarray(values, dtype=np.int64).reshape(-1))
    if np.setdiff1d(junctions, np.asarray(mesh.vertex_global_ids, dtype=np.int64)).size:
        raise ValueError("junction_vertices must be vertex global ids of the mesh.")
    return junctions


def certify_global_embedding(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    validity: CellValidityCertificate,
    /,
    *,
    limits: MeshCertificateLimits | None = None,
    junction_vertices: ArrayLike | None = None,
) -> GlobalEmbeddingCertificate:
    """Certify that the mesh is globally embedded (see the module docstring).

    ``validity`` must certify the same coordinate arrays and topology; it is
    the positive-orientation premise, not a substitute for this certificate.
    ``junction_vertices`` declares the vertex global ids at which an interval
    mesh may join more or fewer than two intervals (curve-network junctions);
    interval contacts beyond shared vertices are still refused there.
    """

    if not isinstance(mesh, CellMesh):
        raise TypeError("mesh must be CellMesh.")
    if not isinstance(geometry, CellGeometrySpec):
        raise TypeError("geometry must be CellGeometrySpec.")
    if not isinstance(validity, CellValidityCertificate):
        raise TypeError("validity must be CellValidityCertificate.")
    validity.require_bound(geometry, mesh=mesh)
    limits_ = MeshCertificateLimits() if limits is None else limits
    if not isinstance(limits_, MeshCertificateLimits):
        raise TypeError("limits must be MeshCertificateLimits or None.")
    junctions = _junction_vertices(mesh, junction_vertices)
    budget = coordinate_enclosure_budget(
        limits_.maximum_work_units, limits_.maximum_scratch_bytes
    )
    with ExitStack() as preparation:
        scope_resource: CoordinateEnclosureResourceError | None = None
        try:
            preparation.enter_context(budget.activate())
            preparation.enter_context(
                budget.bound_stage(
                    limits_.maximum_work_units, limits_.maximum_scratch_bytes
                )
            )
            scope, _, prepared_source = _coordinate_scope(mesh, geometry, budget)
        except CoordinateEnclosureResourceError as error:
            scope = "mapped"
            prepared_source = ()
            scope_resource = error
        binding = MeshCertificateBinding(
            mesh, geometry, scope, limits_, junction_vertices=tuple(junctions.tolist())
        )
        points = np.asarray(mesh.coordinates, dtype=np.float64)
        cell_ids = _cell_global_ids(mesh)
        state = _EmbeddingState([], [])
        _validity_findings(state, validity, cell_ids)
        dimensions = (mesh.topological_dimension, mesh.ambient_dimension)
        if scope_resource is not None:
            finding = (
                "exact_source_resource_budget"
                if geometry.exact_source is not None
                else "mapped_source_expression_resource_budget"
            )
            state.add(
                finding,
                "unresolved",
                "mesh",
                resource_error=scope_resource,
                expression_budget=budget,
            )
        elif not validity.all_certified:
            state.add(
                "cell_validity_embedding_premise",
                "unresolved",
                "cell",
                cell_ids[
                    np.asarray(validity.status) != CellValidityStatus.CERTIFIED_VALID
                ],
            )
        elif mesh.storage is not None:
            state.add("owner_local_global_embedding_premise", "unresolved", "mesh")
        elif mesh.periodic_topology is not None:
            from ._periodic_embedding import certify_periodic_mapped_embedding

            state.periodic_images = certify_periodic_mapped_embedding(
                state, mesh, geometry, cell_ids, limits_
            )
        elif geometry.exact_source is not None:
            try:
                with budget.activate():
                    _exact_source_embedding(
                        state,
                        mesh,
                        geometry,
                        geometry.exact_source,
                        cell_ids,
                        junctions,
                        limits_,
                        scope,
                        budget,
                        prepared_source,
                    )
            except CoordinateEnclosureResourceError as error:
                state.add(
                    "exact_source_resource_budget",
                    "unresolved",
                    "mesh",
                    resource_error=error,
                    expression_budget=budget,
                )
        elif scope == "mapped":
            from ._mapped_embedding import certify_mapped_embedding

            try:
                facets = _mesh_facets(mesh, geometry=geometry)
                _pairing_findings(state, mesh, facets, junctions)
                with budget.activate():
                    certify_mapped_embedding(state, mesh, geometry, cell_ids, limits_)
            except CoordinateEnclosureResourceError as error:
                state.add(
                    "mapped_source_expression_resource_budget",
                    "unresolved",
                    "mesh",
                    resource_error=error,
                    expression_budget=budget,
                )
        else:
            try:
                facets = _mesh_facets(mesh, geometry=geometry)
                _pairing_findings(state, mesh, facets, junctions)
                match dimensions:
                    case (1, 1):
                        _interval_embedding(state, mesh, cell_ids)
                    case (1, 2) | (1, 3):
                        items = np.concatenate(
                            [
                                np.asarray(block.vertices, dtype=np.int64)
                                for block in mesh.blocks
                            ]
                        )
                        _pairwise_contacts(
                            state,
                            points,
                            items,
                            cell_ids,
                            "cell",
                            "cell_contact",
                            limits_,
                        )
                    case (2, 3):
                        _surface_embedding(state, mesh, points, cell_ids, limits_)
                    case (2, 2) | (3, 3):
                        _volume_embedding(state, mesh, points, facets, limits_)
                    case _:
                        state.add("unsupported_embedding_dimension", "unresolved", "mesh")
            except CoordinateEnclosureResourceError as error:
                state.add(
                    "mapped_source_expression_resource_budget",
                    "unresolved",
                    "mesh",
                    resource_error=error,
                    expression_budget=budget,
                )
        state.source_expression_work_units = budget.work_units
        state.source_expression_peak_bytes = budget.peak_bytes_upper
        return GlobalEmbeddingCertificate(
            binding,
            validity.certificate_id,
            tuple(state.findings),
            tuple(state.checks),
            cell_count=cell_ids.size,
            boundary_facet_count=state.boundary_facets,
            shell_count=state.shells,
            candidate_pair_count=state.candidate_pairs,
            ray_test_count=state.ray_tests,
            subdivision_piece_count=state.subdivision_pieces,
            periodic_image_count=state.periodic_images,
            source_expression_work_units=state.source_expression_work_units,
            source_expression_peak_bytes=state.source_expression_peak_bytes,
            boundary_degree=state.boundary_degree,
        )


# Domain and interface coverage ------------------------------------------------------


@dataclass(frozen=True)
class _CoveringPieces:
    """Affine pieces of mesh boundary/interface facets with expected sides."""

    pieces: np.ndarray
    facet_entities: np.ndarray
    own_region: np.ndarray
    other_region: np.ndarray


def _covering_pieces(
    state: _EmbeddingState,
    mesh: CellMesh,
    points: np.ndarray,
    facets: _Facets,
    regions: np.ndarray,
    limits: MeshCertificateLimits,
    /,
) -> _CoveringPieces:
    order = np.argsort(facets.group, kind="stable")
    first_occurrence = np.ones((order.size,), dtype=np.bool_)
    first_occurrence[1:] = facets.group[order[1:]] != facets.group[order[:-1]]
    partner = np.full((order.size,), -1, dtype=np.int64)
    same = ~first_occurrence[1:]
    partner[order[:-1][same]] = order[1:][same]
    own = regions[facets.cells]
    other = np.where(partner >= 0, regions[facets.cells[np.maximum(partner, 0)]], -1)
    selected = np.zeros((order.size,), dtype=np.bool_)
    selected[order[first_occurrence]] = True
    covering = selected & (facets.counts[facets.group] <= 2) & (own != other)
    rows = facets.rows[covering]
    rows = rows[:, : int(np.max(np.sum(rows >= 0, axis=1), initial=1))]
    entities = (
        _facet_entity_ids(mesh, rows) if rows.size else np.empty((0,), dtype=np.int64)
    )
    if mesh.topological_dimension == 2:
        owners = np.arange(rows.shape[0])
        pieces = rows
    else:
        pieces, owners, nonplanar, undecided = _loop_triangles(
            points, rows, limits.maximum_candidate_pairs
        )
        if np.any(nonplanar | undecided):
            state.add(
                "nonplanar_covering_facet",
                "unresolved",
                "facet",
                entities[nonplanar | undecided],
            )
    return _CoveringPieces(
        pieces, entities[owners], own[covering][owners], other[covering][owners]
    )


def _projection_axes(corners: np.ndarray, /) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Coordinate pair on which each declared facet projects injectively.

    Returns ``(axes, orientation, certain)``; the orientation of the projected
    declared facet is exact and nonzero where ``certain``.
    """

    count = corners.shape[0]
    choices = np.asarray(((1, 2), (0, 2), (0, 1)))
    if corners.shape[1] == 2:
        axis = np.where(corners[:, 0, 0] != corners[:, 1, 0], 0, 1)
        sign = np.sign(
            corners[np.arange(count), 1, axis] - corners[np.arange(count), 0, axis]
        )
        return axis[:, None], sign.astype(np.int16), np.ones((count,), dtype=np.bool_)
    normal = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    preference = np.argsort(-np.abs(normal), axis=1, kind="stable")
    axes = np.zeros((count, 2), dtype=np.int64)
    orientation = np.zeros((count,), dtype=np.int16)
    certain = np.zeros((count,), dtype=np.bool_)
    for rank in range(3):
        pending = orientation == 0
        candidate = choices[preference[:, rank]]
        projected = np.take_along_axis(corners, candidate[:, None, :], axis=2)
        sign, known = _orient2d(projected[:, 0], projected[:, 1], projected[:, 2])
        take = pending & known & (sign != 0)
        axes[take] = candidate[take]
        orientation[take] = sign[take]
        certain |= take
    return axes, orientation, certain


def _projected_measures(
    integers: np.ndarray, rows: np.ndarray, axes: np.ndarray, /
) -> np.ndarray:
    """Exact absolute projected measures (scaled integers) of simplices."""

    corners = integers[rows]
    index = np.arange(rows.shape[0])
    if rows.shape[1] == 2:
        axis = axes[:, 0]
        values = corners[index, 1, axis] - corners[index, 0, axis]
    else:
        first = np.stack(
            [
                corners[index, 1, axes[:, k]] - corners[index, 0, axes[:, k]]
                for k in range(2)
            ],
            axis=1,
        )
        second = np.stack(
            [
                corners[index, 2, axes[:, k]] - corners[index, 0, axes[:, k]]
                for k in range(2)
            ],
            axis=1,
        )
        values = _det2(first, second)
    return np.asarray([abs(value) for value in values.tolist()], dtype=object)


def _partition_covering_pieces(
    state: _EmbeddingState,
    covering: _CoveringPieces,
    points: np.ndarray,
    domain: PiecewiseLinearDomain,
    axes: np.ndarray,
    limits: MeshCertificateLimits,
) -> tuple[list[Fraction], np.ndarray, bool, dict[_CoverageOverlapKey, Fraction]]:
    """Partition actual affine facets across the independent source triangulation."""
    from ._planar_coverage import (
        intersection,
        plane_key,
        project,
        rational_points,
        signed_measure,
        source_groups,
    )

    groups, overlaps, used, exceeded = source_groups(
        domain.vertices,
        domain.facets,
        domain.facet_regions,
        limits.maximum_candidate_pairs - state.candidate_pairs,
    )
    state.candidate_pairs += used
    covered = [Fraction(0) for _ in range(domain.facets.shape[0])]
    unmatched = np.zeros(covering.pieces.shape[0], dtype=np.bool_)
    relations: dict[_CoverageOverlapKey, Fraction] = {}
    for first, second in overlaps:
        state.add(
            "overlapping_source_facets", "violated", "source_facet", (first, second)
        )
    if exceeded:
        state.add("coverage_candidate_capacity", "unresolved", "mesh")
        return covered, unmatched, True, relations
    lookup = {(group.plane, group.regions): group for group in groups}
    compatible: dict[tuple[tuple[Fraction, ...], tuple[int, int]], list[int]] = {}
    polygons: dict[int, tuple[tuple[Fraction, ...], ...]] = {}
    whole: dict[int, Fraction] = {}
    accumulated = [Fraction(0) for _ in range(covering.pieces.shape[0])]
    for piece, row in enumerate(covering.pieces):
        polygon = rational_points(points[row])
        key = plane_key(polygon)
        if key is None:
            unmatched[piece] = True
            continue
        plane, _, _, sign = key
        regions = (int(covering.own_region[piece]), int(covering.other_region[piece]))
        canonical_regions = regions if sign > 0 else regions[::-1]
        pair = (plane, canonical_regions)
        group = lookup.get(pair)
        if group is None:
            unmatched[piece] = True
            continue
        polygons[piece] = project(polygon, group.axes)
        whole[piece] = abs(signed_measure(polygons[piece]))
        compatible.setdefault(pair, []).append(piece)
    for pair, pieces in compatible.items():
        group = lookup[pair]
        chosen = np.asarray(pieces, dtype=np.int64)
        source_rows = np.asarray(group.members, dtype=np.int64)
        firsts, seconds, exceeded = _candidate_pairs(
            *_boxes(points, covering.pieces[chosen]),
            limits.maximum_candidate_pairs - state.candidate_pairs,
            other=_boxes(domain.vertices, domain.facets[source_rows]),
        )
        state.candidate_pairs += firsts.size
        for first, local in zip(firsts.tolist(), seconds.tolist(), strict=True):
            piece = pieces[first]
            measure = intersection(polygons[piece], group.simplices[local])
            if measure == 0:
                continue
            accumulated[piece] += measure
            target = group.members[local]
            target_axes = tuple(int(axis) for axis in axes[target])
            target_measure = abs(
                signed_measure(
                    project(
                        rational_points(domain.vertices[domain.facets[target]]),
                        target_axes,
                    )
                )
            )
            actual = measure * target_measure / group.measures[local]
            covered[target] += actual
            relation: _CoverageOverlapKey = (
                int(covering.facet_entities[piece]),
                target,
                target_axes,
                "exact",
                "physical",
            )
            relations[relation] = relations.get(relation, Fraction(0)) + actual
        if exceeded:
            state.add("coverage_candidate_capacity", "unresolved", "mesh")
            return covered, unmatched, True, relations
        for piece in pieces:
            unmatched[piece] = accumulated[piece] != whole[piece]
    return covered, unmatched, False, relations


def _region_measures(
    integers: np.ndarray,
    rows: np.ndarray,
    weights: np.ndarray,
    groups: np.ndarray,
    count: int,
    /,
) -> list[int]:
    """Exact divergence-theorem measures ``sum weight * det`` per group."""

    corners = integers[rows]
    if rows.shape[1] == 2:
        values = _det2(corners[:, 0], corners[:, 1])
    else:
        values = _det3(corners[:, 0], corners[:, 1], corners[:, 2])
    totals = [0] * count
    for group, weight, value in zip(
        groups.tolist(), weights.tolist(), values.tolist(), strict=True
    ):
        if group >= 0:
            totals[group] += weight * value
    return totals


def _cell_measure_pieces(
    state: _EmbeddingState,
    mesh: CellMesh,
    points: np.ndarray,
    facets: _Facets,
    limits: MeshCertificateLimits,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Affine pieces of every outward facet occurrence and their cells."""

    if mesh.topological_dimension == 2:
        return facets.rows[:, :2], facets.cells
    pieces, owners, nonplanar, undecided = _loop_triangles(
        points, facets.rows, limits.maximum_candidate_pairs
    )
    if np.any(nonplanar | undecided):
        state.add(
            "nonplanar_cell_facet",
            "unresolved",
            "facet",
            _facet_entity_ids(mesh, facets.rows[nonplanar | undecided]),
        )
    return pieces, facets.cells[owners]


def _active_expression_work() -> tuple[int, int, int] | None:
    """Actual work, peak, and admission requirement of the active ledger."""
    from ..discretization._coordinate_enclosure import _COORDINATE_BUDGET

    budget = _COORDINATE_BUDGET.get()
    return (
        None
        if budget is None
        else (
            budget.work_units,
            budget.peak_bytes_upper,
            budget.required_work_units,
        )
    )


def _coverage_result(
    binding: MeshCertificateBinding,
    embedding: GlobalEmbeddingCertificate,
    domain: PiecewiseLinearDomain | MappedReferenceDomain,
    state: _EmbeddingState,
    requested: tuple[Fraction, ...],
    achieved: tuple[Fraction | None, ...],
    covered: int,
    overlaps: dict[_CoverageOverlapKey, Fraction],
    premises: tuple[str, ...],
    /,
    *,
    expression_work: tuple[int, int, int] | None,
) -> DomainCoverageCertificate:
    from ._mapped_coverage import enclosure

    mismatched = tuple(
        index
        for index, (actual, target) in enumerate(zip(achieved, requested, strict=True))
        if actual is not None and actual != target
    )
    if mismatched:
        state.add("region_measure", "violated", "source_region", mismatched)
    return DomainCoverageCertificate(
        binding,
        embedding.certificate_id,
        domain,
        tuple(state.findings),
        requested_region_measures=tuple(float(value) for value in requested),
        achieved_region_measures=tuple(
            None if value is None else float(value) for value in achieved
        ),
        requested_region_measure_bounds=tuple(enclosure(value) for value in requested),
        achieved_region_measure_bounds=tuple(
            None if value is None else enclosure(value) for value in achieved
        ),
        integration_error_bounds=tuple(
            None if value is None else 0.0 for value in achieved
        ),
        covered_source_facet_count=covered,
        facet_source_overlaps=tuple(
            (
                facet,
                source,
                axes,
                0.0 if semantics == "candidate" else enclosure(value)[0],
                enclosure(value)[1],
                semantics,
                space,
            )
            for (facet, source, axes, semantics, space), value in sorted(overlaps.items())
        ),
        premise_certificate_ids=premises,
        candidate_pair_count=state.candidate_pairs,
        subdivision_piece_count=state.subdivision_pieces,
        maximum_subdivision_depth_reached=state.subdivision_depth,
        source_expression_work_units=None
        if expression_work is None
        else expression_work[0],
        source_expression_required_work_units=None
        if expression_work is None
        else expression_work[2],
        source_expression_peak_bytes=None
        if expression_work is None
        else expression_work[1],
    )


def _box_jacobian_determinant(
    coordinates: tuple[Polynomial, ...], /
) -> tuple[dict[tuple[int, ...], int], int]:
    """Exact Jacobian determinant of a 2D/3D box map as integers over one denominator.

    The same exact polynomial as ``determinant(jacobian)``; every coefficient
    visit and product term is charged to the active coordinate ledger.
    """
    from ..discretization._coordinate_enclosure import (
        _COORDINATE_BUDGET,
        _reserve_polynomial,
    )

    budget = _COORDINATE_BUDGET.get()
    dimension = len(coordinates)
    if budget is not None:
        budget.reserve(
            sum(len(polynomial) for polynomial in coordinates) * (dimension + 1)
        )
    common = math.lcm(
        *(
            value.denominator
            for polynomial in coordinates
            for value in polynomial.values()
        )
    )
    columns: list[list[dict[tuple[int, ...], int]]] = []
    for polynomial in coordinates:
        row = []
        for axis in range(dimension):
            derivative: dict[tuple[int, ...], int] = {}
            for index, value in polynomial.items():
                if index[axis]:
                    lowered = (*index[:axis], index[axis] - 1, *index[axis + 1 :])
                    derivative[lowered] = derivative.get(lowered, 0) + index[
                        axis
                    ] * value.numerator * (common // value.denominator)
            row.append(derivative)
        columns.append(row)

    def bits(values: dict[tuple[int, ...], int]) -> int:
        return max((abs(value).bit_length() for value in values.values()), default=1)

    def product(
        first: dict[tuple[int, ...], int], second: dict[tuple[int, ...], int]
    ) -> dict[tuple[int, ...], int]:
        _reserve_polynomial(
            len(first) * len(second),
            len(first) * len(second),
            dimension,
            bits(first) + bits(second) + max(len(first) * len(second), 1).bit_length(),
        )
        result: dict[tuple[int, ...], int] = {}
        for a, x in first.items():
            for b, y in second.items():
                index = tuple(i + j for i, j in zip(a, b, strict=True))
                result[index] = result.get(index, 0) + x * y
        return result

    def combine(
        terms: tuple[tuple[int, dict[tuple[int, ...], int]], ...],
    ) -> dict[tuple[int, ...], int]:
        count = sum(len(term) for _, term in terms)
        _reserve_polynomial(
            count,
            count,
            dimension,
            max((bits(term) for _, term in terms), default=1) + len(terms).bit_length(),
        )
        result: dict[tuple[int, ...], int] = {}
        for sign, term in terms:
            for index, value in term.items():
                result[index] = result.get(index, 0) + sign * value
        return result

    j = columns
    if dimension == 2:
        determinant = combine(
            ((1, product(j[0][0], j[1][1])), (-1, product(j[0][1], j[1][0])))
        )
    else:
        minors = (
            combine(((1, product(j[1][1], j[2][2])), (-1, product(j[1][2], j[2][1])))),
            combine(((1, product(j[1][2], j[2][0])), (-1, product(j[1][0], j[2][2])))),
            combine(((1, product(j[1][0], j[2][1])), (-1, product(j[1][1], j[2][0])))),
        )
        determinant = combine(
            tuple((1, product(j[0][axis], minor)) for axis, minor in enumerate(minors))
        )
    return {
        index: value for index, value in determinant.items() if value
    }, common**dimension


def _box_jacobian_measure(coordinates: tuple[Polynomial, ...], /) -> Fraction:
    """Exact integral over the unit box of the Jacobian determinant of a 2D/3D map."""
    from ..discretization._coordinate_enclosure import _COORDINATE_BUDGET

    determinant, denominator = _box_jacobian_determinant(coordinates)
    budget = _COORDINATE_BUDGET.get()
    if budget is not None:
        budget.reserve(2 * len(determinant))
    weights = {index: math.prod(i + 1 for i in index) for index in determinant}
    scale_ = math.lcm(1, *weights.values())
    total = sum(
        value * (scale_ // weights[index]) for index, value in determinant.items()
    )
    return Fraction(total, scale_ * denominator)


def _axis_aligned_root_chart(
    element: CellGeometryElement,
    kind: str,
    dimension: int,
    /,
) -> tuple[CellGeometryElement, tuple[Fraction, ...], tuple[Fraction, ...]] | None:
    """Direct box root, origin and signed steps of an axis-aligned affine chart chain.

    The chain is the authoritative reference composition of the target element;
    any other chart, root kind or rational source returns ``None``.
    """
    from ..discretization._cell_geometry import (
        BarycentricCellGeometryElement,
        PolynomialComposedCellGeometryElement,
        RationalComposedCellGeometryElement,
        RestrictedCellGeometryElement,
        SplineCellGeometryElement,
    )
    from ..discretization._coordinate_enclosure import coordinate_reference_chain

    if kind not in ("quadrilateral", "hexahedron") or dimension not in (2, 3):
        return None
    source = element
    while isinstance(
        source,
        (
            RestrictedCellGeometryElement,
            PolynomialComposedCellGeometryElement,
            RationalComposedCellGeometryElement,
        ),
    ):
        source = source.source_element
    if isinstance(source, BarycentricCellGeometryElement):
        return None
    root, reference = coordinate_reference_chain(element)
    if (
        isinstance(
            root,
            (
                RestrictedCellGeometryElement,
                PolynomialComposedCellGeometryElement,
                RationalComposedCellGeometryElement,
                SplineCellGeometryElement,
            ),
        )
        or root.cell_kind != kind
        or len(reference) != dimension
        or any(not isinstance(value, dict) for value in reference)
    ):
        return None
    polynomial_reference = tuple(value for value in reference if isinstance(value, dict))
    origin, steps = [], []
    for axis, value in enumerate(polynomial_reference):
        unit = tuple(int(axis == other) for other in range(dimension))
        if any(index not in ((0,) * dimension, unit) for index in value) or not value.get(
            unit
        ):
            return None
        origin.append(value.get((0,) * dimension, Fraction(0)))
        steps.append(value[unit])
    return root, tuple(origin), tuple(steps)


def _affine_jacobian_determinant(
    polynomials: tuple[Polynomial, ...], dimension: int, /
) -> Fraction | None:
    """Constant Jacobian determinant of an exactly affine map, else ``None``.

    Only total-degree-one maps qualify; their Jacobian and determinant are the
    canonical exact coordinate-expression algebra, charged to the active ledger.
    """
    from ..discretization import _coordinate_enclosure as algebra

    zero = (0,) * dimension
    if len(polynomials) != dimension or any(
        sum(index) > 1 for value in polynomials for index in value
    ):
        return None
    jacobian = tuple(
        tuple(algebra.derivative(value, axis) for axis in range(dimension))
        for value in polynomials
    )
    return algebra.determinant(jacobian).get(zero, Fraction(0))


def _affine_root_chart(
    element: CellGeometryElement,
    kind: str,
    dimension: int,
    /,
) -> tuple[CellGeometryElement, Fraction] | None:
    """Direct root and exact chart determinant of an affine reference chart chain.

    An affine restriction of an affine root is affine; its measure is the
    product of both constant Jacobian determinants and the reference measure.
    Pyramid charts, composed, spline or nonaffine chains return ``None``.
    """
    from ..discretization._cell_geometry import (
        BarycentricCellGeometryElement,
        PolynomialComposedCellGeometryElement,
        RationalComposedCellGeometryElement,
        RestrictedCellGeometryElement,
        SplineCellGeometryElement,
    )
    from ..discretization._coordinate_enclosure import coordinate_reference_chain

    if kind == "pyramid" or dimension not in (2, 3):
        return None
    source = element
    while isinstance(
        source,
        (
            RestrictedCellGeometryElement,
            PolynomialComposedCellGeometryElement,
            RationalComposedCellGeometryElement,
        ),
    ):
        source = source.source_element
    if isinstance(source, BarycentricCellGeometryElement):
        return None
    root, reference = coordinate_reference_chain(element)
    if (
        isinstance(
            root,
            (
                RestrictedCellGeometryElement,
                PolynomialComposedCellGeometryElement,
                RationalComposedCellGeometryElement,
                SplineCellGeometryElement,
            ),
        )
        or root.cell_kind == "pyramid"
        or len(reference) != dimension
        or any(not isinstance(value, dict) for value in reference)
    ):
        return None
    polynomial_reference = tuple(value for value in reference if isinstance(value, dict))
    determinant = _affine_jacobian_determinant(polynomial_reference, dimension)
    return None if determinant is None else (root, determinant)


def _signed_boundary_region_measures(
    state: _EmbeddingState,
    regions: np.ndarray,
    cells: list[tuple[str, CellGeometryElement, CoordinateCoefficients, tuple[int, ...]]],
    maps: Sequence[tuple[Expression, ...] | None],
    totals: list[Fraction | None],
    rational_regions: set[int],
    facets: _Facets,
) -> None:
    """Integrate the oriented source chain under its earned exact trace premise."""
    from contextlib import nullcontext

    from ..discretization import _coordinate_enclosure as algebra
    from ._mapped_coverage import prepared_face_flux

    budget = algebra._COORDINATE_BUDGET.get()
    if budget is not None:
        budget.reserve(1)
    groups = facets.groups
    unresolved: set[int] = set()
    for group in groups:
        if budget is not None:
            budget.reserve(group.size)
        if group.size > 2:
            raise ValueError(
                "A certified oriented facet chain cannot have more than two owners."
            )
        occurrence = int(group[0])
        owner = int(facets.cells[occurrence])
        own = int(regions[owner])
        other = int(regions[facets.cells[group[1]]]) if group.size == 2 else -1
        if own == other:
            continue
        affected = tuple(region for region in (own, other) if region >= 0)
        if not affected:
            continue
        expression = maps[owner]
        kind, element, controls, vertices = cells[owner]
        if expression is None:
            unresolved.update(affected)
            continue
        with nullcontext() if budget is None else budget.temporary_scope():
            face = tuple(
                vertices.index(int(vertex))
                for vertex in facets.rows[occurrence]
                if vertex >= 0
            )
            flux = prepared_face_flux(element, controls, kind, face)
            if flux is None:
                unresolved.update(affected)
                continue
            if budget is not None:
                budget.reserve(len(affected))
            if own >= 0:
                current = totals[own]
                if current is not None:
                    totals[own] = current + flux
            if other >= 0:
                current = totals[other]
                if current is not None:
                    totals[other] = current - flux
    for region in sorted(unresolved):
        totals[region] = None
        state.add(
            "mapped_rational_boundary_measure"
            if region in rational_regions
            else "mapped_measure_source",
            "unresolved",
            "source_region",
            (region,),
        )


def _mapped_coverage_measures(
    state: _EmbeddingState,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    regions: np.ndarray,
    *,
    embedding: GlobalEmbeddingCertificate | None = None,
    facets: _Facets | None = None,
) -> tuple[
    tuple[Fraction | None, ...],
    list[tuple[str, CellGeometryElement, CoordinateCoefficients, tuple[int, ...]]],
]:
    """Exact region measures; cells keep their element and controls, not expressions.

    A bound, fully traced nonperiodic source chain uses exact physical boundary
    flux, cancelling only earned internal traces. Otherwise each complete source
    Jacobian is integrated over its actual reference chart. No corner surrogate
    replaces either signed integral; all algebra shares the active ledger.
    """
    from contextlib import nullcontext

    from ..discretization import _coordinate_enclosure as algebra
    from ._mapped_coverage import chart_flux, integrate, map_domain, prepared_face_charts

    budget = algebra._COORDINATE_BUDGET.get()
    dimension = mesh.topological_dimension
    boundary_measure = (
        dimension == mesh.ambient_dimension == 3
        and mesh.periodic_topology is None
        and embedding is not None
        and embedding.status == "certified"
        and {"mapped_trace_continuity", "facet_pairing"}
        <= set(embedding.evaluated_checks)
    )
    if boundary_measure and embedding is not None:
        embedding.binding.require(mesh, geometry)
        if budget is not None:
            budget.reserve(1)
            prepared = budget.prepared_cell_cache.get(
                algebra.coordinate_scope_key(mesh, geometry)
            )
            if prepared is not None:
                count = sum(block.cell_count for block in mesh.blocks)
                budget.reserve(2 * count, 256 + 24 * count)
                ids = tuple(
                    int(identifier)
                    for block in mesh.blocks
                    for identifier in np.asarray(block.global_ids, dtype=np.int64)
                )
                if (
                    prepared.cell_ids != ids
                    or len(prepared.descriptors) != count
                    or len(prepared.coordinates) != count
                ):
                    raise ValueError(
                        "Prepared coordinate cells differ from the complete scientific cell bank."
                    )
                totals_: list[Fraction | None] = [
                    Fraction(0) for _ in range(int(np.max(regions, initial=-1)) + 1)
                ]
                cells_: list[
                    tuple[
                        str, CellGeometryElement, CoordinateCoefficients, tuple[int, ...]
                    ]
                ] = list(prepared.descriptors)
                maps_ = list(prepared.coordinates)
                rational_: set[int] = set()
                for owner, polynomial in enumerate(maps_):
                    budget.reserve(len(polynomial))
                    if any(
                        isinstance(value, algebra.RationalPolynomial)
                        for value in polynomial
                    ):
                        region = int(regions[owner])
                        if region >= 0:
                            rational_.add(region)
                _signed_boundary_region_measures(
                    state,
                    regions,
                    cells_,
                    maps_,
                    totals_,
                    rational_,
                    _mesh_facets(mesh, geometry=geometry) if facets is None else facets,
                )
                state.checks.append("mapped_signed_boundary_region_measure")
                return tuple(totals_), cells_
    elements, routes, _ = geometry.resolve(mesh)
    coordinates = algebra.prepared_coordinate_source_bank(geometry)
    totals: list[Fraction | None] = [
        Fraction(0) for _ in range(int(np.max(regions, initial=-1)) + 1)
    ]
    cells: list[
        tuple[str, CellGeometryElement, CoordinateCoefficients, tuple[int, ...]]
    ] = []
    rational_regions: set[int] = set()
    boundary_maps: list[tuple[Expression, ...] | None] = []
    # Root determinants retain complete live element identity and exact controls.
    roots: dict[
        tuple[str, str, CoordinateSourceBank],
        tuple[dict[tuple[int, ...], int], int] | None,
    ] = {}
    affine_roots: dict[tuple[str, str, CoordinateSourceBank], Fraction | None] = {}
    cursor = 0
    for block, element, route in zip(mesh.blocks, elements, routes, strict=True):
        if budget is not None and boundary_measure:
            budget.reserve(
                route.size + block.vertices.size,
                256
                + block.cell_count
                * (128 + 8 * route.shape[1] + 48 * block.vertices.shape[1]),
            )
        local = tuple(
            tuple(coordinates[row] for row in indices)
            for indices in np.asarray(route, dtype=np.int64)
        )
        with nullcontext() if budget is None else budget.temporary_scope():
            affine = (
                _affine_root_chart(element, block.cell_kind, dimension)
                if mesh.ambient_dimension == dimension and not boundary_measure
                else None
            )
            chart = (
                _axis_aligned_root_chart(element, block.cell_kind, dimension)
                if mesh.ambient_dimension == dimension and not boundary_measure
                else None
            )
        reference_measure = integrate(
            {(0,) * dimension: Fraction(1)}, map_domain(block.cell_kind), dimension
        )
        # Signed one-dimensional integrals of the chart's source interval powers.
        powers: dict[tuple[int, int], Fraction] = {}
        for index in range(block.cell_count):
            cells.append(
                (
                    block.cell_kind,
                    element,
                    local[index],
                    tuple(np.asarray(block.vertices[index], dtype=np.int64).tolist()),
                )
            )
            region = int(regions[cursor])
            cursor += 1
            if boundary_measure:
                polynomial = algebra.coordinate_expressions(element, local[index])
                boundary_maps.append(polynomial)
                if polynomial is None:
                    state.add(
                        "mapped_measure_source",
                        "unresolved",
                        "cell",
                        (int(block.global_ids[index]),),
                    )
                    if region >= 0:
                        totals[region] = None
                elif region >= 0 and any(
                    isinstance(value, algebra.RationalPolynomial) for value in polynomial
                ):
                    rational_regions.add(region)
                continue
            with nullcontext() if budget is None else budget.temporary_scope():
                if affine is not None:
                    root, chart_determinant = affine
                    key = (*algebra._element_identity(root), local[index])
                    if key not in affine_roots:
                        expression = algebra.coordinate_expressions(root, local[index])
                        affine_roots[key] = (
                            None
                            if expression is None
                            or any(
                                isinstance(value, algebra.RationalPolynomial)
                                for value in expression
                            )
                            else _affine_jacobian_determinant(
                                tuple(
                                    value
                                    for value in expression
                                    if not isinstance(value, algebra.RationalPolynomial)
                                ),
                                dimension,
                            )
                        )
                        if budget is not None:
                            # The cache survives the cell workspace, including its
                            # exact controls and constant determinant.
                            budget.retain_basis((key, affine_roots[key]))
                    root_determinant = affine_roots[key]
                    if root_determinant is not None:
                        if region >= 0 and totals[region] is not None:
                            if budget is not None:
                                budget.reserve(2)
                            current = totals[region]
                            if current is not None:
                                totals[region] = (
                                    current
                                    + root_determinant
                                    * chart_determinant
                                    * reference_measure
                                )
                        continue
                if chart is not None:
                    root, origin, steps = chart
                    key = (*algebra._element_identity(root), local[index])
                    if key not in roots:
                        expression = algebra.coordinate_expressions(root, local[index])
                        polynomials = (
                            None
                            if expression is None
                            or any(
                                isinstance(value, algebra.RationalPolynomial)
                                for value in expression
                            )
                            else tuple(
                                value
                                for value in expression
                                if not isinstance(value, algebra.RationalPolynomial)
                            )
                        )
                        root_determinant = (
                            None
                            if polynomials is None
                            else _box_jacobian_determinant(polynomials)
                        )
                        if budget is not None:
                            budget.retain_basis((key, root_determinant))
                        roots[key] = root_determinant
                    prepared = roots[key]
                    if prepared is not None:
                        if region < 0 or totals[region] is None:
                            continue
                        determinant, denominator = prepared
                        if budget is not None:
                            budget.reserve((dimension + 2) * len(determinant))
                        measure = Fraction(0)
                        for exponents, value in determinant.items():
                            weight = Fraction(value, denominator)
                            for axis, exponent in enumerate(exponents):
                                integral = powers.get((axis, exponent))
                                if integral is None:
                                    integral = powers[axis, exponent] = (
                                        (origin[axis] + steps[axis]) ** (exponent + 1)
                                        - origin[axis] ** (exponent + 1)
                                    ) / (exponent + 1)
                                weight *= integral
                            measure += weight
                        current = totals[region]
                        if current is not None:
                            totals[region] = current + measure
                        continue
                polynomial = algebra.coordinate_expressions(element, local[index])
                if polynomial is None:
                    state.add(
                        "mapped_measure_source",
                        "unresolved",
                        "cell",
                        (int(block.global_ids[index]),),
                    )
                    if region >= 0:
                        totals[region] = None
                elif region >= 0 and any(
                    isinstance(value, algebra.RationalPolynomial) for value in polynomial
                ):
                    rational_regions.add(region)
                elif region >= 0 and totals[region] is not None:
                    polynomial_coordinates = tuple(
                        value
                        for value in polynomial
                        if not isinstance(value, algebra.RationalPolynomial)
                    )
                    if map_domain(block.cell_kind) == "box" and len(
                        polynomial_coordinates
                    ) == dimension in (2, 3):
                        measure = _box_jacobian_measure(polynomial_coordinates)
                    else:
                        jacobian = tuple(
                            tuple(
                                algebra.derivative(value, axis)
                                for axis in range(dimension)
                            )
                            for value in polynomial_coordinates
                        )
                        measure = integrate(
                            algebra.determinant(jacobian),
                            map_domain(block.cell_kind),
                            dimension,
                        )
                    current = totals[region]
                    if current is not None:
                        totals[region] = current + measure
    if boundary_measure:
        _signed_boundary_region_measures(
            state,
            regions,
            cells,
            boundary_maps,
            totals,
            rational_regions,
            _mesh_facets(mesh, geometry=geometry) if facets is None else facets,
        )
        state.checks.append("mapped_signed_boundary_region_measure")
        return tuple(totals), cells
    if rational_regions:
        facets = _mesh_facets(mesh, geometry=geometry)
        for region in sorted(rational_regions):
            total = Fraction(0)
            resolved = True
            for occurrence, owner in enumerate(facets.cells.tolist()):
                if regions[owner] != region:
                    continue
                group = facets.group[occurrence]
                incident = facets.cells[facets.group == group]
                if incident.size == 2 and np.all(regions[incident] == region):
                    continue
                kind, element, controls, vertices = cells[owner]
                with nullcontext() if budget is None else budget.temporary_scope():
                    expression = algebra.coordinate_expressions(element, controls)
                    if expression is None or dimension != 3:
                        resolved = False
                        break
                    face = tuple(
                        vertices.index(int(vertex))
                        for vertex in facets.rows[occurrence]
                        if vertex >= 0
                    )
                    charts = prepared_face_charts(element, controls, kind, face)
                    if charts is None:
                        resolved = False
                        break
                    for chart_expression, chart_domain in charts:
                        flux = chart_flux(chart_expression, chart_domain)
                        if flux is None:
                            resolved = False
                            break
                        total += flux
                if not resolved:
                    break
            totals[region] = total if resolved else None
            if not resolved:
                state.add(
                    "mapped_rational_boundary_measure",
                    "unresolved",
                    "source_region",
                    (region,),
                )
    return tuple(totals), cells


def _declared_coverage_measures(
    domain: PiecewiseLinearDomain | MappedReferenceDomain,
) -> tuple[Fraction, ...]:
    from ._mapped_reference_domain import MappedReferenceDomain

    if isinstance(domain, MappedReferenceDomain):
        return domain.exact_region_measures()
    integers, exponent = _dyadic_integers(domain.vertices)
    dimension = domain.ambient_dimension
    factor = Fraction(2) ** (dimension * exponent) / (2 if dimension == 2 else 6)
    weights = np.ones(domain.facets.shape[0], dtype=np.int64)
    return tuple(
        (back + front) * factor
        for back, front in zip(
            _region_measures(
                integers,
                domain.facets,
                weights,
                domain.facet_regions[:, 0],
                len(domain.region_ids),
            ),
            _region_measures(
                integers,
                domain.facets,
                -weights,
                domain.facet_regions[:, 1],
                len(domain.region_ids),
            ),
            strict=True,
        )
    )


def _exact_chart_plane(
    chart: tuple[Expression, ...], /
) -> tuple[Fraction, ...] | Literal["nonplanar"] | None:
    """Exact plane of a polynomial facet chart's image, ``"nonplanar"``, or ``None``.

    The images of the parameter origin and unit points lie on the chart's
    polynomial image. When they span a unique plane, the exact polynomial plane
    expression of the chart decides: zero means the whole image lies in that
    plane, keyed with the source groups' exact normalization; nonzero means no
    plane contains the image, so every declared plane rejects it. Rational,
    degenerate or non-hypersurface charts return ``None`` for the full search.
    """
    from ..discretization import _coordinate_enclosure as algebra
    from ._planar_coverage import plane_key

    polynomials = tuple(
        value for value in chart if not isinstance(value, algebra.RationalPolynomial)
    )
    if len(polynomials) != len(chart) or not polynomials:
        return None
    dimension = next((len(index) for value in polynomials for index in value), None)
    if dimension is None or dimension + 1 != len(polynomials):
        return None
    points = tuple(
        tuple(
            algebra.evaluate(
                value, tuple(Fraction(int(axis == unit)) for axis in range(dimension))
            )
            for value in polynomials
        )
        for unit in (-1, *range(dimension))
    )
    key = plane_key(points)
    if key is None:
        return None
    plane = key[0]
    if all(sum(index) <= 1 for value in polynomials for index in value):
        # An affine image is the affine hull of these spanning points.
        return plane
    residual = algebra.add(
        algebra.sum_polynomials(
            tuple(
                algebra.scale(value, weight)
                for value, weight in zip(polynomials, plane[:-1], strict=True)
            )
        ),
        algebra.constant(plane[-1], dimension),
    )
    return "nonplanar" if residual else plane


def _mapped_domain_coverage(
    state: _EmbeddingState,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    domain: PiecewiseLinearDomain,
    regions: np.ndarray,
    limits: MeshCertificateLimits,
    binding: MeshCertificateBinding,
    embedding: GlobalEmbeddingCertificate,
) -> DomainCoverageCertificate:
    from contextlib import nullcontext

    from ..discretization import _coordinate_enclosure as algebra
    from ._mapped_coverage import prepared_face_charts, SubdivisionLedger
    from ._planar_coverage import mapped_containment, source_groups

    budget = algebra._COORDINATE_BUDGET.get()

    facets = _mesh_facets(mesh, geometry=geometry)
    totals, cells = _mapped_coverage_measures(
        state, mesh, geometry, regions, embedding=embedding, facets=facets
    )
    achieved = (
        *totals,
        *(Fraction(0) for _ in range(len(domain.region_ids) - len(totals))),
    )
    source, source_overlaps, used, exceeded = source_groups(
        domain.vertices,
        domain.facets,
        domain.facet_regions,
        limits.maximum_candidate_pairs - state.candidate_pairs,
    )
    state.candidate_pairs += used
    for first, second in source_overlaps:
        state.add(
            "overlapping_source_facets", "violated", "source_facet", (first, second)
        )
    if exceeded:
        state.add("coverage_candidate_capacity", "unresolved", "mesh")
    # Exact source groups by their normalized plane, in source order. A chart
    # whose image lies in another plane, or whose region pair differs, is
    # "absent" for that group by definition of the containment proof.
    by_plane: dict[tuple[Fraction, ...], list[int]] = {}
    if budget is not None:
        budget.reserve(len(source), 128 * len(source))
    for target, declared in enumerate(source):
        by_plane.setdefault(declared.plane, []).append(target)
    every_target = tuple(range(len(source)))
    groups = facets.groups
    # One sorted lookup identifies every facet occurrence, not one per facet.
    entity_ids = _facet_entity_ids(mesh, facets.rows)
    covered = [Fraction(0) for _ in source]
    uncertain_sources: set[int] = set()
    subdivision_work = SubdivisionLedger(candidate_pairs=state.candidate_pairs)
    relations: dict[_CoverageOverlapKey, Fraction] = {}
    for group in groups:
        occurrence = int(group[0])
        own = int(regions[facets.cells[occurrence]])
        other = int(regions[facets.cells[group[1]]]) if group.size == 2 else -1
        if group.size > 2 or own == other:
            continue
        kind, element, controls, vertices = cells[int(facets.cells[occurrence])]
        rows = facets.rows[occurrence]
        entity = int(entity_ids[occurrence])
        facet_keys: list[_CoverageOverlapKey] = []
        # The source map remains ledger-owned; only this facet's workspace ends.
        with nullcontext() if budget is None else budget.temporary_scope():
            face = tuple(vertices.index(int(vertex)) for vertex in rows if vertex >= 0)
            charts = prepared_face_charts(element, controls, kind, face)
            if charts is None:
                state.add("mapped_boundary_source", "unresolved", "facet", (entity,))
                uncertain_sources.update(range(len(source)))
                continue
            for chart, chart_domain in charts:
                matched = False
                unresolved = exceeded
                plane = _exact_chart_plane(chart)
                targets = (
                    every_target
                    if plane is None
                    else ()
                    if plane == "nonplanar"
                    else tuple(
                        target
                        for target in by_plane.get(plane, ())
                        if source[target].regions in ((own, other), (other, own))
                    )
                )
                for target in targets:
                    declared = source[target]
                    outcome, measure, candidates = mapped_containment(
                        chart,
                        chart_domain,
                        declared,
                        own,
                        other,
                        limits.maximum_bernstein_nodes,
                        limits.maximum_subdivision_depth,
                        limits.maximum_subdivision_pieces,
                        limits.maximum_candidate_pairs,
                        subdivision_work,
                    )
                    if outcome == "proven":
                        covered[target] += measure
                        for local in candidates:
                            member = declared.members[local]
                            source_measure = declared.measures[local]
                            key: _CoverageOverlapKey = (
                                entity,
                                member,
                                declared.axes,
                                "candidate",
                                "physical",
                            )
                            if key not in relations:
                                facet_keys.append(key)
                            relations[key] = min(
                                source_measure,
                                relations.get(key, Fraction(0))
                                + min(measure, source_measure),
                            )
                        matched = True
                        break
                    if outcome == "unresolved":
                        uncertain_sources.add(target)
                        unresolved = True
                if not matched:
                    state.add(
                        "mapped_facet_containment"
                        if unresolved
                        else "unmatched_interface_facet"
                        if other >= 0
                        else "unmatched_boundary_facet",
                        "unresolved" if unresolved else "violated",
                        "facet",
                        (entity,),
                    )
        if budget is not None and facet_keys:
            # Overlap relations outlive the facet workspace: their exact keys and
            # measures stay charged to this owner's storage for the whole proof.
            bits = max(
                abs(relations[key].numerator).bit_length()
                + relations[key].denominator.bit_length()
                for key in facet_keys
            )
            algebra._reserve_polynomial(len(facet_keys), len(facet_keys), 5, bits)
    state.candidate_pairs = subdivision_work.candidate_pairs
    state.subdivision_pieces = subdivision_work.pieces
    state.subdivision_depth = subdivision_work.maximum_depth
    requested = _declared_coverage_measures(domain)
    equal = 0
    for target, (actual, declared) in enumerate(zip(covered, source, strict=True)):
        expected = sum(declared.measures, Fraction(0))
        boundary = -1 in declared.regions
        if actual == expected:
            # Compact contained images, interior-disjointness and exact union
            # measure imply coverage of every authoritative member, not merely
            # almost-everywhere coverage of a merged scientific face.
            equal += len(declared.members)
        elif target not in uncertain_sources:
            check = (
                ("uncovered_boundary" if boundary else "omitted_interface")
                if actual < expected
                else (
                    "double_covered_boundary" if boundary else "double_covered_interface"
                )
            )
            state.add(check, "violated", "source_facet", declared.members)
    return _coverage_result(
        binding,
        embedding,
        domain,
        state,
        requested,
        achieved,
        equal,
        relations,
        (),
        expression_work=_active_expression_work(),
    )


def certify_domain_coverage(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    domain: PiecewiseLinearDomain | MappedReferenceDomain,
    cell_regions: ArrayLike,
    /,
    *,
    embedding: GlobalEmbeddingCertificate,
    limits: MeshCertificateLimits | None = None,
) -> DomainCoverageCertificate:
    """Certify coverage of an independently declared planar or mapped volume.

    ``cell_regions`` assigns every cell (mesh block order) exclusively to one
    index of ``domain.region_ids``; ``-1`` marks an unassigned cell. The
    embedding certificate is the interior-disjointness premise.
    """

    from ._mapped_reference_domain import MappedReferenceDomain

    if not isinstance(mesh, CellMesh):
        raise TypeError("mesh must be CellMesh.")
    if not isinstance(domain, (PiecewiseLinearDomain, MappedReferenceDomain)):
        raise TypeError("domain must be PiecewiseLinearDomain or MappedReferenceDomain.")
    if not isinstance(embedding, GlobalEmbeddingCertificate):
        raise TypeError("embedding must be GlobalEmbeddingCertificate.")
    embedding.binding.require(mesh, geometry)
    limits_ = MeshCertificateLimits() if limits is None else limits
    if not isinstance(limits_, MeshCertificateLimits):
        raise TypeError("limits must be MeshCertificateLimits or None.")
    dimension = mesh.topological_dimension
    if dimension != mesh.ambient_dimension or dimension != domain.ambient_dimension:
        raise ValueError(
            "Domain coverage requires a volume mesh of the domain dimension."
        )
    if mesh.storage is not None:
        binding = MeshCertificateBinding(
            mesh,
            geometry,
            embedding.binding.coordinate_scope,
            limits_,
            source_id=domain.source_id,
            source_revision=domain.source_revision,
        )
        state = _EmbeddingState([], ["domain_coverage"])
        state.add("owner_local_domain_coverage_premise", "unresolved", "mesh")
        return _coverage_result(
            binding,
            embedding,
            domain,
            state,
            _declared_coverage_measures(domain),
            tuple(None for _ in domain.region_ids),
            0,
            {},
            (),
            expression_work=(0, 0, 0),
        )
    regions = parse(
        np.asarray(cell_regions, dtype=np.int64), HostInt64[_MeshCellDim], "cell_regions"
    )
    cell_ids = _cell_global_ids(mesh)
    if regions.shape[0] != cell_ids.size:
        raise ValueError("cell_regions must assign every mesh cell.")
    if np.any(regions < -1) or np.any(regions >= len(domain.region_ids)):
        raise ValueError("cell_regions must index the declared regions or be -1.")
    budget = coordinate_enclosure_budget(
        limits_.maximum_work_units, limits_.maximum_scratch_bytes
    )
    with ExitStack() as preparation:
        scope_resource: CoordinateEnclosureResourceError | None = None
        try:
            preparation.enter_context(budget.activate())
            preparation.enter_context(
                budget.bound_stage(
                    limits_.maximum_work_units, limits_.maximum_scratch_bytes
                )
            )
            scope, _, prepared_source = _coordinate_scope(mesh, geometry, budget)
        except CoordinateEnclosureResourceError as error:
            scope = embedding.binding.coordinate_scope
            prepared_source = ()
            scope_resource = error
        binding = MeshCertificateBinding(
            mesh,
            geometry,
            scope,
            limits_,
            source_id=domain.source_id,
            source_revision=domain.source_revision,
        )
        state = _EmbeddingState([], ["domain_coverage"])
        if scope_resource is not None:
            state.add(
                "mapped_coverage_resource_budget",
                "unresolved",
                "mesh",
                resource_error=scope_resource,
                expression_budget=budget,
            )
            return _coverage_result(
                binding,
                embedding,
                domain,
                state,
                _declared_coverage_measures(domain),
                tuple(None for _ in domain.region_ids),
                0,
                {},
                (),
                expression_work=(
                    budget.work_units,
                    budget.peak_bytes_upper,
                    budget.required_work_units,
                ),
            )
        if embedding.status != "certified":
            state.add("embedding_premise", "unresolved", "mesh")
        unassigned = regions < 0
        if np.any(unassigned):
            state.add("unassigned_cell", "violated", "cell", cell_ids[unassigned])
        if isinstance(domain, MappedReferenceDomain) or scope == "mapped":
            from ._mapped_reference_coverage import certify_mapped_reference_coverage

            # Scope preparation and the theorem share this request's one ledger.
            try:
                with budget.activate():
                    if isinstance(domain, MappedReferenceDomain):
                        return certify_mapped_reference_coverage(
                            state,
                            mesh,
                            geometry,
                            domain,
                            regions,
                            limits_,
                            binding,
                            embedding,
                        )
                    return _mapped_domain_coverage(
                        state,
                        mesh,
                        geometry,
                        domain,
                        regions,
                        limits_,
                        binding,
                        embedding,
                    )
            except CoordinateEnclosureResourceError as error:
                state.add(
                    "mapped_coverage_resource_budget",
                    "unresolved",
                    "mesh",
                    resource_error=error,
                    expression_budget=budget,
                )
                return _coverage_result(
                    binding,
                    embedding,
                    domain,
                    state,
                    _declared_coverage_measures(domain),
                    tuple(None for _ in domain.region_ids),
                    0,
                    {},
                    (),
                    expression_work=(
                        budget.work_units,
                        budget.peak_bytes_upper,
                        budget.required_work_units,
                    ),
                )
        from ..discretization._exact_plc_geometry import (
            ExactPlcCellGeometryConvexSource,
            ExactPlcCellGeometrySource,
        )
        from ..discretization._exact_power_geometry import (
            ExactPowerCellGeometryLinearActionSource,
            ExactPowerCellGeometryRestrictionSource,
            ExactPowerCellGeometrySource,
        )

        expression_work: tuple[int, int, int]
        match geometry.exact_source:
            case None:
                # Binary64 mesh-coordinate predicates form no coordinate expression.
                terms = _affine_coverage_terms(
                    state,
                    mesh,
                    domain,
                    regions,
                    np.asarray(mesh.coordinates, dtype=np.float64),
                    limits_,
                )
                expression_work = (
                    budget.work_units,
                    budget.peak_bytes_upper,
                    budget.required_work_units,
                )
            case (
                ExactPowerCellGeometrySource()
                | ExactPowerCellGeometryRestrictionSource()
                | ExactPowerCellGeometryLinearActionSource()
                | ExactPlcCellGeometrySource()
                | ExactPlcCellGeometryConvexSource()
            ):
                # The same request ledger owns source vertices and the proof.
                try:
                    with budget.activate():
                        exact_points = np.asarray(prepared_source, dtype=object)
                        predicate_points, _ = _exact_source_predicate_view(exact_points)
                        terms = _affine_coverage_terms(
                            state, mesh, domain, regions, predicate_points, limits_
                        )
                except CoordinateEnclosureResourceError as error:
                    state.add(
                        "mapped_coverage_resource_budget",
                        "unresolved",
                        "mesh",
                        resource_error=error,
                        expression_budget=budget,
                    )
                    return _coverage_result(
                        binding,
                        embedding,
                        domain,
                        state,
                        _declared_coverage_measures(domain),
                        tuple(None for _ in domain.region_ids),
                        0,
                        {},
                        (),
                        expression_work=(
                            budget.work_units,
                            budget.peak_bytes_upper,
                            budget.required_work_units,
                        ),
                    )
                expression_work = (
                    budget.work_units,
                    budget.peak_bytes_upper,
                    budget.required_work_units,
                )
            case invalid:
                assert_never(invalid)
        requested, achieved, covered, relations = terms
        return _coverage_result(
            binding,
            embedding,
            domain,
            state,
            requested,
            achieved,
            covered,
            relations,
            (),
            expression_work=expression_work,
        )


def _affine_coverage_terms(
    state: _EmbeddingState,
    mesh: CellMesh,
    domain: PiecewiseLinearDomain,
    regions: np.ndarray,
    points: np.ndarray,
    limits: MeshCertificateLimits,
    /,
) -> tuple[
    tuple[Fraction, ...], tuple[Fraction, ...], int, dict[_CoverageOverlapKey, Fraction]
]:
    """Exact affine coverage of binary64 or exact source points.

    Every exact-point operation charges the active coordinate ledger, if any.
    Returns requested and achieved region measures, covered source facets and
    overlap relations.
    """
    dimension = mesh.topological_dimension
    facets = _mesh_facets(mesh)
    covering = _covering_pieces(state, mesh, points, facets, regions, limits)
    declared_corners = domain.vertices[domain.facets]
    axes, orientation, certain = _projection_axes(declared_corners)
    if not np.all(certain):
        state.add(
            "declared_projection_predicates",
            "unresolved",
            "source_facet",
            np.flatnonzero(~certain),
        )
    covered, unmatched, incomplete, relations = _partition_covering_pieces(
        state, covering, points, domain, axes, limits
    )
    interface = covering.other_region >= 0
    for check, rows in (
        ("unmatched_boundary_facet", unmatched & ~interface),
        ("unmatched_interface_facet", unmatched & interface),
    ):
        if np.any(rows):
            state.add(check, "violated", "facet", covering.facet_entities[rows])
    stacked = np.concatenate((points, domain.vertices))
    from ._exact_polyhedral_geometry import coordinate_integers

    integers, coordinate_scale = coordinate_integers(stacked)
    mesh_integers = integers[: points.shape[0]]
    declared_integers = integers[points.shape[0] :]
    projected_factor = coordinate_scale ** (dimension - 1)
    if dimension == 3:
        projected_factor /= 2
    declared_measure = tuple(
        value * projected_factor
        for value in _projected_measures(declared_integers, domain.facets, axes).tolist()
    )
    boundary_facet = np.any(domain.facet_regions < 0, axis=1)
    deficit = np.asarray(
        [value < total for value, total in zip(covered, declared_measure, strict=True)],
        dtype=np.bool_,
    )
    excess = np.asarray(
        [value > total for value, total in zip(covered, declared_measure, strict=True)],
        dtype=np.bool_,
    )
    for check, rows in (
        ("uncovered_boundary", deficit & boundary_facet),
        ("omitted_interface", deficit & ~boundary_facet),
        ("double_covered_boundary", excess & boundary_facet),
        ("double_covered_interface", excess & ~boundary_facet),
    ):
        if np.any(rows) and not incomplete:
            state.add(check, "violated", "source_facet", np.flatnonzero(rows))
    region_count = len(domain.region_ids)
    declared_totals = [
        back + front
        for back, front in zip(
            _region_measures(
                declared_integers,
                domain.facets,
                np.ones((domain.facets.shape[0],), dtype=np.int64),
                domain.facet_regions[:, 0],
                region_count,
            ),
            _region_measures(
                declared_integers,
                domain.facets,
                -np.ones((domain.facets.shape[0],), dtype=np.int64),
                domain.facet_regions[:, 1],
                region_count,
            ),
            strict=True,
        )
    ]
    pieces, owners = _cell_measure_pieces(state, mesh, points, facets, limits)
    achieved_totals = _region_measures(
        mesh_integers,
        pieces,
        np.ones((owners.size,), dtype=np.int64),
        regions[owners],
        region_count,
    )
    denominator = 2 if dimension == 2 else 6
    factor = coordinate_scale**dimension / denominator
    return (
        tuple(value * factor for value in declared_totals),
        tuple(value * factor for value in achieved_totals),
        int(np.count_nonzero(~deficit & ~excess)),
        relations,
    )


# Source fidelity -------------------------------------------------------------------


def _fidelity_items(
    state: _EmbeddingState,
    mesh: CellMesh,
    points: np.ndarray,
    limits: MeshCertificateLimits,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Affine boundary pieces compared with the source and their entity ids."""

    dimension = mesh.topological_dimension
    cell_ids = _cell_global_ids(mesh)
    if dimension == mesh.ambient_dimension:
        facets = _mesh_facets(mesh)
        boundary = _boundary_pieces(state, mesh, points, facets, limits)
        return boundary.pieces, boundary.entities[boundary.piece_owners]
    if dimension == 1:
        items = np.concatenate(
            [np.asarray(block.vertices, dtype=np.int64) for block in mesh.blocks]
        )
        return items, cell_ids
    triangles, owners, nonplanar, undecided = _loop_triangles(
        points, _cell_loops(mesh), limits.maximum_candidate_pairs
    )
    if np.any(nonplanar | undecided):
        state.add("curved_cell", "unresolved", "cell", cell_ids[nonplanar | undecided])
    return triangles, cell_ids[owners]


def _lattice(order: int, vertices: int, /) -> np.ndarray:
    """Barycentric lattice of spacing ``1 / order`` on a simplex."""

    rows = [
        index
        for index in np.ndindex(*((order + 1,) * (vertices - 1)))
        if sum(index) <= order
    ]
    weights = np.asarray(rows, dtype=np.float64) / order
    return np.concatenate((1.0 - np.sum(weights, axis=1, keepdims=True), weights), axis=1)


def _point_simplex_distances(samples: np.ndarray, simplices: np.ndarray, /) -> np.ndarray:
    """Euclidean distances ``(samples, simplices)`` to closed segments/triangles."""

    if simplices.shape[1] == 2:
        start = simplices[None, :, 0]
        edge = simplices[None, :, 1] - start
        offset = samples[:, None, :] - start
        length = np.maximum(np.sum(edge * edge, axis=-1), np.finfo(np.float64).tiny)
        t = np.clip(np.sum(offset * edge, axis=-1) / length, 0.0, 1.0)
        return np.linalg.norm(offset - t[..., None] * edge, axis=-1)
    a = simplices[None, :, 0]
    b = simplices[None, :, 1]
    c = simplices[None, :, 2]
    p = samples[:, None, :]
    distances = np.minimum.reduce(
        [
            _point_simplex_distances(samples, np.stack((u, v), axis=1))
            for u, v in (
                (simplices[:, 0], simplices[:, 1]),
                (simplices[:, 1], simplices[:, 2]),
                (simplices[:, 2], simplices[:, 0]),
            )
        ]
    )
    normal = np.cross(b - a, c - a)
    norm = np.maximum(np.linalg.norm(normal, axis=-1), np.finfo(np.float64).tiny)
    height = np.sum((p - a) * normal, axis=-1) / norm
    foot = p - (height / norm)[..., None] * normal
    # The foot of the perpendicular lies in the triangle iff it is on the inner
    # side of every edge; otherwise the nearest point is on an edge.
    inside = np.ones(distances.shape, dtype=np.bool_)
    for u, v in ((a, b), (b, c), (c, a)):
        inside &= np.sum(np.cross(v - u, foot - u) * normal, axis=-1) >= 0.0
    return np.where(inside, np.minimum(distances, np.abs(height)), distances)


def _mesh_to_source(
    state: _EmbeddingState,
    source: SourceBoundaryQuery,
    points: np.ndarray,
    items: np.ndarray,
    entities: np.ndarray,
    order: int,
    tolerance: float,
    limits: MeshCertificateLimits,
    /,
) -> tuple[SourceBoundSemantics, float, float, int]:
    """Refine a continuous affine cover until tolerance or an explicit limit."""
    corners = points[items]
    edges = np.stack(
        [
            np.linalg.norm(corners[:, i] - corners[:, j], axis=-1)
            for i, j in combinations(range(items.shape[1]), 2)
        ],
        axis=1,
    )
    diameter = np.max(edges, axis=1) * (1 + 16 * _EPSILON)
    rounding = 16 * _EPSILON * np.max(np.abs(corners), axis=(1, 2))
    bounds = np.full((items.shape[0],), math.inf, dtype=np.float64)
    witnesses = np.zeros((items.shape[0],), dtype=np.float64)
    active = np.arange(items.shape[0], dtype=np.int64)
    evaluations, pieces = 0, 0
    semantics: SourceBoundSemantics = "sampled"
    for depth in range(limits.maximum_subdivision_depth + 1):
        dimension = items.shape[1] - 1
        count = active.size * math.comb(order + dimension, dimension)
        added_pieces = active.size * order**dimension
        capacity = next(
            (
                name
                for name, work, maximum in (
                    (
                        "mesh_sample_capacity",
                        evaluations + count,
                        limits.maximum_source_samples,
                    ),
                    (
                        "distance_capacity",
                        evaluations + count,
                        limits.maximum_distance_evaluations,
                    ),
                    (
                        "subdivision_capacity",
                        pieces + added_pieces,
                        limits.maximum_subdivision_pieces,
                    ),
                    (
                        "source_work_capacity",
                        evaluations + count,
                        limits.maximum_work_units,
                    ),
                    (
                        "source_scratch_capacity",
                        count * 256 + corners.nbytes * 4,
                        limits.maximum_scratch_bytes,
                    ),
                )
                if work > maximum
            ),
            None,
        )
        if capacity is not None:
            state.add(capacity, "unresolved", "facet", entities[active])
            break
        lattice = _lattice(order, items.shape[1])
        samples = ein.contract("lv,ivd->ild", lattice, corners[active]).reshape(
            -1, points.shape[1]
        )
        distance = source.boundary_distance(samples)
        if distance.lower.shape != (count,):
            raise ValueError("Source distance bounds must match the query count.")
        evaluations += count
        pieces += added_pieces
        semantics = distance.semantics
        # Affine lattice sub-simplices cover the entire simplex. Their diameter
        # decreases under refinement; source distance is one-Lipschitz.
        upper = np.max(distance.upper.reshape(active.size, -1), axis=1)
        lower = np.max(distance.lower.reshape(active.size, -1), axis=1)
        bounds[active] = np.minimum(
            bounds[active],
            np.nextafter(upper + diameter[active] / order + rounding[active], math.inf),
        )
        witnesses[active] = np.maximum(
            witnesses[active],
            np.maximum(np.nextafter(lower - rounding[active], -math.inf), 0.0),
        )
        certified = semantics == "certified"
        violated = certified & (witnesses[active] > tolerance)
        if np.any(violated):
            state.add(
                "boundary_deviation", "violated", "facet", entities[active[violated]]
            )
        active = active[(bounds[active] > tolerance) & ~violated]
        if not active.size or not certified:
            break
        if depth == limits.maximum_subdivision_depth:
            state.add("subdivision_depth", "unresolved", "facet", entities[active])
            break
        order *= 2
    if active.size:
        state.add("boundary_deviation", "unresolved", "facet", entities[active])
    return semantics, float(np.max(bounds)), float(np.max(witnesses)), evaluations


def _source_to_mesh(
    state: _EmbeddingState,
    source: SourceBoundaryQuery,
    points: np.ndarray,
    items: np.ndarray,
    tolerance: float,
    limits: MeshCertificateLimits,
    /,
) -> tuple[SourceBoundSemantics, float, float, int]:
    samples = source.boundary_samples(limits.maximum_source_samples)
    count = samples.points.shape[0]
    if not samples.complete:
        state.add("source_sample_capacity", "unresolved", "mesh")
        return "sampled", math.inf, 0.0, count
    # Candidate work is admitted during exact packed traversal, not as an
    # unexecuted Cartesian product of every source sample and target facet.
    if count == 0:
        return samples.semantics, 0.0, 0.0, 0
    from .._meshcore import charge_native_geometry_queries

    class DistanceCapacity(Exception):
        def __init__(self, requested: int) -> None:
            self.requested = requested

    budget = coordinate_enclosure_budget(
        limits.maximum_work_units, limits.maximum_scratch_bytes
    )
    evaluations = 0
    owners: list[object] = []

    def work(units: int) -> None:
        budget.reserve(units)
        budget.charge_native_work(units)

    def storage(nbytes: int) -> None:
        budget.reserve(0, nbytes)

    def visit(kind: str, units: int) -> None:
        nonlocal evaluations
        del kind
        requested = evaluations + units
        if requested > limits.maximum_distance_evaluations:
            raise DistanceCapacity(requested)
        budget.reserve(units)
        charge_native_geometry_queries(units)
        budget.charge_native_work(units)
        evaluations = requested

    try:
        with budget.temporary_scope():
            owners.append((points, items, samples))
            work(items.size * points.shape[1])
            storage(items.size * points.shape[1] * 8)
            corners = points[items]
            owners.append(corners)
            work(2 * corners.size)
            storage(2 * corners.shape[0] * corners.shape[2] * 8)
            boxes_min, boxes_max = np.min(corners, axis=1), np.max(corners, axis=1)
            owners.append((boxes_min, boxes_max))
            tree = prepare_bvh(
                boxes_min,
                boxes_max,
                dtype=np.float64,
                _charge_work=work,
                _reserve_storage=storage,
            )
            owners.append(tree)
            work(2 * (points.size + samples.points.size))
            storage(points.nbytes + samples.points.nbytes)
            scale = max(
                float(np.max(np.abs(points))), float(np.max(np.abs(samples.points)))
            )
            slack = 32.0 * _EPSILON * scale
            dimension = points.shape[1]
            # The existing closed-simplex owner uses bounded leaf batches.
            # Its original 512-byte per-pair workspace bound is admitted
            # before dispatch, alongside the explicit box-distance buffers.
            storage(512 * tree.leaf_size + 16 * dimension)
            delta = np.empty((dimension,), dtype=np.float64)
            other = np.empty((dimension,), dtype=np.float64)
            owners.append((delta, other))

            def bounds(
                point: np.ndarray,
                node: int,
                lower: np.ndarray,
                upper: np.ndarray,
                out: np.ndarray,
            ) -> None:
                del node
                np.subtract(lower, point, out=delta)
                np.subtract(point, upper, out=other)
                np.maximum(delta, other, out=delta)
                np.maximum(delta, 0.0, out=delta)
                distance = math.sqrt(float(np.dot(delta, delta)))
                out[0] = max(0.0, distance * (1 - 16 * _EPSILON) - slack)

            def values(point: np.ndarray, selected: np.ndarray, out: np.ndarray) -> None:
                out[:, 0] = _point_simplex_distances(point[None], corners[selected])[0]

            minima, _, _, _ = bvh_host_minima(
                tree,
                samples.points,
                objective_count=1,
                node_lower_bounds=bounds,
                item_values=values,
                visit=visit,
                admit_storage=storage,
                retain_owner=owners.append,
            )
            work(8 * count)
            storage(3 * count * 8 + 4 * count)
            nearest = minima[:, 0]
            upper = nearest * (1.0 + 16.0 * _EPSILON) + slack + samples.covering_radius
            lower = np.maximum(
                nearest * (1.0 - 16.0 * _EPSILON) - slack - samples.point_error,
                0.0,
            )
            certified = samples.semantics == "certified"
            exceeded = lower > tolerance
            open_samples = (upper > tolerance) & ~(exceeded & certified)
            if np.any(exceeded & certified):
                state.add(
                    "source_uncovered",
                    "violated",
                    "source_sample",
                    np.flatnonzero(exceeded),
                )
            if np.any(open_samples):
                state.add(
                    "source_uncovered",
                    "unresolved",
                    "source_sample",
                    np.flatnonzero(open_samples),
                )
            result = samples.semantics, float(np.max(upper)), float(np.max(lower)), count
    except DistanceCapacity as failure:
        state.findings.append(
            MeshCertificateFinding(
                "distance_capacity",
                "unresolved",
                "mesh",
                resource="distance_evaluations",
                requested=(
                    ("limit", limits.maximum_distance_evaluations),
                    ("requested", failure.requested),
                ),
                achieved=(("completed", evaluations),),
            )
        )
        return samples.semantics, math.inf, 0.0, count
    except CoordinateEnclosureResourceError as failure:
        state.add(
            "source_cover_expression_budget",
            "unresolved",
            "mesh",
            resource_error=failure,
            expression_budget=budget,
        )
        return samples.semantics, math.inf, 0.0, count
    return result


type _FidelityPolynomial = Expression


@dataclass(frozen=True)
class _FidelityMap:
    coordinates: tuple[_FidelityPolynomial, ...]
    dimension: int
    entity: int
    entity_kind: MeshCertificateEntityKind = "cell"


def _fidelity_maps(
    state: _EmbeddingState,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    /,
    *,
    target_facet_ids: tuple[int, ...] | None = None,
) -> list[_FidelityMap]:
    """Restrict actual source expressions to an exact boundary simplex chain."""
    from ..discretization._coordinate_enclosure import (
        affine_arguments,
        coordinate_expressions,
        expression_compose as compose,
    )

    elements, routes, _ = geometry.resolve(mesh)
    values = geometry.source_coordinates()
    volume = mesh.topological_dimension == mesh.ambient_dimension
    boundary: set[tuple[int, ...]] = set()
    facet_entities: dict[tuple[int, ...], int] = {}
    if volume:
        facets = _mesh_facets(mesh)
        identifiers = _facet_entity_ids(mesh, facets.rows)
        facet_entities = {
            tuple(sorted(int(v) for v in row if v >= 0)): int(identifier)
            for row, identifier in zip(facets.rows, identifiers, strict=True)
        }
        if mesh.periodic_topology is not None:
            degree = mesh.topological_dimension - 1
            periodic = mesh.periodic_topology
            identifiers = _facet_entity_ids(mesh, facets.rows)
            entity_rows = {
                int(value): index
                for index, value in enumerate(
                    np.asarray(mesh.entity_set(degree).entity_ids)
                )
            }
            orbit = np.asarray(periodic.orbits(degree)[0])
            walls = np.asarray(periodic.quotient.entities(degree).subset("boundary").mask)
            physical = np.asarray(
                [walls[orbit[entity_rows[int(value)]]] for value in identifiers],
                dtype=np.bool_,
            )
            boundary_mask = facets.boundary & physical
        else:
            boundary_mask = facets.boundary
        boundary = {
            tuple(sorted(int(v) for v in row if v >= 0))
            for row in facets.rows[boundary_mask]
        }
        if target_facet_ids is not None:
            selected = set(target_facet_ids)
            if not selected <= set(facet_entities.values()):
                raise ValueError(
                    "Target fidelity scope names absent global facet identifiers."
                )
            boundary = {
                key
                for key, identifier in facet_entities.items()
                if identifier in selected
            }
    result = []
    for block, element, route in zip(mesh.blocks, elements, routes, strict=True):
        if block.cell_kind not in (
            "interval",
            "triangle",
            "quadrilateral",
            "tetrahedron",
            "prism",
            "pyramid",
            "hexahedron",
        ):
            state.add(
                "coordinate_source_expression",
                "unresolved",
                "cell",
                np.asarray(block.global_ids, dtype=np.int64),
            )
            continue
        topology = reference_cell_topology(block.cell_kind)
        reference = np.asarray(topology.vertices, dtype=np.float64)
        vertices = np.asarray(block.vertices, dtype=np.int64)
        local_routes = np.asarray(route, dtype=np.int64)
        for cell in range(local_routes.shape[0]):
            dofs = np.array(local_routes[cell], dtype=np.int64, copy=False)
            if dofs.ndim != 1:
                raise ValueError(
                    "Fidelity coordinate routes must name a complete cell coefficient vector."
                )
            entity = int(np.asarray(block.global_ids)[cell])
            local = tuple(values[int(dof)] for dof in dofs)
            polynomial = coordinate_expressions(element, local)
            if isinstance(element, CellVertexGeometryElement):
                # Native affine vertex maps have no FEM source basis.
                if block.cell_kind in ("interval", "triangle", "tetrahedron"):
                    polynomial = _fidelity_affine(local)
                else:
                    from ..discretization.fem._reference import lagrange_element

                    polynomial = coordinate_expressions(
                        lagrange_element(block.cell_kind, 1), local
                    )
            if polynomial is None:
                state.add("coordinate_source_expression", "unresolved", "cell", (entity,))
                continue
            if volume:
                faces = [
                    face
                    for face in topology.entities[topology.dimension - 1]
                    if tuple(sorted(int(vertices[cell, v]) for v in face)) in boundary
                ]
            else:
                faces = [tuple(range(len(topology.vertices)))]
            face_entities = (
                [
                    facet_entities[tuple(sorted(int(vertices[cell, v]) for v in face))]
                    for face in faces
                ]
                if target_facet_ids is not None
                else [entity] * len(faces)
            )
            if block.cell_kind == "pyramid":
                # The polynomial source lives on the collapsed cube; side
                # squares collapse continuously to the physical apex.
                cube = np.asarray(
                    (
                        (0, 0, 0),
                        (1, 0, 0),
                        (1, 1, 0),
                        (0, 1, 0),
                        (0, 0, 1),
                        (1, 0, 1),
                        (1, 1, 1),
                        (0, 1, 1),
                    ),
                    dtype=np.float64,
                )
                collapsed = {
                    frozenset((0, 1, 2, 3)): (0, 1, 2, 3),
                    frozenset((0, 1, 4)): (0, 1, 5, 4),
                    frozenset((1, 2, 4)): (1, 2, 6, 5),
                    frozenset((2, 3, 4)): (2, 3, 7, 6),
                    frozenset((0, 3, 4)): (3, 0, 4, 7),
                }
                faces = [collapsed[frozenset(face)] for face in faces]
                reference = cube
            for face, face_entity in zip(faces, face_entities, strict=True):
                simplices = (
                    [face]
                    if len(face) <= 3
                    else [
                        (face[0], face[i], face[i + 1]) for i in range(1, len(face) - 1)
                    ]
                )
                for simplex in simplices:
                    corners = reference[np.asarray(simplex)]
                    arguments = affine_arguments(corners[0], (corners[1:] - corners[0]).T)
                    result.append(
                        _FidelityMap(
                            tuple(compose(value, arguments) for value in polynomial),
                            len(simplex) - 1,
                            face_entity,
                            "facet" if target_facet_ids is not None else "cell",
                        )
                    )
    return result


def _fidelity_node_count(
    coordinates: tuple[_FidelityPolynomial, ...], dimension: int
) -> int:
    from ..discretization._coordinate_enclosure import expression_node_count

    return max(
        (expression_node_count(value, "simplex", dimension) for value in coordinates),
        default=0,
    )


def _fidelity_norm_bound(
    coordinates: tuple[_FidelityPolynomial, ...], dimension: int
) -> float:
    from ..discretization._coordinate_enclosure import expression_bernstein_coefficients

    controls = [
        expression_bernstein_coefficients(value, "simplex", dimension)
        for value in coordinates
    ]
    squared = sum(
        (max(abs(value) for value in bank) ** 2 for bank in controls), Fraction(0)
    )
    if squared == 0:
        return 0.0
    return float(
        np.nextafter(math.sqrt(float(np.nextafter(float(squared), math.inf))), math.inf)
    )


def _fidelity_affine(corners: CoordinateCoefficients, /) -> tuple[Polynomial, ...]:
    """Affine interpolation of actual canonical source coefficients."""
    from ..discretization._coordinate_enclosure import _coordinate_source_rows

    rows = _coordinate_source_rows(corners)
    dimension = len(rows) - 1
    return tuple(
        {
            (0,) * dimension: rows[0][axis],
            **{
                tuple(int(i == j) for j in range(dimension)): rows[i + 1][axis]
                - rows[0][axis]
                for i in range(dimension)
            },
        }
        for axis in range(len(rows[0]))
    )


def _fidelity_proxy(item: _FidelityMap, /) -> tuple[np.ndarray, float, float]:
    """Affine proxy and exact continuous coordinate-minus-proxy enclosure."""
    from ..discretization._coordinate_enclosure import (
        expression_add as add,
        expression_evaluate as evaluate,
        expression_scale as scale,
    )

    reference = (tuple(Fraction(0) for _ in range(item.dimension)),) + tuple(
        tuple(Fraction(int(axis == j)) for j in range(item.dimension))
        for axis in range(item.dimension)
    )
    exact = [
        [evaluate(value, point) for value in item.coordinates] for point in reference
    ]
    corners = np.asarray(
        [[float(value) for value in row] for row in exact], dtype=np.float64
    )
    affine = _fidelity_affine(corners)
    residual = tuple(
        add(value, scale(linear, -1))
        for value, linear in zip(item.coordinates, affine, strict=True)
    )
    deviation = _fidelity_norm_bound(residual, item.dimension)
    rounding = max(
        _fidelity_norm_bound(
            tuple(
                {(0,) * item.dimension: value - Fraction(float(rounded))}
                for value, rounded in zip(row, floating, strict=True)
            ),
            item.dimension,
        )
        for row, floating in zip(exact, corners, strict=True)
    )
    return corners, deviation, rounding


def _chart_fidelity(
    state: _EmbeddingState,
    source: SourceBoundaryQuery,
    maps: list[_FidelityMap],
    tolerance: float,
    limits: MeshCertificateLimits,
    /,
) -> (
    tuple[
        tuple[SourceBoundSemantics, float, float, int],
        tuple[SourceBoundSemantics, float, float, int],
        SourceBoundaryChartCover,
    ]
    | None
):
    """Match a source chart chain to actual mesh maps, not their nodal claims."""
    if (
        not maps
        or not isinstance(source, SourceBoundaryChartQuery)
        or any(item.dimension != 2 for item in maps)
    ):
        return None
    cover = source.boundary_chart_cover(limits.maximum_source_samples)
    if (
        cover.source_id != source.source_id
        or cover.source_revision != source.source_revision
    ):
        raise ValueError(
            "Source chart cover is not bound to the queried source revision."
        )
    state.findings.extend(cover.findings)
    if (
        not cover.complete
        or cover.semantics != "certified"
        or cover.simplices.shape[0] == 0
    ):
        return ("sampled", math.inf, 0.0, 0), ("sampled", math.inf, 0.0, 0), cover
    if max(len(maps), cover.simplices.shape[0]) > limits.maximum_source_samples:
        state.add("source_chart_capacity", "unresolved", "mesh")
        return ("certified", math.inf, 0.0, 0), ("certified", math.inf, 0.0, 0), cover
    if len(maps) > limits.maximum_subdivision_pieces:
        state.add("subdivision_capacity", "unresolved", "mesh")
        return ("certified", math.inf, 0.0, 0), ("certified", math.inf, 0.0, 0), cover
    if any(
        max(3, _fidelity_node_count(item.coordinates, 2)) > limits.maximum_bernstein_nodes
        for item in maps
    ):
        state.add("bernstein_capacity", "unresolved", "mesh")
        return ("certified", math.inf, 0.0, 0), ("certified", math.inf, 0.0, 0), cover
    from ..discretization._coordinate_enclosure import _COORDINATE_BUDGET

    ledger = _COORDINATE_BUDGET.get()
    proxies = []
    for item in maps:
        with ledger.temporary_scope() if ledger is not None else nullcontext():
            proxies.append(_fidelity_proxy(item))
    from .._meshcore import MeshcoreError, MeshcoreStatus
    from ..discretization._coordinate_enclosure import CoordinateEnclosureResourceError
    from ._affine_chart_coverage import prove_affine_chart_chain

    try:
        affine_bound, cover, separation = prove_affine_chart_chain(
            proxies, cover, limits, tolerance
        )
    except (CoordinateEnclosureResourceError, MeshcoreError) as error:
        if isinstance(error, MeshcoreError) and error.status not in (
            MeshcoreStatus.CAPACITY_EXCEEDED,
            MeshcoreStatus.TIMEOUT,
        ):
            raise
        state.add(
            "source_affine_chain_resource_budget",
            "unresolved",
            "mesh",
            resource_error=error
            if isinstance(error, CoordinateEnclosureResourceError)
            else None,
        )
        resources = list(cover.resource_counts)
        if isinstance(error, MeshcoreError):
            if error.work_evidence is not None:
                resources.append(
                    (
                        "affine_chain_refused_total_work",
                        int(error.work_evidence[0]),
                        limits.maximum_work_units,
                    )
                )
            if error.memory_evidence is not None:
                resources.append(
                    (
                        "affine_chain_refused_native_peak_bytes",
                        int(error.memory_evidence[2]),
                        limits.maximum_scratch_bytes,
                    )
                )
        else:
            resources.append(
                (f"affine_chain_refused_{error.resource}", error.completed, error.limit)
            )
        cover = SourceBoundaryChartCover(
            cover.simplices,
            cover.deviation_bounds,
            cover.semantics,
            cover.complete,
            cover.source_id,
            cover.source_revision,
            findings=cover.findings,
            resource_counts=tuple(resources),
        )
        return ("certified", math.inf, 0.0, 0), ("certified", math.inf, 0.0, 0), cover
    if separation is not None:
        state.add(
            "boundary_deviation", "violated", maps[0].entity_kind, (maps[0].entity,)
        )
        return (
            ("certified", math.inf, separation, 0),
            ("certified", math.inf, 0.0, 0),
            cover,
        )
    if affine_bound is not None:
        return (
            ("certified", affine_bound, 0.0, 0),
            ("certified", affine_bound, 0.0, 0),
            cover,
        )
    # Exact corner identity establishes a shared affine parameterization. The
    # residual bound applies everywhere, including bowed interiors.
    lookup: dict[tuple[tuple[float, ...], ...], list[int]] = {}
    for i, (corners, _, _) in enumerate(proxies):
        key = tuple(sorted(tuple(float(v) for v in row) for row in corners))
        lookup.setdefault(key, []).append(i)
    matched = np.zeros((len(maps),), dtype=np.bool_)
    source_bounds = []
    forward = np.full((len(maps),), math.inf, dtype=np.float64)
    comparisons = 0
    for triangle, error in zip(cover.simplices, cover.deviation_bounds, strict=True):
        triangle = np.asarray(triangle, dtype=np.float64)
        key = tuple(sorted(tuple(float(v) for v in row) for row in triangle))
        candidates = lookup.get(key, [])
        if not candidates:
            # A collapsed chart triangle can have repeated corners. Its affine
            # segment is exactly contained in a kept triangle sharing endpoints.
            endpoints = set(key)
            candidates = (
                [
                    i
                    for other, ids in lookup.items()
                    if endpoints.issubset(set(other))
                    for i in ids
                ]
                if len(endpoints) < 3
                else []
            )
        comparisons += len(proxies) if len(set(key)) < 3 else len(candidates)
        if comparisons > limits.maximum_distance_evaluations:
            state.add("distance_capacity", "unresolved", "mesh")
            return ("certified", math.inf, 0.0, 0), ("certified", math.inf, 0.0, 0), cover
        if not candidates:
            return None
        bound = min(
            float(np.nextafter(float(error) + proxies[i][1], math.inf))
            for i in candidates
        )
        source_bounds.append(bound)
        for i in candidates:
            if len(set(key)) == 3:
                matched[i] = True
                forward[i] = min(
                    forward[i],
                    float(np.nextafter(float(error) + proxies[i][1], math.inf)),
                )
    if not np.all(matched):
        return None
    forward_upper = float(np.max(forward, initial=0.0))
    backward_upper = max(source_bounds, default=0.0)
    return (
        ("certified", forward_upper, 0.0, len(maps)),
        ("certified", backward_upper, 0.0, len(source_bounds)),
        cover,
    )


def _split_fidelity_map(item: _FidelityMap, /) -> list[_FidelityMap]:
    from ..discretization._coordinate_enclosure import (
        affine_arguments,
        expression_compose as compose,
    )

    if item.dimension == 1:
        children = (
            np.asarray(((0.0,), (0.5,))),
            np.asarray(((0.5,), (1.0,))),
        )
    else:
        children = tuple(
            np.asarray(row, dtype=np.float64)
            for row in (
                ((0, 0), (0.5, 0), (0, 0.5)),
                ((0.5, 0), (1, 0), (0.5, 0.5)),
                ((0, 0.5), (0.5, 0.5), (0, 1)),
                ((0.5, 0), (0.5, 0.5), (0, 0.5)),
            )
        )
    return [
        _FidelityMap(
            tuple(
                compose(value, affine_arguments(child[0], (child[1:] - child[0]).T))
                for value in item.coordinates
            ),
            item.dimension,
            item.entity,
            item.entity_kind,
        )
        for child in children
    ]


def _mapped_domain_source_maps(
    source: MappedDomainBoundarySource, /
) -> tuple[list[_FidelityMap], np.ndarray]:
    from ..discretization._coordinate_enclosure import (
        expression_bounds as polynomial_bounds,
    )
    from ._mapped_reference_coverage import certify_mapped_reference_source
    from ._mapped_reference_domain import MappedReferenceDomain

    domain = source.domain
    if not isinstance(domain, MappedReferenceDomain):
        raise TypeError("The source no longer contains its nominal mapped domain.")
    physical, embedding, coverage = certify_mapped_reference_source(domain, source.limits)
    if embedding.status != "certified" or coverage.status != "certified":
        raise ValueError(
            "Mapped source point queries require established root embedding and reference coverage."
        )
    state = _EmbeddingState([], ["mapped_source_boundary_cover"])
    maps = _fidelity_maps(state, physical, domain.source_geometry)
    if state.findings or not maps:
        raise ValueError("Mapped source boundary coordinate expressions are unavailable.")
    if len(maps) > source.limits.maximum_subdivision_pieces or any(
        _fidelity_node_count(item.coordinates, item.dimension)
        > source.limits.maximum_bernstein_nodes
        for item in maps
    ):
        raise ValueError(
            "Mapped source boundary enclosures exceed the declared source query limits."
        )
    boxes = np.asarray(
        [
            np.asarray(
                [
                    polynomial_bounds(value, "simplex", item.dimension)
                    for value in item.coordinates
                ],
                dtype=np.float64,
            ).T
            for item in maps
        ],
        dtype=np.float64,
    )
    return maps, boxes


def _mapped_domain_source_samples(
    maps: list[_FidelityMap],
    dimension: int,
    requested_radius: float,
    maximum_samples: int,
    limits: MeshCertificateLimits,
    /,
) -> SourceBoundarySamples:
    from ..discretization._coordinate_enclosure import (
        constant,
        expression_add as add,
        expression_evaluate as evaluate,
    )

    pending = [(item, 0) for item in maps]
    points, radii, errors = [], [], []
    complete = len(maps) <= limits.maximum_subdivision_pieces
    work = 0
    if not complete:
        pending.clear()
    budget = min(
        maximum_samples,
        limits.maximum_source_samples,
        limits.maximum_distance_evaluations,
        limits.maximum_subdivision_pieces,
    )
    while pending:
        if work >= budget:
            complete = False
            break
        item, depth = pending.pop()
        if (
            _fidelity_node_count(item.coordinates, item.dimension)
            > limits.maximum_bernstein_nodes
        ):
            complete = False
            break
        reference = (Fraction(1, item.dimension + 1),) * item.dimension
        exact = tuple(evaluate(value, reference) for value in item.coordinates)
        point = np.asarray([float(value) for value in exact], dtype=np.float64)
        error = _fidelity_norm_bound(
            tuple(
                constant(value - Fraction(float(rounded)), item.dimension)
                for value, rounded in zip(exact, point, strict=True)
            ),
            item.dimension,
        )
        radius = _fidelity_norm_bound(
            tuple(
                add(value, constant(-Fraction(float(rounded)), item.dimension))
                for value, rounded in zip(item.coordinates, point, strict=True)
            ),
            item.dimension,
        )
        work += 1
        if radius > requested_radius and depth < limits.maximum_subdivision_depth:
            children = _split_fidelity_map(item)
            if work + len(pending) + len(children) <= limits.maximum_subdivision_pieces:
                pending.extend((child, depth + 1) for child in children)
                continue
        complete = complete and radius <= requested_radius
        points.append(point)
        radii.append(radius)
        errors.append(error)
    return SourceBoundarySamples(
        np.asarray(points, dtype=np.float64).reshape(-1, dimension),
        np.asarray(radii, dtype=np.float64),
        np.asarray(errors, dtype=np.float64),
        "certified",
        complete=complete,
    )


def _mapped_mesh_cover(
    state: _EmbeddingState,
    source: SourceBoundaryQuery,
    maps: list[_FidelityMap],
    tolerance: float,
    limits: MeshCertificateLimits,
    /,
) -> tuple[
    tuple[SourceBoundSemantics, float, float, int], np.ndarray, np.ndarray, np.ndarray
]:
    from ..discretization._coordinate_enclosure import (
        _COORDINATE_BUDGET,
        constant,
        expression_add as add,
        expression_evaluate as evaluate,
    )

    # Initial maps belong to the enclosing coordinate preparation. Child maps
    # are temporary subdivision owners: keep only the still-pending graphs live
    # and release each one after its last bound/query use.
    pending = [(item, 0, 0) for item in maps]
    centers, radii, errors = [], [], []
    upper, lower, count = 0.0, 0.0, 0
    semantics: SourceBoundSemantics = "certified"
    if len(pending) > limits.maximum_subdivision_pieces:
        state.add(
            "subdivision_capacity",
            "unresolved",
            maps[0].entity_kind if maps else "cell",
            tuple(item.entity for item in maps),
        )
        upper = math.inf
        pending.clear()
    ledger = _COORDINATE_BUDGET.get()
    live_bound = 0
    with (
        nullcontext(None) if ledger is None else ledger.live_storage()
    ) as pending_storage:
        while pending:
            item, depth, item_storage = pending.pop()
            child_records: list[tuple[_FidelityMap, int, int]] = []
            next_live_bound = live_bound - item_storage
            stop = False
            with ledger.temporary_scope() if ledger is not None else nullcontext():
                if count >= min(
                    limits.maximum_source_samples, limits.maximum_distance_evaluations
                ):
                    check = (
                        "mesh_sample_capacity"
                        if count >= limits.maximum_source_samples
                        else "distance_capacity"
                    )
                    state.add(
                        check,
                        "unresolved",
                        item.entity_kind,
                        (
                            item.entity,
                            *(value.entity for value, _, _ in pending),
                        ),
                    )
                    upper = math.inf
                    stop = True
                elif (
                    _fidelity_node_count(item.coordinates, item.dimension)
                    > limits.maximum_bernstein_nodes
                ):
                    state.add(
                        "bernstein_capacity",
                        "unresolved",
                        item.entity_kind,
                        (item.entity,),
                    )
                    upper = math.inf
                else:
                    reference = (Fraction(1, item.dimension + 1),) * item.dimension
                    exact = tuple(
                        evaluate(value, reference) for value in item.coordinates
                    )
                    center = np.asarray(
                        [float(value) for value in exact], dtype=np.float64
                    )
                    error = _fidelity_norm_bound(
                        tuple(
                            constant(value - Fraction(float(rounded)), item.dimension)
                            for value, rounded in zip(exact, center, strict=True)
                        ),
                        item.dimension,
                    )
                    radius = _fidelity_norm_bound(
                        tuple(
                            add(
                                value,
                                constant(-Fraction(float(rounded)), item.dimension),
                            )
                            for value, rounded in zip(
                                item.coordinates, center, strict=True
                            )
                        ),
                        item.dimension,
                    )
                    distance = source.boundary_distance(center[None])
                    if distance.lower.shape != (1,):
                        raise ValueError(
                            "Source distance bounds must match the query count."
                        )
                    count += 1
                    if distance.semantics != "certified":
                        state.add(
                            "source_distance_semantics",
                            "unresolved",
                            item.entity_kind,
                            (item.entity,),
                        )
                        ambient = source.ambient_dimension
                        return (
                            ("sampled", math.inf, 0.0, count),
                            np.asarray([center], dtype=np.float64).reshape(-1, ambient),
                            np.asarray([radius], dtype=np.float64),
                            np.asarray([error], dtype=np.float64),
                        )
                    witnessed = max(
                        0.0,
                        float(np.nextafter(distance.lower[0] - error, -math.inf)),
                    )
                    bound = float(np.nextafter(distance.upper[0] + radius, math.inf))
                    lower = max(lower, witnessed)
                    violated = distance.semantics == "certified" and witnessed > tolerance
                    if violated:
                        state.add(
                            "boundary_deviation",
                            "violated",
                            item.entity_kind,
                            (item.entity,),
                        )
                    needs_split = not violated and (
                        bound > tolerance or radius > tolerance * 0.125
                    )
                    split = False
                    if needs_split and depth < limits.maximum_subdivision_depth:
                        children = _split_fidelity_map(item)
                        if (
                            count + len(pending) + len(children)
                            <= limits.maximum_subdivision_pieces
                        ):
                            child_records = [
                                (
                                    child,
                                    depth + 1,
                                    0
                                    if ledger is None
                                    else ledger.live_object_storage_upper(child),
                                )
                                for child in children
                            ]
                            next_live_bound += sum(
                                storage for _, _, storage in child_records
                            )
                            split = True
                        elif bound > tolerance:
                            state.add(
                                "subdivision_capacity",
                                "unresolved",
                                item.entity_kind,
                                (item.entity,),
                            )
                            upper = math.inf
                            stop = True
                    elif needs_split and bound > tolerance:
                        state.add(
                            "subdivision_depth",
                            "unresolved",
                            item.entity_kind,
                            (item.entity,),
                        )
                        upper = math.inf
                        stop = True
                    if not split:
                        if bound > tolerance and not violated:
                            state.add(
                                "boundary_deviation",
                                "unresolved",
                                item.entity_kind,
                                (item.entity,),
                            )
                        upper = max(upper, bound)
                        centers.append(center)
                        radii.append(radius)
                        errors.append(error)
                if pending_storage is not None:
                    pending_storage.set_bound(next_live_bound)
            live_bound = next_live_bound
            pending.extend(child_records)
            if stop:
                break
    ambient = source.ambient_dimension
    return (
        (semantics, upper, lower, count),
        np.asarray(centers, dtype=np.float64).reshape(-1, ambient),
        np.asarray(radii, dtype=np.float64),
        np.asarray(errors, dtype=np.float64),
    )


def _mapped_source_cover(
    state: _EmbeddingState,
    source: SourceBoundaryQuery,
    centers: np.ndarray,
    radii: np.ndarray,
    errors: np.ndarray,
    forward_upper: float,
    forward_evaluations: int,
    tolerance: float,
    limits: MeshCertificateLimits,
    /,
) -> tuple[SourceBoundSemantics, float, float, int]:
    samples = source.boundary_samples(limits.maximum_source_samples)
    count = samples.points.shape[0]
    if not samples.complete or count == 0:
        state.add("source_sample_capacity", "unresolved", "mesh")
        return samples.semantics, math.inf, 0.0, count
    if (
        count > limits.maximum_source_samples
        or forward_evaluations > limits.maximum_distance_evaluations
    ):
        state.add("distance_capacity", "unresolved", "mesh")
        return samples.semantics, math.inf, 0.0, count
    if not math.isfinite(forward_upper) or centers.shape[0] == 0:
        return samples.semantics, math.inf, 0.0, count
    from .._meshcore import charge_native_geometry_queries
    from ..discretization._coordinate_enclosure import coordinate_enclosure_budget

    class DistanceCapacity(Exception):
        def __init__(self, requested: int) -> None:
            self.requested = requested

    budget = coordinate_enclosure_budget(
        limits.maximum_work_units, limits.maximum_scratch_bytes
    )
    distance_evaluations = forward_evaluations
    owners: list[object] = []

    def work(units: int) -> None:
        budget.reserve(units)
        budget.charge_native_work(units)

    def storage(nbytes: int) -> None:
        budget.reserve(0, nbytes)

    def visit(kind: str, units: int) -> None:
        nonlocal distance_evaluations
        del kind
        requested = distance_evaluations + units
        if requested > limits.maximum_distance_evaluations:
            raise DistanceCapacity(requested)
        budget.reserve(units)
        charge_native_geometry_queries(units)
        budget.charge_native_work(units)
        distance_evaluations = requested

    try:
        with budget.temporary_scope():
            owners.append((centers, radii, errors, samples))
            # One canonical immutable center bank; its owning median builder
            # admits actual construction visits and storage before execution.
            tree = prepare_bvh(
                centers,
                centers,
                dtype=np.float64,
                _charge_work=work,
                _reserve_storage=storage,
            )
            owners.append(tree)
            work(2 * (samples.points.size + centers.size) + 2 * centers.shape[0])
            storage(samples.points.nbytes + centers.nbytes)
            slack = (
                32.0
                * _EPSILON
                * max(
                    float(np.max(np.abs(samples.points))),
                    float(np.max(np.abs(centers))),
                    1.0,
                )
            )
            minimum_error, maximum_radius = float(np.min(errors)), float(np.max(radii))
            dimension, leaf_size = centers.shape[1], tree.leaf_size
            storage(8 * (leaf_size * dimension + 2 * leaf_size + 2 * dimension))
            delta = np.empty((leaf_size, dimension), dtype=np.float64)
            distance = np.empty((leaf_size,), dtype=np.float64)
            weights = np.empty((leaf_size,), dtype=np.float64)
            node_delta = np.empty((dimension,), dtype=np.float64)
            node_other = np.empty((dimension,), dtype=np.float64)
            owners.append((delta, distance, weights, node_delta, node_other))

            def bounds(
                point: np.ndarray,
                node: int,
                lower: np.ndarray,
                upper: np.ndarray,
                out: np.ndarray,
            ) -> None:
                del node
                np.subtract(lower, point, out=node_delta)
                np.subtract(point, upper, out=node_other)
                np.maximum(node_delta, node_other, out=node_delta)
                np.maximum(node_delta, 0.0, out=node_delta)
                box_distance = math.sqrt(float(np.dot(node_delta, node_delta)))
                # Outward arithmetic lowers the box distance before applying
                # the same two objectives as the exhaustive center reduction.
                box_lower = max(0.0, box_distance * (1 - 16 * _EPSILON) - slack)
                out[0] = box_lower * (1 + 16 * _EPSILON) + slack + minimum_error
                out[1] = box_lower * (1 - 16 * _EPSILON) - slack - maximum_radius

            def values(point: np.ndarray, items: np.ndarray, out: np.ndarray) -> None:
                active_delta, active_distance, active_weights = (
                    delta[: items.size],
                    distance[: items.size],
                    weights[: items.size],
                )
                np.take(centers, items, axis=0, out=active_delta)
                np.subtract(active_delta, point, out=active_delta)
                np.square(active_delta, out=active_delta)
                np.sum(active_delta, axis=1, out=active_distance)
                np.sqrt(active_distance, out=active_distance)
                np.multiply(active_distance, 1 + 16 * _EPSILON, out=out[:, 0])
                out[:, 0] += slack
                np.take(errors, items, out=active_weights)
                out[:, 0] += active_weights
                np.multiply(active_distance, 1 - 16 * _EPSILON, out=out[:, 1])
                out[:, 1] -= slack
                np.take(radii, items, out=active_weights)
                out[:, 1] -= active_weights

            minima, _, _, _ = bvh_host_minima(
                tree,
                samples.points,
                objective_count=2,
                node_lower_bounds=bounds,
                item_values=values,
                visit=visit,
                admit_storage=storage,
                retain_owner=owners.append,
            )
            work(8 * count)
            storage(3 * count * 8 + 4 * count)
            high = np.nextafter(minima[:, 0] + samples.covering_radius, math.inf)
            low = np.maximum(minima[:, 1] - samples.point_error, 0.0)
            upper, lower = float(np.max(high)), float(np.max(low))
            violated = (low > tolerance) & (samples.semantics == "certified")
            if np.any(violated):
                state.add(
                    "source_uncovered",
                    "violated",
                    "source_sample",
                    np.flatnonzero(violated),
                )
            unresolved = (high > tolerance) & ~violated
            if np.any(unresolved):
                state.add(
                    "source_uncovered",
                    "unresolved",
                    "source_sample",
                    np.flatnonzero(unresolved),
                )
    except DistanceCapacity as failure:
        state.findings.append(
            MeshCertificateFinding(
                "distance_capacity",
                "unresolved",
                "mesh",
                resource="distance_evaluations",
                requested=(
                    ("limit", limits.maximum_distance_evaluations),
                    ("requested", failure.requested),
                ),
                achieved=(("completed", distance_evaluations),),
                observations=(("forward_distance_evaluations", forward_evaluations),),
            )
        )
        return samples.semantics, math.inf, 0.0, count
    except CoordinateEnclosureResourceError as failure:
        state.add(
            "source_cover_expression_budget",
            "unresolved",
            "mesh",
            resource_error=failure,
            expression_budget=budget,
        )
        return samples.semantics, math.inf, 0.0, count
    return samples.semantics, upper, lower, count


def _certify_scientific_source_fidelity(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    source: SourceBoundaryQuery,
    /,
    *,
    tolerance: float,
    budget: CoordinateEnclosureBudget,
    sample_order: int = 4,
    limits: MeshCertificateLimits | None = None,
    target_facet_ids: ArrayLike | None = None,
) -> SourceFidelityCertificate:
    """Bound the two-sided deviation between the mesh boundary and ``source``.

    Volume meshes compare boundary facets, surface and curve meshes their
    cells. Affine maps adaptively refine a barycentric lattice cover starting
    at ``sample_order``. Mapped maps use exact source-expression Bernstein
    enclosures with bounded subdivision. Both routes honor the declared work
    and subdivision limits rather than relaxing the requested tolerance.
    Certified chart chains additionally provide continuous Taylor comparison bounds.
    """

    if not isinstance(mesh, CellMesh):
        raise TypeError("mesh must be CellMesh.")
    if not isinstance(geometry, CellGeometrySpec):
        raise TypeError("geometry must be CellGeometrySpec.")
    if not isinstance(source, SourceBoundaryQuery):
        raise TypeError("source must satisfy SourceBoundaryQuery.")
    if source.ambient_dimension != mesh.ambient_dimension:
        raise ValueError("source and mesh ambient dimensions differ.")
    if mesh.ambient_dimension - mesh.topological_dimension > 1 or (
        mesh.ambient_dimension == 1
    ):
        raise ValueError("Source fidelity needs a volume, surface or planar curve mesh.")
    tolerance_ = float(tolerance)
    if not math.isfinite(tolerance_) or tolerance_ < 0.0:
        raise ValueError("tolerance must be finite and nonnegative.")
    order = positive_integer(sample_order, "sample_order")
    limits_ = MeshCertificateLimits() if limits is None else limits
    if not isinstance(limits_, MeshCertificateLimits):
        raise TypeError("limits must be MeshCertificateLimits or None.")
    selected_facets: tuple[int, ...] | None = None
    source_scope_id: str | None = None
    if target_facet_ids is not None:
        from ._meshing_domain import MeshingDomainBoundarySource

        if (
            mesh.topological_dimension != mesh.ambient_dimension
            or mesh.topological_dimension != 3
        ):
            raise ValueError(
                "Scoped source-facet fidelity requires a three-dimensional volume mesh."
            )
        if not isinstance(source, MeshingDomainBoundarySource):
            raise TypeError(
                "Scoped source fidelity requires an original MeshingDomainBoundarySource query."
            )
        identifiers = np.asarray(target_facet_ids)
        if identifiers.ndim != 1 or identifiers.size == 0:
            raise ValueError("Target facet scope must be a nonempty identifier vector.")
        if identifiers.dtype.kind not in "iu":
            raise TypeError("Target facet identifiers must be integers.")
        selected_facets = tuple(sorted(int(value) for value in identifiers))
        if len(set(selected_facets)) != len(selected_facets) or selected_facets[0] < 0:
            raise ValueError(
                "Target facet scope requires unique nonnegative identifiers."
            )
        if not set(selected_facets) <= set(
            np.asarray(mesh.entity_set(2).entity_ids).tolist()
        ):
            raise ValueError(
                "Target fidelity scope names absent global facet identifiers."
            )
        source_scope_id = source.source_scope_id
    try:
        scope, _, _ = _coordinate_scope(mesh, geometry, budget)
    except CoordinateEnclosureResourceError as error:
        binding = MeshCertificateBinding(
            mesh,
            geometry,
            "mapped",
            limits_,
            source_id=source.source_id,
            source_revision=source.source_revision,
        )
        state = _EmbeddingState([], ["source_fidelity"])
        state.add(
            "source_expression_resource_budget",
            "unresolved",
            "mesh",
            resource_error=error,
            expression_budget=budget,
        )
        return SourceFidelityCertificate(
            binding,
            tuple(state.findings),
            tolerance=tolerance_,
            semantics=("sampled", "sampled"),
            mesh_to_source=(math.inf, 0.0),
            source_to_mesh=(math.inf, 0.0),
            sample_order=order,
            sample_counts=(0, 0),
            target_facet_ids=selected_facets,
            source_scope_id=source_scope_id,
        )
    binding = MeshCertificateBinding(
        mesh,
        geometry,
        scope,
        limits_,
        source_id=source.source_id,
        source_revision=source.source_revision,
    )
    state = _EmbeddingState([], ["source_fidelity"])
    if mesh.storage is not None:
        state.add("owner_local_source_fidelity_premise", "unresolved", "mesh")
        return SourceFidelityCertificate(
            binding,
            tuple(state.findings),
            tolerance=tolerance_,
            semantics=("sampled", "sampled"),
            mesh_to_source=(math.inf, 0.0),
            source_to_mesh=(math.inf, 0.0),
            sample_order=order,
            sample_counts=(0, 0),
            target_facet_ids=selected_facets,
            source_scope_id=source_scope_id,
        )
    from ._radial_source_fidelity import certify_native_radial_fidelity
    from ._surface_source_support import certify_original_surface_restriction_fidelity

    restrictions = (
        certify_original_surface_restriction_fidelity(
            mesh,
            geometry,
            source,
            binding,
            tolerance_,
            order,
            limits_,
        )
        if selected_facets is None
        else None
    )
    if restrictions is not None:
        return restrictions

    radial = (
        certify_native_radial_fidelity(
            mesh,
            geometry,
            source,
            binding,
            tolerance_,
            order,
            limits_,
        )
        if selected_facets is None
        else None
    )
    if radial is not None:
        return radial
    if isinstance(source, MappedDomainBoundarySource):
        coverage = source.boundary_coverage(mesh, geometry, limits=limits_)
        if coverage.binding.binding_id != binding.binding_id:
            raise ValueError(
                "Recomputed mapped boundary coverage has a different binding."
            )
        complete = coverage.status == "certified"
        bound = 0.0 if complete else math.inf
        findings = (
            ()
            if complete
            else (
                MeshCertificateFinding(
                    "mapped_source_boundary_equality_premise",
                    "unresolved",
                    "mesh",
                ),
            )
        )
        return SourceFidelityCertificate(
            binding,
            findings,
            tolerance=tolerance_,
            semantics=("certified", "certified"),
            mesh_to_source=(bound, 0.0),
            source_to_mesh=(bound, 0.0),
            sample_order=order,
            sample_counts=(0, 0),
            domain_coverage=coverage,
        )
    if isinstance(source, ImplicitProjectionBoundarySource):
        evidence = source.projection_coverage(
            mesh,
            geometry,
            tolerance=tolerance_,
            limits=limits_,
        )
        if evidence.binding.binding_id != binding.binding_id:
            raise ValueError(
                "Recomputed source projection evidence has a different binding."
            )
        findings = evidence.findings
        if evidence.forward_upper > tolerance_ and not any(
            finding.check == "boundary_deviation" for finding in findings
        ):
            findings = (
                *findings,
                MeshCertificateFinding(
                    "boundary_deviation",
                    "unresolved",
                    "mesh",
                ),
            )
        backward_upper = (
            evidence.forward_upper if evidence.status == "certified" else math.inf
        )
        return SourceFidelityCertificate(
            binding,
            findings,
            tolerance=tolerance_,
            semantics=("certified", "certified"),
            mesh_to_source=(evidence.forward_upper, evidence.forward_lower),
            source_to_mesh=(backward_upper, 0.0),
            sample_order=order,
            sample_counts=(0, 0),
            projection_coverage=evidence,
        )
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    chart = None
    maps = []
    if (
        selected_facets is not None
        or scope == "mapped"
        or isinstance(source, SourceBoundaryChartQuery)
    ):
        maps = _fidelity_maps(state, mesh, geometry, target_facet_ids=selected_facets)
        chart = _chart_fidelity(state, source, maps, tolerance_, limits_)
    if chart is not None and (
        (chart[0][1] <= tolerance_ and chart[1][1] <= tolerance_)
        or any(
            finding.check == "source_affine_chain_resource_budget"
            for finding in state.findings
        )
        or any(
            finding.check == "boundary_deviation" and finding.status == "violated"
            for finding in state.findings
        )
    ):
        forward, backward, _ = chart
    elif selected_facets is not None or scope == "mapped":
        query_source: SourceBoundaryQuery = source
        from ._meshing_domain import MeshingDomainBoundarySource

        if isinstance(source, MeshingDomainBoundarySource):
            query_source = source.prepare_boundary_queries()
        forward, centers, radii, errors = _mapped_mesh_cover(
            state, query_source, maps, tolerance_, limits_
        )
        if not maps or any(
            value.check == "coordinate_source_expression" for value in state.findings
        ):
            forward = (forward[0], math.inf, forward[2], forward[3])
        backward = (
            ("sampled", math.inf, 0.0, 0)
            if forward[0] != "certified"
            else _mapped_source_cover(
                state,
                query_source,
                centers,
                radii,
                errors,
                forward[1],
                forward[3],
                tolerance_,
                limits_,
            )
        )
    else:
        items, entities = _fidelity_items(state, mesh, points, limits_)
        forward = _mesh_to_source(
            state, source, points, items, entities, order, tolerance_, limits_
        )
        backward = _source_to_mesh(state, source, points, items, tolerance_, limits_)
    return SourceFidelityCertificate(
        binding,
        tuple(state.findings),
        tolerance=tolerance_,
        semantics=(forward[0], backward[0]),
        mesh_to_source=(forward[1], forward[2]),
        source_to_mesh=(backward[1], backward[2]),
        sample_order=order,
        sample_counts=(forward[3], backward[3]),
        chart_coverage=None if chart is None else chart[2],
        target_facet_ids=selected_facets,
        source_scope_id=source_scope_id,
    )


def certify_source_fidelity(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    source: SourceBoundaryQuery,
    /,
    *,
    tolerance: float,
    sample_order: int = 4,
    limits: MeshCertificateLimits | None = None,
    target_facet_ids: ArrayLike | None = None,
) -> SourceFidelityCertificate:
    """Certify the complete source cover and its numerical coefficient carrier.

    A quotient source defines exact group expressions; independently rounded
    local coefficients are only their execution carrier. The continuous error
    of that carrier is added to both directed source bounds, never to a seam
    equality test or the requested tolerance.

    ``target_facet_ids`` compares only the named global volume facets with the
    selected original ``MeshingDomainBoundarySource`` patches. The exact target
    facet list and occurrence-qualified source scope participate in certificate
    identity; the selected query never silently becomes the whole boundary.
    """
    from ..discretization._coordinate_enclosure import RationalEnclosureError

    limits_ = MeshCertificateLimits() if limits is None else limits
    if not isinstance(limits_, MeshCertificateLimits):
        raise TypeError("limits must be MeshCertificateLimits or None.")
    budget = CoordinateEnclosureBudget(
        limits_.maximum_work_units, limits_.maximum_scratch_bytes
    )
    with budget.activate():
        certificate = _certify_scientific_source_fidelity(
            mesh,
            geometry,
            source,
            budget=budget,
            tolerance=tolerance,
            sample_order=sample_order,
            limits=limits_,
            target_facet_ids=target_facet_ids,
        )
        if geometry.periodic_source is None and geometry.exact_source is None:
            return certificate
        findings = certificate.findings
        try:
            with budget.activate():
                error = geometry.source_execution_error(mesh)
        except CoordinateEnclosureResourceError as failure:
            error = math.inf
            state = _EmbeddingState(list(findings), [])
            state.add(
                "coordinate_source_execution_error_budget",
                "unresolved",
                "mesh",
                resource_error=failure,
                expression_budget=budget,
            )
            findings = tuple(state.findings)
        except RationalEnclosureError:
            error = math.inf
            findings = (
                *findings,
                MeshCertificateFinding(
                    "coordinate_source_execution_error_budget", "unresolved", "mesh", ()
                ),
            )
        if error == 0.0:
            return certificate
        forward = float(np.nextafter(certificate.mesh_to_source_upper + error, math.inf))
        backward = float(np.nextafter(certificate.source_to_mesh_upper + error, math.inf))
        if max(forward, backward) > certificate.tolerance:
            findings = (
                *findings,
                MeshCertificateFinding(
                    "coordinate_source_execution_error", "unresolved", "mesh", ()
                ),
            )
        return SourceFidelityCertificate(
            certificate.binding,
            findings,
            tolerance=certificate.tolerance,
            semantics=(
                certificate.mesh_to_source_semantics,
                certificate.source_to_mesh_semantics,
            ),
            mesh_to_source=(forward, max(0.0, certificate.mesh_to_source_lower - error)),
            source_to_mesh=(backward, max(0.0, certificate.source_to_mesh_lower - error)),
            sample_order=certificate.sample_order,
            sample_counts=(
                certificate.mesh_sample_count,
                certificate.source_sample_count,
            ),
            projection_coverage=certificate.projection_coverage,
            domain_coverage=certificate.domain_coverage,
            chart_coverage=certificate.chart_coverage,
            target_facet_ids=certificate.target_facet_ids,
            source_scope_id=certificate.source_scope_id,
        )


__all__ = [
    "CoordinateMapScope",
    "DomainCoverageCertificate",
    "GlobalEmbeddingCertificate",
    "ImplicitBoundarySource",
    "ImplicitProjectionBoundarySource",
    "ImplicitProjectionCoverageEvidence",
    "MappedBoundaryDegreeEvidence",
    "MappedBoundaryDegreeStatus",
    "MappedDomainBoundarySource",
    "MeshCertificateBinding",
    "MeshCertificateEntityKind",
    "MeshCertificateFinding",
    "MeshCertificateLimits",
    "MeshCertificateStatus",
    "MeshFindingStatus",
    "ParametricCurveBoundarySource",
    "PiecewiseLinearDomain",
    "SourceBoundaryChartCover",
    "SourceBoundaryChartQuery",
    "SourceBoundaryDistance",
    "SourceBoundaryQuery",
    "SourceBoundSemantics",
    "SourceBoundarySamples",
    "SourceFidelityCertificate",
    "certify_domain_coverage",
    "certify_global_embedding",
    "certify_source_fidelity",
]
