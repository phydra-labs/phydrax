#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native CAD certification: chart partition, embedding, fidelity, labeled volume.

A native CAD solid candidate (``cad_mesh_solid``) carries its boundary as
mesh vertex triangles together with the per-face chart realization the
construction produced (:class:`NativeCadSurfaceWitness`). Acceptance proves,
in order and each conditional on the previous:

- V4 ``verify_chart_partition``: the chart triangles of every face are
  positively oriented, pairwise interior-disjoint and close against the trim
  chords (exact);
- V2 ``verify_surface_embedding``: the exterior and, for multiple solids,
  each region's outward boundary form embedded closed surfaces (exact);
- the source shell (:mod:`._native_cad_shell`): exact source topology and
  material orientation, with embedded trim-to-chord homotopies and sheets;
- V3 ``verify_cad_fidelity``: every chart triangle's two-sided deviation from
  its source face is bounded by the requested tolerance, and each chart
  triangle is the boundary triangle it names with its orientation;
- the tetrahedra are globally embedded (``verify_tetrahedra``) and tile
  exactly the source-declared solid on each side of every certified exterior
  or interface triangle (``verify_volume``).

The quadratic proof is single-material. It uses the affine tetrahedra only as
the reference complex. A validated global embedding of the actual native
nodal P2 map and exact agreement of its shared edge nodes identify the same
complex's curved physical boundary. The V3 bound is on those quadratic
traces, not the affine shadow.

The witness is construction data, not a claim: every verdict is recomputed
from it, the mesh, cell labels and the source domain.
"""

from __future__ import annotations

from enum import IntEnum
from fractions import Fraction
from itertools import combinations
from typing import TYPE_CHECKING

import equinox as eqx
import numpy as np

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._meshcore import (
    CadFidelityChords,
    CadFidelityMesh,
    ChartFailureKind,
    MeshcoreStatus,
    SurfaceFindingKind,
    VerdictStatus,
    verify_cad_fidelity,
    verify_chart_partition,
    verify_surface_embedding,
    verify_tetrahedra,
    verify_volume,
)
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._cell_geometry import CellGeometrySpec, ScalarCoordinateElement
from ..discretization._cell_geometry_validity import (
    affine_tetrahedra,
    AffineTetrahedra,
    cell_geometry_id,
)
from ..discretization._coordinate_enclosure import (
    add,
    axes,
    constant,
    coordinate_execution_basis,
    coordinate_map_expressions,
    coordinate_point_rounding_bound,
    coordinate_source_basis,
    evaluate_expression,
    expression_sequence_equal,
    multiply,
    RationalPolynomial,
    scale,
    sum_polynomials,
)
from ..discretization._reference_cell import reference_cell_topology
from ..geometry._embedding_state import MeshCertificateFinding
from ..geometry._mesh_certificates import _same_tree_content, certificate_status
from ..geometry._rational import fraction_down, fraction_up


if TYPE_CHECKING:
    from .._meshcore import CadModel, CadSurfaceResult
    from ..discretization._cell_mesh import CellMesh
    from ..geometry._global_embedding import (
        GlobalEmbeddingCertificate,
        ValidatedGlobalEmbedding,
    )
    from ..geometry._mesh_certificates import MeshCertificateStatus
    from ..geometry._meshing_domain import MeshingDomain
    from ._contracts import ProtectedFeature
    from ._native_cad_records import NativeCadRecords


# Subdivision depth of the V3 fidelity bounds per chart triangle.
FIDELITY_DEPTH = 8


def _array(value: object, dtype: type[np.generic], /) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


class NativeCadSurfaceWitness(StrictModule, NonTrainableState):
    """The boundary and per-face chart realization of one native CAD solid.

    ``surface_triangles`` index mesh vertices; chart rows are the
    ``cad_mesh_solid`` chart export (pole slivers have ``chart_to_mesh`` -1).
    """

    mesh_id: str
    surface_triangles: np.ndarray
    chart_triangles: np.ndarray
    chart_patches: np.ndarray
    chart_uv: np.ndarray
    chart_to_mesh: np.ndarray
    edge_chord: np.ndarray
    use_patch: np.ndarray
    use_range: np.ndarray
    loop_count: int = eqx.field(static=True)
    chord_use: np.ndarray
    chord_loop: np.ndarray
    chord_uv: np.ndarray
    chord_range: np.ndarray
    witness_id: str

    def __init__(
        self, mesh: CellMesh, surface: CadSurfaceResult, loop_count: int, /
    ) -> None:
        chart = surface.chart
        self.mesh_id = mesh.mesh_id
        self.surface_triangles = _array(surface.triangles, np.int64)
        self.chart_triangles = _array(chart.triangles, np.int64)
        self.chart_patches = _array(chart.patches, np.int32)
        self.chart_uv = _array(chart.uv, np.float64)
        self.chart_to_mesh = _array(chart.chart_to_mesh, np.int64)
        self.edge_chord = _array(chart.edge_chord, np.int64)
        self.use_patch = _array(chart.use_patch, np.int64)
        self.use_range = _array(chart.use_range, np.float64)
        self.loop_count = int(loop_count)
        self.chord_use = _array(chart.chord_use, np.int64)
        self.chord_loop = _array(chart.chord_loop, np.int64)
        self.chord_uv = _array(chart.chord_uv, np.float64)
        self.chord_range = _array(chart.chord_range, np.float64)
        self.witness_id = self._identity()

    def require_binding(self, mesh: CellMesh, /) -> None:
        """Check stored chart operands before using them as an association premise."""
        if self.mesh_id != mesh.mesh_id or self.witness_id != self._identity():
            raise ValueError(
                "Native CAD association chart changed from its retained binding."
            )

    def _identity(self) -> str:
        return canonical_fingerprint(
            {
                "kind": "native-cad-surface-witness",
                "mesh": self.mesh_id,
                "loop_count": self.loop_count,
                "arrays": array_tree_fingerprint(
                    (
                        self.surface_triangles,
                        self.chart_triangles,
                        self.chart_patches,
                        self.chart_uv,
                        self.chart_to_mesh,
                        self.edge_chord,
                        self.use_patch,
                        self.use_range,
                        self.chord_use,
                        self.chord_loop,
                        self.chord_uv,
                        self.chord_range,
                    )
                ),
            }
        )

    def chords(self, /) -> CadFidelityChords:
        return CadFidelityChords(
            np.asarray(self.use_patch),
            np.asarray(self.use_range),
            self.loop_count,
            np.asarray(self.chord_use),
            np.asarray(self.chord_loop),
            np.asarray(self.chord_uv),
            np.asarray(self.chord_range),
        )


def _counts(kinds: np.ndarray, names: type[IntEnum], /) -> str:
    values, counts = np.unique(kinds, return_counts=True)
    return ", ".join(
        f"{names(int(value)).name}: {int(count)}"
        for value, count in zip(values, counts, strict=True)
    )


def _ids(rows: np.ndarray, /) -> tuple[int, ...]:
    return tuple(int(row) for row in np.unique(rows) if row >= 0)


def _surface_embedding_findings(
    points: np.ndarray,
    triangles: np.ndarray,
    boundary: str,
    /,
    *,
    original_rows: np.ndarray | None = None,
) -> tuple[MeshCertificateFinding, ...]:
    embedding = verify_surface_embedding(points, triangles)
    if embedding.status is VerdictStatus.CERTIFIED:
        return ()
    status = "violated" if embedding.status is VerdictStatus.VIOLATED else "unresolved"
    findings = embedding.findings
    kinds = findings["kind"]
    by_facet = (kinds == SurfaceFindingKind.DEGENERATE) | (
        kinds == SurfaceFindingKind.CONTACT
    )
    triangle_rows = _ids(
        np.concatenate((findings["first"][by_facet], findings["second"][by_facet]))
    )
    if original_rows is not None and triangle_rows:
        triangle_rows = tuple(int(row) for row in original_rows[list(triangle_rows)])
    vertex_rows = _ids(
        np.concatenate((findings["first"][~by_facet], findings["second"][~by_facet]))
    )
    detail = (
        f"V2 {boundary} embedding {embedding.status.name.lower()} ("
        + _counts(kinds, SurfaceFindingKind)
        + ")."
    )
    evidence: list[MeshCertificateFinding] = []
    if triangle_rows:
        evidence.append(
            MeshCertificateFinding(
                "cad_surface_embedding",
                status,
                "facet",
                triangle_rows,
                detail=detail + " Entity IDs index the full physical triangle bank.",
            )
        )
    if vertex_rows:
        evidence.append(
            MeshCertificateFinding(
                "cad_surface_embedding",
                status,
                "vertex",
                vertex_rows,
                detail=detail + " Entity IDs index the mesh coordinate bank.",
            )
        )
    if not evidence:
        evidence.append(
            MeshCertificateFinding("cad_surface_embedding", status, "mesh", detail=detail)
        )
    return tuple(evidence)


def _volume_finding(
    points: np.ndarray,
    cells: np.ndarray,
    triangles: np.ndarray,
    cell_regions: np.ndarray | None,
    surface_regions: np.ndarray | None,
    /,
) -> MeshCertificateFinding | None:
    tiling = verify_tetrahedra(points, cells)
    if tiling.status is not VerdictStatus.CERTIFIED:
        return MeshCertificateFinding(
            "cad_volume",
            "violated" if tiling.status is VerdictStatus.VIOLATED else "unresolved",
            "cell",
            _ids(
                np.asarray([tiling.first_nonpositive_cell, tiling.first_nonmanifold_cell])
            ),
            detail=f"Tetrahedra are not globally embedded ({tiling.status.name}).",
        )
    coverage = verify_volume(
        points,
        cells,
        triangles,
        cell_regions=cell_regions,
        surface_regions=surface_regions,
    )
    if coverage.certified:
        return None
    cell_rows = [coverage.first_unmatched_cell]
    mismatch = ""
    if surface_regions is not None:
        cell_rows.append(coverage.first_region_mismatch_cell)
        mismatch = f"; {coverage.region_mismatches} region mismatches"
    return MeshCertificateFinding(
        "cad_volume",
        "violated",
        "cell",
        _ids(np.asarray(cell_rows)),
        detail="Tetrahedra do not tile the solid the certified surface bounds "
        f"({coverage.unmatched_boundary_faces} unmatched boundary faces, "
        f"{coverage.unmatched_surface_triangles} unmatched surface triangles{mismatch}).",
    )


def _restricted_affine_rounding(
    mesh: CellMesh, geometry: CellGeometrySpec, points: np.ndarray, /
) -> Fraction | None:
    """Prove affine cell maps share one exact point per stored vertex."""
    elements, routes, coefficients = geometry.resolve(mesh)
    bank = np.asarray(coefficients, dtype=np.float64)
    corners: dict[int, tuple[Fraction, ...]] = {}
    for block, element, route in zip(mesh.blocks, elements, routes, strict=True):
        if not isinstance(element, ScalarCoordinateElement):
            return None
        reference = tuple(
            tuple(Fraction(float(value)) for value in point)
            for point in reference_cell_topology(block.cell_kind).vertices
        )
        for row, (cell, dofs) in enumerate(
            zip(np.asarray(block.vertices), np.asarray(route), strict=True)
        ):
            expressions = coordinate_map_expressions(element, bank[dofs], cell_row=row)
            if any(
                isinstance(expression, RationalPolynomial)
                or any(sum(power) > 1 for power in expression)
                for expression in expressions
            ):
                return None
            for vertex, corner in zip(cell, reference, strict=True):
                exact = tuple(evaluate_expression(value, corner) for value in expressions)
                if not all(isinstance(value, Fraction) for value in exact):
                    return None
                rational = tuple(value for value in exact if isinstance(value, Fraction))
                if corners.setdefault(int(vertex), rational) != rational:
                    return None
    if len(corners) != len(points):
        return None
    ordered = tuple(corners[row] for row in range(len(points)))
    if not np.array_equal(np.asarray(ordered, dtype=np.float64), points):
        return None
    return coordinate_point_rounding_bound(ordered, points)


def _cad_affine_carrier(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    embedding: GlobalEmbeddingCertificate | None,
    /,
) -> tuple[AffineTetrahedra, Fraction] | None:
    carrier = affine_tetrahedra(mesh, geometry)
    if carrier is not None:
        return carrier, Fraction(0)
    if (
        geometry.restriction_source is None
        or geometry.exact_coordinate_bank is not None
        or embedding is None
        or embedding.status != "certified"
    ):
        return None
    embedding.require_current(mesh, geometry)
    carrier = affine_tetrahedra(mesh, CellGeometrySpec.affine(mesh))
    if carrier is None:
        return None
    rounding = _restricted_affine_rounding(mesh, geometry, carrier.points)
    if rounding is None:
        return None
    # The exact map and its rounded carrier realize the same oriented cell
    # complex. Embedding of the exact map is proved independently; the
    # carrier's labeled incidence and surface-link checks then apply to it.
    return carrier, rounding


_TETRAHEDRON_EDGES = tuple(combinations(range(4), 2))
_TRIANGLE_EDGES = ((0, 1), (1, 2), (2, 0))


def _edge_key(a: int, b: int, /) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def _quadratic_dofs(
    element: object, cell_kind: str, /
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Authenticate both the effective exact basis and its execution owner."""
    from ..discretization.fem._high_order import SimplexNodalFamily
    from ..discretization.fem._reference import FiniteElementSpec, lagrange_element

    native = lagrange_element(cell_kind, 2)
    if type(element) is not FiniteElementSpec:
        raise ValueError("Native quadratic CAD proof requires a finite element basis.")
    tabulate = element.tabulator
    # Equinox's bound-method record is not a Python MethodType. Match the
    # actual canonical wrapper, its callable identity, and its complete tree.
    if (
        type(tabulate) is not type(native.tabulator)
        or getattr(tabulate, "__func__", None) is not SimplexNodalFamily.tabulate
        or type(getattr(tabulate, "__self__", None)) is not SimplexNodalFamily
        or not _same_tree_content(element, native)
    ):
        raise ValueError(
            "Native quadratic CAD proof requires the native P2 simplex Lagrange basis."
        )

    # Check the full polynomial, not probes at the ten nodes. In particular,
    # basis cache keys and tabulator IDs do not authenticate the executable
    # coefficients, which are checked through the bound owner above.
    vertices = reference_cell_topology(cell_kind).vertices
    row = {tuple(point): i for i, point in enumerate(np.asarray(native.reference_nodes))}
    corner_dofs = tuple(row[tuple(point)] for point in vertices)
    edge_dofs = tuple(
        row[tuple((a + b) / 2 for a, b in zip(vertices[i], vertices[j], strict=True))]
        for i, j in _TETRAHEDRON_EDGES
    )
    variables = axes(3)
    one = constant(1, 3)
    barycentric = (add(one, scale(sum_polynomials(variables, 3), -1)), *variables)
    expected = [constant(0, 3) for _ in range(10)]
    for dof, value in zip(corner_dofs, barycentric, strict=True):
        expected[dof] = multiply(value, add(scale(value, 2), scale(one, -1)))
    for dof, (i, j) in zip(edge_dofs, _TETRAHEDRON_EDGES, strict=True):
        expected[dof] = scale(multiply(barycentric[i], barycentric[j]), 4)
    if not (
        expression_sequence_equal(coordinate_source_basis(element), expected)
        and expression_sequence_equal(coordinate_execution_basis(element), expected)
    ):
        raise ValueError(
            "Native quadratic CAD proof requires the exact native P2 Lagrange basis."
        )
    return corner_dofs, edge_dofs


def _quadratic_cad_carrier(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    domain: MeshingDomain,
    witness: NativeCadSurfaceWitness,
    cell_regions: np.ndarray,
    embedding: GlobalEmbeddingCertificate | ValidatedGlobalEmbedding | None,
    /,
) -> tuple[AffineTetrahedra, np.ndarray, dict[tuple[int, int], int]]:
    """Bind the actual continuous P2 map to the affine cell complex."""
    from ..geometry._global_embedding import ValidatedGlobalEmbedding

    if len(domain.regions) != 1:
        raise ValueError("Native quadratic CAD proof requires exactly one material.")
    if (
        type(geometry) is not CellGeometrySpec
        or geometry.source_map is not None
        or geometry.source_mesh_id is not None
        or geometry.source_topology_id is not None
        or geometry.source_map_id is not None
        or geometry.exact_coordinate_bank is not None
        or geometry.restriction_source is not None
        or mesh.periodic_topology is not None
    ):
        raise ValueError(
            "Native quadratic CAD proof requires a nonperiodic nodal coordinate "
            "map without a source bank or restriction."
        )
    embedding_error = (
        "Native quadratic CAD proof requires a certified embedding of the "
        "current coordinate map."
    )
    if type(embedding) is not ValidatedGlobalEmbedding:
        raise ValueError(embedding_error)
    try:
        certificate = embedding.require(mesh, geometry, embedding.validity)
    except ValueError as error:
        raise ValueError(embedding_error) from error
    if certificate.status != "certified" or not certificate.validity.all_certified:
        raise ValueError(embedding_error)
    witness.require_binding(mesh)
    domain.require_current()
    carrier = affine_tetrahedra(mesh, CellGeometrySpec.affine(mesh))
    if carrier is None or not len(carrier.cells):
        raise ValueError("Native quadratic CAD proof requires a tetrahedral carrier.")
    regions = np.asarray(cell_regions)
    if (
        regions.shape != (len(carrier.cells),)
        or regions.dtype.kind not in "iu"
        or np.any(regions)
    ):
        raise ValueError(
            "Native quadratic CAD proof requires all cells in the one material."
        )

    if geometry.block_names != tuple(sorted(block.name for block in mesh.blocks)):
        raise ValueError("Native quadratic geometry blocks must align with the carrier.")
    if any(np.ndim(route) != 2 for route in geometry.geometry_dofs):
        raise ValueError("Native quadratic coordinate routes must be matrices.")
    elements, routes, coordinates = geometry.resolve(mesh)
    bank = np.asarray(coordinates)
    if (
        bank.dtype != np.float64
        or bank.ndim != 2
        or bank.shape[1] != 3
        or not np.all(np.isfinite(bank))
    ):
        raise ValueError(
            "Native quadratic coordinate bank must contain finite binary64 points."
        )
    bank_bits = bank.view(np.uint64)
    point_bits = carrier.points.view(np.uint64)
    midpoint_rows: dict[tuple[int, int], int] = {}
    for block, element, route in zip(mesh.blocks, elements, routes, strict=True):
        corner_dofs, edge_dofs = _quadratic_dofs(element, block.cell_kind)
        dofs = np.asarray(route)
        cells = np.asarray(block.vertices)
        if (
            dofs.dtype != np.int32
            or dofs.shape != (block.cell_count, 10)
            or np.any(dofs < 0)
            or np.any(dofs >= len(bank))
            or np.any(cells < 0)
            or np.any(cells >= len(carrier.points))
        ):
            raise ValueError(
                "Native quadratic coordinate routes must index the current bank."
            )
        if not np.array_equal(bank_bits[dofs[:, corner_dofs]], point_bits[cells]):
            raise ValueError(
                "Native quadratic corner coordinates must equal the carrier vertices."
            )
        for cell, cell_dofs in zip(cells.tolist(), dofs.tolist(), strict=True):
            for dof, (i, j) in zip(edge_dofs, _TETRAHEDRON_EDGES, strict=True):
                key = _edge_key(cell[i], cell[j])
                index = cell_dofs[dof]
                prior = midpoint_rows.setdefault(key, index)
                if prior != index and not np.array_equal(
                    bank_bits[prior], bank_bits[index]
                ):
                    raise ValueError(
                        "Native quadratic cells disagree on a shared edge midpoint node."
                    )
    # All lengths, domains and scalar types were checked on the retained
    # arrays, before the constructor (which normalizes them) is allowed to run.
    rebuilt = CellGeometrySpec(
        dict(zip(geometry.block_names, geometry.elements, strict=True)),
        dict(zip(geometry.block_names, geometry.geometry_dofs, strict=True)),
        bank,
    )
    if rebuilt.geometry_layout_id != geometry.geometry_layout_id:
        raise ValueError(
            "Native quadratic coordinate layout does not reproduce its binding."
        )
    return carrier, bank, midpoint_rows


def _quadratic_chart_edge_nodes(
    mesh: CellMesh,
    domain: MeshingDomain,
    model: CadModel,
    witness: NativeCadSurfaceWitness,
    points: np.ndarray,
    bank: np.ndarray,
    midpoint_rows: dict[tuple[int, int], int],
    /,
) -> np.ndarray:
    """Restrict the P2 map to each witnessed boundary facet or pole edge."""
    from ._native_cad_records import _chart_face_rows

    triangles, _ = _chart_face_rows(mesh, witness, domain, oriented=True)
    chart = np.asarray(witness.chart_triangles)
    mapping = np.asarray(witness.chart_to_mesh)
    edges = np.asarray(witness.edge_chord)
    uses = np.asarray(witness.chord_use)
    if (
        any(value.dtype != np.int64 for value in (chart, mapping, edges, uses))
        or uses.ndim != 1
        or np.any(uses < 0)
        or np.any(uses >= model.use_kinds.size)
        or np.any(edges < -1)
        or np.any(edges >= uses.size)
    ):
        raise ValueError(
            "Native quadratic chart edges must route to declared trim chords."
        )
    collapsed = np.zeros(chart.shape, dtype=np.bool_)
    trim = edges >= 0
    collapsed[trim] = model.use_kinds[uses[edges[trim]]] == 1
    if np.any(np.count_nonzero(collapsed, axis=1) != (mapping == -1)):
        raise ValueError(
            "Native quadratic slivers require exactly one collapsed trim side."
        )
    boundary_edges = {
        _edge_key(tri[i], tri[j])
        for tri in triangles.tolist()
        for i, j in _TRIANGLE_EDGES
    }
    nodes = np.empty((len(chart), 3, 3), dtype=np.float64)
    for row, tri in enumerate(chart.tolist()):
        for edge, (i, j) in enumerate(_TRIANGLE_EDGES):
            a, b = tri[i], tri[j]
            if (a == b) != collapsed[row, edge]:
                raise ValueError(
                    "Native quadratic collapsed side must repeat its carrier vertex."
                )
            if a == b:
                nodes[row, edge] = points[a]
            else:
                key = _edge_key(a, b)
                dof = midpoint_rows.get(key)
                if key not in boundary_edges or dof is None:
                    raise ValueError(
                        "Native quadratic chart edge is not on the carrier boundary."
                    )
                nodes[row, edge] = bank[dof]
    return nodes


def _quadratic_refusal(detail: str, /) -> tuple[MeshCertificateFinding, ...]:
    return (MeshCertificateFinding("cad_fidelity", "unresolved", "mesh", detail=detail),)


def _native_findings(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    domain: MeshingDomain,
    witness: NativeCadSurfaceWitness,
    cell_regions: np.ndarray,
    tolerance: float,
    protected_features: bool,
    normal_deviation: bool,
    /,
    *,
    embedding: GlobalEmbeddingCertificate | ValidatedGlobalEmbedding | None = None,
    quadratic: bool = False,
) -> tuple[tuple[MeshCertificateFinding, ...], np.ndarray | None, str | None]:
    """Return the first refusal, or the V3 upper bounds and source-shell identity.

    Keep the verifier order explicit: each stage consumes the same carrier,
    chart and source, and runs only after the preceding premise is proved.
    Most lines describe failure evidence, not alternate execution paths.
    """
    from ..geometry._global_embedding import ValidatedGlobalEmbedding
    from ._native_cad_shell import certify_native_trim_references, source_shell
    from .providers._native_cad_bundle import cad_geometry_bundle, coefficient_bounds

    # No native owner proves feature evidence or a normal-deviation bound yet.
    requested = (
        *(("protected features",) if protected_features else ()),
        *(("maximum_normal_deviation",) if normal_deviation else ()),
    )
    if requested:
        return (
            (
                MeshCertificateFinding(
                    "cad_native_unsupported_request",
                    "unresolved",
                    "mesh",
                    detail="The native CAD certificate has no proof of "
                    + " or ".join(requested)
                    + "; the request is refused rather than dropped.",
                ),
            ),
            None,
            None,
        )
    if not quadratic and any(
        element.local_dof_count != 4 for element in geometry.elements
    ):
        return (
            (
                MeshCertificateFinding(
                    "cad_fidelity",
                    "unresolved",
                    "mesh",
                    detail="V3 requires affine boundary triangles or the admitted "
                    "native quadratic trace proof.",
                ),
            ),
            None,
            None,
        )
    quadratic_data = None
    if quadratic:
        try:
            quadratic_data = _quadratic_cad_carrier(
                mesh, geometry, domain, witness, cell_regions, embedding
            )
        except ValueError as error:
            return _quadratic_refusal(str(error)), None, None
        tetrahedra = quadratic_data[0]
        rounding = Fraction(0)
    else:
        affine_embedding = (
            embedding.certificate
            if isinstance(embedding, ValidatedGlobalEmbedding)
            else embedding
        )
        carrier = _cad_affine_carrier(mesh, geometry, affine_embedding)
        if carrier is None:
            return (
                (
                    MeshCertificateFinding(
                        "cad_fidelity",
                        "unresolved",
                        "mesh",
                        detail="Native CAD proof requires the affine tetrahedral map "
                        "interpolating the mesh vertices.",
                    ),
                ),
                None,
                None,
            )
        tetrahedra, rounding = carrier
    points, cells = tetrahedra.points, tetrahedra.cells
    # Matching barycentric weights move every point of a chart facet by at
    # most the largest vertex displacement, in either direction.
    fidelity_tolerance = (
        tolerance if not rounding else fraction_down(Fraction(tolerance) - rounding)
    )
    if fidelity_tolerance <= 0.0:
        return (
            (
                MeshCertificateFinding(
                    "cad_fidelity",
                    "unresolved",
                    "mesh",
                    detail="The exact affine map's carrier rounding bound "
                    f"{fraction_up(rounding):.6g} exhausts the CAD tolerance.",
                ),
            ),
            None,
            None,
        )
    bundle = cad_geometry_bundle(domain)
    model = bundle.model
    chords = witness.chords()
    partition = verify_chart_partition(
        model.surface_nodes.size,
        np.asarray(witness.chart_patches),
        np.asarray(witness.chart_uv),
        chords,
    )
    if partition.size:
        return (
            (
                MeshCertificateFinding(
                    "cad_chart_partition",
                    "violated",
                    "source_cell",
                    _ids(partition["row"]),
                    detail="V4 chart partition failures ("
                    + _counts(partition["kind"], ChartFailureKind)
                    + "); entity ids are chart triangle rows.",
                ),
            ),
            None,
            None,
        )
    triangles = np.asarray(witness.surface_triangles)
    surface_regions = None
    if len(domain.regions) > 1:
        from ._native_cad_records import _native_cad_regional_surface

        regional = _native_cad_regional_surface(mesh, domain, witness)
        surface_regions = regional.regions
        for region in range(-1, len(domain.regions)):
            original, outward = regional.outward(None if region < 0 else region)
            boundary = "union exterior" if region < 0 else f"region {region} surface"
            findings = _surface_embedding_findings(
                points, outward, boundary, original_rows=original
            )
            if findings:
                return findings, None, None
    else:
        findings = _surface_embedding_findings(points, triangles, "surface")
        if findings:
            return findings, None, None
    shell = source_shell(domain)
    if shell.refusal is None:
        sheets, refusal = certify_native_trim_references(domain, model, shell, witness)
    else:
        sheets, refusal = (), shell.refusal
    if refusal is not None:
        return (
            (
                MeshCertificateFinding(
                    "cad_source_shell",
                    "unresolved",
                    "source_entity",
                    detail="Rooted CAD source shell embedding or material orientation "
                    f"remains unresolved (refused at: {refusal}).",
                ),
            ),
            None,
            None,
        )
    shell_id = canonical_fingerprint(
        {
            "kind": "native-cad-source-shell-proof",
            "model_part": shell.certificate_id,
            "witness": witness.witness_id,
            "sheets": sheets,
        }
    )
    triangle_edge_nodes = None
    if quadratic_data is not None:
        # Exact P1 coverage identifies the reference complex's boundary, not
        # the validity or embedding of its curved image. Those were proved
        # on the actual P2 map; every routed trace is its exact restriction.
        finding = _volume_finding(points, cells, triangles, None, None)
        if finding is not None:
            return (finding,), None, None
        _, bank, midpoint_rows = quadratic_data
        try:
            triangle_edge_nodes = _quadratic_chart_edge_nodes(
                mesh, domain, model, witness, points, bank, midpoint_rows
            )
        except ValueError as error:
            return _quadratic_refusal(str(error)), None, None
    lower, upper = coefficient_bounds(bundle, domain)
    fidelity = verify_cad_fidelity(
        model,
        lower,
        upper,
        CadFidelityMesh(
            points,
            np.asarray(witness.chart_triangles),
            np.asarray(witness.chart_patches),
            np.asarray(witness.chart_uv),
            triangles,
            np.asarray(witness.chart_to_mesh),
            np.asarray(witness.edge_chord),
        ),
        chords,
        fidelity_tolerance,
        FIDELITY_DEPTH,
        triangle_edge_nodes=triangle_edge_nodes,
    )
    unresolved = np.flatnonzero(fidelity.item_status != MeshcoreStatus.OK)
    if (
        fidelity.status is not MeshcoreStatus.OK
        or unresolved.size
        or fidelity.findings.size
    ):
        reasons = ", ".join(
            f"reason {int(value)}: {int(count)}"
            for value, count in zip(
                *np.unique(fidelity.reasons[unresolved], return_counts=True),
                strict=True,
            )
        )
        return (
            (
                MeshCertificateFinding(
                    "cad_fidelity",
                    "unresolved",
                    "source_cell",
                    _ids(np.concatenate((unresolved, fidelity.findings["triangle"]))),
                    detail=f"V3: {unresolved.size} of {fidelity.item_status.size} chart "
                    f"triangles unresolved at tolerance {tolerance:.6g} ({reasons}); "
                    f"{fidelity.findings.size} association findings; maximum upper "
                    f"bound {fidelity.maximum_upper:.6g}; entity ids are chart "
                    "triangle rows.",
                ),
            ),
            None,
            None,
        )
    if quadratic_data is None:
        finding = _volume_finding(
            points,
            cells,
            triangles,
            None if surface_regions is None else cell_regions,
            surface_regions,
        )
        if finding is not None:
            return (finding,), None, None
    bounds = fidelity.upper
    if rounding:
        bounds = np.minimum(
            tolerance,
            np.nextafter(bounds + fraction_up(rounding), np.inf),
        )
    return (), bounds, shell_id


class NativeCadCertificate(StrictModule, NonTrainableState):
    """V4, V2, V3 and labeled volume verdicts for native CAD solids.

    ``maximum_deviation`` is V3's certified upper bound of the boundary's
    two-sided deviation from its source faces (``None`` unless certified);
    ``triangle_deviation`` holds it per chart triangle. ``source_shell`` is
    the identity of the embedded source boundary and material proof
    (:mod:`._native_cad_shell`), ``None`` unless proved.
    ``records_id`` binds the native occurrence and curve records
    (:mod:`._native_cad_records`) the candidate's associations must match.
    ``geometry_id`` binds the exact coordinate map used by the proof.
    """

    witness: NativeCadSurfaceWitness
    tolerance: float
    maximum_deviation: float | None
    triangle_deviation: np.ndarray | None
    source_shell: str | None
    findings: tuple[MeshCertificateFinding, ...]
    records_id: str
    geometry_id: str
    status: MeshCertificateStatus = eqx.field(static=True)
    certificate_id: str

    def __init__(
        self,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        domain: MeshingDomain,
        witness: NativeCadSurfaceWitness,
        records: NativeCadRecords,
        cell_regions: np.ndarray,
        tolerance: float,
        protected_features: tuple[ProtectedFeature, ...] | None,
        maximum_normal_deviation: float | None,
        /,
        *,
        embedding: GlobalEmbeddingCertificate | ValidatedGlobalEmbedding | None = None,
        quadratic: bool = False,
    ) -> None:
        if type(witness) is not NativeCadSurfaceWitness:
            raise TypeError("witness must be NativeCadSurfaceWitness.")
        from ._native_cad_records import NativeCadRecords

        if type(records) is not NativeCadRecords:
            raise TypeError("records must be NativeCadRecords.")
        records.require_premises(mesh, domain, cell_regions, witness, tolerance)
        findings, bounds, shell = _native_findings(
            mesh,
            geometry,
            domain,
            witness,
            cell_regions,
            tolerance,
            bool(protected_features),
            maximum_normal_deviation is not None,
            embedding=embedding,
            quadratic=quadratic,
        )
        self.witness = witness
        self.tolerance = float(tolerance)
        self.maximum_deviation = None if bounds is None else float(np.max(bounds))
        self.triangle_deviation = None if bounds is None else _array(bounds, np.float64)
        self.source_shell = shell
        self.findings = findings
        self.records_id = records.records_id
        self.geometry_id = cell_geometry_id(geometry)
        self.status = certificate_status(findings)
        self.certificate_id = self._identity()

    def require_binding(
        self,
        witness: NativeCadSurfaceWitness,
        records: NativeCadRecords,
        geometry_id: str,
        /,
    ) -> None:
        """Require the proof to bind the request's chart, materials and map."""
        if (
            not _same_tree_content(self.witness, witness)
            or self.records_id != records.records_id
            or self.geometry_id != geometry_id
            or self.tolerance != records.tolerance
        ):
            raise ValueError(
                "Native CAD certificate binds another witness, records or geometry."
            )
        if (
            self.status != certificate_status(self.findings)
            or self.certificate_id != self._identity()
        ):
            raise ValueError("Native CAD certificate evidence differs from its identity.")

    def _identity(self) -> str:
        bounds = self.triangle_deviation
        return canonical_fingerprint(
            {
                "kind": "native-cad-certificate",
                "witness": self.witness.witness_id,
                "tolerance": self.tolerance,
                "maximum_deviation": self.maximum_deviation,
                "triangle_deviation": None
                if bounds is None
                else array_tree_fingerprint(bounds),
                "source_shell": self.source_shell,
                "findings": tuple(value.finding_id for value in self.findings),
                "records": self.records_id,
                "geometry": self.geometry_id,
            }
        )


__all__ = ["FIDELITY_DEPTH", "NativeCadCertificate", "NativeCadSurfaceWitness"]
