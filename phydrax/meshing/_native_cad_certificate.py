#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native certification of a CAD solid: chart partition, embedding, fidelity, volume.

A native CAD solid candidate (``cad_mesh_solid``) carries its boundary as
mesh vertex triangles together with the per-face chart realization the
construction produced (:class:`NativeCadSurfaceWitness`). Acceptance proves,
in order and each conditional on the previous:

- V4 ``verify_chart_partition``: the chart triangles of every face are
  positively oriented, pairwise interior-disjoint and close against the trim
  chords (exact);
- V2 ``verify_surface_embedding``: the boundary triangles form an embedded
  closed surface (exact);
- the source shell (:mod:`._native_cad_shell`): the source B-Rep is one
  embedded shell with outward material side, proved from the model once per
  domain and, on the chart export, its trim-to-chord homotopies and sheets;
- V3 ``verify_cad_fidelity``: every chart triangle's two-sided deviation from
  its source face is bounded by the requested tolerance, and each chart
  triangle is the boundary triangle it names with its orientation;
- the tetrahedra are globally embedded (``verify_tetrahedra``) and tile
  exactly the solid the certified surface bounds (``verify_volume``).

The witness is construction data, not a claim: every verdict is recomputed
from it, the mesh and the source domain.
"""

from __future__ import annotations

from enum import IntEnum
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
from ..geometry._embedding_state import MeshCertificateFinding
from ..geometry._mesh_certificates import certificate_status


if TYPE_CHECKING:
    from .._meshcore import CadSurfaceResult
    from ..discretization._cell_geometry import CellGeometrySpec
    from ..discretization._cell_mesh import CellMesh
    from ..geometry._mesh_certificates import MeshCertificateStatus
    from ..geometry._meshing_domain import MeshingDomain
    from ._contracts import ProtectedFeature


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
        self.witness_id = canonical_fingerprint(
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


def _native_findings(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    domain: MeshingDomain,
    witness: NativeCadSurfaceWitness,
    tolerance: float,
    protected_features: bool,
    normal_deviation: bool,
    /,
) -> tuple[tuple[MeshCertificateFinding, ...], np.ndarray | None, str | None]:
    """The first refusing verifier's findings, or none, V3's bounds and the shell proof."""
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
    if any(element.local_dof_count != 4 for element in geometry.elements):
        return (
            (
                MeshCertificateFinding(
                    "cad_fidelity",
                    "unresolved",
                    "mesh",
                    detail="V3 bounds affine boundary triangles; a geometry order 2 CAD "
                    "solid has no certified curved boundary and is constructed "
                    "unproved only (prove=False).",
                ),
            ),
            None,
            None,
        )
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    cells = np.concatenate(
        [np.asarray(block.vertices, dtype=np.int64) for block in mesh.blocks]
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
    embedding = verify_surface_embedding(points, triangles)
    if embedding.status is not VerdictStatus.CERTIFIED:
        return (
            (
                MeshCertificateFinding(
                    "cad_surface_embedding",
                    "violated"
                    if embedding.status is VerdictStatus.VIOLATED
                    else "unresolved",
                    "facet",
                    _ids(embedding.findings["first"]),
                    detail=f"V2 surface embedding {embedding.status.name.lower()} ("
                    + _counts(embedding.findings["kind"], SurfaceFindingKind)
                    + "); entity ids are the findings' first vertex or triangle.",
                ),
            ),
            None,
            None,
        )
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
        tolerance,
        FIDELITY_DEPTH,
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
    tiling = verify_tetrahedra(points, cells)
    if tiling.status is not VerdictStatus.CERTIFIED:
        return (
            (
                MeshCertificateFinding(
                    "cad_volume",
                    "violated"
                    if tiling.status is VerdictStatus.VIOLATED
                    else "unresolved",
                    "cell",
                    _ids(
                        np.asarray(
                            [tiling.first_nonpositive_cell, tiling.first_nonmanifold_cell]
                        )
                    ),
                    detail=f"Tetrahedra are not globally embedded ({tiling.status.name}).",
                ),
            ),
            None,
            None,
        )
    coverage = verify_volume(points, cells, triangles)
    if not coverage.certified:
        return (
            (
                MeshCertificateFinding(
                    "cad_volume",
                    "violated",
                    "cell",
                    _ids(np.asarray([coverage.first_unmatched_cell])),
                    detail="Tetrahedra do not tile the solid the certified surface bounds "
                    f"({coverage.unmatched_boundary_faces} unmatched boundary faces, "
                    f"{coverage.unmatched_surface_triangles} unmatched surface "
                    "triangles).",
                ),
            ),
            None,
            None,
        )
    return (), fidelity.upper, shell_id


# Obligations of the Python CAD certificate no native verifier proves. A
# passing native certificate states them; it does not claim them.
UNPROVED_PREMISES = (
    # Source entity occurrence paths per boundary row: the native witness
    # names model face indices only.
    "source_occurrence_records",
)


class NativeCadCertificate(StrictModule, NonTrainableState):
    """V4, V2, V3 and volume verdicts of one native CAD solid.

    ``maximum_deviation`` is V3's certified upper bound of the boundary's
    two-sided deviation from its source faces (``None`` unless certified);
    ``triangle_deviation`` holds it per chart triangle. ``source_shell`` is
    the identity of the source-shell proof (:mod:`._native_cad_shell`: one
    embedded shell with outward material side), ``None`` unless proved.
    ``unproved_premises`` names the obligations this certificate does not
    prove.
    """

    witness: NativeCadSurfaceWitness
    tolerance: float
    maximum_deviation: float | None
    triangle_deviation: np.ndarray | None
    source_shell: str | None
    findings: tuple[MeshCertificateFinding, ...]
    unproved_premises: tuple[str, ...] = eqx.field(static=True)
    status: MeshCertificateStatus = eqx.field(static=True)
    certificate_id: str

    def __init__(
        self,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        domain: MeshingDomain,
        witness: NativeCadSurfaceWitness,
        tolerance: float,
        protected_features: tuple[ProtectedFeature, ...] | None,
        maximum_normal_deviation: float | None,
        /,
    ) -> None:
        if type(witness) is not NativeCadSurfaceWitness:
            raise TypeError("witness must be NativeCadSurfaceWitness.")
        if witness.mesh_id != mesh.mesh_id:
            raise ValueError("The native CAD surface witness binds another mesh.")
        findings, bounds, shell = _native_findings(
            mesh,
            geometry,
            domain,
            witness,
            tolerance,
            bool(protected_features),
            maximum_normal_deviation is not None,
        )
        self.witness = witness
        self.tolerance = float(tolerance)
        self.maximum_deviation = None if bounds is None else float(np.max(bounds))
        self.triangle_deviation = None if bounds is None else _array(bounds, np.float64)
        self.source_shell = shell
        self.findings = findings
        self.unproved_premises = UNPROVED_PREMISES
        self.status = certificate_status(findings)
        self.certificate_id = canonical_fingerprint(
            {
                "kind": "native-cad-certificate",
                "witness": witness.witness_id,
                "tolerance": self.tolerance,
                "maximum_deviation": self.maximum_deviation,
                "source_shell": shell,
                "findings": tuple(value.finding_id for value in findings),
                "unproved_premises": UNPROVED_PREMISES,
            }
        )


__all__ = ["FIDELITY_DEPTH", "NativeCadCertificate", "NativeCadSurfaceWitness"]
