from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

from phydrax._bvh import PackedBVH
from phydrax.discretization import CellBlock, CellGeometrySpec, CellMesh
from phydrax.discretization._cell_geometry_validity import certify_cell_geometry_validity
from phydrax.discretization._coordinate_enclosure import Expression
from phydrax.discretization._reference_cell import reference_cell_topology
from phydrax.discretization.fem._reference import lagrange_element
from phydrax.geometry import _global_embedding as embedding
from phydrax.geometry._embedding_cells import EmbeddingCell, prepare_embedding_cells
from phydrax.geometry._stratum_subdivision import StratumSubdivisionCache


def _curved_contacts(*, overlap: bool) -> tuple[CellMesh, CellGeometrySpec]:
    element = lagrange_element("tetrahedron", 2)
    vertices = np.asarray(
        reference_cell_topology("tetrahedron").vertices, dtype=np.float64
    )
    nodes = np.asarray(element.reference_nodes, dtype=np.float64)
    points: list[tuple[float, float, float]] = []
    connectivity: list[tuple[int, ...]] = []
    dofs: list[tuple[float, float, float]] = []
    routes: list[tuple[int, ...]] = []

    def image(point: np.ndarray, group: int, second: bool) -> tuple[float, float, float]:
        x, y, z = point
        if second:
            if group == 0:
                x, y, z = -x, -z, -y  # shared vertex
            elif group == 1:
                x, y = -x, -y  # shared edge
            elif group == 2:
                x, y, z = y, x, -z  # shared face
            elif not overlap:
                x, y, z = -x, -z, -y
        return float(x + x * y / 8 + 4 * group), float(y), float(z)

    for group in range(4):
        vertex_rows: dict[tuple[float, float, float], int] = {}
        for second in (False, True):
            row = []
            for point in vertices:
                physical = image(point, group, second)
                # In the overlap case only the origin has a declared identity;
                # other coincident corners are deliberately distinct vertices.
                if group == 3 and second and overlap and physical != (12.0, 0.0, 0.0):
                    index = len(points)
                    points.append(physical)
                else:
                    index = vertex_rows.get(physical, -1)
                    if index == -1:
                        index = len(points)
                        vertex_rows[physical] = index
                        points.append(physical)
                row.append(index)
            connectivity.append(tuple(row))
            routes.append(tuple(range(len(dofs), len(dofs) + len(nodes))))
            dofs.extend(image(point, group, second) for point in nodes)
    mesh = CellMesh(
        np.asarray(points, dtype=np.float64),
        (CellBlock("curved", "tetrahedron", np.asarray(connectivity, dtype=np.int32)),),
    )
    geometry = CellGeometrySpec(
        {"curved": element},
        {"curved": np.asarray(routes, dtype=np.int32)},
        np.asarray(dofs, dtype=np.float64),
    )
    return mesh, geometry


@pytest.mark.parametrize("overlap", (False, True))
def test_curved_contact_certificates_match_independent_sequential_proofs(
    overlap: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    mesh, geometry = _curved_contacts(overlap=overlap)
    validity = certify_cell_geometry_validity(geometry, mesh=mesh)
    assert validity.all_certified

    # Exercise the exact subdivision route, rather than the optional native
    # whole-cell filter, for several independent contact strata in one proof.
    original_blocks = embedding.bvh_overlap_pair_blocks

    def combined_blocks(
        first: PackedBVH, second: PackedBVH, /, *, include_touching: bool = False
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        blocks = tuple(original_blocks(first, second, include_touching=include_touching))
        if blocks:
            yield (
                np.concatenate(tuple(block[0] for block in blocks)),
                np.concatenate(tuple(block[1] for block in blocks)),
            )

    monkeypatch.setattr(embedding, "bvh_overlap_pair_blocks", combined_blocks)
    monkeypatch.setattr(
        embedding._VertexConeFilter, "prepare", lambda *args, **kwargs: None
    )
    cells = prepare_embedding_cells(mesh, geometry)
    values: list[str] = []
    findings = embedding._nonperiodic_contacts(mesh, geometry, cells, values)
    certificate = embedding.certify_global_embedding(mesh, geometry, validity)

    def sequential(
        requests: tuple[
            tuple[EmbeddingCell, EmbeddingCell, tuple[int, ...], tuple[Expression, ...]],
            ...,
        ],
        cache: StratumSubdivisionCache,
        /,
    ) -> tuple[tuple[int, int] | None, ...]:
        return tuple(
            embedding.subdivided_stratum_separation(
                first, second, shared, residuals, cache
            )
            for first, second, shared, residuals in requests
        )

    monkeypatch.setattr(embedding, "subdivided_stratum_separations", sequential)
    sequential_values: list[str] = []
    sequential_findings = embedding._nonperiodic_contacts(
        mesh, geometry, cells, sequential_values
    )
    reference = embedding.certify_global_embedding(mesh, geometry, validity)
    assert findings == sequential_findings
    assert values == sequential_values
    assert certificate.certificate_id == reference.certificate_id
    assert certificate.findings == reference.findings
    assert certificate.status == reference.status
    if overlap:
        assert certificate.status != "certified"
        assert any(
            finding.check == "mapped_cell_contact" for finding in certificate.findings
        )
    else:
        assert certificate.status == "certified", certificate.findings

    def refused(
        requests: tuple[
            tuple[EmbeddingCell, EmbeddingCell, tuple[int, ...], tuple[Expression, ...]],
            ...,
        ],
        cache: StratumSubdivisionCache,
        /,
    ) -> tuple[tuple[int, int] | None, ...]:
        raise ValueError("injected batch refusal")

    monkeypatch.setattr(embedding, "subdivided_stratum_separations", refused)
    refused_values: list[str] = []
    assert (
        embedding._nonperiodic_contacts(mesh, geometry, cells, refused_values) == findings
    )
    assert refused_values == values
