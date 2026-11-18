import numpy as np
import pytest

from phydrax.discretization._cell_geometry import CellGeometrySpec
from phydrax.discretization._cell_geometry_validity import cell_geometry_id
from phydrax.discretization._cell_mesh import CellBlock, CellMesh
from phydrax.discretization._reference_cell import reference_cell_topology
from phydrax.geometry._domain_coverage import PiecewiseLinearDomain
from phydrax.geometry._piecewise_linear_boundary import (
    PiecewiseLinearBoundarySource,
    PiecewiseLinearDomainBoundaryOwner,
)
from phydrax.geometry._source_boundary_records import SourceBoundaryChartWitness
from phydrax.geometry._source_fidelity import (
    _volume_boundary_facet_lookup,
    validate_source_fidelity_witness,
)


def test_fidelity_validation_mixed_blocks_and_inconsistent_facet_rows() -> None:
    vertices = np.asarray(
        ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, -1)),
        dtype=np.float64,
    )
    mesh = CellMesh(
        vertices,
        (
            CellBlock(
                "first",
                "tetrahedron",
                np.asarray(((0, 1, 2, 3),), dtype=np.int32),
                global_ids=(41,),
            ),
            CellBlock(
                "second",
                "tetrahedron",
                np.asarray(((0, 2, 1, 4),), dtype=np.int32),
                global_ids=(19,),
            ),
        ),
    )
    geometry = CellGeometrySpec.affine(mesh)
    owner = PiecewiseLinearDomainBoundaryOwner(
        PiecewiseLinearDomain(
            vertices[:4],
            np.asarray(((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)), dtype=np.int64),
            np.tile(np.asarray(((0, -1),), dtype=np.int64), (4, 1)),
            ("solid",),
            source_id="mixed-target-source",
        )
    )
    source = PiecewiseLinearBoundarySource(
        owner, 2, entity_ids=np.asarray([0], dtype=np.int64)
    )
    topology = reference_cell_topology("tetrahedron")
    local = next(
        index for index, face in enumerate(topology.entities[2]) if set(face) == {0, 1, 2}
    )
    lookup = _volume_boundary_facet_lookup(mesh)
    face_id, is_boundary = lookup[(41, local)]
    assert not is_boundary and lookup[(19, local)] == (face_id, False)
    target = np.asarray(topology.vertices, dtype=np.float64)[
        np.asarray(topology.entities[2][local], dtype=np.int64)
    ]
    standard = np.asarray(((0, 0), (1, 0), (0, 1)), dtype=np.float64)

    def witness(cells: tuple[int, ...]) -> SourceBoundaryChartWitness:
        count = len(cells)
        return SourceBoundaryChartWitness(
            mesh_id=mesh.mesh_id,
            geometry_id=cell_geometry_id(geometry),
            topology_id=mesh.topology_id,
            source_id=source.source_id,
            source_revision=source.source_revision,
            source_query_id=source.source_query_id,
            chart_dimension=2,
            reference_dimension=3,
            cell_global_ids=np.asarray(cells, dtype=np.int64),
            entity_global_ids=np.full(count, face_id, dtype=np.int64),
            local_entity_indices=np.full(count, local, dtype=np.int32),
            target_reference_vertices=np.tile(target[None], (count, 1, 1)),
            source_chart_ids=source.chart_ids * count,
            source_entity_ids=source.source_entity_keys * count,
            occurrence_paths=source.source_occurrence_paths * count,
            source_parameters=np.tile(standard[None], (count, 1, 1)),
            orientations=np.ones(count, dtype=np.int8),
        )

    for cell in (41, 19):
        validate_source_fidelity_witness(mesh, geometry, source, witness((cell,)))
    with pytest.raises(
        ValueError,
        match="Each volume target facet must use one consistent incident-cell side",
    ):
        validate_source_fidelity_witness(mesh, geometry, source, witness((41, 19)))
    with pytest.raises(
        ValueError, match="Volume chart witness changes an actual target facet ancestry"
    ):
        validate_source_fidelity_witness(mesh, geometry, source, witness((999,)))
    with pytest.raises(
        ValueError, match="Volume chart witness changes an actual target facet ancestry"
    ):
        validate_source_fidelity_witness(mesh, geometry, source, witness((999, 19)))
    with pytest.raises(
        ValueError,
        match="Each volume target facet must use one consistent incident-cell side",
    ):
        validate_source_fidelity_witness(mesh, geometry, source, witness((41, 19, 999)))
