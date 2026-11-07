from fractions import Fraction

import numpy as np
import pytest

from phydrax._meshcore import meshcore_available
from phydrax.discretization import CellBlock, CellGeometrySpec, CellMesh
from phydrax.discretization._nested_reference_types import (
    ReferenceCellPartition,
    validate_reference_cell_partition,
)
from phydrax.meshing._layer_core_association import layer_source_associations
from phydrax.meshing._layer_source_association_replay import (
    _affine_witness,
    replay_layer_source_associations,
)
from phydrax.meshing._mixed_adaptation import _affine_chart
from phydrax.meshing._volume_entity_containment import reference_entity_containment
from phydrax.meshing.providers._native_sources import NativeLayerCoreSource
from tests.unit.meshing.test_native_layer_polyhedral_core import (
    _bind_layer_source,
    _make_fixture,
)


def _source() -> NativeLayerCoreSource:
    if not meshcore_available():
        pytest.skip("native layer generation requires meshcore")
    fixture = _make_fixture()
    return _bind_layer_source(
        fixture.layers, fixture.complex, fixture.vertex_layer_ids, fixture.charts, None
    )


def test_authored_layer_replay_rebuilds_all_original_strata() -> None:
    source = _source()
    mesh = source.layers.mesh
    replay = replay_layer_source_associations(
        source, mesh, CellGeometrySpec.affine(mesh), None
    )
    original = layer_source_associations(
        source.layers, mesh, source.layer_regions, source.region_ids
    )
    assert len(replay.associations) == len(original) == 3
    assert set(replay.layer_cell_ids) == set(np.asarray(mesh.entity_set(3).entity_ids))
    assert replay.core_cell_ids == ()
    for actual, expected in zip(replay.associations, original, strict=True):
        actual_rows = {
            int(identifier): (entity, role, path)
            for identifier, entity, role, path in zip(
                np.asarray(actual.target_global_ids),
                actual.source_entity_ids,
                actual.source_roles,
                actual.source_occurrence_paths,
                strict=True,
            )
        }
        expected_rows = {
            int(identifier): (entity, role, path)
            for identifier, entity, role, path in zip(
                np.asarray(expected.target_global_ids),
                expected.source_entity_ids,
                expected.source_roles,
                expected.source_occurrence_paths,
                strict=True,
            )
        }
        assert actual_rows == expected_rows
    assert any(
        item.fine_dimension == 3 and item.coarse_dimension == 3
        for item in replay.containment
    )


def test_authored_layer_replay_refuses_changed_physical_map() -> None:
    source = _source()
    mesh = source.layers.mesh
    points = np.asarray(mesh.coordinates).copy()
    points[:, 0] += 0.125
    changed = CellMesh(points, mesh.blocks, vertex_global_ids=mesh.vertex_global_ids)
    with pytest.raises(ValueError, match="coordinate-map equality"):
        replay_layer_source_associations(
            source, changed, CellGeometrySpec.affine(changed), None
        )


def test_exact_prism_partition_replays_new_interior_incidence() -> None:
    half = Fraction(1, 2)
    points = np.asarray(
        [
            (x, y, z)
            for z in (0.0, 0.5, 1.0)
            for x, y in ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
        ],
        dtype=np.float64,
    )
    coarse = CellMesh(
        points[[0, 1, 2, 6, 7, 8]],
        (
            CellBlock(
                "coarse",
                "prism",
                np.asarray(((0, 1, 2, 3, 4, 5),)),
                global_ids=np.asarray((10,)),
            ),
        ),
        vertex_global_ids=np.asarray((0, 1, 2, 6, 7, 8)),
    )
    fine = CellMesh(
        points,
        (
            CellBlock(
                "fine",
                "prism",
                np.asarray(((0, 1, 2, 3, 4, 5), (3, 4, 5, 6, 7, 8))),
                global_ids=np.asarray((20, 21)),
            ),
        ),
        vertex_global_ids=np.arange(9),
    )
    matrix = (
        (Fraction(1), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(0), half),
    )
    witnesses = []
    for child, bottom in ((20, Fraction(0)), (21, half)):
        corners = tuple(
            (Fraction(x), Fraction(y), z)
            for z in (bottom, bottom + half)
            for x, y in ((0, 0), (1, 0), (0, 1))
        )
        witnesses.append(
            _affine_witness(
                child,
                10,
                "prism",
                "prism",
                corners,
                _affine_chart(matrix, (Fraction(0), Fraction(0), bottom)),
            )
        )
    groups = tuple(witnesses)
    validate_reference_cell_partition(ReferenceCellPartition(groups))
    relations = reference_entity_containment(fine, coarse, groups)
    assert any(
        value.fine_dimension == 0 and value.fine_id == 3 and value.coarse_dimension == 1
        for value in relations
    )
    assert any(
        value.fine_dimension == 1 and value.coarse_dimension == 2 for value in relations
    )
    assert {value.fine_cell_id for value in relations if value.fine_dimension == 3} == {
        20,
        21,
    }
    with pytest.raises(ValueError, match="does not cover"):
        validate_reference_cell_partition(ReferenceCellPartition(groups[:1]))
    # Reusing a corner-keyed affine solve must still check each complete chart:
    # this polynomial has the same exact corners but differs in the interior.
    from phydrax.discretization._coordinate_enclosure import (
        add,
        axes,
        constant,
        multiply,
        RationalPolynomial,
        scale,
        subtract,
    )
    from phydrax.discretization._nested_reference_types import (
        freeze_reference_chart_arguments,
    )

    assert groups[0].source_reference_chart_arguments is not None
    chart = tuple(groups[0].source_reference_chart_arguments[0])
    assert not isinstance(chart[2], RationalPolynomial)
    w = axes(3)[2]
    forged = (
        chart[0],
        chart[1],
        add(
            dict(chart[2]),
            scale(multiply(w, subtract(constant(1, 3), w)), Fraction(1, 8)),
        ),
    )
    changed = groups[0]._replace(
        target_cell_ids=np.asarray((21,), dtype=np.int64),
        source_reference_chart_arguments=freeze_reference_chart_arguments((forged,)),
    )
    with pytest.raises(ValueError, match="affine reference map"):
        reference_entity_containment(fine, coarse, (groups[0], changed))
