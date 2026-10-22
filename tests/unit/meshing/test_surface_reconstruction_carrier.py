# Copyright © 2026 PHYDRA, Inc. All rights reserved.

from decimal import Decimal, localcontext
from fractions import Fraction

import numpy as np
import pytest

from phydrax.discretization import CellMesh
from phydrax.discretization._cell_geometry import CellGeometrySpec
from phydrax.discretization._cell_geometry_transfer import (
    reconstruct_parametric_surface_cell_geometry,
)
from phydrax.discretization._coordinate_budget import CoordinateEnclosureResourceError
from phydrax.meshing.qualification import sphere_surface
from tests._support.coordinate_references import coordinate_reference_budget


@pytest.mark.parametrize("staged_value", (0.0, -0.0, np.nextafter(0.0, 1.0), 1e200))
@pytest.mark.parametrize("permuted", (False, True))
def test_surface_reconstruction_publishes_own_carrier_before_certification(
    staged_value: float, permuted: bool
) -> None:
    from phydrax.geometry._embedding_state import MeshCertificateLimits
    from phydrax.geometry._global_embedding import coordinate_scope

    domain = sphere_surface().domain
    uv = np.asarray([[0.0, 0.0], [0.1, 0.0], [0.1, 0.1]])
    # Coincident, disconnected occurrences must keep their own nominal IDs.
    points = np.tile(domain.evaluate(np.zeros(3, dtype=np.int32), uv), (2, 1))
    cells = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
    source = CellMesh.from_triangles(
        points, cells, cell_global_ids=np.asarray([10, 11], dtype=np.int64)
    )
    staged = points.copy()
    staged[:, 0] = staged_value
    target = source.with_coordinates(staged, numeric_version="staged-controller")
    order = np.asarray([1, 0] if permuted else [0, 1], dtype=np.int64)
    with coordinate_reference_budget() as budget:
        result = reconstruct_parametric_surface_cell_geometry(
            source, CellGeometrySpec.affine(source), target,
            CellGeometrySpec.affine(target), domain,
            domain_id=domain.domain_id,
            cell_ids=np.asarray([10, 11], dtype=np.int64)[order],
            cell_patches=np.zeros(2, dtype=np.int32),
            cell_charts=np.tile(uv, (2, 1, 1))[order],
            cell_geometry_entity_ids=(domain.entity_id(2, 0),) * 2,
            cell_occurrence_paths=(domain.source_occurrences[2][0],) * 2,
            maximum_fidelity=0.1, budget=budget,
        )
        scope = coordinate_scope(result.target_mesh, result.geometry, budget=budget)
        # Independent decimal conversion of the exact nodal source, not a copy
        # of the production rounded-image helper or the staged carrier.
        with localcontext() as context:
            context.prec = 100
            reference_rows = []
            for row in scope.source_values:
                reference_row = []
                for value in row:
                    assert isinstance(value, Fraction)
                    reference_row.append(
                        float(Decimal(value.numerator) / Decimal(value.denominator))
                    )
                reference_rows.append(reference_row)
            reference = np.asarray(reference_rows)
        actual = np.asarray(result.target_mesh.coordinates)
        np.testing.assert_array_equal(actual.view(np.uint64), reference.view(np.uint64))
        np.testing.assert_array_equal(
            actual.view(np.uint64), np.asarray(result.vertex_coordinates).view(np.uint64)
        )
        assert result.target_mesh.topology_id == target.topology_id
        assert result.target_mesh.numeric_version == target.numeric_version
        np.testing.assert_array_equal(
            result.target_mesh.vertex_global_ids, target.vertex_global_ids
        )
        np.testing.assert_array_equal(
            np.asarray(target.coordinates).view(np.uint64), staged.view(np.uint64)
        )
        changed = actual.copy()
        changed[4, 0] = np.nextafter(changed[4, 0], np.inf)
        with pytest.raises(ValueError, match="coordinate_scope.carrier"):
            coordinate_scope(
                result.target_mesh.with_coordinates(changed, numeric_version="fault"),
                result.geometry, budget=budget,
            )
        coefficients = np.asarray(result.geometry.coordinates).copy()
        coefficients[4, 0] = np.nextafter(coefficients[4, 0], np.inf)
        with pytest.raises(ValueError, match="coordinate_scope.carrier"):
            coordinate_scope(
                result.target_mesh, result.geometry.with_coordinates(coefficients),
                budget=budget,
            )
        names = result.geometry.block_names
        routes = {
            name: np.asarray(route).copy()
            for name, route in zip(names, result.geometry.geometry_dofs, strict=True)
        }
        routes[names[0]][0, :2] = routes[names[0]][0, 1::-1]
        foreign = CellGeometrySpec(
            dict(zip(names, result.geometry.elements, strict=True)),
            routes, result.geometry.coordinates,
        )
        with pytest.raises(ValueError, match="coordinate_scope.carrier"):
            coordinate_scope(result.target_mesh, foreign, budget=budget)
    # Measure cold, then replay with fresh ledgers so cache warmth cannot hide
    # the exact arithmetic request boundary.
    with coordinate_reference_budget() as cold:
        coordinate_scope(result.target_mesh, result.geometry, budget=cold)
        required = cold.work_units
    with coordinate_reference_budget(
        MeshCertificateLimits(maximum_work_units=required)
    ) as exact:
        coordinate_scope(result.target_mesh, result.geometry, budget=exact)
        assert exact.work_units == required
    with (
        pytest.raises(CoordinateEnclosureResourceError),
        coordinate_reference_budget(
            MeshCertificateLimits(maximum_work_units=required - 1)
        ) as short,
    ):
        coordinate_scope(result.target_mesh, result.geometry, budget=short)


