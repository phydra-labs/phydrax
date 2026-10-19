#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Validate stored chart bindings without changing numerical record identity."""

import math
from dataclasses import replace

from phydrax import config as pxconfig


pxconfig.configure(pxconfig.RuntimeConfig(enable_x64=True))
import numpy as np
import pytest

from phydrax._meshcore import NativeExecutionLimits, NativeExecutionScope
from phydrax.discretization import (
    CellGeometryTransitionPolicy,
    CellValidityPolicy,
    prepare_sphere_chart_deformation,
    PreparedSphereChartDeformation,
    PreparedSurfaceChartDeformation,
    reconstruct_sphere_material_cell_geometry,
    SphereGeometryReconstruction,
)
from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from phydrax.geometry import MeshCertificateLimits
from tests._support.mapped_cells import curved_surface_chart_fv
from tests._support.sphere_material import accepted_sphere_material


@pytest.fixture(scope="module")
def surface_chart() -> PreparedSurfaceChartDeformation:
    return curved_surface_chart_fv()[2]


@pytest.fixture(scope="module")
def sphere_chart() -> PreparedSphereChartDeformation:
    _, _, atlas = accepted_sphere_material(1.0)
    return prepare_sphere_chart_deformation(
        atlas, atlas, maximum_fidelity=0.3, maximum_displacement=0.5
    )


def test_surface_restore_retains_instance_and_numerical_content(
    surface_chart: PreparedSurfaceChartDeformation,
) -> None:
    identity = surface_chart.deformation_id
    measures = np.asarray(surface_chart.source_cell_measures).copy()
    restored = PreparedSurfaceChartDeformation.from_stored_fields(surface_chart)
    assert restored is surface_chart
    assert restored.deformation_id == identity
    np.testing.assert_array_equal(restored.source_cell_measures, measures)


def test_surface_restore_refuses_stale_geometry_binding(
    surface_chart: PreparedSurfaceChartDeformation,
) -> None:
    stale = replace(surface_chart, target_geometry_id="stale")
    with pytest.raises(ValueError, match=r"surface_chart\.require_bound\.binding"):
        PreparedSurfaceChartDeformation.from_stored_fields(stale)


def test_sphere_restore_retains_instance_and_numerical_content(
    sphere_chart: PreparedSphereChartDeformation,
) -> None:
    identity = sphere_chart.deformation_id
    measures = np.asarray(sphere_chart.target_cell_measures).copy()
    restored = PreparedSphereChartDeformation.from_stored_fields(sphere_chart)
    assert restored is sphere_chart
    assert restored.deformation_id == identity
    np.testing.assert_array_equal(restored.target_cell_measures, measures)


def test_sphere_restore_refuses_stale_geometry_binding(
    sphere_chart: PreparedSphereChartDeformation,
) -> None:
    stale = replace(sphere_chart, target_geometry_id="stale")
    with pytest.raises(ValueError, match=r"sphere_chart\.require_bound\.binding"):
        PreparedSphereChartDeformation.from_stored_fields(stale)


@pytest.fixture(scope="module")
def sphere_reconstruction() -> SphereGeometryReconstruction:
    _, _, atlas = accepted_sphere_material(1.0)
    count = atlas.num_charts
    dimensions = np.empty((count, 3), dtype=np.int32)
    indices = np.empty((count, 3), dtype=np.int32)
    parameters = np.zeros((count, 3, 2), dtype=np.float64)
    patches = np.empty((count,), dtype=np.int32)
    paths: list[tuple[tuple[str, ...], ...]] = [()] * count
    entities: list[tuple[str, ...]] = [()] * count
    # Corner tokens retain source indices and occurrence paths; scientific IDs
    # must be resolved through the domain, not inferred from equal array rows.
    for chart_row, physical_row in enumerate(np.asarray(atlas.physical_rows)):
        row = int(physical_row)
        patches[row] = np.asarray(atlas.patches)[chart_row]
        row_paths, row_entities = [], []
        for corner, (dimension, source_index, path, values) in enumerate(
            atlas.corner_tokens[chart_row]
        ):
            dimensions[row, corner] = dimension
            indices[row, corner] = source_index
            parameters[row, corner, : len(values)] = [float(value) for value in values]
            candidates = tuple(
                local
                for local, (identifier, occurrence) in enumerate(
                    zip(
                        atlas.domain.source_indices[dimension],
                        atlas.domain.source_occurrences[dimension],
                        strict=True,
                    )
                )
                if identifier == source_index and occurrence == path
            )
            assert len(candidates) == 1
            row_paths.append(path)
            row_entities.append(atlas.domain.entity_id(dimension, candidates[0]))
        paths[row] = tuple(row_paths)
        entities[row] = tuple(row_entities)
    work, memory = 10_000_000, 256 * 1024**2
    with NativeExecutionScope(
        NativeExecutionLimits(2 * work, work, 0, memory, math.inf)
    ) as scope:
        with scope.batch(2 * work, queries_upper=work, bytes_upper=memory) as ticket:
            budget = CoordinateEnclosureBudget(
                ticket, maximum_work_units=work, maximum_memory_bytes=memory
            )
            return reconstruct_sphere_material_cell_geometry(
                atlas,
                atlas.source_mesh,
                atlas.source_geometry,
                cell_patches=patches,
                corner_source_dimensions=dimensions,
                corner_source_indices=indices,
                corner_source_parameters=parameters,
                corner_source_occurrence_paths=tuple(paths),
                corner_source_entity_ids=tuple(entities),
                maximum_fidelity=0.3,
                policy=CellGeometryTransitionPolicy(
                    reconstruction="bounded_chart_deformation"
                ),
                budget=budget,
                certificate_limits=MeshCertificateLimits(),
                validity_policy=CellValidityPolicy(),
            )


def test_reconstruction_restore_retains_instance_and_work_evidence(
    sphere_reconstruction: SphereGeometryReconstruction,
) -> None:
    identity = sphere_reconstruction.reconstruction_id
    counts = (
        sphere_reconstruction.evaluation_count,
        sphere_reconstruction.preparation_work_units,
    )
    coordinates = np.asarray(sphere_reconstruction.geometry.coordinates).copy()
    restored = SphereGeometryReconstruction.from_stored_fields(sphere_reconstruction)
    assert restored is sphere_reconstruction
    assert restored.reconstruction_id == identity
    assert (restored.evaluation_count, restored.preparation_work_units) == counts
    np.testing.assert_array_equal(restored.geometry.coordinates, coordinates)


def test_reconstruction_restore_refuses_stale_topology_binding(
    sphere_reconstruction: SphereGeometryReconstruction,
) -> None:
    stale = replace(sphere_reconstruction, target_topology_id="stale")
    with pytest.raises(ValueError, match=r"sphere_reconstruction\.binding"):
        SphereGeometryReconstruction.from_stored_fields(stale)


def test_surface_restore_refuses_wrong_record_kind(
    sphere_chart: PreparedSphereChartDeformation,
) -> None:
    with pytest.raises(TypeError, match="value"):
        PreparedSurfaceChartDeformation.from_stored_fields(
            sphere_chart,  # ty: ignore[invalid-argument-type]
        )


def test_sphere_restore_refuses_wrong_record_kind(
    surface_chart: PreparedSurfaceChartDeformation,
) -> None:
    with pytest.raises(TypeError, match="value"):
        PreparedSphereChartDeformation.from_stored_fields(
            surface_chart,  # ty: ignore[invalid-argument-type]
        )


def test_reconstruction_restore_refuses_wrong_record_kind(
    sphere_chart: PreparedSphereChartDeformation,
) -> None:
    with pytest.raises(TypeError, match="value"):
        SphereGeometryReconstruction.from_stored_fields(
            sphere_chart,  # ty: ignore[invalid-argument-type]
        )
