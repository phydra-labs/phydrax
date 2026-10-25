#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Uniform-median planar construction admission and bounds."""

import numpy as np
import pytest

import phydrax as phx
from phydrax._meshcore import MeshcoreError, MeshcoreStatus, NativeExecutionScope
from phydrax.meshing import MeshingLimits, SurfaceMeshingSpec
from phydrax.meshing._failures import MeshingFailure, MeshingFailureCategory
from phydrax.meshing.providers._native_planar_construction import (
    LATTICE_FRACTION,
    lattice_site_upper,
    lattice_spacing,
    uniform_construction_size,
    uniform_median_strategy,
)
from phydrax.meshing.providers._native_planar_domain import PreparedPlanarDomain
from phydrax.meshing.providers._native_route_policy import publication_policy
from tools._meshing_cases import meshing_case, prepare_case


def _request_with_limits(
    specification: SurfaceMeshingSpec, /, **overrides: int
) -> SurfaceMeshingSpec:
    current = specification.limits
    values = {
        name: getattr(current, name)
        for name in (
            "maximum_vertices",
            "maximum_edges",
            "maximum_faces",
            "maximum_cells",
            "maximum_connectivity_entries",
            "maximum_data_bytes",
            "maximum_work_units",
            "maximum_cavity_cells",
            "maximum_geometry_queries",
            "maximum_scratch_bytes",
        )
    }
    values.update(overrides)
    limits = MeshingLimits(
        **values,
        maximum_wall_seconds=current.maximum_wall_seconds,
    )
    return SurfaceMeshingSpec(
        specification.target,
        specification.scope,
        planar_embedding=specification.planar_embedding,
        size_controls=specification.size_controls,
        protected_features=specification.protected_features,
        region_controls=specification.region_controls,
        patch_controls=specification.patch_controls,
        periodic_constraints=specification.periodic_constraints,
        layer_controls=specification.layer_controls,
        size_combination=specification.size_combination,
        size_compliance=specification.size_compliance,
        limits=limits,
        quality_target=specification.quality_target,
        background_metric=specification.background_metric,
        deterministic=specification.deterministic,
    )


def test_lattice_spacing_follows_the_admitted_uniform_median() -> None:
    _, specification, _ = prepare_case(meshing_case("planar-hole-feature"), 32, 20000, 120.0)
    size = specification.size_controls[0].target_size
    assert uniform_median_strategy(specification, size)
    upper = uniform_construction_size(specification, size)
    assert upper == pytest.approx(1.5 * size)
    assert lattice_spacing(specification, size) == LATTICE_FRACTION * upper
    assert lattice_spacing(specification, 2.0 * size) == 0.0
    assert uniform_construction_size(specification, 2.0 * size) == 2.0 * size


def test_lattice_site_upper_covers_every_visited_row_and_column() -> None:
    lower, upper = np.zeros(2), np.full(2, 2.0)
    spacing = 0.03375
    rows = int(np.floor(2.0 / (spacing * 0.5 * np.sqrt(3.0)))) + 1
    columns = int(np.floor(2.0 / spacing)) + 1
    assert lattice_site_upper(lower, upper, spacing) == columns * rows + rows
    assert lattice_site_upper(lower, upper, 0.0) == 0


def test_planar_publication_retains_observed_preparation_and_cavity_evidence() -> None:
    source, specification, options = prepare_case(
        meshing_case("planar-hole-feature"), 4, 20000, 120.0
    )
    with NativeExecutionScope(specification.limits.native_limits()):
        plan = phx.meshing.NativeMeshingProvider(options).plan(
            source,
            specification,
            coordinate_contract=phx.SpatialCoordinateContract.si(),
        )
        assert isinstance(plan.prepared, PreparedPlanarDomain)
        preparation = plan.preparation_evidence
        assert preparation is not None
        preparation_evidence = preparation.native_evidence()
        assert preparation_evidence.managed_admission_peak_bytes_upper is not None
        assert (
            preparation_evidence.host_storage_peak_bytes_upper
            >= plan.prepared.points.nbytes
        )

        result = plan.execute()
    execution = result.execution_evidence
    assert execution is not None
    execution_evidence = execution.native_evidence()
    assert execution_evidence.managed_admission_peak_bytes_upper is not None
    assert execution_evidence.work_counters.peak_cavity_cells > 0
    assert execution.preparation_evidence is not None
    assert (
        execution.preparation_evidence.native_evidence().managed_admission_peak_bytes_upper
        is not None
    )
    policy = publication_policy("planar")
    assert set(policy.enforced_limits) <= set(result.runtime.enforced_limits)
    assert result.runtime.unenforced_limits == policy.unenforced_limits


def test_planar_native_cavity_cap_refuses_actual_route_construction() -> None:
    source, specification, options = prepare_case(
        meshing_case("planar-hole-feature"), 4, 20000, 120.0
    )
    request = _request_with_limits(specification, maximum_cavity_cells=4)
    plan = phx.meshing.NativeMeshingProvider(options).plan(
        source,
        request,
        coordinate_contract=phx.SpatialCoordinateContract.si(),
    )
    with pytest.raises(MeshingFailure) as caught:
        plan.execute()
    failure = caught.value
    assert failure.category is MeshingFailureCategory.RESOURCE_EXHAUSTED
    assert dict(failure.evidence.achieved)["native_planar:cavity_refusals"] > 0
    assert dict(failure.evidence.requested)["maximum_cavity_cells"] == 4



def test_planar_scratch_cap_refuses_before_publication() -> None:
    source, specification, options = prepare_case(
        meshing_case("planar-hole-feature"), 4, 20000, 120.0
    )
    request = _request_with_limits(specification, maximum_scratch_bytes=1)
    with pytest.raises((MeshcoreError, MeshingFailure)) as caught:
        plan = phx.meshing.NativeMeshingProvider(options).plan(
            source,
            request,
            coordinate_contract=phx.SpatialCoordinateContract.si(),
        )
        plan.execute()
    refusal = caught.value
    if isinstance(refusal, MeshcoreError):
        assert refusal.status is MeshcoreStatus.CAPACITY_EXCEEDED
    else:
        assert refusal.category is MeshingFailureCategory.RESOURCE_EXHAUSTED
        assert dict(refusal.evidence.requested)["maximum_scratch_bytes"] == 1
