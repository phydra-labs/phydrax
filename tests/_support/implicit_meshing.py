"""Box-query builders, resource fixtures and coverage assertions for implicit meshing."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

from phydrax._meshcore import exact_orient3d, meshcore_available, NativeExecutionScope
from phydrax._resource_ticket import HostBatchTicket
from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from phydrax.geometry._contracts import CompiledGeometry
from phydrax.geometry.implicit._adaptive_discovery import ImplicitVolumeQuery
from phydrax.geometry.implicit._enclosure import field_bounds
from phydrax.geometry.implicit._policy import ImplicitDiscoveryEnclosure
from phydrax.meshing._contracts import (
    CellFamilyPolicy,
    CellMeshingTarget,
    MeshingLimits,
    VolumeFillStrategy,
    VolumeMeshingSpec,
)
from phydrax.meshing._implicit_volume import (
    ImplicitDualRootWorkset,
    ImplicitVolumeDomainWorkset,
)
from phydrax.meshing._scope import MeshingEntityKind, MeshingScope
from phydrax.meshing._sizing import UniformSizeControl


IMPLICIT_DOMAIN = np.asarray(((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)), dtype=np.float64)
NATIVE = pytest.mark.skipif(
    not meshcore_available(), reason="native meshcore library is unavailable"
)

# The pre-refactor unit-box oracle used this empirical absolute allowance for
# its independent host determinant volume sum. The value is retained unchanged.
BOX_VOLUME_ROUNDOFF_ATOL = 1.0e-12


@pytest.fixture
def execution_scope() -> Iterator[NativeExecutionScope]:
    with NativeExecutionScope(MeshingLimits().native_limits()) as scope:
        yield scope


@pytest.fixture
def source_budget() -> Iterator[CoordinateEnclosureBudget]:
    limits = MeshingLimits()
    with HostBatchTicket(
        limits.maximum_work_units,
        queries_upper=limits.maximum_geometry_queries,
        bytes_upper=limits.maximum_scratch_bytes,
    ) as ticket:
        budget = CoordinateEnclosureBudget(
            ticket,
            maximum_work_units=limits.maximum_work_units,
            maximum_memory_bytes=limits.maximum_scratch_bytes,
        )
        with budget.activate():
            yield budget


def implicit_query(
    geometry: CompiledGeometry,
    enclosure: ImplicitDiscoveryEnclosure = "interval",
) -> ImplicitVolumeQuery:
    # One box provides the query enclosure independently of surface extraction.
    bounds = field_bounds(geometry, enclosure)
    values = bounds.boxes(IMPLICIT_DOMAIN[:1], IMPLICIT_DOMAIN[1:])
    return ImplicitVolumeQuery(
        bounds,
        IMPLICIT_DOMAIN[:1],
        IMPLICIT_DOMAIN[1:],
        values.value_lower,
        values.value_upper,
        np.asarray((0,), dtype=np.int64),
        IMPLICIT_DOMAIN,
        maximum_level=1,
    )


def implicit_specification(limits: MeshingLimits | None = None) -> VolumeMeshingSpec:
    scope = MeshingScope(
        "implicit",
        "state",
        MeshingEntityKind.GEOMETRY,
        2,
        "boundary",
        np.asarray((0,), dtype=np.int64),
    )
    return VolumeMeshingSpec(
        CellMeshingTarget(3, 3, CellFamilyPolicy(required=("tetrahedron",))),
        scope,
        VolumeFillStrategy.SIMPLEX,
        size_controls=(UniformSizeControl(scope, 0.2),),
        limits=limits,
    )


def assert_box_coverage(work: ImplicitVolumeDomainWorkset) -> None:
    corners = work.points[work.tetrahedra]
    assert np.all(
        exact_orient3d(corners[:, 0], corners[:, 1], corners[:, 2], corners[:, 3]) > 0
    )
    volumes = np.linalg.det(corners[:, 1:] - corners[:, :1]) / 6.0
    assert np.sum(volumes) == pytest.approx(8.0, abs=BOX_VOLUME_ROUNDOFF_ATOL)
    assert np.all(work.points >= IMPLICIT_DOMAIN[0])
    assert np.all(work.points <= IMPLICIT_DOMAIN[1])


def assert_terminal_cover(work: ImplicitDualRootWorkset, segment: int = 0) -> None:
    intervals = work.parameter_intervals[work.segment_ids == segment]
    assert intervals[0, 0] == 0.0
    assert intervals[-1, 1] == 1.0
    np.testing.assert_array_equal(intervals[:-1, 1], intervals[1:, 0])
    assert np.all(intervals[:, 0] < intervals[:, 1])
