#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""PLC source-row identity, query allowance, and consuming-boundary contracts."""

from typing import NamedTuple

import numpy as np
import pytest

from phydrax._meshcore import NativeExecutionScope, TetMesh3D
from phydrax._physical import SpatialCoordinateContract
from phydrax.discretization._cell_geometry_validity import CellValidityPolicy
from phydrax.geometry._domain_coverage import PiecewiseLinearDomain
from phydrax.meshing._association import PlcAssociationTransfer
from phydrax.meshing._contracts import (
    CellFamilyPolicy,
    CellMeshingTarget,
    VolumeFillStrategy,
    VolumeMeshingSpec,
)
from phydrax.meshing._controls import FeatureKind, ProtectedFeature
from phydrax.meshing._plc_planar_support import planar_point_support
from phydrax.meshing._plc_source import PiecewiseLinearComplex
from phydrax.meshing._scope import MeshingEntityKind, MeshingScope
from phydrax.meshing._sizing import SizeControlStrength, UniformSizeControl
from phydrax.meshing._support_work import spend_support_work
from phydrax.meshing._trace import (
    MeshingStageKind,
    MeshingStageReport,
    MeshingStageStatus,
)
from phydrax.meshing._volume_generation import (
    finalize_plc_volume,
    NativeVolumeSchedule,
    plc_source_rows,
    PlcCarrier,
)
from tests._support.coordinate_references import coordinate_reference_budget


class PlcPhaseCase(NamedTuple):
    complex: PiecewiseLinearComplex
    specification: VolumeMeshingSpec
    triangles: np.ndarray
    edges: np.ndarray
    bounds: tuple[float, ...]


@pytest.fixture
def plc_case() -> PlcPhaseCase:
    points = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    triangles = np.asarray(((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)), dtype=np.int64)
    edges = np.asarray(((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)), dtype=np.int64)
    polygons = np.arange(4, dtype=np.int64)
    source = PiecewiseLinearComplex(
        points,
        tuple(triangles),
        polygons,
        np.tile(np.asarray((-1, 0), dtype=np.int64), (4, 1)),
        ("solid",),
    )
    boundary = MeshingScope(
        "source-row-contract",
        "revision",
        MeshingEntityKind.GEOMETRY,
        2,
        "source-facets",
        polygons,
    )
    bounds = (0.25, 0.5, 0.75, 1.0)
    features = tuple(
        ProtectedFeature(
            MeshingScope(
                boundary.source_id,
                boundary.source_revision,
                MeshingEntityKind.GEOMETRY,
                2,
                boundary.entity_set_id,
                np.asarray((facet,), dtype=np.int64),
            ),
            FeatureKind.SURFACE,
            maximum_deviation=bound,
        )
        for facet, bound in enumerate(bounds)
    )
    specification = VolumeMeshingSpec(
        CellMeshingTarget(3, 3, CellFamilyPolicy(required=("tetrahedron",))),
        boundary,
        VolumeFillStrategy.SIMPLEX,
        size_controls=(
            UniformSizeControl(boundary, 4.0, strength=SizeControlStrength.SOFT),
        ),
        protected_features=features,
    )
    return PlcPhaseCase(source, specification, triangles, edges, bounds)


@pytest.mark.parametrize(
    ("remaining", "charge", "expected"),
    ((9, 3, 6), (3, 3, 0), (0, 0, 0)),
    ids=("partial-charge", "exact-limit", "empty-no-charge"),
)
def test_support_queries_retain_the_remaining_allowance(
    remaining: int, charge: int, expected: int
) -> None:
    assert (
        spend_support_work(remaining, charge, "support.work must fit; got excess")
        == expected
    )


def test_support_queries_refuse_one_beyond_the_allowance() -> None:
    with pytest.raises(ValueError, match=r"support\.work"):
        spend_support_work(3, 4, "support.work must fit; got excess")


def test_plc_source_rows_preserve_incidence_and_shared_edge_deviation_bounds(
    plc_case: PlcPhaseCase,
) -> None:
    rows = plc_source_rows(
        plc_case.complex,
        plc_case.specification,
        plc_case.triangles,
        np.arange(4, dtype=np.int64),
        plc_case.edges,
        np.empty((0, 2), dtype=np.int64),
    )
    # Requested bounds and their minima are binary64 dyadic values, so equality
    # needs no floating-point tolerance.
    np.testing.assert_array_equal(rows.points, plc_case.complex.vertices)
    np.testing.assert_array_equal(rows.faces, plc_case.triangles)
    np.testing.assert_array_equal(rows.triangle_facets, np.arange(4, dtype=np.int64))
    np.testing.assert_array_equal(rows.face_tolerances, plc_case.bounds)
    np.testing.assert_array_equal(rows.segments, plc_case.edges)
    np.testing.assert_array_equal(rows.segment_ids, np.arange(6, dtype=np.int64))
    np.testing.assert_array_equal(
        rows.segment_tolerances, (0.25, 0.25, 0.5, 0.25, 0.5, 0.75)
    )
    assert rows.points.dtype == np.dtype(np.float64)
    assert rows.faces.dtype == rows.segments.dtype == np.dtype(np.int32)
    assert rows.triangle_facets.dtype == rows.segment_ids.dtype == np.dtype(np.int64)
    assert (
        rows.face_tolerances.dtype
        == rows.segment_tolerances.dtype
        == np.dtype(np.float64)
    )


def test_finishing_invalid_policy_still_closes_the_admitted_carrier(
    plc_case: PlcPhaseCase,
) -> None:
    with NativeExecutionScope(plc_case.specification.limits.native_limits()) as scope:
        state = TetMesh3D(
            plc_case.complex.vertices,
            np.asarray(((0, 1, 2, 3),), dtype=np.int32),
            np.zeros((1,), dtype=np.int32),
            plc_case.triangles.astype(np.int32),
            np.arange(4, dtype=np.int32),
            plc_case.edges.astype(np.int32),
            np.arange(6, dtype=np.int32),
            boundary_policy="fixed",
            max_vertices=8,
            max_tetrahedra=8,
        )
        carrier = PlcCarrier(
            state,
            plc_case.complex,
            "source-row-contract",
            "revision",
            "carrier-input",
            plc_case.triangles,
            np.arange(4, dtype=np.int64),
            plc_case.edges,
            np.empty((0,), dtype=np.int64),
            MeshingStageReport(MeshingStageKind.VOLUME_FILL, MeshingStageStatus.PASSED),
            (),
            0,
        )
        try:
            with pytest.raises(TypeError, match="validity_policy"):
                finalize_plc_volume(
                    carrier,
                    plc_case.specification,
                    NativeVolumeSchedule(),
                    validity_policy=object(),  # ty: ignore[invalid-argument-type]
                    scope=scope,
                )
            with pytest.raises(ValueError, match="closed"):
                state.arrays()
        finally:
            state.close()


def test_finishing_invalid_carrier_preserves_the_nominal_type_error(
    plc_case: PlcPhaseCase,
) -> None:
    scope = NativeExecutionScope(plc_case.specification.limits.native_limits())
    with pytest.raises(TypeError, match="carrier|PlcCarrier"):
        finalize_plc_volume(
            object(),  # ty: ignore[invalid-argument-type]
            plc_case.specification,
            NativeVolumeSchedule(),
            validity_policy=CellValidityPolicy(),
            scope=scope,
        )


@pytest.fixture
def planar_transfer() -> PlcAssociationTransfer:
    vertices = np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), dtype=np.float64)
    edges = np.asarray(((0, 1), (1, 2), (2, 0)), dtype=np.int64)
    domain = PiecewiseLinearDomain(
        vertices,
        edges,
        np.tile(np.asarray((0, -1), dtype=np.int64), (3, 1)),
        ("solid",),
        source_id="planar-query-refusal",
    )
    return PlcAssociationTransfer(
        domain,
        SpatialCoordinateContract.si(),
        "revision",
        edge_vertices=edges,
        triangle_vertices=np.empty((0, 3), dtype=np.int64),
        triangle_facets=np.empty((0,), dtype=np.int64),
        facet_regions=np.empty((0, 2), dtype=np.int64),
        maximum_support_queries=1,
    )


def test_planar_support_refusal_precedes_source_operation(
    planar_transfer: PlcAssociationTransfer,
) -> None:
    balance = [0]
    with coordinate_reference_budget() as budget, budget.measurement() as measurement:
        with pytest.raises(ValueError, match=r"PlanarPlcSupport\.work"):
            planar_point_support(
                planar_transfer,
                0,
                0,
                np.asarray((0.0, 0.0), dtype=np.float64),
                balance,
                budget=budget,
            )
        # This adapter publishes no attempted debit. The source-coordinate
        # operation is after quota admission, so refused work is not charged.
        assert balance == [0]
        assert budget.work_units == 0
    assert measurement.work_units == 0
    assert measurement.native_charged_work_units == 0


def test_planar_support_refusal_retains_earned_coordinate_evidence(
    planar_transfer: PlcAssociationTransfer,
) -> None:
    balance = [1]
    point = np.asarray((0.0, 0.0), dtype=np.float64)
    with coordinate_reference_budget() as budget, budget.measurement() as measurement:
        assert planar_point_support(planar_transfer, 0, 0, point, balance, budget=budget)
        assert balance == [0]
        charged_work = budget.work_units
        charged_native_work = budget.native_charged_work_units
        assert charged_work > 0
        with pytest.raises(ValueError, match=r"PlanarPlcSupport\.work"):
            planar_point_support(planar_transfer, 0, 0, point, balance, budget=budget)
        assert balance == [0]
        assert budget.work_units == charged_work
        assert budget.native_charged_work_units == charged_native_work
    assert measurement.work_units == charged_work
    assert measurement.native_charged_work_units == charged_native_work
