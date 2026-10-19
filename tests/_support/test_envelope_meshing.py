# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Closed-form anchors for the envelope builders and independent field references."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax import SpatialCoordinateContract
from phydrax.discretization import CellBlock, CellMesh
from phydrax.geometry import SurfaceMetadata
from phydrax.meshing import (
    EnvelopeFeaturePermission,
    EnvelopeTopologyPermission,
    RawTriangleSoup,
    SurfaceEnvelopePolicy,
)
from tests._support.envelope_meshing import (
    envelope_failure,
    envelope_metadata,
    envelope_policy,
    envelope_sheet,
    quadratic_l2_error,
    quadratic_source,
    quadratic_value,
    tetrahedron_measures,
)


# Closed-form simplex moments incur only rounding in host polynomial summation.
_MOMENT_RELATIVE_ROUNDING = 128 * np.finfo(np.float64).eps


def _require_authored_metadata(metadata: SurfaceMetadata) -> None:
    assert metadata.source_id == "sheet-source"
    assert metadata.source_revision == "scan-17"
    assert (
        metadata.coordinate_contract.spatial_id
        == SpatialCoordinateContract.si().spatial_id
    )
    assert metadata.provenance == ("synthetic-open-sheet",)


def _require_authored_sheet(soup: RawTriangleSoup) -> None:
    _require_authored_metadata(soup.metadata)
    np.testing.assert_array_equal(
        soup.vertices, ((0.0, 0.0, 0.0), (0.2, 0.0, 0.0), (0.0, 0.2, 0.0))
    )
    np.testing.assert_array_equal(soup.triangles, ((0, 1, 2),))
    np.testing.assert_array_equal(soup.feature_edges, ((0, 1),))
    assert not soup.vertices.flags.writeable
    assert not soup.triangles.flags.writeable
    assert not soup.feature_edges.flags.writeable


def test_envelope_sheet_preserves_authored_source_and_feature_identity() -> None:
    metadata = envelope_metadata()
    _require_authored_metadata(metadata)
    corrupted_metadata = SurfaceMetadata(
        source_id=metadata.source_id,
        source_revision="changed-revision",
        coordinate_contract=metadata.coordinate_contract,
        provenance=metadata.provenance,
    )
    with pytest.raises(AssertionError):
        _require_authored_metadata(corrupted_metadata)
    soup = envelope_sheet()
    assert soup.metadata.metadata_id == metadata.metadata_id
    _require_authored_sheet(soup)
    corrupted_sheet = RawTriangleSoup(
        soup.vertices,
        soup.triangles[:, ::-1],
        soup.metadata,
        feature_edges=soup.feature_edges,
    )
    with pytest.raises(AssertionError):
        _require_authored_sheet(corrupted_sheet)


def test_envelope_policy_preserves_empirical_geometry_and_requested_capacities() -> None:
    def require_geometry(policy: SurfaceEnvelopePolicy) -> None:
        assert (policy.offset, policy.maximum_deviation) == (0.263, 0.55)
        assert (policy.spacing, policy.certificate_spacing) == (0.11, 0.05)
        assert (policy.max_samples, policy.max_work_units) == (100_000, 10_000_000)
        assert (policy.max_vertices, policy.max_triangles) == (10_000, 20_000)

    default = envelope_policy()
    require_geometry(default)
    with pytest.raises(AssertionError):
        require_geometry(envelope_policy(max_samples=1))
    requested = envelope_policy(
        max_samples=17,
        max_work_units=19,
        max_vertices=23,
        max_triangles=29,
        max_volume_vertices=31,
        max_tetrahedra=37,
        feature_permission=EnvelopeFeaturePermission.PRESERVE,
        topology_permission=EnvelopeTopologyPermission.PRESERVE,
    )

    def require_requested(policy: SurfaceEnvelopePolicy) -> None:
        assert (policy.max_samples, policy.max_work_units) == (17, 19)
        assert (policy.max_vertices, policy.max_triangles) == (23, 29)
        assert (policy.max_volume_vertices, policy.max_tetrahedra) == (31, 37)
        assert policy.feature_permission is EnvelopeFeaturePermission.PRESERVE
        assert policy.topology_permission is EnvelopeTopologyPermission.PRESERVE

    require_requested(requested)
    with pytest.raises(AssertionError):
        require_requested(default)


def test_envelope_failure_retains_field_and_refusal_evidence() -> None:
    def require_evidence(error: RuntimeError) -> None:
        assert type(error) is RuntimeError
        assert str(error) == "field.bound must be positive; got {'bound': -1.0}"

    error = envelope_failure("field.bound", "be positive", {"bound": -1.0})
    require_evidence(error)
    with pytest.raises(AssertionError):
        require_evidence(RuntimeError("field.bound"))


@pytest.mark.parametrize(
    ("vertex_order", "orientation"),
    (((0, 1, 2, 3), 1), ((0, 2, 1, 3), -1)),
    ids=("positive", "reversed"),
)
def test_tetrahedron_measures_detect_orientation_and_scaling(
    vertex_order: tuple[int, ...],
    orientation: int,
) -> None:
    points = np.asarray(
        ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 3.0, 0.0), (0.0, 0.0, 4.0)),
        dtype=np.float64,
    )
    cells = np.asarray((vertex_order,), dtype=np.int64)

    def require_signed_volume(measures: np.ndarray) -> None:
        np.testing.assert_allclose(
            measures,
            (4.0 * orientation,),
            rtol=_MOMENT_RELATIVE_ROUNDING,
            atol=0.0,
        )

    measured = tetrahedron_measures(points, cells)
    require_signed_volume(measured)
    with pytest.raises(AssertionError):
        require_signed_volume(measured * 6.0)
    with pytest.raises(AssertionError):
        require_signed_volume(-measured)


def test_quadratic_coefficients_follow_physical_dimension_and_translation() -> None:
    points = jnp.asarray(((1.0, 2.0, 3.0), (-2.0, 0.0, 1.0)), dtype=jnp.float64)

    def require_coefficients(
        values: np.ndarray,
        source: np.ndarray,
        planar_source: np.ndarray,
        translated_values: np.ndarray,
    ) -> None:
        np.testing.assert_array_equal(values, (14.0, 5.0))
        np.testing.assert_array_equal(source, (-6.0, -6.0))
        np.testing.assert_array_equal(planar_source, (-4.0, -4.0))
        np.testing.assert_array_equal(translated_values, (29.0, 6.0))

    values = np.asarray(quadratic_value(points))
    source = np.asarray(quadratic_source(points, None))
    planar_source = np.asarray(quadratic_source(points[:, :2], None))
    translated_values = np.asarray(quadratic_value(points + 1.0))
    require_coefficients(values, source, planar_source, translated_values)
    with pytest.raises(AssertionError):
        require_coefficients(np.sqrt(values), source, planar_source, translated_values)
    with pytest.raises(AssertionError):
        require_coefficients(values, -source, planar_source, translated_values)
    with pytest.raises(AssertionError):
        require_coefficients(values, source, source, translated_values)


@pytest.mark.parametrize(
    ("values", "squared_error"),
    (
        ((0.0, 0.0, 0.0, 0.0), 2.0 / 105),
        ((0.0, 1.0, 0.0, 0.0), 1.0 / 126),
        ((0.0, 1.0, 1.0, 1.0), 1.0 / 28),
    ),
    ids=("zero-field", "linear-x", "nodal-quadratic"),
)
def test_quadratic_l2_error_detects_nonzero_p1_defect_by_exact_simplex_moments(
    values: tuple[float, ...],
    squared_error: float,
) -> None:
    points = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    mesh = CellMesh(
        points,
        (
            CellBlock(
                "unit",
                "tetrahedron",
                np.asarray(((0, 1, 2, 3),), dtype=np.int64),
            ),
        ),
    )

    # Integral x^a y^b z^c = a! b! c! / (a+b+c+3)! on the unit tetrahedron.
    def require_exact_moment(error: float) -> None:
        assert error > 0.0
        assert error * error == pytest.approx(
            squared_error, rel=_MOMENT_RELATIVE_ROUNDING, abs=0.0
        )

    error = quadratic_l2_error(mesh, jnp.asarray(values, dtype=jnp.float64))
    require_exact_moment(error)
    with pytest.raises(AssertionError):
        require_exact_moment(0.0)
    with pytest.raises(AssertionError):
        require_exact_moment(error * np.sqrt(6.0))
