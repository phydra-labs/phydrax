#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Curve placement and publication contracts through native route boundaries."""

from __future__ import annotations

import jax
import numpy as np
import pytest

import phydrax as phx
from phydrax._meshcore import NativeExecutionScope
from phydrax.geometry import BoundaryAtlas, BSplineCurve
from phydrax.geometry._meshing_domain import PhysicalCurveMap
from phydrax.meshing import (
    CellFamilyPolicy,
    CellMeshAuditPolicy,
    CellMeshingResult,
    CellMeshingTarget,
    CurveMeshingSpec,
    MeshingEntityKind,
    MeshingFailure,
    MeshingFailureCategory,
    MeshingLimits,
    MeshingScope,
    NativeCurveSource,
    NativeMeshingOptions,
    NativeMeshingProvider,
    SizeControlStrength,
    UniformSizeControl,
)
from phydrax.meshing._certification_schedule import MeshCertificationSchedule
from phydrax.meshing.providers._native_publication import (
    NativeCertificationRequest,
    NativePublicationExecution,
    NativePublicationOrganization,
    publish_native_result,
)
from phydrax.meshing.providers._native_route_policy import publication_policy


# Retained from the selected-chart density-root regression: eight-point source
# quadrature compared with an independent 128-point rational Bezier integral.
_DENSITY_FRACTION_TOLERANCE = 2.0e-5
_POINTS = (
    np.asarray(
        ((0.0, 0.0, 0.0), (0.3, 0.6, 0.0), (0.7, -0.2, 0.0), (1.0, 0.0, 0.0)),
        dtype=np.float64,
    ),
    np.asarray(
        ((0.0, 0.0, 0.0), (0.3, 0.2, 0.0), (0.7, 0.5, 0.0), (1.0, 0.0, 0.0)),
        dtype=np.float64,
    ),
)
_WEIGHTS = (
    np.asarray((1.0, 0.8, 1.2, 1.0), dtype=np.float64),
    np.asarray((1.0, 1.2, 0.7, 1.0), dtype=np.float64),
)


def _arc_integral(
    points: np.ndarray, rational_weights: np.ndarray, stop: np.ndarray
) -> np.ndarray:
    nodes, weights = np.polynomial.legendre.leggauss(128)
    parameter = 0.5 * np.asarray(stop)[..., None] * (nodes + 1.0)
    complement = 1.0 - parameter
    basis = np.stack(
        (
            complement**3,
            3.0 * parameter * complement**2,
            3.0 * parameter**2 * complement,
            parameter**3,
        ),
        axis=-1,
    )
    derivative = np.stack(
        (
            -3.0 * complement**2,
            3.0 * complement**2 - 6.0 * parameter * complement,
            6.0 * parameter * complement - 3.0 * parameter**2,
            3.0 * parameter**2,
        ),
        axis=-1,
    )
    weighted = basis * rational_weights
    weighted_derivative = derivative * rational_weights
    numerator = weighted @ points
    denominator = np.sum(weighted, axis=-1)
    velocity = (
        (weighted_derivative @ points) * denominator[..., None]
        - numerator * np.sum(weighted_derivative, axis=-1)[..., None]
    ) / denominator[..., None] ** 2
    return np.asarray(
        0.5
        * np.asarray(stop)
        * np.sum(np.linalg.norm(velocity, axis=-1) * weights, axis=-1)
    )


def test_arc_integral_oracle_recovers_straight_bezier_length() -> None:
    points = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (3.0, 0.0, 0.0)),
        dtype=np.float64,
    )
    stops = np.asarray((0.0, 0.1, 0.5, 0.9, 1.0), dtype=np.float64)

    # For equal rational weights this cubic is the line x(t)=3t. Gauss
    # summation of its constant speed differs only by float64 rounding.
    def require_straight_length(integrals: np.ndarray) -> None:
        np.testing.assert_allclose(
            integrals,
            3.0 * stops,
            rtol=128 * np.finfo(np.float64).eps,
            atol=0.0,
        )

    integrals = _arc_integral(points, np.ones(4, dtype=np.float64), stops)
    require_straight_length(integrals)
    with pytest.raises(AssertionError):
        require_straight_length(integrals / 3.0)


def _curve_request(
    order: tuple[int, ...], selected: int
) -> tuple[NativeCurveSource, CurveMeshingSpec]:
    sources = tuple(
        BSplineCurve.bezier(points, weights)
        for points, weights in zip(_POINTS, _WEIGHTS, strict=True)
    )
    mapping = PhysicalCurveMap(
        tuple(sources[index] for index in order),
        np.tile(np.asarray(((0.0, 1.0),), dtype=np.float64), (len(order), 1)),
    )
    atlas = BoundaryAtlas(
        mapping,
        source_entity_ids=jax.device_put(np.arange(len(order), dtype=np.int32)),
        source_id="rational-root-charts",
        physical_tags=tuple("curve" for _ in order),
    )
    source = NativeCurveSource(atlas, "r1")
    scope = MeshingScope(
        atlas.source_id,
        "r1",
        MeshingEntityKind.GEOMETRY,
        1,
        "curves",
        np.asarray((selected,), dtype=np.int64),
    )
    specification = CurveMeshingSpec(
        CellMeshingTarget(1, 3, CellFamilyPolicy(required=("interval",))),
        scope,
        size_controls=(
            UniformSizeControl(scope, 0.04, strength=SizeControlStrength.SOFT),
        ),
    )
    return source, specification


@pytest.mark.parametrize(
    ("order", "selected"),
    (((0, 1, 0), 0), ((1, 0), 1), ((1,), 0)),
    ids=("repeated-chart", "reordered-chart", "changed-coefficients"),
)
def test_density_targets_follow_selected_chart_and_coefficients(
    order: tuple[int, ...], selected: int
) -> None:
    source, specification = _curve_request(order, selected)
    result = (
        NativeMeshingProvider(NativeMeshingOptions("curve_arc_length"))
        .plan(
            source,
            specification,
            coordinate_contract=phx.SpatialCoordinateContract.si(),
        )
        .execute()
    )
    assert result.certification is not None and result.certification.passed
    assert result.compliance.passed
    association = result.associations[0]
    assert association.parameters is not None
    middles = np.sort(np.asarray(association.parameters)[:, 0])
    # Published interval parameters are endpoint averages. Reconstruct their
    # ordered endpoint sequence independently, starting at the chart origin.
    roots = np.empty(middles.size + 1, dtype=np.float64)
    roots[0] = 0.0
    for row, middle in enumerate(middles):
        roots[row + 1] = 2.0 * middle - roots[row]
    assert np.all((roots[1:-1] > 0.0) & (roots[1:-1] < 1.0))
    assert np.all(np.diff(roots) > 0.0)
    selected_source = order[selected]
    np.testing.assert_allclose(
        _arc_integral(_POINTS[selected_source], _WEIGHTS[selected_source], roots)
        / _arc_integral(
            _POINTS[selected_source], _WEIGHTS[selected_source], np.asarray(1.0)
        ),
        np.linspace(0.0, 1.0, roots.size),
        rtol=0.0,
        atol=_DENSITY_FRACTION_TOLERANCE,
    )


@pytest.fixture(scope="module")
def curve_result() -> tuple[NativeCurveSource, CellMeshingResult]:
    source, specification = _curve_request((0,), 0)
    result = (
        NativeMeshingProvider(NativeMeshingOptions("curve_arc_length"))
        .plan(
            source,
            specification,
            coordinate_contract=phx.SpatialCoordinateContract.si(),
        )
        .execute()
    )
    return source, result


def test_publication_refuses_full_retained_data(
    curve_result: tuple[NativeCurveSource, CellMeshingResult],
) -> None:
    source, result = curve_result
    carrier_bytes = result.mesh.coordinates.nbytes + sum(
        block.vertices.nbytes for block in result.mesh.blocks
    )
    limits = MeshingLimits(maximum_data_bytes=carrier_bytes + 1)
    request = NativeCertificationRequest(
        MeshCertificationSchedule("curve"),
        source.source_id,
        source.source_revision,
        limits,
    )
    with (
        NativeExecutionScope(limits.native_limits()) as scope,
        scope.host_workspace() as workspace,
    ):
        with pytest.raises(
            MeshingFailure, match="publication.retained_data_bytes"
        ) as refused:
            publish_native_result(
                result.mesh,
                result.coordinate_contract,
                result.compliance,
                (),
                result.provider,
                {"kind": "curve-retained-data-refusal"},
                request,
                policy=publication_policy(
                    "curve",
                    audit_policy=CellMeshAuditPolicy(require_complete_association=True),
                ),
                organization=NativePublicationOrganization(
                    associations=result.associations
                ),
                execution=NativePublicationExecution(scope, workspace),
                geometry=result.geometry,
            )
    assert refused.value.category is MeshingFailureCategory.RESOURCE_EXHAUSTED
    assert refused.value.evidence.stage == "canonicalization"
    assert (
        dict(refused.value.evidence.requested)["maximum_data_bytes"]
        == limits.maximum_data_bytes
    )
    assert (
        dict(refused.value.evidence.achieved)["retained_data_bytes"]
        > limits.maximum_data_bytes
    )


def test_publication_checks_policy_kind(
    curve_result: tuple[NativeCurveSource, CellMeshingResult],
) -> None:
    source, result = curve_result
    limits = MeshingLimits()
    request = NativeCertificationRequest(
        MeshCertificationSchedule("curve"),
        source.source_id,
        source.source_revision,
        limits,
    )
    with (
        NativeExecutionScope(limits.native_limits()) as scope,
        scope.host_workspace() as workspace,
    ):
        with pytest.raises(TypeError, match="policy"):
            publish_native_result(
                result.mesh,
                result.coordinate_contract,
                result.compliance,
                (),
                result.provider,
                {"kind": "curve-policy-kind-refusal"},
                request,
                policy="curve",  # ty: ignore[invalid-argument-type]
                execution=NativePublicationExecution(scope, workspace),
                geometry=result.geometry,
            )
