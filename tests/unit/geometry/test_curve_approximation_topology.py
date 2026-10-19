"""Continuous branch topology, rational jets, and trim separation contracts."""

from collections.abc import Callable

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax._fingerprint import array_tree_fingerprint, canonical_fingerprint
from phydrax.geometry.brep import (
    BSplineCurve,
    CurveTrimSegment,
    IntersectionCurve,
    LineCurve,
)
from phydrax.geometry.brep._curve_approximation_topology import (
    _fitted_interval,
    BranchApproximationTopologyResourceError,
    BranchTrimHomotopyTube,
    certify_branch_approximation_topology,
    certify_branch_trim_separation,
    TrimCarrier,
)
from phydrax.interchange import CadCurveApproximation, CadCurveFitPolicy
from tests._support.cad_models import native_intersection_cap


@pytest.fixture(scope="module")
def source() -> IntersectionCurve:
    model = native_intersection_cap(realize=False)
    assert model.geometry is not None
    branch = model.geometry.curves[0]
    assert isinstance(branch, IntersectionCurve)
    return branch


@pytest.fixture(scope="module")
def approximation(source: IntersectionCurve) -> CadCurveApproximation:
    # This policy is the existing unit sphere/plane export qualification policy;
    # it is not a tolerance for the assertions below.
    return phx.interchange.approximate_intersection_curve(
        source,
        CadCurveFitPolicy(tolerance=1e-7, maximum_control_points=512, check_samples=1025),
    )


def test_restricted_rational_derivative_encloses_closed_form_parabola() -> None:
    curve = BSplineCurve(
        jnp.asarray(((0.0, 0.0), (0.5, 1.0), (1.0, 0.0)), dtype=jnp.float64),
        jnp.ones(3, dtype=jnp.float64),
        jnp.asarray((0.0, 0.0, 0.0, 1.0, 1.0, 1.0), dtype=jnp.float64),
        2,
    )
    box, derivative = _fitted_interval(curve.bezier_pieces(), 0.125, 0.875)
    # The independent source is (t, 2t(1-t)); derivative (1, 2-4t).
    expected_lower = np.asarray((1.0, -1.5))
    expected_upper = np.asarray((1.0, 1.5))
    assert np.all(derivative[0] <= expected_lower)
    assert np.all(derivative[1] >= expected_upper)
    # At most 32 elementary outward-rounded operations have magnitude <= 2 in
    # this degree-two unit-weight jet; 128 eps covers both endpoint streams.
    rounding_bound = 128 * np.finfo(np.float64).eps
    np.testing.assert_allclose(
        derivative[0], expected_lower, rtol=0.0, atol=rounding_bound
    )
    np.testing.assert_allclose(
        derivative[1], expected_upper, rtol=0.0, atol=rounding_bound
    )
    assert np.all(box[0] <= (0.125, 0.21875))
    assert np.all(box[1] >= (0.875, 0.5))


def test_topology_records_incidence_and_immutable_interval_arrays(
    source: IntersectionCurve, approximation: CadCurveApproximation
) -> None:
    evidence = approximation.topology_evidence
    assert evidence.source_branch_id == source.branch_id
    assert evidence.cells == evidence.subdivision_cells + evidence.pair_checks
    assert evidence.trim_closure_kinds == ("open-native-period-lift", "closed")
    np.testing.assert_array_equal(evidence.root_references, source.node_references)
    np.testing.assert_array_equal(evidence.root_period_shifts, source.node_period_shifts)
    assert evidence.homotopy_boxes.shape == (evidence.parameter_cells.shape[0], 2, 7)
    with pytest.raises(ValueError, match="WRITEABLE"):
        evidence.homotopy_boxes.setflags(write=True)


def test_topology_byte_refusal_retains_unconsumed_evidence(
    source: IntersectionCurve, approximation: CadCurveApproximation
) -> None:
    with pytest.raises(
        BranchApproximationTopologyResourceError, match="maximum_bytes"
    ) as refusal:
        certify_branch_approximation_topology(
            source,
            approximation.curve,
            approximation.first_pcurve,
            approximation.second_pcurve,
            maximum_cells=100_000,
            maximum_depth=32,
            maximum_bytes=1,
        )
    assert refusal.value.cells == 0
    assert refusal.value.maximum_depth_used == 0
    assert refusal.value.bytes_used == 0


def test_topology_cell_refusal_retains_extraction_work(
    source: IntersectionCurve, approximation: CadCurveApproximation
) -> None:
    with pytest.raises(
        BranchApproximationTopologyResourceError, match="maximum_cells"
    ) as refusal:
        certify_branch_approximation_topology(
            source,
            approximation.curve,
            approximation.first_pcurve,
            approximation.second_pcurve,
            maximum_cells=2,
            maximum_depth=32,
            maximum_bytes=1 << 30,
        )
    assert refusal.value.cells == 2
    assert refusal.value.maximum_depth_used == 0
    assert refusal.value.bytes_used > 0


def _clear_curve(_approximation: CadCurveApproximation) -> TrimCarrier:
    return LineCurve((0.5, 0.5), (0.01, 0.0))


def _clear_segment(_approximation: CadCurveApproximation) -> TrimCarrier:
    return CurveTrimSegment(LineCurve((0.5, 0.5), (0.01, 0.0)), 0.0, 1.0)


@pytest.mark.parametrize(
    "build", (_clear_curve, _clear_segment), ids=("curve", "trim-segment")
)
def test_supported_trim_carriers_have_positive_separation(
    approximation: CadCurveApproximation,
    build: Callable[[CadCurveApproximation], TrimCarrier],
) -> None:
    evidence = certify_branch_trim_separation(
        approximation.topology_evidence,
        build(approximation),
        side="second",
        first=0.0,
        last=1.0,
        maximum_cells=100_000,
        maximum_depth=32,
        maximum_bytes=1 << 30,
    )
    assert evidence.separation_lower_bound > 0.0
    assert evidence.cells > 0
    assert np.all(evidence.interval_evidence[:, -1] > 0.0)
    with pytest.raises(ValueError, match="WRITEABLE"):
        evidence.interval_evidence.setflags(write=True)


def test_topology_keeps_source_and_fitted_carrier_identity_payloads(
    source: IntersectionCurve, approximation: CadCurveApproximation
) -> None:
    carriers = (
        approximation.curve,
        approximation.first_pcurve,
        approximation.second_pcurve,
    )
    payload = {
        "types": [type(carrier).__name__ for carrier in carriers],
        "degrees": [carrier.degree for carrier in carriers],
        "arrays": array_tree_fingerprint(carriers),
    }
    evidence = approximation.topology_evidence
    assert evidence.source_identity == canonical_fingerprint(source.payload())
    assert evidence.approximation_identity == canonical_fingerprint(payload)


def test_homotopy_tube_enclosure_retains_every_overlapping_cell(
    source: IntersectionCurve, approximation: CadCurveApproximation
) -> None:
    evidence = approximation.topology_evidence
    tube = BranchTrimHomotopyTube(evidence, "second", source.parameter_domain)
    boxes = evidence.homotopy_boxes[:, :, 5:7]
    expected = np.stack((np.min(boxes[:, 0], axis=0), np.max(boxes[:, 1], axis=0)))
    np.testing.assert_array_equal(tube.enclosure(*source.parameter_domain), expected)


def test_contacting_homotopy_tube_refuses_with_depth_evidence(
    source: IntersectionCurve, approximation: CadCurveApproximation
) -> None:
    tube = BranchTrimHomotopyTube(
        approximation.topology_evidence, "second", source.parameter_domain
    )
    with pytest.raises(
        BranchApproximationTopologyResourceError, match="maximum_depth"
    ) as refusal:
        certify_branch_trim_separation(
            approximation.topology_evidence,
            tube,
            side="second",
            maximum_cells=100_000,
            maximum_depth=0,
            maximum_bytes=1 << 30,
        )
    assert refusal.value.cells == 2
    assert refusal.value.maximum_depth_used == 0
    assert refusal.value.bytes_used >= 8192 + tube.enclosure_workspace_bytes


def test_trim_side_refuses_unknown_selector(approximation: CadCurveApproximation) -> None:
    with pytest.raises(ValueError, match="BranchTrimSeparation.side"):
        certify_branch_trim_separation(
            approximation.topology_evidence,
            _clear_curve(approximation),
            side="third",  # ty: ignore[invalid-argument-type]
            maximum_cells=100_000,
            maximum_depth=0,
            maximum_bytes=1 << 30,
        )


def test_trim_boundary_refuses_unsupported_object_kind(
    approximation: CadCurveApproximation,
) -> None:
    with pytest.raises(TypeError, match="argument 'trim'"):
        certify_branch_trim_separation(
            approximation.topology_evidence,
            None,  # ty: ignore[invalid-argument-type]
            side="second",
            maximum_cells=100_000,
            maximum_depth=0,
            maximum_bytes=1 << 30,
        )


def test_unbounded_trim_requires_explicit_parameter_range(
    approximation: CadCurveApproximation,
) -> None:
    with pytest.raises(ValueError, match="BranchTrimSeparation.range"):
        certify_branch_trim_separation(
            approximation.topology_evidence,
            _clear_curve(approximation),
            side="second",
            maximum_cells=100_000,
            maximum_depth=0,
            maximum_bytes=1 << 30,
        )
