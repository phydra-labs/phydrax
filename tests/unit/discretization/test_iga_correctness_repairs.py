#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


import math
from fractions import Fraction
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax._interpolation._tspline import (
    LocalKnotVector2D,
    TAnchor2D,
    TMesh2D,
)
from phydrax.discretization.iga import BSplineGrid
from phydrax.discretization.iga._adaptive import AdaptiveDesignEpoch
from phydrax.discretization.iga._basis import (
    IsogeometricFieldSpec,
    IsogeometricQuadraturePolicy,
    TensorSplineBasisSpec,
)
from phydrax.discretization.iga._certificate import (
    _p_matrix_proved,
    _positive_power_interval,
    CertificateDisposition,
    certify_tensor_nurbs,
    IntervalBound,
    TensorNURBSCertificatePolicy,
)
from phydrax.discretization.iga._geometry import NURBSGeometryState
from phydrax.discretization.iga._interfaces import (
    certify_mortar_inf_sup,
    InterfaceCertificate,
    InterfaceParameterMap,
    InterfaceQualificationEvidence,
    MortarCrosspointPlan,
    MortarInterfacePlan,
    PatchInterface,
)
from phydrax.discretization.iga._plan import IsogeometricPlan
from phydrax.discretization.iga._realization import DirectTensorRealization
from phydrax.discretization.iga._thb import THBHierarchy, THBLevel
from phydrax.discretization.iga._topology import SplineSpanTopology
from phydrax.discretization.iga._volume import TensorNURBSVolume


class _LineMap(phx.geometry.BoundaryMap):
    @property
    def num_charts(self) -> int:
        return 1

    @property
    def reference_dimension(self) -> int:
        return 1

    @property
    def ambient_dimension(self) -> int:
        return 2

    def map(self, chart_indices: Any, reference: Any, /) -> Any:
        del chart_indices
        return jnp.concatenate((reference, jnp.zeros_like(reference)), axis=-1)

    def jacobian(self, chart_indices: Any, reference: Any, /) -> Any:
        del reference
        return jnp.ones(jnp.asarray(chart_indices).shape, dtype=jnp.float64)


def _interface_and_certificate() -> Any:
    atlas = phx.geometry.BoundaryAtlas(
        _LineMap(),
        source_entity_ids=jnp.asarray((0,), dtype=jnp.int32),
        source_id="line",
    )
    identity = InterfaceParameterMap.identity(1)
    interface = PatchInterface(
        "left",
        "right",
        atlas,
        atlas,
        left_chart=0,
        right_chart=0,
        left_parameter_map=identity,
        right_parameter_map=identity,
        orientation=1,
    )
    evidence = InterfaceQualificationEvidence(
        jnp.asarray(0.0),
        jnp.asarray(0.0),
        jnp.asarray(1.0),
        jnp.asarray(1.0),
        jnp.asarray(1.0),
        jnp.asarray(1.0),
        2,
        "interface-evidence",
    )
    certificate = InterfaceCertificate(
        evidence,
        interface.interface_id,
        1.0e-9,
        1.0e-9,
        1.0e-8,
        1.0e-10,
        False,
        "interface-certificate",
    )
    return interface, certificate


def test_iga_correctness_repairs_scenario_1() -> None:
    geometry_grid = BSplineGrid.open_uniform(2, 1)
    field_grid = BSplineGrid.open_uniform(1, 2)
    geometry_basis = TensorSplineBasisSpec(
        (geometry_grid, geometry_grid), axis_names=("xi", "eta")
    )
    field_basis = TensorSplineBasisSpec(
        (field_grid, field_grid), axis_names=("xi", "eta")
    )
    coordinates = geometry_grid.greville_abscissae
    xx, yy = jnp.meshgrid(coordinates, coordinates, indexing="ij")
    geometry = NURBSGeometryState(
        jnp.stack((xx, yy), axis=-1),
        jnp.ones(geometry_basis.control_shape),
    )
    plan = IsogeometricPlan(
        geometry_basis,
        geometry,
        IsogeometricFieldSpec("u", field_basis),
        quadrature_policy=IsogeometricQuadraturePolicy(3),
    )
    prepared = plan.prepare()

    regions = prepared.prepare_local_regions(
        prepared.exterior_facet_domain,
        field_names=("u",),
        maximum_derivative_order=1,
        kernel_mode="sum_factorized",
    )

    assert prepared.exterior_facet_domain.entity_indices.size == 8
    assert len(regions) == 4
    assert all(region.field_gathers[0].shape[1] == 4 for region in regions)
    assert all(region.reference_actions[0].local_width == 4 for region in regions)
    # ty: ignore[unresolved-attribute]
    assert all(region.geometry_actions.tensor_plan.local_size == 9 for region in regions)
    quadratic = BSplineGrid.open_uniform(2, 1)
    cubic = BSplineGrid.open_uniform(3, 1)
    basis = TensorSplineBasisSpec((quadratic, cubic), axis_names=("xi", "eta"))
    realization = DirectTensorRealization(basis, SplineSpanTopology(basis))

    assert realization.cell_gathers.shape == (1, 12)
    assert np.unique(np.asarray(realization.cell_gathers)).size == 12
    partial = THBHierarchy(
        # ty: ignore[invalid-argument-type]
        (THBLevel(0, "partial", (True,), (True, False)),),
        (),
    )
    partial_certificate = partial.certify()
    assert not partial_certificate.passed
    assert partial_certificate.partition_defect == 1.0

    source = THBHierarchy(
        # ty: ignore[invalid-argument-type]
        (THBLevel(0, "source", (True,), (True, True)),),
        (),
    )
    target = THBHierarchy(
        # ty: ignore[invalid-argument-type]
        (THBLevel(0, "target", (True,), (True, True, True)),),
        (),
    )
    other = THBHierarchy(
        # ty: ignore[invalid-argument-type]
        (THBLevel(0, "other", (True,), (True, True, True)),),
        (),
    )
    foreign_certificate = other.certify()
    assert foreign_certificate.passed
    epoch = AdaptiveDesignEpoch(0, "source-plan", source, "design")

    with pytest.raises(ValueError, match="certified THB hierarchy"):
        epoch.transition(
            "target-plan",
            target,
            foreign_certificate,
            "next-design",
            completed_iterations=1,
            maximum_transitions=2,
        )


def test_iga_correctness_repairs_scenario_2() -> None:
    interface, certificate = _interface_and_certificate()
    crosspoints = MortarCrosspointPlan(2, 2, owner_patch_id="left")
    coupling = np.eye(2)
    stability = certify_mortar_inf_sup(
        certificate,
        coupling,
        crosspoints,
        required_lower_bound=0.5,
    )

    with pytest.raises(ValueError, match="does not match its inf-sup certificate"):
        MortarInterfacePlan(
            interface,
            certificate,
            np.asarray(((1.0, 0.0), (0.0, 0.0))),
            stability,
        )
    knots = LocalKnotVector2D(
        (0.0, 0.0, 0.0, 0.0, 1.0),
        (0.0, 0.0, 0.0, 0.0, 1.0),
    )
    anchor = TAnchor2D(10, (0.0, 0.0), knots)
    mesh = TMesh2D(((0.0, 1.0, 0.0, 1.0),), (anchor,))
    assert mesh.anchors[0].anchor_id == 10
    assert mesh.anchor_row(10) == 0
    with pytest.raises(KeyError, match="Unknown T-spline anchor ID"):
        mesh.anchor_row(11)
    with pytest.raises(ValueError, match="anchor IDs must be unique"):
        TMesh2D(
            ((0.0, 1.0, 0.0, 1.0),),
            (anchor, TAnchor2D(10, (0.0, 0.0), knots)),
        )
    foreign_knots = LocalKnotVector2D(
        (0.0, 0.0, 1.0, 1.0, 1.0),
        (0.0, 0.0, 0.0, 0.0, 1.0),
    )
    with pytest.raises(ValueError, match="do not follow the T-mesh topology"):
        TMesh2D(
            ((0.0, 1.0, 0.0, 1.0),),
            (TAnchor2D(11, (0.0, 0.0), foreign_knots),),
        )


@pytest.mark.parametrize(
    ("lower", "upper", "exponent"),
    (
        ("0x1.0000000000001p+0", "0x1.0000000000001p+0", 4),
        ("0x1.0000000000001p+0", "0x1.0000000000001p+0", 6),
        ("0x1.0000000000001p+0", "0x1.0000000000001p+0", 8),
        ("0x1.1ce0b608f6030p+0", "0x1.1ce0b608f6031p+0", 8),
    ),
)
def test_weight_power_interval_encloses_exact_powers(
    lower: str, upper: str, exponent: int
) -> None:
    bound = IntervalBound(float.fromhex(lower), float.fromhex(upper))

    power = _positive_power_interval(bound, exponent)

    assert Fraction(power.lower) <= Fraction(bound.lower) ** exponent
    assert Fraction(bound.upper) ** exponent <= Fraction(power.upper)
    assert math.isfinite(power.lower) and math.isfinite(power.upper)


# Integer Jacobian columns near 2^27 whose determinant is exactly -1: the ~2^54
# Bernstein products cancel completely in the measure coefficients.
_CANCELLING_FIRST = (
    float.fromhex("0x1.9b57240000000p+26"),
    float.fromhex("0x1.7be7714000000p+26"),
)
_CANCELLING_SECOND = (
    float.fromhex("0x1.cee7334000000p+26"),
    float.fromhex("0x1.ab86b30000000p+26"),
)


def _affine_volume(
    first: tuple[float, float], second: tuple[float, float]
) -> TensorNURBSVolume:
    grid = BSplineGrid.open_uniform(1, 1, interval=(0.0, 1.0))
    control = np.zeros((2, 2, 2), dtype=np.float64)
    control[1, 0] = first
    control[0, 1] = second
    control[1, 1] = (first[0] + second[0], first[1] + second[1])
    geometry = NURBSGeometryState(control, np.ones((2, 2), dtype=np.float64))
    plan = IsogeometricPlan.isoparametric(
        (grid, grid), geometry, quadrature_policy=IsogeometricQuadraturePolicy(2)
    )
    return TensorNURBSVolume("cancelling-affine", plan.basis, geometry)


def _affine_determinant(
    first: tuple[float, float], second: tuple[float, float]
) -> Fraction:
    return Fraction(first[0]) * Fraction(second[1]) - Fraction(first[1]) * Fraction(
        second[0]
    )


def test_tensor_nurbs_jacobian_enclosure_survives_cancellation() -> None:
    exact = _affine_determinant(_CANCELLING_FIRST, _CANCELLING_SECOND)

    certificate = certify_tensor_nurbs(
        _affine_volume(_CANCELLING_FIRST, _CANCELLING_SECOND),
        policy=TensorNURBSCertificatePolicy(maximum_subdivision_depth=0),
    )

    assert exact == -1
    assert certificate.disposition is not CertificateDisposition.PASS
    assert certificate.cells
    for cell in certificate.cells:
        measure = cell.jacobian_measure
        assert Fraction(measure.lower) <= exact <= Fraction(measure.upper)


def test_tensor_nurbs_cancelling_jacobian_is_certified_negative() -> None:
    exact = _affine_determinant(_CANCELLING_FIRST, _CANCELLING_SECOND)

    certificate = certify_tensor_nurbs(
        _affine_volume(_CANCELLING_FIRST, _CANCELLING_SECOND),
        policy=TensorNURBSCertificatePolicy(maximum_subdivision_depth=0),
    )

    assert exact == -1
    assert certificate.disposition is CertificateDisposition.COUNTEREXAMPLE
    counterexample = certificate.counterexample
    assert counterexample is not None
    assert counterexample.code == "nonpositive-jacobian-measure"
    assert counterexample.cell is not None
    measure = counterexample.cell.jacobian_measure
    assert Fraction(measure.lower) <= exact <= Fraction(measure.upper) < 0
    assert Fraction(measure.upper) - Fraction(measure.lower) <= Fraction(1, 2**50)


def test_p_matrix_minors_survive_cancellation() -> None:
    # Swapping the columns makes the exact determinant +1 with positive diagonal.
    volume = _affine_volume(_CANCELLING_SECOND, _CANCELLING_FIRST)

    proved, diagnostic = _p_matrix_proved(volume, TensorNURBSCertificatePolicy())

    assert _affine_determinant(_CANCELLING_SECOND, _CANCELLING_FIRST) == 1
    assert proved
    assert diagnostic is None
