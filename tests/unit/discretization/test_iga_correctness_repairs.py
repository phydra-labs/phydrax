#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization.iga import BSplineGrid
from phydrax.discretization.iga._adaptive import AdaptiveDesignEpoch
from phydrax.discretization.iga._basis import (
    IsogeometricFieldSpec,
    IsogeometricQuadraturePolicy,
    TensorSplineBasisSpec,
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
from phydrax.discretization.iga._tspline import (
    LocalKnotVector2D,
    TAnchor2D,
    TMesh2D,
)


class _LineMap(phx.geometry.BoundaryMap):
    @property
    def num_charts(self):
        return 1

    @property
    def reference_dimension(self):
        return 1

    @property
    def ambient_dimension(self):
        return 2

    def map(self, chart_indices, reference, /):
        del chart_indices
        return jnp.concatenate((reference, jnp.zeros_like(reference)), axis=-1)

    def jacobian(self, chart_indices, reference, /):
        del reference
        return jnp.ones(jnp.asarray(chart_indices).shape, dtype=jnp.float64)


def _interface_and_certificate():
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


def test_field_specific_facet_actions_use_the_field_basis_on_the_common_overlay():
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
        kernel_mode="dense",
    )

    assert prepared.exterior_facet_domain.entity_indices.size == 8
    assert len(regions) == 4
    assert all(region.field_gathers[0].shape[1] == 4 for region in regions)
    assert all(region.reference_actions[0].local_width == 4 for region in regions)
    assert all(region.geometry_actions.tensor_plan.local_size == 9 for region in regions)


def test_anisotropic_direct_realization_uses_each_axis_degree():
    quadratic = BSplineGrid.open_uniform(2, 1)
    cubic = BSplineGrid.open_uniform(3, 1)
    basis = TensorSplineBasisSpec((quadratic, cubic), axis_names=("xi", "eta"))
    realization = DirectTensorRealization(basis, SplineSpanTopology(basis))

    assert realization.cell_gathers.shape == (1, 12)
    assert np.unique(np.asarray(realization.cell_gathers)).size == 12


def test_partial_thb_basis_fails_partition_and_foreign_certificate_is_rejected():
    partial = THBHierarchy(
        (THBLevel(0, "partial", (True,), (True, False)),),
        (),
    )
    partial_certificate = partial.certify()
    assert not partial_certificate.passed
    assert partial_certificate.partition_defect == 1.0

    source = THBHierarchy(
        (THBLevel(0, "source", (True,), (True, True)),),
        (),
    )
    target = THBHierarchy(
        (THBLevel(0, "target", (True,), (True, True, True)),),
        (),
    )
    other = THBHierarchy(
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


def test_mortar_plan_rejects_a_matrix_other_than_the_certified_coupling():
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


def test_tspline_anchor_ids_must_be_dense_coefficient_indices():
    knots = LocalKnotVector2D(
        (0.0, 0.0, 0.0, 0.0, 1.0),
        (0.0, 0.0, 0.0, 0.0, 1.0),
    )
    anchor = TAnchor2D(10, (0.0, 0.0), knots)

    with pytest.raises(ValueError, match="anchor IDs must be exactly"):
        TMesh2D(((0.0, 1.0, 0.0, 1.0),), (anchor,))
