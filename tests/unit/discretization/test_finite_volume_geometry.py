#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax._sharp_measures import QualifiedSharpGeometry
from phydrax.discretization.finite_volume._mac_cut_cell import (
    MACDiffuseSDFGeometryPlan,
)
from phydrax.discretization.finite_volume._mac_sharp_geometry import (
    MACExactSDFMeasurePlan,
)
from phydrax.geometry._certificate import (
    exact_signed_distance_certificate,
    ExactSDFEnclosureCertificate,
)


def _cell_grid(shape, *, periodic=None):
    periodic = (False,) * len(shape) if periodic is None else tuple(periodic)
    return phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformCellAxisSpec(count, periodic=periodic[axis])
            for axis, count in enumerate(shape)
        ),
        axis_names=tuple("xyz"[: len(shape)]),
    ).prepare(jnp.stack((jnp.zeros(len(shape)), jnp.ones(len(shape)))))


def test_structured_finite_volume_has_exact_cell_and_face_geometry():
    grid = _cell_grid((4, 3))
    discretization = phx.discretization.FiniteVolumePlan(
        grid,
        component_names=("density", "energy"),
    ).prepare()

    assert discretization.cell_shape == (4, 3)
    assert discretization.state_shape == (4, 3, 2)
    assert tuple(layout.shape for layout in discretization.face_layouts) == (
        (5, 3),
        (4, 4),
    )
    np.testing.assert_allclose(discretization.cell_volumes, 1.0 / 12.0)
    np.testing.assert_allclose(discretization.face_measures[0], 1.0 / 3.0)
    np.testing.assert_allclose(discretization.face_measures[1], 1.0 / 4.0)
    np.testing.assert_allclose(jnp.sum(discretization.cell_volumes), 1.0)
    assert discretization.cell_space.representation == "cell_average"
    assert all(
        space.representation == "flux_moment" for space in discretization.face_spaces
    )


def test_periodic_faces_are_unique_and_one_dimensional_measure_is_one():
    grid = _cell_grid((7,), periodic=(True,))
    discretization = phx.discretization.FiniteVolumePlan(grid).prepare()

    assert discretization.face_layouts[0].shape == (7,)
    np.testing.assert_allclose(discretization.face_measures[0], jnp.ones((7,)))
    np.testing.assert_allclose(discretization.cell_volumes, jnp.full((7,), 1.0 / 7.0))


def test_interval_quadrature_weights_define_nonuniform_cell_edges():
    axis = phx.discretization.AxisDiscretization(
        nodes=jnp.asarray([0.1, 0.45, 0.85]),
        quad_weights=jnp.asarray([0.2, 0.5, 0.3]),
        basis="uniform",
        domain=phx.discretization.AxisDomain.interval(0.0, 1.0),
        primary_entity="interval",
        lower_endpoint_included=False,
        upper_endpoint_included=False,
    )
    grid = phx.discretization.PreparedTensorGrid((axis,), axis_names=("x",))
    discretization = phx.discretization.FiniteVolumePlan(grid).prepare()

    np.testing.assert_allclose(
        grid.structured_axes[0].point_coordinates, [0.0, 0.2, 0.7, 1.0]
    )
    np.testing.assert_allclose(discretization.cell_volumes, [0.2, 0.5, 0.3])
    np.testing.assert_allclose(discretization.cell_centers[:, 0], [0.1, 0.45, 0.85])


def test_nonuniform_cell_axis_rejects_inconsistent_centers():
    axis = phx.discretization.AxisDiscretization(
        nodes=jnp.asarray([0.1, 0.4]),
        quad_weights=jnp.asarray([0.2, 0.8]),
        basis="uniform",
        domain=phx.discretization.AxisDomain.interval(0.0, 1.0),
        primary_entity="interval",
        lower_endpoint_included=False,
        upper_endpoint_included=False,
    )
    with pytest.raises(ValueError, match="cell centers"):
        phx.discretization.PreparedTensorGrid((axis,), axis_names=("x",))


def test_finite_volume_rejects_point_primary_support_and_duplicate_components():
    point_grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformAxisSpec(8),), axis_names=("x",)
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    with pytest.raises(ValueError, match="interval-primary"):
        phx.discretization.FiniteVolumePlan(point_grid)

    with pytest.raises(ValueError, match="component_names"):
        phx.discretization.FiniteVolumePlan(_cell_grid((4,)), component_names=("u", "u"))


def test_exact_sdf_enclosure_carries_absolute_bounds_and_source_identity():
    discretization = phx.discretization.FiniteVolumePlan(_cell_grid((4, 4))).prepare()
    operators = phx.discretization.MACOperatorPlan(discretization).prepare()

    def plane(points, time, args):
        del args
        return points[..., 0] - (0.35 + time)

    certificate = ExactSDFEnclosureCertificate(
        exact_signed_distance_certificate(smooth=True)
    )
    coarse = MACExactSDFMeasurePlan(
        operators,
        plane,
        certificate,
        source_id="translating-plane",
        subdivisions=4,
    ).prepare(0.0)
    fine = MACExactSDFMeasurePlan(
        operators,
        plane,
        certificate,
        source_id="translating-plane",
        subdivisions=8,
    ).prepare(0.0)

    assert isinstance(fine, QualifiedSharpGeometry)
    assert fine.accepted
    assert fine.source_id == "translating-plane"
    assert fine.operator_id == operators.prepared_id
    assert fine.support_id == discretization.support.support_id
    assert fine.measure_fidelity.value == "certified_bounded_error"
    assert jnp.all(fine.cell_fluid_measure_lower <= fine.cell_fluid_measure)
    assert jnp.all(fine.cell_fluid_measure <= fine.cell_fluid_measure_upper)
    assert jnp.all(fine.evidence.cell_bound_width <= coarse.evidence.cell_bound_width)
    analytic_cut_volume = 0.15 * 0.25
    assert fine.cell_fluid_measure_lower[1, 0] <= analytic_cut_volume
    assert analytic_cut_volume <= fine.cell_fluid_measure_upper[1, 0]


def test_exact_sdf_refresh_rejects_inconsistent_swept_rate_atomically():
    discretization = phx.discretization.FiniteVolumePlan(_cell_grid((4, 4))).prepare()
    operators = phx.discretization.MACOperatorPlan(discretization).prepare()

    def plane(points, time, args):
        del args
        return points[..., 0] - (0.35 + time)

    plan = MACExactSDFMeasurePlan(
        operators,
        plane,
        ExactSDFEnclosureCertificate(exact_signed_distance_certificate(smooth=True)),
        source_id="moving-plane-without-swept-flux",
        subdivisions=8,
    )
    initial = plan.prepare(0.0)
    refreshed = plan.refresh(initial, 0.2, 0.2)

    assert not refreshed.accepted
    assert refreshed.refresh_required
    np.testing.assert_allclose(
        refreshed.geometry.cell_fluid_measure, initial.cell_fluid_measure
    )
    assert refreshed.geometry.epoch == initial.epoch


def test_diffuse_sdf_ramp_is_honestly_unqualified():
    discretization = phx.discretization.FiniteVolumePlan(_cell_grid((4, 4))).prepare()
    operators = phx.discretization.MACOperatorPlan(discretization).prepare()

    def plane(points, time, args):
        del time, args
        return points[..., 0] - 0.35

    diffuse = MACDiffuseSDFGeometryPlan(
        operators,
        plane,
        lambda points, time, args: jnp.zeros_like(points),
        field_id="diffuse-plane",
        interface_width=0.1,
    ).evaluate(0.0)

    assert diffuse.successful
    assert not isinstance(diffuse, QualifiedSharpGeometry)
