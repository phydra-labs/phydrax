#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax as phx
from phydrax.discretization.fem import FiniteElementDiscretization
from phydrax.equations import FiniteElementExecutionContext
from phydrax.integration import CoordinateMeasure, CoordinateMeasurePlan


def _radial(
    kind: CoordinateMeasure, inner: float, outer: float
) -> FiniteElementDiscretization:
    mesh = phx.discretization.CellMesh(
        jnp.asarray([[inner], [0.5 * (inner + outer)], [outer]]),
        (
            phx.discretization.CellBlock(
                "radial", "interval", jnp.asarray([[0, 1], [1, 2]], dtype=jnp.int32)
            ),
        ),
    )
    return phx.discretization.FiniteElementPlan(
        mesh,
        phx.discretization.FiniteElementFieldSpec(
            "u", phx.discretization.lagrange_element("interval", 1)
        ),
        coordinate_measure=CoordinateMeasurePlan(kind, radial_axis=0),
    ).prepare()


def _rz(radial_axis: int = 0) -> FiniteElementDiscretization:
    points = jnp.asarray([[0.0, 0.0], [2.0, 0.0], [2.0, 3.0], [0.0, 3.0]])
    cells = jnp.asarray([[0, 1, 2], [0, 2, 3]], dtype=jnp.int32)
    if radial_axis == 1:
        points = points[:, ::-1]
        cells = cells[:, ::-1]
    mesh = phx.discretization.CellMesh.from_triangles(points, cells)
    return phx.discretization.FiniteElementPlan(
        mesh,
        phx.discretization.FiniteElementFieldSpec(
            "u", phx.discretization.lagrange_element("triangle", 1)
        ),
        coordinate_measure=CoordinateMeasurePlan(
            "axisymmetric-rz", radial_axis=radial_axis
        ),
    ).prepare()


@pytest.mark.parametrize(
    ("kind", "inner", "outer", "volume"),
    [
        ("cylindrical-radial", 0.0, 2.0, 4.0 * np.pi),
        ("cylindrical-radial", 1.0, 2.0, 3.0 * np.pi),
        ("spherical-radial", 0.0, 2.0, 32.0 * np.pi / 3.0),
        ("spherical-radial", 1.0, 2.0, 28.0 * np.pi / 3.0),
    ],
    ids=["solid-cylinder", "annulus", "solid-sphere", "spherical-shell"],
)
def test_reference_volume_and_consistent_axis_capacity(
    kind: CoordinateMeasure, inner: float, outer: float, volume: float
) -> None:
    discretization = _radial(kind, inner, outer)
    ones = jnp.ones((3,))
    capacity = discretization.mass.mv(ones)
    np.testing.assert_allclose(jnp.sum(capacity), volume, rtol=2.0e-6)
    np.testing.assert_allclose(
        jnp.sum(discretization.block_geometries[0][0].measure), volume, rtol=2.0e-6
    )
    # Only admitted P1 basis has this positive row-sum claim, including r=0.
    assert np.all(np.asarray(capacity) > 0.0)
    radial_field = discretization.dof_maps[0].dof_coordinates[:, 0]
    np.testing.assert_allclose(
        jnp.vdot(radial_field, discretization.stiffness.mv(radial_field)),
        volume,
        rtol=3.0e-6,
    )


@pytest.mark.parametrize("radial_axis", [0, 1])
def test_rz_mass_source_diffusion_and_explicit_radial_identity(radial_axis: int) -> None:
    discretization = _rz(radial_axis)
    ones = jnp.ones((4,))
    expected_volume = 12.0 * np.pi
    radial_field = discretization.dof_maps[0].dof_coordinates[:, radial_axis]
    actions = (
        phx.equations.MassAction("u"),
        phx.equations.SourceAction("u", 1.0),
        phx.equations.DiffusionAction("u"),
    )
    for action in actions:
        compiled = phx.equations.compile_finite_element_problem(
            phx.equations.FiniteElementForm(action.action_id, "u", (action,)),
            discretization,
        )
        if isinstance(action, phx.equations.DiffusionAction):
            result = compiled.full_residual(radial_field)
            observed = jnp.vdot(radial_field, result)
        else:
            observed = jnp.sum(compiled.full_residual(ones))
            if isinstance(action, phx.equations.SourceAction):
                observed = -observed
        np.testing.assert_allclose(observed, expected_volume, rtol=5.0e-6)
    np.testing.assert_array_equal(
        discretization.axis_dof_mask("u"), [True, False, False, True]
    )


@pytest.mark.parametrize("flux_kind", ["area", "divergence"])
def test_axisymmetric_exterior_area_and_divergence_theorem(flux_kind: str) -> None:
    discretization = _rz()

    def flux(
        traces: tuple[Array, ...],
        points: Array,
        weights: Array,
        normal: Array,
        context: FiniteElementExecutionContext,
    ) -> Array:
        del traces, weights, context
        return (
            jnp.ones(points.shape[:-1], dtype=points.dtype)
            if flux_kind == "area"
            else jnp.sum(points * normal, axis=-1)
        )

    action = phx.equations.ExteriorFacetAction(
        "u",
        ("u",),
        flux,
        domain=discretization.exterior_facet_domain,
        action_id=flux_kind,
    )
    compiled = phx.equations.compile_finite_element_problem(
        phx.equations.FiniteElementForm(flux_kind, "u", (action,)), discretization
    )
    observed = jnp.sum(compiled.full_residual(jnp.zeros((4,))))
    # Closed cylinder: lateral area + two caps; div_3D(r e_r + z e_z)=3.
    expected = 20.0 * np.pi if flux_kind == "area" else 36.0 * np.pi
    np.testing.assert_allclose(observed, expected, rtol=5.0e-6)


@pytest.mark.parametrize(
    ("kind", "power"), [("cylindrical-radial", 2), ("spherical-radial", 3)]
)
def test_fixed_topology_geometry_jvp(kind: CoordinateMeasure, power: int) -> None:
    discretization = _radial(kind, 1.0, 2.0)
    reference = discretization.default_runtime.coordinates

    def volume(scale: Array) -> Array:
        runtime = discretization.prepare_runtime(
            scale * reference, numeric_version="scaled"
        )
        mass, _ = discretization.assemble_field_operators("u", runtime)
        return jnp.sum(mass.mv(jnp.ones((3,))))

    primal, tangent = jax.jit(
        lambda scale: jax.jvp(volume, (scale,), (jnp.ones_like(scale),))
    )(jnp.asarray(1.2))
    baseline = np.pi * 3.0 if power == 2 else 4.0 * np.pi * 7.0 / 3.0
    np.testing.assert_allclose(primal, baseline * 1.2**power, rtol=3.0e-6)
    np.testing.assert_allclose(
        tangent, power * baseline * 1.2 ** (power - 1), rtol=4.0e-6
    )


def test_mortar_physical_area_mass_and_dynamic_radius_jvp() -> None:
    axis, weights = np.polynomial.legendre.leggauss(4)
    physical = np.stack((np.full_like(axis, 2.0), 1.5 * (axis + 1.0)), axis=-1)
    measure = CoordinateMeasurePlan("axisymmetric-rz", radial_axis=0, frame="current")
    mortar = phx.discretization.fem.serial_finite_element_mortar_plan(
        np.asarray([-1.0, 1.0]),
        np.asarray([-1.0, 1.0]),
        np.asarray([-1.0, 1.0]),
        axis,
        weights,
        left_physical_coordinates=physical,
        right_physical_coordinates=physical,
        coordinate_measure=np.full_like(axis, 1.5),
        rotational_measure=measure,
        declared_reproduction_degree=1,
    )
    expected_area = 12.0 * np.pi
    np.testing.assert_allclose(
        jnp.sum(mortar.physical_weights), expected_area, rtol=2.0e-6
    )
    np.testing.assert_allclose(jnp.sum(mortar.mortar_mass), expected_area, rtol=2.0e-6)
    left, right = mortar.conservative_flux_contributions(jnp.ones((4,)))
    np.testing.assert_allclose(jnp.sum(left), expected_area, rtol=2.0e-6)
    np.testing.assert_allclose(jnp.sum(right), -expected_area, rtol=2.0e-6)

    def area(radius: Array) -> Array:
        coordinates = jnp.asarray(physical).at[:, 0].set(radius)
        normals = jnp.broadcast_to(jnp.asarray([1.0, 0.0]), coordinates.shape)
        metric = phx.discretization.fem.FiniteElementMortarMetricData(
            coordinates,
            jnp.asarray(weights * 1.5),
            normals,
            -normals,
            metric_id="dynamic-cylinder",
            rotational_measure=measure,
            coordinate_frame="current",
        )
        return jnp.sum(metric.physical_weights)

    observed, derivative = jax.jit(
        lambda radius: jax.jvp(area, (radius,), (jnp.ones_like(radius),))
    )(jnp.asarray(2.0))
    np.testing.assert_allclose(observed, expected_area, rtol=2.0e-6)
    np.testing.assert_allclose(derivative, 6.0 * np.pi, rtol=2.0e-6)


@pytest.mark.parametrize("radius", [-0.1, np.nan, np.inf])
def test_invalid_reference_and_dynamic_geometry_fail_closed(radius: float) -> None:
    with pytest.raises(ValueError, match="finite|nonnegative"):
        _radial("cylindrical-radial", radius, 2.0)
    measure = CoordinateMeasurePlan("cylindrical-radial", radial_axis=0)
    weighted = jax.jit(
        lambda points: measure.weight(points, jnp.ones((1,)), frame="reference")
    )(jnp.asarray([[radius]]))
    assert np.isnan(np.asarray(weighted)[0])


def test_negative_node_is_refused_even_when_gauss_radii_are_positive() -> None:
    discretization = _radial("cylindrical-radial", 0.0, 2.0)
    coordinates = discretization.default_runtime.coordinates.at[0, 0].set(-0.01)
    geometry = discretization.evaluate_geometry("u", coordinates)[0]
    assert np.any(~np.isfinite(np.asarray(geometry.physical_weights)))


def test_nodal_axis_capacity_is_not_silently_zero() -> None:
    discretization = _radial("cylindrical-radial", 0.0, 2.0)
    action = phx.equations.MassAction(
        "u",
        rules={
            "radial": phx.integration.ReferenceIntervalRule(
                phx.integration.GaussLobattoLegendreRule(3)
            )
        },
    )
    compiled = phx.equations.compile_finite_element_problem(
        phx.equations.FiniteElementForm("axis-nodal-capacity", "u", (action,)),
        discretization,
    )
    assert np.any(~np.isfinite(np.asarray(compiled.full_residual(jnp.ones((3,))))))


def test_coordinate_frame_refusal_and_cartesian_unchanged() -> None:
    measure = CoordinateMeasurePlan(
        "cylindrical-radial", radial_axis=0, frame="reference"
    )
    with pytest.raises(ValueError, match="frame"):
        measure.weight(jnp.asarray([[1.0]]), jnp.asarray([1.0]), frame="current")
    discretization = _radial("cylindrical-radial", 0.0, 2.0)
    with pytest.raises(ValueError, match="frame"):
        discretization.prepare_runtime(
            numeric_version="wrong-frame", coordinate_frame="current"
        )
    mesh = _rz().mesh
    field = phx.discretization.FiniteElementFieldSpec(
        "u", phx.discretization.lagrange_element("triangle", 1)
    )
    plain = phx.discretization.FiniteElementPlan(mesh, field)
    cartesian = phx.discretization.FiniteElementPlan(
        mesh, field, coordinate_measure=CoordinateMeasurePlan("cartesian")
    )
    assert plain.plan_id == cartesian.plan_id
    np.testing.assert_array_equal(
        plain.prepare().mass.mv(jnp.ones((4,))),
        cartesian.prepare().mass.mv(jnp.ones((4,))),
    )
    np.testing.assert_allclose(
        jnp.sum(plain.prepare().mass.mv(jnp.ones((4,)))), 6.0, rtol=2.0e-6
    )


@pytest.mark.parametrize(
    ("kind", "area"),
    [("cylindrical-radial", 4.0 * np.pi), ("spherical-radial", 16.0 * np.pi)],
)
def test_radial_exterior_area_and_regular_axis(
    kind: CoordinateMeasure, area: float
) -> None:
    measure = CoordinateMeasurePlan(kind, radial_axis=0)
    weights = measure.weight(
        jnp.asarray([[0.0], [2.0]]), jnp.ones((2,)), frame="reference"
    )
    np.testing.assert_allclose(weights, [0.0, area], rtol=2.0e-6)


def test_complete_reference_rotational_area_receives_nanson_ratio_once() -> None:
    measure = CoordinateMeasurePlan("axisymmetric-rz", radial_axis=0)
    reference_area = measure.weight(
        jnp.asarray([[2.0, 0.0]]), jnp.asarray([3.0]), frame="reference"
    )
    deformation = phx.integration.DeformedMeasurePlan(
        "surface", reference_area, reference_normal=jnp.asarray([[1.0, 0.0, 0.0]])
    )
    state = deformation.evaluate(jnp.diag(jnp.asarray([1.2, 1.2, 1.5]))[None, :, :])
    np.testing.assert_allclose(
        state.measure("current"), [12.0 * np.pi * 1.2 * 1.5], rtol=3.0e-6
    )
    assert bool(state.valid)
