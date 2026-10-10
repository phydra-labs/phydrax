#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._identity import NumericRevision
from phydrax.discretization._spaces import DiscreteFieldSpace, TensorDofLayout
from phydrax.discretization.iga import BSplineGrid
from phydrax.discretization.iga._basis import (
    IsogeometricQuadraturePolicy,
    TensorSplineBasisSpec,
)
from phydrax.discretization.iga._geometry import NURBSGeometryState
from phydrax.discretization.iga._plan import IsogeometricPlan
from phydrax.discretization.iga._transfer import prepare_tensor_transfer
from phydrax.geometry.brep._constructors import reversed_bspline
from phydrax.geometry.brep._patches import BSplineCurve, BSplineSurfacePatch
from phydrax.linalg import ArraySpace


def _field(basis: TensorSplineBasisSpec) -> DiscreteFieldSpace:
    return DiscreteFieldSpace(
        "u",
        "iga-transfer-support",
        TensorDofLayout(basis.axis_names, basis.control_shape, layout_id=basis.layout_id),
        ArraySpace(basis.control_shape, dtype=jnp.float64),
        representation="basis_coefficient",
        conformity="H1",
    )


def test_exact_tensor_transfer_reproduces_linear_spline() -> None:
    for target_knots, target_degree in [
        ((0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0), 2),
        ((0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0), 3),
    ]:
        source_grid = BSplineGrid(jnp.asarray((0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0)), 2)
        target_grid = BSplineGrid(jnp.asarray(target_knots), target_degree)
        source_basis = TensorSplineBasisSpec((source_grid,))
        target_basis = TensorSplineBasisSpec((target_grid,))
        plan = prepare_tensor_transfer(
            source_basis,
            target_basis,
            _field(source_basis),
            _field(target_basis),
            source_plan_id="coarse",
            target_plan_id="fine",
            source_revision=NumericRevision("coarse", {"knots": source_grid.knots}),
            target_revision=NumericRevision("fine", {"knots": target_grid.knots}),
            transfer_class="exact",
        )

        # Greville coefficients represent f(x) = x in every B-spline space.
        source_linear = jnp.asarray(source_grid.greville_abscissae)
        target_linear = np.asarray(target_grid.greville_abscissae)
        np.testing.assert_allclose(plan.apply(source_linear), target_linear, atol=1e-12)
        payload = jnp.stack((jnp.ones_like(source_linear), source_linear), axis=-1)
        np.testing.assert_allclose(
            plan.apply_payload(payload),
            np.stack((np.ones_like(target_linear), target_linear), axis=-1),
            atol=1e-12,
        )
        assert plan.evidence.transfer_class == "exact"
        assert plan.field_transfer.properties.constant_preserving is True
        assert plan.evidence.pointwise_residual < 1e-12


@pytest.mark.parametrize(
    ("target_knots", "target_degree"),
    [
        ((2.0, 2.0, 2.0, 3.0, 5.0, 5.0, 5.0), 2),
        ((2.0, 2.0, 2.0, 2.0, 5.0, 5.0, 5.0, 5.0), 3),
    ],
)
def test_native_rational_source_transfer_preserves_map_and_derivatives(
    target_knots: tuple[float, ...], target_degree: int
) -> None:
    source = BSplineCurve(
        jnp.asarray(((1.0, 0.0), (1.0, 1.0), (0.0, 1.0))),
        7.0 * jnp.asarray((1.0, np.sqrt(0.5), 1.0)),
        jnp.asarray((2.0, 2.0, 2.0, 5.0, 5.0, 5.0)),
        2,
    )
    bound = IsogeometricPlan.from_source(
        source,
        axis_names=("arc",),
        quadrature_policy=IsogeometricQuadraturePolicy(4),
    )
    np.testing.assert_array_equal(bound.geometry.weights, source.weights)
    np.testing.assert_array_equal(bound.geometry.control_points, source.control_points)
    np.testing.assert_array_equal(bound.basis.axes[0].knots, source.knots)
    assert bound.basis.axes[0].parameter_interval == (2.0, 5.0)
    target_grid = BSplineGrid(jnp.asarray(target_knots), target_degree)
    target_basis = TensorSplineBasisSpec((target_grid,), axis_names=("arc",))
    transfer = prepare_tensor_transfer(
        bound.basis,
        target_basis,
        _field(bound.basis),
        _field(target_basis),
        source_plan_id=bound.plan_id,
        target_plan_id="target",
        source_revision=NumericRevision("source", {"weights": source.weights}),
        target_revision=NumericRevision("target", {"knots": target_grid.knots}),
        transfer_class="exact",
    )
    parameters = jnp.asarray((2.0, 2.1, 2.7, 3.0, 4.2, 4.9, 5.0))

    def original(points: jax.Array, weights: jax.Array) -> jax.Array:
        return BSplineCurve(points, weights, source.knots, source.degree).evaluate(
            parameters
        )

    def refined(points: jax.Array, weights: jax.Array) -> jax.Array:
        geometry = transfer.apply_geometry(NURBSGeometryState(points, weights))
        return BSplineCurve(
            geometry.control_points, geometry.weights, target_grid.knots, target_degree
        ).evaluate(parameters)

    arguments = (source.control_points, source.weights)
    tangents = (
        jnp.asarray(((0.2, -0.3), (-0.4, 0.5), (0.6, 0.1))),
        jnp.asarray((0.1, -0.2, 0.3)),
    )
    old_values, old_jvp = jax.jvp(original, arguments, tangents)
    new_values, new_jvp = jax.jvp(refined, arguments, tangents)
    np.testing.assert_allclose(new_values, old_values, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(new_jvp, old_jvp, atol=1e-12, rtol=1e-12)
    _, old_pullback = jax.vjp(original, *arguments)
    _, new_pullback = jax.vjp(refined, *arguments)
    cotangent = jnp.arange(old_values.size, dtype=old_values.dtype).reshape(
        old_values.shape
    )
    for actual, expected in zip(
        new_pullback(cotangent), old_pullback(cotangent), strict=True
    ):
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
    target_geometry = transfer.apply_geometry(bound.geometry)
    np.testing.assert_allclose(
        target_geometry.weights, transfer.apply(source.weights), atol=1e-12, rtol=1e-12
    )
    field, field_weights = transfer.apply_rational_payload(
        source.control_points, source.weights
    )
    np.testing.assert_array_equal(field, target_geometry.control_points)
    np.testing.assert_array_equal(field_weights, target_geometry.weights)


def test_native_surface_source_retains_axes_domains_and_scientific_weight_scale() -> None:
    source = BSplineSurfacePatch(
        jnp.asarray(
            (
                ((0.0, 0.0, 0.0), (0.0, 1.0, 0.2)),
                ((1.0, 0.0, 0.1), (1.0, 1.0, 0.3)),
            )
        ),
        jnp.asarray(((2.0, 3.0), (4.0, 5.0))),
        jnp.asarray((-3.0, -3.0, 2.0, 2.0)),
        jnp.asarray((7.0, 7.0, 11.0, 11.0)),
        1,
        1,
    )
    plan = IsogeometricPlan.from_source(
        source,
        axis_names=("u", "v"),
        quadrature_policy=IsogeometricQuadraturePolicy(3),
    )
    assert plan.basis.axis_names == ("u", "v")
    assert tuple(axis.parameter_interval for axis in plan.basis.axes) == (
        (-3.0, 2.0),
        (7.0, 11.0),
    )
    np.testing.assert_array_equal(plan.geometry.control_points, source.control_points)
    np.testing.assert_array_equal(plan.geometry.weights, source.weights)
    original_revision = NumericRevision(
        "native-source",
        {"points": plan.geometry.control_points, "weights": plan.geometry.weights},
    )
    scaled = NURBSGeometryState(source.control_points, 2.0 * source.weights)
    scaled_revision = NumericRevision(
        "native-source", {"points": scaled.control_points, "weights": scaled.weights}
    )
    assert original_revision.revision_id != scaled_revision.revision_id


def test_projected_transfer_refuses_rational_geometry_preservation() -> None:
    grid = BSplineGrid.open_uniform(1, 2)
    basis = TensorSplineBasisSpec((grid,))
    revision = NumericRevision("source", {"knots": grid.knots})
    transfer = prepare_tensor_transfer(
        basis,
        basis,
        _field(basis),
        _field(basis),
        source_plan_id="source",
        target_plan_id="target",
        source_revision=revision,
        target_revision=revision,
        transfer_class="projected",
        quadrature_id="declared-quadrature",
    )
    geometry = NURBSGeometryState(
        grid.greville_abscissae[:, None], jnp.asarray((2.0, 3.0, 4.0))
    )
    with pytest.raises(ValueError, match="exact qualified transfer"):
        transfer.apply_geometry(geometry)
    with pytest.raises(ValueError, match="exact transfer"):
        transfer.apply_rational_payload(geometry.control_points, geometry.weights)
    with pytest.raises(ValueError, match="exact tensor transfer"):
        transfer.bind_numeric(revision, revision)


@pytest.mark.filterwarnings("error:A JAX array is being set as static:UserWarning")
@pytest.mark.parametrize("elevate", (False, True))
@pytest.mark.parametrize("reverse", (False, True))
def test_original_native_ellipse_source_reuses_prepared_transfer(
    elevate: bool, reverse: bool
) -> None:
    # Same authoritative quadratic, repeated-knot profile used by the native
    # extrusion/revolution construction corpus; no sampled spline surrogate.
    curve = BSplineCurve(
        jnp.asarray(
            (
                (4.0, 0.0),
                (4.0, 0.5),
                (3.0, 0.5),
                (2.0, 0.5),
                (2.0, 0.0),
                (2.0, -0.5),
                (3.0, -0.5),
                (4.0, -0.5),
                (4.0, 0.0),
            )
        ),
        jnp.asarray(
            (
                1.0,
                np.sqrt(0.5),
                1.0,
                np.sqrt(0.5),
                1.0,
                np.sqrt(0.5),
                1.0,
                np.sqrt(0.5),
                1.0,
            )
        ),
        jnp.asarray((0.0, 0.0, 0.0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0, 1.0)),
        2,
    )
    source = reversed_bspline(curve) if reverse else curve
    source_plan = IsogeometricPlan.from_source(
        source,
        axis_names=("profile",),
        quadrature_policy=IsogeometricQuadraturePolicy(4),
    )
    source_knots = np.asarray(source.knots)
    if elevate:
        unique, multiplicity = np.unique(source_knots, return_counts=True)
        target_knots = np.repeat(unique, multiplicity + 1)
        target_degree = 3
    else:
        midpoints = 0.5 * (np.unique(source_knots)[:-1] + np.unique(source_knots)[1:])
        target_knots = np.sort(np.concatenate((source_knots, midpoints)))
        target_degree = 2
    target_grid = BSplineGrid(target_knots, target_degree)
    target_basis = TensorSplineBasisSpec((target_grid,), axis_names=("profile",))
    source_revision = NumericRevision(
        "profile",
        {
            "points": source.control_points,
            "weights": source.weights,
            "knots": source.knots,
        },
    )
    target_revision = NumericRevision("target", {"knots": target_grid.knots})
    transfer = prepare_tensor_transfer(
        source_plan.basis,
        target_basis,
        _field(source_plan.basis),
        _field(target_basis),
        source_plan_id=source_plan.plan_id,
        target_plan_id="target",
        source_revision=source_revision,
        target_revision=target_revision,
        transfer_class="exact",
    )
    parameters = jnp.linspace(*source.parameter_domain, 33)
    source_values = source.evaluate(parameters)
    np.testing.assert_allclose(
        (source_values[:, 0] - 3.0) ** 2 + (source_values[:, 1] / 0.5) ** 2,
        jnp.ones(parameters.shape),
        atol=1e-12,
        rtol=1e-12,
    )
    apply_prepared = eqx.filter_jit(
        lambda prepared, state: prepared.apply_geometry(state)
    )
    geometry = apply_prepared(transfer, source_plan.geometry)
    target_revision = NumericRevision(
        "target",
        {
            "points": geometry.control_points,
            "weights": geometry.weights,
            "knots": target_grid.knots,
        },
    )
    transfer = transfer.bind_numeric(source_revision, target_revision)
    target = BSplineCurve(
        geometry.control_points, geometry.weights, target_grid.knots, target_degree
    )
    np.testing.assert_allclose(
        target.evaluate(parameters), source.evaluate(parameters), atol=1e-12, rtol=1e-12
    )
    if reverse:
        np.testing.assert_allclose(
            source.evaluate(parameters),
            curve.evaluate(-parameters),
            atol=1e-12,
            rtol=1e-12,
        )

    def source_design_values(design: jax.Array) -> jax.Array:
        points = source.control_points * design[:2] + design[2:4]
        weights = source.weights.at[1::2].multiply(design[4])
        return BSplineCurve(points, weights, source.knots, source.degree).evaluate(
            parameters
        )

    def target_design_values(design: jax.Array) -> jax.Array:
        points = source.control_points * design[:2] + design[2:4]
        weights = source.weights.at[1::2].multiply(design[4])
        state = transfer.apply_geometry(NURBSGeometryState(points, weights))
        return BSplineCurve(
            state.control_points, state.weights, target_grid.knots, target_degree
        ).evaluate(parameters)

    design = jnp.asarray((1.0, 1.0, 0.0, 0.0, 1.0))
    direction = jnp.asarray((0.125, -0.25, 0.0625, -0.125, 0.25))
    _, source_jvp = jax.jvp(source_design_values, (design,), (direction,))
    _, target_jvp = jax.jvp(target_design_values, (design,), (direction,))
    np.testing.assert_allclose(target_jvp, source_jvp, atol=1e-12, rtol=1e-12)
    _, source_vjp = jax.vjp(source_design_values, design)
    _, target_vjp = jax.vjp(target_design_values, design)
    cotangent = jnp.ones(source_values.shape)
    np.testing.assert_allclose(
        target_vjp(cotangent)[0], source_vjp(cotangent)[0], atol=1e-12, rtol=1e-12
    )
    np.testing.assert_allclose(
        jnp.vdot(target_jvp, cotangent),
        jnp.vdot(direction, target_vjp(cotangent)[0]),
        atol=1e-12,
        rtol=1e-12,
    )
    edited = NURBSGeometryState(
        source.control_points.at[1, 1].add(0.125),
        source.weights.at[1].multiply(1.125),
    )
    edited_geometry = apply_prepared(transfer, edited)
    edited_source = BSplineCurve(
        edited.control_points, edited.weights, source.knots, source.degree
    )
    edited_target = BSplineCurve(
        edited_geometry.control_points,
        edited_geometry.weights,
        target_grid.knots,
        target_degree,
    )
    np.testing.assert_allclose(
        edited_target.evaluate(parameters),
        edited_source.evaluate(parameters),
        atol=1e-12,
        rtol=1e-12,
    )
    edited_revision = NumericRevision(
        "profile",
        {
            "points": edited.control_points,
            "weights": edited.weights,
            "knots": source.knots,
        },
    )
    assert not transfer.is_valid_for(
        source_plan_id=source_plan.plan_id,
        target_plan_id="target",
        source_layout_id=source_plan.basis.layout_id,
        target_layout_id=target_basis.layout_id,
        source_revision=edited_revision,
        target_revision=target_revision,
    )
    with pytest.raises(ValueError, match="invalidated"):
        transfer.require_valid_for(
            source_plan_id=source_plan.plan_id,
            target_plan_id="target",
            source_layout_id=source_plan.basis.layout_id,
            target_layout_id=target_basis.layout_id,
            source_revision=edited_revision,
            target_revision=target_revision,
        )
    edited_target_revision = NumericRevision(
        "target",
        {
            "points": edited_geometry.control_points,
            "weights": edited_geometry.weights,
            "knots": target_grid.knots,
        },
    )
    rebound = transfer.bind_numeric(edited_revision, edited_target_revision)
    rebound.require_valid_for(
        source_plan_id=source_plan.plan_id,
        target_plan_id="target",
        source_layout_id=source_plan.basis.layout_id,
        target_layout_id=target_basis.layout_id,
        source_revision=edited_revision,
        target_revision=edited_target_revision,
    )
    assert rebound.plan_id != transfer.plan_id
    assert rebound.P.operator_id == transfer.P.operator_id
    assert rebound.PT.operator_id == transfer.PT.operator_id
    rebound_manifest, rebound_archive = rebound.transition_archive_payload()
    assert rebound_manifest["source_content_id"] == edited_revision.content_id
    assert rebound_manifest["target_content_id"] == edited_target_revision.content_id
    for name, original_factor in transfer.transition_archive_payload()[1].items():
        np.testing.assert_array_equal(rebound_archive[name], original_factor)
    manifest, archive = transfer.transition_archive_payload()
    assert manifest["source_content_id"] == source_revision.content_id
    assert manifest["target_content_id"] == target_revision.content_id
    np.testing.assert_allclose(
        archive["P_axis_000"] @ source.weights, geometry.weights, atol=1e-12, rtol=1e-12
    )
