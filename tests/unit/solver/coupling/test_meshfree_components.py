# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Meshfree residual, constraint, reconstruction and physical capacity behavior."""

from __future__ import annotations

from typing import TypedDict

import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from phydrax.discretization import (
    FacetTraceRule,
    PointBoundaryCondition,
    PointBoundaryPlan,
    PointCloudPlan,
    PointCloudPoissonPlan,
    PointCloudReconstruction,
    prepare_point_cloud_field_reconstruction,
    PreparedPointCloudDiscretization,
    PreparedPointCloudPoisson,
)
from phydrax.discretization.meshfree import (
    LocalStencilPolicy,
    PointBlockSystemPlan,
    PointBoundaryCharts,
    PointGhostLayerPlan,
)
from phydrax.geometry import Polygon, Rectangle
from phydrax.linalg import (
    ArraySpace,
    BlockLinearOperator,
    BlockSpace,
    ConstraintMap,
    FunctionLinearOperator,
)
from phydrax.measurement import (
    IndexSampleSupport,
    QuantitySpec,
    SamplingSemantics,
    SpatialSamplingKind,
)
from phydrax.solver.coupling import (
    CoupledGauge,
    CoupledProblemPlan,
    FieldFluxObservation,
    MeshfreeBoundaryTrace,
    MeshfreeComponent,
    MeshfreeReaction,
    MeshfreeTraceComponent,
    prepare_coupled_problem,
)
from phydrax.sparse import SparseCoordinateOperator
from phydrax.units import ONE


class TraceOptions(TypedDict, total=False):
    measure_scale: float
    neighbor: bool
    data: float
    values: np.ndarray | None


def prepared_cloud_component(
    *,
    name: str = "bulk",
    reconstruction: PointCloudReconstruction = "polynomial",
    constrained: bool = False,
) -> MeshfreeComponent:
    axis = np.linspace(0.0, 1.0, 7)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    points = np.stack((x.ravel(), y.ravel()), axis=-1)
    measures = np.linspace(0.5, 1.5, points.shape[0]) / points.shape[0]
    cloud = PointCloudPlan(points, measures, neighbors=20).prepare()
    view = prepare_point_cloud_field_reconstruction(
        cloud,
        support_geometry=Rectangle((0.5, 0.5), (1.0, 1.0)).compile(),
        radius=0.65,
        capacity=49,
        reconstruction=reconstruction,
    )
    full = cloud.field_spaces[0].vector_space
    weights = sum(pair[1] for pair in cloud.derivative_weights)
    native = SparseCoordinateOperator(
        cloud.relation,
        -jnp.asarray(measures)[:, None] * weights,
        source=full,
        target=full,
    )
    offset = np.zeros(points.shape[0])
    constraint = None
    free = None
    if constrained:
        free = jnp.asarray(np.flatnonzero(points[:, 0] > 0.0), dtype=jnp.int32)
        reduced = ArraySpace((free.size,))
        prolongation = FunctionLinearOperator(
            lambda value: jnp.zeros((points.shape[0],)).at[free].set(value),
            source=reduced,
            target=full,
            transpose_action=lambda value: value[free],
        )
        constraint = ConstraintMap(full, reduced, prolongation)
        offset[points[:, 0] == 0] = points[points[:, 0] == 0, 1] ** 2
    return MeshfreeComponent(
        cloud,
        native,
        view,
        measures,
        name=name,
        owner_id=cloud.prepared_id,
        constraint=constraint,
        lift=offset,
        free_rows=free,
        nullspace=None if constrained else np.ones((points.shape[0], 1)),
    )


def test_native_residual_and_capacity_retain_physical_measures() -> None:
    component = prepared_cloud_component()
    points = np.asarray(component.owner.points)
    values = jnp.asarray(np.sum(points**2, axis=-1))
    rows = component.residual((values,), None)[0]
    np.testing.assert_allclose(rows, -4 * component.mass_diagonal, atol=2e-11)
    mass = component.prepare_capacity("concentration", coefficient=2.5).operator(None)
    np.testing.assert_allclose(
        mass.mv(values), 2.5 * component.mass_diagonal * values, atol=1e-14
    )
    query = component.prepare_field_reconstruction("concentration").prepare_query(
        np.asarray([[0.25, 0.45], [0.72, 0.64]])
    )
    np.testing.assert_allclose(
        query.apply(values), [0.25**2 + 0.45**2, 0.72**2 + 0.64**2], atol=2e-11
    )


def test_dirichlet_lift_and_row_pullback_are_applied_once() -> None:
    component = prepared_cloud_component(constrained=True)
    points = np.asarray(component.owner.points)
    field = component.fields[0]
    free = field.free_rows
    assert free is not None
    values = jnp.asarray(np.sum(points**2, axis=-1))
    state = (values[free],)
    np.testing.assert_allclose(
        component.expand("concentration", state, None), values, atol=1e-14
    )
    np.testing.assert_allclose(
        component.residual(state, None)[0],
        (-4 * component.mass_diagonal)[free],
        atol=2e-11,
    )
    operator = component.linear_operator(None)
    assert operator is not None
    direction = jnp.sin(jnp.arange(free.size, dtype=jnp.float64))
    np.testing.assert_allclose(
        operator.mv((direction,))[0],
        component.residual((state[0] + direction,), None)[0]
        - component.residual(state, None)[0],
        atol=2e-11,
    )
    dual = jnp.cos(jnp.arange(free.size, dtype=jnp.float64))
    np.testing.assert_allclose(
        jnp.vdot(operator.mv((direction,))[0], dual),
        jnp.vdot(direction, operator.transpose_mv((dual,))[0]),
        atol=2e-11,
    )


def test_capacity_and_false_kernel_declarations_refuse() -> None:
    component = prepared_cloud_component()
    with pytest.raises(ValueError, match="positive"):
        component.prepare_capacity("concentration", coefficient=-1.0)
    with pytest.raises(ValueError, match="kernel"):
        MeshfreeComponent(
            component.owner,
            component.native_operator,
            component.reconstruction,
            component.mass_diagonal,
            name="wrong",
            owner_id=component.owner_id,
            nullspace=np.asarray(component.owner.points)[:, :1] ** 2,
        )
    with pytest.raises(ValueError, match="strictly positive"):
        MeshfreeComponent(
            component.owner,
            component.native_operator,
            component.reconstruction,
            np.zeros(component.mass_diagonal.shape),
            name="wrong",
            owner_id=component.owner_id,
        )


# --- Chart-authorized traces ----------------------------------------------------------

RESOLUTION = 6
CONDUCTIVITY = 3.0


def _half_plate(
    *,
    shift: float = 0.0,
    neighbor: bool = False,
    resolution: int = RESOLUTION,
    jitter: bool = False,
) -> tuple[PreparedPointCloudDiscretization, PointBoundaryCharts, np.ndarray]:
    """Grid cloud on ``[1, 2] x [0, 1]`` and polygon charts of its ``x = 1`` side.

    ``neighbor`` takes the charts from the polygon of the adjacent half
    ``[0, 1] x [0, 1]``: the same sites with the opposite outward orientation.
    """
    axis = np.linspace(0.0, 1.0, resolution + 1)
    h = 1.0 / resolution
    x, y = np.meshgrid(1.0 + axis, axis, indexing="ij")
    points = np.stack((x.ravel(), y.ravel()), axis=-1)
    weight = np.where((axis > 0) & (axis < 1), h, 0.5 * h)
    left, right = np.isclose(points[:, 0], 1.0), np.isclose(points[:, 0], 2.0)
    bottom, top = np.isclose(points[:, 1], 0.0), np.isclose(points[:, 1], 1.0)
    if jitter:
        interior = ~(left | right | bottom | top)
        points[interior] += (
            np.random.default_rng(7).uniform(-0.15, 0.15, (np.count_nonzero(interior), 2))
            * h
        )
    normals = np.zeros_like(points)
    normals[left], normals[right] = (-1.0, 0.0), (1.0, 0.0)
    normals[bottom], normals[top] = (0.0, -1.0), (0.0, 1.0)
    cloud = PointCloudPlan(
        points,
        np.outer(weight, weight).ravel(),
        boundary_mask=left | right | bottom | top,
        boundary_normals=normals,
        stencil=LocalStencilPolicy(approximation="phs-rbf-fd", polynomial_degree=3),
        neighbors=30,
    ).prepare()
    if neighbor:
        rising = np.stack((np.full(resolution, 1.0), axis[1:]), axis=1)
        vertices = np.concatenate(([[0.0, 0.0], [1.0, 0.0]], rising, [[0.0, 1.0]]))
        selected = tuple(range(1, 1 + resolution))
    else:
        falling = np.stack((np.full(resolution, 1.0), axis[::-1][:-1] + shift), axis=1)
        vertices = np.concatenate(
            ([[1.0, shift], [2.0, shift], [2.0, 1.0 + shift]], falling)
        )
        selected = tuple(range(3, 3 + resolution))
    atlas = Polygon([tuple(v) for v in vertices]).compile().boundary_atlas
    charts = PointBoundaryCharts(
        cloud,
        atlas,
        FacetTraceRule("gauss-lobatto-legendre", points=2),
        charts=selected,
    )
    return cloud, charts, normals


def _trace_component(
    *,
    measure_scale: float = 1.0,
    neighbor: bool = False,
    data: float = 0.0,
    values: np.ndarray | None = None,
) -> MeshfreeTraceComponent:
    cloud, charts, normals = _half_plate(neighbor=neighbor)
    points = np.asarray(cloud.points)
    count = points.shape[0]
    interface = np.flatnonzero(
        np.isclose(points[:, 0], 1.0)
        & (points[:, 1] > 1e-12)
        & (points[:, 1] < 1 - 1e-12)
    )
    boundary = np.flatnonzero(np.asarray(cloud.plan.boundary_mask))
    dirichlet = np.setdiff1d(boundary, interface)
    exact = np.zeros(count) if values is None else values
    plan = PointBoundaryPlan(
        (
            PointBoundaryCondition(
                "dirichlet", dirichlet, exact[dirichlet], label="exterior"
            ),
            PointBoundaryCondition(
                "neumann",
                interface,
                data,
                label="interface",
                normals=normals[interface],
                measure=measure_scale * np.asarray(charts.measure)[interface],
            ),
        ),
        row_count=count,
    )
    poisson = PointCloudPoissonPlan(cloud, plan).prepare(CONDUCTIVITY)
    reconstruction = prepare_point_cloud_field_reconstruction(
        cloud,
        support_geometry=Rectangle((1.5, 0.5), (1.0, 1.0)).compile(),
        radius=0.6,
        capacity=49,
    )
    return MeshfreeTraceComponent(
        poisson,
        reconstruction,
        (MeshfreeBoundaryTrace("u", "interface", charts),),
        name="cloud",
    )


def test_trace_component_publishes_measure_scaled_conormal_reaction() -> None:
    points = _half_plate()[0].points
    exact = np.asarray(2.0 - 0.7 * points[:, 0] + 0.4 * points[:, 1])
    component = _trace_component(values=exact)
    record = component.fields[0]
    free = record.free_rows
    assert free is not None and component.space == "reduced"
    state = (jnp.asarray(exact)[free],)
    np.testing.assert_allclose(component.expand("u", state, None), exact, atol=1e-13)
    domain = component.boundary_domain("u", "interface")
    trace = component.prepare_side_trace(
        "u", domain, rule=FacetTraceRule("gauss-lobatto-legendre", points=3)
    )
    sites = np.asarray(trace.sites)
    np.testing.assert_allclose(
        trace.apply(jnp.asarray(exact)),
        2.0 - 0.7 * sites[..., 0] + 0.4 * sites[..., 1],
        atol=1e-12,
    )
    assert trace.descriptor.trace_degree == 1
    np.testing.assert_allclose(np.sum(np.asarray(trace.weights)), 1.0, rtol=1e-12)
    flux = component.prepare_conormal_flux(trace)
    reaction = np.asarray(flux.evaluate(jnp.asarray(exact), None))
    rows = np.asarray(trace.support_rows)
    interface = np.isin(rows, np.asarray(free))
    measure = np.asarray(component.boundary_trace("u", "interface").charts.measure)
    # Outward normal -x: q = k grad(u) . n = 3 * 0.7 on the interface rows.
    np.testing.assert_allclose(
        reaction[interface], measure[rows][interface] * CONDUCTIVITY * 0.7, rtol=1e-9
    )
    bulk = np.setdiff1d(np.asarray(free), rows)
    residual = np.asarray(component.residual(state, None)[0])
    np.testing.assert_allclose(
        residual[np.searchsorted(np.asarray(free), bulk)], 0.0, atol=1e-9
    )
    capacity = component.prepare_capacity("u").operator(None)
    diagonal = np.asarray(capacity.mv(jnp.ones(exact.shape)))
    owned = np.asarray(component.owner.plan.boundary_mask)
    np.testing.assert_allclose(diagonal, np.where(owned, 0.0, 1.0))


def _weak_trace_component(
    reaction: MeshfreeReaction | None = None,
) -> MeshfreeTraceComponent:
    base = _trace_component()
    owner = PointCloudPoissonPlan(
        base.owner, base.system.plan.boundary, form="dissipative"
    ).prepare(CONDUCTIVITY)
    return MeshfreeTraceComponent(
        owner,
        base.reconstruction,
        base.traces,
        name="weak-cloud",
        source=0.7,
        reaction=reaction,
    )


def test_scalar_weak_trace_residual_preserves_integrated_natural_rows() -> None:
    component = _weak_trace_component()
    assert isinstance(component.system, PreparedPointCloudPoisson)
    record = component.fields[0]
    free = record.free_rows
    assert free is not None
    points = component.owner.points
    state = (jnp.sin(points[free, 0]) + points[free, 1] ** 2,)
    full = component.expand("u", state, None)
    owner_rows = component.system.physical_assembly.operator.mv(full)
    owner_rows = owner_rows - component.system.physical_rhs(jnp.full_like(full, 0.7))
    np.testing.assert_allclose(
        component.residual(state, None)[0], owner_rows[free], rtol=1e-12, atol=1e-12
    )
    operator = component.linear_operator(None)
    assert operator is not None
    np.testing.assert_allclose(
        operator.mv(state)[0],
        component.system.physical_assembly.operator.mv(full)[free],
        rtol=1e-12,
        atol=1e-12,
    )


def test_scalar_weak_trace_capacity_retains_natural_volume_rates() -> None:
    component = _weak_trace_component()
    diagonal = (
        component.prepare_capacity("u")
        .operator(None)
        .mv(jnp.ones_like(component.owner.quadrature_weights))
    )
    dirichlet = component.system.plan.boundary.condition("exterior").rows
    expected = component.owner.quadrature_weights.at[dirichlet].set(0.0)
    np.testing.assert_allclose(diagonal, expected, rtol=1e-12, atol=0.0)
    natural = component.system.plan.boundary.condition("interface").rows
    assert bool(jnp.all(diagonal[natural] > 0.0))


def test_scalar_weak_trace_reaction_retains_natural_volume_rates() -> None:
    def reaction(values: Array, points: Array, args: object) -> Array:
        del points, args
        return values**2

    component = _weak_trace_component(MeshfreeReaction(reaction, reaction_id="square"))
    assert isinstance(component.system, PreparedPointCloudPoisson)
    record = component.fields[0]
    free = record.free_rows
    assert free is not None
    state = (jnp.full(free.shape, 0.8, dtype=jnp.float64),)
    full = component.expand("u", state, None)
    owner_rows = component.system.physical_assembly.operator.mv(full)
    owner_rows = owner_rows - component.system.physical_rhs(jnp.full_like(full, 0.7))
    expected = owner_rows + component.owner.quadrature_weights * full**2
    np.testing.assert_allclose(
        component.residual(state, None)[0], expected[free], rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("declared_gauge", (False, True), ids=("ungauged", "gauged"))
def test_floating_scalar_weak_trace_reaches_coupled_gauge_consumer(
    declared_gauge: bool,
) -> None:
    base = _trace_component()
    interface_rows = np.flatnonzero(np.isclose(np.asarray(base.owner.points)[:, 0], 1.0))
    exterior_rows = np.setdiff1d(
        np.flatnonzero(np.asarray(base.owner.plan.boundary_mask)), interface_rows
    )
    conditions = (
        PointBoundaryCondition(
            "neumann",
            interface_rows,
            0.0,
            label="interface",
            normals=np.tile([[-1.0, 0.0]], (interface_rows.size, 1)),
            measure=np.asarray(base.traces[0].charts.measure)[interface_rows],
        ),
        PointBoundaryCondition(
            "neumann",
            exterior_rows,
            0.0,
            label="exterior",
            normals=np.asarray(base.owner.plan.boundary_normals)[exterior_rows],
            measure=np.ones(exterior_rows.shape, dtype=np.float64),
        ),
    )
    owner = PointCloudPoissonPlan(
        base.owner,
        PointBoundaryPlan(conditions, row_count=base.owner.state_shape[0]),
        form="dissipative",
    ).prepare(CONDUCTIVITY)
    component = MeshfreeTraceComponent(
        owner, base.reconstruction, base.traces, name="floating-cloud"
    )
    plan = CoupledProblemPlan(
        "floating-weak-test",
        components=(component,),
        bindings=(),
        laws=(),
        gauge=CoupledGauge() if declared_gauge else None,
    )
    if not declared_gauge:
        with pytest.raises(ValueError, match="declare a CoupledGauge"):
            prepare_coupled_problem(plan)
        return
    prepared = prepare_coupled_problem(plan)
    policy = prepared.nullspace_policy
    assert policy is not None and policy.right is not None and policy.left is not None
    kernel = component.nullspace(None)
    assert kernel is not None
    operator = component.linear_operator(None)
    assert operator is not None
    vector = kernel[0][:, 0]
    np.testing.assert_allclose(operator.mv((vector,))[0], 0.0, atol=1e-10)
    np.testing.assert_allclose(operator.transpose_mv((vector,))[0], 0.0, atol=1e-10)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"measure_scale": 2.0}, "lumped boundary measure"),
        ({"neighbor": True}, "outward normals of 'interface' disagree"),
        ({"data": 0.5}, "native boundary data"),
    ],
    ids=["stale-measure", "neighbor-orientation", "owner-flux-data"],
)
def test_trace_component_refuses_inconsistent_boundary_authority(
    kwargs: TraceOptions, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        _trace_component(**kwargs)


def test_charts_refuse_sites_that_are_not_cloud_points() -> None:
    with pytest.raises(ValueError, match="coincide with a cloud point"):
        _half_plate(shift=0.5 / RESOLUTION)


def test_trace_component_refuses_pointwise_flux_and_foreign_domains() -> None:
    component = _trace_component()
    domain = component.boundary_domain("u", "interface")
    rule = FacetTraceRule("gauss-lobatto-legendre", points=2)
    trace = component.prepare_side_trace("u", domain, rule=rule)
    with pytest.raises(ValueError, match="zero flux"):
        component.prepare_pointwise_flux(trace)
    with pytest.raises(ValueError, match="trace-inverse"):
        component.certify_flux_stability(component.prepare_conormal_flux(trace))
    # Charts of other edges of the same cloud are a different entity set: a
    # domain the component never declared as a coupling boundary.
    other = PointBoundaryCharts(
        component.owner,
        component.boundary_trace("u", "interface").charts.atlas,
        rule,
        charts=(0, 1),
    ).domain()
    with pytest.raises(ValueError, match="chart-authorized"):
        component.prepare_side_trace("u", other, rule=rule)


def test_block_component_publishes_coupled_fields_and_nonlinear_reaction() -> None:
    component = prepared_cloud_component()
    full = component.fields[0].full_space
    spaces = tuple(
        ArraySpace(full.shape, dtype=full.dtype, space_id=f"block:{name}")
        for name in ("a", "b")
    )
    space = BlockSpace(spaces, names=("a", "b"))
    native = component.native_operator

    def block(row: int, column: int) -> FunctionLinearOperator:
        scale = 1.0 if row == column else 0.25
        return FunctionLinearOperator(
            lambda value: scale * native.mv(value),
            source=spaces[column],
            target=spaces[row],
            transpose_action=lambda value: scale * native.transpose_mv(value),
        )

    operator = BlockLinearOperator(
        [[block(0, 0), block(0, 1)], [block(1, 0), block(1, 1)]],
        source=space,
        target=space,
    )
    linear = MeshfreeComponent(
        component.owner,
        operator,
        component.reconstruction,
        component.mass_diagonal,
        name="pair",
        owner_id="pair",
        field=("a", "b"),
    )
    assert linear.coupled and linear.field_space_id("b") == "block:b"
    first = jnp.sin(jnp.arange(full.size, dtype=jnp.float64))
    second = jnp.cos(jnp.arange(full.size, dtype=jnp.float64))
    published = linear.linear_operator(None)
    assert published is not None
    np.testing.assert_allclose(
        published.mv((first, second))[0],
        native.mv(first) + 0.25 * native.mv(second),
        atol=1e-12,
    )
    reaction = MeshfreeReaction(lambda u, x, args: u**3, reaction_id="cubic")
    weights = np.stack([np.asarray(component.mass_diagonal)] * 2, axis=1)
    nonlinear = MeshfreeComponent(
        component.owner,
        operator,
        component.reconstruction,
        component.mass_diagonal,
        name="pair",
        owner_id="pair",
        field=("a", "b"),
        reaction=reaction,
        reaction_weights=weights,
    )
    assert nonlinear.linear_operator(None) is None
    rows = nonlinear.residual((first, second), None)
    np.testing.assert_allclose(
        rows[1],
        0.25 * native.mv(first) + native.mv(second) + weights[:, 1] * second**3,
        atol=1e-12,
    )
    with pytest.raises(ValueError, match="declared together"):
        MeshfreeComponent(
            component.owner,
            operator,
            component.reconstruction,
            component.mass_diagonal,
            name="pair",
            owner_id="pair",
            field=("a", "b"),
            reaction=reaction,
        )


@pytest.fixture(scope="module")
def coupled_ghost_component() -> MeshfreeTraceComponent:
    cloud, charts, _ = _half_plate(resolution=8, jitter=True)
    points = np.asarray(cloud.points)
    count = points.shape[0]
    interface = np.flatnonzero(np.isclose(points[:, 0], 1.0))
    boundary_rows = np.flatnonzero(np.asarray(cloud.plan.boundary_mask))
    # Ownership differs only at geometric corners, not inside a smooth face.
    flux_rows = (interface, interface[1:-1])
    dirichlet_rows = tuple(np.setdiff1d(boundary_rows, rows) for rows in flux_rows)
    gradients = np.asarray([[-0.7, 0.4], [0.3, -0.2]], dtype=np.float64)
    exact = 2.0 + points @ gradients.T
    conditions = tuple(
        condition
        for index, field in enumerate(("x", "y"))
        for condition in (
            PointBoundaryCondition(
                "dirichlet",
                dirichlet_rows[index],
                exact[dirichlet_rows[index], index],
                label=f"{field}-exterior",
                component=index,
            ),
            PointBoundaryCondition(
                "neumann",
                flux_rows[index],
                0.0,
                label=f"{field}-interface",
                component=index,
                normals=np.tile((-1.0, 0.0), (flux_rows[index].size, 1)),
                measure=np.asarray(charts.measure)[flux_rows[index]],
            ),
        )
    )
    boundary = PointBoundaryPlan(conditions, row_count=count, components=2)
    ghosts = PointGhostLayerPlan(boundary).prepare(cloud)
    coupling = np.asarray([[3.0, 0.6], [0.6, 2.0]], dtype=np.float64)
    coefficients = np.broadcast_to(
        coupling[None, :, :, None, None] * np.eye(2, dtype=np.float64),
        (count, 2, 2, 2, 2),
    )
    owner = PointBlockSystemPlan(
        cloud, boundary, components=("x", "y"), ghosts=ghosts
    ).prepare(coefficients)
    reconstruction = prepare_point_cloud_field_reconstruction(
        cloud,
        support_geometry=Rectangle((1.5, 0.5), (1.0, 1.0)).compile(),
        radius=0.6,
        capacity=count,
    )
    source = np.stack((1.0 + points[:, 0], -2.0 + points[:, 1]), axis=1)
    return MeshfreeTraceComponent(
        owner,
        reconstruction,
        tuple(
            MeshfreeBoundaryTrace(field, f"{field}-interface", charts)
            for field in ("x", "y")
        ),
        name="coupled",
        source=source,
    )


def test_coupled_ghost_rows_preserve_extended_equations(
    coupled_ghost_component: MeshfreeTraceComponent,
) -> None:
    """Independent stencil assembly checks packing, selective row swaps and lifts."""
    component = coupled_ghost_component
    points = np.asarray(component.owner.points)
    count = points.shape[0]
    ghosts = component.ghost_layer
    assert ghosts is not None
    boundary = component.system.plan.boundary
    charts = component.traces[0].charts
    flux_rows = tuple(
        np.asarray(boundary.condition(f"{field}-interface").rows) for field in ("x", "y")
    )
    dirichlet_rows = tuple(
        np.asarray(boundary.condition(f"{field}-exterior").rows) for field in ("x", "y")
    )
    gradients = np.asarray([[-0.7, 0.4], [0.3, -0.2]], dtype=np.float64)
    exact = 2.0 + points @ gradients.T
    coupling = np.asarray([[3.0, 0.6], [0.6, 2.0]], dtype=np.float64)
    source = np.stack((1.0 + points[:, 0], -2.0 + points[:, 1]), axis=1)
    assert tuple(record.name for record in component.fields) == (
        "x",
        "y",
        "x-ghost",
        "y-ghost",
    )
    assert component.ghost_layer is not None
    total = ghosts.row_count
    ghost_points = np.asarray(ghosts.plan.rows)
    offsets = np.asarray(ghosts.offsets)
    relation = ghosts.family.relation
    valid = np.asarray(relation.valid)
    targets = np.broadcast_to(np.arange(total)[:, None], valid.shape)
    sources = np.asarray(relation.source_indices)

    def derivative(index: tuple[int, int]) -> np.ndarray:
        matrix = np.zeros((total, total), dtype=np.float64)
        np.add.at(
            matrix,
            (targets[valid], sources[valid]),
            np.asarray(ghosts.family.weights_for(index))[valid],
        )
        return matrix

    laplace = derivative((2, 0)) + derivative((0, 2))
    normal = -derivative((1, 0))
    native = np.zeros((2 * total, 2 * total), dtype=np.float64)
    native_rhs = np.zeros((total, 2), dtype=np.float64)
    row_order = np.arange(2 * total).reshape((2, total))
    scales = np.ones((2, total), dtype=np.float64)
    for row in range(2):
        flux = np.isin(ghost_points, flux_rows[row])
        for column in range(2):
            block = -coupling[row, column] * laplace
            block[dirichlet_rows[row]] = 0.0
            if row == column:
                block[dirichlet_rows[row], dirichlet_rows[row]] = 1.0
            block[count:] = np.where(
                flux[:, None],
                coupling[row, column] * normal[ghost_points] / offsets[:, None],
                -coupling[row, column] * laplace[ghost_points],
            )
            native[
                row * total : (row + 1) * total,
                column * total : (column + 1) * total,
            ] = block
        native_rhs[:count, row] = source[:, row]
        native_rhs[dirichlet_rows[row], row] = exact[dirichlet_rows[row], row]
        native_rhs[count:, row] = np.where(flux, 0.0, source[ghost_points, row])
        selected = np.flatnonzero(flux)
        natural = ghost_points[selected]
        row_order[row, natural] = row * total + count + selected
        row_order[row, count + selected] = row * total + natural
        scales[row, natural] = np.asarray(charts.measure)[natural] * offsets[selected]
    packing = np.concatenate(
        (
            np.arange(count),
            total + np.arange(count),
            count + np.arange(ghosts.ghost_count),
            total + count + np.arange(ghosts.ghost_count),
        )
    )
    published_rows = row_order.ravel()[packing]
    published_scales = scales.ravel()[packing]
    matrix = published_scales[:, None] * native[np.ix_(published_rows, packing)]
    load = published_scales * native_rhs.T.ravel()[published_rows]
    state = tuple(
        jnp.sin(jnp.arange(block.space.size, dtype=jnp.float64) + index)
        for index, block in enumerate(component.state_blocks)
    )
    full = tuple(
        component.expand(record.name, state, None) for record in component.fields
    )
    full_flat = np.concatenate(tuple(np.asarray(value) for value in full))
    expected = matrix @ full_flat - load
    observed = np.concatenate(
        tuple(np.asarray(value) for value in component.full_rows(full, None))
    )
    np.testing.assert_allclose(
        observed,
        expected,
        rtol=1e-11,
        atol=2e-10,
    )
    # Undo publication and scaling: these are exactly the original extended rows.
    restored = np.empty((2 * total,), dtype=np.float64)
    restored[published_rows] = observed / published_scales
    np.testing.assert_allclose(
        restored,
        native @ full_flat[np.argsort(packing)] - native_rhs.T.ravel(),
        rtol=1e-11,
        atol=2e-10,
    )
    kept = np.concatenate(
        (
            np.setdiff1d(np.arange(count), dirichlet_rows[0]),
            count + np.setdiff1d(np.arange(count), dirichlet_rows[1]),
            np.arange(2 * count, 2 * total),
        )
    )
    np.testing.assert_allclose(
        np.concatenate(
            tuple(np.asarray(value) for value in component.residual(state, None))
        ),
        expected[kept],
        rtol=1e-11,
        atol=2e-10,
    )
    operator = component.linear_operator(None)
    assert operator is not None
    reduced = matrix[np.ix_(kept, kept)]
    direction = tuple(
        jnp.cos(jnp.arange(block.space.size, dtype=jnp.float64) + index)
        for index, block in enumerate(component.state_blocks)
    )
    np.testing.assert_allclose(
        np.concatenate(tuple(np.asarray(value) for value in operator.mv(direction))),
        reduced @ np.concatenate(tuple(np.asarray(value) for value in direction)),
        rtol=1e-11,
        atol=2e-10,
    )
    dual = tuple(value * (index + 1.0) for index, value in enumerate(state))
    reference_transpose = reduced.T @ np.concatenate(
        tuple(np.asarray(value) for value in dual)
    )
    for action in (operator.transpose_mv, operator.adjoint_mv):
        np.testing.assert_allclose(
            np.concatenate(tuple(np.asarray(value) for value in action(dual))),
            reference_transpose,
            rtol=1e-11,
            atol=2e-10,
        )


def test_coupled_ghost_force_work_extension_and_capacity_contract(
    coupled_ghost_component: MeshfreeTraceComponent,
) -> None:
    component = coupled_ghost_component
    points = np.asarray(component.owner.points)
    count = points.shape[0]
    ghosts = component.ghost_layer
    assert ghosts is not None
    boundary = component.system.plan.boundary
    charts = component.traces[0].charts
    flux_rows = tuple(
        np.asarray(boundary.condition(f"{field}-interface").rows) for field in ("x", "y")
    )
    gradients = np.asarray([[-0.7, 0.4], [0.3, -0.2]], dtype=np.float64)
    coupling = np.asarray([[3.0, 0.6], [0.6, 2.0]], dtype=np.float64)
    full_exact = 2.0 + np.asarray(ghosts.points) @ gradients.T
    values = (
        jnp.asarray(full_exact[:count, 0]),
        jnp.asarray(full_exact[:count, 1]),
        jnp.asarray(full_exact[count:, 0]),
        jnp.asarray(full_exact[count:, 1]),
    )
    measure = np.asarray(charts.measure)
    conormal = -(coupling @ gradients[:, 0])
    for index, field in enumerate(("x", "y")):
        trace = component.prepare_side_trace(
            field,
            component.boundary_domain(field, f"{field}-interface"),
            rule=FacetTraceRule("gauss-lobatto-legendre", points=3),
        )
        reaction = np.asarray(
            component.prepare_conormal_flux(trace).evaluate(values, None)
        )
        rows = np.asarray(trace.support_rows)
        owned = np.isin(rows, flux_rows[index])
        np.testing.assert_allclose(
            reaction[owned],
            measure[rows[owned]] * conormal[index],
            rtol=1e-9,
            atol=2e-10,
        )
        virtual = 0.4 + points[:, 1]
        np.testing.assert_allclose(
            reaction[owned] @ virtual[rows[owned]],
            conormal[index]
            * np.sum(measure[flux_rows[index]] * virtual[flux_rows[index]]),
            rtol=1e-9,
            atol=2e-10,
        )
    named = {
        record.name: value for record, value in zip(component.fields, values, strict=True)
    }
    np.testing.assert_allclose(component.ghost_extension_defect(named), 0.0, atol=2e-11)
    named["y-ghost"] = named["y-ghost"] + 0.25
    np.testing.assert_allclose(component.ghost_extension_defect(named), 0.25, atol=2e-11)
    for field in ("x", "y"):
        with pytest.raises(ValueError, match="no transient capacity"):
            component.prepare_capacity(field)
    for field in ("x-ghost", "y-ghost"):
        np.testing.assert_array_equal(
            component.prepare_capacity(field)
            .operator(None)
            .mv(jnp.ones(ghosts.ghost_count)),
            np.zeros(ghosts.ghost_count, dtype=np.float64),
        )
    with pytest.raises(ValueError, match="pointwise reaction"):
        MeshfreeTraceComponent(
            component.system,
            component.reconstruction,
            component.traces,
            name="invalid",
            reaction=MeshfreeReaction(lambda u, x, args: u**3, reaction_id="cubic"),
        )


@pytest.mark.parametrize("field", ("x", "y"), ids=("all-face-rows", "interior-face-rows"))
def test_block_ghost_flux_observation_retains_cross_component_force(
    coupled_ghost_component: MeshfreeTraceComponent, field: str
) -> None:
    component = coupled_ghost_component
    ghosts = component.ghost_layer
    assert ghosts is not None
    count = component.owner.points.shape[0]
    gradients = np.asarray([[-0.7, 0.4], [0.3, -0.2]], dtype=np.float64)
    coupling = np.asarray([[3.0, 0.6], [0.6, 2.0]], dtype=np.float64)
    exact = 2.0 + np.asarray(ghosts.points) @ gradients.T
    fields = {
        (component.name, name): jnp.asarray(values)
        for name, values in (
            ("x", exact[:count, 0]),
            ("y", exact[:count, 1]),
            ("x-ghost", exact[count:, 0]),
            ("y-ghost", exact[count:, 1]),
        )
    }
    observation = FieldFluxObservation(
        f"{field}-force",
        component.name,
        field,
        component.boundary_domain(field, f"{field}-interface"),
        rule=FacetTraceRule("gauss-lobatto-legendre", points=3),
        quantity=QuantitySpec("plate", "force", "force", ONE, "nondimensional-force"),
        support=IndexSampleSupport((1,), ("interface",)),
        sampling=SamplingSemantics(SpatialSamplingKind.PATH_INTEGRAL),
        reaction_unit=ONE,
    ).prepare({component.name: component})
    prediction = observation.evaluate(fields, {component.name: None})
    rows = np.asarray(component.system.plan.boundary.condition(f"{field}-interface").rows)
    index = ("x", "y").index(field)
    expected = -(coupling @ gradients[:, 0])[index] * np.sum(
        np.asarray(component.traces[index].charts.measure)[rows]
    )
    np.testing.assert_allclose(prediction.values, (expected,), rtol=1e-9, atol=2e-10)
    np.testing.assert_array_equal(prediction.valid_mask, (True,))
