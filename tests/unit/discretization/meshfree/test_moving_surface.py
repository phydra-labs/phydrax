# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Stage-correct moving surfaces against independent analytical oracles.

Oracles: uniform dilution exp(-k t)/R(t)^2; sphere and circle mean-curvature
flow R(t)^2 = R0^2 - 2 (d) t for intrinsic dimension d; the measure-rate
identity dw/dt = w (H V_n + div_G u_tau) with H = d/R on the unit sphere and
div_G grad_G z = -2 z; rigid rotation in closed form. Temporal orders are
measured against a fine ARS(4,4,3) reference and, for the geometry, against
the closed-form radius of dR/dt = A/R.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax as phx
from examples.meshfree_moving_surface_reaction_diffusion import (
    perform_epoch_transition,
    prepare_epoch_transition,
    prepare_workflow,
)
from phydrax.discretization import TopologyEpoch
from phydrax.discretization.meshfree import (
    AbstractSurfaceMotionLaw,
    BulkDrivenMotion,
    ChartMotion,
    ImplicitSurfaceGeometry,
    LevelSetMotion,
    LocalStencilPolicy,
    MeanCurvatureMotion,
    MeshfreeCapacityMap,
    MeshfreeCapacityPolicy,
    MeshfreeMetricPolicy,
    MovingArchiveMode,
    MovingGCLPolicy,
    MovingGeometryRefresh,
    MovingMeasureLaw,
    MovingSurfacePlan,
    MovingSurfaceState,
    MovingSurfaceStatus,
    MovingSurfaceStepResult,
    PreparedSurfacePointCloud,
    PrescribedVelocityMotion,
    SampledSurfaceGeometry,
    SmoothSupportEnvelope,
    SurfaceGeometryProvider,
    SurfaceMeshShift,
    SurfacePointCloudPlan,
    SurfaceQuadraturePolicy,
    SurfaceRefreshMode,
    SurfaceShiftPolicy,
)
from phydrax.linalg import (
    ArraySpace,
    FunctionLinearOperator,
    GMRES,
    LinearSolvePolicy,
    TolerancePolicy,
)
from phydrax.metrix import RegularLevelSetManifold
from phydrax.solver.advanced import AdditiveIMEXScheme


type Geometry = Callable[[Array, Array, Any], MovingGeometryRefresh]
type Reaction = Callable[[Array, Array, Array, Any], Array]


def _fibonacci_sphere(count: int) -> np.ndarray:
    index = np.arange(count, dtype=np.float64) + 0.5
    z = 1 - 2 * index / count
    azimuth = math.pi * (3 - math.sqrt(5)) * index
    radial = np.sqrt(1 - z * z)
    return np.column_stack((radial * np.cos(azimuth), radial * np.sin(azimuth), z))


@functools.cache
def _sampled_sphere(count: int) -> PreparedSurfacePointCloud:
    # Degree-4 oversampled height fits bound the curvature/area-rate bias; the
    # smooth fixed-radius support (radius about 2.3 spacings, displacement
    # envelope a tenth of it) makes the refresh trust the declared envelope.
    points = jnp.asarray(_fibonacci_sphere(count))
    radius = 0.5 * math.sqrt(256 / count)
    return SurfacePointCloudPlan(
        points,
        SampledSurfaceGeometry(
            reference_normals=points,
            fit_degree=4,
            oversampling=1.2,
            geometry_id="sampled-sphere",
        ),
        48,
        quadrature=SurfaceQuadraturePolicy(
            "normalized-density",
            density=jnp.ones(count, dtype=jnp.float64),
            total_area=4 * math.pi,
        ),
        stencil_policy=LocalStencilPolicy(
            polynomial_degree=2, support=SmoothSupportEnvelope(radius, 0.1 * radius)
        ),
    ).prepare()


@functools.cache
def _implicit_unit(count: int, ambient: int) -> PreparedSurfacePointCloud:
    """Unit circle (ambient 2) or sphere (ambient 3) with an exact level set."""
    if ambient == 2:
        angle = 2 * math.pi * (np.arange(count, dtype=np.float64) + 0.25) / count
        points = np.column_stack((np.cos(angle), np.sin(angle)))
        # Self plus three neighbors per side: an even count would tie the
        # symmetric last neighbors and leave a zero support trust margin.
        neighbors, area = 7, 2 * math.pi
    else:
        points, neighbors, area = _fibonacci_sphere(count), 16, 4 * math.pi

    def constraint(point: Array) -> Array:
        return jnp.asarray([jnp.dot(point, point) - 1])

    source = RegularLevelSetManifold(
        constraint,
        ambient_dimension=ambient,
        codimension=1,
        manifold_id=f"unit-sphere-{ambient}",
    )
    return SurfacePointCloudPlan(
        jnp.asarray(points),
        ImplicitSurfaceGeometry(
            source, certified_tube_radius=0.5, geometry_id=f"unit-sphere-{ambient}"
        ),
        neighbors,
        quadrature=SurfaceQuadraturePolicy(
            "normalized-density",
            density=jnp.ones(count, dtype=jnp.float64),
            total_area=area,
        ),
        stencil_policy=LocalStencilPolicy(polynomial_degree=2),
        require_tube=True,
    ).prepare()


def _trust_step(
    surface: PreparedSurfacePointCloud, speed: float, fraction: float = 0.25
) -> float:
    """A step moving every point by ``fraction`` of the support trust margin."""
    trust = float(np.min(np.asarray(surface.neighborhood.trust_margin)))
    return fraction * trust / speed


def _no_diffusion(count: int) -> FunctionLinearOperator:
    space = ArraySpace((count,), dtype=np.float64)

    def zero(value: Array) -> Array:
        return jnp.zeros_like(value)

    return FunctionLinearOperator(
        zero, transpose_action=zero, source=space, target=space, operator_id="none"
    )


def _no_reaction(time: Array, points: Array, concentration: Array, args: Any) -> Array:
    return jnp.zeros_like(concentration)


def _surface_plan(
    surface: PreparedSurfacePointCloud,
    motion: AbstractSurfaceMotionLaw,
    /,
    *,
    method: AdditiveIMEXScheme = "ars-222",
    mode: SurfaceRefreshMode = "similarity",
    history_capacity: int = 4,
    gcl: MovingGCLPolicy | None = None,
    shift: SurfaceMeshShift | None = None,
    require_positivity: bool = False,
    measure_law: MovingMeasureLaw = "geometric",
) -> MovingSurfacePlan:
    count = surface.points.shape[0]
    capacity = MeshfreeCapacityPolicy((count,)).allocate(count)
    return MovingSurfacePlan(
        SurfaceGeometryProvider(surface, _no_diffusion(count), mode=mode),
        motion,
        _no_reaction,
        method=method,
        epoch=TopologyEpoch(
            0, "moving-test", surface.neighborhood.neighborhood_id, capacity.mapping_id
        ),
        capacity=capacity,
        history_capacity=history_capacity,
        plan_id=f"moving-test:{motion.law_id}",
        gcl=gcl,
        shift=shift,
        require_positivity=require_positivity,
        measure_law=measure_law,
    )


def _replan(
    plan: MovingSurfacePlan,
    /,
    *,
    geometry: Geometry | None = None,
    motion: AbstractSurfaceMotionLaw | None = None,
    reaction: Reaction | None = None,
    method: AdditiveIMEXScheme = "ars-222",
    history_capacity: int = 12,
    archive: MovingArchiveMode = "rolling",
    gcl: MovingGCLPolicy | None = None,
    linear_policy: LinearSolvePolicy | None = None,
    shift: SurfaceMeshShift | None = None,
) -> MovingSurfacePlan:
    return MovingSurfacePlan(
        plan.geometry if geometry is None else geometry,
        plan.motion if motion is None else motion,
        plan.reaction if reaction is None else reaction,
        method=method,
        epoch=plan.epoch,
        capacity=plan.capacity,
        history_capacity=history_capacity,
        archive=archive,
        gcl=gcl,
        linear_policy=linear_policy,
        shift=shift,
        plan_id="replanned",
    )


@eqx.filter_jit
def _step(
    plan: MovingSurfacePlan, state: MovingSurfaceState, dt: Array, args: Any = None
) -> MovingSurfaceStepResult:
    return plan.step(state, dt, args=args)


@functools.cache
def _workflow() -> tuple[MovingSurfacePlan, MovingSurfaceState]:
    return prepare_workflow(size=48)


def _advance(
    plan: MovingSurfacePlan, state: MovingSurfaceState, dt: float, steps: int
) -> MovingSurfaceState:
    for _ in range(steps):
        result = _step(plan, state, jnp.asarray(dt))
        assert int(result.evidence.status) == int(MovingSurfaceStatus.ACCEPTED), int(
            result.evidence.status
        )
        state = result.state
    return state


def _mean_radius(points: Array) -> float:
    """Mean distance to the origin, the center of every radial test motion."""
    return float(np.mean(np.linalg.norm(np.asarray(points), axis=1)))


@pytest.mark.parametrize(
    ("method", "exact"),
    [
        pytest.param(
            "forward-backward-euler",
            lambda dt: (1 - 0.1 * dt) / (1 + 0.2 * dt) ** 2,
            id="forward-backward-euler",
        ),
        pytest.param(
            "ars-222", lambda dt: math.exp(-0.1 * dt) / (1 + 0.2 * dt) ** 2, id="ars-222"
        ),
    ],
)
def test_growing_sphere_dilution_follows_the_selected_stage_rule(
    method: AdditiveIMEXScheme, exact: Callable[[float], float]
) -> None:
    plan, initial = prepare_workflow(size=48, method=method)
    result = _step(plan, initial, jnp.asarray(0.01))
    evidence = result.evidence
    assert bool(evidence.successful)
    assert bool(evidence.content_conserved) and bool(evidence.gcl_admitted)
    np.testing.assert_allclose(result.state.concentration, exact(0.01), atol=1e-9)
    np.testing.assert_allclose(evidence.conservation_residual, 0.0, atol=1e-12)
    # Measures follow the authoritative chart exactly: w = w0 R(t)^2.
    np.testing.assert_allclose(
        result.state.measures, initial.measures * (1 + 0.2 * 0.01) ** 2, rtol=1e-12
    )


def test_level_set_gradient_floor_owns_validity_and_checkpoint_identity() -> None:
    base, initial = _workflow()

    def level_set(point: Array, time: Array, args: Any) -> Array:
        return 1e-8 * (jnp.dot(point, point) - (1 + 0.2 * time) ** 2)

    admitted = LevelSetMotion(
        level_set, law_id="scaled-growing-sphere", gradient_floor=1e-12
    )
    canonical = LevelSetMotion(
        level_set, law_id="scaled-growing-sphere", gradient_floor=np.float64(1e-12)
    )
    refusing = LevelSetMotion(
        level_set, law_id="scaled-growing-sphere", gradient_floor=1e-6
    )
    geometry = base.geometry(initial.points, initial.time, None)
    assert bool(admitted.motion(initial.time, geometry, None).valid)
    assert not bool(refusing.motion(initial.time, geometry, None).valid)
    assert admitted.law_id != refusing.law_id
    assert admitted.law_id == canonical.law_id

    admitted_plan = _replan(base, motion=admitted)
    canonical_plan = _replan(base, motion=canonical)
    refusing_plan = _replan(base, motion=refusing)
    assert admitted_plan.plan_id != refusing_plan.plan_id
    assert admitted_plan.plan_id == canonical_plan.plan_id
    state = admitted_plan.initialize(initial.points, np.ones(initial.points.shape[0]))
    checkpoint = admitted_plan.checkpoint(state)
    assert canonical_plan.rollback(checkpoint) is state
    with pytest.raises(ValueError, match="different moving plan"):
        refusing_plan.rollback(checkpoint)
    with pytest.raises(ValueError, match="motion is not admitted"):
        refusing_plan.initialize(initial.points, np.ones(initial.points.shape[0]))


def test_nonuniform_concentration_diffuses_conservatively_at_stage_geometry() -> None:
    plan, initial = _workflow()
    perturbed = plan.initialize(
        initial.points, 1 + 0.2 * np.asarray(initial.points[:, 2])
    )
    diffused = _step(plan, perturbed, jnp.asarray(0.1))
    assert bool(diffused.evidence.successful)
    scaled = diffused.state.concentration * (1 + 0.2 * 0.1) ** 2 / math.exp(-0.1 * 0.1)
    assert float(jnp.var(scaled)) < float(jnp.var(perturbed.concentration))
    np.testing.assert_allclose(diffused.evidence.conservation_residual, 0.0, atol=1e-10)
    np.testing.assert_allclose(
        diffused.evidence.reaction_content,
        diffused.evidence.content_after - diffused.evidence.content_before,
        atol=1e-10,
    )


_ORDER_SPEED = 0.5
_ORDER_TIME = 0.4


def _order_reaction(time: Array, points: Array, concentration: Array, args: Any) -> Array:
    return -0.8 * (1 + 0.5 * jnp.sin(3 * time)) * concentration**2


@functools.cache
def _order_plan(method: AdditiveIMEXScheme) -> MovingSurfacePlan:
    return _replan(
        _workflow()[0],
        motion=PrescribedVelocityMotion(
            lambda time, points, args: (
                _ORDER_SPEED * points / jnp.sum(points**2, axis=1)[:, None]
            ),
            tangential="normal-only",
            law_id="radial-inverse-radius",
        ),
        reaction=_order_reaction,
        method=method,
        gcl=MovingGCLPolicy(admission="diagnostic"),
    )


def _order_run(method: AdditiveIMEXScheme, steps: int) -> tuple[np.ndarray, float]:
    plan = _order_plan(method)
    initial = _workflow()[1]
    state = plan.initialize(initial.points, 1 + 0.3 * np.asarray(initial.points[:, 2]))
    final = _advance(plan, state, _ORDER_TIME / steps, steps)
    return np.asarray(final.concentration), _mean_radius(final.points)


@functools.cache
def _order_reference() -> np.ndarray:
    return _order_run("ars-443", 64)[0]


@pytest.mark.parametrize(
    ("method", "order"),
    [
        pytest.param("forward-backward-euler", 1, id="first-order"),
        pytest.param("ars-222", 2, id="second-order"),
        pytest.param("ars-443", 3, id="third-order"),
    ],
)
def test_temporal_refinement_attains_the_selected_imex_order(
    method: AdditiveIMEXScheme, order: int
) -> None:
    reference = _order_reference()
    exact_radius = math.sqrt(1 + 2 * _ORDER_SPEED * _ORDER_TIME)
    field_errors, radius_errors = [], []
    for steps in (8, 16, 32):
        concentration, radius = _order_run(method, steps)
        field_errors.append(float(np.max(np.abs(concentration - reference))))
        radius_errors.append(abs(radius - exact_radius))
    field_rates = np.log2(np.asarray(field_errors[:-1]) / np.asarray(field_errors[1:]))
    radius_rates = np.log2(np.asarray(radius_errors[:-1]) / np.asarray(radius_errors[1:]))
    assert field_rates[-1] > order - 0.3, (field_errors, field_rates)
    assert radius_rates[-1] > order - 0.3, (radius_errors, radius_rates)


def _mean_curvature_flow(
    count: int, ambient: int, dt: float, steps: int
) -> tuple[float, float]:
    surface = _implicit_unit(count, ambient)
    plan = _surface_plan(
        surface,
        MeanCurvatureMotion(1.0, law_id=f"mcf-{ambient}"),
        gcl=MovingGCLPolicy(admission="diagnostic"),
    )
    state = plan.initialize(surface.points, np.ones(count))
    final = _advance(plan, state, dt, steps)
    np.testing.assert_allclose(final.content, state.content, atol=1e-12)
    return _mean_radius(final.points), float(np.sum(np.asarray(final.measures)))


@pytest.mark.parametrize(
    ("ambient", "counts", "dt", "area"),
    [
        # R^2 = 1 - 2 d t with intrinsic dimension d; R(t_end) = sqrt(0.6) in
        # both cases, well before the singularity at R = 0.
        pytest.param(2, (48, 96), 0.004, lambda r: 2 * math.pi * r, id="circle"),
        pytest.param(3, (128, 256), 0.002, lambda r: 4 * math.pi * r**2, id="sphere"),
    ],
)
def test_mean_curvature_flow_converges_to_the_analytic_radius(
    ambient: int,
    counts: tuple[int, int],
    dt: float,
    area: Callable[[float], float],
) -> None:
    exact = math.sqrt(0.6)
    coarse, fine = (_mean_curvature_flow(count, ambient, dt, 50) for count in counts)
    radius_errors = [abs(radius - exact) for radius, _ in (coarse, fine)]
    area_errors = [abs(total - area(exact)) / area(exact) for _, total in (coarse, fine)]
    # Spatial error of the degree-2 Laplace--Beltrami dominates the ARS(2,2,2)
    # temporal error; doubling the samples shrinks h by 2 (circle) or sqrt(2)
    # (sphere), and the second-order operator error must follow.
    reduction = 3.0 if ambient == 2 else 1.6
    assert radius_errors[1] < radius_errors[0] / reduction, radius_errors
    assert area_errors[1] < area_errors[0] / reduction, area_errors
    assert radius_errors[1] < (3e-3 if ambient == 2 else 2e-2), radius_errors


def test_authoritative_rotating_chart_cannot_discard_tangential_velocity() -> None:
    surface = _implicit_unit(64, 2)

    def rotation(reference: Array, time: Array) -> Array:
        cosine, sine = jnp.cos(time), jnp.sin(time)
        return jnp.stack(
            (
                cosine * reference[:, 0] - sine * reference[:, 1],
                sine * reference[:, 0] + cosine * reference[:, 1],
            ),
            axis=1,
        )

    chart = eqx.Partial(rotation, surface.points)
    with pytest.raises(
        ValueError,
        match="cannot discard tangential chart velocity.*PrescribedVelocityMotion",
    ):
        ChartMotion(chart, tangential="normal-only", law_id="rotating-circle")

    material = ChartMotion(chart, tangential="material", law_id="rotating-circle")
    time = jnp.asarray(0.0, dtype=jnp.float64)
    geometry = SurfaceGeometryProvider(surface, _no_diffusion(64))(
        surface.points, time, None
    )
    motion = material.motion(time, geometry, None)
    assert bool(motion.valid)
    np.testing.assert_allclose(
        motion.mesh_velocity,
        jnp.stack((-surface.points[:, 1], surface.points[:, 0]), axis=1),
        atol=1e-14,
    )
    np.testing.assert_array_equal(
        motion.relative_velocity, jnp.zeros_like(surface.points)
    )


def test_sampled_sphere_rotates_over_many_steps_inside_its_support_envelope() -> None:
    surface = _sampled_sphere(256)
    envelope = float(np.min(np.asarray(surface.neighborhood.trust_margin)))
    rate, dt = 0.5, 0.01

    def rotation(time: Array, points: Array, args: Any) -> Array:
        return rate * jnp.stack(
            (-points[:, 1], points[:, 0], jnp.zeros_like(points[:, 2])), axis=1
        )

    # A rotation is divergence free; the sampled strong divergence of it is
    # not exactly zero, so its GCL defect is reported rather than admitted.
    plan = _surface_plan(
        surface,
        PrescribedVelocityMotion(rotation, law_id="rigid-rotation"),
        mode="fixed-support",
        gcl=MovingGCLPolicy(admission="diagnostic"),
    )
    state = plan.initialize(surface.points, 1 + np.asarray(surface.points[:, 0]))
    steps = int(0.9 * envelope / (rate * dt))
    assert steps >= 8
    final = _advance(plan, state, dt, steps)
    angle = steps * rate * dt
    cosine, sine = math.cos(angle), math.sin(angle)
    x0 = np.asarray(state.points)
    exact = np.column_stack(
        (
            cosine * x0[:, 0] - sine * x0[:, 1],
            sine * x0[:, 0] + cosine * x0[:, 1],
            x0[:, 2],
        )
    )
    # ARS(2,2,2) error (rate dt)^3/6 per step; the integrated points are an
    # isometry only to that accuracy, and so are their fitted measures.
    np.testing.assert_allclose(final.points, exact, atol=1e-6)
    np.testing.assert_allclose(final.content, state.content, atol=1e-14)
    np.testing.assert_allclose(final.measures, state.measures, rtol=1e-7)
    # The smooth-support envelope bounds every anchored displacement; an
    # isometry beyond it is refused, not silently admitted.
    refused = _step(plan, final, jnp.asarray(0.3 * envelope / rate))
    assert int(refused.evidence.status) == int(MovingSurfaceStatus.TRUST_REFUSED)
    np.testing.assert_array_equal(refused.state.points, final.points)


def test_nonuniform_normal_motion_has_an_independent_measure_rate() -> None:
    surface = _sampled_sphere(256)
    speed, contrast = 0.3, 0.5

    def normal(time: Array, points: Array, args: Any) -> Array:
        return speed * (1 + contrast * points[:, 2:3]) * points

    law = PrescribedVelocityMotion(
        normal, tangential="normal-only", law_id="nonuniform-normal"
    )
    # The sampled degree-4 curvature is accurate to about 1% at this
    # resolution; the declared GCL rate tolerance covers the difference
    # between that rate and the independently fitted area Jacobian.
    plan = _surface_plan(
        surface, law, mode="fixed-support", gcl=MovingGCLPolicy(rate_tolerance=0.1)
    )
    state = plan.initialize(surface.points, np.ones(256))
    geometry = plan.geometry(state.points, state.time, None)
    motion = law.motion(state.time, geometry, None)
    z = np.asarray(state.points[:, 2])
    analytic = np.asarray(state.measures) * 2 * speed * (1 + contrast * z)
    scale = float(np.max(analytic))
    np.testing.assert_allclose(motion.tangential_measure_rate, 0.0, atol=0)
    np.testing.assert_allclose(motion.measure_rate, analytic, atol=2e-2 * scale)
    dt = _trust_step(surface, speed * (1 + contrast), fraction=0.1)
    stepped = _step(plan, state, jnp.asarray(dt))
    assert bool(stepped.evidence.successful)
    assert bool(stepped.evidence.gcl_admitted)
    # The refreshed area Jacobian is an independent discretization of dw/dt.
    observed = (np.asarray(stepped.state.measures) - np.asarray(state.measures)) / dt
    np.testing.assert_allclose(observed, analytic, atol=5e-2 * scale)


def _redistribution(strength: float) -> PrescribedVelocityMotion:
    def velocity(time: Array, points: Array, args: Any) -> Array:
        axis = jnp.asarray([0.0, 0.0, 1.0])
        return strength * (axis[None, :] - points[:, 2:3] * points)

    return PrescribedVelocityMotion(velocity, law_id="gradient-z-redistribution")


def test_tangential_redistribution_measure_rate_converges_to_the_divergence() -> None:
    strength = 0.2
    law = _redistribution(strength)
    errors = []
    for count in (128, 256):
        surface = _sampled_sphere(count)
        plan = _surface_plan(surface, law, mode="fixed-support")
        state = plan.initialize(surface.points, 1 + np.asarray(surface.points[:, 1]))
        motion = law.motion(
            state.time, plan.geometry(state.points, state.time, None), None
        )
        # u = s grad_G z on the unit sphere, so w div_G u = -2 s z w.
        analytic = (
            -2 * strength * np.asarray(state.points[:, 2]) * np.asarray(state.measures)
        )
        # The velocity is tangent to the exact sphere; its normal speed against
        # the fitted sample normals is bounded by the normal-estimate accuracy.
        np.testing.assert_allclose(motion.normal_speed, 0.0, atol=2e-3 * strength)
        errors.append(
            float(np.max(np.abs(np.asarray(motion.measure_rate) - analytic)))
            / float(np.max(np.abs(analytic)))
        )
    # Degree-2 strong divergence: halving h^2 must shrink the error (h by sqrt 2).
    assert errors[1] < errors[0] / 1.6 and errors[1] < 0.12, errors


def test_tangential_redistribution_conserves_content_on_the_new_measures() -> None:
    strength = 0.2
    surface = _sampled_sphere(256)
    # The declared GCL rate tolerance covers the spatial accuracy of the
    # sampled divergence at this resolution.
    plan = _surface_plan(
        surface,
        _redistribution(strength),
        mode="fixed-support",
        gcl=MovingGCLPolicy(rate_tolerance=0.1),
    )
    state = plan.initialize(surface.points, 1 + np.asarray(surface.points[:, 1]))
    stepped = _step(plan, state, jnp.asarray(_trust_step(surface, strength)))
    assert bool(stepped.evidence.successful)
    np.testing.assert_allclose(
        np.sum(np.asarray(stepped.state.content)),
        np.sum(np.asarray(state.content)),
        rtol=1e-14,
    )
    np.testing.assert_allclose(
        stepped.state.concentration,
        np.asarray(stepped.state.content) / np.asarray(stepped.state.measures),
        rtol=1e-14,
    )


@functools.cache
def _sampled_shift() -> SurfaceMeshShift:
    surface = _sampled_sphere(256)
    exterior = surface.conservative_exterior(
        0.45,
        8192,
        metric_policy=MeshfreeMetricPolicy(sign="nonnegative", acceptance="relaxed"),
    )
    lengths = np.asarray(exterior.lengths)
    return SurfaceMeshShift(
        SurfaceShiftPolicy(
            strength=1.0, target_separation=1.05 * float(np.min(lengths[lengths > 0]))
        ),
        exterior,
    )


def _shift_plan(
    measure_law: MovingMeasureLaw = "conservative",
) -> tuple[PreparedSurfacePointCloud, MovingSurfacePlan]:
    surface = _sampled_sphere(256)
    rest = PrescribedVelocityMotion(
        lambda time, points, args: jnp.zeros_like(points), law_id="material-at-rest"
    )
    return surface, _surface_plan(
        surface,
        rest,
        mode="fixed-support",
        shift=_sampled_shift(),
        require_positivity=True,
        measure_law=measure_law,
    )


def test_mesh_shift_redistributes_conservatively_inside_the_temporal_method() -> None:
    surface, plan = _shift_plan()
    initial = plan.initialize(surface.points, 1 + 0.5 * np.asarray(surface.points[:, 2]))
    state = initial
    for _ in range(10):
        result = _step(plan, state, jnp.asarray(0.05))
        evidence = result.evidence
        assert int(evidence.status) == int(MovingSurfaceStatus.ACCEPTED)
        assert bool(evidence.transport_valid) and bool(evidence.positivity_admitted)
        assert 0 < float(evidence.transport_cfl) < 1
        # The witness is the measure of record: stage-quadrature GCL holds by
        # construction, while the refreshed geometry reports its drift.
        np.testing.assert_allclose(evidence.gcl_defect, 0.0, atol=0)
        assert 0 < float(evidence.geometric_drift) < 0.1
        np.testing.assert_allclose(evidence.conservation_residual, 0.0, atol=1e-13)
        state = result.state
    displacement = np.linalg.norm(
        np.asarray(state.points) - np.asarray(initial.points), axis=1
    )
    # The mesh moved tangentially within the envelope while the material
    # content was only exchanged between nodes, never created.
    assert 1e-4 < float(np.max(displacement)) < 0.05
    np.testing.assert_allclose(
        np.linalg.norm(np.asarray(state.points), axis=1), 1.0, atol=1e-3
    )
    np.testing.assert_allclose(
        np.sum(np.asarray(state.content)), np.sum(np.asarray(initial.content)), rtol=1e-13
    )
    assert float(np.min(np.asarray(state.concentration))) > 0


def test_mesh_shift_preserves_a_constant_concentration_exactly() -> None:
    surface, plan = _shift_plan()
    state = plan.initialize(surface.points, np.full(256, 2.5))
    for _ in range(5):
        result = _step(plan, state, jnp.asarray(0.05))
        assert int(result.evidence.status) == int(MovingSurfaceStatus.ACCEPTED)
        state = result.state
    # Discrete GCL: content and measure advance by one graph divergence.
    np.testing.assert_allclose(state.concentration, 2.5, rtol=1e-13)
    assert (
        float(np.max(np.abs(np.asarray(state.measures) - np.asarray(surface.measures))))
        > 0
    )


def test_mesh_shift_refuses_inconsistent_or_authoritative_measures() -> None:
    with pytest.raises(ValueError, match="conservative measure law"):
        _shift_plan("geometric")
    plan, _state = _workflow()
    with pytest.raises(ValueError, match="authoritative chart"):
        _replan(plan, shift=_sampled_shift())


@functools.cache
def _enveloped_implicit_sphere() -> PreparedSurfacePointCloud:
    def constraint(point: Array) -> Array:
        return jnp.asarray([jnp.dot(point, point) - 1])

    source = RegularLevelSetManifold(
        constraint, ambient_dimension=3, codimension=1, manifold_id="unit-sphere-3"
    )
    return SurfacePointCloudPlan(
        jnp.asarray(_fibonacci_sphere(256)),
        ImplicitSurfaceGeometry(
            source, certified_tube_radius=0.5, geometry_id="unit-sphere-3"
        ),
        48,
        quadrature=SurfaceQuadraturePolicy(
            "normalized-density",
            density=jnp.ones(256, dtype=jnp.float64),
            total_area=4 * math.pi,
        ),
        stencil_policy=LocalStencilPolicy(
            polynomial_degree=2, support=SmoothSupportEnvelope(0.45, 0.04)
        ),
        require_tube=True,
    ).prepare()


def test_support_crossing_on_a_curved_surface_is_continuously_differentiable() -> None:
    surface = _enveloped_implicit_sphere()
    radius = 0.45
    relation = surface.relation
    indices = np.asarray(relation.source_indices)
    points = np.asarray(surface.points)
    valid = np.asarray(relation.valid) & (indices != np.arange(indices.shape[0])[:, None])
    distances = np.linalg.norm(points[indices] - points[:, None, :], axis=2)
    # A source just inside the physical cutoff of its row (a curved chord).
    near = valid & (distances < radius) & (distances > radius - 0.02)
    row, slot = (int(value) for value in np.argwhere(near)[0])
    source = indices[row, slot]
    start = points[source]
    outward = start - points[row]
    direction = outward - np.dot(outward, start) * start
    direction /= np.linalg.norm(direction)

    def moved(parameter: float) -> np.ndarray:
        position = start + parameter * direction
        cloud = points.copy()
        cloud[source] = position / np.linalg.norm(position)
        return cloud

    def distance(parameter: float) -> float:
        return float(np.linalg.norm(moved(parameter)[source] - points[row]))

    lower, upper = 0.0, 0.035
    assert distance(lower) < radius < distance(upper)
    for _ in range(80):
        middle = 0.5 * (lower + upper)
        lower, upper = (middle, upper) if distance(middle) < radius else (lower, middle)
    crossing = 0.5 * (lower + upper)

    @eqx.filter_jit
    def weights(cloud: Array) -> tuple[Array, Array]:
        refreshed = surface.refresh(cloud)
        return refreshed.prepared.laplace_beltrami.coefficients[row], refreshed.accepted

    step = 1e-5
    samples = {}
    for offset in (-2, -1, 0, 1, 2):
        values, accepted = weights(jnp.asarray(moved(crossing + offset * step)))
        assert bool(accepted)
        samples[offset] = np.asarray(values)
    # The crossing source has no weight outside the ambient support, and its
    # weight vanishes at the cutoff (compact kernel of order >= 1).
    np.testing.assert_allclose(samples[1][slot], 0.0, atol=0)
    np.testing.assert_allclose(samples[2][slot], 0.0, atol=0)
    assert abs(samples[-1][slot]) < 1e-6 * np.max(np.abs(samples[0]))
    # Inside the support the row is genuinely sensitive to the moving source.
    interior = [
        np.asarray(weights(jnp.asarray(moved(crossing - 0.01 + offset * step)))[0])
        for offset in (-1, 1)
    ]
    sensitivity = float(np.max(np.abs(interior[1] - interior[0]))) / (2 * step)
    assert sensitivity > 0
    # Across the cutoff the row and its one-sided derivatives are continuous:
    # no jump and no derivative kink relative to that interior sensitivity.
    np.testing.assert_allclose(samples[1], samples[-1], atol=1e-6 * sensitivity * step)
    left = (samples[0] - samples[-2]) / (2 * step)
    right = (samples[2] - samples[0]) / (2 * step)
    np.testing.assert_allclose(left, right, atol=1e-6 * sensitivity)


def _velocity_grid(lower: float, upper: float, growth: float) -> tuple[Any, Array]:
    d = phx.discretization
    names = ("x", "y", "z")
    grid = d.TensorGridPlan(
        tuple(d.UniformAxisSpec(5) for _ in names), axis_names=names
    ).prepare(jnp.asarray(((lower,) * 3, (upper,) * 3), dtype=jnp.float64))
    requests = tuple(
        d.DerivativeRequest(f"d{name}", grid, name, derivative_order=1, accuracy_order=2)
        for name in names
    )
    discretization = d.FiniteDifferencePlan(grid, requests, field_name="v").prepare()
    reconstruction = d.prepare_finite_difference_field_reconstruction(
        discretization,
        interpolation=d.MultilinearGridInterpolation(),
        value_port=phx.ValuePort(
            "bulk-velocity",
            event_shape=(3,),
            component_ids=("vx", "vy", "vz"),
            representation="finite-difference-field",
        ),
    )
    axes = [
        np.asarray(values)
        for values in discretization.grid.primary_entity_layout.coordinates_by_axis
    ]
    nodes = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    return reconstruction, jnp.asarray(growth * nodes)


def test_bulk_driven_motion_reproduces_dilation_and_refuses_unsupported_queries() -> None:
    growth = 0.3
    base, initial = _workflow()
    reconstruction, velocity = _velocity_grid(-1.5, 1.5, growth)

    def bulk(time: Array, args: Any) -> Array:
        return args

    plan = _replan(
        base,
        motion=BulkDrivenMotion(reconstruction, bulk, law_id="bulk-dilation"),
        reaction=_no_reaction,
    )
    state = plan.initialize(
        initial.points, np.ones(initial.points.shape[0]), args=velocity
    )
    for _ in range(10):
        result = _step(plan, state, jnp.asarray(0.01), velocity)
        assert bool(result.evidence.successful)
        state = result.state
    # Multilinear reconstruction of v = g x is exact, so R(t) = exp(g t).
    np.testing.assert_allclose(
        _mean_radius(state.points), math.exp(growth * 0.1), rtol=1e-6
    )
    uncovered, uncovered_velocity = _velocity_grid(0.5, 1.5, growth)
    refusing = _replan(
        base,
        motion=BulkDrivenMotion(uncovered, bulk, law_id="uncovered-bulk"),
        reaction=_no_reaction,
    )
    refused = _step(refusing, state, jnp.asarray(0.01), uncovered_velocity)
    assert int(refused.evidence.status) == int(MovingSurfaceStatus.MOTION_REFUSED)
    assert not bool(refused.evidence.motion_valid)
    np.testing.assert_array_equal(refused.state.points, state.points)


def _overridden(
    plan: MovingSurfacePlan,
    transform: Callable[[MovingGeometryRefresh], MovingGeometryRefresh],
) -> Geometry:
    def geometry(points: Array, time: Array, args: Any, /) -> MovingGeometryRefresh:
        return transform(plan.geometry(points, time, args))

    return geometry


def _leaky(refreshed: MovingGeometryRefresh) -> MovingGeometryRefresh:
    space = ArraySpace(refreshed.measures.shape, dtype=np.float64)

    def leak(value: Array) -> Array:
        return -0.1 * value

    operator = FunctionLinearOperator(
        leak, transpose_action=leak, source=space, target=space, operator_id="leaky"
    )
    return eqx.tree_at(lambda item: item.diffusion, refreshed, operator)


_REFUSALS = (
    "trust_valid",
    "tube_valid",
    "geometry_valid",
    "diffusion_conservative",
    "solver_successful",
    "gcl_admitted",
    "history_admitted",
    "content_conserved",
)


@pytest.mark.parametrize(
    ("case", "status", "flag"),
    [
        pytest.param(
            "trust", MovingSurfaceStatus.TRUST_REFUSED, "trust_valid", id="trust"
        ),
        pytest.param("tube", MovingSurfaceStatus.TUBE_REFUSED, "tube_valid", id="tube"),
        pytest.param(
            "nonconservative",
            MovingSurfaceStatus.NONCONSERVATIVE_DIFFUSION,
            "diffusion_conservative",
            id="nonconservative-diffusion",
        ),
        pytest.param(
            "implicit",
            MovingSurfaceStatus.IMPLICIT_REFUSED,
            "solver_successful",
            id="implicit",
        ),
        pytest.param("gcl", MovingSurfaceStatus.GCL_REFUSED, "gcl_admitted", id="gcl"),
        pytest.param(
            "history",
            MovingSurfaceStatus.HISTORY_EXHAUSTED,
            "history_admitted",
            id="archive-capacity",
        ),
        pytest.param(
            "off-chart",
            MovingSurfaceStatus.INVALID_GEOMETRY,
            "geometry_valid",
            id="off-chart",
        ),
    ],
)
def test_forced_refusal_rolls_back_atomically_with_independent_status(
    case: str, status: MovingSurfaceStatus, flag: str
) -> None:
    base, initial = _workflow()
    match case:
        case "trust":
            plan = _replan(
                base,
                geometry=_overridden(
                    base,
                    lambda item: eqx.tree_at(
                        lambda g: g.trust_valid, item, jnp.asarray(False)
                    ),
                ),
            )
        case "tube":
            plan = _replan(
                base,
                geometry=_overridden(
                    base,
                    lambda item: eqx.tree_at(
                        lambda g: g.tube_valid, item, jnp.asarray(False)
                    ),
                ),
            )
        case "nonconservative":
            plan = _replan(base, geometry=_overridden(base, _leaky))
        case "implicit":
            plan = _replan(
                base,
                linear_policy=LinearSolvePolicy(
                    GMRES(),
                    tolerance=TolerancePolicy(relative=1e-15, absolute=0.0, max_steps=1),
                ),
            )
        case "gcl":
            plan = _replan(
                base,
                method="forward-backward-euler",
                gcl=MovingGCLPolicy(rate_tolerance=1e-14),
            )
        case "history":
            plan = _replan(base, history_capacity=1, archive="acknowledged")
        case "off-chart":
            plan = base
        case _:
            raise AssertionError(case)
    # Initialization admits geometry on the host; a forced trust/tube defect
    # must therefore start from a state admitted by the unforced geometry.
    admitting = base if case in ("trust", "tube") else plan
    state = admitting.initialize(
        initial.points, 1 + 0.2 * np.asarray(initial.points[:, 2])
    )
    if case == "off-chart":
        state = eqx.tree_at(lambda item: item.points, state, state.points * (1 + 1e-6))
    result = _step(plan, state, jnp.asarray(0.05))
    evidence = result.evidence
    flags = {
        "trust_valid": evidence.trust_valid,
        "tube_valid": evidence.tube_valid,
        "geometry_valid": evidence.geometry_valid,
        "diffusion_conservative": evidence.diffusion_conservative,
        "solver_successful": evidence.solver_successful,
        "gcl_admitted": evidence.gcl_admitted,
        "history_admitted": evidence.history_admitted,
        "content_conserved": evidence.content_conserved,
    }
    assert int(evidence.status) == int(status)
    assert not bool(evidence.successful)
    assert not bool(flags[flag])
    if case not in ("nonconservative", "implicit"):
        # The forced defect leaves every other admission independently intact.
        assert all(bool(flags[name]) for name in _REFUSALS if name != flag), case
    for refused, original in zip(
        jax.tree.leaves(result.state), jax.tree.leaves(state), strict=True
    ):
        np.testing.assert_array_equal(refused, original)


def test_plan_refuses_a_tableau_without_shared_stage_times() -> None:
    base, _initial = _workflow()
    with pytest.raises(ValueError, match="shared stage time"):
        _replan(base, method="ssp2-222")


def test_rolling_live_history_runs_beyond_its_capacity() -> None:
    base, initial = _workflow()
    plan = _replan(base, history_capacity=3)
    state = plan.initialize(initial.points, np.ones(initial.points.shape[0]))
    contents = [np.asarray(state.content)]
    for _ in range(10):
        result = _step(plan, state, jnp.asarray(0.01))
        assert bool(result.evidence.successful)
        state = result.state
        contents.append(np.asarray(state.content))
    assert int(state.accepted_steps) == 10 and int(state.history_count) == 3
    np.testing.assert_array_equal(state.live_lifetimes(), [8, 9, 10])
    for slot, lifetime in zip(state.live_slots(), state.live_lifetimes(), strict=True):
        np.testing.assert_array_equal(state.history_content[slot], contents[lifetime])
    np.testing.assert_allclose(
        state.history_times[state.live_slots()], [0.08, 0.09, 0.10], atol=1e-14
    )


def test_acknowledged_archive_applies_backpressure_until_acknowledged() -> None:
    base, initial = _workflow()
    plan = _replan(base, history_capacity=2, archive="acknowledged")
    state = plan.initialize(initial.points, np.ones(initial.points.shape[0]))
    state = _step(plan, state, jnp.asarray(0.01)).state
    blocked = _step(plan, state, jnp.asarray(0.01))
    assert int(blocked.evidence.status) == int(MovingSurfaceStatus.HISTORY_EXHAUSTED)
    assert int(blocked.evidence.evicted_lifetime) == 0
    np.testing.assert_array_equal(blocked.state.history_content, state.history_content)
    with pytest.raises(ValueError, match="monotonically"):
        plan.acknowledge_archive(state, 3)
    archived = plan.acknowledge_archive(state, 1)
    released = _step(plan, archived, jnp.asarray(0.01))
    assert bool(released.evidence.successful)
    assert int(released.evidence.evicted_lifetime) == 0
    np.testing.assert_array_equal(released.state.live_lifetimes(), [1, 2])


def test_epoch_remaps_every_live_history_and_preserves_positive_active_spaces() -> None:
    plan, state = _workflow()
    state = _step(plan, state, jnp.asarray(0.01)).state
    slots = state.live_slots()
    before = np.sum(np.asarray(state.history_content[slots]), axis=1)
    next_plan, transitioned = perform_epoch_transition(plan, state)
    assert transitioned.successful and not transitioned.differentiation_available
    assert transitioned.state.epoch.index == state.epoch.index + 1
    assert next_plan.epoch.epoch_id == transitioned.state.epoch.epoch_id
    assert int(transitioned.state.history_count) == 2
    np.testing.assert_array_equal(transitioned.state.live_slots(), slots)
    np.testing.assert_allclose(
        np.sum(np.asarray(transitioned.state.history_content[slots]), axis=1),
        before,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        np.sum(np.asarray(transitioned.state.content)),
        np.sum(np.asarray(state.content)),
        atol=1e-10,
    )
    with pytest.raises(ValueError, match="changed topology epoch"):
        plan.step(transitioned.state, 0.01)
    continued = _step(next_plan, transitioned.state, jnp.asarray(0.01))
    assert bool(continued.evidence.successful)
    assert int(continued.state.history_count) == 3
    np.testing.assert_allclose(continued.evidence.conservation_residual, 0.0, atol=1e-10)
    mapping = MeshfreeCapacityMap(8, np.asarray([1, 4, 6], dtype=np.int32))
    positive = mapping.positive_space(np.asarray([1.0, 2.0, 3.0], dtype=np.float64))
    assert positive.size == 3
    np.testing.assert_allclose(positive.riesz(jnp.ones(3)), [1.0, 2.0, 3.0], atol=1e-12)
    np.testing.assert_allclose(
        mapping.expand(np.asarray([1.0, 2.0, 3.0], dtype=np.float64)),
        [0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 3.0, 0.0],
        atol=0,
    )
    np.testing.assert_allclose(mapping.compact(jnp.arange(8.0)), [1.0, 4.0, 6.0], atol=0)
    with pytest.raises(ValueError, match="strictly positive"):
        mapping.positive_space(np.asarray([1.0, 0.0, 3.0], dtype=np.float64))


def test_failed_historical_remap_does_not_partially_change_epoch_or_current_field() -> (
    None
):
    plan, state = _workflow()
    state = _step(plan, state, jnp.asarray(0.01)).state
    damaged = eqx.tree_at(
        lambda s: s.history_content, state, state.history_content.at[0, 0].set(jnp.nan)
    )
    held_plan, result = perform_epoch_transition(plan, damaged)
    assert not result.successful
    assert result.state.epoch.epoch_id == damaged.epoch.epoch_id
    assert held_plan is plan
    assert held_plan.epoch.epoch_id == result.state.epoch.epoch_id
    np.testing.assert_array_equal(result.state.points, damaged.points)
    np.testing.assert_array_equal(result.state.content, damaged.content)
    np.testing.assert_array_equal(result.state.history_content, damaged.history_content)
    continued = _step(held_plan, result.state, jnp.asarray(0.01))
    assert bool(continued.evidence.successful)
    assert continued.state.epoch.epoch_id == result.state.epoch.epoch_id
    np.testing.assert_allclose(continued.state.time, damaged.time + 0.01, atol=1e-14)


def test_epoch_values_differentiate_through_the_frozen_transfers() -> None:
    plan, state = _workflow()
    state = _step(plan, state, jnp.asarray(0.01)).state
    prepared = prepare_epoch_transition(state)
    assert prepared is not None
    _target_plan, routes = prepared

    def remapped(content: Array, history: Array) -> tuple[Array, Array]:
        moved = eqx.tree_at(
            lambda s: (s.content, s.history_content), state, (content, history)
        )
        result = plan.transition_epoch(moved, *routes)
        return result.state.content, result.state.history_content

    rng = np.random.default_rng(7)
    tangent = (
        jnp.asarray(rng.normal(size=state.content.shape)),
        jnp.asarray(rng.normal(size=state.history_content.shape)),
    )
    primal, derivative = jax.jvp(
        remapped, (state.content, state.history_content), tangent
    )
    # The frozen routes are linear in the values: a finite difference of any
    # size equals the JVP up to roundoff.
    shifted = remapped(state.content + tangent[0], state.history_content + tangent[1])
    for moved, base, directional in zip(shifted, primal, derivative, strict=True):
        np.testing.assert_allclose(moved - base, directional, atol=1e-12)
    cotangent = (
        jnp.asarray(rng.normal(size=primal[0].shape)),
        jnp.asarray(rng.normal(size=primal[1].shape)),
    )
    _, pullback = jax.vjp(remapped, state.content, state.history_content)
    pulled = pullback(cotangent)
    forward = sum(
        float(jnp.vdot(left, right))
        for left, right in zip(cotangent, derivative, strict=True)
    )
    backward = sum(
        float(jnp.vdot(left, right)) for left, right in zip(pulled, tangent, strict=True)
    )
    np.testing.assert_allclose(forward, backward, rtol=1e-12)
