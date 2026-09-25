import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._physical import RelativityScaleContract
from phydrax._strict import StrictModule
from phydrax.applications.astrophysics._gr_bundles import (
    GRCallableConstantOfMotion,
    GRCoordinateMomentumConstant,
)
from phydrax.applications.astrophysics._gr_events import (
    GRRayEventCode,
    GRRayEventSurfaces,
)
from phydrax.applications.astrophysics._gr_rays import (
    gr_chart_identity,
    gr_metric_identity,
    GRRayPlan,
    GRRayState,
    trace_gr_rays,
)
from phydrax.applications.astrophysics._gr_screens import GRObserverScreenPlan
from phydrax.applications.astrophysics._gr_status import GRRayStatus
from phydrax.metrix._black_hole_metrics import kerr_boyer_lindquist_metric
from phydrax.metrix._chart import CoordinateChart
from phydrax.metrix._lorentzian import minkowski_metric, schwarzschild_metric
from phydrax.metrix._spacetime_conventions import RelativityConvention
from phydrax.units import KILOGRAM


jax.config.update("jax_enable_x64", True)
_SCALE = RelativityScaleContract.geometric(KILOGRAM)
_CONVENTION = RelativityConvention.canonical()
_COORDINATE_UNIT = _SCALE.dimensional_scale.length_unit
_AFFINE_PARAMETER_UNIT = _SCALE.dimensional_scale.length_unit


def _ray_context():
    return {
        "scale": _SCALE,
        "convention": _CONVENTION,
        "coordinate_unit": _COORDINATE_UNIT,
        "affine_parameter_unit": _AFFINE_PARAMETER_UNIT,
    }


class _StatefulMargin(StrictModule):
    threshold: jax.Array

    def __init__(self, threshold):
        self.threshold = jnp.asarray(threshold)

    def __call__(self, affine, point, tangent):
        del affine, tangent
        return self.threshold - point[1]


def _capture_below_negative_half(affine, point, tangent):
    return point[1] + 0.5


def _escape_above_half(affine, point, tangent):
    return 0.5 - point[1]


def _domain_below_half(affine, point, tangent):
    return 0.5 - point[3]


def _escape_above_053(affine, point, tangent):
    return 0.53 - point[1]


def _domain_affine_budget(affine, point, tangent):
    return 2.0 - affine


def _cartesian_metric():
    chart = CoordinateChart("cartesian-spacetime", ("t", "x", "y", "z"))
    return minkowski_metric(chart)


def _spherical_screen(metric, radius=10.0):
    return GRObserverScreenPlan(
        metric,
        jnp.asarray([0.0, radius, jnp.pi / 2.0, 0.0]),
        jnp.asarray([1.0, 0.0, 0.0, 0.0]),
        jnp.asarray([0.0, -1.0, 0.0, 0.0]),
        jnp.asarray([0.0, 0.0, -1.0, 0.0]),
        jnp.asarray([[0.0, 0.0], [0.05, -0.03]]),
        temporal_direction="past",
        **_ray_context(),
    ).initialize()


def test_observer_screen_initialization_is_metric_orthonormal_and_generic():
    flat = _cartesian_metric()
    flat_screen = GRObserverScreenPlan(
        flat,
        jnp.zeros(4),
        jnp.asarray([1.0, 0.0, 0.0, 0.0]),
        jnp.asarray([0.0, 0.0, 0.0, 1.0]),
        jnp.asarray([0.0, 0.0, 1.0, 0.0]),
        jnp.asarray([[0.0, 0.0], [0.1, -0.2]]),
        temporal_direction="past",
        **_ray_context(),
    ).initialize()
    np.testing.assert_allclose(flat_screen.tetrad_residual, 0.0, atol=1.0e-14)
    np.testing.assert_allclose(flat_screen.null_residual, 0.0, atol=1.0e-14)
    assert np.all(flat_screen.valid)
    assert np.all(np.asarray(flat_screen.ray_tangents[:, 0]) < 0.0)

    spherical_chart = CoordinateChart("static-spherical", ("t", "r", "theta", "phi"))
    schwarzschild = schwarzschild_metric(1.0, chart=spherical_chart)
    kerr_chart = CoordinateChart("kerr-bl", ("t", "r", "theta", "phi"))
    kerr = kerr_boyer_lindquist_metric(1.0, 0.4, chart=kerr_chart)
    changed_kerr = kerr_boyer_lindquist_metric(1.0, 0.6, chart=kerr_chart)
    assert gr_metric_identity(kerr) != gr_metric_identity(changed_kerr)
    for screen in (_spherical_screen(schwarzschild), _spherical_screen(kerr)):
        assert np.all(screen.valid)
        assert float(screen.tetrad_residual) < 1.0e-12
        assert float(np.max(screen.null_residual)) < 1.0e-12


def test_observer_screen_promotes_integer_inputs_to_floating_geometry():
    screen = GRObserverScreenPlan(
        _cartesian_metric(),
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [[0, 0], [1, 0]],
        **_ray_context(),
    ).initialize()

    assert jnp.issubdtype(screen.ray_coordinates.dtype, jnp.floating)
    assert np.all(screen.valid)


def test_mixed_batch_resolves_ordered_events_and_retains_work_exhaustion():
    metric = _cartesian_metric()
    coordinates = jnp.zeros((5, 4))
    tangents = jnp.asarray(
        [
            [1.0, -1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
        ]
    )
    plan = GRRayPlan(
        metric,
        GRRayState(
            coordinates,
            tangents,
            active=jnp.asarray([True, True, True, True, False]),
        ),
        jnp.linspace(0.0, 1.0, 9),
        capture_margin=_capture_below_negative_half,
        escape_margin=_escape_above_half,
        domain_margin=_domain_below_half,
        **_ray_context(),
    )
    result = trace_gr_rays(plan)

    np.testing.assert_array_equal(
        result.event_code,
        [
            int(GRRayEventCode.CAPTURE),
            int(GRRayEventCode.ESCAPE),
            int(GRRayEventCode.DOMAIN),
            int(GRRayEventCode.WORK),
            int(GRRayEventCode.NONE),
        ],
    )
    np.testing.assert_array_equal(
        result.status,
        [
            int(GRRayStatus.CAPTURED),
            int(GRRayStatus.ESCAPED),
            int(GRRayStatus.DOMAIN_EXIT),
            int(GRRayStatus.WORK_EXHAUSTED),
            int(GRRayStatus.INACTIVE),
        ],
    )
    assert result.coordinates.shape == (5, 9, 4)
    assert result.active.shape == (5, 9)
    assert result.valid.shape == (5, 9)
    np.testing.assert_array_equal(
        result.physically_valid, [True, True, False, True, False]
    )
    np.testing.assert_array_equal(result.qualified, [True, True, False, False, False])
    np.testing.assert_array_equal(
        result.event_ledger.recorded, [True, True, True, True, False]
    )


def test_event_ledger_retains_backend_root_state_between_history_nodes():
    metric = _cartesian_metric()
    plan = GRRayPlan(
        metric,
        GRRayState(
            jnp.zeros((1, 4)),
            jnp.asarray([[1.0, 1.0, 0.0, 0.0]]),
        ),
        jnp.asarray([0.0, 0.25, 0.5, 0.75, 1.0]),
        escape_margin=_escape_above_053,
        **_ray_context(),
    )
    result = trace_gr_rays(plan)

    np.testing.assert_allclose(result.event_ledger.affine_parameter, [0.53], atol=1.0e-8)
    np.testing.assert_allclose(
        result.event_ledger.coordinates[0],
        [0.53, 0.53, 0.0, 0.0],
        atol=1.0e-8,
    )
    np.testing.assert_allclose(
        result.event_ledger.tangent[0],
        [1.0, 1.0, 0.0, 0.0],
        atol=1.0e-10,
    )
    assert bool(result.event_ledger.state_valid[0])
    assert int(result.event_ledger.history_index[0]) == 2


def test_affine_dependent_domain_evidence_uses_each_history_time():
    metric = _cartesian_metric()
    affine = jnp.asarray([0.0, 0.25, 0.75, 1.0])
    plan = GRRayPlan(
        metric,
        GRRayState(
            jnp.zeros((1, 4)),
            jnp.asarray([[1.0, 1.0, 0.0, 0.0]]),
        ),
        affine,
        domain_margin=_domain_affine_budget,
        affine_budget_is_event=False,
        **_ray_context(),
    )
    result = trace_gr_rays(plan)

    np.testing.assert_allclose(result.domain_evidence.margin[0], 2.0 - affine)
    assert np.all(result.domain_evidence.qualified)


def test_ray_plan_rejects_each_nonfinite_tolerance():
    metric = _cartesian_metric()
    state = GRRayState(
        jnp.zeros((1, 4)),
        jnp.asarray([[1.0, 1.0, 0.0, 0.0]]),
    )
    for name in (
        "relative_tolerance",
        "absolute_tolerance",
        "constraint_tolerance",
        "event_tolerance",
    ):
        with pytest.raises(ValueError, match="finite and positive"):
            GRRayPlan(
                metric,
                state,
                jnp.asarray([0.0, 1.0]),
                **{name: float("inf")},
                **_ray_context(),
            )


def test_stateful_event_margin_identity_includes_numeric_callable_content():
    first = GRRayEventSurfaces(capture_margin=_StatefulMargin(1.0))
    second = GRRayEventSurfaces(capture_margin=_StatefulMargin(2.0))
    assert first.event_id != second.event_id


def test_opaque_event_margins_require_declared_identity():
    near = lambda affine, point, tangent: 1.0 - point[1]
    far = lambda affine, point, tangent: 2.0 - point[1]
    for margin in (near, far):
        with pytest.raises(TypeError, match="explicit semantic_id and numeric_id"):
            GRRayEventSurfaces(capture_margin=margin)
    with pytest.raises(TypeError, match="explicit semantic_id and numeric_id"):
        GRRayPlan(
            _cartesian_metric(),
            GRRayState(jnp.zeros((1, 4)), jnp.asarray([[1.0, 1.0, 0.0, 0.0]])),
            jnp.linspace(0.0, 1.0, 3),
            escape_margin=near,
            **_ray_context(),
        )

    declared_near = GRRayEventSurfaces(
        capture_margin=near,
        capture_margin_semantic_id="radial-capture",
        capture_margin_numeric_id="radius-1",
    )
    declared_far = GRRayEventSurfaces(
        capture_margin=far,
        capture_margin_semantic_id="radial-capture",
        capture_margin_numeric_id="radius-2",
    )
    assert declared_near.event_id != declared_far.event_id
    assert (
        GRRayEventSurfaces(escape_margin=_escape_above_half).event_id
        == GRRayEventSurfaces(escape_margin=_escape_above_half).event_id
    )
    assert (
        GRRayEventSurfaces(escape_margin=_escape_above_half).event_id
        != GRRayEventSurfaces(escape_margin=_escape_above_053).event_id
    )


def test_callable_constant_revisions_change_constant_and_plan_identity():
    def make(offset):
        return lambda metric, coordinates, tangent: tangent[0] + offset

    first = GRCallableConstantOfMotion(
        make(1.0),
        name="closure-constant",
        evaluator_semantic_id="test:closure-constant",
        evaluator_numeric_id="test:closure-constant:offset-1",
    )
    second = GRCallableConstantOfMotion(
        make(2.0),
        name="closure-constant",
        evaluator_semantic_id="test:closure-constant",
        evaluator_numeric_id="test:closure-constant:offset-2",
    )
    assert first.constant_id != second.constant_id

    metric = _cartesian_metric()
    state = GRRayState(
        jnp.zeros((1, 4)),
        jnp.asarray([[1.0, 1.0, 0.0, 0.0]]),
    )
    first_plan = GRRayPlan(
        metric,
        state,
        jnp.asarray([0.0, 1.0]),
        constants_of_motion=(first,),
        **_ray_context(),
    )
    second_plan = GRRayPlan(
        metric,
        state,
        jnp.asarray([0.0, 1.0]),
        constants_of_motion=(second,),
        **_ray_context(),
    )
    assert first_plan.plan_id != second_plan.plan_id


def test_parallel_transport_constants_and_jacobi_map_have_fixed_evidence():
    metric = _cartesian_metric()
    screen = GRObserverScreenPlan(
        metric,
        jnp.zeros(4),
        jnp.asarray([1.0, 0.0, 0.0, 0.0]),
        jnp.asarray([0.0, 0.0, 0.0, 1.0]),
        jnp.asarray([0.0, 0.0, 1.0, 0.0]),
        jnp.asarray([[0.0, 0.0], [0.1, 0.0]]),
        temporal_direction="future",
        **_ray_context(),
    ).initialize()
    affine = jnp.linspace(0.0, 1.0, 5)
    plan = GRRayPlan.from_screen(
        metric,
        screen,
        affine,
        transport_screen_basis=True,
        track_jacobi=True,
        constants_of_motion=(
            GRCoordinateMomentumConstant(0, name="energy", sign=-1.0),
            GRCoordinateMomentumConstant(3, name="axial-momentum"),
        ),
        affine_budget_is_event=False,
        **_ray_context(),
    )
    result = trace_gr_rays(plan)

    transported_basis = result.transported_screen_basis
    assert transported_basis is not None
    assert transported_basis.shape == (2, 5, 2, 4)
    jacobi = result.jacobi_evidence
    assert jacobi is not None
    assert jacobi.jacobi_map.shape == (2, 5, 2, 2)
    np.testing.assert_allclose(
        jacobi.determinant,
        np.broadcast_to(np.asarray(affine**2), (2, 5)),
        atol=2.0e-7,
    )
    assert not np.any(jacobi.caustic)
    assert float(np.max(result.null_residual)) < 1.0e-10
    assert float(np.max(result.transport_residual)) < 1.0e-10
    assert float(np.max(result.bundle_evidence.constant_relative_drift)) < 1.0e-10
    assert np.all(result.derivative_valid)


def test_timelike_trace_is_filter_jittable_and_has_branch_local_gradient():
    metric = _cartesian_metric()
    plan = GRRayPlan(
        metric,
        GRRayState(
            jnp.asarray([[0.0, 0.0, 0.0, 0.0]]),
            jnp.asarray([[1.0, 0.0, 0.0, 0.0]]),
        ),
        jnp.linspace(0.0, 1.0, 4),
        ray_kind="timelike",
        affine_budget_is_event=False,
        **_ray_context(),
    )
    result = eqx.filter_jit(trace_gr_rays)(plan)
    assert int(result.status[0]) == int(GRRayStatus.SUCCESS)
    assert bool(result.derivative_valid[0])
    assert result.metric_id == gr_metric_identity(metric)
    assert result.chart_id == gr_chart_identity(metric)
    np.testing.assert_allclose(result.coordinates[0, :, 0], plan.affine_parameter)
    np.testing.assert_allclose(result.null_residual, 0.0, atol=1.0e-12)

    def final_x(candidate):
        return trace_gr_rays(candidate).coordinates[0, -1, 1]

    gradient = eqx.filter_grad(final_x)(plan)
    np.testing.assert_allclose(gradient.initial_state.tangents[0, 1], 1.0, atol=2.0e-7)
