"""White-box periodic size actions, sparse assembly, and proposal refusal contracts."""

from collections.abc import Callable
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax as phx
from phydrax._meshcore import meshcore_available, NativeExecutionScope
from phydrax.linalg import (
    AbstractLinearOperator,
    BlockSpace,
    SparseFactorizationPreconditionerBuilder,
)
from phydrax.linalg._sparse_factorizations import (
    _eliminate_nonempty_pivot,
    _eliminate_pivot,
    prepare_sparse_factor_congruence,
)
from phydrax.linalg._sparse_triangular import _prepare_sparse_triangular_substitution
from phydrax.meshing.providers import _native_periodic_size as size
from phydrax.meshing.providers._native_route_checks import check_deadline
from phydrax.optim import (
    constraint_layout,
    ConstraintLayout,
    MinimizationProblem,
    PrimalDualKKTSetupResult,
    PrimalDualNewtonKrylov,
)
from phydrax.optim._nonlinear_constraints import _canonical_constraints


pytestmark = pytest.mark.skipif(
    not meshcore_available(), reason="native meshcore library is unavailable"
)

# Empirical float64 bounds from the pre-remediation three-root periodic KKT
# fixture in test_native_periodic.py. These dense-AD and factor-action bounds
# are retained unchanged; the fixture values and operations are unchanged.
_LOCAL_AD_ROUND_OFF = 1.0e-14
_LU_CONGRUENCE_ROUND_OFF = 1.0e-12


@pytest.mark.parametrize("target", (0.25, 0.5))
@pytest.mark.parametrize("margin_fraction,admitted", ((0.75, True), (0.25, False)))
def test_periodic_area_polish_matches_publication_margin(
    target: float,
    margin_fraction: float,
    admitted: bool,
) -> None:
    points = jnp.asarray(
        ((0.0, 0.0), (target, 0.0), (0.0, target * margin_fraction * size._AREA_MARGIN)),
        dtype=np.float64,
    )
    args = size._SizeArgs(
        points,
        jnp.empty(0, dtype=np.int32),
        jnp.empty((0, 2), dtype=np.int32),
        jnp.empty((0, 2), dtype=np.float64),
        jnp.asarray(((0, 1, 2),), dtype=np.int32),
        jnp.zeros((1, 3, 2), dtype=np.float64),
        jnp.asarray(target, dtype=np.float64),
        jnp.empty(0, dtype=np.bool_),
        jnp.asarray(0.0, dtype=np.float64),
    )
    parameters = jnp.empty((0, 2), dtype=np.float64)
    normalized_cross = np.asarray(size._cell_areas(parameters, args))
    assert bool(np.all(normalized_cross > 0.5 * size._AREA_MARGIN)) is admitted
    polish_args = size._PolishArgs(
        args,
        jnp.empty(0, dtype=np.bool_),
        jnp.empty(0, dtype=np.float64),
        jnp.empty(0, dtype=np.int32),
    )
    problem = size._polish_problem(
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.bool_),
        0.0,
    )
    area_constraint = next(
        constraint
        for constraint in problem.constraints
        if constraint.constraint_id == "periodic-positive-cell-areas"
    )
    cross = area_constraint.value(parameters, polish_args)
    lower, upper = area_constraint.bounds(cross)
    assert bool(np.all(np.asarray(cross) >= np.asarray(lower))) is admitted
    assert np.all(np.asarray(cross) <= np.asarray(upper))


def test_placement_bank_keeps_scientific_rows_and_exact_predecessor_ties() -> None:
    # Root 3 acquires a third predecessor when root 2 is placed. Root 4 must
    # then precede it (two predecessors win over three), and root 3 retains
    # the two closest scientific rows with the edge-index tie break.
    edges = np.asarray(
        (
            (0, 2, 0, 0),
            (1, 2, 0, 0),
            (0, 3, 0, 0),
            (1, 3, 0, 0),
            (2, 3, 0, 0),
            (0, 4, 0, 0),
            (2, 4, 0, 0),
            (4, 3, 1, 0),
        ),
        dtype=np.int64,
    )
    closeness = np.asarray((0.1, 0.2, 0.3, 0.2, 0.05, 0.4, 0.3, 0.05))
    order, kept = size._placement_order(
        np.asarray((4, 3, 2)),
        edges,
        np.ones(8, dtype=np.bool_),
        closeness,
        np.asarray((True, True, False, False, False)),
    )
    assert order == [2, 4, 3]
    np.testing.assert_array_equal(np.flatnonzero(kept), (0, 1, 4, 5, 6, 7))


@dataclass(frozen=True)
class _PeriodicKKTFixture:
    edge_keys: np.ndarray
    free: np.ndarray
    kept: np.ndarray
    mutable: np.ndarray
    args: size._PolishArgs
    method: PrimalDualNewtonKrylov
    problem: MinimizationProblem
    layout: ConstraintLayout
    primal: Array
    equality: Array
    inequality: Array
    space: BlockSpace
    setup: size._PeriodicSparseKKTSetup
    builder: SparseFactorizationPreconditionerBuilder
    constraints: Callable[[Array], tuple[Array, Array]]


@pytest.fixture(scope="module")
def periodic_kkt() -> _PeriodicKKTFixture:
    edge_keys = np.asarray(((0, 1, 0, 0), (1, 2, 0, 0), (0, 2, 0, 0)), dtype=np.int64)
    free = np.arange(3, dtype=np.int64)
    kept = np.asarray((True, True, False))
    mutable = np.ones(3, dtype=np.bool_)
    args = size._PolishArgs(
        size._SizeArgs(
            jnp.asarray(((0.0, 0.0), (0.7, 0.1), (0.2, 0.8)), dtype=np.float64),
            jnp.asarray(free, dtype=np.int32),
            jnp.asarray(edge_keys[:, :2], dtype=np.int32),
            jnp.zeros((3, 2), dtype=np.float64),
            jnp.asarray(((0, 1, 2),), dtype=np.int32),
            jnp.zeros((1, 3, 2), dtype=np.float64),
            jnp.asarray(1.0, dtype=np.float64),
            jnp.asarray(mutable),
            jnp.asarray(0.1, dtype=np.float64),
        ),
        jnp.asarray(kept),
        jnp.ones(3, dtype=np.float64),
        jnp.arange(3, dtype=np.int32),
    )
    method, _ = size._prepare_polish_method(
        "authored-periodic-triangle-fixture",
        free,
        edge_keys,
        kept,
        mutable,
        10000000,
        10000000,
        args,
    )
    parameters = jnp.zeros((3, 2), dtype=np.float64)
    problem = size._polish_problem(np.arange(3), kept, 0.1)
    layout = constraint_layout(problem, parameters, args)

    def constraints(point: jax.Array) -> tuple[jax.Array, jax.Array]:
        return _canonical_constraints(problem, layout, point.reshape((3, 2)), args)

    primal = parameters.reshape(-1)
    equality, inequality = constraints(primal)
    space = phx.linalg.BlockSpace(
        (
            phx.linalg.ArraySpace(primal.shape, dtype=np.float64),
            phx.linalg.ArraySpace(equality.shape, dtype=np.float64),
        )
    )
    setup = method.kkt_setup
    assert isinstance(setup, size._PeriodicSparseKKTSetup)
    policy = method.linear_policy.preconditioning
    assert policy is not None
    assert isinstance(policy.builder, phx.linalg.SparseFactorizationPreconditionerBuilder)
    return _PeriodicKKTFixture(
        edge_keys,
        free,
        kept,
        mutable,
        args,
        method,
        problem,
        layout,
        primal,
        equality,
        inequality,
        space,
        setup,
        policy.builder,
        constraints,
    )


@dataclass(frozen=True)
class _PeriodicKKTCase:
    fixture: _PeriodicKKTFixture
    point: Array
    weights: Array
    equality_multipliers: Array
    inequality_multipliers: Array
    current: AbstractLinearOperator
    hessian: AbstractLinearOperator
    tangent: Array
    lagrangian: Callable[[Array], Array]


@pytest.fixture(scope="module", params=(False, True), ids=("initial", "displaced"))
def kkt_case(
    request: pytest.FixtureRequest, periodic_kkt: _PeriodicKKTFixture
) -> _PeriodicKKTCase:
    fixture = periodic_kkt
    point = fixture.primal
    weights = jnp.ones_like(fixture.inequality)
    if request.param:
        point = point + jnp.asarray((0.01, -0.02, 0.03, 0.01, -0.01, 0.02))
        weights = 2.0 * weights
    equality_multipliers = jnp.asarray((0.3, -0.2), dtype=np.float64)
    inequality_multipliers = jnp.linspace(
        0.1, 0.7, fixture.inequality.size, dtype=np.float64
    )
    derivatives = fixture.setup.prepare_derivatives(
        fixture.problem,
        fixture.layout,
        fixture.args,
        point,
        equality_multipliers,
        inequality_multipliers,
    )
    current, hessian = derivatives
    tangent = jnp.asarray((0.2, -0.1, 0.4, 0.3, -0.2, 0.5), dtype=np.float64)

    def lagrangian(candidate: Array) -> Array:
        equal, inequal = fixture.constraints(candidate)
        return (
            size._polish_objective(candidate.reshape((3, 2)), fixture.args)
            + jnp.vdot(equality_multipliers, equal)
            + jnp.vdot(inequality_multipliers, inequal)
        )

    return _PeriodicKKTCase(
        fixture,
        point,
        weights,
        equality_multipliers,
        inequality_multipliers,
        current,
        hessian,
        tangent,
        lagrangian,
    )


def test_periodic_local_derivatives_match_dense_ad(kkt_case: _PeriodicKKTCase) -> None:
    case, fixture = kkt_case, kkt_case.fixture
    local = size._local_constraint_state(case.current)
    assert local.edge_gradients.shape == (3, 2)
    assert local.edge_hessians.shape == (3, 2, 2)
    assert local.cell_gradients.shape == (1, 3, 2)
    expected_push = jax.jvp(fixture.constraints, (case.point,), (case.tangent,))[1]
    actual_push = case.current.mv(case.tangent)
    for actual_component, expected_component in zip(
        actual_push, expected_push, strict=True
    ):
        np.testing.assert_allclose(
            actual_component, expected_component, atol=_LOCAL_AD_ROUND_OFF
        )
    expected_pull = jax.vjp(fixture.constraints, case.point)[1](
        (case.equality_multipliers, case.inequality_multipliers),
    )[0]
    np.testing.assert_allclose(
        case.current.adjoint_mv((case.equality_multipliers, case.inequality_multipliers)),
        expected_pull,
        atol=_LOCAL_AD_ROUND_OFF,
    )
    expected_hessian = jax.jvp(jax.grad(case.lagrangian), (case.point,), (case.tangent,))[
        1
    ]
    np.testing.assert_allclose(
        case.hessian.mv(case.tangent), expected_hessian, atol=_LOCAL_AD_ROUND_OFF
    )


@dataclass(frozen=True)
class _PeriodicSaddleFixture:
    operator: AbstractLinearOperator
    metric: PrimalDualKKTSetupResult


@pytest.fixture(scope="module")
def periodic_saddle(kkt_case: _PeriodicKKTCase) -> _PeriodicSaddleFixture:
    case, fixture = kkt_case, kkt_case.fixture
    operator = fixture.setup.prepare_kkt_operator(
        case.current,
        case.hessian,
        case.weights,
        fixture.method.kkt_regularization,
        fixture.space,
    )
    metric = fixture.setup.prepare(
        case.current,
        case.weights,
        fixture.method.kkt_regularization,
        case.point,
        fixture.equality,
        fixture.space,
        operator,
    )
    return _PeriodicSaddleFixture(operator, metric)


def test_periodic_sparse_saddle_matches_dense_ad(
    kkt_case: _PeriodicKKTCase, periodic_saddle: _PeriodicSaddleFixture
) -> None:
    case, fixture = kkt_case, kkt_case.fixture
    expected_push = jax.jvp(fixture.constraints, (case.point,), (case.tangent,))[1]
    expected_hessian = jax.jvp(jax.grad(case.lagrangian), (case.point,), (case.tangent,))[
        1
    ]
    fused = periodic_saddle.operator
    multiplier = jnp.asarray((0.4, -0.3), dtype=np.float64)
    actual_primal, actual_equality = fused.mv((case.tangent, multiplier))
    expected_curvature = jax.vjp(fixture.constraints, case.point)[1](
        (multiplier, case.weights * expected_push[1])
    )[0]
    np.testing.assert_allclose(
        actual_primal,
        expected_hessian
        + expected_curvature
        + fixture.method.kkt_regularization * case.tangent,
        atol=_LOCAL_AD_ROUND_OFF,
    )
    np.testing.assert_allclose(
        actual_equality, expected_push[0], atol=_LOCAL_AD_ROUND_OFF
    )
    metric = periodic_saddle.metric
    # Dense AD is an independent oracle for this fixed three-root fixture.
    eq_matrix, ineq_matrix = (
        np.asarray(value) for value in jax.jacfwd(fixture.constraints)(case.point)
    )
    primal_matrix = (
        np.asarray(jax.hessian(case.lagrangian)(case.point))
        + ineq_matrix.T @ (np.asarray(case.weights)[:, None] * ineq_matrix)
        + fixture.method.kkt_regularization * np.eye(case.point.size)
    )
    expected_matrix = np.block(
        [
            [primal_matrix, eq_matrix.T],
            [eq_matrix, np.zeros((fixture.equality.size, fixture.equality.size))],
        ]
    )
    np.testing.assert_allclose(
        fixture.space.flatten(metric.operator.mv((case.tangent, multiplier))),
        expected_matrix @ np.r_[np.asarray(case.tangent), np.asarray(multiplier)],
        atol=_LOCAL_AD_ROUND_OFF,
    )
    assert int(metric.jvp_evaluations) == 0
    assert int(metric.status) == int(phx.linalg.LinearSolveStatus.SUCCESS)


def test_periodic_native_lu_congruence_matches_factor_action(
    kkt_case: _PeriodicKKTCase, periodic_saddle: _PeriodicSaddleFixture
) -> None:
    fixture = kkt_case.fixture
    metric = periodic_saddle.metric
    rhs = (jnp.ones(6, dtype=np.float64), jnp.asarray((1.0, -1.0), dtype=np.float64))
    action = fixture.builder.prepare(
        metric.operator, materialization=phx.linalg.MaterializationPolicy()
    )
    actual = action.apply(rhs)
    native = phx.linalg.sparse_preconditioner_factorization(action)
    assert native is not None and native.plan.shape == (8, 8)
    assert int(native.status) == int(phx.linalg.SparseFactorizationStatus.SUCCESS)
    lower = np.eye(native.plan.shape[0])
    positions = np.asarray(native.plan.lower_positions)
    row_indices = np.asarray(native.plan.factor_rows)[positions]
    column_indices = np.asarray(native.plan.factor_indices)[positions]
    coefficients = np.asarray(native.factor_values)[positions]
    strict = column_indices < row_indices
    lower[row_indices[strict], column_indices[strict]] = coefficients[strict]
    magnitudes = np.abs(
        np.asarray(native.factor_values)[np.asarray(native.plan.diagonal_positions)]
    )
    permutation = np.asarray(native.plan.permutation)
    expected_ordered = np.linalg.solve(
        lower.T,
        np.linalg.solve(lower, np.r_[np.asarray(rhs[0]), np.asarray(rhs[1])][permutation])
        / magnitudes,
    )
    expected_metric_action = np.zeros_like(expected_ordered)
    expected_metric_action[permutation] = expected_ordered
    np.testing.assert_allclose(
        fixture.space.flatten(actual),
        expected_metric_action,
        rtol=_LU_CONGRUENCE_ROUND_OFF,
    )
    assert action.properties.certifies("positive_definite")
    assert int(metric.jvp_evaluations) == 0
    assert int(metric.status) == int(phx.linalg.LinearSolveStatus.SUCCESS)


def test_periodic_polish_refuses_changed_source_binding(
    kkt_case: _PeriodicKKTCase,
) -> None:
    case, fixture = kkt_case, kkt_case.fixture
    changed_args = fixture.args._replace(
        size=fixture.args.size._replace(
            origins=fixture.args.size.origins.at[0, 0].set(0.001)
        )
    )
    with pytest.raises(eqx.EquinoxRuntimeError, match=r"periodic\.derivative_source"):
        fixture.setup.prepare_derivatives(
            fixture.problem,
            fixture.layout,
            changed_args,
            fixture.primal,
            case.equality_multipliers,
            case.inequality_multipliers,
        )


@pytest.mark.parametrize(
    "parameters",
    (np.zeros((3, 2)), np.asarray(((0.01, -0.02), (0.03, 0.01), (-0.01, 0.02)))),
    ids=("initial", "displaced"),
)
def test_periodic_host_metrics_match_device_kernel(
    periodic_kkt: _PeriodicKKTFixture, parameters: np.ndarray
) -> None:
    fixture = periodic_kkt
    domain = size._RelocationDomain(
        np.asarray(fixture.args.size.origins),
        fixture.edge_keys,
        np.asarray(((0, 1, 2),), dtype=np.int64),
        np.zeros((1, 3, 2), dtype=np.int64),
        np.eye(2, dtype=np.float64),
        np.zeros(3, dtype=np.bool_),
        1.0,
        0.1,
        np.inf,
    )
    prepared = size._prepare_relocation_domain(domain, ("p50", "p95"))
    metrics = size._host_relocation_metrics(
        prepared,
        parameters,
        edge_offsets=np.asarray(fixture.args.size.edge_offsets),
        cell_offsets=np.asarray(fixture.args.size.cell_offsets),
    )
    np.testing.assert_allclose(
        metrics.normalized,
        size._mutable_lengths(jnp.asarray(parameters), fixture.args.size),
        atol=_LOCAL_AD_ROUND_OFF,
    )
    np.testing.assert_allclose(
        metrics.areas,
        size._cell_areas(jnp.asarray(parameters), fixture.args.size),
        atol=_LOCAL_AD_ROUND_OFF,
    )


def test_periodic_kkt_incidence_refuses_before_storage_growth(
    periodic_kkt: _PeriodicKKTFixture,
) -> None:
    fixture = periodic_kkt
    layout_source = size._polish_derivative_source(fixture.args, fixture.layout)
    with pytest.raises(
        phx.meshing.MeshingFailure, match=r"periodic\.kkt\.incidence"
    ) as refusal:
        size._prepare_kkt_assembly_data(
            layout_source,
            np.asarray(fixture.setup.edge_positions),
            np.asarray(fixture.setup.cell_positions),
            "bounded-incidence-fixture",
            1,
        )
    assert refusal.value.category is phx.meshing.MeshingFailureCategory.RESOURCE_EXHAUSTED
    assert dict(refusal.value.evidence.requested)["kkt_incidence_bytes"] > 1
    assert dict(refusal.value.evidence.achieved)["remaining_bytes"] == 1


@pytest.mark.parametrize(
    "maximum_work,maximum_scratch",
    ((0, 10000000), (10000000, 1)),
    ids=("work", "scratch"),
)
def test_periodic_relocation_refuses_without_mutating_points(
    periodic_kkt: _PeriodicKKTFixture, maximum_work: int, maximum_scratch: int
) -> None:
    fixture = periodic_kkt
    points = np.asarray(fixture.args.size.origins).copy()
    before = points.view(np.uint64).copy()
    limits = phx.meshing.MeshingLimits(
        maximum_work_units=10000000, maximum_scratch_bytes=10000000
    )
    with NativeExecutionScope(limits.native_limits()) as scope:

        def check() -> None:
            check_deadline(
                scope.operation_started,
                limits,
                phx.meshing.MeshingStageKind.SURFACE_MESHING,
            )

        with pytest.raises(
            phx.meshing.MeshingFailure, match=r"periodic\.proposal\.(work|scratch)"
        ) as refusal:
            size.propose_periodic_site_relocation(
                points,
                fixture.edge_keys,
                np.asarray(((0, 1, 2),), dtype=np.int64),
                np.zeros((1, 3, 2), dtype=np.int64),
                np.eye(2, dtype=np.float64),
                np.zeros(3, dtype=np.bool_),
                1.0,
                ("p50", "p95"),
                maximum_work,
                maximum_scratch,
                check,
                source_binding_id="bounded-relocation-fixture",
                scope=scope,
            )
        assert (
            refusal.value.category
            is phx.meshing.MeshingFailureCategory.RESOURCE_EXHAUSTED
        )
    np.testing.assert_array_equal(points.view(np.uint64), before)


def test_fixed_rank_feasibility_keeps_published_interpolation_boundary() -> None:
    target = 0.125
    lengths = np.asarray(
        (np.nextafter(target, -np.inf),) * 4 + (target,) * 4, dtype=np.float64
    )
    fixed = np.asarray((True,) * 4 + (False,) * 4, dtype=np.bool_)
    assert np.quantile(lengths, 0.5) == np.quantile(lengths, 0.95) == target
    assert size._fixed_rank_feasible(lengths, fixed, target, (0.5, 0.95))
    impossible = np.asarray((1.0,) * 4 + (1.01,) * 4, dtype=np.float64)
    assert not size._fixed_rank_feasible(impossible, fixed, 1.01, (0.5, 0.95))


@pytest.mark.parametrize("count", (1, 7, 105, 444))
def test_local_rank_window_matches_full_unmodified_quantiles_bitwise(
    count: int,
) -> None:
    generator = np.random.default_rng(4973 + count)
    target = 0.125
    bank = np.asarray(
        (
            target / 2.0,
            np.nextafter(target, -np.inf),
            target,
            np.nextafter(target, np.inf),
            1.5 * target,
        ),
        dtype=np.float64,
    )
    lengths = generator.choice(bank, size=count)
    changed = np.sort(generator.choice(count, size=min(6, count), replace=False))
    candidates = generator.choice(bank, size=(changed.size, 17))
    mask = np.ones(count, dtype=np.bool_)
    mask[changed] = False
    unchanged = np.sort(lengths[mask], kind="stable")
    charges: list[int] = []
    actual = size._rank_window_quantiles(
        count, unchanged, candidates, (0.5, 0.95), charges.append
    )
    full = np.broadcast_to(lengths[:, None], (count, candidates.shape[1])).copy()
    full[changed] = candidates
    expected = np.quantile(full, (0.5, 0.95), axis=0)
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
    assert sum(charges) > 0


@pytest.mark.parametrize(
    "center,is_neighborhood",
    ((0.8, False), (np.sqrt(0.5) * (1.0 - 2.0e-14), True)),
)
def test_rank_continuation_keeps_source_boundary_and_exact_statistics(
    center: float,
    is_neighborhood: bool,
) -> None:
    points = np.asarray(
        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (center, center)),
        dtype=np.float64,
    )
    before = points.view(np.uint64).copy()
    cells = np.asarray(((0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)), dtype=np.int64)
    edges = np.asarray(
        (
            (0, 1, 0, 0),
            (0, 3, 0, 0),
            (0, 4, 0, 0),
            (1, 2, 0, 0),
            (1, 4, 0, 0),
            (2, 3, 0, 0),
            (2, 4, 0, 0),
            (3, 4, 0, 0),
        ),
        dtype=np.int64,
    )
    shifts = np.zeros((4, 3, 2), dtype=np.int64)
    vectors = 2.0 * np.eye(2, dtype=np.float64)
    protected = np.asarray((True, True, True, True, False), dtype=np.bool_)
    limits = phx.meshing.MeshingLimits(
        maximum_work_units=10000000, maximum_scratch_bytes=10000000
    )
    proposals = []
    for _ in range(2):
        with NativeExecutionScope(limits.native_limits()) as scope:

            def check() -> None:
                check_deadline(
                    scope.operation_started,
                    limits,
                    phx.meshing.MeshingStageKind.SURFACE_MESHING,
                )

            proposals.append(
                size.propose_periodic_rank_placement(
                    points,
                    edges,
                    cells,
                    shifts,
                    vectors,
                    protected,
                    1.0,
                    ("p50", "p95"),
                    limits.maximum_work_units,
                    limits.maximum_scratch_bytes,
                    check,
                    scope=scope,
                    maximum_size=1.5,
                )
            )
    first, second = proposals
    if not is_neighborhood:
        assert first.points is None and second.points is None
        assert first.placement_nodes == second.placement_nodes == 0
        np.testing.assert_array_equal(points.view(np.uint64), before)
        return
    assert first.points is not None and second.points is not None
    np.testing.assert_array_equal(
        first.points.view(np.uint64), second.points.view(np.uint64)
    )
    assert first.record == second.record
    assert first.polish is None and first.iterations == first.value_evaluations == 0
    assert first.placement_nodes > 0 and 0 < first.work_units < limits.maximum_work_units
    np.testing.assert_array_equal(points.view(np.uint64), before)
    np.testing.assert_array_equal(first.points[protected], points[protected])
    lifts = size.periodic_reference_lifts(cells, shifts, points.shape[0])
    lengths = size.quotient_edge_lengths(first.points, edges, lifts, vectors)
    assert np.quantile(lengths, 0.5) == np.quantile(lengths, 0.95) == 1.0
    corners = first.points[cells]
    areas = np.linalg.det(corners[:, 1:] - corners[:, :1]) / 2.0
    assert np.all(areas > 0.0)
    np.testing.assert_allclose(np.sum(areas), 1.0, rtol=0.0, atol=1.0e-14)


def test_circle_rank_placement_proves_fixed_edge_cardinality_obstruction() -> None:
    points = np.asarray(
        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.8, 0.8)),
        dtype=np.float64,
    )
    cells = np.asarray(((0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)), dtype=np.int64)
    edges = np.asarray(
        (
            (0, 1, 0, 0),
            (0, 3, 0, 0),
            (0, 4, 0, 0),
            (1, 2, 0, 0),
            (1, 4, 0, 0),
            (2, 3, 0, 0),
            (2, 4, 0, 0),
            (3, 4, 0, 0),
        ),
        dtype=np.int64,
    )
    limits = phx.meshing.MeshingLimits()
    with NativeExecutionScope(limits.native_limits()) as scope:

        def check() -> None:
            check_deadline(
                scope.operation_started,
                limits,
                phx.meshing.MeshingStageKind.SURFACE_MESHING,
            )

        proposal = size.propose_periodic_rank_placement(
            points,
            edges,
            cells,
            np.zeros((4, 3, 2), dtype=np.int64),
            2.0 * np.eye(2, dtype=np.float64),
            np.asarray((True, True, True, True, False), dtype=np.bool_),
            1.01,
            ("p50", "p95"),
            limits.maximum_work_units,
            limits.maximum_scratch_bytes,
            check,
            scope=scope,
            maximum_size=1.5,
        )
    # Four immutable length-one boundary edges already lie below target, but
    # the eight-edge median permits at most three strictly lower edges.
    assert proposal.points is None
    assert proposal.placement_nodes == 0
    assert proposal.work_units > 0


def test_periodic_relocation_preserves_immutable_early_refusal(
    periodic_kkt: _PeriodicKKTFixture,
) -> None:
    fixture = periodic_kkt
    points = np.asarray(fixture.args.size.origins).copy()
    before = points.view(np.uint64).copy()
    limits = phx.meshing.MeshingLimits(
        maximum_work_units=10000000, maximum_scratch_bytes=10000000
    )
    with NativeExecutionScope(limits.native_limits()) as scope:

        def check() -> None:
            check_deadline(
                scope.operation_started,
                limits,
                phx.meshing.MeshingStageKind.SURFACE_MESHING,
            )

        proposal = size.propose_periodic_site_relocation(
            points,
            fixture.edge_keys,
            np.asarray(((0, 1, 2),), dtype=np.int64),
            np.zeros((1, 3, 2), dtype=np.int64),
            np.eye(2, dtype=np.float64),
            np.ones(3, dtype=np.bool_),
            1.0,
            ("p50", "p95"),
            10000000,
            10000000,
            check,
            source_binding_id="immutable-relocation-fixture",
            scope=scope,
        )
    assert proposal.points is None
    assert proposal.work_units == 0
    assert proposal.status == -1 and proposal.polish_status == -1
    assert proposal.kept_exact_edges == 0 and proposal.placement_nodes == 0
    np.testing.assert_array_equal(points.view(np.uint64), before)


@pytest.mark.parametrize("roots", (16, 64, 256), ids=("small", "medium", "large"))
def test_placement_priority_retains_exhaustive_predecessor_order(roots: int) -> None:
    """A full adjacency reference checks heap invalidation and stable ties."""
    generator = np.random.default_rng(1907 + roots)
    endpoints = generator.integers(0, roots, size=(4 * roots, 2), dtype=np.int64)
    endpoints = endpoints[endpoints[:, 0] != endpoints[:, 1]]
    edges = np.concatenate((endpoints, np.zeros_like(endpoints, dtype=np.int64)), axis=1)
    protected = np.zeros(roots, dtype=np.bool_)
    protected[:3] = True
    free = np.flatnonzero(~protected)[::-1]
    active = generator.random(edges.shape[0]) > 0.2
    # Tied edge priorities must retain the ascending edge-index tie break.
    closeness = generator.integers(0, 4, size=edges.shape[0]).astype(np.float64)
    placed = protected.copy()
    reference_order: list[int] = []
    reference_kept = np.zeros(edges.shape[0], dtype=np.bool_)
    while len(reference_order) != free.size:
        candidates: list[tuple[int, int, int, list[int]]] = []
        for root in sorted(int(value) for value in free if not placed[value]):
            predecessors = [
                edge
                for edge, (first, second) in enumerate(endpoints)
                if active[edge]
                and (
                    (first == root and placed[second])
                    or (second == root and placed[first])
                )
            ]
            candidates.append(
                (-min(len(predecessors), 2), len(predecessors), root, predecessors)
            )
        _, _, root, predecessors = min(candidates)
        selected = sorted(predecessors, key=lambda edge: (closeness[edge], edge))[:2]
        reference_kept[selected] = True
        placed[root] = True
        reference_order.append(root)
    order, kept = size._placement_order(free, edges, active, closeness, protected)
    assert order == reference_order
    np.testing.assert_array_equal(kept, reference_kept)


def test_rank_bank_admission_does_not_charge_unexecuted_enlargement() -> None:
    protected = np.asarray((True, True, False, False, False, False), dtype=np.bool_)
    endpoints = np.asarray(
        ((0, 2), (1, 2), (2, 3), (0, 3), (3, 4), (1, 4), (4, 5), (0, 5)),
        dtype=np.int64,
    )
    edges = np.concatenate((endpoints, np.zeros_like(endpoints)), axis=1)
    admitted: list[int] = []

    def admit_bank(active_edges: int) -> None:
        admitted.append(size._placement_order_work_bound(6, 8, 4, active_edges))

    order, active, kept = size._rank_placement_edges(
        np.flatnonzero(~protected),
        edges,
        np.ones(8, dtype=np.float64),
        np.ones(8, dtype=np.bool_),
        protected,
        8,
        admit_bank=admit_bank,
    )
    assert order == [2, 3, 4, 5]
    assert np.all(active) and np.all(kept)
    # The former all-enlargement bound charged 5,448 units even when the first
    # bank succeeded. The same graph now admits 1,272 bank + 192 rank units.
    assert admitted == [1272]
    assert sum(admitted) + 192 == 1464


@pytest.mark.parametrize("empty", (False, True), ids=("nonempty", "empty"))
def test_periodic_pivot_window_skip_preserves_exact_factor_updates(
    periodic_kkt: _PeriodicKKTFixture, empty: bool
) -> None:
    plan = periodic_kkt.builder.prepared_plan
    assert plan is not None
    widths = np.diff(np.asarray(plan.column_offsets))
    pivots = np.flatnonzero((widths == 0) == empty)
    assert pivots.size > 0
    values = jnp.asarray(
        np.random.default_rng(2319).normal(size=plan.factor_nnz), dtype=np.float64
    )
    marker = jnp.full((plan.shape[0],), -1, dtype=plan.factor_indices.dtype)
    for pivot in pivots:
        index = jnp.asarray(pivot, dtype=plan.factor_indices.dtype)
        denominator = jnp.asarray(0.7, dtype=np.float64)
        actual, actual_marker = _eliminate_pivot(values, marker, plan, index, denominator)
        reference, reference_marker = _eliminate_nonempty_pivot(
            values, marker, plan, index, denominator
        )
        np.testing.assert_array_equal(
            np.asarray(actual).view(np.uint64), np.asarray(reference).view(np.uint64)
        )
        np.testing.assert_array_equal(actual_marker, reference_marker)
        if empty:
            np.testing.assert_array_equal(
                np.asarray(actual).view(np.uint64), np.asarray(values).view(np.uint64)
            )


def test_periodic_polish_bound_charges_only_nonempty_factor_windows(
    periodic_kkt: _PeriodicKKTFixture,
) -> None:
    fixture = periodic_kkt
    setup, plan = size._polish_sparse_owners(fixture.method)
    phases = dict(size._polish_work_bounds(6, 3, 1, 6, 3, 2, fixture.method))
    refreshes = size._POLISH_TERMINATION.maximum_steps + 1
    active = np.count_nonzero(np.diff(np.asarray(plan.column_offsets)))
    expected = 16 * (
        active * (plan.row_width * max(1, plan.column_width) + plan.column_width)
        + plan.shape[0] * (plan.row_width + 1)
    )
    previous = (
        16
        * plan.shape[0]
        * (
            plan.row_width * max(1, plan.column_width)
            + plan.row_width
            + plan.column_width
            + 1
        )
    )
    assert active < plan.shape[0]
    assert phases["native_factor"] == refreshes * expected
    assert phases["native_factor"] < refreshes * previous
    # RHS-dependent substitutions and current numeric-cache preparation have
    # not been waived by the symbolic empty-column evidence.
    assert phases["native_substitution_prepare"] == refreshes * (
        plan.numeric_substitution_preparation_work_units_upper
        + plan.lu_congruence_preparation_work_units_upper
    )
    assert phases["native_congruence_correction"] == (
        size._POLISH_TERMINATION.maximum_steps
        * (size._POLISH_LINEAR_ITERATIONS_PER_TRIAL + 1)
        * plan.lu_congruence_solve_work_units_upper_for(
            setup.kkt_assembly.template.coefficients.dtype,
            phx.linalg.coordinate_dtype(setup.primal_space),
        )
    )


def test_periodic_congruence_reuses_exact_lower_certification(
    periodic_saddle: _PeriodicSaddleFixture,
    kkt_case: _PeriodicKKTCase,
) -> None:
    fixture = kkt_case.fixture
    action = fixture.builder.prepare(
        periodic_saddle.metric.operator,
        materialization=phx.linalg.MaterializationPolicy(),
    )
    factor = phx.linalg.sparse_preconditioner_factorization(action)
    assert factor is not None
    plan = factor.plan
    congruence = prepare_sparse_factor_congruence(factor)
    lower_values = factor.factor_values[plan.lower_positions]
    lower_values = jnp.where(
        plan.lower_analysis.row_indices == plan.lower_analysis.indices,
        jnp.ones((), dtype=lower_values.dtype),
        lower_values,
    )
    reference = _prepare_sparse_triangular_substitution(
        plan.lower_analysis,
        lower_values,
        pivot_tolerance=plan.policy.pivot_tolerance,
        adjoint=True,
    )
    actual = jax.tree.map(lambda leaf: leaf[0], congruence.lower_adjoint_substitution)
    for result, expected in zip(
        jax.tree.leaves(actual), jax.tree.leaves(reference), strict=True
    ):
        np.testing.assert_array_equal(result, expected)
    previous = (
        plan.lower_analysis.numeric_preparation_work_units_upper
        + 4 * plan.lower_positions.size
        + 24 * plan.shape[0]
        + 64
    )
    assert plan.lu_congruence_preparation_work_units_upper < previous


def test_rank_continuation_repairs_excess_tail_across_multiple_free_roots() -> None:
    components = 4
    height = np.sqrt(np.float64(0.75)) + 8.0 * np.spacing(np.float64(1.0))
    points = np.asarray(
        [
            point
            for component in range(components)
            for point in (
                (3.0 * component, 0.0),
                (3.0 * component + 1.0, 0.0),
                (3.0 * component + 0.5, height),
            )
        ],
        dtype=np.float64,
    )
    cells = np.arange(3 * components, dtype=np.int64).reshape((-1, 3))
    edges = np.asarray(
        [
            (3 * component + first, 3 * component + second, 0, 0)
            for component in range(components)
            for first, second in ((0, 1), (0, 2), (1, 2))
        ],
        dtype=np.int64,
    )
    shifts = np.zeros((components, 3, 2), dtype=np.int64)
    vectors = 32.0 * np.eye(2, dtype=np.float64)
    protected = np.tile(np.asarray((True, True, False), dtype=np.bool_), components)
    lifts = size.periodic_reference_lifts(cells, shifts, points.shape[0])
    initial = size.quotient_edge_lengths(points, edges, lifts, vectors)
    above = np.count_nonzero(initial > 1.0)
    incidence = np.bincount(edges[:, :2].reshape(-1), minlength=points.shape[0])
    # No single root can satisfy the final tail bound; several strict
    # improvements must be admitted before the global quantile targets are possible.
    assert above > np.max(incidence[~protected])
    assert np.quantile(initial, 0.5) > 1.0
    assert np.quantile(initial, 0.95) > 1.0
    limits = phx.meshing.MeshingLimits(
        maximum_work_units=10000000, maximum_scratch_bytes=10000000
    )
    before = points.view(np.uint64).copy()
    with NativeExecutionScope(limits.native_limits()) as scope:

        def check() -> None:
            check_deadline(
                scope.operation_started,
                limits,
                phx.meshing.MeshingStageKind.SURFACE_MESHING,
            )

        proposal = size.propose_periodic_rank_placement(
            points,
            edges,
            cells,
            shifts,
            vectors,
            protected,
            1.0,
            ("p50", "p95"),
            limits.maximum_work_units,
            limits.maximum_scratch_bytes,
            check,
            scope=scope,
            maximum_size=1.5,
        )
    assert proposal.points is not None
    assert proposal.placement_nodes >= components
    assert proposal.polish is None
    np.testing.assert_array_equal(points.view(np.uint64), before)
    np.testing.assert_array_equal(proposal.points[protected], points[protected])
    final = size.quotient_edge_lengths(proposal.points, edges, lifts, vectors)
    assert np.quantile(final, 0.5) == np.quantile(final, 0.95) == 1.0
    corners = proposal.points[cells]
    assert np.all(np.linalg.det(corners[:, 1:] - corners[:, :1]) > 0.0)
