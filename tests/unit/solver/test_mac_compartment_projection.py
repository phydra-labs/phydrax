#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Mixed MAC compartment projection: constraint rows, gauge, work, adjoint, breathing."""

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


STEP = 1.0e-3
TOLERANCE = 1.0e-9
REFERENCE_PRESSURE = 1.0e5
LIQUID_DENSITY = 1000.0
GAS_DENSITY = 1.0
GRAVITY = 9.81


def _operators(nx: int, ny: int) -> Any:
    """x-periodic, y-walled unit square."""
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(nx, periodic=True),
            phx.discretization.UniformCellAxisSpec(ny),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray(((0.0, 0.0), (1.0, 1.0))))
    discretization = phx.discretization.FiniteVolumePlan(grid).prepare()
    return phx.discretization.MACOperatorPlan(discretization).prepare()


@eqx.filter_jit
def _project(plan: Any, momentum: Any, inverse: Any, step: float, constraint: Any) -> Any:
    return plan.project(momentum, inverse, step, constraint)


class _Case:
    """Two closed bubbles and a vented gas layer in liquid on a 12 x 12 grid."""

    def __init__(self, gas_density: float) -> None:
        self.operators = _operators(12, 12)
        volumes = self.operators.discretization.cell_volumes
        centers = (jnp.arange(12, dtype=jnp.float64) + 0.5) / 12.0
        x, y = jnp.meshgrid(centers, centers, indexing="ij")
        first = (x - 0.3) ** 2 + (y - 0.35) ** 2 < 0.15**2
        second = (x - 0.7) ** 2 + (y - 0.45) ** 2 < 0.12**2
        core = (x - 0.3) ** 2 + (y - 0.35) ** 2 < 0.1**2
        self.vented = y > 0.8
        self.labels = jnp.where(first, 0, jnp.where(second, 1, -1)).astype(jnp.int32)
        # Partially filled rim cells make the gas-volume weights non-uniform.
        fraction = jnp.where(first & ~core, 0.5, 1.0)
        self.gas = jnp.where(first | second | self.vented, fraction * volumes, 0.0)
        liquid_fraction = 1.0 - self.gas / volumes
        density = liquid_fraction * LIQUID_DENSITY + (1.0 - liquid_fraction) * gas_density
        face_density = self.operators.interpolate_inverse_momentum(density)
        self.inverse = tuple(1.0 / value for value in face_density)
        self.offset = LIQUID_DENSITY * GRAVITY * (1.0 - y)
        self.compartment_volume = jnp.stack(
            (
                jnp.sum(jnp.where(first, self.gas, 0.0)),
                jnp.sum(jnp.where(second, self.gas, 0.0)),
            )
        )
        velocity_x = 0.1 * jnp.sin(2.0 * jnp.pi * x) * jnp.cos(jnp.pi * y)
        velocity_y = jnp.zeros((12, 13))
        velocity_y = velocity_y.at[:, 1:-1].set(
            0.05 * jnp.cos(2.0 * jnp.pi * centers)[:, None]
            + 0.02 * jnp.sin(3.0 * jnp.pi * centers)[None, 1:]
        )
        self.momentum = (face_density[0] * velocity_x, face_density[1] * velocity_y)
        self.face_density = face_density

    def constraint(
        self,
        *,
        active: tuple[bool, bool],
        vented: bool,
        gas: Any = None,
        compliance: Any = None,
    ) -> Any:
        return phx.solver.MACCompartmentConstraint(
            labels=self.labels,
            gas_volume=self.gas if gas is None else gas,
            compliance=(
                self.compartment_volume / REFERENCE_PRESSURE
                if compliance is None
                else compliance
            ),
            pressure=jnp.asarray(
                (REFERENCE_PRESSURE + 200.0, REFERENCE_PRESSURE - 100.0)
            ),
            active=jnp.asarray(active),
            atmosphere=self.vented if vented else jnp.zeros((12, 12), dtype=bool),
            atmosphere_pressure=REFERENCE_PRESSURE,
            pressure_offset=self.offset,
        )

    def gas_mean(self, support: Any, field: Any) -> float:
        weight = jnp.where(support, self.gas, 0.0)
        return float(jnp.sum(weight * field) / jnp.sum(weight))


@pytest.fixture(scope="module")
def case() -> _Case:
    return _Case(GAS_DENSITY)


@pytest.fixture(scope="module")
def plan(case: _Case) -> Any:
    return phx.solver.MACCompartmentProjectionPlan(
        case.operators, compartment_capacity=2, atmosphere=True, tolerance=TOLERANCE
    )


def _volume_norm(operators: Any, value: Any) -> float:
    volumes = operators.discretization.cell_volumes
    return float(jnp.sqrt(jnp.sum(volumes * value**2)))


def _assert_continuity(result: Any) -> None:
    assert float(result.continuity_residual) <= TOLERANCE * float(
        result.continuity_scale
    ) * (1.0 + 1.0e-6)


def test_without_compartments_the_projection_is_the_variable_density_projection(
    case: _Case, plan: Any
) -> None:
    result = _project(
        plan,
        case.momentum,
        case.inverse,
        STEP,
        case.constraint(active=(False, False), vented=False),
    )
    reference = eqx.filter_jit(
        lambda owner, momentum, inverse: owner.project(momentum, inverse, STEP)
    )(plan.projection, case.momentum, case.inverse)

    assert bool(result.successful) and bool(reference.successful)
    assert int(result.gauge) == phx.solver.MACCompartmentGauge.MEAN_ZERO
    np.testing.assert_array_equal(result.volume_rate, 0.0)
    assert float(result.atmosphere_volume_rate) == 0.0
    _assert_continuity(result)
    # The batched lane follows its own floating-point path through PCG, so the
    # two certified projections agree to the solver tolerance.
    pressure_scale = float(jnp.max(jnp.abs(reference.pressure_increment)))
    np.testing.assert_allclose(
        result.pressure,
        reference.pressure_increment,
        rtol=0.0,
        atol=100.0 * TOLERANCE * pressure_scale,
    )
    velocity_scale = max(float(jnp.max(jnp.abs(value))) for value in reference.velocity)
    for projected, expected in zip(result.velocity, reference.velocity, strict=True):
        np.testing.assert_allclose(
            projected, expected, rtol=0.0, atol=100.0 * TOLERANCE * velocity_scale
        )


def test_native_multi_rhs_projection_matches_independent_lane_solutions(
    case: _Case, plan: Any
) -> None:
    centered_target = case.operators.gauge_project(1.0e-4 * case.offset)
    momenta = tuple(
        jnp.stack((value, jnp.zeros_like(value)), axis=0)
        for value in case.momentum
    )
    targets = jnp.stack((jnp.zeros_like(centered_target), centered_target), axis=0)
    guesses = jnp.zeros_like(targets)

    batched = eqx.filter_jit(
        lambda owner, values, inverse, target, guess: owner._project_many(
            values,
            inverse,
            STEP,
            pressure=guess,
            target_divergence=target,
        )
    )(plan.projection, momenta, case.inverse, targets, guesses)

    def solve_lane(
        lane_momentum: Any, target: Any, guess: Any
    ) -> phx.solver.MACVariableDensityProjectionResult:
        return plan.projection.project(
            lane_momentum,
            case.inverse,
            STEP,
            pressure=guess,
            target_divergence=target,
        )

    independent = eqx.filter_jit(jax.vmap(solve_lane))(momenta, targets, guesses)
    assert bool(jnp.all(batched.successful))
    assert bool(jnp.all(independent.successful))
    pressure_scale = float(jnp.max(jnp.abs(independent.pressure_increment)))
    np.testing.assert_allclose(
        batched.pressure_increment,
        independent.pressure_increment,
        rtol=0.0,
        atol=100.0 * TOLERANCE * pressure_scale,
    )
    for actual, expected in zip(
        batched.pressure_impulse, independent.pressure_impulse, strict=True
    ):
        impulse_scale = float(jnp.max(jnp.abs(expected)))
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=0.0,
            atol=100.0 * TOLERANCE * impulse_scale,
        )


def test_compartment_lanes_execute_one_batched_operator_action_stream(
    case: _Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    operator_type = type(case.operators)
    original = operator_type.positive_gauged_weighted_laplacian
    actions: list[None] = []

    def record() -> None:
        actions.append(None)

    def instrumented(
        owner: Any, pressure: Any, face_coefficient: Any
    ) -> jax.Array:
        jax.debug.callback(record, ordered=True)
        return original(owner, pressure, face_coefficient)

    monkeypatch.setattr(
        operator_type,
        "positive_gauged_weighted_laplacian",
        instrumented,
    )
    action_plan = phx.solver.MACCompartmentProjectionPlan(
        case.operators,
        compartment_capacity=2,
        atmosphere=True,
        tolerance=TOLERANCE,
    )
    result = _project(
        action_plan,
        case.momentum,
        case.inverse,
        STEP,
        case.constraint(active=(True, True), vented=True),
    )
    jax.block_until_ready(result)

    reported = np.asarray(result.pressure_solves.diagnostics.matvec_count)
    np.testing.assert_array_equal(reported, np.full(reported.shape, reported[0]))
    # Native evidence excludes the generic solve boundary's final certification
    # action. Every PCG action before that is shared by all pressure lanes.
    assert len(actions) == int(reported[0]) + 1
    assert len(actions) < int(np.sum(reported))


def test_vented_atmosphere_fixes_the_absolute_pressure_level(
    case: _Case, plan: Any
) -> None:
    result = _project(
        plan,
        case.momentum,
        case.inverse,
        STEP,
        case.constraint(active=(False, False), vented=True),
    )

    assert bool(result.successful)
    assert int(result.gauge) == phx.solver.MACCompartmentGauge.ATMOSPHERE_REFERENCE
    vented_mean = case.gas_mean(case.vented, result.absolute_pressure)
    np.testing.assert_allclose(vented_mean, REFERENCE_PRESSURE, rtol=1e-13)
    assert float(result.atmosphere_pressure_residual) <= 1e-13 * REFERENCE_PRESSURE
    _assert_continuity(result)
    # Closed walls admit no net volume change of the atmosphere.
    assert abs(float(result.atmosphere_volume_rate)) <= 1e-14


@pytest.mark.parametrize("vented", [False, True])
def test_closed_compartments_satisfy_constraint_rows_and_the_work_identity(
    case: _Case, plan: Any, vented: bool
) -> None:
    result = _project(
        plan,
        case.momentum,
        case.inverse,
        STEP,
        case.constraint(active=(True, True), vented=vented),
    )

    assert bool(result.successful)
    expected_gauge = (
        phx.solver.MACCompartmentGauge.ATMOSPHERE_REFERENCE
        if vented
        else phx.solver.MACCompartmentGauge.COMPARTMENT_DETERMINED
    )
    assert int(result.gauge) == expected_gauge
    for slot in range(2):
        mean = case.gas_mean(case.labels == slot, result.absolute_pressure)
        np.testing.assert_allclose(
            mean, float(result.compartment_pressure[slot]), rtol=1e-13
        )
    assert float(result.compartment_pressure_residual) <= 1e-13 * REFERENCE_PRESSURE
    compliance = case.compartment_volume / REFERENCE_PRESSURE
    np.testing.assert_allclose(
        result.pressure_increment, -(STEP / compliance) * result.volume_rate, rtol=1e-13
    )
    np.testing.assert_allclose(
        result.compartment_pressure,
        jnp.asarray((REFERENCE_PRESSURE + 200.0, REFERENCE_PRESSURE - 100.0))
        + result.pressure_increment,
        rtol=1e-15,
    )
    assert bool(jnp.all(jnp.abs(result.volume_rate) > 0.0))
    _assert_continuity(result)
    net_flux = float(
        jnp.sum(case.operators.discretization.cell_volumes * result.divergence_before)
    )
    total_rate = float(jnp.sum(result.volume_rate) + result.atmosphere_volume_rate)
    assert abs(total_rate - net_flux) <= 1e-12 * float(
        jnp.sum(jnp.abs(result.volume_rate))
    )

    work_terms = (
        result.compartment_work,
        result.atmosphere_work,
        result.dynamic_pressure_work,
        result.offset_pressure_work,
    )
    assert all(
        term.dtype == jnp.dtype(jnp.float64)
        for term in (*work_terms, result.work_identity_residual)
    )
    work_scale = sum(abs(float(term)) for term in work_terms)
    assert abs(float(result.work_identity_residual)) <= 1e-9 * work_scale
    energy_terms = (
        result.kinetic_energy_change,
        result.projection_dissipation,
        result.dynamic_pressure_work,
    )
    energy_scale = sum(abs(float(term)) for term in energy_terms)
    balance = (
        result.kinetic_energy_change
        + result.projection_dissipation
        - result.dynamic_pressure_work
    )
    assert abs(float(balance)) <= 1e-12 * energy_scale
    assert float(result.projection_dissipation) > 0.0


def test_net_boundary_inflow_is_absorbed_only_by_a_compartment(
    case: _Case, plan: Any
) -> None:
    inflow = 0.01
    momentum = (
        case.momentum[0],
        case.momentum[1].at[:, 0].set(case.face_density[1][:, 0] * inflow),
    )
    absorbed = _project(
        plan,
        momentum,
        case.inverse,
        STEP,
        case.constraint(active=(True, True), vented=False),
    )
    refused = _project(
        plan,
        momentum,
        case.inverse,
        STEP,
        case.constraint(active=(False, False), vented=False),
    )

    assert bool(absorbed.successful)
    # Liquid entering through the wall compresses the gas.
    np.testing.assert_allclose(float(jnp.sum(absorbed.volume_rate)), -inflow, rtol=1e-12)
    _assert_continuity(absorbed)
    # Without a compartment the flux is incompatible, as in the owner projection.
    assert (
        int(refused.status)
        == phx.solver.MACCompartmentProjectionStatus.PRESSURE_SOLVE_FAILED
    )
    for projected, original in zip(refused.momentum, momentum, strict=True):
        np.testing.assert_array_equal(projected, original)


def test_block_operator_is_self_adjoint_and_reproduces_the_projection(
    case: _Case, plan: Any
) -> None:
    constraint = case.constraint(active=(True, True), vented=True)
    operator = plan.block_operator(case.inverse, STEP, constraint)
    keys = jax.random.split(jax.random.key(3), 6)
    first = (
        jax.random.normal(keys[0], (12, 12)),
        jax.random.normal(keys[1], (2,)),
        jax.random.normal(keys[2], ()),
    )
    second = (
        jax.random.normal(keys[3], (12, 12)),
        jax.random.normal(keys[4], (2,)),
        jax.random.normal(keys[5], ()),
    )
    forward = float(operator.pairing(second, operator.mv(first)))
    np.testing.assert_allclose(
        float(operator.pairing(operator.transpose_mv(second), first)), forward, rtol=1e-12
    )
    np.testing.assert_allclose(
        float(operator.pairing(operator.mv(second), first)), forward, rtol=1e-12
    )

    result = _project(plan, case.momentum, case.inverse, STEP, constraint)
    image = operator.mv(
        (result.pressure, result.volume_rate, result.atmosphere_volume_rate)
    )
    right = operator.right_hand_side(case.momentum)
    assert _volume_norm(case.operators, image[0] - right[0]) <= TOLERANCE * float(
        result.continuity_scale
    ) * (1.0 + 1.0e-6)
    np.testing.assert_allclose(
        image[1], right[1], rtol=0.0, atol=1e-13 * REFERENCE_PRESSURE
    )
    np.testing.assert_allclose(
        image[2], right[2], rtol=0.0, atol=1e-13 * REFERENCE_PRESSURE
    )


def test_inactive_slots_and_an_empty_atmosphere_act_as_identity_rows(
    case: _Case, plan: Any
) -> None:
    operator = plan.block_operator(
        case.inverse, STEP, case.constraint(active=(True, False), vented=False)
    )
    zero = jnp.zeros((12, 12))
    pressure_row, rate_row, atmosphere_row = operator.mv(
        (zero, jnp.asarray((0.0, 2.0)), jnp.asarray(3.0))
    )
    np.testing.assert_array_equal(pressure_row, 0.0)
    np.testing.assert_array_equal(rate_row, (0.0, 2.0))
    assert float(atmosphere_row) == 3.0


def _breathing_period(two_slabs: bool) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Piston oscillation of a liquid slab on isothermal gas; returns period and expectation."""
    nx, ny, step, steps, amplitude = 4, 20, 0.01, 520, 1.0e-3
    operators = _operators(nx, ny)
    volumes = operators.discretization.cell_volumes
    y = jnp.broadcast_to(((jnp.arange(ny) + 0.5) / ny)[None, :], (nx, ny))
    lower = y < 0.3
    upper = y > 0.7
    density = jnp.where(lower | upper, GAS_DENSITY, LIQUID_DENSITY)
    inverse = tuple(
        1.0 / value for value in operators.interpolate_inverse_momentum(density)
    )
    gas = jnp.where(lower | upper, volumes, 0.0)
    gas_height, liquid_height = 0.3, 0.4
    if two_slabs:
        pressure0 = 1000.0
        labels = jnp.where(lower, 0, jnp.where(upper, 1, -1)).astype(jnp.int32)
        vented = jnp.zeros((nx, ny), dtype=bool)
        active = jnp.asarray((True, True))
        omega_squared = pressure0 / (LIQUID_DENSITY * liquid_height) * (2.0 / gas_height)
    else:
        pressure0 = 2000.0
        labels = jnp.where(lower, 0, -1).astype(jnp.int32)
        vented = upper
        active = jnp.asarray((True,))
        omega_squared = pressure0 / (LIQUID_DENSITY * liquid_height * gas_height)
    rest_volume = jnp.stack(
        tuple(jnp.sum(jnp.where(labels == slot, gas, 0.0)) for slot in range(active.size))
    )
    plan = phx.solver.MACCompartmentProjectionPlan(
        operators, compartment_capacity=active.size, atmosphere=not two_slabs
    )

    def advance(carry: Any, _: None) -> tuple[Any, tuple[Any, Any, Any]]:
        momentum, volume, pressure = carry
        compartment_pressure = pressure0 * rest_volume / volume
        constraint = phx.solver.MACCompartmentConstraint(
            labels=labels,
            gas_volume=gas,
            compliance=volume / compartment_pressure,
            pressure=compartment_pressure,
            active=active,
            atmosphere=vented,
            atmosphere_pressure=pressure0,
            pressure_offset=jnp.zeros((nx, ny)),
        )
        result = plan.project(momentum, inverse, step, constraint, pressure=pressure)
        return (
            (result.momentum, volume + step * result.volume_rate, result.pressure),
            (result.volume_rate[0], result.status, result.gauge),
        )

    initial = (
        tuple(
            jnp.zeros(layout.shape) for layout in operators.discretization.face_layouts
        ),
        rest_volume.at[0].multiply(1.0 + amplitude),
        jnp.zeros((nx, ny)),
    )
    _, (rate, status, gauge) = jax.jit(
        lambda state: jax.lax.scan(advance, state, None, length=steps)
    )(initial)
    rate = np.asarray(rate)
    sign = np.sign(rate)
    crossing = np.nonzero(sign[1:] * sign[:-1] < 0.0)[0]
    times = crossing + rate[crossing] / (rate[crossing] - rate[crossing + 1])
    assert times.size >= 5
    period = 2.0 * float(np.mean(np.diff(times))) * step
    return (
        period,
        2.0 * np.pi / np.sqrt(omega_squared),
        np.asarray(status),
        np.asarray(gauge),
    )


@pytest.mark.parametrize("two_slabs", [False, True])
def test_linear_breathing_frequency_matches_the_isothermal_piston(
    two_slabs: bool,
) -> None:
    period, expected, status, gauge = _breathing_period(two_slabs)

    np.testing.assert_array_equal(
        status, phx.solver.MACCompartmentProjectionStatus.CONVERGED
    )
    expected_gauge = (
        phx.solver.MACCompartmentGauge.COMPARTMENT_DETERMINED
        if two_slabs
        else phx.solver.MACCompartmentGauge.ATMOSPHERE_REFERENCE
    )
    np.testing.assert_array_equal(gauge, expected_gauge)
    assert abs(period / expected - 1.0) <= 0.02


def test_sharp_density_breathing_basis_reaches_strict_tolerance_with_bounded_work() -> None:
    operators = _operators(4, 32)
    volumes = operators.discretization.cell_volumes
    interface_fraction = 0.996
    profile = jnp.concatenate(
        (
            jnp.zeros((8,), dtype=volumes.dtype),
            jnp.asarray((interface_fraction,), dtype=volumes.dtype),
            jnp.ones((15,), dtype=volumes.dtype),
            jnp.asarray((1.0 - interface_fraction,), dtype=volumes.dtype),
            jnp.zeros((7,), dtype=volumes.dtype),
        )
    )
    alpha = jnp.broadcast_to(profile[None, :], (4, 32))
    gas = volumes * (1.0 - alpha)
    lower = (gas > 0.0) & (
        operators.discretization.cell_centers[..., 1] < 0.5
    )
    upper = (gas > 0.0) & ~lower
    labels = jnp.where(lower, 0, jnp.where(upper, 1, -1)).astype(jnp.int32)
    compartment_volume = jnp.stack(
        (
            jnp.sum(jnp.where(lower, gas, 0.0)),
            jnp.sum(jnp.where(upper, gas, 0.0)),
        )
    )
    pressure = jnp.asarray((1019.3, 980.7), dtype=volumes.dtype)
    density = alpha * LIQUID_DENSITY + (1.0 - alpha) * GAS_DENSITY
    face_density = operators.interpolate_inverse_momentum(density)
    inverse = tuple(1.0 / value for value in face_density)
    momentum = tuple(
        jnp.zeros(layout.shape, dtype=volumes.dtype)
        for layout in operators.discretization.face_layouts
    )
    constraint = phx.solver.MACCompartmentConstraint(
        labels=labels,
        gas_volume=gas,
        compliance=compartment_volume / (1.4 * pressure),
        pressure=pressure,
        active=jnp.asarray((True, True), dtype=jnp.bool_),
        atmosphere=jnp.zeros((4, 32), dtype=jnp.bool_),
        atmosphere_pressure=jnp.asarray(1000.0, dtype=volumes.dtype),
        pressure_offset=jnp.zeros((4, 32), dtype=volumes.dtype),
    )
    plan = phx.solver.MACCompartmentProjectionPlan(
        operators,
        compartment_capacity=2,
        atmosphere=False,
        tolerance=1.0e-11,
        maximum_iterations=256,
    )

    result = _project(plan, momentum, inverse, 6.64e-3, constraint)

    assert bool(result.successful)
    assert int(jnp.max(result.pressure_solves.diagnostics.iterations)) < 256
    assert float(result.continuity_residual) <= 1.0e-11 * float(
        result.continuity_scale
    ) * (1.0 + 1.0e-6)


@pytest.mark.parametrize(
    "gas_scale, compliance_scale",
    [(0.0, 1.0), (1.0, 0.0), (1.0, -1.0)],
)
def test_inadmissible_active_slot_is_refused_and_the_input_returned(
    case: _Case, plan: Any, gas_scale: float, compliance_scale: float
) -> None:
    gas = jnp.where(case.labels == 1, gas_scale * case.gas, case.gas)
    compliance = (
        case.compartment_volume
        / REFERENCE_PRESSURE
        * jnp.asarray((1.0, compliance_scale))
    )
    result = _project(
        plan,
        case.momentum,
        case.inverse,
        STEP,
        case.constraint(active=(True, True), vented=True, gas=gas, compliance=compliance),
    )

    assert (
        int(result.status) == phx.solver.MACCompartmentProjectionStatus.INVALID_CONSTRAINT
    )
    assert not bool(result.successful)
    for projected, original in zip(result.momentum, case.momentum, strict=True):
        np.testing.assert_array_equal(projected, original)
    np.testing.assert_array_equal(result.pressure, 0.0)
    np.testing.assert_array_equal(result.volume_rate, 0.0)
    np.testing.assert_array_equal(
        result.compartment_pressure,
        (REFERENCE_PRESSURE + 200.0, REFERENCE_PRESSURE - 100.0),
    )


@pytest.mark.parametrize("capacity", [0, 33])
def test_capacity_outside_the_tiny_schur_route_is_refused(
    case: _Case, capacity: int
) -> None:
    with pytest.raises(ValueError):
        phx.solver.MACCompartmentProjectionPlan(
            case.operators, compartment_capacity=capacity, atmosphere=False
        )


def test_projected_velocity_jvp_matches_central_differences() -> None:
    # The composed owner's implicit tangent solve (unpreconditioned GMRES)
    # converges only at mild density contrast, so the check uses a 2:1 ratio.
    mild = _Case(0.5 * LIQUID_DENSITY)
    plan = phx.solver.MACCompartmentProjectionPlan(
        mild.operators, compartment_capacity=2, atmosphere=True, tolerance=1e-10
    )
    constraint = mild.constraint(active=(True, True), vented=True)
    keys = jax.random.split(jax.random.key(11), 2)
    direction = (
        mild.face_density[0] * jax.random.normal(keys[0], (12, 12)),
        mild.face_density[1]
        * jax.random.normal(keys[1], (12, 13)).at[:, 0].set(0.0).at[:, -1].set(0.0),
    )

    def velocity(momentum: Any) -> Any:
        return plan.project(momentum, mild.inverse, STEP, constraint).velocity

    _, tangent = eqx.filter_jit(
        lambda momentum, change: jax.jvp(velocity, (momentum,), (change,))
    )(mild.momentum, direction)
    # The projection is affine in u* at fixed constraint, so a unit step is exact.
    evaluate = eqx.filter_jit(velocity)
    plus = evaluate(tuple(a + b for a, b in zip(mild.momentum, direction, strict=True)))
    minus = evaluate(tuple(a - b for a, b in zip(mild.momentum, direction, strict=True)))
    difference = tuple(0.5 * (a - b) for a, b in zip(plus, minus, strict=True))
    error = np.sqrt(
        sum(
            float(jnp.sum((a - b) ** 2)) for a, b in zip(tangent, difference, strict=True)
        )
    )
    scale = np.sqrt(sum(float(jnp.sum(b**2)) for b in difference))
    assert error <= 1e-6 * scale
