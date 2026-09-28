#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from math import comb, factorial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


D = phx.discretization
PIC = phx.discretization.pic
mx = phx.solver.maxwell


def _bridge(count: int) -> Any:
    grid = D.TensorGridPlan(
        tuple(D.UniformCellAxisSpec(count, periodic=True) for _ in range(3)),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray([[0.0] * 3, [1.0] * 3]))
    return D.StructuredCochainBridge(grid)


def _transfer(
    bridge: Any, order: PIC.PICShapeOrder, capacity: int, sign: float, offset: int
) -> Any:
    particles = D.ParticleSetPlan(
        jnp.arange(offset, offset + capacity), jnp.ones((capacity,)), ambient_dimension=3
    ).prepare()
    charged = D.ChargedParticlePlan(
        sign * jnp.ones((capacity,)), f"species-{offset}"
    ).prepare(particles)
    return PIC.PICParticleCochainTransferPlan(bridge, shape_order=order).prepare(charged)


def _solver(bridge: Any, order: PIC.PICShapeOrder, capacity: int) -> Any:
    transfers = (
        _transfer(bridge, order, capacity, -1.0, 0),
        _transfer(bridge, order, capacity, 1.0, 100),
    )
    maxwell = phx.solver.CompatibleMaxwellPlan(
        bridge, sources=(phx.solver.PICMaxwellCurrentSourcePlan(),)
    ).prepare()
    electrostatic = phx.solver.CochainElectrostaticPlan(
        bridge, phx.solver.CochainElectrostaticBoundaryPlan.periodic(bridge)
    )
    return phx.solver.CochainMaxwellPICFieldSolver(
        maxwell,
        electrostatic,
        transfers,
        tuple(PIC.ChargeConservingCurrentPlan(value) for value in transfers),
    )


def _pic(solver: Any, **options: Any) -> Any:
    species = tuple(
        PIC.PICSpeciesPlan(
            D.ParticlePopulationPlan(transfer.species.particles),
            PIC.PICChargeModelPlan(
                sign,
                transfer.species.plan.species_id,
                minimum_charge_number=1,
                maximum_charge_number=1,
                initial_charge_number=1,
            ),
        )
        for transfer, sign in zip(solver.transfers, (-1.0, 1.0), strict=True)
    )
    return phx.solver.ElectromagneticPICPlan(solver, species=species, **options)


def _cardinal_bspline(degree: int, x: np.ndarray) -> np.ndarray:
    """Centered cardinal B-spline by the truncated-power formula."""
    total = np.zeros_like(x)
    for k in range(degree + 2):
        shifted = np.maximum(x + 0.5 * (degree + 1) - k, 0.0)
        total += (-1) ** k * comb(degree + 1, k) * shifted**degree
    return total / factorial(degree)


def _cardinal_bspline_derivative(degree: int, x: np.ndarray) -> np.ndarray:
    return _cardinal_bspline(degree - 1, x + 0.5) - _cardinal_bspline(degree - 1, x - 0.5)


def _paths(seed: int, capacity: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    start = generator.uniform(0.0, 1.0, (capacity, 3))
    # Paths cross cell faces, knots, and the periodic seam.
    start[0] = [0.99, 0.02, 0.5]
    end = start + generator.uniform(-0.15, 0.15, (capacity, 3))
    end[0] = [1.03, -0.03, 0.52]
    return start, end, generator.uniform(-2.0, 1.0, capacity)


@pytest.mark.parametrize("order", [2, 3], ids=["quadratic", "cubic"])
def test_spline_whitney_current_conserves_charge_to_roundoff(
    order: PIC.PICShapeOrder,
) -> None:
    bridge = _bridge(6)
    current = PIC.ChargeConservingCurrentPlan(_transfer(bridge, order, 24, -1.0, 0))
    start, end, charge = _paths(3, 24)
    step = 0.01
    result = jax.jit(
        lambda left, right, value: current.deposit(left, right, step, macrocharge=value)
    )(start, end, charge)
    assert result.successful
    rate = jnp.max(jnp.abs(result.end_charge.cochain - result.start_charge.cochain))
    assert float(result.maximum_continuity_defect) <= 1e-14 * float(rate) / step
    # Independent orientation/magnitude check: the domain integral of the
    # physical current is Σ q Δx / Δt, including the unwrapped seam crossing.
    integrals = np.asarray(
        [np.sum(value) / 6.0**2 for value in bridge.unpack(1, result.current)]
    )
    np.testing.assert_allclose(
        integrals, np.sum(charge[:, None] * (end - start), axis=0) / step, atol=1e-11
    )


@pytest.mark.parametrize("order", [2, 3], ids=["quadratic", "cubic"])
def test_spline_whitney_current_is_invariant_to_particle_slot_order(
    order: PIC.PICShapeOrder,
) -> None:
    current = PIC.ChargeConservingCurrentPlan(_transfer(_bridge(6), order, 24, -1.0, 0))
    start, end, charge = _paths(5, 24)
    permutation = np.random.default_rng(9).permutation(24)
    deposit = jax.jit(
        lambda left, right, value: current.deposit(left, right, 0.01, macrocharge=value)
    )
    ordered = deposit(start, end, charge)
    permuted = deposit(start[permutation], end[permutation], charge[permutation])
    np.testing.assert_array_equal(ordered.current, permuted.current)


@pytest.mark.parametrize("order", [2, 3], ids=["quadratic", "cubic"])
def test_spline_whitney_gather_is_gradient_of_charge_shape_interpolant(
    order: PIC.PICShapeOrder,
) -> None:
    count = 6
    bridge = _bridge(count)
    transfer = _transfer(bridge, order, 5, -1.0, 0)
    potential = np.random.default_rng(11).standard_normal(count**3)
    electric = -bridge.exterior_derivative(0, jnp.asarray(potential))
    position = np.random.default_rng(12).uniform(0.0, 1.0, (5, 3))
    gathered = transfer.gather_electric(transfer.build(position), electric)
    assert gathered.successful
    # Reference: −∇ Σ_i φ_i Π_a N^p(x_a/h − i_a) over periodic vertex images.
    grid = potential.reshape((count,) * 3)
    expected = np.zeros((5, 3))
    nodes = np.arange(-3, count + 3)
    scaled = position * count
    for particle in range(5):
        value = [_cardinal_bspline(order, scaled[particle, a] - nodes) for a in range(3)]
        slope = [
            _cardinal_bspline_derivative(order, scaled[particle, a] - nodes)
            for a in range(3)
        ]
        wrapped = grid[np.ix_(nodes % count, nodes % count, nodes % count)]
        for axis in range(3):
            factors = [slope[b] if b == axis else value[b] for b in range(3)]
            weights = factors[0][:, None, None] * factors[1][None, :, None] * factors[2]
            expected[particle, axis] = -count * np.sum(wrapped * weights)
    np.testing.assert_allclose(gathered.values, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    ("passes", "compensation"),
    [(1, False), (2, False), (1, True), (3, True)],
    ids=["binomial", "binomial-2", "compensated", "compensated-3"],
)
def test_filter_plane_wave_response_matches_binomial_symbol(
    passes: int, compensation: bool
) -> None:
    count = 8
    solver = _solver(_bridge(count), 1, 2)
    plan = phx.solver.PICFilterPlan(passes=passes, compensation=compensation)
    mode = 3
    theta = 2.0 * np.pi * mode / count
    wave = np.cos(theta * np.arange(count))[:, None, None] * np.ones((1, count, count))
    filtered = plan.filter_charge(solver, jnp.asarray(wave.reshape(-1)))
    response = np.cos(0.5 * theta) ** (2 * passes)
    if compensation:
        response *= 1.0 + passes * np.sin(0.5 * theta) ** 2
    np.testing.assert_allclose(filtered, response * wave.reshape(-1), atol=1e-14)


def test_periodic_filter_commutes_with_divergence_and_initializes_gauss() -> None:
    solver = _solver(_bridge(6), 2, 2)
    report = phx.solver.PICFilterPlan(passes=2).continuity_report(solver)
    assert report.periodic == (True, True, True)
    assert report.interior_commutation_defect < 1e-13
    assert report.boundary_commutation_defect < 1e-13
    assert report.gauss_initialization_defect < 1e-12


def _reduced_solver(*, periodic: bool) -> Any:
    grid = D.TensorGridPlan(
        (D.UniformCellAxisSpec(16, periodic=periodic),), axis_names=("x",)
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    pec = mx.MaxwellBoundaryPlan("pec")
    field = phx.solver.CompatibleMaxwell1DPlan(
        grid, boundaries=None if periodic else ((pec, pec),)
    )
    return phx.solver.ReducedMaxwellPICFieldSolver(
        field, PIC.ReducedPICTransferPlan(grid)
    )


def test_filter_mirror_reports_wall_normal_continuity_defect() -> None:
    plan = phx.solver.PICFilterPlan()
    periodic = plan.continuity_report(_reduced_solver(periodic=True))
    walled = plan.continuity_report(_reduced_solver(periodic=False))
    assert periodic.boundary_commutation_defect < 1e-13
    # The mirror commutes while no current crosses a wall ...
    assert walled.interior_commutation_defect < 1e-13
    # ... and reports the defect wall-normal current leaves behind.
    assert walled.boundary_commutation_defect > 1e-3
    assert walled.gauss_initialization_defect < 1e-12


@pytest.mark.parametrize("order", [1, 2, 3], ids=["linear", "quadratic", "cubic"])
def test_filtered_pic_conserves_filtered_charge(order: PIC.PICShapeOrder) -> None:
    solver = _solver(_bridge(4), order, 2)
    pic = _pic(solver, filters=(phx.solver.PICFilterPlan(passes=1),))
    dt = jnp.asarray(0.2 * solver.stable_step)
    position = jnp.asarray([[0.3, 0.4, 0.55], [0.8, 0.15, 0.6]])
    velocity = jnp.asarray([[0.4, -0.2, 0.3], [-0.3, 0.35, 0.1]])
    state = pic.initialize((position, position), (velocity, jnp.zeros((2, 3))), dt)
    step = jax.jit(lambda value, size: pic.step_detailed(value, size))
    for _ in range(4):
        result = step(state, dt)
        assert result.successful
        assert result.diagnostics.particle_field_charge_defect < 1e-12
        assert result.diagnostics.electric_constraint < 1e-10
        state = result.accepted_state


def test_filtered_self_force_vanishes_by_symmetry_and_is_antisymmetric() -> None:
    count = 6
    solver = _solver(_bridge(count), 2, 1)
    plan = phx.solver.PICFilterPlan(passes=2)
    active = jnp.asarray([True])

    def self_force(point: np.ndarray) -> np.ndarray:
        position = jnp.asarray(point[None, :])
        charge, _ = solver.deposit_charge(0, position, jnp.asarray([-1.0]), active)
        # A uniform neutralizing background removes the mean charge.
        field, ok = solver.initialize_field(
            plan.filter_charge(solver, charge - jnp.mean(charge))
        )
        assert ok
        sample = solver.gather(0, position, active, plan.filter_field(solver, field))
        return -np.asarray(sample.electric[0])

    h = 1.0 / count
    vertex = np.asarray([2.0, 3.0, 1.0]) * h
    center = vertex + 0.5 * h
    offset = np.asarray([0.21, -0.13, 0.08]) * h
    scale = np.max(np.abs(self_force(center + offset)))
    assert scale > 0.0
    np.testing.assert_allclose(self_force(vertex), 0.0, atol=1e-12 * scale)
    np.testing.assert_allclose(self_force(center), 0.0, atol=1e-12 * scale)
    np.testing.assert_allclose(
        self_force(center + offset), -self_force(center - offset), atol=1e-12 * scale
    )


def _guarded_pic(speed: float, frequency: float) -> tuple[Any, Any]:
    solver = _solver(_bridge(8), 1, 2)
    dt = 0.5 * float(solver.stable_step)
    audit = mx.CompatibleMaxwellDispersionAudit(
        solver.maxwell, mx.MaxwellMaterialRegion((0, 0, 0), (8, 8, 8)), dt
    )
    regime = mx.CherenkovRegimePlan(
        audit, jnp.asarray([speed, 0.0, 0.0]), jnp.asarray([frequency]), 4
    )
    guard = phx.solver.PICCherenkovGuard(regime, species=0)
    return _pic(solver, cherenkov_guards=(guard,)), dt


def test_cherenkov_guard_refuses_numerical_only_emission() -> None:
    # Vacuum Yee branches near the grid cutoff are slower than 0.95c.
    with pytest.raises(ValueError, match="Numerical Cherenkov emission"):
        _guarded_pic(0.95, 16.0)


def test_cherenkov_guard_admits_audited_step_and_speed_only() -> None:
    pic, dt = _guarded_pic(0.5, 1.0)
    evidence = pic.cherenkov_evidence[0]
    assert not bool(jnp.any(evidence.numerical_only))
    position = jnp.asarray([[0.3, 0.4, 0.55], [0.8, 0.15, 0.6]])
    drift = jnp.zeros((2, 3)).at[:, 0].set(0.45)
    state = pic.initialize((position, position), (drift, jnp.zeros((2, 3))), dt)
    assert pic.step_detailed(state, dt).successful
    flag = int(PIC.PICRejectionReason.NUMERICAL_CHERENKOV)
    other_step = pic.step_detailed(state, 0.5 * dt)
    assert not other_step.successful
    assert int(other_step.diagnostics.rejection_reason) & flag
    fast = pic.initialize(
        (position, position), (drift.at[:, 0].set(0.6), jnp.zeros((2, 3))), dt
    )
    too_fast = pic.step_detailed(fast, dt)
    assert int(too_fast.diagnostics.rejection_reason) & flag
